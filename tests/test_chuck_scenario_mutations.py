#!/usr/bin/env python3
"""Require temporal/action scenario gates to catch isolated source defects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def execute(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return {"command": [str(value) for value in command], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    names = ["notorch.c", "notorch.h", "chuck_architect.h", "chuck_architect_impl.h",
             "tests/test_chuck_architect_scenarios.c"]
    sources = {name: (ROOT / name).read_bytes() for name in names}
    mutations = [
        {
            "name": "resume_rng_reseeded", "gate": "interruptions",
            "before": "    if (payload.failed || payload.at != NT_CA_PAYLOAD_BYTES || !nt_ca_state_valid(&a))",
            "after": "    a.rng = a.config.seed; /* deliberate continuation defect */\n"
                     "    if (payload.failed || payload.at != NT_CA_PAYLOAD_BYTES || !nt_ca_state_valid(&a))",
        },
        {
            "name": "previous_reward_unread", "gate": "temporal_readout",
            "before": "f[15] = a->has_history ? a->prev_reward : 0;",
            "after": "f[15] = 0; /* deliberate history-readout defect */",
        },
        {
            "name": "chosen_action_held", "gate": "masked_exploration",
            "before": "d->action.kind = (nt_chuck_action_kind)(best + NT_CHUCK_ACTION_HOLD);",
            "after": "d->action.kind = NT_CHUCK_ACTION_HOLD; /* deliberate action-readout defect */",
        },
    ]
    sha256 = lambda data: hashlib.sha256(data).hexdigest()
    report = {"sources": {name: sha256(data) for name, data in sources.items()}, "mutations": []}
    compiler = shlex.split(os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="chuck-scenario-mutants-") as temporary:
        temp = Path(temporary)
        for name, data in sources.items():
            (temp / Path(name).name).write_bytes(data)

        def candidate(name, source, gate=None):
            (temp / "chuck_architect_impl.h").write_bytes(source)
            binary = temp / name
            build = execute(compiler + ["-O2", "-std=gnu11", "-pthread", "-I", str(temp),
                "-I", str(ROOT), str(temp / "test_chuck_architect_scenarios.c"),
                str(temp / "notorch.c"), "-lm", "-o", str(binary)])
            result = {"build": build}
            if build["returncode"] == 0:
                result["test"] = execute([str(binary)] + ([gate] if gate else []))
            return result

        original = sources["chuck_architect_impl.h"]
        report["baseline"] = candidate("baseline", original)
        baseline = report["baseline"]
        valid = baseline["build"]["returncode"] == 0 and baseline.get("test", {}).get("returncode") == 0
        print(("PASS" if valid else "FAIL") + " unmodified temporal scenarios", flush=True)
        if not valid:
            print(json.dumps(baseline, indent=2))
        else:
            text = original.decode()
            for mutation in mutations:
                if text.count(mutation["before"]) != 1:
                    raise RuntimeError(f"{mutation['name']}: source anchor is not unique")
                changed = text.replace(mutation["before"], mutation["after"], 1).encode()
                result = {"name": mutation["name"], "gate": mutation["gate"],
                          "mutant_sha256": sha256(changed),
                          **candidate(mutation["name"], changed, mutation["gate"])}
                test = result.get("test", {})
                result["caught"] = (result["build"]["returncode"] == 0 and test.get("returncode") == 1
                                    and f"FAIL {mutation['gate']}:" in test.get("stderr", ""))
                report["mutations"].append(result)
                print(("PASS caught " if result["caught"] else "FAIL escaped/invalid ") + mutation["name"], flush=True)
                print(test.get("stderr", result["build"]["stderr"]).strip(), flush=True)
        report["passed"] = valid and len(report["mutations"]) == len(mutations) and all(
            item["caught"] for item in report["mutations"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
