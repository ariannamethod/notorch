#!/usr/bin/env python3
"""Build deliberate defects in temporary sources and require real red gates.

The working sources are read once, never patched in place. Each candidate must
compile, run its named gate, fail normally, and identify that gate. A compiler
error or crash is an invalid mutation experiment. Optional JSON retains exact
source hashes, build commands, gate output and return codes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invoke(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return {"command": [str(item) for item in command], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write compact mutation receipt")
    args = parser.parse_args()
    names = ("notorch.c", "notorch.h", "chuck_architect.h", "chuck_architect_impl.h",
             "tests/test_chuck_architect.c")
    originals = {name: (ROOT / name).read_bytes() for name in names}
    hashes = {name: digest(data) for name, data in originals.items()}
    compiler = shlex.split(os.environ.get("CC", "cc"))
    mutations = [
        {
            "name": "brake_executes_push", "file": "notorch.c", "gate": "actions",
            "before": "case NT_CHUCK_ACTION_BRAKE:\n        cs->dampen *= NT_CHUCK_DAMP_DOWN;",
            "after": "case NT_CHUCK_ACTION_BRAKE:\n        cs->dampen *= NT_CHUCK_DAMP_UP;",
        },
        {
            "name": "selected_action_replaced", "file": "chuck_architect_impl.h", "gate": "credit",
            "before": "d->action.kind = (nt_chuck_action_kind)(best + NT_CHUCK_ACTION_HOLD);",
            "after": "d->action.kind = NT_CHUCK_ACTION_PUSH;",
        },
        {
            "name": "consequence_sign_reversed", "file": "chuck_architect_impl.h", "gate": "credit",
            "before": "r.error = r.predicted - r.reward; // NT_CA_CREDIT_SIGN:",
            "after": "r.error = r.predicted + r.reward; // NT_CA_CREDIT_SIGN:",
        },
        {
            "name": "consequence_learning_suppressed", "file": "chuck_architect_impl.h", "gate": "credit",
            "before": "float rate = a.config.learning_rate;",
            "after": "float rate = 0;",
        },
    ]
    report = {"source_sha256": hashes, "baseline": None, "mutations": [], "passed": False}
    with tempfile.TemporaryDirectory(prefix="notorch-architect-mutations-") as temporary:
        temp = Path(temporary)
        for name in ("notorch.h", "chuck_architect.h"):
            (temp / name).write_bytes(originals[name])

        def build_and_run(name, changed, gate=None):
            for source in ("notorch.c", "chuck_architect_impl.h"):
                (temp / source).write_bytes(changed.get(source, originals[source]))
            binary = temp / name
            build = invoke(compiler + ["-O2", "-std=gnu11", "-pthread", "-I", str(temp),
                                       "-I", str(ROOT), str(ROOT / "tests/test_chuck_architect.c"),
                                       str(temp / "notorch.c"), "-lm", "-o", str(binary)])
            record = {"build": build}
            if build["returncode"] == 0:
                record["test"] = invoke([str(binary)] + ([gate] if gate else []))
            return record

        report["baseline"] = build_and_run("baseline", {})
        baseline = report["baseline"]
        if baseline["build"]["returncode"] != 0 or baseline.get("test", {}).get("returncode") != 0:
            print("FAIL unmodified baseline must compile and pass all Architect gates")
            print(json.dumps(baseline, indent=2))
        else:
            print("PASS unmodified baseline")
            for mutation in mutations:
                source = originals[mutation["file"]].decode()
                count = source.count(mutation["before"])
                if count != 1:
                    raise RuntimeError(f"{mutation['name']}: expected one mutation anchor, found {count}")
                changed = source.replace(mutation["before"], mutation["after"], 1).encode()
                record = {"name": mutation["name"], "gate": mutation["gate"],
                          "file": mutation["file"], "mutant_sha256": digest(changed),
                          **build_and_run(mutation["name"], {mutation["file"]: changed}, mutation["gate"])}
                test = record.get("test", {})
                record["caught"] = (record["build"]["returncode"] == 0
                                    and test.get("returncode") == 1
                                    and f"FAIL {mutation['gate']}:" in test.get("stderr", ""))
                report["mutations"].append(record)
                print(("PASS caught " if record["caught"] else "FAIL escaped or invalid ") + mutation["name"])
                print(test.get("stderr", record["build"]["stderr"]).strip())
            report["passed"] = all(item["caught"] for item in report["mutations"]) and bool(report["mutations"])
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == data for name, data in originals.items())
    report["passed"] = report["passed"] and report["working_sources_unchanged"]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
