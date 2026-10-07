#!/usr/bin/env python3
"""Require conditioned-credit and frozen-feedback defects to trip named gates.

Every binary is rebuilt from a private source snapshot. Compiler failures and
crashes are not counted as detected defects. Checked-out sources stay intact.
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
SOURCES = ("notorch.c", "notorch.h", "chuck_architect.h", "chuck_architect_impl.h")
TEST = "tests/test_chuck_architect_conditioned.c"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invoke(command):
    result = subprocess.run([str(x) for x in command], cwd=ROOT, capture_output=True,
                            text=True, timeout=180)
    return {"command": [str(x) for x in command], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    originals = {name: (ROOT / name).read_bytes() for name in (*SOURCES, TEST)}
    report = {"source_sha256": {name: digest(data) for name, data in originals.items()},
              "mutations": [], "passed": False}
    compiler = shlex.split(os.environ.get("CC", "cc"))
    mutations = [
        {"name": "relative_hold_denominator_replaces_conditioned_scale", "gate": "conditioned_scale",
         "anchor": "nt_ca_ratio(r->loss_delta[k], out.scale); // NT_CA_CONDITIONED_CREDIT",
         "replacement": "nt_ca_ratio(r->loss_delta[k], fabs((double)r->future_loss[0]) + 1e-6); // NT_CA_CONDITIONED_CREDIT"},
        {"name": "conditioned_brake_push_outcomes_swapped", "gate": "conditioned_choice",
         "anchor": "r->future_loss[k] = sample->future_loss[k]; // NT_CA_CONDITIONED_OUTCOME",
         "replacement": "r->future_loss[k] = sample->future_loss[k == 1 ? 2 : k == 2 ? 1 : 0]; // NT_CA_CONDITIONED_OUTCOME"},
        {"name": "frozen_feedback_accidentally_learns", "gate": "frozen_feedback",
         "anchor": "// NT_CA_FROZEN_COMPLETION: complete measured history without weight learning.",
         "replacement": "return nt_chuck_architect_feedback(architect, after_loss, receipt); // NT_CA_FROZEN_COMPLETION"},
    ]
    with tempfile.TemporaryDirectory(prefix="notorch-conditioned-credit-") as temporary:
        temp = Path(temporary)
        test = temp / "conditioned_test.c"
        test.write_bytes(originals[TEST])

        def build(name, impl):
            directory = temp / name
            directory.mkdir()
            for source in SOURCES:
                (directory / source).write_bytes(impl if source == "chuck_architect_impl.h" else originals[source])
            binary = directory / "gate"
            command = compiler + ["-O2", "-std=gnu11", "-pthread", "-I", str(directory), "-I", str(ROOT),
                                  str(directory / "notorch.c"), str(test), "-lm", "-o", str(binary)]
            return binary, invoke(command)

        binary, built = build("unmodified", originals["chuck_architect_impl.h"])
        report["baseline"] = {"build": built}
        if built["returncode"] == 0:
            report["baseline"]["test"] = invoke([binary])
        baseline_ok = report["baseline"].get("test", {}).get("returncode") == 0
        print(("PASS" if baseline_ok else "FAIL") + " unmodified conditioned-credit gate")
        if not baseline_ok:
            print(json.dumps(report["baseline"], indent=2))
        else:
            for mutation in mutations:
                source = originals["chuck_architect_impl.h"].decode()
                if source.count(mutation["anchor"]) != 1:
                    raise RuntimeError(f"{mutation['name']}: mutation anchor is not unique")
                mutant = source.replace(mutation["anchor"], mutation["replacement"], 1).encode()
                binary, built = build(mutation["name"], mutant)
                record = {"name": mutation["name"], "gate": mutation["gate"], "build": built,
                          "mutant_sha256": digest(mutant), "caught": False}
                if built["returncode"] == 0:
                    result = invoke([binary, mutation["gate"]])
                    record["test"] = result
                    record["caught"] = (result["returncode"] == 1 and
                                        f"FAIL {mutation['gate']}:" in result["stderr"])
                report["mutations"].append(record)
                print(("PASS caught " if record["caught"] else "FAIL escaped or invalid ") + mutation["name"])
                print(record.get("test", {}).get("stderr", built["stderr"]).strip())
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == data for name, data in originals.items())
    report["passed"] = (baseline_ok and len(report["mutations"]) == len(mutations) and
                        all(item["caught"] for item in report["mutations"]) and report["working_sources_unchanged"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
