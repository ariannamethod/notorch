#!/usr/bin/env python3
"""Compile temporary SPA defects; only named normal test failures count as caught.

The working tree is never patched. Baseline must build and pass first. Compile
failures, signals, timeouts, missing anchors and unrelated failures are red for
this harness. Optional JSON records commands, source hashes and actual output.
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


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def invoke(command):
    command = [str(item) for item in command]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "TIMEOUT"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write exact mutation receipts")
    args = parser.parse_args()
    names = ("spa_agent.c", "spa_agent.h", "notorch.c", "notorch.h",
             "chuck_architect.h", "chuck_architect_impl.h", "tests/test_spa_agent.c")
    original = {name: (ROOT / name).read_bytes() for name in names}
    report = {"source_sha256": {name: sha256(data) for name, data in original.items()},
              "mutations": [], "passed": False}
    mutations = [
        {"name": "selection_prefers_lower_score", "gate": "actions",
         "before": "if(available(k,o) && d->scores[k]>d->scores[best]) best=k;",
         "after": "if(available(k,o) && d->scores[k]<d->scores[best]) best=k;"},
        {"name": "consequence_sign_reversed", "gate": "credit",
         "before": "return bounded((float)reward,-1,1);",
         "after": "return bounded(-(float)reward,-1,1);"},
        {"name": "acquired_credit_sign_reversed", "gate": "credit",
         "before": "gradient[p->action.kind]=bounded(r.error,-1,1);",
         "after": "gradient[p->action.kind]=bounded(-r.error,-1,1);"},
        {"name": "consequence_learning_suppressed", "gate": "credit",
         "before": "train_policy(&next.policy,p->features,p->hidden,gradient,a->config.learning_rate);",
         "after": "train_policy(&next.policy,p->features,p->hidden,gradient,0);"},
    ]
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("SPA_TEST_CFLAGS", "-O2 -std=gnu11 -pthread"))
    with tempfile.TemporaryDirectory(prefix="notorch-spa-mutations-") as directory:
        temp = Path(directory)
        for name, data in original.items():
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        report["core_build"] = invoke(compiler + flags + ["-I", temp, "-c", temp / "notorch.c",
                                                           "-o", temp / "notorch.o"])

        def trial(name, source, gate=None):
            (temp / "spa_agent.c").write_text(source)
            binary = temp / name
            build = invoke(compiler + flags + ["-I", temp, temp / "tests/test_spa_agent.c",
                            temp / "spa_agent.c", temp / "notorch.o", "-lm", "-o", binary])
            row = {"build": build}
            if build["returncode"] == 0:
                row["test"] = invoke([binary] + ([gate] if gate else []))
            return row

        if report["core_build"]["returncode"] != 0:
            print("FAIL unmodified notorch build")
            print(report["core_build"]["stderr"])
        else:
            source = original["spa_agent.c"].decode()
            report["baseline"] = trial("baseline", source)
            base = report["baseline"]
            if base["build"]["returncode"] != 0 or base.get("test", {}).get("returncode") != 0:
                print("FAIL unmodified baseline")
                print(json.dumps(base, indent=2))
            else:
                print("PASS unmodified baseline")
                for spec in mutations:
                    matches = source.count(spec["before"])
                    if matches != 1:
                        row = {"name": spec["name"], "gate": spec["gate"], "caught": False,
                               "error": f"expected one mutation anchor, found {matches}"}
                    else:
                        changed = source.replace(spec["before"], spec["after"], 1)
                        row = {"name": spec["name"], "gate": spec["gate"],
                               "mutant_sha256": sha256(changed.encode()),
                               **trial(spec["name"], changed, spec["gate"])}
                        result = row.get("test", {})
                        row["caught"] = (row["build"]["returncode"] == 0
                                         and result.get("returncode") == 1
                                         and f"FAIL {spec['gate']}:" in result.get("stderr", ""))
                    report["mutations"].append(row)
                    print(("PASS caught " if row["caught"] else "FAIL invalid/escaped ") + spec["name"])
                    print(row.get("test", {}).get("stderr", row.get("error", "")).strip())
                report["passed"] = bool(report["mutations"]) and all(row["caught"] for row in report["mutations"])
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == data for name, data in original.items())
    report["passed"] = report["passed"] and report["working_sources_unchanged"]
    if not report["working_sources_unchanged"]:
        print("FAIL sources changed during run; rerun on a stable source snapshot")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
