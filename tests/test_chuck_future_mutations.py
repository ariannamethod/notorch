#!/usr/bin/env python3
"""Require future-credit mutations to fail and preserve pre-fit v1 bytes.

Every binary is rebuilt in a temporary directory. The checked-out sources are
never modified. Compiler errors or crashes do not count as detected mutations.
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
BASELINE = "48512d822a67043e0899a7ac6a2b87779250f42b"
SOURCES = ("notorch.c", "notorch.h", "chuck_architect.h", "chuck_architect_impl.h")
TEST = "tests/test_chuck_architect_future.c"


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
    compiler = shlex.split(os.environ.get("CC", "cc"))
    originals = {name: (ROOT / name).read_bytes() for name in (*SOURCES, TEST)}
    report = {"source_sha256": {n: digest(b) for n, b in originals.items()},
              "mutations": [], "v1_compatibility": {"baseline_commit": BASELINE},
              "passed": False}
    with tempfile.TemporaryDirectory(prefix="notorch-future-credit-") as temporary:
        temp = Path(temporary)
        test = temp / "future_test.c"
        test.write_bytes(originals[TEST])

        def build(name, source_set, compatibility_only=False):
            directory = temp / name
            directory.mkdir()
            for source in SOURCES:
                (directory / source).write_bytes(source_set[source])
            command = compiler + ["-O2", "-std=gnu11", "-pthread", "-I", str(directory),
                                  "-I", str(ROOT)]
            if compatibility_only:
                command.append("-DNT_CHUCK_FUTURE_V1_COMPAT_ONLY")
            binary = directory / "gate"
            command += [str(directory / "notorch.c"), str(test), "-lm", "-o", str(binary)]
            return binary, invoke(command)

        binary, compiled = build("baseline", originals)
        report["baseline"] = {"build": compiled}
        if compiled["returncode"] == 0:
            report["baseline"]["test"] = invoke([binary])
        baseline_ok = report["baseline"].get("test", {}).get("returncode") == 0
        if not baseline_ok:
            print("FAIL unmodified future-credit gate")
            print(json.dumps(report["baseline"], indent=2))
        else:
            print("PASS unmodified future-credit gate")
            outcome = "r.future_loss[k] = sample->future_loss[k]; // NT_CA_COMPARISON_OUTCOME"
            mutations = [
                {"name": "immediate_loss_wired_as_future_credit", "anchor": outcome,
                 "replacement": "r.future_loss[k] = ((const float[3]){2.35505104f, 2.356493f, 2.35361862f})[k]; // NT_CA_COMPARISON_OUTCOME"},
                {"name": "future_learning_disabled",
                 "anchor": "float rate = prior.config.learning_rate; // NT_CA_COMPARISON_RATE",
                 "replacement": "float rate = 0; // NT_CA_COMPARISON_RATE"},
                {"name": "brake_push_outcome_heads_swapped", "anchor": outcome,
                 "replacement": "r.future_loss[k] = sample->future_loss[k == 1 ? 2 : k == 2 ? 1 : 0]; // NT_CA_COMPARISON_OUTCOME"},
            ]
            for mutation in mutations:
                source = originals["chuck_architect_impl.h"].decode()
                if source.count(mutation["anchor"]) != 1:
                    raise RuntimeError(f"{mutation['name']}: mutation anchor not unique")
                mutant = source.replace(mutation["anchor"], mutation["replacement"], 1).encode()
                changed = dict(originals)
                changed["chuck_architect_impl.h"] = mutant
                mutant_binary, built = build(mutation["name"], changed)
                record = {"name": mutation["name"], "gate": "measured_reversal",
                          "mutant_sha256": digest(mutant), "build": built, "caught": False}
                if built["returncode"] == 0:
                    result = invoke([mutant_binary, "measured_reversal"])
                    record["test"] = result
                    record["caught"] = result["returncode"] == 1 and "FAIL measured_reversal:" in result["stderr"]
                report["mutations"].append(record)
                print(("PASS caught " if record["caught"] else "FAIL escaped or invalid ") + mutation["name"])
                print(record.get("test", {}).get("stderr", built["stderr"]).strip())

            # Rebuild the unchanged online emitter against the actual pre-fit
            # source, then compare all bytes of three complete v1 saved lives.
            old_sources = {}
            for name in SOURCES:
                result = subprocess.run(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT,
                                        capture_output=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(f"pinned baseline source unavailable: {name}")
                old_sources[name] = result.stdout
            old_binary, built = build("v1-old", old_sources, compatibility_only=True)
            compatibility = report["v1_compatibility"]
            compatibility["source_sha256"] = {n: digest(b) for n, b in old_sources.items()}
            compatibility["build"] = built
            compatibility["files"] = []
            if built["returncode"] == 0:
                old_output, new_output = temp / "old-output", temp / "new-output"
                old_output.mkdir(); new_output.mkdir()
                compatibility["old_run"] = invoke([old_binary, "--write-v1", old_output])
                compatibility["new_run"] = invoke([binary, "--write-v1", new_output])
                if compatibility["old_run"]["returncode"] == compatibility["new_run"]["returncode"] == 0:
                    for name in ("init.bin", "pending.bin", "complete.bin"):
                        old_bytes, new_bytes = (old_output / name).read_bytes(), (new_output / name).read_bytes()
                        compatibility["files"].append({"name": name, "bytes": len(new_bytes),
                            "old_sha256": digest(old_bytes), "new_sha256": digest(new_bytes),
                            "identical": old_bytes == new_bytes and bool(new_bytes)})
            compatibility["passed"] = len(compatibility["files"]) == 3 and all(x["identical"] for x in compatibility["files"])
            print(("PASS" if compatibility["passed"] else "FAIL") + " unchanged v1 online life bytes")
            report["passed"] = (all(x["caught"] for x in report["mutations"]) and
                                len(report["mutations"]) == 3 and compatibility["passed"])
    report["working_sources_unchanged"] = all((ROOT / n).read_bytes() == data for n, data in originals.items())
    report["passed"] &= report["working_sources_unchanged"]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
