#!/usr/bin/env python3
"""Compile isolated repeated-credit defects; require normal named-gate failures.

Python standard library and a C99-capable compiler are the only dependencies.
Production sources are read-only. No sentence generation or experiment fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ("spa_agent.c", "spa_agent.h", "tests/test_spa_agent_repeated.c",
         "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def invoke(command, timeout=180):
    result = subprocess.run([str(value) for value in command], cwd=ROOT, text=True,
                            capture_output=True, timeout=timeout, check=False)
    return {"command": [str(value) for value in command], "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def portable(value, temporary):
    if isinstance(value, str):
        return value.replace(str(temporary), "<temporary>").replace(str(ROOT), "<source>")
    if isinstance(value, list):
        return [portable(item, temporary) for item in value]
    if isinstance(value, dict):
        return {key: portable(item, temporary) for key, item in value.items()}
    return value


def repeated_replace(source, anchor, replacement):
    start = source.index("int nt_spa_agent_fit_repeated(")
    end = source.index("// Canonical fields are explicitly encoded.", start)
    body = source[start:end]
    if body.count(anchor) != 1:
        raise ValueError("repeated mutation anchor is not unique: " + anchor)
    return source[:start] + body.replace(anchor, replacement, 1) + source[end:]


def mean_before_clamp(source):
    start = source.index("static float reward_for(")
    end = source.index("static int same_float(", start)
    helper = source[start:end].replace("static float reward_for(", "static float mutant_unclipped_reward(", 1)
    anchor = "return bounded((float)reward,-1,1);"
    if helper.count(anchor) != 1:
        raise ValueError("native reward clamp anchor is not unique")
    helper = helper.replace(anchor, "return (float)reward;", 1)
    changed = repeated_replace(source, "sums[i]+=reward_for(&a->config,&alternative->consequence);",
                               "sums[i]+=mutant_unclipped_reward(&a->config,&alternative->consequence);")
    changed = repeated_replace(changed, "r.rewards[i]=(float)(sums[i]/count);",
                               "r.rewards[i]=bounded((float)(sums[i]/count),-1,1);")
    insertion = changed.index("int nt_spa_agent_fit_repeated(")
    return changed[:insertion] + helper + changed[insertion:]


def run(compiler, temporary):
    originals = {name: (ROOT / name).read_bytes() for name in FILES}
    source = originals["spa_agent.c"].decode()
    report = {"schema": 1, "operation": "synthetic compiled repeated-credit mutations",
              "source_sha256": {name: digest(raw) for name, raw in originals.items()},
              "script_sha256": digest(Path(__file__).read_bytes()),
              "mutations": []}
    notorch = temporary / "notorch.o"
    report["build_notorch"] = invoke(compiler + ["-std=gnu11", "-O2", "-pthread", "-I", ROOT,
                                                "-c", ROOT / "notorch.c", "-o", notorch])
    if report["build_notorch"]["returncode"]:
        report["passed"] = False
        return report

    def execute(name, changed, gate=None):
        directory = temporary / name
        directory.mkdir()
        (directory / "spa_agent.c").write_text(changed)
        (directory / "spa_agent.h").write_bytes(originals["spa_agent.h"])
        (directory / "test.c").write_bytes(originals["tests/test_spa_agent_repeated.c"])
        binary = directory / "gate"
        entry = {"build": invoke(compiler + ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-O2", "-pthread", "-I", directory, "-I", ROOT, directory / "test.c",
                    directory / "spa_agent.c", notorch, "-lm", "-o", binary])}
        if entry["build"]["returncode"] == 0:
            entry["run"] = invoke([binary] + ([gate] if gate else []))
        return entry

    report["baseline"] = execute("baseline", source)
    if report["baseline"].get("run", {}).get("returncode") != 0:
        print("FAIL unmodified repeated-credit gate", flush=True)
        report["passed"] = False
        return report
    print("PASS unmodified repeated-credit gate", flush=True)
    sign = "r.targets[i]=r.rewards[i]-r.rewards[NT_SPA_KEEP];"
    update = "if(rate>0) train_policy(&next,e->features,hidden,gradient,rate);"
    mutations = (
        ("reversed_mean_credit", "NT_SPA_REPEATED_TARGET_SIGN", sign,
         repeated_replace(source, sign, "r.targets[i]=r.rewards[NT_SPA_KEEP]-r.rewards[i];")),
        ("suppressed_repeated_update", "NT_SPA_REPEATED_UPDATE", update,
         repeated_replace(source, update, "if(rate>0) train_policy(&next,e->features,hidden,gradient,0);")),
        ("mean_before_clamp", "NT_SPA_REPEATED_CLIP_FIRST",
         "unclipped reward sum followed by bounded mean", mean_before_clamp(source)),
    )
    for name, anchor, expression, changed in mutations:
        entry = {"name": name, "gate": "aggregate", "source_anchor": anchor,
                 "expression": expression, "mutant_source_sha256": digest(changed.encode())}
        entry.update(execute(name, changed, "aggregate"))
        result = entry.get("run", {})
        entry["caught"] = result.get("returncode") == 1 and "FAIL aggregate:" in result.get("stderr", "")
        report["mutations"].append(entry)
        print(("PASS caught " if entry["caught"] else "FAIL escaped or invalid ") + name, flush=True)
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == raw for name, raw in originals.items())
    report["passed"] = report["working_sources_unchanged"] and all(x["caught"] for x in report["mutations"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="spa-repeated-mutations-") as directory:
        temporary = Path(directory)
        report = portable(run(compiler, temporary), temporary)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS repeated mutations" if report["passed"] else "FAIL repeated mutations")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
