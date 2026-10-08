#!/usr/bin/env python3
"""Compile three isolated conditioned-credit defects against named native gates.

Production sources remain read-only. Outcomes are synthetic arithmetic fixtures;
this command runs no model generation or empirical experiment training.
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
FILES = ("spa_agent.c", "spa_agent.h", "tests/test_spa_agent_conditioned.c",
         "tests/test_spa_agent_repeated.c", "tests/test_spa_agent_future.c",
         "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h")
TARGET = "r.comparison.targets[i]=(float)((double)r.comparison.targets[i]/r.scale);"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def invoke(command, timeout=180, env=None):
    result = subprocess.run([str(value) for value in command], cwd=ROOT, text=True,
                            capture_output=True, timeout=timeout, check=False, env=env)
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


def conditioned_replace(source, replacement):
    start = source.index("int nt_spa_agent_fit_conditioned(")
    end = source.index("int nt_spa_agent_fit_repeated(", start)
    body = source[start:end]
    if body.count(TARGET) != 1:
        raise ValueError("conditioned target anchor is not unique")
    return source[:start] + body.replace(TARGET, replacement, 1) + source[end:]


PER_REPEAT = """{
            double wrong_mean=0;
            uint32_t repetition;
            for(repetition=0;repetition<count;++repetition) {
                nt_spa_agent temporary=*a;
                nt_spa_comparison_receipt sample;
                double sample_scale=scale_floor;
                unsigned action;
                if(nt_spa_agent_fit_comparison(&temporary,e,comparisons+repetition,0,&sample)!=NT_SPA_OK)
                    return NT_SPA_E_COMPARISON;
                for(action=0;action<NT_SPA_AGENT_ACTIONS;++action)
                    if(sample.action_mask&(1u<<action)) sample_scale=fmax(sample_scale,fabs((double)sample.targets[action]));
                wrong_mean+=(double)sample.targets[i]/sample_scale;
            }
            r.comparison.targets[i]=(float)(wrong_mean/count);
        }"""


def run(compiler, temporary, sanitize=False):
    originals = {name: (ROOT / name).read_bytes() for name in FILES}
    source = originals["spa_agent.c"].decode()
    report = {"schema": 1, "operation": "synthetic native conditioned-credit gates and compiled mutations",
              "source_sha256": {name: digest(raw) for name, raw in originals.items()},
              "script_sha256": digest(Path(__file__).read_bytes()),
              "mutations": [], "legacy_gates": []}
    notorch = temporary / "notorch.o"
    report["build_notorch"] = invoke(compiler + ["-std=gnu11", "-O2", "-pthread", "-I", ROOT,
                                                "-c", ROOT / "notorch.c", "-o", notorch])
    if report["build_notorch"]["returncode"]:
        report["passed"] = False
        return report

    def execute(name, changed, gate=None, test="tests/test_spa_agent_conditioned.c"):
        directory = temporary / name
        directory.mkdir()
        (directory / "spa_agent.c").write_text(changed)
        (directory / "spa_agent.h").write_bytes(originals["spa_agent.h"])
        (directory / "test.c").write_bytes(originals[test])
        binary = directory / "gate"
        entry = {"build": invoke(compiler + ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-O2", "-pthread", "-I", directory, "-I", ROOT, directory / "test.c",
                    directory / "spa_agent.c", notorch, "-lm", "-o", binary])}
        if entry["build"]["returncode"] == 0:
            entry["run"] = invoke([binary] + ([gate] if gate else []))
        return entry

    report["baseline"] = execute("baseline", source)
    if report["baseline"].get("run", {}).get("returncode") != 0:
        print("FAIL unmodified conditioned gate", flush=True)
        report["passed"] = False
        return report
    print("PASS unmodified conditioned gate", flush=True)
    for name in ("repeated", "future"):
        entry = {"name": name}
        entry.update(execute("legacy-" + name, source, test=f"tests/test_spa_agent_{name}.c"))
        entry["passed"] = entry.get("run", {}).get("returncode") == 0
        report["legacy_gates"].append(entry)
        print(("PASS " if entry["passed"] else "FAIL ") + "existing " + name + " gate", flush=True)
    mutations = (
        ("bypassed_conditioning", "r.comparison.targets[i]=(float)((double)r.comparison.targets[i]/1.0);"),
        ("reversed_conditioned_credit", "r.comparison.targets[i]=(float)(-(double)r.comparison.targets[i]/r.scale);"),
        ("normalize_before_mean", PER_REPEAT),
    )
    for name, replacement in mutations:
        changed = conditioned_replace(source, replacement)
        entry = {"name": name, "gate": "aggregate", "source_anchor": "NT_SPA_CONDITIONED_TARGET",
                 "mutant_source_sha256": digest(changed.encode())}
        entry.update(execute(name, changed, "aggregate"))
        result = entry.get("run", {})
        entry["caught"] = result.get("returncode") == 1 and "FAIL aggregate:" in result.get("stderr", "")
        report["mutations"].append(entry)
        print(("PASS caught " if entry["caught"] else "FAIL escaped or invalid ") + name, flush=True)
    if sanitize:
        binary = temporary / "sanitized-gate"
        entry = {"build": invoke(compiler + ["-std=gnu11", "-O1", "-g", "-pthread", "-I", ROOT,
            "-fno-omit-frame-pointer", "-fsanitize=address,undefined", ROOT / "tests/test_spa_agent_conditioned.c",
            ROOT / "spa_agent.c", ROOT / "notorch.c", "-lm", "-o", binary])}
        if entry["build"]["returncode"] == 0:
            environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1")
            entry["environment"] = {"ASAN_OPTIONS": environment["ASAN_OPTIONS"], "UBSAN_OPTIONS": environment["UBSAN_OPTIONS"]}
            entry["run"] = invoke([binary], env=environment)
        entry["passed"] = entry.get("run", {}).get("returncode") == 0
        report["sanitizers"] = entry
        print(("PASS " if entry["passed"] else "FAIL ") + "ASan/UBSan", flush=True)
    report["working_sources_unchanged"] = all((ROOT / name).read_bytes() == raw for name, raw in originals.items())
    report["passed"] = (report["working_sources_unchanged"] and
                        all(x["caught"] for x in report["mutations"]) and
                        all(x["passed"] for x in report["legacy_gates"]) and
                        report.get("sanitizers", {}).get("passed", True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="spa-conditioned-mutations-") as temporary:
        report = portable(run(compiler, Path(temporary), args.sanitize), Path(temporary))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS conditioned mutations" if report["passed"] else "FAIL conditioned mutations")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
