#!/usr/bin/env python3
"""Prove the value-kernel gate rejects three isolated defects; leave sources intact."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "notorch.c").read_text()
mutations = {
    "lost_bias": ("float value = b[r];", "float value = 0.0f;"),
    "wrong_tanh_vjp": ("dy[i] * (1.0f - y[i] * y[i])", "dy[i] * (1.0f + y[i] * y[i])"),
    "normal_invalid_call_advances": (
        "int nt_rng_normal_values(uint64_t* state, int n, float* out) {",
        "int nt_rng_normal_values(uint64_t* state, int n, float* out) {\n    (void)nt_rng_u32(state);",
    ),
}

with tempfile.TemporaryDirectory(prefix="notorch-values-mutations-") as directory:
    temp = Path(directory)
    for name, replacement in {"control": None, **mutations}.items():
        content = source
        if replacement:
            old, new = replacement
            if source.count(old) != 1:
                raise RuntimeError(f"{name}: expected one mutation site")
            content = source.replace(old, new, 1)
        candidate = temp / f"{name}.c"
        binary = temp / name
        candidate.write_text(content)
        build = subprocess.run([
            os.environ.get("CC", "cc"), "-std=gnu11", "-O1", "-pthread",
            "-I", str(root), str(root / "tests/test_numerical_values.c"),
            str(candidate), "-lm", "-o", str(binary),
        ], capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(f"{name}: build failed, not a rejected mutation:\n{build.stderr}")
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if name == "control":
            if run.returncode or "NUMERICAL_VALUES_OK" not in run.stdout:
                raise RuntimeError("control gate failed:\n" + run.stdout + run.stderr)
            print(run.stdout.rstrip().splitlines()[-1])
        elif run.returncode == 0 or "FAIL " not in run.stderr:
            raise RuntimeError(f"{name}: mutation not rejected by an assertion:\n{run.stdout}{run.stderr}")
        else:
            print(f"REJECTED {name}: {run.stderr.strip().splitlines()[0]}")
print("NUMERICAL_VALUES_MUTATIONS_OK 3/3")
