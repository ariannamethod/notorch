#!/usr/bin/env python3
"""Prove the sampling gate rejects three defects without editing the checkout."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
original = (root / "notorch.c").read_text()
mutations = {
    "no_rejection": (
        "uint32_t threshold = (uint32_t)(0u - bound) % bound;",
        "uint32_t threshold = 0;",
    ),
    "inclusive_cdf": ("if (target < cumulative) {", "if (target <= cumulative) {"),
    "early_state": (
        "if (nt_categorical_index(weights, n, temperature, draw, &selected) != 0)",
        "*state = next;\n    if (nt_categorical_index(weights, n, temperature, draw, &selected) != 0)",
    ),
}
with tempfile.TemporaryDirectory(prefix="notorch-sampling-") as folder:
    tmp = Path(folder)
    for name in ["baseline", *mutations]:
        source = original
        if name != "baseline":
            before, after = mutations[name]
            if source.count(before) != 1:
                raise SystemExit(f"refused {name}: expected exactly one mutation site")
            source = source.replace(before, after)
        path = tmp / f"{name}.c"
        path.write_text(source)
        binary = tmp / name
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + ["-std=gnu11", "-O0", "-pthread", f"-I{root}",
               str(root / "tests/test_sampling.c"), str(path), "-lm", "-o", str(binary)],
            check=True,
        )
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        if name == "baseline":
            if result.returncode:
                raise SystemExit("baseline failed:\n" + result.stdout + result.stderr)
            print("PASS baseline: " + result.stdout.splitlines()[-1], flush=True)
        else:
            if result.returncode != 1 or "FAIL " not in result.stderr:
                raise SystemExit(f"{name}: expected assertion failure, got {result.returncode}\n"
                                 + result.stdout + result.stderr)
            print("KILLED " + name + ": " + result.stderr.splitlines()[0], flush=True)
print("PASS three sampling defects rejected; checkout unchanged")
