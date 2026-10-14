#!/usr/bin/env python3
"""Compile bounded red-hand defects against the real libnotorch boundary gate."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("all parameters", "harness/arch_penelope.c", "if (!b) return a;", "if (1) return a;"),
    ("feed-forward norm", "harness/arch_penelope.c", "b2 ? b2->ffn_norm : NULL", "NULL"),
    ("learned positions", "harness/arch_penelope.c", "pe = alpha*pe + (1.f-alpha)*other->position_data[position+d];", "pe = pe;"),
    ("masked RMS overflow", "harness/arch_penelope.c", "if (!isfinite(ss)) return 0;", "/* deliberately accept overflowing accumulator */"),
    ("complete identity", "pen7.c", "digest(m->snapshot,bytes,m->info.identity)", "digest(m->snapshot,bytes/2,m->info.identity)"),
    ("canonical lowercase", "pen7.c", "c>='A'&&c<='Z' ? c-'A'+'a' : c", "c"),
]


def main():
    baseline = subprocess.run([str(ROOT / "test_pen7")], cwd=ROOT, capture_output=True, text=True)
    if baseline.returncode or "PEN7_OK" not in baseline.stdout:
        raise RuntimeError("baseline gate failed: " + baseline.stdout + baseline.stderr)
    with tempfile.TemporaryDirectory(prefix="pen7_red_") as temp:
        for name, source, before, after in MUTATIONS:
            text = (ROOT / source).read_text()
            if text.count(before) != 1:
                raise RuntimeError(f"mutation anchor changed: {name}")
            mutant = Path(temp) / "mutant.c"
            mutant.write_text(text.replace(before, after))
            binary = Path(temp) / "gate"
            build = subprocess.run([os.environ.get("CC", "cc"), "-O2", "-std=gnu11", "-pthread", "-I.",
                                    "tests/test_pen7.c", str(mutant), "libnotorch.a", "-lm", "-o", str(binary)],
                                   cwd=ROOT, capture_output=True, text=True)
            if build.returncode:
                raise RuntimeError("mutation did not compile: " + name + "\n" + build.stderr)
            result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True)
            if result.returncode == 0 or "FAIL " not in result.stderr:
                raise RuntimeError("gate did not detect: " + name + "\n" + result.stdout + result.stderr)
            print("DETECTED", name)
    print(f"PEN7_MUTATIONS_OK ({len(MUTATIONS)} defects)")


if __name__ == "__main__":
    main()
