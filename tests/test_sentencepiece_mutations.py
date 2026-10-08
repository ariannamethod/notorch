#!/usr/bin/env python3
"""Isolated red-hand defects: the checked-out source stays untouched."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "sentencepiece.c").read_text()
MUTATIONS = [
    ("last equal-score path wins", "candidate > path[end].score", "candidate >= path[end].score", 2, "test_sentencepiece"),
    ("normalizer trie bypassed", "if (m->map_units) {", "if (0 && m->map_units) {", 1, "test_sentencepiece"),
    ("unknown runs split", "if (r.count && r.pieces[r.count - 1].id", "if (0 && r.count && r.pieces[r.count - 1].id", 1, "test_sentencepiece"),
    ("failed output published", 'if (!path) { reason = "out of memory finding tokenizer path";',
     'if (!path) { *out = r; reason = "out of memory finding tokenizer path";', 1, "test_sentencepiece_faults"),
]


def main():
    for name, old, new, count, gate in MUTATIONS:
        if SOURCE.count(old) != count:
            raise SystemExit(f"mutation anchor changed: {name}")
        with tempfile.TemporaryDirectory(prefix="notorch-spm-mutation-") as directory:
            work = Path(directory)
            (work / "tests").mkdir()
            (work / "sentencepiece.c").write_text(SOURCE.replace(old, new))
            shutil.copyfile(ROOT / "sentencepiece.h", work / "sentencepiece.h")
            for file in (gate + ".c", "sentencepiece_reference.h"):
                shutil.copyfile(ROOT / "tests" / file, work / "tests" / file)
            command = ["cc", "-O2", "-std=c11", "-I.", "tests/" + gate + ".c"]
            if gate == "test_sentencepiece":
                command.append("sentencepiece.c")
            command += ["-lm", "-lpthread", "-o", "gate"]
            subprocess.run(command, cwd=work, check=True, capture_output=True)
            result = subprocess.run([str(work / "gate")], cwd=work, capture_output=True, text=True)
            if result.returncode == 0 or "FAIL" not in result.stderr:
                raise SystemExit(f"gate missed {name}: {result.stdout}\n{result.stderr}")
            print(f"PASS detects {name}: {result.stderr.strip().splitlines()[0]}")


if __name__ == "__main__":
    main()
