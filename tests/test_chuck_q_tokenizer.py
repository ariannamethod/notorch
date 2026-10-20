#!/usr/bin/env python3
"""Native Q BPE gates: ordered IDs, byte roundtrip and malformed-table refusal."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "examples/chuck_q_tokenizer.h"
HOST = r'''
#include "chuck_q_tokenizer.h"
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    q_tokenizer q; char error[256];
    if (q_tokenizer_load(&q, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    FILE *f = fopen(argv[2], "rb"); if (!f) return 2;
    if (fseek(f, 0, SEEK_END)) return 2;
    long n = ftell(f); if (n < 0 || fseek(f, 0, SEEK_SET)) return 2;
    unsigned char *bytes = malloc((size_t)n + 1);
    if (!bytes || fread(bytes, 1, (size_t)n, f) != (size_t)n || fclose(f)) return 2;
    uint32_t *ids; size_t count;
    if (q_tokenizer_encode(&q, bytes, (size_t)n, &ids, &count, error, sizeof(error))) return 2;
    for (size_t i = 0; i < count; ++i) printf("%u ", ids[i]);
    puts("");
    for (size_t i = 0; i < count; ++i)
        if (q_tokenizer_emit(&q, ids[i], stdout, error, sizeof(error))) return 2;
    free(ids); free(bytes);
    return fflush(stdout) ? 2 : 0;
}
'''


def table(pairs):
    return struct.pack("<I", len(pairs)) + b"".join(
        struct.pack("<III", a, b, 256 + i) for i, (a, b) in enumerate(pairs))


def reference(raw, pairs):
    ids = list(raw)
    for rank, pair in enumerate(pairs):
        result, at = [], 0
        while at < len(ids):
            if tuple(ids[at:at + 2]) == pair:
                result.append(256 + rank)
                at += 2
            else:
                result.append(ids[at])
                at += 1
        ids = result
    return ids


class QTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="chuck-q-bpe-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.path = Path(cls.temp.name)
        (cls.path / "chuck_q_tokenizer.h").write_bytes(HEADER.read_bytes())
        (cls.path / "host.c").write_text(HOST)
        cls.binary = cls.compile("host")

    @classmethod
    def compile(cls, name):
        binary = cls.path / name
        subprocess.run([os.environ.get("CC", "cc"), "-O2", "-std=c11", "-Wall", "-Wextra",
                        "-I", str(cls.path), str(cls.path / "host.c"), "-o", str(binary)],
                       capture_output=True, check=True, timeout=60)
        return binary

    def run_case(self, merges, raw, binary=None):
        merge_file, text_file = self.path / "merges", self.path / "text"
        merge_file.write_bytes(merges); text_file.write_bytes(raw)
        return subprocess.run([str(binary or self.binary), str(merge_file), str(text_file)],
                              capture_output=True, timeout=10)

    def test_ordered_merges_and_exact_byte_roundtrip(self):
        pairs = [(97, 98), (256, 99), (97, 97), (258, 97)]
        for raw in (b"abc ab aaaaa aababc", bytes(range(256)), "שלום κόσμε!".encode(), b""):
            with self.subTest(raw=raw):
                run = self.run_case(table(pairs), raw)
                self.assertEqual(run.returncode, 0, run.stderr)
                ids, decoded = run.stdout.split(b"\n", 1)
                self.assertEqual(list(map(int, ids.split())), reference(raw, pairs))
                self.assertEqual(decoded, raw)

    def test_zero_merge_table_preserves_all_bytes(self):
        raw = bytes(range(256))
        run = self.run_case(table([]), raw)
        self.assertEqual(run.returncode, 0, run.stderr)
        ids, decoded = run.stdout.split(b"\n", 1)
        self.assertEqual(list(map(int, ids.split())), list(raw))
        self.assertEqual(decoded, raw)

    def test_malformed_merge_tables_are_refused(self):
        valid = table([(97, 98)])
        bad_id = struct.pack("<IIII", 1, 97, 98, 257)
        too_long = table([(97, 97)] + [(255 + i, 255 + i) for i in range(1, 6)])
        for label, data in (("short", b"\x01"), ("truncated", valid[:-1]),
                            ("trailing", valid + b"x"), ("bad-id", bad_id),
                            ("future-reference", table([(256, 97)])),
                            ("too-many", struct.pack("<I", 3841)),
                            ("too-long-token", too_long)):
            with self.subTest(label=label):
                run = self.run_case(data, b"ab")
                self.assertEqual(run.returncode, 1)
                self.assertTrue(run.stderr)
                self.assertFalse(run.stdout)

    def test_reversed_pair_defect_breaks_id_gate(self):
        header = self.path / "chuck_q_tokenizer.h"
        original = header.read_text()
        needle = "tokens[i] == a && tokens[i + 1] == b"
        self.assertEqual(original.count(needle), 1)
        try:
            header.write_text(original.replace(needle, "tokens[i] == b && tokens[i + 1] == a"))
            mutant = self.compile("mutant")
        finally:
            header.write_text(original)
        pairs, raw = [(97, 98), (256, 99)], b"abc ab"
        run = self.run_case(table(pairs), raw, mutant)
        self.assertEqual(run.returncode, 0, run.stderr)
        ids = list(map(int, run.stdout.split(b"\n", 1)[0].split()))
        with self.assertRaises(AssertionError):
            self.assertEqual(ids, reference(raw, pairs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
