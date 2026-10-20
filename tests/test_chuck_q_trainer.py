#!/usr/bin/env python3
"""Bounded Q trainer, checkpoint and native QPTQ export gates; one C build."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from test_chuck_trainer import ROOT, SOURCES, events, policy_weights


def checkpoint_tensors(data):
    """Read canonical checkpoint slots independently of the QPTQ exporter."""
    life = data.index(b"NTCALIFE")
    version, payload = struct.unpack_from("<II", data, life + 8)
    if (version, payload) != (1, 992):
        raise AssertionError("unexpected policy envelope")
    at = life + 24 + payload
    count, = struct.unpack_from("<I", data, at)
    at += 4
    tensors = []
    for _ in range(count):
        ndim, = struct.unpack_from("<I", data, at)
        at += 4
        if ndim not in (1, 2):
            raise AssertionError("invalid tensor rank")
        shape = struct.unpack_from("<" + "I" * ndim, data, at)
        at += ndim * 4
        length, _ = struct.unpack_from("<II", data, at)
        if length != shape[0] * (shape[1] if ndim == 2 else 1):
            raise AssertionError("invalid tensor length")
        # length, Adam t, then local Chuck: 16 gradient floats + 5 scalars.
        at += 8 + 84
        tensors.append((shape, data[at:at + length * 4]))
        at += length * 12  # weights, first moment, second moment
    if at != len(data) - 32:
        raise AssertionError("checkpoint tensors do not consume payload")
    return tensors


class ChuckQTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chuck-q-trainer-gates-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / "source"
        for name in (*SOURCES, "examples/chuck_q_body.h", "examples/chuck_q_tokenizer.h"):
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        cls.env = dict(os.environ, NT_SIMD_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        cls.binary = cls.directory / "train_chuck_q"
        command = [os.environ.get("CC", "cc"), "-O2", "-std=gnu11", "-pthread",
                   "-DNT_CHUCK_Q_TRAINER", "-DNT_CHUCK_ARCHITECT_TEST", "-I", str(source),
                   str(source / "examples/chuck_architect_train.c"), str(source / "notorch.c"),
                   str(source / "sha256.c"), "-lm", "-o", str(cls.binary)]
        build = subprocess.run(command, env=cls.env, capture_output=True, timeout=120)
        if build.returncode:
            raise AssertionError(build.stderr.decode(errors="replace"))
        cls.corpus = cls.directory / "corpus.txt"
        cls.corpus.write_bytes(b"abc ab cab. Q learns from loss.\n" * 240)
        cls.merges = cls.directory / "q.merges"
        cls.merges.write_bytes(struct.pack("<7I", 2, 97, 98, 256, 256, 99, 257))
        cls.full, cls.partial, cls.initial = (cls.directory / name for name in
                                             ("full.ckpt", "partial.ckpt", "initial.ckpt"))
        for label, checkpoint, steps in (("full", cls.full, 73), ("partial", cls.partial, 31),
                                          ("initial", cls.initial, 0)):
            result = cls.invoke("train", "--corpus", cls.corpus, "--merges", cls.merges,
                                "--checkpoint", checkpoint, "--steps", steps,
                                "--trace", cls.directory / (label + ".jsonl"))
            if result.returncode:
                raise AssertionError(result.stderr.decode(errors="replace"))

    @classmethod
    def invoke(cls, *args):
        return subprocess.run([str(cls.binary), *map(str, args)], env=cls.env,
                              capture_output=True, timeout=60)

    def good(self, *args):
        result = self.invoke(*args)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result

    def bad(self, *args):
        result = self.invoke(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr)
        return result

    def test_resume_and_initial_checkpoint_match_uninterrupted_training(self):
        for start, checkpoint in ((31, self.partial), (0, self.initial)):
            with self.subTest(start=start):
                output = self.directory / f"resumed-{start}.ckpt"
                trace = self.directory / f"resumed-{start}.jsonl"
                self.good("resume", "--corpus", self.corpus, "--checkpoint", checkpoint,
                          "--output", output, "--steps", 73, "--trace", trace)
                self.assertEqual(output.read_bytes(), self.full.read_bytes())
                expected = [r for r in events(self.directory / "full.jsonl")
                            if r["type"] in ("step", "acquisition") and r["step"] > start]
                actual = [r for r in events(trace) if r["type"] in ("step", "acquisition")]
                self.assertEqual(actual, expected)

    def test_learned_policy_changes_and_acts(self):
        self.assertNotEqual(policy_weights(self.initial), policy_weights(self.full))
        rows = events(self.directory / "full.jsonl")
        acquired = [r for r in rows if r["type"] == "acquisition"]
        self.assertEqual([r["step"] for r in acquired], [1, 65])
        steps = {r["step"]: r for r in rows if r["type"] == "step"}
        for row in acquired:
            self.assertTrue(row["source_restored"])
            self.assertNotEqual(row["policy_before"], row["policy_after"])
            self.assertEqual(row["action_after"], steps[row["step"]]["action"])
        self.assertEqual(rows[-1]["fit_updates"], 64)

    def test_bpe_generation_is_deterministic_and_read_only(self):
        original = self.full.read_bytes()
        for temperature in (0, .8):
            args = ("generate", "--checkpoint", self.full, "--prompt", "abc ab ",
                    "--tokens", 40, "--temperature", temperature, "--seed", 11)
            first, second = self.good(*args), self.good(*args)
            self.assertEqual(first.stdout, second.stdout)
            self.assertTrue(first.stdout.startswith(b"abc ab "))
            # Output is bytes: 40 tokens, each one byte, 'ab', or 'abc'.
            self.assertGreaterEqual(len(first.stdout), 7 + 40)
            self.assertLessEqual(len(first.stdout), 7 + 120)
        self.assertEqual(self.full.read_bytes(), original)

    def test_qptq_export_matches_every_checkpoint_tensor(self):
        original = self.full.read_bytes()
        output = self.directory / "qptq.bin"
        self.good("export", "--checkpoint", self.full, "--output", output)
        data = output.read_bytes()
        self.assertEqual(struct.unpack_from("<11I", data),
                         (0x51505451, 1, 258, 8, 2, 1, 8, 0, 1, 1, 4))
        tensors = checkpoint_tensors(original)
        self.assertEqual(len(tensors), 10)
        self.assertEqual(tensors[0][0], (258, 8))
        self.assertEqual(tensors[1][0], (8, 8))
        expected = bytearray()
        for slot, (shape, weights) in enumerate(tensors):
            if slot == 6:
                self.assertEqual(shape, (2, 9))
                expected.extend(weights[:32])
                expected.extend(weights[36:68])
                expected.extend(weights[32:36])
                expected.extend(weights[68:72])
            else:
                expected.extend(weights)
        self.assertEqual(data[44:], expected)
        self.assertEqual(len(data), 44 + sum(len(w) for _, w in tensors))
        self.assertEqual(self.full.read_bytes(), original)
        self.bad("export", "--checkpoint", self.full, "--output", self.full)
        self.bad("export", "--checkpoint", self.full, "--output", output)
        self.assertEqual(self.full.read_bytes(), original)
        self.assertEqual(output.read_bytes(), data)

    def test_malformed_merges_and_foreign_corpus_are_refused(self):
        cases = (b"", self.merges.read_bytes()[:-1], self.merges.read_bytes() + b"x",
                 struct.pack("<4I", 1, 97, 98, 999),
                 struct.pack("<4I", 1, 256, 98, 256))
        for i, data in enumerate(cases):
            with self.subTest(merge=i):
                merges = self.directory / f"bad-{i}.merges"; merges.write_bytes(data)
                checkpoint = self.directory / f"bad-{i}.ckpt"
                self.bad("train", "--corpus", self.corpus, "--merges", merges,
                         "--checkpoint", checkpoint, "--steps", 0)
                self.assertFalse(checkpoint.exists())
        foreign = self.directory / "foreign.txt"
        foreign.write_bytes(self.corpus.read_bytes().replace(b"abc", b"abd", 1))
        output = self.directory / "foreign.ckpt"
        self.bad("resume", "--corpus", foreign, "--checkpoint", self.partial,
                 "--output", output, "--steps", 73)
        self.assertFalse(output.exists())
        short = self.directory / "short.txt"; short.write_bytes(b"abc")
        self.bad("train", "--corpus", short, "--merges", self.merges,
                 "--checkpoint", output, "--steps", 0)
        self.assertFalse(output.exists())

    def test_corrupt_merge_graph_and_other_checkpoint_family_are_refused(self):
        original = self.full.read_bytes()
        self.assertEqual(original[:8], b"NTCHTRQ1")
        self.assertEqual(struct.unpack_from("<5I", original, 176), (2, 97, 98, 256, 99))
        corrupt_merge = bytearray(original)
        struct.pack_into("<I", corrupt_merge, 180, 256)  # self-reference in first merge
        other_family = bytearray(original)
        other_family[:8] = b"NTCHTRN1"
        for label, data in (("merge", corrupt_merge), ("simple-family", other_family)):
            with self.subTest(case=label):
                data[-32:] = hashlib.sha256(data[:-32]).digest()
                checkpoint = self.directory / f"invalid-{label}.ckpt"
                checkpoint.write_bytes(data)
                self.bad("generate", "--checkpoint", checkpoint, "--tokens", 1)
                self.assertEqual(checkpoint.read_bytes(), data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
