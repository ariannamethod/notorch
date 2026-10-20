#!/usr/bin/env python3
"""Bounded native gates for training, full-process resumption and generation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h",
    "chuck_architect_impl.h", "sha256.c", "sha256.h",
    "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h",
    "examples/chuck_architect_rollout.h", "examples/chuck_architect_lived.h",
    "examples/chuck_architect_living.h", "examples/chuck_trainer_state.h",
    "examples/chuck_trainer_checkpoint.h", "examples/chuck_trainer.h",
)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def policy_weights(checkpoint):
    """Canonical NTCALIFE v1: 24-byte envelope, 140-byte state prefix, 163 floats."""
    data = checkpoint.read_bytes()
    offset = data.index(b"NTCALIFE")
    version, payload_bytes = struct.unpack_from("<II", data, offset + 8)
    if version != 1 or payload_bytes != 992:
        raise AssertionError("unexpected embedded policy format")
    return data[offset + 164:offset + 164 + 163 * 4]


class ChuckTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chuck-trainer-gates-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.source = cls.directory / "source"
        for name in SOURCES:
            destination = cls.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        cls.env = dict(os.environ, NT_SIMD_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        cls.binary = cls.directory / "train_chuck"
        cls.compile(cls.binary)
        cls.corpus = cls.directory / "corpus.txt"
        cls.corpus.write_text(("Alpha beta. Chuck learns. שלום κόσμε!\n" * 240), encoding="utf-8")
        cls.full, cls.partial, cls.initial = (cls.directory / name for name in
                                             ("full.ckpt", "partial.ckpt", "initial.ckpt"))
        for label, checkpoint, steps in (("full", cls.full, 73), ("partial", cls.partial, 31),
                                          ("initial", cls.initial, 0)):
            result = cls.invoke("train", "--corpus", cls.corpus, "--checkpoint", checkpoint,
                                "--steps", steps, "--trace", cls.directory / (label + ".jsonl"))
            if result.returncode:
                raise AssertionError(result.stderr.decode(errors="replace"))

    @classmethod
    def compile(cls, binary):
        command = [os.environ.get("CC", "cc"), "-O2", "-std=gnu11", "-pthread",
                   "-DNT_CHUCK_TRAINER", "-DNT_CHUCK_ARCHITECT_TEST", "-I", str(cls.source),
                   str(cls.source / "examples/chuck_architect_train.c"), str(cls.source / "notorch.c"),
                   str(cls.source / "sha256.c"), "-lm", "-o", str(binary)]
        result = subprocess.run(command, env=cls.env, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors="replace"))

    @classmethod
    def invoke(cls, *args, binary=None):
        return subprocess.run([str(binary or cls.binary), *map(str, args)], env=cls.env,
                              capture_output=True, timeout=60)

    def good(self, *args, **kwargs):
        result = self.invoke(*args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result

    def bad(self, *args):
        result = self.invoke(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr, "refusal must explain the failure")
        return result

    def test_full_checkpoint_resumes_exactly_across_processes(self):
        for label, checkpoint, start in (("resumed", self.partial, 31), ("from-zero", self.initial, 0)):
            with self.subTest(start=start):
                output = self.directory / (label + ".ckpt")
                trace = self.directory / (label + ".jsonl")
                self.good("resume", "--corpus", self.corpus, "--checkpoint", checkpoint,
                          "--output", output, "--steps", 73, "--trace", trace)
                self.assertEqual(output.read_bytes(), self.full.read_bytes())
                original = [r for r in events(self.directory / "full.jsonl")
                            if r["type"] in ("step", "acquisition") and r["step"] > start]
                continued = [r for r in events(trace) if r["type"] in ("step", "acquisition")]
                self.assertEqual(continued, original)

        unchanged = self.directory / "no-op.ckpt"
        self.good("resume", "--corpus", self.corpus, "--checkpoint", self.partial,
                  "--output", unchanged, "--steps", 31)
        self.assertEqual(unchanged.read_bytes(), self.partial.read_bytes())

    def test_resume_window_rng_defect_is_caught(self):
        """A valid checkpoint must fail continuation parity if its RNG is lost."""
        header = self.source / "examples/chuck_trainer_checkpoint.h"
        original = header.read_text()
        needle = "    *out = s;"
        self.assertEqual(original.count(needle), 1, "resume mutation must remain surgical")
        mutant = self.directory / "train_chuck_lost_window_rng"
        try:
            header.write_text(original.replace(needle,
                              "    s.windows = 1; /* deliberate resume defect */\n" + needle))
            self.compile(mutant)
        finally:
            header.write_text(original)
        output = self.directory / "mutant.ckpt"
        trace = self.directory / "mutant.jsonl"
        self.good("resume", "--corpus", self.corpus, "--checkpoint", self.partial,
                  "--output", output, "--steps", 73, "--trace", trace, binary=mutant)
        # Build failure or refused input cannot count as detecting this defect:
        # the mutated trainer completed, but the normal exact-resume gate goes red.
        with self.assertRaises(AssertionError):
            self.assertEqual(output.read_bytes(), self.full.read_bytes())
        actual = next(r for r in events(trace) if r["type"] == "step")
        expected = next(r for r in events(self.directory / "full.jsonl")
                        if r["type"] == "step" and r["step"] == 32)
        self.assertNotEqual(actual["window_rng"], expected["window_rng"])

    def test_sigterm_saves_completed_step_and_resumes_exactly(self):
        checkpoint = self.directory / "interrupted.ckpt"
        trace = self.directory / "interrupted.jsonl"
        command = [str(self.binary), "train", "--corpus", str(self.corpus),
                   "--checkpoint", str(checkpoint), "--steps", "100000",
                   "--trace", str(trace)]
        process = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            observed_step = False
            while time.monotonic() < deadline and process.poll() is None:
                if trace.exists() and b'"type":"step"' in trace.read_bytes():
                    observed_step = True
                    break
                time.sleep(.005)
            self.assertTrue(observed_step, "trainer did not reach its first step before deadline")
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 143, stderr.decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
        rows = events(trace)
        complete = rows[-1]
        self.assertEqual(complete["type"], "complete")
        self.assertTrue(complete["interrupted"])
        steps = [r for r in rows if r["type"] == "step"]
        self.assertEqual(complete["step"], steps[-1]["step"])
        self.assertEqual(sum(complete["action_counts"]), complete["step"])
        self.assertTrue(checkpoint.is_file())
        target = max(73, complete["step"] + 3)
        resumed = self.directory / "after-signal.ckpt"
        self.good("resume", "--corpus", self.corpus, "--checkpoint", checkpoint,
                  "--output", resumed, "--steps", target)
        reference = self.full
        if target != 73:
            reference = self.directory / "signal-reference.ckpt"
            self.good("train", "--corpus", self.corpus, "--checkpoint", reference,
                      "--steps", target)
        self.assertEqual(resumed.read_bytes(), reference.read_bytes())

    def test_policy_acquires_and_executes_measured_actions(self):
        rows = events(self.directory / "full.jsonl")
        acquired = [r for r in rows if r["type"] == "acquisition"]
        self.assertEqual([r["step"] for r in acquired], [1, 65])
        self.assertNotEqual(policy_weights(self.initial), policy_weights(self.full))
        steps = {r["step"]: r for r in rows if r["type"] == "step"}
        for row in acquired:
            self.assertTrue(row["source_restored"])
            self.assertEqual(row["fits"], 32)
            self.assertEqual(row["horizon"], 16)
            self.assertNotEqual(row["policy_before"], row["policy_after"])
            self.assertEqual(row["action_after"], steps[row["step"]]["action"])
        complete = rows[-1]
        self.assertEqual(complete["acquisitions"], 2)
        self.assertEqual(complete["fit_updates"], 64)
        self.assertEqual(complete["branch_updates"], 96)
        self.assertEqual(sum(complete["action_counts"]), 73)

    def test_generation_is_deterministic_and_checkpoint_is_self_contained(self):
        original = self.full.read_bytes()
        prompt = "שלום "
        for temperature in (0, .8):
            with self.subTest(temperature=temperature):
                args = ("generate", "--checkpoint", self.full, "--prompt", prompt,
                        "--tokens", 40, "--seed", 11, "--temperature", temperature)
                left, right = self.good(*args), self.good(*args)
                self.assertEqual(left.stdout, right.stdout)
                text = left.stdout.decode("utf-8")
                self.assertTrue(text.startswith(prompt))
                self.assertEqual(len(text), len(prompt) + 40)
                self.assertLessEqual(set(text), set(self.corpus.read_text()))
        self.assertEqual(self.full.read_bytes(), original)
        self.bad("generate", "--checkpoint", self.full, "--prompt", "🙂", "--tokens", 1)

    def test_checkpoint_corruption_truncation_and_extra_bytes_are_refused(self):
        original = self.full.read_bytes()
        damaged = bytearray(original); damaged[len(damaged) // 2] ^= 1
        invalid_vocab = bytearray(original)
        struct.pack_into("<I", invalid_vocab, 32, 0)
        invalid_vocab[-32:] = hashlib.sha256(invalid_vocab[:-32]).digest()
        nul_scalar = bytearray(original)
        # V1 fixed metadata ends at176; the sorted scalar table follows.
        struct.pack_into("<I", nul_scalar, 176, 0)
        nul_scalar[-32:] = hashlib.sha256(nul_scalar[:-32]).digest()
        for label, data in (("magic", b"bad!" + original[4:]), ("truncated", original[:-1]),
                            ("corrupted", damaged), ("extra", original + b"x"),
                            ("valid-digest-invalid-vocab", invalid_vocab),
                            ("valid-digest-nul-scalar", nul_scalar)):
            with self.subTest(case=label):
                path = self.directory / (label + ".ckpt")
                path.write_bytes(data)
                self.bad("generate", "--checkpoint", path, "--tokens", 1)
                self.assertEqual(path.read_bytes(), data)

    def test_foreign_corpus_and_invalid_utf8_are_refused(self):
        foreign = self.directory / "foreign.txt"
        foreign.write_bytes(self.corpus.read_bytes().replace(b"Alpha", b"alpha", 1))
        output = self.directory / "foreign.ckpt"
        self.bad("resume", "--corpus", foreign, "--checkpoint", self.partial,
                 "--output", output, "--steps", 73)
        self.assertFalse(output.exists())
        for i, raw in enumerate((b"abc\xc0\xaf", b"abc\xed\xa0\x80", b"abc\xf4\x90\x80\x80",
                                 b"abc\xe2\x82", b"abc\x00def")):
            with self.subTest(utf8=i):
                path = self.directory / f"bad-utf8-{i}.txt"; path.write_bytes(raw)
                destination = self.directory / f"bad-utf8-{i}.ckpt"
                self.bad("train", "--corpus", path, "--checkpoint", destination, "--steps", 0)
                self.assertFalse(destination.exists())

    def test_output_aliases_and_existing_files_are_preserved(self):
        corpus, checkpoint = self.corpus.read_bytes(), self.full.read_bytes()
        self.bad("train", "--corpus", self.corpus, "--checkpoint", self.full, "--steps", 1)
        alias = self.directory / "corpus-alias.txt"; alias.symlink_to(self.corpus)
        self.bad("train", "--corpus", self.corpus, "--checkpoint", alias, "--steps", 1)
        self.bad("resume", "--corpus", self.corpus, "--checkpoint", self.partial,
                 "--output", self.corpus, "--steps", 73)
        self.bad("resume", "--corpus", self.corpus, "--checkpoint", self.partial,
                 "--output", self.directory / "trace-alias.ckpt", "--steps", 73, "--trace", self.partial)
        self.assertEqual(self.corpus.read_bytes(), corpus)
        self.assertEqual(self.full.read_bytes(), checkpoint)
        inplace = self.directory / "inplace.ckpt"; inplace.write_bytes(self.partial.read_bytes())
        self.good("resume", "--corpus", self.corpus, "--checkpoint", inplace,
                  "--output", inplace, "--steps", 73)
        self.assertEqual(inplace.read_bytes(), self.full.read_bytes())

    def test_invalid_arguments_are_refused_before_training(self):
        output = self.directory / "invalid.ckpt"
        for option, value in (("--steps", "-1"), ("--steps", "abc"), ("--seed", "0"),
                              ("--lr", "nan"), ("--save-every", "0"), ("--acquire-every", "0")):
            with self.subTest(option=option, value=value):
                args = ["train", "--corpus", self.corpus, "--checkpoint", output]
                if option != "--steps": args += ["--steps", "1"]
                self.bad(*args, option, value)
                self.assertFalse(output.exists())
        self.bad("resume", "--corpus", self.corpus, "--checkpoint", self.partial,
                 "--output", output, "--steps", 30)
        self.assertFalse(output.exists())
        self.good("--help")


if __name__ == "__main__":
    unittest.main(verbosity=2)
