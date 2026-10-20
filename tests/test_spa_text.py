"""Executable native-body and complete SPA text-host contracts.

    make spa_text BLAS_FLAGS= BLAS_LIBS=
    python3 tests/test_spa_text.py
"""
import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "examples")]
import SPA
from spa_text import BodyLibrary, TextHost


def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


class TextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.native = SPA.Native(ROOT / ("libnotorch." + suffix))
        cls.library = BodyLibrary(ROOT / ("spa_text_body." + suffix))
        host, cls.losses = TextHost.trained(native=cls.native, library=cls.library)
        cls.snapshot = host.snapshot()
        host.close()

    def setUp(self):
        self.host = TextHost(self.native, self.library)
        self.host.restore(self.snapshot)
        self.addCleanup(self.host.close)

    def engine(self, host=None, **options):
        config = SPA.Config.default(native=self.native, mode=SPA.Mode.LEARNED,
                                    seed=42, exploration=.5, **options)
        return SPA.Engine(host or self.host, native=self.native, config=config)

    def test_real_native_training_and_generation(self):
        self.assertEqual(len(self.host.words), 44)
        self.assertLess(self.losses[1], self.losses[0] * .5)
        self.assertTrue(all(1 <= len(s) <= 24 for s in self.host.sentences))
        self.assertTrue(all(all(0 <= t < 44 for t in s) for s in self.host.sentences))
        # The learned token embeddings supply SPA's sentence field.
        observation, metrics = self.host.observe(1)
        self.assertEqual(observation.sentence_count, 4)
        self.assertGreater(metrics.coherence, 0)
        self.assertEqual(self.host.snapshot(), self.snapshot)

    def test_native_checkpoint_exact_generator_continuation(self):
        raw = self.host.body.snapshot()
        clone = self.library.restore(raw)
        self.addCleanup(clone.close)
        self.assertEqual(clone.snapshot(), raw)
        for token in (1, 2, 3, 5, 9):
            previous = clone.snapshot()
            self.assertEqual(self.host.body.generate(token, 0), clone.generate(token, 0))
            self.assertEqual(self.host.body.snapshot(), clone.snapshot())
            self.assertNotEqual(previous, clone.snapshot())

    def test_native_rejects_corruption_and_nonfinite_weights(self):
        raw = self.host.body.snapshot()
        damaged = bytearray(raw)
        damaged[30] ^= 1
        with self.assertRaises(ValueError):
            self.library.restore(bytes(damaged))
        # A repaired checksum cannot make a NaN weight a valid body.
        damaged = bytearray(raw)
        struct.pack_into("<I", damaged, 24, 0x7FC00000)
        struct.pack_into("<Q", damaged, len(damaged) - 8, checksum(damaged[:-8]))
        with self.assertRaises(ValueError):
            self.library.restore(bytes(damaged))
        with self.assertRaises(ValueError):
            self.library.restore(raw + b"x")
        self.assertEqual(self.host.body.snapshot(), raw)

    def test_native_refusal_preserves_body(self):
        raw = self.host.body.snapshot()
        tokens = (C.c_uint32 * 2)(0, 9999)
        losses = (C.c_float * 2)(-7, -9)
        self.assertEqual(self.library.lib.spa_text_train(self.host.body.pointer, tokens, 2, 4, 1, losses), -1)
        out = (C.c_uint32 * 4)(7, 7, 7, 7)
        self.assertEqual(self.library.lib.spa_text_generate(self.host.body.pointer, 1, 0, 0, out, 4), -1)
        self.assertEqual(list(out), [7] * 4)
        self.assertEqual(list(losses), [-7, -9])
        self.assertEqual(self.host.body.snapshot(), raw)

    def test_host_restore_validates_before_committing(self):
        for field, value in (("sentences", [[99999]] * 4), ("cursor", -1),
                             ("reseeds", [0, 0, 0, True]), ("temperature", float("nan")),
                             ("words", [".", "a"]), ("max_tokens", 1)):
            with self.subTest(field=field):
                state = json.loads(self.snapshot)
                state[field] = value
                with self.assertRaises(ValueError):
                    self.host.restore(json.dumps(state).encode())
                self.assertEqual(self.host.snapshot(), self.snapshot)
        with self.assertRaises(ValueError):
            TextHost.trained("a" * 257 + ".", native=self.native, library=self.library)

    def test_actions_and_online_credit(self):
        engine = self.engine()
        policy = bytes(engine.agent.state.policy)
        kinds, rewards = set(), []
        for _ in range(40):
            target = self.host.cursor % 4
            before = [list(s) for s in self.host.sentences]
            result = engine.step(target)
            kinds.add(result.action.kind)
            rewards.append(result.receipt.reward)
            self.assertEqual(result.action.target, target)
            for index in range(4):
                if index != target or result.action.kind == SPA.ActionKind.KEEP:
                    self.assertEqual(self.host.sentences[index], before[index])
            if result.action.kind != SPA.ActionKind.KEEP:
                context = next((t for t in before[result.action.source] if t != 0), 0)
                self.assertEqual(self.host.sentences[target][0], context)
            self.assertEqual(bytes(result.receipt.consequence.after), bytes(result.after))
        self.assertEqual(kinds, {0, 1, 2})
        self.assertTrue(any(r > 0 for r in rewards))
        self.assertTrue(any(r < 0 for r in rewards))
        self.assertNotEqual(bytes(engine.agent.state.policy), policy)
        self.assertEqual(engine.agent.state.observations, 40)

    def test_complete_session_resume_is_exact(self):
        engine = self.engine()
        for _ in range(11):
            engine.step(self.host.cursor % 4)
        other = TextHost(self.native, self.library)
        self.addCleanup(other.close)
        resumed = self.engine(other)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.spa"
            engine.save(path)
            resumed.load(path)
            self.assertEqual(self.host.snapshot(), other.snapshot())
            self.assertEqual(engine.agent.hash, resumed.agent.hash)
            for _ in range(25):
                a = engine.step(self.host.cursor % 4)
                b = resumed.step(other.cursor % 4)
                self.assertEqual(bytes(a.decision), bytes(b.decision))
                self.assertEqual(bytes(a.receipt), bytes(b.receipt))
                self.assertEqual(self.host.snapshot(), other.snapshot())
                self.assertEqual(engine.agent.hash, resumed.agent.hash)

    def test_disabled_keeps_body_byte_exact(self):
        config = SPA.Config.default(native=self.native, mode=SPA.Mode.DISABLED)
        engine = SPA.Engine(self.host, native=self.native, config=config)
        before = engine.agent.hash
        for target in range(4):
            self.assertIsNone(engine.step(target).receipt)
        self.assertEqual(self.host.snapshot(), self.snapshot)
        self.assertEqual(engine.agent.hash, before)

    def test_cli_resume_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "session.spa"
            command = [sys.executable, str(ROOT / "examples/spa_text.py")]
            import os
            env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
            first = subprocess.run(command + ["--steps", "4", "--save", str(checkpoint)],
                                   env=env, capture_output=True, text=True, check=True)
            self.assertIn("cross entropy", first.stdout)
            second = subprocess.run(command + ["--resume", str(checkpoint), "--steps", "5"],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertIn("Resumed at step 4", second.stdout)
            self.assertNotIn("cross entropy", second.stdout)
            loaded = self.engine()
            loaded.load(checkpoint)
            self.assertEqual(self.host.cursor, 9)


if __name__ == "__main__":
    unittest.main()
