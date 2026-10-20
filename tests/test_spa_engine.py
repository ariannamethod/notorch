"""End-to-end gates for SPA.Engine, using the real native policy and checkpoint.

Run after make shared: python3 tests/test_spa_engine.py
The three-sentence host is an executable deterministic environment fixture;
its complete state includes its own RNG and last-action quality measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import SPA
import spa_engine


class Host:
    host_id = "notorch.engine-test.v1"

    def __init__(self, native):
        self.native = native
        self.rng, self.cursor, self.calls, self.quality = 53, 0, 0, .2
        self.field = [[1., .2, 0., -.1], [-.4, .8, .3, 0.], [.6, .1, -.5, .9]]
        self.failure = None

    def snapshot(self):
        return json.dumps([self.rng, self.cursor, self.calls, self.quality, self.field],
                          separators=(",", ":")).encode()

    def restore(self, snapshot):
        values = json.loads(snapshot)
        if self.failure == "restore":
            raise RuntimeError("fixture restore failed")
        if self.failure == "reject_load" and values[1] != self.cursor:
            # Exercise a host that mutates before refusing, and successful rollback.
            self.quality = .99
            raise ValueError("fixture refuses this cursor")
        self.rng, self.cursor, self.calls, self.quality, self.field = values

    def observe(self, target):
        obs = self.native.perceive(self.field, target, temperature=.8)
        metrics = SPA.Metrics(self.quality, obs.mean_sentence_score, obs.coherence,
                              obs.novelty, obs.repetition, 0, obs.coherence)
        if self.failure == "before":
            metrics.collapse = float("nan")
        if self.failure == "target":
            obs.sentence_index = (target + 1) % len(self.field)
        if self.failure == "after" and self.cursor:
            metrics.collapse = 2
        if self.failure == "geometry" and self.cursor:
            obs.sentence_count += 1
        return obs, metrics

    def execute(self, action):
        self.calls += 1
        self.cursor += 1
        self.rng = (1664525 * self.rng + 1013904223) & 0xFFFFFFFF
        self.quality = (.2, .9, .05)[action.kind]
        if self.failure in ("execute", "restore"):
            raise RuntimeError("fixture execution failed after mutation")
        if self.failure == "cost":
            return float("nan")
        return 0 if action.kind == SPA.ActionKind.KEEP else .1


def trace(engine, targets):
    out = []
    for result in engine.run(targets):
        out.append((bytes(result.decision), bytes(result.receipt), engine.agent.hash,
                    engine.host.snapshot()))
    return out


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = SPA.Native(LIBRARY)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spa-engine-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "engine.spa"

    def engine(self, mode=SPA.Mode.LEARNED, exploration=.6):
        config = SPA.Config.default(native=self.native, mode=mode, seed=137,
                                    exploration=exploration, learning_rate=.1)
        return SPA.Engine(Host(self.native), native=self.native, config=config)

    def test_default_mode_and_complete_save_resume(self):
        default = SPA.Engine(Host(self.native), native=self.native)
        self.assertEqual(default.agent.state.config.mode, SPA.Mode.LEARNED)
        engine = self.engine()
        first = trace(engine, [0, 1, 2, 1] * 3)
        self.assertTrue(any(row[0] != first[0][0] for row in first[1:]))
        engine.save(self.path)
        saved_state, saved_host, saved_steps = (bytes(engine.agent.state),
                                                engine.host.snapshot(), engine.steps)
        expected = trace(engine, [1, 2, 0, 1] * 8)
        resumed = self.engine(mode=SPA.Mode.LEGACY)
        self.assertIs(resumed.load(self.path), resumed)
        self.assertEqual(bytes(resumed.agent.state), saved_state)
        self.assertEqual(resumed.host.snapshot(), saved_host)
        self.assertEqual(resumed.steps, saved_steps)
        self.assertEqual(trace(resumed, [1, 2, 0, 1] * 8), expected)
        self.assertEqual(resumed.steps, engine.steps)
        resumed.save(Path(self.tmp.name) / "resumed.spa")
        engine.save(Path(self.tmp.name) / "continuous.spa")
        self.assertEqual((Path(self.tmp.name) / "resumed.spa").read_bytes(),
                         (Path(self.tmp.name) / "continuous.spa").read_bytes())

    def test_live_credit_and_acquired_policy_cause_action_change(self):
        engine = self.engine(exploration=1)
        original_policy = bytes(engine.agent.state.policy)
        for _ in range(160):
            # A new environment trial preserves the acquired agent's own history.
            engine.host.quality = .2
            result = engine.step(1)
            self.assertEqual(bytes(result.receipt.consequence.before), bytes(result.before))
            self.assertEqual(bytes(result.receipt.consequence.after), bytes(result.after))
            self.assertEqual(bytes(result.receipt.action), bytes(result.decision.action))
        self.assertNotEqual(bytes(engine.agent.state.policy), original_policy)
        self.assertEqual(engine.agent.state.observations, 160)
        self.assertEqual(engine.agent.state.updates, 160)
        frozen = self.engine(exploration=0)
        observation = frozen.host.observe(1)[0]
        initial = frozen.agent.select(observation)
        fixed = frozen.agent.state
        frozen.agent.set_policy(engine.agent.state.policy)
        acquired = frozen.agent.select(observation)
        after = frozen.agent.state
        after.policy = fixed.policy
        self.assertEqual(bytes(after), bytes(fixed))
        self.assertEqual(initial.action.kind, SPA.ActionKind.KEEP)
        self.assertEqual(acquired.action.kind, SPA.ActionKind.RESEED_LEFT)

    def test_disabled_is_pure_and_legacy_leaves_policy_fixed(self):
        engine = self.engine(mode=SPA.Mode.DISABLED)
        state, host = bytes(engine.agent.state), engine.host.snapshot()
        with patch.object(engine.host, "snapshot", side_effect=AssertionError("disabled snapshot")):
            result = engine.step(1)
        self.assertEqual(result.action.kind, SPA.ActionKind.KEEP)
        self.assertIsNone(result.receipt)
        self.assertEqual(engine.steps, 0)
        self.assertEqual(bytes(engine.agent.state), state)
        self.assertEqual(engine.host.snapshot(), host)
        engine = self.engine(mode=SPA.Mode.LEGACY)
        policy = bytes(engine.agent.state.policy)
        trace(engine, [0, 1, 2] * 3)
        self.assertEqual(bytes(engine.agent.state.policy), policy)
        self.assertEqual(engine.agent.state.updates, 0)

    def test_invalid_host_and_execution_failures_restore_both_lives(self):
        for failure in ("before", "target", "after", "geometry", "execute", "cost"):
            with self.subTest(failure=failure):
                engine = self.engine()
                state, host = bytes(engine.agent.state), engine.host.snapshot()
                engine.host.failure = failure
                with self.assertRaises((ValueError, RuntimeError)):
                    engine.step(1)
                self.assertEqual(bytes(engine.agent.state), state)
                self.assertEqual(engine.host.snapshot(), host)
                self.assertEqual(engine.steps, 0)
                engine.host.failure = None
                self.assertEqual(trace(engine, [1, 0, 2]), trace(self.engine(), [1, 0, 2]))

    def test_invalid_action_rejected_before_execution(self):
        engine = self.engine()
        state, host = bytes(engine.agent.state), engine.host.snapshot()
        choose = engine.agent.choose
        def defective_choose(observation):
            decision = choose(observation)
            decision.action.source = 99
            return decision
        with patch.object(engine.agent, "choose", defective_choose):
            with self.assertRaises(SPA.Error):
                engine.step(1)
        self.assertEqual(bytes(engine.agent.state), state)
        self.assertEqual(engine.host.snapshot(), host)

    def test_pending_life_refused_and_reentrant_host_refused(self):
        engine = self.engine()
        decision = engine.agent.choose(engine.host.observe(1)[0])
        with self.assertRaises(SPA.Error) as error:
            engine.step(1)
        self.assertEqual(error.exception.status, SPA.Status.PENDING)
        with self.assertRaises(SPA.Error):
            engine.save(self.path)
        engine.agent.cancel(decision.sequence)
        state, host = bytes(engine.agent.state), engine.host.snapshot()
        with patch.object(engine.host, "execute", lambda action: engine.step(1)):
            with self.assertRaisesRegex(RuntimeError, "callbacks"):
                engine.step(1)
        self.assertEqual(bytes(engine.agent.state), state)
        self.assertEqual(engine.host.snapshot(), host)

    def test_rollback_failure_blocks_use_until_complete_reload(self):
        engine = self.engine()
        engine.save(self.path)
        original = bytes(engine.agent.state)
        engine.host.failure = "restore"
        with self.assertRaises(SPA.RollbackError):
            engine.step(1)
        self.assertEqual(bytes(engine.agent.state), original)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            engine.step(1)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            engine.save(self.path)
        engine.host.failure = None
        with patch.object(engine.host, "snapshot", side_effect=RuntimeError("broken host snapshot")):
            engine.load(self.path)
        self.assertEqual(trace(engine, [1, 2, 0]), trace(self.engine(), [1, 2, 0]))

    def test_corruption_wrong_identity_and_native_life_refused_transactionally(self):
        engine = self.engine()
        engine.save(self.path)
        original = self.path.read_bytes()
        state, host = bytes(engine.agent.state), engine.host.snapshot()
        mutations = [original[:-1], original + b"x", b"badmagic" + original[8:]]
        corrupt = bytearray(original)
        corrupt[-33] ^= 1
        mutations.append(bytes(corrupt))
        header = spa_engine._HEADER
        magic, version, meta_size, life_size, host_size = header.unpack(original[:header.size])
        # Valid envelope checksum around an invalid native canonical life.
        corrupt = bytearray(original)
        corrupt[header.size + meta_size] ^= 1
        corrupt[-32:] = hashlib.sha256(corrupt[:-32]).digest()
        mutations.append(bytes(corrupt))
        # A maliciously large host length is refused before payload allocation.
        mutations.append(header.pack(magic, version, meta_size, life_size, 1 << 39)
                         + original[header.size:])
        for mutation in mutations:
            with self.subTest(length=len(mutation)):
                self.path.write_bytes(mutation)
                with self.assertRaises((SPA.CheckpointError, SPA.Error)):
                    engine.load(self.path)
                self.assertEqual(bytes(engine.agent.state), state)
                self.assertEqual(engine.host.snapshot(), host)
        self.path.write_bytes(original)
        stranger = Host(self.native)
        stranger.host_id = "different-host.v1"
        other = SPA.Engine(stranger, native=self.native)
        with self.assertRaisesRegex(SPA.CheckpointError, "identity"):
            other.load(self.path)

    def test_host_restore_rejection_rolls_back_partial_mutation(self):
        engine = self.engine()
        engine.save(self.path)
        trace(engine, [0, 1])
        state, host, steps = bytes(engine.agent.state), engine.host.snapshot(), engine.steps
        engine.host.failure = "reject_load"
        with self.assertRaisesRegex(ValueError, "cursor"):
            engine.load(self.path)
        self.assertEqual(bytes(engine.agent.state), state)
        self.assertEqual(engine.host.snapshot(), host)
        self.assertEqual(engine.steps, steps)

    def test_atomic_save_failure_preserves_previous_file(self):
        engine = self.engine()
        engine.save(self.path)
        original = self.path.read_bytes()
        engine.step(1)
        with patch.object(spa_engine.os, "replace", side_effect=OSError("injected replace failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                engine.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(Path(self.tmp.name).glob(".spa-engine-*")), [])

    def test_host_receives_detached_action_and_result_has_no_live_aliases(self):
        engine = self.engine(exploration=0)
        execute = engine.host.execute
        def modifying_callback(action):
            cost = execute(action)
            action.kind = SPA.ActionKind.RESEED_RIGHT
            return cost
        with patch.object(engine.host, "execute", modifying_callback):
            result = engine.step(1)
        self.assertEqual(result.receipt.action.kind, SPA.ActionKind.KEEP)
        old = bytes(engine.agent.state)
        result.decision.action.kind = SPA.ActionKind.RESEED_LEFT
        result.receipt.reward = .99
        self.assertEqual(bytes(engine.agent.state), old)

    def test_gate_catches_missing_credit_and_restored_rng_mutations(self):
        # Red hands exercise causal checks, not only parser refusals.
        engine = self.engine(exploration=0)
        original_observe = engine.agent.observe
        def no_learning(*args):
            state = engine.agent.state
            result = original_observe(*args)
            engine.agent._state.policy = state.policy
            return result
        # Force a positive LEFT reward, then restoring its old weights must fail
        # the same acquired-state assertion used for operational learning.
        policy = engine.agent.state.policy
        policy.b2[SPA.ActionKind.RESEED_LEFT] = .3
        engine.agent.set_policy(policy)
        old = bytes(engine.agent.state.policy)
        with patch.object(engine.agent, "observe", no_learning):
            engine.step(1)
        with self.assertRaises(AssertionError):
            self.assertNotEqual(bytes(engine.agent.state.policy), old)
        engine = self.engine()
        trace(engine, [0, 1, 2])
        engine.save(self.path)
        expected = trace(engine, [1, 2, 0, 1, 2, 0])
        engine.load(self.path)
        engine.host.rng ^= 1  # Deliberately omit exact host RNG restoration.
        with self.assertRaises(AssertionError):
            self.assertEqual(trace(engine, [1, 2, 0, 1, 2, 0]), expected)


LIBRARY = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    args, rest = parser.parse_known_args()
    LIBRARY = args.library
    unittest.main(argv=[sys.argv[0]] + rest)
