"""Native SPA binding gates. Run after `make shared`, using Python stdlib only."""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import SPA

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = {"schema": "notorch.spa.python.v1", "commands": []}
FIELD = [[1, .2, 0, -.1], [-.4, .8, .3, 0], [.6, .1, -.5, .9]]


def outcome(kind):
    before = SPA.Metrics(.5, .5, .5, .5, .5, .5, .5)
    after = before.copy()
    after.local_connectedness = [.3, .8, .2][kind]
    after.novelty = [.5, .6, .4][kind]
    return SPA.Consequence(before, after, .25 if kind else 0)


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="notorch-spa-python-")
        cls.work = Path(cls.tmp.name)
        cls.native = SPA.Native()
        cls.cc = shlex.split(os.environ.get("CC", "cc"))
        cls.layout = cls.work / "layout"
        cls.oracle = cls.work / "oracle"
        cls.run_command(cls.cc + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                                 str(ROOT / "tests/spa_layout.c"), "-o", str(cls.layout)])
        library = Path(cls.native.path).resolve()
        cls.run_command(cls.cc + ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                 "-I", str(ROOT), str(ROOT / "tests/spa_python_parity.c"),
                                 str(library), "-Wl,-rpath," + str(library.parent),
                                 "-pthread", "-lm", "-o", str(cls.oracle)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_command(cls, command):
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        def normalize(value):
            value = value.replace(str(ROOT), "<repo>").replace(str(cls.work), "<tmp>")
            library = str(Path(cls.native.path).resolve())
            return value.replace(library, "<library>").replace(
                str(Path(library).parent), "<library-dir>")
        RECEIPT["commands"].append({"argv": [normalize(str(x)) for x in command],
                                    "returncode": run.returncode,
                                    "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(),
                                    "stderr": normalize(run.stderr)})
        if run.returncode:
            raise RuntimeError(f"command failed: {normalize(str(command))}\n{run.stderr}")
        return run.stdout

    def learned(self, **changes):
        return SPA.Agent(SPA.Config.default(native=self.native, mode=SPA.Mode.LEARNED,
                                           **changes), native=self.native)

    def test_abi_every_struct_and_field(self):
        compiler = dict(line.rsplit(" ", 1) for line in self.run_command([str(self.layout)]).splitlines())
        expected = SPA._layout()
        self.assertEqual(set(compiler), set(expected))
        for key, value in expected.items():
            with self.subTest(coordinate=key):
                self.assertEqual(int(compiler[key]), value)
                self.assertEqual(self.native.lib.nt_spa_binding_layout(key.encode()), value)
        self.assertEqual(self.native.lib.nt_spa_binding_layout(b"missing"), C.c_size_t(-1).value)
        self.assertEqual(self.native.lib.nt_spa_binding_layout(None), C.c_size_t(-1).value)
        RECEIPT["abi_coordinates"] = len(expected)
        RECEIPT["value_types"] = len(SPA._STRUCTS)

    def test_abi_mutation_and_old_library_refused_before_state_call(self):
        source = (ROOT / "spa_binding.c").read_text()
        self.assertIn("offsetof(t, f)", source)
        mutated = self.work / "bad_abi.c"
        mutated.write_text(source.replace("offsetof(t, f)", "(offsetof(t, f) + 1)"))
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        library = self.work / ("bad_abi" + ext)
        self.run_command(self.cc + ["-std=c11", "-fPIC", "-shared", "-I", str(ROOT),
                                   str(mutated), "-o", str(library)])
        with self.assertRaisesRegex(SPA.ABIError, "ABI mismatch.*offset"):
            SPA.Native(library)
        empty = self.work / "old.c"
        empty.write_text("int old_library(void) { return 1; }\n")
        old = self.work / ("old" + ext)
        self.run_command(self.cc + ["-fPIC", "-shared", str(empty), "-o", str(old)])
        with self.assertRaisesRegex(SPA.ABIError, "lacks the SPA ABI manifest"):
            SPA.Native(old)
        RECEIPT["compiled_abi_mutation"] = "caught before any agent pointer call"

    def test_native_oracle_sensory_actions_learning_checkpoint_continuation(self):
        expected = self.run_command([str(self.oracle), str(self.work)]).splitlines()
        actual = []
        emit = lambda name, value: actual.append(name + " " + bytes(value).hex())
        embedding = self.native.embed_sentence([2, 0, 1], FIELD)
        emit("embedding", embedding)
        connectedness = self.native.connectedness(embedding, FIELD)
        emit("connectedness", C.c_float(connectedness))
        emit("logits", self.native.modulate_logits([1, -2, .5], connectedness))
        agent = self.learned(seed=123, exploration=.35)
        observation = self.native.perceive(FIELD, 1, phase_lock=.2, temperature=.8)
        emit("observation", observation)
        for _ in range(8):
            emit("imitation_loss", C.c_float(agent.imitate(observation, SPA.ActionKind.RESEED_LEFT)))
        experience = agent.capture(observation)
        comparison = SPA.Comparison.from_outcomes(experience, {k: outcome(k) for k in SPA.ActionKind}, horizon=4)
        for _ in range(16):
            emit("fit", agent.fit_comparison(experience, comparison, .025))
        emit("readout", agent.score(experience))
        for i in range(10):
            observation = self.native.perceive(FIELD, i % 3, phase_lock=.2, temperature=.8, reseeds=i)
            decision = agent.choose(observation)
            emit("decision", decision)
            if i == 4:
                agent.save(self.work / "python-pending.life")
            emit("receipt", agent.observe(decision.sequence, decision.action, outcome(decision.action.kind)))
            actual.append("hash " + str(agent.hash))
        agent.save(self.work / "python-final.life")
        self.assertEqual(actual, expected)
        for phase in ("pending", "final"):
            self.assertEqual((self.work / f"native-{phase}.life").read_bytes(),
                             (self.work / f"python-{phase}.life").read_bytes())
        resumed = SPA.Agent.from_file(self.work / "native-pending.life", native=self.native)
        decision = resumed.state.pending_decision
        resumed.observe(decision.sequence, decision.action, outcome(decision.action.kind))
        for i in range(5, 10):
            observation = self.native.perceive(FIELD, i % 3, phase_lock=.2, temperature=.8, reseeds=i)
            decision = resumed.choose(observation)
            resumed.observe(decision.sequence, decision.action, outcome(decision.action.kind))
        resumed.save(self.work / "resumed-final.life")
        final = (self.work / "native-final.life").read_bytes()
        self.assertEqual(final, (self.work / "resumed-final.life").read_bytes())
        RECEIPT["native_python_parity"] = {"records": len(actual), "decisions": 10,
            "imitation_updates": 8, "comparison_fits": 16,
            "pending_checkpoint_bytes": (self.work / "native-pending.life").stat().st_size,
            "final_checkpoint_sha256": hashlib.sha256(final).hexdigest(),
            "resumed_continuation": "byte-identical"}

    def test_shapes_and_integer_bounds_refused(self):
        bad = [lambda: self.native.perceive([[1, 2], [1]], 0),
               lambda: self.native.perceive([1, 2, 3], 0, dim=2),
               lambda: self.native.perceive(FIELD, -1),
               lambda: self.native.perceive(FIELD, 3),
               lambda: self.native.perceive(FIELD, 1, reseeds=1 << 32),
               lambda: self.native.perceive(FIELD, 1, temperature=float("nan")),
               lambda: self.native.connectedness([1, 2], [[1]]),
               lambda: self.native.connectedness([1, 2], [1, 2, 3], dim=2),
               lambda: self.native.embed_sentence([3], FIELD),
               lambda: self.native.embed_sentence([-1], FIELD),
               lambda: self.native.embed_sentence([1.5], FIELD),
               lambda: self.native.embed_sentence([1], []),
               lambda: self.native.modulate_logits([float("inf")], .5),
               lambda: SPA.Config.default(native=self.native, seed=(1 << 32) + 1),
               lambda: SPA.Observation(sentence_count=-1),
               lambda: SPA.Action.at(SPA.ActionKind.RESEED_LEFT, 0)]
        for index, call in enumerate(bad):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                call()
        flat = [x for row in FIELD for x in row]
        self.assertEqual(bytes(self.native.perceive(FIELD, 1)), bytes(self.native.perceive(flat, 1, dim=4)))
        self.assertEqual(bytes(self.native.embed_sentence([1, 2], FIELD)),
                         bytes(self.native.embed_sentence([1, 2], flat, dim=4)))
        self.assertEqual(self.native.connectedness([1, 2], []), 0)
        self.assertEqual(list(self.native.embed_sentence([], FIELD)), [0] * 4)
        self.assertEqual(len(self.native.modulate_logits([], .5)), 0)

    def test_native_status_pending_bad_enum_and_exact_action_witness(self):
        with self.assertRaises(SPA.Error) as failure:
            self.learned(seed=0)
        self.assertEqual(failure.exception.status, SPA.Status.CONFIG)
        bad_config = self.native.config()
        bad_config.mode = 99
        with self.assertRaises(SPA.Error) as failure:
            SPA.Agent(bad_config, native=self.native)
        self.assertEqual(failure.exception.status, SPA.Status.CONFIG)
        agent = self.learned()
        observation = self.native.perceive(FIELD, 1)
        preview, before = agent.select(observation), agent.hash
        self.assertEqual(before, agent.hash)
        decision = agent.choose(observation)
        self.assertEqual(bytes(preview), bytes(decision))
        pending = agent.hash
        with self.assertRaises(SPA.Error) as failure:
            agent.choose(observation)
        self.assertEqual(failure.exception.status, SPA.Status.PENDING)
        wrong = decision.action.copy()
        wrong.kind = 99
        with self.assertRaises(SPA.Error) as failure:
            agent.observe(decision.sequence, wrong, outcome(0))
        self.assertEqual(failure.exception.status, SPA.Status.ACTION)
        self.assertEqual(agent.hash, pending)
        agent.cancel(decision.sequence)
        self.assertEqual(agent.state.cancelled, 1)
        self.assertEqual(agent.state.history_count, 0)
        with self.assertRaises(SPA.Error):
            agent.cancel(decision.sequence)
        with self.assertRaises(ValueError):
            agent.imitate(observation, 99)
        with self.assertRaises(SPA.Error) as failure:
            self.native.validate_action(SPA.Action.at(SPA.ActionKind.RESEED_RIGHT, 2),
                                        self.native.perceive(FIELD, 2))
        self.assertEqual(failure.exception.status, SPA.Status.ACTION)

    def test_disabled_legacy_memory_and_detached_state(self):
        observation = self.native.perceive(FIELD, 1)
        observation.sentence_score, observation.mean_sentence_score = .1, 1
        legacy = SPA.Agent(native=self.native)
        self.assertEqual(bytes(legacy.select(observation).action), bytes(self.native.legacy(observation)))
        self.assertEqual(legacy.select(observation).action.kind, SPA.ActionKind.RESEED_LEFT)
        disabled = SPA.Agent(SPA.Config.default(native=self.native, mode=SPA.Mode.DISABLED), native=self.native)
        before = disabled.hash
        self.assertEqual(disabled.choose(observation).action.kind, SPA.ActionKind.KEEP)
        self.assertEqual(disabled.hash, before)
        agent = self.learned()
        decision = agent.choose(observation)
        agent.observe(decision.sequence, decision.action, outcome(decision.action.kind))
        old = agent.state
        agent.reset_memory()
        reset = agent.state
        self.assertEqual(reset.history_count, 0)
        self.assertEqual(reset.memory_observations, 0)
        self.assertEqual(bytes(reset.policy), bytes(old.policy))
        self.assertEqual((reset.rng, reset.decisions, reset.updates), (old.rng, old.decisions, old.updates))
        current = agent.hash
        detached = agent.state
        detached.rng = 0
        detached.policy.b2[0] += 1
        self.assertEqual(agent.hash, current)
        policy = agent.state.policy.copy()
        policy.b2[0] += .1
        agent.set_policy(policy)
        self.assertNotEqual(agent.hash, current)

    def test_comparison_weight_only_learning_and_rejections(self):
        agent = self.learned(exploration=0)
        observation = self.native.perceive(FIELD, 1)
        experience = agent.capture(observation)
        before = agent.state
        comparison = SPA.Comparison.from_outcomes(experience, {k: outcome(k) for k in SPA.ActionKind}, horizon=4)
        zero = agent.fit_comparison(experience, comparison, 0)
        self.assertEqual(bytes(agent.state), bytes(before))
        self.assertEqual(zero.loss_before, zero.loss_after)
        start = agent.score(experience).action.kind
        for _ in range(100):
            receipt = agent.fit_comparison(experience, comparison, .1)
        after = agent.state
        self.assertEqual(agent.score(experience).action.kind, SPA.ActionKind.RESEED_LEFT)
        self.assertNotEqual(start, SPA.ActionKind.RESEED_LEFT)
        self.assertLess(receipt.loss_after, zero.loss_before)
        after.policy = before.policy
        self.assertEqual(bytes(after), bytes(before))
        current = agent.hash
        comparison.source_life_hash ^= 1
        with self.assertRaises(SPA.Error) as failure:
            agent.fit_comparison(experience, comparison, .1)
        self.assertEqual(failure.exception.status, SPA.Status.COMPARISON)
        self.assertEqual(agent.hash, current)
        RECEIPT["comparison_counterfactual"] = "fixed input: KEEP -> RESEED_LEFT; only policy bytes change"

    def test_checkpoint_refusal_is_transactional(self):
        agent = self.learned()
        before = agent.hash
        bad = self.work / "bad.life"
        bad.write_bytes(b"truncated")
        with self.assertRaises(SPA.Error) as failure:
            agent.load(bad)
        self.assertEqual(failure.exception.status, SPA.Status.FORMAT)
        self.assertEqual(agent.hash, before)
        with self.assertRaises(SPA.Error) as failure:
            agent.save(self.work / "missing" / "life")
        self.assertEqual(failure.exception.status, SPA.Status.IO)
        for path in ("", "bad\0path"):
            with self.assertRaises(ValueError):
                agent.save(path)

    def test_repeated_single_exact_parity_and_refusals(self):
        single, repeated = self.learned(exploration=0), self.learned(exploration=0)
        experience = single.capture(self.native.perceive(FIELD, 1))
        comparison = SPA.Comparison.from_outcomes(experience, {k: outcome(k) for k in SPA.ActionKind}, horizon=4)
        for _ in range(16):
            expected = single.fit_comparison(experience, comparison, .03)
            actual = repeated.fit_repeated(experience, [comparison], .03)
            self.assertEqual(bytes(expected), bytes(actual))
            self.assertEqual(bytes(single.state), bytes(repeated.state))
        single.save(self.work / "single-repeat.life")
        repeated.save(self.work / "repeated-one.life")
        self.assertEqual((self.work / "single-repeat.life").read_bytes(),
                         (self.work / "repeated-one.life").read_bytes())
        before = repeated.hash
        for comparisons in ([], [comparison] * 65, [None]):
            with self.assertRaises((ValueError, TypeError)):
                repeated.fit_repeated(experience, comparisons, .03)
            self.assertEqual(repeated.hash, before)
        bad = comparison.copy()
        bad.source_life_hash ^= 1
        with self.assertRaises(SPA.Error) as failure:
            repeated.fit_repeated(experience, [comparison, bad], .03)
        self.assertEqual(failure.exception.status, SPA.Status.COMPARISON)
        self.assertEqual(repeated.hash, before)
        RECEIPT["repeated_count_one"] = "16 receipts, states and checkpoint bytes equal fit_comparison"

    def test_repeated_averages_clipped_native_rewards(self):
        agent = self.learned(exploration=0, reward_weights=SPA.Metrics(1, 1, 1, 1, 0, 0, 0), cost_weight=0)
        experience = agent.capture(self.native.perceive(FIELD, 1))
        before = SPA.Metrics(.5, .5, .5, .5, .5, .5, .5)
        def comparison(local):
            outcomes = {}
            for kind in SPA.ActionKind:
                after = before.copy()
                if kind == SPA.ActionKind.RESEED_LEFT:
                    after.local_connectedness = after.global_connectedness = local
                    after.coherence = after.novelty = local
                outcomes[kind] = SPA.Consequence(before, after, 0)
            return SPA.Comparison.from_outcomes(experience, outcomes, horizon=4)
        initial = agent.hash
        receipt = agent.fit_repeated(experience, [comparison(1), comparison(.4)], 0)
        self.assertAlmostEqual(receipt.rewards[SPA.ActionKind.RESEED_LEFT], .3, places=6)
        self.assertAlmostEqual(receipt.targets[SPA.ActionKind.RESEED_LEFT], .3, places=6)
        self.assertEqual(receipt.loss_before, receipt.loss_after)
        self.assertEqual(agent.hash, initial)
        mean_raw = agent.fit_comparison(experience, comparison(.7), 0)
        self.assertGreater(abs(receipt.rewards[1] - mean_raw.rewards[1]), .49)
        before_state = agent.state
        fitted = agent.fit_repeated(experience, [comparison(1), comparison(.4)] * 4, .03)
        after_state = agent.state
        self.assertLess(fitted.loss_after, fitted.loss_before)
        after_state.policy = before_state.policy
        self.assertEqual(bytes(after_state), bytes(before_state))
        RECEIPT["repeated_reward_order"] = "clip each repeat in C, then average; mean=.3, mean-raw route=.8"

    def test_conditioned_raw_rewards_floor_rate_zero_and_native_parity(self):
        agent = self.learned(exploration=0)
        captured = agent.capture(self.native.perceive(FIELD, 1))
        comparison = SPA.Comparison.from_outcomes(
            captured, {k: outcome(k) for k in SPA.ActionKind}, horizon=4)
        comparisons = [comparison] * 8
        initial = agent.state
        raw = agent.fit_repeated(captured, comparisons, 0)
        for floor in (.001, 1.0):
            receipt = agent.fit_conditioned(captured, comparisons, 0, floor)
            expected_floor = C.c_float(floor).value
            expected_scale = max(expected_floor, *(abs(v) for v in raw.targets))
            self.assertEqual(receipt.scale_floor, expected_floor)
            self.assertEqual(receipt.scale, expected_scale)
            self.assertEqual(bytes(receipt.comparison.rewards), bytes(raw.rewards))
            expected = (C.c_float * SPA.ACTIONS)(*(v / expected_scale for v in raw.targets))
            self.assertEqual(bytes(receipt.comparison.targets), bytes(expected))
            self.assertEqual(receipt.comparison.loss_before, receipt.comparison.loss_after)
            self.assertEqual(bytes(agent.state), bytes(initial))
        native_state = initial.copy()
        array = (SPA.Comparison * len(comparisons))(*comparisons)
        for _ in range(16):
            receipt = agent.fit_conditioned(captured, comparisons, .03, .001)
            direct = SPA.ConditionedReceipt()
            status = self.native.lib.nt_spa_agent_fit_conditioned(
                C.byref(native_state), C.byref(captured), array, len(array),
                .03, .001, C.byref(direct))
            self.assertEqual(status, SPA.Status.OK)
            self.assertEqual(bytes(receipt), bytes(direct))
            self.assertEqual(bytes(agent.state), bytes(native_state))
        current = agent.state
        self.assertNotEqual(bytes(current.policy), bytes(initial.policy))
        current.policy = initial.policy
        self.assertEqual(bytes(current), bytes(initial))
        path = self.work / "conditioned.life"
        agent.save(path)
        resumed = SPA.Agent.from_file(path, native=self.native)
        self.assertEqual(len(path.read_bytes()), 2296)
        self.assertEqual(bytes(agent.state), bytes(resumed.state))
        receipt = agent.fit_conditioned(captured, comparisons, .03, .001)
        restored = resumed.fit_conditioned(captured, comparisons, .03, .001)
        self.assertEqual(bytes(receipt), bytes(restored))
        self.assertEqual(bytes(agent.state), bytes(resumed.state))
        RECEIPT["conditioned_native_parity"] = {
            "updates": 16, "raw_rewards": "byte-identical to repeated receipt",
            "scale": "max(float32 floor, absolute native float delta) in double",
            "floors": [.001, 1.0], "rate_zero": "all life bytes unchanged",
            "checkpoint_bytes": 2296, "resumed_update": "receipt and life byte-identical"}

    def test_conditioned_refusals_leave_life_unchanged(self):
        agent = self.learned(exploration=0)
        captured = agent.capture(self.native.perceive(FIELD, 1))
        comparison = SPA.Comparison.from_outcomes(
            captured, {k: outcome(k) for k in SPA.ActionKind}, horizon=4)
        initial = bytes(agent.state)
        for floor in (0, -1):
            with self.assertRaises(SPA.Error) as failure:
                agent.fit_conditioned(captured, [comparison], .03, floor)
            self.assertEqual(failure.exception.status, SPA.Status.COMPARISON)
            self.assertEqual(bytes(agent.state), initial)
        for floor in (float("nan"), float("inf"), 1e100):
            with self.assertRaises(ValueError):
                agent.fit_conditioned(captured, [comparison], .03, floor)
            self.assertEqual(bytes(agent.state), initial)
        for values in ([], [comparison] * 65, [None]):
            with self.assertRaises((ValueError, TypeError)):
                agent.fit_conditioned(captured, values, .03, .001)
            self.assertEqual(bytes(agent.state), initial)
        bad = comparison.copy()
        bad.source_life_hash ^= 1
        with self.assertRaises(SPA.Error) as failure:
            agent.fit_conditioned(captured, [comparison, bad], .03, .001)
        self.assertEqual(failure.exception.status, SPA.Status.COMPARISON)
        self.assertEqual(bytes(agent.state), initial)
        bad = comparison.copy()
        bad.alternatives[0].consequence.before.novelty += .1
        with self.assertRaises(SPA.Error) as failure:
            agent.fit_conditioned(captured, [comparison, bad], .03, .001)
        self.assertEqual(failure.exception.status, SPA.Status.CONSEQUENCE)
        self.assertEqual(bytes(agent.state), initial)
        RECEIPT["conditioned_refusals"] = "invalid floor, count, type, source and common-before preserve every life byte"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingTests))
    RECEIPT.update(passed=result.wasSuccessful(), tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors))
    RECEIPT["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
        "python/SPA.py", "python/test_spa_binding.py", "tests/spa_layout.c",
        "tests/spa_python_parity.c", "spa_binding.c", "spa_binding.h", "spa_agent.c", "spa_agent.h")}
    if args.json:
        args.json.write_text(json.dumps(RECEIPT, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
