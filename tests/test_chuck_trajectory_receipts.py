#!/usr/bin/env python3
"""Narrow gates for trajectory credit, action joins and retained identities."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chuck_trajectory_receipts", ROOT / "experiments/chuck_loss_architect/trajectories/run.py")
trajectory = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(trajectory)


class TrajectoryReceiptTests(unittest.TestCase):
    def test_feedback_meets_its_actual_consequence(self):
        reward = trajectory.f32((2.0 - 1.5) / (2.0 + 1e-6))
        policy = {"scores": [0.0, 0.5, -1.0], "action": {"kind": 2},
                  "consequence": {"before_loss": 2.0, "after_loss": 1.5, "reward": reward,
                                  "learned": 0, "nonfinite": 0, "error": 0, "predicted": 0.5}}
        trajectory.validate_feedback(policy)
        wrong = copy.deepcopy(policy)
        wrong["consequence"]["reward"] *= -1
        with self.assertRaisesRegex(AssertionError, "actual consequence"):
            trajectory.validate_feedback(wrong)

    def test_selected_action_binds_its_outcome(self):
        samples = [{"body": "simple", "seed": 42, "checkpoint": 17,
                    "state_hash": "world", "policy_hash": "life", "future_loss": [3.0, 2.0, 4.0]}]
        row = {"body": "simple", "seed": 42, "checkpoint": 17, "label": "student",
               "state_hash": "world", "source_policy_hash": "life", "action": 2,
               "future_loss": 2.0, "hold_loss": 3.0, "regret": 0.0, "advantage": 1.0,
               "optimal": True, "forced": False, "scores": [0.0, 1.0, -1.0]}
        trajectory.validate_readouts([row], samples, "student")
        wrong = dict(row, future_loss=4.0)
        with self.assertRaisesRegex(AssertionError, "wrong action outcome"):
            trajectory.validate_readouts([wrong], samples, "student")
        with self.assertRaises(AssertionError):
            trajectory.validate_readouts([dict(row, action=3)], samples, "student")

    def test_fitting_uses_shared_student_continuation(self):
        fork = {"type": "fork", "step": 17, "state_hash": "world", "policy_hash": "source-life",
                "observation": {}, "features": [0.0] * 16}
        outcomes = [{"type": "comparison", "checkpoint": 17, "continuation": continuation,
                     "action": action, "horizon": 16, "future_after": loss}
                    for continuation, losses in (("policy", [8.0, 7.0, 9.0]), ("student", [2.0, 1.0, 3.0]))
                    for action, loss in zip(trajectory.ACTIONS, losses)]
        rows = [{"type": "rollout_run", "continuation_aliases": {}}, fork, *outcomes]
        prefix, output = Path("/fixture/source"), Path("/fixture")
        samples = trajectory.gather_samples(rows, "simple", 42, "parent", prefix, output)
        self.assertEqual(samples[0]["future_loss"], [2.0, 1.0, 3.0])
        self.assertEqual(samples[0]["executed_continuation"], "student")
        aliased = [{"type": "rollout_run", "continuation_aliases": {"student": "policy"}}, fork, *outcomes[:3]]
        samples = trajectory.gather_samples(aliased, "simple", 42, "student", prefix, output)
        self.assertEqual(samples[0]["future_loss"], [8.0, 7.0, 9.0])
        self.assertEqual(samples[0]["continuation"], "student")
        self.assertEqual(samples[0]["executed_continuation"], "policy")

    def test_source_prefix_compares_matching_steps(self):
        first = {"type": "rollout_step", "step": 1, "state_hash": "one", "action": 2}
        second = {"type": "rollout_step", "step": 2, "state_hash": "two", "action": 3}
        source = [{"type": "evaluation", "step": 0, "heldout_loss": 4.0}, first, second,
                  {"type": "evaluation", "step": 2, "heldout_loss": 3.0}]
        deployment = [source[0], first, second, {"type": "rollout_step", "step": 3, "state_hash": "three"}]
        receipt = trajectory.compare_prefix(source, deployment, 2)
        self.assertEqual(receipt["transitions"], 2)
        self.assertEqual(receipt["final_matching_state_hash"], "two")
        wrong = copy.deepcopy(deployment)
        wrong[2]["state_hash"] = "changed"
        with self.assertRaisesRegex(AssertionError, "deployment prefix"):
            trajectory.compare_prefix(source, wrong, 2)

    def test_alias_requires_identical_saved_lives(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "source"
            left = Path(str(prefix) + ".fork-1.policy.bin")
            right = Path(str(prefix) + ".fork-1.student.policy.bin")
            left.write_bytes(b"same history and weights")
            right.write_bytes(left.read_bytes())
            rows = [{"type": "fork", "step": 1, "continuation_aliases": {"student": "policy"}}]
            self.assertEqual(trajectory.validate_alias_files(rows, prefix), 1)
            right.write_bytes(b"different continuation weights")
            with self.assertRaisesRegex(AssertionError, "aliased continuation lives differ"):
                trajectory.validate_alias_files(rows, prefix)

    def test_terminal_verification_uses_original_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            artifact = output / "source.policy.bin"
            artifact.write_bytes(b"sealed identity")
            anchors = {"files": {artifact.name: trajectory.identity(artifact)}, "protocol": {}, "host_steps": 17,
                       "cohorts": [], "fits": {}, "readouts": {}, "rollouts": [],
                       "source_deployment": [], "continuation": []}
            self.assertEqual(trajectory.verify_terminal_artifacts(output, anchors)["artifact_identities"], 1)
            artifact.write_bytes(b"changed identity")
            with self.assertRaisesRegex(AssertionError, "persisted artifact changed"):
                trajectory.verify_terminal_artifacts(output, anchors)

    def test_durability_restores_first_parsed_bytes_and_keeps_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "trace.jsonl"
            original = b'{"step":1}\n{"step":2}\n'
            path.write_bytes(original)
            path.chmod(0o640)
            durable = trajectory.DurableArtifacts(output)
            damaged = original[:7]
            def parse_and_truncate(text):
                parsed = [json.loads(line) for line in text.splitlines()]
                path.write_bytes(damaged)
                return parsed
            self.assertEqual(durable.parsed(path, parse_and_truncate), [{"step": 1}, {"step": 2}])
            anchor = dict(durable.files[path.name])
            self.assertEqual(durable.events(path), [{"step": 1}, {"step": 2}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(durable.files[path.name], anchor)
            fault = durable.failures[0]
            self.assertEqual(fault["status"], "restored")
            self.assertEqual((output / fault["evidence"] / "damaged.bin").read_bytes(), damaged)
            index = json.loads(durable.index.read_text())
            self.assertEqual(index["files"][path.name], anchor)
            self.assertEqual(index["attempts"][path.name], 1)
            path.write_bytes(b"second corruption")
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                durable.read_bytes(path)
            self.assertEqual(len(durable.failures), 2)
            self.assertEqual(durable.failures[-1]["status"], "refused")

    def test_durability_refuses_corrupt_or_missing_mirror(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                path = output / "life.bin"
                path.write_bytes(b"original acquired life")
                durable = trajectory.DurableArtifacts(output)
                anchor = durable.identity(path)
                mirror = output / durable.mirrors[path.name]["path"]
                if missing: mirror.unlink()
                else: mirror.write_bytes(b"corrupt gzip")
                path.write_bytes(b"damaged")
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    durable.read_bytes(path)
                self.assertEqual(path.read_bytes(), b"damaged")
                self.assertEqual(durable.files[path.name], anchor)
                self.assertEqual(durable.failures[-1]["status"], "refused")

    def test_durability_never_reanchors_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "body.bin"
            original = b"original body"
            path.write_bytes(original)
            durable = trajectory.DurableArtifacts(output)
            anchor = durable.identity(path)
            path.write_bytes(b"changed body")
            with self.assertRaisesRegex(RuntimeError, "refusing to reanchor"):
                durable.anchor(path, path.read_bytes())
            self.assertEqual(durable.files[path.name], anchor)
            durable.prefix(output / "body")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(durable.files[path.name], anchor)


if __name__ == "__main__":
    unittest.main()
