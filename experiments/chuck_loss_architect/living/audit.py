#!/usr/bin/env python3
"""Read-only interpretation of a completed, authenticated living-credit run.

This program never executes a model, fits weights, restores files, or imports
the experiment's receipt validator. It rejoins saved consequences, policy
payloads and actual host actions before deriving the reported comparisons.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct


def identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def strict_json(data):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(data, parse_constant=invalid)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def life(data):
    assert len(data) == 1016 and data[:8] == b"NTCALIFE", "unexpected policy file format"
    version, size, checksum = struct.unpack_from("<IIQ", data, 8)
    payload = data[24:]
    assert version == 1 and size == 992 and fnv(payload) == checksum
    assert struct.unpack_from("<I", payload)[0] == 1
    # v1 state codec: configuration/counters 140 bytes, 163 F32 weights,
    # then temporal history and the completed decision cache (200 bytes).
    weights = payload[140:792]
    assert len(weights) == 163 * 4 and all(math.isfinite(value) for value in struct.unpack("<163f", weights))
    decisions, updates, pending, history = struct.unpack_from("<QQII", payload, 116)
    return {"hash": f"{checksum:016x}", "weights": weights,
            "nonweights": payload[:140] + payload[792:], "decisions": decisions,
            "updates": updates, "pending": pending, "has_history": history,
            "previous": struct.unpack_from("<3f", payload, 792),
            "learning_rate": struct.unpack_from("<f", payload, 100)[0]}


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.index_bytes = (directory / ".sealed/anchors.json").read_bytes()
        self.index = strict_json(self.index_bytes)
        self.verified = {}
        self.listed = None

    def read(self, name):
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts, "artifact escapes run"
        assert name in self.index["files"], "unanchored artifact: " + name
        data = (self.directory / path).read_bytes()
        actual = identity(data)
        assert actual == self.index["files"][name], "artifact differs from original anchor: " + name
        if self.listed is not None:
            assert actual == self.listed[name], "result lists a different identity: " + name
        self.verified[name] = actual
        return data

    def json(self, name):
        return strict_json(self.read(name))

    def events(self, name):
        return [strict_json(line) for line in self.read(name).splitlines()]


def rows_of(rows, kind):
    return [row for row in rows if row["type"] == kind]


def choice(scores):
    assert len(scores) == 3 and all(math.isfinite(value) for value in scores)
    return 1 + max(range(3), key=lambda k: scores[k])


def paired_divergence(left, right):
    path_fields = ("step", "offset", "window_rng", "before", "after", "gradient_norm",
                   "action", "pre_chuck", "post_chuck")
    pairs = list(zip(left, right))
    different = next((i for i, (a, b) in enumerate(pairs) if a["action"] != b["action"]), None)
    if different is None:
        return {"first_action_divergence": None,
                "complete_training_path_equal": all(all(a[k] == b[k] for k in path_fields) for a, b in pairs)}
    a, b = pairs[different]
    prior_equal = all(all(x[k] == y[k] for k in path_fields) for x, y in pairs[:different])
    equal = {key: a[key] == b[key] for key in ("offset", "window_rng", "before", "gradient_norm", "pre_chuck")}
    equal["observation"] = a["architect"]["observation"] == b["architect"]["observation"]
    equal["features"] = a["architect"]["features"] == b["architect"]["features"]
    return {"first_action_divergence": a["step"], "previous_training_path_equal": prior_equal,
            "equal_pre_action_fields": equal, "identical_observation": prior_equal and all(equal.values()),
            "left_action": a["action"], "right_action": b["action"],
            "left_scores": a["architect"]["scores"], "right_scores": b["architect"]["scores"],
            "left_policy_hash": a["architect"]["policy_pre"], "right_policy_hash": b["architect"]["policy_pre"]}


def audit(directory, expected_source):
    evidence = Evidence(directory)
    result = evidence.json("results.json")
    assert result["status"] == "PASS"
    evidence.listed = result["artifacts"]
    protocol = evidence.json("protocol.json")
    manifest = evidence.json("manifest.json")
    assert result["manifest"] == manifest
    assert manifest["source_commit"] == expected_source and manifest["source_status"] == ""
    assert manifest["smoke"] is False
    assert manifest["protocol_sha256"] == evidence.verified["protocol.json"]["sha256"]
    assert manifest["seeds"] == protocol["seeds"] and manifest["steps"] == protocol["steps"]
    initial = life(evidence.read(protocol["input_life"]["file"]))
    assert evidence.verified[protocol["input_life"]["file"]] == {key: protocol["input_life"][key] for key in ("bytes", "sha256")}
    traces, primary, table, changes = {}, {}, [], []
    counts = collections.Counter(total_processes=0, host_updates=0, acquisition_checkpoints=0,
                                 measured_branches=0, branch_updates=0, fit_updates=0,
                                 frozen_source_continuation_checks=0)
    nonweight_checks = 0
    for record in result["runs"]:
        prefix = record["prefix"]
        rows = evidence.events(prefix + ".jsonl")
        assert evidence.verified[prefix + ".jsonl"] == record["trace"]
        header, terminal = rows[0], rows[-1]
        assert header["type"] == "living_run" and terminal["type"] == "living_summary"
        assert not header["fixture"] and not terminal["failed"]
        assert header["arm"] == record["arm"] and header["body"] == record["body"] and header["seed"] == record["seed"]
        arm, body, seed = header["arm"], header["body"], header["seed"]
        assert header["parameters"] == protocol["bodies"][body]
        hosts, evaluations = rows_of(rows, "rollout_step"), rows_of(rows, "evaluation")
        acquisitions, outcomes = rows_of(rows, "acquisition"), rows_of(rows, "acquisition_result")
        fits, comparisons = rows_of(rows, "fit"), rows_of(rows, "comparison")
        assert [r["step"] for r in hosts] == list(range(1, protocol["steps"] + 1))
        assert [r["step"] for r in evaluations] == protocol["evaluations_after_updates"]
        assert evaluations == record["evaluations"] and terminal == record["terminal"]
        first_life = life(evidence.read(prefix + ".policy.initial.bin"))
        last_life = life(evidence.read(prefix + ".policy.final.bin"))
        assert first_life["hash"] == header["initial_policy_hash"] and last_life["hash"] == terminal["policy_hash"]
        if arm != "canonical":
            assert first_life["weights"] == initial["weights"]
            assert first_life["updates"] == first_life["decisions"] == first_life["has_history"] == first_life["pending"] == 0
            assert last_life["updates"] == last_life["decisions"] == protocol["steps"] and last_life["pending"] == 0
        if arm == "frozen":
            assert last_life["weights"] == first_life["weights"]
        if header["resume_step"]:
            resumed = life(evidence.read(prefix + ".policy.resume.bin"))
            assert resumed["hash"] == hosts[header["resume_step"] - 1]["architect"]["policy_post"]
        is_primary = not header["resume_step"] and (arm == "canonical" or header["probes"])
        if is_primary:
            key = body, seed, arm
            assert key not in primary
            primary[key] = prefix
        current_fit_count = 0
        assert len(acquisitions) == len(outcomes)
        for source, acquired in zip(acquisitions, outcomes):
            checkpoint = source["checkpoint"]
            assert acquired["checkpoint"] == checkpoint and source["features"] == acquired["features"]
            assert source["source_hash"] == acquired["source_hash"] == acquired["restored_hash"]
            pre = life(evidence.read(prefix + f".acquire-{checkpoint}.pre.policy.bin"))
            post = life(evidence.read(prefix + f".acquire-{checkpoint}.post.policy.bin"))
            assert pre["hash"] == source["policy_hash"] == acquired["policy_before"]
            assert post["hash"] == acquired["policy_after"]
            assert pre["nonweights"] == post["nonweights"], "saved acquisition changed nonweight fields"
            assert pre["decisions"] == pre["updates"] == checkpoint - 1 and not pre["pending"]
            assert pre["learning_rate"] == f32(protocol["fitting"]["learning_rate"])
            assert f32(source["features"][15]) == pre["previous"][2]
            nonweight_checks += 1
            measured_rows = [r for r in comparisons if r["checkpoint"] == checkpoint]
            assert [r["action"] for r in measured_rows] == ["hold", "brake", "push"]
            assert all(r["horizon"] == 16 and r["finite"] for r in measured_rows)
            measured = [r["future_after"] for r in measured_rows]
            assigned = measured[-1:] + measured[:2] if arm == "rotated" else measured
            assert measured == acquired["measured_future_loss"] and assigned == acquired["assigned_future_loss"]
            own_fits = [r for r in fits if r["checkpoint"] == checkpoint]
            assert len(own_fits) == (0 if arm == "frozen" else 32)
            expected_hash = pre["hash"]
            for fitted in own_fits:
                current_fit_count += 1
                assert fitted["fit_step"] == current_fit_count and fitted["hash_before"] == expected_hash
                assert fitted["assigned_future_loss"] == assigned and fitted["measured_future_loss"] == measured
                values = list(map(f32, assigned))
                scale = max(1e-6 * (abs(values[0]) + 1), *(abs(values[0] - value) for value in values))
                assert fitted["scale"] == scale
                assert list(map(f32, fitted["target"])) == [f32((values[0] - value) / scale) for value in values]
                assert fitted["online_updates"] == fitted["online_decisions"] == checkpoint - 1
                expected_hash = fitted["hash_after"]
            assert expected_hash == post["hash"]
            assert choice(source["scores_before"]) == acquired["action_before"]
            assert choice(acquired["scores_after"]) == acquired["action_after"]
            host = hosts[checkpoint - 1]
            assert host["architect"]["features"] == source["features"]
            assert host["architect"]["observation"] == source["observation"]
            assert host["architect"]["scores"] == acquired["scores_after"]
            assert host["architect"]["policy_pre"] == post["hash"] and host["action"] == acquired["action_after"]
            if is_primary and acquired["action_before"] != acquired["action_after"]:
                changes.append({"body": body, "seed": seed, "arm": arm, "checkpoint": checkpoint,
                                "before_action": acquired["action_before"], "after_action": acquired["action_after"],
                                "measured_HBP": measured, "assigned_HBP": assigned,
                                "scores_before": acquired["scores_before"], "scores_after": acquired["scores_after"],
                                "target_HBP": own_fits[0]["target"], "scale": own_fits[0]["scale"],
                                "fit_count": len(own_fits), "first_fit": own_fits[0], "last_fit": own_fits[-1],
                                "source_hash": source["source_hash"], "restored_hash": acquired["restored_hash"],
                                "features_before_after_host_equal": True, "actual_host_action": host["action"],
                                "policy_before": pre["hash"], "policy_after": post["hash"],
                                "pre_life": evidence.verified[prefix + f".acquire-{checkpoint}.pre.policy.bin"],
                                "post_life": evidence.verified[prefix + f".acquire-{checkpoint}.post.policy.bin"]})
        action_counts = [sum(r["action"] == action for r in hosts) for action in range(4)]
        brake = [r for r in hosts if r["action"] == 2]
        decreases = sum(f32(r["post_chuck"]["dampen"]) < (f32(r["pre_chuck"]["dampen"]) if r["pre_chuck"]["initialized"] else 1) for r in brake)
        saturated = len(brake) - decreases
        floor = next((r["step"] for r in hosts if f32(r["post_chuck"]["dampen"]) == f32(.3)), None)
        assert terminal["action_counts"] == action_counts
        assert terminal["brake_changed_dampen"] == decreases and terminal["brake_at_bound"] == saturated
        assert record["first_dampen_floor"] == floor
        counts.update(total_processes=1, host_updates=len(hosts), acquisition_checkpoints=len(acquisitions),
                      measured_branches=len(comparisons), branch_updates=len(rows_of(rows, "branch_step")),
                      fit_updates=len(fits), frozen_source_continuation_checks=len(rows_of(rows, "continuation_check")))
        if is_primary:
            table.append({"body": body, "seed": seed, "arm": arm, "parameters": header["parameters"],
                          "heldout_curve": [{"step": r["step"], "loss": r["heldout_loss"]} for r in evaluations],
                          "final_loss": evaluations[-1]["heldout_loss"], "action_counts": action_counts,
                          "brake_decreases": decreases, "brake_saturated": saturated, "first_floor": floor,
                          "changed_fit_choices": sum(c["body"] == body and c["seed"] == seed and c["arm"] == arm for c in changes),
                          "policy_final_sha256": evidence.verified[prefix + ".policy.final.bin"]["sha256"]})
        traces[prefix] = rows
    counts["all_body_updates"] = counts["host_updates"] + counts["branch_updates"]
    assert dict(counts) == result["counts"]
    assert all(value == protocol["fixed_counts"][name] for name, value in counts.items())
    assert set(primary) == {(body, seed, arm) for body in protocol["body_order"] for seed in protocol["seeds"] for arm in protocol["arms"]}
    paired, divergence = [], []
    for body in protocol["body_order"]:
        for seed in protocol["seeds"]:
            finals = {arm: rows_of(traces[primary[body, seed, arm]], "evaluation")[-1]["heldout_loss"] for arm in protocol["arms"]}
            paired.append({"body": body, "seed": seed, **finals,
                           "living_minus_frozen": finals["living"] - finals["frozen"],
                           "living_minus_rotated": finals["living"] - finals["rotated"],
                           "living_minus_canonical": finals["living"] - finals["canonical"]})
            for other in ("frozen", "rotated"):
                divergence.append({"body": body, "seed": seed, "left": "living", "right": other,
                                   **paired_divergence(rows_of(traces[primary[body, seed, "living"]], "rollout_step"),
                                                       rows_of(traces[primary[body, seed, other]], "rollout_step"))})
    assert paired == result["paired_final"]
    for pair in result["probe_parity"] + result["resume_parity"]:
        left, right = traces[pair["left"]], traces[pair["right"]]
        full = pair in result["resume_parity"]
        keep = lambda rows: [r for r in rows if r["type"] not in ("living_run", "living_summary", "policy_resume")] if full else [r for r in rows if r["type"] in ("rollout_step", "evaluation")]
        assert keep(left) == keep(right)
        assert all(left[-1][key] == right[-1][key] for key in ("state_hash", "policy_hash", "action_counts"))
    assert evidence.index_bytes == (directory / ".sealed/anchors.json").read_bytes(), "sealed index changed during audit"
    living_changes = [change for change in changes if change["arm"] == "living"]
    return {"status": "PASS", "source_commit": expected_source, "results_identity": evidence.verified["results.json"],
            "sealed_index_identity": identity(evidence.index_bytes), "verified_artifacts": evidence.verified,
            "counts": dict(counts), "saved_nonweight_comparisons": nonweight_checks,
            "primary_runs": table, "paired_final": paired, "first_deployment_divergences": divergence,
            "direct_fit_action_changes": changes, "first_living_witness": living_changes[0] if living_changes else None,
            "witness_rule": "First direct living fit action change in fixed body, seed, checkpoint order; no loss-improvement filter.",
            "integrity_failures": result["integrity_failures"], "numerical_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh audit output path")
    result = audit(args.run.resolve(), args.source_commit)
    result["audit_source_sha256"] = identity(Path(__file__).read_bytes())["sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "primary_runs": len(result["primary_runs"]),
                      "verified_artifacts": len(result["verified_artifacts"]),
                      "direct_fit_action_changes": len(result["direct_fit_action_changes"]),
                      "output_identity": identity(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
