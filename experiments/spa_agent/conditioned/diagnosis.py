#!/usr/bin/env python3
"""Read-only diagnosis of the exposed SPA repeated-consequence experiment.

Only authenticated committed receipts are read. No policy is fitted, no body
is run, and no new-seed outcome is consulted. Python's standard library suffices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "experiments/spa_agent/replicates/receipts.json":
        "de80eb47b78ce0a4474decb1a5317b26d3c27400756eef3829b2dc7d2d2296de",
    "experiments/spa_agent/replicates/conditioning.json":
        "e790afc0eaf94fdbf5957579e8a624bd9f9933c7563890ecf741c59dcf2cf6c0",
    "experiments/spa_agent/replicates/protocol.json":
        "c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75",
}
FLOOR = struct.unpack("<f", struct.pack("<f", .001))[0]


def require(value, message):
    if not value:
        raise ValueError(message)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source(path):
    local = ROOT / path
    # Sparse checkouts may omit an authenticated historical receipt.
    raw = local.read_bytes() if local.exists() else subprocess.check_output(
        ["git", "show", "HEAD:" + path], cwd=ROOT)
    require(identity(raw)["sha256"] == PINS[path], "source identity: " + path)
    return json.loads(raw), identity(raw)


def distribution(values):
    return {"count": len(values), "minimum": min(values),
            "median": statistics.median(values), "maximum": max(values),
            "mean": statistics.mean(values), "sd": statistics.pstdev(values)}


def signed_mass(values):
    positive = sum(value for value in values if value > 0)
    negative = -sum(value for value in values if value < 0)
    return {"positive": sum(value > 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "mean": statistics.mean(values), "positive_l1": positive,
            "negative_l1": negative,
            "positive_to_negative_l1": positive / negative if negative else None}


def target_rows(execution, split):
    original = [row for row in execution[split]["state_comparisons"]
                if row["label"] == "initial" and row["horizon"] == 4]
    require(len(original) == 48, "source count")
    require(len({(r["seed"], r["snapshot"]) for r in original}) == 48,
            "source uniqueness")
    rows = []
    for row in original:
        actions = row["actions"]
        require(actions[0]["kind"] == 0, "KEEP first")
        # Stored means average individually clipped native f32 rewards in double.
        # C then rounds each action mean once and subtracts the rounded KEEP mean.
        means = {a["kind"]: f32(a["mean_reward"]) for a in actions}
        advantages = {kind: f32(value - means[0]) for kind, value in means.items()}
        maximum = max(abs(value) for value in advantages.values())
        scale = max(FLOOR, maximum)
        normalized = {kind: f32(value / scale) for kind, value in advantages.items()}
        rank = sorted(advantages, key=lambda kind: (-advantages[kind], kind))
        require(rank == sorted(normalized, key=lambda kind: (-normalized[kind], kind)),
                "normalization changed measured action order")
        item = {"seed": row["seed"], "snapshot": row["snapshot"],
                "target": row["target"], "max_absolute_advantage": maximum,
                "scale": scale, "gain": 1.0 / scale, "floor_active": maximum < FLOOR,
                "measured_best": rank[0], "actions": []}
        for action in actions:
            kind = action["kind"]
            draws = action["replicate_advantages"]
            require(len(draws) == 8, "replicate count")
            require(abs(statistics.mean(draws) - action["mean_advantage_over_keep"]) < 1e-14,
                    "paired mean differs")
            require(abs(advantages[kind] - statistics.mean(draws)) < 2e-8,
                    "rounded mean reward target differs")
            item["actions"].append({"kind": kind, "native_target": advantages[kind],
                "conditioned_target": normalized[kind],
                "paired_mean": action["mean_advantage_over_keep"],
                "paired_standard_error": action["paired_standard_error"],
                "positive_draws": sum(value > 0 for value in draws),
                "negative_draws": sum(value < 0 for value in draws),
                "draws": draws})
        rows.append(item)
    return rows


def analyze_rows(rows):
    nonkeep = [action for row in rows for action in row["actions"] if action["kind"]]
    positives = [action for action in nonkeep if action["native_target"] > 0]
    groups = {}
    for target in (None, 0, 1, 2, 3):
        for kind in (1, 2):
            chosen = [action for row in rows if target is None or row["target"] == target
                      for action in row["actions"] if action["kind"] == kind]
            if chosen:
                groups[f"target_{target}_action_{kind}"] = {
                    "raw": signed_mass([a["native_target"] for a in chosen]),
                    "conditioned": signed_mass([a["conditioned_target"] for a in chosen])}
    raw_zero_loss = statistics.mean(sum(.5 * a["native_target"] ** 2 for a in r["actions"])
                                    / len(r["actions"]) for r in rows)
    conditioned_zero_loss = statistics.mean(sum(.5 * a["conditioned_target"] ** 2
        for a in r["actions"]) / len(r["actions"]) for r in rows)
    return {"states": len(rows), "valid_nonkeep_alternatives": len(nonkeep),
        "states_with_positive_alternative": sum(any(a["native_target"] > 0
            for a in r["actions"]) for r in rows),
        "positive_alternatives": len(positives),
        "positive_mean_above_one_paired_se": sum(a["paired_mean"] > a["paired_standard_error"]
                                                   for a in positives),
        "positive_mean_above_two_paired_se": sum(a["paired_mean"] > 2 * a["paired_standard_error"]
                                                   for a in positives),
        "positive_with_at_least_three_negative_draws": sum(a["negative_draws"] >= 3 for a in positives),
        "max_absolute_advantage": distribution([r["max_absolute_advantage"] for r in rows]),
        "gain": distribution([r["gain"] for r in rows]),
        "floor_active_sources": [{k: r[k] for k in ("seed", "snapshot", "target", "actions")}
                                   for r in rows if r["floor_active"]],
        "zero_head_mean_huber_raw": raw_zero_loss,
        "zero_head_mean_huber_conditioned": conditioned_zero_loss,
        "groups": groups, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {path: source(path) for path in PINS}
    receipts = loaded["experiments/spa_agent/replicates/receipts.json"][0]
    previous = loaded["experiments/spa_agent/replicates/conditioning.json"][0]
    executions = receipts["executions"]
    require(len(executions) == 2, "two parent executions")
    for name in ("training", "evaluation", "fit", "policy_seal"):
        require(executions[0][name] == executions[1][name], "repeated identity: " + name)
    execution = executions[0]
    report = {"schema": 1, "name": "SPA repeated-target scale diagnosis",
        "operation": "read-only arithmetic over authenticated exposed receipts",
        "body_generation_runs": 0, "policy_fitting_runs": 0,
        "command": "python3 experiments/spa_agent/conditioned/diagnosis.py --output experiments/spa_agent/conditioned/diagnosis.json",
        "script": identity(Path(__file__).read_bytes()),
        "inputs": {path: item[1] for path, item in loaded.items()},
        "sample": {"training_seeds": [42, 73], "exposed_evaluation_seeds": [307, 401],
            "states_per_split": 48, "episodes_per_split": 12, "repeated_draws_per_state": 8,
            "unit": "captured sentence state; eight draws are nested repeated outcomes"},
        "conditioning": {"floor_decimal": .001, "floor_float32": FLOOR,
            "maximum_gain": 1.0 / FLOOR,
            "rule": "f32(raw_KEEP_relative_target / max(f32(.001), max_valid_abs_raw_target))",
            "floor_choice": "Exposed training data only: cap amplification at approximately 1000; preserve sub-floor near-tie magnitude. One training state invokes the floor.",
            "causal_axis": "target conditioning changes both gradient amplitude and relative weighting of states",
            "preserved": "action signs, within-state rank, KEEP zero, raw reward coefficients, source features, capacity, rate, row order and update budget",
            "mechanism_boundary": "This comparison measures amplitude and state reweighting together."},
        "splits": {split: analyze_rows(target_rows(execution, split))
                   for split in ("training", "evaluation")},
        "parent_predictions": {split: [row for row in execution[split]["summaries"]
            if row["horizon"] == 4 and row["group"] == "combined"]
            for split in ("training", "evaluation")},
        "parent_online_fitting": execution["fit"],
        "inherited_feature_diagnosis": {
            "source": "Earlier single_h4 conditioning receipt; training features are unchanged in mean8 experiment.",
            "training_features": previous["splits"]["training"]["features"],
            "single_h4_initial_to_acquired_first_layer": previous["h4_first_layer_change"],
            "single_h4_hidden": previous["policies"]["h4"],
            "scope": "Hidden derivatives here describe the saved single_h4 policy, not unavailable mean8 weights."},
        "mechanism": [
            "KEEP begins with zero output weights and zero bias, receives zero targets, and remains exactly zero in the comparison learner.",
            "Mean8 and shuffled mean8 select KEEP on all 48 training and all 48 exposed evaluation states; their nonKEEP heads never win.",
            "Every nonKEEP action-by-position training mean is negative. Useful measured interventions require conditional selection within a group.",
            "Conditioning magnifies small raw targets and changes which states dominate residual fitting; class counts and observed draw uncertainty remain unchanged.",
            "The target3 continuation overwrites the intervention before reading it; normalized target3 LEFT is -1 on all states because only extra cost survives.",
            "The source features retain their existing scales. Small embedding variation and the fixed order remain separate measured candidates."
        ],
        "falsifiable_questions": [
            "At the fixed 24576-update budget, does conditioned acquisition fit positive and negative training choices more accurately than raw mean8?",
            "With identical observation/history/RNG, do conditioned acquired weights select different actions from initial and raw mean8?",
            "On fresh states, does correctly associated conditioned acquisition beat conditioned shuffled credit and fixed KEEP on the unchanged raw reward?",
            "If conditioning changes actions without improving raw fresh-state consequences, record the changed actions and regression together."
        ]}
    require(report["splits"]["training"]["positive_alternatives"] == 14, "training positives")
    require(report["splits"]["evaluation"]["positive_alternatives"] == 4, "exposed evaluation positives")
    require(len(report["splits"]["training"]["floor_active_sources"]) == 1, "training floor count")
    require(not report["splits"]["evaluation"]["floor_active_sources"], "evaluation floor count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "output": identity(args.output.read_bytes()),
                      "training_positive_alternatives": 14, "evaluation_positive_alternatives": 4,
                      "training_floor_states": 1, "generation_runs": 0, "fit_runs": 0}))


if __name__ == "__main__":
    main()
