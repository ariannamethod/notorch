#!/usr/bin/env python3
"""Read-only diagnostics of the sealed future-credit experiment.

Requires Python 3 and NumPy. Reads retained JSONL receipts and canonical v1
checkpoints; computes descriptive statistics, activations and score Jacobians.
No model fitting, policy update, sentence generation or checkpoint writing.
The double-precision decoded-policy calculation is checked against all native
recorded scores before its activation/derivative statistics enter the report.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import platform
import struct

import numpy as np

ARMS = ("initial", "h4", "h0", "shuffled_h4")
FEATURES = (
    "embedding_0", "embedding_1", "embedding_2", "embedding_3",
    "connectedness", "left_similarity", "right_similarity", "coherence",
    "novelty", "repetition", "phase_gate", "legacy_score_margin", "position",
    "reseed_fraction", "temperature_over_16", "has_left", "has_right",
    "mean_score_fraction", "score_fraction", "ema_reward", "ema_connectedness",
    "ema_novelty", "keep_frequency", "left_frequency", "right_frequency",
    "last_keep", "last_left", "last_right", "last_reward",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def policy(path):
    raw = path.read_bytes()
    require(len(raw) == 2296 and raw[:8] == b"NTSPA001", "unexpected v1 file")
    require(struct.unpack_from("<I", raw, 8)[0] == len(raw), "v1 length differs")
    require(fnv(raw[:-8]) == struct.unpack_from("<Q", raw, len(raw) - 8)[0],
            "v1 checksum differs")
    # Canonical header12 + three versions12 + config56 + config fingerprint8.
    offset = 88
    w1, b1, w2, b2 = [], [], [], []
    for _ in range(8):
        w1.append(struct.unpack_from("<29f", raw, offset)); offset += 116
        b1.append(struct.unpack_from("<f", raw, offset)[0]); offset += 4
    for _ in range(3):
        w2.append(struct.unpack_from("<8f", raw, offset)); offset += 32
        b2.append(struct.unpack_from("<f", raw, offset)[0]); offset += 4
    require(offset == 1156, "unexpected canonical policy extent")
    return tuple(np.array(x, dtype=np.float64) for x in (w1, b1, w2, b2)), f"{fnv(raw):016x}"


def read_set(path):
    rows = [json.loads(line) for line in path.open()]
    imports = [row for row in rows if row["type"] == "import"]
    require(len(imports) == 48, "expected 48 captured states")
    require([r["source"]["index"] for r in imports] == list(range(48)), "source order differs")
    x = np.array([r["features"] for r in imports], dtype=np.float64)
    require(x.shape == (48, 29), "feature shape differs")
    readouts = {(r["label"], r["source"]["index"], r["horizon"]): r
                for r in rows if r["type"] == "readout"}
    return x, imports, readouts


def selected_statistics(rows):
    changed = [r for r in rows if r["action"]["kind"] != 0]
    margins = [r["scores_before"][r["action"]["kind"]] - r["scores_before"][0]
               for r in changed]
    return {
        "states": len(rows), "mean_huber_loss": float(np.mean([r["loss_before"] for r in rows])),
        "mean_advantage_over_keep": float(np.mean([r["advantage_over_keep"] for r in rows])),
        "actions": dict(collections.Counter(str(r["action"]["kind"]) for r in rows)),
        "changed": len(changed), "beneficial_changes": sum(r["advantage_over_keep"] > 0 for r in changed),
        "harmful_changes": sum(r["advantage_over_keep"] < 0 for r in changed),
        "optimal": sum(r["optimal"] for r in rows),
        "changed_margin_min": min(margins) if margins else None,
        "changed_margin_max": max(margins) if margins else None,
        "positive_selected_advantage_sum": sum(r["advantage_over_keep"] for r in rows if r["advantage_over_keep"] > 0),
        "negative_selected_advantage_sum": sum(r["advantage_over_keep"] for r in rows if r["advantage_over_keep"] < 0),
    }


def score_jacobian(x, h, w1, w2, rows):
    result = []
    for i, receipt in enumerate(rows):
        for action in (1, 2):
            if not receipt["action_mask"] & (1 << action):
                continue
            dh = w2[action] * (1 - h[i] ** 2)
            row = np.zeros(267)
            row[:232] = (dh[:, None] * x[i]).ravel()
            row[232:240] = dh
            row[240 + action * 8:248 + action * 8] = h[i]
            row[264 + action] = 1
            result.append(row)
    singular = np.linalg.svd(result, compute_uv=False)
    return {
        "shape": [len(result), 267], "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "rank_relative_1e_8": int(np.sum(singular > singular[0] * 1e-8)),
        "rank_relative_1e_6": int(np.sum(singular > singular[0] * 1e-6)),
        "singular_values": singular.tolist(),
    }


def analyze(directory):
    paths = [directory / f"policy.{arm}.life.bin" for arm in ARMS]
    paths += [directory / f"{split}.readout.jsonl" for split in ("training", "evaluation")]
    paths.append(directory / "policy.fit.jsonl")
    sets = {split: read_set(directory / f"{split}.readout.jsonl")
            for split in ("training", "evaluation")}
    policies = {arm: policy(directory / f"policy.{arm}.life.bin") for arm in ARMS}
    report = {
        "schema": 1, "operation": "read-only conditioning diagnosis; no training",
        "source_experiment": "SPA future-credit v1, authoritative execution-1",
        "body": "frozen 450688-parameter SimpleLLM; policy29->8tanh->3,267 parameters",
        "sample": {"training_seeds": [42, 73], "evaluation_seeds": [101, 211],
                   "states_per_split": 48, "all_96_states_already_exposed": True},
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__},
        "script": artifact(Path(__file__)),
        "command": "python3 experiments/spa_agent/replicates/conditioning.py --execution-dir <retained-execution-1> --output experiments/spa_agent/replicates/conditioning.json",
        "inputs": {p.name: artifact(p) for p in paths},
        "splits": {}, "policies": {}, "fit_trajectory": {},
        "method": {
            "reported_scores": "Native retained scores are authoritative; decoded double scores are only used for activation and derivative diagnostics and checked against them.",
            "loss": "Mean of native per-state mean-Huber losses across48 states.",
            "gradient": "Gradient of the mean per-state loss at the one sealed policy; each state weights each valid head by reciprocal mask size.",
            "scaled_sensitivity": "Per-feature input standard deviation times RMS local score derivative; group statistic is Euclidean norm.",
            "jacobian": "All valid non-KEEP score derivatives at the sealed policy, columns ordered w1,b1,w2,b2. Singular values use NumPy float64 SVD; no solve or parameter update.",
            "fit_curve": "Online before/after losses are recorded at successive parameter states within each epoch, separate from fixed-checkpoint batch loss.",
        },
    }
    all_x = np.vstack([sets[s][0] for s in sets])
    for arm, ((w1, b1, w2, b2), life_hash) in policies.items():
        z = all_x @ w1.T + b1; hidden = np.tanh(z); decoded = hidden @ w2.T + b2
        receipts = [sets[split][2][arm, i, 4] for split in sets for i in range(48)]
        require(all(r["model_hash"] == life_hash for r in receipts), "readout/checkpoint hash differs")
        native = np.array([r["scores_before"] for r in receipts])
        error = float(np.max(np.abs(decoded - native)))
        require(error < 1e-7, "decoded score disagrees with native record")
        report["policies"][arm] = {
            "canonical_hash": life_hash, "maximum_native_score_difference": error,
            "maximum_absolute_parameter": max(float(np.max(np.abs(x))) for x in (w1, b1, w2, b2)),
            "w1_norm": float(np.linalg.norm(w1)), "w2_norm": float(np.linalg.norm(w2)),
            "b1_norm": float(np.linalg.norm(b1)), "b2": b2.tolist(),
            "hidden_pre_activation_range": [float(z.min()), float(z.max())],
            "maximum_absolute_hidden": float(np.abs(hidden).max()),
            "minimum_tanh_derivative": float((1 - hidden ** 2).min()),
            "embedding_preactivation_sd_by_hidden": np.std(all_x[:, :4] @ w1[:, :4].T, axis=0).tolist(),
            "other_preactivation_sd_by_hidden": np.std(all_x[:, 4:] @ w1[:, 4:].T, axis=0).tolist(),
        }
    w1, b1, w2, b2 = policies["h4"][0]
    difference = w1 - policies["initial"][0][0]
    report["h4_first_layer_change"] = {
        "norm": float(np.linalg.norm(difference)),
        "embedding_columns_norm": float(np.linalg.norm(difference[:, :4])),
        "other_columns_norm": float(np.linalg.norm(difference[:, 4:])),
    }
    all_h4_rows = []
    for split, (x, imports, readouts) in sets.items():
        summary = {"arms": {}, "features": [], "h4_per_target": {}, "h4_heads": {}}
        for arm in ARMS:
            summary["arms"][arm] = selected_statistics([readouts[arm, i, 4] for i in range(48)])
        for i, name in enumerate(FEATURES):
            summary["features"].append({"index": i, "name": name, "minimum": float(x[:, i].min()),
                                        "maximum": float(x[:, i].max()), "sd": float(x[:, i].std())})
        rows = [readouts["h4", i, 4] for i in range(48)]; all_h4_rows.extend(rows)
        mask = np.array([[(r["action_mask"] >> a) & 1 for a in range(3)] for r in rows], dtype=bool)
        target = np.array([r["targets"] for r in rows]); prediction = np.array([r["scores_before"] for r in rows])
        require(np.abs((prediction - target)[mask]).max() < 1,
                "fixed-checkpoint diagnostic expects quadratic Huber residuals")
        hidden = np.tanh(x @ w1.T + b1)
        score_gradient = (prediction - target) * mask / mask.sum(1)[:, None] / len(x)
        hidden_gradient = (score_gradient @ w2) * (1 - hidden ** 2)
        gradients = (hidden_gradient.T @ x, hidden_gradient.sum(0), score_gradient.T @ hidden,
                     score_gradient.sum(0))
        summary["fixed_checkpoint_gradient"] = {
            "l2": float(np.sqrt(sum(np.sum(g ** 2) for g in gradients))),
            "block_l2_w1_b1_w2_b2": [float(np.linalg.norm(g)) for g in gradients],
            "b2": gradients[3].tolist(),
            "embedding_w1_l2": float(np.linalg.norm(gradients[0][:, :4])),
            "other_w1_l2": float(np.linalg.norm(gradients[0][:, 4:])),
        }
        summary["maximum_valid_residual"] = float(np.abs((prediction - target)[mask]).max())
        summary["valid_head_rmse"] = float(np.sqrt(np.mean((prediction - target)[mask] ** 2)))
        summary["non_keep_jacobian"] = score_jacobian(x, hidden, w1, w2, rows)
        for action in (1, 2):
            valid = mask[:, action]; targets = target[valid, action]; scores = prediction[valid, action]
            feature_derivative = ((1 - hidden ** 2) * w2[action]) @ w1
            influence = np.std(x, axis=0) * np.sqrt(np.mean(feature_derivative ** 2, axis=0))
            summary["h4_heads"][str(action)] = {
                "mean_target": float(targets.mean()), "mean_score": float(scores.mean()),
                "rmse": float(np.sqrt(np.mean((targets - scores) ** 2))),
                "pearson": float(np.corrcoef(targets, scores)[0, 1]),
                "positive_targets": int(sum(targets > 0)), "positive_scores": int(sum(scores > 0)),
                "embedding_scaled_sensitivity_norm": float(np.linalg.norm(influence[:4])),
                "other_scaled_sensitivity_norm": float(np.linalg.norm(influence[4:])),
                "scaled_sensitivity_by_feature": influence.tolist(),
            }
        for position in range(4):
            selected = [r for r in rows if r["source"]["target"] == position]
            entry = selected_statistics(selected)
            entry["mean_targets"] = np.mean([r["targets"] for r in selected], axis=0).tolist()
            entry["mean_scores"] = np.mean([r["scores_before"] for r in selected], axis=0).tolist()
            summary["h4_per_target"][str(position)] = entry
        worst = sorted(rows, key=lambda r: r["loss_before"], reverse=True)
        loss_sum = sum(r["loss_before"] for r in rows)
        summary["largest_loss_rows"] = [{"index": r["source"]["index"], "seed": r["source"]["seed"],
                                         "target": r["source"]["target"], "loss": r["loss_before"]} for r in worst[:5]]
        summary["top_rows_loss_fraction"] = {str(n): sum(r["loss_before"] for r in worst[:n]) / loss_sum
                                             for n in (1, 3, 5)}
        report["splits"][split] = summary
    report["all_96_non_keep_jacobian"] = score_jacobian(all_x, np.tanh(all_x @ w1.T + b1), w1, w2, all_h4_rows)
    fits = collections.defaultdict(list)
    for line in (directory / "policy.fit.jsonl").open():
        row = json.loads(line)
        if row["type"] == "fit":
            fits[row["arm"]].append(row)
    for arm, rows in fits.items():
        require(len(rows) == 48 * 512, "fit count differs")
        trajectory = []
        for epoch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 480, 511, 512):
            selected = [r for r in rows if r["epoch"] == epoch]
            trajectory.append({"epoch": epoch, "mean_online_loss_before": float(np.mean([r["loss_before"] for r in selected])),
                               "mean_online_loss_after": float(np.mean([r["loss_after"] for r in selected]))})
        report["fit_trajectory"][arm] = trajectory
        if arm == "h4":
            last = [r for r in rows if r["epoch"] == 512]
            report["last_h4_epoch_online_bias_gradient"] = [sum(
                (r["scores_before"][a] - r["targets"][a]) / r["action_mask"].bit_count()
                for r in last if r["action_mask"] & (1 << a)) / 48 for a in range(3)]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.execution_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PASS conditioning:96 retained states,4 sealed policies,1152 native scores checked; no training")


if __name__ == "__main__":
    main()
