#!/usr/bin/env python3
"""Fixed future-credit replay from retained sentence-action consequences."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "134460b76ff7fecc9934eea8187d8aac4f6d0154921db5989de86cc482ee1861"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(result)
    return result


SCENARIOS = module("spa_future_scenarios", HERE.parent / "scenarios/run.py")
V1 = SCENARIOS.V1
TRACE_IO = module("spa_future_trace_io", ROOT / "tests/test_spa_trace_io.py")
AXES = SCENARIOS.AXES
LABELS = ("initial", "h4", "h0", "shuffled_h4", "keep", "left", "right")
POLICIES = LABELS[:4]
HORIZONS = (0, 4)
MEASURED_HORIZONS = (0, 1, 4)
SOURCE_FILES = tuple(dict.fromkeys((*SCENARIOS.SOURCE_FILES,
    "examples/spa_agent_future.c", "experiments/spa_agent/future/run.py",
    "experiments/spa_agent/future/protocol.json", "tests/test_spa_future.py",
    "tests/test_spa_trace_io.py", "tests/test_spa_agent_future.c",
    "experiments/spa_agent/scenarios/receipts.json", "experiments/spa_agent/scenarios/raw_traces.jsonl.gz")))
require = SCENARIOS.require


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fnv_bytes(raw: bytes) -> str:
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def feature_witness(sample: dict) -> str:
    raw = struct.pack("<QII29f", int(sample["life_hash"], 16), sample["target"],
                      sample["sentence_count"], *sample["features"])
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def action_mask(target: int, count: int) -> int:
    return sum(1 << kind for kind in SCENARIOS.valid_actions(target, count))


def parse_raw(raw: bytes) -> tuple[list[dict], list[str]]:
    require(raw.endswith(b"\n"), "source trace lacks terminal newline")
    rows, hashes = [], []
    for line in raw.splitlines(keepends=True):
        row = json.loads(line, object_pairs_hook=SCENARIOS.unique_object)
        require(isinstance(row, dict), "source trace record must be an object")
        SCENARIOS.finite_tree(row)
        rows.append(row)
        hashes.append(digest_bytes(line))
    return rows, hashes


def archive_streams(path: Path, expected: str) -> dict[tuple[int, str], bytes]:
    raw = path.read_bytes()
    require(digest_bytes(raw) == expected, "retained source archive SHA256 mismatch")
    streams = defaultdict(bytearray)
    for line in gzip.decompress(raw).splitlines():
        envelope = json.loads(line, object_pairs_hook=SCENARIOS.unique_object)
        require(set(envelope) == {"seed", "stream", "raw"}, "source archive envelope differs")
        require(type(envelope["seed"]) is int and isinstance(envelope["raw"], str), "invalid source archive record")
        require(envelope["stream"] in ("ordinary_off", "ordinary_on", "scenarios"), "unknown source stream")
        streams[envelope["seed"], envelope["stream"]].extend(envelope["raw"].encode())
    return {key: bytes(raw) for key, raw in streams.items()}


def samples_from_streams(streams: dict, seeds: list[int], scenario_protocol: dict) -> list[dict]:
    """Bind original features to complete, independently checked native forks."""
    require(set(streams) == {(seed, stream) for seed in seeds
                            for stream in ("ordinary_off", "ordinary_on", "scenarios")},
            "source archive stream coverage differs")
    validation_protocol = copy.deepcopy(scenario_protocol)
    validation_protocol["seeds"] = seeds
    samples = []
    for seed in seeds:
        off, on = streams[seed, "ordinary_off"], streams[seed, "ordinary_on"]
        require(off == on, "source ordinary off/on traces differ")
        TRACE_IO.validate(off, seed)
        ordinary, ordinary_hashes = parse_raw(off)
        scenarios, scenario_hashes = parse_raw(streams[seed, "scenarios"])
        SCENARIOS.validate_seed(scenarios, ordinary, validation_protocol)
        ordinary_index = {(row["episode"], row["step"]): (row, sha) for row, sha in zip(ordinary, ordinary_hashes)
                          if row["type"] == "decision" and row["arm"] == "learned"}
        snapshots = {row["snapshot"]: (row, sha) for row, sha in zip(scenarios, scenario_hashes) if row["type"] == "snapshot"}
        measurements = {(row["snapshot"], row["horizon"], row["action"]["kind"]): (row, sha)
                        for row, sha in zip(scenarios, scenario_hashes) if row["type"] == "measurement"}
        for sid in range(validation_protocol["episodes"] * validation_protocol["snapshots_per_episode"]):
            snapshot, snapshot_sha = snapshots[sid]
            main, main_sha = ordinary_index[snapshot["episode"], snapshot["step"]]
            require(snapshot["features"] == main["features"], "retained feature-source association mismatch")
            sample = {"ordinal": len(samples), "seed": seed, "episode": snapshot["episode"],
                "step": snapshot["step"], "snapshot": sid, "target": snapshot["target"], "sentence_count": 4,
                "life_hash": snapshot["life_hash"], "body_hash": snapshot["body_hash"],
                "agent_rng": snapshot["agent_rng"], "host_rng": snapshot["host_rng_before"],
                "features": snapshot["features"], "snapshot_raw_sha256": snapshot_sha,
                "ordinary_decision_raw_sha256": main_sha, "outcomes": {}}
            sample["feature_witness"] = feature_witness(sample)
            sample["action_mask"] = action_mask(sample["target"], sample["sentence_count"])
            for horizon in MEASURED_HORIZONS:
                outcomes = []
                for kind in sorted(SCENARIOS.valid_actions(sample["target"], sample["sentence_count"])):
                    row, sha = measurements[sid, horizon, kind]
                    outcomes.append({"action": row["action"], "before": row["before"], "after": row["after"],
                        "cost": row["cost_normalized"], "reward": row["reward"], "raw_sha256": sha,
                        "initial_generated": row["initial_generated"], "cumulative_generated": row["cumulative_generated"],
                        "chain": row["chain"]})
                sample["outcomes"][horizon] = outcomes
            samples.append(sample)
    return samples


def verify_source_association(samples: list[dict], original: list[dict]) -> None:
    """Reject a changed feature, coordinate, mask, metric or outcome association."""
    require(len(samples) == len(original), "source association sample count mismatch")
    for given, expected in zip(samples, original):
        require(given == expected, "source feature/outcome association mismatch")
        require(given["feature_witness"] == feature_witness(given), "source feature witness mismatch")


def shuffle_donors(samples: list[dict]) -> list[int]:
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample["sentence_count"], sample["target"]].append(index)
    donors = list(range(len(samples)))
    for group in groups.values():
        require(len(group) > 1, "shuffle group needs two source states")
        for offset, index in enumerate(group):
            donors[index] = group[(offset + 1) % len(group)]
    return donors


def write_dataset(path: Path, samples: list[dict], protocol_sha: str, archive_sha: str) -> None:
    require(len(protocol_sha) == len(archive_sha) == 64, "dataset requires SHA256 identities")
    lines = ["NT_SPA_FUTURE_V1", "PROTOCOL " + protocol_sha, "ARCHIVE " + archive_sha, f"COUNT {len(samples)}"]
    floats = lambda values: " ".join(format(f32(value), ".9g") for value in values)
    for ordinal, sample in enumerate(samples):
        require(sample["ordinal"] == ordinal, "dataset ordinal mismatch")
        require(sample["feature_witness"] == feature_witness(sample), "dataset feature witness mismatch")
        fields = (ordinal, sample["seed"], sample["episode"], sample["step"], sample["target"], sample["sentence_count"],
                  sample["life_hash"], sample["body_hash"], sample["agent_rng"], sample["host_rng"], sample["snapshot_raw_sha256"])
        lines.extend(("SNAPSHOT " + " ".join(map(str, fields)), "FEATURES " + floats(sample["features"]),
                      "FEATURE_WITNESS " + sample["feature_witness"]))
        for horizon in HORIZONS:
            lines.append(f"HORIZON {horizon} {sample['action_mask']}")
            for outcome in sample["outcomes"][horizon]:
                action = outcome["action"]
                source = action["source"] if action["source"] is not None else 4294967295
                lines.extend((f"OUTCOME {action['kind']} {action['target']} {source} {outcome['raw_sha256']}",
                              "BEFORE " + floats(outcome["before"][axis] for axis in AXES),
                              "AFTER " + floats(outcome["after"][axis] for axis in AXES),
                              "COST " + floats((outcome["cost"],))))
            lines.append("END_HORIZON")
        lines.append("END_SNAPSHOT")
    lines.append("END")
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def native_rewards(sample: dict, horizon: int) -> tuple[list[float], list[float]]:
    """Independent frozen raw-axis target calculation, allowing F32 rounding."""
    rewards = [0.0, 0.0, 0.0]
    for row in sample["outcomes"][horizon]:
        rewards[row["action"]["kind"]] = SCENARIOS.reward(row["before"], row["after"], row["cost"])
    targets = [rewards[kind] - rewards[0] if sample["action_mask"] & (1 << kind) else 0.0 for kind in range(3)]
    return rewards, targets


def near(actual: float, expected: float, message: str, tolerance=3e-7) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual - expected) <= tolerance, message)


def huber(scores: list[float], targets: list[float], mask: int) -> float:
    errors = [abs(scores[kind] - targets[kind]) for kind in range(3) if mask & (1 << kind)]
    return sum(.5 * error * error if error <= 1 else error - .5 for error in errors) / len(errors)


def source_identity(sample: dict) -> dict:
    return {"index": sample["ordinal"], "seed": sample["seed"], "episode": sample["episode"],
            "step": sample["step"], "target": sample["target"], "count": sample["sentence_count"],
            "life_hash": sample["life_hash"], "body_hash": sample["body_hash"],
            "agent_rng": sample["agent_rng"], "host_rng": sample["host_rng"],
            "snapshot_sha256": sample["snapshot_raw_sha256"], "feature_witness": sample["feature_witness"]}


def outcome_ids(sample: dict, horizon: int) -> list:
    result = [None, None, None]
    for row in sample["outcomes"][horizon]:
        result[row["action"]["kind"]] = row["raw_sha256"]
    return result


def validate_import(rows: list[dict], samples: list[dict], dataset: Path, protocol_sha: str, archive_sha: str) -> None:
    require(rows and rows[0] == {"type": "dataset", "protocol_sha256": protocol_sha,
        "archive_sha256": archive_sha, "dataset_fnv1a": fnv_bytes(dataset.read_bytes()), "samples": len(samples)},
        "native dataset identity mismatch")
    imports = [row for row in rows if row["type"] == "import"]
    require(len(imports) == len(samples), "native import count mismatch")
    for row, sample in zip(imports, samples):
        require(row["source"] == source_identity(sample), "native feature-source association mismatch")
        require([f32(value) for value in row["features"]] == [f32(value) for value in sample["features"]], "native imported features differ")
        require(row["feature_witness_valid"] is True, "native feature witness invalid")
        for horizon in HORIZONS:
            require(row[f"h{horizon}_outcome_sha256"] == outcome_ids(sample, horizon), "native imported outcome identities differ")


def validate_receipt(row: dict, source: dict, horizon: int, learning_rate: float) -> None:
    require(row["horizon"] == horizon, "fit horizon differs from registered arm")
    require(row["action_mask"] == source["action_mask"], "fit action mask differs from source")
    near(row["learning_rate"], learning_rate, "fit learning rate differs", 2e-9)
    rewards, targets = native_rewards(source, horizon)
    for field, expected in (("rewards", rewards), ("targets", targets)):
        require(len(row[field]) == 3, "fit head count differs")
        for kind, value in enumerate(expected):
            near(row[field][kind], value, f"fit {field} differs from raw outcome")
    for field in ("scores_before", "scores_after"):
        require(len(row[field]) == 3 and all(math.isfinite(value) for value in row[field]), "invalid fit scores")
        near(row["loss_before" if field == "scores_before" else "loss_after"],
             huber(row[field], row["targets"], row["action_mask"]), "fit mean Huber loss differs", 2e-7)


def validate_fit(rows: list[dict], samples: list[dict], dataset: Path, protocol: dict,
                 archive_sha: str) -> dict:
    """Every replay update is joined to its original feature and donor outcomes."""
    validate_import(rows, samples, dataset, FROZEN_PROTOCOL_SHA256, archive_sha)
    allowed = {"dataset", "import", "fit_run", "saved_life", "fit", "arm_summary", "fit_summary"}
    require(all(row["type"] in allowed for row in rows), "unknown fit record")
    headers = [row for row in rows if row["type"] == "fit_run"]
    require(len(headers) == 1, "one fit run header required")
    header = headers[0]
    training = protocol["training"]
    require(header["samples"] == len(samples) and header["epochs"] == training["epochs"]
            and header["seed"] == training["policy_seed"] and header["exploration"] == 0
            and header["parameters"] == 267, "native training recipe differs")
    near(header["learning_rate"], training["learning_rate"], "native training rate differs", 2e-9)
    initial_hash = header["initial_hash"]
    fits = [row for row in rows if row["type"] == "fit"]
    per_arm = len(samples) * training["epochs"]
    require(len(fits) == 3 * per_arm, "incomplete native fit trace")
    donors = shuffle_donors(samples)
    epochs, arm_hashes = [], {"initial": initial_hash}
    for ai, arm in enumerate(POLICIES[1:]):
        prior = initial_hash
        for epoch in range(training["epochs"]):
            before_loss = after_loss = 0.0
            for index, source in enumerate(samples):
                local = epoch * len(samples) + index
                total = ai * per_arm + local
                row = fits[total]
                donor = donors[index] if arm == "shuffled_h4" else index
                outcome_source = samples[donor]
                horizon = training["arm_horizons"][arm]
                require((row["arm"], row["fit_step"], row["total_step"], row["epoch"], row["sample_index"], row["donor_index"])
                        == (arm, local + 1, total + 1, epoch + 1, index, donor), "fit order or declared donor mapping differs")
                require(row["feature_source"] == source_identity(source)
                        and row["outcome_source"] == source_identity(outcome_source), "fit source/donor association differs")
                require(row["comparison_feature_hash"] == source["life_hash"], "fit comparison feature binding differs")
                validate_receipt(row, outcome_source, horizon, training["learning_rate"])
                require(row["outcome_sha256"] == outcome_ids(outcome_source, horizon), "fit outcome record association differs")
                require(row["hash_before"] == prior, "fit policy continuation hash differs")
                prior = row["hash_after"]
                require(row["non_policy_state_unchanged"] is True and row["feature_witness_valid"] is True,
                        "fit changed non-policy state or feature witness")
                before_loss += row["loss_before"]; after_loss += row["loss_after"]
            epochs.append({"arm": arm, "epoch": epoch + 1, "mean_online_loss_before": before_loss / len(samples),
                           "mean_online_loss_after": after_loss / len(samples)})
        arm_hashes[arm] = prior
    saved = [row for row in rows if row["type"] == "saved_life"]
    require([row["model"] for row in saved] == list(POLICIES), "saved life coverage differs")
    for row in saved:
        require(row["hash"] == arm_hashes[row["model"]] and row["resume_readouts"] == len(samples)
                and row["same_hash"] is True and row["same_actions_scores"] is True, "saved life/resume differs")
    summaries = [row for row in rows if row["type"] == "arm_summary"]
    require([row["arm"] for row in summaries] == list(POLICIES[1:]), "arm summary coverage differs")
    for row in summaries:
        require(row["fit_steps"] == per_arm and row["hash"] == arm_hashes[row["arm"]]
                and row["non_policy_state_unchanged"] is True, "arm summary differs")
    final = [row for row in rows if row["type"] == "fit_summary"]
    require(len(final) == 1 and rows[-1] == final[0], "complete final fit summary required")
    require(final[0]["fit_steps"] == len(fits) and final[0]["all_lives_saved_resumed"] is True
            and final[0]["lives"] == [{"model": arm, "hash": arm_hashes[arm]} for arm in POLICIES], "fit terminal counters differ")
    return {"fits": len(fits), "samples": len(samples), "epochs_per_arm": training["epochs"],
            "life_hashes": arm_hashes, "per_epoch": epochs,
            "checks": {"complete": True, "source_donor_association": True, "raw_targets_recomputed": True,
                       "all_head_losses_recomputed": True, "policy_only": True, "saved_resume": True}}


def validate_readout(rows: list[dict], samples: list[dict], dataset: Path, archive_sha: str,
                     life_hashes: dict, training_dataset: Path, training_archive_sha: str) -> list[dict]:
    validate_import(rows, samples, dataset, FROZEN_PROTOCOL_SHA256, archive_sha)
    allowed = {"dataset", "import", "read_run", "readout", "read_summary"}
    require(all(row["type"] in allowed for row in rows), "unknown readout record")
    headers = [row for row in rows if row["type"] == "read_run"]
    require(len(headers) == 1 and headers[0] == {"type": "read_run", "fit_dataset_fnv1a": fnv_bytes(training_dataset.read_bytes()),
        "fit_archive_sha256": training_archive_sha, "models": 7, "horizons": [0, 4], "learning_rate": 0}, "native readout recipe differs")
    readouts = [row for row in rows if row["type"] == "readout"]
    require(len(readouts) == len(samples) * len(LABELS) * len(HORIZONS), "incomplete native readout trace")
    by_key = {}
    for row in readouts:
        index, label, horizon = row["source"]["index"], row["label"], row["horizon"]
        require(type(index) is int and 0 <= index < len(samples) and label in LABELS and horizon in HORIZONS, "unregistered readout coordinate")
        key = (index, label, horizon)
        require(key not in by_key, "duplicate native readout")
        by_key[key] = row
        source = samples[index]
        require(row["source"] == source_identity(source), "readout feature-source association differs")
        require(row["model_hash"] == life_hashes[label if label in POLICIES else "initial"], "readout life differs from sealed policy")
        validate_receipt(row, source, horizon, 0)
        require(row["scores_before"] == row["scores_after"], "readout changes predictions")
        require(row["outcome_sha256"] == outcome_ids(source, horizon), "readout outcome identities differ")
        valid = sorted(SCENARIOS.valid_actions(source["target"], source["sentence_count"]))
        requested = {"keep": 0, "left": 1, "right": 2}.get(label)
        if requested is None:
            chosen = max(valid, key=lambda kind: row["scores_before"][kind])
            require(row["forced"] is False and row["boundary_fallback"] is False and row["requested_kind"] == chosen, "learned readout selection differs")
        else:
            chosen = requested if requested in valid else 0
            require(row["forced"] is True and row["requested_kind"] == requested and row["boundary_fallback"] is (requested not in valid), "fixed boundary policy differs")
        outcome = next(o for o in source["outcomes"][horizon] if o["action"]["kind"] == chosen)
        require(row["action"] == outcome["action"] and row["selected_outcome_sha256"] == outcome["raw_sha256"], "selected typed action is not its executed branch")
        for which in ("before", "after"):
            for axis in AXES: near(row[which][axis], outcome[which][axis], "selected raw axis differs", 5e-8)
        near(row["cost"], outcome["cost"], "selected cost differs", 5e-8)
        rewards, targets = native_rewards(source, horizon)
        best = max(rewards[k] for k in valid)
        for field, expected in (("selected_reward", rewards[chosen]), ("keep_reward", rewards[0]),
                                ("advantage_over_keep", targets[chosen]), ("regret", best - rewards[chosen])):
            near(row[field], expected, "selected consequence differs: " + field)
        # The declared optimal flag uses exact C float rewards; all raw scores
        # remain retained, and the report below reapplies the protocol tolerance.
        require(row["optimal"] is (row["regret"] <= 1e-7), "readout optimal flag differs")
        require(row["life_unchanged"] is True and row["features_unchanged"] is True
                and row["feature_witness_valid"] is True, "readout mutated frozen state")
    require(set(by_key) == {(i, label, horizon) for i in range(len(samples)) for label in LABELS for horizon in HORIZONS}, "readout coverage differs")
    for i in range(len(samples)):
        for label in LABELS:
            a, b = by_key[i, label, 0], by_key[i, label, 4]
            require(a["action"] == b["action"] and a["scores_before"] == b["scores_before"], "choice depends on consequence horizon")
    summary = [row for row in rows if row["type"] == "read_summary"]
    require(len(summary) == 1 and rows[-1] == summary[0]
            and summary[0] == {"type": "read_summary", "samples": len(samples), "readouts": len(readouts),
                "sealed_lives_unchanged": True, "source_features_unchanged": True}, "complete readout summary required")
    return readouts


def summarize_readouts(readouts: list[dict], samples: list[dict], split: str, tolerance=1e-7) -> dict:
    choices = {(row["source"]["index"], row["label"]): row["action"]["kind"]
               for row in readouts if row["horizon"] == 0}
    groupings = [("combined", "all", list(range(len(samples))))]
    groupings += [("seed", seed, [i for i, s in enumerate(samples) if s["seed"] == seed]) for seed in sorted({s["seed"] for s in samples})]
    groupings += [("target", target, [i for i, s in enumerate(samples) if s["target"] == target]) for target in range(4)]
    rows = []
    for group, value, indices in groupings:
        for horizon in MEASURED_HORIZONS:
            for label in LABELS:
                reward_sum = advantage_sum = regret_sum = cost_sum = chars_sum = 0.0
                optimal = changes = 0
                counts = dict.fromkeys(SCENARIOS.NAMES, 0)
                axes = {axis: {"before": 0.0, "after": 0.0, "delta": 0.0, "delta_vs_keep": 0.0} for axis in AXES}
                for index in indices:
                    sample = samples[index]
                    outcomes = {row["action"]["kind"]: row for row in sample["outcomes"][horizon]}
                    chosen = choices[index, label]
                    selected, keep = outcomes[chosen], outcomes[0]
                    best = max(row["reward"] for row in outcomes.values())
                    regret = best - selected["reward"]
                    reward_sum += selected["reward"]; advantage_sum += selected["reward"] - keep["reward"]
                    regret_sum += regret; cost_sum += selected["cost"]; chars_sum += selected["cumulative_generated"]
                    optimal += regret <= tolerance
                    changes += chosen != choices[index, "initial"]
                    counts[SCENARIOS.NAMES[chosen]] += 1
                    for axis in AXES:
                        axes[axis]["before"] += selected["before"][axis]
                        axes[axis]["after"] += selected["after"][axis]
                        axes[axis]["delta"] += selected["after"][axis] - selected["before"][axis]
                        axes[axis]["delta_vs_keep"] += selected["after"][axis] - keep["after"][axis]
                n = len(indices)
                rows.append({"split": split, "group": group, "group_value": value, "horizon": horizon, "label": label,
                    "states": n, "mean_selected_reward": reward_sum / n, "mean_advantage_over_keep": advantage_sum / n,
                    "mean_oracle_regret": regret_sum / n, "optimal_count": optimal, "action_counts": counts,
                    "changed_choices_from_initial": changes, "mean_normalized_cost": cost_sum / n,
                    "total_generated_characters": int(chars_sum), "raw_axis_means": {a: {k: v / n for k, v in axis.items()} for a, axis in axes.items()}})
    comparisons = []
    for other in ("initial", "h0", "shuffled_h4", "keep", "left", "right"):
        comparisons.append({"h4_vs": other, "changed_choices": sum(choices[i, "h4"] != choices[i, other] for i in range(len(samples)))})
    return {"split": split, "states": len(samples), "summaries": rows, "h4_choice_comparisons": comparisons}


def run_command(command: list, cwd: Path, output: Path, label: str) -> dict:
    command = list(map(str, command))
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    stdout, stderr = output / f"{label}.stdout.txt", output / f"{label}.stderr.txt"
    stdout.write_text(completed.stdout); stderr.write_text(completed.stderr)
    record = {"command": command, "cwd": str(cwd), "returncode": completed.returncode,
              "wall_seconds": time.monotonic() - started,
              "stdout": {"file": stdout.name, "sha256": V1.digest(stdout), "text": completed.stdout},
              "stderr": {"file": stderr.name, "sha256": V1.digest(stderr), "text": completed.stderr}}
    V1.write_json(output / f"{label}.command.json", record)
    require(completed.returncode == 0, f"{label} exited {completed.returncode}: {completed.stderr}")
    return record


def artifact(path: Path) -> dict:
    return {"sha256": V1.digest(path), "bytes": path.stat().st_size}


def verify_seal(seal: dict, directory: Path) -> None:
    require(seal["protocol_sha256"] == FROZEN_PROTOCOL_SHA256 and seal["status"] == "SEALED", "missing registered policy seal")
    require(set(seal["lives"]) == set(POLICIES), "incomplete sealed life set")
    required = {seal["training_dataset"], seal["fit_trace"], seal["native_seal"]}
    for label in POLICIES:
        life = seal["lives"][label]
        require(life["file"] in seal["files"] and {key: life[key] for key in ("sha256", "bytes")} == seal["files"][life["file"]],
                "sealed life missing from file manifest")
        required.add(life["file"])
    require(set(seal["files"]) == required and len(required) == 7, "sealed training file coverage differs")
    for name, expected in seal["files"].items():
        require(Path(name).name == name and expected["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]), "invalid sealed file identity")
        require(artifact(directory / name) == expected, "sealed training artifact changed: " + name)


def complete_trace(path: Path, seed: int) -> tuple[bytes, dict]:
    receipt, raw = TRACE_IO.inspect(path, seed)
    require(path.read_bytes() == raw, "ordinary trace changed on immediate second parent read")
    return raw, receipt


def complete_scenario(path: Path, ordinary: bytes, seed: int, scenario_protocol: dict) -> tuple[bytes, dict]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    require((before.st_size, before.st_ino) == (after.st_size, after.st_ino) and len(raw) == after.st_size,
            "scenario trace changed during parent read")
    rows, _ = parse_raw(raw); ordinary_rows, _ = parse_raw(ordinary)
    validation_protocol = copy.deepcopy(scenario_protocol); validation_protocol["seeds"] = [seed]
    result = SCENARIOS.validate_seed(rows, ordinary_rows, validation_protocol)
    require(path.read_bytes() == raw, "scenario trace changed on immediate second parent read")
    return raw, {"bytes": len(raw), "sha256": digest_bytes(raw), "snapshots": result["snapshots"],
                 "forks": result["alternatives"], "measurements": result["measurements"],
                 "diagnostic_forwards": result["diagnostic_forwards"], "complete": True}


def write_archive(path: Path, streams: dict[tuple[int, str], bytes]) -> None:
    raw = bytearray()
    for (seed, name), trace in sorted(streams.items()):
        for line in trace.splitlines(keepends=True):
            raw.extend(json.dumps({"stream": name, "seed": seed, "raw": line.decode()}, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    path.write_bytes(gzip.compress(bytes(raw), mtime=0))


def execute_experiment(directory: Path, snapshot: Path, inputs: Path, trainer: Path, body: Path,
                       samples: list[dict], protocol: dict, scenario_protocol: dict, metadata: dict) -> dict:
    directory.mkdir()
    commands, phases, tracked, raw_streams = [], [], {}, {}
    def phase(name, **fields):
        event = {"sequence": len(phases) + 1, "phase": name, **fields}
        phases.append(event)
        V1.write_json(directory / "phases.json", phases)
    phase("training_started", source_archive_sha256=protocol["parents"]["raw_traces.jsonl.gz"]["sha256"])
    old_archive_sha = protocol["parents"]["raw_traces.jsonl.gz"]["sha256"]
    training_table = directory / "training.txt"
    write_dataset(training_table, samples, FROZEN_PROTOCOL_SHA256, old_archive_sha)
    V1.write_json(directory / "training_associations.json", samples)
    prefix = directory / "policy"
    commands.append(run_command([trainer, "train", training_table, prefix], snapshot, directory, "train"))
    fit_path = directory / "policy.fit.jsonl"
    fit_raw = fit_path.read_bytes(); fit_rows, _ = parse_raw(fit_raw)
    fit = validate_fit(fit_rows, samples, training_table, protocol, old_archive_sha)
    require(fit_path.read_bytes() == fit_raw, "complete fit trace changed on immediate second read")
    del fit_rows
    raw_streams[0, "fit"] = fit_raw
    lives = {label: directory / f"policy.{label}.life.bin" for label in POLICIES}
    seal_files = [training_table, fit_path, directory / "policy.seal", *lives.values()]
    seal = {"status": "SEALED", "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "training_dataset": training_table.name, "fit_trace": fit_path.name, "native_seal": "policy.seal",
            "source_files_sha256": metadata["source_files_sha256"], "binaries": metadata["binaries"],
            "training_archive_sha256": old_archive_sha, "life_hashes": fit["life_hashes"],
            "lives": {label: {"file": path.name, **artifact(path)} for label, path in lives.items()},
            "files": {path.name: artifact(path) for path in seal_files}}
    V1.write_json(directory / "sealed_lives.json", seal)
    verify_seal(seal, directory)
    phase("all_four_lives_sealed", seal_sha256=V1.digest(directory / "sealed_lives.json"), fits=fit["fits"])
    training_trace = directory / "training.readout.jsonl"
    commands.append(run_command([trainer, "read", training_table, prefix, training_trace], snapshot, directory, "training-read"))
    training_raw = training_trace.read_bytes(); rows, _ = parse_raw(training_raw)
    training_readouts = validate_readout(rows, samples, training_table, old_archive_sha, fit["life_hashes"], training_table, old_archive_sha)
    training_summary = summarize_readouts(training_readouts, samples, "training", protocol["reporting"]["tie_tolerance"])
    raw_streams[0, "training_readout"] = training_raw
    verify_seal(seal, directory)
    phase("training_readout_complete", readouts=len(training_readouts))
    streams, host_receipts = {}, []
    for seed in protocol["evaluation"]["seeds"]:
        off, on = directory / f"off-s{seed}", directory / f"on-s{seed}"
        scenario_path = directory / f"scenarios-s{seed}.jsonl"
        base = [body, *(inputs / name for name in ("simple.weights", "corpus.tokens.u32", "vocabulary.u32"))]
        for mode, host_prefix in (("off", off), ("on", on)):
            verify_seal(seal, directory)
            phase("evaluation_generation_started", seed=seed, diagnostics=mode, seal_sha256=V1.digest(directory / "sealed_lives.json"))
            command = base + [host_prefix, str(seed)]
            if mode == "on": command += ["--scenarios", scenario_path]
            commands.append(run_command(command, snapshot, directory, f"{mode}-s{seed}"))
            path = Path(str(host_prefix) + ".jsonl")
            raw, receipt = complete_trace(path, seed)
            streams[seed, "ordinary_" + mode] = raw
            TRACE_IO.durable_write(directory / f"{mode}-s{seed}.verified.jsonl.gz", gzip.compress(raw, mtime=0))
            host_receipts.append({"seed": seed, "mode": mode, **receipt})
            verify_seal(seal, directory)
        require(streams[seed, "ordinary_off"] == streams[seed, "ordinary_on"], "new-seed ordinary diagnostics leak")
        for arm in json.loads(V1.PROTOCOL.read_text())["arms"]:
            require(Path(f"{off}.{arm}.life.bin").read_bytes() == Path(f"{on}.{arm}.life.bin").read_bytes(), "new-seed host life diagnostics leak")
        scenario_raw, scenario_receipt = complete_scenario(scenario_path, streams[seed, "ordinary_off"], seed, scenario_protocol)
        streams[seed, "scenarios"] = scenario_raw
        TRACE_IO.durable_write(directory / f"scenarios-s{seed}.verified.jsonl.gz", gzip.compress(scenario_raw, mtime=0))
        host_receipts.append({"seed": seed, "mode": "scenarios", **scenario_receipt})
        phase("paired_evaluation_seed_complete", seed=seed, full_host_and_lives_equal=True)
    evaluation_archive = directory / "evaluation.raw_traces.jsonl.gz"
    write_archive(evaluation_archive, streams)
    evaluation_archive_sha = V1.digest(evaluation_archive)
    evaluation = samples_from_streams(streams, protocol["evaluation"]["seeds"], scenario_protocol)
    evaluation_table = directory / "evaluation.txt"
    write_dataset(evaluation_table, evaluation, FROZEN_PROTOCOL_SHA256, evaluation_archive_sha)
    V1.write_json(directory / "evaluation_associations.json", evaluation)
    evaluation_trace = directory / "evaluation.readout.jsonl"
    verify_seal(seal, directory)
    commands.append(run_command([trainer, "read", evaluation_table, prefix, evaluation_trace], snapshot, directory, "evaluation-read"))
    evaluation_raw = evaluation_trace.read_bytes(); rows, _ = parse_raw(evaluation_raw)
    evaluation_readouts = validate_readout(rows, evaluation, evaluation_table, evaluation_archive_sha, fit["life_hashes"], training_table, old_archive_sha)
    evaluation_summary = summarize_readouts(evaluation_readouts, evaluation, "evaluation", protocol["reporting"]["tie_tolerance"])
    raw_streams[0, "evaluation_readout"] = evaluation_raw
    verify_seal(seal, directory)
    phase("evaluation_readout_complete", readouts=len(evaluation_readouts), all_four_lives_unchanged=True)
    # Reopen every measured stream after training/readout and compare to the
    # complete bytes observed by the same parent immediately after each writer.
    for (seed, mode), raw in streams.items():
        name = f"scenarios-s{seed}.jsonl" if mode == "scenarios" else f"{mode.removeprefix('ordinary_')}-s{seed}.jsonl"
        require((directory / name).read_bytes() == raw, "persisted measured stream changed: " + name)
        mirror = directory / name.replace(".jsonl", ".verified.jsonl.gz")
        require(gzip.decompress(mirror.read_bytes()) == raw, "compressed complete stream changed: " + name)
    for path, raw in ((fit_path, fit_raw), (training_trace, training_raw), (evaluation_trace, evaluation_raw)):
        require(path.read_bytes() == raw, "persisted native learning trace changed: " + path.name)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in ("result.json",): tracked[path.name] = artifact(path)
    phase("later_artifact_recheck_complete", measured_streams=len(streams), native_learning_streams=3)
    tracked["phases.json"] = artifact(directory / "phases.json")
    raw_streams.update(streams)
    archive = directory / "raw_traces.jsonl.gz"
    write_archive(archive, raw_streams)
    tracked[archive.name] = artifact(archive)
    result = {"commands": commands, "phases": phases, "fit": fit, "sealed_lives": seal,
              "host_receipts": host_receipts, "training": training_summary, "evaluation": evaluation_summary,
              "artifacts": tracked, "raw_archive": artifact(archive)}
    V1.write_json(directory / "result.json", result)
    print(json.dumps({"execution": directory.name, "fit_steps": fit["fits"],
        "evaluation_h4": [{key: row[key] for key in ("label", "states", "mean_advantage_over_keep", "mean_oracle_regret", "optimal_count", "action_counts", "changed_choices_from_initial")}
                          for row in evaluation_summary["summaries"] if row["group"] == "combined" and row["horizon"] == 4]}), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != ROOT and ROOT not in output.parents, "experiment output must be outside repository")
    require(not output.exists() or not any(output.iterdir()), "preserve prior experiment: output must be new or empty")
    require(V1.digest(PROTOCOL) == FROZEN_PROTOCOL_SHA256, "future protocol changed after preregistration")
    protocol = json.loads(PROTOCOL.read_text())
    for parent in protocol["parents"].values():
        require(V1.digest(ROOT / parent["path"]) == parent["sha256"], "retained parent identity changed: " + parent["path"])
    source_archive = protocol["parents"]["raw_traces.jsonl.gz"]
    retained = archive_streams(ROOT / source_archive["path"], source_archive["sha256"])
    historical = json.loads((ROOT / protocol["parents"]["receipts.json"]["path"]).read_text())
    for (seed, name), raw in retained.items():
        filename = f"scenarios-s{seed}.jsonl" if name == "scenarios" else f"{name.removeprefix('ordinary_')}-s{seed}.jsonl"
        require(digest_bytes(raw) == historical["artifacts"][filename]["sha256"], "retained raw source identity changed")
    scenario_protocol = json.loads(SCENARIOS.PROTOCOL.read_text())
    samples = samples_from_streams(retained, protocol["training"]["seeds"], scenario_protocol)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"; inputs.mkdir()
    if args.inputs:
        for name in ("simple.weights", "dracula.txt"): shutil.copyfile(args.inputs / name, inputs / name)
    input_identities, _ = V1.prepare(inputs, args.reference, json.loads(V1.PROTOCOL.read_text()))
    source_bytes = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source_hashes = {name: digest_bytes(raw) for name, raw in source_bytes.items()}
    require(all(V1.digest(ROOT / name) == value for name, value in source_hashes.items()), "source changed while taking immutable snapshot")
    snapshot = output / "source_snapshot"
    for name, raw in source_bytes.items():
        path = snapshot / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    commands = []
    trainer, body, core_gate = output / "spa_agent_future", output / "spa_agent_demo", output / "test_spa_agent_future"
    common = ["cc", "-std=c11", "-O2", "-DUSE_SIMD", "-march=native", "-pthread", "-I."]
    for path, source in ((trainer, "examples/spa_agent_future.c"), (body, "examples/spa_agent_demo.c"),
                         (core_gate, "tests/test_spa_agent_future.c")):
        commands.append(run_command(common + [source, "spa_agent.c", "notorch.c", "-lm", "-o", path], snapshot, output, "build-" + path.name))
    commands.append(run_command([core_gate], snapshot, output, "core-gates"))
    commands.append(run_command([sys.executable, snapshot / "tests/test_spa_future.py", "--binary", trainer,
                                 "--json", output / "narrow_gates.json"], snapshot, output, "parser-native-gates"))
    metadata = {"protocol_sha256": FROZEN_PROTOCOL_SHA256, "protocol": protocol,
        "base_commit": V1.git("rev-parse", "HEAD"), "source_status": V1.git("status", "--porcelain"),
        "source_files_sha256": source_hashes, "compiled_from_immutable_snapshot": True,
        "binaries": {path.name: artifact(path) for path in (trainer, body, core_gate)}, "inputs": input_identities,
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "machine": V1.machine(), "narrow_gates": json.loads((output / "narrow_gates.json").read_text()), "commands": commands}
    V1.write_json(output / "metadata.json", metadata)
    executions = [execute_experiment(output / f"execution-{index}", snapshot, inputs, trainer, body, samples,
                                    protocol, scenario_protocol, metadata) for index in (1, 2)]
    comparable = [name for name in executions[0]["artifacts"] if name.endswith((".txt", ".life.bin", ".fit.jsonl", ".readout.jsonl", ".seal"))
                  and not name.endswith((".stdout.txt", ".stderr.txt"))]
    comparable += [name for name in executions[0]["artifacts"] if name.startswith(("off-s", "on-s", "scenarios-s")) and name.endswith((".jsonl", ".gz"))]
    comparable += ["raw_traces.jsonl.gz", "evaluation.raw_traces.jsonl.gz"]
    comparable = sorted(set(comparable))
    for name in comparable:
        require((output / "execution-1" / name).read_bytes() == (output / "execution-2" / name).read_bytes(), "complete experiment repeat differs: " + name)
    require(executions[0]["training"] == executions[1]["training"] and executions[0]["evaluation"] == executions[1]["evaluation"], "repeated experiment summaries differ")
    for execution, result in zip((output / "execution-1", output / "execution-2"), executions):
        verify_seal(result["sealed_lives"], execution)
        for name, expected in result["artifacts"].items(): require(artifact(execution / name) == expected, "later artifact identity changed: " + name)
    require(all(V1.digest(ROOT / name) == value for name, value in source_hashes.items()), "measured source changed during authoritative experiment")
    shutil.copyfile(output / "execution-1/raw_traces.jsonl.gz", output / "raw_traces.jsonl.gz")
    receipts = {"schema": 1, "metadata": metadata, "executions": executions,
                "repeat": {"byte_identical_artifacts": comparable, "count": len(comparable), "all_passed": True},
                "artifacts": {"raw_traces.jsonl.gz": artifact(output / "raw_traces.jsonl.gz")}}
    encoded = json.dumps(receipts, ensure_ascii=False, allow_nan=False).replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    encoded = re.sub(r"/[^\s\"\\]*notorch-spa-future-[^/\s\"\\]*", "<gate_tmp>", encoded)
    V1.write_json(output / "receipts.json", json.loads(encoded))
    print(f"Completed: {output / 'receipts.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
