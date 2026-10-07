#!/usr/bin/env python3
"""Preregistered paired repeated consequences; policy arithmetic stays in SPA/C."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
import statistics
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


FUTURE = module("spa_repeated_future", HERE.parent / "future/run.py")
POLICY = module("spa_repeated_policy", HERE / "policy.py")
SCENARIOS, V1 = FUTURE.SCENARIOS, FUTURE.V1
AXES, HORIZONS = SCENARIOS.AXES, (0, 1, 4)
ARMS = POLICY.ARMS
LABELS = (*ARMS, "keep", "left", "right")
require, f32, parse_raw = FUTURE.require, FUTURE.f32, FUTURE.parse_raw
identity, sha256, fnv = FUTURE.artifact, FUTURE.digest_bytes, FUTURE.fnv_bytes
SOURCE_FILES = tuple(dict.fromkeys((*FUTURE.SOURCE_FILES,
    "spa_binding.c", "spa_binding.h", "python/SPA.py", "python/notorch.py",
    "examples/spa_agent_replicates.c", "tests/spa_replicates_fixture.c",
    "tests/test_spa_agent_repeated.c", "tests/test_spa_replicates.py",
    "experiments/spa_agent/replicates/run.py", "experiments/spa_agent/replicates/policy.py",
    "experiments/spa_agent/replicates/protocol.json")))


def snapshot_witness(sample):
    raw = struct.pack("<5I3QI", sample["seed"], sample["episode"], sample["step"],
        sample["target"], sample["sentence_count"], int(sample["life_hash"], 16),
        int(sample["body_hash"], 16), int(sample["host_rng"], 16), sample["agent_rng"])
    raw += struct.pack("<29f7f4I", *sample["features"],
                       *(sample["before"][axis] for axis in AXES), *sample["reseeds"])
    for sentence in sample["chain"]:
        raw += struct.pack("<" + "I" * (2 + sentence["length"]), sentence["length"],
                           int(sentence["terminated"]), *sentence["tokens"])
    return fnv(raw)


def source_samples(streams, seeds, archive_sha, scenario_protocol):
    samples = FUTURE.samples_from_streams(streams, seeds, scenario_protocol)
    snapshots = {}
    references = {}
    for seed in seeds:
        raw = streams[seed, "scenarios"]
        rows, _ = parse_raw(raw)
        for row, line in zip(rows, raw.splitlines(keepends=True)):
            if row["type"] == "snapshot": snapshots[seed, row["snapshot"]] = row
            if row["type"] == "measurement":
                references[seed, row["snapshot"], row["action"]["kind"], row["horizon"]] = line
    for sample in samples:
        snap = snapshots[sample["seed"], sample["snapshot"]]
        sample.update({key: copy.deepcopy(snap[key]) for key in ("before", "chain", "reseeds")})
        sample["source_archive_sha256"] = archive_sha
        sample["snapshot_witness"] = snapshot_witness(sample)
        sample["outcomes"] = {str(key): value for key, value in sample["outcomes"].items()}
    return samples, references


def source_from_row(row):
    """Fixture adapter: trusted native fixture exports its imported source record."""
    sample = {key: copy.deepcopy(row[key]) for key in ("seed", "episode", "step", "snapshot",
        "target", "sentence_count", "life_hash", "body_hash", "agent_rng", "features",
        "source_archive_sha256", "feature_witness", "snapshot_witness", "before", "chain", "reseeds")}
    sample.update(ordinal=row["source_index"], host_rng=row["host_rng_before"],
        snapshot_raw_sha256=row["snapshot_sha256"], ordinary_decision_raw_sha256="0" * 64,
        action_mask=FUTURE.action_mask(row["target"], row["sentence_count"]))
    return sample


def write_sources(path, samples, vocabulary_fnv, protocol_sha=FROZEN_PROTOCOL_SHA256):
    lines = ["NT_SPA_REPLICATES_V1", "PROTOCOL " + protocol_sha,
             "VOCAB_FNV1A " + vocabulary_fnv, "COUNT " + str(len(samples))]
    floats = lambda xs: " ".join(format(f32(x), ".9g") for x in xs)
    for sample in samples:
        require(sample["feature_witness"] == FUTURE.feature_witness(sample), "source feature witness changed")
        require(sample["snapshot_witness"] == snapshot_witness(sample), "source snapshot witness changed")
        fields = [sample[key] for key in ("ordinal", "seed", "episode", "step", "target", "sentence_count",
            "life_hash", "body_hash", "agent_rng", "host_rng", "source_archive_sha256", "snapshot_raw_sha256")]
        lines += ["SNAPSHOT " + " ".join(map(str, fields)), "FEATURES " + floats(sample["features"]),
                  "BEFORE " + floats(sample["before"][a] for a in AXES),
                  "RESEEDS " + " ".join(map(str, sample["reseeds"]))]
        for index, sentence in enumerate(sample["chain"]):
            lines.append("SENTENCE " + " ".join(map(str, [index, sentence["length"],
                int(sentence["terminated"]), *sentence["tokens"]])))
        lines += ["FEATURE_WITNESS " + sample["feature_witness"],
                  "SNAPSHOT_WITNESS " + sample["snapshot_witness"], "END_SNAPSHOT"]
    lines.append("END")
    with Path(path).open("x") as stream: stream.write("\n".join(lines) + "\n")


def rng_start(sample, replicate, hop):
    if replicate == 0:
        return SCENARIOS.stream(sample["seed"], 0x7370615F6163746E if hop == 0 else 0x7370615F66757472,
            sample["episode"] if hop == 0 else sample["snapshot"], sample["step"] if hop == 0 else hop)
    return SCENARIOS.stream(sample["seed"], 0x7370615F72696E69 if hop == 0 else 0x7370615F72667574,
                            sample["snapshot"] * 8 + replicate, hop)


def outcome(row, raw_sha):
    return {"action": row["action"], "before": row["before"], "after": row["after"],
        "cost": row["cost_normalized"], "reward": row["reward"], "raw_sha256": raw_sha,
        "initial_generated": row["initial_generated"], "cumulative_generated": row["cumulative_generated"],
        "chain": row["chain"]}


def validate_measurement(row, sample, replicate, kind, horizon, earlier):
    for key in ("seed", "episode", "step", "snapshot", "target", "life_hash", "body_hash", "agent_rng"):
        require(row[key] == sample[key], "replicate source association: " + key)
    require(row["horizon"] == horizon, "replicate horizon coverage/order")
    initial_rng = rng_start(sample, replicate, 0)
    # The unchanged writer records this branch's initial RNG as host_rng_before.
    require(int(row["host_rng_before"], 16) == initial_rng, "replicate host RNG association")
    require(row["before"] == sample["before"], "replicate common-before metrics")
    SCENARIOS.check_action(row["action"], sample["target"], 4)
    require(row["action"]["kind"] == kind, "replicate action coverage/order")
    SCENARIOS.check_axes(row["after"]); SCENARIOS.check_chain(row["chain"], 4, 64)
    FUTURE.near(row["after"]["repetition"], SCENARIOS.repetition(row["chain"][sample["target"]]),
                "replicate repetition metric uses wrong target", 5e-8)
    chars = row["initial_generated"]
    require(type(chars) is int and ((chars == 0) if kind == 0 else 12 <= chars <= 64), "replicate initial character count")
    require(row["initial_cost"] == chars / 64, "replicate initial raw cost")
    require(int(row["initial_rng_before"], 16) == initial_rng, "replicate initial RNG pairing")
    require(int(row["initial_rng_after"], 16) == (initial_rng + chars * SCENARIOS.INCREMENT) & SCENARIOS.MASK64,
            "replicate initial RNG consumption")
    require(len(row["continuation"]) == horizon, "replicate future horizon count")
    if earlier:
        require(row["continuation"][:len(earlier["continuation"])] == earlier["continuation"], "replicate branch continuation changed")
        for key in ("initial_rng_before", "initial_rng_after", "initial_generated", "initial_cost"):
            require(row[key] == earlier[key], "replicate initial action changed between horizons")
        rewritten = {future["action"]["target"] for future in row["continuation"][len(earlier["continuation"]):]}
        for index in range(4):
            if index not in rewritten:
                require(row["chain"][index] == earlier["chain"][index], "replicate untouched future sentence changed")
    counters = sample["reseeds"].copy(); counters[sample["target"]] += kind != 0
    total = chars
    for hop, future in enumerate(row["continuation"], 1):
        target = (sample["target"] + hop) % 4
        require(future["hop"] == hop, "replicate future hop order")
        SCENARIOS.check_action(future["action"], target, 4)
        require(future["action"]["kind"] == (1 if target else 2), "replicate frozen continuation action")
        start = rng_start(sample, replicate, hop)
        require(int(future["rng_before"], 16) == start, "replicate future RNG pairing")
        amount = future["generated"]
        require(type(amount) is int and 12 <= amount <= 64 and future["cost"] == amount / 64, "replicate future raw cost")
        require(int(future["rng_after"], 16) == (start + amount * SCENARIOS.INCREMENT) & SCENARIOS.MASK64,
                "replicate future RNG consumption")
        total += amount; counters[target] += 1
    last_lengths = {sample["target"]: chars} if kind != 0 else {}
    for future in row["continuation"]: last_lengths[future["action"]["target"]] = future["generated"]
    for index, length in last_lengths.items():
        require(row["chain"][index]["length"] == length, "replicate future token accounting")
    require(row["cumulative_generated"] == row["branch_forwards"] == total, "replicate unaccounted forwards")
    require(row["cost_denominator"] == 64 * (horizon + 1), "replicate cost denominator")
    FUTURE.near(row["cost_normalized"], total / (64 * (horizon + 1)), "replicate normalized cost", 5e-8)
    require(row["reseeds"] == counters, "replicate reseed history")
    FUTURE.near(row["reward"], SCENARIOS.reward(row["before"], row["after"], row["cost_normalized"]),
                "replicate reward sign/raw axes", 2e-7)
    if horizon == 0:
        for index in range(4):
            if index != sample["target"] or kind == 0:
                require(row["chain"][index] == sample["chain"][index], "replicate initial chain leak")
        if kind == 0:
            require(row["before"] == row["after"] and row["reward"] == 0, "replicate KEEP changed metrics")
        else:
            require(row["chain"][sample["target"]]["length"] == chars, "replicate initial token accounting")


def validate_replicates(raw, samples, references, *, dataset_fnv=None, vocabulary_fnv=None,
                        protocol_sha=FROZEN_PROTOCOL_SHA256):
    rows, hashes = parse_raw(raw); lines = raw.splitlines(keepends=True)
    cursor = 0
    def take(kind):
        nonlocal cursor
        require(cursor < len(rows) and rows[cursor].get("type") == kind,
                "replicate incomplete/reordered stream: expected " + kind)
        row, hashed, line = rows[cursor], hashes[cursor], lines[cursor]
        cursor += 1
        return row, hashed, line
    header, _, _ = take("replicate_run")
    require(header["protocol_sha256"] == protocol_sha and header["sources"] == len(samples) and
            header["replicates_per_source"] == 8 and header["horizons"] == list(HORIZONS), "replicate header/config")
    if dataset_fnv is not None: require(header["dataset_fnv1a"] == dataset_fnv, "replicate imported dataset identity")
    if vocabulary_fnv is not None: require(header["vocabulary_fnv1a"] == vocabulary_fnv, "replicate vocabulary identity")
    total_alternatives = total_forwards = total_measurements = 0
    projected = []
    for sample in samples:
        row, _, _ = take("source")
        expected = source_from_row(row)
        for key in expected:
            if key != "ordinary_decision_raw_sha256": require(expected[key] == sample[key], "replicate imported source changed: " + key)
        require(row["sensory_features_checked"] == 19 and row["metrics_match_field"] is True, "replicate native perception receipt")
        require(header["body_hash"] == sample["body_hash"], "replicate body identity")
        require(sample["snapshot_witness"] == snapshot_witness(sample) and
                sample["feature_witness"] == FUTURE.feature_witness(sample), "replicate source witness")
        kinds = sorted(SCENARIOS.valid_actions(sample["target"], 4))
        repeated = []
        def context(item, replica):
            for key in ("seed", "episode", "step", "snapshot"):
                require(item[key] == sample[key], "replicate bracket source association")
            require(item["source_index"] == sample["ordinal"] and item["replicate"] == replica,
                    "replicate bracket index/count/order")
        for replica in range(8):
            begin, _, _ = take("replicate_begin"); context(begin, replica)
            for key in ("feature_witness", "snapshot_witness", "life_hash", "agent_rng"):
                require(begin[key] == sample[key], "replicate beginning source witness")
            require(begin["rng_starts"] == [f"{rng_start(sample, replica, hop):016x}" for hop in range(5)],
                    "replicate paired RNG starts")
            projected_outcomes = {str(h): [] for h in HORIZONS}
            forwards = 0
            for kind in kinds:
                earlier = None
                for horizon in HORIZONS:
                    measured, hashed, line = take("measurement")
                    validate_measurement(measured, sample, replica, kind, horizon, earlier)
                    if replica == 0:
                        require(line == references[sample["seed"], sample["snapshot"], kind, horizon],
                                "replicate r0 parent raw-line parity")
                    projected_outcomes[str(horizon)].append(outcome(measured, hashed))
                    earlier = measured; total_measurements += 1
                restored, _, _ = take("replicate_restoration"); context(restored, replica)
                require(restored["action"] == earlier["action"], "replicate restoration action")
                for key in ("source_unchanged", "features_unchanged", "rng_witness_unchanged", "body_unchanged",
                            "forwards_restored", "training_mode_restored", "tape_empty"):
                    require(restored[key] is True, "replicate state leak: " + key)
                require(restored["body_hash_before"] == restored["body_hash_after"] == sample["body_hash"], "replicate restored body")
                require(restored["snapshot_witness_before"] == restored["snapshot_witness_after"] == sample["snapshot_witness"],
                        "replicate restored snapshot")
                require(restored["host_forwards"] == 0 and restored["diagnostic_forwards"] == earlier["cumulative_generated"],
                        "replicate restored forward accounting")
                forwards += earlier["cumulative_generated"]; total_alternatives += 1
            end, _, _ = take("replicate_end"); context(end, replica)
            require(end["alternatives"] == len(kinds) and end["measurements"] == len(kinds) * 3 and
                    end["diagnostic_forwards"] == forwards and end["source_unchanged"] is True, "replicate terminal accounting")
            total_forwards += forwards
            repeated.append({"replicate": replica, "outcomes": projected_outcomes})
        value = copy.deepcopy(sample)
        if "outcomes" in value: require(value["outcomes"] == repeated[0]["outcomes"], "replicate projected r0 source association")
        value["outcomes"] = repeated[0]["outcomes"]; value["repeats"] = repeated; projected.append(value)
    summary, _, _ = take("replicate_summary")
    require(cursor == len(rows), "replicate trailing records")
    require(summary["sources"] == len(samples) and summary["replicates"] == len(samples) * 8 and
            summary["alternatives"] == total_alternatives and summary["measurements"] == total_measurements and
            summary["diagnostic_forwards"] == total_forwards, "replicate complete summary accounting")
    require(summary["source_fields_unchanged"] is True and summary["body_unchanged"] is True and
            summary["body_hash"] == header["body_hash"] and summary["host_forwards"] == 0, "replicate final state leak")
    return projected, {"sources": len(samples), "replicates": len(samples) * 8,
        "alternatives": total_alternatives, "measurements": total_measurements,
        "diagnostic_forwards": total_forwards, "r0_raw_line_parity": True,
        "paired_rng_and_consumption": True, "raw_reward_and_cost": True, "complete": True}


def native_reward(value):
    total = 0.0
    for axis, weight in zip(AXES, SCENARIOS.WEIGHTS):
        delta = f32(f32(value["after"][axis]) - f32(value["before"][axis]))
        total += f32(weight) * delta
    total -= f32(.05) * f32(value["cost"])
    return min(1.0, max(-1.0, f32(total)))


def repeated_targets(sample, count=8, horizon=4):
    rewards = [0.0] * 3
    for repeat in sample["repeats"][:count]:
        for value in repeat["outcomes"][str(horizon)]: rewards[value["action"]["kind"]] += native_reward(value)
    rewards = [f32(value / count) for value in rewards]
    targets = [f32(rewards[k] - rewards[0]) if sample["action_mask"] & (1 << k) else 0.0 for k in range(3)]
    return rewards, targets


def validate_receipt(receipt, sample, donor, count, rate):
    require(receipt["source_life_hash"] == sample["life_hash"] and receipt["horizon"] == 4 and
            receipt["action_mask"] == sample["action_mask"], "repeated receipt source/horizon/mask")
    FUTURE.near(receipt["learning_rate"], rate, "repeated receipt rate", 2e-9)
    rewards, targets = repeated_targets(donor, count)
    for name, values in (("rewards", rewards), ("targets", targets)):
        require(len(receipt[name]) == 3, "repeated receipt vector dimensions")
        for actual, expected in zip(receipt[name], values):
            FUTURE.near(actual, expected, "repeated individually-clipped reward/target: " + name, 8e-8)
    for side in ("before", "after"):
        FUTURE.near(receipt["loss_" + side], FUTURE.huber(receipt["scores_" + side], receipt["targets"], sample["action_mask"]),
                    "repeated native mean-Huber loss", 2e-10)


def validate_fit(rows, samples, *, epochs=512, protocol_sha=FROZEN_PROTOCOL_SHA256):
    require(len(rows) == 1 + 3 * len(samples) + 3 * epochs * len(samples), "repeated fit count")
    header = rows[0]
    require(header["type"] == "header" and header["protocol_sha256"] == protocol_sha and
            header["epochs"] == epochs and header["rate"] == .03 and header["arms"] == list(ARMS), "repeated fit header")
    donors = FUTURE.shuffle_donors(samples); cursor = 1
    comparisons = {}
    for arm in ARMS[1:]:
        for index, sample in enumerate(samples):
            donor_index = donors[index] if arm == "shuffled_mean8" else index
            donor, count = samples[donor_index], 1 if arm == "single_h4" else 8
            row = rows[cursor]; cursor += 1
            require(row["type"] == "comparison" and row["arm"] == arm and row["comparison_id"] == f"{arm}:{index}", "repeated comparison order")
            require(row["source"] == POLICY.source_identity(sample) and row["donor"] == POLICY.source_identity(donor) and
                    row["source_features"] == [f32(x) for x in sample["features"]], "repeated feature/outcome source association")
            values, refs = POLICY.comparisons(sample, donor, count)
            require(row["repeats"] == refs and row["native_input_hex"] == [bytes(v).hex() for v in values],
                    "repeated raw/native consequence association")
            validate_receipt(row["initial_evaluation"], sample, donor, count, 0)
            comparisons[arm, index] = (donor_index, count)
    losses = {}
    for arm in ARMS[1:]:
        first = last = 0.0
        for epoch in range(epochs):
            for index, sample in enumerate(samples):
                row = rows[cursor]; cursor += 1
                donor_index, count = comparisons[arm, index]
                require(row["type"] == "fit" and (row["arm"], row["epoch"], row["row"]) == (arm, epoch, index), "repeated fit update order")
                require(row["comparison_id"] == f"{arm}:{index}" and row["source_feature_witness"] == sample["feature_witness"] and
                        row["donor_row"] == donor_index and row["repeat_count"] == count, "repeated fit association/budget")
                validate_receipt(row["receipt"], sample, samples[donor_index], count, .03)
                native = POLICY.SPA.ComparisonReceipt.from_buffer_copy(bytes.fromhex(row["receipt_hex"]))
                require(POLICY.native_receipt(native) == row["receipt"], "repeated canonical receipt bytes")
                if epoch == 0: first += row["receipt"]["loss_before"]
                if epoch == epochs - 1: last += row["receipt"]["loss_after"]
        losses[arm] = {"first_epoch_mean_pre_loss": first / len(samples), "final_epoch_mean_post_loss": last / len(samples)}
    return {"fits": 3 * epochs * len(samples), "comparisons": 3 * len(samples), "epochs": epochs,
        "raw_reward_targets_recalculated": True, "native_receipt_bytes": True, "losses": losses}


def validate_readouts(rows, samples, life_hashes):
    require(len(rows) == len(samples) * 4, "repeated readout count")
    choices = {}
    for sample in samples:
        for arm in ARMS:
            row = rows[len(choices)]
            require(row["type"] == "readout" and row["arm"] == arm and row["source"] == POLICY.source_identity(sample), "repeated readout source order")
            require(row["life_hash"] == life_hashes[arm] and row["action_mask"] == sample["action_mask"], "repeated readout life/mask")
            SCENARIOS.check_action(row["action"], sample["target"], 4)
            kinds = sorted(SCENARIOS.valid_actions(sample["target"], 4))
            best = max(kinds, key=lambda kind: row["scores"][kind])
            require(row["action"]["kind"] == best, "repeated readout masked KEEP-first greedy")
            native = POLICY.SPA.Readout.from_buffer_copy(bytes.fromhex(row["readout_hex"]))
            require(list(native.scores) == row["scores"] and native.action_mask == row["action_mask"] and
                    (native.action.kind, native.action.target, native.action.source) ==
                    (row["action"]["kind"], row["action"]["target"], POLICY.SPA.NO_SOURCE if row["action"]["source"] is None else row["action"]["source"]),
                    "repeated readout canonical bytes")
            choices[sample["ordinal"], arm] = row["action"]["kind"]
    return choices


def mean(values): return sum(values) / len(values)


def variance(values): return statistics.variance(values) if len(values) > 1 else 0.0


def summarize(samples, choices, split):
    state_rows = []
    for sample in samples:
        index = sample["ordinal"]; available = sorted(SCENARIOS.valid_actions(sample["target"], 4))
        kinds = {label: choices[index, label] for label in ARMS}
        kinds.update(keep=0, left=1 if 1 in available else 0, right=2 if 2 in available else 0)
        for horizon in HORIZONS:
            repeats = [{o["action"]["kind"]: o for o in r["outcomes"][str(horizon)]} for r in sample["repeats"]]
            rewards = {kind: [native_reward(r[kind]) for r in repeats] for kind in available}
            paired = {kind: [rewards[kind][r] - rewards[0][r] for r in range(8)] for kind in available}
            means = {kind: mean(rewards[kind]) for kind in available}; oracle = max(means.values())
            actions = [{"kind": kind, "mean_reward": means[kind], "mean_advantage_over_keep": mean(paired[kind]),
                "paired_variance": variance(paired[kind]), "paired_standard_error": math.sqrt(variance(paired[kind]) / 8),
                "positive_replicates": sum(x > 1e-7 for x in paired[kind]), "negative_replicates": sum(x < -1e-7 for x in paired[kind]),
                "sign_disagreement": any(x > 1e-7 for x in paired[kind]) and any(x < -1e-7 for x in paired[kind]),
                "replicate_advantages": paired[kind]} for kind in available]
            for label, kind in kinds.items():
                selected = [r[kind] for r in repeats]
                state_rows.append({"split": split, "ordinal": index, "seed": sample["seed"], "snapshot": sample["snapshot"],
                    "target": sample["target"], "horizon": horizon, "label": label, "kind": kind,
                    "source_snapshot_sha256": sample["snapshot_raw_sha256"],
                    "outcome_sha256": [o["raw_sha256"] for o in selected],
                    "mean_reward": means[kind], "mean_advantage_over_keep": mean(paired[kind]),
                    "r0_advantage": paired[kind][0], "r1_to_r7_advantage": mean(paired[kind][1:]),
                    "oracle_regret": oracle - means[kind], "optimal": oracle - means[kind] <= 1e-7,
                    "changed_choice_from_initial": kind != kinds["initial"],
                    "paired_variance": variance(paired[kind]), "paired_standard_error": math.sqrt(variance(paired[kind]) / 8),
                    "mean_cost": mean([o["cost"] for o in selected]),
                    "mean_cost_vs_keep": mean([r[kind]["cost"] - r[0]["cost"] for r in repeats]),
                    "mean_generated": mean([o["cumulative_generated"] for o in selected]),
                    "mean_after": {axis: mean([o["after"][axis] for o in selected]) for axis in AXES},
                    "mean_axis_delta_vs_keep": {axis: mean([r[kind]["after"][axis] - r[0]["after"][axis] for r in repeats]) for axis in AXES},
                    "actions": actions})
    summaries = []
    groups = [("combined", lambda r: True)]
    groups += [("seed:" + str(seed), lambda r, seed=seed: r["seed"] == seed) for seed in sorted({s["seed"] for s in samples})]
    groups += [("target:" + str(target), lambda r, target=target: r["target"] == target) for target in range(4)]
    for name, included in groups:
        for horizon in HORIZONS:
            for label in LABELS:
                rows = [r for r in state_rows if r["horizon"] == horizon and r["label"] == label and included(r)]
                if not rows: continue
                summaries.append({"group": name, "horizon": horizon, "label": label, "states": len(rows),
                    **{key: mean([r[key] for r in rows]) for key in ("mean_reward", "mean_advantage_over_keep", "r0_advantage",
                        "r1_to_r7_advantage", "paired_variance", "paired_standard_error", "mean_cost", "mean_cost_vs_keep", "mean_generated")},
                    "mean_oracle_regret": mean([r["oracle_regret"] for r in rows]),
                    "optimal_count": sum(r["optimal"] for r in rows), "changed_choices_from_initial": sum(r["changed_choice_from_initial"] for r in rows),
                    "action_counts": {str(k): sum(r["kind"] == k for r in rows) for k in range(3)},
                    "between_state_advantage_variance": variance([r["mean_advantage_over_keep"] for r in rows]),
                    "mean_after": {axis: mean([r["mean_after"][axis] for r in rows]) for axis in AXES},
                    "mean_axis_delta_vs_keep": {axis: mean([r["mean_axis_delta_vs_keep"][axis] for r in rows]) for axis in AXES}})
    return {"split": split, "states": len(samples), "replicates_per_state": 8, "tie_tolerance": 1e-7,
            "reward_arithmetic": "Independently reconstructed clipped native float32 rewards; paired means and sample variance in double precision.",
            "sample_unit": "source state; repeats are nested outcomes", "summaries": summaries, "state_comparisons": state_rows}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def complete_bytes(path):
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    require((before.st_ino, before.st_size) == (after.st_ino, after.st_size) and len(raw) == after.st_size,
            "replicate trace changed during parent read")
    require(Path(path).read_bytes() == raw, "replicate trace changed on second parent read")
    return raw


def run_command(command, cwd, output, label):
    return FUTURE.run_command(command, cwd, output, label)


def verify_barrier(barrier, output, native):
    require(barrier["protocol_sha256"] == FROZEN_PROTOCOL_SHA256 and barrier["status"] == "ALL_EIGHT_LIVES_SEALED",
            "evaluation requires both completed training seals")
    require(set(barrier["executions"]) == {"execution-1", "execution-2"}, "incomplete training execution barrier")
    for name, entry in barrier["executions"].items():
        directory = output / name
        require(set(entry["files"]) == {"training.json", "policies/fit.jsonl", "policies/seal.json",
                *("policies/" + arm + ".life" for arm in ARMS)}, "training barrier coverage")
        for filename, expected in entry["files"].items():
            require(identity(directory / filename) == expected, "sealed training barrier artifact changed: " + filename)
        seal, _ = POLICY.verify_seal(directory / "policies", native, FROZEN_PROTOCOL_SHA256)
        require(seal["lives"]["single_h4"]["sha256"] == POLICY.PARENT_H4_SHA256, "count-one parent control changed")


def authenticate_parents(protocol):
    identities = {}
    for name, entry in protocol["parents"].items():
        path = ROOT / entry["path"]
        if path.exists(): raw = path.read_bytes()
        elif name == "raw_traces.jsonl.gz":
            manifest = json.loads(path.with_name("raw_manifest.json").read_text())
            pieces = []
            for part in manifest["parts"]:
                require(Path(part["path"]).name == part["path"], "unsafe parent archive part")
                piece = path.with_name(part["path"]).read_bytes()
                require(len(piece) == part["bytes"] and sha256(piece) == part["sha256"], "parent archive part identity")
                pieces.append(piece)
            raw = b"".join(pieces)
            require(len(raw) == manifest["archive"]["bytes"] and sha256(raw) == manifest["archive"]["sha256"], "parent archive assembly")
        else: raise AssertionError("missing retained parent: " + str(path))
        require(sha256(raw) == entry["sha256"], "retained parent identity changed: " + name)
        identities[name] = {"sha256": sha256(raw), "bytes": len(raw)}
    for name, entry in protocol["diagnosis"].items():
        require(V1.digest(ROOT / entry["path"]) == entry["sha256"], "exposed diagnosis changed: " + name)
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Previously authenticated public simple.weights and dracula.txt directory")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != ROOT and ROOT not in output.parents, "experiment output must be outside repository")
    require(not output.exists(), "preserve prior experiment: output must be new")
    require(V1.digest(PROTOCOL) == FROZEN_PROTOCOL_SHA256, "replicate protocol changed after preregistration")
    protocol = json.loads(PROTOCOL.read_text()); parents = authenticate_parents(protocol)
    old_archive = ROOT / "experiments/spa_agent/scenarios/raw_traces.jsonl.gz"
    old_sha = "dcee2f733f8a7a4063a44f0654b574dad55ce6b57ea258ee81a7582d39370d22"
    retained = FUTURE.archive_streams(old_archive, old_sha)
    scenario_protocol = json.loads(SCENARIOS.PROTOCOL.read_text())
    historical = json.loads((ROOT / "experiments/spa_agent/scenarios/receipts.json").read_text())
    for (seed, name), raw in retained.items():
        filename = f"scenarios-s{seed}.jsonl" if name == "scenarios" else f"{name.removeprefix('ordinary_')}-s{seed}.jsonl"
        require(sha256(raw) == historical["artifacts"][filename]["sha256"], "retained scenario raw source identity changed")
    training_sources, training_refs = source_samples(retained, protocol["training"]["seeds"], old_sha, scenario_protocol)
    output.mkdir(); inputs = output / "inputs"; inputs.mkdir()
    for name in ("simple.weights", "dracula.txt"): shutil.copyfile(args.inputs / name, inputs / name)
    input_ids, _ = V1.prepare(inputs, None, json.loads(V1.PROTOCOL.read_text()))
    vocab_fnv = fnv((inputs / "vocabulary.u32").read_bytes())
    source_bytes = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source_hashes = {name: sha256(raw) for name, raw in source_bytes.items()}
    snapshot = output / "source_snapshot"
    for name, raw in source_bytes.items():
        path = snapshot / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    binary_ids = {}
    def stable():
        for name, expected in source_hashes.items():
            require(V1.digest(snapshot / name) == expected and V1.digest(ROOT / name) == expected,
                    "measured source changed: " + name)
        for name, expected in binary_ids.items():
            require(identity(output / name) == expected, "measured binary changed: " + name)
    stable()
    commands, stages = [], []
    def phase(name, **fields):
        stages.append({"order": len(stages) + 1, "phase": name, **fields})
        V1.write_json(output / "stages.json", stages)
        print(json.dumps(stages[-1], sort_keys=True), flush=True)
    common = ["cc", "-std=c11", "-O2", "-DUSE_SIMD", "-march=native", "-pthread", "-I."]
    host, ordinary, gate, library = (output / name for name in
        ("spa_agent_replicates", "spa_agent_demo", "test_spa_agent_repeated", "libnotorch.so"))
    for binary, source in ((host, "examples/spa_agent_replicates.c"),
                           (ordinary, "examples/spa_agent_demo.c"), (gate, "tests/test_spa_agent_repeated.c")):
        commands.append(run_command(common + [source, "spa_agent.c", "notorch.c", "-lm", "-o", binary], snapshot, output, "build-" + binary.name))
    commands.append(run_command(common + ["-fPIC", "-shared", "spa_binding.c", "spa_agent.c", "notorch.c", "-lm", "-o", library], snapshot, output, "build-shared"))
    binary_ids.update({path.name: identity(path) for path in (host, ordinary, gate, library)})
    commands.append(run_command([gate], snapshot, output, "core-gates"))
    commands.append(run_command([sys.executable, snapshot / "tests/test_spa_replicates.py", "--library", library,
                                 "--json", output / "narrow_gates.json"], snapshot, output, "host-policy-gates"))
    stable()
    metadata = {"protocol_sha256": FROZEN_PROTOCOL_SHA256, "protocol": protocol, "parents": parents,
        "training_source_archive": identity(old_archive), "inputs": input_ids,
        "source_files_sha256": source_hashes, "compiled_from_immutable_snapshot": True,
        "binaries": {path.name: identity(path) for path in (host, ordinary, gate, library)},
        "base_commit": V1.git("rev-parse", "HEAD"), "source_status": V1.git("status", "--porcelain"),
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "machine": V1.machine(), "narrow_gates": json.loads((output / "narrow_gates.json").read_text()),
        "commands": commands}
    V1.write_json(output / "metadata.json", metadata)
    executions = [output / f"execution-{i}" for i in (1, 2)]
    for directory in executions: directory.mkdir()
    results = {directory.name: {"commands": [], "replicate_gates": {}, "host_gates": []} for directory in executions}
    native = POLICY.SPA.Native(library)
    def parallel(jobs):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(function, *values) for function, values in jobs]
            return [future.result() for future in futures]
    def replicate_job(directory, split, seed, sources, references):
        chunk = [s for s in sources if s["seed"] == seed]
        data = directory / f"{split}-sources-s{seed}.txt"
        trace = directory / f"{split}-repeats-s{seed}.jsonl"
        write_sources(data, chunk, vocab_fnv)
        receipt = run_command([host, inputs / "simple.weights", inputs / "vocabulary.u32", data, trace],
                              snapshot, directory, f"{split}-repeats-s{seed}")
        raw = complete_bytes(trace)
        projected, checked = validate_replicates(raw, chunk, references, dataset_fnv=fnv(data.read_bytes()), vocabulary_fnv=vocab_fnv)
        return directory.name, seed, projected, checked, receipt
    def collect_repeats(split, cohorts):
        jobs = [(replicate_job, (directory, split, seed, *cohorts[directory.name])) for directory in executions
                for seed in (protocol["training"]["seeds"] if split == "training" else protocol["evaluation"]["seeds"])]
        all_samples = {directory.name: [] for directory in executions}
        for execution, seed, projected, checked, command in parallel(jobs):
            all_samples[execution].extend(projected)
            results[execution]["replicate_gates"][f"{split}:{seed}"] = checked
            results[execution]["commands"].append(command)
        for directory in executions:
            samples = sorted(all_samples[directory.name], key=lambda sample: sample["ordinal"])
            require(len(samples) == 48, "full repeated cohort missing")
            write_json(directory / f"{split}.json", {"schema": 1, "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "replicates": 8, "samples": samples})
            all_samples[directory.name] = samples
        return all_samples
    phase("training_repeat_generation_started", independent_processes=4, source_sha256=source_hashes)
    cohorts = {directory.name: (training_sources, training_refs) for directory in executions}
    training = collect_repeats("training", cohorts); stable()
    phase("training_repeats_complete", measurements=2 * 48 * 8 * 7.5)
    helper = snapshot / "experiments/spa_agent/replicates/policy.py"
    policy_args = ["--protocol", snapshot / "experiments/spa_agent/replicates/protocol.json", "--library", library]
    for directory in executions:
        command = run_command([sys.executable, helper, "train", "--dataset", directory / "training.json",
                              *policy_args, "--output", directory / "policies"], snapshot, directory, "policy-train")
        results[directory.name]["commands"].append(command)
        fit_raw = complete_bytes(directory / "policies/fit.jsonl")
        rows, _ = parse_raw(fit_raw)
        results[directory.name]["fit"] = validate_fit(rows, training[directory.name]); del rows
        POLICY.verify_seal(directory / "policies", native, FROZEN_PROTOCOL_SHA256)
        stable()
    barrier = {"status": "ALL_EIGHT_LIVES_SEALED", "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "source_sha256": source_hashes, "binaries": metadata["binaries"], "executions": {}}
    for directory in executions:
        names = ["training.json", "policies/fit.jsonl", "policies/seal.json", *("policies/" + arm + ".life" for arm in ARMS)]
        barrier["executions"][directory.name] = {"files": {name: identity(directory / name) for name in names}}
    write_json(output / "all_eight_lives_sealed.json", barrier)
    verify_barrier(barrier, output, native); stable()
    phase("all_eight_lives_sealed", barrier_sha256=V1.digest(output / "all_eight_lives_sealed.json"))
    def ordinary_job(directory, seed):
        streams, receipts, checked = {}, [], []
        base = [ordinary, *(inputs / name for name in ("simple.weights", "corpus.tokens.u32", "vocabulary.u32"))]
        scenario_path = directory / f"scenarios-s{seed}.jsonl"
        for mode in ("off", "on"):
            prefix = directory / f"{mode}-s{seed}"
            command = [*base, prefix, str(seed)]
            if mode == "on": command += ["--scenarios", scenario_path]
            receipts.append(run_command(command, snapshot, directory, f"{mode}-s{seed}"))
            raw, check = FUTURE.complete_trace(Path(str(prefix) + ".jsonl"), seed)
            streams[seed, "ordinary_" + mode] = raw; checked.append({"mode": mode, "seed": seed, **check})
        require(streams[seed, "ordinary_off"] == streams[seed, "ordinary_on"], "new source diagnostics changed ordinary trace")
        for arm in json.loads(V1.PROTOCOL.read_text())["arms"]:
            require((directory / f"off-s{seed}.{arm}.life.bin").read_bytes() == (directory / f"on-s{seed}.{arm}.life.bin").read_bytes(),
                    "new source diagnostics changed saved life")
        raw, check = FUTURE.complete_scenario(scenario_path, streams[seed, "ordinary_off"], seed, scenario_protocol)
        streams[seed, "scenarios"] = raw; checked.append({"mode": "scenarios", "seed": seed, **check})
        return directory.name, streams, receipts, checked
    verify_barrier(barrier, output, native)
    phase("unseen_source_generation_started", seeds=protocol["evaluation"]["seeds"], independent_processes=4,
          barrier_sha256=V1.digest(output / "all_eight_lives_sealed.json"))
    evaluation_streams = {directory.name: {} for directory in executions}
    for execution, streams, commands, checks in parallel([(ordinary_job, (directory, seed))
            for directory in executions for seed in protocol["evaluation"]["seeds"]]):
        evaluation_streams[execution].update(streams); results[execution]["commands"] += commands
        results[execution]["host_gates"] += checks
    stable(); verify_barrier(barrier, output, native)
    cohorts = {}
    for directory in executions:
        archive = directory / "evaluation_sources.raw.jsonl.gz"
        FUTURE.write_archive(archive, evaluation_streams[directory.name])
        cohorts[directory.name] = source_samples(evaluation_streams[directory.name], protocol["evaluation"]["seeds"],
                                               V1.digest(archive), scenario_protocol)
    phase("unseen_repeated_generation_started", independent_processes=4)
    evaluation = collect_repeats("evaluation", cohorts)
    stable(); verify_barrier(barrier, output, native)
    phase("unseen_repeats_complete")
    for directory in executions:
        execution = results[directory.name]
        seal, _ = POLICY.verify_seal(directory / "policies", native, FROZEN_PROTOCOL_SHA256)
        hashes = {arm: seal["lives"][arm]["life_hash"] for arm in ARMS}
        for split, samples in (("training", training[directory.name]), ("evaluation", evaluation[directory.name])):
            target = directory / f"{split}.readout.jsonl"
            command = [sys.executable, helper, "score", "--dataset", directory / f"{split}.json", *policy_args,
                       "--policies", directory / "policies", "--output", target]
            if split == "training": command += ["--training"]
            execution["commands"].append(run_command(command, snapshot, directory, split + "-readout"))
            rows, _ = parse_raw(complete_bytes(target)); choices = validate_readouts(rows, samples, hashes)
            execution[split] = summarize(samples, choices, split)
            write_json(directory / f"{split}.summary.json", execution[split])
        execution["policy_seal"] = seal
        streams = dict(evaluation_streams[directory.name])
        streams.update({(seed, "training_parent_" + mode): raw for (seed, mode), raw in retained.items()})
        for split, seeds in (("training", protocol["training"]["seeds"]), ("evaluation", protocol["evaluation"]["seeds"])):
            for seed in seeds: streams[seed, split + "_replicates"] = complete_bytes(directory / f"{split}-repeats-s{seed}.jsonl")
            streams[0, split + "_readout"] = complete_bytes(directory / f"{split}.readout.jsonl")
            streams[0, split + "_dataset"] = complete_bytes(directory / f"{split}.json")
        streams[0, "fit"] = complete_bytes(directory / "policies/fit.jsonl")
        streams[0, "policy_seal"] = complete_bytes(directory / "policies/seal.json")
        FUTURE.write_archive(directory / "raw_traces.jsonl.gz", streams)
        execution["artifacts"] = {str(path.relative_to(directory)): identity(path) for path in sorted(directory.rglob("*")) if path.is_file()}
        write_json(directory / "result.json", execution)
        print(json.dumps({"execution": directory.name, "evaluation_h4": [r for r in execution["evaluation"]["summaries"]
              if r["group"] == "combined" and r["horizon"] == 4]}, sort_keys=True), flush=True)
    comparable = [name for name in results["execution-1"]["artifacts"]
                  if not name.endswith((".stdout.txt", ".stderr.txt", ".command.json"))]
    for name in comparable:
        require((executions[0] / name).read_bytes() == (executions[1] / name).read_bytes(), "complete repeated execution differs: " + name)
    for directory in executions:
        for name, expected in results[directory.name]["artifacts"].items():
            require(identity(directory / name) == expected, "later measured artifact changed: " + name)
    stable(); verify_barrier(barrier, output, native)
    phase("complete_two_execution_identity", artifacts=len(comparable))
    shutil.copyfile(executions[0] / "raw_traces.jsonl.gz", output / "raw_traces.jsonl.gz")
    receipts = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages,
        "executions": [results[directory.name] for directory in executions],
        "repeat": {"all_passed": True, "count": len(comparable), "byte_identical_artifacts": comparable},
        "artifacts": {"raw_traces.jsonl.gz": identity(output / "raw_traces.jsonl.gz")}}
    text = json.dumps(receipts, allow_nan=False).replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    text = text.replace(str(sys.executable), "python3")
    text = re.sub(r"/[^\s\"\\]*spa-replicates-test-[^/\s\"\\]*", "<gate_tmp>", text)
    write_json(output / "receipts.json", json.loads(text))
    print("Completed: " + str(output / "receipts.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
