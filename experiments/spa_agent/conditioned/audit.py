#!/usr/bin/env python3
"""Independent read-only audit of SPA conditioned-credit artifacts.

The checker never imports the experiment runner or the policy helper, never
executes native binaries, and never performs policy or body training. It decodes
the saved arithmetic and policy bytes using Python's standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA = "e5bb801a78cf68403c45474858fb7aff96e5aa4337003b6c06dc72253257225a"
PARENT_DATASET_SHA = "1e5f789daef52fc1a8d1bf35fc69a6b522115fafbacb4853bb16d852280af73c"
PARENT_LIFE_SHA = "6fa4aeb4d40598a82728366f02efef20fa8ae8c2a145389864750ed27dab6cb9"
METRIC_REFERENCE_SHA = "4e9024918b0aa7535e13f02d2ac076e8f495246dd7ed8b93cb304b8879998eba"
ARMS = ("initial", "raw_mean8", "conditioned_mean8", "shuffled_conditioned")
LABELS = (*ARMS, "keep", "left", "right")
AXES = ("local_connectedness", "global_connectedness", "coherence", "novelty",
        "repetition", "collapse", "continuity")
WEIGHTS = (.15, .15, .2, .2, -.15, -.1, .05)
SOURCE_KEYS = ("ordinal", "seed", "episode", "step", "snapshot", "target",
    "sentence_count", "life_hash", "body_hash", "agent_rng", "host_rng",
    "snapshot_raw_sha256", "ordinary_decision_raw_sha256", "feature_witness",
    "snapshot_witness", "source_archive_sha256")
SCORE_TOLERANCE = 2e-6
SUMMARY_TOLERANCE = 2e-12
TIE_TOLERANCE = 1e-7


def require(value, name):
    if not value:
        raise AssertionError(name)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    size, digest = 0, hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(raw)
            digest.update(raw)
    return {"bytes": size, "sha256": digest.hexdigest()}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(AssertionError("nonfinite JSON")))


def json_file(path):
    return parse(Path(path).read_bytes())


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def near(actual, expected, name, tolerance=SUMMARY_TOLERANCE):
    require(math.isfinite(actual) and math.isfinite(expected), name + " finite")
    require(abs(actual - expected) <= tolerance, name + ": " + str((actual, expected)))


def tree_near(actual, expected, name):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + " keys")
        for key in expected:
            tree_near(actual[key], expected[key], name + "." + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + " length")
        for index, value in enumerate(expected):
            tree_near(actual[index], value, name + "." + str(index))
    elif isinstance(expected, float):
        near(actual, expected, name)
    else:
        require(type(actual) is type(expected) and actual == expected, name + " value")


def mean(values):
    return sum(values) / len(values)


def variance(values):
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0


def kinds(sample):
    return [kind for kind in range(3) if sample["action_mask"] & (1 << kind)]


def source(sample):
    return {key: sample[key] for key in SOURCE_KEYS}


def raw_reward(outcome):
    total = 0.0
    for axis, weight in zip(AXES, WEIGHTS):
        total += f32(weight) * f32(f32(outcome["after"][axis]) - f32(outcome["before"][axis]))
    total -= f32(.05) * f32(outcome["cost"])
    return min(1.0, max(-1.0, f32(total)))


def targets(sample, conditioned):
    rewards = [0.0, 0.0, 0.0]
    for repeat in sample["repeats"]:
        for outcome in repeat["outcomes"]["4"]:
            rewards[outcome["action"]["kind"]] += raw_reward(outcome)
    rewards = [f32(value / 8) for value in rewards]
    raw = [f32(rewards[kind] - rewards[0]) if kind in kinds(sample) else 0.0 for kind in range(3)]
    scale = max(f32(.001), *(abs(raw[kind]) for kind in kinds(sample)))
    target = [f32(value / scale) for value in raw] if conditioned else raw
    return rewards, raw, scale, target


def huber(scores, target, mask):
    errors = [abs(float(scores[kind]) - target[kind]) for kind in range(3) if mask & (1 << kind)]
    return mean([.5 * value * value if value <= 1 else value - .5 for value in errors])


def decode_receipt(hex_record, conditioned):
    require(isinstance(hex_record, str) and hex_record == hex_record.lower(), "receipt hexadecimal")
    raw = bytes.fromhex(hex_record)
    require(len(raw) == (104 if conditioned else 88), "receipt byte length")
    result = dict(zip(("source_life_hash", "horizon", "action_mask", "learning_rate"),
                      struct.unpack_from("<QIIf", raw)))
    for key, offset in (("rewards", 20), ("targets", 32), ("scores_before", 44), ("scores_after", 56)):
        result[key] = list(struct.unpack_from("<3f", raw, offset))
    result["loss_before"], result["loss_after"] = struct.unpack_from("<2d", raw, 72)
    require(raw[68:72] == b"\0" * 4, "comparison padding")
    if conditioned:
        result["scale_floor"] = struct.unpack_from("<f", raw, 88)[0]
        result["scale"] = struct.unpack_from("<d", raw, 96)[0]
        require(raw[92:96] == b"\0" * 4, "conditioned padding")
    return result


def validate_receipt(record, receiver, donor, arm, rate, expected=None):
    conditioned = arm != "raw_mean8"
    decoded = decode_receipt(record, conditioned)
    require(decoded["source_life_hash"] == int(receiver["life_hash"], 16), "receipt source identity")
    require(decoded["horizon"] == 4 and decoded["action_mask"] == receiver["action_mask"], "receipt horizon/mask")
    require(decoded["learning_rate"] == f32(rate), "receipt rate")
    rewards, _, scale, target = targets(donor, conditioned) if expected is None else expected
    require(decoded["rewards"] == rewards and decoded["targets"] == target, "receipt reward/credit targets")
    if conditioned:
        require(decoded["scale_floor"] == f32(.001) and decoded["scale"] == scale, "receipt conditioning scale")
    for side in ("before", "after"):
        near(decoded["loss_" + side], huber(decoded["scores_" + side], target, receiver["action_mask"]),
             "receipt independent Huber loss")
    return decoded


def comparison_bytes(receiver, donor, repeat):
    raw = bytearray(struct.pack("<QII", int(receiver["life_hash"], 16), 4, receiver["action_mask"]))
    outcomes = {value["action"]["kind"]: value for value in donor["repeats"][repeat]["outcomes"]["4"]}
    for kind in range(3):
        if kind not in kinds(receiver):
            raw.extend(b"\0" * 72)
            continue
        value = outcomes[kind]
        origin = 0xFFFFFFFF if kind == 0 else receiver["target"] + (-1 if kind == 1 else 1)
        raw.extend(struct.pack("<III", kind, receiver["target"], origin))
        raw.extend(struct.pack("<15f", *[value[side][axis] for side in ("before", "after") for axis in AXES], value["cost"]))
    return bytes(raw)


def donor_indices(samples):
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample["sentence_count"], sample["target"]].append(index)
    result = {}
    for group in groups.values():
        require(len(group) > 1, "distinct shuffled donor")
        for index, receiver in enumerate(group):
            result[receiver] = group[(index + 1) % len(group)]
    return result


def decode_life(raw):
    require(len(raw) == 2296 and raw[:8] == b"NTSPA001", "canonical life framing")
    require(struct.unpack_from("<I", raw, 8)[0] == 2296, "canonical length")
    require(struct.unpack_from("<3I", raw, 12) == (1, 1, 1), "canonical versions")
    require(fnv(raw[:-8]) == struct.unpack_from("<Q", raw, 2288)[0], "canonical checksum")
    weights = struct.unpack_from("<267f", raw, 88)
    require(all(math.isfinite(value) and abs(value) <= 8 for value in weights), "policy parameter bounds")
    w1 = [list(weights[row * 30:row * 30 + 29]) for row in range(8)]
    b1 = [weights[row * 30 + 29] for row in range(8)]
    w2 = [list(weights[240 + row * 9:248 + row * 9]) for row in range(3)]
    b2 = [weights[248 + row * 9] for row in range(3)]
    native = struct.pack("<267f", *[v for row in w1 for v in row], *b1,
                         *[v for row in w2 for v in row], *b2)
    return {"weights": (w1, b1, w2, b2), "life_hash": f"{fnv(raw):016x}",
            "policy_sha256": sha(native), "raw": raw}


def predict(policy, features):
    w1, b1, w2, b2 = policy["weights"]
    x = list(map(f32, features))
    hidden = []
    for weights, bias in zip(w1, b1):
        total = bias
        for weight, value in zip(weights, x):
            total = f32(total + f32(weight * value))
        hidden.append(f32(math.tanh(total)))
    result = []
    for weights, bias in zip(w2, b2):
        total = bias
        for weight, value in zip(weights, hidden):
            total = f32(total + f32(weight * value))
        result.append(total)
    return result


def validate_readout(row, sample, policy):
    require(row["type"] == "readout" and row["source"] == source(sample), "readout source association")
    require(row["life_hash"] == policy["life_hash"], "readout sealed life")
    require(row["action_mask"] == sample["action_mask"], "readout mask")
    scores = predict(policy, sample["features"])
    maximum_error = 0.0
    for actual, expected in zip(row["scores"], scores):
        near(actual, expected, "independent saved-weight readout", SCORE_TOLERANCE)
        maximum_error = max(maximum_error, abs(actual - expected))
    choice = max(kinds(sample), key=lambda kind: row["scores"][kind])
    independent = max(kinds(sample), key=lambda kind: scores[kind])
    require(independent == choice, "independent saved-weight choice")
    origin = None if choice == 0 else sample["target"] + (-1 if choice == 1 else 1)
    require(row["action"] == {"kind": choice, "target": sample["target"], "source": origin}, "typed greedy action")
    expected = struct.pack("<IIII3f", choice, sample["target"], 0xFFFFFFFF if origin is None else origin,
                           sample["action_mask"], *row["scores"])
    require(bytes.fromhex(row["readout_hex"]) == expected, "native readout byte identity")
    return choice, maximum_error


def journal(path, samples, policies, dataset_id, expected_epochs=512):
    donors = donor_indices(samples)
    size, digest, lines, fits = 0, hashlib.sha256(), 0, 0
    previous = {arm: (policies["initial"]["life_hash"], policies["initial"]["policy_sha256"]) for arm in ARMS[1:]}
    maximum_losses = {arm: 0.0 for arm in ARMS[1:]}
    last_epoch = defaultdict(list)
    expected_targets = {(arm, index): targets(samples[donors[index] if arm == "shuffled_conditioned" else index],
                                               arm != "raw_mean8")
                        for arm in ARMS[1:] for index in range(len(samples))}
    with gzip.open(path, "rb") as stream:
        header = parse(next(stream))
        require(header["type"] == "header" and header["protocol_sha256"] == PROTOCOL_SHA,
                "fit header protocol")
        require(header["arms"] == list(ARMS) and header["epochs"] == expected_epochs and
                header["rate"] == .03 and header["scale_floor"] == .001 and header["dataset"] == dataset_id,
                "fit header config/dataset")
        require(header["fits_per_arm"] == len(samples) * expected_epochs and header["byteorder"] == "little", "fit header budget/byteorder")
        offsets = {"nt_spa_comparison_receipt.sizeof": 88, "nt_spa_conditioned_receipt.sizeof": 104,
            "nt_spa_comparison_receipt.loss_before.offset": 72,
            "nt_spa_comparison_receipt.loss_after.offset": 80,
            "nt_spa_conditioned_receipt.scale_floor.offset": 88,
            "nt_spa_conditioned_receipt.scale.offset": 96,
            "nt_spa_readout.sizeof": 28, "nt_spa_comparison.sizeof": 232}
        require(all(header["abi"][key] == value for key, value in offsets.items()), "independent native ABI offsets")
        for arm in ARMS[1:]:
            for index, sample in enumerate(samples):
                row = parse(next(stream))
                donor = samples[donors[index] if arm == "shuffled_conditioned" else index]
                require(row["type"] == "comparison" and row["arm"] == arm and
                    row["comparison_id"] == arm + ":" + str(index), "comparison order")
                require(row["source"] == source(sample) and row["donor"] == source(donor), "comparison donor/source")
                require(row["source_features"] == list(map(f32, sample["features"])), "captured feature bits")
                require(row["native_input_hex"] == [comparison_bytes(sample, donor, r).hex() for r in range(8)], "typed native consequence input")
                expected = [{"replicate": r, "outcome_sha256": {str(o["action"]["kind"]): o["raw_sha256"]
                    for o in donor["repeats"][r]["outcomes"]["4"]}} for r in range(8)]
                require(row["repeats"] == expected, "comparison raw outcome association")
                decoded = validate_receipt(row["receipt_hex"], sample, donor, arm, 0, expected_targets[arm, index])
                require(decoded["scores_before"] == [0.0] * 3 and decoded["scores_after"] == [0.0] * 3, "initial pure probe")
                display = {key: value for key, value in decoded.items() if key not in ("scale_floor", "scale")}
                display["source_life_hash"] = f"{display['source_life_hash']:016x}"
                if arm != "raw_mean8":
                    display = {"comparison": display, "scale_floor": decoded["scale_floor"], "scale": decoded["scale"]}
                require(row["initial_evaluation"] == display, "initial displayed/native receipt equality")
        for arm in ARMS[1:]:
            for epoch in range(expected_epochs):
                for index, sample in enumerate(samples):
                    row = parse(next(stream))
                    donor_index = donors[index] if arm == "shuffled_conditioned" else index
                    require(row["type"] == "fit" and row["arm"] == arm and row["epoch"] == epoch and
                            row["row"] == index and row["comparison_id"] == arm + ":" + str(index), "fit exact fixed order")
                    require(row["donor_row"] == donor_index and row["repeat_count"] == 8 and
                            row["source_feature_witness"] == sample["feature_witness"], "fit source/donor association")
                    require((row["life_before"], row["policy_before_sha256"]) == previous[arm], "fit policy identity chain")
                    previous[arm] = row["life_after"], row["policy_after_sha256"]
                    decoded = validate_receipt(row["receipt_hex"], sample, samples[donor_index], arm, .03, expected_targets[arm, index])
                    maximum_losses[arm] = max(maximum_losses[arm], decoded["loss_before"])
                    if epoch == expected_epochs - 1:
                        last_epoch[arm].append(decoded["loss_after"])
                    fits += 1
        require(not stream.read(), "fit journal trailing records")
    # A second streaming pass authenticates the entire decompressed journal,
    # including its header and comparison records, and exercises gzip CRC/EOF.
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "journal final newline")
            size += len(raw); digest.update(raw); lines += 1
    for arm in ARMS[1:]:
        require(previous[arm] == (policies[arm]["life_hash"], policies[arm]["policy_sha256"]), "fit final saved policy")
    return {"fits": fits, "comparisons": 3 * len(samples), "records": lines,
            "compressed": identity(path), "uncompressed": {"bytes": size, "sha256": digest.hexdigest()},
            "maximum_recorded_loss": maximum_losses,
            "last_epoch_mean_post_loss": {arm: mean(last_epoch[arm]) for arm in ARMS[1:]}}


def independent_rows(samples, choices, split):
    rows = []
    for sample in samples:
        index = sample["ordinal"]
        choices_here = {arm: choices[index, arm] for arm in ARMS}
        choices_here.update(keep=0, left=1 if 1 in kinds(sample) else 0, right=2 if 2 in kinds(sample) else 0)
        for horizon in (0, 1, 4):
            outcomes = [{o["action"]["kind"]: o for o in repeat["outcomes"][str(horizon)]} for repeat in sample["repeats"]]
            rewards = {kind: [raw_reward(o[kind]) for o in outcomes] for kind in kinds(sample)}
            paired = {kind: [value - keep for value, keep in zip(rewards[kind], rewards[0])] for kind in kinds(sample)}
            oracle = max(mean(value) for value in rewards.values())
            actions = [{"kind": kind, "mean_reward": mean(rewards[kind]),
                "mean_advantage_over_keep": mean(paired[kind]), "paired_variance": variance(paired[kind]),
                "paired_standard_error": math.sqrt(variance(paired[kind]) / 8),
                "positive_replicates": sum(v > TIE_TOLERANCE for v in paired[kind]),
                "negative_replicates": sum(v < -TIE_TOLERANCE for v in paired[kind]),
                "sign_disagreement": any(v > TIE_TOLERANCE for v in paired[kind]) and any(v < -TIE_TOLERANCE for v in paired[kind]),
                "replicate_advantages": paired[kind]} for kind in kinds(sample)]
            for arm, kind in choices_here.items():
                selected = [o[kind] for o in outcomes]
                rows.append({"split": split, "ordinal": index, "seed": sample["seed"], "snapshot": sample["snapshot"],
                    "target": sample["target"], "horizon": horizon, "label": arm, "kind": kind,
                    "source_snapshot_sha256": sample["snapshot_raw_sha256"], "outcome_sha256": [o["raw_sha256"] for o in selected],
                    "mean_reward": mean(rewards[kind]), "mean_advantage_over_keep": mean(paired[kind]),
                    "r0_advantage": paired[kind][0], "r1_to_r7_advantage": mean(paired[kind][1:]),
                    "oracle_regret": oracle - mean(rewards[kind]), "optimal": oracle - mean(rewards[kind]) <= TIE_TOLERANCE,
                    "changed_choice_from_initial": kind != choices_here["initial"],
                    "paired_variance": variance(paired[kind]), "paired_standard_error": math.sqrt(variance(paired[kind]) / 8),
                    "mean_cost": mean([o["cost"] for o in selected]),
                    "mean_cost_vs_keep": mean([o[kind]["cost"] - o[0]["cost"] for o in outcomes]),
                    "mean_generated": mean([o["cumulative_generated"] for o in selected]),
                    "mean_after": {a: mean([o["after"][a] for o in selected]) for a in AXES},
                    "mean_axis_delta_vs_keep": {a: mean([o[kind]["after"][a] - o[0]["after"][a] for o in outcomes]) for a in AXES},
                    "actions": actions})
    return rows


def independent_groups(rows):
    result = []
    groups = [("combined", lambda _: True)]
    groups += [("seed:" + str(seed), lambda row, seed=seed: row["seed"] == seed) for seed in sorted({r["seed"] for r in rows})]
    groups += [("target:" + str(target), lambda row, target=target: row["target"] == target) for target in range(4)]
    for group, predicate in groups:
        for horizon in (0, 1, 4):
            for arm in LABELS:
                selected = [r for r in rows if r["horizon"] == horizon and r["label"] == arm and predicate(r)]
                if not selected:
                    continue
                result.append({"group": group, "horizon": horizon, "label": arm, "states": len(selected),
                    **{key: mean([r[key] for r in selected]) for key in ("mean_reward", "mean_advantage_over_keep",
                        "r0_advantage", "r1_to_r7_advantage", "paired_variance", "paired_standard_error", "mean_cost", "mean_cost_vs_keep", "mean_generated")},
                    "mean_oracle_regret": mean([r["oracle_regret"] for r in selected]),
                    "optimal_count": sum(r["optimal"] for r in selected),
                    "changed_choices_from_initial": sum(r["changed_choice_from_initial"] for r in selected),
                    "action_counts": {str(k): sum(r["kind"] == k for r in selected) for k in range(3)},
                    "between_state_advantage_variance": variance([r["mean_advantage_over_keep"] for r in selected]),
                    "mean_after": {a: mean([r["mean_after"][a] for r in selected]) for a in AXES},
                    "mean_axis_delta_vs_keep": {a: mean([r["mean_axis_delta_vs_keep"][a] for r in selected]) for a in AXES}})
    return result


def authenticate(path, expected, name):
    require(identity(path) == {key: expected[key] for key in ("bytes", "sha256")}, name + " file identity")


def dataset(path, training):
    value = json_file(path)
    require(value["schema"] == 1 and value["replicates"] == 8 and value["protocol_sha256"] == PROTOCOL_SHA, "dataset header")
    samples = value["samples"]
    require(len(samples) == 48, "dataset source count")
    for index, sample in enumerate(samples):
        seeds = (42, 73) if training else (509, 601)
        require(sample["seed"] == seeds[index // 24] and sample["ordinal"] == index and
                sample["snapshot"] == index % 24, "dataset source order")
        require(sample["sentence_count"] == 4 and sample["target"] == (sample["episode"] + sample["step"]) % 4, "dataset target geometry")
        require(sample["action_mask"] == 1 + (2 if sample["target"] else 0) + (4 if sample["target"] < 3 else 0), "dataset mask")
        feature_bytes = struct.pack("<QII29f", int(sample["life_hash"], 16), sample["target"], 4, *sample["features"])
        require(sample["feature_witness"] == f"{fnv(feature_bytes):016x}", "dataset feature witness")
        require([r["replicate"] for r in sample["repeats"]] == list(range(8)), "dataset paired draw order")
        require(sample["outcomes"] == sample["repeats"][0]["outcomes"], "r0 source outcomes")
        for repeat in sample["repeats"]:
            for horizon in (0, 1, 4):
                outcomes = repeat["outcomes"][str(horizon)]
                require([o["action"]["kind"] for o in outcomes] == kinds(sample), "dataset complete action alternatives")
                for outcome in outcomes:
                    kind = outcome["action"]["kind"]
                    origin = None if kind == 0 else sample["target"] + (-1 if kind == 1 else 1)
                    require(outcome["action"] == {"kind": kind, "target": sample["target"], "source": origin}, "dataset typed action")
                    require(outcome["before"] == sample["before"], "dataset shared before metrics")
                    near(outcome["cost"], f32(outcome["cumulative_generated"] / (64 * (horizon + 1))), "dataset generation cost", 8e-8)
                    near(outcome["reward"], raw_reward(outcome), "dataset native reward", 8e-8)
                if horizon == 4 and sample["target"] == 3:
                    require(outcomes[0]["chain"] == outcomes[1]["chain"] and outcomes[0]["after"] == outcomes[1]["after"], "target3 future erasure control")
    return samples


def rng_start(sample, repeat, hop):
    if repeat == 0:
        domain = 0x7370615F6163746E if hop == 0 else 0x7370615F66757472
        episode, slot = (sample["episode"], sample["step"]) if hop == 0 else (sample["snapshot"], hop)
    else:
        domain = 0x7370615F72696E69 if hop == 0 else 0x7370615F72667574
        episode, slot = sample["snapshot"] * 8 + repeat, hop
    mask = (1 << 64) - 1
    value = ((domain ^ sample["seed"] ^ (episode << 32) ^ (slot << 48)) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return f"{value ^ (value >> 31):016x}"


def measurement_source(row, sample):
    for key in ("seed", "episode", "step", "snapshot", "target", "life_hash", "agent_rng", "body_hash"):
        require(row[key] == sample[key], "raw measurement source field: " + key)


def ordinary_lives(root, seed):
    paths = sorted(root.glob(f"off-s{seed}.*.life.bin"))
    require(len(paths) == 5, "ordinary diagnostics requires all five saved lives")
    for path in paths:
        require(path.read_bytes() == path.with_name(path.name.replace("off-", "on-", 1)).read_bytes(),
                "ordinary diagnostics OFF/ON life parity")


def raw_association(root, samples):
    indexed = {(s["seed"], s["snapshot"]): s for s in samples}
    joined, draws, source_rows = set(), 0, 0
    for seed in (509, 601):
        off, on = root / f"off-s{seed}.jsonl", root / f"on-s{seed}.jsonl"
        require(off.read_bytes() == on.read_bytes(), "ordinary diagnostics OFF/ON trace parity")
        ordinary_lives(root, seed)
        snapshots = {sha(line): parse(line) for line in (root / f"scenarios-s{seed}.jsonl").read_bytes().splitlines(keepends=True)}
        ordinary = {sha(line): parse(line) for line in on.read_bytes().splitlines(keepends=True)}
        current = None
        for line in (root / f"evaluation-repeats-s{seed}.jsonl").read_bytes().splitlines(keepends=True):
            row = parse(line)
            if row["type"] == "source":
                sample = indexed[seed, row["snapshot"]]
                require(row["snapshot_sha256"] == sample["snapshot_raw_sha256"] and
                    sample["snapshot_raw_sha256"] in snapshots and sample["ordinary_decision_raw_sha256"] in ordinary,
                    "new captured source raw identity")
                for key in ("features", "chain", "before", "reseeds", "life_hash", "agent_rng", "body_hash"):
                    require(row[key] == sample[key], "new captured source field: " + key)
                require(row["host_rng_before"] == sample["host_rng"] and
                        row["feature_witness"] == sample["feature_witness"], "new captured source RNG/features")
                source_rows += 1
            elif row["type"] == "replicate_begin":
                current = indexed[seed, row["snapshot"]], row["replicate"]
                sample, repeat = current
                require(row["rng_starts"] == [rng_start(sample, repeat, hop) for hop in range(5)], "independent paired continuation RNG")
                require(row["source_index"] == sample["ordinal"] and row["life_hash"] == sample["life_hash"] and
                        row["agent_rng"] == sample["agent_rng"], "replicate frozen source association")
                draws += 1
            elif row["type"] == "measurement":
                require(current is not None, "measurement outside replicate")
                sample, repeat = current
                measurement_source(row, sample)
                key = sample["ordinal"], repeat, row["horizon"], row["action"]["kind"]
                require(key not in joined, "duplicate raw measured alternative")
                joined.add(key)
                expected = {o["action"]["kind"]: o for o in sample["repeats"][repeat]["outcomes"][str(row["horizon"]) ]}[row["action"]["kind"]]
                projected = {k: row[k] for k in ("action", "before", "after", "reward", "initial_generated", "cumulative_generated", "chain")}
                projected.update(cost=row["cost_normalized"], raw_sha256=sha(line))
                require(projected == expected, "new measured alternative exact raw join")
                require(row["initial_rng_before"] == rng_start(sample, repeat, 0), "initial action paired RNG")
                for hop, transition in enumerate(row["continuation"], 1):
                    require(transition["rng_before"] == rng_start(sample, repeat, hop), "future paired RNG")
    expected_count = sum(len(kinds(sample)) * 8 * 3 for sample in samples)
    require(len(joined) == expected_count and draws == 384 and source_rows == 48, "complete raw source/draw/outcome joins")
    return {"source_rows": source_rows, "paired_draws": draws, "raw_alternative_measurements": len(joined)}


def provenance(root, receipts, training):
    metadata, barrier = json_file(root / "metadata.json"), json_file(root / "all_eight_lives_sealed.json")
    require(metadata["protocol_sha256"] == PROTOCOL_SHA and barrier["protocol_sha256"] == PROTOCOL_SHA,
            "metadata/barrier protocol")
    require(barrier["status"] == "ALL_EIGHT_LIVES_SEALED" and set(barrier["copies"]) == {"fit-1", "fit-2"}, "all eight lives barrier")
    require(barrier == receipts["barrier"], "published barrier equality")
    authenticate(root / "training.json", barrier["training"], "sealed training")
    for name, expected in barrier["copies"].items():
        require(set(expected) == {"seal.json", "fit.jsonl.gz", *(arm + ".life" for arm in ARMS)}, "barrier complete file coverage")
        for filename, pinned in expected.items():
            authenticate(root / name / filename, pinned, "barrier fitting record")
    require(barrier["source_sha256"] == metadata["source_files_sha256"], "sealed source list")
    for name, expected in metadata["source_files_sha256"].items():
        require(identity(ROOT / name)["sha256"] == expected and
                identity(root / "source_snapshot" / name)["sha256"] == expected, "frozen source identity: " + name)
    require("experiments/spa_agent/conditioned/audit.py" in metadata["source_files_sha256"], "audit frozen before fitting")
    for key, parent in (("binaries", root), ("inputs", root / "inputs")):
        require(metadata[key] == barrier[key], "sealed " + key)
        for name, expected in metadata[key].items():
            authenticate(parent / name, expected, "frozen " + key)
    require(metadata["inputs"]["simple.weights"]["sha256"] ==
            "22ce6e5dfe4efb4a00477a9652784970dc53a69fea2825174398af749acda00d", "frozen body checkpoint")
    stages = json_file(root / "stages.json")
    names = [stage["phase"] for stage in stages]
    require(names.count("fitting_started") == names.count("all_eight_lives_sealed") == names.count("new_source_generation_started") == 1, "stage coverage")
    require(names.index("fitting_started") < names.index("all_eight_lives_sealed") < names.index("new_source_generation_started"), "all fits sealed before outcomes")
    barrier_id = identity(root / "all_eight_lives_sealed.json")
    require(stages[names.index("all_eight_lives_sealed")]["barrier"] == barrier_id and
            stages[names.index("new_source_generation_started")]["barrier"] == barrier_id, "generation barrier identity")
    parent = ROOT / "experiments/spa_agent/replicates/raw_traces.jsonl.gz"
    require(identity(parent)["sha256"] == "dac359967bea506047813e25f88d7691010c0d8b47ba0b61fe35051f28300e12", "parent archive")
    retained = bytearray()
    with gzip.open(parent, "rb") as stream:
        for line in stream:
            envelope = parse(line)
            if envelope["seed"] == 0 and envelope["stream"] == "training_dataset":
                retained.extend(envelope["raw"].encode())
    require(sha(retained) == PARENT_DATASET_SHA and parse(retained)["samples"] == training,
            "retained parent training samples exact")
    return {"source_files": len(metadata["source_files_sha256"]), "sealed_lives": 8,
            "source_generation_after_seal": True, "parent_training_samples_exact": True}


def archive_check(root, record):
    authenticate(root / record["file"], record, "full raw archive")
    seen = set()
    with tarfile.open(root / record["file"], "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name not in seen and member.name in record["members"], "archive member coverage")
            require(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts, "archive member path")
            raw = archive.extractfile(member).read()
            require({"bytes": len(raw), "sha256": sha(raw)} == record["members"][member.name], "archive member authenticated bytes")
            require(raw == (root / member.name).read_bytes(), "archive member local byte parity")
            require(Path(member.name).suffix not in (".life", ".bin"), "raw archive excludes binary lives")
            seen.add(member.name)
    require(seen == set(record["members"]), "archive complete member set")
    return {"members": len(seen), **identity(root / record["file"])}


def body_identity(path):
    raw = Path(path).read_bytes()
    require(struct.unpack_from("<2I", raw) == (0x4E544F52, 21), "checkpoint tensor framing")
    cursor, values, parameters = 8, bytearray(), 0
    for _ in range(21):
        dimensions = struct.unpack_from("<i", raw, cursor)[0]; cursor += 4
        require(1 <= dimensions <= 4, "checkpoint rank")
        shape = struct.unpack_from("<" + "i" * dimensions, raw, cursor); cursor += 4 * dimensions
        require(all(value > 0 for value in shape), "checkpoint dimensions")
        count = math.prod(shape)
        values.extend(raw[cursor:cursor + 4 * count]); cursor += 4 * count; parameters += count
    require(cursor == len(raw) and parameters == 450688, "checkpoint complete450688parameter coverage")
    return f"{fnv(values):016x}"


def audit(root):
    """Read only completed outputs from the frozen execution contract."""
    protocol = ROOT / "experiments/spa_agent/conditioned/protocol.json"
    require(identity(protocol)["sha256"] == PROTOCOL_SHA, "frozen protocol")
    datasets = {split: dataset(root / (split + ".json"), split == "training") for split in ("training", "evaluation")}
    training_id = identity(root / "training.json")
    receipts = json_file(root / "receipts.json")
    report = {"schema": 1, "status": "PASS", "name": "Independent SPA conditioned-credit result audit",
        "script": identity(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
        "generation_runs": 0, "policy_fit_runs": 0,
        "command": "python3 experiments/spa_agent/conditioned/audit.py --execution-dir <completed-run> --output experiments/spa_agent/conditioned/result_audit.json",
        "method": {"fit": "Receipt bytes, targets, loss, association and complete policy-hash chain; no optimizer replay.",
            "readout": "267 canonical weights decoded independently; F32 scalar forward with score tolerance2e-6 and identical greedy action.",
            "sample": "48 states per split nested in12episodes/two seeds;8 paired draws within each state."},
        "gates": {}, "primary_h4": {}, "counterfactual": {}}
    report["gates"]["pre_outcome_seal_and_sources"] = provenance(root, receipts, datasets["training"])
    report["gates"]["raw_generation_source_and_outcome_joins"] = raw_association(root, datasets["evaluation"])
    body_hash = body_identity(root / "inputs/simple.weights")
    require(all(sample["body_hash"] == body_hash for samples in datasets.values() for sample in samples),
            "checkpoint to all96source body identities")
    report["gates"]["cross_cohort_body_identity"] = {"parameters": 450688, "body_fnv1a": body_hash,
                                                     "source_states": 96}
    copies = {}
    for copy in (1, 2):
        directory = root / f"fit-{copy}"
        seal = json_file(directory / "seal.json")
        require(seal["protocol_sha256"] == PROTOCOL_SHA and seal["dataset"] == training_id and
            seal["arms"] == list(ARMS) and seal["fits_per_arm"] == 24576, "sealed acquisition budget")
        copies[copy] = {}
        for arm in ARMS:
            path = directory / (arm + ".life")
            authenticate(path, seal["lives"][arm], "sealed life")
            decoded = decode_life(path.read_bytes())
            require(decoded["life_hash"] == seal["lives"][arm]["life_hash"], "sealed canonical FNV")
            copies[copy][arm] = decoded
            initial = copies[copy]["initial"]["raw"]
            require(decoded["raw"][:88] + decoded["raw"][1156:-8] == initial[:88] + initial[1156:-8], "policy-only acquired state")
        require(identity(directory / "raw_mean8.life")["sha256"] == PARENT_LIFE_SHA, "raw mean8 parent life parity")
        checked = journal(directory / "fit.jsonl.gz", datasets["training"], copies[copy], training_id)
        require(checked["compressed"] == seal["trace"]["compressed"], "journal compressed seal")
        require(all(checked["uncompressed"][key] == seal["trace"]["uncompressed"][key] for key in ("bytes", "sha256")), "journal decoded seal")
        report["gates"][f"fitting_copy_{copy}"] = checked
    for arm in ARMS:
        require(copies[1][arm]["raw"] == copies[2][arm]["raw"], "two fitting copies final policy identity")
    require(identity(root / "fit-1/fit.jsonl.gz") == identity(root / "fit-2/fit.jsonl.gz"), "two fitting copies journal identity")
    maximum_score_error, choices_checked = 0.0, 0
    for split, samples in datasets.items():
        paths = [root / f"{split}.readout-{copy}.jsonl" for copy in (1, 2)]
        require(paths[0].read_bytes() == paths[1].read_bytes(), "two fitting copies pure readout identity")
        rows = [parse(line) for line in paths[0].read_bytes().splitlines()]
        require(len(rows) == len(samples) * 4, "complete policy readout count")
        choices, native_scores = {}, {}
        for index, row in enumerate(rows):
            arm = ARMS[index % 4]; sample = samples[index // 4]
            require(row["arm"] == arm, "pure readout order")
            choice, error = validate_readout(row, sample, copies[1][arm])
            choices[sample["ordinal"], arm] = choice
            native_scores[sample["ordinal"], arm] = row["scores"]
            maximum_score_error = max(maximum_score_error, error); choices_checked += 1
        state_rows = independent_rows(samples, choices, split)
        groups = independent_groups(state_rows)
        reported = json_file(root / (split + ".summary.json"))
        require(reported == receipts[split], "receipt standalone summary equality")
        tree_near(reported["state_comparisons"], state_rows, "independent raw state comparisons")
        tree_near(reported["summaries"], groups, "independent raw policy/seed/target summaries")
        report["primary_h4"][split] = [row for row in groups if row["group"] == "combined" and row["horizon"] == 4]
        changed = []
        for sample in samples:
            index = sample["ordinal"]
            if len({choices[index, arm] for arm in ARMS}) > 1:
                changed.append({"source": source(sample), "features": sample["features"],
                    "choices": {arm: choices[index, arm] for arm in ARMS},
                    "scores": {arm: native_scores[index, arm] for arm in ARMS}})
        report["counterfactual"][split] = {"changed_source_states": len(changed), "records": changed,
            "held_fixed": "same captured29features/source history and source RNG; saved lives have identical non-policy bytes"}
    report["gates"]["independent_saved_policy_choices"] = {"readouts": choices_checked,
        "score_components": choices_checked * 3, "maximum_absolute_error": maximum_score_error,
        "score_tolerance": SCORE_TOLERANCE, "exact_typed_choices": True}
    metric_raw = (ROOT / "experiments/spa_agent/future/result_audit.json").read_bytes()
    require(sha(metric_raw) == METRIC_REFERENCE_SHA, "independent token metric reference receipt")
    reference = parse(metric_raw)["reproducibility"]
    require(sha(reference["script"].encode()) == reference["script_sha256"], "independent token metric reference source")
    namespace = {"__name__": "retained_independent_spa_math", "__file__": "<retained-independent-audit>"}
    exec(compile(reference["script"], "<retained-independent-audit>", "exec"), namespace)
    metrics = namespace["Metrics"](namespace["token_table"](root / "inputs/simple.weights"))
    for samples in datasets.values():
        for sample in samples:
            metrics.check(sample["chain"], sample["target"], sample["before"])
            for repeat in sample["repeats"]:
                for outcomes in repeat["outcomes"].values():
                    for outcome in outcomes:
                        metrics.check(outcome["chain"], sample["target"], outcome["after"])
                        metrics.reward(outcome["before"], outcome["after"], outcome["cost"], outcome["reward"])
    report["gates"]["independent_checkpoint_token_metrics"] = {"axis_values": metrics.checks,
        "unique_sentences": len(metrics.cache), "maximum_absolute_error_by_axis": metrics.maximum_error,
        "metric_tolerance": namespace["METRIC_TOLERANCE"], "reward_values": metrics.rewards,
        "maximum_absolute_reward_error": metrics.reward_maximum_error,
        "reference_receipt_sha256": METRIC_REFERENCE_SHA}
    # Result publication names every raw file; verify its bytes against that
    # manifest before authenticating the independently pinned archive copy.
    for name, item in receipts["artifacts"].items():
        authenticate(root / name, item, "final artifact " + name)
    report["gates"]["published_artifact_identities"] = {"files": len(receipts["artifacts"])}
    require(set(receipts["archive"]["members"]) == {name for name in receipts["artifacts"]
        if Path(name).suffix not in (".life", ".bin")}, "archive covers every closed raw artifact")
    report["gates"]["complete_raw_archive"] = archive_check(root, receipts["archive"])
    report["identities"] = {"receipt": identity(root / "receipts.json"),
        "training": training_id, "evaluation": identity(root / "evaluation.json"),
        "initial": identity(root / "fit-1/initial.life"),
        "conditioned": identity(root / "fit-1/conditioned_mean8.life")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.execution_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "gates": len(report["gates"]),
                      "output": identity(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
