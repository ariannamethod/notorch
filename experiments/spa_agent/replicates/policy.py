#!/usr/bin/env python3
"""Import SPA to acquire and read policies from paired repeated consequences.

Dataset association, ordering and receipts live here. Reward evaluation,
per-repeat clipping, averaging, targets, gradients, readout and life files
are native notorch operations through the public Python binding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "python"))
import SPA

ARMS = ("initial", "single_h4", "mean8_h4", "shuffled_mean8")
AXES = tuple(name for name, _ in SPA.Metrics._fields_)
PARENT_H4_SHA256 = "3f8fa08f331a1c34d50201e0c5c567ca1b2b1853e6c9b54de265c9886b0d02cb"
PARENT_H4_LIFE_HASH = "eaf732191c780118"
FROZEN_PROTOCOL_SHA256 = "c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75"
SOURCES = ("python/SPA.py", "python/notorch.py", "spa_agent.c", "spa_agent.h", "spa_binding.c", "spa_binding.h",
           "experiments/spa_agent/replicates/policy.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw)}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def hex_value(value, length, label):
    require(isinstance(value, str) and len(value) == length and
            all(c in "0123456789abcdef" for c in value), "invalid " + label)
    return int(value, 16)


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, "invalid " + label)
    return value


def number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            "invalid " + label)
    return value


def feature_witness(sample):
    raw = struct.pack("<QII29f", int(sample["life_hash"], 16), sample["target"],
                      sample["sentence_count"], *sample["features"])
    result = 14695981039346656037
    for byte in raw:
        result = ((result ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def action_mask(sample):
    target, count = sample["target"], sample["sentence_count"]
    return 1 | (2 if target else 0) | (4 if target + 1 < count else 0)


def source_identity(sample):
    return {key: sample[key] for key in ("ordinal", "seed", "episode", "step", "snapshot",
        "target", "sentence_count", "life_hash", "body_hash", "agent_rng", "host_rng",
        "snapshot_raw_sha256", "ordinary_decision_raw_sha256", "feature_witness",
        "snapshot_witness", "source_archive_sha256")}


def experience(sample):
    require(len(sample["features"]) == SPA.FEATURES, "29 source features required")
    for value in sample["features"]:
        number(value, -1, 1, "source feature")
    return SPA.Experience(version=1, sentence_index=sample["target"],
        sentence_count=sample["sentence_count"], source_life_hash=int(sample["life_hash"], 16),
        features=(C.c_float * SPA.FEATURES)(*sample["features"]))


def validate_source(sample, ordinal):
    require(isinstance(sample, dict), "source must be an object")
    require(integer(sample["ordinal"], 0, 47, "ordinal") == ordinal, "source order changed")
    integer(sample["seed"], 1, 0xFFFFFFFF, "seed")
    episode = integer(sample["episode"], 0, 5, "episode")
    step = integer(sample["step"], 0, 3, "step")
    require(integer(sample["snapshot"], 0, 23, "snapshot") == episode * 4 + step, "source snapshot coordinate changed")
    require(integer(sample["sentence_count"], 1, 4096, "sentence count") == 4, "source sentence count changed")
    require(integer(sample["target"], 0, 3, "target") == (episode + step) % 4, "source target coordinate changed")
    require(integer(sample["action_mask"], 1, 7, "action mask") == action_mask(sample), "source action mask changed")
    integer(sample["agent_rng"], 1, 0xFFFFFFFF, "agent RNG")
    for key in ("life_hash", "body_hash", "host_rng", "feature_witness"):
        hex_value(sample[key], 16, key)
    for key in ("snapshot_raw_sha256", "ordinary_decision_raw_sha256", "source_archive_sha256"):
        hex_value(sample[key], 64, key)
    # The host supplies its native 64-bit field-snapshot witness.
    hex_value(sample["snapshot_witness"], 16, "host snapshot witness")
    experience(sample)
    require(sample["feature_witness"] == feature_witness(sample), "feature-source witness changed")


def read_dataset(path, protocol_sha, *, training):
    value = read_json(path)
    require(value["schema"] == 1 and value["replicates"] == 8, "wrong repeated dataset schema")
    require(value["protocol_sha256"] == protocol_sha, "dataset protocol identity changed")
    samples = value["samples"]
    require(isinstance(samples, list) and len(samples) == 48, "dataset must contain 48 source states")
    seeds = [42, 73] if training else [307, 401]
    for ordinal, sample in enumerate(samples):
        validate_source(sample, ordinal)
        require(sample["seed"] == seeds[ordinal // 24] and sample["snapshot"] == ordinal % 24,
                "registered source order changed")
        require(sample["body_hash"] == samples[0]["body_hash"], "mixed body identities")
    return samples


def metrics(value):
    require(isinstance(value, dict) and set(value) == set(AXES), "raw metric axes changed")
    return SPA.Metrics(*(number(value[axis], 0, 1, axis) for axis in AXES))


def comparisons(receiver, donor, count):
    require((receiver["sentence_count"], receiver["target"]) ==
            (donor["sentence_count"], donor["target"]), "shuffle donor coordinates differ")
    repeats = donor["repeats"]
    require(len(repeats) == 8 and [r["replicate"] for r in repeats] == list(range(8)),
            "repeat order must be exactly 0..7")
    require(donor["outcomes"] == repeats[0]["outcomes"], "r0 differs from retained parent outcomes")
    result, references = [], []
    for repeat in repeats[:count]:
        raw = repeat["outcomes"]["4"]
        expected = [kind for kind in range(3) if action_mask(receiver) & (1 << kind)]
        require([o["action"]["kind"] for o in raw] == expected, "paired action coverage/order changed")
        comparison = SPA.Comparison(source_life_hash=int(receiver["life_hash"], 16),
                                    horizon=4, action_mask=action_mask(receiver))
        ids = [None, None, None]
        for outcome in raw:
            action = outcome["action"]
            kind = integer(action["kind"], 0, 2, "action kind")
            expected_source = None if kind == 0 else receiver["target"] + (-1 if kind == 1 else 1)
            require(action["target"] == receiver["target"] and action["source"] == expected_source,
                    "raw typed action differs from source coordinates")
            ids[kind] = outcome["raw_sha256"]
            hex_value(ids[kind], 64, "outcome record SHA256")
            consequence = SPA.Consequence(metrics(outcome["before"]), metrics(outcome["after"]),
                                          number(outcome["cost"], 0, 1, "cost"))
            comparison.alternatives[kind] = SPA.Alternative(
                SPA.Action.at(kind, receiver["target"]), consequence)
        result.append(comparison)
        references.append({"replicate": repeat["replicate"], "outcome_sha256": ids})
    return result, references


def shuffle_donors(samples):
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample["sentence_count"], sample["target"]].append(index)
    donors = list(range(len(samples)))
    for group in groups.values():
        require(len(group) > 1, "shuffle group must contain distinct source states")
        for index, receiver in enumerate(group):
            donors[receiver] = group[(index + 1) % len(group)]
    return donors


def json_line(stream, record):
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def native_receipt(receipt):
    result = receipt.as_dict()
    result["source_life_hash"] = f"{receipt.source_life_hash:016x}"
    return result


def policy_only(agent, initial):
    current = agent.state
    current.policy = initial.policy
    require(bytes(current) == bytes(initial), "replay changed non-policy life bytes")


def validate_protocol(path):
    before = identity(path)["sha256"]
    require(before == FROZEN_PROTOCOL_SHA256, "frozen protocol SHA256 changed")
    protocol = read_json(path)
    training = protocol["training"]
    require(training["seeds"] == [42, 73] and training["rows"] == 48,
            "registered training cohort changed")
    require(training["epochs"] == 512 and training["learning_rate"] == .03 and
            training["policy_seed"] == 1 and training["exploration"] == 0,
            "registered policy budget/config changed")
    require(tuple(training["arms"]) == ARMS, "registered arm order changed")
    require(protocol["evaluation"]["seeds"] == [307, 401], "registered evaluation cohort changed")
    require(identity(path)["sha256"] == before, "protocol changed while parsing")
    return protocol, before


def fit_policies(samples, native, directory, *, epochs, protocol_sha, dataset_identity,
                 require_parent_parity):
    """Source-frozen worker; synthetic tests may use a short epoch count."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source_ids = {name: identity(ROOT / name)["sha256"] for name in SOURCES}
    library_id = identity(native.path)
    initial_agent = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED,
                                               seed=1, exploration=0), native=native)
    initial = initial_agent.state
    initial_agent.save(directory / "initial.life")
    donors = shuffle_donors(samples)
    native_inputs = {}
    trace_path = directory / "fit.jsonl"
    with trace_path.open("x") as trace:
        json_line(trace, {"type": "header", "schema": 1, "protocol_sha256": protocol_sha,
            "dataset": dataset_identity, "arms": ARMS, "epochs": epochs, "rate": .03,
            "fits_per_arm": epochs * len(samples), "arithmetic": "SPA.Agent.fit_repeated in libnotorch"})
        for arm in ARMS[1:]:
            for index, sample in enumerate(samples):
                donor_index = donors[index] if arm == "shuffled_mean8" else index
                count = 1 if arm == "single_h4" else 8
                captured = experience(sample)
                values, references = comparisons(sample, samples[donor_index], count)
                probe = initial_agent.fit_repeated(captured, values, 0)
                policy_only(initial_agent, initial)
                require(bytes(initial_agent.state.policy) == bytes(initial.policy), "rate-zero validation changed policy")
                comparison_id = arm + ":" + str(index)
                native_inputs[arm, index] = (captured, values, donor_index, comparison_id)
                json_line(trace, {"type": "comparison", "comparison_id": comparison_id, "arm": arm,
                    "source": source_identity(sample), "source_features": list(captured.features),
                    "donor": source_identity(samples[donor_index]), "repeats": references,
                    "native_input_hex": [bytes(value).hex() for value in values],
                    "initial_evaluation": native_receipt(probe)})
        lives = {"initial": {"file": "initial.life", "life_hash": f"{initial_agent.hash:016x}",
                             **identity(directory / "initial.life")}}
        for arm in ARMS[1:]:
            agent = SPA.Agent.from_file(directory / "initial.life", native=native)
            for epoch in range(epochs):
                for index, sample in enumerate(samples):
                    captured, values, donor_index, comparison_id = native_inputs[arm, index]
                    receipt = agent.fit_repeated(captured, values, .03)
                    policy_only(agent, initial)
                    json_line(trace, {"type": "fit", "arm": arm, "epoch": epoch,
                        "row": index, "comparison_id": comparison_id,
                        "source_feature_witness": sample["feature_witness"], "donor_row": donor_index,
                        "repeat_count": len(values), "receipt": native_receipt(receipt),
                        "receipt_hex": bytes(receipt).hex()})
            name = arm + ".life"
            agent.save(directory / name)
            lives[arm] = {"file": name, "life_hash": f"{agent.hash:016x}", **identity(directory / name)}
            if arm == "single_h4" and require_parent_parity:
                require(lives[arm]["sha256"] == PARENT_H4_SHA256 and
                        lives[arm]["life_hash"] == PARENT_H4_LIFE_HASH,
                        "count-one acquisition does not reproduce the retained parent H4 life")
        trace.flush()
        os.fsync(trace.fileno())
    seal = {"schema": 1, "protocol_sha256": protocol_sha, "dataset": dataset_identity,
        "arms": ARMS, "fits_per_arm": epochs * len(samples), "lives": lives,
        "trace": {"file": "fit.jsonl", **identity(trace_path)},
        "count_one_parent_parity": "byte-identical canonical checkpoint" if require_parent_parity else "synthetic fixture",
        "parent_h4_sha256": PARENT_H4_SHA256,
        "source_sha256": source_ids, "native_library": library_id}
    require({name: identity(ROOT / name)["sha256"] for name in SOURCES} == source_ids,
            "policy source changed during fitting")
    require(identity(native.path) == library_id, "native library changed during fitting")
    with (directory / "seal.json").open("x") as stream:
        json.dump(seal, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return seal


def verify_seal(directory, native, protocol_sha):
    directory = Path(directory)
    seal = read_json(directory / "seal.json")
    require(seal["schema"] == 1 and seal["protocol_sha256"] == protocol_sha and
            tuple(seal["arms"]) == ARMS, "policy seal identity changed")
    require(seal["fits_per_arm"] == 24576 and seal["count_one_parent_parity"] ==
            "byte-identical canonical checkpoint", "unregistered policy seal")
    require(seal["parent_h4_sha256"] == PARENT_H4_SHA256 and
            seal["lives"]["single_h4"]["sha256"] == PARENT_H4_SHA256 and
            seal["lives"]["single_h4"]["life_hash"] == PARENT_H4_LIFE_HASH,
            "sealed single_h4 control differs from retained parent life")
    agents = {}
    for arm in ARMS:
        record = seal["lives"][arm]
        require(record["file"] == arm + ".life", "policy filename changed")
        path = directory / record["file"]
        require(identity(path) == {key: record[key] for key in ("bytes", "sha256")},
                "sealed policy bytes changed")
        agents[arm] = SPA.Agent.from_file(path, native=native)
        require(f"{agents[arm].hash:016x}" == record["life_hash"], "sealed native life hash changed")
    require(seal["trace"]["file"] == "fit.jsonl" and identity(directory / "fit.jsonl") ==
            {key: seal["trace"][key] for key in ("bytes", "sha256")}, "sealed fit trace changed")
    for name in SOURCES:
        require(identity(ROOT / name)["sha256"] == seal["source_sha256"][name], "sealed policy source changed")
    require(identity(native.path) == seal["native_library"], "sealed native library changed")
    return seal, agents


def score_samples(samples, agents, stream):
    """Reads source features only; measured outcome fields are never read."""
    hashes = {arm: agent.hash for arm, agent in agents.items()}
    for sample in samples:
        captured = experience(sample)
        for arm in ARMS:
            readout = agents[arm].score(captured)
            require(agents[arm].hash == hashes[arm], "readout mutated a sealed life")
            json_line(stream, {"type": "readout", "arm": arm, "source": source_identity(sample),
                "life_hash": f"{hashes[arm]:016x}", "action_mask": readout.action_mask,
                "action": {"kind": readout.action.kind, "target": readout.action.target,
                           "source": None if readout.action.source == SPA.NO_SOURCE else readout.action.source},
                "scores": list(readout.scores), "readout_hex": bytes(readout).hex()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "score"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--library", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "score":
            child.add_argument("--policies", type=Path, required=True)
            child.add_argument("--training", action="store_true", help="read the registered 42/73 training cohort")
    args = parser.parse_args()
    _, protocol_sha = validate_protocol(args.protocol)
    native = SPA.Native(args.library)
    training = args.command == "train" or args.training
    dataset_id = identity(args.dataset)
    samples = read_dataset(args.dataset, protocol_sha, training=training)
    require(identity(args.dataset) == dataset_id, "dataset changed while parsing")
    if args.command == "train":
        seal = fit_policies(samples, native, args.output, epochs=512,
                            protocol_sha=protocol_sha, dataset_identity=dataset_id, require_parent_parity=True)
        require(identity(args.dataset) == dataset_id, "dataset changed while training")
        require(identity(args.protocol)["sha256"] == protocol_sha, "protocol changed while training")
        print(json.dumps({"sealed": list(ARMS), "fits_per_arm": seal["fits_per_arm"],
                          "count_one_parent_parity": seal["count_one_parent_parity"]}, sort_keys=True))
    else:
        seal_before = identity(args.policies / "seal.json")
        _, agents = verify_seal(args.policies, native, protocol_sha)
        with args.output.open("x") as stream:
            score_samples(samples, agents, stream)
            stream.flush()
            os.fsync(stream.fileno())
        verify_seal(args.policies, native, protocol_sha)
        require(identity(args.policies / "seal.json") == seal_before, "seal changed during readout")
        require(identity(args.dataset) == dataset_id, "dataset changed during readout")
        print(json.dumps({"readouts": len(samples) * len(ARMS), **identity(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
