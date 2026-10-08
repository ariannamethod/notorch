#!/usr/bin/env python3
"""Native SPA acquisition with fixed per-state action-effect conditioning.

The public Python binding executes every reward, mean, target, gradient and
policy readout in C. This helper associates source states with measured
outcomes and writes an exact, compressed receipt for every native update.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as C
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "spa_conditioned_parent_policy", ROOT / "experiments/spa_agent/replicates/policy.py")
PARENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARENT)
SPA = PARENT.SPA

ARMS = ("initial", "raw_mean8", "conditioned_mean8", "shuffled_conditioned")
AXES = PARENT.AXES
PARENT_MEAN8_SHA256 = "6fa4aeb4d40598a82728366f02efef20fa8ae8c2a145389864750ed27dab6cb9"
PARENT_MEAN8_LIFE_HASH = "fcbf95a7bf79af9f"
FROZEN_PROTOCOL_SHA256 = "e5bb801a78cf68403c45474858fb7aff96e5aa4337003b6c06dc72253257225a"
SOURCES = PARENT.SOURCES + ("experiments/spa_agent/conditioned/policy.py",)
require, identity, read_json = PARENT.require, PARENT.identity, PARENT.read_json
experience, comparisons = PARENT.experience, PARENT.comparisons
source_identity, shuffle_donors = PARENT.source_identity, PARENT.shuffle_donors
json_line, policy_only = PARENT.json_line, PARENT.policy_only


def read_dataset(path, protocol_sha, *, training):
    value = read_json(path)
    require(value["schema"] == 1 and value["replicates"] == 8,
            "wrong repeated dataset schema")
    require(value["protocol_sha256"] == protocol_sha, "dataset protocol identity changed")
    samples = value["samples"]
    require(isinstance(samples, list) and len(samples) == 48,
            "dataset must contain 48 source states")
    seeds = [42, 73] if training else [509, 601]
    for ordinal, sample in enumerate(samples):
        PARENT.validate_source(sample, ordinal)
        require(sample["seed"] == seeds[ordinal // 24] and sample["snapshot"] == ordinal % 24,
                "registered source order changed")
        require(sample["body_hash"] == samples[0]["body_hash"], "mixed body identities")
    return samples


def validate_protocol(path):
    before = identity(path)["sha256"]
    require(before == FROZEN_PROTOCOL_SHA256, "frozen protocol SHA256 changed")
    protocol = read_json(path)
    training = protocol["training"]
    require(training["seeds"] == [42, 73] and training["rows"] == 48,
            "registered training cohort changed")
    require(training["epochs"] == 512 and training["learning_rate"] == .03 and
            training["scale_floor"] == .001 and training["policy_seed"] == 1 and
            training["exploration"] == 0 and training["fits_per_arm"] == 24576,
            "registered policy budget/config changed")
    require(tuple(training["arms"]) == ARMS, "registered arm order changed")
    require(protocol["evaluation"]["seeds"] == [509, 601] and
            protocol["evaluation"]["rows"] == 48, "registered evaluation cohort changed")
    require(identity(path)["sha256"] == before, "protocol changed while parsing")
    return protocol, before


def native_receipt(receipt):
    if isinstance(receipt, SPA.ConditionedReceipt):
        return {"comparison": PARENT.native_receipt(receipt.comparison),
                "scale_floor": receipt.scale_floor, "scale": receipt.scale}
    return PARENT.native_receipt(receipt)


def receipt_type(arm):
    return SPA.ComparisonReceipt if arm == "raw_mean8" else SPA.ConditionedReceipt


def decode_receipt(arm, hex_record):
    """Decode a local-ABI native receipt; the journal header pins every offset."""
    typ = receipt_type(arm)
    PARENT.hex_value(hex_record, 2 * C.sizeof(typ), "native receipt bytes")
    return typ.from_buffer_copy(bytes.fromhex(hex_record))


def fit(agent, captured, values, rate, floor, arm):
    if arm == "raw_mean8":
        return agent.fit_repeated(captured, values, rate)
    require(arm in ("conditioned_mean8", "shuffled_conditioned"), "unknown fitted arm")
    return agent.fit_conditioned(captured, values, rate, floor)


def journal_identity(path):
    """Read the complete gzip member, including its CRC, and pin decoded bytes."""
    digest, size, records, counts = hashlib.sha256(), 0, 0, Counter()
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "fit journal has an unterminated record")
            value = json.loads(raw, object_pairs_hook=PARENT.unique_object,
                               parse_constant=lambda x: (_ for _ in ()).throw(
                                   ValueError("nonfinite JSON number: " + x)))
            require(isinstance(value, dict) and isinstance(value.get("type"), str),
                    "fit journal record has no type")
            digest.update(raw)
            size += len(raw)
            records += 1
            counts[value["type"]] += 1
    return {"compressed": identity(path), "uncompressed": {
        "bytes": size, "sha256": digest.hexdigest(), "records": records,
        "record_counts": dict(sorted(counts.items()))}}


class Journal:
    """Create once, stream compressed exact JSONL, then authenticate before publish."""
    def __init__(self, path):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.name + ".part")
        require(not self.path.exists(), "fit journal already exists")
        self.file = self.temporary.open("xb")
        self.gzip = gzip.GzipFile(filename="", fileobj=self.file, mode="wb", mtime=0)
        self.digest, self.size, self.records, self.counts = hashlib.sha256(), 0, 0, Counter()

    def write(self, record):
        raw = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
        self.gzip.write(raw)
        self.digest.update(raw)
        self.size += len(raw)
        self.records += 1
        self.counts[record["type"]] += 1

    def finish(self):
        self.gzip.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        result = journal_identity(self.temporary)
        require(result["uncompressed"] == {
            "bytes": self.size, "sha256": self.digest.hexdigest(), "records": self.records,
            "record_counts": dict(sorted(self.counts.items()))}, "fit journal readback changed")
        os.link(self.temporary, self.path)
        self.temporary.unlink()
        return result

    def abort(self):
        # Retain the partial file as the receipt of an interrupted acquisition.
        self.gzip.close()
        self.file.close()


def fit_policies(samples, native, directory, *, epochs, protocol_sha, dataset_identity,
                 require_parent_parity, scale_floor=.001):
    """Fixed acquisition worker; synthetic fixtures may use a short epoch count."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source_ids = {name: identity(ROOT / name)["sha256"] for name in SOURCES}
    library_id = identity(native.path)
    initial_agent = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED,
                                               seed=1, exploration=0), native=native)
    initial = initial_agent.state
    initial_agent.save(directory / "initial.life")
    donors, native_inputs = shuffle_donors(samples), {}
    trace_path = directory / "fit.jsonl.gz"
    trace = Journal(trace_path)
    try:
        trace.write({"type": "header", "schema": 1, "protocol_sha256": protocol_sha,
            "dataset": dataset_identity, "arms": ARMS, "epochs": epochs, "rate": .03,
            "scale_floor": scale_floor, "fits_per_arm": epochs * len(samples),
            "arithmetic": "SPA.Agent.fit_repeated / fit_conditioned in libnotorch",
            "receipt_format": "exact native structure bytes, lowercase hexadecimal",
            "byteorder": sys.byteorder, "abi": SPA._layout(),
            "receipt_types": {arm: {"name": receipt_type(arm).__name__,
                "bytes": C.sizeof(receipt_type(arm))} for arm in ARMS[1:]}})
        for arm in ARMS[1:]:
            for index, sample in enumerate(samples):
                donor_index = donors[index] if arm == "shuffled_conditioned" else index
                captured = experience(sample)
                values, references = comparisons(sample, samples[donor_index], 8)
                probe = fit(initial_agent, captured, values, 0, scale_floor, arm)
                policy_only(initial_agent, initial)
                require(bytes(initial_agent.state.policy) == bytes(initial.policy),
                        "rate-zero validation changed policy")
                comparison_id = arm + ":" + str(index)
                native_inputs[arm, index] = (captured, values, donor_index, comparison_id)
                trace.write({"type": "comparison", "comparison_id": comparison_id, "arm": arm,
                    "source": source_identity(sample), "source_features": list(captured.features),
                    "donor": source_identity(samples[donor_index]), "repeats": references,
                    "native_input_hex": [bytes(value).hex() for value in values],
                    "initial_evaluation": native_receipt(probe), "receipt_hex": bytes(probe).hex()})
        lives = {"initial": {"file": "initial.life", "life_hash": f"{initial_agent.hash:016x}",
                             **identity(directory / "initial.life")}}
        for arm in ARMS[1:]:
            agent = SPA.Agent.from_file(directory / "initial.life", native=native)
            for epoch in range(epochs):
                for index, sample in enumerate(samples):
                    captured, values, donor_index, comparison_id = native_inputs[arm, index]
                    life_before = f"{agent.hash:016x}"
                    policy_before = PARENT.sha256(bytes(agent.state.policy))
                    receipt = fit(agent, captured, values, .03, scale_floor, arm)
                    policy_only(agent, initial)
                    trace.write({"type": "fit", "arm": arm, "epoch": epoch,
                        "row": index, "comparison_id": comparison_id,
                        "source_feature_witness": sample["feature_witness"], "donor_row": donor_index,
                        "repeat_count": len(values), "life_before": life_before,
                        "life_after": f"{agent.hash:016x}", "policy_before_sha256": policy_before,
                        "policy_after_sha256": PARENT.sha256(bytes(agent.state.policy)),
                        "receipt_hex": bytes(receipt).hex()})
            name = arm + ".life"
            agent.save(directory / name)
            lives[arm] = {"file": name, "life_hash": f"{agent.hash:016x}",
                          **identity(directory / name)}
            if arm == "raw_mean8" and require_parent_parity:
                require(lives[arm]["sha256"] == PARENT_MEAN8_SHA256 and
                        lives[arm]["life_hash"] == PARENT_MEAN8_LIFE_HASH,
                        "raw_mean8 acquisition differs from retained parent mean8 life")
        trace_identity = trace.finish()
    except BaseException:
        if not trace.file.closed:
            trace.abort()
        raise
    expected_counts = {"header": 1, "comparison": 3 * len(samples),
                       "fit": 3 * epochs * len(samples)}
    require(trace_identity["uncompressed"]["record_counts"] == expected_counts,
            "fit journal record counts changed")
    seal = {"schema": 1, "protocol_sha256": protocol_sha, "dataset": dataset_identity,
        "arms": ARMS, "fits_per_arm": epochs * len(samples), "scale_floor": scale_floor,
        "lives": lives, "trace": {"file": "fit.jsonl.gz", **trace_identity},
        "raw_mean8_parent_parity": "byte-identical canonical checkpoint" if require_parent_parity
                                   else "synthetic fixture",
        "parent_mean8_sha256": PARENT_MEAN8_SHA256,
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


def verify_seal(directory, native, protocol_sha, *, synthetic=False):
    directory = Path(directory)
    seal = read_json(directory / "seal.json")
    require(seal["schema"] == 1 and seal["protocol_sha256"] == protocol_sha and
            tuple(seal["arms"]) == ARMS, "policy seal identity changed")
    if not synthetic:
        require(seal["fits_per_arm"] == 24576 and seal["scale_floor"] == .001 and
                seal["raw_mean8_parent_parity"] == "byte-identical canonical checkpoint",
                "unregistered policy seal")
        require(seal["parent_mean8_sha256"] == PARENT_MEAN8_SHA256 and
                seal["lives"]["raw_mean8"]["sha256"] == PARENT_MEAN8_SHA256 and
                seal["lives"]["raw_mean8"]["life_hash"] == PARENT_MEAN8_LIFE_HASH,
                "sealed raw_mean8 control differs from retained parent life")
    agents = {}
    for arm in ARMS:
        record = seal["lives"][arm]
        require(record["file"] == arm + ".life", "policy filename changed")
        path = directory / record["file"]
        require(identity(path) == {key: record[key] for key in ("bytes", "sha256")},
                "sealed policy bytes changed")
        agents[arm] = SPA.Agent.from_file(path, native=native)
        require(f"{agents[arm].hash:016x}" == record["life_hash"], "sealed native life hash changed")
    require(seal["trace"]["file"] == "fit.jsonl.gz" and journal_identity(directory / "fit.jsonl.gz") ==
            {key: seal["trace"][key] for key in ("compressed", "uncompressed")},
            "sealed fit trace changed")
    require(set(seal["source_sha256"]) == set(SOURCES), "sealed source list changed")
    for name in SOURCES:
        require(identity(ROOT / name)["sha256"] == seal["source_sha256"][name],
                "sealed policy source changed")
    require(identity(native.path) == seal["native_library"], "sealed native library changed")
    return seal, agents


def score_samples(samples, agents, stream):
    """Source-only native readout: measured outcomes are never accessed here."""
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
            child.add_argument("--training", action="store_true")
    args = parser.parse_args()
    _, protocol_sha = validate_protocol(args.protocol)
    native = SPA.Native(args.library)
    dataset_id = identity(args.dataset)
    samples = read_dataset(args.dataset, protocol_sha, training=args.command == "train" or args.training)
    require(identity(args.dataset) == dataset_id, "dataset changed while parsing")
    if args.command == "train":
        seal = fit_policies(samples, native, args.output, epochs=512, protocol_sha=protocol_sha,
                            dataset_identity=dataset_id, require_parent_parity=True, scale_floor=.001)
        require(identity(args.dataset) == dataset_id, "dataset changed while training")
        require(identity(args.protocol)["sha256"] == protocol_sha, "protocol changed while training")
        print(json.dumps({"sealed": list(ARMS), "fits_per_arm": seal["fits_per_arm"],
                          "raw_mean8_parent_parity": seal["raw_mean8_parent_parity"]}, sort_keys=True))
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
