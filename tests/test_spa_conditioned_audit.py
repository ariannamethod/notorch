#!/usr/bin/env python3
"""Cheap independent-audit arithmetic and deliberate-record-corruption gates."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("conditioned_independent_audit", ROOT / "experiments/spa_agent/conditioned/audit.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def sample():
    before = {axis: .5 for axis in A.AXES}
    source = {key: "0" * 16 for key in A.SOURCE_KEYS}
    source.update(ordinal=0, seed=42, episode=0, step=1, snapshot=1,
                  target=1, sentence_count=4, life_hash="0000000000000001", agent_rng=1,
                  features=[0.0] * 29, action_mask=7)
    source["repeats"] = []
    for repeat in range(8):
        outcomes = []
        for kind in range(3):
            after = dict(before)
            after["novelty"] = .5 if kind == 0 else (.9 if repeat < 4 else .3) if kind == 1 else .52
            outcomes.append({"action": {"kind": kind, "target": 1,
                                        "source": None if kind == 0 else 0 if kind == 1 else 2},
                "before": before, "after": after, "cost": 0.0, "raw_sha256": str(kind) * 64})
        source["repeats"].append({"replicate": repeat, "outcomes": {"4": outcomes}})
    return source


def receipt(source, conditioned=True):
    reward, raw_target, scale, target = A.targets(source, conditioned)
    result = bytearray(104 if conditioned else 88)
    struct.pack_into("<QIIf", result, 0, 1, 4, 7, 0)
    struct.pack_into("<3f", result, 20, *reward)
    struct.pack_into("<3f", result, 32, *target)
    loss = A.huber([0.0] * 3, target, 7)
    struct.pack_into("<2d", result, 72, loss, loss)
    if conditioned:
        struct.pack_into("<f", result, 88, .001)
        struct.pack_into("<d", result, 96, scale)
    return result


def canonical_life():
    raw = bytearray(2296)
    raw[:8] = b"NTSPA001"
    struct.pack_into("<4I", raw, 8, 2296, 1, 1, 1)
    # A hand-built decoding fixture: zero hidden weights and known biases.
    # It tests the canonical byte layout, not native Agent initialization.
    struct.pack_into("<f", raw, 88 + (240 + 9 + 8) * 4, .25)
    struct.pack_into("<f", raw, 88 + (240 + 18 + 8) * 4, -.25)
    struct.pack_into("<Q", raw, 2288, A.fnv(raw[:2288]))
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    failures = []

    def reject(name, call):
        try:
            call()
        except (AssertionError, ValueError, EOFError) as error:
            failures.append({"name": name, "caught": True, "message": str(error)})
        else:
            raise AssertionError("deliberate defect escaped: " + name)

    source = sample()
    raw = receipt(source)
    decoded = A.validate_receipt(raw.hex(), source, source, "conditioned_mean8", 0)
    A.require(decoded["targets"][0] == 0 and decoded["targets"][1] == 1 and
              abs(decoded["targets"][2] - .2) < 2e-6, "mean-then-normalize fixture")
    raw_receipt = receipt(source, False)
    A.validate_receipt(raw_receipt.hex(), source, source, "raw_mean8", 0)
    tiny = copy.deepcopy(source)
    for repeat in tiny["repeats"]:
        for value in repeat["outcomes"]["4"]:
            value["after"]["novelty"] = .5 + (value["action"]["kind"] * .00001)
    floored = A.validate_receipt(receipt(tiny).hex(), tiny, tiny, "conditioned_mean8", 0)
    A.require(floored["scale"] == A.f32(.001) and 0 < floored["targets"][2] < .01, "tiny effects retain relative magnitude below floor")
    zero = copy.deepcopy(source)
    for repeat in zero["repeats"]:
        for value in repeat["outcomes"]["4"]:
            value["after"] = dict(value["before"])
    A.require(A.targets(zero, True)[3] == [0.0] * 3, "zero effect floor finite")
    bad = bytearray(raw); struct.pack_into("<f", bad, 36, -1)
    reject("reversed_credit", lambda: A.validate_receipt(bad.hex(), source, source, "conditioned_mean8", 0))
    bad_scale = bytearray(raw); struct.pack_into("<d", bad_scale, 96, 1.0)
    reject("forged_scale", lambda: A.validate_receipt(bad_scale.hex(), source, source, "conditioned_mean8", 0))
    bypassed = bytearray(raw); struct.pack_into("<3f", bypassed, 32, *A.targets(source, False)[3])
    reject("bypassed_conditioning", lambda: A.validate_receipt(bypassed.hex(), source, source, "conditioned_mean8", 0))
    per_draw = []
    for repeat in source["repeats"]:
        reward = [A.raw_reward(value) for value in repeat["outcomes"]["4"]]
        scale = max(A.f32(.001), max(abs(value - reward[0]) for value in reward))
        per_draw.append([(value - reward[0]) / scale for value in reward])
    early = bytearray(raw)
    struct.pack_into("<3f", early, 32, *[sum(draw[k] for draw in per_draw) / 8 for k in range(3)])
    reject("normalization_before_mean", lambda: A.validate_receipt(early.hex(), source, source, "conditioned_mean8", 0))
    lost_reward = bytearray(raw); struct.pack_into("<f", lost_reward, 24, 0)
    reject("lost_raw_reward", lambda: A.validate_receipt(lost_reward.hex(), source, source, "conditioned_mean8", 0))
    wrong_loss = bytearray(raw); struct.pack_into("<d", wrong_loss, 72, 0)
    reject("forged_loss", lambda: A.validate_receipt(wrong_loss.hex(), source, source, "conditioned_mean8", 0))
    wrong_source = dict(source, life_hash="0000000000000002")
    reject("wrong_source", lambda: A.validate_receipt(raw.hex(), wrong_source, source, "conditioned_mean8", 0))
    reject("wrong_donor_credit", lambda: A.validate_receipt(raw.hex(), source, tiny, "shuffled_conditioned", 0))
    native = A.comparison_bytes(source, source, 0)
    A.require(len(native) == 232 and struct.unpack_from("<QII", native) == (1, 4, 7), "typed comparison header")
    A.require(struct.unpack_from("<III", native, 16 + 72) == (1, 1, 0), "typed LEFT coordinates")
    life_raw = canonical_life()
    policy = A.decode_life(life_raw)
    A.require(A.predict(policy, source["features"]) == [0.0, .25, -.25], "267-weight canonical decode fixture")
    row = {"type": "readout", "source": A.source(source), "life_hash": policy["life_hash"],
        "action_mask": 7, "action": {"kind": 1, "target": 1, "source": 0}, "scores": [0.0, .25, -.25],
        "readout_hex": struct.pack("<IIII3f", 1, 1, 0, 7, 0, .25, -.25).hex()}
    A.require(A.validate_readout(row, source, policy) == (1, 0.0), "independent native typed choice")
    wrong_choice = copy.deepcopy(row); wrong_choice["action"]["kind"] = 2
    reject("wrong_typed_choice", lambda: A.validate_readout(wrong_choice, source, policy))
    wrong_scores = copy.deepcopy(row); wrong_scores["scores"][1] = .5
    reject("wrong_policy_readout", lambda: A.validate_readout(wrong_scores, source, policy))
    wrong_life = bytearray(life_raw); wrong_life[100] ^= 1
    reject("altered_life_checksum", lambda: A.decode_life(wrong_life))
    reject("duplicate_json_field", lambda: A.parse('{"x":1,"x":2}'))
    reject("nonfinite_json", lambda: A.parse('{"x":NaN}'))
    measured = {key: source[key] for key in ("seed", "episode", "step", "snapshot", "target", "life_hash", "agent_rng", "body_hash")}
    A.measurement_source(measured, source)
    changed_metadata = dict(measured, agent_rng=123)
    reject("forged_raw_source_metadata", lambda: A.measurement_source(changed_metadata, source))
    with tempfile.TemporaryDirectory(prefix="spa-conditioned-audit-") as directory:
        directory = Path(directory)
        reject("empty_off_on_saved_lives", lambda: A.ordinary_lives(directory, 42))
        for index in range(5):
            for mode in ("off", "on"):
                (directory / f"{mode}-s42.fixture{index}.life.bin").write_bytes(bytes([index]))
        A.ordinary_lives(directory, 42)
    output = {"schema": 1, "status": "PASS", "generation_runs": 0, "native_fit_runs": 0,
        "protocol_sha256": A.PROTOCOL_SHA,
        "command": "python3 tests/test_spa_conditioned_audit.py --output experiments/spa_agent/conditioned/audit.json",
        "sources": {"experiments/spa_agent/conditioned/audit.py": A.identity(Path(A.__file__)),
                    "tests/test_spa_conditioned_audit.py": A.identity(Path(__file__))},
        "gates": ["mean before conditioning", "tiny effects retain floor magnitude", "zero effects finite",
            "raw and conditioned native receipt layout", "typed consequence bytes", "canonical267-weight decode",
            "independent saved-weight typed choice", "raw source metadata equality", "all five OFF/ON life pairs"], "named_red_hands": failures,
        "scope": "Hand-built arithmetic and byte fixtures; no experiment fitting or body generation.",
        "frozen_before_experiment_fitting": True}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "red_hands": len(failures), "gates": len(output["gates"])}))


if __name__ == "__main__":
    main()
