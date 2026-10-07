#!/usr/bin/env python3
"""Independent future-credit source, target, native receipt and seal gates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spa_future", ROOT / "experiments/spa_agent/future/run.py")
FUTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FUTURE)


def synthetic_samples() -> list[dict]:
    """48 invented compact states with opposite H0/H4 preferences; no body run."""
    result = []
    for seed in (42, 73):
        for sid in range(24):
            episode, step = divmod(sid, 4)
            target = (episode + step) % 4
            index = len(result)
            features = [0.0] * 29
            features[:4] = [(target - 1.5) / 3, (step - 1.5) / 3, (episode - 2.5) / 5, -.5 if seed == 42 else .5]
            features[4:19] = [.5, .2 if target else 0, .2 if target < 3 else 0,
                .5, .5, 0, .25, .3, target / 3, 0, .05, float(target > 0), float(target < 3), .25, .25]
            sample = {"ordinal": index, "seed": seed, "episode": episode, "step": step, "snapshot": sid,
                "target": target, "sentence_count": 4, "life_hash": f"{1000 + index:016x}",
                "body_hash": "1234567890abcdef", "agent_rng": 1,
                "host_rng": f"{FUTURE.SCENARIOS.stream(seed, 0x7370615f6163746e, episode, step):016x}",
                "features": [FUTURE.f32(value) for value in features],
                "snapshot_raw_sha256": hashlib.sha256(f"synthetic snapshot {index}".encode()).hexdigest(),
                "ordinary_decision_raw_sha256": hashlib.sha256(f"synthetic ordinary {index}".encode()).hexdigest(),
                "action_mask": FUTURE.action_mask(target, 4), "outcomes": {}}
            sample["feature_witness"] = FUTURE.feature_witness(sample)
            for horizon in FUTURE.MEASURED_HORIZONS:
                outcomes = []
                preferred = (1 if target else 2) if horizon else (2 if target < 3 else 1)
                for kind in sorted(FUTURE.SCENARIOS.valid_actions(target, 4)):
                    before = {axis: .5 for axis in FUTURE.AXES}
                    before["repetition"] = before["collapse"] = 0
                    after = before.copy()
                    if kind: after["coherence"] = 1.0 if kind == preferred else 0.0
                    cost = 0.0 if horizon == 0 and kind == 0 else .1 if horizon == 0 else .25 + .05 * (kind != 0)
                    action = {"kind": kind, "target": target, "source": None if kind == 0 else target + (-1 if kind == 1 else 1)}
                    outcomes.append({"action": action, "before": before, "after": after, "cost": cost,
                        "reward": FUTURE.SCENARIOS.reward(before, after, cost),
                        "raw_sha256": hashlib.sha256(f"synthetic outcome {index}/{horizon}/{kind}".encode()).hexdigest(),
                        "initial_generated": 0 if kind == 0 else 16, "cumulative_generated": int(cost * 64 * (horizon + 1)),
                        "chain": [{"length": 12, "terminated": True, "tokens": [(i + j) % 94 for j in range(12)]} for i in range(4)]})
                sample["outcomes"][horizon] = outcomes
            result.append(sample)
    return result


def expect_failure(name: str, operation, expected: str) -> dict:
    try:
        operation()
    except (AssertionError, ValueError, KeyError) as error:
        message = str(error)
        FUTURE.require(expected in message, f"{name}: wrong gate failure: {message}")
        return {"name": name, "caught": True, "failure": message}
    raise AssertionError(name + ": defect escaped")


def invoke(command: list, cwd=ROOT) -> dict:
    command = list(map(str, command))
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    report = {"passed": False, "parser_gates": []}
    protocol = json.loads(FUTURE.PROTOCOL.read_text())
    FUTURE.require(FUTURE.V1.digest(FUTURE.PROTOCOL) == FUTURE.FROZEN_PROTOCOL_SHA256, "future protocol changed")
    archive = ROOT / protocol["parents"]["raw_traces.jsonl.gz"]["path"]
    old_hash = protocol["parents"]["raw_traces.jsonl.gz"]["sha256"]
    streams = FUTURE.archive_streams(archive, old_hash)
    samples = FUTURE.samples_from_streams(streams, protocol["training"]["seeds"], json.loads(FUTURE.SCENARIOS.PROTOCOL.read_text()))
    FUTURE.verify_source_association(samples, samples)
    report["retained_source"] = {"samples": len(samples), "archive_sha256": old_hash,
                                  "feature_witnesses": [sample["feature_witness"] for sample in samples]}
    report["parser_gates"].append(expect_failure("archive_identity", lambda: FUTURE.archive_streams(archive, "0" * 64), "archive SHA256"))
    mutations = {
        "feature_reassociation": lambda x: x[0].__setitem__("features", x[1]["features"]),
        "feature_reassociation_rechecksummed": lambda x: (x[0].__setitem__("features", x[1]["features"]), x[0].__setitem__("feature_witness", FUTURE.feature_witness(x[0]))),
        "outcome_reassociation": lambda x: x[0].__setitem__("outcomes", x[1]["outcomes"]),
        "outcome_identity": lambda x: x[0]["outcomes"][4][0].__setitem__("raw_sha256", "0" * 64),
        "coordinate": lambda x: x[0].__setitem__("target", 1),
        "mask": lambda x: x[0].__setitem__("action_mask", 7),
        "raw_axis": lambda x: x[0]["outcomes"][4][0]["after"].__setitem__("coherence", .25),
    }
    for name, alter in mutations.items():
        changed = copy.deepcopy(samples); alter(changed)
        report["parser_gates"].append(expect_failure(name, lambda: FUTURE.verify_source_association(changed, samples), "source feature/outcome association"))
    donors = FUTURE.shuffle_donors(samples)
    FUTURE.require(sorted(donors) == list(range(48)), "donor mapping is not a permutation")
    for i, donor in enumerate(donors):
        FUTURE.require(donor != i and samples[donor]["target"] == samples[i]["target"], "donor changed typed target")
    report["donors"] = donors
    with tempfile.TemporaryDirectory(prefix="notorch-spa-future-") as directory:
        temp = Path(directory)
        table = temp / "synthetic.txt"
        synthetic = synthetic_samples()
        FUTURE.write_dataset(table, synthetic, FUTURE.FROZEN_PROTOCOL_SHA256, "2" * 64)
        report["synthetic_dataset_sha256"] = FUTURE.V1.digest(table)
        if args.binary:
            binary = args.binary.resolve()
        else:
            binary = temp / "spa_agent_future"
            command = shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-O2", "-pthread", "-I", ROOT,
                ROOT / "examples/spa_agent_future.c", ROOT / "spa_agent.c", ROOT / "notorch.c", "-lm", "-o", binary]
            report["build"] = invoke(command)
            FUTURE.require(report["build"]["returncode"] == 0, "native future fixture compilation failed: " + report["build"]["stderr"])
        prefix = temp / "fit"
        report["train"] = invoke([binary, "train", table, prefix])
        FUTURE.require(report["train"]["returncode"] == 0, "synthetic native train failed: " + report["train"]["stderr"])
        fit_path = Path(str(prefix) + ".fit.jsonl")
        fit_rows, _ = FUTURE.parse_raw(fit_path.read_bytes())
        fit = FUTURE.validate_fit(fit_rows, synthetic, table, protocol, "2" * 64)
        report["fit"] = {k: v for k, v in fit.items() if k != "per_epoch"}
        report["fit"]["trace_sha256"] = FUTURE.V1.digest(fit_path)
        first_fit = next(i for i, row in enumerate(fit_rows) if row["type"] == "fit")
        report["fit_receipt_gates"] = []
        for name, field, value, expected in (
            ("horizon_substitution", "horizon", 0, "fit horizon differs"),
            ("undeclared_donor", "donor_index", 7, "declared donor mapping differs"),
            ("target_sign", "targets", [0, 0, -.25], "fit targets differs"),
            ("missing_tail", None, None, "complete final fit summary")):
            altered = fit_rows.copy()
            if field:
                altered[first_fit] = {**altered[first_fit], field: value}
            else:
                altered = altered[:-1]
            report["fit_receipt_gates"].append(expect_failure(name,
                lambda: FUTURE.validate_fit(altered, synthetic, table, protocol, "2" * 64), expected))
        lives = {label: Path(f"{prefix}.{label}.life.bin") for label in FUTURE.POLICIES}
        saved = {label: path.read_bytes() for label, path in lives.items()}
        trace = temp / "readout.jsonl"
        report["read"] = invoke([binary, "read", table, prefix, trace])
        FUTURE.require(report["read"]["returncode"] == 0, "synthetic native read failed: " + report["read"]["stderr"])
        rows, _ = FUTURE.parse_raw(trace.read_bytes())
        readouts = FUTURE.validate_readout(rows, synthetic, table, "2" * 64, fit["life_hashes"], table, "2" * 64)
        report["readouts"] = len(readouts)
        first_read = next(i for i, row in enumerate(rows) if row["type"] == "readout")
        report["readout_receipt_gates"] = []
        for name, field, value, expected in (
            ("selected_branch_identity", "selected_outcome_sha256", "0" * 64, "executed branch"),
            ("readout_state_mutation", "life_unchanged", False, "readout mutated"),
            ("readout_reward", "selected_reward", .9, "selected consequence differs"),
            ("readout_missing", None, None, "incomplete native readout")):
            altered = rows.copy()
            if field: altered[first_read] = {**altered[first_read], field: value}
            else: altered.pop(first_read)
            report["readout_receipt_gates"].append(expect_failure(name,
                lambda: FUTURE.validate_readout(altered, synthetic, table, "2" * 64, fit["life_hashes"], table, "2" * 64), expected))
        seal = {"status": "SEALED", "protocol_sha256": FUTURE.FROZEN_PROTOCOL_SHA256,
            "training_dataset": table.name, "fit_trace": fit_path.name, "native_seal": "fit.seal",
            "lives": {label: {"file": path.name, **FUTURE.artifact(path)} for label, path in lives.items()},
            "files": {path.name: FUTURE.artifact(path) for path in (table, fit_path, temp / "fit.seal", *lives.values())}}
        FUTURE.verify_seal(seal, temp)
        report["seal_gates"] = []
        missing = copy.deepcopy(seal); missing["lives"].pop("h0")
        report["seal_gates"].append(expect_failure("missing_saved_life", lambda: FUTURE.verify_seal(missing, temp), "incomplete sealed life"))
        no_fit = copy.deepcopy(seal); no_fit["files"].pop(fit_path.name)
        report["seal_gates"].append(expect_failure("unsealed_fit_receipts", lambda: FUTURE.verify_seal(no_fit, temp), "file coverage differs"))
        h4 = lives["h4"]; original_h4 = h4.read_bytes(); h4.write_bytes(original_h4[:-1])
        report["seal_gates"].append(expect_failure("changed_saved_life", lambda: FUTURE.verify_seal(seal, temp), "sealed training artifact changed"))
        h4.write_bytes(original_h4); FUTURE.verify_seal(seal, temp)
        outcomes_changed = copy.deepcopy(synthetic)
        for sample in outcomes_changed:
            for outcome in sample["outcomes"][4]:
                outcome["after"]["coherence"] = .25
                outcome["cost"] = .5
                outcome["reward"] = FUTURE.SCENARIOS.reward(outcome["before"], outcome["after"], outcome["cost"])
                outcome["raw_sha256"] = hashlib.sha256((outcome["raw_sha256"] + "changed").encode()).hexdigest()
        changed_table = temp / "outcomes-changed.txt"
        FUTURE.write_dataset(changed_table, outcomes_changed, FUTURE.FROZEN_PROTOCOL_SHA256, "3" * 64)
        changed_trace = temp / "outcomes-changed.jsonl"
        report["outcome_independence"] = invoke([binary, "read", changed_table, prefix, changed_trace])
        FUTURE.require(report["outcome_independence"]["returncode"] == 0, "changed-outcome read refused")
        changed_rows, _ = FUTURE.parse_raw(changed_trace.read_bytes())
        changed_readouts = FUTURE.validate_readout(changed_rows, outcomes_changed, changed_table, "3" * 64,
                                                  fit["life_hashes"], table, "2" * 64)
        FUTURE.require(all(a["action"] == b["action"] and a["scores_before"] == b["scores_before"]
                           for a, b in zip(readouts, changed_readouts)), "readout choice depends on supplied outcomes")
        FUTURE.require(all(path.read_bytes() == saved[label] for label, path in lives.items()), "readout changed saved life bytes")
        report["outcome_independence"]["same_choices_scores_and_saved_lives"] = True
        encoded = table.read_text()
        first_features = next(line for line in encoded.splitlines() if line.startswith("FEATURES "))
        altered_features = first_features.split(); altered_features[1] = "0.125"
        malformed = {
            "truncation": encoded[:len(encoded) // 2],
            "trailing_data": encoded + "EXTRA\n",
            "feature_witness": encoded.replace(first_features, " ".join(altered_features), 1),
            "nonfinite_feature": encoded.replace(first_features, first_features.replace("-0.5", "nan", 1), 1),
            "horizon": encoded.replace("HORIZON 4", "HORIZON 1", 1),
            "mask": encoded.replace("HORIZON 0 5", "HORIZON 0 7", 1),
            "typed_source": encoded.replace("OUTCOME 2 0 1", "OUTCOME 2 0 2", 1),
        }
        report["native_parser_gates"] = []
        for name, text in malformed.items():
            bad = temp / (name + ".txt"); bad.write_text(text)
            bad_trace = temp / (name + ".jsonl")
            result = invoke([binary, "read", bad, prefix, bad_trace])
            FUTURE.require(result["returncode"] == 1 and result["stderr"].startswith("spa_agent_future:")
                           and not bad_trace.exists(), "native malformed dataset escaped: " + name)
            report["native_parser_gates"].append({"name": name, "caught": True, **result})
        saved_trace = trace.read_bytes()
        collision = invoke([binary, "read", table, prefix, trace])
        FUTURE.require(collision["returncode"] == 1 and "destination must not exist" in collision["stderr"]
                       and trace.read_bytes() == saved_trace, "native readout replaced an existing trace")
        report["existing_trace_preserved"] = collision
        report["partial_trace_gates"] = FUTURE.TRACE_IO.self_test()
    report["passed"] = True
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"SPA_FUTURE_GATES_OK retained={len(samples)} association_mutations={len(report['parser_gates'])} "
          f"native_parser_mutations={len(report['native_parser_gates'])} fits={report['fit']['fits']} readouts={report['readouts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
