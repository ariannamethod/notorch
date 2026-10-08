#!/usr/bin/env python3
"""Synthetic conditioned-helper transcripts, source association and readout gates."""
from __future__ import annotations

import argparse
import copy
import ctypes as C
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURE = load("spa_conditioned_fixture", ROOT / "tests/test_spa_replicates.py")
POLICY = load("spa_conditioned_policy", ROOT / "experiments/spa_agent/conditioned/policy.py")
RUN = FIXTURE.RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    commands, failures = [], []
    source_names = POLICY.SOURCES + ("tests/test_spa_conditioned_policy.py",)
    source_ids = {name: POLICY.identity(ROOT / name)["sha256"] for name in source_names}
    with tempfile.TemporaryDirectory(prefix="spa-conditioned-policy-") as directory:
        work = Path(directory)
        library = args.library.resolve() if args.library else work / "libnotorch.so"

        def normalize(value):
            value = str(value).replace(str(ROOT), "<repo>").replace(str(work), "<tmp>")
            return value.replace(str(library), "<library>").replace(str(library.parent), "<library-dir>")

        def invoke(argv):
            process = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                                     text=True, timeout=240)
            commands.append({"argv": [normalize(value) for value in argv],
                "returncode": process.returncode, "stdout": normalize(process.stdout),
                "stderr": normalize(process.stderr)})
            POLICY.require(process.returncode == 0, "command failed: " + process.stderr)
            return process.stdout

        def reject(name, operation, fragment):
            try:
                operation()
            except (AssertionError, ValueError, EOFError) as error:
                POLICY.require(fragment in str(error), name + ": wrong failure: " + str(error))
                failures.append({"name": name, "caught": True, "failure": str(error)})
            else:
                raise AssertionError(name + ": deliberate defect escaped")

        fixture = work / "fixture"
        invoke(["cc", "-std=c11", "-O2", "-I.", "tests/spa_replicates_fixture.c",
                "spa_agent.c", "notorch.c", "-lm", "-o", fixture])
        if not args.library:
            invoke(["cc", "-std=c11", "-O2", "-fPIC", "-shared", "-I.",
                    "spa_binding.c", "spa_agent.c", "notorch.c", "-lm", "-o", library])
        prefix = work / "source"
        invoke([fixture, "write", prefix])
        host_trace = work / "synthetic-host.jsonl"
        invoke([fixture, "run", str(prefix) + ".sources.txt", host_trace])
        sources, references = FIXTURE.fixture_inputs(prefix)
        projected, host = RUN.validate_replicates(host_trace.read_bytes(), sources, references,
            dataset_fnv=RUN.fnv(Path(str(prefix) + ".sources.txt").read_bytes()))
        invented = FIXTURE.synthetic_samples(projected)
        protocol_sha = "f" * 64
        dataset = work / "synthetic.json"
        dataset.write_text(json.dumps({"schema": 1, "protocol_sha256": protocol_sha,
                                       "replicates": 8, "samples": invented}) + "\n")
        POLICY.require(POLICY.read_dataset(dataset, protocol_sha, training=True) == invented,
                       "synthetic dataset import changed")
        native = POLICY.SPA.Native(library)
        policies = work / "policies"
        seal = POLICY.fit_policies(invented, native, policies, epochs=2,
            protocol_sha=protocol_sha, dataset_identity=POLICY.identity(dataset), require_parent_parity=False)
        verified, agents = POLICY.verify_seal(policies, native, protocol_sha, synthetic=True)
        POLICY.require(seal["trace"] == verified["trace"], "journal seal changed")
        with gzip.open(policies / "fit.jsonl.gz", "rb") as stream:
            rows = [json.loads(line) for line in stream]
        POLICY.require(len(rows) == 433, "synthetic journal coverage changed")
        POLICY.require(rows[0]["abi"] == POLICY.SPA._layout(), "journal ABI layout changed")
        expected_types = {arm: {"name": POLICY.receipt_type(arm).__name__,
                               "bytes": C.sizeof(POLICY.receipt_type(arm))} for arm in POLICY.ARMS[1:]}
        POLICY.require(rows[0]["receipt_types"] == expected_types, "journal receipt sizes changed")
        comparisons = {row["comparison_id"]: row for row in rows if row["type"] == "comparison"}
        POLICY.require(len(comparisons) == 144, "comparison association coverage changed")
        replayed = {arm: POLICY.SPA.Agent.from_file(policies / "initial.life", native=native)
                    for arm in POLICY.ARMS[1:]}
        initial = replayed["raw_mean8"].state
        donors = POLICY.shuffle_donors(invented)
        replay_count = 0
        for row in (value for value in rows if value["type"] == "fit"):
            arm, index = row["arm"], row["row"]
            agent, sample = replayed[arm], invented[index]
            association = comparisons[row["comparison_id"]]
            donor = donors[index] if arm == "shuffled_conditioned" else index
            POLICY.require(row["source_feature_witness"] == sample["feature_witness"] and
                           row["donor_row"] == donor, "source/donor update association changed")
            POLICY.require(association["source"] == POLICY.source_identity(sample) and
                           association["donor"] == POLICY.source_identity(invented[donor]),
                           "comparison source/donor identity changed")
            captured = POLICY.experience(sample)
            values, references = POLICY.comparisons(sample, invented[donor], 8)
            POLICY.require(association["repeats"] == references and
                           association["native_input_hex"] == [bytes(value).hex() for value in values],
                           "raw outcome/native input association changed")
            POLICY.require(row["life_before"] == f"{agent.hash:016x}" and
                           row["policy_before_sha256"] == hashlib.sha256(bytes(agent.state.policy)).hexdigest(),
                           "pre-update identity changed")
            receipt = POLICY.fit(agent, captured, values, .03, .001, arm)
            POLICY.require(bytes(POLICY.decode_receipt(arm, row["receipt_hex"])) == bytes(receipt),
                           "replayed receipt bytes changed")
            POLICY.require(row["life_after"] == f"{agent.hash:016x}" and
                           row["policy_after_sha256"] == hashlib.sha256(bytes(agent.state.policy)).hexdigest(),
                           "post-update identity changed")
            POLICY.policy_only(agent, initial)
            replay_count += 1
        POLICY.require(replay_count == 288, "native receipt replay coverage changed")
        for arm, agent in replayed.items():
            POLICY.require(bytes(agent.state) == bytes(agents[arm].state), "replayed final life changed")
        parent_policies = work / "parent-policies"
        POLICY.PARENT.fit_policies(invented, native, parent_policies, epochs=2,
            protocol_sha=protocol_sha, dataset_identity=POLICY.identity(dataset), require_parent_parity=False)
        POLICY.require((parent_policies / "mean8_h4.life").read_bytes() ==
                       (policies / "raw_mean8.life").read_bytes(), "raw parent fixture life changed")
        clean, poisoned = io.StringIO(), io.StringIO()
        before = {arm: bytes(agent.state) for arm, agent in agents.items()}
        POLICY.score_samples(invented, agents, clean)
        deleted = copy.deepcopy(invented)
        for sample in deleted:
            del sample["outcomes"]
            del sample["repeats"]
        POLICY.score_samples(deleted, agents, poisoned)
        POLICY.require(clean.getvalue() == poisoned.getvalue(), "readout used measured outcomes")
        POLICY.require(before == {arm: bytes(agent.state) for arm, agent in agents.items()},
                       "readout changed full life bytes")
        bad = copy.deepcopy(invented[0])
        bad["features"][0] += .125
        reject("feature_association", lambda: POLICY.PARENT.validate_source(bad, 0), "feature-source witness")
        reject("wrong_donor", lambda: POLICY.comparisons(invented[0], invented[1], 8), "donor coordinates")
        truncated = work / "truncated.gz"
        truncated.write_bytes((policies / "fit.jsonl.gz").read_bytes()[:-5])
        reject("truncated_gzip", lambda: POLICY.journal_identity(truncated), "end-of-stream")
        bad_seal = copy.deepcopy(seal)
        bad_seal["fits_per_arm"] = 24576
        bad_seal["raw_mean8_parent_parity"] = "byte-identical canonical checkpoint"
        (policies / "seal.json").write_text(json.dumps(bad_seal) + "\n")
        reject("resealed_parent_impostor", lambda: POLICY.verify_seal(policies, native, protocol_sha),
               "retained parent")
    POLICY.require(source_ids == {name: POLICY.identity(ROOT / name)["sha256"] for name in source_names},
                   "synthetic gate source changed")
    result = {"passed": True, "body_runs": 0, "host_fixture": host, "fits": 288, "comparisons": 144,
        "exact_native_receipt_replays": replay_count, "journal": seal["trace"],
        "raw_parent_fixture_life_byte_parity": True, "pure_native_readouts": 192,
        "deleted_outcomes_unused": True, "full_life_bytes_unchanged": True,
        "named_red_hands": failures, "commands": commands, "source_sha256": source_ids}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "fits", "comparisons",
                     "exact_native_receipt_replays", "pure_native_readouts")}))


if __name__ == "__main__":
    main()
