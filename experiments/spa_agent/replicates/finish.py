#!/usr/bin/env python3
"""Authenticate the remaining own-archive host trace and publish final manifests."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); snapshot = output / "source_snapshot"
    run = load("spa_frozen_finish", snapshot / "experiments/spa_agent/replicates/run.py")
    finalizer = load("spa_previous_finalizer", HERE / "finalize.py")
    atomic, require, identity = finalizer.atomic, run.require, run.identity
    metadata = json.loads((output / "metadata.json").read_text())
    barrier = json.loads((output / "all_eight_lives_sealed.json").read_text())
    recovery = json.loads((output / "recovery.json").read_text())
    dataset_recovery = json.loads((output / "dataset_recovery.json").read_text())
    native = run.POLICY.SPA.Native(output / "libnotorch.so")
    self_id = identity(Path(__file__))
    def stable():
        for name, expected in metadata["source_files_sha256"].items():
            require(run.V1.digest(ROOT / name) == run.V1.digest(snapshot / name) == expected, "measured source changed")
        for name, expected in metadata["binaries"].items(): require(identity(output / name) == expected, "measured binary changed")
        for name, expected in barrier["input_identities"].items(): require(identity(output / "inputs" / name) == expected, "input changed")
        for record, key in ((recovery, "recovery_source"), (dataset_recovery, "source")):
            source = record[key]
            require(identity(ROOT / source["path"]) == {k: source[k] for k in ("bytes", "sha256")}, "prior recovery source changed")
        require(identity(Path(__file__)) == self_id, "final manifest writer changed")
        run.verify_barrier(barrier, output, native)
    stable()
    directories = [output / f"execution-{i}" for i in (1, 2)]
    results = [json.loads((directory / "result.json").read_text()) for directory in directories]
    for i, directory in enumerate(directories, 1):
        require((directory / "result.json").read_bytes() ==
                (output / f"failure-lane/execution-{i}.result.before-finalization.json").read_bytes(), "original result manifest changed")
    all_streams = []
    original_artifact_checks = 0
    for index, (directory, result) in enumerate(zip(directories, results)):
        for name, expected in result["artifacts"].items():
            if index == 0 and name == "evaluation.json":
                require(identity(directory / name) == {k: dataset_recovery["member"][k] for k in ("bytes", "sha256")}, "restored dataset changed")
            else: require(identity(directory / name) == expected, "original artifact changed: " + name)
            original_artifact_checks += 1
        archive = directory / "raw_traces.jsonl.gz"
        require(identity(archive) == result["artifacts"][archive.name], "enclosing original archive pin changed")
        streams = {}
        for line in gzip.decompress(archive.read_bytes()).splitlines():
            item = json.loads(line, object_pairs_hook=run.SCENARIOS.unique_object)
            require(set(item) == {"seed", "stream", "raw"}, "archive envelope changed")
            streams.setdefault((item["seed"], item["stream"]), bytearray()).extend(item["raw"].encode())
        all_streams.append({key: bytes(value) for key, value in streams.items()})
    require(all_streams[0] == all_streams[1], "complete archived executions differ")
    mappings = {(0, "fit"): "policies/fit.jsonl", (0, "policy_seal"): "policies/seal.json"}
    for split in ("training", "evaluation"):
        mappings[0, split + "_dataset"] = split + ".json"
        mappings[0, split + "_readout"] = split + ".readout.jsonl"
        for seed in ([42, 73] if split == "training" else [307, 401]):
            mappings[seed, split + "_replicates"] = f"{split}-repeats-s{seed}.jsonl"
    for seed in (307, 401):
        for stream, prefix in (("ordinary_off", "off"), ("ordinary_on", "on"), ("scenarios", "scenarios")):
            mappings[seed, stream] = f"{prefix}-s{seed}.jsonl"
    expected_keys = set(mappings) | {(seed, "training_parent_" + kind) for seed in (42, 73)
                                    for kind in ("ordinary_off", "ordinary_on", "scenarios")}
    require(set(all_streams[0]) == expected_keys, "complete archive stream coverage")
    known = (1, "off-s307.jsonl")
    for index, (directory, streams) in enumerate(zip(directories, all_streams)):
        for key, name in mappings.items():
            if (index, name) != known:
                require((directory / name).read_bytes() == streams[key], "additional standalone/archive discrepancy: " + name)
    directory = directories[1]; target = directory / "off-s307.jsonl"
    damaged = target.read_bytes(); complete = all_streams[1][307, "ordinary_off"]
    require(len(damaged) == 211683 and run.sha256(damaged) ==
            "6d6792ad20dc5c50eed3c75d811140c6bc826d59afbe405f15d3ec506bf932d2", "registered host damage changed")
    require(len(complete) == 223818 and run.sha256(complete) ==
            "30c9c9d1acb5fee8a308b3942f000bd01e978c1c38a4398eac77e5623c2991bb" and complete.startswith(damaged),
            "own-archive host member identity changed")
    mirror = directory / "off-s307.verified.jsonl.gz"
    require(identity(mirror) == results[1]["artifacts"][mirror.name] and gzip.decompress(mirror.read_bytes()) == complete,
            "original immediate verified mirror differs")
    require(all_streams[1][307, "ordinary_on"] == complete, "original diagnostics parity changed")
    ordinary_check = run.FUTURE.TRACE_IO.validate(complete, 307)
    ordinary_rows, _ = run.parse_raw(complete)
    scenario_rows, _ = run.parse_raw(all_streams[1][307, "scenarios"])
    protocol = json.loads((snapshot / "experiments/spa_agent/scenarios/protocol.json").read_text()); protocol["seeds"] = [307]
    scenario_check = run.SCENARIOS.validate_seed(scenario_rows, ordinary_rows, protocol)
    preserved = output / "failure-lane/execution-2.off-s307.truncated.jsonl.gz"
    atomic(preserved, gzip.compress(damaged, mtime=0)); atomic(target, complete)
    host_recovery = {"schema": 1, "status": "PINNED_OWN_ARCHIVE_AND_IMMEDIATE_MIRROR_HOST_RESTORATION", "execution": 2,
        "source": {"path": "experiments/spa_agent/replicates/finish.py", **self_id},
        "observed_error": "AssertionError: complete execution repeat differs: off-s307.jsonl",
        "original_result": {"path": "failure-lane/execution-2.result.before-finalization.json",
                            **identity(output / "failure-lane/execution-2.result.before-finalization.json")},
        "enclosing_archive": {"path": "execution-2/raw_traces.jsonl.gz", **results[1]["artifacts"]["raw_traces.jsonl.gz"]},
        "member": {"seed": 307, "stream": "ordinary_off", "bytes": len(complete), "sha256": run.sha256(complete)},
        "immediate_mirror": {"path": "execution-2/off-s307.verified.jsonl.gz", **identity(mirror)},
        "damaged": {"bytes": len(damaged), "sha256": run.sha256(damaged), "exact_prefix": True},
        "damaged_preserved": {"path": str(preserved.relative_to(output)), **identity(preserved)},
        "ordinary_trace_complete": ordinary_check, "source_measurement_association_checked": True,
        "scenario_counts": {key: scenario_check[key] for key in ("snapshots", "alternatives", "measurements")},
        "new_generation": False, "new_fitting": False, "original_artifact_pins_checked": original_artifact_checks,
        "all_other_archive_standalone_members_identical": True}
    run.write_json(output / "host_trace_recovery.json", host_recovery)
    for directory, streams, result in zip(directories, all_streams, results):
        for key, name in mappings.items(): require((directory / name).read_bytes() == streams[key], "final archive/standalone differs")
        result["artifacts"] = {name: identity(directory / name) for name in result["artifacts"]}
    require(set(results[0]["artifacts"]) == set(results[1]["artifacts"]), "final execution artifact coverage")
    comparable = [name for name in results[0]["artifacts"] if not name.endswith((".stdout.txt", ".stderr.txt", ".command.json"))]
    for name in comparable:
        require((directories[0] / name).read_bytes() == (directories[1] / name).read_bytes(), "final repeated artifact differs: " + name)
    require(results[0]["training"] == results[1]["training"] and results[0]["evaluation"] == results[1]["evaluation"], "final summaries differ")
    stable()
    stages = json.loads((output / "stages.json").read_text())
    stages.append({"order": len(stages) + 1, "phase": "own_archives_restored_and_complete_two_execution_identity",
        "artifacts": len(comparable), "dataset_recovery_sha256": run.V1.digest(output / "dataset_recovery.json"),
        "host_trace_recovery_sha256": run.V1.digest(output / "host_trace_recovery.json")})
    atomic(output / "stages.json", (json.dumps(stages, indent=2) + "\n").encode())
    for directory, result in zip(directories, results):
        atomic(directory / "result.json", (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    atomic(output / "raw_traces.jsonl.gz", (directories[0] / "raw_traces.jsonl.gz").read_bytes())
    receipts = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages,
        "recovery": recovery, "dataset_recovery": dataset_recovery, "host_trace_recovery": host_recovery,
        "executions": results, "repeat": {"all_passed": True, "count": len(comparable), "byte_identical_artifacts": comparable},
        "artifacts": {name: identity(output / name) for name in
            ("raw_traces.jsonl.gz", "recovery.json", "dataset_recovery.json", "host_trace_recovery.json")}}
    encoded = json.dumps(receipts, allow_nan=False).replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    encoded = encoded.replace(str(sys.executable), "python3")
    atomic(output / "receipts.json", (json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    stable()
    for directory, result in zip(directories, results):
        for name, expected in result["artifacts"].items(): require(identity(directory / name) == expected, "final artifact changed")
    print(json.dumps({"status": "COMPLETE", "byte_identical_artifacts": len(comparable),
        "archive": identity(output / "raw_traces.jsonl.gz"), "receipts": identity(output / "receipts.json")}))


if __name__ == "__main__":
    main()
