#!/usr/bin/env python3
"""Restore one dataset from its pinned own archive and finalize existing results.

No generation or fitting occurs here. The failed standalone manifest remains
preserved; the complete dataset must also reconstruct from original raw forks.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def atomic(path, raw):
    temporary = path.with_name(path.name + ".verified-finalization.tmp")
    with temporary.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); snapshot = output / "source_snapshot"
    spec = importlib.util.spec_from_file_location("spa_finalization_frozen", snapshot / "experiments/spa_agent/replicates/run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    require, identity = run.require, run.identity
    metadata = json.loads((output / "metadata.json").read_text())
    barrier = json.loads((output / "all_eight_lives_sealed.json").read_text())
    recovery = json.loads((output / "recovery.json").read_text())
    self_id = identity(Path(__file__))
    library = output / "libnotorch.so"; native = run.POLICY.SPA.Native(library)
    def stable():
        for name, expected in metadata["source_files_sha256"].items():
            require(run.V1.digest(ROOT / name) == run.V1.digest(snapshot / name) == expected, "measured source changed")
        for name, expected in metadata["binaries"].items(): require(identity(output / name) == expected, "measured binary changed")
        for name, expected in barrier["input_identities"].items(): require(identity(output / "inputs" / name) == expected, "measured input changed")
        require(identity(Path(__file__)) == self_id, "finalization source changed")
        require(identity(ROOT / recovery["recovery_source"]["path"]) ==
                {key: recovery["recovery_source"][key] for key in ("bytes", "sha256")}, "first recovery source changed")
        run.verify_barrier(barrier, output, native)
    stable()
    executions = [output / f"execution-{i}" for i in (1, 2)]
    original_results = [(directory / "result.json").read_bytes() for directory in executions]
    results = [json.loads(raw) for raw in original_results]
    for directory, result in zip(executions, results):
        for name, expected in result["artifacts"].items(): require(identity(directory / name) == expected, "original failed manifest changed: " + name)
    directory = executions[0]; archive = directory / "raw_traces.jsonl.gz"
    enclosing_pin = results[0]["artifacts"]["raw_traces.jsonl.gz"]
    require(identity(archive) == enclosing_pin, "original enclosing archive pin changed")
    archived = bytearray(); member_records = 0
    for line in gzip.decompress(archive.read_bytes()).splitlines():
        item = json.loads(line, object_pairs_hook=run.SCENARIOS.unique_object)
        if item["seed"] == 0 and item["stream"] == "evaluation_dataset":
            archived.extend(item["raw"].encode()); member_records += 1
    complete = bytes(archived); require(member_records == 1, "archived dataset member coverage")
    dataset = json.loads(complete, object_pairs_hook=run.SCENARIOS.unique_object)
    damaged = (directory / "evaluation.json").read_bytes()
    require(len(damaged) == 2016208 and run.sha256(damaged) ==
            "843ad9b6b1e7d9006b440b2fa429b27935a3515f9a74b4337211268027bbed2b", "registered damaged dataset changed")
    require(len(complete) == 4723513 and run.sha256(complete) ==
            "0390b75e463657f4adb4eccaa54dfefb2a6699665d3dedb673dfd76d1ed7550a" and complete.startswith(damaged),
            "own-archive complete dataset identity or prefix differs")
    source_archive = directory / "evaluation_sources.raw.jsonl.gz"
    require(identity(source_archive) == results[0]["artifacts"][source_archive.name], "own source archive pin changed")
    source_sha = run.V1.digest(source_archive)
    streams = run.FUTURE.archive_streams(source_archive, source_sha)
    scenario_protocol = json.loads((snapshot / "experiments/spa_agent/scenarios/protocol.json").read_text())
    sources, references = run.source_samples(streams, [307, 401], source_sha, scenario_protocol)
    rebuilt, checks = [], []
    vocab_fnv = run.fnv((output / "inputs/vocabulary.u32").read_bytes())
    for seed in (307, 401):
        values, checked = run.validate_replicates((directory / f"evaluation-repeats-s{seed}.jsonl").read_bytes(),
            [s for s in sources if s["seed"] == seed], references,
            dataset_fnv=run.fnv((directory / f"evaluation-sources-s{seed}.txt").read_bytes()), vocabulary_fnv=vocab_fnv)
        rebuilt += values; checks.append(checked)
    rebuilt.sort(key=lambda value: value["ordinal"])
    candidate = {"schema": 1, "protocol_sha256": run.FROZEN_PROTOCOL_SHA256, "replicates": 8, "samples": rebuilt}
    canonical = (json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(canonical == complete, "own-source reconstruction differs from pinned archive member")
    seal, agents = run.POLICY.verify_seal(directory / "policies", native, run.FROZEN_PROTOCOL_SHA256)
    life_bytes = {label: bytes(agent.state) for label, agent in agents.items()}
    scored = io.StringIO(); run.POLICY.score_samples(rebuilt, agents, scored)
    require(scored.getvalue().encode() == (directory / "evaluation.readout.jsonl").read_bytes(), "original native readouts differ")
    require(life_bytes == {label: bytes(agent.state) for label, agent in agents.items()}, "audit readout changed policy life")
    rows, _ = run.parse_raw(scored.getvalue().encode())
    choices = run.validate_readouts(rows, rebuilt, {label: seal["lives"][label]["life_hash"] for label in run.ARMS})
    summary = run.summarize(rebuilt, choices, "evaluation")
    require(summary == results[0]["evaluation"] == json.loads((directory / "evaluation.summary.json").read_text()),
            "reconstructed dataset does not reproduce original summary")
    failure_lane = output / "failure-lane"
    atomic(failure_lane / "execution-1.evaluation.truncated.json.gz", gzip.compress(damaged, mtime=0))
    for number, raw in enumerate(original_results, 1):
        atomic(failure_lane / f"execution-{number}.result.before-finalization.json", raw)
    atomic(directory / "evaluation.json", complete)
    require((directory / "evaluation.json").read_bytes() == complete, "restored own dataset changed")
    dataset_recovery = {"schema": 1, "status": "PINNED_OWN_ARCHIVE_DATASET_RESTORATION", "execution": 1,
        "source": {"path": "experiments/spa_agent/replicates/finalize.py", **self_id},
        "evidence_boundary": "Original result manifest pins the complete enclosing archive but records the already-truncated standalone dataset. No earlier standalone dataset pin is claimed.",
        "original_result": {"path": "failure-lane/execution-1.result.before-finalization.json",
                            "bytes": len(original_results[0]), "sha256": run.sha256(original_results[0])},
        "enclosing_archive": {"path": "execution-1/raw_traces.jsonl.gz", **enclosing_pin},
        "member": {"seed": 0, "stream": "evaluation_dataset", "records": member_records,
                   "bytes": len(complete), "sha256": run.sha256(complete)},
        "damaged": {"bytes": len(damaged), "sha256": run.sha256(damaged), "exact_prefix": True},
        "damaged_preserved": {"path": "failure-lane/execution-1.evaluation.truncated.json.gz",
                               **identity(failure_lane / "execution-1.evaluation.truncated.json.gz")},
        "independently_reconstructed_from_own_raw_sources": True, "native_readouts_reproduced": len(rows),
        "original_summary_reproduced": True, "policy_life_bytes_unchanged": True,
        "new_generation": False, "new_fitting": False, "checks": checks,
        "observed_error": "AssertionError: complete repeated execution differs: evaluation.json"}
    run.write_json(output / "dataset_recovery.json", dataset_recovery)
    # The old manifest remains in failure-lane. This is an explicitly NEW final
    # manifest reflecting the authenticated restoration, not a relabeled pass.
    for directory, result in zip(executions, results):
        result["artifacts"] = {name: identity(directory / name) for name in result["artifacts"]}
    require(set(results[0]["artifacts"]) == set(results[1]["artifacts"]), "complete artifact coverage differs")
    comparable = [name for name in results[0]["artifacts"] if not name.endswith((".stdout.txt", ".stderr.txt", ".command.json"))]
    for name in comparable:
        require((executions[0] / name).read_bytes() == (executions[1] / name).read_bytes(), "complete execution repeat differs: " + name)
    require(results[0]["training"] == results[1]["training"] and results[0]["evaluation"] == results[1]["evaluation"], "complete summaries differ")
    stable()
    stages = json.loads((output / "stages.json").read_text())
    stages.append({"order": len(stages) + 1, "phase": "own_archive_dataset_restored_and_complete_two_execution_identity",
                   "artifacts": len(comparable), "dataset_recovery_sha256": run.V1.digest(output / "dataset_recovery.json")})
    atomic(output / "stages.json", (json.dumps(stages, indent=2) + "\n").encode())
    for directory, result in zip(executions, results):
        atomic(directory / "result.json", (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    atomic(output / "raw_traces.jsonl.gz", (executions[0] / "raw_traces.jsonl.gz").read_bytes())
    receipts = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages,
        "recovery": recovery, "dataset_recovery": dataset_recovery, "executions": results,
        "repeat": {"all_passed": True, "count": len(comparable), "byte_identical_artifacts": comparable},
        "artifacts": {name: identity(output / name) for name in ("raw_traces.jsonl.gz", "recovery.json", "dataset_recovery.json")}}
    text = json.dumps(receipts, allow_nan=False).replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    text = text.replace(str(sys.executable), "python3")
    atomic(output / "receipts.json", (json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    stable()
    for directory, result in zip(executions, results):
        for name, expected in result["artifacts"].items(): require(identity(directory / name) == expected, "final artifact changed")
    print(json.dumps({"status": "COMPLETE", "byte_identical_artifacts": len(comparable),
                      "archive": identity(output / "raw_traces.jsonl.gz"), "receipts": identity(output / "receipts.json")}))


if __name__ == "__main__":
    main()
