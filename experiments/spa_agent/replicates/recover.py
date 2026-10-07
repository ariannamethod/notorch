#!/usr/bin/env python3
"""Authenticate a deterministic fit replay, then continue frozen evaluation.

The original execution stopped at its seal gate before any new evaluation
state. This explicit recovery does not regenerate training outcomes or change
policies: its one replay must reproduce the original sealed fit and four lives.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); snapshot = output / "source_snapshot"
    spec = importlib.util.spec_from_file_location("spa_frozen_recovery", snapshot / "experiments/spa_agent/replicates/run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    require, identity = run.require, run.identity
    metadata = json.loads((output / "metadata.json").read_text())
    protocol = metadata["protocol"]; inputs = output / "inputs"
    original_run_sha = metadata["source_files_sha256"]["experiments/spa_agent/replicates/run.py"]
    require(original_run_sha == "046b1bea1f0676b516afa1125a69d47c401b19e33ef838ad12c2d2be95cdd257", "unregistered failed runner")
    recovery_source = identity(Path(__file__))
    require(not (output / "all_eight_lives_sealed.json").exists(), "original pre-evaluation boundary changed")
    for seed in protocol["evaluation"]["seeds"]:
        require(not list(output.glob(f"execution-*/*s{seed}*")), "evaluation already started before recovery")
    source_hashes = metadata["source_files_sha256"]
    binary_ids = metadata["binaries"]
    input_ids = {name: identity(inputs / name) for name in
                 ("simple.weights", "dracula.txt", "corpus.tokens.u32", "vocabulary.u32")}
    for name, expected in input_ids.items():
        require(expected["sha256"] == metadata["inputs"][name]["sha256"], "original input identity changed")
    def stable():
        for name, expected in source_hashes.items():
            require(run.V1.digest(snapshot / name) == expected and run.V1.digest(ROOT / name) == expected,
                    "frozen measured source changed: " + name)
        for name, expected in binary_ids.items(): require(identity(output / name) == expected, "frozen binary changed: " + name)
        for name, expected in input_ids.items(): require(identity(inputs / name) == expected, "frozen input changed: " + name)
        require(identity(Path(__file__)) == recovery_source, "recovery orchestrator changed")
    stable()
    failure_lane = output / "failure-lane"; failure_lane.mkdir()
    directory = output / "execution-2"; policies = directory / "policies"
    original_seal_raw = (policies / "seal.json").read_bytes()
    original_seal = json.loads(original_seal_raw)
    original_lives = {arm: (policies / (arm + ".life")).read_bytes() for arm in run.ARMS}
    damaged = run.complete_bytes(policies / "fit.jsonl")
    require(len(damaged) == 52978553 and run.sha256(damaged) ==
            "daefc3efdeb9545ea3d521de4aea57d74bf9257ecf8892241222f3d8d4745f1a", "registered damaged trace changed")
    require(original_seal["trace"]["bytes"] == 59168552 and original_seal["trace"]["sha256"] ==
            "a4ae3d8492c240cc91d419b0b78f576e43c36fcb12882ee84538d2fd49ad55ad", "original fit seal changed")
    run.FUTURE.TRACE_IO.durable_write(failure_lane / "execution-2.fit.truncated.jsonl.gz", gzip.compress(damaged, mtime=0))
    run.FUTURE.TRACE_IO.durable_write(failure_lane / "original-execution-2.seal.json", original_seal_raw)
    shutil.copyfile(output / "failure.json", failure_lane / "failure.json")
    helper = snapshot / "experiments/spa_agent/replicates/policy.py"
    library = output / "libnotorch.so"
    native = run.POLICY.SPA.Native(library)
    replay = output / "recovery-replay"
    policy_args = ["--protocol", snapshot / "experiments/spa_agent/replicates/protocol.json", "--library", library]
    print(json.dumps({"phase": "one_identical_fit_replay_started", "new_directory": replay.name}), flush=True)
    started = time.time_ns()
    command = run.run_command([sys.executable, helper, "train", "--dataset", directory / "training.json",
                              *policy_args, "--output", replay], snapshot, output, "recovery-identical-fit")
    stable()
    replay_raw = run.complete_bytes(replay / "fit.jsonl")
    expected_fit = {key: original_seal["trace"][key] for key in ("bytes", "sha256")}
    require(identity(replay / "fit.jsonl") == expected_fit, "deterministic replay did not reproduce original sealed fit")
    require(replay_raw.startswith(damaged), "damaged original is not the replay's exact prefix")
    for arm in run.ARMS:
        require((replay / (arm + ".life")).read_bytes() == original_lives[arm], "replay changed registered life: " + arm)
        require(identity(replay / (arm + ".life"))["sha256"] == original_seal["lives"][arm]["sha256"], "original life seal mismatch")
    require((replay / "seal.json").read_bytes() == original_seal_raw, "replay did not reproduce original complete seal")
    # Preserve a complete closed-file witness before replacing only the damaged
    # trace. The original four lives and original seal are never rewritten.
    run.FUTURE.TRACE_IO.durable_write(output / "recovered-fit.verified.jsonl.gz", gzip.compress(replay_raw, mtime=0))
    replacement = policies / "fit.authenticated-replacement.tmp"
    with replacement.open("xb") as stream:
        stream.write(replay_raw); stream.flush(); os.fsync(stream.fileno())
    require(identity(replacement) == expected_fit, "authenticated replacement bytes changed")
    os.replace(replacement, policies / "fit.jsonl")
    directory_fd = os.open(policies, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    require((policies / "seal.json").read_bytes() == original_seal_raw, "original seal changed during restoration")
    for arm in run.ARMS: require((policies / (arm + ".life")).read_bytes() == original_lives[arm], "original life changed during restoration")
    run.POLICY.verify_seal(policies, native, run.FROZEN_PROTOCOL_SHA256)
    recovery = {"schema": 1, "status": "AUTHENTICATED_DETERMINISTIC_TRACE_REPLAY", "started_unix_ns": started,
        "completed_unix_ns": time.time_ns(), "original_runner_sha256": original_run_sha,
        "recovery_source": {"path": "experiments/spa_agent/replicates/recover.py", **recovery_source},
        "protocol_sha256": run.FROZEN_PROTOCOL_SHA256, "replays": 1, "command": command,
        "replay_directory": replay.name, "original_seal": identity(policies / "seal.json"),
        "original_expected_fit": expected_fit, "damaged_fit": {"bytes": len(damaged), "sha256": run.sha256(damaged), "rows": len(damaged.splitlines())},
        "damaged_trace": {"path": "failure-lane/execution-2.fit.truncated.jsonl.gz", **identity(failure_lane / "execution-2.fit.truncated.jsonl.gz")},
        "restored_fit": identity(policies / "fit.jsonl"), "damaged_is_exact_prefix": True,
        "original_four_lives_unchanged": True, "replay_four_lives_identical": True,
        "original_complete_seal_identical": True, "no_evaluation_before_recovery": True,
        "input_identities": input_ids, "replay_artifacts": {path.name: identity(path) for path in sorted(replay.iterdir()) if path.is_file()}}
    run.write_json(output / "recovery.json", recovery)
    print(json.dumps({"phase": "trace_recovery_authenticated", "fit_sha256": expected_fit["sha256"], "all_four_lives_unchanged": True}), flush=True)
    del damaged, replay_raw
    executions = [output / f"execution-{i}" for i in (1, 2)]
    old_archive = snapshot / "experiments/spa_agent/scenarios/raw_traces.jsonl.gz"
    old_sha = metadata["training_source_archive"]["sha256"]
    retained = run.FUTURE.archive_streams(old_archive, old_sha)
    scenario_protocol = json.loads((snapshot / "experiments/spa_agent/scenarios/protocol.json").read_text())
    old_sources, old_refs = run.source_samples(retained, protocol["training"]["seeds"], old_sha, scenario_protocol)
    vocab_fnv = run.fnv((inputs / "vocabulary.u32").read_bytes())
    results, training = {}, {}
    for directory in executions:
        result = {"commands": [], "replicate_gates": {}, "host_gates": []}
        projected = []
        for seed in protocol["training"]["seeds"]:
            data = directory / f"training-sources-s{seed}.txt"
            values, checked = run.validate_replicates(run.complete_bytes(directory / f"training-repeats-s{seed}.jsonl"),
                [s for s in old_sources if s["seed"] == seed], old_refs,
                dataset_fnv=run.fnv(data.read_bytes()), vocabulary_fnv=vocab_fnv)
            projected.extend(values); result["replicate_gates"][f"training:{seed}"] = checked
            result["commands"].append(json.loads((directory / f"training-repeats-s{seed}.command.json").read_text()))
        samples = sorted(projected, key=lambda value: value["ordinal"])
        require(samples == json.loads((directory / "training.json").read_text())["samples"], "authenticated training dataset changed")
        fit_raw = run.complete_bytes(directory / "policies/fit.jsonl")
        rows, _ = run.parse_raw(fit_raw); result["fit"] = run.validate_fit(rows, samples); del rows
        run.POLICY.verify_seal(directory / "policies", native, run.FROZEN_PROTOCOL_SHA256)
        run.FUTURE.TRACE_IO.durable_write(directory / "fit.verified.jsonl.gz", gzip.compress(fit_raw, mtime=0))
        result["commands"].append(json.loads((directory / "policy-train.command.json").read_text()))
        results[directory.name], training[directory.name] = result, samples
    stable()
    barrier = {"status": "ALL_EIGHT_LIVES_SEALED", "protocol_sha256": run.FROZEN_PROTOCOL_SHA256,
        "source_sha256": source_hashes, "binaries": binary_ids, "input_identities": input_ids,
        "recovery_sha256": run.V1.digest(output / "recovery.json"), "executions": {}}
    for directory in executions:
        names = ["training.json", "policies/fit.jsonl", "policies/seal.json", *("policies/" + arm + ".life" for arm in run.ARMS)]
        barrier["executions"][directory.name] = {"files": {name: identity(directory / name) for name in names}}
    run.write_json(output / "all_eight_lives_sealed.json", barrier)
    run.verify_barrier(barrier, output, native); stable()
    stages = json.loads((output / "stages.json").read_text())
    def phase(name, **fields):
        stages.append({"order": len(stages) + 1, "phase": name, **fields})
        run.V1.write_json(output / "stages.json", stages)
        print(json.dumps(stages[-1], sort_keys=True), flush=True)
    phase("recovery_authenticated_then_all_eight_lives_sealed", barrier_sha256=run.V1.digest(output / "all_eight_lives_sealed.json"),
          recovery_sha256=run.V1.digest(output / "recovery.json"))
    def parallel(jobs):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fn, *args) for fn, args in jobs]
            return [future.result() for future in futures]
    def ordinary_job(directory, seed):
        streams, commands, checked = {}, [], []
        base = [output / "spa_agent_demo", *(inputs / name for name in ("simple.weights", "corpus.tokens.u32", "vocabulary.u32"))]
        scenario = directory / f"scenarios-s{seed}.jsonl"
        for mode in ("off", "on"):
            prefix = directory / f"{mode}-s{seed}"; command = [*base, prefix, str(seed)]
            if mode == "on": command += ["--scenarios", scenario]
            commands.append(run.run_command(command, snapshot, directory, f"{mode}-s{seed}"))
            raw, receipt = run.FUTURE.complete_trace(Path(str(prefix) + ".jsonl"), seed)
            streams[seed, "ordinary_" + mode] = raw; checked.append({"mode": mode, "seed": seed, **receipt})
            run.FUTURE.TRACE_IO.durable_write(directory / f"{mode}-s{seed}.verified.jsonl.gz", gzip.compress(raw, mtime=0))
        require(streams[seed, "ordinary_off"] == streams[seed, "ordinary_on"], "ordinary diagnostic trace leak")
        for arm in json.loads((snapshot / "experiments/spa_agent/protocol.json").read_text())["arms"]:
            require((directory / f"off-s{seed}.{arm}.life.bin").read_bytes() == (directory / f"on-s{seed}.{arm}.life.bin").read_bytes(), "ordinary diagnostic life leak")
        raw, receipt = run.FUTURE.complete_scenario(scenario, streams[seed, "ordinary_off"], seed, scenario_protocol)
        streams[seed, "scenarios"] = raw; checked.append({"mode": "scenarios", "seed": seed, **receipt})
        run.FUTURE.TRACE_IO.durable_write(directory / f"scenarios-s{seed}.verified.jsonl.gz", gzip.compress(raw, mtime=0))
        return directory.name, streams, commands, checked
    run.verify_barrier(barrier, output, native); stable()
    phase("unseen_source_generation_started", seeds=protocol["evaluation"]["seeds"], independent_processes=4,
          barrier_sha256=run.V1.digest(output / "all_eight_lives_sealed.json"))
    evaluation_streams = {directory.name: {} for directory in executions}
    for name, streams, commands, checks in parallel([(ordinary_job, (directory, seed))
            for directory in executions for seed in protocol["evaluation"]["seeds"]]):
        evaluation_streams[name].update(streams); results[name]["commands"] += commands; results[name]["host_gates"] += checks
    run.verify_barrier(barrier, output, native); stable()
    cohorts = {}
    for directory in executions:
        archive = directory / "evaluation_sources.raw.jsonl.gz"
        run.FUTURE.write_archive(archive, evaluation_streams[directory.name])
        cohorts[directory.name] = run.source_samples(evaluation_streams[directory.name], protocol["evaluation"]["seeds"],
                                                   run.V1.digest(archive), scenario_protocol)
        require(all(s["body_hash"] == old_sources[0]["body_hash"] for s in cohorts[directory.name][0]), "cross-cohort body identity changed")
    def replicate_job(directory, seed):
        sources, references = cohorts[directory.name]; chunk = [s for s in sources if s["seed"] == seed]
        data = directory / f"evaluation-sources-s{seed}.txt"; trace = directory / f"evaluation-repeats-s{seed}.jsonl"
        run.write_sources(data, chunk, vocab_fnv)
        command = run.run_command([output / "spa_agent_replicates", inputs / "simple.weights", inputs / "vocabulary.u32", data, trace],
                                  snapshot, directory, f"evaluation-repeats-s{seed}")
        raw = run.complete_bytes(trace)
        run.FUTURE.TRACE_IO.durable_write(directory / f"evaluation-repeats-s{seed}.verified.jsonl.gz", gzip.compress(raw, mtime=0))
        projected, checked = run.validate_replicates(raw, chunk, references, dataset_fnv=run.fnv(data.read_bytes()), vocabulary_fnv=vocab_fnv)
        return directory.name, seed, projected, checked, command
    phase("unseen_repeated_generation_started", independent_processes=4)
    evaluation = {directory.name: [] for directory in executions}
    for name, seed, projected, checked, command in parallel([(replicate_job, (directory, seed))
            for directory in executions for seed in protocol["evaluation"]["seeds"]]):
        evaluation[name] += projected; results[name]["replicate_gates"][f"evaluation:{seed}"] = checked
        results[name]["commands"].append(command)
    for directory in executions:
        evaluation[directory.name].sort(key=lambda sample: sample["ordinal"])
        run.write_json(directory / "evaluation.json", {"schema": 1, "protocol_sha256": run.FROZEN_PROTOCOL_SHA256,
                                                       "replicates": 8, "samples": evaluation[directory.name]})
    stable(); run.verify_barrier(barrier, output, native); phase("unseen_repeats_complete")
    for directory in executions:
        result = results[directory.name]
        seal, _ = run.POLICY.verify_seal(directory / "policies", native, run.FROZEN_PROTOCOL_SHA256)
        hashes = {arm: seal["lives"][arm]["life_hash"] for arm in run.ARMS}
        for split, samples in (("training", training[directory.name]), ("evaluation", evaluation[directory.name])):
            target = directory / f"{split}.readout.jsonl"
            command = [sys.executable, helper, "score", "--dataset", directory / f"{split}.json", *policy_args,
                       "--policies", directory / "policies", "--output", target]
            if split == "training": command += ["--training"]
            result["commands"].append(run.run_command(command, snapshot, directory, split + "-readout"))
            rows, _ = run.parse_raw(run.complete_bytes(target)); choices = run.validate_readouts(rows, samples, hashes)
            result[split] = run.summarize(samples, choices, split)
            run.write_json(directory / f"{split}.summary.json", result[split])
        result["policy_seal"] = seal
        streams = dict(evaluation_streams[directory.name])
        streams.update({(seed, "training_parent_" + mode): raw for (seed, mode), raw in retained.items()})
        for split, seeds in (("training", protocol["training"]["seeds"]), ("evaluation", protocol["evaluation"]["seeds"])):
            for seed in seeds: streams[seed, split + "_replicates"] = run.complete_bytes(directory / f"{split}-repeats-s{seed}.jsonl")
            streams[0, split + "_readout"] = run.complete_bytes(directory / f"{split}.readout.jsonl")
            streams[0, split + "_dataset"] = run.complete_bytes(directory / f"{split}.json")
        streams[0, "fit"] = run.complete_bytes(directory / "policies/fit.jsonl")
        streams[0, "policy_seal"] = run.complete_bytes(directory / "policies/seal.json")
        run.FUTURE.write_archive(directory / "raw_traces.jsonl.gz", streams)
        result["artifacts"] = {str(path.relative_to(directory)): identity(path) for path in sorted(directory.rglob("*")) if path.is_file()}
        run.write_json(directory / "result.json", result)
        print(json.dumps({"execution": directory.name, "evaluation_h4": [r for r in result["evaluation"]["summaries"]
            if r["group"] == "combined" and r["horizon"] == 4]}, sort_keys=True), flush=True)
    comparable = [name for name in results["execution-1"]["artifacts"] if not name.endswith((".stdout.txt", ".stderr.txt", ".command.json"))]
    require(set(results["execution-1"]["artifacts"]) == set(results["execution-2"]["artifacts"]), "execution artifact coverage differs")
    for name in comparable:
        require((executions[0] / name).read_bytes() == (executions[1] / name).read_bytes(), "complete repeated execution differs: " + name)
    for directory in executions:
        for name, expected in results[directory.name]["artifacts"].items(): require(identity(directory / name) == expected, "later artifact changed")
    stable(); run.verify_barrier(barrier, output, native)
    phase("complete_two_execution_identity", artifacts=len(comparable))
    shutil.copyfile(executions[0] / "raw_traces.jsonl.gz", output / "raw_traces.jsonl.gz")
    receipts = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages, "recovery": recovery,
        "executions": [results[directory.name] for directory in executions],
        "repeat": {"all_passed": True, "count": len(comparable), "byte_identical_artifacts": comparable},
        "artifacts": {"raw_traces.jsonl.gz": identity(output / "raw_traces.jsonl.gz"), "recovery.json": identity(output / "recovery.json")}}
    encoded = json.dumps(receipts, allow_nan=False).replace(str(snapshot), "<source_snapshot>").replace(str(output), "<output>")
    encoded = encoded.replace(str(sys.executable), "python3")
    run.write_json(output / "receipts.json", json.loads(encoded))
    print("Completed: " + str(output / "receipts.json"), flush=True)


if __name__ == "__main__":
    main()
