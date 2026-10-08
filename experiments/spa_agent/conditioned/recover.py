#!/usr/bin/env python3
"""One explicit frozen diagnostics replay, then only unstarted evaluation work.

The original runner, protocol, acquisitions and source snapshot stay unchanged.
This recovery is specific to the recorded seed-601 ordinary-trace truncation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_SHA = "2aca5e434396f61509d392dd4167616851a88aff2fc718a44a212c4587170235"
DAMAGED = {"bytes": 120594, "sha256": "b8f32e01ecac0c40a223bd01cd47116152fb96b4441ada12c5ff5e361aa85413"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    snapshot, inputs = output / "source_snapshot", output / "inputs"
    spec = importlib.util.spec_from_file_location("spa_frozen_conditioned_run",
        snapshot / "experiments/spa_agent/conditioned/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    require, identity, write_json = run.require, run.identity, run.write_json
    read = lambda path: json.loads(path.read_bytes())
    require(identity(snapshot / "experiments/spa_agent/conditioned/run.py")["sha256"] == RUN_SHA,
            "unregistered original runner")
    metadata = read(output / "metadata.json")
    barrier = read(output / "all_eight_lives_sealed.json")
    barrier_id = identity(output / "all_eight_lives_sealed.json")
    protocol = metadata["protocol"]
    failure_path = HERE / "durability_failure.json"
    failure = read(failure_path)
    # These are failure-time pins; earlier acquisition pins remain authoritative.
    closed = dict(failure["artifacts"])
    stages = read(output / "stages.json")
    require([s["phase"] for s in stages] == ["fitting_started", "all_eight_lives_sealed",
            "new_source_generation_started"], "failure boundary differs")
    require(not list(output.glob("evaluation*")) and not list(output.glob("*readout*")) and
            not (output / "receipts.json").exists(), "evaluation already advanced")
    require(identity(output / "on-s601.jsonl") == DAMAGED, "recorded damaged trace differs")
    recovery_ids = {str(path.relative_to(ROOT)): identity(path)
                    for path in (Path(__file__), failure_path, HERE / "recovery_preflight.json")}
    binaries = metadata["binaries"]
    native = run.POLICY.SPA.Native(output / "libnotorch.so")
    fits = [output / f"fit-{i}" for i in (1, 2)]
    seals = [run.POLICY.verify_seal(p, native, run.PROTOCOL_SHA)[0] for p in fits]
    training = run.POLICY.read_dataset(output / "training.json", run.PROTOCOL_SHA, training=True)
    commands = [read(output / (label + ".command.json")) for label in (
        "build-spa_agent_replicates", "build-spa_agent_demo", "build-shared", "fit-1", "fit-2",
        "off-s509", "on-s509", "off-s601", "on-s601")]

    def remember(path, raw=None):
        name = str(path.relative_to(output))
        expected = identity(path) if raw is None else {"bytes": len(raw), "sha256": run.sha(raw)}
        require(identity(path) == expected, "closed artifact differs: " + name)
        require(name not in closed or closed[name] == expected, "closed artifact replaced: " + name)
        closed[name] = expected

    def stable():
        for name, digest in metadata["source_files_sha256"].items():
            require(identity(ROOT / name)["sha256"] == digest and
                    identity(snapshot / name)["sha256"] == digest, "frozen source changed: " + name)
        for name, expected in recovery_ids.items():
            require(identity(ROOT / name) == expected, "recovery source changed: " + name)
        require(identity(output / "all_eight_lives_sealed.json") == barrier_id,
                "original acquisition barrier changed")
        require(metadata["source_files_sha256"] == barrier["source_sha256"] and
                metadata["binaries"] == barrier["binaries"] and metadata["inputs"] == barrier["inputs"],
                "original metadata/barrier differs")
        require(identity(output / "training.json") == barrier["training"], "original training changed")
        for name, expected in binaries.items():
            require(identity(output / name) == expected, "frozen binary changed")
        for name, expected in metadata["inputs"].items():
            require(identity(inputs / name) == expected, "frozen input changed")
        for directory, entries in barrier["copies"].items():
            for name, expected in entries.items():
                require(identity(output / directory / name) == expected, "sealed fitting file changed")
        for name, expected in list(closed.items()):
            require(identity(output / name) == expected, "closed artifact changed: " + name)

    def phase(name, **details):
        # stages.json is the sole explicitly mutable record until completion.
        closed.pop("stages.json", None)
        stages.append({"order": len(stages) + 1, "phase": name, **details})
        write_json(output / "stages.json", stages, replace=True)
        print(json.dumps(stages[-1], sort_keys=True), flush=True)

    def command(arguments, label):
        record = run.run_command(arguments, snapshot, output, label)
        commands.append(record)
        for suffix in ("stdout.txt", "stderr.txt", "command.json"):
            remember(output / (label + "." + suffix))
        return record

    stable()
    failure_lane = output / "failure-lane"
    failure_lane.mkdir()
    damaged = (output / "on-s601.jsonl").read_bytes()
    run.write_bytes(failure_lane / "on-s601.truncated.jsonl.gz", gzip.compress(damaged, mtime=0))
    run.write_bytes(failure_lane / "stages-before-recovery.json", (output / "stages.json").read_bytes())
    run.write_bytes(failure_lane / "durability_failure.json", failure_path.read_bytes())
    for path in failure_lane.iterdir():
        remember(path)
    replay = output / "recovery-replay"
    replay.mkdir()
    phase("one_frozen_on601_diagnostics_replay_started", additional_fits=0, additional_seeds=0)
    recovery_command = command([output / "spa_agent_demo",
        *(inputs / n for n in ("simple.weights", "corpus.tokens.u32", "vocabulary.u32")),
        replay / "on-s601", "601", "--scenarios", replay / "scenarios-s601.jsonl"], "recovery-on-s601")
    for path in replay.iterdir():
        if path.is_file():
            remember(path)
    stable()
    restored, trace_gate = run.FUTURE.complete_trace(replay / "on-s601.jsonl", 601)
    require(restored.startswith(damaged), "damaged trace is not exact replay prefix")
    require(restored == (output / "off-s601.jsonl").read_bytes(), "replay changes OFF ordinary trajectory")
    scenario_raw, scenario_gate = run.FUTURE.complete_scenario(replay / "scenarios-s601.jsonl",
        restored, 601, read(snapshot / "experiments/spa_agent/scenarios/protocol.json"))
    require(scenario_raw == (output / "scenarios-s601.jsonl").read_bytes(), "replay changes original scenarios")
    arms = read(snapshot / "experiments/spa_agent/protocol.json")["arms"]
    for arm in arms:
        raw = (replay / f"on-s601.{arm}.life.bin").read_bytes()
        require(raw == (output / f"on-s601.{arm}.life.bin").read_bytes() ==
                (output / f"off-s601.{arm}.life.bin").read_bytes(), "replay changes original life")
    # The original complete ON file had no full hash. Its failed prefix and
    # original complete OFF/scenario/life witnesses are retained explicitly.
    run.write_bytes(output / "on-s601.jsonl", restored, replace=True)
    closed["on-s601.jsonl"] = identity(replay / "on-s601.jsonl")
    remember(output / "on-s601.jsonl", restored)
    recovery = {"schema": 1, "status": "ONE_FROZEN_DIAGNOSTICS_REPLAY_VERIFIED",
        "source_files": recovery_ids, "protocol_sha256": run.PROTOCOL_SHA,
        "original_runner_sha256": RUN_SHA, "failure_receipt": identity(failure_path),
        "additional_diagnostics_processes": 1, "additional_policy_fits": 0,
        "additional_generation_seeds": [], "original_acquisition_barrier": barrier_id,
        "damaged": DAMAGED, "damaged_is_exact_prefix": True,
        "restored": identity(output / "on-s601.jsonl"), "matches_original_off": True,
        "original_scenarios_and_five_lives_identical": True,
        "original_complete_on_identity": "Unavailable; the original ON file failed completeness before any full-file seal.",
        "retained_file_identities": "Failure-time pins plus the earlier acquisition barrier; not retroactive initial-close pins.",
        "command": recovery_command, "trace_gate": trace_gate, "scenario_gate": scenario_gate}
    write_json(output / "recovery.json", recovery)
    remember(output / "recovery.json")
    stable()
    phase("original_cohort_source_recovery_verified", restored=identity(output / "on-s601.jsonl"))

    evaluation_streams, host_gates = {}, []
    scenario_protocol = read(snapshot / "experiments/spa_agent/scenarios/protocol.json")
    for seed in protocol["evaluation"]["seeds"]:
        for mode in ("off", "on"):
            raw, checked = run.FUTURE.complete_trace(output / f"{mode}-s{seed}.jsonl", seed)
            evaluation_streams[seed, "ordinary_" + mode] = raw
            host_gates.append({"seed": seed, "mode": mode, **checked})
        require(evaluation_streams[seed, "ordinary_off"] == evaluation_streams[seed, "ordinary_on"],
                "ordinary diagnostics parity differs")
        for arm in arms:
            require((output / f"off-s{seed}.{arm}.life.bin").read_bytes() ==
                    (output / f"on-s{seed}.{arm}.life.bin").read_bytes(), "saved life diagnostics differ")
        raw, checked = run.FUTURE.complete_scenario(output / f"scenarios-s{seed}.jsonl",
            evaluation_streams[seed, "ordinary_off"], seed, scenario_protocol)
        evaluation_streams[seed, "scenarios"] = raw
        host_gates.append({"seed": seed, "mode": "scenarios", **checked})
    source_archive = output / "evaluation_sources.raw.jsonl.gz"
    run.FUTURE.write_archive(source_archive, evaluation_streams)
    remember(source_archive)
    sources, references = run.REP.source_samples(evaluation_streams, protocol["evaluation"]["seeds"],
        identity(source_archive)["sha256"], scenario_protocol)
    require(all(s["body_hash"] == training[0]["body_hash"] for s in sources), "body differs across cohorts")
    vocabulary_fnv = run.REP.fnv((inputs / "vocabulary.u32").read_bytes())
    stable()

    def repeat_job(seed):
        chunk = [s for s in sources if s["seed"] == seed]
        data, trace = output / f"evaluation-sources-s{seed}.txt", output / f"evaluation-repeats-s{seed}.jsonl"
        run.REP.write_sources(data, chunk, vocabulary_fnv)
        remember(data)
        label = f"evaluation-repeats-s{seed}"
        record = run.run_command([output / "spa_agent_replicates", inputs / "simple.weights",
            inputs / "vocabulary.u32", data, trace], snapshot, output, label)
        for suffix in ("stdout.txt", "stderr.txt", "command.json"):
            remember(output / (label + "." + suffix))
        raw = run.REP.complete_bytes(trace)
        remember(trace, raw)
        projected, checked = run.REP.validate_replicates(raw, chunk, references,
            dataset_fnv=run.REP.fnv(data.read_bytes()), vocabulary_fnv=vocabulary_fnv)
        return projected, record, {"seed": seed, **checked}

    phase("new_repeated_generation_started", sources=len(sources), repeats=8)
    evaluation, repeat_gates = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for projected, record, checked in pool.map(repeat_job, protocol["evaluation"]["seeds"]):
            evaluation.extend(projected)
            commands.append(record)
            repeat_gates.append(checked)
    evaluation.sort(key=lambda row: row["ordinal"])
    require(len(evaluation) == 48, "incomplete evaluation cohort")
    write_json(output / "evaluation.json", {"schema": 1, "protocol_sha256": run.PROTOCOL_SHA,
        "replicates": 8, "samples": evaluation})
    remember(output / "evaluation.json")
    run.POLICY.read_dataset(output / "evaluation.json", run.PROTOCOL_SHA, training=False)
    stable()
    phase("new_generation_complete", measurements=sum(x["measurements"] for x in repeat_gates))
    helper = snapshot / "experiments/spa_agent/conditioned/policy.py"
    policy_args = ["--protocol", snapshot / "experiments/spa_agent/conditioned/protocol.json",
                   "--library", output / "libnotorch.so"]
    summaries = {}
    for split, samples in (("training", training), ("evaluation", evaluation)):
        readouts = []
        for i, directory in enumerate(fits, 1):
            target = output / f"{split}.readout-{i}.jsonl"
            arguments = [sys.executable, helper, "score", "--dataset", output / (split + ".json"),
                         *policy_args, "--policies", directory, "--output", target]
            if split == "training":
                arguments.append("--training")
            command(arguments, f"{split}-readout-{i}")
            raw = run.REP.complete_bytes(target)
            remember(target, raw)
            rows, _ = run.REP.parse_raw(raw)
            choices = run.REP.validate_readouts(rows, samples,
                {arm: seals[i - 1]["lives"][arm]["life_hash"] for arm in run.ARMS})
            readouts.append(raw)
        require(readouts[0] == readouts[1], "sealed copies select different actions")
        summaries[split] = run.REP.summarize(samples, choices, split)
        write_json(output / (split + ".summary.json"), summaries[split])
        remember(output / (split + ".summary.json"))
    stable()
    phase("all_readouts_and_source_checks_complete")
    write_json(output / "commands.json", commands)
    remember(output / "commands.json")
    remember(output / "stages.json")
    artifacts, records = run.record_inventory(output, binaries, closed)
    archive = run.pack_records(output, records, closed)
    stable()
    result = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages,
        "commands": commands, "fit_repeat": {"copies": 2,
        "byte_identical_files": list(barrier["copies"]["fit-1"]), "seals": seals},
        "host_gates": host_gates, "repeat_gates": repeat_gates, "recovery": recovery,
        "training": summaries["training"], "evaluation": summaries["evaluation"],
        "artifacts": artifacts, "archive": archive}
    write_json(output / "receipts.json", result)
    print(json.dumps({"complete": True, "archive": identity(output / archive["file"]),
        "evaluation_h4": [r for r in summaries["evaluation"]["summaries"]
                          if r["group"] == "combined" and r["horizon"] == 4]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
