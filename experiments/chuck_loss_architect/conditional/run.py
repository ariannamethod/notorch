#!/usr/bin/env python3
"""Fixed state-conditioned future credit, sealed new-seed readout and deployment."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("chuck_future", HERE.parent / "future/run.py")
future = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(future)
scenarios, v1 = future.scenarios, future.v1
ACTIONS = ("hold", "brake", "push")
LABELS = ("initial", "oldfuture-simple", "oldfuture-adapted", "conditional-simple",
          "conditional-adapted", "source", "hold", "brake", "push")


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": v1.digest(path)}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def replay_samples(previous: Path, output: Path, old: dict, protocol: dict) -> list[dict]:
    """Reuse authenticated worlds, preserving exact source-history association."""
    recorded = scenarios.strict_json((previous / "development.samples.json").read_text())
    samples = []
    for body in protocol["evaluation"]["bodies"]:
        for seed in protocol["training"]["seeds"]:
            prefix = previous / "cohorts" / f"{body}-s{seed}-diagnostic"
            rows = scenarios.events(Path(str(prefix) + ".jsonl"))
            rebuilt = future.gather_samples(rows, body, seed, prefix, previous, rows)
            assert rebuilt == [s for s in recorded if (s["body"], s["seed"]) == (body, seed)], "archived sample/trace association differs"
            for sample in rebuilt:
                source = previous / sample["source_policy"]
                destination = output / "replay" / sample["source_policy"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                assert identity(destination) == old["artifacts"][sample["source_policy"]]
                sample["source_policy"] = destination.relative_to(output).as_posix()
                samples.append(sample)
    assert len(samples) == protocol["training"]["states"] + protocol["adaptation"]["states"]
    shutil.copyfile(previous / "development.samples.json", output / "replay" / "original-development.samples.json")
    return samples


def readouts(binary: Path, table: Path, samples: list[dict], lives: dict[str, Path],
             output: Path, split: str, env: dict, anchors: dict) -> list[dict]:
    lookup = {(s["body"], s["seed"], s["checkpoint"]): s for s in samples}
    result = []
    for label in LABELS:
        trace = output / f"{split}-{label}.readout.jsonl"
        future.execute([str(binary), "eval", str(table), str(lives[label]) if label in lives else "-",
                        label, str(trace)], output / f"{split}-{label}.readout.log", env)
        rows = scenarios.events(trace)
        anchors[trace.name] = {"identity": identity(trace), "events": [dict(row) for row in rows]}
        assert len(rows) == len(samples)
        seen = set()
        for row in rows:
            key = row["body"], row["seed"], row["checkpoint"]
            assert key not in seen, "duplicate readout identity"
            seen.add(key)
            sample = lookup[key]
            assert row["state_hash"] == sample["state_hash"] and row["source_policy_hash"] == sample["policy_hash"]
            action = ACTIONS[row["action"] - 1]
            arm = sample["arms"][action]
            assert f32(row["future_loss"]) == f32(arm["future_after"]), "selected action/outcome mismatch"
            assert row["label"] == label
            if label in ACTIONS: assert action == label and row["forced"]
            row["split"] = split
            for name in ("immediate_after", "origin_after_horizon", "heldout_after"):
                row[name] = arm[name]
            result.append(row)
        assert seen == set(lookup)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    lookup = {(r["split"], r["body"], r["seed"], r["checkpoint"], r["label"]): r for r in rows}
    result = []
    for split in sorted({r["split"] for r in rows}):
        for body in ("simple", "hevlm"):
            for label in LABELS:
                group = [r for r in rows if (r["split"], r["body"], r["label"]) == (split, body, label)]
                if not group: continue
                changes = {other: sum(r["action"] != lookup[(split, body, r["seed"], r["checkpoint"], other)]["action"]
                                      for r in group)
                           for other in ("initial", "oldfuture-simple", "conditional-simple")}
                result.append({"split": split, "body": body, "label": label, "states": len(group),
                               "actions": dict(collections.Counter(ACTIONS[r["action"] - 1] for r in group)),
                               "optimal": sum(r["optimal"] for r in group),
                               "mean_regret": sum(r["regret"] for r in group) / len(group),
                               "mean_advantage_vs_hold": sum(r["advantage"] for r in group) / len(group),
                               "choice_changes": changes})
    return result


def validate_rollout(rows: list[dict], steps: int, learned: bool, resume_step: int) -> dict:
    assert rows[0]["type"] == "rollout_run" and rows[-1]["type"] == "rollout_summary", "incomplete rollout trace"
    assert rows[0]["steps"] == rows[-1]["steps"] == steps
    events = [r for r in rows if r["type"] == "rollout_step"]
    assert [r["step"] for r in events] == list(range(1, steps + 1)), "incomplete rollout step sequence"
    evaluations = [r for r in rows if r["type"] == "evaluation"]
    expected_evals = sorted({0, steps, *range(128, steps + 1, 128)})
    assert [r["step"] for r in evaluations] == expected_evals
    resumes = [r for r in rows if r["type"] == "policy_resume"]
    assert len(resumes) == bool(resume_step)
    if resume_step: assert resumes[0]["step"] == resume_step and resumes[0]["exact"]
    counts = [sum(r["action"] == action for r in events) for action in range(4)]
    assert rows[-1]["action_counts"] == counts
    for row in events:
        assert isinstance(row["after"], (int, float)), "divergent rollout has no success receipt"
        if learned:
            policy = row["architect"]
            assert policy["sequence"] == policy["consequence"]["decision"] == row["step"]
            assert policy["action"]["kind"] == row["action"]
            assert policy["consequence"]["learned"] == 0 and policy["explored"] == 0
            assert policy["observation"]["loss"] == row["before"]
            assert policy["consequence"]["after_loss"] == row["after"]
            if row["step"] == 1: assert policy["features"][13:] == [0, 0, 0]
            else: assert f32(policy["features"][15]) == f32(events[row["step"] - 2]["architect"]["consequence"]["reward"])
    if learned:
        assert rows[-1]["frozen_weights_unchanged"] and rows[-1]["history_updates"] == steps
        assert rows[-1]["policy_hash"] == events[-1]["architect"]["policy_post"]
    return {"steps": steps, "action_counts": dict(zip(("legacy", *ACTIONS), counts)),
            "heldout": evaluations, "seconds": rows[-1]["seconds"], "max_rss_kib": rows[-1]["max_rss_kib"],
            "frozen_weights_unchanged": rows[-1]["frozen_weights_unchanged"],
            "history_updates": rows[-1]["history_updates"]}


def compare_resume(left: list[dict], right: list[dict], prefix: Path, resumed: Path, steps: int) -> dict:
    retained = ("rollout_step", "evaluation")
    assert [r for r in left if r["type"] in retained] == [r for r in right if r["type"] in retained], "resumed continuation differs"
    artifacts = {}
    for suffix in (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"):
        a, b = Path(str(prefix) + suffix), Path(str(resumed) + suffix)
        assert a.read_bytes() == b.read_bytes(), "resumed final state differs: " + suffix
        artifacts[suffix] = identity(a)
    return {"status": "PASS", "steps_compared": steps, "final_artifacts": artifacts,
            "scope": "complete Architect saved/reloaded in process; real body continues under its next selections"}


def verify_terminal_artifacts(output: Path, anchors: dict) -> dict:
    """Validate final persisted bytes against the receipts acquired during execution."""
    steps = anchors["steps"]
    for label, receipt in anchors["fit_receipts"].items():
        assert v1.digest(output / (label + ".fit.jsonl")) == receipt["fit_trace_sha256"], "persisted fit trace changed"
        assert v1.digest(output / (label + ".samples.bin")) == receipt["sample_table_sha256"], "persisted fitting sample table changed"
    for name, anchor in anchors["readouts"].items():
        assert identity(output / name) == anchor["identity"], "persisted readout trace changed"
        assert scenarios.events(output / name) == anchor["events"], "persisted readout events changed"
    for run in anchors["rollouts"]:
        path = output / (run["prefix"] + ".jsonl")
        assert identity(path) == run["trace"], "persisted rollout trace changed: " + run["prefix"]
        assert validate_rollout(scenarios.events(path), steps, run["arm"] in anchors["learned_labels"], run["resume_step"]) == run["summary"]
        for suffix, recorded in run["final_artifacts"].items():
            assert identity(output / (run["prefix"] + suffix)) == recorded, "persisted rollout final artifact changed"
    for record in anchors["continuation"]:
        prefix = output / "rollouts" / f"{record['body']}-s{record['seed']}-conditional-adapted"
        resumed = Path(str(prefix) + "-resume")
        assert compare_resume(scenarios.events(Path(str(prefix) + ".jsonl")), scenarios.events(Path(str(resumed) + ".jsonl")),
                              prefix, resumed, steps)["status"] == "PASS"
    return {"fit_traces": len(anchors["fit_receipts"]), "readout_traces": len(anchors["readouts"]),
            "rollouts": len(anchors["rollouts"]), "continuations": len(anchors["continuation"]), "status": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output, previous = args.output.resolve(), args.previous_run.resolve()
    if output == ROOT or ROOT in output.parents: parser.error("output must be outside Git")
    if output == previous or previous in output.parents: parser.error("output must not modify the previous run")
    if output.exists(): parser.error("choose a fresh output directory")
    protocol = scenarios.strict_json((HERE / "protocol.json").read_text())
    if v1.digest(previous / "results.json") != protocol["previous_results_sha256"]:
        parser.error("previous result identity mismatch")
    old = scenarios.strict_json((previous / "results.json").read_text())
    previous_verified = future.authenticate_previous(previous, old)
    status = v1.command(["git", "status", "--porcelain"])
    if status and not args.allow_dirty: parser.error("final run requires clean committed source")
    if args.allow_dirty and not args.smoke: parser.error("dirty runs are restricted to development smoke")
    output.mkdir(parents=True)
    seeds = protocol["smoke"]["evaluation_seeds"] if args.smoke else protocol["evaluation"]["seeds"]
    steps = protocol["smoke"]["host_steps"] if args.smoke else protocol["host_steps"]
    env = dict(os.environ, NT_SIMD_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    source_names = ["notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
                    "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h", "examples/chuck_architect_rollout.h",
                    "examples/chuck_architect_future.c", "examples/chuck-loss-architect.json",
                    "experiments/chuck_loss_architect/run.py", "experiments/chuck_loss_architect/scenarios/run.py",
                    "experiments/chuck_loss_architect/scenarios/protocol.json", "experiments/chuck_loss_architect/future/run.py",
                    "experiments/chuck_loss_architect/conditional/run.py", "experiments/chuck_loss_architect/conditional/protocol.json",
                    "experiments/chuck_loss_architect/conditional/development_diagnosis.json"]
    source_hashes = {name: v1.digest(ROOT / name) for name in source_names}
    host_binary, fitter = output / "host_runner", output / "future_runner"
    build_commands = [scenarios.build(host_binary), scenarios.build(fitter, ROOT / "examples/chuck_architect_future.c")]
    config = ROOT / "examples/chuck-loss-architect.json"
    shutil.copyfile(config, output / "architect-config.json")
    shutil.copyfile(HERE / "protocol.json", output / "protocol.json")
    shutil.copyfile(HERE / "development_diagnosis.json", output / "development_diagnosis.json")
    datasets = old["manifest"]["datasets"]
    for body in protocol["evaluation"]["bodies"]:
        for suffix in (".corpus", ".tokens.u32", ".vocab.json"):
            source = previous / (body + suffix)
            shutil.copyfile(source, output / source.name)
            assert identity(output / source.name) == old["artifacts"][source.name]
    manifest = {"source_commit": v1.command(["git", "rev-parse", "HEAD"]), "source_status": status,
                "source_sha256": source_hashes, "protocol": protocol, "protocol_sha256": v1.digest(HERE / "protocol.json"),
                "datasets": datasets, "machine": v1.machine(), "compiler": v1.command(["cc", "--version"]).splitlines()[0],
                "compile_flags": "-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread", "binaries_sha256": {"host": v1.digest(host_binary), "future": v1.digest(fitter)},
                "smoke": args.smoke, "executed_seeds": seeds, "executed_host_steps": steps,
                "previous_results_sha256": v1.digest(previous / "results.json"), "previous_artifacts_verified": previous_verified}
    write_json(output / "manifest.json", manifest)
    timeline = output / "timeline.jsonl"
    def phase(name, **fields):
        with timeline.open("a") as stream:
            stream.write(json.dumps({"phase": name, "time_ns": time.time_ns(), **fields}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    phase("authenticated_replay_started")
    training = replay_samples(previous, output, old, protocol)
    lives = {name: output / (name + ".policy.bin") for name in LABELS[:5]}
    for new, archived in (("initial", "initial"), ("oldfuture-simple", "simple"), ("oldfuture-adapted", "adapted")):
        shutil.copyfile(previous / (archived + ".policy.bin"), lives[new])
        assert v1.digest(lives[new]) == old["sealed_lives"]["lives"][archived]["sha256"]
    fit_receipts = {}
    for body, label, parent, identity_name in (("simple", "conditional-simple", "initial", protocol["training"]["life_id"]),
                                               ("hevlm", "conditional-adapted", "conditional-simple", protocol["adaptation"]["life_id"])):
        samples = sorted((s for s in training if s["body"] == body), key=lambda s: (s["seed"], s["checkpoint"]))
        table, trace = output / (label + ".samples.bin"), output / (label + ".fit.jsonl")
        future.write_samples(table, samples)
        future.execute([str(fitter), "fit-conditioned", str(lives[parent]), str(table), str(lives[label]), str(trace),
                        str(protocol["epochs_per_stage"]), identity_name], output / (label + ".fit.log"), env)
        rows = scenarios.events(trace)
        fits = [r for r in rows if r["type"] == "fit"]
        assert len(fits) == protocol["epochs_per_stage"] * len(samples)
        assert rows[-1]["non_weight_fields_unchanged"]
        for i, row in enumerate(fits):
            sample = samples[i % len(samples)]
            assert (row["body"], row["seed"], row["checkpoint"]) == (sample["body"], sample["seed"], sample["checkpoint"])
            assert row["fit_step"] == i + 1 and row["sample_index"] == i % len(samples)
            assert row["online_decisions"] == row["online_updates"] == 0 and row["scale"] > 0
            losses = list(map(f32, sample["future_loss"]))
            scale = max(max(abs(losses[0] - value) for value in losses), 1e-6 * (abs(losses[0]) + 1))
            assert abs(row["scale"] - scale) <= 1e-15 * max(scale, 1)
            assert list(map(f32, row["target"])) == [f32((losses[0] - value) / scale) for value in losses]
        fit_receipts[label] = {"fit_steps": len(fits), "epochs": protocol["epochs_per_stage"], "samples": len(samples),
                               "initial_hash": rows[0]["initial_hash"], "final_hash": rows[-1]["final_hash"],
                               "sample_table_sha256": v1.digest(table), "fit_trace_sha256": v1.digest(trace)}
    sealed = {label: {"file": path.name, **identity(path)} for label, path in lives.items()}
    seal = {"protocol_sha256": manifest["protocol_sha256"], "lives": sealed, "fit_receipts": fit_receipts,
            "evaluation_generation_started": False}
    write_json(output / "sealed_lives.json", seal)
    phase("lives_sealed", sealed_lives_sha256=v1.digest(output / "sealed_lives.json"))
    dev_table = output / "development.samples.bin"
    future.write_samples(dev_table, training)
    readout_anchors = {}
    outputs = readouts(fitter, dev_table, training, lives, output, "development", env, readout_anchors)
    phase("evaluation_generation_started", sealed_lives_sha256=v1.digest(output / "sealed_lives.json"))
    cohort_dir = output / "cohorts"; cohort_dir.mkdir()
    scenario_protocol = scenarios.strict_json((HERE.parent / "scenarios/protocol.json").read_text())
    cohorts, evaluation, comparisons = [], [], []
    for body in protocol["evaluation"]["bodies"]:
        for seed in seeds:
            prefixes, traces = {}, {}
            for probes, label in ((0, "control"), (1, "diagnostic")):
                name = f"{body}-s{seed}-{label}"; prefix = cohort_dir / name
                print("RUN", name, flush=True)
                future.execute([str(host_binary), "--scenarios", body, str(output / (body + ".tokens.u32")), str(prefix),
                                str(steps), str(seed), str(protocol["body_lr"]), str(config), str(probes)],
                               cohort_dir / (name + ".log"), env)
                prefixes[label] = prefix; traces[label] = scenarios.events(Path(str(prefix) + ".jsonl"))
            parity = scenarios.compare_hosts(traces["control"], traces["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps)
            comparisons.extend(scenarios.comparisons(traces["diagnostic"], body, seed, scenario_protocol, steps))
            evaluation.extend(future.gather_samples(traces["diagnostic"], body, seed, prefixes["diagnostic"], output))
            cohorts.append({"body": body, "seed": seed, "parity": parity,
                            "control_summary": traces["control"][-1], "diagnostic_summary": traces["diagnostic"][-1],
                            "control_trace_sha256": v1.digest(Path(str(prefixes["control"]) + ".jsonl")),
                            "trace_sha256": v1.digest(Path(str(prefixes["diagnostic"]) + ".jsonl"))})
    eval_table = output / "evaluation.samples.bin"; future.write_samples(eval_table, evaluation)
    expected_evaluation = {(body, seed, checkpoint) for body in protocol["evaluation"]["bodies"] for seed in seeds
                           for checkpoint in protocol["checkpoints_before_updates"] if checkpoint <= steps}
    assert {(s["body"], s["seed"], s["checkpoint"]) for s in evaluation} == expected_evaluation
    assert len(evaluation) == len(expected_evaluation), "duplicate evaluation identity"
    outputs.extend(readouts(fitter, eval_table, evaluation, lives, output, "evaluation", env, readout_anchors))
    rollout_dir = output / "rollouts"; rollout_dir.mkdir()
    rollouts, continuation = [], []
    phase("closed_loop_started")
    for body in protocol["closed_loop"]["bodies"]:
        for seed in seeds:
            def deploy(arm, resume_step):
                name = f"{body}-s{seed}-{arm}" + ("-resume" if resume_step else "")
                prefix = rollout_dir / name
                print("RUN", name, flush=True)
                future.execute([str(host_binary), "--rollout", body, str(output / (body + ".tokens.u32")), str(prefix),
                                str(steps), str(seed), str(protocol["body_lr"]), str(config), arm,
                                str(lives[arm]) if arm in lives else "-", str(resume_step)], rollout_dir / (name + ".log"), env)
                trace = Path(str(prefix) + ".jsonl"); rows = scenarios.events(trace)
                summary = validate_rollout(rows, steps, arm in lives, resume_step)
                suffixes = [".final.bin", ".moments.final.bin", ".optimizer.final.json"]
                if arm in lives: suffixes.append(".policy.final.bin")
                rollouts.append({"body": body, "seed": seed, "arm": arm, "resume_step": resume_step,
                                 "prefix": prefix.relative_to(output).as_posix(), "trace": identity(trace), "summary": summary,
                                 "final_body": identity(Path(str(prefix) + ".final.bin")),
                                 "final_artifacts": {suffix: identity(Path(str(prefix) + suffix)) for suffix in suffixes}})
                # Preserve the original anchor even if a later integrity gate fails.
                write_json(output / "rollout_receipts.json", rollouts)
                return prefix, rows
            prefixes = {}
            for arm in protocol["closed_loop"]["arms"]: prefixes[arm] = deploy(arm, 0)
            initial = [Path(str(prefix) + ".initial.bin").read_bytes() for prefix, rows in prefixes.values()]
            assert all(data == initial[0] for data in initial), "rollout arms have different initial bodies"
            resume_step = min(protocol["closed_loop"]["resume_step"], steps // 2)
            resumed, resumed_rows = deploy("conditional-adapted", resume_step)
            prefix, rows = prefixes["conditional-adapted"]
            continuation.append({"body": body, "seed": seed, "resume_step": resume_step,
                                 **compare_resume(rows, resumed_rows, prefix, resumed, steps)})
    for label, path in lives.items(): assert identity(path) == {k: sealed[label][k] for k in ("bytes", "sha256")}, "sealed life changed"
    terminal = future.verify_cohort_traces(cohorts, output, steps)
    terminal_anchors = {"steps": steps, "fit_receipts": fit_receipts, "readouts": readout_anchors,
                        "rollouts": rollouts, "continuation": continuation, "learned_labels": list(lives)}
    write_json(output / "terminal_anchors.json", terminal_anchors)
    terminal["additional_artifacts"] = verify_terminal_artifacts(output, terminal_anchors)
    for name, digest in source_hashes.items(): assert v1.digest(ROOT / name) == digest, "source changed during experiment: " + name
    phase("evaluation_complete", common_states=len(evaluation), rollout_processes=len(rollouts), sealed_lives_unchanged=True)
    result = {"manifest": manifest, "fit_receipts": fit_receipts, "sealed_lives": seal, "cohorts": cohorts,
              "summary": summarize(outputs), "readouts": outputs, "comparisons": comparisons,
              "rollouts": rollouts, "policy_continuation": continuation, "final_trace_recheck": terminal,
              "readout_traces": {name: anchor["identity"] for name, anchor in readout_anchors.items()},
              "source_hashes_rechecked": True, "fitted_lives_unchanged_by_evaluation": True,
              "artifacts": {p.relative_to(output).as_posix(): identity(p) for p in sorted(output.rglob("*"))
                            if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".bin", ".u32", ".corpus")}}
    write_json(output / "results.json", result)
    print("CONDITIONAL_OK", json.dumps({"development_states": len(training), "evaluation_states": len(evaluation),
                                        "fit_steps": sum(r["fit_steps"] for r in fit_receipts.values()),
                                        "rollout_steps": sum(r["summary"]["steps"] for r in rollouts)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
