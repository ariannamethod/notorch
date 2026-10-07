#!/usr/bin/env python3
"""One fixed round of Chuck's own-state experience with matched continuations."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("chuck_conditional", HERE.parent / "conditional/run.py")
conditional = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(conditional)
future, scenarios, v1 = conditional.future, conditional.scenarios, conditional.v1
write_json, identity, f32 = conditional.write_json, conditional.identity, conditional.f32
ACTIONS = ("hold", "brake", "push")
LABELS = ("parent", "lived-hold", "lived-policy", "hold", "brake", "push")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_cohort(rows: list[dict], body: str, seed: int, protocol: dict,
                    steps: int, probes: bool) -> dict:
    """Validate source worlds, both continuations and the naturally selected fork."""
    summary = conditional.validate_rollout(rows, steps, True, 0)
    assert rows[0]["body"] == body and rows[0]["seed"] == seed and rows[0]["arm"] == "parent"
    allowed = {"rollout_run", "evaluation", "rollout_step", "rollout_summary", "fork",
               "branch_step", "comparison", "restore", "continuation_check"}
    assert all(row["type"] in allowed for row in rows), "unexpected or failed cohort event"
    host = {row["step"]: row for row in rows if row["type"] == "rollout_step"}
    checkpoints = [step for step in protocol["checkpoints_before_updates"] if step <= steps] if probes else []
    forks = [row for row in rows if row["type"] == "fork"]
    restores = [row for row in rows if row["type"] == "restore"]
    assert [row["step"] for row in forks] == [row["step"] for row in restores] == checkpoints, "incomplete source world set"
    branch_rows = [row for row in rows if row["type"] == "branch_step"]
    measurements = [row for row in rows if row["type"] == "comparison"]
    checks = [row for row in rows if row["type"] == "continuation_check"]
    horizon = protocol["target_horizon_total_updates"]
    horizons = protocol["diagnostic_horizons_total_updates"]
    assert len(branch_rows) == len(checkpoints) * len(protocol["continuations"]) * len(ACTIONS) * horizon, "incomplete branch trajectory"
    assert len(measurements) == len(checkpoints) * len(protocol["continuations"]) * len(ACTIONS) * len(horizons), "incomplete branch measurements"
    expected_checks = sum(min(horizon, steps - checkpoint + 1) for checkpoint in checkpoints)
    assert len(checks) == expected_checks and all(row["exact"] for row in checks), "incomplete natural continuation checks"
    selected_matches = 0
    for fork, restore in zip(forks, restores):
        checkpoint = fork["step"]
        actual = host[checkpoint]
        assert restore["exact"] and restore["state_hash"] == fork["state_hash"], "source restoration mismatch"
        assert fork["policy_hash"] == actual["architect"]["policy_pre"], "source policy mismatch"
        assert fork["observation"] == actual["architect"]["observation"] and fork["features"] == actual["architect"]["features"], "source observation/features mismatch"
        assert len(fork["features"]) == 16 and all(finite(x) and abs(x) <= 1 for x in fork["features"])
        assert fork["source_action"] == actual["action"] and fork["loss_before"] == actual["before"]
        assert len(fork["offsets"]) == horizon + len(protocol["future_probe_window_indices_zero_based"])
        assert fork["offsets"][0] == actual["offset"]
        for continuation in protocol["continuations"]:
            for action_index, action in enumerate(ACTIONS, 1):
                branch = [row for row in branch_rows if (row["checkpoint"], row["continuation"], row["intervention"]) == (checkpoint, continuation, action)]
                assert [row["update"] for row in branch] == list(range(1, horizon + 1)), "branch update identity/order mismatch"
                for index, row in enumerate(branch):
                    policy = row["architect"]
                    receipt = policy["consequence"]
                    assert row["source_hash"] == fork["state_hash"] and row["offset"] == fork["offsets"][index]
                    assert row["action"] in (1, 2, 3) and row["executed_action"] == ACTIONS[row["action"] - 1]
                    forced = index == 0 or continuation == "hold"
                    assert row["forced"] is forced, "wrong branch continuation mode"
                    if index == 0: assert row["action"] == action_index
                    elif continuation == "hold": assert row["action"] == 1
                    else:
                        scores = policy["scores"]
                        assert row["action"] == 1 + max(range(3), key=lambda k: scores[k]), "branch did not select its recorded policy"
                    assert policy["policy_pending"] is None if forced else isinstance(policy["policy_pending"], str)
                    assert finite(row["before"]) and finite(row["after_same_window"]) and finite(row["gradient_norm"])
                    assert row["gradient_norm_stage"] == "before_clip"
                    assert policy["observation"]["loss"] == receipt["before_loss"] == row["before"]
                    assert receipt["after_loss"] == row["after_same_window"]
                    assert policy["action"]["kind"] == row["action"] and policy["explored"] == 0
                    assert receipt["learned"] == receipt["nonfinite"] == 0 and receipt["error"] == 0
                    assert policy["sequence"] == receipt["decision"] == checkpoint + index
                    assert row["post_chuck"]["global_step"] == row["pre_chuck"]["global_step"] + 1
                    previous_hash = branch[index - 1]["architect"]["policy_post"] if index else fork["policy_hash"]
                    assert policy["policy_pre"] == previous_hash, "branch history hash chain broken"
                    if index:
                        assert f32(policy["features"][15]) == f32(branch[index - 1]["architect"]["consequence"]["reward"]), "branch consequence did not reach next observation"
                    else: assert policy["features"] == fork["features"]
                    if continuation == "policy" and action_index == fork["source_action"] and checkpoint + index <= steps:
                        natural = host[checkpoint + index]
                        for field in ("offset", "window_rng", "before", "gradient_norm", "action", "pre_chuck", "post_chuck", "state_hash"):
                            assert row[field] == natural[field], "selected policy fork differs from actual host: " + field
                        assert row["after_same_window"] == natural["after"]
                        for field in ("policy_pre", "policy_post", "observation", "features", "scores", "action", "consequence"):
                            assert policy[field] == natural["architect"][field], "selected fork policy history differs: " + field
                        selected_matches += 1
                for measured_horizon in horizons:
                    arms = [row for row in measurements if (row["checkpoint"], row["continuation"], row["action"], row["horizon"]) == (checkpoint, continuation, action, measured_horizon)]
                    assert len(arms) == 1, "measurement identity duplicate or missing"
                    measured = arms[0]
                    assert measured["initial_hash"] == fork["state_hash"]
                    assert measured["state_hash"] == branch[measured_horizon - 1]["state_hash"], "outcome belongs to a different executed branch"
                    assert measured["finite"] is True, "failed comparison cannot enter replay"
                    assert measured["immediate_before"] == fork["loss_before"]
                    assert measured["future_before"] == fork["future_probe_before"] and measured["heldout_before"] == fork["heldout_before"]
                    assert measured["immediate_after"] == branch[0]["after_same_window"]
                    for name in ("immediate_after", "origin_after_horizon", "future_after", "heldout_after"):
                        assert finite(measured[name]), "nonfinite measured branch cannot enter replay"
    assert selected_matches == expected_checks
    return {"status": "PASS", "host_steps": steps, "worlds": len(checkpoints), "branch_updates": len(branch_rows),
            "measurements": len(measurements), "selected_policy_transitions_compared": selected_matches, "summary": summary}


def compare_hosts(control: list[dict], diagnostic: list[dict], a: Path, b: Path, steps: int) -> dict:
    retained = ("rollout_step", "evaluation")
    assert [row for row in control if row["type"] in retained] == [row for row in diagnostic if row["type"] in retained], "diagnostics changed source host"
    identities = {}
    for suffix in (".initial.bin", ".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"):
        left, right = Path(str(a) + suffix), Path(str(b) + suffix)
        assert left.read_bytes() == right.read_bytes(), "source host final state mismatch: " + suffix
        identities[suffix] = identity(left)
    return {"status": "PASS", "host_steps_compared": steps, "artifacts": identities}


def gather_samples(rows: list[dict], body: str, seed: int, prefix: Path, output: Path, continuation: str) -> list[dict]:
    samples = []
    for fork in [row for row in rows if row["type"] == "fork"]:
        arms = {row["action"]: row for row in rows if row["type"] == "comparison" and
                (row["checkpoint"], row["continuation"], row["horizon"]) == (fork["step"], continuation, 16)}
        assert set(arms) == set(ACTIONS)
        samples.append({"body": body, "seed": seed, "checkpoint": fork["step"], "continuation": continuation,
                        "state_hash": fork["state_hash"], "policy_hash": fork["policy_hash"],
                        "observation": fork["observation"], "features": fork["features"],
                        "future_loss": [arms[action]["future_after"] for action in ACTIONS],
                        "source_policy": Path(str(prefix) + f".fork-{fork['step']}.policy.bin").relative_to(output).as_posix(),
                        "outcome_source": "executed_parent_world_branches", "arms": arms})
    return samples


def validate_fit_rows(samples: list[dict], rows: list[dict], epochs: int) -> dict:
    assert rows[0]["type"] == "fit_run" and rows[-1]["type"] == "fit_summary"
    fits = [row for row in rows if row["type"] == "fit"]
    assert len(rows) == len(fits) + 2 and len(fits) == epochs * len(samples)
    assert rows[-1]["non_weight_fields_unchanged"] and rows[-1]["fit_steps"] == len(fits)
    for i, row in enumerate(fits):
        sample = samples[i % len(samples)]
        assert (row["body"], row["seed"], row["checkpoint"]) == (sample["body"], sample["seed"], sample["checkpoint"])
        assert row["state_hash"] == sample["state_hash"] and row["source_policy_hash"] == sample["policy_hash"]
        assert row["fit_step"] == i + 1 and row["sample_index"] == i % len(samples)
        assert row["epoch"] == i // len(samples) + 1 and row["online_decisions"] == row["online_updates"] == 0
        losses = list(map(f32, sample["future_loss"]))
        assert list(map(f32, row["future_loss"])) == losses
        scale = max(max(abs(losses[0] - value) for value in losses), 1e-6 * (abs(losses[0]) + 1))
        assert abs(row["scale"] - scale) <= 1e-15 * max(scale, 1)
        assert list(map(f32, row["target"])) == [f32((losses[0] - value) / scale) for value in losses]
    return {"fit_steps": len(fits), "epochs": epochs, "samples": len(samples),
            "initial_hash": rows[0]["initial_hash"], "final_hash": rows[-1]["final_hash"]}


def readouts(binary: Path, table: Path, samples: list[dict], lives: dict[str, Path], output: Path,
             split: str, continuation: str, env: dict, anchors: dict) -> list[dict]:
    lookup = {(sample["body"], sample["seed"], sample["checkpoint"]): sample for sample in samples}
    result = []
    for label in LABELS:
        trace = output / f"{split}-{continuation}-{label}.readout.jsonl"
        future.execute([str(binary), "eval", str(table), str(lives[label]) if label in lives else "-", label, str(trace)],
                       trace.with_suffix(".log"), env)
        rows = scenarios.events(trace)
        anchors[trace.name] = {"identity": identity(trace), "events": [dict(row) for row in rows]}
        assert len(rows) == len(samples)
        seen = set()
        for row in rows:
            key = row["body"], row["seed"], row["checkpoint"]
            assert key in lookup and key not in seen; seen.add(key)
            sample = lookup[key]
            assert row["state_hash"] == sample["state_hash"] and row["source_policy_hash"] == sample["policy_hash"]
            assert row["label"] == label and row["action"] in (1, 2, 3)
            action = ACTIONS[row["action"] - 1]; arm = sample["arms"][action]
            assert f32(row["future_loss"]) == f32(arm["future_after"]), "readout joined to wrong measured action"
            if label in ACTIONS: assert action == label and row["forced"]
            row.update(split=split, continuation=continuation)
            for field in ("immediate_after", "origin_after_horizon", "heldout_after"): row[field] = arm[field]
            result.append(row)
        assert seen == set(lookup)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    indexed = {(row["split"], row["continuation"], row["body"], row["seed"], row["checkpoint"], row["label"]): row for row in rows}
    result = []
    for split in sorted({row["split"] for row in rows}):
        for continuation in ("hold", "policy"):
            for body in ("simple", "hevlm"):
                for label in LABELS:
                    group = [row for row in rows if (row["split"], row["continuation"], row["body"], row["label"]) == (split, continuation, body, label)]
                    if not group: continue
                    changes = {baseline: sum(row["action"] != indexed[(split, continuation, body, row["seed"], row["checkpoint"], baseline)]["action"] for row in group)
                               for baseline in ("parent", "lived-hold")}
                    result.append({"split": split, "continuation": continuation, "body": body, "label": label,
                                   "states": len(group), "actions": dict(collections.Counter(ACTIONS[row["action"] - 1] for row in group)),
                                   "optimal": sum(row["optimal"] for row in group),
                                   "mean_regret": sum(row["regret"] for row in group) / len(group),
                                   "mean_advantage_vs_hold": sum(row["advantage"] for row in group) / len(group), "choice_changes": changes})
    return result


def verify_terminal_artifacts(output: Path, anchors: dict) -> dict:
    """Production terminal gate, also exercised by retained-file mutation tests."""
    protocol, steps = anchors["protocol"], anchors["steps"]
    for relative, recorded in anchors["files"].items():
        assert identity(output / relative) == recorded, "persisted artifact changed: " + relative
    source_steps = 0
    for cohort in anchors["cohorts"]:
        prefix = output / cohort["prefix"]
        control, diagnostic = Path(str(prefix) + "-control"), Path(str(prefix) + "-diagnostic")
        left, right = scenarios.events(Path(str(control) + ".jsonl")), scenarios.events(Path(str(diagnostic) + ".jsonl"))
        assert validate_cohort(left, cohort["body"], cohort["seed"], protocol, steps, False) == cohort["control"]
        assert validate_cohort(right, cohort["body"], cohort["seed"], protocol, steps, True) == cohort["diagnostic"]
        assert compare_hosts(left, right, control, diagnostic, steps) == cohort["parity"]
        source_steps += steps
    for label, fit in anchors["fits"].items():
        samples = scenarios.strict_json((output / (label + ".samples.json")).read_text())
        rows = scenarios.events(output / (label + ".fit.jsonl"))
        assert validate_fit_rows(samples, rows, protocol["fitting"]["epochs"]) == fit["validation"]
    for name, receipt in anchors["readouts"].items():
        assert scenarios.events(output / name) == receipt["events"], "readout events changed"
    for rollout in anchors["rollouts"]:
        prefix = output / rollout["prefix"]
        rows = scenarios.events(Path(str(prefix) + ".jsonl"))
        assert conditional.validate_rollout(rows, steps, rollout["arm"] in anchors["lives"], rollout["resume_step"]) == rollout["summary"]
    for continuation in anchors["continuation"]:
        prefix, resumed = output / continuation["prefix"], output / continuation["resumed_prefix"]
        assert conditional.compare_resume(scenarios.events(Path(str(prefix) + ".jsonl")),
                                           scenarios.events(Path(str(resumed) + ".jsonl")), prefix, resumed, steps) == continuation["validation"]
    for comparison in anchors["source_deployment"]:
        source, deployed = output / comparison["source_prefix"], output / comparison["deployment_prefix"]
        assert compare_hosts(scenarios.events(Path(str(source) + ".jsonl")), scenarios.events(Path(str(deployed) + ".jsonl")),
                             source, deployed, steps) == comparison["validation"]
    return {"status": "PASS", "artifact_identities": len(anchors["files"]), "paired_source_steps": source_steps,
            "cohorts": len(anchors["cohorts"]), "fit_traces": len(anchors["fits"]), "readout_traces": len(anchors["readouts"]),
            "rollouts": len(anchors["rollouts"]), "policy_continuations": len(anchors["continuation"]),
            "source_deployment_pairs": len(anchors["source_deployment"])}


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
    if v1.digest(previous / "results.json") != protocol["previous_results_sha256"]: parser.error("previous results identity mismatch")
    old = scenarios.strict_json((previous / "results.json").read_text())
    previous_verified = future.authenticate_previous(previous, old)
    if identity(previous / protocol["parent"]["file"]) != {key: protocol["parent"][key] for key in ("bytes", "sha256")}:
        parser.error("sealed parent identity mismatch")
    status = v1.command(["git", "status", "--porcelain"])
    if status and not args.allow_dirty: parser.error("final run requires clean committed source")
    if args.allow_dirty and not args.smoke: parser.error("dirty runs are restricted to development smoke")
    steps = protocol["smoke"]["host_steps"] if args.smoke else protocol["host_steps"]
    seeds = {split: protocol["smoke"]["seeds"] if args.smoke else protocol[name]["seeds"]
             for split, name in (("development", "acquisition"), ("evaluation", "evaluation"))}
    output.mkdir(parents=True)
    env = dict(os.environ, NT_SIMD_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    source_names = ["notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
                    "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h", "examples/chuck_architect_rollout.h",
                    "examples/chuck_architect_lived.h", "examples/chuck_architect_future.c", "examples/chuck-loss-architect.json",
                    "experiments/chuck_loss_architect/run.py", "experiments/chuck_loss_architect/scenarios/run.py",
                    "experiments/chuck_loss_architect/future/run.py", "experiments/chuck_loss_architect/conditional/run.py",
                    "experiments/chuck_loss_architect/lived/run.py", "experiments/chuck_loss_architect/lived/protocol.json"]
    source_hashes = {name: v1.digest(ROOT / name) for name in source_names}
    host_binary, fitter = output / "host_runner", output / "future_runner"
    scenarios.build(host_binary); scenarios.build(fitter, ROOT / "examples/chuck_architect_future.c")
    input_identities = {path.name: identity(path) for path in (host_binary, fitter)}
    config = ROOT / "examples/chuck-loss-architect.json"
    shutil.copyfile(config, output / "architect-config.json"); shutil.copyfile(HERE / "protocol.json", output / "protocol.json")
    for name in ("architect-config.json", "protocol.json"): input_identities[name] = identity(output / name)
    datasets = old["manifest"]["datasets"]
    for body in protocol["body_order"]:
        for suffix in (".corpus", ".tokens.u32", ".vocab.json"):
            name = body + suffix; shutil.copyfile(previous / name, output / name)
            assert identity(output / name) == old["artifacts"][name]
            input_identities[name] = identity(output / name)
    lives = {label: output / (label + ".policy.bin") for label in ("parent", *protocol["students"])}
    shutil.copyfile(previous / protocol["parent"]["file"], lives["parent"])
    input_identities[lives["parent"].name] = identity(lives["parent"])
    manifest = {"source_commit": v1.command(["git", "rev-parse", "HEAD"]), "source_status": status,
                "source_sha256": source_hashes, "protocol": protocol, "protocol_sha256": v1.digest(HERE / "protocol.json"),
                "datasets": datasets, "machine": v1.machine(), "compiler": v1.command(["cc", "--version"]).splitlines()[0],
                "compile_flags": "-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread",
                "binaries_sha256": {"host": v1.digest(host_binary), "future": v1.digest(fitter)}, "smoke": args.smoke,
                "executed_seeds": seeds, "executed_host_steps": steps, "previous_results_sha256": v1.digest(previous / "results.json"),
                "previous_artifacts_verified": previous_verified}
    write_json(output / "manifest.json", manifest)
    input_identities["manifest.json"] = identity(output / "manifest.json")
    timeline = output / "timeline.jsonl"
    def phase(name, **fields):
        with timeline.open("a") as stream:
            stream.write(json.dumps({"phase": name, "time_ns": time.time_ns(), **fields}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    files, cohorts, fit_receipts, readout_anchors, rollouts, continuation = dict(input_identities), [], {}, {}, [], []
    source_deployment = []
    def anchor_prefix(prefix):
        for path in sorted(prefix.parent.glob(prefix.name + ".*")):
            if path.is_file(): files[path.relative_to(output).as_posix()] = identity(path)
    def cohort(split, body, seed):
        directory = output / "cohorts"; directory.mkdir(exist_ok=True)
        stem = directory / f"{split}-{body}-s{seed}"
        prefixes, traces, validated = {}, {}, {}
        for probes, label in ((False, "control"), (True, "diagnostic")):
            prefix = Path(str(stem) + "-" + label)
            print("RUN", prefix.name, flush=True)
            future.execute([str(host_binary), "--lived", body, str(output / (body + ".tokens.u32")), str(prefix),
                            str(steps), str(seed), str(protocol["body_lr"]), str(config), str(lives["parent"]), str(int(probes))],
                           Path(str(prefix) + ".log"), env)
            rows = scenarios.events(Path(str(prefix) + ".jsonl"))
            validated[label] = validate_cohort(rows, body, seed, protocol, steps, probes)
            prefixes[label], traces[label] = prefix, rows
            anchor_prefix(prefix)
        parity = compare_hosts(traces["control"], traces["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps)
        cohorts.append({"split": split, "body": body, "seed": seed, "prefix": stem.relative_to(output).as_posix(),
                        "control": validated["control"], "diagnostic": validated["diagnostic"], "parity": parity})
        write_json(output / "cohort_receipts.json", cohorts)
        return {kind: gather_samples(traces["diagnostic"], body, seed, prefixes["diagnostic"], output, kind) for kind in protocol["continuations"]}
    def collect(split):
        collected = {kind: [] for kind in protocol["continuations"]}
        for body in protocol["body_order"]:
            for seed in seeds[split]:
                samples = cohort(split, body, seed)
                for kind in collected: collected[kind].extend(samples[kind])
        expected = {(body, seed, checkpoint) for body in protocol["body_order"] for seed in seeds[split]
                    for checkpoint in protocol["checkpoints_before_updates"] if checkpoint <= steps}
        for kind, samples in collected.items():
            assert len(samples) == len(expected) and {(s["body"], s["seed"], s["checkpoint"]) for s in samples} == expected
            table = output / (split + "-" + kind + ".samples.bin")
            future.write_samples(table, samples)
            files[table.name] = identity(table)
            files[table.with_suffix(".json").name] = identity(table.with_suffix(".json"))
        for hold, policy in zip(collected["hold"], collected["policy"]):
            for field in ("body", "seed", "checkpoint", "state_hash", "policy_hash", "observation", "features", "source_policy"):
                assert hold[field] == policy[field], "students do not share the same captured source worlds"
        return collected
    phase("acquisition_started", parent_sha256=v1.digest(lives["parent"]))
    development = collect("development")
    for label, student in protocol["students"].items():
        samples = development[student["continuation"]]
        table, trace = output / (label + ".samples.bin"), output / (label + ".fit.jsonl")
        future.write_samples(table, samples)
        future.execute([str(fitter), "fit-conditioned", str(lives["parent"]), str(table), str(lives[label]), str(trace),
                        str(protocol["fitting"]["epochs"]), student["life_id"]], output / (label + ".fit.log"), env)
        validation = validate_fit_rows(samples, scenarios.events(trace), protocol["fitting"]["epochs"])
        fit_receipts[label] = {"continuation": student["continuation"], "parent_sha256": v1.digest(lives["parent"]),
                               "validation": validation, "sample_table_sha256": v1.digest(table), "fit_trace_sha256": v1.digest(trace)}
        anchor_prefix(output / label)
    sealed = {label: {"file": path.name, **identity(path)} for label, path in lives.items()}
    seal = {"protocol_sha256": manifest["protocol_sha256"], "lives": sealed, "fit_receipts": fit_receipts, "evaluation_generation_started": False}
    write_json(output / "sealed_lives.json", seal); phase("lives_sealed", sealed_lives_sha256=v1.digest(output / "sealed_lives.json"))
    files["sealed_lives.json"] = identity(output / "sealed_lives.json")
    outputs = []
    for kind, samples in development.items():
        outputs.extend(readouts(fitter, output / ("development-" + kind + ".samples.bin"), samples, lives, output, "development", kind, env, readout_anchors))
    phase("evaluation_generation_started", sealed_lives_sha256=v1.digest(output / "sealed_lives.json"))
    evaluation = collect("evaluation")
    for kind, samples in evaluation.items():
        outputs.extend(readouts(fitter, output / ("evaluation-" + kind + ".samples.bin"), samples, lives, output, "evaluation", kind, env, readout_anchors))
    phase("closed_loop_started")
    rollout_dir = output / "rollouts"; rollout_dir.mkdir()
    for body in protocol["closed_loop"]["bodies"]:
        for seed in seeds["evaluation"]:
            def deploy(arm, resume_step):
                prefix = rollout_dir / (f"{body}-s{seed}-{arm}" + ("-resume" if resume_step else ""))
                print("RUN", prefix.name, flush=True)
                future.execute([str(host_binary), "--rollout", body, str(output / (body + ".tokens.u32")), str(prefix),
                                str(steps), str(seed), str(protocol["body_lr"]), str(config), arm,
                                str(lives[arm]) if arm in lives else "-", str(resume_step)], Path(str(prefix) + ".log"), env)
                rows = scenarios.events(Path(str(prefix) + ".jsonl"))
                summary = conditional.validate_rollout(rows, steps, arm in lives, resume_step)
                rollouts.append({"body": body, "seed": seed, "arm": arm, "resume_step": resume_step,
                                 "prefix": prefix.relative_to(output).as_posix(), "summary": summary})
                anchor_prefix(prefix); write_json(output / "rollout_receipts.json", rollouts)
                return prefix, rows
            runs = {arm: deploy(arm, 0) for arm in protocol["closed_loop"]["arms"]}
            initial = [Path(str(prefix) + ".initial.bin").read_bytes() for prefix, rows in runs.values()]
            assert all(data == initial[0] for data in initial), "deployment initial bodies differ"
            source = next(cohort for cohort in cohorts if (cohort["split"], cohort["body"], cohort["seed"]) == ("evaluation", body, seed))
            source_prefix = output / (source["prefix"] + "-control")
            deployed, deployed_rows = runs["parent"]
            source_deployment.append({"body": body, "seed": seed, "source_prefix": source_prefix.relative_to(output).as_posix(),
                                      "deployment_prefix": deployed.relative_to(output).as_posix(),
                                      "validation": compare_hosts(scenarios.events(Path(str(source_prefix) + ".jsonl")), deployed_rows,
                                                                   source_prefix, deployed, steps)})
            resume_arm = protocol["closed_loop"]["policy_continuation"]["arm"]
            resume_step = min(protocol["closed_loop"]["policy_continuation"]["save_load_after_step"], steps // 2)
            resumed, resumed_rows = deploy(resume_arm, resume_step)
            prefix, rows = runs[resume_arm]
            continuation.append({"body": body, "seed": seed, "resume_step": resume_step,
                                 "prefix": prefix.relative_to(output).as_posix(), "resumed_prefix": resumed.relative_to(output).as_posix(),
                                 "validation": conditional.compare_resume(rows, resumed_rows, prefix, resumed, steps)})
    for name, receipt in readout_anchors.items(): files[name] = receipt["identity"]
    for label, path in lives.items(): assert identity(path) == {key: sealed[label][key] for key in ("bytes", "sha256")}, "sealed life changed"
    anchors = {"protocol": protocol, "steps": steps, "files": files, "cohorts": cohorts, "fits": fit_receipts,
               "readouts": readout_anchors, "rollouts": rollouts, "continuation": continuation, "lives": list(lives),
               "source_deployment": source_deployment}
    write_json(output / "terminal_anchors.json", anchors)
    terminal = verify_terminal_artifacts(output, anchors)
    for name, digest in source_hashes.items(): assert v1.digest(ROOT / name) == digest, "source changed during experiment: " + name
    phase("evaluation_complete", source_states=sum(len(v) for v in evaluation.values()) // 2, rollout_processes=len(rollouts))
    result = {"manifest": manifest, "fit_receipts": fit_receipts, "sealed_lives": seal, "cohorts": cohorts,
              "summary": summarize(outputs), "readouts": outputs, "rollouts": rollouts, "policy_continuation": continuation,
              "source_deployment_parity": source_deployment,
              "terminal_verification": terminal, "source_hashes_rechecked": True, "sealed_lives_unchanged": True,
              "artifacts": {path.relative_to(output).as_posix(): identity(path) for path in sorted(output.rglob("*"))
                            if path.is_file() and (path.suffix in (".json", ".jsonl", ".log", ".bin", ".u32", ".corpus") or
                                                   path.name in ("host_runner", "future_runner"))}}
    write_json(output / "results.json", result)
    print("LIVED_OK", json.dumps({"development_worlds": len(development["policy"]), "evaluation_worlds": len(evaluation["policy"]),
                                  "fit_updates": sum(v["validation"]["fit_steps"] for v in fit_receipts.values()),
                                  "readouts": len(outputs), "rollout_steps": sum(v["summary"]["steps"] for v in rollouts)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
