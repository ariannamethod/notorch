#!/usr/bin/env python3
"""One fixed acquisition round: student worlds versus matched parent worlds."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("chuck_lived", HERE.parent / "lived/run.py")
lived = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(lived)
conditional, future, scenarios, v1 = lived.conditional, lived.future, lived.scenarios, lived.v1
write_json, identity, f32 = lived.write_json, lived.identity, lived.f32
ACTIONS = lived.ACTIONS


class DurableArtifacts:
    """One original byte anchor and one independent gzip copy per closed file."""
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.files, self.mirrors, self.modes, self.attempts = {}, {}, {}, {}
        self.failures = []
        self.index = self.output / ".sealed" / "anchors.json"
        if self.index.exists():
            raise ValueError("durability state already exists; choose a fresh run")

    @staticmethod
    def byte_identity(data: bytes) -> dict:
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    @staticmethod
    def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)

    def persist(self) -> None:
        value = {"schema": 1, "max_restore_attempts_per_file": 1, "files": self.files,
                 "mirrors": self.mirrors, "modes": self.modes, "attempts": self.attempts,
                 "integrity_failures": self.failures}
        self.atomic_write(self.index, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def name(self, path: Path) -> str:
        return path.absolute().relative_to(self.output).as_posix()

    def anchor(self, path: Path, data: bytes) -> None:
        """The parser and mirror receive these exact first-read bytes."""
        name, recorded = self.name(path), self.byte_identity(data)
        if name in self.files:
            if self.files[name] != recorded:
                raise RuntimeError("refusing to reanchor changed artifact: " + name)
            return
        mirror = self.output / ".sealed" / (name + ".gz")
        compressed = gzip.compress(data, mtime=0)
        self.atomic_write(mirror, compressed)
        self.files[name] = recorded
        self.mirrors[name] = {"path": mirror.relative_to(self.output).as_posix(), **self.byte_identity(compressed)}
        self.modes[name] = path.stat().st_mode & 0o777
        self.persist()

    def restore(self, path: Path, damaged: bytes | None) -> bytes:
        name = self.name(path)
        evidence = self.output / "integrity-failures" / f"{len(self.failures) + 1:04d}"
        evidence.mkdir(parents=True, exist_ok=False)
        observed = self.byte_identity(damaged) if damaged is not None else {"missing": True}
        if damaged is not None: self.atomic_write(evidence / "damaged.bin", damaged)
        receipt = {"artifact": name, "original": dict(self.files[name]), "observed": observed,
                   "evidence": evidence.relative_to(self.output).as_posix(), "status": "refused"}
        self.failures.append(receipt)
        try:
            if self.attempts.get(name, 0):
                raise RuntimeError("restore attempt already consumed: " + name)
            self.attempts[name] = 1
            self.persist()
            mirror = self.output / self.mirrors[name]["path"]
            compressed = mirror.read_bytes()
            if self.byte_identity(compressed) != {key: self.mirrors[name][key] for key in ("bytes", "sha256")}:
                raise RuntimeError("sealed gzip identity differs: " + name)
            original = gzip.decompress(compressed)
            if self.byte_identity(original) != self.files[name]:
                raise RuntimeError("sealed gzip does not reproduce original anchor: " + name)
            self.atomic_write(path, original, self.modes[name])
            restored = path.read_bytes()
            if self.byte_identity(restored) != self.files[name]:
                raise RuntimeError("restored artifact differs from original anchor: " + name)
            receipt.update(status="restored", restored=self.byte_identity(restored))
            return restored
        except Exception as error:
            receipt.update(exception=type(error).__name__, reason=str(error))
            raise
        finally:
            self.atomic_write(evidence / "receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
            self.persist()

    def read_bytes(self, path: Path) -> bytes:
        name = self.name(path)
        try: data = path.read_bytes()
        except FileNotFoundError:
            if name not in self.files: raise
            return self.restore(path, None)
        if name in self.files and self.byte_identity(data) != self.files[name]:
            return self.restore(path, data)
        self.anchor(path, data)
        return data

    def parsed(self, path: Path, parser):
        if self.name(path) in self.files:
            return parser(self.read_bytes(path).decode("utf-8"))
        data = path.read_bytes()
        parsed = parser(data.decode("utf-8"))
        self.anchor(path, data)
        return parsed

    def events(self, path: Path) -> list[dict]:
        return self.parsed(path, lambda text: [scenarios.strict_json(line) for line in text.splitlines()])

    def json(self, path: Path):
        return self.parsed(path, scenarios.strict_json)

    def identity(self, path: Path) -> dict:
        self.read_bytes(path)
        return dict(self.files[self.name(path)])

    def require(self, paths) -> None:
        for path in paths: self.read_bytes(path)

    def prefix(self, prefix: Path) -> None:
        self.require(path for path in sorted(prefix.parent.glob(prefix.name + ".*")) if path.is_file())


def compare_files(left: Path, right: Path, suffixes, durable: DurableArtifacts) -> dict:
    artifacts = {}
    for suffix in suffixes:
        a, b = Path(str(left) + suffix), Path(str(right) + suffix)
        assert durable.read_bytes(a) == durable.read_bytes(b), "compared artifact differs: " + suffix
        artifacts[suffix] = dict(durable.files[durable.name(a)])
    return artifacts


def compare_hosts(left, right, a: Path, b: Path, steps: int, durable: DurableArtifacts) -> dict:
    assert [row for row in left if row["type"] in ("rollout_step", "evaluation")] == [row for row in right if row["type"] in ("rollout_step", "evaluation")], "diagnostics changed source host"
    artifacts = compare_files(a, b, (".initial.bin", ".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"), durable)
    return {"status": "PASS", "host_steps_compared": steps, "artifacts": artifacts}


def compare_resume(left, right, a: Path, b: Path, steps: int, durable: DurableArtifacts) -> dict:
    assert [row for row in left if row["type"] in ("rollout_step", "evaluation")] == [row for row in right if row["type"] in ("rollout_step", "evaluation")], "resumed continuation differs"
    artifacts = compare_files(a, b, (".final.bin", ".moments.final.bin", ".optimizer.final.json", ".policy.final.bin"), durable)
    return {"status": "PASS", "steps_compared": steps, "final_artifacts": artifacts,
            "scope": "complete Architect saved/reloaded in process; real body continues under its next selections"}


def validate_feedback(policy: dict) -> None:
    """The recorded consequence must be the actual same-window change."""
    receipt = policy["consequence"]
    before, after = f32(receipt["before_loss"]), f32(receipt["after_loss"])
    assert lived.finite(before) and lived.finite(after), "nonfinite feedback cannot enter this acquisition"
    reward = f32(max(-1.0, min(1.0, (before - after) / (abs(before) + 1e-6))))
    assert f32(receipt["reward"]) == reward, "feedback reward does not match the actual consequence"
    assert receipt["learned"] == receipt["nonfinite"] == receipt["error"] == 0
    assert f32(receipt["predicted"]) == f32(policy["scores"][policy["action"]["kind"] - 1])


def validate_cohort(rows: list[dict], body: str, seed: int, source: str,
                    protocol: dict, steps: int, probes: bool) -> dict:
    """Reuse the lived source gate, then authenticate the distinct continuation."""
    expected = ["policy", "student"] if source == "parent" else ["policy"]
    aliases = {} if source == "parent" else {"student": "policy"}
    header = rows[0]
    assert header["diagnostics"] == "trajectory"
    assert header["executed_continuations"] == expected and header["continuation_aliases"] == aliases
    assert header["source_saved_policy_hash"] == header["sealed_policy_hash"]
    branch_rows = [row for row in rows if row["type"] == "branch_step"]
    measurements = [row for row in rows if row["type"] == "comparison"]
    assert all(row["continuation"] in expected for row in branch_rows + measurements), "unexpected continuation"
    source_rows = [row for row in rows if row["type"] not in ("branch_step", "comparison") or row["continuation"] == "policy"]
    source_protocol = dict(protocol, continuations=["policy"])
    verified = lived.validate_cohort(source_rows, body, seed, source_protocol, steps, probes)
    horizon = protocol["target_horizon_total_updates"]
    horizons = protocol["diagnostic_horizons_total_updates"]
    forks = [row for row in rows if row["type"] == "fork"]
    assert len(branch_rows) == len(forks) * len(expected) * 3 * horizon
    assert len(measurements) == len(forks) * len(expected) * 3 * len(horizons)
    for row in rows:
        if row["type"] in ("rollout_step", "branch_step"):
            validate_feedback(row["architect"])
    for fork in forks:
        assert fork["executed_continuations"] == expected and fork["continuation_aliases"] == aliases
        if aliases:
            assert fork["student_policy_hash"] == fork["policy_hash"]
            assert fork["student_state_hash"] == fork["state_hash"]
        for continuation in expected:
            policy_hash = fork["student_policy_hash"] if continuation == "student" else fork["policy_hash"]
            state_hash = fork["student_state_hash"] if continuation == "student" else fork["state_hash"]
            for action_index, action in enumerate(ACTIONS, 1):
                branch = [row for row in branch_rows if (row["checkpoint"], row["continuation"], row["intervention"]) == (fork["step"], continuation, action)]
                assert [row["update"] for row in branch] == list(range(1, horizon + 1)), "incomplete continuation"
                for index, row in enumerate(branch):
                    policy, previous = row["architect"], branch[index - 1] if index else None
                    receipt = policy["consequence"]
                    assert row["initial_policy_hash"] == policy_hash and row["branch_initial_hash"] == state_hash
                    assert row["source_hash"] == fork["state_hash"] and row["offset"] == fork["offsets"][index]
                    assert policy["policy_pre"] == (previous["architect"]["policy_post"] if previous else policy_hash), "continuation history hash chain broken"
                    assert row["forced"] is (index == 0)
                    assert policy["policy_pending"] is None if index == 0 else isinstance(policy["policy_pending"], str)
                    assert row["action"] in (1, 2, 3) and row["executed_action"] == ACTIONS[row["action"] - 1]
                    if index == 0:
                        assert row["action"] == action_index and policy["features"] == fork["features"]
                    else:
                        assert row["action"] == 1 + max(range(3), key=lambda k: policy["scores"][k]), "continuation did not select its recorded policy"
                        assert f32(policy["features"][15]) == f32(previous["architect"]["consequence"]["reward"])
                    assert policy["sequence"] == receipt["decision"] == fork["step"] + index
                    assert policy["action"]["kind"] == row["action"] and policy["explored"] == 0
                    assert row["post_chuck"]["global_step"] == row["pre_chuck"]["global_step"] + 1
                    assert policy["observation"]["loss"] == receipt["before_loss"] == row["before"]
                    assert receipt["after_loss"] == row["after_same_window"]
                    assert lived.finite(row["gradient_norm"]) and row["gradient_norm_stage"] == "before_clip"
                for measured_horizon in horizons:
                    matches = [row for row in measurements if (row["checkpoint"], row["continuation"], row["action"], row["horizon"]) == (fork["step"], continuation, action, measured_horizon)]
                    assert len(matches) == 1, "measurement identity duplicate or missing"
                    measured = matches[0]
                    assert measured["initial_hash"] == fork["state_hash"]
                    assert measured["initial_policy_hash"] == policy_hash and measured["branch_initial_hash"] == state_hash
                    assert measured["state_hash"] == branch[measured_horizon - 1]["state_hash"], "outcome belongs to another branch"
                    assert measured["finite"] is True
                    assert measured["immediate_before"] == fork["loss_before"]
                    assert measured["immediate_after"] == branch[0]["after_same_window"]
                    assert measured["future_before"] == fork["future_probe_before"]
                    assert measured["heldout_before"] == fork["heldout_before"]
                    assert all(lived.finite(measured[name]) for name in ("immediate_after", "origin_after_horizon", "future_after", "heldout_after"))
    return {**verified, "branch_updates": len(branch_rows), "measurements": len(measurements),
            "executed_continuations": expected, "continuation_aliases": aliases}


def validate_alias_files(rows: list[dict], prefix: Path, durable: DurableArtifacts | None = None) -> int:
    compared = 0
    for fork in (row for row in rows if row["type"] == "fork"):
        source = Path(str(prefix) + f".fork-{fork['step']}.policy.bin")
        student = Path(str(prefix) + f".fork-{fork['step']}.student.policy.bin")
        if durable: durable.require([source, student])
        else: assert source.is_file() and student.is_file()
        if fork["continuation_aliases"]:
            read = durable.read_bytes if durable else Path.read_bytes
            assert read(source) == read(student), "aliased continuation lives differ"
            compared += 1
    return compared


def gather_samples(rows: list[dict], body: str, seed: int, source: str,
                   prefix: Path, output: Path) -> list[dict]:
    table = rows[0]["continuation_aliases"].get("student", "student")
    samples = lived.gather_samples(rows, body, seed, prefix, output, table)
    for sample in samples:
        sample.update(source=source, continuation="student", executed_continuation=table,
                      outcome_source="executed_shared_student_continuation")
    return samples


def compare_prefix(source_rows: list[dict], deployed_rows: list[dict], steps: int) -> dict:
    """The source ends earlier; compare transitions only at matching updates."""
    source = [row for row in source_rows if row["type"] == "rollout_step"]
    deployed = [row for row in deployed_rows if row["type"] == "rollout_step" and row["step"] <= steps]
    assert len(source) == len(deployed) == steps
    assert source == deployed, "student source differs from its deployment prefix"
    left = [row for row in source_rows if row["type"] == "evaluation"]
    right = [row for row in deployed_rows if row["type"] == "evaluation" and row["step"] <= steps]
    # A smoke source ends at 17, which is not a deployment evaluation checkpoint.
    common = {row["step"] for row in right}
    assert [row for row in left if row["step"] in common] == right
    return {"status": "PASS", "transitions": steps, "evaluations": len(right),
            "final_matching_state_hash": source[-1]["state_hash"]}


def validate_readouts(rows: list[dict], samples: list[dict], label: str) -> None:
    lookup = {(sample["body"], sample["seed"], sample["checkpoint"]): sample for sample in samples}
    assert len(lookup) == len(samples) == len(rows)
    seen = set()
    for row in rows:
        key = row["body"], row["seed"], row["checkpoint"]
        assert key in lookup and key not in seen; seen.add(key)
        sample = lookup[key]
        assert row["label"] == label and row["action"] in (1, 2, 3)
        assert row["state_hash"] == sample["state_hash"] and row["source_policy_hash"] == sample["policy_hash"]
        losses = list(map(f32, sample["future_loss"]))
        selected = row["action"] - 1
        assert f32(row["future_loss"]) == losses[selected] and f32(row["hold_loss"]) == losses[0], "readout joined to wrong action outcome"
        assert row["regret"] == losses[selected] - min(losses)
        assert row["advantage"] == losses[0] - losses[selected]
        assert row["optimal"] is (losses[selected] == min(losses))
        if label in ACTIONS:
            assert row["forced"] and selected == ACTIONS.index(label)
        else:
            assert not row["forced"] and selected == max(range(3), key=lambda k: row["scores"][k])
    assert seen == set(lookup)


def readouts(binary: Path, table: Path, samples: list[dict], lives: dict[str, Path],
             output: Path, split: str, source: str, labels: list[str], env: dict,
             anchors: dict, durable: DurableArtifacts) -> list[dict]:
    result = []
    lookup = {(sample["body"], sample["seed"], sample["checkpoint"]): sample for sample in samples}
    for label in labels:
        trace = output / f"{split}-{source}-{label}.readout.jsonl"
        durable.require([binary, table, *(output / sample["source_policy"] for sample in samples)])
        if label in lives: durable.require([lives[label]])
        future.execute([str(binary), "eval", str(table), str(lives[label]) if label in lives else "-", label, str(trace)], trace.with_suffix(".log"), env)
        rows = durable.events(trace)
        durable.require([trace.with_suffix(".log")])
        validate_readouts(rows, samples, label)
        anchors[trace.name] = {"identity": durable.identity(trace), "label": label,
                               "samples": table.with_suffix(".json").name}
        for row in rows:
            sample = lookup[(row["body"], row["seed"], row["checkpoint"])]
            arm = sample["arms"][ACTIONS[row["action"] - 1]]
            row.update(split=split, source=source, continuation="student")
            for name in ("immediate_after", "origin_after_horizon", "heldout_after"):
                row[name] = arm[name]
            result.append(row)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    groups = collections.defaultdict(list)
    indexed = {}
    for row in rows:
        groups[(row["split"], row["source"], row["body"], row["label"])].append(row)
        indexed[(row["split"], row["source"], row["body"], row["seed"], row["checkpoint"], row["label"])] = row
    result = []
    for (split, source, body, label), group in sorted(groups.items()):
        result.append({"split": split, "source": source, "body": body, "label": label,
                       "states": len(group), "actions": dict(collections.Counter(ACTIONS[row["action"] - 1] for row in group)),
                       "optimal": sum(row["optimal"] for row in group),
                       "mean_regret": sum(row["regret"] for row in group) / len(group),
                       "mean_advantage_vs_hold": sum(row["advantage"] for row in group) / len(group),
                       "choice_changes_vs_student": sum(row["action"] != indexed[(split, source, body, row["seed"], row["checkpoint"], "student")]["action"] for row in group)})
    return result


def authenticate_inputs(previous: Path, protocol: dict) -> tuple[dict, dict]:
    assert v1.digest(previous / "results.json") == protocol["previous_results_sha256"], "previous results identity mismatch"
    old = scenarios.strict_json((previous / "results.json").read_text())
    assert v1.digest(previous / "protocol.json") == protocol["previous_protocol_sha256"], "previous protocol identity mismatch"
    selected = {"results.json": identity(previous / "results.json"), "protocol.json": identity(previous / "protocol.json")}
    for body in protocol["body_order"]:
        for suffix in (".corpus", ".tokens.u32", ".vocab.json"):
            name = body + suffix
            assert identity(previous / name) == old["artifacts"][name], "previous corpus identity mismatch: " + name
            selected[name] = old["artifacts"][name]
    for life in protocol["input_lives"].values():
        name = life["file"]
        recorded = {key: life[key] for key in ("bytes", "sha256")}
        assert identity(previous / name) == recorded == old["artifacts"][name], "input life identity mismatch: " + name
        selected[name] = recorded
    name = "architect-config.json"
    assert identity(previous / name) == old["artifacts"][name]
    assert identity(ROOT / "examples/chuck-loss-architect.json") == old["artifacts"][name], "current configuration differs from pinned experience"
    selected[name] = old["artifacts"][name]
    return old, selected


def verify_terminal_artifacts(output: Path, anchors: dict, durable: DurableArtifacts | None = None) -> dict:
    get_identity = durable.identity if durable else identity
    events = durable.events if durable else scenarios.events
    read_json = durable.json if durable else lambda path: scenarios.strict_json(path.read_text())
    for name, recorded in anchors["files"].items():
        assert get_identity(output / name) == recorded, "persisted artifact changed: " + name
    protocol, steps = anchors["protocol"], anchors["host_steps"]
    for cohort in anchors["cohorts"]:
        stem = output / cohort["prefix"]
        prefixes = {label: Path(str(stem) + "-" + label) for label in ("control", "diagnostic")}
        rows = {label: events(Path(str(prefix) + ".jsonl")) for label, prefix in prefixes.items()}
        for label, probes in (("control", False), ("diagnostic", True)):
            assert validate_cohort(rows[label], cohort["body"], cohort["seed"], cohort["source"], protocol, steps, probes) == cohort[label]
        assert validate_alias_files(rows["diagnostic"], prefixes["diagnostic"], durable) == cohort["aliased_lives"]
        comparison = compare_hosts(rows["control"], rows["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps, durable) if durable else lived.compare_hosts(rows["control"], rows["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps)
        assert comparison == cohort["parity"]
    for label, fit in anchors["fits"].items():
        samples = read_json(output / (label + ".samples.json"))
        assert lived.validate_fit_rows(samples, events(output / (label + ".fit.jsonl")), anchors["epochs"]) == fit["validation"]
    for name, readout in anchors["readouts"].items():
        samples = read_json(output / readout["samples"])
        validate_readouts(events(output / name), samples, readout["label"])
    for rollout in anchors["rollouts"]:
        rows = events(output / (rollout["prefix"] + ".jsonl"))
        assert conditional.validate_rollout(rows, anchors["deployment_steps"], rollout["arm"] in anchors["lives"], rollout["resume_step"]) == rollout["summary"]
    for comparison in anchors["source_deployment"]:
        source = events(output / (comparison["source_prefix"] + ".jsonl"))
        deployed = events(output / (comparison["deployment_prefix"] + ".jsonl"))
        assert compare_prefix(source, deployed, steps) == comparison["validation"]
    for continuation in anchors["continuation"]:
        prefix, resumed = output / continuation["prefix"], output / continuation["resumed_prefix"]
        left, right = events(Path(str(prefix) + ".jsonl")), events(Path(str(resumed) + ".jsonl"))
        comparison = compare_resume(left, right, prefix, resumed, anchors["deployment_steps"], durable) if durable else conditional.compare_resume(left, right, prefix, resumed, anchors["deployment_steps"])
        assert comparison == continuation["validation"]
    return {"status": "PASS", "artifact_identities": len(anchors["files"]), "cohorts": len(anchors["cohorts"]),
            "paired_source_steps": len(anchors["cohorts"]) * steps, "fit_traces": len(anchors["fits"]),
            "readout_traces": len(anchors["readouts"]), "rollouts": len(anchors["rollouts"]),
            "source_deployment_pairs": len(anchors["source_deployment"]), "policy_continuations": len(anchors["continuation"])}


def run(args: argparse.Namespace) -> int:
    output, previous = args.output.resolve(), args.previous_run.resolve()
    if output == ROOT or ROOT in output.parents: raise ValueError("output must be outside Git")
    if output == previous or previous in output.parents: raise ValueError("output must not modify the previous run")
    if output.exists(): raise ValueError("choose a fresh output directory")
    protocol = scenarios.strict_json((HERE / "protocol.json").read_text())
    old, selected_inputs = authenticate_inputs(previous, protocol)
    status = v1.command(["git", "status", "--porcelain"])
    if status and not args.allow_dirty: raise ValueError("final run requires clean committed source")
    if args.allow_dirty and not args.smoke: raise ValueError("dirty runs are restricted to development smoke")
    steps = protocol["smoke"]["host_steps"] if args.smoke else protocol["host_steps"]
    deploy_steps = protocol["smoke"]["deployment_steps"] if args.smoke else protocol["closed_loop"]["steps"]
    epochs = protocol["smoke"]["epochs"] if args.smoke else protocol["fitting"]["epochs"]
    seeds = {split: protocol["smoke"]["seeds"] if args.smoke else protocol[key]["seeds"]
             for split, key in (("development", "acquisition"), ("evaluation", "evaluation"))}
    output.mkdir(parents=True)
    durable = DurableArtifacts(output)
    source_names = ["notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
                    "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h", "examples/chuck_architect_rollout.h",
                    "examples/chuck_architect_lived.h", "examples/chuck_architect_future.c", "examples/chuck-loss-architect.json",
                    "experiments/chuck_loss_architect/run.py", "experiments/chuck_loss_architect/scenarios/run.py",
                    "experiments/chuck_loss_architect/future/run.py", "experiments/chuck_loss_architect/conditional/run.py",
                    "experiments/chuck_loss_architect/lived/run.py", "experiments/chuck_loss_architect/trajectories/run.py",
                    "experiments/chuck_loss_architect/trajectories/protocol.json"]
    hashes = {name: v1.digest(ROOT / name) for name in source_names}
    host, fitter = output / "host_runner", output / "future_runner"
    build_commands = [scenarios.build(host), scenarios.build(fitter, ROOT / "examples/chuck_architect_future.c")]
    durable.require([host, fitter])
    config = output / "architect-config.json"
    shutil.copyfile(ROOT / "examples/chuck-loss-architect.json", config)
    shutil.copyfile(HERE / "protocol.json", output / "protocol.json")
    durable.require([config, output / "protocol.json"])
    for body in protocol["body_order"]:
        for suffix in (".corpus", ".tokens.u32", ".vocab.json"):
            name = body + suffix; shutil.copyfile(previous / name, output / name)
            assert durable.identity(output / name) == selected_inputs[name]
    lives = {label: output / (label + ".policy.bin") for label in (*protocol["input_lives"], *protocol["students"])}
    for label, life in protocol["input_lives"].items():
        shutil.copyfile(previous / life["file"], lives[label])
        assert durable.identity(lives[label]) == selected_inputs[life["file"]]
    manifest = {"source_commit": v1.command(["git", "rev-parse", "HEAD"]), "source_status": status,
                "source_sha256": hashes, "protocol": protocol, "protocol_sha256": v1.digest(HERE / "protocol.json"),
                "datasets": old["manifest"]["datasets"], "machine": v1.machine(), "compiler": v1.command(["cc", "--version"]).splitlines()[0],
                "build_commands": build_commands, "compile_flags": "-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread",
                "binaries_sha256": {"host": durable.identity(host)["sha256"], "future": durable.identity(fitter)["sha256"]}, "smoke": args.smoke,
                "executed_seeds": seeds, "executed_host_steps": steps, "executed_deployment_steps": deploy_steps,
                "executed_epochs": epochs, "previous_selected_inputs": selected_inputs,
                "previous_results_sha256": protocol["previous_results_sha256"]}
    write_json(output / "manifest.json", manifest)
    durable.require([output / "manifest.json"])
    env = dict(os.environ, NT_SIMD_THREADS=str(protocol["threads"]), OMP_NUM_THREADS=str(protocol["threads"]), OPENBLAS_NUM_THREADS=str(protocol["threads"]))
    files = durable.files
    cohorts, fits, readout_anchors, rollouts, continuation, source_deployment = [], {}, {}, [], [], []
    timeline = output / "timeline.jsonl"
    def phase(name: str, **fields) -> None:
        with timeline.open("a") as stream:
            stream.write(json.dumps({"phase": name, "time_ns": time.time_ns(), **fields}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    def anchor_prefix(prefix: Path) -> None:
        durable.prefix(prefix)
    def cohort(split: str, source: str, body: str, seed: int) -> list[dict]:
        directory = output / "cohorts"; directory.mkdir(exist_ok=True)
        stem = directory / f"{split}-{source}-{body}-s{seed}"
        prefixes, rows, validations = {}, {}, {}
        for probes, label in ((False, "control"), (True, "diagnostic")):
            prefix = Path(str(stem) + "-" + label)
            print("RUN", prefix.name, flush=True)
            durable.require([host, config, output / (body + ".tokens.u32"), lives[source], lives["student"]])
            future.execute([str(host), "--trajectory", body, str(output / (body + ".tokens.u32")), str(prefix), str(steps), str(seed),
                            str(protocol["body_lr"]), str(config), str(lives[source]), str(int(probes)), str(lives["student"])], Path(str(prefix) + ".log"), env)
            rows[label] = durable.events(Path(str(prefix) + ".jsonl"))
            validations[label] = validate_cohort(rows[label], body, seed, source, protocol, steps, probes)
            prefixes[label] = prefix; anchor_prefix(prefix)
        parity = compare_hosts(rows["control"], rows["diagnostic"], prefixes["control"], prefixes["diagnostic"], steps, durable)
        aliased = validate_alias_files(rows["diagnostic"], prefixes["diagnostic"], durable)
        cohorts.append({"split": split, "source": source, "body": body, "seed": seed,
                        "prefix": stem.relative_to(output).as_posix(), **validations, "parity": parity, "aliased_lives": aliased})
        receipt_path = output / f"cohort-receipt-{len(cohorts):03d}.json"
        write_json(receipt_path, cohorts[-1]); durable.require([receipt_path])
        return gather_samples(rows["diagnostic"], body, seed, source, prefixes["diagnostic"], output)
    def collect(split: str, sources: list[str]) -> dict[str, list[dict]]:
        collected = {}
        expected = {(body, seed, checkpoint) for body in protocol["body_order"] for seed in seeds[split]
                    for checkpoint in protocol["checkpoints_before_updates"] if checkpoint <= steps}
        for source in sources:
            samples = []
            for body in protocol["body_order"]:
                for seed in seeds[split]: samples.extend(cohort(split, source, body, seed))
            assert len(samples) == len(expected) and {(s["body"], s["seed"], s["checkpoint"]) for s in samples} == expected
            table = output / f"{split}-{source}.samples.bin"
            future.write_samples(table, samples); anchor_prefix(table.with_suffix(""))
            collected[source] = samples
        return collected
    phase("acquisition_started", student_sha256=durable.identity(lives["student"])["sha256"])
    development = collect("development", protocol["acquisition"]["sources"])
    for label, student in protocol["students"].items():
        samples = development[student["source"]]
        table, trace = output / (label + ".samples.bin"), output / (label + ".fit.jsonl")
        future.write_samples(table, samples)
        anchor_prefix(table.with_suffix(""))
        durable.require([fitter, lives["student"], table, *(output / sample["source_policy"] for sample in samples)])
        future.execute([str(fitter), "fit-conditioned", str(lives["student"]), str(table), str(lives[label]), str(trace), str(epochs), student["life_id"]], output / (label + ".fit.log"), env)
        fit_rows = durable.events(trace)
        assert f32(fit_rows[0]["learning_rate"]) == f32(protocol["fitting"]["learning_rate"])
        fits[label] = {"source": student["source"], "initial_student_sha256": durable.identity(lives["student"])["sha256"],
                       "validation": lived.validate_fit_rows(samples, fit_rows, epochs),
                       "sample_table_sha256": durable.identity(table)["sha256"], "fit_trace_sha256": durable.identity(trace)["sha256"]}
        anchor_prefix(output / label)
    sealed = {label: {"file": path.name, **durable.identity(path)} for label, path in lives.items()}
    seal = {"protocol_sha256": manifest["protocol_sha256"], "lives": sealed, "fit_receipts": fits, "evaluation_generation_started": False}
    write_json(output / "sealed_lives.json", seal); durable.require([output / "sealed_lives.json"])
    phase("lives_sealed", sealed_lives_sha256=durable.identity(output / "sealed_lives.json")["sha256"])
    outputs = []
    for source, samples in development.items():
        outputs.extend(readouts(fitter, output / f"development-{source}.samples.bin", samples, lives, output, "development", source, protocol["evaluation"]["readouts"], env, readout_anchors, durable))
    phase("evaluation_generation_started", sealed_lives_sha256=durable.identity(output / "sealed_lives.json")["sha256"])
    evaluation = collect("evaluation", [protocol["evaluation"]["source"]])
    for source, samples in evaluation.items():
        outputs.extend(readouts(fitter, output / f"evaluation-{source}.samples.bin", samples, lives, output, "evaluation", source, protocol["evaluation"]["readouts"], env, readout_anchors, durable))
    phase("closed_loop_started")
    directory = output / "rollouts"; directory.mkdir()
    for body in protocol["body_order"]:
        for seed in seeds["evaluation"]:
            def deploy(arm: str, resume_step: int):
                prefix = directory / (f"{body}-s{seed}-{arm}" + ("-resume" if resume_step else ""))
                print("RUN", prefix.name, flush=True)
                durable.require([host, config, output / (body + ".tokens.u32")])
                if arm in lives: durable.require([lives[arm]])
                future.execute([str(host), "--rollout", body, str(output / (body + ".tokens.u32")), str(prefix), str(deploy_steps), str(seed),
                                str(protocol["body_lr"]), str(config), arm, str(lives[arm]) if arm in lives else "-", str(resume_step)], Path(str(prefix) + ".log"), env)
                rows = durable.events(Path(str(prefix) + ".jsonl"))
                summary = conditional.validate_rollout(rows, deploy_steps, arm in lives, resume_step)
                rollouts.append({"body": body, "seed": seed, "arm": arm, "resume_step": resume_step,
                                 "prefix": prefix.relative_to(output).as_posix(), "summary": summary})
                anchor_prefix(prefix)
                receipt_path = output / f"rollout-receipt-{len(rollouts):03d}.json"
                write_json(receipt_path, rollouts[-1]); durable.require([receipt_path])
                return prefix, rows
            runs = {arm: deploy(arm, 0) for arm in protocol["closed_loop"]["arms"]}
            initial = [durable.read_bytes(Path(str(prefix) + ".initial.bin")) for prefix, rows in runs.values()]
            assert all(data == initial[0] for data in initial), "deployment initial bodies differ"
            cohort_receipt = next(item for item in cohorts if (item["split"], item["source"], item["body"], item["seed"]) == ("evaluation", "student", body, seed))
            source_prefix = output / (cohort_receipt["prefix"] + "-control")
            deployed, rows = runs["student"]
            source_deployment.append({"body": body, "seed": seed, "source_prefix": source_prefix.relative_to(output).as_posix(),
                                      "deployment_prefix": deployed.relative_to(output).as_posix(),
                                      "validation": compare_prefix(durable.events(Path(str(source_prefix) + ".jsonl")), rows, steps)})
            resume_arm = protocol["closed_loop"]["policy_continuation"]["arm"]
            resume_step = min(protocol["closed_loop"]["policy_continuation"]["save_load_after_step"], deploy_steps // 2)
            resumed, resumed_rows = deploy(resume_arm, resume_step)
            prefix, rows = runs[resume_arm]
            continuation.append({"body": body, "seed": seed, "resume_step": resume_step,
                                 "prefix": prefix.relative_to(output).as_posix(), "resumed_prefix": resumed.relative_to(output).as_posix(),
                                 "validation": compare_resume(rows, resumed_rows, prefix, resumed, deploy_steps, durable)})
    for label, path in lives.items():
        assert durable.identity(path) == {key: sealed[label][key] for key in ("bytes", "sha256")}, "sealed life changed"
    for name, value in (("cohort_receipts.json", cohorts), ("rollout_receipts.json", rollouts)):
        write_json(output / name, value); durable.require([output / name])
    anchors = {"protocol": protocol, "host_steps": steps, "deployment_steps": deploy_steps, "epochs": epochs,
               "files": dict(files), "cohorts": cohorts, "fits": fits, "readouts": readout_anchors, "rollouts": rollouts,
               "continuation": continuation, "source_deployment": source_deployment, "lives": list(lives)}
    for name, digest in hashes.items(): assert v1.digest(ROOT / name) == digest, "source changed during experiment: " + name
    counts = {"acquisition_worlds": sum(len(samples) for samples in development.values()),
              "evaluation_worlds": sum(len(samples) for samples in evaluation.values()),
              "branch_updates": sum(item["diagnostic"]["branch_updates"] for item in cohorts),
              "paired_host_processes": len(cohorts) * 2, "host_updates": len(cohorts) * 2 * steps,
              "primary_deployment_processes": sum(item["resume_step"] == 0 for item in rollouts),
              "resume_processes": len(continuation), "deployment_updates": len(rollouts) * deploy_steps,
              "fit_updates": sum(item["validation"]["fit_steps"] for item in fits.values()),
              "development_readouts": sum(row["split"] == "development" for row in outputs),
              "evaluation_readouts": sum(row["split"] == "evaluation" for row in outputs), "readouts": len(outputs),
              "source_continuation_transitions": sum(item["diagnostic"]["selected_policy_transitions_compared"] for item in cohorts)}
    counts["executed_branches"] = counts["branch_updates"] // protocol["target_horizon_total_updates"]
    counts["all_body_updates"] = counts["branch_updates"] + counts["host_updates"] + counts["deployment_updates"]
    if not args.smoke: assert counts == protocol["fixed_counts"], "executed experiment counts differ from preregistration"
    phase("evaluation_complete", **counts)
    durable.require([timeline])
    anchors["files"] = dict(files)
    write_json(output / "terminal_anchors.json", anchors)
    durable.require([output / "terminal_anchors.json"])
    terminal = verify_terminal_artifacts(output, anchors, durable)
    result = {"manifest": manifest, "fit_receipts": fits, "sealed_lives": seal, "cohorts": cohorts,
              "summary": summarize(outputs), "readouts": outputs, "rollouts": rollouts, "policy_continuation": continuation,
              "source_deployment_parity": source_deployment, "terminal_verification": terminal, "executed_counts": counts,
              "source_hashes_rechecked": True, "sealed_lives_unchanged": True,
              "durability": {"index": durable.index.relative_to(output).as_posix(), "max_restore_attempts_per_file": 1,
                             "integrity_failures": durable.failures},
              "artifacts": dict(durable.files)}
    write_json(output / "results.json", result)
    durable.require([output / "results.json"])
    print("TRAJECTORY_OK", json.dumps(counts), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        return run(args)
    except Exception as error:
        if not existed and args.output.is_dir():
            write_json(args.output / "failure.json", {"status": "FAIL", "exception": type(error).__name__,
                                                      "message": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    sys.exit(main())
