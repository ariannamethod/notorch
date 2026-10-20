#!/usr/bin/env python3
"""Fixed living acquisition: native C arithmetic, authenticated consequences."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(result)
    return result


previous = module("chuck_trajectories", HERE.parent / "trajectories/run.py")
gates = module("chuck_living_gates", ROOT / "tests/test_chuck_living.py")
v1, scenarios = previous.v1, previous.scenarios
identity, write_json = previous.identity, previous.write_json
DurableArtifacts = previous.DurableArtifacts
SOURCE_FILES = (
    "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h", "chuck_architect_impl.h",
    "examples/chuck_architect_train.c", "examples/chuck_architect_scenarios.h",
    "examples/chuck_architect_rollout.h", "examples/chuck_architect_lived.h",
    "examples/chuck_architect_living.h", "examples/chuck-loss-architect.json",
    "tests/test_chuck_living.py",
)


def build_snapshot(output, durable, source_commit, dirty):
    """Compile exactly the source bytes retained with this run's receipts."""
    source_names = list(SOURCE_FILES)
    source_names += [str(p.relative_to(ROOT)) for p in sorted(HERE.parent.rglob("run.py"))]
    source_names.append(str((HERE / "protocol.json").relative_to(ROOT)))
    snapshot = output / "source"
    source_hashes = {}
    for name in source_names:
        data = (ROOT / name).read_bytes()
        if not dirty:
            committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
            if data != committed:
                raise ValueError("source changed after clean-tree check: " + name)
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        source_hashes[name] = durable.identity(path)["sha256"]
    # Both C translation units, quoted headers, -I and cwd use the snapshot.
    command = scenarios.build(output / "host_runner", root=snapshot)
    return source_hashes, command


def authenticate_inputs(directory, protocol):
    assert v1.digest(directory / "results.json") == protocol["previous_results_sha256"]
    old = scenarios.strict_json((directory / "results.json").read_text())
    selected = {}
    names = ["architect-config.json", protocol["input_life"]["file"]]
    names += [body + suffix for body in protocol["body_order"]
              for suffix in (".corpus", ".tokens.u32", ".vocab.json")]
    for name in names:
        assert identity(directory / name) == old["artifacts"][name], name
        selected[name] = old["artifacts"][name]
    life = protocol["input_life"]
    assert selected[life["file"]] == {k: life[k] for k in ("bytes", "sha256")}
    return old, selected


def selected_rows(rows, full=False):
    ignored = {"living_run", "living_summary", "policy_resume"}
    return [r for r in rows if r["type"] not in ignored] if full else [
        r for r in rows if r["type"] in ("rollout_step", "evaluation")]


def compare(left, right, output, durable, full=False):
    a, b = output / left["prefix"], output / right["prefix"]
    la, lb = durable.events(Path(str(a) + ".jsonl")), durable.events(Path(str(b) + ".jsonl"))
    assert selected_rows(la, full) == selected_rows(lb, full), "continuation events differ"
    for key in ("state_hash", "policy_hash", "steps", "action_counts"):
        assert la[-1][key] == lb[-1][key], "continuation terminal differs: " + key
    suffixes = [".policy.initial.bin", ".policy.final.bin"]
    if full:
        suffixes += [f".acquire-{r['checkpoint']}.{phase}.policy.bin"
                     for r in la if r["type"] == "acquisition" for phase in ("pre", "post")]
    for suffix in suffixes:
        assert durable.read_bytes(Path(str(a) + suffix)) == durable.read_bytes(Path(str(b) + suffix))
    return {"status": "PASS", "left": left["prefix"], "right": right["prefix"],
            "compared_events": len(selected_rows(la, full)), "policy_files": len(suffixes),
            "steps": la[-1]["steps"], "final_state_hash": la[-1]["state_hash"]}


def summarize(rows):
    steps = [r for r in rows if r["type"] == "rollout_step"]
    acquisitions = [r for r in rows if r["type"] == "acquisition_result"]
    header, terminal = rows[0], rows[-1]
    floor = next((r["step"] for r in steps if r["post_chuck"]["dampen"] <= .30000002), None)
    return {"body": header["body"], "seed": header["seed"], "arm": header["arm"],
            "parameters": header["parameters"], "fixture": header["fixture"],
            "evaluations": [r for r in rows if r["type"] == "evaluation"],
            "first_dampen_floor": floor,
            "fit_action_changes": [{"step": r["checkpoint"], "before": r["action_before"],
                                    "after": r["action_after"], "source_hash": r["source_hash"]}
                                   for r in acquisitions if r["action_before"] != r["action_after"]],
            "acquisitions": acquisitions, "terminal": terminal}


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    output, inputs = args.output.resolve(), args.inputs.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        raise ValueError("choose a fresh output directory outside Git")
    source_commit = v1.command(["git", "rev-parse", "HEAD"])
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = scenarios.strict_json(protocol_bytes.decode("utf-8"))
    old, selected = authenticate_inputs(inputs, protocol)
    status = v1.command(["git", "status", "--porcelain"])
    if status and not (args.smoke and args.allow_dirty):
        raise ValueError("final run requires clean committed source")
    if args.allow_dirty and not args.smoke:
        raise ValueError("dirty execution is restricted to the fixed development smoke")
    steps = protocol["smoke"]["steps"] if args.smoke else protocol["steps"]
    seeds = [protocol["smoke"]["seed"]] if args.smoke else protocol["seeds"]
    output.mkdir(parents=True)
    args.output_created = True
    durable = DurableArtifacts(output)
    for name in selected:
        shutil.copyfile(inputs / name, output / name)
        assert durable.identity(output / name) == selected[name]
    (output / "protocol.json").write_bytes(protocol_bytes)
    source_hashes, build_command = build_snapshot(output, durable, source_commit, bool(status))
    protocol_name = str((HERE / "protocol.json").relative_to(ROOT))
    if source_hashes[protocol_name] != v1.digest(output / "protocol.json"):
        raise ValueError("protocol changed after it was read")
    binary = output / "host_runner"
    durable.require([binary, output / "protocol.json"])
    manifest = {"source_commit": source_commit, "source_status": status,
                "source_sha256": source_hashes, "protocol_sha256": v1.digest(output / "protocol.json"),
                "datasets": old["manifest"]["datasets"], "machine": v1.machine(),
                "compiler": v1.command(["cc", "--version"]).splitlines()[0], "build_command": build_command,
                "binary": durable.identity(binary), "smoke": args.smoke, "seeds": seeds, "steps": steps,
                "threads": protocol["threads"], "selected_inputs": selected,
                "previous_results_sha256": protocol["previous_results_sha256"]}
    write_json(output / "manifest.json", manifest); durable.require([output / "manifest.json"])
    env = dict(os.environ, NT_SIMD_THREADS=str(protocol["threads"]), OMP_NUM_THREADS=str(protocol["threads"]),
               OPENBLAS_NUM_THREADS=str(protocol["threads"]))
    (output / "runs").mkdir()
    records, primary, probe_pairs, resume_pairs = [], {}, [], []
    timeline = output / "timeline.jsonl"

    def execute(body, seed, arm, probes, resume=0, suffix=""):
        prefix = output / "runs" / f"{body}-s{seed}-{arm}{suffix}"
        command = [str(binary), "--living", body, str(output / (body + ".tokens.u32")), str(prefix),
                   str(steps), str(seed), str(protocol["body_lr"]), str(output / "architect-config.json"),
                   arm, str(output / protocol["input_life"]["file"]), str(resume), str(int(probes))]
        durable.require([binary, output / (body + ".tokens.u32"), output / "architect-config.json",
                         output / protocol["input_life"]["file"]])
        with timeline.open("a") as stream:
            stream.write(json.dumps({"time_ns": time.time_ns(), "command": command}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        print(f"START {prefix.name}", flush=True)
        code = scenarios.execute(command, Path(str(prefix) + ".log"), env)
        rows = durable.events(Path(str(prefix) + ".jsonl"))
        durable.prefix(prefix)
        if code:
            raise RuntimeError(f"native run {prefix.name} stopped with exit {code}; raw failure retained")
        validation = gates.validate(rows, arm, steps, probes)
        assert rows[0]["parameters"] == protocol["bodies"][body] and rows[0]["fixture"] is False
        assert rows[0]["seed"] == seed and rows[0]["resume_step"] == resume
        record = {"prefix": prefix.relative_to(output).as_posix(), "body": body, "seed": seed,
                  "arm": arm, "probes": probes, "resume_step": resume, "validation": validation,
                  "trace": durable.identity(Path(str(prefix) + ".jsonl")), **summarize(rows)}
        records.append(record)
        print(f"PASS {prefix.name} final={record['evaluations'][-1]['heldout_loss']:.8f}", flush=True)
        return record

    for body in protocol["body_order"]:
        for seed in seeds:
            for arm in protocol["arms"]:
                primary[(body, seed, arm)] = execute(body, seed, arm, arm != "canonical")
            off = execute(body, seed, "frozen", False, suffix="-probe-off")
            probe_pairs.append(compare(primary[(body, seed, "frozen")], off, output, durable))
            if seed == seeds[0]:
                resumed = execute(body, seed, "living", True, steps // 2, "-resume")
                resume_pairs.append(compare(primary[(body, seed, "living")], resumed, output, durable, True))

    # Re-read closed evidence from its original anchor before publishing results.
    for record in records:
        rows = durable.events(output / (record["prefix"] + ".jsonl"))
        assert gates.validate(rows, record["arm"], steps, record["probes"]) == record["validation"]
        assert summarize(rows) == {key: record[key] for key in summarize(rows)}
    for name in list(durable.files):
        durable.read_bytes(output / name)
    counts = {"total_processes": len(records), "host_updates": sum(r["terminal"]["steps"] for r in records),
              "acquisition_checkpoints": sum(r["terminal"]["acquisitions"] for r in records),
              "branch_updates": sum(r["terminal"]["branch_updates"] for r in records),
              "fit_updates": sum(r["terminal"]["fit_updates"] for r in records),
              "frozen_source_continuation_checks": sum(r["terminal"]["continuation_checks"] for r in records)}
    counts["all_body_updates"] = counts["host_updates"] + counts["branch_updates"]
    counts["measured_branches"] = counts["branch_updates"] // protocol["acquisition"]["horizon"]
    if not args.smoke:
        for key, value in counts.items():
            assert value == protocol["fixed_counts"][key], (key, value, protocol["fixed_counts"][key])
    paired, curves, actions = [], [], []
    for body in protocol["body_order"]:
        for seed in seeds:
            finals = {arm: primary[(body, seed, arm)]["evaluations"][-1]["heldout_loss"] for arm in protocol["arms"]}
            paired.append({"body": body, "seed": seed, **finals,
                           "living_minus_frozen": finals["living"] - finals["frozen"],
                           "living_minus_rotated": finals["living"] - finals["rotated"],
                           "living_minus_canonical": finals["living"] - finals["canonical"]})
            for arm in protocol["arms"]:
                r = primary[(body, seed, arm)]
                for e in r["evaluations"]:
                    curves.append({"body": body, "seed": seed, "arm": arm, "step": e["step"], "heldout_loss": e["heldout_loss"]})
                t = r["terminal"]
                actions.append({"body": body, "seed": seed, "arm": arm,
                                **dict(zip(("legacy", "hold", "brake", "push"), t["action_counts"])),
                                "brake_changed_dampen": t["brake_changed_dampen"], "brake_at_bound": t["brake_at_bound"],
                                "first_floor": r["first_dampen_floor"], "fit_action_changes": len(r["fit_action_changes"])})
    csv_file(output / "paired_final.csv", paired)
    csv_file(output / "heldout_curves.csv", curves)
    csv_file(output / "actions.csv", actions)
    durable.require([timeline, *(output / name for name in ("paired_final.csv", "heldout_curves.csv", "actions.csv"))])
    result = {"status": "PASS", "manifest": manifest, "counts": counts, "runs": records,
              "paired_final": paired, "probe_parity": probe_pairs, "resume_parity": resume_pairs,
              "integrity_failures": durable.failures, "artifacts": dict(durable.files)}
    write_json(output / "results.json", result); durable.require([output / "results.json"])
    print(json.dumps({"status": "PASS", "counts": counts, "results": durable.identity(output / "results.json")}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        if getattr(args, "output_created", False):
            write_json(args.output / "failure.json", {"status": "FAIL", "exception": type(error).__name__,
                       "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    sys.exit(main())
