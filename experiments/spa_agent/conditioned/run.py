#!/usr/bin/env python3
"""Fixed SPA target-conditioning experiment; all acquisition runs in native C."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"
PROTOCOL_SHA = "e5bb801a78cf68403c45474858fb7aff96e5aa4337003b6c06dc72253257225a"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REP = module("spa_conditioned_replicates", HERE.parent / "replicates/run.py")
POLICY = module("spa_conditioned_policy", HERE / "policy.py")
FUTURE, V1, SCENARIOS = REP.FUTURE, REP.V1, REP.SCENARIOS
ARMS = POLICY.ARMS
# Reuse the unchanged measurement equations with this protocol's arm labels.
REP.ARMS, REP.LABELS = ARMS, (*ARMS, "keep", "left", "right")
require, identity = REP.require, REP.identity


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_bytes(path, raw, *, replace=False):
    """Publish a complete, read-back-checked record, preserving earlier files."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == raw, "write readback differs: " + path.name)
    if replace:
        os.replace(pending, path)
    else:
        os.link(pending, path)
        pending.unlink()


def write_json(path, value, *, replace=False):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    write_bytes(path, raw, replace=replace)


def run_command(arguments, cwd, output, label):
    arguments = list(map(str, arguments))
    started = time.monotonic()
    process = subprocess.run(arguments, cwd=cwd, capture_output=True)

    def portable(value):
        return (str(value).replace(str(cwd), "<source_snapshot>")
                .replace(str(output), "<output>").replace(str(sys.executable), "python3"))

    record = {"command": [portable(a) for a in arguments], "cwd": "<source_snapshot>",
        "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
        "path_normalization": "Only execution-root/interpreter path spellings are replaced."}
    for name, raw in (("stdout", process.stdout), ("stderr", process.stderr)):
        text = portable(raw.decode("utf-8"))
        path = output / f"{label}.{name}.txt"
        write_bytes(path, text.encode())
        record[name] = {"file": path.name, **identity(path), "text": text,
                        "original_sha256": sha(raw)}
    write_json(output / (label + ".command.json"), record)
    require(process.returncode == 0, f"{label} exited {process.returncode}: {record['stderr']['text']}")
    return record


def archive_member(path, expected, seed, stream):
    require(identity(path)["sha256"] == expected, "parent archive identity changed")
    pieces = []
    with gzip.open(path, "rb") as source:
        for line in source:
            value = json.loads(line, object_pairs_hook=SCENARIOS.unique_object)
            require(set(value) == {"seed", "stream", "raw"}, "archive envelope changed")
            if (value["seed"], value["stream"]) == (seed, stream):
                pieces.append(value["raw"].encode())
    require(pieces, "parent archive member missing")
    return b"".join(pieces)


def source_files():
    names = set(REP.SOURCE_FILES) | {
        "experiments/spa_agent/conditioned/run.py",
        "experiments/spa_agent/conditioned/policy.py",
        "experiments/spa_agent/conditioned/protocol.json",
        "tests/test_spa_agent_conditioned.c",
        "tests/test_spa_conditioned_policy.py",
        "experiments/spa_agent/conditioned/audit.py",
        "experiments/spa_agent/conditioned/audit.json",
        "tests/test_spa_conditioned_audit.py",
        "tests/test_spa_conditioned_runner.py",
        "experiments/spa_agent/conditioned/runner_preflight.json",
    }
    # Include new upstream quoted headers without copying another notorch.
    pending = list(names)
    while pending:
        name = pending.pop()
        if Path(name).suffix not in (".c", ".h"):
            continue
        for header in re.findall(r'^\s*#\s*include\s+"([^"]+)"',
                                 (ROOT / name).read_text(), re.M):
            path = (ROOT / name).parent / header
            if path.is_file():
                rel = str(path.resolve().relative_to(ROOT))
                if rel not in names:
                    names.add(rel)
                    pending.append(rel)
    return sorted(names)


def pack_records(output, paths, closed=None):
    """Archive exact raw records; model weights and life binaries stay external."""
    paths = list(paths)
    for name in paths:
        require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
                "archive member must be a portable relative path")
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts and
                name not in (".", "..") and member.as_posix() == name,
                "archive member must be a portable relative path")
    require(len(paths) == len(set(paths)), "duplicate archive member")
    destination = output / "raw_records.tar.gz"
    pending = output / "raw_records.tar.gz.pending"
    manifest = {name: (closed[name] if closed is not None else identity(output / name))
                for name in sorted(paths)}
    with pending.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in manifest:
                    path = output / name
                    require(identity(path) == manifest[name], "record changed before archive")
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = manifest[name]["bytes"], 0o644, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())
    seen = set()
    with tarfile.open(pending, "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in manifest and member.name not in seen,
                    "archive member coverage changed")
            raw = archive.extractfile(member).read()
            require({"bytes": len(raw), "sha256": sha(raw)} == manifest[member.name],
                    "archive readback mismatch: " + member.name)
            seen.add(member.name)
    require(seen == set(manifest), "incomplete archive")
    os.link(pending, destination)
    pending.unlink()
    return {"file": destination.name, **identity(destination), "members": manifest}


def record_inventory(output, binaries, closed):
    artifacts, records = {}, []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if not path.is_file() or relative.parts[0] in ("source_snapshot", "inputs"):
            continue
        name = str(relative)
        if name in binaries:
            continue
        require(name in closed, "unregistered closed artifact: " + name)
        require(identity(path) == closed[name], "closed artifact differs at final inventory: " + name)
        artifacts[name] = closed[name]
        if path.suffix not in (".life", ".bin"):
            records.append(name)
    return artifacts, records


def run(output, input_directory):
    protocol, protocol_sha = POLICY.validate_protocol(PROTOCOL)
    require(protocol_sha == PROTOCOL_SHA, "runner protocol differs")
    for entry in (*protocol["parents"].values(), *protocol["diagnosis"].values()):
        require(identity(ROOT / entry["path"])["sha256"] == entry["sha256"],
                "registered parent/diagnosis changed: " + entry["path"])
    parent = protocol["parents"]["raw_traces.jsonl.gz"]
    raw_training = archive_member(ROOT / parent["path"], parent["sha256"], 0, "training_dataset")
    expected = protocol["training"]["parent_dataset"]
    require(len(raw_training) == expected["bytes"] and sha(raw_training) == expected["sha256"],
            "retained training dataset changed")
    training_value = json.loads(raw_training)
    training_value["protocol_sha256"] = PROTOCOL_SHA
    output.mkdir()
    write_json(output / "training.json", training_value)
    training = POLICY.read_dataset(output / "training.json", PROTOCOL_SHA, training=True)
    training_id = identity(output / "training.json")
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("simple.weights", "dracula.txt"):
        shutil.copyfile(input_directory / name, inputs / name)
    input_ids, _ = V1.prepare(inputs, None, json.loads(V1.PROTOCOL.read_bytes()))
    input_files = {p.name: identity(p) for p in inputs.iterdir() if p.is_file()}
    vocabulary_fnv = REP.fnv((inputs / "vocabulary.u32").read_bytes())
    snapshot = output / "source_snapshot"
    sources = {name: (ROOT / name).read_bytes() for name in source_files()}
    source_ids = {name: sha(raw) for name, raw in sources.items()}
    for name, raw in sources.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(path, raw)
    binaries, commands, stages, closed = {}, [], [], {}

    def remember(path, raw=None):
        name = str(path.relative_to(output))
        expected = identity(path) if raw is None else {"bytes": len(raw), "sha256": sha(raw)}
        require(identity(path) == expected, "closed artifact differs: " + name)
        require(name not in closed or closed[name] == expected, "closed artifact replaced: " + name)
        closed[name] = expected

    remember(output / "training.json")

    def stable():
        for name, digest in source_ids.items():
            require(identity(ROOT / name)["sha256"] == digest and
                    identity(snapshot / name)["sha256"] == digest, "measured source changed: " + name)
        for name, expected in binaries.items():
            require(identity(output / name) == expected, "measured binary changed")
        for name, expected in input_files.items():
            require(identity(inputs / name) == expected, "measured body input changed")
        require(identity(output / "training.json") == training_id, "retained training changed")
        for name, expected in list(closed.items()):
            require(identity(output / name) == expected, "closed artifact changed later: " + name)

    def phase(name, **details):
        stages.append({"order": len(stages) + 1, "phase": name, **details})
        write_json(output / "stages.json", stages, replace=(output / "stages.json").exists())
        print(json.dumps(stages[-1], sort_keys=True), flush=True)

    def command(arguments, label):
        record = run_command(arguments, snapshot, output, label)
        commands.append(record)
        for suffix in ("stdout.txt", "stderr.txt", "command.json"):
            remember(output / (label + "." + suffix))
        return record

    common = ["cc", "-std=c11", "-O2", "-DUSE_SIMD", "-march=native", "-pthread", "-I."]
    host, ordinary, library = (output / name for name in
                               ("spa_agent_replicates", "spa_agent_demo", "libnotorch.so"))
    for binary, source in ((host, "examples/spa_agent_replicates.c"),
                           (ordinary, "examples/spa_agent_demo.c")):
        command(common + [source, "spa_agent.c", "notorch.c", "-lm", "-o", binary], "build-" + binary.name)
    command(common + ["-fPIC", "-shared", "spa_binding.c", "spa_agent.c", "notorch.c",
                      "-lm", "-o", library], "build-shared")
    binaries.update({p.name: identity(p) for p in (host, ordinary, library)})
    native = POLICY.SPA.Native(library)
    metadata = {"base_commit": V1.git("rev-parse", "HEAD"), "protocol_sha256": PROTOCOL_SHA,
        "protocol": protocol, "source_files_sha256": source_ids, "binaries": binaries,
        "inputs": input_files, "input_preparation": input_ids, "machine": V1.machine(),
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "parent_training": expected, "training_dataset": training_id}
    write_json(output / "metadata.json", metadata)
    remember(output / "metadata.json")
    stable()
    helper = snapshot / "experiments/spa_agent/conditioned/policy.py"
    policy_args = ["--protocol", snapshot / "experiments/spa_agent/conditioned/protocol.json",
                   "--library", library]
    phase("fitting_started", copies=2, new_body_generation=0)
    fits = [output / ("fit-" + str(i)) for i in (1, 2)]
    seals = []
    for i, directory in enumerate(fits, 1):
        command([sys.executable, helper, "train", "--dataset", output / "training.json",
                 *policy_args, "--output", directory], "fit-" + str(i))
        seals.append(POLICY.verify_seal(directory, native, PROTOCOL_SHA)[0])
        for path in directory.iterdir():
            if path.is_file():
                remember(path)
        stable()
    repeated_files = ["seal.json", "fit.jsonl.gz", *(arm + ".life" for arm in ARMS)]
    for name in repeated_files:
        require((fits[0] / name).read_bytes() == (fits[1] / name).read_bytes(),
                "deterministic fitting copies differ: " + name)
    barrier = {"status": "ALL_EIGHT_LIVES_SEALED", "protocol_sha256": PROTOCOL_SHA,
        "source_sha256": source_ids, "binaries": binaries, "inputs": input_files,
        "training": training_id, "copies": {p.name: {name: identity(p / name)
        for name in repeated_files} for p in fits}}
    write_json(output / "all_eight_lives_sealed.json", barrier)
    remember(output / "all_eight_lives_sealed.json")
    barrier_id = identity(output / "all_eight_lives_sealed.json")

    def sealed():
        stable()
        require(identity(output / "all_eight_lives_sealed.json") == barrier_id,
                "evaluation barrier changed")
        for directory in fits:
            for name, expected in barrier["copies"][directory.name].items():
                require(identity(directory / name) == expected, "sealed fitting file changed")

    sealed()
    phase("all_eight_lives_sealed", barrier=barrier_id)
    scenario_protocol = json.loads((snapshot / "experiments/spa_agent/scenarios/protocol.json").read_bytes())
    evaluation_streams, host_gates, repeat_gates = {}, [], []

    def ordinary_job(seed):
        local_streams, local_commands, checks = {}, [], []
        scenario = output / f"scenarios-s{seed}.jsonl"
        for mode in ("off", "on"):
            prefix = output / f"{mode}-s{seed}"
            args = [ordinary, *(inputs / n for n in ("simple.weights", "corpus.tokens.u32", "vocabulary.u32")),
                    prefix, str(seed)]
            if mode == "on":
                args += ["--scenarios", scenario]
            label = f"{mode}-s{seed}"
            local_commands.append(run_command(args, snapshot, output, label))
            for suffix in ("stdout.txt", "stderr.txt", "command.json"):
                remember(output / (label + "." + suffix))
            raw, checked = FUTURE.complete_trace(Path(str(prefix) + ".jsonl"), seed)
            remember(Path(str(prefix) + ".jsonl"), raw)
            local_streams[seed, "ordinary_" + mode] = raw
            checks.append({"seed": seed, "mode": mode, **checked})
        require(local_streams[seed, "ordinary_off"] == local_streams[seed, "ordinary_on"],
                "diagnostics changed ordinary generation")
        for arm in json.loads(V1.PROTOCOL.read_bytes())["arms"]:
            require((output / f"off-s{seed}.{arm}.life.bin").read_bytes() ==
                    (output / f"on-s{seed}.{arm}.life.bin").read_bytes(), "diagnostics changed saved life")
            for mode in ("off", "on"):
                remember(output / f"{mode}-s{seed}.{arm}.life.bin")
        raw, checked = FUTURE.complete_scenario(scenario, local_streams[seed, "ordinary_off"], seed, scenario_protocol)
        local_streams[seed, "scenarios"] = raw
        remember(scenario, raw)
        checks.append({"seed": seed, "mode": "scenarios", **checked})
        return local_streams, local_commands, checks

    sealed()
    phase("new_source_generation_started", seeds=protocol["evaluation"]["seeds"], barrier=barrier_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for streams, records, checks in pool.map(ordinary_job, protocol["evaluation"]["seeds"]):
            evaluation_streams.update(streams)
            commands.extend(records)
            host_gates.extend(checks)
    sealed()
    source_archive = output / "evaluation_sources.raw.jsonl.gz"
    FUTURE.write_archive(source_archive, evaluation_streams)
    remember(source_archive)
    sources, references = REP.source_samples(evaluation_streams, protocol["evaluation"]["seeds"],
                                            identity(source_archive)["sha256"], scenario_protocol)
    require(all(s["body_hash"] == training[0]["body_hash"] for s in sources), "body differs across cohorts")

    def repeat_job(seed):
        chunk = [s for s in sources if s["seed"] == seed]
        data, trace = output / f"evaluation-sources-s{seed}.txt", output / f"evaluation-repeats-s{seed}.jsonl"
        REP.write_sources(data, chunk, vocabulary_fnv)
        remember(data)
        label = f"evaluation-repeats-s{seed}"
        record = run_command([host, inputs / "simple.weights", inputs / "vocabulary.u32", data, trace],
                              snapshot, output, label)
        for suffix in ("stdout.txt", "stderr.txt", "command.json"):
            remember(output / (label + "." + suffix))
        raw = REP.complete_bytes(trace)
        remember(trace, raw)
        projected, checked = REP.validate_replicates(raw, chunk, references,
            dataset_fnv=REP.fnv(data.read_bytes()), vocabulary_fnv=vocabulary_fnv)
        return projected, record, {"seed": seed, **checked}

    phase("new_repeated_generation_started", sources=len(sources), repeats=8)
    evaluation = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for projected, record, checked in pool.map(repeat_job, protocol["evaluation"]["seeds"]):
            evaluation.extend(projected)
            commands.append(record)
            repeat_gates.append(checked)
    evaluation.sort(key=lambda row: row["ordinal"])
    require(len(evaluation) == 48, "incomplete evaluation cohort")
    write_json(output / "evaluation.json", {"schema": 1, "protocol_sha256": PROTOCOL_SHA,
                                            "replicates": 8, "samples": evaluation})
    remember(output / "evaluation.json")
    POLICY.read_dataset(output / "evaluation.json", PROTOCOL_SHA, training=False)
    sealed()
    phase("new_generation_complete", measurements=sum(x["measurements"] for x in repeat_gates))
    summaries = {}
    for split, samples in (("training", training), ("evaluation", evaluation)):
        readouts = []
        for i, directory in enumerate(fits, 1):
            target = output / f"{split}.readout-{i}.jsonl"
            args = [sys.executable, helper, "score", "--dataset", output / (split + ".json"),
                    *policy_args, "--policies", directory, "--output", target]
            if split == "training":
                args.append("--training")
            command(args, f"{split}-readout-{i}")
            raw = REP.complete_bytes(target)
            remember(target, raw)
            rows, _ = REP.parse_raw(raw)
            choices = REP.validate_readouts(rows, samples,
                {arm: seals[i - 1]["lives"][arm]["life_hash"] for arm in ARMS})
            readouts.append(raw)
        require(readouts[0] == readouts[1], "sealed copies select different actions")
        summaries[split] = REP.summarize(samples, choices, split)
        write_json(output / (split + ".summary.json"), summaries[split])
        remember(output / (split + ".summary.json"))
    sealed()
    phase("all_readouts_and_source_checks_complete")
    write_json(output / "commands.json", commands)
    remember(output / "commands.json")
    remember(output / "stages.json")
    artifacts, records = record_inventory(output, binaries, closed)
    archive = pack_records(output, records, closed)
    for name, expected in artifacts.items():
        require(identity(output / name) == expected, "completed artifact changed: " + name)
    sealed()
    result = {"schema": 1, "metadata": metadata, "barrier": barrier, "stages": stages,
        "commands": commands, "fit_repeat": {"copies": 2, "byte_identical_files": repeated_files,
        "seals": seals}, "host_gates": host_gates, "repeat_gates": repeat_gates,
        "training": summaries["training"], "evaluation": summaries["evaluation"],
        "artifacts": artifacts, "archive": archive}
    # Normalize only command path spellings in the public JSON view.
    text = json.dumps(result, allow_nan=False).replace(str(snapshot), "<source_snapshot>")
    text = text.replace(str(output), "<output>").replace(str(sys.executable), "python3")
    write_json(output / "receipts.json", json.loads(text))
    print(json.dumps({"complete": True, "archive": identity(output / archive["file"]),
        "evaluation_h4": [r for r in summaries["evaluation"]["summaries"]
                          if r["group"] == "combined" and r["horizon"] == 4]}, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "experiment output already exists")
    run(args.output.resolve(), args.inputs.resolve())


if __name__ == "__main__":
    main()
