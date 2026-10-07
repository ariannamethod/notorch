#!/usr/bin/env python3
"""Run the frozen SimpleLLM sentence-agent experiment through upstream C.

    python experiments/spa_agent/run.py --output /tmp/spa-receipts \
        --reference /path/to/notorch-simple-llm

The public checkpoint/corpus are pinned and hashed. Numerical work stays in C.
The output holds raw traces, complete saved lives, source identities and decoded
text. The reviewable receipts.json also includes every before/after metric.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
SOURCE_FILES = (
    "notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect.h",
    "chuck_architect_impl.h", "spa_agent.h", "spa_agent.c",
    "examples/spa_agent_demo.c", "experiments/spa_agent/run.py",
    "experiments/spa_agent/protocol.json",
)
AXES = ("local_connectedness", "global_connectedness", "coherence", "novelty",
        "repetition", "collapse", "continuity")
WEIGHTS = (0.15, 0.15, 0.20, 0.20, -0.15, -0.10, 0.05)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare(output: Path, reference: Path | None, protocol: dict) -> tuple[dict, list[str]]:
    body = protocol["body"]
    identities = {}
    if reference:
        ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
        if ref != body["commit"]:
            raise ValueError(f"reference commit: expected {body['commit']}, got {ref}")
    for name, source, expected in (
        ("simple.weights", body["checkpoint"], body["checkpoint_sha256"]),
        ("dracula.txt", body["corpus"], body["corpus_sha256"]),
    ):
        path = output / name
        if not path.exists():
            if reference:
                raw = (reference / source).read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/{body['repository']}/{body['commit']}/{source}"
                with urllib.request.urlopen(url, timeout=30) as response:
                    raw = response.read()
            path.write_bytes(raw)
        if digest(path) != expected:
            raise ValueError(f"{source}: checksum differs from the frozen protocol")
        identities[name] = {"sha256": expected, "bytes": path.stat().st_size}
    text = (output / "dracula.txt").read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    vocabulary = sorted(set(text))
    if len(vocabulary) != 94:
        raise ValueError("SimpleLLM requires exactly 94 sorted Unicode codepoints")
    mapping = {character: i for i, character in enumerate(vocabulary)}
    token_path = output / "corpus.tokens.u32"
    token_path.write_bytes(struct.pack(f"<{len(text)}I", *(mapping[c] for c in text)))
    vocab_path = output / "vocabulary.u32"
    vocab_path.write_bytes(struct.pack("<94I", *(ord(c) for c in vocabulary)))
    write_json(output / "vocabulary.json", vocabulary)
    identities["corpus.tokens.u32"] = {"sha256": digest(token_path), "tokens": len(text)}
    identities["vocabulary.u32"] = {"sha256": digest(vocab_path), "characters": vocabulary}
    return identities, vocabulary


def machine() -> dict:
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count()}
    try:
        result["cpu"] = next(line.split(":", 1)[1].strip() for line in
                             Path("/proc/cpuinfo").read_text().splitlines()
                             if line.startswith("model name"))
        result["memory"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                            for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))}
    except (OSError, StopIteration):
        pass
    thread_env = os.environ.get("NT_SIMD_THREADS")
    try:
        configured = int(thread_env) if thread_env else 0
    except ValueError:
        configured = 0
    hardware_threads = os.sysconf("SC_NPROCESSORS_ONLN") if hasattr(os, "sysconf") else (os.cpu_count() or 1)
    result["simd_threads"] = {"environment": thread_env,
                              "pool_limit": configured if 1 <= configured <= 16 else max(1, min(16, hardware_threads)),
                              "small_matrix_path": "The upstream GEMM chooses its serial fast path below the source-defined size threshold."}
    return result


def decoded_sentence(value: dict, vocabulary: list[str]) -> dict:
    return {"length": value["length"], "terminated": value["terminated"],
            "text": "".join(vocabulary[t] for t in value["tokens"])}


def summarize(events: list[dict], vocabulary: list[str], protocol: dict) -> dict:
    decisions = [event for event in events if event["type"] == "decision"]
    summaries = [event for event in events if event["type"] == "summary"]
    expected_count = protocol["episodes"] * protocol["decisions_per_episode"] * len(protocol["arms"])
    if len(decisions) != expected_count or len(summaries) != 1:
        raise AssertionError("incomplete C experiment trace")
    summary = summaries[0]
    if not summary["body_unchanged"]:
        raise AssertionError("body parameters changed")
    result = {"seed": summary["seed"], "forwards": summary["forwards"],
              "body_unchanged": True, "arms": [], "base_chains": [], "decisions": []}
    for event in events:
        if event["type"] == "base":
            result["base_chains"].append({"episode": event["episode"],
                                          "prompt_offsets": event["prompt_offsets"],
                                          "chain": [decoded_sentence(s, vocabulary) for s in event["chain"]]})
    base_sentences = [sentence for base in result["base_chains"] for sentence in base["chain"]]
    result["initial_generation"] = {"sentences": len(base_sentences),
                                    "terminated": sum(sentence["terminated"] for sentence in base_sentences),
                                    "truncated_at_64": sum(not sentence["terminated"] for sentence in base_sentences)}
    by_arm = {name: [d for d in decisions if d["arm"] == name] for name in protocol["arms"]}
    for arm in summary["arms"]:
        rows = by_arm[arm["name"]]
        if len(rows) != protocol["episodes"] * protocol["decisions_per_episode"]:
            raise AssertionError("missing arm decisions")
        if arm["verified_resumes"] != protocol["episodes"]:
            raise AssertionError("missing exact save/resume continuation")
        positive = sum(row["reward"] > 0 for row in rows)
        negative = sum(row["reward"] < 0 for row in rows)
        arm_result = {**arm, "positive_consequences": positive, "negative_consequences": negative,
                      "mean_reward": sum(row["reward"] for row in rows) / len(rows),
                      "mean_cost": sum(row["cost"] for row in rows) / len(rows),
                      "mean_before": {axis: sum(row["before"][axis] for row in rows) / len(rows) for axis in AXES},
                      "mean_after": {axis: sum(row["after"][axis] for row in rows) / len(rows) for axis in AXES},
                      "mean_delta": {axis: sum(row["after"][axis] - row["before"][axis] for row in rows) / len(rows) for axis in AXES}}
        result["arms"].append(arm_result)
    for row in decisions:
        action = row["action"]
        target = action["target"]
        if action["kind"] == 0:
            if action["source"] is not None or row["cost"] != 0 or row["host_rng_before"] != row["host_rng_after"]:
                raise AssertionError("KEEP changed source/cost/generation RNG")
            if row["before"] != row["after"]:
                raise AssertionError("KEEP changed sentence metrics")
        else:
            expected_source = target - 1 if action["kind"] == 1 else target + 1
            if action["source"] != expected_source or not 0 <= expected_source < protocol["sentences_per_episode"]:
                raise AssertionError("invalid host neighbor execution")
        for moments in (row["before"], row["after"]):
            if any(not math.isfinite(moments[axis]) or not 0 <= moments[axis] <= 1 for axis in AXES):
                raise AssertionError("invalid metric axis")
        reward = max(-1, min(1, sum(weight * (row["after"][axis] - row["before"][axis])
                                    for axis, weight in zip(AXES, WEIGHTS)) - 0.05 * row["cost"]))
        if abs(reward - row["reward"]) > 2e-7:
            raise AssertionError("receipt reward differs from the frozen raw-axis formula")
        if row["arm"] == "learned" and not row["counterfactual_inputs_equal"]:
            raise AssertionError("counterfactual input drift")
        slim = {key: value for key, value in row.items() if key not in ("type", "chain", "seed")}
        slim["chain"] = [decoded_sentence(s, vocabulary) for s in row["chain"]]
        result["decisions"].append(slim)
    disabled = by_arm["disabled"]
    for row in disabled:
        base = next(e for e in events if e["type"] == "base" and e["episode"] == row["episode"])
        if row["chain"] != base["chain"]:
            raise AssertionError("disabled agent changed the complete base chain")
    result["checks"] = {"raw_reward_recomputed": True, "typed_neighbor_bounds": True,
                        "disabled_chain_identical": True, "keep_rng_identical": True,
                        "exact_save_resume": True, "weight_counterfactual_inputs_identical": True}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="local exact pinned notorch-simple-llm checkout")
    parser.add_argument("--binary", type=Path, help="already-built independent runner")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the Git working tree")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    identities, vocabulary = prepare(output, args.reference, protocol)
    if args.prepare_only:
        write_json(output / "inputs.json", identities)
        print(json.dumps(identities, indent=2, ensure_ascii=False))
        return 0
    # Compile an immutable local snapshot so concurrent work in another lane
    # cannot change half of the numerical source during this experiment.
    snapshot = output / "source_snapshot"
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    if any(digest(snapshot / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("an experiment source changed during snapshotting; retry from its settled snapshot")
    binary = args.binary.resolve() if args.binary else output / "spa_agent_demo"
    build = ["cc", "-std=c11", "-O2", "-DUSE_SIMD", "-march=native", "-pthread", "-I.",
             "examples/spa_agent_demo.c", "spa_agent.c", "notorch.c", "-lm", "-o", str(binary)]
    if not args.binary:
        subprocess.run(build, cwd=snapshot, check=True)
    base_commit, source_status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    metadata = {"schema": 1, "protocol_sha256": digest(PROTOCOL), "protocol": protocol,
                "base_commit": base_commit, "source_commit": base_commit if not source_status else None,
                "source_status": source_status, "source_identity": "base commit plus exact per-file SHA-256 snapshot",
                "source_files_sha256": source_hashes, "compiled_from_immutable_snapshot": not bool(args.binary),
                "inputs": identities,
                "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
                "build": build[:-1] + ["<output>/spa_agent_demo"] if not args.binary else None,
                "binary_sha256": digest(binary), "machine": machine()}
    write_json(output / "metadata.json", metadata)
    runs, artifacts = [], {}
    for seed in protocol["seeds"]:
        prefix = output / f"simple-s{seed}"
        command = [str(binary), str(output / "simple.weights"), str(output / "corpus.tokens.u32"),
                   str(output / "vocabulary.u32"), str(prefix), str(seed)]
        started = time.monotonic()
        subprocess.run(command, cwd=ROOT, check=True)
        elapsed = time.monotonic() - started
        trace = prefix.with_suffix(".jsonl")
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        result = summarize(events, vocabulary, protocol)
        result["wall_seconds"] = elapsed
        runs.append(result)
        for path in sorted(output.glob(prefix.name + ".*")):
            if path.is_file():
                artifacts[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        write_json(output / "receipts.partial.json", {"metadata": metadata, "runs": runs, "artifacts": artifacts})
        print(json.dumps({"seed": seed, "wall_seconds": elapsed,
                          "arms": [{key: arm[key] for key in ("name", "actions", "mean_reward", "weights_changed_choices")}
                                   for arm in result["arms"]]}), flush=True)
    raw_archive = output / "raw_traces.jsonl.gz"
    raw_archive.write_bytes(gzip.compress(b"".join((output / f"simple-s{seed}.jsonl").read_bytes()
                                                 for seed in protocol["seeds"]), mtime=0))
    artifacts[raw_archive.name] = {"sha256": digest(raw_archive), "bytes": raw_archive.stat().st_size}
    receipts = {"schema": 1, "metadata": metadata, "runs": runs, "artifacts": artifacts}
    write_json(output / "receipts.json", receipts)
    print(f"Completed: {output / 'receipts.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
