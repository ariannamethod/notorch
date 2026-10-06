#!/usr/bin/env python3
"""Reproducible CPU training receipts for Chuck: Loss Architect.

Numerical work is compiled from this notorch tree. This script retrieves pinned
corpora, creates deterministic token streams, starts independent training
processes, and checks receipts. Store --output outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "simple": {
        "repository": "ariannamethod/notorch-simple-llm",
        "commit": "80b3bd611ed8c937efdc481dc07895c8e13e345d",
        "file": "dracula.txt",
        "sha256": "16faafe5f8e4a958fe4437e561035b1f82235602ab6ef9801cafe2bcc25aef5c",
        "recipe": "train_dracula.py",
        "parameters": 450688,
        "tokenizer": "sorted Unicode codepoints; universal newlines; vocabulary 94",
    },
    "hevlm": {
        "repository": "ariannamethod/notorch-diffusion",
        "commit": "1bc645071b3a0a3525fe89b8f45253f5faa28a1d",
        "file": "hevlm.txt",
        "sha256": "c00b8bcc31785798771aad5fb49f3b420b6a8eb3b7a20c28d0f09c967a52587e",
        "recipe": "ariannamethod/train_hevlm.c",
        "parameters": 1123456,
        "tokenizer": "UTF-8 bytes; vocabulary 256",
    },
}
ARMS = ("adam", "chuck", "legacy", "learned")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], **kwargs) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(body: str, output: Path, references: Path | None) -> dict:
    source = SOURCES[body]
    corpus = output / f"{body}.corpus"
    if not corpus.exists():
        if references:
            reference = references / source["repository"].split("/")[1]
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
            if actual != source["commit"]:
                raise ValueError(f"{reference.name}: expected {source['commit']}, got {actual}")
            raw = (reference / source["file"]).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{source['file']}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
        corpus.write_bytes(raw)
    if digest(corpus) != source["sha256"]:
        raise ValueError(f"{body}: corpus hash mismatch")
    raw = corpus.read_bytes()
    if body == "simple":
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        vocabulary = sorted(set(text))
        if len(vocabulary) != 94:
            raise ValueError("SimpleLLM vocabulary must contain 94 Unicode characters")
        mapping = {c: i for i, c in enumerate(vocabulary)}
        tokens = [mapping[c] for c in text]
    else:
        vocabulary = list(range(256))
        tokens = list(raw)
    token_path = output / f"{body}.tokens.u32"
    token_path.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
    vocab_path = output / f"{body}.vocab.json"
    write_json(vocab_path, vocabulary)
    return {**source, "bytes": len(raw), "tokens": len(tokens), "train_tokens": len(tokens) * 9 // 10,
            "token_sha256": digest(token_path), "vocab_sha256": digest(vocab_path)}


def machine() -> dict:
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count()}
    try:
        text = Path("/proc/cpuinfo").read_text()
        result["cpu_model"] = next(line.split(":", 1)[1].strip() for line in text.splitlines()
                                   if line.startswith("model name"))
        result["memory"] = {line.split(":")[0]: line.split(":", 1)[1].strip()
                            for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))}
    except (OSError, StopIteration):
        pass
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[name] = path.read_text().strip()
    return result


def check_arm_pair(left: dict, right: dict) -> dict:
    a, b = left["events"], right["events"]
    sa = [row for row in a if row["type"] == "step"]
    sb = [row for row in b if row["type"] == "step"]
    fields = ("step", "offset", "window_rng", "loss_before", "loss_after_same_window",
              "gradient_norm_before_clip", "frozen_tensors", "pre_chuck", "post_chuck")
    if len(sa) != len(sb):
        raise AssertionError("canonical/legacy trajectory length differs")
    for i, (x, y) in enumerate(zip(sa, sb), 1):
        for field in fields:
            if x[field] != y[field]:
                raise AssertionError(f"canonical/legacy differs at step {i}: {field}")
    if left["weights_sha256"] != right["weights_sha256"]:
        raise AssertionError("canonical/legacy final weights differ")
    return {"gate": "canonical_legacy_parity", "steps": len(sa), "result": "PASS",
            "final_weights_identical": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seeds", default="42,73")
    parser.add_argument("--bodies", default="simple,hevlm")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--binary", type=Path, help="prebuilt runner; otherwise compile current tree")
    parser.add_argument("--config", type=Path, help="learned Architect JSON config")
    parser.add_argument("--allow-dirty", action="store_true", help="development smoke only; recorded explicitly")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the Git working tree")
    bodies = args.bodies.split(",")
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    if set(bodies) - SOURCES.keys() or set(arms) - set(ARMS):
        parser.error("unknown body or arm")
    if args.steps < 1 or args.threads < 1 or any(s < 1 or s > 0xffffffff for s in seeds):
        parser.error("steps, threads, and seeds must be positive; seeds fit uint32")
    output.mkdir(parents=True, exist_ok=True)
    datasets = {body: prepare(body, output, args.references) for body in bodies}
    write_json(output / "datasets.json", datasets)
    if args.prepare_only:
        print(json.dumps(datasets, indent=2))
        return 0
    status = command(["git", "status", "--porcelain"])
    if status and not args.allow_dirty:
        parser.error("final experiments require a committed tree; use --allow-dirty for development smoke")
    source_files = ["notorch.c", "notorch.h", "notorch_simd.h", "chuck_architect_impl.h", "chuck_architect.h",
                    "examples/chuck_architect_train.c", "experiments/chuck_loss_architect/run.py"]
    source_hashes = {name: digest(ROOT / name) for name in source_files if (ROOT / name).is_file()}
    binary = args.binary.resolve() if args.binary else output / "chuck_architect_train"
    compile_command = ["cc", "-O2", "-std=gnu11", "-DUSE_SIMD", "-march=native", "-pthread", "-I.",
                       "examples/chuck_architect_train.c", "notorch.c"]
    compile_command += ["-lm", "-o", str(binary)]
    if not args.binary:
        subprocess.run(compile_command, cwd=ROOT, check=True)
    metadata = {"schema": 1, "source_commit": command(["git", "rev-parse", "HEAD"]),
                "source_status": status, "source_files_sha256": source_hashes,
                "binary_sha256": digest(binary), "compiler": command(["cc", "--version"]).splitlines()[0],
                "build": ({"type": "caller-supplied", "command": None} if args.binary else
                          {"type": "compiled-current-tree", "command": compile_command[:-1] + ["<output>/chuck_architect_train"]}),
                "machine": machine(), "threads": args.threads, "steps": args.steps, "seeds": seeds,
                "bodies": bodies, "arms": arms, "lr": args.lr, "datasets": datasets,
                "objective": {"feedback": "same-window loss before minus after actual training step",
                              "evaluation": "eight fixed windows in final corpus tenth; initial and final",
                              "window_stream": "SplitMix64 seed, modulo train_tokens-context; context=64"}}
    if args.config:
        metadata["config"] = json.loads(args.config.read_text())
        metadata["config_sha256"] = digest(args.config)
    write_json(output / "manifest.json", metadata)
    env = dict(os.environ, NT_SIMD_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads))
    runs = []
    artifacts = {}
    gates = []
    for body in bodies:
        for seed in seeds:
            group = {}
            for arm in arms:
                name = f"{body}-{arm}-s{seed}"
                prefix = output / name
                invocation = [str(binary), body, arm, str(output / f"{body}.tokens.u32"), str(prefix),
                              str(args.steps), str(seed), str(args.lr)]
                if args.config and arm == "learned":
                    invocation.append(str(args.config.resolve()))
                time_path = output / f"{name}.resources.json"
                external_time = Path("/usr/bin/time")
                measured = ([str(external_time), "-f", '{"max_rss_kib":%M,"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S}',
                             "-o", str(time_path)] + invocation) if external_time.exists() else invocation
                print("RUN", name, flush=True)
                with (output / f"{name}.log").open("w") as log:
                    completed = subprocess.run(measured, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"{name}: exit {completed.returncode}; see {name}.log")
                events = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
                if len([r for r in events if r["type"] == "step"]) != args.steps:
                    raise AssertionError(f"{name}: incomplete step receipts")
                if events[0]["parameters"] != SOURCES[body]["parameters"]:
                    raise AssertionError(f"{name}: body parameter count mismatch")
                summary = events[-1]
                weights_sha = digest(output / f"{name}.final.bin")
                initial_sha = digest(output / f"{name}.initial.bin")
                record = {"name": name, "body": body, "arm": arm, "seed": seed,
                          "parameters": events[0]["parameters"], "initial_weights_sha256": initial_sha,
                          "weights_sha256": weights_sha, "receipts_sha256": digest(output / f"{name}.jsonl"),
                          "summary": summary}
                if time_path.exists():
                    record["resources"] = json.loads(time_path.read_text())
                else:
                    record["resources"] = {"source": "getrusage through final heldout evaluation",
                                           "max_rss_kib": summary["max_rss_kib"],
                                           "wall_seconds": summary["seconds"],
                                           "user_seconds": summary["user_seconds"],
                                           "system_seconds": summary["system_seconds"]}
                if arm in ("legacy", "learned"):
                    for suffix in (".policy.initial.bin", ".policy.final.bin"):
                        path = output / (name + suffix)
                        if path.exists(): record[suffix.strip(".") + "_sha256"] = digest(path)
                actions = [r.get("architect", {}).get("action", {}).get("type") for r in events if r["type"] == "step"]
                record["action_counts"] = {str(a): actions.count(a) for a in sorted(set(actions), key=str) if a is not None}
                runs.append(record)
                group[arm] = {"events": events, **record}
                for path in sorted(output.glob(name + ".*")):
                    artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
                print(json.dumps(record, ensure_ascii=False), flush=True)
                write_json(output / "results.json", {"manifest": metadata, "runs": runs, "gates": gates})
            initials = {r["initial_weights_sha256"] for r in group.values()}
            if len(initials) != 1:
                raise AssertionError(f"{body} seed={seed}: arms initialized with different weights")
            gates.append({"gate": "matched_initial_weights", "body": body, "seed": seed, "result": "PASS"})
            if "chuck" in group and "legacy" in group:
                gates.append({"body": body, "seed": seed, **check_arm_pair(group["chuck"], group["legacy"])})
    result = {"manifest": metadata, "runs": runs, "gates": gates, "artifacts": artifacts}
    write_json(output / "results.json", result)
    print(f"Saved {len(runs)} runs, {len(gates)} gates to {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
