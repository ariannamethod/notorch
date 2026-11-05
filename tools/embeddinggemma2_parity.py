#!/usr/bin/env python3
"""Compare the native C encoder with already frozen NPZ reference fixtures.

NumPy is used only for reading and comparing recorded arrays. Every model
operation is performed by the supplied native executable. No tokenizer, torch,
or model Python implementation is imported here.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def comparison(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64).ravel(), np.asarray(expected, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite recorded values")
    delta, length = a - b, float(np.linalg.norm(b))
    denominator = float(np.linalg.norm(a)) * length
    return {"count": len(a), "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta)) / max(length, 1e-12),
            "cosine": float(a @ b) / denominator if denominator else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fixtures", nargs="+", default=[f"g{i:02d}" for i in range(1, 7)])
    p.add_argument("--max-abs", type=float, default=2e-5)
    p.add_argument("--min-cosine", type=float, default=0.999999)
    p.add_argument("--max-norm-error", type=float, default=2e-6)
    p.add_argument("--internal-relative-l2", type=float, default=2e-3)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    references = {fid: args.reference / f"golden_{fid}.npz" for fid in args.fixtures}
    inputs = args.output / "tokens.tsv"
    with inputs.open("x") as f:
        for fid, path in references.items():
            with np.load(path) as golden:
                ids = golden["input_ids"].ravel().tolist()
                mask = golden["attention_mask"].ravel().tolist()
                f.write(f"{fid}\t" + " ".join(map(str, ids)) + "\t" + " ".join(map(str, mask)) + "\n")
    command = [str(args.binary.resolve()), str(args.model.resolve()), str(inputs.resolve()),
               str((args.output / "vectors.jsonl").resolve()), "--trace", str((args.output / "traces").resolve())]
    settings = {"format": "notorch.embeddinggemma2.parity.v1", "command": command,
                "binary_sha256": sha(args.binary), "model_sha256": sha(args.model),
                "reference_sha256": {k: sha(v) for k, v in references.items()},
                "input_sha256": sha(inputs), "platform": platform.platform(),
                "threads": args.threads, "thresholds": {"normalized_max_abs": args.max_abs,
                "normalized_min_cosine": args.min_cosine, "normalized_max_norm_error": args.max_norm_error,
                "internal_relative_l2": args.internal_relative_l2}}
    (args.output / "manifest.initial.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = dict(os.environ, NT_SIMD_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    start = time.perf_counter()
    with (args.output / "native.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"native encoder exited {result.returncode}; see native.log")
    vectors = {r["id"]: r for r in map(json.loads, (args.output / "vectors.jsonl").read_text().splitlines())}
    with (args.output / "traces/index.tsv").open() as f:
        traces = list(csv.DictReader(f, delimiter="\t"))
    report = {"pass": True, "seconds": time.perf_counter() - start, "fixtures": {}}
    for fid, path in references.items():
        g = np.load(path)
        metrics = {}
        for row in traces:
            if row["id"] != fid:
                continue
            site, layer = row["site"], int(row["layer"])
            key = f"block_{layer:02d}" if site == "block" else site
            if site == "post_attention":
                key = f"layer_{layer:02d}_post_attention_residual"
            elif site == "post_ffn":
                key = f"layer_{layer:02d}_post_mlp_residual"
            if key not in g:
                continue
            actual = np.fromfile(row["path"], dtype="<f4")
            if actual.size != int(row["tokens"]) * int(row["width"]):
                raise ValueError("truncated trace")
            metrics[key] = comparison(actual, g[key])
        metrics["pooled"] = comparison(vectors[fid]["pooled"], g["pooled"])
        metrics["embedding"] = comparison(vectors[fid]["embedding"], g["embeddings"])
        passed = all(v["relative_l2"] <= args.internal_relative_l2 for k, v in metrics.items() if k != "embedding")
        passed &= metrics["embedding"]["max_abs"] <= args.max_abs
        passed &= metrics["embedding"]["cosine"] >= args.min_cosine
        norm_error = abs(float(np.linalg.norm(np.asarray(vectors[fid]["embedding"], dtype=np.float64))) - 1.0)
        passed &= norm_error <= args.max_norm_error
        # Missing hooks must fail rather than silently reduce the parity claim.
        expected = {"input_embeddings", "ple", "final_norm", "projected", "pooled", "embedding"}
        expected.update(f"block_{i:02d}" for i in range(24))
        if not expected.issubset(metrics):
            raise ValueError(f"missing comparisons: {expected - metrics.keys()}")
        report["fixtures"][fid] = {"pass": bool(passed), "tokens": vectors[fid]["tokens"],
                                   "norm_error": norm_error, "metrics": metrics}
        report["pass"] &= bool(passed)
        g.close()
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "seconds": report["seconds"], "output": str(args.output)}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
