#!/usr/bin/env python3
"""Copy the text tensors of an EmbeddingGemma 2 safetensors into NoTorch GGUF.

Standard library only. No tensor arithmetic, dtype conversion, tokenizer repair,
or model execution. Source BF16/F32 bytes are preserved exactly. The companion
receipt binds every tensor to its source bytes. Multimodal tensors are omitted.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as src:
        for b in iter(lambda: src.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def convert(snapshot, output, max_tokens=8192):
    source = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    c = json.loads(config_path.read_text())["text_config"]
    if c["model_type"] != "embedding_gemma2_text":
        raise ValueError("expected embedding_gemma2_text")
    if c["hidden_activation"] != "gelu_pytorch_tanh" or c["attention_bias"]:
        raise ValueError("only bias-free GELU-tanh text checkpoints are supported")
    if not 1 <= max_tokens <= min(8192, c["max_position_embeddings"]):
        raise ValueError("max-tokens must be within 1..8192 and model positions")
    L, E, P = c["num_hidden_layers"], c["hidden_size"], c["hidden_size_per_layer_input"]
    if len(c["layer_types"]) != L:
        raise ValueError("layer_types length mismatch")
    metadata = [("general.architecture", 8, "embeddinggemma2"),
                ("general.name", 8, "EmbeddingGemma 2 text, unchanged source weights")]
    revision = snapshot / "SOURCE_REVISION"
    if revision.exists():
        metadata.append(("general.source.revision", 8, revision.read_text().strip()))
    prefix = "embeddinggemma2."
    for key, val in {
        "block_count": L, "embedding_length": E, "output_length": c["embedding_dim"],
        "vocab_size": c["vocab_size"], "context_length": max_tokens,
        "attention.head_count": c["num_attention_heads"],
        "feed_forward_length": c["intermediate_size"], "ple_length": P,
    }.items():
        metadata.append((prefix + key, 4, val))
    for key, val in {"rms_epsilon": c["rms_norm_eps"], "embed_scale": math.sqrt(E),
                     "ple_scale": E ** -0.5}.items():
        metadata.append((prefix + key, 6, val))
    allowed_overrides = {"head_dim", "num_key_value_heads"}
    for l, layer_type in enumerate(c["layer_types"]):
        override = c.get("per_layer_config", {}).get(str(l).zfill(2), {})
        if set(override) - allowed_overrides:
            raise ValueError(f"unsupported layer override: {override}")
        rope = c["rope_parameters"][layer_type]
        if rope["rope_type"] != "default" or set(rope) - {"rope_type", "rope_theta"}:
            raise ValueError("only unscaled default RoPE is supported")
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError("unsupported attention type")
        for key, val in {
            "head_dim": override.get("head_dim", c["head_dim"]),
            "kv_heads": override.get("num_key_value_heads", c["num_key_value_heads"]),
            "window": c["sliding_window"] if layer_type == "sliding_attention" else 0,
        }.items():
            metadata.append((prefix + f"layer.{l}." + key, 4, val))
        metadata.append((prefix + f"layer.{l}.rope_base", 6, rope["rope_theta"]))
    if len(metadata) > 128:
        raise ValueError("metadata exceeds NoTorch parser limit")
    with source.open("rb") as src:
        header_size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(header_size))
        tensors = []
        offset = 0
        for name, desc in sorted(header.items()):
            if not name.startswith("language_model."):
                continue
            dtype = {"F32": (0, 4), "BF16": (30, 2)}.get(desc["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported dtype: {name}: {desc['dtype']}")
            shape = desc["shape"]
            if not 1 <= len(shape) <= 2 or any(d <= 0 for d in shape):
                raise ValueError(f"unsupported shape: {name}: {shape}")
            lo, hi = desc["data_offsets"]
            if hi - lo != math.prod(shape) * dtype[1] or lo < 0:
                raise ValueError(f"invalid byte span: {name}")
            tensors.append({"source_name": name, "name": name.removeprefix("language_model."),
                            "shape": shape, "dtype": desc["dtype"], "gguf_dtype": dtype[0],
                            "source_offsets": [lo, hi], "offset": offset})
            offset = (offset + hi - lo + 31) & ~31
        expected_count = 5 + L * 17
        if len(tensors) != expected_count:
            raise ValueError(f"expected {expected_count} text tensors, got {len(tensors)}")
        if output.exists() or output.with_suffix(output.suffix + ".json").exists():
            raise FileExistsError("output or receipt already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        created = False
        try:
            with temporary.open("xb") as dst:
                created = True
                dst.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)))
                for key, dtype, value in metadata:
                    dst.write(string(key) + struct.pack("<I", dtype))
                    dst.write(string(value) if dtype == 8 else struct.pack("<I" if dtype == 4 else "<f", value))
                for t in tensors:
                    dst.write(string(t["name"]) + struct.pack("<I", len(t["shape"])))
                    for dim in reversed(t["shape"]):
                        dst.write(struct.pack("<Q", dim))
                    dst.write(struct.pack("<IQ", t["gguf_dtype"], t["offset"]))
                dst.write(b"\0" * (-dst.tell() % 32))
                data_start = dst.tell()
                for t in tensors:
                    dst.write(b"\0" * (data_start + t["offset"] - dst.tell()))
                    lo, hi = t["source_offsets"]
                    src.seek(8 + header_size + lo)
                    remaining, digest = hi - lo, hashlib.sha256()
                    while remaining:
                        b = src.read(min(remaining, 1 << 20))
                        if not b:
                            raise ValueError("truncated source tensor")
                        digest.update(b)
                        dst.write(b)
                        remaining -= len(b)
                    t["sha256"] = digest.hexdigest()
                dst.flush()
                os.fsync(dst.fileno())
            # link is create-only even if another writer races our initial check.
            os.link(temporary, output)
            temporary.unlink()
        except BaseException:
            if created:
                temporary.unlink(missing_ok=True)
            raise
    receipt = {"format": "notorch.embeddinggemma2.convert.v1", "source_revision":
               revision.read_text().strip() if revision.exists() else None,
               "source_sha256": sha256(source), "config_sha256": sha256(config_path),
               "output_sha256": sha256(output), "max_tokens": max_tokens,
               "tensors": tensors, "metadata": metadata,
               "arithmetic": "none; BF16/F32 source bytes copied unchanged"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    receipt = convert(args.snapshot, args.output, args.max_tokens)
    print(json.dumps({"tensors": len(receipt["tensors"]), "sha256": receipt["output_sha256"]}))
