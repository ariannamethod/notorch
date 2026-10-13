#!/usr/bin/env python3
"""Independent GGUF parser and full PEN7 payload recovery gate (stdlib only).

python3 tests/test_pen7_convert.py /path/to/pen7_to_gguf
python3 tests/test_pen7_convert.py /path/to/pen7_to_gguf --sources \
    ../1984/weights/penelope.bin ../janus/weights/nanojanus.bin
The native converter and deployed models do not depend on Python.
"""
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import subprocess
import tempfile

HEADER = (0x50454E37, 2048, 1984, 448, 896, 7, 8, 256)
SOURCE_BYTES = 78_477_152
TOKENIZER_SHA = "664afe692a006a54fc00406209042f6d452748d64a883ecfab9ef66d41f8bdf2"


def directory():
    out = [("token_embd.weight", (448, 2048)),
           ("position_embd.weight", (448, 256)),
           ("output_norm.weight", (448,)), ("output.weight", (448, 2048))]
    layer = [("attn_norm.weight", (448,)), ("attn_q.weight", (448, 448)),
             ("attn_k.weight", (448, 448)), ("attn_v.weight", (448, 448)),
             ("attn_output.weight", (448, 448)), ("resonance.weight", (448, 448)),
             ("resonance_gate.weight", (2,)), ("ffn_norm.weight", (448,)),
             ("ffn_gate.weight", (448, 896)), ("ffn_up.weight", (448, 896)),
             ("ffn_down.weight", (896, 448))]
    return out + [(f"blk.{i}.{n}", s) for i in range(8) for n, s in layer]


class Reader:
    """Deliberately independent of gguf.c and the converter's structures."""
    def __init__(self, data):
        self.data, self.pos, self.values = data, 0, {}

    def number(self, fmt):
        v = struct.unpack_from("<" + fmt, self.data, self.pos)[0]
        self.pos += struct.calcsize("<" + fmt)
        return v

    def string(self):
        n = self.number("Q")
        assert self.pos + n <= len(self.data)
        s = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, n = self.number("I"), self.number("Q")
            assert n < 1_000_000
            return [self.value(item) for _ in range(n)]
        return self.number({0: "B", 1: "b", 2: "H", 3: "h", 4: "I",
                            5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])

    def read(self):
        assert self.number("I") == 0x46554747 and self.number("I") == 3
        nt, nk = self.number("Q"), self.number("Q")
        assert nt == 92 and nk < 128
        meta = {}
        for _ in range(nk):
            key = self.string()
            assert key not in meta
            kind = self.number("I")
            self.values[key] = self.pos
            meta[key] = self.value(kind)
        tensors = []
        for _ in range(nt):
            name, nd = self.string(), self.number("I")
            assert 1 <= nd <= 2
            shape = tuple(self.number("Q") for _ in range(nd))
            tensors.append((name, shape, self.number("I"), self.number("Q")))
        self.pos = (self.pos + 31) & ~31
        return meta, tensors, self.pos


def audit_q8(source, packed):
    """Independent scalar format arithmetic, using Python's IEEE packing oracle."""
    max_error, sum_sq = 0.0, 0.0
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    blocks = len(packed)//34
    for block in range(blocks):
        values = struct.unpack_from("<32f", source, block*128)
        d = f32(max(abs(v) for v in values)/127.0)
        # Both actual PEN7 bodies have normal, nonzero half scales throughout.
        scale_bytes = struct.pack("<e", d)
        assert packed[block*34:block*34+2] == scale_bytes
        scale = struct.unpack("<e", scale_bytes)[0]
        inverse = f32(1/d) if d else 0.0
        scaled = struct.unpack("<32f", struct.pack("<32f", *(v*inverse for v in values)))
        quants = [max(-128, min(127, math.floor(v+0.5) if v>=0 else -math.floor(-v+0.5)))
                  for v in scaled]
        assert packed[block*34+2:block*34+34] == struct.pack("<32b", *quants)
        for original, q in zip(values, quants):
            # A half significand multiplied by an int8 fits exactly in binary32.
            error = abs(original-q*scale)
            max_error = max(max_error, error)
            sum_sq += error*error
    return max_error, sum_sq, blocks


def audit(source, converted):
    with source.open("rb") as sf, converted.open("rb") as gf:
        with mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ) as src, \
                mmap.mmap(gf.fileno(), 0, access=mmap.ACCESS_READ) as dst:
            assert len(src) == SOURCE_BYTES and struct.unpack_from("<8I", src) == HEADER
            meta, tensors, data = Reader(dst).read()
            assert meta["general.architecture"] == "penelope"
            q8 = meta["penelope.conversion"] == "q8_0-matrices-v1"
            assert meta["general.file_type"] == (7 if q8 else 0)
            if q8:
                assert meta["general.quantization_version"] == 2
            assert meta["penelope.source.sha256"] == hashlib.sha256(src).hexdigest()
            assert meta["penelope.conversion"] == ("q8_0-matrices-v1" if q8 else "f32-lossless-v1")
            assert meta["penelope.parameter_count"] == 19_619_280
            dims = {"embedding_length": 448, "context_length": 256, "vocab_size": 2048,
                    "word_vocab_size": 1984, "block_count": 8, "feed_forward_length": 896,
                    "attention.head_count": 7, "attention.head_dim": 64,
                    "rope.dimension_count": 64, "rope.freq_base": 10000.0}
            for key, val in dims.items():
                assert meta["penelope." + key] == val
            assert meta["penelope.position_embedding"] is True
            assert meta["penelope.rope.layout"] == "split-half"
            assert meta["penelope.resonance"] == "dense-gated"
            assert meta["penelope.silu_cutoff"] == -20.0
            assert meta["penelope.attention.layer_norm_rms_epsilon"] == struct.unpack("<f", struct.pack("<f", 1e-5))[0]
            assert meta["tokenizer.ggml.model"] == "penelope"
            assert meta["tokenizer.ggml.add_bos_token"] is False
            assert meta["tokenizer.ggml.add_eos_token"] is False
            assert meta["tokenizer.ggml.bos_token_id"] == meta["tokenizer.ggml.eos_token_id"] == -1
            assert meta["tokenizer.penelope.algorithm"] == "byte-rank-pass-v1"
            assert meta["tokenizer.penelope.source_contract"] == "ariannamethod/1984:penelope.c@a0d20019587ff14e5516eb050ba0876d6edd0baa"
            assert meta["tokenizer.penelope.lowercase_ascii"] is True
            assert meta["tokenizer.penelope.max_input_bytes"] == 8191
            pairs = meta["tokenizer.penelope.merge_pairs"]
            assert len(pairs) == 3584
            assert hashlib.sha256(struct.pack("<3584i", *pairs)).hexdigest() == TOKENIZER_SHA
            assert meta["tokenizer.penelope.sha256"] == TOKENIZER_SHA
            encoded, raw, extra = [], [], 0
            for b in range(256):
                printable = 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255
                cp = b if printable else 256 + extra
                extra += not printable
                encoded.append(chr(cp))
                raw.append(bytes([b]))
            merges = []
            for i in range(1792):
                l, r = pairs[2*i:2*i+2]
                assert 0 <= l < 256+i and 0 <= r < 256+i
                merges.append(encoded[l] + " " + encoded[r])
                encoded.append(encoded[l] + encoded[r])
                raw.append(raw[l] + raw[r])
            assert meta["tokenizer.ggml.tokens"] == encoded
            assert meta["tokenizer.ggml.merges"] == merges
            # Native oracle exercises byte IDs, ASCII lowering and cross-space merges.
            ids = list(b"memory turns through the silence")
            for i in range(1792):
                left, right = pairs[2*i:2*i+2]
                result, j = [], 0
                while j < len(ids):
                    if j+1 < len(ids) and ids[j:j+2] == [left, right]:
                        result.append(256+i)
                        j += 2
                    else:
                        result.append(ids[j])
                        j += 1
                ids = result
            assert ids == [1472, 116, 1383, 256, 1307, 294, 342, 108, 1284]
            assert b"".join(raw[i] for i in ids) == b"memory turns through the silence"
            source_pos, previous_end = 32, 0
            max_error, sum_sq, quantized_blocks = 0.0, 0.0, 0
            for expected, tensor in zip(directory(), tensors):
                name, shape, dtype, off = tensor
                target = 8 if q8 and len(shape)==2 and name!="position_embd.weight" else 0
                assert (name, shape) == expected and dtype == target and off % 32 == 0
                count = shape[0] * (shape[1] if len(shape) == 2 else 1)
                source_size = 4*count
                size = (count//32)*34 if target==8 else source_size
                assert off >= previous_end and data+off+size <= len(dst)
                if target==8:
                    err, sq, blocks = audit_q8(src[source_pos:source_pos+source_size],
                                               dst[data+off:data+off+size])
                    max_error = max(max_error, err)
                    sum_sq += sq
                    quantized_blocks += blocks
                else:
                    assert src[source_pos:source_pos+size] == dst[data+off:data+off+size], name
                previous_end = off+size
                source_pos += source_size
            assert source_pos == SOURCE_BYTES and len(dst) == data+previous_end
            return {"source_sha256": hashlib.sha256(src).hexdigest(),
                    "gguf_sha256": hashlib.sha256(dst).hexdigest(),
                    "gguf_bytes": len(dst), "tensor_count": len(tensors),
                    "parameter_count": 19_619_280, "all_parameter_bytes_equal": not q8,
                    "all_q8_blocks_equal": quantized_blocks if q8 else None,
                    "max_parameter_abs_error": max_error,
                    "rms_parameter_error": math.sqrt(sum_sq/19_619_280)}


def run(binary, source, dest, good=True, q8=False):
    proc = subprocess.run([str(binary)] + (["--q8_0"] if q8 else []) + [str(source), str(dest)],
                          capture_output=True, text=True)
    assert (proc.returncode == 0) == good, (proc.returncode, proc.stdout, proc.stderr)
    if not good:
        assert not dest.exists() and not list(dest.parent.glob(dest.name + ".partial.*"))
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("converter", type=Path)
    ap.add_argument("--sources", type=Path, nargs=2)
    ap.add_argument("--q8_0", action="store_true")
    args = ap.parse_args()
    binary = args.converter.resolve()
    with tempfile.TemporaryDirectory(prefix="nt-pen7-convert-") as td:
        td = Path(td)
        source, dest = td/"fixture.bin", td/"fixture.gguf"
        # Sparse exact-size body with sentinels in every tensor and signed-zero/subnormal boundaries.
        with source.open("wb") as f:
            f.write(struct.pack("<8I", *HEADER))
            f.truncate(SOURCE_BYTES)
            off = 32
            for i, (name, shape) in enumerate(directory()):
                n = shape[0] * (shape[1] if len(shape) == 2 else 1)
                halfscale_ok = args.q8_0 and len(shape)==2 and name!="position_embd.weight"
                tail_bits = (0x3d000000+i) if halfscale_ok else i+1
                for at, bits in ((0, 0x3f800000+i), (n//2, 0x80000000), (n-1, tail_bits)):
                    f.seek(off+at*4)
                    f.write(struct.pack("<I", bits))
                off += n*4
        run(binary, source, dest, q8=args.q8_0)
        result = audit(source, dest)
        twin = td/"twin.gguf"
        run(binary, source, twin, q8=args.q8_0)
        assert hashlib.sha256(twin.read_bytes()).hexdigest() == result["gguf_sha256"]
        # Existing destination, source path, hardlink and dangling symlink are preserved.
        before = dest.read_bytes()
        for output in (dest, source):
            p = subprocess.run([str(binary)] + (["--q8_0"] if args.q8_0 else []) +
                               [str(source), str(output)], capture_output=True)
            assert p.returncode != 0
        hard, sym = td/"hard.bin", td/"symbolic.gguf"
        hard.hardlink_to(source)
        sym.symlink_to(td/"missing-target")
        for output in (hard, sym):
            p = subprocess.run([str(binary)] + (["--q8_0"] if args.q8_0 else []) +
                               [str(source), str(output)], capture_output=True)
            assert p.returncode != 0
        assert dest.read_bytes() == before and sym.is_symlink()
        # Wrong magic, each dimension, NaN and +/-Inf at first/middle/last parameters.
        rejected = 0
        with source.open("r+b") as f:
            patches = [(i*4, struct.pack("<I", HEADER[i]^1)) for i in range(8)]
            patches += [(off, struct.pack("<I", bits))
                        for off in (32, 32+(19_619_280//2)*4, SOURCE_BYTES-4)
                        for bits in (0x7fc00001, 0x7f800000, 0xff800000)]
            if args.q8_0:
                # Finite source values can still overflow the half scale or
                # produce an infinite float reciprocal in an otherwise-zero block.
                patches += [(32+32*4, struct.pack("<I", bits))
                            for bits in (0x7f7fffff, 0x00800000)]
            for off, patch in patches:
                f.seek(off); saved = f.read(4)
                f.seek(off); f.write(patch); f.flush()
                run(binary, source, td/"bad.gguf", good=False, q8=args.q8_0)
                f.seek(off); f.write(saved); f.flush()
                rejected += 1
            f.seek(SOURCE_BYTES-4); tail = f.read(4)
            f.truncate(SOURCE_BYTES-4); f.flush()
            run(binary, source, td/"bad.gguf", good=False, q8=args.q8_0); rejected += 1
            f.seek(SOURCE_BYTES-4); f.write(tail+b"x"); f.flush()
            run(binary, source, td/"bad.gguf", good=False, q8=args.q8_0); rejected += 1
            f.truncate(SOURCE_BYTES); f.flush()
        run(binary, td/"missing.bin", td/"bad.gguf", good=False, q8=args.q8_0); rejected += 1
        # Deliberate conversion defects: payload, dimensions, native merge IDs,
        # alphabet surface and tokenizer algorithm all make the independent gate red.
        with twin.open("r+b") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
                r = Reader(data)
                r.read()
                locations = r.values
                last = data[-1]
            defects = [
                (twin.stat().st_size-1, bytes([last^1])),
                (locations["penelope.embedding_length"], struct.pack("<I", 449)),
                (locations["tokenizer.penelope.merge_pairs"]+12, struct.pack("<i", 2048)),
                (locations["tokenizer.ggml.tokens"]+12+8, "ā".encode()),
                (locations["tokenizer.penelope.algorithm"]+8, b"byte-rank-pass-v2"),
            ]
            for off, patch in defects:
                f.seek(off); saved = f.read(len(patch))
                f.seek(off); f.write(patch); f.flush()
                try:
                    audit(source, twin)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(f"conversion defect at {off} escaped recovery gate")
                f.seek(off); f.write(saved); f.flush()
        print(json.dumps({"fixture": result, "rejected_inputs": rejected,
                          "protected_existing_paths": 4, "deterministic": True,
                          "conversion_defects_detected": len(defects)}, sort_keys=True))
        if args.sources:
            for i, src in enumerate(args.sources):
                dst = td/f"real-{i}.gguf"
                proc = run(binary, src.resolve(), dst, q8=args.q8_0)
                report = audit(src.resolve(), dst)
                assert json.loads(proc.stdout)["gguf_sha256"] == report["gguf_sha256"]
                print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
