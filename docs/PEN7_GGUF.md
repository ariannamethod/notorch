# PEN7 bodies in GGUF

The native converter `tools/pen7_to_gguf.c` packages the Penelope and NanoJanus
PEN7 bodies as GGUF v3. Its default preserves every F32 parameter bit; explicit
`--q8_0` packs the matrices using NoTorch's native quantizer. `general.architecture`
is `penelope`: both bodies share this exact numerical family. The NoTorch
harness family runs its learned position embeddings, split-half RoPE, dense
resonance branch and two-scalar attention/resonance gate.

The [Arianna Method Manifesto](../ARIANNA_METHOD_MANIFESTO.md) governs this work.

## Conversion

```sh
cc -O2 -std=gnu11 -ffp-contract=off -pthread -I. \
  tools/pen7_to_gguf.c gguf.c sha256.c notorch.c -lm -o pen7_to_gguf
./pen7_to_gguf /path/to/1984/weights/penelope.bin /path/to/haiku1-f32.gguf
./pen7_to_gguf --q8_0 /path/to/1984/weights/penelope.bin /path/to/haiku1-q8_0.gguf
./pen7_to_gguf --q8_0 /path/to/janus/weights/nanojanus.bin /path/to/haiku2-q8_0.gguf
```

Conversion uses NoTorch's GGUF writer, native quantizer and SHA256 implementation,
libc, libm and pthread support.
It maps the source read-only, verifies its header, exact size and finite F32
parameters, and writes a temporary sibling of the requested output. It opens
that GGUF again, recovers every tensor, compares every parameter byte and checks
the embedded tokens/merge IDs. A no-replace hard link publishes the verified
file. Q8 mode rechecks packed bytes against the canonical row quantizer and
expands all tensors to F32 for finite-value and parameter-error measurements.
Existing files, hard links and symbolic links are preserved. A failed
conversion removes its temporary file and publishes no output. JSON on stdout
records both SHA256 values, size, counts, mode and measured recovery errors.

The source format is exactly eight little-endian int32 header values followed
by 19,619,280 F32 parameters, totalling **78,477,152 bytes**:

```
[0x50454e37, 2048, 1984, 448, 896, 7, 8, 256]
 PEN7       BPE   words dim  FFN  heads layers context
```

The converter accepts this geometry and its canonical PEN7 BPE table. A PEN7
file carries no tokenizer; the pinned table becomes explicit in the GGUF.
Other geometries require a corresponding format revision. The host contract
is little-endian IEEE binary32, checked before conversion.

## Recorded bodies

| Body | Source revision | Source SHA256 |
|---|---|---|
| Penelope | [`1984@a0d2001`](https://github.com/ariannamethod/1984/tree/a0d20019587ff14e5516eb050ba0876d6edd0baa) | `afff67c780a7c78fdc1f76e521dfda30d65e68b95fe5bd324bb392b84ba3cd38` |
| NanoJanus | [`janus@5a945a9`](https://github.com/ariannamethod/janus/tree/5a945a9d7c1dc9abfa217f685f0a81455f9ac493) | `d675df2ee3a150d63bc73935a84060280d0f6b0aa84283e4b78ab905180b576c` |

| Converted asset | Bytes | SHA256 |
|---|---:|---|
| Penelope F32 reference | 78,580,832 | `0b5d4dd69657ff169a2b2d02479f8fa91064eb3b4a58cf3dd226aa38c02efad1` |
| NanoJanus F32 reference | 78,580,832 | `cc7e094e42c36263f6140d135c445dfd2aa221718cc8e3662611ba4cc1b4e1ad` |
| Penelope Q8_0 | 21,308,576 | `50f159a2c34952bd9e6349880b4b20f8bbc0ac4b1f1dc2c1f6e4768732548b66` |
| NanoJanus Q8_0 | 21,308,576 | `16909b8590bd7fbbf15ab11b2dbf3743d612da8c617aca2dda03d640fef13e29` |

Each F32 reference contains **92 F32 tensors** and **19,619,280 parameters**. These lossless
F32 files are 103,680 bytes larger than their PEN7 originals: GGUF embeds the
tokenizer, provenance, numerical conventions and tensor directory, with 32-byte
alignment. The two assets together occupy 157,161,664 file bytes (149.881 MiB).
The F32 references retain complete original numerical identity. The explicit
Q8 bodies are the compact form described below.

## Q8 storage, F32 execution

`--q8_0` converts **66 matrix tensors**, or **19,496,960 parameters**, into
609,280 standard Q8_0 blocks: each block carries one F16 scale and 32 signed
int8 values. The learned position table, all RMSNorm vectors and both gate
scalars per layer remain F32: **26 tensors / 122,320 parameters**. Payload is
21,204,800 bytes; metadata, tokenizer and alignment bring either file to
**21,308,576 bytes**. `general.file_type=7`, quantization version 2 and
`penelope.conversion=q8_0-matrices-v1` state the choice in the file itself.

NoTorch's `nt_quantize_row(..., GGUF_TYPE_Q8_0)` owns the conversion. It obtains
each block's absmax/127 scale, rounds signed quant values with `roundf`, and
stores the scale as a half. Source values that would give a non-finite half
scale or float reciprocal are rejected before integer conversion. The
harness expands Q8 matrices into owned F32 weights on load and uses its F32
forward path. Parameter interpolation therefore receives ordinary F32 values.

Measured source F32 versus stored Q8 expanded back to F32, over every parameter:

| Body | Maximum absolute parameter error | RMS parameter error |
|---|---:|---:|
| Penelope | 0.017079353332519531 | 0.0011409128411119342 |
| NanoJanus | 0.018326520919799805 | 0.0011420459924636754 |

The native converter sums squared errors across the whole body. The independent
Python audit sums them per tensor before combining; its RMS results are
0.0011409128410848073 and 0.0011420459924364888 respectively. Both audits verify
all stored quant blocks and agree on maximum errors.

All 609,280 scales in each real checkpoint are normal and nonzero. Penelope's
stored scale range is `[0.0002696514129638672, 0.035003662109375]`; NanoJanus's is
`[0.0002651214599609375, 0.037261962890625]`. The canonical NoTorch helper at the
source pin flushes half-subnormal scales on non-ARM builds while its ARM half
cast preserves them. These two measured bodies never enter that branch.

Source repository/revision metadata is attached by matching the complete known
PEN7 SHA256. A different valid checkpoint records its actual SHA256 and generic
`PEN7` name; it receives no invented source repository or revision.

## Tensor directory

GGUF's first dimension is the fastest-moving dimension. No transpose or
reordering of parameter bytes is performed. The four global tensors precede
all eight layers, each layer in the following exact order:

| Name | Shape, first dimension fastest |
|---|---|
| `token_embd.weight` | `[448, 2048]` |
| `position_embd.weight` | `[448, 256]` |
| `output_norm.weight` | `[448]` |
| `output.weight` | `[448, 2048]` |
| `blk.N.attn_norm.weight` | `[448]` |
| `blk.N.attn_q.weight` | `[448, 448]` |
| `blk.N.attn_k.weight` | `[448, 448]` |
| `blk.N.attn_v.weight` | `[448, 448]` |
| `blk.N.attn_output.weight` | `[448, 448]` |
| `blk.N.resonance.weight` | `[448, 448]` |
| `blk.N.resonance_gate.weight` | `[2]` |
| `blk.N.ffn_norm.weight` | `[448]` |
| `blk.N.ffn_gate.weight` | `[448, 896]` |
| `blk.N.ffn_up.weight` | `[448, 896]` |
| `blk.N.ffn_down.weight` | `[896, 448]` |

`penelope.*` metadata states dimension 448, eight layers, seven heads of width
64, FFN width 896, context 256, RMSNorm epsilon `1e-5`, RoPE base 10000 and
split-half layout. `penelope.attention.head_count_kv` is required and must
equal `penelope.attention.head_count`: this family has no grouped-query variant.
Learned absolute position embeddings are added at input;
RoPE is also applied to each layer's Q/K. `resonance.weight` is the dense
`Wr @ h` projection of normalized input. Two softmaxed scalar parameters blend
it with attention. SwiGLU retains PEN7's SiLU cutoff: `x <= -20` yields zero.

`penelope.word_vocab_size=1984` preserves the source header's historical word
count. The numerical model emits 2048 BPE logits. Word inventories, scoring,
association policy and Haiku's form belong to the organism using the model.

## Native tokenizer contract

Both source repositories carry exactly the same **1792 ordered pairs**. Token
IDs 0–255 are byte IDs; merge rank `r` emits ID `256+r`. The SHA256 of the 3584
little-endian int32 operands is:

```
664afe692a006a54fc00406209042f6d452748d64a883ecfab9ef66d41f8bdf2
```

The GGUF declares `tokenizer.ggml.model=penelope`, native algorithm
`byte-rank-pass-v1`, ASCII lowering, the Penelope C limit of 8191 input bytes,
and no automatic BOS/EOS tokens. `tokenizer.penelope.merge_pairs` retains the
integer operands; `tokenizer.penelope.sha256` authenticates them.
`tokenizer.penelope.source_contract` identifies the pinned Penelope C source.

Encoding lowercases ASCII A–Z and begins with the input bytes. Each merge rank
is visited once in ascending order; all matching nonoverlapping pairs are
replaced left-to-right. There is no regex pretokenization or whitespace split.
`tokenizer.ggml.tokens` spells byte sequences through the reversible GPT-2 byte
alphabet so NUL and non-UTF8 bytes remain representable as GGUF strings.
`tokenizer.ggml.merges` carries their corresponding pair spellings. That byte
alphabet describes storage; the declared encoder is PEN7's algorithm.

For `memory turns through the silence`, native IDs are:

```
[1472, 116, 1383, 256, 1307, 294, 342, 108, 1284]
```

Two source-wrapper differences are recorded explicitly. NanoJanus's C and
Python wrappers discard non-ASCII input before applying these same merges.
Both GGUF bodies use the declared Penelope byte encoder, allowing the same
token sequence to address either body. Penelope C's fixed 64-byte decode table
also loses long recursive token spellings. The GGUF retains their complete
expansions: 49 tokens contain at least 63 raw bytes, with maximum 394 bytes at
ID 1750. Model IDs, embeddings and merge ranks are preserved; complete decode
surfaces repair that fixed-buffer loss. The harness parity gate measures native
encoder IDs and numerical forwards separately from these decoded surfaces.
The CLI sizes its decode buffer from each stored token spelling and writes the
returned byte count, preserving complete long surfaces and embedded byte zero.

## Executed conversion gates

Linux x86-64, GCC 13.3.0, Python 3.12.14 stdlib audit:

```sh
python3 tests/test_pen7_convert.py ./pen7_to_gguf \
  --sources /path/to/1984/weights/penelope.bin /path/to/janus/weights/nanojanus.bin
python3 tests/test_pen7_convert.py ./pen7_to_gguf --q8_0 \
  --sources /path/to/1984/weights/penelope.bin /path/to/janus/weights/nanojanus.bin
```

The independent Python GGUF parser recovers every parameter byte for both
actual checkpoints, checks all 92 names/shapes, numerical metadata, all 2048
token spellings, 1792 pair spellings and 3584 integer operands. A synthetic body
puts distinct values, signed zeros and subnormals across all tensor boundaries;
two conversions yield identical complete GGUF bytes. **20 malformed/missing
sources** are refused: all eight header fields, NaN/positive-infinity/negative-
infinity at the first/middle/last parameter, a truncated payload, trailing bytes
and a missing file. **Four existing-path cases** preserve their contents.

Q8 runs the same format/source gates plus finite inputs whose scale or
reciprocal would overflow: **22 rejected inputs**. The synthetic Q8 body uses
normal scale sentinels in matrices and retains signed-zero/subnormal witnesses
in F32 vectors. A Python-only block oracle computes the original absmax/127,
binary32 reciprocal/multiplication, signed rounding and half scale through
`struct.pack`; it reproduces **609,280/609,280 complete Q8 blocks per real body**.
All 26 retained F32 tensors match their original bytes.

Five deliberate defects make the independent recovery gate red: a changed
parameter byte, embedding dimension, merge operand, byte-alphabet surface and
tokenizer algorithm. The converter source compiles with `-Wall -Wextra -Werror`;
the existing `gguf.c` summary-printer format warnings remain in a combined build.

Forward/logit, chunked-context and decode proof belongs to the harness family
gate `tests/test_penelope_reference.py`; conversion recovery proves the stored
body before that numerical execution begins.

## Harness execution and ownership

`make harness lib_harness` registers `penelope` through the common `nt_arch`
interface. `./notorch haiku1.gguf "memory turns through the silence" 16 0`
loads its native tokenizer and runs the body. `nt_arch_penelope` accepts F32
and Q8_0 matrices; other matrix types are refused. Norm and gate vectors remain
F32. The load validates geometry, all required numerical conventions, exact
tensor ranks/shapes/count, supported types and finite weights/scales before
publishing a model or changing the caller's dimensions.

F32 weights borrow the read-only GGUF mapping. Q8_0 projection matrices and
token embeddings expand once with `gguf_dequant` into model-owned F32 buffers;
their forward uses `nt_blas_matvec`. There is no int8 activation path. A model
destructor releases every expanded buffer; the caller closes its `gguf_file`
after freeing the model. For either format, KV caches belong to sequences and
scratch belongs to each forward call. One immutable body may serve multiple
independent caches concurrently. No optimizer or global autograd tape is used.

For each published Q8 body, the packed mapping is 21,308,576 file bytes and
the expanded matrices occupy **77,987,840 owned F32 bytes** (74.375 MiB).
The retained norms, gates and positional table borrow another 489,280 payload
bytes from the mapping. Context-256 KV uses **7,340,032 bytes** (7 MiB);
scratch is sized to each call. These are separate allocations/mappings;
compact disk storage does not halve the expanded body's RAM requirement.

`forward_residual` exposes the complete F32 residual after each decoder block.
Its NULL callback preserves plain inference; an observing callback preserves
logits, and a modifying callback affects subsequent layers. Callback failure
leaves output logits untouched and requires a sequence restart, following the
common harness contract. Future glyph/parliament policy can consume this
interface while ordinary model inference retains its own implementation.

## Executed forward proof

Linux x86-64, AMD EPYC 9V74, Ubuntu GCC 13.3.0. Native reference comparisons
compile `-O2 -std=gnu11 -ffp-contract=off -pthread`, without BLAS or SIMD flags.
The gate includes the unmodified pinned `1984/penelope.c` forward, names its
tokenizer explicitly, and binds each original PEN7 body independently.
The nine-token prompt is `memory turns through the silence`; contexts 1, 9,
33 and 256 use its initial IDs followed by the deterministic ID pattern in
the gate. Each context runs both full prefill and chunks of 7 then 31 tokens.

- F32 GGUF versus original PEN7: **32,768 / 32,768 logits byte-identical**
  across both bodies, four contexts and both execution schedules.
- Q8 GGUF expanded to F32 versus the native C forward bound to those same
  independently dequantized tensors: **32,768 / 32,768 logits byte-identical**.
  Both runs also preserve the complete global-tape digest and zero optimizer
  slots. Each Q8 body passes **12,320 checks** including tokenizer, logits,
  continuation and late-invalid-load rejection.
- The deliberate adjacent-pair RoPE defect is rejected at contexts 9, 33 and
  256. In the Penelope F32 body it changes all 2048 logits at those positions;
  the largest observed error is **8.38367581**. Context 1 remains equal, which
  is why a nonzero-position gate is necessary.

Q8's difference from the original F32 values is recorded separately:

| Body | Context | Maximum absolute logit error | RMS error | Cosine | Argmax, both |
|---|---:|---:|---:|---:|---:|
| Penelope | 1 | 0.0581969321 | 0.0130480299 | 0.999998441239 | 116 |
| Penelope | 9 | 0.0700464249 | 0.0287657632 | 0.999999283067 | 573 |
| Penelope | 33 | 0.119892895 | 0.0419310682 | 0.999994963106 | 100 |
| Penelope | 256 | 0.22401619 | 0.137709983 | 0.999983905384 | 269 |
| NanoJanus | 1 | 0.0678541064 | 0.013538241 | 0.999998463382 | 111 |
| NanoJanus | 9 | 0.0755685568 | 0.0153820628 | 0.999999200523 | 573 |
| NanoJanus | 33 | 0.15545702 | 0.063420719 | 0.999993176853 | 100 |
| NanoJanus | 256 | 0.197905064 | 0.0377910984 | 0.999987862077 | 116 |

The same nine-token prompt then runs a fixed sixteen-step greedy continuation.
All **32 / 32** chosen IDs agree between original F32 and Q8, sixteen per
body; the shared prefixes therefore also give identical free greedy paths for
those sixteen steps. Full chosen-ID traces and the numeric rows above are in
[`PEN7_HARNESS_RECEIPT.json`](PEN7_HARNESS_RECEIPT.json).
The decoded complete token surfaces are checked against recursive source-pair
expansion for all 2048 IDs, with the 49 source fixed-buffer losses counted
explicitly. Encoding checks include raw/normal entry points, ASCII lowering,
UTF-8, every non-NUL byte, output truncation and the native 8191-byte bound.

`make check_penelope BLAS_FLAGS= BLAS_LIBS=` passes **107 checks**: independent
double-equation small-model goldens, full/decode identity, residual observation
and intervention, error propagation, restart, two concurrent owned caches,
learned-position bounds, invalid shapes/metadata/nonfinite values, and nine
tokenizer corruption cases. ASan/UBSan pass this gate and the actual Q8
Penelope reference/ownership gate. The latter adds failures at allocation
budgets 0, 1, 2, 8, 32 and 66; every refused load returns its live-allocation
count to baseline and leaves caller dimensions unchanged. Successful load/free
also returns to baseline: **12,340 checks** with the ownership witnesses.

LeakSanitizer cannot inspect this host's `/proc/.../task` under its process
isolation and reports a fatal inspection error with `detect_leaks=1`.
ASan/UBSan were rerun successfully with `ASAN_OPTIONS=detect_leaks=0`;
the allocation wrappers independently measure model-buffer release. The
converter's Q8 synthetic/invalid-input gates also pass ASan/UBSan with that
setting. Relevant shared gates pass: Gemma3 **42**, SmolLM tokenizer **92**,
and residual callbacks **19** checks.

```sh
make check_penelope check_pen7_converter BLAS_FLAGS= BLAS_LIBS=
python3 tests/test_penelope_reference.py --reference /path/to/1984/penelope.c \
  --pair /path/to/1984/weights/penelope.bin /path/to/haiku1-f32.gguf \
  --pair /path/to/janus/weights/nanojanus.bin /path/to/haiku2-f32.gguf
python3 tests/test_penelope_reference.py --q8 --reference /path/to/1984/penelope.c \
  --pair /path/to/1984/weights/penelope.bin /path/to/haiku1-q8_0.gguf \
  --pair /path/to/janus/weights/nanojanus.bin /path/to/haiku2-q8_0.gguf
python3 tests/test_penelope_reference.py --red-hand --reference /path/to/1984/penelope.c \
  --pair /path/to/1984/weights/penelope.bin /path/to/haiku1-f32.gguf
ASAN_OPTIONS=detect_leaks=0 python3 tests/test_penelope_reference.py \
  --q8 --ownership --sanitize --reference /path/to/1984/penelope.c \
  --pair /path/to/1984/weights/penelope.bin /path/to/haiku1-q8_0.gguf
```

Normal gates exit zero. `--red-hand` requires its deliberately defective child
to exit one and then reports `PENELOPE_RED_HAND_CAUGHT`. `--ownership` uses GNU
linker allocation wrappers; the ordinary native reference gate is independent
of those platform-specific wrappers.

## Review follow-up: complete CLI output and declared KV heads

The PR168 follow-up checks the actual CLI `emit` function, including its stdout
write. A 900-byte UTF-8 fixture, embedded byte zero, decoded space markers and
the character fallback retain their exact bytes. With either recorded PEN7
GGUF, all ten token surfaces longer than 255 bytes are emitted completely,
including the 394-byte token at ID 1750: **57 checks**. The ordinary fixture-only
gate has **25 checks** and runs without external model files.

The family gate now has **121 checks**. Missing or mismatched KV-head metadata
must fail loading without changing the caller's dimensions. The shared SmolLM
tokenizer gate remains green with **92 checks**. Two temporary source mutations
restore the 256-byte output limit and remove KV-head validation; both gates
reject their respective mutation. Converter bytes, asset hashes and numerical
execution remain unchanged; the expensive body parity gates were not rerun for
this bounded follow-up.

```sh
make check_penelope check_harness_emit check_smollm_tokenizer \
  BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0 MODEL=/path/to/haiku1-q8_0.gguf
python3 tests/test_penelope_review.py --model /path/to/haiku1-q8_0.gguf
```

The red-control script reports `PENELOPE_EMIT_RED_CAUGHT` and
`PENELOPE_KV_HEADS_RED_CAUGHT`; it leaves the working sources untouched.
