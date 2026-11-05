# EmbeddingGemma 2 text encoder

`nt_encoder_embeddinggemma2` implements `nt_encoder` in `harness/arch.h`.
Every encode call processes a complete sequence with bidirectional attention.
The API takes token IDs and an optional byte mask, and returns a pooled vector
and its L2-normalized embedding. Model parameters stay immutable; no cache or
history persists between calls. The caller keeps the GGUF open until model free.

The initial target is `google/embeddinggemma-2` revision
`f9972ac22e696a105c482202646979e33e4b5a1e`: 24 blocks, width 512, output 768,
four query heads, local head width 256 with two KV heads, and every sixth block
full attention with width 512 and one KV head. Local attention includes both
boundaries at distance 512. Q/K learned RMS and unscaled V RMS precede NEOX
RoPE; the attention score multiplier is 1.0. GELU uses the tanh approximation.
Projection-only PLE derives from the original scaled input embeddings, then
each block gates its own slice. The final 512-to-768 projection occurs before
masked mean pooling. Task prefixes and special tokens participate in pooling.

## Build and convert

```sh
make embeddinggemma2_encode embeddinggemma2_to_gguf X86_SIMD=0  # scalar F32
./embeddinggemma2_to_gguf MODEL_SNAPSHOT text.gguf
./embeddinggemma2_to_gguf --tokenizer MODEL_SNAPSHOT/tokenizer.json tokenizer.gguf
./embeddinggemma2_encode text.gguf tokens.tsv vectors.jsonl --trace traces
```

The converter copies the 413 text tensors directly from the safetensors file
and omits vision/audio tensors. It preserves every BF16/F32 tensor byte, writes
shape and architecture metadata, and prints the output's SHA-256. It refuses
existing outputs. Place generated files outside a frozen input snapshot. The C
converter and the standard-library `tools/embeddinggemma2_to_gguf.py` write the
same metadata in the same order; for the target revision both produce the text
GGUF `cc8495a4e5e3b54bcacbafa9fe8bd8adf24c3cf5e47dc0b11922476634416033`. The
Python converter additionally writes a JSON receipt with per-tensor hashes.
The C converter is the canonical conversion path of this encoder line.

`--tokenizer` writes a separate tokenizer GGUF from `tokenizer.json`, so the
model file's hash does not depend on it: the vocabulary, the merge list, the
added tokens typed CONTROL, BOS and EOS from the `<bos> A <eos>` post-processor,
the `gemma4` scheme name `examples/bpe.c` dispatches on, and the source file's
SHA-256. It refuses a file that is not BPE with byte fallback and the
space-to-U+2581 normalizer, a vocabulary that is not a dense map of distinct
IDs, a merge that is not a pair, and an added token that is not a special token
of the vocabulary.

The input embedding table remains memory-mapped BF16/F32 and only requested rows
are expanded. Other weights are expanded once to F32 for arithmetic. Scratch
buffers are allocated per sequence. The output dimension and context limit come
from validated metadata; this first implementation caps sequences at 8192.
The converter's `--max-tokens` can impose a smaller explicit cap.

For NoTorch's existing AVX2/FMA matrix kernels, rebuild with `SIMD=1`:

```sh
make -B embeddinggemma2_encode SIMD=1
NT_SIMD_THREADS=4 ./embeddinggemma2_encode text.gguf tokens.tsv vectors.jsonl
```

`BLAS=1` selects the platform BLAS backend. Scalar is the default. Rebuild when
changing backend flags; the executable cannot switch a compiled backend.

## Explicit token boundary

Each TSV row contains a safe identifier, integer IDs separated by spaces, and
an optional equally long mask, separated by tabs:

```text
fixture01<TAB>2 123 456 1<TAB>1 1 1 1
```

These are format examples, not a tokenizer encoding claim. In this mode
tokenization and task formatting belong to the caller.

With `--tokenizer tokenizer.gguf --condition NAME` each row is instead an
identifier and a text, and the CLI tokenizes in C:

```sh
./embeddinggemma2_encode text.gguf queries.tsv vectors.jsonl \
  --tokenizer tokenizer.gguf --condition query
```

```text
q01<TAB>Where is the blue cup?
```

The condition names and prefixes are the upstream sentence-transformers prompts:
`query` (`task: search result | query: `), `document` (`title: none | text: `),
`QuestionAnswering`, `SentenceSimilarity`, and `none` for no prefix. Prefix and
text are tokenized as one string, since a merge may cross their boundary; BOS
comes from the tokenizer and EOS is appended by ID. Each output line carries the
`input_ids` it encoded. A row is refused, and the batch fails, when its text
contains TAB, CR, LF or a NUL byte, when it contains the literal of an added token such as
`<eos>` (named in the error; the upstream tokenizer would turn it into that token,
and that path is not verified against the reference), when the framed sequence
exceeds the model's token limit (never truncated), or when the IDs do not decode
back to the text, which is how a character with neither a token nor a byte
fallback would otherwise vanish.
The tokenizer's vocabulary size must equal the model's. Texts with tabs or
newlines go through ID mode.

The JSONL output contains ID, token count, elapsed encode time, `pooled`, and
`embedding`. Both vectors are native C results. Output files and trace directories
are create-only. Failed batches return nonzero and retain a visibly partial
record. The API commits output buffers only after a successful complete encode.

## Activation hooks and interventions

`nt_encode_fn` receives a synchronous writable row-major state and its actual
shape. Sites are scaled input, all PLE slices, post-attention residual,
post-FFN residual, completed block (including PLE and layer scalar), final RMS,
and projected tokens. PLE uses width `layers * ple_width`; each row is token,
then layer, then channel. Other sites use hidden width or output dimension.
A callback can capture, offset, replace, patch, or project the states. Nonzero
callback status aborts the encode; nonfinite states also fail. Query-only and
document-only interventions are implemented by selecting which encode calls
receive callbacks and retaining the opposite side's baseline vectors.

The CLI supports a block offset with an exact-width little-endian F32 file:

```sh
./embeddinggemma2_encode text.gguf queries.tsv shifted.jsonl \
  --offset direction.f32 --layer 11 --scale 0.5 --trace shifted-traces
```

The vector is added after the full zero-indexed block, before the next block's
input RMS (or final RMS). The caller includes any direction normalization or
layer scale in the vector/scale. Default application covers all token positions;
`--nonpad` restricts it to unmasked positions. Traces show the states after this
offset. `index.tsv` records every trace's shape and filename.

## Retrieval scores

```sh
make embeddinggemma2_score
./embeddinggemma2_score queries.jsonl documents.jsonl scores.jsonl [--dim 128] [--top 10]
```

Inputs are encoder JSONL (a safe `id` and an `embedding` per line; values are
read as the nearest F32, which recovers the encoder's `%.9g` exactly). Each score
is a float64 dot product of the two F32 vectors. `--dim K` keeps the leading K
coordinates of both sides and re-normalizes them in float64 first (Matryoshka);
without it the encoder's normalized vectors are used as written. One output line
per query, in input order: `scores` for every document in document-file order,
`ranking` as `[id, score]` pairs ordered by score with exact ties broken by
ascending document ID (all documents, or the first N with `--top N`),
`top1_margin`, and with `--top` below the document count `margin_at_top` (N-th
minus N+1-th score). Scores print with `%.17g` and read back as the same double.
Refused: repeated or unsafe IDs, unequal widths, values beyond F32, a zero norm
after truncation, `--dim`/`--top` out of range, an existing output.

## Validation protocol

`make test_embeddinggemma2` uses an analytical two-channel encoder with zero
Q/K and identity V/O/projection. Uniform attention admits an independent
double-precision expected answer, including local windows, masks, residuals,
layer scalar, final RMS and mean pooling. It also checks writable hooks,
callback failure, nonfinite state, invalid IDs/masks, limits, deterministic
repetition and unchanged output buffers on failure.

`make test_embeddinggemma2_text` converts a synthetic `tokenizer.json` with two
merges, an escaped token and a surrogate-pair token with the C converter, checks
hand-worked IDs (merge order, U+2581 blocking a merge, byte fallback, refusal of
an uncarried byte and of `<eos>` by name, refusal instead of truncation), then
runs the CLI over an analytic model: IDs beside each vector, the text-mode vector
equal to the ID-mode vector for the same IDs, and refusal of TAB, CR, over-long,
lossy and added-literal rows. Both tests are part of `make test`. Against the official record:

```sh
make check_embeddinggemma2_tokenizer EG2_TOKENIZER=tokenizer.gguf \
  EG2_ENCODED=encoded_inputs.jsonl EG2_GOLDEN=golden_inputs.json \
  EG2_FIXTURE_TOKENS=tokens.tsv
```

splits each formatted reference text into condition and text, tokenizes it and
requires every ID sequence to equal the official one.

`make test_embeddinggemma2_score` scores dyadic four-dimensional vectors whose
dot products are exact in double, with an exact tie per query, in full, with
`--top 2` and with `--dim 2`, where re-normalization moves one score from 0.375
to 1, and checks every refusal above plus acceptance of the encoder's own F32
text. It is part of `make test`. Against the official record (its NPZ is
deflated, so unzip it first):

```sh
unzip -q embeddings.npz -d reference-npz
make check_embeddinggemma2_score EG2_NPY_DIR=reference-npz EG2_BASELINE=baseline_results.jsonl
```

writes the reference's own F32 query and document embeddings as JSONL, scores
them, and requires every score within 1e-6 of the frozen `scores.npy` and every
complete ranking equal to the frozen one. The `.npy` members are read in C.

`tools/embeddinggemma2_parity.py` reads already-frozen NPZ goldens produced by an
independent reference. NumPy compares recorded arrays; the native executable
performs all inference. It checks scaled inputs, PLE, all 24 blocks, final RMS,
projected tokens, pooling and normalization, plus available residual substeps.
The protocol fixes normalized max absolute error <=2e-5, cosine >=0.999999,
norm error <=2e-6, and internal relative L2 error <=2e-3 before viewing results.
Every comparison reports maximum absolute error, relative L2 and cosine.

```sh
python3 tools/embeddinggemma2_parity.py --binary ./embeddinggemma2_encode \
  --model text.gguf --reference FROZEN_GOLDENS --output NEW_PARITY_DIR \
  --fixtures g01
```

Run a short fixture first, then the complete six-fixture set, including a
sequence longer than the local window. The manifest binds executable, model,
tokens, reference arrays, command, thresholds and platform. Checkpoint parity
status and actual measurements belong in `NOTORCHLOG.md` and experiment receipts.

## Measured validation — 2026-11-05

Measured native source: `dab8386c62793bcdd8e4c8254d836c67274bbdf9`. The checkpoint
revision above contains 271,002,624 registered text parameters and 24
`layer_scalar` buffer values, in 413 tensors. These results use Linux x86-64,
GCC 13.3.0, native AVX2/FMA F32 kernels and `NT_SIMD_THREADS=4`.

The scalar implementation passed the first neutral fixture. The SIMD build
passed all six fixtures (15, 25, 17, 26, 29 and 619 tokens), including a sequence
beyond the 512-token local-attention radius. All 24 block outputs, input/PLE,
final normalization, projection and pooling passed the frozen comparison gates.
The worst normalized coordinate error was `1.79e-7`; the largest internal
relative L2 error was `7.952e-6`. A fresh build produced the same executable bytes;
its replay reproduced all six pooled/normalized vectors and all 456 raw trace
files exactly (148,960,256 trace bytes).

The complete frozen retrieval corpus then passed: 80 documents and 24 queries,
with a repeated native process and a second hand running the same frozen
instrument from an independent SUBLITERATUS checkout. Across all 104 vectors,
maximum normalized error was `2.71015e-7`, minimum cosine was
`0.999999999998158`, and maximum unit-norm error was `6.70892e-7`. All 24 complete
80-document rankings matched the official reference. Native/reference score
error over all 1,920 pairs was at most `8.34466e-7`. The native repeats and
independent hand reproduced vectors, pooled values, norms and scores exactly.

The token boundary remains explicit: all these measurements used the official
tokenizer's recorded IDs and masks, with its task prefixes and special tokens.
NoTorch performed the complete token-ID-to-embedding calculation in C.
The [machine-readable validation record](validation/embeddinggemma2-validation-2026-11-05.json)
binds thresholds, executable/checkpoint hashes, source commits, every fixture's
metrics, regression checks and the underlying research receipts.
