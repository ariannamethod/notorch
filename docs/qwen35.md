# Qwen3.5 text and vision on NoTorch

The `qwen35` family implements the dense Qwen3.5 text decoder in C: full gated
GQA with Q/K normalization and partial NEOX RoPE, recurrent GatedDeltaNet,
causal depthwise convolution, zero-centered RMS weights and SwiGLU. Its cache
owns the recurrent matrices, convolution history and full-attention K/V slots.
A continuation must start at the next position; position zero resets the state.
The minimal CLI now honors the existing family-specific cache callbacks.

## Source layout and commands

This first path reads the explicit `qwen35.source_layout_version = 1` format
produced by the native converter below. BF16/F32 tensor payloads are copied
unchanged from `model.safetensors`, or from the shards
`model.safetensors.index.json` names when there is no such file: Qwen's own
0.8B and 2B ship one shard named `-00001-of-00001`, 4B two. The loader expands
matrix weights to F32 for CPU arithmetic; embedding rows remain mapped. Vision
and MTP tensors stay in the original checkpoint and are excluded from this text
pack. A GGUF written by llama.cpp also says `qwen35`; it is refused by name
(`no qwen35.source_layout_version`). Packed quantized text GGUFs, MoE and Metal remain separate integration work.
The native source-layout vision path is described below.

```sh
make qwen35_to_gguf qwen35_tokenize qwen35_eval notorch
./qwen35_to_gguf /path/to/snapshot model.gguf
./qwen35_to_gguf --tokenizer /path/to/snapshot/tokenizer.json tokenizer.gguf
./qwen35_tokenize tokenizer.gguf < prompts.jsonl > inputs.tsv
./qwen35_eval model.gguf --chunk 32 < inputs.tsv > answers.jsonl
NT_BLAS_THREADS=6 ./notorch --tokenizer tokenizer.gguf -n 64 -t 0 model.gguf "The capital of France is"
```

The model file carries no vocabulary, so the CLI takes it with
`--tokenizer`, also for `-T`; a tokenizer for another architecture, one with
more tokens than the model has rows, or a missing file is refused. Without
one the CLI refuses to read text as byte ids, which a 248,320-id model would
take as garbage; `NT_TOKENS` still passes ids straight to the forward. A run
ends on `<|im_end|>` or `<|endoftext|>`. Every matrix here goes through
`cblas_sgemm` in F32, so on an OpenBLAS build `NT_BLAS_THREADS=N` gives it N
threads; the default for every family stays one, and Accelerate threads
itself.

`prompts.jsonl` contains `{ "id": "example", "prompt": "..." }` rows. The
resulting TSV is `id<TAB>space-separated token IDs`. `qwen35_eval` returns the
A/B logits (IDs 32/33) and two-way softmax. `--trace existing-directory` writes
all layer residuals and final full-vocabulary logits as little-endian F32 for
numerical comparison. Trace IDs are restricted to letters, digits, `_` and `-`.
The converter creates outputs exclusively and prints their SHA-256.

`qwen35-ascii` implements the exact ASCII restriction of the checkpoint's
split regex. The source normalizer is NFC; ASCII is unchanged by NFC. Native
text encoding explicitly refuses non-ASCII bytes. The decoder accepts verified
token IDs throughout the vocabulary. `bpe_encode_special` explicitly opts into
literal CONTROL/USER_DEFINED tokens for complete chat prompts; ordinary
`bpe_encode` retains its previous CONTROL behavior.

`qwen35` is the official checkpoint's regex, which keeps marks inside words
(`\p{M}` from `examples/unicode_marks.h`). It is ported over codepoints in the
reference's leftmost-first order, and its `(?i)` folds `'ſ` to `'s` as the
reference does; llama.cpp's `[sS]` does not. The converter writes `qwen35` for
the official tokenizer.json and `qwen35-ascii`, byte for byte as before, for
the older regex; merges are read as pairs or as `"a b"` strings. NFC is a
UAX #15 quick check: invalid UTF-8, or a codepoint in `qwen35_refuse`
(`examples/unicode_qwen35.h`: NFC_QC No or Maybe, 108 marks the reference's
Unicode 9.0 ccc table orders as starters, 4,433 Unicode 16 codepoints its regex
reads as letters, marks or numbers), encodes to nothing. Any other text is NFC
after a stable sort of each run of non-starters by ccc. The check also refuses
some text NFC leaves alone, a Maybe mark that composes with nothing before it:
Russian stress marks (а + U+0301), the Bengali, Tamil and Malayalam vowel sign
AA. So Russian text written with stress marks, and most Bengali, gets no ids
at all: it is refused, not tokenized. Accepting it needs NFC's composition
step, which this tokenizer does not have. `tools/gen_unicode_tables.py`
regenerates both headers.

Against tokenizers 0.22.2 on the official tokenizer.json (sha256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
Qwen/Qwen3.5-0.8B `2fc06364715b967f1860aea9cf38778875588b17`),
`tests/qwen35_tokenizer_parity.py` ran the 211 rows of
`tests/qwen35_unicode_corpus.jsonl` and 300 seeded random rows: all 456
accepted rows had the reference's ids, all 44 refusals were text the
reference's NFC changes, and the 11 limit rows were refused. llama.cpp
e95426f differed from the reference on 95 of the 511 rows, each one text NFC
changes: its BPE path does not normalize. The official tokenizer GGUF is
`2f38049769a7fb0d2b70b5ed329563d932cc366899379f35476419021260a4f4`.

## Verification, 2026-11-06

The independent reference was Transformers 5.15.0, Torch 2.13.0 CPU, F32,
eager attention, four threads. The real checkpoint was
[`OmniJev/OneJev-0.8B`](https://huggingface.co/OmniJev/OneJev-0.8B/tree/c02e305b430b56c1e82f9675ba8c293a22646891),
revision `c02e305b430b56c1e82f9675ba8c293a22646891` (original weights).
The reference's generation and trace comparison ran outside this repository.
The converter, decoder, tokenizer and checked-in regression tests are C.

Fourteen English JOVOVICH review prompts were compared at every one of the
24 layer outputs for every input token, and at all 248,320 final logits.
The thresholds were recorded before the native run. Native arithmetic used
x86_64 Linux / AMD EPYC 9V74, OpenBLAS via `libcasadi-tp-openblas`, with
`OPENBLAS_NUM_THREADS=4 NT_ATTN_THREADS=4`.

| Measurement | Limit | Maximum observed |
| --- | ---: | ---: |
| Layer residual relative L2 | 2e-4 | 1.7839233768302967e-6 |
| Full-logit absolute difference | 2e-3 | 1.811981201171875e-5 |
| A/B probability absolute difference | 2e-4 | 1.3450441778140032e-6 |

All 14 choices and all 14 token-ID sequences matched the reference. The C
converter's model output matched the independently packed source file byte for
byte (321 text tensors). Detailed measured differences are in
[qwen35-parity.json](qwen35-parity.json).

| Artifact | SHA-256 |
| --- | --- |
| Source config.json | `09e987471d022500c939c71e06f86bf1239586e491546a8296d855689c022e2a` |
| Native model GGUF | `b587ecb42b7d022f782ea72c8b84e94b8292ae3f5aa67e1364851b4d4e32b7e6` |
| Native tokenizer GGUF | `9bad2a5c9b99fe69916a6c4bd09f96e385030585b136ee256067e054ec755b8e` |

`make test_qwen35` runs 1,214 checks against frozen independent F32 fixtures:
all-token residuals, logits, chunks of 1/3/5/7, reset, invalid continuations,
callback failure, error-output preservation and a missing convolution tensor.
The fixture uses three delta layers plus a full-attention layer, unequal key
and value head counts, convolution width four and partial RoPE. Disabling
recurrent decay in a temporary source copy caused **968 failures**; the normal
code passed. `make test_qwen35_text` passed 15 tokenizer checks. The full
`make -j4 test` suite passed, including existing SmolLM tokenizer regression
checks. Its optional original Haiku corpus was skipped because `SPM_MODEL`
was not supplied; the 106 embedded SentencePiece oracle cases passed.
An initial attempt stopped on a stale local test executable lacking
execute permission; rebuilding that executable resolved the interruption.

JOVOVICH also ran all 14 prompts through AML + native tokenization + chunk-32
NoTorch inference, preserving all choices and meeting the same 2e-4 probability
gate. The task score remains 12/14: the original index-boundary and disabled
writer misses remain present.

## Official checkpoints against llama.cpp, 2026-11-09

The stock checkpoints Qwen/Qwen3.5-0.8B (`2fc06364715b`), -2B (`15852e8c16360`)
and -4B (`851bf6e806efd8d`), every shard checked against its LFS SHA-256, were
converted here and, for the reference, by llama.cpp `7d7f0287530bc8981690f361d8c1627dbae1c99a`'s
own `convert_hf_to_gguf.py` to BF16. On 2026-11-09 that commit was still the
head of llama.cpp master, and a second run of its benchmark on the same
machine reproduced the first within 2.1%. Greedy, repeat penalty 1.0, through
`--tokenizer`:

- 32-token continuations of "The capital of France is", "1, 2, 3, 4," and
  "def fibonacci(n):" are llama.cpp's text in 9 of 9 cases on x86_64 (0.8B, 2B,
  4B) and 6 of 6 on an M4 Pro (0.8B, 2B).
- A chat prompt ends where llama.cpp's does, on `<|im_end|>`, with the same
  answer, on 0.8B and 2B.
- "Столица Франции это" and "בירת צרפת היא" encode to llama.cpp's ids. On 2B
  the continuations match. On 0.8B both differ from llama.cpp's BF16 run and
  both equal Transformers 5.8.0 in F32, id for id: the F32 path here sides with
  the F32 reference where BF16 arithmetic parts from it.

Throughput, 64-token prompt and 64 generated tokens, t/s, prefill / decode:

| Machine | Model | NoTorch | llama.cpp BF16, CPU |
| --- | --- | ---: | ---: |
| i5-8500T, 6 threads | 0.8B | 83.6 / 6.1 | 209.4 / 17.6 |
| | 2B | 44.7 / 2.4 | 95.4 / 7.8 |
| | 4B | 19.3 / 1.2 | 37.7 / 3.5 |
| i5-8500T, 1 thread | 0.8B | 41.6 / 2.3 | 47.3 / 8.3 |
| | 2B | 18.0 / 0.9 | 19.6 / 3.7 |
| | 4B | 7.5 / 0.4 | 7.6 / 1.5 |
| M4 Pro, Accelerate / 10 threads | 0.8B | ≈450 / 38.4 | 166.2 / 18.8 |
| | 2B | ≈248 / 19.6 | 122.1 / 7.6 |

NoTorch's six-thread x86 numbers are with `NT_BLAS_THREADS=6`, its one-thread
numbers the default. On the M4 Pro, llama.cpp's Metal GPU path reaches 2761.7 /
123.3 (0.8B) and 1413.0 / 59.3 (2B); NoTorch has no GPU path for this family.
On x86 decode runs at a third of llama.cpp's; the F32 weights it streams are
twice the bytes of BF16.

`make test_qwen35` now also runs a second frozen fixture,
`tests/qwen35_gqa_fixture.h` from `tests/qwen35_fixture_gen.py gqa`: two KV
heads, two key heads and two full-attention layers, 1124 checks. Reading the
wrong KV head, the wrong key head or every full layer at slot 0 fails 832, 1056
and 408 of them; the first fixture passes all three.

## Native still images, 2026-11-11

```sh
make qwen35_to_gguf qwen35_vision_probe test_qwen35_vision
./qwen35_to_gguf --vision /path/to/snapshot vision.gguf
./qwen35_vision_probe vision.gguf image.png 262144 image-check
```

The snapshot also needs its original `processor_config.json` (nested
`image_processor`) or `preprocessor_config.json`. The converter checks the
processor's patch, merge, temporal, normalization and resize settings. The
supported profile uses RGB, mean/std 0.5, bicubic resize, temporal patch size
2, spatial merge 2 and no DeepStack outputs. A still image duplicates its
frame in the temporal patch. The source `[out,3,2,patch,patch]` projection is
flattened to a matrix without changing any tensor bytes.

`harness/qwen35_vision.h` exposes image preparation, encoding, placeholder
validation and T/H/W positions. `nt_qwen35_forward_3d` uses physical positions
for cache state and explicit rotary positions for full attention. Each image
run is surrounded by the checkpoint's vision markers; the run contains exactly
one placeholder per merged feature row. Text following the image advances
from the maximum image coordinate. Decoder chunks retain these coordinates.
The shared RGB loader supports PNG, JPEG, BMP and PNM; the image encoder uses
PIL's 22-bit resampling coefficients. LFM3 keeps its existing torchvision
coefficient precision. Image attention works in 128-row tiles.

The generic embedding-row API continues to use ordinary 1D positions. Qwen
image inference uses the explicit 3D API. DOE's persisted media format currently
carries embedding rows and physical offsets; Qwen image replay awaits persisted
T/H/W positions. Existing Qwen text experts keep their current interface.

The probe emits prepared patches, merged features and geometry. Supplying
`TEXT.gguf IDS.i32 CHUNK STEPS` additionally records full logits and greedy IDs;
the IDs must already contain the expanded image placeholders. `IMAGE=-` selects
ordinary text. `tests/check_qwen35_vision.py` creates the independent inputs and
runs both paths, using Transformers **5.15.0**, PyTorch **2.9.1+cpu**, eager/F32
and the original OneJev revision listed above:

```sh
python tests/check_qwen35_vision.py --snapshot /path/to/snapshot \
  --weights model.gguf --vision vision.gguf --out /tmp/qwen35-vision-check
```

Measured on x86_64 Linux / AMD EPYC 7763, OpenBLAS via
`libcasadi-tp-openblas`, `OPENBLAS_NUM_THREADS=1 NT_ATTN_THREADS=2`:

| Check | Fixed gate | Observed maximum |
| --- | --- | --- |
| Prepared patches, five images | absolute 1e-6 | 0, identical bytes |
| Merged features | absolute 2e-4; relative L2 2e-5 | 1.55449e-4; 1.53579e-5 |
| Full logits, six runs × four vectors | absolute 2e-4; relative L2 2e-5 | 3.29018e-5; 2.72524e-6 |
| Three greedy tokens per run | identical IDs | 18/18 |

Inputs include solid red/blue, non-square high-frequency RGB, enlargement and
downsampling. The decoder checks image substitution, text-only input and
chunks 1/7/32. The tiny frozen C fixture additionally checks invalid inputs,
output preservation, restart and disconnected rows/removed spatial positions.

The first resize attempt differed by one RGB level because it used the
existing torchvision coefficient precision. PIL coefficients remove the
mismatch. Sequential F32 softmax accumulation then exceeded the feature gate
on the downsampled image (3.85046e-4 absolute); a double accumulator for the
normalizer brings it to 1.55449e-4. Gates stayed fixed throughout.

The original OneJev vision pack is **201,195,904 bytes**, 153 tensors, SHA-256
`6fe19a97634273c53ca7f8e7dc242ff92a016f11f3b2dc57eb60d78affc1a0be`.
Re-conversion preserves both text and tokenizer hashes listed above.
