# Qwen3.5 dense text on NoTorch

The `qwen35` family implements the dense Qwen3.5 text decoder in C: full gated
GQA with Q/K normalization and partial NEOX RoPE, recurrent GatedDeltaNet,
causal depthwise convolution, zero-centered RMS weights and SwiGLU. Its cache
owns the recurrent matrices, convolution history and full-attention K/V slots.
A continuation must start at the next position; position zero resets the state.
The minimal CLI now honors the existing family-specific cache callbacks.

## Source layout and commands

This first path reads the explicit `qwen35.source_layout_version = 1` format
produced by the native converter below. BF16/F32 tensor payloads are copied
unchanged from a single `model.safetensors`. The loader expands matrix weights
to F32 for CPU arithmetic; embedding rows remain mapped. Vision tensors stay in
the original checkpoint and are excluded from this text pack. Packed quantized
GGUFs, MoE, Metal and the vision forward path are future work.

```sh
make qwen35_to_gguf qwen35_tokenize qwen35_eval
./qwen35_to_gguf /path/to/snapshot model.gguf
./qwen35_to_gguf --tokenizer /path/to/snapshot/tokenizer.json tokenizer.gguf
./qwen35_tokenize tokenizer.gguf < prompts.jsonl > inputs.tsv
./qwen35_eval model.gguf --chunk 32 < inputs.tsv > answers.jsonl
```

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
writer misses remain present. No throughput comparison has been run; memory
packing and a matched llama.cpp speed comparison are subsequent stages.
