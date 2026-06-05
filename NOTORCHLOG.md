# NOTORCHLOG

The running engineering log of notorch. Every fix, every verified change,
every bug-class closed — dated, with commit and proof. The README is the
spec and the manifesto; **this is the work**.

Convention: small fixes (bug fixes, sync-discipline corrections, single-op
work, doc/docstring touch-ups) are recorded **here**. Large changes (a new
backend, a new op family, a new training method, an architecture shift) get
a section in the README too. When in doubt: it goes here first.

Newest entries on top.

---

## 2026-06-06 — nt_qmatvec: agnostic packed quantized CPU matvec (Q4_0/Q5_0/Q8_0/Q4_K/Q6_K)

The CPU/BLAS/SIMD inference path dequantized every GGUF tensor to dense f32 (×6-8 RAM) before
`cblas_sgemv` — only the Apple-Metal path (`nt_metal_q4k_matvec`) and a single example-local
`q6k_rows` inside `examples/infer_gguf_metal.c` kept weights packed. notorch now has a library
primitive, `nt_qmatvec(out, Wq, dtype, x, m, k)` (`notorch.c`, decl `notorch.h`), that keeps the
weights packed in RAM and dequantizes each block inline in registers — the same math as
`gguf_dequant → nt_blas_matvec`, a fraction of the memory and weight bandwidth. It dispatches by
GGUF dtype over the full common quant set: Q4_0, Q5_0, Q8_0 (block-of-32), Q4_K, Q6_K
(super-block-256); the Q6_K kernel is the proven `q6k_rows` lifted out of the example into the
library. **Verified** by a new `tests/test_qmatvec` against the dequant→cblas oracle: all five
formats agree to relative error ~1e-6 (f32 summation-order noise, not unpack error); `notorch_test`
stays 47/47. This is the foundation of an agnostic packed CPU inference path — the CPU no longer
has to blow Q4_0/Q8_0 up to f32. Phase 1 is single-threaded and correctness-first: the RAM win lands
when a runner stops calling `gguf_dequant` and rides `nt_qmatvec` directly, and the speed path
(pthread rows + MNN/llama.cpp-style int8 activation-quant with SDOT/VNNI integer dot) is next.
Branch `feat/nt-qmatvec-packed`, commits `8687137` / `5bc1b84` / `59901df`.

## 2026-06-06 — JS edition: full GGUF RUN (tokenizer + forward + generate), matches C

After the dequant-load landed, `js-edition/infer_gguf.mjs` runs a GGUF end-to-end in pure
JS: a byte-level BPE built **from the GGUF** (mirror of `examples/bpe.c`) + the llama/mistral
forward on notorch.js tape ops (embed / RMSNorm / q-k-v / interleaved-RoPE / GQA-attn /
SwiGLU FFN / tied output) + greedy generate. **Verified vs the C engine:** SmolLM2-135M-Q4_K_M
greedy produces *"The capital of France is Paris. Paris is a city"* — **token-for-token
identical** to `examples/infer_gguf_metal`. The JS edition now loads AND runs real quantized
models with no Python and no llama.cpp. CPU path today; packed/WebGPU quant matvec and the
qwen3 NEOX + per-head q/k-norm arch are the next steps.

## 2026-06-06 — JS edition: GGUF quantized dequant + C-parity test

`js-edition/notorch.js` `loadGGUF` threw on every quantized tensor (F16/F32 only) while the
JS README claimed "F16 + F32 dequant" — a prophetic debt. Ported the five GGML block-dequant
routines from `gguf.c` **byte-for-byte** (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K) into `loadGGUF`; a
real quantized GGUF now loads in browser/Node. **Verified** against the C path with a new test
— `tests/gguf_dequant_ref.c` dumps C `gguf_dequant` values, `js-edition/test_gguf_dequant.mjs`
compares: Q4_K/Q6_K/Q8_0/Q4_0 match C to **~5e-9** across Qwen3-0.6B, smallcoder-Q8_0,
wtf360-Q4_0 → `JS_DEQUANT_OK`. Q5_0 is mirrored from `gguf.c` but had no local Q5_0 file to run
against. Added `js-edition/package.json` (`type:module`) so Node imports the ESM. JS README
corrected to the true state. Open next: a packed / WebGPU quant matvec so big models don't
expand to f32 in-browser.

## 2026-06-05 — README rework: inference is first-class; models split refs vs organisms

The README sold notorch as a training framework; it is training AND inference. Added
an `## inference` section — the packed-Q4_K/Q6_K Metal path (`examples/infer_gguf_metal.c`,
new `make infer_gguf_metal` target, Darwin + non-Darwin guard), the engine matrix, and the
measured oyent-24B numbers (Mistral-Small-24B Q4_K_M on a 24 GB Mac: 0 swap, 10.6 GB,
~1.4 tok/s). Made Apple-Silicon/Metal consistent across the build matrix, dependencies, and
the platform table (it used to appear, then vanish). `what is this` now says trains **and** runs.

Restructured the model list into exactly two sections — **references** (Karpathy ports +
from-scratch notorch models + how-to-train, with the Resonance-200M 3.52→0.59 and
nanollama-88.6M proofs) and **organisms that run on notorch** (appendix). Removed neovlm
(now private) and janus.sonar (too experimental); microgpt-1bit relabeled honestly as the
pure-Python BitNet reference notorch's BitLinear was validated against (not a notorch build);
added nanollama-notorch + siblings. JS README's "F16+F32 dequant" line corrected — `loadGGUF`
throws on quant today; the block-dequant port is the open JS upgrade.

## 2026-06-05 — in-house SIMD (AVX2) matmul: kernel + cache-block pass

A measurement-driven optimization pass on `notorch_simd.h` (the zero-dependency
AVX2 cblas shim), benchmarked against Intel MKL + OpenBLAS on the i5-8500T
(6c no-SMT, perf governor, 7-run medians). Correctness held bit-identical
throughout (`test_simd_loss` = 10.379384 vs the OpenBLAS path).

- **MR-interleaved A packing** (`42eef01`) — the 6×16 micro-kernel read A
  strided by k (6 cache lines per k-step); pack A `[Kc][MR]` so the 6 values
  for one k-step are contiguous. +~20% on NN-forward.
- **4× k-unroll + aligned B loads** (`8b98a6c`) — hoist the per-iteration
  prefetch branch, `_mm256_load_ps` (B_pack is 64-byte aligned). TN
  weight-grad shapes reached MKL parity (Llama dWffn 321 vs MKL 329 GFLOP/s).
- **Re-block Kc=128/Nc=256** (`1db4bf8`) — the Kc=256/Nc=1024 B-panel (1MB)
  spilled to shared L3, so 6 cores contended L3 bandwidth; Kc=128/Nc=256 keeps
  the ~128KB B-panel in private L2. +5–12% on NN-forward at 6T. `#ifndef`
  guards make MC/KC/NC `-D`-overridable per target.

**Honest result:** single-thread the kernel is ~0.82× MKL; TN weight-grad is
at MKL parity; NN-forward stays ~0.5× MKL. The residual gap is multi-core
cache-residency (MKL scales 4×/6c, this 2×/6c) — disproved as kernel, B-pack
(shared-B trial reverted), or malloc (persistent-buffer trial reverted); it is
shared-L3 bandwidth, the deepest machine-specific part of a tuned BLAS. Not
claiming MKL parity on forward GEMM.

## 2026-06-05 — packed-Q4_K + packed-Q6_K GGUF inference on Apple Metal

New `examples/infer_gguf_metal.c` — end-to-end notorch-C inference that keeps
quantized weights **packed** and never materializes the full f32 tensor:
- Q4_K → `nt_metal_q4k_matvec` (Metal, `53f38f2`).
- Q6_K → new CPU per-row dequant matvec (mirrors `gguf.c:dequant_q6_k`), no f32
  buffer. This is what lets a 24B model fit a 24 GB Mac.
- byte-level BPE (`examples/bpe.{c,h}`) reads the tokenizer from the GGUF via new
  `gguf_read_str_array` (gguf.c — `gguf_open` skips array-typed KVs).
- one forward, two RoPE conventions auto-detected: llama/mistral interleaved
  (weights pre-permuted by convert) and qwen2/qwen3 NEOX + per-head q/k-norm.

**Why packed-Q6_K matters — measured on metal (Mac Mini M4 Pro, 24 GB), oyent
(Mistral-Small-24B) Q4_K_M, greedy, `/usr/bin/time -l`:**
- first cut, Q6_K→f32 at load: RSS 7.4 GB + **12.4 GB swap**, load 58.5 s — thrashes.
- packed Q6_K (this pass): **swaps=0**, peak RSS 16.3 GB / footprint 17.3 GB,
  load 3.63 s, coherent+correct → "The capital of France is Paris, and its
  administrative center is the".

Speed is now **compute-bound, not memory-bound**. First the Q6_K per-row CPU
dequant (output 131072×5120 + ~20 ffn_down) dominated at 0.2 t/s; threading that
matvec across cores (work-gated, 12 cores on M4 Pro, disjoint y rows) lifted
oyent-24B to **0.6 t/s** (decode 8 tok 13.2 s, total 66 s → 28.5 s, swaps still 0,
peak 17.3 GB, same correct output). Then the **Metal Q4_K Phase-1 per-call weight
upload** (240 dispatches/token) dominated.

**Phase-2 (resident weights) landed.** `gguf.c` now page-aligns the tensor block
(`posix_memalign`) and records `data_size`; `nt_metal_register_base` wraps it as
zero-copy `newBufferWithBytesNoCopy` MTLBuffer(s) — **segmented**, because one
buffer is capped at `device.maxBufferLength` (14.302 GB on M4 Pro, just under the
14.326 GB block); `nt_metal_q4k_matvec` binds each weight by offset, no per-call
upload (weights straddling a segment edge fall back to upload). Result on oyent-24B:
**0.6 → 1.4 t/s** (0.2 → 1.4 over the whole pass, ~7×), total 28.5 s → 14.4 s,
**RSS 16.3 → 10.6 GB** (zero-copy, weights not duplicated), swaps 0, same correct
output. Llama-3.2-3B on neo (A18 Pro): **0.1 → 1.2 t/s** (~12×). Remaining lift:
optional Q6_K Metal matvec + a tiled/simdgroup Q4_K kernel.

Correctness regression (neo): Qwen3-0.6B-Q4_K greedy still "...Paris..." after the
Q6_K-path change (it uses Q6_K tensors); Llama-3.2-3B-Q4_K greedy 5/5 capitals.

## 2026-06-03 — GPU launch-bound pass: host-sync storm killed

A CUDA-backend performance pass — the bottleneck was launch/sync overhead,
not FLOPs. Six commits (`c1b655a..eaae961`):
- **L1** (`38d6b1a`) — batch per-param grad-norm readback into one D2H
  transfer instead of one sync per parameter; kills the host-sync storm.
- **L2** (`bc02d83`) — wire GPU backward for `NT_OP_MUL` + `NT_OP_SILU`,
  removing mid-backward device→host stalls (those ops now backward on GPU
  instead of bouncing to CPU).
- **L5** (`66f3c0f`) — widen the single-thread softmax / cross-entropy
  kernels to block-parallel.
- **op-33 RRPRAM** (`c1b655a`) — collapse the per-head GEMM loop into a
  cuBLAS strided-batched call.
- (`976d088`) — forward-declare the batched helpers used by the forward
  kernel.

Merged in `eaae961`. `notorch.c` + `notorch_cuda.cu` only; CPU path unchanged.

## 2026-06-02 — sigmoid / scale-by-t GPU sync (CPU-mirror bug class)

`nt_sigmoid` + `nt_scale_by_t` forward & `NT_OP_SCALE_BY_T` backward
joined the GPU/CPU mirror discipline. Surfaced by the molequla Inc2
RRPRAM-gate review: a learnable sigmoid gate sat frozen at sigmoid(0) on
GPU because the CPU backward branch read the stale CPU mirror without
`nt_tensor_sync_cpu(parent->output)`. Fixed forward + backward. With this,
the `NT_OP_*` backward CPU-branch audit for the sync pattern is **complete
— no known remaining candidates**.

## 2026-05-14 — nanollama 89M post-SFT (Arianna)

See `docs/POST_SFT_NANOLLAMA_ARIANNA_2026_05_14.md`.

## 2026-05-11 — Arianna LoRA SFT through notorch + MUL/SILU backward fix

`8ab5062` — `NT_OP_MUL` / `NT_OP_SILU` backward CPU-sync. Proved Chuck
holds at production scale once backward is correct; earlier "Chuck
destabilizes on LoRA scale" notes were downstream of this backward bug.
First production SFT (Resonance 200M Arianna LoRA) landed clean. See
`docs/POST_SFT_RESONANCE_ARIANNA_2026_05_11.md`.

## 2026-05-10 — GPU buffer-leak thread closed

The `ptr_map full — buffer leak` warning was a symptom of upstream tape
ref-accounting at high tensor counts, not a real leak. `3d46007` raised
`GPU_PTR_MAP_SIZE` 8K → 64K and fixed the CE sync; the warning hasn't
reappeared at realistic scales. Full thread:
`docs/GPU_BUFFER_LEAK_HYPOTHESIS_2026_05_10.md` →
`docs/GPU_BUFFER_LEAK_RESOLUTION_2026_05_10.md`. Also see
`docs/GPU_BACKWARD_SEGFAULT_T32_V512_2026_05_10.md`.

## 2026-05-09 — first GPU/CPU mirror bug found and fixed

`3d46007` — `nt_seq_cross_entropy_masked` (Defender). Established the
load-bearing rule: any CPU backward branch reading `parent->output->data`
directly must `nt_tensor_sync_cpu(parent->output)` first when GPU mode can
be on, or it reads the calloc-zero CPU mirror and computes zeros. The
bug-class registry lives in CLAUDE.md «Bug patterns».

---

## Open (carried from CLAUDE.md TODO)

- `gpu_rrpram_lr_forward` `Wrb_h` stride uses current T instead of T_max
  (`notorch_cuda.cu:824`). Workaround: train at T = T_max only. Real fix
  pending.
- `notorch.h:653` alpha-format docstring is stale — `nt_lora_save` writes
  raw IEEE-754 `float32` bytes, not `alpha*1000`. Fix on next pass-through.
- `nt_rrpram_broadcast_attention` (`NT_OP_RRPRAM_BCAST` 34) declared in
  `notorch.h:126,442` but unimplemented in `notorch.c`. JS edition stops
  parity at op 33 awaiting it.
- `phase7_eval.py` — vary RNG seed per cell so the first sampled token
  isn't identical across same-prompt cells.
