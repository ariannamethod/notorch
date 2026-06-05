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

Speed is now **compute-bound, not memory-bound**: 0.2 t/s (~4.7 s/token),
dominated by the Q6_K per-row CPU dequant (output 131072×5120 + ~20 ffn_down)
and the Metal Q4_K Phase-1 per-call weight upload. Next lift: a Q6_K Metal matvec
+ resident weights (Phase-2), both in `notorch_metal.mm`.

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
