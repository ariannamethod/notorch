# GPU backward segfault at T=32 + V=512 (separate from leak hypothesis)

**Status:** Reproducible on A100 SXM 80GB. Distinct from the GPU buffer leak (which closed `RESOLVED — NOT REPRODUCED, fix already in 3d46007`). This is a backward-pass crash in a specific size regime.

## Symptom

`nt_tape_backward(loss_idx)` segfaults (SIGSEGV / exit 139) when the LoRA training step uses **both** `T >= 32` **and** `V >= 512`. Forward pass completes (`lora_forward → lora_forward → cross_entropy_masked` all return valid indices); the crash happens after `loss` is recorded, during backward.

## Bisection (single training step, 2-layer LoRA harness)

| D | T | V | R | Forward | Backward |
|---|---|---|---|---|---|
| 64 | 16 | 128 | 4 | ✓ | ✓ |
| 64 | 32 | 128 | 4 | ✓ | ✓ |
| 64 | 16 | 512 | 4 | ✓ | ✓ |
| 64 | 32 | 128 | 8 | ✓ | ✓ |
| 64 | 16 | 512 | 8 | ✓ | ✓ |
| 128 | 16 | 128 | 4 | ✓ | ✓ |
| 128 | 32 | 128 | 4 | ✓ | ✓ |
| **64** | **32** | **512** | **4** | ✓ | **SIGSEGV** |
| **128** | **32** | **512** | **8** | ✓ | **SIGSEGV** |

Trigger reduces to **T ≥ 32 AND V ≥ 512 simultaneously**. Larger D and R only amplify; they do not gate the crash.

## Hypothesis

Almost certainly an out-of-bounds index or a missing-bounds-check in the cross-entropy backward kernel where `dlogits` shape is `[T, V]`. At T=32, V=512 the tensor is 16,384 floats — modest in bytes, but the access pattern likely hits a tile/block index that was sized for smaller V or T.

Suspect locations to inspect (notorch_cuda.cu):
- `gpu_seq_cross_entropy_*_backward` — softmax(logits) − one_hot(targets), distributed over [T, V]
- Any kernel with hardcoded `BLOCK_DIM` or `THREADS_PER_BLOCK` that assumes V ≤ 256 or T ≤ 16
- Backward through `nt_lora_forward` chain — the LoRA gradient routing may have an off-by-one when both shape axes scale

## Reproduction

1. Build `tools/leak_repro.c` (works, baseline). Edit constants to `D=64 T=32 V=512 R=4`. Rebuild. Run.
2. Crash inside `nt_tape_backward`. `gdb` / `cuda-memcheck` will pinpoint the kernel.

## Next step

Open as follow-up investigation. Run with `cuda-memcheck` to get a precise OOB report, then fix the offending kernel. Lower priority than the (resolved) leak, but blocks any LoRA training at the realistic shape Resonance/Janus actually use (V=16384/32768).

Until fixed, GPU LoRA training on Resonance/Janus through this path will crash. **Workaround:** PyTorch path (which we already used successfully for Run 1-4 today). The notorch GPU LoRA path is not yet ready for the production Resonance config.
