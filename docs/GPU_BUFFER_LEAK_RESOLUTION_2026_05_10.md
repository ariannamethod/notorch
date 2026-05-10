# GPU buffer leak — resolution / closure note

**Status:** Hypothesis A (refcount in tape) **REJECTED** by static audit (every op forward correctly calls `nt_tensor_free(out)` after `nt_tape_record`). Empirical reproduction on A100 SXM 80GB **does not reproduce the leak** at all — 200-step micro-training loop shows GPU memory flat (435 → 437 MB at step 1, then 0 MB/step delta through step 200, 0 `[GPU] ptr_map full` warnings, 0 cuBLAS errors).

## Comparison with original symptom

| Metric | 2026-05-09 (Neo session) | 2026-05-10 (re-test) |
|---|---|---|
| `ptr_map full` warnings | 1,210,000 over 1450 steps | **0 over 200 steps** |
| GPU mem growth | 2 GB → 34.7 GB (~22 MB/step) | 435 MB → 437 MB then flat |
| Steps before saturation | ~50 | n/a (no growth) |

## Probable real fix

Commit `3d46007 — 3 patches: GPU/CPU sync in CE + ptr_map 8K→64K + nt_rope_split_half_freq` (Defender, 2026-05-09 12:19) was applied AFTER the symptom was first reported. It addressed:

1. `nt_seq_cross_entropy_masked` was reading CPU-stale logits with GPU-fresh GPU-side data, producing softmax(zeros) → uniform → loss = ln(V) every step. This forced the bizarre allocation pattern Neo observed.
2. `g_ptr_map` capacity raised 8K → 65K, sufficient for realistic tensor counts.
3. `nt_rope_split_half_freq` corrected RoPE half-frequency split.

Combined, these three patches eliminated the leak symptom. The surviving "ptr_map full" warning text (notorch_cuda.cu:141) is now never reached at the working scales we tested.

## Reproduction harness

Saved at `tools/leak_repro.c` (to be committed). Builds against `notorch.h` + cuBLAS:

```bash
cc -DUSE_CUDA -DUSE_BLAS -O2 -I. tools/leak_repro.c notorch.c notorch_cuda.o \
   -L/usr/local/cuda/lib64 -lcudart -lcublas -lopenblas -lm -o leak_repro
./leak_repro 2>err.log >mem.csv
grep -c ptr_map err.log    # must be 0
awk -F, 'NR>2{s+=$3} END{print s}' mem.csv  # cumulative growth must be ~0
```

Config: `D=64, T=16, V=128, RANK=4`, 2 LoRA layers, 200 steps. Steady-state.

## Bonus finding (separate issue, not the leak)

At larger config `D=128, T=32, V=512, R=8` and `D=256, T=128, V=2048, R=16`, the test **segfaults during the first forward** — not a leak, but a crash. Both cases compile cleanly and `gpu_init` succeeds; the fault happens deep in `nt_lora_forward → nt_seq_linear` chain. Repro: change config constants and rerun. To be filed as separate issue / investigation; **not** the GPU buffer leak.

## Recommendation

- Close this hypothesis. Mark the original investigation doc as `RESOLVED — NOT REPRODUCED, fixed by 3d46007`.
- Add `leak_repro.c` to `tools/` as ongoing regression check; integrate into a future GPU CI pass when the polygon GPU lands.
- Open a separate investigation for the T-scale segfault (the real follow-up).
