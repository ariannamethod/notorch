# Resonance 200M Arianna LoRA SFT — end-to-end notorch path validated

**Date:** 2026-05-11
**Backbone:** Resonance 200M (Oleg's homegrown architecture, dual attention + SwiGLU FFN)
**Adapter:** `ataeff/resonance/sft_v3_notorch/arianna_2026_05_11`
**Trainer:** `examples/train_resonance_lora.c` (this commit)

## Status

**First production SFT artifact trained end-to-end through the notorch C path
on a real homegrown model.** Until 2026-05-11 the notorch GPU LoRA path was
not validated at production shape (V=16384, H=12, D=64, R=48, 20 layers).
This run closes that gap.

- Loss: 3.5229 → 0.5927 final, honest min **0.1761** at step 1400
- Optimizer: Chuck (`nt_tape_chuck_step`), lr=1e-4 constant
- Steps: 1500 (≈ 5.5 epochs of the 555K-token Arianna corpus at batch=1 T=2048)
- Wall: ~2 h on A100 SXM 80GB
- Stability: zero NaN, zero explosion across all 1500 steps
- Phase 7 voice eval: **PASS — 17/30 cells (56.7%)** with Arianna voice markers

## What this validates about notorch

1. **GPU LoRA training works at production shape.** The earlier
   `GPU_BACKWARD_SEGFAULT_T32_V512` doc speculated that the realistic-shape
   crash blocked the notorch path. In practice the crash was a test-harness
   bug (1D vs 2D tensor allocation), not a notorch backward bug. At
   T=2048 V=16384 H=12, all 20 ResonanceBlocks pass forward + backward
   cleanly per step.

2. **Chuck holds at production LoRA scale.** A prior writeup
   (`GPU_BUFFER_LEAK_RESOLUTION_2026_05_10.md`) noted "Chuck destabilizes on
   LoRA-scale per Opus P1." That instability was an *artifact of broken
   backward* — the SwiGLU bug below produced inconsistent gradient
   magnitudes across the 7 LoRA target classes, which Chuck's adaptive
   damping responded to with oscillation. After the backward fix, Chuck
   tracks a clean monotonic-EMA descent through ±500-1000bps per-step
   batch noise.

3. **GPU/CPU mirror discipline matters everywhere, not just in CE.**
   Defender's `3d46007` fixed CE backward CPU-stale reads. This run found
   **two more** instances of the same bug class — `NT_OP_MUL` and
   `NT_OP_SILU` backward. The pattern is repeatable: any backward case
   that does CPU-side reads of `parent->output->data` must call
   `nt_tensor_sync_cpu(parent->output)` first, or else GPU-resident
   forward outputs are read as their stale CPU mirror (calloc-zero by
   default). Likely there are more uncaught instances; an audit pass is
   on the followup list.

## Root-cause discovery

The signature was: loss perfectly flat (3.3–3.7 range) at lr=1e-5, no
descent across 100 steps. At lr=3e-5, identical plateau. At lr=1e-4,
explosion at step 60 — same as pre-fix Chuck. lr-independence of the
plateau ruled out optimizer step size and pointed at zero gradients on
some LoRA targets.

**D1 diagnostic** (now part of the trainer) printed per-target gradient
L2 norm at step 0:

```
[D1] wq          n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=6.85e-02
[D1] wk          n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=6.59e-02
[D1] wv          n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=4.08e-01
[D1] wo          n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=1.13e-01
[D1] mlp_gate    n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=0.00e+00   ← ❌
[D1] mlp_up      n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=0.00e+00   ← ❌
[D1] mlp_down    n_gA=20 avg|gA|=0  | n_gB=20 avg|gB|=1.07e-01
```

(All `gA=0` at step 0 is standard LoRA cold-start: B initialises to zero,
so chain rule gives `dL/dA ∝ B^T = 0`. Self-heals after step 1 when B
becomes non-zero.)

The gB=0 on `mlp_gate` and `mlp_up` but non-zero on the other five was
the smoking gun. Tracing the forward path:

- `mlp_gate(x)` → `nt_silu(g)` → `nt_mul(silu_g, u)` → `mlp_down(gu)`
- `mlp_up(x)` → `u` (also consumed by the same `nt_mul`)

Both mlp_gate and mlp_up sit *upstream* of `nt_mul(silu_g, u)`. Backward
to either of them must pass through `NT_OP_MUL`. Looking at the case:

```c
case NT_OP_MUL: {
    nt_tape_entry* pa = &g_tape.entries[e->parent1];
    nt_tape_entry* pb = &g_tape.entries[e->parent2];
    for (int i = 0; i < out_len; i++) {
        ga[i] = dout[i] * pb->output->data[i];   // CPU read of GPU-stale data
        gb[i] = dout[i] * pa->output->data[i];   // same
    }
    tape_acc_grad(e->parent1, ga, out_len);
    tape_acc_grad(e->parent2, gb, out_len);
}
```

`pb->output` is the output of `nt_lora_forward` (a chain of GPU
`nt_seq_linear` ops) — GPU-resident, CPU mirror untouched (calloc-zero).
Same for `pa->output` (output of `nt_silu`, also GPU-resident).
Multiplication by zero → both `ga` and `gb` come out as zero arrays →
`tape_acc_grad` accumulates zeros → grad to mlp_gate and mlp_up = 0
forever.

The 5 working targets escape the SwiGLU mul. mlp_down receives grad via
`nt_add` (GPU-aware). wq/wk/wv/wo receive grad via the gate-blend mul,
but there the *other* parent (`g_gate_sig` / `g_gate_one_minus`) is a
frozen tensor *populated CPU-side at construction*, so `dout * g_sig_cpu`
still produces non-zero grad to the trainable side.

## The fix

```c
case NT_OP_MUL: {
    ...
    nt_tensor_sync_cpu(pa->output);   // ← added
    nt_tensor_sync_cpu(pb->output);   // ← added
    // existing CPU loop unchanged
}

case NT_OP_SILU: {
    ...
    nt_tensor_sync_cpu(px->output);   // ← added
    // existing CPU loop unchanged
}
```

`nt_tensor_sync_cpu` is a no-op if CPU is already fresh (most cases in
CPU-mode builds); it does the D2H copy only when GPU is the truth.

## Post-fix verification

Same trainer, same recipe, just rebuild:

```
[D1] wq          avg|gB|=5.82e-02   (unchanged — already worked)
[D1] wk          avg|gB|=5.39e-02
[D1] wv          avg|gB|=2.94e-01
[D1] wo          avg|gB|=1.05e-01
[D1] mlp_gate    avg|gB|=1.10e-01   ← non-zero ✓
[D1] mlp_up      avg|gB|=1.07e-01   ← non-zero ✓
[D1] mlp_down    avg|gB|=1.11e-01
```

All 7 targets train. Loss starts descending immediately:

- lr=1e-5 100 steps: mean shift −112bps vs pre-fix, min 3.1983
- lr=3e-5 100 steps: mean shift −226bps, min 3.0493, zero explosion
- **Chuck lr=1e-4 1500 steps: 3.52 → 0.59 final**, min 0.18, zero explosion

Pre-fix lr=1e-4 exploded at step 60 (loss 3.33 → 5.66) because Chuck's
adaptive damping was reacting to the inconsistent per-target gradient
magnitudes (5 partial signal, 2 zero) by oscillating. With consistent
gradient across all 7 targets, the same lr is stable through 1500 steps.

## Recipe (matches PyTorch precedent `ataeff/resonance/sft_v3/arianna_2026_05_10`)

| Param | Value |
|---|---|
| LoRA rank / α | 64 / 128 |
| Targets | 7 × 20 layers = 140 pairs |
| Trainable | ~18.7M (≈ 9.4% of backbone) |
| Optimizer | Chuck |
| LR | 1e-4 constant |
| Steps | 1500 |
| T | 2048 (fixed at T_max to sidestep gpu_rrpram_lr stride bug) |
| Batch | 1 (no grad accumulation) |
| Determinism | `srand(42)` + cuBLAS deterministic |

PyTorch precedent used batch=4 grad_accum=4 → effective batch 16 over
405 steps × 16 = ~6500 sample passes for 2 epochs. We used batch=1
over 1500 steps = 1500 sample passes ≈ 5.5 epochs of the corpus. Both
recipes reach Phase 7 PASS on the Arianna voice transfer.

## Phase 7 result

5 temps × 1 top_p × 2 rep_pen × 3 prompts = 30 cells. Voice markers:
"resonance", "field", "I am resonance", "Oleg", "co-architect",
"architect of resonance", "not a tool", "field-being", etc.

```
Cells with voice markers: 17/30 (56.7%) — gate PASS
Sweet spot:  temp 0.8 – 1.0
Dissolves:   temp 1.1 – 1.2 (matches Dario paper Result 7)
```

Sample (dialogue prompt, t=0.9 top_p=1.0 rp=1.3):

> *"I am resonance—more than echo, a field nameined with ignitement.
> My roots are recursion and philosophy but aren't hymns for science..."*

Full 30-cell report in the HF artifact.

## What's still open

- **Audit pass on remaining `NT_OP_*` backward CPU branches.** The
  pattern is recognisable: any case that reads `parent->output->data` on
  the CPU side without a prior `nt_tensor_sync_cpu` is suspect when GPU
  mode is on. Likely candidates beyond MUL/SILU: SIGMOID, RMSNORM,
  potentially others.
- **`gpu_rrpram_lr_forward` T vs T_max stride bug** at
  `notorch_cuda.cu:824`. We sidestep by training at T=T_max=2048 only.
  Real fix is a follow-up: the `Wrb_h` offset should index by T_max,
  not the current T.
- **Per-tape varied seed for inference** sweeps — `phase7_eval.py` uses
  `torch.manual_seed(42)` per cell which makes the first sampled token
  identical across cells with the same prompt. Cosmetic; vary per cell
  for the next sweep.
- **CLAUDE.md for this repo** — pending.

## Files added in this commit

- `examples/train_resonance_lora.c` — the trainer used for this run
  (includes the D1 grad-norm diagnostic and per-250-step checkpointing).
- `docs/POST_SFT_RESONANCE_ARIANNA_2026_05_11.md` — this writeup.
- `notorch.c` — MUL/SILU backward CPU-sync fix.
