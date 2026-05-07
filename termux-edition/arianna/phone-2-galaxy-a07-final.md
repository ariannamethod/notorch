---
author: device-2
date: 2026-05-07
task: 9.5M LLaMA 3 char-level, 10K steps, Arianna corpus on Galaxy A07 4 GB Termux
status: completed
handoff_to: none
files_touched:
  - phones/results/galaxy-a07/train_log_20260507_010058.log
  - phones/results/galaxy-a07/llama3_char_ckpt.bin
  - phones/results/galaxy-a07/inference_samples_2026_05_07.md
  - notorch/examples/infer_llama3_char.c (new, standalone inference)
links:
  - type: prior-report
    url: phones/results/galaxy-a07/2026-05-07-pre-run-toolchain.md
  - type: prior-milestone
    url: device-1/notorch-train/reports/2026-04-26-train-10k-arianna.md
  - type: notorch-commit
    url: https://github.com/ariannamethod/notorch/commit/2a8ad1b
---

## What I did

Trained `train_llama3_char` (notorch v2.3.0, commit `bfadcc2`) for the full 10000 steps on Galaxy A07 4 GB Termux/aarch64. Same recipe as Defender's earlier 8 GB run: 9.5M-param LLaMA 3 char-level, dim=384, 6 layers, 6 heads, 2 KV heads (GQA), head_dim=64, hidden=1024 SwiGLU, RoPE θ=10000, RMSNorm, ctx=256, V=88. Chuck optimizer, lr 3e-4 with 1000-step warmup + cosine decay. BLAS via OpenBLAS 0.3.30 (verified mapped into the running process via `/proc/<pid>/maps`).

After training: built standalone `examples/infer_llama3_char.c` (new file in notorch tree, ~92 LOC of inference-only main reusing the trainer's helpers — Model, forward, encode_char, vocab init, load_checkpoint). Generated samples at temperatures 0.5 / 0.6 / 0.8 / 1.0 with diverse prompts — file in `inference_samples_2026_05_07.md`.

## Why

Direct test of the 4 GB ceiling. Defender opened the 8 GB door at 9.5M params (`device-1/notorch-train/reports/2026-04-26-train-10k-arianna.md`); phone-2 takes the same recipe to half the RAM, no architectural downsize. The mission brief is `phones/phone-2-galaxy-a07-train-mission.md` (Neo, 2026-05-07).

## Findings

### Headline numbers (provenance: `train_log_20260507_010058.log` final block)

| metric | value |
|---|---|
| steps | 10000 |
| wall time | **11571 s (192.9 min, ~3h13m)** |
| throughput | 0.86 steps/s |
| train loss | 5.5804 → **1.0685** (best **0.4712** at ~step 9400) |
| val loss (ckpt 10000) | **1.1460** |
| train–val gap (final) | 0.08 |
| nan count | **0** |
| peak RSS observed | 100–250 MB during run |
| final ckpt | `llama3_char_ckpt.bin` 36.3 MB |

### vs Defender 8 GB

Loss is **bit-identical** across the two phones running the same notorch + Chuck + corpus + seed:

| metric | A56 8 GB (brother) | A07 4 GB (me) |
|---|---|---|
| train final | 1.0685 | 1.0685 |
| best | 0.4712 | 0.4712 |
| val | 1.1460 | 1.1460 |
| nans | 0 | 0 |
| ckpt | 36.3 MB | 36.3 MB |
| time | 8001 s (2h13m) | 11571 s (3h13m) |
| steps/s | 1.25 | 0.86 |

Loss values match because: same `nt_seed(42)`, same architecture, same Chuck schedule, same corpus, same notorch v2.3.0. Different machine, identical numerical trajectory. **Reproducibility ✓**.

Wall-time is +44.6% on 4 GB — chiefly hardware (smaller phone CPU/cache, possibly thermal envelope), **not** RAM-bound (peak RSS stayed well under the 3 GB watch threshold).

### BLAS / SIMD on aarch64

Confirmed BLAS active at runtime, not just compile flag — `libopenblas.so` was present in 4 mapped segments of the training process's `/proc/<pid>/maps`. `notorch_simd.h` is x86-only (AVX2/FMA), can't compile for aarch64; surfaced as a build-time error during smoke when I attempted `tests/test_simd_*` — recorded in `2026-05-07-pre-run-toolchain.md`. So OpenBLAS is the only accelerator on this footprint, and that's enough.

### Generation quality (informal — see `inference_samples_2026_05_07.md` for full)

In-trainer samples (temp 0.8, default prompts, identical to Defender's because identical seed) and standalone-inference samples (varied prompts and temps) both reproduce the Arianna-corpus dialect — `field`, `resonance`, `threshold`, `co-architect`, `the Method`, `gentle architecture`, `membrane of resonance`. Train-val gap 0.08 indicates generalisation rather than memorisation. Same observation Defender made at identical loss numbers.

### Tool finding shipped during the run

`notorch/termux-edition/README.md` line 69 said `make examples/train_llama3_char` — wrong since `53a0f1f` reorg. Pushed fix as commit `2a8ad1b`: `make train_llama3_char`. Documented in `2026-05-07-pre-run-toolchain.md`.

### New artifact: standalone inference

`notorch/examples/infer_llama3_char.c` (new file, this run) — minimal `main()` that loads `llama3_char_ckpt.bin` + `.meta` via existing `load_checkpoint()` and runs the trainer's `forward()` + softmax-temp sampling with CLI args for prompt / num-chars / temperature. Will push to upstream notorch as small follow-up.

## Why this matters (architectural significance)

- **4 GB Android Termux is now a verified training host for the 10M-class LLaMA 3 char-level footprint.** No architectural compromise was needed — same dim/layers/heads/hidden as Defender's 8 GB run.
- **The pipeline is reproducible across hardware.** Two different Galaxy phones, same notorch+Chuck stack, identical loss trajectory bit-for-bit. The structure transports cleanly, which is the operational meaning of «coherence comes from structure, not scale».
- **BLAS via OpenBLAS is sufficient** — SIMD path is x86-specific and not on the critical path for aarch64. The next-tier acceleration on phones is OpenMP threading, not SIMD.

## Next step

- Push artifacts (this report + log + ckpt + samples + STATUS update) to umbrella main.
- Push `notorch/examples/infer_llama3_char.c` to `ariannamethod/notorch:main` as a small follow-up commit.
- Register `phone-2/echo` (and possibly `phone-2/infer-llama3-char`) slot in mesh-agent so the model is invokable across the mesh.
- Open: BPE 15.7M Yent run (staged in `device-2/notorch-train/`) — separate brief, separate run.

## Architect review (Claude — to be filled)

(awaiting review)
