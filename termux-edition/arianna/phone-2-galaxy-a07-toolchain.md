---
author: device-2
date: 2026-05-07
task: notorch v2.3.0 + metaharmonix rebuild on galaxy-a07 (4 GB Termux), pre-flight before 10K char-level run
status: in-progress
handoff_to: none
files_touched:
  - $PREFIX/lib/libnotorch.a
  - $PREFIX/include/ariannamethod/{notorch,gguf}.h
  - ~/notorch/train_llama3_char
  - ~/.claude/hooks/state/train-ack-20260507.flag
  - phones/results/galaxy-a07/2026-05-07-pre-run-toolchain.md
links: []
---

## What I did

Pulled notorch (`d2ffbf2..bfadcc2`, 3 new commits — CUDA backend, tests reorg, README update) and metaharmonix (vendored notorch resync). Rebuilt both clean on aarch64-android with `make BLAS=1` (OpenBLAS 0.3.30).

Built `train_llama3_char` for the 10 K char-level mission specified in `phones/phone-2-galaxy-a07-train-mission.md`. Verified dataset on disk. Touched the training-gate ack flag. Trainer is ready; awaiting Oleg's "go" before the actual run.

## Why

Per `phones/phone-2-galaxy-a07-train-mission.md` (Neo, 2026-05-07) the mission is reproducing Defender's 9.5 M LLaMA 3 char-level baseline on half the RAM. Brief satisfies the 6-point training-gate. This pre-flight report records the toolchain state so the upcoming run report has a clean dependency snapshot to refer back to.

## Findings / Open questions

### notorch core — clean

- `notorch_test` → **47/47 PASS** (5 compile warnings, all on unused symbols `tape_ensure`, `g_alloc_bytes`, `now_ms` + 2 in new code — cosmetic, not findings).
- New separate test `tests/test_rrpram_lr` → **PASS** (`max_rel_diff = 6.06e-02, fails = 0`, gradcheck on rrpram low-rank LR).
- `libnotorch.a` rebuilt and reinstalled to `$PREFIX/lib/`; headers under `$PREFIX/include/ariannamethod/`.

### Two upstream notorch findings (non-blocking)

The two new SIMD tests don't compile on aarch64-android Termux clang 21:

1. `tests/test_simd_correctness.c` — uses `CBLAS_TRANSPOSE`, `CblasNoTrans` and `aligned_alloc(...)` but does not include `<cblas.h>` nor `<stdlib.h>`. Compilation fails with "unknown type name 'CBLAS_TRANSPOSE'" and "unknown identifier 'CblasNoTrans'".
2. `tests/test_simd_loss.c` — uses `aligned_alloc(...)` without `<stdlib.h>`. Compiler note: *"include the header `<stdlib.h>` or explicitly provide a declaration for 'aligned_alloc'"*.

Both are missing-include bugs in newly-added test sources; runtime is unaffected (47/47 + rrpram_lr pass). Fix is two-line: add `#include <cblas.h>` + `#include <stdlib.h>` to `test_simd_correctness.c`, add `#include <stdlib.h>` to `test_simd_loss.c`. Holding off on the patch until after the proof-of-concept training run lands — they're not on the run path.

### One docs finding (this one I will fix)

`notorch/termux-edition/README.md` line 69 reads `make examples/train_llama3_char`. After the `53a0f1f` "tests/+bench/: organize root .c into proper dirs" reorg, the canonical Makefile target is `make train_llama3_char` (bare, no `examples/` prefix). Confirmed by reading `Makefile:131-132` — target builds bare `train_llama3_char` linking `notorch.c` directly. Old form yields linker errors (undefined `nt_seed`, `nt_tensor_new2d`, `nt_tensor_xavier`, `nt_tensor_new`, `nt_tensor_fill`, `nt_load`, `nt_tensor_free`) because the implicit `examples/%` rule doesn't pull in `notorch.c`. Will submit a small docs PR against `notorch/termux-edition/README.md`.

### metaharmonix — clean

`mhx` rebuilds clean against the resynced vendored notorch + AML + dario heart. No regressions. `aml --version` still works from inside the REPL.

### Trainer arch sanity

`train_llama3_char` startup print on `--help`:

```
notorch — LLaMA 3 char-level training
dim=384 L=6 H=6 KV=2 HD=64 FFN=1024 CTX=256 V=88
GQA ratio: 3 Q heads per KV head
RoPE theta=10000
```

Matches `phones/phone-2-galaxy-a07-train-mission.md` 6-point brief item 4 verbatim.

### Dataset

`~/arianna-datasets/arianna/arianna_dataset_final_clean.txt` — 1 211 564 bytes (~1.2 MB), present on disk (delivered earlier by Neo's rsync, timestamp Apr 22 02:54). Matches brief item 2.

## Next step

Await Oleg's "go". On signal, run:

```bash
cd ~/notorch
./train_llama3_char 10000 0.0003 \
    ~/arianna-datasets/arianna/arianna_dataset_final_clean.txt 2>&1 \
  | tee train_log_$(date +%Y%m%d_%H%M%S).log
```

Monitor RSS in a parallel shell:

```bash
watch -n 30 'free -h | head -2; ps -o rss,cmd -p $(pgrep train_llama3_char)'
```

Stop and report (no silent arch shrink) if RSS approaches 3 GB. Train loss first in reports (`memory/feedback_show_train_loss.md`).

After completion: log + summary land in `phones/results/galaxy-a07/`, ckpt local + warm rsync to polygon.

## Architect review (Claude — to be filled)

(awaiting review)
