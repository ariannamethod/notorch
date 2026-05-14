# notorch — Termux Edition

**Pure-C neural network training on Android, via [Termux](https://termux.dev). No PyTorch, no CUDA, no datacenter, no root.**

This is not a fork. It is upstream `notorch` (`git clone https://github.com/ariannamethod/notorch`) plus the one-time setup needed to build and train inside Termux on an Android phone.

## Why Termux

Termux turns an Android device into a complete POSIX UNIX environment without rooting it:

- **APT package manager** — `pkg install` mirrors the Debian workflow. `git`, `make`, `clang`, `gdb`, `vim`, `openssh`, `python`, hundreds of others.
- **Native ARM64 toolchain** via `clang`. Pure-C projects compile unchanged from Linux — no cross-compilation, no NDK gymnastics.
- **OpenBLAS native on aarch64.** Standard `cblas_*` API, full vector-instruction performance. The scalar baseline on ARM Cortex is weaker than on x86, so BLAS pulls more — measured 8× on a 9.5 M LLaMA 3 forward+backward pass.
- **No root required.** Termux ships its own `$PREFIX` (`/data/data/com.termux/files/usr`) and never touches `/system`. Standard package, standard Play / F-Droid install.
- **Standard UNIX tooling end-to-end.** `make`, `pkg-config`, `nm`, `objcopy`, `time`, all behave as on Debian. Build scripts that work on a Linux workstation work here.

The interesting consequence: a sub-$500 phone running Termux is now a viable training host for the 10–100 M parameter regime. That covers small language models, persona LoRAs, narrow code-completion, micro-translators — work that has been wrongly assumed to require cloud GPUs.

## What's verified

- Galaxy A56, Android 15, ARM64 (aarch64-linux-android), 8 GB RAM
- libopenblas 0.3.30 via `pkg install libopenblas`
- 47/47 notorch tests pass
- 9.5 M LLaMA 3 char-level model, 10 000 steps, 2 h 13 m wall, train 5.58 → 1.07, val 1.94 → 1.15, train-val gap 0.08, 0 NaN
- Peak RSS 130–155 MB steady — phone never swapped
- ~8× speedup of OpenBLAS over scalar baseline on aarch64
- **15.7 M LLaMA 3 BPE (vocab 2048) on Yent v11 corpus, 5.4 MB / 1.95 M tokens, 15 000 steps, 4 h 42 m loop wall + ~18 min one-time encode, 1.10 s/step sustained, train 7.81 → 4.35 (best 2.90), val 4.94 → 3.93, 0 NaN, 60 MB ckpt** — first BPE LLaMA 3 trained end-to-end on Android Termux.

Reference runs with full logs:
- char-level 9.5 M: [`device-1/notorch-train/`](https://github.com/ariannamethod/ariannamethod/tree/main/device-1/notorch-train) in the umbrella repo.
- BPE 15.7 M: [`phones/results/arianna-method/2026-05-07-yent-bpe-15k-final.md`](https://github.com/ariannamethod/ariannamethod/blob/main/phones/results/arianna-method/2026-05-07-yent-bpe-15k-final.md).
- 4 GB peer (Galaxy A07) char-level, bit-identical loss to the 8 GB run thanks to `nt_seed(42)` determinism: [`phones/results/galaxy-a07/2026-05-07-10k-char-arianna-final.md`](https://github.com/ariannamethod/ariannamethod/blob/main/phones/results/galaxy-a07/2026-05-07-10k-char-arianna-final.md).

## Setup (one-time)

```bash
# Inside Termux
pkg install -y git build-essential binutils libopenblas pkg-config

# 1) ar / llvm-ar
#    Termux GNU binutils ships with a `g` prefix (gar, gnm, gobjcopy),
#    and `ar` itself is occupied by `arp` from net-tools. Symlink llvm-ar:
ln -sf "$(command -v llvm-ar)" "$PREFIX/bin/ar"

# 2) cblas.h / openblas_config.h
#    Termux installs them under $PREFIX/include/openblas/, not the
#    include root. notorch's Makefile asks pkg-config for openblas
#    flags, which resolves this automatically. If pkg-config is
#    missing, set BLAS_FLAGS manually:
#    export BLAS_FLAGS="-I$PREFIX/include/openblas -lopenblas"

# 3) Clone and build
git clone https://github.com/ariannamethod/notorch ~/notorch
cd ~/notorch
make            # CPU-only, scalar fallback
make BLAS=1     # CPU + OpenBLAS (recommended on aarch64)
./notorch_test  # expect 47/47 pass
```

If `make BLAS=1` fails on `cblas.h`, double-check pkg-config:

```bash
pkg-config --cflags openblas    # → -I$PREFIX/include/openblas
pkg-config --libs openblas      # → -L... -lopenblas
```

## Train your first model

`examples/train_llama3_char.c` is the reference 9.5 M LLaMA 3 char-level model, paired with a ~1 MB-class corpus:

```bash
cd ~/notorch
make train_llama3_char

# Drop a corpus next to the binary
cp /path/to/corpus.txt examples/corpus.txt

cd examples
./train_llama3_char 10000 0.0003 corpus.txt
# 10K steps, lr 3e-4 — Karpathy formula territory:
#   1.1 MB × 10M params × 10K iters → train ≤ 1.0 / val ≤ 1.5
# Logs every 100 steps, checkpoints every 1000 to llama3_char_ckpt.bin
# Resume after interruption:
#   ./train_llama3_char --resume 10000 0.0003 corpus.txt
```

Expected on a Galaxy 8 GB (aarch64, 1 core, BLAS):

- ~0.76 s/step → ~2 h for 10K steps
- 130–155 MB RSS steady-state
- 0 NaN with Chuck cosine decay

## Why Chuck (and not Adam) for this scale

`notorch` ships its own optimizer — Chuck — which suits micro-scale training without per-run β tuning:

- 9.5 M params, char-level, single-byte sequences
- 1.21 MB corpus → narrow batch distribution
- Long cosine decay over 10K steps with stable behaviour through the schedule

Use Chuck for anything in the same regime; the training scripts in `examples/` are wired for it.

## Inference

After `train_llama3_char` finishes the checkpoint is a portable binary blob. The matching inference helper is `examples/infer_llama.c` (or `infer_llama3_bpe.c` for BPE-trained variants):

```bash
make examples/infer_llama
./infer_llama llama3_char_ckpt.bin "the field is"
```

## Portability surface

Two upstream patches make Termux build cleanly out of the box (merged via [#5](https://github.com/ariannamethod/notorch/pull/5)):

1. `tests/test_notorch.c` honours `$TMPDIR` (was hardcoded `/tmp`, fails under Termux sandbox and hermetic CI runners more generally)
2. `Makefile` uses `AR ?= ar` and asks `pkg-config` for openblas flags

Known follow-ups, not yet patched:

- `tests/test_vision.c` has 16+ `/tmp/*.bmp` hardcodes
- A few `examples/` may want the same TMPDIR treatment if they write checkpoints

## Hardware envelope

Verified:

- 9.5 M params char-level LLaMA 3 — 2 h 13 m for 10K steps, 130–155 MB RSS, 8 GB phone

Likely viable (extrapolated from RSS scaling and Chuck stability):

- 1–3 M params on a 4 GB phone
- 30–50 M params overnight on 8 GB (~6–12 h)
- 80–100 M params over a couple of days with babysitting (RSS toward ~1.5 GB, may swap)

Out of scope on phone CPU:

- 1 B+ params (no realistic memory budget without aggressive quantization + sharding)
- Anything that would normally demand multi-GPU

## Status after Intel 2026-05-11 CPU-sync audit

Intel's `8ab5062` (NT_OP_MUL / NT_OP_SILU backward CPU-sync) and the parallel AML-side fix `ff7fb97` (16 ops) both gate behind `#ifdef USE_CUDA`. Termux builds CPU-only — `nt_tensor_sync_cpu()` is compiled out, zero runtime delta on aarch64. notorch `8e6658b` rebuilt clean on phone-1, `make notorch_test` → 47/47 pass post-update. README and recipe above unchanged.

For an AML system-wide install on the same Termux substrate, see the sibling [`ariannamethod.ai/termux-edition/`](https://github.com/ariannamethod/ariannamethod.ai/tree/main/termux-edition) — it carries the `pkg-config` Makefile patch that AML's upstream Makefile still needs for `make BLAS=1` on Termux.

## Files in this folder

- `README.md` — this document

The actual code, tests, and training scripts are in the standard notorch tree (`notorch.c`, `tests/`, `examples/`). This folder is the Termux narrative; the implementation is universal.
