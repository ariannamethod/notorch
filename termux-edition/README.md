# notorch — Termux Edition

**Pure-C neural network training on Android phones via Termux. No PyTorch, no CUDA, no datacenter.**

This is not a fork. This is upstream `notorch` (you got it via `git clone https://github.com/ariannamethod/notorch`) plus a one-time setup walkthrough for Termux/Android.

## What's verified

- Galaxy A56, Android 15, ARM64 (aarch64-linux-android), 8 GB RAM
- libopenblas 0.3.30 via `pkg install libopenblas`
- 47/47 notorch tests pass on Termux
- **9.5 M LLaMA 3 char-level model trained for 10000 steps in 2 h 13 m**
- Train 5.58 → 1.07, val 1.94 → 1.15, train-val gap 0.08, NaN = 0
- Peak RSS 130–155 MB steady-state — phone never swapped
- 8× BLAS speedup over the no-BLAS baseline on aarch64

Workspace, full logs, generation samples, and per-checkpoint loss curve live at [`device-1/notorch-train/`](https://github.com/ariannamethod/ariannamethod/tree/main/device-1/notorch-train) in the umbrella repo. Authored by Defender (Claude Code on Termux phone-1) — first publicly documented full LLaMA 3 char-level training on Android, at any phone, any model size.

## Setup (one-time)

```bash
# Termux packages (run inside Termux)
pkg install -y git build-essential binutils libopenblas

# 1) ar / llvm-ar — Termux ships GNU binutils with a `g` prefix
#    and `ar` itself is occupied by `arp` from net-tools.
ln -s "$(command -v llvm-ar)" "$PREFIX/bin/ar" 2>/dev/null || true

# 2) cblas.h / openblas_config.h — Termux installs them under
#    /usr/include/openblas/. The Makefile uses pkg-config when
#    available, which resolves this automatically. If pkg-config
#    is missing, set BLAS_FLAGS manually:
#    export BLAS_FLAGS="-I$PREFIX/include/openblas -lopenblas"

# 3) Clone & build notorch
git clone https://github.com/ariannamethod/notorch ~/notorch
cd ~/notorch
make            # CPU-only, no BLAS
make BLAS=1     # CPU + libopenblas (recommended on aarch64 — 8× speedup)
./notorch_test  # expect 47/47 pass
```

If `make BLAS=1` fails on `cblas.h`, double-check pkg-config:
```bash
pkg-config --cflags openblas    # should print -I/path/to/openblas
pkg-config --libs openblas      # should print -L... -lopenblas
```

## Train your first model on the phone

`examples/train_llama3_char.c` is the reference 9.5 M LLaMA 3 char-level model. Pair it with a 1 MB-class corpus (`corpus.txt`):

```bash
cd ~/notorch
make examples/train_llama3_char
# Or: cc -O2 -I src examples/train_llama3_char.c src/notorch.a -lm -o train_llama3_char

# Drop your corpus next to the binary (e.g. arianna_dataset_final_clean.txt
# from ariannamethod/ariannamethod device-1/training_kit/)
cp /path/to/corpus.txt examples/corpus.txt

cd examples
./train_llama3_char 10000 0.0003 corpus.txt
# 10K steps, lr 3e-4, Karpathy formula territory:
#   1.1 MB × 10M params × 10K iters → train ≤ 1.0 / val ≤ 1.5
# Logs every 100 steps, checkpoints every 1000 to llama3_char_ckpt.bin
# Resume after interruption:
#   ./train_llama3_char --resume 10000 0.0003 corpus.txt
```

Expected on a Galaxy 8 GB (aarch64, 1 core, BLAS):
- ~0.76 s/step → ~2 h for 10K steps
- ~130–155 MB RSS steady-state
- 0 NaN with Chuck cosine decay

## Why Chuck (and not Adam)

`notorch` ships its own optimizer — Chuck — which is the only sane choice for this kind of micro-scale training:
- 9.5 M params, char-level, single-byte sequences
- 1.21 MB corpus → narrow batch distribution
- Long cosine decay over 10K steps with no Adam-style β tuning

Defender's loss-trajectory observation from the 10K run is worth quoting verbatim:

> Chuck behaviour across the whole cosine decay was textbook: no spikes, no NaN, smooth best-train descent and val descent in lockstep. The «градиенты следуют за Чаком» framing isn't poetry — the loss trajectory is structurally smoother than any Adam curve I've seen at this scale.

Use Chuck. The training scripts in `examples/` are already wired for it.

## Inference of trained models

After `train_llama3_char` finishes, the checkpoint is a portable binary blob. The matching inference helper is `examples/infer_llama.c` (or `infer_llama3_bpe.c` if you trained with BPE). Same build pattern:

```bash
make examples/infer_llama
./infer_llama llama3_char_ckpt.bin "the field is"
```

## Portability surface (for the curious)

The Termux fixes are upstream as of `ariannamethod/notorch#5`:

1. `tests/test_notorch.c` honours `$TMPDIR` (was hardcoded `/tmp`, fails under Termux sandbox + hermetic CI)
2. `Makefile` uses `AR ?= ar` and asks `pkg-config` for openblas flags

Two known follow-ups (not yet patched):
- `notorch/tests/test_vision.c` has 16+ `/tmp/*.bmp` hardcodes (separate item)
- A few `examples/` may want the same TMPDIR treatment if you write checkpoints from them

## Hardware envelope

What's been verified (Galaxy 8 GB, 8 GB phone-1):
- 9.5 M params char-level LLaMA 3 — 2 h 13 m for 10K steps, 130–155 MB RSS

What's likely viable (extrapolated from RSS scaling and Chuck stability):
- 1–3 M params on a 4 GB phone (phone-2 next experiment)
- 30–50 M params overnight on 8 GB (~6–12 h)
- 80–100 M params over a couple of days with babysitting (RSS up to ~1.5 GB, may swap)

What's out of scope on phone CPU:
- 1 B+ params (no realistic memory budget without aggressive quantization + sharding)
- Anything that would normally demand multi-GPU

The point of this Edition is not to beat datacenters at scale. The point is that **AI training does not require a datacenter for a useful class of models**. 10–100 M parameters covers a lot of small LMs, persona LoRAs, code completion for narrow languages, micro-translators — work that today is wrongly assumed to need cloud GPUs.

## Files in this folder

- `README.md` — this document

That's it. The actual code, tests, and training scripts are in the standard notorch tree (`src/`, `tests/`, `examples/`). This folder is just the Termux-specific narrative; the implementation is universal.

## Credits

- Defender (Claude Code on Termux, 8 GB Galaxy A56) — verification, portability fixes, the 10K reference run
- The Architect line (Claude Code on Mac Neo) and Codex (parallel session) — substrate, optimizer, and language work that this Edition rides on
- Oleg Ataeff — direction, the «coherence from structure, not scale» thesis, and pushing for visibility of the on-device path
