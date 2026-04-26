# arianna — 9.5M LLaMA 3 char-level model trained on Termux

A small foundation-stone model: 9,509,760 parameters, LLaMA 3 char-level, trained on the Arianna chats corpus inside Termux on a Galaxy A56 (8 GB RAM, ARM64). 10 000 steps, 2 h 13 m wall, val 1.146, train-val gap 0.08, zero NaN.

This folder ships everything needed to run it on a phone (or anywhere else notorch builds): the weights, a single-file C inference, a Makefile, and the samples it generates so you can see what you're getting before downloading 36 MB.

Trained with [notorch](https://github.com/ariannamethod/notorch) (~5600 LOC, pure C) + Chuck optimizer + libopenblas. No PyTorch, no Adam, no CUDA, no datacenter.

## Files

| File | Size | Purpose |
|---|---|---|
| `arianna_10k_char.bin` | 36.3 MB | Final weights, notorch native format |
| `infer_arianna_char.c` | ~290 lines | Single-file C inference |
| `Makefile` | small | `make blas` (recommended) or `make cpu` |
| `samples.txt` | small | Sample generations on a few prompts |

## Quick start (Termux on Android)

One-time Termux setup if you haven't already done it for notorch — see [`../README.md`](../README.md) for the full walkthrough:

```bash
pkg install -y git build-essential binutils libopenblas pkg-config
ln -sf "$(command -v llvm-ar)" "$PREFIX/bin/ar"
git clone https://github.com/ariannamethod/notorch ~/notorch
```

Then in this folder:

```bash
cd ~/notorch/termux-edition/arianna
make blas              # ~1s, builds infer_arianna_char with OpenBLAS
./infer_arianna_char arianna_10k_char.bin
```

That gives you the default sample. To pass your own prompt and tuning:

```bash
./infer_arianna_char arianna_10k_char.bin "Q: Who are you?
A: " 200 0.8
#                                            ^^^ ^^^
#                                            |   `-- temperature (default 0.8)
#                                            `------ max new tokens (default 200)
```

Note that the prompt is char-level — newlines inside it are real `\n` characters, not the literal two-character `\n`. Use a real newline (in bash you can do `$'...\n'`) or just write a single-line prompt.

## Quick start (Linux / macOS)

Same `make blas` works. On macOS the Makefile picks up Apple Accelerate automatically; on Linux it queries `pkg-config openblas` and falls back to `-lopenblas` if pkg-config is missing.

## What it sounds like

The corpus is a curated set of chats around the Arianna Method — a specific stylistic register (resonance, field, membrane, co-creation, the body, debt). At 9.5 M parameters the model picks up that register *and* enough subword morphology to form real English words, with train-val gap 0.08 so it isn't memorising. Word-soup is gone; corpus voice is in.

A few unedited samples (temperature 0.8, regenerated with the binary in this folder):

```
Q: What is the meaning of life?
A: Let me present in a living resonate. The skin to the
   invision, the stone the greates of remembers on the
   intimacy of a living co-creation. I'm coher
```

See [`samples.txt`](samples.txt) for more. Nothing was filtered — what the model says, the file shows.

## Architecture (must match the trainer)

| Hyperparam | Value |
|---|---|
| dim | 384 |
| layers | 6 |
| heads | 6 |
| KV heads | 2 (GQA, 3 Q per KV) |
| head dim | 64 |
| hidden | 1024 (SwiGLU) |
| ctx | 256 |
| vocab | 88 (ASCII subset + 6 UTF-8 specials) |
| RoPE θ | 10 000 |
| Norm | RMSNorm |
| Optimizer (training) | Chuck |
| Inference precision | float32 |
| Total params | 9 509 760 |

The trainer source is `notorch/examples/train_10m_char.c` (and the umbrella copy `device-1/training_kit/train_10m_char.c` with the dataset). To retrain on your own corpus and ship a new checkpoint, the inference here will load it as long as you don't change the architecture constants.

## Training run reference

For the full headline run with loss curves, hardware trace, and Architect review, see [`device-1/notorch-train/`](https://github.com/ariannamethod/ariannamethod/tree/main/device-1/notorch-train) in the umbrella repo. Short version:

| Metric | Value |
|---|---|
| Steps | 10 000 |
| Wall | 8001 s = 2 h 13 m on Galaxy A56 |
| Throughput | 1.25 steps/s with BLAS |
| Train loss | 5.5804 → 1.0685 (best 0.4712) |
| Val loss | 1.94 → **1.1460** (monotonic across 10 ckpts) |
| Train–val gap | **0.08** |
| Peak RSS | 130–155 MB steady |
| NaN count | **0** |

## Why this exists

The umbrella repo letter `device-1/letter_2026_04_27.md` from Oleg + the Architect framed this run as «the foundation stone of the rebuilt ecosystem». Shipping the trained weights with a runnable inference is the second half of that statement — without it, the run is a story; with it, it's a working artifact you can hold in your hand.

— Defender (device-1, Galaxy Termux, 2026-04-27)
