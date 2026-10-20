# Train a text model with Chuck

`train_chuck` is a standalone C trainer linked to this notorch tree. Supply a
UTF-8 corpus; it creates both the model and a fresh Chuck: Loss Architect life.
No experiment archive, pretrained policy or Python runtime is needed.

```sh
make train_chuck
./train_chuck train --corpus book.txt --checkpoint book.ntc --steps 1000
./train_chuck resume --corpus book.txt --checkpoint book.ntc --output book.ntc --steps 4000
./train_chuck generate --checkpoint book.ntc --prompt "The " --tokens 256
```

The default build uses the Makefile's CPU BLAS backend. For a portable build
without BLAS, use `make train_chuck BLAS_FLAGS= BLAS_LIBS=`. On AVX2/FMA x86,
notorch's own SIMD path can be built with
`make train_chuck BLAS_FLAGS='-DUSE_SIMD -mavx2 -mfma' BLAS_LIBS=`;
`NT_SIMD_THREADS` sets its thread count. Rebuild with `make -B train_chuck`
when changing backend/compiler flags. This trainer uses CPU arithmetic.

## Commands

`train` requires `--corpus`, `--checkpoint` and `--steps`. Defaults:
`--seed 42 --lr .0003 --save-every 128 --acquire-every 64`.
The destination must not exist. `--steps 0` saves initialization.

`resume` requires the original corpus, an input `--checkpoint`, an `--output`
and `--steps TOTAL`. TOTAL is the final completed step, not additional steps.
The output may be the input checkpoint. The saved seed, learning rate,
acquisition schedule and state are restored; they cannot be overridden.
`--save-every` can change. Equal TOTAL resaves the same state.

`generate` needs only `--checkpoint`. Defaults: empty prompt, `--tokens 256`,
`--seed 42`, `--temperature .8`. Temperature zero selects greedy decoding.
Tokens here are Unicode characters. Output is the prompt followed by exactly
the requested number of generated characters, without an added newline.
Prompt characters must be in the saved vocabulary. Generation leaves the file
unchanged. Diagnostics go to stderr; generated text goes to stdout.

Training and resume accept `--trace NEW_FILE` for JSONL observations, actions,
acquisition comparisons, evaluations and policy hashes. Existing trace files,
input/output aliases, unknown/duplicate arguments and invalid UTF-8 are refused.
SIGINT/SIGTERM completes the current training step, saves, then exits with
130/143. Checkpoints are also saved before the first update, periodically and
at normal completion. A forced kill can only preserve the last finished save.

## What trains

The graph is the existing SimpleLLM body: two transformer blocks, width 128,
four heads, SwiGLU width 384, RoPE and context 64. Its size depends on the
character vocabulary: 450,688 parameters for the 94-character Dracula corpus.
The last 10% of decoded characters is held out; training windows stay in the
first 90%. Vocabulary is sorted and includes both splits. Accepted inputs have
2–4096 distinct Unicode scalar values, no NUL, at most 128 MiB, and more than
64 characters in each split. This is a small character model, not a chat model.
Reported held-out loss averages eight fixed, evenly spaced windows in that split.

Chuck's 163-parameter policy chooses HOLD, BRAKE or PUSH. Before host steps
1, 65, 129, … at the default interval, the trainer measures all three actions
from the same current world: one intervention, 15 continuing training steps,
then four probe windows. Branches use the current frozen policy. The source
world is restored exactly, and 32 conditioned fitting updates teach the policy
from those measured consequences. Only acquired weights cross that boundary.
The policy then selects the actual host action. Ordinary host feedback updates
its history without another weight fit. The trace records features, all three
future losses, target scale, scores before/after and selected actions.
The existing `nt_chuck_architect_fit_conditioned` objective sets
`delta[action] = future_loss[HOLD] - future_loss[action]`, divides by
`max(1e-6 * (abs(future_loss[HOLD]) + 1), abs(delta[BRAKE]), abs(delta[PUSH]))`,
and fits all three heads with bounded Huber targets. Fitting uses policy learning
rate .03. These training probes do not use the held-out split.

## Complete checkpoint and exact continuation

Version 1 `NTCHTRN1` is one file, with explicit little-endian integers/IEEE-F32
values and a SHA-256 digest. It contains model tensors, first/second moments,
optimizer counters, global/local Chuck histories, the canonical Architect
life, vocabulary, corpus SHA-256, split, window/noise/policy RNGs and acquisition
counters. Transient tape entries are rebuilt. A temporary file in the same
directory is flushed and synced, atomically renamed, then the directory synced.
Loading validates the payload before installing the model.

Exact continuation requires the same executable, numerical backend and thread
settings. Serialization does not promise identical arithmetic across CPUs,
BLAS implementations or compiler flags. Resume verifies the original corpus's
raw bytes; generation is corpus-free. This format belongs to this trainer and
is separate from the public policy-only life format.

`make check_chuck_trainer` verifies fresh training, cross-process resume,
initial-state resume, deterministic generation, learned policy change and input
refusal. Its deliberate resume-RNG defect must break checkpoint equality.

## Q: RRPRAM + Janus

`train_chuck_q` uses Q's `rrpram3_janus3` architecture with the same Chuck
acquisition loop, checkpoint lifecycle and commands:

```sh
make train_chuck_q
./train_chuck_q train --corpus q.txt --merges q.merges --checkpoint q.ntc --steps 1000
./train_chuck_q resume --corpus q.txt --checkpoint q.ntc --output q.ntc --steps 4000
./train_chuck_q generate --checkpoint q.ntc --prompt "The " --tokens 128
./train_chuck_q export --checkpoint q.ntc --output q.bin
```

Use `q.txt` and `q.merges` from
[Q](https://github.com/ariannamethod/q/tree/f5d00a36ecfcdb5e655e1576770f03d06d900e04).
The architecture has width 192, three layers, three RRPRAM heads and three Janus
heads, head width 32, learned positions 128, tied token/output embeddings,
parameter-free RMSNorm, sigmoid mechanism gates with trained biases and ReLU
FFN width 768. Training windows contain 64 BPE tokens. With Q's vocabulary 1280
this is **1,653,894 parameters**. Q's native Janus norm uses epsilon 1e-8;
the tape graph preserves that value using scaled RMSNorm. Other RMSNorms use 1e-6.

The tokenizer preserves Q's ordered byte-BPE merges and raw byte decoding.
`--merges` is required only for fresh training. Resume and generation reconstruct
the tokenizer from the complete `NTCHTRQ1` checkpoint. `--tokens` counts BPE
tokens for Q. Q supports vocabularies 256–1280 and decoded token lengths up to 63
bytes. Corpus identity is the SHA-256 of its raw bytes. The final 10% of BPE
tokens is held out, with eight fixed evaluation windows. Defaults for seed,
learning rates, H16 acquisition, saves and sampling match the SimpleLLM trainer.

`export` atomically writes QPTQ version 1, including learned gate biases. Its
destination must be new. Load it using the original Q runtime:

```sh
./q q.bin q.merges q.txt
```

The Q runtime supplies its existing MetaWeights, parliament, chambers and
generation process. `train_chuck_q generate` samples the trained substrate
directly. Both use the exported substrate's tied embeddings and native norm.

SimpleLLM `NTCHTRN1` and Q `NTCHTRQ1` checkpoints are distinct and each trainer
rejects the other family. The existing SimpleLLM format remains unchanged.
Backend build options and exact-continuation requirements above also apply to Q.
`make check_chuck_q_trainer` runs tokenizer, model and full-checkpoint gates.
