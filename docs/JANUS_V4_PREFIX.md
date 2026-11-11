# Janus v4: published-graph prefix inference

`janus.forward_graph = "pytorch-v4-prefix"` executes the published low-rank
Janus v4 graph in native C. It restores historical Echo attention, simultaneous
embedding smear, the complete-block backout snapshot and the global RRPRAM
reduction. The model's weight tensors are unchanged.

The reference is [`janus_gpt_v4_lowrank.py`](https://huggingface.co/ataeff/janus4/blob/8c87c65c062ad535d12d562f1693fadac05c086c/janus/janus_gpt_v4_lowrank.py),
SHA256 `76541ae11822c8bc0c0eb19d898efa42e27509f113e76eaa01b04ed89f605752`.
The reference execution used CPU FP32 and PyTorch causal SDPA for the external
FlashAttention call. Its ordinary RMSNorm epsilon is `FLT_EPSILON`.

## File contract

| GGUF key | Behavior |
| --- | --- |
| `general.architecture = "janus"` | Existing Janus family and tensor layout |
| `janus.forward_graph = "pytorch-v4-prefix"` | Published Python graph, prefix recomputation |
| `janus.rms_norm_epsilon` (optional F32) | Positive finite normalization epsilon; default `FLT_EPSILON` in the prefix profile |
| `janus.forward_graph = "arianna-v4-incremental"`, or absent | Existing incremental arianna.c-compatible graph |

Unknown graph names, wrong metadata types and invalid explicit epsilon are
refused. Existing GGUF exports retain their original execution profile.
The corrected profile is selected in each new export. All Janus tensor names,
outer-first shapes and supported packed weight storage remain the same.
The prefix profile uses floating activations, including with packed weights.

## Operations

Smear reads the preceding **original normalized embedding**, and accepts a
positive or negative learned smear scalar. It then supplies the smeared
embedding to the per-layer residual/x0 remix.

Echo computes `echo = Wj(x)`, the soft-capped self-resonance score, causal
attention from the outer product of those scores, and history-weighted Echo
values. The native score uses `||Wj x||² / sqrt(E)`, algebraically equal to
`x · (Wjᵀ Wj x) / sqrt(E)`. The independent PyTorch oracle performs the explicit
back-projection and compares every output logit.

Backout snapshots layer index `block_count / 2` after both attention and MLP.
Normalization uses the profile's explicit/default epsilon, and the output
retains the model's `15 * tanh(logit / 15)` soft cap.

The published RRPRAM einsum `bte,her->bhr` reduces the entire supplied sequence
before broadcasting position scores. A newly supplied token changes earlier
hidden rows, so those rows are recomputed. A frozen KV history produces a
different result. Sequence state therefore contains input token IDs; the
scratch tensors are rebuilt for each requested output. Scratch storage grows
linearly with the available prefix, while causal attention work is quadratic.
Decode cost grows with prefix length.

For CLI chat with a 1024-position Janus export, set `NT_CTX=1024`; the CLI's
general default is 2048. Explicit cache allocations beyond the file's context
are refused. Allocate through the family hooks when embedding the decoder.

## Cache contract

Use `nt_model_kv_new` / `nt_model_kv_free` or the architecture's cache hooks.
Each prefix cache owns an independent token history. A call at position zero
starts a new sequence; subsequent calls append exactly at the stored length.
Gaps, nonzero rewinds, foreign caches, invalid tokens and capacity overruns are
refused without changing the output buffer. Calls with `logits = NULL` retain
the input IDs and defer numerical work until logits are requested. This makes
full prefill, chunked prefill and token-at-a-time input converge to the same
available-prefix computation. Multiple caches may be interleaved on one body.

The current entry point accepts token IDs. Tokenizer conversion is a separate
boundary: use the exact original tokenizer's ID mapping for parity checks.

## Proof

```
make check_janus_prefix BLAS_FLAGS= BLAS_LIBS= BLAS_NAME=native
python tests/gen_janus_prefix.py  # optional oracle regeneration; needs torch
```

The fixture has V=37, E=32, H=2, D=16, B=3, M=64, R=4, context=16. Every branch
is active; the header contains ten full-logit FP32 reference vectors. The native
gate checks every prefix, alternate input, chunking, restart, interleaved
caches, invalid calls, negative smear and an explicit epsilon. The old graph
differs from the oracle by 0.0262018, so restoring the old computation is
detectable. The initial 58-check run has maximum reference error 4.76837e-7.

The real checkpoint is `ataeff/janus4` revision
`8c87c65c062ad535d12d562f1693fadac05c086c`, file
`janus/bins/janus_v4_base_22k.bin`, SHA256
`5b2422e18dd3e3e161ef1f3760e0ccb7b96291f7e34e91453bfe6c6919b10971`.
It contains 176,292,506 F32 parameters. BIN-to-GGUF conversion preserves every
tensor byte. On three prompts (6, 13 and 21 tokens including BOS) and three
greedy continuation steps per prompt, all 12 argmax IDs agree with the Python
reference. Maximum full-vocabulary error is 4.38690e-5. The build is native
x86_64 C with floating activations; the CPU reference is PyTorch 2.15.0 FP32.

Additional 32-, 128- and 1024-token prefixes agree on the next-token argmax,
with maximum logit errors 1.71661e-5, 2.38419e-5 and 2.62260e-5. The incremental
profile's logits remain bit-identical to a freshly rebuilt pre-patch decoder
on two original base prompts. Full `make -j2 test BLAS_FLAGS= BLAS_LIBS=
BLAS_NAME=native` passes. The focused gate also passes ASan/UBSan; leak scanning
was disabled because the execution host's ptrace setup prevents LeakSanitizer.

The prompts are:

- `The capital of France is`
- `A small bird landed on the roof. It was looking for`
- `I spent three hours fixing a bug that turned out to be a missing comma. Honestly,`

For likelihood evaluation of this published graph, score only the final row
of each available prefix against the unseen next token. Earlier rows in a
full-sequence pass have received the global RRPRAM reduction of later inputs.
