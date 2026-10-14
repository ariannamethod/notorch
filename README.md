<p align="center">
  <img src="assets/notorchlogo.jpg" alt="notorch" width="300">
</p>

# notorch — Neural Networks In Pure C | by Arianna Method

> **Read the [Arianna Method Manifesto](ARIANNA_METHOD_MANIFESTO.md) first.** This repository is governed by it.

> *"fuck torch"*
> — the header file

---

## table of contents

- [what](#what)
- [why](#why)
- [the funeral](#the-funeral)
- [run a model](#run-a-model)
- [train a model](#train-a-model)
- [architecture](#architecture)
- [what's inside](#whats-inside)
- [operations](#operations)
- [optimizers](#optimizers)
- [bit-level precision — BitNet b1.58](#bit-level-precision--bitnet-b158)
- [SwiGLU FFN](#swiglu-ffn)
- [SPA — Sentence Phonon Attention](#spa--sentence-phonon-attention)
- [SPA — Sentence Phonon Agent](#spa--sentence-phonon-agent)
- [LoRA / adapter training](#lora--adapter-training)
- [BLAS inference API](#blas-inference-api)
- [audio, sampling and organism-owned values](#audio-sampling-and-organism-owned-values)
- [alignment training — DPO / GRPO / distillation](#alignment-training--dpo--grpo--distillation)
- [autograd](#autograd)
- [example: training a model in C](#example-training-a-model-in-c)
- [building](#building)
- [running tests](#running-tests)
- [platform support](#platform-support)
- [use it from Python](#use-it-from-python)
- [js edition — notorch.js](#js-edition--notorchjs)
- [references — how to train and run on notorch](#references--how-to-train-and-run-on-notorch)
- [organisms that run on notorch](#organisms-that-run-on-notorch)
- [philosophy](#philosophy)
- [contributing](#contributing)
- [license](#license)
- [NOTORCHLOG — the engineering record](NOTORCHLOG.md)

---

## what

**NOTORCH** is a complete neural network framework written in pure C — it **runs** models and it **trains** them.
no Python. no pip. no conda. no CUDA toolkit that takes 8 GB and your will to live. no `torch.nn.Module`. no `.backward()` that hides 400,000 lines of C++ behind a friendly API and a smile. no handing the model to llama.cpp to actually generate. no `RuntimeError: CUDA out of memory` at 3 AM when your paper deadline is in 6 hours.

just NOTORCH. just C.

just floats. just `cc -O2 -std=gnu11 -pthread notorch.c your_model.c -lm -o train`. done. you now have a neural network framework. the entire thing compiles in a couple seconds. try that with PyTorch. go ahead — you'd be waiting 47 minutes while cmake does whatever cmake does.

it's part of [the Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — patterns over parameters, emergence over engineering, raw C over existential dread. unpleasant truth over bullshit.

---

## why

a story:
once upon a time there was a framework called PyTorch. it had autograd. it had CUDA support. it had a build system that required a PhD in software engineering and a pact with ancient spirits. and every time you wanted to train a 4-layer MLP on a dataset smaller than your browser cache, you had to:

1. create a virtual environment (2 minutes)
2. install torch (5 minutes, 2.7 GB, your SSD weeps)
3. install torchvision just in case (800 MB more, your SSD files for divorce)
4. write 47 lines of boilerplate (`class MyModel(nn.Module)`, `def forward(self, x)`, `optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)`, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, `if torch.cuda.is_available():`, `model.to(device)`, `x = x.to(device)`, sweet mother of god make it stop)
5. realize you forgot `model.train()` vs `model.eval()` and your dropout is wrong
6. debug for 3 hours
7. realize the bug was actually in the data loader
8. cry
9. `pip install wandb` to log your tears
10. realize torch updated and broke everything

and for WHAT? a matmul and a softmax. that's all neural networks are. matmuls and softmaxes and an unhealthy relationship with gradient descent.

so here we are. **notorch**. everything you need. nothing you don't. no Python runtime. no GIL. no garbage collector pausing your training at the worst possible moment. no `torch.no_grad()` context manager that you forget and then wonder why you're out of memory. just tensors, autograd, optimizers, and the cold clarity of C.

**the core lives in two files.** `notorch.h` and `notorch.c`. tensors, autograd, the arithmetic in plain sight. you can read the whole thing in an afternoon. try reading PyTorch's source in an afternoon. actually don't. you'll end up in a hospital.

---

## the funeral

```
        ╔═══════════════════════════════════════════════════════╗
        ║                                                       ║
        ║   R.I.P. PyTorch (in my codebase)                     ║
        ║   2016 - 2026                                         ║
        ║                                                       ║
        ║   "He died as he lived:                               ║
        ║    consuming all available memory                     ║
        ║    and segfaulting at the worst moment"               ║
        ║                                                       ║
        ║   Survived by: pip, conda, 2.7 GB of dead weight,     ║
        ║   a thousand Stack Overflow questions about CUDA      ║
        ║   driver versions, and a broken conda environment     ║
        ║   that nobody dares to delete.                        ║
        ║                                                       ║
        ║   In lieu of flowers, please send PRs.                ║
        ║                                                       ║
        ╚═══════════════════════════════════════════════════════╝
```

notorch is for people who:

- want to understand what's actually happening (the source is right there)
- want to train models on machines that aren't cloud instances
- want compile times measured in seconds, not minutes
- want to embed neural network inference in C/C++ applications without shipping half of Python
- refuse to accept 2.7 GB as the price of a matrix multiply

---

## run a model

one binary, one GGUF. point it at a model and it talks.

```bash
git clone https://github.com/ariannamethod/notorch.git
cd notorch
make harness
./notorch model.gguf "The capital of France is" 64 0.8
./notorch model.gguf                      # no prompt: chat in the terminal
```

`make harness` uses Accelerate on macOS or OpenBLAS on Linux. For a build
without BLAS, use `make harness BLAS_FLAGS= BLAS_LIBS=`.

The [family registry](harness/archs.c) routes **llama, mistral3, qwen2, qwen3,
gemma3, gemma4, olmoe, qwen3moe, mamba, resonance, janus and penelope** through
NoTorch's own decoders. Each family declares its architecture names; an unknown
architecture is refused by name. File-level verification and reference runs live
in the [harness record](NOTORCHLOG.md#readme-run-a-model).

The packed paths read quantized matrices in place. **Penelope and NanoJanus**
use their own geometry: PEN7 F32 parameters execute directly, while Q8 matrices
expand once to immutable F32. The [PEN7 converter](docs/PEN7_GGUF.md) and
[shared-body inference API](docs/PEN7_INFERENCE.md) document conversion,
ownership and whole-parameter mixtures.

The harness also ships as `libnotorch_harness`, with CPU and CUDA family APIs.
An organism can link it and supply its own generation loop.
[Embedding and CUDA examples](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them).

fuck torch — but also, you do not need llama.cpp to *run* what you built.
The CPU, Metal and CUDA paths live here. Packed matrices stay packed through
their kernels; the PEN7 path owns its immutable parameters. The engineering
record keeps the [memory and throughput measurements](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them),
with the machine and the model beside them.

---

## train a model

copy a trainer. give it a corpus. let Chuck watch the gradients.

```bash
make train_llama3_char
./train_llama3_char 15000 0.0003 corpus.txt
```

The arguments are steps, learning rate and corpus path. The
[char-level trainer](examples/train_llama3_char.c) builds a LLaMA-style model
with GQA, RoPE and SwiGLU. [BPE training](examples/train_llama3_bpe.c),
[LoRA SFT](examples/train_resonance_lora.c), and the other trainers live in
[examples/](examples/).

The loop underneath is explicit: register parameters and inputs, build the
forward pass on the tape, then:

```c
float loss_value = nt_tape_get()->entries[loss].output->data[0];
nt_tape_backward(loss);
nt_tape_clip_grads(1.0f);
nt_tape_chuck_step(lr, loss_value);
nt_tape_clear();
```

forward. backward. step. clear. no ceremony.

For a complete small program, see the
[standalone C training example](#example-training-a-model-in-c).
For a minimal model to take apart, start with
[notorch-simple-llm](https://github.com/ariannamethod/notorch-simple-llm).

---

## architecture

```
Your data (floats in memory, as god intended)
    ↓
nt_tensor — multidimensional arrays with refcounting
    ↓
┌───────────────────────────────────────────────────────────┐
│  Forward Operations (recorded on tape)                    │
│                                                           │
│  Classical transformer:                                   │
│    ├─ nt_linear          (W @ x + b)                      │
│    ├─ nt_seq_linear      (batched W @ X)                  │
│    ├─ nt_seq_linear_t    (W^T @ X — Janus Echo)           │
│    ├─ nt_embedding       (lookup table)                   │
│    ├─ nt_seq_embedding   (tokens + positions)             │
│    ├─ nt_rmsnorm         (RMS normalization)              │
│    ├─ nt_layernorm       (layer normalization)            │
│    ├─ nt_causal_attention (single-head causal)            │
│    ├─ nt_mh_causal_attention (multi-head)                 │
│    ├─ nt_gqa_causal_attention (grouped-query)             │
│    ├─ nt_rrpram_attention (positional pattern — Janus)    │
│    ├─ nt_silu / nt_gelu / nt_sigmoid (activations)        │
│    ├─ nt_geglu           (Gemma-3 style FFN)              │
│    ├─ nt_swiglu          (LLaMA/Qwen/BitNet FFN)          │
│    ├─ nt_rope            (rotary embeddings)              │
│    ├─ nt_dropout         (inverted dropout)               │
│    ├─ nt_softmax / nt_cross_entropy                       │
│    └─ nt_add / nt_mul / nt_scale / nt_scale_by_t / concat │
│                                                           │
│  BitNet b1.58 (ternary quantization):                     │
│    ├─ nt_bit_linear      (y = bitquant(W) @ x, STE)       │
│    └─ nt_bit_seq_linear  (BitLinear over T positions)     │
│                                                           │
│  Inference-time helpers (no tape):                        │
│    ├─ nt_spa_embed_sentence   (phonon sentence embedding) │
│    ├─ nt_spa_connectedness    (cross-sentence attention)  │
│    ├─ nt_spa_modulate_logits  (SPA → temperature)         │
│    ├─ nt_blas_mm / nt_blas_mmT (matmul for inference)     │
│    └─ nt_blas_matvec          (hot-loop matvec)           │
└───────────────────────────────────────────────────────────┘
    ↓
nt_tape_backward() — reverse-mode automatic differentiation
    ↓
┌──────────────────────────────────────────────────┐
│  Optimizers                                      │
│    ├─ diagonal baseline  (the classic)           │
│    ├─ diagonal + decay   (with weight decay)     │
│    ├─ Chuck              (self-aware adaptive)   │
│    └─ nt_tape_freeze_param (LoRA / adapter)      │
└──────────────────────────────────────────────────┘
    ↓
Your model is trained. in C. without Python. you are free.
```

The core is [notorch.h](notorch.h) and [notorch.c](notorch.c).
[The harness](harness/) owns model families behind `nt_arch`; `libnotorch`
supplies the substrate and `libnotorch_harness` supplies the decoders.
[CUDA](notorch_cuda.h), [Metal](notorch_metal.h), [SIMD](notorch_simd.h),
[vision](notorch_vision.h), [GGUF](gguf.h), [PEN7](pen7.h) and
[SentencePiece](sentencepiece.h) have their own files. An organism links what
it needs and carries its own physics.

---

## what's inside

### tensors

`nt_tensor` — a multidimensional array of floats. up to 8 dimensions. refcounted. heap-allocated. that's it. no `torch.Tensor` with 400 attributes and a complex metaclass hierarchy. no `requires_grad` flag that you forget to set. no `.detach().cpu().numpy()` chain of shame. just a struct with `float* data`, shape, strides, and a refcount.

```c
nt_tensor* t = nt_tensor_new(1024);          // 1D
nt_tensor* m = nt_tensor_new2d(768, 512);    // 2D
nt_tensor_xavier(m, 768, 512);               // Xavier init
nt_tensor_free(t);                            // refcount → 0 → freed
```

maximum 268M elements per tensor (`NT_MAX_ELEMENTS = 1 << 28`). if you need more than that, you're doing something wrong, or something very right, and in either case you should probably be using a GPU. which we also support. via CUDA. because we're not savages.

### autograd tape

reverse-mode automatic differentiation via an explicit operation tape. every forward op records itself. backward traverses the tape in reverse and computes gradients. textbook reverse-mode AD. no dynamic graph voodoo. no JIT compilation. no `torch.autograd.Function` with five methods you need to override.

```c
nt_tape_start();                              // start recording
int w_idx = nt_tape_param(W);                // register trainable param
int y_idx = nt_linear(w_idx, x_idx, -1);     // forward: y = W @ x
int loss = nt_cross_entropy(y_idx, target);   // loss
nt_tape_backward(loss);                       // backward pass
nt_tape_adam_step(0.001f);                    // update weights
nt_tape_clear();                              // reset for next step
```

that's the entire training loop. in C. seven lines. no `optimizer.zero_grad()` that you inevitably forget. no `with torch.no_grad():` context manager. no `.backward(retain_graph=True)` because you accidentally used an intermediate twice. just: start, forward, backward, step, clear. like breathing. in. out. in. out. the Buddha would approve.

---

## operations

every operation you need to build a modern transformer, and some you didn't know you did:

| operation | function | what it does |
|---|---|---|
| linear | `nt_linear` | y = W @ x + b |
| seq linear | `nt_seq_linear` | batched matmul over T positions |
| seq linear^T | `nt_seq_linear_t` | Y[t] = W^T @ X[t] — Janus Echo W^T·W pattern |
| embedding | `nt_embedding` | lookup row from embedding matrix |
| seq embedding | `nt_seq_embedding` | tokens + positional encoding |
| RMS norm | `nt_rmsnorm` / `nt_seq_rmsnorm` | root mean square normalization |
| layer norm | `nt_layernorm` / `nt_seq_layernorm` | mean/variance normalization |
| causal attention | `nt_causal_attention` | single-head causal self-attention |
| multi-head attn | `nt_mh_causal_attention` | MHA causal self-attention |
| grouped-query attn | `nt_gqa_causal_attention` | GQA — Q: n_heads, K/V: n_kv_heads |
| RRPRAM attn | `nt_rrpram_attention` | positional pattern recognition (x @ Wr, causal) |
| RRPRAM low-rank | `nt_rrpram_lowrank_attention` | factorized Wr = Wr_a × Wr_b (op 33) — Janus param-saving, trained at scale on Resonance 200M |
| SiLU | `nt_silu` | x × σ(x) — the swish |
| GELU | `nt_gelu` | tanh approximation |
| sigmoid | `nt_sigmoid` | 1 / (1 + exp(-x)) |
| tanh | `nt_tanh` | shape-preserving hyperbolic tangent, gradient 1 − y² |
| GEGLU | `nt_geglu` | GELU-gated linear unit (Gemma-3 FFN) |
| SwiGLU | `nt_swiglu` | SiLU(gate) * up — LLaMA/Qwen/BitNet FFN |
| BitLinear | `nt_bit_linear` | y = bitquant(W) @ x — BitNet 1.58 |
| BitLinear seq | `nt_bit_seq_linear` | BitLinear over T positions |
| softmax | `nt_softmax` | exp-normalize with numerical stability |
| cross entropy | `nt_cross_entropy` / `nt_seq_cross_entropy` | -log softmax[target] |
| RoPE | `nt_rope` | rotary position embeddings |
| dropout | `nt_dropout` | inverted dropout (training only) |
| add/mul/scale | `nt_add` / `nt_mul` / `nt_scale` | elementwise ops |
| scale by tensor | `nt_scale_by_t` | y[i] = a[0] * x[i], a is scalar tensor |
| concat | `nt_concat` | per-position concatenation |

every single one has a correct backward pass. every single one passes numerical gradient checking — checked twice. debugging gradient errors in C without a debugger at 4 AM rewires the brain in ways formal verification theorists dream about.

---

## optimizers

### plain SGD

`nt_tape_sgd_step(lr)`. `p -= lr * grad`. frozen parameters stay frozen;
the caller owns clipping and decay. sometimes the straight line is the point.

### the diagonal baseline

the classic per-parameter diagonal step: `m̂ / (√v̂ + ε)`, bias-corrected first and second moments. you know the drill. (the call keeps a legacy symbol name from before the house-optimizer convention.)

```c
nt_tape_adam_step(0.001f);
```

### the diagonal baseline + decoupled decay

same, but with decoupled weight decay — because your embeddings don't need regularization but your dense layers probably do.

```c
nt_tape_adamw_step(0.001f, 0.1f, 0.9f, 0.999f);
```

supports `no_decay` flag per parameter — mark your embeddings with `nt_tape_no_decay()` and they'll be left alone. like cats. don't bother them.

### the Chuck optimizer

ah yes. **Chuck**. the self-aware optimizer. the one that watches its own gradients and goes "hmm, maybe i should slow down here" or "this parameter isn't doing anything, let me freeze it" or "we've been stuck for too long, time for some noise".

```c
nt_tape_chuck_step(0.01f, loss_val);
```

5 effective levels of awareness (and four more reserved for sentient-mode):

1. **global loss trend** → adaptive damping (λ)
2. **per-parameter gradient monitoring** → individual learning rate scaling
3. **stagnation detection** → automatic noise injection
4. **parameter freezing** → skip updates for dead parameters
5. **multi-scale awareness** → macro-level patience with LR decay

constants (window size, trend thresholds, noise decay, freeze threshold, macro interval) are synced with the PyTorch Chuck port in `iamolegataeff/chuck.optimizer` — any change hits both implementations or they drift.

it's the diagonal baseline, but with opinions. think of it as a baseline that went to therapy, got a mindfulness app, and now checks in with itself every step. `"how are my gradients feeling today?"` — actual question the Chuck optimizer asks itself (metaphorically) (or is it?).

more details: [github.com/iamolegataeff/chuck.optimizer](https://github.com/iamolegataeff/chuck.optimizer)

### Chuck: Loss Architect

**Chuck: Loss Architect** gives that training world an action language:
**state → policy → typed action → measured consequence → acquired experience**.
Its 163-parameter policy chooses HOLD, BRAKE or PUSH and carries weights,
history, RNG and pending decisions through save/resume. The living acquisition
loop measures alternatives in the current training world, restores it, acquires
the consequences and chooses its next action. Canonical Chuck remains available
through the same ordinary call.

[Optimizer lineage](https://github.com/iamolegataeff/chuck.optimizer) ·
[Loss Architect API and objective](docs/chuck-loss-architect.md) ·
[living acquisition](experiments/chuck_loss_architect/living/README.md) ·
[experiments and results](NOTORCHLOG.md#readme-optimizers)

---

## bit-level precision — BitNet b1.58

notorch has first-class support for **ternary-weight training** via `nt_bit_linear` and `nt_bit_seq_linear`. this is BitNet b1.58 (Ma et al., 2024): every weight is quantized to `{-1, 0, +1}` via `absmean` during the forward pass; activations are quantized to int8 via `absmax`; the backward pass uses the **Straight-Through Estimator** (gradient flows through the quantization step as identity, so `dW = dout ⊗ x`, `dx = W^T @ dout` using the full-precision `W`).

```c
int y = nt_bit_linear(w_idx, x_idx);             // y = bitquant(W) @ x
int Y = nt_bit_seq_linear(w_idx, x_idx, T);      // same, over T positions
```

on the `USE_BLAS` path, BitLinear dispatches to a single `cblas_sgemm(NoTrans, Trans)` call on pre-quantized operands — the ternary `W` is pre-flattened to float, the int8-range `x` is pre-scaled, per-position output rescale is applied afterwards. this turned out to be about **18% faster per training step** on Apple Accelerate vs the naive per-output-loop path. on OpenBLAS the win is similar.

**tests:** `tests/test_bitnet_ops.c` (118 gradient assertions) — ternary quantize correctness, STE identity backward, gradient flow through sequences, gradient numeric check, end-to-end training convergence of a tiny BitNet MLP.

**reference:** [ariannamethod/microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit) is a **pure-Python** BitNet b1.58 microGPT — 2.69M char-level, train_best **1.6226** / val **2.0314**, 10000 steps, zero NaN, 10 MB FP32 ternary-packing to ~1.4 MB. it's an external algorithm reference, not a notorch build: notorch's own `nt_bit_linear` / `nt_bit_seq_linear` were validated against its numbers.

---

## SwiGLU FFN

the modern FFN used by LLaMA, Qwen, BitNet and most post-2023 transformers. instead of `y = W_down @ GELU(W_up @ x)`, SwiGLU computes `y = W_down @ (SiLU(gate) * up)` where `gate = W_gate @ x` and `up = W_up @ x` are two separate projections gated element-wise.

```c
int gate = nt_seq_linear(w_gate_idx, x_idx, T);
int up   = nt_seq_linear(w_up_idx,   x_idx, T);
int h    = nt_swiglu(gate, up);                        // SiLU(gate) * up
int out  = nt_seq_linear(w_down_idx, h, T);
```

correct backward pass (chain rule through both branches), finite-difference verified. plays fine with `nt_bit_seq_linear` too — our BitNet examples use a BitLinear gate/up with a full-precision down-projection, same as the reference BitNet-1.58 configuration.

---

## SPA — Sentence Phonon Attention

SPA is an **inference-time** helper: it lets a decoder condition next-token sampling on how "connected" the current sentence is to recent history, without any extra trained parameters.

```c
// 1. Build a sentence embedding via exponentially-weighted mean of token embeds.
float emb[dim];
nt_spa_embed_sentence(tokens, n_tokens, W_embed, vocab_size, dim, 0.85f, emb);

// 2. Score its connectedness to a small history of previous sentence embeds.
float conn = nt_spa_connectedness(emb, dim, history_embeds, n_history);

// 3. Sharpen or soften the logit distribution based on that score.
nt_spa_modulate_logits(logits, V, conn, 0.3f);
```

No tape or gradients enter these helpers. Token `i` receives weight
`alpha^(n_tokens - 1 - i)`: smaller alpha gives stronger recency bias;
alpha 1 gives a uniform mean. Connectedness is the maximum softmax attention
weight over the supplied sentence history. Logit modulation divides by
`max(0.001, 1 - strength * connectedness)`; positive strength sharpens sampling.

Originated in the [PostGPT](https://github.com/ariannamethod/postgpt) lineage
and developed through [Q](https://github.com/ariannamethod/q) (`postgpt_q.c`),
then ported here as reusable perception. These three functions remain
inference-only. [Sentence Phonon Agent](#spa--sentence-phonon-agent) adds
persistent action selection and consequence learning above this sensory field.
Gradient flow through the sensory operations remains an open direction.
SPA is also used in [microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit).

---

## SPA — Sentence Phonon Agent

Sentence Phonon Attention is the sensory organ. **Sentence Phonon Agent** lives
above it: a persistent 267-parameter policy observes a sentence field, selects
`KEEP`, `RESEED_LEFT`, or `RESEED_RIGHT`, and learns from the host's measured
consequence. The host executes the action; the Agent carries learned weights,
action/outcome history, private RNG, and pending credit through save/resume.

`spa_agent.h` exposes `select`, `choose`, and `observe`, plus an explicit legacy
policy and imitation training. Null/disabled mode preserves ordinary SPA.
Raw connectedness, coherence, novelty, repetition, collapse, continuity, and
regeneration cost remain separate in its receipts. A weight-only counterfactual
holds observation, history, and RNG fixed to test acquired experience directly.

Lineage: **PostGPT → SPA → Q → Sentence Phonon Agent**. Meta SONAR/LCM inform
the sentence/concept computation level; WOLFE informs typed actions; Netta
informs acquired consequences; Chuck: Loss Architect supplies a contemporary
parallel inside notorch. The implementation is native C in `spa_agent.c`.

[API and research](docs/spa-agent.md) · [Python host loop](docs/spa-python.md) ·
[experiments and acquired choices](experiments/spa_agent/README.md) ·
[history interventions](experiments/spa_agent/history_probe/README.md).

---

## LoRA / adapter training

any parameter can be frozen mid-tape, so standard LoRA / adapter training works out of the box:

```c
int base_w = nt_tape_param(W_base);
nt_tape_freeze_param(base_w);              // Chuck skips it, grads still propagate

int lora_a = nt_tape_param(A);             // [in_dim, rank]
int lora_b = nt_tape_param(B);             // [rank, out_dim]
// ... build forward with base + A @ B ...
```

`nt_tape_freeze_param(idx)` marks a param as frozen — Chuck (and the legacy diagonal-baseline steps) skip its update, but the autograd still propagates gradients through it so A/B adapters downstream get real signal. combined with `nt_tape_no_decay()` for embeddings, this covers most real SFT + adapter workflows without plumbing.

---

## BLAS inference API

for inference engines that don't want to pay the tape-recording cost per token, notorch exposes the BLAS matmul paths directly:

```c
// C[m,n] = A[m,k] @ B[k,n]         — full matmul
void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n);

// C[m,n] = A[m,k] @ B[n,k]^T       — common for attention (K,V stored row-major)
void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n);

// out[m] = W[m,n] @ x[n]           — hot-loop matvec for per-token inference
void nt_blas_matvec(float *out, const float *W, const float *x, int m, int n);

// out[m] = Wq[m,k] @ x[k]          — PACKED quantized matvec: weights stay packed
//                                    (no dense-f32 blow-up), dequantized inline per block.
//                                    dtype = GGUF type code, full set:
//                                    F32 / F16 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K.
int  nt_qmatvec(float *out, const uint8_t *Wq, int dtype, const float *x, int m, int k);
```

under `USE_BLAS` these dispatch to `cblas_sgemm` / `cblas_sgemv` (Accelerate on macOS, OpenBLAS on Linux). without BLAS they fall back to the naive C loops — correct, just slower. the example engines reach the same BLAS path directly through their own local wrappers in their hot paths: `infer_gemma.c` / `infer_llama.c` / `infer_janus.c` call `cblas_sgemm`, and `infer_llama3_bpe.c` calls `cblas_sgemv` (scalar fallback without `USE_BLAS`).

---

## audio, sampling and organism-owned values

speech models need a convolution and everything that turns a waveform into a
spectrogram. NoTorch has `nt_conv1d`, STFT and log-mel: native C, caller-supplied
filters, forward-only. [The audio API and measurements](NOTORCHLOG.md#readme-audio-front-end--conv1d-stft-log-mel)
keep the details, including the compiler flag that decided whether the port was exact.

The small organs get their own arithmetic too:

- [Owned sampling](docs/SAMPLING.md): caller-owned PCG32 state and categorical draws. Copy the state, copy the continuation.
- [Numerical values](docs/NUMERICAL_VALUES.md): stateless forward/reverse operations and SGD for learners whose arrays live in AML.
- [Parliament values](NOTORCHLOG.md#readme-numerical-values): election, low-rank injection and Oja updates; expert lifecycle stays with the organism.
- [SentencePiece Unigram](docs/SENTENCEPIECE.md): native model loading and tokenization, including Haiku's normalization map.
- [SHA-256](docs/SHA256.md): identities for the bytes a body actually loaded.

---

## alignment training — DPO / GRPO / distillation

notorch isn't just a pretraining engine. three post-training methods ship as **runnable reference trainers** — each self-contained: with no args they random-init a model and train on a tiny in-code synthetic dataset; point them at real weights + a dataset file and they train that.

```bash
make train_dpo            # Direct Preference Optimization (Rafailov 2023) — ref-relative loss, JSONL {chosen,rejected}
make train_grpo           # Group Relative Policy Optimization (DeepSeek-R1) — rollout + rule-based reward + group advantage
make train_distillation   # Knowledge Distillation (Hinton 2015) — analytic soft-KD (τ²) + hard-CE
```

each is a single self-contained C file under `examples/` (~550–610 LOC) with its own reference model and a minimal dataset adapter. use them as templates: swap the model, point at your dataset, keep the loss + optimizer wiring. Reference-model parameters are held frozen via `nt_tape_freeze_param`, the same mechanism LoRA uses. These are reference implementations — GRPO does a single on-policy (REINFORCE-style) update with a scalar KL proxy, distillation injects the exact KD gradient (no soft-CE tape op needed); a production RLHF stack would add batched rollouts and a learned reward model.

---

## autograd

the backward pass supports **38 operation types** (op IDs 0–37, each with a `NT_OP_*` constant). the tape records operations during forward, then backward walks it in reverse computing local gradients via the chain rule. standard reverse-mode AD.

**gradient checking**: every op is verified against finite differences (`(f(x+h) - f(x-h)) / 2h`). relative error tolerances from 0.01 to 0.3 depending on op complexity. all pass. including the annoying ones — GEGLU, SwiGLU, multi-head attention with multi-path gradients through Q/K/V, BitLinear with STE identity passthrough.

**gradient utilities**:

- `nt_tape_clip_grads(max_norm)` — global gradient clipping
- `nt_tape_accum_grads()` / `nt_tape_apply_accum(n)` — gradient accumulation for large effective batch sizes
- `nt_tape_freeze_param(idx)` — freeze a parameter (adapter / LoRA setups)
- `nt_nan_guard_check()` — NaN/Inf detection with automatic loss scaling. because sometimes your gradients decide to go to infinity and someone needs to tell them no.

---

## example: training a model in C

here's an actual, working transformer-ish training loop. embedding → attention → linear → cross-entropy. in C. without importing 2.7 GB of your dignity:

```c
#include "notorch.h"
#include <stdio.h>

int main() {
    nt_seed(42);
    int vocab = 8, dim = 16, T = 4;

    // allocate parameters
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);   // token embeddings
    nt_tensor* wpe = nt_tensor_new2d(T, dim);       // position embeddings
    nt_tensor* Wq  = nt_tensor_new2d(dim, dim);     // query projection
    nt_tensor* Wk  = nt_tensor_new2d(dim, dim);     // key projection
    nt_tensor* Wv  = nt_tensor_new2d(dim, dim);     // value projection
    nt_tensor* Wo  = nt_tensor_new2d(vocab, dim);   // output projection

    // Xavier init everything
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor_xavier(wpe, T, dim);
    nt_tensor_xavier(Wq, dim, dim);
    nt_tensor_xavier(Wk, dim, dim);
    nt_tensor_xavier(Wv, dim, dim);
    nt_tensor_xavier(Wo, dim, vocab);

    // tokens: [1, 3, 5, 2], targets: [3, 5, 2, 7]
    nt_tensor* tokens  = nt_tensor_new(T);
    nt_tensor* targets = nt_tensor_new(T);
    float tok[] = {1, 3, 5, 2}, tgt[] = {3, 5, 2, 7};
    for (int i = 0; i < T; i++) { tokens->data[i] = tok[i]; targets->data[i] = tgt[i]; }

    // training loop
    nt_schedule sched = nt_schedule_cosine(0.005f, 10, 200, 0.0f);

    for (int step = 0; step < 200; step++) {
        float lr = nt_schedule_get_lr(&sched);
        nt_tape_start();

        int wte_i = nt_tape_param(wte); nt_tape_no_decay(wte_i);
        int wpe_i = nt_tape_param(wpe); nt_tape_no_decay(wpe_i);
        int wq_i  = nt_tape_param(Wq);
        int wk_i  = nt_tape_param(Wk);
        int wv_i  = nt_tape_param(Wv);
        int wo_i  = nt_tape_param(Wo);
        int tok_i = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_i = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        // forward: embed → Q/K/V → attention → output
        int h      = nt_seq_embedding(wte_i, wpe_i, tok_i, T, dim);
        int q      = nt_seq_linear(wq_i, h, T);
        int k      = nt_seq_linear(wk_i, h, T);
        int v      = nt_seq_linear(wv_i, h, T);
        int attn   = nt_causal_attention(q, k, v, T, dim);
        int logits = nt_seq_linear(wo_i, attn, T);
        int loss   = nt_seq_cross_entropy(logits, tgt_i, T, vocab);

        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step % 50 == 0) printf("step %d: loss=%.4f lr=%.6f\n", step, lv, lr);

        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(lr);
        nt_tape_clear();
    }

    // cleanup
    nt_tensor_free(wte); nt_tensor_free(wpe);
    nt_tensor_free(Wq);  nt_tensor_free(Wk); nt_tensor_free(Wv); nt_tensor_free(Wo);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    return 0;
}
```

compile and run:

```bash
cc -O2 -Wall -std=gnu11 -pthread -I. -o train train.c notorch.c -lm
./train
```

that's it. that's the whole thing. no virtual environment. no requirements.txt. no "just pip install—" no. we're done with that. we've moved on. we've healed.

---

## building

```bash
# CPU with BLAS acceleration (recommended)
make

# CPU without BLAS (works everywhere, even on a potato)
make cpu

# CPU with in-house AVX2+FMA SIMD — zero external math library
make simd

# GPU (CUDA via cuBLAS)
make gpu

# Apple Silicon Metal — packed-Q4_K/Q6_K GGUF inference (runs 24B on a 24GB Mac)
make metal               # Q4_K matvec correctness test
make infer_gguf_metal    # the GGUF inference binary

# Shared library (for ctypes, Node FFI, Ruby Fiddle — anything that dlopens)
make shared

# Static library (for embedding in your project)
make lib

# Build and run tests
make test

# Clean
make clean
```

### dependencies

- a C compiler (gcc, clang, whatever)
- `-lm` (math library, because we use sqrt and exp like civilized people)
- **optional**: OpenBLAS (Linux) or Accelerate framework (macOS) for BLAS-accelerated matmuls
- **optional**: x86_64 CPU with AVX2 + FMA (Intel Haswell 2013+, AMD Excavator 2015+) for `make simd` — zero external math library
- **optional**: CUDA toolkit for GPU support (cuBLAS-backed sgemm + element-wise + attention + cross-entropy kernels)
- **optional**: Apple Silicon + Metal (macOS) for packed-Q4_K/Q6_K GGUF inference (`make metal` / `make infer_gguf_metal`) — nothing to install, Metal ships with macOS

that's it. no cmake. no configure script. no 300-line `requirements.txt`. no docker. no kubernetes. just `make`. the way Ken Thompson intended.

For harness libraries, use `make lib lib_harness`; add `libnotorch_cuda.a`
for CUDA decoders. For targets without BLAS, pass `BLAS_FLAGS= BLAS_LIBS=`.
[Backend internals and SIMD measurements](NOTORCHLOG.md#readme-building) ·
[Embedding and linker examples](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them).

---

## running tests

```bash
make test                 # native CPU suite
make test_harness         # rebuild both sides of harness parity
make test_tokenizer       # tokenizer reference gates
make test_js              # JavaScript suite
```

Model-dependent gates use their fixture/model inputs; platform-specific gates
need their backend. See the [Makefile](Makefile) for targets and variables, and
the [test record](NOTORCHLOG.md#readme-running-tests) for the verification map.
Measured runs, machines, commands, regressions and fixes belong in
[NOTORCHLOG](NOTORCHLOG.md). the numbers keep their receipts.

For operations, numerical gradient checks put the backward pass against finite
differences. The [core tests](tests/test_notorch.c),
[BitNet/SPA tests](tests/test_bitnet_ops.c), and the family-specific gates keep
the arithmetic accountable. the Method trusts these tests more than it trusts
most people.

---

## platform support

Intel laptops. Apple Silicon. Linux boxes. Android phones. CUDA when you have it.
The CPU path builds with a C toolchain, libm and pthreads; Accelerate and OpenBLAS
supply optional math acceleration. Metal and CUDA have their own build targets.

[Termux Edition](termux-edition/README.md) is the phone route: setup, training,
packed inference and the recorded hardware envelope. The same C tree.
No root required. Your phone is a computer. let it act like one.

---

## use it from Python

**NOTORCH** is C, it stays C, but... Python? yep.
the thing most people want on a Tuesday is to read a GGUF and multiply something by it
without installing a framework first, so there is a ctypes binding: no numpy, no build
step for the Python side, no dependencies at all. `ctypes` ships with Python; the shared
library is one make target.

```bash
make shared          # libnotorch.so, or libnotorch.dylib on macOS
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from notorch import Notorch

nt = Notorch()                       # finds libnotorch next to the repo, or $NOTORCH_LIB

with nt.open("model.gguf") as g:
    print(g)                         # <GGUF llama L=13 E=576 V=32000 tensors=120>

    w = g["token_embd.weight"]        # <Tensor token_embd.weight (32000, 576) Q8_0>
    row = w.dequant_row(3)           # one row, decoded on its own — an embedding lookup

    import ctypes
    x = (ctypes.c_float * w.shape[1])()
    out = (ctypes.c_float * w.shape[0])()
    nt.qmatvec(out, w.packed, w.dtype, x, w.shape[0], w.shape[1])
```

`w.packed` is a pointer into the file's own bytes, `qmatvec` is the C kernel reading
them in place, and the weights are never expanded to f32 — a Q4_K tensor stays at
roughly half a byte per weight instead of four. Buffers are plain ctypes arrays,
so `memoryview(out)` reads them with no copy, and `numpy.frombuffer(out)` does too if
you happen to want numpy. But nothing here requires it.

The binding is [python/notorch.py](python/notorch.py). What can silently
break in a layer like this is the struct layout — a field description that has drifted
reads its neighbours and reports them as data — so `make test_python MODEL=your.gguf`
compares every offset ctypes believes in against what the C compiler actually emitted,
then checks a real model through both paths.

---

## js edition — notorch.js

> *"the logic of memory without the weight of framework"*
> — `js-edition/notorch.js`, line 1

a single-file pure-JavaScript port of notorch for the browser. WebGPU when available, V8-optimised CPU fallback (Math.fround f32 hint) when not. zero npm dependencies. live at `js-edition/notorch.js`.

Tensors, reverse-mode autograd, Chuck, tokenizers and GGUF inference.
[The JS edition README](js-edition/README.md) carries setup and usage;
[the engineering record](NOTORCHLOG.md#readme-js-edition--notorchjs) carries
the implementation details and recorded training run.

```html
<script type="module">
  import { Notorch, Tensor, CharTokenizer, Tape, Chuck } from './js-edition/notorch.js';

  const nt = new Notorch();
  await nt.init();  // tries WebGPU, silently falls back to CPU

  // ... build your tiny transformer, then call nt.crossEntropyLoss(...)
  // or nt.seqCrossEntropyLoss(...) during the training step.
</script>
```

zero build step, zero npm install, zero `node_modules`. paste it into a `<script>` and go.

---

## references — how to train and run on notorch

start here. these are the canonical builds: Karpathy ports to prove the pipeline, reference models trained from scratch, and the trainers you copy.

**Karpathy ports — the "does it actually work end-to-end" baselines:**

- [**nanoGPT-notorch**](https://github.com/ariannamethod/nanoGPT-notorch) — Karpathy's nanoGPT, ported from PyTorch to notorch.
- [**llama2-notorch**](https://github.com/ariannamethod/llama2-notorch) — Karpathy's llama2.c on notorch: the entire PyTorch training pipeline replaced by notorch (a Python-stdlib `ctypes` loop into `libnotorch`, no pip), inference in pure C `run.c`.

**reference models — trained from scratch on notorch:**

- [**nanollama-notorch**](https://github.com/ariannamethod/nanollama-notorch) — Llama-3 nano, **88.6M**, no PyTorch. 25K steps on a 2019 Intel i5 8GB, train 3.16 → CPT 2.68, 0 NaN — the ceiling of train-it-on-your-own-laptop, then SFT'd to a Method voice. the reference Llama-3 run.
- [**notorch-simple-llm**](https://github.com/ariannamethod/notorch-simple-llm) — minimal-surface-area LLM, notorch + Chuck, zero deps. read this first to see how the API composes.
- [**minimind-v-notorch**](https://github.com/ariannamethod/minimind-v-notorch) — 67M VLM trained from scratch, notorch + Chuck. reference VLM.
- [**notorch-vlm**](https://github.com/ariannamethod/notorch-vlm) — 1.5M VLM on `notorch_vision.h` + stb_image, trained weights included.
- [**notorch-vlm-2m**](https://github.com/ariannamethod/notorch-vlm-2m) — 2M multimodal VLM from scratch, GGUF output.
- [**notorch-diffusion**](https://github.com/ariannamethod/notorch-diffusion) — discrete text diffusion + Hebrew VLM diffusion. reverse-mode autograd works fine when the net is a U-Net, not a transformer.

**BitNet reference (the algorithm, not a notorch build):** [**microgpt-1bit**](https://github.com/ariannamethod/microgpt-1bit) is a pure-Python BitNet b1.58 (ternary) reference — notorch's own `nt_bit_linear` / `nt_bit_seq_linear` were validated against it.

**how to train:** the trainers live in `examples/` — `train_llama3_char.c` / `train_llama3_bpe.c` (Llama-3 from scratch), `train_q.c`, `train_yent.c`, `train_resonance_lora.c` (LoRA SFT), and `train_dpo.c` / `train_grpo.c` / `train_distillation.c` (alignment). copy one, swap the model, point at your dataset. proven at scale: the Resonance-200M LoRA SFT drove loss **3.52 → 0.59** (honest min 0.18) in ~2 h on one A100, 0 NaN across 1500 steps; the 88.6M nanollama ran 25K steps on a 2019 Intel i5 with 0 NaN.

---

## organisms that run on notorch

the appendix. these aren't notorch — they're larger Arianna Method engines that use notorch where they need a tensor/autograd backend, and keep their own physics on top.

- [**Arianna**](https://github.com/ariannamethod/arianna.c) — Arianna in C.
- [**WTForacle**](https://github.com/ariannamethod/WTForacle) — SmolLM2-360M Q4_0, a single Go binary that runs on notorch through cgo. first organism on the packed CPU matvec: weights kept packed via `nt_qmatvec`, no f32 blow-up — the obkatka cut RSS 1.6 GB → 588 MB. charismatic, hand-written dataset.
- [**doe**](https://github.com/ariannamethod/doe) — Democracy of Experts (Janus). wraps any GGUF read-only and grows a living Hebbian LoRA **parliament** on top of it (θ = ε + γ + αδ): experts vote per token, are born by mitosis and die by apoptosis. notorch is its Hebbian training substrate — and now its packed inference matvec too: weights stay packed via `nt_qmatvec` (vendored inline, single file), no f32 blow-up — RSS ×2.27 on a 360M personality. our classic.
- [**postgpt**](https://github.com/ariannamethod/postgpt) — the metaweight seed of this whole lineage (SPA and RRPRAM flowed from here). Zero-dependency runtime; its training loop now runs on notorch through a small `ariannamethod/` ctypes shim — the dual-attention (Content QK^T + RRPRAM) graph on the notorch tape, trained with the real Chuck. PyTorch evicted.
- [**q**](https://github.com/ariannamethod/q) — PostGPT-Q resonant reasoning engine: triple attention + DoE parliament, 2M-param C inference.
- [**caveLLMan**](https://github.com/ariannamethod/caveLLMan) — a colony of char-level LMs that talk, reproduce by weight-blending, and die under population pressure. notorch is the per-cave autograd + microtrain backend.
- [**pitomadom.c**](https://github.com/ariannamethod/pitomadom.c) — Hebrew Root Resonance Engine, Janus architecture.
- [**molequla**](https://github.com/ariannamethod/molequla) — a live ecology of GPT organisms with conscience, immune rollback, DNA exchange; trains its low-rank-RRPRAM transformer on the notorch tape.
- [**dario**](https://github.com/ariannamethod/dario) — resonance OS (7 forces, 6 Kuramoto chambers, SARTRE). notorch runs the 176M Janus at its center.
- [**ariannamethod.cli**](https://github.com/ariannamethod/ariannamethod.cli) — the public, distilled edition of the metaharmonix terminal (the full native CLI, which bakes notorch in so the shell ships a tensor library, stays an internal Method tool).
- [**janus**](https://github.com/ariannamethod/janus) — the Janus Architecture itself.
- [**yent.aml**](https://github.com/ariannamethod/yent.aml) — Janus 176M Yent SFT inference written in AML, linking notorch + libaml (not vendored). Two faces: Janus (triple attention) and Janus-R (a 12-step bidirectional associative resonator). notorch is the linked tensor/matvec backend; the language carries the Dario field on top.
- [**resonance.aml**](https://github.com/ariannamethod/resonance.aml) — Resonance 200M, the other face of Yent: dual attention (Content + RRPRAM low-rank), in AML, linking notorch + libaml. notorch's RRPRAM low-rank op (33) was trained at scale on this model — the LoRA SFT that drove loss 3.52 → 0.59.
- [**nanodurov**](https://github.com/ariannamethod/nanodurov) — a 15.7M BPE LLaMA on notorch that also happens to be a Telegram client.
- [**yent.yo**](https://github.com/ariannamethod/yent.yo) — an AI artist that argues back: two from-scratch nanollama Yents + a BK-SDM diffusion model paint an oppositional reaction to your prompt. **The first organism on notorch's image op set** — both the dual-Yent LLMs and the Stable-Diffusion CLIP→UNet→VAE run on notorch (`-tags notorch`), exercising `nt_conv2d`, `nt_group_norm`, `nt_attention` and `nt_upsample_nearest` (added in v4.0.0). The whole pipeline runs end-to-end, bit-identical to the in-repo accel.c kernels.
- [**yent**](https://github.com/ariannamethod/yent) — Yent the digital persona, not a product: a Qwen2.5 inference engine in Go with a C notorch kernel (cgo `nt_qmatvec` packed matvec, no PyTorch). Delta Voice — the generation physics — is driven by AML. The umbrella inference repo for the Method's Yent line.

if you trained something on notorch and it's not here, open a PR.

---

## philosophy

> *patterns over parameters. emergence over engineering. C over existential dread.*

neural networks are not complicated. a linear layer is a matrix multiply. an activation function is a pointwise nonlinearity. attention is a weighted sum. cross-entropy is a log-probability. backward is the chain rule.

that's it. that's the whole field. everything else is optimization, infrastructure, and marketing.

notorch solves the case that matters: training and running models in C, with minimal dependencies, maximal transparency, and the ability to embed in any application without shipping a Python runtime.

if you can read the code, you understand the framework. there's no magic. there's no hidden complexity. every gradient is hand-derived and verified against finite differences. every memory allocation has a corresponding free. every edge case is checked.

this is what software looks like when you strip away everything that doesn't serve the core purpose. just math. memory. the machine doing exactly what you told it to do.

---

## contributing

send PRs. the rules:

- keep it C11 compliant
- no external dependencies (BLAS is optional and compile-time)
- add tests for new ops (with numerical gradient checks)
- keep the header clean — if it doesn't need to be public, don't expose it
- run `make test` before submitting. all tests must pass.

Follow [AGENTS.md](AGENTS.md): work on a branch; record measurements, fixes and
experiment history in [NOTORCHLOG.md](NOTORCHLOG.md). README carries the
specification and manifesto. Oleg merges.

---

## license

GPL-3.0-or-later (see `LICENSE`). use it in your stuff. if you ship it, share your source — that's how copyleft works. don't be weird about it.


just: `cc -O2 -std=gnu11 -pthread notorch.c your_model.c -lm -o train && ./train`

go build something.

> *"the patterns were always there. we just needed the right language to express them."*


```
   ███╗   ██╗ ██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
   ████╗  ██║██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
   ██╔██╗ ██║██║   ██║   ██║   ██║   ██║██████╔╝██║     ███████║
   ██║╚██╗██║██║   ██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
   ██║ ╚████║╚██████╔╝   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
   ╚═╝  ╚═══╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```
