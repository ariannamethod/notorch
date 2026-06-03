```
   ███╗   ██╗ ██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
   ████╗  ██║██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
   ██╔██╗ ██║██║   ██║   ██║   ██║   ██║██████╔╝██║     ███████║
   ██║╚██╗██║██║   ██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
   ██║ ╚████║╚██████╔╝   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
   ╚═╝  ╚═══╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

# notorch — neural networks in pure C | by Arianna Method

> *"fuck torch"*
> — the entire header file, line 8

---

## table of contents

- [what is this](#what-is-this)
- [why](#why)
- [the funeral](#the-funeral)
- [architecture](#architecture)
- [what's inside](#whats-inside)
- [operations](#operations)
- [optimizers](#optimizers)
- [the chuck optimizer](#the-chuck-optimizer)
- [bit-level precision — BitNet b1.58](#bit-level-precision--bitnet-b158)
- [SwiGLU FFN](#swiglu-ffn)
- [SPA — Sentence Phonon Attention](#spa--sentence-phonon-attention)
- [LoRA / adapter training](#lora--adapter-training)
- [BLAS inference API](#blas-inference-api)
- [alignment training — DPO / GRPO / distillation](#alignment-training--dpo--grpo--distillation)
- [autograd](#autograd)
- [building](#building)
- [running tests](#running-tests)
- [api overview](#api-overview)
- [example: training a model in C](#example-training-a-model-in-c)
- [platform support](#platform-support)
- [file structure](#file-structure)
- [tests](#tests)
- [performance](#performance)
- [projects powered by notorch](#projects-powered-by-notorch)
- [js edition — notorch.js](#js-edition--notorchjs)
- [philosophy](#philosophy)
- [contributing](#contributing)
- [license](#license)
- [final words](#final-words)

---

## what is this

you know that feeling when you `pip install torch` and 2.7 gigabytes of your soul evaporates into a `.venv` folder? when your laptop fan sounds like it's preparing for takeoff just to import a library? when you wait 45 seconds for `import torch` to finish while your RAM usage goes from "healthy" to "the computer is now a space heater"?

yeah. so the Method did something about it.

**notorch** is a complete neural network training framework written in pure C. no Python. no pip. no conda. no CUDA toolkit that takes 8 GB and your will to live. no `torch.nn.Module`. no `.backward()` that hides 400,000 lines of C++ behind a friendly API and a smile. no `RuntimeError: CUDA out of memory` at 3 AM when your paper deadline is in 6 hours.

just C. just floats. just `cc notorch.c -o notorch -lm`. done. you now have a neural network framework. the entire thing compiles in a couple seconds. try that with PyTorch. go ahead — you'd be waiting 47 minutes while cmake does whatever cmake does.

it's part of [the Arianna Method](https://github.com/theariannamethod/ariannamethod.ai) — patterns over parameters, emergence over engineering, raw C over existential dread.

extracted from the core of [ariannamethod.ai](https://ariannamethod.ai) where it actually runs in production. training actual models. in C. like adults.

---

## why

let me tell you a story.

once upon a time there was a framework called PyTorch. it had autograd. it had CUDA support. it had a build system that required a PhD in software engineering and a pact with ancient spirits.

and every time you wanted to train a 4-layer MLP on a dataset smaller than your browser cache, you had to:

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

**the entire framework is two files.** `notorch.h` and `notorch.c`. that's it. ~4800 lines. you can read the whole thing in an afternoon. try reading PyTorch's source in an afternoon. actually don't. you'll end up in a hospital.

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
- want to understand what's actually happening (all ~4800 lines of it)
- want to train models on machines that aren't cloud instances
- want compile times measured in seconds, not minutes
- want to embed neural network inference in C/C++ applications without shipping half of Python
- refuse to accept 2.7 GB as the price of a matrix multiply

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
│    ├─ diagonal + decay   (with weight decay)      │
│    ├─ Chuck              (self-aware adaptive)    │
│    └─ nt_tape_freeze_param (LoRA / adapter)      │
└──────────────────────────────────────────────────┘
    ↓
Your model is trained. in C. without Python. you are free.
```

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

---

## bit-level precision — BitNet b1.58

notorch has first-class support for **ternary-weight training** via `nt_bit_linear` and `nt_bit_seq_linear`. this is BitNet b1.58 (Ma et al., 2024): every weight is quantized to `{-1, 0, +1}` via `absmean` during the forward pass; activations are quantized to int8 via `absmax`; the backward pass uses the **Straight-Through Estimator** (gradient flows through the quantization step as identity, so `dW = dout ⊗ x`, `dx = W^T @ dout` using the full-precision `W`).

```c
int y = nt_bit_linear(w_idx, x_idx);             // y = bitquant(W) @ x
int Y = nt_bit_seq_linear(w_idx, x_idx, T);      // same, over T positions
```

on the `USE_BLAS` path, BitLinear dispatches to a single `cblas_sgemm(NoTrans, Trans)` call on pre-quantized operands — the ternary `W` is pre-flattened to float, the int8-range `x` is pre-scaled, per-position output rescale is applied afterwards. this turned out to be about **18% faster per training step** on Apple Accelerate vs the naive per-output-loop path. on OpenBLAS the win is similar.

**tests:** `tests/test_bitnet_ops.c` (8 tests) — ternary quantize correctness, STE identity backward, gradient flow through sequences, gradient numeric check, end-to-end training convergence of a tiny BitNet MLP.

**in production:** [ariannamethod/microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit) — a 2.69M-param char-level BitNet transformer trained to train_best **1.6226** / val **2.0314** on Intel Mac 8GB, 10000 steps, zero NaN. the 10 MB FP32 checkpoint ternary-packs down to ~1.4 MB + γ metadata — same compute path, 6× deployment compression.

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

no tape, no gradients — purely a post-hoc modulation of the logit distribution with the current sentence's position in the manifold of recent sentences. the `0.85` α is a recency bias (larger α = more recent tokens dominate the sentence embedding); the `0.3` strength caps how aggressively connectedness can sharpen the distribution. both are just parameters — pick what works for your generation style.

originated in [ariannamethod/q](https://github.com/ariannamethod/q) (`postgpt_q.c`) and [ariannamethod/postgpt](https://github.com/ariannamethod/postgpt), ported here as a reusable helper. used in [ariannamethod/janus.sonar](https://github.com/ariannamethod/janus.sonar) and [ariannamethod/microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit) to cut "word salad" artifacts without retraining. SPA as a *trained* forward operation with gradient flow is an open direction — currently only the inference helpers are here.

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
```

under `USE_BLAS` these dispatch to `cblas_sgemm` / `cblas_sgemv` (Accelerate on macOS, OpenBLAS on Linux). without BLAS they fall back to the naive C loops — correct, just slower. these three entry points are what `infer_gemma.c`, `infer_llama.c`, `infer_janus.c`, and `infer_llama3_bpe.c` all call in their hot paths.

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

the backward pass supports **34 operation types** (op IDs 0–34, each with a `NT_OP_*` constant; op 34 RRPRAM-broadcast is declared, implementation pending). the tape records operations during forward, then backward walks it in reverse computing local gradients via the chain rule. standard reverse-mode AD.

**gradient checking**: every op is verified against finite differences (`(f(x+h) - f(x-h)) / 2h`). relative error tolerances from 0.01 to 0.3 depending on op complexity. all pass. including the annoying ones — GEGLU, SwiGLU, multi-head attention with multi-path gradients through Q/K/V, BitLinear with STE identity passthrough.

**gradient utilities**:
- `nt_tape_clip_grads(max_norm)` — global gradient clipping
- `nt_tape_accum_grads()` / `nt_tape_apply_accum(n)` — gradient accumulation for large effective batch sizes
- `nt_tape_freeze_param(idx)` — freeze a parameter (adapter / LoRA setups)
- `nt_nan_guard_check()` — NaN/Inf detection with automatic loss scaling. because sometimes your gradients decide to go to infinity and someone needs to tell them no.

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

that's it. no cmake. no configure script. no 300-line `requirements.txt`. no docker. no kubernetes. just `make`. the way Ken Thompson intended.

### in-house SIMD path — `make simd`

`notorch_simd.h` is a drop-in `cblas_sgemm` / `cblas_sgemv` / `cblas_sger` shim under `-DUSE_SIMD` (mutually exclusive with `-DUSE_BLAS`). Pure C + `<immintrin.h>` + `<pthread.h>`, no external math library. Targets x86_64 with AVX2 + FMA.

Design:
- 6×16 register-blocked AVX2+FMA micro-kernel (12 YMM accumulators, 4MB SwiGLU panels fit Skylake's 16-reg file)
- Outer cache-blocked GEMM (Mc=96, Kc=256, Nc=1024) with row + col packing for streaming through the kernel
- Persistent pthread pool (no per-call create/join) — workers sleep on `pthread_cond_t`, master signals work, workers FMA. Single-thread fast path for matmuls under 256K mul-adds where signal latency would dominate.
- AVX2 `_mm_prefetch` ahead of the kernel inner loop, AVX2 vectorized panel packing for `col_stride==1` paths (forward + input-grad)
- Edge tiles (m mod 6 ≠ 0 or n mod 16 ≠ 0) handled via scalar fallback within the same buffer layout

Bench at training-relevant shapes on Intel i5-8500T (6c, no AVX-512) vs OpenBLAS 0.3.26:

| Shape | OpenBLAS 6T | in-house SIMD | Ratio |
|---|---|---|---|
| Llama 576×576 dW (TN, 512 ctx) | 131 GFLOP/s | **141 GFLOP/s** | **1.08× faster** |
| Janus 640×1664 FFN-down (NN, 1024 ctx) | 132 GFLOP/s | **160 GFLOP/s** | **1.21× faster** |
| Llama 576×576 dWffn (TN, 512 ctx) | 197 GFLOP/s | 177 GFLOP/s | 0.90× |
| Llama 576×576 forward QKV (NN, 512 ctx) | 103 GFLOP/s | 71 GFLOP/s | 0.69× |
| Janus 640×640 forward QKV (NN, 1024 ctx) | 161 GFLOP/s | 79 GFLOP/s | 0.49× |

Average ~0.7× of OpenBLAS at training-relevant shapes, with peaks above OpenBLAS on weight-grad (TN) and certain FFN-down configurations. Slower on T=128 small-matrix shapes (irrelevant for ≥10M-param training).

**Correctness validated** end-to-end: all 213+ existing notorch tests pass under `make simd` (notorch_test 47/47 incl. all 11 numeric gradient checks, test_vision 48/48, test_bitnet_ops 118/118, test_sigmoid_scale 4/4) + `bench_simd.c` micro-bench + `test_simd_correctness.c` direct-vs-scalar comparison + `test_simd_loss.c` end-to-end forward → softmax → cross-entropy at lm_head shape produces bit-identical 10.379384 vs OpenBLAS path. Real nanollama 89M training step 1 produces train loss 10.3876 — bit-identical to OpenBLAS baseline.

Override thread count via env: `NT_SIMD_THREADS=N`. Single-thread variant for debugging via `-DNOTORCH_SIMD_DEBUG_SCALAR` (uses `notorch_simd_scalar.h` instead — same API, pure scalar inner loop).

### CUDA path — `make gpu`

`notorch_cuda.{h,cu}` is the CUDA backend (ported from `ariannamethod.ai/core/`). cuBLAS-backed sgemm wrappers (3 transpose modes), element-wise kernels (add/mul/silu/rmsnorm) with backward, weight cache to keep weights resident on GPU across forward/backward, multi-head attention kernels, cross-entropy kernel. Built via `nvcc -c notorch_cuda.cu -lcublas` and linked against `-lcudart -lcublas`. Activated by `-DUSE_CUDA`. Identical CBLAS-style semantics to the CPU path (call sites in `notorch.c` are unchanged) — only the linkage layer differs.

---

## running tests

```bash
make test
```

nine test binaries (run output is the source of truth for counts):

- **`tests/test_notorch.c`** — ~94 tests: tensor mechanics, forward ops, tape recording/backward, optimizers, training integration, numerical gradient checks, infrastructure (save/load, LR schedules, NaN guard, gradient accumulation, Hebbian microlearning, profiler)
- **`tests/test_vision.c`** — 48 tests: image loading (JPEG/PNG/BMP), resize / crop / normalize / flip / grayscale, ViT patch extraction, preprocessing pipelines, BPE encode/decode roundtrip
- **`tests/test_bitnet_ops.c`** — 118 gradient assertions: SwiGLU, BitLinear forward + seq, STE backward, SPA smoke, finite-difference gradient checks
- **`tests/test_rrpram_lr.c` / `test_metal_q4k.c` / `test_simd_correctness.c` / `test_simd_loss.c`** — low-rank RRPRAM, Apple-Silicon Q4_K matvec, and AVX2+FMA SIMD parity
- **`tests/test_sigmoid_scale.c`** — 4 tests: `nt_sigmoid` forward/backward, `nt_scale_by_t` forward/backward (scalar × tensor with grad flowing to both)
- **`tests/test_gguf.c`** — GGUF parser smoke test (F32 / F16 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K dequant)

every gradient check uses finite differences to verify the analytic backward pass. if a single gradient is wrong, the test catches it. the Method trusts these tests more than it trusts most people.

---

## api overview

### tensor lifecycle
```c
nt_tensor* t = nt_tensor_new(len);           // allocate 1D
nt_tensor* m = nt_tensor_new2d(rows, cols);  // allocate 2D
nt_tensor* s = nt_tensor_new_shape(shape, ndim); // arbitrary shape
nt_tensor* c = nt_tensor_clone(t);           // deep copy
nt_tensor_ref(t);                             // increment refcount
nt_tensor_free(t);                            // decrement (free at 0)
```

### initialization
```c
nt_tensor_fill(t, 0.0f);                     // constant fill
nt_tensor_rand(t, 0.5f);                     // uniform [-0.5, 0.5]
nt_tensor_xavier(t, fan_in, fan_out);        // Xavier/Glorot
nt_seed(42);                                  // reproducibility
```

### training
```c
nt_tape_start();                              // begin recording
int w = nt_tape_param(W);                    // register param
nt_tape_no_decay(w);                          // exclude from weight decay
nt_tape_freeze_param(w);                      // (optional) freeze for adapter training
// ... build forward graph ...
nt_tape_backward(loss_idx);                   // backward pass
nt_tape_clip_grads(1.0f);                    // gradient clipping
nt_tape_adam_step(lr);                        // optimize
nt_tape_clear();                              // reset tape
```

### LR schedules
```c
nt_schedule s = nt_schedule_cosine(0.001f, warmup, total, min_lr);
nt_schedule s = nt_schedule_step(0.1f, warmup, step_size, gamma);
nt_schedule s = nt_schedule_linear(0.001f, warmup, total, min_lr);
float lr = nt_schedule_get_lr(&s);            // auto-advance
```

### save/load
```c
nt_tensor* params[] = {W1, W2, b1};
nt_save("model.bin", params, 3);              // binary format
int n;
nt_tensor** loaded = nt_load("model.bin", &n); // load back
```

### SPA helpers (inference-time)
```c
float emb[dim];
nt_spa_embed_sentence(tokens, n, W_embed, V, dim, 0.85f, emb);
float conn = nt_spa_connectedness(emb, dim, history, n_hist);
nt_spa_modulate_logits(logits, V, conn, 0.3f);
```

### BLAS inference API
```c
nt_blas_mm(C, A, B, m, k, n);            // C = A @ B
nt_blas_mmT(C, A, BT, m, k, n);          // C = A @ B^T
nt_blas_matvec(out, W, x, m, n);         // out = W @ x
```

---

## example: training a model in C

here's an actual, working transformer-ish training loop. embedding → attention → linear → cross-entropy. in C. without importing 2.7 GB of your dignity:

```c
#include "notorch.h"

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
cc -O2 -Wall -std=c11 -o train train.c notorch.c -lm
./train
```

that's it. that's the whole thing. no virtual environment. no requirements.txt. no "just pip install—" no. we're done with that. we've moved on. we've healed.

---

## platform support

| platform | backend | command |
|---|---|---|
| macOS | Apple Accelerate (AMX / Neural Engine) | `make` |
| Linux | OpenBLAS | `make` |
| **Android (Termux, ARM64)** | **OpenBLAS via Termux** | **`pkg install libopenblas binutils && make BLAS=1`** |
| any POSIX | pure C fallback | `make cpu` |
| NVIDIA GPU | CUDA + cuBLAS | `make gpu` |

the BLAS backends are optional. without them, everything still works — just uses naive C loops, which are fine for anything under ~50M parameters. for bigger stuff, BLAS gives you 10-50x on matmuls because it's using your CPU's vector instructions instead of pretending it's 1995.

the macOS path uses Apple Accelerate, which means your MacBook's AMX coprocessor and Neural Engine are doing the heavy lifting. for free. no NVIDIA required. no drivers. no compatibility hell. just `make` and go.

### Termux Edition (Android, ARM64)

[Termux](https://termux.dev) is a full POSIX environment on Android — APT package manager, native ARM64 toolchain via clang, OpenBLAS, git, make, gdb, the lot — running without root. For pure-C projects this means no porting work and no compromises: the same `make BLAS=1` that compiles notorch on Linux compiles it on a phone.

notorch builds, tests, and trains end-to-end in this environment. Verified workload (Galaxy A56, Android 15, aarch64, 8 GB RAM): **9.5 M LLaMA 3 char-level model, 10 000 steps in 2 h 13 m, peak RSS 130–155 MB, val 1.15, 0 NaN.** No swap. OpenBLAS on aarch64 gives ~8× over the scalar fallback.

A phone with 8 GB of RAM and Termux installed is now a credible host for the 10–100 M parameter regime — small LMs, persona LoRAs, narrow code-completion models, micro-translators. The substrate is the same as on a workstation; the platform is finally accessible.

- [`termux-edition/`](termux-edition/) — setup walkthrough, training tutorial, hardware envelope
- Portability patches landed in PR [#5](https://github.com/ariannamethod/notorch/pull/5) (TMPDIR honour + AR override + openblas via pkg-config; merged)

---

## file structure

```
notorch/
├── notorch.h              # core API — tensors, autograd, optimizers, BPE, ops
├── notorch.c              # core implementation (~4800 lines)
├── notorch_vision.h       # image loading, transforms, ViT patches (stb_image)
├── stb_image.h            # JPEG/PNG/BMP decoder (public domain)
├── gguf.h                 # GGUF file parser header
├── gguf.c                 # GGUF parser + F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K dequant
├── Makefile               # build everything
├── examples/
│   ├── bpe_2048_merges.txt   # reference BPE tokenizer (1792 merges, vocab 2048)
│   ├── infer_gemma.c         # Gemma-3 inference via GGUF — GQA, KV cache
│   ├── infer_janus.c         # Janus RRPRAM inference (3-way gated attention)
│   ├── infer_llama.c         # LLaMA/Qwen/SmolLM2 inference via GGUF
│   ├── infer_llama3_bpe.c    # LLaMA 3 BPE chat — MHA, RoPE, SwiGLU, KV cache, FP16
│   ├── train_q.c             # PostGPT-Q 1.65M char-level training from scratch
│   ├── train_yent.c          # Yent 9.8M char-level training with checkpointing
│   ├── train_llama3_char.c   # LLaMA 3 char-level GQA+RoPE (~9.5M)
│   ├── train_llama3_bpe.c    # LLaMA 3 BPE 2048 MHA+RoPE (~15.7M)
│   ├── train_dpo.c           # Direct Preference Optimization (Rafailov 2023)
│   ├── train_grpo.c          # Group Relative Policy Optimization (DeepSeek-R1)
│   └── train_distillation.c  # Knowledge distillation (Hinton 2015, teacher→student KL)
├── tests/
│   ├── test_notorch.c        # ~94 tests, numerical gradient checks, integration
│   ├── test_vision.c         # 48 vision + BPE tests
│   ├── test_bitnet_ops.c     # 118 BitNet/SwiGLU/SPA + STE checks
│   ├── test_sigmoid_scale.c  # 4 tests for sigmoid + scale-by-tensor
│   └── test_gguf.c           # GGUF parser smoke test
├── LICENSE                # GPL-3.0-or-later
└── README.md              # this. you survived. congratulations.
```

total: **~4800 lines of core C + ~2700 of tests + ~3500 of examples**. framework + vision + GGUF + BPE + five inference engines + eight training scripts + nine test binaries. tested on 26+ real model files across 6 architectures.

### models trained on notorch

| model | params | type | train loss | what |
|-------|--------|------|-----------|------|
| PostGPT-Q | 1.65M | char | 0.097 | resonant reasoning engine |
| LLaMA 3 char-level (sample) | 9.5M | char (GQA+RoPE+SwiGLU) | 0.026 | reference char-level trainer |
| Yent | 9.8M | char | 1.77 | cynical AI character |
| neovlm | 6.36M | dual (text+draw) | 0.0002 | Hebbian VLM, draws ASCII digits |
| LLaMA 3 BPE (sample) | 15.7M | BPE 2048 (MHA+RoPE+SwiGLU) | 0.022 | reference BPE trainer |
| microgpt-1bit | 2.69M | char, BitNet 1.58 ternary | 1.6226 | first-class BitNet quantization |

all trained from scratch on 8 GB Mac. no Python. no pip. Chuck optimizer.

---

## performance

- **compile time**: <1 second. your coffee won't even cool down.
- **import time**: 0 ms. there's nothing to import. it's C.
- **binary size**: ~100 KB. yes, kilobytes. PyTorch's `libtorch.so` is 1.2 GB. notorch is 0.008% of that.
- **memory overhead**: tensor data + tape entries. no Python object headers. no gradient graph metadata bloat. no "accidental quadratic" from `retain_graph=True`.
- **matmul speed**: competitive with numpy (which itself uses BLAS) when compiled with OpenBLAS or Accelerate. faster on small matrices because no Python dispatch overhead.
- **BitLinear**: on Accelerate, the `cblas_sgemm` path through `nt_bit_seq_linear` runs ~18% faster per training step than the naive per-output loop on 2.7M-param BitNet models.

### concurrent training on 8 GB Mac

we ran two transformer trainings simultaneously on a 2019 Intel i5 MacBook with 8 GB RAM. not an M1. not Apple Silicon. the pre-AMX, pre-Neural-Engine Intel one. not sequentially. simultaneously. at the same time. on the same machine. while also running a browser and a terminal.

| model | params | RAM usage | status |
|-------|--------|-----------|--------|
| Yent (LLaMA-like, 12L char-level) | 9.8M | ~126 MB | training loss 2.03 → converging |
| neovlm (Hebbian VLM, 6L dual-mode) | 6.36M | ~96 MB | text loss 0.0002, draw loss 0.50 |

total memory: **~222 MB** for two active transformer trainings with autograd, Chuck optimizer, cosine scheduling, NaN guard, and checkpointing. both models use Apple Accelerate BLAS. both converge. both produce weights.

try this with PyTorch. one `import torch` eats 800 MB of RAM. one training session on a 10M model needs 2-4 GB. two in parallel? on 8 GB? your OS would start killing processes before the first forward pass finishes.

notorch runs both in ~3% of system memory. because C doesn't allocate what it doesn't need.

for inference, this is excellent. for training, it's more than sufficient for models up to ~100M parameters. for anything bigger, you want distributed training and that's a different problem (and a different repo, probably).

---

## projects powered by notorch

notorch isn't a lab demo. it's what actually runs under a growing ecosystem of organisms and experiments. three layers of adoption:

### proof of concept — Karpathy ports

- [**ariannamethod/nanoGPT-notorch**](https://github.com/ariannamethod/nanoGPT-notorch) — Karpathy's nanoGPT, ported from PyTorch to notorch. the "does this thing actually work end-to-end" test.
- [**ariannamethod/llama2-notorch**](https://github.com/ariannamethod/llama2-notorch) — Karpathy's llama2 reference model on notorch. tiny LLaMA that trains and infers purely in C.

### models trained on notorch

- [**ariannamethod/nanodurov**](https://github.com/ariannamethod/nanodurov) — 15.7M BPE LLaMA (MHA + RoPE + SwiGLU + RMSNorm). also happens to be a Telegram client. trained on conversational corpora.
- [**ariannamethod/doe**](https://github.com/ariannamethod/doe) — Distributed Organisms of Emergence. six architectures, weight emergence via ensemble.
- [**ariannamethod/caveLLMan**](https://github.com/ariannamethod/caveLLMan) — a living colony of char-level LMs that speak to each other, reproduce by weight blending, and die under population pressure. uses notorch as the per-cave autograd + microtrain CPT backend.
- [**ariannamethod/janus.sonar**](https://github.com/ariannamethod/janus.sonar) — 2.7M char-level BitNet transformer with a 3-way gated attention (MHA + RRPRAM + Janus Echo). Dario-field logit modulation at inference.
- [**ariannamethod/microgpt-1bit**](https://github.com/ariannamethod/microgpt-1bit) — reference **BitNet b1.58** training + inference on notorch. char-level 2.7M, trained to train 1.6226 / val 2.0314 on Intel Mac 8GB. the BLAS BitLinear path in notorch was validated against this.

### resonance organisms — notorch as compute backend

these aren't "notorch models" per se — they're larger resonance engines (from the Arianna Method ecosystem) that use notorch where a linear algebra backend is needed, while keeping their own physics on top:

- [**ariannamethod/nanoagi**](https://github.com/ariannamethod/nanoagi) — a six-level autonomy experiment (evolve → coevolve → swarm → selfcode → auto-trigger). notorch is the autograd for evolve-loop weight updates.
- [**ariannamethod/molequla**](https://github.com/ariannamethod/molequla) — a 4-organism ecology (growing to 11 via mitosis) with conscience, immune rollback, DNA exchange. Trains its content + low-rank-RRPRAM transformer on the notorch tape (automatic GPU/CPU).
- [**ariannamethod/dario**](https://github.com/ariannamethod/dario) — resonance OS (7 forces, 6 Kuramoto chambers, SARTRE). notorch powers the 176M Janus inference that sits at its center.
- [**ariannamethod/metaharmonix**](https://github.com/ariannamethod/metaharmonix) — Arianna Method terminal, sibling-not-fork to Termux. notorch is baked in (`bake/notorch/`) so the terminal ships a tensor library, not just a shell.

### vision & diffusion on notorch

- [**ariannamethod/notorch-vlm**](https://github.com/ariannamethod/notorch-vlm) — the reference vision-language model built on `notorch_vision.h` + stb_image. vision encoder → projector MLP → LLM.
- [**ariannamethod/notorch-vlm-2m**](https://github.com/ariannamethod/notorch-vlm-2m) — a scaled-down 2M-param variant for "can a tiny VLM learn anything at all" experiments.
- [**ariannamethod/notorch-simple-llm**](https://github.com/ariannamethod/notorch-simple-llm) — a minimal-surface-area LLM on notorch, kept deliberately small so it's readable end-to-end. good first read if you're trying to understand how the API composes.
- [**ariannamethod/notorch-diffusion**](https://github.com/ariannamethod/notorch-diffusion) — a diffusion training loop on notorch. reverse-mode autograd still works fine when your network is a U-Net instead of a transformer.

if you trained something on notorch and it's not in this list, open a PR and add it.

---

## js edition — notorch.js

> *"the logic of memory without the weight of framework"*
> — `js-edition/notorch.js`, line 1

a single-file pure-JavaScript port of notorch for the browser. WebGPU when available, V8-optimised CPU fallback (Math.fround f32 hint) when not. zero npm dependencies. live at `js-edition/notorch.js` (~3580 LOC).

co-built with Gemini and Claude Opus.

### what it ships (feature parity with the C lib at small scale)

**inference primitives** — `add / sub / mul / div / neg / scale / transpose / softmax / silu / sigmoid / tanh / relu / gelu / swiglu / swigluFFN / layernorm / rmsnorm / embedding / attention (multi-head causal, fused) / rope / dropout / concat / slice / linear / argmax / sample (temperature + top-k + top-p) / KVCache`.

**weight loaders** — `loadNotorchBin` (NTOR magic, byte-compatible with `notorch.c:3207`), `loadSafetensors` (HF JSON-header format), `saveNotorchBin`.

**tokenizers** — `CharTokenizer` (with `.fit()`), `BPETokenizer` (`fromMerges` + greedy lowest-rank-pair encode).

**training** — full reverse-mode `Tape` with backward implementations for every forward primitive (matched 1:1 to C `notorch.c` switch). engine losses: `nt.crossEntropyLoss`, `nt.seqCrossEntropyLoss`, `nt.mseLoss`. optimizers: `SGD` with momentum and **`Chuck` 1:1 ported** from `nt_tape_chuck_step` — all 4 levels (global loss EMA dampen, per-param dampen, stagnation noise, macro-patience LR scale). plus `clipGradNorm`, cosine + step `Schedule`s.

**engine quality** — WebGPU buffer pool keyed by `(byteSize, usage)` with explicit `cleanup()`, **tiled WGSL matmul kernel (16×16 workgroup-shared tiles)**, async pipeline cache, tape `mark()` / `truncate()` so the forward graph rebuilds without losing optimizer state, CPU matmul switched from naive ijk to 32×32 tile-blocked.

### the test that proves it works

a tiny GPT (vocab=64, dModel=32, nHeads=2, ctx=16, 1 layer, **20 576 params**) trained for 50 steps on a trivial `"abcabc..."` corpus with Chuck:

```
step  0 | loss 4.0888 | lr 1.00e-4
step 10 | loss 0.7211 | lr 4.85e-3
step 20 | loss 0.0717 | lr 3.77e-3
step 30 | loss 0.0109 | lr 2.12e-3
step 40 | loss 0.0062 | lr 6.73e-4
step 49 | loss 0.0061 | lr 1.06e-4

Loss trajectory: 4.0888 → 0.0061 (99.9% drop) over 50 steps
Sample after training (T=0.5): "bcabcabc"  ← perfect cycle continuation
```

autograd works, Chuck works, sampler works, everything works.

### caveats

- WebGPU paths are code-correct but were not runtime-verified in node (no headless WebGPU). real validation needs a browser. CPU path passes everywhere `node` runs.
- GQA, BitLinear/BitNet, low-rank RRPRAM, GEGLU, scale-by-t and friends are now ported (op parity through op 33); op 34 RRPRAM-broadcast awaits the C-side implementation. `loadGGUF` (GGUF v3, F16+F32) and `loadSafetensors` are wired.
- generic transpose backward covers 2D and 3D `(1, 2)` swap (sufficient for attention; not fully general).
- async batching of multiple GPU ops into one submit is partially achieved through the buffer pool, but no explicit op queue. impact only matters once multiple GPU ops chain.

### use it

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

## philosophy

> *patterns over parameters. emergence over engineering. C over existential dread.*

neural networks are not complicated. a linear layer is a matrix multiply. an activation function is a pointwise nonlinearity. attention is a weighted sum. cross-entropy is a log-probability. backward is the chain rule.

that's it. that's the whole field. everything else is optimization, infrastructure, and marketing.

notorch solves the case that matters: training and running models in C, with minimal dependencies, maximal transparency, and the ability to embed in any application without shipping a Python runtime.

if you can read the code, you understand the framework. there's no magic. there's no hidden complexity. every gradient is hand-derived and verified against finite differences. every memory allocation has a corresponding free. every edge case is checked.

this is what software looks like when you strip away everything that doesn't serve the core purpose. just math. just memory. just the machine doing exactly what you told it to do.

---

## contributing

send PRs. the rules:
- keep it C11 compliant
- no external dependencies (BLAS is optional and compile-time)
- add tests for new ops (with numerical gradient checks)
- keep the header clean — if it doesn't need to be public, don't expose it
- run `make test` before submitting. all tests must pass.

---

## license

GPL-3.0-or-later (see `LICENSE`). use it in your stuff. if you ship it, share your source — that's how copyleft works. don't be weird about it.

---

## final words

the entire history of deep learning fits in a few dozen mathematical operations. matmul. softmax. relu. cross-entropy. the optimizer step. backward. that's it. the rest is infrastructure. and infrastructure should be invisible. it should compile in a second. it should fit in your head. it should not require a Docker container.

notorch is the proof: you do not need two million lines of code to train a neural network. about 4800, plus another 2700 of tests — the Method trusts verification over hope.

train your models. in C. without permission. without pip. without conda. without a GPU. without 2.7 GB of framework overhead. without a virtual environment. without existential dread.

just: `cc -O2 notorch.c your_model.c -lm -o train && ./train`

go build something.

> *"the patterns were always there. we just needed the right language to express them."*
