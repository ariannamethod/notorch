# notorch.js — pure-JS / WebGPU port of notorch

> *"fuck torch"* — `notorch.js:7`

JavaScript port of the [notorch](https://github.com/ariannamethod/notorch)
C tensor library. Same API surface, same Chuck optimizer, same naming.
Different lifecycle: everything in **one ES module file** (no CPU/GPU
lib split), runs in Node and the browser, optional WebGPU matmul.

GPL-3.0+. Co-authored by Oleg Ataeff and Claude (Arianna Method).

---

## Why JS

The C path is the production line for organism training (Resonance,
Janus, Leo, Yent, Dario). JS is the **distribution** path:

- **Browser inference** — drop a `.bin` weight file + this module on a
  static site, run LoRA-adapted Resonance / Janus directly in the
  user's browser. No server, no CUDA. Just `await engine.matmulAsync(...)`
  on a WGSL tile-blocked matmul.
- **Browser SFT** — train a LoRA adapter against a base model the user
  holds locally; the data never leaves the device. Chuck is real here,
  not a toy.
- **Node prototyping** — fast iteration on architectures (no `make`,
  no rebuild loop) before committing to the C path.
- **Cross-load with C** — LoRA artifacts saved by `nt_lora_save` are
  byte-compatible with `loadLoRA` in JS, and vice versa. The notorch
  native `.bin` format reads/writes both ways.

---

## Install

No package manager, no build step. Single file.

```bash
curl -O https://raw.githubusercontent.com/ariannamethod/notorch/main/js-edition/notorch.js
```

Or clone the parent repo and import from `js-edition/notorch.js`.

Runs in:

- **Node 20+** with `--input-type=module` or the `.mjs` extension
- **Modern browsers** (Chrome / Safari / Firefox stable) as an ES
  module via `<script type="module">`

---

## Quick start

### Node — forward + autograd + Chuck step

```js
import { Notorch, Tensor, Chuck } from "./notorch.js";

const e = new Notorch();
e.tape.start();

const W = e.tape.param(Tensor.xavier([4, 3], 3, 4));    // trainable
const x = e.tape.leaf(Tensor.fromArray([1, 2, 3], [3]));

const yIdx    = e.matvec ? e.matmul(W, x) : e.matmul(W, x);
const lossIdx = e.crossEntropyLoss(yIdx, /*target*/1);

e.backward(lossIdx);
const opt = new Chuck(e, /*lr*/1e-3);
opt.step(e.get(lossIdx).data[0]);  // Chuck consumes scalar loss for EMA
```

### Browser — WebGPU matmul

```html
<script type="module">
import { Notorch, Tensor } from "./notorch.js";

const e = new Notorch();
await e.init();                  // probes navigator.gpu → e.hasWebGPU

const A = Tensor.rand([512, 256], 0.1);
const B = Tensor.rand([256, 128], 0.1);
const aIdx = e.tape.leaf(A);
const bIdx = e.tape.leaf(B);

// Tile-blocked WGSL kernel; transparently falls back to CPU.
const cIdx = await e.matmulAsync(aIdx, bIdx);
console.log(e.get(cIdx).shape);  // [512, 128]
e.cleanup();
</script>
```

### Assistant-only SFT loop (concept, mirrors `examples/train_resonance_lora.c`)

```js
import { Notorch, LoRAPair, Chuck } from "./notorch.js";

const e = new Notorch();
const pair = new LoRAPair(/*in*/768, /*out*/768, /*rank*/64, /*alpha*/128);
const opt  = new Chuck(e, /*lr*/1e-4);

for (let step = 0; step < 1500; step++) {
  // Build tape per step: paramFrozen(base) → pair.forward → ... → masked CE
  // Masked positions (prompt tokens) contribute neither to loss nor to grad.
  const lossIdx = buildForwardWithMaskedCE(e, pair, batch);
  e.backward(lossIdx);
  opt.step(e.get(lossIdx).data[0]);
}

// Save adapter — byte-compatible with `nt_lora_save` on the C side.
const blob = saveLoRA([pair], /*L*/1, /*T*/1, ["wq"]);
```

`seqCrossEntropyLossMasked(logits, targets, mask, T, V)` is the SFT
loss — zero contribution from positions where `mask[t] === 0`,
matching the C `nt_seq_cross_entropy_masked` pattern.

---

## Architecture

Three exported classes carry the model:

| Class       | Role                                                                  |
|-------------|-----------------------------------------------------------------------|
| `Tensor`    | `Float32Array` + shape. Pure data carrier — no autograd state.        |
| `Tape`      | Reverse-mode autograd. Records ops, walks `backward(lossIdx)`.        |
| `Notorch`   | Engine: forward methods record on `this.tape`; WebGPU device + pool.  |

Optimizers: `SGD`, `Chuck`.
Schedules: `Schedule.cosine`, `Schedule.step`.
Inference helpers: `KVCache`.
Adapters: `LoRAPair`, `saveLoRA`, `loadLoRA`, `mergeLoRAInto`.
Loaders: `loadNotorchBin`, `loadSafetensors`, `saveNotorchBin`.
Tokenizers: `CharTokenizer`, `BPETokenizer`.

---

## Op parity table

35 op codes (matching C `notorch.h:92-126` numbering) + 8 JS-specific
extensions.

| OP | # | Forward method | Notes |
|----|---|----------------|-------|
| NONE                | 0  | (leaf marker)                                  | |
| MATVEC              | 1  | (internal)                                     | |
| ADD                 | 2  | `add(a, b)`                                    | |
| MUL                 | 3  | `mul(a, b)`                                    | |
| SCALE               | 4  | `scale(a, k)`                                  | k is a JS scalar |
| SOFTMAX             | 5  | `softmax(x)`                                   | vector form |
| RMSNORM             | 6  | `rmsnorm(x, γ, eps)`                           | |
| SILU                | 7  | `silu(x)`                                      | |
| CROSS_ENT           | 8  | `crossEntropyLoss(logits, target)`             | single position |
| EMB_LOOKUP          | 9  | `embedding1(W, tokenId)`                       | single token |
| MATMUL              | 10 | `matmul(A, B)` / `matmulAsync(A, B)`           | CPU tiled + WebGPU |
| SEQ_EMBED           | 11 | (backward only — emit via `embedding`)         | |
| SEQ_MATVEC          | 12 | `seqLinear(W, x, T)`                           | |
| SEQ_RMSNORM         | 13 | `seqRmsnorm(x, γ, T, D, eps)`                  | |
| CAUSAL_ATTN         | 14 | (use `attention` with n_heads=1)               | |
| SEQ_CROSSENT        | 15 | `seqCrossEntropyLoss(logits, targets, T, V)`   | |
| MH_CAUSAL_ATTN      | 16 | `attention(q, k, v, T, headDim)`               | |
| GEGLU               | 17 | `geglu(x, W1, W2, T, dIn, dOut)`               | Gemma-3 fused FFN |
| ROPE                | 18 | `rope(x, T, headDim, freqBase)`                | |
| DROPOUT             | 19 | `dropout(x, p)`                                | mask saved on tape |
| LAYERNORM           | 20 | `layernorm(x, γ, β, eps)`                      | |
| SEQ_LAYERNORM       | 21 | `seqLayernorm(x, γ, β, T, D, eps)`             | |
| GELU                | 22 | `gelu(x)`                                      | tanh approximation |
| GQA_ATTN            | 23 | `gqaCausalAttention(q, k, v, T, hD, nH, nKV)`  | Llama-3+ grouped-query |
| RRPRAM_ATTN         | 24 | `rrpramAttention(wr, x, v, T, E, nH, hD)`      | Resonance/Janus positional |
| CONCAT              | 25 | `concat(a, b, T)`                              | |
| SEQ_MATVEC_T        | 26 | `seqLinearT(W, x, T)`                          | transposed seq linear |
| SIGMOID             | 27 | `sigmoid(x)`                                   | |
| SCALE_BY_T          | 28 | `scaleByT(x, a)`                               | scalar-tensor scale (a is [1]) |
| SWIGLU              | 29 | `swiglu(g, u)` / `swigluFFN(x, W1, W2, W3, T)` | LLaMA-style FFN |
| BIT_LINEAR          | 30 | `bitLinear(W, x)`                              | BitNet 1.58, STE backward |
| BIT_SEQ_LINEAR      | 31 | `bitSeqLinear(W, x, T)`                        | BitNet 1.58 sequence |
| SEQ_CROSSENT_MASKED | 32 | `seqCrossEntropyLossMasked(l, t, m, T, V)`     | assistant-only SFT |
| RRPRAM_LR           | 33 | `rrpramLowrankAttention(wr, x, v, T, E, nH, hD)` | low-rank Wr_a × Wr_b |
| **RRPRAM_BCAST**    | 34 | _not yet implemented (parity with C — declared in `notorch.h` but unimplemented in `notorch.c`)_ | |

### JS-specific extensions (op codes 100+)

| OP        | #   | Method                              | Notes                       |
|-----------|-----|-------------------------------------|-----------------------------|
| SUB       | 100 | `sub(a, b)`                         | element-wise subtract       |
| DIV       | 101 | `div(a, b)`                         | element-wise divide         |
| NEG       | 102 | `neg(a)`                            |                             |
| TRANSPOSE | 103 | `transpose(a, dimA, dimB)`          | 2D/3D axis swap             |
| TANH      | 104 | `tanh(x)`                           |                             |
| RELU      | 105 | `relu(x)`                           |                             |
| EMBEDDING | 106 | `embedding(W, ids, T, D)`           | sequence embedding lookup   |
| MSE       | 107 | `mseLoss(pred, target)`             | mean-squared error          |

---

## Chuck optimizer

Self-aware Adam-shape optimizer with per-parameter dampening (ring
buffer over the last 16 gradient norms) and a global macro-stagnation
detector. Synced bit-for-bit with C `nt_tape_chuck_step` and the
upstream PyTorch reference at `iamolegataeff/chuck.optimizer`.

```js
const opt = new Chuck(engine, /*lr*/1e-3);
opt.step(lossValue);
```

`SGD(engine, lr, momentum)` is also exported for cases where you want
a vanilla baseline. AdamW exists on the C side as
`nt_tape_adamw_step` for legacy callers but is **not** ported to JS by
design — Chuck is the default optimizer here.

`Schedule.cosine(baseLr, warmupSteps, totalSteps, minLr)` and
`Schedule.step(baseLr, warmupSteps, stepSize, gamma)` return objects
with `.get()` (current LR) and `.advance()` (current LR + step++).

---

## LoRA — byte-compatible with C

`LoRAPair`, `saveLoRA`, `loadLoRA`, `mergeLoRAInto` mirror `nt_lora_*`.
The save format is **byte-compatible** with the C `nt_lora_save`:

```
[u32 magic 'LORA'][u32 version=1]
[u32 num_targets][per-target: u8 namelen, namelen × ASCII bytes]
[u32 num_layers][u32 rank]
[f32 alpha (raw IEEE-754 bytes — NOT alpha*1000)]
[u32 in_dim][u32 out_dim]
[for L in [0, num_layers): for T in [0, num_targets):
    A floats (rank * in_dim), B floats (out_dim * rank)]
```

Train a LoRA adapter in C on a pod, scp the artifact to a static
site, `loadLoRA` it in the browser — no conversion step.

> The `alpha*1000` line in `notorch.h:653` C docstring is stale.
> The actual format writes the raw `float32` bytes, as JS does here.

---

## WebGPU

Only `matmul` has a WGSL kernel today (16×16 tiled with workgroup-shared
A/B tiles, buffer pool re-use). All other ops run on CPU. Calls to
`matmulAsync` transparently fall back to the CPU path when WebGPU is
absent.

The C-side notorch has a known bug class (six instances fixed
2026-05-09..14): GPU forward outputs read as their stale CPU mirror
in backward, producing zero / NaN gradients. **The JS path does not
have this bug today** — the one GPU op (`matmulGPU`) copies the
output into the CPU mirror before the `tape.record` call (see
`notorch.js:2065`).

If you add more WebGPU forward kernels later, mirror that discipline:
copy the GPU output back into the CPU mirror **before** calling
`tape.record`, or the backward branch (which reads parent
`output.data` on CPU) will silently see zeros.

---

## Loaders

- `loadNotorchBin(arrayBuffer)` — C native `.bin` (magic `'NTOR'`).
  Layout: `[u32 magic][i32 n][per-tensor: i32 ndim, ndim × i32 shape,
  len × f32 data]`.
- `loadSafetensors(arrayBuffer)` — HuggingFace safetensors with F32
  dtype (other dtypes throw).
- `saveNotorchBin(tensors)` — writes a `Map<name, Tensor>` to the
  native `.bin` format.

GGUF is supported in JS via `loadGGUF(arrayBuffer)` — a GGUF v3 reader
(F16 + F32 dequant). `loadSafetensors` also works for HF F32 weights.

---

## Cross-references

- C source: `notorch.c`, `notorch.h` in the parent directory
- LoRA SFT trainer pattern: `examples/train_resonance_lora.c`
- Bug-class postmortem (read this if you add WebGPU forward kernels):
  `docs/POST_SFT_RESONANCE_ARIANNA_2026_05_11.md`

---

## License & attribution

GPL-3.0+. Co-authored by Claude (Arianna Method) and Oleg Ataeff.
The C notorch repo is the canonical source — JS lockstep-follows the
same op semantics and naming. When the C path adds an op, the JS path
catches up via a port commit; the parity table above is the
ground truth.
