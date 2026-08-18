# notorch.js — pure-JS / WebGPU port of notorch

> *"fuck torch"* — `notorch.js:7`

JavaScript port of the [notorch](https://github.com/ariannamethod/notorch)
C tensor library. Same API surface, same Chuck optimizer, same naming.
Different lifecycle: everything in **one ES module file** (no CPU/GPU
lib split), runs in Node and the browser, optional WebGPU matmul.

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
Packed kernels: `qmatvec` (exact), `qmatvecI8` / `qmatvecI8Rows` / `quantAct`
(int8-activation fast path).
Tokenizers: `CharTokenizer`, `BPETokenizer`.

---

## Op parity table

37 C op codes (0–36, matching C `notorch.h:91-127` numbering) + 7 JS-specific
extension codes (100+). RELU is C op 35 in both runtimes.

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
| RRPRAM_BCAST        | 34 | `rrpramBroadcastAttention(wr, x, v, T, E, nH, hD, rank)` | broadcast RRPRAM — canonical Janus, sc=1/√hd |
| RELU                | 35 | `relu(x)`                                       | C op 35 |
| SEQ_GATE            | 36 | `seqGate(x, g, T, nm, gi)`                      | per-position mechanism gate |

### JS-specific extensions (op codes 100+)

| OP        | #   | Method                              | Notes                       |
|-----------|-----|-------------------------------------|-----------------------------|
| SUB       | 100 | `sub(a, b)`                         | element-wise subtract       |
| DIV       | 101 | `div(a, b)`                         | element-wise divide         |
| NEG       | 102 | `neg(a)`                            |                             |
| TRANSPOSE | 103 | `transpose(a, dimA, dimB)`          | 2D/3D axis swap             |
| TANH      | 104 | `tanh(x)`                           |                             |
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

GGUF in JS via `loadGGUF(arrayBuffer)` reads GGUF v3 and dequantizes
**F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K** to f32 on load. the block routines
mirror `gguf.c` byte-for-byte and are verified against the C path by
`test_gguf_dequant.mjs` — Q4_K/Q6_K/Q8_0/Q4_0 match C to ~5e-9 across real
models (Qwen3-0.6B, smallcoder Q8_0, wtf360 Q4_0), Q5_0 to 5e-8 on
nano_arianna Q4_K_M, whose 32000×576 `token_embd.weight` is Q5_0.
`loadSafetensors` works for HF F32 weights.

`loadGGUF` still expands every tensor to f32 on load, which is 4 B/weight
where Q4_K on disk is ~0.55 — a 170 MB file becomes north of a gigabyte in a
tab. `qmatvec(out, Wq, dtype, x, m, k)` is the way out and the port of C
`nt_qmatvec`: it dots a row straight out of the packed bytes, unpacking one
block at a time into locals, so no dense tensor is ever built. It covers F32,
F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K and returns -1 on a dtype or a `k` it has no
kernel for, same contract as C.

`qmatvecI8` is the same matvec with the activation quantized to per-32 int8 and
the dot accumulated in integers — C's `nt_qmatvec_i8`, the llama.cpp / MNN fast
path. Approximate by construction, `qmatvec` stays the exact reference, and the
gate holds it to the C tolerance of 2e-2 (measured 3.1e-3 to 4.1e-3). Where C
banks on SDOT and VNNI, JS banks on the type: an int32 accumulator stays in V8's
small-integer form instead of running an f32 dependency chain, and no weight is
ever widened to a float. `quantAct` + `qmatvecI8Rows` are the split entry, for a
consumer dotting one row against many matrices — a MoE against its experts —
that should quantize the row once rather than once per matrix.

Two details there are load-bearing and neither is visible on random input.
C's `lrintf` rounds ties to even; `Math.round` rounds them up, and on activations
that land on exact halves that is a whole int8 step — 16 of 32 in the test case.
And C holds the scale, its reciprocal, and the scaled activation at f32 where JS
would carry f64: 13 of 6.4M random activations move by one step when that is
dropped. Both are pinned by searched-out literals in `test_qmatvec.mjs`, because
a uniform-random check finds the second maybe one run in thirty.

`loadGGUF(ab, { packed: true })` stops expanding the quantized families
altogether: the tensor keeps a `Uint8Array` view onto the file's own bytes, and
`Tensor.fromPacked` carries the GGUF dtype alongside it. `seqLinear` and
`embedding` branch on that — the first through `qmatvec`, the second through
`dequantRow`, which decodes one row where the dense path decodes the table.
F32 and F16 still expand: their consumers here are norms and other non-matvec
ops that read dense data, and F16 only ever buys a factor of two.

On nano_arianna Q4_K_M, 69.4 MB on disk, 93 of 120 tensors packed:

| | dense | packed |
|---|---|---|
| f32 bytes built at load | 354,546,432 | 62,208 |
| heap + external after load | +339.8 MB | +1.9 MB |
| load time | 190 ms | 5 ms |
| peak RSS, 8-token generation | 538 MB | 287 MB |
| wall time, 8-token generation | 4.68 s | 6.55 s |

Generation is byte-for-byte identical across 24 greedy tokens — that is the
gate, and it is what makes the memory number mean anything. The load collapses
to 5 ms because the packed tensor is a view onto the buffer already read, not a
copy of it.

The 40% slower generation is real and is the honest trade as it stands:
`qmatvec` re-decodes a block on every pass where the dense path decoded once at
load. `qmatvecI8` is the answer and already exists, but `seqLinear` does not
reach for it yet — that is a separate step with its own gate, since an
approximate kernel can move the tokens and the check above would have to change
shape.

Packed weights are inference-only. `SEQ_MATVEC` and `EMBEDDING` backward refuse
them by name rather than reading an empty `data` and filling the gradient with
NaN.

Then the WebGPU quant matvec (mirroring the C Metal `nt_metal_q4k_matvec`).

To run the parity test:
```bash
cc -O2 -I. tests/gguf_dequant_ref.c gguf.c -lm -o /tmp/gguf_dequant_ref
/tmp/gguf_dequant_ref model.gguf token_embd.weight blk.0.attn_q.weight > ref.json
node js-edition/test_gguf_dequant.mjs model.gguf ref.json   # → JS_DEQUANT_OK
```

The lightweight JS/C op-contract gate is:
```bash
make test_js
# or: cd js-edition && npm test
```

`test_qmatvec.mjs` (part of both of the above) checks the packed kernels
against an independent dequant oracle at the C threshold of 1e-3. Pass
`--cref` to add the second hand — the C kernel run on the identical bytes
rather than on a second generator believed to agree:
```bash
cc -std=c11 -O2 -I. tests/js_qmatvec_ref.c notorch.c -lm -o /tmp/js_qmatvec_ref
cd js-edition && node test_qmatvec.mjs --cref /tmp/js_qmatvec_ref
# → each format PASS with C ~1e-6 (f64 accumulator in JS vs f32 in C)
```

### Running a GGUF end-to-end (`infer_gguf.mjs`)

`infer_gguf.mjs` is the full RUN on notorch.js: load a quantized GGUF, build the
byte-level BPE tokenizer **from the GGUF**, run the llama/mistral transformer
forward (embed → RMSNorm → q/k/v → interleaved RoPE → GQA causal attention →
SwiGLU FFN → tied/output projection), and generate. Same module the browser
loads — no Python, no llama.cpp.

```bash
node js-edition/infer_gguf.mjs model.gguf "The capital of France is" 6 0
```

**Verified against the C engine** (`examples/infer_gguf_metal`): on
SmolLM2-135M-Q4_K_M, greedy, the JS RUN produces *"The capital of France is
Paris. Paris is a city"* — **token-for-token identical** to the C output. The
forward runs on the CPU path today; a packed / WebGPU quant matvec (mirroring
the C Metal `nt_metal_q4k_matvec`) is the next step, so very large models still
favour the C edition. RoPE is interleaved — correct for llama/mistral; qwen2/qwen3
(NEOX RoPE + per-head q/k-norm) is the next arch to wire in.

---

## Cross-references

- C source: `notorch.c`, `notorch.h` in the parent directory
- LoRA SFT trainer pattern: `examples/train_resonance_lora.c`
- Bug-class postmortem (read this if you add WebGPU forward kernels):
  `docs/POST_SFT_RESONANCE_ARIANNA_2026_05_11.md`

---

## License & attribution

GPL-3.0+ — by Arianna Method.
The C notorch repo is the canonical source — JS lockstep-follows the
same op semantics and naming. When the C path adds an op, the JS path
catches up via a port commit; the parity table above is the
ground truth.
