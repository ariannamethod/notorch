// notorch.js — "The logic of memory without the weight of framework"
// Pure JS/WebGPU AI engine. No dependencies.
// (C) 2026 Arianna Method contributors
//
// Port of the C library at github.com/ariannamethod/notorch.
// Same naming, same semantics, same Chuck.
// "fuck torch"

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Multi-dim Float32 tensor. Shape is row-major; the innermost dim is shape[ndim-1].
 * Tensors are plain data carriers — autograd lives on the tape, not on the tensor.
 */
export class Tensor {
  constructor(data, shape) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = Array.isArray(shape) ? shape.slice() : [this.data.length];
    this.len = this.data.length;
    this.gpuBuffer = null;
  }

  static zeros(shape) {
    const len = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(len), shape);
  }

  static rand(shape, scale = 1.0) {
    const len = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(len);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * scale;
    return new Tensor(data, shape);
  }

  /** Xavier/Kaiming-ish uniform: scale = sqrt(6 / (fan_in + fan_out)). */
  static xavier(shape, fanIn, fanOut) {
    const s = Math.sqrt(6 / (fanIn + fanOut));
    return Tensor.rand(shape, s);
  }

  /** Normal-distributed via Box–Muller. */
  static randn(shape, std = 1.0) {
    const len = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(len);
    for (let i = 0; i < len; i += 2) {
      const u1 = Math.max(Math.random(), 1e-9);
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const t = 2 * Math.PI * u2;
      data[i] = r * Math.cos(t) * std;
      if (i + 1 < len) data[i + 1] = r * Math.sin(t) * std;
    }
    return new Tensor(data, shape);
  }

  static fromArray(arr, shape) {
    return new Tensor(new Float32Array(arr), shape);
  }

  clone() {
    return new Tensor(new Float32Array(this.data), this.shape.slice());
  }

  fill(v) {
    this.data.fill(v);
    return this;
  }

  reshape(newShape) {
    const len = newShape.reduce((a, b) => a * b, 1);
    if (len !== this.len) throw new Error(`reshape: size mismatch ${this.len} vs ${len}`);
    this.shape = newShape.slice();
    return this;
  }

  /**
   * In-place Kaiming uniform init (mirrors C nt_kaiming_uniform_init).
   * Samples uniformly from [-sqrt(3/fanIn), +sqrt(3/fanIn)] so Var = 1/fanIn.
   * Note: scale is per fan_in, NOT per rank — this matches PyTorch's
   * `kaiming_uniform_(a=sqrt(5))` convention used for LoRA A init.
   */
  kaimingUniform_(fanIn) {
    if (fanIn <= 0) throw new Error(`kaimingUniform_: fanIn must be > 0, got ${fanIn}`);
    const scale = Math.sqrt(3 / fanIn);
    for (let i = 0; i < this.len; i++) this.data[i] = (Math.random() * 2 - 1) * scale;
    return this;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// OP CODES — match notorch.h NT_OP_* numbering for cross-reference clarity.
// ═══════════════════════════════════════════════════════════════════════════

export const OP = Object.freeze({
  NONE: 0,
  MATVEC: 1,
  ADD: 2,
  MUL: 3,
  SCALE: 4,
  SOFTMAX: 5,
  RMSNORM: 6,
  SILU: 7,
  CROSS_ENT: 8,
  EMB_LOOKUP: 9,
  MATMUL: 10,
  SEQ_EMBED: 11,
  SEQ_MATVEC: 12,
  SEQ_RMSNORM: 13,
  CAUSAL_ATTN: 14,
  SEQ_CROSSENT: 15,
  MH_CAUSAL_ATTN: 16,
  GEGLU: 17,
  ROPE: 18,
  DROPOUT: 19,
  LAYERNORM: 20,
  SEQ_LAYERNORM: 21,
  GELU: 22,
  GQA_ATTN: 23,
  RRPRAM_ATTN: 24,
  CONCAT: 25,
  SEQ_MATVEC_T: 26,
  SIGMOID: 27,
  SCALE_BY_T: 28,
  SWIGLU: 29,
  BIT_LINEAR: 30,
  BIT_SEQ_LINEAR: 31,
  SEQ_CROSSENT_MASKED: 32,
  RRPRAM_LR: 33,
  // JS-specific extensions
  SUB: 100,
  DIV: 101,
  NEG: 102,
  TRANSPOSE: 103,
  TANH: 104,
  RELU: 105,
  EMBEDDING: 106,
  MSE: 107,
});

// ═══════════════════════════════════════════════════════════════════════════
// AUTOGRAD TAPE
// Reverse-mode AD. Mirrors C nt_tape_* API: every forward op records an entry
// (output tensor + op code + parent indices + aux). backward(idx) walks the
// tape from idx down to 0, accumulating grads into entry.grad.
// ═══════════════════════════════════════════════════════════════════════════

class TapeEntry {
  constructor(output, op, p1 = -1, p2 = -1, p3 = -1, aux = 0, aux2 = 0, aux3 = 0, aux4 = 0) {
    this.output = output;
    this.grad = null;          // Float32Array, lazily allocated on first acc
    this.op = op;
    this.parent1 = p1;
    this.parent2 = p2;
    this.parent3 = p3;
    this.aux = aux;
    this.aux2 = aux2;
    this.aux3 = aux3;
    this.aux4 = aux4;
    this.isParam = false;
    this.noDecay = false;
    this.frozen = false;
  }
}

export class Tape {
  constructor() {
    this.entries = [];
    this.active = false;
    // Optimizer state per parameter (parallel arrays keyed by param order).
    this.adamState = [];      // {m, v, accGrad, t}
    this.chuckState = null;   // global Chuck state
    this.chuckParams = [];    // per-param Chuck state
    this.training = true;
  }

  start() { this.active = true; }
  stop() { this.active = false; }
  clear() { this.entries.length = 0; this.adamState.length = 0; this.chuckParams.length = 0; this.chuckState = null; }

  /** Truncate tape back to the first `n` entries. Useful for preserving
   *  parameter leaves while discarding the forward graph between steps. */
  truncate(n) {
    if (n < this.entries.length) this.entries.length = n;
  }

  /** Snapshot the current entry count — call after building parameters. */
  mark() { return this.entries.length; }

  /** Record an op. Returns the entry index. */
  record(output, op, p1 = -1, p2 = -1, p3 = -1, aux = 0, aux2 = 0, aux3 = 0, aux4 = 0) {
    const entry = new TapeEntry(output, op, p1, p2, p3, aux, aux2, aux3, aux4);
    this.entries.push(entry);
    return this.entries.length - 1;
  }

  /** Wrap a tensor as a leaf (no gradient parent). */
  leaf(tensor) {
    return this.record(tensor, OP.NONE);
  }

  /** Mark a leaf as a trainable parameter. Returns its tape index. */
  param(tensor) {
    const idx = this.leaf(tensor);
    this.entries[idx].isParam = true;
    // Allocate Adam slots lazily on first step; reserve a placeholder now.
    this.adamState.push({ m: null, v: null, accGrad: null, t: 0 });
    this.chuckParams.push({
      gradHist: new Float32Array(NT_CHUCK_WINDOW),
      dampen: 1.0, frozen: 0, pos: 0, full: 0, stag: 0,
    });
    return idx;
  }

  /**
   * Mark a leaf as a FROZEN parameter (mirrors C nt_tape_param_frozen).
   * Like param() but does NOT allocate an optimizer slot — Chuck/SGD slots
   * stay 1:1 with truly trainable params. Backward also skips dw accumulation
   * for frozen entries (see accGrad guard on e.frozen above).
   *
   * Use for LoRA base weights: nt_tape_param_frozen(W) before nt_tape_param(A),
   * so A,B occupy clean leading Chuck slots while W stays read-only.
   */
  paramFrozen(tensor) {
    const idx = this.leaf(tensor);
    const e = this.entries[idx];
    e.isParam = true;
    e.frozen = true;
    // INTENTIONAL: do NOT push to adamState / chuckParams. Optimizer loops
    // skip via `!e.grad || e.frozen` and never consume an optimizer slot.
    return idx;
  }

  noDecay(idx) { this.entries[idx].noDecay = true; }
  freezeParam(idx) { this.entries[idx].frozen = true; }

  /** Accumulate `vals` into entry[idx].grad (allocate if needed). */
  accGrad(idx, vals) {
    if (idx < 0 || idx >= this.entries.length) return;
    const e = this.entries[idx];
    if (e.frozen) return;
    if (!e.grad) e.grad = new Float32Array(e.output.len);
    const n = Math.min(vals.length, e.grad.length);
    for (let i = 0; i < n; i++) e.grad[i] += vals[i];
  }

  /** Reset gradients on all entries. Called at the start of each train step. */
  zeroGrads() {
    for (const e of this.entries) {
      if (e.grad) e.grad.fill(0);
    }
  }

  /**
   * Reverse-mode backward from loss entry index. Loss entry's gradient is
   * seeded to 1.0. Each op's gradient routine computes contributions to its
   * parents' grads. Mirrors notorch.c switch (e->op) { ... }.
   */
  backward(lossIdx) {
    if (lossIdx < 0 || lossIdx >= this.entries.length) return;
    // Seed loss grad
    const lossEntry = this.entries[lossIdx];
    if (!lossEntry.grad) lossEntry.grad = new Float32Array(lossEntry.output.len);
    if (lossEntry.grad.length > 0) lossEntry.grad[0] = 1.0;

    for (let i = lossIdx; i >= 0; i--) {
      const e = this.entries[i];
      if (!e.grad) continue;
      this._backwardOp(e);
    }
  }

  // Big switch — one branch per OP. Kept inline to avoid call overhead.
  _backwardOp(e) {
    const dout = e.grad;
    const outLen = e.output.len;

    switch (e.op) {

      case OP.NONE: return;

      case OP.ADD: {
        if (e.parent1 >= 0) this.accGrad(e.parent1, dout);
        if (e.parent2 >= 0) this.accGrad(e.parent2, dout);
        return;
      }

      case OP.SUB: {
        if (e.parent1 >= 0) this.accGrad(e.parent1, dout);
        if (e.parent2 >= 0) {
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = -dout[i];
          this.accGrad(e.parent2, g);
        }
        return;
      }

      case OP.NEG: {
        if (e.parent1 >= 0) {
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = -dout[i];
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.MUL: {
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const a = this.entries[e.parent1].output.data;
          const b = this.entries[e.parent2].output.data;
          const ga = new Float32Array(outLen);
          const gb = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            ga[i] = Math.fround(dout[i] * b[i]);
            gb[i] = Math.fround(dout[i] * a[i]);
          }
          this.accGrad(e.parent1, ga);
          this.accGrad(e.parent2, gb);
        }
        return;
      }

      case OP.DIV: {
        // y = a / b; dy/da = 1/b; dy/db = -a/b^2
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const a = this.entries[e.parent1].output.data;
          const b = this.entries[e.parent2].output.data;
          const ga = new Float32Array(outLen);
          const gb = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const bi = b[i];
            ga[i] = dout[i] / bi;
            gb[i] = -dout[i] * a[i] / (bi * bi);
          }
          this.accGrad(e.parent1, ga);
          this.accGrad(e.parent2, gb);
        }
        return;
      }

      case OP.SCALE: {
        if (e.parent1 >= 0) {
          const ga = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) ga[i] = dout[i] * e.aux;
          this.accGrad(e.parent1, ga);
        }
        return;
      }

      case OP.MATMUL: {
        // C[M,N] = A[M,K] @ B[K,N]
        // dA = dC @ B^T,  dB = A^T @ dC
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const A = this.entries[e.parent1].output;
          const B = this.entries[e.parent2].output;
          const M = A.shape[0], K = A.shape[1], N = B.shape[1];
          const dA = new Float32Array(M * K);
          const dB = new Float32Array(K * N);
          const Ad = A.data, Bd = B.data;
          // dA[m,k] = Σ_n dC[m,n] * B[k,n]
          for (let m = 0; m < M; m++) {
            for (let k = 0; k < K; k++) {
              let s = 0;
              for (let n = 0; n < N; n++) s += dout[m * N + n] * Bd[k * N + n];
              dA[m * K + k] = s;
            }
          }
          // dB[k,n] = Σ_m A[m,k] * dC[m,n]
          for (let k = 0; k < K; k++) {
            for (let n = 0; n < N; n++) {
              let s = 0;
              for (let m = 0; m < M; m++) s += Ad[m * K + k] * dout[m * N + n];
              dB[k * N + n] = s;
            }
          }
          this.accGrad(e.parent1, dA);
          this.accGrad(e.parent2, dB);
        }
        return;
      }

      case OP.MATVEC: {
        // y = W @ x ; W: [rows, cols] ; x: [cols]
        // dW[i,j] = dout[i] * x[j] ;  dx[j] = Σ_i W[i,j] * dout[i]
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const W = this.entries[e.parent1].output;
          const x = this.entries[e.parent2].output;
          const rows = W.shape[0];
          const cols = W.shape.length >= 2 ? W.shape[1] : (W.len / rows);
          const dW = new Float32Array(rows * cols);
          const dx = new Float32Array(cols);
          for (let i = 0; i < rows; i++) {
            const di = dout[i];
            for (let j = 0; j < cols; j++) dW[i * cols + j] = di * x.data[j];
          }
          for (let j = 0; j < cols; j++) {
            let s = 0;
            for (let i = 0; i < rows; i++) s += W.data[i * cols + j] * dout[i];
            dx[j] = s;
          }
          this.accGrad(e.parent1, dW);
          this.accGrad(e.parent2, dx);
        }
        return;
      }

      case OP.SEQ_MATVEC: {
        // Y[t] = W @ X[t] for t=0..T-1 ; W: [out, in] ; X: [T, in] ; Y: [T, out]
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const W = this.entries[e.parent1].output;
          const X = this.entries[e.parent2].output;
          const T = e.aux | 0;
          const outDim = W.shape[0];
          const inDim = W.shape.length >= 2 ? W.shape[1] : (W.len / outDim);
          const dW = new Float32Array(outDim * inDim);
          const dX = new Float32Array(T * inDim);
          for (let t = 0; t < T; t++) {
            const xt = X.data.subarray(t * inDim, (t + 1) * inDim);
            const dyt = dout.subarray(t * outDim, (t + 1) * outDim);
            // dW += dyt ⊗ xt
            for (let i = 0; i < outDim; i++) {
              const di = dyt[i];
              for (let j = 0; j < inDim; j++) dW[i * inDim + j] += di * xt[j];
            }
            // dX[t] = W^T @ dyt
            for (let j = 0; j < inDim; j++) {
              let s = 0;
              for (let i = 0; i < outDim; i++) s += W.data[i * inDim + j] * dyt[i];
              dX[t * inDim + j] = s;
            }
          }
          this.accGrad(e.parent1, dW);
          this.accGrad(e.parent2, dX);
        }
        return;
      }

      case OP.TRANSPOSE: {
        // y = transpose(x, dimA, dimB); reverse-rotate gradient
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output;
          const dimA = e.aux | 0, dimB = e.aux2 | 0;
          const dx = transposeData(dout, e.output.shape, dimA, dimB);
          this.accGrad(e.parent1, dx);
        }
        return;
      }

      case OP.SILU: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const xi = x[i];
            const sig = 1 / (1 + Math.exp(-xi));
            g[i] = dout[i] * sig * (1 + xi * (1 - sig));
          }
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.SIGMOID: {
        if (e.parent1 >= 0) {
          const y = e.output.data;
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = dout[i] * y[i] * (1 - y[i]);
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.TANH: {
        if (e.parent1 >= 0) {
          const y = e.output.data;
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = dout[i] * (1 - y[i] * y[i]);
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.RELU: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = x[i] > 0 ? dout[i] : 0;
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.GELU: {
        // GELU tanh-approx: derivative computed numerically from x
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const g = new Float32Array(outLen);
          const k = Math.sqrt(2 / Math.PI);
          for (let i = 0; i < outLen; i++) {
            const xi = x[i];
            const inner = k * (xi + 0.044715 * xi * xi * xi);
            const t = Math.tanh(inner);
            const dInner = k * (1 + 3 * 0.044715 * xi * xi);
            const dt = (1 - t * t) * dInner;
            g[i] = dout[i] * (0.5 * (1 + t) + 0.5 * xi * dt);
          }
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.SOFTMAX: {
        if (e.parent1 >= 0) {
          // For a vector softmax: dx = y * (dout - dot(dout, y))
          let dotDy = 0;
          for (let i = 0; i < outLen; i++) dotDy += dout[i] * e.output.data[i];
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = e.output.data[i] * (dout[i] - dotDy);
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.RMSNORM: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const n = outLen;
          let ss = 0;
          for (let i = 0; i < n; i++) ss += x[i] * x[i];
          const rms = Math.sqrt(ss / n + 1e-6);
          const rms3 = rms * rms * rms;
          const hasGamma = e.parent2 >= 0;
          const gammaData = hasGamma ? this.entries[e.parent2].output.data : null;
          const gammaLen = hasGamma ? this.entries[e.parent2].output.len : 0;
          const doutEff = hasGamma ? new Float32Array(n) : dout;
          if (hasGamma) {
            for (let i = 0; i < n; i++) doutEff[i] = dout[i] * gammaData[i % gammaLen];
          }
          let sumDoutX = 0;
          for (let i = 0; i < n; i++) sumDoutX += doutEff[i] * x[i];
          const gx = new Float32Array(n);
          for (let i = 0; i < n; i++) gx[i] = (doutEff[i] / rms) - (x[i] * sumDoutX / (n * rms3));
          this.accGrad(e.parent1, gx);
          if (hasGamma) {
            const gg = new Float32Array(gammaLen);
            for (let i = 0; i < n; i++) gg[i % gammaLen] += dout[i] * (x[i] / rms);
            this.accGrad(e.parent2, gg);
          }
        }
        return;
      }

      case OP.LAYERNORM: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const n = outLen;
          const hasGamma = e.parent2 >= 0;
          const hasBeta = e.parent3 >= 0;
          const gamma = hasGamma ? this.entries[e.parent2].output.data : null;
          let mean = 0;
          for (let i = 0; i < n; i++) mean += x[i];
          mean /= n;
          let varv = 0;
          for (let i = 0; i < n; i++) { const d = x[i] - mean; varv += d * d; }
          varv /= n;
          const invStd = 1 / Math.sqrt(varv + 1e-5);
          const doutEff = new Float32Array(n);
          for (let i = 0; i < n; i++) doutEff[i] = hasGamma ? dout[i] * gamma[i] : dout[i];
          let sumDe = 0, sumDeXhat = 0;
          for (let i = 0; i < n; i++) {
            const xhat = (x[i] - mean) * invStd;
            sumDe += doutEff[i];
            sumDeXhat += doutEff[i] * xhat;
          }
          const gx = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            const xhat = (x[i] - mean) * invStd;
            gx[i] = invStd * (doutEff[i] - sumDe / n - xhat * sumDeXhat / n);
          }
          this.accGrad(e.parent1, gx);
          if (hasGamma) {
            const gn = this.entries[e.parent2].output.len;
            const gg = new Float32Array(gn);
            for (let i = 0; i < n && i < gn; i++) gg[i] += dout[i] * (x[i] - mean) * invStd;
            this.accGrad(e.parent2, gg);
          }
          if (hasBeta) {
            const bn = this.entries[e.parent3].output.len;
            const gb = new Float32Array(bn);
            for (let i = 0; i < n && i < bn; i++) gb[i] += dout[i];
            this.accGrad(e.parent3, gb);
          }
        }
        return;
      }

      case OP.SEQ_LAYERNORM: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const T = e.aux | 0, D = e.aux2 | 0;
          const hasGamma = e.parent2 >= 0;
          const hasBeta = e.parent3 >= 0;
          const gamma = hasGamma ? this.entries[e.parent2].output.data : null;
          const gx = new Float32Array(T * D);
          const gg = hasGamma ? new Float32Array(D) : null;
          const gb = hasBeta ? new Float32Array(D) : null;
          for (let t = 0; t < T; t++) {
            const off = t * D;
            let mean = 0;
            for (let d = 0; d < D; d++) mean += x[off + d];
            mean /= D;
            let varv = 0;
            for (let d = 0; d < D; d++) { const dd = x[off + d] - mean; varv += dd * dd; }
            varv /= D;
            const invStd = 1 / Math.sqrt(varv + 1e-5);
            let sumDe = 0, sumDeXhat = 0;
            for (let d = 0; d < D; d++) {
              const de = hasGamma ? dout[off + d] * gamma[d] : dout[off + d];
              const xhat = (x[off + d] - mean) * invStd;
              sumDe += de;
              sumDeXhat += de * xhat;
            }
            for (let d = 0; d < D; d++) {
              const de = hasGamma ? dout[off + d] * gamma[d] : dout[off + d];
              const xhat = (x[off + d] - mean) * invStd;
              gx[off + d] = invStd * (de - sumDe / D - xhat * sumDeXhat / D);
            }
            if (gg) for (let d = 0; d < D; d++) gg[d] += dout[off + d] * (x[off + d] - mean) * invStd;
            if (gb) for (let d = 0; d < D; d++) gb[d] += dout[off + d];
          }
          this.accGrad(e.parent1, gx);
          if (gg) this.accGrad(e.parent2, gg);
          if (gb) this.accGrad(e.parent3, gb);
        }
        return;
      }

      case OP.SEQ_RMSNORM: {
        if (e.parent1 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const T = e.aux | 0, D = e.aux2 | 0;
          const hasGamma = e.parent2 >= 0;
          const gamma = hasGamma ? this.entries[e.parent2].output.data : null;
          const gx = new Float32Array(T * D);
          const gg = hasGamma ? new Float32Array(D) : null;
          for (let t = 0; t < T; t++) {
            const off = t * D;
            let ss = 0;
            for (let d = 0; d < D; d++) ss += x[off + d] * x[off + d];
            const rms = Math.sqrt(ss / D + 1e-6);
            const rms3 = rms * rms * rms;
            const doutEff = hasGamma ? new Float32Array(D) : dout.subarray(off, off + D);
            if (hasGamma) for (let d = 0; d < D; d++) doutEff[d] = dout[off + d] * gamma[d];
            let sumDoutX = 0;
            for (let d = 0; d < D; d++) sumDoutX += doutEff[d] * x[off + d];
            for (let d = 0; d < D; d++)
              gx[off + d] = (doutEff[d] / rms) - (x[off + d] * sumDoutX / (D * rms3));
            if (gg) for (let d = 0; d < D; d++) gg[d] += dout[off + d] * (x[off + d] / rms);
          }
          this.accGrad(e.parent1, gx);
          if (gg) this.accGrad(e.parent2, gg);
        }
        return;
      }

      case OP.SWIGLU: {
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const g = this.entries[e.parent1].output.data;
          const u = this.entries[e.parent2].output.data;
          const dg = new Float32Array(outLen);
          const du = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const gi = g[i];
            const s = 1 / (1 + Math.exp(-gi));
            const silu = gi * s;
            const dsilu = s * (1 + gi * (1 - s));
            dg[i] = dout[i] * u[i] * dsilu;
            du[i] = dout[i] * silu;
          }
          this.accGrad(e.parent1, dg);
          this.accGrad(e.parent2, du);
        }
        return;
      }

      case OP.MH_CAUSAL_ATTN: {
        if (e.parent1 >= 0 && e.parent2 >= 0 && e.parent3 >= 0) {
          const Q = this.entries[e.parent1].output.data;
          const K = this.entries[e.parent2].output.data;
          const V = this.entries[e.parent3].output.data;
          const T = e.aux | 0, headDim = e.aux2 | 0;
          const D = (e.output.len / T) | 0;
          const nHeads = (D / headDim) | 0;
          const sc = 1 / Math.sqrt(headDim);
          const dQ = new Float32Array(T * D);
          const dK = new Float32Array(T * D);
          const dV = new Float32Array(T * D);
          for (let h = 0; h < nHeads; h++) {
            const ho = h * headDim;
            for (let i = 0; i < T; i++) {
              const qiOff = i * D + ho;
              const doutiOff = i * D + ho;
              const scores = new Float32Array(i + 1);
              const attn = new Float32Array(i + 1);
              let mx = -Infinity;
              for (let j = 0; j <= i; j++) {
                const kjOff = j * D + ho;
                let dot = 0;
                for (let d = 0; d < headDim; d++) dot += Q[qiOff + d] * K[kjOff + d];
                scores[j] = dot * sc;
                if (scores[j] > mx) mx = scores[j];
              }
              let sm = 0;
              for (let j = 0; j <= i; j++) { attn[j] = Math.exp(scores[j] - mx); sm += attn[j]; }
              if (sm > 0) for (let j = 0; j <= i; j++) attn[j] /= sm;
              const dAttn = new Float32Array(i + 1);
              for (let j = 0; j <= i; j++) {
                const vjOff = j * D + ho;
                for (let d = 0; d < headDim; d++) dAttn[j] += dout[doutiOff + d] * V[vjOff + d];
              }
              for (let j = 0; j <= i; j++) {
                const dvjOff = j * D + ho;
                const aj = attn[j];
                for (let d = 0; d < headDim; d++) dV[dvjOff + d] += aj * dout[doutiOff + d];
              }
              let dotDa = 0;
              for (let j = 0; j <= i; j++) dotDa += dAttn[j] * attn[j];
              for (let j = 0; j <= i; j++) {
                const ds = attn[j] * (dAttn[j] - dotDa) * sc;
                const kjOff = j * D + ho;
                for (let d = 0; d < headDim; d++) {
                  dQ[i * D + ho + d] += ds * K[kjOff + d];
                  dK[j * D + ho + d] += ds * Q[qiOff + d];
                }
              }
            }
          }
          this.accGrad(e.parent1, dQ);
          this.accGrad(e.parent2, dK);
          this.accGrad(e.parent3, dV);
        }
        return;
      }

      case OP.CAUSAL_ATTN: {
        // Single-head reduces to multi-head with nHeads=1 ; dispatch handled via MH_CAUSAL_ATTN.
        // For clarity of the C-mirror we keep a dedicated branch.
        if (e.parent1 >= 0 && e.parent2 >= 0 && e.parent3 >= 0) {
          const Q = this.entries[e.parent1].output.data;
          const K = this.entries[e.parent2].output.data;
          const V = this.entries[e.parent3].output.data;
          const T = e.aux | 0, D = e.aux2 | 0;
          const sc = 1 / Math.sqrt(D);
          const dQ = new Float32Array(T * D);
          const dK = new Float32Array(T * D);
          const dV = new Float32Array(T * D);
          for (let i = 0; i < T; i++) {
            const scores = new Float32Array(i + 1);
            const attn = new Float32Array(i + 1);
            let mx = -Infinity;
            for (let j = 0; j <= i; j++) {
              let dot = 0;
              for (let d = 0; d < D; d++) dot += Q[i * D + d] * K[j * D + d];
              scores[j] = dot * sc;
              if (scores[j] > mx) mx = scores[j];
            }
            let sm = 0;
            for (let j = 0; j <= i; j++) { attn[j] = Math.exp(scores[j] - mx); sm += attn[j]; }
            if (sm > 0) for (let j = 0; j <= i; j++) attn[j] /= sm;
            const dAttn = new Float32Array(i + 1);
            for (let j = 0; j <= i; j++)
              for (let d = 0; d < D; d++) dAttn[j] += dout[i * D + d] * V[j * D + d];
            for (let j = 0; j <= i; j++)
              for (let d = 0; d < D; d++) dV[j * D + d] += attn[j] * dout[i * D + d];
            let dotDa = 0;
            for (let j = 0; j <= i; j++) dotDa += dAttn[j] * attn[j];
            for (let j = 0; j <= i; j++) {
              const ds = attn[j] * (dAttn[j] - dotDa) * sc;
              for (let d = 0; d < D; d++) {
                dQ[i * D + d] += ds * K[j * D + d];
                dK[j * D + d] += ds * Q[i * D + d];
              }
            }
          }
          this.accGrad(e.parent1, dQ);
          this.accGrad(e.parent2, dK);
          this.accGrad(e.parent3, dV);
        }
        return;
      }

      case OP.SEQ_CROSSENT: {
        // Loss = -(1/T) Σ_t log softmax(logits[t])[targets[t]]
        // dlogits[t,j] = (softmax[j] - 1{j==target}) * (dout[0] / T)
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const logits = this.entries[e.parent1].output.data;
          const targets = this.entries[e.parent2].output.data;
          const T = e.aux | 0, V = e.aux2 | 0;
          const dl = new Float32Array(T * V);
          const sd = dout[0] / T;
          for (let t = 0; t < T; t++) {
            const off = t * V;
            const tgt = targets[t] | 0;
            let mx = logits[off];
            for (let j = 1; j < V; j++) if (logits[off + j] > mx) mx = logits[off + j];
            let sum = 0;
            for (let j = 0; j < V; j++) { dl[off + j] = Math.exp(logits[off + j] - mx); sum += dl[off + j]; }
            for (let j = 0; j < V; j++) dl[off + j] /= sum;
            if (tgt >= 0 && tgt < V) dl[off + tgt] -= 1;
            for (let j = 0; j < V; j++) dl[off + j] *= sd;
          }
          this.accGrad(e.parent1, dl);
        }
        return;
      }

      case OP.CROSS_ENT: {
        if (e.parent1 >= 0) {
          const logits = this.entries[e.parent1].output.data;
          const n = logits.length;
          const tgt = e.aux | 0;
          let mx = logits[0];
          for (let i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
          const sm = new Float32Array(n);
          let sum = 0;
          for (let i = 0; i < n; i++) { sm[i] = Math.exp(logits[i] - mx); sum += sm[i]; }
          for (let i = 0; i < n; i++) sm[i] /= sum;
          if (tgt >= 0 && tgt < n) sm[tgt] -= 1;
          for (let i = 0; i < n; i++) sm[i] *= dout[0];
          this.accGrad(e.parent1, sm);
        }
        return;
      }

      case OP.MSE: {
        // L = (1/N) Σ (pred - target)^2 ; stored: parent1 = pred, parent2 = target
        // dpred = (2/N) (pred - target) * dout[0]
        if (e.parent1 >= 0) {
          const pred = this.entries[e.parent1].output.data;
          const tgt = this.entries[e.parent2].output.data;
          const n = pred.length;
          const g = new Float32Array(n);
          const k = 2 * dout[0] / n;
          for (let i = 0; i < n; i++) g[i] = k * (pred[i] - tgt[i]);
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.EMB_LOOKUP: {
        if (e.parent1 >= 0) {
          const W = this.entries[e.parent1].output;
          const tokenId = e.aux | 0;
          const cols = W.shape.length >= 2 ? W.shape[1] : outLen;
          const rows = (W.len / cols) | 0;
          if (tokenId >= 0 && tokenId < rows) {
            const gw = new Float32Array(W.len);
            for (let i = 0; i < cols && i < outLen; i++) gw[tokenId * cols + i] = dout[i];
            this.accGrad(e.parent1, gw);
          }
        }
        return;
      }

      case OP.SEQ_EMBED: {
        // h[t] = wte[tokens[t]] + (wpe[t] if wpe given)
        if (e.parent1 >= 0) {
          const wte = this.entries[e.parent1].output;
          const tokens = e.parent3 >= 0 ? this.entries[e.parent3].output.data : null;
          const T = e.aux | 0, D = e.aux2 | 0;
          const wteRows = (wte.len / D) | 0;
          if (tokens) {
            const gw = new Float32Array(wte.len);
            for (let t = 0; t < T; t++) {
              let tid = tokens[t] | 0;
              if (tid < 0) tid = 0;
              if (tid >= wteRows) tid = wteRows - 1;
              for (let d = 0; d < D; d++) gw[tid * D + d] += dout[t * D + d];
            }
            this.accGrad(e.parent1, gw);
          }
          if (e.parent2 >= 0) {
            const wpe = this.entries[e.parent2].output;
            const wpeRows = (wpe.len / D) | 0;
            const gp = new Float32Array(wpe.len);
            for (let t = 0; t < T; t++) {
              const pos = t < wpeRows ? t : wpeRows - 1;
              for (let d = 0; d < D; d++) gp[pos * D + d] += dout[t * D + d];
            }
            this.accGrad(e.parent2, gp);
          }
        }
        return;
      }

      case OP.EMBEDDING: {
        // y[t,d] = table[ids[t], d]; dtable[ids[t], d] += dout[t,d]
        if (e.parent1 >= 0) {
          const table = this.entries[e.parent1].output;
          const ids = this.entries[e.parent2].output.data;
          const T = e.aux | 0, D = e.aux2 | 0;
          const rows = (table.len / D) | 0;
          const gw = new Float32Array(table.len);
          for (let t = 0; t < T; t++) {
            let tid = ids[t] | 0;
            if (tid < 0) tid = 0;
            if (tid >= rows) tid = rows - 1;
            for (let d = 0; d < D; d++) gw[tid * D + d] += dout[t * D + d];
          }
          this.accGrad(e.parent1, gw);
        }
        return;
      }

      case OP.CONCAT: {
        // out = [a, b] per position; aux=T, aux2=Da, aux3=Db
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const T = e.aux | 0, Da = e.aux2 | 0, Db = e.aux3 | 0;
          const dA = new Float32Array(T * Da);
          const dB = new Float32Array(T * Db);
          const stride = Da + Db;
          for (let t = 0; t < T; t++) {
            for (let d = 0; d < Da; d++) dA[t * Da + d] = dout[t * stride + d];
            for (let d = 0; d < Db; d++) dB[t * Db + d] = dout[t * stride + Da + d];
          }
          this.accGrad(e.parent1, dA);
          this.accGrad(e.parent2, dB);
        }
        return;
      }

      case OP.ROPE: {
        // Inverse rotation. Pairs (2k, 2k+1).
        if (e.parent1 >= 0) {
          const T = e.aux | 0, headDim = e.aux2 | 0;
          const total = outLen;
          const D = (total / T) | 0;
          const nHeads = (D / headDim) | 0;
          const freqBase = e.aux3 || 10000;
          const gx = new Float32Array(total);
          for (let t = 0; t < T; t++) {
            for (let h = 0; h < nHeads; h++) {
              const off = t * D + h * headDim;
              for (let k = 0; k < headDim; k += 2) {
                const theta = t / Math.pow(freqBase, k / headDim);
                const c = Math.cos(theta), s = Math.sin(theta);
                const dx = dout[off + k];
                const dy = dout[off + k + 1];
                gx[off + k]     = dx * c + dy * s;
                gx[off + k + 1] = -dx * s + dy * c;
              }
            }
          }
          this.accGrad(e.parent1, gx);
        }
        return;
      }

      case OP.DROPOUT: {
        // Mask was stored in e.mask (allocated at forward).
        if (e.parent1 >= 0 && e._mask) {
          const g = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) g[i] = dout[i] * e._mask[i];
          this.accGrad(e.parent1, g);
        }
        return;
      }

      case OP.SCALE_BY_T: {
        // y = a[0] * x ; gx = a[0] * dout ; ga = sum(dout * x)
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const x = this.entries[e.parent1].output.data;
          const aVal = this.entries[e.parent2].output.data[0];
          const gx = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) gx[i] = aVal * dout[i];
          this.accGrad(e.parent1, gx);
          let ga = 0;
          for (let i = 0; i < outLen; i++) ga += dout[i] * x[i];
          this.accGrad(e.parent2, new Float32Array([ga]));
        }
        return;
      }

      case OP.SEQ_MATVEC_T: {
        // Y[t] = W^T @ X[t] ; W:[W_rows, W_cols] ; X[t]:[W_rows] ; Y[t]:[W_cols]
        // dX[t][i] = Σ_j W[i,j] * dout[t][j]   → dX[t] = W @ dout[t]
        // dW[i][j] = Σ_t X[t][i] * dout[t][j]  → dW = X^T @ dout
        if (e.parent1 >= 0 && e.parent2 >= 0) {
          const W = this.entries[e.parent1].output;
          const X = this.entries[e.parent2].output;
          const T = e.aux | 0;
          const W_rows = W.shape[0];
          const W_cols = W.shape.length >= 2 ? W.shape[1] : (W.len / W_rows);
          const dW = new Float32Array(W_rows * W_cols);
          const dX = new Float32Array(T * W_rows);
          for (let t = 0; t < T; t++) {
            const doutOff = t * W_cols;
            const xOff = t * W_rows;
            const dxOff = t * W_rows;
            // dX[t] = W @ dout[t]
            for (let i = 0; i < W_rows; i++) {
              let s = 0;
              const wRow = i * W_cols;
              for (let j = 0; j < W_cols; j++) s += W.data[wRow + j] * dout[doutOff + j];
              dX[dxOff + i] = s;
            }
            // dW[i,j] += x_t[i] * dout[t][j]
            for (let i = 0; i < W_rows; i++) {
              const xi = X.data[xOff + i];
              const wRow = i * W_cols;
              for (let j = 0; j < W_cols; j++) dW[wRow + j] += xi * dout[doutOff + j];
            }
          }
          this.accGrad(e.parent1, dW);
          this.accGrad(e.parent2, dX);
        }
        return;
      }

      case OP.SEQ_CROSSENT_MASKED: {
        // Masked sequence cross-entropy. parent3 = mask [T] (0=skip).
        // dlogits[t,j] = (softmax[j] - 1{j==target}) * (mask[t] * dout[0] / n_active)
        if (e.parent1 >= 0 && e.parent2 >= 0 && e.parent3 >= 0) {
          const logits = this.entries[e.parent1].output.data;
          const targets = this.entries[e.parent2].output.data;
          const mask = this.entries[e.parent3].output.data;
          const T = e.aux | 0;
          const V = e.aux2 | 0;
          let nActive = 0;
          for (let t = 0; t < T; t++) nActive += mask[t];
          if (nActive <= 0) return;
          const dl = new Float32Array(T * V);
          for (let t = 0; t < T; t++) {
            const m = mask[t];
            if (m === 0) continue;
            const off = t * V;
            let tgt = targets[t] | 0;
            if (tgt < 0 || tgt >= V) tgt = 0;
            let mx = logits[off];
            for (let j = 1; j < V; j++) if (logits[off + j] > mx) mx = logits[off + j];
            let sum = 0;
            for (let j = 0; j < V; j++) { dl[off + j] = Math.exp(logits[off + j] - mx); sum += dl[off + j]; }
            for (let j = 0; j < V; j++) dl[off + j] /= sum;
            dl[off + tgt] -= 1;
            const s = m * dout[0] / nActive;
            for (let j = 0; j < V; j++) dl[off + j] *= s;
          }
          this.accGrad(e.parent1, dl);
        }
        return;
      }

      case OP.GEGLU: {
        // y[t,i] = GELU(gate[t,i]) * val[t,i], gate = x @ W1^T, val = x @ W2^T.
        // Recompute gate/val/GELU(gate) and propagate to x, W1, W2.
        if (e.parent1 >= 0 && e.parent2 >= 0 && e.parent3 >= 0) {
          const x = this.entries[e.parent1].output;
          const W1 = this.entries[e.parent2].output;
          const W2 = this.entries[e.parent3].output;
          const D_out = W1.shape[0];
          const D_in = W1.shape.length >= 2 ? W1.shape[1] : (W1.len / D_out);
          const T = (x.len / D_in) | 0;
          const k = 0.7978845608;  // sqrt(2/pi)
          const dx = new Float32Array(x.len);
          const dW1 = new Float32Array(W1.len);
          const dW2 = new Float32Array(W2.len);
          for (let t = 0; t < T; t++) {
            const xOff = t * D_in;
            const yOff = t * D_out;
            for (let i = 0; i < D_out; i++) {
              const wRow = i * D_in;
              let gate = 0, val = 0;
              for (let j = 0; j < D_in; j++) {
                gate += W1.data[wRow + j] * x.data[xOff + j];
                val  += W2.data[wRow + j] * x.data[xOff + j];
              }
              const g3 = gate * gate * gate;
              const inner = k * (gate + 0.044715 * g3);
              const th = Math.tanh(inner);
              const geluGate = 0.5 * gate * (1 + th);
              const dy = dout[yOff + i];
              const dVal = dy * geluGate;
              const geluGrad = 0.5 * (1 + th)
                + 0.5 * gate * (1 - th * th) * k * (1 + 3 * 0.044715 * gate * gate);
              const dGate = dy * val * geluGrad;
              for (let j = 0; j < D_in; j++) {
                const xj = x.data[xOff + j];
                dW1[wRow + j] += dGate * xj;
                dW2[wRow + j] += dVal * xj;
                dx[xOff + j] += dGate * W1.data[wRow + j] + dVal * W2.data[wRow + j];
              }
            }
          }
          this.accGrad(e.parent1, dx);
          this.accGrad(e.parent2, dW1);
          this.accGrad(e.parent3, dW2);
        }
        return;
      }

      default: return;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function transposeData(data, outShape, dimA, dimB) {
  // Generic 2-axis transpose for arbitrary-rank tensors. outShape is the
  // already-transposed shape; we invert by swapping axes back.
  // For now we support 2D and 3D — sufficient for attention QKᵀ paths.
  if (outShape.length === 2) {
    const [R, C] = outShape;
    const out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out[c * R + r] = data[r * C + c];
    return out;
  }
  if (outShape.length === 3) {
    const [A, B, C] = outShape;
    const out = new Float32Array(A * B * C);
    if (dimA === 1 && dimB === 2) {
      for (let a = 0; a < A; a++)
        for (let b = 0; b < B; b++)
          for (let c = 0; c < C; c++)
            out[a * C * B + c * B + b] = data[a * B * C + b * C + c];
    } else {
      // Generic fallback: identity (caller error)
      out.set(data);
    }
    return out;
  }
  return new Float32Array(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// CHUCK CONSTANTS — synced with notorch.h
// ═══════════════════════════════════════════════════════════════════════════

const NT_CHUCK_WINDOW       = 16;
const NT_CHUCK_DAMP_LO      = 0.3;
const NT_CHUCK_DAMP_HI      = 2.0;
const NT_CHUCK_DAMP_DOWN    = 0.97;
const NT_CHUCK_DAMP_UP      = 1.03;
const NT_CHUCK_TREND_BRAKE  = 0.02;
const NT_CHUCK_TREND_PUSH   = -0.02;
const NT_CHUCK_STAG_THRESH  = 0.001;
const NT_CHUCK_STAG_STEPS   = 8;
const NT_CHUCK_NOISE_MAG    = 0.001;
const NT_CHUCK_NOISE_DECAY  = 0.9;
const NT_CHUCK_FREEZE_THRESH = 0.01;
const NT_CHUCK_MACRO_INT    = 1000;
const NT_CHUCK_MACRO_PAT    = 3;
const NT_CHUCK_MACRO_DECAY  = 0.5;
const NT_CHUCK_MEAN_REVERT  = 0.999;

function chuckRingAvg(buf, pos, full, start, q) {
  let s = 0;
  for (let i = 0; i < q; i++) s += buf[(start + i) % NT_CHUCK_WINDOW];
  return s / q;
}

function chuckRandn() {
  const u1 = Math.max(Math.random(), 1e-9);
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZERS — work directly on Tape entries with isParam=true.
// ═══════════════════════════════════════════════════════════════════════════

/** Accept either a Notorch engine (uses engine.tape) or a Tape directly. */
function _resolveTape(arg) {
  if (arg instanceof Tape) return arg;
  if (arg && arg.tape instanceof Tape) return arg.tape;
  throw new Error("Optimizer: expected Notorch engine or Tape");
}

export class SGD {
  constructor(tapeOrEngine, lr = 0.01, momentum = 0) {
    this.tape = _resolveTape(tapeOrEngine);
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = new Map();
  }

  step() {
    const t = this.tape;
    for (let i = 0; i < t.entries.length; i++) {
      const e = t.entries[i];
      if (!e.isParam || !e.grad || e.frozen) continue;
      const data = e.output.data;
      if (this.momentum > 0) {
        let v = this.velocity.get(i);
        if (!v) { v = new Float32Array(data.length); this.velocity.set(i, v); }
        for (let k = 0; k < data.length; k++) {
          v[k] = this.momentum * v[k] + e.grad[k];
          data[k] -= this.lr * v[k];
        }
      } else {
        for (let k = 0; k < data.length; k++) data[k] -= this.lr * e.grad[k];
      }
    }
  }
}

/** Chuck — self-aware Adam, ported 1:1 from notorch.c nt_tape_chuck_step. */
export class Chuck {
  constructor(tapeOrEngine, lr = 1e-3) {
    this.tape = _resolveTape(tapeOrEngine);
    this.lr = lr;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.eps = 1e-8;
  }

  step(lossVal) {
    const t = this.tape;
    if (!t.chuckState) {
      t.chuckState = {
        lossHist: new Float32Array(NT_CHUCK_WINDOW),
        dampen: 1.0, noise: 0, lossEma: 0, macroEma: 0, bestMacro: 1e9,
        lrScale: 1.0, macroStag: 0, globalStep: 0, pos: 0, full: 0, stag: 0,
        initialized: 1,
      };
    }
    const cs = t.chuckState;
    if (cs.lossEma === 0) cs.lossEma = lossVal;
    else cs.lossEma = 0.99 * cs.lossEma + 0.01 * lossVal;
    cs.lossHist[cs.pos] = cs.lossEma;
    cs.pos = (cs.pos + 1) % NT_CHUCK_WINDOW;
    if (cs.pos === 0) cs.full = 1;

    const len = cs.full ? NT_CHUCK_WINDOW : cs.pos;
    if (len >= 8) {
      let q = (len / 4) | 0; if (q < 1) q = 1;
      const oldStart = cs.full ? (cs.pos % NT_CHUCK_WINDOW) : 0;
      const recentStart = cs.full
        ? ((cs.pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW)
        : (cs.pos - q);
      const oldAvg = chuckRingAvg(cs.lossHist, cs.pos, cs.full, oldStart, q);
      const recentAvg = chuckRingAvg(cs.lossHist, cs.pos, cs.full, recentStart, q);
      if (oldAvg > this.eps) {
        const trend = (recentAvg - oldAvg) / oldAvg;
        if (trend >  NT_CHUCK_TREND_BRAKE) cs.dampen *= NT_CHUCK_DAMP_DOWN;
        if (trend <  NT_CHUCK_TREND_PUSH ) cs.dampen *= NT_CHUCK_DAMP_UP;
        if (Math.abs(trend) < NT_CHUCK_STAG_THRESH) {
          cs.stag++;
          if (cs.stag >= NT_CHUCK_STAG_STEPS) {
            cs.noise = NT_CHUCK_NOISE_MAG;
            cs.stag = 0;
          }
        } else {
          cs.stag = 0;
          cs.noise *= NT_CHUCK_NOISE_DECAY;
        }
      }
    }
    cs.dampen = NT_CHUCK_MEAN_REVERT * cs.dampen + (1 - NT_CHUCK_MEAN_REVERT) * 1.0;
    if (cs.dampen < NT_CHUCK_DAMP_LO) cs.dampen = NT_CHUCK_DAMP_LO;
    if (cs.dampen > NT_CHUCK_DAMP_HI) cs.dampen = NT_CHUCK_DAMP_HI;

    cs.globalStep++;
    if (cs.macroEma === 0) cs.macroEma = lossVal;
    else cs.macroEma = 0.999 * cs.macroEma + 0.001 * lossVal;
    if (cs.globalStep % NT_CHUCK_MACRO_INT === 0 && cs.globalStep > NT_CHUCK_WINDOW) {
      if (cs.macroEma > cs.bestMacro * 0.999) {
        cs.macroStag++;
        if (cs.macroStag >= NT_CHUCK_MACRO_PAT) {
          cs.lrScale *= NT_CHUCK_MACRO_DECAY;
          if (cs.lrScale < 0.05) cs.lrScale = 0.05;
          cs.macroStag = 0;
        }
      } else {
        cs.bestMacro = cs.macroEma;
        cs.macroStag = 0;
        if (cs.lrScale < 1.0) {
          cs.lrScale *= 1.2;
          if (cs.lrScale > 1.0) cs.lrScale = 1.0;
        }
      }
    }

    const globalLambda = cs.dampen;
    const noiseMag = cs.noise;

    let paramIdx = 0;
    for (let i = 0; i < t.entries.length && paramIdx < t.adamState.length; i++) {
      const e = t.entries[i];
      if (!e.isParam || !e.grad) continue;
      const as = t.adamState[paramIdx];
      const cp = t.chuckParams[paramIdx];
      if (cp.dampen === 0) cp.dampen = 1.0;
      if (cp.frozen) { paramIdx++; continue; }
      const n = e.output.len;
      if (!as.m) as.m = new Float32Array(n);
      if (!as.v) as.v = new Float32Array(n);

      let gnorm = 0;
      for (let j = 0; j < n; j++) gnorm += e.grad[j] * e.grad[j];
      gnorm = Math.sqrt(gnorm);

      cp.gradHist[cp.pos] = gnorm;
      cp.pos = (cp.pos + 1) % NT_CHUCK_WINDOW;
      if (cp.pos === 0) cp.full = 1;
      const plen = cp.full ? NT_CHUCK_WINDOW : cp.pos;
      if (plen >= 8) {
        let q = (plen / 4) | 0; if (q < 1) q = 1;
        const oldStart = cp.full ? (cp.pos % NT_CHUCK_WINDOW) : 0;
        const recentStart = cp.full
          ? ((cp.pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW)
          : (cp.pos - q);
        const oldGn = chuckRingAvg(cp.gradHist, cp.pos, cp.full, oldStart, q);
        const recentGn = chuckRingAvg(cp.gradHist, cp.pos, cp.full, recentStart, q);
        if (oldGn > this.eps) {
          const gtrend = (recentGn - oldGn) / oldGn;
          if (gtrend >  0.05) cp.dampen *= NT_CHUCK_DAMP_UP;
          if (gtrend < -0.05) cp.dampen *= NT_CHUCK_DAMP_DOWN;
        }
        if (gnorm < NT_CHUCK_FREEZE_THRESH) {
          cp.stag++;
          if (cp.stag >= NT_CHUCK_STAG_STEPS) cp.frozen = 1;
        } else cp.stag = 0;
        cp.dampen = NT_CHUCK_MEAN_REVERT * cp.dampen + (1 - NT_CHUCK_MEAN_REVERT) * 1.0;
        if (cp.dampen < NT_CHUCK_DAMP_LO) cp.dampen = NT_CHUCK_DAMP_LO;
        if (cp.dampen > NT_CHUCK_DAMP_HI) cp.dampen = NT_CHUCK_DAMP_HI;
      }

      const paramLambda = cp.dampen;
      const effectiveLr = this.lr * globalLambda * paramLambda * cs.lrScale;
      as.t++;
      const data = e.output.data;
      const bc1 = 1 - Math.pow(this.beta1, as.t);
      const bc2 = 1 - Math.pow(this.beta2, as.t);
      for (let j = 0; j < n; j++) {
        const g = e.grad[j];
        as.m[j] = this.beta1 * as.m[j] + (1 - this.beta1) * g;
        as.v[j] = this.beta2 * as.v[j] + (1 - this.beta2) * g * g;
        const mh = as.m[j] / bc1;
        const vh = as.v[j] / bc2;
        let upd = effectiveLr * mh / (Math.sqrt(vh) + this.eps);
        if (noiseMag > 0) upd += noiseMag * chuckRandn();
        data[j] -= upd;
      }
      paramIdx++;
    }
  }
}

/** Clip total parameter gradient norm to maxNorm. Returns the original norm. */
export function clipGradNorm(tapeOrEngine, maxNorm) {
  const tape = _resolveTape(tapeOrEngine);
  let totalSq = 0;
  for (const e of tape.entries) {
    if (!e.isParam || !e.grad) continue;
    for (let j = 0; j < e.grad.length; j++) totalSq += e.grad[j] * e.grad[j];
  }
  const total = Math.sqrt(totalSq);
  if (total > maxNorm) {
    const scale = maxNorm / (total + 1e-6);
    for (const e of tape.entries) {
      if (!e.isParam || !e.grad) continue;
      for (let j = 0; j < e.grad.length; j++) e.grad[j] *= scale;
    }
  }
  return total;
}

// ═══════════════════════════════════════════════════════════════════════════
// WGSL SHADERS — kept minimal but real. Tiled matmul + element-wise.
// ═══════════════════════════════════════════════════════════════════════════

const TILED_MATMUL_WGSL = `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> u: Uniforms;

const TILE: u32 = 16u;
var<workgroup> tileA: array<f32, 256>;  // 16 * 16
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  var sum: f32 = 0.0;
  let nTiles = (u.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let aCol = t * TILE + lid.x;
    let bRow = t * TILE + lid.y;
    if (row < u.M && aCol < u.K) {
      tileA[lid.y * TILE + lid.x] = A[row * u.K + aCol];
    } else {
      tileA[lid.y * TILE + lid.x] = 0.0;
    }
    if (bRow < u.K && col < u.N) {
      tileB[lid.y * TILE + lid.x] = B[bRow * u.N + col];
    } else {
      tileB[lid.y * TILE + lid.x] = 0.0;
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum = sum + tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
    }
    workgroupBarrier();
  }
  if (row < u.M && col < u.N) {
    C[row * u.N + col] = sum;
  }
}
`;

const ELEMENTWISE_WGSL = (opCode) => `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
struct Uniforms { N: u32, scalar: f32 };
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.N) { return; }
  ${opCode}
}
`;

// ═══════════════════════════════════════════════════════════════════════════
// ENGINE
// Public surface. Holds the tape, GPU device, buffer pool, pipelines.
// All forward ops are methods on Notorch. They optionally record on the tape.
// ═══════════════════════════════════════════════════════════════════════════

export class Notorch {
  constructor() {
    this.device = null;
    this.hasWebGPU = false;
    this.pipelines = new Map();
    this.tape = new Tape();
    // Buffer pool: Map<size_in_bytes, GPUBuffer[]>
    this.bufferPool = new Map();
    this.allocatedBuffers = new Set();
  }

  async init() {
    if (typeof navigator !== "undefined" && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.device = await adapter.requestDevice();
          this.hasWebGPU = true;
          console.log("notorch.js: WebGPU active");
        }
      } catch (e) {
        console.warn("notorch.js: WebGPU init failed, CPU fallback", e);
      }
    }
    return this.hasWebGPU;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // TAPE PASSTHROUGHS
  // ═════════════════════════════════════════════════════════════════════════

  param(tensor) { return this.tape.param(tensor); }
  paramFrozen(tensor) { return this.tape.paramFrozen(tensor); }
  leaf(tensor) { return this.tape.leaf(tensor); }
  noDecay(idx) { this.tape.noDecay(idx); }
  freezeParam(idx) { this.tape.freezeParam(idx); }
  zeroGrads() { this.tape.zeroGrads(); }
  backward(idx) { this.tape.backward(idx); }
  trainMode(b = true) { this.tape.training = b; }
  isTraining() { return this.tape.training; }
  /** Get the output tensor of a tape entry (so caller can inspect). */
  get(idx) { return this.tape.entries[idx].output; }
  /** Reset tape but keep parameter values. Returns a snapshot of param idxs to re-leaf. */
  resetTape() { this.tape.entries.length = 0; }

  /** Mark current tape position (use after building params). */
  mark() { return this.tape.mark(); }

  /** Truncate tape to the marked position — keeps params, discards forward graph. */
  truncate(n) { this.tape.truncate(n); }

  // ═════════════════════════════════════════════════════════════════════════
  // ELEMENT-WISE OPS (CPU + GPU)
  // ═════════════════════════════════════════════════════════════════════════

  add(aIdx, bIdx) {
    const a = this.tape.entries[aIdx].output;
    const b = this.tape.entries[bIdx].output;
    const out = Tensor.zeros(a.shape);
    const da = a.data, db = b.data, dOut = out.data;
    const n = Math.min(da.length, db.length, dOut.length);
    for (let i = 0; i < n; i++) dOut[i] = Math.fround(da[i] + db[i]);
    return this.tape.record(out, OP.ADD, aIdx, bIdx);
  }

  sub(aIdx, bIdx) {
    const a = this.tape.entries[aIdx].output;
    const b = this.tape.entries[bIdx].output;
    const out = Tensor.zeros(a.shape);
    const n = a.len;
    for (let i = 0; i < n; i++) out.data[i] = Math.fround(a.data[i] - b.data[i]);
    return this.tape.record(out, OP.SUB, aIdx, bIdx);
  }

  mul(aIdx, bIdx) {
    const a = this.tape.entries[aIdx].output;
    const b = this.tape.entries[bIdx].output;
    const out = Tensor.zeros(a.shape);
    const n = a.len;
    for (let i = 0; i < n; i++) out.data[i] = Math.fround(a.data[i] * b.data[i]);
    return this.tape.record(out, OP.MUL, aIdx, bIdx);
  }

  div(aIdx, bIdx) {
    const a = this.tape.entries[aIdx].output;
    const b = this.tape.entries[bIdx].output;
    const out = Tensor.zeros(a.shape);
    const n = a.len;
    for (let i = 0; i < n; i++) out.data[i] = a.data[i] / b.data[i];
    return this.tape.record(out, OP.DIV, aIdx, bIdx);
  }

  neg(aIdx) {
    const a = this.tape.entries[aIdx].output;
    const out = Tensor.zeros(a.shape);
    for (let i = 0; i < a.len; i++) out.data[i] = -a.data[i];
    return this.tape.record(out, OP.NEG, aIdx);
  }

  scale(aIdx, factor) {
    const a = this.tape.entries[aIdx].output;
    const out = Tensor.zeros(a.shape);
    const f = Math.fround(factor);
    for (let i = 0; i < a.len; i++) out.data[i] = Math.fround(a.data[i] * f);
    return this.tape.record(out, OP.SCALE, aIdx, -1, -1, factor);
  }

  /**
   * Scale-by-tensor: y[i] = a[0] * x[i] where `a` is a scalar tensor [1].
   * Mirrors C nt_scale_by_t (notorch.c:3075). Gradient flows to both `x`
   * and the scalar `a` (single scalar grad = sum(dout * x)).
   */
  scaleByT(xIdx, aIdx) {
    const x = this.tape.entries[xIdx].output;
    const a = this.tape.entries[aIdx].output;
    if (a.len !== 1) throw new Error(`scaleByT: a must be scalar [1], got len=${a.len}`);
    const out = Tensor.zeros(x.shape);
    const aVal = a.data[0];
    for (let i = 0; i < x.len; i++) out.data[i] = Math.fround(aVal * x.data[i]);
    return this.tape.record(out, OP.SCALE_BY_T, xIdx, aIdx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // ACTIVATIONS
  // ═════════════════════════════════════════════════════════════════════════

  silu(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    for (let i = 0; i < x.len; i++) {
      const xi = x.data[i];
      const sig = 1 / (1 + Math.exp(-xi));
      out.data[i] = Math.fround(xi * sig);
    }
    return this.tape.record(out, OP.SILU, xIdx);
  }

  sigmoid(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    for (let i = 0; i < x.len; i++) out.data[i] = 1 / (1 + Math.exp(-x.data[i]));
    return this.tape.record(out, OP.SIGMOID, xIdx);
  }

  tanh(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    for (let i = 0; i < x.len; i++) out.data[i] = Math.tanh(x.data[i]);
    return this.tape.record(out, OP.TANH, xIdx);
  }

  relu(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    for (let i = 0; i < x.len; i++) out.data[i] = x.data[i] > 0 ? x.data[i] : 0;
    return this.tape.record(out, OP.RELU, xIdx);
  }

  gelu(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    const k = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < x.len; i++) {
      const xi = x.data[i];
      const t = Math.tanh(k * (xi + 0.044715 * xi * xi * xi));
      out.data[i] = Math.fround(0.5 * xi * (1 + t));
    }
    return this.tape.record(out, OP.GELU, xIdx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // SOFTMAX (numerically stable, vector form)
  // ═════════════════════════════════════════════════════════════════════════

  softmax(xIdx) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    let max = -Infinity;
    for (let i = 0; i < x.len; i++) if (x.data[i] > max) max = x.data[i];
    let sum = 0;
    for (let i = 0; i < x.len; i++) { out.data[i] = Math.exp(x.data[i] - max); sum += out.data[i]; }
    for (let i = 0; i < x.len; i++) out.data[i] /= sum;
    return this.tape.record(out, OP.SOFTMAX, xIdx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // MATMUL — CPU (tile-blocked) + GPU (tiled WGSL)
  // ═════════════════════════════════════════════════════════════════════════

  /** A: [M,K], B: [K,N] → C: [M,N]. Returns tape index. */
  matmul(aIdx, bIdx) {
    const A = this.tape.entries[aIdx].output;
    const B = this.tape.entries[bIdx].output;
    const out = Notorch.matmulCPU(A, B);
    return this.tape.record(out, OP.MATMUL, aIdx, bIdx);
  }

  /** Async GPU matmul; if no WebGPU, transparently falls back to CPU. */
  async matmulAsync(aIdx, bIdx) {
    if (this.hasWebGPU) {
      const A = this.tape.entries[aIdx].output;
      const B = this.tape.entries[bIdx].output;
      const out = await this.matmulGPU(A, B);
      return this.tape.record(out, OP.MATMUL, aIdx, bIdx);
    }
    return this.matmul(aIdx, bIdx);
  }

  static matmulCPU(A, B) {
    const [M, K] = A.shape;
    const [K2, N] = B.shape;
    if (K !== K2) throw new Error(`matmul shape mismatch ${K} vs ${K2}`);
    const out = Tensor.zeros([M, N]);
    const dA = A.data, dB = B.data, dOut = out.data;
    // Tiled CPU matmul — better cache use than naive ijk for matrices > ~64.
    const TILE = 32;
    for (let ii = 0; ii < M; ii += TILE) {
      for (let jj = 0; jj < N; jj += TILE) {
        for (let kk = 0; kk < K; kk += TILE) {
          const iEnd = Math.min(ii + TILE, M);
          const jEnd = Math.min(jj + TILE, N);
          const kEnd = Math.min(kk + TILE, K);
          for (let i = ii; i < iEnd; i++) {
            const iK = i * K, iN = i * N;
            for (let k = kk; k < kEnd; k++) {
              const a = dA[iK + k];
              const kN = k * N;
              for (let j = jj; j < jEnd; j++) {
                dOut[iN + j] = Math.fround(dOut[iN + j] + Math.fround(a * dB[kN + j]));
              }
            }
          }
        }
      }
    }
    return out;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // SEQ_LINEAR — Y[t] = W @ X[t] for T positions. W:[out, in] X:[T, in]
  // ═════════════════════════════════════════════════════════════════════════

  /** Sequence linear projection — typical of attention/FFN per-token. */
  seqLinear(wIdx, xIdx, T) {
    const W = this.tape.entries[wIdx].output;
    const X = this.tape.entries[xIdx].output;
    const outDim = W.shape[0];
    const inDim = W.shape.length >= 2 ? W.shape[1] : (W.len / outDim);
    const out = Tensor.zeros([T, outDim]);
    const Wd = W.data, Xd = X.data, Yd = out.data;
    for (let t = 0; t < T; t++) {
      const xOff = t * inDim, yOff = t * outDim;
      for (let i = 0; i < outDim; i++) {
        let s = 0;
        const wRow = i * inDim;
        for (let j = 0; j < inDim; j++) s += Wd[wRow + j] * Xd[xOff + j];
        Yd[yOff + i] = Math.fround(s);
      }
    }
    return this.tape.record(out, OP.SEQ_MATVEC, wIdx, xIdx, -1, T);
  }

  /**
   * Transposed sequence linear: Y[t] = W^T @ X[t]. W:[W_rows, W_cols],
   * X[t] has W_rows elements, Y[t] has W_cols elements. Mirrors C
   * nt_seq_linear_t (notorch.c:2899) — used by Janus Echo and other
   * patterns that need W^T projection without an explicit transpose op.
   */
  seqLinearT(wIdx, xIdx, T) {
    const W = this.tape.entries[wIdx].output;
    const X = this.tape.entries[xIdx].output;
    const W_rows = W.shape[0];
    const W_cols = W.shape.length >= 2 ? W.shape[1] : (W.len / W_rows);
    const out = Tensor.zeros([T, W_cols]);
    const Wd = W.data, Xd = X.data, Yd = out.data;
    for (let t = 0; t < T; t++) {
      const xOff = t * W_rows;
      const yOff = t * W_cols;
      for (let j = 0; j < W_cols; j++) {
        let s = 0;
        for (let i = 0; i < W_rows; i++) s += Wd[i * W_cols + j] * Xd[xOff + i];
        Yd[yOff + j] = Math.fround(s);
      }
    }
    return this.tape.record(out, OP.SEQ_MATVEC_T, wIdx, xIdx, -1, T);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // TRANSPOSE
  // ═════════════════════════════════════════════════════════════════════════

  transpose(aIdx, dimA = 0, dimB = 1) {
    const a = this.tape.entries[aIdx].output;
    if (a.shape.length === 2) {
      const [R, C] = a.shape;
      const out = Tensor.zeros([C, R]);
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out.data[c * R + r] = a.data[r * C + c];
      return this.tape.record(out, OP.TRANSPOSE, aIdx, -1, -1, dimA, dimB);
    }
    if (a.shape.length === 3) {
      const [A0, B0, C0] = a.shape;
      // Swap dimA <-> dimB
      const newShape = a.shape.slice();
      [newShape[dimA], newShape[dimB]] = [newShape[dimB], newShape[dimA]];
      const out = Tensor.zeros(newShape);
      // Generic 3D for axes (1,2)
      if (dimA === 1 && dimB === 2) {
        for (let i = 0; i < A0; i++)
          for (let j = 0; j < B0; j++)
            for (let k = 0; k < C0; k++)
              out.data[i * C0 * B0 + k * B0 + j] = a.data[i * B0 * C0 + j * C0 + k];
      } else {
        out.data.set(a.data);
      }
      return this.tape.record(out, OP.TRANSPOSE, aIdx, -1, -1, dimA, dimB);
    }
    throw new Error(`transpose: unsupported rank ${a.shape.length}`);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // NORMALIZATION
  // ═════════════════════════════════════════════════════════════════════════

  layernorm(xIdx, gammaIdx = -1, betaIdx = -1, eps = 1e-5) {
    const x = this.tape.entries[xIdx].output;
    const n = x.len;
    const out = Tensor.zeros(x.shape);
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x.data[i];
    mean /= n;
    let varv = 0;
    for (let i = 0; i < n; i++) { const d = x.data[i] - mean; varv += d * d; }
    varv /= n;
    const invStd = 1 / Math.sqrt(varv + eps);
    for (let i = 0; i < n; i++) out.data[i] = (x.data[i] - mean) * invStd;
    if (gammaIdx >= 0) {
      const g = this.tape.entries[gammaIdx].output.data;
      const gl = g.length;
      for (let i = 0; i < n; i++) out.data[i] *= g[i % gl];
    }
    if (betaIdx >= 0) {
      const b = this.tape.entries[betaIdx].output.data;
      const bl = b.length;
      for (let i = 0; i < n; i++) out.data[i] += b[i % bl];
    }
    return this.tape.record(out, OP.LAYERNORM, xIdx, gammaIdx, betaIdx);
  }

  seqLayernorm(xIdx, gammaIdx, betaIdx, T, D, eps = 1e-5) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros([T, D]);
    const gamma = gammaIdx >= 0 ? this.tape.entries[gammaIdx].output.data : null;
    const beta = betaIdx >= 0 ? this.tape.entries[betaIdx].output.data : null;
    for (let t = 0; t < T; t++) {
      const off = t * D;
      let mean = 0;
      for (let d = 0; d < D; d++) mean += x.data[off + d];
      mean /= D;
      let varv = 0;
      for (let d = 0; d < D; d++) { const dd = x.data[off + d] - mean; varv += dd * dd; }
      varv /= D;
      const invStd = 1 / Math.sqrt(varv + eps);
      for (let d = 0; d < D; d++) {
        let v = (x.data[off + d] - mean) * invStd;
        if (gamma) v *= gamma[d];
        if (beta)  v += beta[d];
        out.data[off + d] = v;
      }
    }
    return this.tape.record(out, OP.SEQ_LAYERNORM, xIdx, gammaIdx, betaIdx, T, D);
  }

  rmsnorm(xIdx, gammaIdx = -1, eps = 1e-6) {
    const x = this.tape.entries[xIdx].output;
    const n = x.len;
    const out = Tensor.zeros(x.shape);
    let ss = 0;
    for (let i = 0; i < n; i++) ss += x.data[i] * x.data[i];
    const rms = Math.sqrt(ss / n + eps);
    if (gammaIdx >= 0) {
      const g = this.tape.entries[gammaIdx].output.data;
      const gl = g.length;
      for (let i = 0; i < n; i++) out.data[i] = (x.data[i] / rms) * g[i % gl];
    } else {
      for (let i = 0; i < n; i++) out.data[i] = x.data[i] / rms;
    }
    return this.tape.record(out, OP.RMSNORM, xIdx, gammaIdx);
  }

  seqRmsnorm(xIdx, gammaIdx, T, D, eps = 1e-6) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros([T, D]);
    const gamma = gammaIdx >= 0 ? this.tape.entries[gammaIdx].output.data : null;
    for (let t = 0; t < T; t++) {
      const off = t * D;
      let ss = 0;
      for (let d = 0; d < D; d++) ss += x.data[off + d] * x.data[off + d];
      const rms = Math.sqrt(ss / D + eps);
      for (let d = 0; d < D; d++) {
        let v = x.data[off + d] / rms;
        if (gamma) v *= gamma[d];
        out.data[off + d] = v;
      }
    }
    return this.tape.record(out, OP.SEQ_RMSNORM, xIdx, gammaIdx, -1, T, D);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // EMBEDDING
  // ═════════════════════════════════════════════════════════════════════════

  /** Single-token embed (slow path for inference one token at a time). */
  embedding1(tableIdx, tokenId) {
    const W = this.tape.entries[tableIdx].output;
    const cols = W.shape.length >= 2 ? W.shape[1] : W.len;
    const out = Tensor.zeros([cols]);
    out.data.set(W.data.subarray(tokenId * cols, (tokenId + 1) * cols));
    return this.tape.record(out, OP.EMB_LOOKUP, tableIdx, -1, -1, tokenId);
  }

  /**
   * Sequence embedding: y[t,d] = table[ids[t], d]. ids is a tensor of token IDs.
   * Returns tape index.
   */
  embedding(tableIdx, idsIdx, T, D) {
    const W = this.tape.entries[tableIdx].output;
    const ids = this.tape.entries[idsIdx].output.data;
    const rows = (W.len / D) | 0;
    const out = Tensor.zeros([T, D]);
    for (let t = 0; t < T; t++) {
      let tid = ids[t] | 0;
      if (tid < 0) tid = 0;
      if (tid >= rows) tid = rows - 1;
      out.data.set(W.data.subarray(tid * D, (tid + 1) * D), t * D);
    }
    return this.tape.record(out, OP.EMBEDDING, tableIdx, idsIdx, -1, T, D);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // ATTENTION
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Multi-head causal self-attention. Q/K/V: [T, n_heads * head_dim].
   * Single fused op with backward via tape (MH_CAUSAL_ATTN).
   * If `mask` is null, uses standard causal mask (j <= i).
   */
  attention(qIdx, kIdx, vIdx, T, headDim) {
    const Q = this.tape.entries[qIdx].output.data;
    const K = this.tape.entries[kIdx].output.data;
    const V = this.tape.entries[vIdx].output.data;
    const D = (this.tape.entries[qIdx].output.len / T) | 0;
    const nHeads = (D / headDim) | 0;
    const sc = 1 / Math.sqrt(headDim);
    const out = Tensor.zeros([T, D]);
    const Yd = out.data;
    for (let h = 0; h < nHeads; h++) {
      const ho = h * headDim;
      for (let i = 0; i < T; i++) {
        const qOff = i * D + ho;
        const scores = new Float32Array(i + 1);
        let mx = -Infinity;
        for (let j = 0; j <= i; j++) {
          const kOff = j * D + ho;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += Q[qOff + d] * K[kOff + d];
          scores[j] = dot * sc;
          if (scores[j] > mx) mx = scores[j];
        }
        let sm = 0;
        for (let j = 0; j <= i; j++) { scores[j] = Math.exp(scores[j] - mx); sm += scores[j]; }
        if (sm > 0) for (let j = 0; j <= i; j++) scores[j] /= sm;
        // out[i, ho:ho+headDim] = Σ_j attn[j] * V[j, ho:ho+headDim]
        for (let j = 0; j <= i; j++) {
          const vOff = j * D + ho;
          const aj = scores[j];
          for (let d = 0; d < headDim; d++) Yd[i * D + ho + d] += aj * V[vOff + d];
        }
      }
    }
    return this.tape.record(out, OP.MH_CAUSAL_ATTN, qIdx, kIdx, vIdx, T, headDim);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // SwiGLU MLP — full block: gate = silu(x @ W1), out = (gate * (x @ W3)) @ W2
  // Matches LLaMA-style FFN: gate_proj, up_proj, down_proj.
  // Returns tape index of the down projection output.
  // ═════════════════════════════════════════════════════════════════════════

  swigluFFN(xIdx, w1Idx, w2Idx, w3Idx, T) {
    // gate = W1 @ X, up = W3 @ X (per-token)
    const gateIdx = this.seqLinear(w1Idx, xIdx, T);
    const upIdx   = this.seqLinear(w3Idx, xIdx, T);
    // h = silu(gate) * up — fused in OP.SWIGLU
    const hIdx    = this.swiglu(gateIdx, upIdx);
    // out = W2 @ h
    return this.seqLinear(w2Idx, hIdx, T);
  }

  /** Element-wise SwiGLU: silu(gate) * up. */
  swiglu(gateIdx, upIdx) {
    const g = this.tape.entries[gateIdx].output;
    const u = this.tape.entries[upIdx].output;
    const out = Tensor.zeros(g.shape);
    for (let i = 0; i < g.len; i++) {
      const gi = g.data[i];
      const sig = 1 / (1 + Math.exp(-gi));
      out.data[i] = (gi * sig) * u.data[i];
    }
    return this.tape.record(out, OP.SWIGLU, gateIdx, upIdx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // GEGLU FFN — fused: y = GELU(x @ W1^T) * (x @ W2^T). Gemma-3 style FFN.
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Fused GEGLU. x:[T, D_in], W1/W2:[D_out, D_in], output [T, D_out].
   * Mirrors C nt_geglu (notorch.c:3090). Tanh GELU approximation matches
   * PyTorch `gelu(approximate="tanh")` and the C path bit-for-bit
   * (constants 0.7978845608, 0.044715).
   */
  geglu(xIdx, w1Idx, w2Idx, T, dIn, dOut) {
    const x = this.tape.entries[xIdx].output;
    const W1 = this.tape.entries[w1Idx].output;
    const W2 = this.tape.entries[w2Idx].output;
    const out = Tensor.zeros([T, dOut]);
    const k = 0.7978845608;
    for (let t = 0; t < T; t++) {
      const xOff = t * dIn;
      const yOff = t * dOut;
      for (let i = 0; i < dOut; i++) {
        const wRow = i * dIn;
        let gate = 0, val = 0;
        for (let j = 0; j < dIn; j++) {
          gate += W1.data[wRow + j] * x.data[xOff + j];
          val  += W2.data[wRow + j] * x.data[xOff + j];
        }
        const g3 = gate * gate * gate;
        const inner = k * (gate + 0.044715 * g3);
        const gelu = 0.5 * gate * (1 + Math.tanh(inner));
        out.data[yOff + i] = Math.fround(gelu * val);
      }
    }
    return this.tape.record(out, OP.GEGLU, xIdx, w1Idx, w2Idx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // ROPE — rotary positional embedding (in-place style; new tensor with rotation)
  // ═════════════════════════════════════════════════════════════════════════

  rope(xIdx, T, headDim, freqBase = 10000) {
    const x = this.tape.entries[xIdx].output;
    const out = Tensor.zeros(x.shape);
    const total = x.len;
    const D = (total / T) | 0;
    const nHeads = (D / headDim) | 0;
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < nHeads; h++) {
        const off = t * D + h * headDim;
        for (let k = 0; k < headDim; k += 2) {
          const theta = t / Math.pow(freqBase, k / headDim);
          const c = Math.cos(theta), s = Math.sin(theta);
          const x0 = x.data[off + k], x1 = x.data[off + k + 1];
          out.data[off + k]     = x0 * c - x1 * s;
          out.data[off + k + 1] = x0 * s + x1 * c;
        }
      }
    }
    return this.tape.record(out, OP.ROPE, xIdx, -1, -1, T, headDim, freqBase);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // DROPOUT — train-only; saves the mask on the tape entry for backward.
  // ═════════════════════════════════════════════════════════════════════════

  dropout(xIdx, p) {
    const x = this.tape.entries[xIdx].output;
    if (!this.tape.training || p <= 0) {
      const out = x.clone();
      return this.tape.record(out, OP.NONE, xIdx);
    }
    const out = Tensor.zeros(x.shape);
    const mask = new Float32Array(x.len);
    const keep = 1 - p;
    const inv = 1 / keep;
    for (let i = 0; i < x.len; i++) {
      const m = Math.random() < keep ? inv : 0;
      mask[i] = m;
      out.data[i] = x.data[i] * m;
    }
    const idx = this.tape.record(out, OP.DROPOUT, xIdx);
    this.tape.entries[idx]._mask = mask;
    return idx;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // CONCAT / SLICE — needed for KV cache and friends
  // ═════════════════════════════════════════════════════════════════════════

  /** Per-position concat: a:[T,Da], b:[T,Db] → [T, Da+Db]. */
  concat(aIdx, bIdx, T) {
    const a = this.tape.entries[aIdx].output;
    const b = this.tape.entries[bIdx].output;
    const Da = (a.len / T) | 0, Db = (b.len / T) | 0;
    const out = Tensor.zeros([T, Da + Db]);
    const stride = Da + Db;
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < Da; d++) out.data[t * stride + d] = a.data[t * Da + d];
      for (let d = 0; d < Db; d++) out.data[t * stride + Da + d] = b.data[t * Db + d];
    }
    return this.tape.record(out, OP.CONCAT, aIdx, bIdx, -1, T, Da, Db);
  }

  /** Slice along an axis (inference helper, no autograd). */
  slice(aIdx, start, end, dim = 0) {
    const a = this.tape.entries[aIdx].output;
    const shape = a.shape.slice();
    shape[dim] = end - start;
    const out = Tensor.zeros(shape);
    if (dim === 0) {
      const stride = a.len / a.shape[0];
      out.data.set(a.data.subarray(start * stride, end * stride));
    } else {
      // Generic for dim=1 in 2D: per-row slice
      if (a.shape.length === 2 && dim === 1) {
        const [R, C] = a.shape;
        for (let r = 0; r < R; r++)
          for (let c = start; c < end; c++)
            out.data[r * (end - start) + (c - start)] = a.data[r * C + c];
      } else {
        throw new Error(`slice: dim=${dim} on rank ${a.shape.length} not supported`);
      }
    }
    return this.tape.leaf(out);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // LOSSES
  // ═════════════════════════════════════════════════════════════════════════

  /** Single-position cross-entropy. logits:[V], target: int. */
  crossEntropyLoss(logitsIdx, target) {
    const logits = this.tape.entries[logitsIdx].output;
    const n = logits.len;
    let mx = logits.data[0];
    for (let i = 1; i < n; i++) if (logits.data[i] > mx) mx = logits.data[i];
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.exp(logits.data[i] - mx);
    const logZ = mx + Math.log(sum);
    const loss = -(logits.data[target] - logZ);
    const out = new Tensor(new Float32Array([loss]), [1]);
    return this.tape.record(out, OP.CROSS_ENT, logitsIdx, -1, -1, target);
  }

  /** Sequence cross-entropy. logits:[T,V], targets tape idx points to int tensor [T]. */
  seqCrossEntropyLoss(logitsIdx, targetsIdx, T, V) {
    const logits = this.tape.entries[logitsIdx].output;
    const targets = this.tape.entries[targetsIdx].output.data;
    let lossSum = 0;
    for (let t = 0; t < T; t++) {
      const off = t * V;
      let mx = logits.data[off];
      for (let j = 1; j < V; j++) if (logits.data[off + j] > mx) mx = logits.data[off + j];
      let sum = 0;
      for (let j = 0; j < V; j++) sum += Math.exp(logits.data[off + j] - mx);
      const logZ = mx + Math.log(sum);
      const tgt = targets[t] | 0;
      lossSum += -(logits.data[off + tgt] - logZ);
    }
    const out = new Tensor(new Float32Array([lossSum / T]), [1]);
    return this.tape.record(out, OP.SEQ_CROSSENT, logitsIdx, targetsIdx, -1, T, V);
  }

  /**
   * Masked sequence cross-entropy — assistant-only SFT loss.
   * logits:[T,V], targets:[T] (int), mask:[T] (float, 0=skip). Loss is
   * averaged over the `mask[t] > 0` positions only, so prompt tokens with
   * mask=0 contribute neither to forward nor backward. Mirrors C
   * nt_seq_cross_entropy_masked at notorch.c:3778; sister of
   * `seqCrossEntropyLoss`.
   */
  seqCrossEntropyLossMasked(logitsIdx, targetsIdx, maskIdx, T, V) {
    const logits = this.tape.entries[logitsIdx].output;
    const targets = this.tape.entries[targetsIdx].output.data;
    const mask = this.tape.entries[maskIdx].output.data;
    let totalLoss = 0;
    let nActive = 0;
    for (let t = 0; t < T; t++) {
      const m = mask[t];
      if (m === 0) continue;
      const off = t * V;
      let mx = logits.data[off];
      for (let j = 1; j < V; j++) if (logits.data[off + j] > mx) mx = logits.data[off + j];
      let sum = 0;
      for (let j = 0; j < V; j++) sum += Math.exp(logits.data[off + j] - mx);
      let tgt = targets[t] | 0;
      if (tgt < 0 || tgt >= V) tgt = 0;
      totalLoss += m * -(logits.data[off + tgt] - mx - Math.log(sum));
      nActive += m;
    }
    const out = new Tensor(new Float32Array([nActive > 0 ? totalLoss / nActive : 0]), [1]);
    return this.tape.record(out, OP.SEQ_CROSSENT_MASKED, logitsIdx, targetsIdx, maskIdx, T, V);
  }

  /** Mean-squared error. pred and target both Tensors of same shape. */
  mseLoss(predIdx, targetIdx) {
    const p = this.tape.entries[predIdx].output;
    const t = this.tape.entries[targetIdx].output;
    let s = 0;
    for (let i = 0; i < p.len; i++) { const d = p.data[i] - t.data[i]; s += d * d; }
    const out = new Tensor(new Float32Array([s / p.len]), [1]);
    return this.tape.record(out, OP.MSE, predIdx, targetIdx);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // SAMPLING
  // ═════════════════════════════════════════════════════════════════════════

  /** Greedy argmax over dim=-1 (returns int). */
  argmax(tensor) {
    const data = tensor.data || tensor;  // accept Tensor or Float32Array
    let bi = 0, bv = data[0];
    for (let i = 1; i < data.length; i++) if (data[i] > bv) { bv = data[i]; bi = i; }
    return bi;
  }

  /**
   * Sample one token from a logits vector. Supports temperature, top-k, top-p.
   * temperature <= 0 → greedy.
   */
  sample(logits, { temperature = 1.0, topK = 0, topP = 0 } = {}) {
    const data = logits.data || logits;
    const V = data.length;
    if (temperature <= 0) return this.argmax(data);
    // Apply temperature, find max for stability
    const scaled = new Float32Array(V);
    let mx = -Infinity;
    for (let i = 0; i < V; i++) {
      scaled[i] = data[i] / temperature;
      if (scaled[i] > mx) mx = scaled[i];
    }
    let probs = new Float32Array(V);
    let sum = 0;
    for (let i = 0; i < V; i++) { probs[i] = Math.exp(scaled[i] - mx); sum += probs[i]; }
    for (let i = 0; i < V; i++) probs[i] /= sum;

    // Index list for top-k / top-p filtering
    let idx = null;
    if (topK > 0 && topK < V) {
      idx = Array.from({ length: V }, (_, i) => i);
      idx.sort((a, b) => probs[b] - probs[a]);
      idx = idx.slice(0, topK);
    }
    if (topP > 0 && topP < 1) {
      if (!idx) {
        idx = Array.from({ length: V }, (_, i) => i);
        idx.sort((a, b) => probs[b] - probs[a]);
      }
      let cum = 0;
      const cutoff = [];
      for (const i of idx) {
        cum += probs[i];
        cutoff.push(i);
        if (cum >= topP) break;
      }
      idx = cutoff;
    }

    if (idx) {
      let s = 0;
      for (const i of idx) s += probs[i];
      const r = Math.random() * s;
      let acc = 0;
      for (const i of idx) {
        acc += probs[i];
        if (r <= acc) return i;
      }
      return idx[idx.length - 1];
    }
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < V; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return V - 1;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // GPU MATMUL (tiled, with buffer pool)
  // ═════════════════════════════════════════════════════════════════════════

  /** Acquire a GPU buffer of given byte size from pool, or create new. */
  _acquireBuffer(byteSize, usage) {
    const key = `${byteSize}|${usage}`;
    const list = this.bufferPool.get(key);
    if (list && list.length > 0) {
      const buf = list.pop();
      this.allocatedBuffers.add(buf);
      return buf;
    }
    const buf = this.device.createBuffer({ size: byteSize, usage });
    this.allocatedBuffers.add(buf);
    buf._poolKey = key;
    return buf;
  }

  /** Return a buffer to the pool (or destroy if pool too big). */
  _releaseBuffer(buf) {
    if (!buf || !buf._poolKey) { try { buf.destroy(); } catch {} return; }
    this.allocatedBuffers.delete(buf);
    const list = this.bufferPool.get(buf._poolKey) || [];
    if (list.length < 8) { list.push(buf); this.bufferPool.set(buf._poolKey, list); }
    else { try { buf.destroy(); } catch {} }
  }

  /** Free all pooled buffers. Call before tearing down the engine. */
  cleanup() {
    for (const list of this.bufferPool.values()) {
      for (const b of list) try { b.destroy(); } catch {}
    }
    this.bufferPool.clear();
    for (const b of this.allocatedBuffers) try { b.destroy(); } catch {}
    this.allocatedBuffers.clear();
  }

  async matmulGPU(A, B) {
    const [M, K] = A.shape;
    const [, N] = B.shape;
    const out = Tensor.zeros([M, N]);
    const dev = this.device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const bufA = this._acquireBuffer(A.data.byteLength, usage);
    const bufB = this._acquireBuffer(B.data.byteLength, usage);
    const bufC = this._acquireBuffer(out.data.byteLength, usage);
    const bufU = this._acquireBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    dev.queue.writeBuffer(bufA, 0, A.data);
    dev.queue.writeBuffer(bufB, 0, B.data);
    dev.queue.writeBuffer(bufU, 0, new Uint32Array([M, N, K, 0]));

    const pipeline = await this.getPipeline("matmul_tiled", TILED_MATMUL_WGSL);
    const bindGroup = dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufU } },
      ],
    });

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
    pass.end();

    const readBuf = dev.createBuffer({
      size: out.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    enc.copyBufferToBuffer(bufC, 0, readBuf, 0, out.data.byteLength);
    dev.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    out.data.set(new Float32Array(readBuf.getMappedRange()));
    readBuf.unmap();
    readBuf.destroy();

    this._releaseBuffer(bufA);
    this._releaseBuffer(bufB);
    this._releaseBuffer(bufC);
    this._releaseBuffer(bufU);
    return out;
  }

  async getPipeline(name, code) {
    if (this.pipelines.has(name)) return this.pipelines.get(name);
    const module = this.device.createShaderModule({ code });
    const pipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// KV CACHE — append-only per-layer K/V buffer, recall by position.
// Separate from the tape — KV cache is for inference, no gradients.
// ═══════════════════════════════════════════════════════════════════════════

export class KVCache {
  /**
   * @param {number} maxLen   maximum context length
   * @param {number} dim      total per-token K/V dimension (n_heads * head_dim)
   */
  constructor(maxLen, dim) {
    this.maxLen = maxLen;
    this.dim = dim;
    this.K = new Float32Array(maxLen * dim);
    this.V = new Float32Array(maxLen * dim);
    this.length = 0;
  }

  /** Append one token's K and V. Both vectors must be length=dim. */
  append(kVec, vVec) {
    if (this.length >= this.maxLen) throw new Error("KV cache full");
    this.K.set(kVec, this.length * this.dim);
    this.V.set(vVec, this.length * this.dim);
    this.length++;
  }

  /** Bulk append T tokens at once (KSeq/VSeq each [T, dim]). */
  appendBatch(KSeq, VSeq, T) {
    if (this.length + T > this.maxLen) throw new Error("KV cache full");
    this.K.set(KSeq.subarray(0, T * this.dim), this.length * this.dim);
    this.V.set(VSeq.subarray(0, T * this.dim), this.length * this.dim);
    this.length += T;
  }

  /** Read K up to current length as a tensor view. */
  Ktensor() { return new Tensor(this.K.subarray(0, this.length * this.dim), [this.length, this.dim]); }
  Vtensor() { return new Tensor(this.V.subarray(0, this.length * this.dim), [this.length, this.dim]); }

  reset() { this.length = 0; }
}

// ═══════════════════════════════════════════════════════════════════════════
// NOTORCH LoRA — low-rank adapters on a frozen base weight
// Mirrors notorch.c nt_lora_* (commit f995aae / 1de7c69, branch
// lora-primitives-restore-2026-05-10). File format is byte-compatible with
// the C side so checkpoints cross-load between C trainers and JS inference.
//
// Forward:  y = W_frozen @ x + (alpha/rank) * B @ (A @ x)
//   A : [rank, in_dim]   — kaimingUniform_(fanIn=inDim) init, trainable
//   B : [out_dim, rank]  — zeros init  (so initial Δ output = 0)
//   W : [out_dim, in_dim] — supplied externally via tape.paramFrozen(W)
// ═══════════════════════════════════════════════════════════════════════════

const NT_LORA_MAGIC   = 0x4C4F5241;  // 'LORA'
const NT_LORA_VERSION = 1;

/**
 * LoRA pair: persistent A, B tensors plus rank / alpha / scaling metadata.
 * Owns its tensors across training steps; forward() re-registers A and B
 * into each step's tape so Chuck advances their optimizer state correctly.
 */
export class LoRAPair {
  /**
   * @param {number} inDim   input feature dim
   * @param {number} outDim  output feature dim
   * @param {number} rank    low-rank dim (typically 8..64)
   * @param {number} alpha   LoRA alpha (scaling = alpha / rank)
   */
  constructor(inDim, outDim, rank, alpha) {
    if (inDim <= 0 || outDim <= 0 || rank <= 0) {
      throw new Error(`LoRAPair: dims must be > 0 (in=${inDim}, out=${outDim}, rank=${rank})`);
    }
    this.inDim   = inDim;
    this.outDim  = outDim;
    this.rank    = rank;
    this.alpha   = alpha;
    this.scaling = alpha / rank;
    this.A = Tensor.zeros([rank, inDim]).kaimingUniform_(inDim);
    this.B = Tensor.zeros([outDim, rank]);   // zeros init — Δ output = 0 at step 0
  }

  /**
   * Per-step forward: registers A,B as trainable in the active tape, then
   * composes y = seqLinear(W,x,T) + scaling * seqLinear(B, seqLinear(A,x,T), T).
   *
   * MUST be called inside an active tape (caller's responsibility), and the
   * base weight W identified by `wIdx` MUST have been registered via
   * `tape.paramFrozen(W)` so optimizer slot indexing stays clean.
   *
   * @param {Notorch} engine  notorch engine (provides tape + seqLinear + add + scale)
   * @param {number}  wIdx    tape entry index of the frozen base weight W
   * @param {number}  xIdx    tape entry index of input x [T, inDim]
   * @param {number}  T       sequence length
   * @returns {number} tape entry index of y (the final sum)
   */
  forward(engine, wIdx, xIdx, T) {
    if (wIdx < 0 || xIdx < 0) throw new Error("LoRAPair.forward: invalid wIdx/xIdx");
    // Register persistent A,B as trainable in THIS step's tape. Chuck slot
    // allocation for them happens here, AFTER any base paramFrozen() calls,
    // so Chuck slot indices stay clean for the optimizer.
    const aIdx = engine.tape.param(this.A);
    const bIdx = engine.tape.param(this.B);
    // Compose y = W@x + scaling * B@(A@x)
    const wxIdx     = engine.seqLinear(wIdx, xIdx, T);    // [T, outDim]
    const axIdx     = engine.seqLinear(aIdx, xIdx, T);    // [T, rank]
    const baxIdx    = engine.seqLinear(bIdx, axIdx, T);   // [T, outDim]
    const scaledIdx = engine.scale(baxIdx, this.scaling);
    return engine.add(wxIdx, scaledIdx);
  }
}

/**
 * Serialize all LoRA pairs into a single binary blob (mirrors C nt_lora_save).
 * `pairs` is flat-indexed [layer * numTargets + targetIdx]; all pairs must
 * share rank / alpha / inDim / outDim (single-shape per artifact — see
 * notorch/CLAUDE.md "Single-shape nt_lora_save per file").
 *
 * File format (little-endian, byte-compatible with C):
 *   [u32 magic 'LORA'][u32 version=1]
 *   [u32 num_targets]
 *   [per-target: u8 namelen, namelen × ASCII bytes]
 *   [u32 num_layers][u32 rank]
 *   [f32 alpha (raw IEEE-754 bytes — NOT alpha*1000; header docstring stale)]
 *   [u32 in_dim][u32 out_dim]
 *   [for L in [0,num_layers): for T in [0,num_targets):
 *       A floats (rank*in_dim), B floats (out_dim*rank)]
 *
 * @param {LoRAPair[]} pairs       flat-indexed [layer * numTargets + targetIdx]
 * @param {number}     numLayers
 * @param {number}     numTargets
 * @param {string[]}   targetNames length=numTargets, ASCII, ≤255 bytes each
 * @returns {Uint8Array} binary blob; caller persists (e.g. fs.writeFileSync in Node)
 */
export function saveLoRA(pairs, numLayers, numTargets, targetNames) {
  if (!Array.isArray(pairs) || !Array.isArray(targetNames)) {
    throw new Error("saveLoRA: pairs and targetNames must be arrays");
  }
  if (numLayers <= 0 || numTargets <= 0) {
    throw new Error(`saveLoRA: numLayers/numTargets must be > 0 (got ${numLayers}, ${numTargets})`);
  }
  if (targetNames.length !== numTargets) {
    throw new Error(`saveLoRA: targetNames.length=${targetNames.length} != numTargets=${numTargets}`);
  }
  if (pairs.length !== numLayers * numTargets) {
    throw new Error(`saveLoRA: pairs.length=${pairs.length} != numLayers*numTargets=${numLayers * numTargets}`);
  }

  // Validate ALL pairs first (matches C: dimension check pre-fopen to avoid
  // truncating a destination file on shape mismatch — JS has no destination
  // yet, but the discipline catches the bug at the same point).
  const rank   = pairs[0].rank;
  const inDim  = pairs[0].inDim;
  const outDim = pairs[0].outDim;
  const alpha  = pairs[0].alpha;
  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    if (!p || !p.A || !p.B) throw new Error(`saveLoRA: pair[${i}] missing A/B`);
    if (p.rank !== rank || p.inDim !== inDim || p.outDim !== outDim || p.alpha !== alpha) {
      throw new Error(`saveLoRA: pair[${i}] shape/alpha mismatch (single-shape per artifact)`);
    }
  }

  // Pre-encode names so we know the total byte size up front.
  const encoder = new TextEncoder();
  const nameBytes = new Array(numTargets);
  let namesByteLen = 0;
  for (let t = 0; t < numTargets; t++) {
    const name = targetNames[t] || "";
    let bytes = encoder.encode(name);
    if (bytes.length > 255) bytes = bytes.subarray(0, 255);   // C clamps to u8
    nameBytes[t] = bytes;
    namesByteLen += 1 + bytes.length;   // u8 namelen + bytes
  }

  const aN = rank * inDim;
  const bN = outDim * rank;
  const headerBytes = 4 + 4 + 4 + namesByteLen + 4 + 4 + 4 + 4 + 4;   // magic..outDim
  const tensorBytes = numLayers * numTargets * (aN + bN) * 4;
  const totalBytes  = headerBytes + tensorBytes;

  const buf = new ArrayBuffer(totalBytes);
  const dv  = new DataView(buf);
  const u8  = new Uint8Array(buf);
  let off = 0;

  dv.setUint32(off, NT_LORA_MAGIC, true);  off += 4;
  dv.setUint32(off, NT_LORA_VERSION, true); off += 4;
  dv.setUint32(off, numTargets, true);     off += 4;
  for (let t = 0; t < numTargets; t++) {
    const nb = nameBytes[t];
    u8[off++] = nb.length;
    u8.set(nb, off);
    off += nb.length;
  }
  dv.setUint32(off, numLayers, true);      off += 4;
  dv.setUint32(off, rank, true);           off += 4;
  // Alpha as raw f32 little-endian (NOT alpha*1000 — see notorch/CLAUDE.md;
  // we verified empirically that nt_lora_save writes float bits).
  dv.setFloat32(off, alpha, true);         off += 4;
  dv.setUint32(off, inDim, true);          off += 4;
  dv.setUint32(off, outDim, true);         off += 4;

  // Tensor payload — A then B per pair, layer-major.
  // Use Float32Array view at correct offset for fast bulk copy.
  // NOTE: the offset is always 4-byte aligned because everything before
  // the tensor block sums to (3*4 + namesByteLen + 5*4) bytes; the only
  // odd-sized chunks are the name bytes themselves, but the C side writes
  // them with no padding and reads them back the same way — so any padding
  // here would break cross-load. We must write floats by setFloat32 per
  // element OR pre-copy the whole tensor block to a temp aligned buffer.
  // The names sum to (numTargets * 1) + sum(name_bytes), which is generally
  // NOT a multiple of 4 → use DataView.setFloat32 per-element to stay safe
  // across all alignments.
  for (let L = 0; L < numLayers; L++) {
    for (let T = 0; T < numTargets; T++) {
      const p = pairs[L * numTargets + T];
      const aData = p.A.data;
      for (let k = 0; k < aN; k++) { dv.setFloat32(off, aData[k], true); off += 4; }
      const bData = p.B.data;
      for (let k = 0; k < bN; k++) { dv.setFloat32(off, bData[k], true); off += 4; }
    }
  }

  return new Uint8Array(buf);
}

/**
 * Deserialize a LoRA artifact (mirrors C nt_lora_load). Caller pre-allocates
 * `pairs` (via `new LoRAPair(...)`) so dims/rank/alpha can be validated
 * against the file header before any A/B bytes are read.
 *
 * Validates magic, version, numTargets, numLayers, dims, rank, alpha
 * (tolerance 1e-4 — float exact equality is brittle even across raw-byte
 * round-trips), and target names. Throws on any mismatch.
 *
 * @param {Uint8Array} blob        binary artifact (e.g. fs.readFileSync output)
 * @param {LoRAPair[]} pairs       flat-indexed, length=numLayers*numTargets
 * @param {number}     numLayers
 * @param {number}     numTargets
 * @param {string[]}   targetNames length=numTargets
 */
export function loadLoRA(blob, pairs, numLayers, numTargets, targetNames) {
  if (!(blob instanceof Uint8Array)) throw new Error("loadLoRA: blob must be Uint8Array");
  if (!Array.isArray(pairs) || !Array.isArray(targetNames)) {
    throw new Error("loadLoRA: pairs and targetNames must be arrays");
  }
  if (numLayers <= 0 || numTargets <= 0) {
    throw new Error(`loadLoRA: numLayers/numTargets must be > 0`);
  }
  if (pairs.length !== numLayers * numTargets) {
    throw new Error(`loadLoRA: pairs.length=${pairs.length} != ${numLayers * numTargets}`);
  }
  if (targetNames.length !== numTargets) {
    throw new Error(`loadLoRA: targetNames.length=${targetNames.length} != numTargets=${numTargets}`);
  }

  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  let off = 0;
  const decoder = new TextDecoder("utf-8");

  const magic = dv.getUint32(off, true); off += 4;
  if (magic !== NT_LORA_MAGIC) throw new Error(`loadLoRA: bad magic 0x${magic.toString(16)}`);
  const version = dv.getUint32(off, true); off += 4;
  if (version !== NT_LORA_VERSION) throw new Error(`loadLoRA: unsupported version ${version}`);
  const fileNumTargets = dv.getUint32(off, true); off += 4;
  if (fileNumTargets !== numTargets) {
    throw new Error(`loadLoRA: numTargets mismatch (file=${fileNumTargets}, caller=${numTargets})`);
  }

  for (let t = 0; t < numTargets; t++) {
    const nl = blob[off]; off += 1;
    const nameBytes = blob.subarray(off, off + nl);
    off += nl;
    const name = decoder.decode(nameBytes);
    if (targetNames[t] != null && name !== targetNames[t]) {
      throw new Error(`loadLoRA: target[${t}] name mismatch (file="${name}", caller="${targetNames[t]}")`);
    }
  }

  const fileNumLayers = dv.getUint32(off, true); off += 4;
  if (fileNumLayers !== numLayers) {
    throw new Error(`loadLoRA: numLayers mismatch (file=${fileNumLayers}, caller=${numLayers})`);
  }
  const rank   = dv.getUint32(off, true);  off += 4;
  const alpha  = dv.getFloat32(off, true); off += 4;
  const inDim  = dv.getUint32(off, true);  off += 4;
  const outDim = dv.getUint32(off, true);  off += 4;

  // Validate caller's pre-allocated pairs against the header (matches C).
  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    if (p.rank !== rank || p.inDim !== inDim || p.outDim !== outDim) {
      throw new Error(`loadLoRA: pair[${i}] dim mismatch (file rank=${rank} in=${inDim} out=${outDim})`);
    }
    if (Math.abs(p.alpha - alpha) > 1e-4) {
      throw new Error(`loadLoRA: pair[${i}] alpha mismatch (file=${alpha}, caller=${p.alpha})`);
    }
  }

  const aN = rank * inDim;
  const bN = outDim * rank;
  for (let L = 0; L < numLayers; L++) {
    for (let T = 0; T < numTargets; T++) {
      const p = pairs[L * numTargets + T];
      // Per-element setFloat32 on save → per-element getFloat32 on load.
      // (Cannot use Float32Array view because `off` is rarely 4-byte aligned
      // relative to blob.byteOffset after the variable-length name section.)
      for (let k = 0; k < aN; k++) { p.A.data[k] = dv.getFloat32(off, true); off += 4; }
      for (let k = 0; k < bN; k++) { p.B.data[k] = dv.getFloat32(off, true); off += 4; }
    }
  }
}

/**
 * Merge LoRA delta into a CPU float buffer (mirrors C nt_lora_merge_into):
 *     W_dst[i,j] = W_frozen[i,j] + scaling * sum_k B[i,k] * A[k,j]
 * Layout: row-major [outDim, inDim]. WDst and WFrozen MAY alias for in-place.
 *
 * @param {Float32Array} WDst     destination buffer, length outDim*inDim
 * @param {Float32Array} WFrozen  source frozen weight, length outDim*inDim
 * @param {LoRAPair}     pair
 * @param {number}       inDim
 * @param {number}       outDim
 */
export function mergeLoRAInto(WDst, WFrozen, pair, inDim, outDim) {
  if (!(WDst instanceof Float32Array) || !(WFrozen instanceof Float32Array)) {
    throw new Error("mergeLoRAInto: WDst and WFrozen must be Float32Array");
  }
  if (!pair || !pair.A || !pair.B) throw new Error("mergeLoRAInto: pair missing A/B");
  if (inDim <= 0 || outDim <= 0) throw new Error("mergeLoRAInto: dims must be > 0");
  if (pair.inDim !== inDim || pair.outDim !== outDim) {
    throw new Error(`mergeLoRAInto: pair dim mismatch (pair in=${pair.inDim} out=${pair.outDim}, caller in=${inDim} out=${outDim})`);
  }
  const need = outDim * inDim;
  if (WDst.length < need || WFrozen.length < need) {
    throw new Error(`mergeLoRAInto: buffers too small (need ${need})`);
  }
  const rank  = pair.rank;
  const scale = pair.scaling;
  const A = pair.A.data;  // [rank, inDim]
  const B = pair.B.data;  // [outDim, rank]
  // Δ[i,j] = sum_k B[i,k] * A[k,j], then W_dst[i,j] = W_frozen[i,j] + scale*Δ[i,j].
  // Fuse the two: accumulate Δ into a scalar, write once. Saves the temp.
  for (let i = 0; i < outDim; i++) {
    const bRow = i * rank;
    const wRow = i * inDim;
    for (let j = 0; j < inDim; j++) {
      let d = 0;
      for (let k = 0; k < rank; k++) d += B[bRow + k] * A[k * inDim + j];
      WDst[wRow + j] = WFrozen[wRow + j] + scale * d;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// WEIGHT LOADERS
// ═══════════════════════════════════════════════════════════════════════════

const NT_MAGIC = 0x4E544F52;  // 'NTOR' little-endian as in notorch.c

/**
 * Load notorch native .bin format. Layout: [magic u32][n i32]
 *   per tensor: [ndim i32][shape_d * i32 ...][float32 data ...]
 * Returns Map<index_string, Tensor> — no names in the format, so caller maps by order.
 */
export function loadNotorchBin(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  let off = 0;
  const magic = dv.getUint32(off, true); off += 4;
  if (magic !== NT_MAGIC) throw new Error(`loadNotorchBin: bad magic 0x${magic.toString(16)}`);
  const n = dv.getInt32(off, true); off += 4;
  const out = new Map();
  for (let i = 0; i < n; i++) {
    const ndim = dv.getInt32(off, true); off += 4;
    const shape = [];
    let len = 1;
    for (let d = 0; d < ndim; d++) {
      const s = dv.getInt32(off, true); off += 4;
      shape.push(s);
      len *= s;
    }
    // Float32 view directly over the buffer (no copy).
    const data = new Float32Array(arrayBuffer, off, len);
    off += len * 4;
    // Copy out so the tensor owns its memory (callers typically reuse arrayBuffer).
    out.set(String(i), new Tensor(new Float32Array(data), shape));
  }
  return out;
}

/**
 * Load HuggingFace safetensors format.
 * Layout: [u64 header_len LE][JSON header][raw tensor data].
 * Returns Map<name, Tensor>. dtype must be F32 — others throw.
 */
export function loadSafetensors(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  // u64 header length, little-endian. JS BigInt then coerce.
  const hdrLenBig = dv.getBigUint64(0, true);
  const hdrLen = Number(hdrLenBig);
  const decoder = new TextDecoder("utf-8");
  const headerJSON = decoder.decode(new Uint8Array(arrayBuffer, 8, hdrLen));
  const header = JSON.parse(headerJSON);
  const dataStart = 8 + hdrLen;
  const out = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`safetensors: dtype ${info.dtype} not supported (only F32)`);
    const [start, end] = info.data_offsets;
    const len = (end - start) / 4;
    const view = new Float32Array(arrayBuffer, dataStart + start, len);
    out.set(name, new Tensor(new Float32Array(view), info.shape));
  }
  return out;
}

/** Save a Map<name, Tensor> as a notorch .bin (sorted by insertion order). */
export function saveNotorchBin(tensors) {
  const arr = [...tensors.values()];
  let total = 8;  // magic + n
  for (const t of arr) total += 4 + 4 * t.shape.length + t.data.byteLength;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  let off = 0;
  dv.setUint32(off, NT_MAGIC, true); off += 4;
  dv.setInt32(off, arr.length, true); off += 4;
  for (const t of arr) {
    dv.setInt32(off, t.shape.length, true); off += 4;
    for (const s of t.shape) { dv.setInt32(off, s, true); off += 4; }
    new Float32Array(buf, off, t.len).set(t.data);
    off += t.data.byteLength;
  }
  return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Char-level tokenizer. Pass an explicit vocab string OR call .fit(text)
 * to derive vocab from a sample.
 */
export class CharTokenizer {
  constructor(vocab = "") {
    this.itos = Array.from(vocab);
    this.stoi = new Map();
    this.itos.forEach((c, i) => this.stoi.set(c, i));
  }

  static fit(text) {
    const set = new Set(text);
    const vocab = Array.from(set).sort();
    return new CharTokenizer(vocab.join(""));
  }

  get vocabSize() { return this.itos.length; }

  encode(text) {
    const out = new Uint32Array(text.length);
    let n = 0;
    for (const c of text) {
      const id = this.stoi.get(c);
      if (id === undefined) continue;  // skip OOV
      out[n++] = id;
    }
    return out.subarray(0, n);
  }

  decode(ids) {
    let s = "";
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id < this.itos.length) s += this.itos[id];
    }
    return s;
  }
}

/**
 * Minimal BPE tokenizer. Build via `BPETokenizer.fromMerges(mergesText, vocabPath)`
 * where mergesText is a string of "a b\n" pairs and vocabPath is an optional
 * Map<string, int> for the base vocabulary (defaults to single-byte 0..255).
 */
export class BPETokenizer {
  constructor() {
    this.merges = [];        // [[a, b], ...] as token-id pairs
    this.mergeRank = new Map();   // "a,b" → priority (lower = earlier)
    this.itos = [];          // token id → byte sequence (Uint8Array)
    this.stoi = new Map();   // base token strings → ids (for tokenization)
  }

  static fromMerges(mergesText, baseVocab = null) {
    const tk = new BPETokenizer();
    // base vocab — bytes 0..255
    for (let i = 0; i < 256; i++) {
      const u8 = new Uint8Array([i]);
      tk.itos.push(u8);
      tk.stoi.set(String.fromCharCode(i), i);
    }
    // Optional named base vocab on top of byte vocab (safetensors-style HF)
    if (baseVocab && typeof baseVocab.entries === "function") {
      for (const [tok, id] of baseVocab.entries()) {
        if (id >= tk.itos.length) {
          while (tk.itos.length <= id) tk.itos.push(new Uint8Array(0));
        }
        tk.itos[id] = new TextEncoder().encode(tok);
        tk.stoi.set(tok, id);
      }
    }
    const lines = mergesText.split("\n").filter(l => l.trim() && !l.startsWith("#"));
    for (let r = 0; r < lines.length; r++) {
      const parts = lines[r].split(/\s+/);
      if (parts.length < 2) continue;
      const aStr = parts[0], bStr = parts[1];
      const aId = tk.stoi.get(aStr);
      const bId = tk.stoi.get(bStr);
      if (aId === undefined || bId === undefined) continue;
      const newId = tk.itos.length;
      const merged = new Uint8Array(tk.itos[aId].length + tk.itos[bId].length);
      merged.set(tk.itos[aId], 0);
      merged.set(tk.itos[bId], tk.itos[aId].length);
      tk.itos.push(merged);
      tk.stoi.set(aStr + bStr, newId);
      tk.merges.push([aId, bId]);
      tk.mergeRank.set(`${aId},${bId}`, r);
    }
    return tk;
  }

  get vocabSize() { return this.itos.length; }

  /** Greedy BPE encode: start from byte tokens, repeatedly merge lowest-rank pair. */
  encode(text) {
    // Start with single-byte tokens
    const bytes = new TextEncoder().encode(text);
    const ids = [];
    for (const b of bytes) ids.push(b);
    while (ids.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      let bestNew = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.mergeRank.get(`${ids[i]},${ids[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r; bestIdx = i; bestNew = 256 + r;
        }
      }
      if (bestIdx < 0) break;
      ids.splice(bestIdx, 2, bestNew);
    }
    return new Uint32Array(ids);
  }

  decode(ids) {
    let total = 0;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id < this.itos.length) total += this.itos[id].length;
    }
    const all = new Uint8Array(total);
    let off = 0;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id < this.itos.length) {
        all.set(this.itos[id], off);
        off += this.itos[id].length;
      }
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(all);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// LR SCHEDULES
// ═══════════════════════════════════════════════════════════════════════════

export const Schedule = {
  cosine: (baseLr, warmupSteps, totalSteps, minLr = 0) => ({
    type: "cosine", baseLr, warmupSteps, totalSteps, minLr, step: 0,
    get() {
      if (this.warmupSteps > 0 && this.step < this.warmupSteps) {
        return this.minLr + (this.baseLr - this.minLr) * (this.step / this.warmupSteps);
      }
      const progress = (this.step - this.warmupSteps) / Math.max(1, this.totalSteps - this.warmupSteps);
      const cos = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)));
      return this.minLr + (this.baseLr - this.minLr) * cos;
    },
    advance() { const lr = this.get(); this.step++; return lr; },
  }),
  step: (baseLr, warmupSteps, stepSize, gamma) => ({
    type: "step", baseLr, warmupSteps, stepSize, gamma, step: 0,
    get() {
      if (this.warmupSteps > 0 && this.step < this.warmupSteps) {
        return this.baseLr * (this.step / this.warmupSteps);
      }
      return this.baseLr * Math.pow(this.gamma, Math.floor((this.step - this.warmupSteps) / this.stepSize));
    },
    advance() { const lr = this.get(); this.step++; return lr; },
  }),
};

// ═══════════════════════════════════════════════════════════════════════════
// END OF notorch.js
// ═══════════════════════════════════════════════════════════════════════════
