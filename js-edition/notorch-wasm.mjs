// notorch-wasm.mjs — the SIMD kernels, for runtimes that have WebAssembly.
//
// notorch.js computes everything in plain JS and needs nothing else. This is
// the optional fast path: the same int8-activation matvec compiled to wasm with
// i8x16 SIMD, where sixteen products issue as one instruction. Plain JS has no
// way to express that, which is the entire reason for a second artifact.
//
// The .wasm is checked in. Nobody needs a toolchain to use it; wasm/build.sh is
// there for whoever changes the kernels.
//
// Approximate by construction, exactly as C's nt_qmatvec_i8 is: the activation
// is quantized per 32-value block. qmatvec in notorch.js stays the exact
// reference, and the gate holds this to the same 2e-2 the C test uses.
import { readFileSync } from 'node:fs';

const PAGE = 65536;

export class WasmKernels {
  /**
   * @param {ArrayBuffer|Uint8Array} [bytes] the module. In Node it is read from
   *   disk when omitted; in a browser, fetch it and pass it in.
   */
  static async create(bytes) {
    const self = new WasmKernels();
    if (!bytes) {
      const url = new URL('./wasm/qkernels.wasm', import.meta.url);
      bytes = readFileSync(url);
    }
    const { instance } = await WebAssembly.instantiate(
      bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes), {});
    self.exports = instance.exports;
    self.memory = instance.exports.memory;
    self._sync();
    self.top = PAGE;                 // first page reserved; allocations start after
    return self;
  }

  _sync() {
    this.u8 = new Uint8Array(this.memory.buffer);
    this.i8 = new Int8Array(this.memory.buffer);
    this.f32 = new Float32Array(this.memory.buffer);
    this.i32 = new Int32Array(this.memory.buffer);
  }

  /** Reserve `bytes` in the module's memory, growing it if needed. */
  alloc(bytes) {
    const align = 16;
    const ptr = (this.top + align - 1) & ~(align - 1);
    const end = ptr + bytes;
    if (end > this.memory.buffer.byteLength) {
      this.memory.grow(Math.ceil((end - this.memory.buffer.byteLength) / PAGE));
      this._sync();
    }
    this.top = end;
    return ptr;
  }

  /** Copy bytes into the module's memory and return the pointer. */
  put(bytes) {
    const ptr = this.alloc(bytes.byteLength);
    this.u8.set(bytes, ptr);
    return ptr;
  }

  /**
   * Scratch for one k: the quantized activation, its per-block scales, and the
   * per-block activation sums Q5_0 folds its -16 into. Reused across calls, so
   * a decode loop does not allocate per matvec.
   */
  scratch(k) {
    if (!this._scratchK || this._scratchK < k) {
      this._qa = this.alloc(k);
      this._da = this.alloc((k / 32) * 4);
      this._asum = this.alloc((k / 32) * 4);
      this._x = this.alloc(k * 4);
      this._scratchK = k;
    }
    return this;
  }

  /**
   * out[m] = W[m,k] @ x[k] through the SIMD kernel.
   *
   * @param {Float32Array} out written in place
   * @param {number} wPtr pointer to the packed weights inside this memory
   * @param {Float32Array} x the activation
   * @returns {number} 0, or -1 if this dtype has no wasm kernel — the caller
   *   falls back to notorch.js exactly as for any other miss.
   */
  /**
   * Quantize one activation row, exposed so a test can look at the integers
   * rather than only at what they add up to. Returns the Int8Array view.
   */
  quantAct(x, k) {
    this.scratch(k);
    this.f32.set(x.subarray(0, k), this._x >> 2);
    this.exports.quant_act(this._x, k, this._qa, this._da);
    return this.i8.subarray(this._qa, this._qa + k);
  }

  qmatvecI8(out, wPtr, dtype, x, m, k) {
    this.scratch(k);
    const outPtr = this._outPtr && this._outM >= m ? this._outPtr
      : (this._outPtr = this.alloc(m * 4), this._outM = m, this._outPtr);
    this.f32.set(x.subarray(0, k), this._x >> 2);
    const rc = this.exports.qmatvec_i8(outPtr, wPtr, dtype, this._x, m, k,
                                       this._qa, this._da, this._asum);
    if (rc !== 0) return rc;
    out.set(this.f32.subarray(outPtr >> 2, (outPtr >> 2) + m));
    return 0;
  }
}
