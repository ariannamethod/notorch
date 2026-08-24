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
//
// The module imports its memory rather than owning it, which is what makes it
// usable on a real model. A wasm kernel can only read the one address space it
// was handed, so either the weights are copied in per call — which costs more
// than the kernel saves — or the model lives there from the start. It lives
// there: `create({ modelBytes })` sizes the memory for the file, `modelBase`
// says where to read it, and from then on a tensor's byte offset inside that
// buffer IS the pointer the kernel wants. The memory is shared, so the same
// bytes are also what notorch-workers.mjs binds a pool to.
import { readFileSync, statSync, openSync, readSync, closeSync } from 'node:fs';

const PAGE = 65536;

/** Where the model goes. Below this live the module's stack and globals. */
const RESERVED = 1048576;   // matches --initial-memory in wasm/build.sh

export class WasmKernels {
  /**
   * @param {object|Uint8Array|ArrayBuffer} [opts] the module bytes, or:
   * @param {ArrayBuffer|Uint8Array} [opts.bytes] the module. In Node it is read
   *   from disk when omitted; in a browser, fetch it and pass it in.
   * @param {number} [opts.modelBytes=0] room to reserve at `modelBase` for a
   *   model the caller is about to read in. Zero means kernels only.
   * @param {number} [opts.maximumPages=32768] ceiling for the memory, in 64 KB
   *   pages. 32768 is the 2 GB the module was linked with.
   */
  static async create(opts = {}) {
    if (opts instanceof Uint8Array || opts instanceof ArrayBuffer) opts = { bytes: opts };
    const { modelBytes = 0, maximumPages = 32768 } = opts;
    const self = new WasmKernels();
    let bytes = opts.bytes;
    if (!bytes) {
      const url = new URL('./wasm/qkernels.wasm', import.meta.url);
      bytes = readFileSync(url);
    }
    // Scratch (activation, its int8 form, the output row) is allocated on
    // demand; growing a shared memory keeps the buffer, so views the caller
    // already holds on the weights stay valid.
    const initial = Math.ceil((RESERVED + modelBytes) / PAGE) + 16;
    if (initial > maximumPages) {
      throw new Error(`WasmKernels.create: ${modelBytes} bytes of model needs `
        + `${initial} pages, over the ${maximumPages}-page ceiling`);
    }
    self.memory = new WebAssembly.Memory({ initial, maximum: maximumPages, shared: true });
    const { instance } = await WebAssembly.instantiate(
      bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
      { env: { memory: self.memory } });
    self.exports = instance.exports;
    const heapBase = instance.exports.__heap_base?.value ?? 0;
    if (heapBase > RESERVED) {
      throw new Error(`WasmKernels.create: module wants ${heapBase} bytes below `
        + `the model, more than the ${RESERVED} reserved`);
    }
    self._sync();
    self.modelBase = RESERVED;
    self.top = RESERVED + modelBytes;
    // Calls the kernels took and calls they refused. A fast path that is
    // silently not running looks exactly like a fast path that is slow.
    self.calls = 0;
    self.misses = 0;
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
   * The region a model was sized for, as bytes to read a file into. Its
   * `byteOffset` is `modelBase`, so every tensor view loadGGUF cuts from this
   * buffer already carries the pointer the kernels need.
   */
  modelRegion(byteLength) {
    return new Uint8Array(this.memory.buffer, this.modelBase, byteLength);
  }

  /**
   * Kernels with a GGUF file already inside their memory. Read straight into
   * the region — no intermediate copy of the model, which for a file of any
   * size is the difference between one allocation and three.
   *
   *   const w = await WasmKernels.fromModelFile(path);
   *   const { tensors } = loadGGUF(w.memory.buffer, { base: w.modelBase });
   */
  static async fromModelFile(path, opts = {}) {
    const size = statSync(path).size;
    const self = await WasmKernels.create({ ...opts, modelBytes: size });
    const view = self.modelRegion(size);
    const fd = openSync(path, 'r');
    try {
      for (let got = 0; got < size; ) {
        const n = readSync(fd, view, got, size - got, got);
        if (n <= 0) throw new Error(`fromModelFile: short read at ${got} of ${size}`);
        got += n;
      }
    } finally { closeSync(fd); }
    self.modelBytes = size;
    return self;
  }

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

  /**
   * out[m] = W[m,k] @ x[k] through the SIMD kernel.
   *
   * @param {Float32Array} out written in place
   * @param {number} wPtr byte offset of the packed weights inside this memory —
   *   for a model read into `modelRegion`, that is the tensor's own byteOffset
   * @param {Float32Array} x the activation
   * @returns {number} 0, or -1 if this dtype has no wasm kernel — the caller
   *   falls back to notorch.js exactly as for any other miss.
   */
  qmatvecI8(out, wPtr, dtype, x, m, k) {
    this.scratch(k);
    const outPtr = this._outPtr && this._outM >= m ? this._outPtr
      : (this._outPtr = this.alloc(m * 4), this._outM = m, this._outPtr);
    this.f32.set(x.subarray(0, k), this._x >> 2);
    const rc = this.exports.qmatvec_i8(outPtr, wPtr, dtype, this._x, m, k,
                                       this._qa, this._da, this._asum);
    if (rc !== 0) { this.misses++; return rc; }
    this.calls++;
    out.set(this.f32.subarray(outPtr >> 2, (outPtr >> 2) + m));
    return 0;
  }
}
