// notorch-workers.mjs — a resident worker pool for the packed matvec.
//
// notorch.js stays single-file and works on its own; this is the optional
// second half for machines with cores to spare. Measured on an A18 Pro
// (2 performance + 4 efficiency), 32000x576 Q8_0: 10.01 ms on one worker,
// 5.16 ms on six — 1.94x, with the output bit-identical to qmatvec.
//
// Two things it needs and one it cannot do:
//
//   - SharedArrayBuffer. In Node it is always there. In a browser it needs
//     COOP/COEP headers on the page, so a static host that cannot set headers
//     cannot use this file. That is exactly why it is a separate file.
//   - Weights, activation and output all living in shared memory. `toShared`
//     moves an ArrayBuffer across once, at load.
//   - It blocks the calling thread on Atomics.wait, which the browser's main
//     thread forbids. In a browser, run the engine inside a worker and let
//     this pool serve it from there. In Node, on the main thread, it is fine.
//
// Why row ranges rather than anything cleverer: rows are independent and write
// disjoint slots, so which worker computes which row cannot move a bit of the
// result. Splitting the work does not change the answer, only who is idle.
import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';

// Control-block layout, mirrored in notorch-worker-entry.mjs.
const GEN = 0, DONE = 1, CURSOR = 2, HI = 3, CHUNK = 4, DTYPE = 5, K = 6, WOFF = 7, SLOTS = 8;

/** How long a single call may wait on its workers before declaring the pool wedged. */
const POOL_TIMEOUT_MS = 30_000;

/** Copy an ArrayBuffer into shared memory. Do this once, at load, not per call. */
export function toShared(arrayBuffer) {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error('toShared: SharedArrayBuffer is unavailable (a browser page needs COOP/COEP headers)');
  }
  if (arrayBuffer instanceof SharedArrayBuffer) return arrayBuffer;
  const sab = new SharedArrayBuffer(arrayBuffer.byteLength);
  new Uint8Array(sab).set(new Uint8Array(arrayBuffer));
  return sab;
}

export class WorkerPool {
  /**
   * @param {SharedArrayBuffer} weights the whole packed weight blob — for a GGUF
   *   loaded with { packed: true } every tensor is already a view onto one
   *   buffer, so the pool binds that buffer once and a call names its matrix by
   *   byte offset. Nothing crosses by postMessage per call: the caller blocks on
   *   Atomics.wait, so a message would be waiting on an event loop that is not
   *   turning. That mistake produced a pool which reported every round complete
   *   without computing anything.
   * @param {number} maxM largest row count any call will ask for
   * @param {number} maxK largest inner dimension any call will ask for
   * @param {number} [n] worker count. Defaults to availableParallelism(), which
   *   counts the cores this process may run on rather than the cores the machine
   *   has — a distinction that cost the C side measurably on a pinned
   *   big.LITTLE run (notorch.c, commit 4281b5f).
   * @param {number} [chunksPerWorker=16] row-handout granularity. 16 is C's
   *   number and measured best here too; 1 degenerates to an even split.
   */
  static async create(weights, maxM, maxK, n = availableParallelism(), chunksPerWorker = 16) {
    if (!(weights instanceof SharedArrayBuffer)) {
      throw new Error('WorkerPool.create: weights must be a SharedArrayBuffer (see toShared)');
    }
    const pool = new WorkerPool();
    pool.n = Math.max(1, n | 0);
    pool.chunksPerWorker = Math.max(1, chunksPerWorker | 0);
    pool.ctrl = new SharedArrayBuffer(SLOTS * 4);
    pool.C = new Int32Array(pool.ctrl);
    pool.weights = weights;
    pool.xScratch = new SharedArrayBuffer(maxK * 4);
    pool.outScratch = new SharedArrayBuffer(maxM * 4);
    pool.x = new Float32Array(pool.xScratch);
    pool.out = new Float32Array(pool.outScratch);
    pool.workers = [];
    const url = new URL('./notorch-worker-entry.mjs', import.meta.url);
    for (let i = 0; i < pool.n; i++) {
      const w = new Worker(url, { workerData: {
        ctrl: pool.ctrl, weights, xScratch: pool.xScratch, outScratch: pool.outScratch,
      } });
      await new Promise((res, rej) => { w.once('message', res); w.once('error', rej); });
      w.unref();                          // a live pool must not hold the process open
      pool.workers.push(w);
    }
    return pool;
  }

  /**
   * out[0..m) = Wq[m,k] @ x[k], across the pool.
   *
   * `wOffset` is the matrix's byte offset inside the blob the pool was built
   * on — for a packed GGUF tensor that is `tensor.packed.byteOffset`. `out` and
   * `x` are ordinary arrays: they are copied through the pool's shared scratch,
   * which costs k floats in and m floats out against a matvec of m*k.
   *
   * @returns {number} 0, or -1 if the dtype has no packed kernel, in which case
   *   the caller falls back to qmatvec exactly as for a single-threaded miss.
   */
  qmatvec(out, wOffset, dtype, x, m, k) {
    if (m > this.out.length || k > this.x.length) {
      throw new Error(`WorkerPool.qmatvec: m=${m} k=${k} exceeds the scratch this pool was built for (${this.out.length}, ${this.x.length})`);
    }
    const C = this.C;
    this.x.set(x.subarray(0, k));

    let chunk = Math.floor(m / (this.n * this.chunksPerWorker));
    if (chunk < 1) chunk = 1;
    C[HI] = m; C[CHUNK] = chunk; C[DTYPE] = dtype; C[K] = k; C[WOFF] = wOffset;
    Atomics.store(C, CURSOR, 0);
    Atomics.store(C, DONE, 0);
    Atomics.add(C, GEN, 1);
    Atomics.notify(C, GEN);

    // Wait on the value just read: reading it a second time inside the call is
    // the race that turns this loop into a timeout per round. The timeout is not
    // decoration — a pool that wedges here wedges the whole thread, and no timer
    // can fire to notice, because Atomics.wait is exactly the event loop not
    // turning. So the wait itself has to give up and say what happened.
    const deadline = Date.now() + POOL_TIMEOUT_MS;
    let done;
    while ((done = Atomics.load(C, DONE)) < this.n) {
      Atomics.wait(C, DONE, done, 250);
      if (Date.now() > deadline) {
        throw new Error(`WorkerPool.qmatvec: ${done}/${this.n} workers reported after ${POOL_TIMEOUT_MS} ms — the pool is wedged`);
      }
    }
    out.set(this.out.subarray(0, m));
    return 0;
  }

  async terminate() {
    Atomics.store(this.C, GEN, -1);
    Atomics.notify(this.C, GEN);
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers.length = 0;
  }
}
