// notorch-worker-entry.mjs — one worker of the matvec pool.
//
// Everything it needs arrives once, in workerData: the shared weight blob, the
// shared activation and output scratch, and the control block. Nothing crosses
// by postMessage after that, because the calling thread blocks on Atomics.wait
// and a message would need an event loop that is not turning.
import { workerData, parentPort } from 'node:worker_threads';
import { qmatvecRows } from './notorch.js';

const { ctrl, weights, xScratch, outScratch } = workerData;
const C = new Int32Array(ctrl);
const W = new Uint8Array(weights);
const X = new Float32Array(xScratch);
const OUT = new Float32Array(outScratch);

// Control slots, mirrored in notorch-workers.mjs.
const GEN = 0, DONE = 1, CURSOR = 2, HI = 3, CHUNK = 4, DTYPE = 5, K = 6, WOFF = 7;

parentPort.postMessage({ ready: true });

let seen = 0;
for (;;) {
  Atomics.wait(C, GEN, seen);           // returns at once if GEN already moved
  const gen = Atomics.load(C, GEN);
  if (gen < 0) break;                   // pool shutting down
  seen = gen;

  const hi = C[HI], chunk = C[CHUNK], dtype = C[DTYPE], k = C[K], woff = C[WOFF];
  const Wm = W.subarray(woff);          // this call's matrix inside the blob
  // Rows are claimed, not assigned. An even split leaves fast cores waiting on
  // slow ones — on 2P+4E that measured 1.72x against 1.94x for the cursor, the
  // same reason C hands out chunks (notorch.c, commit 8c7026e).
  for (;;) {
    const r0 = Atomics.add(C, CURSOR, chunk);
    if (r0 >= hi) break;
    const r1 = r0 + chunk > hi ? hi : r0 + chunk;
    qmatvecRows(OUT, Wm, dtype, X, r0, r1, k);
  }
  Atomics.add(C, DONE, 1);
  Atomics.notify(C, DONE);
}
