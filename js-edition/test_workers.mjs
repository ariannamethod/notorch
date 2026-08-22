// test_workers.mjs — the pool must compute what qmatvec computes, exactly, and
// must not hang.
//
// Splitting rows across workers cannot change a single bit: rows are
// independent and write disjoint slots. So the gate is equality, not a
// tolerance. The second gate is liveness — every deadlock this pool could have
// shows up as a test that never returns, so the run is under a hard timeout and
// a hang is a failure rather than a hung CI job.
//
//   node test_workers.mjs   → JS_WORKERS_OK
import { qmatvec } from './notorch.js';
import { WorkerPool, toShared } from './notorch-workers.mjs';

// No setTimeout guard here on purpose: a wedged pool blocks this thread inside
// Atomics.wait, the event loop stops turning, and a timer never fires. The
// liveness guarantee has to live in the pool itself, which gives up on its own
// wait and throws — so this test simply lets the exception through.

let _s = 0x13579BDF;
const rnd32 = () => { _s ^= _s << 13; _s >>>= 0; _s ^= _s >>> 17; _s ^= _s << 5; _s >>>= 0; return _s; };
const rndUnit = () => (rnd32() / 0xFFFFFFFF) * 2 - 1;

let fails = 0;

// Q8_0, and a row count that no worker count divides evenly.
const M = 501, K = 256, NB = K / 32, STRIDE = NB * 34;
const wPlain = new ArrayBuffer(M * STRIDE);
{
  const w = new Uint8Array(wPlain);
  for (let i = 0; i < w.length; i++) w[i] = rnd32() & 0xFF;
  for (let r = 0; r < M; r++) for (let b = 0; b < NB; b++) { const o = r * STRIDE + b * 34; w[o] = 0x66; w[o + 1] = 0x2A; }
}
const Wsab = toShared(wPlain);
const W = new Uint8Array(Wsab);
const x = new Float32Array(K);
for (let i = 0; i < K; i++) x[i] = rndUnit();

const ref = new Float32Array(M);
qmatvec(ref, W, 8, x, M, K);

for (const nw of [1, 2, 4, 6]) {
  for (const cpw of [1, 16]) {
    const pool = await WorkerPool.create(Wsab, M, K, nw, cpw);
    const out = new Float32Array(M);
    const rc = pool.qmatvec(out, 0, 8, x, M, K);
    let bad = 0;
    for (let i = 0; i < M; i++) if (out[i] !== ref[i]) bad++;

    // Liveness: a pool that works once and wedges on the second round is the
    // failure mode worth catching, so hammer it before believing the first.
    let stillExact = true;
    for (let rep = 0; rep < 100; rep++) {
      out.fill(0);
      pool.qmatvec(out, 0, 8, x, M, K);
      if (rep === 99) for (let i = 0; i < M; i++) if (out[i] !== ref[i]) stillExact = false;
    }
    await pool.terminate();

    const ok = rc === 0 && bad === 0 && stillExact;
    if (!ok) fails++;
    console.log(`pool    [n=${nw} chunks/worker=${cpw}] ${bad ? `${bad}/${M} differ` : 'bit-identical'}`
      + `, 100 rounds ${stillExact ? 'still exact' : 'DRIFTED'}  ${ok ? 'PASS' : 'FAIL'}`);
  }
}

// Contract: non-shared weights are refused at construction, and a call that
// outgrows the scratch says so instead of writing past it.
{
  let threw = '';
  try { await WorkerPool.create(new ArrayBuffer(64), M, K, 2); }
  catch (err) { threw = String(err.message); }
  if (!/SharedArrayBuffer/.test(threw)) {
    console.log(`plain weights buffer: expected a refusal, got "${threw || 'no throw'}"  FAIL`); fails++;
  }
  const pool = await WorkerPool.create(Wsab, M, K, 2);
  let threw2 = '';
  try { pool.qmatvec(new Float32Array(M + 1), 0, 8, x, M + 1, K); }
  catch (err) { threw2 = String(err.message); }
  await pool.terminate();
  if (!/exceeds the scratch/.test(threw2)) {
    console.log(`oversized call: expected a refusal, got "${threw2 || 'no throw'}"  FAIL`); fails++;
  }
  if (/SharedArrayBuffer/.test(threw) && /exceeds the scratch/.test(threw2)) {
    console.log('pool    [contract] plain weights and oversized calls are refused  PASS');
  }
}

// A second matrix in the same blob, reached by byte offset — the shape a real
// model uses, where every tensor is a view into one file.
{
  const off = 64 * STRIDE;                     // rows 64.. treated as a matrix of its own
  const m2 = 32;
  const pool = await WorkerPool.create(Wsab, M, K, 4);
  const out = new Float32Array(m2), want = new Float32Array(m2);
  pool.qmatvec(out, off, 8, x, m2, K);
  qmatvec(want, W.subarray(off), 8, x, m2, K);
  await pool.terminate();
  let bad = 0;
  for (let i = 0; i < m2; i++) if (out[i] !== want[i]) bad++;
  if (bad) { console.log(`offset matrix: ${bad}/${m2} differ  FAIL`); fails++; }
  else console.log('pool    [offset] a matrix at a byte offset matches qmatvec  PASS');
}

// A pool that wedges must report it rather than hang the caller forever.
{
  const pool = await WorkerPool.create(Wsab, M, K, 2);
  await Promise.all(pool.workers.map((w) => w.terminate()));   // kill the hands, keep the pool
  let threw = '';
  try { pool.qmatvec(new Float32Array(M), 0, 8, x, M, K); }
  catch (err) { threw = String(err.message); }
  if (!/wedged/.test(threw)) {
    console.log(`pool with dead workers: expected a wedge report, got "${threw || 'no throw'}"  FAIL`);
    fails++;
  } else {
    console.log('pool    [liveness] a pool with no live workers reports instead of hanging  PASS');
  }
}

console.log(fails === 0 ? 'JS_WORKERS_OK' : `${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
