// test_wasm_e2e.mjs — the wasm kernels must run a real model, not three shapes.
//
// test_wasm.mjs proves the arithmetic on synthetic blocks. This proves the
// wiring: a GGUF read into the module's own memory, every packed tensor
// multiplied at the shape the model actually uses, and a real forward compared
// against the same forward with the kernels switched off.
//
// Two questions a shape test cannot answer:
//   - is the fast path running at all? A pointer that misses, a dtype that
//     falls through, a buffer identity that never matches — each looks exactly
//     like "wasm is attached" and none of them computes anything. The kernels
//     count what they take and what they refuse, and this asserts on the count.
//   - does the model still mean the same thing? A base or offset bug reads
//     neighbouring weights and still produces logits, just not these logits.
//
// What this does NOT assert is that generation is unchanged. The wasm path
// quantizes the activation to int8, the same approximation C's nt_qmatvec_i8
// makes, and greedy decoding is a chain of argmaxes over numbers that moved:
// on nano_arianna Q8_0 the two paths agree for eleven tokens and split at the
// twelfth. Logit agreement is the property; token identity is a coincidence
// with a shelf life, so it is measured and printed, never gated on.
//
//   node test_wasm_e2e.mjs model.gguf   → JS_WASM_E2E_OK
//
// Without a file it says so and exits 0: `make test_js` has no model to hand.
import { Notorch, KVCache, qmatvec, qmatvecI8 } from './notorch.js';

const MODEL = process.argv[2] || process.env.NT_TEST_MODEL;
if (!MODEL) {
  console.log('wasm-e2e  (no model given — pass a .gguf to run this gate)');
  console.log('JS_WASM_E2E_SKIPPED');
  process.exit(0);
}

const DTYPE = { 2: 'Q4_0', 6: 'Q5_0', 8: 'Q8_0', 12: 'Q4_K', 14: 'Q6_K' };
/** vs the JS i8 kernel this ports: same quantization, only the sum order differs. */
const PORT_TOL = 1e-5;
/** vs the exact path: int8 activations, the tolerance C uses at these row counts. */
const APPROX_TOL = 2e-2;

let fails = 0;
const bad = (msg) => { console.log(`wasm-e2e  ${msg}  FAIL`); fails++; };

let _s = 0x13579BDF;
const rndUnit = () => {
  _s ^= _s << 13; _s >>>= 0; _s ^= _s >>> 17; _s ^= _s << 5; _s >>>= 0;
  return (_s / 0xFFFFFFFF) * 2 - 1;
};

function relErr(ref, got, n) {
  let maxAbs = 0, maxRef = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(got[i]) || !Number.isFinite(ref[i])) return Infinity;
    const d = Math.abs(ref[i] - got[i]);
    if (d > maxAbs) maxAbs = d;
    const a = Math.abs(ref[i]);
    if (a > maxRef) maxRef = a;
  }
  return maxRef > 0 ? maxAbs / maxRef : maxAbs;
}
const argmax = (v) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; };

// One load, both paths. A packed tensor inside wasm memory is an ordinary
// Uint8Array view, so the exact JS kernels read the very same bytes — the
// comparison is like for like, on one copy of the weights.
process.env.NT_WASM = '1';
const { loadModel, forwardLastLogits } = await import('./infer_gguf.mjs');
const m = await loadModel(MODEL);
if (!m.wasm) { bad('[load] NT_WASM=1 produced no kernels'); process.exit(1); }

// ── A. every packed tensor, at the shape the model uses ──────────────────────
const seen = new Map();          // dtype -> [taken, refused]
let worstVsI8 = 0, worstVsExact = 0, worstShape = '';
for (const [name, t] of m.tensors) {
  if (!t.packed || t.shape.length < 2) continue;
  const rows = t.shape[0], k = t.shape[1];
  const x = new Float32Array(k);
  for (let i = 0; i < k; i++) x[i] = rndUnit();

  const yWasm = new Float32Array(rows);
  const rc = m.wasm.qmatvecI8(yWasm, t.packed.byteOffset, t.dtype, x, rows, k);
  const tally = seen.get(t.dtype) || [0, 0];
  tally[rc === 0 ? 0 : 1]++;
  seen.set(t.dtype, tally);
  if (rc !== 0) continue;        // no kernel for this dtype: the JS path has it

  const yI8 = new Float32Array(rows);
  if (qmatvecI8(yI8, t.packed, t.dtype, x, rows, k) !== 0) {
    bad(`[${name}] wasm computed dtype ${t.dtype} that the JS i8 path refuses`);
    continue;
  }
  const yExact = new Float32Array(rows);
  if (qmatvec(yExact, t.packed, t.dtype, x, rows, k) !== 0) {
    bad(`[${name}] exact path refuses dtype ${t.dtype} at k=${k}`);
    continue;
  }
  const eI8 = relErr(yI8, yWasm, rows);
  const eEx = relErr(yExact, yWasm, rows);
  if (!(eI8 <= PORT_TOL)) {
    bad(`[${name} ${DTYPE[t.dtype]} ${rows}x${k}] vs js-i8 ${eI8.toExponential(2)} over ${PORT_TOL}`);
  }
  if (eI8 > worstVsI8) worstVsI8 = eI8;
  if (eEx > worstVsExact) { worstVsExact = eEx; worstShape = `${DTYPE[t.dtype]} ${rows}x${k}`; }
}

let taken = 0;
for (const [d, [t0, t1]] of [...seen].sort((a, b) => a[0] - b[0])) {
  taken += t0;
  console.log(`wasm-e2e  [${(DTYPE[d] || d).padEnd(4)}] ${t0} tensors taken, ${t1} refused to the JS path`);
}
if (taken === 0) bad('[coverage] no tensor in this model reached a wasm kernel');
console.log(`wasm-e2e  [agreement] worst vs js-i8 ${worstVsI8.toExponential(2)}, `
  + `worst vs exact ${worstVsExact.toExponential(2)} at ${worstShape}  `
  + (worstVsI8 <= PORT_TOL && worstVsExact <= APPROX_TOL ? 'PASS' : 'FAIL'));
if (worstVsExact > APPROX_TOL) bad(`[agreement] vs exact ${worstVsExact.toExponential(2)} over ${APPROX_TOL}`);

// ── B. a real forward, kernels on and off ────────────────────────────────────
const PROMPT = 'The capital of France is';
const ids = m.tok.encode(PROMPT);

function runForward(useWasm, steps) {
  const nt = new Notorch();
  if (useWasm) nt.wasm = m.wasm;
  const caches = Array.from({ length: m.L }, () => new KVCache(ids.length + steps + 1, m.KV * m.HD));
  const seq = ids.slice();
  let logits = forwardLastLogits(nt, m, seq, caches, 0);
  const out = [];
  for (let s = 0; s < steps; s++) {
    const next = argmax(logits);
    out.push(next);
    logits = forwardLastLogits(nt, m, [next], caches, ids.length + s);
  }
  return { logits, out, calls: useWasm ? nt.wasm.calls : 0 };
}

const before = m.wasm.calls;
const a = runForward(true, 4);
const b = runForward(false, 4);
if (m.wasm.calls === before) bad('[forward] the wasm path was attached and never called');
const eLogits = relErr(b.logits, a.logits, b.logits.length);
if (eLogits <= APPROX_TOL) {
  console.log(`wasm-e2e  [forward] ${m.wasm.calls - before} matvecs through wasm, `
    + `logits within ${eLogits.toExponential(2)} of the exact path  PASS`);
} else {
  bad(`[forward] logits diverge by ${eLogits.toExponential(2)}, over ${APPROX_TOL}`);
}
const agree = a.out.findIndex((t, i) => t !== b.out[i]);
console.log(`wasm-e2e  [generation] greedy tokens agree for ${agree < 0 ? a.out.length : agree} of ${a.out.length}`
  + ` (reported, not gated — int8 activations move argmaxes)`);

console.log(fails ? `JS_WASM_E2E_FAIL (${fails})` : 'JS_WASM_E2E_OK');
process.exit(fails ? 1 : 0);
