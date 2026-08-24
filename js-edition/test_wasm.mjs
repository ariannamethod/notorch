// test_wasm.mjs — the SIMD kernels must agree with the ones they replace.
//
// Two comparisons, because they answer different questions. Against
// notorch.js's own int8 path the wasm kernel should be nearly exact: same
// quantization, same round-half-to-even, same per-block scales, differing only
// in the order sixteen products get summed. Against the exact packed matvec it
// is approximate by construction — activations are quantized — and held to the
// same 2e-2 the C test uses (tests/test_qmatvec.c:227).
//
//   node test_wasm.mjs   → JS_WASM_OK
import { qmatvec, qmatvecI8 } from './notorch.js';
import { WasmKernels } from './notorch-wasm.mjs';

let _s = 0x5EED1234;
const rnd32 = () => { _s ^= _s << 13; _s >>>= 0; _s ^= _s >>> 17; _s ^= _s << 5; _s >>>= 0; return _s; };
const rndUnit = () => (rnd32() / 0xFFFFFFFF) * 2 - 1;

let fails = 0;
const w = await WasmKernels.create();

// Random bytes make a fine quantized weight everywhere except the f16 scales,
// which would land on NaN and infinity; those are written by hand. `vals` is
// how many weights a block holds — 32 for the row formats, 256 for the
// K-quants, whose scales are 6-bit fields that random bytes fill legally.
const F16 = (W, o) => { W[o] = 0x66; W[o + 1] = 0x2A; };
const FORMATS = [
  { name: 'Q4_0', dtype: 2,  vals: 32,  blkBytes: 18,  setScale: F16 },
  { name: 'Q5_0', dtype: 6,  vals: 32,  blkBytes: 22,  setScale: F16 },
  { name: 'Q8_0', dtype: 8,  vals: 32,  blkBytes: 34,  setScale: F16 },
  { name: 'Q4_K', dtype: 12, vals: 256, blkBytes: 144, setScale: (W, o) => { F16(W, o); F16(W, o + 2); } },
  { name: 'Q6_K', dtype: 14, vals: 256, blkBytes: 210, setScale: (W, o) => F16(W, o + 208) },
];

function relErr(ref, got, m) {
  let maxAbs = 0, maxRef = 0;
  for (let i = 0; i < m; i++) {
    if (!Number.isFinite(got[i]) || !Number.isFinite(ref[i])) return Infinity;
    const d = Math.abs(ref[i] - got[i]);
    if (d > maxAbs) maxAbs = d;
    const a = Math.abs(ref[i]);
    if (a > maxRef) maxRef = a;
  }
  return maxRef > 0 ? maxAbs / maxRef : maxAbs;
}

// Row counts that no block size divides evenly, and a k of several blocks.
for (const M of [1, 63, 256]) {
  for (const f of FORMATS) {
    const K = 512, nb = K / f.vals, stride = nb * f.blkBytes;
    const W = new Uint8Array(M * stride);
    for (let i = 0; i < W.length; i++) W[i] = rnd32() & 0xFF;
    for (let r = 0; r < M; r++) for (let b = 0; b < nb; b++) f.setScale(W, r * stride + b * f.blkBytes);
    const x = new Float32Array(K);
    for (let i = 0; i < K; i++) x[i] = rndUnit();

    const exact = new Float32Array(M), js8 = new Float32Array(M), wa = new Float32Array(M);
    qmatvec(exact, W, f.dtype, x, M, K);
    qmatvecI8(js8, W, f.dtype, x, M, K);
    const ptr = w.put(W);
    const rc = w.qmatvecI8(wa, ptr, f.dtype, x, M, K);

    const vsExact = relErr(exact, wa, M), vsJs = relErr(js8, wa, M);
    // Agreement with notorch.js's own int8 path is the real check on the port,
    // and it holds at every m. The 2e-2 against the exact path is a statistic
    // over many rows — C measures it at m=512 — and it does not mean anything
    // at m=1, where rel is one number divided by one number and int8
    // quantization alone lands at 5.8e-2. The JS kernel misses by exactly the
    // same amount there, which is how we know the tolerance is what is wrong.
    const ok = rc === 0 && vsJs < 1e-5 && (M < 16 || vsExact < 2e-2);
    if (!ok) fails++;
    const note = M < 16 ? ' (exact-path tolerance not applied at this m)' : '';
    console.log(`wasm    [${f.name} m=${String(M).padStart(3)} k=${K}] vs exact ${vsExact.toExponential(2)}`
      + `  vs js-i8 ${vsJs.toExponential(2)}  ${ok ? 'PASS' : 'FAIL'}${note}`);
  }
}

// Contract: a dtype with no wasm kernel, and a k that is not whole blocks, are
// refused so the caller can fall back rather than get a wrong answer quietly.
{
  const out = new Float32Array(4), x = new Float32Array(64);
  const ptr = w.put(new Uint8Array(4096));
  for (const [dtype, k, why] of [[12, 96, 'Q4_K needs k a multiple of 256'],
                                 [14, 96, 'Q6_K needs k a multiple of 256'],
                                 [1, 512, 'F16 is not an int8 path'],
                                 [8, 48, 'k is not a whole number of blocks']]) {
    const rc = w.qmatvecI8(out, ptr, dtype, x, 4, k);
    if (rc !== -1) { console.log(`  ${why}: expected -1, got ${rc}  FAIL`); fails++; }
  }
  if (!fails) console.log('wasm    [contract] unsupported dtypes and shapes are refused  PASS');
}

// The wasm quantizer must round ties to even, as lrintf does and Math.round
// does not. Random input never lands on an exact half — measured 0 in 14336 —
// so every check above is blind to it, exactly as the JS gate was until the
// same case was built there. x[0]=127 forces d=1, so each entry IS the value
// being rounded, and the expectations are written from the definition.
{
  const K = 32, x = new Float32Array(K);
  x[0] = 127;
  for (let i = 1; i < K; i++) x[i] = i - 16.5;
  const want = [127,
    -16, -14, -14, -12, -12, -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0,
    0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14];
  const qa = w.quantAct(x, K);
  let bad = 0;
  for (let i = 0; i < K; i++) if (qa[i] !== want[i]) bad++;
  if (bad) { console.log(`wasm    [ties] ${bad}/${K} activations round the wrong way  FAIL`); fails++; }
  else console.log('wasm    [ties] 32/32 activations round to even  PASS');
}

console.log(fails === 0 ? 'JS_WASM_OK' : `${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
