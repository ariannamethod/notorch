// test_qmatvec.mjs — packed matvec vs an independent dequant oracle.
//
// Mirrors tests/test_qmatvec.c: for every GGUF dtype the packed kernel must
// compute the same matvec as dequantizing the block and running a dense dot.
// The oracle here is deliberately NOT notorch.js's own ggufDequant* — those
// share a reading of the block layout with the kernel under test, and an
// oracle that shares the bug proves nothing. These reference unpackers are
// ported from the C test's own independent set (tests/test_qmatvec.c:42-113),
// down to the f16 reconstruction, which goes through raw bits rather than the
// Math.pow path notorch.js uses.
//
// Compared by RELATIVE error (max|ref-got| / max|ref|) at the C threshold of
// 1e-3 (tests/test_qmatvec.c:148): a real unpack/scale bug lands at O(1),
// float summation-order noise at ~1e-7.
//
//   node test_qmatvec.mjs                       # JS oracle only
//   node test_qmatvec.mjs --cref /path/to/bin   # + second hand: the C kernel
//                                               #   on the identical bytes
import { writeFileSync, readFileSync, mkdtempSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { qmatvec, qmatvecI8, qmatvecI8Rows, quantAct, dequantRow, Notorch, Tensor } from './notorch.js';

// ── deterministic input, so a failure is reproducible and C sees these bytes ──
let _s = 0x9E3779B9;
function rnd32() { _s ^= _s << 13; _s >>>= 0; _s ^= _s >>> 17; _s ^= _s << 5; _s >>>= 0; return _s; }
function rndByte() { return rnd32() & 0xFF; }
function rndUnit() { return (rnd32() / 0xFFFFFFFF) * 2 - 1; }

// ── independent f16 → f32, by bits (mirrors ref_f16, tests/test_qmatvec.c:25) ──
const _u32 = new Uint32Array(1), _f32 = new Float32Array(_u32.buffer);
function refF16(h) {
  const s = (h >> 15) & 1;
  let e = (h >> 10) & 0x1F, m = h & 0x3FF, b;
  if (e === 0) {
    if (m === 0) b = s << 31;
    else {
      e = 127 - 15 + 1;
      while (!(m & 0x400)) { m <<= 1; e--; }
      m &= 0x3FF;
      b = (s << 31) | (e << 23) | (m << 13);
    }
  } else if (e === 0x1F) {
    b = (s << 31) | (0xFF << 23) | (m << 13);
  } else {
    b = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
  }
  _u32[0] = b >>> 0;
  return _f32[0];
}

// ── independent reference dequants (mirror the C test's, not notorch.js's) ────
function refQ4_0(W, off, out, n) {
  for (let bk = 0; bk < n / 32; bk++) {
    const b = off + bk * 18, sc = refF16(W[b] | (W[b + 1] << 8));
    for (let i = 0; i < 16; i++) {
      out[bk * 32 + i]      = ((W[b + 2 + i] & 0x0F) - 8) * sc;
      out[bk * 32 + i + 16] = ((W[b + 2 + i] >> 4)   - 8) * sc;
    }
  }
}
function refQ8_0(W, off, out, n) {
  for (let bk = 0; bk < n / 32; bk++) {
    const b = off + bk * 34, sc = refF16(W[b] | (W[b + 1] << 8));
    for (let i = 0; i < 32; i++) out[bk * 32 + i] = ((W[b + 2 + i] << 24) >> 24) * sc;
  }
}
function refQ5_0(W, off, out, n) {
  for (let bk = 0; bk < n / 32; bk++) {
    const b = off + bk * 22, sc = refF16(W[b] | (W[b + 1] << 8));
    const qh = (W[b + 2] | (W[b + 3] << 8) | (W[b + 4] << 16) | (W[b + 5] << 24)) >>> 0;
    for (let j = 0; j < 16; j++) {
      const lo = W[b + 6 + j] & 0x0F, hi = W[b + 6 + j] >> 4;
      const h0 = (qh >>> j) & 1, h1 = (qh >>> (j + 16)) & 1;
      out[bk * 32 + j]      = ((lo | (h0 << 4)) - 16) * sc;
      out[bk * 32 + j + 16] = ((hi | (h1 << 4)) - 16) * sc;
    }
  }
}
function refScaleMinK4(j, W, sc) {
  if (j < 4) return [W[sc + j] & 63, W[sc + j + 4] & 63];
  return [(W[sc + j + 4] & 0x0F) | ((W[sc + j - 4] >> 6) << 4),
          (W[sc + j + 4] >> 4)   | ((W[sc + j]     >> 6) << 4)];
}
function refQ4_K(W, off, out, n) {
  for (let i = 0; i < n / 256; i++) {
    const b = off + i * 144;
    const d = refF16(W[b] | (W[b + 1] << 8)), dmin = refF16(W[b + 2] | (W[b + 3] << 8));
    const sc = b + 4, qs = b + 16;
    let is = 0, qi = 0;
    for (let j = 0; j < 256; j += 64) {
      const [sc0, m0] = refScaleMinK4(is, W, sc), [sc1, m1] = refScaleMinK4(is + 1, W, sc);
      const d1 = d * sc0, mm1 = dmin * m0, d2 = d * sc1, mm2 = dmin * m1;
      for (let l = 0; l < 32; l++) out[i * 256 + j + l]      = d1 * (W[qs + qi + l] & 0x0F) - mm1;
      for (let l = 0; l < 32; l++) out[i * 256 + j + 32 + l] = d2 * (W[qs + qi + l] >> 4)   - mm2;
      qi += 32; is += 2;
    }
  }
}
function refQ6_K(W, off, out, n) {
  for (let i = 0; i < n / 256; i++) {
    const b = off + i * 210, ql = b, qh = b + 128, sc = b + 192;
    const d = refF16(W[b + 208] | (W[b + 209] << 8));
    for (let nn = 0; nn < 256; nn += 128) {
      const h = (nn / 128) | 0;
      const qlh = ql + h * 64, qhh = qh + h * 32, sch = sc + h * 8;
      for (let l = 0; l < 32; l++) {
        const is = (l / 16) | 0;
        const lo = W[qlh + l], lo32 = W[qlh + l + 32], hb = W[qhh + l];
        const q1 = ((lo   & 0x0F) | (((hb >> 0) & 3) << 4)) - 32;
        const q2 = ((lo32 & 0x0F) | (((hb >> 2) & 3) << 4)) - 32;
        const q3 = ((lo   >> 4)   | (((hb >> 4) & 3) << 4)) - 32;
        const q4 = ((lo32 >> 4)   | (((hb >> 6) & 3) << 4)) - 32;
        out[i * 256 + nn + l]      = d * ((W[sch + is]     << 24) >> 24) * q1;
        out[i * 256 + nn + l + 32] = d * ((W[sch + is + 2] << 24) >> 24) * q2;
        out[i * 256 + nn + l + 64] = d * ((W[sch + is + 4] << 24) >> 24) * q3;
        out[i * 256 + nn + l + 96] = d * ((W[sch + is + 6] << 24) >> 24) * q4;
      }
    }
  }
}

// Sane normal f16 scales (0x2A66) at the per-format offsets; other bytes stay
// random (set_s32 / set_q4k / set_q6k, tests/test_qmatvec.c:119-121).
const setS32 = (W, o) => { W[o] = 0x66; W[o + 1] = 0x2A; };
const setQ4K = (W, o) => { W[o] = 0x66; W[o + 1] = 0x2A; W[o + 2] = 0x66; W[o + 3] = 0x26; };
const setQ6K = (W, o) => { W[o + 208] = 0x66; W[o + 209] = 0x2A; };

const M = 512, K = 2048;
const FORMATS = [
  { name: 'Q4_0', dtype: 2,  blkBytes: 18,  blkVals: 32,  ref: refQ4_0, setScale: setS32 },
  { name: 'Q5_0', dtype: 6,  blkBytes: 22,  blkVals: 32,  ref: refQ5_0, setScale: setS32 },
  { name: 'Q8_0', dtype: 8,  blkBytes: 34,  blkVals: 32,  ref: refQ8_0, setScale: setS32 },
  { name: 'Q4_K', dtype: 12, blkBytes: 144, blkVals: 256, ref: refQ4_K, setScale: setQ4K },
  { name: 'Q6_K', dtype: 14, blkBytes: 210, blkVals: 256, ref: refQ6_K, setScale: setQ6K },
];

const cref = (() => {
  const i = process.argv.indexOf('--cref');
  return i >= 0 ? process.argv[i + 1] : null;
})();
const tmp = cref ? mkdtempSync(join(tmpdir(), 'ntjs-qmv-')) : null;

/** Run the C kernel on the identical bytes and return its out[], or null. */
function cSecondHand(tag, W, dtype, x, m, k, i8 = false) {
  if (!cref) return null;
  const inPath = join(tmp, `${tag}.in`), outPath = join(tmp, `${tag}.out`);
  const head = Buffer.alloc(16);
  head.writeInt32LE(dtype, 0); head.writeInt32LE(m, 4); head.writeInt32LE(k, 8);
  head.writeInt32LE(W.length, 12);
  writeFileSync(inPath, Buffer.concat([head, Buffer.from(W.buffer, W.byteOffset, W.byteLength),
                                       Buffer.from(x.buffer, x.byteOffset, x.byteLength)]));
  execFileSync(cref, i8 ? [inPath, outPath, '--i8'] : [inPath, outPath]);
  const raw = readFileSync(outPath);
  return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}

function relErr(ref, got, m) {
  let maxAbs = 0, maxRef = 0;
  for (let i = 0; i < m; i++) {
    // A NaN fails every comparison below and would leave maxAbs at 0 — which is
    // exactly how a byte-swapped f16 kernel passed this test on its first run.
    if (!Number.isFinite(got[i]) || !Number.isFinite(ref[i])) return Infinity;
    const d = Math.abs(ref[i] - got[i]);
    if (d > maxAbs) maxAbs = d;
    const a = Math.abs(ref[i]);
    if (a > maxRef) maxRef = a;
  }
  return maxRef > 0 ? maxAbs / maxRef : maxAbs;
}

let fails = 0;

function report(name, rel, relC, tol = 1e-3) {
  const ok = rel < tol && (relC === null || relC < tol);
  const cTxt = relC === null ? '' : `  C ${relC.toExponential(2)}`;
  console.log(`${name.padEnd(7)} [m=${M} k=${K}] rel ${rel.toExponential(2)}${cTxt}  ${ok ? 'PASS' : 'FAIL'}`);
  if (!ok) fails++;
}

for (const f of FORMATS) {
  const nb = K / f.blkVals, stride = nb * f.blkBytes;
  const W = new Uint8Array(M * stride);
  for (let i = 0; i < W.length; i++) W[i] = rndByte();
  for (let row = 0; row < M; row++)
    for (let b = 0; b < nb; b++) f.setScale(W, row * stride + b * f.blkBytes);
  const x = new Float32Array(K);
  for (let i = 0; i < K; i++) x[i] = rndUnit();

  const rowVals = new Float32Array(K), ref = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    f.ref(W, row * stride, rowVals, K);
    let acc = 0;
    for (let j = 0; j < K; j++) acc += rowVals[j] * x[j];
    ref[row] = acc;
  }

  const got = new Float32Array(M);
  const rc = qmatvec(got, W, f.dtype, x, M, K);
  if (rc !== 0) { console.log(`${f.name.padEnd(5)} qmatvec rc=${rc}  FAIL`); fails++; continue; }

  const c = cSecondHand(f.name, W, f.dtype, x, M, K);
  report(f.name, relErr(ref, got, M), c ? relErr(c, got, M) : null);

  // dequantRow decodes ONE row where the dense path decodes the tensor. Checked
  // against the same independent unpacker the matvec above is checked against,
  // on rows chosen at both ends and in the middle, and required to be exact —
  // it is a copy of the block decode, not an accumulation, so anything but
  // equality is a bug (gguf_dequant_row, gguf.c).
  {
    const stride1 = (K / f.blkVals) * f.blkBytes;
    const rowVals2 = new Float32Array(K), rowGot = new Float32Array(K);
    let rowBad = 0;
    for (const rw of [0, 1, 200, M - 1]) {
      if (dequantRow(rowGot, W, f.dtype, rw, K) !== 0) { rowBad++; continue; }
      f.ref(W, rw * stride1, rowVals2, K);
      for (let i = 0; i < K; i++) if (rowGot[i] !== rowVals2[i]) { rowBad++; break; }
    }
    if (rowBad) { console.log(`row${f.name} dequantRow mismatch on ${rowBad} row(s)  FAIL`); fails++; }
  }

  // int8 dynamic-activation path against the exact packed path just measured.
  // APPROXIMATE by construction (notorch.h:551) — tolerance, not bit-close,
  // at the C threshold (run_i8, tests/test_qmatvec.c:227). The second hand is
  // C's OWN i8 kernel, not its f32 one: two approximations of the same shape.
  const gotI8 = new Float32Array(M);
  const rcI8 = qmatvecI8(gotI8, W, f.dtype, x, M, K);
  if (rcI8 !== 0) { console.log(`i8${f.name} qmatvecI8 rc=${rcI8}  FAIL`); fails++; continue; }
  const cI8 = cSecondHand(`i8${f.name}`, W, f.dtype, x, M, K, true);
  report(`i8${f.name}`, relErr(got, gotI8, M), cI8 ? relErr(cI8, gotI8, M) : null, 2e-2);

  // qmatvecI8Rows over two disjoint ranges must reproduce the single call
  // exactly — the row split is arithmetic-free, so this is equality, not
  // tolerance. A shared accumulator or a row-base off by a block shows here.
  const qa = new Int8Array(K), da = new Float32Array(K / 32);
  quantAct(x, K, qa, da);
  const split = new Float32Array(M);
  const rcA = qmatvecI8Rows(split, W, f.dtype, qa, da, 0, 173, K);
  const rcB = qmatvecI8Rows(split, W, f.dtype, qa, da, 173, M, K);
  let identical = rcA === 0 && rcB === 0;
  for (let i = 0; identical && i < M; i++) if (split[i] !== gotI8[i]) identical = false;
  if (!identical) { console.log(`i8${f.name} rows split != whole call  FAIL`); fails++; }
}

// F16 / F32 have no block structure — sane weights, dedicated runners
// (run_f16 / run_f32, tests/test_qmatvec.c:157,179).
{
  const Wh = new Uint16Array(M * K);
  for (let i = 0; i < Wh.length; i++) Wh[i] = ((rnd32() & 0x8000) | 0x2000 | (rnd32() & 0x03FF));
  const W = new Uint8Array(Wh.buffer);
  const x = new Float32Array(K);
  for (let i = 0; i < K; i++) x[i] = rndUnit();
  const ref = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let acc = 0;
    for (let j = 0; j < K; j++) acc += refF16(Wh[row * K + j]) * x[j];
    ref[row] = acc;
  }
  const got = new Float32Array(M);
  const rc = qmatvec(got, W, 1, x, M, K);
  if (rc !== 0) { console.log(`F16   qmatvec rc=${rc}  FAIL`); fails++; }
  else {
    const c = cSecondHand('F16', W, 1, x, M, K);
    report('F16', relErr(ref, got, M), c ? relErr(c, got, M) : null);
  }
}
{
  const Wf = new Float32Array(M * K);
  for (let i = 0; i < Wf.length; i++) Wf[i] = rndUnit();
  const W = new Uint8Array(Wf.buffer);
  const x = new Float32Array(K);
  for (let i = 0; i < K; i++) x[i] = rndUnit();
  const ref = new Float32Array(M);
  for (let row = 0; row < M; row++) {
    let acc = 0;
    for (let j = 0; j < K; j++) acc += Wf[row * K + j] * x[j];
    ref[row] = acc;
  }
  const got = new Float32Array(M);
  const rc = qmatvec(got, W, 0, x, M, K);
  if (rc !== 0) { console.log(`F32   qmatvec rc=${rc}  FAIL`); fails++; }
  else {
    const c = cSecondHand('F32', W, 0, x, M, K);
    report('F32', relErr(ref, got, M), c ? relErr(c, got, M) : null);
  }
}

// Shape contract: k that is not a whole number of blocks must be refused, not
// silently read past the row (nt_qrows_for, notorch.c:5230).
for (const [dtype, badK, name] of [[2, 48, 'Q4_0'], [6, 48, 'Q5_0'], [8, 48, 'Q8_0'],
                                   [12, 128, 'Q4_K'], [14, 128, 'Q6_K'], [99, 32, 'unknown']]) {
  const rc = qmatvec(new Float32Array(1), new Uint8Array(1024), dtype, new Float32Array(badK), 1, badK);
  if (rc !== -1) { console.log(`${name} k=${badK} expected rc=-1, got ${rc}  FAIL`); fails++; }
}

// ggufHalfToFloat rebuilds the f32 bit pattern instead of going through
// Math.pow twice. Faster only matters if it is also the same function, so every
// one of the 65536 half patterns is checked — subnormals, both infinities, and
// the whole NaN payload range included. The comparison runs each side through
// the kernel's own arithmetic (accumulate from 0, land in an f32), because a
// negative-zero half is -0 as a value and +0 once summed, and that difference
// belongs to the accumulator, not to the conversion.
{
  const x1 = new Float32Array(1); x1[0] = 1;
  const o1 = new Float32Array(1);
  const half = new Uint16Array(1);
  const halfBytes = new Uint8Array(half.buffer);
  const viaPow = (h) => {
    const sg = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1F, f = h & 0x3FF;
    if (e === 0) return sg * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : sg * Infinity;
    return sg * Math.pow(2, e - 15) * (1 + f / 1024);
  };
  let bad = 0, firstBad = null;
  for (let h = 0; h < 65536; h++) {
    half[0] = h;
    if (qmatvec(o1, halfBytes, 1, x1, 1, 1) !== 0) { bad++; continue; }
    const want = Math.fround(0 + viaPow(h) * 1), got = o1[0];
    if (!(Object.is(want, got) || (Number.isNaN(want) && Number.isNaN(got)))) {
      bad++;
      if (!firstBad) firstBad = `0x${h.toString(16)}: ${want} vs ${got}`;
    }
  }
  if (bad) { console.log(`ggufHalfToFloat: ${bad}/65536 patterns disagree (${firstBad})  FAIL`); fails++; }
  else console.log('f16     [65536]  every half pattern matches the Math.pow form  PASS');
}

// quantAct must round ties to even, the way C's lrintf does under the default
// FE_TONEAREST (notorch.c:5515) — Math.round rounds them up instead. On uniform
// input the two never disagree: measured 0 exact halves in 14336 activations, so
// every random-input check above is blind to the difference. This case builds the
// halves on purpose. x[0] = 127 forces amax = 127, hence d = 1 and id = 1, so the
// value being rounded IS the activation, and every other entry sits on .5 exactly.
// The expectations are written out from the definition rather than computed, so
// this cannot agree with the implementation by sharing its mistake.
{
  const K = 32, x = new Float32Array(K);
  x[0] = 127;
  for (let i = 1; i < K; i++) x[i] = i - 16.5;
  const want = [127,
    -16, -14, -14, -12, -12, -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0,
    0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14];
  const qa = new Int8Array(K), da = new Float32Array(1);
  quantAct(x, K, qa, da);
  let bad = 0;
  for (let i = 0; i < K; i++) if (qa[i] !== want[i]) bad++;
  if (da[0] !== 1) { console.log(`quantAct halves: expected d=1, got ${da[0]}  FAIL`); fails++; }
  if (bad) {
    console.log(`quantAct ties-to-even: ${bad}/${K} activations off  FAIL`);
    fails++;
  } else {
    console.log(`ties    [k=${K}] 32/32 activations round to even  PASS`);
  }
}

// quantAct must also hold the scale and the scaled activation at f32, because C
// does (notorch.c:5511-5512) and JS would otherwise carry f64 into the rounding.
// This is rarer than the ties case — 13 of 6.4M random activations move by one
// int8 step when the frounds are dropped — so a random-input check finds it maybe
// one run in thirty. These two literals were searched out to make it certain:
// with the frounds the pair quantizes to 76, without them to 75.
// Two pairs, because the three frounds are not one guard: the first moves only
// when the fround on the product goes, the second only when the scale and its
// reciprocal stop being f32. Each was searched out to make a rare event certain.
for (const [name, amax, v, want] of [
  ['product', 0.24229010939598083, 0.14403860270977020, 76],
  ['scale',   0.92275577783584595, 0.41051736474037170, 56],
]) {
  const K = 32, x = new Float32Array(K);
  x[0] = amax;   // largest in the block, so it sets d and its reciprocal
  x[1] = v;
  const qa = new Int8Array(K), da = new Float32Array(1);
  quantAct(x, K, qa, da);
  if (qa[1] !== want) {
    console.log(`quantAct f32 ${name}: qa[1]=${qa[1]}, expected ${want}  FAIL`);
    fails++;
  } else {
    console.log(`fround  [${name.padEnd(7)}] activation held at f32, qa[1]=${want}  PASS`);
  }
}

// The i8 dispatch refuses the same shapes plus the two dense dtypes, which have
// no integer kernel at all (nt_qrows_i8_for, notorch.c:6294).
for (const [dtype, badK, name] of [[2, 48, 'Q4_0'], [6, 48, 'Q5_0'], [8, 48, 'Q8_0'],
                                   [12, 128, 'Q4_K'], [14, 128, 'Q6_K'],
                                   [0, 2048, 'F32'], [1, 2048, 'F16'], [99, 32, 'unknown']]) {
  const rc = qmatvecI8(new Float32Array(1), new Uint8Array(65536), dtype, new Float32Array(badK), 1, badK);
  if (rc !== -1) { console.log(`i8${name} k=${badK} expected rc=-1, got ${rc}  FAIL`); fails++; }
  const rcRows = qmatvecI8Rows(new Float32Array(1), new Uint8Array(65536), dtype,
                               new Int8Array(badK), new Float32Array(badK / 32 + 1), 0, 1, badK);
  if (rcRows !== -1) { console.log(`i8${name}Rows k=${badK} expected rc=-1, got ${rcRows}  FAIL`); fails++; }
}

// dequantRow refuses what it cannot decode whole, rather than reading into the
// neighbouring row: unknown dtype, a column count that is not whole blocks, and
// a row index off either end.
{
  const bytes = new Uint8Array(4096);
  const dst = new Float32Array(256);
  const bad = [
    [1, 0, 256, 'F16 has no block decode'],
    [12, 0, 128, 'Q4_K with 128 columns'],
    [8, -1, 256, 'negative row'],
    [8, 1 << 20, 256, 'row past the end'],
    [99, 0, 256, 'unknown dtype'],
  ];
  for (const [dtype, row, cols, why] of bad) {
    const rc = dequantRow(dst, bytes, dtype, row, cols);
    if (rc !== -1) { console.log(`dequantRow ${why}: expected -1, got ${rc}  FAIL`); fails++; }
  }
}

// A packed weight is inference-only. Backward must say so instead of reading
// W.data[...] = undefined and filling the gradient with NaN.
{
  const K2 = 256, OUT = 8, nb2 = K2 / 32;
  const W = new Uint8Array(OUT * nb2 * 34);          // Q8_0
  for (let i = 0; i < W.length; i++) W[i] = rndByte();
  for (let row = 0; row < OUT; row++)
    for (let b = 0; b < nb2; b++) setS32(W, (row * nb2 + b) * 34);

  for (const which of ['seqLinear', 'embedding']) {
    const nt = new Notorch();
    const wIdx = nt.leaf(Tensor.fromPacked(W, which === 'seqLinear' ? [OUT, K2] : [OUT, K2], 8));
    let yIdx;
    if (which === 'seqLinear') {
      const x = new Float32Array(K2);
      for (let i = 0; i < K2; i++) x[i] = rndUnit();
      yIdx = nt.seqLinear(wIdx, nt.leaf(new Tensor(x, [1, K2])), 1);
    } else {
      yIdx = nt.embedding(wIdx, nt.leaf(new Tensor(Float32Array.from([2]), [1])), 1, K2);
    }
    if (!Number.isFinite(nt.get(yIdx).data[0])) {
      console.log(`${which} packed forward produced a non-finite value  FAIL`); fails++;
    }
    let threw = '';
    try { nt.backward(yIdx); } catch (err) { threw = String(err.message); }
    if (!/packed/.test(threw)) {
      console.log(`${which} backward on a packed weight: expected a refusal, got "${threw || 'no throw'}"  FAIL`);
      fails++;
    }
  }
  console.log('packed  [backward] seqLinear and embedding refuse packed weights  PASS');
}

console.log(fails === 0 ? 'JS_QMATVEC_OK' : `${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
