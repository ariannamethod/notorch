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
import { qmatvec } from './notorch.js';

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
function cSecondHand(tag, W, dtype, x, m, k) {
  if (!cref) return null;
  const inPath = join(tmp, `${tag}.in`), outPath = join(tmp, `${tag}.out`);
  const head = Buffer.alloc(16);
  head.writeInt32LE(dtype, 0); head.writeInt32LE(m, 4); head.writeInt32LE(k, 8);
  head.writeInt32LE(W.length, 12);
  writeFileSync(inPath, Buffer.concat([head, Buffer.from(W.buffer, W.byteOffset, W.byteLength),
                                       Buffer.from(x.buffer, x.byteOffset, x.byteLength)]));
  execFileSync(cref, [inPath, outPath]);
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

function report(name, rel, relC) {
  const ok = rel < 1e-3 && (relC === null || relC < 1e-3);
  const cTxt = relC === null ? '' : `  C ${relC.toExponential(2)}`;
  console.log(`${name.padEnd(5)} [m=${M} k=${K}] rel ${rel.toExponential(2)}${cTxt}  ${ok ? 'PASS' : 'FAIL'}`);
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

console.log(fails === 0 ? 'JS_QMATVEC_OK' : `${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
