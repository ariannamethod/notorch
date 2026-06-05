// test_gguf_dequant.mjs — verify notorch.js loadGGUF block-dequant matches the
// C path (gguf.c) byte-for-byte. Prophetic-debt test for the JS GGUF upgrade.
//
//   1. /tmp/gguf_dequant_ref model.gguf t1 t2 ... > ref.json   (C reference)
//   2. node test_gguf_dequant.mjs model.gguf ref.json          (this — compares)
//
// PASS if every checked value matches C within 1e-4 absolute.
import { readFileSync } from 'fs';
import { loadGGUF } from './notorch.js';

const [, , ggufPath, refPath] = process.argv;
if (!ggufPath || !refPath) { console.error('usage: node test_gguf_dequant.mjs model.gguf ref.json'); process.exit(2); }

const ref = JSON.parse(readFileSync(refPath, 'utf8'));
const buf = readFileSync(ggufPath);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const { tensors } = loadGGUF(ab);

const DTYPE = { 0: 'F32', 1: 'F16', 2: 'Q4_0', 6: 'Q5_0', 8: 'Q8_0', 12: 'Q4_K', 14: 'Q6_K' };
let maxAbs = 0, maxRel = 0, checked = 0, fail = 0;

for (const [name, r] of Object.entries(ref)) {
  const t = tensors.get(name);
  if (!t) { console.log(`  MISSING in JS: ${name}`); fail++; continue; }
  let tAbs = 0;
  for (let i = 0; i < r.vals.length; i++) {
    const c = r.vals[i], j = t.data[i];
    const abs = Math.abs(c - j), rel = abs / (Math.abs(c) + 1e-8);
    if (abs > tAbs) tAbs = abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    checked++;
  }
  console.log(`  ${name.padEnd(26)} ${(DTYPE[r.dtype] || r.dtype).padEnd(5)} n=${String(r.n).padEnd(9)} maxAbs=${tAbs.toExponential(2)}`);
}

console.log(`\nchecked=${checked} values | maxAbs=${maxAbs.toExponential(3)} maxRel=${maxRel.toExponential(3)} | missing=${fail}`);
const ok = fail === 0 && maxAbs < 1e-4 && checked > 0;
console.log(ok ? 'JS_DEQUANT_OK' : 'JS_DEQUANT_FAIL');
process.exit(ok ? 0 : 1);
