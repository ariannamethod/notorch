import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Notorch, OP, Tensor } from './notorch.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg}: got ${a}, expected ${b}`);
}

function assertThrows(fn, pattern, msg) {
  try {
    fn();
  } catch (err) {
    if (!pattern || pattern.test(String(err.message))) return;
    throw new Error(`${msg}: threw wrong error "${err.message}"`);
  }
  throw new Error(`${msg}: did not throw`);
}

function testOpCodesMirrorC() {
  const header = readFileSync(resolve(root, 'notorch.h'), 'utf8');
  const defs = [...header.matchAll(/^#define\s+NT_OP_([A-Z0-9_]+)\s+(\d+)/gm)];
  assert(defs.length >= 37, `expected at least 37 C op defines, got ${defs.length}`);
  for (const [, name, codeText] of defs) {
    const code = Number(codeText);
    if (code > 36) continue;
    assert(Object.prototype.hasOwnProperty.call(OP, name), `JS OP missing C op ${name}`);
    assert(OP[name] === code, `JS OP.${name}=${OP[name]} != C ${code}`);
  }
  for (const [name, code] of Object.entries(OP)) {
    if (code > 36) assert(code >= 100, `JS extension ${name} overlaps C op range`);
  }
}

function testReluUsesCOp35() {
  const nt = new Notorch();
  const x = nt.param(Tensor.fromArray([-1, 0, 2], [3]));
  const y = nt.relu(x);
  assert(nt.tape.entries[y].op === 35, `relu recorded op ${nt.tape.entries[y].op}, expected 35`);
  assertClose(nt.get(y).data[0], 0, 1e-6, 'relu(-1)');
  assertClose(nt.get(y).data[1], 0, 1e-6, 'relu(0)');
  assertClose(nt.get(y).data[2], 2, 1e-6, 'relu(2)');

  nt.tape.entries[y].grad = new Float32Array([1, 1, 1]);
  nt.tape._backwardOp(nt.tape.entries[y]);
  assertClose(nt.tape.entries[x].grad[0], 0, 1e-6, 'relu grad negative');
  assertClose(nt.tape.entries[x].grad[1], 0, 1e-6, 'relu grad zero');
  assertClose(nt.tape.entries[x].grad[2], 1, 1e-6, 'relu grad positive');
}

function testSeqGateGuards() {
  const nt = new Notorch();
  const x = nt.param(Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]));
  const g = nt.param(Tensor.fromArray([0.5, 2.0, 0.5, 3.0], [2, 2]));
  const y = nt.seqGate(x, g, 2, 2, 1);
  assert(nt.tape.entries[y].op === OP.SEQ_GATE, 'seqGate op code');
  assertClose(nt.get(y).data[0], 2, 1e-6, 'seqGate t0 d0');
  assertClose(nt.get(y).data[5], 18, 1e-6, 'seqGate t1 d2');

  assertThrows(() => nt.seqGate(x, g, 2, 2, 2), /invalid T\/nm\/gi/, 'seqGate rejects gi >= nm');
  assertThrows(() => nt.seqGate(x, g, 4, 2, 1), /not divisible/, 'seqGate rejects non-divisible x length');
  assertThrows(() => nt.seqGate(x, g, 2, 3, 1), /gate len/, 'seqGate rejects gate length mismatch');
}

function testRrpramBroadcastGuards() {
  const nt = new Notorch();
  const T = 2, E = 4, H = 2, hD = 2, rank = 1, ctxT = 2;
  const wrLen = H * E * rank + H * rank * ctxT;
  const wr = nt.param(Tensor.zeros([wrLen]));
  const x = nt.param(Tensor.zeros([T * E]));
  const v = nt.param(Tensor.zeros([T * E]));
  const y = nt.rrpramBroadcastAttention(wr, x, v, T, E, H, hD, rank);
  assert(nt.tape.entries[y].op === OP.RRPRAM_BCAST, 'rrpramBroadcast op code');

  const shortX = nt.param(Tensor.zeros([T * E - 1]));
  assertThrows(
    () => nt.rrpramBroadcastAttention(wr, shortX, v, T, E, H, hD, rank),
    /x len/,
    'rrpramBroadcast rejects short x',
  );
  const shortV = nt.param(Tensor.zeros([T * E - 1]));
  assertThrows(
    () => nt.rrpramBroadcastAttention(wr, x, shortV, T, E, H, hD, rank),
    /v len/,
    'rrpramBroadcast rejects short v',
  );
}

testOpCodesMirrorC();
testReluUsesCOp35();
testSeqGateGuards();
testRrpramBroadcastGuards();

console.log('JS_OP_PARITY_OK');
