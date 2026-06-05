// infer_gguf.mjs — run a GGUF model end-to-end in pure JS on notorch.js.
// Loads a quantized GGUF (dequant verified vs C), builds the byte-level BPE
// tokenizer from the GGUF, runs the llama/mistral transformer forward, and
// generates. This is the browser-viable RUN path (CPU here; WebGPU packed
// matvec is the next step). RoPE is interleaved — correct for llama/mistral;
// qwen3 (NEOX + q/k-norm) is a follow-up.
//
//   node infer_gguf.mjs model.gguf "prompt" [maxTokens] [temp]
//
import { readFileSync } from 'fs';
import { Notorch, Tensor, loadGGUF } from './notorch.js';

// ── GGUF byte-level BPE (mirror of examples/bpe.c) ──────────────────────────
function buildByteTable() {
  const cp = new Int32Array(256), cp2b = new Int32Array(512).fill(-1);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    cp[b] = printable ? b : (256 + n);
    if (!printable) n++;
  }
  for (let b = 0; b < 256; b++) if (cp[b] < 512) cp2b[cp[b]] = b;
  return { cp, cp2b };
}
class GgufBPE {
  constructor(tokens, merges) {
    this.tokens = tokens;
    this.vocab = new Map();
    for (let i = 0; i < tokens.length; i++) this.vocab.set(tokens[i], i);
    this.rank = new Map();
    if (merges) for (let r = 0; r < merges.length; r++) this.rank.set(merges[r], r);
    const { cp, cp2b } = buildByteTable();
    this.cp = cp; this.cp2b = cp2b;
    this.enc = new TextEncoder(); this.dec = new TextDecoder();
  }
  // byte -> the unicode codepoint GGUF uses (space->Ġ etc.), as a JS string char
  byteChar(b) { return String.fromCodePoint(this.cp[b]); }
  encode(text) {
    const bytes = this.enc.encode(text);
    const ids = [];
    let i = 0;
    while (i < bytes.length) {
      let j = i + 1;
      while (j < bytes.length && bytes[j] !== 0x20) j++;   // split on space, leading space attaches
      let sym = [];
      for (let b = i; b < j; b++) sym.push(this.byteChar(bytes[b]));
      // greedy merge by lowest rank
      while (sym.length > 1) {
        let bestRank = Infinity, bi = -1;
        for (let k = 0; k < sym.length - 1; k++) {
          const r = this.rank.get(sym[k] + ' ' + sym[k + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; bi = k; }
        }
        if (bi < 0) break;
        sym.splice(bi, 2, sym[bi] + sym[bi + 1]);
      }
      for (const s of sym) { const id = this.vocab.get(s); if (id !== undefined) ids.push(id); }
      i = j;
    }
    return ids;
  }
  decodeOne(id) {
    const s = this.tokens[id]; if (!s) return new Uint8Array(0);
    const out = [];
    for (const ch of s) { const b = this.cp2b[ch.codePointAt(0)]; if (b >= 0) out.push(b); }
    return new Uint8Array(out);
  }
  decode(ids) {
    let bytes = [];
    for (const id of ids) bytes.push(...this.decodeOne(id));
    return this.dec.decode(new Uint8Array(bytes));
  }
}

// ── model load ──────────────────────────────────────────────────────────────
function loadModel(path) {
  const buf = readFileSync(path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const { metadata, tensors } = loadGGUF(ab);
  const arch = metadata.get('general.architecture');
  const g = (k, d) => { const v = metadata.get(`${arch}.${k}`); return v === undefined ? d : v; };
  const m = {
    arch,
    E: g('embedding_length'), L: g('block_count'),
    H: g('attention.head_count'), KV: g('attention.head_count_kv'),
    FFN: g('feed_forward_length'),
    ropeBase: g('rope.freq_base', 10000),
    eps: g('attention.layer_norm_rms_epsilon', 1e-5),
    tensors,
  };
  m.HD = (tensors.get('blk.0.attn_q.weight').shape[0] / m.H) | 0; // [out,in] -> out=H*HD
  m.tokEmb = tensors.get('token_embd.weight');
  m.outNorm = tensors.get('output_norm.weight');
  m.output = tensors.get('output.weight') || m.tokEmb;   // tied if absent
  m.vocab = m.tokEmb.shape[0];
  const tokens = metadata.get('tokenizer.ggml.tokens');
  const merges = metadata.get('tokenizer.ggml.merges');
  m.tok = new GgufBPE(tokens, merges);
  m.eos = metadata.get('tokenizer.ggml.eos_token_id');
  return m;
}

// ── forward (full sequence; returns last-position logits Float32Array) ───────
function forwardLastLogits(nt, m, ids) {
  const T = ids.length, E = m.E, H = m.H, KV = m.KV, HD = m.HD;
  nt.resetTape();
  const W = (t) => nt.leaf(t);
  const idsT = nt.leaf(new Tensor(Float32Array.from(ids), [T]));
  let x = nt.embedding(W(m.tokEmb), idsT, T, E);          // [T,E]
  for (let l = 0; l < m.L; l++) {
    const g = (n) => m.tensors.get(`blk.${l}.${n}`);
    const xn = nt.seqRmsnorm(x, W(g('attn_norm.weight')), T, E, m.eps);
    const q = nt.rope(nt.seqLinear(W(g('attn_q.weight')), xn, T), T, HD, m.ropeBase);
    const k = nt.rope(nt.seqLinear(W(g('attn_k.weight')), xn, T), T, HD, m.ropeBase);
    const v = nt.seqLinear(W(g('attn_v.weight')), xn, T);
    const ao = nt.gqaCausalAttention(q, k, v, T, HD, H, KV);
    const o = nt.seqLinear(W(g('attn_output.weight')), ao, T);
    x = nt.add(x, o);
    const fn = nt.seqRmsnorm(x, W(g('ffn_norm.weight')), T, E, m.eps);
    const ff = nt.swigluFFN(fn, W(g('ffn_gate.weight')), W(g('ffn_down.weight')), W(g('ffn_up.weight')), T);
    x = nt.add(x, ff);
  }
  const xn = nt.seqRmsnorm(x, W(m.outNorm), T, E, m.eps);
  // last-position hidden → logits
  const last = nt.slice(xn, T - 1, T, 0);                 // [1,E]
  const logitsIdx = nt.seqLinear(W(m.output), last, 1);   // [1,vocab]
  return nt.get(logitsIdx).data;
}

// ── main ─────────────────────────────────────────────────────────────────────
const [, , path, prompt = 'The capital of France is', maxTokStr = '6', tempStr = '0'] = process.argv;
const maxTok = parseInt(maxTokStr), temp = parseFloat(tempStr);
const m = loadModel(path);
console.error(`model: arch=${m.arch} E=${m.E} H=${m.H} KV=${m.KV} HD=${m.HD} FFN=${m.FFN} L=${m.L} V=${m.vocab} rope=${m.ropeBase} eos=${m.eos}`);
const nt = new Notorch();
const ids = m.tok.encode(prompt);
console.error(`prompt "${prompt}" -> ${ids.length} tokens`);
let out = '';
for (let step = 0; step < maxTok; step++) {
  const logits = forwardLastLogits(nt, m, ids);
  let next = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[next]) next = i;
  if (next === m.eos) break;
  ids.push(next);
  out += m.tok.decode([next]);
  process.stderr.write('.');
}
console.error('');
console.log(prompt + out);
