// test_kvcache.mjs — incremental attention must equal the full-sequence one.
//
// The property that makes a KV cache safe is not "it runs faster": it is that
// feeding T positions at once and feeding them one at a time produce the same
// numbers. Everything a cache can get wrong — an off-by-one in the causal
// mask, a RoPE position taken from the window instead of the sequence, a stale
// prefix — breaks that equality and nothing else has to be inspected.
//
//   node test_kvcache.mjs   → JS_KVCACHE_OK
import { Notorch, Tensor, KVCache } from './notorch.js';

let _s = 0x2545F491;
function rnd32() { _s ^= _s << 13; _s >>>= 0; _s ^= _s >>> 17; _s ^= _s << 5; _s >>>= 0; return _s; }
function rndUnit() { return (rnd32() / 0xFFFFFFFF) * 2 - 1; }
function randArr(n) { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = rndUnit(); return a; }

let fails = 0;
const T = 9, HD = 8, H = 6, KV = 3, QD = H * HD, KVD = KV * HD;

// 1. With Tq === Tkv the cached op is the plain causal one, to the bit.
{
  const nt = new Notorch();
  const q = nt.leaf(new Tensor(randArr(T * QD), [T, QD]));
  const k = nt.leaf(new Tensor(randArr(T * KVD), [T, KVD]));
  const v = nt.leaf(new Tensor(randArr(T * KVD), [T, KVD]));
  const a = nt.get(nt.gqaCausalAttention(q, k, v, T, HD, H, KV)).data;
  const b = nt.get(nt.gqaAttentionKV(q, k, v, T, HD, H, KV)).data;
  let diff = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
  if (diff) { console.log(`gqaAttentionKV at Tq=Tkv: ${diff}/${a.length} elements differ  FAIL`); fails++; }
  else console.log(`equiv   [T=${T}] cached op equals gqaCausalAttention exactly  PASS`);
}

// 2. One pass over T positions == T passes of one position through a cache.
{
  const Q = randArr(T * QD), K = randArr(T * KVD), V = randArr(T * KVD);
  const nt = new Notorch();
  const whole = nt.get(nt.gqaAttentionKV(
    nt.leaf(new Tensor(Q, [T, QD])),
    nt.leaf(new Tensor(K, [T, KVD])),
    nt.leaf(new Tensor(V, [T, KVD])), T, HD, H, KV)).data;

  const cache = new KVCache(T, KVD);
  const stepwise = new Float32Array(T * QD);
  for (let t = 0; t < T; t++) {
    nt.resetTape();
    cache.append(K.subarray(t * KVD, (t + 1) * KVD), V.subarray(t * KVD, (t + 1) * KVD));
    const row = nt.get(nt.gqaAttentionKV(
      nt.leaf(new Tensor(Q.slice(t * QD, (t + 1) * QD), [1, QD])),
      nt.leaf(cache.Ktensor()), nt.leaf(cache.Vtensor()), 1, HD, H, KV)).data;
    stepwise.set(row, t * QD);
  }
  let diff = 0, worst = 0;
  for (let i = 0; i < whole.length; i++) {
    if (whole[i] !== stepwise[i]) { diff++; worst = Math.max(worst, Math.abs(whole[i] - stepwise[i])); }
  }
  if (diff) { console.log(`incremental: ${diff}/${whole.length} differ, worst ${worst.toExponential(2)}  FAIL`); fails++; }
  else console.log(`incr    [T=${T}] one pass equals ${T} cached single-token passes  PASS`);
}

// 3. RoPE at posOffset p over one position == the p-th row of RoPE over p+1.
{
  const nt = new Notorch();
  const D = QD;
  const x = randArr(T * D);
  const full = nt.get(nt.rope(nt.leaf(new Tensor(x, [T, D])), T, HD, 10000)).data;
  let diff = 0;
  for (let t = 0; t < T; t++) {
    nt.resetTape();
    const one = nt.get(nt.rope(
      nt.leaf(new Tensor(x.slice(t * D, (t + 1) * D), [1, D])), 1, HD, 10000, t)).data;
    for (let d = 0; d < D; d++) if (one[d] !== full[t * D + d]) diff++;
  }
  if (diff) { console.log(`rope posOffset: ${diff} elements differ from the windowed rope  FAIL`); fails++; }
  else console.log(`rope    [T=${T}] posOffset row equals the full-window row  PASS`);

  // posOffset defaults to 0, i.e. the pre-cache behaviour is untouched.
  nt.resetTape();
  const dflt = nt.get(nt.rope(nt.leaf(new Tensor(x, [T, D])), T, HD, 10000)).data;
  let same = true;
  for (let i = 0; i < dflt.length; i++) if (dflt[i] !== full[i]) same = false;
  if (!same) { console.log('rope with no posOffset changed behaviour  FAIL'); fails++; }
}

// 4. Cached attention is inference-only and backward says so.
{
  const nt = new Notorch();
  const q = nt.leaf(new Tensor(randArr(1 * QD), [1, QD]));
  const k = nt.leaf(new Tensor(randArr(T * KVD), [T, KVD]));
  const v = nt.leaf(new Tensor(randArr(T * KVD), [T, KVD]));
  const y = nt.gqaAttentionKV(q, k, v, 1, HD, H, KV);
  let threw = '';
  try { nt.backward(y); } catch (err) { threw = String(err.message); }
  if (!/inference-only/.test(threw)) {
    console.log(`GQA_ATTN_KV backward: expected a refusal, got "${threw || 'no throw'}"  FAIL`);
    fails++;
  } else console.log('packed  [backward] cached attention refuses to produce a gradient  PASS');
}

// 5. Shape contract: fewer keys than queries is a bug, not a clamp.
{
  const nt = new Notorch();
  const q = nt.leaf(new Tensor(randArr(4 * QD), [4, QD]));
  const k = nt.leaf(new Tensor(randArr(2 * KVD), [2, KVD]));
  const v = nt.leaf(new Tensor(randArr(2 * KVD), [2, KVD]));
  let threw = '';
  try { nt.gqaAttentionKV(q, k, v, 4, HD, H, KV); } catch (err) { threw = String(err.message); }
  if (!/fewer than/.test(threw)) {
    console.log(`gqaAttentionKV with Tkv < Tq: expected a refusal, got "${threw || 'no throw'}"  FAIL`);
    fails++;
  }
}

console.log(fails === 0 ? 'JS_KVCACHE_OK' : `${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
