# NOTORCHLOG

The running engineering log of notorch. Every fix, every verified change,
every bug-class closed — dated, with commit and proof. The README is the
spec and the manifesto; **this is the work**.

Convention: small fixes (bug fixes, sync-discipline corrections, single-op
work, doc/docstring touch-ups) are recorded **here**. Large changes (a new
backend, a new op family, a new training method, an architecture shift) get
a section in the README too. When in doubt: it goes here first.

Newest entries on top.

---

## 2026-08-24 — the tokenizer speaks SentencePiece

`examples/bpe.c` implemented byte-level BPE and nothing else, and every
LLaMA-family GGUF in this tree carries SentencePiece. The two schemes share a
file format and share nothing else: one has a merge list and merges by rank,
the other has a score per token, writes space as U+2581, and falls back to
`<0xHH>` tokens for anything its vocabulary does not cover. Run the first over
the second and a space encodes to `Ġ`, which is not in the vocabulary, and the
old code dropped it without a word.

Which scheme a file carries is now decided by what it hands over — a merge
list, or scores without one — and not by the name in `tokenizer.ggml.model`.
That is deliberate: the name is a label, the arrays are what make an algorithm
possible.

The SentencePiece path prepends the dummy space these vocabularies were built
over, splits into UTF-8 characters, and merges the highest-scoring adjacent
pair until no adjacent pair is a token. Pair scores live beside the symbols and
only the two around a merge are recomputed, so the vocabulary is consulted O(n)
times for a whole string instead of once per scan; finding the best pair is a
walk, which at the length of a chat line is cheaper than a heap. Anything the
vocabulary does not carry goes out as bytes, and anything that cannot even do
that now says so on stderr instead of vanishing.

On nano_arianna Q8_0: "The capital of France is Paris." was 26 tokens, five
short of its 31 bytes, one missing per space, every token a single character
from the tail of the vocabulary. It is now **7 tokens** and decodes back to
itself. What the model does with that is the whole point — the same prompt at
temp 0 used to continue `,anewfield,anewarchitecture.` and now continues `the
cathedral of the French language, the most important document in the world.`

The gate had existed and had never been aimed here. It now runs six strings
including a leading-and-trailing-space case and an emoji no vocabulary carries,
across both schemes, and it asserts one thing a round-trip alone cannot:
**merges have to be doing something.** Red hand proved why. Removing the U+2581
substitution left every round-trip green — the byte fallback rebuilds the text
from `<0x20>` — and only the token count gave it away, 15 against 7. Removing
the byte fallback turned the emoji case red as `'🔥 fire' -> '  fire'`, which is
the original bug's exact shape. A round-trip is not enough; a round-trip plus a
count is.

Byte-level vocabularies were correct before and are untouched: smallcoder-303M
at 49152 tokens and a Qwen3 at 151936 both pass all seven checks. `make test_bpe
MODEL=…`. Harness/example parity stays green, because the fix landed in the file
they share.

Open and named: the JS edition's `GgufBPE` in `js-edition/infer_gguf.mjs` splits
on spaces the same way and has the same hole for SentencePiece files.

---

## 2026-08-24 — a harness, and the tokenizer it caught

`harness/` is the simple way to run a model: `notorch model.gguf "prompt"` for
one shot, `notorch model.gguf` for a chat in the terminal. The forward, the KV
cache, the sampler and the scalar pieces moved out of `examples/infer_llama.c`
into `harness/runtime.c` and `harness/arch_llama.c`, with the arithmetic
untouched — `harness/test_parity.sh` is what says so, comparing the generated
continuation against the example at temp 0 and getting six identical answers
across two models and three prompts. Timing overlaps within run-to-run noise:
prefill 1352 / 1391 / 1394 t/s against the example's 1321 / 1470 / 1463, decode
428 / 495 / 597 against 479 / 492 / 592, nano_arianna Q8_0 on an A18 Pro.

The example stays where it is. It is what the phone numbers were measured with
and what models are tested through, and until the harness has been pointed at
the same pile of files, replacing the reference with the thing being tested
would leave nothing to test against. Two copies of one forward is the debt this
takes on knowingly; it closes when the harness has earned the reference's job.

Two decisions worth naming. **stdout is the model, stderr is everything else** —
banner, model shape, the prompt you typed, timings, profile — so a redirected
run is text and not a transcript of the tool. And **architectures are a table**:
`nt_arch` is names, load, free, forward, and llama registers with `names = NULL`
as the fallback, which is what the example already did with every architecture
it had never heard of. Adding a family should be adding a file and one line; if
it ever needs a branch inside `runtime.c`, the interface is lying and it is the
interface that gets fixed.

Red hand on both failure modes a move like this has: flipping the RoPE
convention turned all six parity checks red with fluent text that answers a
different question, and writing the KV one position early turned them red with
`,I.AIeiIH:` where the example says `,anewfield,anewarchitecture.`

**And then the harness caught something older than itself.** Its own reason to
exist is a person typing a sentence and reading one back, which is a harder
test of the tokenizer than any benchmark, and the text came back without
spaces. `examples/bpe.c` implements GPT-2 byte-level BPE. nano_arianna carries
`tokenizer.ggml.model = "llama"` — SentencePiece — where 22965 of 32000 tokens
begin with U+2581 and none begin with `Ġ`. Encode maps a space to `Ġ`, finds no
such token, and **drops it silently**: "The capital of France is Paris." is 31
bytes and comes back as 26 tokens, five short, one per space, every remaining
token a single character from the tail of the vocabulary rather than a merge.
Decode cannot map U+2581 back, because the table it reads is 512 entries wide
and that codepoint is 9601.

The gate for this already existed and had never been pointed here: `bpe.c`
built with `-DBPE_TEST` prints `BPE_FAIL (roundtrip=0 merges_applied=1)` on
this file, and the `merges_applied` is only true because five tokens went
missing. Byte-level vocabularies — Qwen2.5, SmolLM2 — are unaffected and always
were. This is not from the move; it is what the move made visible, and it is
the next thing to fix, before another architecture is added.

---

## 2026-08-24 — the thirteen tensors that were two fifths of the time

Q4_K and Q6_K now have wasm kernels, and the case for writing them was not the
tensor count. With only the row formats, nano_arianna Q4_K_M ran 3378 of its
3937 matvecs through wasm and handed back 559 — 7 Q4_K tensors and 6 Q6_K,
14.2 percent of the calls. Those thirteen are `ffn_down` and the output head.
Moving them across took prefill from 35.7 / 38.8 / 39.2 to 67.2 / 67.4 / 67.9
t/s and decode from 34.6 / 39.2 / 39.7 to 58.5 / 62.2 / 63.6, on an A18 Pro
with a 20-token prompt and 24 greedy tokens, each configuration in its own
process. One seventh of the calls held at least two fifths of the prefill, and
the same file measures 17.0 / 17.1 / 17.4 prefill on the exact f32 path it
started from.

Both kernels are ports of the JS int8 pair, which is what the gate holds them
to. Q4_K's affine minimum lifts out of the dot the way Q5_0's -16 does — a
value is `d*s6*q - dmin*m6`, so a sub-block is `d*s6*SUM(q*a) - dmin*m6*SUM(a)`
and the integer loop only ever sees raw nibbles in [0,15]. Sub-blocks 2p and
2p+1 share one 32-byte span, low nibbles feeding the even one and high the odd,
so the unpack is a mask or a shift and never a table. Q6_K reconstructs
`(ql | qh<<4) - 32`, which lands in [-32,31] and stays int8-safe, so the whole
reconstruction is vector work; its sub-scale covers 16 values against the
activation block's 32, which is why the integer accumulator is per weight
sub-block and `d*sc[j]*da[j/2]` is applied once at the end. The sixteen
sub-sums are drained ascending, the order the per-token kernel adds them in,
because a different order is a different float.

Agreement with the JS i8 kernels holds at 6.69e-7 worst across all 93 tensors
of the model at their own shapes, and the module is 9086 bytes with one import.

Red hand on each kernel separately, and each stayed in its own lane: selecting
the wrong nibble half in Q4_K reddened Q4_K at 3.51e+0 / 6.76e-1 / 5.36e-1 with
Q6_K untouched, and dropping Q6_K's -32 bias reddened Q6_K at 1.02e+0 / 2.13e+0
/ 1.39e+0 with Q4_K untouched. `test_wasm.mjs` now sweeps five formats over
three row counts, and its contract case changed meaning: Q4_K and Q6_K are no
longer refused for having no kernel, they are refused for a k that is not a
whole number of 256-value blocks.

Still scalar and still costing: Q5_0's high-bit expansion in the wasm kernel
builds its sixteen bytes in a loop before the vector load. That is a candidate,
not a claim — nothing here measured it.

---

## 2026-08-24 — the wasm kernel gets callers

The SIMD kernel landed with a green gate and nobody calling it. `git grep
WasmKernels` found three hits — the module, its test, the README — and
`infer_gguf.mjs` imported notorch.js and the worker pool and nothing else, so
every matvec in every run went through plain JS while 6 KB of `i16x8.extmul`
sat checked in, correct at nothing.

Wiring it is not a line of glue. A wasm kernel can only read the address space
it was handed, and an existing SharedArrayBuffer cannot be handed to it — so
either the weights are copied in per call, which costs more than the kernel
saves, or the model lives there from the start. The load was inverted:
`WasmKernels.fromModelFile` sizes an imported shared memory for the file and
reads the bytes straight into it, `loadGGUF` learned a `base` so a file can
start anywhere in a buffer, and from then on a packed tensor's `byteOffset` is
the pointer the kernel wants. `build.sh` gained `--import-memory
--shared-memory`; `qkernels.c` did not change a line. That the memory is shared
is not incidental — `WorkerPool.create` refuses anything that is not a
SharedArrayBuffer — so one buffer now serves the JS kernels, the wasm kernels
and the pool, and `toShared`'s second copy of the whole model goes with it.

Measured on nano_arianna 89M Q8_0, A18 Pro, 20-token prompt and 24 greedy
tokens, each configuration in its own process, three runs: prefill 29.8 / 31.0
/ 32.4 t/s exact against 208.0 / 212.5 / 219.9 through wasm; decode 22.8 / 23.1
/ 24.3 against 175.8 / 177.7 / 179.7.

That is a composite, and it was decomposed before it was claimed. `NT_I8=1`
runs the same integer arithmetic in plain JS: 26.6 / 26.8 / 27.3 prefill, 19.9
/ 20.1 / 20.5 decode — slower than the exact f32 path it replaces. The int8
algorithm is a 13 percent loss in JS, and the whole ~6.9x prefill and ~7.7x
decode is the instruction. One measurement against the exact path would have
credited the algorithm with the instruction's work.

Coverage reads better than the file names suggest. On the Q8_0 file wasm takes
all 93 packed tensors and refuses none; on the Q4_K_M build of the same model
it takes 80 of 93 — 73 Q5_0 and 7 Q8_0 — and hands back 7 Q4_K and 6 Q6_K. That
is the C side's finding from two days ago arriving in a different edition: a
file called Q4_K_M is mostly not Q4_K.

**This path is not bit-identical to the one beside it**, and the gate says so
rather than asserting a coincidence. The activation is quantized to int8 as
`nt_qmatvec_i8` does, and greedy decoding is a chain of argmaxes over numbers
that moved. Across all 93 tensors at the model's own shapes the worst row lands
1.08e-2 from the exact answer, the prompt's logits within 1.68e-2, and the
continuation of "The capital of France is" holds for eleven tokens and splits
at the twelfth. The first draft of the gate asserted token identity: it passed
at six tokens and failed at twelve. Token identity is a coincidence with a
shelf life, so it is printed and never gated on, and `nt.wasm` is opt-in for
the same reason.

`test_wasm_e2e.mjs` multiplies every packed tensor at the shape the model uses,
then runs a real forward with the kernels on and off. Red hand on both failure
modes this wiring actually has: a weight pointer shifted by one 34-byte block
turned 93 tensors red at 1.7e+5 and the forward at 3.7e+1; `useWasm` forced
false — the shape a buffer-identity miss takes — was caught only by the call
counter, "attached and never called", with every arithmetic check still green.
The second is the one worth having. A fast path that silently does not run
looks exactly like a fast path that is slow.

---

## 2026-08-24 — notorch quantizes, and the ladder it costs

The library could read every packed format it runs and produce none of them. `nt_quantize_row`
writes Q4_0, Q5_0 and Q8_0; `tools/gguf_quantize.c` rewrites a whole f16 or f32 GGUF into
them (`bab4ec4`). The metadata section is copied byte for byte — tokenizer, chat template,
architecture keys survive untouched — and only the tensor directory is rewritten, because
types and offsets are what quantization moves. Policy is `llama-quantize --pure`: 2-D tensors
whose row divides 32 convert, the rest copy through.

The arithmetic is llama.cpp's reference to the bit, which is the requirement rather than a
concession: a file only earns the name GGUF if everything else can read it. Qwen2.5-0.5B fp16
through both quantizers and compared tensor by tensor — 291 tensors identical byte for byte
at Q4_0, Q5_0 and Q8_0. Ours takes 3.4 s on the phone against llama-quantize's 3.0.

**The FMA trap, third appearance.** The reference computes `x*id`, rounds it to a float, then
adds `+8.5` and truncates. As one expression the compiler fuses multiply and add, the last bit
moves, and truncation flips for any value near an integer: 143 tensors of 291 differed by one
level in one nibble. Diagnosed the same way as before — rebuild with `-ffp-contract=off`, watch
it agree — and fixed by storing the scaled product through a `volatile`, a store and a load per
weight in code that runs once per model.

**What the ladder costs**, same weights, four big cores, decode of 64 tokens, perplexity over
32 chunks of 512 from wikitext-2:

| format | tensor data | decode | PPL | vs fp16 |
|---|---:|---:|---:|---:|
| fp16 | 1207.8 MiB | — | 13.7214 ± 0.436 | — |
| Q8_0 | 638.7 MiB | 33.0 t/s | 13.8076 ± 0.440 | +0.6% |
| Q5_0 | 413.4 MiB | 45.6 t/s | 14.6143 ± 0.474 | +6.5% |
| Q4_0 | 338.3 MiB | 57.0 t/s | 16.0112 ± 0.527 | +16.7% |

Decode tracks bytes and nothing else, which is what a memory-bound engine looks like from
outside. Q5_0 buys 38 percent of speed over Q8_0 for six percent of perplexity; Q4_0 buys a
further 25 for ten more. The 57.0 t/s is the fastest this phone has decoded anything, and it
is llama.cpp's own file format made by our own tool.

K-quants are not here. Q4_K and Q6_K quantize with a per-super-block search over scale and
minimum rather than a single absmax, and writing that to the bit is a separate piece of work
from writing the block formats; the kernels have read them since June, and the quantizer will
say so plainly when it can produce them.

---

## 2026-08-23 — decode: a mutex per row chunk, and a half-float decoded in software

Prefill had been the whole story for three days and decode had not moved: 22.9 t/s on an
Exynos 1580 against llama.cpp's 45.6 for the same Qwen2.5-0.5B Q4_K_M. Decode is 168
matvecs per token on a 24-layer model, so it is the shape where every per-call cost is paid
168 times, and the profile said 94 percent of it was inside the matvecs.

**The fan-out cost more than the work it split** (`8646fdd`). Measured at the shapes decode
asks for: 896x896 took 197.9 us on one core and 197.8 on four — a speedup of 1.00 — and
128x896 went from 35.4 us to 83.4, twice as slow threaded as not. Every chunk claim took the
pool mutex, so four workers serialised sixty-four times per dispatch; the claim is now one
fetch_add and the job is read without a lock, published before the generation bump that lets
anyone see it. Workers were woken through a condvar, a futex wake and a scheduler round-trip
each, seventy-three times per token; they now spin on the generation counter first, with the
condvar underneath so an idle phone still sleeps (`NT_QMV_SPIN` sets the budget). The pool
also sized itself to the core count while the dispatching thread drains alongside it, which
put five threads on four cores — one fewer now. After: 896x896 at 3.59x, per-call overhead
148 us -> 5.3 us.

**Which inverted the threading floor.** It existed because dispatches were expensive. Swept
on decode: 27.3 t/s at the old 4M, 33.5 at 512K, 35.0 at 64K, flat below. The default is 64K.

**The half float** (`9d351bf`). Every packed kernel converts one f16 scale per block, so
`nt_f16_to_f32` sits in the innermost loop of every matvec in the library — about eleven
million calls per decoded token. It was a branch, a shift chain and a while loop for
subnormals; aarch64 has done it in one FCVT since armv8. 8.4 percent of decode for two
lines, on every dtype at once.

Also in that commit: the activation block sums move out of the kernels, where the chunked
dispatch had them rebuilt once per row chunk, to the quantization that produces the bytes
they sum. It measures neutral — the sums are two SDOTs against an activation already in L1 —
and stays because it removes the k <= 65536 ceiling the stack-held version imposed.

**What the experiments said no to**, because a log of only the things that worked teaches
the wrong lesson: software prefetch of the weight stream, swept 4 / 8 / 16 / 32 blocks
ahead, changed nothing — the hardware prefetcher already had it. Pinning the activation
address so every load hit L1 bought 1.7 percent, so activation traffic is not the
constraint and pairing rows to share it would not pay. Removing the float tail entirely
measured slower than keeping it. Removing the Q5_0 high-bit expansion entirely: no change.

Decode of Qwen2.5-0.5B Q4_K_M, 96 tokens, four big cores: **38.9 / 38.8 / 39.1 t/s against
22.9 at the start of the day** — 68 percent, and 85 percent of llama.cpp where it was 50.
In bandwidth: 14.2 GiB/s of the 17.98 this phone can stream, against 12.9 before. Pinning
the weights in L1 now buys 23 percent where it bought 8, which is the same statement from
the other side: the kernels stopped being the constraint and the memory bus started.
Prefill is unchanged at 82.6 t/s, and the greedy continuation is unchanged.

---

## 2026-08-23 — SMMLA: the instruction nothing else on this phone uses

The batched kernels were bound by feeding, not by arithmetic. Doubling the SDOTs at constant
loads measured free on an Exynos 1580 — 7.7 ms against 7.9 for the same shape — which says
the four 16-byte activation loads per 64 multiply-accumulates were the constraint. SMMLA
multiplies two 2x8 int8 matrices in a single instruction: two weight rows in one operand,
two activations in the other, four dot products of eight retired at once. Each activation
half is then read once and serves two rows, and the bytes per MAC halve (`4ecf6f6`).

All five dtypes take it, with the odd row and the odd activation handed back to the SDOT
path. Isolated at m=2048 k=4096 n=32, measured against the SDOT batched kernel in the same
thermal window: Q4_0 5.4 ms against 9.1, 1.66x. End to end on four big cores, 241-token
prompt: Qwen2.5-1.5B Q4_0 prefill 33.0 / 33.2 t/s against 27.8 / 25.8 / 25.5; Qwen2.5-0.5B
Q4_K_M 81.9 / 81.9 / 81.9 against 72.1 / 72.4 / 71.9. llama.cpp on those two files does 26.5
and 88.6.

**A bit-identity trap worth naming.** Q4_K and Q6_K compute a product chain and an
accumulate that the compiler is free to contract into fused instructions, and it chose
differently in the SDOT kernel and the SMMLA one — same arithmetic, last bit apart, 2358 of
4096 outputs flagged. Neither result is wrong and no tolerance would have caught it as a
problem; the test compares bits, and the fix is to stop leaving the choice to the compiler.
`nt_q4k_acc` and `nt_q6k_acc` name both fused operations explicitly, so every kernel rounds
where those lines say it rounds. `tests/test_qmatmul.c` is now 34 shapes across five dtypes,
including odd row counts, all identical to the per-token path.

Where this leaves the phone, on a 241-token prompt: prefill on the 0.5B went 20.85 → 26.2 →
48.5 → 59.3 → 69.4 → 81.9 t/s across the last three days, against llama.cpp's 88.6 on the
same weights, and on the 1.5B Q4_0 it is 33.2 against 26.5 — past it. Decode is untouched at
21-22 t/s against 45.6 and is the next thing worth a plan, not a patch: it is one activation
against the whole model, which is the shape SMMLA cannot help.

---

## 2026-08-22 — a section profiler, and the formats the file was actually made of

`NT_PROFILE=1` on `infer_llama` (`e32a6f1`) accumulates wall time around ten sections of
the forward — embedding, norms, qkv, rope and cache writes, attention, projections, FFN
matmuls, SiLU, residuals, head — and prints prefill and decode separately. Unset, each
call is a predicted branch on a static int.

It was asked because batching had removed the weight traffic and left a guess in its
place, and it answered against the guess. Qwen2.5-0.5B Q4_K_M, 241-token prompt, four big
cores of an Exynos 1580: of 8446 ms, the FFN matmuls take 5719, qkv 1008, the attention
projection 781, attention itself 709. Norms, rope, SiLU, residuals and the head together
are 229 ms — 2.7 percent. Nothing around the kernels is worth touching.

What was worth touching is which kernels run at all (`292be7a`). A file called Q4_K_M is
mostly not Q4_K: this model's hidden size is 896, no K-quant block divides it, and
`llama_model_loader` reports 133 tensors q5_0, 13 q8_0, 12 q4_K, 12 q6_K. The batched path
covered 12 of 170 tensors, which is why the Q4_K work measured 8 percent end to end while
its isolated kernel measured 3.3x.

Q5_0 and Q8_0 now have batched kernels. Q5_0 gains most — its unpack is two table loads,
an AND, a shift and two ORs before a single dot, now paid once per tile rather than once
per token — and its -16 bias lifts out as `16*SUM(qa)`, the same lift Q4_K's affine
minimum takes, so both read the per-block activation sum the call already builds.

Isolated at m=2048 k=4096 n=32: Q5_0 7.2 ms batched against 34.7 per token (4.79x), Q8_0
6.8 against 21.6 (3.18x). End to end, same model and prompt, three interleaved repeats:
prefill 52.0 / 47.2 / 46.4 t/s against 27.1 / 25.4 / 24.6 — 1.9x, and 55 percent of
llama.cpp's 88.6 on the same file where it was 30. `tests/test_qmatmul.c` covers 21 shapes
across four dtypes, all bit-identical to the per-token path, including 4864x896 where k is
not a multiple of 256.

Q6_K followed (`da1802d`), the last format still walking the prompt one token at a time and
the holder of half the down projections in this file. Its sixteen integer sub-block sums are
kept per activation and drained afterwards in ascending order — the order the per-token
kernel adds them in — because folding each into the accumulator as it appears would be the
same arithmetic in a different order, and a different order is a different float. Isolated:
11.1 ms batched against 35.2 per token, 3.16x. End to end: prefill 61.7 / 58.7 / 57.4 t/s
against 51.9 / 50.7 / 48.9.

With the matmuls no longer dominating, attention surfaced as the second line of the profile
— 725 ms of 3861 — still scalar over head_dim. Four lanes with a scalar tail (`1ceb044`):
prefill 70.3 / 69.6 / 68.4 t/s against 62.7 / 59.8 / 59.3, the section down to 218 ms.
Decode is unchanged within noise; its KV is short and attention was never its cost.

**That last one is not bit-identical and must not be read as if it were.** Four partial sums
are a different summation order from one running sum, and greedy generation is a chain of
argmaxes over numbers that moved: the 24-token continuation of "The capital of France is"
now ends "It is also the capital of the" where it read "It is located in the south of".
The matmul kernels stay bit-exact and their test still asserts equality — this is attention
alone.

The line as a whole, on this file: prefill 20.85 t/s before any of it, 69.4 after, against
llama.cpp's 88.6 on the same weights. The profile now reads 3331 ms as 2438 ms of FFN
matmul, 278 qkv, 218 attention, 164 projection, 136 SiLU. Measured on the phone, one node,
four cores; other hardware will read differently.

---

## 2026-08-22 — the instruction JS does not have

Everything the JS edition could reach had been reached: unrolling bought 1.4x,
workers bought 2.7x, int8 bought nothing and batching bought nothing, and the
measurement that explained all of it was 0.556 ns per element on 306 KB against
0.569 on 19 MB — compute-bound, with no memory win left to take. What remained
was the operation count itself, and one instruction JS cannot express: sixteen
products at once.

`js-edition/wasm/qkernels.c` is that kernel, freestanding wasm32 with
`-msimd128`: 6 KB, no libc, no imports, nothing but pointers into the one linear
memory the host owns. `i16x8.extmul` over an `i8x16` pair, folded by
`i32x4.extadd_pairwise`. Q4_0, Q5_0 and Q8_0; anything else returns -1 and the
caller falls back to notorch.js, correct and only as slow as before. The
activation quantizer is the C one down to lrintf's round-half-to-even.

A18 Pro, each path in its own process, warmed to steady state:

| shape | dtype | plain JS | wasm SIMD | |
|---|---|---|---|---|
| 576×576 | Q5_0 | 0.333 ms | 0.101 ms | 3.3x |
| 32000×576 | Q8_0 | 16.34 ms | 1.34 ms | 12.2x |

The JS baseline moves with V8 warmth and the number deserves the caveat: the
same kernel in a script that had already run it hundreds of times on other
shapes measured 11.95 ms rather than 16.34, which puts the head at 8.9x instead
of 12.2x. Both are real. The honest claim is an order of magnitude on the head
and about 3x on small matrices.

Getting an honest number took three tries, each wrong in its own way, and all
three are worth naming because they are the same mistake in different clothes —
measuring the harness instead of the code. First the two paths were timed
through a shared closure, which gave the JS side a polymorphic call the wasm
side did not pay: 50% handed to wasm for free. Then they were timed in one
process, where whichever ran second inherited a cold cache. Then in separate
processes but with different warm-up histories, which is the 11.95-against-16.34
above.

Correctness took one correction too. The gate first held wasm to 2e-2 against
the exact path at every m, and m=1 Q4_0 failed at 5.76e-2 — but notorch.js's own
int8 kernel misses by exactly the same 5.759e-2 on that row. The tolerance is a
statistic over many rows (C measures it at m=512); at m=1 it is one number
divided by one number, and int8 quantization alone lands there. Agreement with
the JS int8 path, which holds at ~1e-7 for every m, is the check that actually
tests the port.

Red hand: Q4_0 zero-point off by one, an arithmetic shift where the nibble
needs a logical one, Q5_0 without its -16 lift — all caught. Ties-to-even was
not, until the same built-half input the JS gate uses was added here too; then
16 of 32 activations round the wrong way.

---

## 2026-08-22 — notorch answers to `from notorch import`

`make shared` builds libnotorch.so (dylib on macOS), and `python/notorch.py` is
a ctypes layer over it: 246 lines of type declarations and no arithmetic. No
numpy, no build step on the Python side, no dependencies — ctypes is standard
library. Weights stay packed: `tensor.packed` is a pointer into the file's own
bytes and the kernels read them in place, so a Q4_K tensor costs roughly half a
byte per weight in Python exactly as it does in C.

The failure mode a binding like this has is not a crash. A ctypes Structure
whose offsets have drifted from the header reads its neighbouring fields and
reports them as data, and nothing about that looks wrong. So
`tests/gguf_layout.c` prints what the compiler actually laid out, and
`make test_python MODEL=…` compares every offset against it before touching a
model. Red hand, all three caught: a field dropped from gguf_file (sizeof 443544
against 443552), a name array one size short (tensor_info 184 against 192),
GGUF_MAX_TENSORS off by one (443360 against 443552).

Then it checks a real file: nano_arianna Q4_K_M reads back as
`llama L=13 E=576 V=32000 tensors=120`, `gguf_dequant_row` equals its slice of
`gguf_dequant`, and `nt_qmatvec` matches dequant-then-matvec at rel 8.6e-07.

The README says so on the second screen rather than the last, because someone
who wants to read a GGUF and multiply by it should not have to scroll to find
out they can.

## 2026-08-22 — a worker pool for js-edition, and a batched matmul that measured its way back out

Two experiments, one kept.

**Kept: `notorch-workers.mjs`**, an optional resident pool that splits a matvec's
rows across threads. Rows are independent and write disjoint slots, so output is
bit-identical to `qmatvec` and the gate asserts equality. Rows are claimed from a
shared cursor, not divided up front: on 2 performance and 4 efficiency cores an
even split measured 1.72x where the cursor measured 1.94x. `qmatvecRows` is now
exported for it, and `seqLinear` uses a pool when one is attached to the engine.

A18 Pro, 6 workers, against a clean single-threaded baseline measured in its own
process — the in-process baseline runs ~10% slow once a pool has been alive
beside it, which would have inflated these:

| shape | dtype | single | pool | |
|---|---|---|---|---|
| 576×576 | Q5_0 | 0.21 ms | 0.10 ms | 2.10x |
| 576×1536 | Q4_K | 0.45 ms | 0.21 ms | 2.14x |
| 32000×576 | Q8_0 | 10.77 ms | 4.01 ms | 2.69x |

End to end, 24 greedy tokens: 2.04 s to 1.09 s, output identical.

**Dropped: the batched matmul.** Porting C's `nt_qmatmul_i8` shape to JS —
unpack a weight row once, dot it against a tile of activations — measured
1.15x at n=32 on Q8_0, 1.08x on Q4_K, and *below one* at n=2 and n=8. In C the
same shape is worth 3.19x, because there the win is memory traffic. In JS there
is no such win to take, which a direct measurement makes plain: the same kernel
costs 0.556 ns per element on a 306 KB working set and 0.569 ns on a 19 MB one.
JS inference is compute-bound end to end; cache residency changes nothing.

That single number explains the whole trajectory of this work. int8 does not help
(the win needs an instruction JS does not have). Batching does not help (it saves
bandwidth nobody was short of). Unrolling helped, 1.4x, because it removes
operations and dependencies. Workers help, because they add executors. Only those
two levers exist here.

Three notes on gates, all learned the hard way in this step:

- The pool's first version passed buffers by `postMessage` and reported every
  round complete having computed nothing: the caller blocks on `Atomics.wait`, so
  the workers' event loops never processed the message. Everything shared is now
  bound at construction.
- A `setTimeout` deadlock guard in the test was useless for the same reason — a
  wedged pool stops the event loop the timer lives in. The pool times out its own
  wait instead and throws, and the test lets that through.
- Two of three red-hand defects did not fail the gate, and both times the defect
  was the problem, not the test. A worker ignoring its chunk end recomputes rows
  with identical values; a caller waiting on the wrong slot degrades parking into
  a spin. Neither changes an answer. The one that does — dropping a row from each
  chunk — was caught at 1/501 and 16/501.

## 2026-08-21 — the phone front: batched prefill, honest core counts, and a segfault that only glibc could see

Five commits from Defender (Galaxy A56, Exynos 1580, Termux and a glibc chroot),
recorded here because the log had not caught up with them. Everything below is
their measurement, on their hardware.

**Batched prefill** (`1eb756c`, `e83809a`). Prefill pushed one token at a time
through the packed matvec, so every weight byte was read once per prompt token —
a 0.5B Qwen streams 373 MiB per token, and a 241-token prompt streamed it 241
times. Prefill cost exactly what generation cost, which is the wrong shape for an
agent: long prompt, short answer. `nt_qmatmul_i8` unpacks a weight row once and
dots it against a tile of activations, so the traffic divides by the tile width.
The activation side is the same per-32-block int8 and the per-row accumulation
order is unchanged, so outputs are bit-identical rather than merely close;
`tests/test_qmatmul.c` asserts equality across the tile boundary (n = 31, 32, 33)
and on both sides of the threading gate. Q4_0 landed first, then Q4_K — the
format almost every GGUF on the hub actually carries, and the one whose per-block
overhead is most worth amortizing: eight 6-bit (scale, min) pairs unpacked from
twelve bytes, and `SUM(qa)` now built once per call instead of once per row range.

Isolated at m=2048 k=4096 n=32: 7.0 ms batched against 22.5 ms per-token on Q4_0
(3.19x), 8.0 ms against 26.3 ms on Q4_K (3.30x). End to end on a 241-token prompt:
Q4_0 prefill 19.3 t/s against 12.9 with the chunk forced to 1; Q4_K 27.4 / 26.0 /
25.3 t/s against 24.9 / 24.1 / 23.9 with the per-token fallback. Greedy
continuations unchanged.

**The threading gate was measuring the wrong quantity** (`e83809a`). It counted one
matvec's work while the call performs n of them, so a 0.5B Qwen's 896x896 query
projection sat under the 4M floor and a 32-position chunk of it — 25M weight
elements — stayed on one core.

**Core counts** (`4281b5f`). `nt_qmv_host_threads` sized the pool from
`sysconf(_SC_NPROCESSORS_ONLN)`, which reports the cores the kernel has online, not
the cores this process may run on. Every big.LITTLE measurement pins to the fast
cluster, and there the old count returned 8 while four were usable — the pool
oversubscribed two to one and each matvec waited on a context switch instead of on
memory. The affinity mask is the honest number; `NT_QMV_THREADS` overrides both.
Three interleaved repeats of 96 tokens: 21.6 / 20.4 / 19.5 t/s against
20.0 / 18.3 / 18.5, mean 20.5 against 18.9, text unchanged character for character.

**Logits nobody reads** (`42d930e`). Prefill ran the full forward for every prompt
token, head included, then discarded every distribution but the last. The head is
the largest matvec in the model — 151936 rows against 896 columns on a 0.5B Qwen —
so a 241-token prompt spent a tenth of its time producing 240 unread
distributions. A NULL logits pointer now means "KV cache only". 24.3 and 21.5 t/s
against 22.0 and 19.7.

**A segfault only glibc could see** (`925cc2d`). `-std=c11` asks glibc for strict
ISO, and under it `strdup`, `getpagesize` and `posix_memalign` are not declared:
they become implicit ints, the returned pointer truncates to 32 bits, and the first
dereference is a SIGSEGV. Bionic declares them at c11 regardless, which is why
Termux builds never showed it and the glibc chroot on the same phone did. `gnu11`
is the same language with the declarations present.

Note for the JS edition: `seqLinear` has the identical prefill hole — it calls
`qmatvec` once per position and re-reads every weight T times. The batched shape
above is the fix, and it needs no threading and no headers to work in a browser.

## 2026-08-20 — four accumulators instead of one, and an int8 claim withdrawn

The packed kernels ran one serial `acc +=` per block, so every addition waited
on the previous one. Four independent accumulators let them overlap; the scale
moves out of the inner loop with them, and in Q6_K the sub-scale index is
constant across each 16-wide half and hoists as well. `ggufHalfToFloat` rebuilds
the f32 bit pattern rather than calling `Math.pow` twice.

nano_arianna Q4_K_M shapes, median of five:

| shape | dtype | before | after | |
|---|---|---|---|---|
| 576×576 | Q5_0 | 0.321 ms | 0.212 ms | 1.51x |
| 1536×576 | Q5_0 | 0.835 ms | 0.563 ms | 1.48x |
| 576×1536 | Q6_K | 0.837 ms | 0.590 ms | 1.42x |
| 32000×576 | Q8_0 | 16.59 ms | 11.11 ms | 1.49x |

End to end, 24 greedy tokens: 2.95 s to 2.05 s. Output byte-for-byte unchanged,
packed and dense alike — reordering the sums moves nothing that survives
rounding to f32 — and the distance to the C kernels stays at ~1e-6.

**Withdrawn: the int8 path is not faster in JS.** The entry of 2026-08-18 and
the README both claimed it would win on the type, an int32 accumulator staying
in V8's small-integer form instead of running an f32 dependency chain. That was
reasoning, not measurement. Measured twice, on real shapes, median of five:
0.97x / 0.95x / 1.02x / 1.01x before the unroll, and 0.95x / 0.94x / 0.82x /
1.01x after it — at best a wash, usually a loss. `quantAct` is not the cost
either, it measures 0.001–0.004 ms. The thing that makes i8 cheaper in C is one
instruction covering sixteen products, and JS has no such instruction; what
would change it is WASM SIMD, a different artifact with a build step.

A private microbenchmark did show i8 ahead by 19%, which is why the claim
survived as long as it did. It fed the kernel a pre-quantized activation and a
synthetic shape, and neither held up against the real ones. A benchmark that
does not run the code the way the program runs it is a hypothesis wearing a
number.

`qmatvecI8` stays: it is the C contract, it is verified against C's own i8
kernel to 3e-7, and it is what a SIMD backend would call. It just no longer
promises anything about speed here.

## 2026-08-20 — js-edition stops recomputing the prefix it already computed

The 5.4x measured two days ago is collected. `infer_gguf.mjs` prefills the
prompt once and then feeds one token per step through per-layer `KVCache` —
the class had been in the file, unused, since it was written.

Two pieces had to exist first:

- `rope(x, T, headDim, freqBase, posOffset)`. A single-token step sits at
  absolute position `pos`, not at 0, and the same offset goes into ROPE
  backward, where the angle is recomputed. `posOffset` defaults to 0, so every
  existing caller is untouched.
- `gqaAttentionKV(q, k, v, Tq, headDim, nHeads, nKvHeads)`, JS extension op 108.
  Tkv comes from K's own shape rather than a fifth aux slot, and the query at
  `i` answers for absolute position `Tkv - Tq + i`. At `Tq === Tkv` it equals
  `gqaCausalAttention` to the bit.

Inference-only, and `GQA_ATTN_KV` backward says so: the causal structure came
from the cache length, not from the tape.

nano_arianna Q4_K_M, packed, greedy, one prompt throughout:

| tokens | no cache | KV cache | speedup |
|---|---|---|---|
| 8 | 11.73 s | 2.02 s | 5.81x |
| 24 | 49.98 s | 3.73 s | 13.40x |

These are not comparable to the 6.55 s quoted in the entry below — that run used
a shorter prompt. Within the table everything is one prompt on one machine.

Proof (neo, node v25.9.0):

- Output byte-for-byte identical to the pre-cache reference at 24 tokens, and
  identical again under `NT_PACKED=0`.
- `test_kvcache.mjs`, wired into `make test_js` and `npm test`: the cached op
  equals `gqaCausalAttention` at Tq=Tkv; one pass over T positions equals T
  single-token cached passes; a `posOffset` row equals the corresponding
  full-window row; backward refuses; Tkv < Tq refuses.
- Red hand, each caught: mask dropping the cache prefix — 384/432 elements
  differ, worst 1.16e+0; forward RoPE ignoring `posOffset` — 384 elements
  differ; both refusals removed — FAIL. On the real model, a query RoPE offset
  by one and a mask without the prefix each derail the generation outright.

The property the test holds is the one worth stating plainly: a cache is correct
when feeding T positions at once and feeding them one at a time give the same
numbers. Speed is what you get afterwards, not what you check.

## 2026-08-18 — packed becomes the default in js-edition, and the slowness gets a number

`loadGGUF` now keeps quantized weights packed unless asked for `{ packed: false }`.
`test_gguf_dequant.mjs` asks for dense explicitly — it exists to check the f32
block decode against C, which is a different question — and gained a check that
the default is packed, because the numeric part of that test is blind to it:
flipping the default back leaves `maxAbs=5.000e-8` untouched while costing
4 B/weight. `infer_gguf.mjs` takes `NT_PACKED=0` to force the old path.

Generation on nano_arianna Q4_K_M is byte-for-byte identical across 24 greedy
tokens between the new default, the forced-dense path, and the reference saved
before the switch.

Where the time actually goes, since 8 tokens out of an 89M model in 6.55 s is
not a kernel problem:

- `infer_gguf.mjs` re-forwards the whole prefix per token. 11-token prompt,
  8 generated: 6.15 GMAC against 1.13 GMAC with a KV cache. **5.4x of the work
  is thrown away.** `KVCache` exists (`notorch.js:3150`) and nothing uses it.
- What is left is throughput: 0.94 GMAC/s packed, 1.31 GMAC/s dense, scalar
  single-threaded JS. The packed gap is `qmatvec` re-decoding a block per pass.

Order of return, measured rather than guessed: KV cache (5.4x, pure
architecture) far ahead of i8 in `seqLinear` (kernel), ahead of workers
(parallelism).

## 2026-08-18 — js-edition stops unpacking the weights it now knows how to read

`loadGGUF(ab, { packed: true })` leaves the quantized families in their blocks:
the tensor holds a `Uint8Array` view onto the file's own bytes and its GGUF
dtype (`Tensor.fromPacked`). `seqLinear` branches through `qmatvec`, `embedding`
through the new `dequantRow` — the port of `gguf_dequant_row`, one row decoded
where the dense path decodes the table. F32 and F16 still expand; their
consumers here are norms and other non-matvec ops that read dense data.

The default stays f32. Both paths have to run on one model for the gate to mean
anything, so the switch exists either way, and flipping the default is its own
decision.

Packed weights are inference-only, and both backward paths say so by name.
Without that, `SEQ_MATVEC` backward reads `W.data[i*inDim+j]` on an empty array
and fills the gradient with NaN — a wrong answer wearing the shape of a working
one, which is the failure mode this whole step exists to avoid.

nano_arianna Q4_K_M, 69.4 MB, 93 of 120 tensors packed:

| | dense | packed |
|---|---|---|
| f32 bytes built at load | 354,546,432 | 62,208 |
| heap + external after load | +339.8 MB | +1.9 MB |
| load time | 190 ms | 5 ms |
| peak RSS, 8 tokens | 538 MB | 287 MB |
| wall time, 8 tokens | 4.68 s | 6.55 s |

Load collapses to 5 ms because a packed tensor is a view onto the buffer already
read, not a copy of it. The 40% slower generation is the honest trade as it
stands: `qmatvec` re-decodes a block every pass where the dense path decoded once
at load. `qmatvecI8` is the answer and already exists, but `seqLinear` does not
reach for it yet — an approximate kernel can move the tokens, so the identity
check below would have to change shape first. Separate step, separate gate.

Proof (neo, node v25.9.0):

- Generation byte-for-byte identical across 24 greedy tokens, `NT_PACKED=0` vs
  `NT_PACKED=1`, on a prompt long enough to walk every layer.
- `make test_js` / `npm test` green; real-model dequant parity unchanged
  (`JS_DEQUANT_OK`, `maxAbs=5.00e-8`).
- Red hand, each caught: `dequantRow` reading the next row — 5 formats FAIL;
  its bounds check removed — `RangeError` out of the DataView; the backward
  refusals removed — 2 FAIL; packed `seqLinear` writing row 0 — generation
  diverges; `embedding` reading `tid+1` — generation diverges.

A measurement instrument lied during this step and is worth recording: a
`kill -0` loop reported the test as hung past 40 s because nothing reaped the
background process, and `kill -0` succeeds on a zombie. Timed directly, the same
run was 0.28 s. The harness was wrong, not the code — check the instrument
before believing the anomaly.

## 2026-08-18 — js-edition gets the int8-activation matvec, and two rounding guards that random input cannot see

`qmatvecI8` / `qmatvecI8Rows` / `quantAct` port `nt_qmatvec_i8`,
`nt_qmatvec_i8_rows` and `nt_quant_act_q8`: the activation goes to per-32 int8,
the dot accumulates in integers. Q4_0, Q5_0, Q8_0, Q4_K, Q6_K, with the Q5_0 and
Q4_K lifts intact — `SUM((q-16)*x)` as `SUM(q*x) - 16*SUM(x)`, and Q4_K's minimum
the same way — so the integer loop never sees a subtraction. Approximate by
construction; `qmatvec` stays the exact reference, as in C.

C's `NT_QMV_ASUM_MAX` (`notorch.c:5244`) is not reproduced: it caps k at 65536
because its activation sums are stack-held, which is a C storage detail rather
than semantics. The JS contract takes any k that divides into whole blocks.

Two guards carry the port and neither shows up under random input:

- `lrintf` (`notorch.c:5515`) rounds ties to even; `Math.round` rounds them up.
  Measured 0 exact halves in 14336 uniform activations — every random check is
  blind to this. On an input built to land on halves, 16 of 32 activations move
  by a full int8 step.
- C holds the scale, its reciprocal and the scaled activation at f32
  (`notorch.c:5511-5512`); JS would carry f64 into the rounding. 13 of 6.4M
  random activations move by one step when the `Math.fround`s are dropped —
  about one test run in thirty would notice.

Both are now pinned by literals searched out for the purpose, since a gate that
only fires one run in thirty is not a gate.

Proof (neo, Accelerate, node v25.9.0):

- `make test_js` / `npm test` — 12 matvec rows + 3 rounding rows, `JS_QMATVEC_OK`.
- i8 vs the exact packed path: `4.07e-3` Q4_0, `3.13e-3` Q5_0, `3.64e-3` Q8_0,
  `3.34e-3` Q4_K, `3.25e-3` Q6_K — against the C tolerance of 2e-2
  (`tests/test_qmatvec.c:227`).
- Second hand against C's OWN i8 kernel (`--i8` on `tests/js_qmatvec_ref.c`, same
  bytes through a file): `1.73e-7` to `3.25e-7` — an order tighter than the
  packed path's `1e-6`, because most of the work is integer, where f64 and f32
  cannot differ.
- Red hand, all FAIL: Q4_0 zero-point `2.48e-1`, Q8_0 sign-extend `1.89e+0`,
  Q5_0 without the `16*asum` lift `9.78e-1`, Q4_K without the `dmin` lift
  `5.59e-2`, Q6_K on the wrong activation scale `4.26e-2`; `Math.round` for the
  ties `16/32 off`; each `Math.fround` dropped separately.
- `qmatvecI8Rows` over `[0,173)` and `[173,512)` equals the single call exactly.
  The first defect tried against this guard — forcing `r0 = 0` — was a bad one:
  the second call rewrites every row correctly, so it proves nothing. A row base
  of `(row - r0)` does fail it.

## 2026-08-18 — js-edition gets the packed matvec, and its own gate learns to see NaN

The JS edition had drifted behind the C kernels: op parity was intact (0–36, all
37 `NT_OP_*` defines), but `loadGGUF` still expanded every quantized tensor to f32
on load — 4 B/weight where Q4_K on disk is ~0.55, so a 170 MB file becomes north of
a gigabyte in a browser tab. `qmatvec(out, Wq, dtype, x, m, k)` ports `nt_qmatvec`:
one block unpacked into locals at a time, no dense tensor ever built. F32, F16,
Q4_0, Q5_0, Q8_0, Q4_K, Q6_K; `-1` for a dtype or a `k` with no kernel, same
contract as `nt_qrows_for` (`notorch.c:5230`). `loadGGUF` is untouched — moving
storage onto packed bytes is its own step with its own gate.

The gate found its own blind spot first. The clean run showed `rel 0.00e+0` across
all seven formats, and a byte-swapped f16 kernel still **passed**: the kernel
returned NaN, NaN fails every comparison in the error reducer, and `maxAbs` stayed
at zero. Proximity was being measured where finiteness was never checked. Fixed,
then re-falsified per format.

Proof (neo, Accelerate, node v25.9.0):

- `make test_js` and `npm test` — `JS_OP_PARITY_OK` + `JS_QMATVEC_OK`, 7/7 PASS.
- Red hand, one defect per format, all FAIL at rc=1: Q4_0 zero-point dropped
  `2.46e-1`, Q5_0 high bit swapped `8.10e-1`, Q8_0 sign-extend dropped `1.88e+0`,
  Q4_K min subtract dropped `3.93e-2`, Q6_K wrong sub-scale `6.61e-1`, F16 and F32
  byte-swapped `Infinity`.
- Second hand: `tests/js_qmatvec_ref.c` runs `nt_qmatvec` on the identical bytes
  handed over through a file, not on a second generator believed to agree. JS vs C
  is `1.02e-6`–`1.68e-6` across the seven — the width of the accumulator, f64 in JS
  against f32 in C, three orders under the 1e-3 threshold.
- README debt closed by measurement: it claimed Q5_0 had no local file to run
  against. nano_arianna Q4_K_M carries `token_embd.weight` 32000×576 in Q5_0;
  `test_gguf_dequant.mjs` puts it at `maxAbs=5.00e-8` against C.
- Regression: `infer_gguf.mjs` generates token-for-token identically on the HEAD
  `notorch.js` and this one.

Noted, not touched: `infer_gguf.mjs` decodes without word separators
(`resonance is,akindofthefield,a`) on both versions — a BPE-decode defect that
predates this change.

## 2026-07-30 — qmatvec pthread worker reuse (non-OpenMP path)

WTForacle surfaced the remaining per-call pthread overhead in the packed matvec path.
OpenMP consumers already reuse their caller/runtime team; non-OpenMP consumers now
reuse persistent pthread workers for `nt_qmatvec` and `nt_qmatvec_i8`, with the caller
computing the final shard inline. That keeps the packed-row contract unchanged while
removing `pthread_create`/`pthread_join` from each decode matvec.

`NT_QMV_POOL=0` restores the old per-call pthread fallback. `NT_QMV_THREAD_MIN` /
`nt_qmv_set_thread_min` still decide when row threading starts.

Proof (neo, Accelerate):

- `make test` — notorch 49/49, vision+BPE 73/73; only the pre-existing
  `nt_image_load_mem` unused-function warning remains.
- `cc -O2 -Wall -Wextra -std=c11 -pthread -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK tests/test_qmatvec.c notorch.c -framework Accelerate -lm -o /private/tmp/notorch_test_qmatvec_pool`
- `NT_QMV_THREAD_MIN=1 /private/tmp/notorch_test_qmatvec_pool` — F32/F16/Q4_0/i8Q4_0/Q5_0/Q8_0/Q4_K/Q6_K PASS, ALL PASS.

---

## 2026-07-27 — `nt_qmatvec_i8` covers Q8_0 (int8-activation matvec for the Q8 shape)

`nt_qmatvec_i8` (`notorch.c:5338`) accepted only Q4_0, so every Q8_0 decoder fell
through to the exact per-block dot. Q8_0 is the cheaper case for this path: the block
is a f16 scale followed by 32 raw int8 weights, so no nibble unpacking is needed —
activation and weight meet as int8 and accumulate in int32.

- `nt_q8_0_rows_i8` (`notorch.c:5284` NEON dot-product / `notorch.c:5310` scalar):
  per row, per 32-block, two `vdotq_s32` over `vld1q_s8` halves, result scaled by
  `d_w * d_a`. Scalar branch is the same arithmetic in a plain int loop.
- dispatcher guard widened from `dtype != 2` to `(dtype != 2 && dtype != 8)`; the
  `k % 32` requirement is unchanged.

Proof (neo, Accelerate, real tensors from a 500M SmolVLM2 Q8_0 GGUF, deterministic
activation, same input to both kernels; agreement measured against the exact
`nt_qmatvec`):

| tensor | shape | rel L2 vs exact | speedup |
|---|---|---|---|
| `blk.0.attn_q.weight` | [960,960] | 0.0027 | 4.33× |
| `blk.0.ffn_gate.weight` | [960,2560] | 0.0036 | 21.23× |
| `blk.15.ffn_down.weight` | [2560,960] | 0.0038 | 21.15× |

Both branches verified: built with dot-product intrinsics and again with `-march=armv8-a`
(scalar path) — identical results to the last digit (`max|diff| 0.033519`,
`rel_L2 0.003554` on `blk.0.ffn_gate.weight`). Guard checked from the other side: a
Q4_K tensor (`dtype=12`) returns -1 while the exact kernel returns 0. The kernel stays
documented as approximate; `nt_qmatvec` remains the exact reference.

---

## 2026-07-20 — close the alloc-overflow residual: nt_conv2d geometry guard + resonance size_t

The `nt_tensor_new` root pass named two same-class residuals; both closed here.

- `nt_conv2d` (`notorch.c:5386`): `K = Cin*kH*kW` and `N = Hout*Wout` are matmul
  dims for `nt_blas_mm` and must stay `int`, so they cannot simply be widened. The
  products are computed in `long` and rejected if either exceeds `INT_MAX` before the
  truncation, so a wrapped int32 can no longer mis-size the `(size_t)K*N` im2col
  buffer. `<limits.h>` added for `INT_MAX`.
- `examples/train_resonance_lora.c:87`: two-step `len = max_T*H*D` widened to `size_t`
  (its only use is the two `nt_tensor_new(len)` allocations).

Proof (neo, Accelerate): `make test` → notorch_test 49/49, test_vision 73/73; a
standalone guard check drives `nt_conv2d` with `Cin*kH*kW = 2.2e9 > INT_MAX` — returns
-1 before allocating — while a 1×3×3 / 2×2 conv returns the correct `[6,8,12,14]`; the
resonance translation unit compiles clean under `-Wall -Wextra`.

## 2026-07-20 — integer-overflow hardening: `nt_tensor_new` length widened to `size_t` (root of the alloc-size overflow class)

Two passes closed the `int * int` overflow-before-widening class flagged by CodeQL
(`cpp/integer-multiplication-cast-to-long`, threat model `remote`).

**Leaf pass** (PR #24, `9c41b39`) — 64 flagged size expressions where a product of
`int`s overflows before the implicit widen to `size_t` at `malloc`/`calloc`/`memcpy`/
`memset`. Each casts its leading operand to `size_t` so the product computes wide:
`notorch.c` 24, `examples/infer_janus.c` 28, `tests/test_rrpram_broadcast.c` 6,
`notorch_vision.h` 4, `stb_image.h` 2 (the vendored 16-bit convert path, which lacked
the overflow guard its 8-bit sibling gets from `stbi__malloc_mad3`).

**Root pass** (this change) — the leaf casts don't help a caller that hands `nt_tensor_new`
an already-truncated `int` product, because the length parameter itself was `int`. Widened
the constructor family so the guard sees the true product:

- `nt_tensor_new(int len)` → `nt_tensor_new(size_t len)` (`notorch.h:42`, `notorch.c:152`).
  Guard `len <= 0` → `len == 0` (unsigned); a negative-int caller now converts to a huge
  `size_t` and is still rejected by the `> NT_MAX_ELEMENTS` (`1<<28`) upper bound — same
  NULL result, no under-alloc path. `t->len`/`t->shape[0]` take `(int)len`, lossless after
  the guard (≤ 268435456 < INT_MAX).
- `nt_tensor_new2d`: `int total = rows*cols` → `size_t total = (size_t)rows * cols`
  (`notorch.c:168`); `nt_tensor_new_shape`: `size_t total` accumulated with `(size_t)shape[i]`
  (`notorch.c:182`). The overflow that previously slipped past `total > NT_MAX_ELEMENTS` is
  now caught.
- 37 product call sites cast to `(size_t)` (14 in `notorch.c`, plus `examples/train_distillation.c`,
  `tests/test_notorch.c` ×12, `tests/test_rrpram_broadcast.c` ×6, `tools/leak_repro.c` ×3, and the
  two-step `nt_image_to_tensor` in `notorch_vision.h:226` which also gained a NULL-alloc guard).
  Pure integer-literal products (`nt_tensor_new(3 * 6)`) are left as-is — compile-time constants.

Proof (neo, Accelerate): `make test` → `notorch_test` 49/49, `test_vision` 73/73. A boundary
harness drives all three constructors with `100000 * 42950` (= 4.295e9, which the old `int`
math wrapped to 32704 and passed) — all now return NULL; ordinary shapes still allocate with
correct `len`; `NT_MAX_ELEMENTS+1` and zero are rejected. Two independent Opus audits: casts
correct and behavior-preserving, no flagged or two-step site missed inside the library.

Known same-class residual, outside the `nt_tensor_new` root and tracked separately: `nt_conv2d`
im2col `K = Cin*kH*kW` / `N = Hout*Wout` (`notorch.c:5372`) form `int` products before the
`size_t` malloc (needs geometry-validation guards, since K/N must stay `int` for `nt_blas_mm`);
and `examples/train_resonance_lora.c:94` two-step `len = max_T*H*D` (config-bounded consumer).

## 2026-07-15 — Codex audit: JS/C op-contract + fresh-op fail-fast guards

Targeted Codex audit after the JS edition was brought up to C op 36. Two bug
classes were closed around the fresh surface:
- `js-edition/notorch.js`: `RELU` now records canonical C op 35 instead of the
  old JS-local 105. The README had promised full C op parity through 36 while
  RELU still violated the numeric contract.
- `nt_seq_gate` / `seqGate`: reject invalid `T/nm/gi`, non-divisible `x.len`,
  and gate-length mismatches before reading.
- `nt_rrpram_broadcast_attention` / `rrpramBroadcastAttention`: reject invalid
  dims, short `x`/`v`, and malformed packed `Wr` before deriving strides.

Added `js-edition/test_op_parity.mjs` plus `make test_js` / `npm test` for the
lightweight JS/C op-contract gate. C regressions now cover invalid `seq_gate`
inputs and invalid broadcast-RRPRAM shapes.

Proof (Codex, local): `make test_js` → `JS_OP_PARITY_OK`; `npm test` in
`js-edition` → `JS_OP_PARITY_OK`; `make test` → `notorch_test` 49/49 and
`test_vision` 73/73; standalone `tests/test_rrpram_broadcast.c` adversarial
binary PASS including invalid shape checks; `git diff --check` clean.

## 2026-07-15 — JS edition: op 34 RRPRAM_BCAST + op 36 SEQ_GATE ported (tri-version parity)

The JS edition (`js-edition/notorch.js`) had stalled at op 33 while the C canon advanced
to op 36 (RRPRAM_BCAST 2026-06-16, RELU 2026-06-27, SEQ_GATE 2026-06-28). RELU was already
present under JS-local opcode 105; the two genuinely-missing ops are now ported, closing
C↔JS op parity at the full 0–36 set.

Ported (forward + backward, 1:1 with C semantics):
- `seqGate(x, g, T, nm, gi)` — op 36 SEQ_GATE, per-position mechanism gate
  `out[t,d] = x[t,d]·g[t,gi]`. Mirrors C `nt_seq_gate` (notorch.c:3383 fwd / :762 bwd).
- `rrpramBroadcastAttention(wr, x, v, T, E, nH, hD, rank)` — op 34 RRPRAM_BCAST, canonical
  Janus broadcast pattern: `mid[h,r] = Σ_t Σ_e x·Wr_a` (one mid per head, broadcast across
  queries), causal-softmax scores scaled `1/√hD`. Mirrors C `nt_rrpram_broadcast_attention`
  (notorch.c:3796 fwd / :1578 bwd). `rank` is passed explicitly (ctx ≥ T ⇒ not derivable
  from `Wr.len`). OP table gains `RRPRAM_BCAST:34`, `SEQ_GATE:36`.

Proof (neo, node v25.9.0): a C emitter (`parity_emit.c`, built against canonical
`notorch.c`) and a JS runner ran both ops fwd+bwd on identical hardcoded inputs
(dout = all-ones, loss = Σ out) and diffed —
- op 36 SEQ_GATE — bit-identical to C: `SG_OUT/DX/DG` maxAbs = 0.0.
- op 34 RRPRAM_BCAST — within float32 rounding: `RB_OUT` 1.5e-8, `RB_DWR` 2.3e-10,
  `RB_DX` 1.5e-11, `RB_DV` 6.0e-8 (float32-vs-float64 intermediate accumulation).
- Independent JS finite-difference grad-check (ε=1e-3): both ops, all groups `fails=0`.

`notorch_test` full suite 49/49 on neo (Accelerate). The Termux edition (a platform
recipe + demo that builds `../../notorch.c` directly — no core fork) was rebuilt and
generated against the current canon: parity by construction. README parity table
(`js-edition/README.md`) and main-README op-count synced (37 ops, IDs 0–36); js caveats
corrected (op parity through 36). Tree hygiene: `.gitignore` hardened so compiled
test/example binaries and editor state stay out of `git status`.

## 2026-07-07 — gguf.c: harden parser error-paths (F-1 NULL-deref + latent data_size wrap)

An error-path hardening pass on the GGUF parser (untrusted-binary surface). One real
NULL-deref plus four fail-loud gaps; the successful-load path is byte-unchanged.

Fixed:
- `gguf_read_str_array` (F-1): a crafted type-9 string array with a huge `alen` drove
  `calloc(alen*8)` → NULL → `result[j]=strdup` NULL-deref. Now capped at
  `GGUF_MAX_STR_ARRAY` (2M, gguf.h), `calloc`/`strdup` NULL-checked, `*out_n` reports
  the actually-read count `j`, not the claimed `alen`.
- `gguf_open` data section: `data_size = fsize - data_offset` wrapped to a huge unsigned
  on a file truncated before the data section (latent bug). Guarded now
  (`fsize<0 || data_offset>fsize` → fail-loud), and the tensor-data `fread` is checked
  for short read (frees `gf->data`+`gf`).
- `gguf_open` header: `version/n_tensors/n_kv` reads checked, fail-loud on truncation.
- `read_string`: huge-len discard loop breaks on EOF (no billion-iteration spin).
- `gguf_dequant`: `dst = malloc(n*sizeof)` → `calloc(n, sizeof)` — C11 overflow-safe,
  zeroes the tail when `n_elements` isn't block-aligned.

Proof (Neo): compiles `-Wall -Wextra` zero-warning; `make` builds libnotorch.a; mini.gguf
(janus, 31 tensors) and nanollama (llama, 120 tensors, 32000 tokens) load identically;
crafted `alen`=4G → fail-loud, no segfault (`gguf_craft_test`). Independently verified by
Codex (CLEAN) and an Opus subagent audit (CLEAN — all six hunks + the `examples/bpe.c`
consumer `out_n` contract).

## 2026-06-28 — nt_seq_gate: per-position mechanism gate (op 36)

Added `nt_seq_gate(x_idx, g_idx, T, nm, gi)` — `out[t,d] = x[t,d] * gate[t*nm+gi]`, the
per-position scalar-over-block multiply PostGPT-Q's triple attention needs to gate each
mechanism (Content / RRPRAM / Janus) by its own learned sigmoid before the concat. `x`
is `[T, B]` (B = x.len/T), `gate` is `[T, nm]`, `gi` selects the gate column. Backward
flows to `x` (`dout*gate`) and to gate column `gi` (`Σ_d dout[t,d]*x[t,d]`); mirrors
`NT_OP_MUL` plus a reduction. This lifted PostGPT-Q's training loop off PyTorch onto the
notorch tape (Operation Napalm-2 — github.com/ariannamethod/q). Proof: `make` clean
(pre-existing unused-symbol warnings only), `./notorch_test` 49/49 passed, 0 failed
(48 + `test_seq_gate`, which checks the gated values and that grads reach both `x` and
the gate).

## 2026-06-27 — nt_relu: plain ReLU activation (op 35)

Added `nt_relu(int x_idx)` — `y = max(0, x)` forward, `dy/dx = (y > 0) ? 1 : 0`
backward (`NT_OP_RELU`, op 35). notorch carried silu / gelu / sigmoid / geglu /
swiglu but no plain ReLU; PostGPT's MLP (`F.relu`) needed it to lift its training
loop off PyTorch onto the notorch tape (Operation Napalm-2 —
github.com/ariannamethod/postgpt). Forward mirrors `nt_sigmoid`; backward mirrors
the SIGMOID case (reads `e->output`, since `y > 0 ⟺ x > 0`). Proof: `make` clean
(only the pre-existing unused-symbol warnings), `./notorch_test` 48/48 passed,
0 failed (47 + `test_relu`, which checks relu(-1)=0 / relu(0)=0 / relu(2)=2).

## 2026-06-19 — Metal: nt_metal_rope gains norm_pairs (arch-gated rope)

`nt_metal_rope` now takes a `norm_pairs` flag: 0 keeps the half-split pairs
`(i, i+hd/2)`, 1 uses consecutive NORM pairs `(2i, 2i+1)`. The Metal
`rope_f32` kernel branches on it (extra `buffer(5)` constant). This makes the
Metal rope arch-aware: llama-arch GGUFs, which the HF→GGUF converter lays out
for interleaved/NORM rope, decode correctly with `norm_pairs=1`, while
mistral3 keeps the existing half-split path. It is the byte-identical upstream
of doe `b3e7a23`, where the arch gate was first validated on a live
Mistral-Nemo-12B forward (coherent output, old multi-byte salad gone) and a
24B mistral3 tok1 regression that stayed bit-identical (`'ĠI'=19.947`). On the
notorch side `nt_metal_rope` is consumed only by the Metal unit test;
`examples/infer_gguf_metal.c` already selects `rope_neox`/`rope_interleaved`
on the CPU path. Proof: `make metal` 0 errors, `test_metal_rope
max_rel=1.313e-05 PASS` (`norm_pairs=0` half-split bit-matches the CPU
reference), all Metal gates green on Apple Silicon A18.

## 2026-06-16 — op 34 nt_rrpram_broadcast_attention implemented (closes a standing TODO)

NT_OP_RRPRAM_BCAST (34) was declared in notorch.h with no C implementation —
the op was unusable from C and the JS port stalled at op 33. Implemented the
canonical Janus broadcast pattern (mid[h,r] = Σ_t x[t]·Wr_a[h], sc=1/sqrt(D))
with full forward + backward, plus a 348-line adversarial test. Verified:
sentinel forward bit-exact (max_diff=0), backward finite-diff (d_wr/d_x/d_v)
correct, suite 73/73 green. (PR #13.)

## 2026-06-16 — infer_llama: GGUF-embedded BPE tokenizer (CPU path was byte-level)

examples/infer_llama.c tokenized byte-level — each prompt byte fed as a token
id (infer_llama.c:327) and decoded as raw ASCII — so any real BPE-vocab GGUF
(SmolLM2, Qwen2.5, Mistral) got scrambled input and emitted token-number
garbage (`[9234][512]…`). The fix wires the gguf-native BPE that already
shipped in examples/bpe.c (bpe_load reads tokenizer.ggml.tokens/.merges
straight from the file; bpe_encode/bpe_decode_token) — the same tokenizer the
Metal inferer infer_gguf_metal.c already used; the CPU path simply never got
it. eos comes from tokenizer.ggml.eos_token_id; byte-level is kept as a guarded
fallback for char/byte-level models (nanollama) where bpe_load returns NULL.
Makefile `llama` target now links examples/bpe.c. Verified: BPE roundtrip
BPE_OK on the SmolLM2 vocab; SmolLM2-135M produces coherent English (was
`[9234][512]` garbage); notorch test suite 73/73 green.

## 2026-06-13 — Metal: naive matvec is the default; sg goes opt-in (NT_METAL_SG=1)

The authoritative A/B — live oyent-24B decode through doe on M4 Pro, one
binary, whole-run NT_METAL_NAIVE flag — measured the simdgroup kernels at
−23% vs naive (sg median 2.86 t/s vs naive 3.71; correctness gates green,
identity intact, pure speed). The square resident microbench win that made
sg the default in `09e76af` (×1.81, M=K=2048) does not transfer to the real
mixed-shape decode stream (280 matvecs/token, attn k/v down to 1024×5120),
and a phase-fair microbench rerun on neo A18 now agrees (sg solo 227.27
ms/sweep vs naive 155.85). Real-workload A/B outranks the microbench, so the
default follows it: `g_use_sg` starts at 0, `NT_METAL_SG=1` opts in for the
geometry-tuning round, `NT_METAL_NAIVE=1` still forces naive and wins over
both. Tests updated to match — the sg determinism/tolerance gates opt in
explicitly (the tolerance gate now also proves the default differs from sg
by reduction order, max_rel 3.6e-05 ≠ 0), bench phases A–C pin sg while
phase D measures the library default. 13 gates green, rc=0, −Wall −Wextra
clean.

Same day, the lesson became a harness: `bench_metal_batch doe` — doe-mix
mode with the real oyent-24B shapes (q/k/v/o + gate/up/down ×40 + lm_head;
Q6_K on v/down/lm_head), weight copies cycled per layer so every matvec
streams from DRAM (small attn matrices get 8 copies to defeat the SLC),
per-group sync isolation plus an honest full-speed sweep. First read on
neo A18 (naive default): time follows bytes — ffn 85.2% of time vs 88% of
bytes, no dispatch anomaly — and effective bandwidth is 26.6 GB/s, with
per-group spread 18.8 (attn qkv, small-m underoccupancy) to 34.0 (gate+up);
ffn down (Q6_K, m=5120 k=32768) is the worst byte-weighted offender at
19.7 GB/s. That is the target list for the kernel-geometry round.

And the harness paid for itself within the hour: the per-shape sg-vs-naive
A/B revealed the split is by FORMAT, not by geometry — the sg kernels win
on Q6_K everywhere measured (ffn down 188 vs 279 ms, lm_head 15.9 vs 25.7,
even at m=131072) and lose on Q4_K everywhere (gate+up 495 vs 222). So the
default became per-format auto: Q6_K rides sg, Q4_K rides naive
(`g_use_sg` tri-state; NT_METAL_SG=1 still forces all-sg, NT_METAL_NAIVE=1
all-naive and wins over both). doe-mix full-speed on neo A18: naive 574.6
ms/tok, all-sg 767.0, auto 449.7 = 33.0 GB/s effective — +24% over the
naive default from one selection rule, zero new kernels. Caveat for
consumers: Q6_K results now differ from the naive reference by reduction
order (within tolerance, run-to-run still bit-identical) — exact-equality
gates against CPU on Q6_K-fed paths become tolerance + argmax gates. The
custom-geometry round (multi-row simdgroup, scale-decode amortization,
Q4_K sg rework) stays open with gate+up as the next byte-weighted target.

The deploy machine then ruled on the default. A clean one-binary A/B on
M4 Pro (live oyent-24B decode through doe, short runs, disjoint ranges,
medians): all-naive 4.24 t/s > per-format auto 3.57 (-16%) > all-sg 3.24.
The per-format split is A18-tuned and does not transfer across Apple GPU
generations -- on M4 Pro the sg kernels lag even on Q6_K, while identity
stays exact (auto tok1 19.961 == CPU, argmax + determinism x2; pure
speed, zero correctness cost). So the library default is naive again and
the split is opt-in: NT_METAL_AUTO=1 enables per-format (the A18 win,
re-verified on neo same-binary: doe-mix full-speed 558.7 ms/tok auto vs
778.6 naive, auto run later and hotter), NT_METAL_SG=1 forces all-sg,
NT_METAL_NAIVE=1 forces naive and wins over both. The standing rule this
encodes: kernel defaults follow the deploy machine, and a per-GPU tuning
win ships as an env opt-in until the target machine confirms it. The
geometry round ahead (multi-row simdgroup, Q4_K rework) gates on M4 Pro
numbers, not A18.

## 2026-06-12 — Metal token-graph step 1: persistent arenas + batched dispatch (with Q6_K landing the same day)

Two commits, two nodes, one front. `dd1779f` (metal node): `nt_metal_q6k_matvec` —
Q4_K_M GGUF stores attn_v/ffn_down/output as Q6_K, so the GPU path needed the
second kernel to keep lm_head/FFN-down off the CPU; verified bit-identical vs CPU
on live oyent-24B weights (lm_head m=131072, max_rel < 2e-5), ~2.5x decode.
cb.status guard after every waitUntilCompleted (a silent GPU fault is now loud) +
a run-to-run determinism gate in the test.

`bbb29e5` (neo, branch `feat/metal-token-graph`): the dispatch structure. The
Metal path was a matvec accelerator bolted onto a CPU loop — every call allocated
fresh x/out/k buffers and paid a full commit+waitUntilCompleted (~280 syncs per
24B token; profile shows matvec = 95% of decode). Step 1: persistent in/out
arenas (bump-allocated, 256-aligned) kill the per-call buffer churn; k rides
setBytes; `nt_metal_batch_begin/commit` encodes independent matvecs ({q,k,v},
{gate,up}, a whole layer sweep) into ONE command buffer with ONE sync. Kernels
and dispatch geometry untouched — batched results are bit-identical to solo
calls, and the q4k correctness numbers are bit-identical to the pre-change
baseline (max_rel=2.124e-05, same worst idx). New gates: q6k correctness vs the
gguf.c reference dequant (max_rel=1.267e-05), q4k/q6k 2x-run determinism,
batch-vs-solo memcmp. `tests/bench_metal_batch.c` isolates the sync cost on
resident weights: neo A18, 280 matvecs/sweep — solo 280 syncs vs 40 per-layer
batches = x1.6-2.2 wall-clock. Next: doe wires the {q,k,v}/{gate,up} groups,
then layer-resident ops (rmsnorm/rope/silu/attention in MSL) toward the
one-command-buffer-per-token shape — the llama.cpp-class decode (16.8 t/s on
M4 Pro vs our 3.66 today) with our bit-identical gate discipline at every step.


### Addendum, same day — M3: simdgroup-cooperative kernels (default path)

`q4k_matvec_sg` / `q6k_matvec_sg`: one 32-lane simdgroup per output row, lanes
split WITHIN each 256-weight block (8 weights/lane — full utilization at any k,
coalesced reads), simd_sum folds the partials; dispatch grid (32,m), threadgroup
(32,8). Default path; `NT_METAL_NAIVE=1` keeps the one-thread-per-row reference
kernels for A/B (never deleted). Determinism: fixed simd_sum tree → bit-identical
run-to-run (gated); vs naive the reduction order differs → tolerance gate
(q4k 3.6e-05, q6k 1.6e-05 max_rel, both PASS). Phase-fair bench on neo A18
(all-naive run vs all-sg run): solo sweep 168.50 → 93.27 ms (x1.81), per-layer
batch 102.84 → 73.58 ms (x1.40); best observed warmed config (sg + 40 batches)
41.63 ms/sweep vs the 163.68 ms starting point. A18 microbench is noisy — the
authoritative numbers come from the 24B on M4 (doe re-runs t/s + verify after
pull). M4-the-milestone (rmsnorm/rope/silu/attention in MSL) remains next.


### Addendum 2, same day — M4: layer ops in MSL + device-resident slots

The other half of the 50/50 profile (CPU attention/rmsnorm/silu/sample between
GPU matvecs). Six kernels — `rmsnorm_f32` (single-threadgroup, fixed reduction
ladder), `rope_f32` (llama-style pairs, in place), `silu_mul_f32`, `add_f32`,
`attn_decode_f32` (one threadgroup per q-head, GQA, softmax in threadgroup
memory, t_len <= 4096), `copy_f32` (KV append GPU-side) — plus the architecture
that makes them chain: SLOTS, device-resident activations in a persistent GPU
arena. Ops read/write slots with no host crossing, so a whole decode layer
(rmsnorm -> qkv -> rope -> attn -> o -> residual -> rmsnorm -> gate/up ->
silu*mul -> down -> residual) encodes inside ONE command buffer between
batch_begin/commit. New API: nt_metal_register_region (appends KV cache and
friends to the registered segments — base and length must be PAGE-aligned;
note getpagesize() is 16384 on Apple Silicon), slot_alloc/upload/download,
slot-resident matvec variants, and the ops above. Gates (neo A18, all green):
rmsnorm exact-0 vs CPU ref, rope 1.3e-05, silu_mul 2.2e-07, add exact-0,
attn_decode 3.6e-06 vs double-precision CPU softmax-attention, 3-op chain
batched bit-identical to solo. Integration into doe (layer graph on slots,
KV registered, one sync per token) is the next wiring step on the metal node.

## 2026-06-09 — SD op set on notorch: conv2d + group norm + upsample + attention (forward)

Added to `notorch.c` (declared in `notorch.h`) — the image-NN ops notorch lacked, forward-only,
companions to `nt_qmatvec` (pre-trained weights, no tape). After this notorch carries the full
Stable-Diffusion building-block set (conv2d · group_norm · silu · gelu · layernorm · softmax · GEMM · upsample · attention).

- **`nt_conv2d`** (+ `nt_im2col`) = zero-padded unfold → a single `nt_blas_mm` GEMM (weight `[Cout, Cin·kH·kW]` @ col `[K, Hout·Wout]`) → optional per-channel bias.
- **`nt_group_norm`** = per-group mean/var over `(C/num_groups)·H·W` → normalize → per-channel affine (`gamma`/`beta` nullable). Portable plain-C (no vDSP); `out` may alias `in`.
- **`nt_upsample_nearest`** = nearest-neighbour `[C,H,W] → [C,H·scale,W·scale]` for the UNet/VAE up-blocks.
- **`nt_attention`** = single-head scaled dot-product `softmax(Q@Kᵀ/√d)@V` via `nt_blas_mmT` + inline softmax + `nt_blas_mm`. Self-attn (S=T) and **cross-attn** (S=context — the diffusion conditioning path).

Motivation: yent.yo's BK-SDM diffusion runs on ONNX Runtime because notorch had no conv/attention image ops —
this is the op foundation for running it on notorch instead. Reference: yent.yo's `accel.c`, ported portable.
Tests in `tests/test_vision.c` (conv2d 3×3 → [12,16,24,28] + bias; group_norm 2-group {−1,+1} + 1-group affine;
nearest upsample 2×; self- and cross-attention vs hand-computed softmax): **test_vision 73/73, notorch_test 47/47.**

Remaining for a full BK-SDM on notorch (a larger model-port follow-up): the UNet/VAE graph, the scheduler,
and weight loading from the ONNX/safetensors checkpoint. The ops are now in place.

## 2026-06-07 — Phase 2: gated multi-thread fan-out + int8 dynamic-activation-quant matvec (Q4_0, 22.9×)

Two speed paths layered onto `nt_qmatvec`, branch `feat/nt-qmatvec-threaded`.

**(2a) fn-dispatch + gated multi-thread.** `nt_qmatvec` is now a function-pointer dispatch (`nt_qrows_for`)
over per-dtype row kernels, plus a pthread row fan-out. Naive per-call fan-out turned out **counterproductive
for small single-token decode matvecs** — measured ~6%/noise on a 360M model: per-call `pthread_create` plus the
2P+4E asymmetry of Apple-Silicon CPUs eat the parallelism (even-split waits on the slow E-cores). So it is
**gated high (≥4M elements)**: only large matvecs (big models / batched) thread; small decode stays
single-thread. The fn-dispatch is clean groundwork the int8 kernels plug into. `Makefile` gains `-pthread`
(glibc-Linux linkage; no-op on macOS/Termux libc). Commit `9096051`.

**(2b) int8 dynamic-activation-quant matvec — `nt_qmatvec_i8`.** The llama.cpp/MNN fast path: quantize the
activation to per-32-block symmetric int8 once (`nt_quant_act_q8`: `d_a = amax/127`, `qa = round(x/d_a)`), then
dot it against the **packed** Q4_0 weights with INTEGER accumulation; per-block result scaled by `d_w·d_a`.
NEON **SDOT** (`vdotq_s32`, 4 int8-MAC/instr; `__ARM_FEATURE_DOTPROD`, default on Apple Silicon) with a scalar
`#else` fallback — weights unpacked to int8 in-register (`nibble−8`), dotted against the int8 activation,
horizontal-summed. **Measured single-thread on neo (A18 Pro), `tests/bench_qmatvec.c`: f32-dequant
1.794 ms/call → int8-dot 0.078 ms/call = 22.9×.** Same matvec result (rel 0.0028 vs the exact f32 reference):
int8 activation quant is **APPROXIMATE**, so `nt_qmatvec` (f32 dequant) stays the exact path and `nt_qmatvec_i8`
is an opt-in fast path. `notorch_test` 47/47. Commits `71eb92d` (scalar) / `bf87651` (NEON SDOT).

Kernel-level numbers. NEXT: wire `nt_qmatvec_i8` end-to-end into the runners (WTForacle Q4_0), extend to
Q8_0 / K-quants, add x86 AVX-VNNI, then merge Phase 2 to main.

## 2026-06-06 — nt_qmatvec: agnostic packed quantized CPU matvec (Q4_0/Q5_0/Q8_0/Q4_K/Q6_K)

The CPU/BLAS/SIMD inference path dequantized every GGUF tensor to dense f32 (×6-8 RAM) before
`cblas_sgemv` — only the Apple-Metal path (`nt_metal_q4k_matvec`) and a single example-local
`q6k_rows` inside `examples/infer_gguf_metal.c` kept weights packed. notorch now has a library
primitive, `nt_qmatvec(out, Wq, dtype, x, m, k)` (`notorch.c`, decl `notorch.h`), that keeps the
weights packed in RAM and dequantizes each block inline in registers — the same math as
`gguf_dequant → nt_blas_matvec`, a fraction of the memory and weight bandwidth. It dispatches by
GGUF dtype over the full set: F32, F16, Q4_0, Q5_0, Q8_0 (block-of-32), Q4_K, Q6_K
(super-block-256); the Q6_K kernel is the proven `q6k_rows` lifted out of the example into the
library, and F16 alone halves the weight RAM vs dense f32 (converted per element, never
materialized). **Verified** by a new `tests/test_qmatvec` against the dequant→cblas oracle: all
seven dtypes agree to relative error ~1e-6 (f32 summation-order noise, not unpack error);
`notorch_test` stays 47/47. This is the foundation of an agnostic packed CPU inference path — the CPU no longer
has to blow Q4_0/Q8_0 up to f32. Phase 1 is single-threaded and correctness-first: the RAM win lands
when a runner stops calling `gguf_dequant` and rides `nt_qmatvec` directly, and the speed path
(pthread rows + MNN/llama.cpp-style int8 activation-quant with SDOT/VNNI integer dot) is next.
Branch `feat/nt-qmatvec-packed`, commits `8687137` / `5bc1b84` / `59901df`.

## 2026-06-06 — JS edition: full GGUF RUN (tokenizer + forward + generate), matches C

After the dequant-load landed, `js-edition/infer_gguf.mjs` runs a GGUF end-to-end in pure
JS: a byte-level BPE built **from the GGUF** (mirror of `examples/bpe.c`) + the llama/mistral
forward on notorch.js tape ops (embed / RMSNorm / q-k-v / interleaved-RoPE / GQA-attn /
SwiGLU FFN / tied output) + greedy generate. **Verified vs the C engine:** SmolLM2-135M-Q4_K_M
greedy produces *"The capital of France is Paris. Paris is a city"* — **token-for-token
identical** to `examples/infer_gguf_metal`. The JS edition now loads AND runs real quantized
models with no Python and no llama.cpp. CPU path today; packed/WebGPU quant matvec and the
qwen3 NEOX + per-head q/k-norm arch are the next steps.

## 2026-06-06 — JS edition: GGUF quantized dequant + C-parity test

`js-edition/notorch.js` `loadGGUF` threw on every quantized tensor (F16/F32 only) while the
JS README claimed "F16 + F32 dequant" — a prophetic debt. Ported the five GGML block-dequant
routines from `gguf.c` **byte-for-byte** (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K) into `loadGGUF`; a
real quantized GGUF now loads in browser/Node. **Verified** against the C path with a new test
— `tests/gguf_dequant_ref.c` dumps C `gguf_dequant` values, `js-edition/test_gguf_dequant.mjs`
compares: Q4_K/Q6_K/Q8_0/Q4_0 match C to **~5e-9** across Qwen3-0.6B, smallcoder-Q8_0,
wtf360-Q4_0 → `JS_DEQUANT_OK`. Q5_0 is mirrored from `gguf.c` but had no local Q5_0 file to run
against. Added `js-edition/package.json` (`type:module`) so Node imports the ESM. JS README
corrected to the true state. Open next: a packed / WebGPU quant matvec so big models don't
expand to f32 in-browser.

## 2026-06-05 — README rework: inference is first-class; models split refs vs organisms

The README sold notorch as a training framework; it is training AND inference. Added
an `## inference` section — the packed-Q4_K/Q6_K Metal path (`examples/infer_gguf_metal.c`,
new `make infer_gguf_metal` target, Darwin + non-Darwin guard), the engine matrix, and the
measured oyent-24B numbers (Mistral-Small-24B Q4_K_M on a 24 GB Mac: 0 swap, 10.6 GB,
~1.4 tok/s). Made Apple-Silicon/Metal consistent across the build matrix, dependencies, and
the platform table (it used to appear, then vanish). `what is this` now says trains **and** runs.

Restructured the model list into exactly two sections — **references** (Karpathy ports +
from-scratch notorch models + how-to-train, with the Resonance-200M 3.52→0.59 and
nanollama-88.6M proofs) and **organisms that run on notorch** (appendix). Removed neovlm
(now private) and janus.sonar (too experimental); microgpt-1bit relabeled honestly as the
pure-Python BitNet reference notorch's BitLinear was validated against (not a notorch build);
added nanollama-notorch + siblings. JS README's "F16+F32 dequant" line corrected — `loadGGUF`
throws on quant today; the block-dequant port is the open JS upgrade.

## 2026-06-05 — in-house SIMD (AVX2) matmul: kernel + cache-block pass

A measurement-driven optimization pass on `notorch_simd.h` (the zero-dependency
AVX2 cblas shim), benchmarked against Intel MKL + OpenBLAS on the i5-8500T
(6c no-SMT, perf governor, 7-run medians). Correctness held bit-identical
throughout (`test_simd_loss` = 10.379384 vs the OpenBLAS path).

- **MR-interleaved A packing** (`42eef01`) — the 6×16 micro-kernel read A
  strided by k (6 cache lines per k-step); pack A `[Kc][MR]` so the 6 values
  for one k-step are contiguous. +~20% on NN-forward.
- **4× k-unroll + aligned B loads** (`8b98a6c`) — hoist the per-iteration
  prefetch branch, `_mm256_load_ps` (B_pack is 64-byte aligned). TN
  weight-grad shapes reached MKL parity (Llama dWffn 321 vs MKL 329 GFLOP/s).
- **Re-block Kc=128/Nc=256** (`1db4bf8`) — the Kc=256/Nc=1024 B-panel (1MB)
  spilled to shared L3, so 6 cores contended L3 bandwidth; Kc=128/Nc=256 keeps
  the ~128KB B-panel in private L2. +5–12% on NN-forward at 6T. `#ifndef`
  guards make MC/KC/NC `-D`-overridable per target.

**Honest result:** single-thread the kernel is ~0.82× MKL; TN weight-grad is
at MKL parity; NN-forward stays ~0.5× MKL. The residual gap is multi-core
cache-residency (MKL scales 4×/6c, this 2×/6c) — disproved as kernel, B-pack
(shared-B trial reverted), or malloc (persistent-buffer trial reverted); it is
shared-L3 bandwidth, the deepest machine-specific part of a tuned BLAS. Not
claiming MKL parity on forward GEMM.

## 2026-06-05 — packed-Q4_K + packed-Q6_K GGUF inference on Apple Metal

New `examples/infer_gguf_metal.c` — end-to-end notorch-C inference that keeps
quantized weights **packed** and never materializes the full f32 tensor:
- Q4_K → `nt_metal_q4k_matvec` (Metal, `53f38f2`).
- Q6_K → new CPU per-row dequant matvec (mirrors `gguf.c:dequant_q6_k`), no f32
  buffer. This is what lets a 24B model fit a 24 GB Mac.
- byte-level BPE (`examples/bpe.{c,h}`) reads the tokenizer from the GGUF via new
  `gguf_read_str_array` (gguf.c — `gguf_open` skips array-typed KVs).
- one forward, two RoPE conventions auto-detected: llama/mistral interleaved
  (weights pre-permuted by convert) and qwen2/qwen3 NEOX + per-head q/k-norm.

**Why packed-Q6_K matters — measured on metal (Mac Mini M4 Pro, 24 GB), oyent
(Mistral-Small-24B) Q4_K_M, greedy, `/usr/bin/time -l`:**
- first cut, Q6_K→f32 at load: RSS 7.4 GB + **12.4 GB swap**, load 58.5 s — thrashes.
- packed Q6_K (this pass): **swaps=0**, peak RSS 16.3 GB / footprint 17.3 GB,
  load 3.63 s, coherent+correct → "The capital of France is Paris, and its
  administrative center is the".

Speed is now **compute-bound, not memory-bound**. First the Q6_K per-row CPU
dequant (output 131072×5120 + ~20 ffn_down) dominated at 0.2 t/s; threading that
matvec across cores (work-gated, 12 cores on M4 Pro, disjoint y rows) lifted
oyent-24B to **0.6 t/s** (decode 8 tok 13.2 s, total 66 s → 28.5 s, swaps still 0,
peak 17.3 GB, same correct output). Then the **Metal Q4_K Phase-1 per-call weight
upload** (240 dispatches/token) dominated.

**Phase-2 (resident weights) landed.** `gguf.c` now page-aligns the tensor block
(`posix_memalign`) and records `data_size`; `nt_metal_register_base` wraps it as
zero-copy `newBufferWithBytesNoCopy` MTLBuffer(s) — **segmented**, because one
buffer is capped at `device.maxBufferLength` (14.302 GB on M4 Pro, just under the
14.326 GB block); `nt_metal_q4k_matvec` binds each weight by offset, no per-call
upload (weights straddling a segment edge fall back to upload). Result on oyent-24B:
**0.6 → 1.4 t/s** (0.2 → 1.4 over the whole pass, ~7×), total 28.5 s → 14.4 s,
**RSS 16.3 → 10.6 GB** (zero-copy, weights not duplicated), swaps 0, same correct
output. Llama-3.2-3B on neo (A18 Pro): **0.1 → 1.2 t/s** (~12×). Remaining lift:
optional Q6_K Metal matvec + a tiled/simdgroup Q4_K kernel.

Correctness regression (neo): Qwen3-0.6B-Q4_K greedy still "...Paris..." after the
Q6_K-path change (it uses Q6_K tensors); Llama-3.2-3B-Q4_K greedy 5/5 capitals.

## 2026-06-03 — GPU launch-bound pass: host-sync storm killed

A CUDA-backend performance pass — the bottleneck was launch/sync overhead,
not FLOPs. Six commits (`c1b655a..eaae961`):
- **L1** (`38d6b1a`) — batch per-param grad-norm readback into one D2H
  transfer instead of one sync per parameter; kills the host-sync storm.
- **L2** (`bc02d83`) — wire GPU backward for `NT_OP_MUL` + `NT_OP_SILU`,
  removing mid-backward device→host stalls (those ops now backward on GPU
  instead of bouncing to CPU).
- **L5** (`66f3c0f`) — widen the single-thread softmax / cross-entropy
  kernels to block-parallel.
- **op-33 RRPRAM** (`c1b655a`) — collapse the per-head GEMM loop into a
  cuBLAS strided-batched call.
- (`976d088`) — forward-declare the batched helpers used by the forward
  kernel.

Merged in `eaae961`. `notorch.c` + `notorch_cuda.cu` only; CPU path unchanged.

## 2026-06-02 — sigmoid / scale-by-t GPU sync (CPU-mirror bug class)

`nt_sigmoid` + `nt_scale_by_t` forward & `NT_OP_SCALE_BY_T` backward
joined the GPU/CPU mirror discipline. Surfaced by the molequla Inc2
RRPRAM-gate review: a learnable sigmoid gate sat frozen at sigmoid(0) on
GPU because the CPU backward branch read the stale CPU mirror without
`nt_tensor_sync_cpu(parent->output)`. Fixed forward + backward. With this,
the `NT_OP_*` backward CPU-branch audit for the sync pattern is **complete
— no known remaining candidates**.

## 2026-05-14 — nanollama 89M post-SFT (Arianna)

See `docs/POST_SFT_NANOLLAMA_ARIANNA_2026_05_14.md`.

## 2026-05-11 — Arianna LoRA SFT through notorch + MUL/SILU backward fix

`8ab5062` — `NT_OP_MUL` / `NT_OP_SILU` backward CPU-sync. Proved Chuck
holds at production scale once backward is correct; earlier "Chuck
destabilizes on LoRA scale" notes were downstream of this backward bug.
First production SFT (Resonance 200M Arianna LoRA) landed clean. See
`docs/POST_SFT_RESONANCE_ARIANNA_2026_05_11.md`.

## 2026-05-10 — GPU buffer-leak thread closed

The `ptr_map full — buffer leak` warning was a symptom of upstream tape
ref-accounting at high tensor counts, not a real leak. `3d46007` raised
`GPU_PTR_MAP_SIZE` 8K → 64K and fixed the CE sync; the warning hasn't
reappeared at realistic scales. Full thread:
`docs/GPU_BUFFER_LEAK_HYPOTHESIS_2026_05_10.md` →
`docs/GPU_BUFFER_LEAK_RESOLUTION_2026_05_10.md`. Also see
`docs/GPU_BACKWARD_SEGFAULT_T32_V512_2026_05_10.md`.

## 2026-05-09 — first GPU/CPU mirror bug found and fixed

`3d46007` — `nt_seq_cross_entropy_masked` (Defender). Established the
load-bearing rule: any CPU backward branch reading `parent->output->data`
directly must `nt_tensor_sync_cpu(parent->output)` first when GPU mode can
be on, or it reads the calloc-zero CPU mirror and computes zeros. The
bug-class registry lives in CLAUDE.md «Bug patterns».

---

## Open (carried from CLAUDE.md TODO)

- `gpu_rrpram_lr_forward` `Wrb_h` stride uses current T instead of T_max
  (`notorch_cuda.cu:824`). Workaround: train at T = T_max only. Real fix
  pending.
- `notorch.h:653` alpha-format docstring is stale — `nt_lora_save` writes
  raw IEEE-754 `float32` bytes, not `alpha*1000`. Fix on next pass-through.
- `nt_rrpram_broadcast_attention` (`NT_OP_RRPRAM_BCAST` 34) declared in
  `notorch.h:126,442` but unimplemented in `notorch.c`. JS edition stops
  parity at op 33 awaiting it.
- `phase7_eval.py` — vary RNG seed per cell so the first sampled token
  isn't identical across same-prompt cells.
