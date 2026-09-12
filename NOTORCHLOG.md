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

## 2026-09-12 — the reference gate called a documented difference a defect

Qwen3 and the audit repairs met in one tree, and the meeting turned up something
neither had touched. `harness/test_reference.sh` on nano_arianna Q8_0 reports
`NOTORCH_REFERENCE_FAIL (3 of 3)` — every prompt DIVERGED. It does so at
`30ae5fa`, at `claude/qwen3`, and on the merge, so it belongs to none of them.

The reference is reading a different prompt. `llama-simple` echoes
`<s> def fibonacci(n):` and writes `: With::::::::::::::`; this tree, given the
same text, writes `the pattern of resonance that is not possible, but the field
that is the source`. The file declares no `add_bos_token`, the reference prepends
one anyway, and forcing the same token in front here degenerates this model in
the same way — `6coding the fibrous structure the fibrous structure`. That is the
09-10 entry, the one where following the reference's default cost this model its
voice, arriving back as a red gate.

So the gate now checks that first, reusing the signal `test_tokenizer.sh`
already prints, and skips with the reason instead of reporting three defects
that are not there.

**Unverified, and named rather than left to be discovered:** every GGUF on this
machine has an undeclared BOS, so nothing here proves the gate still exercises a
model that declares one. That check waits for a file that does.

Merged in the same commit: Qwen3, which turned out not to be a family — two
optional per-head norms and two lines in the rotation loop, in `arch_llama.c`
rather than a file of its own — and the coin-flip gate's re-anchoring rewrite.
Both are Defender's, both survive the new call contract unchanged: `q_norm` and
`k_norm` load through the optional path, so the strict loader added hours
earlier does not refuse a file that lacks them.

Gates on the merged tree: `NOTORCH_PARITY_OK (6 checks)`,
`NOTORCH_CONSUMER_OK (3 checks)`, `NOTORCH_REPEAT_OK (3 checks)`,
`NOTORCH_TOKENIZER_OK (8 checks)`, `JANUS_OK`, `RESONANCE_OK`,
`NOTORCH_REFERENCE_SKIPPED` on the only model this machine can offer it,
notorch_test 49/49 and 73/73, test_qmatmul 46/46. `test_quantize` still FAILs
Q8_0 at 2.081e-04 over 2.067e-04, unchanged since `cd659e8`.

---


## 2026-09-12 — the coin-flip gate was wrong twice before it was right

Yesterday's entry below describes a gate that re-anchors on the single word where this tree and
the reference part, and calls agreement from there a tie-break. Qwen3 broke it within an hour
of arriving, and the two repairs are worth more than the original.

**One word of help is not enough, because ties cluster.** Qwen3 0.6B parted twice on two
prompts of ten and stayed apart after one word, so the gate said DIVERGED for a model whose
arithmetic is sound. **And a longer anchor is not the fix either.** Handing back the reference's
whole answer and comparing what each writes next moved the count from one DIVERGED to three,
because a free run from any context collects ties of its own. Lengthening the rope does not
help when the rope is the problem.

The question that separates the two cases has to be asked in a way that cannot accumulate:
given identical context, does the **next token** agree? That is one argmax over one forward
pass. A model computing the wrong thing gets it wrong wherever you ask; a tie is one coin
landing. It is asked at three points along the reference's answer rather than one, since a
single position can itself be a tie, and two of three must agree.

The separation is not a matter of threshold. On Qwen3 0.6B with ten prompts:

    sound build          6 identical, 4 tie-break, 0 diverged   —  11 of 12 forced points agreed
    QK norm not loaded   0 identical, 0 tie-break, 10 diverged  —   0 of 30 forced points agreed

Across six models and ten prompts: 38 identical, 22 tie-break, 0 diverged, 62 of 66 forced
points agreed.

**What it cannot see, stated because it was measured.** Dividing rmsnorm by n-1 instead of n —
a 0.05 percent error at n=1024 — passes. Both builds report OK on SmolLM2, and the forced points
agree 9 of 9 under the defect. The older, noisier version of the gate did flag it, so this is a
trade and not an improvement in every direction: the new rule stops crying wolf and in exchange
stops seeing perturbations below the tie threshold. Something that small only moves which
near-ties fall which way, and catching it needs logits, which the reference does not print.

One more shape it now names rather than fails on. A reference answer that is the prompt and a
few newlines — OLMoE on a Cyrillic prompt — gives nothing to anchor on, and the gate says
INCONCLUSIVE, which is neither colour.

---

## 2026-09-12 — Qwen3, which turned out not to be a family

`general.architecture = qwen3` is Qwen2's shape plus two tensors. It drops the qkv bias Qwen2
carries, which was already optional in this tree, and adds an RMS norm over each head of q and
of k before the rotation — `attn_q_norm.weight` and `attn_k_norm.weight`, both `[head_dim]`,
both applied with the model's own epsilon. Before the rotation, not after: RoPE mixes lanes
within a head, so a norm taken afterwards is a different function.

So it lives in `arch_llama.c` behind two optional weights and two lines in the rotation loop.
`arch.h` asks whether a family can be added without editing `runtime.c`; the question before
that is whether it is a family at all, and a separate file would have been 280 lines copied to
hold two tensors. `qwen3moe` is a different answer — it routes its feed-forward to experts and
belongs beside olmoe.

One trap was already closed and worth recording as closed. Qwen3 0.6B has `head_count = 16` and
`embedding_length = 1024`, which divide to 64, while the file's `key_length` is 128 and
`attn_q.weight` is `[1024, 2048]`. Anything deriving head_dim as embed/n_heads gets this model
wrong. This tree reads it off the q tensor, has since before Qwen3 arrived, and the same
arithmetic is what makes Mistral Nemo work.

Qwen3 0.6B Q8_0, ten prompts against llama.cpp: 6 identical, 4 tie-break, 0 diverged, with the
forced next token agreeing at 11 of 12 points. Tokenizer identical on 8. Not loading the QK
norm turns all ten DIVERGED and every one of 30 forced points wrong, which is how you can tell
the two lines are doing the work and not decorating it.

Speed on four big cores, Q8_0, 0.6B: prefill 79.6 t/s, decode 26.3.

The same file now also takes SmolLM2 and DeepSeek-R1-Distill-Qwen-1.5B without a line of code —
both are this shape already, and both were verified rather than assumed.

---
## 2026-09-12 — the harness answers an audit: forward can refuse, and the archive stops lying

An independent audit at `781f135` returned eleven findings and deliberately
withheld its commands, so every one of them was a hypothesis here until a tool
said otherwise. Nine were reproduced and repaired, one was already fixed, and
one turned out to have a different cause than the one it was filed under. What
follows is the reproduction, not the report.

### The archive kept members the build had dropped

`ar rcs` adds and replaces; it never deletes. Take `arch_mamba.o` out of
`HARNESS_LIB_OBJ`, rebuild, and `ar t` still lists it with `_nt_arch_mamba`
still exported — a consumer links a family the build no longer contains and the
link test passes. Worse, and not in the report: without `Makefile` in the
prerequisites the archive is newer than every surviving object, so make answers
`Nothing to be done` and the stale member is never even given the chance to be
overwritten. Both archives are now removed before they are written and both
depend on the Makefile. Red hand: with `arch_mamba.o` dropped, `ar t` no longer
lists it and `nm` no longer exports it, from a list change alone.

This went first, ahead of everything else in the repair order, because every
later fix is verified through these archives.

### forward returned void, and had three ways to not do what it said

A scratch allocation could fail, two families stopped quietly at the end of the
cache while three ran past it, and any caller could hand in a token id the model
has no row for. In all of them the caller got its buffer back unchanged and no
way to know.

`nt_arch.forward` now returns a code — `NT_OK`, or one of `NT_E_ARG`,
`NT_E_TOKEN`, `NT_E_CAPACITY`, `NT_E_CACHE`, `NT_E_MEMORY`, `NT_E_STATE` — and
on a refusal the output buffer is not written at all rather than half written.
`nt_check_call` in the runtime holds the checks no family should be writing for
itself, and a forward's first line is a call to it. The per-family bounds checks
in `arch_janus.c` and `arch_resonance.c` are gone: the boundary is one place now
and it refuses before the call instead of truncating inside it. Every caller in
the tree checks the code, the CLI included.

Measured, all seven cases, logits pre-filled with a sentinel to see whether
anything touched them:

    id == vocab          rc=2  token id outside the vocabulary    logits untouched
    id negative          rc=2  token id outside the vocabulary    logits untouched
    n = 0                rc=1  bad argument                       logits untouched
    pos0 negative        rc=1  bad argument                       logits untouched
    pos0+n one past      rc=3  sequence does not fit the cache    logits untouched
    pos0+n exactly fits  rc=0  ok                                 logits WRITTEN
    the good call        rc=0  ok                                 logits WRITTEN

### kv_new returned a cache with no storage

Both `calloc` results went unchecked, so a request too large for the machine
came back non-NULL with the right dimensions in its fields and NULL where the
memory should be. `kv_new(1, INT_MAX, INT_MAX)` printed
`kv_new=NONNULL k=NULL v=NULL max_seq=2147483647`. It is all-or-nothing now,
and non-positive dimensions are refused before the multiplication rather than
after: `kv_new=NULL (atomic)`.

### A model missing one weight loaded, ran, and answered

`arch_llama.c` threw away every `wt_load` result in the per-layer loop and
checked only `token_embd` and `out_norm`. Reproduced by renaming
`blk.7.attn_q.weight` to `blk.7.attn_Q.weight` in a copy of nano_arianna Q8_0 —
one byte, same length, same file size. Before: the Accelerate build exited 255
inside `cblas_sgemm` with `Parameter number 9 ... had an invalid value`, and the
scalar build exited **0** and answered `the Arianna, the city of the Arianna`
where the intact file says `the cathedral of the French language`. Same file,
same source, two platforms, one silent and one fatal.

Now both exit 1 with `llama: required tensor 'blk.7.attn_q.weight' missing or
unreadable` and write nothing to stdout. Gemma 4 gets the same treatment, with
one difference the file forces: `attn_k` and `attn_v` exist only on layers that
own a cache slot, so the requirement is per-layer rather than blanket. **That
loader could not be exercised here** — no Gemma 4 GGUF is on this machine — so
the change is reasoned from `arch_gemma4.c:366`, where the forward reads those
two only when `l < n_kv_layers`, and it is untested until a file arrives.

A refused load also used to leak: `free(m)` dropped the struct and left every
tensor already expanded behind. Both loaders now go out through their own free.

### Janus ate the previous prompt

Its running low-rank sum is model-owned, and nothing cleared it. Two identical
`forward(pos0 = 0)` calls on one model came out `0.492908299` apart on the
logits and the next pair `0.502647579` apart — accumulating, not settling — with
the argmax unchanged, which is why every gate stayed green. The comment above
`janus_state` had claimed for weeks that a new run started the sum over; the
code reallocated `vr` and left `mid` holding the last prompt.

Isolated before fixing: clearing `mid` alone drives both numbers to exactly 0,
so the `vr` clear that was in the first version of the patch is not there — `vr`
is position-indexed and every position is written before it is read.

`make test_repeat` is the gate that would have caught it, over every model on
the machine, two runs sharing a cache and a third on a fresh one, and the
invariant is exact equality rather than a tolerance. Red hand: with the reset
removed it prints `janus same-cache 0.492908299 (id 310) new-cache 0.526572466
(id 13431) FAIL`.

**And the gate's own first version was wrong in the way this log spent the
morning fixing.** The probe exits 1 on drift and 3 when it cannot run, and the
script collapsed both into SKIPPED — so the red-hand run reported the drift as
"skipped". A gate written hours after `test_parity.sh` was repaired for exactly
that shipped with exactly that. It separates the two now.

### One finding had a different cause than the one it was filed under

The report said the two archives do not close their own symbols in the
documented order. They do: `cc ... ./libnotorch_harness.a ./libnotorch.a
-framework Accelerate -lm` links at rc=0, and so does the `-l` form. The
undefined `_gguf_read_f32_array` came from a `libnotorch.dylib` dated **22
August** sitting in the working tree — `-lnotorch` prefers a shared library over
a static one, so three weeks of kernel changes were silently replaced by a file
that predates them. Real, and worse than an ordering bug, because nothing in the
output points at it. `make clean` now removes the dylib along with everything
else this Makefile can produce, and the README says which way the linker
resolves.

The documented link command was genuinely incomplete: no `-L`, no backend flag.
Both are there now, with the reason.

Also reproduced and not reproduced: the report has the scalar build's output on
a broken model as byte-identical to the intact one. Here it is different text,
not identical text. The platform fork is real; that detail is not, and it
changes how the failure looks in production — not "quietly the same" but
"confidently different".

Gates: `NOTORCH_PARITY_OK (6 checks)`, `NOTORCH_CONSUMER_OK (3 checks)`,
`NOTORCH_REPEAT_OK (3 checks)`, `NOTORCH_TOKENIZER_OK (8 checks)`, `JANUS_OK`,
`RESONANCE_OK`, notorch_test 49/49 and 73/73, test_qmatmul 46/46.
`test_quantize` still FAILs Q8_0 at 2.081e-04 over 2.067e-04, unchanged since
`cd659e8`.

---


## 2026-09-12 — telling a coin-flip from a defect, because the coin flips constantly

`harness/test_parity.sh` compares this tree against its own example. That catches a harness
that drifted from the arithmetic it was lifted from, and cannot catch the arithmetic being
wrong in both. For the families llama.cpp supports, an outside reference can — and comparing
text against one fails for a reason that is not a defect.

Greedy decoding takes an argmax. Our summation order is not the reference's, so wherever the
top two logits sit within float noise the two implementations pick differently, and everything
after that follows from the one token. A single tie-break reads as a completely different
answer. In one afternoon: SmolLM2 360M Q8_0 parted on one prompt of three,
DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M on two of three, qwen05b Q4_K_M on two of three, and
qwen05b Q4_0 on one of three. Every one of them, handed the reference's own token at the point
of disagreement, produced byte-identical text for the rest of the run.

`harness/test_reference.sh` does that re-anchoring itself and reports three outcomes:
IDENTICAL, TIE-BREAK, DIVERGED. A defect does not survive re-anchoring — a model computing the
wrong thing goes on computing the wrong thing. Across every model on this machine:

    12 identical, 6 tie-break, 0 diverged, 1 skipped

The skip is Qwen3, which this tree does not load and which therefore reports SKIPPED rather
than either colour.

**It was built by breaking it, and the breaking taught two things.** Dividing rmsnorm by n-1
instead of n turns one prompt DIVERGED with both continuations printed, and the script exits
1; restoring the line returns 0. But the other two prompts stayed IDENTICAL under that same
defect, so three prompts is a thin net — `NT_REF_PROMPTS` widens it, and a new family should.
And multiplying rmsnorm's epsilon by ten changed no output at all, because eps sits five orders
below the mean square. Not every wrong number is a detectable one.

A third thing, about method rather than code: the first version of the comparison piped both
answers into awk and matched them with NR==1/NR==2, which silently compared only their first
lines. Model answers are routinely several. On a Python prompt it found a shared head of
nothing and re-anchored on a fragment of a word, then reported TIE-BREAK — the right verdict
for the wrong reason, which is the failure mode a gate is least likely to be caught in. Both
strings now arrive through the environment, uncut.

Two things this does not do. It cannot say how close a tie was, because llama-simple prints no
logits; TIE-BREAK is evidence and not proof. And it is an oracle only for the families the
reference implements — the Method's own architectures have no outside reference by
construction and keep the Method's own goldens.

Along the way, one finding that is not about the gate: DeepSeek-R1-Distill-Qwen-1.5B declares
`tokenizer.ggml.pre = deepseek-r1-qwen`, a third pre-tokenizer family beside the gpt2 and
qwen2 ones this tree knows, and fails one tokenizer check of eight on indented code. The model
runs and its parity is sound; the whitespace rule is not this file's.

---

## 2026-09-12 — README promised a fallback that had been removed under it

`32ffc9e` made an unclaimed architecture a refusal instead of a trip through the
llama forward, and updated `harness/archs.h` to say so. The README section
describing the harness as a linkable surface, written hours earlier, still said
`llama` was the fallback and carried a code sample commenting
`nt_pick_arch` as `never NULL`. A body written against that sample would
dereference NULL on the first file the table does not claim.

Two lines, no code. What makes it worth an entry is where the lie lived: the
header and the implementation were changed together and correctly, and the
document a caller reads first was not, so the surface was consistent everywhere
except at its front door.

---


## 2026-09-12 — the scan walks in order; the projections around it never did

`mamba_layer` was written per token, and the note above it said so on purpose: the scan is
sequential by construction, token t's state is token t-1's output, so a batched version would
have to walk them in order and prefill must cost what decode costs. Half of that is true. The
scan walks in order and the convolution walks in order. The four projections around them do
not look across positions at all, and they are where the weights are — a 350-token prompt was
streaming the whole file 350 times to compute what a handful of passes could.

The layer now takes a chunk. Each projection runs once for it through `qmm`; the scan and the
convolution keep their loops over positions, ascending as before.

Measured on mamba-130m-f16, 350-token prompt, three interleaved pairs of the two binaries:

    prefill  40.1 / 41.0 / 40.8   ->   74.0 / 78.1 / 78.3 t/s
    decode   36.4 / 35.4 / 35.5   ->   37.8 / 36.2 / 35.8

Parity with the reference is unchanged: byte-identical on three short prompts and on the
350-token one, which crosses eleven chunk boundaries. Nothing about the arithmetic moved —
`qmm` is bit-identical to `qmv` per row, and both sequential loops visit positions in the same
order.

The profile says where the win came from and what is left:

    per token   qkv 4675 ms   out 1950   scan 1868   total 8672
    per chunk   qkv 2144 ms   out 1003   scan 1457   total 4684

The projections halve, and the scan drops a fifth it was not asked for — the same arithmetic
reading its inputs in chunk order instead of walking a scratch buffer per token. The
projections are still two thirds of the time, but they are no longer memory-bound: 7.9 GB of
weight traffic in 3.1 seconds is 2.5 GB/s where this machine gives 18. The cost that remains
is the conversion and the FMAs, at about a quarter of what four cores can retire.

**Three things measured wrong along the way, and the wrong numbers are here because they were
believable.** A first A/B compared the chunked build against itself — the two binaries had the
same md5, which is the check that should have come first, and it said chunking bought nothing.
A per-call `calloc` looked like the reason the first pass of a repeat read twice what the next
ones did; holding the buffer changed nothing and the real cause was thread affinity in the
pool, a floor below this file. And a tile-width sweep taken on this model at this hour read
78, 82, 72 and 87 for widths 8, 12, 16 and 24 — not a curve, and the same build read qwen
fp16 at 28 where it had read 42 earlier in the day. The tile stays at the width that was
measured cleanly. The sweep is open, not answered.

The reference reads 329 t/s of prefill on this file, so a quarter of the way.
## 2026-09-12 — the public registry stopped guessing Llama

The newly linkable harness exposed `nt_pick_arch`, and an independent consumer
asked it for an architecture that does not exist. It returned Llama. That made
the public surface contradict the repository's own rule: an unfamiliar GGUF
must not be repaired by silently pretending it is Llama.

The Llama-family implementation now explicitly claims the two architecture
names whose distinct paths it actually carries: `llama` (interleaved RoPE) and
`qwen2` (NEOX RoPE). Unknown and NULL names return NULL, so the caller refuses
the file before any model-family arithmetic runs. The installed consumer gate
checks both positive names and both refusals before opening its real GGUF.

Red hand: at the harness-library pin `e4c553d`, the independent strict lookup
probe exits 2 with `unknown architecture silently selected a family`. With the
fix it prints `STRICT_ARCH_OK`; the real installed consumer remains
`arch=llama tokens=5 vocab=32000 argmax=264`, and the archive with `archs.o`
removed still fails to link on `_nt_archs` and `_nt_pick_arch`.

---

## 2026-09-12 — a prefill that halved after the first answer

The pool gave each of its threads one core of its own and pinned it there, including the
thread that starts it. That arrangement is ten days old — `79950c9`, 2026-09-02, "the method
gives each thread a core instead of a promise" — and two things follow from it that were
never measured.

A pinned thread cannot be moved off a busy core onto an idle one, so a pool thread waking for
one matvec preempts whatever the scheduler had put there rather than going somewhere free.
And on Linux a new thread inherits its parent's affinity mask — so from the first decode
onward, every fan-out a batched matmul opened was born onto the single core the pool had
pinned its parent to. The prompt was processed by one core with three idle beside it.

Which reads as a prefill that drops by a third or a half after the first answer: every turn of
a conversation but the first. A 350-token prompt, three passes in one process,
`taskset -c 4-7`:

    qwen05b Q4_0    96.7  47.5  47.8   ->   104.4  98.2  96.8
    qwen05b Q4_K_M  89.0  36.0  35.7   ->    83.3  84.2  83.5
    mamba-130m-f16  92.8  47.5  47.0   ->    85.6  94.2  95.8

Decode, the thing the narrowing was for, measures the same or better on every model here:
gemma-4 Q4_0 10.8 against 10.8, qwen Q4_0 51.0 against 56.9, mamba f16 36.2 against 38.7.
Nothing was traded. The core *selection* is untouched and still worth what it was measured at
— `NT_QMV_PIN=0` gives that up and remains the way to.

**Two explanations were tried first and both were wrong.** The allocator: a new per-call chunk
buffer had just been added elsewhere, and calloc over recycled heap writes every byte where
calloc over fresh kernel pages does not — holding the buffer across calls changed the numbers
by nothing. Heat: the die reads 55-57 C either way, and under `NT_QMV_PIN=0` the same three
passes at the same temperature stay flat. What named the cause was deleting the caller's pin
outright and watching the collapse go, then putting it back and watching it return.

Widening only the fan-out was tried too and was not enough: the caller drains chunks as well,
so a worker that cannot leave one core holds the whole call. Holding every thread to the whole
chosen set is the change; there is no wrapper around the batched entries.

`tests/test_affinity.c` asserted the old arrangement — "the driving thread holds one core of
its own" — which is the bug written down as a requirement. It now asserts the set, and carries
the consequence as its own check: a thread created after the pool exists must see every core
the plan chose. Restoring one-core pinning turns both red, and the child's mask in the failure
reads `cpus 4` where four were expected.

Gates: 17 suites green, tokenizer identical on 24, harness parity OK, ThreadSanitizer clean on
both pools.

---

## 2026-09-12 — prefill stopped reading the file once per token, for f16

`nt_qmatmul_i8` has said since it was written that a prompt of n tokens streams the weights n
times if nobody batches, and the unpacked formats had nobody. `qmm` asked the int8 entry, was
refused, and fell to the per-token loop — so every f16 model here ran its prefill at decode
speed. `nt_qmatmul` is the other half: one pass over an f16 row against a tile of activations,
no activation quantization, since f16 weights meet float activations directly.

qwen05b_fp16, 350-token prompt, two runs each, `taskset -c 4-7`:

    prefill  20.1-20.6  ->  41.9-42.0 t/s

Per (row, activation) the walk over k is the same eight-wide two-accumulator shape as
`nt_f16_rows` with the scalar tail added in the same order, so batched and per-token agree bit
for bit. `tests/test_qmatmul.c` grew eleven f16 cases asserting equality rather than a
tolerance, including three where k is not a multiple of eight — dropping the tail turns exactly
those three red, which is how you can tell they are there on purpose. Writing the result
through a factor of 1.0000001 turns nine red. Restoring gives 46 passed, 0 failed.

Tile width, measured where it is used: four gives 36.4 t/s, **eight 41.9**, sixteen 39.2, and
the int8 path's thirty-two collapses to 25.7-28.9 where the register spills start. Eight
activations need sixteen vector registers for partial sums plus two for the converted weights,
against thirty-two that exist.

**The first version of that table measured nothing.** It was taken on mamba-130m-f16, which
walks its prompt one position at a time — so the kernel under test never ran, and four, eight
and sixteen read 37.3, 40.2 and 39.6, which is the spread of an idle measurement. A tile width
has to be measured where the tile is used.

Which names the next piece rather than hiding it. `mamba_forward` loops positions and calls
`qmv` inside each, because the scan is a recurrence — but only the scan is. The projections
around it do not look across positions and could go through this kernel for the whole chunk at
once, which is what the reference does and why it reads 309 t/s of prefill where this tree
reads 40.

F32 is deliberately not batched. Its tensors in every file on this machine are norms and
biases — 121 of them in qwen05b_fp16, none large — so there is no traffic to save, and writing
the kernel would take the summation order away from the compiler for nothing.

Gates: 17 suites green, tokenizer identical on 24, harness parity OK on Q4_0, both f16 models
byte-identical to the reference on three prompts, and qwen05b_fp16 byte-identical on the
350-token prompt that actually exercises the new path.

---

## 2026-09-12 — the goldens re-derived, and the harness made linkable

Two things that both come down to the same question: does the thing everyone is
about to build on actually hold.

### The goldens had one witness, and it was us

`tests/resonance_golden_v3.txt` and `tests/janus_golden_v4.txt` are the only
reference for the Method's own two families — llama.cpp refuses both with
`unknown model architecture`, so there is no outside oracle. Three green gates
hang off two text files, and the header of each says where its numbers came
from. Nobody had ever run that recipe back.

Both were re-derived from `arianna.c`'s own forwards, with drivers written from
scratch rather than recovered, and both were re-derived a second time with the
entire substrate swapped for `arianna.c`'s vendored notorch — a different
`notorch.c` by 3867 lines and a different `gguf.c` by 353.

- **Resonance** comes back with a worst distance of **7.15e-06** on this tree's
  substrate and **8.58e-06** on the vendored one, against a 1e-3 tolerance. The
  two substrates sit **3.34e-06** apart, so on this path our kernels and the
  ones the model was actually run with are not covering for each other.
- **Janus** comes back **bit-identical** on both, all eight logits to the
  printed digit.

Three things the exercise turned up that the headers did not say.

The Janus recipe as written **does not run**. That file is Q8_0, dtype 8, and
the reference's packed path takes F16 only: `not F16 (packed path needs F16;
use YENT_DENSE=1)`. `YENT_DENSE=1` is required and was never recorded, which
also means the reference runs dense F32 there while the port runs packed — the
tolerance measures dequantization and the exact integer matvec against dense
arithmetic, not one packed path against another.

The reference forward applies a low-rank delta mid-block, and it loads that
delta by a path **relative to the working directory**
(`weights/arianna.delta.r`, inside `resonance_load_gguf`). It is absent here —
rank 8, alpha 0, `sum|A| = 0` — so the goldens are clean, but a driver run from
a directory where that file exists would have frozen a different model with no
sign of it. Both headers now say so.

And the number that had been asserted in a comment is now measured: the shipped
integer-matvec build sits **1.240e-01** from the reference on those eight ids,
8 of 8 matched. The exact-matvec build, which is what the tolerance gate
compares, sits at 3.338e-05.

The drivers stay out of this tree. They include `arianna.c` headers, and a
build here that reaches into a sibling repository is the contamination rule.
What travels is the number and a recipe that has now been run.

### The harness became a linkable surface

`libnotorch.a` was `notorch.o` and `gguf.o`, nothing else, and `make install`
shipped four headers, none of them the harness's. A body wanting to run a GGUF
through this tree had two options: copy `harness/*.c` into its own build, or
link `main.c` and inherit its `main()`. The first is a fork; the second is not
a library. Yent's inference is being rebuilt on this harness, so it needed a
third option.

The family table and its lookup moved out of `main.c` into `harness/archs.c`
behind `harness/archs.h` — `main.c` had kept everything else `static`, so that
table was the only thing a caller needed and could not reach.
`libnotorch_harness.a` now carries the six families, the runtime, the KV cache
and the tokenizer; `make install` adds it and four more headers under
`include/ariannamethod/harness/` and `.../examples/`. Two archives and not one,
because the split is real: `libnotorch` is the substrate anything can use,
`libnotorch_harness` is family code that only means something to a caller
running models.

`harness/test_consumer_link.sh` installs into a throwaway prefix and builds
`tests/consumer_link.c` against it with `-lnotorch_harness -lnotorch` and no
notorch source on the command line — `arch=llama tokens=5 vocab=32000
argmax=264`. It runs the negative case in the same pass and requires the build
**without** `-lnotorch_harness` to fail, because a link test that would pass
without the archive is testing nothing. Red hand: dropping `archs.o` from the
archive puts the gate straight into `Undefined symbols: _nt_pick_arch`.

Gates unmoved by the split: `NOTORCH_PARITY_OK (6 checks)`, `JANUS_OK`,
`RESONANCE_OK`, `NOTORCH_CONSUMER_OK (3 checks)`, notorch_test 49/49 and 73/73,
test_qmatmul 34/34. `test_quantize` still FAILs Q8_0 at 2.081e-04 over
2.067e-04, unchanged since `cd659e8`.

---


## 2026-09-12 — the parity gate could not run, and said FAIL

The agent rules in this tree carry "a gate that cannot run must report neither
green nor red" as one of four things already paid for. `harness/test_parity.sh`
had never been brought up to it, and it had all three failure directions at once.

Against Janus Q8_0 the reference exits 1 with `llama: missing critical weights`
and the gate printed **three FAILs** with an empty `example:` line under each —
a red about the harness produced by a reference that never loaded the model. That
is the same shape as the eight tokenizer "mismatches" from 09-10, in the script
next to it.

Against a truncated GGUF neither binary loads, `A=$(./notorch ...)` fails, `set -e`
kills the script, and it printed **nothing at all**: no verdict line, rc=1. A
wrapper reading stdout for `NOTORCH_PARITY_*` sees neither.

And with both sides empty the comparison is `"" = ""`, which is **PASS**. Proved
rather than argued: the same script with the two capture lines forced empty prints
`NOTORCH_PARITY_OK (3 checks)` under the old condition and `NOTORCH_PARITY_FAIL (3)`
under the new one.

Each model is now offered to both binaries before anything is compared, and one
that will not load is `SKIPPED` with the reason taken off stderr — the first line
that reads like an error, since the first line is the shape banner, and the last
line said before it gave up when nothing matches. `NOTORCH_PARITY_OK` now carries
its check count and is unreachable at zero: every model skipped prints
`NOTORCH_PARITY_SKIPPED`. An empty harness side is a failure even when the
reference side is empty too.

One thing this cost twice: `set -e` kills a function at `OUT=$(cmd)` when the
command exits non-zero, so the first version of the reason lookup silently
returned nothing at all — the truncated file reported `SKIPPED — ` with the reason
missing. A pipeline hides it (the status is the last stage) and a bare assignment
does not.

Janus, the truncated file and a path that does not exist all skip with a real
reason at rc=0. `make test_harness` unmoved: 6 PASS, `NOTORCH_PARITY_OK (6 checks)`.
Mixed input skips Janus and still compares nano_arianna, 3 checks, rc=0.

---


## 2026-09-12 — the wrapping moves into the file, and two guards that were never guarding

`NT_CHAT` proved the mechanism and left the ids in the operator's hands, which is
the wrong place for them: they belong to the weights. `tools/gguf_add_tokenizer`
now takes `--chat "before|after|stop"` and writes `notorch.chat.{before,after,stop}`
into the file as INT32 arrays, and the harness reads them when `NT_CHAT` is unset.
The environment still wins when it is set, because trying a different wrapping is
how the right one is found in the first place.

Janus v4 Q8_0 with `32759,32760|32761,32762|32763` baked in, no environment at all,
against the same weights driven by `NT_CHAT`: byte-identical continuation, the
harness reporting `chat: 2 ids before, 2 after, 1 stop (file)` where the other says
`(NT_CHAT)`. The file grew 128 bytes and kept all 247 tensors. The original, read
with no environment, prints no `chat:` line at all.

**Both refusals in that tool were decoration.** The guard against writing a second
tokenizer and the guard against writing a second wrapping both asked `gguf_get_kv`,
and `gguf_open` does not parse array-valued keys — so the table answers "absent"
for a key that is sitting in the file. Writing twice produced a GGUF carrying
`notorch.chat.before` twice, with no rule about which copy a reader takes. This was
found by running the refusal rather than by reading it: `--chat` on an
already-wrapped file returned 0 and wrote the file. Both guards now go through the
path readers, `gguf_read_str_array` and `gguf_read_i32_array`, which do parse
arrays. The second wrapping is refused; the second tokenizer is refused and now
says how many tokens the file already has. The comment three lines below the dead
guards had stated the reason the whole time — the fix is to read the comment next
to the code you are trusting.

A `--chat` id outside the vocabulary the file declares is refused at write time as
well as at read time: `--chat "32759|99999|32763"` against a 32768-token Janus exits
1 and writes nothing.

Gates: `NOTORCH_PARITY_OK` on six checks, `JANUS_OK` worst 3.338e-05 against 1e-3
plus 8 greedy tokens identical on both matvec paths, `RESONANCE_OK` worst 4.290e-06,
`NOTORCH_TOKENIZER_OK` on 8 checks. Nothing moved for a file with no chat keys:
nano_arianna Q8_0 still encodes `The capital of France is` to `338,3228,282,4135,313`
and still answers `the cathedral of the French language, the most important document
in the world.`

---


## 2026-09-12 — a chat template made of ids, because the strings are not always there

Some families were trained with the prompt wrapped in tokens of their own, and
fed a bare prompt they do not fail — they drift. Janus v4, same weights, same
question, decoded through its own merge table:

    without the wrapping:  "The Method of the Method of the Method of"
    with it:               "I sense the resonance of the field: the field"

That is the whole argument for the feature, and it was measured before anything
was built.

`NT_CHAT="<before>|<after>|<stop>"` takes three comma-separated id lists: what
goes ahead of the user's text, what goes behind it, and what ends the turn. For
Janus that is `32759,32760|32761,32762|32763` — open, user turn, hand over to the
model, stop when it hands back.

**Ids and not strings, deliberately.** The ids are what the weights were trained
on, and the strings that spell them are not always recoverable: Janus carries
nine special ids above what its merge list reconstructs and only five of them
are named anywhere on this machine. A template expressed as text would need
those names, a template engine to interpolate them, and a tokenizer able to
round-trip them. A template expressed as ids needs none of the three and works
on a file with no tokenizer at all — which is exactly the file that needs it.

The wrapping goes around whatever the tokenizer produced, so it composes with
the byte-level fallback as well as with a real vocabulary, and the byte path's
own BOS steps aside when a wrapping is active rather than fighting it. Ids
outside the vocabulary are refused at parse time instead of indexing a row the
model does not have. In chat mode every turn is wrapped, which is what a
multi-turn conversation with such a family requires.

Nothing changes without `NT_CHAT`: nano_arianna Q8_0 encodes to the same
`338,3228,282,4135,313` and answers "the cathedral of the French language, the
most important document in the world" exactly as before.

One thing the tracing turned up that is worth stating rather than filing away.
Driving Janus by hand a position at a time and driving it through the harness
disagree on the second generated token, and the cause is not a defect: the
harness prefills the prompt as a group, and this family's smear only runs over a
group — the asymmetry inherited from its own engine, where the per-token path
carries a TODO where the smear should be. So `harness/test_janus.sh`, which
steps position by position, exercises the path without the smear, and a prompt
run through `notorch` exercises the path with it. Both are the reference's
behaviour. Which one the model was meant to have is a question for whoever
trained it.

This is the mechanism. Storing a file's own wrapping inside the file, so nobody
has to type ids, is the step after it — and the deep body coming later will want
roles per turn rather than one fixed pair, which this shape can grow into
without becoming a template language.

---

## 2026-09-12 — Janus, three attentions in one block, and a prefill that is not causal

`harness/arch_janus.c` runs Janus v4 176M — E=640, H=10, D=64, FFN=1664,
V=32768, 20 blocks, ctx 1024, rrpram rank 64. The second of the Method's own
families to come home, and the first architecture here that needs state the
shared cache does not carry.

A block blends three attentions per head by a softmax over three learned
logits: ordinary content attention, RRPRAM, and Echo. RRPRAM is not Resonance's.
Its low-rank state accumulates across positions — `mid[r] += Σ_e norm(x)[e]·wr_a[h,e,r]`
— and it reads its values through a projection of its own, so the cache holds
three tensors and not two. Around the blocks sit a smear that mixes the previous
token's embedding in through a gate on 24 dimensions, a per-layer remix
`resid_lambda·x + x0_lambda·x0`, a mid-depth snapshot subtracted at the end, and
a soft cap of `15·tanh(l/15)` on every logit.

**The source's two paths are two models, not two implementations.** Its batched
prefill sums the RRPRAM state over *every* position of the prompt and hands that
to all of them, so row 0's scores carry tokens that come after it; its per-token
path accumulates, which is causal. They answer differently on the same prompt —
8.84566784 against 8.80177402 for the same argmax — and the smear is in the
first and marked TODO in the second. The port takes the causal reading and says
so in the file; the asymmetry is measured rather than quietly repaired.

**The divergence that took the longest was not a bug in the port.** Logits sat
6.278e-01 apart with the argmax agreeing, and the bisection put it inside block 0
after the blend: `cat` matched to 1.050e-05, and the residual after the output
projection differed by 2.148e-01. The cause is that Janus ships Q8_0 weights and
`qmv` prefers `nt_qmatvec_i8` where this family's own engine calls the exact
`nt_qmatvec`. Building the port with the exact matvec closes it to **6.103e-05**
on the full vector. Resonance never showed this because its weights are F16 and
there is no integer kernel for those.

That is a difference between two engines, not a defect, and every quantized
family in this harness has been taking the integer path since D0 — which is why
parity with `examples/infer_llama.c` holds: the example does the same. So the
harness keeps its convention, and the gate asks the question that matters
instead of inventing a tolerance around the approximation: eight greedy tokens
are identical on both matvec paths.

`harness/test_janus.sh` is therefore two assertions with no grey zone. The
exact-matvec build against the reference's eight largest logits, frozen in
`tests/janus_golden_v4.txt`, worst 3.338e-05 against 1e-3. And the shipped build
generating the same tokens as the exact one.

Red hand on both halves of the block, and the second is why this gate reads
logits rather than text — the third time in three ports. Rotating adjacent pairs
instead of the split halves this family uses moved the argmax 575 → 5593 and the
logits by 8.783e+00. Pointing RRPRAM at the ordinary value cache instead of its
own projection **left the argmax at 575** and moved the logits by 3.688e+00.

Adding the family changed no line of `harness/runtime.c` and no line of the
interface beyond one `extern`. It is the first one to keep private state — a
second value cache and the running low-rank sum, both owned by the architecture
and sized against the cache it is handed, rather than widening `kv_cache` for
one family.

Open, and the same shape as Resonance's: the file carries no tokenizer, and this
one cannot simply be baked. Its merge list makes 32759 ids where the file says
32768; the nine above are special, and only five of them are named anywhere on
this machine — 32759 BOS, 32760 and 32761 the user turn, 32762 and 32763 the
assistant turn. The generated text is degenerate for a reason the source states
outright: this family was trained with its prompt wrapped in those tokens, and
a raw prompt produces off-voice salad. The harness has no notion of a chat
template, and the deep body coming after this one will need the same thing in a
different dialect.

---

## 2026-09-12 — one work contract for every agent in the tree

`AGENTS.md` makes the repository's existing engineering discipline explicit for
Codex, Claude, Gemini, and any later routed agent. It keeps the branch-only rule,
the measured-claim rule, red-hand verification, the split between README and
this log, and the Method commit format. It also records what the new harness has
made load-bearing: architecture arithmetic stays behind `arch.h`, standard
families earn tokenizer/logit/output parity against the reference, Metal remains
a first-class backend, and DoE must not make the single-body path lie.

This changes no arithmetic. The proof for the change is therefore structural:
the instructions name only files and targets that exist in the current tree,
and their branch, test, and logging rules agree with `CLAUDE.md` rather than
creating a second process for another agent.

---

## 2026-09-11 — the format nobody optimized, because nobody ships it

`nt_f16_rows` was a scalar loop over one half at a time, and so it read 9.02 GiB/s on
[50304, 2048] where the Q8_0 kernel beside it read 18.90 — half the rate for the format that
has the least to do, since f16 unpacks nothing. Eight halves a step through two FCVTLs and two
f32 accumulators brings it to 17.92-18.74 across three runs, which is Q8_0's rate and therefore
the machine's, so the matvec has nothing further to give. `taskset -c 4-7`, Exynos 1580,
`make bench_dtype`.

End to end on the two f16 files here, prompt and length fixed, four interleaved pairs of old
binary against new so the machine's drift falls on both sides:

    qwen05b_fp16.gguf   decode 8.1-8.4 -> 14.7-15.3 t/s    prefill 10.7-10.9 -> 19.3-20.7
    mamba-130m-f16      decode 21.4-23.6 -> 39.2-40.0      prefill 24.7-30.3 -> 41.0-45.7

Interleaving is not decoration. The first attempt at the qwen numbers ran all three old passes
and then all three new ones, and reported the new kernel *slower* — 6.6 against 8.3 — with
prefill swinging from 2106 ms to 734 ms inside one process. The file was 100 percent resident
both times, checked with `mincore`, so the usual suspect was not it. Alternating the two
binaries produced four consecutive agreeing pairs instead.

Which leaves a fact about this machine worth writing down, because it will mislead the next
person who looks: **load average here is permanently around 14 and means nothing.** Thirteen
kernel threads — `tz_worker_thread/0` through `/7`, `ree_time`, `tz_iwlog_thread` and their
neighbours — sit in uninterruptible sleep forever. Linux counts D state toward load, so the
number reads as an overloaded machine while no core is contended.

Why it matters beyond one format: f16 is what a model is before anybody quantizes it. Mamba's
whole decode is f16 because that is the only checkpoint published, the first download of
anything is f16, and a dtype with no packed kernel falls back to it.

**The gate is new and the first two versions of it were worthless.** `tests/test_f16_matvec.c`
compares against a double accumulation across every k from 1 to 40 — every remainder of the
vector step several times over — plus long rows where a dropped tail would be a small fraction
of the answer. Version one used tidy values on both sides: every partial sum was exact in f32
and the worst error came out a clean zero for every k, which is not a strong result, it is a
measurement that cannot see the thing that changed. Version two gave the activation a full
mantissa and went red on correct code, because it normalised the error by the answer — and a
dot product of 2048 signed terms cancels, so a sum of magnitude one over terms of magnitude
one divides by nearly nothing. Normalised by the sum of the magnitudes it reports what it
means. Red-hand: dropping the scalar tail turns 37 checks red and the binary returns 1;
restoring it returns 0.

The tolerance is one part in a hundred thousand, set by the machine without a vector path
rather than this one — a serial sum of 2048 terms drifts by about the square root of that many
epsilons. The vector path measures 2.5e-8 against it.

`bench_dtype` grew F16 and F32 columns, which is how the gap was found in the first place.

Gates: 17 suites green, tokenizer identical on 24, harness parity OK on Q4_0, and both f16
models byte-identical to the reference on three prompts each.

---

## 2026-09-11 — a family with no attention, and the interface holds a fifth time

`harness/arch_mamba.c` runs Mamba. Everything this tree had run until now keeps a KV cache and
looks back at every token; Mamba carries a fixed-size state instead — a four-wide convolution
window and sixteen numbers per channel — and each token updates it and reads an answer out.
The file declares a context of 1048576 and means it, because context length is no longer a
memory question.

Parity with the reference is exact on three prompts at the first attempt. The arithmetic came
from ggml's own kernels rather than from a description of the architecture:
`ggml_compute_forward_ssm_conv_f32` and `ggml_compute_forward_ssm_scan_f32`, with the layer
assembled in `models/mamba-base.cpp`. Two details would have been wrong from the paper alone —
softplus carries a threshold at twenty, above which it returns its input unchanged, and the
decay factor is per state element, which is Mamba-1; Mamba-2 has one per head and is a
different family.

**What it cost the interface: nothing, and one parameter is meaningless.** `arch.h` says that a
family which cannot be added without editing `runtime.c` means the interface is lying. It did
not have to be edited. But `kv_cache` has nowhere to go here, so the state lives in the model
and resets when `pos0` is zero — which the harness's `-r` mode makes load-bearing, since a
second run of the same prompt would otherwise answer with the first one still inside it. Two
runs in one process produce identical text, which is the check for that.

**A reader bug fell out of the bring-up, and it was not about Mamba.** `gguf_dequant_row`
returned success on a row that is not a whole number of blocks, and filled the tail with
whatever followed in memory: 48 values of Q8_0 decoded as the 34 bytes of one block, the last
sixteen floats uninitialised, one of them reading -3.2e28. The guard was `rb == 0`, which only
catches a row shorter than a block — `gguf_dtype_nbytes` divides, so the remainder vanished.
Every embedding lookup in this tree goes through that function. Fixed by checking the block
size directly; the file that exposed it is a Mamba checkpoint whose `ssm_dt` rows are 48 wide,
which current llama.cpp refuses to load at all.

Speed, warm, against `llama-bench` on the same f16 file: decode **21.4 / 23.6 t/s against
58.64**, prefill 24.7 / 30.3 against 309.74. The profile says where, and it is not where the
architecture would suggest: the scan with its exponential per state element is 10.5 percent,
the convolution 1.3, and the projections 46 with the head at 21.5. The model is f16 and this
tree's fast kernels are for the packed formats — 257 MB a token at 23.6 t/s is 5.6 GiB/s where
Q4_0 reaches 17. The reference manages 14 on the same file. So the gap is an f16 matvec
question rather than a Mamba question, and it belongs to every f16 model here.

---

## 2026-09-11 — what stands between this tokenizer and qwen2, named exactly

The two whitespace shapes Qwen still disagrees on were attempted and the attempt is reverted.
It is worth the entry because it found the blocker, which is not where anyone would look for it.

qwen2's pattern, read from llama-vocab.cpp rather than remembered:

    (?:'[sS]|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

Implementing its whitespace alternatives faithfully — a run ending in newlines goes whole, any
other run gives up its last character, and the word after it may take a character of any kind
rather than only a space — fixed both cases that failed and broke two that passed. Not a
mistake in the implementation. The alternative that decides those two is the fourth one,
` ?[^\s\p{L}\p{N}]+[\r\n]*`: **a run of punctuation swallows the newlines that follow it**,
and it is tried before any whitespace rule. `def f():\n` is `def`, ` f`, `():\n` — the
reference has one token for `():Ċ` where a whitespace-first reading produces `():` and `Ċ`.

The crude rule this tree carried before — scan to the next space — reproduced that by accident,
which is why space-indented code always matched and tab-indented code never did.

So the blocker is not whitespace at all. To know where a run of punctuation ends, one has to
know what `\p{L}` and `\p{N}` are over Unicode, and this gate has Cyrillic in it and an
emoji: letters and symbols land in different alternatives of that pattern, and "anything above
0x80 is a letter" gets the Cyrillic right and the emoji wrong. Approximating character classes
is how the previous attempt in this file became a regression, so no third approximation was
built.

What it would take, stated so the next person does not rediscover it: codepoint category
tables for L and N, which is what llama.cpp's unicode.cpp carries, and then the alternatives
in their written order. Until then Qwen keeps two known divergences, both in this log with
their ids, and OLMoE has none.

---

## 2026-09-10 — one pre-tokenizer where the reference keeps several

The byte-level splitter treated `' '` as the only whitespace. Everything else — tab, newline,
carriage return — fell through to the branch that scans for the next space, so `x\n\t\tdeep`
became `x` and `\n\t\tdeep` where the reference reads `x`, `\n\t`, `\t`, `deep`. Space-indented
code agreed by luck, which is why it went unseen through twenty-four gate checks.

The rule GPT-2 actually uses: `\s+(?!\S)` takes a whitespace run whole when nothing follows and
otherwise gives up its last character, and what happens to that character is decided by the
alternatives after it — ` ?\p{L}+` and its siblings begin with an optional *space*, not with
optional whitespace. A trailing space joins the word; a trailing tab or newline stands alone.

Correcting that alone was a regression, and the measurement said so before it was committed:
Qwen went from passing every whitespace shape in the battery to failing four of them. The
reference does not have one pre-tokenizer. It has a regex per family and picks by
`tokenizer.ggml.pre`, a key every file here carries — `olmo` for OLMoE, `qwen2` for the Qwen
checkpoints, absent for Gemma 4, which takes a different path anyway. qwen2's word may absorb
one preceding character of any kind rather than a space, and a run of newlines is a single
piece; those are different rules, not a better or worse version of the same one.

So both are kept and the key chooses. The qwen2 branch is what this tree has always done, now
named rather than assumed; the other is GPT-2's rule, and a file with no key or an unrecognised
one gets it, which is where llama.cpp falls back too.

Twelve whitespace shapes against `llama-tokenize`, before and after:

| | before | after |
|---|---|---|
| OLMoE (`pre=olmo`) | 3 differ | **none** |
| Qwen 0.5B (`pre=qwen2`) | 2 differ | 2 differ, the same two |

Not a claim that the tokenizer is right — a claim that it is right in one more family and no
worse in the other. The two Qwen cases are qwen2's own rules, still unimplemented, and they are
written down rather than left to be rediscovered:

    x\n\t\tdeep      ours 87,198,298,32880        reference 87,198,197,197,32880
    one\n   \n  two   ours 603,198,256,715,220,1378 reference 603,58958,220,1378

The first is a word taking a tab; the second is `\s*[\r\n]+` swallowing a whole mixed run.
Both need the qwen2 pattern rather than an approximation of it.

Also in this entry, from the same pass: three warnings this tree gained with the Q4_K work are
gone. Two were an `sub_o` left unused when the accumulate moved, and one was a positional
initializer that stopped mentioning `nt_qjob_i8`'s new fields — which is the shape of a bug
this file has had before, so it is spelled out with designators now. `notorch.c` compiles
without a warning again.

---

## 2026-09-10 — the opening token, and the default that was costing a model its voice

The harness prepended a beginning-of-text token for Gemma 4 and for nothing else. The
other two encoders — SentencePiece and byte-level — did not read the key at all, so on
a llama-family file the model was reading a prompt whose first position was not the one
it was trained to see. `llama-tokenize` on nano_arianna Q8_0 answered
`1,338,3228,282,4135,313` where this answered `338,3228,282,4135,313`, every id after
the first identical, on all four texts that ran.

**The obvious fix was wrong and the model said so.** Following the reference — prepend
for SentencePiece when the file does not declare — gave, at temp 0 on the same prompt,
`the pain,  there there there there there there there` where the same model without a
prepended `<s>` says `the cathedral of the French language, the most important document
in the world.` "Resonance is" degenerated to whitespace. Three prompts, three
degenerations. This model was not trained with an opening marker, and the file does not
record that.

**So the rule is the file, and silence means nothing rather than a guess.** Every
foreign model on this machine states the key outright: Ministral-3-3B `true`,
Qwen3.5-0.8B, Qwen3-4B, smallcoder-303M, wtforacle and doe-coder all `false`. The only
file that omits it is one of ours. Ministral is the one that matters twice, because it
is a *byte-level* file that wants the marker — so "byte-level means no BOS" is not a
safe default either. The key is the answer; the scheme is not.

Gemma 4 keeps its default of one, because that default was measured rather than assumed:
without the marker it answers a different question. That is a statement about that
family and it does not spread.

Where that leaves the gate, across four families: nano_arianna 8 of 8, Qwen3 8 of 8,
**Ministral 8 of 8** — the last is the check on the other half of the rule, a foreign
file that declares `true`, now prepended and identical to the reference. smallcoder
stays at 7 of 8 on the pre-tokenizer split already recorded below, unchanged by any of
this.

The disagreement that remains is declared rather than hidden. `notorch -T` says on
stderr whether the file asked, and the gate prints `identical after the reference's
undeclared bos 1` instead of either a green that conceals a difference or a red for a
question the file never answered.

And the fork this looked like it needed did not exist: the fix landed in
`examples/bpe.c`, which `examples/infer_llama.c` shares, so the example prepends where
the file asks too and `test_parity.sh` stays green — no second reference, no third
binary, and the runs made through the example get the same correction.

---

## 2026-09-10 — the vocabulary goes into the file, and three things the gate found on the way

`tools/gguf_add_tokenizer.c` writes a byte-level BPE vocabulary into a GGUF
that has none. Resonance 200M was one: 243 tensors, eleven architecture keys,
and a vocabulary that lived in its trainer's source as 16128 integer merge
pairs — readable by the AML program it was written for and by nothing else.

The integers are ids in the space the weights were trained in: id *i* is byte
*i*, id 256+*k* is the *k*-th merge. GGUF wants strings, so each id is spelled
in the GPT-2 byte-level alphabet and the merges are those spellings in pairs.
Order is the weights' order, not the alphabet's: sorted differently, the
tokenizer addresses different rows of the embedding table and the model answers
fluently and wrongly.

Nothing in the tensor data moves. The header is rewritten with three more keys,
the existing keys are copied as bytes — `gguf_open` skips array-valued ones and
cannot give them back — the tensor directory is re-emitted with its offsets
unchanged, and the data section goes across whole. 398 408 608 bytes in,
398 908 384 out.

**The tool's own check caught the tool's first bug.** Reading "every integer in
the file" is the obvious way to parse the merge list and the wrong one: the
trainer's C header also states its vocabulary size and merge count in its
`#define`s and comments, which added four pairs and made a vocabulary four rows
longer than the model has. `resonance.vocab_size` in the file said 16384
against the 16388 those merges produced, and the run stopped. Integers now
count only inside brackets when the file brackets them.

After: `make test_bpe` on the new file reads a byte-level scheme with 16384
tokens and passes all seven round-trips, and `notorch <file> "The field is"`
tokenizes, runs and prints *"not a fixed point, but a living field where every
wave is a wave in the air, a"* — the same text the reference produces from ids,
now from one file with nothing beside it.

**Three findings, in falling order of how much they matter.**

*The harness never prepends BOS.* Pointing `test_tokenizer.sh` at a
SentencePiece model for the first time made it visible: on nano_arianna Q8_0
the reference answers `1,338,3228,282,4135,313` where we answer
`338,3228,282,4135,313`, and every id after the first is identical on all four
texts that ran. The file carries no `add_bos_token` key and `bos_token_id = 1`;
the reference's rule is the key when present and otherwise the tokenizer's
default, which is true for llama/SPM and false for gpt2 — smallcoder-303M says
`add_bos_token = false` explicitly, which is why byte-level models passed. So
for most llama-family files the model has been receiving a prompt without the
token it was trained to see. Not fixed here, and the reason is a fork rather
than an omission: `examples/infer_llama.c` does not prepend either, so adding
it breaks `test_parity.sh` by construction. The example is the wrong reference
for this question and `llama-tokenize` is the right one, but retiring a frozen
reference is a decision, not a patch.

*`bpe_encode` splits on spaces where the reference applies a regex.* On
smallcoder-303M, an indented multi-line text diverges at one position: ours
`...711,203,264,442...` against theirs `...711,284,442...`, one of their tokens
against two of ours, everything before and after identical. A newline followed
by indentation is one merge in their pre-tokenizer and two symbols in ours.
Seven of eight texts pass; this is the eighth. Same shape as the OLMoE
added-token find — the gate was not weak in its texts, it was weak in its
corpus, and Qwen and Gemma happen to split this text the same way we do.

*llama.cpp cannot load `resonance` at all* — `unknown model architecture` — so
parity against the reference engine is unavailable for the Method's own
families by construction. It reads the file, reports its format and size, and
stops at the architecture table. Worth knowing before a campaign that plans to
hold llama.cpp beside every model: Resonance and Janus are measured against the
Method's own forward, and only the standard families get Gerganov.

**Three gate defects fixed in passing, all exposed by running it here.** A
model the reference cannot load is now SKIPPED with the reason printed instead
of reporting eight mismatches against an empty answer. Zero checks now report
SKIPPED rather than a green with nothing behind it — the same lie as the red
with nothing behind it. And the reference's output is read under `LC_ALL=C`,
because the awk macOS ships aborts on a multibyte character it cannot convert
and two of the gate's texts are Cyrillic and emoji; those two now compare and
pass.

---

## 2026-09-10 — Resonance comes home, and the file's shapes are written backwards

`harness/arch_resonance.c` runs Resonance 200M — E=768, H=12, D=64, FFN=2048,
V=16384, 20 blocks, ctx 2048, rrpram_rank 48. It is the Method's own
architecture and the first one here that was already running somewhere else:
the forward has lived in `arianna.c/tools/resonance_forward.h`, behind an AML
program, since long before this harness existed.

A block runs two attentions over one set of values. The first is the ordinary
content attention. The second, RRPRAM, never looks at a key: it maps the
normalised input through a learned low-rank basis, `temp[r] = Σ_e xn[e]·wr_a[h,e,r]`,
and scores position *j* against a second learned basis, `Σ_r temp[r]·wr_b[h,r,j]`.
One is addressed by content and the other by position, over the same V, and a
per-head sigmoid decides the mix. The second basis carries the context length
in its shape — `wr_b` is [H, R, T] — so past T the model has no basis at all.

**The port was wrong and the gate said where.** Not the architecture: the
shapes. This file writes `ne` outer dimension first, the reverse of the ggml
convention `wt_load` reads. `mlp.w_gate.weight` is `ne=[2048,768]` where a
llama-family file would write `ne=[768,2048]`, and `wr_a` is `ne=[12,768,48]`,
which only reads as [H,E,R] outer-first. Square matrices hide it completely —
`wq/wk/wv/wo` come out identical either way — and every other tensor comes out
transposed, which is fluent, confident, wrong text. The shapes now come from
the architecture's own config, the way the reference always took them; the
dtype still comes from the file.

Finding it took a bisection rather than a reading. Position 0 was the lever:
there the softmax is over one score, so both attentions reduce to V exactly and
the blend is V whatever the gate says, and RoPE at pos 0 is the identity. That
left embedding, norms, projections and MLP — and the residual stream matched
bit for bit at the entry to block 0 and differed by 4.39 at block 1, so the
fault was inside one block with two thirds of it already excluded. Dumping the
blend narrowed it to what came after.

**What the comparison measures now.** Logits, not text, against the reference
forward built standalone against this tree: at one position the two agree bit
for bit, 0.000e+00 across all 16384; from two positions on, 1.3e-05 to 1.8e-05
absolute against a mean |logit| of 1.589. That residue is named rather than
tolerated — rebuilding the port with `-U__ARM_NEON` gives 0.000e+00 at two,
three and twelve positions, so it is exactly the four-lane accumulation in
`dot_f32`/`axpy_f32` against the reference's scalar loops. Twenty greedy tokens
are identical: "The field is not a fixed point, but a living field where every
wave is a wave in the air, a".

Red hand on both halves of the block, and the second one is why this gate reads
logits. Swapping content and rrpram inside the per-head gate moves the argmax
from 432 to 262 and the logits by 4.8. Transposing `wr_a` **leaves the argmax at
432** and moves the logits by 5.6 — a text comparison would have called that a
pass. `harness/test_resonance.sh` holds the reference's eight largest logits in
`tests/resonance_golden_v3.txt` at a 1e-3 tolerance, and reads 4.29e-06.

Adding the family needed no change to `harness/runtime.c` or to the interface —
one file, one line in the table, and one local loader that overrides the shape
convention for this file.

Still open, and it is what stops this being finished: **the GGUF carries no
tokenizer.** No `tokenizer.ggml.tokens`, no merges, no scores — 243 tensors and
eleven architecture keys. The vocabulary lives in `arianna.c`'s source as 16128
integer merge pairs, so neither this harness nor llama.cpp can read the file on
its own, and the gate above had to be driven by ids. The canonical tokenizer on
the Hub confirms what those integers mean: all 16128 merge pairs match its
string pairs exactly, and its `idmap` shows the model's id space is byte-order
— id *i* is byte *i*, id 256+*k* is merge *k* — while the Hub's JSON is in
`tokenizers`' own codepoint-rank order. Baking the vocabulary into a copy of
the file, in byte-order, is the next piece.

---

## 2026-09-09 — Q4_K keeps four running sums, and stops being the slow format

Q4_K read 13.0 GiB/s where Q4_0 read 17.3 on identical byte counts — 144 bytes per 256 values
in both. The difference was never memory; it was eight scalar unpackings of a six-bit (scale,
min) pair and eight scalar float accumulates per super-block, where the block format has one
cheap accumulate per 32 values and nothing to unpack.

Both are now vector work. `nt_q4k_scales4` produces all eight pairs from the twelve packed
bytes without leaving the vector file — fifteen operations for what `nt_get_scale_min_k4` does
in about six per pair — and `nt_q4k_acc4` spends them four sub-blocks at a time into four
running sums, added together once at the end of a row.

Measured, warm, three runs each:

| | before | after |
|---|---|---|
| kernel alone, [50304, 2048] | 13.1 / 13.3 / 11.7 GiB/s | **17.5 / 17.8 / 15.6** |
| decode, a 100% Q4_K model | 9.4 / 8.7 / 8.9 t/s | **11.3 / 10.8 / 10.7** |
| prefill, same model | 6.0 / 5.4 / 5.4 t/s | **9.8 / 9.0 / 8.9** |

Q4_K now reads at the rate Q4_0 does. Prefill gains most because the batched and SMMLA kernels
got the same treatment, which they had to: **four running sums is a different order of
additions from one, and `tests/test_qmatmul.c` compares the per-token and batched kernels by
bit pattern rather than by tolerance.** Changing one kernel alone turned that gate red, and the
only honest way forward was to move all three together — per-token, batched sdot, and the
SMMLA pair — so lane *i* carries the sub-blocks whose index is *i* mod 4, ascending by block,
in every one of them. The gate is green again and it means what it always meant.

What is genuinely different from before this change: every Q4_K result now rounds in a new
place, and numbers recorded in this log before today will differ in their last bit. The oracle
in `tests/test_qmatvec.c` — an independent dequantise-then-BLAS path — puts the packed kernel
at relative error 1.5e-06 and the integer one at 0.0049 against a tolerance of 2e-2, which is
where they were. Greedy text on a Q4_K model is identical to the reference.

Two things found on the way. `gguf_quantize` skips 3-D tensors entirely, so requantising a
mixture leaves its experts untouched — OLMoE came back 6.9 percent Q4_K with 93 percent still
Q4_0, and the tool is unusable for mixtures until that is fixed. And `qwen05b_q4km.gguf`, which
this repo has treated as its Q4_K model, is 6.1 percent Q4_K: at n_embd 896 almost nothing
divides by 256, so the quantizer's documented fallback sent it to Q5_0. Measurements of "a
K-quant model" taken on that file measured Q5_0.

Also: `test_qgather`, a 146 KB binary, was committed by accident and is untracked now —
`.gitignore` does not reach a file that is already in the index.

---

## 2026-09-09 — where the K-quant kernel loses its quarter, and two ways of not getting it back

`bench_dtype` put Q4_K at 13.0 GiB/s against Q4_0's 17.3 on identical byte counts — 144 bytes
per 256 values in both formats, so the difference is arithmetic and not memory. This entry is
what that quarter is made of and two attempts at it that made things worse, which is the useful
half.

**Where it goes.** Replacing `nt_get_scale_min_k4` with a wrong-but-cheap substitute takes
Q4_K from 13.1 GiB/s to **15.4**. That is not a fix — the arithmetic is nonsense — but it is an
honest clock: unpacking eight (scale, min) pairs out of twelve packed 6-bit fields is roughly
eighteen percent of this kernel. Everything else it does, Q4_0 does too.

**First attempt: fold the eight sub-block totals with a pairwise tree.** The AVX2 body has done
this since it was written; the NEON one still drained each of the eight with its own `vaddvq`,
which leaves the vector unit for a scalar register with the next drain waiting behind it. Six
`vpaddq` instead of eight `vaddvq` measured **11.7 to 13.1 GiB/s against a 12.6 to 13.8
baseline** — no better, probably worse. Collecting the partials into an array to fold them
appears to cost more than the drains it removes.

**Second attempt: hoist the unpack above the dot products,** so the scalar work overlaps the
vector work rather than sitting between each dot and the multiply that consumes it. The AVX2
body does this too. Measured **10.9 to 11.6** — clearly worse. Two small arrays of eight bytes
are enough to spill.

Both attempts were structural changes that the compiler was already handling better on its own,
and both are reverted. The tree is unchanged by this entry; only the knowledge is new.

**What would capture it, and why it is not done here.** The unpack has to become vector work
feeding a vector accumulate, with nothing round-tripping through memory in between. That means
reordering the float sum inside a super-block — and `nt_q4k_acc` exists precisely to stop that:
it spells out both fused operations so that the per-token, batched and SMMLA kernels round in
the same place, and `tests/test_qmatmul.c` compares them by bit pattern rather than tolerance.
The three would have to move together, and the results would differ in the last bit from every
number this tree has recorded. That is a decision about what the library guarantees, not an
optimization, and it belongs to whoever makes such decisions rather than to whoever noticed the
eighteen percent.

---

## 2026-09-09 — eight experts, one fan-out, and a smaller number than the one predicted

`nt_qmatvec_i8_gather` takes one activation and a list of weight bases and dispatches them as
one matrix: the activation is quantized once instead of once per slice, the pool is woken once
instead of once per slice, and a worker's run of rows is as long as the whole gather rather
than one expert. The experts stay where they are — a mixture's chosen eight are scattered
through sixty-four and copying them adjacent would cost exactly the bandwidth this is meant to
save. The pool's row cursor walks the slices as if they were one matrix and maps each claimed
chunk back, splitting it where it straddles a boundary.

What it is worth, warm, six runs: decode **18.2 / 19.0 / 18.8 / 18.8 t/s against 18.4 / 18.1 /
18.4 / 18.2**, and in the profile the expert matmuls go **717 ms to 662** for the same 24
tokens — 432 MiB a token at 13.25 GiB/s before, 15.3 after.

That is two or three percent of decode, and the estimate that justified building it said
fifteen. The estimate was wrong in a way worth writing down: it priced all 432 MiB of expert
weight at the 17.3 GiB/s the attention weights already reach, but only gate and up share a
token's activation and can be gathered — 288 of the 432. `down` takes a different vector from
each expert and is not the same operation. Two thirds of the bytes, moved most of the way, is
the whole of it.

Kept regardless, for two reasons that are not the speed. Parity is bit-identical — the output
is the same floats, which `tests/test_qgather.c` checks against the loop it replaces rather
than trusting, at five dtypes and at shapes where the chunk size does not divide the slice, so
chunks land across boundaries. And the entry is not OLMoE's: any mixture this tree meets reads
its experts this way.

The gate was built by breaking it twice. Writing a slice's rows into the gather's base rather
than its own stretch of the output fails six checks; ignoring the slice boundary when a chunk
straddles one fails five. The first version of the test failed for a reason of its own making
— weights filled with counted bytes give a NaN f16 scale, and NaN compares unequal to itself,
so it reported a broken kernel where the kernel was fine. The weights are quantized from
floats now, and the note is in the file for whoever writes the next one of these.

Still open, with numbers: the head reads 80.6 MiB a token at 12.5 GiB/s where attention manages
17.3, and it is the one Q6_K tensor in the file. Whether that is the format or our kernel for it
is a measurement nobody has taken.

---

## 2026-09-09 — the harness can be asked twice, and says how much of the model is in memory

`-r N` runs the same prompt N times in one process. The load happens once, so what is timed is
the model running rather than the model arriving. Before every run — including a lone one — the
harness prints how much of itself is resident and how much the machine could still give it,
because on a phone those two numbers decide the result more than anything in the arithmetic
does.

The reason for both is a measurement that was worthless and had already been written down.
OLMoE is 3.66 GB on a machine with 7.6, and the same binary on the same file gave 6.7 t/s and
18.8 t/s within one afternoon — the difference being whether 1.2 GB happened to be free. On
that basis this log carried "16.5 t/s, 77 percent of the reference" and a decode profile
putting 62 percent in the expert matmuls. Both are withdrawn. Three explanations were built on
them and all three are withdrawn with them: that the gap was in 384 dispatches per token, that
gathering the chosen experts into one call was worth 21 percent, and that scattered expert
reads cost twice contiguous ones. The first was contradicted by a bench on the same shapes; the
second measured adjacent experts, which routing never produces; the third was two runs, and six
repeats of the same command gave 13.2 to 15.2 GiB/s with nothing near the 7.5 it rested on.

What the mode gives instead, on a warm process with residency printed beside each line: run 1
at 2.86 GiB resident decodes 7.9 t/s, runs 3 through 6 at 3.17 GiB decode 18.4, 18.1, 18.4,
18.2. Two percent apart, and the line above each says why. Against `llama-bench` at `-t 4`,
24.30 ± 0.19 on the same file, that is **75 percent** — and now both sides are warm, which the
earlier comparison was not: `llama-bench` loads once and iterates, while a fresh harness
process faults in 3.66 GB first.

The warm profile is a different picture from the cold one: FFN 61.2 percent, `qkv+bias` 16.1,
head 11.8. The FFN reads 453 MB per token in 29.9 ms, which is 14.8 GiB/s and matches a
standalone bench of the same shapes — so the expert matmuls are running at the rate this
machine gives, and there is no mystery left in them.

One finding is left deliberately unused. The block holding the router took 26.7 percent of
decode for seven percent of the bytes: `ffn_gate_inp` is the family's one f32 weight, and f32
goes through a matvec without the integer kernel's dot instruction. Packing it to Q8_0 at load
is worth about nine percent of decode, 18.6 to 20.5 t/s — and it breaks parity on one prompt in
three, because routing is a discrete decision. A small numeric change reorders neighbouring
scores, a different eight of sixty-four run, and the text stops matching the reference, which
routes in f32. The measurement is in the source beside the code that does not use it. Trading
correctness for nine percent is a decision to be made out loud, not a side effect.

Also from review, and the same class as two before it: a check labelled "starts at the right
row" whose predicate also tested the row count, so a slice with the right base and the wrong
height failed under a label that did not mention height.

---

## 2026-09-04 — four on the mixture, and the one that would have answered with seven eighths

Review on the OLMoE merge, all four fair, and one of them a wrong answer rather than a crash.

`wt_expert` refused to slice a weight held as expanded f32, taking only the packed form. That
did not fail loudly: the caller logged once and skipped that expert, so a model whose expert
tensors fell back to f32 — which is exactly what happens to a dtype with no packed kernel —
would have run with seven of its eight and said nothing. The expanded form is a contiguous
`[rows, cols]` and slices by the same arithmetic; refusing it was refusing the case the
fallback exists for. The docstring said so too and now does not.

`tests/test_wt_expert.c` gates both forms without a model: rows are filled with their own
index, so a slice that lands wrong reads a number naming where it landed. Thirteen checks,
including the three refusals that must stay refusals — past the end, negative, and a weight
with no data at all. Restoring the old `!src->q` condition fails the five f32 cases and
nothing else, which is how the gate was shown to bite.

The rest: `olmoe.expert_count` came out of the file and went unbounded into a `taken[1024]` on
the stack, so a header claiming more experts than that would have written past it — the count
is now checked against the buffer it indexes. `bpe_encode` handed a NULL tokenizer to the span
path, which reads `t->byte_cp` on its first line. And the expert loop's failure branch dropped
one expert of eight; it now drops the whole feed-forward for that token, because a missing
contribution shows in the output and a quietly missing expert does not.

Gates: 15 green including the new one, tokenizer identical on 24, llama parity identical on
three prompts, olmoe parity identical on three.

---

## 2026-09-04 — the first mixture, and the tokens that were never in the alphabet

`harness/arch_olmoe.c` runs OLMoE-1B-7B: sixteen layers, sixty-four experts each, eight used
per token. It is the first architecture here whose weights are not read in one sweep. A dense
model streams every byte of every layer; this one reads an eighth of its feed-forward, but
reads it gathered — eight slices chosen per token out of a 75 MB region per tensor per layer.
Everything this library has been tuned on assumed the sweep.

Adding it needed no change to the family interface, which is what `arch.h` claims of itself.
It needed one helper: `wt_expert`, which is arithmetic rather than a new idea. A stacked
expert tensor is 3-D, `[n_embd, n_ff, n_expert]`, and `wt_load` already reads that as one
matrix of `n_ff * n_expert` rows — so expert *e* is that matrix with a shorter row count and a
shifted base, and nothing is copied. `gguf_type_size` became public to make the shift
computable, and `tools/gguf_quantize.c` lost its private copy of it.

The routing is the reference's and not the usual shape of these things: softmax over all
sixty-four, top eight by that probability, and the weights are those probabilities as they
stand — no renormalisation over the chosen eight (`norm_w = false` at the call site) and no
scale (the file carries no `expert_weights_scale`). Both are common elsewhere and wrong here.
Q and K are RMS-normalised over the whole projection before the heads are split out, with a
weight vector of `n_embd` — gemma4 normalises per head, and the two are one reshape apart.

**The tokenizer was the real find, and it was ours.** OLMoE's vocabulary carries twenty-five
tokens the file marks USER_DEFINED, and they are runs of *real spaces* — token 50274 is four
bytes of 0x20, not four 'Ġ'. No sequence of merges over the byte-level alphabet can ever spell
one, so the reference matches them against the raw text before BPE runs and encodes only what
lies between. We had no such pass, on either side: indentation was lost when encoding and lost
again when printing, because a literal space is not in the byte-level table and got dropped.
`gguf_read_i32_array` reads `tokenizer.ggml.token_type` to find them; CONTROL tokens are left
out on purpose, since the reference only splits those when asked to parse specials.

Why it went unseen is the part worth keeping. `harness/test_tokenizer.sh` **already** had a
repeated-spaces text and an indented-code text, and they had passed for as long as they
existed — on Qwen and Gemma, whose vocabularies have no such tokens, where both
implementations happen to split the same way. The gate was not weak in its texts; it was weak
in its corpus. OLMoE is in its default list now, 24 checks instead of 16, and removing the
added-token pass fails exactly those two texts and only on that model.

Against the reference, cold: parity is identical on all three prompts, tokenizer identical on
24. Decode **16.1 / 16.8 t/s against llama.cpp's 21.41 ± 1.15 — 77 percent**, where the dense
models sit at 97. The gap is not a mystery and is not a defect: the reference gathers the
eight chosen experts into one `mul_mat_id`, while this does eight separate matvecs, twenty-four
per layer against three, 384 dispatches per token. That is the next piece of work and it now
has a number.

---

## 2026-09-04 — six review points, and one of them was a comment that lied

The worst of the six is the smallest. `NT_CACHELINE` carried a note saying the line size was
"read from sysfs … rather than assumed", beside a `#define` of 64. Sysfs is where **I** read it,
by hand, while writing the change; the code reads nothing, and alignment is a compile-time
decision so it could not. A comment that describes behaviour the code does not have is the same
failure as a commit message that does, and this log has spent a week saying so. It now states
what it is: a constant, chosen because every core this has run on reports 64, wasteful rather
than wrong if some machine reports more.

The rest, all real:

- `tests/bench_claim.c` took its thread count straight from `atoi` while sizing its arrays at
  64, so `./bench_claim 128` wrote past `pthread_t th[]`, and a zero divided by zero in
  `id % nthreads`. Both arguments now go through a parser that takes the whole string in range
  or says why it did not, and `sched_getaffinity`'s return is checked rather than assumed.
- The same file uses Linux affinity APIs and would not compile on macOS, which this repo
  builds on. Guarded, with a `SKIP` main elsewhere — verified by compiling the other branch
  with the condition forced false under `-Wall -Wextra`, not by reading it.
- `1L << 40` as the upper bound for `NT_QMV_THREAD_MIN` is undefined where `long` is 32 bits.
  `LONG_MAX` and `INT_MAX` say the same thing without the shift.
- `(void)e;` left over from the plan refactor, where `e` is used three lines above.
- Integer-to-pointer casts through `long` in the benchmark's thread argument, now `intptr_t`.

Nothing here moves a number; the gates are unchanged and decode is where it was.

---

## 2026-09-04 — the false sharing was real and it was not the point

The pool coordinates through three counters. `offsetof` said where they sat: `shutdown` at
232, `generation` 236, `busy` 240, `next` 244 — one 64-byte line, with the job description
starting at 248 in the same line. Every one of the sixty-four row claims a matvec makes wrote
that line while the other workers were reading it.

Separating them changes nothing measurable in decode. Four runs each, cold: Gemma
11.2 / 11.1 / 11.1 / 11.1 against 11.1 four times, Qwen 56.4 / 56.5 / 56.9 against
56.2 / 57.2 / 56.3 / 56.3, and at `NT_QMV_CHUNKS=64`, where claims are four times as frequent,
53.0 against 53.3. That is a null result and it is reported as one.

But "no effect in decode" and "no effect" are different claims, and four workers cannot settle
the second. `tests/bench_claim.c` measures the claim alone — a worker reads `generation` the
way the spin loop does, then claims a row — and the sharing costs plenty: **51.9 Mclaims/s in
one line against 121.3 separated at four threads, 2.34x**; 1.25x at one thread, 2.31x at three,
1.95x at eight.

Both facts fit together in arithmetic. A matvec makes 68 claims: 1.3 us shared, 0.6 separated.
A token runs a few hundred matvecs and takes about ninety milliseconds. The saving is a
fraction of a millisecond, which is exactly the nothing the decode measurement found.

The layout is separated anyway, and the reason is the ratio rather than the milliseconds: it
costs 192 bytes in one global, the operation genuinely runs twice as fast, and anything that
claims more often — more cores, finer chunks, smaller matrices — moves the arithmetic. Both
pools get the same treatment. What this entry does not claim is a speedup, because there
isn't one here.

Three review points on the merges, all fair. `NT_QMV_SPIN` was parsed with `atoi`, which maps
junk to zero — and zero is a valid setting meaning park on every wait, the worst one there is.
`NT_QMV_SPIN=off`, written by analogy with `NT_QMV_POOL` where "off" is accepted, would have
silently chosen it. Every knob now goes through one parser that takes the whole string or
prints why it did not. `tests/test_plan_race.c` used `pthread_barrier_t`, which POSIX makes
optional and macOS does not have, and that test is in the default `make test` — replaced with a
mutex and a condition variable. The `test_tsan` target compiled without `$(CFLAGS)` and
`$(BLAS_FLAGS)`, so it sanitized a different configuration than the one that ships.

One part of that review did not hold up: the concern that `A && { …; exit 1; } || C` lets a
detected race exit zero. Checked against a genuinely racy build — 3215ed0, before the plan was
made once — and the recipe prints RACE REPORTED and exits 1. `exit` inside the group ends the
shell before `||` is reached.

---

## 2026-09-03 — the workers were going to sleep between matvecs

A worker looks for its next job for a while before parking on a condvar, and the budget for
that looking was 20000 iterations. Between two matvecs of one token the gap is the scalar work
— norms, rope, softmax, quantizing the activation — and some of those gaps are longer than
20000 iterations covers. Every one of them parked three workers and woke them again through a
futex.

Raising the budget, four cores, decode t/s on Gemma 4 E2B and Qwen 2.5 0.5B: 20000 -> 10.5 /
52.7, 200000 -> 10.9 / 56.1, **500000 -> 11.1 / 56.5**, 1000000 -> 11.1 / 56.1, 4000000 ->
11.1 / 57.6. Flat past the knee at half a million, which is the new default.

Spinning does not cost what it looks like it costs, and the CPU seconds beside the wall clock
say so: 11.6 cpu-s at 20000 against 11.3 at 500000 for the same 24 tokens, wall 5.53 s against
4.98. Parking and waking spends more cycles than looking does. What it does cost is a core held
for about ten milliseconds after the last dispatch before the worker gives up — free for
continuous decoding, a small drain for a process that runs one matvec and waits.

Cold, three runs each, against the previous default and the reference at `-t 4` on the same
files: Gemma **10.6 / 10.0 / 10.6 -> 11.0 / 11.0 / 11.1** against llama.cpp's 11.33, Qwen
**52.3 / 52.1 / 52.5 -> 56.4 / 56.0 / 55.7** against 57.98. Decode is at 97 percent of the
reference on both, from 93 after the core work and 76 and 55 before it.

An old note in this log said the opposite — that a long spin halved throughput on two cores,
2.8 t/s against 5.1. That measurement was taken before the pool pinned one thread per core,
and it was measuring two threads sharing one core, where a spinner starves the thread doing the
arithmetic. With placement settled the sign reverses: on two cores now, 8.1 at zero budget
against 8.95 at 200000. The old number was about placement, not about spinning.

One guess died on the way. The three counters the pool coordinates through — `generation`,
`busy`, `next` — are consecutive ints in one cache line, so every chunk claim invalidates the
line the others spin on, and false sharing predicts that more spinning hurts. It helps, all the
way to four million. Worth separating those fields anyway, but that is a different change with
its own measurement, not the explanation for this one.

`test_qpool` now also runs at `NT_QMV_SPIN=0`, which is the only way the park-and-wake path
executes at all — at the default budget a worker essentially never reaches the condvar. Said
exactly, because it was tried: that run caught nothing the default run missed. A lost wakeup is
a race and no bit comparison finds it; corrupting the busy count, the hazard the dispatch
comment warns about, fails both runs. It is coverage before the next refactor touches that
path, not a second detector.

---

## 2026-09-03 — one plan behind one guard, and the five races a review only half saw

Review on the pinning merge pointed at `nt_qmv_target_cpus`: a `cpu_set_t` cached in a
function-scope static behind a lazy `if (n < 0)`, with nothing synchronising it. The concern
is right and it is not narrow. This library has two worker pools behind two separate
`pthread_once` guards, so a program calling the float matvec from one thread and the integer
matvec from another runs both initialisers at the same instant, and four separate functions
each cached their answer that way. Patching them one at a time would have left the shape
intact, so everything the fan-out decides once — thread count, core set, pinning, granularity,
the threading floor — is now one struct built under one `pthread_once`.

The claim that the races are gone is a tool's answer, not an argument. ThreadSanitizer runs
here under `setarch -R`, because Android's ASLR is wider than its shadow mapping expects, and
`tests/test_plan_race.c` releases two threads through a barrier so the two initialisers
overlap on purpose. On the version before this change it reports **five**:

    host.7             nt_qmv_host_threads
    n.6                nt_qmv_fast_cpus
    set.2 (128 bytes)  the cpu_set_t the review predicted
    g_qmv_thread_min   not in the review
    enabled.1          not in the review

The last two were found by the sanitizer alone — neither the review nor I had them. After the
change: none. `make test_tsan` fails on any warning, and the test also runs in the ordinary
suite, where it checks that both pools agree on the plan and that each path matches its own
serial result. `g_qmv_thread_min` keeps a public setter, so it stays a global and is read and
written atomically, with a compare-and-exchange for the default so an explicit
`nt_qmv_set_thread_min` wins whichever happens first.

The refactor cost speed twice, and both cost was found by measuring rather than by reading.
`nt_qmv_spin()` is called from inside the innermost wait loop; routing it through the plan put
a libpthread call on every spin iteration, and hoisting it into a local per thread took decode
from 9.6-9.8 back toward 10.5. Then the accessor itself, holding a `pthread_once` call, could
not be inlined into `nt_qmatvec_i8` beside the shape checks: 10.4-10.6 with an occasional 9.3
against a flat 10.7 before, twelve runs each. Splitting it — an inlinable acquire load of a
pointer the initialiser publishes with release ordering, and an out-of-line slow path for the
first caller — removed the dips and left a steady 10.6.

**What remains, stated rather than buried: 10.6 against 10.7, about one percent, twelve runs
each and not noise.** Four atomic loads per matvec account for microseconds per token, so the
rest is most likely code layout, and chasing it further costs more than it returns. The trade
is one percent of decode for five removed data races, one of them a partially built CPU set
under concurrent start. Worth it, and the number is here so anyone can disagree.

Also from the review, both fair: the affinity test still said "three modes" after gaining a
fourth, and its `NT_QMV_PIN=0` mode reported a failure as "expected the plan's first core"
when what it expects there is the mask untouched — a message that would send whoever reads a
failure looking in the wrong place.

---

## 2026-09-02 — the prime core is the slow one, and the fix was in the chunking

Three big cores beat four: `cpu4-6` decoded Gemma 4 at 10.7 t/s against 10.5 on `cpu4-7`, so
adding the *faster* prime core cost three percent. Four explanations were tried and three died.

**Memory bandwidth, refuted.** A standalone streaming-read probe — threads reading disjoint
slices of a 384 MiB buffer, pinned before touching anything — gives 18.9 GiB/s on one core,
18.4 on two, 17.6 on three, 17.5 on four, 16.4 on all eight. One core already saturates this
memory system, and three against four is flat. Which also corrects an older claim in this log:
decode is not simply memory-bound. If it were, four cores could not decode 72 percent faster
than one, since the bytes arrive at the same rate either way. The scaling comes from the work
between the bytes — unpacking nibbles, applying scales, accumulating — and that part is
compute.

**Android holding the prime core, refuted.** Twenty idle seconds of `/proc/stat` put cpu7 at 34
busy ticks and cpu6 at 57, against 519-636 on the small cores. The prime core is the *least*
contended on this phone, not the most.

**A thermal or policy cap, refuted.** `scaling_max_freq` for cpu7 reads 2910000, every cpufreq
cooling device sits at state 0.

**What it actually is.** Sampling `scaling_cur_freq` during a decode: a thread pinned to cpu7
runs at **1.6-2.0 GHz while its neighbours hold 2.6**. The governor is `energy_aware`, the
prime core is expensive in its energy model, and a pinned thread cannot be migrated away — so
it is simply run slowly. Nothing in `cpuinfo_max_freq`, which reports 2.91 GHz for that core,
says any of this. In practice cpu7 is the slowest of the four big cores, and a chunk landing
there is the straggler everybody waits for.

That is a tail, and a tail is a granularity problem. Chunks per worker against decode t/s,
four cores, Gemma then Qwen 2.5 0.5B: 1 -> 9.9 / 55.9, 2 -> 10.3 / 52.9, 4 -> 10.5 / 54.4,
8 -> 10.6 / 55.1, **16 -> 10.7 / 55.9**, 32 -> 10.7 / 55.3. The default moves from four to
sixteen, and at sixteen four cores match the three that exclude the prime core — 10.7 either
way — so the answer is finer chunks rather than throwing a core away. Confirmed at three runs
each: `cpu4-7` 10.5 / 10.5 / 10.5 at four chunks against 10.7 / 10.7 / 10.7 at sixteen, Qwen
53.5 / 53.2 / 53.7 against 54.3 / 55.7 / 55.7. The float pool had arrived at sixteen on its
own; only the integer pool was still on four.

The core-selection rule is left alone. It would have had to exclude cpu7 on evidence that no
static file provides, and the thing it was compensating for is gone.

`tests/test_qpool.c` now runs at three granularities from the Makefile, since the division is
read once per process: coarse, default and fine must agree to the bit with the single-threaded
reference. Injecting the header's documented bug — one row skipped per chunk — fails all five
dispatches at every granularity, which is how the new dimension was shown to bite.

---

## 2026-09-02 — one thread per core, because the scheduler stacks them two to a core

The core-class work left an oddity in its sweep: two cores measured slower than one. Gemma 4
decode on `taskset -c 6,7` gave 5.1 t/s against 6.1 on `cpu7` alone, and `cpu5,6` — two cores
of the same 2.6 GHz class — gave 4.6, 8.8, 4.6 across three runs where every other
configuration repeated itself to the decimal.

Not a spread, two plateaus, and the ratio between them is 1.9. Sampling `/proc/<pid>/task`
through six runs said what that means: in the fast ones the two threads accumulate time on
cpu5 and cpu6, in the slow ones both sit on cpu5 and the second collects a quarter of the
first's cycles. One core doing the work of two. The scheduler is not being careless — a thread
that alternates spinning with parking on a condvar reads as half idle, two halves fit on one
core by that arithmetic, and the placement is decided once and kept for the run.

So the pool pins one thread per core, in order, instead of handing every thread the same mask.
Honouring an affinity mask somebody else set means honouring **which** cores they gave us; it
does not mean declining to use all of them, and a mask from `taskset` or a cgroup is usually
somebody reserving cores for exactly this. `NT_QMV_PIN=0` turns all of it off.

Six runs each, cold, with an unpinned control in the same sweep on the same hardware:

| | runs |
|---|---|
| `cpu5,6` pinned | 8.3 8.3 8.3 8.3 8.3 8.3 |
| `cpu5,6` unpinned | 4.5 8.8 4.6 4.6 5.8 8.8 |
| `cpu4,6` pinned | 8.3 8.3 8.3 8.3 8.3 8.3 |
| `cpu6,7` pinned | 7.7 7.7 7.7 7.8 7.7 7.7 (was 5.1) |
| all cores, default | 10.5 10.4 10.4 (unchanged) |

The control still swings, which is what makes this a difference rather than a story. `cpu6,7`
gains half again. Stated plainly: pinned 8.3 is about six percent below the *lucky* unpinned
8.8, because a pinned thread cannot step aside when Android wants that core for a moment — the
trade is six percent off the best case against eighty percent off the worst, and a number that
repeats. The default four-core path does not move; this is for masks somebody narrowed.

`tests/test_affinity.c` gains a fourth mode, `NT_QMV_PIN=0`, and its placement assertion now
reads "one core of its own" rather than "somewhere in the plan" — the old wording passed
whether or not the threads were stacked, which is exactly the failure this entry is about.

Left open and measured, not explained: three big cores beat four. `cpu4-6` gives 10.8 / 10.8 /
10.7 against `cpu4-7` at 10.5 / 10.5 / 10.6, so adding the *faster* prime core costs three
percent. Either bandwidth is already saturated at three, or cpu7 is where Android puts its own
foreground work and we are sharing it. Next.

---

## 2026-09-02 — the library picks its cores, because counting them was the wrong question

`nt_qmv_host_threads` asked `sched_getaffinity` how many CPUs it was allowed and used all of
them. On a phone that is the wrong question. This SoC runs four cores at 1.95 GHz, three at
2.6 and one at 2.91; the matvec splits into equal chunks, decode is already at the memory
ceiling, and a chunk that lands on a small core holds up the whole operation. Which chunk that
is changes from run to run, which is what made a benchmark disagree with itself between days —
the earlier figures were taken under `taskset -c 4-7` and the later ones were not, and nothing
had regressed at all.

Measured by core set, Gemma 4 E2B decode: `cpu7` alone **6.0** t/s, `cpu6-7` 5.2, `cpu5-7`
8.3, `cpu4-7` **10.2**, all eight 7.9, the four small ones 3.4. A single small core added to
the big four costs a fifth of the throughput. More cores are not more arithmetic when the
weights arrive at 91 percent of what the memory can deliver.

The rule that survives measurement is **drop the slowest class**, not keep the fastest. The
first version of this patch kept the fastest, which on three classes leaves the one prime core,
and it benchmarked at 6.0 against the 8.0 it was meant to improve. Dropping only the slowest
leaves `cpu4-7`, which is the measured optimum, and needs no threshold constant to say so.

Engaged only when nobody else has decided: an explicit `NT_QMV_THREADS`, an affinity mask
already narrower than the machine (`taskset`, a cgroup, a container), a uniform machine, no
cpufreq, or `NT_QMV_BIG_ONLY=0` all leave the choice alone. Fewer than two surviving cores is
treated as an unfamiliar topology rather than an opportunity. Worker threads and the thread
that drives them are held there with `pthread_setaffinity_np` — the dispatcher drains a chunk
alongside the pool, and a straggler is a straggler wherever it sits.

Cold, three runs each, interleaved. Gemma decode **8.7 / 8.8 / 8.6 to 10.5 / 10.5 / 10.5**,
prefill 9.4 / 9.6 / 10.1 to 13.4 / 20.9 / 20.4. Qwen 2.5 0.5B decode **33.0 / 33.1 / 34.0 to
55.8 / 55.0 / 57.0**. Against the reference at `-t 4` on the same two files: 11.31 and 59.91,
so decode is at 93 percent of it on both, where it had been 76 and 55. Prefill is not compared
here — ours is an 11-token prompt and theirs is a 32-token batch, which are different
quantities until they are measured the same way.

`tests/test_affinity.c` holds it, in three processes because the decision is cached on first
use: the default, `NT_QMV_BIG_ONLY=0`, and a mask the test narrows itself before the first
call. It derives the expected core set from sysfs rather than asking the library, since a test
that calls the function it is checking agrees with itself whatever that function does. Both
ways of breaking it were tried: restoring the keep-the-fastest rule fails both assertions
(planned 1 against 4), and removing the pinning fails exactly the one written for it.

---

## 2026-09-01 — the packed matvec was keeping its scratch

Review on the three merges of the day raised eight points. Six were fair and are fixed below.
Reading the code to answer one of them turned up something nobody had flagged: `nt_qmatvec_i8`
takes three buffers per call — the quantized activation, its scales, and the per-block sums —
and frees two of them on every exit path. `asum` is freed on the allocation-failure path and
nowhere else: not on the unsupported-kernel return, not on the single-threaded shortcut, not
on the pool return, not at the end. That is the hottest function in the library, and it leaked
`k/32 × 4` bytes every time it ran. Watched on a 0.5B model, resident memory climbed
380228 → 387180 → 394976 kB across ten seconds of decode; with the frees in place the same
window gives 380560 → 384068 → 387412, and what is left is the KV cache becoming resident as
the context fills — 24 layers × 2 KV heads × 64 × 2 × 4 bytes is 24 kB per token of context,
which is the slope that remains.

`tests/test_qmatvec_leak.c` is the gate, and it is `mallinfo2` rather than RSS: RSS moves for
reasons that have nothing to do with the caller, `uordblks` is exactly the sum of live
allocations. Two thousand calls now retain **0 bytes**; with the frees removed again the same
test reports **416000**, which is 208 per call against the 192 the arithmetic predicts plus
the allocator's header. Where `mallinfo2` is absent the test says so and skips, because a gate
that cannot measure must not claim to.

The same review was right that `nt_qmatvec_i8` validated dtype and shape but never the
pointers, so a caller whose weight failed to load reached the kernel and died there instead of
receiving the `-1` it was testing for — `harness/arch_gemma4.c` tests for exactly that. Guard
added; removing it again turns the new test into a segfault rather than a failed assertion,
which is its own kind of proof.

Also from the review, all confirmed before changing anything: the Python ctypes binding had
drifted from `gguf_file` by two additions, `kv_end` from the quantizer work and `map_base` /
`map_len` from the mapping, and a `Structure` with a gap does not fail loudly — every field
past the gap reads its neighbour. `tests/gguf_layout.c` puts `data` at 443432 and `n_layers`
at 443472 on this platform, and only the corrected field list reproduces both. (The Python
gate itself was not executed here; this node does not run Python. It needs a run where it is
allowed.) `tools/gguf_quantize.c` ignored unknown flags and accepted `--only` with no value,
which made a typo look like a successful conversion — both are errors now; and `--requantize`
read as "any dtype that is not float", which let types `gguf_dequant_row` has no reader for
past the shape test, so it names the five it can decode. Wording in the previous entry was
loose in three places and now says: populating is the default *where the platform has*
`MAP_POPULATE`, `USE_METAL` compiles the mapping out rather than failing to build, and the
two load timings come from two different runs rather than being one disputed number.

---

## 2026-09-01 — the model is mapped, not read

`gguf_open` allocated the whole tensor block and read the file into it. The file passes through
the page cache on its way into that buffer, so a 2.6 GB model occupies 5.2 GB at the moment of
load, and two processes running the same model repeat every byte of it. A mapping *is* the page
cache: neither cost exists.

The first version of this entry, and of the commit that carried it, named a different reason —
that the copy is anonymous memory and therefore swapped and decompressed under pressure — and
tied the day-to-day variance to it. **That was checked afterwards and is wrong.** Under 3 GiB
of deliberate pressure, with 2.3 GB genuinely moved into swap system-wide, the inference
process reported no swapped pages of its own on either path: weights touched every token stay
hot, so the kernel takes something colder. The correction stands in the record rather than
being quietly dropped, because the number that motivated the work is still unexplained (below).

The tensor block is now a read-only `MAP_PRIVATE` view. The data section is 32-byte aligned
and a mapping must start on a page, so it maps from the page below and hands out the offset
view; `data_size` on this path is the exact byte count rather than the page-rounded one, which
makes the bounds checks in `gguf_dequant_row` stricter, not looser. `map_base` and `map_len`
carry what `munmap` needs. Any refusal from the kernel falls back to the read path, which is
kept whole.

Populating is the default where the platform has `MAP_POPULATE`; where it does not, the
mapping is lazy and the paragraph below describes what that costs. Which to prefer took a
measurement to settle. Faulted lazily the model
arrives one page at a time inside the first forward pass, which is prefill: 7.5 t/s against
12.3 on an 11-token prompt. `MADV_WILLNEED` did not move it — 7.4 / 7.5 / 8.2 with the call
against 7.0 / 7.1 / 8.3 without — so the call is not in the tree. `MAP_POPULATE` pays the same
cost at load as one sequential stream and removes it from prefill entirely, 12.5 / 12.3, while
still reaching first output faster than reading did because nothing is copied: **1.90 s
against 4.13** in one run and 2.09 against 3.62 in another. The cost is fixed rather than per token, which is why all three variants are
indistinguishable on a 577-token prompt — 10.0, 10.0, 10.1 — and why the short prompt is the
only place the choice shows. `NT_GGUF_MMAP=lazy` skips populating for a model larger than RAM:
peak RSS **1 557 448 kB against 2 851 072**, at the price named above.

Decode moved nowhere, which is the expected result and worth stating: 8.6 to 8.9 t/s across
all eight runs of a cold-start A/B, mapped and read alike. Prefill with the file already in
cache favours the mapping, 12.5 / 12.9 against 12.2 / 8.5, because mapping is free there and
`fread` still copies 2.6 GB.

**The open number.** Decode on this Gemma file was 10.2 t/s on 2026-08-31 and 8.6–9.0 on
2026-09-01, and nothing found so far accounts for it. Eliminated, each by measurement rather
than by argument: a code regression (the same revision built clean from `53eef56` gives
8.9 / 8.8 / 9.0 against the branch's 8.8 / 9.1 / 8.9); generation length (`-n` 8, 16, 24, 32 →
9.0, 8.7, 8.7, 9.0); the prompt (six of them, 7.9 to 8.3 warm); temperature (a 33 °C start
gives the same 8.6–9.0); memory pressure (2 GiB and 3 GiB change nothing); the weak
`openblas_set_num_threads` hook (no effect — BLAS left the token loop when the per-layer
projection was quantized); transparent huge pages (`[never]` in this kernel, so never a factor
on either day). The reference reproduces itself across the same two days, 10.92 and 10.83.
Either something in the environment is still unidentified, or the earlier figure was measured
wrong. It is recorded as open rather than explained.

Metal keeps the read path unchanged. `nt_metal_register_base` wraps `data` as a NoCopy
MTLBuffer and needs a page-aligned pointer with a page-rounded length, which an offset view
into a file does not provide, so `USE_METAL` compiles the mapping out — `gguf.c` itself builds
as it always did, with `NOTORCH_GGUF_MMAP` set to 0.
`NT_GGUF_MMAP=0` forces the read path anywhere else.

The gate was checked by breaking it: shifting the mapped view four bytes makes the same model
answer `!!!!!!!!` where it answered ` Paris.`, and parity catches it. Suites 49, 73, 34 and 3
pass, parity is identical on three prompts, tokenizer on 16.

---

## 2026-09-01 — the quantizer learns to take quantized input, and Gemma's head halves

Gemma 4's output projection is its embedding table, so the whole of `token_embd.weight` is
read once per decoded token. `llama-quantize` deliberately keeps that tensor at a higher
precision than the rest — in the file on this node everything is q4_0 and the head is q8_0,
408 MiB against the 876 MiB the FFN reads. Dropping it to q4_0 was arithmetic worth testing
and the tool could not do it: it accepted f32 and f16 sources only, and Gemma's float original
is a 9.3 GB bf16 checkpoint that does not fit in this phone's 8 GB, since `gguf_open` reads
the data section into memory whole.

Two flags on `tools/gguf_quantize.c`. `--requantize` accepts an already-quantized source,
which loses what the first quantization lost and then some, and is the only path left when the
floats are gone. `--only SUBSTR` converts just the tensors whose name contains the substring
and copies every other one byte for byte, so a single tensor can be priced on its own. Also
bf16 (type 30) in the sizer and the type namer — without it the tool aborted on
`per_layer_model_proj.weight`, the file's one bf16 tensor, which is copied through untouched
rather than converted.

    ./gguf_quantize gemma-4-E2B-it-Q4_0.gguf out.gguf q4_0 --requantize --only token_embd

Head 408 → 216 MiB, file 2710 → 2518 MiB. Decode on the harness, three runs each, interleaved,
from a cold phone: **8.8 / 8.5 / 8.6 t/s against 9.5 / 9.2 / 9.0, plus 7.0 percent**, prefill
unchanged at 12.0 against 12.1. The prediction was 13 percent and it did not arrive: a q4_0
matvec spends more on unpacking than a q8_0 one, so the head went 25.4 ms to about 16, not to
12.7. Quality, `llama-perplexity` over 16 chunks of 512 on the same corpus: 64.8869 ± 4.234
against 65.2808 ± 4.251 — six tenths of a percent, inside the interval. `llama.cpp` on the same
two files moves the same way on decode, 10.83 ± 0.38 to 11.48 ± 0.02, and pays 5.5 percent of
prefill for it, 16.61 to 15.70; we pay nothing there.

Two review findings from the Gemma bring-up, both real, both in `harness/arch_gemma4.c`. The
per-layer projection ignored the return of `nt_qmatvec_i8`, so a failure would have fed the
residual whatever was in the scratch buffer; it now zeroes the section and says so once.
And the loader published `ple_proj_q` before checking that its rows and columns were set,
leaving a non-NULL pointer describing a zero-sized matrix on the allocation-failure path;
the pointer is now published only after a probe matvec returns 0.

One measurement note worth more than the patch. Decode on this file does not reproduce across
days: 10.2 t/s on 2026-08-31, 8.9 today, from the same commit — built from `53eef56` in a
clean tree and A/B'd against the branch, 8.9 / 8.8 / 9.0 against 8.8 / 9.1 / 8.9, no
difference. `llama.cpp` reproduces its own number (10.92 then, 10.83 now). The asymmetry is
`gguf.c:179` — `posix_memalign` plus `fread` of the whole data section into anonymous memory,
which under memory pressure goes to swap and comes back decompressed, and during load holds
the file twice: once in the page cache, once in our copy. With `Cached` at 1.9 GB against a
2.64 GB model and 2.55 GB already swapped, part of every token streams. The reference mmaps.
So should we — separately, with the Metal NoCopy alignment guarantee kept intact.

---

## 2026-08-31 — Gemma 4 gets faster, and neither reason was in the kernels

The profile refused the obvious guess. Attention was 0.9 percent of a decode, not the
bottleneck it looked like from the code; the FFN matmuls were 48.9, the head 18.6, and the
per-layer embedding block 10.0 with its projection another 5.7.

**Two thread pools on four cores** (`46fd424`). The per-layer matvecs took 237 us inside the
model where the same shape measures 41 us on its own. Not cold weights — pointing every layer
at one matrix changed nothing. Not an f32 fallback — every weight reported the packed path.
OpenBLAS was the answer: it starts a pool of its own, woken by the single sgemm this family
runs per token for its f32 projection, and its workers spin exactly like notorch's do. Four
of theirs and four of ours, taking turns evicting each other. One thread for BLAS, set at
startup through a weak symbol: decode 9.5 / 8.6 / 9.2 t/s against 7.5 / 6.7 / 6.5, thirty
percent, with prefill unchanged and Qwen — which touches BLAS nowhere — the same within noise.

**One weight read in floats** (`c7fa6a6`). `per_layer_model_proj` is the file's only bf16
tensor and was expanded to f32 at load: 55 MB read in full per token against 15 MB for the
rest of that path. Quantized to Q8_0 at load with this tree's own quantizer, row by row, and
read afterwards by the same integer matvec as everything else. The section holding it goes
227 / 160 / 215 ms to 24 / 20 / 15 over sixteen tokens, and the last BLAS call leaves the
token loop. Q8_0 is not bf16 and the commit says so: scale per 32 values, normalized right
after, and every other matrix in this file is Q4_0.

Where Gemma 4 stands on four big cores: decode 10.2 / 10.1 / 8.7 t/s against 7.0 this morning,
prefill 23.5 / 19.9 / 22.3 against 14.9. llama.cpp on the same file does 10.92 and 17.84 —
decode within seven percent, prefill past it.

The profile is flat in the way that means finished: 51.6 percent FFN and 26.5 percent head,
both at the memory ceiling — 876 MB of FFN weights per token in 49.6 ms is 16.4 GiB/s of the
17.98 this phone streams, the head 428 MB in 25.4 ms is 15.7. The next lever is fewer bytes,
not faster code, and that is a decision about the file rather than the loop.

---

## 2026-08-31 — K-quants, and the harness re-measured against the example it came from

**The harness carries the arithmetic and the speed** — the check the phone owed it after the
forward moved out of `examples/infer_llama.c`. Parity on Qwen2.5-0.5B Q5_0, three prompts at
temp 0: identical output, `NOTORCH_PARITY_OK`. Speed on the same 121-token prompt, three
interleaved repeats each: harness prefill 92.8 / 95.6 / 94.8 t/s against the example's
96.8 / 95.9 / 96.8, decode 41.6 / 41.8 / 40.5 against 39.2 / 40.4 / 40.3. Within noise both
ways.

One difference that is not noise and is in the harness's favour: the example clamps its
context to 256 tokens and silently truncates a longer prompt — a 241-token prompt arrives as
207. The harness takes `NT_CTX` and encoded all 241. Anyone comparing prefill numbers between
the two on long prompts was comparing different amounts of work.

**Q4_K and Q6_K now quantize** (`09803e1`). The block formats take an absmax; a K-quant
searches — Q6_K walks eighteen candidate scales per 16-value sub-block and keeps the lowest
weighted error, Q4_K solves a weighted least squares for scale and minimum across twenty-one
candidates — and then quantizes those scales to six bits against a super-block scale. Ported
from ggml's reference (MIT) rather than reinvented. `gguf_quantize` mirrors llama-quantize's
fallback for rows 256 does not divide, observed rather than assumed: `--pure Q4_K` writes 24
tensors of this model as q4_K and 146 as q5_0, `--pure Q6_K` writes 24 as q6_K and 146 as
q8_0. File sizes match to the byte: 426350432 and 650379104.

**Where byte-identity stops, and why that belongs to the algorithm rather than to us.** A
K-quant search runs on multiply-adds. Whether the compiler fuses them decides which candidate
wins a near-tie, so two correct builds disagree by one level in one sub-block. Against ggml's
reference *source* compiled with contraction off, our output is identical — 64 real rows,
19 super-blocks each, both formats, byte for byte. Against the shipped llama-quantize
*binary*, 24 tensors of 291 differ, and nothing in our code can fix that. The block formats
have no such expression in their critical path, which is why they still match a binary
exactly, and they still do.

The test therefore gained the gate that survives a compiler: for K-quant tensors that differ,
dequantize both and compare reconstruction error against the original f16 weights — ours must
be no worse. Q4_K mean rms 8.503983e-04 against 8.503976e-04, Q6_K 2.169588e-04 against
2.169600e-04. Perplexity agrees: Q4_K 14.8192 ± 0.480 against 14.8334 ± 0.481, Q6_K
13.8092 ± 0.440 against 13.8219 ± 0.441, over 32 chunks of 512 from wikitext-2.

The ladder on this phone now reads, decode of 64 tokens on four big cores: Q8_0 33.0,
Q6_K 32.0, Q5_0 45.6, Q4_K 45.8, Q4_0 57.0 t/s.

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
