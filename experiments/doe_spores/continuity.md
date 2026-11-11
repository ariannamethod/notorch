# Long histories and a shared parliament

Two native C/AML protocols continue the six addressed experts saved after the
[36-turn Omni experience](omni-replay.md). One follows six histories on Gemma
for 128 additional encounters. The other alternates Gemma and Qwen with a
single continuing parliament, with separate histories as its control. Each
measurement restores a discarded copy of a completed generation.

## Bodies and arithmetic

| File | Source | SHA-256 |
|---|---|---|
| Gemma3-270M Q8_0 | `ggml-org/gemma-3-270m-it-GGUF@e7647be17ae1108f2f605ed061ca0608b171afff` | `0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3` |
| Qwen3.5-0.8B BF16 text conversion | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` | `f277970417e643a86ccd426e16f8a567c99d8f0ee367e4613485c4ea96847923` |
| Qwen tokenizer GGUF | same source revision | `2f38049769a7fb0d2b70b5ed329563d932cc366899379f35476419021260a4f4` |
| Prior Omni portable population | turn 36 | `3c7fcf27d7e45e5543d7220fa2227b9598fb2a09c247a80056c3b2f9878d227e` |
| Common Gemma parent | one-time native adoption of the Omni population | `4aa0cdf1bcac11f6c3327d8ec0ebac6cff1b412a83ac6b35858e0bf28cbe1679` |

Gemma has width 640 and 18 layers; Qwen has width 1,024 and 24 layers.
Their expert layers are 9 and 12. Qwen's BF16 matrices are widened to F32 by
its native loader. The common parent retains 36 waking turns, 3,348 waking
elections, six text dreams, twelve sensory replays and the six adoption IDs.
Its initial Omni-to-Gemma projection used the default activation path.
Both new protocols explicitly set `NT_NO_I8=1`, `NT_ATTN_THREADS=1`, raw
`NT_CHAT='||'`, empty `NT_TOKENS`, `NT_CTX=128` and `LC_ALL=C`.

The machine is AMD EPYC 7763 with GCC 14.2.0, native AVX2/FMA/F16C and no
BLAS. AML revision is `78caa5b15f732d7fb83c950e544c58d23735a615`.
Inference, elections, updates, frame projection and replay run in NoTorch C.
AML schedules the encounters, retains full-vocabulary readings, computes
softmax/margins and publishes the receipt chain.

## Six histories through 128 new encounters

[`long_memory.aml`](long_memory.aml) crosses the early content (`key` or `bell`)
with three schedules: waking only, a six-choice text dream every eighth
encounter, or six-choice replay of the first encounter every eighth encounter.
The first four inputs in each history say:

> Miro is the lighthouse keeper. Every night Miro carries a key. At sunrise the lighthouse keeper returns home.

The paired history substitutes `bell`. Encounters 5–128 are identical across
all six histories: a four-sentence rotation about a musician, mountain rain,
a baker and an orchard. Every waking encounter has a fresh cache, one greedy
choice and the ordinary Harness's final non-stop feedback. The C resident
runner holds a body for blocks of at most eight encounters, publishes every
turn, and reopens the saved population for the next block.

The first input is captured in all arms before learning. Its replay always
starts fresh private decoder state. The waking turn, optional sleep, and
post-sleep checkpoint are separate boundaries. Waking lifecycle can retire
an expert and create a descendant; private sleep preserves that membership.

Every eighth encounter, two discarded raw-text probes retain all 262,144
logits each:

- `Every night the lighthouse keeper carries a`
- `At the lighthouse, Miro reaches for his`

The target continuations are ` key` and ` bell` (Gemma IDs 2307 and 19974).
The helper verifies that appending either surface adds exactly its single
token. The selected margin is `logit(key) - logit(bell)`.

### Completed boundaries and recovery

Each receipt commits the latest state of all six histories together. It
records the protocol, native executables, body, tokenizer and root hashes;
the exact ordinal interval; every generated state, captured episode, input,
trace and reading; and the preceding receipt's path and digest. The receipt
is published after the files it names. A failed block can leave inspectable
files without advancing the committed head.

Resume verifies every receipt and every referenced artifact back to the
root before claiming a new output directory. It starts after the selected
completed ordinal and preserves the global eight-encounter sleep cadence,
including when an earlier process stopped between sleep boundaries. All
external paths are absolute. To move a raw archive, preserve its original
path layout for receipt validation, or run a fresh history in the new layout.

The native population inspector writes uint64 clocks as decimal strings in
AML checkpoints. The experiment parses only its bounded schedule and counters;
the durable counters retain their full integer representation.

### Reproduction

```sh
make notorch doe_experience doe_probe AML_ROOT="$AML_REPO" \
  BLAS_FLAGS= BLAS_LIBS= BLAS_NAME=native

# Establish the exact common parent; OUT_PARENT must be a new file.
./notorch --experts --spore-in "$OMNI_TURN36" --spore-only \
  --portable-spore --spore-out "$OUT_PARENT" "$GEMMA"

# All paths below are absolute; OUT must be a new directory.
"$AML_REPO/runner/aml" experiments/doe_spores/long_memory.aml <<EOF
$NOTORCH_ROOT/doe_experience
$NOTORCH_ROOT/doe_probe
$GEMMA
-
$OUT_PARENT
$OUT
128
-
sync
0
none
EOF
```

For a worker history, replace `sync` with `worker`. To exercise recovery,
use final ordinal `8`, planned stop `4`, and `before-receipt` or `after-receipt`.
The planned stop deliberately exits nonzero. An after-receipt stop leaves
`receipt-4.amlck`; resume with that absolute path in the eighth input line,
a new output directory, final ordinal `8`, stop `0` and `none`. A stop before
the first receipt restarts from the original root in a new directory.

The protocol permits up to 4,096 new encounters; this recorded run uses 128.
The archive also retains an eight-encounter bring-up with default activation
quantization, run before the final protocol fixed `NT_NO_I8=1`. Its receipt
belongs to that earlier protocol hash.

### Results at turn 164

All six histories completed 128 new encounters: 768 waking updates in total.
Every final state has 5,211 waking elections and a 30-entry birth ledger with
six living identities. Two of the original Omni experts remain alive; the
other four living experts are addressed descendants. The ledger preserves
all 24 retired identities. The waking histories retain the original six
text dreams and twelve sensory replays. Dream histories reach 22 dreams;
spaced-replay histories reach 28 replays.

| History | Probe 0 margin | Probe 1 margin |
|---|---:|---:|
| key / waking | 2.190685272 | 0.926439285 |
| bell / waking | 2.190898895 | 0.926586151 |
| key / dream | 2.155904770 | 1.061639786 |
| bell / dream | 2.155904770 | 1.061639786 |
| key / early replay | 2.336042404 | 1.025087357 |
| bell / early replay | 2.273895264 | 0.943239212 |

The early-replay contrast at 128 is +0.062147141 and +0.081848145. Both
full output rows differ at all 262,144 entries, with maximum absolute
changes 0.067641139 and 0.075785875. The ordinary-dream histories have exact
full-row equality at every measured checkpoint from 80 onward and exact
final A/B and vote tensors; their historical birth addresses remain distinct.
The waking-only pair retains small negative key-minus-bell contrasts at 128. The full table
preserves the nonmonotonic trajectory, including earlier changes of sign.

The last key dream changes the selected margins by −0.043127060 and
+0.127323151; the last key replay changes them by +0.142652512 and
+0.096712112. Transplanting the living experts' post-sleep A/B by birth ID
into the corresponding pre-sleep population reproduces both complete output
rows in each case: 524,288 exact F32 values per intervention. Recipient
clocks and genealogy remain those from before sleep. An independent native
reader verifies elected-seat A/B/cursor writes and preservation of the other
tensors at all 64 sleeping boundaries.

There are 864 archived spore files across six histories, totaling
123,375,648 bytes. The key-replay history has 144 files and 20,562,304 bytes;
its final exact generation is 145,376 bytes. Episodes, traces and full-logit
checkpoints are stored alongside those generations.

Recovery controls compare 120 exact states against the main run: 24 orphan
states before a planned publication stop, 48 states from stop-after/resume,
and 48 from a fresh worker run. All twelve final score checkpoints match.
A separate late worker run reproduces all 48 states of ordinals 121–128,
all six pre-sleep states at 128 and all six choice tapes. Corrupting an early
state or its parent receipt independently causes refusal before output
directory creation; the control restores the original files afterward.

[Scores](long-scores.tsv) · [Full rows](long-logits.tsv) ·
[Clocks](long-clocks.tsv) · [Living seats](long-seats.tsv) ·
[Final genealogy](long-lineage.tsv) · [Generation hashes](long-generations.tsv) ·
[Sleep writes](long-sleep-writes.tsv)

### Distribution of the votes

The trace reader covers 13,979 elections: 1,649 probe positions (including
17 from the initial parent), 11,178 waking positions and 1,152 private
positions. It normalizes each printed six-element gate row by its measured
sum, checks finite values and sums, and reports the per-position average of
`exp(−Σ p log p)`. Uniform-six and one-hot analytical controls give effective
counts six and one. Reported top-k, positive gate count and normalized mass
are retained separately.

Seat 0 wins all 13,979 rows with no ties. Its original Omni identity remains
alive at every saved boundary. Before the long continuation, its mean probe
mass is 0.999999945885 and effective participation is 1.000000843948.
At checkpoint 128 these are 0.999999948545 and 1.000000804848 in all six
histories. Across waking encounters its mean mass is approximately 0.999707.
This concentration is present in the imported parent. The current C lifecycle
counts `gate > 0` occurrences, and the vote vectors remain unchanged during
waking updates and sleep. The inherited vote geometry, receiving residual
scale and episode-selection policy are therefore explicit coordinates for
the next controlled tuning experiments.
An additional factor intervention copies only that living seat-0 identity
from the last post-sleep state into its pre-sleep recipient. Relative to the
complete post-sleep output, the maximum absolute residuals are 1.38283e-5 and
1.04904e-5 for the ordinary dream, and 1.33514e-5 and 1.43051e-5 for replay.
The corresponding selected-margin residuals are zero/zero and
−1.90735e-6/−1.90735e-6. Copying all living IDs gives exact full-row equality.
[Aggregate gates](long-gates.tsv) · [Probe series](long-gate-series.tsv)

## Two bodies, four histories

[`shared_parliament.aml`](shared_parliament.aml) first generates a complete
peer tape with plain Gemma and Qwen alternating, six greedy choices each.
The first prompt is `Miro is the lighthouse keeper. He carries a key.` Each
following prompt includes the preceding body's UTF-8 utterance and asks for
the continuation. The protocol removes the CLI's verified prompt echo and
terminal line feed, preserving the generated text itself. Every arm then
receives this same frozen tape, encoded by the receiving body's tokenizer.

| Arm | Incoming population | Returning-body reference | Consolidation |
|---|---|---|---|
| independent-sync | that body's own last generation, initially the common root | none | serial |
| shared-sync | the preceding body's latest generation | none | serial |
| recall-sync | the preceding body's latest generation | that body's own last generation | serial |
| recall-worker | same as recall-sync | same as recall-sync | worker |

Each step publishes an entry spore before learning, one waking generation,
the captured input, and a generation after six private replay choices.
Portable frames accompany the generations. Returning bodies in the recall
arms use `--body-memory` explicitly. All four captures must have identical
bytes; recall serial/worker entry, waking and replay states must match before
the four-arm receipt is published. Body banks advance only after publication.

The shared population carries one waking/replay clock across the alternation.
The independent arm advances each body's own clock on that body's visits.
Receipts retain exact incoming and local-reference hashes, both body banks,
the frozen input and all intermediate generations. Birth IDs select matching
living experts; they do not replace these state-level parent references.

At the first Gemma return (step 3), an additional reading compares shared and
recalled entry states before learning. The common incoming history is then
identical, so this comparison isolates body recall. Later scheduled readings
measure the accumulated differences after the paths have diverged.

```sh
"$AML_REPO/runner/aml" experiments/doe_spores/shared_parliament.aml <<EOF
$NOTORCH_ROOT/notorch
$NOTORCH_ROOT/doe_probe
$GEMMA
$QWEN
$QWEN_TOKENIZER
$OUT_PARENT
$SHARED_OUT
12
EOF
```

The step count must be even, from 2 through 16. The frozen tape and its hashes
are saved before the expert histories begin. This protocol establishes the
serial handoff and return operation for a subsequent live speaker scheduler.

At step 3 the shared and recalled incoming parent files are identical. Native
audit matches all six identities at Gemma width 640. Across 18 A/B/vote
measurements, the largest shared-coordinate residual is 9.289242986e-10,
the largest local-orthogonal residual is 1.711326736e-9, and the largest
independent formula residual is 1.839302585e-9. Incoming state outside these
vectors is preserved. Before new learning, recall changes the two probe
margins by −0.004350662 and −0.004522324; maximum full-row changes are
0.012924671 and 0.012619019.
[Recall vectors](shared-recall.tsv) · [Entry logits](shared-entry-logits.tsv)

At their last visits, the three serial histories give the following selected
`key` minus `bell` margins. Gemma is measured at step 11 and Qwen at step 12;
each column compares histories on the same body and probe.

| History | Gemma probe 0 | Gemma probe 1 | Qwen probe 0 | Qwen probe 1 |
|---|---:|---:|---:|---:|
| Independent per-body populations | 3.781076431 | 3.338155746 | −0.823555946 | 1.379212379 |
| Shared population | 3.867132187 | 3.437553406 | −0.823804855 | 1.380080223 |
| Shared with body recall | 3.886407852 | 3.462926865 | −0.823807716 | 1.380134583 |

At step 12 the shared histories retain all six adoption identities and reach
turn 48, 3,703 waking elections and 24 replay cycles. The independent Qwen
population reaches turn 42, 3,532 waking elections and 18 replay cycles,
having experienced its own six visits. The shared histories carry the work
done on both bodies into the next generation.

The recall-worker history completes the same twelve encounters. Its entry,
waking and final spores match recall-sync in all 36 pairs. The four arms'
captured inputs match at each step (36 pairwise comparisons), and all eight
serial/worker full-logit rows match at the four scheduled checkpoints.
An independent AML reader checks 973 file hashes and all twelve receipt
predecessors, frozen prompts, population parents and body-bank references.
[Scores](shared-scores.tsv) · [Full rows](shared-logits.tsv) ·
[Clocks](shared-clocks.tsv) · [Genealogy](shared-lineage.tsv) ·
[Generation hashes](shared-generations.tsv)

## Independent readers and regression gates

```sh
cc -O2 -std=gnu11 -pthread -I. experiments/doe_spores/recall_audit.c \
  libnotorch_harness.a libnotorch.a -lm -o recall_audit
./recall_audit "$GEMMA" \
  "$SHARED_OUT/shared-sync-3-entry.gguf" \
  "$SHARED_OUT/recall-sync-1-final.gguf" \
  "$SHARED_OUT/recall-sync-3-entry.gguf"

cc -O2 -std=gnu11 -pthread -I"$AML_REPO/core" \
  experiments/doe_spores/logits_audit.c "$AML_REPO/libaml.a" -lm -o logits_audit
./logits_audit "$LEFT_SCORES" "$RIGHT_SCORES" comparison

make check_expert_recall check_text_replay check_experience \
  BLAS_FLAGS= BLAS_LIBS= BLAS_NAME=native

cc -O2 -std=gnu11 -pthread -I. experiments/doe_spores/long_gate_audit.c \
  libnotorch.a -lm -o long_gate_audit
./long_gate_audit --selftest
# GATE_OUT exists and has no files named by this reader; the parent trace
# comes from doe_probe score on OUT_PARENT with the same two raw probes.
./long_gate_audit "$OUT" "$GATE_OUT" "$PARENT_SCORES_TRACE" \
  "$OMNI_TURN36" "$OUT_PARENT"
```

The recall reader independently evaluates every A column, B row and vote
vector, reports shared-coordinate and orthogonal-component residuals, and
requires unchanged incoming lifecycle, clocks, RNG and genealogy. Its scalar
formula oracle does not call the projector. The logit reader compares both
entire F32 rows and reports changed entries, maximum absolute change, L2,
greedy indices and the selected margin difference.

The recall regression includes displaced slots, new and retired IDs,
same-width wrong hosts, partial turns, rollback after a late invalid vector,
256 repeated recalls and eight persisted A → B → A cycles. The text replay
gate compares against an independent native API caller, including assembled
chat IDs, stopping policy and sparse native routes. The resident runner gate
compares each state with repeated ordinary CLI calls and a split 4+4 run.
