# Receiving-body votes and addressed sleep selection

This study continues the turn-164 key-replay population from
[long continuity](continuity.md). The C Harness supplies explicit vote
calibration and a read-only response measurement for sealed replay episodes.
AML schedules the experiences, compares selection policies and commits the
population together with its episode-round history.

## Bodies, inputs and arithmetic

The Gemma3-270M Q8_0 body, Qwen3.5-0.8B BF16 text conversion and tokenizer
are the exact files identified in [the preceding study](continuity.md#bodies-and-arithmetic).
The machine is AMD EPYC 7763, Linux x86_64, GCC 14.2.0, native
AVX2/FMA/F16C without BLAS. Qwen's loader expands its BF16 matrices to F32.
Every real run uses `NT_NO_I8=1`, `NT_ATTN_THREADS=1`, `NT_CHAT='||'`, empty
`NT_TOKENS`, `NT_CTX=128` and `LC_ALL=C`. AML is revision
`78caa5b15f732d7fb83c950e544c58d23735a615`.

| Input | SHA-256 |
|---|---|
| Gemma turn-164 key-replay population | `fcbd3d4ff573f49de1de5aab858e100340e920153b57509cb019a38a6252d932` |
| Earlier Qwen local memory, shared step 12 | `d4c8446b0816c33baf5ea5c9c5d50b0b85013bc2f11f2f9c19dafcdfa5df3bd2` |
| Earlier Qwen captured episode, shared step 12 | `0e2c4028deacb879b3cac6c042cb42cb690b3972461c55afd7b0fdf76e541367` |
| `vote_sleep.aml` protocol | `e3a6faf52608d6f9fa5ed932e4b9c1ef4ed66d310e4aeb9eb52cfd503ff7c0ad` |

## Separating vote scale from preference

[`vote_scale_audit.c`](vote_scale_audit.c) samples 32 raw anchor residuals
before QR and every position of eight fixed texts. It restores the early
turn-36 and late turn-164 populations independently, retains their saved
consensus/history, and feeds disposable copies into the native election hook.
No learning occurs. The native library and helper were rebuilt with the same
compiler flags as the experiment.

Both populations have byte-identical vote tensors. Their living centered
anchor score RMS is **45.3926721746**, so target one uses the common scale
**0.0220299874868**. The six vote-row norms range from 0.24374 to 0.25895;
mean input RMS is 255.057 on the anchors and 483.775 over the other 116
positions. Seat zero has the largest raw dot on all 148 positions.

| Population and probe | Original leader mass | RMS-one leader mass | RMS-one effective seats |
|---|---:|---:|---:|
| Turn 36, lighthouse carries | ~1 | 0.628539 | 2.478641 |
| Turn 36, Miro reaches | ~1 | 0.587928 | 2.578196 |
| Turn 164, lighthouse carries | ~1 | 0.721146 | 1.746674 |
| Turn 164, Miro reaches | ~1 | 0.699486 | 1.781973 |

Effective seats is the position mean of `exp(-sum(p*log(p)))`, with gates
normalized by their actual sum. The late population begins with consensus
0.999999523; its short calibrated probes still select k=2. The audit also
retains target 0.5/2, individual vote-row normalization and cosine controls.
Unit vote normalization increases concentration here; cosine moves the
earlier population close to equal mass among its five selected seats.
[Raw scores](vote-scale-raw.tsv) · [Gates](vote-scale-gates.tsv) ·
[Calibration summary](vote-scale-summary.tsv)

The implemented operator uses raw anchor rows, a living-expert centered RMS,
one positive scale for every stored vote row, and a staged finite/bound check.
It changes no A/B, birth addresses, lifecycle values, history, consensus, RNG
or clocks. Values within eight F32 epsilons of unit scale are a no-op. The
existing schemas 5/6 persist the resulting rows directly. Calibration remains
an explicit `--expert-vote-rms` choice after transfer and optional body recall.

## Six continuations and a four-episode bank

[`vote_sleep.aml`](vote_sleep.aml) crosses original/RMS-one votes with three
sleep policies. Every history begins from the same completed turn-164 state;
the three calibrated histories share one explicitly calibrated initial spore.
There are 64 further waking encounters, each with a fresh decoder cache and
one greedy choice plus the ordinary final non-stop feedback. The common
inputs cycle through musician, mountain rain, baker and orchard sentences.
Every eighth encounter is followed by six private greedy replay choices.

The sealed bank is captured once without waking or learning:

| Index | Episode | Content address |
|---|---|---|
| 0 | The earlier lighthouse/key encounter | `07eefcb7f67382960b8b8fdd4e9f73f23f19e42884c24a2fbdd894f734c52e0b` |
| 1 | Musician and wooden floor | `81e8d00fbfdf0fe4f2e402752504cd192f9406e3211f5a6f24bcda3f517ee522` |
| 2 | Mountain rain and stream | `0d96d42cc00293e0d9f3af829a5ade4c7f62992ed8dee1d56ef3f594a425e42d` |
| 3 | Baker and morning market | `3cfb0191b14d0f691ee175a42b9ab75890938a7589fa523400fd31839c0fdf10` |

Fixed replay always selects index 0. Rotation selects the first unvisited
bank entry. Disagreement selects the unvisited episode with the greatest
current normalized dispersion of living A(Bx) proposals, then mean proposal
energy; equal F32 scores use the lexicographic episode address. Both bounded
policies visit all four entries before restarting their round. Candidate
measurements use the entire saved prompt, private KV and the same saved
consensus/history for every candidate. They generate no new token and alter
no owner buffers or persistent state. Native double readings accompany the
F32 AML checkpoint used for selection.

Each night's receipt stores all six parents/finals, the bank and file hashes,
candidate readings, selected indices, previous/next round histories and the
previous receipt's hash. Resume verifies the entire ancestry and recomputes
each selection before creating an output directory. A worker receipt records
the execution mode; a later segment can switch between serial and worker
consolidation while preserving the same numerical protocol.

## The continuation through turn 228

All six histories complete the 64 encounters: turn **228**, **6,123** waking
elections, **36** lifetime replay cycles and **44** birth addresses, with six
living experts and 38 retired ancestors. The previous six ordinary text-dream
cycles remain unchanged. The original Omni leader is still living at its
original address. Original and calibrated vote tensors remain unchanged
through all new waking and sleeping updates.

The two rotating policies cover the same four addresses twice, in different
orders. Rotation selects `0 1 2 3 0 1 2 3`; current disagreement selects
`3 2 1 0 3 2 1 0` in both scale conditions. Fixed replay selects index 0 all
eight nights. The candidate response values are measured again each night.

| Votes and sleep | Final key/bell margin, probe 0 | Final key/bell margin, probe 1 |
|---|---:|---:|
| Original, fixed | 2.336054 | 1.025265 |
| Original, rotation | 2.164680 | 0.885490 |
| Original, disagreement | 2.334709 | 1.025675 |
| RMS one, fixed | 2.334839 | 1.037363 |
| RMS one, rotation | 2.164640 | 0.884760 |
| RMS one, disagreement | 2.334656 | 1.022118 |

The probes and target IDs are those of the preceding Gemma study. They read
all 262,144 logits from discarded copies. In the calibrated condition, fixed
and disagreement histories both finish by replaying the lighthouse episode,
yet their final full rows differ by maxima **0.198219/0.216997**. Rotation
finishes with the baker episode. Its margins differ from calibrated fixed
replay by **−0.170198/−0.152603**. The reported greedy argmax IDs on these final
probe rows remain 41865 and 15123 across all six histories.

On a fresh read of the saved lighthouse episode at turn 228, original fixed
replay gives mean leader mass **0.999877** and effective participation
**1.000852**; calibrated fixed replay gives **0.729265** and **1.761267**.
The requested k is two in both. The final-bank tables also retain all six
per-expert masses and ungated responses for every episode.
[Scores](vote-sleep-scores.tsv) · [Selection](vote-sleep-selection.tsv) ·
[Full-row contrasts](vote-sleep-logits.tsv) · [Final bank](vote-sleep-final-bank.tsv) ·
[Genealogy](vote-sleep-lineage.tsv)

The independent native reader verifies the staged writes at all **48** sleep
boundaries. In the final calibrated disagreement night, copying all living
post-sleep A/B by birth address into the pre-sleep population reproduces all
**524,288** logits exactly while retaining the recipient's clocks and history.
That night's own key/bell margin changes are **+0.142668/+0.097111**.
[Sleep writes](vote-sleep-writes.tsv) · [Clocks](vote-sleep-clocks.tsv)

Both planned interruptions at 32 were exercised from receipt 24. Before
publication, receipt 24 remains the continuation point; after publication,
receipt 32 becomes it. The resumed run switches to worker consolidation and
continues through 64, including the round reset at 40. Against uninterrupted
serial execution, **420 spore pairs**, **126 reading checkpoints**, **42 choice
tapes**, the complete score table and every recorded choice match exactly.
The final uninterrupted receipt SHA is
`5638631be71776c2e90f8d610ccfe0b7da5e0d42ec7ae3f65a29a7aec4db86e8`.
Four corrupted copies are rejected by the real continuation protocol before
creating an output directory: an early state reached through the complete
eight-receipt ancestry, a parent-receipt hash, a bank episode and a recorded
selection. All 997 original input/artifact hashes remain unchanged.

## Transfer and local recall on Qwen

The same Gemma turn-164 population is exported portably and imported into
Qwen3.5-0.8B. Explicit body recall matches two still-living identities in its
earlier local spore. In this body the centered anchor RMS is **0.025481375**;
target one multiplies vote rows by **39.244350375**. On the saved 29-position
Qwen episode, the original mean maximum gate is 0.300254 and effective
participation is 3.826892. After target-one calibration they are 0.611641
and 1.905977; mean selected k changes from 3.827586 to 2. Ungated response
and disagreement are identical in both reads because their A/B are unchanged.

These two bodies therefore give opposite participation changes for the same
target. The target is explicit and the measured scale is reported. Staged
import/recall/calibration and their combined CLI invocation produce identical
Qwen spores; an ordinary restart preserves the calibrated file, and its
six-choice serial/worker replay produces identical final spores.
[Qwen episode readings](vote-qwen-bank.tsv)

## Native verification

`make -j2 test BLAS_FLAGS= BLAS_LIBS= BLAS_NAME=native` passes, including
110 calibration checks, 517 replay-reader checks and the CLI flag/transfer/
restart/worker gate. The independent coordinate decoder covers uniform and
one-hot gates, equal and opposed proposals, zero input, input scale, episode
length, sensory rows on both sides of prefill boundaries, late failure and
all owner buffers. Fully instrumented native libraries pass those 110/517
checks under ASan/UBSan with leak detection disabled in this environment.

Separate mutants make the gates fail: omitted vote publication (70 failures),
total energy substituted for centered disagreement (6), and borrowed live
injection scratch (11). AML selection controls exercise 54 expected choices;
a repeated-address mutant is rejected. The audit's three real TSV outputs
reproduce byte for byte. Its file/argument errors and the probe's `/dev/full`
failure are exercised explicitly.

The unchanged default path reproduces the prior Gemma waking/replay states.
On Qwen it reproduces plain output and choice/logit traces, portable local
recall, captured input, worker waking and worker replay states exactly.

## Reproduction

Build the native Harness and optional AML adapter:

```sh
make -j2 notorch doe_experience doe_probe \
  BLAS_FLAGS= BLAS_LIBS= BLAS_NAME=native AML_ROOT=/absolute/path/to/aml
```

From `experiments/doe_spores`, feed `vote_sleep.aml` twelve lines: absolute
paths to `notorch`, `doe_experience`, `doe_probe`, host, tokenizer or `-`,
initial exact spore, a new output directory; final ordinal; resume receipt or
`-`; `sync` or `worker`; planned stop ordinal (`0` for none); stop location
(`none`, `before`, `after`). The final ordinal is a multiple of eight through
4096. The documented run uses 64. A stop before publication leaves the prior
receipt authoritative; a stop after publication resumes from the new one.

Feed the final receipt and a new report directory to
[`vote_sleep_report.aml`](vote_sleep_report.aml) for chronological score and
selection tables. It verifies the files before reading them. The initial
parent and earlier body files are retained in the preceding experiment
archive; the current experiment retains its runtime binaries, full vocabulary
rows, every completed state, replay traces, bank, receipts and recovery runs.

For the scale ablations, compile `vote_scale_audit.c` against
`libnotorch_harness.a libnotorch.a -lm`, then pass the Gemma host, turn-36
parent, turn-164 parent and a new empty output directory in that order.
Use the environment above and no BLAS. Its three TSV files reproduce the
committed raw scores, gate readings and calibration summary.
