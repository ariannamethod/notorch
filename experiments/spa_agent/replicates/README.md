# SPA measures repeated future consequences

This experiment changes one axis: how many stochastic continuations measure
an action's consequence. It keeps the 267-parameter policy, captured features,
reward, row order and fitting budget from the [future-credit experiment](../future/README.md).
Each sentence state supplies eight paired continuations. The first, `r0`,
reproduces the previous continuation exactly.

## Why repeat the consequence?

The preceding H4 policy changed 17 of 48 evaluation decisions: six interventions
were beneficial and eleven harmful. Its fixed-checkpoint mean Huber loss fell
24.04% on training states and rose 3.23% on the exposed evaluation states.
Position accounted for 90.6% of the LEFT prediction variance, compared with
2.69% of the measured LEFT advantage variance. One training outcome at
seed 42, snapshot 11 changed target-1 LEFT's mean advantage from negative to
positive. These are retained diagnostics, with per-state evidence in
[`diagnosis.json`](diagnosis.json).

[`conditioning.json`](conditioning.json) inspects the same four sealed policies
and 96 exposed states. The hidden layer is unsaturated, while embedding inputs
have much less local influence than the other features. Fitting budget, row
order and feature conditioning remain candidates for separate experiments.
This run holds them fixed and measures variation across generation RNG first.
Both diagnostic scripts perform analysis without training or generation;
the conditioning script uses NumPy for its descriptive Jacobian SVD.

## Frozen design

[`protocol.json`](protocol.json) was frozen before new rollouts and experiment
fitting. Its SHA-256 is
`c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75`.

The body is the same frozen **450,688-parameter SimpleLLM Dracula transformer**:
width 128, two layers, four heads, FFN 384, context 64, and 94 Unicode characters.
Its source is `ariannamethod/notorch-simple-llm` at
`80b3bd611ed8c937efdc481dc07895c8e13e345d`; checkpoint SHA-256 is
`22ce6e5dfe4efb4a00477a9652784970dc53a69fea2825174398af749acda00d`.
Generation uses upstream notorch arithmetic, temperature 0.8, 12–64 generated
characters, and punctuation termination. Reseeding uses the neighbour's last
three characters.

The two F32 CPU executions use an AMD EPYC 9V74, GCC 13.3.0 and Python 3.12.14.
Native builds use `-O2 -DUSE_SIMD -march=native -pthread`. The available CPU
pool limit is nine; small matrices retain the upstream serial dispatch path.
The command, memory snapshot, sources, inputs and compiled identities are
recorded in [`receipts.json`](receipts.json).

Training uses 48 retained states at seeds **42/73**, in their original order.
Evaluation uses 48 new states at seeds **307/401**. Seeds 101/211 belong to the
exposed preceding experiment and diagnostic analysis. Each source state keeps
its complete sentence field, reseed counts, 29 captured float features,
source-life identity and RNG witnesses. The imported history features retain
their original values; the snapshot host independently recomputes the 19
sensory features and all seven before-measurements from the sentence field.

| Policy | Consequences used | Updates |
|---|---|---:|
| initial | None | 0 |
| single_h4 | Original `r0`, horizon 4 | 24,576 |
| mean8_h4 | Mean native reward over `r0..7`, horizon 4 | 24,576 |
| shuffled_mean8 | Eight outcomes from the next state at matching target coordinates | 24,576 |

Every fitted arm starts from the same native seed-1 life and receives 512
epochs at rate 0.03, one comparison update per state. Online history, counters
and RNG remain fixed. The shuffled donor mapping is the preceding experiment's
cyclic mapping within `(sentence_count, target)` groups; donor provenance
remains separate from the receiver's captured input.

Both complete executions fit and seal all four lives before the first body
process for seeds 307/401. The single-outcome arm must reproduce the previous
H4 life exactly, SHA-256
`3f8fa08f331a1c34d50201e0c5c567ca1b2b1853e6c9b54de265c9886b0d02cb`.
Readout uses the actual `import SPA` binding to native C and consumes source
features only. Its typed choice is then joined to the measured alternatives.

## Paired trajectories and native aggregation

For every `(source, replicate, hop)`, all valid initial actions receive the
same precomputed generation RNG start. Every later hop has an independently
constructed start; earlier token counts cannot shift it. `r0` uses the old
streams. Repeats 1–7 use the two registered RNG domains in the protocol.
The original measurement writer is reused, allowing complete `r0` JSONL lines
to match their parent records byte for byte.

The initial intervention is KEEP, LEFT or RIGHT subject to the source boundary.
Future target is `(original_target + hop) mod 4`, with LEFT except at target
zero, which uses RIGHT. H0, H1 and H4 keep their complete generated chains,
termination flags, cost and separate axes. Measurements refer to the original
target. At target 3, this continuation overwrites the initial intervention
before reading it; the per-target results retain that geometry.

The registered reward is unchanged:

```text
clamp(0.15*delta_local + 0.15*delta_global + 0.20*delta_coherence
    + 0.20*delta_novelty - 0.15*delta_repetition - 0.10*delta_collapse
    + 0.05*delta_continuity - 0.05*cost, -1, 1)
cost = cumulative_generated_characters / (64 * (horizon + 1))
```

`nt_spa_agent_fit_repeated` computes and clips each native reward separately,
sums in double precision and rounds the mean to float once per action.
Targets are mean action reward minus mean KEEP reward. All valid heads fit
their mean Huber loss in one simultaneous update. Averaging raw axes before
reward clipping would change this operation; a deliberate compiled mutation
tests that distinction. Count 1 delegates exactly to `fit_comparison`.

The API accepts 1–64 comparisons and validates the complete array before
mutation. Its receipt reports mean rewards, targets, scores and losses;
the caller retains every raw consequence and its source/RNG association.
The original sensory API, online credit and canonical 2,296-byte life remain
unchanged. Python exposes the same operation as `Agent.fit_repeated`.

The primary measurement is mean paired H4 advantage over KEEP on the 48 new
states, averaged over all eight continuations. Separate `r0` and `r1..7`
results, per-state paired variance, action counts, regret, every raw axis and
cost are also retained. The sample unit is a source state: **48 states with
eight repeated outcomes each**.

## Measured result

Both complete body executions give the same numerical result. On the 48 new
states at seeds 307/401, the eight-outcome policy chooses KEEP everywhere,
as do the initial life and the shuffled-outcome control. The single-outcome
policy makes 21 interventions and loses to KEEP on the registered reward.

| Policy | KEEP / LEFT / RIGHT | Mean H4 advantage over KEEP | Mean regret | Best action, including ties |
|---|---:|---:|---:|---:|
| initial | 48 / 0 / 0 | 0 | 0.000637765 | 46 / 48 |
| single_h4 | 27 / 20 / 1 | −0.004025041 | 0.004662805 | 28 / 48 |
| mean8_h4 | 48 / 0 / 0 | 0 | 0.000637765 | 46 / 48 |
| shuffled_mean8 | 48 / 0 / 0 | 0 | 0.000637765 | 46 / 48 |
| fixed LEFT, KEEP at the left boundary | 12 / 36 / 0 | −0.007144097 | 0.007781861 | 13 / 48 |
| fixed RIGHT, KEEP at the right boundary | 12 / 0 / 36 | −0.006623239 | 0.007261004 | 13 / 48 |

These are means over states, each measured by eight paired continuations.
The 48 states are nested in 12 episodes and two generation seeds. Regret is
the best measured mean action reward minus the selected mean action reward;
the registered implementation uses a `1e-7` tolerance when counting ties.
They are measurements of this frozen body and continuation host.

The training states provide a direct noise check without changing their
sentence fields. The single-outcome policy has **+0.000479821** advantage on
its original `r0` outcomes, **−0.004265353** on the seven additional outcomes
of those same states, and **−0.003672206** across all eight. On new states,
its advantage is **−0.005690422** on `r0`, **−0.003787129** on `r1..7`, and
negative separately at both seeds: −0.003564616 at 307 and −0.004485465 at 401.

Among 72 available non-KEEP alternatives per cohort, the paired advantage
changes sign across repeats for **52 training alternatives and 57 evaluation
alternatives**. Only four evaluation alternatives, belonging to two states,
have a positive eight-outcome mean. The host therefore offers few measured
beneficial interventions in this cohort. Target 3's prescribed continuation
erases the original intervention before reading it; its remaining action
disadvantage comes from regeneration cost. Per-target results stay separate.

The mean8 life has different acquired weights but makes **zero action changes
from the initial life** on these evaluation states. It improves utility over
the harmful single-outcome policy and remains behaviorally indistinguishable
from both the initial life and the shuffled control here. This experiment
exposes stochastic outcome noise; it does **not** establish useful conditional
intervention or improved language quality. Capacity, features, fitting order,
budget and reward were held fixed throughout.

Raw axes remain available alongside the reward. For example, the single-outcome
policy's evaluation changes relative to KEEP include mean coherence
+0.000165826, novelty +0.001956870, repetition +0.002764935 and normalized
generation cost +0.082006834. A small coherence gain does not establish success.

## Recorded interruption and exact recovery

The second execution's fit trace was sealed at **59,168,552 bytes**, SHA-256
`a4ae3d8492c240cc91d419b0b78f576e43c36fcb12882ee84538d2fd49ad55ad`.
A later check found **52,978,553 bytes**, ending at shuffled-mean8 epoch 350,
row 27. The seal gate stopped the runner before any seed-307/401 generation.
Both complete training outcome sets and all eight policy lives were intact.
The observed shorter file was an exact prefix, missing 6,189,999 bytes.
The cause of this later truncation is unknown.

[`recover.py`](recover.py) preserves the damaged file and original seal, then
performs **one additional deterministic fitting replay** in a new directory.
It uses the same immutable helper, library, source data and budget. Recovery
requires the full fit trace, all four life files and the complete seal to match
the original second-execution identities exactly. All matched. Only the fit
trace was replaced, using a checked temporary file, fsync and atomic rename;
the original policies and seal stayed intact.

The resumed orchestration verifies every original source, binary and input
hash, seals all eight original lives, and starts the previously unstarted
evaluation stages. The protocol, numerical sources, training outcomes and
acquired policies retain their original identities. The two registered body
executions are accompanied by this explicit extra fitting replay. The failure
receipt is [`durability_failure.json`](durability_failure.json); its damaged
bytes can be reconstructed as the recorded prefix of the full archived fit
stream and checked against their original hash.

After both executions completed generation and readout, the final inventory
found two more truncated standalone files:

| File | Observed bytes | Complete bytes | Intact evidence from that same execution |
|---|---:|---:|---|
| execution 1, `evaluation.json` | 2,016,208 | 4,723,513 | Originally pinned closed archive; independently rebuilt dataset and 192 native readouts |
| execution 2, `off-s307.jsonl` | 211,683 | 223,818 | Originally pinned closed archive and immediate verified gzip mirror |

Both damaged files were exact prefixes. Unlike the fitting trace, their
original result manifests already recorded the shortened standalone files.
Those original manifests are failed evidence: they cannot authenticate a
complete standalone file. They do authenticate each execution's closed archive,
which retains the complete member. Dataset recovery additionally requires a
canonical rebuild from that execution's source/fork traces and exact reproduction
of all 192 evaluation readouts and the original summary. Ordinary-trace recovery
requires agreement with its immediate compressed mirror. Neither restoration
runs generation or fitting.

[`finalize.py`](finalize.py) preserves the original manifests and authenticates
the dataset restoration; it stops when the remaining trace mismatch appears.
[`finish.py`](finish.py) authenticates that trace, completes the full inventory
and emits new final artifact manifests. Both preserve the damaged originals.
The independent failure captures are [`durability_failure2.json`](durability_failure2.json)
and [`durability_failure3.json`](durability_failure3.json). The cause of all three
truncations remains unknown. No numerical result is repaired by substituting
another execution's outcome.

## Reproduction

```sh
make check_spa_repeated BLAS_FLAGS= BLAS_LIBS=
make test_spa_repeated_mutations
python3 experiments/spa_agent/replicates/run.py \
  --output /path/to/new-run --inputs /path/to/body-inputs
```

The inputs directory supplies the registered `simple.weights` and
`dracula.txt`. The runner verifies their hashes, builds an immutable source
snapshot, preserves complete raw traces and checks both executions. Native
child processes own separate bodies, RNGs and output files; their scheduling
does not change the registered numerical settings. Model weights and generated
life checkpoints remain outside the repository.

The published raw archive contains 22 complete streams. Its 18,444,383 bytes
have SHA-256 `dac359967bea506047813e25f88d7691010c0d8b47ba0b61fe35051f28300e12`.
The three lossless parts are indexed by [`raw_manifest.json`](raw_manifest.json).
Reassemble them into the ignored `raw_traces.jsonl.gz` with:

```sh
python3 experiments/spa_agent/replicates/traces.py
```

The helper validates each part's length, SHA-256 and Git identity, then the
complete archive. It refuses an existing destination. It neither generates
model output nor retrains policies. [`publication.json`](publication.json)
records the exact original and published identities where ephemeral path
spellings were normalized in receipts; numerical evidence is unchanged.

## Gates and engineering receipts

The native repeated API passes **3,457 checks in five groups**, including
1,602 finite differences covering all 267 parameters, exact count-one parity,
policy-only updates, hostile inputs and save/resume. Three compiled defects
reverse the credit, suppress the update, or average before clipping; each is
caught. An independent float-accumulation defect is also caught by exact mean
bits. The original v1 initial, pending, completed and imitation lives reproduce
freshly rebuilt `b289119` bytes.

The cheap host fixture supplies four sources, 32 repeats, 80 alternatives and
240 measurements. Its 30 `r0` measurement lines match the original host's
58,163 bytes exactly. Twenty-three malformed/path cases are refused, and one
finite-subnormal history fixture round-trips successfully. Four compiled host
defects fail native checks; the wrong-target defect reaches the independent
measurement validator and fails there. Fourteen named data corruptions test
repeat coverage, RNG pairing, token continuity, source/donor association and
the sealed parent control.

The audit discovered two missing validator checks before real generation:
an untouched H1 sentence could change, and an H4 sentence length could disagree
with its recorded generation. Both now fail named gates. The Python helper
also refuses malformed snapshot witnesses and a replaced, resealed parent
control. Deleting every outcome from a source leaves its policy readouts byte
identical. These discoveries and their repaired gates are retained explicitly.

Python passes **10 binding groups**, verifies 253 native ABI coordinates and
preserves the original 59 C/Python oracle records. The first clipping fixture
used an invalid weight and was correctly refused; its retained failure and
corrected legal fixture are separate receipts. Fresh main `5eb709f` was
integrated before the body experiment. The combined SPA/Chuck build passes all
**43 CPU recipe commands**; original sensory parity remains 1,024 steps and
114,688 bytes. Chuck files match that main exactly.

| Receipt | Evidence |
|---|---|
| [`protocol.json`](protocol.json) | Frozen cohorts, streams, objective, arms, budget and generation barrier. |
| [`diagnosis.json`](diagnosis.json), [`conditioning.json`](conditioning.json) | Read-only analysis of the exposed parent experiment. |
| [`core_mutations.json`](core_mutations.json) | Native clipping, credit and update defects on integrated sources. |
| [`audit.json`](audit.json) | Independent compiled host/precision defects, repaired validator gaps and v1 bytes. |
| [`python.json`](python.json) | Native ABI and Python parity. |
| [`verification.json`](verification.json), [`merge_verification.json`](merge_verification.json) | Exact integration commands, outputs and source hashes. |
| [`receipts.json`](receipts.json) | Two complete executions, all policy controls, source/RNG associations and separate raw axes. |
| [`durability_failure.json`](durability_failure.json), [`durability_failure2.json`](durability_failure2.json), [`durability_failure3.json`](durability_failure3.json) | Three original artifact refusals, including the failed manifests. |
| [`recovery.json`](recovery.json), [`dataset_recovery.json`](dataset_recovery.json), [`host_trace_recovery.json`](host_trace_recovery.json) | Exact fitting replay and two same-execution archive restorations. |
| [`result_audit.json`](result_audit.json) | Independent reward, perception, acquired-policy, durability and publication checks. |
| [`archive_verification.json`](archive_verification.json), [`raw_manifest.json`](raw_manifest.json) | Synthetic refusal checks and exact real-archive reconstruction. |

The pre-main mutation receipt and initial fixture failure remain alongside
their subsequent passing records. The initial verification command omitted
the old Python binding test's required C-layout file; its usage refusal is
retained, followed by the corrected invocation and PASS.
