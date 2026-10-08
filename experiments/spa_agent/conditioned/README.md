# SPA: conditioning repeated consequence targets

This experiment asks whether statewise target conditioning helps the existing
267-parameter policy select useful sentence interventions. It keeps the
29 features, 8 tanh units, three typed actions, reward, learning rate, fitting
order and update budget fixed. The native operation and Python binding expose
the same arithmetic. The old perception and unconditioned learning APIs retain
their behavior.

## Result: training choices changed; new-state utility did not improve

The two acquisitions reproduce all four saved lives and both 73,728-update
journals exactly. The `raw_mean8` life is byte-identical to the retained parent
control. Both pure-readout copies also match. Results below use the registered
original H4 reward, averaged over eight paired continuations per state.

| New-state policy | KEEP / LEFT / RIGHT | Advantage over KEEP | Mean regret | Measured-best choices |
|---|---:|---:|---:|---:|
| `initial` | 48 / 0 / 0 | 0 | 0.000812401 | 41 / 48 |
| `raw_mean8` | 48 / 0 / 0 | 0 | 0.000812401 | 41 / 48 |
| `conditioned_mean8` | 48 / 0 / 0 | 0 | 0.000812401 | 41 / 48 |
| `shuffled_conditioned` | 44 / 0 / 4 | −0.000838234 | 0.001650635 | 39 / 48 |
| fixed KEEP | 48 / 0 / 0 | 0 | 0.000812401 | 41 / 48 |
| fixed LEFT, KEEP at the boundary | 12 / 36 / 0 | −0.006439643 | 0.007252044 | 9 / 48 |
| fixed RIGHT, KEEP at the boundary | 12 / 0 / 36 | −0.006833424 | 0.007645825 | 18 / 48 |

Regret is the best measured mean reward minus the selected mean reward.
Measured-best counts use the registered `1e-7` tolerance.

On the retained training states, associated conditioning selects KEEP 45 times,
LEFT once and RIGHT twice. All three changes from initial KEEP have positive
measured H4 advantage and select the measured-best action. Mean advantage is
**+0.000328626**, compared with zero for initial/raw mean8 and **−0.000581098**
for shuffled conditioning. Keeping captured perception/history and RNG fixed,
acquired weights alone change those three native decisions. This establishes
the acquired-experience action path on the training field.

On new states, associated conditioning makes **zero changes** from initial/raw
mean8. It misses seven measured beneficial alternatives in seven states. It
stays at KEEP separately on both seeds and all four target positions. The four
shuffled-policy interventions all occur at target 0 on seed 601: one is
beneficial and three harmful. Its mean advantage is zero on seed 509 and
−0.001676468 on seed 601. Its `r0` advantage is +0.001054307, while `r1..7`
average −0.001108597. A single outcome would give a different impression.

Raw metrics expose the cost of those shuffled choices: versus KEEP, mean
coherence changes by −0.000513074, novelty by +0.000085241, repetition by
+0.000134245 and normalized generation cost by +0.015698242. Associated
conditioning's new-state deltas are all exactly zero because its choices
equal KEEP. Collapse remains a separately retained measurement.

**FAIL for improved new-state utility at this registered floor and budget.**
Conditioning supplies a trainable signal that changes retained-state actions,
but it does not establish useful transfer to new states. Its advantage over
the harmful shuffled control comes from retaining KEEP in this cohort.
Neither policy capacity, reward nor generation recipe was changed after the
result. A next experiment must distinguish limitations of the captured state
representation from sparse, noisy intervention effects; this result does not
select between those explanations.

## Registered comparison

The [read-only diagnosis](diagnosis.json) precedes fitting and new generation.
Among 72 non-KEEP alternatives on the retained 48 training states, 14 have a
positive eight-outcome mean, belonging to 12 states. Thirteen of those 14
also have at least three negative draws. Three positive means exceed one
paired standard error; none exceeds two. These descriptive errors use eight
repeated continuations of a state, not eight independent source states.

The [protocol](protocol.json) tests one combined axis: target conditioning
changes gradient amplitude and relative state weighting together. It divides
the existing float KEEP-relative mean rewards by their largest absolute valid
value, with a float32 floor of `0.001`. The floor activates on one training
state and bounds amplification at about 1,000. The original rewards and each
scale remain in the native receipts. Conditioning does not resolve noisy credit.

| Policy | Acquisition |
|---|---|
| `initial` | Fresh native seed-1 life; no fitting. |
| `raw_mean8` | Existing `fit_repeated`; must reproduce the parent mean8 life exactly. |
| `conditioned_mean8` | New `fit_conditioned`, associated state/outcome pairs. |
| `shuffled_conditioned` | Same operation; next cyclic outcome donor within matching sentence-count/target groups. |

Each fitted policy receives 24,576 native updates: 512 passes over the same
48 retained states from seeds 42/73, rate `0.03`, eight H4 consequences per
state. Two independent fitting copies must reproduce all four lives and the
complete fitting journals byte for byte. All eight lives are sealed before
any new body process. Training requires no new body generation.

Evaluation uses one new cohort: 48 states from seeds 509/601, eight paired
continuations per available action, retaining horizons 0, 1 and 4. The unchanged
host links upstream notorch with the frozen 450,688-parameter SimpleLLM body
(float32, width 128, two layers, four heads, FFN 384, context 64, vocabulary 94).
The two fitted copies receive pure native readouts on identical captured
features and masks. Every readout and saved life must agree. This evaluates
acquired choices on captured reference trajectories; it does not measure a
new online trajectory driven throughout by the acquired policy.

The primary measurement is original H4 reward advantage over paired KEEP.
Regret, action changes, per-seed/target results, seven raw axes and generation
cost remain separate. The 48 states are nested in 12 episodes and two seeds;
the 384 state/repeat pairs are not 384 independent states. Target 3's fixed
continuation overwrites the intervention before reading it, so its cost-only
outcomes are reported separately. Capacity and the host continuation are unchanged.

## Reproduction and evidence

```sh
make check_spa_conditioned BLAS_FLAGS= BLAS_LIBS=
python3 tests/test_spa_conditioned_mutations.py --sanitize
make shared test_spa_python BLAS_FLAGS= BLAS_LIBS=
python3 tests/test_spa_conditioned_policy.py --library ./libnotorch.so
python3 tests/test_spa_conditioned_audit.py
python3 tests/test_spa_conditioned_runner.py
python3 experiments/spa_agent/replicates/traces.py
python3 experiments/spa_agent/conditioned/run.py \
  --output /path/to/new-run --inputs /path/to/body-inputs
python3 experiments/spa_agent/conditioned/audit_final.py \
  --execution-dir /path/to/new-run --correction /path/to/schema-check.json \
  --output /path/to/audit.json
```

The inputs directory contains the registered `simple.weights` and `dracula.txt`;
their hashes and download source are in the protocol. The runner refuses an
existing output directory. It builds a source snapshot, hashes sources,
binaries and body inputs, records closed-file identities immediately and
checks them again through archive publication. Each fitting journal uses
deterministic gzip and retains every exact native receipt, source/donor join,
and policy identity. Model weights and life checkpoint binaries stay external.
The full execution audit needs those external lives; archive inspection alone
does not replace that audit.

The [protocol correction](protocol_revision.json) records a pre-fit repair to
duplicated primary-measurement prose that still named the parent's arms. Only
the arm names changed; the previous exact protocol bytes can be reconstructed.
No rate, reward, cohort, objective or budget changed.

## Recorded interruption

The first execution stopped during source generation, before any K8 evaluation
or policy readout. Both fitting journals and all eight lives were already sealed.
The seed-601 diagnostics process returned zero, but its ordinary trace ended
inside JSONL line 74: 120,594 bytes, SHA-256
`b8f32e01ecac0c40a223bd01cd47116152fb96b4441ada12c5ff5e361aa85413`.
It is the exact prefix of the complete 216,380-byte OFF trace, missing 95,786
bytes. The full original scenario trace and all five saved lives were intact;
the lives matched OFF. The cause and timing of truncation are unknown.

[durability_failure.json](durability_failure.json) preserves the failed gate
and pins 69 files at failure inspection. Those are failure-time identities,
not retroactive evidence that the original files passed completeness at close.
The original ON file has no surviving complete-file hash.

[recover.py](recover.py) preserves the failed prefix, original stage record
and failure receipt. It permits exactly one additional seed-601 ON diagnostics
process using the frozen binary and inputs, in a new directory. Before replacing
only the damaged trace, it requires the full replay to equal original OFF,
the failed bytes to be its exact prefix, and the entire original scenario and
all five lives to match. All original fitting, source, binary and input pins
remain checked. The additional process is disclosed separately from the
registered run; it introduces no new seed or policy fitting. Remaining K8
generation and pure readouts use the original sealed policies and unchanged
measurement helpers. The failed original command receipt remains in the archive.

The one replay passed all of those gates. [recovery_audit.json](recovery_audit.json)
independently verifies the original pins and complete replay witnesses.

The first result-audit run then exposed a checker schema error: it expected
per-action SHA dictionaries where the original helper records three-position
lists, with `null` for unavailable actions. [audit_failure.json](audit_failure.json)
retains that refusal. [audit_final.py](audit_final.py) authenticates the frozen
checker and applies one explicitly recorded expectation correction in memory.
The original checker, data and numerical measurements remain unchanged.
Swapped action hashes and a populated invalid-action slot both still fail.
[audit_correction.json](audit_correction.json) retains the exact before/after
patch and targeted checks.

The corrected independent audit passes all nine result gates: 147,456 fit
receipts, 384 independently decoded policy readouts, 40,992 raw axis values,
5,760 rewards, 114 retained artifacts and all 81 archive members. It performs
no policy fitting or model generation.

## Published raw evidence

[receipts.json](receipts.json) contains the full numerical result and closed
artifact manifest. The exact raw archive is **17,542,082 bytes**, SHA-256
`595bf9f75e68f1b8fe562bc2bbb68c3908d71a4817205d2684714657a2f68dca`.
Its three lossless parts are indexed by [raw_manifest.json](raw_manifest.json).
Reassemble them into the ignored `raw_records.tar.gz` with:

```sh
python3 experiments/spa_agent/conditioned/traces.py
```

The helper verifies each part's bytes, SHA-256 and Git identity and the complete
archive identity. It refuses an existing destination. [publication.json](publication.json)
records a real reconstruction from the committed parts, exact original receipt
bytes and complete archive membership; it performs no numerical recomputation.

| Receipt | Evidence |
|---|---|
| [native.json](native.json) | 5,928 checks, 1,602 finite differences over all 267 parameters, three compiled defects caught, ASan/UBSan pass with LeakSanitizer disabled. |
| [python.json](python.json) | 12 groups, 261 ABI coordinates across 15 types, original 59 exact C/Python records and v1 continuation. |
| [helper.json](helper.json) | Synthetic fixture: 288 fits and exact receipt replays, 192 pure readouts, four corrupted controls caught. |
| [audit.json](audit.json) | Independent pre-fit audit gate: nine groups, 15 deliberate defects caught, zero generation or native fitting. |
| [runner_preflight.json](runner_preflight.json) | 40 IO/source-closure checks, 16 refused corruptions, zero generation or native fitting. |
| [integration.json](integration.json) | Exact successful commands for 41 CPU recipes, shared build, Python binding and helper fixture. |
| [legacy.json](legacy.json) | Ordinary sensory parity: 1,024 steps and 114,688 identical bytes against main `420fa54`. |
| [finalization.json](finalization.json) | Integration with main `9f146ed`: 45 measured sources unchanged; native 5,928 checks, shared build and 12 Python groups pass after exact Git recovery of the staged deliverables. No new numerical experiment. |
| [result_audit.json](result_audit.json) | Independent native-target, acquired-policy, original-reward, raw-metric and archive checks. |

The runner review caught unsafe archive member names and an inventory exclusion
that used absolute path components. The latter could omit records if an output
ancestor was named `inputs` or `source_snapshot`. Both now have failing fixtures.
Independent audit review also added exact raw-source metadata joins and a
requirement for all five OFF/ON saved-life pairs. These repairs preceded all
real fitting and new outcomes.
