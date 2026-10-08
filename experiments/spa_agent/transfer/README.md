# SPA: where the acquired decisions stop transferring

This is a read-only, post-result diagnosis of the [conditioned experiment](../conditioned/README.md).
It reads the original authenticated archive and receipts. Policy fitting and
body generation both remain at **zero**. The exposed evaluation seeds 509/601
serve as diagnostic data in this report; subsequent transfer evaluation needs
a newly registered cohort.

The measured body remains the frozen 450,688-parameter float32 SimpleLLM
(width128, two layers, four heads, FFN384, context64, vocabulary94), with a
267-parameter SPA policy: 29 inputs, eight tanh units and three typed actions.
Each split has 48 captured states nested in 12 episodes and two seeds, with
eight paired H4 consequences per available action. Training seeds are 42/73.

## Decisions follow the scores

The analysis decodes each native readout, joins its complete source witness,
checks its available-action mask and reproduces the recorded masked argmax.
Both acquisition copies have identical readout bytes. There are 384 unique
state/policy readouts across the two splits.

| Associated conditioned policy | Training | Exposed evaluation |
|---|---:|---:|
| Available non-KEEP alternatives | 72 | 72 |
| Positive measured mean alternatives | 14 | 7 |
| Positive learned score margins over KEEP | 3 | 0 |
| Positive-mean alternatives with negative learned margins | 11 | 7 |
| Largest non-KEEP margin | +0.904243231 | −0.177349791 |
| Smallest non-KEEP margin | −1.221085429 | −1.934276104 |
| Mean Huber error against associated conditioned targets | 0.067545001 | 0.092554194 |

Every available evaluation intervention scores below KEEP. Their gaps are
finite negative readouts, with no zero-margin ties. The score-to-action path
reproduces the original choices exactly. On training, the policy still places
11 of the 14 positive-mean alternatives below KEEP. This locates a retained-data
fit gap alongside the new-state transfer failure.

Mask handling matters: training state `42/s7` has positive LEFT score
`+0.551815987` at target0, where LEFT is unavailable. The correct choice is KEEP.
The analysis excludes that unavailable head; counting it would falsely report
a fourth training intervention.

## A specific history shift

All three acquired training interventions occur where feature26, the indicator
for the previous action being LEFT, equals one:

| State | Chosen action | Previous LEFT indicator | LEFT history frequency |
|---|---|---:|---:|
| 42/s4 | LEFT | 1 | 0.25 |
| 42/s23 | RIGHT | 1 | 0.125 |
| 73/s10 | RIGHT | 1 | 0.125 |

Previous-LEFT states occur **5/48** times in training and **0/48** times in
evaluation. Any LEFT in the recorded short action history occurs **25/48**
times in training and **0/48** times in evaluation. The report retains all
source identities and the feature witnesses behind this association.

A broad coordinate-range explanation also has a measured boundary: 22/48
evaluation vectors exceed at least one training coordinate minimum or maximum,
while four of the seven missed positive states remain within every coordinate
range (`509/s11`, `509/s13`, `601/s14`, `601/s16`). The min/max box is a
coordinate description, not a learned support model.

The next causal question is whether the acquired policy uses that history
block to select its interventions. A fixed-policy probe should hold sentence
perception, target, mask and RNG fixed, substitute a consistent captured history
block, and measure the action scores. Record intact-history and zero-history
controls and same-target donor identities before reading outcomes. Preserve
all original raw reward axes. This tests the influence of the policy's history
input; a subsequent registered rollout evaluates consequences along actual
histories produced by the acting agent.

## Credit remains variable across paired continuations

Three of 14 training positive means and two of seven evaluation positive means
exceed one paired standard error; none exceeds two. Five training positive
means and three evaluation positive means remain positive in all eight
leave-one-draw-out means. These are descriptive repeated-continuation measures
within a state; their draws are nested, not independent source states.

The full report includes every valid alternative's original mean reward
advantage, score margin, conditioned target residual, paired standard error,
sign counts and leave-one-out range. Target3's cost-only alternatives remain
separate. The existing objective, threshold and action choices are unchanged.

## Reproduce

```sh
python3 tests/test_spa_transfer_diagnosis.py
python3 experiments/spa_agent/transfer/analyze.py \
  --output experiments/spa_agent/transfer/diagnosis.json
```

Python's standard library suffices. Missing sparse-checkout evidence is read
from the committed Git tree. Archive members are read in memory; no archive
extraction, native inference or fitting is needed.

[diagnosis.json](diagnosis.json) authenticates the original receipt SHA256,
full archive SHA256, six selected member identities and this analysis script.
[validation.json](validation.json) records exact commands and results. The gate
rejects twelve deliberate corruptions, including coherent score-byte changes with
an incorrect selected action, reversed consequence credit, forged source joins,
missing/duplicate readouts and altered paired errors. The real positive-but-
unavailable LEFT case is also checked.
