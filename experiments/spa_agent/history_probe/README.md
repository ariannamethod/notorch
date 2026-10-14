# SPA: frozen-policy history intervention

The [transfer diagnosis](../transfer/README.md) found that all three acquired
training interventions shared a previous LEFT, while the exposed evaluation
cohort had no previous LEFT. This experiment changes the complete captured
history input while holding sentence perception and learned weights fixed.

## Result: acquired memory changes the action

The reconstruction reproduces all **24,576** original update receipts and
both canonical life identities. All **192** original native readouts match
before the first composed-history probe. The acquired policy then changes
choices when the history input changes, with perception, weights, agent state,
RNG and action masks held fixed. The initial policy chooses KEEP for all
**1,344** of its recorded probes.

| Acquired policy / input | Readouts | KEEP / LEFT / RIGHT | Changes from original history | Distinct changed source states |
|---|---:|---:|---:|---:|
| Training, original | 48 | 45 / 1 / 2 | 0 | 0 |
| Training, zero history | 48 | 45 / 3 / 0 | 4 | 4 |
| Training, all matching donors | 576 | 519 / 18 / 39 | 69 | 27 |
| Evaluation, original | 48 | 48 / 0 / 0 | 0 | 0 |
| Evaluation, zero history | 48 | 48 / 0 / 0 | 0 | 0 |
| Evaluation, all matching donors | 576 | 539 / 11 / 26 | 37 | 17 |

Within the evaluation donor grid, histories whose previous action was LEFT
produce **29 changes in 60 cells**, across 17 source states. Other histories
produce **8 changes in 516 cells**, across two source states. Those source
sets overlap; their counts must not be added. Every donor in the matching
stratum was used, including histories that produced KEEP.

Zeroing training history removes the original RIGHT choices at `42/s23` and
`73/s10`, retains LEFT at `42/s4`, and introduces LEFT at `42/s18` and `73/s20`.
Thus the acquired response includes both suppression and activation by history.
The complete ten-coordinate block is the causal intervention in this test.

An independent audit decodes the saved policy and reconstructs all 2,688
readouts with scalar float32 arithmetic. Maximum score difference is
**3.576278687e−7**, within the fixed `3e−6` tolerance. The audit reproduces every
summary and catches four deliberate record corruptions. Native preflight
passes ten checks and catches fifteen defects, including altered state after
a readout, malformed history, perception leakage and invalid masked actions.

## Frozen protocol

[protocol.json](protocol.json) was written before probing, SHA256
`af116f2d153e3494cba93eac55033604e8c01851431d1713658bd133c1d25af6`.

For every one of the 96 retained source states, read both the initial policy
and the original acquired conditioned policy under:

- the original history;
- all ten history coordinates zeroed;
- each of the twelve training histories with the same target and sentence count.

This gives 2,688 native readouts. All matching donors are retained in source
order, including self-donors. Donor selection uses no reward or readout filter.
The whole history block (features19–28) contains three running means, three
action frequencies, the previous-action one-hot vector and the last reward.
Perception features0–18, recipient metadata, action mask and RNG stay fixed.

The policy is the original 267-parameter model: 29 features, eight tanh units,
three action scores. All arithmetic runs through `import SPA` in native C.
The host body that originally supplied the states was the frozen float32
450,688-parameter SimpleLLM (width128, two layers, four heads, FFN384, context64,
vocabulary94). This probe generates no text and executes no host actions.

## Recover the acquired weights

The previous working directory was removed by workspace maintenance. The
published archive retains the exact inputs and every native learning receipt.
The runner reconstructs the published policy with precisely 24,576 recorded
updates (512 passes over 48 training states at rate0.03, scale floor0.001).
Every recovered update receipt and the final canonical saved life must match
the originals before any history intervention. Recovery is counted separately
from the frozen-policy measurements. The original training experiment is fixed.

The recovered canonical lives stay outside Git. The original-history readouts
must reproduce the published native bytes. Every intervention records recipient
and donor provenance together with composed input bytes; donor history never
replaces recipient source identity.

## Interpretation and next step

This intervention measures the acquired policy's dependence on the complete
history block. Previous-LEFT groups describe donor histories; all ten history
coordinates move jointly. Recorded observations are native score changes and
typed action changes. The composed inputs have no new measured host consequence,
and the report assigns no original reward to their counterfactual decisions.

A subsequent registered rollout can let SPA generate its own action history
and measure future sentence consequences. The retained cohorts here are exposed
diagnostic data: 48 training states from seeds42/73 and 48 evaluation states
from509/601, each split nested in twelve episodes.

## Reproduce and inspect

Build the same native SPA surface, then run the preflight, recovery/probe and
independent audit. Use a new execution directory; the runner refuses an existing
one and the audit refuses an existing output file.

```sh
cc -std=c11 -O2 -DUSE_SIMD -march=native -pthread -I. -fPIC -shared \
  spa_binding.c spa_agent.c notorch.c -lm -o /tmp/spa-history.so
python3 tests/test_spa_history_probe.py --library /tmp/spa-history.so
python3 experiments/spa_agent/history_probe/run.py \
  --library /tmp/spa-history.so \
  --protocol experiments/spa_agent/history_probe/protocol.json \
  --output /tmp/spa-history-execution
python3 experiments/spa_agent/history_probe/audit.py \
  --execution-dir /tmp/spa-history-execution --output /tmp/spa-history-audit.json
```

The recorded run used GCC13.3.0 on Linux x86-64. Exact receipts, ABI and life
identity gate recovery on a different build. [build.json](build.json) records
compiler, source and library hashes; [publication.json](publication.json)
records the exact commands and results with execution-directory placeholders.

[summary.json](summary.json) contains every aggregate and changed source list;
[recovery.json](recovery.json) records the exact update recovery;
[audit.json](audit.json) holds independent checks;
[preflight.json](preflight.json) names all synthetic and native defects.
[probes.jsonl.gz](probes.jsonl.gz) is a lossless gzip of the complete 2,688-record
native trace, including source/donor witnesses and input/readout bytes. Its
compressed and decoded SHA256 identities are in publication.json. Decompress
with Python's `gzip` module or `gzip -dc`; no model weights are in the archive.
The full independent audit uses the exactly recovered external life files.
