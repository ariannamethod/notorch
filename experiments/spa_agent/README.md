# SPA Agent: first sentence-space experiment

The native agent passes the controlled acquired-experience test: replacing only
initial policy weights with acquired weights changes `KEEP` to `RESEED_LEFT`
while observation, temporal memory, action history, and RNG remain identical.
The first pretrained-body experiment records a different result: **zero of 48
choices changed through acquired weights**, and learned/frozen trajectories
remain identical. Both results belong to v1.

## Frozen protocol and body

[protocol.json](protocol.json) was fixed before the first run. Its SHA-256 is
`0a1272697270e93b8d0dbdeb2d8ce7706dbbbda77fcc146c2f29be7eddc1fae2`.
The [C host](../../examples/spa_agent_demo.c) loads the public 450,688-parameter
[notorch-simple-llm](https://github.com/ariannamethod/notorch-simple-llm/tree/80b3bd611ed8c937efdc481dc07895c8e13e345d)
Dracula checkpoint: width 128, two layers, four heads, SwiGLU width 384,
context 64, and 94 Unicode character tokens. The numerical engine is upstream
notorch. The body weights stay frozen; the separate 267-parameter SPA policy
receives sentence consequences.

Two seeds (42, 73), six episodes per seed, four sentence units per episode,
and four decisions per arm produce 240 host decisions across five arms:
disabled, registered legacy, uniform random, frozen policy, and learned policy.
Each episode's complete initial chain is shared by all arms. Host generation
and policy exploration use separate RNG streams. Every valid reseed is applied,
including one with a negative measured consequence. The policy retains its
history and weights across episodes and is saved/reloaded after each episode.

Generation stops on `.`, `!`, or `?` after at least 12 generated characters,
or at the fixed 64-character cap. Only 4/24 initial units for seed 42 and 2/24
for seed 73 ended on punctuation; the others reached the cap. Every token,
termination flag, and decoded unit is retained. These are the actual bounded
sentence units on which this experiment acts.

The immediate consequence window measures all seven axes separately:
adjacent cosine connectedness, legacy SPA global connectedness, mean-adjacent
coherence, novelty, repeated character trigrams, cosine collapse, and minimum
adjacent continuity. Cost is generated characters divided by 64. Exact formulas
and the fixed reward coefficients are in the protocol and
[architecture document](../../docs/spa-agent.md#consequences-stay-separate).

## Results

Every row contains 24 host decisions. Action counts are `KEEP/LEFT/RIGHT`.
Positive/negative counts exclude zero rewards. The last column is the exact
same-observation/history/RNG counterfactual replacing only acquired weights.

| Seed | Arm | Actions | Mean reward | Positive/negative | Changed choice |
| --- | --- | --- | ---: | ---: | ---: |
| 42 | disabled | 24/0/0 | 0.000000000 | 0/0 | — |
| 42 | legacy | 24/0/0 | 0.000000000 | 0/0 | — |
| 42 | random | 8/9/7 | -0.029748971 | 3/13 | — |
| 42 | frozen | 20/3/1 | -0.005917228 | 1/3 | — |
| 42 | learned | 20/3/1 | -0.005917228 | 1/3 | 0/24 |
| 73 | disabled | 24/0/0 | 0.000000000 | 0/0 | — |
| 73 | legacy | 24/0/0 | 0.000000000 | 0/0 | — |
| 73 | random | 9/9/6 | -0.024320280 | 0/15 | — |
| 73 | frozen | 21/2/1 | -0.006427699 | 0/3 | — |
| 73 | learned | 21/2/1 | -0.006427699 | 0/3 | 0/24 |

Mean per-decision changes in the raw axes follow. Frozen equals learned;
disabled and legacy have zero deltas. No pair crossed the protocol's collapse
threshold in these runs.

| Seed | Arm | Local | Global | Coherence | Novelty | Repetition | Collapse | Continuity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | random | +0.0057426 | +0.0001003 | +0.0043808 | -0.0256394 | -0.0193276 | +0.0000000 | +0.0056890 |
| 42 | learned | -0.0171547 | -0.0000588 | -0.0058117 | +0.0177818 | -0.0133289 | +0.0000000 | -0.0041806 |
| 73 | random | -0.0072825 | -0.0001074 | +0.0005343 | +0.0291130 | -0.0034703 | +0.0000000 | +0.0031182 |
| 73 | learned | +0.0007901 | -0.0000092 | +0.0001839 | -0.0015421 | -0.0000000 | +0.0000000 | -0.0004637 |

The learned arm receives 48 feedback updates. Its seven exploratory reseeds
produce six negative rewards and one positive reward. Initial action scores are
zero, with `KEEP` as the tie winner. Acquired reseed scores become negative;
the single positive result does not change their rank against `KEEP`.
The frozen reward, horizon, exploration, and generation cap were retained.
This run establishes execution, measurement, and acquisition on the real body.
Acquired action selection on that body remains equal to the frozen policy.
The controlled causal test establishes that the mechanism can change an action.

All 60 episode save/resume checks reproduce the canonical life exactly, and
body parameter bytes remain unchanged. Repeated ordinary runs reproduce both
raw traces and all ten final saved lives byte for byte. The trace SHA-256 values
are `51abdaee81b0e5cf1689ce48f0ae5d3783fe343672b0e7ea16d995f66680bf6b`
and `fe65bd7d17447de9153a12f2787083138fab0bb20415bd44b6570c29991884c6`.

## Reproduce and inspect

From the notorch root, the runner fetches only the pinned public checkpoint and
corpus, checks their hashes, snapshots the local C sources, builds, and validates
the complete output. Numerical execution links the upstream C implementation.

```sh
python3 experiments/spa_agent/run.py --output /tmp/spa-agent-run
```

For already downloaded material, add `--reference "$SPA_BODY_REPO"`; the
checkout must be at the exact pinned commit. Output includes the complete
JSONL traces, decoded sentences, source identities, and saved agent lives.
The runner records compiler flags, CPU, memory, and thread settings.

Checked-in evidence:

- [receipts.json](receipts.json): protocol, source/model/corpus hashes, every
  decision's raw measurements, action scores, counterfactuals, and replay checks.
- [raw_traces.jsonl.gz](raw_traces.jsonl.gz): complete original C trace records,
  deterministically compressed; checkpoints are regenerated rather than committed.
- [verification.json](verification.json): exact test commands, outputs, source
  hashes, mutation outcomes, and sanitizer status.
- [Q geometry probe](q_legacy_geometry.py) and
  [receipt](q_legacy_geometry.json): pinned Q C perception, 4,100 fields,
  three phase values, zero reseed triggers; the analytical lower bound is
  0.71228839, above its maximum trigger ratio of 0.70.

## Verification boundaries

```sh
make check_spa_agent BLAS_FLAGS= BLAS_LIBS=
LD_LIBRARY_PATH=. make test BLAS_FLAGS= BLAS_LIBS=
make test_spa_legacy_parity
python3 tests/test_spa_agent_mutations.py --json /tmp/spa-mutations.json
```

The agent gate covers 10 groups and 12,618 checks; the independent state gate
covers five groups and 78 checks. Legacy imitation learns nine controlled
examples and classifies all 108 held-out observations correctly. Four deliberate
mutations must compile and exit normally through their named failing gates:
reversed action ranking, reversed reward, reversed learning credit, and
suppressed acquisition. A crash or build failure does not count as detection.
Legacy perception matches rebuilt `origin/main` through 1,024 trace steps and
114,688 identical bytes; the existing BitNet/SwiGLU/SPA gate passes 118 checks.

The local full CPU run uses `LD_LIBRARY_PATH=.` because shared-library tests
link to the locally built `libnotorch.so`. Both libraries are rebuilt from the
integrated source before that final run.

Core and adversarial-state ASan/UBSan runs pass with leak detection disabled.
LeakSanitizer cannot inspect this sandbox's process tasks and is marked SKIPPED.
The body sanitizer's process result and artifact-completeness result are
recorded separately in receipts; incomplete artifacts are not accepted as a
successful body sanitizer gate.

The first consumer is SimpleLLM. Q's registered rule and exact perception are
covered as reference fixtures; a complete second generation consumer is not
part of this experiment. Sentence-target prediction and specialist routing
remain future work.
