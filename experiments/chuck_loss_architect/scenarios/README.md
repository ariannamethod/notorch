# One action, one common training state

This diagnostic compares `hold`, `brake`, and `push` from exactly the same saved
body, pending gradients, Adam state, Chuck state, policy life and random streams.
It asks how one intervention changes its immediate consequence and subsequent
training trajectory. The policy remains the 163-parameter v1 Architect.

The [fixed protocol](protocol.json) is written before the measurements: SimpleLLM
and HeVLM, seeds 42 and 73, 512 normal host updates, checkpoints before updates
1, 32, 128 and 384, and horizons of 1, 4 and 16 updates. Every branch applies its
chosen action once, then applies `hold` on identical subsequent training windows.
The host keeps its normal learned policy and never learns from diagnostic forks.

The horizon counts the intervention update. Four future probe windows come from
indices 16–19 of the common window stream, with the current window at index zero.
They are the same for every action and horizon and lie beyond the 16 branch
training windows. Eight fixed held-out windows provide a separate measurement.
Immediate loss, the original window's later loss, future probe loss and held-out
loss remain distinct fields. Action advantage for each metric is `hold - action`;
exactly equal losses retain all tied actions.

## Reproduce

```bash
python3 experiments/chuck_loss_architect/scenarios/run.py \
  --output /tmp/chuck-action-scenarios --verify
```

Use `--references /path/to/reference-checkouts` to reuse the pinned v1 corpus
checkouts. The default route downloads and checks their immutable SHA-256 corpus
identities. Final experiments require a clean committed tree. Development gates
can use `--allow-dirty --smoke`; the manifest labels that mode explicitly.

The diagnostic is an opt-in `--scenarios` mode of the existing C example. Its
ordinary v1 command keeps its successful training behavior. A post-action
non-finite result now receives its negative consequence, a valid JSON failure
receipt and saved final state before the runner stops.

## Restoration gate

The CPU snapshot deep-copies every live tape output and gradient, all operation
metadata, body parameters, Adam moments/accumulators/counters, global and local
Chuck state, the complete policy, Chuck's noise RNG, the window RNG and the
training flag. Restoration reconnects parameter entries to the original body
tensors. Each restored tensor and state field is compared byte for byte with its
snapshot. A GPU-enabled execution is refused by this CPU-only diagnostic.

Independent processes run the normal host with and without forks. Every host
step must have identical losses, action, policy identity and full state identity.
Final body weights, Adam moments, optimizer controls and policy files must also
match byte for byte. A deliberate omission of parameter restoration must make
this gate fail. The verification runner retains the failed mutation receipt.

The two bodies use the notorch initialization RNG only while constructing their
initial parameters; no branch operation consumes that stream. Each fork saves
body parameters, pending parameter gradients, optimizer controls/moments and the
policy life. Complete in-memory tapes support exact restoration; deterministic
forward/backward plus the saved window reconstructs their transient operations.
Source/corpus/configuration/protocol hashes and every branch step accompany the
snapshot artifacts outside Git.

## Measured run — 2026-10-07 UTC

The fixed experiment ran from clean source
`0dea13331077d9c5da504c5c6608b4f804f345ed`, with all numerical source hashes checked
again after execution. [Compact receipts](receipts.json) retain every comparison,
source and corpus identity, final state hashes, timings and the negative gates.

All four cohorts passed: 2,048 corresponding host steps had identical losses,
actions, policy identities and full state identities with and without forks.
The final body, moments, optimizer controls and policy files matched byte for
byte in each cohort. The 16 snapshots produced 48 fork rollouts, 768 branch
updates and 144 action/horizon measurements, grouped into 48 comparisons.

| Body | Parameters | Seed | Final held-out loss, both host processes | Compared host steps |
| --- | ---: | ---: | ---: | ---: |
| SimpleLLM | 450,688 | 42 | 2.53987360 | 512 identical |
| SimpleLLM | 450,688 | 73 | 2.68184018 | 512 identical |
| HeVLM | 1,123,456 | 42 | 1.57145250 | 512 identical |
| HeVLM | 1,123,456 | 73 | 1.58826482 | 512 identical |

PUSH produced the lowest immediate same-window loss at all 16 snapshots. Future
consequences selected different actions in several states. Each cell below
counts winners among eight snapshots in the order **PUSH / BRAKE / HOLD+BRAKE
tie**; every recorded tie is retained.

| Body | Horizon | Future-probe winners | Held-out winners |
| --- | ---: | ---: | ---: |
| SimpleLLM | 1 | 8 / 0 / 0 | 7 / 1 / 0 |
| SimpleLLM | 4 | 7 / 1 / 0 | 7 / 1 / 0 |
| SimpleLLM | 16 | 5 / 2 / 1 | 6 / 2 / 0 |
| HeVLM | 1 | 5 / 1 / 2 | 5 / 2 / 1 |
| HeVLM | 4 | 6 / 1 / 1 | 4 / 3 / 1 |
| HeVLM | 16 | 4 / 2 / 2 | 4 / 2 / 2 |

At horizon 16 the future-probe winner differs from the immediate winner in
7 of 16 snapshots; the held-out winner differs in 6. HOLD and BRAKE produce
identical state hashes at the dampening floor in the four seed-73 snapshots at
updates 128 and 384, explaining their ties. The policy's 163 parameters and its
immediate normalized-improvement learning target remain unchanged.

One concrete reversal occurs in SimpleLLM, seed 42, before update 128. All three
actions start from the same pending body and gradients; every subsequent update
uses HOLD on the same window sequence. Lower loss is better.

| Action | Immediate own-window loss | Future loss, horizon 4 | Future loss, horizon 16 | Held-out loss, horizon 16 |
| --- | ---: | ---: | ---: | ---: |
| HOLD | 2.35505104 | 2.92756820 | 2.87523842 | 2.85701990 |
| BRAKE | 2.35649300 | 2.92720723 | 2.87194562 | 2.85494900 |
| PUSH | 2.35361862 | 2.92799807 | 2.87865996 | 2.85925555 |

Relative to HOLD, BRAKE increases immediate loss by `0.00144196` and reduces
horizon-16 future loss by `0.00329280`. PUSH reduces immediate loss by
`0.00143242` and increases horizon-16 future loss by `0.00342154`. The environment
supplies distinct immediate and future consequences for this one intervention.

The restore defect was caught with exit 1. A post-action NaN was recorded as
reward −1, consumed exactly once, saved with policy decisions/updates/pending
`1/1/0`, and stopped with exit 3 before another training step. The 640-token
corpus gate refused the insufficient held-out split before evaluation. Together
with the other scenario suites, the restore defect is the twelfth caught
deliberate mutation; NaN injection and short-corpus refusal are separate gates.

The eight host processes took 188.02 seconds in aggregate, excluding compilation
and negative gates, with two SIMD threads per process. Peak recorded RSS was
52,852 KiB. Raw traces, fork parameters and gradients, optimizer states, saved
policies, corpus/token files and verification evidence are retained in the
external archive identified by the compact receipt.
