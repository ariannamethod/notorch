# Learning from future consequences

The 163-parameter Architect receives three measured consequences from one common
training state. It learns the horizon-16 advantage of each action relative to
HOLD. A separate saved life acquires HeVLM experience after its SimpleLLM life.
The policy network, ordinary same-window feedback and version-1 life format stay
as specified in the earlier experiments.

The numerical design in the [protocol](protocol.json) was fixed before the first
smoke measurement. Eight
SimpleLLM states from seeds 42 and 73 train the first life. Eight HeVLM states
from those seeds adapt its copy. Both stages use 512 epochs, LR 0.03 and the
ascending `(seed, checkpoint)` order on every epoch. All three outcome heads
learn together from one pre-update weight snapshot. The initial policy seed is
1; checkpoints are 1, 32, 128 and 384.

Seeds 101 and 211 provide sixteen evaluation states across both bodies.
Both fitted lives are saved and hashed before any fresh evaluation outcome is
generated. Evaluation fits no weights, changes no source history, and checks
that every saved life retains its sealed hash.

The first dirty smoke exposed seed 101 at checkpoint 1 on both bodies: two of
the sixteen planned evaluation outcomes. Its original receipt and protocol hashes
are retained in `prior_smoke_exposure`; the target, epochs, rate, ordering and
capacity were fixed before that smoke and remain unchanged. Seed 101 is reported
as fixed-protocol evaluation with prior smoke exposure. The whole seed trajectory
is the separation unit, so seed 211's eight states also receive a separate
untouched-confirmation summary. Later smoke runs use seed 907. Final-run sealing
still precedes final-run outcome generation for both evaluation seeds.

## Measured target and source association

Each branch applies HOLD, BRAKE or PUSH once, then fifteen HOLD updates on the
same window sequence. Its target is

```text
target[action] = clamp((future_loss[HOLD] - future_loss[action])
                       / (abs(future_loss[HOLD]) + 1e-6), -1, 1)
```

The future loss averages the same four windows at stream indices 16–19. Huber
regression averages the three heads, with delta 1. Raw losses, action advantages,
targets, predictions and regression errors have separate receipts. Immediate,
origin-window and held-out losses remain separate measurements. The complete
branch records also retain H1 and H4 measurements, allowing the same probe to be
compared across horizons.

The sixteen earlier states already have exact action/outcome receipts, but their
original traces lack the current observation/features. The runner reproduces
the old hosts, captures the current pre-action observation with its saved policy
history, and requires every old step, fork state, branch result and saved state
byte to match. Only then does it pair captured features with the old H16 losses.
A preceding completed decision's features are never substituted for the current
state. A deliberately corrupted feature table must fail this association gate.

## What the controls ask

| Readout | Question |
| --- | --- |
| Initial weights | Does acquired future experience change choices at the same source history? |
| SimpleLLM life | How does the frozen first life act on new states of either body? |
| Adapted HeVLM life | What does experience of the second body change relative to its frozen parent? |
| Original source life | How does the source's acquired immediate-feedback policy choose there? |
| HOLD | What is the zero relative-action baseline? |
| PUSH | How does always choosing the immediate winner from the development states behave? |

The C readout loads each source life, verifies its observation/features, replaces
only the 163 network parameters, disables exploration on the temporary copy,
and calls the ordinary C action selector. Each choice is joined to the actual
common-state branch that executed that action. Reports retain future advantage,
regret against the best measured branch, tie-aware optimal choices and changes
from the initial and parent choices. Equal F32 losses retain every tied action.

Fitting changes weights alone. Its update count and objective identity live in
external receipts, including the source sample table and protocol hashes. The
online history, RNG, decision/update counters and serialization format retain
their existing meaning. Python orchestrates and records; C performs feature
capture, network arithmetic, fitting and action selection.

## Reproduce

Use the expanded raw common-state run identified in `protocol.json` as the
previous-run directory. It contains the original `results.json`, traces and
saved fork/final lives.

```bash
python3 experiments/chuck_loss_architect/future/run.py \
  --previous-run /path/to/previous-common-state-run \
  --output /tmp/chuck-future-credit
```

`--references /path/to/reference-checkouts` reuses the pinned corpus checkouts.
Final measurements require clean committed source, and every numerical source
hash is checked again after the run. The explicit development mode
`--allow-dirty --smoke` uses both bodies, old seed 42, seed 907 and eight host
steps while retaining full H16 branches and the 512-epoch fit schedule.

The runner writes full JSONL fit/readout traces, source and binary hashes,
`sealed_lives.json`, sample tables, source policy lives, body snapshots and a
phase timeline outside Git. Independent probe and no-probe processes must keep
identical host steps and final body/optimizer/policy bytes.
