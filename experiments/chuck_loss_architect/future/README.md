# Learning from future consequences

The 163-parameter Architect receives three measured consequences from one common
training state. It learns the horizon-16 advantage of each action relative to
HOLD. A separate saved life acquires HeVLM experience after its SimpleLLM life.
The policy network, ordinary same-window feedback and version-1 life format stay
as specified in the earlier experiments.

The numerical design in the [protocol](protocol.json) was fixed before the first
smoke measurement. Eight SimpleLLM states from seeds 42 and 73 train the first
life. Eight HeVLM states
from those seeds adapt its copy. Both stages use 512 epochs, LR 0.03 and the
ascending `(seed, checkpoint)` order on every epoch. All three outcome heads
learn together from one pre-update weight snapshot. The initial policy seed is
1; checkpoints are 1, 32, 128 and 384.

Seeds 101 and 211 provide sixteen evaluation states across both bodies.
Within the final run, both fitted lives are saved and hashed before evaluation
outcomes are generated. Evaluation fits no weights, changes no source history, and checks
that every saved life retains its sealed hash.

The first dirty smoke exposed seed 101 at checkpoint 1 on both bodies: two of
the sixteen planned evaluation outcomes. Its original receipt and protocol hashes
are retained in `prior_smoke_exposure`; the target, epochs, rate, ordering and
capacity were fixed before that smoke and remain unchanged. Seed 101 is reported
as fixed-protocol evaluation with prior smoke exposure. The whole seed trajectory
is the separation unit, so seed 211's eight states also receive a separate
untouched-confirmation summary. Subsequent `--smoke` calls use seed 907. Final-run sealing
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

## Measured run, 2026-10-07

Source `634a77ba8e28daa9aa71ebc1e000e074017ce955` completed the fixed protocol
on SimpleLLM (450,688 parameters) and HeVLM (1,123,456), F32, two SIMD threads,
AMD EPYC 9V74 and GCC 13.3, using the command above.
The [compact receipts](receipts.json) retain corpus/source/protocol hashes,
sealed lives, every state's features and branch losses, all 192 C readouts,
fit identities, host parity and the separate seed-211 summary. The two fitting
stages completed 4,096 updates each. All 4,096 paired host steps and final
body/moments/controls/policy bytes matched; all sixteen old states reproduced
their prior trajectories and outcomes before feature association.

Both fitted lives changed the initial HOLD choice to PUSH at all 32 development
and evaluation states. HeVLM adaptation changed scores at all sixteen evaluation
states but changed no selected action from the frozen SimpleLLM parent. Both
lives therefore matched the constant PUSH control in this run. Conditional
BRAKE selection was not acquired: the measured H16 winner on both fresh bodies
and seeds was PUSH at checkpoints 1 and 32, then BRAKE at 128 and 384.

The table reports mean H16 advantage over HOLD and regret against the best
measured branch. Lower regret is better. `Simple / adapted / PUSH` groups three
separately recorded readouts whose selected actions and outcomes were identical.

| Evaluation states | Readout | H16 optimal | Mean advantage | Mean regret |
| --- | --- | ---: | ---: | ---: |
| SimpleLLM, seeds 101/211 | Initial / HOLD | 0/8 | 0 | 0.00452873111 |
| SimpleLLM, seeds 101/211 | Simple / adapted / PUSH | 4/8 | 0.00316423178 | 0.00136449933 |
| HeVLM, seeds 101/211 | Initial / HOLD | 0/8 | 0 | 0.00401799381 |
| HeVLM, seeds 101/211 | Simple / adapted / PUSH | 4/8 | 0.00344905257 | 0.00056894124 |
| SimpleLLM, untouched seed 211 | Simple / adapted / PUSH | 2/4 | 0.00378119946 | 0.00080406666 |
| HeVLM, untouched seed 211 | Simple / adapted / PUSH | 2/4 | 0.00316631794 | 0.00044593215 |

The source immediate-feedback readout had the same mean evaluation metrics as
HOLD: it selected HOLD on all eight SimpleLLM states and seven HeVLM states;
the remaining HeVLM PUSH choice had the same measured loss as HOLD. Development
optimal choices were 9/16 for each fitted life and 3/16 for the initial life,
with ties retained. Host trajectories use the original immediate-feedback
policy; fitted lives select among the measured common-state branches.

SimpleLLM, seed 211, checkpoint 384 preserves a concrete failure. Both fitted
lives chose PUSH, although BRAKE had the lowest future loss:

| Action | Immediate origin loss | H16 origin loss | H16 future loss | H16 held-out loss |
| --- | ---: | ---: | ---: | ---: |
| HOLD | 2.53764606 | 2.38298011 | 2.43854713 | 2.65397024 |
| BRAKE | 2.53842854 | 2.38665915 | 2.43694258 | 2.65299416 |
| PUSH | 2.53686452 | 2.37936616 | 2.44011188 | 2.65493321 |

The selected PUSH regret was `0.0031692981719970703`, computed in C from the
F32 branch losses. The H16 origin-versus-future columns isolate a probe change
at the same horizon. The raw H1/H4/H16 records also permit a horizon comparison
using the same future probe; immediate origin versus H16 future changes both.
No new epochs, rate, capacity or target were tried after this result.

The sixteen original host processes took 364.0760502 seconds in total, with
60,804 KiB peak RSS. These timings exclude compilation, fitting/readout, fault
gates and the two later integrity diagnostic replays.

## Observed artifact failure and terminal gate

Independent audit found the persisted HeVLM seed-73 diagnostic trace truncated
after the run: 168,242 bytes, SHA `b79a64040d75e237e7fcca1d7637fb83ea0b1078902d57103601e6d42cc35cdc`.
Its earlier complete per-cohort receipt recorded 170,856 bytes, SHA
`be6482a0302e52bf7fa579a7038058fc400744b5ecca903c61a8c3e88736394b`.
The later artifact inventory contained the truncated hash, exposing a missing
terminal consistency check. The cause of the file change remains unknown.

The original truncated bytes and original results are retained. An isolated
replay with the exact measured binary reproduced all deterministic rows and
all 24 saved state artifacts. Retaining the already-recorded original timing
and RSS summary reconstructed the complete trace with exactly its original
SHA; that authenticated candidate was atomically restored. The corrected raw
results explicitly record this repair and preserve all numerical metrics,
policy lives, targets and protocol choices.

The runner now rereads every persisted control/diagnostic trace before success,
checks its recorded hash, complete step sequence and summary, and repeats host
and final-state parity. This gate went RED on the retained truncated file and
GREEN on the restored set of eight cohorts. The verifier was added after the
measurement; both its source hash and the original numerical source commit are
recorded separately in the receipts.

The [verification receipt](verification.json) binds the narrow tests, independent
probes, v1 byte comparisons, 38-command CPU recipe coverage and canonical
6,000-step trajectory. Full raw artifacts, including both integrity replays,
are retained in `chuck-future-credit-634a77b-raw.tar.gz` (429,943,659 bytes,
570 verified members, SHA-256
`0c46257580b881ce03940f07284b3a4f106cbe37268c20a17c7e1e0252ebdf3e`).
The archive includes its per-file hash index and the original failed receipts.
