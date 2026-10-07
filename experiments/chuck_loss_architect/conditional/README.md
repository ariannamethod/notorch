# Conditional action credit

Chuck's next experience keeps the 163-parameter network, sixteen features,
three actions, learning rate .03 and 512 epochs per body. Each world's measured
action differences now set the scale of its own comparison. A second interface
completes real action feedback while freezing weights, so the acquired policy
can drive a whole training trajectory and retain its evolving history.

## Development diagnosis

Six fixed trials use only the sixteen archived development states from seeds
42 and 73. The earlier raw-target run reproduces its saved weights exactly.
Its mean target favors PUSH on both bodies; the first SimpleLLM stage changes
the input weights by an L2 norm of only 0.0000211272. Sixteen times as many
epochs and a global target gain of 100 retain the missed conditional choices.
Frozen-hidden and constant-feature controls accompany the gain-100 trial.

The per-state scale trial learns all eight SimpleLLM choices, selects an optimal
action on seven of eight HeVLM states before adaptation, and reaches eight on
each body after adaptation. Exact outcome ties count as optimal. All six trials,
including their failures, are retained in
[development_diagnosis.json](development_diagnosis.json). These development
results select the next fixed experiment; its new seeds are 307 and 509.

## Objective and persistent state

For measured future losses in HOLD/BRAKE/PUSH order:

```text
delta[a] = loss[HOLD] - loss[a]
scale = max(1e-6 * (abs(loss[HOLD]) + 1),
            max(abs(delta[a]) for finite loss[a]))
target[a] = delta[a] / scale
```

A non-finite HOLD refuses fitting. A non-finite alternative receives target
−1 and retains its raw outcome/status. Finite equal outcomes receive equal
targets; the floor bounds the effect of very small differences. Mean Huber
loss with delta 1 and simultaneous gradients teaches all three heads. Existing
weight bounds remain −16 to 16. A receipt includes raw losses, deltas, scale,
targets, pre/post scores, objective values and life hashes.

The heads express within-state normalized advantages. The original
`nt_chuck_architect_fit_comparison` continues to teach its specified relative
loss improvement. Separate saved-life identities and the protocol bind the
chosen objective. Both fit interfaces change weights only; ordinary online
history, RNG, counters and v1 encoding retain their meaning.

`nt_chuck_architect_feedback_frozen` completes a pending executed action. It
records the measured same-window consequence, advances history and completed
feedback counters, clears pending credit and returns `learned=0`. Every weight
stays identical. Its receipt retains the selected score and measured reward;
`error=0` records that no regression error is computed for this completion.
Ordinary `feedback` retains its existing learning arithmetic.

## Fixed evaluation

The [protocol](protocol.json) binds two distinct experiments. Both conditioned
lives are saved and hashed before either new seed generates outcomes.

Common-state evaluation uses the original host policy to reach checkpoints
1, 32, 128 and 384 on both bodies and both new seeds. Each action executes
once, followed by fifteen HOLD updates. The same future four-window probe
measures all branches. Initial, original-future, conditioned, source and fixed
action readouts share the exact source observation and policy history.
Independent diagnostic-on/off hosts must keep identical complete trajectories
and final states.

Closed-loop deployment starts each body from its seed's initial parameters.
Canonical Chuck, fixed HOLD, fixed PUSH, the old Simple-trained future life,
the conditioned Simple life and its HeVLM-adapted copy each train for 512
updates. A learned arm chooses anew at every step and receives frozen feedback;
its choices therefore influence the states it sees next. All arms receive the
same corpus stream. Held-out loss, action counts, measured consequences and
body/optimizer/policy identities remain separate receipts. Repeating the
adapted arm with a policy save/load at step 256 checks exact continuation while
the body remains in process.

The earlier future experiment's seeds 101/211 remain historical evidence.
The new protocol uses only 42/73 development targets for fitting and 307/509
for evaluation. Smoke runs reuse development seed 42. Target, capacity and training settings
stay fixed after evaluation begins.

## Reproduce

```bash
python3 experiments/chuck_loss_architect/conditional/run.py \
  --previous-run /path/to/expanded-future-credit-run \
  --output /tmp/chuck-conditional-credit
```

The previous run is the authenticated archive identified in the preceding
[future-credit report](../future/README.md). Source, corpus, protocol, fitted
lives and output hashes accompany the raw traces. CPU execution uses F32 and
two SIMD threads; numerical policy work stays in C.

## Measured run, 2026-10-07

Clean source `ff9d3f41c0b9826c3743a154aff6ddc2c899edfd` executes the frozen
protocol on AMD EPYC 9V74, GCC 13.3, F32, two SIMD threads. The two bodies
retain 450,688 and 1,123,456 parameters. Archived development states supply
8,192 comparison fits. Five sealed lives precede every new-seed outcome.
The run records 288 C readouts across sixteen development and sixteen fresh
states, 48 fresh action branches with 768 branch updates, and 4,096 control/
diagnostic host updates. Every paired host trajectory and final state matches.

The [receipts](receipts.json) retain all sources, corpus identities, features,
targets, action outcomes, readouts, sealed lives and complete summary tables.
The protocol SHA-256 is
`8b4e4292805950a066817f9b13018ab92a0d4dd9f7d4fae3a6020243c51796f2`.
No fitting setting changes after the new seeds begin.

### Choices from common states

Each body contributes eight new states, four per seed. Counts preserve exact
measured ties. Regret is selected H16 future loss minus the best measured loss.

| Body | Acquired life | HOLD / BRAKE / PUSH | Optimal | Mean regret |
| --- | --- | --- | --- | --- |
| SimpleLLM | Old relative-loss life | 0 / 0 / 8 | 4/8 | 0.000982791185 |
| SimpleLLM | Conditioned Simple or adapted copy | 0 / 2 / 6 | 6/8 | 0.000310242176 |
| HeVLM | Old relative-loss life | 0 / 0 / 8 | 4/8 | 0.000817790627 |
| HeVLM | Conditioned Simple | 1 / 3 / 4 | 5/8 | 0.0000748187304 |
| HeVLM | Adapted copy | 0 / 4 / 4 | 6/8 | 0.0000558793545 |

The old adapted life has the same choices as its parent. New HeVLM adaptation
changes one of eight HeVLM choices, and zero SimpleLLM choices. The two missed
SimpleLLM states and two missed adapted-HeVLM states remain in the receipts.
Initial and source lives select HOLD throughout; fixed BRAKE selects an
optimal action at 3/8 states per body. All nine controls remain reported.

### Consequences through a complete training loop

Twenty-four primary runs execute 12,288 body updates. Four independent
save/load repeats add 2,048 updates. Every conditioned run uses all three
actions; the old future life chooses PUSH throughout and matches fixed PUSH
on all 2,048 step comparisons and final body, moments and control bytes.

The table gives held-out loss after update 512; lower values are better.
The raw traces also retain evaluations at steps 0, 128, 256 and 384.

| Body | Seed | Canonical Chuck | HOLD | PUSH / old future | Conditioned Simple | Adapted copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleLLM | 307 | 2.48433638 | 2.50896096 | 2.48682046 | 2.54674840 | 2.53935456 |
| SimpleLLM | 509 | 2.53324080 | 2.54700208 | 2.54189587 | 2.57628131 | 2.56634331 |
| HeVLM | 307 | 1.58709764 | 1.59234679 | 1.59718883 | 1.58298719 | 1.58755374 |
| HeVLM | 509 | 1.58882868 | 1.57306290 | 1.57452559 | 1.58084154 | 1.58228230 |

Both conditioned lives regress against PUSH and canonical Chuck at the end
of both SimpleLLM runs. HeVLM improves against PUSH at seed 307 and regresses
at seed 509. The Simple-trained life beats canonical Chuck on both HeVLM
seeds; its adapted copy is worse than its parent on both HeVLM seeds.
These are separate measured outcomes of the same acquired action language.

SimpleLLM seed 307 shows how the trajectory changes: the conditioned Simple
life selects 35 HOLD, 329 BRAKE and 148 PUSH actions. Its step-256 held-out
loss is 2.60634041 against PUSH's 2.63895655; its final loss is 2.54674840
against 2.48682046. The earlier advantage and later regression are both
retained. The next learning question is consequence credit on the training
states reached through repeated acquired actions.

All four policy save/load repeats reproduce every step, held-out evaluation
and final body/moments/control/policy byte. Each completes 512 feedback events
with unchanged policy weights and retained history. The body remains in
process during these policy-continuation tests.

The eight common-state host processes report 185.0287185 seconds; the 28
deployment processes report 582.2871297 seconds and maximum process RSS
125,920 KiB. These are the recorded per-process measurements for this fixed
CPU run; compilation, fitting/readout, hashing and archive work are separate.

### Verification and retained failures

The new API gate passes 2,474 checks; three intentional learning defects are
caught. Independent probes check 1,531 assertions, including every network
derivative across eight loss regimes. The checkpoint gate passes 111 checks
and catches omitted directory synchronization. All 41 CPU recipe commands
pass after rebuilding dependencies. Canonical Chuck's 6,000-step trajectory
and all three pinned v1 compatibility lives remain byte-identical. Focused
ASan/UBSan pass; LeakSanitizer retains the previously recorded sandbox
task-inspection limitation. CUDA execution remains pending hardware access.

[Smoke validation](smoke_validation.json) retains three failed artifact checks
and the complete fourth smoke. Their numerical protocol and development seed
42 stay fixed. The successful smoke and final run stage intermediate artifacts
in `/tmp`, then verify the complete archives before publication. Nine deliberate
receipt faults are caught, including the actual changed smoke trace. The cause
of the earlier pathname-content changes remains unknown.

The verified numerical archive is `chuck-conditional-final-ff9d3f4-raw.tar.gz`:
420 members, 372,856,293 bytes, SHA-256
`adfcd981ffe6990cda483b46826aec5b3a01d67a2642ef0d10fb22a31c8faeaf`.
Its raw results SHA-256 is
`c18eae909ccbdd8f2bd62b752ab465a6c630da821033eef876c583d0a4626ce9`.

The [independent verification](verification.json) completes 126,920 checks:
all 416 recorded artifact identities, 420 archive members, both complete C fit
traces and lives, all 288 C readouts, and every reported summary. It reconstructs
8,192 learned-policy history transitions from saved initial lives and measured
observations/consequences, reproducing pre/pending/post hashes and final policy
bytes. All 14,336 deployment steps and four continuation pairs are authenticated.

The complete retained phase is `chuck-conditional-credit-ff9d3f4-receipts.tar.gz`:
962 members, 482,540,391 bytes, SHA-256
`94f802e20733c89d02cca9707b51d6e6824c1307beb6e8f325036a9903954e9a`.
It contains the numerical archive, development diagnosis, all four smoke
attempts, compatibility and mutation receipts, and independent audits. Every
member and the published copy were verified; the index hash is recorded in
`verification.json`. Compact results and documentation pass a further 2,254
independent checks.

### Integration with the next SPA merge

Before publication, main `b289119` adds SPA future comparisons and its Python
binding. The [integration receipt](merge_verification.json) verifies all
seventeen measured Chuck sources against `ff9d3f4` and twenty-eight incoming
SPA files against `b289119`. Forced shared/static and host builds, both agents'
narrow gates, Python and all 42 current CPU recipe commands pass. One scenario
executable initially lacks execute permission; its targeted rebuild has identical
bytes and its rerun passes. The initial exit 126 remains recorded. This later
receipt is separate from the frozen numerical archive and changes no acquired
life or measured body trajectory.
