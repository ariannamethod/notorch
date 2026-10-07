# Credit from Chuck's own training trajectories

This preregistration fixes one acquisition round for Chuck's existing
163-parameter policy. A frozen parent acts throughout real training; the worlds
those choices reach supply the next experience. At each saved world, HOLD,
BRAKE and PUSH encounter two measured continuations: fifteen HOLD updates, or
fifteen new selections by the same parent from each branch's evolving state.

The [protocol](protocol.json) fixes the two bodies, seeds, source states,
continuations, fitting order and evaluation before these new experiments run.

## Why this next experience

The preceding [conditional run](../conditional/README.md) acquired fresh-state
choices: SimpleLLM reached 6/8 measured-best actions, and HeVLM reached 5/8 before
adaptation and 6/8 afterward. Its 512-step deployments then created different
trajectories. Both conditioned lives finished behind PUSH and canonical Chuck
on the two SimpleLLM seeds; HeVLM improved against PUSH on one seed and regressed
on the other. Those complete outcomes remain in the earlier receipts.

That experience used one intervention followed by fifteen HOLD updates, at
states reached by the earlier immediate-feedback host. The present acquisition
uses the frozen conditioned parent's own states and measures what follows when
that parent keeps selecting actions. A same-state HOLD-continuation student
isolates the change in the subsequent policy.

## Parent worlds and measured branches

The parent is the earlier sealed `conditional-simple` life, SHA-256
`e93349d96f31a6a8be37e0f35571ad48d32d9c3213613d223416158c9d572dcb`.
Its 163 weights stay frozen during acquisition. SimpleLLM has 450,688 parameters;
HeVLM has 1,123,456. Each body trains for 512 updates on development seeds 42
and 73, using its retained corpus and recipe, F32 and two SIMD threads.

Every host begins with fresh temporal history, the matched initial body and
window/noise seeds, and the parent's weights. Its executed actions and measured
same-window consequences determine its subsequent history. Snapshots precede
updates 1, 64, 128, 192, 256, 320, 384 and 448: sixteen worlds per body,
thirty-two in total.

Each snapshot retains the body, gradients, optimizer moments, global/local
Chuck state, policy history/cache/RNG, tape, training mode and window/noise RNG.
All six branches begin from those identical bytes:

| Continuation | Update 1 | Updates 2–16 |
| --- | --- | --- |
| HOLD | Execute the named HOLD/BRAKE/PUSH intervention | Execute HOLD |
| Parent policy | Execute the same named intervention | Let the frozen parent select at every new branch state |

A synchronous native intervention validates and executes the first action,
invokes one same-window after-loss callback and completes its real frozen
feedback. Every subsequent executed action in both continuations also receives
real feedback. History advances even during forced HOLD. Receipts identify the
externally supplied first action, with actual weights, configuration and RNG. Each branch uses the same future window
stream and the same parent weights; its later choices can differ because its
own earlier consequences changed its observations and history.

At horizons 1, 4 and 16, retain origin-window loss, the same four-window future
probe and held-out loss separately. Only the H16 future probe, at stream indices
16–19 relative to the fork, supplies training targets. The probe windows remain
identical across branches and horizons. Restoring the source snapshot lets the
ordinary host continue exactly. The branch beginning with the parent's actual
choice and continuing under that parent must reproduce the next sixteen
ordinary host transitions.

## Two students, one fixed fitting schedule

Both students start independently from the same parent weights and consume
features captured from the same thirty-two worlds. `lived-hold` receives the
three measured H16 losses under HOLD continuation. `lived-policy` receives the
three measured H16 losses under parent-policy continuation.

The existing conditioned objective stays fixed:

```text
delta[a] = loss[HOLD] - loss[a]
scale = max(1e-6 * (abs(loss[HOLD]) + 1), max_finite_abs(delta))
target[a] = delta[a] / scale
```

Mean three-head Huber loss, learning rate .03, and 512 epochs teach each
163-parameter student. Each epoch visits SimpleLLM then HeVLM, ascending seed
then checkpoint. Each student receives 16,384 fits; total fitting is 32,768.
There is no separate body-adaptation stage in this round. Raw outcomes, scale,
targets and errors remain separate receipts. Fitting changes weights only.

A failed branch retains its actual non-finite loss, gradient or callback
failure; the host snapshot is restored and the run stops. It supplies no
partial comparison to fitting. The generic conditioned API retains its
existing non-finite contract; this experiment's sample tables contain complete
finite comparisons. Exact F32 ties retain every best action, including aliases
at control bounds.

## Sealed evaluation and complete deployment

Parent and student lives are saved and hashed before either new seed, 701 or
907, generates outcomes. Independent parent hosts then reach the same eight
checkpoints on both bodies and both seeds, producing thirty-two new common
worlds. Both continuations are measured there. Parent, `lived-hold`,
`lived-policy`, fixed HOLD, BRAKE and PUSH are joined to their executed branch
outcomes. Report each continuation's regret, advantage and optimal-choice count
separately. The primary action consequence is measured under parent-policy
continuation; the HOLD table provides its parallel control. Six readouts over
both continuation tables yield 384 development and 384 evaluation records.

Complete deployments run six arms for 512 updates on each body/seed:
canonical Chuck, fixed HOLD, fixed PUSH, parent, `lived-hold` and `lived-policy`.
All policy arms begin with fresh history and their sealed weights. Actual
feedback advances history without online weight fitting. Held-out evaluations
at updates 0, 128, 256, 384 and 512 retain the whole measured curve.

The comparisons ask:

- `lived-policy` versus `lived-hold`: what changes when future continuation
  matches the parent's repeated selection?
- Each student versus parent: what does the additional experience on its own
  reached states change?
- Policy arms versus canonical/HOLD/PUSH: how do their complete trajectories
  and final held-out losses compare?

Four additional `lived-policy` runs save/load the complete Architect after
update 256 while the body stays in process. Their next actions, all evaluations
and final body/moments/control/policy bytes must reproduce uninterrupted runs.

The fixed full run contains 8,192 paired-source host updates, 6,144 fork
updates and 14,336 deployment updates: 28,672 body updates, plus 32,768 policy
fits. Development smoke reuses seed 42 and eight host updates; its reduced
counts stay separate. No new target, epoch, learning rate, capacity or recipe
is chosen after the evaluation seeds start.

## Receipts and gates

Every world binds source commit, corpus/config/seed, complete source hashes,
observation/features, parent identity, selected/executed action, continuation,
actual feedback and future consequence. Every fitted life binds its input
sample-table hashes and protocol. Whole-file hashes accompany complete raw
traces; compact summaries retain exact losses and comparisons.

Independent probe-on/off hosts must keep the same ordinary trajectories and
final states. Gates check exact snapshot restoration, the parent-selected
branch's ordinary continuation, branch-specific history, identical window
streams, frozen parent/student weights, sample-to-state association, full
expected identity sets and save/load continuation. Terminal verification
rereads recorded artifacts against their earlier anchors. Failed gates and
measured regressions remain part of the report.

Implementation and measurements will be recorded here after the fixed run.
The earlier relative-loss objective, conditioned objective, ordinary online
feedback and canonical Chuck remain available with their existing contracts.
