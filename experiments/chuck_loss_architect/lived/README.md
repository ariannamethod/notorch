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

## Measured result: changed futures, deployment regression

The first full fixed run completed on clean source
`6029e1378986aa91518cc90ef82032a36a5cc6dd`, under the preregistered
protocol SHA-256 `790f4ff88fd089331f2139dbe5e78d83e838d91d0c2214625e606903bea932b5`.
Both students changed their acquired choices. **Neither student beat the parent
on final held-out loss in any of the four complete deployments.** This result
includes both bodies and both previously unseen seeds; no recipe was changed
and no world was discarded after evaluation began.

Changing only the continuation changed the exact set of best H16 actions in
11/32 new worlds: 6/16 SimpleLLM and 5/16 HeVLM. On development worlds it changed
7/32. Thus the future policy changes what an intervention earns. For example,
on SimpleLLM seed 701 before update 64, BRAKE is best under fifteen HOLD updates;
HOLD and PUSH tie for best when the parent continues selecting. Both branches
begin at the same world with global dampening 2.

Under the primary **parent-policy continuation**, the new-state comparisons
are below. Regret is selected H16 future loss minus the measured best loss at
that same world; lower is better. Counts include all actions tied at the exact
F32 minimum. Displayed mean regrets are rounded; full precision is retained in
the tables and receipts.

| Body | Life | Best action | Mean regret |
| --- | --- | ---: | ---: |
| SimpleLLM | `parent` | 10/16 | 0.000056028 |
| SimpleLLM | `lived-hold` | 10/16 | 0.000039056 |
| SimpleLLM | `lived-policy` | 12/16 | 0.000036791 |
| HeVLM | `parent` | 9/16 | 0.000073545 |
| HeVLM | `lived-hold` | 9/16 | 0.000362307 |
| HeVLM | `lived-policy` | 9/16 | 0.000362307 |

The POLICY student improves the SimpleLLM common-state count from 10/16 to
12/16 and lowers mean regret. On HeVLM both students still choose a best action
in 9/16 worlds, yet mean regret increases from approximately .000073545 to
.000362307. The same best-choice count can conceal more costly misses. Both
continuation tables, all six fixed/learned readouts and every miss remain in
[common_state.csv](common_state.csv).

Complete 512-update deployments answer a different question. Final held-out
losses are:

| Body / seed | Canonical | HOLD | PUSH | Parent | `lived-hold` | `lived-policy` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleLLM / 701 | 2.55249667 | 2.60917759 | 2.54636097 | 2.61984873 | 2.70259333 | 2.6376853 |
| SimpleLLM / 907 | 2.54698443 | 2.59399152 | 2.55647969 | 2.59760356 | 2.70563793 | 2.62439394 |
| HeVLM / 701 | 1.59340107 | 1.55677295 | 1.56599045 | 1.56524408 | 1.56944704 | 1.56889033 |
| HeVLM / 907 | 1.5716759 | 1.55485332 | 1.56948733 | 1.57769823 | 1.58197331 | 1.58368015 |

The POLICY student beats the HOLD student on three of four final comparisons,
but both finish behind the parent in every case. All learned deployments use
HOLD, BRAKE and PUSH. This is a measured regression after a real change in
acquired behavior.

The traces locate an early trajectory change. On HeVLM, the HOLD student first
chooses BRAKE where the parent chooses PUSH at update 2 on both seeds. The
POLICY student first makes that choice at updates 8 and 7. At each first
separation, the recorded observation, features, pre-action Chuck state and
before-loss still agree exactly. Dampening first reaches its floor at update
42 for the HOLD student and 61 for the POLICY student, compared with 216 and
220 for the parent. These facts identify changed choices and the subsequent
trajectories recorded in this run.
Late HOLD/BRAKE action aliases at the floor remain explicit ties in the
common-state comparisons.

[deployment.csv](deployment.csv) retains all five held-out checkpoints for all
24 primary deployments and four save/load repeats. Its action and history
counts are **final-run totals**, repeated on each curve row. Learned-policy
weights remain frozen throughout; only real feedback history advances.
Canonical and fixed-action arms have no learned-policy frozen-weight claim.

## Exact receipts and reproduction

[results.json](results.json) contains the compact measured receipt: all 128
state/continuation records (64 distinct worlds), their observations, features,
three measured action consequences and exact F32 conditioned targets; sealed
life identities; all common-state summaries; complete deployment curves;
source/probe and resume parity; and raw artifact identities.
[smoke_validation.json](smoke_validation.json) is the separate seed-42 wiring
run. The first smoke and first full numerical attempt both passed.

The full run executed 28,672 body updates and 32,768 native policy fits. Gates
validated 4,096 probe-on/off source transitions, 1,024 selected-branch ordinary
continuation transitions, all 6,144 fork updates, four exact policy save/load
continuations, and four independently reproduced parent deployments. Terminal
verification reread 707 anchored identities and all semantic traces; the final
raw inventory contains 735 artifacts. Source hashes and all sealed lives were
unchanged. The two students were sealed before seeds 701/907 produced outcomes.

The raw result is SHA-256
`a684c8c9863b9c08973880d6efdfd1ea3c6197a1cd894fcf4deaf4b9c04e0354`.
The complete numerical archive `chuck-lived-final-6029e13-raw.tar.gz` has
941,976,873 bytes, 737 verified members and SHA-256
`7773c68e24245227467d526228b2149193b65768da449fa5571c6b8da69e9960`.
It retains corpora, initial and final bodies, policy lives, every raw action and
outcome trace, source/binary identities and the archive index. Two independent
member-reading methods authenticated the same bytes; the slower backward-link
reader completed normally. These large raw artifacts remain outside Git.

To reproduce, use the measured source commit and the verified preceding
conditional raw run identified in [protocol.json](protocol.json):

```sh
python3 experiments/chuck_loss_architect/lived/run.py \
  --previous-run "$CONDITIONAL_RAW" --output "$RUN_OUTPUT"
```

Use a fresh output directory on the host's temporary filesystem. The runner
builds against the current upstream tree, checks the preceding run's identities,
acquires the fixed development worlds, seals both students, and only then runs
the new seeds. This receipt covers CPU F32/SIMD with two threads, compiler and
binary hashes in `results.json`; it contains no GPU measurement.

The next architectural question follows from the retained failure: how can
Chuck value repeated choices over the longer trajectory that they create?
The present H16 parent-continuation experience is real and changes its next
choices, but this fixed round does not improve complete deployment. The earlier
relative-loss objective, conditioned objective, ordinary online feedback and
canonical Chuck remain available with their existing contracts.
