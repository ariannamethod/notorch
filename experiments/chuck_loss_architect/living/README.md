# Chuck acquires consequences inside its training life

The same 163-parameter life can now measure an action, acquire its consequence,
and choose again while the real training body continues. `--living` implements
this loop in C, using the existing snapshot, intervention, frozen feedback and
conditioned fitting APIs. Core optimizer arithmetic and the saved-life format
stay unchanged.

The [preregistered protocol](protocol.json) fixes two bodies, fresh seeds
1711/2357 and 512 steps. The input is the preceding experiment's sealed
`refit-parent` life. At eight points, before steps 1/65/129/193/257/321/385/449,
the host captures its complete current world. Three branches force HOLD, BRAKE
or PUSH once, then let the current frozen policy select fifteen actions from
actual branch history. The H16 mean loss on the next four fixed training windows
provides the measured action comparison. Held-out loss remains a separate report.

Every branch restores the source body, gradients, moments, Chuck controls,
Architect history, tape and RNG. Acquisition performs 32 native conditioned
fits at learning rate .03; only policy weights may change. The host then selects
its next action. Receipts retain scores and choices before and after fitting at
identical source observations, alongside the measured outcomes and target scale.
Same-window feedback advances history between acquisitions without regression.

| Arm | Acquisition | Question |
|---|---|---|
| canonical | Ordinary Chuck | Reference trajectory |
| frozen | Measures three branches; no fits | What does acquisition change? |
| living | Fits measured H/B/P credit | Can current consequences alter the next choice? |
| rotated | Assigns H/B/P the measured P/H/B losses | Does the action-to-outcome association matter? |

The negative control changes credit deliberately and identifies it in every
receipt. It is not evidence of a wiring defect. A separate compiled defect
silently rotates the **living** arm's outcomes and must fail the receipt gate.

Four frozen probe-off runs check that diagnostics preserve all host transitions.
Two living repeats save/reload the complete Architect at step 256, one per body,
while the training body remains in process. They check all acquisition and host
events and final policy bytes. This is not out-of-process body resumption.
The full budget is **16,640 body updates and 2,560 policy fits**, including these
controls. BRAKE selections, actual dampen decreases and saturated BRAKEs are
counted separately.

```sh
python3 tests/test_chuck_living.py --json /absolute/path/living-gates.json
python3 experiments/chuck_loss_architect/living/run.py \
  --inputs /absolute/path/recovered-trajectory-inputs \
  --output /absolute/path/new-living-run
```

The runner refuses a dirty production tree and authenticates the previous
results, input life, corpora, tokens and vocabulary. `--smoke --allow-dirty`
uses only development seed 42 and 16 host steps. It does not spend the new seeds.
All numerical execution is native C; Python launches processes, checks receipts
and summarizes measurements. Full traces, checkpoint lives and first-byte
anchors remain outside Git. The completed run is recorded in
[results.json](results.json), with complete [held-out curves](heldout_curves.csv),
[action counts](actions.csv), [paired finals](paired_final.csv),
[independent audit](audit.json), and [archive identity](archive.json).

Clean published source `52dca982df285d06c71e58c0d458839702be746c` executes all
22 processes: 16,640 body updates and 2,560 policy fits. Both bodies use F32,
context 64, body LR .0003 and 2 SIMD threads on an AMD EPYC 9V74 host; compiler,
memory state, pinned corpus/token identities and source hashes are in the result.
These are two fixed seed repetitions, not a throughput benchmark or a population
estimate. Final held-out loss after 512 steps:

| Body | Seed | Canonical | Frozen | Living | Rotated credit |
|---|---:|---:|---:|---:|---:|
| Simple (450,688) | 1711 | 2.462270 | 2.556019 | 2.529372 | 2.635262 |
| Simple (450,688) | 2357 | 2.498046 | 2.553459 | 2.567729 | 2.583310 |
| HeVLM (1,123,456) | 1711 | 1.576849 | 1.566473 | 1.558620 | 1.569660 |
| HeVLM (1,123,456) | 2357 | 1.560185 | 1.557627 | 1.555800 | 1.558470 |

Living improves frozen in 3/4 pairs and rotated credit in 4/4. It improves over
canonical on both HeVLM seeds and regresses on both Simple seeds. Simple 2357
also regresses against frozen by .01427006. Its first dampen-floor entry moves
from 284 to 202 even though total BRAKE selections are 340 versus 339. Aggregate
action counts alone do not describe the acquired trajectory.

Across the four primary living runs, **16 of 32 acquisitions change the next
action at identical captured features**, with all non-weight state preserved:
Simple 5/3 changes, HeVLM 6/2. Earlier acquisition also changes later choices:
the first living/frozen action divergences are steps 48/82 and 39/43 respectively.
Every pair has an identical preceding training path, current loss, gradient,
Chuck controls, observation and features at that first divergence.

A direct witness is Simple 1711 at checkpoint 129. Measured H/B/P future losses
are 2.63455939 / 2.63477254 / 2.63438725. The 32 native fits use targets
`[0, -1, .80760628]`, changing scores from `[0, -.286639512, -.323161602]` to
`[0, -1.08126473, .72249949]`. HOLD becomes PUSH; the restored host immediately
executes that PUSH at the same features. The receipt preserves every fit, the
raw losses, scale .00021314620971679688, and both saved lives.

All 512 selected frozen-branch transitions, four complete probe-off controls and
two living save/load continuations are exact. Independent receipts analysis
authenticates 296 artifacts and 112 saved pre/post-fit life pairs; no non-weight
bytes change during fitting. Full-run and smoke artifact integrity failures=0.
The [native gate](verification.json), [smoke](smoke.json) and
[integration report](integration.json) retain their separate work budgets.
The archive verifies 350 full-run byte anchors, 102 smoke anchors and all 932
members, including the complete raw audit and the negative credit fixture.

Recheck the raw archive without executing a model:

```sh
python3 experiments/chuck_loss_architect/living/audit.py \
  --run EXTRACTED/raw --output NEW_AUDIT.json \
  --source-commit 52dca982df285d06c71e58c0d458839702be746c
```

The H16 continuation is frozen **before** the 32 fits. Subsequent updated choices
can still differ from that measured continuation. This experiment brings
acquisition into the current life; it does not establish that this remaining
credit mismatch is solved. There is no capacity increase, recurrent model,
hyperparameter sweep or inference-side DOE change. CPU F32 results cannot stand
in for a GPU experiment, and the mechanism need not beat canonical Chuck to
demonstrate acquired action selection.
