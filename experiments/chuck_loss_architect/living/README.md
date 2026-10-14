# Chuck acquires consequences inside its training life

The same 163-parameter life can now measure an action, acquire its consequence,
and choose again while the real training body continues. `--living` implements
this loop in C, using the existing snapshot, intervention, frozen feedback and
conditioned fitting APIs. Core optimizer arithmetic and the saved-life format
stay unchanged.

The [preregistered protocol](protocol.json) fixes two bodies, fresh seeds
1711/2357 and 512 steps. The input is the preceding experiment's sealed
`refit-parent` life. At eight points, before steps1/65/129/193/257/321/385/449,
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
Two living repeats save/reload the complete Architect at step256, one per body,
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
uses only development seed42 and16 host steps. It does not spend the new seeds.
All numerical execution is native C; Python launches processes, checks receipts
and summarizes measurements. Full traces, checkpoint lives and first-byte
anchors remain outside Git. Compact results and identities belong beside this
protocol after execution.

The H16 continuation is frozen **before** the 32 fits. Subsequent updated choices
can still differ from that measured continuation. This experiment brings
acquisition into the current life; it does not establish that this remaining
credit mismatch is solved. There is no capacity increase, recurrent model,
hyperparameter sweep or inference-side DOE change. CPU F32 results cannot stand
in for a GPU experiment, and the mechanism need not beat canonical Chuck to
demonstrate acquired action selection.
