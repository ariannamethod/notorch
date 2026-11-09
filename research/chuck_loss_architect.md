# Chuck, the loss architect of notorch

A technical account of the training controller written in C inside notorch: what it does, what
the repository verifies about it today, and what has not been measured. Citations name a file and
line in the tree at the time of writing; a test citation names the test group and states what it
asserts. Log entries are cited by their dated heading in `NOTORCHLOG.md`.

## Summary

Chuck is the optimizer of notorch, the Arianna Method's C library that runs and trains neural
networks. Its Loss Architect layer puts a 163-parameter policy in charge of the global step-size
multiplier. At each training step the policy chooses HOLD, BRAKE or PUSH, and it is credited with
the loss change that choice was measured to cause. The repository verifies that the layer
reproduces canonical Chuck when it is absent, disabled or in legacy mode, refuses malformed input
without side effects, and saves and resumes exactly. Recorded experiments on two small models show
that measured consequences change what the policy chooses. They have not shown that the changed
choices lower held-out loss: in the eight SimpleLLM seed-runs recorded across four experiments, no
learned life finished below canonical Chuck, and the HeVLM body gives mixed results. Section 4
lists what remains unmeasured.

## 1. What Chuck is and what problem it answers

Training a neural network is a loop. Each step measures how wrong the model is on a batch of data
(the loss), computes for every weight which direction would lower that loss (the gradient), and
moves the weights a small distance that way. The distance is the step size, or learning rate, and
the component that applies the move is the optimizer. Chuck is notorch's own optimizer. Its
per-weight update is the familiar moment-based adaptive rule: a running average of each gradient
and of its square, bias-corrected, with the move scaled by their ratio (notorch.c:3432-3440). A
layer of controllers rescales that move while the run is in progress. The README describes the
idea as an optimizer that watches its own gradients, slows down when it should, freezes a
parameter that has stopped moving, and adds noise when it is stuck (README.md:384).

The effective step for a parameter is the base learning rate times three factors, a global
multiplier called dampen, a per-parameter dampen and a global lr_scale, with a noise term added
afterward (notorch.c:3398-3399, 3438-3440). Canonical Chuck, called "legacy" in the source, sets
those factors by fixed rules. The smoothed loss (decay 0.99, notorch.c:3006) is kept in a ring of
16 values (notorch.h:168); once the ring holds eight, its newest quarter is compared with its
oldest. A rise above 2% multiplies dampen by 0.97 and a fall below -2% multiplies it by 1.03,
inside the bounds 0.3 and 2.0 (notorch.h:169-174, notorch.c:3012-3024). A trend of magnitude below
0.001 for eight evaluations in a row switches on noise of amplitude 0.001 (notorch.h:175-177,
notorch.c:3027-3032). A parameter whose gradient norm stays below 0.01 for eight steps freezes
(notorch.c:3386-3388). Three consecutive stalled checks, taken every 1000 steps, halve lr_scale
(notorch.h:180-182, notorch.c:3050-3057).

Those constants were chosen by people (notorch.h:168-183). The Loss Architect, added on 2026-10-06
(`NOTORCHLOG.md`, entry "Chuck: Loss Architect"), asks whether the global part of that decision
can be acquired from experience. The first question of its experiment protocol is "whether
measured action consequences change future action selection while the canonical trajectory stays
reproducible" (docs/chuck-loss-architect.md:66-68). The second clause constrains the design: an
absent, disabled or legacy Architect runs canonical Chuck exactly (chuck_architect_impl.h:514-523;
test_chuck_architect `legacy`, section 3).

## 2. How it works

The core exposes two functions. `nt_tape_chuck_observe` returns an `nt_chuck_observation`: the loss
with its smoothed value and trend, the slow macro average and its best value, the three controls,
the gradient norm and its trend, the fraction of frozen parameters, and four counters
(notorch.h:293-299). It computes them on a copy of Chuck's state and changes nothing
(notorch.c:3243-3246; test_chuck_architect `observation`, tests/test_chuck_architect.c:195).
`nt_tape_chuck_step_action` executes one `nt_chuck_action` of seven kinds (notorch.h:263-272):
LEGACY runs the canonical rules, HOLD leaves the controls alone, BRAKE and PUSH multiply dampen by
0.97 and 1.03 inside configured bounds, and SET_DAMPEN, SET_LR_SCALE and SET_NOISE set a control
to an absolute value (notorch.c:3209-3229). A non-legacy action replaces the global legacy
reactions and leaves the per-parameter rules running (notorch.h:320-321; notorch.c:3023-3065). The
executor validates the whole request on a copy before it mutates anything (notorch.c:3464-3496).

`chuck_architect_impl.h` sits between the two. Its only calls into the tape are those two
functions (chuck_architect_impl.h:517-532, 629-636), and it lives outside the tape
(notorch.h:261-262). The mode is disabled, legacy or learned, and the first two forward to
canonical Chuck (chuck_architect.h:20-24). In learned mode the policy picks among HOLD, BRAKE and
PUSH and does not select the three setters, which a host can still call directly
(chuck_architect_impl.h:472-483). The header's title line calls the whole thing "a small action
language, acquired training experience" (chuck_architect.h:1). Nothing in this layer builds or
edits the loss function, the model graph or the data. Its entire influence on training passes
through the two calls above, and so through the step multiplier, lr_scale and noise amplitude.

The policy is a small network (chuck_architect_impl.h:400-433). Sixteen features, each bounded to
[-1, 1], feed eight tanh units, which feed three linear outputs, one score per action, for
128 + 8 + 24 + 3 = 163 parameters (chuck_architect.h:14-17). Thirteen features come from the
observation. The other three come from the previous decision: the relative loss change since it,
its trend, and the reward it earned (chuck_architect_impl.h:413-418), which is how the last
consequence reaches the next choice. A fresh life draws only its first-layer weights at random,
within ±0.15 from its seed; everything else starts at zero (chuck_architect_impl.h:320, 327-331),
so the three scores tie and the tie resolves to HOLD (chuck_architect_impl.h:476;
test_chuck_architect `credit`, tests/test_chuck_architect.c:427). By default, with probability
0.15, the policy picks uniformly among the enabled actions instead (chuck_architect_impl.h:72,
479-482).

Online, the caller runs the step and then evaluates the same batch again. The reward is
(before - after) / (|before| + 1e-6), clamped to [-1, 1], and a non-finite after-loss earns -1
(chuck_architect_impl.h:551-553). Only the selected head and the hidden layer learn, by Huber
regression (quadratic near zero, linear beyond one unit) toward the reward at learning rate 0.03,
with weights clipped to ±16 (chuck_architect_impl.h:556-572, 71). That signal covers one step on a
batch the model has just trained on. The scenario experiment measured what it misses: from 16
saved training states, PUSH gave the lowest immediate same-window loss at all 16, and the action
with the lowest future-probe loss after 16 updates was not PUSH at 7 of them
(experiments/chuck_loss_architect/scenarios/README.md:82-97). Two replay functions respond. The
caller restores a saved state, runs HOLD, BRAKE and PUSH forward from it, and passes the three
measured future losses with that state's features to `nt_chuck_architect_fit_comparison` or
`nt_chuck_architect_fit_conditioned` (chuck_architect.h:181-207). All three heads take one
mean-Huber step from a single pre-update snapshot (chuck_architect_impl.h:706-744). Targets are
measured differences from HOLD. The first function divides by the magnitude of the HOLD loss plus
1e-6; the conditioned one divides by the largest finite action effect in that state, floored at
1e-6 (|HOLD loss| + 1), so targets span [-1, 1] (chuck_architect_impl.h:717-718, 779-786). A
non-finite alternative gets target -1. Two more entry points, `nt_chuck_architect_feedback_frozen`
and `nt_chuck_architect_intervene`, let a trained policy act inside real training while its history
accumulates and its 163 weights stay fixed (chuck_architect.h:140-167).

A "life" is the saved Architect: configuration, weights, exploration generator, counters, temporal
features and any pending decision (chuck_architect.h:209-212). The file is 1016 bytes: the magic
NTCALIFE, a version, a length and a 64-bit FNV-1a checksum, then a 992-byte payload
(chuck_architect_impl.h:25-27, 914-918, 941-945). Loading validates size, checksum, every field's
range and a pending decision's cached features and scores, and changes the destination life only
on success (chuck_architect_impl.h:363-394, 435-462, 1010-1037). Saving writes a temporary file,
syncs it, renames it over the destination and syncs the directory (chuck_architect_impl.h:977-990).
`train_chuck` wires these pieces into a loop: before step 1 and every 64 steps after it, it
snapshots the training world, runs each action followed by 15 updates and four probe windows,
restores the world, takes 32 conditioned fits and lets the updated policy choose the real action
(docs/chuck-trainer.md:58-71).

## 3. What is verified today

The evidence has three layers: C tests that assert behavior on fixtures, mutation scripts that
check those tests can fail, and recorded experiments on real training runs. The make targets are
in Makefile:692-795 (`check_chuck_architect`, `check_chuck_scenarios` and their siblings). The
temporal fixtures use analytic bodies to isolate optimizer chronology (docs/chuck-scenarios.md,
"CPU checks").

- **Absent, disabled and legacy Architects reproduce canonical Chuck.** test_chuck_architect
  `legacy` (tests/test_chuck_architect.c:126-188) runs nine entry routes (the plain call; the
  action call with no action and with LEGACY; disabled and legacy Architects; an empty
  configuration as `{}`, whitespace and NULL; a NULL Architect) for 25 steps against a table
  captured from commit b14dd36, before the Architect existed (:95-124). Floats match within
  1e-7 + 2e-6·|x| (:15); counters, history ring and the freeze at step 24 match exactly
  (:171-182); no random draw is consumed (:184). tests/test_chuck_legacy_parity.sh (:8-20) builds
  notorch.c from commit 097fc06 and from the tree, runs 6000 steps with noise, freezing and macro
  decay, each required to occur (tests/chuck_legacy_parity.c:44-47), and requires `cmp` to find
  the traces identical (digest 306275db…, `NOTORCHLOG.md` entry "Chuck: Loss Architect"). Both
  sides are notorch code, so this guards against regression.
- **Bad requests and corrupt state are refused with nothing changed.** `action_rejection`
  (:244-275) sends 11 invalid actions (:248-255), unavailable actions, inverted or non-finite
  bounds, a NaN learning rate and an infinite loss; weights, moments, histories and the noise
  generator stay byte-equal to before (:90-93). test_chuck_actions_edge
  (tests/test_chuck_actions_edge.c:303-308) adds NaN, infinite and overflowing gradients
  (:118-131), eight corrupted per-parameter states (:142-157) and all seven combinations of
  too-short gradient and moment buffers (:174-190), each expected to return `NT_CHUCK_E_STATE` with
  state intact.
- **Credit reaches the head that earned it, once, and a refused step changes nothing.** `credit`
  (tests/test_chuck_architect.c:414-447) runs two identical lives through one HOLD step from loss 1
  with after-losses 0.5 and 1.5. The selected head's bias moves up in one life and down in the
  other, the unselected heads stay exactly zero, and the next action differs (:442-444).
  `policy_transactions` (:341-412): a NaN loss, or a core refusal after selection, leaves policy
  hash, body, noise generator and outputs unchanged (:382-395); a second step before feedback and a
  second feedback are refused (:401-409); a non-finite after-loss gives reward -1 (:404-406).
- **Save, resume and save failure are exact.** `resume` (tests/test_chuck_architect.c:478-539)
  saves a life with a pending decision, continues 16 steps, restores the body and loads the life,
  and requires identical pending credit, a bit-identical body, equal policy hash and equal bytes;
  a NaN-weight life fails to save and leaves the old file intact (:520-524), and a flipped or
  truncated file fails to load with the resident life unchanged (:525-535).
  test_chuck_architect_durability (tests/test_chuck_architect_durability.c:170-183) interposes
  open, fstat, mkstemp, fsync, rename and close, injects eight failures and asserts the call order
  `ODTFRPc` (:174). A failure before the rename leaves the prior 1016-byte file byte for byte
  (:131, :150); a failure after it returns `NT_CHUCK_E_IO` with the replacement installed
  (:145-147).
- **Inputs are strict.** `configuration` (tests/test_chuck_architect.c:279-329) refuses 46 malformed
  or contradictory JSON strings (:280-307), each with an error text and the destination unchanged.
  test_chuck_architect_state (tests/test_chuck_architect_state.c) parses the same JSON to identical
  bytes under a decimal-comma locale, with two threads in different locales converting 2000 times
  in total (`numeric_locales`, :63-126; skipped when no comma locale is installed, :88-92), and
  refuses 34 single-field corruptions of a pending life, each carrying a valid checksum
  (`hostile_lives`, :166-242, cases :182-205).
- **Interrupted, interleaved and mode-switched lives continue exactly.**
  test_chuck_architect_scenarios (tests/test_chuck_architect_scenarios.c): `mixed_replay`
  (:146-175) replays 1200 steps cut at 613, uses all seven action kinds and compares the whole body
  after every step; `interruptions` (:177-227) cuts a 96-step run at 8 points and 3 stages (before
  selection, after execution, after feedback) and resumes through the life file;
  `interleaved_lives` (:229-266) alternates two saved identities 192 times; `mode_transitions`
  (:268-312) keeps the acquired policy hash through legacy and disabled phases, matches ordinary
  Chuck on the body, and refuses a mode switch while credit is pending; `temporal_readout`
  (:319-350) shows an earlier consequence changing features and scores at fixed observation and
  weights.
- **Replay credit has the right sign, gradient and scale.** test_chuck_architect_future
  `measured_reversal` (tests/test_chuck_architect_future.c:86-118) fits the recorded SimpleLLM
  horizon-16 losses (HOLD 2.87523842, BRAKE 2.87194562, PUSH 2.87865996, synthetic features,
  :60-63) 128 times and asserts BRAKE-favoring targets and the order BRAKE > HOLD > PUSH;
  `gradient_and_determinism` (:214-251) matches the applied update at seven coordinates to a
  finite-difference derivative of an independently coded objective within 2e-4.
  test_chuck_architect_conditioned (tests/test_chuck_architect_conditioned.c) checks the scale rule
  and invariance under an eightfold loss rescaling (`conditioned_scale`, :51-96), opposite
  preferences held by one life in two states (`conditioned_choice`, :98-131), and that weight-frozen
  completion leaves all 163 weights unchanged over 32 steps of a two-weight body while history and
  counters equal ordinary feedback (`frozen_feedback`, :243-288).
- **An intervention equals the ordinary step.** test_chuck_lived (tests/test_chuck_lived.c):
  `selected_equivalence` (:83-111) intervenes with the action the policy would have picked and
  matches the ordinary step plus frozen feedback bit for bit over 32 steps; `unselected_resume`
  (:113-157) forces BRAKE against a known greedy PUSH and asserts the real pre-action scores and
  16 identical later steps after a save and load.
- **The host/device mirror protocol holds.** test_chuck_architect_device.c (:171-232) runs four
  40-step trajectories (typed actions with noise, legacy noise, CPU/device alternation, learned
  policy) and requires CPU, device and resumed-device states to agree within 3e-6 scaled by
  1 + |value| (:39-43, 111-130). The host-emulation backend (tests/chuck_device_host_backend.c) uses
  separate buffers and runs no CUDA kernel, and tests/run_chuck_architect_device.sh `host-mutations`
  (:42-69) deletes the first-moment download or invalidation and requires the gate to fail.
- **The trainer resumes across processes.** tests/test_chuck_trainer.py (nine tests): a 73-step
  run, a resume to 73 from step 31 and a resume from step 0 give byte-identical checkpoints
  (:93-109); SIGTERM saves a completed step whose resumed continuation matches an uninterrupted run
  (:137-178); dropping the saved window generator turns the gate red (:111-135).

Five of the C suites have a companion Python script that copies the sources, plants one defect,
compiles, and requires the named group to fail with a normal red exit; a compiler error or a crash
does not count as detection. The scripts define four defects for the architect gates, among them
BRAKE executing PUSH and a reversed credit sign (tests/test_chuck_architect_mutations.py:41-62),
three each for scenarios, conditioned and future credit
(tests/test_chuck_scenario_mutations.py:28-45, tests/test_chuck_conditioned_mutations.py:40-50,
tests/test_chuck_future_mutations.py:71-80), and four compiled plus seven semantic defects for
interventions (tests/test_chuck_lived_mutations.py:141-157, 188-194).

Behavior on real training is recorded under `experiments/chuck_loss_architect/`; paths below that
begin `scenarios/`, `conditional/`, `trajectories/` or `living/` sit there. All runs are CPU, F32,
on SimpleLLM (450,688 parameters, trained on the text of Dracula) and HeVLM (1,123,456
parameters, a Hebrew corpus), with two seeds per experiment (docs/chuck-loss-architect.md:71-79).
The quality measure is held-out cross-entropy on eight fixed windows of text the model did not
train on; lower is better. The first experiment ran 512 updates per arm with batch size one,
gradient clipping at 1, a constant body learning rate of 0.0003, and identical initial weights and
data windows in every arm (experiments/chuck_loss_architect/README.md:11, 23-26, 30-31, 38-39, 93).
The stock arm is the standard moment-based optimizer.

| Body, seed | Stock arm | Canonical Chuck | Learned Architect |
|---|---:|---:|---:|
| SimpleLLM, 42 | 2.568933 | 2.465579 | 2.539874 |
| SimpleLLM, 73 | 2.586830 | 2.531280 | 2.681840 |
| HeVLM, 42 | 1.571433 | 1.576577 | 1.571453 |
| HeVLM, 73 | 1.529242 | 1.587225 | 1.588265 |

Canonical Chuck ends below the stock arm on SimpleLLM at both seeds and above it on HeVLM at both
seeds. The legacy Architect matched canonical Chuck at every step and final weight byte in all four
pairs (experiments/chuck_loss_architect/README.md:106-107). In seed 73 the learned life chose
differently from the same life with its initial weights in 426 of 512 decisions on each body, and
in seed 42 in none (same file, :109-114). The conditioned replay objective raised the share of
optimal choices at fresh SimpleLLM states from 4 of 8 under the earlier relative-loss objective to
6 of 8, and its full 512-step deployments still ended above canonical Chuck on both SimpleLLM
seeds (conditional/README.md:118-121, 142-147).

The living loop of section 2 was recorded on fresh seeds 1711 and 2357, with 16,640 body updates
and 2,560 policy fits (living/README.md:67-72). Frozen measures the three branches and performs no
fits. Rotated assigns each action another action's measured outcome, as a control on the
action-to-outcome association (living/README.md:24-29).

| Body, seed | Canonical | Frozen | Living | Rotated credit |
|---|---:|---:|---:|---:|
| SimpleLLM, 1711 | 2.462270 | 2.556019 | 2.529372 | 2.635262 |
| SimpleLLM, 2357 | 2.498046 | 2.553459 | 2.567729 | 2.583310 |
| HeVLM, 1711 | 1.576849 | 1.566473 | 1.558620 | 1.569660 |
| HeVLM, 2357 | 1.560185 | 1.557627 | 1.555800 | 1.558470 |

Acquisition changed the next action at identical captured features in 16 of 32 acquisitions.
Living ended below frozen in 3 of 4 pairs and below rotated in 4 of 4, and below canonical Chuck on
both HeVLM seeds and above it on both SimpleLLM seeds (living/README.md:81-92). In the trajectory
experiment, canonical Chuck finished below every learned life in all four body-seed pairs
(trajectories/README.md:114-118). The standalone trainer took SimpleLLM held-out cross-entropy
from 4.80649185 to 2.538402 at step 1000 and 2.159829 at step 4000, executing 954 HOLD, 2,846
BRAKE and 200 PUSH with 3,024 branch updates beside the 4,000 host steps; its log entry states
that this shows a working trained model and continuation path and no advantage over another
optimizer (`NOTORCHLOG.md`, entry "Ship Chuck's standalone train / resume / generate path").

## 4. What is not yet measured

- **Advantage over the stock optimizer stack.** Recorded: two small bodies, two seeds, 512 updates,
  one constant learning rate, a split result by body, and seeds the living README calls "not a
  throughput benchmark or a population estimate" (living/README.md:71-72). Exploration 0.15
  (chuck_architect_impl.h:72), 32 fits and fitting rate 0.03 (living/README.md:17-19) are fixed by
  protocol with no recorded sweep (living/README.md:120-121). Probe: a per-arm learning-rate sweep,
  a sweep over those three settings, enough seeds for an interval, a body above 1.1 million
  parameters.
- **A learned global policy beating canonical Chuck on SimpleLLM.** A hypothesis; all eight
  SimpleLLM seed-runs went the other way. Probe: the longer-horizon subset, reversed body transfer
  and time/history ablations named in docs/chuck-learned-policy-research.md:61-62.
- **The cause of the SimpleLLM gap.** That brake saturation at the dampen floor drives it is a
  hypothesis; the seed-73 learned life brakes to the lower bound
  (experiments/chuck_loss_architect/README.md:124-127) and the regressing living run reaches the
  floor earlier than frozen (living/README.md:82-85). Probe: rerun with `dampen_min` raised or
  BRAKE masked in `enabled_actions` (chuck_architect.h:103-108).
- **Scale and modality.** A 10.2M-parameter body and a 1.5M vision-language body are listed as next
  steps (experiments/chuck_loss_architect/README.md:61, 63) and no recorded experiment uses them.
  Probe: the same four-arm protocol on each.
- **Learned policy against canonical Chuck past step 1000.** Canonical Chuck's macro
  learning-rate rule first checks at step 1000 (notorch.h:180; notorch.c:3050) and a non-legacy
  action replaces it (notorch.c:3053-3065). The paired experiments stop at 512 steps, and the
  4000-step trainer run has no canonical arm. Probe: a paired run of at least 5000 steps with a
  canonical arm.
- **Cost.** Online credit needs a second forward pass on the same window
  (experiments/chuck_loss_architect/README.md:35-37) and each acquisition adds three 16-update
  branches (docs/chuck-trainer.md:58-62). Probe: a paired wall-clock run of canonical, learned and
  living Chuck on one machine and binary.
- **Execution on CUDA.** The CUDA arm exits 77 without nvcc or a device
  (tests/run_chuck_architect_device.sh:72-75; tests/test_chuck_architect_device.c:344-347),
  scenarios/verification.json records it SKIPPED, and conditional/README.md:181 states that CUDA
  execution "remains pending hardware access". Probe: `sh tests/run_chuck_architect_device.sh cuda`
  on a machine with an NVIDIA device, recording the `CHUCK_DEVICE_OK backend=CUDA` line.
- **Exact continuation across machines.** The trainer promises it for the same executable,
  backend and thread settings only (docs/chuck-trainer.md:83-86). Probe: resume one checkpoint on a
  different CPU and BLAS and report the first diverging step.
- **Generated-text quality.** Only held-out cross-entropy is reported, and the 4000-step sample is
  described as "word fragments and short recognizable phrases" (`NOTORCHLOG.md`, entry "Ship
  Chuck's standalone train / resume / generate path"). Probe: a fixed generation evaluation
  against a canonical-Chuck model of equal budget.
- **Agreement with the reference optimizer.** The header says the C version diverged from the
  PyTorch reference (notorch.h:162-165), and no test or experiment runs both on one trajectory.
  Probe: feed one recorded loss and gradient sequence to both and report per-step differences.
- **Concurrent writers and abrupt kills during save.** The durability gate injects call failures
  (tests/test_chuck_architect_durability.c:20-23) and the trainer gate sends SIGTERM
  (tests/test_chuck_trainer.py:137-178); no probe saves from two processes to one path or sends
  SIGKILL between the file sync and the rename (chuck_architect_impl.h:984-987). Probe: a
  two-writer race and a kill-point sweep on real storage.

## 5. Lineage

Chuck lives in C inside notorch. It began as a PyTorch optimizer (`chuck.optimizer`); the C
version is a reduced per-parameter variant that its header says has diverged from that reference,
which carries more levels and finer granularity, and the header records its use in an Arianna LoRA
fine-tune on 2026-05-11 (notorch.h:158-167). notorch itself is the Method's route away from
PyTorch. The README describes a framework in pure C that runs and trains models with no Python
(README.md:53-54) and lists organisms that run on it, such as Yent's inference engine, built with a
C notorch kernel and no PyTorch (README.md:921). The Loss Architect layer takes its action
vocabulary from WOLFE and its action, consequence and acquired-state loop from Netta, two other
projects of the Method (docs/chuck-loss-architect.md:82-93).
