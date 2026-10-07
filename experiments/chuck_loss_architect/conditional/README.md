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
