# Chuck: Loss Architect

Chuck inhabits the training trajectory. Its state meets a policy, the policy
chooses a typed action, and the measured consequence becomes acquired experience.

## State, policy, action

`nt_tape_chuck_observe` reads the pending loss/history state and the current
gradient structure. `nt_tape_chuck_step_action` executes the selected action.
`nt_chuck_architect_step` connects the observation, learned policy and executor;
`nt_chuck_architect_feedback` credits the successful action with its consequence.

The existing `nt_tape_chuck_step` selects the canonical legacy policy. An absent,
disabled, empty-config or explicitly legacy Architect follows the same trajectory.
The original 25-step golden vectors remain the baseline. The tape layout is
unchanged. Per-parameter gradient history, dampening, freezing and Adam moments
remain Chuck's local policy; the Architect selects the global action.

| Action | Consequence |
| --- | --- |
| `legacy` | Execute the canonical global and local policies. |
| `hold` | Retain global controls. |
| `brake` | Multiply global dampening by 0.97 within configured bounds. |
| `push` | Multiply global dampening by 1.03 within configured bounds. |
| `set_global_dampen` | Set bounded global dampening. |
| `set_lr_scale` | Set bounded global LR scale. |
| `set_noise` | Set bounded noise amplitude. |

The executor rejects unavailable actions, non-finite inputs, malformed bounds and
out-of-range setters before mutation. Custom actions replace global legacy
reactions, including mean reversion, noise decay and macro LR adjustment.

## A small learned life

The policy is a **163-parameter MLP**: 16 bounded features, eight tanh units and
three outcome heads for `hold`, `brake`, `push`. Loss/gradient trends, Chuck's
history, and the previous loss, trend and consequence carry the trajectory into
the next decision. Exploration uses a separate seeded RNG.

After a successful step, the caller evaluates the **same batch/window** again.
The selected head and hidden layer learn a Huber regression target:

```
improvement = before_loss - after_loss
target = clamp(improvement / (abs(before_loss) + 1e-6), -1, 1)
```

A non-finite consequence receives target -1 and a distinct receipt flag.
Receipts retain before/after loss, raw improvement, target, prediction and error
separately. Held-out loss is recorded separately by the experiment runner.
Each successful learned step opens one pending transition; feedback consumes it
once. Previewing an action creates no credit.

[The example JSON](../examples/chuck-loss-architect.json) declares the life,
available actions, bounds, learning rate, exploration rate and seed. Unknown or
duplicate fields/actions and malformed numeric syntax are refused. Empty input
selects legacy. JSON exposes the architecture's configuration.

Save/load preserves weights, configuration, temporal features, counters, RNG and
pending credit in a versioned little-endian representation with an integrity
checksum. The training body saves its own weights, optimizer and random streams.
Restoring both sides reproduces continuation.

## Experiment protocol v1

The first question is whether measured action consequences change future action
selection while the canonical trajectory stays reproducible. The training
comparison uses Adam, canonical Chuck, Architect legacy and learned Architect
with matched initial weights, windows, step count and learning rate.

| Body | Parameters | Reference commit | Corpus SHA-256 |
| --- | ---: | --- | --- |
| SimpleLLM, E128/L2/FF384/RoPE | 450,688 | `notorch-simple-llm@80b3bd611ed8c937efdc481dc07895c8e13e345d` | `16faafe5f8e4a958fe4437e561035b1f82235602ab6ef9801cafe2bcc25aef5c` |
| HeVLM, E128/L4/FF512/learned positions | 1,123,456 | `notorch-diffusion@1bc645071b3a0a3525fe89b8f45253f5faa28a1d` | `c00b8bcc31785798771aad5fb49f3b420b6a8eb3b7a20c28d0f09c967a52587e` |

The runner uses context 64, a contiguous 90/10 train/held-out split, fixed held-out
windows, and two training seeds. Same-window loss is the online credit signal;
next-window loss remains the training trajectory. Compact receipts retain source,
corpus, configuration, observations, actions, state identities and artifact hashes.
Weights and complete traces are retained outside Git.

## Lineage

- [WOLFE](https://github.com/ariannamethod/wolfe/tree/a04d2b7e6488656b715ca6b5b5ef5b1227eea79d):
  `doom/tools.json`, `doom/run.py:action_vector`, and the execution receipts give
  the finite action vocabulary and its checked execution boundary.
- [Netta Code / Netta Lee](https://github.com/ariannamethod/netta.code/tree/3dbb938662791efe1f5cc58cf97f9ed59a2b9aa5):
  `OutcomeHead`, `observe`, saved life and pending decision credit give the
  action → consequence → acquired state → altered choice loop.
- [Q](https://github.com/ariannamethod/q/tree/f5d00a36ecfcdb5e655e1576770f03d06d900e04)
  and [DOE](https://github.com/ariannamethod/doe/tree/588cf0009d3933dc5b7e72ac553fdd6e7932fa65):
  persistent experts and changing contribution remain available for a later
  training policy. Inference parliament stays in its separate harness work.

Implementation belongs to upstream notorch. Reference bodies supply architectures,
corpora and recipes. Their dependency migration remains a separate change.
