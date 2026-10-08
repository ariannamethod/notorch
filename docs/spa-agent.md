# SPA: Sentence Phonon Agent

Sentence Phonon Attention supplies the sensory field. Sentence Phonon Agent
remembers its own actions, chooses an operation on that field, and acquires the
operation's measured consequence. Both live upstream in notorch.

`spa_agent.h` / `spa_agent.c` contain a 267-parameter policy: 29 compact inputs,
eight tanh units, and three action-value heads. The host executes `KEEP`,
`RESEED_LEFT`, or `RESEED_RIGHT`. Its model, tokenizer, sentence segmentation,
generation RNG, and candidate acceptance rule remain host responsibilities.

## Perception and the registered legacy rule

`nt_spa_embed_sentence`, `nt_spa_connectedness`, and `nt_spa_modulate_logits`
retain their existing implementation and ABI. The Agent is an additional
translation unit. A null or disabled Agent selects `KEEP` without changing its
life or the host's perception and generation state.

The optional `nt_spa_agent_perceive` consumes a contiguous sentence-embedding
field. It calls ordinary SPA connectedness against the other sentences, folds
the current embedding into four coordinates, and measures adjacent cosine
continuity and maximum pairwise similarity. A host may instead supply an
observation from its own sentence field. `spa_agent.h` specifies the ranges and
definitions of these features.

Q supplies the behavioral reference. In the C implementation at
[`f5d00a3`](https://github.com/ariannamethod/q/blob/f5d00a36ecfcdb5e655e1576770f03d06d900e04/postgpt_q.c),
the host chooses the earliest weakest sentence, then applies:

```c
sentence_score < mean_sentence_score * (0.52f + 0.18f * (1.0f - phase_gate))
```

The registered legacy policy uses this strict comparison and reseeds from the
left neighbor when present, otherwise the right. Q's separate host acceptance
rule is `new_score > 0.7f * old_score || new_length > old_length`.
The Agent API leaves acceptance to its host.

Q's sensory scores are sums of exponentiated, distance-biased dot products of
normalized 32-dimensional sentence embeddings. Upstream SPA connectedness is
the maximum softmax weight over raw scaled dot products. The policy accepts
either host's declared scores; the legacy decision fixture registers the rule
independently of those different sensory definitions.

The pinned Q geometry also yields a concrete legacy result. For 12 vectors of
norm at most one, each undirected edge is between
`m = exp(-1/sqrt(32) + .1/12)` and `M = exp(1/sqrt(32) + .1/2)`.
Symmetry bounds the weakest score divided by the mean from below by
`12*11*m / (2*11*m + 110*M) = 0.71228839`, above the maximum trigger ratio
`0.70`. The [source-extraction probe](../experiments/spa_agent/q_legacy_geometry.py)
runs the exact pinned C scoring function: 4,100 fields and three phase values
produce zero reseeds, with minimum observed ratio `0.734477997`.
[Receipts](../experiments/spa_agent/q_legacy_geometry.json) preserve the source
hash, formula, compiler, and measurements. Controlled low-score fixtures
exercise the registered action rule's reseed branches.

## A decision's life

`select` previews with a copy of the private RNG. `choose` opens one pending
decision with its sequence, observation, features, hidden activations, scores,
and selected action. The host executes that action and reports its identity
and consequence to `observe`. Feedback consumes the decision exactly once.
An unexecuted action can be cancelled explicitly.

The persistent life contains the policy weights, private RNG, frozen
configuration, counters, eight raw action/outcome receipts, and exponential
history features. Pending lives refuse a second decision, imitation, policy
replacement, or memory reset. `reset_memory` clears temporal experience while
preserving acquired weights and lifetime state. `set_policy` replaces only
weights, providing the exact acquired-weight counterfactual boundary.

Imitation uses supervised softmax cross-entropy. Consequence learning fits
the selected action-value head to the measured reward with a Huber gradient
clipped to [-1,1]. The policy operates
entirely in native C; it consumes compact sentence observations and history.

The binary save format records canonical little-endian fields, binary32
weights and measurements, format/perception/reward versions, checksum, and
pending state. Load validates a temporary life before replacing the resident
one. Save writes a unique temporary file beside the destination, flushes and
fsyncs it, atomically renames it, and fsyncs the parent directory. Failures
before rename preserve the previous checkpoint. A directory sync or close
failure after rename returns an I/O error with the new checkpoint already
installed.
Same-platform save/resume reproduces continuation exactly. Hosts save
their sentence field and model/generation state alongside the Agent.

## Consequences stay separate

Every receipt retains before and after values for local connectedness,
global connectedness, coherence, novelty, repetition, collapse, and continuity,
plus normalized regeneration cost. Their operational definitions are fixed
in each host experiment's protocol.

The default reward is the following bounded combination, with all deltas
computed as `after - before`:

```text
clamp(0.15*delta(local) + 0.15*delta(global) + 0.20*delta(coherence)
    + 0.20*delta(novelty) - 0.15*delta(repetition) - 0.10*delta(collapse)
    + 0.05*delta(continuity) - 0.05*regeneration_cost, -1, 1)
```

Reward weights are part of the frozen life configuration. Individual axes
remain available for comparisons, including repetition and sentence collapse.
The [experiment protocol](../experiments/spa_agent/protocol.json) fixes the
host's metrics, horizon, seeds, and comparison arms before its runs.

The [common-state scenario experiment](../experiments/spa_agent/scenarios/README.md)
forks each valid host action from the same pre-decision state, measures fixed
later continuations and checks an exact return to the ordinary trajectory.
These diagnostic consequences are retained separately; ordinary learning
continues to use its registered immediate reward.

## Learning comparisons of later consequences

`nt_spa_agent_capture_experience` captures the 29 perception and temporal
features, sentence coordinates and source-life hash before an intervention.
`nt_spa_agent_score_experience` reads that same captured field with the current
policy and returns a bounded greedy action. It preserves the agent and its RNG.

`nt_spa_agent_fit_comparison` receives the measured consequences of every valid
action at one host-declared horizon. The existing reward formula gives each
action a reward; subtracting the same-state KEEP reward gives its learning
target. All available heads fit their mean Huber loss in one simultaneous
update. The hidden gradient uses the same pre-update weights as the head
gradients. The host chooses the rate and records the fit receipt, including
raw rewards, relative targets, scores and loss before and after the update.

Comparison replay changes policy weights. The online history, counters,
configuration and RNG remain at their saved values, and the canonical v1 life
format retains the acquired weights. A pending online decision refuses replay.
Source bindings, action masks, typed coordinates and common before-measurements
are validated before an update. Hosts retain the actual measurement provenance
alongside captured experience.

The [future-credit protocol](../experiments/spa_agent/future/protocol.json)
fixes training on retained sentence scenarios, an immediate-outcome control,
a declared shuffled-outcome control, and evaluation on new generation seeds.

`nt_spa_agent_fit_repeated` accepts 1–64 comparisons of the same captured
state, action mask and horizon. Every action and repetition must share the
same before-measurements. Native reward calculation and clipping happen
separately for each outcome; their double-precision sum is averaged and
rounded to float once per action. The target is that mean reward minus
the mean KEEP reward. One simultaneous mean-Huber update fits all valid heads.
Count 1 delegates to `fit_comparison` and preserves its exact state and receipt.

The existing `nt_spa_comparison_receipt` reports mean rewards, relative targets,
scores and losses. The caller retains the repeat count, raw consequences and
their source/RNG provenance. Validation covers the entire comparison array
before mutation; replay preserves temporal state, configuration, RNG and the
v1 save format. The [paired-continuation protocol](../experiments/spa_agent/replicates/protocol.json)
uses eight paired continuations per state with the original policy capacity
and update budget.

## Conditioning the learning targets

`nt_spa_agent_fit_conditioned` accepts the same 1–64 paired comparisons, a
learning rate, a finite positive `scale_floor`, and an
`nt_spa_conditioned_receipt`. It first computes exactly the rewards and
KEEP-relative float targets used by `fit_repeated`. For each captured state:

```text
delta[action] = float(mean_reward[action] - mean_reward[KEEP])
scale = max(double(scale_floor), max_valid abs(double(delta[action])))
target[action] = float(double(delta[action]) / scale)
```

The existing simultaneous mean-Huber update fits these conditioned targets.
Only valid action heads participate; KEEP retains target zero. The receipt's
`comparison` contains original mean rewards, conditioned targets, scores and
losses, alongside the supplied float floor and the actual double scale.
The host retains every raw consequence and its provenance. A rate of zero
produces the receipt without changing policy bytes.

Every valid head in a state uses the same positive scale. This changes gradient
amplitude and relative weighting between states.
The floor limits amplification of near ties; conditioning does not establish
that a small, noisy advantage is reliable. Evaluate acquired actions using
the original reward and separate raw measurements. One comparison still
receives conditioning; `fit_repeated` remains the unconditioned API.

Replay changes only policy weights and keeps the canonical v1 life format.
Validation, pending-credit refusal and transaction boundaries match repeated
learning. The [fixed conditioning protocol](../experiments/spa_agent/conditioned/protocol.json)
specifies its floor, controls and new-state evaluation before fitting.

## Lineage and research sources

The code lineage is **PostGPT → SPA → Q → Sentence Phonon Agent**.
PostGPT's MetaWeights form a probability field; its Content and RRPRAM heads
combine content relations with positional patterns. Q develops a sentence
chain, cross-sentence perception, reseeding, and persistent coherence/phase
state. The upstream helpers entered notorch in
[`609dbfd`](https://github.com/ariannamethod/notorch/commit/609dbfd6d0fff2e9fa1a581d0edc7ccc2df3be96).

| Reference | Contribution studied |
| --- | --- |
| [PostGPT](https://github.com/ariannamethod/postgpt) | MetaWeights, probability fields, Content/RRPRAM architecture. |
| [Q](https://github.com/ariannamethod/q) | Sentence chains, SPA perception, reseeding, coherence/phase memory, candidate acceptance. |
| [SONAR paper](https://arxiv.org/abs/2308.11466) and [code](https://github.com/facebookresearch/SONAR) | Fixed-size sentence representations with encoders and decoders. |
| [Large Concept Models](https://arxiv.org/abs/2412.08821) | Autoregressive computation over sentence representations, with separate sentence encoder and decoder. |
| [WOLFE](https://github.com/ariannamethod/wolfe/tree/a04d2b7e6488656b715ca6b5b5ef5b1227eea79d) | Small model selects a finite typed action; the environment executes it. |
| [netta.code](https://github.com/ariannamethod/netta.code/tree/3dbb938662791efe1f5cc58cf97f9ed59a2b9aa5) | Executed decisions receive consequence credit, and saved experience changes later actions. |
| [Chuck: Loss Architect](chuck-loss-architect.md) | Contemporary separation of observation, learned policy, typed action, and acquired consequence in notorch. |

SONAR/LCM supply conceptual inspiration for the computational level. SPA Agent
has its own C implementation and uses the existing native sentence field.
There are no added external model or runtime dependencies. Chuck, DOE, SPA,
and tool execution retain their individual environments and interfaces.

Predicting a semantic target for a token model to realize is a later research
direction. The implemented action vocabulary operates on sentence trajectories.
