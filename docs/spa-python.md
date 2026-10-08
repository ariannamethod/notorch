# SPA from Python and other hosts

`import SPA` opens native Sentence Phonon Attention and Sentence Phonon Agent
to Python hosts. The module uses `ctypes` from the standard library. Sentence
embeddings, connectedness, modulation, perception, policy, learning and life
serialization execute in the same upstream `libnotorch` used by C hosts.

## Build and import

From the notorch checkout:

```sh
make shared BLAS_FLAGS= BLAS_LIBS=
PYTHONPATH=python python3 examples/spa_python.py
make test_spa_python BLAS_FLAGS= BLAS_LIBS=
```

For an external Python project, put the checkout's `python/` directory on
`PYTHONPATH`. It contains `SPA.py` and the existing `notorch.py` library locator.
Choose a library with `SPA.Native(path)`, `NOTORCH_LIB`, or automatic discovery
next to the checkout. The filename is `libnotorch.so` on Linux and
`libnotorch.dylib` on macOS. This module is distributed with the repository.

```python
import SPA

native = SPA.Native()                 # or Native("path/to/libnotorch.so")
config = SPA.Config.default(native=native, mode=SPA.Mode.LEARNED, seed=42)
agent = SPA.Agent(config, native=native)

sentences = [[1.0, 0.2, 0.0], [-0.4, 0.8, 0.3], [0.6, 0.1, -0.5]]
observation = native.perceive(sentences, target=1, temperature=0.8)
decision = agent.choose(observation)
print(SPA.ActionKind(decision.action.kind).name, decision.action.target)
agent.cancel(decision.sequence)       # this snippet leaves execution to its host
```

`Config.default()` calls the native defaults; its default mode is `LEGACY`.
`LEARNED` selects the policy. `DISABLED` returns `KEEP` without reserving credit.
`agent.state` is a detached snapshot, including config, policy, receipt history,
RNG and counters. Use `agent.set_policy(policy)` for an acquired-weight
counterfactual. Methods serialize access to each Agent life.

## Perception and the host loop

The original sensory functions are available through `native.embed_sentence`,
`native.connectedness` and `native.modulate_logits`. Inputs accept sequences;
matrices accept equal-width rows or a flat sequence with explicit `dim`.
The binding checks shape, integer bounds and finite float32 buffers before
passing a pointer. Invalid token IDs are refused. Outputs are owned ctypes
float arrays; `modulate_logits` returns a new array.

```python
embedding = native.embed_sentence([2, 0, 1], token_embedding_rows, alpha=0.85)
connectedness = native.connectedness(embedding, prior_sentence_embeddings)
logits = native.modulate_logits(logits, connectedness, strength=0.3)
```

Host execution stays outside the policy:

```python
decision = agent.choose(observation)
native.validate_action(decision.action, observation)
host.execute(decision.action)
consequence = SPA.Consequence(before=before, after=host.measure(),
                              regeneration_cost=host.normalized_cost())
receipt = agent.observe(decision.sequence, decision.action, consequence)
```

Here `before` and `host.measure()` are `SPA.Metrics` values. The seven named
components are local connectedness, global connectedness, coherence, novelty,
repetition, collapse and continuity; each is in `[0,1]`. Each host defines and
records those measurements. Generation cost is also in `[0,1]`. Native credit
uses the frozen config's coefficients and retains all components in the receipt.
The exact action witness and sequence must match the pending choice. Call
`cancel` when the host declines execution. A second decision while one is
pending raises `SPA.Error` with `Status.PENDING`.

`examples/spa_python.py` is an executable three-sentence token-host fixture.
It uses nine fixed word embeddings, implements neighbour reseeding, measures
all seven axes, supplies consequence credit, and resumes the saved native life
halfway through its loop. It prints the operational metrics and generated
sentences on every step. The pretrained-body experiment remains in
[`experiments/spa_agent/`](../experiments/spa_agent/README.md).

## Learning and persistence

`agent.imitate(observation, ActionKind.RESEED_LEFT)` invokes native supervised
policy training and returns its loss. Frozen-input comparison learning uses
the same policy parameters:

```python
experience = agent.capture(observation)
comparison = SPA.Comparison.from_outcomes(experience, {
    SPA.ActionKind.KEEP: keep_consequence,
    SPA.ActionKind.RESEED_LEFT: left_consequence,
    SPA.ActionKind.RESEED_RIGHT: right_consequence,
}, horizon=4)
receipt = agent.fit_comparison(experience, comparison, learning_rate=0.03)
readout = agent.score(experience)
```

Supply every valid action at the target coordinates. At the left edge omit
`RESEED_LEFT`; at the right edge omit `RESEED_RIGHT`. The native implementation
checks the action mask, frozen source association and common before metrics.
Comparison fitting changes policy bytes while retaining temporal memory,
online counters and RNG. Raw rewards, KEEP-relative targets, scores and Huber
losses are available in `ComparisonReceipt`.

For repeated executions from exactly the same captured state, build one
comparison per paired continuation and call:

```python
receipt = agent.fit_repeated(experience, comparisons, learning_rate=0.03)
```

The sequence contains 1–64 comparisons with the same source, horizon, mask
and before-measurements. C computes and clips each outcome's reward, averages
those rewards, and performs one update toward their KEEP-relative means.
The receipt contains the mean rewards; retain the individual comparisons and
their generation provenance alongside it. A single comparison reproduces
`fit_comparison` byte for byte.

For explicit per-state target conditioning, use:

```python
receipt = agent.fit_conditioned(
    experience, comparisons, learning_rate=0.03, scale_floor=0.001)
raw_mean_rewards = tuple(receipt.comparison.rewards)
conditioned_targets = tuple(receipt.comparison.targets)
actual_scale = receipt.scale
```

The floor is a required host choice: C divides the original KEEP-relative
targets by the larger of that floor and their largest absolute valid value.
It computes and retains the original mean rewards first. `ConditionedReceipt`
also exposes `scale_floor`; its nested comparison scores and losses refer to
the conditioned update. This changes the learning amplitude and relative
weighting of states. Keep original-reward evaluation and raw consequences
alongside it. Count one receives conditioning too. Policy-only mutation and
the canonical v1 saved life remain unchanged.

`agent.save(path)` and `SPA.Agent.from_file(path)` use the native canonical
checkpoint, including pending credit. Python and C load the same file.
`agent.load(path)` replaces an existing life transactionally;
`agent.reset_memory()` preserves policy, RNG and lifetime counters.
The host saves its own sentence field, body state and generation RNG alongside
the Agent life when restoring the complete environment.

Native refusals raise `SPA.Error`, with the exact `Status` and operation name.
Shape/type checks raise `ValueError` or `TypeError` before entering C. Native
save can return `Status.IO` after rename if directory sync or close fails;
the replacement checkpoint is already installed in that case, as specified
by [`spa_agent.h`](../spa_agent.h).

## ABI and parity gates

The loaded library exports two read-only scalar queries from `spa_binding.h`.
Before passing any structure pointer, `Native` checks 261 coordinates: native
versions, dimensions and enum values, plus size, alignment, every field offset
and every field size for all 15 exposed value types. A missing manifest or
mismatched coordinate raises `SPA.ABIError` with a rebuild instruction or the
specific difference. ABI metadata is compiled into that actual library.

`python/test_spa_binding.py` builds `tests/spa_layout.c` and a separate native
oracle in an isolated temporary directory. Its twelve groups cover all ABI
coordinates, malformed buffers/config/actions, pending credit, reset boundaries,
disabled and legacy behavior, comparison learning, repeated reward aggregation,
count-one parity, conditioned targets and invalid floors, policy-only updates,
and checkpoint refusal.
The oracle gives 59 byte-exact C/Python records: sensory output, observation,
eight imitation losses, sixteen comparison receipts, readout, ten decisions,
ten online receipts and ten canonical life hashes. Pending checkpoints and the
continued final checkpoint are byte-identical. A compiled offset mutation is
rejected before a structure-pointer call. Reproduce a machine-readable receipt:

```sh
python3 python/test_spa_binding.py --json python-receipt.json
```

## Molequla's integration boundary

Read-only inspection is pinned to molequla commit
[`c1cbc4792c422be4a5515cc68daad2b6c6603158`](https://github.com/ariannamethod/molequla/commit/c1cbc4792c422be4a5515cc68daad2b6c6603158).
Its primary Go body already links the system `libnotorch` through
[`cgo_notorch.go`](https://github.com/ariannamethod/molequla/blob/c1cbc4792c422be4a5515cc68daad2b6c6603158/cgo_notorch.go)
and [`cgo_notorch_cpu.go`](https://github.com/ariannamethod/molequla/blob/c1cbc4792c422be4a5515cc68daad2b6c6603158/cgo_notorch_cpu.go).
The direct boundary for that body is CGO over `spa_agent.h`. A Python engine
uses this module; a C engine includes the header and links upstream notorch.

Molequla's [`spa_coherence.go`](https://github.com/ariannamethod/molequla/blob/c1cbc4792c422be4a5515cc68daad2b6c6603158/spa_coherence.go)
normalizes its weighted sentence embeddings and scores each with a sum of
cross-sentence exponentials. The generation block in
[`molequla.go`, lines 5351–5449](https://github.com/ariannamethod/molequla/blob/c1cbc4792c422be4a5515cc68daad2b6c6603158/molequla.go#L5351-L5449)
splits sentences, selects the weakest, applies the `0.6 * mean` gate, regenerates
once from the neighbour's last three tokens and clips the replacement to a
sentence. The Agent can receive that host's existing scores through an explicit
`Observation`, with the host retaining its segmentation and action execution.
Those measurements have their own definitions; notorch's native convenience
perception uses the original softmax-maximum SPA connectedness.

The inspected Molequla source supplies a concrete next consumer boundary.
Its host integration will carry a separate parity fixture and acquisition
experiment. This change publishes the shared upstream primitive and Python
entry point.
