# SPA: consequences from the same sentence state

The [first body experiment](../README.md) left an explicit question: its policy
received experience, but acquired weights changed zero of 48 actions. Were
better actions available at those states, and did an immediate reward describe
their later consequences?

This experiment measures every valid `KEEP`, `RESEED_LEFT` and `RESEED_RIGHT`
from each learned-arm state. An alternative beats `KEEP` at seven of 48 states
immediately. After four fixed host steps, the set of best actions differs from
the immediate set at 19 states. The original learned policy still changes
**0/48 choices through acquired weights**. The diagnostic branches supply
measurements; they do not train the agent.

## Registered comparison

[protocol.json](protocol.json) was frozen before the first scenario run:
SHA-256 `25aeed3248bcde9920a4a4eed29fd5ef06e5373da670c26fbb87696d819edcf4`.
The parent protocol, model, generation limits, reward weights and original
receipts remain intact. The body is the pinned 450,688-parameter native
SimpleLLM Dracula checkpoint at `80b3bd611ed8c937efdc481dc07895c8e13e345d`.
Its bounded sentence units, including the original 42/48 initial units ending
at the 64-character cap, are the units evaluated here.

Seeds 42 and 73 each contribute 24 snapshots, immediately before the learned
arm's ordinary decision. Each fork starts from the same complete sentence
chain, reseed counters, persistent agent and policy RNG, body weights and host
generation stream. Boundary targets allow two actions; interior targets allow
three. The experiment therefore contains **48 snapshots, 120 action forks and
360 measurements**.

The C host executes the intervention on a private chain. No diagnostic branch
calls agent `choose`, `observe`, imitation or policy replacement. The ordinary
decision preview uses the existing pure `select` API. The selected fork at
horizon zero must exactly match the subsequent ordinary execution, including
its tokens, metrics, action features, scores, cost and host RNG.

The three registered horizons are 0, 1 and 4 additional host reseeds. At hop
`h`, the fixed continuation targets `(original_target + h) % 4` and reseeds from
the left neighbor, or from the right at target zero. Each hop has a separately
constructed host RNG stream shared by all initial alternatives. The neighbor
prompt remains three characters, temperature 0.8, and generation min 12/max 64.
This continuation measures a declared host intervention, not the learned
policy's future rollout.

All seven metrics refer to the **original target index**, including at later
horizons. Raw before/after connectedness, coherence, novelty, repetition,
collapse and continuity are retained. The reward keeps the parent's exact
weighted delta formula and its `-0.05 * cost` term. Horizon cost is cumulative
generated characters divided by `64 * (horizon + 1)`. Initial, future and total
character costs are also retained separately. Each horizon's comparison uses
its own declared cost denominator.

Fork completion checks that the original agent, chain, reseed counters, body
parameters, training mode and empty tape remain unchanged. Diagnostic body
forward calls are counted separately; the ordinary counter is restored.

## Results

An opportunity means that at least one alternative exceeds `KEEP` by more
than `1e-7`. Selected regret is the best measured reward minus the reward of
the actual original-policy action. Winner sets include ties within `1e-7`.
These are measurements on the registered states, not estimates over other
bodies or prompts.

| Additional steps | States beating KEEP | Positive selected regret | Mean selected regret | Winner set changed from step 0 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7/48 | 12/48 | 0.00762913 | — |
| 1 | 14/48 | 21/48 | 0.00987689 | 9/48 |
| 4 | 14/48 | 17/48 | 0.00457661 | 19/48 |

The first registered state (seed 42, episode 0, step 0, target 0) shows why the
axes and horizon matter. Its actual policy chooses `KEEP`.

| Initial action | Reward at step 0 | Reward after 1 step | Reward after 4 steps |
| --- | ---: | ---: | ---: |
| KEEP | 0.000000000 | -0.018034888 | -0.007562072 |
| RESEED_RIGHT | +0.008609292 | -0.015011920 | -0.108164445 |

The immediate right reseed increases novelty by `0.303105294` while decreasing
local connectedness by `0.021688103` and coherence by `0.007229447`; repetition
increases by `0.005184334`. Its positive combined reward is not a coherence
improvement. Four steps later, `KEEP` has the better measured consequence.

Two complete executions reproduce all 27 retained trajectory/life/archive
artifacts exactly. With diagnostics on or off, both complete ordinary traces
and all ten final agent lives reproduce the original v1 identities. The
diagnostic branches add 32,661 body forward calls, recorded separately from
the unchanged ordinary totals of 2,803 and 2,789. These are operation counts,
not a throughput benchmark.

Seven immediate opportunities establish that useful alternatives under this
reward exist in some visited states. They do not establish a successful
learning rule for finding them. The original sparse experience and unchanged
0/48 weight-only action result remain available beside the scenario evidence.

## Gates and saved-state durability

The native fixture exercises production fork, action, measurement and restore
code with deterministic generated tokens. Eight damaged traces fail independent
parser checks. Four compiled defects fail their named gates: leaked host chain,
unpaired future RNG, reversed consequence sign and a wrong later metric target.
Compile failures and crashes never count as detection. The earlier fixture
accepted the wrong-target mutation. A [transcript-derived audit](fixture_audit.json)
preserves that command and outcome; its temporary artifacts were deleted and
source hashes were not captured. Distinct target repetition now makes the
same defect observable without changing the production experiment or protocol.

The optional scenario sink is create-only. Five path checks cover distinct
outputs, `./` aliases, future saved-life aliases and symlinks; an existing sink
is refused without changing its bytes. Ordinary output is opened only after
these checks.

The separate [PR #153 review](review.json) found a real persistence omission.
Agent save now opens and validates its parent directory, writes and fsyncs a
unique temporary file, closes and renames it, then fsyncs the directory. Failure
before rename preserves the previous checkpoint. A directory sync/close failure
after rename reports `NT_SPA_E_IO` with the new checkpoint already installed.
Policy arithmetic and the canonical 2,296-byte format stay unchanged.
[Durability receipts](durability.json) contain 10 passing cases/111 checks,
eight injected failures, cleanup/errno checks, an omitted-directory-fsync
mutation caught with exit 1, and a focused ASan/UBSan pass.

The review's metadata finding names source commit `6e834f3`, whose actual Git
object contains both `Quote:` and `Method:`. Merge commit `79707d8` has only
its title. Both exact messages are preserved in [review.json](review.json).

## The incomplete sanitizer trace

The original body sanitizer receipt remains PARTIAL with a failed artifact
gate. A follow-up using its immutable source and sanitized binary reproduced
the failure and separated the open descriptor from the published pathname.
The descriptor reached the complete 223,280 bytes while the pathname referred
to a different inode containing a 62,337-byte prefix. `/proc/self/fd` marked
the writer's original file deleted. All measured flush/close calls succeeded,
and rereading the open descriptor matched the full baseline FNV hash.

This identifies a pathname replacement during execution. The replacing actor
is unknown; process tracing was blocked. A subprocess-parent run of the same
binary and inputs retained the complete original SHA-256 trace, and a later
invocation rechecked its named file, raw mirror and compressed mirror. Seven
deliberate completeness mutations are rejected. The production writer is
unchanged. [trace_io.json](trace_io.json) retains the boundary evidence and
distinguishes complete artifacts from an exit-zero process.

## Reproduce

From the repository root, use a new empty output directory. The runner fetches
only the pinned public checkpoint/corpus, verifies their hashes, snapshots
sources, runs narrow gates and builds the native host. `--inputs "$SPA_INPUTS"`
can reuse previously prepared and hash-checked public inputs.

```sh
make check_spa_agent check_spa_scenarios BLAS_FLAGS= BLAS_LIBS=
python3 experiments/spa_agent/scenarios/run.py --output /tmp/spa-scenarios-run
make shared spa_agent_demo BLAS_FLAGS= BLAS_LIBS=
LD_LIBRARY_PATH=. make -j1 test BLAS_FLAGS= BLAS_LIBS=
make test_spa_legacy_parity
python3 tests/test_spa_agent_mutations.py --json /tmp/spa-mutations.json
```

The full CPU recipe passes all 38 commands. A first attempt stopped because an
existing ignored `test_multi_decode` binary had mode 0644. Its bytes were
preserved; rebuilding produced the identical SHA-256 with mode 0755, followed
by the complete successful run. The source of the mode change is unknown.
Old SPA parity remains 1,024 steps/114,688 identical bytes. Existing four
action/credit mutations are still caught.

Evidence is in [receipts.json](receipts.json), [verification.json](verification.json)
and [raw_traces.jsonl.gz](raw_traces.jsonl.gz). The archive contains six complete
streams, each record tagged with seed and `ordinary_off`, `ordinary_on` or
`scenarios`; its `raw` value retains each original C JSONL line. Receipts pin
the immutable body-run source hashes and separately identify later path-gate
additions. No generated executable, model weight or agent checkpoint is
committed.
