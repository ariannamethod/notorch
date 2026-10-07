# SPA acquires future-action choices; KEEP wins the new-state comparison

The native 267-parameter policy learns from paired sentence-action consequences.
On 48 new sentence states, its acquired H4 weights change **17/48 choices**
with the captured input, action history and RNG held fixed. Its mean reward
advantage over KEEP is **−0.002473649** at the registered four-step horizon.
The same policy beats fixed LEFT and RIGHT; KEEP and the equal-budget H0
policy score higher. The correctly associated H4 policy also scores below
the shuffled-H4 control. These are the recorded outcomes of this experiment.

Two complete executions reproduce **44 artifacts byte for byte**, including
datasets, fitted lives, every fit/readout, ordinary trajectories, scenario
trajectories and compressed archives. They repeat the same 48 evaluation
states; the sample count remains 48.

## Frozen question and body

[`protocol.json`](protocol.json) was frozen before experiment fitting and
before generating seeds 101/211. SHA-256:
`134460b76ff7fecc9934eea8187d8aac4f6d0154921db5989de86cc482ee1861`.

The body is the same frozen **450,688-parameter SimpleLLM Dracula model** from
[`ariannamethod/notorch-simple-llm`](https://github.com/ariannamethod/notorch-simple-llm),
commit `80b3bd611ed8c937efdc481dc07895c8e13e345d`. Its checkpoint SHA-256 is
`22ce6e5dfe4efb4a00477a9652784970dc53a69fea2825174398af749acda00d`.
The native host links the upstream notorch arithmetic. The body has width 128,
two layers, four heads, FFN 384, context 64 and 94 Unicode characters.

Training consumes all 48 retained states from seeds 42/73 and their already
measured alternatives in the [parent scenario experiment](../scenarios/README.md).
Evaluation consumes 48 newly generated states from seeds 101/211. Each source
state retains 29 float32 features, typed coordinates, source-life and body
identities, RNG witnesses, and the raw outcome-record identity for every action.

Each fitted arm receives 512 epochs in fixed order, SGD rate 0.03: **24,576
comparison updates per arm**, 73,728 per execution. The arms start from the
same seed-1 learned policy with exploration zero:

| Life | Training target | Question |
|---|---|---|
| initial | No fitting | What does the original readout choose? |
| h4 | Measured H4 action reward minus same-state KEEP reward | Does later credit improve later choices? |
| h0 | Measured H0 advantage, identical fitting budget | Does the credit horizon matter? |
| shuffled_h4 | H4 outcomes from the next training row at identical target coordinates | Does correct state/outcome association improve choice? |

The shuffle is the registered cyclic donor mapping within `(sentence_count,
target)` groups. Feature-source identity and actual outcome-donor identity are
recorded separately. Fixed KEEP/LEFT/RIGHT controls have no fitted parameters;
LEFT or RIGHT falls back to KEEP at its unavailable boundary.

All four canonical lives and the complete training trace are saved and hashed
**before the first evaluation body process**. Readout then remains pure and
greedy. The existing reference host supplies the evaluation states; its ordinary
online learning recipe is unchanged. Each frozen policy's choice is joined to
the actually executed alternative from that same state.

## Measurement

H0 is the immediate intervention; H1 and H4 add one or four fixed host reseeds.
Future target is `(original_target + hop) mod 4`; the host uses LEFT except at
target zero, where it uses RIGHT. All alternatives share the registered future
RNG stream. Seven measurements continue to refer to the original target.

The reward remains:

```text
clamp(0.15*delta_local + 0.15*delta_global + 0.20*delta_coherence
    + 0.20*delta_novelty - 0.15*delta_repetition - 0.10*delta_collapse
    + 0.05*delta_continuity - 0.05*cost, -1, 1)
cost = cumulative_generated_characters / (64 * (horizon + 1))
```

All valid heads learn their KEEP-relative targets with mean Huber loss,
delta 1, from the same pre-update weights. Replay changes policy bytes only.
Config, temporal memory, RNG, online counters and the v1 checkpoint schema
remain unchanged. Every raw axis, generated token, termination flag and
initial/future generation cost is retained. Metric definitions and generation
recipe are explicit in the protocol.

## New-state result

H4 over 48 states, seeds 101/211. Regret is the best measured valid action's
reward minus the selected action's reward. Best-action counts use the
registered `1e-7` tie tolerance.

| Policy | KEEP / LEFT / RIGHT | Mean advantage over KEEP | Mean regret | Best action |
|---|---:|---:|---:|---:|
| initial | 48 / 0 / 0 | 0 | 0.006626973 | 32/48 |
| h4 | 31 / 17 / 0 | −0.002473649 | 0.009100622 | 27/48 |
| h0 | 48 / 0 / 0 | 0 | 0.006626973 | 32/48 |
| shuffled_h4 | 44 / 3 / 1 | −0.001379491 | 0.008006464 | 28/48 |
| fixed KEEP | 48 / 0 / 0 | 0 | 0.006626973 | 32/48 |
| fixed LEFT | 12 / 36 / 0 | −0.005310106 | 0.011937079 | 14/48 |
| fixed RIGHT | 12 / 0 / 36 | −0.005267728 | 0.011894702 | 22/48 |

The H4 policy changes 17 choices relative to initial/H0/KEEP and 21 relative
to shuffled H4. For seed 101 it selects 15 KEEP / 9 LEFT, mean advantage
−0.002754205; for seed 211, 16 KEEP / 8 LEFT, −0.002193093.

| Target index | KEEP / LEFT / RIGHT | H4 advantage over KEEP | Best action |
|---|---:|---:|---:|
| 0 | 12 / 0 / 0 | 0 | 8/12 |
| 1 | 1 / 11 / 0 | −0.003337985 | 3/12 |
| 2 | 6 / 6 / 0 | −0.006556611 | 4/12 |
| 3 | 12 / 0 / 0 | 0 | 12/12 |

At target 3, this continuation overwrites the initial intervention before
reading it. The training inspection registered that geometry: all 12 training
states have identical H4 after-chains and seven after-axes across actions;
generation cost differs. The per-target breakdown keeps that structure visible.

The separate **training** readout gives H4 advantage +0.000479821, regret
0.006019361, 32/48 best actions, and 31 KEEP / 15 LEFT / 2 RIGHT. Initial KEEP
has regret 0.006499182 and 34/48 best actions. Training acquisition and new-state
utility are therefore measured separately.

## Raw-axis result and the next question

For the H4 policy's selected actions, these are evaluation means relative to
the same-state KEEP branch. Higher repetition/collapse and cost are penalized
by the registered objective.

| Component | H0 | H1 | H4 |
|---|---:|---:|---:|
| Local connectedness | −0.010814587 | −0.005281866 | +0.010329938 |
| Global connectedness | +0.000058940 | −0.000063291 | +0.000099190 |
| Coherence | −0.007209726 | −0.003652883 | +0.002638487 |
| Novelty | +0.006372822 | +0.012729351 | −0.009278271 |
| Repetition | +0.000234261 | +0.000234261 | −0.000093869 |
| Collapse | 0 | 0 | 0 |
| Continuity | −0.009300657 | +0.002801181 | +0.000269773 |
| Normalized generation cost | +0.337565104 | +0.144856771 | +0.054752602 |
| Registered reward advantage | −0.019159155 | −0.006124399 | −0.002473649 |

Later coherence rises while novelty falls and regeneration costs increase.
Coherence alone would give a different verdict from the frozen seven-axis
objective. Useful transfer fails against KEEP on both evaluation seeds;
correct source association also loses to the registered shuffle control.
Acquired weights nevertheless alter choices on identical captured states.

The next question is which sentence/history information distinguishes useful
interventions from costly ones. This experiment localizes the missed choices
to the two interior target positions. Representation, training coverage and
fitting budget are candidates for separate future controls; they were held
fixed throughout this run. The existing body, objective and negative result
remain registered here.

## Reproduction and receipts

The authoritative runner builds an immutable source snapshot, runs the native
and parser gates, fits/seals the lives, generates paired diagnostics-OFF/ON
trajectories, reads both datasets, then repeats the complete process.

```sh
make check_spa_future BLAS_FLAGS= BLAS_LIBS=
python3 tests/test_spa_future_mutations.py --json /path/to/audit.json
python3 experiments/spa_agent/future/run.py \
  --output /path/to/new-run --inputs /path/to/body-inputs
```

`--inputs` supplies the parent recipe's `simple.weights` and `dracula.txt`.
The runner verifies their registered hashes and prepares token/vocabulary
buffers. The output directory must be outside the repository and new or empty.
Compilation uses C11, `-O2 -DUSE_SIMD -march=native -pthread`, without BLAS.
The recorded machine is Linux x86_64, AMD EPYC 9V74, a nine-CPU process view;
compiler, memory state, source hashes and exact commands are in the receipts.

- [`receipts.json`](receipts.json): both execution manifests, seals, every
  per-epoch loss, all per-seed/per-target summaries, repeat and completeness gates.
- [`raw_manifest.json`](raw_manifest.json): three lossless transport parts of
  the one canonical `raw_traces.jsonl.gz` trace set;
  24,000,308 bytes, SHA-256
  `359d2983c0c58d35ca025d7bfbcb26e55b0556beff3a4eb4ac95859ea3c95cf5`.
  Each wrapper records `stream`, `seed`, and the original JSONL `raw` line.
  It retains all 73,728 fits, 672 training and 672 evaluation readouts, full
  ordinary traces, and all 48 snapshots / 120 forks / 360 scenario measurements.
- [`audit.json`](audit.json): all 267 policy derivatives, 6,277 native checks,
  six compiled red-hand mutations, exact four-life v1 compatibility, source
  association controls and the earlier concurrent-header stability failure.
- [`result_audit.json`](result_audit.json): independent final artifact, source
  association, selected-outcome and summary verification.
- [`verification.json`](verification.json): existing SPA/Chuck gates, original
  1,024-step / 114,688-byte SPA parity, all 40 CPU recipe commands, shared/static
  integration and Python gates.
- [`python.json`](python.json): 252 loaded-library ABI coordinates, 59 exact
  C/Python records, canonical checkpoint continuation and compiled ABI mutation.

The initial and acquired lives are saved during reproduction; their hashes
are retained here. Checkpoints, model weights and executables are not committed.
The original parent 0/48 acquired-choice result and its measured scenario
artifacts remain unchanged.

The publishing transport caps requests at 16 MiB. Parts are byte ranges of the
original archive; concatenation reproduces it without recompression. Authenticate
all part hashes and reconstruct into a new destination with:

```sh
python3 experiments/spa_agent/future/traces.py \
  --output /path/to/raw_traces.jsonl.gz
```
