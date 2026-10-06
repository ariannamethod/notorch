# Chuck: Loss Architect — training receipts

`run.py` trains the SimpleLLM and HeVLM reference bodies through the current
upstream `notorch.c`. It retrieves pinned corpora, verifies SHA-256 identities,
builds the C runner, and executes four independent optimizer arms for each seed:
Adam, canonical Chuck, Architect legacy, and learned Architect.

```bash
python3 experiments/chuck_loss_architect/run.py \
  --output /tmp/chuck-loss-architect \
  --steps 512 --seeds 42,73 --threads 2 \
  --config examples/chuck-loss-architect.json
```

`--references /path/to/reference-checkouts` reads corpora from local checkouts
at the pinned commits. Without that option, the script fetches immutable GitHub
raw-file URLs. Output must be outside the repository. Final measurements require
a committed source tree; `--allow-dirty` labels development smoke runs explicitly.

## Protocol

The [architecture and objective](../../docs/chuck-loss-architect.md) describe the
163-parameter policy. This experiment uses context 64, F32 parameters/gradients,
batch size one, gradient clipping at 1, and a constant body learning rate of
`0.0003`. Both reference bodies retain their tensor shapes, initialization and
forward computations. SimpleLLM uses the repository's 94-character Unicode
Dracula vocabulary; HeVLM uses all 256 UTF-8 byte IDs.

Each corpus splits contiguously: first 90% for training, final 10% for held-out
evaluation. Every arm receives identical initial weights and a SplitMix64 window
stream independent of model initialization and policy exploration. The run seed
initializes the body, window stream, and Architect exploration; it overrides the
JSON `seed`. The exact effective Architect configuration is the first JSONL row.

Every update is followed by a second forward pass on **the same window**. Only
this measured loss difference supplies online policy credit. The next JSONL
step's `loss_before` measures the next sampled training window. Eight fixed
held-out windows are evaluated initially and after the final update. Their
losses remain separate from the online credit target.

The script checks matched initial weights for all arms and exact canonical
Chuck/Architect-legacy equality for every loss, gradient norm, global Chuck
state, frozen count, and final body weight. The first comparison answers whether
adding the architecture preserves the canonical organism. Learned-arm action
and consequence receipts describe how experience changes its policy state.

A read-only counterfactual asks: **do acquired network weights change the chosen
action at encountered states, holding history and RNG fixed?** Before each
learned step, the runner copies the current life and replaces only its network
weights with their initialization. After the actual decision, a read-only
selection on the identical observation records `initial_weights_action`.
`weight_dependent_choices` counts differences. This is a decision diagnostic;
the copied policy never acts in the training environment.

## Reviewed bodies

| Repository and inspected commit | Body | Choice for this experiment |
| --- | --- | --- |
| `notorch-simple-llm@80b3bd611ed8c937efdc481dc07895c8e13e345d` | SimpleLLM: E128, two layers, FF384, RoPE, 94 Unicode characters, 450,688 parameters on current Dracula | First body; native upstream operations and real repository corpus. |
| `notorch-diffusion@1bc645071b3a0a3525fe89b8f45253f5faa28a1d` | HeVLM: E128, four layers, FF512, learned positions, UTF-8 bytes, 1,123,456 parameters | Second body; C recipe links directly, Hebrew corpus changes the training environment and positional representation. |
| `notorch-vlm@0d53de7a2ccac87c1e2bee1ce4ac1e3fd640a019` | 1.5M VLM: E160, four self/cross-attention blocks, synthetic image features and captions | Inspected for the next distinct modality; first pair uses existing text corpora. |
| `q@f5d00a36ecfcdb5e655e1576770f03d06d900e04` | Approximately 2M Q substrate: BPE, RRPRAM/Janus, gated attention, 439,326-byte `q.txt` | Method-native extension target; first pair needs no tokenizer/gated-attention adaptation. |
| `nanoGPT-notorch@9fe2eb98fccd637e904a0e638d08fa7da26e9650` | 10.2M: E320, eight layers, FF896, RoPE, Dracula | Next scale after the small-body mechanism and receipts. |

The selected architectures come from `train_dracula.py` and
`ariannamethod/train_hevlm.c`. Each numerical operation is supplied by this
upstream tree. Vendored copies in the reference repositories remain untouched.

## Artifacts

- `manifest.json`: source commit/status, numerical-source hashes, compiler and
  binary identity, machine/CPU quota/memory, configuration and corpus identities.
- `BODY-ARM-sSEED.jsonl`: every training window, observation/features, selected
  action, actual same-window consequence, pre/pending/post policy identities,
  pre/post global Chuck state, and initial/final held-out loss.
- `*.initial.bin`, `*.final.bin`: body parameters in `nt_save` format.
- `*.policy.initial.bin`, `*.policy.final.bin`: versioned Architect lives.
- `*.moments.final.bin`: Adam first/second moments, interleaved in parameter
  order. `*.optimizer.final.json` holds Adam counters, complete Chuck global and
  per-parameter state, Chuck's noise RNG and the window RNG.
- `*.resources.json`: process peak RSS and elapsed/user/system CPU time,
  independently measured by `/usr/bin/time` when available. The C summary also
  records `getrusage` peak RSS and CPU time through the final held-out evaluation.
- `results.json`: compact comparisons, causal gates and artifact hashes.

Weights, corpus copies and complete receipts stay outside Git. Results committed
here name the exact source and artifact identities used for the measurements.

## Measured run — 2026-10-06 UTC

Source `d5103e89797b47920f76c730cf5a24613791d04d`, clean tree, GCC 13.3,
`-O2 -std=gnu11 -DUSE_SIMD -march=native -pthread`, Linux x86_64,
AMD EPYC 9V74, eight-core CPU quota, `NT_SIMD_THREADS=2`, F32. The exact command
above trained **16 runs × 512 updates**, 32,768 training tokens per run.
Source hashes were checked again after the final run and remained identical.

Final held-out cross entropy on the eight fixed windows:

| Body | Seed | Initial | Adam | Canonical Chuck | Architect legacy | Learned Architect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleLLM, 450,688 parameters | 42 | 4.806492 | 2.568933 | 2.465579 | 2.465579 | 2.539874 |
| SimpleLLM, 450,688 parameters | 73 | 5.208465 | 2.586830 | 2.531280 | 2.531280 | 2.681840 |
| HeVLM, 1,123,456 parameters | 42 | 5.744682 | 1.571433 | 1.576577 | 1.576577 | 1.571453 |
| HeVLM, 1,123,456 parameters | 73 | 6.111211 | 1.529242 | 1.587225 | 1.587225 | 1.588265 |

All four canonical/legacy pairs matched every recorded training step and final
weight byte. All four initialization groups matched across optimizer arms.

| Learned body | Seed | Hold | Brake | Push | Weight-dependent choices | Feedback updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleLLM | 42 | 460 | 25 | 27 | 0 / 512 | 512 |
| SimpleLLM | 73 | 21 | 464 | 27 | 426 / 512 | 512 |
| HeVLM | 42 | 460 | 25 | 27 | 0 / 512 | 512 |
| HeVLM | 73 | 21 | 464 | 27 | 426 / 512 | 512 |

In seed 73, the first exploratory `brake` produces an actual same-window loss
improvement. At step two, the updated policy selects `brake` without exploration;
the initial-weight policy on exactly the same observation, history and RNG
selects `hold`. For SimpleLLM the step-two outcome scores are
`[0, 0.00577885238, 0]`. Both bodies preserve this acquired preference through
426 non-exploratory choices. The initial and final policy lives and the first
credited transition are retained in the receipts.

The seed-42 readout records zero weight-dependent choices: its acquired outcome
head keeps `hold` ahead throughout this run. The seed-73 preference repeatedly
brakes to the configured lower dampening bound. Its measured SimpleLLM held-out
loss is 2.681840 against canonical Chuck's 2.531280. These two observations are
the next controlled-policy question: action comparisons from common starting
body/optimizer states can teach the outcome heads the relative consequences of
their alternatives. The current one-step credit target and all four learned
results remain recorded unchanged.

Across the sixteen processes, the recorded training/evaluation wall times sum
to 211.500319 seconds. Process peak RSS is 37,976–39,356 KiB; the per-run values,
CPU time, first/final 16-window means, exact losses, source identities and all
artifact hashes are in [receipts.json](receipts.json). Raw receipts and saved
bodies/optimizer/policy lives are in the archive identified there.
