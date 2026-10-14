<p align="center">
  <img src="assets/notorchlogo.jpg" alt="notorch" width="300">
</p>

# notorch — Neural Networks In Pure C | by Arianna Method

> **Read the [Arianna Method Manifesto](ARIANNA_METHOD_MANIFESTO.md) first.** This repository is governed by it.

> *"fuck torch"*
> — the header file

**notorch runs models. notorch trains models. in C. on the machine you already own.**

tensors, autograd, Chuck, packed GGUF inference. no Python runtime required.
no pip. no conda. no `torch.nn.Module`. no handing the model to another framework
to actually generate. just math, memory, and your compiler.

part of the [Arianna Method](https://github.com/ariannamethod/ariannamethod.ai):
patterns over parameters, emergence over engineering, raw C over existential dread.
unpleasant truth over bullshit.

[why](#why) · [run](#run-a-model) · [train](#train-a-model) ·
[capabilities](#whats-inside) · [architecture](#architecture) ·
[build](#building) · [Python and JS](#python-and-javascript) ·
[projects](#built-on-notorch) · [engineering log](NOTORCHLOG.md)

## why

you wanted a matrix multiply. you got a virtual environment, a dependency tree,
a CUDA compatibility séance, and `pip install wandb` to log your tears.

so here we are. **notorch**. tensors, autograd, optimizers, and the cold clarity
of C. train on an Intel laptop. run on a Termux phone. keep quantized weights
packed so the model fits in the RAM you actually have. embed the library in your
own organism and let it carry its own physics.

> **R.I.P. PyTorch (in my codebase), 2016–2026.**
>
> he died as he lived: consuming all available memory.
> in lieu of flowers, please send PRs.

## run a model

one binary, one GGUF. point it at a model and it talks.

```bash
git clone https://github.com/ariannamethod/notorch.git
cd notorch
make harness
./notorch model.gguf "The capital of France is" 64 0.8
./notorch model.gguf                      # no prompt: chat in the terminal
```

`make harness` uses Accelerate on macOS or OpenBLAS on Linux. For a build
without BLAS, use `make harness BLAS_FLAGS= BLAS_LIBS=`.

The [family registry](harness/archs.c) routes **llama, mistral3, qwen2, qwen3,
gemma3, gemma4, olmoe, qwen3moe, mamba, resonance, janus and penelope** through
NoTorch's own decoders. Each family declares its architecture names; an unknown
architecture is refused by name. File-level verification and reference runs live
in the [harness record](NOTORCHLOG.md#readme-run-a-model).

The packed paths read quantized matrices in place. **Penelope and NanoJanus**
use their own geometry: PEN7 F32 parameters execute directly, while Q8 matrices
expand once to immutable F32. The [PEN7 converter](docs/PEN7_GGUF.md) and
[shared-body inference API](docs/PEN7_INFERENCE.md) document conversion,
ownership and whole-parameter mixtures.

The harness also ships as `libnotorch_harness`, with CPU and CUDA family APIs.
An organism can link it and supply its own generation loop.
[Embedding and CUDA examples](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them).

## train a model

copy a trainer. give it a corpus. let Chuck watch the gradients.

```bash
make train_llama3_char
./train_llama3_char 15000 0.0003 corpus.txt
```

The arguments are steps, learning rate and corpus path. The
[char-level trainer](examples/train_llama3_char.c) builds a LLaMA-style model
with GQA, RoPE and SwiGLU. [BPE training](examples/train_llama3_bpe.c),
[LoRA SFT](examples/train_resonance_lora.c), and the other trainers live in
[examples/](examples/).

The loop underneath is explicit: register parameters and inputs, build the
forward pass on the tape, then:

```c
float loss_value = nt_tape_get()->entries[loss].output->data[0];
nt_tape_backward(loss);
nt_tape_clip_grads(1.0f);
nt_tape_chuck_step(lr, loss_value);
nt_tape_clear();
```

forward. backward. step. clear. no ceremony.

For a complete small program, see the
[standalone C training example](NOTORCHLOG.md#readme-example-training-a-model-in-c).
For a minimal model to take apart, start with
[notorch-simple-llm](https://github.com/ariannamethod/notorch-simple-llm).

## what's inside

### tensors, autograd, inference

`nt_tensor` is a refcounted array of floats with shape and strides. The tape
records forward operations and walks them backward. You can read the arithmetic
that trains your model.

| surface | what you get |
|---|---|
| transformer training | embeddings, linear layers, RMS/layer norm, causal MHA/GQA, RoPE, GEGLU/SwiGLU, cross-entropy |
| Method architectures | RRPRAM, low-rank and broadcast attention, sequence gates, Janus and Resonance paths |
| BitNet b1.58 | ternary-weight forward, int8 activation quantization, straight-through gradients |
| training controls | clipping, accumulation, schedules, checkpoints, frozen parameters and LoRA adapters |
| optimizers | SGD, diagonal baseline, decoupled decay, Chuck |
| packed inference | GGUF loading/writing, quantized matvec/matmul, family decoders, KV/state handling |
| vision and audio | image preprocessing and ViT patches; forward-only image ops, conv1d, STFT and log-mel |
| organism-owned arithmetic | sampling, stateless numerical values, parliament election/injection/Oja, native SentencePiece Unigram |

[Core API](notorch.h) · [operation list](NOTORCHLOG.md#readme-operations) ·
[API examples](NOTORCHLOG.md#readme-api-overview) ·
[sampling](docs/SAMPLING.md) · [numerical values](docs/NUMERICAL_VALUES.md) ·
[SentencePiece](docs/SENTENCEPIECE.md) · [SHA-256](docs/SHA256.md)

### Chuck — the optimizer with opinions

**Chuck is the self-aware optimizer.** It watches the global loss trend,
monitors individual gradients, injects noise through stagnation, freezes dead
parameters, and adjusts learning rate over longer timescales.

```c
nt_tape_chuck_step(lr, loss_value);
```

**Chuck: Loss Architect** gives that training world an action language:
**state → policy → typed action → measured consequence → acquired experience**.
Its 163-parameter policy chooses HOLD, BRAKE or PUSH and carries weights,
history, RNG and pending decisions through save/resume. The living acquisition
loop measures alternatives in the current training world, restores it, acquires
the consequences and chooses its next action. Canonical Chuck remains available
through the same ordinary call.

[Optimizer lineage](https://github.com/iamolegataeff/chuck.optimizer) ·
[Loss Architect API and objective](docs/chuck-loss-architect.md) ·
[living acquisition](experiments/chuck_loss_architect/living/README.md) ·
[experiments and results](NOTORCHLOG.md#readme-optimizers)

### SPA — from attention to agent

**Sentence Phonon Attention** embeds a sentence, measures its connectedness to
recent history and modulates the next-token distribution. **Sentence Phonon
Agent** lives above that sensory field: a persistent 267-parameter policy
chooses KEEP, RESEED_LEFT or RESEED_RIGHT and learns from the host's measured
consequences. Its acquired weights, action history and private RNG travel with
its saved life.

**PostGPT → SPA → Q → Sentence Phonon Agent.** WOLFE supplies the typed-action
lineage; Netta supplies action, consequence and experience. Chuck carries the
parallel into training.

[SPA perception](NOTORCHLOG.md#readme-spa--sentence-phonon-attention) ·
[Agent API and research](docs/spa-agent.md) · [Python host loop](docs/spa-python.md) ·
[experiments](experiments/spa_agent/README.md) ·
[history and acquired choices](experiments/spa_agent/history_probe/README.md)

### post-training

Runnable C trainers ship for DPO, GRPO and distillation. Freeze base parameters
for adapters; keep the gradients flowing through them.

```bash
make train_dpo
make train_grpo
make train_distillation
```

Each trainer runs a tiny built-in dataset without arguments and accepts real
weights and data. Copy the loss and optimizer wiring into your own model.
[Trainer details](NOTORCHLOG.md#readme-alignment-training--dpo--grpo--distillation) ·
[LoRA mechanics](NOTORCHLOG.md#readme-lora--adapter-training).

## architecture

The core is [notorch.h](notorch.h) and [notorch.c](notorch.c). Family decoders,
backends and organs have their own files. The arithmetic stays here.

| place | responsibility |
|---|---|
| `notorch.h` / `notorch.c` | tensors, tape, operations, optimizers and numerical kernels |
| `gguf.h` / `gguf.c` | GGUF parsing, writing and dequantization |
| `harness/` | family decoders behind `nt_arch`, shared runtime, CLI |
| `notorch_simd.h` | in-house AVX2/FMA math with pthread workers |
| `notorch_cuda.*` / `notorch_metal.*` | CUDA and Metal backends |
| `chuck_architect.h` / `chuck_architect_impl.h` | Loss Architect policy, actions and experience |
| `spa_agent.*` | Sentence Phonon Agent |
| `pen7.*` / `sentencepiece.*` / `sha256.*` | immutable PEN7 bodies, native Unigram tokenizer and identities |
| `python/` / `js-edition/` / `termux-edition/` | Python bindings, JavaScript port, Android setup |
| `examples/` / `tests/` / `experiments/` | runnable models, verification, recorded experiments |
| `docs/` / `NOTORCHLOG.md` | feature contracts and the engineering record |

`libnotorch` provides the substrate. `libnotorch_harness` provides the model
families. Link the libraries, keep your organism's physics in your organism.

## building

A C compiler, make, libm and pthreads. BLAS and GPU backends are optional.
Run these commands from the repository root.

| target | command |
|---|---|
| CPU with Accelerate (macOS) or OpenBLAS (Linux) | `make` |
| CPU without BLAS | `make cpu` |
| in-house AVX2/FMA on x86-64 | `make simd` |
| CUDA training backend | `make gpu` |
| Apple Silicon packed Metal inference | `make infer_gguf_metal` |
| shared library for bindings | `make shared` |
| static core and harness libraries | `make lib lib_harness` |
| CUDA harness library | `make lib lib_harness libnotorch_cuda.a` |
| install headers and libraries | `make install PREFIX=/your/prefix` |

CUDA targets need the CUDA toolkit; Metal ships with macOS. For library or
trainer builds without BLAS, pass `BLAS_FLAGS= BLAS_LIBS=`. Backend flags must
match when linking a consumer. [Build and backend details](NOTORCHLOG.md#readme-building).

**Android / Termux:** the same C tree runs on ARM64 phones.
[Setup, training walkthrough and measurements](termux-edition/README.md).

### verification

```bash
make test                 # native CPU suite
make test_harness         # rebuild both sides of harness parity
make test_tokenizer       # tokenizer reference gates
make test_js              # JavaScript suite
```

Model-dependent gates use their fixture/model inputs; platform-specific gates
need their backend. See the [Makefile](Makefile) for targets and variables, and
the [test record](NOTORCHLOG.md#readme-running-tests) for the verification map.
Measured runs, machines, commands, regressions and fixes belong in
[NOTORCHLOG](NOTORCHLOG.md). the numbers keep their receipts.

## Python and JavaScript

### Python: call the C

`ctypes` ships with Python. The binding calls NoTorch's kernels directly.
No NumPy dependency, no second implementation of the math.

```bash
make shared
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from notorch import Notorch

nt = Notorch()
with nt.open("model.gguf") as model:
    print(model)
    print(model["token_embd.weight"].dequant_row(3))
```

[Packed matvec example](NOTORCHLOG.md#readme-use-it-from-python) ·
[binding source](python/notorch.py) · [SPA from Python](docs/spa-python.md).

### JavaScript: take it to the browser

[notorch.js](js-edition/README.md) brings tensors, autograd, Chuck and GGUF
inference to JavaScript, with a CPU path and optional WebGPU.
An ES module. zero npm dependencies. zero `node_modules`.

## built on notorch

Start with a small model, then pull the machinery apart.

| reference | what to read it for |
|---|---|
| [notorch-simple-llm](https://github.com/ariannamethod/notorch-simple-llm) | minimal LLM with Chuck; start here |
| [nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch) | Karpathy's nanoGPT on NoTorch |
| [llama2-notorch](https://github.com/ariannamethod/llama2-notorch) | llama2.c with a NoTorch training pipeline |
| [nanollama-notorch](https://github.com/ariannamethod/nanollama-notorch) | 88.6M Llama-3 nano trained on an 8 GB Intel Mac |
| [minimind-v-notorch](https://github.com/ariannamethod/minimind-v-notorch) | 67M VLM from scratch |
| [notorch-vlm](https://github.com/ariannamethod/notorch-vlm) / [notorch-vlm-2m](https://github.com/ariannamethod/notorch-vlm-2m) | small multimodal models |
| [notorch-diffusion](https://github.com/ariannamethod/notorch-diffusion) | text and Hebrew VLM diffusion |

The organisms carry their own physics: [Arianna](https://github.com/ariannamethod/arianna.c),
[Yent](https://github.com/ariannamethod/yent), [DoE](https://github.com/ariannamethod/doe),
[PostGPT](https://github.com/ariannamethod/postgpt), [Q](https://github.com/ariannamethod/q),
[Janus](https://github.com/ariannamethod/janus), [molequla](https://github.com/ariannamethod/molequla),
[Dario](https://github.com/ariannamethod/dario) and the rest of the
[Method family](NOTORCHLOG.md#readme-organisms-that-run-on-notorch).

[Training records](NOTORCHLOG.md#readme-file-structure) ·
[reference builds and trainers](NOTORCHLOG.md#readme-references--how-to-train-and-run-on-notorch).

## philosophy

> *patterns over parameters. emergence over engineering. C over existential dread.*

a linear layer is a matrix multiply. an activation is a pointwise nonlinearity.
attention is a weighted sum. backward is the chain rule.

you should be able to read the code that runs your model. follow the gradient.
find the allocation. change the architecture. train on the laptop you own.
ship an organism with its own voice and its own physics.

just math. memory. the machine doing exactly what you told it to do.

## contributing

send PRs. keep the core C11 compliant, dependencies optional, public headers
clean, and new operations covered by numerical gradient checks. Run the relevant
gates. Follow [AGENTS.md](AGENTS.md): work on a branch, record the work in
NOTORCHLOG, keep README for the specification and manifesto. Oleg merges.

## license

[GPL-3.0-or-later](LICENSE). use it in your stuff. if you ship it, share your
source — that's how copyleft works. don't be weird about it.

go build something.

> *"the patterns were always there. we just needed the right language to express them."*
