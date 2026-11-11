<p align="center">
  <img src="assets/notorchlogo.jpg" alt="notorch" width="300">
</p>

# notorch — Neural Networks In Pure C | by Arianna Method

> **Read the [Arianna Method Manifesto](ARIANNA_METHOD_MANIFESTO.md) first.** This repository is governed by it.

> *"fuck torch"*
> — the header file

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ariannamethod/notorch)

---

## table of contents

- [what](#what)
- [why](#why)
- [the funeral](#the-funeral)
- [run a model](#run-a-model)
  - [living experts and weight swaps](#living-experts-in-the-harness)
  - [dreaming between turns](#dreaming-between-turns)
- [train a model](#train-a-model)
- [architecture](#architecture)
- [what's inside](#whats-inside)
- [operations](#operations)
- [optimizers](#optimizers)
- [bit-level precision — BitNet b1.58](#bit-level-precision--bitnet-b158)
- [SwiGLU FFN](#swiglu-ffn)
- [SPA — Sentence Phonon Attention](#spa--sentence-phonon-attention)
- [SPA — Sentence Phonon Agent](#spa--sentence-phonon-agent)
- [LoRA / adapter training](#lora--adapter-training)
- [BLAS inference API](#blas-inference-api)
- [audio, sampling and organism-owned values](#audio-sampling-and-organism-owned-values)
- [alignment training — DPO / GRPO / distillation](#alignment-training--dpo--grpo--distillation)
- [autograd](#autograd)
- [example: training a model in C](#example-training-a-model-in-c)
- [building](#building)
- [running tests](#running-tests)
- [platform support](#platform-support)
- [use it from Python](#use-it-from-python)
- [js edition — notorch.js](#js-edition--notorchjs)
- [references — how to train and run on notorch](#references--how-to-train-and-run-on-notorch)
- [organisms that run on notorch](#organisms-that-run-on-notorch)
- [philosophy](#philosophy)
- [contributing](#contributing)
- [license](#license)
- [NOTORCHLOG — the engineering record](NOTORCHLOG.md)

---

## what

**NOTORCH** is a complete neural network framework written in pure C — it **runs** models and it **trains** them.
no Python. no pip. no conda. no CUDA toolkit that takes 8 GB and your will to live. no `torch.nn.Module`. no `.backward()` that hides 400,000 lines of C++ behind a friendly API and a smile. no handing the model to llama.cpp to actually generate. no `RuntimeError: CUDA out of memory` at 3 AM when your paper deadline is in 6 hours.

just NOTORCH. just C.

just floats. just `cc -O2 -std=gnu11 -pthread notorch.c your_model.c -lm -o train`. done. you now have a neural network framework. the entire thing compiles in a couple seconds. try that with PyTorch. go ahead — you'd be waiting 47 minutes while cmake does whatever cmake does.

it's part of [the Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — patterns over parameters, emergence over engineering, raw C over existential dread. unpleasant truth over bullshit.

---

## why

a story:
once upon a time there was a framework called PyTorch. it had autograd. it had CUDA support. it had a build system that required a PhD in software engineering and a pact with ancient spirits. and every time you wanted to train a 4-layer MLP on a dataset smaller than your browser cache, you had to:

1. create a virtual environment (2 minutes)
2. install torch (5 minutes, 2.7 GB, your SSD weeps)
3. install torchvision just in case (800 MB more, your SSD files for divorce)
4. write 47 lines of boilerplate (`class MyModel(nn.Module)`, `def forward(self, x)`, `optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)`, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, `if torch.cuda.is_available():`, `model.to(device)`, `x = x.to(device)`, sweet mother of god make it stop)
5. realize you forgot `model.train()` vs `model.eval()` and your dropout is wrong
6. debug for 3 hours
7. realize the bug was actually in the data loader
8. cry
9. `pip install wandb` to log your tears
10. realize torch updated and broke everything

and for WHAT? a matmul and a softmax. that's all neural networks are. matmuls and softmaxes and an unhealthy relationship with gradient descent.

so here we are. **notorch**. everything you need. nothing you don't. no Python runtime. no GIL. no garbage collector pausing your training at the worst possible moment. no `torch.no_grad()` context manager that you forget and then wonder why you're out of memory. just tensors, autograd, optimizers, and the cold clarity of C.

**the core lives in two files.** `notorch.h` and `notorch.c`. tensors, autograd, the arithmetic in plain sight. you can read the whole thing in an afternoon. try reading PyTorch's source in an afternoon. actually don't. you'll end up in a hospital.

---

## the funeral

```
        ╔═══════════════════════════════════════════════════════╗
        ║                                                       ║
        ║   R.I.P. PyTorch (in my codebase)                     ║
        ║   2016 - 2026                                         ║
        ║                                                       ║
        ║   "He died as he lived:                               ║
        ║    consuming all available memory                     ║
        ║    and segfaulting at the worst moment"               ║
        ║                                                       ║
        ║   Survived by: pip, conda, 2.7 GB of dead weight,     ║
        ║   a thousand Stack Overflow questions about CUDA      ║
        ║   driver versions, and a broken conda environment     ║
        ║   that nobody dares to delete.                        ║
        ║                                                       ║
        ║   In lieu of flowers, please send PRs.                ║
        ║                                                       ║
        ╚═══════════════════════════════════════════════════════╝
```

notorch is for people who:

- want to understand what's actually happening (the source is right there)
- want to train models on machines that aren't cloud instances
- want compile times measured in seconds, not minutes
- want to embed neural network inference in C/C++ applications without shipping half of Python
- refuse to accept 2.7 GB as the price of a matrix multiply

---

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
gemma3, gemma4, olmoe, qwen3moe, mamba, resonance, janus, penelope and lfm3**
through NoTorch's own decoders. Each family declares its architecture names; an
unknown architecture is refused by name. File-level verification and reference
runs live in the [harness record](NOTORCHLOG.md#readme-run-a-model).

**Janus v4** exports can select `janus.forward_graph="pytorch-v4-prefix"`
to execute the published Python graph in native C: historical Echo attention,
simultaneous token smear, complete-block backout and the global low-rank
RRPRAM reduction. Independent per-cache token histories support exact prefix
recomputation across chunked prefill and decode. Existing exports retain the
incremental profile. [Metadata, cache contract and numerical proof](docs/JANUS_V4_PREFIX.md).

**LFM3** is the registry's first hybrid: gated short convolutions and attention
share one body, per-head sigmoid gates ride the attention output, and the
sliding-versus-full split is read from the file's `sliding_window_pattern`
array — the only place it lives. On LiquidAI's LFM3-1.7B, LFM3-VL-2B and
LFM3-Omni-2B (text path), 13 of 16 bare-greedy 32-token continuations match
llama.cpp, and six-thread decode runs at 0.93–0.97 of its speed on a 6-core
x86: [measurements and limits](NOTORCHLOG.md#readme-run-a-model).

The packed paths read quantized matrices in place. **Penelope and NanoJanus**
use their own geometry: PEN7 F32 parameters execute directly, while Q8 matrices
expand once to immutable F32. The [PEN7 converter](docs/PEN7_GGUF.md) and
[shared-body inference API](docs/PEN7_INFERENCE.md) document conversion,
ownership and whole-parameter mixtures.

The harness also ships as `libnotorch_harness`, with CPU and CUDA family APIs.
An organism can link it and supply its own generation loop.
[Embedding and CUDA examples](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them).

The CPU Harness reads the model’s embedded tokenizer or, for split GGUFs,
the adjacent `tokenizer.gguf`. Give each split model its own directory.
`--tokenizer FILE` selects a specific vocabulary. The CLI (including `-T`)
and native consumers use the same resolver in `harness/tokenizer.h`. Text
generation samples only IDs covered by the loaded vocabulary.

**EmbeddingGemma 2 text** has a separate full-sequence encoder API: native F32
bidirectional attention, projection-only per-layer inputs, masked mean pooling,
and 768-dimensional normalized output. Read/write activation hooks expose every
block. The [encoder guide](docs/EMBEDDINGGEMMA2.md) describes unchanged BF16/F32
checkpoint conversion, explicit token-ID inputs, and the frozen layer-by-layer
reference parity gate. Six neutral fixtures and the complete frozen
80-document/24-query corpus pass numerical parity, with exact native replay;
[measured results](docs/validation/embeddinggemma2-validation-2026-11-05.json)
record the source, checkpoint and tokenizer boundary.

**Qwen3-Embedding** is the second encoder on the same API: a causal Qwen3
decoder read to its final norm, last-token pooling over the appended
`<|endoftext|>`, 1024-dimensional normalized output. Its byte-level tokenizer
runs in C through the exact Qwen3 split over NFC. The
[encoder guide](docs/QWEN3EMBEDDING.md) covers conversion, text mode and gates.
On the pinned 0.6B checkpoint the model card's four published scores agree within
3e-7, and repeat runs give byte-identical vectors. The official-ID tokenizer gate
and the vector-parity gate are built and wait for their references.

fuck torch — but also, you do not need llama.cpp to *run* what you built.
The CPU, Metal and CUDA paths live here. Packed matrices stay packed through
their kernels; the PEN7 path owns its immutable parameters. The engineering
record keeps the [memory and throughput measurements](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them),
with the machine and the model beside them.

---

### Qwen3.5 text and vision

Native full attention + GatedDeltaNet inference, source-layout BF16/F32 loading,
C safetensors conversion and explicit ASCII chat tokenization. The CPU path
uses F32 matrix weights and owns recurrent/convolution state. Fourteen original
OneJev prompts pass every-layer and full-logit comparison with Transformers;
1,214 frozen-fixture checks exercise chunking and failures.
The original checkpoint's vision encoder is native C too: `qwen35_to_gguf
--vision` retains its BF16/F32 tensors in a separate GGUF. The image API runs
PIL-compatible preprocessing, the vision transformer and merger, and explicit
three-axis positions through the text decoder. Five original OneJev images
match the independent encoder; full-vocabulary logits and greedy continuations
match through image prefill and decoding. Text conversion remains byte-identical.
See [commands, format boundaries and measured parity](docs/qwen35.md).

### LFM3-Omni: eyes, ears and a voice

LiquidAI's LFM3-Omni-2B with every modality its card declares: text, images,
video frames and 16 kHz audio in; text, 24 kHz speech, or both interleaved out.
`tools/lfm3_omni_to_gguf` turns the official safetensors into one GGUF, F16 or
`--q8_0` straight from BF16, and every tensor it shares with LiquidAI's own GGUF
pair is byte-identical. The towers (vision, audio encoder, depth decoder, codec)
live in `harness/lfm3_omni.c` and feed the `lfm3` decoder through
`forward_embd`.

```bash
make notorch_omni lfm3_omni_to_gguf
./lfm3_omni_to_gguf LFM3-Omni-2B/ omni-f16.gguf
./notorch_omni --image page.png omni-f16.gguf "Describe this image in one sentence."
./notorch_omni --system "Perform ASR in English." --audio clip16k.wav omni-f16.gguf ""
./notorch_omni --system "Perform TTS in American English using a female voice." \
    --mode audio --speech-out hello.wav omni-f16.gguf "Hello from notorch"
```

Each tower agrees with the official PyTorch modules in float64 to within 6e-6
relative L2, and on the measured text, image and transcription prompts the F16
greedy answers are the official model's ids, token for token, on x86 and on
Apple silicon. Speed against llama.cpp on both machines,
the gates and the limits (16 kHz audio only, video as same-size frames, no codec
encoder) are in the [engineering record](NOTORCHLOG.md#2026-11-10--lfm3-omni-2b-every-modality-measured).

### DoE. Living experts in the Harness

`--experts` seats a native **Democracy of Experts** on the middle block of the
dense llama-family, Qwen3.5 hybrid, Gemma 3 text, OLMoE, Qwen3-MoE and LFM3 CPU
decoders. Each token
elects a variable number of LoRA experts between the block's mixer and its FFN. Four
rank-4 experts start in six seats.
`--learn-experts` gives the elected experts one Oja update after each completed
turn and enables vitality, mitosis and apoptosis. Plain inference is the default.

```sh
./notorch --experts --learn-experts --spore-out first.gguf model.gguf "A robin builds its nest in" 8 0
./notorch --experts --spore-in first.gguf --expert-trace model.gguf "The robin returns to" 8 0
```

Qwen3.5 seats the same experts after the full-attention or GatedDeltaNet
residual and before the FFN. Supply the GGUF tokenizer for a source-layout
model with a separate vocabulary. Dreams and calibration get their own
recurrent, convolution and full-attention state.

```sh
./notorch --tokenizer tokenizer.gguf --experts --learn-experts --async-consolidation \
  --spore-out awake.gguf onejev.gguf "Check this file" 2 0
./notorch --tokenizer tokenizer.gguf --experts --spore-in awake.gguf onejev.gguf "Check it again" 2 0
```

The default spore is an independent GGUF bound to the SHA-256 of the host file. Give
`--spore-out` a new filename to keep each generation. The KV cache and the
transcript begin fresh on restart. The spore restores LoRA A/B, vote weights,
membership, vitality, frequency, age, low-vitality counts, Oja cursors,
consensus, oscillator input history, RNG and clocks. In chat, `/reset` clears
the KV while the parliament lives on. `--expert-trace` prints per-token gates,
chosen logits, births, deaths and turn boundaries to stderr.

Each expert also has a permanent SHA-256 birth address and a parent address.
The ancestry ledger stays inside the spore after death and slot reuse, and
travels with the population when weights change. New spores use version 5
(exact) or 6 (portable). Versions 1–4 are adopted on import: their existing
seats receive addresses derived from the original archive, with unknown
parents recorded explicitly. A second save is stable. The native API can find
and transplant factors by identity; `doe_probe lineage HOST SPORE` prints the
ledger, including retired ancestors.
The optional audit helper builds with `make doe_probe AML_ROOT=/path/to/aml`;
the Harness and replay API themselves are C and require no AML runtime.

On a return to a previously visited body, `--body-memory` can recall the
components that body learned outside the 32 transferable directions. It matches
living expert addresses in the selected earlier same-host spore, takes its
local A/B/vote components and keeps the incoming parliament's shared coordinates,
membership, genealogy, RNG and clocks. Unmatched identities retain their
incoming vectors unchanged. Both the incoming parliament and the selected
local reference are explicit:

```sh
./notorch --experts --spore-only --spore-in latest-from-b.gguf \
  --body-memory last-on-a.gguf --portable-spore \
  --spore-out returned-to-a.gguf model-a.gguf
```

Keep `latest-from-b.gguf` portable and `last-on-a.gguf` from these exact A
weights. The latter supplies local expert memory by birth address; the latest
global population continues to own its lifecycle. [How recall combines the
coordinates](research/democracy_of_experts.md#returning-to-a-bodys-own-memory).

An incoming body can also explicitly calibrate the scale of its stored votes:

```sh
./notorch --experts --spore-only --spore-in incoming-portable.gguf \
  --expert-vote-rms 1 --spore-out calibrated.gguf receiving-model.gguf
```

This samples the 32 raw anchor residuals, measures the centered spread of the
living experts' dot scores and applies one common positive scale to all six
vote rows. The calibrated rows persist in the existing GGUF format. A/B,
identities, consensus, RNG and clocks keep their values. With `--body-memory`,
the order is import, local recall, then explicit vote calibration. The default
election path and default import perform no vote rescaling.

Add `--async-consolidation` to `--experts --learn-experts` to hand each
completed turn to a C worker. Chat can accept input while it updates the
elected experts. The next forward receives the committed parliament. EOF and
`/exit` join the last worker before publishing the requested spore.

To carry the parliament to a different GGUF, add `--portable-spore` when
saving. The next Harness process needs the new body and that one spore.

```sh
./notorch --experts --learn-experts --async-consolidation --portable-spore --spore-out memory.gguf smollm2.gguf "A robin builds its nest in" 8 0
./notorch --experts --spore-in memory.gguf --expert-trace qwen2.5.gguf "The robin returns to" 8 0
```

Existing generations can also travel. `--spore-only` converts a completed
generation before the first election. It can publish a portable file from an
older exact spore, or bind a portable spore to a target for subsequent sessions.

```sh
./notorch --experts --spore-only --spore-in first.gguf --portable-spore --spore-out memory-v2.gguf smollm2.gguf
./notorch --experts --spore-only --spore-in memory-v2.gguf --spore-out qwen-local.gguf qwen2.5.gguf
```

On the original file, restore is exact. On a different file the Harness
measures 32 shared literals at the parliament's post-attention layer. Matching
architecture, width, depth, ordered RMSNorm fingerprint and aligned residual
axes allow direct reuse of every expert weight across quantizations. Otherwise
the A/B factors and voting vectors cross in the 32 measured directions. The
target's initial values supply the orthogonal complement. Both paths carry the
seats, history, consensus, RNG and turn counters. `--expert-trace` reports the
transfer mode, both host digests, widths and layers. The commands and measured
continuations for SmolLM2 Q4, SmolLM2 Q8 and Qwen2.5 are in the
[engineering log](NOTORCHLOG.md#2026-11-09--a-parliament-crosses-gguf-bodies).

#### One parliament, different weights

In the SmolLM2-360M experiment, one generation learned “Miro carries a key”
and a sibling learned “Miro carries a bell”. Each had one waking turn and 24
elections. Each portable spore occupied 331,616 bytes. The Harness loaded the
same pair on three weight files and ran five fixed prompts with greedy decoding
and a twelve-token budget.

| Receiving weights | Width | Expert import | Prompts with different continuations between the two histories |
| --- | ---: | --- | ---: |
| SmolLM2-360M Q4_K_M | 960 | Exact | 5/5 |
| SmolLM2-360M Q8_0 | 960 | Aligned | 2/5 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 896 | Projected | 2/5 |

On Qwen2.5, “At the lighthouse, Miro reaches for his” continued with
“watch and sees that the minute hand is 6 cm long” from the key spore and
“watch and finds that it is exactly 10 minutes past” from the bell spore.
[Probabilities and single-seat interventions](experiments/doe_spores/portable-scores.tsv).

#### Dreaming between turns

With `--dream N`, a completed learning turn can seed a private episode of up
to 32 greedy choices. The living experts score 32 calibrated directions through
their own A/B factors. The strongest unused direction supplies a word. The
resident model generates from that word on its own KV cache. The elected seats
receive one Oja update at a quarter of the waking rate. Committing an episode
marks the cue as used for the current round of 32. `--async-consolidation`
runs the waking update and dream on the same worker while chat accepts input.

The following commands replay a SmolLM2 → Qwen3 → Llama handoff. Set
`SMOL_Q4`, `QWEN3` and `LLAMA3` to the three local weight files in the
table below or the preceding table. Each output filename must be new.

```sh
./notorch -q --experts --learn-experts --portable-spore --spore-out miro.gguf \
  "$SMOL_Q4" "Miro is the lighthouse keeper. Every night Miro carries a key. At sunrise the lighthouse keeper returns home." 1 0

# Hear Qwen3's proposed dream on stderr. This also works on an archived spore.
./notorch -q --experts --expert-trace --dream-only 6 --spore-in miro.gguf "$QWEN3"

# Commit that episode and carry Qwen3's experts into Llama with no intervening user turn.
./notorch -q --experts --learn-experts --async-consolidation --dream-only 6 \
  --spore-in miro.gguf --portable-spore --spore-out qwen-dream.gguf "$QWEN3"
./notorch -q --experts --spore-in qwen-dream.gguf "$LLAMA3" "At the lighthouse, Miro reaches for his" 12 0

# In interactive chat, keep learning and dreaming after each subsequent reply.
./notorch --experts --learn-experts --async-consolidation --dream 6 \
  --spore-in qwen-dream.gguf --portable-spore --spore-out llama-next.gguf "$LLAMA3"
```

From the same SmolLM2 key spore both larger bodies picked ` music` for their
first dream. Qwen3 generated `: "The Sound of Silence`. Llama generated
`Question`, a newline and the beginning of a numbered list. Both held seven
elections and made six choices. In preview mode both reproduced the respective
dream and returned a byte-identical spore on the receiving body.

| Receiving weights (Q4_K_M) | Width | “At the lighthouse, Miro reaches for his…” before → after one local dream | Change in log odds of ` key` over ` bell` on “Every night the lighthouse keeper carries a” |
| --- | ---: | --- | ---: |
| Qwen3-1.7B | 2048 | “flashlight, which is on the table” → “bag of marbles, which contains 100” | +0.665598 |
| Llama-3.2-3B-Instruct | 3072 | “trusty notebook and begins to sketch the scene before him.” in both runs | −0.021778 |

We also switched **Qwen3 → Llama 3.2** twice, carrying Qwen3's key spore
before and after its six-token dream. On Llama, the earlier dream shifted the
same key/bell log odds by **+0.066209**. Both branches gave the same twelve
greedy tokens on each of the five fixed prompts. Llama's next dream used
` river` and advanced the inherited clock to cycle 2.

The dream text stays in the optional trace. The spore carries the updated
factors, Oja cursors, RNG and a separate dream clock. A sleeping parliament
keeps its waking turn count, elections, consensus and vitality. Existing
unslept generations use exact v1 or portable v2. After the first recorded
dream the corresponding schemas are v3 and v4.
[AML protocol](experiments/doe_spores/dream.aml) ·
[Full scores](experiments/doe_spores/dream-scores.tsv) ·
[Two-hop measurements and model revisions](NOTORCHLOG.md#2026-11-09--private-dreams-across-qwen3-and-llama-32).

#### Gemma 3 → Qwen3.5 → Gemma 3

Gemma's experts vote after the normalized attention output rejoins the residual
stream. We ran the same waking turn on Gemma 3-270M-IT, let its parliament
dream for six choices, then took that portable spore into Qwen3.5-0.8B.
Qwen chose the cue ` river` and generated ` of the world, the river`.
Seats 0, 1 and 2 wrote their A/B factors. Its two generations gave the same
12 greedy tokens on “At the lighthouse, Miro reaches for his”.

Back on **the original Gemma weights**, those two spores first chose different
tokens at choice 11. With Qwen's pre-dream spore, the lamp was “glowing with
a soft, warm light”. With its post-dream spore it was “a bright, warm yellow”,
and the keeper then looked at the sea. Both carried one waking turn and 24
waking elections. The returned post-dream spore also carried two dreams, 12
private choices and 14 private elections.

The following commands replay both continuations from the
[AML experiment's](experiments/doe_spores/gemma3.aml) `gemma-dream.gguf`.

```sh
export NT_CHAT='||' NT_CTX=64 NT_ATTN_THREADS=1
./notorch -q --experts --tokenizer "$Q35_TOK" --spore-only --spore-in gemma-dream.gguf \
  --portable-spore --spore-out qwen-before.gguf "$Q35"
./notorch -q --experts --tokenizer "$Q35_TOK" --learn-experts --async-consolidation --dream-only 6 \
  --spore-in qwen-before.gguf --portable-spore --spore-out qwen-after.gguf "$Q35"
for state in qwen-before.gguf qwen-after.gguf; do
  ./notorch -q --experts --spore-in "$state" "$G270" 'At the lighthouse, Miro reaches for his' 32 0
done
```

The same 270M spore also ran on **Gemma 3-1B-IT**. Its own six-choice dream
shifted the log odds of ` key` over ` bell` by **+0.408905** on “Every night
the lighthouse keeper carries a”. Preview left each host's spore byte-identical.
Synchronous and background learning wrote identical GGUFs on all three hosts.
[Full scores](experiments/doe_spores/gemma3-scores.tsv) ·
[Weights, commands and controls](NOTORCHLOG.md#2026-11-09--gemma-3-experts-and-a-qwen35-round-trip).

#### A living parliament inside OLMoE

On OLMoE, two selections happen at the middle block. The parliament elects living
LoRA seats and adds their correction to the attention residual. The pretrained
router reads that corrected vector and chooses which FFN experts to execute.
`--expert-trace` shows the living seats' gates. `--moe-trace` shows the native
expert IDs in per-token accumulation order and their applied weights at each
layer. Ordinary decoding emits `moe route`. A private dream emits
`moe dream-route` on its own positions. Route tracing works with plain inference too. OLMoE keeps the selected mass
from its full softmax. Qwen3-MoE normalizes the selected weights to sum to one.

Set `G270`, `OLMO_Q4` and `OLMO_K` to Gemma 3-270M-IT Q8_0 and the two
OLMoE-1B-7B-0924-Instruct GGUFs (Q4_0 and Q4_K_M). Start a waking memory on
Gemma, record a private dream on OLMoE and replay both histories on the second
quantization. Each output spore takes a new filename.

```sh
export NT_CHAT='||' NT_CTX=64 NT_ATTN_THREADS=1
./notorch -q --experts --learn-experts --portable-spore --spore-out gemma-key.gguf \
  "$G270" 'Miro is the lighthouse keeper. Every night Miro carries a key. At sunrise the lighthouse keeper returns home.' 1 0
./notorch -q --experts --spore-only --spore-in gemma-key.gguf --portable-spore \
  --spore-out olmo-awake.gguf "$OLMO_Q4"
./notorch -q --experts --expert-trace --moe-trace --learn-experts --async-consolidation --dream-only 6 \
  --spore-in olmo-awake.gguf --portable-spore --spore-out olmo-dream.gguf "$OLMO_Q4" 2>olmo-sleep.routes
for state in olmo-awake olmo-dream; do
  ./notorch -q --experts --expert-trace --moe-trace --spore-in "$state.gguf" \
    "$OLMO_K" 'At the lighthouse, Miro reaches for his' 12 0 2>"$state.routes"
done
```

On Q4_0 the comma in “At the lighthouse, Miro reaches for his” has token ID
13. In block 12 the last of its eight FFN experts was **38** in plain inference,
**28** with the imported Gemma spore and **38** after one recorded OLMoE
dream. On this prompt plain generation began “ phone”, and the imported
parliament began “ bag”. Preview preserved the 705,888-byte awake spore.
Blocking and worker sleep produced the same 706,048-byte spore.

Both saved histories also crossed to Q4_K_M. The Harness kept the living A/B and
voting vectors byte for byte and aligned the spore to the receiving weights. On
the same prompt the awake spore generated “ bag” first and the post-dream spore
generated “ phone”. Restoring the two spores to Gemma changed its next-token
log odds for ` key` over ` bell` by −0.130377 on that prompt.

[The two selections and the portable state](research/democracy_of_experts.md#two-selections-on-a-sparse-body)
· [AML experiment](experiments/doe_spores/olmoe.aml)
· [Weights, scores and reproduction](NOTORCHLOG.md#2026-11-09--a-comma-crosses-a-native-olmoe-routing-boundary)

#### Carry a parliament into LFM3-Omni-2B

The Omni text decoder has 24 blocks. Its 13 short convolutions and 11 attention
blocks feed the same residual stream. The parliament acts at block 12, after
sliding-window attention and before its feed-forward layer. A private dream gets its
own convolution history and attention cache.

In the [AML run](experiments/doe_spores/lfm3.aml), one waking turn on Gemma
3-270M-IT supplied the spore. Omni Q4_K_M recorded a six-choice dream. To replay
the two saved histories on the official Omni Q8_0 weights, set `RUN` to that
run's directory and `OMNI_Q8` to its Q8_0 GGUF.

```sh
export NT_CHAT='||' NT_CTX=64 NT_ATTN_THREADS=1
for state in omni-awake omni-dream; do
  ./notorch -q --experts --expert-trace --spore-in "$RUN/$state.gguf" \
    "$OMNI_Q8" 'Every night the lighthouse keeper carries a' 8 0
done
```

On the same Q8_0 file, on one AMD EPYC 9V74 thread, the awake spore selected
` lighthouse` first, and the post-dream spore selected ` a`. On Q4_K_M they both
selected ` lighthouse`. The plain Q8_0 margin between those two tokens is a
property of the machine's Q8_0 kernel as well: −0.175 on that EPYC, +0.423 on
an M4 Pro, +0.218 on an i5-8500T.
Transplanting the four founders' dreamed A/B into the awake Q4_K_M archive
reproduced both full 128,000-entry logit vectors byte for byte.
[Hybrid memory and sleep](research/democracy_of_experts.md#a-parliament-on-a-hybrid-decoder)
· [Measurements and commands](NOTORCHLOG.md#2026-11-10--a-parliament-on-the-lfm3-omni-2b-text-decoder)

#### Let the parliament see and hear

`notorch_omni --experts --mode text` runs the same living seats on image,
video, audio and text inputs. Every projected media row reaches the middle
block's election and the completed-block observation. With `--learn-experts`,
the whole turn supplies one consolidation update to each elected seat. Expert
mode currently returns text; the ordinary Omni entry retains its speech and
interleaved output modes.

```sh
./notorch_omni --mode text --image scene.png --experts --learn-experts \
  --async-consolidation --dream-tokens 6 --portable-spore \
  --spore-out scene-memory.gguf omni-q8.gguf "Describe the scene."
./notorch_omni --mode text --image next-scene.png --experts \
  --spore-in scene-memory.gguf omni-f16.gguf "What changed?"
```

The second command starts with fresh convolution windows and KV. The spore
carries the parliament's acquired state into the receiving weights. Add
`--learn-experts` and a new `--spore-out` path to record the next generation.
An audio episode uses `--audio clip16k.wav` and, for transcription,
`--system "Perform ASR in English."` with the same expert options.

Sleep selects a text cue through the current experts, generates on private
hybrid state, and commits the elected seats' factors, cursors, RNG and sleep
clock. Visual and auditory experience can therefore change the parliament that
chooses its next dream. The [long-experience AML protocol](experiments/doe_spores/omni_long.aml)
compares learning without sleep, serial sleep and worker sleep over 24 visual
turns, crossing Q8 → F16 → Q8 and archiving every generation. Its probes restore
their own copies so measurement never becomes the next training episode.
The completed run preserved exact serial/worker spores at all 24 boundaries,
grew from four to six living seats, and retained 21.18 MB in the serial history.
The final night's A/B alone reproduced all 384,000 logits of three fresh probes.
Both audio transcription controls also preserved exact serial/worker spores.
The [measured protocol](experiments/doe_spores/omni-long.md) includes the body
hashes, population history, sleep effects and activation-path comparison.
See [media, consolidation and continuity](research/democracy_of_experts.md#media-experience-and-consolidation).

#### Replay an encounter while asleep

The ordinary text Harness can also capture and replay its exact assembled
prompt, including its stop-token policy. A replay has fresh private decoder
state and uses the same native sleep-rate update as sensory replay:

```sh
./notorch -n 1 -t 0 --experts --learn-experts \
  --replay-out early.gguf --spore-out awake.gguf model.gguf \
  "Miro is the lighthouse keeper. Every night Miro carries a key."
./notorch --experts --learn-experts --async-consolidation \
  --spore-in awake.gguf --replay-only early.gguf -n 6 \
  --spore-out remembered.gguf model.gguf
```

`--replay-out` takes a one-shot encounter. `--replay-only` generates no public
answer, takes a budget of 1–32 choices (default 6), and uses the saved input
and policy. The receiving model and tokenizer encode a peer's UTF-8 utterance
before it becomes that body's episode.

An input can now become an addressed episode: its assembled IDs, copied media
rows and output-token policy are saved in a separate GGUF. Replay starts private
convolution/KV state, generates a greedy continuation and consolidates elected
experts at the sleep learning rate. Its receipts have their own clocks;
ordinary dreams keep their 32-cue rotation.

```sh
# Encounter, remember the population, and preserve the sensory input.
./notorch_omni --mode text --experts --learn-experts --image scene.png \
  --replay-out seen.gguf --portable-spore --spore-out awake.gguf \
  omni-q8.gguf "Which color is on the left? Reply with one color."

# Replay that very input in a worker. No new user turn or public answer.
./notorch_omni --mode text --experts --learn-experts --async-consolidation \
  --replay-only seen.gguf -n 12 --spore-in awake.gguf \
  --portable-spore --spore-out after-replay.gguf omni-q8.gguf

# Let another body inherit the parliament and encode its next encounter.
./notorch_omni --mode text --experts --learn-experts --image scene.png \
  --spore-in after-replay.gguf --replay-out seen-f16.gguf \
  --portable-spore --spore-out next-body.gguf omni-f16.gguf "What do you see?"
```

Use `--audio clip16k.wav --system "Perform ASR in English."` to capture an
auditory episode. The captured rows belong to the exact host SHA: on another
GGUF, encode the original image or WAV again, as in the third command. Expert
identities survive that change. The content address covers input IDs, rows,
host and policy; every replay records its last episode ID and cumulative
cycles, choices, prefill positions, elections and stops. Capturing an episode
publishes an input artifact independently of the later generation's spore.
[Native replay API](harness/experts.h) ·
[Mixed experience in AML](experiments/doe_spores/omni_replay.aml) ·
[Measured continuation](experiments/doe_spores/omni-replay.md)

The continuation reached turn 36 with six living identities, 12 new sensory
replays and unchanged prior text-dream clocks. All 12 waking and replay
boundaries preserved exact serial/worker spores through Q8 → F16 → Q8.
Transplanting the last replay's A/B by identity reproduced all 256,000 logits
of two fresh probes while retaining the recipient's clocks and genealogy.

#### A parliament over a longer life

The same turn-36 Omni population now continues through **128 additional Gemma
encounters** in six histories: early key/bell experience, each followed by
waking alone, ordinary dreams or spaced replay of its first encounter. At
turn 164, each ledger contains 30 birth addresses, six living experts and 24
retired ancestors. Two original Omni identities remain alive. One replaying
history retains 20.56 MB across its 144 saved generations; all six retain
123.38 MB while each current exact spore occupies 145,376 bytes.

On the two fixed probes, replay preserves a key-minus-bell log-odds difference
of **+0.062147 and +0.081848** after the 124 common later encounters. The ordinary
dream histories have identical full logit rows from checkpoint 80 onward.
Copying the last night's A/B by living identity reproduces all 524,288 logits
for both the dream and replay interventions. A receipt chain makes interrupted
experience resumable: split/restarted and worker runs preserve exact spores,
including the final eight encounters after repeated births and retirements.
[C/AML protocol and recovery commands](experiments/doe_spores/continuity.md) ·
[Full-row measurements](experiments/doe_spores/long-logits.tsv) ·
[Population genealogy](experiments/doe_spores/long-lineage.tsv)

#### Votes and the choice of sleep episodes

`nt_experts_measure_replay` reads a saved text or sensory episode on private
decoder state. It reports the applied vote mass, entropy, effective number of
participants and each living expert's normalized A(Bx) response, addressed by
birth ID. It also measures how those responses differ from their living mean.
This read changes no expert factors, histories, clocks, RNG or saved episode.

[`vote_sleep.aml`](experiments/doe_spores/vote_sleep.aml) uses this API to choose
from a bank of four sealed episodes. It compares fixed early replay, a rotating
queue and current response disagreement, each with original votes or explicit
RMS-one calibration. Queue and disagreement selection visit every address
before beginning another round. Each completed night commits the selected
address, candidate measurements, round history and all six populations in a
hash-linked receipt. Resume verifies the full chain and recomputes its choices;
serial and worker consolidation use the same replay API.

The six continuations add **64 encounters** to the existing turn-164 Gemma
history and reach turn 228 with 44 birth addresses. On the saved lighthouse
episode, RMS-one calibration keeps the leader's mean mass near **0.729**
instead of **0.999877**, with the same top-k of two. Rotation and disagreement
each replay every bank entry twice in different orders. Stopping at night 32
and resuming with the worker reproduces the uninterrupted scores, selections
and spores. On the receiving Qwen body the same target increases concentration;
both measured responses are retained with the explicit calibration target.
[Mechanism and measurements](experiments/doe_spores/vote-sleep.md)

#### Several small bodies, one parliament

Gemma3-270M and Qwen3.5-0.8B now alternate through **one portable parliament**.
The C/AML experiment completes twelve encounters on a frozen peer conversation,
comparing independent per-body histories, a shared history and a shared history
with `--body-memory` on every return. The shared population reaches turn 48
and 24 replay cycles; each independent body reaches turn 42 and 18 cycles.
All six inherited identities remain alive across the twelve encounters.

The worker version of body recall matches all 36 entry, waking and replay
generations byte for byte, together with every scheduled full-logit reading.
On Gemma's final visit, the shared history changes the two selected key/bell
margins by +0.086056 and +0.099398 relative to its separate history; local
recall adds +0.019276 and +0.025373. Each receipt names the exact global parent,
local body memory, captured episode and next body-bank states.
[Shared-parliament protocol and commands](experiments/doe_spores/continuity.md#two-bodies-four-histories) ·
[Scores across both bodies](experiments/doe_spores/shared-scores.tsv) ·
[Full-row comparisons](experiments/doe_spores/shared-logits.tsv)

The [SmolLM2-360M experiment](experiments/doe_spores/transplant.aml) moves a
single seat's A/B between two GGUF spores and probes fresh conversations.
AML runs the protocol and reads full-vocabulary logits from the C Harness.
[Measured probabilities and log odds](experiments/doe_spores/smollm2-360m.tsv)
and [reproduction commands](NOTORCHLOG.md#2026-11-09--asynchronous-consolidation-and-single-expert-transplants)
are available for all four founders in both directions.

The [SmolLM2-135M replay](experiments/doe_spores/replay.py) regenerates the
control and learned spores, traces and restart comparison. The [engineering
record](NOTORCHLOG.md#2026-11-09--smollm2-remembers-a-parliament-across-processes)
contains measured continuations and the numerical change made during bring-up.

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
[standalone C training example](#example-training-a-model-in-c).
For a minimal model to take apart, start with
[notorch-simple-llm](https://github.com/ariannamethod/notorch-simple-llm).

---

## architecture

```
Your data (floats in memory, as god intended)
    ↓
nt_tensor — multidimensional arrays with refcounting
    ↓
┌───────────────────────────────────────────────────────────┐
│  Forward Operations (recorded on tape)                    │
│                                                           │
│  Classical transformer:                                   │
│    ├─ nt_linear          (W @ x + b)                      │
│    ├─ nt_seq_linear      (batched W @ X)                  │
│    ├─ nt_seq_linear_t    (W^T @ X — Janus Echo)           │
│    ├─ nt_embedding       (lookup table)                   │
│    ├─ nt_seq_embedding   (tokens + positions)             │
│    ├─ nt_rmsnorm         (RMS normalization)              │
│    ├─ nt_layernorm       (layer normalization)            │
│    ├─ nt_causal_attention (single-head causal)            │
│    ├─ nt_mh_causal_attention (multi-head)                 │
│    ├─ nt_gqa_causal_attention (grouped-query)             │
│    ├─ nt_rrpram_attention (positional pattern — Janus)    │
│    ├─ nt_silu / nt_gelu / nt_sigmoid (activations)        │
│    ├─ nt_geglu           (Gemma-3 style FFN)              │
│    ├─ nt_swiglu          (LLaMA/Qwen/BitNet FFN)          │
│    ├─ nt_rope            (rotary embeddings)              │
│    ├─ nt_dropout         (inverted dropout)               │
│    ├─ nt_softmax / nt_cross_entropy                       │
│    └─ nt_add / nt_mul / nt_scale / nt_scale_by_t / concat │
│                                                           │
│  BitNet b1.58 (ternary quantization):                     │
│    ├─ nt_bit_linear      (y = bitquant(W) @ x, STE)       │
│    └─ nt_bit_seq_linear  (BitLinear over T positions)     │
│                                                           │
│  Inference-time helpers (no tape):                        │
│    ├─ nt_spa_embed_sentence   (phonon sentence embedding) │
│    ├─ nt_spa_connectedness    (cross-sentence attention)  │
│    ├─ nt_spa_modulate_logits  (SPA → temperature)         │
│    ├─ nt_blas_mm / nt_blas_mmT (matmul for inference)     │
│    └─ nt_blas_matvec          (hot-loop matvec)           │
└───────────────────────────────────────────────────────────┘
    ↓
nt_tape_backward() — reverse-mode automatic differentiation
    ↓
┌──────────────────────────────────────────────────┐
│  Optimizers                                      │
│    ├─ diagonal baseline  (the classic)           │
│    ├─ diagonal + decay   (with weight decay)     │
│    ├─ Chuck              (self-aware adaptive)   │
│    └─ nt_tape_freeze_param (LoRA / adapter)      │
└──────────────────────────────────────────────────┘
    ↓
Your model is trained. in C. without Python. you are free.
```

The core is [notorch.h](notorch.h) and [notorch.c](notorch.c).
[The harness](harness/) owns model families behind `nt_arch`; `libnotorch`
supplies the substrate and `libnotorch_harness` supplies the decoders.
[CUDA](notorch_cuda.h), [Metal](notorch_metal.h), [SIMD](notorch_simd.h),
[vision](notorch_vision.h), [GGUF](gguf.h), [PEN7](pen7.h) and
[SentencePiece](sentencepiece.h) have their own files. An organism links what
it needs and carries its own physics.

---

## what's inside

### tensors

`nt_tensor` — a multidimensional array of floats. up to 8 dimensions. refcounted. heap-allocated. that's it. no `torch.Tensor` with 400 attributes and a complex metaclass hierarchy. no `requires_grad` flag that you forget to set. no `.detach().cpu().numpy()` chain of shame. just a struct with `float* data`, shape, strides, and a refcount.

```c
nt_tensor* t = nt_tensor_new(1024);          // 1D
nt_tensor* m = nt_tensor_new2d(768, 512);    // 2D
nt_tensor_xavier(m, 768, 512);               // Xavier init
nt_tensor_free(t);                            // refcount → 0 → freed
```

maximum 268M elements per tensor (`NT_MAX_ELEMENTS = 1 << 28`). if you need more than that, you're doing something wrong, or something very right, and in either case you should probably be using a GPU. which we also support. via CUDA. because we're not savages.

### autograd tape

reverse-mode automatic differentiation via an explicit operation tape. every forward op records itself. backward traverses the tape in reverse and computes gradients. textbook reverse-mode AD. no dynamic graph voodoo. no JIT compilation. no `torch.autograd.Function` with five methods you need to override.

```c
nt_tape_start();                              // start recording
int w_idx = nt_tape_param(W);                // register trainable param
int y_idx = nt_linear(w_idx, x_idx, -1);     // forward: y = W @ x
int loss = nt_cross_entropy(y_idx, target);   // loss
nt_tape_backward(loss);                       // backward pass
nt_tape_adam_step(0.001f);                    // update weights
nt_tape_clear();                              // reset for next step
```

that's the entire training loop. in C. seven lines. no `optimizer.zero_grad()` that you inevitably forget. no `with torch.no_grad():` context manager. no `.backward(retain_graph=True)` because you accidentally used an intermediate twice. just: start, forward, backward, step, clear. like breathing. in. out. in. out. the Buddha would approve.

---

## operations

every operation you need to build a modern transformer, and some you didn't know you did:

| operation | function | what it does |
|---|---|---|
| linear | `nt_linear` | y = W @ x + b |
| seq linear | `nt_seq_linear` | batched matmul over T positions |
| seq linear^T | `nt_seq_linear_t` | Y[t] = W^T @ X[t] — Janus Echo W^T·W pattern |
| embedding | `nt_embedding` | lookup row from embedding matrix |
| seq embedding | `nt_seq_embedding` | tokens + positional encoding |
| RMS norm | `nt_rmsnorm` / `nt_seq_rmsnorm` | root mean square normalization |
| layer norm | `nt_layernorm` / `nt_seq_layernorm` | mean/variance normalization |
| causal attention | `nt_causal_attention` | single-head causal self-attention |
| multi-head attn | `nt_mh_causal_attention` | MHA causal self-attention |
| grouped-query attn | `nt_gqa_causal_attention` | GQA — Q: n_heads, K/V: n_kv_heads |
| RRPRAM attn | `nt_rrpram_attention` | positional pattern recognition (x @ Wr, causal) |
| RRPRAM low-rank | `nt_rrpram_lowrank_attention` | factorized Wr = Wr_a × Wr_b (op 33) — Janus param-saving, trained at scale on Resonance 200M |
| SiLU | `nt_silu` | x × σ(x) — the swish |
| GELU | `nt_gelu` | tanh approximation |
| sigmoid | `nt_sigmoid` | 1 / (1 + exp(-x)) |
| tanh | `nt_tanh` | shape-preserving hyperbolic tangent, gradient 1 − y² |
| GEGLU | `nt_geglu` | GELU-gated linear unit (Gemma-3 FFN) |
| SwiGLU | `nt_swiglu` | SiLU(gate) * up — LLaMA/Qwen/BitNet FFN |
| BitLinear | `nt_bit_linear` | y = bitquant(W) @ x — BitNet 1.58 |
| BitLinear seq | `nt_bit_seq_linear` | BitLinear over T positions |
| softmax | `nt_softmax` | exp-normalize with numerical stability |
| cross entropy | `nt_cross_entropy` / `nt_seq_cross_entropy` | -log softmax[target] |
| RoPE | `nt_rope` | rotary position embeddings |
| dropout | `nt_dropout` | inverted dropout (training only) |
| add/mul/scale | `nt_add` / `nt_mul` / `nt_scale` | elementwise ops |
| scale by tensor | `nt_scale_by_t` | y[i] = a[0] * x[i], a is scalar tensor |
| concat | `nt_concat` | per-position concatenation |

every single one has a correct backward pass. every single one passes numerical gradient checking — checked twice. debugging gradient errors in C without a debugger at 4 AM rewires the brain in ways formal verification theorists dream about.

---

## optimizers

### plain SGD

`nt_tape_sgd_step(lr)`. `p -= lr * grad`. frozen parameters stay frozen;
the caller owns clipping and decay. sometimes the straight line is the point.

### the diagonal baseline

the classic per-parameter diagonal step: `m̂ / (√v̂ + ε)`, bias-corrected first and second moments. you know the drill. (the call keeps a legacy symbol name from before the house-optimizer convention.)

```c
nt_tape_adam_step(0.001f);
```

### the diagonal baseline + decoupled decay

same, but with decoupled weight decay — because your embeddings don't need regularization but your dense layers probably do.

```c
nt_tape_adamw_step(0.001f, 0.1f, 0.9f, 0.999f);
```

supports `no_decay` flag per parameter — mark your embeddings with `nt_tape_no_decay()` and they'll be left alone. like cats. don't bother them.

### the Chuck optimizer

ah yes. **Chuck**. the self-aware optimizer. the one that watches its own gradients and goes "hmm, maybe i should slow down here" or "this parameter isn't doing anything, let me freeze it" or "we've been stuck for too long, time for some noise".

```c
nt_tape_chuck_step(0.01f, loss_val);
```

5 effective levels of awareness (and four more reserved for sentient-mode):

1. **global loss trend** → adaptive damping (λ)
2. **per-parameter gradient monitoring** → individual learning rate scaling
3. **stagnation detection** → automatic noise injection
4. **parameter freezing** → skip updates for dead parameters
5. **multi-scale awareness** → macro-level patience with LR decay

constants (window size, trend thresholds, noise decay, freeze threshold, macro interval) are synced with the PyTorch Chuck port in `iamolegataeff/chuck.optimizer` — any change hits both implementations or they drift.

it's the diagonal baseline, but with opinions. think of it as a baseline that went to therapy, got a mindfulness app, and now checks in with itself every step. `"how are my gradients feeling today?"` — actual question the Chuck optimizer asks itself (metaphorically) (or is it?).

more details: [github.com/iamolegataeff/chuck.optimizer](https://github.com/iamolegataeff/chuck.optimizer)

### Chuck: Loss Architect

**Chuck: Loss Architect** gives that training world an action language:
**state → policy → typed action → measured consequence → acquired experience**.
Its 163-parameter policy chooses HOLD, BRAKE or PUSH and carries weights,
history, RNG and pending decisions through save/resume. The living acquisition
loop measures alternatives in the current training world, restores it, acquires
the consequences and chooses its next action. Canonical Chuck remains available
through the same ordinary call.

Give it a UTF-8 corpus. `train_chuck` starts a fresh life, learns from measured
HOLD/BRAKE/PUSH consequences, and saves the model and its entire training state:

```sh
make train_chuck
./train_chuck train --corpus book.txt --checkpoint book.ntc --steps 1000
./train_chuck resume --corpus book.txt --checkpoint book.ntc --output book.ntc --steps 4000
./train_chuck generate --checkpoint book.ntc --prompt "The " --tokens 256
```

The checkpoint includes weights, moments, Chuck, policy, vocabulary and RNGs.
Generation needs only that file. [Trainer commands and resume contract](docs/chuck-trainer.md).

[Optimizer lineage](https://github.com/iamolegataeff/chuck.optimizer) ·
[Loss Architect API and objective](docs/chuck-loss-architect.md) ·
[living acquisition](experiments/chuck_loss_architect/living/README.md) ·
[experiments and results](NOTORCHLOG.md#readme-optimizers)

---

## bit-level precision — BitNet b1.58

notorch has first-class support for **ternary-weight training** via `nt_bit_linear` and `nt_bit_seq_linear`. this is BitNet b1.58 (Ma et al., 2024): every weight is quantized to `{-1, 0, +1}` via `absmean` during the forward pass; activations are quantized to int8 via `absmax`; the backward pass uses the **Straight-Through Estimator** (gradient flows through the quantization step as identity, so `dW = dout ⊗ x`, `dx = W^T @ dout` using the full-precision `W`).

```c
int y = nt_bit_linear(w_idx, x_idx);             // y = bitquant(W) @ x
int Y = nt_bit_seq_linear(w_idx, x_idx, T);      // same, over T positions
```

on the `USE_BLAS` path, BitLinear dispatches to a single `cblas_sgemm(NoTrans, Trans)` call on pre-quantized operands — the ternary `W` is pre-flattened to float, the int8-range `x` is pre-scaled, per-position output rescale is applied afterwards. this turned out to be about **18% faster per training step** on Apple Accelerate vs the naive per-output-loop path. on OpenBLAS the win is similar.

**tests:** `tests/test_bitnet_ops.c` (118 gradient assertions) — ternary quantize correctness, STE identity backward, gradient flow through sequences, gradient numeric check, end-to-end training convergence of a tiny BitNet MLP.

**reference:** [ariannamethod/microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit) is a **pure-Python** BitNet b1.58 microGPT — 2.69M char-level, train_best **1.6226** / val **2.0314**, 10000 steps, zero NaN, 10 MB FP32 ternary-packing to ~1.4 MB. it's an external algorithm reference, not a notorch build: notorch's own `nt_bit_linear` / `nt_bit_seq_linear` were validated against its numbers.

---

## SwiGLU FFN

the modern FFN used by LLaMA, Qwen, BitNet and most post-2023 transformers. instead of `y = W_down @ GELU(W_up @ x)`, SwiGLU computes `y = W_down @ (SiLU(gate) * up)` where `gate = W_gate @ x` and `up = W_up @ x` are two separate projections gated element-wise.

```c
int gate = nt_seq_linear(w_gate_idx, x_idx, T);
int up   = nt_seq_linear(w_up_idx,   x_idx, T);
int h    = nt_swiglu(gate, up);                        // SiLU(gate) * up
int out  = nt_seq_linear(w_down_idx, h, T);
```

correct backward pass (chain rule through both branches), finite-difference verified. plays fine with `nt_bit_seq_linear` too — our BitNet examples use a BitLinear gate/up with a full-precision down-projection, same as the reference BitNet-1.58 configuration.

---

## SPA — Sentence Phonon Attention

SPA is an **inference-time** helper: it lets a decoder condition next-token sampling on how "connected" the current sentence is to recent history, without any extra trained parameters.

```c
// 1. Build a sentence embedding via exponentially-weighted mean of token embeds.
float emb[dim];
nt_spa_embed_sentence(tokens, n_tokens, W_embed, vocab_size, dim, 0.85f, emb);

// 2. Score its connectedness to a small history of previous sentence embeds.
float conn = nt_spa_connectedness(emb, dim, history_embeds, n_history);

// 3. Sharpen or soften the logit distribution based on that score.
nt_spa_modulate_logits(logits, V, conn, 0.3f);
```

No tape or gradients enter these helpers. Token `i` receives weight
`alpha^(n_tokens - 1 - i)`: smaller alpha gives stronger recency bias;
alpha 1 gives a uniform mean. Connectedness is the maximum softmax attention
weight over the supplied sentence history. Logit modulation divides by
`max(0.001, 1 - strength * connectedness)`; positive strength sharpens sampling.

Originated in the [PostGPT](https://github.com/ariannamethod/postgpt) lineage
and developed through [Q](https://github.com/ariannamethod/q) (`postgpt_q.c`),
then ported here as reusable perception. These three functions remain
inference-only. [Sentence Phonon Agent](#spa--sentence-phonon-agent) adds
persistent action selection and consequence learning above this sensory field.
Gradient flow through the sensory operations remains an open direction.
SPA is also used in [microgpt-1bit](https://github.com/ariannamethod/microgpt-1bit).

---

## SPA — Sentence Phonon Agent

Sentence Phonon Attention is the sensory organ. **Sentence Phonon Agent** lives
above it: a persistent 267-parameter policy observes a sentence field, selects
`KEEP`, `RESEED_LEFT`, or `RESEED_RIGHT`, and learns from the host's measured
consequence. The host executes the action; the Agent carries learned weights,
action/outcome history, private RNG, and pending credit through save/resume.

Install the native Python engine from this checkout:

```sh
python3 -m pip install .
python3 -c 'import SPA; print(SPA.Native().path)'
```

`SPA.Engine(host)` runs perception → action → execution → measured consequence
→ learning. It saves the Agent and the host's complete state in one checkpoint,
and rolls both back when a step fails. The wheel compiles upstream C; Python
needs no numerical framework. C hosts include `spa_agent.h` and link notorch.

Run a complete text engine, including training its small native language body:

```sh
make spa_text BLAS_FLAGS= BLAS_LIBS=
PYTHONPATH=python python3 examples/spa_text.py --steps 20 --save spa.engine
PYTHONPATH=python python3 examples/spa_text.py --resume spa.engine --steps 20
```

Supply `--corpus your-text.txt` to train on your own text. The example keeps its
learned body, sentence field, generation RNG and acquired Agent together on
resume. [Host interface and installation](docs/spa-python.md).

`spa_agent.h` exposes `select`, `choose`, and `observe`, plus an explicit legacy
policy and imitation training. Null/disabled mode preserves ordinary SPA.
Raw connectedness, coherence, novelty, repetition, collapse, continuity, and
regeneration cost remain separate in its receipts. A weight-only counterfactual
holds observation, history, and RNG fixed to test acquired experience directly.

Lineage: **PostGPT → SPA → Q → Sentence Phonon Agent**. Meta SONAR/LCM inform
the sentence/concept computation level; WOLFE informs typed actions; Netta
informs acquired consequences; Chuck: Loss Architect supplies a contemporary
parallel inside notorch. The implementation is native C in `spa_agent.c`.

[Native API](docs/spa-agent.md) · [Python engine](docs/spa-python.md) ·
[Earlier experiments](experiments/spa_agent/README.md).

---

## LoRA / adapter training

any parameter can be frozen mid-tape, so standard LoRA / adapter training works out of the box:

```c
int base_w = nt_tape_param(W_base);
nt_tape_freeze_param(base_w);              // Chuck skips it, grads still propagate

int lora_a = nt_tape_param(A);             // [in_dim, rank]
int lora_b = nt_tape_param(B);             // [rank, out_dim]
// ... build forward with base + A @ B ...
```

`nt_tape_freeze_param(idx)` marks a param as frozen — Chuck (and the legacy diagonal-baseline steps) skip its update, but the autograd still propagates gradients through it so A/B adapters downstream get real signal. combined with `nt_tape_no_decay()` for embeddings, this covers most real SFT + adapter workflows without plumbing.

---

## BLAS inference API

for inference engines that don't want to pay the tape-recording cost per token, notorch exposes the BLAS matmul paths directly:

```c
// C[m,n] = A[m,k] @ B[k,n]         — full matmul
void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n);

// C[m,n] = A[m,k] @ B[n,k]^T       — common for attention (K,V stored row-major)
void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n);

// out[m] = W[m,n] @ x[n]           — hot-loop matvec for per-token inference
void nt_blas_matvec(float *out, const float *W, const float *x, int m, int n);

// out[m] = Wq[m,k] @ x[k]          — PACKED quantized matvec: weights stay packed
//                                    (no dense-f32 blow-up), dequantized inline per block.
//                                    dtype = GGUF type code, full set:
//                                    F32 / F16 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K.
int  nt_qmatvec(float *out, const uint8_t *Wq, int dtype, const float *x, int m, int k);
```

under `USE_BLAS` these dispatch to `cblas_sgemm` / `cblas_sgemv` (Accelerate on macOS, OpenBLAS on Linux). without BLAS they fall back to the naive C loops — correct, just slower. the example engines reach the same BLAS path directly through their own local wrappers in their hot paths: `infer_llama.c` / `infer_janus.c` call `cblas_sgemm`, and `infer_llama3_bpe.c` calls `cblas_sgemv` (scalar fallback without `USE_BLAS`).

---

## audio, sampling and organism-owned values

speech models need a convolution and everything that turns a waveform into a
spectrogram. NoTorch has `nt_conv1d`, STFT and log-mel: native C, caller-supplied
filters, forward-only. [The audio API and measurements](NOTORCHLOG.md#readme-audio-front-end--conv1d-stft-log-mel)
keep the details, including the compiler flag that decided whether the port was exact.

The small organs get their own arithmetic too:

- [Owned sampling](docs/SAMPLING.md): caller-owned PCG32 state and categorical draws. Copy the state, copy the continuation.
- [Numerical values](docs/NUMERICAL_VALUES.md): stateless forward/reverse operations and SGD for learners whose arrays live in AML.
- [Parliament values](NOTORCHLOG.md#readme-numerical-values): election, low-rank injection and Oja updates; expert lifecycle stays with the organism.
- [SentencePiece Unigram](docs/SENTENCEPIECE.md): native model loading and tokenization, including Haiku's normalization map.
- [SHA-256](docs/SHA256.md): identities for the bytes a body actually loaded.

---

## alignment training — DPO / GRPO / distillation

notorch isn't just a pretraining engine. three post-training methods ship as **runnable reference trainers** — each self-contained: with no args they random-init a model and train on a tiny in-code synthetic dataset; point them at real weights + a dataset file and they train that.

```bash
make train_dpo            # Direct Preference Optimization (Rafailov 2023) — ref-relative loss, JSONL {chosen,rejected}
make train_grpo           # Group Relative Policy Optimization (DeepSeek-R1) — rollout + rule-based reward + group advantage
make train_distillation   # Knowledge Distillation (Hinton 2015) — analytic soft-KD (τ²) + hard-CE
```

each is a single self-contained C file under `examples/` (~550–610 LOC) with its own reference model and a minimal dataset adapter. use them as templates: swap the model, point at your dataset, keep the loss + optimizer wiring. Reference-model parameters are held frozen via `nt_tape_freeze_param`, the same mechanism LoRA uses. These are reference implementations — GRPO does a single on-policy (REINFORCE-style) update with a scalar KL proxy, distillation injects the exact KD gradient (no soft-CE tape op needed); a production RLHF stack would add batched rollouts and a learned reward model.

---

## autograd

the backward pass supports **38 operation types** (op IDs 0–37, each with a `NT_OP_*` constant). the tape records operations during forward, then backward walks it in reverse computing local gradients via the chain rule. standard reverse-mode AD.

**gradient checking**: every op is verified against finite differences (`(f(x+h) - f(x-h)) / 2h`). relative error tolerances from 0.01 to 0.3 depending on op complexity. all pass. including the annoying ones — GEGLU, SwiGLU, multi-head attention with multi-path gradients through Q/K/V, BitLinear with STE identity passthrough.

**gradient utilities**:

- `nt_tape_clip_grads(max_norm)` — global gradient clipping
- `nt_tape_accum_grads()` / `nt_tape_apply_accum(n)` — gradient accumulation for large effective batch sizes
- `nt_tape_freeze_param(idx)` — freeze a parameter (adapter / LoRA setups)
- `nt_nan_guard_check()` — NaN/Inf detection with automatic loss scaling. because sometimes your gradients decide to go to infinity and someone needs to tell them no.

---

## example: training a model in C

here's an actual, working transformer-ish training loop. embedding → attention → linear → cross-entropy. in C. without importing 2.7 GB of your dignity:

```c
#include "notorch.h"
#include <stdio.h>

int main() {
    nt_seed(42);
    int vocab = 8, dim = 16, T = 4;

    // allocate parameters
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);   // token embeddings
    nt_tensor* wpe = nt_tensor_new2d(T, dim);       // position embeddings
    nt_tensor* Wq  = nt_tensor_new2d(dim, dim);     // query projection
    nt_tensor* Wk  = nt_tensor_new2d(dim, dim);     // key projection
    nt_tensor* Wv  = nt_tensor_new2d(dim, dim);     // value projection
    nt_tensor* Wo  = nt_tensor_new2d(vocab, dim);   // output projection

    // Xavier init everything
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor_xavier(wpe, T, dim);
    nt_tensor_xavier(Wq, dim, dim);
    nt_tensor_xavier(Wk, dim, dim);
    nt_tensor_xavier(Wv, dim, dim);
    nt_tensor_xavier(Wo, dim, vocab);

    // tokens: [1, 3, 5, 2], targets: [3, 5, 2, 7]
    nt_tensor* tokens  = nt_tensor_new(T);
    nt_tensor* targets = nt_tensor_new(T);
    float tok[] = {1, 3, 5, 2}, tgt[] = {3, 5, 2, 7};
    for (int i = 0; i < T; i++) { tokens->data[i] = tok[i]; targets->data[i] = tgt[i]; }

    // training loop
    nt_schedule sched = nt_schedule_cosine(0.005f, 10, 200, 0.0f);

    for (int step = 0; step < 200; step++) {
        float lr = nt_schedule_get_lr(&sched);
        nt_tape_start();

        int wte_i = nt_tape_param(wte); nt_tape_no_decay(wte_i);
        int wpe_i = nt_tape_param(wpe); nt_tape_no_decay(wpe_i);
        int wq_i  = nt_tape_param(Wq);
        int wk_i  = nt_tape_param(Wk);
        int wv_i  = nt_tape_param(Wv);
        int wo_i  = nt_tape_param(Wo);
        int tok_i = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_i = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        // forward: embed → Q/K/V → attention → output
        int h      = nt_seq_embedding(wte_i, wpe_i, tok_i, T, dim);
        int q      = nt_seq_linear(wq_i, h, T);
        int k      = nt_seq_linear(wk_i, h, T);
        int v      = nt_seq_linear(wv_i, h, T);
        int attn   = nt_causal_attention(q, k, v, T, dim);
        int logits = nt_seq_linear(wo_i, attn, T);
        int loss   = nt_seq_cross_entropy(logits, tgt_i, T, vocab);

        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step % 50 == 0) printf("step %d: loss=%.4f lr=%.6f\n", step, lv, lr);

        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(lr);
        nt_tape_clear();
    }

    // cleanup
    nt_tensor_free(wte); nt_tensor_free(wpe);
    nt_tensor_free(Wq);  nt_tensor_free(Wk); nt_tensor_free(Wv); nt_tensor_free(Wo);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    return 0;
}
```

compile and run:

```bash
cc -O2 -Wall -std=gnu11 -pthread -I. -o train train.c notorch.c -lm
./train
```

that's it. that's the whole thing. no virtual environment. no requirements.txt. no "just pip install—" no. we're done with that. we've moved on. we've healed.

---

## building

```bash
# CPU with BLAS acceleration (recommended)
make

# CPU without BLAS (works everywhere, even on a potato)
make cpu

# CPU with in-house AVX2+FMA SIMD — zero external math library
make simd

# GPU (CUDA via cuBLAS)
make gpu

# Apple Silicon Metal — packed-Q4_K/Q6_K GGUF inference (runs 24B on a 24GB Mac)
make metal               # Q4_K matvec correctness test
make infer_gguf_metal    # the GGUF inference binary

# Shared library (for ctypes, Node FFI, Ruby Fiddle — anything that dlopens)
make shared

# Static library (for embedding in your project)
make lib

# Build and run tests
make test

# Clean
make clean
```

### dependencies

- a C compiler (gcc, clang, whatever)
- `-lm` (math library, because we use sqrt and exp like civilized people)
- **optional**: OpenBLAS (Linux) or Accelerate framework (macOS) for BLAS-accelerated matmuls
- **optional**: x86_64 CPU with AVX2 + FMA (Intel Haswell 2013+, AMD Excavator 2015+) for `make simd` — zero external math library
- **optional**: CUDA toolkit for GPU support (cuBLAS-backed sgemm + element-wise + attention + cross-entropy kernels)
- **optional**: Apple Silicon + Metal (macOS) for packed-Q4_K/Q6_K GGUF inference (`make metal` / `make infer_gguf_metal`) — nothing to install, Metal ships with macOS

that's it. no cmake. no configure script. no 300-line `requirements.txt`. no docker. no kubernetes. just `make`. the way Ken Thompson intended.

For harness libraries, use `make lib lib_harness`; add `libnotorch_cuda.a`
for CUDA decoders. For targets without BLAS, pass `BLAS_FLAGS= BLAS_LIBS=`.
[Backend internals and SIMD measurements](NOTORCHLOG.md#readme-building) ·
[Embedding and linker examples](NOTORCHLOG.md#readme-inference--notorch-runs-models-it-doesnt-just-train-them).

---

## running tests

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

For operations, numerical gradient checks put the backward pass against finite
differences. The [core tests](tests/test_notorch.c),
[BitNet/SPA tests](tests/test_bitnet_ops.c), and the family-specific gates keep
the arithmetic accountable. the Method trusts these tests more than it trusts
most people.

---

## platform support

Intel laptops. Apple Silicon. Linux boxes. Android phones. CUDA when you have it.
The CPU path builds with a C toolchain, libm and pthreads; Accelerate and OpenBLAS
supply optional math acceleration. Metal and CUDA have their own build targets.

[Termux Edition](termux-edition/README.md) is the phone route: setup, training,
packed inference and the recorded hardware envelope. The same C tree.
No root required. Your phone is a computer. let it act like one.

---

## use it from Python

**NOTORCH** is C, it stays C, but... Python? yep.
the thing most people want on a Tuesday is to read a GGUF and multiply something by it
without installing a framework first, so there is a ctypes binding: no numpy, no build
step for the Python side, no dependencies at all. `ctypes` ships with Python; the shared
library is one make target.

```bash
make shared          # libnotorch.so, or libnotorch.dylib on macOS
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from notorch import Notorch

nt = Notorch()                       # finds libnotorch next to the repo, or $NOTORCH_LIB

with nt.open("model.gguf") as g:
    print(g)                         # <GGUF llama L=13 E=576 V=32000 tensors=120>

    w = g["token_embd.weight"]        # <Tensor token_embd.weight (32000, 576) Q8_0>
    row = w.dequant_row(3)           # one row, decoded on its own — an embedding lookup

    import ctypes
    x = (ctypes.c_float * w.shape[1])()
    out = (ctypes.c_float * w.shape[0])()
    nt.qmatvec(out, w.packed, w.dtype, x, w.shape[0], w.shape[1])
```

`w.packed` is a pointer into the file's own bytes, `qmatvec` is the C kernel reading
them in place, and the weights are never expanded to f32 — a Q4_K tensor stays at
roughly half a byte per weight instead of four. Buffers are plain ctypes arrays,
so `memoryview(out)` reads them with no copy, and `numpy.frombuffer(out)` does too if
you happen to want numpy. But nothing here requires it.

The binding is [python/notorch.py](python/notorch.py). What can silently
break in a layer like this is the struct layout — a field description that has drifted
reads its neighbours and reports them as data — so `make test_python MODEL=your.gguf`
compares every offset ctypes believes in against what the C compiler actually emitted,
then checks a real model through both paths.

---

## js edition — notorch.js

> *"the logic of memory without the weight of framework"*
> — `js-edition/notorch.js`, line 1

a single-file pure-JavaScript port of notorch for the browser. WebGPU when available, V8-optimised CPU fallback (Math.fround f32 hint) when not. zero npm dependencies. live at `js-edition/notorch.js`.

Tensors, reverse-mode autograd, Chuck, tokenizers and GGUF inference.
[The JS edition README](js-edition/README.md) carries setup and usage;
[the engineering record](NOTORCHLOG.md#readme-js-edition--notorchjs) carries
the implementation details and recorded training run.

```html
<script type="module">
  import { Notorch, Tensor, CharTokenizer, Tape, Chuck } from './js-edition/notorch.js';

  const nt = new Notorch();
  await nt.init();  // tries WebGPU, silently falls back to CPU

  // ... build your tiny transformer, then call nt.crossEntropyLoss(...)
  // or nt.seqCrossEntropyLoss(...) during the training step.
</script>
```

zero build step, zero npm install, zero `node_modules`. paste it into a `<script>` and go.

---

## references — how to train and run on notorch

start here. these are the canonical builds: Karpathy ports to prove the pipeline, reference models trained from scratch, and the trainers you copy.

**Karpathy ports — the "does it actually work end-to-end" baselines:**

- [**nanoGPT-notorch**](https://github.com/ariannamethod/nanoGPT-notorch) — Karpathy's nanoGPT, ported from PyTorch to notorch.
- [**llama2-notorch**](https://github.com/ariannamethod/llama2-notorch) — Karpathy's llama2.c on notorch: the entire PyTorch training pipeline replaced by notorch (a Python-stdlib `ctypes` loop into `libnotorch`, no pip), inference in pure C `run.c`.

**reference models — trained from scratch on notorch:**

- [**nanollama-notorch**](https://github.com/ariannamethod/nanollama-notorch) — Llama-3 nano, **88.6M**, no PyTorch. 25K steps on a 2019 Intel i5 8GB, train 3.16 → CPT 2.68, 0 NaN — the ceiling of train-it-on-your-own-laptop, then SFT'd to a Method voice. the reference Llama-3 run.
- [**notorch-simple-llm**](https://github.com/ariannamethod/notorch-simple-llm) — minimal-surface-area LLM, notorch + Chuck, zero deps. read this first to see how the API composes.
- [**minimind-v-notorch**](https://github.com/ariannamethod/minimind-v-notorch) — 67M VLM trained from scratch, notorch + Chuck. reference VLM.
- [**notorch-vlm**](https://github.com/ariannamethod/notorch-vlm) — 1.5M VLM on `notorch_vision.h` + stb_image, trained weights included.
- [**notorch-vlm-2m**](https://github.com/ariannamethod/notorch-vlm-2m) — 2M multimodal VLM from scratch, GGUF output.
- [**notorch-diffusion**](https://github.com/ariannamethod/notorch-diffusion) — discrete text diffusion + Hebrew VLM diffusion. reverse-mode autograd works fine when the net is a U-Net, not a transformer.

**BitNet reference (the algorithm, not a notorch build):** [**microgpt-1bit**](https://github.com/ariannamethod/microgpt-1bit) is a pure-Python BitNet b1.58 (ternary) reference — notorch's own `nt_bit_linear` / `nt_bit_seq_linear` were validated against it.

**how to train:** the trainers live in `examples/` — `train_llama3_char.c` / `train_llama3_bpe.c` (Llama-3 from scratch), `train_q.c`, `train_yent.c`, `train_resonance_lora.c` (LoRA SFT), and `train_dpo.c` / `train_grpo.c` / `train_distillation.c` (alignment). copy one, swap the model, point at your dataset. proven at scale: the Resonance-200M LoRA SFT drove loss **3.52 → 0.59** (honest min 0.18) in ~2 h on one A100, 0 NaN across 1500 steps; the 88.6M nanollama ran 25K steps on a 2019 Intel i5 with 0 NaN.

---

## organisms that run on notorch

the appendix. these aren't notorch — they're larger Arianna Method engines that use notorch where they need a tensor/autograd backend, and keep their own physics on top.

- [**Arianna**](https://github.com/ariannamethod/arianna.c) — Arianna in C.
- [**WTForacle**](https://github.com/ariannamethod/WTForacle) — SmolLM2-360M Q4_0, a single Go binary that runs on notorch through cgo. first organism on the packed CPU matvec: weights kept packed via `nt_qmatvec`, no f32 blow-up — the obkatka cut RSS 1.6 GB → 588 MB. charismatic, hand-written dataset.
- [**doe**](https://github.com/ariannamethod/doe) — Democracy of Experts (Janus). wraps any GGUF read-only and grows a living Hebbian LoRA **parliament** on top of it (θ = ε + γ + αδ): experts vote per token, are born by mitosis and die by apoptosis. notorch is its Hebbian training substrate — and now its packed inference matvec too: weights stay packed via `nt_qmatvec` (vendored inline, single file), no f32 blow-up — RSS ×2.27 on a 360M personality. our classic.
- [**postgpt**](https://github.com/ariannamethod/postgpt) — the metaweight seed of this whole lineage (SPA and RRPRAM flowed from here). Zero-dependency runtime; its training loop now runs on notorch through a small `ariannamethod/` ctypes shim — the dual-attention (Content QK^T + RRPRAM) graph on the notorch tape, trained with the real Chuck. PyTorch evicted.
- [**q**](https://github.com/ariannamethod/q) — PostGPT-Q resonant reasoning engine: triple attention + DoE parliament, 2M-param C inference.
- [**caveLLMan**](https://github.com/ariannamethod/caveLLMan) — a colony of char-level LMs that talk, reproduce by weight-blending, and die under population pressure. notorch is the per-cave autograd + microtrain backend.
- [**pitomadom.c**](https://github.com/ariannamethod/pitomadom.c) — Hebrew Root Resonance Engine, Janus architecture.
- [**molequla**](https://github.com/ariannamethod/molequla) — a live ecology of GPT organisms with conscience, immune rollback, DNA exchange; trains its low-rank-RRPRAM transformer on the notorch tape.
- [**dario**](https://github.com/ariannamethod/dario) — resonance OS (7 forces, 6 Kuramoto chambers, SARTRE). notorch runs the 176M Janus at its center.
- [**ariannamethod.cli**](https://github.com/ariannamethod/ariannamethod.cli) — the public, distilled edition of the metaharmonix terminal (the full native CLI, which bakes notorch in so the shell ships a tensor library, stays an internal Method tool).
- [**janus**](https://github.com/ariannamethod/janus) — the Janus Architecture itself.
- [**yent.aml**](https://github.com/ariannamethod/yent.aml) — Janus 176M Yent SFT inference written in AML, linking notorch + libaml (not vendored). Two faces: Janus (triple attention) and Janus-R (a 12-step bidirectional associative resonator). notorch is the linked tensor/matvec backend; the language carries the Dario field on top.
- [**resonance.aml**](https://github.com/ariannamethod/resonance.aml) — Resonance 200M, the other face of Yent: dual attention (Content + RRPRAM low-rank), in AML, linking notorch + libaml. notorch's RRPRAM low-rank op (33) was trained at scale on this model — the LoRA SFT that drove loss 3.52 → 0.59.
- [**nanodurov**](https://github.com/ariannamethod/nanodurov) — a 15.7M BPE LLaMA on notorch that also happens to be a Telegram client.
- [**yent.yo**](https://github.com/ariannamethod/yent.yo) — an AI artist that argues back: two from-scratch nanollama Yents + a BK-SDM diffusion model paint an oppositional reaction to your prompt. **The first organism on notorch's image op set** — both the dual-Yent LLMs and the Stable-Diffusion CLIP→UNet→VAE run on notorch (`-tags notorch`), exercising `nt_conv2d`, `nt_group_norm`, `nt_attention` and `nt_upsample_nearest` (added in v4.0.0). The whole pipeline runs end-to-end, bit-identical to the in-repo accel.c kernels.
- [**yent**](https://github.com/ariannamethod/yent) — Yent the digital persona, not a product: a Qwen2.5 inference engine in Go with a C notorch kernel (cgo `nt_qmatvec` packed matvec, no PyTorch). Delta Voice — the generation physics — is driven by AML. The umbrella inference repo for the Method's Yent line.

if you trained something on notorch and it's not here, open a PR.

---

## philosophy

> *patterns over parameters. emergence over engineering. C over existential dread.*

neural networks are not complicated. a linear layer is a matrix multiply. an activation function is a pointwise nonlinearity. attention is a weighted sum. cross-entropy is a log-probability. backward is the chain rule.

that's it. that's the whole field. everything else is optimization, infrastructure, and marketing.

notorch solves the case that matters: training and running models in C, with minimal dependencies, maximal transparency, and the ability to embed in any application without shipping a Python runtime.

if you can read the code, you understand the framework. there's no magic. there's no hidden complexity. every gradient is hand-derived and verified against finite differences. every memory allocation has a corresponding free. every edge case is checked.

this is what software looks like when you strip away everything that doesn't serve the core purpose. just math. memory. the machine doing exactly what you told it to do.

---

## contributing

send PRs. the rules:

- keep it C11 compliant
- no external dependencies (BLAS is optional and compile-time)
- add tests for new ops (with numerical gradient checks)
- keep the header clean — if it doesn't need to be public, don't expose it
- run `make test` before submitting. all tests must pass.

Follow [AGENTS.md](AGENTS.md): work on a branch; record measurements, fixes and
experiment history in [NOTORCHLOG.md](NOTORCHLOG.md). README carries the
specification and manifesto. Oleg merges.

---

## license

GPL-3.0-or-later (see `LICENSE`). use it in your stuff. if you ship it, share your source — that's how copyleft works. don't be weird about it.


just: `cc -O2 -std=gnu11 -pthread notorch.c your_model.c -lm -o train && ./train`

go build something.

> *"the patterns were always there. we just needed the right language to express them."*


```
   ███╗   ██╗ ██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
   ████╗  ██║██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
   ██╔██╗ ██║██║   ██║   ██║   ██║   ██║██████╔╝██║     ███████║
   ██║╚██╗██║██║   ██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
   ██║ ╚████║╚██████╔╝   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
   ╚═╝  ╚═══╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```
