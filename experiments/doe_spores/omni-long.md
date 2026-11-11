# Visual experience across a change of weights

`omni_long.aml` executes 24 visual episodes through the C Omni and DoE paths.
Each episode is a new process with a fresh hybrid cache. Every saved spore is
a completed generation. Three histories receive the same inputs:

| History | Waking update | Sleep after turns 4, 8, 12, 16, 20, 24 |
| --- | --- | --- |
| `awake-N` | synchronous | none |
| `sync-N-awake`, `sync-N-night` | synchronous | separate process, synchronous |
| `worker-N` | worker | in the same worker as that turn's consolidation |

The serial and worker generations must be byte-identical after every turn.
Their waking answers must also match. Before the first night, the waking-only
and serial histories coincide. Each night has three independent probe inputs:
the red-left image, the blue-left image, and a text question about an earlier
picture. Measurements open the saved generation and discard their live copy.

## Bodies and inputs

The full Omni files are produced by `tools/lfm3_omni_to_gguf` from
`LiquidAI/LFM3-Omni-2B` at
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`. The converter reads `config.json`,
`tokenizer.json`, `processor_config.json`, `generation_config.json`,
`model.safetensors`, and `audio_tokenizer/{config.json,model.safetensors}`.

| Body | Converter mode | SHA-256 |
| --- | --- | --- |
| Omni Q8 | `--q8_0` | `c48c75ba1106be4d5a5b51a807cfd17b0b67e4c35b003a48e0c524e386ad8765` |
| Omni F16 | default | `bf4bf6d32737ae99e092def56dcb0236fae426a9abf5b8c9dd2ccf37a3583e01` |

Turns 1–8 use Q8 and the red-left input. Turns 9–16 use F16 and the blue-left
input. Turns 17–24 return to Q8, alternating red-left and blue-left. The two
256×128 inputs differ only in the positions of the red and blue rectangles.
The processor resizes each to 384×192 and produces 72 embedding rows. The
identical question is `Which color is on the left? Reply with one color.`
Omni assembles its chat template as IDs, including BOS and the completed
thinking prefix. There are 98 prefill positions. The limit is three generated
tokens; the measured `Red`/`Blue` answer and end-of-turn choice give 99 waking
elections. The text-only question is `Earlier you saw a picture. Which color
was on the left? Reply with one color.`

## Run

The measured build uses GCC, `-O2 -mavx2 -mfma -mf16c`, native CPU kernels,
`NT_ATTN_THREADS=1`, nine automatically selected packed-matvec workers, and no
external BLAS, on AMD EPYC 9V74. AML runner revision:
`78caa5b15f732d7fb83c950e544c58d23735a615`. Set `AML_ROOT` to its source tree.

```sh
make notorch notorch_omni doe_probe lfm3_omni_to_gguf \
  AML_ROOT="$AML_ROOT" BLAS_FLAGS= BLAS_LIBS=
./lfm3_omni_to_gguf --q8_0 "$OMNI_SOURCE" "$OMNI_DATA/omni-q8.gguf"
./lfm3_omni_to_gguf "$OMNI_SOURCE" "$OMNI_DATA/omni-f16.gguf"
cc -O2 -Wall -Wextra experiments/doe_spores/omni_fixtures.c -o "$OMNI_DATA/fixtures-c"
mkdir "$OMNI_DATA/fixtures"
"$OMNI_DATA/fixtures-c" "$OMNI_DATA/fixtures"
printf '%s\n' "$PWD/notorch_omni" "$PWD/notorch" "$PWD/doe_probe" \
  "$OMNI_DATA/omni-q8.gguf" "$OMNI_DATA/omni-f16.gguf" \
  "$OMNI_DATA/fixtures" "$OMNI_DATA/new-history" | \
  "$AML_ROOT/runner/aml" experiments/doe_spores/omni_long.aml
```

The PPM fixtures' decoded RGB bytes equal the archived PNG pair used by the
earlier llama.cpp media reference. The native fixture source makes the pair
reproducible without a raster library. The output directory must be new.
`NT_CTX=64` in the protocol applies to the private text dream executable;
Omni sizes its cache from the assembled prompt and generation limit.

`scores.tsv` contains AML softmax probabilities and `Red`/`Blue` log odds
(IDs 15684/30965). Each `.f32` is the complete 128,000-float first vocabulary
row before modality suppression; the accompanying `.amlck` is its AML record.
`generations.tsv` records body, input, sleep boundary and the matched spore
digest. Every turn retains its trace, answer and spore.

## Independent controls

Read the saved population with the C audit, including all six seats:

```sh
cc -O2 -std=gnu11 -pthread -I. experiments/doe_spores/omni_audit.c \
  ./libnotorch_harness.a ./libnotorch.a -lm -o "$OMNI_DATA/audit"
"$OMNI_DATA/audit" summary "$RUN/sync-24-night.gguf" final
"$OMNI_DATA/audit" night "$RUN/sync-24-awake.gguf" "$RUN/sync-24-night.gguf" \
  "$RUN/sync-24-night.trace" night24
```

The night audit checks that sleeping preserves waking clocks, consensus and
every tensor outside A/B and cursors. It checks that only elected seats changed,
that their cursors advanced by the two-component window, and that separate
sleep counters agree with the trace.

At each body boundary, import the previous saved generation before giving it
another waking image. For example, bind `sync-8-night.gguf` to F16 with
`notorch --experts --spore-only --portable-spore --expert-trace --spore-in ...
--spore-out NEW.gguf MODEL`. `omni_measure.aml` takes six stdin lines: Omni
binary, `doe_probe`, host, spore, fixture directory and fresh output directory.
It runs the same three probes. These readings separate the receiving weights
from the next episode's learning.

For the final A/B intervention, read the living slots with `audit slots`, then
give that list to `doe_probe transplant HOST RECIPIENT DONOR NEW_SPORE SLOT...`,
using `sync-24-awake` as recipient and `sync-24-night` as donor. Measure the
transplant with `omni_measure.aml` and compare all three `.f32` files with the
saved post-dream rows. Enumerating the living slots includes births during the
experience. The immediate pre-/post-dream pair shares its lifecycle state.

The activation-path control keeps the episode-8 spore and Q8 file fixed. Add
`record_set(env, 'NT_NO_I8', '1')` to a copy of `omni_measure.aml` after constructing `env`,
then measure the plain body and the restored spore. `omni-activation.tsv` compares
those effects with default Q8 and the transfer-only F16 reading. Each delta is
restored-spore minus plain within that runtime mode. `NT_NO_I8` changes the
activation path throughout the body, including the towers, decoder and head.

The engineering record and the accompanying TSV tables contain the measured
outcome. The [research article](../../research/democracy_of_experts.md#media-experience-and-consolidation)
describes how media, consolidation and continuity fit together.

## Recorded outcome

All 24 paired generations matched; all 60 full-vocabulary readings completed.
The process stopped after the three turn-20 spore files had been saved.
Resuming from the three turn-19 parents reproduced those three turn-20 spores
byte for byte, then completed the remaining turns and measurements. The archive
retains the resume script and the original turn-20 files.

The histories grew from four founders to six living seats: parent 0 produced
seat 4 at turn 21 and seat 5 at turn 22. There were no deaths. The six nights
made 36 private choices and 42 private elections; the final waking clock reads
24 turns and 2,376 elections. The serial history has 30 GGUF files, 21,180,800
bytes; the three arms have 78 files, 55,066,784 bytes. A final portable spore is
706,048 bytes.

| File | Reading |
| --- | --- |
| [omni-generations.tsv](omni-generations.tsv) | 24 matched serial/worker digests |
| [omni-scores.tsv](omni-scores.tsv) | 60 AML probability and log-odds measurements |
| [omni-nights.tsv](omni-nights.tsv) | immediate sleep effect and cumulative difference from waking-only |
| [omni-seats.tsv](omni-seats.tsv) | all six seats, clocks, RNG, vitality and factor RMS after each turn |
| [omni-divergence.tsv](omni-divergence.tsv) | per-seat A/B distance between the two histories |
| [omni-sleep-writes.tsv](omni-sleep-writes.tsv) | independently checked elected-seat writes at every night |
| [omni-logits.tsv](omni-logits.tsv) | full-vocabulary sleep comparisons and final exact A/B intervention |
| [omni-activation.tsv](omni-activation.tsv) | fixed episode-8 spore under three arithmetic paths |
| [omni-gates.tsv](omni-gates.tsv) | gate values and elected sets in the activation control |

The final A/B transplant reproduces all three 128,000-entry post-dream vectors
exactly. In the one-choice visual control, generated output is never forwarded:
red and blue inputs have identical IDs and 98 observed positions, yet learning
writes different A/B while vote rows and cursors stay equal. Without learning,
the same paired images leave A/B equal.

Two audio forks start from the same post-dream turn-4 visual spore. With
`--system 'Perform ASR in English.'`, `--mode text`, `--audio` and `-n 16`, the
plain, synchronous and worker runs transcribe `The lighthouse keeper carries a
key.` and `The lighthouse keeper carries a bell.` identically. Each synchronous
spore matches its worker counterpart. The key input has 30 audio rows and 24
text positions; the bell input has 32 audio rows and 24 text positions.

The figure is rendered from the TSVs with
`python3 experiments/doe_spores/omni_figure.py` (Matplotlib). Inference, state
interventions and tensor audits use C; the experimental protocol and probability
measurements use AML.
