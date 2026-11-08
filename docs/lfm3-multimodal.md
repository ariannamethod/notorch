# LFM3-VL-2B and LFM3-Omni-2B on NoTorch: the plan before the C

The text decoder of both checkpoints already runs here: `harness/arch_lfm3.c`
reads the official `lfm3` text GGUFs of LFM3-1.7B, LFM3-VL-2B and LFM3-Omni-2B
(NOTORCHLOG 2026-11-07). Nothing that sees an image or hears a sound exists
yet. This page fixes what is built, from which tensors, in which order, and
against what each step is gated, before any of it is written.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| Omni checkpoint, remote code, configs | `LiquidAI/LFM3-Omni-2B` | `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67` |
| Omni GGUF pair (text + mmproj) | `LiquidAI/LFM3-Omni-2B-GGUF` | as published 2026-10-27 |
| VL checkpoint | `LiquidAI/LFM3-VL-2B` | pinned when its fixtures are cut |
| Reference graphs | llama.cpp `tools/mtmd/models/{lfm3vl,lfm3a,lfm3a-gen}.cpp` | `7d7f0287530bc8981690f361d8c1627dbae1c99a` |

## What the files hold

The official Omni GGUF pair is complete. The text file (`lfm3`, 287 tensors)
is the decoder `arch_lfm3.c` already loads. The mmproj file
(`general.architecture = clip`, 458 tensors, F16/BF16/Q8_0/Q4_K_M/Q4_0) holds
every other part:

| Prefix | Count | Part | Projector type |
| --- | ---: | --- | --- |
| `v.*` | 133 | vision tower: patch embedding (two halves), embedding norm, 14 blocks | `lfm3vl` |
| `mm.0`, `mm.2`, `mm.input_norm` | 6 | vision projector: LayerNorm, unshuffle 2×2, 4096, GELU, 2048 (`lfm3vl.cpp:153`) | |
| `a.conv1d.*`, `a.pre_encode.*` | 12 | audio downsampler: three stride-2 conv2d, two pointwise, 1×1 out | `lfm3a` |
| `a.blk.*`, `a.post_ln` | 158 | audio encoder: LFM2 backbone, 17 blocks, width 768 | |
| `mm.a.mlp.*` | 6 | audio projector: LayerNorm, 768→2048, GELU, 2048→2048 | |
| `a.gen.code.*` | 43 | depth decoder: LFM3, 3 blocks, width 1024, 12 codebooks of 2017 | `lfm3a_gen` |
| `a.gen.wav.*` | 100 | codec decoder from codes to waveform | |

The safetensors checkpoint carries the same parts under `model.vision_tower`,
`model.multi_modal_projector`, `model.audio_encoder`, `model.audio_projector`,
`model.depth_decoder`, `model.codebook_embedding` and a separate
`audio_tokenizer/` codec (665 tensors in the main file).

Position: read the official mmproj GGUF, as the text side reads the official
text GGUF. The tensors are the ones llama.cpp's mtmd runs, so every parity
number is like for like, and no converter is written until a part turns out to
be missing from the file. The VL-2B tower has the same configuration as the
Omni tower (width 1024, 14 layers, the same conv/attention order, 16 query and
8 KV heads, patch 16, axial RoPE θ=1e6, merge 2, temporal patch 2), so one
implementation serves both.

## The parts, from the upstream code

**Vision tower** (`Lfm3VlVisionModel`). Patches of 3×2×16×16 values, grouped
by 2×2 merge block, go through a bias-free linear patchifier to width 1024 and
an RMSNorm. Each of the 14 layers is pre-norm with a residual around the
operator and around a SwiGLU-style MLP (`intermediate_size` 2560). The operator
is one of two:

- `conv` (8 layers): `in_proj` to 3×1024, split B, C, x; `B*x` is scattered from
  merge-block order to the raster grid of its frame, run through a 3×3
  depthwise conv2d with padding 1 and no bias, gathered back, multiplied by C
  and sent through `out_proj`. Frames are convolved separately.
- `full_attention` (6 layers): bidirectional GQA inside one frame, RMSNorm on
  every query and key head, axial 2D RoPE. The rotary table is built over
  `head_dim/2 = 32` frequencies per axis; height and width angles are
  concatenated and repeated, so the rotation covers the whole head.

**Vision projector** (`Lfm3VlMultiModalProjector`): LayerNorm (eps 1e-6) over
each tower output, each 2×2 block concatenated to 4096, `linear_1` (4096→4096,
bias), GELU, `linear_2` (4096→2048, bias). One language token per 2×2 block.

**Audio front end** (`Lfm3OmniFeatureExtractor`): 16 kHz, Hann window 400, FFT
400, hop 160, 128 Slaney mel bands to 8 kHz, power spectrum, `log10`, floor −12,
then `(x + 4) / 4`. `notorch.c` already has a log-mel; whether its padding and
scale match this is the first thing its fixture settles.

**Audio encoder** (`Lfm3OmniEncoder`): three 3×3 stride-2 conv2d over the
(mel, time) plane with SiLU (the second and third depthwise, each followed by a
1×1), padding masked between them, 128×256/8 channels flattened and projected
to 768 by a 1×1 conv1d: eight frames become one vector. Then an LFM2 backbone
of 17 layers (conv, conv, attention, repeated, three attention layers at the
end), 8 query and 4 KV heads, RoPE θ=1e6, short convolution window 3, with a
causal sliding-window mask of 32 on the attention layers, and a final norm.

**Audio projector** (`Lfm3OmniProjector`): LayerNorm, 768→2048, GELU,
2048→2048, all with bias.

**Prompt assembly**: image and audio features replace placeholder tokens in the
text sequence (`<|image|>` 125030, `<|audio|>` 125031, `<|audio_start|>`
125012, `<|audio_out|>` 125032, mixed start 125015, text end 125014). The text
forward therefore has to accept embedding rows, not only token ids.

**Speech out** (`Lfm3OmniDepthDecoder`, codec): per audio frame the depth
decoder (LFM3, 3 layers, width 1024, 32 query and 8 KV heads, head 32) predicts
12 codebooks of 2017 codes under a MaskGIT schedule of 4, 4, 4 steps, and the
codec turns the codes into a waveform. Text and audio interleave 6:12.

## Order of work

Omni comes first. Every step has its fixture before its C and its gate before
the next step.

1. **Interface.** A way for a family to take embedding rows for some positions
   (a `forward_embd` hook, or token id −1 plus a row table). This changes
   `harness/arch.h`, and every family that does not implement it must keep
   refusing. Decided with the maintainer before it is written.
2. **Vision tower + projector** over the mmproj GGUF, F32 first. Gate: per-layer
   outputs against the upstream PyTorch module on a fixed 64×64 and a fixed
   512×384 image (relative L2 ≤ 2e-4 per layer), projector output against
   llama.cpp's mtmd embedding of the same image, then greedy answers to image
   prompts against `llama-mtmd-cli` on the same files.
3. **Audio front end + encoder + projector.** Gate: mel frames against the
   upstream feature extractor on a fixed 1 s 440 Hz tone and a fixed spoken
   clip, per-layer encoder outputs, projector rows against mtmd, transcription
   prompts greedy against `llama-mtmd-cli`.
4. **Quantized towers** (Q8_0, Q4_K_M mmproj) through the existing packed
   matvec, gated against the F32 path of step 2/3 and against mtmd at the same
   quantization.
5. **Speech out**: depth decoder, MaskGIT schedule, codec. Gate: codes against
   the upstream model for a fixed seed, then waveform against the codec.
6. **Speed**, prefill and decode separately, image and audio encode separately,
   against fresh llama.cpp on the same files, on x86 and on Apple silicon, each
   verdict OK / FAILED / SKIPPED with its reason.

## Fixtures

Two kinds, as for Qwen3.5 (`tests/qwen35_fixture_gen.py`):

- **Synthetic, in `make test`.** Tiny configurations of the upstream modules
  (two vision layers, one of each operator, two frames of different shapes; a
  two-layer audio encoder; a one-layer depth decoder), weights from a fixed
  arithmetic pattern over the module's parameter order, outputs from the
  upstream PyTorch code, frozen as C headers. Gated by mutation: a wrong merge
  order, a swapped RoPE axis or a causal vision mask must turn them red.
- **Real, outside the repository.** Per-layer dumps of the official checkpoint
  on fixed inputs, with SHA-256 of every input and output recorded here.

The upstream modeling code needs a newer Transformers than the one these
fixtures were first attempted with: 5.8.0 has no
`masking_utils.create_recurrent_attention_mask`, which `modeling_lfm3_omni.py`
imports. Until a newer one is installed, the reference is llama.cpp's mtmd
on the official GGUF pair, end to end.

## End-to-end references recorded, 2026-11-09

llama.cpp `7d7f028` `llama-mtmd-cli`, CPU, six threads of an i5-8500T,
greedy (`--temp 0 --top-k 1`), on the official GGUF files and on llama.cpp's own
mtmd test inputs. A NoTorch tower is not done until it reproduces these answers
token for token on the same files.

| File | SHA-256 |
| --- | --- |
| `LFM3-Omni-2B-Q8_0.gguf` (GGUF repo `d508b88f`) | `59118f661eaacb93d204723d76355b53347f0940b508c11412f6290fd49158d4` |
| `mmproj-LFM3-Omni-2B-F16.gguf` | `361bc95192c7d9156b566fcd4e48930363b78218e493fb1be607456bd51586e3` |
| `LFM3-VL-2B-Q8_0.gguf` (GGUF repo `1add221a`) | `e138e54189a751d90ea8c0b236360490613ecd465d28d68fa6094d3db7fa6727` |
| `mmproj-LFM3-VL-2B-F16.gguf` | `1d8e6e58eefc093e09d6332f081516e1ddbd5b35cfca7f35463fb270db230b1e` |
| `tools/mtmd/tests/test-1-positive.png` | `02ef128ec125a13f4ebca550fab584d189a794dcc597595b93da3d527593f343` |
| `tools/mtmd/test-2.mp3` | `cdeac0ded280e18b99afbb7fac86130e4dda4b7d0b252cc22aa3d20679270a1e` |

| Case | Encode | Answer (start) | Output SHA-256 |
| --- | ---: | --- | --- |
| Omni, image, "Describe this image in one sentence.", 64 tokens | 1699 ms | `<think></think>This image is a vertical newspaper article headlined "A Powdery Surface Is Closely Explored," by John Noble Wilford…` | `e5757af435e6c698dee0116ceaa80b3a1f875be2f4fc174d4c09bc4bb656f053` |
| Omni, audio, "Transcribe the audio.", 96 tokens | 278 ms | `<think></think>The New York Times from July 21st, 1969. This isn't just newsprint and ink…` | `23f78a82b3c6900c530edb17d2b22ea9df4869bbf6db20135b9c6719e73a4d9a` |
| VL-2B, image, same prompt, 64 tokens | 1703 ms | `The image is a scanned page from The New York Times, featuring an article titled "A Powdery Surface Is Closely Explored"…` | `b9de3dbd8746549f6f284e4ecd8e3cd8528d299f52955ad167f5af3c3bb1f645` |
