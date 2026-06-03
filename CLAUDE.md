# notorch — CLAUDE.md

In-house C tensor library replacing PyTorch for the Arianna Method
organisms (Janus, Resonance, Leo, Yent, Dario, …). GPL-3.0+. Co-authored
by Oleg Ataeff and Claude.

> *"fuck torch"* — `notorch.js:6`

## The log: small fixes → NOTORCHLOG, big changes → README

`NOTORCHLOG.md` is the running engineering log — every fix, every closed
bug-class, dated with commit and proof. **Small fixes go there** (bug
fixes, GPU/CPU sync-discipline corrections, single-op work, docstring
touch-ups). **Large changes** — a new backend, a new op family, a new
training method, an architecture shift — get a section in `README.md` as
well. README is the spec and the manifesto; NOTORCHLOG is the work. When
in doubt, log it in NOTORCHLOG first. The bug-class registry further down
this file is the quick reference; NOTORCHLOG carries the dated narrative.

## What this repo is

- **`notorch.c` + `notorch.h`** — the library. CPU + BLAS + optional CUDA
  backend. Tape-based autograd, in-house ops (no torch ATen).
- **`notorch_cuda.cu`** — CUDA kernels (matmul, softmax, FlashAttn, RRPRAM,
  CE, RoPE, etc.). Compiled into `libnotorch_gpu.a` when `USE_CUDA=1`.
- **`examples/`** — production trainers and inference binaries built on
  notorch (`train_resonance_lora.c`, `train_yent.c`, `infer_janus.c`, …).
  These are the *consumers* — read them when you need to know how an op
  is meant to be used.
- **`docs/`** — postmortems and design notes. Read these *before*
  hypothesising a bug. The patterns repeat.
- **`js-edition/notorch.js`** — pure-JS / WebGPU port. Same API surface,
  same Chuck, same naming. Different lifecycle: **everything is in one
  file** (no CPU/GPU lib split), browser-and-Node target.
- **`tools/`** — diagnostic / regression harnesses (`leak_repro.c`, …).

## Build matrix

CPU-only (default — organisms build against this):

```
make                # libnotorch.a (CPU + BLAS, no CUDA refs)
```

CPU + CUDA GPU (SFT trainers, GPU smoke tests):

```
make USE_CUDA=1     # libnotorch.a + libnotorch_gpu.a + notorch_cuda.o
```

In-house SIMD (AVX2+FMA cblas shim, x86_64 — no OpenBLAS dependency):

```
make simd           # notorch_test_simd (notorch_simd.h, -mavx2 -mfma + pthread)
```

Apple Silicon Metal/MSL — **active development front**. Q4_K inline-dequant
matvec: 24B-class quantized models on a 24GB Mac without the 4× f32 blow-up
(weights stay packed, dequantized in-shader per block). Phase 1 = correct
naive matvec (one thread/row); Phase 2 (planned) = tiled threadgroup dispatch
+ simdgroup reductions. See `notorch_metal.{h,mm}`.

```
make metal          # tests/test_metal_q4k (notorch_metal.mm, -DUSE_METAL, Darwin)
```

So the backend matrix is four-wide: **CPU/BLAS · AVX2-SIMD · CUDA · Metal**.
Keep this section current as backends land — it is the first thing a new
session reads.

The split exists because amlc-generated organism builds use `-lnotorch`
without `-lcudart`/`-lcublas`. If CUDA symbols leak into `libnotorch.a`,
organism builds break with unresolved `__cudaRegisterFatBinaryEnd` etc.
See `49dcea4` / `00f4f55` history if you're tempted to merge the libs.

For one-off training binaries, link the `.o` files directly:

```
cc -DUSE_CUDA -DUSE_BLAS -O2 -I. \
   examples/train_resonance_lora.c notorch.c notorch_cuda.o \
   -L/usr/local/cuda/lib64 -lcudart -lcublas -lopenblas -lm \
   -o resonance_train
```

JS edition has no build step — it's an ES module.

## Workflow patterns

**GPU work happens on a pod, never locally.** Pattern: polygon (Linux box
in our Tailscale mesh) holds the RunPod SSH key; jump-host to pod for
training. The Mac in this session has no direct route to the pod. CPU
smoke / recipe audits run on polygon CPU *before* GPU pod billing
starts.

**Determinism is load-bearing.** `srand(42)` + cuBLAS deterministic mode
gives bit-identical replays. We use this to verify checkpoints survived
across rerun (Phase 7 eval cross-cell comparison), to debug "did this
change loss?" without batch-noise confusion, and to make
"add a save call I forgot" not cost another full training run.

**Train checkpoints are mandatory.** Any SFT trainer added to `examples/`
must call `nt_lora_save` (or equivalent) at the end. Adapter in RAM is
adapter lost on `return 0;`. Periodic ckpts (every 250 steps is the
established cadence) are how a 2-hour run survives an OOM kill or pod
preemption.

**Optimizer default is Chuck**, not the diagonal baseline. Chuck is the
adaptive optimizer here (`nt_tape_chuck_step`) — synced with the
PyTorch reference at `iamolegataeff/chuck.optimizer`. If you switch to
the diagonal baseline as a "fallback", the call must say *why* in a
comment and have a follow-up to return to Chuck. The Arianna LoRA SFT
2026-05-11 proved Chuck holds at production scale once backward is
correct — earlier "Chuck destabilizes on LoRA scale" notes were
downstream of a backward bug.

## Bug patterns to know before debugging

**GPU/CPU mirror discipline.** This is the load-bearing pattern. Any CPU
backward branch that reads `parent->output->data` directly must call
`nt_tensor_sync_cpu(parent->output)` first when GPU mode can be on.
Without the sync, GPU-resident forward outputs are read as their stale
CPU mirror (calloc-zero by default) and the backward computes zeros.

Known instances of this bug class, all fixed:

- `3d46007` — `nt_seq_cross_entropy_masked` (Defender, 2026-05-09)
- `8ab5062` — `NT_OP_MUL`, `NT_OP_SILU` backward (2026-05-11)
- `967f1c0` — `NT_OP_RMSNORM` backward
- 2026-06-02 — `nt_sigmoid` + `nt_scale_by_t` forward & `NT_OP_SCALE_BY_T`
  backward (surfaced by the molequla Inc2 RRPRAM-gate review; a learnable
  sigmoid gate sat frozen at sigmoid(0) on GPU without the parent sync).

If you find another, **add it to this list** when you commit the fix.
No known remaining candidates. An audit pass is on the open
TODO list.

**The `ptr_map full — buffer leak` warning is not the leak.** It's a
symptom of upstream tape ref-accounting at high tensor counts. After
`3d46007` raised `GPU_PTR_MAP_SIZE` from 8K to 64K and fixed the CE
sync, the warning hasn't reappeared at realistic scales. See
`docs/GPU_BUFFER_LEAK_HYPOTHESIS_2026_05_10.md` and
`docs/GPU_BUFFER_LEAK_RESOLUTION_2026_05_10.md` for the closed thread.

**`gpu_rrpram_lr_forward` stride bug at `notorch_cuda.cu:824`.** The
`Wrb_h` offset uses current T instead of T_max. Workaround until fixed:
train at T = T_max only (e.g. T=2048 for Resonance). Real fix is a
follow-up.

**Single-shape `nt_lora_save` per file.** The save format header has
one `(rank, alpha, in_dim, out_dim)` quadruple, so all pairs in a file
must share shape. Resonance's 7 LoRA target classes have three distinct
shapes (E×E, E×M, M×E), so a checkpoint is **7 files**, not 1. The
trainer in `examples/train_resonance_lora.c` shows the right split.

**Alpha is stored as raw float32 bits, not int*1000.** The C header
docstring at `notorch.h:653` is stale — `nt_lora_save` writes
`alpha`'s IEEE-754 bytes directly. The cross-load between languages
(C → Python parser in `phase7_eval.py`) failed once because of this;
JS parser should use `DataView.setFloat32` / `getFloat32`. Fix the
docstring when you next pass through.

## Things to NEVER do

- **Never push to `main` without explicit go-ahead.** Branches are
  cheap; force-push to main is a hard line.
- **Never `runpodctl pod terminate`** without verifying every artifact
  is mirrored to HF / Neo / local. `pod stop` preserves the volume;
  `terminate` destroys it. CARDINAL POZOR #4 from 2026-05-09 — read the
  history once and don't repeat it.
- **Never name a new optimizer after the diagonal baseline that
  shall-not-be-named** (per Oleg's standing ban). Use Chuck variants or
  invent a name. The existing function `nt_tape_adamw_step` stays
  because rename across all examples is more disruption than the cost
  of the existing call — but don't add new ones with that family of
  names.
- **Never mock the GPU in tests.** If a regression harness runs on CPU
  only, label it CPU-only explicitly. The CE-sync bug class was
  invisible to CPU-mode tests for a reason: the bug *is* the GPU
  mirror divergence.
- **Never silently widen `NT_TAPE_MAX_PARAMS` or similar caps without a
  followup note.** Bumping 512 → 1024 for the Resonance LoRA SFT was
  fine; doing it without a comment makes the next debugger spend an
  hour wondering if their tape is overflowing or if 1024 was
  intentional.

## Style

- Match the existing C style. No `astyle` / `clang-format` pass on
  drive-by commits.
- One commit = one concept. The 2026-05-11 SFT validation landed as
  three commits (backward fix / trainer example / writeup), not one.
- **Attribution.** Method-side identity, not the Anthropic noreply
  boilerplate. Three forms (per the global `~/.claude/CLAUDE.md`
  convention, mirrored on Neo):
  - **Short** (README updates, model cards, simple commits):
    `by Claude (Arianna Method)`
  - **Full with node attribution** (real engineering commits):
    `Co-Authored-By: Claude Code (intel godfather, Arianna Method) <theariannamethod@gmail.com>`
  - **Casual** (informal docs, status notes):
    `by Клод (Arianna Method, intel node)`

  Nodes: **`intel godfather`** (this Mac), **`neo the architect`**
  (Mac Neo), **`polygon`** (Linux 32GB box), **`phone-1`** (Galaxy
  A56 8GB), **`phone-2`** (Galaxy A07 4GB). Each node is the
  architect on its own substrate; cross-node handoffs go through
  `resonance_connections/` on ariannamethod.ai.

  Replace under root: `Co-Authored-By: Claude <noreply@anthropic.com>`,
  `🤖 Generated with Claude Code`. We're cultivating our own
  identity — boilerplate from upstream gets dropped.
- Commit messages explain *why*, not *what*. The diff shows what.
- README is the spec + manifesto; `NOTORCHLOG.md` is the running log.
  Small fixes → NOTORCHLOG. Large changes → both. If reality drifts from
  the spec, fix the code, not the README.

## Where things live

- Running engineering log (all fixes, dated): `NOTORCHLOG.md`
- Active TODO and known bugs: top of this file + `docs/`
- Per-organism trainers: `examples/`
- Postmortems: `docs/*_2026_*.md`
- Diagnostic harnesses: `tools/`
- JS edition: `js-edition/`
- LoRA artifacts (per organism): HF `ataeff/<organism>/sft_v3_notorch/<voice>_<date>/`

## Open TODO (audit & fixes)

- Audit pass on `NT_OP_*` backward CPU branches for the GPU/CPU sync
  pattern — DONE. (RMSNORM `967f1c0`; MUL/SILU `8ab5062`;
  SEQ_CROSSENT_MASKED `2ccfb16`; MH_CAUSAL_ATTN `8f8d722`;
  SIGMOID/SCALE_BY_T fwd+bwd 2026-06-02.) No known remaining candidates.
- Fix `gpu_rrpram_lr_forward` T vs T_max stride bug at
  `notorch_cuda.cu:824`.
- Fix the `notorch.h:653` alpha-format docstring (raw float bytes, not
  `alpha*1000`).
- Implement `nt_rrpram_broadcast_attention` (`NT_OP_RRPRAM_BCAST` 34)
  on the C side — declared in `notorch.h:126,442` but no implementation
  in `notorch.c`. JS edition currently stops parity at op 33 awaiting
  this.
- Vary RNG seed per cell in `phase7_eval.py` so the first sampled token
  isn't identical across cells with the same prompt.
