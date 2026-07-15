# notorch — Makefile
# "fuck torch"

CC = cc
AR ?= ar
CFLAGS = -O2 -Wall -Wextra -std=c11 -pthread -I.

# Detect platform
UNAME := $(shell uname)

# ── macOS: Apple Accelerate (AMX/Neural Engine) ──
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_LIBS  =
  BLAS_NAME  = Accelerate
endif

# ── Linux: OpenBLAS ──
# Prefer pkg-config when available — handles distros that ship cblas.h in
# a subdir (Termux: /usr/include/openblas/) and custom $PREFIX layouts.
# Fallback to bare -lopenblas for minimal environments.
#
# Linux note: GNU ld requires -l<lib> AFTER the .o/.c that reference its
# symbols, so split compile flags from link flags. (-framework on Darwin
# works at any position, hence Mac keeps a single BLAS_FLAGS.)
ifeq ($(UNAME), Linux)
  ifneq ($(shell command -v pkg-config 2>/dev/null),)
    BLAS_FLAGS = -DUSE_BLAS $(shell pkg-config --cflags openblas 2>/dev/null)
    BLAS_LIBS  = $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
  else
    BLAS_FLAGS = -DUSE_BLAS
    BLAS_LIBS  = -lopenblas
  endif
  BLAS_NAME = OpenBLAS
endif

# ── In-house SIMD: AVX2 + FMA (no external BLAS) ──
# Hand-rolled cblas_* shim in notorch_simd.h. Pure C + intrinsics + pthread.
# Drop-in replacement on x86_64 Haswell+ (Intel 2013, AMD 2015). Mutually
# exclusive with USE_BLAS — the shim re-#defines USE_BLAS internally so the
# call sites in notorch.c keep working unchanged.
SIMD_FLAGS = -DUSE_SIMD -mavx2 -mfma
SIMD_LIBS  = -lpthread

# ── Targets ──

.PHONY: all test test_js clean cpu gpu simd help lib install metal test_metal infer_gguf_metal

all: notorch_test
	@echo "Built with $(BLAS_NAME). Run: ./notorch_test"

# CPU with BLAS
notorch_test: notorch.c notorch.h tests/test_notorch.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o notorch_test tests/test_notorch.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: notorch_test (CPU + $(BLAS_NAME))"

# CPU without BLAS (portable scalar fallback)
cpu: notorch.c notorch.h tests/test_notorch.c
	$(CC) $(CFLAGS) -o notorch_test tests/test_notorch.c notorch.c -lm
	@echo "Compiled: notorch_test (CPU, scalar — no BLAS, no SIMD)"

# In-house AVX2+FMA SIMD shim — zero external BLAS dependency
simd: notorch.c notorch.h notorch_simd.h tests/test_notorch.c
	$(CC) $(CFLAGS) $(SIMD_FLAGS) -o notorch_test_simd tests/test_notorch.c notorch.c -lm $(SIMD_LIBS)
	@echo "Compiled: notorch_test_simd (in-house AVX2+FMA, pthread)"

# GPU (CUDA)
gpu: notorch.c notorch.h notorch_cuda.cu tests/test_notorch.c
	nvcc -O2 -DUSE_CUDA -c notorch_cuda.cu -o notorch_cuda.o
	$(CC) $(CFLAGS) -DUSE_CUDA -DUSE_BLAS -o notorch_test_gpu \
		tests/test_notorch.c notorch.c notorch_cuda.o \
		-L/usr/local/cuda/lib64 -lcudart -lcublas -lm $(BLAS_LIBS)
	@echo "Compiled: notorch_test_gpu (CUDA + BLAS)"

# Static library — bundles notorch.o + gguf.o so a single -lnotorch linkage
# satisfies both tensor ops (nt_blas_mmT, nt_bpe_*) and GGUF reader
# (gguf_open, gguf_dequant, gguf_get_kv).
lib: libnotorch.a $(if $(filter 1,$(USE_CUDA)),libnotorch_gpu.a)

# CPU-only library — always built, organism binaries link this (no CUDA deps).
libnotorch.a: notorch.c notorch.h gguf.c gguf.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c notorch.c -o notorch.o
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c gguf.c -o gguf.o
	$(AR) rcs libnotorch.a notorch.o gguf.o
	@echo "Built: libnotorch.a (CPU + BLAS)"

# GPU-enabled library — only when USE_CUDA=1. SFT trainer links this +
# -lcudart -lcublas. Compiled with -DUSE_CUDA so #ifdef blocks activate.
libnotorch_gpu.a: notorch.c notorch.h gguf.c gguf.h notorch_cuda.cu notorch_cuda.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -DUSE_CUDA -I/usr/local/cuda/include -c notorch.c -o notorch_gpu.o
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c gguf.c -o gguf_gpu.o
	nvcc -O2 -DUSE_CUDA -c notorch_cuda.cu -o notorch_cuda.o
	$(AR) rcs libnotorch_gpu.a notorch_gpu.o gguf_gpu.o notorch_cuda.o
	@echo "Built: libnotorch_gpu.a (CPU + BLAS + CUDA)"

# ── Install — system-wide baseline at $PREFIX (default /opt/homebrew) ──
PREFIX ?= /opt/homebrew

install: lib
	install -d $(PREFIX)/lib $(PREFIX)/include/ariannamethod
	install -m 0644 libnotorch.a $(PREFIX)/lib/libnotorch.a
	install -m 0644 notorch.h    $(PREFIX)/include/ariannamethod/notorch.h
	install -m 0644 gguf.h       $(PREFIX)/include/ariannamethod/gguf.h
ifeq ($(USE_CUDA),1)
	install -m 0644 libnotorch_gpu.a $(PREFIX)/lib/libnotorch_gpu.a
	install -m 0644 notorch_cuda.h $(PREFIX)/include/ariannamethod/notorch_cuda.h
	@echo "Installed: $(PREFIX)/lib/{libnotorch.a, libnotorch_gpu.a} + headers"
else
	@echo "Installed: $(PREFIX)/lib/libnotorch.a + headers"
endif

# ── Inference ──

infer: examples/infer_janus.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_nt examples/infer_janus.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: infer_janus_nt (Janus RRPRAM, $(BLAS_NAME))"

gemma: examples/infer_gemma.c gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_gemma examples/infer_gemma.c gguf.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: infer_gemma (Gemma-3 GGUF, $(BLAS_NAME))"

llama: examples/infer_llama.c examples/bpe.c examples/bpe.h gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama examples/infer_llama.c examples/bpe.c gguf.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: infer_llama (LLaMA/Qwen GGUF + GGUF-BPE tokenizer, $(BLAS_NAME))"

# ── Training ──

train_q: examples/train_q.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_q examples/train_q.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_q (PostGPT-Q 1.65M, $(BLAS_NAME))"

train_yent: examples/train_yent.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_yent examples/train_yent.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_yent (Yent 9.8M, $(BLAS_NAME))"

# LLaMA 3 BPE training (MHA + RoPE + SwiGLU, 15.7M params, vocab 2048)
train_llama3_bpe: examples/train_llama3_bpe.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_llama3_bpe examples/train_llama3_bpe.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_llama3_bpe (MHA + RoPE + BPE 2048, 15.7M, $(BLAS_NAME))"

# LLaMA 3 BPE inference (interactive chat, KV cache, optional FP16 weights)
infer_llama3_bpe: examples/infer_llama3_bpe.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama3_bpe examples/infer_llama3_bpe.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: infer_llama3_bpe (MHA + RoPE + BPE 2048, $(BLAS_NAME))"

# LLaMA 3 char-level training (GQA + RoPE + SwiGLU, ~9.5M params)
train_llama3_char: examples/train_llama3_char.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_llama3_char examples/train_llama3_char.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_llama3_char (GQA + RoPE, $(BLAS_NAME))"

infer_llama3_char: examples/infer_llama3_char.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama3_char examples/infer_llama3_char.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: infer_llama3_char (char-level inference, $(BLAS_NAME))"

# DPO — Direct Preference Optimization (Rafailov 2023)
train_dpo: examples/train_dpo.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_dpo examples/train_dpo.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_dpo (DPO alignment, $(BLAS_NAME))"

# GRPO — Group Relative Policy Optimization (DeepSeek-R1)
train_grpo: examples/train_grpo.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_grpo examples/train_grpo.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_grpo (GRPO self-play RL, $(BLAS_NAME))"

# Knowledge Distillation (Hinton 2015)
train_distillation: examples/train_distillation.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_distillation examples/train_distillation.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: train_distillation (teacher→student KL, $(BLAS_NAME))"

# Vision + BPE tests
test_vision: tests/test_vision.c notorch.c notorch.h notorch_vision.h stb_image.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o test_vision tests/test_vision.c notorch.c -lm $(BLAS_LIBS)
	@echo "Compiled: test_vision (vision + BPE, $(BLAS_NAME))"

# ── Apple Silicon Metal/MSL backend (Phase 1: Q4_K matvec) ──
# Inline-dequant Q4_K matvec on the Apple GPU — the critical path for
# 24B-class quantized models on 24GB nodes. Weights stay in their packed
# layout; the shader streams blocks and reconstructs f32 values per
# matvec, never materializing the 4× f32 buffer. See notorch_metal.h.
ifeq ($(UNAME), Darwin)
# Obj-C++ needs a C++ standard; -std=c11 is invalid for .mm. Use a
# dedicated MM_FLAGS so the rest of notorch (pure C99/C11) keeps its
# C dialect.
MM_FLAGS = -O2 -Wall -Wextra -std=c++17 -I.

notorch_metal.o: notorch_metal.mm notorch_metal.h
	clang++ $(MM_FLAGS) -DUSE_METAL -fobjc-arc -c notorch_metal.mm -o notorch_metal.o
	@echo "Compiled: notorch_metal.o (Metal/MSL, Apple Silicon)"

tests/test_metal_q4k: tests/test_metal_q4k.c notorch_metal.o notorch_metal.h
	$(CC) $(CFLAGS) -DUSE_METAL -o tests/test_metal_q4k tests/test_metal_q4k.c notorch_metal.o \
		-framework Metal -framework Foundation -lc++ -lm
	@echo "Compiled: tests/test_metal_q4k (Metal Q4_K correctness)"

metal: tests/test_metal_q4k
	@echo "Metal backend built. Run: make test_metal"

test_metal: tests/test_metal_q4k
	./tests/test_metal_q4k

# Packed-Q4_K/Q6_K GGUF inference on Apple Metal — runs 24B-class quantized
# models on a 24GB Mac (weights stay packed; Q4_K on the GPU, Q6_K per-block
# on CPU cores; resident weights, no per-call upload). See examples/infer_gguf_metal.c.
infer_gguf_metal: examples/infer_gguf_metal.c examples/bpe.c examples/bpe.h gguf.c gguf.h notorch_metal.o notorch_metal.h
	$(CC) $(CFLAGS) -DUSE_METAL -I. -o examples/infer_gguf_metal \
		examples/infer_gguf_metal.c examples/bpe.c gguf.c notorch_metal.o \
		-framework Metal -framework Foundation -lc++ -lm
	@echo "Compiled: examples/infer_gguf_metal (packed Q4_K/Q6_K GGUF inference, Apple Metal)"
else
metal:
	@echo "Metal is Apple-only; this is $(UNAME). Skipping."
test_metal:
	@echo "Metal is Apple-only; skipping."
infer_gguf_metal:
	@echo "infer_gguf_metal needs Apple Metal; this is $(UNAME). Skipping."
endif

# ── Test & Clean ──

test: notorch_test test_vision
	./notorch_test
	./test_vision

test_js:
	node js-edition/test_op_parity.mjs

# SIMD correctness harness (vs scalar reference at nanollama shapes)
tests/test_simd_correctness: tests/test_simd_correctness.c notorch_simd.h
	$(CC) -O2 -mavx2 -mfma -DUSE_SIMD -I. -o tests/test_simd_correctness tests/test_simd_correctness.c -lm -lpthread

# SIMD end-to-end loss validation (lm_head shape)
tests/test_simd_loss: tests/test_simd_loss.c notorch_simd.h
	$(CC) -O2 -mavx2 -mfma -DUSE_SIMD -I. -o tests/test_simd_loss tests/test_simd_loss.c -lm -lpthread

# RRPRAM low-rank attention finite-difference grad check
tests/test_rrpram_lr: tests/test_rrpram_lr.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -I. -o tests/test_rrpram_lr tests/test_rrpram_lr.c notorch.c -lm $(BLAS_LIBS)

test_simd: tests/test_simd_correctness tests/test_simd_loss
	./tests/test_simd_correctness
	./tests/test_simd_loss

# SIMD vs OpenBLAS micro-benchmark at hot-path shapes
bench/bench_simd: bench/bench_simd.c notorch.c notorch.h notorch_simd.h
	$(CC) -O2 -mavx2 -mfma -DUSE_SIMD -I. -o bench/bench_simd bench/bench_simd.c notorch.c -lm -lpthread

bench/bench_blas: bench/bench_simd.c notorch.c notorch.h
	$(CC) -O2 -mavx2 -mfma $(BLAS_FLAGS) -I. -o bench/bench_blas bench/bench_simd.c notorch.c -lm $(BLAS_LIBS)

bench: bench/bench_simd bench/bench_blas

clean:
	rm -f notorch_test notorch_test_gpu notorch.o gguf.o libnotorch.a notorch_cuda.o \
		infer_janus_nt infer_gemma infer_llama \
		train_q train_yent train_llama3_bpe train_llama3_char infer_llama3_bpe \
		train_dpo train_grpo train_distillation test_vision test_gguf \
		tests/test_simd_correctness tests/test_simd_loss tests/test_rrpram_lr \
		bench/bench_simd bench/bench_blas

help:
	@echo "notorch — neural networks in pure C"
	@echo ""
	@echo "  make                    Build and run tests with BLAS"
	@echo "  make cpu                Build tests without BLAS (portable)"
	@echo "  make gpu                Build tests with CUDA"
	@echo "  make lib                Build static library (libnotorch.a)"
	@echo ""
	@echo "  inference:"
	@echo "    make infer            Janus RRPRAM inference"
	@echo "    make gemma            Gemma-3 GGUF inference"
	@echo "    make llama            LLaMA/Qwen/SmolLM2 GGUF inference"
	@echo "    make infer_llama3_bpe LLaMA 3 BPE chat (vocab 2048)"
	@echo ""
	@echo "  training:"
	@echo "    make train_q          PostGPT-Q 1.65M (char-level research)"
	@echo "    make train_yent       Yent 9.8M char-level"
	@echo "    make train_llama3_char LLaMA 3 char-level (GQA, ~9.5M)"
	@echo "    make train_llama3_bpe  LLaMA 3 BPE 2048 (MHA, ~15.7M)"
	@echo "    make train_dpo        DPO alignment"
	@echo "    make train_grpo       GRPO self-play RL"
	@echo "    make train_distillation  Knowledge distillation"
	@echo ""
	@echo "  make test               Build and run tests"
	@echo "  make clean              Remove build artifacts"
