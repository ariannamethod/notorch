# notorch — Makefile
# "fuck torch"

CC = cc
AR ?= ar
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

# Detect platform
UNAME := $(shell uname)

# ── macOS: Apple Accelerate (AMX/Neural Engine) ──
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif

# ── Linux: OpenBLAS ──
# Prefer pkg-config when available — handles distros that ship cblas.h in
# a subdir (Termux: /usr/include/openblas/) and custom $PREFIX layouts.
# Fallback to bare -lopenblas for minimal environments.
ifeq ($(UNAME), Linux)
  ifneq ($(shell command -v pkg-config 2>/dev/null),)
    BLAS_FLAGS = -DUSE_BLAS $(shell pkg-config --cflags openblas 2>/dev/null) $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
  else
    BLAS_FLAGS = -DUSE_BLAS -lopenblas
  endif
  BLAS_NAME = OpenBLAS
endif

# ── Targets ──

.PHONY: all test clean cpu gpu help lib install

all: notorch_test
	@echo "Built with $(BLAS_NAME). Run: ./notorch_test"

# CPU with BLAS
notorch_test: notorch.c notorch.h tests/test_notorch.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o notorch_test tests/test_notorch.c notorch.c -lm
	@echo "Compiled: notorch_test (CPU + $(BLAS_NAME))"

# CPU without BLAS (portable fallback)
cpu: notorch.c notorch.h tests/test_notorch.c
	$(CC) $(CFLAGS) -o notorch_test tests/test_notorch.c notorch.c -lm
	@echo "Compiled: notorch_test (CPU, no BLAS)"

# GPU (CUDA)
gpu: notorch.c notorch.h notorch_cuda.cu tests/test_notorch.c
	nvcc -O2 -DUSE_CUDA -c notorch_cuda.cu -o notorch_cuda.o
	$(CC) $(CFLAGS) -DUSE_CUDA -DUSE_BLAS -o notorch_test_gpu \
		tests/test_notorch.c notorch.c notorch_cuda.o \
		-L/usr/local/cuda/lib64 -lcudart -lcublas -lm
	@echo "Compiled: notorch_test_gpu (CUDA + BLAS)"

# Static library — bundles notorch.o + gguf.o so a single -lnotorch linkage
# satisfies both tensor ops (nt_blas_mmT, nt_bpe_*) and GGUF reader
# (gguf_open, gguf_dequant, gguf_get_kv).
lib: libnotorch.a

libnotorch.a: notorch.c notorch.h gguf.c gguf.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c notorch.c -o notorch.o
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c gguf.c -o gguf.o
	$(AR) rcs libnotorch.a notorch.o gguf.o
	@echo "Built: libnotorch.a (notorch + gguf)"

# ── Install — system-wide baseline at $PREFIX (default /opt/homebrew) ──
PREFIX ?= /opt/homebrew

install: libnotorch.a
	install -d $(PREFIX)/lib $(PREFIX)/include/ariannamethod
	install -m 0644 libnotorch.a $(PREFIX)/lib/libnotorch.a
	install -m 0644 notorch.h    $(PREFIX)/include/ariannamethod/notorch.h
	install -m 0644 gguf.h       $(PREFIX)/include/ariannamethod/gguf.h
	@echo "Installed: $(PREFIX)/lib/libnotorch.a + $(PREFIX)/include/ariannamethod/{notorch,gguf}.h"

# ── Inference ──

infer: examples/infer_janus.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_nt examples/infer_janus.c notorch.c -lm
	@echo "Compiled: infer_janus_nt (Janus RRPRAM, $(BLAS_NAME))"

gemma: examples/infer_gemma.c gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_gemma examples/infer_gemma.c gguf.c notorch.c -lm
	@echo "Compiled: infer_gemma (Gemma-3 GGUF, $(BLAS_NAME))"

llama: examples/infer_llama.c gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama examples/infer_llama.c gguf.c notorch.c -lm
	@echo "Compiled: infer_llama (LLaMA/Qwen GGUF, $(BLAS_NAME))"

# ── Training ──

train_q: examples/train_q.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_q examples/train_q.c notorch.c -lm
	@echo "Compiled: train_q (PostGPT-Q 1.65M, $(BLAS_NAME))"

train_yent: examples/train_yent.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_yent examples/train_yent.c notorch.c -lm
	@echo "Compiled: train_yent (Yent 9.8M, $(BLAS_NAME))"

# LLaMA 3 BPE training (MHA + RoPE + SwiGLU, 15.7M params, vocab 2048)
train_llama3_bpe: examples/train_llama3_bpe.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_llama3_bpe examples/train_llama3_bpe.c notorch.c -lm
	@echo "Compiled: train_llama3_bpe (MHA + RoPE + BPE 2048, 15.7M, $(BLAS_NAME))"

# LLaMA 3 BPE inference (interactive chat, KV cache, optional FP16 weights)
infer_llama3_bpe: examples/infer_llama3_bpe.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama3_bpe examples/infer_llama3_bpe.c notorch.c -lm
	@echo "Compiled: infer_llama3_bpe (MHA + RoPE + BPE 2048, $(BLAS_NAME))"

# LLaMA 3 char-level training (GQA + RoPE + SwiGLU, ~9.5M params)
train_llama3_char: examples/train_llama3_char.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_llama3_char examples/train_llama3_char.c notorch.c -lm
	@echo "Compiled: train_llama3_char (GQA + RoPE, $(BLAS_NAME))"

# DPO — Direct Preference Optimization (Rafailov 2023)
train_dpo: examples/train_dpo.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_dpo examples/train_dpo.c notorch.c -lm
	@echo "Compiled: train_dpo (DPO alignment, $(BLAS_NAME))"

# GRPO — Group Relative Policy Optimization (DeepSeek-R1)
train_grpo: examples/train_grpo.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_grpo examples/train_grpo.c notorch.c -lm
	@echo "Compiled: train_grpo (GRPO self-play RL, $(BLAS_NAME))"

# Knowledge Distillation (Hinton 2015)
train_distillation: examples/train_distillation.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_distillation examples/train_distillation.c notorch.c -lm
	@echo "Compiled: train_distillation (teacher→student KL, $(BLAS_NAME))"

# Vision + BPE tests
test_vision: tests/test_vision.c notorch.c notorch.h notorch_vision.h stb_image.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o test_vision tests/test_vision.c notorch.c -lm
	@echo "Compiled: test_vision (vision + BPE, $(BLAS_NAME))"

# ── Test & Clean ──

test: notorch_test test_vision
	./notorch_test
	./test_vision

clean:
	rm -f notorch_test notorch_test_gpu notorch.o gguf.o libnotorch.a notorch_cuda.o \
		infer_janus_nt infer_gemma infer_llama \
		train_q train_yent train_llama3_bpe train_llama3_char infer_llama3_bpe \
		train_dpo train_grpo train_distillation test_vision test_gguf

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
