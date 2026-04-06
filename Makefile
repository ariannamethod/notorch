# notorch — Makefile
# "fuck torch"

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

# Detect platform
UNAME := $(shell uname)

# ── macOS: Apple Accelerate (AMX/Neural Engine) ──
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif

# ── Linux: OpenBLAS ──
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  BLAS_NAME = OpenBLAS
endif

# ── Targets ──

.PHONY: all test clean cpu gpu help

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

# Static library
lib: notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c notorch.c -o notorch.o
	ar rcs libnotorch.a notorch.o
	@echo "Built: libnotorch.a"

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

# Dubrovsky training (GQA + RoPE)
train_dubrovsky: examples/train_dubrovsky.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_dubrovsky examples/train_dubrovsky.c notorch.c -lm
	@echo "Compiled: train_dubrovsky (Dubrovsky GQA+RoPE, $(BLAS_NAME))"

# ── Test & Clean ──

test: notorch_test
	./notorch_test

clean:
	rm -f notorch_test notorch_test_gpu notorch.o libnotorch.a notorch_cuda.o \
		infer_janus_nt infer_gemma infer_llama train_q train_yent chat_yent test_gguf

help:
	@echo "notorch — neural networks in pure C"
	@echo ""
	@echo "  make            Build and run tests with BLAS"
	@echo "  make cpu        Build tests without BLAS (portable)"
	@echo "  make gpu        Build tests with CUDA"
	@echo "  make lib        Build static library (libnotorch.a)"
	@echo "  make infer      Build Janus RRPRAM inference"
	@echo "  make gemma      Build Gemma-3 GGUF inference"
	@echo "  make llama      Build LLaMA/Qwen/SmolLM2 inference"
	@echo "  make train_q    Build PostGPT-Q training"
	@echo "  make train_yent Build Yent 9.8M training"
	@echo "  make test       Build and run tests"
	@echo "  make clean      Remove build artifacts"
