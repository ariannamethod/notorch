# notorch — Makefile
# "fuck torch"

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11

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
notorch_test: notorch.c notorch.h test_notorch.c
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o notorch_test test_notorch.c notorch.c -lm
	@echo "Compiled: notorch_test (CPU + $(BLAS_NAME))"

# CPU without BLAS (portable fallback)
cpu: notorch.c notorch.h test_notorch.c
	$(CC) $(CFLAGS) -o notorch_test test_notorch.c notorch.c -lm
	@echo "Compiled: notorch_test (CPU, no BLAS)"

# GPU (CUDA)
gpu: notorch.c notorch.h notorch_cuda.cu test_notorch.c
	nvcc -O2 -DUSE_CUDA -c notorch_cuda.cu -o notorch_cuda.o
	$(CC) $(CFLAGS) -DUSE_CUDA -DUSE_BLAS -o notorch_test_gpu \
		test_notorch.c notorch.c notorch_cuda.o \
		-L/usr/local/cuda/lib64 -lcudart -lcublas -lm
	@echo "Compiled: notorch_test_gpu (CUDA + BLAS)"

# Static library
lib: notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -c notorch.c -o notorch.o
	ar rcs libnotorch.a notorch.o
	@echo "Built: libnotorch.a"

# Janus RRPRAM inference
infer: infer_janus.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_nt infer_janus.c notorch.c -lm
	@echo "Compiled: infer_janus_nt (Janus RRPRAM, $(BLAS_NAME))"

# Gemma-3 inference via GGUF
gemma: infer_gemma.c gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_gemma infer_gemma.c gguf.c notorch.c -lm
	@echo "Compiled: infer_gemma (Gemma-3 GGUF, $(BLAS_NAME))"

# LLaMA/Qwen/SmolLM2 inference via GGUF
llama: infer_llama.c gguf.c gguf.h notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_llama infer_llama.c gguf.c notorch.c -lm
	@echo "Compiled: infer_llama (LLaMA/Qwen GGUF, $(BLAS_NAME))"

test: notorch_test
	./notorch_test

clean:
	rm -f notorch_test notorch_test_gpu notorch.o libnotorch.a notorch_cuda.o infer_janus_nt

help:
	@echo "notorch — PyTorch replacement in C"
	@echo ""
	@echo "  make          Build tests with BLAS (Accelerate/OpenBLAS)"
	@echo "  make cpu      Build tests without BLAS (portable)"
	@echo "  make gpu      Build tests with CUDA"
	@echo "  make lib      Build static library"
	@echo "  make test     Build and run tests"
	@echo "  make clean    Remove build artifacts"
