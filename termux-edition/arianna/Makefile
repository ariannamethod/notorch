# Makefile — infer_arianna_char build helper
#
# Two targets: with BLAS (recommended, ~8x faster on aarch64) and without.
# Both produce a single-file static binary linked against ../../notorch.c.

CC      ?= cc
CFLAGS  ?= -O2 -Wall
NOTORCH := ../../notorch.c
INCLUDE := -I../..

# Detect platform / BLAS layout
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
endif

ifeq ($(UNAME), Linux)
  ifneq ($(shell command -v pkg-config 2>/dev/null),)
    BLAS_FLAGS = -DUSE_BLAS $(shell pkg-config --cflags openblas 2>/dev/null) $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
  else
    BLAS_FLAGS = -DUSE_BLAS -lopenblas
  endif
endif

.PHONY: all blas cpu clean run

all: blas

blas: infer_arianna_char.c $(NOTORCH)
	$(CC) $(CFLAGS) $(BLAS_FLAGS) $(INCLUDE) infer_arianna_char.c $(NOTORCH) -lm -o infer_arianna_char
	@echo "Built: infer_arianna_char (BLAS on)"

cpu: infer_arianna_char.c $(NOTORCH)
	$(CC) $(CFLAGS) $(INCLUDE) infer_arianna_char.c $(NOTORCH) -lm -o infer_arianna_char
	@echo "Built: infer_arianna_char (scalar, no BLAS)"

run: infer_arianna_char arianna_10k_char.bin
	./infer_arianna_char arianna_10k_char.bin

clean:
	rm -f infer_arianna_char
