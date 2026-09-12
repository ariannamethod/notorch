#!/bin/sh
# test_repeat.sh — a model must answer the same question the same way twice.
#
# Families keep state the KV cache does not: Mamba's convolution and SSM state,
# Janus's running low-rank sum. It lives in the model object, so a fresh cache
# does not clear it and only the family can, when it sees pos0 = 0. Janus did
# not, and drifted 0.49 on the logits between two identical runs while keeping
# the same argmax — every gate in the tree stayed green.
#
#   ./harness/test_repeat.sh [model.gguf ...]
#
# Exact equality, not a tolerance. The same arithmetic on the same inputs has no
# reason to move, and a tolerance here is where the next drift would hide.
set -eu
cd "$(dirname "$0")/.."

MODELS="$*"
if [ -z "$MODELS" ]; then
  for m in "$HOME/arianna/weights/nano_arianna_full_resft_2026_07_09/nano_arianna_q8_0.gguf" \
           "$HOME/arianna/weights/janus_arianna_full_resft_2026_07_09/janus_arianna_q8_0.gguf" \
           "$HOME/arianna-shared/arianna.c/weights/arianna_resonance_v3_f16.gguf"; do
    [ -f "$m" ] && MODELS="$MODELS $m"
  done
fi
if [ -z "$MODELS" ]; then
  echo "repeat  (no model given and no default on this machine — pass a .gguf)"
  echo "NOTORCH_REPEAT_SKIPPED"
  exit 0
fi

BIN=$(mktemp -t nt_repeat)
trap 'rm -f "$BIN"' EXIT
SRC="tests/test_repeat.c harness/archs.c harness/runtime.c harness/arch_llama.c \
     harness/arch_gemma4.c harness/arch_olmoe.c harness/arch_mamba.c \
     harness/arch_resonance.c harness/arch_janus.c examples/bpe.c gguf.c notorch.c"
# shellcheck disable=SC2086
${CC:-cc} -O2 -std=gnu11 -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
  -framework Accelerate -o "$BIN" $SRC -lm 2>/dev/null \
  || ${CC:-cc} -O2 -std=gnu11 -I. -o "$BIN" $SRC -lm

# Exit 1 means the probe ran and the model drifted; 2 and 3 mean it never got
# far enough to measure anything. Collapsing those was this script's own first
# defect: the red-hand run that put the drift back reported SKIPPED, which is
# the same lie test_parity.sh was carrying this morning, written fresh.
FAILS=0
CHECKS=0
for M in $MODELS; do
  NAME=$(basename "$M")
  set +e
  OUT=$("$BIN" "$M" 2>/dev/null)
  RC=$?
  set -e
  if [ "$RC" -gt 1 ]; then
    WHY=$("$BIN" "$M" 2>&1 >/dev/null | grep -m1 -iE 'error|cannot|missing|refus|no family|too small' || echo "it exited $RC with nothing to say")
    echo "repeat  [$NAME] SKIPPED — $WHY"
    continue
  fi
  CHECKS=$((CHECKS + 1))
  echo "repeat  [$NAME] $OUT"
  [ "$RC" -eq 0 ] || FAILS=$((FAILS + 1))
done

if [ "$FAILS" -ne 0 ]; then
  echo "NOTORCH_REPEAT_FAIL ($FAILS)"
  exit 1
elif [ "$CHECKS" -eq 0 ]; then
  echo "repeat  (every model given was skipped — nothing was compared)"
  echo "NOTORCH_REPEAT_SKIPPED"
else
  echo "NOTORCH_REPEAT_OK ($CHECKS checks)"
fi
