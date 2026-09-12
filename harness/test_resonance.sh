#!/bin/sh
# test_resonance.sh — the ported Resonance forward against the one it was
# ported from.
#
# The reference is arianna.c's tools/resonance_forward.h, which lives in another
# repository and stays there; what travels here is its answer.
# tests/resonance_golden_v3.txt holds the eight largest logits of the last
# position for a fixed prompt, and this recomputes them.
#
#   ./harness/test_resonance.sh [model.gguf]
#
# Without a model it says so and exits 0 — the weights are not in this tree.
set -eu
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/arianna-shared/arianna.c/weights/arianna_resonance_v3_f16.gguf}"
GOLDEN=tests/resonance_golden_v3.txt
IDS="487 1955 306"
TOL=1e-3

if [ ! -f "$MODEL" ]; then
  echo "resonance  (no model at $MODEL — pass one to run this gate)"
  echo "RESONANCE_SKIPPED"
  exit 0
fi

BIN=$(mktemp "${TMPDIR:-/tmp}/nt_res_parity.XXXXXX")
trap 'rm -f "$BIN" "$BIN.out"' EXIT
${CC:-cc} -O2 -std=gnu11 -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
  -framework Accelerate -o "$BIN" \
  tests/test_resonance_parity.c harness/arch_resonance.c harness/runtime.c \
  gguf.c notorch.c -lm 2>/dev/null \
  || ${CC:-cc} -O2 -std=gnu11 -I. -DUSE_BLAS -o "$BIN" \
       tests/test_resonance_parity.c harness/arch_resonance.c harness/runtime.c \
       gguf.c notorch.c -lopenblas -lm 2>/dev/null \
  || ${CC:-cc} -O2 -std=gnu11 -I. -o "$BIN" \
       tests/test_resonance_parity.c harness/arch_resonance.c harness/runtime.c \
       gguf.c notorch.c -lm

# shellcheck disable=SC2086
"$BIN" "$MODEL" $IDS > "$BIN.out" 2>/dev/null

grep -v '^#' "$GOLDEN" | grep -v '^[[:space:]]*$' | while read -r id want; do
  got=$(sed -n "$((id + 1))p" "$BIN.out")
  echo "$id $want $got"
done > "$BIN.cmp"

awk -v tol="$TOL" '
  { d = $2 - $3; if (d < 0) d = -d
    printf "resonance  [id %-5s] golden %-12s got %-12s |diff|=%.3e  %s\n",
           $1, $2, $3, d, (d <= tol ? "PASS" : "FAIL")
    if (d > tol) fails++
    if (d > worst) worst = d }
  END {
    printf "resonance  [worst] %.3e against a %s tolerance\n", worst, tol
    print (fails ? "RESONANCE_FAIL" : "RESONANCE_OK")
    exit (fails ? 1 : 0)
  }' "$BIN.cmp"
