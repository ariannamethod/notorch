#!/bin/sh
# test_janus.sh — the ported Janus forward against the one it was ported from,
# and against itself.
#
# Two assertions, and neither of them needs a tolerance invented to fit.
#
#   1. Built with JANUS_EXACT_MATVEC, the port must reproduce the reference's
#      eight largest logits to 1e-3. That is the forward, measured against the
#      engine in arianna.c that stays in arianna.c — what travels here is its
#      answer, in tests/janus_golden_v4.txt.
#
#   2. Built as it ships, the port prefers the integer matvec where that engine
#      calls the exact one. The question that matters is not how far the logits
#      move but whether the model says anything different, so the two builds
#      generate greedily from the same ids and the token sequences must match.
#
#   ./harness/test_janus.sh [model.gguf]
#
# Without a model it says so and exits 0 — the weights are not in this tree.
set -eu
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/arianna/weights/janus_arianna_full_resft_2026_07_09/janus_arianna_q8_0.gguf}"
GOLDEN=tests/janus_golden_v4.txt
IDS="84 104 101 32 102 105 101 108 100"
TOL=1e-3
STEPS=8

if [ ! -f "$MODEL" ]; then
  echo "janus  (no model at $MODEL — pass one to run this gate)"
  echo "JANUS_SKIPPED"
  exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

SRC="tests/test_janus_parity.c harness/arch_janus.c harness/runtime.c gguf.c notorch.c"
BASE="-O2 -std=gnu11 -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate"
# shellcheck disable=SC2086
${CC:-cc} $BASE -DJANUS_EXACT_MATVEC -o "$TMP/exact" $SRC -lm 2>/dev/null \
  || ${CC:-cc} -O2 -std=gnu11 -I. -DJANUS_EXACT_MATVEC -o "$TMP/exact" $SRC -lm
# shellcheck disable=SC2086
${CC:-cc} $BASE -o "$TMP/shipped" $SRC -lm 2>/dev/null \
  || ${CC:-cc} -O2 -std=gnu11 -I. -o "$TMP/shipped" $SRC -lm

# ── 1. the forward, against the reference's frozen answer ────────────────────
# shellcheck disable=SC2086
"$TMP/exact" "$MODEL" $IDS > "$TMP/exact.out" 2>/dev/null

grep -v '^#' "$GOLDEN" | grep -v '^[[:space:]]*$' | while read -r id want; do
  got=$(sed -n "$((id + 1))p" "$TMP/exact.out")
  echo "$id $want $got"
done > "$TMP/cmp"

awk -v tol="$TOL" '
  { d = $2 - $3; if (d < 0) d = -d
    printf "janus  [id %-6s] golden %-12s got %-12s |diff|=%.3e  %s\n",
           $1, $2, $3, d, (d <= tol ? "PASS" : "FAIL")
    if (d > tol) fails++
    if (d > worst) worst = d }
  END {
    printf "janus  [forward] worst %.3e against a %s tolerance  %s\n",
           worst, tol, (fails ? "FAIL" : "PASS")
    exit (fails ? 1 : 0)
  }' "$TMP/cmp" || { echo "JANUS_FAIL"; exit 1; }

# ── 2. the approximation, measured by what the model says ───────────────────
greedy() {
  bin=$1; seq=$IDS
  i=0
  while [ "$i" -lt "$STEPS" ]; do
    # shellcheck disable=SC2086
    a=$("$bin" "$MODEL" $seq 2>&1 >/dev/null | sed -n 's/.*argmax=\([0-9]*\).*/\1/p')
    [ -n "$a" ] || { echo "no argmax from $bin" >&2; return 1; }
    seq="$seq $a"
    i=$((i + 1))
  done
  echo "$seq" | cut -d' ' -f$(( $(echo $IDS | wc -w | tr -d ' ') + 1 ))-
}

GE=$(greedy "$TMP/exact")
GS=$(greedy "$TMP/shipped")
if [ "$GE" = "$GS" ]; then
  echo "janus  [generation] $STEPS greedy tokens identical on both matvec paths  PASS"
  echo "  $GS"
  echo "JANUS_OK"
else
  echo "janus  [generation] the integer matvec changed what the model says  FAIL"
  echo "  exact:   $GE"
  echo "  shipped: $GS"
  echo "JANUS_FAIL"
  exit 1
fi
