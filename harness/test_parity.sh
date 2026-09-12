#!/bin/sh
# test_parity.sh — the harness must say exactly what the example says.
#
# examples/infer_llama.c stays in the tree as the reference: it is what the
# phone numbers were measured with and what Defender runs models through. This
# moved its forward into harness/ and claimed the arithmetic came along
# unchanged. At temp 0 that claim is checkable to the byte.
#
#   ./harness/test_parity.sh [model.gguf ...]
#
# With no argument it uses whichever of its default models are on this machine
# and says so when none are. Needs ./notorch and ./infer_llama built.
set -eu
cd "$(dirname "$0")/.."

[ -x ./notorch ] || { echo "test_parity: ./notorch not built (make harness)"; exit 1; }
[ -x ./infer_llama ] || { echo "test_parity: ./infer_llama not built (make llama)"; exit 1; }

MODELS="$*"
if [ -z "$MODELS" ]; then
  for m in "$HOME/arianna/weights/nano_arianna_full_resft_2026_07_09/nano_arianna_q8_0.gguf" \
           "$HOME/arianna/weights/nano_arianna_full_resft_2026_07_09/nano_arianna_q4_k_m.gguf"; do
    [ -f "$m" ] && MODELS="$MODELS $m"
  done
fi
if [ -z "$MODELS" ]; then
  echo "parity  (no model given and no default on this machine — pass a .gguf)"
  echo "NOTORCH_PARITY_SKIPPED"
  exit 0
fi

# The reference prints its diagnostics on stdout with the text. Take what sits
# between the prompt line and the timing rule, minus the blank line the rule
# brings with it. The harness needs no such surgery: its stdout is the text.
ref_text() {
  awk '
    /^prompt: "/  { on = 1; next }
    on && /^── prefill:/ { exit }
    on           { buf[n++] = $0 }
    END          { for (i = 0; i < n - 1; i++) print buf[i] }
  '
}

# A comparison needs both sides to have run. The reference loads only the
# standard families, so on resonance or janus it prints nothing and every
# prompt reads as a mismatch — three FAILs that say nothing about the harness.
# Ask each binary to load the model before comparing anything, and when one
# cannot, say so and move on: neither green nor red.
can_load() {
  "$1" "$2" "x" 1 0 >/dev/null 2>&1
}
# The first line of stderr is the shape banner, so the reason is the first line
# that reads like one, and failing that the last thing said before it gave up.
why_not() {
  OUT=$("$1" "$2" "x" 1 0 2>&1 >/dev/null) || true
  R=$(printf '%s\n' "$OUT" | grep -m1 -iE 'error|cannot|missing|unsupported|unknown|failed|truncat|short') || true
  [ -n "$R" ] || R=$(printf '%s\n' "$OUT" | grep . | tail -1) || true
  [ -n "$R" ] || R="it exited non-zero with nothing to say"
  printf '%s\n' "$R"
}

FAILS=0
CHECKS=0
for M in $MODELS; do
  NAME=$(basename "$M")
  if ! can_load ./notorch "$M"; then
    echo "parity  [$NAME] SKIPPED — the harness could not load it: $(why_not ./notorch "$M")"
    continue
  fi
  if ! can_load ./infer_llama "$M"; then
    echo "parity  [$NAME] SKIPPED — the reference could not load it: $(why_not ./infer_llama "$M")"
    continue
  fi
  for P in "The capital of France is" "Resonance is" "def fibonacci(n):"; do
    A=$(./notorch "$M" "$P" 24 0 2>/dev/null)
    B=$(./infer_llama "$M" "$P" 24 0 2>/dev/null | ref_text)
    CHECKS=$((CHECKS + 1))
    # Two empty strings are equal and prove nothing, so an empty side is a
    # failure even when both sides are empty.
    if [ -n "$A" ] && [ "$A" = "$B" ]; then
      echo "parity  [$NAME] \"$P\"  identical  PASS"
    else
      echo "parity  [$NAME] \"$P\"  FAIL"
      echo "  harness: $A"
      echo "  example: $B"
      FAILS=$((FAILS + 1))
    fi
  done
done

if [ "$FAILS" -ne 0 ]; then
  echo "NOTORCH_PARITY_FAIL ($FAILS)"
  exit 1
elif [ "$CHECKS" -eq 0 ]; then
  echo "parity  (every model given was skipped — nothing was compared)"
  echo "NOTORCH_PARITY_SKIPPED"
else
  echo "NOTORCH_PARITY_OK ($CHECKS checks)"
fi
