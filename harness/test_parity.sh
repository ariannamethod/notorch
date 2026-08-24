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

FAILS=0
for M in $MODELS; do
  NAME=$(basename "$M")
  for P in "The capital of France is" "Resonance is" "def fibonacci(n):"; do
    A=$(./notorch "$M" "$P" 24 0 2>/dev/null)
    B=$(./infer_llama "$M" "$P" 24 0 2>/dev/null | ref_text)
    if [ "$A" = "$B" ]; then
      echo "parity  [$NAME] \"$P\"  identical  PASS"
    else
      echo "parity  [$NAME] \"$P\"  FAIL"
      echo "  harness: $A"
      echo "  example: $B"
      FAILS=$((FAILS + 1))
    fi
  done
done

if [ "$FAILS" -eq 0 ]; then
  echo "NOTORCH_PARITY_OK"
else
  echo "NOTORCH_PARITY_FAIL ($FAILS)"
  exit 1
fi
