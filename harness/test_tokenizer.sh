#!/bin/sh
# test_tokenizer.sh — our ids against llama.cpp's, on the same file, for the same text.
#
# A tokenizer is either identical to the reference or it is a different model wearing the
# same weights: one token of drift moves every position after it. So the gate is the id
# sequence, not the round-trip — a tokenizer can round-trip perfectly and still split
# differently, which is exactly how gemma-4 arrived here (nine tokens where llama.cpp makes
# six, and the text decoded back fine).
#
#   ./harness/test_tokenizer.sh <model.gguf> [more.gguf ...]
#
# Needs ./notorch built (make harness) and llama-tokenize on PATH or in the Termux prefix.
# Says so and skips when either is missing, because a gate that cannot run should not pass.
set -eu
cd "$(dirname "$0")/.."

[ -x ./notorch ] || { echo "test_tokenizer: ./notorch not built (make harness)"; exit 1; }

REF=""
for c in llama-tokenize /data/data/com.termux/files/usr/bin/llama-tokenize; do
  command -v "$c" >/dev/null 2>&1 && REF="$c" && break
  [ -x "$c" ] && REF="$c" && break
done
if [ -z "$REF" ]; then
  echo "tokenizer  (no llama-tokenize on this machine — nothing to compare against)"
  echo "NOTORCH_TOKENIZER_SKIPPED"
  exit 0
fi

MODELS="$*"
if [ -z "$MODELS" ]; then
  # $HOME is not always where the models are — this tree is often run from a chroot whose
  # home is elsewhere than the one holding the files, so look beside the repo as well.
  # olmoe earns its place here rather than for variety: its vocabulary carries USER_DEFINED
  # tokens — runs of real spaces, stored as literal text — and no other model in this list
  # does. The indented-code and repeated-space texts below were already here and passed for
  # years, because on a vocabulary without such tokens both implementations split the same way.
  for m in "$HOME/models/gemma-4-E2B-it-Q4_0.gguf" "$HOME/models/qwen05b_q5_0_ours.gguf" \
           "$HOME/models/olmoe-1b-7b-q4_0.gguf" \
           "../models/gemma-4-E2B-it-Q4_0.gguf" "../models/qwen05b_q5_0_ours.gguf" \
           "../models/olmoe-1b-7b-q4_0.gguf" \
           "../../models/gemma-4-E2B-it-Q4_0.gguf" "../../models/qwen05b_q5_0_ours.gguf" \
           "../../models/olmoe-1b-7b-q4_0.gguf"; do
    [ -f "$m" ] && MODELS="$MODELS $m"
  done
fi
if [ -z "$MODELS" ]; then
  echo "tokenizer  (no model given and no default on this machine — pass a .gguf)"
  echo "NOTORCH_TOKENIZER_SKIPPED"
  exit 0
fi

# Cases chosen for the ways a tokenizer breaks: a plain sentence, leading and repeated
# spaces, punctuation runs, digits, a non-Latin script, code with indentation, and text
# that has no whole-token spelling and must fall back to bytes.
set -- \
  "The capital of France is" \
  "Hello, world!" \
  "  leading and   repeated   spaces" \
  "2 + 2 = 4, right?!" \
  "Привет, как дела?" \
  "def fibonacci(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)" \
  "мороженое 🍦 и ещё emoji 🚀" \
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

FAILS=0
CHECKS=0
for M in $MODELS; do
  NAME=$(basename "$M")
  for P in "$@"; do
    OURS=$(./notorch -T "$M" "$P" 2>/dev/null)
    THEIRS=$("$REF" -m "$M" -p "$P" 2>/dev/null | awk -F" -> " 'NF>1 { gsub(/[^0-9]/, "", $1); if ($1 != "") printf "%s%s", (n++ ? "," : ""), $1 } END { print "" }')
    CHECKS=$((CHECKS + 1))
    SHORT=$(printf '%s' "$P" | tr '\n' ' ' | cut -c1-40)
    if [ "$OURS" = "$THEIRS" ]; then
      echo "tokenizer  [$NAME] \"$SHORT\"  $(printf '%s' "$OURS" | tr ',' '\n' | grep -c .) ids  identical  PASS"
    else
      echo "tokenizer  [$NAME] \"$SHORT\"  FAIL"
      echo "  ours:   $OURS"
      echo "  theirs: $THEIRS"
      FAILS=$((FAILS + 1))
    fi
  done
done

if [ "$FAILS" -eq 0 ]; then
  echo "NOTORCH_TOKENIZER_OK ($CHECKS checks)"
else
  echo "NOTORCH_TOKENIZER_FAIL ($FAILS of $CHECKS)"
  exit 1
fi
