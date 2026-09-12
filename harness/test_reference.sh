#!/bin/sh
# test_reference.sh — compare against llama.cpp, and tell a tie-break from a defect.
#
# harness/test_parity.sh compares this tree against its own example, which catches a harness
# that drifted from the arithmetic it was lifted from. It cannot catch the arithmetic being
# wrong in both. For the families llama.cpp supports, that is what an outside reference is for.
#
# The problem with a text comparison against an outside implementation is that it fails for a
# reason that is not a defect. Greedy decoding takes an argmax, our summation order is not the
# reference's, and at a position where the top two logits sit within float noise the two pick
# differently. Everything after that follows from the one token, so a single tie-break reads as
# a completely different answer. Measured on this machine in one afternoon: SmolLM2 360M Q8_0
# diverged on one prompt of three, DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M on three of three, and
# qwen05b Q4_K_M on one of two. In every case, feeding the reference's own token at the point
# of disagreement and continuing produced byte-identical text for the rest of the run.
#
# So this script does that automatically. On disagreement it re-anchors: takes the agreed text
# plus the reference's next word, hands it back as a prompt, and compares the continuation. A
# defect does not survive re-anchoring — a model computing the wrong thing goes on computing
# the wrong thing. A tie-break does.
#
# Three outcomes, and the middle one is the reason this exists:
#
#   IDENTICAL  — the same bytes with no help
#   TIE-BREAK  — disagreed, and agreed again from the reference's token. Arithmetic sound.
#   DIVERGED   — disagreed, and disagreed again after re-anchoring. Something is wrong.
#
# What it cannot tell you: how close the tie was. That needs logits from both sides, and
# llama-simple does not print them. A TIE-BREAK is evidence, not proof — a defect small enough
# to flip one argmax and then never matter again would look the same. Nothing seen here so far
# has behaved that way, and if one does, the re-anchor count below will start climbing.
#
# It was built by breaking it. Dividing rmsnorm by n-1 instead of n — the kind of off-by-one
# this is for — turns one prompt DIVERGED with the two continuations printed side by side, and
# the script exits 1; restoring the line puts it back to 0. Worth knowing from the same
# experiment: the other two prompts stayed IDENTICAL under that defect, so three prompts is a
# thin net. Widen it with NT_REF_PROMPTS when a family is new. And a second thing that is not
# about this script — multiplying rmsnorm's epsilon by ten changed no output at all, because
# eps sits five orders below the mean square. Not every wrong number is a detectable one.
#
#   ./harness/test_reference.sh <model.gguf> [more.gguf ...]
#
# Needs ./notorch built and llama-simple on PATH. Says SKIPPED, not OK, when either is missing:
# a gate that cannot run must report neither green nor red.
set -eu
cd "$(dirname "$0")/.."

REF=${NT_REF:-llama-simple}
command -v "$REF" >/dev/null 2>&1 || {
  echo "reference  (no $REF on this machine — set NT_REF or install llama.cpp)"
  echo "NOTORCH_REFERENCE_SKIPPED"
  exit 0
}
[ -x ./notorch ] || { echo "test_reference: ./notorch not built (make harness)"; exit 1; }

N=${NT_REF_TOKENS:-16}
PIN=${NT_REF_PIN:-}                       # e.g. "taskset -c 4-7"; empty runs unpinned
PROMPTS_FILE=${NT_REF_PROMPTS:-}

fails=0; ties=0; ok=0; skipped=0

# The reference prints the prompt back, and prepends the BOS token as text when the file asks
# for one. Ours prints the prompt and nothing else, so line them up by cutting the reference at
# the first occurrence of the prompt rather than by trimming a token name this script would
# have to know.
ref_run() {
  _m=$1; _p=$2
  $REF -m "$_m" -n "$N" "$_p" 2>/dev/null | awk -v p="$_p" '
    BEGIN { RS = "\0" }
    { i = index($0, p); if (i > 0) printf "%s", substr($0, i); else printf "%s", $0 }
  '
}

our_run() {
  _m=$1; _p=$2
  # shellcheck disable=SC2086
  $PIN ./notorch -q -n "$N" -t 0 "$_m" "$_p" 2>/dev/null
}

# The agreed head of two strings, cut back to the last whitespace so the re-anchor prompt ends
# on a word rather than inside one. Tokenizers split on word boundaries often enough that
# cutting mid-word would ask the model a different question than the reference was asked.
#
# Both strings arrive through the environment rather than through stdin, and that is the whole
# reason this is not three lines shorter. A model's answer is routinely several lines — the
# first version of this piped the two strings in and compared them with NR==1/NR==2, which
# silently compared only their first lines. On a Python prompt, where both answers begin with
# the same "def fibonacci(n):" line and part on the second, it reported a shared head of
# nothing and re-anchored on a fragment of a word. It said TIE-BREAK, which happened to be
# true, for a reason that was not.
common_head() {
  NT_A=$1 NT_B=$2 awk '
    BEGIN {
      a = ENVIRON["NT_A"]; b = ENVIRON["NT_B"]
      n = length(a); if (length(b) < n) n = length(b)
      for (i = 1; i <= n; i++) if (substr(a, i, 1) != substr(b, i, 1)) break
      head = substr(a, 1, i - 1)
      sub(/[^ \t\n]*$/, "", head)
      printf "%s", head
      exit
    }'
}

# The reference word that this tree did not pick — what re-anchoring hands back.
next_ref_word() {
  NT_REFTXT=$1 NT_HEAD=$2 awk '
    BEGIN {
      rest = substr(ENVIRON["NT_REFTXT"], length(ENVIRON["NT_HEAD"]) + 1)
      n = split(rest, w, /[ \t\n]+/)
      for (i = 1; i <= n; i++) if (w[i] != "") { printf "%s", w[i]; break }
      exit
    }'
}

check_model() {
  m=$1
  [ -f "$m" ] || { echo "  SKIP $(basename "$m") — not on this machine"; skipped=$((skipped + 1)); return 0; }

  if ! $PIN ./notorch -q -n 1 -t 0 "$m" "x" >/dev/null 2>&1; then
    echo "  SKIP $(basename "$m") — this tree does not load it"; skipped=$((skipped + 1)); return 0
  fi
  if ! $REF -m "$m" -n 1 "x" >/dev/null 2>&1; then
    echo "  SKIP $(basename "$m") — the reference does not load it"; skipped=$((skipped + 1)); return 0
  fi

  while IFS= read -r p; do
    [ -n "$p" ] || continue
    a=$(our_run "$m" "$p"); b=$(ref_run "$m" "$p")
    if [ "$a" = "$b" ]; then
      echo "  IDENTICAL  [$(basename "$m")] \"$p\""
      ok=$((ok + 1))
      continue
    fi

    head=$(common_head "$a" "$b")
    word=$(next_ref_word "$b" "$head")
    if [ -z "$word" ]; then
      echo "  DIVERGED   [$(basename "$m")] \"$p\" — no shared head to re-anchor on"
      printf '      ours: %s\n      ref:  %s\n' "$a" "$b"
      fails=$((fails + 1)); continue
    fi

    anchor="$head$word"
    a2=$(our_run "$m" "$anchor"); b2=$(ref_run "$m" "$anchor")
    if [ "$a2" = "$b2" ]; then
      # Say where, so a reader can see it was one token and not a region.
      echo "  TIE-BREAK  [$(basename "$m")] \"$p\" — split after $(printf '%s' "$head" | wc -c | tr -d ' ') chars, identical from the reference's \"$word\""
      ties=$((ties + 1))
    else
      echo "  DIVERGED   [$(basename "$m")] \"$p\" — still apart after re-anchoring on \"$word\""
      printf '      ours: %s\n      ref:  %s\n' "$a2" "$b2"
      fails=$((fails + 1))
    fi
  done <<EOF
$(if [ -n "$PROMPTS_FILE" ]; then cat "$PROMPTS_FILE"; else
printf '%s\n' \
  'The capital of France is' \
  'Resonance is' \
  'def fibonacci(n):'
fi)
EOF
}

echo "against the reference ($REF), $N tokens, temperature 0"
for m in "$@"; do check_model "$m"; done

[ $# -gt 0 ] || { echo "  (no model given — pass one or more .gguf)"; echo "NOTORCH_REFERENCE_SKIPPED"; exit 0; }

echo
echo "  $ok identical, $ties tie-break, $fails diverged, $skipped skipped"
if [ "$fails" -gt 0 ]; then
  echo "NOTORCH_REFERENCE_FAIL ($fails of $((ok + ties + fails)))"
  exit 1
fi
if [ $((ok + ties)) -eq 0 ]; then
  echo "NOTORCH_REFERENCE_SKIPPED"
  exit 0
fi
echo "NOTORCH_REFERENCE_OK ($ok identical, $ties tie-break)"
