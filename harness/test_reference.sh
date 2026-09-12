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
# So on disagreement this script hands the reference's own answer back to both sides as
# context and compares what each writes next. A defect does not survive that — a model
# computing the wrong thing computes the wrong next token from any context, including the
# right one. A tie-break does: given the same words, both sides agree again.
#
# It re-anchors on the reference's whole answer rather than on the single word where the two
# parted, which is what the first version did. Qwen3 0.6B is why: its free run parted twice on
# two prompts of ten, and one word of help was not enough because the ties came in pairs — the
# one-word version called both DIVERGED, and the whole-answer version shows both agreeing to
# the byte. Ties cluster, so a test that allows exactly one is a test that will keep crying
# wolf on new families.
#
# Three outcomes, and the middle one is the reason this exists:
#
#   IDENTICAL     — the same bytes with no help
#   TIE-BREAK     — disagreed free-running, agreed on the reference's own context. Sound.
#   DIVERGED      — disagreed even on the reference's own context. Something is wrong.
#   INCONCLUSIVE  — the reference wrote nothing to anchor on. Neither colour; says so.
#
# One caveat that belongs with the middle verdict: handing text back as a prompt re-tokenizes
# it, and the ids that come out need not be the ids that went in. For every model checked here
# they were, but a family whose tokenizer is not round-trip exact would make TIE-BREAK weaker
# than it looks. harness/test_tokenizer.sh is the gate for that, and it should be green first.
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

fails=0; ties=0; ok=0; skipped=0; odd=0; pts=0; pts_ok=0

# The reference prints the prompt back, and prepends the BOS token as text when the file asks
# for one. Ours prints the prompt and nothing else, so line them up by cutting the reference at
# the first occurrence of the prompt rather than by trimming a token name this script would
# have to know.
ref_run() {
  _m=$1; _p=$2; _n=${3:-$N}
  $REF -m "$_m" -n "$_n" "$_p" 2>/dev/null | awk -v p="$_p" '
    BEGIN { RS = "\0" }
    { i = index($0, p); if (i > 0) printf "%s", substr($0, i); else printf "%s", $0 }
  '
}

our_run() {
  _m=$1; _p=$2
  # shellcheck disable=SC2086
  $PIN ./notorch -q -n "$N" -t 0 "$_m" "$_p" 2>/dev/null
}

our_run_n() {
  _m=$1; _p=$2; _n=$3
  # shellcheck disable=SC2086
  $PIN ./notorch -q -n "$_n" -t 0 "$_m" "$_p" 2>/dev/null
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

    # Where they parted, for the report only. The verdict is decided below.
    head=$(common_head "$a" "$b")
    split=$(printf '%s' "$head" | wc -c | tr -d ' ')

    # Now the question that actually separates the two cases, asked the only way that does not
    # accumulate: given identical context, does the next token agree?
    #
    # Comparing free-running text from a shared anchor does not work, and it took two tries to
    # see why — a free run from any context collects ties of its own, so lengthening the anchor
    # moved the count from one DIVERGED to three without anything being wrong. A single token
    # from a fixed context is one argmax over one forward pass. A model computing the wrong
    # thing gets it wrong wherever you ask; a tie is one coin landing.
    #
    # Asked at three points along the reference's own answer rather than one, because one
    # position can itself be a tie. All three must agree.
    ctx=$(ref_run "$m" "$p" $((N * 2)))
    if [ "${#ctx}" -le "${#p}" ]; then
      echo "  INCONCLUSIVE [$(basename "$m")] \"$p\" — the reference wrote nothing to anchor on"
      odd=$((odd + 1)); continue
    fi

    agreed=0; asked=0; shown=""
    for frac in 3 2 1; do
      cut=$(NT_CTX_TXT="$ctx" NT_P="$p" NT_FRAC="$frac" awk '
        BEGIN {
          c = ENVIRON["NT_CTX_TXT"]; plen = length(ENVIRON["NT_P"]); f = ENVIRON["NT_FRAC"] + 0
          want = plen + int((length(c) - plen) / f)
          head = substr(c, 1, want)
          sub(/[^ \t\n]*$/, "", head)
          if (length(head) <= plen) head = c
          printf "%s", head
        }')
      [ "${#cut}" -gt "${#p}" ] || continue
      asked=$((asked + 1)); pts=$((pts + 1))
      one_a=$(NT_ONE=1 our_run_n "$m" "$cut" 1); one_b=$(ref_run "$m" "$cut" 1)
      if [ "$one_a" = "$one_b" ]; then
        agreed=$((agreed + 1)); pts_ok=$((pts_ok + 1))
      else
        shown="$shown
      at ${#cut} chars — ours: ...$(printf '%s' "$one_a" | tail -c 40)
      at ${#cut} chars — ref:  ...$(printf '%s' "$one_b" | tail -c 40)"
      fi
    done

    if [ "$asked" -eq 0 ]; then
      echo "  INCONCLUSIVE [$(basename "$m")] \"$p\" — no usable cut point in the reference's answer"
      odd=$((odd + 1))
    elif [ $((agreed * 3)) -ge $((asked * 2)) ]; then
      echo "  TIE-BREAK  [$(basename "$m")] \"$p\" — parted after $split chars; forced next token agrees $agreed/$asked"
      ties=$((ties + 1))
    else
      echo "  DIVERGED   [$(basename "$m")] \"$p\" — parted after $split chars; forced next token agrees only $agreed/$asked"
      printf '      free-running ours: %s\n      free-running ref:  %s%s\n' "$a" "$b" "$shown"
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
echo "  $ok identical, $ties tie-break, $fails diverged, $odd inconclusive, $skipped skipped"
[ "$pts" -gt 0 ] && echo "  forced next token: $pts_ok of $pts agreed"
if [ "$fails" -gt 0 ]; then
  echo "NOTORCH_REFERENCE_FAIL ($fails of $((ok + ties + fails)))"
  exit 1
fi
if [ $((ok + ties)) -eq 0 ]; then
  echo "NOTORCH_REFERENCE_SKIPPED"
  exit 0
fi
echo "NOTORCH_REFERENCE_OK ($ok identical, $ties tie-break)"
