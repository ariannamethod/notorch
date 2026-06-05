#!/usr/bin/env bash
# bench_gguf_metal.sh — reproducible benchmark for infer_gguf_metal (Apple Metal,
# packed-Q4_K GGUF inference). Deterministic greedy (srand(42), temp=0), fixed
# factual prompt so correctness is checkable byte-for-byte.
#
# Usage:  ./examples/bench_gguf_metal.sh "label=path/to/model.gguf" [more...]
#         (no args -> uses the DoE.aml zoo paths on this Metal node)
#
# Emits: per-model generated text (proof) + a markdown table
# (arch | prefill t/s | decode t/s | peak RSS | "Paris"? ).
# Reproduce on any Apple-Silicon + Metal machine with the same GGUF files.
set -u
BIN="$(dirname "$0")/infer_gguf_metal"
PROMPT="The capital of France is"
NTOK=8
ZOO="$HOME/arianna/DoE.aml/weights/zoo"

if [ $# -eq 0 ]; then
  set -- \
    "SmolLM2-360M(llama)=$ZOO/SmolLM2-360M/SmolLM2-360M-Instruct-Q4_K_M.gguf" \
    "Qwen3-0.6B(qwen3)=$ZOO/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf" \
    "Qwen3-1.7B(qwen3)=$ZOO/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf" \
    "Llama-3.2-3B(llama)=$ZOO/Llama-3.2-3B/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
fi

[ -x "$BIN" ] || { echo "build first: cc -O2 -DUSE_METAL -I. examples/infer_gguf_metal.c examples/bpe.c gguf.c notorch_metal.o -framework Metal -framework Foundation -lc++ -lm -o examples/infer_gguf_metal"; exit 1; }

rows=""
for spec in "$@"; do
  label="${spec%%=*}"; path="${spec#*=}"
  [ -f "$path" ] || { echo "SKIP $label (missing $path)"; continue; }
  echo "═══ $label ═══"
  out="$(/usr/bin/time -l "$BIN" "$path" "$PROMPT" "$NTOK" 0 2>&1)"
  echo "$out" | sed -n '/^---$/,/^---$/p' | sed '1d;$d'         # generated text
  arch="$(echo "$out" | sed -n 's/^model: arch=\([a-z0-9]*\).*/\1/p')"
  pre="$(echo "$out" | sed -n 's/.*prefill:.*(\([0-9.]*\) t\/s).*/\1/p')"
  dec="$(echo "$out" | sed -n 's/.*decode:.*(\([0-9.]*\) t\/s)/\1/p')"
  rss="$(echo "$out" | awk '/peak memory footprint/{printf "%.2f GB", $1/1073741824}')"
  paris="no"; echo "$out" | grep -q "Paris" && paris="yes"
  echo "  arch=$arch prefill=${pre}t/s decode=${dec}t/s rss=$rss paris=$paris"
  rows="$rows| $label | $arch | $pre | $dec | $rss | $paris |\n"
done

echo
echo "## benchmark (greedy, temp=0, prompt=\"$PROMPT\", $NTOK tokens)"
echo "| model | arch | prefill t/s | decode t/s | peak RSS | \"Paris\"? |"
echo "|---|---|---|---|---|---|"
printf "$rows"
