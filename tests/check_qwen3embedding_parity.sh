#!/bin/sh
# Vector parity against official reference vectors, run from the repository root:
#   tests/check_qwen3embedding_parity.sh MODEL.gguf REFERENCE.jsonl
# REFERENCE.jsonl: one line per input, {"id": ..., "input_ids": [...], "embedding": [...]},
# the official model's normalized float32 output for exactly those IDs (format and
# provenance: docs/QWEN3EMBEDDING.md). The native encoder encodes the reference's own
# IDs in ID mode, so the tokenizer boundary stays outside this gate, as in EG2's RG3.
# Thresholds, fixed before any reference exists (EG2 RG3): per vector
# max |delta| <= 2e-5, cosine >= 0.999999, native |norm - 1| <= 2e-6.
set -eu
if [ $# -ne 2 ] || [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "usage: $0 MODEL.gguf REFERENCE.jsonl" >&2; exit 2
fi
DIR=$(mktemp -d /tmp/notorch-qe-parity-XXXXXX)
trap 'rm -rf "$DIR"' EXIT
awk '{
    id = $0; sub(/.*"id": *"/, "", id); sub(/".*/, "", id)
    ids = $0; sub(/.*"input_ids": *\[/, "", ids); sub(/\].*/, "", ids); gsub(/[ ,]+/, " ", ids)
    print id "\t" ids
}' "$2" > "$DIR/ids.tsv"
./qwen3embedding_encode "$1" "$DIR/ids.tsv" "$DIR/native.jsonl" 2>"$DIR/encode.err" || { cat "$DIR/encode.err" >&2; exit 1; }
awk '
function vec(line, a,   e) { e = line; sub(/.*"embedding": *\[/, "", e); sub(/\].*/, "", e); return split(e, a, ",") }
function key(line,   id) { id = line; sub(/.*"id": *"/, "", id); sub(/".*/, "", id); return id }
FNR == NR { n = vec($0, r); k = key($0); width[k] = n; for (i = 1; i <= n; i++) ref[k, i] = r[i]; total++; next }
{
    k = key($0); n = vec($0, v); seen++
    if (!(k in width) || width[k] != n) { printf "%s: width %d, reference %d FAIL\n", k, n, width[k]; bad = 1; next }
    d = 0; dot = 0; nn = 0; rr = 0
    for (i = 1; i <= n; i++) {
        x = v[i] - ref[k, i]; if (x < 0) x = -x; if (x > d) d = x
        dot += v[i] * ref[k, i]; nn += v[i] * v[i]; rr += ref[k, i] * ref[k, i]
    }
    cosv = dot / sqrt(nn * rr); ne = sqrt(nn) - 1; if (ne < 0) ne = -ne
    ok = d <= 2e-5 && cosv >= 0.999999 && ne <= 2e-6
    printf "%s: max |delta| %.3g, cosine %.15g, |norm-1| %.3g %s\n", k, d, cosv, ne, ok ? "PASS" : "FAIL"
    if (!ok) bad = 1; else good++
    if (d > worst) worst = d
}
END { printf "vector parity: %d/%d vectors within thresholds, max |delta| %.3g\n", good, total, worst
      exit bad || seen != total || total == 0 }' "$2" "$DIR/native.jsonl"
