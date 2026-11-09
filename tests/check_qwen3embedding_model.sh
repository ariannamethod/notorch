#!/bin/sh
# Real-checkpoint gates for the Qwen3-Embedding encoder, run from the repository root:
#   tests/check_qwen3embedding_model.sh MODEL.gguf TOKENIZER.gguf
# 1. determinism: the same texts encoded by two processes give byte-identical
#    pooled and embedding values;
# 2. shape and norm: every embedding has the model's width, 1024 for the
#    0.6B checkpoint, and |norm - 1| <= 2e-6;
# 3. published scores: the four query/document scores the model card prints for
#    its Transformers recipe (README.md of Qwen/Qwen3-Embedding-0.6B at revision
#    97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3), within 1e-4, with the card's
#    own prompt and texts.
set -eu
if [ $# -ne 2 ] || [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "usage: $0 MODEL.gguf TOKENIZER.gguf" >&2; exit 2
fi
MODEL=$1 TOK=$2 WIDTH=1024
DIR=$(mktemp -d /tmp/notorch-qe-model-XXXXXX)
trap 'rm -rf "$DIR"' EXIT
printf 'q1\tWhat is the capital of China?\nq2\tExplain gravity\n' > "$DIR/queries.tsv"
printf 'd1\tThe capital of China is Beijing.\nd2\tGravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.\n' > "$DIR/documents.tsv"
for run in 1 2; do
    ./qwen3embedding_encode "$MODEL" "$DIR/queries.tsv" "$DIR/q$run.jsonl" --tokenizer "$TOK" --condition query 2>"$DIR/q$run.err" ||
        { cat "$DIR/q$run.err" >&2; exit 1; }
    ./qwen3embedding_encode "$MODEL" "$DIR/documents.tsv" "$DIR/d$run.jsonl" --tokenizer "$TOK" --condition document 2>"$DIR/d$run.err" ||
        { cat "$DIR/d$run.err" >&2; exit 1; }
done
fail=0
for side in q d; do
    sed 's/"seconds":[^,]*,//' "$DIR/${side}1.jsonl" > "$DIR/${side}1.cmp"
    sed 's/"seconds":[^,]*,//' "$DIR/${side}2.jsonl" > "$DIR/${side}2.cmp"
    if cmp -s "$DIR/${side}1.cmp" "$DIR/${side}2.cmp"; then echo "determinism $side: two processes, byte-identical vectors PASS"
    else echo "determinism $side: vectors differ FAIL"; fail=1; fi
done
cat "$DIR/q1.jsonl" "$DIR/d1.jsonl" | awk -v width="$WIDTH" '
{
    id = $0; sub(/^\{"id":"/, "", id); sub(/".*/, "", id)
    ids = $0; sub(/.*"input_ids":\[/, "", ids); sub(/\].*/, "", ids)
    e = $0; sub(/.*"embedding":\[/, "", e); sub(/\].*/, "", e)
    n = split(e, v, ","); s = 0
    for (i = 1; i <= n; i++) s += v[i] * v[i]
    err = sqrt(s) - 1; if (err < 0) err = -err
    ok = n == width && err <= 2e-6
    printf "%s: %d IDs [%s], width %d, |norm-1| %.3g %s\n", id, split(ids, x, ","), ids, n, err, ok ? "PASS" : "FAIL"
    if (!ok) bad = 1
}
END { exit bad }' || fail=1
./embeddinggemma2_score "$DIR/q1.jsonl" "$DIR/d1.jsonl" "$DIR/scores.jsonl"
awk '
BEGIN { want["q1,1"] = 0.7645568251609802; want["q1,2"] = 0.14142508804798126
        want["q2,1"] = 0.13549736142158508; want["q2,2"] = 0.5999549627304077 }
{
    id = $0; sub(/^\{"query_id":"/, "", id); sub(/".*/, "", id)
    s = $0; sub(/.*"scores":\[/, "", s); sub(/\].*/, "", s)
    split(s, v, ",")
    for (j = 1; j <= 2; j++) {
        if (!((id "," j) in want)) { printf "unexpected score %s,d%d\n", id, j; seen = -99; continue }
        d = v[j] - want[id "," j]; if (d < 0) d = -d
        if (d > worst) worst = d
        printf "score %s,d%d: %.17g published %.17g |delta| %.3g\n", id, j, v[j], want[id "," j], d
        seen++
    }
}
END { ok = seen == 4 && worst <= 1e-4
      printf "published scores: max |delta| %.3g (<= 1e-4) %s\n", worst, ok ? "PASS" : "FAIL"; exit !ok }' "$DIR/scores.jsonl" || fail=1
[ $fail -eq 0 ] && echo "Qwen3-Embedding model gates PASS" || echo "Qwen3-Embedding model gates FAIL"
exit $fail
