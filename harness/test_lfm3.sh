#!/bin/sh
# test_lfm3.sh — LFM3 parity against llama.cpp, the reference implementation.
#
# The reference for this family is not infer_llama.c (it loads only the standard
# families) but llama.cpp itself, at temperature 0 — and tonight's calibration
# says the comparison only means anything with sampling stripped to bare greedy:
# llama-server's defaults carry repeat_penalty 1.1, which flips continuations at
# any position whose history contains the natural next token. Every request
# here pins repeat_penalty to 1.0.
#
# The llama-server binary is not part of this repository and has no default
# path: point LFM3_LLAMA_BIN at one. A run that compares nothing — no binary,
# no models, every model skipped — is red, not green: a gate that checked
# nothing has said nothing. notorch must be built (make harness).
#
#   LFM3_LLAMA_BIN=/path/to/llama-server ./harness/test_lfm3.sh model.gguf ...
#
set -eu
cd "$(dirname "$0")/.."

[ -x ./notorch ] || { echo "test_lfm3: ./notorch not built (make harness)"; exit 1; }

not_run() {
    echo "parity(lfm3): NOT RUN — $1"
    echo "NOTORCH_LFM3_PARITY_FAILED"
    exit 1
}
if [ -z "${LFM3_LLAMA_BIN:-}" ] || [ ! -x "${LFM3_LLAMA_BIN}" ]; then
    not_run "LFM3_LLAMA_BIN is unset or not executable"
fi
command -v curl >/dev/null 2>&1 || not_run "curl not found"
command -v python3 >/dev/null 2>&1 || not_run "python3 not found (JSON parsing)"
[ $# -gt 0 ] || not_run "no model given"

PORT=${LFM3_PORT:-31999}
REF_URL="http://127.0.0.1:${PORT}"
REF_PID=""
REQ=$(mktemp)
trap '[ -n "$REF_PID" ] && kill -9 "$REF_PID" 2>/dev/null; rm -f "$REQ"' EXIT

ref_start() {
    # A server already answering on the port would be compared against in place of
    # the model named here.
    if curl -s --max-time 2 "${REF_URL}/health" 2>/dev/null | grep -q ok; then
        echo "parity(lfm3): port ${PORT} already serves something; refusing to compare against it"
        return 1
    fi
    "${LFM3_LLAMA_BIN}" -m "$1" -ngl 0 --port "${PORT}" >/dev/null 2>&1 &
    REF_PID=$!
    i=0
    while [ $i -lt 120 ]; do
        if curl -s --max-time 2 "${REF_URL}/health" 2>/dev/null | grep -q ok; then return 0; fi
        sleep 1; i=$((i + 1))
    done
    echo "parity(lfm3): reference server did not come up"
    kill -9 $REF_PID 2>/dev/null || true
    return 1
}

# reference completion: bare greedy, no penalties, no truncated top-k
ref() {
    printf '{"prompt":%s,"n_predict":%d,"temperature":0.0,"repeat_penalty":1.0,"top_k":0,"top_p":1.0,"min_p":0.0}' \
        "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")" "$2" > "$REQ"
    curl -s --max-time 300 "${REF_URL}/completion" -H "Content-Type: application/json" \
        -d @"$REQ" | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])'
}

PROMPTS='
The capital of France is:24
The first man on the moon was:32
1, 2, 3, 4,:32
Столица Франции это:8
'

FAILS=0; CHECKS=0
for M in ${LFM3_MODELS:-}; do :; done
for M in "$@"; do
    NAME=$(basename "$M")
    if ! ./notorch "$M" "x" 1 0 >/dev/null 2>&1; then
        echo "parity(lfm3) [$NAME] SKIPPED — the harness could not load it"
        continue
    fi
    if ! ref_start "$M"; then
        echo "parity(lfm3) [$NAME] SKIPPED — the reference could not load it"
        continue
    fi
    # A plain pipeline would run the loop in a subshell and FAILS would die
    # inside it — the gate would print OK over its own red. Read the plan into
    # the shell instead.
    PLAN=$(echo "$PROMPTS")
    while IFS=: read -r P N; do
        [ -n "$P" ] || continue
        WANT=$(ref "$P" "$N")
        RAW=$(./notorch -q "$M" "$P" "$N" 0 2>/dev/null)
        # notorch stdout is the prompt echoed back with the continuation after
        # it; the reference returns the continuation alone.
        case "$RAW" in
            "$P"*) GOT=${RAW#"$P"} ;;
            *)     GOT="$RAW" ;;
        esac
        CHECKS=$((CHECKS + 1))
        if [ "$WANT" = "$GOT" ]; then
            echo "parity(lfm3) [$NAME] PASS  ${P}"
        else
            echo "parity(lfm3) [$NAME] FAIL  ${P}"
            echo "    ref:  ${WANT}"
            echo "    ours: ${GOT}"
            FAILS=$((FAILS + 1))
        fi
    done <<EOF2
$PLAN
EOF2
    kill -9 $REF_PID 2>/dev/null || true
    wait $REF_PID 2>/dev/null || true
    REF_PID=""
done

if [ "$CHECKS" -eq 0 ]; then
    echo "parity(lfm3): no comparison ran — $# model(s) given, every one skipped"
    echo "NOTORCH_LFM3_PARITY_FAILED"
    exit 1
fi
echo "parity(lfm3): $CHECKS comparisons, $FAILS failed"
if [ "$FAILS" -gt 0 ]; then
    echo "NOTORCH_LFM3_PARITY_FAILED"
    exit 1
fi
echo "NOTORCH_LFM3_PARITY_OK"
