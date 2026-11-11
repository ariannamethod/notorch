#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
export NT_ATTN_THREADS=1 NT_CHAT='||' NT_TOKENS= NT_NO_I8=1
cli="$root/notorch"
"$cli" -q -n 1 -t 0 --experts --learn-experts --portable-spore \
    --spore-out parent source ' exposure' >seed.out 2>seed.trace
convert() {
    "$cli" -q --experts --spore-only --spore-in "$1" --spore-out "$2" \
        --expert-vote-rms 1 "$3" >convert.out 2>convert.trace
}
convert parent calibrated source
convert calibrated repeated source
cmp calibrated repeated
"$cli" -q --experts --spore-only --spore-in calibrated --spore-out restored source >copy.out 2>copy.trace
cmp calibrated restored
convert parent direct wider
"$cli" -q --experts --spore-only --spore-in parent --spore-out imported wider >import.out 2>import.trace
convert imported staged wider
cmp direct staged
"$cli" -q -n 1 -t 0 --experts --learn-experts --spore-in calibrated --spore-out sync \
    source ' probe' >sync.out 2>sync.trace
"$cli" -q -n 1 -t 0 --experts --learn-experts --async-consolidation --spore-in calibrated \
    --spore-out worker source ' probe' >worker.out 2>worker.trace
cmp sync worker
cmp sync.out worker.out
reject() {
    if "$cli" "$@" >bad.out 2>bad.trace; then
        echo 'accepted invalid vote calibration' >&2; exit 1
    else code=$?; fi
    test "$code" -eq 1
    test ! -e rejected
    if grep -Eq 'Sanitizer|runtime error:' bad.trace; then cat bad.trace >&2; exit 1; fi
}
for target in 0 -1 nan inf 1e999 1e-999 nope 1tail; do
    reject --experts --spore-only --spore-in parent --spore-out rejected --expert-vote-rms "$target" source
done
reject --expert-vote-rms
reject --expert-vote-rms 1 source
reject --experts --expert-vote-rms 1 source
reject --experts --spore-in parent --expert-vote-rms 1 --expert-vote-rms 2 source
echo 'Vote CLI: explicit transfer, idempotent restart, sync/worker and invalid flags passed'
