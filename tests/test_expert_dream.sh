#!/bin/sh
set -eu
root=$(pwd)
bin=${NT_TEST_HARNESS:-$root/notorch}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
export NT_ATTN_THREADS=1 NT_CHAT='||' NT_CTX=32
run() { "$bin" -q -t 0 -n 1 --experts --expert-trace "$@" </dev/null >out 2>trace; }
sleep_on() { run --dream-only 6 --spore-in "$1" --spore-out "$2" --portable-spore --learn-experts "$3"; }
reject() {
    if run "$@"; then return 1; else rc=$?; fi
    test "$rc" -eq 1 && ! grep -Eq 'Sanitizer|runtime error:' trace
}

run --learn-experts --portable-spore --spore-out awake source ' exposure'
cp out awake.out
run --dream-only 6 --spore-in awake --spore-out preview --portable-spore source
test ! -s out
cmp awake preview
sed -n '/doe dream-start /,/doe dream-end /p' trace | sed 's/learn=0/learn=1/' >preview.trace
sleep_on awake one source
test ! -s out
cp trace one.trace
sed -n '/doe dream-start /,/doe dream-end /p' trace | cmp preview.trace -
"$root/test_expert_dream" awake one one.trace
cmp awake failure-copy

# Waking output, sync, async, resumed, and both serialization modes.
for mode in sync async; do
    extra=
    if [ "$mode" = async ]; then extra=--async-consolidation; fi
    run --learn-experts $extra --dream 6 --portable-spore --spore-out "$mode" source ' exposure'
    cmp awake.out out
    cmp one "$mode"
done
run --spore-only --spore-in one --spore-out exact source
run --spore-only --spore-in exact --spore-out exact-packed --portable-spore source
cmp one exact-packed
sleep_on one two source
run --dream-only 6 --learn-experts --async-consolidation --spore-in exact --portable-spore --spore-out two-async source
cmp two two-async
run --spore-only --spore-in one --spore-out moved wider
grep -q 'width=64->96' trace
run --dream-only 1 --async-consolidation --learn-experts --spore-in moved --spore-out onward wider
grep -q 'dream-end cycle=2 choices=1 ' trace

# Keep dialogue KV across two turns. /reset clears only KV; EOF waits for the worker.
for mode in sync async; do
    extra=
    if [ "$mode" = async ]; then extra=--async-consolidation; fi
    printf ' exposure\n probe\n/reset\n probe\n' |
        "$bin" -q -t 0 -n 1 --experts --expert-trace --learn-experts $extra --dream 6 \
        --spore-out "chat-$mode" source >"$mode.out" 2>"$mode.trace"
    grep -q 'dream-end cycle=3 ' "$mode.trace"
done
cmp sync.out async.out
cmp chat-sync chat-async

# 32 cues are consumed once; the 33rd begins the next round after a disk restart.
parent=awake
i=1
while [ "$i" -le 33 ]; do
    run --dream-only 1 --learn-experts --spore-in "$parent" --spore-out "round-$i" source
    sed -n 's/.*dream-start .*cue=\([0-9]*\) literal=.*/\1/p' trace >>cues
    parent=round-$i
    i=$((i+1))
done
test "$(head -32 cues | sort -u | wc -l)" -eq 32
test "$(wc -l <cues)" -eq 33
grep -q 'dream-end cycle=33 choices=1' trace

for n in 0 -1 33 2junk; do reject --dream-only "$n" --spore-in awake source; done
reject --dream 2 source ' exposure'
reject --dream 2 --learn-experts --dream-only 1 --spore-in awake source
reject --dream-only 1 source
reject --dream-only 1 --spore-in awake source ' exposure'
reject --dream-only 1 --spore-in awake --learn-experts source
reject --dream-only 1 --spore-in awake --spore-only --spore-out refused source
reject --dream-only 1 --spore-in exact --learn-experts --spore-out refused wider
run --learn-experts --spore-out flat-awake flat ' exposure'
reject --async-consolidation --dream-only 1 --spore-in flat-awake --learn-experts --spore-out refused flat
test ! -e refused
echo 'Private dreams: cue energy, elected factors, EOS, isolated state, two hops, worker, failure and 33 cycles passed'
