#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/expert_fixtures" host.gguf second.gguf
cp host.gguf same-shape.gguf
printf x >> same-shape.gguf
"$root/test_expert_consolidation"
export NT_TOKENS=3,4,5 NT_CTX=64 NT_CHAT='||' NT_ATTN_THREADS=1
run() {
    "$root/notorch" -q -n 2 -t 0 --experts --learn-experts --expert-trace "$@" host.gguf
}
printf 'one\ntwo\n/reset\nthree\n' > turns
run --spore-out sync < turns > sync.txt 2> sync.trace
run --async-consolidation --spore-out async < turns > async.txt 2> async.trace
cmp sync async
cmp sync.txt async.txt
# Chat writes prompts to stderr; a worker may finish on either side of "> ".
for mode in sync async; do
    sed -nE 's/.*(doe (choice|vote|turn=).*)/\1/p' "$mode.trace" > "$mode.events"
done
cmp sync.events async.events
test "$(grep -c 'doe turn=' async.events)" -eq 3
cat turns > quit
printf '/exit\n' >> quit
run --async-consolidation --spore-out exit < quit > quit.txt 2> quit.trace
cmp sync exit
# Compare a live second turn after a KV reset with restart from its parent.
printf 'one\n/reset\ntwo\n' | run --async-consolidation --spore-out live > live.txt 2> live.trace
printf 'one\n' | run --async-consolidation --spore-out first > first.txt 2> first.trace
printf 'two\n' | run --async-consolidation --spore-in first --spore-out resumed > resumed.txt 2> resumed.trace
cmp live resumed
cat first.txt resumed.txt > replay.txt
cmp live.txt replay.txt
if "$root/notorch" --experts --async-consolidation --spore-out invalid host.gguf; then exit 1; fi
test ! -e invalid
test ! -e failed.gguf
test ! -e premature.gguf
echo 'chat: immediate next input, reset, EOF and explicit exit: PASS'
