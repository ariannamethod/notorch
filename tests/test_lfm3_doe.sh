#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_lfm3" fixtures
"$root/test_portable_spore" fixtures
"$root/test_portable_spore" tokenizer lfm3
export NT_ATTN_THREADS=1 NT_CHAT='||' NT_CTX=32
run() { "$root/notorch" -q -t 0 -n 1 --experts --expert-trace "$@" </dev/null >out 2>trace; }
bind() { run --spore-only --spore-in "$1" --spore-out "$2" "$3"; }

# The three versions differ only in the presence of pairs of unit post norms.
run --learn-experts --portable-spore --spore-out donor source ' exposure'
for host in lfm-mixed lfm-unit lfm-bare; do
    run --spore-only --spore-in donor --portable-spore --spore-out "import-$host" "$host"
    grep -q 'transfer mode=projected anchors=32 width=64->64 layer=1->2' trace
    test ! -s out
done
bind import-lfm-mixed aligned-copy lfm-aligned
grep -q 'transfer mode=aligned ' trace
bind import-lfm-mixed unit-copy lfm-unit
# Changing the last block leaves all 32 sampled middle-block axes identical.
grep -q 'transfer mode=projected ' trace
"$root/test_portable_spore" check-norm-change import-lfm-mixed import-lfm-unit
cp lfm-mixed edited
"$root/test_portable_spore" set-first edited blk.2.post_ffw_norm.weight
run --spore-only --spore-in import-lfm-mixed --portable-spore --spore-out changed-post edited
grep -q 'transfer mode=projected ' trace
"$root/test_portable_spore" check-norm-change import-lfm-mixed changed-post

run --learn-experts --spore-in import-lfm-mixed --spore-out awake lfm-mixed ' probe'
cp out waking.out
grep '^doe choice ' trace >waking.choice
run --dream-only 2 --spore-in awake --spore-out preview lfm-mixed
test ! -s out
cmp awake preview
grep '^doe dream-choice ' trace >private.choices
for mode in sync async; do
    extra=
    if [ "$mode" = async ]; then extra=--async-consolidation; fi
    run --learn-experts $extra --dream 2 --spore-in import-lfm-mixed --spore-out "$mode" lfm-mixed ' probe'
    cmp waking.out out
    grep '^doe choice ' trace | cmp waking.choice -
    grep '^doe dream-choice ' trace | cmp private.choices -
done
cmp sync async
run --learn-experts --dream-only 2 --spore-in awake --spore-out resumed lfm-mixed
cmp resumed sync
if cmp -s awake sync; then exit 1; fi
echo 'LFM3 parliament: hybrid anchors, optional norms, paired waking and private choices, sync/worker/restart passed'
