#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
export NT_ATTN_THREADS=1 NT_CTX=32 NT_CHAT='||'
run() { "$root/notorch" -q -t 0 -n 1 --experts --expert-trace "$@" </dev/null >text 2>trace; }
bind() { run --spore-only "$@"; }
learn() { run --learn-experts --async-consolidation "$@"; }
reject() {
    if run "$@"; then return 1; else rc=$?; fi
    test "$rc" -eq 1 && ! grep -Eq 'Sanitizer|runtime error:' trace
}

learn --spore-out exact source ' exposure'
learn --portable-spore --spore-out portable source ' exposure'
bind --spore-in portable --spore-out source-copy source
cmp exact source-copy
bind --spore-in exact --portable-spore --spore-out packed source
cmp portable packed
mv source donor-offline
bind --spore-in portable --spore-out aligned-exact aligned
grep -q 'transfer mode=aligned ' trace
learn --spore-out aligned-trained aligned ' exposure'
cmp aligned-exact aligned-trained
learn --spore-in aligned-exact --spore-out restart aligned ' probe'
learn --spore-in portable --spore-out direct aligned ' probe'
cmp restart direct

run --spore-out fresh wider ' exposure'
bind --spore-in portable --spore-out projected wider
grep -q 'transfer mode=projected .*width=64->96 layer=1->2' trace
if grep -q 'doe vote pos=' trace; then exit 1; fi
"$root/test_portable_spore" check exact fresh projected
learn --spore-in projected --spore-out wider-restart wider ' probe'
learn --spore-in portable --spore-out wider-direct wider ' probe'
cmp wider-restart wider-direct
bind --spore-in portable --portable-spore --spore-out onward wider
bind --spore-in onward --spore-out twice wider
cmp projected twice

for host in rotated norm-change; do
    run --spore-in portable --spore-out "$host-state" "$host"
    grep -q 'transfer mode=projected ' trace
done
cp portable bad-frame
"$root/test_portable_spore" break-frame bad-frame
for mode in 0 1; do
    export NT_GGUF_MMAP=$mode
    reject --spore-in bad-frame --spore-out rejected wider
    reject --spore-in exact --spore-out rejected aligned
done
reject --spore-only --spore-in portable --spore-out rejected short-norm
reject --portable-spore --spore-out rejected flat ' exposure'
reject --spore-only --spore-in portable --spore-out rejected wider ' probe'
reject --learn-experts --spore-only --spore-in portable --spore-out rejected wider
reject --portable-spore aligned ' probe'
reject --spore-in portable --spore-out portable aligned
cmp portable packed
test ! -e rejected

# Zero branch weights make the sampled frames identical in every variant.
# Changing one final-block norm must still select projection rather than copying A/B.
learn --portable-spore --spore-out gemma-spore gemma ' exposure'
bind --spore-in gemma-spore --spore-out gemma-copy gemma-aligned
grep -q 'transfer mode=aligned ' trace
for norm in attn_norm post_attention_norm ffn_norm post_ffw_norm; do
    cp gemma-aligned gemma-edited
    "$root/test_portable_spore" set-first gemma-edited "blk.1.$norm.weight"
    bind --spore-in gemma-spore --spore-out "gemma-$norm" gemma-edited
    grep -q 'transfer mode=projected ' trace
done
echo 'Portable spore: exact, aligned, projected, two-hop, async restart and refusals passed'
