#!/bin/sh
set -eu
root=$(pwd)
bin=${NT_TEST_HARNESS:-$root/notorch}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
export NT_ATTN_THREADS=1 NT_CHAT='3,4|5|5,5' NT_TOKENS=

run() { "$bin" -q -t 0 -n 1 --experts --expert-trace "$@" </dev/null >out 2>trace; }
reject() {
    if "$bin" -q "$@" </dev/null >out 2>trace; then
        cat trace >&2
        echo 'unexpectedly accepted invalid replay request' >&2
        exit 1
    else rc=$?; fi
    test "$rc" -eq 1
    test ! -s out
    test ! -e refused
    test ! -e refused-episode
    if grep -Eq 'Sanitizer|runtime error:' trace; then cat trace >&2; exit 1; fi
}

# The wrapper is part of the encounter, and repeated stop IDs are one policy.
run --learn-experts --portable-spore --spore-out awake --replay-out episode source ' exposure'
cp out captured.out
run --learn-experts --portable-spore --spore-out no-capture source ' exposure'
cmp awake no-capture
cmp captured.out out

# An independent caller supplies the known fixture IDs and validates receipts.
cat >api.c <<'C'
#include "harness/experts.h"
#include "harness/archs.h"
#include "harness/tokenizer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
static uint64_t counter(const gguf_file *g, const char *key) {
    const gguf_kv *v=gguf_get_kv(g,key);
    assert(v && v->type==10);
    return v->val.u64;
}
int main(void) {
    gguf_file *host=gguf_open("source"); assert(host);
    const nt_arch *arch=nt_pick_arch(host->arch); assert(arch && arch->doe);
    nt_dims dims={0}; void *model=arch->load(host,&dims); assert(model);
    bpe_tokenizer *tok=NULL; int eos=-1;
    assert(!nt_model_tokenizer_open(host,"source",dims.vocab,NULL,&tok,&eos) && tok);
    nt_expert_body body={model,tok};
    nt_experts *p=nt_experts_open_on_body(host,"awake",NULL,&body); assert(p);
    int ids[]={3,4,32,5}, stops[]={5,eos};
    nt_replay_episode *e=nt_replay_capture(p,ids,NULL,4,NULL,0,stops,eos>=0 && eos!=5 ? 2 : 1);
    assert(e && !nt_replay_save(e,"api-episode"));
    assert(!nt_experts_replay(p,e,6,1) && !nt_experts_save_portable(p,"api-replay"));
    nt_replay_free(e); nt_experts_free(p);
    gguf_file *before=gguf_open("awake"), *after=gguf_open("api-replay");
    assert(before && after);
    assert(counter(after,"doe.turns")==counter(before,"doe.turns"));
    assert(counter(after,"doe.elections")==counter(before,"doe.elections"));
    assert(counter(after,"doe.replay.cycles")==1);
    assert(counter(after,"doe.replay.choices")==1);
    assert(counter(after,"doe.replay.prefill")==4);
    assert(counter(after,"doe.replay.elections")==4);
    assert(counter(after,"doe.replay.stops")==1);
    gguf_close(after); gguf_close(before);
    bpe_free(tok); arch->free(model); gguf_close(host);
    return 0;
}
C
${CC:-cc} ${NT_TEST_CFLAGS:--O2 -std=gnu11} -I"$root" -o api api.c \
    "$root/libnotorch_harness.a" "$root/libnotorch.a" ${NT_TEST_LIBS:--lm -pthread}
./api
cmp episode api-episode

# Replay uses the saved policy even when the receiving process has another
# NT_CHAT setting. Its default budget is six; it produces no public text.
NT_CHAT='1|2|3' "$bin" -q --experts --learn-experts --expert-trace \
    --replay-only episode --spore-in awake --portable-spore --spore-out sync source >out 2>trace
test ! -s out
grep -q '^doe replay-start cycle=1 .* positions=4 media=0 limit=6$' trace
grep -q '^doe replay-end cycle=1 .* choices=1 elections=4 stop=1 learn=1$' trace
cmp api-replay sync
NT_CHAT='||' "$bin" -q --experts --learn-experts --expert-trace --async-consolidation \
    --replay-only episode --spore-in awake --portable-spore --spore-out worker source >out 2>trace
test ! -s out
cmp sync worker

# The raw-ID path captures its actual IDs rather than applying the wrapper twice.
NT_TOKENS='3,4,32,5' NT_CHAT='1|2|5' run --learn-experts --portable-spore \
    --spore-out raw-awake --replay-out raw-episode source ' ignored'
cmp episode raw-episode
cmp awake raw-awake

# With no explicit stop, replay traverses the requested budget and restarts.
NT_CHAT='3,4|5|' run --learn-experts --portable-spore --spore-out continuing \
    --replay-out continuing-episode source ' exposure'
NT_CHAT='||' "$bin" -q --experts --learn-experts --expert-trace --replay-only continuing-episode \
    --spore-in continuing --portable-spore --spore-out default-budget source >out 2>trace
test ! -s out
grep -q '^doe replay-end cycle=1 .* choices=6 elections=10 stop=0 learn=1$' trace
run --learn-experts --async-consolidation --replay-only continuing-episode -n 32 \
    --spore-in default-budget --portable-spore --spore-out max-budget source
test ! -s out
grep -q '^doe replay-end cycle=2 .* choices=32 elections=36 stop=0 learn=1$' trace

# Sparse replay exposes only private native routes on the fresh cache.
NT_CHAT='||' run --learn-experts --portable-spore --spore-out sparse-awake \
    --replay-out sparse-episode sparse ' exposure'
run --moe-trace --learn-experts --replay-only sparse-episode --spore-in sparse-awake \
    --spore-out sparse-replayed sparse
test ! -s out
grep -q '^moe replay-route ' trace
if grep -Eq '^(moe route|doe (choice|vote)) ' trace; then exit 1; fi

# Body recall happens before conversion or any replay and requires a local file.
run --spore-only --spore-in awake --spore-out moved --portable-spore wider
run --spore-only --spore-in moved --body-memory awake --spore-out recalled --portable-spore source
test ! -s out
grep -q '^doe body-memory matched=4$' trace
run --learn-experts --replay-only episode --spore-in moved --body-memory awake \
    --spore-out recalled-replay source
test ! -s out
grep -q '^doe body-memory matched=4$' trace

for n in 0 -1 33 2junk 999999999999999999999999999; do
    reject --experts --learn-experts --replay-only episode -n "$n" --spore-in awake --spore-out refused source
done
for t in 0.1 -1 nan inf invalid; do
    reject --experts --learn-experts --replay-only episode -t "$t" --spore-in awake --spore-out refused source
done
reject --learn-experts --replay-only episode --spore-in awake --spore-out refused source
reject --experts --replay-only episode --spore-in awake --spore-out refused source
reject --experts --learn-experts --replay-only episode --spore-out refused source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused source ' exposure'
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused --dream 1 source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused --spore-only source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused --replay-out refused-episode source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused -T source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused -A source
reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused -r 2 source
NT_TOKENS=3 reject --experts --learn-experts --replay-only episode --spore-in awake --spore-out refused source
reject --experts --learn-experts --replay-only episode --spore-in moved --spore-out refused wider
reject --experts --body-memory awake --spore-out refused source ' exposure'
reject --experts --spore-in awake --body-memory moved --spore-out refused source ' exposure'
reject --experts --replay-out refused-episode --spore-out refused source
reject --experts --replay-out refused-episode --spore-out refused -T source ' exposure'
reject --experts --replay-out refused-episode --spore-out refused -A source ' exposure'
reject --replay-out refused-episode --spore-out refused source ' exposure'
reject --experts --replay-out refused-episode --spore-out refused -n 0 source ' exposure'
reject --experts --replay-out refused-episode --spore-out refused -t nan source ' exposure'
reject --experts --learn-experts --replay-only missing-episode --spore-in awake --spore-out refused source
NT_CHAT='||34' reject --experts --replay-out refused-episode --spore-out refused source ' exposure'
cp episode immutable-copy
reject --experts --replay-out episode --spore-out refused source ' exposure'
cmp episode immutable-copy
echo 'Text replay: assembled input, saved stops, API/worker equality, private routes, body recall and refused combinations passed'
