#!/bin/sh
set -eu
root=$(pwd)
bin=${NT_TEST_EXPERIENCE:-$root/doe_experience}
cli=${NT_TEST_HARNESS:-$root/notorch}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
export NT_ATTN_THREADS=1 NT_CHAT='||' NT_TOKENS=
cat >inputs <<'INPUTS'
 exposure
 probe
 river
 glass
 seven
 green
 bread
 music
INPUTS
head -n 4 inputs >first-four
tail -n 4 inputs >last-four
"$cli" -q -n 1 -t 0 --experts --learn-experts --spore-out parent source ' probe' >seed.out 2>seed.trace

for mode in awake dream replay; do
    for worker in sync worker; do
        name=$mode-$worker
        mkdir "$name" "$name-cli" "$name-first" "$name-last"
        "$bin" source - parent - "$mode" "$worker" 1 8 "$name" inputs >"$name.tsv" 2>"$name.trace"
        awk -F '\t' 'NF!=3 || $1!=NR || $2!="state-" NR ".gguf" || $3!~/^[0-9]+$/ {bad=1}
                     END {exit bad || NR!=8}' "$name.tsv"
        extra=
        if [ "$worker" = worker ]; then extra=--async-consolidation; fi
        step=1
        previous=parent
        while IFS= read -r input; do
            if [ "$step" -eq 1 ]; then
                "$cli" -q -n 1 -t 0 --experts --learn-experts $extra --spore-in "$previous" \
                    --replay-out "$name-cli/episode.gguf" --spore-out "$name-cli/wake-$step.gguf" \
                    source "$input" >cli.out 2>cli.trace
            else
                "$cli" -q -n 1 -t 0 --experts --learn-experts $extra --spore-in "$previous" \
                    --spore-out "$name-cli/wake-$step.gguf" source "$input" >cli.out 2>cli.trace
            fi
            if [ "$step" -eq 8 ] && [ "$mode" = dream ]; then
                "$cli" -q --experts --learn-experts $extra --spore-in "$name-cli/wake-$step.gguf" \
                    --dream-only 6 --spore-out "$name-cli/state-$step.gguf" source >cli.out 2>cli.trace
            elif [ "$step" -eq 8 ] && [ "$mode" = replay ]; then
                "$cli" -q --experts --learn-experts $extra --spore-in "$name-cli/wake-$step.gguf" \
                    --replay-only "$name-cli/episode.gguf" --spore-out "$name-cli/state-$step.gguf" \
                    source >cli.out 2>cli.trace
            else cp "$name-cli/wake-$step.gguf" "$name-cli/state-$step.gguf"; fi
            cmp "$name/state-$step.gguf" "$name-cli/state-$step.gguf"
            previous=$name-cli/state-$step.gguf
            step=$((step+1))
        done <inputs
        cmp "$name/wake-8.gguf" "$name-cli/wake-8.gguf"
        cmp "$name/episode.gguf" "$name-cli/episode.gguf"
        test ! -e "$name/wake-7.gguf"

        # Resume at an ordinal that is not a sleep boundary, with the same
        # sealed early encounter and a fresh process/cache for the next block.
        "$bin" source - parent - "$mode" "$worker" 1 4 "$name-first" first-four >split-first.tsv 2>split-first.trace
        "$bin" source - "$name-first/state-4.gguf" "$name-first/episode.gguf" \
            "$mode" "$worker" 5 8 "$name-last" last-four >split-last.tsv 2>split-last.trace
        test ! -e "$name-last/episode.gguf"
        cat split-first.tsv split-last.tsv | cmp "$name.tsv" -
        step=1
        while [ "$step" -le 8 ]; do
            split=$name-first
            if [ "$step" -gt 4 ]; then split=$name-last; fi
            cmp "$name/state-$step.gguf" "$split/state-$step.gguf"
            step=$((step+1))
        done
        cmp "$name/wake-8.gguf" "$name-last/wake-8.gguf"
    done
    cmp "$mode-sync/state-8.gguf" "$mode-worker/state-8.gguf"
    cmp "$mode-sync.tsv" "$mode-worker.tsv"
done

mkdir refused
reject() {
    if "$bin" "$@" >refused.out 2>refused.trace; then
        cat refused.trace >&2; echo 'unexpectedly accepted an invalid experience' >&2; exit 1
    else rc=$?; fi
    test "$rc" -eq 1
    test ! -s refused.out
    test -z "$(ls -A refused)"
    if grep -Eq 'Sanitizer|runtime error:' refused.trace; then cat refused.trace >&2; exit 1; fi
}
for n in 0 -1 4097 2junk 999999999999999999999; do
    reject source - parent - awake sync "$n" 8 refused inputs
done
reject source - parent - awake sync 8 1 refused inputs
reject source - parent - bad sync 1 8 refused inputs
reject source - parent - awake bad 1 8 refused inputs
reject source - parent - awake sync 1 7 refused inputs
reject source - parent - awake sync 1 9 refused inputs
NT_CHAT='3|4|' reject source - parent - awake sync 1 8 refused inputs
NT_TOKENS=3 reject source - parent - awake sync 1 8 refused inputs
printf '\300\257\n' >invalid-utf8
reject source - parent - awake sync 1 1 refused invalid-utf8
printf ' exposure\000hidden\n' >embedded-nul
reject source - parent - awake sync 1 1 refused embedded-nul
awk 'BEGIN {for(i=0;i<513;i++)printf " exposure"; print ""}' >too-long
reject source - parent - awake sync 1 1 refused too-long
reject source - parent missing-episode replay sync 1 8 refused inputs
reject wider - parent awake-sync/episode.gguf replay sync 1 8 refused inputs

# New-path admission preserves existing generations, including a partial block.
cp awake-sync/state-8.gguf expected-final
if "$bin" source - parent - awake sync 1 8 awake-sync inputs >refused.out 2>refused.trace; then exit 1; fi
cmp expected-final awake-sync/state-8.gguf
test ! -s refused.out

# A bad later encounter leaves its valid prefix, never that turn's state.
mkdir partial
printf ' exposure\n' >late-error
cat too-long >>late-error
if "$bin" source - parent - awake worker 1 2 partial late-error >partial.tsv 2>partial.trace; then exit 1; fi
test -s partial/state-1.gguf
cmp partial/state-1.gguf awake-sync/state-1.gguf
test ! -e partial/state-2.gguf
test "$(wc -l <partial.tsv)" -eq 1
echo 'Resident experience: 48 states equal restarted CLI, 48 split-resume states, six episode captures, sync/worker and bounded failures passed'
