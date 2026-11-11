#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_portable_spore" fixtures
NT_ATTN_THREADS=1 "$root/test_expert_vote"
