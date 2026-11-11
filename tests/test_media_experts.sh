#!/bin/sh
set -eu
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
"$root/test_lfm3" fixtures
"$root/test_portable_spore" tokenizer lfm3
NT_ATTN_THREADS=1 "$root/test_media_experts"
