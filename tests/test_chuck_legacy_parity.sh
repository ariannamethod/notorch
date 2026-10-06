#!/bin/sh
# Rebuild both sides: 6000-step noisy CPU trajectory against the pinned baseline.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
parity_tmp=$(mktemp -d "${TMPDIR:-/tmp}/notorch-chuck-parity.XXXXXX")
trap 'rm -rf "$parity_tmp"' EXIT HUP INT TERM
baseline=097fc062418bc9f29de8f0a89d781608680aef4d
compiler=${CC:-cc}

git -C "$repo_root" show "$baseline:notorch.c" > "$parity_tmp/legacy.c"
"$compiler" -O2 -std=gnu11 -pthread -I"$repo_root" \
    "$parity_tmp/legacy.c" "$repo_root/tests/chuck_legacy_parity.c" \
    -lm -o "$parity_tmp/legacy"
"$compiler" -O2 -std=gnu11 -pthread -I"$repo_root" \
    "$repo_root/notorch.c" "$repo_root/tests/chuck_legacy_parity.c" \
    -lm -o "$parity_tmp/current"
"$parity_tmp/legacy" > "$parity_tmp/legacy.bin"
"$parity_tmp/current" > "$parity_tmp/current.bin"
cmp "$parity_tmp/legacy.bin" "$parity_tmp/current.bin"

digest=unavailable
if command -v sha256sum >/dev/null 2>&1; then
    digest=$(sha256sum "$parity_tmp/current.bin" | cut -d ' ' -f 1)
elif command -v shasum >/dev/null 2>&1; then
    digest=$(shasum -a 256 "$parity_tmp/current.bin" | cut -d ' ' -f 1)
fi
printf 'CHUCK_LEGACY_PARITY_OK steps=6000 baseline=%s sha256=%s\n' "$baseline" "$digest"
