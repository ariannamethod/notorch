#!/bin/sh
# Rebuild origin/main and the working branch independently before comparison.
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
parity_tmp=$(mktemp -d "${TMPDIR:-/tmp}/notorch-spa-parity.XXXXXX")
trap 'rm -rf "$parity_tmp"' EXIT HUP INT TERM
compiler=${CC:-cc}
baseline=$(git -C "$repo_root" rev-parse "${SPA_LEGACY_REF:-origin/main}")
mkdir "$parity_tmp/base"
# Extract only build inputs; no second checkout and no writes to tracked files.
git -C "$repo_root" archive "$baseline" notorch.c notorch.h chuck_architect.h chuck_architect_impl.h | tar -x -C "$parity_tmp/base"
"$compiler" -O2 -std=gnu11 -pthread -I"$parity_tmp/base" \
    "$parity_tmp/base/notorch.c" "$repo_root/tests/test_spa_legacy_parity.c" \
    -lm -o "$parity_tmp/legacy"
"$compiler" -O2 -std=gnu11 -pthread -I"$repo_root" \
    "$repo_root/notorch.c" "$repo_root/tests/test_spa_legacy_parity.c" \
    -lm -o "$parity_tmp/current"
"$parity_tmp/legacy"
"$parity_tmp/current"
"$parity_tmp/legacy" --trace > "$parity_tmp/legacy.bin"
"$parity_tmp/current" --trace > "$parity_tmp/current.bin"
cmp "$parity_tmp/legacy.bin" "$parity_tmp/current.bin"
digest=unavailable
if command -v sha256sum >/dev/null 2>&1; then
    digest=$(sha256sum "$parity_tmp/current.bin" | cut -d ' ' -f 1)
elif command -v shasum >/dev/null 2>&1; then
    digest=$(shasum -a 256 "$parity_tmp/current.bin" | cut -d ' ' -f 1)
fi
printf 'SPA_LEGACY_PARITY_OK steps=1024 bytes=114688 baseline=%s sha256=%s\n' "$baseline" "$digest"
