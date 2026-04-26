#!/usr/bin/env bash
# tests/test_libnotorch_lib.sh — sanity test for `make lib` output.
#
# Verifies that libnotorch.a (built by `make lib`) is a self-sufficient
# static archive: it must contain both notorch.o and gguf.o, and export
# the four GGUF entry points consumers actually call (gguf_open,
# gguf_find_tensor, gguf_get_kv, gguf_dequant) plus the BLAS matvec
# (nt_blas_mmT) and BPE init (nt_bpe_init).
#
# Catch regression: pre-2026-04-26 the lib target only bundled notorch.o,
# so consumers wanting GGUF had to compile gguf.c themselves. yent.aml's
# amlc-driven build hit this and failed at link time.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== test_libnotorch_lib =="

make lib >/tmp/notorch_lib_build.log 2>&1
if [ ! -f libnotorch.a ]; then
    echo "  FAIL: make lib did not produce libnotorch.a"
    cat /tmp/notorch_lib_build.log
    exit 1
fi

REQUIRED=(
    "_gguf_open"
    "_gguf_find_tensor"
    "_gguf_get_kv"
    "_gguf_dequant"
    "_nt_blas_mmT"
    "_nt_bpe_init"
)

NM_OUT=$(nm libnotorch.a 2>/dev/null || true)
MISSING=0
for sym in "${REQUIRED[@]}"; do
    if printf '%s\n' "$NM_OUT" | grep -qE "^[0-9a-f]+ T $sym$"; then
        echo "  PASS [$sym]"
    else
        echo "  FAIL [$sym]: not exported by libnotorch.a"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo
    echo "$MISSING required symbol(s) missing — make lib regression."
    exit 1
fi

echo
echo "libnotorch.a OK ($(wc -c < libnotorch.a) bytes, all symbols present)"
