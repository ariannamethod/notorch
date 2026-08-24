#!/bin/sh
# Build the SIMD kernels to wasm. Needs clang with a wasm32 target and wasm-ld;
# on macOS that is llvm + lld from Homebrew, on Linux the distro's clang/lld.
#
#   ./build.sh            # writes qkernels.wasm next to this script
#
# The result is checked in: nobody should need a toolchain to use notorch.js.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
CLANG=${CLANG:-$(command -v clang)}
for c in /opt/homebrew/opt/llvm@21/bin/clang /opt/homebrew/opt/llvm/bin/clang; do
  [ -x "$c" ] && CLANG=$c && break
done
for d in /opt/homebrew/opt/lld@21/bin /opt/homebrew/opt/lld/bin; do
  [ -d "$d" ] && PATH="$d:$PATH"
done
export PATH
# The memory is imported and shared, which is what lets the host put a whole
# model inside it: the weights are read straight into this address space, so a
# matvec names its matrix by byte offset instead of copying it in, and the same
# SharedArrayBuffer is what a worker pool already wants. --max-memory is the
# ceiling the host's Memory must fit under; 2 GB is half of what wasm32 can
# address and more than any model that belongs in a browser tab.
"$CLANG" --target=wasm32 -O3 -msimd128 -matomics -mbulk-memory \
  -nostdlib -ffreestanding \
  -Wall -Wextra \
  -Wl,--no-entry -Wl,--export-dynamic \
  -Wl,--import-memory -Wl,--shared-memory \
  -Wl,--initial-memory=1048576 -Wl,--max-memory=2147483648 \
  -Wl,--export=__heap_base \
  -o "$DIR/qkernels.wasm" "$DIR/qkernels.c"
echo "built: $DIR/qkernels.wasm ($(wc -c < "$DIR/qkernels.wasm") bytes)"
