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
"$CLANG" --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding \
  -Wall -Wextra \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=1048576 \
  -o "$DIR/qkernels.wasm" "$DIR/qkernels.c"
echo "built: $DIR/qkernels.wasm ($(wc -c < "$DIR/qkernels.wasm") bytes)"
