#!/bin/sh
# test_consumer_link.sh — the harness is a library, or it is not.
#
# Installs into a throwaway prefix and builds tests/consumer_link.c against it
# with nothing but -lnotorch_harness -lnotorch and the installed headers. No
# notorch source on the command line: if a symbol a body needs is still trapped
# in main.c, this fails to link, which is the whole point.
#
#   ./harness/test_consumer_link.sh [model.gguf]
#
# The negative case runs too. A link test that would pass without the archive
# is testing nothing, so it also builds without -lnotorch_harness and requires
# that one to fail.
set -eu
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/arianna/weights/nano_arianna_full_resft_2026_07_09/nano_arianna_q8_0.gguf}"
if [ ! -f "$MODEL" ]; then
  echo "consumer  (no model at $MODEL — pass one to run this gate)"
  echo "NOTORCH_CONSUMER_SKIPPED"
  exit 0
fi

PREFIX=$(mktemp -d "${TMPDIR:-/tmp}/nt_consumer.XXXXXX")
trap 'rm -rf "$PREFIX"' EXIT

make -s lib lib_harness >/dev/null
make -s install PREFIX="$PREFIX" >/dev/null

INC="$PREFIX/include/ariannamethod"
for h in harness/arch.h harness/archs.h harness/runtime.h examples/bpe.h gguf.h notorch.h; do
  [ -f "$INC/$h" ] || { echo "consumer  header $h was not installed  FAIL"; echo "NOTORCH_CONSUMER_FAIL"; exit 1; }
done
[ -f "$PREFIX/lib/libnotorch_harness.a" ] || { echo "consumer  libnotorch_harness.a was not installed  FAIL"; echo "NOTORCH_CONSUMER_FAIL"; exit 1; }

# The two archives are the point of this gate, but they still call into whatever
# BLAS the tree was built with, and a consumer has to name it. "Accelerate or
# nothing" was written on a Mac and reached the polygon as
# `undefined reference to cblas_sgemm` — the third gate today to have only ever
# run on one machine. Ask the linker rather than guess: a build made without
# BLAS links fine with no flag at all, and the probe below finds that too.
EXTRA=""
if [ "$(uname)" = "Darwin" ]; then
  EXTRA="-framework Accelerate"
else
  for cand in "-lopenblas" "-lblas" ""; do
    if echo 'int main(void){return 0;}' | ${CC:-cc} -x c - $cand -o /dev/null 2>/dev/null; then
      EXTRA="$cand"; break
    fi
  done
fi

# shellcheck disable=SC2086
if ! ${CC:-cc} -O2 -std=gnu11 -I"$INC" -o "$PREFIX/consumer" tests/consumer_link.c \
       -L"$PREFIX/lib" -lnotorch_harness -lnotorch $EXTRA -lm 2>"$PREFIX/link.err"; then
  echo "consumer  linking against the installed archives  FAIL"
  sed 's/^/  /' "$PREFIX/link.err"
  echo "NOTORCH_CONSUMER_FAIL"
  exit 1
fi
echo "consumer  links with -lnotorch_harness -lnotorch and no notorch source  PASS"

OUT=$("$PREFIX/consumer" "$MODEL" "The capital of France is" 2>/dev/null) || {
  echo "consumer  running the linked binary  FAIL"; echo "NOTORCH_CONSUMER_FAIL"; exit 1
}
case "$OUT" in
  arch=*tokens=*vocab=*argmax=*) echo "consumer  ran a model through the archive: $OUT  PASS" ;;
  *) echo "consumer  unexpected output: $OUT  FAIL"; echo "NOTORCH_CONSUMER_FAIL"; exit 1 ;;
esac

# shellcheck disable=SC2086
if ${CC:-cc} -O2 -std=gnu11 -I"$INC" -o "$PREFIX/consumer_red" tests/consumer_link.c \
     -L"$PREFIX/lib" -lnotorch $EXTRA -lm 2>/dev/null; then
  echo "consumer  it linked WITHOUT -lnotorch_harness — this gate proves nothing  FAIL"
  echo "NOTORCH_CONSUMER_FAIL"
  exit 1
fi
echo "consumer  refuses to link without the harness archive  PASS"

echo "NOTORCH_CONSUMER_OK (3 checks)"
