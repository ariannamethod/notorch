# Native SentencePiece Unigram

`sentencepiece.c/.h` load an original binary ModelProto directly into NoTorch.
The file supplies the vocabulary, float32 scores, piece types, and compiled
normalization map. Deterministic encoding returns normalized UTF-8 plus token
IDs and byte spans. Consecutive unknown characters form one surface piece.

Haiku uses the original 650-piece model from
[`harmonix` at `abb878c`](https://github.com/ariannamethod/harmonix/tree/abb878c52d763b73e5ad7a4d6a68a9ea7a248a39/haiku).
Its 256,534-byte ModelProto has SHA-256
`b8fb56e049977498be0f59586534e67b43b23fd1d7a2efd06cc6c62d9213b942`.
The embedded `nfkc_cf` map is 247,028 bytes. This map preserves `ß`, folds Greek
final sigma, and preserves tabs/newlines. The AML tokenizer applies Python-lineage
Unicode lowercase before native encoding, then removes every `▁` and empty
piece. Those Haiku choices remain in AML.

## API and ownership

```c
#include "sentencepiece.h"
#include <stdio.h>

char error[160];
nt_spm_model *model = nt_spm_load("haiku_sp.model", error, sizeof(error));
nt_spm_result result = {0};
if (model && nt_spm_encode(model, "cloud remembers", 15,
                          &result, error, sizeof(error)) == 0) {
    for (size_t i = 0; i < result.count; ++i) {
        nt_spm_piece piece = result.pieces[i];
        fwrite(result.normalized + piece.offset, 1, piece.length, stdout);
    }
}
nt_spm_result_free(&result);
nt_spm_free(model);
```

`nt_spm_load_memory` copies borrowed ModelProto bytes. File loading owns its
contents after the file is removed. Models stay immutable through concurrent
encodes; the caller keeps a model alive until every call returns. Each result
owns its normalized buffer and spans and remains valid after model destruction.
All scratch belongs to the call. Failure preserves the caller's result object;
success publishes the complete result. Free a previous successful result before
reusing its destination.

The supported inference profile is UNIGRAM with at least one NORMAL piece,
exactly one UNKNOWN piece, and optional CONTROL, USER_DEFINED, and UNUSED
entries. Longest user-defined strings bypass normalizer rules. All three
normalizer whitespace flags and `treat_whitespace_as_suffix` are honored.
Control IDs are omitted from normal text matching. Encoding inserts no BOS/EOS.
Denormalizer metadata concerns decoding and is retained as model data.

Unsupported model algorithms and byte fallback produce load errors. Malformed
protobuf, nonfinite scores, duplicate entries, invalid UTF-8 vocabulary,
invalid compiled-map references, and missing unknown/normal pieces also fail
loading. Limits are 64 MiB of ModelProto, 1,048,576 pieces, 7,999 bytes per piece,
and 1 MiB each for input and final normalized output. A removed trailing run
can exceed the final-output limit transiently without allocating that run.
Input NUL is rejected; malformed UTF-8 bytes become U+FFFD one byte at a time.
Allocation failures return an explicit error and release all partial state.

## Arithmetic and normalization

The native oracle is [SentencePiece 0.2.2](https://github.com/google/sentencepiece/tree/e0cce7d37b065b5140349dbe12c6bcf6192fdd78).
Viterbi visits starting UTF-8 positions in order and keeps the first path at
an exact score tie. Scores accumulate in float32. An unknown scalar receives
the minimum NORMAL score minus 10; a user-defined symbol receives
`(float)(0.1 * (byte_length - 1))`. Active frontier scores are recentered when
the current score crosses ±100,000. After backtracking, adjacent UNKNOWN spans
are fused while retaining their normalized surface.

The embedded Darts map supplies longest-prefix normalization. Its full array
bounds, replacement offsets, NUL terminators, and UTF-8 boundaries are checked
before publication. Normalization first determines the final output size;
its writing pass retains only that prefix and trims the same trailing symbols.
This keeps the exact cap valid when a dummy prefix temporarily enlarges text
whose trailing `▁` will be removed.

## Reproduce

```sh
make lib shared BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0
make check_sentencepiece SPM_MODEL=/path/to/haiku_sp.model
python3 tests/test_sentencepiece_mutations.py
```

The normal native gate needs C and pthreads. Its 106 embedded cases cover
Unigram ties, unknown runs, overlapping user symbols, UNUSED entries, all
16 whitespace/suffix configurations, and long-score recentering. Supplying
`SPM_MODEL` adds 36 pinned original-Haiku cases. Omitting it prints a SKIP for
that external model corpus while running every embedded case.

The complete gate passes 21,689 checks on Linux x86_64 / GCC 13.3, including
8,000 concurrent encodes through one model, final-size boundaries, missing
files, NUL/UTF-8 handling, and 2,048 deterministic corrupted-model probes.
The separate fault gate passes 462 checks over all five loader and three
encoder allocation sites, output preservation, released state, and allocation
end canaries. ASan/UBSan pass both gates with `ASAN_OPTIONS=detect_leaks=0`;
the host blocks LeakSanitizer's process inspection.

Four isolated red-hand mutations fail: changing strict score ties to `>=`,
bypassing the compiled normalizer, splitting unknown runs, and publishing a
failed result. The mutation script uses temporary copies and leaves the
checkout untouched.

Reference regeneration needs development-only `sentencepiece==0.2.2`:

```sh
python3 tests/reference_sentencepiece.py \
  --haiku-model /path/to/haiku_sp.model \
  --output tests/sentencepiece_reference.h
```

The generator authenticates the model SHA-256 and constructs small compiled
normalizer fixtures before asking SentencePiece for normalized bytes, IDs,
and piece strings. The generated C oracle is committed; no Python package is
needed to run it. Attribution is in [third-party notices](../THIRD_PARTY_NOTICES.md).
