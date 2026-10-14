# Immutable PEN7 inference

`pen7.h` exposes frozen Penelope/NanoJanus bodies through `libnotorch.a`.
`pen7.c` owns their loaded bytes, tokenizer, lifetime and workspaces; the
arithmetic stays in `harness/arch_penelope.c`, shared with the ordinary harness.
The API uses no autodiff tape, optimizer, RNG, sampling policy or global
sequence. AML owns fallback, lexical decisions and each Haiku's history.

## Lifetime and identity

`nt_pen7_load(path, &model)` reads one regular GGUF file into an owned snapshot,
validates its complete PEN7 schema and expands Q8_0 matrices through
`gguf_dequant`. F32 tensors borrow the snapshot. It publishes the model only
when weights, geometry, tokenizer metadata, token surfaces, merge strings and
merge IDs agree. Missing or malformed files return a status without writing
stdout/stderr or changing the caller's output handle.

The model identity is the lowercase SHA-256 of **every byte of that exact
snapshot**, including metadata. The tokenizer identity is the SHA-256 of its
little-endian int32 merge-pair array. The accepted tokenizer contract fixes
`byte-rank-pass-v1`, ASCII lowercase, raw byte IDs 0..255, an 8191-byte input
limit and no automatically added BOS/EOS. The declared tokenizer hash must
match the loaded array, and its complete redundant surfaces are validated.
Path replacement, in-place writes and unlink after loading cannot change the
model or either identity.

Handles have atomic retain/release. Every caller must hold a live reference
while using a model; a retain cannot race the last release. Read-only model
operations can run concurrently. `nt_pen7_get_info` and
`nt_pen7_token_piece` return borrowed immutable views valid for model life.
Token pieces are exact bytes: NUL and partial UTF-8 sequences are representable.

`nt_pen7_encode` accepts an explicit byte count, performs each merge rank once
left to right over the entire lowercased input, and returns all token IDs or a
capacity error. Empty input encodes to zero IDs. It never silently truncates.
Outputs and their returned count stay unchanged on refusal.

## Workspaces and parameter mixtures

Each workspace retains its creation model and owns its KV cache, activation
scratch, result staging and a single matrix-sized mixture buffer. It accepts
compatible model handles. Each forward starts at learned position zero and
reads precisely the supplied sequence; previous calls leave no sequence
history. Simultaneous forwards require different workspaces. Reusing a
workspace for another body, context or alpha needs no reset.

`nt_pen7_forward` returns last-token logits and the final RMS-normalized
contextual vector consumed by the output head. Either output can be omitted.
At least one is required. Requested outputs are staged and checked before
publication, including rejection of RMS-accumulator overflow which would
otherwise become finite zeros. Invalid IDs, geometry, capacities, overlapping
output ranges, coefficients or numerical results preserve both outputs.

`nt_pen7_mix_forward(A, B, alpha, ...)` evaluates parameters
`alpha*A + (1-alpha)*B`. Every matrix, token embedding, learned position,
normalization weight and two-logit resonance gate participates. It mixes one
matrix into reusable scratch and applies that matrix to all current rows
before reusing the buffer. No third full model is retained. Alpha must be
finite in [0,1]; endpoints use their exact source. All dimensions and tokenizer
identities must agree even at endpoints.

A file snapshot and total expanded parameter count are each bounded at
512 MiB; decoded token surfaces at 32 MiB; a workspace at 256 MiB. With the
published 2048-vocabulary, 448-width, eight-layer Q8 bodies on Linux x86-64,
a 32-token workspace occupies **5,187,424 bytes**, and the full 256-token
workspace **15,738,720 bytes**. Each source retains its 21,308,576-byte GGUF
snapshot and its Q8-to-F32 expansions; those costs are shared by retained
model references. Workspace bytes are the requested object and float-buffer
allocations; the shared attention primitive also allocates at most four bytes
per active token for temporary scores. These numbers report storage, not throughput.

## C consumer

```c
#include "pen7.h"
#include <string.h>

nt_pen7_model *body = NULL;
nt_pen7_workspace *work = NULL;
int ids[256];
size_t count = 0;
float logits[2048], representation[448];
const char *text = "the trees remember the rain";

int rc = nt_pen7_load("haiku1.gguf", &body);
if (!rc) rc = nt_pen7_workspace_new(body, 256, &work);
if (!rc) rc = nt_pen7_encode(body, text, strlen(text), ids, 256, &count);
if (!rc) rc = nt_pen7_forward(body, work, ids, count,
                             logits, 2048, representation, 448);
/* rc is explicit; the caller chooses what an unavailable source means. */
nt_pen7_workspace_free(work);
nt_pen7_release(body);
```

```sh
make lib BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0
cc -O2 -std=gnu11 -pthread -I. consumer.c libnotorch.a -lm -o consumer
```

`make install` includes `pen7.h`. The shared library exports the same API.
The PEN7 family and shared runtime objects now also live in `libnotorch.a`;
other architecture selection and tokenizer/harness interfaces retain the
`libnotorch_harness.a` link surface.

## Executed gates

GCC 13.3.0, Linux x86-64, scalar build (`BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0
ARM_SIMD=0`):

```sh
make -j2 test_pen7 test_penelope BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0
./test_pen7
./test_penelope
python3 tests/test_pen7_mutations.py
./test_pen7 /path/to/haiku1.gguf /path/to/haiku2.gguf
```

The fixture boundary passes **79 checks**: a complete materialized mixed GGUF
matches all logits and representation values exactly, endpoint controls,
context sensitivity, four concurrent workspace owners, immutable file identity,
source overwrite/unlink, lifetime, strict tokenizer, malformed containers, nonblocking nonregular-file refusal
and atomic refusal paths.
The existing family gate passes **121 checks**, including its independent
small-model double-equation goldens and full/chunked contexts.

Both published Q8 sources add real 1/9/33/256-token comparisons: **16,384 logits
and 3,584 representation values** match the rebuilt shared harness exactly.
The representation oracle captures the final contextual residual then applies
the file's output norm. The real pair adds exact endpoint controls, an interior
mixture and unchanged source readouts afterwards. The combined gate passes
**130 checks**. On the fixed twelve-token context from
`The trees remember the rain.`, alpha .37 differs from source A by maximum
absolute 7.30835009 in logits and 7.66948795 in representation.

Complete source identities:

| Body | Loaded GGUF SHA-256 |
| --- | --- |
| Haiku 1 | `50f159a2c34952bd9e6349880b4b20f8bbc0ac4b1f1dc2c1f6e4768732548b66` |
| Haiku 2 | `16909b8590bd7fbbf15ab11b2dbf3743d612da8c617aca2dda03d640fef13e29` |

Both tokenizer identities are
`664afe692a006a54fc00406209042f6d452748d64a883ecfab9ef66d41f8bdf2`.

The compiled mutation gate detects six defects: removed parameter mixing,
omitted FF normalization mixing, omitted learned-position mixing, accepted
RMS overflow, hashing only half the snapshot, and omitted ASCII lowercase.
The overflow fixture first reproduced a success with finite zero outputs;
the accumulator guard now refuses it with outputs untouched.

For downstream tests, `./test_pen7 --fixtures DIR` writes `a.gguf`, `b.gguf`
and an exact alpha-.37 `m.gguf` into an existing temporary directory. All are
small deterministic F32 fixtures with vocabulary 260, width 8, two layers,
context 40 and four merge ranks. Model binaries are never checked in.

The full scalar `make -j2 test BLAS_FLAGS= BLAS_LIBS= X86_SIMD=0 ARM_SIMD=0`
suite passes. Installed static and shared-library consumers both load a real
body, tokenize a sentence and return finite logits and representation.
ASan/UBSan runs the complete 79-check fixture gate with
`ASAN_OPTIONS=detect_leaks=0`. An explicit leak-scanning attempt terminates
with LeakSanitizer's fatal ptrace restriction; it reports no completed leak result.
