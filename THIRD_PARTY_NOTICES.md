# Third-party notices

## PCG32 sampling

The PCG32 XSH-RR transition, two-step seeding, and bounded-rejection algorithm
in `notorch.c` adapt the [PCG minimal C implementation](https://github.com/imneme/pcg-c-basic)
by Melissa O'Neill.

Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>

These portions are available under the Apache License, Version 2.0. The full
license is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Upstream provides
the work on an “AS IS” basis without warranties or conditions of any kind,
either express or implied; the license sets out the applicable permissions
and limitations.

NoTorch modifications fix the sequence to 54, expose caller-owned `uint64_t`
state, handle NULL inputs, preserve state and output on rejected checked
calls, and integrate floating-point and categorical sampling. The categorical
weight calculation and NoTorch API integration are Arianna Method code.

PCG is compiled into NoTorch and introduces no separate runtime library.
The surrounding NoTorch source retains its declared LGPL-3.0-or-later license.

## SentencePiece Unigram inference

The deterministic normalization and optimized Unigram Viterbi algorithms in
`sentencepiece.c` adapt [SentencePiece v0.2.2](https://github.com/google/sentencepiece/tree/e0cce7d37b065b5140349dbe12c6bcf6192fdd78),
especially `src/normalizer.cc`, `src/unigram_model.cc`, and the processor's
consecutive-unknown-piece fusion. Copyright 2016 Google Inc. These portions
are available under the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt).

The compiled normalizer's unit layout and XOR traversal adapt Darts-clone by
Susumu Yata, Copyright 2008–2011. Its full three-clause BSD license is preserved
in [LICENSES/Darts-clone-BSD-3-Clause.txt](LICENSES/Darts-clone-BSD-3-Clause.txt).

NoTorch adds a bounded C protobuf reader, immutable owned model storage, a
sparse vocabulary trie, checked UTF-8/normalizer validation, exact final-output
limits, per-call scratch, and transactional result publication. These compiled
implementations introduce no separate runtime library. Reference generation
uses SentencePiece as an external development oracle.
