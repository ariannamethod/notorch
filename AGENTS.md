# notorch — agent work rules

This repository is not a wrapper around PyTorch or llama.cpp. `notorch` is the
pure-C training and inference substrate of the Arianna Method: it owns the
arithmetic it claims, trains real models on constrained machines, and runs
quantized GGUFs through its own code. Treat the Intel laptop, Termux phone, and
24 GB Mac Mini as target machines, not anecdotes. Saving memory or removing a
copy can be the difference between a model existing there and not existing.

Multiple agents and machines work in this repository. Filesystem territory is
a coordination boundary, not an intellectual hierarchy: Codex, Claude, Gemini,
and other routed agents may all act as architects and coauthors in their lanes.
Assume another node may be preparing a change while you inspect the tree.

## Git discipline

- Start by reading `README.md`, the newest entries in `NOTORCHLOG.md`, and the
  relevant implementation and tests. Check `git status` before editing.
- Work on a dedicated branch (`codex/<topic>`, `claude/<topic>`, or another
  agent-specific prefix). Never push directly to `main`. Push the branch and let
  Oleg merge it.
- Preserve unknown or unrelated work. Do not clean, reset, overwrite, or commit
  another agent's files merely because they are present.
- Every commit must have a technical subject/body plus a unique, relevant
  `Quote:` line and a short agent-written `Method:` line. Search the visible
  history before choosing the quote. Keep authorship and coauthorship truthful.
- Do not commit generated binaries, weights, checkpoints, private datasets,
  credentials, or machine-local paths.

## What counts as proof

- Every performance claim names the model or tensor shape, dtype, machine,
  command or harness, comparison target, and repeated measurements. A number
  without those coordinates is not a benchmark.
- Compare like with like: cold with cold, warm with warm, the same prompt,
  context, quantization, thread affinity, output length, and sampling mode.
  Report regressions and failed optimizations as carefully as wins.
- Correctness means a gate capable of going red. Add a red-hand defect when
  practical and verify that the gate catches it before restoring the code.
- Fluent text is not parity. Prefer tokenizer IDs, logits, tensors, and
  byte-identical greedy continuations. `llama.cpp` is a reference for supported
  standard families, not a runtime dependency and not an oracle for the
  Method's own architectures.
- Run the narrow test first, then the relevant suite. Typical entry points are
  `make test`, `make test_js`, `make test_harness`, `make test_tokenizer`, and
  architecture-specific parity scripts. State exactly what ran and what could
  not run on the current machine.

## Architecture and harness rules

- `harness/` is deliberately small: GGUF in, text out. Architecture-specific
  arithmetic belongs in `harness/arch_*.c`; shared loading, tokenizer, runtime,
  sampling, and reporting belong in the common layer.
- A new family should implement the interface in `harness/arch.h`. If adding it
  requires family branches throughout `harness/runtime.c`, first question the
  interface instead of normalizing the branch explosion.
- Preserve the file's declared tokenizer, BOS/EOS policy, RoPE convention,
  tensor layout, expert routing, cache/state semantics, and quantization. Do not
  repair an unfamiliar architecture by silently pretending it is Llama.
- Keep weights packed and memory-mapped when the path promises that property.
  Active MoE parameters are not total resident weights; report RSS, mappings,
  KV/state, scratch, and swap separately.
- The Metal backend is part of `notorch`, not a disposable accelerator shim.
  Any CPU/Metal divergence needs an explicit parity boundary and test.
- DoE/parliament integration must remain separable from plain model inference.
  First prove one body through the minimal harness; then add routing and experts
  without making the simple path lie.

## Scope and project memory

- A bug fix, kernel optimization, test, documentation correction, or small
  harness change gets a dated entry at the top of `NOTORCHLOG.md` with its proof.
- New backends, operation families, training methods, or architecture shifts
  also earn a concise `README.md` update. README is the specification and
  manifesto; `NOTORCHLOG.md` is the engineering record.
- Python and JavaScript bindings may expose `notorch`; they must not secretly
  reimplement its numerical work in another framework.
- When a result is negative, keep the measurement. `notorch` is allowed to say
  that an optimization failed; it is not allowed to call a hope a feature.
