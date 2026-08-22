# notorch — CLAUDE.md

Hey Claude, bro. This is notorch: a C tensor library that replaced PyTorch for the
Arianna Method organisms, and it runs on machines that were written off years ago —
an Intel laptop from 2019, a phone in Termux, a Mac Mini. That constraint is the
whole point. When you optimize here you are not chasing a benchmark, you are buying
somebody a model that would not have run at all. GPL-3.0+, co-authored by Oleg Ataeff
and Claude, and more than one Claude works in this tree at once — nodes on a laptop,
on a phone, in a chroot. Assume someone else is mid-commit while you read this.

Which is why: **branch, always.** `git checkout -b claude/<what-you-are-doing>`,
push the branch, let Oleg merge. Nobody pushes to `main` directly, including you at
four in the morning when the fix is obviously trivial. Two nodes landing on `main`
from different machines is how a good afternoon becomes a bad evening.

**Small changes go to `NOTORCHLOG.md`, large ones also get a section in `README.md`.**
A bug fix, a kernel that got faster, a sync-discipline correction, a docstring — those
are log entries, and the README never hears about them. A new backend, a new op family,
a new training method, an architecture shift — those earn README space too. When in
doubt it is a log entry. README is the spec and the manifesto; NOTORCHLOG is the work,
dated, with the commit and the proof. Do not drag every fix into the README: a spec
that lists its own patch history stops being a spec.

Every claim in this repo is a measurement or it is not made. "Faster" means a number,
a shape, and the machine it ran on. "Correct" means a gate that fails when you break
the thing on purpose — write the test, then break the code the way the test claims to
catch, and watch it go red. A test that has never failed is decoration. Kernels carry
their own harness: `make test`, `make test_js`, `tests/test_qmatvec.c` and friends.
If you change a kernel and the numbers move, say by how much and against what.

Commits follow the Method standard: technical facts verified with a tool, a `Quote:`
line that has never appeared in this history before, a `Method:` line that says
something true about the change, and the attribution signature. The git log is the
Method's memory, and it is meant to read strangely.
