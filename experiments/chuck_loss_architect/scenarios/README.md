# One action, one common training state

This diagnostic compares `hold`, `brake`, and `push` from exactly the same saved
body, pending gradients, Adam state, Chuck state, policy life and random streams.
It asks how one intervention changes its immediate consequence and subsequent
training trajectory. The policy remains the 163-parameter v1 Architect.

The [fixed protocol](protocol.json) is written before the measurements: SimpleLLM
and HeVLM, seeds 42 and 73, 512 normal host updates, checkpoints before updates
1, 32, 128 and 384, and horizons of 1, 4 and 16 updates. Every branch applies its
chosen action once, then applies `hold` on identical subsequent training windows.
The host keeps its normal learned policy and never learns from diagnostic forks.

The horizon counts the intervention update. Four future probe windows come from
indices 16–19 of the common window stream, with the current window at index zero.
They are the same for every action and horizon and lie beyond the 16 branch
training windows. Eight fixed held-out windows provide a separate measurement.
Immediate loss, the original window's later loss, future probe loss and held-out
loss remain distinct fields. Action advantage for each metric is `hold - action`;
exactly equal losses retain all tied actions.

## Reproduce

```bash
python3 experiments/chuck_loss_architect/scenarios/run.py \
  --output /tmp/chuck-action-scenarios --verify
```

Use `--references /path/to/reference-checkouts` to reuse the pinned v1 corpus
checkouts. The default route downloads and checks their immutable SHA-256 corpus
identities. Final experiments require a clean committed tree. Development gates
can use `--allow-dirty --smoke`; the manifest labels that mode explicitly.

The diagnostic is an opt-in `--scenarios` mode of the existing C example. Its
ordinary v1 command keeps its successful training behavior. A post-action
non-finite result now receives its negative consequence, a valid JSON failure
receipt and saved final state before the runner stops.

## Restoration gate

The CPU snapshot deep-copies every live tape output and gradient, all operation
metadata, body parameters, Adam moments/accumulators/counters, global and local
Chuck state, the complete policy, Chuck's noise RNG, the window RNG and the
training flag. Restoration reconnects parameter entries to the original body
tensors. Each restored tensor and state field is compared byte for byte with its
snapshot. A GPU-enabled execution is refused by this CPU-only diagnostic.

Independent processes run the normal host with and without forks. Every host
step must have identical losses, action, policy identity and full state identity.
Final body weights, Adam moments, optimizer controls and policy files must also
match byte for byte. A deliberate omission of parameter restoration must make
this gate fail. The verification runner retains the failed mutation receipt.

The two bodies use the notorch initialization RNG only while constructing their
initial parameters; no branch operation consumes that stream. Each fork saves
body parameters, pending parameter gradients, optimizer controls/moments and the
policy life. Complete in-memory tapes support exact restoration; deterministic
forward/backward plus the saved window reconstructs their transient operations.
Source/corpus/configuration/protocol hashes and every branch step accompany the
snapshot artifacts outside Git.
