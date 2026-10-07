# Chuck: Loss Architect — scenario gates

This round follows the first two-body training receipts. It checks the boundaries
that carry an action into a consequence: input state, pending credit, saved lives,
device mirrors and the training runner's failure record.

## Executable boundaries

| Boundary | Required behavior | Gate |
| --- | --- | --- |
| Canonical trajectory | Existing Chuck arithmetic and valid legacy trajectories remain reproducible. | `test_chuck_architect`, `test_chuck_legacy_parity` |
| Gradient and local state | Malformed shapes, non-finite gradient norms/history and invalid ring/counter state are refused before an update. | `test_chuck_actions_edge` |
| Configuration locale | JSON decimal dots have the same meaning under C and comma-decimal caller locales. The caller's locale remains intact. | `test_chuck_architect_state` |
| Pending life | Cached features, hidden state and predictions agree with the pending observation and unchanged policy weights. Contradictory saved state is refused transactionally. | `test_chuck_architect_state` |
| Interrupted experience | Restoring before selection, after execution or after feedback reproduces the continuation. | `test_chuck_architect_scenarios` |
| Multiple lives and modes | Separate lives keep their histories; mode changes respect pending credit and preserve canonical execution. | `test_chuck_architect_scenarios` |
| Action and history readout | Enabled actions bound exploration; a prior consequence can change scores at identical current observation and weights. | `test_chuck_architect_scenarios` |
| CPU/device boundary | CPU fallbacks read current parameters, gradients and moments; later device steps see CPU-written moments. | `run_chuck_architect_device.sh` |
| Divergent consequence | The executed action receives its measured non-finite consequence, writes a valid JSON receipt and saves its acquired life before the run stops. | Training scenario runner's fault gate |
| Alternative actions | Every fork begins with the same body, optimizer, policy and RNG state. Restoring after probes preserves the host continuation. | Two-body scenario runner |

The checked core API validates tape structure, gradient inputs and Chuck history.
Its gradient reduction is reused by the update loop. The original void CPU
legacy entrypoint retains its arithmetic. The public feedback contract still
requires the caller to evaluate the same window as the executed decision.

## CPU checks

```sh
make check_chuck_scenarios BLAS_FLAGS= BLAS_LIBS=
make test_chuck_architect_mutations
make test_chuck_scenario_mutations
make test_chuck_legacy_parity
make test BLAS_FLAGS= BLAS_LIBS=
```

The state-test driver builds a private comma-decimal locale when `localedef` and
locale sources are available. Otherwise it tries installed locales and reports
the unavailable locale case as SKIPPED. It never changes the system locale
archive. Temporal tests use analytical bodies to isolate optimizer chronology;
the real-body experiments exercise the numerical training kernels.

## Device checks

```sh
sh tests/run_chuck_architect_device.sh cpu
sh tests/run_chuck_architect_device.sh host
sh tests/run_chuck_architect_device.sh host-mutations
sh tests/run_chuck_architect_device.sh cuda
```

`host` uses separately allocated CPU and emulated device buffers. Its receipt is
labelled `HOST_EMULATION`; it exercises the host-side mirror protocol and catches
deliberately omitted moment downloads and invalidations. `cuda` compiles
`notorch_cuda.cu`, executes CUDA Chuck kernels and checks a linear-body training
loop with actual GPU forward/backward dispatch. Missing toolkit/device reports
SKIPPED and returns 77. `CHUCK_DEVICE_OUTPUT` retains build products and logs in
an external output directory.

The current workspace has no CUDA toolkit or NVIDIA device. The CUDA command is
ready for a Linux GPU machine. The workload uses small tensors: 257-element
optimizer trajectories and a 512-parameter linear training body. A CUDA devel
image with `nvcc`, cuBLAS, a C compiler and Python is sufficient; an 8 GiB GPU,
four CPU cores, 8 GiB host RAM and a 15-minute run cap cover the planned gate.
The Metal backend supplies inference operations; this training gate targets
notorch's CUDA tape backend.

## Controlled action comparisons

The experiment snapshots the training body, complete tape, optimizer, Chuck,
policy life, training mode and window/noise RNG. Each branch executes one of
`hold`, `brake`, `push` and then follows a fixed `hold` continuation. Horizons
1, 4 and 16 measure the effect of that intervention under the declared
continuation. Current-window loss, common future-window loss and held-out loss
are separate observations.

An independent host run without probes must match the probe-bearing run at every
recorded step and in final state. This checks that studying alternatives leaves
the acquired host life intact. The policy remains the 163-parameter v1 model;
this experiment records the comparative consequences used to design its next
learning objective.
