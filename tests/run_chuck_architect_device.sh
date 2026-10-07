#!/bin/sh
# CPU: genuine scalar training; host: separate-buffer host mirror emulation;
# cuda: genuine CUDA kernels, forward/backward and Chuck (requires a device).
# host-mutations: deliberately omit first-moment sync/invalidation in temporary
# core copies and require the same host gate to catch each defect.
# CUDA resource budget: 512-parameter linear body, eight steps; four 257-element
# optimizer trajectories, forty steps each. Gate tensors occupy under 64 KiB;
# CUDA context/cuBLAS residency is hardware/toolkit dependent and is measured
# on the target machine. A 4 GiB device is ample for this bounded gate.
# Override CC/NVCC as usual. CHUCK_DEVICE_OUTPUT retains binaries/logs outside Git.
set -eu

mode=${1:-cpu}
case "$mode" in cpu|host|cuda|host-mutations) ;; *)
    echo 'usage: sh tests/run_chuck_architect_device.sh cpu|host|cuda|host-mutations' >&2
    exit 2;; esac
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
compiler=${CC:-cc}
if [ -n "${CHUCK_DEVICE_OUTPUT:-}" ]; then
    output=$CHUCK_DEVICE_OUTPUT
    mkdir -p "$output"
else
    output=$(mktemp -d "${TMPDIR:-/tmp}/notorch-chuck-device.XXXXXX")
    trap 'rm -rf "$output"' EXIT HUP INT TERM
fi
core_source=${CHUCK_CORE_SOURCE:-"$repo_root/notorch.c"}

build_host() {
    "$compiler" -O2 -std=gnu11 -pthread -ffunction-sections -fdata-sections \
        -DUSE_CUDA -DCHUCK_HOST_EMULATION -I"$repo_root" \
        "$1" "$repo_root/tests/test_chuck_architect_device.c" \
        "$repo_root/tests/chuck_device_host_backend.c" -Wl,--gc-sections -lm -o "$2"
}

if [ "$mode" = host ]; then
    build_host "$core_source" "$output/chuck-device-host"
    "$output/chuck-device-host"
elif [ "$mode" = cpu ]; then
    "$compiler" -O2 -std=gnu11 -pthread -I"$repo_root" "$core_source" \
        "$repo_root/tests/test_chuck_architect_device.c" -lm -o "$output/chuck-device-cpu"
    "$output/chuck-device-cpu"
elif [ "$mode" = host-mutations ]; then
    build_host "$core_source" "$output/chuck-device-host"
    "$output/chuck-device-host" > "$output/host-baseline.log" 2>&1
    for defect in missing-moment-download missing-moment-invalidation; do
        python3 - "$core_source" "$output/$defect.c" "$defect" <<'PY'
import pathlib, sys
source, destination, defect = sys.argv[1:]
text = pathlib.Path(source).read_text()
start = text.index('        if (!chuck_done_gpu) {')
end = text.index('\n    }\n#ifdef USE_CUDA', start)
region = text[start:end]
needle = ('nt_tensor_ensure_cpu(as->m);' if defect == 'missing-moment-download'
          else 'nt_tensor_mark_cpu_dirty(as->m);')
if region.count(needle) != 1:
    raise SystemExit('refusing mutation: expected one exact Chuck fallback site')
region = region.replace(needle, '(void)as; /* deliberate host/device coherence defect */')
pathlib.Path(destination).write_text(text[:start] + region + text[end:])
PY
        build_host "$output/$defect.c" "$output/$defect"
        status=0
        "$output/$defect" > "$output/$defect.log" 2>&1 || status=$?
        if [ "$status" -ne 1 ]; then
            echo "CHUCK_DEVICE_MUTATION_FAIL defect=$defect exit=$status" >&2
            cat "$output/$defect.log" >&2
            exit 1
        fi
        echo "CHUCK_DEVICE_MUTATION_CAUGHT defect=$defect exit=$status backend=HOST_EMULATION"
    done
else
    nvcc=${NVCC:-nvcc}
    if ! command -v "$nvcc" >/dev/null 2>&1; then
        echo 'CHUCK_DEVICE_SKIPPED backend=CUDA reason=nvcc_unavailable'
        exit 77
    fi
    "$nvcc" --version
    # No external BLAS dependency: the same CUDA binary retains scalar CPU as
    # its comparison arm. nvcc supplies CUDA runtime paths at the link step.
    "$nvcc" -O2 -DUSE_CUDA -c "$repo_root/notorch_cuda.cu" -o "$output/notorch_cuda.o"
    "$compiler" -O2 -std=gnu11 -pthread -DUSE_CUDA -I"$repo_root" \
        -c "$core_source" -o "$output/notorch-device.o"
    "$compiler" -O2 -std=gnu11 -pthread -DUSE_CUDA -I"$repo_root" \
        -c "$repo_root/tests/test_chuck_architect_device.c" -o "$output/test-device.o"
    "$nvcc" "$output/notorch-device.o" "$output/test-device.o" "$output/notorch_cuda.o" \
        -lcublas -Xcompiler -pthread -lm -o "$output/chuck-device-cuda"
    "$output/chuck-device-cuda"
fi
