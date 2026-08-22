"""test_binding.py — the binding must describe the library, not resemble it.

Two gates. First, every struct offset ctypes believes in is compared against
what the C compiler actually laid out (tests/gguf_layout.c prints it): a
Structure whose fields have drifted reads its neighbours and reports them as
data, which is the kind of wrong that looks right. Second, values read through
the binding are compared against the same values read by a C program.

    cc -O2 -I. tests/gguf_layout.c -o /tmp/gguf_layout && /tmp/gguf_layout > /tmp/layout.txt
    python3 python/test_binding.py /tmp/layout.txt [model.gguf]
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notorch import GGUF_TYPES, Notorch, _GGUFFile, _KV, _TensorInfo  # noqa: E402

fails = 0


def check(name, got, want):
    global fails
    if got != want:
        print(f"  {name}: binding says {got}, C says {want}  FAIL")
        fails += 1


layout_path = sys.argv[1] if len(sys.argv) > 1 else None
if not layout_path or not os.path.exists(layout_path):
    print("usage: test_binding.py layout.txt [model.gguf]")
    print("  layout.txt comes from tests/gguf_layout.c — see the docstring")
    sys.exit(2)

want = {}
for line in open(layout_path):
    key, value = line.rsplit(" ", 1)
    want[key.strip()] = int(value)

off = lambda s, f: getattr(s, f).offset  # noqa: E731

check("sizeof tensor_info", ctypes.sizeof(_TensorInfo), want["tensor_info"])
for field in ("name", "ndim", "shape", "dtype", "offset", "n_elements"):
    check(f"tensor_info.{field}", off(_TensorInfo, field), want[f"tensor_info.{field}"])
check("sizeof kv", ctypes.sizeof(_KV), want["kv"])
for field in ("type", "val"):
    check(f"kv.{field}", off(_KV, field), want[f"kv.{field}"])
check("sizeof file", ctypes.sizeof(_GGUFFile), want["file"])
for field in ("n_tensors", "kv", "tensors", "data", "data_size", "n_layers",
              "rope_freq_base", "arch"):
    check(f"file.{field}", off(_GGUFFile, field), want[f"file.{field}"])

if fails == 0:
    print("layout  every struct offset matches what the C compiler produced  PASS")

model = sys.argv[2] if len(sys.argv) > 2 else None
if model and os.path.exists(model):
    nt = Notorch()
    with nt.open(model) as g:
        print(f"model   {g!r}")
        if g.n_layers <= 0 or g.embed_dim <= 0 or not g.arch:
            print(f"  metadata came back empty: arch={g.arch!r} L={g.n_layers} "
                  f"E={g.embed_dim}  FAIL")
            fails += 1
        # A tensor the kernels can actually consume, dequantized two ways: the
        # whole thing, and one row on its own. They must agree exactly — one is
        # a slice of the other, not an approximation of it.
        picked = None
        for t in g.tensors():
            if t.dtype in (2, 6, 8, 12, 14) and len(t.shape) == 2 and t.shape[0] > 4:
                picked = t
                break
        if picked is None:
            print("  no quantized 2-D tensor in this file  FAIL")
            fails += 1
        else:
            cols = picked.shape[-1]
            full = picked.dequant()
            row = picked.dequant_row(3)
            bad = sum(1 for i in range(cols) if full[3 * cols + i] != row[i])
            if bad:
                print(f"  {picked.name}: dequant_row differs from dequant in "
                      f"{bad}/{cols} values  FAIL")
                fails += 1
            else:
                print(f"row     {picked.name} {picked.dtype_name}: one row equals "
                      f"its slice of the whole tensor  PASS")

            # The packed kernel against the dequantized reference, the same
            # comparison tests/test_qmatvec.c makes on the C side.
            m, k = picked.shape[0], picked.shape[1]
            m = min(m, 64)
            x = (ctypes.c_float * k)()
            for i in range(k):
                x[i] = ((i * 37) % 101) / 101.0 - 0.5
            got = (ctypes.c_float * m)()
            rc = nt.qmatvec(got, picked.packed, picked.dtype, x, m, k)
            if rc != 0:
                print(f"  nt_qmatvec returned {rc} for {picked.dtype_name}  FAIL")
                fails += 1
            else:
                dense = (ctypes.c_float * (m * k))()
                for i in range(m * k):
                    dense[i] = full[i]
                ref = (ctypes.c_float * m)()
                nt.matvec(ref, dense, x, m, k)
                worst = max(abs(ref[i] - got[i]) for i in range(m))
                scale = max(abs(ref[i]) for i in range(m)) or 1.0
                if worst / scale > 1e-3:
                    print(f"  qmatvec vs dequant+matvec: rel {worst/scale:.2e}  FAIL")
                    fails += 1
                else:
                    print(f"kernel  nt_qmatvec matches dequant+matvec, rel "
                          f"{worst/scale:.1e}  PASS")
elif model:
    print(f"model file not found: {model}")
    fails += 1

print("NOTORCH_PY_OK" if fails == 0 else f"{fails} FAILED")
sys.exit(0 if fails == 0 else 1)
