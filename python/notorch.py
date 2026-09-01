"""notorch — Python access to the C library.

    from notorch import Notorch, GGUF

Nothing is computed here. Every kernel call goes straight into libnotorch, and
this file is the thin layer that finds it and describes its types. No numpy, no
build step, no dependencies: ctypes is in the standard library and the .so or
.dylib is what `make shared` produces.

Buffers are plain ctypes arrays, and `memoryview` on one of them is a zero-copy
view any Python code can read. If you want numpy, `numpy.frombuffer` takes them
without copying either — but nothing here requires it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (POINTER, Structure, c_char, c_char_p, c_float, c_int,
                    c_int32, c_uint8, c_uint32, c_uint64, c_void_p)

__all__ = ["Notorch", "GGUF", "GGUF_TYPES", "find_library"]

GGUF_MAX_NAME = 128
GGUF_MAX_TENSORS = 2048
GGUF_MAX_KV = 128

#: GGUF type code -> name. These are the types the packed kernels understand.
GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 8: "Q8_0",
              12: "Q4_K", 14: "Q6_K"}


class _TensorInfo(Structure):
    _fields_ = [("name", c_char * GGUF_MAX_NAME),
                ("ndim", c_uint32),
                ("shape", c_uint64 * 4),
                ("dtype", c_uint32),
                ("offset", c_uint64),
                ("n_elements", c_uint64)]


class _KVValue(ctypes.Union):
    _fields_ = [("u32", c_uint32), ("i32", c_int32), ("f32", c_float),
                ("b", c_uint8), ("str", c_char * 256), ("u64", c_uint64)]


class _KV(Structure):
    _fields_ = [("key", c_char * GGUF_MAX_NAME),
                ("type", c_uint32),
                ("val", _KVValue)]


class _GGUFFile(Structure):
    _fields_ = [("version", c_uint32),
                ("n_tensors", c_uint64),
                ("n_kv", c_uint64),
                ("kv", _KV * GGUF_MAX_KV),
                ("n_kv_parsed", c_int),
                ("tensors", _TensorInfo * GGUF_MAX_TENSORS),
                # kv_end, map_base and map_len were added to gguf_file on the C side and
                # not here, which does not fail loudly: every field after the gap simply
                # reads its neighbour. The C compiler puts data at 443432 and n_layers at
                # 443472 on this platform, and only this field list reproduces both.
                ("kv_end", c_uint64),
                ("data", POINTER(c_uint8)),
                ("data_offset", c_uint64),
                ("data_size", c_uint64),
                ("map_base", c_void_p),
                ("map_len", c_uint64),
                ("n_layers", c_int), ("n_heads", c_int), ("n_kv_heads", c_int),
                ("embed_dim", c_int), ("ffn_dim", c_int), ("vocab_size", c_int),
                ("ctx_len", c_int),
                ("rope_freq_base", c_float), ("rms_eps", c_float),
                ("arch", c_char * 64)]


def find_library(path: str | None = None) -> str:
    """Locate libnotorch. Pass a path, set NOTORCH_LIB, or run `make shared`."""
    if path:
        return path
    env = os.environ.get("NOTORCH_LIB")
    if env:
        return env
    ext = "dylib" if sys.platform == "darwin" else "so"
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, os.pardir, f"libnotorch.{ext}"),
                      os.path.join(here, f"libnotorch.{ext}"),
                      f"libnotorch.{ext}"):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise OSError(
        f"libnotorch.{ext} not found. Build it with `make shared`, or point "
        f"NOTORCH_LIB at it.")


class Notorch:
    """The loaded library, with its signatures declared."""

    def __init__(self, path: str | None = None):
        self.path = find_library(path)
        self.lib = ctypes.CDLL(self.path)
        lib = self.lib

        lib.gguf_open.argtypes = [c_char_p]
        lib.gguf_open.restype = POINTER(_GGUFFile)
        lib.gguf_close.argtypes = [POINTER(_GGUFFile)]
        lib.gguf_close.restype = None
        lib.gguf_find_tensor.argtypes = [POINTER(_GGUFFile), c_char_p]
        lib.gguf_find_tensor.restype = c_int
        lib.gguf_dequant.argtypes = [POINTER(_GGUFFile), c_int]
        lib.gguf_dequant.restype = POINTER(c_float)
        lib.gguf_dequant_row.argtypes = [POINTER(_GGUFFile), c_int, c_uint64,
                                         POINTER(c_float)]
        lib.gguf_dequant_row.restype = c_int

        lib.nt_qmatvec.argtypes = [POINTER(c_float), POINTER(c_uint8), c_int,
                                   POINTER(c_float), c_int, c_int]
        lib.nt_qmatvec.restype = c_int
        lib.nt_qmatvec_i8.argtypes = lib.nt_qmatvec.argtypes
        lib.nt_qmatvec_i8.restype = c_int
        lib.nt_blas_matvec.argtypes = [POINTER(c_float), POINTER(c_float),
                                       POINTER(c_float), c_int, c_int]
        lib.nt_blas_matvec.restype = None
        lib.nt_qmv_set_thread_min.argtypes = [ctypes.c_long]
        lib.nt_qmv_set_thread_min.restype = None

    # ── kernels ─────────────────────────────────────────────────────────────

    def qmatvec(self, out, packed, dtype: int, x, m: int, k: int) -> int:
        """out[m] = W[m,k] @ x[k], weights read straight from their GGUF blocks.

        Returns 0, or -1 if the dtype has no packed kernel for this k.
        """
        return self.lib.nt_qmatvec(
            ctypes.cast(out, POINTER(c_float)),
            ctypes.cast(packed, POINTER(c_uint8)), dtype,
            ctypes.cast(x, POINTER(c_float)), m, k)

    def qmatvec_i8(self, out, packed, dtype: int, x, m: int, k: int) -> int:
        """The int8-activation path: faster on ARM, approximate by construction."""
        return self.lib.nt_qmatvec_i8(
            ctypes.cast(out, POINTER(c_float)),
            ctypes.cast(packed, POINTER(c_uint8)), dtype,
            ctypes.cast(x, POINTER(c_float)), m, k)

    def matvec(self, out, W, x, m: int, n: int) -> None:
        """Dense f32 matvec, through BLAS where the build has one."""
        self.lib.nt_blas_matvec(ctypes.cast(out, POINTER(c_float)),
                                ctypes.cast(W, POINTER(c_float)),
                                ctypes.cast(x, POINTER(c_float)), m, n)

    def set_thread_floor(self, elems: int) -> None:
        """Weight-element count below which a matvec stays single-threaded."""
        self.lib.nt_qmv_set_thread_min(elems)

    def open(self, path: str) -> "GGUF":
        return GGUF(self, path)


class Tensor:
    """One tensor in a GGUF: its shape, its type, and its packed bytes."""

    def __init__(self, gguf: "GGUF", index: int):
        self._gguf = gguf
        self.index = index
        info = gguf._f.contents.tensors[index]
        self.name = info.name.decode()
        self.ndim = info.ndim
        # GGML stores dims innermost-first; row-major is what callers expect.
        self.shape = tuple(reversed([info.shape[i] for i in range(info.ndim)]))
        self.dtype = info.dtype
        self.dtype_name = GGUF_TYPES.get(info.dtype, str(info.dtype))
        self.n_elements = info.n_elements
        self._offset = info.offset

    @property
    def packed(self):
        """A ctypes view of this tensor's bytes inside the file, no copy."""
        base = ctypes.cast(self._gguf._f.contents.data, c_void_p).value
        return ctypes.cast(base + self._offset, POINTER(c_uint8))

    def dequant(self):
        """Dequantize the whole tensor to f32. Allocates n_elements floats."""
        ptr = self._gguf._nt.lib.gguf_dequant(self._gguf._f, self.index)
        if not ptr:
            raise RuntimeError(f"gguf_dequant failed for {self.name}")
        buf = (c_float * self.n_elements)()
        ctypes.memmove(buf, ptr, self.n_elements * 4)
        return buf

    def dequant_row(self, row: int, out=None):
        """One row, decoded in place of the whole tensor — an embedding lookup."""
        cols = self.shape[-1]
        if out is None:
            out = (c_float * cols)()
        rc = self._gguf._nt.lib.gguf_dequant_row(
            self._gguf._f, self.index, row, ctypes.cast(out, POINTER(c_float)))
        if rc != 0:
            raise ValueError(
                f"gguf_dequant_row({self.name}, row={row}) refused: the row is "
                f"not a whole number of blocks, or it is past the end")
        return out

    def __repr__(self):
        return f"<Tensor {self.name} {self.shape} {self.dtype_name}>"


class GGUF:
    """An open GGUF file. Use as a context manager, or call close()."""

    def __init__(self, nt: Notorch, path: str):
        self._nt = nt
        self._f = nt.lib.gguf_open(path.encode())
        if not self._f:
            raise OSError(f"gguf_open failed: {path}")
        c = self._f.contents
        self.arch = c.arch.decode()
        self.n_layers, self.n_heads = c.n_layers, c.n_heads
        self.n_kv_heads, self.embed_dim = c.n_kv_heads, c.embed_dim
        self.ffn_dim, self.vocab_size, self.ctx_len = c.ffn_dim, c.vocab_size, c.ctx_len
        self.rope_freq_base, self.rms_eps = c.rope_freq_base, c.rms_eps
        self._tensors = {}
        for i in range(int(c.n_tensors)):
            t = Tensor(self, i)
            self._tensors[t.name] = t

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._f:
            self._nt.lib.gguf_close(self._f)
            self._f = None

    def __getitem__(self, name: str) -> Tensor:
        return self._tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __len__(self) -> int:
        return len(self._tensors)

    def tensors(self):
        return self._tensors.values()

    def __repr__(self):
        return (f"<GGUF {self.arch} L={self.n_layers} E={self.embed_dim} "
                f"V={self.vocab_size} tensors={len(self._tensors)}>")
