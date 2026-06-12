/* notorch_metal.h — Apple Silicon Metal/MSL backend for notorch.
 *
 * Why this exists: 24B-class quantized models do not fit on a 24GB Mac
 * if weights have to be dequantized to f32 (4× blow-up). The Metal
 * backend keeps Q4_K weights in their packed layout and dequantizes
 * inline inside the shader, one block at a time per output row. The
 * activations and accumulator stay in fp32; only the substrate is
 * read in its native 4.5-bit-per-weight encoding.
 *
 * Phase 1 (this file): correct, naive matvec — one thread per output
 * row, no threadgroup memory sharing, no simdgroup_matrix tiling. The
 * point is to land a working GPU path against which the next round of
 * optimisations can be benchmarked.
 *
 * Phase 2 (planned): tiled threadgroup dispatch with x[] in shared
 * memory; simdgroup reductions for the inner dot; multiple rows per
 * thread; async + double buffering.
 *
 * Build: add `-DUSE_METAL` to CFLAGS, compile notorch_metal.mm with the
 *        Obj-C++ driver and link `-framework Metal -framework Foundation
 *        -lc++`. See Makefile target `metal`.
 *
 * by Claude (Arianna Method)
 */

#ifndef NOTORCH_METAL_H
#define NOTORCH_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if a Metal device is available on this host (Apple Silicon
 * macOS or a discrete GPU on Intel macOS), 0 otherwise. Safe to call
 * before init. */
int nt_metal_available(void);

/* Lazily initialise the Metal backend: pick the default device, build
 * the command queue, compile the kernel library. Idempotent. Returns 0
 * on success; on failure writes a diagnostic to stderr and returns a
 * non-zero code (1=no device, 2=no queue, 3=compile, 4=missing kernel,
 * 5=pipeline). */
int nt_metal_init(void);

/* Release Metal resources. Safe to call without a prior successful
 * init. After shutdown the next API call re-initialises. */
void nt_metal_shutdown(void);

/* Q4_K matrix-vector multiply on the Apple GPU, with inline dequant.
 *
 *   out[i]  =  sum_{j=0..k-1}  dequant_q4_k(W)[i*k + j] * x[j]
 *
 * W is GGML Q4_K-packed, row-major: m rows of (k/256) blocks × 144
 * bytes each. k MUST be a multiple of 256. The host never materialises
 * the f32 weights — they are reconstructed inside the shader from the
 * super-block scale/min (fp16) plus the 6-bit per-subblock scales/mins
 * and the 4-bit nibbles, matching gguf.c:dequant_q4_k exactly.
 *
 * x and out are fp32 in host memory; they are uploaded to GPU shared
 * buffers for the call and read back synchronously. For repeated calls
 * a Phase 2 API will keep weight buffers resident.
 *
 * Returns 0 on success; non-zero on failure (10=k not multiple of 256,
 * 11=buffer alloc, propagated init codes). */
int nt_metal_q4k_matvec(float *out,
                        const uint8_t *W_q4k,
                        const float *x,
                        int m, int k);

/* Q6_K matrix-vector multiply on the Apple GPU, inline dequant (block 210 bytes per 256
 * weights). Mirrors gguf dequant_q6_k / doe.c:pq_q6_k_rows. Same ABI/codes as q4k. */
int nt_metal_q6k_matvec(float *out,
                        const uint8_t *W_q6k,
                        const float *x,
                        int m, int k);

/* Phase 2 — resident weights. Register one base region (e.g. the whole packed
 * GGUF tensor block) as a single zero-copy GPU buffer (Apple unified memory).
 * After this, nt_metal_q4k_matvec binds any W that falls inside
 * [base, base+nbytes) by offset instead of re-uploading it every call — the
 * per-token weight upload disappears. `base` MUST be page-aligned
 * (posix_memalign / mmap). Returns 0 on success, 12 if the NoCopy wrap fails.
 * Calls nt_metal_init if needed. Weights outside the region fall back to upload. */
int nt_metal_register_base(const void *base, uint64_t nbytes);

/* ── Token-graph batch mode ──────────────────────────────────────────
 * Collect multiple matvec dispatches into ONE command buffer with ONE
 * GPU sync at commit. Between begin and commit, nt_metal_q4k_matvec /
 * nt_metal_q6k_matvec ENQUEUE instead of executing: x is consumed at
 * call time (the caller may reuse its memory immediately), but `out`
 * is written only during commit — do not read `out`, and keep its
 * memory alive, until nt_metal_batch_commit returns. Intended for
 * groups of independent matvecs that share an input ({q,k,v},
 * {gate,up}, the whole per-token weight sweep): 1 CPU-GPU sync per
 * group instead of 1 per matvec. The kernels and dispatch geometry are
 * identical to the solo path, so batched results are bit-identical to
 * sequential calls. begin is idempotent; commit without begin is a
 * no-op. Returns 0 on success, 14 if the GPU reports an error (every
 * pending `out` of that batch is then undefined). */
int nt_metal_batch_begin(void);
int nt_metal_batch_commit(void);
int nt_metal_batch_active(void);

#ifdef __cplusplus
}
#endif

#endif /* NOTORCH_METAL_H */
