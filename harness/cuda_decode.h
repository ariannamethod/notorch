/* cuda_decode.h — the llama and gemma3 decoders on an NVIDIA GPU.
 *
 * Each is an nt_arch like the CPU family it mirrors, loaded through that family's loader,
 * with forward, forward_residual and forward_multi. Its KV caches live in device memory:
 * a caller takes them from the arch's kv_new and returns them to its kv_free, never from
 * the host kv_new. Built only by the CUDA targets of the Makefile (libnotorch_cuda.a). */
#ifndef NT_HARNESS_CUDA_DECODE_H
#define NT_HARNESS_CUDA_DECODE_H

#include "harness/arch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const nt_arch nt_arch_llama_cuda;
extern const nt_arch nt_arch_gemma3_cuda;

/* The device twin of a CPU family (&nt_arch_llama, &nt_arch_gemma3), or NULL. */
const nt_arch *nt_cuda_arch_for(const nt_arch *cpu);

/* out[n, w->rows] = X[n, w->cols] through w with the decoder's kernels, w and X uploaded
 * for this one call: the entry that checks the device matmul against qmm, not a hot path.
 * Returns NT_OK or NT_E_STATE. */
int nt_cuda_qmm(float *out, const wt *w, const float *X, int n);

#ifdef __cplusplus
}
#endif

#endif
