/* arch.h — the one interface a model family implements.
 *
 * Adding a family is adding a file next to arch_llama.c and one line to the
 * table in main.c. If a family cannot be added without also editing runtime.c
 * or the forward of another family, this interface is lying and it is the
 * interface that gets fixed, not the family.
 *
 * `names` is matched against the GGUF's general.architecture. NULL means the
 * fallback: the family that takes a file no other family claims. There is
 * exactly one of those, and today it is llama — which is what the reference
 * example already did with every architecture it had never heard of. */
#ifndef NT_HARNESS_ARCH_H
#define NT_HARNESS_ARCH_H

#include "harness/runtime.h"

/* What the caller needs to size its buffers, filled by load. */
typedef struct {
    int n_layers, kv_dim, vocab;
} nt_dims;

typedef struct {
    const char *const *names;    /* NULL-terminated, or NULL for the fallback */
    void *(*load)(gguf_file *gf, nt_dims *dims);
    void  (*free)(void *model);
    /* One forward for a group of consecutive positions. Decode calls it with
     * n = 1; prefill calls it with a chunk of the prompt. `logits` may be NULL,
     * which asks for the KV cache alone, and when it is not NULL it is filled
     * for the LAST row only — the only position anybody samples from. */
    void  (*forward)(void *model, kv_cache *kv, const int *tokens, int n,
                     int pos0, float *logits);
} nt_arch;

extern const nt_arch nt_arch_llama;
extern const nt_arch nt_arch_gemma4;

#endif
