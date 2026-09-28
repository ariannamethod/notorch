/* arch.h — the one interface a model family implements.
 *
 * Adding a family is adding a file next to arch_llama.c and one line to the
 * table in archs.c. If a family cannot be added without also editing runtime.c
 * or the forward of another family, this interface is lying and it is the
 * interface that gets fixed, not the family.
 *
 * `names` is matched against the GGUF's general.architecture. Every family
 * names what it actually implements: an unknown architecture is refused
 * rather than sent through arithmetic that merely looks similar. */
#ifndef NT_HARNESS_ARCH_H
#define NT_HARNESS_ARCH_H

#include "harness/runtime.h"

/* What the caller needs to size its buffers, filled by load. */
typedef struct {
    int n_layers, kv_dim, vocab;
} nt_dims;

/* Why forward returns a code.
 *
 * It used to return void, and it had three ways to not do what it said: a
 * scratch allocation could fail, a family could stop early at the end of the
 * cache, and a caller could hand it a token id or a position the model has no
 * room for. In all three the caller got its buffer back unchanged and no way to
 * know — a CLI prints one wrong token, a body serving turns keeps going. An
 * outside audit named this as the reason the harness was not yet an honest
 * library boundary, and it is right: a function that cannot fail out loud is
 * not a contract.
 *
 * Zero is success and means logits were written when logits were asked for.
 * Everything else is a refusal, and on a refusal the output buffer is not
 * written at all rather than half written. */
enum {
    NT_OK          = 0,
    NT_E_ARG       = 1,   /* n < 1, pos0 < 0, a NULL the interface requires */
    NT_E_TOKEN     = 2,   /* a token id outside the model's vocabulary */
    NT_E_CAPACITY  = 3,   /* pos0 + n does not fit the cache handed in */
    NT_E_CACHE     = 4,   /* the cache does not match the model's shape */
    NT_E_MEMORY    = 5,   /* an allocation inside the forward failed */
    NT_E_STATE     = 6,   /* the family's own state could not be prepared */
};

const char *nt_strerror(int rc);

/* Optional observation / intervention after a complete decoder layer, before
 * the next layer (or final output norm). residual is a borrowed, writable
 * row-major [n, width] view, valid only during this call. Layer and absolute
 * token positions are zero-based; row j is position pos0+j. A caller may
 * inspect rows or mutate them in place, and selects layers/positions itself.
 * Return NT_OK to continue or a nonzero caller error to stop the forward.
 * The callback runs synchronously on the caller's thread. */
typedef int (*nt_residual_fn)(void *user, int layer, int pos0, int n,
                              int width, float *residual);

typedef struct {
    const char *const *names;    /* NULL-terminated list of supported names */
    void *(*load)(gguf_file *gf, nt_dims *dims);
    void  (*free)(void *model);
    /* One forward for a group of consecutive positions. Decode calls it with
     * n = 1; prefill calls it with a chunk of the prompt. `logits` may be NULL,
     * which asks for the KV cache alone, and when it is not NULL it is filled
     * for the LAST row only — the only position anybody samples from.
     *
     * Returns NT_OK, or one of the codes above. A non-zero return means
     * nothing was written to `logits` and the KV cache may hold a partial
     * prefill: the sequence has to be restarted, not continued. */
    int   (*forward)(void *model, kv_cache *kv, const int *tokens, int n,
                     int pos0, float *logits);
    /* NULL when this family does not expose residuals. With callback=NULL,
     * identical to forward. Callback errors propagate unchanged, leave logits
     * untouched, and require restarting the sequence just like forward errors.
     * No callback is retained in the model between calls. */
    int   (*forward_residual)(void *model, kv_cache *kv, const int *tokens, int n,
                              int pos0, float *logits,
                              nt_residual_fn callback, void *user);
    /* NULL when this family has no multi-sequence decode. One decode step for n
     * independent sequences: row j is tokens[j] at position pos[j] of the sequence whose
     * cache is kvs[j], and the n caches must be distinct. `logits`, when not NULL, is
     * [n, vocab] with one row per sequence. Each row's arithmetic is exactly that of
     * forward_residual(model, kvs[j], &tokens[j], 1, pos[j], ..., users[j]) — the same
     * logits and the same cache contents, bit for bit — while each weight matrix is read
     * once for all rows. The callback, if any, runs per row and layer with n = 1, that
     * row's position and users[j]; `users` is required when a callback is given. Errors
     * follow forward_residual: on a nonzero return no logits are written and every
     * sequence in the call has to be restarted. */
    int   (*forward_multi)(void *model, kv_cache *const *kvs, const int *tokens,
                           const int *pos, int n, float *logits,
                           nt_residual_fn callback, void *const *users);
    /* NULL when the family's cache is the host cache kv_new makes. A family that keeps
     * its cache elsewhere — the CUDA decoder keeps it in device memory — allocates it
     * here: the struct carries n_layers, max_seq and kv_dim as kv_new's does, its k and
     * v belong to the family and are never read by the caller, and it is released with
     * the same family's kv_free. */
    kv_cache *(*kv_new)(void *model, int max_seq);
    void      (*kv_free)(void *model, kv_cache *kv);
} nt_arch;

/* The checks every family needs and none should be writing for itself: token
 * ids inside the vocabulary, a sane group size, a position that fits, a cache
 * shaped for this model. Call it first in a forward and return what it returns.
 *
 * The last three are what this family's load reported in nt_dims, passed as
 * numbers because a family keeps its own struct and not a copy of dims. A
 * family that does not use the shared cache passes kv_dim 0 and the cache shape
 * is not compared. */
int nt_check_call(const kv_cache *kv, const int *tokens, int n, int pos0,
                  int vocab, int n_layers, int kv_dim);

extern const nt_arch nt_arch_llama;
extern const nt_arch nt_arch_gemma3;
extern const nt_arch nt_arch_gemma4;
extern const nt_arch nt_arch_olmoe;
extern const nt_arch nt_arch_mamba;
extern const nt_arch nt_arch_resonance;
extern const nt_arch nt_arch_janus;

#endif
