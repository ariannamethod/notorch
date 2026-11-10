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

/* A pretrained mixture's actual FFN choices for one input token. The k
 * borrowed, read-only IDs and applied weights are in accumulation order.
 * Observation is synchronous and scoped to this forward call. */
typedef void (*nt_moe_route_fn)(void *user, int layer, int pos, int token,
                                int k, const int *ids, const float *weights);
typedef int nt_moe_forward(void *model, kv_cache *kv, const int *tokens, int n,
                           int pos0, float *logits, nt_residual_fn post_attention,
                           nt_residual_fn post_block, void *user,
                           nt_moe_route_fn route, void *route_user);

#define NT_DOE_DECODER_API 1

/* CPU decoders offering DOE expose both sides of the FFN. Calibration hashes
 * the full residual-width norms in block execution order, then the final norm.
 * The original two-norm layout is used when norms_per_layer is zero. */
typedef struct {
    int (*forward)(void *model, kv_cache *kv, const int *tokens, int n, int pos0,
                   float *logits, nt_residual_fn post_attention, nt_residual_fn post_block, void *user);
    const gguf_file *(*describe)(const void *model, nt_dims *dims);
    const float *(*norm)(const void *model, int index);
    int norms_per_layer;
} nt_doe_decoder;

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
    const nt_doe_decoder *doe; /* NULL for a decoder without native expert hooks. */
    nt_moe_forward *forward_routes; /* Optional observation of the native sparse FFN. */
    /* NULL when the family reads token ids only; a caller with embedding rows refuses
     * such a family by name. Otherwise forward, with two additions. rows, when not
     * NULL, holds n pointers: position j reads rows[j] (the model's embedding width)
     * in place of the embedding of tokens[j], or its token when rows[j] is NULL.
     * tokens[j] is still checked against the vocabulary, so a placeholder id marks
     * the position and no sentinel enters the stream. hidden, when not NULL, receives
     * the final-norm output of the last row, the vector the head reads. Errors follow
     * forward: on a nonzero return neither logits nor hidden are written. */
    int   (*forward_embd)(void *model, kv_cache *kv, const int *tokens,
                          const float *const *rows, int n, int pos0,
                          float *logits, float *hidden);
} nt_arch;

static inline kv_cache *nt_model_kv_new(const nt_arch *arch, void *model, nt_dims dims, int capacity) {
    return arch->kv_new ? arch->kv_new(model, capacity) : kv_new(dims.n_layers, capacity, dims.kv_dim);
}
static inline void nt_model_kv_free(const nt_arch *arch, void *model, kv_cache *kv) {
    if (!kv) return;
    if (arch->kv_free) arch->kv_free(model, kv);
    else kv_free(kv);
}

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
/* The dense llama-family decoder also exposes the DOE boundary between the
 * attention residual and the FFN. Both hooks follow nt_residual_fn's contract.
 * model must come from nt_arch_llama.load. */
int nt_llama_forward_hooks(void *model, kv_cache *kv, const int *tokens, int n,
                           int pos0, float *logits, nt_residual_fn post_attention,
                           nt_residual_fn post_block, void *user);

extern const nt_arch nt_arch_qwen35;
extern const nt_arch nt_arch_gemma3;
extern const nt_arch nt_arch_gemma4;
extern const nt_arch nt_arch_olmoe;
extern const nt_arch nt_arch_mamba;
extern const nt_arch nt_arch_resonance;
extern const nt_arch nt_arch_janus;
extern const nt_arch nt_arch_penelope;
extern const nt_arch nt_arch_lfm3;

/* Full-sequence encoders have a different contract from causal decoders: no
 * token-logit output and no KV state carried between calls. All n positions
 * participate in one bidirectional call. mask is NULL (all positions) or n
 * bytes in {0,1}; masked positions cannot be keys or enter the mean pool.
 * Callbacks run synchronously and may inspect or alter each named boundary.
 * The caller's pooled/output buffers are committed only on NT_OK. */
typedef struct {
    int n_layers, width, output_dim, vocab, max_tokens;
} nt_encode_dims;

typedef enum {
    NT_ENCODE_INPUT, NT_ENCODE_PLE, NT_ENCODE_POST_ATTENTION,
    NT_ENCODE_POST_FFN, NT_ENCODE_BLOCK, NT_ENCODE_FINAL_NORM,
    NT_ENCODE_PROJECTED
} nt_encode_site;

typedef int (*nt_encode_fn)(void *user, nt_encode_site site, int layer,
                            int n, int width, float *states);

typedef struct {
    const char *name;
    void *(*load)(gguf_file *gf, nt_encode_dims *dims);
    void (*free)(void *model);
    int (*encode)(void *model, const int *tokens, const unsigned char *mask,
                  int n, float *pooled, float *normalized,
                  nt_encode_fn callback, void *user);
} nt_encoder;

extern const nt_encoder nt_encoder_embeddinggemma2;
/* Causal attention: a masked position is never a key, and the pooled vector is
 * the final state of the last unmasked position. Sites: INPUT, POST_ATTENTION,
 * BLOCK (the post-FFN residual), FINAL_NORM; output_dim equals width. */
extern const nt_encoder nt_encoder_qwen3embedding;

#endif
