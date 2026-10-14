/* Immutable PEN7 bodies; no tape, optimizer, sampling or global sequence state. */
#ifndef NT_PEN7_H
#define NT_PEN7_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct nt_pen7_model nt_pen7_model;
typedef struct nt_pen7_workspace nt_pen7_workspace;
typedef struct {
    int vocab, width, context, layers, ffn, heads;
    char identity[65];            /* SHA-256 of the exact loaded GGUF snapshot */
    char tokenizer_identity[65];  /* SHA-256 of LE int32 merge pairs; fixed v1 rules */
} nt_pen7_info;

enum {
    NT_PEN7_OK = 0, NT_PEN7_E_ARG, NT_PEN7_E_IO, NT_PEN7_E_FORMAT,
    NT_PEN7_E_MEMORY, NT_PEN7_E_CAPACITY, NT_PEN7_E_NUMERIC,
    NT_PEN7_E_COMPAT, NT_PEN7_E_TOKEN
};
const char *nt_pen7_error_string(int status);

/* Quiet load: no stdout/stderr. Copies a regular file into an owned snapshot;
 * path replacement, deletion or later writes cannot alter a loaded body.
 * Only the declared PEN7 F32/Q8_0 schema is accepted: snapshot and expanded
 * parameter storage are each bounded at 512 MiB, decoded token surfaces at
 * 32 MiB. On failure *out is unchanged.
 * Retain/release are atomic. A caller must own a live reference while using it;
 * retaining a handle concurrently with its last release is not valid. */
int nt_pen7_load(const char *path, nt_pen7_model **out);
nt_pen7_model *nt_pen7_retain(nt_pen7_model *model);
void nt_pen7_release(nt_pen7_model *model);
const nt_pen7_info *nt_pen7_get_info(const nt_pen7_model *model);

/* Canonical raw-byte ranked-pass-v1 BPE with ASCII lowercase. No BOS/EOS or
 * pre-tokenizer. Explicit byte length permits NUL; input is bounded at 8191
 * bytes and is never truncated. Empty input succeeds with count zero. Output
 * ids and count are published together on success, unchanged on any refusal. */
int nt_pen7_encode(const nt_pen7_model *model, const char *text, size_t bytes,
                   int *ids, size_t capacity, size_t *count);
/* Borrowed exact bytes, possibly NUL or a UTF-8 fragment, valid for model life.
 * Invalid id/model/length pointer returns NULL and leaves *bytes untouched. */
const unsigned char *nt_pen7_token_piece(const nt_pen7_model *model, int id,
                                        size_t *bytes);

/* A workspace owns bounded sequence and mixing scratch, never model weights.
 * It can serve compatible models and alpha values repeatedly. One workspace
 * per simultaneous call; it has no history between calls. max_tokens is 1..ctx.
 * A workspace is bounded at 256 MiB; larger requests return E_CAPACITY.
 * The workspace retains its model; caller handles can be released afterwards. */
int nt_pen7_workspace_new(nt_pen7_model *model, size_t max_tokens,
                          nt_pen7_workspace **out);
void nt_pen7_workspace_free(nt_pen7_workspace *workspace);
size_t nt_pen7_workspace_bytes(const nt_pen7_workspace *workspace);

/* Each call starts at learned position zero, using exactly count ids. At least
 * one output is required; NULL outputs require capacity zero. logits[vocab]
 * and repr[width] must not overlap. repr is the final last-token RMS-normalized
 * contextual vector consumed by the output head. No writes on any error,
 * including nonfinite results and overflow in RMS accumulation. Capacities are float counts, not byte counts. */
int nt_pen7_forward(const nt_pen7_model *model, nt_pen7_workspace *workspace,
                    const int *ids, size_t count,
                    float *logits, size_t logits_capacity,
                    float *repr, size_t repr_capacity);
/* Every parameter participates: alpha*A + (1-alpha)*B before forward. alpha
 * must be finite in [0,1]. Endpoints use the exact source body. Compatibility
 * requires all geometry and tokenizer identity to agree. One tensor at a time
 * uses reusable scratch, so no permanent full mixed body is allocated. */
int nt_pen7_mix_forward(const nt_pen7_model *left, const nt_pen7_model *right,
                        float alpha, nt_pen7_workspace *workspace,
                        const int *ids, size_t count,
                        float *logits, size_t logits_capacity,
                        float *repr, size_t repr_capacity);
#ifdef __cplusplus
}
#endif
#endif
