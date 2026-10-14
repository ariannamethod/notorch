/* Family-private interface shared by the harness and immutable library owner. */
#ifndef NT_PENELOPE_INTERNAL_H
#define NT_PENELOPE_INTERNAL_H
#include "harness/arch.h"
#define PEN_MAX_BLOCKS 32
typedef struct {
    wt q, k, v, out, resonance, gate, up, down;
    const float *attn_norm, *ffn_norm, *blend;
} pen_block;
typedef struct {
    gguf_file *gf;
    int vocab, embed, heads, hd, layers, ffn, ctx, tok, pos;
    float eps, base;
    const float *norm, *token_data, *position_data;
    float *token_owned, *position_owned;
    wt output;
    pen_block block[PEN_MAX_BLOCKS];
} pen_model;
pen_model *nt_penelope_load_quiet(gguf_file *gf, nt_dims *dims);
/* Quiet checked load: NT_OK, NT_E_ARG for an invalid family contract, or
 * NT_E_MEMORY for allocation failure. On refusal dims and *out are unchanged.
 * The pointer-only entry above remains available to existing family callers. */
int nt_penelope_load_checked(gguf_file *gf, nt_dims *dims, pen_model **out);
void nt_penelope_drop(void *model);
/* scratch uses 6*n*E + 2*n*F + 2*total*hd floats. mix_scratch is sized to
 * max(vocab*E, E*E, F*E); its two last small vectors are separate in core. */
int nt_penelope_eval(const pen_model *m, const pen_model *other, float alpha,
    kv_cache *kv, const int *ids, int n, int pos0, float *logits, float *repr,
    float *scratch, float *mix_scratch, nt_residual_fn callback, void *user);
#endif
