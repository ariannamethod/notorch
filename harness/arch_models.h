/* arch_models.h — the llama and gemma3 model structs, shared with the CUDA decoder.
 *
 * The CUDA decoder loads a model through the CPU loaders and uploads what they parsed, so
 * the metadata checks, tensor lookups and dtype probes exist once. These are the structs
 * those loaders fill, moved here unchanged from arch_llama.c and arch_gemma3.c. */
#ifndef NT_HARNESS_ARCH_MODELS_H
#define NT_HARNESS_ARCH_MODELS_H

#include "harness/arch.h"

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps;
    int rope_neox;         /* 1 = pair i with i+hd/2 (qwen2 and most non-llama) */
    int has_output_weight; /* 0 = tied embeddings */

    gguf_file *gf;      /* the packed weights point into it; must outlive this */
    int emb_ti;         /* token_embd tensor index, for the per-token row read */

    wt tok_emb;         /* [vocab, embed] — also the lm_head when tied */
    float *out_norm;    /* [embed] */
    wt out_weight;      /* [vocab, embed], absent when tied */

    struct {
        float *attn_norm;
        wt wq, wk, wv, wo;
        float *q_bias, *k_bias, *v_bias;   /* Qwen2 has bias; Qwen3 does not */
        float *q_norm, *k_norm;            /* [head_dim] — Qwen3's QK norm, absent elsewhere */
        float *ffn_norm;
        float *ffn_down_bias;             /* [embed], optional residual shift */
        wt wgate, wup, wdown;
    } layers[];
} llama_model;

typedef struct {
    float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm;
    wt q, k, v, o, gate, up, down;
} gemma3_layer;
typedef struct {
    int E, H, KV, HD, QD, KD, FF, V, L, window, pattern, emb_ti;
    float eps, base, local_base, rope_scale, query_scale, softcap;
    gguf_file *gf;
    wt emb, output;
    float *norm;
    gemma3_layer layers[];
} gemma3_model;

#endif
