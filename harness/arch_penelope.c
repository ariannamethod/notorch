/* PEN7 / Penelope: learned positions plus split-half RoPE, causal attention,
 * a dense resonance branch, a two-logit gate, and SwiGLU. The model borrows
 * immutable GGUF bytes; Q8_0 matrices expand once into owned F32 weights.
 * Caches belong to sequences and scratch to each call.
 * No optimizer, tape or global sequence state participates in inference. */
#include "harness/penelope_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pen_u32(gguf_file *gf, const char *key, int *out, int max) {
    const gguf_kv *v = gguf_get_kv(gf, key);
    if (!v || v->type != 4 || !v->val.u32 || v->val.u32 > (uint32_t)max) return 0;
    *out = (int)v->val.u32;
    return 1;
}
static int pen_float(gguf_file *gf, const char *key, float expected) {
    const gguf_kv *v = gguf_get_kv(gf, key);
    return v && v->type == 6 && v->val.f32 == expected;
}
static int pen_string(gguf_file *gf, const char *key, const char *expected) {
    const gguf_kv *v = gguf_get_kv(gf, key);
    return v && v->type == 8 && !strcmp(v->val.str, expected);
}
static int pen_shape(gguf_file *gf, const char *name, int rows, int cols) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return -1;
    const gguf_tensor_info *t = &gf->tensors[ti];
    if ((t->dtype != GGUF_TYPE_F32 && t->dtype != GGUF_TYPE_Q8_0) ||
        t->ndim != (rows ? 2u : 1u) || t->shape[0] != (uint64_t)cols ||
        (rows && t->shape[1] != (uint64_t)rows)) return -1;
    uint64_t bytes = gguf_type_size(t->dtype, t->n_elements);
    if (!bytes || t->offset > gf->data_size || bytes > gf->data_size - t->offset) return -1;
    if (t->dtype == GGUF_TYPE_F32) {
        const float *p = (const float *)(gf->data + t->offset);
        if (t->offset % sizeof(float)) return -1;
        for (uint64_t i = 0; i < t->n_elements; i++) if (!isfinite(p[i])) return -1;
    } else {
        if (cols % 32) return -1;
        const unsigned char *p = gf->data + t->offset;
        for (uint64_t i = 0; i < t->n_elements/32; i++) {
            uint16_t bits; memcpy(&bits, p+34*i, sizeof(bits));
            if ((bits & 0x7c00u) == 0x7c00u) return -1;
        }
    }
    return ti;
}
static const float *pen_vector(gguf_file *gf, const char *name, int n) {
    int ti = pen_shape(gf, name, 0, n);
    /* Small norms and the gate remain F32 in the PEN7 schema. */
    return ti >= 0 && gf->tensors[ti].dtype == GGUF_TYPE_F32
        ? (const float *)(gf->data + gf->tensors[ti].offset) : NULL;
}
static int pen_matrix(wt *w, gguf_file *gf, const char *name, int rows, int cols) {
    int ti = pen_shape(gf, name, rows, cols);
    if (ti < 0) return 0;
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->dtype == GGUF_TYPE_F32) {
        w->rows = rows; w->cols = cols; w->dtype = GGUF_TYPE_F32;
        w->f32 = (float *)(gf->data + t->offset); /* borrowed, never freed */
        return 1;
    }
    /* Quantization is the storage format. The body explicitly unfolds into
     * float32 for its projections and eventual parameter-space mixtures. */
    w->rows = rows; w->cols = cols; w->dtype = GGUF_TYPE_Q8_0;
    w->f32 = gguf_dequant(gf, ti);
    return w->f32 != NULL;
}
static void pen_drop_weight(wt *w) {
    if (w->dtype != GGUF_TYPE_F32) free(w->f32);
}
void nt_penelope_drop(void *model) {
    pen_model *m = model;
    if (!m) return;
    free(m->token_owned); free(m->position_owned);
    pen_drop_weight(&m->output);
    for (int l = 0; l < m->layers; l++) {
        pen_block *b = &m->block[l];
        pen_drop_weight(&b->q); pen_drop_weight(&b->k); pen_drop_weight(&b->v);
        pen_drop_weight(&b->out); pen_drop_weight(&b->resonance);
        pen_drop_weight(&b->gate); pen_drop_weight(&b->up); pen_drop_weight(&b->down);
    }
    free(m);
}
pen_model *nt_penelope_load_quiet(gguf_file *gf, nt_dims *dims) {
    if (!gf || !dims || strcmp(gf->arch, "penelope")) return NULL;
    pen_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->gf = gf; m->eps = 1e-5f; m->base = 10000.f;
#define U(k, member, max) if (!pen_u32(gf, "penelope." k, &m->member, max)) goto fail
    U("embedding_length", embed, 8192); U("attention.head_count", heads, 128);
    U("attention.head_dim", hd, 8192); U("block_count", layers, PEN_MAX_BLOCKS);
    U("feed_forward_length", ffn, 32768); U("context_length", ctx, 65536);
    U("vocab_size", vocab, 1048576);
#undef U
    if (gf->n_tensors != (uint64_t)(4 + 11*m->layers) ||
        !pen_string(gf, "tokenizer.ggml.model", "penelope") ||
        !pen_string(gf, "tokenizer.penelope.algorithm", "byte-rank-pass-v1")) goto fail;
    int rotary = 0, kv_heads = 0;
    const gguf_kv *position = gguf_get_kv(gf, "penelope.position_embedding");
    if (m->heads * m->hd != m->embed || m->hd % 2 ||
        !pen_u32(gf, "penelope.attention.head_count_kv", &kv_heads, 128) || kv_heads != m->heads ||
        !pen_u32(gf, "penelope.rope.dimension_count", &rotary, 8192) || rotary != m->hd ||
        !pen_float(gf, "penelope.attention.layer_norm_rms_epsilon", m->eps) ||
        !pen_float(gf, "penelope.rope.freq_base", m->base) ||
        !pen_float(gf, "penelope.silu_cutoff", -20.f) ||
        !pen_string(gf, "penelope.rope.layout", "split-half") ||
        !pen_string(gf, "penelope.resonance", "dense-gated") ||
        !position || !((position->type == 7 && position->val.b == 1) ||
                        (position->type == 4 && position->val.u32 == 1))) goto fail;
    m->tok = pen_shape(gf, "token_embd.weight", m->vocab, m->embed);
    m->pos = pen_shape(gf, "position_embd.weight", m->ctx, m->embed);
    m->norm = pen_vector(gf, "output_norm.weight", m->embed);
    if (m->tok < 0 || m->pos < 0 || !m->norm ||
        !pen_matrix(&m->output, gf, "output.weight", m->vocab, m->embed)) goto fail;
    if (gf->tensors[m->tok].dtype == GGUF_TYPE_F32)
        m->token_data = (const float *)(gf->data + gf->tensors[m->tok].offset);
    else m->token_data = m->token_owned = gguf_dequant(gf, m->tok);
    if (gf->tensors[m->pos].dtype == GGUF_TYPE_F32)
        m->position_data = (const float *)(gf->data + gf->tensors[m->pos].offset);
    else m->position_data = m->position_owned = gguf_dequant(gf, m->pos);
    if (!m->token_data || !m->position_data) goto fail;
    for (int l = 0; l < m->layers; l++) {
        pen_block *b = &m->block[l]; char name[128];
#define V(member, suffix, len) do { \
    snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
    b->member = pen_vector(gf, name, len); if (!b->member) goto fail; \
} while (0)
#define W(member, suffix, rows, cols) do { \
    snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
    if (!pen_matrix(&b->member, gf, name, rows, cols)) goto fail; \
} while (0)
        V(attn_norm, "attn_norm", m->embed); V(ffn_norm, "ffn_norm", m->embed);
        V(blend, "resonance_gate", 2);
        W(q, "attn_q", m->embed, m->embed); W(k, "attn_k", m->embed, m->embed);
        W(v, "attn_v", m->embed, m->embed); W(out, "attn_output", m->embed, m->embed);
        W(resonance, "resonance", m->embed, m->embed);
        W(gate, "ffn_gate", m->ffn, m->embed); W(up, "ffn_up", m->ffn, m->embed);
        W(down, "ffn_down", m->embed, m->ffn);
#undef V
#undef W
    }
    dims->n_layers = m->layers; dims->kv_dim = m->embed; dims->vocab = m->vocab;
    return m;
fail:
    nt_penelope_drop(m);
    return NULL;
}

/* The same equations serve the harness and immutable owner. Mixing happens
 * before multiplication for every tensor, including positions, norms and gates.
 * A matrix is mixed once for all current token rows into one reusable buffer. */
static const float *pen_mix(const float *a, const float *b, float alpha,
                            size_t count, float *scratch) {
    if (!b) return a;
    for (size_t i = 0; i < count; ++i)
        scratch[i] = alpha*a[i] + (1.f-alpha)*b[i];
    return scratch;
}
static void pen_rows(float *out, size_t stride_out, const wt *a, const wt *b,
                      float alpha, const float *x, size_t stride_x, int n,
                      float *mix_scratch) {
    const float *w = pen_mix(a->f32, b ? b->f32 : NULL, alpha,
                            (size_t)a->rows*a->cols, mix_scratch);
    for (int t = 0; t < n; ++t)
        nt_blas_matvec(out+(size_t)t*stride_out, w,
                       x+(size_t)t*stride_x, a->rows, a->cols);
}
/* A finite zero after 1/sqrt(infinity) is still an overflow, not a valid
 * contextual vector. Check the accumulator before the shared norm runs. */
static int pen_norm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i]*x[i];
    if (!isfinite(ss)) return 0;
    rmsnorm(out, x, w, n, eps);
    for (int i = 0; i < n; ++i) if (!isfinite(out[i])) return 0;
    return 1;
}
int nt_penelope_eval(const pen_model *m, const pen_model *other, float alpha,
    kv_cache *kv, const int *tokens, int n, int pos0, float *logits, float *repr,
    float *mem, float *mix_scratch, nt_residual_fn callback, void *user) {
    const int E = m->embed, F = m->ffn, D = m->hd, total = pos0+n;
    const size_t ne = (size_t)n*E, nf = (size_t)n*F, nh = (size_t)total*D;
    float *x = mem, *h = x+ne, *q = h+ne, *av = q+ne;
    float *proj = av+ne, *rr = proj+ne, *fg = rr+ne, *fu = fg+nf;
    float *kh = fu+nf, *vh = kh+nh;
    int rc = NT_OK;
    for (int t = 0; t < n; ++t) {
        size_t token = (size_t)tokens[t]*E, position = (size_t)(pos0+t)*E;
        for (int d = 0; d < E; ++d) {
            float te = m->token_data[token+d], pe = m->position_data[position+d];
            if (other) {
                te = alpha*te + (1.f-alpha)*other->token_data[token+d];
                pe = alpha*pe + (1.f-alpha)*other->position_data[position+d];
            }
            x[(size_t)t*E+d] = te+pe;
        }
    }
    for (int l = 0; l < m->layers; ++l) {
        const pen_block *b = &m->block[l], *b2 = other ? &other->block[l] : NULL;
        float *keys = kv->k+(size_t)l*kv->max_seq*E;
        float *values = kv->v+(size_t)l*kv->max_seq*E;
        const float *norm = pen_mix(b->attn_norm, b2 ? b2->attn_norm : NULL,
                                    alpha, E, mix_scratch);
        for (int t = 0; t < n; ++t)
            if (!pen_norm(h+(size_t)t*E, x+(size_t)t*E, norm, E, m->eps)) return NT_E_STATE;
        pen_rows(q, E, &b->q, b2 ? &b2->q : NULL, alpha, h, E, n, mix_scratch);
        pen_rows(keys+(size_t)pos0*E, E, &b->k, b2 ? &b2->k : NULL, alpha, h, E, n, mix_scratch);
        pen_rows(values+(size_t)pos0*E, E, &b->v, b2 ? &b2->v : NULL, alpha, h, E, n, mix_scratch);
        for (size_t i = 0; i < ne; ++i)
            if (!isfinite(q[i]) || !isfinite(keys[(size_t)pos0*E+i]) ||
                !isfinite(values[(size_t)pos0*E+i])) return NT_E_STATE;
        for (int t = 0; t < n; ++t) for (int a = 0; a < m->heads; ++a) {
            rope(q+(size_t)t*E+a*D, pos0+t, D, m->base, 1);
            rope(keys+(size_t)(pos0+t)*E+a*D, pos0+t, D, m->base, 1);
        }
        for (int a = 0; a < m->heads; ++a) {
            for (int t = 0; t < total; ++t) {
                memcpy(kh+(size_t)t*D, keys+(size_t)t*E+a*D, (size_t)D*sizeof(float));
                memcpy(vh+(size_t)t*D, values+(size_t)t*E+a*D, (size_t)D*sizeof(float));
            }
            for (int t = 0; t < n; ++t)
                if (nt_attention(av+(size_t)t*E+a*D, q+(size_t)t*E+a*D,
                                  kh, vh, 1, pos0+t+1, D)) return NT_E_MEMORY;
        }
        const float *blend = pen_mix(b->blend, b2 ? b2->blend : NULL, alpha, 2, mix_scratch);
        float gmax = blend[0] > blend[1] ? blend[0] : blend[1];
        float e0 = expf(blend[0]-gmax), e1 = expf(blend[1]-gmax);
        float w0 = e0/(e0+e1), w1 = e1/(e0+e1);
        pen_rows(proj, E, &b->out, b2 ? &b2->out : NULL, alpha, av, E, n, mix_scratch);
        pen_rows(rr, E, &b->resonance, b2 ? &b2->resonance : NULL, alpha, h, E, n, mix_scratch);
        norm = pen_mix(b->ffn_norm, b2 ? b2->ffn_norm : NULL, alpha, E, mix_scratch);
        for (int t = 0; t < n; ++t) {
            size_t row = (size_t)t*E;
            for (int d = 0; d < E; ++d) x[row+d] += w0*proj[row+d] + w1*rr[row+d];
            if (!pen_norm(h+row, x+row, norm, E, m->eps)) return NT_E_STATE;
        }
        pen_rows(fg, F, &b->gate, b2 ? &b2->gate : NULL, alpha, h, E, n, mix_scratch);
        pen_rows(fu, F, &b->up, b2 ? &b2->up : NULL, alpha, h, E, n, mix_scratch);
        for (size_t d = 0; d < nf; ++d) {
            float z = fg[d];
            if (!isfinite(z) || !isfinite(fu[d])) return NT_E_STATE;
            fg[d] = (z > -20.f ? z/(1.f+expf(-z)) : 0.f)*fu[d];
        }
        pen_rows(proj, E, &b->down, b2 ? &b2->down : NULL, alpha, fg, F, n, mix_scratch);
        for (size_t d = 0; d < ne; ++d) x[d] += proj[d];
        if (callback && (rc = callback(user, l, pos0, n, E, x)) != NT_OK) return rc;
    }
    if (logits || repr) {
        const float *norm = pen_mix(m->norm, other ? other->norm : NULL, alpha, E, mix_scratch);
        if (!pen_norm(h, x+(size_t)(n-1)*E, norm, E, m->eps)) return NT_E_STATE;
        if (repr) memcpy(repr, h, (size_t)E*sizeof(float));
        if (logits) pen_rows(logits, m->vocab, &m->output,
                            other ? &other->output : NULL, alpha, h, E, 1, mix_scratch);
    }
    return rc;
}
static void *pen_load(gguf_file *gf, nt_dims *dims) {
    pen_model *m = nt_penelope_load_quiet(gf, dims);
    if (!m) fprintf(stderr, "penelope: unsupported geometry, metadata, tensor shape/dtype or nonfinite weight\n");
    return m;
}
static int pen_forward_residual(void *model, kv_cache *kv, const int *tokens, int n,
                                int pos0, float *logits, nt_residual_fn callback, void *user) {
    const pen_model *m = model;
    if (!m) return NT_E_ARG;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->layers, m->embed);
    if (rc) return rc;
    if (pos0 > m->ctx || n > m->ctx-pos0) return NT_E_CAPACITY;
    size_t count = 6*(size_t)n*m->embed + 2*(size_t)n*m->ffn + 2*(size_t)(pos0+n)*m->hd;
    if (count > SIZE_MAX/sizeof(float)) return NT_E_MEMORY;
    float *mem = malloc(count*sizeof(float));
    if (!mem) return NT_E_MEMORY;
    rc = nt_penelope_eval(m, NULL, 1.f, kv, tokens, n, pos0, logits, NULL, mem, NULL, callback, user);
    free(mem);
    return rc;
}
static int pen_forward(void *m, kv_cache *kv, const int *tokens, int n, int pos0, float *logits) {
    return pen_forward_residual(m, kv, tokens, n, pos0, logits, NULL, NULL);
}
static const char *const pen_names[] = {"penelope", NULL};
const nt_arch nt_arch_penelope = {
    .names = pen_names, .load = pen_load, .free = nt_penelope_drop,
    .forward = pen_forward, .forward_residual = pen_forward_residual,
};
