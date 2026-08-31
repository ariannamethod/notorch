/* arch_gemma4.c — Gemma 4, and the first family that is not a llama with different numbers.
 *
 * Everything the harness ran until now was the same graph: norm, qkv, rope, attention,
 * gated FFN, residual. Gemma 4 keeps that skeleton and hangs four things on it that the
 * interface had never been asked for, which is why this file is the first real test of
 * whether arch.h is honest.
 *
 *  - Two attention geometries in one model. Every fifth layer attends to everything with
 *    512-wide heads; the other four attend inside a 512-position window with 256-wide
 *    heads. Head size, rope base and rope length all change with the layer.
 *
 *  - Only the first fifteen layers own a KV cache. The other twenty read one: a sliding
 *    layer reads layer 13's, a full layer reads layer 14's. That is not an optimization
 *    we chose, it is the model — the weights for those layers have no k or v projection
 *    at all, and there are 541 tensors in the file precisely because of what is missing.
 *
 *  - Per-layer embeddings. Half the file (1.32 GB of 2.6) is a second embedding table with
 *    a 256-wide row per layer per token. Each layer gates its own output through its slice
 *    of it. Nothing else in this tree has a weight that is indexed by token AND by layer.
 *
 *  - Attention with no 1/sqrt(head_dim). Gemma 4 sets the scale to 1 and lets the query
 *    norm carry it, so the one line every transformer shares is the one to leave out.
 *
 * The arithmetic follows llama.cpp's graph for this family (src/models/gemma4.cpp), which
 * is the reference the output is compared against: same file, same prompt, temperature 0,
 * same tokens or it is wrong.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, vocab;
    int n_kv_layers;      /* layers 0..n_kv_layers-1 own a cache; the rest borrow */
    int ple_dim;          /* 256 — width of one layer's slice of the second table */
    int swa_window;
    float rope_base, rope_base_swa, rms_eps, softcap;

    gguf_file *gf;
    int emb_ti, ple_ti;   /* both embedding tables are read a row at a time */

    wt tok_emb;           /* [vocab, embed] — also the head, tied */
    float *out_norm;
    float *ple_proj_w;    /* per_layer_model_proj, bf16 in the file, expanded here */
    float *ple_proj_norm;
    float *rope_freqs;    /* proportional rope, full-attention layers only */

    struct {
        float *attn_norm, *q_norm, *k_norm, *post_attn_norm;
        float *ffn_norm, *post_ffw_norm, *ple_post_norm;
        float out_scale;
        wt wq, wk, wv, wo, wgate, wup, wdown, ple_gate, ple_proj;
        int head_dim, q_dim, kv_dim, ffn, is_swa, kv_slot;
    } layers[];
} gemma4_model;

static float *dequant_named(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    return ti >= 0 ? gguf_dequant(gf, ti) : NULL;
}

static void *gemma4_load(gguf_file *gf, nt_dims *dims) {
    int nl = gf->n_layers;
    gemma4_model *m = (gemma4_model*)calloc(1, sizeof(gemma4_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->gf = gf;

    const gguf_kv *kv;
    /* Read by exact key, not by the reader's convenience fields. gguf.c fills those by
     * matching a suffix, and this family has two keys per suffix: embedding_length and
     * embedding_length_per_layer_input, rope.freq_base and rope.freq_base_swa. Whichever
     * came last in the file won, which is how a 1536-wide model announced itself as 256
     * wide and its full-attention layers took the sliding rope. */
    #define KVF(key, dst, dflt) do { \
        (dst) = (dflt); \
        if ((kv = gguf_get_kv(gf, key))) (dst) = kv->val.f32; \
    } while (0)
    #define KVU(key, dst, dflt) do { \
        (dst) = (dflt); \
        if ((kv = gguf_get_kv(gf, key))) (dst) = (int)kv->val.u32; \
    } while (0)
    KVU("gemma4.embedding_length", m->embed, gf->embed_dim);
    KVU("gemma4.attention.head_count", m->n_heads, gf->n_heads);
    KVU("gemma4.attention.head_count_kv", m->n_kv_heads, gf->n_kv_heads);
    KVF("gemma4.attention.layer_norm_rms_epsilon", m->rms_eps, gf->rms_eps);
    KVF("gemma4.rope.freq_base", m->rope_base, gf->rope_freq_base);
    KVF("gemma4.rope.freq_base_swa", m->rope_base_swa, 10000.0f);
    KVF("gemma4.final_logit_softcapping", m->softcap, 0.0f);
    KVU("gemma4.attention.sliding_window", m->swa_window, 512);
    KVU("gemma4.embedding_length_per_layer_input", m->ple_dim, 256);
    int shared_kv = 0;
    KVU("gemma4.attention.shared_kv_layers", shared_kv, 0);
    #undef KVF
    #undef KVU
    m->n_kv_layers = nl - shared_kv;
    if (m->n_kv_layers < 2 || m->n_kv_layers > nl) m->n_kv_layers = nl;

    int ti = gguf_find_tensor(gf, "token_embd.weight");
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;
    m->emb_ti = ti;
    m->ple_ti = gguf_find_tensor(gf, "per_layer_token_embd.weight");
    wt_load(&m->tok_emb, gf, "token_embd.weight");
    m->out_norm      = dequant_named(gf, "output_norm.weight");
    m->ple_proj_w    = dequant_named(gf, "per_layer_model_proj.weight");
    m->ple_proj_norm = dequant_named(gf, "per_layer_proj_norm.weight");
    m->rope_freqs    = dequant_named(gf, "rope_freqs.weight");

    int max_kv_dim = 0;
    for (int l = 0; l < nl; l++) {
        char name[128];
        #define L(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            m->layers[l].field = dequant_named(gf, name); \
        } while (0)
        #define W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            wt_load(&m->layers[l].field, gf, name); \
        } while (0)
        L(attn_norm,      "blk.%d.attn_norm.weight");
        L(q_norm,         "blk.%d.attn_q_norm.weight");
        L(k_norm,         "blk.%d.attn_k_norm.weight");
        L(post_attn_norm, "blk.%d.post_attention_norm.weight");
        L(ffn_norm,       "blk.%d.ffn_norm.weight");
        L(post_ffw_norm,  "blk.%d.post_ffw_norm.weight");
        L(ple_post_norm,  "blk.%d.post_norm.weight");
        W(wq,       "blk.%d.attn_q.weight");
        W(wk,       "blk.%d.attn_k.weight");
        W(wv,       "blk.%d.attn_v.weight");
        W(wo,       "blk.%d.attn_output.weight");
        W(wgate,    "blk.%d.ffn_gate.weight");
        W(wup,      "blk.%d.ffn_up.weight");
        W(wdown,    "blk.%d.ffn_down.weight");
        W(ple_gate, "blk.%d.inp_gate.weight");
        W(ple_proj, "blk.%d.proj.weight");
        #undef L
        #undef W

        snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", l);
        float *sc = dequant_named(gf, name);
        m->layers[l].out_scale = sc ? sc[0] : 1.0f;
        free(sc);

        /* Geometry is read off the weights rather than off a per-layer metadata array,
         * because the shapes cannot disagree with themselves. q is present on every layer
         * even where k and v are not, so it is what the head count divides. */
        m->layers[l].q_dim = m->layers[l].wq.rows;
        m->layers[l].head_dim = m->layers[l].q_dim / m->n_heads;
        m->layers[l].kv_dim = m->layers[l].head_dim * m->n_kv_heads;
        m->layers[l].ffn = m->layers[l].wgate.rows;
        if (m->layers[l].kv_dim > max_kv_dim) max_kv_dim = m->layers[l].kv_dim;
    }

    /* A layer is sliding unless its heads are the widest in the model — the full-attention
     * layers are exactly the ones that kept the large head. Layers past the KV boundary
     * have no k of their own, so they inherit the question from the layer they borrow. */
    for (int l = 0; l < nl; l++)
        m->layers[l].is_swa = m->layers[l].kv_dim < max_kv_dim;
    for (int l = 0; l < nl; l++) {
        if (l < m->n_kv_layers) m->layers[l].kv_slot = l;
        else m->layers[l].kv_slot = m->n_kv_layers - (m->layers[l].is_swa ? 2 : 1);
    }

    fprintf(stderr, "gemma4: E=%d H=%d KV=%d V=%d L=%d (KV on %d) PLE=%d window=%d softcap=%.0f\n",
            m->embed, m->n_heads, m->n_kv_heads, m->vocab, nl, m->n_kv_layers,
            m->ple_dim, m->swa_window, (double)m->softcap);
    fprintf(stderr, "  heads: %d wide on full layers, %d on sliding | rope %.0f / %.0f\n",
            max_kv_dim / m->n_kv_heads,
            m->layers[0].head_dim, (double)m->rope_base, (double)m->rope_base_swa);

    if (!(m->tok_emb.q || m->tok_emb.f32) || !m->out_norm || !m->ple_proj_w ||
        !m->ple_proj_norm || m->ple_ti < 0) {
        fprintf(stderr, "gemma4: missing critical weights\n");
        free(m);
        return NULL;
    }

    dims->n_layers = m->n_kv_layers;   /* only the layers that store need a slot */
    dims->kv_dim   = max_kv_dim;       /* sliding layers use the front of a wide row */
    dims->vocab    = m->vocab;
    return m;
}

static void gemma4_free(void *model) {
    gemma4_model *m = (gemma4_model*)model;
    if (!m) return;
    free(m->tok_emb.f32); free(m->out_norm);
    free(m->ple_proj_w); free(m->ple_proj_norm); free(m->rope_freqs);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm); free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].post_attn_norm); free(m->layers[l].ffn_norm);
        free(m->layers[l].post_ffw_norm); free(m->layers[l].ple_post_norm);
        free(m->layers[l].wq.f32); free(m->layers[l].wk.f32); free(m->layers[l].wv.f32);
        free(m->layers[l].wo.f32); free(m->layers[l].wgate.f32); free(m->layers[l].wup.f32);
        free(m->layers[l].wdown.f32);
        free(m->layers[l].ple_gate.f32); free(m->layers[l].ple_proj.f32);
    }
    free(m);
}

/* RMS norm with no learned weight — Gemma normalizes V that way, and the per-layer
 * projection reuses the same shape with a weight. */
static void rmsnorm_plain(float *out, const float *x, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

/* The exact GELU, not the tanh approximation: llama.cpp's ggml_gelu uses erf here and the
 * two differ in the third decimal, which is enough to move a greedy argmax. */
static float gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }

/* NEOX rope with optional per-lane frequency factors. rot is the number of lanes that
 * rotate, which for this family is the whole head. */
static void rope_freq(float *x, int pos, int head_dim, int rot, float base,
                      const float *factors) {
    int half = rot / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, 2.0f * (float)i / (float)rot);
        if (factors) freq /= factors[i];
        float angle = (float)pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = x[i], x1 = x[i + half];
        x[i]        = x0 * cs - x1 * sn;
        x[i + half] = x0 * sn + x1 * cs;
    }
    (void)head_dim;
}

static void gemma4_forward(void *model, kv_cache *kv, const int *tokens, int n,
                           int pos0, float *logits) {
    gemma4_model *m = (gemma4_model*)model;
    int E = m->embed, H = m->n_heads, KVH = m->n_kv_heads, P = m->ple_dim;
    int NL = m->n_layers;
    float eps = m->rms_eps;

    float *x  = (float*)calloc((size_t)n * E, sizeof(float));
    float *xn = (float*)calloc((size_t)n * E, sizeof(float));
    /* One row of the second table per token: 8960 values, one 256-wide slice per layer. */
    float *ple = (float*)calloc((size_t)n * P * NL, sizeof(float));
    float *ple_row = (float*)calloc((size_t)P * NL, sizeof(float));
    float *proj = (float*)calloc((size_t)P * NL, sizeof(float));
    if (!x || !xn || !ple || !ple_row || !proj) {
        free(x); free(xn); free(ple); free(ple_row); free(proj);
        return;
    }

    double pft = pf_mark();
    float emb_scale = sqrtf((float)E), ple_scale = sqrtf((float)P);
    for (int j = 0; j < n; j++) {
        float *xj = x + (long)j * E;
        if (m->tok_emb.f32) memcpy(xj, m->tok_emb.f32 + (long)tokens[j] * E, E * sizeof(float));
        else if (gguf_dequant_row(m->gf, m->emb_ti, (uint64_t)tokens[j], xj) != 0)
            memset(xj, 0, E * sizeof(float));
        for (int i = 0; i < E; i++) xj[i] *= emb_scale;

        /* per-layer input = ( norm(model_proj @ x / sqrt(E)) + table_row * sqrt(P) ) / sqrt(2) */
        if (gguf_dequant_row(m->gf, m->ple_ti, (uint64_t)tokens[j], ple_row) != 0)
            memset(ple_row, 0, (size_t)P * NL * sizeof(float));
        mm_t(proj, xj, m->ple_proj_w, 1, E, P * NL);
        float inv_e = 1.0f / sqrtf((float)E), inv_sqrt2 = 1.0f / sqrtf(2.0f);
        for (int l = 0; l < NL; l++) {
            float *pl = proj + (long)l * P;
            for (int i = 0; i < P; i++) pl[i] *= inv_e;
            rmsnorm(pl, pl, m->ple_proj_norm, P, eps);
            float *dst = ple + ((long)j * NL + l) * P;
            for (int i = 0; i < P; i++)
                dst[i] = (pl[i] + ple_row[(long)l * P + i] * ple_scale) * inv_sqrt2;
        }
    }
    pf_add(PF_EMBED, pft);

    int maxq = 0, maxkv = 0, maxffn = 0;
    for (int l = 0; l < NL; l++) {
        if (m->layers[l].q_dim > maxq) maxq = m->layers[l].q_dim;
        if (m->layers[l].kv_dim > maxkv) maxkv = m->layers[l].kv_dim;
        if (m->layers[l].ffn > maxffn) maxffn = m->layers[l].ffn;
    }
    float *q_all    = (float*)calloc((size_t)n * maxq, sizeof(float));
    float *k_new    = (float*)calloc((size_t)n * maxkv, sizeof(float));
    float *v_new    = (float*)calloc((size_t)n * maxkv, sizeof(float));
    float *attn_out = (float*)calloc((size_t)n * maxq, sizeof(float));
    float *proj_out = (float*)calloc((size_t)n * E, sizeof(float));
    float *g_buf    = (float*)calloc((size_t)n * maxffn, sizeof(float));
    float *u_buf    = (float*)calloc((size_t)n * maxffn, sizeof(float));
    float *pe_in    = (float*)calloc((size_t)n * E, sizeof(float));
    float *pe_buf   = (float*)calloc((size_t)n * P, sizeof(float));

    for (int l = 0; l < NL; l++) {
        int HD = m->layers[l].head_dim, Q_DIM = m->layers[l].q_dim;
        int KVD = m->layers[l].kv_dim, FFN = m->layers[l].ffn;
        int gqa = H / KVH;
        float base = m->layers[l].is_swa ? m->rope_base_swa : m->rope_base;
        const float *factors = m->layers[l].is_swa ? NULL : m->rope_freqs;
        long cache_base = (long)m->layers[l].kv_slot * kv->max_seq * kv->kv_dim;

        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (long)j * E, x + (long)j * E, m->layers[l].attn_norm, E, eps);
        pf_add(PF_NORM, pft);

        pft = pf_mark();
        qmm(q_all, &m->layers[l].wq, xn, n);
        int owns_kv = (l < m->n_kv_layers);
        if (owns_kv) {
            qmm(k_new, &m->layers[l].wk, xn, n);
            qmm(v_new, &m->layers[l].wv, xn, n);
        }
        pf_add(PF_QKV, pft);

        pft = pf_mark();
        for (int j = 0; j < n; j++) {
            int pos = pos0 + j;
            float *qj = q_all + (long)j * Q_DIM;
            for (int h = 0; h < H; h++) {
                rmsnorm(qj + h * HD, qj + h * HD, m->layers[l].q_norm, HD, eps);
                rope_freq(qj + h * HD, pos, HD, HD, base, factors);
            }
            if (owns_kv) {
                float *kj = k_new + (long)j * KVD, *vj = v_new + (long)j * KVD;
                for (int h = 0; h < KVH; h++) {
                    rmsnorm(kj + h * HD, kj + h * HD, m->layers[l].k_norm, HD, eps);
                    rmsnorm_plain(vj + h * HD, vj + h * HD, HD, eps);
                    rope_freq(kj + h * HD, pos, HD, HD, base, factors);
                }
                memcpy(kv->k + cache_base + (long)pos * kv->kv_dim, kj, KVD * sizeof(float));
                memcpy(kv->v + cache_base + (long)pos * kv->kv_dim, vj, KVD * sizeof(float));
            }
        }
        pf_add(PF_ROPE, pft);

        /* Attention with the scale left out on purpose: this family sets it to 1. */
        pft = pf_mark();
        memset(attn_out, 0, (size_t)n * Q_DIM * sizeof(float));
        for (int j = 0; j < n; j++) {
            int pos = pos0 + j;
            int first = 0;
            if (m->layers[l].is_swa && pos >= m->swa_window) first = pos - m->swa_window + 1;
            int span = pos - first + 1;
            for (int h = 0; h < H; h++) {
                int kvh = h / gqa;
                const float *q = q_all + (long)j * Q_DIM + h * HD;
                float *scores = (float*)calloc(span, sizeof(float));
                for (int t = 0; t < span; t++) {
                    const float *kt = kv->k + cache_base + (long)(first + t) * kv->kv_dim + kvh * HD;
                    scores[t] = dot_f32(q, kt, HD);
                }
                softmax(scores, span);
                float *out_h = attn_out + (long)j * Q_DIM + h * HD;
                for (int t = 0; t < span; t++) {
                    const float *vt = kv->v + cache_base + (long)(first + t) * kv->kv_dim + kvh * HD;
                    axpy_f32(out_h, scores[t], vt, HD);
                }
                free(scores);
            }
        }
        pf_add(PF_ATTN, pft);

        pft = pf_mark();
        qmm(proj_out, &m->layers[l].wo, attn_out, n);
        for (int j = 0; j < n; j++)
            rmsnorm(proj_out + (long)j * E, proj_out + (long)j * E,
                    m->layers[l].post_attn_norm, E, eps);
        for (long i = 0; i < (long)n * E; i++) x[i] += proj_out[i];   /* attn_out = cur + x */
        pf_add(PF_PROJ, pft);

        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (long)j * E, x + (long)j * E, m->layers[l].ffn_norm, E, eps);
        qmm(g_buf, &m->layers[l].wgate, xn, n);
        qmm(u_buf, &m->layers[l].wup, xn, n);
        pf_add(PF_FFN, pft);

        pft = pf_mark();
        for (long i = 0; i < (long)n * FFN; i++) g_buf[i] = gelu(g_buf[i]) * u_buf[i];
        pf_add(PF_SILU, pft);

        pft = pf_mark();
        qmm(proj_out, &m->layers[l].wdown, g_buf, n);
        for (int j = 0; j < n; j++)
            rmsnorm(proj_out + (long)j * E, proj_out + (long)j * E,
                    m->layers[l].post_ffw_norm, E, eps);
        for (long i = 0; i < (long)n * E; i++) x[i] += proj_out[i];
        pf_add(PF_FFN, pft);

        /* Per-layer embedding: gate the block output through this layer's slice of the
         * second table, project it back and add. */
        pft = pf_mark();
        memcpy(pe_in, x, (size_t)n * E * sizeof(float));
        qmm(pe_buf, &m->layers[l].ple_gate, x, n);
        for (int j = 0; j < n; j++) {
            float *pj = pe_buf + (long)j * P;
            const float *slice = ple + ((long)j * NL + l) * P;
            for (int i = 0; i < P; i++) pj[i] = gelu(pj[i]) * slice[i];
        }
        qmm(proj_out, &m->layers[l].ple_proj, pe_buf, n);
        for (int j = 0; j < n; j++)
            rmsnorm(proj_out + (long)j * E, proj_out + (long)j * E,
                    m->layers[l].ple_post_norm, E, eps);
        for (long i = 0; i < (long)n * E; i++) x[i] = pe_in[i] + proj_out[i];

        float os = m->layers[l].out_scale;
        if (os != 1.0f) for (long i = 0; i < (long)n * E; i++) x[i] *= os;
        pf_add(PF_RESID, pft);
    }

    if (logits) {
        pft = pf_mark();
        rmsnorm(xn, x + (long)(n - 1) * E, m->out_norm, E, eps);
        qmv(logits, &m->tok_emb, xn);
        if (m->softcap > 0.0f)
            for (int i = 0; i < m->vocab; i++)
                logits[i] = tanhf(logits[i] / m->softcap) * m->softcap;
        pf_add(PF_HEAD, pft);
    }

    free(x); free(xn); free(ple); free(ple_row); free(proj);
    free(q_all); free(k_new); free(v_new); free(attn_out); free(proj_out);
    free(g_buf); free(u_buf); free(pe_in); free(pe_buf);
}

static const char *const gemma4_names[] = { "gemma4", NULL };

const nt_arch nt_arch_gemma4 = {
    gemma4_names,
    gemma4_load,
    gemma4_free,
    gemma4_forward,
};
