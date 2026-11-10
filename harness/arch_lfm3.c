/* arch_lfm3.c — LiquidAI LFM3: two kinds of layer in one body, and the first
 * hybrid the harness runs.
 *
 * Mamba was the first family here with no attention; LFM3 is the first that
 * mixes the two. Some layers are a gated short convolution over the channel
 * dimension — they carry a window of recent inputs per channel and no keys or
 * values at all — and the rest are ordinary attention with per-head query and
 * key norms, partial RoPE on the layers that see everything (one rope_freqs
 * table for all of them, freezing the pairs the model does not rotate), and a
 * plain base on the layers held to a sliding window. Which is which is not
 * announced by the weights: a layer with no attn_q owns a convolution, but the
 * sliding-versus-full split among attention layers lives only in the
 * lfm3.attention.sliding_window_pattern metadata array — both kinds carry the
 * same tensors and the rope table is global — so the array is read, and a file
 * that does not carry it is refused rather than routed by guesswork.
 *
 * The two things this family does that nothing here had done before:
 *
 *  - The attention layers gate their own output per head. A small matrix reads
 *    the normed layer input down to one number per head, a sigmoid turns it
 *    into a gate, and each head's answer is multiplied by its own gate before
 *    the output projection. It costs a matvec the width of the model and gives
 *    the layer a volume knob per head.
 *
 *  - The convolution layers are gated before they convolve: in_proj produces
 *    three copies of the input (b, c, x), the convolved signal is b * x and the
 *    answer is c * conv(b * x). No activation after the window — the gating is
 *    the nonlinearity, and out_proj takes the result straight back to width.
 *    The state is the last l_cache-1 gated inputs per channel. It belongs to
 *    the sequence, not to the weights: it lives in the family's cache next to
 *    the keys, is cleared when a sequence starts over, and a continuation is
 *    accepted only at the position the cache last reached.
 *
 * The arithmetic follows llama.cpp's graph for this family (src/models/lfm3.cpp
 * and the shortconv in src/models/lfm-base.cpp), which is the reference the
 * output is compared against: same file, same prompt, temperature 0, same
 * tokens or it is wrong.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Bounds on the keys every shape is computed from, so 3 * embed * sizeof(float)
 * and its kin stay inside int before any comparison is made. */
#define LFM3_MAX_EMBED (INT_MAX / 12)
#define LFM3_MAX_HEAD  4096
#define LFM3_MAX_LC    4096

typedef struct {
    int n_layers, n_recr, n_attn, embed, vocab, max_kv;
    int head_dim, n_rot, l_cache, swa_window;
    float rope_base, rope_base_swa, rms_eps;

    gguf_file *gf;
    int emb_ti, has_output_weight;

    wt tok_emb;            /* token_embd, also the head when output.weight is absent */
    wt out_weight;         /* output.weight — absent in LiquidAI's files, which tie it */
    float *out_norm;
    float *rope_freqs;     /* rope_freqs.weight, n_rot/2 divisors, full-attention layers only */

    struct {
        int is_recr;       /* convolution layer: no qkv at all */
        int is_swa;        /* attention layer held to the sliding window */
        int kv_slot;       /* attention layers only; conv layers use recr_slot */
        int recr_slot;
        int n_head, n_kv_head, ffn;
        float *attn_norm, *q_norm, *k_norm, *post_attn_norm;
        float *ffn_norm, *post_ffn_norm;
        wt wq, wk, wv, wo, wgate, wup, wdown;      /* attention layers' FFN */
        wt gate_head;      /* attn_gate: [n_head, embed] — the per-head volume knob */
        wt in_proj, out_proj;                        /* convolution layers */
        float *conv_w;    /* [embed][l_cache], one window per channel */
    } layers[];
} lfm3_model;

/* The family's cache. k/v hold the attention layers only, rows kvd wide; conv
 * holds one window slice per convolution layer. base.kv_dim carries a sentinel
 * so a host cache from kv_new, which has no conv slice, is refused instead of
 * being read past its end. */
#define LFM3_KV_SENTINEL (-3)
typedef struct {
    kv_cache base;
    const lfm3_model *owner;
    int next, kvd;
    float *conv;           /* [n_recr][embed][l_cache-1] */
} lfm3_cache;

static int lfm3_key_int(gguf_file *gf, const char *key, int fallback) {
    const gguf_kv *kv = gguf_get_kv(gf, key);
    if (!kv) return fallback;
    if (kv->type == 4) return (int)kv->val.u32;
    if (kv->type == 5) return kv->val.i32;
    return fallback;
}

static float lfm3_key_f32(gguf_file *gf, const char *key, float fallback) {
    const gguf_kv *kv = gguf_get_kv(gf, key);
    return kv ? kv->val.f32 : fallback;
}

/* A tensor this family reads has the rank it expects, no empty or int-sized-out
 * dimension, and a shape that multiplies out — without wrapping — to the element
 * count the file states. Anything else is a crafted or broken header. */
static int lfm3_dims(const gguf_file *gf, int ti, uint32_t ndim) {
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->ndim != ndim) return 0;
    uint64_t n = 1;
    for (uint32_t d = 0; d < ndim; d++)
        if (!t->shape[d] || t->shape[d] > (uint64_t)INT_MAX ||
            __builtin_mul_overflow(n, t->shape[d], &n))
            return 0;
    return n == t->n_elements;
}

static int lfm3_dims_named(gguf_file *gf, const char *name, uint32_t ndim) {
    int ti = gguf_find_tensor(gf, name);
    if (ti >= 0 && lfm3_dims(gf, ti, ndim)) return 1;
    fprintf(stderr, "lfm3: '%s' is not a %u-D tensor with a shape that multiplies out\n",
            name, ndim);
    return 0;
}

/* A vector the forward reads n values of: absent is NULL (refused when required),
 * present with another length is refused — rmsnorm would read past it. */
static float *lfm3_vec(gguf_file *gf, const char *name, int n, int required, int *bad) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) {
        if (required) { fprintf(stderr, "lfm3: required tensor '%s' missing\n", name); *bad = 1; }
        return NULL;
    }
    if (!lfm3_dims(gf, ti, 1)) {
        fprintf(stderr, "lfm3: '%s' is not a 1-D tensor with a consistent shape\n", name);
        *bad = 1;
        return NULL;
    }
    if (gf->tensors[ti].n_elements != (uint64_t)n) {
        fprintf(stderr, "lfm3: '%s' holds %llu values, %d expected\n", name,
                (unsigned long long)gf->tensors[ti].n_elements, n);
        *bad = 1;
        return NULL;
    }
    float *v = gguf_dequant(gf, ti);
    if (!v) { fprintf(stderr, "lfm3: '%s' unreadable\n", name); *bad = 1; }
    return v;
}

/* Compared in 64 bits: an expected row count such as 3 * embed is formed by the
 * caller as uint64_t, so a hostile key cannot wrap it into agreement. */
static int lfm3_shape(const wt *w, uint64_t rows, uint64_t cols, int l, const char *what) {
    if (w->rows > 0 && w->cols > 0 && (uint64_t)w->rows == rows && (uint64_t)w->cols == cols)
        return 1;
    fprintf(stderr, "lfm3: layer %d %s is %d x %d, %llu x %llu expected\n", l, what,
            w->rows, w->cols, (unsigned long long)rows, (unsigned long long)cols);
    return 0;
}

static void lfm3_free(void *model);

static void *lfm3_load(gguf_file *gf, nt_dims *dims) {
    if (!gf || !dims) return NULL;
    if (gf->n_layers < 1 || gf->n_layers > GGUF_MAX_TENSORS) {
        fprintf(stderr, "lfm3: block_count %u outside 1..%d; refusing\n",
                (unsigned)gf->n_layers, GGUF_MAX_TENSORS);
        return NULL;
    }
    int nl = gf->n_layers;
    lfm3_model *m = (lfm3_model*)calloc(1, sizeof(lfm3_model) + (size_t)nl * sizeof(m->layers[0]));
    if (!m) return NULL;
    m->gf = gf;
    m->n_layers = nl;

    /* Which attention layer is which is NOT derivable from the shapes: sliding and
     * full layers carry the same tensors and the rope table is one for all of them.
     * The file decides, and a file that does not say — refuses. */
    int n_pat = 0;
    uint8_t *swa_pat = gguf_read_bool_array(gf->path, "lfm3.attention.sliding_window_pattern", &n_pat);
    if (!swa_pat || n_pat != nl) {
        fprintf(stderr, "lfm3: sliding_window_pattern array %s (%d entries for %d layers) — "
                        "the routing is not derivable from the weights, refusing\n",
                swa_pat ? "has the wrong length" : "is missing", n_pat, nl);
        free(swa_pat);
        lfm3_free(m);
        return NULL;
    }
    for (int l = 0; l < nl; l++) {
        m->layers[l].is_swa = swa_pat[l] ? 1 : 0;
    }
    free(swa_pat);

    /* Read by exact key. gguf.c fills its convenience fields by suffix, and this
     * family pairs two keys per suffix the way gemma4 does: rope.freq_base and
     * rope.freq_base_swa. Whichever came last in the file would have won. */
    m->embed     = lfm3_key_int(gf, "lfm3.embedding_length", gf->embed_dim);
    m->head_dim  = lfm3_key_int(gf, "lfm3.attention.key_length", 0);
    m->l_cache   = lfm3_key_int(gf, "lfm3.shortconv.l_cache", 0);
    m->swa_window= lfm3_key_int(gf, "lfm3.attention.sliding_window", 0);
    m->n_rot     = lfm3_key_int(gf, "lfm3.rope.dimension_count", m->head_dim);
    m->rms_eps   = lfm3_key_f32(gf, "lfm3.attention.layer_norm_rms_epsilon", gf->rms_eps);
    m->rope_base = lfm3_key_f32(gf, "lfm3.rope.freq_base", gf->rope_freq_base);
    m->rope_base_swa = lfm3_key_f32(gf, "lfm3.rope.freq_base_swa", 10000.0f);
    int val_dim = lfm3_key_int(gf, "lfm3.attention.value_length", m->head_dim);

    if (m->head_dim <= 0 || m->embed <= 0) {
        fprintf(stderr, "lfm3: key_length/embedding_length missing or unusable "
                        "(head=%d embed=%d)\n", m->head_dim, m->embed);
        free(m);
        return NULL;
    }
    if (m->embed > LFM3_MAX_EMBED || m->head_dim > LFM3_MAX_HEAD || m->l_cache > LFM3_MAX_LC) {
        fprintf(stderr, "lfm3: embedding_length %d, key_length %d or shortconv.l_cache %d "
                        "exceeds %d / %d / %d; refusing before any shape is computed from it\n",
                m->embed, m->head_dim, m->l_cache, LFM3_MAX_EMBED, LFM3_MAX_HEAD, LFM3_MAX_LC);
        free(m);
        return NULL;
    }
    if (val_dim != m->head_dim) {
        fprintf(stderr, "lfm3: key_length %d and value_length %d differ; "
                        "the reference refuses this file and so does this\n",
                m->head_dim, val_dim);
        free(m);
        return NULL;
    }
    if (m->l_cache < 2) {
        fprintf(stderr, "lfm3: shortconv.l_cache %d is below the causal minimum of 2\n",
                m->l_cache);
        free(m);
        return NULL;
    }
    if (m->n_rot <= 0 || m->n_rot > m->head_dim || m->n_rot % 2) {
        fprintf(stderr, "lfm3: rope.dimension_count %d unusable for head %d\n",
                m->n_rot, m->head_dim);
        free(m);
        return NULL;
    }
    /* A d1 checkpoint reads its prompt both ways on the full-attention layers and
     * centres the convolution window. This forward is causal only, so such a file
     * is refused rather than run the wrong way round. */
    const gguf_kv *causal = gguf_get_kv(gf, "lfm3.attention.causal");
    if (causal && causal->type == 7 && !causal->val.b) {
        fprintf(stderr, "lfm3: attention.causal is false (a non-causal checkpoint); "
                        "this forward is causal only, refusing\n");
        free(m);
        return NULL;
    }
    int any_swa = 0;
    for (int l = 0; l < nl; l++) any_swa |= m->layers[l].is_swa;
    if (any_swa && (m->swa_window <= 0 || !gguf_get_kv(gf, "lfm3.rope.freq_base_swa"))) {
        fprintf(stderr, "lfm3: sliding layers without attention.sliding_window or "
                        "rope.freq_base_swa; the reference requires both, refusing\n");
        free(m);
        return NULL;
    }

    int missing = 0;
    /* The vocabulary is the embedding's row count — the number of rows a token id
     * may index — not a header field read beside it. A vocab_size key that names
     * another count is a contradiction and is refused. */
    int ti = gguf_find_tensor(gf, "token_embd.weight");
    m->emb_ti = ti;
    if (ti < 0 || !lfm3_dims_named(gf, "token_embd.weight", 2) ||
        !wt_load(&m->tok_emb, gf, "token_embd.weight")) {
        fprintf(stderr, "lfm3: token_embd.weight missing, unreadable or not 2-D\n");
        free(m);
        return NULL;
    }
    m->vocab = m->tok_emb.rows;
    const gguf_kv *vs = gguf_get_kv(gf, "lfm3.vocab_size");
    if (vs && vs->type == 4 && (uint64_t)vs->val.u32 != (uint64_t)m->vocab) {
        fprintf(stderr, "lfm3: lfm3.vocab_size %u but token_embd has %d rows; refusing\n",
                vs->val.u32, m->vocab);
        missing = 1;
    }
    if (!lfm3_shape(&m->tok_emb, (uint64_t)m->vocab, (uint64_t)m->embed, -1, "token_embd"))
        missing = 1;
    m->has_output_weight = gguf_find_tensor(gf, "output.weight") >= 0;
    if (m->has_output_weight &&
        (!lfm3_dims_named(gf, "output.weight", 2) || !wt_load(&m->out_weight, gf, "output.weight") ||
         !lfm3_shape(&m->out_weight, (uint64_t)m->vocab, (uint64_t)m->embed, -1, "output")))
        missing = 1;
    m->out_norm = lfm3_vec(gf, "output_norm.weight", m->embed, 1, &missing);
    /* One table for every full-attention layer, as the converter writes it and the
     * reference loads it (TENSOR_DUPLICATED): 1.0 on rotated pairs, 1e30 on the pairs
     * partial RoPE leaves still. */
    int rf = gguf_find_tensor(gf, "rope_freqs.weight");
    if (rf >= 0) {
        if (!lfm3_dims(gf, rf, 1) || gf->tensors[rf].n_elements != (uint64_t)(m->n_rot / 2)) {
            fprintf(stderr, "lfm3: rope_freqs.weight has %llu entries, n_rot/2 is %d\n",
                    (unsigned long long)gf->tensors[rf].n_elements, m->n_rot / 2);
            missing = 1;
        }
        m->rope_freqs = gguf_dequant(gf, rf);
        if (!m->rope_freqs) missing = 1;
    }
    if (!(m->tok_emb.q || m->tok_emb.f32) || !m->out_norm) {
        fprintf(stderr, "lfm3: missing critical weights\n");
        missing = 1;
    }

    /* Convolution or attention is read off the tensors: a layer without attn_q
     * owns a convolution. Sliding or full came from the pattern array above.
     * The shapes then carry the geometry, and each is checked against the keys
     * before the forward may read it. */
    for (int l = 0; l < nl; l++) {
        char name[128];
        int E = m->embed, HD = m->head_dim;
        #define V(field, fmt, len, req) do { \
            snprintf(name, sizeof(name), fmt, l); \
            m->layers[l].field = lfm3_vec(gf, name, len, req, &missing); \
        } while (0)
        #define W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            if (!lfm3_dims_named(gf, name, 2) || !wt_load(&m->layers[l].field, gf, name)) { \
                fprintf(stderr, "lfm3: required tensor '%s' missing, malformed or unreadable\n", name); \
                missing = 1; \
            } \
        } while (0)
        #define OPT(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            wt_load(&m->layers[l].field, gf, name); \
        } while (0)

        V(attn_norm,      "blk.%d.attn_norm.weight", E, 1);
        V(ffn_norm,       "blk.%d.ffn_norm.weight", E, 1);
        V(post_attn_norm, "blk.%d.post_attention_norm.weight", E, 0);   /* sandwich */
        V(post_ffn_norm,  "blk.%d.post_ffw_norm.weight", E, 0);
        {
            char pa[128], pf[128];
            snprintf(pa, sizeof(pa), "blk.%d.post_attention_norm.weight", l);
            snprintf(pf, sizeof(pf), "blk.%d.post_ffw_norm.weight", l);
            if ((gguf_find_tensor(gf, pa) < 0) != (gguf_find_tensor(gf, pf) < 0)) {
                fprintf(stderr, "lfm3: layer %d carries one post norm without the other; "
                                "the reference refuses this file and so does this\n", l);
                missing = 1;
            }
        }
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        m->layers[l].is_recr = (gguf_find_tensor(gf, name) < 0);

        if (m->layers[l].is_recr) {
            W(in_proj,  "blk.%d.shortconv.in_proj.weight");
            W(out_proj, "blk.%d.shortconv.out_proj.weight");
            if (!lfm3_shape(&m->layers[l].in_proj, 3ULL * (uint64_t)E, (uint64_t)E, l, "shortconv.in_proj") ||
                !lfm3_shape(&m->layers[l].out_proj, (uint64_t)E, (uint64_t)E, l, "shortconv.out_proj"))
                missing = 1;
            snprintf(name, sizeof(name), "blk.%d.shortconv.conv.weight", l);
            int ci = gguf_find_tensor(gf, name);
            if (ci < 0 || !lfm3_dims(gf, ci, 2) || gf->tensors[ci].shape[0] != (uint64_t)m->l_cache ||
                gf->tensors[ci].n_elements != (uint64_t)E * (uint64_t)m->l_cache) {
                fprintf(stderr, "lfm3: '%s' missing or not [%d x %d]\n", name, m->l_cache, E);
                missing = 1;
            } else if (!(m->layers[l].conv_w = gguf_dequant(gf, ci))) {
                fprintf(stderr, "lfm3: '%s' unreadable\n", name);
                missing = 1;
            }
            m->layers[l].recr_slot = m->n_recr++;
        } else {
            W(wq,        "blk.%d.attn_q.weight");
            W(wk,        "blk.%d.attn_k.weight");
            W(wv,        "blk.%d.attn_v.weight");
            W(wo,        "blk.%d.attn_output.weight");
            W(gate_head, "blk.%d.attn_gate.weight");
            V(q_norm,    "blk.%d.attn_q_norm.weight", HD, 1);
            V(k_norm,    "blk.%d.attn_k_norm.weight", HD, 1);
            m->layers[l].kv_slot = m->n_attn++;
            int qr = m->layers[l].wq.rows, kr = m->layers[l].wk.rows;
            m->layers[l].n_head    = qr / HD;
            m->layers[l].n_kv_head = kr / HD;
            int H = m->layers[l].n_head, KVH = m->layers[l].n_kv_head;
            if (qr % HD || kr % HD || H <= 0 || KVH <= 0 || H % KVH) {
                fprintf(stderr, "lfm3: layer %d attention geometry unusable "
                                "(q rows %d, k rows %d, head %d)\n", l, qr, kr, HD);
                missing = 1;
            } else if (!lfm3_shape(&m->layers[l].wq, qr, E, l, "attn_q") ||
                       !lfm3_shape(&m->layers[l].wk, kr, E, l, "attn_k") ||
                       !lfm3_shape(&m->layers[l].wv, kr, E, l, "attn_v") ||
                       !lfm3_shape(&m->layers[l].wo, E, qr, l, "attn_output") ||
                       !lfm3_shape(&m->layers[l].gate_head, H, E, l, "attn_gate")) {
                missing = 1;
            }
        }
        W(wgate, "blk.%d.ffn_gate.weight");
        W(wup,   "blk.%d.ffn_up.weight");
        W(wdown, "blk.%d.ffn_down.weight");
        m->layers[l].ffn = m->layers[l].wgate.rows;
        if (!lfm3_shape(&m->layers[l].wgate, m->layers[l].ffn, E, l, "ffn_gate") ||
            !lfm3_shape(&m->layers[l].wup, m->layers[l].ffn, E, l, "ffn_up") ||
            !lfm3_shape(&m->layers[l].wdown, E, m->layers[l].ffn, l, "ffn_down"))
            missing = 1;
        #undef V
        #undef W
        #undef OPT
    }

    if (m->n_attn == 0) {
        fprintf(stderr, "lfm3: no attention layers; this family is a hybrid and "
                        "an all-convolution file means the loader misread it\n");
        missing = 1;
    }

    int max_kv = 0;
    for (int l = 0; l < nl; l++)
        if (!m->layers[l].is_recr && m->layers[l].n_kv_head * m->head_dim > max_kv)
            max_kv = m->layers[l].n_kv_head * m->head_dim;
    m->max_kv = max_kv;

    fprintf(stderr, "lfm3: E=%d V=%d L=%d (attn %d, conv %d) head=%d window=%d "
                    "l_cache=%d rope %.0f/%.0f eps=%g\n",
            m->embed, m->vocab, nl, m->n_attn, m->n_recr, m->head_dim,
            m->swa_window, m->l_cache, (double)m->rope_base,
            (double)m->rope_base_swa, (double)m->rms_eps);

    if (missing) { lfm3_free(m); return NULL; }

    dims->n_layers = m->n_attn;   /* only attention layers store keys */
    dims->kv_dim   = max_kv;
    dims->vocab    = m->vocab;
    return m;
}

static void lfm3_free(void *model) {
    lfm3_model *m = (lfm3_model*)model;
    if (!m) return;
    free(m->tok_emb.f32); free(m->out_weight.f32); free(m->out_norm);
    free(m->rope_freqs);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm); free(m->layers[l].ffn_norm);
        free(m->layers[l].post_attn_norm); free(m->layers[l].post_ffn_norm);
        free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].wq.f32); free(m->layers[l].wk.f32); free(m->layers[l].wv.f32);
        free(m->layers[l].wo.f32); free(m->layers[l].gate_head.f32);
        free(m->layers[l].wgate.f32); free(m->layers[l].wup.f32); free(m->layers[l].wdown.f32);
        free(m->layers[l].in_proj.f32); free(m->layers[l].out_proj.f32);
        free(m->layers[l].conv_w);
    }
    free(m);
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

/* NEOX rope with optional per-lane frequency divisors — ggml's freq_factors and the
 * gemma4 family's convention, rotated over n_rot lanes of the head. */
static void rope_lanes(float *x, int pos, int n_rot, float base, const float *factors) {
    int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, 2.0f * (float)i / (float)n_rot);
        if (factors) freq /= factors[i];
        float angle = (float)pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = x[i], x1 = x[i + half];
        x[i]        = x0 * cs - x1 * sn;
        x[i + half] = x0 * sn + x1 * cs;
    }
}

/* One convolution layer over a group of positions. The gated input b*x is the
 * signal; the window is the state plus the group, exactly the walk Mamba's
 * kernel takes, with no activation after it — the gating is the nonlinearity. */
static int lfm3_conv_layer(lfm3_model *m, int l, float *conv, const float *xn, float *y_out,
                           int n) {
    int E = m->embed, LC = m->l_cache;
    int w_rows = m->layers[l].in_proj.rows;   /* 3*E */
    float *bcx = (float*)malloc((size_t)n * w_rows * sizeof(float));
    float *y   = (float*)malloc((size_t)n * E * sizeof(float));
    if (!bcx || !y) { free(bcx); free(y); return NT_E_MEMORY; }

    qmm(bcx, &m->layers[l].in_proj, xn, n);

    float *state = conv + (size_t)m->layers[l].recr_slot * E * (LC - 1);
    const float *w = m->layers[l].conv_w;    /* [E][LC], w[i*LC + k]; k = LC-1 is now */

    for (int j = 0; j < n; j++) {
        const float *row = bcx + (size_t)j * w_rows;
        const float *b = row, *c = row + E, *x = row + 2 * E;
        float *yj = y + (size_t)j * E;
        for (int i = 0; i < E; i++) {
            float z = b[i] * x[i];
            float *st = state + (size_t)i * (LC - 1);
            const float *wi = w + (size_t)i * LC;
            float sum = 0.0f;
            for (int k = 0; k < LC - 1; k++) sum += st[k] * wi[k];
            sum += z * wi[LC - 1];
            for (int k = 0; k + 1 < LC - 1; k++) st[k] = st[k + 1];
            st[LC - 2] = z;
            yj[i] = c[i] * sum;
        }
    }
    qmm(y_out, &m->layers[l].out_proj, y, n);
    free(bcx); free(y);
    return NT_OK;
}

/* rows[j], where given, stands in for the embedding of tokens[j]: an image patch, an
 * audio frame or a speech frame's codebook sum, put where the placeholder id sits. */
static int lfm3_run(void *model, kv_cache *kv, const int *tokens, const float *const *rows,
                    int n, int pos0, float *logits, float *hidden, nt_residual_fn pre_ffn,
                    nt_residual_fn post_block, void *user) {
    lfm3_model *m = (lfm3_model*)model;
    if (!m) return NT_E_ARG;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->n_attn, 0);
    if (rc != NT_OK) return rc;
    if (kv->kv_dim != LFM3_KV_SENTINEL) return NT_E_CACHE;
    lfm3_cache *cache = (lfm3_cache *)kv;
    if (cache->owner != m) return NT_E_CACHE;
    /* The window cannot be rewound, so a continuation is accepted only where the
     * cache stopped; position zero is a new sequence and starts from silence. */
    if (pos0 && pos0 != cache->next) return NT_E_STATE;

    int E = m->embed, HD = m->head_dim, ROT = m->n_rot, KVW = cache->kvd;
    float eps = m->rms_eps;

    float *x  = (float*)calloc((size_t)n * E, sizeof(float));
    float *xn = (float*)calloc((size_t)n * E, sizeof(float));
    if (!x || !xn) { free(x); free(xn); return NT_E_MEMORY; }
    if (pos0 == 0 && m->n_recr > 0)
        memset(cache->conv, 0, (size_t)m->n_recr * E * (m->l_cache - 1) * sizeof(float));
    cache->next = -1;

    double pft = pf_mark();
    for (int j = 0; j < n; j++) {
        float *xj = x + (size_t)j * E;
        if (rows && rows[j]) memcpy(xj, rows[j], E * sizeof(float));
        else if (m->tok_emb.f32) memcpy(xj, m->tok_emb.f32 + (long)tokens[j] * E, E * sizeof(float));
        else if (gguf_dequant_row(m->gf, m->emb_ti, (uint64_t)tokens[j], xj) != 0)
            memset(xj, 0, E * sizeof(float));
    }
    pf_add(PF_EMBED, pft);

    int max_q = 0, max_kv = 0, max_ffn = 0, max_gate = 0;
    for (int l = 0; l < m->n_layers; l++) {
        if (m->layers[l].is_recr) continue;
        if (m->layers[l].n_head * HD > max_q) max_q = m->layers[l].n_head * HD;
        if (m->layers[l].n_kv_head * HD > max_kv) max_kv = m->layers[l].n_kv_head * HD;
        if (m->layers[l].ffn > max_ffn) max_ffn = m->layers[l].ffn;
        if (m->layers[l].n_head > max_gate) max_gate = m->layers[l].n_head;
    }
    for (int l = 0; l < m->n_layers; l++)
        if (m->layers[l].is_recr && m->layers[l].ffn > max_ffn) max_ffn = m->layers[l].ffn;

    float *q_all = (float*)calloc((size_t)n * (max_q ? max_q : 1), sizeof(float));
    float *k_new = (float*)calloc((size_t)n * (max_kv ? max_kv : 1), sizeof(float));
    float *v_new = (float*)calloc((size_t)n * (max_kv ? max_kv : 1), sizeof(float));
    float *attn_out = (float*)calloc((size_t)n * (max_q ? max_q : 1), sizeof(float));
    float *proj_out = (float*)calloc((size_t)n * E, sizeof(float));
    float *gate_buf = (float*)calloc((size_t)n * (max_gate ? max_gate : 1), sizeof(float));
    float *g_buf   = (float*)calloc((size_t)n * (max_ffn ? max_ffn : 1), sizeof(float));
    float *u_buf   = (float*)calloc((size_t)n * (max_ffn ? max_ffn : 1), sizeof(float));
    if (!q_all || !k_new || !v_new || !attn_out || !proj_out || !gate_buf || !g_buf || !u_buf) {
        rc = NT_E_MEMORY;
        goto done;
    }

    for (int l = 0; l < m->n_layers; l++) {
        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (size_t)j * E, x + (size_t)j * E, m->layers[l].attn_norm, E, eps);
        pf_add(PF_NORM, pft);

        if (m->layers[l].is_recr) {
            pft = pf_mark();
            rc = lfm3_conv_layer(m, l, cache->conv, xn, proj_out, n);
            pf_add(PF_ATTN, pft);
            if (rc != NT_OK) goto done;
        } else {
            int H = m->layers[l].n_head, KVH = m->layers[l].n_kv_head;
            int Q_DIM = H * HD, KVD = KVH * HD;
            int gqa = H / KVH;
            float scale = 1.0f / sqrtf((float)HD);
            float base = m->layers[l].is_swa ? m->rope_base_swa : m->rope_base;
            const float *factors = m->layers[l].is_swa ? NULL : m->rope_freqs;
            long cache_base = (long)m->layers[l].kv_slot * kv->max_seq * KVW;

            pft = pf_mark();
            qmm(q_all, &m->layers[l].wq, xn, n);
            qmm(k_new, &m->layers[l].wk, xn, n);
            qmm(v_new, &m->layers[l].wv, xn, n);
            pf_add(PF_QKV, pft);

            pft = pf_mark();
            for (int j = 0; j < n; j++) {
                int pos = pos0 + j;
                float *qj = q_all + (size_t)j * Q_DIM, *kj = k_new + (size_t)j * KVD;
                for (int h = 0; h < H; h++)
                    rmsnorm(qj + h * HD, qj + h * HD, m->layers[l].q_norm, HD, eps);
                for (int h = 0; h < KVH; h++)
                    rmsnorm(kj + h * HD, kj + h * HD, m->layers[l].k_norm, HD, eps);
                for (int h = 0; h < H; h++)
                    rope_lanes(qj + h * HD, pos, ROT, base, factors);
                for (int h = 0; h < KVH; h++)
                    rope_lanes(kj + h * HD, pos, ROT, base, factors);
                memcpy(kv->k + cache_base + (long)pos * KVW, kj, KVD * sizeof(float));
                memcpy(kv->v + cache_base + (long)pos * KVW, v_new + (size_t)j * KVD,
                       KVD * sizeof(float));
            }
            pf_add(PF_ROPE, pft);

            /* The reference's scale is 1/sqrt(head_dim); only the sliding layers
             * clamp their span, and a layer without a window reads everything. */
            pft = pf_mark();
            memset(attn_out, 0, (size_t)n * Q_DIM * sizeof(float));
            for (int j = 0; j < n; j++) {
                int pos = pos0 + j;
                int first = 0;
                if (m->layers[l].is_swa && m->swa_window > 0 && pos >= m->swa_window)
                    first = pos - m->swa_window + 1;
                int span = pos - first + 1;
                for (int h = 0; h < H; h++) {
                    int kvh = h / gqa;
                    const float *q = q_all + (size_t)j * Q_DIM + h * HD;
                    float *scores = (float*)calloc((size_t)span, sizeof(float));
                    if (!scores) { rc = NT_E_MEMORY; goto done; }
                    for (int t = 0; t < span; t++) {
                        const float *kt = kv->k + cache_base + (long)(first + t) * KVW + kvh * HD;
                        scores[t] = dot_f32(q, kt, HD) * scale;
                    }
                    softmax(scores, span);
                    float *out_h = attn_out + (size_t)j * Q_DIM + h * HD;
                    for (int t = 0; t < span; t++) {
                        const float *vt = kv->v + cache_base + (long)(first + t) * KVW + kvh * HD;
                        axpy_f32(out_h, scores[t], vt, HD);
                    }
                    free(scores);
                }
            }
            pf_add(PF_ATTN, pft);

            /* Per-head volume knob: one sigmoid per head off the normed input,
             * multiplied into that head's answer before the projection. */
            pft = pf_mark();
            qmm(gate_buf, &m->layers[l].gate_head, xn, n);
            for (int j = 0; j < n; j++) {
                float *gj = gate_buf + (size_t)j * H;
                for (int h = 0; h < H; h++) gj[h] = 1.0f / (1.0f + expf(-gj[h]));
                float *aj = attn_out + (size_t)j * Q_DIM;
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < HD; i++) aj[h * HD + i] *= gj[h];
            }
            pf_add(PF_SILU, pft);

            pft = pf_mark();
            qmm(proj_out, &m->layers[l].wo, attn_out, n);
            pf_add(PF_PROJ, pft);
        }

        /* Both mixers meet the FFN at the same residual, after the optional
         * sandwich norm and addition. */
        pft = pf_mark();
        if (m->layers[l].post_attn_norm)
            for (int j = 0; j < n; j++)
                rmsnorm(proj_out + (size_t)j * E, proj_out + (size_t)j * E,
                        m->layers[l].post_attn_norm, E, eps);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += proj_out[i];
        pf_add(PF_RESID, pft);
        if (pre_ffn && (rc = pre_ffn(user, l, pos0, n, E, x)) != NT_OK) goto done;

        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (size_t)j * E, x + (size_t)j * E, m->layers[l].ffn_norm, E, eps);
        qmm(g_buf, &m->layers[l].wgate, xn, n);
        qmm(u_buf, &m->layers[l].wup, xn, n);
        for (size_t i = 0; i < (size_t)n * m->layers[l].ffn; i++) g_buf[i] = silu(g_buf[i]) * u_buf[i];
        qmm(proj_out, &m->layers[l].wdown, g_buf, n);
        if (m->layers[l].post_ffn_norm)
            for (int j = 0; j < n; j++)
                rmsnorm(proj_out + (size_t)j * E, proj_out + (size_t)j * E,
                        m->layers[l].post_ffn_norm, E, eps);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += proj_out[i];
        pf_add(PF_FFN, pft);
        if (post_block && (rc = post_block(user, l, pos0, n, E, x)) != NT_OK) goto done;
    }

    if (logits || hidden) {
        pft = pf_mark();
        rmsnorm(xn, x + (size_t)(n - 1) * E, m->out_norm, E, eps);
        if (hidden) memcpy(hidden, xn, (size_t)E * sizeof(float));
        if (logits) qmv(logits, m->has_output_weight ? &m->out_weight : &m->tok_emb, xn);
        pf_add(PF_HEAD, pft);
    }

    cache->next = pos0 + n;
done:
    free(x); free(xn); free(q_all); free(k_new); free(v_new); free(attn_out);
    free(proj_out); free(gate_buf); free(g_buf); free(u_buf);
    return rc;
}

static int lfm3_forward_hooks(void *model, kv_cache *kv, const int *tokens, int n,
                              int pos0, float *logits, nt_residual_fn pre_ffn,
                              nt_residual_fn post_block, void *user) {
    return lfm3_run(model, kv, tokens, NULL, n, pos0, logits, NULL, pre_ffn, post_block, user);
}

static int lfm3_forward(void *model, kv_cache *kv, const int *tokens, int n, int pos, float *logits) {
    return lfm3_run(model, kv, tokens, NULL, n, pos, logits, NULL, NULL, NULL, NULL);
}

static int lfm3_forward_residual(void *model, kv_cache *kv, const int *tokens, int n, int pos,
                                 float *logits, nt_residual_fn post_block, void *user) {
    return lfm3_forward_hooks(model, kv, tokens, n, pos, logits, NULL, post_block, user);
}

static const gguf_file *lfm3_describe(const void *model, nt_dims *dims) {
    const lfm3_model *m = model;
    if (!m || !dims || m->gf->embed_dim != m->embed || m->gf->n_layers != m->n_layers) return NULL;
    /* KV counts attention slots. Elections and anchor capture count all blocks. */
    *dims = (nt_dims){m->n_layers, m->max_kv, m->vocab};
    return m->gf;
}

enum { LFM3_RESIDUAL_NORMS = 4 };
static const float *lfm3_doe_norm(const void *model, int index) {
    const lfm3_model *m = model;
    if (!m || index < 0 || index > LFM3_RESIDUAL_NORMS*m->n_layers) return NULL;
    if (index == LFM3_RESIDUAL_NORMS*m->n_layers) return m->out_norm;
    int l = index / LFM3_RESIDUAL_NORMS;
    const float *const norms[] = {m->layers[l].attn_norm, m->layers[l].post_attn_norm,
                                  m->layers[l].ffn_norm, m->layers[l].post_ffn_norm};
    return norms[index % LFM3_RESIDUAL_NORMS];
}

static int lfm3_forward_embd(void *model, kv_cache *kv, const int *tokens,
                             const float *const *rows, int n, int pos0,
                             float *logits, float *hidden) {
    return lfm3_run(model, kv, tokens, rows, n, pos0, logits, hidden, NULL, NULL, NULL);
}

static void lfm3_kv_free(void *model, kv_cache *kv) {
    (void)model;
    if (!kv) return;
    lfm3_cache *c = (lfm3_cache *)kv;
    free(c->base.k); free(c->base.v); free(c->conv); free(c);
}

static int lfm3_cache_cells(int slices, int rows, int cols, size_t *cells) {
    if (slices < 1 || rows < 1 || cols < 1 ||
        (size_t)slices > SIZE_MAX / sizeof(float) / (size_t)rows / (size_t)cols) return 0;
    *cells = (size_t)slices * (size_t)rows * (size_t)cols;
    return 1;
}

static kv_cache *lfm3_kv_new(void *model, int max_seq) {
    lfm3_model *m = (lfm3_model *)model;
    size_t kv_cells, conv_cells;
    if (!m || m->l_cache < 2 ||
        !lfm3_cache_cells(m->n_attn, max_seq, m->max_kv, &kv_cells) ||
        !lfm3_cache_cells(m->n_recr ? m->n_recr : 1, m->embed, m->l_cache-1, &conv_cells)) return NULL;
    lfm3_cache *c = (lfm3_cache *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->owner = m;
    c->kvd = m->max_kv;
    c->base.max_seq = max_seq;
    c->base.n_layers = m->n_attn;
    c->base.kv_dim = LFM3_KV_SENTINEL;
    c->base.k = (float *)calloc(kv_cells, sizeof(float));
    c->base.v = (float *)calloc(kv_cells, sizeof(float));
    c->conv = (float *)calloc(conv_cells, sizeof(float));
    if (!c->base.k || !c->base.v || !c->conv) { lfm3_kv_free(m, &c->base); return NULL; }
    return &c->base;
}

static const char *const lfm3_names[] = { "lfm3", NULL };
static const nt_doe_decoder lfm3_doe = {
    .forward = lfm3_forward_hooks, .describe = lfm3_describe,
    .norm = lfm3_doe_norm, .norms_per_layer = LFM3_RESIDUAL_NORMS,
    .optional_norm_slots = (1u << 1) | (1u << 3)
};

const nt_arch nt_arch_lfm3 = {
    .names   = lfm3_names,
    .load    = lfm3_load,
    .free    = lfm3_free,
    .forward = lfm3_forward,
    .forward_residual = lfm3_forward_residual,
    .kv_new  = lfm3_kv_new,
    .kv_free = lfm3_kv_free,
    .doe = &lfm3_doe,
    .forward_embd = lfm3_forward_embd,
};
