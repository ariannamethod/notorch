/* arch_lfm3.c — LiquidAI LFM3: two kinds of layer in one body, and the first
 * hybrid the harness runs.
 *
 * Mamba was the first family here with no attention; LFM3 is the first that
 * mixes the two. Some layers are a gated short convolution over the channel
 * dimension — they carry a window of recent inputs per channel and no keys or
 * values at all — and the rest are ordinary attention with per-head query and
 * key norms, a learned frequency table on the layers that see everything, and
 * a plain base on the layers held to a sliding window. Which is which is not
 * announced by the weights: a layer with no attn_q owns a convolution, but the
 * sliding-versus-full split among attention layers lives only in the
 * lfm3.attention.sliding_window_pattern metadata array — a file whose layers
 * all ship a frequency table and a file whose layers ship none can carry the
 * same routing, so the array is read, and a file that does not carry it is
 * refused rather than routed by guesswork.
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
 *    The state is the last l_cache-1 gated inputs per channel, kept in the
 *    model like Mamba keeps its own, and cleared when a sequence starts over.
 *
 * The arithmetic follows llama.cpp's graph for this family (src/models/lfm3.cpp
 * and the shortconv in src/models/lfm-base.cpp), which is the reference the
 * output is compared against: same file, same prompt, temperature 0, same
 * tokens or it is wrong.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n_layers, n_recr, n_attn, embed, vocab;
    int head_dim, n_rot, l_cache, swa_window;
    float rope_base, rope_base_swa, rms_eps;

    gguf_file *gf;
    int emb_ti, has_output_weight;

    wt tok_emb;            /* token_embd, also the head when output.weight is absent */
    wt out_weight;         /* output.weight — this family ships it untied */
    float *out_norm;

    /* The recurrent state, one slice per convolution layer. Each channel keeps
     * the last l_cache-1 gated inputs; there is no third place a token could
     * hide in, exactly like Mamba's window. */
    float *conv_state;     /* [n_recr][embed][l_cache-1] */

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
        float *rope_freqs;   /* full-attention layers only, head_dim/2 */
    } layers[];
} lfm3_model;

static float *lfm3_f32(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    return ti >= 0 ? gguf_dequant(gf, ti) : NULL;
}

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

static void lfm3_free(void *model);

static void *lfm3_load(gguf_file *gf, nt_dims *dims) {
    int nl = gf->n_layers;
    lfm3_model *m = (lfm3_model*)calloc(1, sizeof(lfm3_model) + (size_t)nl * sizeof(m->layers[0]));
    if (!m) return NULL;
    m->gf = gf;
    m->n_layers = nl;

    /* Which attention layer is which is NOT derivable from the shapes: the sliding
     * window pattern is a per-layer metadata array, and a file that ships every
     * layer's rope_freqs looks different from one that ships none while keeping
     * the same routing. The file decides, and a file that does not say — refuses. */
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

    int ti = gguf_find_tensor(gf, "token_embd.weight");
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;
    m->emb_ti = ti;
    wt_load(&m->tok_emb, gf, "token_embd.weight");
    m->has_output_weight = wt_load(&m->out_weight, gf, "output.weight");
    m->out_norm = lfm3_f32(gf, "output_norm.weight");

    int missing = 0;
    if (!(m->tok_emb.q || m->tok_emb.f32) || !m->out_norm) {
        fprintf(stderr, "lfm3: missing critical weights\n");
        missing = 1;
    }

    /* Which layer is which is read off the tensors, not off the metadata
     * arrays the converter also writes. A layer without attn_q owns a
     * convolution; an attention layer without rope_freqs is the sliding kind.
     * The shapes then carry the geometry, because a shape cannot disagree
     * with itself. */
    for (int l = 0; l < nl; l++) {
        char name[128];
        #define L(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            m->layers[l].field = lfm3_f32(gf, name); \
        } while (0)
        #define W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            if (!wt_load(&m->layers[l].field, gf, name)) { \
                fprintf(stderr, "lfm3: required tensor '%s' missing or unreadable\n", name); \
                missing = 1; \
            } \
        } while (0)
        #define OPT(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            wt_load(&m->layers[l].field, gf, name); \
        } while (0)

        L(attn_norm,     "blk.%d.attn_norm.weight");
        L(ffn_norm,      "blk.%d.ffn_norm.weight");
        L(post_attn_norm,"blk.%d.post_attention_norm.weight");   /* sandwich, optional */
        L(post_ffn_norm, "blk.%d.post_ffw_norm.weight");          /* sandwich, optional */
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        m->layers[l].is_recr = (gguf_find_tensor(gf, name) < 0);

        if (m->layers[l].is_recr) {
            W(in_proj,  "blk.%d.shortconv.in_proj.weight");
            W(out_proj, "blk.%d.shortconv.out_proj.weight");
            snprintf(name, sizeof(name), "blk.%d.shortconv.conv.weight", l);
            m->layers[l].conv_w = lfm3_f32(gf, name);
            if (!m->layers[l].conv_w) {
                fprintf(stderr, "lfm3: required tensor '%s' missing or unreadable\n", name);
                missing = 1;
            }
            m->layers[l].recr_slot = m->n_recr++;
        } else {
            /* The learned frequency table belongs to the full-attention layers; a
             * file that carries none runs them on the plain base, as the reference
             * does. The routing itself came from the pattern array, not from here. */
            if (!m->layers[l].is_swa) {
                snprintf(name, sizeof(name), "blk.%d.rope_freqs.weight", l);
                m->layers[l].rope_freqs = lfm3_f32(gf, name);   /* optional */
            }
            W(wq,        "blk.%d.attn_q.weight");
            W(wk,        "blk.%d.attn_k.weight");
            W(wv,        "blk.%d.attn_v.weight");
            W(wo,        "blk.%d.attn_output.weight");
            W(gate_head, "blk.%d.attn_gate.weight");
            L(q_norm,    "blk.%d.attn_q_norm.weight");
            L(k_norm,    "blk.%d.attn_k_norm.weight");
            m->layers[l].kv_slot = m->n_attn++;
            m->layers[l].n_head    = m->layers[l].wq.rows / m->head_dim;
            m->layers[l].n_kv_head= m->layers[l].wk.rows / m->head_dim;
            if (m->layers[l].n_head <= 0 || m->layers[l].n_kv_head <= 0 ||
                m->layers[l].gate_head.rows != m->layers[l].n_head) {
                fprintf(stderr, "lfm3: layer %d geometry unusable "
                                "(heads=%d kv_heads=%d gate_rows=%d)\n", l,
                        m->layers[l].n_head, m->layers[l].n_kv_head,
                        m->layers[l].gate_head.rows);
                missing = 1;
            }
        }
        W(wgate, "blk.%d.ffn_gate.weight");
        W(wup,   "blk.%d.ffn_up.weight");
        W(wdown, "blk.%d.ffn_down.weight");
        m->layers[l].ffn = m->layers[l].wgate.rows;
        #undef L
        #undef W
        #undef OPT
    }

    if (m->n_attn == 0) {
        fprintf(stderr, "lfm3: no attention layers; this family is a hybrid and "
                        "an all-convolution file means the loader misread it\n");
        missing = 1;
    }
    if (m->n_recr > 0) {
        m->conv_state = (float*)calloc((size_t)m->n_recr * m->embed * (m->l_cache - 1),
                                       sizeof(float));
        if (!m->conv_state) { lfm3_free(m); return NULL; }
    }

    int max_kv = 0;
    for (int l = 0; l < nl; l++)
        if (!m->layers[l].is_recr && m->layers[l].n_kv_head * m->head_dim > max_kv)
            max_kv = m->layers[l].n_kv_head * m->head_dim;

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
    free(m->conv_state);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm); free(m->layers[l].ffn_norm);
        free(m->layers[l].post_attn_norm); free(m->layers[l].post_ffn_norm);
        free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].wq.f32); free(m->layers[l].wk.f32); free(m->layers[l].wv.f32);
        free(m->layers[l].wo.f32); free(m->layers[l].gate_head.f32);
        free(m->layers[l].wgate.f32); free(m->layers[l].wup.f32); free(m->layers[l].wdown.f32);
        free(m->layers[l].in_proj.f32); free(m->layers[l].out_proj.f32);
        free(m->layers[l].conv_w); free(m->layers[l].rope_freqs);
    }
    free(m);
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

/* NEOX rope with optional per-lane frequency factors — the same convention the
 * gemma4 family runs, rotated over n_rot lanes of the head. */
static void rope_lanes(float *x, int pos, int n_rot, float base, const float *factors) {
    int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, 2.0f * (float)i / (float)n_rot);
        if (factors) freq *= factors[i];
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
static void lfm3_conv_layer(lfm3_model *m, int l, const float *xn, float *y_out,
                            int n) {
    int E = m->embed, LC = m->l_cache;
    int w_rows = m->layers[l].in_proj.rows;   /* 3*E */
    float *bcx = (float*)malloc((size_t)n * w_rows * sizeof(float));
    float *y   = (float*)malloc((size_t)n * E * sizeof(float));
    if (!bcx || !y) { free(bcx); free(y); return; }

    qmm(bcx, &m->layers[l].in_proj, xn, n);

    float *state = m->conv_state + (size_t)m->layers[l].recr_slot * E * (LC - 1);
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
}

static int lfm3_forward(void *model, kv_cache *kv, const int *tokens, int n,
                        int pos0, float *logits) {
    lfm3_model *m = (lfm3_model*)model;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->n_attn, 0);
    if (rc != NT_OK) return rc;

    int E = m->embed, HD = m->head_dim, ROT = m->n_rot;
    float eps = m->rms_eps;

    /* A sequence starting over starts from silence — same reason Mamba resets
     * here: a second run under -r must not answer with the first run's window. */
    if (pos0 == 0 && m->n_recr > 0)
        memset(m->conv_state, 0,
               (size_t)m->n_recr * E * (m->l_cache - 1) * sizeof(float));

    float *x  = (float*)calloc((size_t)n * E, sizeof(float));
    float *xn = (float*)calloc((size_t)n * E, sizeof(float));
    if (!x || !xn) { free(x); free(xn); return NT_E_MEMORY; }

    double pft = pf_mark();
    for (int j = 0; j < n; j++) {
        float *xj = x + (size_t)j * E;
        if (m->tok_emb.f32) memcpy(xj, m->tok_emb.f32 + (long)tokens[j] * E, E * sizeof(float));
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
        free(x); free(xn); free(q_all); free(k_new); free(v_new); free(attn_out);
        free(proj_out); free(gate_buf); free(g_buf); free(u_buf);
        return NT_E_MEMORY;
    }

    for (int l = 0; l < m->n_layers; l++) {
        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (size_t)j * E, x + (size_t)j * E, m->layers[l].attn_norm, E, eps);
        pf_add(PF_NORM, pft);

        if (m->layers[l].is_recr) {
            pft = pf_mark();
            lfm3_conv_layer(m, l, xn, proj_out, n);
            pf_add(PF_ATTN, pft);
        } else {
            int H = m->layers[l].n_head, KVH = m->layers[l].n_kv_head;
            int Q_DIM = H * HD, KVD = KVH * HD;
            int gqa = H / KVH;
            float scale = 1.0f / sqrtf((float)HD);
            float base = m->layers[l].is_swa ? m->rope_base_swa : m->rope_base;
            const float *factors = m->layers[l].is_swa ? NULL : m->layers[l].rope_freqs;
            long cache_base = (long)m->layers[l].kv_slot * kv->max_seq * kv->kv_dim;

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
                memcpy(kv->k + cache_base + (long)pos * kv->kv_dim, kj, KVD * sizeof(float));
                memcpy(kv->v + cache_base + (long)pos * kv->kv_dim, v_new + (size_t)j * KVD,
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
                    if (!scores) { free(x); free(xn); free(q_all); free(k_new); free(v_new);
                        free(attn_out); free(proj_out); free(gate_buf); free(g_buf); free(u_buf);
                        return NT_E_MEMORY; }
                    for (int t = 0; t < span; t++) {
                        const float *kt = kv->k + cache_base + (long)(first + t) * kv->kv_dim + kvh * HD;
                        scores[t] = dot_f32(q, kt, HD) * scale;
                    }
                    softmax(scores, span);
                    float *out_h = attn_out + (size_t)j * Q_DIM + h * HD;
                    for (int t = 0; t < span; t++) {
                        const float *vt = kv->v + cache_base + (long)(first + t) * kv->kv_dim + kvh * HD;
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

        /* Sandwich norm on the block output, then the residual. Either both
         * post norms exist or neither — the loader checks nothing because the
         * file cannot lie about a tensor it did not write. */
        pft = pf_mark();
        if (m->layers[l].post_attn_norm)
            for (int j = 0; j < n; j++)
                rmsnorm(proj_out + (size_t)j * E, proj_out + (size_t)j * E,
                        m->layers[l].post_attn_norm, E, eps);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += proj_out[i];
        pf_add(PF_RESID, pft);

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
    }

    if (logits) {
        pft = pf_mark();
        rmsnorm(xn, x + (size_t)(n - 1) * E, m->out_norm, E, eps);
        qmv(logits, m->has_output_weight ? &m->out_weight : &m->tok_emb, xn);
        pf_add(PF_HEAD, pft);
    }

    free(x); free(xn); free(q_all); free(k_new); free(v_new); free(attn_out);
    free(proj_out); free(gate_buf); free(g_buf); free(u_buf);
    return NT_OK;
}

static const char *const lfm3_names[] = { "lfm3", NULL };

const nt_arch nt_arch_lfm3 = {
    .names  = lfm3_names,
    .load   = lfm3_load,
    .free   = lfm3_free,
    .forward= lfm3_forward,
};
