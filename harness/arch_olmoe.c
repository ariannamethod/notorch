/* arch_olmoe.c — OLMoE, and the first mixture this tree has run.
 *
 * Everything before the feed-forward is llama with two additions: Q and K are RMS-normalised
 * after their projections and before the heads are split out, with a weight vector as long as
 * the whole projection rather than one per head. Read that from the file rather than from a
 * family resemblance — gemma4 normalises per head, this one does not, and the two are one
 * reshape apart.
 *
 * The feed-forward is the new part and it changes how weights are read. A dense model streams
 * every byte of every layer for every token; this one has sixty-four experts per layer and
 * uses eight, so it reads an eighth of the feed-forward — but reads it *gathered*, eight
 * slices chosen per token out of a 75 MB region, instead of one sweep. Everything this
 * library has been tuned on assumed the sweep.
 *
 * Routing, from the reference and not from the usual shape of these things: softmax over all
 * sixty-four, top eight by that probability, and the weights are those probabilities taken as
 * they are. No renormalisation over the chosen eight (llama.cpp passes norm_w = false here)
 * and no scale (the file carries no expert_weights_scale). Both are common in other mixtures
 * and wrong for this one.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OLMOE_MAX_USED 32

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    int n_expert, n_expert_used;
    float rope_base, rms_eps;

    gguf_file *gf;
    int emb_ti;

    wt tok_emb;
    float *out_norm;
    wt out_weight;

    struct {
        float *attn_norm;
        wt wq, wk, wv, wo;
        float *q_norm, *k_norm;       /* over the whole projection, not per head */
        float *ffn_norm;
        wt gate_inp;                  /* [n_expert, embed] — the router */
        wt gate_exps, up_exps, down_exps;   /* stacked: n_expert slices each */
    } layers[];
} olmoe_model;

static float *load_f32(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    return ti < 0 ? NULL : gguf_dequant(gf, ti);
}

static void *olmoe_load(gguf_file *gf, nt_dims *dims) {
    int nl = gf->n_layers;
    olmoe_model *m = (olmoe_model*)calloc(1, sizeof(olmoe_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;
    m->gf = gf;

    /* Expert counts have no home in gguf_file's convenience fields, and a mixture without
     * them is not a mixture. Exact keys, because this family's names are not suffixes of
     * anything else. */
    const gguf_kv *kv = gguf_get_kv(gf, "olmoe.expert_count");
    m->n_expert = kv ? (int)kv->val.u32 : 0;
    kv = gguf_get_kv(gf, "olmoe.expert_used_count");
    m->n_expert_used = kv ? (int)kv->val.u32 : 0;
    if (m->n_expert <= 0 || m->n_expert_used <= 0 || m->n_expert_used > OLMOE_MAX_USED ||
        m->n_expert_used > m->n_expert) {
        fprintf(stderr, "olmoe: expert counts missing or unusable (%d of %d)\n",
                m->n_expert_used, m->n_expert);
        free(m);
        return NULL;
    }

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) {
        m->q_dim = (int)gf->tensors[ti].shape[1];
        m->head_dim = m->q_dim / m->n_heads;
    } else {
        m->q_dim = m->embed;
        m->head_dim = m->embed / m->n_heads;
    }
    m->kv_dim = m->n_kv_heads * m->head_dim;

    m->emb_ti = gguf_find_tensor(gf, "token_embd.weight");
    if (!wt_load(&m->tok_emb, gf, "token_embd.weight") || m->emb_ti < 0) {
        fprintf(stderr, "olmoe: no token_embd\n");
        free(m);
        return NULL;
    }
    m->vocab = m->tok_emb.rows;
    m->out_norm = load_f32(gf, "output_norm.weight");
    /* Not tied: this family ships a separate head, and in this checkpoint at a different
     * quantisation from the embedding table. */
    if (!wt_load(&m->out_weight, gf, "output.weight")) {
        fprintf(stderr, "olmoe: no output.weight\n");
        free(m);
        return NULL;
    }

    int ok = m->out_norm != NULL;
    for (int l = 0; l < nl && ok; l++) {
        char nm[128];
        #define T(dst, fmt) (snprintf(nm, sizeof(nm), fmt, l), wt_load(&m->layers[l].dst, gf, nm))
        #define F(dst, fmt) (snprintf(nm, sizeof(nm), fmt, l), m->layers[l].dst = load_f32(gf, nm), \
                             m->layers[l].dst != NULL)
        ok = F(attn_norm, "blk.%d.attn_norm.weight")
          && T(wq, "blk.%d.attn_q.weight")   && T(wk, "blk.%d.attn_k.weight")
          && T(wv, "blk.%d.attn_v.weight")   && T(wo, "blk.%d.attn_output.weight")
          && F(q_norm, "blk.%d.attn_q_norm.weight")
          && F(k_norm, "blk.%d.attn_k_norm.weight")
          && F(ffn_norm, "blk.%d.ffn_norm.weight")
          && T(gate_inp, "blk.%d.ffn_gate_inp.weight")
          && T(gate_exps, "blk.%d.ffn_gate_exps.weight")
          && T(up_exps, "blk.%d.ffn_up_exps.weight")
          && T(down_exps, "blk.%d.ffn_down_exps.weight");
        #undef T
        #undef F
        /* The stacked tensors must divide into the experts the metadata promised, or a slice
         * would silently start mid-expert and the model would answer with somebody else's
         * arithmetic. */
        if (ok && (m->layers[l].gate_exps.rows != m->ffn * m->n_expert ||
                   m->layers[l].up_exps.rows   != m->ffn * m->n_expert ||
                   m->layers[l].down_exps.rows != m->embed * m->n_expert)) {
            fprintf(stderr, "olmoe: layer %d expert stack does not match %d experts\n",
                    l, m->n_expert);
            ok = 0;
        }
    }
    if (!ok) {
        fprintf(stderr, "olmoe: missing or mismatched weights\n");
        free(m);
        return NULL;
    }

    fprintf(stderr, "olmoe: E=%d H=%d KV=%d HD=%d FFN=%d V=%d L=%d experts=%d/%d\n",
            m->embed, m->n_heads, m->n_kv_heads, m->head_dim, m->ffn, m->vocab,
            m->n_layers, m->n_expert_used, m->n_expert);

    dims->n_layers = m->n_layers;
    dims->kv_dim = m->kv_dim;
    dims->vocab = m->vocab;
    return m;
}

static void olmoe_free(void *model) {
    olmoe_model *m = (olmoe_model*)model;
    if (!m) return;
    free(m->tok_emb.f32); free(m->out_norm); free(m->out_weight.f32);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm);
        free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].ffn_norm);
        free(m->layers[l].wq.f32); free(m->layers[l].wk.f32);
        free(m->layers[l].wv.f32); free(m->layers[l].wo.f32);
        free(m->layers[l].gate_inp.f32);
        free(m->layers[l].gate_exps.f32); free(m->layers[l].up_exps.f32);
        free(m->layers[l].down_exps.f32);
    }
    free(m);
}

/* Top-k by value, k small and n sixty-four, so k passes of argmax beat sorting and beat
 * being clever. Returns the indices; the caller reads the weights out of the untouched
 * probability vector. */
static void top_k(const float *p, int n, int k, int *idx) {
    char taken[1024];
    memset(taken, 0, (size_t)n);
    for (int i = 0; i < k; i++) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < n; e++)
            if (!taken[e] && p[e] > bv) { bv = p[e]; best = e; }
        idx[i] = best < 0 ? 0 : best;
        taken[idx[i]] = 1;
    }
}

static void olmoe_forward(void *model, kv_cache *kv, const int *tokens, int n,
                          int pos0, float *logits) {
    olmoe_model *m = (olmoe_model*)model;
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    int NE = m->n_expert, NU = m->n_expert_used;
    float eps = m->rms_eps;
    int gqa = H / KV;

    double pft = pf_mark();
    float *x = (float*)calloc((size_t)n * E, sizeof(float));
    for (int j = 0; j < n; j++) {
        float *xj = x + (long)j * E;
        if (m->tok_emb.f32) memcpy(xj, m->tok_emb.f32 + (long)tokens[j] * E, E * sizeof(float));
        else if (gguf_dequant_row(m->gf, m->emb_ti, (uint64_t)tokens[j], xj) != 0)
            memset(xj, 0, E * sizeof(float));
    }
    pf_add(PF_EMBED, pft);

    float *xn = (float*)calloc((size_t)n * E, sizeof(float));
    float *q_all = (float*)calloc((size_t)n * Q_DIM, sizeof(float));
    float *k_new = (float*)calloc((size_t)n * KVD, sizeof(float));
    float *v_new = (float*)calloc((size_t)n * KVD, sizeof(float));
    float *attn_out = (float*)calloc((size_t)n * Q_DIM, sizeof(float));
    float *ffn_out = (float*)calloc((size_t)n * E, sizeof(float));
    float *router = (float*)calloc((size_t)n * NE, sizeof(float));
    float *eg = (float*)calloc((size_t)FFN, sizeof(float));
    float *eu = (float*)calloc((size_t)FFN, sizeof(float));
    float *eo = (float*)calloc((size_t)E, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (long)j * E, x + (long)j * E, m->layers[l].attn_norm, E, eps);
        pf_add(PF_NORM, pft);

        pft = pf_mark();
        qmm(q_all, &m->layers[l].wq, xn, n);
        qmm(k_new, &m->layers[l].wk, xn, n);
        qmm(v_new, &m->layers[l].wv, xn, n);
        pf_add(PF_QKV, pft);

        /* The projection is normalised whole, then split into heads. Doing it after the
         * split would divide by a different scale per head and is a different model. */
        pft = pf_mark();
        for (int j = 0; j < n; j++) {
            rmsnorm(q_all + (long)j * Q_DIM, q_all + (long)j * Q_DIM,
                    m->layers[l].q_norm, Q_DIM, eps);
            rmsnorm(k_new + (long)j * KVD, k_new + (long)j * KVD,
                    m->layers[l].k_norm, KVD, eps);
        }
        pf_add(PF_NORM, pft);

        pft = pf_mark();
        long base = (long)l * kv->max_seq * KVD;
        for (int j = 0; j < n; j++) {
            int pos = pos0 + j;
            float *qj = q_all + (long)j * Q_DIM, *kj = k_new + (long)j * KVD;
            for (int h = 0; h < H; h++) rope(qj + h*HD, pos, HD, m->rope_base, 1);
            for (int h = 0; h < KV; h++) rope(kj + h*HD, pos, HD, m->rope_base, 1);
            memcpy(kv->k + base + (long)pos * KVD, kj, KVD * sizeof(float));
            memcpy(kv->v + base + (long)pos * KVD, v_new + (long)j * KVD, KVD * sizeof(float));
        }
        pf_add(PF_ROPE, pft);

        pft = pf_mark();
        float scale = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, (size_t)n * Q_DIM * sizeof(float));
        for (int j = 0; j < n; j++) {
            int pos = pos0 + j;
            for (int h = 0; h < H; h++) {
                int kv_h = h / gqa;
                float *q = q_all + (long)j * Q_DIM + h * HD;
                float *scores = (float*)calloc(pos + 1, sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    const float *kt = kv->k + base + (long)t * KVD + kv_h * HD;
                    scores[t] = dot_f32(q, kt, HD) * scale;
                }
                softmax(scores, pos + 1);
                float *out_h = attn_out + (long)j * Q_DIM + h * HD;
                for (int t = 0; t <= pos; t++) {
                    const float *vt = kv->v + base + (long)t * KVD + kv_h * HD;
                    axpy_f32(out_h, scores[t], vt, HD);
                }
                free(scores);
            }
        }
        pf_add(PF_ATTN, pft);

        pft = pf_mark();
        qmm(ffn_out, &m->layers[l].wo, attn_out, n);
        pf_add(PF_PROJ, pft);
        pft = pf_mark();
        for (long i = 0; i < (long)n * E; i++) x[i] += ffn_out[i];
        pf_add(PF_RESID, pft);

        pft = pf_mark();
        for (int j = 0; j < n; j++)
            rmsnorm(xn + (long)j * E, x + (long)j * E, m->layers[l].ffn_norm, E, eps);
        pf_add(PF_NORM, pft);

        /* Router first, for every row at once: it is a thin matrix and the batched path
         * costs nothing here. The experts cannot follow, because each row picks its own. */
        pft = pf_mark();
        qmm(router, &m->layers[l].gate_inp, xn, n);
        pf_add(PF_QKV, pft);

        memset(ffn_out, 0, (size_t)n * E * sizeof(float));
        for (int j = 0; j < n; j++) {
            float *probs = router + (long)j * NE;
            const float *xj = xn + (long)j * E;
            float *dst = ffn_out + (long)j * E;
            int idx[OLMOE_MAX_USED];

            pft = pf_mark();
            softmax(probs, NE);
            top_k(probs, NE, NU, idx);
            pf_add(PF_SILU, pft);

            for (int e = 0; e < NU; e++) {
                wt ge, ue, de;
                if (!wt_expert(&ge, &m->layers[l].gate_exps, idx[e], FFN) ||
                    !wt_expert(&ue, &m->layers[l].up_exps,   idx[e], FFN) ||
                    !wt_expert(&de, &m->layers[l].down_exps, idx[e], E)) {
                    static int warned = 0;
                    if (!warned) { fprintf(stderr, "olmoe: expert slice failed\n"); warned = 1; }
                    continue;
                }
                pft = pf_mark();
                qmv(eg, &ge, xj);
                qmv(eu, &ue, xj);
                pf_add(PF_FFN, pft);
                pft = pf_mark();
                for (int i = 0; i < FFN; i++) {
                    float g = eg[i];
                    eg[i] = (g / (1.0f + expf(-g))) * eu[i];
                }
                pf_add(PF_SILU, pft);
                pft = pf_mark();
                qmv(eo, &de, eg);
                pf_add(PF_FFN, pft);
                /* The weight is the softmax probability as it stands: this family neither
                 * renormalises over the chosen eight nor scales them. */
                axpy_f32(dst, probs[idx[e]], eo, E);
            }
        }
        pft = pf_mark();
        for (long i = 0; i < (long)n * E; i++) x[i] += ffn_out[i];
        pf_add(PF_RESID, pft);
    }

    if (logits) {
        pft = pf_mark();
        rmsnorm(xn, x + (long)(n - 1) * E, m->out_norm, E, eps);
        qmv(logits, &m->out_weight, xn);
        pf_add(PF_HEAD, pft);
    }

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_out); free(router); free(eg); free(eu); free(eo);
}

static const char *const olmoe_names[] = { "olmoe", NULL };

const nt_arch nt_arch_olmoe = {
    .names = olmoe_names,
    .load = olmoe_load,
    .free = olmoe_free,
    .forward = olmoe_forward,
};
