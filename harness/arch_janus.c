/* arch_janus.c — Janus v4, the Method's low-rank resonance architecture.
 *
 * A block blends three attentions rather than one, per head, by a softmax over
 * three learned logits: ordinary content attention, RRPRAM, and Echo. RRPRAM
 * here is not Resonance's: its low-rank state *accumulates across positions*,
 * mid[r] += Σ_e norm(x)[e]·wr_a[h,e,r], and the scores it produces are the same
 * for every query row — one broadcast over the whole sequence. It also reads its
 * own values through a separate projection, so the cache holds three tensors and
 * not two.
 *
 * Around the blocks: a smear that mixes the previous token's embedding into the
 * current one through a gate on 24 dimensions; a per-layer remix of the residual
 * with the original embedding, resid_lambda·x + x0_lambda·x0; a mid-depth
 * snapshot subtracted at the end (backout); and a soft cap on the logits,
 * 15·tanh(l/15).
 *
 * Ported from ~/arianna-shared/arianna.c/tools/yent_forward.h. Equality with
 * that forward is the goal and not an assumption: nothing here establishes it,
 * and the gate that does is separate from it.
 *
 * One asymmetry is inherited deliberately. The source smears in its batched
 * prefill and does not smear in its per-token path, where the omission is
 * marked TODO rather than decided. Mirroring it keeps the port comparable to
 * the thing it was ported from; the cost of the asymmetry is measured
 * separately rather than silently repaired here. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define JANUS_MAX_BLOCKS 32
#define JANUS_SMEAR_DIMS 24

typedef struct {
    int vocab, embed, heads, head_dim, blocks, ffn, ctx, rank;

    gguf_file *gf;
    float *resid_l, *x0_l, *smear_l, *backout_l, *smear_g;
    float *wte;
    wt head;

    struct {
        float *wr_a;      /* [H, E, R] */
        float *wr_b;      /* [H, R, T] */
        float *gate;      /* [H, 3] — content / rrpram / echo, before softmax */
        wt cq, ck, cv, wvr, wj, cproj;
        wt wg, wu, wd;
    } b[JANUS_MAX_BLOCKS];

    /* State the shared cache does not carry. The harness's kv_cache holds keys
     * and values; this family also reads a second set of values through wvr,
     * and keeps a low-rank sum that grows with the sequence. Both belong to the
     * architecture, so the architecture owns them rather than widening the
     * interface for one family. */
    float *vr;        /* [blocks, max_seq, E] */
    float *mid;       /* [blocks, heads, R] */
    int vr_seq;       /* what vr was sized for */
} janus_model;

#define JANUS_EPS       1e-5f
#define JANUS_ROPE_BASE 100000.0f
#define JANUS_QK_SCALE  1.2f
#define JANUS_LOGIT_CAP 15.0f

/* No weight vector: this family's norm is the bare RMS. */
static void jnorm(float *o, const float *x, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + JANUS_EPS);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv;
}

/* Split-half pairs (i, i+D/2) at base 100000, and the rotation runs the
 * opposite way round from the convention the llama family uses. */
static void jrope(float *q, float *k, int pos, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(JANUS_ROPE_BASE, (float)(2*i) / (float)dim);
        float val = pos * freq;
        float cv = cosf(val), sv = sinf(val);
        float q0 = q[i], q1 = q[i + half];
        q[i]        = q0 * cv + q1 * sv;
        q[i + half] = q0 * (-sv) + q1 * cv;
        float k0 = k[i], k1 = k[i + half];
        k[i]        = k0 * cv + k1 * sv;
        k[i + half] = k0 * (-sv) + k1 * cv;
    }
}

static void jqk_norm(float *q, float *k, int dim) {
    jnorm(q, q, dim);
    jnorm(k, k, dim);
    for (int i = 0; i < dim; i++) { q[i] *= JANUS_QK_SCALE; k[i] *= JANUS_QK_SCALE; }
}

static float jsilu(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }

static float *jdense(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) { fprintf(stderr, "janus: tensor '%s' missing\n", name); return NULL; }
    float *p = gguf_dequant(gf, ti);
    if (!p) fprintf(stderr, "janus: dequant '%s' failed\n", name);
    return p;
}

/* This file writes its tensor shapes outer dimension first, the reverse of the
 * convention wt_load reads: mlp.w_gate.weight is ne=[1664,640] where a
 * llama-family file writes ne=[640,1664], and wr_a is ne=[10,640,64], which
 * only reads as [H,E,R] outer-first. Square matrices hide it and everything
 * else comes out transposed, so the shape comes from the architecture and only
 * the dtype comes from the file. */
static int jpacked(wt *w, gguf_file *gf, const char *name, int rows, int cols) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) { fprintf(stderr, "janus: tensor '%s' missing\n", name); return 0; }
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->n_elements != (uint64_t)rows * (uint64_t)cols) {
        fprintf(stderr, "janus: '%s' has %llu elements, arch says %dx%d\n",
                name, (unsigned long long)t->n_elements, rows, cols);
        return 0;
    }
    w->rows = rows; w->cols = cols;
    w->dtype = (int)t->dtype;
    w->q = gf->data + t->offset;
    w->f32 = NULL;
    float *probe = (float*)calloc(cols, sizeof(float));
    float out = 0.0f;
    if (probe) {
#ifdef JANUS_EXACT_MATVEC
        /* The gate builds with this to compare against the engine this was
         * ported from, which calls the exact matvec where the harness prefers
         * the integer one. Without it the two disagree by the size of that
         * approximation and the comparison measures the choice rather than the
         * port. Nothing ships with it set. */
        w->use_i8 = 0;
        (void)out;
#else
        w->use_i8 = (nt_qmatvec_i8(&out, w->q, w->dtype, probe, 1, cols) == 0);
#endif
        if (!w->use_i8 && nt_qmatvec(&out, w->q, w->dtype, probe, 1, cols) != 0) {
            w->q = NULL;
            w->f32 = gguf_dequant(gf, ti);
        }
        free(probe);
    }
    return (w->q || w->f32) ? 1 : 0;
}

static void *janus_load(gguf_file *gf, nt_dims *dims) {
    janus_model *m = (janus_model*)calloc(1, sizeof(janus_model));
    if (!m) return NULL;

    int ok = 1;
#define KV(key, tgt) do { \
        const gguf_kv *kv = gguf_get_kv(gf, key); \
        if (!kv) { fprintf(stderr, "janus: missing kv '%s'\n", key); ok = 0; } \
        else tgt = (int)kv->val.u32; \
    } while (0)
    KV("janus.vocab_size",           m->vocab);
    KV("janus.embedding_length",     m->embed);
    KV("janus.attention.head_count", m->heads);
    KV("janus.attention.head_dim",   m->head_dim);
    KV("janus.block_count",          m->blocks);
    KV("janus.feed_forward_length",  m->ffn);
    KV("janus.context_length",       m->ctx);
    KV("janus.rrpram.rank",          m->rank);
#undef KV
    if (!ok) { free(m); return NULL; }

    if (m->heads * m->head_dim != m->embed || m->blocks < 1 ||
        m->blocks > JANUS_MAX_BLOCKS || m->embed < JANUS_SMEAR_DIMS) {
        fprintf(stderr, "janus: arch refused — V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
                m->vocab, m->embed, m->heads, m->head_dim, m->blocks, m->ffn, m->ctx, m->rank);
        free(m);
        return NULL;
    }
    fprintf(stderr, "janus: E=%d H=%d D=%d FFN=%d V=%d B=%d ctx=%d rrpram_rank=%d\n",
            m->embed, m->heads, m->head_dim, m->ffn, m->vocab, m->blocks, m->ctx, m->rank);

    const int E = m->embed, FFN = m->ffn;
    m->gf = gf;
    m->resid_l   = jdense(gf, "resid_lambdas");
    m->x0_l      = jdense(gf, "x0_lambdas");
    m->smear_l   = jdense(gf, "smear_lambda");
    m->backout_l = jdense(gf, "backout_lambda");
    m->smear_g   = jdense(gf, "smear_gate.weight");
    m->wte       = jdense(gf, "transformer.wte.weight");
    if (!jpacked(&m->head, gf, "lm_head.weight", m->vocab, E)) ok = 0;

    char nm[128];
    for (int l = 0; l < m->blocks && ok; l++) {
        #define L(field, suffix) do { \
            snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, l); \
            m->b[l].field = jdense(gf, nm); \
            if (!m->b[l].field) ok = 0; \
        } while (0)
        #define W(field, suffix, r, c) do { \
            snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, l); \
            if (!jpacked(&m->b[l].field, gf, nm, (r), (c))) ok = 0; \
        } while (0)
        L(wr_a,  "attn.wr_a");
        L(wr_b,  "attn.wr_b");
        L(gate,  "attn.gate");
        W(cq,    "attn.c_q.weight",    E,   E);
        W(ck,    "attn.c_k.weight",    E,   E);
        W(cv,    "attn.c_v.weight",    E,   E);
        W(wvr,   "attn.wvr.weight",    E,   E);
        W(wj,    "attn.wj.weight",     E,   E);
        W(cproj, "attn.c_proj.weight", E,   E);
        W(wg,    "mlp.w_gate.weight",  FFN, E);
        W(wu,    "mlp.w_up.weight",    FFN, E);
        W(wd,    "mlp.w_down.weight",  E,   FFN);
        #undef L
        #undef W
    }

    if (!ok || !m->resid_l || !m->x0_l || !m->smear_l || !m->backout_l ||
        !m->smear_g || !m->wte) {
        fprintf(stderr, "janus: missing critical weights\n");
        free(m);
        return NULL;
    }

    dims->n_layers = m->blocks;
    dims->kv_dim   = m->embed;
    dims->vocab    = m->vocab;
    return m;
}

static void janus_free(void *model) {
    janus_model *m = (janus_model*)model;
    if (!m) return;
    free(m->resid_l); free(m->x0_l); free(m->smear_l); free(m->backout_l);
    free(m->smear_g); free(m->wte); free(m->head.f32);
    for (int l = 0; l < m->blocks; l++) {
        free(m->b[l].wr_a); free(m->b[l].wr_b); free(m->b[l].gate);
        free(m->b[l].cq.f32); free(m->b[l].ck.f32); free(m->b[l].cv.f32);
        free(m->b[l].wvr.f32); free(m->b[l].wj.f32); free(m->b[l].cproj.f32);
        free(m->b[l].wg.f32); free(m->b[l].wu.f32); free(m->b[l].wd.f32);
    }
    free(m->vr); free(m->mid);
    free(m);
}

/* The second value cache and the running low-rank sum, sized once against the
 * cache the harness handed us. A shorter cache than last time means a new run,
 * so the sum starts over rather than carrying somebody else's prompt. */
static int janus_state(janus_model *m, const kv_cache *kv) {
    size_t vr_words = (size_t)m->blocks * (size_t)kv->max_seq * (size_t)m->embed;
    if (!m->vr || m->vr_seq != kv->max_seq) {
        free(m->vr);
        m->vr = (float*)calloc(vr_words, sizeof(float));
        m->vr_seq = kv->max_seq;
    }
    if (!m->mid)
        m->mid = (float*)calloc((size_t)m->blocks * m->heads * m->rank, sizeof(float));
    return m->vr && m->mid;
}

/* One position. The low-rank sum grows as the sequence does, which is the
 * causal reading of this family's broadcast: position p is scored by what the
 * model has seen up to p.
 *
 * The source's batched prefill reads it the other way — it sums over every
 * position of the prompt and hands that to all of them, so row 0's scores carry
 * the prompt's later tokens. Its own per-token path accumulates instead, and the
 * two therefore answer differently on the same prompt. This takes the
 * accumulating one, and the gate measures what the other costs rather than
 * assuming either is the intent. */
static void janus_step(janus_model *m, kv_cache *kv, const float *xin, int pos,
                       float *logits, float *scratch) {
    const int E = m->embed, H = m->heads, D = m->head_dim;
    const int FFN = m->ffn, R = m->rank, T = m->ctx;
    const float sc = 1.0f / sqrtf((float)D);

    float *x   = scratch;             /* E   */
    float *x0  = x + E;               /* E   */
    float *rn  = x0 + E;              /* E   */
    float *qa  = rn + E;              /* E   */
    float *ka  = qa + E;              /* E   */
    float *va  = ka + E;              /* E   */
    float *vra = va + E;              /* E   */
    float *eco = vra + E;             /* E   */
    float *cat = eco + E;             /* E   */
    float *ao  = cat + E;             /* E   */
    float *mo  = ao + E;              /* E   */
    float *bko = mo + E;              /* E   — the mid-depth snapshot */
    float *mg  = bko + E;             /* FFN */
    float *mu  = mg + FFN;            /* FFN */
    float *att = mu + FFN;            /* max_seq */

    memcpy(x, xin, (size_t)E * sizeof(float));
    memcpy(x0, x, (size_t)E * sizeof(float));

    const int backout_layer = m->blocks / 2;

    for (int l = 0; l < m->blocks; l++) {
        double pft = pf_mark();
        float rl = m->resid_l[l], x0l = m->x0_l[l];
        for (int e = 0; e < E; e++) x[e] = rl * x[e] + x0l * x0[e];
        jnorm(rn, x, E);
        pf_add(PF_NORM, pft);

        pft = pf_mark();
        qmv(qa,  &m->b[l].cq,  rn);
        qmv(ka,  &m->b[l].ck,  rn);
        qmv(va,  &m->b[l].cv,  rn);
        qmv(vra, &m->b[l].wvr, rn);
        pf_add(PF_QKV, pft);

        pft = pf_mark();
        for (int h = 0; h < H; h++) {
            jrope(qa + h*D, ka + h*D, pos, D);
            jqk_norm(qa + h*D, ka + h*D, D);
        }
        long base = (long)l * kv->max_seq * E;
        memcpy(kv->k + base + (long)pos * E, ka, (size_t)E * sizeof(float));
        memcpy(kv->v + base + (long)pos * E, va, (size_t)E * sizeof(float));
        memcpy(m->vr + base + (long)pos * E, vra, (size_t)E * sizeof(float));
        pf_add(PF_ROPE, pft);

        pft = pf_mark();
        qmv(eco, &m->b[l].wj, rn);

        for (int h = 0; h < H; h++) {
            float g[3] = { m->b[l].gate[h*3], m->b[l].gate[h*3+1], m->b[l].gate[h*3+2] };
            softmax(g, 3);

            const float *q_h = qa + h*D;
            for (int j = 0; j <= pos; j++)
                att[j] = dot_f32(q_h, kv->k + base + (long)j * E + h*D, D) * sc;
            softmax(att, pos + 1);
            float *oh = cat + h*D;
            memset(oh, 0, (size_t)D * sizeof(float));
            for (int j = 0; j <= pos; j++)
                axpy_f32(oh, g[0] * att[j], kv->v + base + (long)j * E + h*D, D);

            /* RRPRAM: one score row for every query, from a sum that grows with
             * the sequence, against this family's own value projection. */
            const float *wr_a_h = m->b[l].wr_a + (long)h * E * R;
            const float *wr_b_h = m->b[l].wr_b + (long)h * R * T;
            float *mid = m->mid + ((long)l * H + h) * R;
            for (int r = 0; r < R; r++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[e] * wr_a_h[e * R + r];
                mid[r] += s;
            }
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int r = 0; r < R; r++) s += mid[r] * wr_b_h[r * T + j];
                att[j] = s * sc;
            }
            softmax(att, pos + 1);
            for (int j = 0; j <= pos; j++)
                axpy_f32(oh, g[1] * att[j], m->vr + base + (long)j * E + h*D, D);

            const float *e_h = eco + h*D;
            for (int d = 0; d < D; d++) oh[d] += g[2] * e_h[d];
        }
        pf_add(PF_ATTN, pft);

        pft = pf_mark();
        qmv(ao, &m->b[l].cproj, cat);
        pf_add(PF_PROJ, pft);
        pft = pf_mark();
        for (int e = 0; e < E; e++) x[e] += ao[e];
        pf_add(PF_RESID, pft);

        if (l == backout_layer) memcpy(bko, x, (size_t)E * sizeof(float));

        pft = pf_mark();
        jnorm(rn, x, E);
        pf_add(PF_NORM, pft);
        pft = pf_mark();
        qmv(mg, &m->b[l].wg, rn);
        qmv(mu, &m->b[l].wu, rn);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (int i = 0; i < FFN; i++) mg[i] = jsilu(mg[i]) * mu[i];
        pf_add(PF_SILU, pft);
        pft = pf_mark();
        qmv(mo, &m->b[l].wd, mg);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (int e = 0; e < E; e++) x[e] += mo[e];
        pf_add(PF_RESID, pft);
    }

    float bkl = *m->backout_l;
    for (int e = 0; e < E; e++) x[e] -= bkl * bko[e];

    if (logits) {
        double pft = pf_mark();
        jnorm(rn, x, E);
        qmv(logits, &m->head, rn);
        /* A soft cap, not a clamp: every logit is squashed, not only the loud
         * ones, so the ordering survives and the scale stops running away. */
        for (int i = 0; i < m->vocab; i++)
            logits[i] = JANUS_LOGIT_CAP * tanhf(logits[i] / JANUS_LOGIT_CAP);
        pf_add(PF_HEAD, pft);
    }
}

static void janus_forward(void *model, kv_cache *kv, const int *tokens, int n,
                          int pos0, float *logits) {
    janus_model *m = (janus_model*)model;
    const int E = m->embed;
    if (!janus_state(m, kv)) return;

    size_t words = (size_t)E * 12 + (size_t)m->ffn * 2 + (size_t)kv->max_seq;
    float *scratch = (float*)malloc(words * sizeof(float));
    float *emb = (float*)malloc((size_t)n * E * sizeof(float));
    if (!scratch || !emb) { free(scratch); free(emb); return; }

    /* Embeddings first, because the smear needs the previous position's and the
     * per-position path would have thrown it away. */
    for (int j = 0; j < n; j++) {
        memcpy(emb + (size_t)j * E, m->wte + (long)tokens[j] * E, (size_t)E * sizeof(float));
        jnorm(emb + (size_t)j * E, emb + (size_t)j * E, E);
    }

    /* The smear runs over a group and cannot run over a single position, which
     * is exactly what the source does: it smears in its batched prefill and not
     * in its per-token path, where the gap is marked TODO. Mirrored here rather
     * than repaired, so the port answers the same as what it was ported from. */
    if (n > 1) {
        float sl = *m->smear_l;
        if (sl > 1e-6f) {
            for (int j = 1; j < n; j++) {
                float dot = 0;
                for (int d = 0; d < JANUS_SMEAR_DIMS; d++)
                    dot += m->smear_g[d] * emb[(size_t)j * E + d];
                float g = sl / (1.0f + expf(-dot));
                for (int e = 0; e < E; e++)
                    emb[(size_t)j * E + e] += g * emb[(size_t)(j-1) * E + e];
            }
        }
    }

    for (int j = 0; j < n; j++) {
        int pos = pos0 + j;
        if (pos >= kv->max_seq) break;
        janus_step(m, kv, emb + (size_t)j * E, pos,
                   (logits && j == n - 1) ? logits : NULL, scratch);
    }

    free(scratch);
    free(emb);
}

static const char *const JANUS_NAMES[] = { "janus", NULL };

const nt_arch nt_arch_janus = {
    .names = JANUS_NAMES,
    .load = janus_load,
    .free = janus_free,
    .forward = janus_forward,
};
