/* arch_resonance.c — Resonance, the Method's own architecture.
 *
 * A block runs two attentions over one set of values. The first is the usual
 * content attention, Q·Kᵀ/√D. The second — RRPRAM — does not look at keys at
 * all: it maps the normalised input through a learned low-rank basis,
 * temp[r] = Σ_e xn[e]·wr_a[h,e,r], and scores position j by that against a
 * second learned basis, Σ_r temp[r]·wr_b[h,r,j]. One is addressed by content,
 * the other by position, and a per-head sigmoid decides how much of each:
 * g·content + (1−g)·rrpram.
 *
 * That second basis carries the context length in its shape — wr_b is
 * [H, R, T] — so the model has no basis at all past T positions. T is read off
 * the file and announced; it is not a soft limit.
 *
 * Ported from ~/arianna-shared/arianna.c/tools/resonance_forward.h. Equality
 * with that forward is the goal and not an assumption: nothing in this file
 * establishes it, and the gate that does is separate from it.
 *
 * Tensor names are the training graph's own (transformer.h.N.attn.wq.weight),
 * not the llama convention, which is what an architecture table is for. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int vocab, embed, heads, head_dim, blocks, ffn, ctx, rank;

    gguf_file *gf;
    float *tok_emb;          /* [vocab, embed] — dense, one row per token */
    float *norm_f;
    wt out_head;

    struct {
        float *norm1, *norm2;
        float *wr_a;         /* [H, E, R] */
        float *wr_b;         /* [H, R, T] */
        float *gate;         /* [H] */
        wt wq, wk, wv, wo;
        wt mlp_gate, mlp_up, mlp_down;
    } b[64];
} resonance_model;

/* eps and rope base are constants of this architecture rather than metadata:
 * the training graph hard-codes both, and the file carries neither. */
#define RES_EPS       1e-5f
#define RES_ROPE_BASE 10000.0f

static void rmsnorm_p(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + RES_EPS);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

/* Pairs (2i, 2i+1) rotate together — the older of the two conventions, and the
 * one the training graph applies. */
static void rope_even_odd(float *q, float *k, int pos, int dim) {
    int n_pairs = dim / 2;
    for (int i = 0; i < n_pairs; i++) {
        float freq = 1.0f / powf(RES_ROPE_BASE, (float)(2 * i) / (float)dim);
        float val = pos * freq;
        float cs = cosf(val), sn = sinf(val);
        float qe = q[2*i], qo = q[2*i + 1];
        float ke = k[2*i], ko = k[2*i + 1];
        q[2*i]     = qe * cs - qo * sn;
        q[2*i + 1] = qe * sn + qo * cs;
        k[2*i]     = ke * cs - ko * sn;
        k[2*i + 1] = ke * sn + ko * cs;
    }
}

static float siluf(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }
static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* This file writes its tensor shapes outer dimension first, which is the
 * reverse of the ggml convention wt_load reads: mlp.w_gate.weight is
 * ne=[2048,768] where a llama-family file would write ne=[768,2048], and
 * wr_a is ne=[12,768,48] which only reads as [H,E,R] outer-first. Square
 * matrices hide it — wq/wk/wv/wo come out identical either way — and every
 * other one comes out transposed, which is fluent, wrong text.
 *
 * So the shape comes from the architecture here and not from the file. The
 * dtype probe is still the file's answer: ask the two entry points whether
 * they take this tensor rather than assuming F16 forever. */
static int packed(wt *w, gguf_file *gf, const char *name, int rows, int cols) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) { fprintf(stderr, "resonance: tensor '%s' missing\n", name); return 0; }
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->n_elements != (uint64_t)rows * (uint64_t)cols) {
        fprintf(stderr, "resonance: '%s' has %llu elements, arch says %dx%d\n",
                name, (unsigned long long)t->n_elements, rows, cols);
        return 0;
    }
    w->rows = rows;
    w->cols = cols;
    w->dtype = (int)t->dtype;
    w->q = gf->data + t->offset;
    w->f32 = NULL;
    float *probe = (float*)calloc(cols, sizeof(float));
    float out = 0.0f;
    if (probe) {
        w->use_i8 = (nt_qmatvec_i8(&out, w->q, w->dtype, probe, 1, cols) == 0);
        if (!w->use_i8 && nt_qmatvec(&out, w->q, w->dtype, probe, 1, cols) != 0) {
            w->q = NULL;
            w->f32 = gguf_dequant(gf, ti);
        }
        free(probe);
    }
    return (w->q || w->f32) ? 1 : 0;
}

static float *dense(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) { fprintf(stderr, "resonance: tensor '%s' missing\n", name); return NULL; }
    float *p = gguf_dequant(gf, ti);
    if (!p) fprintf(stderr, "resonance: dequant '%s' failed\n", name);
    return p;
}

static void *resonance_load(gguf_file *gf, nt_dims *dims) {
    resonance_model *m = (resonance_model*)calloc(1, sizeof(resonance_model));
    if (!m) return NULL;

    int ok = 1;
#define KV(key, tgt) do { \
        const gguf_kv *kv = gguf_get_kv(gf, key); \
        if (!kv) { fprintf(stderr, "resonance: missing kv '%s'\n", key); ok = 0; } \
        else tgt = (int)kv->val.u32; \
    } while (0)
    KV("resonance.vocab_size",           m->vocab);
    KV("resonance.embedding_length",     m->embed);
    KV("resonance.attention.head_count", m->heads);
    KV("resonance.attention.head_dim",   m->head_dim);
    KV("resonance.block_count",          m->blocks);
    KV("resonance.feed_forward_length",  m->ffn);
    KV("resonance.context_length",       m->ctx);
    KV("resonance.rrpram_rank",          m->rank);
#undef KV
    if (!ok) { free(m); return NULL; }

    /* H*D == E is not a convention here but an assumption the forward rests on:
     * the KV rows are E wide and the blend runs over E, so a head layout that
     * does not fill E would read past its own head. */
    if (m->heads * m->head_dim != m->embed || m->blocks < 1 ||
        m->blocks > (int)(sizeof(m->b) / sizeof(m->b[0]))) {
        fprintf(stderr, "resonance: arch refused — V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
                m->vocab, m->embed, m->heads, m->head_dim, m->blocks, m->ffn, m->ctx, m->rank);
        free(m);
        return NULL;
    }
    fprintf(stderr, "resonance: E=%d H=%d D=%d FFN=%d V=%d B=%d ctx=%d rrpram_rank=%d\n",
            m->embed, m->heads, m->head_dim, m->ffn, m->vocab, m->blocks, m->ctx, m->rank);

    const int E = m->embed, FFN = m->ffn;
    m->gf = gf;
    m->tok_emb = dense(gf, "tok_emb");
    m->norm_f  = dense(gf, "norm_f.weight");
    if (!packed(&m->out_head, gf, "out_head.weight", m->vocab, E)) ok = 0;

    char nm[128];
    for (int l = 0; l < m->blocks && ok; l++) {
        #define L(field, suffix) do { \
            snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, l); \
            m->b[l].field = dense(gf, nm); \
            if (!m->b[l].field) ok = 0; \
        } while (0)
        #define W(field, suffix, r, c) do { \
            snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, l); \
            if (!packed(&m->b[l].field, gf, nm, (r), (c))) ok = 0; \
        } while (0)
        L(wr_a,     "attn.wr_a");
        L(wr_b,     "attn.wr_b");
        L(gate,     "attn.gate");
        L(norm1,    "norm1.weight");
        W(wq,       "attn.wq.weight",     E,   E);
        W(wk,       "attn.wk.weight",     E,   E);
        W(wv,       "attn.wv.weight",     E,   E);
        W(wo,       "attn.wo.weight",     E,   E);
        L(norm2,    "norm2.weight");
        W(mlp_gate, "mlp.w_gate.weight",  FFN, E);
        W(mlp_up,   "mlp.w_up.weight",    FFN, E);
        W(mlp_down, "mlp.w_down.weight",  E,   FFN);
        #undef L
        #undef W
    }

    if (!ok || !m->tok_emb || !m->norm_f) {
        fprintf(stderr, "resonance: missing critical weights\n");
        free(m);
        return NULL;
    }

    dims->n_layers = m->blocks;
    dims->kv_dim   = m->embed;      /* K and V are stored whole, no GQA here */
    dims->vocab    = m->vocab;
    return m;
}

static void resonance_free(void *model) {
    resonance_model *m = (resonance_model*)model;
    if (!m) return;
    free(m->tok_emb); free(m->norm_f); free(m->out_head.f32);
    for (int l = 0; l < m->blocks; l++) {
        free(m->b[l].norm1); free(m->b[l].norm2);
        free(m->b[l].wr_a); free(m->b[l].wr_b); free(m->b[l].gate);
        free(m->b[l].wq.f32); free(m->b[l].wk.f32);
        free(m->b[l].wv.f32); free(m->b[l].wo.f32);
        free(m->b[l].mlp_gate.f32); free(m->b[l].mlp_up.f32); free(m->b[l].mlp_down.f32);
    }
    free(m);
}

/* One position. Prefill calls this per row rather than in a batch: the two
 * attentions read the cache row by row and the source forward is per-token, so
 * a batched form would be a different summation order for no weight traffic
 * saved — every matrix here is already read once per position. */
static void resonance_step(resonance_model *m, kv_cache *kv, int tok, int pos,
                           float *logits, float *scratch) {
    const int E = m->embed, H = m->heads, D = m->head_dim;
    const int FFN = m->ffn, R = m->rank, T = m->ctx;
    const float sc = 1.0f / sqrtf((float)D);

    float *x   = scratch;                 /* E       */
    float *xn  = x + E;                   /* E       */
    float *qa  = xn + E;                  /* E       */
    float *ka  = qa + E;                  /* E       */
    float *va  = ka + E;                  /* E       */
    float *cot = va + E;                  /* E       */
    float *rot = cot + E;                 /* E       */
    float *bl_ = rot + E;                 /* E       */
    float *ao  = bl_ + E;                 /* E       */
    float *mg  = ao + E;                  /* FFN     */
    float *mu  = mg + FFN;                /* FFN     */
    float *mo  = mu + FFN;                /* E       */
    float *att = mo + E;                  /* max_seq */
    float *tmp = att + kv->max_seq;       /* R       */

    double pft = pf_mark();
    memcpy(x, m->tok_emb + (long)tok * E, E * sizeof(float));
    pf_add(PF_EMBED, pft);

    for (int l = 0; l < m->blocks; l++) {
        pft = pf_mark();
        rmsnorm_p(xn, x, m->b[l].norm1, E);
        pf_add(PF_NORM, pft);

        pft = pf_mark();
        qmv(qa, &m->b[l].wq, xn);
        qmv(ka, &m->b[l].wk, xn);
        qmv(va, &m->b[l].wv, xn);
        pf_add(PF_QKV, pft);

        pft = pf_mark();
        for (int h = 0; h < H; h++)
            rope_even_odd(qa + h * D, ka + h * D, pos, D);
        long base = (long)l * kv->max_seq * E;
        memcpy(kv->k + base + (long)pos * E, ka, E * sizeof(float));
        memcpy(kv->v + base + (long)pos * E, va, E * sizeof(float));
        pf_add(PF_ROPE, pft);

        pft = pf_mark();
        /* content attention */
        for (int h = 0; h < H; h++) {
            const float *q_h = qa + h * D;
            for (int j = 0; j <= pos; j++) {
                const float *kj = kv->k + base + (long)j * E + h * D;
                att[j] = dot_f32(q_h, kj, D) * sc;
            }
            softmax(att, pos + 1);
            float *oh = cot + h * D;
            memset(oh, 0, D * sizeof(float));
            for (int j = 0; j <= pos; j++)
                axpy_f32(oh, att[j], kv->v + base + (long)j * E + h * D, D);
        }

        /* RRPRAM: scored by position through a learned basis, over the same V */
        for (int h = 0; h < H; h++) {
            const float *wr_a_h = m->b[l].wr_a + (long)h * E * R;
            const float *wr_b_h = m->b[l].wr_b + (long)h * R * T;
            for (int r = 0; r < R; r++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += xn[e] * wr_a_h[e * R + r];
                tmp[r] = s;
            }
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int r = 0; r < R; r++) s += tmp[r] * wr_b_h[r * T + j];
                att[j] = s * sc;
            }
            softmax(att, pos + 1);
            float *oh = rot + h * D;
            memset(oh, 0, D * sizeof(float));
            for (int j = 0; j <= pos; j++)
                axpy_f32(oh, att[j], kv->v + base + (long)j * E + h * D, D);
        }

        for (int h = 0; h < H; h++) {
            float g = sigmoidf_(m->b[l].gate[h]);
            for (int d = 0; d < D; d++)
                bl_[h * D + d] = g * cot[h * D + d] + (1.0f - g) * rot[h * D + d];
        }
        pf_add(PF_ATTN, pft);

        pft = pf_mark();
        qmv(ao, &m->b[l].wo, bl_);
        pf_add(PF_PROJ, pft);
        pft = pf_mark();
        for (int e = 0; e < E; e++) x[e] += ao[e];
        pf_add(PF_RESID, pft);

        pft = pf_mark();
        rmsnorm_p(xn, x, m->b[l].norm2, E);
        pf_add(PF_NORM, pft);
        pft = pf_mark();
        qmv(mg, &m->b[l].mlp_gate, xn);
        qmv(mu, &m->b[l].mlp_up, xn);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (int i = 0; i < FFN; i++) mg[i] = siluf(mg[i]) * mu[i];
        pf_add(PF_SILU, pft);
        pft = pf_mark();
        qmv(mo, &m->b[l].mlp_down, mg);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (int e = 0; e < E; e++) x[e] += mo[e];
        pf_add(PF_RESID, pft);
    }

    if (logits) {
        pft = pf_mark();
        rmsnorm_p(xn, x, m->norm_f, E);
        qmv(logits, &m->out_head, xn);
        pf_add(PF_HEAD, pft);
    }
}

static int resonance_forward(void *model, kv_cache *kv, const int *tokens, int n,
                             int pos0, float *logits) {
    resonance_model *m = (resonance_model*)model;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->blocks, m->embed);
    if (rc != NT_OK) return rc;

    /* Ten E-wide buffers (x, xn, q, k, v, content, rrpram, blend, proj, mlp-out),
     * two FFN-wide, one score row per cached position, one rank row. Counting
     * these wrong does not crash: the score row lands on top of the MLP output
     * and the model answers with different tokens. */
    size_t words = (size_t)m->embed * 10 + (size_t)m->ffn * 2
                 + (size_t)kv->max_seq + (size_t)m->rank;
    float *scratch = (float*)malloc(words * sizeof(float));
    if (!scratch) return NT_E_MEMORY;
    /* The bounds check that used to live on this loop is gone: nt_check_call
     * refused the call before any of it ran, so a position past the cache can
     * no longer arrive here to be silently dropped mid-prompt. */
    for (int j = 0; j < n; j++) {
        int pos = pos0 + j;
        resonance_step(m, kv, tokens[j], pos,
                       (logits && j == n - 1) ? logits : NULL, scratch);
    }
    free(scratch);
    return NT_OK;
}

static const char *const RESONANCE_NAMES[] = { "resonance", NULL };

const nt_arch nt_arch_resonance = {
    .names = RESONANCE_NAMES,
    .load = resonance_load,
    .free = resonance_free,
    .forward = resonance_forward,
};
