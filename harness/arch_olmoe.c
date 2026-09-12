/* arch_olmoe.c — the mixtures: OLMoE, and Qwen3-MoE.
 *
 * Everything before the feed-forward is llama with two additions: Q and K are RMS-normalised
 * after their projections and before the heads are split out. Whether that norm is one weight
 * for the projection or one for each head differs between the two families, so it is read off
 * the tensor's own length rather than assumed — OLMoE ships [q_dim], Qwen3-MoE ships
 * [head_dim]. The two are one reshape apart and both produce fluent text.
 *
 * The feed-forward is the part that changes how weights are read. A dense model streams every
 * byte of every layer for every token; a mixture reads a slice — eight of sixty-four on OLMoE,
 * eight of a hundred and twenty-eight on Qwen3-MoE — but reads it *gathered*, chosen per token
 * out of one large region, instead of one sweep. Everything this library has been tuned on
 * assumed the sweep.
 *
 * Routing, from the reference and not from the usual shape of these things: softmax over all
 * experts, top k by that probability. Then the families part. OLMoE takes those probabilities
 * as they stand — llama.cpp's src/models/olmoe.cpp passes norm_w = false. Qwen3-MoE
 * renormalises them over the chosen k — src/models/qwen3moe.cpp:144 passes true. Neither GGUF
 * records which, so the name decides and the reference is cited beside it. Neither carries
 * expert_weights_scale, and llama.cpp skips the scale at 0.0 (src/llama-graph.cpp:1955), so
 * there is none here either.
 *
 * Qwen3-MoE is here rather than in a file of its own for the reason qwen3 is in arch_llama.c:
 * two switches read at load are not a family. What it does add is a name collision worth
 * knowing about — its feed_forward_length is the dense-equivalent 6144 while an expert is 768,
 * so the expert's own key wins when the file carries one.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OLMOE_MAX_USED 32
/* top_k marks the experts it has taken in a buffer this wide. The count comes out of the file,
 * so it is checked against this rather than trusted: a header claiming more experts than the
 * buffer holds would write past it. */
#define OLMOE_MAX_EXPERTS 1024

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    int n_expert, n_expert_used;
    /* The two places two mixtures disagree, decided at load and not per token.
     * qk_norm_per_head comes off the tensor: a [head_dim] weight is applied to
     * each head, a [q_dim] one to the projection whole, and the two are one
     * reshape apart with entirely different arithmetic. renorm_topk cannot come
     * off the file — nothing in the GGUF records it — so it comes off the name,
     * with the reference's line beside it. */
    int qk_norm_per_head, renorm_topk;
    float rope_base, rms_eps;

    gguf_file *gf;
    int emb_ti;

    wt tok_emb;
    float *out_norm;
    wt out_weight;

    struct {
        float *attn_norm;
        wt wq, wk, wv, wo;
        float *q_norm, *k_norm;       /* [q_dim] or [head_dim]; see qk_norm_per_head */
        float *ffn_norm;
        wt gate_inp;                  /* [n_expert, embed] — the router, as the file has it */
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
     * them is not a mixture. The key is prefixed with the file's own architecture, because
     * this loader now answers to more than one name and a hardcoded "olmoe." would read as
     * "no experts" on the other. */
    char key[96];
    snprintf(key, sizeof(key), "%s.expert_count", gf->arch);
    const gguf_kv *kv = gguf_get_kv(gf, key);
    m->n_expert = kv ? (int)kv->val.u32 : 0;
    snprintf(key, sizeof(key), "%s.expert_used_count", gf->arch);
    kv = gguf_get_kv(gf, key);
    m->n_expert_used = kv ? (int)kv->val.u32 : 0;

    /* feed_forward_length is the dense-equivalent width on qwen3moe — 6144 where an expert
     * is 768 — so the expert's own key wins when the file carries it. Getting this wrong is
     * caught below by the stack-geometry check rather than by the output, but the message
     * would blame the expert count for a feed-forward mistake. */
    snprintf(key, sizeof(key), "%s.expert_feed_forward_length", gf->arch);
    kv = gguf_get_kv(gf, key);
    if (kv && (int)kv->val.u32 > 0) m->ffn = (int)kv->val.u32;

    /* llama.cpp renormalises the chosen weights for qwen3moe and does not for olmoe —
     * src/models/qwen3moe.cpp passes norm_w = true where src/models/olmoe.cpp passes false.
     * Nothing in either GGUF says so, so it is the name that decides and the reference that
     * is cited. */
    m->renorm_topk = (strcmp(gf->arch, "qwen3moe") == 0);
    if (m->n_expert <= 0 || m->n_expert > OLMOE_MAX_EXPERTS ||
        m->n_expert_used <= 0 || m->n_expert_used > OLMOE_MAX_USED ||
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

    /* Whether the QK norm is one weight for the projection or one for each head is a
     * property of the tensor, so it is read off the tensor. OLMoE ships [q_dim]; qwen3moe
     * ships [head_dim], the same shape qwen3 does. Both are plausible on sight and the
     * difference is invisible until the logits are compared. */
    ti = gguf_find_tensor(gf, "blk.0.attn_q_norm.weight");
    if (ti >= 0) {
        uint64_t ne = gf->tensors[ti].n_elements;
        if (ne == (uint64_t)m->head_dim)      m->qk_norm_per_head = 1;
        else if (ne == (uint64_t)m->q_dim)    m->qk_norm_per_head = 0;
        else {
            fprintf(stderr, "olmoe: attn_q_norm is %llu long, neither head_dim %d nor q_dim %d\n",
                    (unsigned long long)ne, m->head_dim, m->q_dim);
            free(m);
            return NULL;
        }
    }

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
    char taken[OLMOE_MAX_EXPERTS];
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

static int olmoe_forward(void *model, kv_cache *kv, const int *tokens, int n,
                         int pos0, float *logits) {
    olmoe_model *m = (olmoe_model*)model;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->n_layers, m->kv_dim);
    if (rc != NT_OK) return rc;

    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    int NE = m->n_expert, NU = m->n_expert_used;
    float eps = m->rms_eps;
    int gqa = H / KV;

    double pft = pf_mark();
    float *x = (float*)calloc((size_t)n * E, sizeof(float));
    if (!x) return NT_E_MEMORY;
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
    /* Room for every chosen expert's gate and up at once: the two matrices that share a
     * token's activation and can therefore be read in one fan-out. `down` cannot join them —
     * each expert feeds it a different vector. */
    float *eg = (float*)calloc((size_t)NU * FFN, sizeof(float));
    float *eu = (float*)calloc((size_t)NU * FFN, sizeof(float));
    float *eo = (float*)calloc((size_t)E, sizeof(float));

    /* The expert-grouped prefill's working set, allocated only when there is a group to
     * make. grp_part is the big one — one E-wide result per (position, slot) so the eight
     * contributions can be summed in slot order after being computed out of it — and at 32
     * positions of a 2048-wide body that is 2 MB. A failure here is not fatal: grouping is
     * an optimisation and the per-position path below is the definition of the answer. */
    float *grp_part = NULL, *grp_x = NULL, *grp_g = NULL, *grp_u = NULL, *grp_o = NULL, *grp_w = NULL;
    /* The quantized layer input, its per-expert gather, and the same pair for the down
     * projection. These are why the grouping is worth doing at all: without them every
     * expert re-quantized about two positions of activation and paid three mallocs for the
     * privilege, which measured 2.4x slower than not grouping. */
    int8_t *grp_qa = NULL, *grp_cqa = NULL, *grp_dqa = NULL;
    float *grp_da = NULL, *grp_cda = NULL, *grp_dda = NULL;
    int32_t *grp_as = NULL, *grp_cas = NULL, *grp_das = NULL;
    int *grp_idx = NULL, *grp_pos = NULL, *grp_slot = NULL;
    if (n > 1) {
        long nsubE = E / 32, nsubF = FFN / 32;
        grp_part = (float*)malloc((size_t)n * NU * E * sizeof(float));
        grp_x    = (float*)malloc((size_t)n * E * sizeof(float));
        grp_g    = (float*)malloc((size_t)n * FFN * sizeof(float));
        grp_u    = (float*)malloc((size_t)n * FFN * sizeof(float));
        grp_o    = (float*)malloc((size_t)n * E * sizeof(float));
        grp_w    = (float*)malloc((size_t)n * NU * sizeof(float));
        grp_qa   = (int8_t*)malloc((size_t)n * E);
        grp_cqa  = (int8_t*)malloc((size_t)n * E);
        grp_dqa  = (int8_t*)malloc((size_t)n * FFN);
        grp_da   = (float*)malloc((size_t)n * nsubE * sizeof(float));
        grp_cda  = (float*)malloc((size_t)n * nsubE * sizeof(float));
        grp_dda  = (float*)malloc((size_t)n * nsubF * sizeof(float));
        grp_as   = (int32_t*)malloc((size_t)n * nsubE * sizeof(int32_t));
        grp_cas  = (int32_t*)malloc((size_t)n * nsubE * sizeof(int32_t));
        grp_das  = (int32_t*)malloc((size_t)n * nsubF * sizeof(int32_t));
        grp_idx  = (int*)malloc((size_t)n * NU * sizeof(int));
        grp_pos  = (int*)malloc((size_t)n * NU * sizeof(int));
        grp_slot = (int*)malloc((size_t)n * NU * sizeof(int));
        if (!grp_part || !grp_x || !grp_g || !grp_u || !grp_o || !grp_w ||
            !grp_qa || !grp_cqa || !grp_dqa || !grp_da || !grp_cda || !grp_dda ||
            !grp_as || !grp_cas || !grp_das ||
            !grp_idx || !grp_pos || !grp_slot) {
            free(grp_part); free(grp_x); free(grp_g); free(grp_u); free(grp_o); free(grp_w);
            free(grp_qa); free(grp_cqa); free(grp_dqa); free(grp_da); free(grp_cda);
            free(grp_dda); free(grp_as); free(grp_cas); free(grp_das);
            free(grp_idx); free(grp_pos); free(grp_slot);
            grp_part = NULL;
        }
    }

    if (!xn || !q_all || !k_new || !v_new || !attn_out || !ffn_out || !router ||
        !eg || !eu || !eo) {
        free(x); free(xn); free(q_all); free(k_new); free(v_new);
        free(attn_out); free(ffn_out); free(router); free(eg); free(eu); free(eo);
        return NT_E_MEMORY;
    }

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

        /* OLMoE normalises the projection whole and then splits it into heads; qwen3moe
         * normalises each head. Doing either one the other way divides by a different scale
         * and is a different model, and both still produce fluent text. */
        pft = pf_mark();
        for (int j = 0; j < n; j++) {
            if (m->qk_norm_per_head) {
                float *qj = q_all + (long)j * Q_DIM, *kj = k_new + (long)j * KVD;
                for (int h = 0; h < H; h++)
                    rmsnorm(qj + h * HD, qj + h * HD, m->layers[l].q_norm, HD, eps);
                for (int h = 0; h < KV; h++)
                    rmsnorm(kj + h * HD, kj + h * HD, m->layers[l].k_norm, HD, eps);
            } else {
                rmsnorm(q_all + (long)j * Q_DIM, q_all + (long)j * Q_DIM,
                        m->layers[l].q_norm, Q_DIM, eps);
                rmsnorm(k_new + (long)j * KVD, k_new + (long)j * KVD,
                        m->layers[l].k_norm, KVD, eps);
            }
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

        /* Router: sixty-four scores over the layer's normalised input, for every row at once.
         *
         * This is the one f32 weight the family ships, and it is measurably expensive for its
         * size — the block holding it took 26.7 percent of decode against seven percent of the
         * bytes read, because f32 goes through a matvec without the integer kernel's dot
         * instruction rather than because of its width. Packing it to Q8_0 at load was tried
         * and is worth about nine percent of decode (18.6 -> 20.5 t/s), but routing is a
         * discrete decision: a small numeric change reorders neighbouring scores, a different
         * eight of sixty-four run, and the text diverges from the reference where every other
         * prompt matches it. The reference routes in f32. Until that trade is somebody's
         * decision rather than a side effect, so does this. */
        pft = pf_mark();
        qmm(router, &m->layers[l].gate_inp, xn, n);
        pf_add(PF_QKV, pft);

        memset(ffn_out, 0, (size_t)n * E * sizeof(float));

        /* Prefill groups the chunk by expert; decode cannot and does not try.
         *
         * A mixture reads a different eighth of the layer for every position, so the tile a
         * dense body carries through one pass over the weights has nothing in common here —
         * which is why arch_olmoe was the one family the batched kernels did not help. The
         * grouping is the answer llama.cpp reaches through ggml_mul_mat_id: invert the loop,
         * collect the positions that chose each expert, and read that expert once for all of
         * them. With 32 positions over 128 experts that is about two positions per read; over
         * OLMoE's 64 it is about four.
         *
         * What must not change is the order the contributions are added in. Regrouping
         * computes them out of order by construction, so each position's eight results are
         * kept apart and summed afterwards ascending by slot, which is the order the
         * per-position path adds them in and therefore the same float. The path below stays
         * as the fallback and as the definition of that order.
         *
         * Routing is done for the whole chunk first because the grouping needs to see every
         * position's choice before it can read any expert. */
        int grouped = (n > 1 && grp_part != NULL);
        if (grouped) {
            long nsubE = E / 32;
            for (int j = 0; j < n; j++) {
                float *probs = router + (long)j * NE;
                int *ridx = grp_idx + (long)j * NU;
                float *rw  = grp_w  + (long)j * NU;
                pft = pf_mark();
                softmax(probs, NE);
                top_k(probs, NE, NU, ridx);
                for (int e = 0; e < NU; e++) rw[e] = probs[ridx[e]];
                if (m->renorm_topk) {
                    float s = 0;
                    for (int e = 0; e < NU; e++) s += rw[e];
                    if (s > 0) for (int e = 0; e < NU; e++) rw[e] /= s;
                }
                pf_add(PF_SILU, pft);
            }

            /* The layer's input, quantized once for every expert that will read it. This
             * line is the difference between the grouping paying and costing. */
            pft = pf_mark();
            /* The granularity the gate and up experts want. They share a dtype in every
             * mixture here; a per-256 scale is a valid scale for any kernel that reads it,
             * only a coarser one, so a mismatch would cost accuracy and not correctness. */
            nt_quant_act_batch(xn, m->layers[l].gate_exps.dtype, E, n, grp_qa, grp_da, grp_as);
            pf_add(PF_FFN, pft);

            for (int ex = 0; ex < NE; ex++) {
                int cnt = 0;
                for (int j = 0; j < n; j++)
                    for (int e = 0; e < NU; e++)
                        if (grp_idx[(long)j * NU + e] == ex) {
                            grp_pos[cnt] = j; grp_slot[cnt] = e; cnt++;
                        }
                if (!cnt) continue;

                /* No abandoning the grouping once routing has run: softmax has already been
                 * applied in place, and the per-position path below would apply it a second
                 * time. A weight the batched entry will not take is handled here instead,
                 * per position through the same matvec the fallback would have used, which
                 * writes into the same slots and keeps the same order. */
                wt gw, uw, dw;
                int have = wt_expert(&gw, &m->layers[l].gate_exps, ex, FFN)
                        && wt_expert(&uw, &m->layers[l].up_exps,   ex, FFN)
                        && wt_expert(&dw, &m->layers[l].down_exps, ex, E);
                if (!have) {
                    for (int c = 0; c < cnt; c++)
                        memset(grp_part + ((long)grp_pos[c] * NU + grp_slot[c]) * E, 0,
                               (size_t)E * sizeof(float));
                    continue;
                }

                /* This expert's columns out of the tile quantized once above. A column is
                 * k bytes plus two small arrays; against the matmul that follows, free. */
                for (int c = 0; c < cnt; c++) {
                    long jj = grp_pos[c];
                    memcpy(grp_x   + (long)c * E,     xn      + jj * E, (size_t)E * sizeof(float));
                    memcpy(grp_cqa + (long)c * E,     grp_qa  + jj * E, (size_t)E);
                    memcpy(grp_cda + (long)c * nsubE, grp_da  + jj * nsubE, (size_t)nsubE * sizeof(float));
                    memcpy(grp_cas + (long)c * nsubE, grp_as  + jj * nsubE, (size_t)nsubE * sizeof(int32_t));
                }

                pft = pf_mark();
                int batched = gw.q && uw.q && dw.q &&
                    nt_qmatmul_i8_pre(grp_g, gw.q, gw.dtype, grp_cqa, grp_cda, grp_cas,
                                      FFN, E, cnt) == 0 &&
                    nt_qmatmul_i8_pre(grp_u, uw.q, uw.dtype, grp_cqa, grp_cda, grp_cas,
                                      FFN, E, cnt) == 0;
                if (!batched)
                    for (int c = 0; c < cnt; c++) {
                        qmv(grp_g + (long)c * FFN, &gw, grp_x + (long)c * E);
                        qmv(grp_u + (long)c * FFN, &uw, grp_x + (long)c * E);
                    }
                pf_add(PF_FFN, pft);

                pft = pf_mark();
                for (long i = 0; i < (long)cnt * FFN; i++) {
                    float g = grp_g[i];
                    grp_g[i] = (g / (1.0f + expf(-g))) * grp_u[i];
                }
                pf_add(PF_SILU, pft);

                pft = pf_mark();
                if (!(batched &&
                      nt_quant_act_batch(grp_g, dw.dtype, FFN, cnt, grp_dqa, grp_dda, grp_das) == 0 &&
                      nt_qmatmul_i8_pre(grp_o, dw.q, dw.dtype, grp_dqa, grp_dda, grp_das,
                                        E, FFN, cnt) == 0))
                    for (int c = 0; c < cnt; c++)
                        qmv(grp_o + (long)c * E, &dw, grp_g + (long)c * FFN);
                pf_add(PF_FFN, pft);

                for (int c = 0; c < cnt; c++)
                    memcpy(grp_part + ((long)grp_pos[c] * NU + grp_slot[c]) * E,
                           grp_o + (long)c * E, (size_t)E * sizeof(float));
            }
        }
        if (grouped) {
            for (int j = 0; j < n; j++) {
                float *dst = ffn_out + (long)j * E;
                const float *rw = grp_w + (long)j * NU;
                for (int e = 0; e < NU; e++)
                    axpy_f32(dst, rw[e], grp_part + ((long)j * NU + e) * E, E);
            }
            pft = pf_mark();
            for (long i = 0; i < (long)n * E; i++) x[i] += ffn_out[i];
            pf_add(PF_RESID, pft);
            continue;
        }

        for (int j = 0; j < n; j++) {
            float *probs = router + (long)j * NE;
            const float *xj = xn + (long)j * E;
            float *dst = ffn_out + (long)j * E;
            int idx[OLMOE_MAX_USED];

            pft = pf_mark();
            softmax(probs, NE);
            top_k(probs, NE, NU, idx);
            /* Renormalised over the chosen experts, or taken as they stand. Nothing in
             * either file records which, so it was decided at load from the name and the
             * reference. Writing it back into `probs` keeps the one place downstream that
             * reads a weight reading one thing. */
            float w[OLMOE_MAX_USED];
            for (int e = 0; e < NU; e++) w[e] = probs[idx[e]];
            if (m->renorm_topk) {
                float s = 0;
                for (int e = 0; e < NU; e++) s += w[e];
                if (s > 0) for (int e = 0; e < NU; e++) w[e] /= s;
            }
            pf_add(PF_SILU, pft);

            /* Slice first, all of them, before any arithmetic. Load has already checked that
             * every stack divides into n_expert slices of the right height, so this cannot
             * fire on a file that loaded. If it ever does, the token gets no feed-forward at
             * all rather than seven eighths of one: a missing contribution is visible in the
             * output, a quietly missing expert is not. */
            wt ge[OLMOE_MAX_USED], ue[OLMOE_MAX_USED], de[OLMOE_MAX_USED];
            int sliced = 1;
            for (int e = 0; e < NU && sliced; e++)
                sliced = wt_expert(&ge[e], &m->layers[l].gate_exps, idx[e], FFN)
                      && wt_expert(&ue[e], &m->layers[l].up_exps,   idx[e], FFN)
                      && wt_expert(&de[e], &m->layers[l].down_exps, idx[e], E);
            if (!sliced) {
                static int warned = 0;
                if (!warned) {
                    fprintf(stderr, "olmoe: an expert of layer %d would not slice; "
                                    "this token's feed-forward is dropped\n", l);
                    warned = 1;
                }
                memset(dst, 0, (size_t)E * sizeof(float));
                continue;
            }

            /* Gate and up read a different eighth of the layer for the same activation, so
             * they go out as one fan-out each rather than eight: the activation is quantized
             * once instead of eight times, the pool is woken once, and a worker's run of rows
             * is eight times longer. The experts stay where they are — copying them adjacent
             * would cost the bandwidth this is trying to save. */
            const uint8_t *gs[OLMOE_MAX_USED], *us[OLMOE_MAX_USED];
            int gatherable = ge[0].use_i8 && ue[0].use_i8;
            for (int e = 0; e < NU && gatherable; e++) {
                gs[e] = ge[e].q; us[e] = ue[e].q;
                gatherable = gs[e] && us[e];
            }

            pft = pf_mark();
            if (gatherable &&
                nt_qmatvec_i8_gather(eg, gs, NU, ge[0].dtype, xj, FFN, E) == 0 &&
                nt_qmatvec_i8_gather(eu, us, NU, ue[0].dtype, xj, FFN, E) == 0) {
                /* done */
            } else {
                for (int e = 0; e < NU; e++) {
                    qmv(eg + (long)e * FFN, &ge[e], xj);
                    qmv(eu + (long)e * FFN, &ue[e], xj);
                }
            }
            pf_add(PF_FFN, pft);

            pft = pf_mark();
            for (long i = 0; i < (long)NU * FFN; i++) {
                float g = eg[i];
                eg[i] = (g / (1.0f + expf(-g))) * eu[i];
            }
            pf_add(PF_SILU, pft);

            for (int e = 0; e < NU; e++) {
                pft = pf_mark();
                qmv(eo, &de[e], eg + (long)e * FFN);
                pf_add(PF_FFN, pft);
                axpy_f32(dst, w[e], eo, E);
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
    free(grp_part); free(grp_x); free(grp_g); free(grp_u); free(grp_o); free(grp_w);
    free(grp_qa); free(grp_cqa); free(grp_dqa); free(grp_da); free(grp_cda);
    free(grp_dda); free(grp_as); free(grp_cas); free(grp_das);
    free(grp_idx); free(grp_pos); free(grp_slot);
    return NT_OK;
}

static const char *const olmoe_names[] = { "olmoe", "qwen3moe", NULL };

const nt_arch nt_arch_olmoe = {
    .names = olmoe_names,
    .load = olmoe_load,
    .free = olmoe_free,
    .forward = olmoe_forward,
};
