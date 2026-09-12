/* arch_llama.c — the llama, qwen2 and qwen3 family.
 *
 * SmolLM2, nanollama, Qwen2.5, Qwen3, LLaMA, Mistral, and the distills of any of
 * them: any GGUF whose blocks are attn_norm / attn_{q,k,v,output} / ffn_norm /
 * ffn_{gate,up,down}. GQA, bias and tied embeddings are read off the file rather
 * than configured.
 *
 * Qwen3 lives here rather than in a file of its own because it is this shape plus
 * two tensors. It drops the qkv bias Qwen2 carries — already optional here, so it
 * costs nothing — and adds an RMS norm over each head of q and of k before the
 * rotation. Two optional weights and two lines in the rotation loop is not a
 * family; a separate file would be 280 lines copied to hold them. arch.h asks
 * whether a family can be added without editing runtime.c, and the question
 * before that is whether it is a family at all.
 *
 * What is not here: qwen3moe, which routes its feed-forward to experts and
 * belongs beside olmoe rather than beside this.
 *
 * Moved from examples/infer_llama.c with the arithmetic untouched. That file
 * stays put as the reference this is measured against.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
        wt wgate, wup, wdown;
    } layers[];
} llama_model;

static void llama_free(void *model);

static void *llama_load(gguf_file *gf, nt_dims *dims) {
    int nl = gf->n_layers;
    llama_model *m = (llama_model*)calloc(1, sizeof(llama_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;
    /* Which lanes the rotation pairs is a property of the converter, not of the
     * file, so it is a list and the list is named rather than inferred. llama
     * and mistral come out of llama.cpp's converter with the weights permuted
     * for adjacent-lane rotation; qwen2 and qwen3 rotate halves. Writing this as
     * "anything that is not llama is neox" made the answer for a new name a
     * coin-flip, and mistral3 is the name that would have lost it. */
    m->rope_neox = !(strcmp(gf->arch, "llama") == 0 || strcmp(gf->arch, "mistral3") == 0);

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) {
        m->q_dim = (int)gf->tensors[ti].shape[1];
        m->head_dim = m->q_dim / m->n_heads;
    } else {
        m->head_dim = m->embed / m->n_heads;
        m->q_dim = m->n_heads * m->head_dim;
    }
    m->kv_dim = m->head_dim * m->n_kv_heads;

    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->vocab = (int)gf->tensors[ti].shape[1];
    else if (gf->vocab_size > 0) m->vocab = gf->vocab_size;
    else m->vocab = 32000;

    fprintf(stderr, "llama: E=%d H=%d KV=%d FFN=%d V=%d L=%d HD=%d Q=%d\n",
            m->embed, m->n_heads, m->n_kv_heads, m->ffn, m->vocab, nl,
            m->head_dim, m->q_dim);

    m->gf = gf;
    m->emb_ti = gguf_find_tensor(gf, "token_embd.weight");
    wt_load(&m->tok_emb, gf, "token_embd.weight");
    ti = gguf_find_tensor(gf, "output_norm.weight");
    if (ti >= 0) m->out_norm = gguf_dequant(gf, ti);   /* [embed], f32 either way */
    m->has_output_weight = wt_load(&m->out_weight, gf, "output.weight");

    int missing = 0;
    for (int l = 0; l < nl; l++) {
        char name[128];
        /* 1-D: norms and biases are a few thousand floats and are read
         * elementwise, so they are expanded. 2-D: everything the matvec
         * touches stays packed. */
        #define L(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            ti = gguf_find_tensor(gf, name); \
            if (ti >= 0) m->layers[l].field = gguf_dequant(gf, ti); \
        } while(0)
        /* wt_load's answer was thrown away here until 2026-09-12, and a file
         * missing one attention matrix loaded, ran, and answered — with a
         * different sentence on a scalar build and a BLAS abort on Accelerate,
         * from the same file. A weight this forward will dereference is not
         * optional, so the loop stops at the first one that is not there and
         * names it. The 1-D reads above stay optional: biases genuinely are. */
        #define W(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            if (!wt_load(&m->layers[l].field, gf, name)) { \
                fprintf(stderr, "llama: required tensor '%s' missing or unreadable\n", name); \
                missing = 1; \
            } \
        } while(0)
        L(attn_norm, "blk.%d.attn_norm.weight");
        W(wq, "blk.%d.attn_q.weight");
        W(wk, "blk.%d.attn_k.weight");
        W(wv, "blk.%d.attn_v.weight");
        W(wo, "blk.%d.attn_output.weight");
        L(q_bias, "blk.%d.attn_q.bias");
        L(k_bias, "blk.%d.attn_k.bias");
        L(v_bias, "blk.%d.attn_v.bias");
        L(q_norm, "blk.%d.attn_q_norm.weight");
        L(k_norm, "blk.%d.attn_k_norm.weight");
        L(ffn_norm, "blk.%d.ffn_norm.weight");
        W(wgate, "blk.%d.ffn_gate.weight");
        W(wup, "blk.%d.ffn_up.weight");
        W(wdown, "blk.%d.ffn_down.weight");
        #undef L
        #undef W
    }

    if (!(m->tok_emb.q || m->tok_emb.f32) || !m->out_norm) {
        fprintf(stderr, "llama: missing critical weights\n");
        missing = 1;
    }
    /* llama_free and not free: every tensor already expanded belongs to this
     * model, and dropping the struct alone left them behind — on a 24B that is
     * the difference between a refused load and a refused load that took the
     * machine down with it. */
    if (missing) { llama_free(m); return NULL; }
    if (m->layers[0].q_bias) fprintf(stderr, "  (has attention bias — qwen-style)\n");
    if (!m->has_output_weight) fprintf(stderr, "  (tied embeddings)\n");
    fprintf(stderr, "  rope: %s | weights: packed%s\n", m->rope_neox ? "neox" : "norm",
            m->tok_emb.use_i8 ? " (int8 dot)" : "");

    dims->n_layers = m->n_layers;
    dims->kv_dim = m->kv_dim;
    dims->vocab = m->vocab;
    return m;
}

/* Only the expanded copies are ours; a packed pointer belongs to the gguf_file. */
static void llama_free(void *model) {
    llama_model *m = (llama_model*)model;
    if (!m) return;
    free(m->tok_emb.f32); free(m->out_norm); free(m->out_weight.f32);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm);
        free(m->layers[l].wq.f32); free(m->layers[l].wk.f32);
        free(m->layers[l].wv.f32); free(m->layers[l].wo.f32);
        free(m->layers[l].q_bias); free(m->layers[l].k_bias); free(m->layers[l].v_bias);
        free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].ffn_norm);
        free(m->layers[l].wgate.f32); free(m->layers[l].wup.f32); free(m->layers[l].wdown.f32);
    }
    free(m);
}

/* One forward for a group of consecutive positions. Decode calls it with n = 1
 * and gets exactly the old arithmetic; prefill calls it with a chunk of the
 * prompt and the weight traffic drops by the chunk width, which is the whole
 * difference between a prompt that costs the same as generating it and one
 * that does not. Attention still runs per row — it reads the KV cache rather
 * than the weights, so batching it buys little and costs a mask. */
static int llama_forward(void *model, kv_cache *kv, const int *tokens, int n,
                         int pos0, float *logits) {
    llama_model *m = (llama_model*)model;
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, m->n_layers, m->kv_dim);
    if (rc != NT_OK) return rc;

    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    float eps = m->rms_eps;
    int gqa = H / KV;

    /* Rows of the embedding table, decoded where they lie. For a 1.5B Qwen
     * that table is the largest tensor in the file and expanding it would cost
     * 933 MB to read 1536 floats per token. */
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
    float *ffn_gate = (float*)calloc((size_t)n * FFN, sizeof(float));
    float *ffn_up = (float*)calloc((size_t)n * FFN, sizeof(float));
    float *ffn_out = (float*)calloc((size_t)n * E, sizeof(float));
    if (!xn || !q_all || !k_new || !v_new || !attn_out || !ffn_gate || !ffn_up || !ffn_out) {
        free(x); free(xn); free(q_all); free(k_new); free(v_new);
        free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
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
        for (int j = 0; j < n; j++) {
            add_bias(q_all + (long)j * Q_DIM, m->layers[l].q_bias, Q_DIM);
            add_bias(k_new + (long)j * KVD, m->layers[l].k_bias, KVD);
            add_bias(v_new + (long)j * KVD, m->layers[l].v_bias, KVD);
        }
        pf_add(PF_QKV, pft);

        pft = pf_mark();
        long base = (long)l * kv->max_seq * KVD;
        for (int j = 0; j < n; j++) {
            int pos = pos0 + j;
            float *qj = q_all + (long)j * Q_DIM, *kj = k_new + (long)j * KVD;
            /* Qwen3 normalises each head of q and k before rotating it, with the
             * model's own epsilon. Before, not after: the rotation mixes lanes
             * within a head, so a norm taken afterwards is a different function.
             * Absent weights mean an older file and the loop is the old loop. */
            for (int h = 0; h < H; h++) {
                if (m->layers[l].q_norm)
                    rmsnorm(qj + h*HD, qj + h*HD, m->layers[l].q_norm, HD, eps);
                rope(qj + h*HD, pos, HD, m->rope_base, m->rope_neox);
            }
            for (int h = 0; h < KV; h++) {
                if (m->layers[l].k_norm)
                    rmsnorm(kj + h*HD, kj + h*HD, m->layers[l].k_norm, HD, eps);
                rope(kj + h*HD, pos, HD, m->rope_base, m->rope_neox);
            }
            memcpy(kv->k + base + (long)pos * KVD, kj, KVD * sizeof(float));
            memcpy(kv->v + base + (long)pos * KVD, v_new + (long)j * KVD, KVD * sizeof(float));
        }
        pf_add(PF_ROPE, pft);

        /* GQA attention, causal: row j sees 0..pos0+j, all already in cache */
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
        pft = pf_mark();
        qmm(ffn_gate, &m->layers[l].wgate, xn, n);
        qmm(ffn_up, &m->layers[l].wup, xn, n);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (long i = 0; i < (long)n * FFN; i++) {
            float g = ffn_gate[i];
            ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i];
        }
        pf_add(PF_SILU, pft);
        pft = pf_mark();
        qmm(ffn_out, &m->layers[l].wdown, ffn_gate, n);
        pf_add(PF_FFN, pft);
        pft = pf_mark();
        for (long i = 0; i < (long)n * E; i++) x[i] += ffn_out[i];
        pf_add(PF_RESID, pft);
    }

    /* The head is the single largest matvec in the model — 151936 rows against
     * 896 for a 0.5B Qwen — so running it at every prompt position spends a
     * tenth of the prefill on distributions nobody reads. */
    if (logits) {
        pft = pf_mark();
        rmsnorm(xn, x + (long)(n - 1) * E, m->out_norm, E, eps);
        const wt *lm_head = m->has_output_weight ? &m->out_weight : &m->tok_emb;
        qmv(logits, lm_head, xn);
        pf_add(PF_HEAD, pft);
    }

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
    return NT_OK;
}

static const char *const llama_names[] = { "llama", "mistral3", "qwen2", "qwen3", NULL };

const nt_arch nt_arch_llama = {
    .names = llama_names,
    .load = llama_load,
    .free = llama_free,
    .forward = llama_forward,
};
