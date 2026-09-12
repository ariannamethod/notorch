/* arch_mamba.c — the first family here with no attention at all.
 *
 * Everything this tree has run so far keeps a KV cache: every token looks back at every token
 * before it, and the cost of a forward grows with the conversation. Mamba does not look back.
 * It carries a state of fixed size — a short convolution window and a small matrix per channel
 * — and each token updates that state and reads an answer out of it. Context length stops
 * being a memory question. The file says ctx = 1048576 and means it.
 *
 * Which makes this the test `arch.h` set for itself: a family that cannot be added without
 * editing runtime.c would mean the interface is lying. It is not, but one parameter turns out
 * to be meaningless here — `kv_cache` has nowhere to go, because the state is not per position.
 * It lives in the model instead and resets when a sequence starts. See the note on `pos0`.
 *
 * The arithmetic is the reference's, read from ggml's own kernels rather than from the paper:
 * ggml_compute_forward_ssm_conv_f32 and ggml_compute_forward_ssm_scan_f32 in ggml-cpu/ops.cpp,
 * with the layer assembled in llama.cpp's models/mamba-base.cpp. Where they differ from what
 * a description of Mamba would suggest, they win — in particular the decay factor here is
 * per state element, which is Mamba-1; Mamba-2 has one per head and is a different family.
 *
 * Prints go to stderr: stdout belongs to the model. */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n_layers, embed, vocab;
    int d_conv, d_inner, d_state, dt_rank;
    float rms_eps;

    gguf_file *gf;
    int emb_ti;

    wt tok_emb;            /* also the head: this family ties them */
    float *out_norm;
    int has_output_weight;
    wt out_weight;

    /* The recurrent state, one slice per layer. conv holds the last d_conv-1 inputs per
     * channel; ssm holds d_state numbers per channel. Both are the whole of what this model
     * remembers — there is no third place a token could hide in. */
    float *conv_state;     /* [n_layers][d_inner][d_conv-1] */
    float *ssm_state;      /* [n_layers][d_inner][d_state]  */

    /* Working room for one chunk, kept rather than taken per call, and nothing in it is read
     * before it is written — so malloc, not calloc.
     *
     * It was taken per call at first, and the first pass of a repeat then read 99.1 t/s of
     * prefill against 47.8 and 47.6 for the two after it. The allocator looked like the
     * answer: calloc over pages the kernel has just handed out is free because they arrive
     * zeroed, while calloc over recycled heap has to write every byte. It was not the
     * answer. Holding the buffer here changed those numbers by nothing, and the cause turned
     * out to be thread affinity in the pool, a floor below this file. The buffer stays
     * because it belongs here; the explanation is recorded as wrong because it was. */
    float *work;
    int    work_pos;       /* positions the buffer is sized for; grows, never shrinks */

    struct {
        float *attn_norm;
        wt in;             /* [2*d_inner, embed] */
        float *conv_w;     /* [d_inner][d_conv] */
        float *conv_b;     /* [d_inner] */
        wt x_proj;         /* [dt_rank + 2*d_state, d_inner] */
        wt dt_proj;        /* [d_inner, dt_rank] */
        float *dt_b;       /* [d_inner] */
        float *A;          /* [d_inner][d_state] */
        float *D;          /* [d_inner] */
        wt out;            /* [embed, d_inner] */
    } layers[];
} mamba_model;

static float *load_f32(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    return ti < 0 ? NULL : gguf_dequant(gf, ti);
}

static int key_int(gguf_file *gf, const char *name, int fallback) {
    const gguf_kv *kv = gguf_get_kv(gf, name);
    if (!kv) return fallback;
    if (kv->type == 4) return (int)kv->val.u32;
    if (kv->type == 5) return kv->val.i32;
    return fallback;
}

/* The reference's softplus, threshold included. Above twenty the exponential overflows long
 * before the logarithm would change the answer, so it is returned unchanged — dropping that
 * guard is a NaN on any prompt that drives dt high enough. */
static float softplus(float v) {
    return v > 20.0f ? v : logf(1.0f + expf(v));
}

static void *mamba_load(gguf_file *gf, nt_dims *dims) {
    int nl = gf->n_layers;
    mamba_model *m = (mamba_model*)calloc(1, sizeof(mamba_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->embed = gf->embed_dim;
    m->rms_eps = gf->rms_eps;
    m->gf = gf;

    m->d_conv  = key_int(gf, "mamba.ssm.conv_kernel", 0);
    m->d_inner = key_int(gf, "mamba.ssm.inner_size", 0);
    m->d_state = key_int(gf, "mamba.ssm.state_size", 0);
    m->dt_rank = key_int(gf, "mamba.ssm.time_step_rank", 0);
    if (m->d_conv < 2 || m->d_inner <= 0 || m->d_state <= 0 || m->dt_rank <= 0) {
        fprintf(stderr, "mamba: ssm parameters missing or unusable "
                        "(conv=%d inner=%d state=%d dt_rank=%d)\n",
                m->d_conv, m->d_inner, m->d_state, m->dt_rank);
        free(m);
        return NULL;
    }
    /* The reference refuses anything but an expansion of two, and the layer arithmetic below
     * assumes it as well — the gate for that assumption is here rather than in a comment. */
    if (m->d_inner != 2 * m->embed) {
        fprintf(stderr, "mamba: inner size %d is not twice the embedding %d\n",
                m->d_inner, m->embed);
        free(m);
        return NULL;
    }

    m->emb_ti = gguf_find_tensor(gf, "token_embd.weight");
    if (!wt_load(&m->tok_emb, gf, "token_embd.weight") || m->emb_ti < 0) {
        fprintf(stderr, "mamba: no token_embd\n");
        free(m);
        return NULL;
    }
    m->vocab = m->tok_emb.rows;
    m->out_norm = load_f32(gf, "output_norm.weight");
    m->has_output_weight = wt_load(&m->out_weight, gf, "output.weight");

    int ok = m->out_norm != NULL;
    for (int l = 0; l < nl && ok; l++) {
        char nm[128];
        #define T(dst, fmt) (snprintf(nm, sizeof(nm), fmt, l), wt_load(&m->layers[l].dst, gf, nm))
        #define F(dst, fmt) (snprintf(nm, sizeof(nm), fmt, l), m->layers[l].dst = load_f32(gf, nm), \
                             m->layers[l].dst != NULL)
        ok = F(attn_norm, "blk.%d.attn_norm.weight")
          && T(in,      "blk.%d.ssm_in.weight")
          && F(conv_w,  "blk.%d.ssm_conv1d.weight")
          && F(conv_b,  "blk.%d.ssm_conv1d.bias")
          && T(x_proj,  "blk.%d.ssm_x.weight")
          && T(dt_proj, "blk.%d.ssm_dt.weight")
          && F(dt_b,    "blk.%d.ssm_dt.bias")
          && F(A,       "blk.%d.ssm_a")          /* no "weight" suffix on these two */
          && F(D,       "blk.%d.ssm_d")
          && T(out,     "blk.%d.ssm_out.weight");
        #undef T
        #undef F
    }
    if (!ok) {
        fprintf(stderr, "mamba: missing weights\n");
        free(m);
        return NULL;
    }

    size_t conv_n = (size_t)nl * m->d_inner * (m->d_conv - 1);
    size_t ssm_n  = (size_t)nl * m->d_inner * m->d_state;
    m->conv_state = (float*)calloc(conv_n, sizeof(float));
    m->ssm_state  = (float*)calloc(ssm_n, sizeof(float));
    if (!m->conv_state || !m->ssm_state) {
        fprintf(stderr, "mamba: no room for the recurrent state\n");
        free(m->conv_state); free(m->ssm_state); free(m);
        return NULL;
    }

    fprintf(stderr, "mamba: E=%d L=%d V=%d d_conv=%d d_inner=%d d_state=%d dt_rank=%d, "
                    "state %.2f MiB (no kv cache)\n",
            m->embed, m->n_layers, m->vocab, m->d_conv, m->d_inner, m->d_state, m->dt_rank,
            (double)((conv_n + ssm_n) * sizeof(float)) / (1024.0 * 1024.0));

    dims->n_layers = m->n_layers;
    /* Nothing per position to cache. One float a layer keeps the caller's allocation honest
     * and unused rather than asking it to special-case this family. */
    dims->kv_dim = 1;
    dims->vocab = m->vocab;
    return m;
}

static void mamba_free(void *model) {
    mamba_model *m = (mamba_model*)model;
    if (!m) return;
    free(m->tok_emb.f32); free(m->out_norm); free(m->out_weight.f32);
    free(m->conv_state); free(m->ssm_state); free(m->work);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm);
        free(m->layers[l].in.f32);
        free(m->layers[l].conv_w); free(m->layers[l].conv_b);
        free(m->layers[l].x_proj.f32); free(m->layers[l].dt_proj.f32);
        free(m->layers[l].dt_b); free(m->layers[l].A); free(m->layers[l].D);
        free(m->layers[l].out.f32);
    }
    free(m);
}

/* A chunk of consecutive positions through one layer, state in and state out.
 *
 * This was written per token, on the reasoning that the scan is sequential by construction —
 * token t's state is token t-1's output — so a batched version would have to walk them in
 * order anyway and prefill must cost what decode costs. Half of that is true and the half
 * that is false was expensive. The scan walks in order. The convolution walks in order. The
 * four projections around them do not look across positions at all, and they are where the
 * weights are: a 350-token prompt was streaming the whole file 350 times to compute what one
 * pass could. So the order survives and the traffic does not — each projection now runs once
 * for the chunk through qmm, and the two sequential parts keep their loops.
 *
 * Nothing about the arithmetic moves. qmm is bit-identical to qmv called per row, which
 * tests/test_qmatmul.c asserts as equality rather than a tolerance, and both sequential loops
 * visit positions ascending exactly as before. The channels of the convolution are
 * independent of each other, so their order is free; the loop below keeps the per-token one
 * anyway, since a reader comparing the two files should not have to prove that. */
static void mamba_layer(mamba_model *m, int l, float *X, int n, float *scratch) {
    int E = m->embed, DI = m->d_inner, DS = m->d_state, DC = m->d_conv, DT = m->dt_rank;
    int XDBW = DT + 2 * DS;
    float *XN   = scratch;                          /* [n, E]      */
    float *XZ   = XN   + (size_t)n * E;             /* [n, 2*DI]   */
    float *XDB  = XZ   + (size_t)n * 2 * DI;        /* [n, XDBW]   */
    float *DTIN = XDB  + (size_t)n * XDBW;          /* [n, DT]     */
    float *DTO  = DTIN + (size_t)n * DT;            /* [n, DI]     */
    float *CONV = DTO  + (size_t)n * DI;            /* [n, DI]     */
    float *Y    = CONV + (size_t)n * DI;            /* [n, DI]     */
    float *OUTV = Y    + (size_t)n * DI;            /* [n, E]      */

    float *cstate = m->conv_state + (size_t)l * DI * (DC - 1);
    float *sstate = m->ssm_state + (size_t)l * DI * DS;

    double pft = pf_mark();
    for (int j = 0; j < n; j++)
        rmsnorm(XN + (size_t)j * E, X + (size_t)j * E, m->layers[l].attn_norm, E, m->rms_eps);
    pf_add(PF_NORM, pft);

    pft = pf_mark();
    qmm(XZ, &m->layers[l].in, XN, n);       /* [2*DI] each: x in the first half, z in the second */
    pf_add(PF_QKV, pft);

    /* Depthwise convolution over time. Each channel keeps the d_conv-1 inputs before this one;
     * the window is those followed by the new value, and the state moves along by one. */
    pft = pf_mark();
    for (int j = 0; j < n; j++) {
        const float *xz = XZ + (size_t)j * 2 * DI;
        float *conv = CONV + (size_t)j * DI;
        for (int i = 0; i < DI; i++) {
            const float *w = m->layers[l].conv_w + (size_t)i * DC;
            float *st = cstate + (size_t)i * (DC - 1);
            float sum = 0.0f;
            for (int k = 0; k < DC - 1; k++) sum += st[k] * w[k];
            sum += xz[i] * w[DC - 1];
            for (int k = 0; k + 1 < DC - 1; k++) st[k] = st[k + 1];
            st[DC - 2] = xz[i];
            sum += m->layers[l].conv_b[i];
            conv[i] = sum / (1.0f + expf(-sum));      /* silu */
        }
    }
    pf_add(PF_ATTN, pft);

    pft = pf_mark();
    qmm(XDB, &m->layers[l].x_proj, CONV, n);   /* dt_rank + 2*d_state: dt, then B, then C */
    /* dt_proj reads only the first dt_rank of each row, and a batched call wants those rows
     * contiguous — which is what this copy is for. It is dt_rank floats a position, 48 on
     * this model against the 1536 the projection then produces from them. */
    for (int j = 0; j < n; j++)
        memcpy(DTIN + (size_t)j * DT, XDB + (size_t)j * XDBW, (size_t)DT * sizeof(float));
    qmm(DTO, &m->layers[l].dt_proj, DTIN, n);
    pf_add(PF_QKV, pft);

    /* The selective scan. Per channel: a time step of its own, a decay per state element,
     * and an answer read out by C. This is the whole of the recurrence, and it stays here. */
    pft = pf_mark();
    for (int j = 0; j < n; j++) {
        const float *xdb  = XDB + (size_t)j * XDBW;
        const float *B = xdb + DT, *C = xdb + DT + DS;
        const float *dt   = DTO + (size_t)j * DI;
        const float *conv = CONV + (size_t)j * DI;
        const float *z    = XZ + (size_t)j * 2 * DI + DI;
        float *y = Y + (size_t)j * DI;
        for (int i = 0; i < DI; i++) {
            float dt_sp = softplus(dt[i] + m->layers[l].dt_b[i]);
            float x_dt = conv[i] * dt_sp;
            const float *A = m->layers[l].A + (size_t)i * DS;
            float *s = sstate + (size_t)i * DS;
            float sum = 0.0f;
            for (int q = 0; q < DS; q++) {
                float st = s[q] * expf(dt_sp * A[q]) + B[q] * x_dt;
                sum += st * C[q];
                s[q] = st;
            }
            y[i] = sum + conv[i] * m->layers[l].D[i];
            y[i] *= z[i] / (1.0f + expf(-z[i]));      /* silu(z) gates the answer */
        }
    }
    pf_add(PF_SILU, pft);

    pft = pf_mark();
    qmm(OUTV, &m->layers[l].out, Y, n);
    pf_add(PF_PROJ, pft);
    pft = pf_mark();
    for (long i = 0; i < (long)n * E; i++) X[i] += OUTV[i];
    pf_add(PF_RESID, pft);
}

static int mamba_forward(void *model, kv_cache *kv, const int *tokens, int n,
                         int pos0, float *logits) {
    mamba_model *m = (mamba_model*)model;
    /* This family keeps no per-position cache — see the note at the top of the
     * file — so it passes kv_dim 0 and the cache shape is not compared. The
     * rest of the contract still applies: token ids and a sane group size are
     * checked here exactly as everywhere else. */
    int rc = nt_check_call(kv, tokens, n, pos0, m->vocab, 0, 0);
    if (rc != NT_OK) return rc;

    int E = m->embed, DI = m->d_inner, DS = m->d_state, DT = m->dt_rank;

    /* A sequence starting over starts from silence. The harness runs a prompt more than once
     * under -r, and without this the second run would answer with the first one still in its
     * state — which reads as a model that has lost its mind rather than as a bug. */
    if (pos0 == 0) {
        memset(m->conv_state, 0,
               (size_t)m->n_layers * DI * (m->d_conv - 1) * sizeof(float));
        memset(m->ssm_state, 0, (size_t)m->n_layers * DI * DS * sizeof(float));
    }

    /* Per position rather than per token now, because a projection wants the whole chunk at
     * once. The harness already caps a prefill chunk at NT_PREFILL_CHUNK, so this is bounded
     * by that and not by the prompt: 1.4 MB at 32 positions on this model, held across calls
     * for the reason written beside the field. */
    size_t per_pos = (size_t)E + 2 * DI + (DT + 2 * DS) + DT + DI + DI + DI + E; /* mamba_layer */
    size_t need    = (per_pos + (size_t)E) * (size_t)n                           /* + X         */
                   + (size_t)E;                                                  /* + head_in   */
    if (n > m->work_pos) {
        float *w = (float*)realloc(m->work, need * sizeof(float));
        if (!w) return NT_E_MEMORY;
        m->work = w; m->work_pos = n;
    }
    float *scratch = m->work;                        /* [per_pos, n], carved up by the layer */
    float *X       = scratch + per_pos * (size_t)n;  /* [n, E] */
    float *head_in = X + (size_t)n * E;              /* [E]    */

    double pft = pf_mark();
    for (int j = 0; j < n; j++) {
        float *x = X + (size_t)j * E;
        if (m->tok_emb.f32) memcpy(x, m->tok_emb.f32 + (long)tokens[j] * E, E * sizeof(float));
        else if (gguf_dequant_row(m->gf, m->emb_ti, (uint64_t)tokens[j], x) != 0)
            memset(x, 0, E * sizeof(float));
    }
    pf_add(PF_EMBED, pft);

    for (int l = 0; l < m->n_layers; l++) mamba_layer(m, l, X, n, scratch);

    /* Only the last position of a chunk is ever sampled from, so the head runs once. */
    if (logits) {
        pft = pf_mark();
        rmsnorm(head_in, X + (size_t)(n - 1) * E, m->out_norm, E, m->rms_eps);
        const wt *head = m->has_output_weight ? &m->out_weight : &m->tok_emb;
        qmv(logits, head, head_in);
        pf_add(PF_HEAD, pft);
    }
    return NT_OK;
}

static const char *const mamba_names[] = { "mamba", NULL };

const nt_arch nt_arch_mamba = {
    .names = mamba_names,
    .load = mamba_load,
    .free = mamba_free,
    .forward = mamba_forward,
};
