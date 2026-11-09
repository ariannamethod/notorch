/* lfm3_omni.c — see lfm3_omni.h. The arithmetic follows LiquidAI's remote code at
 * revision 21d75123 (modeling_lfm3_omni.py, modeling_lfm3_vl.py, modeling_lfm3.py,
 * modeling_liquid_codec.py, feature_extraction_lfm3_omni.py, image_processing_lfm3_vl.py),
 * which is also the oracle these towers are measured against.
 *
 * Three of the four towers are the same kind of stack the text decoder is — an LFM block of
 * either a gated short convolution or attention, then a SwiGLU — differing in width, heads,
 * whether attention looks back only (audio encoder, codec), both ways (depth decoder), how
 * far (a window of 32 encoder steps, 30 codec steps), and whether the LFM3 extras are there
 * (sandwich norms, a sigmoid gate per head). One runner, lfm_run, serves the three. The
 * vision tower convolves in two dimensions and rotates by row and column, so it has its own
 * layer loop but the same attention core.
 *
 * The vision tower runs in raster order. The reference keeps patches in 2x2-block order and
 * scatters them to raster order for each convolution; attention and the MLP do not see the
 * order, so the two are the same computation, and the blocks are gathered once, at the
 * projector, where the reference concatenates them. */
#include "harness/lfm3_omni.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── weights ─────────────────────────────────────────────────────────────────── */

/* A matrix whose rows are `cols` wide, whatever rank the file declares: the patch
 * embedding is 4-D, the downsampler's 1x1 convolutions are 4-D, the depth embeddings 3-D.
 * The packed kernels read F16/F32/Q8_0... rows in place; anything else is expanded. */
static int mat(wt *w, gguf_file *gf, const char *name, int cols) {
    memset(w, 0, sizeof *w);
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0 || cols <= 0 || gf->tensors[ti].n_elements % (uint64_t)cols) {
        fprintf(stderr, "lfm3_omni: %s missing or not a multiple of %d wide\n", name, cols);
        return 0;
    }
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->n_elements / (uint64_t)cols > INT_MAX) return 0;
    w->cols = cols;
    w->rows = (int)(t->n_elements / (uint64_t)cols);
    w->dtype = (int)t->dtype;
    w->q = gf->data + t->offset;
    float probe = 0;
    float *x = calloc((size_t)cols, sizeof(float));
    if (!x) return 0;
    int ok = nt_qmatvec(&probe, w->q, w->dtype, x, 1, cols) == 0;
    free(x);
    if (!ok) { w->q = NULL; w->f32 = gguf_dequant(gf, ti); }
    return w->q || w->f32;
}

static float *vec(gguf_file *gf, const char *name, int n, int *bad) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0 || gf->tensors[ti].n_elements != (uint64_t)n) {
        fprintf(stderr, "lfm3_omni: %s missing or not %d values\n", name, n);
        *bad = 1;
        return NULL;
    }
    float *v = gguf_dequant(gf, ti);
    if (!v) *bad = 1;
    return v;
}

static int key_u32(gguf_file *gf, const char *key, int *out) {
    const gguf_kv *kv = gguf_get_kv(gf, key);
    if (!kv || kv->type != 4) { fprintf(stderr, "lfm3_omni: key %s missing\n", key); return 0; }
    *out = (int)kv->val.u32;
    return 1;
}
static int key_f32(gguf_file *gf, const char *key, float *out) {
    const gguf_kv *kv = gguf_get_kv(gf, key);
    if (!kv || kv->type != 6) { fprintf(stderr, "lfm3_omni: key %s missing\n", key); return 0; }
    *out = kv->val.f32;
    return 1;
}

/* ── small arithmetic ────────────────────────────────────────────────────────── */

static float silu(float v) { return v / (1.0f + expf(-v)); }
static float gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }

static void rms_rows(float *y, const float *x, const float *w, int n, int d, float eps) {
    for (int j = 0; j < n; j++) rmsnorm(y + (size_t)j * d, x + (size_t)j * d, w, d, eps);
}

static void layernorm(float *y, const float *x, const float *w, const float *b, int d, float eps) {
    double mean = 0, var = 0;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= d;
    float inv = (float)(1.0 / sqrt(var + eps));
    for (int i = 0; i < d; i++) y[i] = (float)(x[i] - mean) * inv * w[i] + b[i];
}

static void bias_rows(float *x, const float *b, int n, int d) {
    for (int j = 0; j < n; j++) for (int i = 0; i < d; i++) x[(size_t)j * d + i] += b[i];
}

/* NEOX rotation of one head by an angle per pair: lane i turns with lane i + hd/2. */
static void rotate(float *x, const float *cs, const float *sn, int hd) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float a = x[i], b = x[i + h];
        x[i] = a * cs[i] - b * sn[i];
        x[i + h] = b * cs[i] + a * sn[i];
    }
}

/* ── attention core: roped q [n][H*hd], k/v [n][KVH*hd] -> out [n][H*hd] ───────── */

typedef struct {
    const float *q, *k, *v;
    float *out;
    int n, H, KVH, hd, causal, window, base;   /* rows base..base+n of a group */
} attn_job;

static void attn_items(void *ctx, int i0, int i1) {
    attn_job *a = ctx;
    float *s = malloc((size_t)a->n * sizeof(float));
    if (!s) return;
    float scale = 1.0f / sqrtf((float)a->hd);
    int qd = a->H * a->hd, kd = a->KVH * a->hd, g = a->H / a->KVH;
    for (int it = i0; it < i1; it++) {
        int h = it % a->H, j = it / a->H;
        const float *q = a->q + (size_t)j * qd + (size_t)h * a->hd;
        int lo = 0, hi = a->n;
        if (a->causal) { hi = j + 1; if (a->window > 0 && j - a->window + 1 > 0) lo = j - a->window + 1; }
        int kvh = h / g;
        float mx = -INFINITY;
        for (int t = lo; t < hi; t++) {
            s[t] = dot_f32(q, a->k + (size_t)t * kd + (size_t)kvh * a->hd, a->hd) * scale;
            if (s[t] > mx) mx = s[t];
        }
        float sum = 0;
        for (int t = lo; t < hi; t++) { s[t] = expf(s[t] - mx); sum += s[t]; }
        float *o = a->out + (size_t)j * qd + (size_t)h * a->hd;
        memset(o, 0, (size_t)a->hd * sizeof(float));
        for (int t = lo; t < hi; t++)
            axpy_f32(o, s[t] / sum, a->v + (size_t)t * kd + (size_t)kvh * a->hd, a->hd);
    }
    free(s);
}

static void attention(const float *q, const float *k, const float *v, float *out, int n,
                      int H, int KVH, int hd, int causal, int window) {
    attn_job a = {q, k, v, out, n, H, KVH, hd, causal, window, 0};
    nt_par_for(attn_items, &a, n * H, 64);
}

/* ── the LFM stack shared by the audio encoder, the depth decoder and the codec ── */

typedef struct {
    int conv, gate;
    float *ln1, *ln2, *post1, *post2, *qn, *kn, *dw;
    wt q, k, v, o, g, pw1, pw2, fg, fu, fd;
} lfm_layer;

typedef struct {
    const char *tag;
    int n, width, heads, kv_heads, head_dim, ffn, causal, window, ck;
    float theta, eps;
    lfm_layer *L;
    float *out_norm;
} lfm_stack;

/* Every layer of a stack shares one geometry, because the runner sizes its scratch once:
 * a layer whose FFN or convolution window differs from layer 0's is refused here, not
 * discovered as a write past a buffer. */
static int lfm_load(lfm_stack *s, gguf_file *gf, const char *prefix, const char *out_norm) {
    char nm[160];
    int bad = 0, W = s->width;
    s->ffn = 0; s->ck = 0;
    if (s->n < 1 || s->n > 256 || W < 1 || W > (1 << 20) || s->heads < 1 || s->heads > 1024 || s->kv_heads < 1 ||
        s->kv_heads > s->heads || s->heads % s->kv_heads || s->head_dim < 2 || s->head_dim > 4096 || s->head_dim % 2 ||
        (long long)s->heads * s->head_dim != W || s->window < 0) {
        fprintf(stderr, "lfm3_omni: %s stack geometry unusable (layers %d, width %d, heads %d/%d, head %d)\n",
                prefix, s->n, W, s->heads, s->kv_heads, s->head_dim);
        return 0;
    }
    s->L = calloc((size_t)s->n, sizeof *s->L);
    if (!s->L) return 0;
    for (int l = 0; l < s->n; l++) {
        lfm_layer *y = &s->L[l];
#define N(x) (snprintf(nm, sizeof nm, "%s.blk.%d.%s", prefix, l, x), nm)
        y->ln1 = vec(gf, N("ln1.weight"), W, &bad);
        y->ln2 = vec(gf, N("ln2.weight"), W, &bad);
        if (gguf_find_tensor(gf, N("attn_post_norm.weight")) >= 0) {
            y->post1 = vec(gf, N("attn_post_norm.weight"), W, &bad);
            y->post2 = vec(gf, N("ffn_post_norm.weight"), W, &bad);
        }
        y->conv = gguf_find_tensor(gf, N("conv_pw1.weight")) >= 0;
        if (!mat(&y->fg, gf, N("ffn_gate.weight"), W) || !mat(&y->fu, gf, N("ffn_up.weight"), W)) { bad = 1; break; }
        if (!s->ffn) s->ffn = y->fg.rows;
        if (y->fg.rows != s->ffn || y->fu.rows != s->ffn || !mat(&y->fd, gf, N("ffn_down.weight"), s->ffn) || y->fd.rows != W) {
            fprintf(stderr, "lfm3_omni: %s layer %d FFN is not %d wide like layer 0, or does not return to %d\n",
                    prefix, l, s->ffn, W);
            bad = 1;
        }
        if (y->conv) {
            int ti = gguf_find_tensor(gf, N("conv_dw.weight"));
            int ck = ti >= 0 ? (int)gf->tensors[ti].shape[0] : 0;
            if (!s->ck) s->ck = ck;
            if (ck < 1 || ck > 64 || ck != s->ck) {
                fprintf(stderr, "lfm3_omni: %s layer %d convolution window %d (first conv layer: %d)\n", prefix, l, ck, s->ck);
                bad = 1;
            } else y->dw = vec(gf, nm, W * s->ck, &bad);
            if (!mat(&y->pw1, gf, N("conv_pw1.weight"), W) || !mat(&y->pw2, gf, N("conv_pw2.weight"), W) ||
                y->pw1.rows != 3 * W || y->pw2.rows != W) bad = 1;
        } else {
            int hd = s->head_dim;
            y->qn = vec(gf, N("attn_q_norm.weight"), hd, &bad);
            y->kn = vec(gf, N("attn_k_norm.weight"), hd, &bad);
            if (!mat(&y->q, gf, N("attn_q.weight"), W) || !mat(&y->k, gf, N("attn_k.weight"), W) ||
                !mat(&y->v, gf, N("attn_v.weight"), W) || !mat(&y->o, gf, N("attn_out.weight"), s->heads * hd) ||
                y->q.rows != s->heads * hd || y->k.rows != s->kv_heads * hd || y->v.rows != s->kv_heads * hd || y->o.rows != W)
                bad = 1;
            if (gguf_find_tensor(gf, N("attn_gate.weight")) >= 0) {
                y->gate = 1;
                if (!mat(&y->g, gf, N("attn_gate.weight"), W) || y->g.rows != s->heads) bad = 1;
            }
        }
#undef N
    }
    s->out_norm = out_norm ? vec(gf, out_norm, W, &bad) : NULL;
    if (bad) fprintf(stderr, "lfm3_omni: %s stack does not load\n", prefix);
    return !bad;
}

/* The head width, read off the first attention layer's query norm. */
static int first_head_dim(gguf_file *gf, const char *prefix, int n) {
    char nm[160];
    for (int l = 0; l < n && l < 256; l++) {
        snprintf(nm, sizeof nm, "%s.blk.%d.attn_q_norm.weight", prefix, l);
        int ti = gguf_find_tensor(gf, nm);
        if (ti >= 0) return gf->tensors[ti].n_elements <= 4096 ? (int)gf->tensors[ti].n_elements : 0;
    }
    return 0;
}

static void lfm_free(lfm_stack *s) {
    if (!s->L) return;
    for (int l = 0; l < s->n; l++) {
        lfm_layer *y = &s->L[l];
        free(y->ln1); free(y->ln2); free(y->post1); free(y->post2); free(y->qn); free(y->kn); free(y->dw);
        wt *ws[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->pw1, &y->pw2, &y->fg, &y->fu, &y->fd};
        for (size_t i = 0; i < sizeof ws / sizeof ws[0]; i++) free(ws[i]->f32);
    }
    free(s->L);
    free(s->out_norm);
}

struct lfm3_omni {
    gguf_file *gf;
    lfm3_omni_info info;
    lfm3_omni_probe probe;
    void *probe_user;
    /* vision */
    int v_layers, v_width, v_heads, v_kv_heads, v_ffn;
    float v_theta, v_eps, p_eps;
    wt v_patch[2];
    float *v_embed_norm, *mm_norm_w, *mm_norm_b, *mm0_b, *mm2_b;
    wt mm0, mm2;
    struct { int conv; float *ln1, *ln2, *qn, *kn, *dw; wt q, k, v, o, in, out, fg, fu, fd; } *vl;
    /* audio */
    int a_ch, a_width;
    float *a_w[5], *a_b[5];      /* conv_in, conv_mid, lin_mid, conv_end, lin_end */
    wt a_out, a_lin_mid, a_lin_end;
    float *a_out_b, *mel, *am_norm_w, *am_norm_b, *am1_b, *am3_b, a_floor, a_offset, a_div;
    wt am1, am3;
    int n_fft;
    lfm_stack enc;
    /* speech out */
    lfm_stack depth, codec;
    wt d_proj, d_embd, d_out;
    float *d_proj_b, *d_mask, *d_head_norm;
    float *fsq, *q_out_w, *q_out_b, *c_in_b, *c_lin_b, *window;
    wt c_in, c_lin;
    int n_lv, c_fs, c_nfft, c_hop;
};

const lfm3_omni_info *lfm3_omni_get_info(const lfm3_omni *o) { return &o->info; }
void lfm3_omni_set_probe(lfm3_omni *o, lfm3_omni_probe fn, void *user) { o->probe = fn; o->probe_user = user; }
static void probe(const lfm3_omni *o, const char *tag, int layer, const float *x, int n, int w) {
    if (o->probe) o->probe(o->probe_user, tag, layer, x, n, w);
}

/* x [n][W] through one LFM stack, positions 0..n-1, final norm included. */
static int lfm_run(const lfm3_omni *om, const lfm_stack *s, float *x, int n) {
    int W = s->width, hd = s->head_dim, F = s->ffn, H = s->heads, KVH = s->kv_heads;
    int qd = H * hd, kd = KVH * hd, rc = -1;
    float *xn = malloc((size_t)n * W * sizeof(float)), *y = malloc((size_t)n * W * sizeof(float));
    size_t aw = (size_t)3 * W > (size_t)F ? (size_t)3 * W : (size_t)F;
    size_t bw = (size_t)F > (size_t)qd ? (size_t)F : (size_t)qd;
    if (bw < (size_t)W) bw = (size_t)W;
    float *a = malloc((size_t)n * aw * sizeof(float));
    float *b = malloc((size_t)n * bw * sizeof(float));
    float *q = malloc((size_t)n * qd * sizeof(float)), *k = malloc((size_t)n * kd * sizeof(float));
    float *v = malloc((size_t)n * kd * sizeof(float)), *g = malloc((size_t)n * H * sizeof(float));
    float *cs = malloc((size_t)n * hd / 2 * sizeof(float)), *sn = malloc((size_t)n * hd / 2 * sizeof(float));
    if (!xn || !y || !a || !b || !q || !k || !v || !g || !cs || !sn) goto out;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < hd / 2; i++) {
            float inv = (float)(1.0 / pow((double)s->theta, 2.0 * i / hd));
            float ang = (float)j * inv;
            cs[(size_t)j * hd / 2 + i] = cosf(ang);
            sn[(size_t)j * hd / 2 + i] = sinf(ang);
        }
    for (int l = 0; l < s->n; l++) {
        const lfm_layer *L = &s->L[l];
        rms_rows(xn, x, L->ln1, n, W, s->eps);
        if (L->conv) {
            int K = s->ck;
            qmm(a, &L->pw1, xn, n);                         /* [n][B | C | x] */
            for (int j = 0; j < n; j++) {
                float *r = a + (size_t)j * 3 * W;
                for (int c = 0; c < W; c++) r[c] *= r[2 * W + c];   /* B*x in place of B */
            }
            for (int j = 0; j < n; j++)
                for (int c = 0; c < W; c++) {
                    float acc = 0;
                    for (int t = 0; t < K; t++) {
                        int src = j - (K - 1) + t;
                        if (src >= 0) acc += L->dw[(size_t)c * K + t] * a[(size_t)src * 3 * W + c];
                    }
                    b[(size_t)j * W + c] = acc * a[(size_t)j * 3 * W + W + c];
                }
            qmm(y, &L->pw2, b, n);
        } else {
            qmm(q, &L->q, xn, n); qmm(k, &L->k, xn, n); qmm(v, &L->v, xn, n);
            for (int j = 0; j < n; j++) {
                for (int h = 0; h < H; h++) {
                    float *qh = q + (size_t)j * qd + (size_t)h * hd;
                    rmsnorm(qh, qh, L->qn, hd, s->eps);
                    rotate(qh, cs + (size_t)j * hd / 2, sn + (size_t)j * hd / 2, hd);
                }
                for (int h = 0; h < KVH; h++) {
                    float *kh = k + (size_t)j * kd + (size_t)h * hd;
                    rmsnorm(kh, kh, L->kn, hd, s->eps);
                    rotate(kh, cs + (size_t)j * hd / 2, sn + (size_t)j * hd / 2, hd);
                }
            }
            attention(q, k, v, b, n, H, KVH, hd, s->causal, s->window);
            if (L->gate) {
                qmm(g, &L->g, xn, n);
                for (int j = 0; j < n; j++)
                    for (int h = 0; h < H; h++) {
                        float gv = 1.0f / (1.0f + expf(-g[(size_t)j * H + h]));
                        float *bh = b + (size_t)j * qd + (size_t)h * hd;
                        for (int i = 0; i < hd; i++) bh[i] *= gv;
                    }
            }
            qmm(y, &L->o, b, n);
        }
        if (L->post1) rms_rows(y, y, L->post1, n, W, s->eps);
        for (size_t i = 0; i < (size_t)n * W; i++) x[i] += y[i];
        rms_rows(xn, x, L->ln2, n, W, s->eps);
        qmm(a, &L->fg, xn, n); qmm(b, &L->fu, xn, n);
        for (size_t i = 0; i < (size_t)n * F; i++) a[i] = silu(a[i]) * b[i];
        qmm(y, &L->fd, a, n);
        if (L->post2) rms_rows(y, y, L->post2, n, W, s->eps);
        for (size_t i = 0; i < (size_t)n * W; i++) x[i] += y[i];
        probe(om, s->tag, l, x, n, W);
    }
    if (s->out_norm) rms_rows(x, x, s->out_norm, n, W, s->eps);
    rc = 0;
out:
    free(xn); free(y); free(a); free(b); free(q); free(k); free(v); free(g); free(cs); free(sn);
    return rc;
}

/* ── load ────────────────────────────────────────────────────────────────────── */

void lfm3_omni_free(lfm3_omni *o) {
    if (!o) return;
    if (o->vl) {
        for (int l = 0; l < o->v_layers; l++) {
            free(o->vl[l].ln1); free(o->vl[l].ln2); free(o->vl[l].qn); free(o->vl[l].kn); free(o->vl[l].dw);
            wt *ws[] = {&o->vl[l].q, &o->vl[l].k, &o->vl[l].v, &o->vl[l].o, &o->vl[l].in, &o->vl[l].out,
                        &o->vl[l].fg, &o->vl[l].fu, &o->vl[l].fd};
            for (size_t i = 0; i < sizeof ws / sizeof ws[0]; i++) free(ws[i]->f32);
        }
        free(o->vl);
    }
    wt *ws[] = {&o->v_patch[0], &o->v_patch[1], &o->mm0, &o->mm2, &o->a_out, &o->a_lin_mid, &o->a_lin_end,
                &o->am1, &o->am3, &o->d_proj, &o->d_embd, &o->d_out, &o->c_in, &o->c_lin};
    for (size_t i = 0; i < sizeof ws / sizeof ws[0]; i++) free(ws[i]->f32);
    float *fs[] = {o->v_embed_norm, o->mm_norm_w, o->mm_norm_b, o->mm0_b, o->mm2_b, o->a_out_b, o->mel,
                   o->am_norm_w, o->am_norm_b, o->am1_b, o->am3_b, o->d_proj_b, o->d_mask, o->d_head_norm,
                   o->fsq, o->q_out_w, o->q_out_b, o->c_in_b, o->c_lin_b, o->window};
    for (size_t i = 0; i < sizeof fs / sizeof fs[0]; i++) free(fs[i]);
    for (int i = 0; i < 5; i++) { free(o->a_w[i]); free(o->a_b[i]); }
    lfm_free(&o->enc); lfm_free(&o->depth); lfm_free(&o->codec);
    free(o);
}

lfm3_omni *lfm3_omni_load(gguf_file *gf) {
    lfm3_omni *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->gf = gf;
    lfm3_omni_info *I = &o->info;
    int ok = 1, bad = 0;
    ok &= key_u32(gf, "lfm3.embedding_length", &I->width);
    ok &= key_u32(gf, "lfm3_omni.vision.layers", &o->v_layers) && key_u32(gf, "lfm3_omni.vision.heads", &o->v_heads) &&
          key_u32(gf, "lfm3_omni.vision.kv_heads", &o->v_kv_heads) && key_u32(gf, "lfm3_omni.vision.patch", &I->patch) &&
          key_u32(gf, "lfm3_omni.vision.merge", &I->merge) && key_u32(gf, "lfm3_omni.vision.min_pixels", &I->min_pixels) &&
          key_u32(gf, "lfm3_omni.vision.max_pixels", &I->max_pixels) && key_f32(gf, "lfm3_omni.vision.theta", &o->v_theta) &&
          key_f32(gf, "lfm3_omni.vision.eps", &o->v_eps) && key_f32(gf, "lfm3_omni.vision.projector_eps", &o->p_eps);
    ok &= key_u32(gf, "lfm3_omni.audio.layers", &o->enc.n) && key_u32(gf, "lfm3_omni.audio.heads", &o->enc.heads) &&
          key_u32(gf, "lfm3_omni.audio.kv_heads", &o->enc.kv_heads) && key_u32(gf, "lfm3_omni.audio.window", &o->enc.window) &&
          key_f32(gf, "lfm3_omni.audio.theta", &o->enc.theta) && key_f32(gf, "lfm3_omni.audio.eps", &o->enc.eps) &&
          key_u32(gf, "lfm3_omni.audio.sample_rate", &I->sample_rate) && key_u32(gf, "lfm3_omni.audio.n_fft", &o->n_fft) &&
          key_u32(gf, "lfm3_omni.audio.hop", &I->hop) && key_u32(gf, "lfm3_omni.audio.stride", &I->stride) &&
          key_f32(gf, "lfm3_omni.audio.log_floor", &o->a_floor) && key_f32(gf, "lfm3_omni.audio.log_offset", &o->a_offset) &&
          key_f32(gf, "lfm3_omni.audio.log_divisor", &o->a_div);
    ok &= key_u32(gf, "lfm3_omni.depth.layers", &o->depth.n) && key_u32(gf, "lfm3_omni.depth.heads", &o->depth.heads) &&
          key_u32(gf, "lfm3_omni.depth.kv_heads", &o->depth.kv_heads) && key_f32(gf, "lfm3_omni.depth.theta", &o->depth.theta) &&
          key_f32(gf, "lfm3_omni.depth.eps", &o->depth.eps) && key_u32(gf, "lfm3_omni.depth.codebooks", &I->n_codebooks) &&
          key_u32(gf, "lfm3_omni.depth.codebook_size", &I->codebook_size) && key_u32(gf, "lfm3_omni.depth.eos_code", &I->eos_code);
    ok &= key_u32(gf, "lfm3_omni.gen.interleave_text", &I->interleave_text) &&
          key_u32(gf, "lfm3_omni.gen.interleave_audio", &I->interleave_audio) &&
          key_f32(gf, "lfm3_omni.gen.audio_temperature", &I->audio_temperature) &&
          key_f32(gf, "lfm3_omni.gen.audio_repetition_penalty", &I->audio_rep_penalty) &&
          key_u32(gf, "lfm3_omni.gen.audio_repetition_window", &I->audio_rep_window);
    ok &= key_u32(gf, "lfm3_omni.codec.layers", &o->codec.n) && key_u32(gf, "lfm3_omni.codec.heads", &o->codec.heads) &&
          key_u32(gf, "lfm3_omni.codec.kv_heads", &o->codec.kv_heads) && key_u32(gf, "lfm3_omni.codec.window", &o->codec.window) &&
          key_f32(gf, "lfm3_omni.codec.theta", &o->codec.theta) && key_f32(gf, "lfm3_omni.codec.eps", &o->codec.eps) &&
          key_u32(gf, "lfm3_omni.codec.frame_stack", &o->c_fs) && key_u32(gf, "lfm3_omni.codec.n_fft", &o->c_nfft) &&
          key_u32(gf, "lfm3_omni.codec.hop", &o->c_hop) && key_u32(gf, "lfm3_omni.codec.sample_rate", &I->codec_rate);
    const struct { const char *k; int *v; } toks[] = {
        {"lfm3_omni.token.image", &I->tok_image}, {"lfm3_omni.token.video", &I->tok_video},
        {"lfm3_omni.token.audio", &I->tok_audio}, {"lfm3_omni.token.audio_out", &I->tok_audio_out},
        {"lfm3_omni.token.audio_start", &I->tok_audio_start}, {"lfm3_omni.token.mixed_start", &I->tok_mixed_start},
        {"lfm3_omni.token.text_end", &I->tok_text_end}, {"lfm3_omni.token.media_start", &I->tok_media_start},
        {"lfm3_omni.token.media_end", &I->tok_media_end}, {"lfm3_omni.token.im_start", &I->tok_im_start},
        {"lfm3_omni.token.im_end", &I->tok_im_end}};
    for (size_t i = 0; i < sizeof toks / sizeof toks[0]; i++) ok &= key_u32(gf, toks[i].k, toks[i].v);
    if (!ok) { lfm3_omni_free(o); return NULL; }
    {
        int n = 0;
        int32_t *s = gguf_read_i32_array(gf->path, "lfm3_omni.depth.schedule", &n);
        int sum = 0;
        for (int i = 0; s && i < n && i < 16; i++) { I->schedule[i] = s[i]; sum += s[i]; }
        I->n_schedule = n;
        free(s);
        if (!s || n < 1 || n > 16 || sum != I->n_codebooks) {
            fprintf(stderr, "lfm3_omni: depth.schedule missing or not covering the codebooks\n");
            lfm3_omni_free(o); return NULL;
        }
    }
    if (I->patch != 16 || I->merge != 2 || I->n_codebooks < 1 || I->n_codebooks > 32 || o->n_fft < 16 || o->n_fft > 4096) {
        fprintf(stderr, "lfm3_omni: geometry outside what this file implements\n");
        lfm3_omni_free(o); return NULL;
    }

    /* vision */
    char nm[160];
    int pd = 3 * I->patch * I->patch;
    if (!mat(&o->v_patch[0], gf, "v.patch_embd.weight", pd) || !mat(&o->v_patch[1], gf, "v.patch_embd.weight.1", pd)) bad = 1;
    o->v_width = o->v_patch[0].rows;
    int VW = o->v_width, vhd = o->v_heads > 0 ? VW / o->v_heads : 0;
    /* the head geometry every attention layer and the rope table are sized from */
    if (bad || VW < 1 || VW > (1 << 20) || o->v_patch[1].rows != VW || o->v_layers < 1 || o->v_layers > 256 ||
        o->v_heads < 1 || o->v_heads > 1024 ||
        o->v_kv_heads < 1 || o->v_kv_heads > o->v_heads || o->v_heads % o->v_kv_heads || vhd * o->v_heads != VW || vhd % 4) {
        fprintf(stderr, "lfm3_omni: vision geometry unusable (width %d, heads %d/%d, layers %d)\n",
                VW, o->v_heads, o->v_kv_heads, o->v_layers);
        lfm3_omni_free(o);
        return NULL;
    }
    o->v_embed_norm = vec(gf, "v.embedding_norm.weight", VW, &bad);
    o->vl = calloc((size_t)o->v_layers, sizeof *o->vl);
    if (!o->vl) bad = 1;
    for (int l = 0; !bad && l < o->v_layers; l++) {
#define V(x) (snprintf(nm, sizeof nm, "v.blk.%d.%s", l, x), nm)
        o->vl[l].ln1 = vec(gf, V("ln1.weight"), VW, &bad);
        o->vl[l].ln2 = vec(gf, V("ln2.weight"), VW, &bad);
        if (!mat(&o->vl[l].fg, gf, V("ffn_gate.weight"), VW) || !mat(&o->vl[l].fu, gf, V("ffn_up.weight"), VW)) { bad = 1; break; }
        if (l == 0) o->v_ffn = o->vl[l].fg.rows;
        if (o->vl[l].fg.rows != o->v_ffn || o->vl[l].fu.rows != o->v_ffn ||
            !mat(&o->vl[l].fd, gf, V("ffn_down.weight"), o->v_ffn) || o->vl[l].fd.rows != VW) {
            fprintf(stderr, "lfm3_omni: vision layer %d FFN is not %d wide like layer 0, or does not return to %d\n", l, o->v_ffn, VW);
            bad = 1;
        }
        o->vl[l].conv = gguf_find_tensor(gf, V("conv2d_in_proj.weight")) >= 0;
        if (o->vl[l].conv) {
            if (!mat(&o->vl[l].in, gf, V("conv2d_in_proj.weight"), VW) || !mat(&o->vl[l].out, gf, V("conv2d_out_proj.weight"), VW) ||
                o->vl[l].in.rows != 3 * VW || o->vl[l].out.rows != VW) {
                fprintf(stderr, "lfm3_omni: vision layer %d convolution projections are not %d -> 3x%d -> %d\n", l, VW, VW, VW);
                bad = 1;
            }
            o->vl[l].dw = vec(gf, V("conv2d_spatial.weight"), 9 * VW, &bad);
        } else {
            o->vl[l].qn = vec(gf, V("attn_q_norm.weight"), vhd, &bad);
            o->vl[l].kn = vec(gf, V("attn_k_norm.weight"), vhd, &bad);
            if (!mat(&o->vl[l].q, gf, V("attn_q.weight"), VW) || !mat(&o->vl[l].k, gf, V("attn_k.weight"), VW) ||
                !mat(&o->vl[l].v, gf, V("attn_v.weight"), VW) || !mat(&o->vl[l].o, gf, V("attn_out.weight"), VW) ||
                o->vl[l].q.rows != VW || o->vl[l].k.rows != o->v_kv_heads * vhd ||
                o->vl[l].v.rows != o->v_kv_heads * vhd || o->vl[l].o.rows != VW) {
                fprintf(stderr, "lfm3_omni: vision layer %d attention is not q %d, k/v %d, out %d rows\n",
                        l, VW, o->v_kv_heads * vhd, VW);
                bad = 1;
            }
        }
#undef V
    }
    o->mm_norm_w = vec(gf, "mm.input_norm.weight", VW, &bad);
    o->mm_norm_b = vec(gf, "mm.input_norm.bias", VW, &bad);
    if (!mat(&o->mm0, gf, "mm.0.weight", 4 * VW) || !mat(&o->mm2, gf, "mm.2.weight", o->mm0.rows) || o->mm2.rows != I->width) bad = 1;
    o->mm0_b = vec(gf, "mm.0.bias", o->mm0.rows, &bad);
    o->mm2_b = vec(gf, "mm.2.bias", I->width, &bad);

    /* audio */
    int ti = gguf_find_tensor(gf, "a.conv1d.0.weight");
    o->a_ch = ti >= 0 ? (int)(gf->tensors[ti].n_elements / 9) : 0;
    int C = o->a_ch;
    const char *cw[5] = {"a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6"};
    const int cn[5] = {9, 9, 0, 9, 0};
    for (int i = 0; i < 5; i++) {
        snprintf(nm, sizeof nm, "%s.bias", cw[i]);
        o->a_b[i] = vec(gf, nm, C, &bad);
        if (cn[i]) { snprintf(nm, sizeof nm, "%s.weight", cw[i]); o->a_w[i] = vec(gf, nm, cn[i] * C, &bad); }
    }
    if (C < 1 || C > 65536 || !mat(&o->a_lin_mid, gf, "a.conv1d.3.weight", C) || !mat(&o->a_lin_end, gf, "a.conv1d.6.weight", C) ||
        o->a_lin_mid.rows != C || o->a_lin_end.rows != C) {
        fprintf(stderr, "lfm3_omni: audio downsampler 1x1 convolutions are not %d x %d\n", C, C);
        bad = 1;
    }
    ti = gguf_find_tensor(gf, "a.mel_filters");
    I->n_mel = ti >= 0 && gf->tensors[ti].ndim == 2 ? (int)gf->tensors[ti].shape[1] : 0;
    int nbins = o->n_fft / 2 + 1;
    /* three stride-2 convolutions: eight mel frames per row, n_mel / 8 bins left */
    if (I->n_mel < 8 || I->n_mel > 1024 || I->n_mel % 8 || I->stride != 8 || I->hop < 1 || I->hop > o->n_fft ||
        o->n_fft % 4 || (int)gf->tensors[ti].shape[0] != nbins) {
        fprintf(stderr, "lfm3_omni: audio front end unusable (mel %d, n_fft %d, hop %d, stride %d)\n",
                I->n_mel, o->n_fft, I->hop, I->stride);
        lfm3_omni_free(o);
        return NULL;
    }
    o->mel = vec(gf, "a.mel_filters", I->n_mel * nbins, &bad);
    if (!mat(&o->a_out, gf, "a.pre_encode.out.weight", C * I->n_mel / 8)) bad = 1;
    o->a_width = o->a_out.rows;
    o->a_out_b = vec(gf, "a.pre_encode.out.bias", o->a_width, &bad);
    o->enc.tag = "audio";
    o->enc.width = o->a_width;
    o->enc.causal = 1;
    o->enc.head_dim = first_head_dim(gf, "a", o->enc.n);
    if (o->enc.head_dim * o->enc.heads != o->a_width || !lfm_load(&o->enc, gf, "a", "a.post_ln.weight")) bad = 1;
    o->am_norm_w = vec(gf, "mm.a.mlp.0.weight", o->a_width, &bad);
    o->am_norm_b = vec(gf, "mm.a.mlp.0.bias", o->a_width, &bad);
    if (!mat(&o->am1, gf, "mm.a.mlp.1.weight", o->a_width) || !mat(&o->am3, gf, "mm.a.mlp.3.weight", o->am1.rows) ||
        o->am3.rows != I->width) bad = 1;
    o->am1_b = vec(gf, "mm.a.mlp.1.bias", o->am1.rows, &bad);
    o->am3_b = vec(gf, "mm.a.mlp.3.bias", I->width, &bad);

    /* depth decoder */
    int NC = I->n_codebooks, CS = I->codebook_size;
    if (CS < 2 || CS > 65536 || I->eos_code != CS - 1 || !mat(&o->d_proj, gf, "a.gen.code.proj_in.weight", I->width) ||
        o->d_proj.rows % NC) {
        fprintf(stderr, "lfm3_omni: depth decoder unusable (codebooks %d of %d codes, eos %d)\n", NC, CS, I->eos_code);
        bad = 1;
    }
    o->depth.width = o->d_proj.rows / NC;
    int DW = o->depth.width;
    o->d_proj_b = vec(gf, "a.gen.code.proj_in.bias", NC * DW, &bad);
    o->d_mask = vec(gf, "a.gen.code.mask_embd", DW, &bad);
    o->d_head_norm = vec(gf, "a.gen.code.head_norm.weight", NC * DW, &bad);
    if (!mat(&o->d_embd, gf, "a.gen.code.embd.weight", DW) || o->d_embd.rows != NC * CS) bad = 1;
    if (!mat(&o->d_out, gf, "a.gen.code.out_embd.weight", I->width) || o->d_out.rows != NC * CS) bad = 1;
    o->depth.tag = "depth";
    o->depth.causal = 0;
    o->depth.head_dim = first_head_dim(gf, "a.gen.code", o->depth.n);
    if (!lfm_load(&o->depth, gf, "a.gen.code", "a.gen.code.output_norm.weight")) bad = 1;

    /* codec: the iSTFT reads frames of n_fft at a stride of hop and divides by the summed
     * squared window, which every trimmed sample must have at least two frames under */
    if (o->c_nfft < 4 || o->c_nfft > 65536 || o->c_nfft % 2 || o->c_hop < 1 || 2 * o->c_hop > o->c_nfft ||
        o->c_fs < 1 || o->c_fs > 64) {
        fprintf(stderr, "lfm3_omni: codec geometry unusable (n_fft %d, hop %d, frame stack %d)\n", o->c_nfft, o->c_hop, o->c_fs);
        lfm3_omni_free(o);
        return NULL;
    }
    ti = gguf_find_tensor(gf, "a.gen.wav.quant.first.codebook.weight");
    o->n_lv = ti >= 0 && gf->tensors[ti].ndim == 2 ? (int)gf->tensors[ti].shape[0] : 0;
    if (o->n_lv < 1 || o->n_lv > 16) bad = 1;
    else o->fsq = vec(gf, "a.gen.wav.quant.first.codebook.weight", o->n_lv * (CS - 1), &bad);
    ti = gguf_find_tensor(gf, "a.gen.wav.quant.first.out_proj.weight");
    int qdim = ti >= 0 && gf->tensors[ti].ndim == 2 ? (int)gf->tensors[ti].shape[1] : 0;
    if (qdim < 1 || qdim > 65536) bad = 1;
    else {
        o->q_out_w = vec(gf, "a.gen.wav.quant.first.out_proj.weight", qdim * NC * o->n_lv, &bad);
        o->q_out_b = vec(gf, "a.gen.wav.quant.first.out_proj.bias", qdim, &bad);
        if (!mat(&o->c_in, gf, "a.gen.wav.tfm.in_proj.weight", qdim)) bad = 1;
    }
    o->codec.width = o->c_in.rows;
    o->c_in_b = vec(gf, "a.gen.wav.tfm.in_proj.bias", o->codec.width, &bad);
    o->codec.tag = "codec";
    o->codec.causal = 1;
    o->codec.head_dim = first_head_dim(gf, "a.gen.wav.tfm", o->codec.n);
    if (!lfm_load(&o->codec, gf, "a.gen.wav.tfm", "a.gen.wav.tfm.output_norm.weight")) bad = 1;
    if (!mat(&o->c_lin, gf, "a.gen.wav.tfm.out_proj.weight", o->codec.width) || o->c_lin.rows != 2 * (o->c_nfft / 2 + 1)) bad = 1;
    o->c_lin_b = vec(gf, "a.gen.wav.tfm.out_proj.bias", o->c_lin.rows, &bad);
    o->window = vec(gf, "a.gen.wav.istft.window", o->c_nfft, &bad);
    I->codec_samples_per_frame = o->c_fs * o->c_hop;

    if (bad) { fprintf(stderr, "lfm3_omni: towers do not load\n"); lfm3_omni_free(o); return NULL; }
    fprintf(stderr, "lfm3_omni: vision %d x %d (%d heads), audio %d x %d, depth %d x %d (%d codebooks), codec %d x %d\n",
            o->v_layers, VW, o->v_heads, o->enc.n, o->a_width, o->depth.n, DW, NC, o->codec.n, o->codec.width);
    return o;
}

/* ── vision ──────────────────────────────────────────────────────────────────── */

int lfm3_omni_image_size(const lfm3_omni *o, int h, int w, int *out_h, int *out_w) {
    const lfm3_omni_info *I = &o->info;
    int f = I->patch * I->merge;
    if (h < 1 || w < 1 || (double)(h > w ? h : w) / (h > w ? w : h) > 200) return -1;
    /* Python's round: half to even */
    double hb = nearbyint((double)h / f) * f, wb = nearbyint((double)w / f) * f;
    if (hb * wb > I->max_pixels) {
        double beta = sqrt((double)h * w / I->max_pixels);
        hb = floor(h / beta / f) * f; if (hb < f) hb = f;
        wb = floor(w / beta / f) * f; if (wb < f) wb = f;
    } else if (hb * wb < I->min_pixels) {
        double beta = sqrt((double)I->min_pixels / ((double)h * w));
        hb = ceil(h * beta / f) * f;
        wb = ceil(w * beta / f) * f;
    }
    *out_h = (int)hb; *out_w = (int)wb;
    return 0;
}

/* Antialiased bicubic (a = -0.5) on bytes, separable, rounding back to bytes after each
 * pass — the uint8 path torchvision's resize takes, which is PIL's. */
static double cubic(double x) {
    const double a = -0.5;
    x = fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}
static int resample_axis(const uint8_t *src, uint8_t *dst, int n_in, int n_out, int lines, int stride_in,
                         int stride_line_in, int stride_out, int stride_line_out) {
    double scale = (double)n_in / n_out, support = 2.0 * (scale > 1 ? scale : 1), fs = scale > 1 ? scale : 1;
    int ksize = (int)ceil(support) * 2 + 1;
    double *k = malloc((size_t)ksize * sizeof(double));
    if (!k) return -1;
    for (int i = 0; i < n_out; i++) {
        double center = (i + 0.5) * scale;
        int lo = (int)(center - support + 0.5); if (lo < 0) lo = 0;
        int hi = (int)(center + support + 0.5); if (hi > n_in) hi = n_in;
        double ws = 0;
        for (int x = lo; x < hi; x++) { k[x - lo] = cubic((x - center + 0.5) / fs); ws += k[x - lo]; }
        for (int x = lo; x < hi; x++) k[x - lo] = ws != 0 ? k[x - lo] / ws : 0;
        for (int l = 0; l < lines; l++) {
            double acc = 0;
            for (int x = lo; x < hi; x++) acc += k[x - lo] * src[(size_t)l * stride_line_in + (size_t)x * stride_in];
            long v = lround(acc);
            dst[(size_t)l * stride_line_out + (size_t)i * stride_out] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    free(k);
    return 0;
}

int lfm3_omni_image_prepare(const lfm3_omni *o, const uint8_t *rgb, int h, int w, float **out, int *oh, int *ow) {
    int H, W;
    if (lfm3_omni_image_size(o, h, w, &H, &W)) return -1;
    uint8_t *tmp = malloc((size_t)h * W * 3), *res = malloc((size_t)H * W * 3);
    float *px = malloc((size_t)3 * H * W * sizeof(float));
    int rc = -1;
    if (!tmp || !res || !px) goto done;
    /* horizontal pass first, then vertical, per channel (interleaved RGB bytes) */
    for (int c = 0; c < 3; c++) {
        if (W == w) { for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) tmp[((size_t)y * W + x) * 3 + c] = rgb[((size_t)y * w + x) * 3 + c]; }
        else if (resample_axis(rgb + c, tmp + c, w, W, h, 3, w * 3, 3, W * 3)) goto done;
        if (H == h) { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) res[((size_t)y * W + x) * 3 + c] = tmp[((size_t)y * W + x) * 3 + c]; }
        else if (resample_axis(tmp + c, res + c, h, H, W, W * 3, 3, W * 3, 3)) goto done;
    }
    for (int c = 0; c < 3; c++)
        for (size_t i = 0; i < (size_t)H * W; i++)
            px[(size_t)c * H * W + i] = (res[i * 3 + c] * (float)(1.0 / 255.0) - 0.5f) / 0.5f;
    *out = px; px = NULL; *oh = H; *ow = W;
    rc = 0;
done:
    free(tmp); free(res); free(px);
    return rc;
}

int lfm3_omni_vision_encode(lfm3_omni *o, const float *const (*frames)[2], int n_groups, int h, int w, float *out) {
    const lfm3_omni_info *I = &o->info;
    int P = I->patch, gh = h / P, gw = w / P, N = gh * gw, VW = o->v_width;
    int Hh = o->v_heads, KVH = o->v_kv_heads, hd = VW / Hh, F = o->v_ffn;
    if (n_groups < 1 || h % (P * I->merge) || w % (P * I->merge) || !N) return -1;
    size_t NW = (size_t)N * VW;
    float *pt = malloc((size_t)N * 3 * P * P * sizeof(float)), *x = malloc(NW * sizeof(float));
    float *xn = malloc(NW * sizeof(float)), *y = malloc(NW * sizeof(float)), *t2 = malloc(NW * sizeof(float));
    float *a = malloc((size_t)N * (3 * VW > F ? 3 * VW : F) * sizeof(float)), *b = malloc((size_t)N * F * sizeof(float));
    float *q = malloc(NW * sizeof(float)), *k = malloc(NW * sizeof(float)), *v = malloc(NW * sizeof(float));
    float *cs = malloc((size_t)N * hd / 2 * sizeof(float)), *sn = malloc((size_t)N * hd / 2 * sizeof(float));
    float *m = malloc((size_t)(N / 4) * 4 * VW * sizeof(float)), *mh = malloc((size_t)(N / 4) * o->mm0.rows * sizeof(float));
    int rc = -1;
    if (!pt || !x || !xn || !y || !t2 || !a || !b || !q || !k || !v || !cs || !sn || !m || !mh) goto done;
    /* axial rope: the first hd/4 pairs turn with the row, the next hd/4 with the column */
    for (int r = 0; r < gh; r++)
        for (int c = 0; c < gw; c++)
            for (int i = 0; i < hd / 2; i++) {
                int j = i % (hd / 4);
                float inv = (float)(1.0 / pow((double)o->v_theta, (2.0 * j) / (hd / 2)));
                float ang = (float)(i < hd / 4 ? r : c) * inv;
                size_t at = ((size_t)r * gw + c) * hd / 2 + i;
                cs[at] = cosf(ang); sn[at] = sinf(ang);
            }
    for (int gi = 0; gi < n_groups; gi++) {
        memset(x, 0, NW * sizeof(float));
        for (int tt = 0; tt < 2; tt++) {
            const float *fr = frames[gi][tt];
            for (int r = 0; r < gh; r++)
                for (int c = 0; c < gw; c++) {
                    float *p = pt + ((size_t)r * gw + c) * 3 * P * P;
                    for (int ch = 0; ch < 3; ch++)
                        for (int py = 0; py < P; py++)
                            memcpy(p + ((size_t)ch * P + py) * P, fr + ((size_t)ch * h + r * P + py) * w + c * P, P * sizeof(float));
                }
            qmm(y, &o->v_patch[tt], pt, N);
            for (size_t i = 0; i < NW; i++) x[i] += y[i];
        }
        rms_rows(x, x, o->v_embed_norm, N, VW, o->v_eps);
        probe(o, "vision.embed", -1, x, N, VW);
        for (int l = 0; l < o->v_layers; l++) {
            rms_rows(xn, x, o->vl[l].ln1, N, VW, o->v_eps);
            if (o->vl[l].conv) {
                qmm(a, &o->vl[l].in, xn, N);                 /* [N][B | C | x] */
                for (int j = 0; j < N; j++)
                    for (int ch = 0; ch < VW; ch++) t2[(size_t)j * VW + ch] = a[(size_t)j * 3 * VW + ch] * a[(size_t)j * 3 * VW + 2 * VW + ch];
                const float *dw = o->vl[l].dw;
                for (int r = 0; r < gh; r++)
                    for (int c = 0; c < gw; c++)
                        for (int ch = 0; ch < VW; ch++) {
                            float acc = 0;
                            for (int dy = -1; dy <= 1; dy++)
                                for (int dx = -1; dx <= 1; dx++) {
                                    int rr = r + dy, cc = c + dx;
                                    if (rr < 0 || rr >= gh || cc < 0 || cc >= gw) continue;
                                    acc += dw[(size_t)ch * 9 + (dy + 1) * 3 + dx + 1] * t2[((size_t)rr * gw + cc) * VW + ch];
                                }
                            size_t j = (size_t)r * gw + c;
                            b[j * VW + ch] = acc * a[j * 3 * VW + VW + ch];
                        }
                qmm(y, &o->vl[l].out, b, N);
            } else {
                qmm(q, &o->vl[l].q, xn, N); qmm(k, &o->vl[l].k, xn, N); qmm(v, &o->vl[l].v, xn, N);
                int kd = KVH * hd;
                for (int j = 0; j < N; j++) {
                    for (int hh = 0; hh < Hh; hh++) {
                        float *qh = q + (size_t)j * VW + (size_t)hh * hd;
                        rmsnorm(qh, qh, o->vl[l].qn, hd, o->v_eps);
                        rotate(qh, cs + (size_t)j * hd / 2, sn + (size_t)j * hd / 2, hd);
                    }
                    for (int hh = 0; hh < KVH; hh++) {
                        float *kh = k + (size_t)j * kd + (size_t)hh * hd;
                        rmsnorm(kh, kh, o->vl[l].kn, hd, o->v_eps);
                        rotate(kh, cs + (size_t)j * hd / 2, sn + (size_t)j * hd / 2, hd);
                    }
                }
                attention(q, k, v, t2, N, Hh, KVH, hd, 0, 0);
                qmm(y, &o->vl[l].o, t2, N);
            }
            for (size_t i = 0; i < NW; i++) x[i] += y[i];
            rms_rows(xn, x, o->vl[l].ln2, N, VW, o->v_eps);
            qmm(a, &o->vl[l].fg, xn, N); qmm(b, &o->vl[l].fu, xn, N);
            for (size_t i = 0; i < (size_t)N * F; i++) a[i] = silu(a[i]) * b[i];
            qmm(y, &o->vl[l].fd, a, N);
            for (size_t i = 0; i < NW; i++) x[i] += y[i];
            probe(o, "vision.layer", l, x, N, VW);
        }
        /* projector: LayerNorm per patch, each 2x2 block concatenated in the reference's
         * (row, col) order within the block, block rows in raster order */
        for (int j = 0; j < N; j++) layernorm(xn + (size_t)j * VW, x + (size_t)j * VW, o->mm_norm_w, o->mm_norm_b, VW, o->p_eps);
        int bh = gh / 2, bw = gw / 2, nb = bh * bw;
        for (int by = 0; by < bh; by++)
            for (int bx = 0; bx < bw; bx++)
                for (int my = 0; my < 2; my++)
                    for (int mx = 0; mx < 2; mx++)
                        memcpy(m + (((size_t)by * bw + bx) * 4 + my * 2 + mx) * VW,
                               xn + ((size_t)(2 * by + my) * gw + 2 * bx + mx) * VW, VW * sizeof(float));
        qmm(mh, &o->mm0, m, nb);
        bias_rows(mh, o->mm0_b, nb, o->mm0.rows);
        for (size_t i = 0; i < (size_t)nb * o->mm0.rows; i++) mh[i] = gelu(mh[i]);
        float *dst = out + (size_t)gi * nb * I->width;
        qmm(dst, &o->mm2, mh, nb);
        bias_rows(dst, o->mm2_b, nb, I->width);
        probe(o, "vision.out", gi, dst, nb, I->width);
    }
    rc = 0;
done:
    free(pt); free(x); free(xn); free(y); free(t2); free(a); free(b); free(q); free(k); free(v);
    free(cs); free(sn); free(m); free(mh);
    return rc;
}

/* ── audio ───────────────────────────────────────────────────────────────────── */

/* The power spectrum in double, by a direct real DFT over a cosine table: at n_fft = 400
 * that is 80 thousand multiply-adds a frame, and the bins a pure tone leaves near zero keep
 * their digits instead of a float transform's rounding noise, which log10 would magnify. */
typedef struct { const float *sig; const double *win, *tab; double *pw; int nf, hop, bins; } mel_job;

static void mel_items(void *ctx, int f0, int f1) {
    mel_job *j = ctx;
    int N = j->nf;
    double *x = malloc((size_t)N * sizeof(double));
    if (!x) return;
    for (int f = f0; f < f1; f++) {
        for (int t = 0; t < N; t++) x[t] = (double)j->sig[(size_t)f * j->hop + t] * j->win[t];
        for (int k = 0; k < j->bins; k++) {
            double re = 0, im = 0;
            long idx = 0;
            for (int t = 0; t < N; t++) {
                re += x[t] * j->tab[idx];
                im -= x[t] * j->tab[(idx + 3L * N / 4) % N];
                idx += k; if (idx >= N) idx -= N;
            }
            j->pw[(size_t)f * j->bins + k] = re * re + im * im;
        }
    }
    free(x);
}

int lfm3_omni_audio_mel(const lfm3_omni *o, const float *pcm, int n, float **mel, int *n_frames) {
    const lfm3_omni_info *I = &o->info;
    int nf = o->n_fft, hop = I->hop, bins = nf / 2 + 1, M = I->n_mel;
    if (n < 1 || nf % 4) return -1;
    long padded = n < nf ? nf : n + ((-(long)(n - nf)) % hop + hop) % hop;
    int frames = (int)(1 + (padded - nf) / hop);
    float *sig = calloc((size_t)padded, sizeof(float)), *out = malloc((size_t)M * frames * sizeof(float));
    double *win = malloc((size_t)nf * sizeof(double)), *tab = malloc((size_t)nf * sizeof(double));
    double *pw = malloc((size_t)frames * bins * sizeof(double));
    int rc = -1;
    if (!sig || !win || !tab || !pw || !out) goto done;
    memcpy(sig, pcm, (size_t)n * sizeof(float));
    for (int t = 0; t < nf; t++) {
        win[t] = 0.5 - 0.5 * cos(2.0 * M_PI * t / nf);   /* periodic Hann */
        tab[t] = cos(2.0 * M_PI * t / nf);               /* sin(a) = cos(a - pi/2): index + 3N/4 */
    }
    mel_job job = {sig, win, tab, pw, nf, hop, bins};
    nt_par_for(mel_items, &job, frames, 8);
    for (int m = 0; m < M; m++) {
        const float *fm = o->mel + (size_t)m * bins;
        for (int f = 0; f < frames; f++) {
            const double *p = pw + (size_t)f * bins;
            double acc = 0;
            for (int k = 0; k < bins; k++) acc += (double)fm[k] * p[k];
            double lg = log10(acc);
            if (!(lg >= o->a_floor)) lg = o->a_floor;   /* also catches log10(0) = -inf */
            out[(size_t)m * frames + f] = (float)((lg + o->a_offset) / o->a_div);
        }
    }
    *mel = out; out = NULL; *n_frames = frames;
    rc = 0;
done:
    free(sig); free(win); free(tab); free(pw); free(out);
    return rc;
}

/* 3x3, stride 2, padding 1, one input plane per output channel (depthwise) or a single
 * input plane shared by every channel (the first convolution). in [C or 1][H][W]. */
static void conv3s2(float *out, const float *in, const float *w, const float *b, int C, int depthwise, int H, int W, int Ho, int Wo) {
    for (int c = 0; c < C; c++) {
        const float *src = in + (depthwise ? (size_t)c * H * W : 0);
        const float *k = w + (size_t)c * 9;
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                float acc = b[c];
                for (int ky = 0; ky < 3; ky++) {
                    int iy = 2 * oy - 1 + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = 2 * ox - 1 + kx;
                        if (ix >= 0 && ix < W) acc += k[ky * 3 + kx] * src[(size_t)iy * W + ix];
                    }
                }
                out[((size_t)c * Ho + oy) * Wo + ox] = acc;
            }
    }
}

/* 1x1 over [C][S]: transpose to [S][C], one matmul, back, bias and SiLU. */
static int pointwise(float *x, const wt *w, const float *b, int C, int S) {
    float *t = malloc((size_t)S * C * sizeof(float)), *u = malloc((size_t)S * C * sizeof(float));
    if (!t || !u) { free(t); free(u); return -1; }
    for (int c = 0; c < C; c++) for (int s = 0; s < S; s++) t[(size_t)s * C + c] = x[(size_t)c * S + s];
    qmm(u, w, t, S);
    for (int c = 0; c < C; c++) for (int s = 0; s < S; s++) x[(size_t)c * S + s] = silu(u[(size_t)s * C + c] + b[c]);
    free(t); free(u);
    return 0;
}

int lfm3_omni_audio_encode(lfm3_omni *o, const float *mel, int T, float *out) {
    const lfm3_omni_info *I = &o->info;
    int C = o->a_ch, M = I->n_mel, AW = o->a_width;
    if (T < 1) return -1;
    int M1 = (M + 1) / 2, T1 = (T + 1) / 2, M2 = (M1 + 1) / 2, T2 = (T1 + 1) / 2, M3 = (M2 + 1) / 2, T3 = (T2 + 1) / 2;
    float *h1 = malloc((size_t)C * M1 * T1 * sizeof(float)), *h2 = malloc((size_t)C * M2 * T2 * sizeof(float));
    float *h3 = malloc((size_t)C * M3 * T3 * sizeof(float)), *fl = malloc((size_t)T3 * C * M3 * sizeof(float));
    float *x = malloc((size_t)T3 * AW * sizeof(float)), *xn = malloc((size_t)T3 * AW * sizeof(float));
    float *hid = malloc((size_t)T3 * o->am1.rows * sizeof(float));
    int rc = -1;
    if (!h1 || !h2 || !h3 || !fl || !x || !xn || !hid || C * M3 != o->a_out.cols) goto done;
    conv3s2(h1, mel, o->a_w[0], o->a_b[0], C, 0, M, T, M1, T1);
    for (size_t i = 0; i < (size_t)C * M1 * T1; i++) h1[i] = silu(h1[i]);
    conv3s2(h2, h1, o->a_w[1], o->a_b[1], C, 1, M1, T1, M2, T2);
    if (pointwise(h2, &o->a_lin_mid, o->a_b[2], C, M2 * T2)) goto done;
    conv3s2(h3, h2, o->a_w[3], o->a_b[3], C, 1, M2, T2, M3, T3);
    if (pointwise(h3, &o->a_lin_end, o->a_b[4], C, M3 * T3)) goto done;
    /* flatten(1, 2): channel-major then mel, one vector per time step */
    for (int t = 0; t < T3; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < M3; f++) fl[(size_t)t * C * M3 + (size_t)c * M3 + f] = h3[((size_t)c * M3 + f) * T3 + t];
    qmm(x, &o->a_out, fl, T3);
    bias_rows(x, o->a_out_b, T3, AW);
    probe(o, "audio.downsample", -1, x, T3, AW);
    if (lfm_run(o, &o->enc, x, T3)) goto done;
    probe(o, "audio.encoder", -1, x, T3, AW);
    for (int j = 0; j < T3; j++) layernorm(xn + (size_t)j * AW, x + (size_t)j * AW, o->am_norm_w, o->am_norm_b, AW, 1e-5f);
    qmm(hid, &o->am1, xn, T3);
    bias_rows(hid, o->am1_b, T3, o->am1.rows);
    for (size_t i = 0; i < (size_t)T3 * o->am1.rows; i++) hid[i] = gelu(hid[i]);
    qmm(out, &o->am3, hid, T3);
    bias_rows(out, o->am3_b, T3, I->width);
    probe(o, "audio.out", -1, out, T3, I->width);
    rc = 0;
done:
    free(h1); free(h2); free(h3); free(fl); free(x); free(xn); free(hid);
    return rc;
}

/* ── speech out ──────────────────────────────────────────────────────────────── */

int lfm3_omni_depth_logits(lfm3_omni *o, const float *hidden, const int *codes, const uint8_t *masked, float *logits) {
    const lfm3_omni_info *I = &o->info;
    int NC = I->n_codebooks, CS = I->codebook_size, DW = o->depth.width;
    float *x = malloc((size_t)NC * DW * sizeof(float)), *e = malloc((size_t)DW * sizeof(float));
    float *nrm = malloc((size_t)DW * sizeof(float));
    int rc = -1;
    if (!x || !e || !nrm) goto done;
    for (int k = 0; k < NC; k++) if (!masked[k] && (codes[k] < 0 || codes[k] >= CS)) goto done;
    qmv(x, &o->d_proj, hidden);
    bias_rows(x, o->d_proj_b, 1, NC * DW);
    int ti = gguf_find_tensor(o->gf, "a.gen.code.embd.weight");
    for (int k = 0; k < NC; k++) {
        const float *add = o->d_mask;
        if (!masked[k]) {
            if (gguf_dequant_row(o->gf, ti, (uint64_t)k * CS + (uint64_t)codes[k], e)) goto done;
            add = e;
        }
        for (int i = 0; i < DW; i++) x[(size_t)k * DW + i] += add[i];
    }
    if (lfm_run(o, &o->depth, x, NC)) goto done;
    for (int k = 0; k < NC; k++) {
        rmsnorm(nrm, x + (size_t)k * DW, o->d_head_norm + (size_t)k * DW, DW, o->depth.eps);
        wt rows;
        if (!wt_expert(&rows, &o->d_embd, k, CS)) goto done;
        qmv(logits + (size_t)k * CS, &rows, nrm);
    }
    rc = 0;
done:
    free(x); free(e); free(nrm);
    return rc;
}

int lfm3_omni_frame_embed(const lfm3_omni *o, const int *codes, float *out) {
    const lfm3_omni_info *I = &o->info;
    int W = I->width, ti = gguf_find_tensor(o->gf, "a.gen.code.out_embd.weight");
    float *r = malloc((size_t)W * sizeof(float));
    if (!r || ti < 0) { free(r); return -1; }
    memset(out, 0, (size_t)W * sizeof(float));
    for (int k = 0; k < I->n_codebooks; k++) {
        if (codes[k] < 0 || codes[k] >= I->codebook_size ||
            gguf_dequant_row(o->gf, ti, (uint64_t)k * I->codebook_size + (uint64_t)codes[k], r)) { free(r); return -1; }
        for (int i = 0; i < W; i++) out[i] += r[i];
    }
    free(r);
    return 0;
}

typedef struct { const float *spec; float *frames; const float *window; int N, bins; } istft_job;

/* irfft with norm="backward" (1/N), then the synthesis window. The DC and Nyquist bins
 * contribute their real parts only, as a Hermitian inverse reads them. */
static void istft_items(void *ctx, int f0, int f1) {
    istft_job *j = ctx;
    int N = j->N, B = j->bins;
    for (int f = f0; f < f1; f++) {
        const float *mag = j->spec + (size_t)f * 2 * B, *ang = mag + B;
        float *dst = j->frames + (size_t)f * N;
        for (int t = 0; t < N; t++) {
            double acc = exp((double)mag[0]) * cos((double)ang[0]);
            acc += exp((double)mag[B - 1]) * cos((double)ang[B - 1]) * ((t & 1) ? -1.0 : 1.0);
            for (int k = 1; k < B - 1; k++) {
                double ph = 2.0 * M_PI * (double)((long)k * t % N) / N;
                acc += 2.0 * exp((double)mag[k]) * cos((double)ang[k] + ph);
            }
            dst[t] = (float)(acc / N) * j->window[t];
        }
    }
}

int lfm3_omni_codec_decode(lfm3_omni *o, const int *codes, int T, float *pcm) {
    const lfm3_omni_info *I = &o->info;
    int NC = I->n_codebooks, LV = o->n_lv, QD = o->q_out_w ? NC * LV : 0, CW = o->codec.width, FS = o->c_fs;
    int N = o->c_nfft, hop = o->c_hop, B = N / 2 + 1, n = T * FS;
    int qdim = o->c_in.cols;
    /* 2^20 decoder positions is more than six hours of speech and keeps every product
     * below (size_t and int) without a question */
    if (T < 1 || (long long)T * FS > (1LL << 20)) return -1;
    for (size_t i = 0; i < (size_t)T * NC; i++) if (codes[i] < 0 || codes[i] >= I->codebook_size - 1) return -1;
    long olen = (long)(n - 1) * hop + N;
    float *lat = malloc((size_t)T * QD * sizeof(float)), *q = malloc((size_t)T * qdim * sizeof(float));
    float *e = malloc((size_t)T * CW * sizeof(float)), *x = malloc((size_t)n * CW * sizeof(float));
    float *spec = malloc((size_t)n * 2 * B * sizeof(float)), *frames = malloc((size_t)n * N * sizeof(float));
    double *acc = calloc((size_t)olen, sizeof(double)), *env = calloc((size_t)olen, sizeof(double));
    int rc = -1;
    if (!lat || !q || !e || !x || !spec || !frames || !acc || !env) goto done;
    for (int t = 0; t < T; t++)
        for (int k = 0; k < NC; k++)
            memcpy(lat + (size_t)t * QD + (size_t)k * LV, o->fsq + (size_t)codes[(size_t)t * NC + k] * LV, LV * sizeof(float));
    mm_t(q, lat, o->q_out_w, T, QD, qdim);
    bias_rows(q, o->q_out_b, T, qdim);
    qmm(e, &o->c_in, q, T);
    bias_rows(e, o->c_in_b, T, CW);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < FS; r++) memcpy(x + ((size_t)t * FS + r) * CW, e + (size_t)t * CW, CW * sizeof(float));
    if (lfm_run(o, &o->codec, x, n)) goto done;
    qmm(spec, &o->c_lin, x, n);
    bias_rows(spec, o->c_lin_b, n, 2 * B);
    probe(o, "codec.spec", -1, spec, n, 2 * B);
    istft_job job = {spec, frames, o->window, N, B};
    nt_par_for(istft_items, &job, n, 4);
    for (int f = 0; f < n; f++)
        for (int t = 0; t < N; t++) {
            acc[(size_t)f * hop + t] += frames[(size_t)f * N + t];
            env[(size_t)f * hop + t] += (double)o->window[t] * o->window[t];
        }
    int trim = (N - hop) / 2;
    long total = (long)T * I->codec_samples_per_frame;
    if (total + trim > olen) goto done;
    for (long i = 0; i < total; i++) if (!(env[i + trim] > 0)) goto done;
    for (long i = 0; i < total; i++) pcm[i] = (float)(acc[i + trim] / env[i + trim]);
    rc = 0;
done:
    free(lat); free(q); free(e); free(x); free(spec); free(frames); free(acc); free(env);
    return rc;
}
