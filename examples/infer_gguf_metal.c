/* infer_gguf_metal.c — notorch-C packed-Q4_K GGUF inference on Apple Metal.
 *
 * Keeps Q4_K weights PACKED (no f32 blow-up) and runs them through
 * nt_metal_q4k_matvec; Q6_K / F32 / F16 tensors go through gguf_dequant + CPU
 * mm. This is the runtime path for oyent (24B on metal) and qyent (4B on Neo).
 *
 * Handles two model families with one forward:
 *   - llama / mistral  : interleaved RoPE (2i,2i+1), weights pre-permuted by
 *                        llama.cpp convert. No q/k-norm.   (oyent)
 *   - qwen2 / qwen3    : NEOX RoPE (i, i+HD/2), no permutation. qwen3 adds
 *                        per-head q/k RMSNorm before RoPE. (qyent)
 *
 * Tokenizer: byte-level BPE read from the GGUF (examples/bpe.{c,h}).
 *
 * Build: cc -O2 -DUSE_METAL -I. examples/infer_gguf_metal.c examples/bpe.c \
 *          gguf.c notorch_metal.o -framework Metal -framework Foundation \
 *          -lc++ -lm -o examples/infer_gguf_metal
 * Run:   ./examples/infer_gguf_metal <model.gguf> [prompt] [max_tokens] [temp]
 *        temp<=0 => greedy argmax (deterministic, best for correctness smoke).
 */
#include "gguf.h"
#include "notorch_metal.h"
#include "examples/bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

/* ── weight: packed Q4_K (q4k) / packed Q6_K (q6k) / dequantized f32 ─────────── */
typedef struct { const uint8_t *q4k; const uint8_t *q6k; float *f32; int m, k; } Weight;

static float f16f(uint16_t h) {
    uint32_t s = (h>>15)&1, e = (h>>10)&0x1F, mant = h&0x3FF, bits;
    if (e == 0) { if (mant == 0) bits = s<<31;
        else { e = 127-15+1; while(!(mant&0x400)){ mant<<=1; e--; } mant &= 0x3FF; bits = (s<<31)|(e<<23)|(mant<<13); } }
    else if (e == 0x1F) bits = (s<<31)|(0xFFu<<23)|(mant<<13);
    else bits = (s<<31)|((e-15+127)<<23)|(mant<<13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* y[m] = W_q6k @ x[k], dequantizing each 256-block on the fly (no f32 blow-up).
 * Mirrors gguf.c:dequant_q6_k (210 B/block: ql[128] qh[64] scales[16] d). */
static void q6k_rows(float *y, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t*)(b + 192);
            float d = f16f((uint16_t)(b[208] | (b[209] << 8)));
            const float *xb = x + (long)blk * 256;
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n/128)*64, *qhh = qh + (n/128)*32;
                const int8_t *sch = sc + (n/128)*8;
                for (int l = 0; l < 32; l++) {
                    int is = l/16;
                    int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * sch[is+0] * q1 * xb[n + l];
                    acc += d * sch[is+2] * q2 * xb[n + l + 32];
                    acc += d * sch[is+4] * q3 * xb[n + l + 64];
                    acc += d * sch[is+6] * q4 * xb[n + l + 96];
                }
            }
        }
        y[row] = acc;
    }
}

/* q6k matvec parallelized across CPU cores — rows are independent, disjoint y[]. */
typedef struct { float *y; const uint8_t *W; const float *x; int r0, r1, k; } q6k_job;
static void *q6k_worker(void *p) { q6k_job *j = (q6k_job*)p; q6k_rows(j->y, j->W, j->x, j->r0, j->r1, j->k); return NULL; }
static void q6k_matvec(float *y, const uint8_t *W, const float *x, int m, int k) {
    int nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nt < 1) nt = 1; if (nt > 16) nt = 16; if (nt > m) nt = m;
    long work = (long)m * (k / 256);   /* total 256-blocks; thread only big tensors */
    if (work < 100000 || nt <= 1) { q6k_rows(y, W, x, 0, m, k); return; }
    pthread_t th[16]; q6k_job jobs[16];
    int per = (m + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per, r1 = (r0 + per > m) ? m : r0 + per;
        jobs[t] = (q6k_job){ y, W, x, r0, r1, k };
        pthread_create(&th[t], NULL, q6k_worker, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

/* y[m] = W @ x[k]. Q4_K -> Metal; Q6_K -> packed CPU per-row dequant; else CPU f32. */
static void matvec(const Weight *W, const float *x, float *y) {
    if (W->q4k) nt_metal_q4k_matvec(y, W->q4k, x, W->m, W->k);
    else if (W->q6k) q6k_matvec(y, W->q6k, x, W->m, W->k);
    else for (int i = 0; i < W->m; i++) {
        const float *row = W->f32 + (long)i * W->k; float s = 0;
        for (int j = 0; j < W->k; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

/* GGUF 2D weight: ne[0]=in (k), ne[1]=out (m). Q4_K stays packed, else dequant.
 * Returns 0 if the tensor is absent. */
static int load_weight(gguf_file *gf, const char *name, Weight *W) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) { W->q4k = W->q6k = NULL; W->f32 = NULL; W->m = W->k = 0; return 0; }
    gguf_tensor_info *t = &gf->tensors[ti];
    W->k = (int)t->shape[0];
    W->m = (int)t->shape[1];
    W->q4k = W->q6k = NULL; W->f32 = NULL;
    if (t->dtype == GGUF_TYPE_Q4_K && (W->k % 256) == 0)      W->q4k = gf->data + t->offset;
    else if (t->dtype == GGUF_TYPE_Q6_K && (W->k % 256) == 0) W->q6k = gf->data + t->offset;
    else                                                      W->f32 = gguf_dequant(gf, ti);
    return 1;
}

static float *deq(gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    return ti >= 0 ? gguf_dequant(gf, ti) : NULL;
}

/* ── math ──────────────────────────────────────────────────────────────────── */
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* interleaved RoPE (llama/mistral; weights pre-permuted by convert) */
static void rope_interleaved(float *x, int pos, int hd, float base) {
    for (int i = 0; i < hd / 2; i++) {
        float a = pos / powf(base, 2.0f * i / hd);
        float c = cosf(a), s = sinf(a);
        float x0 = x[2*i], x1 = x[2*i+1];
        x[2*i] = x0*c - x1*s; x[2*i+1] = x0*s + x1*c;
    }
}

/* NEOX RoPE (qwen2/qwen3; pairs (i, i+hd/2)) */
static void rope_neox(float *x, int pos, int hd, float base) {
    int h2 = hd / 2;
    for (int i = 0; i < h2; i++) {
        float a = pos / powf(base, 2.0f * i / hd);
        float c = cosf(a), s = sinf(a);
        float x0 = x[i], x1 = x[i + h2];
        x[i] = x0*c - x1*s; x[i + h2] = x0*s + x1*c;
    }
}

static void add_bias(float *x, const float *b, int n) { if (b) for (int i = 0; i < n; i++) x[i] += b[i]; }

/* ── model ─────────────────────────────────────────────────────────────────── */
typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps;
    int has_output, is_qwen3, neox;
    float *tok_emb, *out_norm;
    Weight output;                /* tied -> output.f32 borrows tok_emb */
    struct {
        float *attn_norm, *ffn_norm;
        Weight wq, wk, wv, wo, wgate, wup, wdown;
        float *q_norm, *k_norm;   /* qwen3, [head_dim] */
        float *q_bias, *k_bias, *v_bias; /* qwen2.5 */
    } *L;
} model_t;

static model_t *model_load(gguf_file *gf) {
    model_t *m = (model_t*)calloc(1, sizeof(model_t));
    m->n_layers = gf->n_layers; m->n_heads = gf->n_heads; m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim; m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base; m->rms_eps = gf->rms_eps;

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    m->q_dim = ti >= 0 ? (int)gf->tensors[ti].shape[1] : m->embed;
    m->head_dim = m->q_dim / m->n_heads;
    m->kv_dim = m->head_dim * m->n_kv_heads;
    ti = gguf_find_tensor(gf, "token_embd.weight");
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;

    m->neox = (strstr(gf->arch, "qwen") || strstr(gf->arch, "gemma") || strstr(gf->arch, "phi")) ? 1 : 0;
    m->is_qwen3 = (gguf_find_tensor(gf, "blk.0.attn_q_norm.weight") >= 0) ? 1 : 0;

    m->tok_emb  = deq(gf, "token_embd.weight");
    m->out_norm = deq(gf, "output_norm.weight");
    if (load_weight(gf, "output.weight", &m->output)) m->has_output = 1;
    else { m->output.q4k = NULL; m->output.q6k = NULL; m->output.f32 = m->tok_emb; m->output.m = m->vocab; m->output.k = m->embed; m->has_output = 0; }

    m->L = calloc(m->n_layers, sizeof(*m->L));
    char nm[128];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); } while(0)
        #define LW(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); load_weight(gf, nm, &m->L[l].f); } while(0)
        LD(attn_norm, "blk.%d.attn_norm.weight"); LD(ffn_norm, "blk.%d.ffn_norm.weight");
        LW(wq, "blk.%d.attn_q.weight"); LW(wk, "blk.%d.attn_k.weight");
        LW(wv, "blk.%d.attn_v.weight"); LW(wo, "blk.%d.attn_output.weight");
        LW(wgate, "blk.%d.ffn_gate.weight"); LW(wup, "blk.%d.ffn_up.weight"); LW(wdown, "blk.%d.ffn_down.weight");
        LD(q_norm, "blk.%d.attn_q_norm.weight"); LD(k_norm, "blk.%d.attn_k_norm.weight");
        LD(q_bias, "blk.%d.attn_q.bias"); LD(k_bias, "blk.%d.attn_k.bias"); LD(v_bias, "blk.%d.attn_v.bias");
        #undef LD
        #undef LW
    }
    printf("model: arch=%s E=%d H=%d KV=%d HD=%d Q=%d FFN=%d V=%d L=%d | %s rope%s%s\n",
           gf->arch, m->embed, m->n_heads, m->n_kv_heads, m->head_dim, m->q_dim, m->ffn, m->vocab, m->n_layers,
           m->neox ? "NEOX" : "interleaved", m->is_qwen3 ? " +qk-norm" : "", m->has_output ? "" : " tied");
    if (!m->tok_emb || !m->out_norm) { fprintf(stderr, "missing tok_emb/out_norm\n"); return NULL; }
    return m;
}

/* ── KV cache ──────────────────────────────────────────────────────────────── */
typedef struct { float *k, *v; int max_seq, kv_dim; } kv_cache;
static kv_cache *kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache *c = calloc(1, sizeof(kv_cache));
    c->k = calloc((long)nl * max_seq * kv_dim, sizeof(float));
    c->v = calloc((long)nl * max_seq * kv_dim, sizeof(float));
    c->max_seq = max_seq; c->kv_dim = kv_dim; return c;
}

/* ── forward (single token, KV-cached) ─────────────────────────────────────── */
static void forward(model_t *m, kv_cache *kv, int token, int pos, float *logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim;
    int KVD = m->kv_dim, FFN = m->ffn, QD = m->q_dim, gqa = H / KV;
    float eps = m->rms_eps;
    void (*ropef)(float*,int,int,float) = m->neox ? rope_neox : rope_interleaved;

    float *x  = calloc(E, sizeof(float));
    memcpy(x, m->tok_emb + (long)token * E, E * sizeof(float));
    float *xn = calloc(E, sizeof(float));
    float *q  = calloc(QD, sizeof(float));
    float *kk = calloc(KVD, sizeof(float));
    float *vv = calloc(KVD, sizeof(float));
    float *ao = calloc(QD, sizeof(float));
    float *g  = calloc(FFN, sizeof(float));
    float *u  = calloc(FFN, sizeof(float));
    float *t  = calloc(E, sizeof(float));
    float *sc = calloc(m->n_layers && kv->max_seq ? kv->max_seq : 1, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->L[l].attn_norm, E, eps);
        matvec(&m->L[l].wq, xn, q);
        matvec(&m->L[l].wk, xn, kk);
        matvec(&m->L[l].wv, xn, vv);
        add_bias(q, m->L[l].q_bias, QD); add_bias(kk, m->L[l].k_bias, KVD); add_bias(vv, m->L[l].v_bias, KVD);

        if (m->is_qwen3) {                       /* per-head q/k RMSNorm before RoPE */
            for (int h = 0; h < H;  h++) rmsnorm(q + h*HD, q + h*HD, m->L[l].q_norm, HD, eps);
            for (int h = 0; h < KV; h++) rmsnorm(kk + h*HD, kk + h*HD, m->L[l].k_norm, HD, eps);
        }
        for (int h = 0; h < H;  h++) ropef(q + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) ropef(kk + h*HD, pos, HD, m->rope_base);

        long base = (long)l * kv->max_seq * KVD;
        memcpy(kv->k + base + (long)pos * KVD, kk, KVD * sizeof(float));
        memcpy(kv->v + base + (long)pos * KVD, vv, KVD * sizeof(float));

        float scale = 1.0f / sqrtf((float)HD);
        memset(ao, 0, QD * sizeof(float));
        for (int h = 0; h < H; h++) {
            int kvh = h / gqa;
            float *qh = q + h*HD;
            for (int j = 0; j <= pos; j++) {
                float *kj = kv->k + base + (long)j*KVD + kvh*HD, d = 0;
                for (int t2 = 0; t2 < HD; t2++) d += qh[t2] * kj[t2];
                sc[j] = d * scale;
            }
            softmax(sc, pos + 1);
            float *oh = ao + h*HD;
            for (int j = 0; j <= pos; j++) {
                float *vj = kv->v + base + (long)j*KVD + kvh*HD, w = sc[j];
                for (int t2 = 0; t2 < HD; t2++) oh[t2] += w * vj[t2];
            }
        }
        matvec(&m->L[l].wo, ao, t);
        for (int i = 0; i < E; i++) x[i] += t[i];

        rmsnorm(xn, x, m->L[l].ffn_norm, E, eps);
        matvec(&m->L[l].wgate, xn, g);
        matvec(&m->L[l].wup, xn, u);
        for (int i = 0; i < FFN; i++) { float gi = g[i]; g[i] = (gi / (1.0f + expf(-gi))) * u[i]; }
        matvec(&m->L[l].wdown, g, t);
        for (int i = 0; i < E; i++) x[i] += t[i];
    }

    rmsnorm(xn, x, m->out_norm, E, eps);
    matvec(&m->output, xn, logits);

    free(x); free(xn); free(q); free(kk); free(vv); free(ao); free(g); free(u); free(t); free(sc);
}

static int argmax(const float *x, int n) { int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b; }
static int sample(float *x, int n, float temp) {
    if (temp <= 0) return argmax(x, n);
    for (int i = 0; i < n; i++) x[i] /= temp;
    softmax(x, n);
    float r = (float)rand() / RAND_MAX, c = 0;
    for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; }
    return n - 1;
}
static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp]\n", argv[0]); return 1; }
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 40;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.0f;
    srand(42);

    double t0 = now_ms();
    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;
    nt_metal_register_base(gf->data, gf->data_size);  /* Phase 2: weights resident on GPU, no per-call upload */
    model_t *m = model_load(gf);
    if (!m) return 1;
    bpe_tokenizer *tok = bpe_load(argv[1]);
    if (!tok) { fprintf(stderr, "bpe_load failed\n"); return 1; }
    int eos = -1;
    const gguf_kv *e = gguf_get_kv(gf, "tokenizer.ggml.eos_token_id");
    if (e) eos = (int)e->val.u32;
    printf("loaded in %.0f ms (vocab=%d eos=%d)\n", now_ms() - t0, bpe_n_vocab(tok), eos);

    int max_seq = 512;
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = calloc(m->vocab, sizeof(float));

    int ids[512];
    int n = bpe_encode(tok, prompt, ids, max_seq - max_tokens - 1);
    printf("\nprompt: \"%s\" (%d tokens, temp=%.2f)\n---\n%s", prompt, n, temp, prompt);
    fflush(stdout);

    double g0 = now_ms();
    for (int i = 0; i < n; i++) forward(m, kv, ids[i], i, logits);
    double prefill = now_ms() - g0;

    int gen = 0; char buf[256];
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, m->vocab, temp);
        if (next == eos) break;
        bpe_decode_token(tok, next, buf, sizeof(buf));
        printf("%s", buf); fflush(stdout);
        gen++;
        int pos = n + step;
        if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits);
    }
    double total = now_ms() - g0;
    printf("\n---\nprefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s)\n",
           n, prefill, n*1000.0/prefill, gen, total-prefill, gen > 0 ? gen*1000.0/(total-prefill) : 0);
    return 0;
}
