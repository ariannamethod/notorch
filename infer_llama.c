/*
 * infer_llama.c — LLaMA-family inference on notorch via GGUF
 *
 * Supports: SmolLM2, nanollama, Qwen2.5, LLaMA, Mistral — any GGUF with
 * llama/qwen2 architecture. Auto-detects GQA, bias, tied embeddings.
 *
 * Build: make llama
 * Run:   ./infer_llama <model.gguf> [prompt] [max_tokens] [temp]
 */

#include "gguf.h"
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

// C[m,n] = A[m,k] @ B^T[n,k]
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

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

static void rope(float *x, int pos, int head_dim, float freq_base) {
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = x[2*i], x1 = x[2*i+1];
        x[2*i]   = x0 * cs - x1 * sn;
        x[2*i+1] = x0 * sn + x1 * cs;
    }
}

static void add_bias(float *x, const float *bias, int n) {
    if (bias) for (int i = 0; i < n; i++) x[i] += bias[i];
}

// ── LLaMA model ──────────────────────────────────────────────────────────────

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps;
    int has_output_weight; // 0 = tied embeddings

    float *tok_emb;     // [vocab, embed]
    float *out_norm;    // [embed]
    float *out_weight;  // [vocab, embed] or NULL (tied)

    struct {
        float *attn_norm;
        float *wq, *wk, *wv, *wo;
        float *q_bias, *k_bias, *v_bias; // Qwen has bias
        float *ffn_norm;
        float *wgate, *wup, *wdown;
    } layers[];
} llama_model;

static llama_model* llama_load(gguf_file* gf) {
    int nl = gf->n_layers;
    llama_model* m = (llama_model*)calloc(1, sizeof(llama_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;

    // Detect head_dim and vocab from tensor shapes
    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) {
        m->q_dim = (int)gf->tensors[ti].shape[1];
        m->head_dim = m->q_dim / m->n_heads;
    } else {
        m->head_dim = m->embed / m->n_heads;
        m->q_dim = m->n_heads * m->head_dim;
    }
    m->kv_dim = m->head_dim * m->n_kv_heads;

    // Vocab from embedding tensor
    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->vocab = (int)gf->tensors[ti].shape[1];
    else if (gf->vocab_size > 0) m->vocab = gf->vocab_size;
    else m->vocab = 32000;

    printf("llama: E=%d H=%d KV=%d FFN=%d V=%d L=%d HD=%d Q=%d\n",
           m->embed, m->n_heads, m->n_kv_heads, m->ffn, m->vocab, nl, m->head_dim, m->q_dim);

    // Global weights
    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->tok_emb = gguf_dequant(gf, ti);
    ti = gguf_find_tensor(gf, "output_norm.weight");
    if (ti >= 0) m->out_norm = gguf_dequant(gf, ti);
    ti = gguf_find_tensor(gf, "output.weight");
    if (ti >= 0) { m->out_weight = gguf_dequant(gf, ti); m->has_output_weight = 1; }

    // Per-layer
    for (int l = 0; l < nl; l++) {
        char name[128];
        #define L(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            ti = gguf_find_tensor(gf, name); \
            if (ti >= 0) m->layers[l].field = gguf_dequant(gf, ti); \
        } while(0)
        L(attn_norm, "blk.%d.attn_norm.weight");
        L(wq, "blk.%d.attn_q.weight");
        L(wk, "blk.%d.attn_k.weight");
        L(wv, "blk.%d.attn_v.weight");
        L(wo, "blk.%d.attn_output.weight");
        L(q_bias, "blk.%d.attn_q.bias");
        L(k_bias, "blk.%d.attn_k.bias");
        L(v_bias, "blk.%d.attn_v.bias");
        L(ffn_norm, "blk.%d.ffn_norm.weight");
        L(wgate, "blk.%d.ffn_gate.weight");
        L(wup, "blk.%d.ffn_up.weight");
        L(wdown, "blk.%d.ffn_down.weight");
        #undef L
    }

    if (!m->tok_emb || !m->out_norm) {
        fprintf(stderr, "llama: missing critical weights\n");
        return NULL;
    }
    if (m->layers[0].q_bias) printf("  (has attention bias — qwen-style)\n");
    if (!m->has_output_weight) printf("  (tied embeddings)\n");

    return m;
}

static void llama_free(llama_model* m) {
    if (!m) return;
    free(m->tok_emb); free(m->out_norm); free(m->out_weight);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm);
        free(m->layers[l].wq); free(m->layers[l].wk);
        free(m->layers[l].wv); free(m->layers[l].wo);
        free(m->layers[l].q_bias); free(m->layers[l].k_bias); free(m->layers[l].v_bias);
        free(m->layers[l].ffn_norm);
        free(m->layers[l].wgate); free(m->layers[l].wup); free(m->layers[l].wdown);
    }
    free(m);
}

// ── KV Cache ─────────────────────────────────────────────────────────────────

typedef struct {
    float *k, *v;
    int max_seq, n_layers, kv_dim;
} kv_cache;

static kv_cache* kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache* kv = (kv_cache*)calloc(1, sizeof(kv_cache));
    kv->k = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->v = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->max_seq = max_seq; kv->n_layers = nl; kv->kv_dim = kv_dim;
    return kv;
}

static void kv_free(kv_cache* kv) { if (!kv) return; free(kv->k); free(kv->v); free(kv); }

// ── Forward (single token with KV cache) ─────────────────────────────────────

static void llama_forward(llama_model* m, kv_cache* kv, int token, int pos, float* logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    float eps = m->rms_eps;
    int gqa = H / KV;

    float *x = (float*)calloc(E, sizeof(float));
    memcpy(x, m->tok_emb + token * E, E * sizeof(float));

    float *xn = (float*)calloc(E, sizeof(float));
    float *q_all = (float*)calloc(Q_DIM, sizeof(float));
    float *k_new = (float*)calloc(KVD, sizeof(float));
    float *v_new = (float*)calloc(KVD, sizeof(float));
    float *attn_out = (float*)calloc(Q_DIM, sizeof(float));
    float *ffn_gate = (float*)calloc(FFN, sizeof(float));
    float *ffn_up = (float*)calloc(FFN, sizeof(float));
    float *ffn_out = (float*)calloc(E, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->layers[l].attn_norm, E, eps);

        mm_t(q_all, xn, m->layers[l].wq, 1, E, Q_DIM);
        mm_t(k_new, xn, m->layers[l].wk, 1, E, KVD);
        mm_t(v_new, xn, m->layers[l].wv, 1, E, KVD);
        add_bias(q_all, m->layers[l].q_bias, Q_DIM);
        add_bias(k_new, m->layers[l].k_bias, KVD);
        add_bias(v_new, m->layers[l].v_bias, KVD);

        // RoPE
        for (int h = 0; h < H; h++)
            rope(q_all + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++)
            rope(k_new + h*HD, pos, HD, m->rope_base);

        // KV cache
        long base = (long)l * kv->max_seq * KVD;
        memcpy(kv->k + base + pos * KVD, k_new, KVD * sizeof(float));
        memcpy(kv->v + base + pos * KVD, v_new, KVD * sizeof(float));

        // GQA attention
        float scale = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, Q_DIM * sizeof(float));
        for (int h = 0; h < H; h++) {
            int kv_h = h / gqa;
            float *q = q_all + h * HD;
            float *scores = (float*)calloc(pos + 1, sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *kj = kv->k + base + j * KVD + kv_h * HD;
                float dot = 0;
                for (int d = 0; d < HD; d++) dot += q[d] * kj[d];
                scores[j] = dot * scale;
            }
            softmax(scores, pos + 1);
            float *out_h = attn_out + h * HD;
            for (int j = 0; j <= pos; j++) {
                float *vj = kv->v + base + j * KVD + kv_h * HD;
                for (int d = 0; d < HD; d++) out_h[d] += scores[j] * vj[d];
            }
            free(scores);
        }

        // Output projection + residual
        float *proj = (float*)calloc(E, sizeof(float));
        mm_t(proj, attn_out, m->layers[l].wo, 1, Q_DIM, E);
        for (int i = 0; i < E; i++) x[i] += proj[i];
        free(proj);

        // FFN: SiLU-gated
        rmsnorm(xn, x, m->layers[l].ffn_norm, E, eps);
        mm_t(ffn_gate, xn, m->layers[l].wgate, 1, E, FFN);
        mm_t(ffn_up, xn, m->layers[l].wup, 1, E, FFN);
        for (int i = 0; i < FFN; i++) {
            float g = ffn_gate[i];
            ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i];
        }
        mm_t(ffn_out, ffn_gate, m->layers[l].wdown, 1, FFN, E);
        for (int i = 0; i < E; i++) x[i] += ffn_out[i];
    }

    rmsnorm(xn, x, m->out_norm, E, eps);
    float *lm_head = m->has_output_weight ? m->out_weight : m->tok_emb;
    mm_t(logits, xn, lm_head, 1, E, m->vocab);

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
}

// ── Sampling + timing ────────────────────────────────────────────────────────

static int sample(float *logits, int n, float temp) {
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    float r = (float)rand() / (float)RAND_MAX, cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("notorch LLaMA/Qwen inference via GGUF\n");
        printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp]\n", argv[0]);
        return 1;
    }

    double t0 = now_ms();
    gguf_file* gf = gguf_open(argv[1]);
    if (!gf) return 1;

    llama_model* model = llama_load(gf);
    if (!model) { gguf_close(gf); return 1; }
    printf("loaded in %.0f ms\n", now_ms() - t0);

    const char* prompt = argc > 2 ? argv[2] : "Hello";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 50;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.8f;

    int max_seq = 256;
    kv_cache* kv = kv_new(model->n_layers, max_seq, model->kv_dim);
    float *logits = (float*)calloc(model->vocab, sizeof(float));

    // Simple byte-level tokenization (BOS=1 for llama, BOS=2 for others)
    int tokens[256]; int n_tok = 0;
    tokens[n_tok++] = 1; // BOS
    for (int i = 0; prompt[i] && n_tok < max_seq - max_tokens; i++)
        tokens[n_tok++] = (unsigned char)prompt[i];

    printf("\nprompt: \"%s\" (%d tokens, temp=%.2f)\n", prompt, n_tok, temp);

    // Prefill
    double gen0 = now_ms();
    for (int i = 0; i < n_tok; i++)
        llama_forward(model, kv, tokens[i], i, logits);
    double prefill_ms = now_ms() - gen0;

    printf("%s", prompt);
    fflush(stdout);

    // Decode
    int gen = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, model->vocab, temp);
        if (next <= 2) break; // EOS/BOS/PAD

        // Print: if printable ASCII or newline
        if (next >= 32 && next < 127) printf("%c", (char)next);
        else if (next == 10) printf("\n");
        else printf("[%d]", next);
        fflush(stdout);
        gen++;

        int pos = n_tok + step;
        if (pos >= max_seq - 1) break;
        llama_forward(model, kv, next, pos, logits);
    }

    double total_ms = now_ms() - gen0;
    printf("\n\n── prefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s) ──\n",
           n_tok, prefill_ms, n_tok * 1000.0 / prefill_ms,
           gen, total_ms - prefill_ms, gen > 0 ? gen * 1000.0 / (total_ms - prefill_ms) : 0);

    free(logits); kv_free(kv); llama_free(model); gguf_close(gf);
    return 0;
}
