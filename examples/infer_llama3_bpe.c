/*
 * infer_llama3_bpe.c — Interactive chat with a LLaMA 3 BPE model on notorch
 *   (MHA + RoPE + SwiGLU + RMSNorm, 15.7M params, vocab 2048)
 *
 * Build: make infer_llama3_bpe
 * Run:   ./infer_llama3_bpe [weights.bin] [merges.txt]
 *
 * Default: llama3_bpe.bin + examples/bpe_2048_merges.txt
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM       384
#define NLAYERS   8
#define NHEADS    8
#define HEAD_DIM  (DIM / NHEADS)
#define HIDDEN    1024
#define CTX       256
#define VOCAB     2048

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo, *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static int model_n_tensors(void) { return 1 + NLAYERS * 9 + 2; }

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(DIM, DIM);
        m->L[l].wv = nt_tensor_new2d(DIM, DIM);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM);
        m->L[l].rms2 = nt_tensor_new(DIM);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_up = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN);
    }
    m->rms_f = nt_tensor_new(DIM);
    m->head = nt_tensor_new2d(VOCAB, DIM);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

/* FP16 → FP32 */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) { float z = 0; uint32_t r = sign; memcpy(&z, &r, 4); return z; }
    if (exp == 31) exp = 255; else exp = exp - 15 + 127;
    uint32_t r = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

static int load_weights_f16(Model* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic; int n;
    fread(&magic, 4, 1, f); fread(&n, 4, 1, f);
    if (magic != 0x3631544E) { fclose(f); return -1; } /* "NT16" */
    int expected = model_n_tensors();
    if (n != expected) { fclose(f); return -1; }
    nt_tensor* params[75];
    int pi = 0;
    params[pi++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        params[pi++]=m->L[l].rms1; params[pi++]=m->L[l].wq; params[pi++]=m->L[l].wk;
        params[pi++]=m->L[l].wv; params[pi++]=m->L[l].wo; params[pi++]=m->L[l].rms2;
        params[pi++]=m->L[l].w_gate; params[pi++]=m->L[l].w_up; params[pi++]=m->L[l].w_down;
    }
    params[pi++] = m->rms_f; params[pi++] = m->head;
    for (int t = 0; t < expected; t++) {
        int ndim; fread(&ndim, 4, 1, f);
        for (int d = 0; d < ndim; d++) { int s; fread(&s, 4, 1, f); }
        for (int i = 0; i < params[t]->len; i++) {
            uint16_t h; fread(&h, 2, 1, f);
            params[t]->data[i] = f16_to_f32(h);
        }
    }
    fclose(f);
    return 0;
}

static int load_weights(Model* m, const char* path) {
    /* Try FP16 first */
    if (load_weights_f16(m, path) == 0) { printf("loaded FP16 weights\n"); return 0; }
    /* Fallback to FP32 (notorch format) */
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        printf("WARN: expected %d tensors, got %d\n", expected, n_loaded);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    nt_tensor* params[] = {
        m->wte,
        m->L[0].rms1, m->L[0].wq, m->L[0].wk, m->L[0].wv, m->L[0].wo, m->L[0].rms2,
        m->L[0].w_gate, m->L[0].w_up, m->L[0].w_down,
        m->L[1].rms1, m->L[1].wq, m->L[1].wk, m->L[1].wv, m->L[1].wo, m->L[1].rms2,
        m->L[1].w_gate, m->L[1].w_up, m->L[1].w_down,
        m->L[2].rms1, m->L[2].wq, m->L[2].wk, m->L[2].wv, m->L[2].wo, m->L[2].rms2,
        m->L[2].w_gate, m->L[2].w_up, m->L[2].w_down,
        m->L[3].rms1, m->L[3].wq, m->L[3].wk, m->L[3].wv, m->L[3].wo, m->L[3].rms2,
        m->L[3].w_gate, m->L[3].w_up, m->L[3].w_down,
        m->L[4].rms1, m->L[4].wq, m->L[4].wk, m->L[4].wv, m->L[4].wo, m->L[4].rms2,
        m->L[4].w_gate, m->L[4].w_up, m->L[4].w_down,
        m->L[5].rms1, m->L[5].wq, m->L[5].wk, m->L[5].wv, m->L[5].wo, m->L[5].rms2,
        m->L[5].w_gate, m->L[5].w_up, m->L[5].w_down,
        m->L[6].rms1, m->L[6].wq, m->L[6].wk, m->L[6].wv, m->L[6].wo, m->L[6].rms2,
        m->L[6].w_gate, m->L[6].w_up, m->L[6].w_down,
        m->L[7].rms1, m->L[7].wq, m->L[7].wk, m->L[7].wv, m->L[7].wo, m->L[7].rms2,
        m->L[7].w_gate, m->L[7].w_up, m->L[7].w_down,
        m->rms_f, m->head
    };
    for (int i = 0; i < expected; i++) {
        memcpy(params[i]->data, loaded[i]->data, params[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded);
    return 0;
}

/* ── Forward (inference only, no tape) ── */

static void rmsnorm(float* out, const float* x, const float* w, int d) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = x[i] * ss * w[i];
}

static void matmul(float* out, const float* x, const float* w, int out_d, int in_d) {
    for (int o = 0; o < out_d; o++) {
        float s = 0;
        for (int i = 0; i < in_d; i++) s += w[o * in_d + i] * x[i];
        out[o] = s;
    }
}

static void rope(float* x, int pos, int dim, int head_dim) {
    for (int h = 0; h < dim / head_dim; h++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * i) / head_dim);
            float theta = pos * freq;
            float cs = cosf(theta), sn = sinf(theta);
            int idx = h * head_dim + i * 2;
            float x0 = x[idx], x1 = x[idx + 1];
            x[idx]     = x0 * cs - x1 * sn;
            x[idx + 1] = x0 * sn + x1 * cs;
        }
    }
}

static void softmax(float* x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sm = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sm;
}

/* KV cache */
static float kv_k[NLAYERS][CTX][DIM];
static float kv_v[NLAYERS][CTX][DIM];

static void forward_pos(Model* m, int token, int pos, float* logits) {
    float x[DIM], xn[DIM], q[DIM], k[DIM], v[DIM], attn_out[DIM];
    float gate[HIDDEN], up[HIDDEN], down[DIM], ffn_out[DIM];

    /* Token embedding */
    memcpy(x, m->wte->data + token * DIM, DIM * sizeof(float));

    for (int l = 0; l < NLAYERS; l++) {
        /* Attn norm */
        rmsnorm(xn, x, m->L[l].rms1->data, DIM);

        /* QKV */
        matmul(q, xn, m->L[l].wq->data, DIM, DIM);
        matmul(k, xn, m->L[l].wk->data, DIM, DIM);
        matmul(v, xn, m->L[l].wv->data, DIM, DIM);

        /* RoPE */
        rope(q, pos, DIM, HEAD_DIM);
        rope(k, pos, DIM, HEAD_DIM);

        /* Store in KV cache */
        memcpy(kv_k[l][pos], k, DIM * sizeof(float));
        memcpy(kv_v[l][pos], v, DIM * sizeof(float));

        /* Multi-head attention */
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        memset(attn_out, 0, DIM * sizeof(float));
        for (int h = 0; h < NHEADS; h++) {
            int ho = h * HEAD_DIM;
            float scores[CTX];
            for (int j = 0; j <= pos; j++) {
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[ho + d] * kv_k[l][j][ho + d];
                scores[j] = dot * scale;
            }
            /* Softmax over 0..pos */
            float mx = scores[0];
            for (int j = 1; j <= pos; j++) if (scores[j] > mx) mx = scores[j];
            float sm = 0;
            for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j] - mx); sm += scores[j]; }
            for (int j = 0; j <= pos; j++) scores[j] /= sm;
            /* Weighted sum of values */
            for (int j = 0; j <= pos; j++)
                for (int d = 0; d < HEAD_DIM; d++)
                    attn_out[ho + d] += scores[j] * kv_v[l][j][ho + d];
        }

        /* Output projection + residual */
        float proj[DIM];
        matmul(proj, attn_out, m->L[l].wo->data, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += proj[i];

        /* FFN norm */
        rmsnorm(xn, x, m->L[l].rms2->data, DIM);

        /* SwiGLU FFN */
        matmul(gate, xn, m->L[l].w_gate->data, HIDDEN, DIM);
        matmul(up, xn, m->L[l].w_up->data, HIDDEN, DIM);
        for (int i = 0; i < HIDDEN; i++)
            gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i]; /* SiLU(gate) * up */
        matmul(down, gate, m->L[l].w_down->data, DIM, HIDDEN);
        for (int i = 0; i < DIM; i++) x[i] += down[i];
    }

    /* Final norm + lm_head */
    rmsnorm(xn, x, m->rms_f->data, DIM);
    matmul(logits, xn, m->head->data, VOCAB, DIM);
}

static int sample(float* logits, float temperature, int top_k) {
    for (int i = 0; i < VOCAB; i++) logits[i] /= temperature;
    /* Top-k: find k-th largest, zero out rest */
    if (top_k > 0 && top_k < VOCAB) {
        float threshold = -1e30f;
        float tmp[VOCAB];
        memcpy(tmp, logits, VOCAB * sizeof(float));
        for (int k = 0; k < top_k; k++) {
            float mx = -1e30f; int mi = 0;
            for (int i = 0; i < VOCAB; i++) if (tmp[i] > mx) { mx = tmp[i]; mi = i; }
            threshold = mx;
            tmp[mi] = -1e30f;
        }
        for (int i = 0; i < VOCAB; i++) if (logits[i] < threshold) logits[i] = -1e30f;
    }
    softmax(logits, VOCAB);
    float r = (float)rand() / (float)RAND_MAX, cum = 0;
    for (int i = 0; i < VOCAB; i++) { cum += logits[i]; if (cum >= r) return i; }
    return VOCAB - 1;
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "llama3_bpe.bin";
    const char* merges_path = argc > 2 ? argv[2] : "examples/bpe_2048_merges.txt";

    srand((unsigned)time(NULL));

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch — LLaMA 3 BPE chat (15.7M, MHA + RoPE + SwiGLU)\n");
    printf("════════════════════════════════════════════════════════\n");

    /* Load BPE */
    nt_bpe bpe;
    if (nt_bpe_load(&bpe, merges_path) < 0) {
        printf("cannot load %s\n", merges_path); return 1;
    }
    printf("bpe: %d merges, vocab %d\n", bpe.n_merges, bpe.vocab_size);

    /* Load model */
    Model* model = model_new();
    if (load_weights(model, weights_path) < 0) {
        printf("cannot load %s\n", weights_path); return 1;
    }
    printf("model loaded: %s\n", weights_path);
    printf("────────────────────────────────────────────────────\n");
    printf("  type your message (or 'quit' to exit)\n");
    printf("────────────────────────────────────────────────────\n\n");

    char input[4096];
    while (1) {
        printf("You: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = (int)strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = 0;
        if (len == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;

        /* Build prompt: "Q: {input}\nA:" */
        char prompt[4096];
        snprintf(prompt, sizeof(prompt), "Q: %s\nA:", input);
        int tokens[CTX];
        int n = nt_bpe_encode(&bpe, prompt, (int)strlen(prompt), tokens, CTX / 2);

        /* Generate */
        printf("AI: ");
        fflush(stdout);

        /* Prefill */
        float logits[VOCAB];
        for (int i = 0; i < n; i++)
            forward_pos(model, tokens[i], i, logits);

        /* Decode */
        int pos = n;
        for (int s = 0; s < CTX - n; s++) {
            int next = sample(logits, 0.8f, 40);
            tokens[pos] = next;

            /* Decode token and print */
            char decoded[NT_BPE_MAX_TOKEN_LEN + 1];
            nt_bpe_decode(&bpe, &next, 1, decoded, NT_BPE_MAX_TOKEN_LEN);
            /* Stop on Q: boundary */
            if (strstr(decoded, "\nQ") != NULL || strstr(decoded, "\n\n") != NULL) break;
            printf("%s", decoded);
            fflush(stdout);

            /* Next step */
            forward_pos(model, next, pos, logits);
            pos++;
            if (pos >= CTX) break;
        }
        printf("\n\n");
    }

    model_free(model);
    printf("\n  bye.\n");
    return 0;
}
