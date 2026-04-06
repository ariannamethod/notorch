/*
 * infer_gemma.c — Gemma-3 270M inference on notorch via GGUF
 *
 * Loads any Gemma-3 GGUF (Q8_0, Q4_0, F16, F32).
 * Architecture: 18 layers, E=640, H=4, KV_heads=1 (GQA),
 *               FFN=2048, V=262144, RoPE, SiLU-gated FFN,
 *               QK-norm, post-attention/FFN norms
 *
 * Build: make gemma
 * Run:   ./infer_gemma ~/Downloads/gemma-notorch/leo-q8_0.gguf [prompt] [max] [temp]
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

// ── Matmul ───────────────────────────────────────────────────────────────────

// C[m,n] = A[m,k] @ B^T[n,k]  (GGUF stores weights as [out,in] like PyTorch)
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
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = x[2*i];
        float x1 = x[2*i+1];
        x[2*i]   = x0 * cos_a - x1 * sin_a;
        x[2*i+1] = x0 * sin_a + x1 * cos_a;
    }
}

// ── Gemma-3 model ────────────────────────────────────────────────────────────

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim;
    float rope_base, rms_eps;

    // Weights (dequantized to f32)
    float *tok_emb;     // [vocab, embed]
    float *out_norm;    // [embed]

    struct {
        float *attn_norm, *post_attn_norm;
        float *wq, *wk, *wv, *wo;
        float *q_norm, *k_norm;
        float *ffn_norm, *post_ffn_norm;
        float *wgate, *wup, *wdown;
    } layers[];
} gemma_model;

static gemma_model* gemma_load(gguf_file* gf) {
    int nl = gf->n_layers;
    gemma_model* m = (gemma_model*)calloc(1, sizeof(gemma_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;

    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->vocab = gf->vocab_size ? gf->vocab_size : 262144;  // Gemma-3 default
    // head_dim comes from Q weight shape, NOT embed/heads
    // Q weight in GGUF: [embed, n_heads*head_dim] — ne[1] = total Q output dim
    // We'll detect from tensor shape after loading
    m->head_dim = m->embed / m->n_heads; // default, overridden below
    m->kv_dim = m->head_dim * m->n_kv_heads;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;

    printf("gemma: E=%d H=%d KV=%d FFN=%d V=%d L=%d head_dim=%d\n",
           m->embed, m->n_heads, m->n_kv_heads, m->ffn, m->vocab, nl, m->head_dim);

    // Load global weights
    int ti;
    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->tok_emb = gguf_dequant(gf, ti);

    ti = gguf_find_tensor(gf, "output_norm.weight");
    if (ti >= 0) m->out_norm = gguf_dequant(gf, ti);

    // Load per-layer weights
    for (int l = 0; l < nl; l++) {
        char name[128];
        #define LOAD(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, l); \
            ti = gguf_find_tensor(gf, name); \
            if (ti >= 0) m->layers[l].field = gguf_dequant(gf, ti); \
            else fprintf(stderr, "warning: missing %s\n", name); \
        } while(0)

        LOAD(attn_norm, "blk.%d.attn_norm.weight");
        LOAD(post_attn_norm, "blk.%d.post_attention_norm.weight");
        LOAD(wq, "blk.%d.attn_q.weight");
        LOAD(wk, "blk.%d.attn_k.weight");
        LOAD(wv, "blk.%d.attn_v.weight");
        LOAD(wo, "blk.%d.attn_output.weight");
        LOAD(q_norm, "blk.%d.attn_q_norm.weight");
        LOAD(k_norm, "blk.%d.attn_k_norm.weight");
        LOAD(ffn_norm, "blk.%d.ffn_norm.weight");
        LOAD(post_ffn_norm, "blk.%d.post_ffw_norm.weight");
        LOAD(wgate, "blk.%d.ffn_gate.weight");
        LOAD(wup, "blk.%d.ffn_up.weight");
        LOAD(wdown, "blk.%d.ffn_down.weight");
        #undef LOAD
    }

    // Check critical weights
    if (!m->tok_emb || !m->out_norm) {
        fprintf(stderr, "gemma: missing critical weights\n");
        return NULL;
    }

    // Detect head_dim from Q weight shape
    // GGUF shape: [ne0, ne1] where ne1 = total Q output dim = n_heads * head_dim
    ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) {
        int q_out = (int)gf->tensors[ti].shape[1]; // ne[1] = output dim
        m->head_dim = q_out / m->n_heads;
        m->kv_dim = m->head_dim * m->n_kv_heads;
        printf("gemma: head_dim=%d (from Q weight), kv_dim=%d\n", m->head_dim, m->kv_dim);
    }

    return m;
}

static void gemma_free(gemma_model* m) {
    if (!m) return;
    free(m->tok_emb); free(m->out_norm);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->layers[l].attn_norm); free(m->layers[l].post_attn_norm);
        free(m->layers[l].wq); free(m->layers[l].wk);
        free(m->layers[l].wv); free(m->layers[l].wo);
        free(m->layers[l].q_norm); free(m->layers[l].k_norm);
        free(m->layers[l].ffn_norm); free(m->layers[l].post_ffn_norm);
        free(m->layers[l].wgate); free(m->layers[l].wup); free(m->layers[l].wdown);
    }
    free(m);
}

// ── KV Cache ─────────────────────────────────────────────────────────────────

typedef struct {
    float *k;  // [n_layers, max_seq, kv_dim]
    float *v;  // [n_layers, max_seq, kv_dim]
    int max_seq;
    int n_layers;
    int kv_dim;
} kv_cache;

static kv_cache* kv_cache_new(int n_layers, int max_seq, int kv_dim) {
    kv_cache* kv = (kv_cache*)calloc(1, sizeof(kv_cache));
    kv->k = (float*)calloc((long)n_layers * max_seq * kv_dim, sizeof(float));
    kv->v = (float*)calloc((long)n_layers * max_seq * kv_dim, sizeof(float));
    kv->max_seq = max_seq;
    kv->n_layers = n_layers;
    kv->kv_dim = kv_dim;
    return kv;
}

static void kv_cache_free(kv_cache* kv) {
    if (!kv) return;
    free(kv->k); free(kv->v); free(kv);
}

// ── Forward pass (single token, uses KV cache) ──────────────────────────────

static void gemma_forward(gemma_model* m, kv_cache* kv, int token, int pos, float* logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn;
    float eps = m->rms_eps;
    int gqa_ratio = H / KV;  // how many Q heads per KV head

    // Embedding + Gemma scaling
    float *x = (float*)calloc(E, sizeof(float));
    for (int i = 0; i < E; i++)
        x[i] = m->tok_emb[token * E + i] * sqrtf((float)E);

    float *xn = (float*)calloc(E, sizeof(float));
    int Q_DIM = H * HD;   // total Q output dimension
    float *q_all = (float*)calloc(Q_DIM, sizeof(float));
    float *k_new = (float*)calloc(KVD, sizeof(float));
    float *v_new = (float*)calloc(KVD, sizeof(float));
    float *attn_out = (float*)calloc(Q_DIM, sizeof(float));
    float *ffn_gate = (float*)calloc(FFN, sizeof(float));
    float *ffn_up = (float*)calloc(FFN, sizeof(float));
    float *ffn_out = (float*)calloc(E, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        // Pre-attention RMSNorm
        rmsnorm(xn, x, m->layers[l].attn_norm, E, eps);

        // Q, K, V projections
        mm_t(q_all, xn, m->layers[l].wq, 1, E, Q_DIM);
        mm_t(k_new, xn, m->layers[l].wk, 1, E, KVD);
        mm_t(v_new, xn, m->layers[l].wv, 1, E, KVD);

        // QK-norm (per-head RMSNorm on Q and K)
        if (m->layers[l].q_norm) {
            for (int h = 0; h < H; h++)
                rmsnorm(q_all + h*HD, q_all + h*HD, m->layers[l].q_norm, HD, eps);
        }
        if (m->layers[l].k_norm) {
            for (int h = 0; h < KV; h++)
                rmsnorm(k_new + h*HD, k_new + h*HD, m->layers[l].k_norm, HD, eps);
        }

        // RoPE on Q and K
        for (int h = 0; h < H; h++)
            rope(q_all + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++)
            rope(k_new + h*HD, pos, HD, m->rope_base);

        // Store K, V in cache
        long kv_base = (long)l * kv->max_seq * KVD;
        memcpy(kv->k + kv_base + pos * KVD, k_new, KVD * sizeof(float));
        memcpy(kv->v + kv_base + pos * KVD, v_new, KVD * sizeof(float));

        // Multi-head attention with GQA
        float scale = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, Q_DIM * sizeof(float));

        for (int h = 0; h < H; h++) {
            int kv_head = h / gqa_ratio;
            float *q = q_all + h * HD;

            // Attention scores over all cached positions
            float *scores = (float*)calloc(pos + 1, sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *kj = kv->k + kv_base + j * KVD + kv_head * HD;
                float dot = 0;
                for (int d = 0; d < HD; d++) dot += q[d] * kj[d];
                scores[j] = dot * scale;
            }
            softmax(scores, pos + 1);

            // Weighted sum of values
            float *out_h = attn_out + h * HD;
            for (int j = 0; j <= pos; j++) {
                float *vj = kv->v + kv_base + j * KVD + kv_head * HD;
                for (int d = 0; d < HD; d++)
                    out_h[d] += scores[j] * vj[d];
            }
            free(scores);
        }

        // Output projection: attn_out[Q_DIM] → proj[E]
        float *proj = (float*)calloc(E, sizeof(float));
        mm_t(proj, attn_out, m->layers[l].wo, 1, Q_DIM, E);

        // Post-attention norm + residual
        if (m->layers[l].post_attn_norm) {
            rmsnorm(proj, proj, m->layers[l].post_attn_norm, E, eps);
        }
        for (int i = 0; i < E; i++) x[i] += proj[i];
        free(proj);

        // FFN: pre-norm
        rmsnorm(xn, x, m->layers[l].ffn_norm, E, eps);

        // SiLU-gated FFN: gate = silu(xn @ Wgate^T), up = xn @ Wup^T, out = (gate * up) @ Wdown^T
        mm_t(ffn_gate, xn, m->layers[l].wgate, 1, E, FFN);
        mm_t(ffn_up, xn, m->layers[l].wup, 1, E, FFN);
        for (int i = 0; i < FFN; i++) {
            float g = ffn_gate[i];
            ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i]; // silu(gate) * up
        }
        mm_t(ffn_out, ffn_gate, m->layers[l].wdown, 1, FFN, E);

        // Post-FFN norm + residual
        if (m->layers[l].post_ffn_norm) {
            rmsnorm(ffn_out, ffn_out, m->layers[l].post_ffn_norm, E, eps);
        }
        for (int i = 0; i < E; i++) x[i] += ffn_out[i];
    }

    // Final norm + logits
    rmsnorm(xn, x, m->out_norm, E, eps);

    // Gemma-3 ties embeddings: output = xn @ tok_emb^T
    mm_t(logits, xn, m->tok_emb, 1, E, m->vocab);

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
}

// ── Sampling ─────────────────────────────────────────────────────────────────

static int sample(float *logits, int n, float temp) {
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Simple BPE-style token decoder (reads tokenizer.json) ────────────────────

#define MAX_VOCAB 300000
#define MAX_TOK_LEN 256

static char** load_tokenizer(const char* dir, int* vocab_size) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE* f = fopen(path, "r");
    if (!f) {
        // Try same directory as model
        snprintf(path, sizeof(path), "%s/../tokenizer.json", dir);
        f = fopen(path, "r");
    }
    if (!f) { *vocab_size = 0; return NULL; }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json = (char*)malloc(sz + 1);
    fread(json, 1, sz, f);
    json[sz] = 0;
    fclose(f);

    // Quick & dirty: find "vocab" section, extract "token": id pairs
    // Gemma tokenizer.json has entries like: "content": "▁the", "id": 235
    char** tokens = (char**)calloc(MAX_VOCAB, sizeof(char*));
    int max_id = 0;

    // Find "added_tokens" or "model"."vocab"
    char* p = json;
    while ((p = strstr(p, "\"content\"")) != NULL) {
        p += 9; // skip "content"
        // Find the string value
        char* q = strchr(p, '"');
        if (!q) break;
        q++; // skip opening quote
        char* end = q;
        while (*end && !(*end == '"' && *(end-1) != '\\')) end++;
        int len = (int)(end - q);
        if (len >= MAX_TOK_LEN) len = MAX_TOK_LEN - 1;

        // Find "id": number
        char* id_str = strstr(end, "\"id\"");
        if (!id_str || id_str - end > 200) { p = end; continue; }
        id_str += 4;
        while (*id_str && (*id_str == ':' || *id_str == ' ')) id_str++;
        int id = atoi(id_str);

        if (id >= 0 && id < MAX_VOCAB) {
            tokens[id] = (char*)malloc(len + 1);
            // Handle escape sequences
            int j = 0;
            for (int i = 0; i < len && j < MAX_TOK_LEN - 1; i++) {
                if (q[i] == '\\' && i + 1 < len) {
                    i++;
                    if (q[i] == 'n') tokens[id][j++] = '\n';
                    else if (q[i] == 't') tokens[id][j++] = '\t';
                    else if (q[i] == '"') tokens[id][j++] = '"';
                    else if (q[i] == '\\') tokens[id][j++] = '\\';
                    else { tokens[id][j++] = '\\'; tokens[id][j++] = q[i]; }
                } else {
                    tokens[id][j++] = q[i];
                }
            }
            tokens[id][j] = 0;
            if (id > max_id) max_id = id;
        }
        p = end;
    }

    free(json);
    *vocab_size = max_id + 1;
    printf("tokenizer: %d tokens loaded\n", *vocab_size);
    return tokens;
}

// Decode token to string. Handles Gemma's ▁ (U+2581) → space mapping.
static void print_token(char** vocab, int id, int vocab_size) {
    if (!vocab || id < 0 || id >= vocab_size || !vocab[id]) {
        printf("[%d]", id);
        return;
    }
    char* s = vocab[id];
    // Replace ▁ (UTF-8: E2 96 81) with space
    while (*s) {
        if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
            putchar(' ');
            s += 3;
        } else {
            putchar(*s);
            s++;
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("notorch Gemma-3 inference\n");
        printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp]\n", argv[0]);
        return 1;
    }

    double t0 = now_ms();

    // Load GGUF
    printf("loading %s...\n", argv[1]);
    gguf_file* gf = gguf_open(argv[1]);
    if (!gf) return 1;

    // Load model
    gemma_model* model = gemma_load(gf);
    if (!model) { gguf_close(gf); return 1; }

    double load_ms = now_ms() - t0;
    printf("loaded in %.0f ms\n", load_ms);

    // Load tokenizer
    // Try directory of model file
    char dir[512];
    strncpy(dir, argv[1], sizeof(dir));
    char* last_slash = strrchr(dir, '/');
    if (last_slash) *last_slash = 0; else strcpy(dir, ".");

    int tok_vocab = 0;
    char** vocab = load_tokenizer(dir, &tok_vocab);

    // KV cache
    int max_seq = 512;  // reasonable for 8GB Mac
    kv_cache* kv = kv_cache_new(model->n_layers, max_seq, model->kv_dim);

    // Prompt tokens (simple: use byte-level + BOS)
    const char* user_prompt = argc > 2 ? argv[2] : "Hello, who are you?";
    // Gemma instruct format
    char prompt[2048];
    snprintf(prompt, sizeof(prompt), "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_prompt);
    // If user passes --raw, skip template
    if (argc > 2 && strcmp(argv[2], "--raw") == 0) {
        strcpy(prompt, "");
        user_prompt = "(raw BOS)";
    }
    int max_tokens = argc > 3 ? atoi(argv[3]) : 100;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.7f;

    // Simple prompt encoding: character-level fallback
    // Gemma uses BOS=2, EOS=1
    int tokens[512];
    int n_tokens = 0;
    // Gemma special tokens
    #define TOK_BOS 2
    #define TOK_EOS 1
    #define TOK_START_TURN 106
    #define TOK_END_TURN 107

    tokens[n_tokens++] = TOK_BOS;

    for (int i = 0; prompt[i] && n_tokens < max_seq - max_tokens; ) {
        // Check special tokens first
        if (strncmp(prompt + i, "<start_of_turn>", 15) == 0) {
            tokens[n_tokens++] = TOK_START_TURN; i += 15; continue;
        }
        if (strncmp(prompt + i, "<end_of_turn>", 13) == 0) {
            tokens[n_tokens++] = TOK_END_TURN; i += 13; continue;
        }
        if (strncmp(prompt + i, "<bos>", 5) == 0) { i += 5; continue; } // already added
        if (strncmp(prompt + i, "<eos>", 5) == 0) {
            tokens[n_tokens++] = TOK_EOS; i += 5; continue;
        }

        // Greedy vocab match
        int best_id = -1, best_len = 0;
        if (vocab) {
            for (int v = 0; v < tok_vocab && v < model->vocab; v++) {
                if (!vocab[v]) continue;
                int vlen = (int)strlen(vocab[v]);
                if (vlen > best_len && vlen <= 32 && strncmp(prompt + i, vocab[v], vlen) == 0) {
                    best_id = v;
                    best_len = vlen;
                }
                // Space → ▁ mapping
                if (prompt[i] == ' ' && vlen >= 4 &&
                    (unsigned char)vocab[v][0] == 0xE2 &&
                    (unsigned char)vocab[v][1] == 0x96 &&
                    (unsigned char)vocab[v][2] == 0x81) {
                    int match_len = vlen - 3 + 1;
                    if (match_len > best_len && match_len <= 32 &&
                        strncmp(prompt + i + 1, vocab[v] + 3, vlen - 3) == 0) {
                        best_id = v;
                        best_len = match_len;
                    }
                }
            }
        }
        if (best_id >= 0 && best_len > 0) {
            tokens[n_tokens++] = best_id;
            i += best_len;
        } else {
            // Byte fallback
            tokens[n_tokens++] = (unsigned char)prompt[i] + 3;
            i++;
        }
    }

    printf("\nprompt: \"%s\" (%d tokens)\n", prompt, n_tokens);
    printf("generating %d tokens (temp=%.2f)...\n\n", max_tokens, temp);

    // Process prompt tokens (prefill)
    double gen_start = now_ms();
    float *logits = (float*)calloc(model->vocab, sizeof(float));

    // Debug: print token IDs
    printf("token_ids: ");
    for (int i = 0; i < n_tokens && i < 20; i++) printf("%d ", tokens[i]);
    if (n_tokens > 20) printf("...");
    printf("\n");

    for (int i = 0; i < n_tokens; i++) {
        gemma_forward(model, kv, tokens[i], i, logits);
    }

    // Debug: print top-5 logits
    {
        int top5[5] = {0};
        for (int i = 0; i < model->vocab; i++)
            for (int j = 0; j < 5; j++)
                if (logits[i] > logits[top5[j]]) {
                    for (int k = 4; k > j; k--) top5[k] = top5[k-1];
                    top5[j] = i; break;
                }
        printf("top-5 logits: ");
        for (int j = 0; j < 5; j++) printf("[%d]=%.2f ", top5[j], logits[top5[j]]);
        printf("\n");
    }

    // Print user prompt
    printf("> %s\n", user_prompt);
    fflush(stdout);

    double prefill_ms = now_ms() - gen_start;

    // Generate
    int gen_tokens = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, model->vocab, temp);
        if (next == 1 || next == 107) break; // EOS or <end_of_turn>

        print_token(vocab, next, tok_vocab);
        fflush(stdout);
        gen_tokens++;

        int pos = n_tokens + step;
        if (pos >= max_seq - 1) break;
        gemma_forward(model, kv, next, pos, logits);
    }

    double total_ms = now_ms() - gen_start;
    double decode_ms = total_ms - prefill_ms;

    printf("\n\n── stats ──\n");
    printf("  prefill: %d tokens, %.0f ms (%.1f tok/s)\n",
           n_tokens, prefill_ms, n_tokens * 1000.0 / prefill_ms);
    printf("  decode:  %d tokens, %.0f ms (%.1f tok/s)\n",
           gen_tokens, decode_ms, gen_tokens > 0 ? gen_tokens * 1000.0 / decode_ms : 0);
    printf("  total:   %.0f ms\n", total_ms);

    // Cleanup
    free(logits);
    kv_cache_free(kv);
    gemma_free(model);
    if (vocab) {
        for (int i = 0; i < tok_vocab; i++) free(vocab[i]);
        free(vocab);
    }
    gguf_close(gf);

    return 0;
}
