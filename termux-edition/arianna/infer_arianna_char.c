/*
 * infer_arianna_char.c — char-level inference for the 9.5M Arianna LLaMA 3
 *
 * Companion to notorch/examples/train_10m_char.c (= device-1/training_kit/
 * train_10m_char.c in the umbrella repo). Loads a checkpoint produced by
 * that trainer and generates text.
 *
 * Architecture (must match train_10m_char.c exactly):
 *   dim=384, layers=6, heads=6, kv_heads=2 (GQA), vocab=88, ctx=256
 *   hidden=1024 (SwiGLU), RoPE theta=10000, RMSNorm
 *   ~9.5M parameters
 *
 * Build:
 *   cc -O2 -DUSE_BLAS -I../.. \
 *      -I/data/data/com.termux/files/usr/include/openblas \
 *      infer_arianna_char.c ../../notorch.c -lopenblas -lm \
 *      -o infer_arianna_char
 *   # or without BLAS:
 *   cc -O2 -I../.. infer_arianna_char.c ../../notorch.c -lm \
 *      -o infer_arianna_char
 *
 * Run:
 *   ./infer_arianna_char arianna_10k_char.bin ["prompt"] [max_tokens] [temp]
 *   defaults: max_tokens=200, temp=0.8, prompt = built-in sample
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define DIM       384
#define NLAYERS   6
#define NHEADS    6
#define NKV_HEADS 2
#define HEAD_DIM  (DIM / NHEADS)
#define KV_DIM    (NKV_HEADS * HEAD_DIM)
#define HIDDEN    1024
#define CTX       256
#define VOCAB     88

/* ── Vocabulary (must mirror train_10m_char.c) ── */

static char vocab_chars[VOCAB];
static int  char_to_id[256];
static const char* utf8_decode[] = {"\xC3\x97", "\xC3\xA0", "\xC3\xA9",
                                    "\xC3\xB6", "\xE2\x80\x94", "\xE2\x84\xA2"};

static void init_vocab(void) {
    memset(char_to_id, -1, sizeof(char_to_id));
    const char* ascii_order = "\n !\"$%&'()*+,-./"
                              "0123456789:;=?"
                              "ABCDEFGHIJKLMNOPQRSTUVWYZ"
                              "_abcdefghijklmnopqrstuvwxyz";
    int idx = 0;
    for (int i = 0; ascii_order[i] && idx < VOCAB; i++) {
        vocab_chars[idx] = ascii_order[i];
        char_to_id[(unsigned char)ascii_order[i]] = idx;
        idx++;
    }
    vocab_chars[82] = '*'; vocab_chars[83] = 'a'; vocab_chars[84] = 'e';
    vocab_chars[85] = 'o'; vocab_chars[86] = '-'; vocab_chars[87] = 'T';
}

static int encode_char(unsigned char c) {
    int id = char_to_id[c];
    return id >= 0 ? id : 1;  /* unknown → space */
}

/* Print a token id, expanding ids 82-87 back to multi-byte UTF-8. */
static void print_token(int id) {
    if (id >= 82 && id <= 87) { fputs(utf8_decode[id - 82], stdout); return; }
    char c = vocab_chars[id];
    if (c == '\n') { putchar('\n'); return; }
    if (c >= 32 && c < 127) putchar(c);
    /* anything else: skip silently — this only happens for unmapped slots */
}

/* ── Model (must mirror train_10m_char.c parameter order exactly) ── */

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo, *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static Model* model_alloc(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1   = nt_tensor_new(DIM);
        m->L[l].wq     = nt_tensor_new2d(DIM, DIM);
        m->L[l].wk     = nt_tensor_new2d(KV_DIM, DIM);
        m->L[l].wv     = nt_tensor_new2d(KV_DIM, DIM);
        m->L[l].wo     = nt_tensor_new2d(DIM, DIM);
        m->L[l].rms2   = nt_tensor_new(DIM);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_up   = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN);
    }
    m->rms_f = nt_tensor_new(DIM);
    m->head  = nt_tensor_new2d(VOCAB, DIM);
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

static int model_n_tensors(void) { return 1 + NLAYERS * 9 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1; p[i++]=m->L[l].wq; p[i++]=m->L[l].wk;
        p[i++]=m->L[l].wv; p[i++]=m->L[l].wo; p[i++]=m->L[l].rms2;
        p[i++]=m->L[l].w_gate; p[i++]=m->L[l].w_up; p[i++]=m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static int load_model(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -2;
    }
    nt_tensor** mp = model_param_array(m);
    for (int i = 0; i < expected; i++) {
        if (loaded[i]->len != mp[i]->len) {
            for (int j = 0; j < n_loaded; j++) nt_tensor_free(loaded[j]);
            free(loaded); free(mp); return -3;
        }
        memcpy(mp[i]->data, loaded[i]->data, mp[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(mp);
    return 0;
}

/* ── Forward (logits-only, no loss head) ── */

static int forward_logits(Model* m, const int* tokens) {
    int wte_i = nt_tape_param(m->wte);
    int li[NLAYERS][9];
    for (int l = 0; l < NLAYERS; l++) {
        li[l][0] = nt_tape_param(m->L[l].rms1);
        li[l][1] = nt_tape_param(m->L[l].wq);
        li[l][2] = nt_tape_param(m->L[l].wk);
        li[l][3] = nt_tape_param(m->L[l].wv);
        li[l][4] = nt_tape_param(m->L[l].wo);
        li[l][5] = nt_tape_param(m->L[l].rms2);
        li[l][6] = nt_tape_param(m->L[l].w_gate);
        li[l][7] = nt_tape_param(m->L[l].w_up);
        li[l][8] = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) tok_t->data[i] = (float)tokens[i];
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, DIM);
        int q = nt_rope(nt_seq_linear(li[l][1], xn, CTX), CTX, HEAD_DIM);
        int k = nt_rope(nt_seq_linear(li[l][2], xn, CTX), CTX, HEAD_DIM);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        int attn = nt_gqa_causal_attention(q, k, v, CTX, HEAD_DIM, NHEADS, NKV_HEADS);
        h = nt_add(h, nt_seq_linear(li[l][4], attn, CTX));
        xn = nt_seq_rmsnorm(h, li[l][5], CTX, DIM);
        int gate = nt_silu(nt_seq_linear(li[l][6], xn, CTX));
        int up   = nt_seq_linear(li[l][7], xn, CTX);
        int down = nt_seq_linear(li[l][8], nt_mul(gate, up), CTX);
        h = nt_add(h, down);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    return nt_seq_linear(head_i, hf, CTX);   /* [CTX, VOCAB] logits */
}

/* ── Temperature sampling ── */

static int sample_token(float* logits, float temp) {
    float scale = 1.0f / (temp > 1e-6f ? temp : 1e-6f);
    float mx = logits[0] * scale;
    for (int i = 1; i < VOCAB; i++) {
        float v = logits[i] * scale;
        if (v > mx) mx = v;
    }
    float sum = 0;
    float p[VOCAB];
    for (int i = 0; i < VOCAB; i++) {
        p[i] = expf(logits[i] * scale - mx);
        sum += p[i];
    }
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cum = 0;
    for (int i = 0; i < VOCAB; i++) {
        cum += p[i];
        if (cum >= r) return i;
    }
    return VOCAB - 1;
}

/* ── Generate ── */

static void generate(Model* m, const char* prompt, int max_new, float temp) {
    int ctx[CTX];
    int gen_len = 0;
    for (int i = 0; prompt[i] && gen_len < CTX/2; i++) {
        unsigned char c = (unsigned char)prompt[i];
        if (c < 0x80) {
            ctx[gen_len++] = encode_char(c);
        } else {
            /* simple UTF-8 rejection — fall back to space for unmapped multi-byte */
            ctx[gen_len++] = 1;
            while (prompt[i+1] && (((unsigned char)prompt[i+1]) & 0xC0) == 0x80) i++;
        }
    }

    fputs(prompt, stdout); fflush(stdout);

    nt_train_mode(0);
    int tokens[CTX];
    for (int s = 0; s < max_new; s++) {
        for (int i = 0; i < gen_len; i++) tokens[i] = ctx[i];
        for (int i = gen_len; i < CTX; i++) tokens[i] = 0;

        nt_tape_start();
        int logits_idx = forward_logits(m, tokens);
        nt_tape* tape = nt_tape_get();
        float* last_logits = tape->entries[logits_idx].output->data + (gen_len-1)*VOCAB;

        int next = sample_token(last_logits, temp);
        print_token(next);
        fflush(stdout);

        ctx[gen_len++] = next;
        nt_tape_clear();

        if (gen_len >= CTX - 1) break;
    }
    putchar('\n');
}

/* ── Main ── */

static const char* DEFAULT_PROMPT = "Q: What is the meaning of life?\nA: ";

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <model.bin> [\"prompt\"] [max_tokens=200] [temp=0.8]\n"
            "  model.bin   notorch checkpoint produced by train_10m_char.c\n"
            "  prompt      seed text (default: a built-in sample)\n"
            "  max_tokens  number of tokens to generate (default 200)\n"
            "  temp        sampling temperature (default 0.8)\n",
            argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt     = argc > 2 ? argv[2] : DEFAULT_PROMPT;
    int   max_tokens       = argc > 3 ? atoi(argv[3]) : 200;
    float temp             = argc > 4 ? (float)atof(argv[4]) : 0.8f;

    init_vocab();
    srand((unsigned)time(NULL));

    fprintf(stderr, "── arianna char-level inference ──\n");
    fprintf(stderr, "model: %s\n", model_path);
    fprintf(stderr, "arch:  dim=%d L=%d H=%d KV=%d hidden=%d ctx=%d vocab=%d\n",
            DIM, NLAYERS, NHEADS, NKV_HEADS, HIDDEN, CTX, VOCAB);
    fprintf(stderr, "gen:   max=%d temp=%.2f\n\n", max_tokens, temp);

    Model* m = model_alloc();
    int rc = load_model(m, model_path);
    if (rc != 0) {
        fprintf(stderr,
            "load_model failed (rc=%d). Expected %d tensors with the exact\n"
            "shape produced by train_10m_char.c (dim=%d, %d layers, KV=%d).\n",
            rc, model_n_tensors(), DIM, NLAYERS, NKV_HEADS);
        model_free(m);
        return 2;
    }

    generate(m, prompt, max_tokens, temp);

    model_free(m);
    return 0;
}
