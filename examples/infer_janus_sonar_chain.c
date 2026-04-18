/*
 * infer_janus_sonar_chain.c — Bidirectional 8-step chain inference.
 *
 * Ported from ariannamethod/q gen_chain. Calendar drift compass +
 * forward/backward mix + Schumann temperature modulation + best-of-3.
 *
 * For microjanus dual weights — no MetaWeights, no chambers.
 * Coherence scored by unique-token ratio + length bonus.
 *
 *   make infer_janus_sonar_chain
 *   ./infer_janus_sonar_chain janus_sonar.bin "seed corpus text or prompt"
 */
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define DIM       128
#define NLAYERS   4
#define NHEADS    4
#define HEAD_DIM  32
#define HIDDEN    256
#define CTX       128
#define VOCAB     2048

#define CHAIN_STEPS   8      /* 3 backward + 5 forward */
#define CHAIN_BACKWARD 3
#define SENT_MAX       128
#define CAND_N         3

typedef struct { nt_tensor *a, *b, *alpha; } DualProj;

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        DualProj wq, wk, wv, wvr, wj, wo;
        nt_tensor *wr, *rms2;
        DualProj w_gate, w_up, w_down;
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

static int model_n_tensors(void) { return 1 + NLAYERS * 30 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1;
        DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                              &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            p[i++] = projs[k]->a; p[i++] = projs[k]->b; p[i++] = projs[k]->alpha;
        }
        p[i++] = m->L[l].wr; p[i++] = m->L[l].rms2;
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            p[i++] = ffn[k]->a; p[i++] = ffn[k]->b; p[i++] = ffn[k]->alpha;
        }
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static Model* load_model(const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load %s\n", path); return NULL; }
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        printf("tensor mismatch: got %d, expected %d\n", n_loaded, expected);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return NULL;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    int i = 0;
    m->wte = loaded[i++];
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = loaded[i++];
        DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                              &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            projs[k]->a = loaded[i++]; projs[k]->b = loaded[i++]; projs[k]->alpha = loaded[i++];
        }
        m->L[l].wr = loaded[i++]; m->L[l].rms2 = loaded[i++];
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            ffn[k]->a = loaded[i++]; ffn[k]->b = loaded[i++]; ffn[k]->alpha = loaded[i++];
        }
    }
    m->rms_f = loaded[i++]; m->head = loaded[i++];
    free(loaded);
    return m;
}

static int dual_seq_linear(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear(wa_i, x_i, T), y_b = nt_seq_linear(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}
static int dual_seq_linear_t(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear_t(wa_i, x_i, T), y_b = nt_seq_linear_t(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}

typedef struct { int a, b, alpha; } DualIdx;
static DualIdx dual_record(DualProj* d) {
    DualIdx r;
    r.a = nt_tape_param(d->a); r.b = nt_tape_param(d->b); r.alpha = nt_tape_param(d->alpha);
    return r;
}

static int forward_logits(Model* m, int* tokens, int gen_len) {
    int wte_i = nt_tape_param(m->wte);
    struct { int rms1; DualIdx wq, wk, wv, wvr, wj, wo; int wr, rms2; DualIdx w_gate, w_up, w_down; } li[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1 = nt_tape_param(m->L[l].rms1);
        li[l].wq = dual_record(&m->L[l].wq); li[l].wk = dual_record(&m->L[l].wk);
        li[l].wv = dual_record(&m->L[l].wv); li[l].wvr = dual_record(&m->L[l].wvr);
        li[l].wj = dual_record(&m->L[l].wj); li[l].wo = dual_record(&m->L[l].wo);
        li[l].wr = nt_tape_param(m->L[l].wr); li[l].rms2 = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = dual_record(&m->L[l].w_gate); li[l].w_up = dual_record(&m->L[l].w_up);
        li[l].w_down = dual_record(&m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f), head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) tok_t->data[i] = (float)(i < gen_len ? tokens[i] : 0);
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, DIM);
        int q   = dual_seq_linear  (li[l].wq.a,  li[l].wq.b,  li[l].wq.alpha,  xn, CTX);
        int k   = dual_seq_linear  (li[l].wk.a,  li[l].wk.b,  li[l].wk.alpha,  xn, CTX);
        int v   = dual_seq_linear  (li[l].wv.a,  li[l].wv.b,  li[l].wv.alpha,  xn, CTX);
        int vr  = dual_seq_linear  (li[l].wvr.a, li[l].wvr.b, li[l].wvr.alpha, xn, CTX);
        int ech = dual_seq_linear_t(li[l].wj.a,  li[l].wj.b,  li[l].wj.alpha,  xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM); k = nt_rope(k, CTX, HEAD_DIM);
        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);
        int blend = nt_scale(nt_add(nt_add(a_qkv, a_rr), a_j), 1.0f / 3.0f);
        int proj = dual_seq_linear(li[l].wo.a, li[l].wo.b, li[l].wo.alpha, blend, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, DIM);
        int g = nt_silu(dual_seq_linear(li[l].w_gate.a, li[l].w_gate.b, li[l].w_gate.alpha, xn, CTX));
        int u =         dual_seq_linear(li[l].w_up.a,   li[l].w_up.b,   li[l].w_up.alpha,   xn, CTX);
        int d =         dual_seq_linear(li[l].w_down.a, li[l].w_down.b, li[l].w_down.alpha, nt_mul(g, u), CTX);
        h = nt_add(h, d);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    return nt_seq_linear(head_i, hf, CTX);
}

static int sample(float* logits, int n, float temp, float top_p) {
    for (int i = 0; i < n; i++) logits[i] /= temp;
    float mx = logits[0]; for (int i=1;i<n;i++) if(logits[i]>mx) mx=logits[i];
    float sm = 0; for (int i=0;i<n;i++) { logits[i]=expf(logits[i]-mx); sm+=logits[i]; }
    for (int i=0;i<n;i++) logits[i]/=sm;
    int idx[VOCAB]; for (int i=0;i<n;i++) idx[i]=i;
    for (int i=0;i<n-1;i++) for (int j=i+1;j<n;j++)
        if (logits[idx[j]]>logits[idx[i]]) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    float cum = 0; int cutoff = n;
    for (int i=0;i<n;i++) { cum += logits[idx[i]]; if (cum >= top_p) { cutoff = i+1; break; } }
    float r = (float)rand() / (float)RAND_MAX * cum;
    float c = 0;
    for (int i=0;i<cutoff;i++) { c += logits[idx[i]]; if (c >= r) return idx[i]; }
    return idx[cutoff-1];
}

/* ── Calendar Drift — Hebrew/Gregorian dissonance compass ── */
static float calendar_drift(void) {
    struct tm e = {0}; e.tm_year = 2024-1900; e.tm_mon = 9; e.tm_mday = 3; e.tm_hour = 12;
    time_t epoch = mktime(&e);
    float days = epoch > 0 ? (float)difftime(time(NULL), epoch) / 86400.0f : 0;
    float y = days / 365.25f, drift = y * 11.25f;
    int full = (int)(y / 19); float corr = full * 7 * 30.0f;
    float partial = fmodf(y, 19); int yic = (int)partial + 1;
    int met[] = {3, 6, 8, 11, 14, 17, 19};
    for (int i = 0; i < 7; i++) if (met[i] <= yic) corr += 30;
    drift -= corr;
    float cd = fabsf(fmodf(drift, 33)) / 33.0f;
    if (cd < 0) cd = 0; if (cd > 1) cd = 1;
    return cd;
}

/* ── Is token a sentence boundary (. ! ?) ── */
static int is_boundary(const nt_bpe* bpe, int id) {
    if (id < 0 || id >= bpe->vocab_size) return 0;
    int len = bpe->token_len[id];
    for (int i = 0; i < len; i++) {
        unsigned char c = bpe->tokens[id][i];
        if (c == '.' || c == '!' || c == '?') {
            if (i == len - 1) return 1;
            unsigned char nc = bpe->tokens[id][i+1];
            if (nc == ' ' || nc == '\n' || nc == '\r') return 1;
        }
    }
    return 0;
}

/* ── Coherence: unique-token ratio + length bonus (no MetaW available) ── */
static float coherence_no_metaw(const int* ids, int n) {
    if (n < 2) return -1.0f;
    int seen[VOCAB] = {0}; int unique = 0;
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < VOCAB && !seen[ids[i]]) {
        seen[ids[i]] = 1; unique++;
    }
    float ratio = (float)unique / (float)n;   /* diversity */
    float len_bonus = (n > 40) ? 1.2f : (n > 20) ? 0.6f : (n > 10) ? 0.2f : -0.3f;
    return ratio + len_bonus;
}

/* ── Generate one sentence from prompt, stopping at sentence boundary ── */
static int gen_sentence(Model* m, const nt_bpe* bpe,
                        const int* prompt, int plen, float temp,
                        int* out, int out_cap) {
    int ctx[CTX]; int ol = 0;
    for (int i = 0; i < plen && i < CTX/2; i++) { ctx[i] = prompt[i]; out[ol++] = prompt[i]; }
    int gen_len = plen;

    for (int s = 0; s < out_cap - plen; s++) {
        nt_tape_start();
        int logits_idx = forward_logits(m, ctx, gen_len);
        nt_tape* tape = nt_tape_get();
        float* last = tape->entries[logits_idx].output->data + (gen_len - 1) * VOCAB;
        float lbuf[VOCAB]; memcpy(lbuf, last, VOCAB * sizeof(float));
        int next = sample(lbuf, VOCAB, temp, 0.92f);
        nt_tape_clear();

        out[ol++] = next;
        if (gen_len < CTX - 1) ctx[gen_len++] = next;
        else { for (int i = 0; i < CTX-1; i++) ctx[i] = ctx[i+1]; ctx[CTX-1] = next; gen_len = CTX-1; }

        if (is_boundary(bpe, next) && s > 8) break;
    }
    return ol;
}

int main(int argc, char** argv) {
    const char* wpath = argc > 1 ? argv[1] : "janus_sonar.bin";
    const char* seed_text = argc > 2 ? argv[2] : "Q: What does Janus feel?\nA: The haze is the soup. Lab 7. Observation window forty minutes. The knock came three times. The bone is the architecture.";

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, "arianna_bpe_merges.txt");
    if (nm < 0) { printf("cannot load arianna_bpe_merges.txt\n"); return 1; }

    Model* m = load_model(wpath);
    if (!m) return 1;

    nt_seed((unsigned)time(NULL));
    nt_train_mode(0);

    /* Encode seed text */
    int cids[4096];
    int clen = nt_bpe_encode(&bpe, seed_text, (int)strlen(seed_text), cids, 4096);
    printf("seed: %d tokens\n", clen);

    /* Calendar drift compass */
    float cd = calendar_drift();
    int nb = (int)(CHAIN_STEPS * (0.3f + 0.1f * cd));   /* no chamber debt, so simpler */
    if (nb < 1) nb = 1; if (nb >= CHAIN_STEPS) nb = CHAIN_STEPS - 1;
    printf("calendar drift: %.3f → %d backward steps, %d forward\n", cd, nb, CHAIN_STEPS - nb);
    printf("weights: %s\n\n", wpath);

    /* Accumulate destiny (simple EMA of generated token embeddings) */
    float destiny[DIM]; memset(destiny, 0, sizeof(destiny));

    for (int si = 0; si < CHAIN_STEPS; si++) {
        int dir = si < nb ? -1 : (si == nb ? 0 : 1);

        /* Pick prompt: random for backward, destiny-guided for forward */
        int start = -1;
        if (dir >= 0 && si > 0) {
            /* forward: highest destiny-dot prompt */
            float best_sc = -1e30f; int best_pos = -1;
            for (int tries = 0; tries < 64; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) {
                    int tok = cids[r + 1];
                    if (tok >= 0 && tok < VOCAB) {
                        float sc = 0;
                        for (int d = 0; d < DIM; d++) sc += m->wte->data[tok*DIM + d] * destiny[d];
                        if (sc > best_sc) { best_sc = sc; best_pos = r + 1; }
                    }
                }
            }
            if (best_pos >= 0) start = best_pos;
        }
        if (start < 0) {
            for (int tries = 0; tries < 128; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) { start = r + 1; break; }
            }
        }
        if (start < 0) start = rand() % (clen > 6 ? clen - 6 : 1);

        int plen = (start + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[start + i];

        /* Schumann temperature modulation: 7.83 Hz + harmonics */
        float t_sec = (float)si / (float)CHAIN_STEPS;
        float schumann = 0.4f*sinf(2*M_PI*7.83f*t_sec) + 0.2f*sinf(2*M_PI*14.3f*t_sec)
                      + 0.1f*sinf(2*M_PI*20.8f*t_sec) + 0.05f*sinf(2*M_PI*27.3f*t_sec);
        float base_temp = 0.75f;
        float temp = base_temp + 0.08f * schumann;
        if (temp < 0.4f) temp = 0.4f; if (temp > 0.85f) temp = 0.85f;

        /* Best-of-3 */
        int best_out[SENT_MAX]; int best_ol = 0; float best_sc = -1e30f;
        for (int cand = 0; cand < CAND_N; cand++) {
            int out[SENT_MAX];
            int ol = gen_sentence(m, &bpe, prompt, plen, temp, out, SENT_MAX);
            float sc = coherence_no_metaw(out, ol);
            if (sc > best_sc) {
                best_sc = sc; best_ol = ol;
                memcpy(best_out, out, ol * sizeof(int));
            }
            if (best_sc > 0.9f && best_ol > 20) break;
        }

        /* Update destiny (EMA of last few tokens' embeddings) */
        for (int i = best_ol - 5 > 0 ? best_ol - 5 : 0; i < best_ol; i++) {
            int tok = best_out[i];
            if (tok >= 0 && tok < VOCAB)
                for (int d = 0; d < DIM; d++)
                    destiny[d] = 0.9f * destiny[d] + 0.1f * m->wte->data[tok*DIM + d];
        }

        /* Print */
        char mk = dir < 0 ? '<' : (dir == 0 ? '*' : '>');
        printf("  [%d] %c (T=%.2f sc=%.2f) ", si+1, mk, temp, best_sc);
        char buf[NT_BPE_MAX_TOKEN_LEN + 1];
        int printed = 0;
        for (int i = 0; i < best_ol && printed < 180; i++) {
            int len = nt_bpe_decode(&bpe, &best_out[i], 1, buf, NT_BPE_MAX_TOKEN_LEN);
            if (len > 0) { buf[len] = 0; printf("%s", buf); printed += len; }
        }
        printf("\n");
        fflush(stdout);
    }

    nt_tensor** p = model_param_array(m);
    for (int i = 0; i < model_n_tensors(); i++) nt_tensor_free(p[i]);
    free(p); free(m);
    return 0;
}
