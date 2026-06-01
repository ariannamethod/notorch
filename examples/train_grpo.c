/*
 * train_grpo.c — Group Relative Policy Optimization on notorch
 *
 * GRPO (DeepSeek-R1 style): generate G responses per prompt, compute rewards,
 * normalize advantages within group, policy gradient with KL penalty.
 *
 * Per group: REINFORCE with a baseline. log π(y|x) = -T·CE(y), so for the
 * policy-gradient surrogate L = -E[A · log π(y|x)] the parameter gradient is
 *   ∂L/∂θ = -A · ∂log π/∂θ = A · ∂CE/∂θ.
 * We run forward_on_tape with the generated tokens as targets, backward to get
 * ∂CE/∂θ, scale the tape grads by the (group-normalized) advantage A, then take
 * a Chuck step — same grad-injection pattern dpo_step uses. An optional KL-vs-ref
 * penalty (BETA_KL) and a 1-step PPO clip on the importance ratio are folded in.
 *
 * No external reward model — uses rule-based scoring (length / repetition /
 * diversity), see compute_reward().
 *
 * Self-contained: with no args (or unreadable files) it random-inits the policy
 * (Xavier) and generates a tiny synthetic prompt set in-code, so the binary runs
 * a real RL demo with NO external files. The documented CLI still works.
 *
 * Build: make train_grpo
 * Run:   ./train_grpo                                  # synthetic demo, random init
 *        ./train_grpo <prompts.txt> <base_weights.bin> [steps] [lr]
 *
 * By Arianna Method. DOI: 10.5281/zenodo.19638451
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Model config — same as train_dpo.c */
#ifndef DIM
#define DIM       512
#endif
#ifndef NLAYERS
#define NLAYERS   8
#endif
#ifndef NHEADS
#define NHEADS    8
#endif
#ifndef NKV_HEADS
#define NKV_HEADS 4
#endif
#define HEAD_DIM  (DIM / NHEADS)
#define KV_DIM    (NKV_HEADS * HEAD_DIM)
#ifndef HIDDEN
#define HIDDEN    (DIM * 4)
#endif
#ifndef CTX
#define CTX       256
#endif
#ifndef VOCAB
#define VOCAB     6400
#endif

/* GRPO config */
#ifndef NUM_GENERATIONS
#define NUM_GENERATIONS 4    /* responses per prompt */
#endif
#ifndef MAX_GEN_LEN
#define MAX_GEN_LEN     128  /* max tokens to generate */
#endif
#define EPSILON         0.2f /* PPO clip */
#define BETA_KL         0.1f /* KL penalty */
#define GEN_TEMP        1.0f /* sampling temperature */
#define GEN_TOP_P       0.95f
#define EOS_TOKEN       2    /* synthetic EOS id (also stops on this in CLI mode) */
#define MAX_PROMPT_LEN  16
#define MAX_PROMPTS     64
#define LOG_EVERY       1
#define CKPT_EVERY      50

/* ── Model struct (identical to train_dpo.c) ── */

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *qnorm, *knorm, *wo;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static int model_n_tensors(void) { return 1 + NLAYERS * 11 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1; p[i++]=m->L[l].wq; p[i++]=m->L[l].wk;
        p[i++]=m->L[l].wv; p[i++]=m->L[l].qnorm; p[i++]=m->L[l].knorm;
        p[i++]=m->L[l].wo; p[i++]=m->L[l].rms2;
        p[i++]=m->L[l].w_gate; p[i++]=m->L[l].w_up; p[i++]=m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static Model* model_new(void) {
    Model* m = calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->wte, VOCAB, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq, DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(KV_DIM, DIM); nt_tensor_xavier(m->L[l].wk, DIM, KV_DIM);
        m->L[l].wv = nt_tensor_new2d(KV_DIM, DIM); nt_tensor_xavier(m->L[l].wv, DIM, KV_DIM);
        m->L[l].qnorm = nt_tensor_new(HEAD_DIM); nt_tensor_fill(m->L[l].qnorm, 1.0f);
        m->L[l].knorm = nt_tensor_new(HEAD_DIM); nt_tensor_fill(m->L[l].knorm, 1.0f);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo, DIM, DIM);
        m->L[l].rms2 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_gate, DIM, HIDDEN);
        m->L[l].w_up = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_up, DIM, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN); nt_tensor_xavier(m->L[l].w_down, HIDDEN, DIM);
    }
    m->rms_f = nt_tensor_new(DIM); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->head, DIM, VOCAB);
    return m;
}

static Model* model_clone(Model* src) {
    Model* m = calloc(1, sizeof(Model));
    nt_tensor** sp = model_param_array(src);
    int n = model_n_tensors();
    for (int i = 0; i < n; i++) {
        nt_tensor* t = nt_tensor_new(sp[i]->len);
        t->ndim = sp[i]->ndim;
        memcpy(t->shape, sp[i]->shape, sizeof(sp[i]->shape));
        memcpy(t->stride, sp[i]->stride, sizeof(sp[i]->stride));
        memcpy(t->data, sp[i]->data, sp[i]->len * sizeof(float));
        /* assign in order */
        if (i == 0) m->wte = t;
        else if (i == n-2) m->rms_f = t;
        else if (i == n-1) m->head = t;
        else {
            int li = (i - 1) / 11, fi = (i - 1) % 11;
            switch(fi) {
                case 0: m->L[li].rms1=t; break; case 1: m->L[li].wq=t; break;
                case 2: m->L[li].wk=t; break; case 3: m->L[li].wv=t; break;
                case 4: m->L[li].qnorm=t; break; case 5: m->L[li].knorm=t; break;
                case 6: m->L[li].wo=t; break; case 7: m->L[li].rms2=t; break;
                case 8: m->L[li].w_gate=t; break; case 9: m->L[li].w_up=t; break;
                case 10: m->L[li].w_down=t; break;
            }
        }
    }
    free(sp);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].qnorm);
        nt_tensor_free(m->L[l].knorm); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

/* ── Reward function (rule-based, no external model) ──
 *
 * GOOD_TOKEN shaping: the length/diversity terms below are already near-max for
 * a random-init policy (uniform sampling over VOCAB gives in-band length + high
 * diversity by chance), so on their own they leave no headroom and a smoke run
 * looks flat. The GOOD_TOKEN term gives the demo a *learnable* target the random
 * policy does NOT already max: reward ∝ fraction of generated tokens equal to a
 * fixed id. REINFORCE then has a clear gradient — emit GOOD_TOKEN more often —
 * which is exactly what proves the rollout→advantage→Chuck path trains. Real
 * alignment runs replace this with the task reward; the length/rep/diversity
 * terms stay as the documented rule-based base.                                */
#ifndef GOOD_TOKEN
#define GOOD_TOKEN 7
#endif

static float compute_reward(int* tokens, int len) {
    float reward = 0;
    /* Length reward */
    if (len >= 5 && len <= 200) reward += 0.5f;
    else reward -= 0.5f;
    /* Repetition penalty: count repeated trigrams */
    int reps = 0, total = 0;
    for (int i = 0; i + 2 < len; i++) {
        total++;
        for (int j = i + 3; j + 2 < len; j++)
            if (tokens[i]==tokens[j] && tokens[i+1]==tokens[j+1] && tokens[i+2]==tokens[j+2])
                { reps++; break; }
    }
    if (total > 0) reward -= 0.5f * (float)reps / total;
    /* Diversity: unique tokens / total */
    int unique[VOCAB]; memset(unique, 0, sizeof(unique));
    for (int i = 0; i < len; i++) if (tokens[i] < VOCAB) unique[tokens[i]] = 1;
    int u = 0; for (int i = 0; i < VOCAB; i++) u += unique[i];
    reward += 0.3f * (float)u / (len + 1);
    /* Learnable target term (see header): fraction of tokens == GOOD_TOKEN. */
    int good = 0;
    for (int i = 0; i < len; i++) if (tokens[i] == GOOD_TOKEN) good++;
    reward += 1.0f * (float)good / (len + 1);
    return reward;
}

/* ── GRPO advantage: normalize rewards within group ── */

static void compute_advantages(float* rewards, float* advantages, int G) {
    float mean = 0, var = 0;
    for (int i = 0; i < G; i++) mean += rewards[i];
    mean /= G;
    for (int i = 0; i < G; i++) var += (rewards[i] - mean) * (rewards[i] - mean);
    float std = sqrtf(var / G + 1e-4f);
    for (int i = 0; i < G; i++) advantages[i] = (rewards[i] - mean) / std;
}

/* ── Tape forward to logits (returns logits tape index; CE not appended) ──
 * Identical transformer body to train_dpo.c's forward_on_tape, but stops at
 * the LM head so the caller can read per-position logits for sampling.        */

static int forward_logits_on_tape(Model* m, int* tok, int T) {
    nt_tensor** params = model_param_array(m);
    int n = model_n_tensors();
    int* pi_ids = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        pi_ids[i] = nt_tape_param(params[i]);
    nt_tape_no_decay(pi_ids[0]); /* embedding — no weight decay */

    nt_tensor* tok_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) tok_t->data[t] = (float)tok[t];
    int tok_idx = nt_tape_record(tok_t, 0, -1, -1, 0);
    nt_tensor_free(tok_t);

    int pi = 0;
    int h = nt_seq_embedding(pi_ids[pi++], -1, tok_idx, T, DIM);

    for (int l = 0; l < NLAYERS; l++) {
        int rms1=pi_ids[pi++], wq=pi_ids[pi++], wk=pi_ids[pi++], wv=pi_ids[pi++];
        int qn=pi_ids[pi++], kn=pi_ids[pi++], wo=pi_ids[pi++], rms2=pi_ids[pi++];
        int wg=pi_ids[pi++], wu=pi_ids[pi++], wd=pi_ids[pi++];

        int xn = nt_seq_rmsnorm(h, rms1, T, DIM);
        int q = nt_seq_linear(wq, xn, T);
        int k = nt_seq_linear(wk, xn, T);
        int v = nt_seq_linear(wv, xn, T);
        q = nt_seq_rmsnorm(q, qn, T * NHEADS, HEAD_DIM);
        k = nt_seq_rmsnorm(k, kn, T * NKV_HEADS, HEAD_DIM);
        q = nt_rope(q, T, HEAD_DIM);
        k = nt_rope(k, T, HEAD_DIM);
        int attn = nt_gqa_causal_attention(q, k, v, T, HEAD_DIM, NHEADS, NKV_HEADS);
        h = nt_add(h, nt_seq_linear(wo, attn, T));

        xn = nt_seq_rmsnorm(h, rms2, T, DIM);
        int gate = nt_silu(nt_seq_linear(wg, xn, T));
        int up = nt_seq_linear(wu, xn, T);
        h = nt_add(h, nt_seq_linear(wd, nt_mul(gate, up), T));
    }

    int rmsf = pi_ids[pi++], head_p = pi_ids[pi++];
    int hf = nt_seq_rmsnorm(h, rmsf, T, DIM);
    int logits = nt_seq_linear(head_p, hf, T);

    free(params); free(pi_ids);
    return logits;
}

/* ── Full forward + CE loss (same as train_dpo's forward_on_tape) ── */

static int forward_on_tape(Model* m, int* tok, int* tgt, int T) {
    int logits = forward_logits_on_tape(m, tok, T);
    nt_tensor* tgt_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) tgt_t->data[t] = (float)tgt[t];
    int tgt_idx = nt_tape_record(tgt_t, 0, -1, -1, 0);
    nt_tensor_free(tgt_t);
    return nt_seq_cross_entropy(logits, tgt_idx, T, VOCAB);
}

static float read_tape_scalar(int idx) {
    return nt_tape_get()->entries[idx].output->data[0];
}

/* ── Hand-rolled top-p sampler (template: infer_janus.c sample_top_p) ──
 * Samples in-place over a copy so the tape tensor is left untouched.          */

static int sample_top_p(const float* logits_row, int V, float temp, float top_p) {
    float* p = malloc(V * sizeof(float));
    float mx = logits_row[0];
    for (int i = 1; i < V; i++) if (logits_row[i] > mx) mx = logits_row[i];
    float sum = 0;
    for (int i = 0; i < V; i++) { p[i] = expf((logits_row[i] - mx) / temp); sum += p[i]; }
    for (int i = 0; i < V; i++) p[i] /= sum;

    /* nucleus: keep mass up to top_p by descending prob (simple selection) */
    float kept = 0; float thresh = 0;
    /* find the prob value at which cumulative mass crosses top_p */
    float* sorted = malloc(V * sizeof(float));
    memcpy(sorted, p, V * sizeof(float));
    for (int a = 0; a < V; a++)
        for (int b = a + 1; b < V; b++)
            if (sorted[b] > sorted[a]) { float t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }
    for (int i = 0; i < V; i++) { kept += sorted[i]; if (kept >= top_p) { thresh = sorted[i]; break; } }
    free(sorted);

    float renorm = 0;
    for (int i = 0; i < V; i++) { if (p[i] < thresh) p[i] = 0; renorm += p[i]; }
    if (renorm <= 0) { free(p); /* fall back to argmax */
        int best = 0; for (int i = 1; i < V; i++) if (logits_row[i] > logits_row[best]) best = i; return best; }
    float r = (float)rand() / (float)RAND_MAX * renorm;
    float cum = 0; int out = 0;
    for (int i = 0; i < V; i++) { cum += p[i]; if (cum >= r) { out = i; break; } out = i; }
    free(p);
    return out;
}

/* ── Rollout: autoregressively sample one completion from the policy ──
 * Returns generated length (>=1). Fills `out` (the FULL sequence: prompt +
 * generation). `gen_start` is the first generated index. Each step runs a fresh
 * forward over the current sequence (no KV cache — clearest reference form),
 * reads the last-position logit row, samples, appends. Stops on EOS or cap.    */

static int rollout(Model* policy, const int* prompt, int prompt_len,
                   int* out, int max_total) {
    int T = prompt_len;
    for (int i = 0; i < prompt_len; i++) out[i] = prompt[i];

    nt_train_mode(0);
    int target_len = prompt_len + MAX_GEN_LEN;
    if (target_len > max_total) target_len = max_total;
    if (target_len > CTX) target_len = CTX;

    while (T < target_len) {
        nt_tape_start();
        int logits_idx = forward_logits_on_tape(policy, out, T);
        nt_tensor* lt = nt_tape_get()->entries[logits_idx].output; /* [T, VOCAB] */
        /* copy last-position logit row out before clearing the tape */
        float* row = malloc(VOCAB * sizeof(float));
        memcpy(row, lt->data + (size_t)(T - 1) * VOCAB, VOCAB * sizeof(float));
        nt_tape_clear();

        int next = sample_top_p(row, VOCAB, GEN_TEMP, GEN_TOP_P);
        free(row);
        out[T++] = next;
        if (next == EOS_TOKEN) break;
    }
    nt_train_mode(1);
    return T;
}

/* ── KL(π||π_ref) over the generated tokens, sequence-mean ──
 * Cheap proxy used as a scalar penalty added into the advantage: forward both
 * policy and ref once over the full sequence, compare per-token CE on the
 * generated targets. logπ = -CE, so KL_token ≈ CE_ref - CE_pi (a positive
 * scalar when the policy has drifted toward higher-prob outputs than ref).
 * Honest simplification: this is the standard GRPO scalar-KL proxy, not the
 * full per-token softmax-KL, because notorch has no soft-target CE tape op.    */

static float kl_divergence_softmax(float ce_pi, float ce_ref) {
    return ce_ref - ce_pi;  /* ≈ Σ_t (logπ - logπ_ref) / T, GRPO scalar proxy */
}

/* ── GRPO step over one group of generations for a single prompt ──
 * grads is a scratch [n][len] buffer reused across calls. Returns mean reward. */

static float grpo_step(Model* policy, Model* ref,
                       int gens[][CTX], int gen_lens[], int prompt_len,
                       int G, float lr) {
    int n = model_n_tensors();
    nt_tensor** pp = model_param_array(policy);

    float rewards[NUM_GENERATIONS], advantages[NUM_GENERATIONS];
    float ce_pi[NUM_GENERATIONS], ce_ref[NUM_GENERATIONS];

    /* 1. Score each completion on the generated portion only. */
    for (int g = 0; g < G; g++) {
        int glen = gen_lens[g] - prompt_len;
        if (glen < 1) glen = 1;
        rewards[g] = compute_reward(gens[g] + prompt_len, glen);
    }
    compute_advantages(rewards, advantages, G);

    /* 2. Reference CE per generation (frozen, no grad) for the KL penalty. */
    nt_train_mode(0);
    for (int g = 0; g < G; g++) {
        int T = gen_lens[g];
        if (T < 2) { ce_ref[g] = 0; continue; }
        /* targets[t] = token at t+1; loss over t=0..T-2 (predict-next). */
        int tgt[CTX];
        for (int t = 0; t < T - 1; t++) tgt[t] = gens[g][t + 1];
        nt_tape_start();
        int loss_idx = forward_on_tape(ref, gens[g], tgt, T - 1);
        ce_ref[g] = read_tape_scalar(loss_idx);
        nt_tape_clear();
    }

    /* 3. Policy gradient: for each generation, backward CE then scale grads by
     *    the (advantage - β·KL) coefficient before accumulating. REINFORCE:
     *    ∂L/∂θ = -A·∂logπ/∂θ = A·∂CE/∂θ. PPO 1-step ratio = 1 here (on-policy,
     *    single update per rollout) so the clip is a no-op guard kept for shape. */
    nt_train_mode(1);

    float** accum = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) accum[i] = calloc(pp[i]->len, sizeof(float));

    float mean_reward = 0, surrogate = 0;
    for (int g = 0; g < G; g++) {
        mean_reward += rewards[g];
        int T = gen_lens[g];
        if (T < 2) continue;
        int tgt[CTX];
        for (int t = 0; t < T - 1; t++) tgt[t] = gens[g][t + 1];

        nt_tape_start();
        int loss_idx = forward_on_tape(policy, gens[g], tgt, T - 1);
        ce_pi[g] = read_tape_scalar(loss_idx);
        nt_tape_backward(loss_idx);

        float kl = kl_divergence_softmax(ce_pi[g], ce_ref[g]);
        /* coeff multiplies ∂CE/∂θ; advantage drives REINFORCE, KL pulls toward ref */
        float coeff = advantages[g] + BETA_KL * kl;
        /* PPO clip guard (ratio=1 on-policy → clamp coeff magnitude for stability) */
        float lim = (1.0f + EPSILON) * fabsf(advantages[g]) + BETA_KL * fabsf(kl) + 1e-3f;
        if (coeff >  lim) coeff =  lim;
        if (coeff < -lim) coeff = -lim;
        surrogate += -advantages[g] * (-(T - 1) * ce_pi[g]); /* -A·logπ, for logging */

        nt_tape* tape = nt_tape_get();
        for (int i = 0; i < n; i++)
            if (tape->entries[i].grad)
                for (int j = 0; j < pp[i]->len; j++)
                    accum[i][j] += coeff * tape->entries[i].grad->data[j];
        nt_tape_clear();
    }
    mean_reward /= G;
    surrogate /= G;

    /* 4. Inject the group-summed REINFORCE grads onto a fresh tape and step. */
    nt_tape_start();
    nt_tensor** pp2 = model_param_array(policy);
    for (int i = 0; i < n; i++) nt_tape_param(pp2[i]);
    nt_tape_no_decay(0);
    nt_tape* tape = nt_tape_get();
    for (int i = 0; i < n; i++) {
        if (!tape->entries[i].grad) {
            tape->entries[i].grad = nt_tensor_new(pp2[i]->len);
        }
        for (int j = 0; j < pp2[i]->len; j++)
            tape->entries[i].grad->data[j] = accum[i][j] / G;
    }
    nt_tape_clip_grads(1.0f);
    nt_tape_chuck_step(lr, -mean_reward); /* loss ≈ -reward so Chuck schedules on reward-up */
    nt_tape_clear();

    for (int i = 0; i < n; i++) free(accum[i]);
    free(accum); free(pp2); free(pp);
    (void)surrogate;
    return mean_reward;
}

/* ── Synthetic prompt set (used when no prompts file is given/readable) ──
 * Tiny fixed prompts in token-id space; the reward favours length-in-band +
 * token diversity + low repetition, so a random-init policy has clear signal. */

static int make_synthetic_prompts(int prompts[][MAX_PROMPT_LEN], int* lens) {
    int base[][6] = {
        {5, 11, 23, 7, 0, 0},
        {9, 3, 17, 0, 0, 0},
        {13, 4, 8, 21, 6, 0},
        {7, 19, 2 + 1, 10, 0, 0},
    };
    int plens[] = {4, 3, 5, 4};
    int np = 4;
    for (int i = 0; i < np; i++) {
        lens[i] = plens[i];
        for (int t = 0; t < plens[i]; t++) prompts[i][t] = base[i][t] % VOCAB;
        if (prompts[i][0] < 3) prompts[i][0] = 3; /* avoid EOS at start */
    }
    return np;
}

/* ── Load prompts file: one line of whitespace-separated token ids per prompt ── */

static int load_prompts(const char* path, int prompts[][MAX_PROMPT_LEN], int* lens) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int np = 0;
    char line[4096];
    while (np < MAX_PROMPTS && fgets(line, sizeof(line), f)) {
        int t = 0; char* tok = strtok(line, " \t\r\n");
        while (tok && t < MAX_PROMPT_LEN) {
            int id = atoi(tok);
            if (id < 0) id = 0;
            if (id >= VOCAB) id = VOCAB - 1;
            prompts[np][t++] = id;
            tok = strtok(NULL, " \t\r\n");
        }
        if (t > 0) lens[np++] = t;
    }
    fclose(f);
    return np;
}

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Main ── */

int main(int argc, char** argv) {
    /* All args optional — self-contained demo with no files. */
    const char* data_path    = argc > 1 ? argv[1] : NULL;
    const char* weights_path = argc > 2 ? argv[2] : NULL;
    int   max_steps = argc > 3 ? atoi(argv[3]) : 200;
    float lr        = argc > 4 ? atof(argv[4]) : 1e-4f;
    const char* ckpt_out = "grpo_policy.bin";

    srand(42);  /* determinism is load-bearing (notorch/CLAUDE.md) */

    printf("═══════════════════════════════════════════════════\n");
    printf("  notorch — GRPO training\n");
    printf("  Group Relative Policy Optimization (DeepSeek)\n");
    printf("  DIM=%d LAYERS=%d G=%d ε=%.2f β_kl=%.2f\n", DIM, NLAYERS, NUM_GENERATIONS, EPSILON, BETA_KL);
    printf("═══════════════════════════════════════════════════\n");

    int n = model_n_tensors();
    Model* policy = model_new();  /* Xavier random init */

    /* Optional weights load; fall back to random init if missing/mismatched. */
    if (weights_path) {
        int loaded_n = 0;
        nt_tensor** loaded = nt_load(weights_path, &loaded_n);
        if (loaded && loaded_n == n) {
            nt_tensor** pp = model_param_array(policy);
            for (int i = 0; i < n; i++)
                memcpy(pp[i]->data, loaded[i]->data, pp[i]->len * sizeof(float));
            free(pp);
            printf("  loaded policy from %s\n", weights_path);
        } else {
            printf("  could not load %s (got %d, need %d) — random init (Xavier)\n",
                   weights_path, loaded_n, n);
        }
    } else {
        printf("  no weights file — random init (Xavier)\n");
    }

    Model* ref = model_clone(policy);  /* frozen reference snapshot */
    printf("  cloned reference model (frozen)\n");

    /* Optional prompts; fall back to synthetic in-code set. */
    static int prompts[MAX_PROMPTS][MAX_PROMPT_LEN];
    int plens[MAX_PROMPTS];
    int np = -1;
    if (data_path) np = load_prompts(data_path, prompts, plens);
    if (np <= 0) {
        np = make_synthetic_prompts(prompts, plens);
        printf("  %s — using %d synthetic prompts\n",
               data_path ? "prompts file unreadable" : "no prompts file", np);
    } else {
        printf("  loaded %d prompts from %s\n", np, data_path);
    }
    printf("  lr=%.1e  steps=%d  gen_len≤%d\n\n", lr, max_steps, MAX_GEN_LEN);

    /* ── Training loop ── */
    static int gens[NUM_GENERATIONS][CTX];
    int gen_lens[NUM_GENERATIONS];
    double t0 = now_ms();
    float ema_reward = 0; int ema_init = 0;

    for (int step = 0; step < max_steps; step++) {
        int* prompt = prompts[step % np];
        int prompt_len = plens[step % np];

        /* Rollout: sample G completions from the policy for this prompt. */
        for (int g = 0; g < NUM_GENERATIONS; g++)
            gen_lens[g] = rollout(policy, prompt, prompt_len, gens[g], CTX);

        /* Score + group-normalize advantages + REINFORCE/Chuck update. */
        float mean_reward = grpo_step(policy, ref, gens, gen_lens, prompt_len,
                                      NUM_GENERATIONS, lr);

        if (!ema_init) { ema_reward = mean_reward; ema_init = 1; }
        else ema_reward = 0.9f * ema_reward + 0.1f * mean_reward;

        if (step % LOG_EVERY == 0 || step == max_steps - 1) {
            double dt = (now_ms() - t0) / 1000.0;
            printf("step %4d | reward %+.4f | ema %+.4f | %.1fs\n",
                   step, mean_reward, ema_reward, dt);
        }

        if ((step + 1) % CKPT_EVERY == 0) {
            nt_tensor** pp = model_param_array(policy);
            nt_save(ckpt_out, pp, n);
            free(pp);
            printf("  ckpt → %s (step %d)\n", ckpt_out, step + 1);
        }
    }

    /* Final checkpoint. */
    nt_tensor** pp = model_param_array(policy);
    nt_save(ckpt_out, pp, n);
    free(pp);
    printf("\n  final ckpt → %s\n", ckpt_out);

    model_free(policy); model_free(ref);
    return 0;
}
