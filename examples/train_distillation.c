/*
 * train_distillation.c — Knowledge Distillation on notorch
 *
 * Teacher → Student transfer via soft labels at temperature τ:
 *   L = α·τ²·KL(softmax(teacher/τ) || softmax(student/τ))
 *     + (1-α)·CE(student_logits, hard_labels)
 *
 * notorch has a HARD-target sequence cross-entropy tape op
 * (nt_seq_cross_entropy) but NO soft-CE / KL tape op. The soft term is
 * therefore handled by INJECTING the analytic KD gradient straight into
 * the student-logits tape entry's grad before backward — the same
 * manual grad-injection pattern train_dpo.c uses. The KD gradient of
 * α·τ²·KL(P_t || P_s) w.r.t. the student logits at position t is
 *     α·τ · (softmax(student_t/τ) − softmax(teacher_t/τ)) / T
 * (one τ from τ² cancels against the 1/τ chain-rule factor of the inner
 * softmax). This is the EXACT gradient — it transfers the teacher's full
 * distribution, not just its argmax. The hard-CE term rides the real
 * nt_seq_cross_entropy backward path. See inject_kd_grad() for the
 * single-pass combine.
 *
 * Teacher is a frozen notorch model of the SAME arch (loaded, or
 * random-init Xavier as a self-contained demo teacher). A real run would
 * point --teacher at a larger/better checkpoint.
 *
 * Self-contained: with NO args it random-inits both models and trains on
 * a tiny synthetic token dataset, so the binary is a runnable demo.
 *
 * Build: make train_distillation
 * Run:   ./train_distillation [data.bin] [teacher.bin] [student.bin] [steps] [lr] [temp]
 *        ./train_distillation                       # full synthetic demo
 *
 * data.bin format: raw int32 token stream (little-endian). Missing/failed
 * load → synthetic stream generated in-code.
 *
 * By Arianna Method. DOI: 10.5281/zenodo.19638451
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Student config */
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

/* Distillation params */
#ifndef TEMPERATURE
#define TEMPERATURE 3.0f   /* soft label temperature τ */
#endif
#ifndef ALPHA
#define ALPHA       0.7f   /* weight of KD soft loss vs hard CE */
#endif
#define LOG_EVERY   10
#define CKPT_EVERY  100
#define SEQ_LEN     32     /* tokens per training window */

/* ── Student / teacher model (same struct as DPO/GRPO) ── */

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
    /* Deep copy for frozen teacher / reference */
    Model* m = calloc(1, sizeof(Model));
    nt_tensor** sp = model_param_array(src);
    int n = model_n_tensors();
    nt_tensor** dp = malloc(n * sizeof(nt_tensor*));
    for (int i = 0; i < n; i++) {
        dp[i] = nt_tensor_new(sp[i]->len);
        dp[i]->ndim = sp[i]->ndim;
        memcpy(dp[i]->shape, sp[i]->shape, sizeof(sp[i]->shape));
        memcpy(dp[i]->stride, sp[i]->stride, sizeof(sp[i]->stride));
        memcpy(dp[i]->data, sp[i]->data, sp[i]->len * sizeof(float));
    }
    int j = 0;
    m->wte = dp[j++];
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = dp[j++]; m->L[l].wq = dp[j++]; m->L[l].wk = dp[j++];
        m->L[l].wv = dp[j++]; m->L[l].qnorm = dp[j++]; m->L[l].knorm = dp[j++];
        m->L[l].wo = dp[j++]; m->L[l].rms2 = dp[j++];
        m->L[l].w_gate = dp[j++]; m->L[l].w_up = dp[j++]; m->L[l].w_down = dp[j++];
    }
    m->rms_f = dp[j++]; m->head = dp[j++];
    free(sp); free(dp);
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

/* Load weights into model in-place; return 1 on success, 0 on any failure
 * (caller keeps the Xavier-init weights from model_new). */
static int model_load_inplace(Model* m, const char* path) {
    if (!path) return 0;
    int n = model_n_tensors();
    int loaded_n = 0;
    nt_tensor** loaded = nt_load(path, &loaded_n);
    if (!loaded || loaded_n != n) return 0;
    nt_tensor** pp = model_param_array(m);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (loaded[i]->len != pp[i]->len) { ok = 0; break; }
    }
    if (ok) {
        for (int i = 0; i < n; i++)
            memcpy(pp[i]->data, loaded[i]->data, pp[i]->len * sizeof(float));
    }
    for (int i = 0; i < loaded_n; i++) nt_tensor_free(loaded[i]);
    free(loaded);
    free(pp);
    return ok;
}

/* ── Tape-based forward (returns cross-entropy loss index on tape) ──
 * Identical to train_dpo.c / train_grpo.c. The student-logits tape entry
 * is loss_entry.parent1 (nt_seq_cross_entropy records logits as parent1). */

static int forward_on_tape(Model* m, int* tok, int* tgt, int T) {
    nt_tensor** params = model_param_array(m);
    int n = model_n_tensors();
    int* pi_ids = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        pi_ids[i] = nt_tape_param(params[i]);
    nt_tape_no_decay(pi_ids[0]); /* embedding — no weight decay */

    nt_tensor* tok_t = nt_tensor_new(T);
    nt_tensor* tgt_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) { tok_t->data[t] = (float)tok[t]; tgt_t->data[t] = (float)tgt[t]; }
    int tok_idx = nt_tape_record(tok_t, 0, -1, -1, 0);
    int tgt_idx = nt_tape_record(tgt_t, 0, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

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
    int loss_idx = nt_seq_cross_entropy(logits, tgt_idx, T, VOCAB);

    free(params); free(pi_ids);
    return loss_idx;
}

/* ── Read a tape entry's output / value ── */

static float read_tape_loss(int loss_idx) {
    nt_tape* tape = nt_tape_get();
    return tape->entries[loss_idx].output->data[0];
}

/* Copy a tape entry's output tensor data (e.g. logits, len T*V). */
static void read_tape_output(int idx, float* dst, int len) {
    nt_tape* tape = nt_tape_get();
    memcpy(dst, tape->entries[idx].output->data, len * sizeof(float));
}

/* ── KL divergence (scalar, for logging the soft-loss value) ── */

static float kl_divergence_softmax(const float* teacher_logits, const float* student_logits,
                                    int vocab, float temperature) {
    /* KL(P_teacher || P_student) where P = softmax(logits/τ), scaled by τ² */
    float t_max = teacher_logits[0], s_max = student_logits[0];
    for (int i = 1; i < vocab; i++) {
        if (teacher_logits[i] > t_max) t_max = teacher_logits[i];
        if (student_logits[i] > s_max) s_max = student_logits[i];
    }
    float t_sum = 0, s_sum = 0;
    float* t_soft = malloc(vocab * sizeof(float));
    float* s_soft = malloc(vocab * sizeof(float));
    for (int i = 0; i < vocab; i++) {
        t_soft[i] = expf((teacher_logits[i] - t_max) / temperature);
        s_soft[i] = expf((student_logits[i] - s_max) / temperature);
        t_sum += t_soft[i]; s_sum += s_soft[i];
    }
    for (int i = 0; i < vocab; i++) { t_soft[i] /= t_sum; s_soft[i] /= s_sum; }
    float kl = 0;
    for (int i = 0; i < vocab; i++) {
        if (t_soft[i] > 1e-10f)
            kl += t_soft[i] * logf(t_soft[i] / (s_soft[i] + 1e-10f));
    }
    free(t_soft); free(s_soft);
    return kl * temperature * temperature;
}

/* ── KD gradient injection ──
 *
 * After the student forward built the tape (loss_idx = hard CE), we want
 * to backprop the COMBINED loss
 *     L = α·τ²·KL(P_t || P_s) + (1-α)·CE_hard
 * in a single pass. nt_tape_backward(loss_idx) seeds loss.grad = 1.0 and
 * the CE op deposits the FULL hard-CE grad g_ce = (softmax(s) - onehot)/T
 * into the student-logits entry (weight 1.0). We want weight (1-α) there
 * plus α·KD. tape_acc_grad ACCUMULATES, so we pre-seed the logits entry
 * grad with
 *     g_inject = α·KD_grad − α·g_ce
 *              = α·τ·(softmax(s/τ) − softmax(t/τ))/T − α·(softmax(s) − onehot)/T
 * Then CE backward adds g_ce → total = α·KD + (1-α)·g_ce. Exact, single
 * pass, propagates through the whole network. (DPO uses the same manual
 * grad surgery on tape entries.)
 *
 * Returns the combined per-token soft-loss value (α·τ²·KL averaged) for
 * logging. Must be called AFTER forward and BEFORE nt_tape_backward, and
 * the logits entry's grad is created here so backward accumulates onto it.
 */
static float inject_kd_grad(int logits_idx, const float* teacher_logits,
                            const float* student_logits, const int* hard_tgt,
                            int T, int V, float alpha, float tau) {
    float* g = calloc((size_t)T * V, sizeof(float));
    float kl_total = 0;
    float* sp = malloc(V * sizeof(float)); /* softmax(student/τ) */
    float* tp = malloc(V * sizeof(float)); /* softmax(teacher/τ) */
    float* s1 = malloc(V * sizeof(float)); /* softmax(student) at τ=1 (hard CE grad) */

    for (int t = 0; t < T; t++) {
        const float* sl = student_logits + (size_t)t * V;
        const float* tl = teacher_logits + (size_t)t * V;

        /* softmax(student/τ) */
        float mx = sl[0] / tau;
        for (int j = 1; j < V; j++) { float z = sl[j] / tau; if (z > mx) mx = z; }
        float sum = 0;
        for (int j = 0; j < V; j++) { sp[j] = expf(sl[j] / tau - mx); sum += sp[j]; }
        for (int j = 0; j < V; j++) sp[j] /= sum;

        /* softmax(teacher/τ) */
        mx = tl[0] / tau;
        for (int j = 1; j < V; j++) { float z = tl[j] / tau; if (z > mx) mx = z; }
        sum = 0;
        for (int j = 0; j < V; j++) { tp[j] = expf(tl[j] / tau - mx); sum += tp[j]; }
        for (int j = 0; j < V; j++) tp[j] /= sum;

        /* softmax(student) at τ=1 — matches what CE backward uses */
        mx = sl[0];
        for (int j = 1; j < V; j++) if (sl[j] > mx) mx = sl[j];
        sum = 0;
        for (int j = 0; j < V; j++) { s1[j] = expf(sl[j] - mx); sum += s1[j]; }
        for (int j = 0; j < V; j++) s1[j] /= sum;

        int tgt = hard_tgt[t];
        if (tgt < 0 || tgt >= V) tgt = 0;

        /* KD grad of α·τ²·KL : α·τ·(sp − tp)/T
         * minus α × hard-CE grad : α·(s1 − onehot)/T  (cancels the α share
         * of the weight-1.0 grad CE backward will add). */
        float invT = 1.0f / (float)T;
        for (int j = 0; j < V; j++) {
            float g_kd = alpha * tau * (sp[j] - tp[j]) * invT;
            float g_ce = (s1[j] - (j == tgt ? 1.0f : 0.0f)) * invT;
            g[(size_t)t * V + j] = g_kd - alpha * g_ce;
        }

        /* logging: α·τ²·KL(P_t || P_s) for this position */
        float kl = 0;
        for (int j = 0; j < V; j++)
            if (tp[j] > 1e-10f) kl += tp[j] * logf(tp[j] / (sp[j] + 1e-10f));
        kl_total += kl * tau * tau;
    }

    /* Seed the logits grad tape entry so CE backward accumulates g_ce on
     * top (tape_acc_grad += onto an existing grad). Same direct tape-entry
     * grad surgery train_dpo.c performs after backward — done here BEFORE
     * backward because the entry's grad is NULL until something writes it. */
    nt_tape* tape = nt_tape_get();
    nt_tape_entry* le = &tape->entries[logits_idx];
    if (!le->grad) le->grad = nt_tensor_new((size_t)T * V);
    memcpy(le->grad->data, g, (size_t)T * V * sizeof(float));

    free(g); free(sp); free(tp); free(s1);
    return alpha * (kl_total / T);
}

/* ── Synthetic token dataset (self-contained demo) ──
 * Structured stream so there is a real pattern for the teacher to know and
 * the student to learn: token[i] depends on token[i-1] (a noisy bigram
 * lattice). Not random — random would make CE flat and nothing to distill. */
static int* make_synthetic_tokens(int count, unsigned seed) {
    int* toks = malloc(count * sizeof(int));
    unsigned x = seed ? seed : 1u;
    int prev = 1;
    for (int i = 0; i < count; i++) {
        x = x * 1664525u + 1013904223u;            /* LCG */
        int noise = (x >> 13) % 7;                 /* small jitter */
        int next = (prev * 3 + 7 + noise) % VOCAB; /* deterministic bigram + noise */
        if (next < 0) next += VOCAB;
        toks[i] = next;
        prev = next;
    }
    return toks;
}

/* Load a raw int32 token file; NULL on failure. Sets *count. */
static int* load_token_file(const char* path, int* count) {
    *count = 0;
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < (long)(2 * sizeof(int32_t))) { fclose(f); return NULL; }
    long n = bytes / sizeof(int32_t);
    int32_t* raw = malloc(n * sizeof(int32_t));
    size_t got = fread(raw, sizeof(int32_t), n, f);
    fclose(f);
    if (got != (size_t)n) { free(raw); return NULL; }
    int* toks = malloc(n * sizeof(int));
    for (long i = 0; i < n; i++) {
        int v = (int)raw[i];
        if (v < 0 || v >= VOCAB) v = ((v % VOCAB) + VOCAB) % VOCAB;
        toks[i] = v;
    }
    free(raw);
    *count = (int)n;
    return toks;
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Main ── */

int main(int argc, char** argv) {
    /* All args OPTIONAL — self-contained demo when omitted. */
    const char* data_path    = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : NULL;
    const char* teacher_path = (argc > 2 && strcmp(argv[2], "-") != 0) ? argv[2] : NULL;
    const char* student_path = (argc > 3 && strcmp(argv[3], "-") != 0) ? argv[3] : NULL;
    int   max_steps = argc > 4 ? atoi(argv[4]) : 300;
    float lr        = argc > 5 ? (float)atof(argv[5]) : 1e-4f;
    float tau       = argc > 6 ? (float)atof(argv[6]) : TEMPERATURE;
    float alpha     = ALPHA;
    const char* out_path = student_path ? student_path : "distill_student.bin";

    srand(42);

    printf("═══════════════════════════════════════════════════\n");
    printf("  notorch — Knowledge Distillation\n");
    printf("  Teacher→Student via KL soft labels (Hinton 2015)\n");
    printf("  Student: DIM=%d L=%d H=%d KV=%d VOCAB=%d\n", DIM, NLAYERS, NHEADS, NKV_HEADS, VOCAB);
    printf("  τ=%.1f α=%.2f lr=%.1e steps=%d seq=%d\n", tau, alpha, lr, max_steps, SEQ_LEN);
    printf("═══════════════════════════════════════════════════\n");

    /* ── Teacher (frozen). Load if given; else random-init demo teacher. ── */
    Model* teacher = model_new();
    if (teacher_path && model_load_inplace(teacher, teacher_path))
        printf("  teacher: loaded %s (frozen)\n", teacher_path);
    else
        printf("  teacher: random-init Xavier (frozen demo teacher)%s\n",
               teacher_path ? " [load failed]" : "");

    /* ── Student. Load if given; else random-init (will learn). ── */
    Model* student = model_new();
    if (student_path && model_load_inplace(student, student_path))
        printf("  student: loaded %s\n", student_path);
    else
        printf("  student: random-init Xavier%s\n",
               student_path ? " [load failed]" : "");

    /* ── Dataset: real token file, else synthetic. ── */
    int n_toks = 0;
    int* toks = load_token_file(data_path, &n_toks);
    if (toks) {
        printf("  data: %s (%d tokens)\n", data_path, n_toks);
    } else {
        n_toks = SEQ_LEN * 64 + 1;
        toks = make_synthetic_tokens(n_toks, 1337u);
        printf("  data: synthetic bigram stream (%d tokens)%s\n",
               n_toks, data_path ? " [load failed]" : "");
    }
    if (n_toks < SEQ_LEN + 1) {
        fprintf(stderr, "  dataset too small (%d < %d); aborting\n", n_toks, SEQ_LEN + 1);
        free(toks); model_free(teacher); model_free(student); return 1;
    }
    int n_windows = (n_toks - 1) / SEQ_LEN;
    printf("  windows: %d × %d tokens\n", n_windows, SEQ_LEN);
    printf("───────────────────────────────────────────────────\n");

    nt_tensor** student_params = model_param_array(student);
    int n_params = model_n_tensors();

    int T = SEQ_LEN;
    int* in_tok  = malloc(T * sizeof(int));
    int* tgt_tok = malloc(T * sizeof(int));
    float* teacher_logits = malloc((size_t)T * VOCAB * sizeof(float));
    float* student_logits = malloc((size_t)T * VOCAB * sizeof(float));

    double t0 = now_ms();
    float first_loss = 0, last_loss = 0;

    for (int step = 0; step < max_steps; step++) {
        /* Pick a window (round-robin over the stream). */
        int w = step % n_windows;
        int base = w * T;
        for (int t = 0; t < T; t++) {
            in_tok[t]  = toks[base + t];
            tgt_tok[t] = toks[base + t + 1];
        }

        /* 1. Teacher forward — NO grad, read per-position logits. */
        nt_train_mode(0);
        nt_tape_start();
        int t_loss_idx = forward_on_tape(teacher, in_tok, tgt_tok, T);
        int t_logits_idx = nt_tape_get()->entries[t_loss_idx].parent1;
        read_tape_output(t_logits_idx, teacher_logits, T * VOCAB);
        nt_tape_clear();

        /* 2. Student forward — WITH tape + grad. */
        nt_train_mode(1);
        nt_tape_start();
        int s_loss_idx = forward_on_tape(student, in_tok, tgt_tok, T);
        int s_logits_idx = nt_tape_get()->entries[s_loss_idx].parent1;
        float hard_ce = read_tape_loss(s_loss_idx);
        read_tape_output(s_logits_idx, student_logits, T * VOCAB);

        /* 3. Inject α·KD − α·hardCE into the logits grad (CE backward adds
         *    the weight-1.0 hard-CE grad → net α·KD + (1-α)·hardCE). */
        float soft_loss = inject_kd_grad(s_logits_idx, teacher_logits, student_logits,
                                         tgt_tok, T, VOCAB, alpha, tau);

        /* 4. Backward — propagates the combined logits grad through net. */
        nt_tape_backward(s_loss_idx);

        /* soft_loss already carries the α factor (α·τ²·KL). */
        float combined = soft_loss + (1.0f - alpha) * hard_ce;

        /* 5. Chuck step. */
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, combined);
        nt_tape_clear();

        if (step == 0) first_loss = combined;
        last_loss = combined;

        if (step % LOG_EVERY == 0 || step == max_steps - 1) {
            double el = (now_ms() - t0) / 1000.0;
            printf("  step %4d | loss %.4f | soft(α·τ²·KL) %.4f | hardCE %.4f | %.1fs\n",
                   step, combined, soft_loss, hard_ce, el);
        }

        if ((step > 0 && step % CKPT_EVERY == 0)) {
            char path[512];
            snprintf(path, sizeof(path), "%s.step%d", out_path, step);
            nt_save(path, student_params, n_params);
            printf("    ckpt → %s\n", path);
        }
    }

    /* Final checkpoint. */
    nt_save(out_path, student_params, n_params);
    printf("───────────────────────────────────────────────────\n");
    printf("  saved student → %s (%d tensors)\n", out_path, n_params);
    printf("  loss %.4f → %.4f  (Δ %.4f)\n", first_loss, last_loss, last_loss - first_loss);

    /* Sanity: scalar KL helper on first vs last position of last window. */
    float kl_demo = kl_divergence_softmax(teacher_logits, student_logits, VOCAB, tau);
    printf("  KL(teacher‖student) last-window pos0: %.4f\n", kl_demo);

    free(in_tok); free(tgt_tok); free(teacher_logits); free(student_logits);
    free(student_params); free(toks);
    model_free(teacher); model_free(student);
    return 0;
}
