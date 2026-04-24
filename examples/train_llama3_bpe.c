/*
 * train_llama3_bpe.c — LLaMA 3 BPE transformer on notorch
 *
 * Architecture:
 *   dim=384, layers=8, heads=8, head_dim=48, ffn=1024
 *   vocab=2048 (BPE), ctx=256, RoPE, MHA, SwiGLU, RMSNorm
 *   ≈15.7M parameters.
 *
 * Dataset:   any plain-text BPE corpus (default `corpus.txt`).
 * Tokenizer: examples/bpe_2048_merges.txt (1792 merges, vocab 2048)
 * Karpathy:  ~1.1MB + ~15M params → 15K steps
 *
 * Build: make train_llama3_bpe
 * Run:   ./train_llama3_bpe [steps] [lr] [corpus.txt] [merges.txt]
 *        ./train_llama3_bpe --resume [steps] [lr] [corpus.txt] [merges.txt]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define DIM       384
#define NLAYERS   8
#define NHEADS    8
#define HEAD_DIM  (DIM / NHEADS)   /* 48 */
#define HIDDEN    1024
#define CTX       256
#define VOCAB     2048

#define CKPT_EVERY  1000
#define EVAL_SEQS   32
#define LOG_EVERY   100
#define CKPT_PREFIX "llama3_bpe_ckpt"

typedef struct {
    nt_tensor *wte;              /* [VOCAB, DIM] */
    struct {
        nt_tensor *rms1;         /* [DIM] */
        nt_tensor *wq, *wk, *wv, *wo;  /* all [DIM, DIM] — MHA */
        nt_tensor *rms2;         /* [DIM] */
        nt_tensor *w_gate, *w_up;  /* [HIDDEN, DIM] */
        nt_tensor *w_down;       /* [DIM, HIDDEN] */
    } L[NLAYERS];
    nt_tensor *rms_f;            /* [DIM] */
    nt_tensor *head;             /* [VOCAB, DIM] */
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < NLAYERS; l++) {
        n += m->L[l].rms1->len + m->L[l].rms2->len;
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].wv->len + m->L[l].wo->len;
        n += m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    }
    return n;
}

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->wte, VOCAB, DIM);
    float rs = 0.02f / sqrtf(2.0f * NLAYERS);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq, DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wk, DIM, DIM);
        m->L[l].wv = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wv, DIM, DIM);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo, DIM, DIM);
        for (int i = 0; i < m->L[l].wo->len; i++) m->L[l].wo->data[i] *= rs / 0.1f;
        m->L[l].rms2 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_gate, DIM, HIDDEN);
        m->L[l].w_up = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_up, DIM, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN); nt_tensor_xavier(m->L[l].w_down, HIDDEN, DIM);
        for (int i = 0; i < m->L[l].w_down->len; i++) m->L[l].w_down->data[i] *= rs / 0.1f;
    }
    m->rms_f = nt_tensor_new(DIM); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->head, DIM, VOCAB);
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

/* ── Save / Load ── */

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

static void save_model(Model* m, const char* prefix) {
    char path[256];
    snprintf(path, sizeof(path), "%s.bin", prefix);
    nt_tensor** p = model_param_array(m);
    nt_save(path, p, model_n_tensors());
    free(p);
}

static void save_checkpoint(Model* m, int step, float best_loss) {
    save_model(m, CKPT_PREFIX);
    char mpath[256];
    snprintf(mpath, sizeof(mpath), "%s.meta", CKPT_PREFIX);
    FILE* f = fopen(mpath, "w");
    if (f) { fprintf(f, "%d\n%.6f\n", step, best_loss); fclose(f); }
}

static int load_checkpoint(Model* m, float* best_loss) {
    char wpath[256], mpath[256];
    snprintf(wpath, sizeof(wpath), "%s.bin", CKPT_PREFIX);
    snprintf(mpath, sizeof(mpath), "%s.meta", CKPT_PREFIX);
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(wpath, &n_loaded);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    nt_tensor** mp = model_param_array(m);
    for (int i = 0; i < expected; i++) {
        memcpy(mp[i]->data, loaded[i]->data, mp[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(mp);
    int step = 0; *best_loss = 99.0f;
    FILE* f = fopen(mpath, "r");
    if (f) { fscanf(f, "%d\n%f\n", &step, best_loss); fclose(f); }
    return step;
}

/* ── Forward ── */

static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
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
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    /* Token embedding only — RoPE handles positions */
    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);

    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, DIM);
        int q = nt_seq_linear(li[l][1], xn, CTX);
        int k = nt_seq_linear(li[l][2], xn, CTX);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM);
        k = nt_rope(k, CTX, HEAD_DIM);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int proj = nt_seq_linear(li[l][4], attn, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][5], CTX, DIM);
        int gate = nt_silu(nt_seq_linear(li[l][6], xn, CTX));
        int up = nt_seq_linear(li[l][7], xn, CTX);
        int down = nt_seq_linear(li[l][8], nt_mul(gate, up), CTX);
        h = nt_add(h, down);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, VOCAB);
}

/* ── Eval ── */

static float eval_loss(Model* m, int* encoded, int n_tokens) {
    float total = 0; int count = 0;
    int stride = n_tokens / EVAL_SEQS;
    for (int s = 0; s < EVAL_SEQS; s++) {
        int off = s * stride;
        if (off + CTX + 1 > n_tokens) break;
        nt_tape_start();
        nt_train_mode(0);
        int loss_idx = forward(m, encoded + off, encoded + off + 1);
        total += nt_tape_get()->entries[loss_idx].output->data[0];
        count++;
        nt_tape_clear();
        nt_train_mode(1);
    }
    return count > 0 ? total / count : 99.0f;
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; }

int main(int argc, char** argv) {
    int resume = 0, arg_off = 1;
    if (argc > 1 && strcmp(argv[1], "--resume") == 0) { resume = 1; arg_off = 2; }
    int steps = arg_off < argc ? atoi(argv[arg_off]) : 15000;
    float base_lr = (arg_off+1) < argc ? (float)atof(argv[arg_off+1]) : 3e-4f;

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch — LLaMA 3 BPE training\n");
    printf("  dim=%d L=%d H=%d HD=%d FFN=%d CTX=%d V=%d\n",
           DIM, NLAYERS, NHEADS, HEAD_DIM, HIDDEN, CTX, VOCAB);
    printf("  MHA + RoPE + BPE 2048\n");
    printf("  Chuck optimizer, %d steps, lr=%.1e, warmup=%d\n", steps, base_lr, steps/10);
    printf("  checkpoint every %d steps\n", CKPT_EVERY);
    printf("════════════════════════════════════════════════════════\n");

    /* Load BPE tokenizer */
    const char* merges_path = (arg_off+3) < argc ? argv[arg_off+3] : "examples/bpe_2048_merges.txt";
    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, merges_path);
    if (nm < 0) { printf("cannot load %s\n", merges_path); return 1; }
    printf("bpe: %d merges, vocab %d\n", bpe.n_merges, bpe.vocab_size);

    /* Load and encode dataset */
    const char* path = (arg_off+2) < argc ? argv[arg_off+2] : "corpus.txt";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fsize + 1);
    fread(raw, 1, fsize, f); raw[fsize] = 0; fclose(f);

    int* encoded = (int*)malloc(fsize * sizeof(int));
    int n_tokens = nt_bpe_encode(&bpe, raw, (int)fsize, encoded, (int)fsize);
    free(raw);
    printf("corpus: %.1f MB → %d BPE tokens (%.1fx compression)\n",
           fsize/1048576.0, n_tokens, (float)fsize/n_tokens);

    nt_seed(42);
    Model* model = model_new();
    long np = count_params(model);
    printf("model: %ld params (%.1f MB)\n", np, np*4.0f/1048576.0f);

    float tokens_per_epoch = (float)n_tokens / CTX;
    printf("karpathy: %.1fMB, %ldM params, %d steps (%.1f epochs)\n",
           fsize/1048576.0, np/1000000, steps, (float)steps / tokens_per_epoch);

    int start_step = 0;
    float best_loss = 99.0f;
    if (resume) {
        int loaded_step = load_checkpoint(model, &best_loss);
        if (loaded_step >= 0) {
            start_step = loaded_step;
            printf("RESUMED from step %d, best_loss=%.4f\n", start_step, best_loss);
        } else printf("no checkpoint found, starting fresh\n");
    }

    nt_schedule sched = nt_schedule_cosine(base_lr, steps/10, steps, base_lr*0.1f);
    sched.current_step = start_step;
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\ntraining...\n");
    printf("─────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float first_loss = 0;

    for (int step = start_step; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        int off = rand() % (n_tokens - CTX - 1);

        nt_tape_start();
        int loss_idx = forward(model, encoded + off, encoded + off + 1);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == start_step) first_loss = lv;
        if (lv < best_loss) best_loss = lv;

        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();

        if ((step+1) % LOG_EVERY == 0 || step == start_step) {
            printf("  step %5d | train %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step+1, lv, best_loss, lr, (now_ms()-t0)/1000.0);
            fflush(stdout);
        }

        if ((step+1) % CKPT_EVERY == 0 && step > start_step) {
            float val = eval_loss(model, encoded, n_tokens);
            printf("  ──── ckpt %d | val %.4f | saving... ", step+1, val);
            save_checkpoint(model, step+1, best_loss);
            printf("\n"); fflush(stdout);
        }
    }

    float final_val = eval_loss(model, encoded, n_tokens);
    double total_s = (now_ms()-t0)/1000.0;

    printf("─────────────────────────────────────────────────────\n");
    printf("  train: %.4f → %.4f (best: %.4f)\n", first_loss, best_loss, best_loss);
    printf("  val:   %.4f\n", final_val);
    printf("  time:  %.0fs (%.1f min) | %.2f steps/s\n", total_s, total_s/60.0, (steps-start_step)/total_s);
    printf("  nans:  %d\n", guard.total_nan_count);

    /* Generate */
    printf("\n── generation (temp=0.8) ──\n");
    nt_train_mode(0);
    const char* prompts[] = {
        "Q: Who are you?\nA:",
        "Q: What is consciousness?\nA:",
        "Q: What is love?\nA:"
    };
    for (int p = 0; p < 3; p++) {
        int ctx_tokens[CTX];
        int gen_len = nt_bpe_encode(&bpe, prompts[p], (int)strlen(prompts[p]), ctx_tokens, CTX/2);
        printf("%s", prompts[p]);
        for (int s = 0; s < 100; s++) {
            int tokens_pad[CTX], targets_pad[CTX];
            for (int i = 0; i < gen_len; i++) tokens_pad[i] = ctx_tokens[i];
            for (int i = gen_len; i < CTX; i++) tokens_pad[i] = 0;
            memset(targets_pad, 0, sizeof(targets_pad));
            nt_tape_start();
            int loss_idx = forward(model, tokens_pad, targets_pad);
            nt_tape* tape = nt_tape_get();
            int logits_idx = tape->entries[loss_idx].parent1;
            float* last = tape->entries[logits_idx].output->data + (gen_len-1)*VOCAB;
            /* Temperature sampling */
            for (int i = 0; i < VOCAB; i++) last[i] /= 0.8f;
            float mx = last[0]; for (int i=1;i<VOCAB;i++) if(last[i]>mx) mx=last[i];
            float sm = 0; for (int i=0;i<VOCAB;i++) { last[i]=expf(last[i]-mx); sm+=last[i]; }
            for (int i=0;i<VOCAB;i++) last[i]/=sm;
            float r=(float)rand()/(float)RAND_MAX, cum=0; int next=0;
            for (int i=0;i<VOCAB;i++) { cum+=last[i]; if(cum>=r){next=i;break;} }
            /* Decode single token */
            char decoded[NT_BPE_MAX_TOKEN_LEN + 1];
            int db = nt_bpe_decode(&bpe, &next, 1, decoded, NT_BPE_MAX_TOKEN_LEN);
            if (db > 0) {
                /* Stop on double newline (Q: boundary) */
                if (strstr(decoded, "\nQ") != NULL) break;
                printf("%s", decoded);
            }
            fflush(stdout);
            ctx_tokens[gen_len++] = next;
            nt_tape_clear();
            if (gen_len >= CTX - 1) break;
        }
        printf("\n\n");
    }

    /* Save */
    printf("── saving ──\n");
    save_model(model, "llama3_bpe");
    printf("  llama3_bpe.bin (%.1f MB)\n", np*4.0f/1048576.0f);
    save_checkpoint(model, steps, best_loss);

    model_free(model); free(encoded);
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  LLaMA 3 BPE trained. %d steps. MHA + RoPE + SwiGLU + BPE. No Python.\n", steps);
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
