/*
 * train_yent.c — Train a 9.8M LLaMA-like transformer on Yent dataset
 *
 * Architecture: V=256, E=224, H=8, FFN=896, CTX=128, L=12
 * Dataset: yent_v11_en_final.txt (5.6MB, cynical AI character)
 * Optimizer: Chuck (synced with PyTorch)
 *
 * Build: make train_yent
 * Run:   ./train_yent [steps] [lr]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define V     256
#define EMB   224
#define HEADS 8
#define HD    (EMB / HEADS)  // 28
#define FFN_D 896
#define CTX   128
#define NLAYERS 12

typedef struct {
    nt_tensor *wte, *wpe;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->wpe->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < NLAYERS; l++) {
        n += m->L[l].rms1->len + m->L[l].rms2->len;
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].wv->len + m->L[l].wo->len;
        n += m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    }
    return n;
}

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(V, EMB); nt_tensor_xavier(m->wte, V, EMB);
    m->wpe = nt_tensor_new2d(CTX, EMB); nt_tensor_xavier(m->wpe, CTX, EMB);
    float res_scale = 0.02f / sqrtf(2.0f * NLAYERS);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(EMB); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(EMB, EMB); nt_tensor_xavier(m->L[l].wq, EMB, EMB);
        m->L[l].wk = nt_tensor_new2d(EMB, EMB); nt_tensor_xavier(m->L[l].wk, EMB, EMB);
        m->L[l].wv = nt_tensor_new2d(EMB, EMB); nt_tensor_xavier(m->L[l].wv, EMB, EMB);
        m->L[l].wo = nt_tensor_new2d(EMB, EMB); nt_tensor_xavier(m->L[l].wo, EMB, EMB);
        for (int i = 0; i < m->L[l].wo->len; i++) m->L[l].wo->data[i] *= res_scale / 0.1f;
        m->L[l].rms2 = nt_tensor_new(EMB); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(FFN_D, EMB); nt_tensor_xavier(m->L[l].w_gate, EMB, FFN_D);
        m->L[l].w_up = nt_tensor_new2d(FFN_D, EMB); nt_tensor_xavier(m->L[l].w_up, EMB, FFN_D);
        m->L[l].w_down = nt_tensor_new2d(EMB, FFN_D); nt_tensor_xavier(m->L[l].w_down, FFN_D, EMB);
        for (int i = 0; i < m->L[l].w_down->len; i++) m->L[l].w_down->data[i] *= res_scale / 0.1f;
    }
    m->rms_f = nt_tensor_new(EMB); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(V, EMB); nt_tensor_xavier(m->head, EMB, V);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

static int forward(Model* m, int* tokens, int* targets) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);
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

    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, CTX, EMB);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, EMB);
        int q = nt_seq_linear(li[l][1], xn, CTX);
        int k = nt_seq_linear(li[l][2], xn, CTX);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HD);
        int proj = nt_seq_linear(li[l][4], attn, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][5], CTX, EMB);
        int gate = nt_silu(nt_seq_linear(li[l][6], xn, CTX));
        int up = nt_seq_linear(li[l][7], xn, CTX);
        int down = nt_seq_linear(li[l][8], nt_mul(gate, up), CTX);
        h = nt_add(h, down);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, EMB);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, V);
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; }

int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 5000;
    float base_lr = argc > 2 ? (float)atof(argv[2]) : 3e-4f;

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch — Yent 9.8M LLaMA training\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", V, EMB, HEADS, FFN_D, CTX, NLAYERS);
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("════════════════════════════════════════════════════════\n");

    const char* path = "/Users/ataeff/Downloads/yent-datasets/yent_v11_en_final.txt";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(fsize);
    fread(data, 1, fsize, f); fclose(f);
    printf("corpus: %.1f MB (%ld bytes)\n", fsize/1048576.0, fsize);

    nt_seed(42);
    Model* model = model_new();
    printf("model: %ld params (%.1f MB)\n", count_params(model), count_params(model)*4.0f/1048576.0f);

    nt_schedule sched = nt_schedule_cosine(base_lr, steps/10, steps, base_lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\ntraining...\n");
    printf("─────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float first_loss = 0, best_loss = 99;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        int off = rand() % (int)(fsize - CTX - 1);
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < CTX; i++) { tokens[i] = data[off+i]; targets[i] = data[off+i+1]; }

        nt_tape_start();
        int loss_idx = forward(model, tokens, targets);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];
        if (step == 0) first_loss = lv;
        if (lv < best_loss) best_loss = lv;

        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();

        if ((step+1) % 50 == 0 || step == 0)
            printf("  step %5d | train %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step+1, lv, best_loss, lr, (now_ms()-t0)/1000.0);
    }

    double total_s = (now_ms()-t0)/1000.0;
    float last_loss = 0;
    { // eval last
        int off = 0;
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < CTX; i++) { tokens[i]=data[off+i]; targets[i]=data[off+i+1]; }
        nt_tape_start();
        int loss_idx = forward(model, tokens, targets);
        last_loss = nt_tape_get()->entries[loss_idx].output->data[0];
        nt_tape_clear();
    }

    printf("─────────────────────────────────────────────────\n");
    printf("  loss: %.4f → %.4f (best: %.4f)\n", first_loss, last_loss, best_loss);
    printf("  time: %.0fs (%.1f steps/s)\n", total_s, steps/total_s);
    printf("  nans: %d\n", guard.total_nan_count);

    // Generate
    printf("\n── generation (temp=0.8) ──\n");
    nt_train_mode(0);
    int ctx[CTX]; int gen_len = 0;
    const char* seed = "Q: Who are you?\nA: ";
    for (int i = 0; seed[i] && gen_len < CTX/2; i++) ctx[gen_len++] = (unsigned char)seed[i];
    printf("%s", seed);

    for (int step = 0; step < CTX - gen_len - 1; step++) {
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < gen_len; i++) tokens[i] = ctx[i];
        for (int i = gen_len; i < CTX; i++) tokens[i] = 0;
        memset(targets, 0, sizeof(targets));
        nt_tape_start();
        int loss_idx = forward(model, tokens, targets);
        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        float* last = tape->entries[logits_idx].output->data + (gen_len-1)*V;
        for (int i = 0; i < V; i++) last[i] /= 0.8f;
        float mx = last[0]; for (int i=1;i<V;i++) if(last[i]>mx) mx=last[i];
        float sm = 0; for (int i=0;i<V;i++) { last[i]=expf(last[i]-mx); sm+=last[i]; }
        for (int i=0;i<V;i++) last[i]/=sm;
        float r=(float)rand()/(float)RAND_MAX, cum=0; int next=0;
        for (int i=0;i<V;i++) { cum+=last[i]; if(cum>=r){next=i;break;} }
        char c = (char)next;
        if (c >= 32 && c < 127) printf("%c", c); else if (c=='\n') printf("\n"); else printf(".");
        fflush(stdout);
        ctx[gen_len++] = next;
        nt_tape_clear();
    }
    printf("\n");

    // Save
    printf("\n── saving ──\n");
    int n_tensors = 2 + NLAYERS*9 + 2;
    nt_tensor** params = (nt_tensor**)malloc(n_tensors * sizeof(nt_tensor*));
    int pi = 0;
    params[pi++] = model->wte; params[pi++] = model->wpe;
    for (int l = 0; l < NLAYERS; l++) {
        params[pi++]=model->L[l].rms1; params[pi++]=model->L[l].wq; params[pi++]=model->L[l].wk;
        params[pi++]=model->L[l].wv; params[pi++]=model->L[l].wo; params[pi++]=model->L[l].rms2;
        params[pi++]=model->L[l].w_gate; params[pi++]=model->L[l].w_up; params[pi++]=model->L[l].w_down;
    }
    params[pi++] = model->rms_f; params[pi++] = model->head;
    nt_save("yent_10m.bin", params, pi);
    printf("  saved %d tensors to yent_10m.bin (%.1f MB)\n", pi, count_params(model)*4.0f/1048576.0f);
    free(params);

    model_free(model); free(data);
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Yent 9.8M trained. No Python harmed. Chuck approves.\n");
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
