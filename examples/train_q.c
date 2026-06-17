/*
 * train_q.c — Train a 1.65M transformer on postgpt.txt using notorch + Chuck
 *
 * First real training on notorch. No Python. No pip. No torch.
 * Char-level (V=256), 6 layers, E=128, H=4, FFN=512, ctx=64.
 *
 * Build: make train_q
 * Run:   ./train_q [steps] [lr]
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

// ── Config ───────────────────────────────────────────────────────────────────

#define V     256
#define E     128
#define H     4
#define HD    (E / H)  // 32
#define FFN   512
#define CTX   64
#define N_LAYERS 6

// ── Model weights ────────────────────────────────────────────────────────────

typedef struct {
    nt_tensor *wte;     // [V, E]
    nt_tensor *wpe;     // [CTX, E]
    struct {
        nt_tensor *rms1;       // [E]
        nt_tensor *wq, *wk, *wv, *wo;  // [E, E]
        nt_tensor *rms2;       // [E]
        nt_tensor *w_gate;     // [FFN, E]  (SiLU gate)
        nt_tensor *w_up;       // [FFN, E]
        nt_tensor *w_down;     // [E, FFN]
    } layers[N_LAYERS];
    nt_tensor *rms_f;   // [E]
    nt_tensor *head;    // [V, E] (tied with wte for training, separate for clarity)
} Model;

static long count_params(Model* m) {
    long n = m->wte->len + m->wpe->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < N_LAYERS; l++) {
        n += m->layers[l].rms1->len + m->layers[l].rms2->len;
        n += m->layers[l].wq->len + m->layers[l].wk->len;
        n += m->layers[l].wv->len + m->layers[l].wo->len;
        n += m->layers[l].w_gate->len + m->layers[l].w_up->len + m->layers[l].w_down->len;
    }
    return n;
}

static Model* model_create(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));

    m->wte = nt_tensor_new2d(V, E);
    nt_tensor_xavier(m->wte, V, E);
    m->wpe = nt_tensor_new2d(CTX, E);
    nt_tensor_xavier(m->wpe, CTX, E);

    for (int l = 0; l < N_LAYERS; l++) {
        m->layers[l].rms1 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wq, E, E);
        m->layers[l].wk = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wk, E, E);
        m->layers[l].wv = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wv, E, E);
        m->layers[l].wo = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wo, E, E);
        // Scale wo init for stable residuals
        float scale = 0.02f / sqrtf(2.0f * N_LAYERS);
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale / 0.1f;

        m->layers[l].rms2 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_gate = nt_tensor_new2d(FFN, E);
        nt_tensor_xavier(m->layers[l].w_gate, E, FFN);
        m->layers[l].w_up = nt_tensor_new2d(FFN, E);
        nt_tensor_xavier(m->layers[l].w_up, E, FFN);
        m->layers[l].w_down = nt_tensor_new2d(E, FFN);
        nt_tensor_xavier(m->layers[l].w_down, FFN, E);
        for (int i = 0; i < m->layers[l].w_down->len; i++)
            m->layers[l].w_down->data[i] *= scale / 0.1f;
    }

    m->rms_f = nt_tensor_new(E);
    nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(V, E);
    nt_tensor_xavier(m->head, E, V);

    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    for (int l = 0; l < N_LAYERS; l++) {
        nt_tensor_free(m->layers[l].rms1); nt_tensor_free(m->layers[l].rms2);
        nt_tensor_free(m->layers[l].wq); nt_tensor_free(m->layers[l].wk);
        nt_tensor_free(m->layers[l].wv); nt_tensor_free(m->layers[l].wo);
        nt_tensor_free(m->layers[l].w_gate); nt_tensor_free(m->layers[l].w_up);
        nt_tensor_free(m->layers[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head);
    free(m);
}

// ── Forward pass on tape ─────────────────────────────────────────────────────

static int model_forward(Model* m, int* tokens, int* targets) {
    // Register params
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);

    int layer_idx[N_LAYERS][9]; // store tape indices per layer
    for (int l = 0; l < N_LAYERS; l++) {
        layer_idx[l][0] = nt_tape_param(m->layers[l].rms1);
        layer_idx[l][1] = nt_tape_param(m->layers[l].wq);
        layer_idx[l][2] = nt_tape_param(m->layers[l].wk);
        layer_idx[l][3] = nt_tape_param(m->layers[l].wv);
        layer_idx[l][4] = nt_tape_param(m->layers[l].wo);
        layer_idx[l][5] = nt_tape_param(m->layers[l].rms2);
        layer_idx[l][6] = nt_tape_param(m->layers[l].w_gate);
        layer_idx[l][7] = nt_tape_param(m->layers[l].w_up);
        layer_idx[l][8] = nt_tape_param(m->layers[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    // Input tokens and targets as tensors
    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) {
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

    // Embed: h = wte[tokens] + wpe[positions]
    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, CTX, E);

    // Transformer blocks
    for (int l = 0; l < N_LAYERS; l++) {
        // RMSNorm → Attention
        int xn = nt_seq_rmsnorm(h, layer_idx[l][0], CTX, E);
        int q = nt_seq_linear(layer_idx[l][1], xn, CTX);
        int k = nt_seq_linear(layer_idx[l][2], xn, CTX);
        int v = nt_seq_linear(layer_idx[l][3], xn, CTX);
        int attn = nt_mh_causal_attention(q, k, v, CTX, HD);
        int proj = nt_seq_linear(layer_idx[l][4], attn, CTX);
        h = nt_add(h, proj); // residual

        // RMSNorm → SiLU-gated FFN
        xn = nt_seq_rmsnorm(h, layer_idx[l][5], CTX, E);
        int gate = nt_seq_linear(layer_idx[l][6], xn, CTX);
        int up = nt_seq_linear(layer_idx[l][7], xn, CTX);
        // SiLU gate: silu(gate) * up
        gate = nt_silu(gate);
        int ffn_h = nt_mul(gate, up);
        int down = nt_seq_linear(layer_idx[l][8], ffn_h, CTX);
        h = nt_add(h, down); // residual
    }

    // Final norm + head
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, E);
    int logits = nt_seq_linear(head_i, hf, CTX);

    // Loss
    int loss = nt_seq_cross_entropy(logits, tgt_i, CTX, V);
    return loss;
}

// ── Timer ────────────────────────────────────────────────────────────────────

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 200;
    float base_lr = argc > 2 ? (float)atof(argv[2]) : 3e-4f;

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch training — PostGPT-Q transformer\n");
    printf("  V=%d E=%d H=%d FFN=%d CTX=%d L=%d\n", V, E, H, FFN, CTX, N_LAYERS);
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("════════════════════════════════════════════════════════\n");

    // Load corpus
    const char* corpus_path = "./postgpt.txt";
    FILE* f = fopen(corpus_path, "rb");
    if (!f) { printf("cannot open %s\n", corpus_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);
    printf("corpus: %ld bytes\n", fsize);

    // Create model
    nt_seed(42);
    Model* model = model_create();
    long np = count_params(model);
    printf("model: %ld params (%.2f MB)\n", np, np * 4.0f / 1048576.0f);

    // LR schedule: cosine with warmup
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 10, steps, base_lr * 0.1f);

    // NaN guard
    nt_nan_guard guard = nt_nan_guard_new();

    // Training loop
    printf("\ntraining...\n");
    printf("─────────────────────────────────────────────────\n");

    double t0 = now_ms();
    float first_loss = 0, last_loss = 0;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        // Random batch: pick random position in corpus
        int off = rand() % (int)(fsize - CTX - 1);
        int tokens[CTX], targets[CTX];
        for (int i = 0; i < CTX; i++) {
            tokens[i] = data[off + i];
            targets[i] = data[off + i + 1];
        }

        // Forward
        nt_tape_start();
        int loss_idx = model_forward(model, tokens, targets);
        float loss_val = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        // Backward
        nt_tape_backward(loss_idx);

        // NaN check
        if (!nt_nan_guard_check(&guard)) {
            if (step % 10 == 0)
                printf("  step %4d: NaN detected, skipping (scale=%.4f)\n", step + 1, guard.loss_scale);
            nt_tape_clear();
            continue;
        }

        // Gradient clip + Chuck step
        float gnorm = nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

        // Log
        if ((step + 1) % 10 == 0 || step == 0) {
            double elapsed = (now_ms() - t0) / 1000.0;
            printf("  step %4d | train %.4f | lr %.2e | gnorm %.2f | %.1fs\n",
                   step + 1, loss_val, lr, gnorm, elapsed);
        }
    }

    double total_s = (now_ms() - t0) / 1000.0;
    printf("─────────────────────────────────────────────────\n");
    printf("  loss: %.4f → %.4f (%.1f%% reduction)\n",
           first_loss, last_loss,
           first_loss > 0 ? (first_loss - last_loss) / first_loss * 100.0f : 0);
    printf("  time: %.1f seconds (%.1f steps/s)\n", total_s, steps / total_s);
    printf("  nans: %d detected, %d steps skipped\n", guard.total_nan_count, guard.skipped_steps);

    // Generate
    printf("\n── generation ──\n");
    nt_train_mode(0); // eval mode (no dropout)
    int ctx[CTX];
    // Seed with "The "
    ctx[0] = 'T'; ctx[1] = 'h'; ctx[2] = 'e'; ctx[3] = ' ';
    int gen_len = 4;

    for (int step = 0; step < 200 && gen_len < CTX; step++) {
        nt_tape_start();
        int tokens_arr[CTX], targets_arr[CTX];
        for (int i = 0; i < gen_len; i++) tokens_arr[i] = ctx[i];
        for (int i = gen_len; i < CTX; i++) tokens_arr[i] = 0;
        for (int i = 0; i < CTX; i++) targets_arr[i] = 0;

        int loss_idx = model_forward(model, tokens_arr, targets_arr);

        // Get logits for last position
        // The logits are at seq_linear output, which is CTX * V
        // We need the last position's logits
        nt_tape* tape = nt_tape_get();
        // Find the seq_cross_entropy parent (logits)
        int logits_idx = tape->entries[loss_idx].parent1;
        nt_tensor* logits = tape->entries[logits_idx].output;

        // Sample from last position
        float* last_logits = logits->data + (gen_len - 1) * V;
        float temp = 0.8f;
        for (int i = 0; i < V; i++) last_logits[i] /= temp;
        // Softmax
        float mx = last_logits[0];
        for (int i = 1; i < V; i++) if (last_logits[i] > mx) mx = last_logits[i];
        float sum = 0;
        for (int i = 0; i < V; i++) { last_logits[i] = expf(last_logits[i] - mx); sum += last_logits[i]; }
        for (int i = 0; i < V; i++) last_logits[i] /= sum;
        // Sample
        float r = (float)rand() / (float)RAND_MAX;
        float cum = 0;
        int next = 0;
        for (int i = 0; i < V; i++) { cum += last_logits[i]; if (cum >= r) { next = i; break; } }

        ctx[gen_len++] = next;
        nt_tape_clear();
    }

    // Print
    printf("  \"");
    for (int i = 0; i < gen_len; i++) {
        char c = (char)ctx[i];
        if (c >= 32 && c < 127) printf("%c", c);
        else if (c == '\n') printf("\\n");
        else printf(".");
    }
    printf("\"\n");

    // Save weights
    printf("\n── saving weights ──\n");
    nt_tensor* params[] = {
        model->wte, model->wpe,
        model->layers[0].rms1, model->layers[0].wq, model->layers[0].wk,
        model->layers[0].wv, model->layers[0].wo, model->layers[0].rms2,
        model->layers[0].w_gate, model->layers[0].w_up, model->layers[0].w_down,
        model->layers[1].rms1, model->layers[1].wq, model->layers[1].wk,
        model->layers[1].wv, model->layers[1].wo, model->layers[1].rms2,
        model->layers[1].w_gate, model->layers[1].w_up, model->layers[1].w_down,
        model->layers[2].rms1, model->layers[2].wq, model->layers[2].wk,
        model->layers[2].wv, model->layers[2].wo, model->layers[2].rms2,
        model->layers[2].w_gate, model->layers[2].w_up, model->layers[2].w_down,
        model->layers[3].rms1, model->layers[3].wq, model->layers[3].wk,
        model->layers[3].wv, model->layers[3].wo, model->layers[3].rms2,
        model->layers[3].w_gate, model->layers[3].w_up, model->layers[3].w_down,
        model->layers[4].rms1, model->layers[4].wq, model->layers[4].wk,
        model->layers[4].wv, model->layers[4].wo, model->layers[4].rms2,
        model->layers[4].w_gate, model->layers[4].w_up, model->layers[4].w_down,
        model->layers[5].rms1, model->layers[5].wq, model->layers[5].wk,
        model->layers[5].wv, model->layers[5].wo, model->layers[5].rms2,
        model->layers[5].w_gate, model->layers[5].w_up, model->layers[5].w_down,
        model->rms_f, model->head
    };
    int n_save = sizeof(params) / sizeof(params[0]);
    nt_save("q_weights.bin", params, n_save);
    printf("  saved %d tensors to q_weights.bin\n", n_save);

    model_free(model);
    free(data);

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Training complete. No Python was harmed.\n");
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
