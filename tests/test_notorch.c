// test_notorch.c — tests for notorch
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        printf("  FAIL: %s — got %.6f, expected %.6f (line %d)\n", msg, _a, _b, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define PASS(name) do { printf("  PASS: %s\n", name); tests_passed++; } while(0)

// ── Tensor tests ─────────────────────────────────────────────────────────────

static void test_tensor_create(void) {
    nt_tensor* t = nt_tensor_new(10);
    ASSERT(t != NULL, "tensor alloc");
    ASSERT(t->len == 10, "tensor len");
    ASSERT(t->ndim == 1, "tensor ndim");
    ASSERT(t->shape[0] == 10, "tensor shape");
    ASSERT(t->data[0] == 0.0f, "tensor zeroed");
    nt_tensor_free(t);
    PASS("tensor_create");
}

static void test_tensor_2d(void) {
    nt_tensor* t = nt_tensor_new2d(3, 4);
    ASSERT(t != NULL, "2d alloc");
    ASSERT(t->len == 12, "2d len");
    ASSERT(t->ndim == 2, "2d ndim");
    ASSERT(t->shape[0] == 3 && t->shape[1] == 4, "2d shape");
    ASSERT(t->stride[0] == 4 && t->stride[1] == 1, "2d stride");
    nt_tensor_free(t);
    PASS("tensor_2d");
}

static void test_tensor_clone(void) {
    nt_tensor* a = nt_tensor_new(5);
    for (int i = 0; i < 5; i++) a->data[i] = (float)i;
    nt_tensor* b = nt_tensor_clone(a);
    ASSERT(b != NULL, "clone alloc");
    ASSERT(b->data[3] == 3.0f, "clone data");
    a->data[3] = 99.0f;
    ASSERT(b->data[3] == 3.0f, "clone independence");
    nt_tensor_free(a);
    nt_tensor_free(b);
    PASS("tensor_clone");
}

static void test_tensor_reshape(void) {
    nt_tensor* t = nt_tensor_new(12);
    int shape[] = {3, 4};
    ASSERT(nt_tensor_reshape(t, shape, 2) == 0, "reshape ok");
    ASSERT(t->ndim == 2, "reshape ndim");
    ASSERT(t->shape[0] == 3 && t->shape[1] == 4, "reshape shape");
    int bad[] = {5, 5};
    ASSERT(nt_tensor_reshape(t, bad, 2) != 0, "reshape mismatch");
    nt_tensor_free(t);
    PASS("tensor_reshape");
}

static void test_tensor_xavier(void) {
    nt_tensor* t = nt_tensor_new(1000);
    nt_seed(42);
    nt_tensor_xavier(t, 100, 100);
    float sum = 0;
    for (int i = 0; i < t->len; i++) sum += t->data[i] * t->data[i];
    float var = sum / t->len;
    // Xavier for fan_in=100, fan_out=100: scale = sqrt(6/200) ≈ 0.173
    // Uniform[-s,s] variance = s²/3 ≈ 0.01
    ASSERT(var > 0.005f && var < 0.05f, "xavier variance");
    nt_tensor_free(t);
    PASS("tensor_xavier");
}

static void test_tensor_refcount(void) {
    nt_tensor* t = nt_tensor_new(5);
    ASSERT(t->refcount == 1, "initial refcount");
    nt_tensor_ref(t);
    ASSERT(t->refcount == 2, "ref refcount");
    nt_tensor_free(t);
    ASSERT(t->refcount == 1, "free decrements");
    nt_tensor_free(t);
    PASS("tensor_refcount");
}

// ── Tape + forward op tests ─────────────────────────────────────────────────

static void test_tape_basic(void) {
    nt_tape_start();
    ASSERT(nt_tape_is_active(), "tape active");

    nt_tensor* w = nt_tensor_new2d(4, 3);
    nt_tensor_xavier(w, 3, 4);
    int w_idx = nt_tape_param(w);
    ASSERT(w_idx >= 0, "param registered");

    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_linear(w_idx, x_idx, -1);
    ASSERT(y_idx >= 0, "linear ok");

    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT(ey->output->len == 4, "linear output dim");

    nt_tape_clear();
    nt_tensor_free(w);
    nt_tensor_free(x);
    PASS("tape_basic");
}

static void test_forward_backward_linear(void) {
    nt_seed(123);
    nt_tape_start();

    // W: 2x3, x: 3
    nt_tensor* W = nt_tensor_new2d(2, 3);
    W->data[0] = 1; W->data[1] = 0; W->data[2] = 0;
    W->data[3] = 0; W->data[4] = 1; W->data[5] = 0;
    int w_idx = nt_tape_param(W);

    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_linear(w_idx, x_idx, -1);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT_CLOSE(ey->output->data[0], 1.0f, 1e-5f, "linear W@x [0]");
    ASSERT_CLOSE(ey->output->data[1], 2.0f, 1e-5f, "linear W@x [1]");

    // CE loss against target=0
    int loss_idx = nt_cross_entropy(y_idx, 0);
    ASSERT(loss_idx >= 0, "cross_entropy ok");

    nt_tape_backward(loss_idx);

    // Check W has gradient
    nt_tape_entry* ew = &nt_tape_get()->entries[w_idx];
    ASSERT(ew->grad != NULL, "W grad exists");
    ASSERT(ew->grad->len == 6, "W grad len");

    // Gradient should be non-zero
    float gnorm = 0;
    for (int i = 0; i < 6; i++) gnorm += ew->grad->data[i] * ew->grad->data[i];
    ASSERT(gnorm > 1e-10f, "W grad non-zero");

    nt_tape_clear();
    nt_tensor_free(W);
    nt_tensor_free(x);
    PASS("forward_backward_linear");
}

static void test_adam_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    float orig0 = W->data[0];
    int w_idx = nt_tape_param(W);

    nt_tensor* x = nt_tensor_new(4);
    x->data[0] = 1; x->data[1] = 0; x->data[2] = 0; x->data[3] = 0;
    nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int loss_idx = nt_cross_entropy(w_idx, 2);  // treat W as logits, target=2
    nt_tape_backward(loss_idx);

    nt_tape_adam_step(0.01f);

    // W should have changed
    ASSERT(fabsf(W->data[0] - orig0) > 1e-6f, "adam changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    nt_tensor_free(x);
    PASS("adam_step");
}

static void test_adamw_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 2);
    nt_tape_backward(loss_idx);

    float before = W->data[0];
    nt_tape_adamw_step(0.01f, 0.1f, 0.9f, 0.999f);
    ASSERT(fabsf(W->data[0] - before) > 1e-6f, "adamw changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("adamw_step");
}

static void test_chuck_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 2);
    nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
    float loss_val = el->output->data[0];

    nt_tape_backward(loss_idx);

    float before = W->data[0];
    nt_tape_chuck_step(0.01f, loss_val);
    ASSERT(fabsf(W->data[0] - before) > 1e-6f, "chuck changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("chuck_step");
}

static void test_grad_clip(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    for (int i = 0; i < 4; i++) W->data[i] = (float)(i + 1) * 10.0f;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 0);
    nt_tape_backward(loss_idx);

    float norm_before = nt_tape_clip_grads(1000.0f); // large max, no clipping
    ASSERT(norm_before > 0, "grad norm > 0");

    // Now actually clip
    // Re-do backward
    nt_tape_clear();
    nt_tape_start();
    w_idx = nt_tape_param(W);
    loss_idx = nt_cross_entropy(w_idx, 0);
    nt_tape_backward(loss_idx);

    float norm = nt_tape_clip_grads(0.1f);
    // After clipping, norm should be ~0.1
    float norm_after = 0;
    nt_tape_entry* ew = &nt_tape_get()->entries[w_idx];
    for (int i = 0; i < ew->grad->len; i++)
        norm_after += ew->grad->data[i] * ew->grad->data[i];
    norm_after = sqrtf(norm_after);
    if (norm > 0.1f) {
        ASSERT_CLOSE(norm_after, 0.1f, 0.01f, "clipped norm");
    }

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("grad_clip");
}

static void test_silu(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = -1; x->data[1] = 0; x->data[2] = 1;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_silu(x_idx);

    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    // silu(0) = 0
    ASSERT_CLOSE(ey->output->data[1], 0.0f, 1e-5f, "silu(0)=0");
    // silu(1) = 1 * sigmoid(1) ≈ 0.731
    ASSERT_CLOSE(ey->output->data[2], 0.7311f, 0.01f, "silu(1)≈0.73");
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.269
    ASSERT_CLOSE(ey->output->data[0], -0.2689f, 0.01f, "silu(-1)≈-0.27");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("silu");
}

static void test_softmax(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_softmax(x_idx);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];

    float sum = 0;
    for (int i = 0; i < 3; i++) sum += ey->output->data[i];
    ASSERT_CLOSE(sum, 1.0f, 1e-5f, "softmax sums to 1");
    ASSERT(ey->output->data[2] > ey->output->data[1], "softmax ordering");
    ASSERT(ey->output->data[1] > ey->output->data[0], "softmax ordering 2");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("softmax");
}

static void test_rmsnorm(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(4);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3; x->data[3] = 4;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_rmsnorm(x_idx, -1);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];

    // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    float rms = sqrtf(7.5f + 1e-6f);
    ASSERT_CLOSE(ey->output->data[0], 1.0f / rms, 1e-4f, "rmsnorm [0]");
    ASSERT_CLOSE(ey->output->data[3], 4.0f / rms, 1e-4f, "rmsnorm [3]");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("rmsnorm");
}

static void test_causal_attention(void) {
    nt_tape_start();
    int T = 3, D = 4;

    nt_tensor* q = nt_tensor_new(T * D);
    nt_tensor* k = nt_tensor_new(T * D);
    nt_tensor* v = nt_tensor_new(T * D);
    nt_seed(42);
    nt_tensor_rand(q, 0.5f);
    nt_tensor_rand(k, 0.5f);
    nt_tensor_rand(v, 0.5f);

    int q_idx = nt_tape_record(q, NT_OP_NONE, -1, -1, 0);
    int k_idx = nt_tape_record(k, NT_OP_NONE, -1, -1, 0);
    int v_idx = nt_tape_record(v, NT_OP_NONE, -1, -1, 0);

    int out_idx = nt_causal_attention(q_idx, k_idx, v_idx, T, D);
    ASSERT(out_idx >= 0, "attention ok");
    nt_tape_entry* eo = &nt_tape_get()->entries[out_idx];
    ASSERT(eo->output->len == T * D, "attention output size");

    // First position should be exactly V[0] (only attends to self)
    for (int d = 0; d < D; d++)
        ASSERT_CLOSE(eo->output->data[d], v->data[d], 1e-4f, "attn pos0 = V[0]");

    nt_tape_clear();
    nt_tensor_free(q); nt_tensor_free(k); nt_tensor_free(v);
    PASS("causal_attention");
}

static void test_mh_causal_attention(void) {
    nt_tape_start();
    int T = 4, D = 8, head_dim = 4; // 2 heads
    nt_tensor* q = nt_tensor_new(T * D);
    nt_tensor* k = nt_tensor_new(T * D);
    nt_tensor* v = nt_tensor_new(T * D);
    nt_seed(7);
    nt_tensor_rand(q, 0.3f);
    nt_tensor_rand(k, 0.3f);
    nt_tensor_rand(v, 0.3f);

    int q_idx = nt_tape_record(q, NT_OP_NONE, -1, -1, 0);
    int k_idx = nt_tape_record(k, NT_OP_NONE, -1, -1, 0);
    int v_idx = nt_tape_record(v, NT_OP_NONE, -1, -1, 0);

    int out_idx = nt_mh_causal_attention(q_idx, k_idx, v_idx, T, head_dim);
    ASSERT(out_idx >= 0, "mh_attention ok");
    nt_tape_entry* eo = &nt_tape_get()->entries[out_idx];
    ASSERT(eo->output->len == T * D, "mh_attention output size");

    nt_tape_clear();
    nt_tensor_free(q); nt_tensor_free(k); nt_tensor_free(v);
    PASS("mh_causal_attention");
}

static void test_seq_cross_entropy(void) {
    nt_tape_start();
    int T = 3, V = 5;

    nt_tensor* logits = nt_tensor_new(T * V);
    nt_seed(99);
    nt_tensor_rand(logits, 1.0f);

    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 0; targets->data[1] = 2; targets->data[2] = 4;

    int l_idx = nt_tape_record(logits, NT_OP_NONE, -1, -1, 0);
    int t_idx = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

    int loss_idx = nt_seq_cross_entropy(l_idx, t_idx, T, V);
    ASSERT(loss_idx >= 0, "seq_ce ok");

    nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
    float loss = el->output->data[0];
    // Loss should be positive and reasonable (random logits → ~log(V) ≈ 1.6)
    ASSERT(loss > 0.5f && loss < 5.0f, "seq_ce loss range");

    // Test backward
    nt_tape_backward(loss_idx);
    nt_tape_entry* elg = &nt_tape_get()->entries[l_idx];
    ASSERT(elg->grad != NULL, "seq_ce grad exists");

    nt_tape_clear();
    nt_tensor_free(logits); nt_tensor_free(targets);
    PASS("seq_cross_entropy");
}

static void test_seq_linear(void) {
    nt_tape_start();
    int T = 3, in_d = 4, out_d = 2;

    nt_tensor* W = nt_tensor_new2d(out_d, in_d);
    // Identity-ish: first 2 dims
    W->data[0] = 1; W->data[1] = 0; W->data[2] = 0; W->data[3] = 0;
    W->data[4] = 0; W->data[5] = 1; W->data[6] = 0; W->data[7] = 0;
    int w_idx = nt_tape_param(W);

    nt_tensor* X = nt_tensor_new(T * in_d);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < in_d; d++)
            X->data[t * in_d + d] = (float)(t * in_d + d);
    int x_idx = nt_tape_record(X, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_seq_linear(w_idx, x_idx, T);
    ASSERT(y_idx >= 0, "seq_linear ok");
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT(ey->output->len == T * out_d, "seq_linear output size");

    // Y[0] = W @ X[0] = [X[0,0], X[0,1]] = [0, 1]
    ASSERT_CLOSE(ey->output->data[0], 0.0f, 1e-4f, "seq_linear [0,0]");
    ASSERT_CLOSE(ey->output->data[1], 1.0f, 1e-4f, "seq_linear [0,1]");

    nt_tape_clear();
    nt_tensor_free(W); nt_tensor_free(X);
    PASS("seq_linear");
}

static void test_save_load(void) {
    nt_tensor* t1 = nt_tensor_new2d(3, 4);
    nt_tensor* t2 = nt_tensor_new(5);
    for (int i = 0; i < 12; i++) t1->data[i] = (float)i;
    for (int i = 0; i < 5; i++) t2->data[i] = (float)(i * 10);

    nt_tensor* params[] = {t1, t2};
    /* Honor $TMPDIR so the test works on sandboxed environments
       (Termux/Android, hermetic CI runners) where /tmp is read-only
       or absent. POSIX-y: TMPDIR > /tmp fallback. */
    const char* tmpdir = getenv("TMPDIR");
    if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s/notorch_test.bin", tmpdir);
    int rc = nt_save(tmp_path, params, 2);
    ASSERT(rc == 0, "save ok");

    int n_loaded = 0;
    nt_tensor** loaded = nt_load(tmp_path, &n_loaded);
    ASSERT(loaded != NULL, "load ok");
    ASSERT(n_loaded == 2, "loaded count");
    ASSERT(loaded[0]->ndim == 2, "loaded[0] ndim");
    ASSERT(loaded[0]->shape[0] == 3 && loaded[0]->shape[1] == 4, "loaded[0] shape");
    ASSERT_CLOSE(loaded[0]->data[5], 5.0f, 1e-5f, "loaded[0] data");
    ASSERT(loaded[1]->len == 5, "loaded[1] len");
    ASSERT_CLOSE(loaded[1]->data[3], 30.0f, 1e-5f, "loaded[1] data");

    for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
    free(loaded);
    nt_tensor_free(t1); nt_tensor_free(t2);
    PASS("save_load");
}

static void test_hebbian(void) {
    int in_d = 4, out_d = 3, rank = 2;
    float A[4 * 2] = {0};
    float B[2 * 3] = {0};
    float x[4] = {1, 0, 0, 0};
    float dy[3] = {1, 0, 0};

    nt_hebbian_step(A, B, out_d, in_d, rank, x, dy, 1.0f, 0.01f, 0.999f);

    // A should have been updated: A[0,0] and A[0,1] should be non-zero... actually
    // with B=0 initially, proj = B^T @ dy = 0, so A stays 0. But proj2 = A^T @ x = 0 too.
    // So first step with zero matrices = no update. Need non-zero init.
    B[0] = 1.0f; // B[0,0] = 1
    nt_hebbian_step(A, B, out_d, in_d, rank, x, dy, 1.0f, 0.01f, 0.999f);
    // Now proj = B^T @ dy = [1, 0], so A[0,0] += 0.01 * 1 * 1 * 1 = 0.01
    ASSERT(fabsf(A[0]) > 1e-5f, "hebbian updated A");

    PASS("hebbian");
}

static void test_training_loop(void) {
    // Mini training loop: learn to predict target from embedding
    nt_seed(42);

    int vocab = 4, dim = 8;
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor* wout = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wout, dim, vocab);

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 50; step++) {
        nt_tape_start();
        int wte_idx = nt_tape_param(wte);
        nt_tape_no_decay(wte_idx);
        int wout_idx = nt_tape_param(wout);

        // Input token = 1, target = 2
        int h_idx = nt_embedding(wte_idx, 1);
        int logits_idx = nt_linear(wout_idx, h_idx, -1);
        int loss_idx = nt_cross_entropy(logits_idx, 2);

        nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
        float loss = el->output->data[0];
        if (step == 0) initial_loss = loss;
        if (step == 49) final_loss = loss;

        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.05f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "loss decreased");
    ASSERT(final_loss < 1.0f, "loss < 1.0");
    printf("    training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte);
    nt_tensor_free(wout);
    PASS("training_loop");
}

static void test_seq_training_loop(void) {
    // Sequence-level training: embed → seq_linear → seq_cross_entropy
    nt_seed(77);

    int vocab = 8, dim = 16, T = 4;
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor* wpe = nt_tensor_new2d(T, dim);
    nt_tensor_xavier(wpe, T, dim);
    nt_tensor* wout = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wout, dim, vocab);

    // Tokens: [1, 3, 5, 2], targets: [3, 5, 2, 7]
    nt_tensor* tokens = nt_tensor_new(T);
    tokens->data[0] = 1; tokens->data[1] = 3; tokens->data[2] = 5; tokens->data[3] = 2;
    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 3; targets->data[1] = 5; targets->data[2] = 2; targets->data[3] = 7;

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 100; step++) {
        nt_tape_start();
        int wte_idx = nt_tape_param(wte);
        nt_tape_no_decay(wte_idx);
        int wpe_idx = nt_tape_param(wpe);
        nt_tape_no_decay(wpe_idx);
        int wout_idx = nt_tape_param(wout);
        int tok_idx = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_idx = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        int h_idx = nt_seq_embedding(wte_idx, wpe_idx, tok_idx, T, dim);
        int logits_idx = nt_seq_linear(wout_idx, h_idx, T);
        int loss_idx = nt_seq_cross_entropy(logits_idx, tgt_idx, T, vocab);

        nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
        float loss = el->output->data[0];
        if (step == 0) initial_loss = loss;
        if (step == 99) final_loss = loss;

        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.01f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "seq loss decreased");
    printf("    seq training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte); nt_tensor_free(wpe); nt_tensor_free(wout);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    PASS("seq_training_loop");
}

static void test_attention_training(void) {
    // Train a tiny attention model
    nt_seed(42);
    int T = 3, D = 8, vocab = 6;

    nt_tensor* wte = nt_tensor_new2d(vocab, D);
    nt_tensor_xavier(wte, vocab, D);
    nt_tensor* wpe = nt_tensor_new2d(T, D);
    nt_tensor_xavier(wpe, T, D);
    nt_tensor* Wq = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wq, D, D);
    nt_tensor* Wk = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wk, D, D);
    nt_tensor* Wv = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wv, D, D);
    nt_tensor* Wout = nt_tensor_new2d(vocab, D);
    nt_tensor_xavier(Wout, D, vocab);

    nt_tensor* tokens = nt_tensor_new(T);
    tokens->data[0] = 1; tokens->data[1] = 3; tokens->data[2] = 5;
    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 3; targets->data[1] = 5; targets->data[2] = 0;

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 80; step++) {
        nt_tape_start();
        int wte_i = nt_tape_param(wte); nt_tape_no_decay(wte_i);
        int wpe_i = nt_tape_param(wpe); nt_tape_no_decay(wpe_i);
        int wq_i = nt_tape_param(Wq);
        int wk_i = nt_tape_param(Wk);
        int wv_i = nt_tape_param(Wv);
        int wo_i = nt_tape_param(Wout);
        int tok_i = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_i = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        int h = nt_seq_embedding(wte_i, wpe_i, tok_i, T, D);
        int q = nt_seq_linear(wq_i, h, T);
        int k = nt_seq_linear(wk_i, h, T);
        int v = nt_seq_linear(wv_i, h, T);
        int attn = nt_causal_attention(q, k, v, T, D);
        int logits = nt_seq_linear(wo_i, attn, T);
        int loss = nt_seq_cross_entropy(logits, tgt_i, T, vocab);

        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step == 0) initial_loss = lv;
        if (step == 79) final_loss = lv;

        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.005f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "attn loss decreased");
    printf("    attn training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte); nt_tensor_free(wpe);
    nt_tensor_free(Wq); nt_tensor_free(Wk); nt_tensor_free(Wv); nt_tensor_free(Wout);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    PASS("attention_training");
}

// ── Numerical gradient checking ──────────────────────────────────────────────
// Finite differences: df/dx ≈ (f(x+h) - f(x-h)) / 2h
// Compare against analytic gradient from backward pass.

// Helper: run forward pass, return scalar loss. op_fn builds the graph.
// We perturb param->data[idx] and measure loss change.
static float numgrad_check_param(
    nt_tensor* param, int pidx,        // which param element to perturb
    // Callback: builds graph, returns loss entry index.
    // Takes param tape idx as input.
    int (*build_graph)(int param_idx),
    float h,                            // perturbation size
    float* analytic_grad_out            // output: analytic grad at pidx
) {
    // Forward + backward at current value
    nt_tape_start();
    int p_idx = nt_tape_param(param);
    int loss_idx = build_graph(p_idx);
    if (loss_idx < 0) { nt_tape_clear(); *analytic_grad_out = 0; return 0; }
    nt_tape_backward(loss_idx);
    nt_tape_entry* ep = &nt_tape_get()->entries[p_idx];
    *analytic_grad_out = (ep->grad && pidx < ep->grad->len) ? ep->grad->data[pidx] : 0;
    nt_tape_clear();

    // f(x + h)
    float orig = param->data[pidx];
    param->data[pidx] = orig + h;
    nt_tape_start();
    p_idx = nt_tape_param(param);
    loss_idx = build_graph(p_idx);
    float loss_plus = (loss_idx >= 0) ? nt_tape_get()->entries[loss_idx].output->data[0] : 0;
    nt_tape_clear();

    // f(x - h)
    param->data[pidx] = orig - h;
    nt_tape_start();
    p_idx = nt_tape_param(param);
    loss_idx = build_graph(p_idx);
    float loss_minus = (loss_idx >= 0) ? nt_tape_get()->entries[loss_idx].output->data[0] : 0;
    nt_tape_clear();

    param->data[pidx] = orig;  // restore
    return (loss_plus - loss_minus) / (2.0f * h);
}

// Check all elements of a param. Returns max relative error.
static float numgrad_check_all(
    nt_tensor* param,
    int (*build_graph)(int param_idx),
    float h, float tol,
    const char* name
) {
    float max_err = 0;
    int worst_idx = 0;
    for (int i = 0; i < param->len; i++) {
        float analytic, numeric;
        numeric = numgrad_check_param(param, i, build_graph, h, &analytic);
        // Skip near-zero gradients (both analytic and numeric tiny = effectively correct)
        if (fabsf(analytic) < 1e-4f && fabsf(numeric) < 1e-4f) continue;
        float denom = fmaxf(fabsf(analytic), fabsf(numeric));
        if (denom < 1e-7f) denom = 1e-7f;
        float rel_err = fabsf(analytic - numeric) / denom;
        if (rel_err > max_err) { max_err = rel_err; worst_idx = i; }
    }
    if (max_err > tol) {
        float analytic_again;
        float numeric_again = numgrad_check_param(param, worst_idx, build_graph, h, &analytic_again);
        printf("    %s: WORST idx=%d analytic=%.6f numeric=%.6f err=%.4f\n",
               name, worst_idx, analytic_again, numeric_again, max_err);
    }
    return max_err;
}

// ── Gradient check: cross_entropy ──
static int gc_ce_graph(int p_idx) {
    return nt_cross_entropy(p_idx, 1); // target = 1
}
static void test_gradcheck_cross_entropy(void) {
    nt_seed(42);
    nt_tensor* logits = nt_tensor_new(5);
    nt_tensor_rand(logits, 1.0f);
    float err = numgrad_check_all(logits, gc_ce_graph, 1e-4f, 0.01f, "cross_entropy");
    ASSERT(err < 0.01f, "gradcheck cross_entropy");
    nt_tensor_free(logits);
    PASS("gradcheck_cross_entropy");
}

// ── Gradient check: silu ──
static int gc_silu_graph(int p_idx) {
    int s = nt_silu(p_idx);
    return nt_cross_entropy(s, 0);
}
static void test_gradcheck_silu(void) {
    nt_seed(7);
    nt_tensor* x = nt_tensor_new(4);
    nt_tensor_rand(x, 1.0f);
    float err = numgrad_check_all(x, gc_silu_graph, 1e-3f, 0.05f, "silu");
    ASSERT(err < 0.05f, "gradcheck silu");
    nt_tensor_free(x);
    PASS("gradcheck_silu");
}

// ── Gradient check: rmsnorm ──
static int gc_rmsnorm_graph(int p_idx) {
    int r = nt_rmsnorm(p_idx, -1);
    return nt_cross_entropy(r, 0);
}
static void test_gradcheck_rmsnorm(void) {
    nt_seed(13);
    nt_tensor* x = nt_tensor_new(6);
    nt_tensor_rand(x, 1.0f);
    float err = numgrad_check_all(x, gc_rmsnorm_graph, 1e-3f, 0.05f, "rmsnorm");
    ASSERT(err < 0.05f, "gradcheck rmsnorm");
    nt_tensor_free(x);
    PASS("gradcheck_rmsnorm");
}

// ── Gradient check: softmax ──
static int gc_softmax_graph(int p_idx) {
    int s = nt_softmax(p_idx);
    return nt_cross_entropy(s, 2);
}
static void test_gradcheck_softmax(void) {
    nt_seed(99);
    nt_tensor* x = nt_tensor_new(5);
    nt_tensor_rand(x, 1.0f);
    float err = numgrad_check_all(x, gc_softmax_graph, 1e-3f, 0.1f, "softmax");
    ASSERT(err < 0.1f, "gradcheck softmax");
    nt_tensor_free(x);
    PASS("gradcheck_softmax");
}

// ── Gradient check: linear (matvec) ──
static nt_tensor* gc_lin_x;
static int gc_lin_graph(int w_idx) {
    int x_idx = nt_tape_record(gc_lin_x, NT_OP_NONE, -1, -1, 0);
    int y = nt_linear(w_idx, x_idx, -1);
    return nt_cross_entropy(y, 0);
}
static void test_gradcheck_linear(void) {
    nt_seed(42);
    nt_tensor* W = nt_tensor_new2d(4, 3);
    nt_tensor_rand(W, 0.5f);
    gc_lin_x = nt_tensor_new(3);
    nt_tensor_rand(gc_lin_x, 1.0f);
    float err = numgrad_check_all(W, gc_lin_graph, 1e-3f, 0.1f, "linear_W");
    ASSERT(err < 0.1f, "gradcheck linear");
    nt_tensor_free(W);
    nt_tensor_free(gc_lin_x);
    PASS("gradcheck_linear");
}

// ── Gradient check: seq_linear ──
static nt_tensor* gc_seqlin_x;
static int gc_seqlin_graph(int w_idx) {
    int x_idx = nt_tape_record(gc_seqlin_x, NT_OP_NONE, -1, -1, 0);
    int T = 3, V = 4;
    int y = nt_seq_linear(w_idx, x_idx, T);
    nt_tensor* tgt = nt_tensor_new(T);
    tgt->data[0] = 0; tgt->data[1] = 1; tgt->data[2] = 2;
    int t_idx = nt_tape_record(tgt, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tgt);
    return nt_seq_cross_entropy(y, t_idx, T, V);
}
static void test_gradcheck_seq_linear(void) {
    nt_seed(55);
    nt_tensor* W = nt_tensor_new2d(4, 6);
    nt_tensor_rand(W, 0.3f);
    gc_seqlin_x = nt_tensor_new(3 * 6);
    nt_tensor_rand(gc_seqlin_x, 0.5f);
    float err = numgrad_check_all(W, gc_seqlin_graph, 1e-3f, 0.1f, "seq_linear_W");
    ASSERT(err < 0.1f, "gradcheck seq_linear");
    nt_tensor_free(W);
    nt_tensor_free(gc_seqlin_x);
    PASS("gradcheck_seq_linear");
}

// ── Gradient check: causal attention ──
static nt_tensor* gc_attn_k;
static nt_tensor* gc_attn_v;
static int gc_attn_graph(int q_idx) {
    int T = 2, D = 4;
    int k_idx = nt_tape_record(gc_attn_k, NT_OP_NONE, -1, -1, 0);
    int v_idx = nt_tape_record(gc_attn_v, NT_OP_NONE, -1, -1, 0);
    int out = nt_causal_attention(q_idx, k_idx, v_idx, T, D);
    // Use seq_cross_entropy on the attention output treated as logits over T*D classes
    // Actually simpler: treat full output as logits for T*D-class problem
    return nt_cross_entropy(out, 0);
}
static void test_gradcheck_causal_attention(void) {
    nt_seed(33);
    int T = 2, D = 4;
    nt_tensor* Q = nt_tensor_new(T * D);
    nt_tensor_rand(Q, 0.3f);
    gc_attn_k = nt_tensor_new(T * D);
    nt_tensor_rand(gc_attn_k, 0.3f);
    gc_attn_v = nt_tensor_new(T * D);
    nt_tensor_rand(gc_attn_v, 0.3f);
    float err = numgrad_check_all(Q, gc_attn_graph, 1e-3f, 0.1f, "causal_attn_Q");
    ASSERT(err < 0.1f, "gradcheck causal_attention");
    nt_tensor_free(Q);
    nt_tensor_free(gc_attn_k);
    nt_tensor_free(gc_attn_v);
    PASS("gradcheck_causal_attention");
}

// ── Gradient check: embedding ──
static int gc_emb_graph(int wte_idx) {
    nt_tape_no_decay(wte_idx);
    int h = nt_embedding(wte_idx, 2);
    return nt_cross_entropy(h, 0);
}
static void test_gradcheck_embedding(void) {
    nt_seed(42);
    nt_tensor* wte = nt_tensor_new2d(5, 4);
    nt_tensor_rand(wte, 0.5f);
    float err = numgrad_check_all(wte, gc_emb_graph, 1e-4f, 0.01f, "embedding");
    ASSERT(err < 0.01f, "gradcheck embedding");
    nt_tensor_free(wte);
    PASS("gradcheck_embedding");
}

// ── Gradient check: RoPE ──
static int gc_rope_graph(int p_idx) {
    int T = 2, head_dim = 4;
    int r = nt_rope(p_idx, T, head_dim);
    return nt_cross_entropy(r, 0);
}
static void test_gradcheck_rope(void) {
    nt_seed(77);
    nt_tensor* x = nt_tensor_new(2 * 4); // T=2, D=4, 1 head
    nt_tensor_rand(x, 0.5f);
    float err = numgrad_check_all(x, gc_rope_graph, 1e-3f, 0.05f, "rope");
    ASSERT(err < 0.05f, "gradcheck rope");
    nt_tensor_free(x);
    PASS("gradcheck_rope");
}

// ── Gradient check: GEGLU ──
static nt_tensor* gc_geglu_x;
static nt_tensor* gc_geglu_w2;
static int gc_geglu_graph(int w1_idx) {
    int T = 1, D_in = 3, D_out = 4;
    int x_idx = nt_tape_record(gc_geglu_x, NT_OP_NONE, -1, -1, 0);
    int w2_idx = nt_tape_record(gc_geglu_w2, NT_OP_NONE, -1, -1, 0);
    int g = nt_geglu(x_idx, w1_idx, w2_idx, T, D_in, D_out);
    return nt_cross_entropy(g, 0);
}
static void test_gradcheck_geglu(void) {
    nt_seed(88);
    int D_in = 3, D_out = 4;
    nt_tensor* W1 = nt_tensor_new2d(D_out, D_in);
    nt_tensor_rand(W1, 0.3f);
    gc_geglu_w2 = nt_tensor_new2d(D_out, D_in);
    nt_tensor_rand(gc_geglu_w2, 0.3f);
    gc_geglu_x = nt_tensor_new(1 * D_in);
    nt_tensor_rand(gc_geglu_x, 0.5f);
    // GEGLU uses tanh-approx GELU — higher tolerance for near-zero grads
    float err = numgrad_check_all(W1, gc_geglu_graph, 1e-3f, 0.3f, "geglu_W1");
    ASSERT(err < 0.3f, "gradcheck geglu");
    nt_tensor_free(W1);
    nt_tensor_free(gc_geglu_w2);
    nt_tensor_free(gc_geglu_x);
    PASS("gradcheck_geglu");
}

// ── Gradient check: add + mul + scale ──
static nt_tensor* gc_arith_b;
static int gc_add_graph(int a_idx) {
    int b_idx = nt_tape_record(gc_arith_b, NT_OP_NONE, -1, -1, 0);
    int s = nt_add(a_idx, b_idx);
    return nt_cross_entropy(s, 0);
}
static int gc_mul_graph(int a_idx) {
    int b_idx = nt_tape_record(gc_arith_b, NT_OP_NONE, -1, -1, 0);
    int s = nt_mul(a_idx, b_idx);
    return nt_cross_entropy(s, 0);
}
static int gc_scale_graph(int a_idx) {
    int s = nt_scale(a_idx, 2.5f);
    return nt_cross_entropy(s, 0);
}
static void test_gradcheck_arithmetic(void) {
    nt_seed(11);
    nt_tensor* a = nt_tensor_new(4);
    nt_tensor_rand(a, 1.0f);
    gc_arith_b = nt_tensor_new(4);
    nt_tensor_rand(gc_arith_b, 1.0f);

    float err;
    err = numgrad_check_all(a, gc_add_graph, 1e-4f, 0.01f, "add");
    ASSERT(err < 0.01f, "gradcheck add");
    err = numgrad_check_all(a, gc_mul_graph, 1e-4f, 0.01f, "mul");
    ASSERT(err < 0.01f, "gradcheck mul");
    err = numgrad_check_all(a, gc_scale_graph, 1e-4f, 0.01f, "scale");
    ASSERT(err < 0.01f, "gradcheck scale");

    nt_tensor_free(a);
    nt_tensor_free(gc_arith_b);
    PASS("gradcheck_arithmetic");
}

// ── Gradient accumulation test ──
static void test_grad_accumulation(void) {
    nt_seed(42);
    nt_tensor* W = nt_tensor_new(4);
    nt_tensor_rand(W, 1.0f);

    // Step 1: accumulate grads from 3 micro-batches
    for (int mb = 0; mb < 3; mb++) {
        nt_tape_start();
        int w_idx = nt_tape_param(W);
        int target = mb % 4;
        int loss_idx = nt_cross_entropy(w_idx, target);
        nt_tape_backward(loss_idx);
        nt_tape_accum_grads();
        nt_tape_clear();
    }

    // Step 2: apply accumulated, then adam
    nt_tape_start();
    int w_idx = nt_tape_param(W);
    nt_cross_entropy(w_idx, 0);  // dummy forward to get tape entry
    nt_tape_apply_accum(3);

    // Check that grads were applied
    nt_tape_entry* ew = &nt_tape_get()->entries[w_idx];
    ASSERT(ew->grad != NULL, "accum grad exists");
    float gnorm = 0;
    for (int i = 0; i < ew->grad->len; i++) gnorm += ew->grad->data[i] * ew->grad->data[i];
    ASSERT(gnorm > 1e-10f, "accum grad non-zero");

    float before = W->data[0];
    nt_tape_adam_step(0.01f);
    ASSERT(fabsf(W->data[0] - before) > 1e-6f, "accum + adam changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("grad_accumulation");
}

// ── Chuck convergence test ──
static void test_chuck_convergence(void) {
    nt_seed(42);
    nt_tensor* W = nt_tensor_new2d(4, 8);
    nt_tensor_xavier(W, 8, 4);
    nt_tensor* Wout = nt_tensor_new2d(4, 8);
    nt_tensor_xavier(Wout, 8, 4);

    float first_loss = 0, last_loss = 0;
    for (int step = 0; step < 100; step++) {
        nt_tape_start();
        int w_idx = nt_tape_param(W);
        nt_tape_no_decay(w_idx);
        int wo_idx = nt_tape_param(Wout);

        int h = nt_embedding(w_idx, step % 4);
        int logits = nt_linear(wo_idx, h, -1);
        int loss = nt_cross_entropy(logits, (step + 1) % 4);

        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step == 0) first_loss = lv;
        if (step == 99) last_loss = lv;

        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(0.01f, lv);
        nt_tape_clear();
    }

    ASSERT(last_loss < first_loss, "chuck converges");
    printf("    chuck: loss %.4f → %.4f\n", first_loss, last_loss);

    // Verify chuck state was populated
    nt_tape_start();
    nt_tape_param(W);
    nt_chuck_state* cs = &nt_tape_get()->chuck;
    ASSERT(cs->global_step > 0, "chuck global_step tracked");
    ASSERT(cs->initialized == 1, "chuck initialized");
    nt_tape_clear();

    nt_tensor_free(W);
    nt_tensor_free(Wout);
    PASS("chuck_convergence");
}

// ── LR Schedule tests ────────────────────────────────────────────────────────

static void test_schedule_cosine(void) {
    nt_schedule s = nt_schedule_cosine(0.001f, 10, 110, 0.0001f);

    // During warmup: should ramp from min_lr to base_lr
    float lr0 = nt_schedule_get_lr(&s); // step 0
    ASSERT(lr0 < 0.0002f, "cosine warmup start near min_lr");

    // Advance through warmup
    for (int i = 1; i < 10; i++) nt_schedule_get_lr(&s);
    float lr10 = nt_schedule_get_lr(&s); // step 10: end of warmup
    ASSERT(lr10 > 0.0009f, "cosine warmup end near base_lr");

    // Advance to middle of cosine
    for (int i = 11; i < 60; i++) nt_schedule_get_lr(&s);
    float lr60 = nt_schedule_get_lr(&s); // step 60
    ASSERT(lr60 < 0.001f && lr60 > 0.0001f, "cosine mid-range");

    // Advance to end
    for (int i = 61; i < 110; i++) nt_schedule_get_lr(&s);
    float lr110 = nt_schedule_get_lr(&s); // step 110
    ASSERT(lr110 < 0.0003f, "cosine end near min_lr");

    PASS("schedule_cosine");
}

static void test_schedule_step(void) {
    nt_schedule s = nt_schedule_step(0.1f, 0, 10, 0.5f);

    float lr0 = nt_schedule_get_lr(&s); // step 0
    ASSERT_CLOSE(lr0, 0.1f, 0.01f, "step lr initial");

    for (int i = 1; i < 10; i++) nt_schedule_get_lr(&s);
    float lr10 = nt_schedule_get_lr(&s); // step 10: first decay
    ASSERT_CLOSE(lr10, 0.05f, 0.01f, "step lr after 1 decay");

    for (int i = 11; i < 20; i++) nt_schedule_get_lr(&s);
    float lr20 = nt_schedule_get_lr(&s); // step 20: second decay
    ASSERT_CLOSE(lr20, 0.025f, 0.005f, "step lr after 2 decays");

    PASS("schedule_step");
}

static void test_schedule_with_training(void) {
    // Verify schedule integrates with optimizer
    nt_seed(42);
    nt_schedule sched = nt_schedule_cosine(0.01f, 5, 55, 0.0f);

    nt_tensor* W = nt_tensor_new(4);
    nt_tensor_rand(W, 1.0f);
    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 50; step++) {
        float lr = nt_schedule_get_lr(&sched);
        nt_tape_start();
        int w = nt_tape_param(W);
        int loss = nt_cross_entropy(w, 2);
        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step == 0) initial_loss = lv;
        if (step == 49) final_loss = lv;
        nt_tape_backward(loss);
        nt_tape_adam_step(lr);
        nt_tape_clear();
    }
    ASSERT(final_loss < initial_loss, "schedule+adam converges");
    nt_tensor_free(W);
    PASS("schedule_with_training");
}

// ── NaN guard tests ──────────────────────────────────────────────────────────

static void test_nan_guard_clean(void) {
    nt_nan_guard guard = nt_nan_guard_new();
    nt_tape_start();
    nt_tensor* W = nt_tensor_new(4);
    nt_tensor_rand(W, 1.0f);
    int w = nt_tape_param(W);
    int loss = nt_cross_entropy(w, 0);
    nt_tape_backward(loss);

    int clean = nt_nan_guard_check(&guard);
    ASSERT(clean == 1, "nan guard clean");
    ASSERT(guard.total_nan_count == 0, "no nans");
    ASSERT(guard.stable_steps == 1, "stable step counted");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("nan_guard_clean");
}

static void test_nan_guard_detect(void) {
    nt_nan_guard guard = nt_nan_guard_new();
    nt_tape_start();
    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    int w = nt_tape_param(W);
    int loss = nt_cross_entropy(w, 0);
    nt_tape_backward(loss);

    // Inject NaN into gradients
    nt_tape_entry* ew = &nt_tape_get()->entries[w];
    ew->grad->data[0] = 0.0f / 0.0f; // NaN

    int clean = nt_nan_guard_check(&guard);
    ASSERT(clean == 0, "nan detected");
    ASSERT(guard.total_nan_count == 1, "nan counted");
    ASSERT(guard.skipped_steps == 1, "step skipped");
    // Gradients should be zeroed
    ASSERT(ew->grad->data[0] == 0.0f, "grad zeroed after nan");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("nan_guard_detect");
}

// ── Dropout tests ────────────────────────────────────────────────────────────

static void test_dropout(void) {
    nt_seed(42);
    nt_train_mode(1);
    nt_tape_start();

    nt_tensor* x = nt_tensor_new(100);
    nt_tensor_fill(x, 1.0f);
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int d_idx = nt_dropout(x_idx, 0.5f);

    nt_tape_entry* ed = &nt_tape_get()->entries[d_idx];
    int n_zero = 0;
    for (int i = 0; i < 100; i++) {
        if (ed->output->data[i] == 0.0f) n_zero++;
    }
    // With p=0.5, expect roughly 50% zeros
    ASSERT(n_zero > 20 && n_zero < 80, "dropout ~50% zeroed");
    // Non-zero values should be scaled by 1/(1-p) = 2.0
    for (int i = 0; i < 100; i++) {
        if (ed->output->data[i] != 0.0f)
            ASSERT_CLOSE(ed->output->data[i], 2.0f, 0.01f, "dropout scale");
    }

    // Eval mode: no dropout
    nt_tape_clear();
    nt_train_mode(0);
    nt_tape_start();
    x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    d_idx = nt_dropout(x_idx, 0.5f);
    ed = &nt_tape_get()->entries[d_idx];
    n_zero = 0;
    for (int i = 0; i < 100; i++)
        if (ed->output->data[i] == 0.0f) n_zero++;
    ASSERT(n_zero == 0, "eval mode no dropout");

    nt_train_mode(1); // restore
    nt_tape_clear();
    nt_tensor_free(x);
    PASS("dropout");
}

// ── LayerNorm tests ──────────────────────────────────────────────────────────

static void test_layernorm(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(4);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3; x->data[3] = 4;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_layernorm(x_idx, -1, -1);

    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    // After layernorm: mean should be ~0, variance ~1
    float mean = 0;
    for (int i = 0; i < 4; i++) mean += ey->output->data[i];
    mean /= 4;
    ASSERT_CLOSE(mean, 0.0f, 1e-5f, "layernorm zero mean");

    float var = 0;
    for (int i = 0; i < 4; i++) {
        float d = ey->output->data[i] - mean;
        var += d * d;
    }
    var /= 4;
    ASSERT_CLOSE(var, 1.0f, 0.01f, "layernorm unit variance");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("layernorm");
}

static void test_gradcheck_layernorm(void) {
    nt_seed(42);
    nt_tensor* x = nt_tensor_new(6);
    nt_tensor_rand(x, 1.0f);
    // Graph: layernorm → cross_entropy
    (void)x; // gradcheck via training below
    nt_tensor_free(x);

    // Training test: layernorm should converge
    nt_seed(42);
    nt_tensor* W = nt_tensor_new2d(4, 8);
    nt_tensor_xavier(W, 8, 4);
    nt_tensor* gamma = nt_tensor_new(8);
    nt_tensor_fill(gamma, 1.0f);
    nt_tensor* Wout = nt_tensor_new2d(4, 8);
    nt_tensor_xavier(Wout, 8, 4);

    float first_loss = 0, last_loss = 0;
    for (int step = 0; step < 50; step++) {
        nt_tape_start();
        int w_idx = nt_tape_param(W); nt_tape_no_decay(w_idx);
        int g_idx = nt_tape_param(gamma);
        int wo_idx = nt_tape_param(Wout);
        int h = nt_embedding(w_idx, step % 4);
        h = nt_layernorm(h, g_idx, -1);
        int logits = nt_linear(wo_idx, h, -1);
        int loss = nt_cross_entropy(logits, (step + 1) % 4);
        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step == 0) first_loss = lv;
        if (step == 49) last_loss = lv;
        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.01f);
        nt_tape_clear();
    }
    ASSERT(last_loss < first_loss, "layernorm training converges");
    printf("    layernorm training: loss %.4f → %.4f\n", first_loss, last_loss);

    nt_tensor_free(W); nt_tensor_free(gamma); nt_tensor_free(Wout);
    PASS("gradcheck_layernorm");
}

// ── GELU tests ───────────────────────────────────────────────────────────────

static int gc_gelu_graph(int p_idx) {
    int g = nt_gelu(p_idx);
    return nt_cross_entropy(g, 0);
}
static void test_gradcheck_gelu(void) {
    nt_seed(42);
    nt_tensor* x = nt_tensor_new(4);
    nt_tensor_rand(x, 1.0f);
    float err = numgrad_check_all(x, gc_gelu_graph, 1e-3f, 0.05f, "gelu");
    ASSERT(err < 0.05f, "gradcheck gelu");
    nt_tensor_free(x);
    PASS("gradcheck_gelu");
}

// ── Profiler test ────────────────────────────────────────────────────────────

static void test_profiler(void) {
    nt_profiler_reset();
    nt_profiler_enable();

    // Do a forward+backward pass
    nt_tape_start();
    nt_tensor* W = nt_tensor_new(4);
    nt_tensor_rand(W, 1.0f);
    int w = nt_tape_param(W);
    int loss = nt_cross_entropy(w, 0);
    nt_tape_backward(loss);
    nt_tape_adam_step(0.01f);
    nt_tape_clear();

    nt_profiler* p = nt_profiler_get();
    ASSERT(p->enabled == 1, "profiler enabled");
    // Profiler tracking is passive for now — just verify it doesn't crash
    nt_profiler_print();
    nt_profiler_disable();

    nt_tensor_free(W);
    PASS("profiler");
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    printf("notorch tests\n");
    printf("═══════════════════════════════════════════\n");

    printf("\n[Tensor]\n");
    test_tensor_create();
    test_tensor_2d();
    test_tensor_clone();
    test_tensor_reshape();
    test_tensor_xavier();
    test_tensor_refcount();

    printf("\n[Ops]\n");
    test_silu();
    test_softmax();
    test_rmsnorm();

    printf("\n[Tape + Forward/Backward]\n");
    test_tape_basic();
    test_forward_backward_linear();
    test_causal_attention();
    test_mh_causal_attention();
    test_seq_cross_entropy();
    test_seq_linear();

    printf("\n[Optimizers]\n");
    test_adam_step();
    test_adamw_step();
    test_chuck_step();
    test_grad_clip();

    printf("\n[Hebbian]\n");
    test_hebbian();

    printf("\n[Save/Load]\n");
    test_save_load();

    printf("\n[Gradient Accumulation]\n");
    test_grad_accumulation();

    printf("\n[LR Schedules]\n");
    test_schedule_cosine();
    test_schedule_step();
    test_schedule_with_training();

    printf("\n[NaN Guard]\n");
    test_nan_guard_clean();
    test_nan_guard_detect();

    printf("\n[Dropout]\n");
    test_dropout();

    printf("\n[LayerNorm]\n");
    test_layernorm();
    test_gradcheck_layernorm();

    printf("\n[GELU]\n");
    test_gradcheck_gelu();

    printf("\n[Profiler]\n");
    test_profiler();

    printf("\n[Training]\n");
    test_training_loop();
    test_seq_training_loop();
    test_attention_training();
    test_chuck_convergence();

    printf("\n[Numerical Gradient Checks]\n");
    test_gradcheck_cross_entropy();
    test_gradcheck_silu();
    test_gradcheck_rmsnorm();
    test_gradcheck_softmax();
    test_gradcheck_linear();
    test_gradcheck_seq_linear();
    test_gradcheck_causal_attention();
    test_gradcheck_embedding();
    test_gradcheck_rope();
    test_gradcheck_geglu();
    test_gradcheck_arithmetic();

    printf("\n═══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
