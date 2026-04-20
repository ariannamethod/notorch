/*
 * test_bitnet_ops.c — gradient check для NT_OP_SWIGLU + NT_OP_BIT_LINEAR + NT_OP_BIT_SEQ_LINEAR
 *
 * Verifies analytical backward against numerical (finite-difference) gradient.
 * Builds on notorch.c's tape + nt_tensor primitives.
 *
 * Build:
 *   cc -O2 -Wall -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *       test_bitnet_ops.c notorch.c -o test_bitnet_ops \
 *       -framework Accelerate -lm
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
static int tests = 0;

static float compute_loss_sum(int out_idx) {
    /* Loss = sum of all outputs (simple scalar for grad check) */
    nt_tape_entry* e = nt_tape_get()->entries + out_idx;
    float s = 0;
    for (int i = 0; i < e->output->len; i++) s += e->output->data[i];
    return s;
}

static int check_close(const char* name, float a, float b, float rtol, float atol) {
    float diff = fabsf(a - b);
    float tol = atol + rtol * fabsf(b);
    tests++;
    if (diff > tol || !isfinite(a) || !isfinite(b)) {
        printf("  FAIL %s: analytical=%.6g numerical=%.6g diff=%.3e tol=%.3e\n",
               name, a, b, diff, tol);
        failures++;
        return 0;
    }
    return 1;
}

/* ─── helper: create parameter tensor from float array ─── */
static int make_param(const float* data, int len, int rows, int cols) {
    nt_tensor* t;
    if (rows > 0 && cols > 0) t = nt_tensor_new2d(rows, cols);
    else                       t = nt_tensor_new(len);
    memcpy(t->data, data, len * sizeof(float));
    int idx = nt_tape_param(t);
    nt_tensor_free(t);
    return idx;
}

/* ─── SwiGLU gradient check ─── */
static void test_swiglu(void) {
    printf("[swiglu] forward+backward gradient check\n");
    int N = 12;
    float gate_data[12] = { 0.5f, -1.2f, 0.3f, 2.1f, -0.7f, 0.9f,
                            -0.4f, 1.5f, -2.0f, 0.0f, 0.8f, -0.6f };
    float up_data[12]   = { 1.0f,  0.5f,-0.8f, 1.3f,  0.2f,-1.1f,
                             0.7f,-0.9f, 0.4f, 1.2f,-0.3f, 0.6f };

    /* Analytical gradient via tape.backward */
    nt_tape_start();
    int g_idx = make_param(gate_data, N, 0, 0);
    int u_idx = make_param(up_data,   N, 0, 0);
    int out_idx = nt_swiglu(g_idx, u_idx);
    nt_tape_backward(out_idx);
    nt_tape_entry* eg = nt_tape_get()->entries + g_idx;
    nt_tape_entry* eu = nt_tape_get()->entries + u_idx;
    float analytical_dg[12], analytical_du[12];
    for (int i = 0; i < N; i++) analytical_dg[i] = eg->grad->data[i];
    for (int i = 0; i < N; i++) analytical_du[i] = eu->grad->data[i];
    nt_tape_clear();

    /* Numerical gradient via finite difference */
    float eps = 1e-3f;
    for (int i = 0; i < N; i++) {
        float gp[12], gm[12];
        memcpy(gp, gate_data, sizeof(gate_data));
        memcpy(gm, gate_data, sizeof(gate_data));
        gp[i] += eps; gm[i] -= eps;

        nt_tape_start();
        int gp_idx = make_param(gp, N, 0, 0);
        int u_idx2 = make_param(up_data, N, 0, 0);
        int op = nt_swiglu(gp_idx, u_idx2);
        float Lp = compute_loss_sum(op);
        nt_tape_clear();

        nt_tape_start();
        int gm_idx = make_param(gm, N, 0, 0);
        int u_idx3 = make_param(up_data, N, 0, 0);
        int om = nt_swiglu(gm_idx, u_idx3);
        float Lm = compute_loss_sum(om);
        nt_tape_clear();

        float numerical = (Lp - Lm) / (2 * eps);
        char name[64]; snprintf(name, sizeof(name), "swiglu dgate[%d]", i);
        check_close(name, analytical_dg[i], numerical, 1e-3f, 1e-4f);
    }
    for (int i = 0; i < N; i++) {
        float up[12], um[12];
        memcpy(up, up_data, sizeof(up_data));
        memcpy(um, up_data, sizeof(up_data));
        up[i] += eps; um[i] -= eps;

        nt_tape_start();
        int g_idx2 = make_param(gate_data, N, 0, 0);
        int up_idx = make_param(up, N, 0, 0);
        int op = nt_swiglu(g_idx2, up_idx);
        float Lp = compute_loss_sum(op);
        nt_tape_clear();

        nt_tape_start();
        int g_idx3 = make_param(gate_data, N, 0, 0);
        int um_idx = make_param(um, N, 0, 0);
        int om = nt_swiglu(g_idx3, um_idx);
        float Lm = compute_loss_sum(om);
        nt_tape_clear();

        float numerical = (Lp - Lm) / (2 * eps);
        char name[64]; snprintf(name, sizeof(name), "swiglu dup[%d]", i);
        check_close(name, analytical_du[i], numerical, 1e-3f, 1e-4f);
    }
}

/* ─── BitSeqLinear gradient check ─── */
/* STE: backward treats quant as identity, so numerical grad should match analytical */
static void test_bit_seq_linear(void) {
    printf("[bit_seq_linear] forward+backward (STE) gradient check\n");
    int ROWS = 4, COLS = 6, T = 3;
    /* Random-ish but seeded for repeatability */
    float W[24] = { 0.12f,-0.34f, 0.56f,-0.11f, 0.87f, 0.25f,
                   -0.43f, 0.28f,-0.72f, 0.13f, 0.45f,-0.09f,
                    0.31f,-0.18f, 0.64f, 0.22f,-0.51f, 0.07f,
                   -0.62f, 0.41f, 0.17f,-0.29f, 0.38f, 0.95f };
    float X[18] = { 0.8f, 0.2f,-0.5f, 0.7f, 0.1f,-0.3f,
                   -0.4f, 0.6f, 0.9f,-0.2f, 0.5f, 0.3f,
                    0.1f,-0.7f, 0.4f, 0.8f,-0.1f, 0.2f };

    /* Analytical */
    nt_tape_start();
    int w_idx = make_param(W, ROWS * COLS, ROWS, COLS);
    int x_idx = make_param(X, T * COLS, T, COLS);
    int out_idx = nt_bit_seq_linear(w_idx, x_idx, T);
    nt_tape_backward(out_idx);
    nt_tape_entry* ew = nt_tape_get()->entries + w_idx;
    nt_tape_entry* ex = nt_tape_get()->entries + x_idx;
    float dW[24], dX[18];
    for (int i = 0; i < ROWS * COLS; i++) dW[i] = ew->grad->data[i];
    for (int i = 0; i < T * COLS; i++)    dX[i] = ex->grad->data[i];
    nt_tape_clear();

    /* STE means backward is analytical identity through quantization.
     * Numerical finite-difference would give 0 for dX most of the time because int8 quantization
     * rounds small perturbations back to the same bucket. That's expected behavior for STE —
     * what we verify is that analytical dX matches the STE formula (W^T @ dout using FP W).
     *
     * With loss = sum(outputs) → dout[t,i] = 1 for all t, i.
     * Expected dX[t,j] = Σ_i W[i,j] × 1 = column_sum(W)[j]  (same for all t).
     */
    float expected_dX_per_t[6] = {0};
    for (int j = 0; j < COLS; j++)
        for (int i = 0; i < ROWS; i++)
            expected_dX_per_t[j] += W[i * COLS + j];
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < COLS; j++) {
            char name[64]; snprintf(name, sizeof(name), "bit_seq_linear dX[%d,%d] (STE)", t, j);
            check_close(name, dX[t * COLS + j], expected_dX_per_t[j], 1e-5f, 1e-5f);
        }
    }
    /* For dW — STE says analytical = dout ⊗ x (full precision).
     * Let's check that analytical dW matches the standard matmul formula (sanity check). */
    float expected_dW[24] = {0};
    /* dout is all 1s (loss = sum), so dW[i,j] = Σ_t x[t,j] */
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            for (int t = 0; t < T; t++)
                expected_dW[i * COLS + j] += X[t * COLS + j];
    for (int i = 0; i < ROWS * COLS; i++) {
        char name[64]; snprintf(name, sizeof(name), "bit_seq_linear dW[%d] (STE)", i);
        check_close(name, dW[i], expected_dW[i], 1e-5f, 1e-5f);
    }
}

/* ─── BitLinear (single-position) gradient check ─── */
static void test_bit_linear(void) {
    printf("[bit_linear] forward+backward (STE) gradient check\n");
    int ROWS = 5, COLS = 8;
    float W[40] = {
         0.1f,-0.2f, 0.3f,-0.4f, 0.5f,-0.6f, 0.7f,-0.8f,
        -0.15f, 0.25f,-0.35f, 0.45f,-0.55f, 0.65f,-0.75f, 0.85f,
         0.12f, 0.22f,-0.32f, 0.42f, 0.52f,-0.62f, 0.72f,-0.82f,
        -0.05f, 0.15f,-0.25f, 0.35f,-0.45f, 0.55f,-0.65f, 0.75f,
         0.18f,-0.28f, 0.38f,-0.48f, 0.58f,-0.68f, 0.78f,-0.88f };
    float x[8] = { 0.9f, -0.4f, 0.2f, 0.7f, -0.1f, 0.5f, -0.6f, 0.3f };

    nt_tape_start();
    int w_idx = make_param(W, ROWS * COLS, ROWS, COLS);
    int x_idx = make_param(x, COLS, 0, 0);
    int out_idx = nt_bit_linear(w_idx, x_idx);
    nt_tape_backward(out_idx);
    nt_tape_entry* ew = nt_tape_get()->entries + w_idx;
    nt_tape_entry* ex = nt_tape_get()->entries + x_idx;
    float dW[40], dx[8];
    for (int i = 0; i < ROWS * COLS; i++) dW[i] = ew->grad->data[i];
    for (int i = 0; i < COLS; i++) dx[i] = ex->grad->data[i];
    nt_tape_clear();

    /* STE expected: dW[i,j] = dout[i] * x[j] = 1 * x[j] */
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            char name[64]; snprintf(name, sizeof(name), "bit_linear dW[%d,%d]", i, j);
            check_close(name, dW[i * COLS + j], x[j], 1e-5f, 1e-5f);
        }
    }
    /* STE expected: dx[j] = Σ_i W[i,j] * dout[i] = Σ_i W[i,j] */
    for (int j = 0; j < COLS; j++) {
        float expected = 0;
        for (int i = 0; i < ROWS; i++) expected += W[i * COLS + j];
        char name[64]; snprintf(name, sizeof(name), "bit_linear dx[%d]", j);
        check_close(name, dx[j], expected, 1e-5f, 1e-5f);
    }
}

/* ─── SPA smoke test (no grad, just verify mathematical sanity) ─── */
static void test_spa(void) {
    printf("[spa] smoke test (embed + connectedness + modulate)\n");
    int V = 10, D = 4, n_tokens = 3;
    float W_embed[40] = {
        0.1f, 0.2f, 0.3f, 0.4f,    0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,    1.3f, 1.4f, 1.5f, 1.6f,
        1.7f, 1.8f, 1.9f, 2.0f,    2.1f, 2.2f, 2.3f, 2.4f,
        2.5f, 2.6f, 2.7f, 2.8f,    2.9f, 3.0f, 3.1f, 3.2f,
        3.3f, 3.4f, 3.5f, 3.6f,    3.7f, 3.8f, 3.9f, 4.0f
    };
    int tokens[3] = { 0, 5, 9 };
    float emb[4] = { 0 };
    nt_spa_embed_sentence(tokens, n_tokens, W_embed, V, D, 0.85f, emb);
    tests++;
    int ok = 1;
    for (int d = 0; d < D; d++) if (!isfinite(emb[d])) ok = 0;
    if (!ok) { printf("  FAIL spa embed not finite\n"); failures++; }

    /* Recency bias: with alpha=0.85, token 9 (most recent) should dominate. */
    /* sanity: embedding should be closer to tok9 row (3.7,3.8,3.9,4.0) than tok0 row (0.1...) */
    float d9 = (emb[0]-3.7f)*(emb[0]-3.7f) + (emb[1]-3.8f)*(emb[1]-3.8f);
    float d0 = (emb[0]-0.1f)*(emb[0]-0.1f) + (emb[1]-0.2f)*(emb[1]-0.2f);
    tests++;
    if (d9 >= d0) {
        printf("  FAIL spa embed recency: d9=%.3f d0=%.3f (emb should be closer to recent tok9)\n", d9, d0);
        failures++;
    }

    /* Connectedness — self-query should give max attention = 1 (with only 1 entry in history = self) */
    float conn = nt_spa_connectedness(emb, D, emb, 1);
    tests++;
    if (fabsf(conn - 1.0f) > 1e-3f) {
        printf("  FAIL spa connectedness self: got %.4f, expected 1.0\n", conn);
        failures++;
    }

    /* Modulate: connectedness=1 with strength=0.3 should scale logits by 1/(1-0.3)=1.428 */
    float logits[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    nt_spa_modulate_logits(logits, 5, 1.0f, 0.3f);
    float expected0 = 1.0f / (1.0f - 0.3f);
    tests++;
    if (fabsf(logits[0] - expected0) > 1e-3f) {
        printf("  FAIL spa modulate: logit[0]=%.4f expected %.4f\n", logits[0], expected0);
        failures++;
    }
}

int main(void) {
    printf("═══ BitNet/SwiGLU/SPA grad + smoke tests ═══\n");
    nt_seed(42);
    test_swiglu();
    test_bit_seq_linear();
    test_bit_linear();
    test_spa();
    printf("\n═══ %d tests, %d failures ═══\n", tests, failures);
    return failures ? 1 : 0;
}
