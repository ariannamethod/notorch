// test_rrpram_lr.c — finite-difference grad check for nt_rrpram_lowrank_attention.
// Build: cc -O2 -DUSE_BLAS -I. -o test_rrpram_lr test_rrpram_lr.c notorch.c -lm -lopenblas
// Uses nt_cross_entropy on output[0] for scalar loss reduction (same pattern as
// existing test_notorch.c gradchecks).

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float rand_uniform(void) { return ((float)((rand() & 0xFFFF) / 65535.0f) - 0.5f) * 2.0f; }

static nt_tape_entry* tget(int idx) { return &nt_tape_get()->entries[idx]; }

/* Globals for forward-only re-runs at perturbed θ */
static int gT = 6, gE = 8, gH = 2, ghd = 4;
static nt_tensor* gWr;
static nt_tensor* gX;
static nt_tensor* gV;

static float forward_loss(void) {
    nt_tape_start();
    int wr = nt_tape_param(gWr);
    int x  = nt_tape_param(gX);
    int v  = nt_tape_param(gV);
    int o  = nt_rrpram_lowrank_attention(wr, x, v, gT, gE, gH, ghd);
    int l  = nt_cross_entropy(o, 0);
    float L = tget(l)->output->data[0];
    nt_tape_clear();
    return L;
}

static float numgrad(float* slot) {
    float saved = *slot;
    float h = 1e-3f;
    *slot = saved + h; float L1 = forward_loss();
    *slot = saved - h; float L0 = forward_loss();
    *slot = saved;
    return (L1 - L0) / (2.0f * h);
}

int main(void) {
    int T = gT, E = gE, H = gH, hd = ghd, T_r = T, rank = 3;

    long wr_len = (long)H * rank * (E + T_r);
    long x_len  = (long)T * E;
    long v_len  = (long)T * H * hd;

    nt_seed(42);
    gWr = nt_tensor_new(wr_len);
    gX  = nt_tensor_new2d(T, E);
    gV  = nt_tensor_new(v_len);
    if (!gWr || !gX || !gV) { printf("alloc fail\n"); return 1; }

    srand(123);
    for (long i = 0; i < wr_len; i++) gWr->data[i] = rand_uniform();
    for (long i = 0; i < x_len;  i++) gX->data[i]  = rand_uniform();
    for (long i = 0; i < v_len;  i++) gV->data[i]  = rand_uniform();

    /* Analytic backward via tape */
    nt_tape_start();
    int wr_idx = nt_tape_param(gWr);
    int x_idx  = nt_tape_param(gX);
    int v_idx  = nt_tape_param(gV);
    int out_idx = nt_rrpram_lowrank_attention(wr_idx, x_idx, v_idx, T, E, H, hd);
    if (out_idx < 0) { printf("FATAL: forward returned -1\n"); return 2; }
    int loss_idx = nt_cross_entropy(out_idx, 0);
    printf("forward OK: loss=%.6f\n", tget(loss_idx)->output->data[0]);
    nt_tape_backward(loss_idx);

    nt_tape_entry* ewr = tget(wr_idx);
    nt_tape_entry* ex  = tget(x_idx);
    nt_tape_entry* ev  = tget(v_idx);

    if (!ewr->grad || !ex->grad || !ev->grad) {
        printf("FATAL: missing grads. ewr=%p ex=%p ev=%p\n",
               (void*)(ewr->grad), (void*)(ex->grad), (void*)(ev->grad));
        return 3;
    }

    /* Copy analytic grads into local arrays — forward_loss() below will reset tape */
    float* g_wr = (float*)malloc(wr_len * sizeof(float));
    float* g_x  = (float*)malloc(x_len  * sizeof(float));
    float* g_v  = (float*)malloc(v_len  * sizeof(float));
    for (long i = 0; i < wr_len; i++) g_wr[i] = ewr->grad->data[i];
    for (long i = 0; i < x_len;  i++) g_x[i]  = ex->grad->data[i];
    for (long i = 0; i < v_len;  i++) g_v[i]  = ev->grad->data[i];
    nt_tape_clear();

    int n_check = 8;
    /* Random spot-check indexes */
    long idx_wr[] = { 1, 11, 23, 45, 67, 89, 109, wr_len - 2 };
    long idx_x[]  = { 0, 5, 11, 17, 23, 29, 35, x_len - 1 };
    long idx_v[]  = { 0, 3, 8, 16, 23, 30, 37, v_len - 1 };

    int fails = 0; double max_rel = 0;

    printf("\nWr grads:\n");
    for (int k = 0; k < n_check; k++) {
        long i = idx_wr[k]; if (i < 0 || i >= wr_len) continue;
        float num = numgrad(&gWr->data[i]);
        float ana = g_wr[i];
        float abs_diff = fabsf(num - ana);
        float rel = abs_diff / (fabsf(num) + fabsf(ana) + 1e-6f);
        int ok = (rel < 8e-2f) || (abs_diff < 5e-4f);   /* FP noise floor */
        printf("  Wr[%ld]: num=%+.5f ana=%+.5f rel=%.2e %s\n",
               i, num, ana, rel, ok ? "OK" : "FAIL");
        if (rel > max_rel) max_rel = rel;
        if (!ok) fails++;
    }

    printf("\nX grads:\n");
    for (int k = 0; k < n_check; k++) {
        long i = idx_x[k]; if (i < 0 || i >= x_len) continue;
        float num = numgrad(&gX->data[i]);
        float ana = g_x[i];
        float abs_diff = fabsf(num - ana);
        float rel = abs_diff / (fabsf(num) + fabsf(ana) + 1e-6f);
        int ok = (rel < 8e-2f) || (abs_diff < 5e-4f);   /* FP noise floor */
        printf("  X[%ld]:  num=%+.5f ana=%+.5f rel=%.2e %s\n",
               i, num, ana, rel, ok ? "OK" : "FAIL");
        if (rel > max_rel) max_rel = rel;
        if (!ok) fails++;
    }

    printf("\nV grads:\n");
    for (int k = 0; k < n_check; k++) {
        long i = idx_v[k]; if (i < 0 || i >= v_len) continue;
        float num = numgrad(&gV->data[i]);
        float ana = g_v[i];
        float abs_diff = fabsf(num - ana);
        float rel = abs_diff / (fabsf(num) + fabsf(ana) + 1e-6f);
        int ok = (rel < 8e-2f) || (abs_diff < 5e-4f);   /* FP noise floor */
        printf("  V[%ld]:  num=%+.5f ana=%+.5f rel=%.2e %s\n",
               i, num, ana, rel, ok ? "OK" : "FAIL");
        if (rel > max_rel) max_rel = rel;
        if (!ok) fails++;
    }

    printf("\n══════════════════════════════════════════\n");
    printf("  max_rel_diff = %.2e   fails = %d\n", max_rel, fails);
    printf("  result: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}
