/*
 * test_rrpram_broadcast.c — synthetic gradient check for nt_rrpram_broadcast_attention.
 *
 * Tiny dims (T=4, E=8, R=2, H=2, hd=4) keep ε-sweep cheap.
 * For every element of Wr_combined / x / v we compute analytic gradient
 * (one backward pass with grad-of-loss = ones into out) and finite-difference
 * gradient ((L(x+ε) - L(x-ε)) / 2ε), where L = Σ out. They must match
 * within 1e-3 absolute error, else the broadcast op is wrong.
 *
 * Build (CPU only):
 *   cc -O2 -I. tests/test_rrpram_broadcast.c notorch.c -o /tmp/test_rrpram_bcast -lm
 *
 * Run:
 *   /tmp/test_rrpram_bcast
 *   exits 0 if all gates pass, 1 otherwise.
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define T_LEN   4
#define E_LEN   8
#define R_LEN   2
#define H_LEN   2
#define HD_LEN  4
#define OUT_DIM (H_LEN * HD_LEN)

#define EPS_FD     1e-3f
#define TOL_ABS    1e-3f
#define TOL_REL    5e-3f

static float frand_unit(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float forward_only(const float* wr_data, long wr_len,
                          const float* x_data,  int t_len, int e_len,
                          const float* v_data,  int out_dim) {
    nt_tape_start();

    nt_tensor* tw = nt_tensor_new(wr_len);
    memcpy(tw->data, wr_data, (size_t)wr_len * sizeof(float));
    int wr_idx = nt_tape_param(tw);
    nt_tape_freeze_param(wr_idx);

    nt_tensor* tx = nt_tensor_new(t_len * e_len);
    memcpy(tx->data, x_data, (size_t)t_len * e_len * sizeof(float));
    int x_idx = nt_tape_param(tx);
    nt_tape_freeze_param(x_idx);

    nt_tensor* tv = nt_tensor_new(t_len * out_dim);
    memcpy(tv->data, v_data, (size_t)t_len * out_dim * sizeof(float));
    int v_idx = nt_tape_param(tv);
    nt_tape_freeze_param(v_idx);

    int out_idx = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                                 t_len, e_len, H_LEN, HD_LEN);
    if (out_idx < 0) { nt_tape_clear(); return NAN; }

    nt_tape_entry* po = &nt_tape_get()->entries[out_idx];
    float loss = 0.0f;
    for (long i = 0; i < po->output->len; i++) loss += po->output->data[i];

    nt_tape_clear();
    return loss;
}

static int analytic_grads(const float* wr_data, long wr_len,
                          const float* x_data,  int t_len, int e_len,
                          const float* v_data,  int out_dim,
                          float** out_dwr, float** out_dx, float** out_dv) {
    nt_tape_start();

    nt_tensor* tw = nt_tensor_new(wr_len);
    memcpy(tw->data, wr_data, (size_t)wr_len * sizeof(float));
    int wr_idx = nt_tape_param(tw);
    nt_tape_no_decay(wr_idx);

    nt_tensor* tx = nt_tensor_new(t_len * e_len);
    memcpy(tx->data, x_data, (size_t)t_len * e_len * sizeof(float));
    int x_idx = nt_tape_param(tx);
    nt_tape_no_decay(x_idx);

    nt_tensor* tv = nt_tensor_new(t_len * out_dim);
    memcpy(tv->data, v_data, (size_t)t_len * out_dim * sizeof(float));
    int v_idx = nt_tape_param(tv);
    nt_tape_no_decay(v_idx);

    int out_idx = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                                 t_len, e_len, H_LEN, HD_LEN);
    if (out_idx < 0) { nt_tape_clear(); return -1; }

    nt_tape_entry* po = &nt_tape_get()->entries[out_idx];
    if (!po->grad) {
        po->grad = nt_tensor_new(po->output->len);
    }
    for (long i = 0; i < po->grad->len; i++) po->grad->data[i] = 1.0f;

    nt_tape_backward(out_idx);

    nt_tape_entry* ew = &nt_tape_get()->entries[wr_idx];
    nt_tape_entry* ex = &nt_tape_get()->entries[x_idx];
    nt_tape_entry* ev = &nt_tape_get()->entries[v_idx];

    *out_dwr = (float*)malloc(wr_len * sizeof(float));
    *out_dx  = (float*)malloc(t_len * e_len * sizeof(float));
    *out_dv  = (float*)malloc(t_len * out_dim * sizeof(float));

    if (ew->grad) memcpy(*out_dwr, ew->grad->data, wr_len * sizeof(float));
    else memset(*out_dwr, 0, wr_len * sizeof(float));
    if (ex->grad) memcpy(*out_dx,  ex->grad->data,  t_len * e_len * sizeof(float));
    else memset(*out_dx,  0, t_len * e_len * sizeof(float));
    if (ev->grad) memcpy(*out_dv,  ev->grad->data,  t_len * out_dim * sizeof(float));
    else memset(*out_dv,  0, t_len * out_dim * sizeof(float));

    nt_tape_clear();
    return 0;
}

static int compare_grads(const char* name, const float* analytic,
                         float* x_data, long n_elem,
                         const float* wr_data, long wr_len,
                         const float* x_full,  int t_len, int e_len,
                         const float* v_data,  int out_dim,
                         int which) {
    int n_fail = 0;
    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    int max_idx = -1;

    for (long k = 0; k < n_elem; k++) {
        float orig = x_data[k];
        x_data[k] = orig + EPS_FD;
        float l_plus = forward_only(
            (which == 0) ? x_data : wr_data, wr_len,
            (which == 1) ? x_data : x_full, t_len, e_len,
            (which == 2) ? x_data : v_data, out_dim);
        x_data[k] = orig - EPS_FD;
        float l_minus = forward_only(
            (which == 0) ? x_data : wr_data, wr_len,
            (which == 1) ? x_data : x_full, t_len, e_len,
            (which == 2) ? x_data : v_data, out_dim);
        x_data[k] = orig;

        float fd  = (l_plus - l_minus) / (2.0f * EPS_FD);
        float ana = analytic[k];
        float abs_err = fabsf(fd - ana);
        float denom = fmaxf(fabsf(fd), fabsf(ana));
        float rel_err = (denom > 1e-9f) ? abs_err / denom : 0.0f;

        if (abs_err > max_abs_err) { max_abs_err = abs_err; max_idx = (int)k; }
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (abs_err > TOL_ABS && rel_err > TOL_REL) {
            if (n_fail < 5)
                printf("  [%s] FAIL k=%ld fd=%.6e ana=%.6e abs=%.6e rel=%.6e\n",
                       name, k, fd, ana, abs_err, rel_err);
            n_fail++;
        }
    }

    printf("  [%s] n=%ld max_abs=%.4e (idx %d) max_rel=%.4e fails=%d\n",
           name, n_elem, max_abs_err, max_idx, max_rel_err, n_fail);
    return n_fail;
}

int main(void) {
    srand(42);
    nt_seed(42);

    long wr_len = (long)H_LEN * E_LEN * R_LEN + (long)H_LEN * R_LEN * T_LEN;
    float* wr   = (float*)malloc(wr_len * sizeof(float));
    float* x    = (float*)malloc(T_LEN * E_LEN * sizeof(float));
    float* v    = (float*)malloc(T_LEN * OUT_DIM * sizeof(float));

    for (long i = 0; i < wr_len; i++) wr[i] = 0.1f * frand_unit();
    for (long i = 0; i < T_LEN * E_LEN; i++) x[i] = 0.5f * frand_unit();
    for (long i = 0; i < T_LEN * OUT_DIM; i++) v[i] = 0.5f * frand_unit();

    printf("=== nt_rrpram_broadcast_attention gradient check ===\n");
    printf("dims: T=%d E=%d R=%d H=%d hd=%d  wr_len=%ld\n",
           T_LEN, E_LEN, R_LEN, H_LEN, HD_LEN, wr_len);

    float *dwr = NULL, *dx = NULL, *dv = NULL;
    if (analytic_grads(wr, wr_len, x, T_LEN, E_LEN, v, OUT_DIM,
                       &dwr, &dx, &dv) < 0) {
        printf("FAIL: analytic forward returned error\n");
        return 1;
    }

    float L0 = forward_only(wr, wr_len, x, T_LEN, E_LEN, v, OUT_DIM);
    printf("L0 = %.6f (loss = sum(out))\n", L0);

    int n_fail = 0;

    float* wr_copy = (float*)malloc(wr_len * sizeof(float));
    memcpy(wr_copy, wr, wr_len * sizeof(float));
    n_fail += compare_grads("d_wr", dwr, wr_copy, wr_len,
                             wr_copy, wr_len, x, T_LEN, E_LEN, v, OUT_DIM, 0);

    float* x_copy = (float*)malloc(T_LEN * E_LEN * sizeof(float));
    memcpy(x_copy, x, T_LEN * E_LEN * sizeof(float));
    n_fail += compare_grads("d_x", dx, x_copy, T_LEN * E_LEN,
                             wr, wr_len, x_copy, T_LEN, E_LEN, v, OUT_DIM, 1);

    float* v_copy = (float*)malloc(T_LEN * OUT_DIM * sizeof(float));
    memcpy(v_copy, v, T_LEN * OUT_DIM * sizeof(float));
    n_fail += compare_grads("d_v", dv, v_copy, T_LEN * OUT_DIM,
                             wr, wr_len, x, T_LEN, E_LEN, v_copy, OUT_DIM, 2);

    free(dwr); free(dx); free(dv);
    free(wr_copy); free(x_copy); free(v_copy);
    free(wr); free(x); free(v);

    printf("\n=== %s — total fails: %d ===\n",
           n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
