/*
 * test_rrpram_broadcast.c — gradient check for nt_rrpram_broadcast_attention.
 *
 * Adversarial config: T_input=3 < ctx_T=7 (packed weight size > runtime sequence).
 * This is the regression case that the original test missed (T_packed == T_input
 * synthetic shape coincidentally produced the right answer with the old buggy
 * rank/stride inference).
 *
 * Layout: Wr_combined = [H,E,R] concat [H,R,ctx_T] — packed at full ctx, sliced
 * at runtime to T_input columns of Wr_b. New API takes `rank` explicitly,
 * derives `ctx_T` from combined_len / (H*rank) - n_embd.
 *
 * Build (CPU only):
 *   cc -O2 -I. tests/test_rrpram_broadcast.c notorch.c -o /tmp/test_rrpram_bcast -lm
 * Run:  /tmp/test_rrpram_bcast  ; exits 0 if all gates pass, 1 otherwise.
 *
 * Negative control: revert notorch broadcast op fix, rebuild, run — must FAIL.
 * Re-apply fix, rebuild, run — must PASS. If both PASS, test has no diagnostic
 * power (the original failure mode).
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Adversarial: T_input < CTX_T (packed > runtime), H*HD=E invariant. */
#define T_LEN    3
#define CTX_T    7
#define E_LEN    8
#define R_LEN    2
#define H_LEN    2
#define HD_LEN   4
#define OUT_DIM  (H_LEN * HD_LEN)

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

    nt_tensor* tx = nt_tensor_new((size_t)t_len * e_len);
    memcpy(tx->data, x_data, (size_t)t_len * e_len * sizeof(float));
    int x_idx = nt_tape_param(tx);
    nt_tape_freeze_param(x_idx);

    nt_tensor* tv = nt_tensor_new((size_t)t_len * out_dim);
    memcpy(tv->data, v_data, (size_t)t_len * out_dim * sizeof(float));
    int v_idx = nt_tape_param(tv);
    nt_tape_freeze_param(v_idx);

    int out_idx = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                                 t_len, e_len, H_LEN, HD_LEN, R_LEN);
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

    nt_tensor* tx = nt_tensor_new((size_t)t_len * e_len);
    memcpy(tx->data, x_data, (size_t)t_len * e_len * sizeof(float));
    int x_idx = nt_tape_param(tx);
    nt_tape_no_decay(x_idx);

    nt_tensor* tv = nt_tensor_new((size_t)t_len * out_dim);
    memcpy(tv->data, v_data, (size_t)t_len * out_dim * sizeof(float));
    int v_idx = nt_tape_param(tv);
    nt_tape_no_decay(v_idx);

    int out_idx = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                                 t_len, e_len, H_LEN, HD_LEN, R_LEN);
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

static int invalid_shape_checks(void) {
    int failures = 0;
    long wr_len = (long)H_LEN * E_LEN * R_LEN + (long)H_LEN * R_LEN * CTX_T;

    nt_tape_start();
    nt_tensor* tw = nt_tensor_new(wr_len);
    int wr_idx = nt_tape_param(tw);
    nt_tensor* tx = nt_tensor_new(T_LEN * E_LEN - 1);
    int x_idx = nt_tape_param(tx);
    nt_tensor* tv = nt_tensor_new(T_LEN * OUT_DIM);
    int v_idx = nt_tape_param(tv);
    int rc = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                           T_LEN, E_LEN, H_LEN, HD_LEN, R_LEN);
    if (rc >= 0) { printf("FAIL: short x accepted\n"); failures++; }
    nt_tape_clear();

    nt_tape_start();
    tw = nt_tensor_new(wr_len);
    wr_idx = nt_tape_param(tw);
    tx = nt_tensor_new(T_LEN * E_LEN);
    x_idx = nt_tape_param(tx);
    tv = nt_tensor_new(T_LEN * OUT_DIM - 1);
    v_idx = nt_tape_param(tv);
    rc = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                       T_LEN, E_LEN, H_LEN, HD_LEN, R_LEN);
    if (rc >= 0) { printf("FAIL: short v accepted\n"); failures++; }
    nt_tape_clear();

    nt_tape_start();
    tw = nt_tensor_new(wr_len - 1);
    wr_idx = nt_tape_param(tw);
    tx = nt_tensor_new(T_LEN * E_LEN);
    x_idx = nt_tape_param(tx);
    tv = nt_tensor_new(T_LEN * OUT_DIM);
    v_idx = nt_tape_param(tv);
    rc = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                       T_LEN, E_LEN, H_LEN, HD_LEN, R_LEN);
    if (rc >= 0) { printf("FAIL: bad packed Wr accepted\n"); failures++; }
    nt_tape_clear();

    printf("invalid shape checks: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures;
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

/* Sentinel-coded check: pack Wr_b with position-tag values so that wrong stride
 * (T_input vs ctx_T) produces dramatically different output. Independently
 * compute expected output via direct formula, compare to op output. */
static int sentinel_layout_check(void) {
    long wr_len = (long)H_LEN * E_LEN * R_LEN + (long)H_LEN * R_LEN * CTX_T;
    float* wr = (float*)calloc(wr_len, sizeof(float));
    float* x  = (float*)calloc(T_LEN * E_LEN, sizeof(float));
    float* v  = (float*)calloc(T_LEN * OUT_DIM, sizeof(float));
    if (!wr || !x || !v) { free(wr); free(x); free(v); return -1; }

    /* Wr_a[h,e,r] = 1 only for r==0, else 0. So mid[h,r=0] = Σ_t,e x[t,e]. */
    long wra_total = (long)H_LEN * E_LEN * R_LEN;
    for (int h = 0; h < H_LEN; h++)
        for (int e = 0; e < E_LEN; e++)
            wr[(long)h*E_LEN*R_LEN + (long)e*R_LEN + 0] = 1.0f;

    /* Wr_b[h,r=0,j] = (j+1) * 100 + h*10. Sentinel = position-coded.
     * Wrong stride (T_input=3 vs ctx_T=7) reads from wr_a region or earlier
     * positions, producing distinct output values. */
    for (int h = 0; h < H_LEN; h++)
        for (int j = 0; j < CTX_T; j++)
            wr[wra_total + (long)h*R_LEN*CTX_T + (long)0*CTX_T + j] = (j+1)*100.0f + h*10.0f;

    /* x[t,e] = 1 for all (so mid[h,0] = T_LEN * E_LEN regardless of h). */
    for (int t = 0; t < T_LEN; t++)
        for (int e = 0; e < E_LEN; e++)
            x[t*E_LEN+e] = 1.0f;
    /* v[j, h*hd+d] = 10*j + h + 0.01*d — position-coded so wrong attn weights
     * (from wrong stride read) yield distinct output values, NOT just 1. */
    for (int j = 0; j < T_LEN; j++)
        for (int h = 0; h < H_LEN; h++)
            for (int d = 0; d < HD_LEN; d++)
                v[j*OUT_DIM + h*HD_LEN + d] = 10.0f*j + (float)h + 0.01f*(float)d;

    /* Expected (correct ctx_T stride):
     *   mid[h, r=0] = Σ_t,e x[t,e] * Wr_a[h,e,r=0] = T*E*1 = 24 (all r=0 weights = 1).
     *   mid[h, r=1] = 0 (all r=1 weights = 0).
     *   score[h, j] = mid[h,0] * Wr_b[h,0,j] = 24 * ((j+1)*100 + h*10).
     *   scaled = score * sc = score / sqrt(4) = score / 2.
     *   With T=3 positions, score values are large → softmax peaks at j=last visible.
     *   For i=0: j∈{0}; attn[0,0]=1; out[0,h*hd+d] = v[0,h*hd+d] = 0+h+0.01d
     *   For i=1: j∈{0,1}; softmax dominated by j=1 (larger score); out ≈ v[1,h,d] = 10+h+0.01d
     *   For i=2: j∈{0,1,2}; softmax dominated by j=2; out ≈ v[2,h,d] = 20+h+0.01d
     * Wrong stride (T_input=3 vs ctx_T=7) reads from wrong Wr_b columns →
     * different score values → different attn dist → different output values
     * → max_diff != 0. */
    nt_tape_start();
    nt_tensor* tw = nt_tensor_new(wr_len);
    memcpy(tw->data, wr, (size_t)wr_len * sizeof(float));
    int wr_idx = nt_tape_param(tw); nt_tape_freeze_param(wr_idx);
    nt_tensor* tx = nt_tensor_new((size_t)T_LEN * E_LEN);
    memcpy(tx->data, x, (size_t)T_LEN * E_LEN * sizeof(float));
    int x_idx = nt_tape_param(tx); nt_tape_freeze_param(x_idx);
    nt_tensor* tv = nt_tensor_new((size_t)T_LEN * OUT_DIM);
    memcpy(tv->data, v, (size_t)T_LEN * OUT_DIM * sizeof(float));
    int v_idx = nt_tape_param(tv); nt_tape_freeze_param(v_idx);

    int out_idx = nt_rrpram_broadcast_attention(wr_idx, x_idx, v_idx,
                                                 T_LEN, E_LEN, H_LEN, HD_LEN, R_LEN);
    if (out_idx < 0) {
        printf("  [sentinel] op returned -1 (likely shape mismatch reject — investigate)\n");
        nt_tape_clear(); free(wr); free(x); free(v);
        return -1;
    }

    nt_tape_entry* po = &nt_tape_get()->entries[out_idx];
    int n_fail = 0;
    float max_diff = 0.0f;
    /* Compute analytic expected output via direct ctx_T stride formula.
     * (wra_total declared at line 186 above.) */
    float sc = 1.0f / sqrtf((float)HD_LEN);
    float* expected = (float*)calloc(T_LEN * OUT_DIM, sizeof(float));
    for (int h = 0; h < H_LEN; h++) {
        float mid[R_LEN];
        for (int r = 0; r < R_LEN; r++) mid[r] = 0.0f;
        for (int t = 0; t < T_LEN; t++)
            for (int e = 0; e < E_LEN; e++)
                for (int r = 0; r < R_LEN; r++)
                    mid[r] += x[t*E_LEN+e] * wr[(long)h*E_LEN*R_LEN + (long)e*R_LEN + r];
        float scores[T_LEN];
        for (int j = 0; j < T_LEN; j++) {
            float s = 0.0f;
            for (int r = 0; r < R_LEN; r++)
                s += mid[r] * wr[wra_total + (long)h*R_LEN*CTX_T + (long)r*CTX_T + j];
            scores[j] = s * sc;
        }
        for (int i = 0; i < T_LEN; i++) {
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) if (scores[j] > mx) mx = scores[j];
            float sm = 0.0f;
            float att[T_LEN] = {0};
            for (int j = 0; j <= i; j++) { att[j] = expf(scores[j] - mx); sm += att[j]; }
            if (sm > 0) for (int j = 0; j <= i; j++) att[j] /= sm;
            for (int d = 0; d < HD_LEN; d++) {
                float o = 0.0f;
                for (int j = 0; j <= i; j++)
                    o += att[j] * v[j*OUT_DIM + h*HD_LEN + d];
                expected[i*OUT_DIM + h*HD_LEN + d] = o;
            }
        }
    }
    for (long i = 0; i < po->output->len; i++) {
        float diff = fabsf(po->output->data[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) n_fail++;
    }
    free(expected);
    printf("  [sentinel] expected vs op output (position-coded v), max_diff=%.4e fails=%d\n",
           max_diff, n_fail);

    nt_tape_clear();
    free(wr); free(x); free(v);
    return n_fail;
}

int main(void) {
    srand(42);
    nt_seed(42);

    long wr_len = (long)H_LEN * E_LEN * R_LEN + (long)H_LEN * R_LEN * CTX_T;
    float* wr   = (float*)malloc(wr_len * sizeof(float));
    float* x    = (float*)malloc(T_LEN * E_LEN * sizeof(float));
    float* v    = (float*)malloc(T_LEN * OUT_DIM * sizeof(float));

    for (long i = 0; i < wr_len; i++) wr[i] = 0.1f * frand_unit();
    for (long i = 0; i < T_LEN * E_LEN; i++) x[i] = 0.5f * frand_unit();
    for (long i = 0; i < T_LEN * OUT_DIM; i++) v[i] = 0.5f * frand_unit();

    printf("=== nt_rrpram_broadcast_attention gradient check (adversarial) ===\n");
    printf("dims: T_input=%d ctx_T=%d E=%d R=%d H=%d hd=%d  wr_len=%ld\n",
           T_LEN, CTX_T, E_LEN, R_LEN, H_LEN, HD_LEN, wr_len);
    printf("(T_input != ctx_T — exposes rank/stride layout bugs)\n");

    int n_fail = 0;

    printf("\n--- sentinel layout check ---\n");
    int sentinel = sentinel_layout_check();
    n_fail += (sentinel < 0) ? 1 : sentinel;

    printf("\n--- invalid shape checks ---\n");
    n_fail += invalid_shape_checks();

    printf("\n--- finite-diff gradient check ---\n");
    float *dwr = NULL, *dx = NULL, *dv = NULL;
    if (analytic_grads(wr, wr_len, x, T_LEN, E_LEN, v, OUT_DIM,
                       &dwr, &dx, &dv) < 0) {
        printf("FAIL: analytic forward returned error\n");
        free(wr); free(x); free(v);
        return 1;
    }

    float L0 = forward_only(wr, wr_len, x, T_LEN, E_LEN, v, OUT_DIM);
    printf("L0 = %.6f (loss = sum(out))\n", L0);

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
