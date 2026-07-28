/* q4k_i8_micro — Q4_K int8-activation dot vs the f32-activation reference.
 *
 * nt_qmatvec (dtype 12) dequantizes to f32 and is the exact reference; nt_qmatvec_i8
 * quantizes the activation to per-32 int8 first. The only difference between them is that
 * activation quantization — same weight bytes, same unpack, same (scale,min) pairs. An
 * indexing or minus-term mistake does not produce a small relative error, it produces a
 * scrambled dot, so the magnitude here is the test.
 *
 * Also checks that the fan-out never changes a result: rows are disjoint, so the threaded
 * and single-threaded paths must agree bit for bit, and the floor can be set either through
 * the API or the environment — four cuts of the same rows, one checksum.
 *
 * build: cc -O2 -I. tools/q4k_i8_micro.c libnotorch.a -lm [-framework Accelerate]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "notorch.h"

static void fill_q4k(uint8_t *W, long nblk, unsigned seed) {
    srand(seed);
    for (long i = 0; i < nblk; i++) {
        uint8_t *b = W + i * 144;
        b[0] = 0x00; b[1] = 0x20;              /* d    = 2^-7 in f16 */
        b[2] = 0x00; b[3] = 0x1C;              /* dmin = 2^-9 in f16 */
        for (int j = 4; j < 144; j++) b[j] = (uint8_t)(rand() & 0xFF);
    }
}

int main(void) {
    const int k = 2048, m = 64, nblk_row = k / 256;
    uint8_t *W = malloc((size_t)m * nblk_row * 144);
    float *x = malloc((size_t)k * sizeof(float));
    float *y_ref = malloc((size_t)m * sizeof(float));
    float *y_i8 = malloc((size_t)m * sizeof(float));
    if (!W || !x || !y_ref || !y_i8) return 1;
    fill_q4k(W, (long)m * nblk_row, 7);
    for (int i = 0; i < k; i++) x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    if (nt_qmatvec(y_ref, W, 12, x, m, k) != 0) { puts("nt_qmatvec refused dtype 12"); return 1; }
    if (nt_qmatvec_i8(y_i8, W, 12, x, m, k) != 0) { puts("nt_qmatvec_i8 refused dtype 12"); return 1; }

    double num = 0, den = 0, dot = 0, na = 0, nb = 0, mx = 0;
    for (int i = 0; i < m; i++) {
        double a = y_i8[i], r = y_ref[i], e = fabs(a - r);
        num += e; den += fabs(r); if (e > mx) mx = e;
        dot += a * r; na += a * a; nb += r * r;
    }
    printf("Q4_K i8-act vs f32-act reference: rows=%d k=%d\n", m, k);
    printf("  relL1 = %.6f   max|diff| = %.4e   cos(f64) = %.10f\n",
           num / den, mx, dot / (sqrt(na) * sqrt(nb)));
    printf("  verdict: %s\n", (num / den) < 0.02
           ? "activation-quant scale (expected)" : "TOO LARGE — kernel/index bug");
    printf("  guard k%%256 -> %d, guard dtype 13 -> %d (both want -1)\n",
           nt_qmatvec_i8(y_i8, W, 12, x, m, k - 32), nt_qmatvec_i8(y_i8, W, 13, x, m, k));

    /* threading: same rows, four ways to cut them */
    const int mb = 2048;
    uint8_t *WB = malloc((size_t)mb * nblk_row * 144);
    float *yb = malloc((size_t)mb * sizeof(float));
    if (WB && yb) {
        fill_q4k(WB, (long)mb * nblk_row, 11);
        const char *how = getenv("MICRO_FLOOR_API");
        if (how) nt_qmv_set_thread_min(atol(how));        /* API wins over the environment */
        if (nt_qmatvec_i8(yb, WB, 12, x, mb, k) == 0) {
            double s = 0, mxv = 0;
            for (int i = 0; i < mb; i++) { s += yb[i]; if (fabs(yb[i]) > mxv) mxv = fabs(yb[i]); }
            const char *e = getenv("NT_QMV_THREAD_MIN");
            printf("  threaded m=%d (API=%s env=%s): checksum %.9f max %.9f\n",
                   mb, how ? how : "unset", e ? e : "unset", s, mxv);
        }
    }
    free(WB); free(yb); free(W); free(x); free(y_ref); free(y_i8);
    return (num / den) < 0.02 ? 0 : 2;
}
