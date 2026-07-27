/* q6_i8_micro — Q6_K int8-activation dot vs the f32-activation reference.
 *
 * nt_qmatvec (dtype 14) dequantizes to f32 and is the exact reference; nt_qmatvec_i8
 * quantizes the activation to per-32 int8 first. Any difference between them is the
 * ACTIVATION quantization alone — the weights are the same bytes, read by the same
 * unpack. A packing or sub-scale-indexing mistake in the i8 kernel does not produce a
 * small relative error, it produces a scrambled dot, so the magnitude here is the test.
 *
 * Random Q6 payloads are legal by construction (any ql/qh/sc bytes decode); d is pinned
 * to 2^-7 so the reference stays in a sane range instead of riding a random f16.
 *
 * build: cc -O2 -I. tools/q6_i8_micro.c libnotorch.a -lm [-framework Accelerate]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "notorch.h"

int main(void) {
    const int k = 2048, m = 64, nb = k / 256;
    uint8_t *W = malloc((size_t)m * nb * 210);
    float *x = malloc((size_t)k * sizeof(float));
    float *y_ref = malloc((size_t)m * sizeof(float));
    float *y_i8 = malloc((size_t)m * sizeof(float));
    if (!W || !x || !y_ref || !y_i8) return 1;
    srand(7);
    for (long i = 0; i < (long)m * nb; i++) {
        uint8_t *b = W + i * 210;
        for (int j = 0; j < 192; j++) b[j] = (uint8_t)(rand() & 0xFF);
        for (int j = 0; j < 16; j++) b[192 + j] = (uint8_t)(rand() % 96 + 1);  /* sc in [1,96] */
        b[208] = 0x00; b[209] = 0x20;                                          /* d = 2^-7 f16 */
    }
    for (int i = 0; i < k; i++) x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    if (nt_qmatvec(y_ref, W, 14, x, m, k) != 0) { puts("nt_qmatvec refused"); return 1; }
    if (nt_qmatvec_i8(y_i8, W, 14, x, m, k) != 0) { puts("nt_qmatvec_i8 refused"); return 1; }

    double num = 0, den = 0, dot = 0, na = 0, nbn = 0, mx = 0;
    for (int i = 0; i < m; i++) {
        double dref = y_ref[i], di8 = y_i8[i], e = fabs(di8 - dref);
        num += e; den += fabs(dref); if (e > mx) mx = e;
        dot += di8 * dref; na += di8 * di8; nbn += dref * dref;
    }
    printf("Q6_K i8-act vs f32-act reference: rows=%d k=%d\n", m, k);
    printf("  relL1 = %.6f   max|diff| = %.4e   cos(f64) = %.10f\n",
           num / den, mx, dot / (sqrt(na) * sqrt(nbn)));
    printf("  verdict: %s\n", (num / den) < 0.02
           ? "activation-quant scale (expected)" : "TOO LARGE — kernel/index bug");
    /* dtype/shape guards */
    printf("  guard k%%256!=0 -> %d (want -1)\n", nt_qmatvec_i8(y_i8, W, 14, x, m, 2048 - 32));
    printf("  guard dtype 12  -> %d (want -1)\n", nt_qmatvec_i8(y_i8, W, 12, x, m, k));

    /* Threading: rows are disjoint, so the fan-out must be bit-identical to one thread.
     * The floor is cached on first use, so the two sides are compared ACROSS runs via a
     * checksum — run once with NT_QMV_THREAD_MIN=1 and once with a huge floor. m*k here
     * clears the default 4M floor so the threaded path is the one under test. */
    const int mb = 2048;
    uint8_t *WB = malloc((size_t)mb * nb * 210);
    float *yb = malloc((size_t)mb * sizeof(float));
    if (WB && yb) {
        for (long i = 0; i < (long)mb * nb; i++) {
            uint8_t *b = WB + i * 210;
            for (int j = 0; j < 192; j++) b[j] = (uint8_t)(rand() & 0xFF);
            for (int j = 0; j < 16; j++) b[192 + j] = (uint8_t)(rand() % 96 + 1);
            b[208] = 0x00; b[209] = 0x20;
        }
        if (nt_qmatvec_i8(yb, WB, 14, x, mb, k) == 0) {
            double s = 0, mxv = 0;
            for (int i = 0; i < mb; i++) { s += yb[i]; if (fabs(yb[i]) > mxv) mxv = fabs(yb[i]); }
            const char *e = getenv("NT_QMV_THREAD_MIN");
            printf("  threaded m=%d k=%d (floor=%s): checksum %.9f max %.9f\n",
                   mb, k, e ? e : "default 4M", s, mxv);
        }
    }
    free(WB); free(yb);
    free(W); free(x); free(y_ref); free(y_i8);
    return (num / den) < 0.02 ? 0 : 2;
}
