/*
 * test_qmatmul.c — the batched packed matvec must agree with the per-token one exactly.
 *
 * nt_qmatmul_i8 exists to stop re-reading the weights once per token, and the only way
 * that is a free win is if it changes nothing else: same activation quantization, same
 * per-row accumulation order, therefore the same float out of the same adds. So the
 * assertion here is equality of the bit pattern, not a tolerance — a tolerance would let
 * a reordered accumulation through, and a reordered accumulation is exactly the bug this
 * kernel invites.
 *
 * Shapes cover the tile boundary (n below, at, and above NT_QMM_TILE = 32) and both sides
 * of the threading gate, because the row fan-out is where a batched kernel goes wrong.
 *
 * Build: cc -O2 -I. tests/test_qmatmul.c notorch.c -lm [-DUSE_BLAS -lopenblas]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_s(void) { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec / 1e6; }

/* Q4_0 blocks with a plausible scale — a random f16 would hand us inf/NaN half the time
 * and the equality check would pass on payloads nobody will ever run. */
static uint8_t *make_q4_0(int m, int k, unsigned seed) {
    long nb = k / 32;
    uint8_t *W = (uint8_t *)malloc((size_t)m * nb * 18);
    if (!W) return NULL;
    srand(seed);
    for (long r = 0; r < m; r++)
        for (long b = 0; b < nb; b++) {
            uint8_t *bl = W + (r * nb + b) * 18;
            bl[0] = 0x66; bl[1] = 0x2A;                 /* f16 ~0.05 */
            for (int i = 0; i < 16; i++) bl[2 + i] = (uint8_t)(rand() & 0xFF);
        }
    return W;
}

static int check(int m, int k, int n, int *pass, int *fail) {
    uint8_t *W = make_q4_0(m, k, 1234u + (unsigned)n);
    float *X   = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *ref = (float *)malloc(sizeof(float) * (size_t)m * n);
    float *got = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (!W || !X || !ref || !got) { printf("  alloc failed\n"); (*fail)++; return -1; }

    for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);

    for (int j = 0; j < n; j++)
        if (nt_qmatvec_i8(ref + (long)j * m, W, 2, X + (long)j * k, m, k) != 0) {
            printf("  m=%d k=%d n=%d: reference matvec refused\n", m, k, n); (*fail)++;
            free(W); free(X); free(ref); free(got); return -1;
        }
    if (nt_qmatmul_i8(got, W, 2, X, m, k, n) != 0) {
        printf("  m=%d k=%d n=%d: batched matmul refused\n", m, k, n); (*fail)++;
        free(W); free(X); free(ref); free(got); return -1;
    }

    long bad = 0;
    for (long i = 0; i < (long)m * n; i++)
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) bad++;
    if (bad) {
        printf("  FAIL m=%d k=%d n=%d: %ld of %ld outputs differ (first ref=%g got=%g)\n",
               m, k, n, bad, (long)m * n, ref[0], got[0]);
        (*fail)++;
    } else {
        printf("  PASS m=%d k=%d n=%d: %ld outputs identical\n", m, k, n, (long)m * n);
        (*pass)++;
    }
    free(W); free(X); free(ref); free(got);
    return bad ? -1 : 0;
}

int main(void) {
    printf("notorch batched packed matmul (Q4_0)\n");
    int pass = 0, fail = 0;

    check(64,   256,  1, &pass, &fail);   /* n = 1 delegates to the matvec entry */
    check(64,   256,  3, &pass, &fail);
    check(128,  512, 31, &pass, &fail);   /* just under the tile */
    check(128,  512, 32, &pass, &fail);   /* exactly the tile */
    check(128,  512, 33, &pass, &fail);   /* one past it: two passes, second is short */
    check(2048, 4096, 8, &pass, &fail);   /* over the threading gate: fan-out engaged */

    /* Not an assertion, a number to look at: how much the single unpack actually buys. */
    int m = 2048, k = 4096, n = 32;
    uint8_t *W = make_q4_0(m, k, 7u);
    float *X   = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *out = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (W && X && out) {
        for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX);
        double t0 = now_s();
        for (int j = 0; j < n; j++) nt_qmatvec_i8(out + (long)j * m, W, 2, X + (long)j * k, m, k);
        double per_token = now_s() - t0;
        double t1 = now_s();
        nt_qmatmul_i8(out, W, 2, X, m, k, n);
        double batched = now_s() - t1;
        printf("  timing m=%d k=%d n=%d: per-token %.1f ms, batched %.1f ms (%.2fx)\n",
               m, k, n, per_token * 1e3, batched * 1e3,
               batched > 0 ? per_token / batched : 0.0);
    }
    free(W); free(X); free(out);

    printf("\nResults: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
