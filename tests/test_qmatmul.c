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

/* Q4_K super-block: d, dmin (f16), 12 bytes of packed 6-bit (scale,min) pairs, 128 nibble
 * bytes. The scale bytes are left random on purpose — every 6-bit pair the packing can
 * produce is a legal weight, and pinning them would test one arm of nt_get_scale_min_k4. */
static uint8_t *make_q4_k(int m, int k, unsigned seed) {
    long nb = k / 256;
    uint8_t *W = (uint8_t *)malloc((size_t)m * nb * 144);
    if (!W) return NULL;
    srand(seed);
    for (long r = 0; r < m; r++)
        for (long b = 0; b < nb; b++) {
            uint8_t *bl = W + (r * nb + b) * 144;
            bl[0] = 0x66; bl[1] = 0x2A;                 /* d    ~0.05 */
            bl[2] = 0x00; bl[3] = 0x24;                 /* dmin ~0.016 */
            for (int i = 4; i < 144; i++) bl[i] = (uint8_t)(rand() & 0xFF);
        }
    return W;
}

/* Q5_0: f16 scale, a 32-bit high-bit mask, 16 nibble bytes. Q8_0: f16 scale, 32 int8. */
static uint8_t *make_block32(int m, int k, int bytes, unsigned seed) {
    long nb = k / 32;
    uint8_t *W = (uint8_t *)malloc((size_t)m * nb * bytes);
    if (!W) return NULL;
    srand(seed);
    for (long r = 0; r < m; r++)
        for (long b = 0; b < nb; b++) {
            uint8_t *bl = W + (r * nb + b) * bytes;
            bl[0] = 0x66; bl[1] = 0x2A;                 /* f16 ~0.05 */
            for (int i = 2; i < bytes; i++) bl[i] = (uint8_t)(rand() & 0xFF);
        }
    return W;
}

static uint8_t *make_weights(int dtype, int m, int k, unsigned seed) {
    switch (dtype) {
    case 2:  return make_q4_0(m, k, seed);
    case 6:  return make_block32(m, k, 22, seed);
    case 8:  return make_block32(m, k, 34, seed);
    default: return make_q4_k(m, k, seed);
    }
}

static int check(int dtype, int m, int k, int n, int *pass, int *fail) {
    uint8_t *W = make_weights(dtype, m, k, 1234u + (unsigned)n);
    float *X   = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *ref = (float *)malloc(sizeof(float) * (size_t)m * n);
    float *got = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (!W || !X || !ref || !got) { printf("  alloc failed\n"); (*fail)++; return -1; }

    for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);

    for (int j = 0; j < n; j++)
        if (nt_qmatvec_i8(ref + (long)j * m, W, dtype, X + (long)j * k, m, k) != 0) {
            printf("  dtype=%d m=%d k=%d n=%d: reference matvec refused\n", dtype, m, k, n); (*fail)++;
            free(W); free(X); free(ref); free(got); return -1;
        }
    if (nt_qmatmul_i8(got, W, dtype, X, m, k, n) != 0) {
        printf("  dtype=%d m=%d k=%d n=%d: batched matmul refused\n", dtype, m, k, n); (*fail)++;
        free(W); free(X); free(ref); free(got); return -1;
    }

    long bad = 0;
    for (long i = 0; i < (long)m * n; i++)
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) bad++;
    if (bad) {
        printf("  FAIL dtype=%d m=%d k=%d n=%d: %ld of %ld outputs differ (first ref=%g got=%g)\n",
               dtype, m, k, n, bad, (long)m * n, ref[0], got[0]);
        (*fail)++;
    } else {
        printf("  PASS dtype=%2d m=%4d k=%4d n=%3d: %ld outputs identical\n", dtype, m, k, n, (long)m * n);
        (*pass)++;
    }
    free(W); free(X); free(ref); free(got);
    return bad ? -1 : 0;
}

/* Not an assertion, a number to look at: what the single unpack actually buys. */
static void timing(int dtype, int m, int k, int n) {
    uint8_t *W = make_weights(dtype, m, k, 7u);
    float *X   = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *out = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (W && X && out) {
        for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX);
        double t0 = now_s();
        for (int j = 0; j < n; j++)
            nt_qmatvec_i8(out + (long)j * m, W, dtype, X + (long)j * k, m, k);
        double per_token = now_s() - t0;
        double t1 = now_s();
        nt_qmatmul_i8(out, W, dtype, X, m, k, n);
        double batched = now_s() - t1;
        printf("  timing dtype=%2d m=%d k=%d n=%d: per-token %.1f ms, batched %.1f ms (%.2fx)\n",
               dtype, m, k, n, per_token * 1e3, batched * 1e3,
               batched > 0 ? per_token / batched : 0.0);
    }
    free(W); free(X); free(out);
}

int main(void) {
    printf("notorch batched packed matmul (Q4_0 = dtype 2, Q4_K = dtype 12)\n");
    int pass = 0, fail = 0;

    check(2,  64,   256,  1, &pass, &fail);   /* n = 1 delegates to the matvec entry */
    check(2,  64,   256,  3, &pass, &fail);
    check(2,  128,  512, 31, &pass, &fail);   /* just under the tile */
    check(2,  128,  512, 32, &pass, &fail);   /* exactly the tile */
    check(2,  128,  512, 33, &pass, &fail);   /* one past it: two passes, second is short */
    check(2,  2048, 4096, 8, &pass, &fail);   /* over the threading gate: fan-out engaged */

    check(12, 64,   256,  1, &pass, &fail);
    check(12, 64,   256,  3, &pass, &fail);
    check(12, 128,  512, 31, &pass, &fail);
    check(12, 128,  512, 32, &pass, &fail);
    check(12, 128,  512, 33, &pass, &fail);
    check(12, 2048, 4096, 8, &pass, &fail);
    check(12, 256, 1024, 17, &pass, &fail);   /* k with four super-blocks, odd tile */

    check(6,  64,   256,  3, &pass, &fail);
    check(6,  128,  512, 32, &pass, &fail);
    check(6,  128,  512, 33, &pass, &fail);
    check(6,  2048, 4096, 8, &pass, &fail);
    check(6,  4864,  896, 32, &pass, &fail);  /* a real FFN shape: k is not a multiple of 256 */

    check(8,  64,   256,  3, &pass, &fail);
    check(8,  128,  512, 33, &pass, &fail);
    check(8,  2048, 4096, 8, &pass, &fail);

    timing(2,  2048, 4096, 32);
    timing(6,  2048, 4096, 32);
    timing(8,  2048, 4096, 32);
    timing(12, 2048, 4096, 32);

    printf("\nResults: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
