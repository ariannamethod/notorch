/* test_f16_matvec.c — the unpacked formats, which had no gate until their kernels changed.
 *
 * F16 and F32 reach nt_qmatvec rather than the i8 entry, and every other dtype in this tree
 * is covered by a test that the packed ones share. These two were not, which was survivable
 * while both were a scalar loop over one element at a time and stopped being so the moment
 * eight halves started moving at once.
 *
 * Two things a vector loop can get wrong and a scalar one cannot. The tail: a row whose
 * length is not a multiple of the step has a remainder, and a kernel that drops it answers
 * with a number rather than an error — so every k from 1 to 40 runs here, which covers every
 * remainder of eight several times over, and 2048 plus a remainder covers the case where the
 * vector body is long enough to hide a wrong tail in the noise. And the sum: partial
 * accumulators reassociate the addition, so the last bit moves. That is allowed and it is
 * measured rather than assumed — the reference here is a double accumulation, which both
 * float orders approximate, and the check is on the distance to it.
 *
 * The reference is written out in this file instead of borrowed from the library, because a
 * test that calls the code it is checking agrees with it however wrong it is.
 */
#include "notorch.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

/* IEEE half -> float, written here rather than taken from notorch.c for the reason above. */
static float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
               bits = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1F) bits = (s << 31) | (0xFFu << 23) | (m << 13);
    else bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* float -> IEEE half, round-to-nearest-even, for building the weights. The values below are
 * chosen to be exactly representable, so this rounds nothing and a bug in it cannot pass for
 * a bug in the kernel — but it is written correctly anyway, since the next person will widen
 * the value range before they read this comment. */
static uint16_t float_to_half(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s;                        /* underflow to zero: not exercised */
    if (e >= 0x1F) return (uint16_t)(s | 0x7C00);
    uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
    uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}

/* A value that a half holds exactly: a small integer scaled by a power of two. Weights are
 * built from these because the point of the test is the kernel, not float_to_half's rounding. */
static float nice_value(long i) { return (float)((i % 61) - 30) * 0.125f; }

/* The activation is the opposite: a full mantissa, and an exponent range wide enough that the
 * products do not all land on the same scale. Without this the test passes trivially — the
 * first version used tidy values on both sides, every partial sum was exact in f32, and the
 * worst error came out as a clean zero for every k. Zero error is not a strong result here,
 * it means the measurement cannot see reassociation at all, and reassociation is exactly what
 * changed. With a ragged activation the two summation orders genuinely differ and the
 * tolerance starts holding something. */
static float ragged_value(long i) {
    uint32_t s = (uint32_t)(i * 2654435761u + 12345u);
    s ^= s >> 13; s *= 1274126177u; s ^= s >> 16;
    float m = 1.0f + (float)(s & 0x7FFFFF) / (float)0x800000;   /* [1, 2) with every bit set */
    int e = (int)((s >> 24) % 9) - 4;                            /* spread over 2^-4 .. 2^4 */
    return ((s >> 23) & 1 ? -m : m) * ldexpf(1.0f, e);
}

static int run_f16(int rows, int k, double tol) {
    uint16_t *W = (uint16_t *)malloc((size_t)rows * k * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *got = (float *)malloc((size_t)rows * sizeof(float));
    if (!W || !x || !got) { free(W); free(x); free(got); printf("  FAIL out of memory\n"); fails++; return 0; }

    for (long r = 0; r < rows; r++)
        for (int j = 0; j < k; j++) W[r * k + j] = float_to_half(nice_value(r * 7 + j * 3));
    for (int j = 0; j < k; j++) x[j] = ragged_value(j * 5 + 1);

    char detail[224];
    int rc = nt_qmatvec(got, (const uint8_t *)W, GGUF_TYPE_F16, x, rows, k);
    if (rc != 0) {
        snprintf(detail, sizeof(detail), "k=%d returned %d", k, rc);
        check("f16 matvec takes the shape", 0, detail);
        free(W); free(x); free(got); return 0;
    }

    /* Normalised by the sum of the magnitudes rather than by the answer. A dot product of
     * signed terms cancels — 2048 products of size one can sum to 0.01 — and dividing the
     * error by that near-zero reports a catastrophe where the arithmetic did nothing wrong.
     * The first version of this test did exactly that and went red on the correct kernel.
     * Against the magnitudes the number means what it is meant to: how much of the
     * summation's own precision was lost. */
    double worst = 0.0; int worst_row = -1;
    for (long r = 0; r < rows; r++) {
        double want = 0.0, mag = 0.0;
        for (int j = 0; j < k; j++) {
            double p = (double)half_to_float(W[r * k + j]) * (double)x[j];
            want += p; mag += fabs(p);
        }
        double scale = mag > 1e-30 ? mag : 1.0;
        double err = fabs((double)got[r] - want) / scale;
        if (err > worst) { worst = err; worst_row = (int)r; }
    }
    snprintf(detail, sizeof(detail), "k=%d worst relative error %.3g at row %d, limit %.3g",
             k, worst, worst_row, tol);
    check("f16 matvec matches a double accumulation", worst <= tol, detail);

    free(W); free(x); free(got);
    return worst <= tol;
}

static int run_f32(int rows, int k, double tol) {
    float *W = (float *)malloc((size_t)rows * k * sizeof(float));
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *got = (float *)malloc((size_t)rows * sizeof(float));
    if (!W || !x || !got) { free(W); free(x); free(got); printf("  FAIL out of memory\n"); fails++; return 0; }

    for (long r = 0; r < rows; r++)
        for (int j = 0; j < k; j++) W[r * k + j] = ragged_value(r * 11 + j * 4);
    for (int j = 0; j < k; j++) x[j] = ragged_value(j * 5 + 1);

    char detail[224];
    int rc = nt_qmatvec(got, (const uint8_t *)W, GGUF_TYPE_F32, x, rows, k);
    if (rc != 0) {
        snprintf(detail, sizeof(detail), "k=%d returned %d", k, rc);
        check("f32 matvec takes the shape", 0, detail);
        free(W); free(x); free(got); return 0;
    }

    double worst = 0.0;
    for (long r = 0; r < rows; r++) {
        double want = 0.0, mag = 0.0;
        for (int j = 0; j < k; j++) {
            double p = (double)W[r * k + j] * (double)x[j];
            want += p; mag += fabs(p);
        }
        double scale = mag > 1e-30 ? mag : 1.0;
        double err = fabs((double)got[r] - want) / scale;
        if (err > worst) worst = err;
    }
    snprintf(detail, sizeof(detail), "k=%d worst relative error %.3g, limit %.3g", k, worst, tol);
    check("f32 matvec matches a double accumulation", worst <= tol, detail);

    free(W); free(x); free(got);
    return worst <= tol;
}

int main(void) {
    printf("unpacked matvec against a double accumulation\n");

    /* Every remainder of the vector step, several times over, at a row count that crosses the
     * threading floor in both directions so a threaded split is covered too. */
    int worst_k = -1;
    double tol = 1e-5;
    for (int k = 1; k <= 40; k++) {
        if (!run_f16(9, k, tol) && worst_k < 0) worst_k = k;
    }
    if (worst_k >= 0) printf("  first failing k was %d\n", worst_k);

    /* A body long enough that a dropped tail is a small fraction of the answer — which is
     * how a wrong tail survives a test that only looks at short rows. */
    run_f16(64, 2048, 1e-5);
    run_f16(64, 2048 + 1, 1e-5);
    run_f16(64, 2048 + 7, 1e-5);
    /* Above the threading floor: 64K elements is where nt_qmatvec starts splitting rows. */
    run_f16(512, 2048, 1e-5);

    /* Same limit for both, and it is set by the slowest machine rather than this one. Where
     * there is no vector path the sum runs serially over every element, and a serial sum of
     * 2048 terms drifts about the square root of that many epsilons — a few parts in a
     * million. One part in a hundred thousand clears that everywhere and still sits three
     * orders above what the vector path measures here, which is what a portable limit costs.
     * The structural failures this is built to catch are not close to it: a dropped tail on
     * the shapes below moves the answer by percent, not by parts per million. */
    run_f32(9, 33, 1e-5);
    run_f32(64, 2048 + 5, 1e-5);
    run_f32(512, 2048, 1e-5);

    /* What must be refused rather than answered. */
    float out[4], x[8] = {0};
    uint16_t w[32] = {0};
    check("an unknown dtype is refused",
          nt_qmatvec(out, (const uint8_t *)w, 99, x, 4, 8) != 0, NULL);

    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
