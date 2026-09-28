/* test_q8_0_rows.c — the Q8_0 kernel behind NT_NO_I8, which stopped being one scalar chain.
 *
 * nt_qmatvec runs a Q8_0 weight against float activations through nt_q8_0_rows, and every
 * consumer that sets NT_NO_I8 (quantized weights, no extra int8 quantization of the
 * activation) decodes through it. On AVX2+FMA it now carries 32 partial sums per row instead
 * of one. That moves the last bits, so three things are checked here rather than assumed.
 *
 * The distance: against a double accumulation, normalised by the sum of the magnitudes, for
 * the reason tests/test_f16_matvec.c gives — a signed dot product cancels, and dividing by a
 * cancelled answer reports damage the arithmetic did not do. The activation is ragged, with
 * outliers two to three orders above the rest, because the models that need NT_NO_I8 are the
 * ones whose residual stream looks like that; tidy values make every partial exact and hide
 * the reassociation this test exists to measure.
 *
 * The bits: the order is written out below — each term the scalar loop's own d*w, added to
 * its position's partial by fmaf, folded (p0..7 + p8..15) + (p16..23 + p24..31) and then
 * halves, 0+2 and 1+3, last pair — and the kernel must reproduce it exactly on AVX2+FMA. On
 * x86 without FMA the kernel is the scalar loop and must reproduce that. Anywhere else the
 * compiler may fuse the scalar loop's multiply-add on its own, so only the distance is held.
 *
 * The threads: the same call with the fan-out forced on and forced off must give identical
 * bits, since a row is one thread's work.
 *
 * The exception: non-finite SIMD sums retry in scalar order, so finite cancellation
 * survives overflow within a partial or at the fold. Genuine infinity and NaN survive too.
 *
 * The reference is written out in this file instead of borrowed from the library, because a
 * test that calls the code it is checking agrees with it however wrong it is. */
#include "notorch.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

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

static uint32_t mix(uint32_t s) { s ^= s >> 13; s *= 1274126177u; s ^= s >> 16; return s; }

/* Block scales: a half with a full ten-bit mantissa and exponents 2^-12 .. 2^1, the range
 * real Q8_0 scales occupy. Written as raw half bits, so no rounding happens building them. */
static uint16_t scale_bits(long i) {
    uint32_t s = mix((uint32_t)(i * 2246822519u + 7u));
    uint32_t e = 3u + s % 14u;                       /* biased 3..16 -> 2^-12 .. 2^1 */
    return (uint16_t)((e << 10) | ((s >> 8) & 0x3FF));
}

static int8_t weight(long i) {
    uint32_t s = mix((uint32_t)(i * 2654435761u + 99u));
    return (int8_t)((int)(s % 255u) - 127);           /* -127 .. 127, the Q8_0 range */
}

static float ragged_value(long i) {
    uint32_t s = mix((uint32_t)(i * 2654435761u + 12345u));
    float m = 1.0f + (float)(s & 0x7FFFFF) / (float)0x800000;
    int e = (int)((s >> 24) % 9) - 4;
    if (i % 97 == 13) e += 9;                          /* an outlier channel, ~2^9 above */
    return ((s >> 23) & 1 ? -m : m) * ldexpf(1.0f, e);
}

static uint8_t *build(int rows, int k) {
    int nb = k / 32;
    uint8_t *W = (uint8_t *)malloc((size_t)rows * nb * 34);
    if (!W) return NULL;
    for (long r = 0; r < rows; r++)
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = W + (r * nb + b) * 34;
            uint16_t d = scale_bits(r * nb + b);
            blk[0] = (uint8_t)(d & 0xFF); blk[1] = (uint8_t)(d >> 8);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)weight((r * nb + b) * 32 + i);
        }
    return W;
}

/* The claimed order on AVX2+FMA, one row. */
static float lanes_row(const uint8_t *rb, const float *x, int nb) {
    float p[32] = {0};
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rb + (long)b * 34;
        float d = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        for (int i = 0; i < 32; i++) {
            float t = d * (float)(int8_t)blk[2 + i];
            p[i] = fmaf(t, x[b * 32 + i], p[i]);
        }
    }
    float s[8], h[4];
    for (int l = 0; l < 8; l++) s[l] = (p[l] + p[8 + l]) + (p[16 + l] + p[24 + l]);
    for (int l = 0; l < 4; l++) h[l] = s[l] + s[l + 4];
    return (h[0] + h[2]) + (h[1] + h[3]);
}

/* The scalar loop's order, one row. */
static float scalar_row(const uint8_t *rb, const float *x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = rb + (long)b * 34;
        float d = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        for (int i = 0; i < 32; i++) acc += d * (float)(int8_t)blk[2 + i] * x[b * 32 + i];
    }
    return acc;
}

static void run(int rows, int k, double tol) {
    int nb = k / 32;
    uint8_t *W = build(rows, k);
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *got = (float *)malloc((size_t)rows * sizeof(float));
    float *got1 = (float *)malloc((size_t)rows * sizeof(float));
    char detail[256];
    if (!W || !x || !got || !got1) { check("allocation", 0, NULL); goto out; }
    for (int j = 0; j < k; j++) x[j] = ragged_value(j * 5 + 1);

    nt_qmv_set_thread_min(1);                          /* fan out whenever there are rows */
    if (nt_qmatvec(got, W, GGUF_TYPE_Q8_0, x, rows, k) != 0) {
        snprintf(detail, sizeof(detail), "k=%d", k); check("q8_0 matvec takes the shape", 0, detail); goto out;
    }
    nt_qmv_set_thread_min(LONG_MAX);                   /* never fan out */
    nt_qmatvec(got1, W, GGUF_TYPE_Q8_0, x, rows, k);
    snprintf(detail, sizeof(detail), "rows=%d k=%d", rows, k);
    check("threaded and single-thread results are bit-identical",
          memcmp(got, got1, (size_t)rows * sizeof(float)) == 0, detail);

    double worst = 0.0, worst_scalar = 0.0; int worst_row = -1, exact = 0, exact_checked = 0;
    for (long r = 0; r < rows; r++) {
        const uint8_t *rb = W + r * nb * 34;
        double want = 0.0, mag = 0.0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 34;
            double d = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
            for (int i = 0; i < 32; i++) {
                double p = d * (double)(int8_t)blk[2 + i] * (double)x[b * 32 + i];
                want += p; mag += fabs(p);
            }
        }
        double scale = mag > 1e-30 ? mag : 1.0;
        double err = fabs((double)got[r] - want) / scale;
        double err_s = fabs((double)scalar_row(rb, x, nb) - want) / scale;
        if (err > worst) { worst = err; worst_row = (int)r; }
        if (err_s > worst_scalar) worst_scalar = err_s;
#if defined(__AVX2__) && defined(__FMA__)
        exact_checked++; exact += got[r] == lanes_row(rb, x, nb);
#elif (defined(__x86_64__) || defined(__i386__)) && !defined(__FMA__)
        exact_checked++; exact += got[r] == scalar_row(rb, x, nb);
#endif
    }
    snprintf(detail, sizeof(detail), "k=%d worst %.3g at row %d (scalar order %.3g), limit %.3g",
             k, worst, worst_row, worst_scalar, tol);
    check("q8_0 matvec matches a double accumulation", worst <= tol, detail);
    if (exact_checked) {
        snprintf(detail, sizeof(detail), "k=%d %d/%d rows", k, exact, exact_checked);
        check("q8_0 matvec reproduces its documented summation order", exact == exact_checked, detail);
    }
out:
    free(W); free(x); free(got); free(got1);
}

/* Finite products may cancel in scalar order while overflowing a SIMD partial,
 * or while folding finite partials. Both must take the scalar retry. */
static void cancellation(void) {
    uint8_t W[7 * 2 * 34] = {0};
    float x[64], got[7], got1[7];
    /* Unit weights make every product exact, including on scalar FMA targets. */
    for (int i = 0; i < 64; i++) x[i] = 0.6f * FLT_MAX;
    for (int r = 0; r < 7; r++) {
        for (int b = 0; b < 2; b++) {
            uint8_t *p = W + (r * 2 + b) * 34;
            p[1] = 0x3c;                            /* scale = 1 */
            p[2] = 1;
            if (r % 2 == 0) {
                p[2 + 8] = (uint8_t)-1;             /* overflow within partials */
            } else if (b == 0) {
                p[2 + 1] = (uint8_t)-1;
                p[2 + 8] = 1;
                p[2 + 9] = (uint8_t)-1;             /* overflow only at the fold */
            } else {
                p[2] = 0;
            }
        }
    }
    nt_qmv_set_thread_min(1);
    nt_qmatvec(got, W, GGUF_TYPE_Q8_0, x, 7, 64);
    nt_qmv_set_thread_min(LONG_MAX);
    nt_qmatvec(got1, W, GGUF_TYPE_Q8_0, x, 7, 64);
    int finite_zero = 1;
    for (int r = 0; r < 7; r++) finite_zero &= got[r] == 0.0f;
    check("large finite products cancel without SIMD overflow", finite_zero, NULL);
    check("scalar retry is bit-identical across thread counts",
          memcmp(got, got1, sizeof(got)) == 0, NULL);

    /* A retry is not saturation: a genuinely overflowing dot or a NaN input
     * retains its non-finite scalar result. */
    W[2 + 8] = W[34 + 2 + 8] = 1;
    nt_qmatvec(got, W, GGUF_TYPE_Q8_0, x, 1, 64);
    check("genuine overflow remains positive infinity", isinf(got[0]) && got[0] > 0, NULL);
    x[0] = NAN;
    nt_qmatvec(got, W, GGUF_TYPE_Q8_0, x, 1, 64);
    check("NaN input remains NaN", isnan(got[0]), NULL);
}

int main(void) {
#if defined(__AVX2__) && defined(__FMA__)
    printf("test_q8_0_rows — AVX2+FMA kernel, 32 partial sums\n");
#elif (defined(__x86_64__) || defined(__i386__)) && !defined(__FMA__)
    printf("test_q8_0_rows — scalar kernel (x86 without FMA)\n");
#else
    printf("test_q8_0_rows — scalar kernel, compiler may fuse; distance only\n");
#endif
    /* Shapes: one block, a few, and the widths of the decoders that run NT_NO_I8
     * (1152 and 6912 are Gemma 3 1B's residual and feed-forward). 1e-5 of the magnitude sum
     * is roughly a hundred float epsilons: loose enough for any honest order over 216
     * blocks, tight enough that a dropped block or a lost partial cannot hide in it. */
    int ks[] = {32, 64, 96, 1152, 2048, 6912};
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) run(i < 3 ? 7 : 64, ks[i], 1e-5);
    cancellation();
    printf("%s (%d failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
