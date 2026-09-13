/*
 * test_conv1d.c — the 1-D convolution against the definition, and the f16-column
 * variant against the rounding it claims to do.
 *
 * nt_conv1d unfolds into columns and hands one GEMM to BLAS. That is an
 * optimisation of a triple loop, so the triple loop is what it is checked against:
 * an independent implementation of the same arithmetic, written from the definition
 * rather than from the kernel, at shapes that put taps outside the signal on both
 * ends and at strides that make the output shorter than the input.
 *
 * nt_conv1d_f16cols is NOT checked against nt_conv1d — they are deliberately
 * different arithmetic. It is checked against a triple loop that rounds each tap to
 * f16 where the kernel rounds, which is the only statement worth making about it,
 * and separately against nt_conv1d for INEQUALITY on a signal fine enough to feel
 * the rounding: a "f16" path that quietly did nothing would pass every tolerance
 * test in this file.
 *
 * Build: cc -O2 -I. tests/test_conv1d.c notorch.c -lm [-DUSE_BLAS -lopenblas]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;

static void ok(int cond, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf(cond ? "  PASS " : "  FAIL ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (cond) n_pass++; else n_fail++;
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static float f16r(float v) {
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
    return (float)(__fp16)v;
#else
    /* round-to-nearest-even, the same fallback nt_f32_to_f16_round uses */
    uint32_t bits; memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    uint16_t h;
    if (exp >= 0x1F) h = (uint16_t)(sign | 0x7C00);
    else if (exp <= 0) h = (uint16_t)sign;
    else {
        h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
        uint32_t rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    }
    float out; uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t o;
    if (e == 0) o = s;                                   /* denormals -> 0, good enough here */
    else if (e == 0x1F) o = s | 0x7F800000 | (m << 13);
    else o = s | ((e - 15 + 127) << 23) | (m << 13);
    memcpy(&out, &o, 4);
    return out;
#endif
}

/* The definition, written out: out[co][o] = sum_c sum_k w[co][c][k] * in[c][o*s-p+k].
 * `round_taps` rounds each tap the way ggml's f16 columns do. Accumulation is double
 * so that the reference is not itself a float-order experiment. */
static void conv1d_naive(float *out, const float *in, const float *w, const float *b,
                         int Cin, int Lin, int Cout, int K, int stride, int pad,
                         int round_taps) {
    int Lout = (Lin + 2 * pad - K) / stride + 1;
    for (int co = 0; co < Cout; co++)
        for (int o = 0; o < Lout; o++) {
            double acc = b ? b[co] : 0.0;
            for (int c = 0; c < Cin; c++)
                for (int k = 0; k < K; k++) {
                    int t = o * stride - pad + k;
                    float v = (t >= 0 && t < Lin) ? in[(size_t)c * Lin + t] : 0.0f;
                    if (round_taps) v = f16r(v);
                    acc += (double)w[((size_t)co * Cin + c) * K + k] * (double)v;
                }
            out[(size_t)co * Lout + o] = (float)acc;
        }
}

static double maxabs_diff(const float *a, const float *b, size_t n, size_t *where) {
    double worst = 0.0; *where = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) { worst = d; *where = i; }
    }
    return worst;
}

/* One shape through both the kernel and the reference. `f16` picks the variant.
 *
 * The limit is Cin*K * 1e-7 and not a flat constant. The reference accumulates in
 * double and the kernel in f32, so the gap between them IS the f32 accumulation of
 * Cin*K products, which grows with the sum length: at Cin=384, K=3 that is 1152
 * terms of order one, and 1152 * FLT_EPSILON is already 6.9e-5. A flat 1e-5 would
 * therefore pass the small shapes and fail the whisper stem for being arithmetic.
 * 1e-7 per term is a little under two f32 eps and leaves the measured worst case
 * (2.3e-5 against a 1.2e-4 budget at Cin=384) a factor of five of headroom, which is
 * tight enough that a transposed index or a dropped tap does not fit inside it. */
static void check(int Cin, int Lin, int Cout, int K, int stride, int pad,
                  int f16, int with_bias, unsigned seed) {
    const double limit = (double)Cin * K * 1e-7;
    int Lout = (Lin + 2 * pad - K) / stride + 1;
    if (Lout <= 0) { ok(0, "geometry Cin=%d Lin=%d K=%d s=%d p=%d is empty", Cin, Lin, K, stride, pad); return; }

    size_t nw = (size_t)Cout * Cin * K, ni = (size_t)Cin * Lin, no = (size_t)Cout * Lout;
    float *in = malloc(ni * sizeof(float)), *w = malloc(nw * sizeof(float));
    float *b  = malloc((size_t)Cout * sizeof(float));
    float *got = malloc(no * sizeof(float)), *ref = malloc(no * sizeof(float));
    if (!in || !w || !b || !got || !ref) { ok(0, "allocation"); goto done; }

    srand(seed);
    for (size_t i = 0; i < ni; i++) in[i] = frand();
    for (size_t i = 0; i < nw; i++) w[i]  = frand();
    for (int i = 0; i < Cout; i++)  b[i]  = frand();
    const float *bias = with_bias ? b : NULL;

    int rc = f16 ? nt_conv1d_f16cols(got, in, w, bias, Cin, Lin, Cout, K, stride, pad)
                 : nt_conv1d       (got, in, w, bias, Cin, Lin, Cout, K, stride, pad);
    if (rc != 0) { ok(0, "nt_conv1d%s returned %d", f16 ? "_f16cols" : "", rc); goto done; }

    conv1d_naive(ref, in, w, bias, Cin, Lin, Cout, K, stride, pad, f16);
    size_t at; double d = maxabs_diff(got, ref, no, &at);
    ok(d <= limit, "conv1d%s Cin=%d Lin=%d Cout=%d K=%d s=%d p=%d bias=%d "
                   "vs the triple loop — max|d| = %.3e at %zu, limit %.3e (%d terms)",
       f16 ? "_f16cols" : "", Cin, Lin, Cout, K, stride, pad, with_bias, d, at, limit, Cin * K);

done:
    free(in); free(w); free(b); free(got); free(ref);
}

/* The f16-column path has to actually round. Same input through both entries: if the
 * outputs are identical the rounding is not happening, and every tolerance above is
 * being passed by a kernel that does nothing. */
static void check_f16_actually_rounds(void) {
    const int Cin = 8, Lin = 64, Cout = 8, K = 3;
    size_t nw = (size_t)Cout * Cin * K, ni = (size_t)Cin * Lin;
    int Lout = Lin;
    size_t no = (size_t)Cout * Lout;
    float *in = malloc(ni * sizeof(float)), *w = malloc(nw * sizeof(float));
    float *a = malloc(no * sizeof(float)), *bb = malloc(no * sizeof(float));

    srand(99);
    /* Values with mantissa bits below f16's 10 — rounding must bite. */
    for (size_t i = 0; i < ni; i++) in[i] = frand() * 1.0001234f + 1e-4f;
    for (size_t i = 0; i < nw; i++) w[i]  = frand();

    nt_conv1d       (a,  in, w, NULL, Cin, Lin, Cout, K, 1, 1);
    nt_conv1d_f16cols(bb, in, w, NULL, Cin, Lin, Cout, K, 1, 1);

    size_t at; double d = maxabs_diff(a, bb, no, &at);
    ok(d > 0.0, "f16 columns change the result — max|d| vs f32 columns = %.3e at %zu "
                "(zero would mean the rounding never ran)", d, at);
    /* ...but only by about what an f16 mantissa is worth. */
    ok(d < 1e-2, "f16 columns stay within an f16 mantissa of the f32 path — %.3e < 1e-2", d);

    free(in); free(w); free(a); free(bb);
}

/* im2col_1d is public, so its tap order is part of the contract the GEMM relies on. */
static void check_im2col(void) {
    const int Cin = 2, Lin = 5, K = 3, stride = 1, pad = 1;
    const int Lout = (Lin + 2 * pad - K) / stride + 1;   /* 5 */
    float in[10]; for (int i = 0; i < 10; i++) in[i] = (float)(i + 1);
    float col[2 * 3 * 5];
    nt_im2col_1d(col, in, Cin, Lin, K, stride, pad);

    /* row (c*K + k), column o, value in[c][o - 1 + k] or 0 */
    int good = 1;
    for (int c = 0; c < Cin; c++)
        for (int k = 0; k < K; k++)
            for (int o = 0; o < Lout; o++) {
                int t = o - pad + k;
                float want = (t >= 0 && t < Lin) ? in[c * Lin + t] : 0.0f;
                if (col[(c * K + k) * Lout + o] != want) good = 0;
            }
    ok(good, "nt_im2col_1d lays taps out as row (c*K + k) across output positions");
    ok(col[0] == 0.0f && col[2 * 5 + Lout - 1] == 0.0f,
       "nt_im2col_1d zeroes the taps that fall outside the signal");
}

static void check_rejects(void) {
    float x[8] = {0}, w[8] = {0}, o[8] = {0};
    ok(nt_conv1d(o, x, w, NULL, 1, 2, 1, 5, 1, 0) == -1, "a kernel longer than the signal is refused");
    ok(nt_conv1d(o, x, w, NULL, 1, 8, 1, 3, 0, 0) == -1, "stride 0 is refused");
    ok(nt_conv1d(o, x, w, NULL, 1, 8, 1, 3, 1, -1) == -1, "negative padding is refused");
    ok(nt_conv1d(o, x, NULL, NULL, 1, 8, 1, 3, 1, 0) == -1, "a NULL weight is refused");
}

int main(void) {
    printf("nt_conv1d — im2col + one GEMM against the definition\n\n");

    check(1,  16,   1,  3, 1, 1, 0, 0, 1);
    check(4,  32,   8,  3, 1, 1, 0, 1, 2);
    check(4,  32,   8,  3, 2, 1, 0, 1, 3);
    check(3,  17,   5,  5, 1, 2, 0, 1, 4);
    check(3,  17,   5,  5, 3, 2, 0, 0, 5);
    check(8,  64,  16,  1, 1, 0, 0, 1, 6);       /* K=1: a pointwise conv */
    check(16, 100, 32,  7, 2, 3, 0, 1, 7);
    check(80, 300, 384, 3, 1, 1, 0, 1, 8);       /* the whisper stem, first conv */
    check(384, 300, 384, 3, 2, 1, 0, 1, 9);      /* the whisper stem, second conv */

    /* f16 columns against a triple loop that rounds in the same place. The residual is
     * the f32 accumulation, not the rounding, so the budget is the same per term. */
    check(4,  32,   8,  3, 1, 1, 1, 1, 12);
    check(80, 300, 384, 3, 1, 1, 1, 1, 13);
    check(384, 300, 384, 3, 2, 1, 1, 1, 14);

    check_f16_actually_rounds();
    check_im2col();
    check_rejects();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
