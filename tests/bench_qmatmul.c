/* bench_qmatmul.c — how much of the machine the batched packed matmul is using.
 *
 * Prefill on the polygon reads 11.6 t/s where llama.cpp reads 49.63, and three hypotheses
 * about why measured to zero: hoisting the nibble unpack out of the column loop, widening
 * the tile from 32 to 128, and dropping the bit-exact float tail for a cheaper one. None of
 * them moved the number. That leaves the question this answers — is the kernel near what the
 * instruction set can do, or a long way from it — which end-to-end timings cannot tell you
 * because they carry attention, the router, and the memory system with them.
 *
 * The ceiling is arithmetic, not a guess. One _mm256_maddubs_epi16 consumes 32 int8 pairs
 * and one _mm256_madd_epi16 folds them, so a sub-block of 32 weights costs two vector
 * instructions. On a core that retires one of each per cycle that is 32 MACs per cycle; the
 * figure printed below divides by cores * clock * 32. NEON's SDOT does 16 MACs in one
 * instruction, so the same division is by cores * clock * 16 there.
 *
 *   cc -O2 -std=gnu11 -I. -mavx2 -mfma tests/bench_qmatmul.c notorch.c -lm -o bench_qmatmul
 *   ./bench_qmatmul [cores] [GHz]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_s(void) { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec / 1e6; }

/* Q4_K: 144 bytes per 256 weights — an f16 scale, an f16 min, twelve packed 6-bit
 * (scale, min) pairs, and 128 nibble bytes. Only the two f16 fields have to be sane for the
 * arithmetic to be representative; the rest is any bit pattern. */
static uint8_t *make_q4_k(int m, int k, unsigned seed) {
    long nb = (long)k / 256;
    uint8_t *W = (uint8_t *)malloc((size_t)m * nb * 144);
    if (!W) return NULL;
    srand(seed);
    for (long r = 0; r < m; r++)
        for (long b = 0; b < nb; b++) {
            uint8_t *bl = W + (r * nb + b) * 144;
            for (int i = 0; i < 144; i++) bl[i] = (uint8_t)(rand() & 0xFF);
            bl[0] = 0x66; bl[1] = 0x2A;          /* d    ~ 0.05 */
            bl[2] = 0x00; bl[3] = 0x28;          /* dmin ~ 0.03 */
        }
    return W;
}


#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* The reference's activation layout, and the kernel shape it makes possible.
 *
 * Seven attempts at our own kernel measured to nothing or worse, and reading
 * ggml_vec_dot_q4_K_q8_K in llama.cpp's ggml/src/ggml-cpu/arch/x86/quants.c said why. It is
 * not the instruction selection. It is that their quantized activation carries ONE float
 * scale per 256 values (block_q8_K) where ours carries one per 32, plus per-16 block sums.
 *
 * With one scale per superblock the whole sub-block scaling stays in the integer domain:
 * `_mm256_madd_epi16(scale_l, p16l)` multiplies the int16 partial products by the sub-block
 * scale in the same instruction that folds them, everything accumulates into one int32
 * vector, and a block costs one cvtepi32_ps and one fmadd at the end. Eight scalar FMAs and
 * a six-deep hadd tree per (row, column, block) disappear.
 *
 * Ours cannot do that because eight different activation scales per superblock have to
 * multiply eight different sub-block sums, which forces them out to scalars first.
 *
 * The min term rides along: mins dot the per-32 activation sums in one _mm_madd_epi16 and
 * one fmadd, instead of eight scalar multiply-subtracts.
 *
 * This measures what changing the layout would buy. It is not bit-compatible with anything
 * in the tree — a coarser activation scale is a different number, not a different order. */
static void bench_scale_min(int j, const uint8_t *sc, uint8_t *s, uint8_t *mn) {
    if (j < 4) { *s = sc[j] & 63; *mn = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *mn = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4); }
}

static void q8k_quant(const float *X, int k, int n, int8_t *qa, float *dsuper, int32_t *bs32) {
    int nsb = k / 256, nsub = k / 32;
    for (int j = 0; j < n; j++) {
        const float *x = X + (long)j * k;
        for (int b = 0; b < nsb; b++) {
            const float *xb = x + (long)b * 256;
            float amax = 0.0f;
            for (int i = 0; i < 256; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
            float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            dsuper[(long)j * nsb + b] = d;
            for (int i = 0; i < 256; i++) {
                int q = (int)lrintf(xb[i] * id);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                qa[(long)j * k + (long)b * 256 + i] = (int8_t)q;
            }
        }
        for (int s = 0; s < nsub; s++) {
            const int8_t *p = qa + (long)j * k + (long)s * 32;
            int32_t t = 0;
            for (int i = 0; i < 32; i++) t += p[i];
            bs32[(long)j * nsub + s] = t;
        }
    }
}

static void q4k_q8k(float *out, int m, const uint8_t *W, const int8_t *qa,
                    const float *dsuper, const int32_t *bs32, int k, int n) {
    int nb = k / 256, nsub = k / 32;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const int TILE = 8;
    for (int j0 = 0; j0 < n; j0 += TILE) {
        int jn = n - j0; if (jn > TILE) jn = TILE;
        for (int row = 0; row < m; row++) {
            __m256 accv[32];
            float accm[32];
            for (int j = 0; j < jn; j++) { accv[j] = _mm256_setzero_ps(); accm[j] = 0.0f; }
            const uint8_t *rb = W + (long)row * nb * 144;
            for (int blk = 0; blk < nb; blk++) {
                const uint8_t *b = rb + (long)blk * 144;
                float dw = _cvtsh_ss((uint16_t)(b[0] | (b[1] << 8)));
                float mw = _cvtsh_ss((uint16_t)(b[2] | (b[3] << 8)));
                const uint8_t *sc = b + 4, *qs = b + 16;
                uint8_t ls[8], lm[8];
                for (int s = 0; s < 8; s++) bench_scale_min(s, sc, &ls[s], &lm[s]);
                const __m256i q0 = _mm256_loadu_si256((const __m256i *)(qs));
                const __m256i q1 = _mm256_loadu_si256((const __m256i *)(qs + 32));
                const __m256i q2 = _mm256_loadu_si256((const __m256i *)(qs + 64));
                const __m256i q3 = _mm256_loadu_si256((const __m256i *)(qs + 96));
                const __m256i s0 = _mm256_set1_epi16((short)ls[0]), s1 = _mm256_set1_epi16((short)ls[1]);
                const __m256i s2 = _mm256_set1_epi16((short)ls[2]), s3 = _mm256_set1_epi16((short)ls[3]);
                const __m256i s4 = _mm256_set1_epi16((short)ls[4]), s5 = _mm256_set1_epi16((short)ls[5]);
                const __m256i s6 = _mm256_set1_epi16((short)ls[6]), s7 = _mm256_set1_epi16((short)ls[7]);
                for (int j = 0; j < jn; j++) {
                    const int8_t *ac = qa + (long)(j0 + j) * k + (long)blk * 256;
                    const int32_t *scol = bs32 + (long)(j0 + j) * nsub;
                    #define LO(qv, sv, off) _mm256_madd_epi16((sv), _mm256_maddubs_epi16(  \
                        _mm256_and_si256((qv), m4),                                        \
                        _mm256_loadu_si256((const __m256i *)(ac + (off) * 32))))
                    #define HI(qv, sv, off) _mm256_madd_epi16((sv), _mm256_maddubs_epi16(  \
                        _mm256_and_si256(_mm256_srli_epi16((qv), 4), m4),                  \
                        _mm256_loadu_si256((const __m256i *)(ac + (off) * 32))))
                    __m256i sumi = _mm256_add_epi32(
                        _mm256_add_epi32(LO(q0,s0,0), HI(q0,s1,1)),
                        _mm256_add_epi32(LO(q1,s2,2), HI(q1,s3,3)));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(
                        _mm256_add_epi32(LO(q2,s4,4), HI(q2,s5,5)),
                        _mm256_add_epi32(LO(q3,s6,6), HI(q3,s7,7))));
                    #undef LO
                    #undef HI
                    /* The drain stays a vector across the whole row: one horizontal sum per
                     * (row, column) instead of one per block, which at k=14336 is 56 hadd
                     * chains saved. This is the shape of ggml_vec_dot_q4_K_q8_K. */
                    float dj = dsuper[(long)(j0 + j) * nb + blk];
                    accv[j] = _mm256_fmadd_ps(_mm256_set1_ps(dw * dj),
                                              _mm256_cvtepi32_ps(sumi), accv[j]);
                    float mt = 0.0f;
                    for (int s = 0; s < 8; s++) mt += (float)lm[s] * (float)scol[blk * 8 + s];
                    accm[j] += mw * dj * mt;
                }
            }
            for (int j = 0; j < jn; j++) {
                __m128 h = _mm_add_ps(_mm256_castps256_ps128(accv[j]),
                                      _mm256_extractf128_ps(accv[j], 1));
                h = _mm_hadd_ps(h, h); h = _mm_hadd_ps(h, h);
                out[(long)(j0 + j) * m + row] = _mm_cvtss_f32(h) - accm[j];
            }
        }
    }
}
#endif

int main(int argc, char **argv) {
    int cores = argc > 1 ? atoi(argv[1]) : 1;
    double ghz = argc > 2 ? atof(argv[2]) : 0.0;

    /* A feed-forward of the size the 4B bodies carry, against a prefill chunk. */
    const int dtype = 12;
    int n = argc > 3 ? atoi(argv[3]) : 32;
    int k = argc > 4 ? atoi(argv[4]) : 2048;
    int m = argc > 5 ? atoi(argv[5]) : 4096;
    uint8_t *W = make_q4_k(m, k, 7u);
    float *X = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *O = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (!W || !X || !O) return 1;
    for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);

    if (nt_qmatmul_i8(O, W, dtype, X, m, k, 2) != 0) {
        printf("bench_qmatmul: the batched entry refused Q4_K at k=%d\n", k);
        return 1;
    }

    /* Enough repetitions that building the weights — one rand() per byte, single
     * threaded — is noise beside the kernel. Without this the process spends most of its
     * life in setup and `time -v` reports a CPU percentage that belongs to the setup, which
     * is how a per-core comparison against another implementation goes wrong. */
    double probe0 = now_s();
    nt_qmatmul_i8(O, W, dtype, X, m, k, n);
    double one = now_s() - probe0;
    int reps = (int)(2.0 / (one > 1e-6 ? one : 1e-6));
    if (reps < 20) reps = 20;
    if (reps > 4000) reps = 4000;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) nt_qmatmul_i8(O, W, dtype, X, m, k, n);
    double dt = (now_s() - t0) / reps;

    double macs  = (double)m * k * n;
    double bytes = (double)m * ((double)k / 256.0) * 144.0;   /* the weights, read once */
    printf("q4_k batched  m=%d k=%d n=%d   %.2f ms   %.1f GMAC/s   %.1f GB/s of weights\n",
           m, k, n, dt * 1e3, macs / dt / 1e9, bytes / dt / 1e9);

    if (ghz > 0.0) {
#if defined(__AVX2__)
        double per_cycle = 32.0;        /* maddubs + madd, one sub-block */
        const char *isa = "AVX2";
#elif defined(__ARM_FEATURE_DOTPROD)
        double per_cycle = 16.0;        /* SDOT */
        const char *isa = "NEON SDOT";
#else
        double per_cycle = 1.0;
        const char *isa = "scalar";
#endif
        double ceil_g = cores * ghz * per_cycle;
        printf("              %s ceiling %.1f GMAC/s on %d cores at %.2f GHz — %.1f%% of it\n",
               isa, ceil_g, cores, ghz, 100.0 * (macs / dt / 1e9) / ceil_g);
    }

#if defined(__AVX2__) && defined(__FMA__)
    {
        int nsb = k / 256, nsub = k / 32;
        int8_t  *q2 = (int8_t *)malloc((size_t)k * n);
        float   *ds = (float *)malloc((size_t)nsb * n * sizeof(float));
        int32_t *bs = (int32_t *)malloc((size_t)nsub * n * sizeof(int32_t));
        float   *O2 = (float *)malloc(sizeof(float) * (size_t)m * n);
        if (q2 && ds && bs && O2) {
            q8k_quant(X, k, n, q2, ds, bs);
            q4k_q8k(O2, m, W, q2, ds, bs, k, n);
            double t2 = now_s();
            for (int r = 0; r < reps; r++) q4k_q8k(O2, m, W, q2, ds, bs, k, n);
            double d2 = (now_s() - t2) / reps;
            /* Per-element relative error is meaningless where an output lands near zero,
             * and with random weights some do. Against the RMS of the reference it says
             * what a different activation granularity actually costs. */
            double se = 0, sr = 0, worst = 0;
            for (long i = 0; i < (long)m * n; i++) {
                double a = O[i], b2 = O2[i], e = a - b2;
                se += e * e; sr += a * a;
                if (fabs(e) > worst) worst = fabs(e);
            }
            double rms_ref = sqrt(sr / ((double)m * n));
            printf("q4_k q8_K-style   %.2f ms   %.1f GMAC/s   %.2fx   rms err %.3g of rms %.3g"
                   " (%.2e), worst abs %.3g\n",
                   d2 * 1e3, macs / d2 / 1e9, dt / d2,
                   sqrt(se / ((double)m * n)), rms_ref,
                   sqrt(se / ((double)m * n)) / (rms_ref > 0 ? rms_ref : 1), worst);
        }
        free(q2); free(ds); free(bs); free(O2);
    }
#endif

    free(W); free(X); free(O);
    return 0;
}
