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
#include <math.h>

static void bench_scale_min(int j, const uint8_t *sc, uint8_t *s, uint8_t *mn) {
    if (j < 4) { *s = sc[j] & 63; *mn = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *mn = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4); }
}

/* The variant the ceiling measurement points at.
 *
 * The shipped kernel drains eight sub-block dots to scalars through a hadd tree — six
 * dependent horizontal adds, two permutes and a store — and then adds them one at a time.
 * That drain is per (row, column, block) and it is latency, not throughput: nothing else
 * issues while it resolves.
 *
 * It is avoidable because the scale is a scalar and the sum is distributive: sum(lanes) * c
 * is sum(lanes * c). So each sub-block's vector of partial sums is converted to float,
 * multiplied by its own coefficient, and accumulated into a float vector that is drained
 * once per row. Two accumulators, because one is a serial dependency eight deep per block.
 *
 * The min term stays scalar — it multiplies SUM(qa), which is already a scalar — and gets
 * its own accumulator so it does not lengthen the other chain.
 *
 * This changes the order the floats are added in, so it is not bit-identical to the
 * per-token kernel and would cost tests/test_qmatmul its memcmp. That is the trade this
 * measures. */
static void q4k_vecdrain(float *out, int m, const uint8_t *W, const int8_t *qa,
                         const float *da, const int32_t *asum, int k, int n, int tile) {
    int nb = k / 256, nsub = k / 32;
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    for (int j0 = 0; j0 < n; j0 += tile) {
        int jn = n - j0; if (jn > tile) jn = tile;
        for (int row = 0; row < m; row++) {
            for (int j = 0; j < jn; j++) {
                const int8_t *ac0 = qa + (long)(j0 + j) * k;
                const float *dac = da + (long)(j0 + j) * nsub;
                const int32_t *asc = asum + (long)(j0 + j) * nsub;
                __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_setzero_ps();
                float mins = 0.0f;
                for (int blk = 0; blk < nb; blk++) {
                    const uint8_t *b = W + (long)row * nb * 144 + (long)blk * 144;
                    float d    = (float)(*(const _Float16 *)(const void *)b);
                    float dmin = (float)(*(const _Float16 *)(const void *)(b + 2));
                    const uint8_t *sc = b + 4, *qs = b + 16;
                    const int8_t *ac = ac0 + (long)blk * 256;
                    for (int p = 0; p < 4; p++) {
                        __m256i qsv = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                        __m256i lo  = _mm256_and_si256(qsv, m4);
                        __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(qsv, 4), m4);
                        __m256i a0  = _mm256_loadu_si256((const __m256i *)(ac + (2*p)     * 32));
                        __m256i a1  = _mm256_loadu_si256((const __m256i *)(ac + (2*p + 1) * 32));
                        __m256i d0  = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a0), ones);
                        __m256i d1  = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a1), ones);
                        uint8_t ls0, lm0, ls1, lm1;
                        bench_scale_min(2*p,     sc, &ls0, &lm0);
                        bench_scale_min(2*p + 1, sc, &ls1, &lm1);
                        int s0 = blk * 8 + 2*p, s1 = s0 + 1;
                        f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0),
                                             _mm256_set1_ps(d * (float)ls0 * dac[s0]), f0);
                        f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1),
                                             _mm256_set1_ps(d * (float)ls1 * dac[s1]), f1);
                        mins += dac[s0] * dmin * (float)lm0 * (float)asc[s0]
                              + dac[s1] * dmin * (float)lm1 * (float)asc[s1];
                    }
                }
                __m256 f = _mm256_add_ps(f0, f1);
                __m128 lo4 = _mm_add_ps(_mm256_castps256_ps128(f), _mm256_extractf128_ps(f, 1));
                lo4 = _mm_hadd_ps(lo4, lo4);
                lo4 = _mm_hadd_ps(lo4, lo4);
                out[(long)(j0 + j) * m + row] = _mm_cvtss_f32(lo4) - mins;
            }
        }
    }
}
#endif

int main(int argc, char **argv) {
    int cores = argc > 1 ? atoi(argv[1]) : 1;
    double ghz = argc > 2 ? atof(argv[2]) : 0.0;

    /* A feed-forward of the size the 4B bodies carry, against a prefill chunk. */
    const int m = 4096, k = 2048, dtype = 12;
    int n = argc > 3 ? atoi(argv[3]) : 32;
    uint8_t *W = make_q4_k(m, k, 7u);
    float *X = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *O = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (!W || !X || !O) return 1;
    for (long i = 0; i < (long)k * n; i++) X[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);

    if (nt_qmatmul_i8(O, W, dtype, X, m, k, 2) != 0) {
        printf("bench_qmatmul: the batched entry refused Q4_K at k=%d\n", k);
        return 1;
    }

    int reps = 20;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) nt_qmatmul_i8(O, W, dtype, X, m, k, n);
    double dt = (now_s() - t0) / reps;
    double dt_1 = dt;   /* the shipped kernel, for the variant to be read against */

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
        int nsub = k / 32;
        int8_t *qa = (int8_t *)malloc((size_t)k * n);
        float *da = (float *)malloc((size_t)nsub * n * sizeof(float));
        int32_t *as = (int32_t *)malloc((size_t)nsub * n * sizeof(int32_t));
        float *O2 = (float *)malloc(sizeof(float) * (size_t)m * n);
        if (qa && da && as && O2 && nt_quant_act_batch(X, k, n, qa, da, as) == 0) {
            q4k_vecdrain(O2, m, W, qa, da, as, k, n, 32);
            double t1 = now_s();
            for (int r = 0; r < reps; r++) q4k_vecdrain(O2, m, W, qa, da, as, k, n, 32);
            double dv = (now_s() - t1) / reps;
            double worst = 0;
            for (long i = 0; i < (long)m * n; i++) {
                double a = O[i], b = O2[i];
                double rel = fabs(a) > 1e-6 ? fabs(a - b) / fabs(a) : fabs(a - b);
                if (rel > worst) worst = rel;
            }
            printf("q4_k vec-drain, 1 thread   %.2f ms   %.1f GMAC/s   %.2fx   worst rel %.3g\n",
                   dv * 1e3, macs / dv / 1e9, dt_1 / dv, worst);
        }
        free(qa); free(da); free(as); free(O2);
    }
#endif

    free(W); free(X); free(O);
    return 0;
}
