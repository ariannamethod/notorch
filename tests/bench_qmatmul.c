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


int main(int argc, char **argv) {
    int cores = argc > 1 ? atoi(argv[1]) : 1;
    double ghz = argc > 2 ? atof(argv[2]) : 0.0;

    /* A feed-forward of the size the 4B bodies carry, against a prefill chunk. */
    const int m = 4096, dtype = 12;
    int n = argc > 3 ? atoi(argv[3]) : 32;
    int k = argc > 4 ? atoi(argv[4]) : 2048;
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
    free(W); free(X); free(O);
    return 0;
}
