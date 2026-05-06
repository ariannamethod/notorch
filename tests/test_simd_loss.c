// test_simd_loss.c — does my SIMD sgemm + softmax + cross-entropy give right loss?
// Mimics nanollama's lm_head forward at random init: logits = X @ W^T, softmax, CE.
// Build (SIMD): cc -O2 -mavx2 -mfma -DUSE_SIMD -I. -o test_simd_loss test_simd_loss.c -lm -lpthread
// Build (BLAS): cc -O2 -DUSE_BLAS -I. -o test_blas_loss test_simd_loss.c -lm -lopenblas
// Expected at random init: cross_entropy ~ log(V) = log(32000) ≈ 10.37

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef USE_SIMD
#include "notorch_simd.h"
#endif
#ifdef USE_BLAS
#include <cblas.h>
#endif

static void fill_xavier(float* p, long n, int fan_in, unsigned int* seed) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (long i = 0; i < n; i++) {
        // Box-Muller for normal noise (rough, deterministic via LCG)
        *seed = (*seed) * 1103515245u + 12345u;
        float u1 = ((*seed >> 8) & 0xFFFFFF) / 16777216.0f + 1e-7f;
        *seed = (*seed) * 1103515245u + 12345u;
        float u2 = ((*seed >> 8) & 0xFFFFFF) / 16777216.0f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        p[i] = scale * z;
    }
}

int main(void) {
    int T = 512;     // ctx
    int E = 576;     // dim
    int V = 32000;   // vocab
    long c_elems = (long)T * V;
    long x_elems = (long)T * E;
    long w_elems = (long)V * E;

    float* X = (float*)aligned_alloc(64, x_elems * sizeof(float));
    float* W = (float*)aligned_alloc(64, w_elems * sizeof(float));   // [V, E] row-major
    float* C = (float*)aligned_alloc(64, c_elems * sizeof(float));   // logits [T, V]

    unsigned int seed1 = 0xC0FFEE;
    unsigned int seed2 = 0xBADC0DE;
    fill_xavier(X, x_elems, E, &seed1);    // hidden state
    fill_xavier(W, w_elems, E, &seed2);    // lm_head weights

    // Logits: C[T, V] = X[T, E] @ W^T[V, E]^T  → use NoTrans×Trans
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, V, E,
                1.0f, X, E, W, E,
                0.0f, C, V);

    // Compute mean cross-entropy with random targets
    double total_ce = 0.0;
    int has_nan = 0, has_inf = 0;
    float min_logit = +1e30f, max_logit = -1e30f;
    for (long i = 0; i < c_elems; i++) {
        if (isnan(C[i])) has_nan++;
        if (isinf(C[i])) has_inf++;
        if (C[i] < min_logit) min_logit = C[i];
        if (C[i] > max_logit) max_logit = C[i];
    }

    for (int t = 0; t < T; t++) {
        // Pick a "target" deterministically
        int target = (t * 17 + 5) % V;
        float* logits = C + t * V;
        // softmax + cross-entropy with numerical stability
        float maxl = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > maxl) maxl = logits[v];
        double sum_exp = 0.0;
        for (int v = 0; v < V; v++) sum_exp += exp((double)(logits[v] - maxl));
        double log_softmax_t = (double)(logits[target] - maxl) - log(sum_exp);
        total_ce += -log_softmax_t;
    }
    double mean_ce = total_ce / T;

    printf("Logits stats:\n");
    printf("  min logit: %.4f\n", min_logit);
    printf("  max logit: %.4f\n", max_logit);
    printf("  NaN count: %d\n", has_nan);
    printf("  Inf count: %d\n", has_inf);
    printf("  sample C[0][0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", C[i]);
    printf("\n");
    printf("  sample C[256][0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", C[256*V + i]);
    printf("\n");
    printf("Cross-entropy mean: %.6f  (expected ~%.4f for random V=%d)\n",
           mean_ce, log((double)V), V);

    free(X); free(W); free(C);
    return 0;
}
