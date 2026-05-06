// bench_simd.c — micro-benchmark of cblas_sgemm/sgemv at notorch shapes.
// Compiles separately for OpenBLAS path (-DUSE_BLAS -lopenblas) and the
// in-house AVX2 path (-DUSE_SIMD -mavx2 -mfma -lpthread). Same source.
//
// Build OpenBLAS variant: cc bench_simd.c notorch.c -DUSE_BLAS -O2 -mavx2 -mfma -o bench_blas -lm -lopenblas
// Build SIMD variant:     cc bench_simd.c notorch.c -DUSE_SIMD -O2 -mavx2 -mfma -o bench_simd -lm -lpthread
//
// Time-tests representative shapes from notorch hot path.

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// notorch.c is included separately via the build line — we just need its
// public API. The cblas symbols come from notorch_simd.h / cblas.h depending
// on flags; declare them here for the bench-side calls so this TU compiles
// regardless of which include path notorch.c picks.
#ifdef USE_BLAS
  #include <cblas.h>
#endif
#ifdef USE_SIMD
  #include "notorch_simd.h"
#endif

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Fill matrix with deterministic values that fit single-precision well.
static void fill_random(float* p, long n, unsigned int* seed) {
    for (long i = 0; i < n; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        p[i] = (float)((*seed >> 16) & 0xFFFF) / 65536.0f - 0.5f;
    }
}

typedef struct {
    int M, N, K;
    int trans_a;  // 0 = NoTrans, 1 = Trans
    int trans_b;
    const char* label;
} bench_case;

int main(int argc, char** argv) {
    int repeat = (argc > 1) ? atoi(argv[1]) : 5;

#ifdef USE_BLAS
    const char* backend_label = "OpenBLAS";
#elif defined(USE_SIMD)
    const char* backend_label = "in-house AVX2+FMA";
#else
    const char* backend_label = "scalar";
#endif

    printf("════════════════════════════════════════════════════════════════\n");
    printf("  notorch matmul micro-bench [%s]\n", backend_label);
    printf("  repeat=%d, GFLOP/s = 2*M*N*K / time / 1e9\n", repeat);
    printf("════════════════════════════════════════════════════════════════\n");

    // Shapes representative of notorch training hot loops.
    // Most are (T=512 or 1024) × (dim 384..1664) × (dim 384..1664).
    bench_case cases[] = {
        // small: char-level Yent block (E=224, FFN=896, T=128)
        {128, 224, 224, 0, 0, "Yent attn QKV  (T=128, E=224, NN)"},
        {128, 896, 224, 0, 0, "Yent FFN up    (T=128, h=896, NN)"},
        {128, 224, 896, 0, 0, "Yent FFN down  (T=128, NN)"},

        // medium: nanollama-notorch (T=512, dim=576, FFN=1536)
        {512, 576, 576, 0, 0, "Llama attn QKV (T=512, E=576, NN)"},
        {512, 1536, 576, 0, 0, "Llama FFN up   (T=512, h=1536, NN)"},
        {512, 576, 1536, 0, 0, "Llama FFN down (T=512, NN)"},

        // weight-grad shapes: dW[out, in] = dout^T @ X
        {576, 576, 512, 1, 0, "Llama dWqkv    (E×E, T=512, TN)"},
        {1536, 576, 512, 1, 0, "Llama dWffn   (h×E, T=512, TN)"},

        // input-grad shapes: dX = dout @ W^T   (NoTrans × Trans)
        {512, 576, 576, 0, 1, "Llama dX QKV   (T=512, E×E, NT)"},
        {512, 576, 1536, 0, 1, "Llama dX FFN   (T=512, NT)"},

        // big: Janus 285M (E=640, FFN=1664, T=1024)
        {1024, 640, 640, 0, 0, "Janus QKV      (T=1024, E=640, NN)"},
        {1024, 1664, 640, 0, 0, "Janus FFN up   (T=1024, h=1664, NN)"},
        {1024, 640, 1664, 0, 0, "Janus FFN down (T=1024, NN)"},
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    unsigned int seed = 0xCAFEBABE;
    for (int c = 0; c < n_cases; c++) {
        bench_case bc = cases[c];

        // Allocate and fill A, B, C
        long a_elems = (long)bc.M * bc.K;
        long b_elems = (long)bc.K * bc.N;
        long c_elems = (long)bc.M * bc.N;
        float* A = (float*)aligned_alloc(64, a_elems * sizeof(float));
        float* B = (float*)aligned_alloc(64, b_elems * sizeof(float));
        float* C = (float*)aligned_alloc(64, c_elems * sizeof(float));
        if (!A || !B || !C) { fprintf(stderr, "OOM\n"); return 1; }
        fill_random(A, a_elems, &seed);
        fill_random(B, b_elems, &seed);
        memset(C, 0, c_elems * sizeof(float));

        // Strides depend on transpose mode (CblasRowMajor convention)
        // NoTrans: ld = inner-loop dimension
        int lda = (bc.trans_a == 0) ? bc.K : bc.M;
        int ldb = (bc.trans_b == 0) ? bc.N : bc.K;
        int ldc = bc.N;

        // Warmup
        cblas_sgemm(CblasRowMajor,
                    bc.trans_a ? CblasTrans : CblasNoTrans,
                    bc.trans_b ? CblasTrans : CblasNoTrans,
                    bc.M, bc.N, bc.K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);

        double total = 0.0;
        for (int r = 0; r < repeat; r++) {
            double t0 = now_seconds();
            cblas_sgemm(CblasRowMajor,
                        bc.trans_a ? CblasTrans : CblasNoTrans,
                        bc.trans_b ? CblasTrans : CblasNoTrans,
                        bc.M, bc.N, bc.K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
            total += now_seconds() - t0;
        }
        double avg = total / repeat;
        double flops = 2.0 * (double)bc.M * (double)bc.N * (double)bc.K;
        double gflops = flops / avg / 1e9;

        printf("  %-42s  %.2f ms  %6.2f GFLOP/s\n",
               bc.label, avg * 1000.0, gflops);

        // Sanity: checksum to prevent compiler from elising the work
        double sum = 0;
        for (long i = 0; i < c_elems; i += 4096) sum += C[i];
        if (sum == 0.0 && c_elems > 0) printf("    (warn: zero checksum)\n");

        free(A); free(B); free(C);
    }

    printf("════════════════════════════════════════════════════════════════\n");
    return 0;
}
