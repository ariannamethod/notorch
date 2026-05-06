// test_simd_correctness.c — compare SIMD sgemm vs scalar at large nanollama shapes.
// Build: cc -O2 -mavx2 -mfma -DUSE_SIMD -I. -o test_simd_correctness test_simd_correctness.c -lm -lpthread
//
// Tests cblas_sgemm correctness on shapes where unit tests don't reach:
//   * lm_head: m=T=128 (subset of 512), k=E=576, n=V=32000 (NoTrans×NoTrans)
//   * weight grad: m=V, k=T, n=E (Trans×NoTrans)
//   * input grad: m=T, k=V, n=E (NoTrans×Trans)
// Compares against bit-identical scalar reference. Reports max relative error.

#include "notorch_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static void scalar_sgemm_NN(const float* A, const float* B, float* C,
                             int m, int k, int n, int lda, int ldb, int ldc) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += (double)A[i*lda + p] * B[p*ldb + j];
            C[i*ldc + j] = (float)s;
        }
}

static void scalar_sgemm_TN(const float* A, const float* B, float* C,
                             int m, int k, int n, int lda, int ldb, int ldc) {
    // C[m,n] = A^T @ B, A is [k, m]
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += (double)A[p*lda + i] * B[p*ldb + j];
            C[i*ldc + j] = (float)s;
        }
}

static void scalar_sgemm_NT(const float* A, const float* B, float* C,
                             int m, int k, int n, int lda, int ldb, int ldc) {
    // C[m,n] = A @ B^T, B is [n, k]
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += (double)A[i*lda + p] * B[j*ldb + p];
            C[i*ldc + j] = (float)s;
        }
}

static double max_rel_diff(const float* a, const float* b, long n) {
    double maxd = 0;
    for (long i = 0; i < n; i++) {
        float ref = b[i];
        float got = a[i];
        double abs_diff = fabsf(got - ref);
        double denom = fabsf(ref) + 1e-6;
        double rel = abs_diff / denom;
        if (rel > maxd) maxd = rel;
    }
    return maxd;
}

static void worst_diff_diag(const float* simd, const float* ref, long n, int M, int N) {
    double worst_rel = 0;
    long worst_idx = 0;
    for (long i = 0; i < n; i++) {
        double abs_diff = fabsf(simd[i] - ref[i]);
        double rel = abs_diff / (fabsf(ref[i]) + 1e-6);
        if (rel > worst_rel) { worst_rel = rel; worst_idx = i; }
    }
    int row = worst_idx / N;
    int col = worst_idx % N;
    printf("    worst at row=%d (%.0f%% of M=%d), col=%d  ref=%.6f simd=%.6f abs_diff=%.6f\n",
           row, 100.0 * row / M, M, col, ref[worst_idx], simd[worst_idx],
           simd[worst_idx] - ref[worst_idx]);
    // also count how many are bad
    int bad_count = 0;
    for (long i = 0; i < n; i++) {
        double rel = fabsf(simd[i] - ref[i]) / (fabsf(ref[i]) + 1e-6);
        if (rel > 1e-3) bad_count++;
    }
    printf("    %d/%ld outputs have rel_diff > 1e-3 (%.2f%%)\n",
           bad_count, n, 100.0 * bad_count / n);
}

static void fill(float* p, long n, unsigned int* seed) {
    for (long i = 0; i < n; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        p[i] = (float)((*seed >> 16) & 0xFFFF) / 65536.0f - 0.5f;
    }
}

static int run_case(const char* name,
                    CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                    int M, int N, int K) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;

    long a_elems = (TransA == CblasNoTrans) ? (long)M * K : (long)K * M;
    long b_elems = (TransB == CblasNoTrans) ? (long)K * N : (long)N * K;
    long c_elems = (long)M * N;

    float* A = (float*)aligned_alloc(64, a_elems * sizeof(float));
    float* B = (float*)aligned_alloc(64, b_elems * sizeof(float));
    float* C_simd = (float*)aligned_alloc(64, c_elems * sizeof(float));
    float* C_ref = (float*)aligned_alloc(64, c_elems * sizeof(float));
    if (!A || !B || !C_simd || !C_ref) { printf("OOM\n"); return 1; }

    unsigned int seed = 0xDEADBEEF;
    fill(A, a_elems, &seed);
    fill(B, b_elems, &seed);

    // Scalar reference
    if      (TransA == CblasNoTrans && TransB == CblasNoTrans) scalar_sgemm_NN(A, B, C_ref, M, K, N, lda, ldb, ldc);
    else if (TransA == CblasTrans   && TransB == CblasNoTrans) scalar_sgemm_TN(A, B, C_ref, M, K, N, lda, ldb, ldc);
    else if (TransA == CblasNoTrans && TransB == CblasTrans  ) scalar_sgemm_NT(A, B, C_ref, M, K, N, lda, ldb, ldc);
    else { printf("unsupported transpose mode\n"); return 1; }

    // SIMD path
    cblas_sgemm(CblasRowMajor, TransA, TransB,
                M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C_simd, ldc);

    double rel = max_rel_diff(C_simd, C_ref, c_elems);

    // Sample a few values for diagnostic
    printf("  %-50s  M=%d K=%d N=%d  max_rel_diff=%.2e   ", name, M, K, N, rel);
    if (rel < 1e-3) printf("OK\n");
    else {
        printf("FAIL\n");
        worst_diff_diag(C_simd, C_ref, c_elems, M, N);
    }

    free(A); free(B); free(C_simd); free(C_ref);
    return (rel < 1e-3) ? 0 : 1;
}

int main(void) {
    printf("=== SIMD cblas_sgemm correctness (vs scalar reference, NT_SIMD_THREADS auto) ===\n");
    int fails = 0;

    // Small (test-scale, should pass — tests already do)
    fails += run_case("small NN T=8 K=64 V=64",      CblasNoTrans, CblasNoTrans, 8, 64, 64);
    fails += run_case("small TN T=8 K=64 V=64",      CblasTrans,   CblasNoTrans, 8, 64, 64);
    fails += run_case("small NT T=8 K=64 V=64",      CblasNoTrans, CblasTrans,   8, 64, 64);

    // Medium (notorch_test gradcheck scale)
    fails += run_case("med NN T=64 K=128 V=128",     CblasNoTrans, CblasNoTrans, 64, 128, 128);
    fails += run_case("med NN T=128 K=224 V=896",    CblasNoTrans, CblasNoTrans, 128, 224, 896);

    // Larger (Llama nanollama-notorch hot path)
    fails += run_case("nano hidden×Wproj T=512 E=576 H=576",   CblasNoTrans, CblasNoTrans, 512, 576, 576);
    fails += run_case("nano FFN up T=512 E=576 H=1536",        CblasNoTrans, CblasNoTrans, 512, 576, 1536);
    fails += run_case("nano FFN down T=512 H=1536 E=576",      CblasNoTrans, CblasNoTrans, 512, 1536, 576);

    // Smaller subset of lm_head (M=128 to reduce N×N memory)
    fails += run_case("lm_head subset M=128 K=576 V=32000",    CblasNoTrans, CblasNoTrans, 128, 576, 32000);

    // Full lm_head
    fails += run_case("lm_head full M=512 K=576 V=32000",      CblasNoTrans, CblasNoTrans, 512, 576, 32000);

    // dW shapes
    fails += run_case("dW Llama TN T=512 E=576 -> 576×576",    CblasTrans, CblasNoTrans, 576, 512, 576);
    fails += run_case("dX Llama NT T=512 E×E NT",              CblasNoTrans, CblasTrans, 512, 576, 576);

    printf("\n=== %d failure(s) ===\n", fails);
    return fails;
}
