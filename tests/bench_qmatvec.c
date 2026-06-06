/*
 * bench_qmatvec.c — micro-benchmark for the packed Q4_0 matvec paths, single-thread:
 *   f32-dequant (nt_qmatvec) vs int8 dynamic-activation-quant (nt_qmatvec_i8, SDOT/scalar).
 * Sized below the threading gate (m*k < 4M) so both run single-thread — an apples-to-apples
 * kernel comparison. The end-to-end win is measured separately in the WTForacle engine.
 *
 * Build (neo / Darwin):
 *   cc -std=c11 -O2 -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *      -Wno-deprecated-declarations tests/bench_qmatvec.c notorch.c \
 *      -framework Accelerate -lm -o tests/bench_qmatvec
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_s(void) { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec / 1e6; }

int main(void) {
    srand(42);
    int m = 1024, k = 2048;            /* m*k = 2.1M < 4M -> both single-thread */
    long nb = (long)k / 32;
    uint8_t *W = malloc((long)m * nb * 18);
    for (long i = 0; i < (long)m * nb * 18; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (long r = 0; r < m; r++)
        for (long b = 0; b < nb; b++) { uint8_t *bl = W + (r*nb + b)*18; bl[0] = 0x66; bl[1] = 0x2A; }
    float *x = malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) x[i] = (float)((double)rand()/RAND_MAX*2.0 - 1.0);
    float *o = malloc(sizeof(float) * m);

    int N = 300;
    for (int w = 0; w < 10; w++) { nt_qmatvec(o, W, 2, x, m, k); nt_qmatvec_i8(o, W, 2, x, m, k); }
    double t0 = now_s(); for (int it = 0; it < N; it++) nt_qmatvec(o, W, 2, x, m, k);    double tf = now_s() - t0;
    double t1 = now_s(); for (int it = 0; it < N; it++) nt_qmatvec_i8(o, W, 2, x, m, k); double ti = now_s() - t1;

    printf("Q4_0 matvec, m=%d k=%d, single-thread, N=%d:\n", m, k, N);
    printf("  f32-dequant   : %.3f ms/call\n", tf / N * 1000.0);
    printf("  int8-dot      : %.3f ms/call   (%.2fx vs f32-dequant)\n", ti / N * 1000.0, tf / ti);
    free(W); free(x); free(o);
    return 0;
}
