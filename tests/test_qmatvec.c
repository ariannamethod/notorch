/*
 * test_qmatvec.c — nt_qmatvec packed quantized matvec vs the dequant->cblas oracle.
 *
 * For each quant format the packed kernel must compute the same matvec as the
 * established path (independent dequant -> nt_blas_matvec), within f32
 * summation-order epsilon (< 1e-3 abs on these scales).
 *
 * Build (neo / Darwin):
 *   cc -std=c11 -O2 -I. -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *      -Wno-deprecated-declarations tests/test_qmatvec.c notorch.c \
 *      -framework Accelerate -lm -o tests/test_qmatvec
 * Build (Linux):
 *   cc -std=c11 -O2 -I. -DUSE_BLAS tests/test_qmatvec.c notorch.c -lopenblas -lm \
 *      -o tests/test_qmatvec
 */
#include "notorch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* independent f16 -> f32 (oracle must not borrow the kernel's helper) */
static float ref_f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, b;
    if (e == 0) {
        if (m == 0) b = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
               b = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1F) b = (s << 31) | (0xFFu << 23) | (m << 13);
    else b = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &b, 4); return f;
}

/* independent reference dequants (mirror gguf.c / wtf_kernels.c block layouts) */
static void ref_q4_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 18; uint16_t sh; memcpy(&sh, b, 2);
        float sc = ref_f16(sh);
        for (int i = 0; i < 16; i++) {
            d[bk*32+i]    = (float)((int)(b[2+i] & 0x0F) - 8) * sc;
            d[bk*32+i+16] = (float)((int)(b[2+i] >> 4)   - 8) * sc;
        }
    }
}
static void ref_q8_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 34; uint16_t sh; memcpy(&sh, b, 2);
        float sc = ref_f16(sh);
        for (int i = 0; i < 32; i++) d[bk*32+i] = (float)(int8_t)b[2+i] * sc;
    }
}
static void ref_q5_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 22; uint16_t sh; memcpy(&sh, b, 2);
        float sc = ref_f16(sh);
        uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3]<<8) |
                      ((uint32_t)b[4]<<16) | ((uint32_t)b[5]<<24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int h0 = (qh>>j)&1, h1 = (qh>>(j+16))&1;
            d[bk*32+j]    = (float)((lo | (h0<<4)) - 16) * sc;
            d[bk*32+j+16] = (float)((hi | (h1<<4)) - 16) * sc;
        }
    }
}

typedef void (*deqfn)(const uint8_t *, float *, long);

/* all three block-of-32 formats carry their f16 scale at block bytes [0..1] */
static int run_fmt(const char *name, int dtype, int blkbytes, int blkvals, deqfn ref) {
    int m = 512, k = 2048;
    long nb = (long)k / blkvals, stride = nb * blkbytes;
    uint8_t *W = malloc((long)m * stride);
    for (long i = 0; i < (long)m * stride; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (long row = 0; row < m; row++)            /* set a sane normal f16 scale */
        for (long bk = 0; bk < nb; bk++) {
            uint8_t *b = W + row*stride + bk*blkbytes; b[0] = 0x66; b[1] = 0x2A;
        }
    float *x = malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) x[i] = (float)((double)rand()/RAND_MAX*2.0 - 1.0);

    float *Wf = malloc(sizeof(float) * (long)m * k);
    for (int row = 0; row < m; row++) ref(W + (long)row*stride, Wf + (long)row*k, k);
    float *r = malloc(sizeof(float)*m), *g = malloc(sizeof(float)*m);
    nt_blas_matvec(r, Wf, x, m, k);

    int rc = nt_qmatvec(g, W, dtype, x, m, k), ok;
    float maxabs = 0.0f;
    if (rc != 0) { printf("FAIL  %-5s nt_qmatvec rc=%d\n", name, rc); ok = 0; }
    else {
        for (int i = 0; i < m; i++) { float dd = fabsf(r[i]-g[i]); if (dd>maxabs) maxabs = dd; }
        ok = maxabs < 1e-3f;
        printf("%-5s nt_qmatvec [m=%d k=%d] vs dequant->cblas: max abs err %.3g  %s\n",
               name, m, k, maxabs, ok ? "PASS" : "FAIL");
    }
    free(W); free(x); free(Wf); free(r); free(g);
    return ok ? 0 : 1;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += run_fmt("Q4_0", 2, 18, 32, ref_q4_0);
    fails += run_fmt("Q5_0", 6, 22, 32, ref_q5_0);
    fails += run_fmt("Q8_0", 8, 34, 32, ref_q8_0);
    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("%d format(s) FAILED\n", fails); return 1;
}
