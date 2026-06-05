/*
 * test_qmatvec.c — nt_qmatvec packed Q4_0 vs the dequant->cblas_sgemv oracle.
 *
 * The packed kernel must compute the same matvec as the established path
 * (gguf_dequant -> nt_blas_matvec), within f32 summation-order epsilon.
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

/* independent f16 -> f32 (so the oracle does not borrow the kernel's helper) */
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

/* independent Q4_0 dequant (mirrors gguf.c:dequant_q4_0, 18 B / 32 vals) */
static void ref_dequant_q4_0(const uint8_t *src, float *dst, long n) {
    long nb = n / 32;
    for (long bk = 0; bk < nb; bk++) {
        const uint8_t *bl = src + bk * 18;
        uint16_t sh; memcpy(&sh, bl, 2);
        float sc = ref_f16(sh);
        for (int i = 0; i < 16; i++) {
            int lo = (int)(bl[2 + i] & 0x0F) - 8;
            int hi = (int)(bl[2 + i] >> 4)   - 8;
            dst[bk * 32 + i]      = (float)lo * sc;
            dst[bk * 32 + i + 16] = (float)hi * sc;
        }
    }
}

int main(void) {
    srand(42);
    int m = 512, k = 2048;                 /* k % 32 == 0 */
    long nb = (long)k / 32;
    long wbytes = (long)m * nb * 18;

    uint8_t *W = malloc(wbytes);
    for (long i = 0; i < wbytes; i++) W[i] = (uint8_t)(rand() & 0xFF);
    /* deterministic normal f16 scale (0x2A66) per block — avoid inf/nan */
    for (long row = 0; row < m; row++)
        for (long bk = 0; bk < nb; bk++) {
            uint8_t *b = W + (row * nb + bk) * 18; b[0] = 0x66; b[1] = 0x2A;
        }

    float *x = malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) x[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);

    /* oracle: dequant whole W to dense f32, then cblas matvec */
    float *Wf = malloc(sizeof(float) * (long)m * k);
    for (int row = 0; row < m; row++)
        ref_dequant_q4_0(W + (long)row * nb * 18, Wf + (long)row * k, k);
    float *ref = malloc(sizeof(float) * m), *got = malloc(sizeof(float) * m);
    nt_blas_matvec(ref, Wf, x, m, k);

    int rc = nt_qmatvec(got, W, 2 /* GGUF_TYPE_Q4_0 */, x, m, k);
    if (rc != 0) { printf("FAIL: nt_qmatvec returned %d\n", rc); return 1; }

    float maxabs = 0.0f; int worst = -1;
    for (int i = 0; i < m; i++) {
        float d = fabsf(ref[i] - got[i]);
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    printf("nt_qmatvec Q4_0 [m=%d k=%d] vs dequant->cblas: max abs err %.3g "
           "(ref=%.5f got=%.5f @row %d)\n",
           m, k, maxabs, worst >= 0 ? ref[worst] : 0, worst >= 0 ? got[worst] : 0, worst);

    free(W); free(x); free(Wf); free(ref); free(got);
    if (maxabs < 1e-3f) { printf("PASS\n"); return 0; }
    printf("FAIL (max abs err >= 1e-3)\n"); return 1;
}
