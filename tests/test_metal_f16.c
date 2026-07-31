/* test_metal_f16 — the GPU F16 matvec against nt_qmatvec's F16 reference.
 *
 * The K-quant kernels require k % 256 == 0, which locks out any architecture
 * whose inner dimension is not a multiple of 256. Janus v4 is E=640, M=1664 —
 * so those two shapes are tested explicitly, not only the friendly power-of-two
 * case. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "notorch.h"
#include "notorch_metal.h"

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static int one_shape(int m, int k) {
    uint16_t *W = malloc((size_t)m * k * 2);
    float *x    = malloc((size_t)k * sizeof(float));
    float *ref  = malloc((size_t)m * sizeof(float));
    float *gpu  = malloc((size_t)m * sizeof(float));
    if (!W || !x || !ref || !gpu) { fprintf(stderr, "oom\n"); return 1; }

    srand(1234 + m + k);
    for (long i = 0; i < (long)m * k; i++)
        W[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 0.5f);
    for (int i = 0; i < k; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    if (nt_qmatvec(ref, (const uint8_t *)W, 1 /* GGUF F16 */, x, m, k) != 0) {
        fprintf(stderr, "nt_qmatvec F16 reference failed for m=%d k=%d\n", m, k);
        return 1;
    }
    int rc = nt_metal_f16_matvec(gpu, (const uint8_t *)W, x, m, k);
    if (rc != 0) { fprintf(stderr, "nt_metal_f16_matvec rc=%d (m=%d k=%d)\n", rc, m, k); return 1; }

    double abs_max = 0, ref_mag = 0;
    for (int i = 0; i < m; i++) {
        double d = fabs((double)gpu[i] - (double)ref[i]);
        if (d > abs_max) abs_max = d;
        if (fabs((double)ref[i]) > ref_mag) ref_mag = fabs((double)ref[i]);
    }
    double rel = ref_mag > 0 ? abs_max / ref_mag : abs_max;
    int pass = rel < 1e-5;
    printf("F16 [m=%-5d k=%-5d] abs %.3e / |ref| %.3g = rel %.2g  %s\n",
           m, k, abs_max, ref_mag, rel, pass ? "PASS" : "FAIL");

    free(W); free(x); free(ref); free(gpu);
    return pass ? 0 : 1;
}

int main(void) {
    if (!nt_metal_available()) { printf("no Metal device — skipping\n"); return 0; }
    int bad = 0;
    bad |= one_shape(4096, 2048);   /* the friendly case */
    bad |= one_shape(640,  640);    /* Janus attention: k not a multiple of 256 */
    bad |= one_shape(1664, 640);    /* Janus FFN gate/up */
    bad |= one_shape(640,  1664);   /* Janus FFN down */
    bad |= one_shape(32768, 640);   /* Janus lm_head */
    bad |= one_shape(7,    33);     /* tail rows and an odd k */
    printf(bad ? "FAILURES\n" : "ALL PASS\n");
    return bad;
}
