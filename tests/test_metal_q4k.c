/* tests/test_metal_q4k.c — correctness of the Metal Q4_K matvec kernel
 * against a scalar reference dequant + double-precision matvec.
 *
 * The reference dequant here is a private copy of gguf.c:dequant_q4_k —
 * we keep it local so the test is self-contained and the comparison is
 * literally byte-for-byte against the production C code that the
 * shader was ported from.
 *
 * Tolerance: max relative error per output element. Because the kernel
 * runs the inner dot in fp32 and the reference accumulates in double
 * before downcast, we expect ~1e-5 on random inputs at k=512.
 *
 * by Claude (Arianna Method)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef USE_METAL
int main(void) {
    fprintf(stderr, "test_metal_q4k: not built with -DUSE_METAL, skipping\n");
    return 0;
}
#else

#include "notorch_metal.h"

/* ── fp16 → fp32 (IEEE-754 half, same conversion as gguf.c) ──────────── */
static float f16_to_f32_(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            exp += 127 - 15;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        u = sign | (exp << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

/* ── Reference dequant, identical to gguf.c:dequant_q4_k ─────────────── */
static void get_scale_min_k4_ref(int j, const uint8_t *sc, uint8_t *s, uint8_t *m)
{
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else {
        *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4)    | ((sc[j]     >> 6) << 4);
    }
}

static void dequant_q4_k_ref(const uint8_t *data, float *out, uint64_t n)
{
    uint64_t nblocks = n / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * 144;
        float d    = f16_to_f32_((uint16_t)(b[0] | (b[1] << 8)));
        float dmin = f16_to_f32_((uint16_t)(b[2] | (b[3] << 8)));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0; int oi = (int)(i * 256);
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4_ref(is,     sc, &sc0, &m0);
            get_scale_min_k4_ref(is + 1, sc, &sc1, &m1v);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                out[oi + j + l]      = d1 * (float)(qs[qi + l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                out[oi + j + 32 + l] = d2 * (float)(qs[qi + l] >> 4) - mm2;
            qi += 32; is += 2;
        }
    }
}

/* Construct a random-but-structurally-valid Q4_K row. d and dmin are
 * chosen as small positive fp16 values so dequantized weights have
 * sensible magnitude; the scale/min/quant bytes are arbitrary. */
static void make_random_q4k_row(uint8_t *row, uint64_t k, unsigned *rng)
{
    uint64_t nblocks = k / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        uint8_t *b = row + i * 144;
        uint16_t d_bits    = (uint16_t)(0x3000 + (rand_r(rng) & 0x0FFF));  /* ~[0.125, 0.25) */
        uint16_t dmin_bits = (uint16_t)(0x2C00 + (rand_r(rng) & 0x07FF));  /* ~[0.0625, 0.094) */
        b[0] = (uint8_t)(d_bits    & 0xFF); b[1] = (uint8_t)(d_bits    >> 8);
        b[2] = (uint8_t)(dmin_bits & 0xFF); b[3] = (uint8_t)(dmin_bits >> 8);
        for (int j = 4; j < 144; j++) b[j] = (uint8_t)(rand_r(rng) & 0xFF);
    }
}

int main(void)
{
    if (!nt_metal_available()) {
        fprintf(stderr, "test_metal_q4k: no Metal device available, skipping\n");
        return 0;
    }
    int init_rc = nt_metal_init();
    if (init_rc != 0) {
        fprintf(stderr, "test_metal_q4k: nt_metal_init failed rc=%d\n", init_rc);
        return 1;
    }

    const int m = 128;
    const int k = 512;                /* 2 Q4_K blocks per row */
    const uint64_t row_bytes = (uint64_t)(k / 256) * 144;

    uint8_t *W         = (uint8_t *)malloc((size_t)m * row_bytes);
    float   *x         = (float *)  malloc((size_t)k * sizeof(float));
    float   *W_ref_f32 = (float *)  malloc((size_t)m * (size_t)k * sizeof(float));
    float   *out_ref   = (float *)  calloc((size_t)m, sizeof(float));
    float   *out_gpu   = (float *)  calloc((size_t)m, sizeof(float));
    if (!W || !x || !W_ref_f32 || !out_ref || !out_gpu) {
        fprintf(stderr, "test_metal_q4k: host alloc failed\n");
        return 2;
    }

    unsigned rng = 0xC0FFEE;
    for (int i = 0; i < m; i++)
        make_random_q4k_row(W + (uint64_t)i * row_bytes, (uint64_t)k, &rng);
    for (int j = 0; j < k; j++)
        x[j] = ((float)rand_r(&rng) / (float)RAND_MAX) - 0.5f;

    /* Reference: dequant every row, then scalar matvec in double. */
    for (int i = 0; i < m; i++)
        dequant_q4_k_ref(W + (uint64_t)i * row_bytes,
                         W_ref_f32 + (uint64_t)i * k,
                         (uint64_t)k);
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = 0; j < k; j++)
            s += (double)W_ref_f32[i * k + j] * (double)x[j];
        out_ref[i] = (float)s;
    }

    /* GPU path under test. */
    int rc = nt_metal_q4k_matvec(out_gpu, W, x, m, k);
    if (rc != 0) {
        fprintf(stderr, "test_metal_q4k: nt_metal_q4k_matvec failed rc=%d\n", rc);
        return 3;
    }

    float max_abs = 0.f, max_rel = 0.f;
    int   worst_i = 0;
    for (int i = 0; i < m; i++) {
        float diff = fabsf(out_gpu[i] - out_ref[i]);
        float rel  = diff / (fabsf(out_ref[i]) + 1e-6f);
        if (diff > max_abs) max_abs = diff;
        if (rel  > max_rel) { max_rel = rel; worst_i = i; }
    }

    const float tol = 5e-4f;            /* fp32 GPU dot vs double host dot */
    int ok = (max_rel < tol);

    printf("test_metal_q4k: m=%d k=%d  max_abs=%.3e  max_rel=%.3e  "
           "(worst idx=%d ref=%.5f gpu=%.5f)\n",
           m, k, max_abs, max_rel, worst_i, out_ref[worst_i], out_gpu[worst_i]);
    printf("test_metal_q4k: %s (tol=%.0e)\n", ok ? "PASS" : "FAIL", tol);

    free(W); free(x); free(W_ref_f32); free(out_ref); free(out_gpu);
    nt_metal_shutdown();
    return ok ? 0 : 4;
}

#endif /* USE_METAL */
