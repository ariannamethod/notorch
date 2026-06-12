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
#include <sys/mman.h>
#include <unistd.h>

#ifndef USE_METAL
/* ── Reference dequant for Q6_K, identical to gguf.c:dequant_q6_k ────── */

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

/* Random-but-valid Q6_K row (210 bytes/block: ql[128] qh[64] sc[16] d[2]). */
static void make_random_q6k_row(uint8_t *row, uint64_t k, unsigned *rng)
{
    uint64_t nblocks = k / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        uint8_t *b = row + i * 210;
        for (int j = 0; j < 208; j++) b[j] = (uint8_t)(rand_r(rng) & 0xFF);
        uint16_t d_bits = (uint16_t)(0x3000 + (rand_r(rng) & 0x0FFF));  /* ~[0.125,0.25) */
        b[208] = (uint8_t)(d_bits & 0xFF); b[209] = (uint8_t)(d_bits >> 8);
    }
}

/* ── Reference dequant for Q6_K, identical to gguf.c:dequant_q6_k ────── */
static void dequant_q6_k_ref(const uint8_t *data, float *out, uint64_t n)
{
    uint64_t nblocks = n / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b  = data + i * 210;
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 192);
        float d = f16_to_f32_((uint16_t)(b[208] | (b[209] << 8)));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            const uint8_t *qlh = ql + (n_ / 128) * 64;
            const uint8_t *qhh = qh + (n_ / 128) * 32;
            const int8_t  *sch = sc + (n_ / 128) * 8;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                out[i*256 + n_ + l]      = d * sch[is + 0] * q1;
                out[i*256 + n_ + l + 32] = d * sch[is + 2] * q2;
                out[i*256 + n_ + l + 64] = d * sch[is + 4] * q3;
                out[i*256 + n_ + l + 96] = d * sch[is + 6] * q4;
            }
        }
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

    /* ── Q6_K determinism gate (Mythos): the same (W,x) run twice must be
     * bit-identical. A stability tripwire for any future non-deterministic
     * Metal Q6_K path (the cleaned eviction-debug class) — not a correctness
     * check, a regression guard. */
    {
        const int q6m = 64, q6k = 512;
        const uint64_t q6_row = (uint64_t)(q6k / 256) * 210;
        uint8_t *Wq6 = (uint8_t *)malloc((size_t)q6m * q6_row);
        float   *xq6 = (float *)  malloc((size_t)q6k * sizeof(float));
        float   *o1  = (float *)  calloc((size_t)q6m, sizeof(float));
        float   *o2  = (float *)  calloc((size_t)q6m, sizeof(float));
        unsigned r6 = 0xB16B00B5u;
        for (int i = 0; i < q6m; i++)
            make_random_q6k_row(Wq6 + (uint64_t)i * q6_row, (uint64_t)q6k, &r6);
        for (int j = 0; j < q6k; j++)
            xq6[j] = ((float)rand_r(&r6) / (float)RAND_MAX) - 0.5f;
        nt_metal_q6k_matvec(o1, Wq6, xq6, q6m, q6k);
        nt_metal_q6k_matvec(o2, Wq6, xq6, q6m, q6k);
        int det = (memcmp(o1, o2, (size_t)q6m * sizeof(float)) == 0);
        printf("test_metal_q6k_determinism: %s (2x same input bit-identical)\n",
               det ? "PASS" : "FAIL");
        ok = ok && det;
        free(Wq6); free(xq6); free(o1); free(o2);
    }

    /* ── Q6_K correctness vs the gguf.c reference dequant ──────────────── */
    {
        const int m6 = 96, k6 = 768;                 /* 3 Q6_K blocks per row */
        const uint64_t r6b = (uint64_t)(k6 / 256) * 210;
        uint8_t *W6  = (uint8_t *)malloc((size_t)m6 * r6b);
        float   *x6  = (float *)  malloc((size_t)k6 * sizeof(float));
        float   *Wf  = (float *)  malloc((size_t)m6 * (size_t)k6 * sizeof(float));
        float   *ref = (float *)  calloc((size_t)m6, sizeof(float));
        float   *gpu = (float *)  calloc((size_t)m6, sizeof(float));
        unsigned rq = 0xDECAF;
        for (int i = 0; i < m6; i++)
            make_random_q6k_row(W6 + (uint64_t)i * r6b, (uint64_t)k6, &rq);
        for (int j = 0; j < k6; j++)
            x6[j] = ((float)rand_r(&rq) / (float)RAND_MAX) - 0.5f;
        for (int i = 0; i < m6; i++)
            dequant_q6_k_ref(W6 + (uint64_t)i * r6b, Wf + (uint64_t)i * k6, (uint64_t)k6);
        for (int i = 0; i < m6; i++) {
            double s = 0.0;
            for (int j = 0; j < k6; j++) s += (double)Wf[i * k6 + j] * (double)x6[j];
            ref[i] = (float)s;
        }
        int rc6 = nt_metal_q6k_matvec(gpu, W6, x6, m6, k6);
        float mr = 0.f; int wi = 0;
        for (int i = 0; i < m6; i++) {
            float rel = fabsf(gpu[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f);
            if (rel > mr) { mr = rel; wi = i; }
        }
        int ok6 = (rc6 == 0) && (mr < 5e-4f);
        printf("test_metal_q6k: m=%d k=%d  max_rel=%.3e  (worst idx=%d ref=%.5f gpu=%.5f)  %s\n",
               m6, k6, mr, wi, ref[wi], gpu[wi], ok6 ? "PASS" : "FAIL");
        ok = ok && ok6;
        free(W6); free(x6); free(Wf); free(ref); free(gpu);
    }

    /* ── Token-graph gate: solo determinism + batch ≡ solo bit-identical ── */
    {
        const int bm = 80, bk = 512;
        const uint64_t r4b = (uint64_t)(bk / 256) * 144;
        const uint64_t r6b = (uint64_t)(bk / 256) * 210;
        uint8_t *W4 = (uint8_t *)malloc((size_t)bm * r4b);
        uint8_t *W6 = (uint8_t *)malloc((size_t)bm * r6b);
        float *x1 = (float *)malloc((size_t)bk * sizeof(float));
        float *x2 = (float *)malloc((size_t)bk * sizeof(float));
        float *s1 = (float *)calloc((size_t)bm, sizeof(float));
        float *s2 = (float *)calloc((size_t)bm, sizeof(float));
        float *s3 = (float *)calloc((size_t)bm, sizeof(float));
        float *d1 = (float *)calloc((size_t)bm, sizeof(float));
        float *b1 = (float *)calloc((size_t)bm, sizeof(float));
        float *b2 = (float *)calloc((size_t)bm, sizeof(float));
        float *b3 = (float *)calloc((size_t)bm, sizeof(float));
        unsigned rb = 0xFEED;
        for (int i = 0; i < bm; i++) make_random_q4k_row(W4 + (uint64_t)i * r4b, (uint64_t)bk, &rb);
        for (int i = 0; i < bm; i++) make_random_q6k_row(W6 + (uint64_t)i * r6b, (uint64_t)bk, &rb);
        for (int j = 0; j < bk; j++) x1[j] = ((float)rand_r(&rb) / (float)RAND_MAX) - 0.5f;
        for (int j = 0; j < bk; j++) x2[j] = ((float)rand_r(&rb) / (float)RAND_MAX) - 0.5f;

        /* solo reference trio: a {q,k,v}-style share of x1 + a second input */
        nt_metal_q4k_matvec(s1, W4, x1, bm, bk);
        nt_metal_q6k_matvec(s2, W6, x1, bm, bk);
        nt_metal_q4k_matvec(s3, W4, x2, bm, bk);
        /* solo determinism (q4k twin of the q6k gate below) */
        nt_metal_q4k_matvec(d1, W4, x1, bm, bk);
        int det4 = (memcmp(s1, d1, (size_t)bm * sizeof(float)) == 0);
        printf("test_metal_q4k_determinism: %s (2x same input bit-identical)\n",
               det4 ? "PASS" : "FAIL");
        ok = ok && det4;

        /* the same trio through ONE command buffer */
        int brc = nt_metal_batch_begin();
        if (brc == 0) {
            nt_metal_q4k_matvec(b1, W4, x1, bm, bk);
            nt_metal_q6k_matvec(b2, W6, x1, bm, bk);
            nt_metal_q4k_matvec(b3, W4, x2, bm, bk);
            brc = nt_metal_batch_commit();
        }
        int okb = (brc == 0) &&
                  (memcmp(b1, s1, (size_t)bm * sizeof(float)) == 0) &&
                  (memcmp(b2, s2, (size_t)bm * sizeof(float)) == 0) &&
                  (memcmp(b3, s3, (size_t)bm * sizeof(float)) == 0);
        printf("test_metal_batch: %s (batched trio bit-identical to solo, rc=%d)\n",
               okb ? "PASS" : "FAIL", brc);
        ok = ok && okb;
        free(W4); free(W6); free(x1); free(x2);
        free(s1); free(s2); free(s3); free(d1); free(b1); free(b2); free(b3);
    }

    /* ── M3 gate: simdgroup path vs naive reference + determinism ──────── */
    {
        const int sm = 128, sk = 1024;
        const uint64_t r4 = (uint64_t)(sk / 256) * 144;
        const uint64_t r6 = (uint64_t)(sk / 256) * 210;
        uint8_t *W4 = (uint8_t *)malloc((size_t)sm * r4);
        uint8_t *W6 = (uint8_t *)malloc((size_t)sm * r6);
        float *xs = (float *)malloc((size_t)sk * sizeof(float));
        float *g1 = (float *)calloc((size_t)sm, sizeof(float));
        float *g1b = (float *)calloc((size_t)sm, sizeof(float));
        float *g2 = (float *)calloc((size_t)sm, sizeof(float));
        float *n1 = (float *)calloc((size_t)sm, sizeof(float));
        float *n2 = (float *)calloc((size_t)sm, sizeof(float));
        unsigned rs = 0xA11CE;
        for (int i = 0; i < sm; i++) make_random_q4k_row(W4 + (uint64_t)i * r4, (uint64_t)sk, &rs);
        for (int i = 0; i < sm; i++) make_random_q6k_row(W6 + (uint64_t)i * r6, (uint64_t)sk, &rs);
        for (int j = 0; j < sk; j++) xs[j] = ((float)rand_r(&rs) / (float)RAND_MAX) - 0.5f;

        /* simdgroup path is opt-in (NT_METAL_SG=1) since the 24B doe A/B
         * regression; library default is naive */
        nt_metal_shutdown();
        setenv("NT_METAL_SG", "1", 1);
        nt_metal_q4k_matvec(g1, W4, xs, sm, sk);
        nt_metal_q4k_matvec(g1b, W4, xs, sm, sk);
        nt_metal_q6k_matvec(g2, W6, xs, sm, sk);
        int sdet = (memcmp(g1, g1b, (size_t)sm * sizeof(float)) == 0);
        printf("test_metal_sg_determinism: %s (2x same input bit-identical)\n",
               sdet ? "PASS" : "FAIL");
        ok = ok && sdet;

        /* default (naive) path: different reduction order — tolerance gate */
        unsetenv("NT_METAL_SG");
        nt_metal_shutdown();
        nt_metal_q4k_matvec(n1, W4, xs, sm, sk);
        nt_metal_q6k_matvec(n2, W6, xs, sm, sk);
        nt_metal_shutdown();
        float mr4 = 0.f, mr6 = 0.f;
        for (int i = 0; i < sm; i++) {
            float r4e = fabsf(g1[i] - n1[i]) / (fabsf(n1[i]) + 1e-6f);
            float r6e = fabsf(g2[i] - n2[i]) / (fabsf(n2[i]) + 1e-6f);
            if (r4e > mr4) mr4 = r4e;
            if (r6e > mr6) mr6 = r6e;
        }
        int oksg = (mr4 < 5e-4f) && (mr6 < 5e-4f);
        printf("test_metal_sg_vs_naive: q4k max_rel=%.3e q6k max_rel=%.3e  %s\n",
               mr4, mr6, oksg ? "PASS" : "FAIL");
        ok = ok && oksg;
        free(W4); free(W6); free(xs); free(g1); free(g1b); free(g2); free(n1); free(n2);
    }

    /* ── M4 gates: layer ops vs CPU references + chained batch ─────────── */
    {
        const int E = 512, NH = 4, KVH = 2, HD = 64, T = 9;
        const float EPS = 1e-5f, THETA = 10000.0f;
        unsigned rm = 0x4D4;
        float *xv = (float *)malloc((size_t)E * sizeof(float));
        float *wn = (float *)malloc((size_t)E * sizeof(float));
        float *ref = (float *)malloc((size_t)E * sizeof(float));
        float *gpu = (float *)malloc((size_t)E * sizeof(float));
        for (int i = 0; i < E; i++) {
            xv[i] = ((float)rand_r(&rm) / (float)RAND_MAX) - 0.5f;
            wn[i] = 0.5f + (float)rand_r(&rm) / (float)RAND_MAX;
        }
        int m4ok = 1;

        /* rmsnorm */
        nt_metal_slot_alloc(0, (uint64_t)E * 4);
        nt_metal_slot_alloc(1, (uint64_t)E * 4);
        nt_metal_slot_upload(0, xv, (uint64_t)E * 4);
        nt_metal_rmsnorm(1, 0, wn, E, EPS);
        nt_metal_slot_download(1, gpu, (uint64_t)E * 4);
        {
            double ss = 0; for (int i = 0; i < E; i++) ss += (double)xv[i] * xv[i];
            float rinv = 1.0f / sqrtf((float)(ss / E) + EPS);
            float mr = 0;
            for (int i = 0; i < E; i++) {
                ref[i] = xv[i] * rinv * wn[i];
                float r = fabsf(gpu[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f);
                if (r > mr) mr = r;
            }
            printf("test_metal_rmsnorm: max_rel=%.3e %s\n", mr, mr < 5e-4f ? "PASS" : "FAIL");
            m4ok = m4ok && (mr < 5e-4f);
        }

        /* rope (in place on slot 0, NH heads x HD) */
        {
            const int QN = NH * HD;  /* 256 <= E */
            const int POS = 7;
            nt_metal_slot_upload(0, xv, (uint64_t)QN * 4);
            nt_metal_rope(0, NH, HD, POS, THETA);
            nt_metal_slot_download(0, gpu, (uint64_t)QN * 4);
            memcpy(ref, xv, (size_t)QN * 4);
            for (int h = 0; h < NH; h++)
                for (int i = 0; i < HD / 2; i++) {
                    float fr = powf(THETA, -2.0f * (float)i / (float)HD);
                    float c = cosf((float)POS * fr), s = sinf((float)POS * fr);
                    float x0 = ref[h * HD + i], x1 = ref[h * HD + i + HD / 2];
                    ref[h * HD + i]          = x0 * c - x1 * s;
                    ref[h * HD + i + HD / 2] = x0 * s + x1 * c;
                }
            float mr = 0;
            for (int i = 0; i < QN; i++) {
                float r = fabsf(gpu[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f);
                if (r > mr) mr = r;
            }
            printf("test_metal_rope: max_rel=%.3e %s\n", mr, mr < 5e-4f ? "PASS" : "FAIL");
            m4ok = m4ok && (mr < 5e-4f);
        }

        /* silu_mul + add */
        {
            float *g2 = (float *)malloc((size_t)E * 4);
            for (int i = 0; i < E; i++) g2[i] = ((float)rand_r(&rm) / (float)RAND_MAX) - 0.5f;
            nt_metal_slot_alloc(2, (uint64_t)E * 4);
            nt_metal_slot_upload(0, xv, (uint64_t)E * 4);
            nt_metal_slot_upload(1, g2, (uint64_t)E * 4);
            nt_metal_silu_mul(2, 0, 1, E);
            nt_metal_slot_download(2, gpu, (uint64_t)E * 4);
            float mr = 0;
            for (int i = 0; i < E; i++) {
                float si = xv[i] / (1.0f + expf(-xv[i]));
                float rf = si * g2[i];
                float r = fabsf(gpu[i] - rf) / (fabsf(rf) + 1e-6f);
                if (r > mr) mr = r;
            }
            nt_metal_add(2, 0, 1, E);
            nt_metal_slot_download(2, gpu, (uint64_t)E * 4);
            float mra = 0;
            for (int i = 0; i < E; i++) {
                float rf = xv[i] + g2[i];
                float r = fabsf(gpu[i] - rf) / (fabsf(rf) + 1e-6f);
                if (r > mra) mra = r;
            }
            printf("test_metal_silu_mul: max_rel=%.3e %s\n", mr, mr < 5e-4f ? "PASS" : "FAIL");
            printf("test_metal_add: max_rel=%.3e %s\n", mra, mra < 5e-4f ? "PASS" : "FAIL");
            m4ok = m4ok && (mr < 5e-4f) && (mra < 5e-4f);
            free(g2);
        }

        /* attn_decode: KV in a registered (page-aligned) region */
        {
            const int QDIM = NH * HD;
            size_t pgsz = (size_t)getpagesize();
            size_t kv_bytes = ((size_t)KVH * T * HD * 4 * 2 + pgsz - 1) & ~(pgsz - 1);
            uint8_t *kvbase = (uint8_t *)mmap(NULL, kv_bytes, PROT_READ | PROT_WRITE,
                                              MAP_ANON | MAP_PRIVATE, -1, 0);
            float *Kc = (float *)kvbase;
            float *Vc = Kc + (size_t)KVH * T * HD;
            float *qv = (float *)malloc((size_t)QDIM * 4);
            float *att = (float *)malloc((size_t)QDIM * 4);
            for (int i = 0; i < KVH * T * HD; i++) {
                Kc[i] = ((float)rand_r(&rm) / (float)RAND_MAX) - 0.5f;
                Vc[i] = ((float)rand_r(&rm) / (float)RAND_MAX) - 0.5f;
            }
            for (int i = 0; i < QDIM; i++) qv[i] = ((float)rand_r(&rm) / (float)RAND_MAX) - 0.5f;
            nt_metal_register_region(kvbase, kv_bytes);
            nt_metal_slot_alloc(3, (uint64_t)QDIM * 4);
            nt_metal_slot_alloc(4, (uint64_t)QDIM * 4);
            nt_metal_slot_upload(3, qv, (uint64_t)QDIM * 4);
            float scale = 1.0f / sqrtf((float)HD);
            /* layout: K[kvh][t][d] -> head stride T*HD, pos stride HD */
            int rc = nt_metal_attn_decode(4, 3, Kc, Vc, T, NH, KVH, HD,
                                          (uint32_t)HD, (uint32_t)(T * HD),
                                          (uint32_t)HD, (uint32_t)(T * HD), scale);
            nt_metal_slot_download(4, att, (uint64_t)QDIM * 4);
            float mr = 0;
            for (int h = 0; h < NH && rc == 0; h++) {
                int g = h / (NH / KVH);
                const float *qh = qv + h * HD;
                const float *Kh = Kc + (size_t)g * T * HD;
                const float *Vh = Vc + (size_t)g * T * HD;
                double sc[64]; double smax = -1e30;
                for (int p = 0; p < T; p++) {
                    double d = 0;
                    for (int dd = 0; dd < HD; dd++) d += (double)qh[dd] * Kh[p * HD + dd];
                    sc[p] = d * scale;
                    if (sc[p] > smax) smax = sc[p];
                }
                double ssum = 0;
                for (int p = 0; p < T; p++) { sc[p] = exp(sc[p] - smax); ssum += sc[p]; }
                for (int dd = 0; dd < HD; dd++) {
                    double a = 0;
                    for (int p = 0; p < T; p++) a += sc[p] * Vh[p * HD + dd];
                    float rf = (float)(a / ssum);
                    float r = fabsf(att[h * HD + dd] - rf) / (fabsf(rf) + 1e-6f);
                    if (r > mr) mr = r;
                }
            }
            printf("test_metal_attn_decode: rc=%d max_rel=%.3e %s\n",
                   rc, mr, (rc == 0 && mr < 5e-4f) ? "PASS" : "FAIL");
            m4ok = m4ok && (rc == 0) && (mr < 5e-4f);

            /* chained batch: rmsnorm -> silu_mul -> add in ONE command buffer,
             * bit-identical to the solo sequence above re-run */
            float *solo3 = (float *)malloc((size_t)E * 4);
            float *bat3  = (float *)malloc((size_t)E * 4);
            nt_metal_slot_upload(0, xv, (uint64_t)E * 4);
            nt_metal_rmsnorm(1, 0, wn, E, EPS);
            nt_metal_silu_mul(2, 1, 0, E);
            nt_metal_add(1, 2, 0, E);
            nt_metal_slot_download(1, solo3, (uint64_t)E * 4);
            nt_metal_slot_upload(0, xv, (uint64_t)E * 4);
            nt_metal_batch_begin();
            nt_metal_rmsnorm(1, 0, wn, E, EPS);
            nt_metal_silu_mul(2, 1, 0, E);
            nt_metal_add(1, 2, 0, E);
            int brc2 = nt_metal_batch_commit();
            nt_metal_slot_download(1, bat3, (uint64_t)E * 4);
            int chain_ok = (brc2 == 0) && (memcmp(solo3, bat3, (size_t)E * 4) == 0);
            printf("test_metal_chain_batch: %s (3-op chain batched bit-identical to solo)\n",
                   chain_ok ? "PASS" : "FAIL");
            m4ok = m4ok && chain_ok;
            free(qv); free(att); free(solo3); free(bat3);
            munmap(kvbase, kv_bytes);
        }
        ok = ok && m4ok;
        free(xv); free(wn); free(ref); free(gpu);
    }

    nt_metal_shutdown();
    return ok ? 0 : 4;
}

#endif /* USE_METAL */
