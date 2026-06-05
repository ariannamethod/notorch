/*
 * test_qmatvec.c — nt_qmatvec packed quantized matvec vs the dequant->cblas oracle.
 *
 * For each quant format the packed kernel must compute the same matvec as the
 * established path (independent dequant -> nt_blas_matvec). Compared by RELATIVE
 * error (max|ref-got| / max|ref|) so it is robust across the very different output
 * magnitudes of block-of-32 vs super-block-256 formats. A real unpack/scale bug
 * gives rel ~ O(1); f32 summation-order noise gives rel ~ 1e-5.
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
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *mn) {
    if (j < 4) { *s = sc[j] & 63; *mn = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *mn = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4); }
}

/* independent reference dequants (mirror gguf.c block layouts) */
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
static void ref_q4_k(const uint8_t *s, float *out, long n) {
    for (long i = 0; i < n / 256; i++) {
        const uint8_t *b = s + i * 144;
        float d = ref_f16((uint16_t)(b[0]|(b[1]<<8)));
        float dmin = ref_f16((uint16_t)(b[2]|(b[3]<<8)));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0,m0,sc1,m1;
            get_scale_min_k4(is, sc, &sc0,&m0);
            get_scale_min_k4(is+1, sc, &sc1,&m1);
            float d1=d*sc0,mm1=dmin*m0,d2=d*sc1,mm2=dmin*m1;
            for (int l=0;l<32;l++) out[i*256+j+l]    = d1*(float)(qs[qi+l]&0x0F)-mm1;
            for (int l=0;l<32;l++) out[i*256+j+32+l] = d2*(float)(qs[qi+l]>>4)  -mm2;
            qi+=32; is+=2;
        }
    }
}
static void ref_q6_k(const uint8_t *s, float *out, long n) {
    for (long i = 0; i < n / 256; i++) {
        const uint8_t *b = s + i*210, *ql = b, *qh = b+128;
        const int8_t *sc = (const int8_t*)(b+192);
        float d = ref_f16((uint16_t)(b[208]|(b[209]<<8)));
        for (int n_=0;n_<256;n_+=128) {
            const uint8_t *qlh=ql+(n_/128)*64,*qhh=qh+(n_/128)*32;
            const int8_t *sch=sc+(n_/128)*8;
            for (int l=0;l<32;l++) {
                int is=l/16;
                int q1=(int)((qlh[l]   &0x0F)|(((qhh[l]>>0)&3)<<4))-32;
                int q2=(int)((qlh[l+32]&0x0F)|(((qhh[l]>>2)&3)<<4))-32;
                int q3=(int)((qlh[l]   >>4)  |(((qhh[l]>>4)&3)<<4))-32;
                int q4=(int)((qlh[l+32]>>4)  |(((qhh[l]>>6)&3)<<4))-32;
                out[i*256+n_+l]    = d*sch[is+0]*q1;
                out[i*256+n_+l+32] = d*sch[is+2]*q2;
                out[i*256+n_+l+64] = d*sch[is+4]*q3;
                out[i*256+n_+l+96] = d*sch[is+6]*q4;
            }
        }
    }
}

typedef void (*deqfn)(const uint8_t *, float *, long);
typedef void (*setfn)(uint8_t *);

/* sane normal f16 scales (0x2A66) at the per-format offsets; other bytes stay random */
static void set_s32(uint8_t *b) { b[0]=0x66; b[1]=0x2A; }                       /* Q4_0/Q5_0/Q8_0 */
static void set_q4k(uint8_t *b) { b[0]=0x66; b[1]=0x2A; b[2]=0x66; b[3]=0x26; } /* d, dmin */
static void set_q6k(uint8_t *b) { b[208]=0x66; b[209]=0x2A; }                   /* d */

static int run_fmt(const char *name, int dtype, int blkbytes, int blkvals,
                   deqfn ref, setfn setblk) {
    int m = 512, k = 2048;
    long nb = (long)k / blkvals, stride = nb * blkbytes;
    uint8_t *W = malloc((long)m * stride);
    for (long i = 0; i < (long)m * stride; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (long row = 0; row < m; row++)
        for (long bk = 0; bk < nb; bk++) setblk(W + row*stride + bk*blkbytes);
    float *x = malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) x[i] = (float)((double)rand()/RAND_MAX*2.0 - 1.0);

    float *Wf = malloc(sizeof(float) * (long)m * k);
    for (int row = 0; row < m; row++) ref(W + (long)row*stride, Wf + (long)row*k, k);
    float *r = malloc(sizeof(float)*m), *g = malloc(sizeof(float)*m);
    nt_blas_matvec(r, Wf, x, m, k);

    int rc = nt_qmatvec(g, W, dtype, x, m, k), ok;
    if (rc != 0) { printf("FAIL  %-5s nt_qmatvec rc=%d\n", name, rc); ok = 0; }
    else {
        float maxabs = 0, maxref = 0;
        for (int i = 0; i < m; i++) {
            float dd = fabsf(r[i]-g[i]); if (dd > maxabs) maxabs = dd;
            float a = fabsf(r[i]);       if (a  > maxref) maxref = a;
        }
        float rel = maxref > 0 ? maxabs / maxref : maxabs;
        ok = rel < 1e-3f;
        printf("%-5s [m=%d k=%d] abs %.3g / |ref| %.3g = rel %.2g  %s\n",
               name, m, k, maxabs, maxref, rel, ok ? "PASS" : "FAIL");
    }
    free(W); free(x); free(Wf); free(r); free(g);
    return ok ? 0 : 1;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += run_fmt("Q4_0", 2,  18,  32, ref_q4_0, set_s32);
    fails += run_fmt("Q5_0", 6,  22,  32, ref_q5_0, set_s32);
    fails += run_fmt("Q8_0", 8,  34,  32, ref_q8_0, set_s32);
    fails += run_fmt("Q4_K", 12, 144, 256, ref_q4_k, set_q4k);
    fails += run_fmt("Q6_K", 14, 210, 256, ref_q6_k, set_q6k);
    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("%d format(s) FAILED\n", fails); return 1;
}
