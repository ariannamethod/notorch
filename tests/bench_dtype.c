/* bench_dtype.c — the same matvec in every packed format, so a slow tensor can be told from a
 * slow format.
 *
 * Written to answer one question and answered a larger one. OLMoE's output head is its file's
 * only Q6_K tensor and it read 80.6 MiB a token at 12.5 GiB/s where the Q4_0 attention weights
 * beside it reached 17.3 — which could have been the head's shape, its place in the forward,
 * or the format. It is the format: Q6_K measures 12.7 GiB/s here in isolation, so the head runs
 * at exactly the rate this kernel gives and holds nothing model-specific.
 *
 * The larger answer is in the same table. On identical byte counts, 55.27 MiB either way, Q4_0
 * reaches 17.3 GiB/s and Q4_K 13.0 — a quarter slower for the same reading, which is the
 * super-block unpacking rather than the memory. Q8_0 reads nearly twice Q4_K's bytes and takes
 * a third longer. Anyone measuring a K-quant model on this library is measuring that gap, and
 * Q4_K_M is what most downloads are.
 *
 * Exynos 1580, four big cores, [50304, 2048], three runs each within two percent:
 *   Q4_0  55.27 MiB  3.1 ms  17.3 GiB/s     Q4_K  55.27 MiB  4.2 ms  13.0 GiB/s
 *   Q8_0 104.39 MiB  5.5 ms  18.7 GiB/s     Q6_K  80.60 MiB  6.2 ms  12.7 GiB/s
 *   Q5_0  67.55 MiB  4.2 ms  15.8 GiB/s
 *
 * The ordering holds at a tenth the rows, where the tensor is small enough to sit partly in
 * cache and every rate rises: Q4_0 16.3-16.5, Q8_0 15.5-17.8, Q4_K 9.0-12.1, Q6_K 11.0-11.9.
 * Pass a shape rather than trusting these — they are one SoC's answer, and the point of the
 * program is that the question is worth asking again elsewhere.
 *
 * Build: make bench_dtype
 * Run:   taskset -c 4-7 ./bench_dtype [rows] [cols] [passes]
 */
#define _GNU_SOURCE
#include "notorch.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1e3 + (double)t.tv_nsec*1e-6; }
static size_t rb_of(int d,int k){ switch(d){
    case GGUF_TYPE_F32:  return (size_t)k*4;        case GGUF_TYPE_F16:  return (size_t)k*2;
    case GGUF_TYPE_Q4_0: return (size_t)(k/32)*18;  case GGUF_TYPE_Q5_0: return (size_t)(k/32)*22;
    case GGUF_TYPE_Q8_0: return (size_t)(k/32)*34;  case GGUF_TYPE_Q4_K: return (size_t)(k/256)*144;
    case GGUF_TYPE_Q6_K: return (size_t)(k/256)*210; default: return 0; } }

/* The unpacked formats are not quantized, so they are written here rather than through
 * nt_quantize_row, and they do not go through the i8 entry either — that one is for the
 * packed dtypes and refuses these. Both differences are the reason they belong in this
 * table: whatever the unpacked path costs, no packed measurement reveals it. */
static int fill_plain(uint8_t *dst, const float *row, int k, int d) {
    if (d == GGUF_TYPE_F32) { memcpy(dst, row, (size_t)k*4); return 1; }
#if defined(__ARM_NEON) || defined(__FP16_VALUE__) || defined(__ARM_FP16_FORMAT_IEEE)
    if (d == GGUF_TYPE_F16) {
        __fp16 *h = (__fp16 *)dst;
        for (int i = 0; i < k; i++) h[i] = (__fp16)row[i];
        return 1;
    }
#endif
    return 0;
}
static int is_plain(int d){ return d == GGUF_TYPE_F32 || d == GGUF_TYPE_F16; }
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 50304;   /* OLMoE vocab */
    int k    = argc > 2 ? atoi(argv[2]) : 2048;    /* n_embd */
    int reps = argc > 3 ? atoi(argv[3]) : 30;
    int dts[] = { GGUF_TYPE_Q4_0, GGUF_TYPE_Q5_0, GGUF_TYPE_Q8_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
                  GGUF_TYPE_F16, GGUF_TYPE_F32 };
    const char *nm[] = { "Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q6_K", "F16", "F32" };
    float *x = malloc((size_t)k*sizeof(float)), *o = malloc((size_t)rows*sizeof(float));
    float *row = malloc((size_t)k*sizeof(float));
    for (int i=0;i<k;i++) x[i]=(float)((i%31)-15)*0.0625f;
    printf("head shape [%d, %d], %d passes\n", rows, k, reps);
    for (unsigned d = 0; d < sizeof(dts)/sizeof(dts[0]); d++) {
        size_t rb = rb_of(dts[d], k);
        if (!rb) continue;
        uint8_t *W = malloc(rb*(size_t)rows);
        if (!W) { printf("  %s: no memory\n", nm[d]); continue; }
        int ok = 1, plain = is_plain(dts[d]);
        for (long r = 0; r < rows && ok; r++) {
            for (int i=0;i<k;i++) row[i]=(float)(((r*7919+i*104729)%2001)-1000)/500.0f;
            if (plain) ok = fill_plain(W + rb*(size_t)r, row, k, dts[d]);
            else if (nt_quantize_row(row, W + rb*(size_t)r, k, dts[d]) != 0) ok = 0;
        }
        if (!ok) { printf("  %s: this build cannot pack it\n", nm[d]); free(W); continue; }
        for (int r=0;r<3;r++)
            if (plain) nt_qmatvec(o,W,dts[d],x,rows,k); else nt_qmatvec_i8(o,W,dts[d],x,rows,k);
        double t0=now();
        for (int r=0;r<reps;r++)
            if (plain) nt_qmatvec(o,W,dts[d],x,rows,k); else nt_qmatvec_i8(o,W,dts[d],x,rows,k);
        double dt=now()-t0;
        double mib = (double)rb*rows/(1024.0*1024.0);
        printf("  %-5s %7.2f MiB  %6.2f ms/pass  %5.2f GiB/s\n",
               nm[d], mib, dt/reps, mib*reps/(dt/1e3)/1024.0);
        free(W);
    }
    return 0;
}
