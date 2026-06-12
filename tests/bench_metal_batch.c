/* tests/bench_metal_batch.c — dispatch-overhead microbench: solo matvecs
 * (one command buffer + wait each) vs token-graph batches. Models the doe
 * per-token sweep: L "layers" x 7 matvecs over resident Q4_K weights.
 * Numbers isolate CPU-GPU sync cost — the same kernels run either way.
 *
 * by Claude (Arianna Method)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#ifndef USE_METAL
int main(void) { fprintf(stderr, "bench_metal_batch: built without -DUSE_METAL, skipping\n"); return 0; }
#else
#include "notorch_metal.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ── doe-mix mode (`bench_metal_batch doe`): per-shape time distribution of
 * the oyent-24B decode sweep on the DEFAULT kernels (mirrors doe: D=5120,
 * heads 32x128, kv 8x128, ffn 32768, vocab 131072; Q4_K_M rides attn_v /
 * ffn_down / lm_head as Q6_K). Weight copies are cycled per layer so every
 * matvec streams from DRAM — small attn matrices get 8 copies so they cannot
 * sit in the SLC, big ffn ones get 2 (one already exceeds any cache). Each
 * group is timed around its own batch_begin/commit, so group numbers carry
 * the isolation-sync cost; the full-speed sweep (1 batch/layer) is the
 * honest total. The question this mode answers: does decode time follow
 * bytes? */
static int doe_mix(void)
{
    if (nt_metal_init() != 0) return 1;

    enum { D = 5120, QO = 4096, KVD = 1024, FF = 32768, VOC = 131072 };
    enum { LAYERS = 40, TOKENS = 4, NGRP = 5, MAXCP = 8 };

    struct shape { const char *grp; int gi; int m, k, q6, per_layer; };
    static const struct shape S[] = {
        { "attn qkv",    0, QO,  D,  0, 1 },
        { "attn qkv",    0, KVD, D,  0, 1 },
        { "attn qkv",    0, KVD, D,  1, 1 },   /* v rides Q6_K */
        { "attn o",      1, D,   QO, 0, 1 },
        { "ffn gate+up", 2, FF,  D,  0, 1 },
        { "ffn gate+up", 2, FF,  D,  0, 1 },
        { "ffn down",    3, D,   FF, 1, 1 },
        { "lm_head",     4, VOC, D,  1, 0 },   /* once per token */
    };
    enum { NS = sizeof(S) / sizeof(S[0]) };

    uint64_t offs[NS][MAXCP], wB[NS], total = 0;
    int cps[NS], bsz[NS];
    for (int s = 0; s < (int)NS; s++) {
        uint64_t row = (uint64_t)(S[s].k / 256) * (S[s].q6 ? 210 : 144);
        wB[s]  = (uint64_t)S[s].m * row;
        bsz[s] = S[s].q6 ? 210 : 144;
        cps[s] = !S[s].per_layer ? 1 : (wB[s] < ((uint64_t)32 << 20) ? MAXCP : 2);
        for (int c = 0; c < cps[s]; c++) { offs[s][c] = total; total += wB[s]; }
        for (int c = cps[s]; c < MAXCP; c++) offs[s][c] = offs[s][c % cps[s]];
    }
    total = (total + 16383) & ~(uint64_t)16383;

    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                                    MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    unsigned rng = 0xD0E;
    uint32_t *w32 = (uint32_t *)(void *)base;
    for (uint64_t i = 0; i < total / 4; i++) w32[i] = (uint32_t)rand_r(&rng) * 2654435761u;
    /* pin every block scale small-positive so dequant stays finite */
    for (int s = 0; s < (int)NS; s++)
        for (int c = 0; c < cps[s]; c++)
            for (uint64_t b = 0; b < wB[s]; b += (uint64_t)bsz[s]) {
                uint8_t *blk = base + offs[s][c] + b;
                if (S[s].q6) { blk[208] = 0x00; blk[209] = 0x1C; }
                else { blk[0] = 0x00; blk[1] = 0x30; blk[2] = 0x00; blk[3] = 0x2C; }
            }
    if (nt_metal_register_base(base, total) != 0) { fprintf(stderr, "register_base failed\n"); return 1; }

    float *xD  = (float *)malloc((size_t)D  * sizeof(float));
    float *xQO = (float *)malloc((size_t)QO * sizeof(float));
    float *xFF = (float *)malloc((size_t)FF * sizeof(float));
    float *out = (float *)malloc((size_t)VOC * sizeof(float));
    for (int j = 0; j < D;  j++) xD[j]  = ((float)rand_r(&rng) / (float)RAND_MAX) - 0.5f;
    for (int j = 0; j < QO; j++) xQO[j] = ((float)rand_r(&rng) / (float)RAND_MAX) - 0.5f;
    for (int j = 0; j < FF; j++) xFF[j] = ((float)rand_r(&rng) / (float)RAND_MAX) - 0.5f;

    uint64_t g_b[NGRP] = {0}; int g_d[NGRP] = {0};
    for (int s = 0; s < (int)NS; s++) {
        int times = S[s].per_layer ? LAYERS : 1;
        g_b[S[s].gi] += wB[s] * (uint64_t)times;
        g_d[S[s].gi] += times;
    }

#define DM_ENC(s, c, mo) do { \
        const float *xv = S[s].k == D ? xD : S[s].k == QO ? xQO : xFF; \
        if (S[s].q6) nt_metal_q6k_matvec(out + (mo), base + offs[s][c], xv, S[s].m, S[s].k); \
        else         nt_metal_q4k_matvec(out + (mo), base + offs[s][c], xv, S[s].m, S[s].k); \
    } while (0)

    /* warmup: one untimed full-speed token (faults the region in) */
    for (int l = 0; l < LAYERS; l++) {
        nt_metal_batch_begin();
        uint64_t mo = 0;
        for (int s = 0; s < (int)NS; s++)
            if (S[s].per_layer) { DM_ENC(s, l % cps[s], mo); mo += (uint64_t)S[s].m; }
        nt_metal_batch_commit();
    }
    nt_metal_q6k_matvec(out, base + offs[NS-1][0], xD, VOC, D);

    /* grouped pass: each group behind its own sync */
    double g_s[NGRP] = {0}, tok_ms[TOKENS];
    for (int t = 0; t < TOKENS; t++) {
        double tk0 = now_s();
        for (int l = 0; l < LAYERS; l++) {
            int s = 0;
            while (s < (int)NS && S[s].per_layer) {
                int gi = S[s].gi;
                double t0 = now_s();
                nt_metal_batch_begin();
                uint64_t mo = 0;
                while (s < (int)NS && S[s].per_layer && S[s].gi == gi) {
                    DM_ENC(s, l % cps[s], mo); mo += (uint64_t)S[s].m; s++;
                }
                nt_metal_batch_commit();
                g_s[gi] += now_s() - t0;
            }
        }
        double t0 = now_s();
        nt_metal_q6k_matvec(out, base + offs[NS-1][0], xD, VOC, D);
        g_s[NGRP-1] += now_s() - t0;
        tok_ms[t] = (now_s() - tk0) * 1e3;
    }

    /* full-speed pass: one batch per layer */
    double full_ms[TOKENS];
    for (int t = 0; t < TOKENS; t++) {
        double tk0 = now_s();
        for (int l = 0; l < LAYERS; l++) {
            nt_metal_batch_begin();
            uint64_t mo = 0;
            for (int s = 0; s < (int)NS; s++)
                if (S[s].per_layer) { DM_ENC(s, l % cps[s], mo); mo += (uint64_t)S[s].m; }
            nt_metal_batch_commit();
        }
        nt_metal_q6k_matvec(out, base + offs[NS-1][0], xD, VOC, D);
        full_ms[t] = (now_s() - tk0) * 1e3;
    }
#undef DM_ENC

    uint64_t tot_b = 0; double tot_s = 0;
    for (int g = 0; g < NGRP; g++) { tot_b += g_b[g]; tot_s += g_s[g]; }
    static const char *gname[NGRP] = { "attn qkv", "attn o", "ffn gate+up", "ffn down", "lm_head" };
    printf("bench doe-mix: %d layers x {qkv, o, gate+up, down} + lm_head, %d tokens, %.2f GB resident\n",
           LAYERS, TOKENS, (double)total / 1e9);
    printf("  %-12s %8s %8s %9s %8s %7s\n", "group", "disp/tok", "GB/tok", "ms/tok", "GB/s", "share");
    for (int g = 0; g < NGRP; g++) {
        double ms = g_s[g] / TOKENS * 1e3;
        printf("  %-12s %8d %8.2f %9.2f %8.1f %6.1f%%\n", gname[g], g_d[g],
               (double)g_b[g] / 1e9, ms, (double)g_b[g] / 1e9 / (g_s[g] / TOKENS),
               100.0 * g_s[g] / tot_s);
    }
    double gmin = tok_ms[0], gmax = tok_ms[0], fav = 0, fmin = full_ms[0], fmax = full_ms[0];
    for (int t = 0; t < TOKENS; t++) {
        if (tok_ms[t] < gmin) gmin = tok_ms[t];
        if (tok_ms[t] > gmax) gmax = tok_ms[t];
        fav += full_ms[t];
        if (full_ms[t] < fmin) fmin = full_ms[t];
        if (full_ms[t] > fmax) fmax = full_ms[t];
    }
    fav /= TOKENS;
    printf("  grouped total: %.2f GB/tok, %.1f ms/tok [%.1f..%.1f] — includes %d isolation syncs\n",
           (double)tot_b / 1e9, tot_s / TOKENS * 1e3, gmin, gmax, LAYERS * (NGRP - 1) + 1);
    printf("  full-speed (1 batch/layer + lm): %.1f ms/tok [%.1f..%.1f] = %.1f GB/s effective\n",
           fav, fmin, fmax, (double)tot_b / 1e9 / (fav / 1e3));

    free(xD); free(xQO); free(xFF); free(out);
    munmap(base, (size_t)total);
    nt_metal_shutdown();
    return 0;
}

int main(int argc, char **argv)
{
    if (!nt_metal_available()) { fprintf(stderr, "no Metal device, skipping\n"); return 0; }
    if (argc > 1 && strcmp(argv[1], "doe") == 0) return doe_mix();
    setenv("NT_METAL_SG", "1", 1);   /* phases A-C bench the sg kernels; D = naive (library default) */
    if (nt_metal_init() != 0) return 1;

    const int M = 2048, K = 2048;            /* small-model-layer sized */
    const int NW = 7;                        /* q,k,v,o,gate,up,down    */
    const int LAYERS = 40, REPS = 3;
    const uint64_t row_bytes = (uint64_t)(K / 256) * 144;
    const uint64_t W_bytes   = (uint64_t)M * row_bytes;
    const uint64_t total     = W_bytes * NW;

    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                                    MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    unsigned rng = 0xBEEF;
    for (uint64_t i = 0; i < total; i++) base[i] = (uint8_t)(rand_r(&rng) & 0xFF);
    /* make every fp16 block scale small-positive so dequant stays finite */
    for (int w = 0; w < NW; w++)
        for (uint64_t b = 0; b < W_bytes; b += 144) {
            uint8_t *blk = base + (uint64_t)w * W_bytes + b;
            uint16_t d_bits    = (uint16_t)(0x3000 + (rand_r(&rng) & 0x0FFF));
            uint16_t dmin_bits = (uint16_t)(0x2C00 + (rand_r(&rng) & 0x07FF));
            blk[0] = (uint8_t)(d_bits & 0xFF);    blk[1] = (uint8_t)(d_bits >> 8);
            blk[2] = (uint8_t)(dmin_bits & 0xFF); blk[3] = (uint8_t)(dmin_bits >> 8);
        }
    if (nt_metal_register_base(base, total) != 0) { fprintf(stderr, "register_base failed\n"); return 1; }

    float *x   = (float *)malloc((size_t)K * sizeof(float));
    float *out = (float *)malloc((size_t)M * NW * sizeof(float));
    for (int j = 0; j < K; j++) x[j] = ((float)rand_r(&rng) / (float)RAND_MAX) - 0.5f;

    /* warmup */
    nt_metal_q4k_matvec(out, base, x, M, K);

    /* A: solo — 1 sync per matvec (the current doe pattern) */
    double t0 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int l = 0; l < LAYERS; l++)
            for (int w = 0; w < NW; w++)
                nt_metal_q4k_matvec(out + (size_t)w * M, base + (uint64_t)w * W_bytes, x, M, K);
    double tA = (now_s() - t0) / REPS;

    /* B: grouped — {q,k,v} and {gate,up} share a sync (4 syncs / layer) */
    t0 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int l = 0; l < LAYERS; l++) {
            nt_metal_batch_begin();
            for (int w = 0; w < 3; w++)
                nt_metal_q4k_matvec(out + (size_t)w * M, base + (uint64_t)w * W_bytes, x, M, K);
            nt_metal_batch_commit();
            nt_metal_q4k_matvec(out + 3 * (size_t)M, base + 3 * W_bytes, x, M, K);
            nt_metal_batch_begin();
            for (int w = 4; w < 6; w++)
                nt_metal_q4k_matvec(out + (size_t)w * M, base + (uint64_t)w * W_bytes, x, M, K);
            nt_metal_batch_commit();
            nt_metal_q4k_matvec(out + 6 * (size_t)M, base + 6 * W_bytes, x, M, K);
        }
    double tB = (now_s() - t0) / REPS;

    /* C: one batch per layer — 1 sync / layer (the M4 target shape) */
    t0 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int l = 0; l < LAYERS; l++) {
            nt_metal_batch_begin();
            for (int w = 0; w < NW; w++)
                nt_metal_q4k_matvec(out + (size_t)w * M, base + (uint64_t)w * W_bytes, x, M, K);
            nt_metal_batch_commit();
        }
    double tC = (now_s() - t0) / REPS;

    int n_mv = LAYERS * NW;
    printf("bench_metal_batch: M=%d K=%d, %d matvecs/token-sweep (resident W)\n", M, K, n_mv);
    printf("  A solo   (%3d syncs): %7.2f ms/sweep  %6.1f us/matvec\n", n_mv,        tA * 1e3, tA * 1e6 / n_mv);
    printf("  B groups (%3d syncs): %7.2f ms/sweep  %6.1f us/matvec  x%.2f\n", LAYERS * 4, tB * 1e3, tB * 1e6 / n_mv, tA / tB);
    printf("  C /layer (%3d syncs): %7.2f ms/sweep  %6.1f us/matvec  x%.2f\n", LAYERS,     tC * 1e3, tC * 1e6 / n_mv, tA / tC);


    /* D: naive kernels (library default), solo — isolates the simdgroup
     * kernel multiplier */
    nt_metal_shutdown();
    unsetenv("NT_METAL_SG");
    if (nt_metal_init() != 0 || nt_metal_register_base(base, total) != 0) {
        fprintf(stderr, "naive re-init failed\n"); return 1;
    }
    nt_metal_q4k_matvec(out, base, x, M, K);   /* warmup */
    t0 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int l = 0; l < LAYERS; l++)
            for (int w = 0; w < NW; w++)
                nt_metal_q4k_matvec(out + (size_t)w * M, base + (uint64_t)w * W_bytes, x, M, K);
    double tD = (now_s() - t0) / REPS;
    printf("  D naive  (%3d syncs): %7.2f ms/sweep  %6.1f us/matvec  (sg kernel x%.2f vs naive)\n",
           n_mv, tD * 1e3, tD * 1e6 / n_mv, tD / tA);
    free(x); free(out);
    munmap(base, (size_t)total);
    nt_metal_shutdown();
    return 0;
}
#endif
