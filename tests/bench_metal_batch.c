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

int main(void)
{
    if (!nt_metal_available()) { fprintf(stderr, "no Metal device, skipping\n"); return 0; }
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
