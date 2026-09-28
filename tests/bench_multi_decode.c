/* bench_multi_decode.c — decode B sequences one at a time, then together, on a real GGUF.
 *
 * Usage: bench_multi_decode MODEL.gguf B STEPS
 * Each sequence starts from its own short prompt (fixed token ids, prefilled alone), then
 * both runs feed the same greedy tokens: the single run's choices. Prints the wall time of
 * the STEPS decode steps each way and whether every row's logits matched bit for bit. The
 * numerical mode is the environment's (NT_NO_I8 or not), as for any consumer. Built as
 * bench_cuda_decode (-DNT_CUDA) it runs the family's CUDA decoder, with device caches. */
#include "harness/archs.h"
#ifdef NT_CUDA
#include "harness/cuda_decode.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static int argmax(const float *x, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: bench_multi_decode MODEL.gguf B STEPS\n"); return 1; }
    int B = atoi(argv[2]), S = atoi(argv[3]);
    if (B < 1 || B > 64 || S < 1) return 1;
    gguf_file *gf = gguf_open(argv[1]); if (!gf) return 2;
    const nt_arch *arch = nt_pick_arch(gf->arch);
#ifdef NT_CUDA
    if (arch) arch = nt_cuda_arch_for(arch);
#endif
    if (!arch || !arch->forward_multi) { fprintf(stderr, "no multi-sequence decode for %s\n", gf->arch); return 3; }
    nt_dims d; void *model = arch->load(gf, &d); if (!model) return 4;
    int V = d.vocab, P = 8, cap = P + S + 1;
    float *single = malloc((size_t)B * (S + 1) * V * sizeof(float));
    float *multi = malloc((size_t)B * V * sizeof(float));
    int *chosen = malloc((size_t)B * S * sizeof(int)), *tokens = malloc((size_t)B * sizeof(int));
    int *pos = malloc((size_t)B * sizeof(int));
    kv_cache **kv = malloc((size_t)B * sizeof(kv_cache*)), **kvm = malloc((size_t)B * sizeof(kv_cache*));
    if (!single || !multi || !chosen || !tokens || !pos || !kv || !kvm) return 5;
    int prompt[8];
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < P; i++) prompt[i] = 100 + ((b * 131 + i * 17) % 5000);
        kv[b] = arch->kv_new ? arch->kv_new(model, cap) : kv_new(d.n_layers, cap, d.kv_dim);
        kvm[b] = arch->kv_new ? arch->kv_new(model, cap) : kv_new(d.n_layers, cap, d.kv_dim);
        if (!kv[b] || !kvm[b]) return 6;
        if (arch->forward(model, kv[b], prompt, P, 0, single + (size_t)b * (S + 1) * V)) return 7;
        if (arch->forward(model, kvm[b], prompt, P, 0, multi + (size_t)b * V)) return 7;
    }
    double t0 = now();
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++) {
            float *cur = single + ((size_t)b * (S + 1) + s) * V;
            chosen[b * S + s] = argmax(cur, V);
            if (arch->forward(model, kv[b], &chosen[b * S + s], 1, P + s, cur + V)) return 8;
        }
    double t1 = now();
    int same = 1;
    for (int s = 0; s < S; s++) {
        for (int b = 0; b < B; b++) { tokens[b] = chosen[b * S + s]; pos[b] = P + s; }
        if (arch->forward_multi(model, kvm, tokens, pos, B, multi, NULL, NULL)) return 9;
        for (int b = 0; b < B; b++)
            same &= !memcmp(multi + (size_t)b * V, single + ((size_t)b * (S + 1) + s + 1) * V, (size_t)V * sizeof(float));
    }
    double t2 = now();
    printf("{\"model\":\"%s\",\"B\":%d,\"steps\":%d,\"single_s\":%.3f,\"multi_s\":%.3f,"
           "\"single_tok_s\":%.2f,\"multi_tok_s\":%.2f,\"identical\":%s,\"nt_no_i8\":%d,\"device\":\"%s\"}\n",
           argv[1], B, S, t1 - t0, t2 - t1, B * S / (t1 - t0), B * S / (t2 - t1),
           same ? "true" : "false", getenv("NT_NO_I8") != NULL, arch->kv_new ? "cuda" : "cpu");
    for (int b = 0; b < B; b++) {
        if (arch->kv_free) { arch->kv_free(model, kv[b]); arch->kv_free(model, kvm[b]); }
        else { kv_free(kv[b]); kv_free(kvm[b]); }
    }
    arch->free(model); gguf_close(gf);
    return same ? 0 : 10;
}
