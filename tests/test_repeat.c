/* test_repeat.c — the same sequence, twice, on one model.
 *
 * A family may keep state the KV cache does not carry: Mamba's convolution and
 * SSM state, Janus's running low-rank sum. That state belongs to the model
 * object, so it survives a new kv_cache, and the only thing that can clear it
 * is the family itself when it sees pos0 = 0.
 *
 * When one forgets, nothing visible breaks. Janus drifted 0.49 on the logits
 * between two identical runs and kept the same argmax, so every gate in the
 * tree stayed green while a body serving two turns answered the second one
 * from a model that had eaten the first.
 *
 * The invariant is exact equality, not a tolerance: the same arithmetic on the
 * same inputs has no reason to move at all, and a tolerance here would be a
 * place for the next drift to hide.
 *
 *   test_repeat model.gguf
 */
#include "harness/archs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }

    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;

    const nt_arch *arch = nt_pick_arch(gf->arch);
    if (!arch) { fprintf(stderr, "no family claims '%s'\n", gf->arch); return 3; }

    nt_dims dims;
    void *model = arch->load(gf, &dims);
    if (!model) { fprintf(stderr, "load failed\n"); return 3; }

    /* Bytes, so this runs on a file with any vocabulary or none. */
    static const int ids[] = { 84, 104, 101, 32, 102, 105, 101, 108, 100 };
    int n = (int)(sizeof ids / sizeof ids[0]);
    for (int i = 0; i < n; i++)
        if (ids[i] >= dims.vocab) { fprintf(stderr, "vocab %d too small\n", dims.vocab); return 3; }

    float *a = (float *)calloc((size_t)dims.vocab, sizeof(float));
    float *b = (float *)calloc((size_t)dims.vocab, sizeof(float));
    float *c = (float *)calloc((size_t)dims.vocab, sizeof(float));
    if (!a || !b || !c) return 1;

    /* Two runs share a cache, the third gets a new one — a body may or may not
     * reuse the cache between turns, and the model's own state must not care. */
    kv_cache *kv = kv_new(dims.n_layers, n + 8, dims.kv_dim);
    if (!kv) return 1;
    int rc = arch->forward(model, kv, ids, n, 0, a);
    if (rc == NT_OK) rc = arch->forward(model, kv, ids, n, 0, b);
    if (rc != NT_OK) { fprintf(stderr, "forward refused: %s\n", nt_strerror(rc)); return 3; }
    kv_free(kv);

    kv_cache *kv2 = kv_new(dims.n_layers, n + 8, dims.kv_dim);
    if (!kv2) return 1;
    rc = arch->forward(model, kv2, ids, n, 0, c);
    if (rc != NT_OK) { fprintf(stderr, "forward refused: %s\n", nt_strerror(rc)); return 3; }
    kv_free(kv2);

    double w_same = 0, w_new = 0;
    int at_same = -1, at_new = -1;
    for (int i = 0; i < dims.vocab; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > w_same) { w_same = d; at_same = i; }
        double e = fabs((double)a[i] - (double)c[i]);
        if (e > w_new) { w_new = e; at_new = i; }
    }

    int fail = (w_same != 0.0) || (w_new != 0.0);
    printf("%s  same-cache %.9g (id %d)  new-cache %.9g (id %d)  %s\n",
           gf->arch, w_same, at_same, w_new, at_new, fail ? "FAIL" : "PASS");

    arch->free(model);
    gguf_close(gf);
    return fail ? 1 : 0;
}
