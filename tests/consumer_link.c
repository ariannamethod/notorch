/* consumer_link.c — the harness used the way a body uses it.
 *
 * Not a correctness test: harness/test_parity.sh and the architecture goldens
 * do that. This asserts one thing those cannot, because they all compile the
 * tree's own sources — that a program outside the tree can run a model by
 * linking two installed archives and including four installed headers, with no
 * notorch source on its command line.
 *
 * Yent's inference is being rebuilt on this harness, so the day that stops
 * being true is a day somebody discovers by hand. harness/test_consumer_link.sh
 * installs into a throwaway prefix and builds this against it.
 */
#include "harness/archs.h"
#include "examples/bpe.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf \"prompt\"\n", argv[0]); return 2; }

    /* Architecture selection is part of the installed surface. A body must
     * never discover support by executing the wrong family's arithmetic. */
    if (nt_pick_arch("llama") != &nt_arch_llama ||
        nt_pick_arch("qwen2") != &nt_arch_llama ||
        nt_pick_arch("notorch-red-hand-unknown") != NULL ||
        nt_pick_arch(NULL) != NULL) {
        fprintf(stderr, "architecture registry contract failed\n");
        return 1;
    }

    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;

    const nt_arch *arch = nt_pick_arch(gf->arch);
    if (!arch) { fprintf(stderr, "no family for '%s'\n", gf->arch); return 1; }

    nt_dims dims;
    void *model = arch->load(gf, &dims);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    bpe_tokenizer *tok = bpe_load(argv[1]);
    int ids[512], n = 0;
    if (tok) n = bpe_encode(tok, argv[2], ids, 512);
    else { ids[n++] = 1; for (int i = 0; argv[2][i] && n < 512; i++) ids[n++] = (unsigned char)argv[2][i]; }

    kv_cache *kv = kv_new(dims.n_layers, n + 8, dims.kv_dim);
    float *logits = (float *)malloc((size_t)dims.vocab * sizeof(float));
    if (!kv || !logits) return 1;

    arch->forward(model, kv, ids, n, 0, logits);

    int best = 0;
    for (int i = 1; i < dims.vocab; i++) if (logits[i] > logits[best]) best = i;

    printf("arch=%s tokens=%d vocab=%d argmax=%d\n", gf->arch, n, dims.vocab, best);
    return 0;
}
