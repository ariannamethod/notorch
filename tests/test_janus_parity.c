/* test_janus_parity.c — the ported Janus forward, in logits.
 *
 * The harness prints text, and text is a poor instrument for a port: an argmax
 * hides how far apart two vectors are and says nothing about where they parted.
 * This calls the architecture through its own table entry — the same entry
 * `notorch` uses — and writes the last position's logits, one per line, in the
 * same %.9g form the standalone reference writes them.
 *
 *   cc -O2 -I. -o test_janus_parity tests/test_janus_parity.c \
 *      harness/arch_janus.c harness/runtime.c gguf.c notorch.c -lm
 *   ./test_janus_parity model.gguf 487 1955 306 > port.txt
 *   diff <(...) port.txt     # or the max-abs comparison in the harness gate
 */
#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <id> [id ...]\n", argv[0]);
        return 1;
    }
    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;

    nt_dims dims = {0};
    void *model = nt_arch_janus.load(gf, &dims);
    if (!model) { gguf_close(gf); return 1; }

    int n = argc - 2;
    int *ids = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        ids[i] = atoi(argv[2 + i]);
        if (ids[i] < 0 || ids[i] >= dims.vocab) {
            fprintf(stderr, "id %d out of vocab %d\n", ids[i], dims.vocab);
            return 1;
        }
    }

    /* One position per call, the way decode drives it, so this exercises the
     * same path a generation does rather than a prefill-only shape. */
    kv_cache *kv = kv_new(dims.n_layers, n + 1, dims.kv_dim);
    float *logits = (float*)calloc(dims.vocab, sizeof(float));
    for (int i = 0; i < n; i++) {
        int rc = nt_arch_janus.forward(model, kv, &ids[i], 1, i, logits);
        if (rc != NT_OK) { fprintf(stderr, "forward refused: %s\n", nt_strerror(rc)); return 1; }
    }

    for (int i = 0; i < dims.vocab; i++) printf("%.9g\n", logits[i]);

    int best = 0;
    for (int i = 1; i < dims.vocab; i++) if (logits[i] > logits[best]) best = i;
    fprintf(stderr, "port: argmax=%d logit=%.9g\n", best, logits[best]);

    free(logits); free(ids);
    kv_free(kv);
    nt_arch_janus.free(model);
    gguf_close(gf);
    return 0;
}
