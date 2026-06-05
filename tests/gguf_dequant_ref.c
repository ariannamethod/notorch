/* gguf_dequant_ref.c — reference dequant dump for the JS-vs-C parity test.
 * Dequantizes named tensors via the C path (gguf.c) and prints the first 64
 * values of each as JSON, so js-edition/test_gguf_dequant.mjs can verify the
 * JS block-dequant matches byte-for-byte.
 *
 * Build: cc -O2 -I. tests/gguf_dequant_ref.c gguf.c -lm -o /tmp/gguf_dequant_ref
 * Run:   /tmp/gguf_dequant_ref model.gguf tensor1 [tensor2 ...]
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tensor1 [tensor2 ...]\n", argv[0]); return 1; }
    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) { fprintf(stderr, "open failed: %s\n", argv[1]); return 1; }
    printf("{\n");
    int first = 1;
    for (int a = 2; a < argc; a++) {
        int ti = gguf_find_tensor(gf, argv[a]);
        if (ti < 0) { fprintf(stderr, "missing tensor: %s\n", argv[a]); continue; }
        float *d = gguf_dequant(gf, ti);
        if (!d) { fprintf(stderr, "dequant failed: %s\n", argv[a]); continue; }
        long n = (long)gf->tensors[ti].n_elements;
        long m = n < 64 ? n : 64;
        printf("%s  \"%s\": {\"n\": %ld, \"dtype\": %u, \"vals\": [",
               first ? "" : ",\n", argv[a], n, gf->tensors[ti].dtype);
        for (long i = 0; i < m; i++) printf("%s%.8g", i ? "," : "", d[i]);
        printf("]}");
        first = 0;
        free(d);
    }
    printf("\n}\n");
    gguf_close(gf);
    return 0;
}
