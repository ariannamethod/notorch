// test_gguf.c — quick GGUF parser test
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : NULL;
    if (!path) {
        printf("usage: %s <file.gguf>\n", argv[0]);
        return 1;
    }

    gguf_file* gf = gguf_open(path);
    if (!gf) return 1;

    gguf_print_info(gf);

    // Test dequantization of first tensor
    if (gf->n_tensors > 0) {
        printf("\nDequantizing tensor 0 (%s)...\n", gf->tensors[0].name);
        float* data = gguf_dequant(gf, 0);
        if (data) {
            int n = gf->tensors[0].n_elements;
            float sum = 0, min = 1e30f, max = -1e30f;
            for (int i = 0; i < n; i++) {
                sum += data[i];
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }
            printf("  elements=%d mean=%.6f min=%.4f max=%.4f\n",
                   n, sum / n, min, max);
            free(data);
        }
    }

    gguf_close(gf);
    return 0;
}
