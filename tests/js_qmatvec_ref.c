/* js_qmatvec_ref.c — run nt_qmatvec on bytes handed over by the JS test, so the
 * two runtimes are compared on ONE input rather than on two generators that are
 * only believed to agree. Companion to js-edition/test_qmatvec.mjs (--cref),
 * same role tests/gguf_dequant_ref.c plays for the block dequant.
 *
 * in:  int32 dtype, int32 m, int32 k, int32 wbytes, uint8 W[wbytes], float x[k]
 * out: float out[m]
 *
 * With --i8 the same input goes through nt_qmatvec_i8 instead, so the JS int8
 * path is checked against the C int8 path rather than against the exact one —
 * two approximations of the same shape, not an approximation against a truth.
 *
 * Build: cc -std=c11 -O2 -I. tests/js_qmatvec_ref.c notorch.c -lm -o /tmp/js_qmatvec_ref
 * Run:   node js-edition/test_qmatvec.mjs --cref /tmp/js_qmatvec_ref
 */
#include "notorch.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) { fprintf(stderr, "usage: %s in.bin out.bin [--i8]\n", argv[0]); return 2; }
    int i8 = (argc == 4 && strcmp(argv[3], "--i8") == 0);
    if (argc == 4 && !i8) { fprintf(stderr, "unknown flag: %s\n", argv[3]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", argv[1]); return 1; }

    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, f) != 4) { fclose(f); fprintf(stderr, "short header\n"); return 1; }
    int32_t dtype = hdr[0], m = hdr[1], k = hdr[2], wbytes = hdr[3];
    if (m <= 0 || k <= 0 || wbytes <= 0) { fclose(f); fprintf(stderr, "bad shape\n"); return 1; }

    uint8_t *W = malloc((size_t)wbytes);
    float *x = malloc((size_t)k * sizeof(float));
    float *out = calloc((size_t)m, sizeof(float));
    if (!W || !x || !out) { fclose(f); fprintf(stderr, "oom\n"); return 1; }
    if (fread(W, 1, (size_t)wbytes, f) != (size_t)wbytes ||
        fread(x, sizeof(float), (size_t)k, f) != (size_t)k) {
        fclose(f); fprintf(stderr, "short body\n"); return 1;
    }
    fclose(f);

    int rc = i8 ? nt_qmatvec_i8(out, W, dtype, x, m, k)
                : nt_qmatvec(out, W, dtype, x, m, k);
    if (rc != 0) {
        fprintf(stderr, "nt_qmatvec%s rc=%d (dtype=%d k=%d)\n", i8 ? "_i8" : "", rc, dtype, k);
        return 1;
    }

    FILE *g = fopen(argv[2], "wb");
    if (!g) { fprintf(stderr, "open failed: %s\n", argv[2]); return 1; }
    if (fwrite(out, sizeof(float), (size_t)m, g) != (size_t)m) {
        fclose(g); fprintf(stderr, "short write\n"); return 1;
    }
    fclose(g);
    free(W); free(x); free(out);
    return 0;
}
