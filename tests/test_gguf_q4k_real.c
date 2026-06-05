/* test_gguf_q4k_real.c — verify nt_metal_q4k_matvec against gguf_dequant + CPU mm
 * on a REAL Q4_K tensor from a GGUF file (not synthetic). Foundation for the
 * notorch-C packed-Q4_K inference forward.
 *
 * Build: cc -O2 -DUSE_METAL -I. tests/test_gguf_q4k_real.c gguf.c notorch_metal.o \
 *          -framework Metal -framework Foundation -lc++ -lm -o tests/test_gguf_q4k_real
 * Run:   ./tests/test_gguf_q4k_real <model.gguf>
 */
#include "gguf.h"
#include "notorch_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <q4k.gguf>\n", argv[0]); return 1; }
    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) { printf("gguf_open failed\n"); return 1; }

    /* GGUF weight: ne[0]=in (k, contiguous), ne[1]=out (m). matvec wants
       W = m rows x (k/256) blocks, k multiple of 256. */
    int idx = -1; long m = 0, k = 0;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (gf->tensors[i].dtype != GGUF_TYPE_Q4_K || gf->tensors[i].ndim != 2) continue;
        long s0 = (long)gf->tensors[i].shape[0];
        long s1 = (long)gf->tensors[i].shape[1];
        if (s0 % 256 != 0) continue;
        idx = (int)i; k = s0; m = s1; break;
    }
    if (idx < 0) { printf("no 2D Q4_K tensor with k%%256==0\n"); gguf_close(gf); return 1; }
    printf("tensor[%d] %s  m(out)=%ld k(in)=%ld dtype=Q4_K\n", idx, gf->tensors[idx].name, m, k);

    const uint8_t *W = gf->data + gf->tensors[idx].offset;
    float *x = (float*)malloc(k * sizeof(float));
    for (long j = 0; j < k; j++) x[j] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    float *out_metal = (float*)malloc(m * sizeof(float));

    float *deq = gguf_dequant(gf, idx);  /* f32 [m*k], row-major out x in */
    if (!deq) { printf("gguf_dequant failed\n"); return 1; }

    int rc = nt_metal_q4k_matvec(out_metal, W, x, (int)m, (int)k);
    if (rc != 0) { printf("nt_metal_q4k_matvec rc=%d\n", rc); return 1; }

    double maxabs = 0, maxrel = 0;
    for (long i = 0; i < m; i++) {
        double ref = 0;
        for (long j = 0; j < k; j++) ref += (double)deq[i*k + j] * (double)x[j];
        double d = fabs((double)out_metal[i] - ref);
        if (d > maxabs) maxabs = d;
        double r = d / (fabs(ref) + 1e-6);
        if (r > maxrel) maxrel = r;
    }
    printf("packed-matvec vs gguf_dequant+CPU-mm: max_abs=%.3e max_rel=%.3e\n", maxabs, maxrel);
    int ok = maxrel < 5e-3;
    printf("%s\n", ok ? "REAL_Q4K_MATVEC_OK" : "MISMATCH");

    free(x); free(out_metal); free(deq); gguf_close(gf);
    return ok ? 0 : 1;
}
