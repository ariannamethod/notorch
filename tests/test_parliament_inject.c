/* test_parliament_inject.c — isolated gate for nt_metal_parliament_inject.
 * Builds a tiny registered expert arena (ne=4, D=512, rank=16), known x/gate,
 * runs the GPU inject and compares to the CPU reference (the doe parliament
 * inject math). The GPU reduction order differs from the CPU sequential sum,
 * so the bar is max|Δ| < 1e-3, not bit-identical. Also checks the all-gate-0
 * no-op (dead/unelected experts must contribute exactly nothing).
 *
 * build (neo / Apple Silicon):
 *   clang++ -DUSE_METAL -fobjc-arc -c notorch_metal.mm -o notorch_metal.o
 *   cc -DUSE_METAL -O2 tests/test_parliament_inject.c notorch_metal.o \
 *      -framework Metal -framework Foundation -lc++ -lm -o test_parliament_inject
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../notorch_metal.h"

#define NE   4
#define D    512
#define RANK 16

/* deterministic LoRA-scale weights so CPU and GPU read identical bytes */
static float bval(int e, int r, int j) { return 0.01f * sinf(0.013f*(float)(e*7919 + r*131 + j)); }
static float aval(int e, int i, int r) { return 0.01f * cosf(0.017f*(float)(e*5003 + i*97 + r)); }

int main(void) {
    if (!nt_metal_available()) { fprintf(stderr, "metal unavailable\n"); return 2; }

    size_t per_expert   = (size_t)2 * RANK * D;            /* B rank*D + A D*rank */
    size_t layer_floats = (size_t)NE * D + (size_t)NE * per_expert;
    size_t pg    = (size_t)getpagesize();
    size_t bytes = (layer_floats * sizeof(float) + pg - 1) / pg * pg;

    float *arena = NULL;
    if (posix_memalign((void**)&arena, pg, bytes) != 0 || !arena) { perror("posix_memalign"); return 2; }
    memset(arena, 0, bytes);

    /* layout: [ w_vote NE*D | per-expert (B rank*D, A D*rank) x NE ] */
    float *experts = arena + (size_t)NE * D;
    for (int e = 0; e < NE; e++) {
        float *Bb = experts + (size_t)e * per_expert;          /* [rank, D] */
        float *Ab = Bb + (size_t)RANK * D;                     /* [D, rank] */
        for (int r = 0; r < RANK; r++) for (int j = 0; j < D; j++) Bb[(size_t)r*D + j] = bval(e,r,j);
        for (int i = 0; i < D; i++) for (int r = 0; r < RANK; r++) Ab[(size_t)i*RANK + r] = aval(e,i,r);
    }

    float x[D], gate[NE] = { 0.5f, 0.0f, 0.3f, 0.2f };         /* expert 1 unelected */
    for (int j = 0; j < D; j++) x[j] = 0.1f * cosf(0.05f * (float)j);
    float alpha = 0.1f;

    /* CPU reference — exactly the doe parliament inject math */
    float xref[D]; memcpy(xref, x, sizeof(x));
    for (int e = 0; e < NE; e++) {
        if (gate[e] == 0.0f) continue;
        float *Bb = experts + (size_t)e * per_expert;
        float *Ab = Bb + (size_t)RANK * D;
        float tmp[RANK];
        for (int r = 0; r < RANK; r++) { float s = 0.0f; for (int j = 0; j < D; j++) s += Bb[(size_t)r*D+j]*x[j]; tmp[r] = s; }
        for (int i = 0; i < D; i++) {
            float s = 0.0f; for (int r = 0; r < RANK; r++) s += Ab[(size_t)i*RANK+r]*tmp[r];
            xref[i] += alpha * gate[e] * s;
        }
    }

    /* GPU path */
    if (nt_metal_register_region(arena, bytes) != 0) { fprintf(stderr, "register_region failed\n"); return 2; }
    enum { SLOT_X = 0, SLOT_TMP = 1, SLOT_GATE = 2 };
    if (nt_metal_slot_alloc(SLOT_X,   (uint64_t)D*4)        ||
        nt_metal_slot_alloc(SLOT_TMP, (uint64_t)NE*RANK*4)  ||
        nt_metal_slot_alloc(SLOT_GATE,(uint64_t)NE*4)) { fprintf(stderr, "slot_alloc failed\n"); return 2; }
    nt_metal_slot_upload(SLOT_X,    x,    (uint64_t)D*4);
    nt_metal_slot_upload(SLOT_GATE, gate, (uint64_t)NE*4);
    int rc = nt_metal_parliament_inject(SLOT_X, SLOT_TMP, SLOT_GATE, arena, D, RANK, NE, alpha);
    if (rc) { fprintf(stderr, "parliament_inject rc=%d\n", rc); return 2; }
    float xg[D]; nt_metal_slot_download(SLOT_X, xg, (uint64_t)D*4);

    double maxd = 0.0; int argi = 0;
    for (int i = 0; i < D; i++) { double d = fabs((double)xg[i] - (double)xref[i]); if (d > maxd) { maxd = d; argi = i; } }
    printf("[inject] max|x_gpu - x_cpu| = %.3e at i=%d (xg=%.6f xref=%.6f)\n", maxd, argi, xg[argi], xref[argi]);

    /* all-gate-0 no-op: dead/unelected experts contribute exactly nothing */
    float zgate[NE] = {0,0,0,0};
    nt_metal_slot_upload(SLOT_X,    x,     (uint64_t)D*4);
    nt_metal_slot_upload(SLOT_GATE, zgate, (uint64_t)NE*4);
    rc = nt_metal_parliament_inject(SLOT_X, SLOT_TMP, SLOT_GATE, arena, D, RANK, NE, alpha);
    if (rc) { fprintf(stderr, "parliament_inject(zero) rc=%d\n", rc); return 2; }
    float xz[D]; nt_metal_slot_download(SLOT_X, xz, (uint64_t)D*4);
    double maxz = 0.0; for (int i = 0; i < D; i++) { double d = fabs((double)xz[i]-(double)x[i]); if (d > maxz) maxz = d; }
    printf("[inject] all-gate-0 max|x_out - x_in| = %.3e (must be 0)\n", maxz);

    int pass = (maxd < 1e-3) && (maxz == 0.0);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
