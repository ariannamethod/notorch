/* test_qmatvec_leak.c — the packed matvec must return every byte it took.
 *
 * nt_qmatvec_i8 allocated three buffers per call and freed two of them on every exit path
 * but one, so the hottest function in the library leaked its block-sum scratch on every
 * invocation: about 1.4 MB per second of decode on a 0.5B model, unbounded in a server.
 * Reading the code says it; this says it in bytes, and goes red if the free is removed.
 *
 * The gate is glibc's mallinfo2, not RSS. RSS is a measure of what the allocator has asked
 * the kernel for and it moves for reasons that have nothing to do with the caller, while
 * uordblks is exactly the sum of live allocations. Where mallinfo2 is unavailable the test
 * says so and passes, because a gate that cannot measure must not pretend to.
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GLIBC__)
#include <malloc.h>
#define HAVE_MALLINFO2 (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
#else
#define HAVE_MALLINFO2 0
#endif

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

int main(void) {
    printf("packed matvec — allocation balance\n");

    /* Q4_0: 18 bytes per block of 32. One row of k weights, m rows. Shapes are the ones a
     * decode actually asks for, so the threaded path is exercised rather than only the
     * single-threaded shortcut. */
    const int k = 1536, m = 512;
    size_t rowsz = (size_t)(k / 32) * 18;
    uint8_t *W = (uint8_t *)malloc(rowsz * (size_t)m);
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *out = (float *)malloc((size_t)m * sizeof(float));
    if (!W || !x || !out) { printf("  FAIL out of memory\n"); return 1; }
    for (size_t i = 0; i < rowsz * (size_t)m; i++) W[i] = (uint8_t)(i * 31u + 7u);
    for (int i = 0; i < k; i++) x[i] = (float)((i % 17) - 8) * 0.125f;

    /* A NULL weight must come back as -1 rather than as a signal. This is the contract the
     * callers test against, and it did not hold until the guard was added. */
    check("NULL weight returns -1", nt_qmatvec_i8(out, NULL, 2, x, m, k) == -1, NULL);
    check("NULL activation returns -1", nt_qmatvec_i8(out, W, 2, NULL, m, k) == -1, NULL);
    check("NULL output returns -1", nt_qmatvec_i8(NULL, W, 2, x, m, k) == -1, NULL);

    if (nt_qmatvec_i8(out, W, 2, x, m, k) != 0) {
        check("matvec runs", 0, "returned non-zero on a valid call");
        return 1;
    }

#if HAVE_MALLINFO2
    /* Warm first: the allocator grows its arenas on the early calls and that growth is not
     * a leak. Measure only the steady state. */
    for (int i = 0; i < 64; i++) nt_qmatvec_i8(out, W, 2, x, m, k);
    size_t before = mallinfo2().uordblks;
    const int reps = 2000;
    for (int i = 0; i < reps; i++) nt_qmatvec_i8(out, W, 2, x, m, k);
    size_t after = mallinfo2().uordblks;
    long delta = (long)after - (long)before;
    /* Each leaked scratch would be k/32 int32 = 192 bytes, so 2000 calls would show about
     * 384 kB. Anything under 64 kB is allocator bookkeeping, not per-call retention. */
    char detail[128];
    snprintf(detail, sizeof(detail), "%ld bytes held after %d calls", delta, reps);
    check("nothing retained per call", delta < 65536, detail);
#else
    printf("  SKIP mallinfo2 unavailable — allocation balance not measured here\n");
#endif

    free(W); free(x); free(out);
    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
