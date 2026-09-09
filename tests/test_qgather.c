/* test_qgather.c — one activation through many matrices must equal many calls, bit for bit.
 *
 * nt_qmatvec_i8_gather exists to stop a mixture paying for eight dispatches and eight
 * quantizations of the same activation. It is worth having only if it changes nothing but the
 * time: rows are independent and write disjoint outputs, so a gathered dispatch and a loop of
 * separate ones must produce identical floats — not close ones. A tolerance here would hide
 * exactly the bug this is for, a chunk that straddles two slices and lands rows in the wrong
 * expert's output.
 *
 * The shapes are chosen to make that straddle happen: a row count per slice that the chunk
 * size does not divide, and a slice count that is not a multiple of the thread count.
 */
#include "notorch.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

/* Packed bytes per row, for the dtypes this test drives. */
static size_t row_bytes(int dtype, int k) {
    switch (dtype) {
    case GGUF_TYPE_Q4_0: return (size_t)(k / 32) * 18;
    case GGUF_TYPE_Q5_0: return (size_t)(k / 32) * 22;
    case GGUF_TYPE_Q8_0: return (size_t)(k / 32) * 34;
    case GGUF_TYPE_Q4_K: return (size_t)(k / 256) * 144;
    case GGUF_TYPE_Q6_K: return (size_t)(k / 256) * 210;
    default:             return 0;
    }
}

static int one_dtype(int dtype, const char *name, int k, int n_slices, int rows_each) {
    size_t rb = row_bytes(dtype, k);
    if (!rb) return 0;

    /* One stack far larger than what is used, with the slices taken from scattered positions —
     * a mixture never gets to read adjacent experts, and a gather that quietly assumed it
     * would should fail here. */
    int stack = n_slices * 5;
    uint8_t *W = (uint8_t *)malloc(rb * (size_t)rows_each * (size_t)stack);
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *a = (float *)calloc((size_t)n_slices * rows_each, sizeof(float));
    float *b = (float *)calloc((size_t)n_slices * rows_each, sizeof(float));
    const uint8_t **slices = (const uint8_t **)malloc((size_t)n_slices * sizeof(*slices));
    if (!W || !x || !a || !b || !slices) { free(W); free(x); free(a); free(b); free(slices); return 0; }

    /* Quantized from floats rather than filled with bytes. A packed block carries an f16
     * scale, and random bytes make that scale a NaN — which compares unequal to itself, so a
     * byte-filled weight fails this test whatever the kernel does. Learned here by doing it. */
    float *row = (float *)malloc((size_t)k * sizeof(float));
    if (!row) { free(W); free(x); free(a); free(b); free(slices); return 0; }
    for (long r = 0; r < (long)rows_each * stack; r++) {
        for (int i = 0; i < k; i++)
            row[i] = (float)(((r * 7919 + i * 104729) % 2001) - 1000) / 500.0f;
        if (nt_quantize_row(row, W + rb * (size_t)r, k, dtype) != 0) {
            free(row); free(W); free(x); free(a); free(b); free(slices);
            printf("  SKIP %s — this build cannot pack that dtype\n", name);
            return 0;
        }
    }
    free(row);
    for (int i = 0; i < k; i++) x[i] = (float)((i % 31) - 15) * 0.0625f;
    for (int s = 0; s < n_slices; s++)
        slices[s] = W + rb * (size_t)rows_each * (size_t)(s * 5 + 2);

    char detail[192];
    int rc_loop = 0;
    for (int s = 0; s < n_slices; s++)
        if (nt_qmatvec_i8(a + (long)s * rows_each, slices[s], dtype, x, rows_each, k) != 0)
            rc_loop = -1;
    int rc_gather = nt_qmatvec_i8_gather(b, slices, n_slices, dtype, x, rows_each, k);

    snprintf(detail, sizeof(detail), "loop returned %d, gather %d", rc_loop, rc_gather);
    check(name, rc_loop == 0 && rc_gather == 0, (rc_loop == 0 && rc_gather == 0) ? NULL : detail);

    if (rc_loop == 0 && rc_gather == 0) {
        long n = (long)n_slices * rows_each, bad = -1;
        for (long i = 0; i < n; i++) if (a[i] != b[i]) { bad = i; break; }
        if (bad >= 0)
            snprintf(detail, sizeof(detail),
                     "%s: row %ld of %ld differs — loop %.9g, gather %.9g (slice %ld, local %ld)",
                     name, bad, n, (double)a[bad], (double)b[bad],
                     bad / rows_each, bad % rows_each);
        check("gathered result is identical to the loop", bad < 0, bad < 0 ? NULL : detail);
    }

    free(W); free(x); free(a); free(b); free(slices);
    return 1;
}

int main(void) {
    printf("gathered matvec against the loop it replaces\n");

    /* rows_each deliberately awkward: the pool cuts chunks of (rows)/(threads*16), and 260 is
     * not a multiple of anything that produces, so chunks land across slice boundaries. */
    one_dtype(GGUF_TYPE_Q4_0, "Q4_0", 2048, 7, 260);
    one_dtype(GGUF_TYPE_Q5_0, "Q5_0", 1024, 5, 132);
    one_dtype(GGUF_TYPE_Q8_0, "Q8_0", 1024, 3, 68);
    one_dtype(GGUF_TYPE_Q4_K, "Q4_K", 1024, 5, 132);
    one_dtype(GGUF_TYPE_Q6_K, "Q6_K", 1024, 3, 68);

    /* And the shape a mixture actually runs, where the chunk size does divide the slice. */
    one_dtype(GGUF_TYPE_Q4_0, "Q4_0 at the OLMoE shape", 2048, 8, 1024);

    /* What must be refused rather than guessed at. */
    const uint8_t *one[1] = { NULL };
    float out[4], x[32] = {0};
    check("a NULL slice is refused",
          nt_qmatvec_i8_gather(out, one, 1, GGUF_TYPE_Q4_0, x, 4, 32) != 0, NULL);
    check("no slices at all is refused",
          nt_qmatvec_i8_gather(out, (const uint8_t *const *)one, 0, GGUF_TYPE_Q4_0, x, 4, 32) != 0,
          NULL);

    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
