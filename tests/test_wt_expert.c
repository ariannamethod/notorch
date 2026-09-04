/* test_wt_expert.c — slicing one expert out of a stacked tensor, in both forms a weight takes.
 *
 * A mixture keeps its experts in one 3-D tensor and wt_load reads that as a single matrix, so
 * an expert is a row range: a shifted base and a shorter row count, nothing copied. Two things
 * make that worth a test rather than a reading. The packed form has to advance by the dtype's
 * row size, which differs per dtype and is easy to get right for Q4_0 and wrong for Q6_K. And
 * the f32 form has to advance at all — it was refused at first, and refusing it did not fail
 * loudly: the caller skipped that expert and answered with seven of its eight.
 *
 * No model is needed. The weights here are counted bytes and counted floats, so a wrong offset
 * is a wrong number rather than a wrong sentence.
 */
#include "harness/runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

int main(void) {
    printf("expert slice out of a stacked tensor\n");

    const int cols = 256, rows_each = 4, n_expert = 5;
    const int rows = rows_each * n_expert;
    char detail[160];

    /* Packed: Q4_0 is 18 bytes per 32 values, so a row is cols/32*18. Fill each row with its
     * own index so a slice that lands wrong reads a number that names where it landed. */
    size_t rb = (size_t)(cols / 32) * 18;
    uint8_t *packed = (uint8_t *)malloc(rb * (size_t)rows);
    if (!packed) { printf("  FAIL out of memory\n"); return 1; }
    for (int r = 0; r < rows; r++) memset(packed + rb * (size_t)r, r, rb);

    wt stacked = { .q = packed, .f32 = NULL, .dtype = 2 /* Q4_0 */,
                   .rows = rows, .cols = cols, .use_i8 = 0 };

    for (int e = 0; e < n_expert; e++) {
        wt slice;
        if (!wt_expert(&slice, &stacked, e, rows_each)) {
            snprintf(detail, sizeof(detail), "expert %d refused", e);
            check("packed slice returns a weight", 0, detail);
            continue;
        }
        int want = e * rows_each;
        int got = slice.q[0];
        snprintf(detail, sizeof(detail), "expert %d starts at row %d, expected %d", e, got, want);
        check("packed slice starts at the right row", got == want && slice.rows == rows_each,
              got == want ? NULL : detail);
    }

    /* Expanded: the same shape as floats, each row holding its own index. */
    float *flat = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!flat) { printf("  FAIL out of memory\n"); return 1; }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) flat[(size_t)r * cols + c] = (float)r;

    wt expanded = { .q = NULL, .f32 = flat, .dtype = 0 /* F32 */,
                    .rows = rows, .cols = cols, .use_i8 = 0 };

    for (int e = 0; e < n_expert; e++) {
        wt slice;
        if (!wt_expert(&slice, &expanded, e, rows_each)) {
            snprintf(detail, sizeof(detail), "expert %d refused", e);
            check("f32 slice returns a weight", 0, detail);
            continue;
        }
        int want = e * rows_each;
        int got = (int)slice.f32[0];
        snprintf(detail, sizeof(detail), "expert %d starts at row %d, expected %d", e, got, want);
        check("f32 slice starts at the right row", got == want && slice.rows == rows_each,
              got == want ? NULL : detail);
    }

    /* What must be refused. A slice past the end would read somebody else's memory, and a
     * weight that is neither packed nor expanded has nothing to slice. */
    wt slice;
    check("a slice past the end is refused",
          !wt_expert(&slice, &stacked, n_expert, rows_each), NULL);
    check("a negative index is refused", !wt_expert(&slice, &stacked, -1, rows_each), NULL);
    wt empty = { .q = NULL, .f32 = NULL, .dtype = 2, .rows = rows, .cols = cols, .use_i8 = 0 };
    check("a weight with no data is refused", !wt_expert(&slice, &empty, 0, rows_each), NULL);

    free(packed); free(flat);
    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
