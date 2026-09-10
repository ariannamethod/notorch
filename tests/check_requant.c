/* check_requant.c — did a requantised file keep its rows where they were?
 *
 * gguf_quantize walks a tensor row by row, and a tensor of rank three is only a matrix seen
 * from a distance: GGUF stores shape[0] fastest, so rows are contiguous whatever the rank and
 * there are n_elements / shape[0] of them. That is why supporting mixtures took one condition
 * rather than a rewrite — but it is also why an error would be silent. A stride computed
 * wrongly does not crash; it writes each row somewhere plausible and the model answers with
 * somebody else's weights.
 *
 * So: dequantise the same row from both files and compare. Rows are probed at the beginning,
 * the middle and the far end, because a stride that is wrong by a little is only obvious far
 * from row zero. Correct mapping leaves quantisation noise — Q4_0 to Q4_K measured 4.3 to 5.2
 * percent on OLMoE's expert stacks of 65536 and 131072 rows, beside 6.3 percent for an
 * ordinary 2-D tensor in the same file. A wrong mapping leaves two unrelated vectors, which
 * is a relative error near one.
 *
 * This compares a file against the file it was made from; it is not a substitute for running
 * the model, and it says nothing about whether the quantisation was a good idea.
 *
 * Build: make check_requant
 * Run:   ./check_requant source.gguf requantised.gguf
 */
#define _GNU_SOURCE
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static double relerr(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i]-b[i]; num += d*d; den += (double)a[i]*a[i]; }
    return den > 0 ? sqrt(num/den) : (num > 0 ? 1.0 : 0.0);
}
int main(int argc, char **argv) {
    (void)argc;
    gguf_file *A = gguf_open(argv[1]), *B = gguf_open(argv[2]);
    if (!A || !B) return 1;
    /* Every tensor the two files share, rather than a list that would go stale with the next
     * architecture. Rank three first in the report, since those are the ones this exists for. */
    int checked = 0, bad = 0;
    for (uint64_t ti = 0; ti < A->n_tensors; ti++) {
        const char *nm = A->tensors[ti].name;
        int ta = (int)ti, tb = gguf_find_tensor(B, nm);
        if (tb < 0) { printf("  %-34s отсутствует во втором файле\n", nm); continue; }
        if (A->tensors[ta].ndim < 2) continue;      /* 1-D stays float in both */
        const gguf_tensor_info *ia = &A->tensors[ta], *ib = &B->tensors[tb];
        if (ia->shape[0] != ib->shape[0] || ia->n_elements != ib->n_elements) {
            printf("  %-34s форма разошлась\n", nm); bad++; continue;
        }
        int cols = (int)ia->shape[0];
        long rows = (long)(ia->n_elements / ia->shape[0]);
        float *ra = malloc((size_t)cols*4), *rb = malloc((size_t)cols*4);
        double worst = 0; long worst_row = -1;
        /* first, last, and a scatter through the middle — a wrong stride shows at the far end */
        long probes[7] = { 0, 1, rows/4, rows/2, (3*rows)/4, rows-2, rows-1 };
        for (int p = 0; p < 7; p++) {
            long r = probes[p]; if (r < 0 || r >= rows) continue;
            if (gguf_dequant_row(A, ta, (uint64_t)r, ra) || gguf_dequant_row(B, tb, (uint64_t)r, rb)) {
                printf("  %-34s строка %ld не декодируется\n", nm, r); bad++; continue;
            }
            double e = relerr(ra, rb, cols);
            if (e > worst) { worst = e; worst_row = r; }
        }
        checked++;
        int ok = worst < 0.15;
        if (!ok) bad++;
        if (!ok || ia->ndim > 2)
            printf("  %-34s ndim=%u rows=%-7ld худшая rel %.4f (строка %ld)  %s\n",
                   nm, ia->ndim, rows, worst, worst_row,
                   ok ? "шум квантования" : "СТРОКИ НЕ СОВПАДАЮТ");
        free(ra); free(rb);
    }
    printf("\n  проверено тензоров: %d, расхождений: %d\n", checked, bad);
    gguf_close(A); gguf_close(B);
    return bad ? 1 : 0;
}
