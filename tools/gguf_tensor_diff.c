/* gguf_tensor_diff — does OURS carry the same tensors, byte for byte, as REF?
 *
 *   gguf_tensor_diff OURS.gguf REF.gguf [REF2.gguf ...]
 *
 * For every tensor of every REF: found in OURS by name, same dtype, same element count,
 * same bytes. Shapes are printed when they differ but do not fail the tensor (a bias
 * declared [256] or [1,1,256] is the same 256 values). With --tokenizer, the
 * tokenizer.ggml.{tokens,merges,token_type} arrays are compared too. Exit 0 only when
 * every compared tensor is identical and at least one was compared. */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int same_strs(const char *a, const char *b, const char *key) {
    int na = 0, nb = 0, diff = 0;
    char **x = gguf_read_str_array(a, key, &na), **y = gguf_read_str_array(b, key, &nb);
    if (!x || !y || na != nb) diff = -1;
    for (int i = 0; diff >= 0 && i < na; i++)
        if (strcmp(x[i], y[i])) { if (!diff) fprintf(stderr, "  %s[%d]: '%s' vs '%s'\n", key, i, x[i], y[i]); diff++; }
    printf("%s: %d vs %d entries, %d differ\n", key, na, nb, diff < 0 ? -1 : diff);
    for (int i = 0; x && i < na; i++) free(x[i]);
    for (int i = 0; y && i < nb; i++) free(y[i]);
    free(x); free(y);
    return diff == 0;
}

static int same_i32(const char *a, const char *b, const char *key) {
    int na = 0, nb = 0, diff = 0;
    int32_t *x = gguf_read_i32_array(a, key, &na), *y = gguf_read_i32_array(b, key, &nb);
    if (!x || !y || na != nb) diff = -1;
    for (int i = 0; diff >= 0 && i < na; i++)
        if (x[i] != y[i]) { if (diff < 4) fprintf(stderr, "  %s[%d]: %d vs %d\n", key, i, x[i], y[i]); diff++; }
    printf("%s: %d vs %d entries, %d differ\n", key, na, nb, diff < 0 ? -1 : diff);
    free(x); free(y);
    return diff == 0;
}

int main(int argc, char **argv) {
    int ai = 1, tok = 0;
    if (ai < argc && !strcmp(argv[ai], "--tokenizer")) { tok = 1; ai++; }
    if (argc - ai < 2) { fprintf(stderr, "usage: %s [--tokenizer] OURS.gguf REF.gguf [REF2.gguf ...]\n", argv[0]); return 2; }
    gguf_file *ours = gguf_open(argv[ai]);
    if (!ours) return 2;
    int compared = 0, identical = 0, missing = 0;
    for (int r = ai + 1; r < argc; r++) {
        gguf_file *ref = gguf_open(argv[r]);
        if (!ref) { gguf_close(ours); return 2; }
        for (uint64_t i = 0; i < ref->n_tensors && i < GGUF_MAX_TENSORS; i++) {
            const gguf_tensor_info *t = &ref->tensors[i];
            int j = gguf_find_tensor(ours, t->name);
            if (j < 0) { fprintf(stderr, "missing in ours: %s\n", t->name); missing++; continue; }
            const gguf_tensor_info *u = &ours->tensors[j];
            compared++;
            uint64_t bytes = gguf_type_size(t->dtype, t->n_elements);
            int ok = u->dtype == t->dtype && u->n_elements == t->n_elements && bytes &&
                     !memcmp(ours->data + u->offset, ref->data + t->offset, bytes);
            int shape = u->ndim == t->ndim;
            for (uint32_t d = 0; shape && d < t->ndim; d++) shape = u->shape[d] == t->shape[d];
            if (ok) identical++;
            else {
                uint64_t first = 0;
                if (u->dtype == t->dtype && u->n_elements == t->n_elements && bytes)
                    for (; first < bytes && ours->data[u->offset + first] == ref->data[t->offset + first]; first++) {}
                fprintf(stderr, "differs: %s (dtype %u vs %u, n %llu vs %llu, first differing byte %llu)\n", t->name,
                        u->dtype, t->dtype, (unsigned long long)u->n_elements, (unsigned long long)t->n_elements,
                        (unsigned long long)first);
            }
            if (!shape) fprintf(stderr, "shape declared differently: %s\n", t->name);
        }
        if (tok && r == ai + 1) {
            int ok = same_strs(argv[ai], argv[r], "tokenizer.ggml.tokens") &
                     same_strs(argv[ai], argv[r], "tokenizer.ggml.merges") &
                     same_i32(argv[ai], argv[r], "tokenizer.ggml.token_type");
            if (!ok) missing++;
        }
        gguf_close(ref);
    }
    printf("compared %d, identical %d, differing %d, missing %d\n", compared, identical, compared - identical, missing);
    gguf_close(ours);
    return (compared && identical == compared && !missing) ? 0 : 1;
}
