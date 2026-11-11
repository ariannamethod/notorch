/* Read two native probe checkpoints, compare every F32 logit, and report the
 * selected margin alongside full-row changes. No inference or learning runs. */
#include "ariannamethod.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const AML_Var *get(const AM_Record *r, const char *name) {
    AM_String *key = am_string_new(name);
    const AML_Var *value = key ? am_record_get(r, key) : NULL;
    am_string_free(key);
    return value;
}

static int target(const AM_Record *r, const char *name, int n) {
    const AML_Var *v = get(r, name);
    if (!v || v->type != AML_TYPE_FLOAT || !isfinite(v->value) ||
        v->value < 0 || v->value >= n || floorf(v->value) != v->value) return -1;
    return (int)v->value;
}

typedef struct {
    int n, changed, left_choice, right_choice;
    double max_abs, l2, margin_delta;
} comparison;

static int compare(const AM_Record *a, const AM_Record *b, int probe, comparison *out) {
    char name[24];
    snprintf(name, sizeof name, "probe%d", probe);
    const AML_Var *x = get(a, name), *y = get(b, name);
    if (!x || !y || x->type != AML_TYPE_ARRAY || y->type != AML_TYPE_ARRAY ||
        !x->array || !y->array || x->array->len <= 0 ||
        x->array->len != y->array->len || x->array->rows != y->array->rows ||
        x->array->cols != y->array->cols) return 0;
    int n = x->array->len;
    int left = target(a, "left_id", n), right = target(a, "right_id", n);
    if (left < 0 || right < 0 || left == right ||
        left != target(b, "left_id", n) || right != target(b, "right_id", n)) return 0;
    const float *u = x->array->data, *v = y->array->data;
    *out = (comparison){.n = n};
    double ss = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(u[i]) || !isfinite(v[i])) return 0;
        double d = (double)v[i] - u[i];
        out->changed += memcmp(u+i, v+i, sizeof(float)) != 0;
        out->max_abs = fmax(out->max_abs, fabs(d));
        ss += d*d;
        if (u[i] > u[out->left_choice]) out->left_choice = i;
        if (v[i] > v[out->right_choice]) out->right_choice = i;
    }
    out->l2 = sqrt(ss);
    /* Match the selected F32 subtraction used by the AML protocols. */
    float ma = u[left]-u[right], mb = v[left]-v[right];
    if (!isfinite(ma) || !isfinite(mb)) return 0;
    out->margin_delta = (double)mb-ma;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 || strpbrk(argv[3], "\t\r\n")) {
        fputs("usage: logits_audit LEFT.amlck RIGHT.amlck LABEL\n", stderr);
        return 1;
    }
    char error[256] = {0};
    AM_Record *a = am_checkpoint_load(argv[1], error, sizeof error);
    AM_Record *b = a ? am_checkpoint_load(argv[2], error, sizeof error) : NULL;
    comparison rows[2];
    int ok = a && b && compare(a,b,0,rows) && compare(a,b,1,rows+1);
    if (ok) for (int i = 0; i < 2; i++) {
        const comparison *r = rows+i;
        printf("%s\t%d\t%d\t%d\t%.12g\t%.12g\t%d\t%d\t%.12g\n",
            argv[3],i,r->n,r->changed,r->max_abs,r->l2,
            r->left_choice,r->right_choice,r->margin_delta);
    }
    else fprintf(stderr,"logits audit: incompatible or invalid native probe records: %s\n",error);
    am_record_free(b); am_record_free(a);
    return !ok || fflush(stdout) || ferror(stdout);
}
