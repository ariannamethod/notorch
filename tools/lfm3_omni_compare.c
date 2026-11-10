/* lfm3_omni_compare — two NTD1 dumps (see lfm3_omni_probe.c), tag by tag.
 *
 *   lfm3_omni_compare OURS.ntd ORACLE.ntd [MAX_REL_L2]
 *
 * For every oracle record the record of ours with the same tag and layer is found and
 * compared as flat vectors: relative L2 error ||ours - oracle|| / ||oracle||, the largest
 * absolute difference, and the cosine. An oracle record ours does not have is a failure;
 * so is a size mismatch, and, when MAX_REL_L2 is given, any record above it. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char tag[96]; int layer; uint32_t rows, cols; double *x; } rec;
typedef struct { rec *r; int n; } dumpf;

static int load(const char *path, dumpf *d) {
    FILE *f = fopen(path, "rb");
    char magic[4];
    if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "NTD1", 4)) { fprintf(stderr, "%s: not an NTD1 dump\n", path); if (f) fclose(f); return -1; }
    for (;;) {
        uint32_t tl, rows, cols, dt;
        int32_t layer;
        if (fread(&tl, 4, 1, f) != 1) break;
        rec r = {0};
        if (tl >= sizeof r.tag || fread(r.tag, 1, tl, f) != tl || fread(&layer, 4, 1, f) != 1 || fread(&rows, 4, 1, f) != 1 ||
            fread(&cols, 4, 1, f) != 1 || fread(&dt, 4, 1, f) != 1 || dt > 1 || (uint64_t)rows * cols > (1ull << 31)) {
            fprintf(stderr, "%s: malformed record\n", path); fclose(f); return -1;
        }
        size_t n = (size_t)rows * cols;
        r.layer = layer; r.rows = rows; r.cols = cols;
        r.x = malloc((n ? n : 1) * sizeof(double));
        if (!r.x) { fclose(f); return -1; }
        for (size_t i = 0; i < n; i++) {
            if (dt == 0) { float v; if (fread(&v, 4, 1, f) != 1) { fclose(f); return -1; } r.x[i] = v; }
            else if (fread(&r.x[i], 8, 1, f) != 1) { fclose(f); return -1; }
        }
        rec *nr = realloc(d->r, (size_t)(d->n + 1) * sizeof(rec));
        if (!nr) { fclose(f); return -1; }
        d->r = nr;
        d->r[d->n++] = r;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s OURS ORACLE [MAX_REL_L2]\n", argv[0]); return 2; }
    double limit = argc > 3 ? atof(argv[3]) : -1;
    dumpf a = {0}, b = {0};
    if (load(argv[1], &a) || load(argv[2], &b)) return 2;
    int bad = 0, compared = 0;
    double worst = 0;
    for (int i = 0; i < b.n; i++) {
        const rec *o = &b.r[i], *u = NULL;
        for (int j = 0; j < a.n; j++)
            if (!strcmp(a.r[j].tag, o->tag) && a.r[j].layer == o->layer) { u = &a.r[j]; break; }
        if (!u) { printf("%-22s %3d  MISSING in ours\n", o->tag, o->layer); bad++; continue; }
        size_t n = (size_t)o->rows * o->cols;
        if ((size_t)u->rows * u->cols != n) {
            printf("%-22s %3d  SIZE %ux%u vs oracle %ux%u\n", o->tag, o->layer, u->rows, u->cols, o->rows, o->cols);
            bad++; continue;
        }
        double num = 0, den = 0, dot = 0, nu = 0, mx = 0;
        for (size_t k = 0; k < n; k++) {
            double d = u->x[k] - o->x[k];
            num += d * d; den += o->x[k] * o->x[k]; dot += u->x[k] * o->x[k]; nu += u->x[k] * u->x[k];
            if (fabs(d) > mx) mx = fabs(d);
        }
        double rel = den > 0 ? sqrt(num / den) : sqrt(num), cosv = (den > 0 && nu > 0) ? dot / sqrt(den * nu) : 1;
        compared++;
        if (rel > worst) worst = rel;
        int over = limit >= 0 && !(rel <= limit);
        if (over) bad++;
        printf("%-22s %3d  %6ux%-6u rel_l2 %.3e  max_abs %.3e  cos %.9f%s\n", o->tag, o->layer, o->rows, o->cols,
               rel, mx, cosv, over ? "  OVER" : "");
    }
    printf("compared %d records, worst rel_l2 %.3e, %d failing\n", compared, worst, bad);
    return (compared && !bad) ? 0 : 1;
}
