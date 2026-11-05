/* Retrieval scores from encoder vectors: every query against every document as a
 * float64 dot product of the F32 "embedding" vectors, the complete ranking per
 * query with exact ties broken by document ID, and optionally the leading K
 * coordinates of both sides re-normalized first (Matryoshka truncation).
 *
 *   embeddinggemma2_score QUERIES.jsonl DOCUMENTS.jsonl RESULTS.jsonl [--dim K] [--top N]
 *
 * Input lines are embeddinggemma2_encode output, or any JSONL with a safe "id"
 * and an "embedding" array. Values are read as the nearest F32, which recovers
 * exactly what the encoder's %.9g wrote; one outside the F32 range is refused.
 * One output line per query, in input order:
 * "scores" for every document in document-file order, "ranking" as [ID, score]
 * pairs (the first N, all by default), "top1_margin" (first minus second) and,
 * when N is below the document count, "margin_at_top" (N-th minus N+1-th).
 * Scores print as %.17g, which reads back as the same double. Create-only output. */
#include "tools/narrow_json.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char **id; double *v; int n, dim; } vectors;

static int safe_id(const char *s) {
    if (!*s || strlen(s) > 128) return 0;
    for (; *s; s++) if (!isalnum((unsigned char)*s) && *s != '_' && *s != '-' && *s != '.') return 0;
    return 1;
}
static int load(const char *path, vectors *s) {
    FILE *f = fopen(path, "r");
    char *line = NULL;
    size_t cap = 0, have = 0, used = 0;
    int row = 0, rc = -1;
    if (!f) { perror(path); return -1; }
    while (getline(&line, &cap, f) >= 0) {
        int bad = 0, k = 0;
        row++;
        for (const char *x = nj_first(nj_get(line, "embedding"), &bad); x && !bad; x = nj_next(x, ']', &bad), k++) {
            char *e;
            float v = strtof(x, &e);
            if (e == x || !isfinite(v)) { bad = 1; break; }
            if (used == have) {
                double *grown = realloc(s->v, (have = have ? 2 * have : 4096) * sizeof(double));
                if (!grown) goto done;
                s->v = grown;
            }
            s->v[used++] = v;
        }
        if (!s->n) s->dim = k;
        if (bad || k < 1 || k != s->dim) { fprintf(stderr, "%s:%d: embedding is not %d finite F32 values\n", path, row, s->dim); goto done; }
        char *id = nj_string(nj_get(line, "id")), **ids = NULL;
        int seen = 0;
        for (int i = 0; id && i < s->n; i++) seen |= !strcmp(s->id[i], id);
        if (!id || !safe_id(id) || seen) { fprintf(stderr, "%s:%d: missing, unsafe or repeated id\n", path, row); free(id); goto done; }
        if (!(ids = realloc(s->id, (size_t)(s->n + 1) * sizeof(*ids)))) { free(id); goto done; }
        s->id = ids;
        s->id[s->n++] = id;
    }
    rc = s->n ? 0 : -1;
    if (!s->n) fprintf(stderr, "%s: no vectors\n", path);
done:
    free(line);
    fclose(f);
    return rc;
}
/* Leading K coordinates of every row, re-normalized; K = dim keeps rows as given. */
static int truncate_rows(vectors *s, int k) {
    for (int i = 0; i < s->n; i++) {
        double *row = s->v + (size_t)i * (size_t)s->dim, *to = s->v + (size_t)i * (size_t)k, norm = 0;
        for (int j = 0; j < k; j++) norm += row[j] * row[j];
        norm = sqrt(norm);
        if (k < s->dim && !(norm > 0)) return -1;
        for (int j = 0; j < k; j++) to[j] = k < s->dim ? row[j] / norm : row[j];
    }
    s->dim = k;
    return 0;
}
static const double *rank_scores;
static char **rank_ids;
static int by_score(const void *a, const void *b) {
    int i = *(const int *)a, j = *(const int *)b;
    if (rank_scores[i] != rank_scores[j]) return rank_scores[i] > rank_scores[j] ? -1 : 1;
    return strcmp(rank_ids[i], rank_ids[j]);
}

int main(int argc, char **argv) {
    vectors q = {0}, d = {0};
    long k = 0, top = 0;
    int status = 1;
    double *scores = NULL;
    int *order = NULL;
    FILE *out = NULL;
    if (argc < 4 || argc % 2) goto usage;
    for (int i = 4; i < argc; i += 2) {
        char *end;
        long v = strtol(argv[i + 1], &end, 10);
        if (*end || v < 1) goto usage;
        if (!strcmp(argv[i], "--dim")) k = v;
        else if (!strcmp(argv[i], "--top")) top = v;
        else goto usage;
    }
    if (load(argv[1], &q) || load(argv[2], &d)) goto done;
    if (q.dim != d.dim) { fprintf(stderr, "query width %d differs from document width %d\n", q.dim, d.dim); goto done; }
    if (k > q.dim || top > d.n) { fprintf(stderr, "--dim must be within 1..%d and --top within 1..%d\n", q.dim, d.n); goto done; }
    if (!k) k = q.dim;
    if (!top) top = d.n;
    if (truncate_rows(&q, (int)k) || truncate_rows(&d, (int)k)) { fprintf(stderr, "a truncated vector has zero norm\n"); goto done; }
    if (!(scores = malloc((size_t)d.n * sizeof(double))) || !(order = malloc((size_t)d.n * sizeof(int)))) goto done;
    if (!(out = fopen(argv[3], "wx"))) { perror(argv[3]); goto done; }
    rank_scores = scores;
    rank_ids = d.id;
    for (int i = 0; i < q.n; i++) {
        const double *a = q.v + (size_t)i * (size_t)k;
        fprintf(out, "{\"query_id\":\"%s\",\"dim\":%ld,\"scores\":[", q.id[i], k);
        for (int j = 0; j < d.n; j++) {
            const double *b = d.v + (size_t)j * (size_t)k;
            double s = 0;
            for (int c = 0; c < k; c++) s += a[c] * b[c];
            scores[j] = s;
            order[j] = j;
            fprintf(out, "%s%.17g", j ? "," : "", s);
        }
        qsort(order, (size_t)d.n, sizeof(int), by_score);
        fputs("],\"ranking\":[", out);
        for (int r = 0; r < top; r++) fprintf(out, "%s[\"%s\",%.17g]", r ? "," : "", d.id[order[r]], scores[order[r]]);
        fputs("],\"top1_margin\":", out);
        if (d.n > 1) fprintf(out, "%.17g", scores[order[0]] - scores[order[1]]);
        else fputs("null", out);
        if (top < d.n) fprintf(out, ",\"margin_at_top\":%.17g", scores[order[top - 1]] - scores[order[top]]);
        fputs("}\n", out);
    }
    status = ferror(out) ? 1 : 0;
done:
    if (out && fclose(out)) status = 1;
    for (int i = 0; i < q.n; i++) free(q.id[i]);
    for (int i = 0; i < d.n; i++) free(d.id[i]);
    free(q.id); free(q.v); free(d.id); free(d.v); free(scores); free(order);
    return status;
usage:
    fprintf(stderr, "usage: %s QUERIES.jsonl DOCUMENTS.jsonl RESULTS.jsonl [--dim K] [--top N]\n", argv[0]);
    return 2;
}
