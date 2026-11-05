/* embeddinggemma2_score, run as a binary.
 *
 * Without arguments: four-dimensional dyadic vectors whose dot products are exact
 * in double, with an exact tie in each query, scored in full, with --top 2 and
 * with --dim 2 (where re-normalization changes a score from 0.375 to 1), and the
 * refusals: a value beyond the F32 range, mismatched widths, a repeated ID, a zero
 * norm after truncation, out-of-range --dim/--top, an existing output.
 *
 * With NPY_DIR BASELINE.jsonl: the frozen gate. NPY_DIR holds the members of the
 * reference embeddings.npz (unzip it; the archive is deflated). The reference's
 * own F32 query and document embeddings are written as JSONL, scored, and the
 * result compared with its frozen scores.npy (|difference| <= 1e-6 for every
 * pair) and with every complete ranking of BASELINE.jsonl, which must be equal. */
#include "tools/narrow_json.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(c, what) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s (line %d)\n", what, __LINE__); } } while (0)

static int sh(const char *cmd) {
    int s = system(cmd);
    return WIFEXITED(s) ? WEXITSTATUS(s) : 128;
}
static void put(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f || fputs(text, f) < 0 || fclose(f)) { perror(path); exit(1); }
}
static int score(const char *dir, const char *q, const char *d, const char *out, const char *opts) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "./embeddinggemma2_score %s/%s %s/%s %s/%s %s 2>/dev/null", dir, q, dir, d, dir, out, opts);
    return sh(cmd);
}
/* Document IDs of a ranking value: [["id",score],...] or [{"document_id":...},...]. */
static int ranking_ids(const char *v, char **ids, int cap) {
    int bad = 0, n = 0;
    for (const char *x = nj_first(v, &bad); x && n < cap; x = nj_next(x, ']', &bad)) {
        int b = 0;
        const char *id = *x == '[' ? nj_first(x, &b) : nj_get(x, "document_id");
        if (!(ids[n] = nj_string(id))) return -1;
        n++;
    }
    return bad ? -1 : n;
}
static int doubles(const char *v, double *out, int cap) {
    int bad = 0, n = 0;
    for (const char *x = nj_first(v, &bad); x && n < cap; x = nj_next(x, ']', &bad))
        if (nj_number(x, &out[n++])) return -1;
    return bad ? -1 : n;
}

/* ── frozen gate ── */
typedef struct { char descr[8]; long shape[2]; int ndim; size_t count, item; unsigned char *buf; const unsigned char *data; } npy;
static int npy_load(const char *dir, const char *name, npy *a) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    long size = f && !fseek(f, 0, SEEK_END) ? ftell(f) : -1;
    memset(a, 0, sizeof(*a));
    if (size < 12 || fseek(f, 0, SEEK_SET) || !(a->buf = malloc((size_t)size + 1)) ||
        fread(a->buf, 1, (size_t)size, f) != (size_t)size || memcmp(a->buf, "\x93NUMPY", 6)) goto bad;
    const unsigned char *b = a->buf;
    size_t off = b[6] == 1 ? 10 : 12;
    size_t len = b[6] == 1 ? (size_t)(b[8] | b[9] << 8) : (size_t)b[8] | (size_t)b[9] << 8 | (size_t)b[10] << 16 | (size_t)b[11] << 24;
    if (off + len > (size_t)size) goto bad;
    char header[1024] = {0};
    memcpy(header, b + off, len < sizeof(header) - 1 ? len : sizeof(header) - 1);
    const char *p = strstr(header, "'descr': '"), *s = strstr(header, "'shape': (");
    if (!p || !s || !strstr(header, "'fortran_order': False")) goto bad;
    sscanf(p + 10, "%7[^']", a->descr);
    a->count = 1;
    for (char *e, *q = (char *)s + 10; *q != ')' && a->ndim < 2; q = e) {
        while (*q == ' ' || *q == ',') q++;
        if (*q == ')') break;
        a->shape[a->ndim] = strtol(q, &e, 10);
        if (e == q || a->shape[a->ndim] < 1) goto bad;
        a->count *= (size_t)a->shape[a->ndim++];
    }
    if (!strcmp(a->descr, "<f4")) a->item = 4;
    else if (!strncmp(a->descr, "<U", 2)) a->item = 4 * (size_t)atoi(a->descr + 2);
    if (!a->item || !a->ndim || off + len + a->count * a->item != (size_t)size) goto bad;
    a->data = b + off + len;
    fclose(f);
    return 0;
bad:
    fprintf(stderr, "%s: not a little-endian C-order <f4 or <U .npy\n", path);
    if (f) fclose(f);
    free(a->buf);
    return -1;
}
static float f4(const npy *a, size_t i) { float v; memcpy(&v, a->data + 4 * i, 4); return v; }
static void unicode_id(const npy *a, size_t i, char *out, size_t cap) {
    size_t n = 0;
    for (size_t k = 0; k < a->item / 4 && n + 1 < cap; k++) {
        uint32_t c;
        memcpy(&c, a->data + i * a->item + 4 * k, 4);
        if (!c) break;
        out[n++] = c < 128 ? (char)c : '?';
    }
    out[n] = 0;
}
static int write_jsonl(const char *path, const npy *ids, const npy *v) {
    FILE *f = fopen(path, "w");
    char id[64];
    for (long i = 0; f && i < v->shape[0]; i++) {
        unicode_id(ids, (size_t)i, id, sizeof(id));
        fprintf(f, "{\"id\":\"%s\",\"embedding\":[", id);
        for (long j = 0; j < v->shape[1]; j++) fprintf(f, "%s%.9g", j ? "," : "", f4(v, (size_t)(i * v->shape[1] + j)));
        fputs("]}\n", f);
    }
    return f && !fclose(f) ? 0 : -1;
}
static int gate(const char *npy_dir, const char *baseline_path) {
    npy qi, di, qe, de, sc;
    char dir[] = "/tmp/notorch-eg2-score-gate-XXXXXX", path[256], id[64];
    if (npy_load(npy_dir, "query_ids.npy", &qi) || npy_load(npy_dir, "document_ids.npy", &di) ||
        npy_load(npy_dir, "queries_embeddings.npy", &qe) || npy_load(npy_dir, "documents_embeddings.npy", &de) ||
        npy_load(npy_dir, "scores.npy", &sc)) return 1;
    long Q = qe.shape[0], N = de.shape[0];
    if (qe.ndim != 2 || de.ndim != 2 || qe.shape[1] != de.shape[1] || qi.shape[0] != Q || di.shape[0] != N ||
        sc.ndim != 2 || sc.shape[0] != Q || sc.shape[1] != N || qi.descr[1] != 'U' || di.descr[1] != 'U') {
        fprintf(stderr, "frozen arrays disagree in shape\n");
        return 1;
    }
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/q.jsonl", dir);
    int ok = !write_jsonl(path, &qi, &qe);
    snprintf(path, sizeof(path), "%s/d.jsonl", dir);
    ok = ok && !write_jsonl(path, &di, &de) && score(dir, "q.jsonl", "d.jsonl", "out.jsonl", "") == 0;
    snprintf(path, sizeof(path), "%s/out.jsonl", dir);
    char *out = ok ? nj_load(path, NULL) : NULL, *base = nj_load(baseline_path, NULL);
    double worst = 0, *s = malloc((size_t)N * sizeof(double));
    char **got = calloc((size_t)N, sizeof(char *)), **want = calloc((size_t)N, sizeof(char *));
    long pairs = 0, lines = 0, rankings = 0, baseline_queries = 0;
    for (const char *p = base; p && *p; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL) if (*p != '\n') baseline_queries++;
    for (const char *p = out; ok && p && *p && lines < Q; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL, lines++) {
        char *qid = nj_string(nj_get(p, "query_id"));
        unicode_id(&qi, (size_t)lines, id, sizeof(id));
        if (!qid || strcmp(qid, id) || doubles(nj_get(p, "scores"), s, (int)N) != N) { ok = 0; free(qid); break; }
        for (long j = 0; j < N; j++, pairs++) {
            double d = fabs(s[j] - (double)f4(&sc, (size_t)(lines * N + j)));
            if (d > worst) worst = d;
        }
        const char *b = NULL;
        for (const char *r = base; r && *r && !b; r = strchr(r, '\n') ? strchr(r, '\n') + 1 : NULL)
            if (nj_is(nj_get(r, "query_id"), qid)) b = r;
        int ng = ranking_ids(nj_get(p, "ranking"), got, (int)N), nw = b ? ranking_ids(nj_get(b, "ranking"), want, (int)N) : -1, same = ng == N && nw == N;
        for (long j = 0; j < N; j++) {
            if (same && strcmp(got[j], want[j])) same = 0;
            free(got[j]); free(want[j]); got[j] = want[j] = NULL;
        }
        if (same) rankings++;
        else fprintf(stderr, "RANKING DIFFERS %s\n", qid);
        free(qid);
    }
    printf("score gate: %ld queries x %ld documents, max |score - frozen| = %.3g over %ld pairs, %ld/%ld rankings identical\n",
           Q, N, worst, pairs, rankings, baseline_queries);
    const char *made[] = {"q.jsonl", "d.jsonl", "out.jsonl"};
    for (int i = 0; i < 3; i++) { snprintf(path, sizeof(path), "%s/%s", dir, made[i]); unlink(path); }
    rmdir(dir);
    free(out); free(base); free(s); free(got); free(want);
    free(qi.buf); free(di.buf); free(qe.buf); free(de.buf); free(sc.buf);
    return ok && lines == Q && pairs == Q * N && worst <= 1e-6 && rankings == Q && baseline_queries == Q ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 3) return gate(argv[1], argv[2]);
    if (argc != 1) { fprintf(stderr, "usage: %s [NPY_DIR BASELINE.jsonl]\n", argv[0]); return 2; }
    char dir[] = "/tmp/notorch-eg2-score-XXXXXX", path[256];
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
#define FILE_IN(name, text) do { snprintf(path, sizeof(path), "%s/%s", dir, name); put(path, text); } while (0)
    FILE_IN("q.jsonl", "{\"id\":\"q1\",\"embedding\":[1,0,0,0]}\n{\"id\":\"q2\",\"tokens\":4,\"embedding\":[0,0.5,0.25,0]}\n");
    FILE_IN("d.jsonl", "{\"id\":\"d3\",\"embedding\":[0.5,0.5,0.5,0.5]}\n{\"id\":\"d2\",\"embedding\":[1,0,0,0]}\n"
                       "{\"id\":\"d1\",\"embedding\":[1,0,0,0]}\n{\"id\":\"d4\",\"embedding\":[0,0.75,0.25,0.5]}\n");
    FILE_IN("notf32.jsonl", "{\"id\":\"d1\",\"embedding\":[1e39,0,0,0]}\n");
    FILE_IN("encoder.jsonl", "{\"id\":\"d1\",\"embedding\":[0.100000001,0,0,0]}\n"); /* %.9g of 0.1f, as the encoder writes it */
    FILE_IN("narrow.jsonl", "{\"id\":\"d1\",\"embedding\":[1,0,0]}\n");
    FILE_IN("repeat.jsonl", "{\"id\":\"d1\",\"embedding\":[1,0,0,0]}\n{\"id\":\"d1\",\"embedding\":[0,1,0,0]}\n");
    FILE_IN("zero.jsonl", "{\"id\":\"d1\",\"embedding\":[0,0,1,0]}\n");
    CHECK(score(dir, "q.jsonl", "d.jsonl", "full.jsonl", "") == 0, "full scoring runs");
    CHECK(score(dir, "q.jsonl", "d.jsonl", "top.jsonl", "--top 2") == 0, "--top runs");
    CHECK(score(dir, "q.jsonl", "d.jsonl", "dim.jsonl", "--dim 2") == 0, "--dim runs");
    CHECK(score(dir, "q.jsonl", "notf32.jsonl", "r1.jsonl", "") != 0, "a value beyond the F32 range is refused");
    CHECK(score(dir, "q.jsonl", "encoder.jsonl", "r8.jsonl", "") == 0, "the encoder's %.9g F32 text is accepted");
    CHECK(score(dir, "q.jsonl", "narrow.jsonl", "r2.jsonl", "") != 0, "mismatched widths are refused");
    CHECK(score(dir, "q.jsonl", "repeat.jsonl", "r3.jsonl", "") != 0, "a repeated ID is refused");
    CHECK(score(dir, "q.jsonl", "zero.jsonl", "r4.jsonl", "--dim 2") != 0, "a zero norm after truncation is refused");
    CHECK(score(dir, "q.jsonl", "d.jsonl", "r5.jsonl", "--dim 5") != 0 && score(dir, "q.jsonl", "d.jsonl", "r6.jsonl", "--top 5") != 0 &&
          score(dir, "q.jsonl", "d.jsonl", "r7.jsonl", "--dim 0") != 0, "--dim and --top out of range are refused");
    CHECK(score(dir, "q.jsonl", "d.jsonl", "full.jsonl", "") != 0, "an existing output is refused");

    const double r = 0.5 / sqrt(0.5);
    struct { const char *file; int line; double scores[4]; const char *ranking[4]; int top; double margin, at_top; } want[] = {
        {"full.jsonl", 0, {0.5, 1, 1, 0}, {"d1", "d2", "d3", "d4"}, 4, 0, NAN},       /* d1/d2 tie: ID decides */
        {"full.jsonl", 1, {0.375, 0, 0, 0.4375}, {"d4", "d3", "d1", "d2"}, 4, 0.0625, NAN},
        {"top.jsonl", 0, {0.5, 1, 1, 0}, {"d1", "d2"}, 2, 0, 0.5},
        {"top.jsonl", 1, {0.375, 0, 0, 0.4375}, {"d4", "d3"}, 2, 0.0625, 0.375},
        {"dim.jsonl", 0, {r, 1, 1, 0}, {"d1", "d2", "d3", "d4"}, 4, 0, NAN},
        {"dim.jsonl", 1, {r, 0, 0, 1}, {"d4", "d3", "d1", "d2"}, 4, 1 - r, NAN},     /* d4 is 0.375 untruncated */
    };
    for (size_t i = 0; i < sizeof(want) / sizeof(*want); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, want[i].file);
        char *text = nj_load(path, NULL), *ids[4] = {0};
        const char *p = text;
        for (int k = 0; p && k < want[i].line; k++) if ((p = strchr(p, '\n'))) p++;
        double s[4], margin = NAN, at_top = NAN;
        int ns = p ? doubles(nj_get(p, "scores"), s, 4) : -1, nr = p ? ranking_ids(nj_get(p, "ranking"), ids, 4) : -1, ok = ns == 4 && nr == want[i].top;
        for (int k = 0; ok && k < 4; k++) ok = fabs(s[k] - want[i].scores[k]) <= 1e-15;
        for (int k = 0; ok && k < nr; k++) ok = !strcmp(ids[k], want[i].ranking[k]);
        if (ok && nj_number(nj_get(p, "top1_margin"), &margin) == 0) ok = fabs(margin - want[i].margin) <= 1e-15;
        else ok = 0;
        if (nj_get(p, "margin_at_top")) nj_number(nj_get(p, "margin_at_top"), &at_top);
        ok = ok && (isnan(want[i].at_top) ? isnan(at_top) : fabs(at_top - want[i].at_top) <= 1e-15);
        if (!ok) fprintf(stderr, "case %s line %d\n", want[i].file, want[i].line);
        CHECK(ok, "hand-worked scores, ranking with ID tie-break, margins");
        for (int k = 0; k < 4; k++) free(ids[k]);
        free(text);
    }
    const char *made[] = {"q.jsonl", "d.jsonl", "notf32.jsonl", "encoder.jsonl", "narrow.jsonl", "repeat.jsonl", "zero.jsonl",
                          "full.jsonl", "top.jsonl", "dim.jsonl", "r1.jsonl", "r2.jsonl", "r3.jsonl", "r4.jsonl", "r5.jsonl",
                          "r6.jsonl", "r7.jsonl", "r8.jsonl"};
    for (size_t i = 0; i < sizeof(made) / sizeof(*made); i++) { snprintf(path, sizeof(path), "%s/%s", dir, made[i]); unlink(path); }
    CHECK(rmdir(dir) == 0, "no stray output left behind");
    if (failures) { fprintf(stderr, "EmbeddingGemma2 score: %d FAILED\n", failures); return 1; }
    puts("EmbeddingGemma2 score: float64 scores, ID tie-break, top-k margins, Matryoshka re-normalization and refusals PASS");
    return 0;
}
