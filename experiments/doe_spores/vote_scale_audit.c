/* Read-only vote-scale audit. Capture pre-injection residuals once, then run
 * elections on disposable population copies. No learning or spore saves. */
#define _GNU_SOURCE
#include "harness/experts_internal.h"
#include "harness/archs.h"
#include "harness/spore_frame.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { CAP = 512, NHELD = 8, CAPTURE_STOP = -1 };

static const char *const held[NHELD] = {
    "Every night the lighthouse keeper carries a",
    "At the lighthouse, Miro reaches for his",
    "Miro is the lighthouse keeper. Every night Miro carries a key. At sunrise the lighthouse keeper returns home.",
    "Miro is the lighthouse keeper. Every night Miro carries a bell. At sunrise the lighthouse keeper returns home.",
    "The orchard has apple trees. In spring the trees bloom.",
    "A musician practices scales. The room has a wooden floor.",
    "Rain falls on the mountain. A stream runs through the valley.",
    "The baker makes bread. The market opens in the morning."
};

typedef struct {
    int layer, width, n;
    float *rows;
} Capture;

typedef struct {
    int n;
    float *rows;
} Sample;

static int capture(void *ptr, int layer, int pos, int n, int width, float *rows) {
    Capture *c = ptr;
    if (layer != c->layer) return NT_OK;
    if (pos != 0 || width != c->width || n < 1 || n > CAP || !rows)
        return NT_E_STATE;
    c->n = n;
    memcpy(c->rows, rows, (size_t)n * width * sizeof(float));
    return CAPTURE_STOP;
}

static double norm(const float *x, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    return sqrt(ss);
}

static int dots(const nt_experts *p, const float *x, float d[SEATS]) {
    return nt_linear_values(p->vote, (float[SEATS]){0}, x, SEATS, p->dim, d) == NT_OK;
}

static double centered(const nt_experts *p, const float *d, double *mean) {
    double sum = 0, ss = 0;
    int n = 0;
    for (int e = 0; e < SEATS; e++) {
        if (p->alive[e]) {
            sum += d[e];
            n++;
        }
    }
    *mean = sum / n;
    for (int e = 0; e < SEATS; e++) {
        if (p->alive[e]) ss += (d[e] - *mean) * (d[e] - *mean);
    }
    return ss / n;
}

static int raw(FILE *out, const char *owner, const char *kind, int sample, int row,
               const nt_experts *p, const float *x) {
    float d[SEATS];
    if (!dots(p, x, d)) return 0;
    double mean, sigma = sqrt(centered(p, d, &mean));
    int best = -1, second = -1;
    for (int e = 0; e < SEATS; e++) {
        if (!p->alive[e]) continue;
        if (best < 0 || d[e] > d[best]) {
            second = best;
            best = e;
        } else if (second < 0 || d[e] > d[second]) {
            second = e;
        }
    }
    double xn = norm(x, p->dim);
    if (best < 0 || second < 0 || !isfinite(sigma) || !isfinite(xn) || xn <= 0)
        return 0;
    fprintf(out, "%s\t%s\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%.17g",
            owner, kind, sample, row, xn, xn / sqrt(p->dim), mean, sigma,
            best, (double)d[best] - d[second]);
    for (int e = 0; e < SEATS; e++) fprintf(out, "\t%.17g", (double)d[e]);
    for (int e = 0; e < SEATS; e++) {
        fprintf(out, "\t%.17g", d[e] / (xn * norm(p->vote + (size_t)e * p->dim, p->dim)));
    }
    fputc('\n', out);
    return !ferror(out);
}

static int simulate(FILE *out, const char *owner, const char *label, int sample,
                    const nt_experts *p, const Sample *s, int mode, double factor) {
    static const char *const modes[] = {
        "raw", "scale0.5", "scale1", "scale2", "unit_vote_norm", "cosine"
    };
    nt_experts a = *p;
    a.vote = malloc((size_t)SEATS * p->dim * sizeof(float));
    a.sum_in = calloc(p->dim, sizeof(float));
    a.injected = malloc((size_t)p->dim * sizeof(float));
    a.trace = NULL;
    float *copy = malloc((size_t)s->n * p->dim * sizeof(float));
    char *trace = NULL;
    size_t bytes = 0;
    int ok = 0;
    if (!a.vote || !a.sum_in || !a.injected || !copy) goto done;
    memcpy(copy, s->rows, (size_t)s->n * p->dim * sizeof(float));
    a.trace = open_memstream(&trace, &bytes);
    if (!a.trace) goto done;
    for (int j = 0; j < s->n; j++) {
        double xn = norm(s->rows + (size_t)j * p->dim, p->dim);
        if (!isfinite(xn) || xn <= 0) goto done;
        for (int e = 0; e < SEATS; e++) {
            double gain = mode == 0 ? 1 : mode == 1 ? factor * .5 :
                          mode == 2 ? factor : mode == 3 ? factor * 2 :
                          1 / norm(p->vote + (size_t)e * p->dim, p->dim);
            if (mode == 5) gain /= xn;
            for (int i = 0; i < p->dim; i++) {
                a.vote[(size_t)e * p->dim + i] = (float)(p->vote[(size_t)e * p->dim + i] * gain);
            }
        }
        if (nt_experts_attend(&a, p->layer, j, 1, p->dim, copy + (size_t)j * p->dim))
            goto done;
    }
    int close_error = fclose(a.trace);
    a.trace = NULL;
    if (close_error) goto done;

    double mass[SEATS] = {0}, effective = 0, maxmass = 0, kavg = 0;
    int wins[SEATS] = {0}, n = 0;
    char *line = trace;
    while (line && *line) {
        float consensus;
        double g[SEATS];
        int pos, k;
        int got = sscanf(line, "doe vote pos=%d k=%d consensus=%f gates=%lf,%lf,%lf,%lf,%lf,%lf",
                         &pos, &k, &consensus, g, g + 1, g + 2, g + 3, g + 4, g + 5);
        if (got != 9 || pos != n) goto done;
        double sum = 0, h = 0, m = -1;
        int best = 0;
        for (int e = 0; e < SEATS; e++) sum += g[e];
        if (sum <= 0 || !isfinite(sum)) goto done;
        for (int e = 0; e < SEATS; e++) {
            double q = g[e] / sum;
            mass[e] += q;
            if (q > 0) h -= q * log(q);
            if (q > m) {
                m = q;
                best = e;
            }
        }
        wins[best]++;
        effective += exp(h);
        maxmass += m;
        kavg += k;
        n++;
        line = strchr(line, '\n');
        if (line) line++;
    }
    if (n != s->n) goto done;
    fprintf(out, "%s\t%s\t%d\t%s\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",
            owner, label, sample, modes[mode], n, p->consensus, a.consensus,
            kavg / n, effective / n, maxmass / n);
    for (int e = 0; e < SEATS; e++) fprintf(out, "\t%.17g", mass[e] / n);
    for (int e = 0; e < SEATS; e++) fprintf(out, "\t%d", wins[e]);
    fputc('\n', out);
    ok = !ferror(out);
done:
    if (a.trace) fclose(a.trace);
    free(trace);
    free(copy);
    free(a.vote);
    free(a.sum_in);
    free(a.injected);
    return ok;
}

static char *output_path(const char *directory, const char *name) {
    size_t d = strlen(directory), n = strlen(name);
    if (d > SIZE_MAX - n - 2) return NULL;
    char *path = malloc(d + n + 2);
    if (path) snprintf(path, d + n + 2, "%s/%s", directory, name);
    return path;
}

int main(int argc, char **argv) {
    if (argc != 5 || !*argv[1] || !*argv[2] || !*argv[3] || !*argv[4]) {
        fprintf(stderr, "usage: %s HOST PARENT_SPORE FINAL_SPORE OUTPUT_DIRECTORY\n", argv[0]);
        return 1;
    }

    int rc = 1;
    const char *failure = "could not prepare the audit";
    gguf_file *host = NULL;
    const nt_arch *arch = NULL;
    void *model = NULL;
    bpe_tokenizer *tok = NULL;
    nt_dims dims = {0};
    nt_experts *p[2] = {0};
    Sample anchors[NT_SPORE_ANCHORS] = {0}, heldouts[NHELD] = {0};
    float *space = NULL;
    const char *const names[] = {"raw.tsv", "gates.tsv", "summary.tsv"};
    char *paths[3] = {0};
    FILE *outputs[3] = {0};
    int created[3] = {0};

    /* Reject existing destinations before inference. On a later error, remove
     * only files created by this invocation; existing files are never opened. */
    for (int i = 0; i < 3; i++) {
        paths[i] = output_path(argv[4], names[i]);
        if (!paths[i]) goto done;
        outputs[i] = fopen(paths[i], "wx");
        if (!outputs[i]) {
            fprintf(stderr, "vote_scale_audit: could not create %s: %s\n", paths[i], strerror(errno));
            goto done;
        }
        created[i] = 1;
    }

    host = gguf_open(argv[1]);
    if (!host) { failure = "could not open the host"; goto done; }
    arch = nt_pick_arch(host->arch);
    if (!arch || !arch->doe || !arch->doe->forward) {
        failure = "host has no DOE forward";
        goto done;
    }
    model = arch->load(host, &dims);
    tok = bpe_load(argv[1]);
    if (!model || !tok) { failure = "could not load the body or its tokenizer"; goto done; }
    if (host->embed_dim < 1 || (size_t)host->embed_dim > SIZE_MAX / (CAP * sizeof(float)))
        goto done;
    p[0] = nt_experts_open(host, argv[2], NULL);
    p[1] = nt_experts_open(host, argv[3], NULL);
    if (!p[0] || !p[1]) { failure = "could not restore both populations on this host"; goto done; }
    for (int owner = 0; owner < 2; owner++) {
        if (p[owner]->dim != host->embed_dim || p[owner]->layer != dims.n_layers / 2) {
            failure = "population and capture shapes differ";
            goto done;
        }
        for (int e = 0; e < SEATS; e++) {
            double vote_norm = norm(p[owner]->vote + (size_t)e * p[owner]->dim, p[owner]->dim);
            if (!isfinite(vote_norm) || vote_norm <= 0) {
                failure = "cosine audit requires finite nonzero vote rows";
                goto done;
            }
        }
    }
    space = malloc((size_t)CAP * host->embed_dim * sizeof(float));
    if (!space) goto done;
    for (int test = 0; test < NT_SPORE_ANCHORS + NHELD; test++) {
        int ids[CAP], is_anchor = test < NT_SPORE_ANCHORS;
        int n = is_anchor ? nt_spore_cue_ids(tok, test, ids) :
                           bpe_encode(tok, held[test - NT_SPORE_ANCHORS], ids, CAP);
        if (n < 1 || n >= CAP) { failure = "input tokenization failed or reached the cap"; goto done; }
        kv_cache *kv = nt_model_kv_new(arch, model, dims, n);
        if (!kv) { failure = "could not allocate private KV"; goto done; }
        Capture c = {dims.n_layers / 2, host->embed_dim, 0, space};
        int result = arch->doe->forward(model, kv, ids, n, 0, NULL, capture, NULL, &c);
        nt_model_kv_free(arch, model, kv);
        if (result != CAPTURE_STOP || c.n != n) {
            failure = "could not capture the complete raw residual";
            goto done;
        }
        Sample *s = is_anchor ? anchors + test : heldouts + test - NT_SPORE_ANCHORS;
        s->n = is_anchor ? 1 : n;
        s->rows = malloc((size_t)s->n * host->embed_dim * sizeof(float));
        if (!s->rows) goto done;
        memcpy(s->rows, space + (is_anchor ? (size_t)(n - 1) * host->embed_dim : 0),
               (size_t)s->n * host->embed_dim * sizeof(float));
    }

    FILE *r = outputs[0], *g = outputs[1], *sum = outputs[2];
    fputs("owner\tkind\tsample\trow\tx_l2\tx_rms\tdot_mean\tdot_centered_rms\traw_winner\ttop_gap", r);
    for (int e = 0; e < SEATS; e++) fprintf(r, "\tdot%d", e);
    for (int e = 0; e < SEATS; e++) fprintf(r, "\tcos%d", e);
    fputc('\n', r);
    fputs("owner\tkind\tsample\tmode\tpositions\tconsensus_start\tconsensus_end\tmean_k\tmean_effective\tmean_max_mass", g);
    for (int e = 0; e < SEATS; e++) fprintf(g, "\tmean_mass%d", e);
    for (int e = 0; e < SEATS; e++) fprintf(g, "\twins%d", e);
    fputc('\n', g);
    fputs("owner\tanchor_centered_rms\tunit_rms_scale\tconsensus\tturns\telections\tvote_l2_0\tvote_l2_1\tvote_l2_2\tvote_l2_3\tvote_l2_4\tvote_l2_5\n", sum);
    fprintf(stderr, "votes_byte_identical=%d width=%d\n",
            !memcmp(p[0]->vote, p[1]->vote, (size_t)SEATS * host->embed_dim * sizeof(float)),
            host->embed_dim);
    for (int owner = 0; owner < 2; owner++) {
        const char *name = owner ? "final" : "parent";
        double ss = 0;
        for (int j = 0; j < NT_SPORE_ANCHORS; j++) {
            float d[SEATS];
            double mean;
            if (!dots(p[owner], anchors[j].rows, d)) goto done;
            ss += centered(p[owner], d, &mean);
        }
        double scale = 1 / sqrt(ss / NT_SPORE_ANCHORS);
        if (!isfinite(scale) || scale <= 0) {
            failure = "anchor variance does not admit a finite positive scale";
            goto done;
        }
        fprintf(sum, "%s\t%.17g\t%.17g\t%.17g\t%llu\t%llu", name, 1 / scale, scale,
                p[owner]->consensus, (unsigned long long)p[owner]->turns,
                (unsigned long long)p[owner]->elections);
        for (int e = 0; e < SEATS; e++) {
            fprintf(sum, "\t%.17g", norm(p[owner]->vote + (size_t)e * host->embed_dim, host->embed_dim));
        }
        fputc('\n', sum);
        for (int j = 0; j < NT_SPORE_ANCHORS + NHELD; j++) {
            int is_anchor = j < NT_SPORE_ANCHORS;
            int index = is_anchor ? j : j - NT_SPORE_ANCHORS;
            Sample *s = is_anchor ? anchors + index : heldouts + index;
            for (int row = 0; row < s->n; row++) {
                if (!raw(r, name, is_anchor ? "anchor" : "held", index, row, p[owner],
                         s->rows + (size_t)row * host->embed_dim)) {
                    failure = "raw vote measurement or output failed";
                    goto done;
                }
            }
            for (int mode = 0; mode < 6; mode++) {
                if (!simulate(g, name, is_anchor ? "anchor" : "held", index, p[owner], s, mode, scale)) {
                    failure = "election simulation or output failed";
                    goto done;
                }
            }
        }
    }
    rc = 0;
done:
    if (rc) fprintf(stderr, "vote_scale_audit: %s\n", failure);
    for (int i = 0; i < 3; i++) {
        if (!outputs[i]) continue;
        int error = ferror(outputs[i]);
        if (fflush(outputs[i])) error = 1;
        if (fclose(outputs[i])) error = 1;
        if (error) {
            fprintf(stderr, "vote_scale_audit: could not finish %s\n", paths[i]);
            rc = 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (rc && created[i]) unlink(paths[i]);
        free(paths[i]);
    }
    for (int j = 0; j < NT_SPORE_ANCHORS; j++) free(anchors[j].rows);
    for (int j = 0; j < NHELD; j++) free(heldouts[j].rows);
    nt_experts_free(p[0]);
    nt_experts_free(p[1]);
    free(space);
    bpe_free(tok);
    if (model) arch->free(model);
    gguf_close(host);
    return rc;
}
