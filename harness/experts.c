/* Native DOE for a single dense decoder block. Tensor layout is shared with
 * nt_parliament_*_values. Each turn collects residuals; Oja runs at its end. */
#include "harness/experts.h"
#include "sha256.h"
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { SEATS = 6, FOUNDERS = 4, RANK = 4, WINDOW = 2, HISTORY = 16 };
#define TAU 6.28318530718f

struct nt_experts {
    int dim, layer;
    char host[65];
    FILE *trace;
    uint32_t rng;
    uint64_t turns, elections;
    float consensus;
    float *a, *b, *vote;
    float alive[SEATS], vitality[SEATS], frequency[SEATS], cursor[SEATS];
    float age[SEATS], low[SEATS], history[HISTORY];
    /* The history is oldest first and zero-padded at birth. */
    int count, observed, choices, hits[SEATS];
    double signal;
    float *sum_in, *sum_out, *injected, *update;
};

static float uniform(nt_experts *p) {
    uint32_t v = p->rng;
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    p->rng = v;
    return ((v >> 8) + 1.0f) * 0x1p-24f;
}

static float normal(nt_experts *p) {
    float u = uniform(p);
    return sqrtf(-2.0f * logf(u)) * cosf(TAU * uniform(p));
}

static int host_hash(const char *path, char hex[65]) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    void *map = MAP_FAILED;
    size_t size = 0;
    if (!fstat(fd, &st) && st.st_size > 0 && (uintmax_t)st.st_size <= SIZE_MAX) {
        size = (size_t)st.st_size;
        map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);
    if (map == MAP_FAILED) return -1;
    unsigned char digest[32];
    int rc = nt_sha256(map, size, digest);
    munmap(map, size);
    if (rc) return -1;
    for (int i = 0; i < 32; i++) snprintf(hex + 2*i, 3, "%02x", digest[i]);
    return 0;
}

static int restore(nt_experts *p, const char *path);

nt_experts *nt_experts_open(const gguf_file *host, const char *spore, FILE *trace) {
    if (!host || host->embed_dim < 1 || host->n_layers < 1 ||
        host->embed_dim > NT_MAX_ELEMENTS / (SEATS * RANK)) return NULL;
    nt_experts *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->dim = host->embed_dim;
    p->layer = host->n_layers / 2;
    p->trace = trace;
    p->rng = 0x4e54444f;
    p->consensus = 0.5f;
    size_t dr = (size_t)p->dim * RANK;
    p->a = calloc(SEATS * dr, sizeof(float));
    p->b = calloc(SEATS * dr, sizeof(float));
    p->vote = calloc((size_t)SEATS * p->dim, sizeof(float));
    p->sum_in = calloc(p->dim, sizeof(float));
    p->sum_out = calloc(p->dim, sizeof(float));
    p->injected = calloc(p->dim, sizeof(float));
    p->update = calloc(2 * dr + 1, sizeof(float));
    if (!p->a || !p->b || !p->vote || !p->sum_in || !p->sum_out ||
        !p->injected || !p->update || host_hash(host->path, p->host)) goto fail;
    if (spore) {
        if (restore(p, spore)) goto fail;
    } else {
        for (size_t i = 0; i < (size_t)SEATS * p->dim; i++) p->vote[i] = 0.01f * normal(p);
        for (int e = 0; e < FOUNDERS; e++) {
            p->alive[e] = 1;
            p->vitality[e] = 0.7f;
            p->frequency[e] = TAU * e / FOUNDERS;
            for (size_t i = 0; i < dr; i++) {
                p->a[e*dr+i] = 0.01f * normal(p);
                p->b[e*dr+i] = 0.01f * normal(p);
            }
        }
    }
    return p;
fail:
    nt_experts_free(p);
    return NULL;
}

void nt_experts_free(nt_experts *p) {
    if (!p) return;
    free(p->a); free(p->b); free(p->vote);
    free(p->sum_in); free(p->sum_out); free(p->injected); free(p->update);
    free(p);
}

/* DOE's Fourier amplitudes and frequency-distance vote bias. Here the history
 * is the bounded RMS of this layer's last sixteen input residuals. */
static void resonate(nt_experts *p, const float *x, float bias[SEATS]) {
    double ss = 0;
    for (int i = 0; i < p->dim; i++) ss += (double)x[i] * x[i];
    double rms = sqrt(ss / p->dim);
    memmove(p->history, p->history + 1, (HISTORY-1) * sizeof(float));
    p->history[HISTORY-1] = (float)(rms / (1.0 + rms));
    for (int k = 0; k < HISTORY/2; k++) {
        float re = 0, im = 0;
        for (int i = 0; i < HISTORY; i++) {
            float angle = TAU * (k*i) / HISTORY;
            re += p->history[i] * cosf(angle);
            im += p->history[i] * sinf(angle);
        }
        float amplitude = hypotf(re, im) / HISTORY;
        for (int e = 0; e < SEATS; e++) {
            float d = fabsf(p->frequency[e] - TAU*k/(HISTORY/2));
            d = fminf(d, TAU - d);
            bias[e] += 0.1f * amplitude * expf(-2.0f*d*d);
        }
    }
}

int nt_experts_attend(void *user, int layer, int pos, int n, int width, float *rows) {
    nt_experts *p = user;
    if (layer != p->layer) return NT_OK;
    if (width != p->dim || n < 1 || !rows || n > INT32_MAX - p->count ||
        (uint64_t)n > UINT64_MAX - p->elections) return NT_E_STATE;
    for (int j = 0; j < n; j++) {
        float *x = rows + (size_t)j * width;
        float dots[SEATS], bias[SEATS] = {0}, result[SEATS+2];
        if (nt_linear_values(p->vote, (float[SEATS]){0}, x, SEATS, width, dots)) return NT_E_STATE;
        resonate(p, x, bias);
        if (nt_parliament_elect_values(dots, bias, p->alive, SEATS, 2, p->consensus, result) ||
            nt_parliament_inject_values(p->a, p->b, result, x, SEATS, width, RANK, 0.1f,
                                        p->injected)) return NT_E_STATE;
        p->consensus = result[SEATS];
        p->elections++;
        p->count++;
        if (p->trace) fprintf(p->trace, "doe vote pos=%d k=%.0f consensus=%.9g gates=",
                              pos+j, result[SEATS+1], p->consensus);
        for (int e = 0; e < SEATS; e++) {
            p->hits[e] += result[e] > 0;
            if (p->trace) fprintf(p->trace, "%s%.9g", e ? "," : "", result[e]);
        }
        if (p->trace) fputc('\n', p->trace);
        for (int i = 0; i < width; i++) p->sum_in[i] += x[i];
        memcpy(x, p->injected, (size_t)width * sizeof(float));
    }
    return NT_OK;
}

int nt_experts_observe(void *user, int layer, int pos, int n, int width, float *rows) {
    (void)pos;
    nt_experts *p = user;
    if (layer != p->layer) return NT_OK;
    if (width != p->dim || n < 1 || !rows || n > p->count - p->observed) return NT_E_STATE;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < width; i++) p->sum_out[i] += rows[(size_t)j*width+i];
    p->observed += n;
    return NT_OK;
}

void nt_experts_choice(nt_experts *p, float best_logit, float chosen_logit) {
    double gap = (double)best_logit - chosen_logit;
    double debt = gap / (1.0 + gap);
    p->signal += debt > 0.3 ? -debt : (1.0 - debt) * 0.1;
    p->choices++;
}

/* DOE vitality thresholds. Age is counted in completed turns; its only
 * threshold is 20. Bounded counters stay exact in the spore's F32 tensors. */
static void lifecycle(nt_experts *p) {
    int alive = 0, parent = -1, child = -1, retire = -1;
    for (int e = 0; e < SEATS; e++) alive += (int)p->alive[e];
    for (int e = 0; e < SEATS; e++) {
        if (!p->alive[e]) {
            if (child < 0) child = e;
            continue;
        }
        float share = (float)p->hits[e] / p->count * alive;
        p->vitality[e] = fminf(1, fmaxf(0, p->vitality[e] + 0.05f * (share - 1)));
        p->age[e] = fminf(21, p->age[e] + 1);
        p->low[e] = p->vitality[e] < 0.1f ? fminf(8, p->low[e]+1) : 0;
        if (parent < 0 && p->vitality[e] > 0.8f && p->age[e] > 20) parent = e;
        if (retire < 0 && p->low[e] >= 8) retire = e;
    }
    if (parent >= 0 && child >= 0) {
        size_t dr = (size_t)p->dim * RANK;
        for (size_t i = 0; i < dr; i++) {
            p->a[child*dr+i] = p->a[parent*dr+i] + 0.01f*normal(p);
            p->b[child*dr+i] = p->b[parent*dr+i] + 0.01f*normal(p);
        }
        p->frequency[child] = fmodf(p->frequency[parent] + 0.5f*TAU/(alive+1), TAU);
        p->alive[child] = 1;
        p->age[child] = p->low[child] = p->cursor[child] = 0;
        p->vitality[child] = 0.5f;
        p->vitality[parent] *= 0.8f;
        alive++;
        if (p->trace) fprintf(p->trace, "doe birth parent=%d child=%d\n", parent, child);
    }
    if (retire >= 0 && alive > 2) {
        p->alive[retire] = p->vitality[retire] = 0;
        if (p->trace) fprintf(p->trace, "doe death slot=%d\n", retire);
    }
}

/* SmolLM2 has large residual outliers at the middle block. On raw means an
 * elected expert crossed the quarantine bound on the second turn. Bound
 * the RMS of the learning inputs only; elections and injection see the
 * actual residual. A quiet channel keeps its original scale. */
static void turn_mean(float *v, int dim, int count) {
    double ss = 0;
    for (int i = 0; i < dim; i++) {
        v[i] /= count;
        ss += (double)v[i] * v[i];
    }
    double scale = fmax(1.0, sqrt(ss / dim));
    for (int i = 0; i < dim; i++) v[i] = (float)(v[i] / scale);
}

int nt_experts_finish(nt_experts *p, int learn) {
    if (p->count == 0 || p->observed != p->count || p->choices < 1 ||
        p->turns == UINT64_MAX) return NT_E_STATE;
    size_t dr = (size_t)p->dim * RANK;
    float signal = (float)(p->signal / p->choices);
    turn_mean(p->sum_in, p->dim, p->count);
    turn_mean(p->sum_out, p->dim, p->count);
    for (int e = 0; e < SEATS; e++) {
        if (!learn || !p->hits[e]) continue;
        float *a = p->a + e*dr, *b = p->b + e*dr;
        float noise[WINDOW];
        for (int i = 0; i < WINDOW; i++) noise[i] = normal(p);
        if (nt_oja_values(a, b, p->sum_in, p->sum_out, noise, p->dim, RANK,
                          (int)p->cursor[e], WINDOW, 0.01f, signal, p->update)) return NT_E_STATE;
        for (size_t i = 0; i < 2*dr; i++)
            if (fabsf(p->update[i]) > 1e4f) return NT_E_STATE;
        memcpy(a, p->update, dr*sizeof(float));
        memcpy(b, p->update + dr, dr*sizeof(float));
        p->cursor[e] = p->update[2*dr];
    }
    if (learn) lifecycle(p);
    p->turns++;
    if (p->trace) fprintf(p->trace, "doe turn=%llu elections=%llu signal=%.9g\n",
                          (unsigned long long)p->turns, (unsigned long long)p->elections, signal);
    p->count = p->observed = p->choices = 0;
    p->signal = 0;
    memset(p->hits, 0, sizeof(p->hits));
    memset(p->sum_in, 0, (size_t)p->dim*sizeof(float));
    memset(p->sum_out, 0, (size_t)p->dim*sizeof(float));
    return NT_OK;
}

/* Shared tensor directory for the reader and writer. In GGUF shape[0] is the
 * contiguous dimension. The spore contains no prompt, host weights or KV. */
typedef struct {
    const char *name;
    float *data;
    unsigned ndim;
    uint64_t shape[3];
} spore_tensor;

enum { TENSORS = 10 };
static void directory(nt_experts *p, spore_tensor out[TENSORS]) {
    uint64_t d = (uint64_t)p->dim;
    const spore_tensor fields[TENSORS] = {
        {"doe.lora_a", p->a, 3, {RANK,d,SEATS}},
        {"doe.lora_b", p->b, 3, {d,RANK,SEATS}},
        {"doe.vote", p->vote, 2, {d,SEATS}},
        {"doe.alive", p->alive, 1, {SEATS}},
        {"doe.vitality", p->vitality, 1, {SEATS}},
        {"doe.frequency", p->frequency, 1, {SEATS}},
        {"doe.cursor", p->cursor, 1, {SEATS}},
        {"doe.age", p->age, 1, {SEATS}},
        {"doe.low_vitality", p->low, 1, {SEATS}},
        {"doe.rms_history", p->history, 1, {HISTORY}}
    };
    memcpy(out, fields, sizeof(fields));
}

static uint64_t elements(const spore_tensor *t) {
    uint64_t n = 1;
    for (unsigned j = 0; j < t->ndim; j++) n *= t->shape[j];
    return n;
}

int nt_experts_save(nt_experts *p, const char *path) {
    if (!p || !path || !*path || p->count || !p->turns) return -1;
    char *tmp = malloc(strlen(path) + sizeof(".XXXXXX"));
    if (!tmp) return -1;
    sprintf(tmp, "%s.XXXXXX", path);
    int fd = mkstemp(tmp), rc = -1;
    if (fd < 0) { free(tmp); return -1; }
    gguf_writer *w = gguf_write_open(tmp);
    if (!w) goto done;
    gguf_write_kv_str(w, "general.architecture", "notorch-doe-spore");
    gguf_write_kv_str(w, "doe.host_sha256", p->host);
    gguf_write_kv_u32(w, "doe.version", 1);
    gguf_write_kv_u32(w, "doe.layer", (uint32_t)p->layer);
    gguf_write_kv_u32(w, "doe.rng", p->rng);
    gguf_write_kv_u64(w, "doe.turns", p->turns);
    gguf_write_kv_u64(w, "doe.elections", p->elections);
    gguf_write_kv_f32(w, "doe.consensus", p->consensus);
    spore_tensor t[TENSORS];
    directory(p, t);
    for (int i = 0; i < TENSORS; i++)
        gguf_write_tensor_decl(w, t[i].name, t[i].ndim, t[i].shape, GGUF_TYPE_F32);
    for (int i = 0; i < TENSORS; i++)
        gguf_write_tensor_f32(w, t[i].name, t[i].data, elements(&t[i]));
    /* link publishes all bytes at once and refuses any existing destination,
     * including the host, the parent spore, and a concurrent writer's child. */
    if (!gguf_write_close(w) && !fsync(fd)) rc = link(tmp, path);
done:
    close(fd);
    unlink(tmp);
    free(tmp);
    return rc;
}

static int bounded_int(float v, int max) {
    return v >= 0 && v <= max && v == floorf(v);
}

static int restore(nt_experts *p, const char *path) {
    struct stat st;
    if (stat(path, &st) || st.st_size <= 0) return -1;
    gguf_file *g = gguf_open(path);
    if (!g) return -1;
    int rc = -1;
    const gguf_kv *host = gguf_get_kv(g, "doe.host_sha256");
    const gguf_kv *version = gguf_get_kv(g, "doe.version");
    const gguf_kv *layer = gguf_get_kv(g, "doe.layer");
    const gguf_kv *rng = gguf_get_kv(g, "doe.rng");
    const gguf_kv *turns = gguf_get_kv(g, "doe.turns");
    const gguf_kv *elections = gguf_get_kv(g, "doe.elections");
    const gguf_kv *consensus = gguf_get_kv(g, "doe.consensus");
    if (strcmp(g->arch, "notorch-doe-spore") || g->version != 3 || g->n_tensors != TENSORS ||
        !host || host->type != 8 || strcmp(host->val.str, p->host) ||
        !version || version->type != 4 || version->val.u32 != 1 ||
        !layer || layer->type != 4 || layer->val.u32 != (unsigned)p->layer ||
        !rng || rng->type != 4 || !rng->val.u32 ||
        !turns || turns->type != 10 || !turns->val.u64 ||
        !elections || elections->type != 10 || elections->val.u64 < turns->val.u64 ||
        !consensus || consensus->type != 6 || !isfinite(consensus->val.f32) ||
        consensus->val.f32 < 0 || consensus->val.f32 > 1 ||
        g->data_offset > (uint64_t)st.st_size) goto done;
    p->rng = rng->val.u32;
    p->turns = turns->val.u64;
    p->elections = elections->val.u64;
    p->consensus = consensus->val.f32;
    spore_tensor t[TENSORS];
    directory(p, t);
    uint64_t offset = 0;
    uint64_t physical = (uint64_t)st.st_size - g->data_offset;
    for (int i = 0; i < TENSORS; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        uint64_t n = elements(&t[i]), bytes = n*sizeof(float);
        if (strcmp(ti->name, t[i].name) || ti->dtype != GGUF_TYPE_F32 || ti->ndim != t[i].ndim ||
            memcmp(ti->shape, t[i].shape, ti->ndim*sizeof(uint64_t)) ||
            ti->offset != offset || offset > physical || bytes > physical-offset ||
            offset > g->data_size || bytes > g->data_size-offset) goto done;
        memcpy(t[i].data, g->data + offset, (size_t)bytes);
        for (uint64_t j = 0; j < n; j++)
            if (!isfinite(t[i].data[j]) || fabsf(t[i].data[j]) > 1e4f) goto done;
        offset = (offset + bytes + 31) & ~(uint64_t)31;
    }
    if (offset != physical) goto done;
    int alive = 0;
    for (int i = 0; i < SEATS; i++) {
        if (!bounded_int(p->alive[i], 1) || !bounded_int(p->cursor[i], RANK-1) ||
            !bounded_int(p->age[i], 21) || !bounded_int(p->low[i], 8) ||
            p->vitality[i] < 0 || p->vitality[i] > 1 ||
            p->frequency[i] < 0 || p->frequency[i] >= TAU) goto done;
        alive += (int)p->alive[i];
    }
    for (int j = 0; j < HISTORY; j++)
        if (p->history[j] < 0 || p->history[j] > 1) goto done;
    if (alive >= 2) rc = 0;
done:
    gguf_close(g);
    return rc;
}
