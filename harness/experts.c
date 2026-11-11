/* Native DOE for a single dense decoder block. Tensor layout is shared with
 * nt_parliament_*_values. Each turn collects residuals; Oja runs at its end. */
#include "harness/spore_frame.h"
#include "harness/experts_internal.h"
#include "harness/replay_internal.h"
#include "harness/archs.h"
#include "sha256.h"
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int nt_experts_copy_factors(nt_experts *recipient, const nt_experts *donor, int slot) {
    if (!recipient || !donor || slot < 0 || slot >= SEATS) return NT_E_ARG;
    if (!recipient->turns || !donor->turns || recipient->count || donor->count ||
        recipient->choices || donor->choices || !recipient->alive[slot] || !donor->alive[slot] ||
        recipient->dim != donor->dim || recipient->layer != donor->layer ||
        strcmp(recipient->host, donor->host)) return NT_E_STATE;
    if (recipient != donor) {
        size_t n = (size_t)recipient->dim * RANK;
        size_t offset = (size_t)slot * n;
        memcpy(recipient->a + offset, donor->a + offset, n * sizeof(float));
        memcpy(recipient->b + offset, donor->b + offset, n * sizeof(float));
    }
    return NT_OK;
}

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

static void seed_weights(nt_experts *p) {
    size_t dr = (size_t)p->dim * RANK;
    for (size_t i = 0; i < (size_t)SEATS * p->dim; i++) p->vote[i] = 0.01f * normal(p);
    for (int e = 0; e < FOUNDERS; e++) {
        for (size_t i = 0; i < dr; i++) {
            p->a[e*dr+i] = 0.01f * normal(p);
            p->b[e*dr+i] = 0.01f * normal(p);
        }
    }
}

static int restore(nt_experts *p, const char *path);

nt_experts *nt_experts_open(const gguf_file *host, const char *spore, FILE *trace) {
    return nt_experts_open_on_body(host, spore, trace, NULL);
}

nt_experts *nt_experts_open_on_body(const gguf_file *host, const char *spore, FILE *trace,
                                    const nt_expert_body *body) {
    if (!host || host->embed_dim < 1 || host->n_layers < 1 ||
        host->embed_dim > NT_MAX_ELEMENTS / (SEATS * RANK)) return NULL;
    nt_experts *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->dim = host->embed_dim;
    p->depth = host->n_layers;
    p->layer = p->depth / 2;
    snprintf(p->arch, sizeof(p->arch), "%s", host->arch);
    p->decoder = nt_pick_arch(host->arch);
    if (body) {
        if (!p->decoder || !p->decoder->doe || !body->model ||
            p->decoder->doe->describe(body->model, &p->dims) != host ||
            (body->tokenizer && bpe_n_vocab(body->tokenizer) > p->dims.vocab)) goto fail;
        p->body = *body;
        p->frame_host = host;
    }
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
        seed_weights(p);
        for (int e = 0; e < FOUNDERS; e++) {
            p->alive[e] = 1;
            p->vitality[e] = 0.7f;
            p->frequency[e] = TAU * e / FOUNDERS;
        }
        if (nt_lineage_founders(p)) goto fail;
    }
    return p;
fail:
    nt_experts_free(p);
    return NULL;
}

void nt_experts_free(nt_experts *p) {
    if (!p) return;
    free(p->a); free(p->b); free(p->vote); free(p->frame);
    free(p->ancestors);
    free(p->sum_in); free(p->sum_out); free(p->injected); free(p->update);
    free(p);
}

int nt_experts_set_dream_route(nt_experts *p, nt_moe_route_fn route, void *user) {
    if (!p) return NT_E_ARG;
    if (route && (!p->body.model || !p->decoder->forward_routes)) return NT_E_STATE;
    p->dream_route = route;
    p->dream_route_user = route ? user : NULL;
    return NT_OK;
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
        if (p->trace) {
            flockfile(p->trace);
            fprintf(p->trace, "doe %svote pos=%d k=%.0f consensus=%.9g gates=",
                    p->dreaming==2 ? "replay-" : p->dreaming ? "dream-" : "", pos+j, result[SEATS+1], p->consensus);
        }
        for (int e = 0; e < SEATS; e++) {
            p->hits[e] += result[e] > 0;
            if (p->trace) fprintf(p->trace, "%s%.9g", e ? "," : "", result[e]);
        }
        if (p->trace) { fputc('\n', p->trace); funlockfile(p->trace); }
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
static int lifecycle(nt_experts *p) {
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
        int rc=nt_lineage_birth(p,parent,child);
        if(rc) return rc;
        alive++;
        if (p->trace) fprintf(p->trace, "doe birth parent=%d child=%d\n", parent, child);
    }
    if (retire >= 0 && alive > 2) {
        nt_lineage_retire(p,retire);
        p->alive[retire] = p->vitality[retire] = 0;
        if (p->trace) fprintf(p->trace, "doe death slot=%d\n", retire);
    }
    return NT_OK;
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

int nt_experts_update_factors(nt_experts *p, float learning_rate, float signal) {
    size_t dr = (size_t)p->dim * RANK;
    turn_mean(p->sum_in, p->dim, p->count);
    turn_mean(p->sum_out, p->dim, p->count);
    for (int e = 0; e < SEATS; e++) {
        if (!p->hits[e]) continue;
        float *a = p->a + e*dr, *b = p->b + e*dr;
        float noise[WINDOW];
        for (int i = 0; i < WINDOW; i++) noise[i] = normal(p);
        if (nt_oja_values(a, b, p->sum_in, p->sum_out, noise, p->dim, RANK,
                          (int)p->cursor[e], WINDOW, learning_rate, signal, p->update)) return NT_E_STATE;
        for (size_t i = 0; i < 2*dr; i++)
            if (fabsf(p->update[i]) > 1e4f) return NT_E_STATE;
        memcpy(a, p->update, dr*sizeof(float));
        memcpy(b, p->update + dr, dr*sizeof(float));
        p->cursor[e] = p->update[2*dr];
    }
    return NT_OK;
}

int nt_experts_finish(nt_experts *p, int learn) {
    if (p->count == 0 || p->observed != p->count || p->choices < 1 ||
        p->turns == UINT64_MAX) return NT_E_STATE;
    float signal = (float)(p->signal / p->choices);
    if (learn && nt_experts_update_factors(p, 0.01f, signal)) return NT_E_STATE;
    if (learn) { int rc=lifecycle(p); if(rc) return rc; }
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

struct nt_consolidation {
    pthread_t thread;
    nt_experts *state;
    int finish_turn, dream_tokens, eos, result;
    nt_replay_episode *episode;
    int replay_tokens;
};

static void *consolidate_turn(void *arg) {
    nt_consolidation *job = arg;
    job->result = job->finish_turn ? nt_experts_finish(job->state, 1) : NT_OK;
    if (job->result == NT_OK && job->dream_tokens)
        job->result = nt_experts_dream(job->state, job->dream_tokens, job->eos, 1);
    if (job->result == NT_OK && job->episode)
        job->result = nt_experts_replay(job->state, job->episode, job->replay_tokens, 1);
    return NULL;
}

static int submit(nt_experts **owner, nt_consolidation **job, int tokens, int eos,
                   const nt_replay_episode *episode, int replay_tokens) {
    if (!owner || !*owner || !job || *job) return NT_E_ARG;
    if (episode && !nt_replay_matches(*owner,episode)) return NT_E_STATE;
    nt_consolidation *next = calloc(1,sizeof(*next));
    if (!next) return NT_E_MEMORY;
    if (episode && !(next->episode=nt_replay_clone(episode))) { free(next); return NT_E_MEMORY; }
    next->replay_tokens=replay_tokens;
    next->state = *owner;
    next->finish_turn = (!tokens && !episode) || next->state->count != 0;
    next->dream_tokens = tokens;
    next->eos = eos;
    if (pthread_create(&next->thread, NULL, consolidate_turn, next)) {
        nt_replay_free(next->episode);
        free(next);
        return NT_E_STATE;
    }
    *owner = NULL;
    *job = next;
    return NT_OK;
}

int nt_experts_submit(nt_experts **owner, nt_consolidation **job) {
    return submit(owner, job, 0, -1, NULL, 0);
}

int nt_experts_submit_dream(nt_experts **owner, nt_consolidation **job, int tokens, int eos) {
    if (tokens < 1 || tokens > NT_DREAM_MAX_TOKENS || eos < -1) return NT_E_ARG;
    return submit(owner, job, tokens, eos, NULL, 0);
}

int nt_experts_submit_replay(nt_experts **owner,nt_consolidation **job,
                             const nt_replay_episode *episode,int tokens) {
    if(!episode || tokens<1 || tokens>NT_DREAM_MAX_TOKENS) return NT_E_ARG;
    return submit(owner,job,0,-1,episode,tokens);
}

int nt_experts_await(nt_experts **owner, nt_consolidation **job) {
    if (!owner || !job) return NT_E_ARG;
    nt_consolidation *pending = *job;
    if (!pending) return NT_OK;
    if (*owner || pthread_join(pending->thread, NULL)) return NT_E_STATE;
    int result = pending->result;
    if (result == NT_OK) *owner = pending->state;
    else nt_experts_free(pending->state);
    nt_replay_free(pending->episode);
    free(pending);
    *job = NULL;
    return result;
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
static void directory(nt_experts *p, uint64_t d, spore_tensor out[TENSORS]) {
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

int nt_experts_calibrate(nt_experts *p) {
    if (p->frame) return 0;
    if (!p->decoder || !p->decoder->doe || p->dim < NT_SPORE_ANCHORS ||
        p->dim > NT_MAX_ELEMENTS / NT_SPORE_ANCHORS || !p->frame_host)
        return -1;
    float *frame = malloc((size_t)p->dim * NT_SPORE_ANCHORS * sizeof(*frame));
    if (!frame) return -1;
    if (nt_spore_frame_build(&p->body, p->frame_host, p->decoder, p->dims, frame, p->norm_sha)) { free(frame); return -1; }
    p->frame = frame;
    return 0;
}

static spore_tensor frame_tensor(float *frame, int width) {
    return (spore_tensor){"doe.anchor_frame", frame, 2, {(uint64_t)width, NT_SPORE_ANCHORS}};
}

static int save_spore(nt_experts *p, const char *path, int portable) {
    if (!p || !path || !*path || p->count || !p->turns) return -1;
    if (portable && nt_experts_calibrate(p)) return -1;
    char *tmp = malloc(strlen(path) + sizeof(".XXXXXX"));
    if (!tmp) return -1;
    sprintf(tmp, "%s.XXXXXX", path);
    int fd = mkstemp(tmp), rc = -1;
    if (fd < 0) { free(tmp); return -1; }
    gguf_writer *w = gguf_write_open(tmp);
    if (!w) goto done;
    gguf_write_kv_str(w, "general.architecture", "notorch-doe-spore");
    gguf_write_kv_str(w, "doe.host_sha256", p->host);
    gguf_write_kv_u32(w, "doe.version", portable ? 6 : 5);
    gguf_write_kv_u32(w, "doe.layer", (uint32_t)p->layer);
    gguf_write_kv_u32(w, "doe.rng", p->rng);
    gguf_write_kv_u64(w, "doe.turns", p->turns);
    gguf_write_kv_u64(w, "doe.elections", p->elections);
    gguf_write_kv_f32(w, "doe.consensus", p->consensus);
    if (p->sleep.cycles) {
        gguf_write_kv_u64(w, "doe.dream.cycles", p->sleep.cycles);
        gguf_write_kv_u64(w, "doe.dream.choices", p->sleep.choices);
        gguf_write_kv_u64(w, "doe.dream.elections", p->sleep.elections);
        gguf_write_kv_u32(w, "doe.dream.seen", p->sleep.seen);
    }
    if (p->replay.cycles) {
        gguf_write_kv_u64(w, "doe.replay.cycles", p->replay.cycles);
        gguf_write_kv_u64(w, "doe.replay.choices", p->replay.choices);
        gguf_write_kv_u64(w, "doe.replay.elections", p->replay.elections);
        gguf_write_kv_u64(w, "doe.replay.prefill", p->replay.prefill);
        gguf_write_kv_u64(w, "doe.replay.stops", p->replay.stops);
        gguf_write_kv_str(w, "doe.replay.last", p->replay.last);
    }
    if (nt_lineage_write(p,w)) { gguf_write_close(w); goto done; }
    spore_tensor t[TENSORS+1];
    directory(p, p->dim, t);
    if (portable) {
        gguf_write_kv_str(w, "doe.frame", NT_SPORE_FRAME_ID);
        gguf_write_kv_str(w, "doe.source_arch", p->arch);
        gguf_write_kv_str(w, "doe.norm_sha256", p->norm_sha);
        gguf_write_kv_u32(w, "doe.width", (uint32_t)p->dim);
        gguf_write_kv_u32(w, "doe.depth", (uint32_t)p->depth);
        t[TENSORS] = frame_tensor(p->frame, p->dim);
    }
    int count = TENSORS + portable;
    for (int i = 0; i < count; i++)
        gguf_write_tensor_decl(w, t[i].name, t[i].ndim, t[i].shape, GGUF_TYPE_F32);
    for (int i = 0; i < count; i++)
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

int nt_experts_save(nt_experts *p, const char *path) {
    return save_spore(p, path, 0);
}

int nt_experts_save_portable(nt_experts *p, const char *path) {
    return save_spore(p, path, 1);
}

static int bounded_int(float v, int max) {
    return v >= 0 && v <= max && v == floorf(v);
}

static int valid_lifecycle(const nt_experts *p) {
    int alive = 0;
    for (int i = 0; i < SEATS; i++) {
        if (!bounded_int(p->alive[i], 1) || !bounded_int(p->cursor[i], RANK-1) ||
            !bounded_int(p->age[i], 21) || !bounded_int(p->low[i], 8) ||
            p->vitality[i] < 0 || p->vitality[i] > 1 ||
            p->frequency[i] < 0 || p->frequency[i] >= TAU) return 0;
        alive += (int)p->alive[i];
    }
    for (int j = 0; j < HISTORY; j++)
        if (p->history[j] < 0 || p->history[j] > 1) return 0;
    return alive >= 2;
}

/* Validate the complete donor directory before any donor dimension is used to
 * allocate memory. Tensor views are borrowed until this spore is closed. */
static int tensor_views(const gguf_file *g, uint64_t physical, spore_tensor *t, int count,
                        const float **values) {
    uint64_t offset = 0;
    for (int i = 0; i < count; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        uint64_t n = elements(&t[i]), bytes = n * sizeof(float);
        if (strcmp(ti->name, t[i].name) || ti->dtype != GGUF_TYPE_F32 || ti->ndim != t[i].ndim ||
            memcmp(ti->shape, t[i].shape, ti->ndim*sizeof(uint64_t)) ||
            ti->offset != offset || offset > physical || bytes > physical-offset ||
            offset > g->data_size || bytes > g->data_size-offset) return -1;
        values[i] = (const float *)(g->data + offset);
        for (uint64_t j = 0; j < n; j++)
            if (!isfinite(values[i][j]) || fabsf(values[i][j]) > 1e4f) return -1;
        offset = (offset + bytes + 31) & ~(uint64_t)31;
    }
    return offset == physical ? 0 : -1;
}

static int project_seats(nt_experts *p, int width, const float *frame, const float **values) {
    seed_weights(p);
    for (int e = 0; e < SEATS; e++) {
        size_t src = (size_t)e * width, dst = (size_t)e * p->dim;
        if (nt_spore_project(frame, width, values[2]+src, 1,
                             p->frame, p->dim, p->vote+dst, 1)) return -1;
        for (int r = 0; r < RANK; r++) {
            if (nt_spore_project(frame, width, values[0]+src*RANK+r, RANK,
                                 p->frame, p->dim, p->a+dst*RANK+r, RANK) ||
                nt_spore_project(frame, width, values[1]+src*RANK+(size_t)r*width, 1,
                                 p->frame, p->dim, p->b+dst*RANK+(size_t)r*p->dim, 1))
                return -1;
        }
    }
    return 0;
}

static const gguf_kv *field(const gguf_file *g, const char *name, unsigned type) {
    const gguf_kv *kv = gguf_get_kv(g, name);
    return kv && kv->type == type ? kv : NULL;
}

static int restore_sleep(nt_experts *p, const gguf_file *g, int present) {
    const gguf_kv *cycles = gguf_get_kv(g, "doe.dream.cycles");
    const gguf_kv *choices = gguf_get_kv(g, "doe.dream.choices");
    const gguf_kv *elections = gguf_get_kv(g, "doe.dream.elections");
    const gguf_kv *seen = gguf_get_kv(g, "doe.dream.seen");
    if (!present) return cycles || choices || elections || seen ? -1 : 0;
    if (!cycles || cycles->type != 10 || !choices || choices->type != 10 ||
        !elections || elections->type != 10 || !seen || seen->type != 4) return -1;
    uint64_t n = cycles->val.u64, c = choices->val.u64, e = elections->val.u64;
    uint32_t mask = seen->val.u32;
    unsigned used = 0;
    for (uint32_t m = mask; m; m &= m-1) used++;
    /* Each cycle has 1..32 choices; its last choice may be a stop.
     * Elections add at most one short raw cue per cycle. Division avoids
     * overflowing when a saved clock is close to UINT64_MAX. */
    if (!n || c < n || (c-1) / NT_DREAM_MAX_TOKENS >= n || e < c ||
        (e > c && (e-c-1) / (NT_SPORE_CUE_CAP-1) >= n) ||
        used != 1+(n-1)%NT_SPORE_ANCHORS) return -1;
    p->sleep.cycles = n;
    p->sleep.choices = c;
    p->sleep.elections = e;
    p->sleep.seen = mask;
    return 0;
}

static int is_digest(const gguf_kv *v) {
    return v && strlen(v->val.str) == 64 && strspn(v->val.str, "0123456789abcdef") == 64;
}

static int restore_replay(nt_experts *p,const gguf_file *g,int modern) {
    const char *names[]={"doe.replay.cycles","doe.replay.choices","doe.replay.elections",
                         "doe.replay.prefill","doe.replay.stops","doe.replay.last"};
    const gguf_kv *v[6]; int present=0;
    for(int i=0;i<6;i++) { v[i]=gguf_get_kv(g,names[i]); present+=v[i]!=NULL; }
    if(!present) return 0;
    if(!modern || present!=6 || v[5]->type!=8 || !is_digest(v[5])) return -1;
    for(int i=0;i<5;i++) if(v[i]->type!=10) return -1;
    uint64_t n=v[0]->val.u64,c=v[1]->val.u64,e=v[2]->val.u64,
             f=v[3]->val.u64,s=v[4]->val.u64;
    /* Every cycle has a nonempty prefix and 1..32 choices; a final stop does
     * not enter the decoder. Subtraction keeps the clock identity overflow-free. */
    if(!n || c<n || (c-1)/NT_DREAM_MAX_TOKENS>=n || f<n || s>n ||
       e<f || e-f!=c-s) return -1;
    p->replay.cycles=n; p->replay.choices=c; p->replay.elections=e;
    p->replay.prefill=f; p->replay.stops=s; memcpy(p->replay.last,v[5]->val.str,65);
    return 0;
}

static int restore(nt_experts *p, const char *path) {
    struct stat st;
    if (stat(path, &st) || st.st_size <= 0) return -1;
    gguf_file *g = gguf_open(path);
    if (!g) return -1;
    int rc = -1;
    const gguf_kv *host = field(g, "doe.host_sha256", 8);
    const gguf_kv *version = field(g, "doe.version", 4);
    const gguf_kv *layer = field(g, "doe.layer", 4);
    const gguf_kv *rng = field(g, "doe.rng", 4);
    const gguf_kv *turns = field(g, "doe.turns", 10);
    const gguf_kv *elections = field(g, "doe.elections", 10);
    const gguf_kv *consensus = field(g, "doe.consensus", 6);
    if (strcmp(g->arch, "notorch-doe-spore") || g->version != 3 ||
        !version || version->val.u32 < 1 || version->val.u32 > 6 ||
        !is_digest(host) ||
        !layer || !rng || !rng->val.u32 || !turns || !turns->val.u64 ||
        !elections || elections->val.u64 < turns->val.u64 ||
        !consensus || !isfinite(consensus->val.f32) ||
        consensus->val.f32 < 0 || consensus->val.f32 > 1 ||
        g->data_offset > (uint64_t)st.st_size) goto done;
    int modern=version->val.u32>=5;
    if (!modern && (gguf_get_kv(g,"doe.lineage.version") ||
                    gguf_get_kv(g,"doe.lineage.entries") || gguf_get_kv(g,"doe.lineage.seats"))) goto done;
    int slept=modern ? gguf_get_kv(g,"doe.dream.cycles")!=NULL : version->val.u32>=3;
    if (restore_sleep(p, g, slept)) goto done;
    if (restore_replay(p, g, modern)) goto done;
    int portable = (version->val.u32 & 1)==0;
    int same_host = !strcmp(host->val.str, p->host);
    if (g->n_tensors != (uint64_t)(TENSORS + portable) ||
        (!same_host && !portable) || (same_host && layer->val.u32 != (unsigned)p->layer))
        goto done;

    int width = p->dim, same_shape = 0;
    const gguf_kv *norm_sha = field(g, "doe.norm_sha256", 8);
    spore_tensor t[TENSORS+1];
    const float *values[TENSORS+1];
    if (portable) {
        const gguf_kv *frame = field(g, "doe.frame", 8);
        const gguf_kv *arch = field(g, "doe.source_arch", 8);
        const gguf_kv *w = field(g, "doe.width", 4);
        const gguf_kv *depth = field(g, "doe.depth", 4);
        const nt_arch *source = arch ? nt_pick_arch(arch->val.str) : NULL;
        if (!frame || strcmp(frame->val.str, NT_SPORE_FRAME_ID) || !is_digest(norm_sha) ||
            !source || !source->doe ||
            !w || w->val.u32 < NT_SPORE_ANCHORS || w->val.u32 > NT_MAX_ELEMENTS / NT_SPORE_ANCHORS ||
            !depth || !depth->val.u32 || depth->val.u32 / 2 != layer->val.u32) goto done;
        width = (int)w->val.u32;
        same_shape = width == p->dim && depth->val.u32 == (unsigned)p->depth &&
                     !strcmp(arch->val.str, p->arch);
        if (same_host && !same_shape) goto done;
        t[TENSORS] = frame_tensor(NULL, width);
    }
    directory(p, width, t);
    if (tensor_views(g, (uint64_t)st.st_size-g->data_offset, t, TENSORS+portable, values) ||
        (portable && !nt_spore_frame_valid(values[TENSORS], width))) goto done;
    for (int i = 3; i < TENSORS; i++)
        memcpy(t[i].data, values[i], (size_t)elements(&t[i])*sizeof(float));
    if (!valid_lifecycle(p)) goto done;
    if (modern) {
        if (nt_lineage_read(p,g,turns->val.u64)) goto done;
    } else {
        char digest[65];
        if (host_hash(path,digest) || nt_lineage_adopt(p,digest,host->val.str,
            turns->val.u64,width,values[0],values[1])) goto done;
    }
    int copy_weights = 1;
    if (!same_host) {
        if (nt_experts_calibrate(p)) goto done;
        const float *frame = values[TENSORS];
        double minimum = same_shape ? nt_spore_frame_min_cos(frame, p->frame, width) : 0;
        /* Each measured axis must stay closer to its original direction than
         * to the perpendicular plane (45 degrees), and all norm digests match. */
        int aligned = same_shape && !strcmp(norm_sha->val.str, p->norm_sha) &&
                      minimum > 0.7071067811865475;
        if (!aligned) {
            if (project_seats(p, width, frame, values)) goto done;
            copy_weights = 0;
        }
        if (p->trace) {
            fprintf(p->trace, "doe transfer mode=%s anchors=%d width=%d->%d layer=%u->%d",
                    aligned ? "aligned" : "projected", NT_SPORE_ANCHORS, width, p->dim,
                    layer->val.u32, p->layer);
            if (same_shape) fprintf(p->trace, " min_cos=%.9g", minimum);
            fprintf(p->trace, " source=%s target=%s\n", host->val.str, p->host);
        }
    }
    if (copy_weights) {
        for (int i = 0; i < 3; i++)
            memcpy(t[i].data, values[i], (size_t)elements(&t[i])*sizeof(float));
    }
    p->rng = rng->val.u32;
    p->turns = turns->val.u64;
    p->elections = elections->val.u64;
    p->consensus = consensus->val.f32;
    rc = 0;
done:
    gguf_close(g);
    return rc;
}
