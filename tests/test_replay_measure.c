/* An explicit coordinate decoder exercises the native reader and election
 * kernel. Its two living proposals have independently known energies and
 * disagreement, and its private cache counts every input position. */
#include "harness/experts_internal.h"
#include "harness/replay_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define CHECK(test, label) do { checks++; if (!(test)) { \
    fprintf(stderr, "FAIL replay measure: %s (line %d)\n", label, __LINE__); failures++; \
} } while (0)

typedef struct {
    int width, layer, vocab, fail_pos, bad_input, skip_observe;
    int allocated, released, heads, media_rows, calls;
} decoder;

static kv_cache *cache_new(void *model, int capacity) {
    decoder *m = model;
    kv_cache *kv = kv_new(1, capacity, 1);
    if (kv) m->allocated++;
    return kv;
}

static void cache_free(void *model, kv_cache *kv) {
    decoder *m = model;
    if (kv) m->released++;
    kv_free(kv);
}

static int forward_rows(void *model, kv_cache *kv, const int *ids, const float *const *media,
                        int n, int pos, float *logits, float *hidden,
                        nt_residual_fn before, nt_residual_fn after, void *user) {
    decoder *m = model;
    m->calls++;
    if (!kv || pos != (int)kv->v[0] || pos + n > kv->max_seq) return NT_E_CACHE;
    float *row = calloc((size_t)m->width, sizeof(float));
    if (!row) return NT_E_MEMORY;
    int rc = NT_OK;
    for (int j = 0; j < n; j++) {
        memset(row, 0, (size_t)m->width * sizeof(float));
        if (media && media[j]) {
            memcpy(row, media[j], (size_t)m->width * sizeof(float));
            m->media_rows++;
        } else if (ids[j] == 0) row[0] = 1;
        else if (ids[j] == 1) row[1] = 2;
        else if (ids[j] == 3) row[0] = 4;
        if (pos + j == m->fail_pos) {
            if (!m->bad_input) { rc = NT_E_STATE; break; }
            row[0] = NAN;
        }
        if (before && (rc = before(user, m->layer, pos + j, 1, m->width, row))) break;
        if (after && !m->skip_observe && (rc = after(user, m->layer, pos + j, 1, m->width, row))) break;
        kv->v[0]++;
    }
    if (!rc && logits) {
        m->heads++;
        for (int i = 0; i < m->vocab; i++) logits[i] = i ? -1 : 2;
    }
    if (!rc && hidden) memcpy(hidden, row, (size_t)m->width * sizeof(float));
    free(row);
    return rc;
}

static int forward(void *model, kv_cache *kv, const int *ids, int n, int pos, float *logits,
                   nt_residual_fn before, nt_residual_fn after, void *user) {
    return forward_rows(model, kv, ids, NULL, n, pos, logits, NULL, before, after, user);
}

static const nt_doe_decoder methods = {.forward = forward, .forward_embd = forward_rows};
static const nt_arch architecture = {.kv_new = cache_new, .kv_free = cache_free, .doe = &methods};

static void attach(nt_experts *p, gguf_file *host, decoder *m, bpe_tokenizer *tok) {
    p->body = (nt_expert_body){m, tok};
    p->frame_host = host;
    p->decoder = &architecture;
    p->dims = (nt_dims){1, 1, bpe_n_vocab(tok)};
    m->width = p->dim; m->layer = p->layer; m->vocab = p->dims.vocab; m->fail_pos = -1;
}

/* mode0: u0=(2*x0,0), u1=(0,3*x1).
 * mode1: both u=(2*x0,0). mode2: u1=-u0. */
static void factors(nt_experts *p, int mode) {
    size_t dr = (size_t)p->dim * RANK;
    memset(p->a, 0, SEATS * dr * sizeof(float));
    memset(p->b, 0, SEATS * dr * sizeof(float));
    p->a[0] = 2; p->b[0] = 1;
    if (!mode) { p->a[dr + RANK] = 3; p->b[dr + 1] = 1; }
    else { p->a[dr] = mode == 1 ? 2 : -2; p->b[dr] = 1; }
}

static nt_experts *population(gguf_file *host, decoder *m, bpe_tokenizer *tok) {
    nt_experts *p = nt_experts_open(host, NULL, NULL); assert(p);
    attach(p, host, m, tok);
    p->turns = 7; p->elections = 19;
    p->rng = UINT32_C(0xa0123456); p->consensus = .37f;
    for (int i = 0; i < HISTORY; i++) p->history[i] = (float)i / 32;
    for (int e = 0; e < SEATS; e++) {
        p->frequency[e] = 0;
        if (e >= 2) {
            p->alive[e] = p->vitality[e] = 0;
            if (p->seat_ancestor[e]) p->ancestors[p->seat_ancestor[e] - 1].death_turn = 1;
        }
    }
    memset(p->vote, 0, (size_t)SEATS * p->dim * sizeof(float));
    factors(p, 0);
    /* Disposable buffers still belong to the caller; even these must remain
     * byte-identical when a const-population measurement returns. */
    for (int i = 0; i < p->dim; i++) {
        p->sum_in[i] = .25f; p->sum_out[i] = -.5f; p->injected[i] = .75f;
    }
    for (size_t i = 0; i < (size_t)2 * p->dim * RANK + 1; i++) p->update[i] = -.125f;
    return p;
}

typedef struct {
    nt_experts value;
    nt_lineage_entry *lineage;
    float *buffers;
    size_t count;
} snapshot;

static size_t buffer_copy(const nt_experts *p, float *out, int compare) {
    const float *arrays[] = {p->a, p->b, p->vote, p->sum_in, p->sum_out, p->injected, p->update};
    size_t width = (size_t)p->dim, dr = width * RANK;
    size_t lengths[] = {SEATS * dr, SEATS * dr, SEATS * width, width, width, width, 2 * dr + 1};
    size_t at = 0;
    for (unsigned i = 0; i < sizeof(arrays) / sizeof(*arrays); i++) {
        if (out) {
            if (compare) CHECK(!memcmp(out + at, arrays[i], lengths[i] * sizeof(float)), "caller buffer preserved");
            else memcpy(out + at, arrays[i], lengths[i] * sizeof(float));
        }
        at += lengths[i];
    }
    return at;
}

static snapshot remember(const nt_experts *p) {
    snapshot s = {0}; memcpy(&s.value, p, sizeof(*p));
    s.count = buffer_copy(p, NULL, 0);
    s.buffers = malloc(s.count * sizeof(float));
    s.lineage = malloc((size_t)p->ancestor_count * sizeof(*p->ancestors));
    assert(s.buffers && s.lineage);
    buffer_copy(p, s.buffers, 0);
    memcpy(s.lineage, p->ancestors, (size_t)p->ancestor_count * sizeof(*p->ancestors));
    return s;
}

static void unchanged(const nt_experts *p, snapshot *s) {
    CHECK(!memcmp(p, &s->value, sizeof(*p)), "all caller fields, histories and clocks preserved");
    CHECK(!memcmp(p->ancestors, s->lineage, (size_t)p->ancestor_count * sizeof(*p->ancestors)), "lineage preserved");
    buffer_copy(p, s->buffers, 1);
    free(s->buffers); free(s->lineage);
}

static int close_to(double a, double b) { return fabs(a - b) < 1e-11; }

static nt_replay_episode *text(nt_experts *p, int token, int n) {
    int *ids = malloc((size_t)n * sizeof(int)); assert(ids);
    for (int i = 0; i < n; i++) ids[i] = token;
    nt_replay_episode *e = nt_replay_capture(p, ids, NULL, n, NULL, 0, NULL, 0);
    free(ids); assert(e); return e;
}

static nt_replay_measurement measure(nt_experts *p, const nt_replay_episode *e) {
    nt_replay_measurement out; memset(&out, 0xa5, sizeof(out));
    snapshot before = remember(p);
    CHECK(nt_experts_measure_replay(p, e, &out) == NT_OK, "measurement succeeds");
    unchanged(p, &before);
    CHECK(!strcmp(out.episode, nt_replay_id(e)), "exact episode address");
    CHECK(out.positions == (uint64_t)e->n, "whole prompt and no generated positions");
    CHECK(out.alive_mask == 3, "living mask");
    for (int i = 0; i < SEATS; i++) {
        nt_expert_identity id;
        if (i < 2) {
            assert(!nt_experts_seat_identity(p, i, &id));
            CHECK(!strcmp(out.identities[i], id.id), "living identity address");
        } else CHECK(!out.identities[i][0] && out.response[i] == 0, "vacant identities and response empty");
    }
    return out;
}

static int equal_files(const char *a, const char *b) {
    FILE *x = fopen(a, "rb"), *y = fopen(b, "rb"); assert(x && y);
    int left, right, same = 1;
    do { left = fgetc(x); right = fgetc(y); if (left != right) same = 0; } while (left != EOF && right != EOF);
    same = same && !ferror(x) && !ferror(y);
    assert(!fclose(x) && !fclose(y)); return same;
}

static void refuse(nt_experts *p, const nt_replay_episode *e, int expected) {
    nt_replay_measurement out, untouched; memset(&out, 0xa5, sizeof(out));
    memcpy(&untouched, &out, sizeof(out));
    snapshot before = remember(p);
    CHECK(nt_experts_measure_replay(p, e, &out) == expected, "invalid measurement refused");
    CHECK(!memcmp(&out, &untouched, sizeof(out)), "failed output is atomic");
    unchanged(p, &before);
}

int main(void) {
    gguf_file *host = gguf_open("source"); assert(host);
    host->ctx_len = 128;
    bpe_tokenizer *tok = bpe_load("source"); assert(tok);
    decoder model = {0};
    nt_experts *p = population(host, &model, tok);
    p->trace = tmpfile(); assert(p->trace);
    FILE *trace = p->trace;
    nt_replay_episode *unit = text(p, 0, 1), *other = text(p, 1, 1), *zero = text(p, 2, 1), *scaled = text(p, 3, 1);
    nt_replay_measurement u = measure(p, unit);
    CHECK(close_to(u.response[0], 4) && close_to(u.response[1], 0), "orthogonal coordinate responses");
    CHECK(close_to(u.mean_response, 2) && close_to(u.mean_disagreement, 1), "independent response/variance oracle");
    CHECK(close_to(u.gate_mass[0], .5) && close_to(u.gate_mass[1], .5), "actual uniform gates");
    CHECK(close_to(u.mean_max_gate, .5) && close_to(u.mean_entropy, log(2.0)) && close_to(u.mean_effective_seats, 2), "uniform entropy oracle");
    CHECK(u.mean_selected == 2 && u.mean_nonzero == 2, "selection counts");
    nt_replay_measurement s = measure(p, scaled);
    CHECK(s.mean_response == u.mean_response && s.mean_disagreement == u.mean_disagreement, "input-scale invariance");
    nt_replay_measurement v = measure(p, other);
    CHECK(v.response[0] == 0 && v.response[1] == 9 && v.mean_response == 4.5 && v.mean_disagreement == 2.25, "second coordinate independent oracle");
    nt_replay_measurement z = measure(p, zero);
    CHECK(z.mean_response == 0 && z.mean_disagreement == 0 && z.response[0] == 0 && z.response[1] == 0, "exact zero-input policy");
    for (int mode = 1; mode <= 2; mode++) {
        factors(p, mode); s = measure(p, unit);
        CHECK(s.mean_response == 4 && s.mean_disagreement == (mode == 1 ? 0 : 4), "identical/opposed proposal oracle");
    }
    factors(p, 0);
    p->vote[0] = 1000;
    s = measure(p, unit);
    CHECK(s.gate_mass[0] == 1 && s.gate_mass[1] == 0 && s.mean_max_gate == 1, "actual one-hot gate oracle");
    CHECK(s.mean_entropy == 0 && s.mean_effective_seats == 1 && s.mean_selected == 2 && s.mean_nonzero == 1, "one-hot entropy and selected/nonzero distinction");
    CHECK(s.mean_response == 2 && s.mean_disagreement == 1, "ungated response survives suppressed participation");
    p->vote[0] = 0;

    nt_replay_episode *long_text = text(p, 0, 70);
    int calls = model.calls;
    s = measure(p, long_text);
    CHECK(s.mean_response == 2 && s.mean_disagreement == 1 && model.calls - calls == 3, "length normalization and three full chunks");
    int ids[70] = {0}, at[] = {0, 31, 32, 63, 64, 69};
    float *row = calloc((size_t)p->dim, sizeof(float)); assert(row); row[1] = 2;
    const float *media[70] = {0};
    for (unsigned i = 0; i < sizeof(at) / sizeof(*at); i++) media[at[i]] = row;
    nt_replay_episode *mixed = nt_replay_capture(p, ids, media, 70, NULL, 0, NULL, 0); assert(mixed);
    row[1] = 19; free(row);
    int seen = model.media_rows;
    s = measure(p, mixed);
    CHECK(model.media_rows - seen == 6, "sparse media across chunk boundaries");
    CHECK(close_to(s.response[0], 256.0 / 70) && close_to(s.response[1], 54.0 / 70), "mixed-input independent energy oracle");
    CHECK(close_to(s.mean_response, 31.0 / 14) && close_to(s.mean_disagreement, 31.0 / 28), "mixed-input centered dispersion oracle");
    nt_replay_measurement again = measure(p, unit);
    CHECK(!memcmp(&u, &again, sizeof(u)), "candidate-order independence and fresh private history");
    for (int i = 0; i < 8; i++) {
        again = measure(p, unit);
        CHECK(!memcmp(&u, &again, sizeof(u)), "repeated reads are deterministic");
    }
    CHECK(model.heads == 0 && model.allocated == model.released, "no generation/head and every private cache freed");
    CHECK(ftell(trace) == 0 && !p->frame, "no live trace or calibration mutation");

    nt_replay_measurement invalid;
    CHECK(nt_experts_measure_replay(NULL, unit, &invalid) == NT_E_ARG, "null population");
    CHECK(nt_experts_measure_replay(p, NULL, &invalid) == NT_E_ARG, "null episode");
    CHECK(nt_experts_measure_replay(p, unit, NULL) == NT_E_ARG, "null output");
    p->count = 1; refuse(p, unit, NT_E_STATE); p->count = 0;
    p->observed = 1; refuse(p, unit, NT_E_STATE); p->observed = 0;
    p->choices = 1; refuse(p, unit, NT_E_STATE); p->choices = 0;
    char h = p->host[0]; p->host[0] = h == '0' ? '1' : '0'; refuse(p, unit, NT_E_STATE); p->host[0] = h;
    model.fail_pos = 69; refuse(p, long_text, NT_E_STATE);
    model.bad_input = 1; refuse(p, long_text, NT_E_STATE); model.bad_input = 0; model.fail_pos = -1;
    model.skip_observe = 1; refuse(p, unit, NT_E_STATE); model.skip_observe = 0;
    uint64_t turns = p->turns, elections = p->elections, replays = p->replay.cycles;
    p->turns = p->elections = p->replay.cycles = UINT64_MAX;
    again = measure(p, unit);
    CHECK(!memcmp(&u, &again, sizeof(u)), "read-only at persistent counter limits");
    p->turns = turns; p->elections = elections; p->replay.cycles = replays;
    CHECK(model.allocated == model.released && !model.heads, "failed reads free caches and request no logits");

    CHECK(!nt_experts_save(p, "measure-before"), "save pre-read state");
    again = measure(p, unit);
    CHECK(!nt_experts_save(p, "measure-after") && equal_files("measure-before", "measure-after"), "serialized state is unchanged");
    nt_experts *resumed = nt_experts_open(host, "measure-before", NULL); assert(resumed);
    attach(resumed, host, &model, tok);
    again = measure(resumed, unit);
    CHECK(!memcmp(&u, &again, sizeof(u)), "measurement survives exact restart");
    CHECK(!nt_experts_replay(p, unit, 2, 1), "serial learned replay after measurements");
    nt_consolidation *job = NULL;
    CHECK(!nt_experts_submit_replay(&resumed, &job, unit, 2) && !resumed && job, "worker accepts same chosen episode");
    CHECK(!nt_experts_await(&resumed, &job) && resumed && !job, "worker rejoins");
    CHECK(!nt_experts_save(p, "measure-serial") && !nt_experts_save(resumed, "measure-worker") &&
          equal_files("measure-serial", "measure-worker"), "readout leaves serial/worker learning byte-identical");
    CHECK(model.allocated == model.released, "all preview and replay caches freed");
    nt_experts_free(resumed);
    nt_replay_free(unit); nt_replay_free(other); nt_replay_free(zero); nt_replay_free(scaled);
    nt_replay_free(long_text); nt_replay_free(mixed);
    p->trace = NULL; fclose(trace); nt_experts_free(p); bpe_free(tok); gguf_close(host);
    printf("Replay measurement: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
