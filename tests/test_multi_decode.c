/* test_multi_decode.c — several sequences decoded in one forward, against each decoded alone.
 *
 * forward_multi promises identity, not closeness: row j of a multi-sequence step must leave
 * the same logits and the same cache as forward_residual on that sequence alone, bit for
 * bit, while the weights are read once for all rows. The fixture is a small real GGUF with
 * Q8_0 matrices, 64 wide so every matrix goes through the packed kernels and their batched
 * forms — the int8 path by default, the float path under NT_NO_I8 — once with the thread
 * fan-out forced on and once with it off, for a qwen2 body (attention biases), a qwen3
 * body (per-head q/k norms) and a gemma3 body (sliding-window layers). Three sequences of different lengths decode greedily; the
 * multi-sequence run feeds the tokens the single runs chose, so any difference is the
 * forward's. A per-row steering callback with its own user pointer runs both ways, and the
 * refusals are checked for leaving the logits unwritten. */
#include "tests/decode_fixtures.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failed, checks;
#define CHECK(c, name) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL: %s\n", name); failed++; } } while (0)

typedef struct { float scale; int calls, bad; } steer;

/* Adds scale * (i-th unit ramp) at layer 0; checks that per-row calls come one row at a time. */
static int push(void *user, int layer, int pos0, int n, int width, float *x) {
    steer *s = user;
    (void)pos0;
    s->calls++;
    if (width != E) s->bad = 1;
    if (layer == 0)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < width; i++) x[j * width + i] += s->scale * (float)(i % 5 - 2);
    return NT_OK;
}

static int argmax(const float *x) {
    int b = 0; for (int i = 1; i < V; i++) if (x[i] > x[b]) b = i;
    return b;
}

static const int prompts[SEQ][5] = {{1, 7, 3}, {9, 2, 30, 4, 11}, {5, 5}};
static const int lengths[SEQ] = {3, 5, 2};

/* One configuration: decode every sequence alone, then all together, and compare. */
static void run(const nt_arch *arch, void *model, nt_dims dims, int steered, const char *label) {
    char what[160];
    float alone[SEQ][STEPS + 1][V], multi[SEQ][V];
    int chosen[SEQ][STEPS];
    kv_cache *solo[SEQ], *batch[SEQ];
    steer ss[SEQ], sm[SEQ];
    for (int s = 0; s < SEQ; s++) {
        ss[s] = (steer){0.05f * (float)(s + 1), 0, 0}; sm[s] = ss[s];
        solo[s] = kv_new(dims.n_layers, CAP, dims.kv_dim);
        batch[s] = kv_new(dims.n_layers, CAP, dims.kv_dim);
        if (!solo[s] || !batch[s]) { CHECK(0, "allocate caches"); return; }
        int rc = arch->forward_residual(model, solo[s], prompts[s], lengths[s], 0, alone[s][0],
                                        steered ? push : NULL, &ss[s]);
        rc |= arch->forward_residual(model, batch[s], prompts[s], lengths[s], 0, multi[s],
                                     steered ? push : NULL, &sm[s]);
        CHECK(rc == NT_OK, "prefill");
        for (int t = 0; t < STEPS; t++) {
            chosen[s][t] = argmax(alone[s][t]);
            rc = arch->forward_residual(model, solo[s], &chosen[s][t], 1, lengths[s] + t,
                                        alone[s][t + 1], steered ? push : NULL, &ss[s]);
            CHECK(rc == NT_OK, "single decode");
        }
    }
    int same = 1, users_ok = 1;
    void *users[SEQ];
    for (int s = 0; s < SEQ; s++) users[s] = &sm[s];
    for (int t = 0; t < STEPS; t++) {
        int tokens[SEQ], pos[SEQ];
        for (int s = 0; s < SEQ; s++) { tokens[s] = chosen[s][t]; pos[s] = lengths[s] + t; }
        int before[SEQ];
        for (int s = 0; s < SEQ; s++) before[s] = sm[s].calls;
        int rc = arch->forward_multi(model, batch, tokens, pos, SEQ, &multi[0][0],
                                     steered ? push : NULL, steered ? users : NULL);
        CHECK(rc == NT_OK, "multi-sequence decode");
        for (int s = 0; s < SEQ; s++) {
            same &= memcmp(multi[s], alone[s][t + 1], sizeof(multi[s])) == 0;
            if (steered) users_ok &= sm[s].calls - before[s] == dims.n_layers && !sm[s].bad;
        }
    }
    int caches = 1;
    for (int s = 0; s < SEQ; s++) {
        size_t bytes = (size_t)dims.n_layers * CAP * dims.kv_dim * sizeof(float);
        caches &= !memcmp(solo[s]->k, batch[s]->k, bytes) && !memcmp(solo[s]->v, batch[s]->v, bytes);
    }
    snprintf(what, sizeof(what), "%s: multi-sequence logits equal single decode bit for bit", label);
    CHECK(same, what);
    snprintf(what, sizeof(what), "%s: multi-sequence caches equal single decode bit for bit", label);
    CHECK(caches, what);
    if (steered) {
        snprintf(what, sizeof(what), "%s: per-row callback, own user, one row per call", label);
        CHECK(users_ok, what);
    }
    for (int s = 0; s < SEQ; s++) { kv_free(solo[s]); kv_free(batch[s]); }
}

static void refusals(const nt_arch *arch, void *model, nt_dims dims) {
    kv_cache *a = kv_new(dims.n_layers, CAP, dims.kv_dim), *b = kv_new(dims.n_layers, CAP, dims.kv_dim);
    kv_cache *shared[2] = {a, a}, *pair[2] = {a, b};
    int tokens[2] = {1, 2}, bad[2] = {1, V}, pos[2] = {0, 0}, far[2] = {0, CAP};
    float logits[2][V], sentinel[2][V];
    for (int i = 0; i < V; i++) logits[0][i] = logits[1][i] = sentinel[0][i] = sentinel[1][i] = 1234.5f;
    steer s = {0.1f, 0, 0};
    CHECK(arch->forward_multi(model, shared, tokens, pos, 2, &logits[0][0], NULL, NULL) == NT_E_ARG,
          "a cache shared by two rows is refused");
    CHECK(arch->forward_multi(model, pair, bad, pos, 2, &logits[0][0], NULL, NULL) == NT_E_TOKEN,
          "a token outside the vocabulary is refused");
    CHECK(arch->forward_multi(model, pair, tokens, far, 2, &logits[0][0], NULL, NULL) == NT_E_CAPACITY,
          "a position beyond the cache is refused");
    CHECK(arch->forward_multi(model, pair, tokens, pos, 2, &logits[0][0], push, NULL) == NT_E_ARG,
          "a callback without per-row users is refused");
    (void)s;
    CHECK(memcmp(logits, sentinel, sizeof(logits)) == 0, "refusals leave the logits unwritten");
    kv_free(a); kv_free(b);
}

int main(void) {
    const char *families[] = {"qwen2", "qwen3", "gemma3"};
    for (int family = 0; family < 3; family++) {
        const nt_arch *arch = family == 2 ? &nt_arch_gemma3 : &nt_arch_llama;
        char path[] = "/tmp/nt_multi_XXXXXX";
        int fd = mkstemp(path);
        if (fd < 0) return 1;
        close(fd);
        CHECK((family == 2 ? fixture_gemma3(path) : fixture(path, family == 1)) == 0, "write Q8_0 fixture");
        for (int noi8 = 0; noi8 < 2; noi8++) {
            if (noi8) setenv("NT_NO_I8", "1", 1); else unsetenv("NT_NO_I8");
            gguf_file *gf = gguf_open(path);
            nt_dims dims;
            void *model = gf ? arch->load(gf, &dims) : NULL;
            CHECK(model != NULL, "load Q8_0 fixture");
            if (!model) { gguf_close(gf); continue; }
            CHECK(arch->forward_multi != NULL, "family exposes multi-sequence decode");
            for (int fan = 0; fan < 2; fan++) {
                nt_qmv_set_thread_min(fan ? 1 : LONG_MAX);
                for (int steered = 0; steered < 2; steered++) {
                    char label[96];
                    snprintf(label, sizeof(label), "%s %s %s %s", families[family],
                             noi8 ? "NT_NO_I8" : "int8", fan ? "threaded" : "one thread",
                             steered ? "steered" : "plain");
                    run(arch, model, dims, steered, label);
                }
            }
            refusals(arch, model, dims);
            arch->free(model);
            gguf_close(gf);
        }
        unlink(path);
    }
    unsetenv("NT_NO_I8");
    if (failed) { fprintf(stderr, "MULTI_DECODE FAILED (%d of %d)\n", failed, checks); return 1; }
    printf("MULTI_DECODE_OK (%d checks)\n", checks);
    return 0;
}
