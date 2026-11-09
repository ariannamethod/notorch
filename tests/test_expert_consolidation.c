/* The locker holds the trace stream until submit returns. A synchronous
 * finish inside submit or an immediate join deadlocks and trips alarm(). */
#include "harness/experts.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    FILE *trace;
    int held[2], release[2];
} gate;

static void *hold_trace(void *arg) {
    gate *g = arg;
    char b;
    flockfile(g->trace);
    assert(write(g->held[1], "x", 1) == 1);
    assert(read(g->release[0], &b, 1) == 1);
    funlockfile(g->trace);
    return NULL;
}

static void observe(nt_experts *p, const gguf_file *host, float chosen) {
    int width = host->embed_dim, layer = host->n_layers / 2;
    float *x = malloc((size_t)width * sizeof(float));
    assert(x);
    for (int i = 0; i < width; i++) x[i] = (float)(i + 1) / width;
    assert(nt_experts_attend(p, layer, 0, 1, width, x) == NT_OK);
    assert(nt_experts_observe(p, layer, 0, 1, width, x) == NT_OK);
    nt_experts_choice(p, 1, chosen);
    free(x);
}

static unsigned char *bytes(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    assert(f && !fseek(f, 0, SEEK_END));
    long n = ftell(f);
    assert(n > 0);
    rewind(f);
    unsigned char *data = malloc((size_t)n);
    assert(data && fread(data, 1, (size_t)n, f) == (size_t)n);
    assert(!fclose(f));
    *size = (size_t)n;
    return data;
}

static void verify_transplants(gguf_file *host, nt_experts *recipient, nt_experts *donor) {
    assert(!nt_experts_save(recipient, "recipient.gguf"));
    assert(!nt_experts_save(donor, "donor.gguf"));
    gguf_file *dir = gguf_open("recipient.gguf");
    assert(dir);
    size_t n, dn;
    unsigned char *parent = bytes("recipient.gguf", &n), *other = bytes("donor.gguf", &dn);
    assert(n == dn);
    int changed = 0;
    for (int slot = 0; slot < 4; slot++) {
        nt_experts *dst = nt_experts_open(host, "recipient.gguf", NULL);
        assert(dst && !nt_experts_copy_factors(dst, donor, slot));
        char path[32];
        snprintf(path, sizeof(path), "seat%d.gguf", slot);
        assert(!nt_experts_save(dst, path));
        size_t hn;
        unsigned char *hybrid = bytes(path, &hn), *expected = malloc(n);
        assert(expected && hn == n);
        memcpy(expected, parent, n);
        for (int j = 0; j < 2; j++) {
            int idx = gguf_find_tensor(dir, j ? "doe.lora_b" : "doe.lora_a");
            assert(idx >= 0);
            gguf_tensor_info *t = &dir->tensors[idx];
            assert(t->ndim == 3 && t->shape[2] == 6);
            size_t width = t->shape[0] * t->shape[1] * sizeof(float);
            size_t offset = dir->data_offset + t->offset + slot * width;
            assert(offset + width <= n);
            memcpy(expected + offset, other + offset, width);
        }
        assert(!memcmp(expected, hybrid, n));
        changed += memcmp(parent, hybrid, n) != 0;
        free(expected);
        free(hybrid);
        nt_experts_free(dst);
    }
    assert(changed > 0);
    gguf_close(dir);
    free(parent);
    free(other);
}

int main(void) {
    alarm(20);
    gguf_file *host = gguf_open("host.gguf");
    int messages[2];
    assert(host && !pipe(messages));
    FILE *stream = fdopen(messages[1], "w"), *reader = fdopen(messages[0], "r");
    assert(stream && reader && !setvbuf(stream, NULL, _IONBF, 0));
    nt_experts *state = nt_experts_open(host, NULL, stream);
    nt_experts *other = nt_experts_open(host, NULL, NULL);
    assert(state && other);
    observe(state, host, 1);
    observe(other, host, -1);
    assert(nt_experts_copy_factors(state, other, 0) == NT_E_STATE);
    assert(!nt_experts_finish(other, 1));

    gate g = {.trace = stream};
    pthread_t locker;
    char b, line[512];
    assert(!pipe(g.held) && !pipe(g.release));
    assert(!pthread_create(&locker, NULL, hold_trace, &g));
    assert(read(g.held[0], &b, 1) == 1);
    nt_consolidation *job = NULL;
    assert(!nt_experts_submit(&state, &job) && !state && job);
    assert(nt_experts_submit(&other, &job) == NT_E_ARG && other);
    assert(nt_experts_save(state, "premature.gguf") != 0);
    assert(write(g.release[1], "x", 1) == 1 && !pthread_join(locker, NULL));
    /* Observe actual completion while the foreground still owns no state. */
    do { assert(fgets(line, sizeof(line), reader)); } while (!strstr(line, "doe turn=1 "));
    assert(!nt_experts_await(&state, &job) && state && !job);
    assert(!nt_experts_await(&state, &job));
    assert(nt_experts_copy_factors(state, other, 4) == NT_E_STATE);
    assert(nt_experts_copy_factors(state, other, -1) == NT_E_ARG);
    gguf_file *foreign = gguf_open("same-shape.gguf");
    assert(foreign);
    nt_experts *stranger = nt_experts_open(foreign, NULL, NULL);
    assert(stranger);
    observe(stranger, foreign, 1);
    assert(!nt_experts_finish(stranger, 1));
    assert(nt_experts_copy_factors(state, stranger, 0) == NT_E_STATE);
    nt_experts_free(stranger);
    gguf_close(foreign);
    verify_transplants(host, state, other);
    observe(state, host, NAN);
    assert(nt_experts_copy_factors(other, state, 0) == NT_E_STATE);
    assert(!nt_experts_submit(&state, &job));
    assert(nt_experts_await(&state, &job) == NT_E_STATE && !job && !state);
    assert(nt_experts_save(state, "failed.gguf") != 0);
    nt_experts_free(other);
    fclose(stream);
    fclose(reader);
    for (int i = 0; i < 2; i++) { close(g.held[i]); close(g.release[i]); }
    gguf_close(host);
    puts("background handoff, idle completion, failure and seat-local bytes: PASS");
    return 0;
}
