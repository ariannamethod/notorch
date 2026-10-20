/* Native trainable word model. See spa_text_body.h for the host boundary. */
#include "spa_text_body.h"
#include "notorch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TEXT_DIM 16u
#define TEXT_VOCAB 512u
#define TEXT_TOKENS 4096u
struct spa_text_body {
    uint32_t vocabulary;
    uint64_t rng;
    nt_tensor *embedding, *output;
};

static int valid(const spa_text_body *b) {
    if (!b || b->vocabulary < 2 || b->vocabulary > TEXT_VOCAB ||
        !b->embedding || !b->output) return 0;
    for (size_t i = 0; i < (size_t)b->vocabulary * TEXT_DIM; ++i)
        if (!isfinite(b->embedding->data[i]) || !isfinite(b->output->data[i])) return 0;
    return 1;
}

void spa_text_free(spa_text_body *b) {
    if (b) { nt_tensor_free(b->embedding); nt_tensor_free(b->output); free(b); }
}

spa_text_body *spa_text_create(uint32_t vocabulary, uint64_t seed) {
    if (vocabulary < 2 || vocabulary > TEXT_VOCAB) return NULL;
    spa_text_body *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->vocabulary = vocabulary;
    b->rng = seed;
    b->embedding = nt_tensor_new2d((int)vocabulary, TEXT_DIM);
    b->output = nt_tensor_new2d((int)vocabulary, TEXT_DIM);
    if (!b->embedding || !b->output) { spa_text_free(b); return NULL; }
    size_t n = (size_t)vocabulary * TEXT_DIM;
    if (nt_rng_normal_values(&b->rng, (int)n, b->embedding->data) ||
        nt_rng_normal_values(&b->rng, (int)n, b->output->data)) {
        spa_text_free(b); return NULL;
    }
    for (size_t i = 0; i < n; ++i) {
        b->embedding->data[i] *= .25f;
        b->output->data[i] *= .25f;
    }
    return b;
}

uint32_t spa_text_vocabulary(const spa_text_body *b) { return b ? b->vocabulary : 0; }
uint32_t spa_text_dimension(void) { return TEXT_DIM; }

int spa_text_train(spa_text_body *b, const uint32_t *tokens, size_t count,
                   uint32_t epochs, float rate, float losses[2]) {
    if (!valid(b) || !tokens || !losses || count < 2 || count > TEXT_TOKENS ||
        epochs < 1 || epochs > 10000 || !isfinite(rate) || rate <= 0 || rate > 4 ||
        nt_tape_is_active()) return -1;
    for (size_t i = 0; i < count; ++i) if (tokens[i] >= b->vocabulary) return -1;
    size_t n = (size_t)b->vocabulary * TEXT_DIM;
    nt_tensor *e = nt_tensor_new2d((int)b->vocabulary, TEXT_DIM);
    nt_tensor *w = nt_tensor_new2d((int)b->vocabulary, TEXT_DIM);
    nt_tensor *x = nt_tensor_new(count - 1), *y = nt_tensor_new(count - 1);
    int status = -1;
    float first = 0, last = 0;
    if (!e || !w || !x || !y) goto done;
    memcpy(e->data, b->embedding->data, n * sizeof(float));
    memcpy(w->data, b->output->data, n * sizeof(float));
    for (size_t i = 0; i < count - 1; ++i) {
        x->data[i] = (float)tokens[i]; y->data[i] = (float)tokens[i + 1];
    }
    for (uint32_t step = 0; step <= epochs; ++step) {
        nt_tape_start();
        int ei = nt_tape_param(e), wi = nt_tape_param(w);
        int xi = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
        int yi = nt_tape_record(y, NT_OP_NONE, -1, -1, 0);
        if (ei < 0 || wi < 0 || xi < 0 || yi < 0) goto done;
        int h = nt_seq_embedding(ei, -1, xi, (int)count - 1, TEXT_DIM);
        if (h < 0) goto done;
        h = nt_tanh(h);
        if (h < 0) goto done;
        int z = nt_seq_linear(wi, h, (int)count - 1);
        if (z < 0) goto done;
        int loss = nt_seq_cross_entropy(z, yi, (int)count - 1, (int)b->vocabulary);
        if (loss < 0) goto done;
        last = nt_tape_get()->entries[loss].output->data[0];
        if (!isfinite(last)) goto done;
        if (!step) first = last;
        if (step < epochs) { nt_tape_backward(loss); nt_tape_clip_grads(1.f); nt_tape_sgd_step(rate); }
        nt_tape_clear();
    }
    for (size_t i = 0; i < n; ++i)
        if (!isfinite(e->data[i]) || !isfinite(w->data[i])) goto done;
    memcpy(b->embedding->data, e->data, n * sizeof(float));
    memcpy(b->output->data, w->data, n * sizeof(float));
    losses[0] = first; losses[1] = last; status = 0;
done:
    nt_tape_clear();
    nt_tensor_free(e); nt_tensor_free(w); nt_tensor_free(x); nt_tensor_free(y);
    return status;
}

int spa_text_embeddings(const spa_text_body *b, float *out, size_t capacity) {
    if (!valid(b) || !out || capacity != (size_t)b->vocabulary * TEXT_DIM) return -1;
    memcpy(out, b->embedding->data, capacity * sizeof(float));
    return 0;
}

int spa_text_generate(spa_text_body *b, uint32_t context, uint32_t end,
                      float temperature, uint32_t *out, size_t capacity) {
    if (!valid(b) || !out || context >= b->vocabulary || end >= b->vocabulary ||
        capacity < 1 || capacity > 64 || !isfinite(temperature) ||
        temperature <= 0 || temperature > 16) return -1;
    uint64_t rng = b->rng;
    uint32_t result[64];
    size_t length = 0;
    while (length < capacity) {
        float hidden[TEXT_DIM], logits[TEXT_VOCAB], weights[TEXT_VOCAB], bias[TEXT_VOCAB] = {0};
        if (nt_tanh_values(b->embedding->data + context * TEXT_DIM, TEXT_DIM, hidden) ||
            nt_linear_values(b->output->data, bias, hidden, (int)b->vocabulary, TEXT_DIM, logits)) return -1;
        float maximum = logits[0];
        for (uint32_t i = 1; i < b->vocabulary; ++i) if (logits[i] > maximum) maximum = logits[i];
        for (uint32_t i = 0; i < b->vocabulary; ++i) weights[i] = expf(logits[i] - maximum);
        int next;
        if (nt_rng_categorical(&rng, weights, (int)b->vocabulary, temperature, &next)) return -1;
        result[length++] = (uint32_t)next;
        if ((uint32_t)next == end) break;
        context = (uint32_t)next;
    }
    memcpy(out, result, length * sizeof(*out)); b->rng = rng;
    return (int)length;
}

static void put32(unsigned char *p, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i) p[i] = (unsigned char)(x >> (8 * i));
}
static uint32_t get32(const unsigned char *p) {
    uint32_t x = 0; for (unsigned i = 0; i < 4; ++i) x |= (uint32_t)p[i] << (8 * i); return x;
}
static void put64(unsigned char *p, uint64_t x) {
    for (unsigned i = 0; i < 8; ++i) p[i] = (unsigned char)(x >> (8 * i));
}
static uint64_t get64(const unsigned char *p) {
    uint64_t x = 0; for (unsigned i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i); return x;
}
static uint64_t checksum(const unsigned char *p, size_t n) {
    uint64_t x = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) { x ^= p[i]; x *= UINT64_C(1099511628211); } return x;
}
size_t spa_text_snapshot_size(const spa_text_body *b) {
    return valid(b) ? 32 + (size_t)b->vocabulary * TEXT_DIM * 8 : 0;
}
int spa_text_snapshot(const spa_text_body *b, void *out, size_t capacity) {
    size_t size = spa_text_snapshot_size(b);
    if (!size || !out || capacity != size) return -1;
    unsigned char *p = out;
    memcpy(p, "SPATXT01", 8); put32(p + 8, b->vocabulary); put32(p + 12, TEXT_DIM); put64(p + 16, b->rng);
    size_t n = (size_t)b->vocabulary * TEXT_DIM;
    for (size_t i = 0; i < 2 * n; ++i) {
        float f = i < n ? b->embedding->data[i] : b->output->data[i - n];
        uint32_t bits; memcpy(&bits, &f, 4); put32(p + 24 + i * 4, bits);
    }
    put64(p + size - 8, checksum(p, size - 8)); return 0;
}
spa_text_body *spa_text_restore(const void *data, size_t size) {
    if (!data || size < 32 || sizeof(float) != 4) return NULL;
    const unsigned char *p = data;
    uint32_t vocabulary = get32(p + 8);
    if (memcmp(p, "SPATXT01", 8) || vocabulary < 2 || vocabulary > TEXT_VOCAB ||
        get32(p + 12) != TEXT_DIM || size != 32 + (size_t)vocabulary * TEXT_DIM * 8 ||
        get64(p + size - 8) != checksum(p, size - 8)) return NULL;
    size_t n = (size_t)vocabulary * TEXT_DIM;
    for (size_t i = 0; i < 2 * n; ++i) {
        uint32_t bits = get32(p + 24 + i * 4); float f; memcpy(&f, &bits, 4);
        if (!isfinite(f)) return NULL;
    }
    spa_text_body *b = spa_text_create(vocabulary, 1);
    if (!b) return NULL;
    b->rng = get64(p + 16);
    for (size_t i = 0; i < 2 * n; ++i) {
        uint32_t bits = get32(p + 24 + i * 4); float f; memcpy(&f, &bits, 4);
        if (i < n) b->embedding->data[i] = f; else b->output->data[i - n] = f;
    }
    return b;
}
