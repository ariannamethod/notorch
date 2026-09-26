/* Exercise the public residual API using a tiny, real qwen2 GGUF. No download. */
#include "harness/arch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { E = 4, V = 5, L = 2, FF = 8 };
static int failed, checks;
#define CHECK(c, name) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL: %s\n", name); failed++; } } while (0)
static const float shift[E] = { 0.2f, -0.3f, 0.4f, 0.1f };

typedef struct { char name[64]; int ndim, rows, cols; float data[FF*E]; } tensor;

static void put(tensor *t, const char *name, int rows, int cols, int norm) {
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->ndim = rows ? 2 : 1; t->rows = rows ? rows : 1; t->cols = cols;
    for (int i = 0; i < t->rows * cols; i++)
        t->data[i] = norm ? 1.0f : (float)((i * 7 + 3) % 17 - 8) * 0.03125f;
}

static int fixture(const char *path, int bias_width) {
    tensor ts[30]; int n = 0;
    put(&ts[n++], "token_embd.weight", V, E, 0);
    put(&ts[n++], "output_norm.weight", 0, E, 1);
    for (int l = 0; l < L; l++) {
        const char *fields[] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "attn_q.bias", "attn_k.bias", "attn_v.bias"};
        int rows[] = {0,E,2,2,E,0,FF,FF,E,0,0,0};
        int cols[] = {E,E,E,E,E,E,E,E,FF,E,2,2};
        for (int k = 0; k < 12; k++) {
            char name[64]; snprintf(name, sizeof(name), "blk.%d.%s", l, fields[k]);
            put(&ts[n++], name, rows[k], cols[k], k == 0 || k == 5);
        }
    }
    if (bias_width) {
        put(&ts[n], "blk.0.ffn_down.bias", 0, bias_width, 0);
        memcpy(ts[n++].data, shift, (size_t)bias_width * sizeof(float));
    }
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w, "general.architecture", "qwen2");
    gguf_write_kv_u32(w, "qwen2.block_count", L);
    gguf_write_kv_u32(w, "qwen2.embedding_length", E);
    gguf_write_kv_u32(w, "qwen2.feed_forward_length", FF);
    gguf_write_kv_u32(w, "qwen2.attention.head_count", 2);
    gguf_write_kv_u32(w, "qwen2.attention.head_count_kv", 1);
    gguf_write_kv_u32(w, "qwen2.context_length", 8);
    gguf_write_kv_f32(w, "qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
    for (int i = 0; i < n; i++) {
        uint64_t shape[] = {(uint64_t)ts[i].cols, (uint64_t)ts[i].rows};
        gguf_write_tensor_decl(w, ts[i].name, ts[i].ndim, shape, GGUF_TYPE_F32);
    }
    for (int i = 0; i < n; i++)
        gguf_write_tensor_f32(w, ts[i].name, ts[i].data,
                             (uint64_t)ts[i].rows * (uint64_t)ts[i].cols);
    return gguf_write_close(w);
}

typedef struct {
    int calls, count, start, bad, apply, target_layer, target_pos, fail_code;
    float last[E];
} probe;

static int observe(void *user, int layer, int pos0, int n, int width, float *x) {
    probe *p = user;
    if (layer != p->calls % L || width != E || n != p->count || pos0 != p->start)
        p->bad = 1;
    p->calls++;
    if (p->fail_code) return p->fail_code;
    if (p->apply && layer == p->target_layer)
        for (int j = 0; j < n; j++)
            if (p->target_pos < 0 || pos0 + j == p->target_pos)
                for (int i = 0; i < width; i++) x[j * width + i] += shift[i];
    if (layer == L - 1) memcpy(p->last, x + (n - 1) * E, sizeof(p->last));
    return NT_OK;
}

static int close_logits(const float *a, const float *b) {
    for (int i = 0; i < V; i++)
        if (!isfinite(a[i]) || !isfinite(b[i]) || fabsf(a[i] - b[i]) > 2e-6f) return 0;
    return 1;
}

int main(void) {
    char path[] = "/tmp/nt_residual_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    close(fd);
    char bias_path[128];
    snprintf(bias_path, sizeof(bias_path), "%s_bias", path);
    CHECK(fixture(path, 0) == 0, "write qwen2 fixture");
    gguf_file *gf = gguf_open(path);
    nt_dims dims;
    const nt_arch *arch = &nt_arch_llama;
    void *model = gf ? arch->load(gf, &dims) : NULL;
    if (!model) { unlink(path); return 1; }
    CHECK(arch->forward_residual != NULL, "family exposes residual interface");
    int ids[] = {1, 2, 3};
    float baseline[V], observed[V], disabled[V], steered[V], saved[V], expected[V];
    kv_cache *kv = kv_new(dims.n_layers, 8, dims.kv_dim);
    if (!kv) return 1;
    CHECK(arch->forward(model, kv, ids, 3, 0, baseline) == NT_OK, "plain forward");
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, disabled, NULL, NULL) == NT_OK
          && memcmp(baseline, disabled, sizeof(baseline)) == 0, "NULL callback byte parity");
    probe p = {.count = 3};
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, observed, observe, &p) == NT_OK
          && !p.bad && p.calls == L, "layer order, row count, width and position");
    CHECK(memcmp(baseline, observed, sizeof(baseline)) == 0, "observation byte parity");
    /* The callback sees the completed final block BEFORE output normalization. */
    float ss = 0;
    for (int i = 0; i < E; i++) ss += p.last[i] * p.last[i];
    for (int v = 0; v < V; v++) {
        expected[v] = 0;
        for (int i = 0; i < E; i++)
            expected[v] += ((float)(((v*E+i)*7+3)%17-8)*0.03125f)
                         * p.last[i] / sqrtf(ss/E + 1e-5f);
    }
    CHECK(close_logits(expected, observed), "observed residual predicts final logits");

    p = (probe){.count = 3, .apply = 1, .target_layer = 0, .target_pos = -1};
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, steered, observe, &p) == NT_OK
          && !p.bad && memcmp(baseline, steered, sizeof(baseline)) != 0,
          "mutation propagates into downstream layer and logits");
    CHECK(fixture(bias_path, E) == 0, "write persisted residual shift");
    gguf_file *bgf = gguf_open(bias_path);
    nt_dims bdims;
    void *bmodel = bgf ? arch->load(bgf, &bdims) : NULL;
    CHECK(bmodel != NULL, "load persisted bias");
    if (bmodel) {
        CHECK(arch->forward(bmodel, kv, ids, 3, 0, saved) == NT_OK &&
              memcmp(steered, saved, sizeof(saved)) == 0, "persisted bias matches hook byte for byte");
        arch->free(bmodel);
    }
    gguf_close(bgf);

    /* Position-selective intervention must mean the same thing in prefill and decode. */
    p = (probe){.count = 3, .apply = 1, .target_layer = 1, .target_pos = 2};
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, steered, observe, &p) == NT_OK,
          "select one absolute position in prefill");
    p = (probe){.count = 1, .apply = 1, .target_layer = 1, .target_pos = 2};
    int rc = NT_OK;
    for (int j = 0; j < 3; j++) {
        p.start = j;
        rc |= arch->forward_residual(model, kv, ids+j, 1, j, observed, observe, &p);
    }
    CHECK(rc == NT_OK && !p.bad && p.calls == 3*L && close_logits(steered, observed),
          "selected position agrees across prefill and decode");
    p = (probe){.count = 3, .fail_code = 77};
    for (int i = 0; i < V; i++) observed[i] = -123.0f;
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, observed, observe, &p) == 77
          && p.calls == 1, "callback error stops and propagates unchanged");
    int untouched = 1;
    for (int i = 0; i < V; i++) if (observed[i] != -123.0f) untouched = 0;
    CHECK(untouched, "callback failure leaves logits untouched");
    CHECK(arch->forward(model, kv, ids, 3, 0, observed) == NT_OK &&
          memcmp(baseline, observed, sizeof(baseline)) == 0, "plain restart retains no intervention");
    p = (probe){.count = 3};
    CHECK(arch->forward_residual(model, kv, ids, 3, 0, NULL, observe, &p) == NT_OK
          && p.calls == L, "collect residuals without output logits");
    CHECK(fixture(bias_path, E-1) == 0, "write malformed bias fixture");
    bgf = gguf_open(bias_path); bmodel = bgf ? arch->load(bgf, &bdims) : NULL;
    CHECK(bgf && !bmodel, "reject mismatched bias width");
    if (bmodel) arch->free(bmodel);
    gguf_close(bgf);
    kv_free(kv); arch->free(model); gguf_close(gf); unlink(path); unlink(bias_path);
    printf("RESIDUAL_%s (%d checks)\n", failed ? "FAIL" : "OK", checks);
    return failed ? 1 : 0;
}
