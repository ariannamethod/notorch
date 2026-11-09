/* Qwen3-Embedding encoder against a double-precision reference written from
 * modeling_qwen3.py: two layers, four query heads over two KV heads (so a head
 * mapped to the wrong group shows), per-head Q/K RMSNorm, NEOX RoPE with a base
 * small enough to turn every pair, SiLU gate, final RMSNorm, last-token pooling.
 * Weights are seeded pseudo-random F32; the embedding table is BF16 with exact
 * values, so the mapped row path is the one the checkpoint takes. Two properties
 * need no reference at all: right padding leaves the pooled state of a causal
 * encoder as it was, and left padding with raw positions changes nothing but
 * rounding, because RoPE scores depend on position differences only. */
#include "harness/arch.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { V = 7, E = 8, H = 4, KV = 2, HD = 4, F = 6, L = 2, MAXT = 8 };
#define EPS 1e-6f
#define BASE 100.0f

typedef struct { char name[64]; int rows, cols, rank; float *data; } tensor;
static tensor ts[64];
static int nts;
static unsigned rng = 12345u;
static float uniform(void) { rng = rng * 1664525u + 1013904223u; return (float)((rng >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
static float *add(const char *name, int rows, int cols, int rank, int norm) {
    tensor *t = &ts[nts++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->rows = rows; t->cols = cols; t->rank = rank;
    t->data = malloc((size_t)rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) t->data[i] = norm ? 1.0f + 0.25f * uniform() : 0.5f * uniform();
    return t->data;
}
static float *emb, *out_norm;
static struct { float *in, *post, *qn, *kn, *q, *k, *v, *o, *gate, *up, *down; } lw[L];
static void build(void) {
    char name[64];
    emb = add("embed_tokens.weight", V, E, 2, 0);
    for (int i = 0; i < V * E; i++) emb[i] = (float)((int)(uniform() * 16)) / 16.0f; /* exact in BF16 */
    out_norm = add("norm.weight", 1, E, 1, 1);
    for (int l = 0; l < L; l++) {
        #define T(dst, s, r, c, rank, nm) do { snprintf(name, sizeof(name), "layers.%d." s, l); lw[l].dst = add(name, r, c, rank, nm); } while (0)
        T(in, "input_layernorm.weight", 1, E, 1, 1); T(post, "post_attention_layernorm.weight", 1, E, 1, 1);
        T(qn, "self_attn.q_norm.weight", 1, HD, 1, 1); T(kn, "self_attn.k_norm.weight", 1, HD, 1, 1);
        T(q, "self_attn.q_proj.weight", H * HD, E, 2, 0); T(k, "self_attn.k_proj.weight", KV * HD, E, 2, 0);
        T(v, "self_attn.v_proj.weight", KV * HD, E, 2, 0); T(o, "self_attn.o_proj.weight", E, H * HD, 2, 0);
        T(gate, "mlp.gate_proj.weight", F, E, 2, 0); T(up, "mlp.up_proj.weight", F, E, 2, 0);
        T(down, "mlp.down_proj.weight", E, F, 2, 0);
        #undef T
    }
}
static void write_model(const char *path, const char *arch, int skip_tensor) {
    gguf_writer *w = gguf_write_open(path); assert(w);
    assert(!gguf_write_kv_str(w, "general.architecture", arch));
    const char *keys[] = {"block_count", "embedding_length", "vocab_size", "context_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length", "feed_forward_length"};
    int vals[] = {L, E, V, MAXT, H, KV, HD, F};
    char key[128];
    for (int i = 0; i < 8; i++) { snprintf(key, sizeof(key), "qwen3embedding.%s", keys[i]); assert(!gguf_write_kv_u32(w, key, (uint32_t)vals[i])); }
    assert(!gguf_write_kv_f32(w, "qwen3embedding.rms_epsilon", EPS));
    assert(!gguf_write_kv_f32(w, "qwen3embedding.rope.freq_base", BASE));
    for (int i = 0; i < nts; i++) {
        if (i == skip_tensor) continue;
        uint64_t dims[] = {(uint64_t)ts[i].cols, (uint64_t)ts[i].rows};
        assert(!gguf_write_tensor_decl(w, ts[i].name, (uint32_t)ts[i].rank, dims, i == 0 ? GGUF_TYPE_BF16 : GGUF_TYPE_F32));
    }
    for (int i = 0; i < nts; i++) {
        if (i == skip_tensor) continue;
        size_t count = (size_t)ts[i].rows * ts[i].cols;
        if (i == 0) {
            uint16_t *bf = malloc(count * sizeof(uint16_t));
            for (size_t j = 0; j < count; j++) { uint32_t b; memcpy(&b, &ts[i].data[j], 4); bf[j] = (uint16_t)(b >> 16); }
            assert(!gguf_write_tensor(w, ts[i].name, bf, count * sizeof(uint16_t)));
            free(bf);
        } else assert(!gguf_write_tensor_f32(w, ts[i].name, ts[i].data, count));
    }
    assert(!gguf_write_close(w));
}

/* ── the reference, in double ── */
static void ref_rms(double *y, const double *x, const float *w, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0 / sqrt(s / n + (double)EPS);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}
static void ref_lin(double *y, const float *w, const double *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) { double s = 0; for (int c = 0; c < cols; c++) s += (double)w[r * cols + c] * x[c]; y[r] = s; }
}
static void ref_rope(double *z, int pos) {
    for (int i = 0; i < HD / 2; i++) {
        double a = pos * pow((double)BASE, -2.0 * i / HD), x1 = z[i], x2 = z[i + HD / 2];
        z[i] = x1 * cos(a) - x2 * sin(a);
        z[i + HD / 2] = x2 * cos(a) + x1 * sin(a);
    }
}
static void reference(const int *ids, const unsigned char *mask, int n, double *pooled, double *unit, double *blocks) {
    static double x[MAXT][E], a[MAXT][E], q[MAXT][H * HD], k[MAXT][KV * HD], v[MAXT][KV * HD], at[MAXT][H * HD];
    for (int t = 0; t < n; t++) for (int j = 0; j < E; j++) x[t][j] = emb[ids[t] * E + j];
    for (int l = 0; l < L; l++) {
        for (int t = 0; t < n; t++) {
            ref_rms(a[t], x[t], lw[l].in, E);
            ref_lin(q[t], lw[l].q, a[t], H * HD, E); ref_lin(k[t], lw[l].k, a[t], KV * HD, E); ref_lin(v[t], lw[l].v, a[t], KV * HD, E);
            for (int h = 0; h < H; h++) { ref_rms(q[t] + h * HD, q[t] + h * HD, lw[l].qn, HD); ref_rope(q[t] + h * HD, t); }
            for (int h = 0; h < KV; h++) { ref_rms(k[t] + h * HD, k[t] + h * HD, lw[l].kn, HD); ref_rope(k[t] + h * HD, t); }
        }
        for (int t = 0; t < n; t++) for (int h = 0; h < H; h++) {
            int g = h / (H / KV);
            double s[MAXT], max = -INFINITY, total = 0;
            for (int j = 0; j <= t; j++) {
                s[j] = -INFINITY;
                if (mask && !mask[j]) continue;
                double d = 0;
                for (int z = 0; z < HD; z++) d += q[t][h * HD + z] * k[j][g * HD + z];
                s[j] = d / sqrt((double)HD);
                if (s[j] > max) max = s[j];
            }
            for (int j = 0; j <= t; j++) { s[j] = isinf(s[j]) ? 0 : exp(s[j] - max); total += s[j]; }
            for (int z = 0; z < HD; z++) {
                double acc = 0;
                for (int j = 0; j <= t; j++) acc += s[j] * v[j][g * HD + z];
                at[t][h * HD + z] = total > 0 ? acc / total : 0;
            }
        }
        for (int t = 0; t < n; t++) {
            double o[E], gt[F], up[F], dn[E];
            ref_lin(o, lw[l].o, at[t], E, H * HD);
            for (int j = 0; j < E; j++) x[t][j] += o[j];
            ref_rms(a[t], x[t], lw[l].post, E);
            ref_lin(gt, lw[l].gate, a[t], F, E); ref_lin(up, lw[l].up, a[t], F, E);
            for (int j = 0; j < F; j++) gt[j] = gt[j] / (1 + exp(-gt[j])) * up[j];
            ref_lin(dn, lw[l].down, gt, E, F);
            for (int j = 0; j < E; j++) { x[t][j] += dn[j]; if (l == 0) blocks[t * E + j] = x[t][j]; }
        }
    }
    int last = -1;
    for (int t = 0; t < n; t++) if (!mask || mask[t]) last = t;
    ref_rms(pooled, x[last], out_norm, E);
    double len = 0;
    for (int j = 0; j < E; j++) len += pooled[j] * pooled[j];
    for (int j = 0; j < E; j++) unit[j] = pooled[j] / sqrt(len);
}

static int failures;
static double worst;
static void close_to(const float *got, const double *want, int n, double tol, const char *what) {
    double d = 0;
    for (int i = 0; i < n; i++) if (fabs(got[i] - want[i]) > d) d = fabs(got[i] - want[i]);
    if (d > worst) worst = d;
    if (d > tol) { failures++; fprintf(stderr, "FAIL %s: max |delta| %.3g > %.1g\n", what, d, tol); }
}
typedef struct { float block[MAXT * E]; int calls, action; } capture;
static int cb(void *user, nt_encode_site site, int layer, int n, int width, float *x) {
    capture *c = user;
    c->calls++;
    assert(width == E);
    if (site == NT_ENCODE_BLOCK && layer == 0) {
        memcpy(c->block, x, (size_t)n * E * sizeof(float));
        if (c->action == 1) for (int t = 0; t < n; t++) x[t * E] += 0.5f;
        if (c->action == 2) return -77;
        if (c->action == 3) x[0] = NAN;
    }
    return 0;
}
int main(void) {
    char path[] = "/tmp/notorch-qe-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); close(fd);
    build();
    write_model(path, "embeddinggemma2", -1);
    gguf_file *gf = gguf_open(path); nt_encode_dims d;
    assert(gf && !nt_encoder_qwen3embedding.load(gf, &d)); gguf_close(gf);
    write_model(path, "qwen3embedding", 5);
    gf = gguf_open(path);
    assert(gf && !nt_encoder_qwen3embedding.load(gf, &d)); gguf_close(gf);
    write_model(path, "qwen3embedding", -1);
    gf = gguf_open(path); assert(gf);
    void *m = nt_encoder_qwen3embedding.load(gf, &d);
    assert(m && d.width == E && d.output_dim == E && d.n_layers == L && d.vocab == V && d.max_tokens == MAXT);

    const int ids[] = {3, 1, 6, 0, 5, 2};
    const unsigned char holes[] = {1, 0, 1, 1, 0, 0};    /* a masked key, and a masked tail */
    for (int variant = 0; variant < 2; variant++) {
        const unsigned char *mk = variant ? holes : NULL;
        float p[E], e[E], again[E];
        double rp[E], re[E], rb[MAXT * E];
        capture c = {0};
        reference(ids, mk, 6, rp, re, rb);
        assert(!nt_encoder_qwen3embedding.encode(m, ids, mk, 6, p, e, cb, &c));
        if (c.calls != 2 * L + 2) { failures++; fprintf(stderr, "FAIL hook calls %d\n", c.calls); }
        close_to(p, rp, E, 1e-5, "pooled");
        close_to(e, re, E, 1e-5, "normalized");
        close_to(c.block, rb, 6 * E, 1e-5, "block 0 residual");
        double sq = 0;
        for (int j = 0; j < E; j++) sq += (double)e[j] * e[j];
        if (fabs(sqrt(sq) - 1) > 2e-6) { failures++; fprintf(stderr, "FAIL unit norm %.3g\n", fabs(sqrt(sq) - 1)); }
        assert(!nt_encoder_qwen3embedding.encode(m, ids, mk, 6, NULL, again, NULL, NULL));
        if (memcmp(e, again, sizeof(e))) { failures++; fputs("FAIL repeat is not byte-identical\n", stderr); }
        c.action = 1;
        assert(!nt_encoder_qwen3embedding.encode(m, ids, mk, 6, NULL, again, cb, &c));
        if (fabsf(again[0] - e[0]) < 1e-4f) { failures++; fputs("FAIL hook write had no effect\n", stderr); }
        for (int action = 2; action <= 3; action++) {
            c.action = action;
            for (int j = 0; j < E; j++) p[j] = e[j] = 42;
            int rc = nt_encoder_qwen3embedding.encode(m, ids, mk, 6, p, e, cb, &c);
            assert(rc == (action == 2 ? -77 : NT_E_STATE));
            for (int j = 0; j < E; j++) assert(p[j] == 42 && e[j] == 42);
        }
    }
    /* right padding: rows before the pooled one never see the pads */
    {
        const int pad_ids[] = {3, 1, 6, 4, 4};
        const unsigned char right[] = {1, 1, 1, 0, 0};
        float a[E], b[E];
        assert(!nt_encoder_qwen3embedding.encode(m, ids, NULL, 3, NULL, a, NULL, NULL));
        assert(!nt_encoder_qwen3embedding.encode(m, pad_ids, right, 5, NULL, b, NULL, NULL));
        double ad[E];
        for (int j = 0; j < E; j++) ad[j] = a[j];
        close_to(b, ad, E, 1e-6, "right padding");
    }
    /* left padding, positions shifted by two: RoPE is relative */
    {
        const int pad_ids[] = {4, 4, 3, 1, 6};
        const unsigned char left[] = {0, 0, 1, 1, 1};
        float a[E], b[E];
        assert(!nt_encoder_qwen3embedding.encode(m, ids, NULL, 3, NULL, a, NULL, NULL));
        assert(!nt_encoder_qwen3embedding.encode(m, pad_ids, left, 5, NULL, b, NULL, NULL));
        double ad[E];
        for (int j = 0; j < E; j++) ad[j] = a[j];
        close_to(b, ad, E, 1e-5, "left padding");
    }
    float out[E];
    for (int j = 0; j < E; j++) out[j] = 42;
    const int bad_ids[] = {0, V};
    const unsigned char none[] = {0, 0, 0}, two[] = {1, 2, 1};
    assert(nt_encoder_qwen3embedding.encode(m, bad_ids, NULL, 2, NULL, out, NULL, NULL) == NT_E_TOKEN);
    assert(nt_encoder_qwen3embedding.encode(m, ids, none, 3, NULL, out, NULL, NULL) == NT_E_ARG);
    assert(nt_encoder_qwen3embedding.encode(m, ids, two, 3, NULL, out, NULL, NULL) == NT_E_ARG);
    assert(nt_encoder_qwen3embedding.encode(m, ids, NULL, MAXT + 1, NULL, out, NULL, NULL) == NT_E_CAPACITY);
    assert(nt_encoder_qwen3embedding.encode(m, ids, NULL, 0, NULL, out, NULL, NULL) == NT_E_ARG);
    for (int j = 0; j < E; j++) assert(out[j] == 42);
    nt_encoder_qwen3embedding.free(m);
    gguf_close(gf);
    unlink(path);
    for (int i = 0; i < nts; i++) free(ts[i].data);
    if (failures) { fprintf(stderr, "Qwen3-Embedding: %d FAILED\n", failures); return 1; }
    printf("Qwen3-Embedding: reference forward max |delta| %.3g (<= 1e-5), GQA, RoPE, causal mask, padding, "
           "hooks, refusals and failure atomicity PASS\n", worst);
    return 0;
}
