/* test_cuda_decode.c — the CUDA decoder against the CPU decoder it mirrors.
 *
 * The fixtures of test_multi_decode (a qwen2, a qwen3 and a gemma3 body, Q8_0 matrices 64
 * wide) and the CPU forward under NT_NO_I8, whose Q8_0 dot the device kernel reproduces in
 * the same order. Three sequences are prefilled and decoded greedily on both, the GPU fed
 * the tokens the CPU chose: every step's argmax must agree and every logit must stay within
 * TOL of the CPU's; the largest difference seen is printed. On the GPU the claims are exact:
 * a second run repeats every logit bit for bit, and a multi-sequence step leaves each row
 * the logits and the cache a single decode leaves it. A per-row steering callback runs on
 * both devices, and the refusals — a host cache handed to the device among them — leave
 * the logits unwritten. */
#include "tests/decode_fixtures.h"
#include "harness/cuda_decode.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TOL 1e-4f
static int failed, checks;
static float worst;
#define CHECK(c, name) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL: %s\n", name); failed++; } } while (0)

typedef struct { float scale; int calls, bad; } steer;

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

static int same_cache(const kv_cache *a, const kv_cache *b, nt_dims dims) {
    size_t n = (size_t)dims.n_layers * CAP * dims.kv_dim;
    float *ha = malloc(4 * n * sizeof(float));
    if (!ha) return 0;
    int ok = cudaMemcpy(ha, a->k, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(ha + n, a->v, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(ha + 2 * n, b->k, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(ha + 3 * n, b->v, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             !memcmp(ha, ha + 2 * n, 2 * n * sizeof(float));
    free(ha);
    return ok;
}

static const int prompts[SEQ][5] = {{1, 7, 3}, {9, 2, 30, 4, 11}, {5, 5}};
static const int lengths[SEQ] = {3, 5, 2};

static void run(const nt_arch *cpu, void *cm, const nt_arch *gpu, void *gm, nt_dims dims,
                int steered, const char *label) {
    char what[160];
    float host[SEQ][STEPS + 1][V], dev[SEQ][STEPS + 1][V], again[SEQ][STEPS + 1][V], multi[SEQ][V];
    int chosen[SEQ][STEPS];
    kv_cache *h[SEQ], *d[SEQ], *d2[SEQ], *b[SEQ];
    steer sh[SEQ], sd[SEQ], sd2[SEQ], sb[SEQ];
    nt_residual_fn fn = steered ? push : NULL;
    int agree = 1, close = 1, repeat = 1;
    for (int s = 0; s < SEQ; s++) {
        sh[s] = (steer){0.05f * (float)(s + 1), 0, 0}; sd[s] = sd2[s] = sb[s] = sh[s];
        h[s] = kv_new(dims.n_layers, CAP, dims.kv_dim);
        d[s] = gpu->kv_new(gm, CAP); d2[s] = gpu->kv_new(gm, CAP); b[s] = gpu->kv_new(gm, CAP);
        if (!h[s] || !d[s] || !d2[s] || !b[s]) { CHECK(0, "allocate caches"); return; }
        int rc = cpu->forward_residual(cm, h[s], prompts[s], lengths[s], 0, host[s][0], fn, &sh[s]);
        rc |= gpu->forward_residual(gm, d[s], prompts[s], lengths[s], 0, dev[s][0], fn, &sd[s]);
        rc |= gpu->forward_residual(gm, d2[s], prompts[s], lengths[s], 0, again[s][0], fn, &sd2[s]);
        rc |= gpu->forward_residual(gm, b[s], prompts[s], lengths[s], 0, multi[s], fn, &sb[s]);
        CHECK(rc == NT_OK, "prefill");
        for (int t = 0; t < STEPS; t++) {
            chosen[s][t] = argmax(host[s][t]);
            int p = lengths[s] + t;
            rc = cpu->forward_residual(cm, h[s], &chosen[s][t], 1, p, host[s][t + 1], fn, &sh[s]);
            rc |= gpu->forward_residual(gm, d[s], &chosen[s][t], 1, p, dev[s][t + 1], fn, &sd[s]);
            rc |= gpu->forward_residual(gm, d2[s], &chosen[s][t], 1, p, again[s][t + 1], fn, &sd2[s]);
            CHECK(rc == NT_OK, "decode");
        }
        for (int t = 0; t <= STEPS; t++) {
            agree &= argmax(dev[s][t]) == argmax(host[s][t]);
            for (int i = 0; i < V; i++) {
                float diff = fabsf(dev[s][t][i] - host[s][t][i]);
                if (!(diff <= TOL)) close = 0;
                if (diff > worst) worst = diff;
            }
        }
        repeat &= !memcmp(dev[s], again[s], sizeof(dev[s]));
    }
    snprintf(what, sizeof(what), "%s: every argmax agrees with the CPU", label);
    CHECK(agree, what);
    snprintf(what, sizeof(what), "%s: every logit within %g of the CPU", label, (double)TOL);
    CHECK(close, what);
    snprintf(what, sizeof(what), "%s: a second GPU run repeats every logit bit for bit", label);
    CHECK(repeat, what);

    int same = 1, users_ok = 1;
    void *users[SEQ];
    for (int s = 0; s < SEQ; s++) users[s] = &sb[s];
    for (int t = 0; t < STEPS; t++) {
        int tokens[SEQ], pos[SEQ], before[SEQ];
        for (int s = 0; s < SEQ; s++) { tokens[s] = chosen[s][t]; pos[s] = lengths[s] + t; before[s] = sb[s].calls; }
        int rc = gpu->forward_multi(gm, b, tokens, pos, SEQ, &multi[0][0], fn, steered ? users : NULL);
        CHECK(rc == NT_OK, "multi-sequence decode");
        for (int s = 0; s < SEQ; s++) {
            same &= !memcmp(multi[s], dev[s][t + 1], sizeof(multi[s]));
            if (steered) users_ok &= sb[s].calls - before[s] == dims.n_layers && !sb[s].bad;
        }
    }
    int caches = 1;
    for (int s = 0; s < SEQ; s++) caches &= same_cache(d[s], b[s], dims);
    snprintf(what, sizeof(what), "%s: GPU multi-sequence logits equal single decode bit for bit", label);
    CHECK(same, what);
    snprintf(what, sizeof(what), "%s: GPU multi-sequence caches equal single decode bit for bit", label);
    CHECK(caches, what);
    if (steered) {
        snprintf(what, sizeof(what), "%s: per-row callback on the GPU, own user, one row per call", label);
        CHECK(users_ok, what);
        int calls = 1;
        for (int s = 0; s < SEQ; s++) calls &= sd[s].calls == sh[s].calls && !sd[s].bad;
        snprintf(what, sizeof(what), "%s: the GPU calls the hook as often as the CPU", label);
        CHECK(calls, what);
    }
    for (int s = 0; s < SEQ; s++) {
        kv_free(h[s]); gpu->kv_free(gm, d[s]); gpu->kv_free(gm, d2[s]); gpu->kv_free(gm, b[s]);
    }
}

static void refusals(const nt_arch *gpu, void *gm, nt_dims dims) {
    kv_cache *a = gpu->kv_new(gm, CAP), *b = gpu->kv_new(gm, CAP);
    kv_cache *host = kv_new(dims.n_layers, CAP, dims.kv_dim);
    kv_cache *shared[2] = {a, a}, *pair[2] = {a, b}, *mixed[2] = {a, host};
    int tokens[2] = {1, 2}, bad[2] = {1, V}, pos[2] = {0, 0}, far[2] = {0, CAP};
    float logits[2][V], sentinel[2][V];
    for (int i = 0; i < V; i++) logits[0][i] = logits[1][i] = sentinel[0][i] = sentinel[1][i] = 1234.5f;
    CHECK(a && b && host, "allocate refusal caches");
    if (!a || !b || !host) return;
    CHECK(gpu->forward_multi(gm, shared, tokens, pos, 2, &logits[0][0], NULL, NULL) == NT_E_ARG,
          "a cache shared by two rows is refused");
    CHECK(gpu->forward_multi(gm, pair, bad, pos, 2, &logits[0][0], NULL, NULL) == NT_E_TOKEN,
          "a token outside the vocabulary is refused");
    CHECK(gpu->forward_multi(gm, pair, tokens, far, 2, &logits[0][0], NULL, NULL) == NT_E_CAPACITY,
          "a position beyond the cache is refused");
    CHECK(gpu->forward_multi(gm, pair, tokens, pos, 2, &logits[0][0], push, NULL) == NT_E_ARG,
          "a callback without per-row users is refused");
    CHECK(gpu->forward_multi(gm, mixed, tokens, pos, 2, &logits[0][0], NULL, NULL) == NT_E_CACHE,
          "a host cache in a multi-sequence step is refused");
    CHECK(gpu->forward_residual(gm, host, tokens, 2, 0, &logits[0][0], NULL, NULL) == NT_E_CACHE,
          "a host cache in a prefill is refused");
    CHECK(memcmp(logits, sentinel, sizeof(logits)) == 0, "refusals leave the logits unwritten");
    CHECK(gpu->kv_new(gm, 0) == NULL, "an empty device cache is refused");
    CHECK(gpu->kv_new(gm, 1 << 20) == NULL, "a device cache beyond the RoPE table is refused");
    gpu->kv_free(gm, a); gpu->kv_free(gm, b); kv_free(host);
}

/* The device Q8_0 matmul against qmm under NT_NO_I8, bit for bit: the kernel claims the CPU
 * kernel's order — 32 partial sums per block position, (d*w) then a fused add, the same fold
 * — and a claim of order is only checked by equality. 3584 columns (112 blocks, the vector
 * path) and 3680 (115 blocks, the byte path), group sizes on both sides of the 8-row tile.
 * Then the row whose partial sums overflow where the sequential sum does not: the CPU
 * retries it in the sequential order, and so must the GPU. */
static void matmul_exact(void) {
    enum { R = 45, KMAX = 3680, N = 11 };
    float *w = malloc((size_t)R * KMAX * sizeof(float)), *x = malloc((size_t)N * KMAX * sizeof(float));
    float *cpu = malloc((size_t)N * R * sizeof(float)), *gpu = malloc((size_t)N * R * sizeof(float));
    uint8_t *packed = malloc((size_t)R * (KMAX / 32) * 34);
    if (!w || !x || !cpu || !gpu || !packed) { CHECK(0, "allocate matmul buffers"); return; }
    const int widths[] = {3584, 3680}, groups[] = {1, 3, 8, 11};
    int ok = 1;
    for (int s = 0; s < 2; s++) {
        int K = widths[s];
        for (long i = 0; i < (long)R * K; i++) w[i] = sinf((float)i * 0.37f) * (1.0f + (float)(i % 13));
        for (long i = 0; i < (long)N * K; i++) x[i] = cosf((float)i * 0.11f) * 0.1f;
        for (int r = 0; r < R; r++)
            ok &= nt_quantize_row(w + (size_t)r * K, packed + (size_t)r * (K / 32) * 34, K, GGUF_TYPE_Q8_0) == 0;
        wt m = {packed, NULL, GGUF_TYPE_Q8_0, R, K, 0};
        int same = 1;
        for (int g = 0; ok && g < 4; g++) {
            qmm(cpu, &m, x, groups[g]);
            same &= nt_cuda_qmm(gpu, &m, x, groups[g]) == NT_OK &&
                    !memcmp(cpu, gpu, (size_t)groups[g] * R * sizeof(float));
        }
        char what[96];
        snprintf(what, sizeof(what), "device Q8_0 matmul equals qmm bit for bit, %d columns, 1 to 11 rows", K);
        CHECK(ok && same, what);
    }

    /* Every weight 2^21 quantizes to q = 127 with d = 16512, so w = 2097024; x = +-1.5 * 2^106
     * makes each product about 2.55e38, finite, and the sequential sum 0. Positions 0 and 8
     * meet in the first step of the fold and overflow, as do 1 and 9. */
    float row[64] = {0}, xs[64] = {0};
    for (int i = 0; i < 64; i++) row[i] = 2097152.0f;
    xs[0] = xs[8] = ldexpf(1.5f, 106); xs[1] = xs[9] = -ldexpf(1.5f, 106);
    uint8_t one[2 * 34];
    float c = 1.0f, d = 1.0f;
    ok = nt_quantize_row(row, one, 64, GGUF_TYPE_Q8_0) == 0;
    wt r1 = {one, NULL, GGUF_TYPE_Q8_0, 1, 64, 0};
    qmm(&c, &r1, xs, 1);
    ok &= nt_cuda_qmm(&d, &r1, xs, 1) == NT_OK;
    CHECK(ok && isfinite(c) && !memcmp(&c, &d, sizeof(c)), "an overflowing fold is retried in the sequential order");
    free(w); free(x); free(cpu); free(gpu); free(packed);
}

/* The widest row RMSNorm can stage on this device, in floats: dim + 1 of them must fit the
 * default 48 KB or the device's opt-in limit, whichever is larger. */
static int device_norm_width(void) {
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) optin = 0;
    if (optin < 48 * 1024) optin = 48 * 1024;
    return optin / (int)sizeof(float) - 1;
}

/* A gemma3 body WE wide (one head of 32, FF 32, one layer). 12288 is the width the first
 * decoder accepted at load and then could not normalize, because 48 KB of staged row plus one
 * static float is past the default per-block shared memory. The rule is the same on every
 * device: a body within its limit loads, prefills and decodes with the CPU's argmax at every
 * step, and a body beyond it is refused at load. */
static void wide_body(int WE) {
    enum { WH = 32, WF = 32, STEPS_W = 3 };
    int fits = WE <= device_norm_width();
    char what[128];
    const char *fields[] = {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm",
        "attn_k_norm", "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm"};
    int rows[] = {0, WH, WH, WH, WE, 0, 0, 0, 0, WF, WF, WE, 0};
    int cols[] = {WE, WE, WE, WE, WH, WH, WH, WE, WE, WE, WE, WF, WE};
    char path[] = "/tmp/nt_cuda_wide_XXXXXX", name[96];
    int fd = mkstemp(path);
    float *norm = malloc(WE * sizeof(float));
    if (fd < 0 || !norm) { CHECK(0, "wide body: temporary file"); free(norm); return; }
    close(fd);
    for (int i = 0; i < WE; i++) norm[i] = 0.9f + 0.02f * (float)(i % 7);
    gguf_writer *w = gguf_write_open(path);
    int ok = w != NULL;
    if (ok) {
        gguf_write_kv_str(w, "general.architecture", "gemma3");
        gguf_write_kv_u32(w, "gemma3.block_count", 1);
        gguf_write_kv_u32(w, "gemma3.embedding_length", WE);
        gguf_write_kv_u32(w, "gemma3.feed_forward_length", WF);
        gguf_write_kv_u32(w, "gemma3.attention.head_count", 1);
        gguf_write_kv_u32(w, "gemma3.attention.head_count_kv", 1);
        gguf_write_kv_u32(w, "gemma3.attention.key_length", WH);
        gguf_write_kv_u32(w, "gemma3.attention.value_length", WH);
        gguf_write_kv_u32(w, "gemma3.context_length", CAP);
        gguf_write_kv_f32(w, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
        gguf_write_kv_f32(w, "gemma3.rope.freq_base", 10000);
        gguf_write_tensor_decl(w, "token_embd.weight", 2, (uint64_t[]){WE, V}, GGUF_TYPE_Q8_0);
        gguf_write_tensor_decl(w, "output_norm.weight", 1, (uint64_t[]){WE}, GGUF_TYPE_F32);
        for (int f = 0; f < 13; f++) {
            snprintf(name, sizeof(name), "blk.0.%s.weight", fields[f]);
            if (rows[f]) gguf_write_tensor_decl(w, name, 2, (uint64_t[]){(uint64_t)cols[f], (uint64_t)rows[f]}, GGUF_TYPE_Q8_0);
            else gguf_write_tensor_decl(w, name, 1, (uint64_t[]){(uint64_t)cols[f]}, GGUF_TYPE_F32);
        }
        ok = put_q8(w, "token_embd.weight", V, WE, 3) == 0 &&
             gguf_write_tensor_f32(w, "output_norm.weight", norm, WE) == 0;
        for (int f = 0; ok && f < 13; f++) {
            snprintf(name, sizeof(name), "blk.0.%s.weight", fields[f]);
            ok = rows[f] ? put_q8(w, name, rows[f], cols[f], 7 + f) == 0
                         : gguf_write_tensor_f32(w, name, norm, (uint64_t)cols[f]) == 0;
        }
        ok = gguf_write_close(w) == 0 && ok;
    }
    snprintf(what, sizeof(what), "wide body %d: write the GGUF", WE);
    CHECK(ok, what);
    const nt_arch *gpu = &nt_arch_gemma3_cuda;
    gguf_file *gf = ok && fits ? gguf_open(path) : NULL, *gf2 = ok ? gguf_open(path) : NULL;
    nt_dims dims, gdims;
    void *cm = gf ? nt_arch_gemma3.load(gf, &dims) : NULL;
    void *gm = gf2 ? gpu->load(gf2, &gdims) : NULL;
    if (fits) {
        snprintf(what, sizeof(what), "wide body %d: within this device's limit, the GPU loads it", WE);
        CHECK(cm && gm, what);
    } else {
        snprintf(what, sizeof(what), "wide body %d: beyond this device's limit, the GPU refuses it at load", WE);
        CHECK(ok && !gm, what);
    }
    if (cm && gm) {
        kv_cache *hk = kv_new(dims.n_layers, CAP, dims.kv_dim), *dk = gpu->kv_new(gm, CAP);
        float host[V], dev[V];
        int tokens[3] = {1, 7, 3}, same = 1, rc = NT_OK;
        rc |= nt_arch_gemma3.forward(cm, hk, tokens, 3, 0, host);
        rc |= gpu->forward(gm, dk, tokens, 3, 0, dev);
        for (int t = 0; t < STEPS_W && rc == NT_OK; t++) {
            same &= argmax(host) == argmax(dev);
            int next = argmax(host);
            rc |= nt_arch_gemma3.forward(cm, hk, &next, 1, 3 + t, host);
            rc |= gpu->forward(gm, dk, &next, 1, 3 + t, dev);
        }
        snprintf(what, sizeof(what), "wide body %d: the GPU prefills and decodes it", WE);
        CHECK(rc == NT_OK, what);
        snprintf(what, sizeof(what), "wide body %d: every argmax agrees with the CPU", WE);
        CHECK(rc == NT_OK && same && argmax(host) == argmax(dev), what);
        kv_free(hk); gpu->kv_free(gm, dk);
    }
    if (cm) nt_arch_gemma3.free(cm);
    if (gm) gpu->free(gm);
    gguf_close(gf); gguf_close(gf2);
    unlink(path); free(norm);
}

int main(void) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        fprintf(stderr, "CUDA_DECODE: no CUDA device\n");
        return 1;
    }
    setenv("NT_NO_I8", "1", 1);
    matmul_exact();
    wide_body(12288);
    wide_body(device_norm_width() / 32 * 32);         /* the widest this device can stage */
    wide_body((device_norm_width() / 32 + 1) * 32);   /* the first width past this device */
    printf("RMSNorm staging on this device: up to %d floats per row\n", device_norm_width());
    const char *families[] = {"qwen2", "qwen3", "gemma3"};
    for (int family = 0; family < 3; family++) {
        const nt_arch *cpu = family == 2 ? &nt_arch_gemma3 : &nt_arch_llama;
        const nt_arch *gpu = nt_cuda_arch_for(cpu);
        CHECK(gpu && gpu->kv_new && gpu->kv_free && gpu->forward_multi, "the family has a device twin");
        if (!gpu) continue;
        char path[] = "/tmp/nt_cuda_XXXXXX";
        int fd = mkstemp(path);
        if (fd < 0) return 1;
        close(fd);
        CHECK((family == 2 ? fixture_gemma3(path) : fixture(path, family == 1)) == 0, "write Q8_0 fixture");
        gguf_file *gf = gguf_open(path), *gf2 = gguf_open(path);
        nt_dims dims, gdims;
        void *cm = gf ? cpu->load(gf, &dims) : NULL;
        void *gm = gf2 ? gpu->load(gf2, &gdims) : NULL;
        CHECK(cm && gm, "load the fixture on both devices");
        if (cm && gm) {
            CHECK(!memcmp(&dims, &gdims, sizeof(dims)), "both loads report the same dimensions");
            for (int steered = 0; steered < 2; steered++) {
                char label[64];
                snprintf(label, sizeof(label), "%s %s", families[family], steered ? "steered" : "plain");
                run(cpu, cm, gpu, gm, dims, steered, label);
            }
            refusals(gpu, gm, dims);
        }
        if (cm) cpu->free(cm);
        if (gm) gpu->free(gm);
        gguf_close(gf); gguf_close(gf2);
        unlink(path);
    }
    printf("largest |GPU - CPU| logit difference: %.3g\n", (double)worst);
    if (failed) { fprintf(stderr, "CUDA_DECODE FAILED (%d of %d)\n", failed, checks); return 1; }
    printf("CUDA_DECODE_OK (%d checks)\n", checks);
    return 0;
}
