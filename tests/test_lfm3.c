/* test_lfm3.c — the LFM3 family executed on a file written here, so make test runs it.
 *
 * The family source is included rather than linked to reach its static RoPE. The file is
 * written by hand because the sliding-window pattern is a GGUF bool array, which the
 * library's writer does not emit. Layers: conv, full attention, conv, sliding attention;
 * a prompt longer than the window, so the sliding span is exercised. */
#include "harness/arch_lfm3.c"
#include <stdint.h>
#include <unistd.h>

enum { E = 64, V = 34, HD = 8, H = 2, KVH = 1, LC = 3, F = 24, NL = 4, SWA = 4, N = 10 };

static int checks, failed;
#define CHECK(c, s) do { checks++; if (!(c)) { fprintf(stderr, "FAIL: %s\n", s); failed++; } } while (0)

typedef struct { char name[96]; uint32_t nd; uint64_t ne[3]; float *d; uint64_t n; } T;
static T ts[64];
static int nts;
static uint32_t rng;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) / 16777216.0f) - 0.5f; }
static void add(const char *name, uint32_t nd, uint64_t ne0, uint64_t ne1, float base, float scale) {
    T *t = &ts[nts++];
    snprintf(t->name, sizeof t->name, "%s", name);
    t->nd = nd; t->ne[0] = ne0; t->ne[1] = ne1; t->n = ne0 * (nd > 1 ? ne1 : 1);
    t->d = malloc(t->n * sizeof(float));
    for (uint64_t i = 0; i < t->n; i++) t->d[i] = base + scale * rnd();
}
static void wu32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wu64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void wstr(FILE *f, const char *s) { wu64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }
static void kvu32(FILE *f, const char *k, uint32_t v) { wstr(f, k); wu32(f, 4); wu32(f, v); }
static void kvf32(FILE *f, const char *k, float v) { wstr(f, k); wu32(f, 6); fwrite(&v, 4, 1, f); }
static void kvstr(FILE *f, const char *k, const char *v) { wstr(f, k); wu32(f, 8); wstr(f, v); }

/* rope: 0 no table, 1 partial-RoPE table [1,1,1e30,1e30], 2 table of ones.
 * The rest break one thing each; causal: 0 key absent, 1 true, 2 false. */
typedef struct {
    int rope, drop_qnorm, short_in_proj, one_post, causal, odd_rot;
    int emb3d, vocab_key, huge_embed, unit_post, omit_post_mask;
} opts;

static int write_model(const char *path, opts o) {
    nts = 0; rng = 12345;
    const int is_attn[NL] = {0, 1, 0, 1};
    const uint8_t pat[NL] = {0, 0, 0, 1};
    char nm[96];
    add("token_embd.weight", 2, E, V, 0, 1);
    if (o.emb3d) { ts[0].nd = 3; ts[0].ne[1] = V / 2; ts[0].ne[2] = 2; }   /* same bytes, rank 3 */
    add("output_norm.weight", 1, E, 0, 1, 0.1f);
    for (int l = 0; l < NL; l++) {
#define A(s, nd, a, b, base, sc) do { snprintf(nm, sizeof nm, "blk.%d." s, l); add(nm, nd, a, b, base, sc); } while (0)
        A("attn_norm.weight", 1, E, 0, 1, 0.1f);
        A("ffn_norm.weight", 1, E, 0, 1, 0.1f);
        A("post_attention_norm.weight", 1, E, 0, 1, o.unit_post ? 0 : 0.1f);
        if (!(o.one_post && l == 0)) A("post_ffw_norm.weight", 1, E, 0, 1, o.unit_post ? 0 : 0.1f);
        if (o.omit_post_mask & (1 << l)) {
            /* Advance the RNG identically so every other tensor stays fixed. */
            free(ts[--nts].d); free(ts[--nts].d);
        }
        A("ffn_gate.weight", 2, E, F, 0, 0.5f);
        A("ffn_up.weight", 2, E, F, 0, 0.5f);
        A("ffn_down.weight", 2, F, E, 0, 0.5f);
        if (is_attn[l]) {
            A("attn_q.weight", 2, E, H * HD, 0, 0.5f);
            A("attn_k.weight", 2, E, KVH * HD, 0, 0.5f);
            A("attn_v.weight", 2, E, KVH * HD, 0, 0.5f);
            A("attn_output.weight", 2, H * HD, E, 0, 0.5f);
            A("attn_gate.weight", 2, E, H, 0, 1);
            if (!(o.drop_qnorm && l == 1)) A("attn_q_norm.weight", 1, HD, 0, 1, 0.1f);
            A("attn_k_norm.weight", 1, HD, 0, 1, 0.1f);
        } else {
            A("shortconv.in_proj.weight", 2, E, (o.short_in_proj ? 2 : 3) * E, 0, 0.5f);
            A("shortconv.out_proj.weight", 2, E, E, 0, 0.5f);
            A("shortconv.conv.weight", 2, LC, E, 0, 1);
        }
#undef A
    }
    if (o.rope) {
        add("rope_freqs.weight", 1, HD / 2, 0, 1, 0);
        if (o.rope == 1) { ts[nts - 1].d[2] = 1e30f; ts[nts - 1].d[3] = 1e30f; }
    }
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    wu32(f, 0x46554747); wu32(f, 3); wu64(f, (uint64_t)nts);
    wu64(f, 12 + (o.causal ? 1 : 0) + (o.vocab_key ? 1 : 0));
    kvstr(f, "general.architecture", "lfm3");
    kvu32(f, "lfm3.block_count", NL);
    kvu32(f, "lfm3.embedding_length", o.huge_embed ? 715827883u : E);   /* 3 * this wraps int32 */
    if (o.vocab_key) kvu32(f, "lfm3.vocab_size", V + 1);
    kvu32(f, "lfm3.attention.key_length", HD);
    kvu32(f, "lfm3.attention.value_length", HD);
    kvu32(f, "lfm3.rope.dimension_count", o.odd_rot ? HD - 1 : HD);
    if (o.causal) { wstr(f, "lfm3.attention.causal"); wu32(f, 7); fputc(o.causal == 1, f); }
    kvu32(f, "lfm3.shortconv.l_cache", LC);
    kvu32(f, "lfm3.attention.sliding_window", SWA);
    kvf32(f, "lfm3.rope.freq_base", 1e6f);
    kvf32(f, "lfm3.rope.freq_base_swa", 1e4f);
    kvf32(f, "lfm3.attention.layer_norm_rms_epsilon", 1e-5f);
    wstr(f, "lfm3.attention.sliding_window_pattern"); wu32(f, 9); wu32(f, 7); wu64(f, NL);
    fwrite(pat, 1, NL, f);
    uint64_t off = 0;
    for (int i = 0; i < nts; i++) {
        wstr(f, ts[i].name); wu32(f, ts[i].nd);
        for (uint32_t d = 0; d < ts[i].nd; d++) wu64(f, ts[i].ne[d]);
        wu32(f, 0); wu64(f, off);
        off += (ts[i].n * 4 + 31) & ~31ull;
    }
    long p = ftell(f);
    while (p % 32) { fputc(0, f); p++; }
    for (int i = 0; i < nts; i++) {
        fwrite(ts[i].d, 4, ts[i].n, f);
        for (uint64_t k = ts[i].n * 4; k % 32; k++) fputc(0, f);
        free(ts[i].d);
    }
    return fclose(f);
}

typedef struct { gguf_file *gf; void *m; nt_dims d; } M;
static int open_model(M *x, opts o) {
    char path[] = "/tmp/notorch-lfm3-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    close(fd);
    memset(x, 0, sizeof *x);
    if (write_model(path, o) == 0 && (x->gf = gguf_open(path)))
        x->m = nt_arch_lfm3.load(x->gf, &x->d);
    unlink(path);
    return x->m != NULL;
}
static void close_model(M *x) { if (x->m) nt_arch_lfm3.free(x->m); if (x->gf) gguf_close(x->gf); }

static kv_cache *cache_new(M *x, int max_seq) {
    return nt_arch_lfm3.kv_new ? nt_arch_lfm3.kv_new(x->m, max_seq)
                               : kv_new(x->d.n_layers, max_seq, x->d.kv_dim);
}
static void cache_free(M *x, kv_cache *kv) {
    if (nt_arch_lfm3.kv_free) nt_arch_lfm3.kv_free(x->m, kv); else kv_free(kv);
}

static const int seq_a[N] = {3, 17, 5, 29, 11, 2, 8, 31, 14, 6};
static const int seq_b[N] = {9, 1, 22, 4, 27, 13, 30, 7, 19, 25};

static int run(M *x, const int *tok, int n, int chunk, float *logits) {
    kv_cache *kv = cache_new(x, 16);
    int rc = kv ? NT_OK : NT_E_MEMORY;
    for (int i = 0; i < n && rc == NT_OK; i += chunk) {
        int c = n - i < chunk ? n - i : chunk;
        rc = nt_arch_lfm3.forward(x->m, kv, tok + i, c, i, logits);
    }
    if (kv) cache_free(x, kv);
    return rc;
}

/* ggml's freq_factors semantics written independently: theta / ff (ops.cpp,
 * ggml_rope_cache_init), the same per-lane powf as the family so equality is exact. */
static void rope_ref(float *x, int pos, int n_rot, float base, const float *ff) {
    int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, 2.0f * (float)i / (float)n_rot);
        if (ff) freq = freq / ff[i];
        float a = (float)pos * freq, cs = cosf(a), sn = sinf(a), x0 = x[i], x1 = x[i + half];
        x[i] = x0 * cs - x1 * sn;
        x[i + half] = x0 * sn + x1 * cs;
    }
}

/* The window state belongs to the sequence: two caches on one model must not see
 * each other, a continuation must start where its cache stopped, and position zero
 * must start from silence. */
static void test_caches(M *x, const float *solo_a) {
    float solo_b[V], la[V], lb[V], t[V], r[V];
    CHECK(run(x, seq_b, N, 1, solo_b) == NT_OK, "sequence B alone");
    kv_cache *ka = cache_new(x, 16), *kb = cache_new(x, 16);
    int ok = ka && kb;
    for (int i = 0; ok && i < N; i++)
        ok = nt_arch_lfm3.forward(x->m, ka, &seq_a[i], 1, i, la) == NT_OK &&
             nt_arch_lfm3.forward(x->m, kb, &seq_b[i], 1, i, lb) == NT_OK;
    CHECK(ok, "two caches decode interleaved on one model");
    CHECK(ok && !memcmp(la, solo_a, sizeof la), "sequence A is untouched by sequence B");
    CHECK(ok && !memcmp(lb, solo_b, sizeof lb), "sequence B is untouched by sequence A");

    for (int i = 0; i < V; i++) t[i] = 1234.0f;
    CHECK(ok && nt_arch_lfm3.forward(x->m, ka, &seq_b[0], 1, N + 1, t) == NT_E_STATE,
          "a gap in positions is refused");
    CHECK(ok && nt_arch_lfm3.forward(x->m, ka, &seq_b[0], 1, 3, t) == NT_E_STATE,
          "a rewind is refused");
    int untouched = 1;
    for (int i = 0; i < V; i++) if (t[i] != 1234.0f) untouched = 0;
    CHECK(untouched, "a refused call writes no logits");
    CHECK(ok && nt_arch_lfm3.forward(x->m, ka, &seq_b[0], 1, N, t) == NT_OK,
          "the next position is accepted");

    int rc = ok ? NT_OK : NT_E_CACHE;
    for (int i = 0; i < N && rc == NT_OK; i++)
        rc = nt_arch_lfm3.forward(x->m, kb, &seq_a[i], 1, i, r);
    CHECK(rc == NT_OK && !memcmp(r, solo_a, sizeof r), "position zero on a used cache starts from silence");
    if (ka) cache_free(x, ka);
    if (kb) cache_free(x, kb);

    kv_cache *host = kv_new(x->d.n_layers, 16, x->d.kv_dim);
    kv_cache *half = kv_new(x->d.n_layers, 16, x->d.kv_dim / 2);
    CHECK(host && nt_arch_lfm3.forward(x->m, host, seq_a, N, 0, t) == NT_E_CACHE,
          "a host cache without the window state is refused");
    CHECK(half && nt_arch_lfm3.forward(x->m, half, seq_a, N, 0, t) == NT_E_CACHE,
          "a cache narrower than the model is refused");
    kv_free(host);
    kv_free(half);

    M other = {0};
    CHECK(open_model(&other, (opts){.rope = 2}), "a second model loads");
    kv_cache *foreign = other.m ? cache_new(&other, 16) : NULL;
    CHECK(foreign && nt_arch_lfm3.forward(x->m, foreign, seq_a, N, 0, t) == NT_E_CACHE,
          "another model's cache is refused");
    if (foreign) cache_free(&other, foreign);
    close_model(&other);
}

/* The tensor-header guard on its own, against headers no writer here would emit. */
static void test_dims(void) {
    lfm3_model huge = {.n_attn=128, .max_kv=1<<27, .n_recr=1, .embed=16, .l_cache=3};
    CHECK(!nt_arch_lfm3.kv_new(&huge,1<<30), "2^64 KV cells cannot wrap into an empty cache");
    huge.n_attn=1; huge.max_kv=1; huge.n_recr=128; huge.embed=1<<27; huge.l_cache=(1<<30)+1;
    CHECK(!nt_arch_lfm3.kv_new(&huge,1), "2^64 convolution cells cannot wrap into an empty window");
    gguf_file *g = calloc(1, sizeof(*g));
    if (!g) { CHECK(0, "allocate a tensor table"); return; }
    gguf_tensor_info *t = &g->tensors[0];
    t->ndim = 2; t->shape[0] = 1ULL << 33; t->shape[1] = 1ULL << 32;
    t->n_elements = t->shape[0] * t->shape[1];                      /* wraps to 0 */
    CHECK(!lfm3_dims(g, 0, 2), "a [2^33, 2^32] shape is refused");
    t->shape[0] = 16; t->shape[1] = 32; t->n_elements = 512;
    CHECK(lfm3_dims(g, 0, 2), "a consistent [16, 32] shape is accepted");
    CHECK(!lfm3_dims(g, 0, 1), "a matrix where a vector is expected is refused");
    t->shape[1] = 0; t->n_elements = 0;
    CHECK(!lfm3_dims(g, 0, 2), "a [16, 0] shape is refused");
    t->shape[1] = 32; t->n_elements = 513;
    CHECK(!lfm3_dims(g, 0, 2), "an element count the shape does not give is refused");
    free(g);
}

static void test_rope(void) {
    float ff[64];
    for (int i = 0; i < 64; i++) ff[i] = i < 32 ? 1.0f : 1e30f;
    const int poss[] = {0, 16, 2048, 8192};
    for (int p = 0; p < 4; p++) {
        float a[128], b[128];
        rng = 777;
        for (int i = 0; i < 128; i++) a[i] = b[i] = rnd();
        rope_lanes(a, poss[p], 128, 1e6f, ff);
        rope_ref(b, poss[p], 128, 1e6f, ff);
        CHECK(!memcmp(a, b, sizeof a), "rope_lanes divides by the table as ggml does");
        int still = 1;
        rng = 777;
        for (int i = 0; i < 128; i++) { float v = rnd(); if ((i % 64) >= 32 && a[i] != v) still = 0; }
        CHECK(still, "pairs the table freezes do not rotate");
    }
}

/* Embedding rows in place of ids: the rows a token would read, put under placeholder ids,
 * must give that token's logits bit for bit, or the rows are not what the forward read. */
static void test_rows(M *x, const float *ref) {
    lfm3_model *m = (lfm3_model *)x->m;
    float emb[N][E], lg[V], hid[E], head[V];
    const float *rows[N], *none[N] = {0};
    int ph[N];
    for (int j = 0; j < N; j++) {
        CHECK(gguf_dequant_row(x->gf, m->emb_ti, (uint64_t)seq_a[j], emb[j]) == 0, "read an embedding row");
        rows[j] = emb[j];
        ph[j] = 0;
    }
    kv_cache *kv = cache_new(x, 16);
    CHECK(kv && nt_arch_lfm3.forward_embd(x->m, kv, ph, rows, N, 0, lg, hid) == NT_OK,
          "forward_embd over placeholder ids");
    CHECK(!memcmp(lg, ref, sizeof lg), "rows under placeholders give the tokens' logits, bit for bit");
    qmv(head, &m->tok_emb, hid);
    CHECK(!memcmp(head, lg, sizeof head), "hidden is the vector the head reads");
    CHECK(kv && nt_arch_lfm3.forward_embd(x->m, kv, seq_a, none, N, 0, lg, NULL) == NT_OK &&
          !memcmp(lg, ref, sizeof lg), "all-NULL rows equal forward");
    emb[N - 2][0] += 0.5f;
    CHECK(kv && nt_arch_lfm3.forward_embd(x->m, kv, ph, rows, N, 0, lg, NULL) == NT_OK &&
          memcmp(lg, ref, sizeof lg) != 0, "a changed row changes the logits");
    for (int i = 0; i < V; i++) lg[i] = 99.0f;
    for (int i = 0; i < E; i++) hid[i] = 99.0f;
    ph[3] = V;
    int refused = kv && nt_arch_lfm3.forward_embd(x->m, kv, ph, rows, N, 0, lg, hid) == NT_E_TOKEN;
    for (int i = 0; i < V; i++) refused &= lg[i] == 99.0f;
    for (int i = 0; i < E; i++) refused &= hid[i] == 99.0f;
    CHECK(refused, "a placeholder outside the vocabulary is refused, nothing written");
    if (kv) cache_free(x, kv);
}

/* Evaluate the dense FFN from the residual handed to the first hook. Its
 * result is the independent oracle for the second hook on both mixer types. */
typedef struct {
    lfm3_model *m;
    int pos, n, intervening_layer, stop, events;
    double expected[N][E];
} probe;

static double row_dot(const wt *w, int row, const double *v) {
    const float *f = w->f32 ? w->f32 : (const float *)(const void *)w->q;
    double sum = 0;
    for (int i = 0; i < w->cols; i++) sum += f[row*w->cols+i] * v[i];
    return sum;
}

static int before_ffn(void *user, int l, int pos, int n, int e, float *rows) {
    probe *p = user;
    if (pos != p->pos || n != p->n || e != E || p->events++ != 2*l) return 73;
    if (p->stop == 1 && l == p->intervening_layer) return 71;
    for (int j = 0; j < n; j++) {
        float *x = rows + (size_t)j*E;
        if (l == p->intervening_layer) x[0] += 2.5f;
        double v[E], inner[F], delta[E], ss = 0;
        for (int i = 0; i < E; i++) ss += (double)x[i]*x[i];
        double rms = sqrt(ss/E+p->m->rms_eps);
        for (int i = 0; i < E; i++) v[i] = (double)x[i] * p->m->layers[l].ffn_norm[i] / rms;
        for (int k = 0; k < F; k++) {
            double g = row_dot(&p->m->layers[l].wgate, k, v);
            inner[k] = g/(1+exp(-g))*row_dot(&p->m->layers[l].wup, k, v);
        }
        ss = 0;
        for (int i = 0; i < E; i++) {
            delta[i] = row_dot(&p->m->layers[l].wdown, i, inner);
            ss += delta[i]*delta[i];
        }
        const float *post = p->m->layers[l].post_ffn_norm;
        for (int i = 0; i < E; i++)
            p->expected[j][i] = x[i] + (post ? delta[i]*post[i]/sqrt(ss/E+p->m->rms_eps) : delta[i]);
    }
    return NT_OK;
}

static int after_ffn(void *user, int l, int pos, int n, int e, float *rows) {
    probe *p = user;
    if (pos != p->pos || n != p->n || e != E || p->events++ != 2*l+1) return 73;
    if (p->stop == 2 && l == p->intervening_layer) return 72;
    for (int j = 0; j < n; j++) for (int i = 0; i < E; i++)
        if (!(fabs(rows[j*E+i] - p->expected[j][i]) < 2e-5)) return 74;
    return NT_OK;
}

static void test_doe(M *x) {
    const nt_doe_decoder *d = nt_arch_lfm3.doe;
    nt_dims geometry = {0};
    CHECK(d && d->describe(x->m, &geometry) == x->gf && geometry.n_layers == NL &&
          geometry.kv_dim == x->d.kv_dim && geometry.vocab == V && x->d.n_layers == 2,
          "calibrate at logical depth, allocate KV for the two attention slots");
    if (!d) return;
    ++x->gf->embed_dim;
    CHECK(!d->describe(x->m, &geometry), "inconsistent generic width cannot enter the frame");
    --x->gf->embed_dim;
    lfm3_model *m = x->m;
    for (int l = 0; l < NL; l++)
        CHECK(d->norm(m,4*l)==m->layers[l].attn_norm && d->norm(m,4*l+1)==m->layers[l].post_attn_norm &&
              d->norm(m,4*l+2)==m->layers[l].ffn_norm && d->norm(m,4*l+3)==m->layers[l].post_ffn_norm,
              "each block exposes its full-width RMS norms in forward order");
    CHECK(d->norm(m,4*NL)==m->out_norm && !d->norm(m,-1) && !d->norm(m,4*NL+1), "norm bounds and final norm");
    kv_cache *kv = cache_new(x,16);
    CHECK(kv, "allocate a hybrid cache for the hooks");
    if (!kv) return;
    float plain[V], logits[V], sentinel[V], active[V];
    CHECK(run(x,seq_a,N,N,plain)==NT_OK,"independent plain cache");
    for (int target=-1; target<NL; target++) {
        for (int chunk=1; chunk<=N; chunk+=N-1) {
            probe p = {.m=m, .intervening_layer=target};
            for (int pos=0; pos<N; pos+=chunk) {
                p.pos=pos; p.n=chunk; p.events=0;
                CHECK(d->forward(m,kv,seq_a+pos,chunk,pos,logits,before_ffn,after_ffn,&p)==NT_OK &&
                      p.events==2*NL, "hybrid hooks enclose the exact FFN, in layer and position order");
            }
            CHECK((memcmp(plain,logits,sizeof plain)!=0)==(target>=0),"passive logits match; an intervention changes them");
            if (chunk==1) memcpy(active,logits,sizeof active);
            else CHECK(!memcmp(active,logits,sizeof active),"active grouped and scalar paths agree");
        }
    }
    for (int stop=1; stop<=2; stop++) for (int target=0; target<NL; target++) {
        probe p = {.m=m, .pos=1, .n=2, .intervening_layer=target, .stop=stop};
        CHECK(nt_arch_lfm3.forward(m,kv,seq_a,1,0,NULL)==NT_OK,"init prefix before interruption");
        for (int i=0; i<V; i++) sentinel[i]=logits[i]=-1234;
        CHECK(d->forward(m,kv,seq_a+1,2,1,logits,before_ffn,after_ffn,&p)==70+stop &&
              !memcmp(sentinel,logits,sizeof logits), "callback error preserves exact status and all logits");
        CHECK(nt_arch_lfm3.forward(m,kv,seq_a+1,1,1,NULL)==NT_E_STATE &&
              nt_arch_lfm3.forward(m,kv,seq_a+3,1,3,NULL)==NT_E_STATE,
              "a partial conv/KV cache refuses continuation at either end of the failed chunk");
        CHECK(nt_arch_lfm3.forward(m,kv,seq_a,N-1,0,NULL)==NT_OK &&
              nt_arch_lfm3.forward(m,kv,seq_a+N-1,1,N-1,logits)==NT_OK && !memcmp(plain,logits,sizeof plain),
              "restart at zero and subsequent decode reproduce fresh-cache logits");
    }
    cache_free(x,kv);
}

/* Mixed media rows and ordinary ids must reach both sides of every FFN. Use
 * token embeddings as an independent input oracle, then change one media row.
 * This also prices the final hidden vector consumed by the speech decoder. */
static void test_doe_rows(M *x) {
    const nt_doe_decoder *d = nt_arch_lfm3.doe;
    CHECK(d && d->forward_embd, "the hybrid exposes media rows to DoE");
    if (!d || !d->forward_embd) return;
    lfm3_model *m = x->m;
    float emb[N][E], plain[V], hplain[E], logits[V], hidden[E], head[V], scalar[V], hs[E];
    const float *rows[N];
    int ids[N];
    for (int j=0; j<N; j++) {
        CHECK(!gguf_dequant_row(x->gf,m->emb_ti,(uint64_t)seq_a[j],emb[j]), "read media oracle row");
        rows[j] = j%2 ? emb[j] : NULL;
        ids[j] = rows[j] ? 0 : seq_a[j];
    }
    kv_cache *kv = cache_new(x,16);
    CHECK(kv, "allocate media cache");
    if (!kv) return;
    CHECK(nt_arch_lfm3.forward_embd(m,kv,ids,rows,N,0,plain,hplain)==NT_OK &&
          d->forward_embd(m,kv,ids,rows,N,0,logits,hidden,NULL,NULL,NULL)==NT_OK &&
          !memcmp(plain,logits,sizeof plain) && !memcmp(hplain,hidden,sizeof hidden),
          "NULL hooks preserve plain media logits and final hidden exactly");
    for (int target=-1; target<NL; target++) {
        probe p = {.m=m, .pos=0, .n=N, .intervening_layer=target};
        CHECK(d->forward(m,kv,seq_a,N,0,plain,before_ffn,after_ffn,&p)==NT_OK,
              "token-only oracle for an active media intervention");
        for (int chunk=1; chunk<=N; chunk+=N-1) {
            p.events=0;
            for (int pos=0; pos<N; pos+=chunk) {
                p.pos=pos; p.n=chunk; p.events=0;
                CHECK(d->forward_embd(m,kv,ids+pos,rows+pos,chunk,pos,logits,hidden,
                                     before_ffn,after_ffn,&p)==NT_OK && p.events==2*NL,
                      "media callbacks enclose every exact FFN");
            }
            CHECK(!memcmp(plain,logits,sizeof plain), "active media rows equal the independent token path");
            qmv(head,&m->tok_emb,hidden);
            CHECK(!memcmp(head,logits,sizeof head), "expert hidden produces expert logits");
            if (chunk==1) { memcpy(scalar,logits,sizeof scalar); memcpy(hs,hidden,sizeof hs); }
            else CHECK(!memcmp(scalar,logits,sizeof scalar) && !memcmp(hs,hidden,sizeof hs),
                       "active media chunks preserve logits and hidden bit for bit");
        }
    }
    emb[3][0] += 0.5f;
    CHECK(d->forward_embd(m,kv,ids,rows,N,0,plain,hplain,NULL,NULL,NULL)==NT_OK &&
          nt_arch_lfm3.forward_embd(m,kv,seq_a,NULL,N,0,logits,hidden)==NT_OK &&
          memcmp(plain,logits,sizeof plain), "changed media reaches the DoE path");
    CHECK(d->forward_embd(m,kv,ids,rows,N-1,0,NULL,NULL,NULL,NULL,NULL)==NT_OK &&
          d->forward_embd(m,kv,ids+N-1,rows+N-1,1,N-1,logits,hidden,NULL,NULL,NULL)==NT_OK &&
          !memcmp(plain,logits,sizeof plain) && !memcmp(hplain,hidden,sizeof hidden),
          "media prefix continuation equals fresh full replay");
    for (int stop=1; stop<=2; stop++) for (int target=0; target<NL; target++) {
        probe p = {.m=m, .pos=1, .n=2, .intervening_layer=target, .stop=stop};
        CHECK(d->forward_embd(m,kv,ids,rows,1,0,NULL,NULL,NULL,NULL,NULL)==NT_OK,
              "media prefix before an abort");
        for (int i=0; i<V; i++) logits[i]=-1234;
        for (int i=0; i<E; i++) hidden[i]=-1234;
        int r=d->forward_embd(m,kv,ids+1,rows+1,2,1,logits,hidden,before_ffn,after_ffn,&p);
        int untouched=1;
        for (int i=0; i<V; i++) untouched &= logits[i]==-1234;
        for (int i=0; i<E; i++) untouched &= hidden[i]==-1234;
        CHECK(r==70+stop && untouched, "media callback abort preserves status and both output buffers");
        CHECK(d->forward_embd(m,kv,ids+1,rows+1,1,1,NULL,NULL,NULL,NULL,NULL)==NT_E_STATE &&
              d->forward_embd(m,kv,ids+3,rows+3,1,3,NULL,NULL,NULL,NULL,NULL)==NT_E_STATE,
              "aborted media cache refuses continuation");
        CHECK(d->forward_embd(m,kv,ids,rows,N,0,logits,hidden,NULL,NULL,NULL)==NT_OK &&
              !memcmp(plain,logits,sizeof plain) && !memcmp(hplain,hidden,sizeof hidden),
              "media restart restores fresh logits and hidden");
    }
    cache_free(x,kv);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1],"fixtures")) {
        return write_model("lfm-unit",(opts){.unit_post=1}) ||
               write_model("lfm-mixed",(opts){.unit_post=1,.omit_post_mask=1<<(NL-1)}) ||
               write_model("lfm-aligned",(opts){.unit_post=1,.omit_post_mask=1<<(NL-1),.rope=2}) ||
               write_model("lfm-bare",(opts){.unit_post=1,.omit_post_mask=(1<<NL)-1});
    }
    if (argc != 1) return 1;
    M x;
    CHECK(open_model(&x, (opts){.rope = 0}), "synthetic lfm3 file loads");
    if (!x.m) { printf("LFM3: %d checks, %d failures\n", checks, failed); return 1; }

    float one[V], dec[V], ch3[V];
    CHECK(run(&x, seq_a, N, N, one) == NT_OK, "one-shot prefill");
    CHECK(run(&x, seq_a, N, 1, dec) == NT_OK, "token-by-token decode");
    CHECK(run(&x, seq_a, N, 3, ch3) == NT_OK, "chunk-3 prefill");
    CHECK(!memcmp(one, dec, sizeof one), "one-shot equals token-by-token, bit for bit");
    CHECK(!memcmp(one, ch3, sizeof one), "one-shot equals chunk-3, bit for bit");
    test_rows(&x, one);
    test_caches(&x, dec);
    test_doe(&x);
    test_doe_rows(&x);
    close_model(&x);
    M mixed = {0};
    CHECK(open_model(&mixed,(opts){.omit_post_mask=1<<2}), "mixed sandwich and pre-norm blocks load");
    if (mixed.m) { test_doe(&mixed); test_doe_rows(&mixed); }
    close_model(&mixed);

    test_rope();
    test_dims();

    M ones = {0}, part = {0};
    float l_ones[V], l_part[V];
    CHECK(open_model(&ones, (opts){.rope = 2}) && open_model(&part, (opts){.rope = 1}),
          "files with rope_freqs.weight load");
    if (ones.m && part.m) {
        run(&ones, seq_a, N, N, l_ones);
        run(&part, seq_a, N, N, l_part);
        CHECK(!memcmp(l_ones, one, sizeof one), "a table of ones changes nothing");
        CHECK(memcmp(l_part, one, sizeof one) != 0, "the partial-RoPE table reaches the full-attention layer");
    }
    close_model(&ones);
    close_model(&part);

    /* Each of these breaks one thing the forward depends on or the reference refuses. */
    const struct { opts o; const char *what; } bad[] = {
        {{.drop_qnorm = 1},    "a file without attn_q_norm is refused"},
        {{.short_in_proj = 1}, "an in_proj of 2E rows is refused"},
        {{.one_post = 1},      "one post norm without the other is refused"},
        {{.causal = 2},        "a non-causal file is refused"},
        {{.odd_rot = 1},       "an odd rope.dimension_count is refused"},
        {{.emb3d = 1},         "a 3-D token_embd is refused"},
        {{.vocab_key = 1},     "a vocab_size that disagrees with token_embd is refused"},
        {{.huge_embed = 1},    "an embedding_length whose triple wraps int32 is refused"},
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        M b = {0};
        CHECK(!open_model(&b, bad[i].o), bad[i].what);
        close_model(&b);
    }
    M c = {0};
    CHECK(open_model(&c, (opts){.causal = 1}), "a file that says causal loads");
    close_model(&c);

    printf("LFM3: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
