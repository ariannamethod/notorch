/* runtime.c — see runtime.h. Lifted from examples/infer_llama.c unchanged in
 * arithmetic; the gate that keeps it that way is harness/test_parity.sh. */
#include "harness/arch.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* Both entry points report whether they could take the shape, so ask them once
 * with a single row rather than duplicating their dtype tables here and letting
 * the two drift. */
static void wt_probe(wt *w) {
    if (!w->q || w->cols <= 0) return;
    float *x = (float*)calloc(w->cols, sizeof(float));
    float out = 0.0f;
    if (x) {
        w->use_i8 = (nt_qmatvec_i8(&out, w->q, w->dtype, x, 1, w->cols) == 0);
        if (!w->use_i8 && nt_qmatvec(&out, w->q, w->dtype, x, 1, w->cols) != 0) {
            w->q = NULL;   /* no packed path — the loader will expand it */
        }
        free(x);
    }
}

int wt_load(wt *w, gguf_file *gf, const char *name) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return 0;
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (!t->shape[0] || t->n_elements % t->shape[0]) return 0;
    w->cols  = (int)t->shape[0];
    w->rows  = (int)(t->n_elements / t->shape[0]);
    w->dtype = (int)t->dtype;
    w->q     = gf->data + t->offset;
    wt_probe(w);
    if (!w->q) w->f32 = gguf_dequant(gf, ti);
    return (w->q || w->f32) ? 1 : 0;
}

int wt_expert(wt *dst, const wt *src, int index, int rows_each) {
    if (!src || index < 0 || rows_each <= 0) return 0;
    if ((long)(index + 1) * rows_each > src->rows) return 0;
    if (src->q) {
        uint64_t row_bytes = gguf_type_size((uint32_t)src->dtype, (uint64_t)src->cols);
        if (!row_bytes) return 0;
        *dst = *src;
        dst->rows = rows_each;
        dst->q    = src->q + row_bytes * (uint64_t)index * (uint64_t)rows_each;
        return 1;
    }
    /* The expanded copy is a contiguous [rows, cols] of floats and slices the same way. It
     * used to be refused here, which did not fail — the caller skipped that expert and
     * answered with seven of its eight, quietly. A dtype with no packed kernel is exactly
     * when this path is taken, so refusing it is refusing the case it exists for. */
    if (src->f32) {
        *dst = *src;
        dst->rows = rows_each;
        dst->f32  = src->f32 + (long)index * rows_each * src->cols;
        return 1;
    }
    return 0;
}

/* All of it or none of it.
 *
 * This used to return the struct whatever happened to the two allocations
 * inside it, so a cache too large for the machine came back looking like a
 * cache — non-NULL, with the right dimensions in its fields and NULL where the
 * memory should be. Every caller checked the pointer it was given; none checked
 * the two behind it. An outside audit reached it with kv_new(1, INT_MAX,
 * INT_MAX) and got a live object with no storage.
 *
 * The dimensions are checked before the multiplication rather than after: a
 * negative one turns into an enormous unsigned request, and a zero makes a
 * cache that every later bounds check treats as valid and empty. */
kv_cache *kv_new(int nl, int max_seq, int kv_dim) {
    if (nl < 1 || max_seq < 1 || kv_dim < 1) return NULL;
    size_t cells = (size_t)nl * (size_t)max_seq * (size_t)kv_dim;
    if (cells > SIZE_MAX / sizeof(float)) return NULL;

    kv_cache *kv = (kv_cache*)calloc(1, sizeof(kv_cache));
    if (!kv) return NULL;
    kv->k = (float*)calloc(cells, sizeof(float));
    kv->v = (float*)calloc(cells, sizeof(float));
    if (!kv->k || !kv->v) { free(kv->k); free(kv->v); free(kv); return NULL; }
    kv->max_seq = max_seq; kv->n_layers = nl; kv->kv_dim = kv_dim;
    return kv;
}

void kv_free(kv_cache *kv) { if (!kv) return; free(kv->k); free(kv->v); free(kv); }

// C[m,n] = A[m,k] @ B^T[n,k]
void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

void qmv(float *out, const wt *w, const float *x) {
    if (w->use_i8 && nt_qmatvec_i8(out, w->q, w->dtype, x, w->rows, w->cols) == 0) return;
    if (w->q && nt_qmatvec(out, w->q, w->dtype, x, w->rows, w->cols) == 0) return;
    mm_t(out, x, w->f32, 1, w->cols, w->rows);
}

void qmm(float *out, const wt *w, const float *X, int n) {
    if (n > 1 && w->use_i8 && w->q &&
        nt_qmatmul_i8(out, w->q, w->dtype, X, w->rows, w->cols, n) == 0) return;
    /* An unpacked weight reaches neither the line above nor use_i8, and fell through to
     * the loop — which re-reads the whole matrix per token and is why an f16 prefill ran
     * at decode speed. Both entries refuse what they cannot take, so asking costs a
     * comparison and the loop stays underneath as the answer for everything else. */
    if (n > 1 && w->q &&
        nt_qmatmul(out, w->q, w->dtype, X, w->rows, w->cols, n) == 0) return;
    for (int j = 0; j < n; j++)
        qmv(out + (long)j * w->rows, w, X + (long)j * w->cols);
}

/* Attention reads the KV cache, not the weights, so batching left it untouched
 * — and once the matmuls stopped dominating it surfaced as the second line of
 * the profile. Both loops are f32 over head_dim, which is 64 or 128 in every
 * model here and always a multiple of four, so four lanes cover them with a
 * scalar tail for anything odd. The four partial sums make this a different
 * summation order from the scalar loop, hence a different last bit; that is a
 * change to attention's arithmetic, not to its meaning. */
float dot_f32(const float *a, const float *b, int n) {
    int i = 0; float s = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    s = vaddvq_f32(acc);
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

void axpy_f32(float *y, float alpha, const float *x, int n) {
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
#endif
    for (; i < n; i++) y[i] += alpha * x[i];
}

void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void rope(float *x, int pos, int head_dim, float freq_base, int neox) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        int a = neox ? i : 2*i, b = neox ? i + half : 2*i + 1;
        float x0 = x[a], x1 = x[b];
        x[a] = x0 * cs - x1 * sn;
        x[b] = x0 * sn + x1 * cs;
    }
}

void add_bias(float *x, const float *bias, int n) {
    if (bias) for (int i = 0; i < n; i++) x[i] += bias[i];
}

int sample(float *logits, int n, float temp) {
    /* temp 0 means greedy, and dividing by it means every logit becomes an
     * infinity, the softmax becomes NaN, the cumulative comparison never fires
     * and the caller silently receives the last token in the vocabulary on
     * every step. Take the argmax instead. */
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    float r = (float)rand() / (float)RAND_MAX, cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Section timing ───────────────────────────────────────────────────────────

static const char *pf_name[PF_N] = { "embed", "rmsnorm", "qkv+bias", "rope+kv",
                                     "attention", "attn proj", "ffn matmul",
                                     "silu", "residual", "head" };
static double pf_acc[PF_N];
int pf_on = 0;

static double pf_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
double pf_mark(void) { return pf_on ? pf_now() : 0.0; }
void   pf_add(int slot, double t0) { if (pf_on) pf_acc[slot] += pf_now() - t0; }
void   pf_reset(void) { for (int i = 0; i < PF_N; i++) pf_acc[i] = 0.0; }

void pf_report(const char *phase, double wall_ms) {
    if (!pf_on) return;
    double sum = 0;
    for (int i = 0; i < PF_N; i++) sum += pf_acc[i];
    fprintf(stderr, "\n[profile] %s — %.0f ms wall, %.0f ms accounted\n",
            phase, wall_ms, sum * 1e3);
    for (int i = 0; i < PF_N; i++)
        if (pf_acc[i] > 0)
            fprintf(stderr, "  %-11s %8.0f ms  %5.1f%%\n", pf_name[i], pf_acc[i] * 1e3,
                    wall_ms > 0 ? pf_acc[i] * 1e5 / wall_ms : 0.0);
}

/* ── The call contract, checked once for every family ──────────────────────
 *
 * Before this, a token id outside the vocabulary read a row past the end of the
 * embedding table, a prompt longer than the cache wrote past the end of it, and
 * two families out of six stopped quietly at the boundary while the rest did
 * not. None of that was any family's fault: the interface never said whose job
 * it was, so it was nobody's. It is here now, and a forward's first line is a
 * call to it. */
int nt_check_call(const kv_cache *kv, const int *tokens, int n, int pos0,
                  int vocab, int n_layers, int kv_dim) {
    if (!tokens || n < 1 || pos0 < 0) return NT_E_ARG;
    if (!kv || !kv->k || !kv->v) return NT_E_CACHE;
    if (kv->n_layers < n_layers) return NT_E_CACHE;
    if (kv_dim > 0 && kv->kv_dim != kv_dim) return NT_E_CACHE;
    /* The cache is indexed by absolute position, so the last position this call
     * writes has to be inside it. Signed comparison on ints that are already
     * bounded by the cache's own dimensions. */
    if (pos0 > kv->max_seq - n) return NT_E_CAPACITY;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= vocab) return NT_E_TOKEN;
    return NT_OK;
}

const char *nt_strerror(int rc) {
    switch (rc) {
        case NT_OK:         return "ok";
        case NT_E_ARG:      return "bad argument";
        case NT_E_TOKEN:    return "token id outside the vocabulary";
        case NT_E_CAPACITY: return "sequence does not fit the cache";
        case NT_E_CACHE:    return "cache does not match the model";
        case NT_E_MEMORY:   return "allocation failed";
        case NT_E_STATE:    return "model state could not be prepared";
        default:            return "unknown error";
    }
}


/* ── a fan-out for work the matvec pool does not own ───────────────────────── */
typedef struct {
    nt_par_fn fn;
    void *ctx;
    int n, next, chunk;
} nt_par_job;

static void nt_par_drain(nt_par_job *j) {
    for (;;) {
        int i0 = __atomic_fetch_add(&j->next, j->chunk, __ATOMIC_RELAXED);
        if (i0 >= j->n) return;
        int i1 = i0 + j->chunk; if (i1 > j->n) i1 = j->n;
        j->fn(j->ctx, i0, i1);
    }
}

static void *nt_par_worker(void *p) { nt_par_drain((nt_par_job *)p); return NULL; }

void nt_par_for(nt_par_fn fn, void *ctx, int n_items, int min_items) {
    if (!fn || n_items <= 0) return;
    int nt = 0;
    const char *env = getenv("NT_ATTN_THREADS");
    if (env && *env) nt = atoi(env);
    if (nt <= 0) {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        nt = online > 0 ? (int)online : 1;
    }
    if (nt > 64) nt = 64;
    /* Pulled, not dealt: heads late in a causal prefill carry the same work as early ones,
     * but a big.LITTLE machine does not run them at the same speed. */
    if (nt <= 1 || n_items < min_items) { fn(ctx, 0, n_items); return; }

    nt_par_job job = { fn, ctx, n_items, 0, 1 };
    pthread_t th[64];
    int launched = 0;
    for (int t = 0; t + 1 < nt; t++) {
        if (pthread_create(&th[t], NULL, nt_par_worker, &job) != 0) break;
        launched++;
    }
    nt_par_drain(&job);
    for (int t = 0; t < launched; t++) pthread_join(th[t], NULL);
}
