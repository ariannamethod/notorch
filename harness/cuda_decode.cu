/* cuda_decode.cu — the llama and gemma3 decoders on an NVIDIA GPU, behind the same nt_arch.
 *
 * A model is loaded by the CPU loader of its family (arch_llama.c, arch_gemma3.c), so the
 * metadata checks, tensor lookups and dtype probes exist once; what they parsed is uploaded,
 * Q8_0 matrices still packed. The forward runs over rows that each name their cache and
 * position, exactly as the CPU forwards do: a prefill is one cache and consecutive positions,
 * forward_multi is one row per sequence. The caches live in device memory and come from this
 * family's kv_new.
 *
 * Arithmetic. Every reduction has a fixed order and no atomics, so a run repeats bit for bit
 * on the same device, and a row's result does not depend on which other rows share its call.
 * Where following the CPU order costs nothing it is followed: the Q8_0 dot is the AVX2
 * kernel's 32 partial sums per block position, (d*w) then a fused add, folded in the same
 * tree; attention's q·k is dot_f32's fold; RMSNorm and softmax sum sequentially; the RoPE
 * angles' cosines and sines are tabulated on the host with the CPU's own libm. expf, tanhf
 * and the order of a few multiplies can still differ from the CPU in the last bit; the
 * difference is measured by tests/test_cuda_decode.c, not assumed away.
 *
 * The residual callback runs on the host: the rows are copied out after each layer, the
 * callback sees them exactly as it would on the CPU, and whatever it writes is copied back.
 */
#include <math.h>      /* the C headers first, so their C++ forms are not opened inside extern "C" */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "harness/arch_models.h"
}
#include "harness/cuda_decode.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define NT_CUDA_MAX_POS 32768   /* longest cache kv_new hands out; the RoPE tables cover it */
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return NT_E_STATE; } } while (0)

typedef struct { const void *p; int dtype, rows, cols; } dwt;   /* device weight */

typedef struct {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *post_attn_norm, *post_ffw_norm;
    const float *q_bias, *k_bias, *v_bias, *ffn_down_bias;
    dwt wq, wk, wv, wo, wgate, wup, wdown;
    int local;            /* gemma3 sliding-window layer */
    const float *cs, *sn; /* this layer's RoPE table, [NT_CUDA_MAX_POS][HD/2] */
} dlayer;

typedef struct {
    int gemma;
    const nt_arch *cpu; void *cpu_model;
    int E, H, KVH, HD, QD, KVD, FF, V, L, window, neox;
    float eps, query_scale, softcap, emb_scale, attn_scale;
    dwt emb, head;
    const float *out_norm;
    dlayer *layers;       /* host array of device pointers */
    void **owned; int n_owned, cap_owned;
    int failed;           /* an upload failed: an absent optional tensor and a failed one both read NULL */
    /* scratch, grown on demand */
    int cap_rows, cap_stride, cap_head;
    float *x, *xn, *q, *k, *v, *att, *gate, *up, *out, *logits, *scores, *hx;
    int *tok, *pos, *maxseq; float **kptr, **vptr;
} cuda_model;

/* ─── kernels ─────────────────────────────────────────────────────────────────── */

__device__ __forceinline__ float fold32(float p) {
    /* The AVX2 fold over 32 lane partials: (p0..7 + p8..15) + (p16..23 + p24..31), then the
     * halves, then 0+2 and 1+3, then the last pair; lane 0 ends with the sum. */
    float a = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 8));
    float s = __fadd_rn(a, __shfl_down_sync(0xffffffffu, a, 16));
    float h = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 4));
    float t = __fadd_rn(h, __shfl_down_sync(0xffffffffu, h, 2));
    return __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 1));
}

__device__ __forceinline__ float fold_dot(float p) {
    /* dot_f32's fold: (p0..7 + p8..15) + (p16..23 + p24..31), halves, then two hadds:
     * (h0 + h1) + (h2 + h3). */
    float a = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 8));
    float s = __fadd_rn(a, __shfl_down_sync(0xffffffffu, a, 16));
    float h = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, 4));
    float t = __fadd_rn(h, __shfl_down_sync(0xffffffffu, h, 1));
    return __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 2));
}

#define TILE 8      /* activation rows per pass over a weight row */
#define SLICE 4096  /* floats of X staged per pass, 16 KB: one row of 4096, or TILE rows of 512 */
#define AHEAD 8     /* weight blocks a lane loads before multiplying them (byte path) */
/* out[j * rows + r] = W[r] . X[j]; one warp per output row, lane = position in a block, eight
 * rows to a thread block of 256.
 *
 * The warps of a block share X: each slice of it is staged in shared memory once instead of
 * being read from L2 by every warp. The weights come in one of two ways. When a row holds a
 * multiple of eight blocks, eight blocks are 272 bytes, seventeen 16-byte words starting on a
 * 16-byte boundary, and seventeen lanes load them in one instruction into a buffer of the
 * warp while the previous eight are being multiplied (VEC). Otherwise each lane loads the
 * scale and its byte of AHEAD blocks before multiplying any of them. Either way a lane's
 * multiply-adds run in block order, the order of the CPU kernel, which a staged or prefetched
 * operand does not change: tests/test_cuda_decode.c holds both paths to qmm bit for bit. */
template <bool VEC>
__global__ void __launch_bounds__(256) k_q8_matmul(float *__restrict__ out, const uint8_t *__restrict__ W,
                                                   const float *__restrict__ X, int rows, int k, int n) {
    __shared__ float xs[SLICE];
    __shared__ uint4 wbuf[8][17];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, r = blockIdx.x * 8 + warp;
    int active = r < rows, nb = k / 32;
    const uint8_t *rb = W + (size_t)(active ? r : 0) * nb * 34;
    const uint8_t *wb = (const uint8_t *)wbuf[warp];
    for (int j0 = 0; j0 < n; j0 += TILE) {
        int jn = n - j0 < TILE ? n - j0 : TILE, per = SLICE / 32 / jn;
        if (VEC) per &= ~7;
        float acc[TILE];
        #pragma unroll
        for (int t = 0; t < TILE; t++) acc[t] = 0.0f;
        for (int c0 = 0; c0 < nb; c0 += per) {
            int cn = nb - c0 < per ? nb - c0 : per, width = cn * 32;
            __syncthreads();
            for (int i = threadIdx.x; i < jn * width; i += blockDim.x)
                xs[i] = X[(size_t)(j0 + i / width) * k + (size_t)c0 * 32 + i % width];
            __syncthreads();
            if (!active) continue;
            if (VEC) {
                const uint4 *src = (const uint4 *)(rb + (size_t)c0 * 34);
                uint4 next = lane < 17 ? src[lane] : make_uint4(0, 0, 0, 0);
                for (int g = 0; g < cn; g += 8) {
                    __syncwarp();
                    if (lane < 17) wbuf[warp][lane] = next;
                    __syncwarp();
                    if (g + 8 < cn && lane < 17) next = src[(g / 8 + 1) * 17 + lane];
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        float dw = __fmul_rn(__half2float(*(const __half *)(wb + u * 34)),
                                             (float)(int8_t)wb[u * 34 + 2 + lane]);
                        #pragma unroll
                        for (int t = 0; t < TILE; t++)
                            if (t < jn) acc[t] = __fmaf_rn(dw, xs[t * width + (g + u) * 32 + lane], acc[t]);
                    }
                }
            } else {
                for (int b = 0; b < cn; b += AHEAD) {
                    float dw[AHEAD];
                    #pragma unroll
                    for (int u = 0; u < AHEAD; u++)
                        if (b + u < cn) {
                            const uint8_t *blk = rb + (size_t)(c0 + b + u) * 34;
                            dw[u] = __fmul_rn(__half2float(*(const __half *)blk), (float)(int8_t)blk[2 + lane]);
                        }
                    #pragma unroll
                    for (int u = 0; u < AHEAD; u++)
                        if (b + u < cn) {
                            #pragma unroll
                            for (int t = 0; t < TILE; t++)
                                if (t < jn) acc[t] = __fmaf_rn(dw[u], xs[t * width + (b + u) * 32 + lane], acc[t]);
                        }
                }
            }
        }
        if (!active) continue;
        for (int t = 0; t < jn; t++) {
            float s = fold32(acc[t]);
            if (lane == 0) {
                if (!isfinite(s)) {   /* the CPU kernel's retry: the sequential order */
                    float a = 0.0f;
                    const float *xj = X + (size_t)(j0 + t) * k;
                    for (int b = 0; b < nb; b++) {
                        const uint8_t *blk = rb + (size_t)b * 34;
                        float d = __half2float(*(const __half *)blk);
                        for (int i = 0; i < 32; i++)
                            a = __fmaf_rn(__fmul_rn(d, (float)(int8_t)blk[2 + i]), xj[b * 32 + i], a);
                    }
                    s = a;
                }
                out[(size_t)(j0 + t) * rows + r] = s;
            }
        }
    }
}

/* F32 or F16 weights: one warp per row, lanes stride the columns, the same fold. */
__global__ void k_f_matmul(float *out, const void *W, int f16, const float *X, int rows, int k, int n) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int c = lane; c < k; c += 32) {
            float w = f16 ? __half2float(((const __half *)W)[(size_t)r * k + c])
                          : ((const float *)W)[(size_t)r * k + c];
            acc = __fmaf_rn(w, X[(size_t)j * k + c], acc);
        }
        float s = fold32(acc);
        if (lane == 0) out[(size_t)j * rows + r] = s;
    }
}

/* Embedding rows: the dequantized row, times `scale` when it is not 1 (gemma3's sqrt(E)). */
__global__ void k_embed(float *x, dwt w, const int *tok, int n, int E, float scale) {
    int j = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n || i >= E) return;
    size_t row = (size_t)tok[j];
    float v;
    if (w.dtype == 8) {
        const uint8_t *blk = (const uint8_t *)w.p + (row * (E / 32) + i / 32) * 34;
        v = __fmul_rn(__half2float(*(const __half *)blk), (float)(int8_t)blk[2 + i % 32]);
    } else if (w.dtype == 1) v = __half2float(((const __half *)w.p)[row * E + i]);
    else v = ((const float *)w.p)[row * E + i];
    if (scale != 1.0f) v = __fmul_rn(v, scale);
    x[(size_t)j * E + i] = v;
}

/* RMSNorm per row: the row staged in shared memory by the whole block, the sum of squares
 * then sequential in one thread, as the CPU loop runs it, and w * x * inv in parallel. The
 * kernel has no static shared memory: dim + 1 floats of dynamic memory hold the row and inv,
 * so the widest row is set by the device alone (norm_width_ok). */
__global__ void k_rmsnorm(float *out, const float *x, const float *w, int dim, float eps) {
    extern __shared__ float xs[];
    const float *xr = x + (size_t)blockIdx.x * dim;
    float *o = out + (size_t)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) xs[i] = xr[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss = __fmaf_rn(xs[i], xs[i], ss);
        xs[dim] = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(ss, (float)dim), eps)));
    }
    __syncthreads();
    float inv = xs[dim];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = __fmul_rn(__fmul_rn(w[i], xs[i]), inv);
}

/* Whether this device can stage a row of `dim`: up to 48 KB of dynamic shared memory needs
 * nothing, beyond it the kernel opts into the device's larger per-block limit. */
static int norm_width_ok(int dim) {
    size_t need = ((size_t)dim + 1) * sizeof(float);
    if (need <= 48 * 1024) return 1;
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        need > (size_t)optin ||
        cudaFuncSetAttribute(k_rmsnorm, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return 1;
}

__global__ void k_add_rows(float *x, const float *bias, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim * n) x[i] = __fadd_rn(x[i], bias[i % dim]);
}
__global__ void k_add(float *x, const float *y, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __fadd_rn(x[i], y[i]);
}
__global__ void k_silu_mul(float *g, const float *u, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = g[i];
    g[i] = __fmul_rn(__fdiv_rn(v, __fadd_rn(1.0f, expf(-v))), u[i]);
}
__global__ void k_gelu_mul(float *g, const float *u, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = g[i];
    float inner = 0.7978845608028654f * v * (1.0f + 0.044715f * v * v);
    g[i] = 0.5f * v * (1.0f + tanhf(inner)) * u[i];
}
__global__ void k_softcap(float *x, float cap, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = cap * tanhf(x[i] / cap);
}

/* Per row and head: optional per-head RMSNorm, RoPE from the host tables, optional query
 * scale; the k and v rows are then written to their sequence's cache at this layer. */
__global__ void k_qk(float *q, float *k, const float *v, int H, int KVH, int HD, int neox,
                     const float *qn, const float *kn, float eps, float qscale,
                     const float *cs, const float *sn, const int *pos,
                     float *const *kptr, float *const *vptr, const int *maxseq, int layer) {
    int j = blockIdx.x, h = blockIdx.y, heads = H + KVH;
    if (h >= heads) return;
    int isq = h < H, hh = isq ? h : h - H, half = HD / 2;
    float *row = isq ? q + ((size_t)j * H + hh) * HD : k + ((size_t)j * KVH + hh) * HD;
    const float *nw = isq ? qn : kn;
    __shared__ float inv;
    if (nw) {
        if (threadIdx.x == 0) {
            float ss = 0.0f;
            for (int i = 0; i < HD; i++) ss = __fmaf_rn(row[i], row[i], ss);
            inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(ss, (float)HD), eps)));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < HD; i += blockDim.x) row[i] = __fmul_rn(__fmul_rn(nw[i], row[i]), inv);
        __syncthreads();
    }
    const float *c = cs + (size_t)pos[j] * half, *s = sn + (size_t)pos[j] * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
        float x0 = row[a], x1 = row[b];
        row[a] = __fsub_rn(__fmul_rn(x0, c[i]), __fmul_rn(x1, s[i]));
        row[b] = __fadd_rn(__fmul_rn(x0, s[i]), __fmul_rn(x1, c[i]));
    }
    __syncthreads();
    if (isq && qscale != 1.0f) for (int i = threadIdx.x; i < HD; i += blockDim.x) row[i] = __fmul_rn(row[i], qscale);
    if (!isq) {
        size_t off = ((size_t)layer * maxseq[j] + pos[j]) * (KVH * HD) + (size_t)hh * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            kptr[j][off + i] = row[i];
            vptr[j][off + i] = v[((size_t)j * KVH + hh) * HD + i];
        }
    }
}

/* Attention per row and head: q.k per position with dot_f32's fold (one warp each), the
 * softmax with a sequential max and sum, then the weighted sum of v in position order. */
__global__ void k_attn(float *out, const float *q, float *const *kptr, float *const *vptr,
                       const int *maxseq, const int *pos, int H, int KVH, int HD, int layer,
                       int window, float scale, float *scores, int stride) {
    int j = blockIdx.x, h = blockIdx.y, kvh = h / (H / KVH);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    int p = pos[j], first = window > 0 && p >= window ? p - window + 1 : 0, span = p - first + 1;
    size_t base = (size_t)layer * maxseq[j] * (KVH * HD) + (size_t)kvh * HD;
    const float *qh = q + ((size_t)j * H + h) * HD;
    float *sc = scores + ((size_t)j * H + h) * stride;
    for (int t = warp; t < span; t += warps) {
        const float *kt = kptr[j] + base + (size_t)(first + t) * (KVH * HD);
        float acc = 0.0f;
        for (int c = lane; c < HD; c += 32) acc = __fmaf_rn(qh[c], kt[c], acc);
        float s = fold_dot(acc);
        if (lane == 0) sc[t] = scale == 1.0f ? s : __fmul_rn(s, scale);
    }
    __syncthreads();
    __shared__ float mx, sum;
    if (threadIdx.x == 0) {
        float m = sc[0];
        for (int t = 1; t < span; t++) if (sc[t] > m) m = sc[t];
        mx = m;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < span; t += blockDim.x) sc[t] = expf(__fsub_rn(sc[t], mx));
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int t = 0; t < span; t++) s = __fadd_rn(s, sc[t]);
        sum = s;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < span; t += blockDim.x) sc[t] = __fdiv_rn(sc[t], sum);
    __syncthreads();
    float *o = out + ((size_t)j * H + h) * HD;
    for (int c = threadIdx.x; c < HD; c += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < span; t++)
            acc = __fmaf_rn(sc[t], vptr[j][base + (size_t)(first + t) * (KVH * HD) + c], acc);
        o[c] = acc;
    }
}

/* ─── host side ───────────────────────────────────────────────────────────────── */

static int own(cuda_model *m, void *p) {
    if (m->n_owned == m->cap_owned) {
        int cap = m->cap_owned ? m->cap_owned * 2 : 256;
        void **more = (void **)realloc(m->owned, (size_t)cap * sizeof(void *));
        if (!more) return 0;
        m->owned = more; m->cap_owned = cap;
    }
    m->owned[m->n_owned++] = p;
    return 1;
}
static const float *up_floats(cuda_model *m, const float *h, size_t n) {
    if (!h) return NULL;
    void *d = NULL;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) { m->failed = 1; return NULL; }
    if (!own(m, d)) { cudaFree(d); m->failed = 1; return NULL; }
    if (cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { m->failed = 1; return NULL; }
    return (const float *)d;
}
static int up_wt(cuda_model *m, dwt *dst, const wt *w) {
    size_t bytes;
    const void *src;
    dst->rows = w->rows; dst->cols = w->cols;
    if (w->q && w->dtype == 8 && w->cols % 32 == 0) { dst->dtype = 8; bytes = (size_t)w->rows * (w->cols / 32) * 34; src = w->q; }
    else if (w->q && w->dtype == 1) { dst->dtype = 1; bytes = (size_t)w->rows * w->cols * 2; src = w->q; }
    else if (w->q && w->dtype == 0) { dst->dtype = 0; bytes = (size_t)w->rows * w->cols * 4; src = w->q; }
    else if (w->f32) { dst->dtype = 0; bytes = (size_t)w->rows * w->cols * 4; src = w->f32; }
    else { fprintf(stderr, "cuda: weight dtype %d has no device kernel\n", w->dtype); return 0; }
    void *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess || !own(m, d)) return 0;
    if (cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    dst->p = d;
    return 1;
}

/* cos and sin of every angle a position can take, with the CPU forward's own expressions and
 * libm, so the rotation starts from the same numbers. */
static int rope_table(cuda_model *m, int gemma, float base, float scale, const float **cs, const float **sn) {
    int half = m->HD / 2;
    size_t n = (size_t)NT_CUDA_MAX_POS * half;
    float *c = (float *)malloc(n * sizeof(float)), *s = (float *)malloc(n * sizeof(float));
    if (!c || !s) { free(c); free(s); return 0; }
    for (int p = 0; p < NT_CUDA_MAX_POS; p++)
        for (int i = 0; i < half; i++) {
            float angle;
            if (gemma) angle = (float)p * scale / powf(base, 2.0f * (float)i / (float)m->HD);
            else { float freq = 1.0f / powf(base, 2.0f * i / m->HD); angle = p * freq; }
            c[(size_t)p * half + i] = cosf(angle); s[(size_t)p * half + i] = sinf(angle);
        }
    *cs = up_floats(m, c, n); *sn = up_floats(m, s, n);
    free(c); free(s);
    return *cs && *sn;
}

static void cuda_free(void *ptr) {
    cuda_model *m = (cuda_model *)ptr;
    if (!m) return;
    for (int i = 0; i < m->n_owned; i++) cudaFree(m->owned[i]);
    free(m->owned); free(m->layers); free(m->hx);
    void *scratch[] = {m->x, m->xn, m->q, m->k, m->v, m->att, m->gate, m->up, m->out, m->logits,
                       m->scores, m->tok, m->pos, m->maxseq, m->kptr, m->vptr};
    for (size_t i = 0; i < sizeof(scratch) / sizeof(scratch[0]); i++) cudaFree(scratch[i]);
    if (m->cpu && m->cpu_model) m->cpu->free(m->cpu_model);
    free(m);
}

static cuda_model *cuda_new(const nt_arch *cpu, gguf_file *gf, nt_dims *dims, void **cpu_model) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) { fprintf(stderr, "cuda: no device\n"); return NULL; }
    *cpu_model = cpu->load(gf, dims);
    if (!*cpu_model) return NULL;
    cuda_model *m = (cuda_model *)calloc(1, sizeof(cuda_model));
    if (!m) { cpu->free(*cpu_model); return NULL; }
    m->cpu = cpu; m->cpu_model = *cpu_model;
    return m;
}

static void *cuda_llama_load(gguf_file *gf, nt_dims *dims) {
    void *cm = NULL;
    cuda_model *m = cuda_new(&nt_arch_llama, gf, dims, &cm);
    if (!m) return NULL;
    llama_model *c = (llama_model *)cm;
    m->E = c->embed; m->H = c->n_heads; m->KVH = c->n_kv_heads; m->HD = c->head_dim;
    m->QD = c->q_dim; m->KVD = c->kv_dim; m->FF = c->ffn; m->V = c->vocab; m->L = c->n_layers;
    m->eps = c->rms_eps; m->neox = c->rope_neox; m->query_scale = 1.0f; m->softcap = 0.0f;
    m->emb_scale = 1.0f; m->attn_scale = 1.0f / sqrtf((float)m->HD);
    if (!norm_width_ok(m->E)) { fprintf(stderr, "cuda: embedding width %d exceeds this device's shared memory for RMSNorm\n", m->E); cuda_free(m); return NULL; }
    m->layers = (dlayer *)calloc((size_t)m->L, sizeof(dlayer));
    const float *cs, *sn;
    int ok = m->layers && up_wt(m, &m->emb, &c->tok_emb) &&
             up_wt(m, &m->head, c->has_output_weight ? &c->out_weight : &c->tok_emb) &&
             (m->out_norm = up_floats(m, c->out_norm, m->E)) && rope_table(m, 0, c->rope_base, 1.0f, &cs, &sn);
    for (int l = 0; ok && l < m->L; l++) {
        dlayer *d = &m->layers[l];
        d->attn_norm = up_floats(m, c->layers[l].attn_norm, m->E);
        d->ffn_norm = up_floats(m, c->layers[l].ffn_norm, m->E);
        d->q_norm = up_floats(m, c->layers[l].q_norm, m->HD);
        d->k_norm = up_floats(m, c->layers[l].k_norm, m->HD);
        d->q_bias = up_floats(m, c->layers[l].q_bias, m->QD);
        d->k_bias = up_floats(m, c->layers[l].k_bias, m->KVD);
        d->v_bias = up_floats(m, c->layers[l].v_bias, m->KVD);
        d->ffn_down_bias = up_floats(m, c->layers[l].ffn_down_bias, m->E);
        d->cs = cs; d->sn = sn;
        ok = d->attn_norm && d->ffn_norm &&
             up_wt(m, &d->wq, &c->layers[l].wq) && up_wt(m, &d->wk, &c->layers[l].wk) &&
             up_wt(m, &d->wv, &c->layers[l].wv) && up_wt(m, &d->wo, &c->layers[l].wo) &&
             up_wt(m, &d->wgate, &c->layers[l].wgate) && up_wt(m, &d->wup, &c->layers[l].wup) &&
             up_wt(m, &d->wdown, &c->layers[l].wdown);
    }
    if (!ok || m->failed) { cuda_free(m); return NULL; }
    fprintf(stderr, "cuda: llama family on device, E=%d L=%d V=%d\n", m->E, m->L, m->V);
    return m;
}

static void *cuda_gemma3_load(gguf_file *gf, nt_dims *dims) {
    void *cm = NULL;
    cuda_model *m = cuda_new(&nt_arch_gemma3, gf, dims, &cm);
    if (!m) return NULL;
    gemma3_model *c = (gemma3_model *)cm;
    m->gemma = 1;
    m->E = c->E; m->H = c->H; m->KVH = c->KV; m->HD = c->HD; m->QD = c->QD; m->KVD = c->KD;
    m->FF = c->FF; m->V = c->V; m->L = c->L; m->window = c->window; m->neox = 1;
    m->eps = c->eps; m->query_scale = c->query_scale; m->softcap = c->softcap;
    m->emb_scale = sqrtf((float)c->E); m->attn_scale = 1.0f;
    if (!norm_width_ok(m->E)) { fprintf(stderr, "cuda: embedding width %d exceeds this device's shared memory for RMSNorm\n", m->E); cuda_free(m); return NULL; }
    m->layers = (dlayer *)calloc((size_t)m->L, sizeof(dlayer));
    const float *gcs, *gsn, *lcs, *lsn;
    int ok = m->layers && up_wt(m, &m->emb, &c->emb) &&
             up_wt(m, &m->head, c->output.q || c->output.f32 ? &c->output : &c->emb) &&
             (m->out_norm = up_floats(m, c->norm, m->E)) &&
             rope_table(m, 1, c->base, c->rope_scale, &gcs, &gsn) &&
             rope_table(m, 1, c->local_base, 1.0f, &lcs, &lsn);
    for (int l = 0; ok && l < m->L; l++) {
        dlayer *d = &m->layers[l];
        gemma3_layer *s = &c->layers[l];
        d->local = c->window > 0 && (l + 1) % c->pattern != 0;
        d->cs = d->local ? lcs : gcs; d->sn = d->local ? lsn : gsn;
        d->attn_norm = up_floats(m, s->attn_norm, m->E);
        d->ffn_norm = up_floats(m, s->ffn_norm, m->E);
        d->q_norm = up_floats(m, s->q_norm, m->HD);
        d->k_norm = up_floats(m, s->k_norm, m->HD);
        d->post_attn_norm = up_floats(m, s->post_attn_norm, m->E);
        d->post_ffw_norm = up_floats(m, s->post_ffw_norm, m->E);
        ok = d->attn_norm && d->ffn_norm && d->q_norm && d->k_norm && d->post_attn_norm && d->post_ffw_norm &&
             up_wt(m, &d->wq, &s->q) && up_wt(m, &d->wk, &s->k) && up_wt(m, &d->wv, &s->v) &&
             up_wt(m, &d->wo, &s->o) && up_wt(m, &d->wgate, &s->gate) && up_wt(m, &d->wup, &s->up) &&
             up_wt(m, &d->wdown, &s->down);
    }
    if (!ok || m->failed) { cuda_free(m); return NULL; }
    fprintf(stderr, "cuda: gemma3 on device, E=%d L=%d V=%d\n", m->E, m->L, m->V);
    return m;
}

/* Scratch for n rows and caches up to `stride` positions, and logits for head_rows rows: the
 * head runs on the last row of a prefill only, so its buffer is not sized by the prompt. */
static int grow(cuda_model *m, int n, int head_rows, int stride) {
    if (head_rows > m->cap_head) {
        cudaFree(m->logits); m->logits = NULL; m->cap_head = 0;
        CK(cudaMalloc((void **)&m->logits, (size_t)head_rows * m->V * 4));
        m->cap_head = head_rows;
    }
    if (n <= m->cap_rows && stride <= m->cap_stride) return NT_OK;
    int rows = n > m->cap_rows ? n : m->cap_rows, st = stride > m->cap_stride ? stride : m->cap_stride;
    void **bufs[] = {(void **)&m->x, (void **)&m->xn, (void **)&m->q, (void **)&m->k, (void **)&m->v,
                     (void **)&m->att, (void **)&m->gate, (void **)&m->up, (void **)&m->out,
                     (void **)&m->scores, (void **)&m->tok, (void **)&m->pos,
                     (void **)&m->maxseq, (void **)&m->kptr, (void **)&m->vptr};
    size_t sizes[] = {(size_t)rows * m->E * 4, (size_t)rows * m->E * 4, (size_t)rows * m->QD * 4,
                      (size_t)rows * m->KVD * 4, (size_t)rows * m->KVD * 4, (size_t)rows * m->QD * 4,
                      (size_t)rows * m->FF * 4, (size_t)rows * m->FF * 4, (size_t)rows * m->E * 4,
                      (size_t)rows * m->H * st * 4, (size_t)rows * 4,
                      (size_t)rows * 4, (size_t)rows * 4, (size_t)rows * sizeof(float *),
                      (size_t)rows * sizeof(float *)};
    m->cap_rows = m->cap_stride = 0;   /* until every buffer below exists again */
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        cudaFree(*bufs[i]); *bufs[i] = NULL;
        CK(cudaMalloc(bufs[i], sizes[i]));
    }
    free(m->hx);
    m->hx = (float *)malloc((size_t)rows * m->E * sizeof(float));
    if (!m->hx) return NT_E_MEMORY;
    m->cap_rows = rows; m->cap_stride = st;
    return NT_OK;
}

static int matmul(float *out, const dwt *w, const float *X, int n) {
    int threads = 256, blocks = (w->rows * 32 + threads - 1) / threads;
    if (w->dtype == 8 && (w->cols / 32) % 8 == 0)
        k_q8_matmul<true><<<blocks, threads>>>(out, (const uint8_t *)w->p, X, w->rows, w->cols, n);
    else if (w->dtype == 8) k_q8_matmul<false><<<blocks, threads>>>(out, (const uint8_t *)w->p, X, w->rows, w->cols, n);
    else k_f_matmul<<<blocks, threads>>>(out, w->p, w->dtype == 1, X, w->rows, w->cols, n);
    CK(cudaGetLastError());
    return NT_OK;
}
static int rms(float *out, const float *x, const float *w, int dim, int n, float eps) {
    k_rmsnorm<<<n, 256, ((size_t)dim + 1) * sizeof(float)>>>(out, x, w, dim, eps);
    CK(cudaGetLastError());
    return NT_OK;
}
#define LAUNCH1(kernel, count, ...) do { int c_ = (count); \
    kernel<<<(c_ + 255) / 256, 256>>>(__VA_ARGS__); CK(cudaGetLastError()); } while (0)

/* The forward over rows, the CUDA twin of llama_rows and gemma3_rows. */
static int cuda_rows(cuda_model *m, int n, const int *tokens, kv_cache *const *kvs, const int *pos,
                     float *logits, int head_rows, nt_residual_fn cb, void *user, void *const *users) {
    int stride = 0;
    for (int j = 0; j < n; j++) if (kvs[j]->max_seq > stride) stride = kvs[j]->max_seq;
    int rc = grow(m, n, logits ? head_rows : 0, stride);   /* a cache-only call needs no logits buffer */
    if (rc) return rc;
    int E = m->E;
    int *maxseq = (int *)malloc((size_t)n * sizeof(int));
    float **kp = (float **)malloc((size_t)n * sizeof(float *)), **vp = (float **)malloc((size_t)n * sizeof(float *));
    if (!maxseq || !kp || !vp) { free(maxseq); free(kp); free(vp); return NT_E_MEMORY; }
    for (int j = 0; j < n; j++) { maxseq[j] = kvs[j]->max_seq; kp[j] = kvs[j]->k; vp[j] = kvs[j]->v; }
    cudaError_t e = cudaMemcpy(m->tok, tokens, (size_t)n * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(m->pos, pos, (size_t)n * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(m->maxseq, maxseq, (size_t)n * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(m->kptr, kp, (size_t)n * sizeof(float *), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(m->vptr, vp, (size_t)n * sizeof(float *), cudaMemcpyHostToDevice);
    free(maxseq); free(kp); free(vp);
    CK(e);
    k_embed<<<dim3((E + 255) / 256, n), 256>>>(m->x, m->emb, m->tok, n, E, m->emb_scale);
    CK(cudaGetLastError());
    for (int l = 0; l < m->L; l++) {
        dlayer *d = &m->layers[l];
        if ((rc = rms(m->xn, m->x, d->attn_norm, E, n, m->eps))) return rc;
        if ((rc = matmul(m->q, &d->wq, m->xn, n)) || (rc = matmul(m->k, &d->wk, m->xn, n)) ||
            (rc = matmul(m->v, &d->wv, m->xn, n))) return rc;
        if (d->q_bias) LAUNCH1(k_add_rows, m->QD * n, m->q, d->q_bias, m->QD, n);
        if (d->k_bias) LAUNCH1(k_add_rows, m->KVD * n, m->k, d->k_bias, m->KVD, n);
        if (d->v_bias) LAUNCH1(k_add_rows, m->KVD * n, m->v, d->v_bias, m->KVD, n);
        k_qk<<<dim3(n, m->H + m->KVH), 128>>>(m->q, m->k, m->v, m->H, m->KVH, m->HD, m->neox,
            d->q_norm, d->k_norm, m->eps, m->gemma ? m->query_scale : 1.0f, d->cs, d->sn, m->pos,
            m->kptr, m->vptr, m->maxseq, l);
        CK(cudaGetLastError());
        k_attn<<<dim3(n, m->H), 128>>>(m->att, m->q, m->kptr, m->vptr, m->maxseq, m->pos, m->H, m->KVH,
            m->HD, l, d->local ? m->window : 0, m->attn_scale, m->scores, stride);
        CK(cudaGetLastError());
        if ((rc = matmul(m->out, &d->wo, m->att, n))) return rc;
        if (m->gemma && (rc = rms(m->out, m->out, d->post_attn_norm, E, n, m->eps))) return rc;
        LAUNCH1(k_add, E * n, m->x, m->out, E * n);
        if ((rc = rms(m->xn, m->x, d->ffn_norm, E, n, m->eps))) return rc;
        if ((rc = matmul(m->gate, &d->wgate, m->xn, n)) || (rc = matmul(m->up, &d->wup, m->xn, n))) return rc;
        if (m->gemma) LAUNCH1(k_gelu_mul, m->FF * n, m->gate, m->up, m->FF * n);
        else LAUNCH1(k_silu_mul, m->FF * n, m->gate, m->up, m->FF * n);
        if ((rc = matmul(m->out, &d->wdown, m->gate, n))) return rc;
        if (m->gemma && (rc = rms(m->out, m->out, d->post_ffw_norm, E, n, m->eps))) return rc;
        LAUNCH1(k_add, E * n, m->x, m->out, E * n);
        if (d->ffn_down_bias) LAUNCH1(k_add_rows, E * n, m->x, d->ffn_down_bias, E, n);
        if (cb) {
            CK(cudaMemcpy(m->hx, m->x, (size_t)n * E * 4, cudaMemcpyDeviceToHost));
            if (!users) rc = cb(user, l, pos[0], n, E, m->hx);
            else for (int j = 0; j < n && rc == NT_OK; j++) rc = cb(users[j], l, pos[j], 1, E, m->hx + (size_t)j * E);
            if (rc != NT_OK) return rc;
            CK(cudaMemcpy(m->x, m->hx, (size_t)n * E * 4, cudaMemcpyHostToDevice));
        }
    }
    if (logits) {
        int first = n - head_rows;
        if ((rc = rms(m->xn, m->x + (size_t)first * E, m->out_norm, E, head_rows, m->eps))) return rc;
        if ((rc = matmul(m->logits, &m->head, m->xn, head_rows))) return rc;
        if (m->softcap > 0) LAUNCH1(k_softcap, m->V * head_rows, m->logits, m->softcap, m->V * head_rows);
        CK(cudaMemcpy(logits, m->logits, (size_t)head_rows * m->V * 4, cudaMemcpyDeviceToHost));
    }
    CK(cudaDeviceSynchronize());
    return NT_OK;
}

static int cuda_kv_layers(const cuda_model *m) { return m->L; }

/* A cache from the host kv_new would be read by the kernels as a device address; it is
 * refused here, as a cache of the wrong shape is. */
static int on_device(const kv_cache *kv) {
    cudaPointerAttributes a;
    for (int i = 0; i < 2; i++) {
        const void *p = i ? (const void *)kv->v : (const void *)kv->k;
        if (cudaPointerGetAttributes(&a, p) != cudaSuccess || a.type != cudaMemoryTypeDevice) {
            cudaGetLastError();
            return 0;
        }
    }
    return 1;
}

static int cuda_forward_residual(void *model, kv_cache *kv, const int *tokens, int n, int pos0,
                                 float *logits, nt_residual_fn cb, void *user) {
    cuda_model *m = (cuda_model *)model;
    if (!m) return NT_E_ARG;
    int rc = nt_check_call(kv, tokens, n, pos0, m->V, cuda_kv_layers(m), m->KVD);
    if (rc) return rc;
    if (!on_device(kv)) return NT_E_CACHE;
    int *pos = (int *)malloc((size_t)n * sizeof(int));
    kv_cache **kvs = (kv_cache **)malloc((size_t)n * sizeof(kv_cache *));
    if (!pos || !kvs) { free(pos); free(kvs); return NT_E_MEMORY; }
    for (int j = 0; j < n; j++) { pos[j] = pos0 + j; kvs[j] = kv; }
    rc = cuda_rows(m, n, tokens, kvs, pos, logits, 1, cb, user, NULL);
    free(pos); free(kvs);
    return rc;
}
static int cuda_forward(void *model, kv_cache *kv, const int *tokens, int n, int pos0, float *logits) {
    return cuda_forward_residual(model, kv, tokens, n, pos0, logits, NULL, NULL);
}
static int cuda_forward_multi(void *model, kv_cache *const *kvs, const int *tokens, const int *pos,
                              int n, float *logits, nt_residual_fn cb, void *const *users) {
    cuda_model *m = (cuda_model *)model;
    if (!m || !kvs || !tokens || !pos || n < 1 || (cb && !users)) return NT_E_ARG;
    for (int j = 0; j < n; j++) {
        if (!kvs[j]) return NT_E_ARG;
        for (int i = 0; i < j; i++) if (kvs[i] == kvs[j]) return NT_E_ARG;
        int rc = nt_check_call(kvs[j], &tokens[j], 1, pos[j], m->V, cuda_kv_layers(m), m->KVD);
        if (rc) return rc;
        if (!on_device(kvs[j])) return NT_E_CACHE;
    }
    return cuda_rows(m, n, tokens, kvs, pos, logits, n, cb, NULL, users);
}

static kv_cache *cuda_kv_new(void *model, int max_seq) {
    cuda_model *m = (cuda_model *)model;
    if (!m || max_seq < 1 || max_seq > NT_CUDA_MAX_POS) return NULL;
    kv_cache *kv = (kv_cache *)calloc(1, sizeof(kv_cache));
    if (!kv) return NULL;
    size_t bytes = (size_t)m->L * max_seq * m->KVD * sizeof(float);
    if (cudaMalloc((void **)&kv->k, bytes) != cudaSuccess) { free(kv); return NULL; }
    if (cudaMalloc((void **)&kv->v, bytes) != cudaSuccess) { cudaFree(kv->k); free(kv); return NULL; }
    cudaMemset(kv->k, 0, bytes); cudaMemset(kv->v, 0, bytes);
    kv->max_seq = max_seq; kv->n_layers = m->L; kv->kv_dim = m->KVD;
    return kv;
}
static void cuda_kv_free(void *model, kv_cache *kv) {
    (void)model;
    if (!kv) return;
    cudaFree(kv->k); cudaFree(kv->v); free(kv);
}

/* The decoder's matmul on its own, for checking it against qmm: W and X are uploaded for
 * this call and freed after it, so it is a test entry, not a hot path. */
extern "C" int nt_cuda_qmm(float *out, const wt *w, const float *X, int n) {
    if (!out || !w || !X || n < 1) return NT_E_ARG;
    cuda_model tmp;
    memset(&tmp, 0, sizeof(tmp));
    dwt d;
    float *dx = NULL, *dout = NULL;
    int rc = NT_E_STATE;
    if (up_wt(&tmp, &d, w) &&
        cudaMalloc((void **)&dx, (size_t)n * d.cols * 4) == cudaSuccess &&
        cudaMalloc((void **)&dout, (size_t)n * d.rows * 4) == cudaSuccess &&
        cudaMemcpy(dx, X, (size_t)n * d.cols * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
        matmul(dout, &d, dx, n) == NT_OK &&
        cudaMemcpy(out, dout, (size_t)n * d.rows * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
        rc = NT_OK;
    cudaFree(dx); cudaFree(dout);
    for (int i = 0; i < tmp.n_owned; i++) cudaFree(tmp.owned[i]);
    free(tmp.owned);
    return rc;
}

static const char *const llama_cuda_names[] = {"llama", "mistral3", "qwen2", "qwen3", NULL};
static const char *const gemma3_cuda_names[] = {"gemma3", NULL};

extern "C" {
const nt_arch nt_arch_llama_cuda = {llama_cuda_names, cuda_llama_load, cuda_free, cuda_forward,
                                    cuda_forward_residual, cuda_forward_multi, cuda_kv_new, cuda_kv_free};
const nt_arch nt_arch_gemma3_cuda = {gemma3_cuda_names, cuda_gemma3_load, cuda_free, cuda_forward,
                                     cuda_forward_residual, cuda_forward_multi, cuda_kv_new, cuda_kv_free};

/* The device twin of a CPU family, or NULL when there is none. */
const nt_arch *nt_cuda_arch_for(const nt_arch *cpu) {
    if (cpu == &nt_arch_llama) return &nt_arch_llama_cuda;
    if (cpu == &nt_arch_gemma3) return &nt_arch_gemma3_cuda;
    return NULL;
}
}
