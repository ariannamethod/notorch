// notorch.c — Neural Networks in pure C
// Extracted from ariannamethod.ai/core/ (Arianna Method)
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/* sched_getaffinity is a GNU extension behind _GNU_SOURCE on glibc; Bionic and musl expose
 * it unguarded. The define has to land before the first libc header so features.h sees it. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#include <stdlib.h>

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_SIMD
  #ifdef USE_BLAS
    #error "USE_SIMD and USE_BLAS are mutually exclusive — pick one matmul backend."
  #endif
  // In-house AVX2 + FMA shim for cblas_sgemm / sgemv / sger.
  // Lets every existing cblas_* call site stay unchanged.
  #ifdef NOTORCH_SIMD_DEBUG_SCALAR
    #include "notorch_simd_scalar.h"
  #else
    #include "notorch_simd.h"
  #endif
  // Also satisfy the original `#ifdef USE_BLAS` guards in this file by aliasing
  // them on. The shim defines the same CBLAS_* enums and functions.
  #define USE_BLAS 1
#endif

#ifdef USE_CUDA
  #include "notorch_cuda.h"
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// GPU MODE — runtime flag + per-tensor lazy CPU↔GPU mirror helpers
// All compiled out when USE_CUDA is undefined.
// ═══════════════════════════════════════════════════════════════════════════════

static int g_use_gpu = 0;

void nt_set_gpu_mode(int on_off) {
#ifdef USE_CUDA
    g_use_gpu = on_off ? 1 : 0;
#else
    (void)on_off;
    g_use_gpu = 0;
#endif
}

int nt_get_gpu_mode(void) { return g_use_gpu; }

#ifdef USE_CUDA
// Lazy upload: ensure t->d_data is allocated and contains current CPU values.
// If gpu_valid == 1 the GPU buffer is up to date and no transfer happens.
// If cpu_dirty == 1 the GPU is the source of truth — caller already wrote
// there. Do not overwrite it with stale CPU data.
static float* nt_tensor_ensure_gpu(nt_tensor* t) {
    if (!t || t->len <= 0) return NULL;
    if (!t->d_data) {
        t->d_data = gpu_alloc(t->len);
        t->gpu_valid = 0;
    }
    if (!t->gpu_valid && !t->cpu_dirty && t->d_data) {
        gpu_upload(t->d_data, t->data, t->len);
        t->gpu_valid = 1;
    }
    return t->d_data;
}

// Lazy download: pull GPU data into CPU mirror only if a CPU op needs it.
// Called at the start of any CPU-only op that reads tensor data.
static void nt_tensor_ensure_cpu(nt_tensor* t) {
    if (!t || !t->d_data || !t->cpu_dirty) return;
    gpu_download(t->data, t->d_data, t->len);
    t->cpu_dirty = 0;
}

// Mark a tensor as freshly written by a GPU kernel: GPU is source of truth,
// CPU mirror is stale. Avoids the eager D2H copy of v1 dispatch (one transfer
// per op was killing throughput more than the kernels saved).
static void nt_tensor_mark_gpu_fresh(nt_tensor* t) {
    if (!t) return;
    t->gpu_valid = 1;
    t->cpu_dirty = 1;
}

// Mark CPU as authoritative (e.g. after Chuck step on CPU writes weights).
static void nt_tensor_mark_cpu_dirty(nt_tensor* t) {
    if (!t) return;
    t->gpu_valid = 0;  /* next ensure_gpu re-uploads */
    t->cpu_dirty = 0;  /* CPU is now the source of truth */
}
#endif

/* Public sync wrapper for external callers (e.g. nanoarianna LoRA trainer that
 * needs to read a parameter's CPU data after Chuck step). On non-USE_CUDA build
 * this is a no-op since CPU is always authoritative. */
void nt_tensor_sync_cpu(nt_tensor* t) {
#ifdef USE_CUDA
    nt_tensor_ensure_cpu(t);
#else
    (void)t;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// RNG
// ═══════════════════════════════════════════════════════════════════════════════

static uint64_t g_rng_state = 2463534242ULL;

void nt_seed(uint64_t seed) {
    g_rng_state = seed ? seed : 2463534242ULL;
}

static uint32_t xorshift32(void) {
    uint64_t s = g_rng_state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    g_rng_state = s;
    return (uint32_t)s;
}

static float rand_uniform(void) {
    return (float)xorshift32() / 4294967296.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR
// ═══════════════════════════════════════════════════════════════════════════════

static void compute_strides(nt_tensor* t) {
    if (t->ndim <= 0) return;
    t->stride[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--)
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
}

nt_tensor* nt_tensor_new(size_t len) {
    if (len == 0 || len > NT_MAX_ELEMENTS) return NULL;
    nt_tensor* t = (nt_tensor*)calloc(1, sizeof(nt_tensor));
    if (!t) return NULL;
    t->data = (float*)calloc(len, sizeof(float));
    if (!t->data) { free(t); return NULL; }
    t->len = (int)len;
    t->ndim = 1;
    t->shape[0] = (int)len;
    t->stride[0] = 1;
    t->refcount = 1;
    return t;
}

nt_tensor* nt_tensor_new2d(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    size_t total = (size_t)rows * cols;
    if (total > NT_MAX_ELEMENTS) return NULL;
    nt_tensor* t = nt_tensor_new(total);
    if (!t) return NULL;
    t->ndim = 2;
    t->shape[0] = rows;
    t->shape[1] = cols;
    compute_strides(t);
    return t;
}

nt_tensor* nt_tensor_new_shape(const int* shape, int ndim) {
    if (ndim <= 0 || ndim > NT_MAX_DIMS) return NULL;
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
        total *= (size_t)shape[i];
        if (total > NT_MAX_ELEMENTS) return NULL;
    }
    nt_tensor* t = nt_tensor_new(total);
    if (!t) return NULL;
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    compute_strides(t);
    return t;
}

void nt_tensor_free(nt_tensor* t) {
    if (!t) return;
    t->refcount--;
    if (t->refcount <= 0) {
        free(t->data);
#ifdef USE_CUDA
        if (t->d_data) { gpu_free(t->d_data); t->d_data = NULL; }
#endif
        free(t);
    }
}

nt_tensor* nt_tensor_ref(nt_tensor* t) {
    if (t) t->refcount++;
    return t;
}

nt_tensor* nt_tensor_clone(const nt_tensor* src) {
    if (!src) return NULL;
    nt_tensor* dst = nt_tensor_new(src->len);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->len * sizeof(float));
    dst->ndim = src->ndim;
    for (int i = 0; i < src->ndim; i++) {
        dst->shape[i] = src->shape[i];
        dst->stride[i] = src->stride[i];
    }
    return dst;
}

void nt_tensor_fill(nt_tensor* t, float val) {
    if (!t) return;
    for (int i = 0; i < t->len; i++) t->data[i] = val;
}

void nt_tensor_rand(nt_tensor* t, float scale) {
    if (!t) return;
    for (int i = 0; i < t->len; i++)
        t->data[i] = (2.0f * rand_uniform() - 1.0f) * scale;
}

void nt_tensor_xavier(nt_tensor* t, int fan_in, int fan_out) {
    if (!t || fan_in <= 0 || fan_out <= 0) return;
    float scale = sqrtf(6.0f / (float)(fan_in + fan_out));
    nt_tensor_rand(t, scale);
}

void nt_kaiming_uniform_init(nt_tensor* t, int fan_in) {
    // Uniform in [-sqrt(3/fan_in), +sqrt(3/fan_in)] → variance a²/3 = 1/fan_in.
    if (!t || fan_in <= 0) return;
    float scale = sqrtf(3.0f / (float)fan_in);
    nt_tensor_rand(t, scale);
}

int nt_tensor_reshape(nt_tensor* t, const int* new_shape, int new_ndim) {
    if (!t || new_ndim <= 0 || new_ndim > NT_MAX_DIMS) return -1;
    int total = 1;
    for (int i = 0; i < new_ndim; i++) total *= new_shape[i];
    if (total != t->len) return -1;
    t->ndim = new_ndim;
    for (int i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];
    compute_strides(t);
    return 0;
}

void nt_tensor_print(const nt_tensor* t, const char* name) {
    if (!t) { printf("%s: NULL\n", name ? name : "tensor"); return; }
    printf("%s: [", name ? name : "tensor");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? "×" : "");
    }
    printf("] (%d params)", t->len);
    if (t->len > 0) {
        printf(" first=%.4f", t->data[0]);
        if (t->len > 1) printf(" last=%.4f", t->data[t->len - 1]);
    }
    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD TAPE
// ═══════════════════════════════════════════════════════════════════════════════

static nt_tape g_tape = {0};

void nt_tape_start(void) {
    nt_tape_clear();
    g_tape.active = 1;
}

void nt_tape_clear(void) {
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output)
            nt_tensor_free(g_tape.entries[i].output);
        if (g_tape.entries[i].grad) {
            nt_tensor_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
        /* Reset frozen flag — defense-in-depth so reused slots can't leak
         * frozen=1 from prior session into ops that don't init it explicitly. */
        g_tape.entries[i].frozen = 0;
    }
    g_tape.count = 0;
    g_tape.active = 0;
    g_tape.n_params = 0;
}

void nt_tape_destroy(void) {
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output) {
            nt_tensor_free(g_tape.entries[i].output);
            g_tape.entries[i].output = NULL;
        }
        if (g_tape.entries[i].grad) {
            nt_tensor_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
    }
    for (int i = 0; i < g_tape.n_params; i++) {
        if (g_tape.adam[i].m) { nt_tensor_free(g_tape.adam[i].m); g_tape.adam[i].m = NULL; }
        if (g_tape.adam[i].v) { nt_tensor_free(g_tape.adam[i].v); g_tape.adam[i].v = NULL; }
        if (g_tape.adam[i].acc_grad) { nt_tensor_free(g_tape.adam[i].acc_grad); g_tape.adam[i].acc_grad = NULL; }
        g_tape.adam[i].t = 0;
    }
    memset(&g_tape, 0, sizeof(g_tape));
}

int nt_tape_is_active(void) { return g_tape.active; }
nt_tape* nt_tape_get(void) { return &g_tape; }

int nt_tape_record(nt_tensor* output, int op, int p1, int p2, float aux) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = -1;
    e->aux = aux;
    e->aux2 = 0;
    e->is_param = 0;
    e->no_decay = 0;
    e->frozen = 0;  /* clear leftover from prior tape session sharing this slot */
    g_tape.count++;
    return idx;
}

int nt_tape_record3(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = p3;
    e->aux = aux;
    e->aux2 = aux2;
    e->is_param = 0;
    e->no_decay = 0;
    e->frozen = 0;  /* clear leftover from prior tape session sharing this slot */
    g_tape.count++;
    return idx;
}

int nt_tape_record4(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2, float aux3, float aux4) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = p3;
    e->aux = aux;
    e->aux2 = aux2;
    e->aux3 = aux3;
    e->aux4 = aux4;
    e->is_param = 0;
    e->no_decay = 0;
    e->frozen = 0;  /* clear leftover from prior tape session sharing this slot */
    g_tape.count++;
    return idx;
}

int nt_tape_param(nt_tensor* param) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = param;
    nt_tensor_ref(param);
    e->grad = NULL;
    e->op = NT_OP_NONE;
    e->parent1 = -1;
    e->parent2 = -1;
    e->parent3 = -1;
    e->aux = 0;
    e->aux2 = 0;
    e->is_param = 1;
    e->frozen = 0;  /* clear leftover from prior tape session sharing this slot */
    e->no_decay = 0;
    e->frozen = 0;       // explicit reset — prevents sticky frozen flag from
                         // a previous nt_tape_param_frozen() that reused this slot.
                         // Per Codex notorch-pass-1 P2 #1.

    if (g_tape.n_params < NT_TAPE_MAX_PARAMS) {
        int pi = g_tape.n_params;
        if (!g_tape.adam[pi].m) {
            g_tape.adam[pi].m = nt_tensor_new(param->len);
            g_tape.adam[pi].v = nt_tensor_new(param->len);
            g_tape.adam[pi].t = 0;
        } else if (g_tape.adam[pi].m->len != param->len) {
            nt_tensor* new_m = nt_tensor_new(param->len);
            nt_tensor* new_v = nt_tensor_new(param->len);
            int copy_len = g_tape.adam[pi].m->len < param->len ? g_tape.adam[pi].m->len : param->len;
            memcpy(new_m->data, g_tape.adam[pi].m->data, copy_len * sizeof(float));
            memcpy(new_v->data, g_tape.adam[pi].v->data, copy_len * sizeof(float));
            nt_tensor_free(g_tape.adam[pi].m);
            nt_tensor_free(g_tape.adam[pi].v);
            g_tape.adam[pi].m = new_m;
            g_tape.adam[pi].v = new_v;
        }
        g_tape.n_params++;
    }

    g_tape.count++;
    return idx;
}

void nt_tape_no_decay(int idx) {
    if (idx >= 0 && idx < g_tape.count)
        g_tape.entries[idx].no_decay = 1;
}

void nt_tape_freeze_param(int param_idx) {
    if (param_idx >= 0 && param_idx < g_tape.n_params)
        g_tape.chuck_params[param_idx].frozen = 1;
    // Also set the per-entry frozen flag so backward can skip computation.
    // Note: param_idx in this API is the *tape entry index*, returned by nt_tape_param().
    if (param_idx >= 0 && param_idx < g_tape.count)
        g_tape.entries[param_idx].frozen = 1;
}

int nt_tape_param_frozen(nt_tensor* param) {
    // Mirror nt_tape_param body, but DO NOT allocate a Chuck optimizer slot.
    // Set entry->frozen=1 so backward skips dw accumulation.
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = param;
    nt_tensor_ref(param);
    e->grad = NULL;
    e->op = NT_OP_NONE;
    e->parent1 = -1;
    e->parent2 = -1;
    e->parent3 = -1;
    e->aux = 0;
    e->aux2 = 0;
    e->is_param = 1;
    e->no_decay = 0;
    e->frozen = 1;            // backward skips dw via this flag (notorch.c:845 path)
    // INTENTIONAL: do NOT increment g_tape.n_params, do NOT touch g_tape.adam[].
    // Chuck slots stay 1:1 with truly trainable params registered via nt_tape_param().
    g_tape.count++;
    return idx;
}

// Accumulate gradient into a tape entry
static void tape_acc_grad(int idx, const float* grad, int len) {
    if (idx < 0 || idx >= g_tape.count) return;
    nt_tape_entry* e = &g_tape.entries[idx];
    if (e->frozen) return;   // skip allocation + accumulation for frozen params
    if (!e->grad) {
        e->grad = nt_tensor_new(len);
        if (!e->grad) return;
    }
#ifdef USE_CUDA
    /* If GPU is the source of truth for e->grad, sync to CPU first so this
     * CPU contribution lands on the latest accumulated value. */
    nt_tensor_ensure_cpu(e->grad);
#endif
    int n = e->grad->len < len ? e->grad->len : len;
    for (int i = 0; i < n; i++) e->grad->data[i] += grad[i];
#ifdef USE_CUDA
    /* CPU just modified — invalidate GPU mirror. */
    e->grad->gpu_valid = 0;
    e->grad->cpu_dirty = 0;
#endif
}

#ifdef USE_CUDA
/* Accumulate a GPU-resident contribution into e->grad's GPU buffer.
 * If e->grad doesn't have GPU storage yet, allocate + zero. If e->grad
 * is currently CPU-fresh, upload the existing CPU values first so the
 * GPU buffer sees full accumulated state, then axpy. */
static void tape_acc_grad_gpu(int idx, const float* d_grad, int len) {
    if (idx < 0 || idx >= g_tape.count) return;
    nt_tape_entry* e = &g_tape.entries[idx];
    if (e->frozen) return;
    if (!e->grad) {
        e->grad = nt_tensor_new(len);
        if (!e->grad) return;
    }
    int n = e->grad->len < len ? e->grad->len : len;
    /* Ensure GPU buffer exists and contains current CPU state. */
    float* d_dst = nt_tensor_ensure_gpu(e->grad);
    if (!d_dst) return;
    /* Use cuBLAS axpy: dst += d_grad. */
    extern void gpu_axpy(float* d_y, const float* d_x, int n, float alpha);
    gpu_axpy(d_dst, d_grad, n, 1.0f);
    /* GPU is now fresh. */
    e->grad->gpu_valid = 1;
    e->grad->cpu_dirty = 1;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// BACKWARD PASS
// ═══════════════════════════════════════════════════════════════════════════════

void nt_tape_backward(int loss_idx) {
    if (loss_idx < 0 || loss_idx >= g_tape.count) return;

    /* Lazy GPU/CPU mirror model — no eager D2H prelude. Each bw op-case is
     * responsible for either staying GPU-resident (GPU branch) or pulling
     * the specific parents/grads it consumes via nt_tensor_ensure_cpu().
     * Avoids the avg ~18% GPU-utilization ceiling caused by syncing all
     * activations at the start of backward. */

    nt_tape_entry* loss = &g_tape.entries[loss_idx];
    if (!loss->grad) loss->grad = nt_tensor_new(loss->output->len);
    for (int i = 0; i < loss->grad->len; i++) loss->grad->data[i] = 1.0f;
#ifdef USE_CUDA
    /* Loss grad is a CPU-authored fresh value — invalidate any stale GPU mirror. */
    loss->grad->gpu_valid = 0;
    loss->grad->cpu_dirty = 0;
#endif

    for (int idx = loss_idx; idx >= 0; idx--) {
        nt_tape_entry* e = &g_tape.entries[idx];
        if (!e->grad) continue;
#ifdef USE_CUDA
        /* CPU bw cases read e->grad->data (`dout`) directly. If a downstream
         * GPU bw kernel deposited the grad via tape_acc_grad_gpu, the CPU
         * mirror is stale — pull it down now. Cost: one D2H per active grad.
         * GPU bw cases that ensure_gpu(e->grad) below will see cpu_dirty=0,
         * gpu_valid=1 and skip the upload. */
        nt_tensor_ensure_cpu(e->grad);
#endif
        float* dout = e->grad->data;
        int out_len = e->output->len;

        switch (e->op) {

        case NT_OP_ADD: {
#ifdef USE_CUDA
            if (g_use_gpu) {
                int p1_match = e->parent1 >= 0 &&
                    g_tape.entries[e->parent1].output &&
                    g_tape.entries[e->parent1].output->len == out_len;
                int p2_match = e->parent2 >= 0 &&
                    g_tape.entries[e->parent2].output &&
                    g_tape.entries[e->parent2].output->len == out_len;
                if (p1_match && p2_match) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    if (d_dout) {
                        tape_acc_grad_gpu(e->parent1, d_dout, out_len);
                        tape_acc_grad_gpu(e->parent2, d_dout, out_len);
                        break;
                    }
                }
            }
#endif
            if (e->parent1 >= 0) tape_acc_grad(e->parent1, dout, out_len);
            if (e->parent2 >= 0) tape_acc_grad(e->parent2, dout, out_len);
            break;
        }

        case NT_OP_MUL: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pa = &g_tape.entries[e->parent1];
                nt_tape_entry* pb = &g_tape.entries[e->parent2];
#ifdef USE_CUDA
                /* L2 (2026-06-03): GPU mul backward — gpu_mul_backward existed but
                 * was unused, so each MUL did a D2H sync (SwiGLU + gate-blend = 3
                 * MULs/hybrid layer → ~30 mid-backward stalls/step, the residual
                 * 0%-util cause after L1). GPU path reads parent outputs on-device
                 * (NO sync_cpu — that download is exactly what the CPU path guards;
                 * tape_acc_grad_gpu sets gpu_valid/cpu_dirty, mirroring NT_OP_SCALE). */
                if (g_use_gpu) {
                    extern void gpu_mul_backward(float*, float*, const float*, const float*, const float*, int);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_a = nt_tensor_ensure_gpu(pa->output);
                    float* d_b = nt_tensor_ensure_gpu(pb->output);
                    float* d_ga = gpu_scratch(3, out_len);
                    float* d_gb = gpu_scratch(4, out_len);
                    if (d_dout && d_a && d_b && d_ga && d_gb) {
                        gpu_mul_backward(d_ga, d_gb, d_dout, d_a, d_b, out_len);
                        tape_acc_grad_gpu(e->parent1, d_ga, out_len);
                        tape_acc_grad_gpu(e->parent2, d_gb, out_len);
                        break;
                    }
                }
#endif
                /* SwiGLU / gate-blend FIX 2026-05-11: forward output of both
                 * parents may live on GPU; CPU mirror is stale calloc-zero.
                 * Without sync, ga=gb=0 — masks all LoRA gradients on the
                 * mlp_gate + mlp_up SwiGLU branch. */
                nt_tensor_sync_cpu(pa->output);
                nt_tensor_sync_cpu(pb->output);
                float* ga = (float*)calloc(out_len, sizeof(float));
                float* gb = (float*)calloc(out_len, sizeof(float));
                if (ga && gb) {
                    for (int i = 0; i < out_len; i++) {
                        ga[i] = dout[i] * pb->output->data[i];
                        gb[i] = dout[i] * pa->output->data[i];
                    }
                    tape_acc_grad(e->parent1, ga, out_len);
                    tape_acc_grad(e->parent2, gb, out_len);
                }
                free(ga); free(gb);
            }
            break;
        }

        case NT_OP_SCALE: {
            if (e->parent1 >= 0) {
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_ga   = gpu_scratch(3, out_len);
                    if (d_dout && d_ga) {
                        gpu_scale(d_ga, d_dout, out_len, e->aux);
                        tape_acc_grad_gpu(e->parent1, d_ga, out_len);
                        break;
                    }
                }
#endif
                float* ga = (float*)calloc(out_len, sizeof(float));
                if (ga) {
                    for (int i = 0; i < out_len; i++) ga[i] = dout[i] * e->aux;
                    tape_acc_grad(e->parent1, ga, out_len);
                }
                free(ga);
            }
            break;
        }

        case NT_OP_MATVEC: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0) {
                    float* dw = (float*)calloc((size_t)rows * cols, sizeof(float));
                    if (dw) {
                        for (int i = 0; i < rows; i++)
                            for (int j = 0; j < cols; j++)
                                dw[i * cols + j] = dout[i] * px->output->data[j];
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc(cols, sizeof(float));
                    if (dx) {
                        for (int j = 0; j < cols; j++)
                            for (int i = 0; i < rows; i++)
                                dx[j] += pw->output->data[i * cols + j] * dout[i];
                        tape_acc_grad(e->parent2, dx, cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        case NT_OP_SILU: {
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
#ifdef USE_CUDA
                /* L2 (2026-06-03): GPU silu backward — kernel existed, was unused
                 * (one D2H sync/SiLU/hybrid layer). GPU path reads x on-device. */
                if (g_use_gpu) {
                    extern void gpu_silu_backward(float*, const float*, const float*, int);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_x = nt_tensor_ensure_gpu(px->output);
                    float* d_gx = gpu_scratch(3, out_len);
                    if (d_dout && d_x && d_gx) {
                        gpu_silu_backward(d_gx, d_dout, d_x, out_len);
                        tape_acc_grad_gpu(e->parent1, d_gx, out_len);
                        break;
                    }
                }
#endif
                /* FIX 2026-05-11: parent output may be GPU-resident; CPU stale
                 * gives sigmoid(0)=0.5 partial grad — still corrupts the SiLU
                 * derivative used in SwiGLU mlp_gate path. */
                nt_tensor_sync_cpu(px->output);
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float x = px->output->data[i];
                        float sig = 1.0f / (1.0f + expf(-x));
                        gx[i] = dout[i] * sig * (1.0f + x * (1.0f - sig));
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SIGMOID: {
            /* y = sigmoid(x); dy/dx = y * (1 - y) */
            if (e->parent1 >= 0) {
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float y = e->output->data[i];
                        gx[i] = dout[i] * y * (1.0f - y);
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_RELU: {
            /* y = max(0, x); dy/dx = (y > 0) ? 1 : 0  (y>0 ⟺ x>0) */
            if (e->parent1 >= 0) {
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        gx[i] = (e->output->data[i] > 0.0f) ? dout[i] : 0.0f;
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SEQ_GATE: {
            /* out[t,d] = x[t,d] * g[t,gi];
             * dx[t,d] = dout[t,d] * g[t,gi];  dg[t,gi] = Σ_d dout[t,d] * x[t,d] */
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                nt_tape_entry* pg = &g_tape.entries[e->parent2];
                int T = (int)e->aux, nm = (int)e->aux2, gi = (int)e->aux3;
                int B = (T > 0) ? out_len / T : 0;
                nt_tensor_sync_cpu(px->output);
                nt_tensor_sync_cpu(pg->output);
                float* dx = (float*)calloc(out_len, sizeof(float));
                float* dg = (float*)calloc(pg->output->len, sizeof(float));
                if (dx && dg) {
                    for (int t = 0; t < T; t++) {
                        float gv = pg->output->data[t * nm + gi];
                        float acc = 0.0f;
                        for (int d = 0; d < B; d++) {
                            dx[t * B + d] = dout[t * B + d] * gv;
                            acc += dout[t * B + d] * px->output->data[t * B + d];
                        }
                        dg[t * nm + gi] = acc;
                    }
                    tape_acc_grad(e->parent1, dx, out_len);
                    tape_acc_grad(e->parent2, dg, pg->output->len);
                }
                free(dx); free(dg);
            }
            break;
        }

        case NT_OP_SCALE_BY_T: {
            /* y = a[0] * x; gx = a[0] * dout; ga = sum(dout * x) */
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                nt_tape_entry* pa = &g_tape.entries[e->parent2];
                /* GPU-sync FIX (2026-06-02): px (the scaled tensor) is often a
                 * GPU-fresh attention output; without sync ga = Σ dout·x reads
                 * stale calloc-zero and the gate gradient vanishes. */
                nt_tensor_sync_cpu(px->output);
                nt_tensor_sync_cpu(pa->output);
                float a_val = pa->output->data[0];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) gx[i] = a_val * dout[i];
                    tape_acc_grad(e->parent1, gx, out_len);
                    free(gx);
                }
                float ga = 0;
                for (int i = 0; i < out_len; i++) ga += dout[i] * px->output->data[i];
                float ga_buf[1] = { ga };
                tape_acc_grad(e->parent2, ga_buf, 1);
            }
            break;
        }

        case NT_OP_SOFTMAX: {
            if (e->parent1 >= 0) {
                float dot_dy = 0;
                for (int i = 0; i < out_len; i++)
                    dot_dy += dout[i] * e->output->data[i];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++)
                        gx[i] = e->output->data[i] * (dout[i] - dot_dy);
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_RMSNORM: {
            // y = (x / rms) * gamma (if gamma provided)
            // parent1 = x, parent2 = gamma (-1 if none)
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                /* GPU/CPU mirror discipline (4th instance of this bug class
                 * after CE 3d46007 + MUL/SILU 8ab5062): backward below reads
                 * px->output->data and gamma_data on CPU side. In GPU mode
                 * the mirror is stale → garbage gx → NaN explosion. Verified
                 * 2026-05-14 on nanollama-notorch SFT: 27 RMSNorms per
                 * forward exploded at step ~40, lr=1e-4 (same shape as
                 * Resonance pre-fix lr=1e-4 step 60 explosion). */
                nt_tensor_sync_cpu(px->output);
                int n = out_len;
                float ss = 0;
                for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
                float rms = sqrtf(ss / n + 1e-6f);
                float rms3 = rms * rms * rms;

                // If gamma exists, dout_eff = dout * gamma for x-gradient
                float* dout_eff = dout;
                float* gamma_data = NULL;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                if (has_gamma) {
                    nt_tape_entry* pg = &g_tape.entries[e->parent2];
                    nt_tensor_sync_cpu(pg->output);
                    gamma_data = pg->output->data;
                    dout_eff = (float*)calloc(n, sizeof(float));
                    if (dout_eff) {
                        for (int i = 0; i < n; i++)
                            dout_eff[i] = dout[i] * gamma_data[i % pg->output->len];
                    } else {
                        dout_eff = dout;
                        has_gamma = 0;
                    }
                }

                float sum_dout_x = 0;
                for (int i = 0; i < n; i++)
                    sum_dout_x += dout_eff[i] * px->output->data[i];
                float* gx = (float*)calloc(n, sizeof(float));
                if (gx) {
                    for (int i = 0; i < n; i++)
                        gx[i] = (dout_eff[i] / rms) - (px->output->data[i] * sum_dout_x / (n * rms3));
                    tape_acc_grad(e->parent1, gx, n);
                }
                free(gx);

                // Gamma gradient: d_gamma[i] = dout[i] * (x[i] / rms)
                if (has_gamma && e->parent2 >= 0) {
                    nt_tape_entry* pg = &g_tape.entries[e->parent2];
                    float* gg = (float*)calloc(pg->output->len, sizeof(float));
                    if (gg) {
                        for (int i = 0; i < n; i++)
                            gg[i % pg->output->len] += dout[i] * (px->output->data[i] / rms);
                        tape_acc_grad(e->parent2, gg, pg->output->len);
                    }
                    free(gg);
                }

                if (has_gamma && dout_eff != dout) free(dout_eff);
            }
            break;
        }

        case NT_OP_CROSS_ENT: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                int n = pl->output->len;
                int target = (int)e->aux;
                float mx = pl->output->data[0];
                for (int i = 1; i < n; i++)
                    if (pl->output->data[i] > mx) mx = pl->output->data[i];
                float* sm = (float*)calloc(n, sizeof(float));
                if (sm) {
                    float sum = 0;
                    for (int i = 0; i < n; i++) {
                        sm[i] = expf(pl->output->data[i] - mx);
                        sum += sm[i];
                    }
                    for (int i = 0; i < n; i++) sm[i] /= sum;
                    if (target >= 0 && target < n) sm[target] -= 1.0f;
                    for (int i = 0; i < n; i++) sm[i] *= dout[0];
                    tape_acc_grad(e->parent1, sm, n);
                }
                free(sm);
            }
            break;
        }

        case NT_OP_EMB_LOOKUP: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                int token_id = (int)e->aux;
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : out_len;
                int rows = pw->output->len / cols;
                if (cols > 0 && token_id >= 0 && token_id < rows) {
                    float* gw = (float*)calloc(pw->output->len, sizeof(float));
                    if (gw) {
                        for (int i = 0; i < cols && i < out_len; i++)
                            gw[token_id * cols + i] = dout[i];
                        tape_acc_grad(e->parent1, gw, pw->output->len);
                    }
                    free(gw);
                }
            }
            break;
        }

        case NT_OP_SEQ_EMBED: {
            if (e->parent1 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwte = &g_tape.entries[e->parent1];
                nt_tape_entry* ptok = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int wte_rows = pwte->output->ndim >= 2 ? pwte->output->shape[0] : pwte->output->len / D;
#ifdef USE_CUDA
                int seqemb_done_gpu = 0;
                /* GPU bw — only when no WPE branch (parent2 < 0). WPE handled CPU. */
                if (g_use_gpu && e->parent2 < 0) {
                    float* d_dwte = gpu_scratch(3, pwte->output->len);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_tok  = nt_tensor_ensure_gpu(ptok->output);
                    if (d_dwte && d_dout && d_tok) {
                        gpu_zero(d_dwte, pwte->output->len);
                        gpu_seq_embedding_backward(d_dwte, d_dout, d_tok, T, D, wte_rows);
                        tape_acc_grad_gpu(e->parent1, d_dwte, pwte->output->len);
                        seqemb_done_gpu = 1;
                    }
                }
                if (seqemb_done_gpu) break;
                nt_tensor_ensure_cpu(ptok->output);
#endif
                float* dwte = (float*)calloc(pwte->output->len, sizeof(float));
                if (dwte) {
                    for (int t = 0; t < T; t++) {
                        int tok = (int)ptok->output->data[t];
                        if (tok < 0) tok = 0;
                        if (tok >= wte_rows) tok = wte_rows - 1;
                        for (int d = 0; d < D; d++)
                            dwte[tok * D + d] += dout[t * D + d];
                    }
                    tape_acc_grad(e->parent1, dwte, pwte->output->len);
                }
                free(dwte);
                /* Position embedding gradients (if present) */
                if (e->parent2 >= 0) {
                    nt_tape_entry* pwpe = &g_tape.entries[e->parent2];
                    float* dwpe = (float*)calloc(pwpe->output->len, sizeof(float));
                    if (dwpe) {
                        int wpe_rows = pwpe->output->ndim >= 2 ? pwpe->output->shape[0] : pwpe->output->len / D;
                        for (int t = 0; t < T; t++) {
                            int pos = t < wpe_rows ? t : wpe_rows - 1;
                            for (int d = 0; d < D; d++)
                                dwpe[pos * D + d] += dout[t * D + d];
                        }
                        tape_acc_grad(e->parent2, dwpe, pwpe->output->len);
                    }
                    free(dwpe);
                }
            }
            break;
        }

        case NT_OP_SEQ_MATVEC: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int out_d = pw->output->shape[0];
                int in_d = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / out_d;
                int w_frozen = pw->frozen;     // skip dw if W is frozen (LoRA on frozen base)
                int x_frozen = px->frozen;     // also skip dx if X chain is frozen (rare)
                float* dw = NULL;
                float* dx = NULL;
                int bw_done_gpu = 0;
#ifdef USE_CUDA
                /* GPU backward path: stays GPU-resident.
                 * dW grad accumulates directly on pw->grad->d_data via cuBLAS
                 * axpy. Same for dX → px->grad->d_data. Saves the full
                 * download → calloc → CPU-add chain of v1. */
                if (g_use_gpu && (!w_frozen || !x_frozen)) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_W = nt_tensor_ensure_gpu(pw->output);
                    float* d_X = nt_tensor_ensure_gpu(px->output);
                    float* d_dx = !x_frozen ? gpu_scratch(3, px->output->len) : NULL;
                    float* d_dw = !w_frozen ? gpu_scratch(4, pw->output->len) : NULL;
                    if (d_dout && d_W && d_X &&
                        ((x_frozen) || d_dx) && ((w_frozen) || d_dw)) {
                        if (!x_frozen)
                            gpu_sgemm_nn(T, in_d, out_d, d_dout, d_W, d_dx);
                        if (!w_frozen)
                            gpu_sgemm_tn(out_d, in_d, T, d_dout, d_X, d_dw);
                        if (!w_frozen)
                            tape_acc_grad_gpu(e->parent1, d_dw, pw->output->len);
                        if (!x_frozen)
                            tape_acc_grad_gpu(e->parent2, d_dx, px->output->len);
                        bw_done_gpu = 1;
                    }
                }
                if (!bw_done_gpu) {
                    dw = w_frozen ? NULL : (float*)calloc(pw->output->len, sizeof(float));
                    dx = x_frozen ? NULL : (float*)calloc(px->output->len, sizeof(float));
                }
#else
                dw = w_frozen ? NULL : (float*)calloc(pw->output->len, sizeof(float));
                dx = x_frozen ? NULL : (float*)calloc(px->output->len, sizeof(float));
#endif
                if (!bw_done_gpu && ((dw || w_frozen) && (dx || x_frozen))) {
                    float* Wd = pw->output->data;
                    float* Xd = px->output->data;
#ifdef USE_BLAS
                    if (!x_frozen) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    T, in_d, out_d,
                                    1.0f, dout, out_d, Wd, in_d,
                                    0.0f, dx, in_d);
                    }
                    if (!w_frozen) {
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                    out_d, in_d, T,
                                    1.0f, dout, out_d, Xd, in_d,
                                    0.0f, dw, in_d);
                    }
#else
                    if (!x_frozen) {
                        for (int t = 0; t < T; t++) {
                            float* dout_t = dout + t * out_d;
                            for (int j = 0; j < in_d; j++)
                                for (int i = 0; i < out_d; i++)
                                    dx[t * in_d + j] += Wd[i * in_d + j] * dout_t[i];
                        }
                    }
                    if (!w_frozen) {
                        for (int t = 0; t < T; t++) {
                            float* dout_t = dout + t * out_d;
                            float* x_t = Xd + t * in_d;
                            for (int i = 0; i < out_d; i++)
                                for (int j = 0; j < in_d; j++)
                                    dw[i * in_d + j] += dout_t[i] * x_t[j];
                        }
                    }
#endif
                }
                if (!bw_done_gpu && ((dw || w_frozen) && (dx || x_frozen))) {
                    if (!w_frozen) tape_acc_grad(e->parent1, dw, pw->output->len);
                    if (!x_frozen) tape_acc_grad(e->parent2, dx, px->output->len);
                }
                if (dw) free(dw);
                if (dx) free(dx);
            }
            break;
        }

        case NT_OP_SEQ_RMSNORM: {
            // y[t] = (x[t] / rms[t]) * gamma (if gamma provided)
            // parent1 = x, parent2 = gamma (-1 if none)
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int srn_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_X    = nt_tensor_ensure_gpu(px->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_gamma = NULL;
                    if (has_gamma) {
                        nt_tape_entry* pg = &g_tape.entries[e->parent2];
                        d_gamma = nt_tensor_ensure_gpu(pg->output);
                    }
                    float* d_gx = gpu_scratch(3, T * D);
                    float* d_gg = has_gamma ? gpu_scratch(4, D) : NULL;
                    if (d_X && d_dout && d_gx && (!has_gamma || (d_gamma && d_gg))) {
                        if (d_gg) gpu_zero(d_gg, D);
                        gpu_seq_rmsnorm_backward(d_gx, d_gg, d_dout, d_X, d_gamma, T, D);
                        tape_acc_grad_gpu(e->parent1, d_gx, T * D);
                        if (has_gamma && d_gg)
                            tape_acc_grad_gpu(e->parent2, d_gg, D);
                        srn_done_gpu = 1;
                    }
                }
#endif
                if (srn_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(px->output);
                if (has_gamma) nt_tensor_ensure_cpu(g_tape.entries[e->parent2].output);
#endif
                float* gamma_data = NULL;
                if (has_gamma) gamma_data = g_tape.entries[e->parent2].output->data;

                float* gx = (float*)calloc((size_t)T * D, sizeof(float));
                float* gg = has_gamma ? (float*)calloc(D, sizeof(float)) : NULL;
                if (gx) {
                    float* Xrn = px->output->data;
                    for (int t = 0; t < T; t++) {
                        float* x_t = Xrn + t * D;
                        float* dout_t = dout + t * D;
                        float ss = 0;
                        for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
                        float rms = sqrtf(ss / D + 1e-6f);
                        float rms3 = rms * rms * rms;

                        // dout_eff = dout * gamma for x-gradient
                        float sum_dx = 0;
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            sum_dx += de * x_t[d];
                        }
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            gx[t * D + d] = (de / rms) - (x_t[d] * sum_dx / (D * rms3));
                        }
                        // gamma gradient: d_gamma[d] += dout[t,d] * (x[t,d] / rms[t])
                        if (gg) {
                            for (int d = 0; d < D; d++)
                                gg[d] += dout_t[d] * (x_t[d] / rms);
                        }
                    }
                    tape_acc_grad(e->parent1, gx, T * D);
                    if (gg && has_gamma)
                        tape_acc_grad(e->parent2, gg, D);
                }
                free(gx);
                free(gg);
            }
            break;
        }

        case NT_OP_CAUSAL_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                float sc = 1.0f / sqrtf((float)D);
                float* dq = (float*)calloc((size_t)T * D, sizeof(float));
                float* dk = (float*)calloc((size_t)T * D, sizeof(float));
                float* dv = (float*)calloc((size_t)T * D, sizeof(float));
                if (dq && dk && dv) {
                    for (int i = 0; i < T; i++) {
                        float* qi = pq->output->data + i * D;
                        float* dout_i = dout + i * D;
                        float* scores = (float*)calloc(i + 1, sizeof(float));
                        float* attn = (float*)calloc(i + 1, sizeof(float));
                        if (!scores || !attn) { free(scores); free(attn); continue; }
                        float mx = -1e30f;
                        for (int j = 0; j <= i; j++) {
                            float* kj = pk->output->data + j * D;
                            float dot = 0;
                            for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                            scores[j] = dot * sc;
                            if (scores[j] > mx) mx = scores[j];
                        }
                        float sm = 0;
                        for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                        if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                        float* d_attn = (float*)calloc(i + 1, sizeof(float));
                        if (d_attn) {
                            for (int j = 0; j <= i; j++) {
                                float* vj = pv->output->data + j * D;
                                for (int d = 0; d < D; d++) d_attn[j] += dout_i[d] * vj[d];
                            }
                            for (int j = 0; j <= i; j++) {
                                float* dvj = dv + j * D;
                                for (int d = 0; d < D; d++) dvj[d] += attn[j] * dout_i[d];
                            }
                            float dot_da = 0;
                            for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                            for (int j = 0; j <= i; j++) {
                                float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                float* kj = pk->output->data + j * D;
                                for (int d = 0; d < D; d++) {
                                    dq[i * D + d] += ds * kj[d];
                                    dk[j * D + d] += ds * qi[d];
                                }
                            }
                        }
                        free(scores); free(attn); free(d_attn);
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_MH_CAUSAL_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int head_dim = (int)e->aux2;
                int D = e->output->len / T;
                int n_heads = D / head_dim;
                float sc = 1.0f / sqrtf((float)head_dim);
                float* dq = (float*)calloc((size_t)T * D, sizeof(float));
                float* dk = (float*)calloc((size_t)T * D, sizeof(float));
                float* dv = (float*)calloc((size_t)T * D, sizeof(float));
                int mh_done_gpu = 0;
#ifdef USE_CUDA
                /* GPU backward: kernel needs softmaxed scores. Forward did not
                 * persist them, so re-run forward into scratch first.
                 * Slot map (GPU_SCRATCH_SLOTS=16):
                 *   0 silu, 1 mh-attn scores, 2 cross_ent losses,
                 *   3,4 seq_matvec_bw d_dx/d_dw,
                 *   5,6 mh_bw scratch_TT/scratch_TT2, 7 mh recompute out,
                 *   8,9,10 mh_bw d_dQ/d_dK/d_dV.
                 *
                 * Diagnostic: env NT_DISABLE_MH_GPU=1 forces CPU fallback,
                 * for isolating nanollama-on-GPU NaN hypothesis 2026-05-14. */
                int mh_gpu_disabled = getenv("NT_DISABLE_MH_GPU") != NULL;
                if (g_use_gpu && !mh_gpu_disabled && dq && dk && dv) {
                    float* d_Q = nt_tensor_ensure_gpu(pq->output);
                    float* d_K = nt_tensor_ensure_gpu(pk->output);
                    float* d_V = nt_tensor_ensure_gpu(pv->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_scores = gpu_scratch(1, n_heads * T * T);
                    float* d_scratch_TT  = gpu_scratch(5, n_heads * T * T);
                    float* d_scratch_TT2 = gpu_scratch(6, n_heads * T * T);
                    float* d_out_tmp = gpu_scratch(7, T * D);
                    float* d_dQ_buf  = gpu_scratch(8, T * D);
                    float* d_dK_buf  = gpu_scratch(9, T * D);
                    float* d_dV_buf  = gpu_scratch(10, T * D);
                    if (d_Q && d_K && d_V && d_dout && d_scores && d_scratch_TT &&
                        d_scratch_TT2 && d_out_tmp && d_dQ_buf && d_dK_buf && d_dV_buf) {
                        /* Recompute softmaxed scores (kernel writes them). */
                        gpu_multi_head_attention(d_Q, d_K, d_V, d_out_tmp, d_scores, T, D, n_heads);
                        gpu_multi_head_attention_backward(d_Q, d_K, d_V, d_scores, d_dout,
                                                          d_dQ_buf, d_dK_buf, d_dV_buf,
                                                          d_scratch_TT, d_scratch_TT2,
                                                          T, D, n_heads);
                        /* GPU-resident grad accumulation — no D2H. */
                        tape_acc_grad_gpu(e->parent1, d_dQ_buf, T * D);
                        tape_acc_grad_gpu(e->parent2, d_dK_buf, T * D);
                        tape_acc_grad_gpu(e->parent3, d_dV_buf, T * D);
                        mh_done_gpu = 1;
                    }
                }
#endif
                if (mh_done_gpu) {
                    free(dq); free(dk); free(dv);
                    break;
                }
                /* GPU/CPU mirror discipline (6th instance): CPU fallback below
                 * reads pq/pk/pv ->output->data to compute scores, d_attn,
                 * ds, dq, dk. Without sync, GPU-resident mirrors are stale
                 * (calloc-zero) → ds = attn * (d_attn - dot_da) * sc = 0 →
                 * dq, dk accumulate zero → wq, wk LoRA targets receive no
                 * grad (verified 2026-05-14 with NT_DISABLE_MH_GPU=1).
                 * dv survives because it uses dout, not q/k. */
                nt_tensor_sync_cpu(pq->output);
                nt_tensor_sync_cpu(pk->output);
                nt_tensor_sync_cpu(pv->output);
                if (dq && dk && dv) {
                    for (int h = 0; h < n_heads; h++) {
                        int ho = h * head_dim;
                        for (int i = 0; i < T; i++) {
                            float* qi = pq->output->data + i * D + ho;
                            float* dout_i = dout + i * D + ho;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float* kj = pk->output->data + j * D + ho;
                                float dot = 0;
                                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * sc;
                                if (scores[j] > mx) mx = scores[j];
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                    float* kj = pk->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) {
                                        dq[i * D + ho + d] += ds * kj[d];
                                        dk[j * D + ho + d] += ds * qi[d];
                                    }
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_GQA_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int head_dim = (int)e->aux2;
                int n_heads = (int)e->aux3;
                int n_kv_heads = (int)e->aux4;
                int Q_D = n_heads * head_dim;
                int KV_D = n_kv_heads * head_dim;
                int gqa_ratio = n_heads / n_kv_heads;
                float sc = 1.0f / sqrtf((float)head_dim);
                float* dq = (float*)calloc((size_t)T * Q_D, sizeof(float));
                float* dk = (float*)calloc((size_t)T * KV_D, sizeof(float));
                float* dv = (float*)calloc((size_t)T * KV_D, sizeof(float));
                if (dq && dk && dv) {
                    for (int h = 0; h < n_heads; h++) {
                        int kv_h = h / gqa_ratio;
                        int q_off = h * head_dim;
                        int kv_off = kv_h * head_dim;
                        for (int i = 0; i < T; i++) {
                            float* qi = pq->output->data + i * Q_D + q_off;
                            float* dout_i = dout + i * Q_D + q_off;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float* kj = pk->output->data + j * KV_D + kv_off;
                                float dot = 0;
                                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * sc;
                                if (scores[j] > mx) mx = scores[j];
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                    float* kj = pk->output->data + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) {
                                        dq[i * Q_D + q_off + d] += ds * kj[d];
                                        dk[j * KV_D + kv_off + d] += ds * qi[d];
                                    }
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dq, T * Q_D);
                    tape_acc_grad(e->parent2, dk, T * KV_D);
                    tape_acc_grad(e->parent3, dv, T * KV_D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_RRPRAM_LR: {
            /* Low-rank RRPRAM backward.
             * Forward: u = X @ Wr_a[h]; scores = u @ Wr_b[h]; attn = softmax(causal); out = Σ attn·V.
             * dout flows back through:
             *   d_attn  = dout · V               (per i, h, j)
             *   d_v     = attn · dout            (per j, h)
             *   d_score = softmax_bwd(d_attn, attn)
             *   d_u     = d_score @ Wr_b[h]^T
             *   d_Wr_b  = u^T @ d_score (causal-masked outer-product)
             *   d_x     = Σ_h d_u · Wr_a[h]^T
             *   d_Wr_a  = Σ_h x^T @ d_u
             */
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwr = &g_tape.entries[e->parent1];
                nt_tape_entry* px  = &g_tape.entries[e->parent2];
                nt_tape_entry* pv  = &g_tape.entries[e->parent3];
                int T = (int)e->aux; int n_embd = (int)e->aux2;
                int nr = (int)e->aux3; int hd = (int)e->aux4;
                int out_dim = nr * hd;
                int T_r = T;   /* same assumption as forward */
                long combined_len = pwr->output->len;
                int rank = (int)(combined_len / ((long)nr * (n_embd + T_r)));
                long wra_total = (long)nr * n_embd * rank;

                float* dwr = (float*)calloc(combined_len, sizeof(float));
                float* dx  = (float*)calloc((long)T * n_embd, sizeof(float));
                float* dv  = (float*)calloc((long)T * out_dim, sizeof(float));

#ifdef USE_CUDA
                int rrlr_bw_gpu = 0;
                if (g_use_gpu && dwr && dx && dv) {
                    /* Recompute U and scores on GPU (forward did not persist
                     * across tape boundary cleanly — this is cheap: H·T·R + H·T·T floats). */
                    float* d_X  = nt_tensor_ensure_gpu(px->output);
                    float* d_Wr = nt_tensor_ensure_gpu(pwr->output);
                    float* d_V  = nt_tensor_ensure_gpu(pv->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_U      = gpu_scratch(12, nr * T * rank);
                    float* d_scores = gpu_scratch(1,  nr * T * T);
                    float* d_O_tmp  = gpu_scratch(7,  T * out_dim);
                    float* d_d_attn  = gpu_scratch(13, nr * T * T);
                    float* d_d_score = gpu_scratch(14, nr * T * T);
                    /* All scratch via persistent slots — avoid per-call cudaMalloc. */
                    float* d_dX  = gpu_scratch(15, T * n_embd);
                    /* Slots 11..14 already used by other backward paths above — but
                     * those paths run sequentially per backward pass (different ops),
                     * so slot reuse across distinct op-cases in the same backward
                     * call is safe. d_dWr fits in slot 11 (CE backward path scratch),
                     * d_dV in slot 0 (forward silu, not running here). */
                    float* d_dWr = gpu_scratch(11, combined_len);
                    float* d_dV  = gpu_scratch(0, T * out_dim);
                    if (d_X && d_Wr && d_V && d_dout && d_U && d_scores && d_O_tmp &&
                        d_d_attn && d_d_score && d_dX && d_dWr && d_dV) {
                        /* Recompute forward (writes U and scores). */
                        gpu_rrpram_lr_forward(d_X, d_Wr, d_V, d_O_tmp, d_U, d_scores,
                                              T, n_embd, nr, rank, hd);
                        gpu_rrpram_lr_backward(d_X, d_Wr, d_V, d_U, d_scores, d_dout,
                                               d_dWr, d_dX, d_dV,
                                               d_d_attn, d_d_score,
                                               T, n_embd, nr, rank, hd);
                        /* GPU-resident grad accumulation. */
                        tape_acc_grad_gpu(e->parent1, d_dWr, combined_len);
                        tape_acc_grad_gpu(e->parent2, d_dX,  (long)T * n_embd);
                        tape_acc_grad_gpu(e->parent3, d_dV,  (long)T * out_dim);
                        rrlr_bw_gpu = 1;
                    }
                }
                if (rrlr_bw_gpu) {
                    free(dwr); free(dx); free(dv);
                    break;
                }
#endif

                float* u_buf      = (float*)malloc(rank * sizeof(float));
                float* du_buf     = (float*)malloc(rank * sizeof(float));
                float* scores_buf = (float*)malloc(T_r  * sizeof(float));
                float* attn_buf   = (float*)malloc(T_r  * sizeof(float));
                float* d_attn_buf = (float*)malloc(T_r  * sizeof(float));
                float* d_score_buf= (float*)malloc(T_r  * sizeof(float));

                if (dwr && dx && dv && u_buf && du_buf && scores_buf && attn_buf && d_attn_buf && d_score_buf) {
                    for (int h = 0; h < nr; h++) {
                        long wr_a_base = (long)h * n_embd * rank;
                        long wr_b_base = wra_total + (long)h * rank * T_r;
                        int  v_off     = h * hd;
                        for (int i = 0; i < T; i++) {
                            float* xi = px->output->data + i * n_embd;
                            float* dout_i = dout + i * out_dim + v_off;

                            /* recompute forward: u, scores, attn */
                            for (int r = 0; r < rank; r++) u_buf[r] = 0.0f;
                            for (int d = 0; d < n_embd; d++) {
                                float xd = xi[d];
                                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                                for (int r = 0; r < rank; r++) u_buf[r] += xd * wa_row[r];
                            }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float s = 0.0f;
                                for (int r = 0; r < rank; r++) {
                                    s += u_buf[r] * pwr->output->data[wr_b_base + (long)r * T_r + j];
                                }
                                scores_buf[j] = s;
                                if (s > mx) mx = s;
                            }
                            float sm = 0.0f;
                            for (int j = 0; j <= i; j++) { attn_buf[j] = expf(scores_buf[j] - mx); sm += attn_buf[j]; }
                            if (sm > 0.0f) for (int j = 0; j <= i; j++) attn_buf[j] /= sm;

                            /* d_attn[j] = Σ_d dout_i[d] · v[j, h_off+d]
                             * d_v [j, h_off+d] += attn[j] · dout_i[d] */
                            for (int j = 0; j <= i; j++) d_attn_buf[j] = 0.0f;
                            for (int j = 0; j <= i; j++) {
                                const float* vj = pv->output->data + j * out_dim + v_off;
                                float* dvj      = dv + j * out_dim + v_off;
                                for (int d = 0; d < hd; d++) {
                                    d_attn_buf[j] += dout_i[d] * vj[d];
                                    dvj[d]        += attn_buf[j] * dout_i[d];
                                }
                            }

                            /* softmax backward → d_score */
                            float dot_da = 0.0f;
                            for (int j = 0; j <= i; j++) dot_da += d_attn_buf[j] * attn_buf[j];
                            for (int j = 0; j <= i; j++) d_score_buf[j] = attn_buf[j] * (d_attn_buf[j] - dot_da);

                            /* d_u[r] = Σ_j d_score[j] · Wr_b[h, r, j] (j ≤ i)
                             * d_Wr_b[h, r, j] += d_score[j] · u[r]   (j ≤ i) */
                            for (int r = 0; r < rank; r++) du_buf[r] = 0.0f;
                            for (int j = 0; j <= i; j++) {
                                float ds = d_score_buf[j];
                                for (int r = 0; r < rank; r++) {
                                    du_buf[r] += ds * pwr->output->data[wr_b_base + (long)r * T_r + j];
                                    dwr[wr_b_base + (long)r * T_r + j] += ds * u_buf[r];
                                }
                            }

                            /* d_xi[d] += Σ_r d_u[r] · Wr_a[h, d, r]
                             * d_Wr_a[h, d, r] += d_u[r] · xi[d] */
                            for (int d = 0; d < n_embd; d++) {
                                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                                float* dwa_row     = dwr + wr_a_base + (long)d * rank;
                                float dxd = 0.0f;
                                float xd = xi[d];
                                for (int r = 0; r < rank; r++) {
                                    dxd        += du_buf[r] * wa_row[r];
                                    dwa_row[r] += du_buf[r] * xd;
                                }
                                dx[i * n_embd + d] += dxd;
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, dwr, combined_len);
                    tape_acc_grad(e->parent2, dx,  (long)T * n_embd);
                    tape_acc_grad(e->parent3, dv,  (long)T * out_dim);
                }
                free(dwr); free(dx); free(dv);
                free(u_buf); free(du_buf); free(scores_buf); free(attn_buf); free(d_attn_buf); free(d_score_buf);
            }
            break;
        }

        case NT_OP_RRPRAM_BCAST: {
            /* Broadcast RRPRAM backward (canonical Janus scale included).
             * Forward: mid = Σ_t x·Wr_a (broadcast); raw_s = mid·Wr_b (per layer);
             *          score = raw_s * sc, sc = 1/sqrt(D);
             *          attn[i,:] = softmax_causal(score)[0..i]; out[i] = Σ attn·v.
             * d_v[j,h,d] += Σ_i attn[i,j] · dout[i,h,d]
             * d_attn[i,j] = Σ_d dout[i,h,d] · v[j,h,d]
             * d_score[j] = Σ_i softmax_bwd(attn[i],d_attn[i])[j]   (only j ≤ i)
             * d_raw_s[j] = d_score[j] * sc                          (chain rule through scale)
             * d_mid[r] = Σ_j d_raw_s[j] · Wr_b[h,r,j]
             * d_Wr_b[h,r,j] = mid[r] · d_raw_s[j]
             * d_x[t,e] += Σ_r d_mid[r] · Wr_a[h,e,r]   (broadcast — same dxe added to every t)
             * d_Wr_a[h,e,r] += Σ_t x[t,e] · d_mid[r]
             */
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwr = &g_tape.entries[e->parent1];
                nt_tape_entry* px  = &g_tape.entries[e->parent2];
                nt_tape_entry* pv  = &g_tape.entries[e->parent3];
                int T = (int)e->aux; int n_embd = (int)e->aux2;
                int nr = (int)e->aux3;
                int rank = (int)e->aux4;  /* aux4 = rank, head_dim = E/H */
                int hd = n_embd / nr;
                int out_dim = nr * hd;
                long combined_len = pwr->output->len;
                int ctx_T = (int)(combined_len / ((long)nr * rank) - n_embd);
                long wra_total = (long)nr * n_embd * rank;
                float sc = 1.0f / sqrtf((float)hd);

#ifdef USE_CUDA
                nt_tensor_ensure_cpu(pwr->output);
                nt_tensor_ensure_cpu(px->output);
                nt_tensor_ensure_cpu(pv->output);
                nt_tensor_ensure_cpu(e->grad);
#endif

                float* dwr = (float*)calloc(combined_len, sizeof(float));
                float* dx  = (float*)calloc((long)T * n_embd, sizeof(float));
                float* dv  = (float*)calloc((long)T * out_dim, sizeof(float));

                float* mid_buf       = (float*)malloc(rank * sizeof(float));
                float* d_mid_buf     = (float*)malloc(rank * sizeof(float));
                float* all_scores    = (float*)malloc(T  * sizeof(float));
                float* attn_buf      = (float*)malloc(T  * sizeof(float));
                float* d_attn_buf    = (float*)malloc(T  * sizeof(float));
                float* d_score_global= (float*)calloc(T,   sizeof(float));

                float* dout = e->grad ? e->grad->data : NULL;

                if (dwr && dx && dv && mid_buf && d_mid_buf && all_scores &&
                    attn_buf && d_attn_buf && d_score_global && dout) {
                    for (int h = 0; h < nr; h++) {
                        long wr_a_base = (long)h * n_embd * rank;
                        long wr_b_base = wra_total + (long)h * rank * ctx_T;
                        int  v_off     = h * hd;

                        for (int r = 0; r < rank; r++) mid_buf[r] = 0.0f;
                        for (int t = 0; t < T; t++) {
                            const float* xt = px->output->data + (long)t * n_embd;
                            for (int e2 = 0; e2 < n_embd; e2++) {
                                float xe = xt[e2];
                                const float* wa_row = pwr->output->data + wr_a_base + (long)e2 * rank;
                                for (int r = 0; r < rank; r++) mid_buf[r] += xe * wa_row[r];
                            }
                        }

                        for (int j = 0; j < T; j++) {
                            float s = 0.0f;
                            for (int r = 0; r < rank; r++) {
                                s += mid_buf[r] * pwr->output->data[wr_b_base + (long)r * ctx_T + j];
                            }
                            all_scores[j] = s * sc;
                        }

                        for (int j = 0; j < T; j++) d_score_global[j] = 0.0f;

                        for (int i = 0; i < T; i++) {
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                attn_buf[j] = all_scores[j];
                                if (attn_buf[j] > mx) mx = attn_buf[j];
                            }
                            float sm = 0.0f;
                            for (int j = 0; j <= i; j++) { attn_buf[j] = expf(attn_buf[j] - mx); sm += attn_buf[j]; }
                            if (sm > 0.0f) for (int j = 0; j <= i; j++) attn_buf[j] /= sm;

                            const float* dout_i = dout + (long)i * out_dim + v_off;

                            for (int j = 0; j <= i; j++) d_attn_buf[j] = 0.0f;
                            for (int j = 0; j <= i; j++) {
                                const float* vj = pv->output->data + (long)j * out_dim + v_off;
                                float* dvj      = dv + (long)j * out_dim + v_off;
                                for (int d = 0; d < hd; d++) {
                                    d_attn_buf[j] += dout_i[d] * vj[d];
                                    dvj[d]        += attn_buf[j] * dout_i[d];
                                }
                            }

                            float dot_da = 0.0f;
                            for (int j = 0; j <= i; j++) dot_da += d_attn_buf[j] * attn_buf[j];
                            for (int j = 0; j <= i; j++) {
                                d_score_global[j] += attn_buf[j] * (d_attn_buf[j] - dot_da);
                            }
                        }

                        for (int r = 0; r < rank; r++) d_mid_buf[r] = 0.0f;
                        for (int j = 0; j < T; j++) {
                            /* Chain rule through forward scale: d_raw_s[j] = d_score[j] * sc. */
                            float ds = d_score_global[j] * sc;
                            if (ds == 0.0f) continue;
                            for (int r = 0; r < rank; r++) {
                                d_mid_buf[r] += ds * pwr->output->data[wr_b_base + (long)r * ctx_T + j];
                                dwr[wr_b_base + (long)r * ctx_T + j] += ds * mid_buf[r];
                            }
                        }

                        for (int t = 0; t < T; t++) {
                            const float* xt = px->output->data + (long)t * n_embd;
                            float* dxt = dx + (long)t * n_embd;
                            for (int e2 = 0; e2 < n_embd; e2++) {
                                const float* wa_row = pwr->output->data + wr_a_base + (long)e2 * rank;
                                float* dwa_row     = dwr + wr_a_base + (long)e2 * rank;
                                float dxe = 0.0f;
                                float xe  = xt[e2];
                                for (int r = 0; r < rank; r++) {
                                    dxe         += d_mid_buf[r] * wa_row[r];
                                    dwa_row[r]  += d_mid_buf[r] * xe;
                                }
                                dxt[e2] += dxe;
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, dwr, combined_len);
                    tape_acc_grad(e->parent2, dx,  (long)T * n_embd);
                    tape_acc_grad(e->parent3, dv,  (long)T * out_dim);
                }
                free(dwr); free(dx); free(dv);
                free(mid_buf); free(d_mid_buf); free(all_scores);
                free(attn_buf); free(d_attn_buf); free(d_score_global);
            }
            break;
        }

        case NT_OP_RRPRAM_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwr = &g_tape.entries[e->parent1];
                nt_tape_entry* px  = &g_tape.entries[e->parent2];
                nt_tape_entry* pv  = &g_tape.entries[e->parent3];
                int T = (int)e->aux; int n_embd = (int)e->aux2;
                int nr = (int)e->aux3; int hd = (int)e->aux4;
                int out_dim = nr * hd;
                int ctx = pwr->output->len / (nr * n_embd);
                float* dwr = (float*)calloc(pwr->output->len, sizeof(float));
                float* dx  = (float*)calloc((size_t)T * n_embd, sizeof(float));
                float* dv  = (float*)calloc((size_t)T * out_dim, sizeof(float));
                if (dwr && dx && dv) {
                    for (int h = 0; h < nr; h++) {
                        int wr_base = h * n_embd * ctx; int v_off = h * hd;
                        for (int i = 0; i < T; i++) {
                            float* xi = px->output->data + i * n_embd;
                            float* dout_i = dout + i * out_dim + v_off;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float dot = 0;
                                for (int d = 0; d < n_embd; d++)
                                    dot += xi[d] * pwr->output->data[wr_base + d * ctx + j];
                                scores[j] = dot; if (dot > mx) mx = dot;
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * out_dim + v_off;
                                    for (int d = 0; d < hd; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * out_dim + v_off;
                                    for (int d = 0; d < hd; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da);
                                    for (int d = 0; d < n_embd; d++)
                                        dx[i * n_embd + d] += ds * pwr->output->data[wr_base + d * ctx + j];
                                    for (int d = 0; d < n_embd; d++)
                                        dwr[wr_base + d * ctx + j] += ds * xi[d];
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dwr, pwr->output->len);
                    tape_acc_grad(e->parent2, dx, T * n_embd);
                    tape_acc_grad(e->parent3, dv, T * out_dim);
                }
                free(dwr); free(dx); free(dv);
            }
            break;
        }

        case NT_OP_CONCAT: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pa = &g_tape.entries[e->parent1];
                nt_tape_entry* pb = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int Da = pa->output->len / T; int Db = pb->output->len / T; int Dc = Da + Db;
                float* da = (float*)calloc((size_t)T * Da, sizeof(float));
                float* db = (float*)calloc((size_t)T * Db, sizeof(float));
                if (da && db) {
                    for (int t = 0; t < T; t++) {
                        for (int d = 0; d < Da; d++) da[t * Da + d] = dout[t * Dc + d];
                        for (int d = 0; d < Db; d++) db[t * Db + d] = dout[t * Dc + Da + d];
                    }
                    tape_acc_grad(e->parent1, da, T * Da);
                    tape_acc_grad(e->parent2, db, T * Db);
                }
                free(da); free(db);
            }
            break;
        }

        case NT_OP_SEQ_MATVEC_T: {
            /* Y[t] = W^T @ X[t]. W[W_rows, W_cols], X[t] has W_rows elems, Y[t] has W_cols elems.
             * dX[t][i] = sum_j dout[t][j] * W[i][j]  → dX[t] = W @ dout[t]
             * dW[i][j] = sum_t dout[t][j] * X[t][i]  → dW = X^T @ dout
             */
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int W_rows = pw->output->shape[0];
                int W_cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / W_rows;
                float* dw = (float*)calloc(pw->output->len, sizeof(float));
                float* dx = (float*)calloc(px->output->len, sizeof(float));
                int bw_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu && dw && dx) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_W = nt_tensor_ensure_gpu(pw->output);
                    float* d_X = nt_tensor_ensure_gpu(px->output);
                    float* d_dx = gpu_scratch(3, px->output->len);
                    float* d_dw = gpu_scratch(4, pw->output->len);
                    if (d_dout && d_W && d_X && d_dx && d_dw) {
                        /* dX[T, W_rows] = dout[T, W_cols] @ W^T[W_cols, W_rows] — NT gemm
                         *   M=T, N=W_rows, K=W_cols, A=dout, B=W */
                        gpu_sgemm_nt(T, W_rows, W_cols, d_dout, d_W, d_dx);
                        /* dW[W_rows, W_cols] = X^T[W_rows, T] @ dout[T, W_cols] — TN gemm
                         *   M=W_rows, N=W_cols, K=T, A=X(T,W_rows), B=dout(T,W_cols) */
                        gpu_sgemm_tn(W_rows, W_cols, T, d_X, d_dout, d_dw);
                        gpu_download(dx, d_dx, px->output->len);
                        gpu_download(dw, d_dw, pw->output->len);
                        bw_done_gpu = 1;
                    }
                }
#endif
                if (!bw_done_gpu && dw && dx) {
                    float* Wd = pw->output->data;
                    float* Xd = px->output->data;
#ifdef USE_BLAS
                    /* dX[T, W_rows] = dout[T, W_cols] @ W^T[W_cols, W_rows] */
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                T, W_rows, W_cols,
                                1.0f, dout, W_cols, Wd, W_cols,
                                0.0f, dx, W_rows);
                    /* dW[W_rows, W_cols] = X^T[W_rows, T] @ dout[T, W_cols] */
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                W_rows, W_cols, T,
                                1.0f, Xd, W_rows, dout, W_cols,
                                0.0f, dw, W_cols);
#else
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * W_cols;
                        for (int i = 0; i < W_rows; i++)
                            for (int j = 0; j < W_cols; j++)
                                dx[t * W_rows + i] += Wd[i * W_cols + j] * dout_t[j];
                    }
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * W_cols;
                        float* x_t = Xd + t * W_rows;
                        for (int i = 0; i < W_rows; i++)
                            for (int j = 0; j < W_cols; j++)
                                dw[i * W_cols + j] += x_t[i] * dout_t[j];
                    }
#endif
                }
                if (dw && dx) {
                    tape_acc_grad(e->parent1, dw, pw->output->len);
                    tape_acc_grad(e->parent2, dx, px->output->len);
                }
                free(dw); free(dx);
            }
            break;
        }

        case NT_OP_SEQ_CROSSENT: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                nt_tape_entry* pt = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int V = (int)e->aux2;
                int ce_done_gpu = 0;
#ifdef USE_CUDA
                /* Pure GPU backward: write straight into pl->grad GPU buffer.
                 * Kernel produces (softmax - one_hot) / T. The loss tape entry
                 * carries dout[0] via e->grad which we read on CPU (single
                 * scalar — cheap). Bake (dout[0] / T) into per-T kernel scale. */
                if (g_use_gpu) {
                    float* d_logits  = nt_tensor_ensure_gpu(pl->output);
                    float* d_targets = nt_tensor_ensure_gpu(pt->output);
                    float* d_grad_logits = gpu_scratch(11, T * V);
                    if (d_logits && d_targets && d_grad_logits) {
                        gpu_cross_entropy_backward(d_grad_logits, d_logits, d_targets, T, V);
                        if (dout[0] != 1.0f) {
                            extern void gpu_axpy(float*, const float*, int, float);
                            /* Multiply in-place: scratch *= dout[0]; do it via
                             * a brief CPU read of dout[0] (already on CPU) and
                             * a kernel-side scale. Reuse gpu_scale for in-place. */
                            gpu_scale(d_grad_logits, d_grad_logits, T * V, dout[0]);
                        }
                        tape_acc_grad_gpu(e->parent1, d_grad_logits, T * V);
                        ce_done_gpu = 1;
                    }
                }
#endif
                float* dl = ce_done_gpu ? NULL : (float*)calloc((size_t)T * V, sizeof(float));
                if (!ce_done_gpu && dl && pt) {
                    for (int t = 0; t < T; t++) {
                        float* logits_t = pl->output->data + t * V;
                        int target = (int)pt->output->data[t];
                        if (target < 0 || target >= V) target = 0;
                        float mx = logits_t[0];
                        for (int j = 1; j < V; j++)
                            if (logits_t[j] > mx) mx = logits_t[j];
                        float sum = 0;
                        for (int j = 0; j < V; j++) {
                            dl[t * V + j] = expf(logits_t[j] - mx);
                            sum += dl[t * V + j];
                        }
                        for (int j = 0; j < V; j++) dl[t * V + j] /= sum;
                        dl[t * V + target] -= 1.0f;
                        float s = dout[0] / T;
                        for (int j = 0; j < V; j++) dl[t * V + j] *= s;
                    }
                }
                if (dl) tape_acc_grad(e->parent1, dl, T * V);
                free(dl);
            }
            break;
        }

        case NT_OP_SEQ_CROSSENT_MASKED: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                nt_tape_entry* pt = &g_tape.entries[e->parent2];
                nt_tape_entry* pm = &g_tape.entries[e->parent3];
                /* GPU/CPU mirror discipline (5th instance — sibling of 3d46007
                 * which fixed non-masked NT_OP_CROSS_ENT). Masked variant was
                 * never synced. In GPU mode pl->output->data (logits, line 1697)
                 * is stale (CPU mirror untouched since the GPU output linear).
                 * dl computed via softmax(stale_logits) - target produces a
                 * gradient pointing at the wrong direction → feeds garbage up
                 * 13 layers → Chuck oscillates → NaN at step 40-220 regardless
                 * of LoRA scale. Verified 2026-05-14 nanollama-notorch SFT.
                 * Matches Olego «не из-за оптимайзера» and Intel POST_SFT note
                 * that lr=1e-5/3e-5 plateau is lr-independent (= zero/garbage
                 * grad somewhere upstream). */
                nt_tensor_sync_cpu(pl->output);
                nt_tensor_sync_cpu(pm->output);
                nt_tensor_sync_cpu(pt->output);
                int T = (int)e->aux;
                int V = (int)e->aux2;
                float n_active = 0;
                for (int t = 0; t < T; t++) n_active += pm->output->data[t];
                if (n_active <= 0) break;
                float* dl = (float*)calloc((size_t)T * V, sizeof(float));
                if (dl) {
                    for (int t = 0; t < T; t++) {
                        float m = pm->output->data[t];
                        if (m == 0.0f) continue;   // dl row stays zero
                        float* logits_t = pl->output->data + t * V;
                        int target = (int)pt->output->data[t];
                        if (target < 0 || target >= V) target = 0;
                        float mx = logits_t[0];
                        for (int j = 1; j < V; j++)
                            if (logits_t[j] > mx) mx = logits_t[j];
                        float sum = 0;
                        for (int j = 0; j < V; j++) {
                            dl[t * V + j] = expf(logits_t[j] - mx);
                            sum += dl[t * V + j];
                        }
                        for (int j = 0; j < V; j++) dl[t * V + j] /= sum;
                        dl[t * V + target] -= 1.0f;
                        float s = m * dout[0] / n_active;
                        for (int j = 0; j < V; j++) dl[t * V + j] *= s;
                    }
                    tape_acc_grad(e->parent1, dl, T * V);
                }
                free(dl);
            }
            break;
        }

        case NT_OP_GEGLU: {
            // y = GELU(x @ W1) * (x @ W2)
            // Stored: parent1 = x, parent2 = W1, parent3 = W2
            // aux = T*D_out (output total), aux2 encodes T and D_in
            // For backward: we need the intermediate values, recompute from parents
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                nt_tape_entry* pw1 = &g_tape.entries[e->parent2];
                nt_tape_entry* pw2 = &g_tape.entries[e->parent3];
                int D_out = pw1->output->shape[0];
                int D_in = pw1->output->ndim >= 2 ? pw1->output->shape[1] : pw1->output->len / D_out;
                int T = px->output->len / D_in;

                // Recompute gate and value
                float* gate = (float*)calloc((size_t)T * D_out, sizeof(float));
                float* val = (float*)calloc((size_t)T * D_out, sizeof(float));
                float* gelu_gate = (float*)calloc((size_t)T * D_out, sizeof(float));
                float* dx = (float*)calloc(px->output->len, sizeof(float));
                float* dw1 = (float*)calloc(pw1->output->len, sizeof(float));
                float* dw2 = (float*)calloc(pw2->output->len, sizeof(float));

                if (gate && val && gelu_gate && dx && dw1 && dw2) {
                    // Forward recompute: gate = x @ W1^T, val = x @ W2^T
                    for (int t = 0; t < T; t++) {
                        float* x_t = px->output->data + t * D_in;
                        for (int i = 0; i < D_out; i++) {
                            float g = 0, v = 0;
                            for (int j = 0; j < D_in; j++) {
                                g += pw1->output->data[i * D_in + j] * x_t[j];
                                v += pw2->output->data[i * D_in + j] * x_t[j];
                            }
                            gate[t * D_out + i] = g;
                            val[t * D_out + i] = v;
                            // GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
                            float x3 = g * g * g;
                            float inner = 0.7978845608f * (g + 0.044715f * x3);
                            float th = tanhf(inner);
                            gelu_gate[t * D_out + i] = 0.5f * g * (1.0f + th);
                        }
                    }

                    // Backward: dy = dout, y = gelu(gate) * val
                    // d_val = dout * gelu(gate)
                    // d_gelu_gate = dout * val
                    // d_gate = d_gelu_gate * gelu'(gate)
                    for (int t = 0; t < T; t++) {
                        for (int i = 0; i < D_out; i++) {
                            int ti = t * D_out + i;
                            float d_val = dout[ti] * gelu_gate[ti];
                            float g = gate[ti];
                            float x3 = g * g * g;
                            float inner = 0.7978845608f * (g + 0.044715f * x3);
                            float th = tanhf(inner);
                            float gelu_grad = 0.5f * (1.0f + th) +
                                0.5f * g * (1.0f - th * th) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * g * g);
                            float d_gate = dout[ti] * val[ti] * gelu_grad;

                            // Accumulate into weight and input grads
                            float* x_t = px->output->data + t * D_in;
                            for (int j = 0; j < D_in; j++) {
                                dw1[i * D_in + j] += d_gate * x_t[j];
                                dw2[i * D_in + j] += d_val * x_t[j];
                                dx[t * D_in + j] += d_gate * pw1->output->data[i * D_in + j];
                                dx[t * D_in + j] += d_val * pw2->output->data[i * D_in + j];
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, dx, px->output->len);
                    tape_acc_grad(e->parent2, dw1, pw1->output->len);
                    tape_acc_grad(e->parent3, dw2, pw2->output->len);
                }
                free(gate); free(val); free(gelu_gate);
                free(dx); free(dw1); free(dw2);
            }
            break;
        }

        case NT_OP_DROPOUT: {
            // y = x * mask (mask encoded in output: 0 = dropped, scale = kept)
            if (e->parent1 >= 0) {
                float p = e->aux;
                float scale = (p > 0.0f && p < 1.0f) ? 1.0f / (1.0f - p) : 1.0f;
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        // If output was zero, the mask dropped it
                        gx[i] = (e->output->data[i] != 0.0f) ? dout[i] * scale : 0.0f;
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_GELU: {
            // y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float x = px->output->data[i];
                        float x3 = x * x * x;
                        float inner = 0.7978845608f * (x + 0.044715f * x3);
                        float th = tanhf(inner);
                        float gelu_grad = 0.5f * (1.0f + th) +
                            0.5f * x * (1.0f - th * th) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                        gx[i] = dout[i] * gelu_grad;
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_LAYERNORM: {
            // y = gamma * (x - mean) / sqrt(var + eps) + beta
            // parent1 = x, parent2 = gamma, parent3 = beta
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int n = out_len;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int has_beta = (e->parent3 >= 0 && e->parent3 < g_tape.count);
                float* gamma_data = has_gamma ? g_tape.entries[e->parent2].output->data : NULL;

                // Recompute stats
                float mean = 0;
                for (int i = 0; i < n; i++) mean += px->output->data[i];
                mean /= n;
                float var = 0;
                for (int i = 0; i < n; i++) { float d = px->output->data[i] - mean; var += d * d; }
                var /= n;
                float inv_std = 1.0f / sqrtf(var + 1e-5f);

                // dout_eff = dout * gamma for x-gradient
                float* dout_eff = (float*)calloc(n, sizeof(float));
                if (dout_eff) {
                    for (int i = 0; i < n; i++)
                        dout_eff[i] = has_gamma ? dout[i] * gamma_data[i] : dout[i];

                    // x gradient (standard layernorm backward)
                    float sum_dout = 0, sum_dout_xhat = 0;
                    for (int i = 0; i < n; i++) {
                        float xhat = (px->output->data[i] - mean) * inv_std;
                        sum_dout += dout_eff[i];
                        sum_dout_xhat += dout_eff[i] * xhat;
                    }
                    float* gx = (float*)calloc(n, sizeof(float));
                    if (gx) {
                        for (int i = 0; i < n; i++) {
                            float xhat = (px->output->data[i] - mean) * inv_std;
                            gx[i] = inv_std * (dout_eff[i] - sum_dout / n - xhat * sum_dout_xhat / n);
                        }
                        tape_acc_grad(e->parent1, gx, n);
                    }
                    free(gx);
                    free(dout_eff);
                }

                // Gamma gradient: d_gamma[i] = dout[i] * xhat[i]
                if (has_gamma) {
                    int gn = g_tape.entries[e->parent2].output->len;
                    float* gg = (float*)calloc(gn, sizeof(float));
                    if (gg) {
                        for (int i = 0; i < n && i < gn; i++)
                            gg[i] += dout[i] * (px->output->data[i] - mean) * inv_std;
                        tape_acc_grad(e->parent2, gg, gn);
                    }
                    free(gg);
                }
                // Beta gradient: d_beta[i] = dout[i]
                if (has_beta) {
                    int bn = g_tape.entries[e->parent3].output->len;
                    float* gb = (float*)calloc(bn, sizeof(float));
                    if (gb) {
                        for (int i = 0; i < n && i < bn; i++)
                            gb[i] += dout[i];
                        tape_acc_grad(e->parent3, gb, bn);
                    }
                    free(gb);
                }
            }
            break;
        }

        case NT_OP_SEQ_LAYERNORM: {
            // Same as LAYERNORM but per-position
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int has_beta = (e->parent3 >= 0 && e->parent3 < g_tape.count);
                float* gamma_data = has_gamma ? g_tape.entries[e->parent2].output->data : NULL;

                float* gx = (float*)calloc((size_t)T * D, sizeof(float));
                float* gg = has_gamma ? (float*)calloc(D, sizeof(float)) : NULL;
                float* gb = has_beta ? (float*)calloc(D, sizeof(float)) : NULL;

                if (gx) {
                    for (int t = 0; t < T; t++) {
                        float* x_t = px->output->data + t * D;
                        float* dout_t = dout + t * D;
                        float mean = 0;
                        for (int d = 0; d < D; d++) mean += x_t[d];
                        mean /= D;
                        float var = 0;
                        for (int d = 0; d < D; d++) { float dd = x_t[d] - mean; var += dd * dd; }
                        var /= D;
                        float inv_std = 1.0f / sqrtf(var + 1e-5f);

                        float sum_de = 0, sum_de_xhat = 0;
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            float xhat = (x_t[d] - mean) * inv_std;
                            sum_de += de;
                            sum_de_xhat += de * xhat;
                        }
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            float xhat = (x_t[d] - mean) * inv_std;
                            gx[t * D + d] = inv_std * (de - sum_de / D - xhat * sum_de_xhat / D);
                        }
                        if (gg) for (int d = 0; d < D; d++)
                            gg[d] += dout_t[d] * (x_t[d] - mean) * inv_std;
                        if (gb) for (int d = 0; d < D; d++)
                            gb[d] += dout_t[d];
                    }
                    tape_acc_grad(e->parent1, gx, T * D);
                    if (gg && has_gamma) tape_acc_grad(e->parent2, gg, D);
                    if (gb && has_beta) tape_acc_grad(e->parent3, gb, D);
                }
                free(gx); free(gg); free(gb);
            }
            break;
        }

        case NT_OP_ROPE: {
            // RoPE: rotation is orthogonal, backward = inverse rotation (transpose)
            // forward: x' = x*cos - y*sin, y' = x*sin + y*cos
            // backward: dx = dx'*cos + dy'*sin, dy = -dx'*sin + dy'*cos
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int total = px->output->len;
                int T = (int)e->aux;
                int D = total / T;
                // Recover head_dim from aux2 (stored when we fix forward)
                int head_dim = (int)e->aux2;
                if (head_dim <= 0) head_dim = D; // fallback: single head
                int n_heads = D / head_dim;

                float fb = (e->aux3 > 0.0f) ? e->aux3 : 10000.0f;
                int split_half = (e->aux4 > 0.5f) ? 1 : 0;
                int rope_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu && !split_half) {  /* GPU only handles even/odd */
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_gx   = gpu_scratch(3, total);
                    if (d_dout && d_gx) {
                        gpu_rope_backward(d_gx, d_dout, T, D, n_heads, head_dim, fb);
                        tape_acc_grad_gpu(e->parent1, d_gx, total);
                        rope_done_gpu = 1;
                    }
                }
#endif
                if (rope_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(e->grad);
#endif
                float* gx = (float*)calloc(total, sizeof(float));
                if (gx) {
                    int half = head_dim / 2;
                    for (int t = 0; t < T; t++) {
                        for (int h = 0; h < n_heads; h++) {
                            int base = t * D + h * head_dim;
                            for (int i = 0; i < half; i++) {
                                float freq = 1.0f / powf(fb, 2.0f * i / head_dim);
                                float angle = t * freq;
                                float cos_a = cosf(angle);
                                float sin_a = sinf(angle);
                                int o0 = split_half ? (base + i)        : (base + 2 * i);
                                int o1 = split_half ? (base + half + i) : (base + 2 * i + 1);
                                float dx0 = dout[o0];
                                float dx1 = dout[o1];
                                if (split_half) {
                                    /* Forward (Janus): n0 = x0*c + x1*s; n1 = -x0*s + x1*c
                                     * Backward (transpose): dx0 = c*dn0 - s*dn1; dx1 = s*dn0 + c*dn1 */
                                    gx[o0] =  dx0 * cos_a - dx1 * sin_a;
                                    gx[o1] =  dx0 * sin_a + dx1 * cos_a;
                                } else {
                                    /* Forward (notorch even/odd): n0 = x0*c - x1*s; n1 = x0*s + x1*c
                                     * Backward (transpose): dx0 = c*dn0 + s*dn1; dx1 = -s*dn0 + c*dn1 */
                                    gx[o0] = dx0 * cos_a + dx1 * sin_a;
                                    gx[o1] = -dx0 * sin_a + dx1 * cos_a;
                                }
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, gx, total);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SWIGLU: {
            // y = SiLU(gate) * up, silu(g) = g * σ(g)
            // d/dg silu(g) = σ(g) + g*σ(g)*(1-σ(g)) = σ(g) * (1 + g*(1-σ(g)))
            // dgate = dout * up * silu'(gate)
            // dup   = dout * silu(gate)
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pg = &g_tape.entries[e->parent1];
                nt_tape_entry* pu = &g_tape.entries[e->parent2];
                int n = out_len;
                int swi_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_G    = nt_tensor_ensure_gpu(pg->output);
                    float* d_U    = nt_tensor_ensure_gpu(pu->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_dg = gpu_scratch(3, n);
                    float* d_du = gpu_scratch(4, n);
                    if (d_G && d_U && d_dout && d_dg && d_du) {
                        gpu_swiglu_backward(d_dg, d_du, d_dout, d_G, d_U, n);
                        tape_acc_grad_gpu(e->parent1, d_dg, n);
                        tape_acc_grad_gpu(e->parent2, d_du, n);
                        swi_done_gpu = 1;
                    }
                }
#endif
                if (swi_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(pg->output);
                nt_tensor_ensure_cpu(pu->output);
#endif
                float* dg = (float*)calloc(n, sizeof(float));
                float* du = (float*)calloc(n, sizeof(float));
                if (dg && du) {
                    for (int i = 0; i < n; i++) {
                        float g = pg->output->data[i];
                        float u = pu->output->data[i];
                        float s = 1.0f / (1.0f + expf(-g));
                        float silu = g * s;
                        float dsilu_dg = s * (1.0f + g * (1.0f - s));
                        dg[i] = dout[i] * u * dsilu_dg;
                        du[i] = dout[i] * silu;
                    }
                    tape_acc_grad(e->parent1, dg, n);
                    tape_acc_grad(e->parent2, du, n);
                }
                free(dg); free(du);
            }
            break;
        }

        case NT_OP_BIT_LINEAR: {
            // STE: treat quantization as identity, so backward = standard matvec
            // dW[i,j] = dout[i] * x[j]
            // dx[j]   = Σ_i W[i,j] * dout[i]   (using full-precision W, per BitNet paper)
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0) {
                    float* dw = (float*)calloc((size_t)rows * cols, sizeof(float));
                    if (dw) {
                        for (int i = 0; i < rows; i++)
                            for (int j = 0; j < cols; j++)
                                dw[i * cols + j] = dout[i] * px->output->data[j];
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc(cols, sizeof(float));
                    if (dx) {
                        for (int j = 0; j < cols; j++) {
                            float acc = 0;
                            for (int i = 0; i < rows; i++)
                                acc += pw->output->data[i * cols + j] * dout[i];
                            dx[j] = acc;
                        }
                        tape_acc_grad(e->parent2, dx, cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        case NT_OP_BIT_SEQ_LINEAR: {
            // STE backward over T positions: dW = Σ_t dout[t] ⊗ x[t]; dx[t] = W^T @ dout[t]
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0 && T > 0) {
                    float* dw = (float*)calloc((size_t)rows * cols, sizeof(float));
                    if (dw) {
                        for (int t = 0; t < T; t++) {
                            const float* dout_t = dout + t * rows;
                            const float* x_t = px->output->data + t * cols;
                            for (int i = 0; i < rows; i++) {
                                float dot_i = dout_t[i];
                                float* dw_row = dw + i * cols;
                                for (int j = 0; j < cols; j++)
                                    dw_row[j] += dot_i * x_t[j];
                            }
                        }
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc((size_t)T * cols, sizeof(float));
                    if (dx) {
                        for (int t = 0; t < T; t++) {
                            const float* dout_t = dout + t * rows;
                            float* dx_t = dx + t * cols;
                            for (int j = 0; j < cols; j++) {
                                float acc = 0;
                                for (int i = 0; i < rows; i++)
                                    acc += pw->output->data[i * cols + j] * dout_t[i];
                                dx_t[j] = acc;
                            }
                        }
                        tape_acc_grad(e->parent2, dx, T * cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        default:
            break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPTIMIZERS
// ═══════════════════════════════════════════════════════════════════════════════

void nt_tape_adam_step(float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        if (!e->grad) { param_idx++; continue; }   // registered param w/o grad this step: keep slot alignment, skip update
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (!as->m || !as->v) { param_idx++; continue; }
        as->t++;
        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
            as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
            float m_hat = as->m->data[j] / (1.0f - powf(beta1, (float)as->t));
            float v_hat = as->v->data[j] / (1.0f - powf(beta2, (float)as->t));
            e->output->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
#ifdef USE_CUDA
        nt_tensor_mark_cpu_dirty(e->output);
#endif
        param_idx++;
    }
#ifdef USE_CUDA
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

void nt_tape_adamw_step(float lr, float weight_decay, float beta1, float beta2) {
    float eps = 1e-8f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        if (!e->grad) { param_idx++; continue; }   // registered param w/o grad this step: keep slot alignment, skip update
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (!as->m || !as->v) { param_idx++; continue; }
        as->t++;
        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        float bc1 = 1.0f - powf(beta1, (float)as->t);
        float bc2 = 1.0f - powf(beta2, (float)as->t);
        float wd = (e->no_decay) ? 0.0f : weight_decay;
        for (int j = 0; j < n; j++) {
            if (wd > 0.0f)
                e->output->data[j] -= lr * wd * e->output->data[j];
            float g = e->grad->data[j];
            as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
            as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
            float m_hat = as->m->data[j] / bc1;
            float v_hat = as->v->data[j] / bc2;
            e->output->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
#ifdef USE_CUDA
        nt_tensor_mark_cpu_dirty(e->output);
#endif
        param_idx++;
    }
#ifdef USE_CUDA
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

// ── Chuck optimizer ──────────────────────────────────────────────────────────

static float chuck_ring_avg(const float* buf, int pos, int full, int start, int count) {
    int len = full ? NT_CHUCK_WINDOW : pos;
    if (len == 0 || count == 0) return 0.0f;
    float sum = 0.0f;
    int actual = 0;
    for (int i = 0; i < count && i < len; i++) {
        int idx = (start + i) % NT_CHUCK_WINDOW;
        if (idx < len || full) { sum += buf[idx]; actual++; }
    }
    return actual > 0 ? sum / actual : 0.0f;
}

static uint32_t chuck_rng = 2463534242u;
static float chuck_randn(void) {
    chuck_rng ^= chuck_rng << 13;
    chuck_rng ^= chuck_rng >> 17;
    chuck_rng ^= chuck_rng << 5;
    return 2.0f * (float)(chuck_rng) / 4294967296.0f - 1.0f;
}

// Synced with PyTorch chuck.py (iamolegataeff/chuck.optimizer) 2026-04-06
// θ -= (α × S × λ × λ_l) × m̂/(√v̂ + ε) + η
void nt_tape_chuck_step(float lr, float loss_val) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // ── Level 1: Global loss trend → λ (dampen) ──
    nt_chuck_state* cs = &g_tape.chuck;
    if (!cs->initialized) {
        cs->dampen = 1.0f;
        cs->noise = 0.0f;
        cs->lr_scale = 1.0f;
        cs->best_macro = 1e9f;
        cs->initialized = 1;
    }
    if (cs->loss_ema == 0.0f) cs->loss_ema = loss_val;
    else cs->loss_ema = 0.99f * cs->loss_ema + 0.01f * loss_val;
    cs->loss_hist[cs->pos] = cs->loss_ema;
    cs->pos = (cs->pos + 1) % NT_CHUCK_WINDOW;
    if (cs->pos == 0) cs->full = 1;

    int len = cs->full ? NT_CHUCK_WINDOW : cs->pos;
    if (len >= 8) {
        int q = len / 4;
        if (q < 1) q = 1;
        int old_start = cs->full ? ((cs->pos) % NT_CHUCK_WINDOW) : 0;
        int recent_start = cs->full ? ((cs->pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW) : (cs->pos - q);
        float old_avg = chuck_ring_avg(cs->loss_hist, cs->pos, cs->full, old_start, q);
        float recent_avg = chuck_ring_avg(cs->loss_hist, cs->pos, cs->full, recent_start, q);
        if (old_avg > eps) {
            float trend = (recent_avg - old_avg) / old_avg;
            // Symmetric thresholds (synced with PyTorch: 0.02 / -0.02)
            if (trend > NT_CHUCK_TREND_BRAKE) cs->dampen *= NT_CHUCK_DAMP_DOWN;
            if (trend < NT_CHUCK_TREND_PUSH)  cs->dampen *= NT_CHUCK_DAMP_UP;

            // ── Level 3: Stagnation escape ──
            if (fabsf(trend) < NT_CHUCK_STAG_THRESH) {
                cs->stag++;
                if (cs->stag >= NT_CHUCK_STAG_STEPS) {
                    cs->noise = NT_CHUCK_NOISE_MAG;
                    cs->stag = 0;  // reset counter (PyTorch behavior)
                }
            } else {
                cs->stag = 0;
                cs->noise *= NT_CHUCK_NOISE_DECAY;  // exponential decay (was: reset to 0)
            }
        }
    }
    // Mean reversion: pull dampen toward 1.0 (prevents drift)
    cs->dampen = NT_CHUCK_MEAN_REVERT * cs->dampen + (1.0f - NT_CHUCK_MEAN_REVERT) * 1.0f;
    if (cs->dampen < NT_CHUCK_DAMP_LO) cs->dampen = NT_CHUCK_DAMP_LO;
    if (cs->dampen > NT_CHUCK_DAMP_HI) cs->dampen = NT_CHUCK_DAMP_HI;

    // ── Level 9: Multi-scale awareness (macro patience) ──
    cs->global_step++;
    if (cs->macro_ema == 0.0f) cs->macro_ema = loss_val;
    else cs->macro_ema = 0.999f * cs->macro_ema + 0.001f * loss_val;
    if (cs->global_step % NT_CHUCK_MACRO_INT == 0 && cs->global_step > NT_CHUCK_WINDOW) {
        if (cs->macro_ema > cs->best_macro * 0.999f) {
            cs->macro_stag++;
            if (cs->macro_stag >= NT_CHUCK_MACRO_PAT) {
                cs->lr_scale *= NT_CHUCK_MACRO_DECAY;
                if (cs->lr_scale < 0.05f) cs->lr_scale = 0.05f;
                cs->macro_stag = 0;
            }
        } else {
            cs->best_macro = cs->macro_ema;
            cs->macro_stag = 0;
            // LR recovery when improving (PyTorch: lr_scale *= 1.2)
            if (cs->lr_scale < 1.0f) {
                cs->lr_scale *= 1.2f;
                if (cs->lr_scale > 1.0f) cs->lr_scale = 1.0f;
            }
        }
    }

    float global_lambda = cs->dampen;
    float noise_mag = cs->noise;

    // ── Level 2: Per-param gradient norm + Adam update ──
#ifdef USE_CUDA
    /* L1 (2026-06-03): pre-compute ALL per-param grad norms in ONE batched device
     * readback (DEVICE pointer-mode, no per-call stall) instead of a blocking
     * cublasSnrm2-to-host per param in the loop below — the teen 0%-util sync
     * storm. Indexed by the same is_param+grad counter the update loop uses, so
     * chuck_gnorms[param_idx] aligns. n matches the loop's min(output,m) for the
     * params that use it → bit-identical norms. */
    float chuck_gnorms[NT_TAPE_MAX_PARAMS]; int chuck_gn_have = 0;
    if (g_use_gpu) {
        extern void gpu_nrm2_batch(const float**, const int*, int, float*);
        const float* d_gs[NT_TAPE_MAX_PARAMS]; int ns_arr[NT_TAPE_MAX_PARAMS];
        int pj = 0;
        for (int i = 0; i < g_tape.count && pj < g_tape.n_params; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            int n = e->output->len;
            nt_adam_state* as = &g_tape.adam[pj];
            if (as->m && as->m->len < n) n = as->m->len;
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            d_gs[pj] = d_g; ns_arr[pj] = d_g ? n : 0;
            pj++;
        }
        gpu_nrm2_batch(d_gs, ns_arr, pj, chuck_gnorms);
        chuck_gn_have = 1;
    }
#endif
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        if (!e->grad) { param_idx++; continue; }   // registered param w/o grad this step: keep slot alignment, skip update
        nt_adam_state* as = &g_tape.adam[param_idx];
        nt_chuck_param_state* cp = &g_tape.chuck_params[param_idx];
        if (cp->dampen == 0.0f) cp->dampen = 1.0f;
        if (cp->frozen) { param_idx++; continue; }
        if (!as->m || !as->v) { param_idx++; continue; }

        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        float gnorm = 0.0f;
#ifdef USE_CUDA
        if (g_use_gpu) {
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) {
                gnorm = chuck_gn_have ? chuck_gnorms[param_idx] : gpu_nrm2(d_g, n); /* L1 batched readback */
            } else {
                nt_tensor_ensure_cpu(e->grad);
                for (int j = 0; j < n; j++) gnorm += e->grad->data[j] * e->grad->data[j];
                gnorm = sqrtf(gnorm);
            }
        } else
#endif
        {
            for (int j = 0; j < n; j++) gnorm += e->grad->data[j] * e->grad->data[j];
            gnorm = sqrtf(gnorm);
        }

        cp->grad_hist[cp->pos] = gnorm;
        cp->pos = (cp->pos + 1) % NT_CHUCK_WINDOW;
        if (cp->pos == 0) cp->full = 1;

        int plen = cp->full ? NT_CHUCK_WINDOW : cp->pos;
        if (plen >= 8) {
            int q = plen / 4; if (q < 1) q = 1;
            int old_start = cp->full ? ((cp->pos) % NT_CHUCK_WINDOW) : 0;
            int recent_start = cp->full ? ((cp->pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW) : (cp->pos - q);
            float old_gn = chuck_ring_avg(cp->grad_hist, cp->pos, cp->full, old_start, q);
            float recent_gn = chuck_ring_avg(cp->grad_hist, cp->pos, cp->full, recent_start, q);
            if (old_gn > eps) {
                float gtrend = (recent_gn - old_gn) / old_gn;
                // Per-param: 0.05 thresholds (symmetric, PyTorch)
                if (gtrend > 0.05f)  cp->dampen *= NT_CHUCK_DAMP_UP;   // grad rising → boost
                if (gtrend < -0.05f) cp->dampen *= NT_CHUCK_DAMP_DOWN;  // grad settling → ease
            }
            if (gnorm < NT_CHUCK_FREEZE_THRESH) {
                cp->stag++;
                if (cp->stag >= NT_CHUCK_STAG_STEPS) cp->frozen = 1;
            } else {
                cp->stag = 0;
            }
            // Per-param mean reversion
            cp->dampen = NT_CHUCK_MEAN_REVERT * cp->dampen + (1.0f - NT_CHUCK_MEAN_REVERT) * 1.0f;
            if (cp->dampen < NT_CHUCK_DAMP_LO) cp->dampen = NT_CHUCK_DAMP_LO;
            if (cp->dampen > NT_CHUCK_DAMP_HI) cp->dampen = NT_CHUCK_DAMP_HI;
        }

        float param_lambda = cp->dampen;
        float effective_lr = lr * global_lambda * param_lambda * cs->lr_scale;
        as->t++;
        float bc1 = 1.0f - powf(beta1, (float)as->t);
        float bc2 = 1.0f - powf(beta2, (float)as->t);
        int chuck_done_gpu = 0;
#ifdef USE_CUDA
        /* GPU path: trivially parallel m,v update + param step. Skip when
         * Chuck noise injection is active (rare stagnation escape) since
         * CPU RNG is harder to port deterministically. */
        if (g_use_gpu && noise_mag == 0.0f) {
            float* d_p = nt_tensor_ensure_gpu(e->output);
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            float* d_m = nt_tensor_ensure_gpu(as->m);
            float* d_v = nt_tensor_ensure_gpu(as->v);
            if (d_p && d_g && d_m && d_v) {
                gpu_chuck_inner(d_p, d_m, d_v, d_g, n, beta1, beta2, bc1, bc2, effective_lr, eps);
                /* GPU is now source of truth for param, m, v. Mark CPU stale —
                 * next forward will read GPU directly without re-upload. */
                e->output->cpu_dirty = 1; e->output->gpu_valid = 1;
                as->m->cpu_dirty = 1;     as->m->gpu_valid = 1;
                as->v->cpu_dirty = 1;     as->v->gpu_valid = 1;
                chuck_done_gpu = 1;
            }
        }
#endif
        if (!chuck_done_gpu) {
            for (int j = 0; j < n; j++) {
                float g = e->grad->data[j];
                as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
                as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
                float m_hat = as->m->data[j] / bc1;
                float v_hat = as->v->data[j] / bc2;
                float update = effective_lr * m_hat / (sqrtf(v_hat) + eps);
                if (noise_mag > 0.0f) update += noise_mag * chuck_randn();
                e->output->data[j] -= update;
            }
#ifdef USE_CUDA
            /* CPU just mutated param weights — invalidate GPU mirror so next
             * forward re-uploads. */
            nt_tensor_mark_cpu_dirty(e->output);
#endif
        }
        param_idx++;
    }
#ifdef USE_CUDA
    /* Conservative belt-and-braces: mark global weight cache dirty too.
     * Cached entries (gpu_cache_weight) are not the same as per-tensor
     * d_data, but coa_v1_janus does not use the named weight cache today;
     * harmless if empty. */
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRADIENT UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

float nt_tape_clip_grads(float max_norm) {
    float total_norm_sq = 0.0f;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* L1 (2026-06-03): batch all per-param grad norms into ONE device readback
         * instead of one blocking cublasSnrm2-to-host per param. Plain gpu_nrm2
         * drains the stream every call (~42 here + 42 in Chuck = the 0%-util sync
         * storm). Numerically identical — same L2 norms, just read once. */
        extern void gpu_nrm2_batch(const float**, const int*, int, float*);
        const float* d_gs[NT_TAPE_MAX_PARAMS]; int ns_arr[NT_TAPE_MAX_PARAMS]; int k = 0;
        for (int i = 0; i < g_tape.count && k < NT_TAPE_MAX_PARAMS; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            int n = e->output->len;
            if (e->grad->len < n) n = e->grad->len;
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) { d_gs[k] = d_g; ns_arr[k] = n; k++; }
            else {
                nt_tensor_ensure_cpu(e->grad);
                for (int j = 0; j < n; j++) { float g = e->grad->data[j]; total_norm_sq += g * g; }
            }
        }
        if (k > 0) {
            float norms[NT_TAPE_MAX_PARAMS];
            gpu_nrm2_batch(d_gs, ns_arr, k, norms);
            for (int i = 0; i < k; i++) total_norm_sq += norms[i] * norms[i];
        }
    } else
#endif
    {
        for (int i = 0; i < g_tape.count; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            int n = e->output->len;
            if (e->grad->len < n) n = e->grad->len;
            for (int j = 0; j < n; j++) {
                float g = e->grad->data[j];
                total_norm_sq += g * g;
            }
        }
    }
    float total_norm = sqrtf(total_norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < g_tape.count; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            int n = e->output->len;
            if (e->grad->len < n) n = e->grad->len;
#ifdef USE_CUDA
            if (g_use_gpu) {
                float* d_g = nt_tensor_ensure_gpu(e->grad);
                if (d_g) {
                    gpu_sscal(d_g, n, scale);
                    /* GPU is now source of truth — mark CPU stale so later
                     * reads pull fresh values. */
                    e->grad->gpu_valid = 1;
                    e->grad->cpu_dirty = 1;
                    continue;
                }
            }
#endif
            for (int j = 0; j < n; j++) e->grad->data[j] *= scale;
#ifdef USE_CUDA
            e->grad->gpu_valid = 0;
            e->grad->cpu_dirty = 0;
#endif
        }
    }
    return total_norm;
}

void nt_tape_accum_grads(void) {
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        if (!e->grad) { param_idx++; continue; }   // registered param w/o grad this step: keep slot alignment, skip update
        nt_adam_state* as = &g_tape.adam[param_idx];
        int n = e->output->len;
        if (!as->acc_grad) {
            as->acc_grad = nt_tensor_new(n);
        } else if (as->acc_grad->len < n) {
            nt_tensor_free(as->acc_grad);
            as->acc_grad = nt_tensor_new(n);
        }
        for (int j = 0; j < n && j < as->acc_grad->len; j++)
            as->acc_grad->data[j] += e->grad->data[j];
        param_idx++;
    }
}

void nt_tape_apply_accum(int n_accum) {
    float scale = (n_accum > 1) ? 1.0f / (float)n_accum : 1.0f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (as->acc_grad) {
            int n = e->output->len;
            if (as->acc_grad->len < n) n = as->acc_grad->len;
            if (!e->grad) e->grad = nt_tensor_new(n);
            for (int j = 0; j < n; j++) {
                e->grad->data[j] = as->acc_grad->data[j] * scale;
                as->acc_grad->data[j] = 0.0f;
            }
        }
        param_idx++;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING MODE
// ═══════════════════════════════════════════════════════════════════════════════

static int g_training_mode = 1;

void nt_train_mode(int training) { g_training_mode = training; }
int  nt_is_training(void) { return g_training_mode; }

// ═══════════════════════════════════════════════════════════════════════════════
// LR SCHEDULE
// ═══════════════════════════════════════════════════════════════════════════════

nt_schedule nt_schedule_cosine(float base_lr, int warmup_steps, int total_steps, float min_lr) {
    nt_schedule s = {0};
    s.type = NT_SCHED_COSINE;
    s.base_lr = base_lr;
    s.min_lr = min_lr;
    s.warmup_steps = warmup_steps;
    s.total_steps = total_steps > 0 ? total_steps : 1;
    return s;
}

nt_schedule nt_schedule_step(float base_lr, int warmup_steps, int step_size, float gamma) {
    nt_schedule s = {0};
    s.type = NT_SCHED_STEP;
    s.base_lr = base_lr;
    s.warmup_steps = warmup_steps;
    s.step_size = step_size > 0 ? step_size : 1;
    s.step_gamma = gamma > 0 ? gamma : 0.1f;
    return s;
}

nt_schedule nt_schedule_linear(float base_lr, int warmup_steps, int total_steps, float min_lr) {
    nt_schedule s = {0};
    s.type = NT_SCHED_LINEAR;
    s.base_lr = base_lr;
    s.min_lr = min_lr;
    s.warmup_steps = warmup_steps;
    s.total_steps = total_steps > 0 ? total_steps : 1;
    return s;
}

float nt_schedule_get_lr(nt_schedule* s) {
    if (!s) return 0.001f;
    int step = s->current_step++;
    float lr = s->base_lr;

    // Warmup phase: linear ramp from min_lr to base_lr
    if (step < s->warmup_steps && s->warmup_steps > 0) {
        float t = (float)step / (float)s->warmup_steps;
        return s->min_lr + t * (s->base_lr - s->min_lr);
    }

    int decay_step = step - s->warmup_steps;

    switch (s->type) {
    case NT_SCHED_COSINE: {
        int decay_total = s->total_steps - s->warmup_steps;
        if (decay_total <= 0) return lr;
        float progress = (float)decay_step / (float)decay_total;
        if (progress > 1.0f) progress = 1.0f;
        lr = s->min_lr + 0.5f * (s->base_lr - s->min_lr) * (1.0f + cosf(3.14159265f * progress));
        break;
    }
    case NT_SCHED_STEP: {
        int n_decays = decay_step / s->step_size;
        lr = s->base_lr * powf(s->step_gamma, (float)n_decays);
        break;
    }
    case NT_SCHED_LINEAR: {
        int decay_total = s->total_steps - s->warmup_steps;
        if (decay_total <= 0) return lr;
        float progress = (float)decay_step / (float)decay_total;
        if (progress > 1.0f) progress = 1.0f;
        lr = s->base_lr - progress * (s->base_lr - s->min_lr);
        break;
    }
    default:
        break;
    }
    return lr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// NaN/Inf GUARD
// ═══════════════════════════════════════════════════════════════════════════════

nt_nan_guard nt_nan_guard_new(void) {
    nt_nan_guard g = {0};
    g.loss_scale = 1.0f;
    g.scale_factor = 2.0f;
    g.scale_window = 100;
    return g;
}

int nt_nan_guard_check(nt_nan_guard* guard) {
    if (!guard) return 1;
    int has_nan = 0;

    for (int i = 0; i < g_tape.count; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        int n = e->grad->len;
#ifdef USE_CUDA
        if (g_use_gpu) {
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) {
                /* NaN/Inf propagate through Snrm2: result = NaN if any input is NaN,
                 * Inf if any input is Inf. Cheap O(n) GPU reduction vs CPU loop. */
                float nrm = gpu_nrm2(d_g, n);
                if (nrm != nrm || nrm == 1.0f/0.0f || nrm == -1.0f/0.0f) {
                    has_nan = 1;
                }
                if (has_nan) break;
                continue;
            }
        }
#endif
        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            if (g != g || g == 1.0f/0.0f || g == -1.0f/0.0f) {  // NaN or Inf
                has_nan = 1;
                break;
            }
        }
        if (has_nan) break;
    }

    if (has_nan) {
        // Zero all gradients — don't apply this step
        for (int i = 0; i < g_tape.count; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            memset(e->grad->data, 0, e->grad->len * sizeof(float));
        }
        guard->loss_scale /= guard->scale_factor;
        if (guard->loss_scale < 1e-8f) guard->loss_scale = 1e-8f;
        guard->stable_steps = 0;
        guard->total_nan_count++;
        guard->skipped_steps++;
        return 0;
    }

    // Clean step
    guard->stable_steps++;
    if (guard->stable_steps >= guard->scale_window) {
        guard->loss_scale *= guard->scale_factor;
        if (guard->loss_scale > 65536.0f) guard->loss_scale = 65536.0f;
        guard->stable_steps = 0;
    }
    return 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILER
// ═══════════════════════════════════════════════════════════════════════════════

static nt_profiler g_profiler = {0};

void nt_profiler_enable(void)  { g_profiler.enabled = 1; }
void nt_profiler_disable(void) { g_profiler.enabled = 0; }
void nt_profiler_reset(void)   { memset(&g_profiler, 0, sizeof(g_profiler)); }
nt_profiler* nt_profiler_get(void) { return &g_profiler; }

void nt_profiler_print(void) {
    printf("── notorch profiler ──\n");
    printf("  ops: %d, params: %d (%ld elements, %.2f MB)\n",
           g_profiler.n_ops, g_profiler.n_params,
           g_profiler.total_param_elems,
           (float)g_profiler.total_param_elems * 4.0f / 1048576.0f);
    printf("  forward:   %.2f ms\n", g_profiler.forward_ms);
    printf("  backward:  %.2f ms\n", g_profiler.backward_ms);
    printf("  optimizer: %.2f ms\n", g_profiler.optimizer_ms);
    printf("  peak mem:  %.2f MB\n", (float)g_profiler.peak_memory / 1048576.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD OPS
// ═══════════════════════════════════════════════════════════════════════════════

int nt_embedding(int wte_idx, int token_id) {
    if (wte_idx < 0 || wte_idx >= g_tape.count) return -1;
    nt_tape_entry* wte = &g_tape.entries[wte_idx];
    int cols = wte->output->ndim >= 2 ? wte->output->shape[1] : wte->output->len;
    int rows = wte->output->len / cols;
    if (token_id < 0 || token_id >= rows) return -1;
    nt_tensor* out = nt_tensor_new(cols);
    if (!out) return -1;
    memcpy(out->data, wte->output->data + token_id * cols, cols * sizeof(float));
    int idx = nt_tape_record(out, NT_OP_EMB_LOOKUP, wte_idx, -1, (float)token_id);
    nt_tensor_free(out); // tape holds ref
    return idx;
}

int nt_seq_embedding(int wte_idx, int wpe_idx, int tokens_idx, int T, int D) {
    if (wte_idx < 0 || tokens_idx < 0) return -1;
    nt_tape_entry* wte = &g_tape.entries[wte_idx];
    nt_tape_entry* tok = &g_tape.entries[tokens_idx];
    int wte_rows = wte->output->ndim >= 2 ? wte->output->shape[0] : wte->output->len / D;

    nt_tensor* out = nt_tensor_new((size_t)T * D);
    if (!out) return -1;

#ifdef USE_CUDA
    if (g_use_gpu && wpe_idx < 0) {
        /* Pure WTE lookup on GPU. Skip WPE branch to keep kernel simple. */
        float* d_wte = nt_tensor_ensure_gpu(wte->output);
        float* d_tok = nt_tensor_ensure_gpu(tok->output);
        float* d_out = nt_tensor_ensure_gpu(out);
        if (d_wte && d_tok && d_out) {
            gpu_seq_embedding_forward(d_out, d_wte, d_tok, T, D, wte_rows);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record3(out, NT_OP_SEQ_EMBED, wte_idx, wpe_idx, tokens_idx, (float)T, (float)D);
            nt_tensor_free(out);
            return idx;
        }
    }
    nt_tensor_ensure_cpu(wte->output);
    nt_tensor_ensure_cpu(tok->output);
#endif
    for (int t = 0; t < T; t++) {
        int tid = (int)tok->output->data[t];
        if (tid < 0) tid = 0;
        if (tid >= wte_rows) tid = wte_rows - 1;
        for (int d = 0; d < D; d++)
            out->data[t * D + d] = wte->output->data[tid * D + d];
    }
    /* Add position embeddings if provided */
    if (wpe_idx >= 0) {
        nt_tape_entry* wpe = &g_tape.entries[wpe_idx];
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(wpe->output);
#endif
        int wpe_rows = wpe->output->ndim >= 2 ? wpe->output->shape[0] : wpe->output->len / D;
        for (int t = 0; t < T; t++) {
            int pos = t < wpe_rows ? t : wpe_rows - 1;
            for (int d = 0; d < D; d++)
                out->data[t * D + d] += wpe->output->data[pos * D + d];
        }
    }
    int idx = nt_tape_record3(out, NT_OP_SEQ_EMBED, wte_idx, wpe_idx, tokens_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_linear(int w_idx, int x_idx, int bias_idx) {
    if (w_idx < 0 || x_idx < 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;

    nt_tensor* out = nt_tensor_new(rows);
    if (!out) return -1;
    for (int i = 0; i < rows; i++) {
        float s = 0;
        for (int j = 0; j < cols; j++)
            s += pw->output->data[i * cols + j] * px->output->data[j];
        out->data[i] = s;
    }
    int idx = nt_tape_record(out, NT_OP_MATVEC, w_idx, x_idx, 0);
    nt_tensor_free(out);

    if (bias_idx >= 0) {
        idx = nt_add(idx, bias_idx);
    }
    return idx;
}

int nt_seq_linear(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int out_dim = pw->output->shape[0];
    int in_dim = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / out_dim;

    nt_tensor* out = nt_tensor_new((size_t)T * out_dim);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* Y(T, out_dim) = X(T, in_dim) @ W^T(in_dim, out_dim)
         * gpu_sgemm_nt: C(M,N) = A(M,K) × B^T(N,K), so M=T, N=out_dim, K=in_dim. */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_W = nt_tensor_ensure_gpu(pw->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_W && d_Y) {
            gpu_sgemm_nt(T, out_dim, in_dim, d_X, d_W, d_Y);
            nt_tensor_mark_gpu_fresh(out);  /* keep CPU mirror coherent for non-GPU ops */
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float* W = pw->output->data;
        float* X = px->output->data;
        float* Y = out->data;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, out_dim, in_dim,
                    1.0f, X, in_dim, W, in_dim,
                    0.0f, Y, out_dim);
#else
        for (int t = 0; t < T; t++) {
            float* x_t = X + t * in_dim;
            float* y_t = Y + t * out_dim;
            for (int i = 0; i < out_dim; i++) {
                float s = 0;
                for (int j = 0; j < in_dim; j++)
                    s += W[i * in_dim + j] * x_t[j];
                y_t[i] = s;
            }
        }
#endif
    }

    int idx = nt_tape_record3(out, NT_OP_SEQ_MATVEC, w_idx, x_idx, -1, (float)T, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_linear_t(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int W_rows = pw->output->shape[0];
    int W_cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / W_rows;

    /* W^T @ X[t]: input dim = W_rows, output dim = W_cols */
    nt_tensor* out = nt_tensor_new((size_t)T * W_cols);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* Y[T, W_cols] = X[T, W_rows] @ W[W_rows, W_cols] — NN gemm.
         * gpu_sgemm_nn(M, N, K, A, B, C):  C(M,N) = A(M,K) × B(K,N)
         *   M = T, N = W_cols, K = W_rows. */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_W = nt_tensor_ensure_gpu(pw->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_W && d_Y) {
            gpu_sgemm_nn(T, W_cols, W_rows, d_X, d_W, d_Y);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float* W = pw->output->data;
        float* X = px->output->data;
        float* Y = out->data;
#ifdef USE_BLAS
        /* Y[T, W_cols] = X[T, W_rows] @ W[W_rows, W_cols] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, W_cols, W_rows,
                    1.0f, X, W_rows, W, W_cols,
                    0.0f, Y, W_cols);
#else
        for (int t = 0; t < T; t++) {
            float* x_t = X + t * W_rows;
            float* y_t = Y + t * W_cols;
            for (int j = 0; j < W_cols; j++) {
                float s = 0;
                for (int i = 0; i < W_rows; i++)
                    s += W[i * W_cols + j] * x_t[i];
                y_t[j] = s;
            }
        }
#endif
    }

    int idx = nt_tape_record3(out, NT_OP_SEQ_MATVEC_T, w_idx, x_idx, -1, (float)T, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_rmsnorm(int x_idx, int gamma_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;

    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    float ss = 0;
    for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
    float rms = sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out->data[i] = px->output->data[i] / rms;

    // Apply gamma scale if provided
    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int i = 0; i < n && i < pg->output->len; i++)
            out->data[i] *= pg->output->data[i];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int idx = nt_tape_record(out, NT_OP_RMSNORM, x_idx, g_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_rmsnorm(int x_idx, int gamma_idx, int T, int D) {
    if (x_idx < 0 || T <= 0 || D <= 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];

    nt_tensor* out = nt_tensor_new((size_t)T * D);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* GPU forward: y = (x / rms) * gamma (single dispatch, gamma optional). */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        float* d_gamma = NULL;
        if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
            nt_tape_entry* pg = &g_tape.entries[gamma_idx];
            d_gamma = nt_tensor_ensure_gpu(pg->output);
        }
        if (d_X && d_Y) {
            gpu_seq_rmsnorm_gamma(d_Y, d_X, d_gamma, T, D);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int t = 0; t < T; t++) {
            float* x_t = px->output->data + t * D;
            float* o_t = out->data + t * D;
            float ss = 0;
            for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
            float rms = sqrtf(ss / D + 1e-6f);
            for (int d = 0; d < D; d++) o_t[d] = x_t[d] / rms;
        }
        if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
            nt_tape_entry* pg = &g_tape.entries[gamma_idx];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D && d < pg->output->len; d++)
                    out->data[t * D + d] *= pg->output->data[d];
        }
    }

    int g_idx2 = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_SEQ_RMSNORM, x_idx, g_idx2, -1, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_silu(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_silu(d_Y, d_X, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int i = 0; i < n; i++) {
            float x = px->output->data[i];
            out->data[i] = x / (1.0f + expf(-x));
        }
    }
    int idx = nt_tape_record(out, NT_OP_SILU, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_sigmoid(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    /* GPU-sync FIX (2026-06-02): the parent's forward output may live on GPU
     * with a stale calloc-zero CPU mirror; without this sync the sigmoid reads
     * zeros and a learnable gate sits frozen at sigmoid(0). Same bug class as
     * MUL/SILU/RMSNORM/CE. */
    nt_tensor_sync_cpu(px->output);
    for (int i = 0; i < n; i++) {
        float x = px->output->data[i];
        /* numerically stable */
        out->data[i] = (x >= 0) ? 1.0f / (1.0f + expf(-x))
                                : expf(x) / (1.0f + expf(x));
    }
    int idx = nt_tape_record(out, NT_OP_SIGMOID, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_relu(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    /* parent output may be GPU-resident with a stale CPU mirror — sync before
     * read (same bug class as SIGMOID/SILU). */
    nt_tensor_sync_cpu(px->output);
    for (int i = 0; i < n; i++) {
        float x = px->output->data[i];
        out->data[i] = x > 0.0f ? x : 0.0f;
    }
    int idx = nt_tape_record(out, NT_OP_RELU, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_gate(int x_idx, int g_idx, int T, int nm, int gi) {
    if (x_idx < 0 || g_idx < 0 || x_idx >= g_tape.count || g_idx >= g_tape.count) return -1;
    if (T <= 0 || nm <= 0 || gi < 0 || gi >= nm) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tape_entry* pg = &g_tape.entries[g_idx];
    if (!px->output || !pg->output) return -1;
    int n = px->output->len;
    if (n <= 0 || (n % T) != 0) return -1;
    if (pg->output->len != (long)T * nm) return -1;
    int B = n / T;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    /* parents may be GPU-resident with stale CPU mirrors — sync before read. */
    nt_tensor_sync_cpu(px->output);
    nt_tensor_sync_cpu(pg->output);
    for (int t = 0; t < T; t++) {
        float gv = pg->output->data[t * nm + gi];
        for (int d = 0; d < B; d++)
            out->data[t * B + d] = px->output->data[t * B + d] * gv;
    }
    int idx = nt_tape_record4(out, NT_OP_SEQ_GATE, x_idx, g_idx, -1,
                              (float)T, (float)nm, (float)gi, 0.0f);
    nt_tensor_free(out);
    return idx;
}

int nt_scale_by_t(int x_idx, int a_idx) {
    if (x_idx < 0 || a_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    if (pa->output->len != 1) return -1;  /* scalar required */
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    /* GPU-sync FIX (2026-06-02): both parents' forward outputs may be GPU-fresh
     * with stale CPU mirrors — without sync the scaled output is computed from
     * calloc-zero. Same bug class as MUL/SILU/RMSNORM/CE. */
    nt_tensor_sync_cpu(px->output);
    nt_tensor_sync_cpu(pa->output);
    float a_val = pa->output->data[0];
    for (int i = 0; i < n; i++) out->data[i] = a_val * px->output->data[i];
    int idx = nt_tape_record3(out, NT_OP_SCALE_BY_T, x_idx, a_idx, -1, 0, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_geglu(int x_idx, int w1_idx, int w2_idx, int T, int D_in, int D_out) {
    if (x_idx < 0 || w1_idx < 0 || w2_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tape_entry* pw1 = &g_tape.entries[w1_idx];
    nt_tape_entry* pw2 = &g_tape.entries[w2_idx];

    nt_tensor* out = nt_tensor_new((size_t)T * D_out);
    if (!out) return -1;

    for (int t = 0; t < T; t++) {
        float* x_t = px->output->data + t * D_in;
        for (int i = 0; i < D_out; i++) {
            float gate = 0, val = 0;
            for (int j = 0; j < D_in; j++) {
                gate += pw1->output->data[i * D_in + j] * x_t[j];
                val += pw2->output->data[i * D_in + j] * x_t[j];
            }
            // GELU approximation
            float x3 = gate * gate * gate;
            float inner = 0.7978845608f * (gate + 0.044715f * x3);
            float gelu = 0.5f * gate * (1.0f + tanhf(inner));
            out->data[t * D_out + i] = gelu * val;
        }
    }

    int idx = nt_tape_record3(out, NT_OP_GEGLU, x_idx, w1_idx, w2_idx, (float)(T * D_out), 0);
    nt_tensor_free(out);
    return idx;
}

int nt_softmax(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    float mx = px->output->data[0];
    for (int i = 1; i < n; i++) if (px->output->data[i] > mx) mx = px->output->data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out->data[i] = expf(px->output->data[i] - mx); sum += out->data[i]; }
    for (int i = 0; i < n; i++) out->data[i] /= sum;
    int idx = nt_tape_record(out, NT_OP_SOFTMAX, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_causal_attention(int q_idx, int k_idx, int v_idx, int T, int D) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];
    float scale = 1.0f / sqrtf((float)D);
    nt_tensor* out = nt_tensor_new((size_t)T * D);
    if (!out) return -1;
    for (int i = 0; i < T; i++) {
        float* qi = pq->output->data + i * D;
        float* scores = (float*)calloc(i + 1, sizeof(float));
        if (!scores) { nt_tensor_free(out); return -1; }
        float mx = -1e30f;
        for (int j = 0; j <= i; j++) {
            float* kj = pk->output->data + j * D;
            float dot = 0;
            for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0;
        for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        if (sum > 0) for (int j = 0; j <= i; j++) scores[j] /= sum;
        float* oi = out->data + i * D;
        for (int d = 0; d < D; d++) oi[d] = 0;
        for (int j = 0; j <= i; j++) {
            float* vj = pv->output->data + j * D;
            for (int d = 0; d < D; d++) oi[d] += scores[j] * vj[d];
        }
        free(scores);
    }
    int idx = nt_tape_record3(out, NT_OP_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_mh_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    int D = pq->output->len / T;
    int n_heads = D / head_dim;
    if (n_heads <= 0 || D % head_dim != 0) return -1;
    float scale = 1.0f / sqrtf((float)head_dim);

    nt_tensor* out = nt_tensor_new((size_t)T * D);
    if (!out) return -1;
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];

#ifdef USE_CUDA
    /* NT_DISABLE_MH_GPU env-guard also gates forward (extends prior backward
     * guard). Diagnostic for nanollama-Llama-3 forward kernel suspected NaN
     * source — Resonance bypassed plain MH via RRPRAM dual-attn, never
     * production-tested gpu_multi_head_attention forward on this shape. */
    int mh_gpu_disabled = getenv("NT_DISABLE_MH_GPU") != NULL;
    if (g_use_gpu && !mh_gpu_disabled) {
        float* d_Q = nt_tensor_ensure_gpu(pq->output);
        float* d_K = nt_tensor_ensure_gpu(pk->output);
        float* d_V = nt_tensor_ensure_gpu(pv->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        /* Scratch buffer for attention scores: n_heads * T * T floats. */
        float* d_scores = gpu_scratch(1, n_heads * T * T);
        if (d_Q && d_K && d_V && d_Y && d_scores) {
            gpu_multi_head_attention(d_Q, d_K, d_V, d_Y, d_scores, T, D, n_heads);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record3(out, NT_OP_MH_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)head_dim);
            nt_tensor_free(out);
            return idx;
        }
    }
    /* CPU fallback reads pq/pk/pv mirrors — sync first when GPU forward is on
     * for those parents (q/k/v come from nt_rope which is GPU-aware). */
    nt_tensor_sync_cpu(pq->output);
    nt_tensor_sync_cpu(pk->output);
    nt_tensor_sync_cpu(pv->output);
#endif

    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int ho = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* qi = pq->output->data + i * D + ho;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float* kj = pk->output->data + j * D + ho;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                scores_buf[j] = dot * scale;
                if (scores_buf[j] > mx) mx = scores_buf[j];
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sum += scores_buf[j]; }
            if (sum > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sum;
            float* oi = out->data + i * D + ho;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * D + ho;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);

    int idx = nt_tape_record3(out, NT_OP_MH_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

int nt_gqa_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim, int n_heads, int n_kv_heads) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    int Q_D = n_heads * head_dim;
    int KV_D = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    nt_tensor* out = nt_tensor_new((size_t)T * Q_D);
    if (!out) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];

    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        int q_off = h * head_dim;
        int kv_off = kv_h * head_dim;
        for (int i = 0; i < T; i++) {
            float* qi = pq->output->data + i * Q_D + q_off;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float* kj = pk->output->data + j * KV_D + kv_off;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                scores_buf[j] = dot * scale;
                if (scores_buf[j] > mx) mx = scores_buf[j];
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sum += scores_buf[j]; }
            if (sum > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sum;
            float* oi = out->data + i * Q_D + q_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * KV_D + kv_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);

    int idx = nt_tape_record4(out, NT_OP_GQA_ATTN, q_idx, k_idx, v_idx,
                              (float)T, (float)head_dim, (float)n_heads, (float)n_kv_heads);
    nt_tensor_free(out);
    return idx;
}

int nt_rrpram_attention(int wr_idx, int x_idx, int v_idx, int T, int n_embd, int nr_heads, int head_dim) {
    if (wr_idx < 0 || x_idx < 0 || v_idx < 0) return -1;
    int out_dim = nr_heads * head_dim;
    nt_tensor* out = nt_tensor_new((size_t)T * out_dim);
    if (!out) return -1;
    nt_tape_entry* pwr = &g_tape.entries[wr_idx];
    nt_tape_entry* px  = &g_tape.entries[x_idx];
    nt_tape_entry* pv  = &g_tape.entries[v_idx];
    int ctx = pwr->output->len / (nr_heads * n_embd);
    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < nr_heads; h++) {
        int wr_base = h * n_embd * ctx;
        int v_off = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* xi = px->output->data + i * n_embd;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float dot = 0;
                for (int d = 0; d < n_embd; d++)
                    dot += xi[d] * pwr->output->data[wr_base + d * ctx + j];
                scores_buf[j] = dot;
                if (dot > mx) mx = dot;
            }
            float sm = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sm += scores_buf[j]; }
            if (sm > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sm;
            float* oi = out->data + i * out_dim + v_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * out_dim + v_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);
    int idx = nt_tape_record4(out, NT_OP_RRPRAM_ATTN, wr_idx, x_idx, v_idx,
                              (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

/* ════════════════════════════════════════════════════════════════════════
 * Low-rank RRPRAM: Wr = Wr_a × Wr_b factorized.
 *
 * wr_combined layout: [Wr_a flat | Wr_b flat]
 *   Wr_a: H*E*R floats — head h offset = h*E*R, indexed [d, r] = h*E*R + d*R + r
 *   Wr_b: H*R*T_r floats — head h offset = H*E*R + h*R*T_r, indexed [r, j] = ... + r*T_r + j
 *   Total length = H*R*(E + T_r)
 *
 * Assumption: T_r == T (positional dim equals current ctx).
 * Rank derived: R = wr_combined->len / (H * (E + T))
 *
 * Per head h, position i (causal: j ≤ i):
 *   u[r]      = Σ_d xi[d] · Wr_a[h, d, r]              (matmul X[i,:] @ Wr_a[h])
 *   scores[j] = Σ_r u[r]   · Wr_b[h, r, j]              (matmul u @ Wr_b[h])
 *   attn[j]   = softmax(scores[0..i])
 *   out[d]    = Σ_j attn[j] · v[j, h_off+d]              (weighted sum of V)
 * ════════════════════════════════════════════════════════════════════════ */
int nt_rrpram_lowrank_attention(int wr_combined_idx, int x_idx, int v_idx,
                                 int T, int n_embd, int nr_heads, int head_dim) {
    if (wr_combined_idx < 0 || x_idx < 0 || v_idx < 0) return -1;
    int out_dim = nr_heads * head_dim;
    nt_tensor* out = nt_tensor_new((size_t)T * out_dim);
    if (!out) return -1;
    nt_tape_entry* pwr = &g_tape.entries[wr_combined_idx];
    nt_tape_entry* px  = &g_tape.entries[x_idx];
    nt_tape_entry* pv  = &g_tape.entries[v_idx];

    int T_r = T;   /* assumption */
    long combined_len = pwr->output->len;
    int rank = (int)(combined_len / ((long)nr_heads * (n_embd + T_r)));
    if (rank < 1) { nt_tensor_free(out); return -1; }
    long wra_total = (long)nr_heads * n_embd * rank;          /* offset of Wr_b section */

    int rrlr_done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X  = nt_tensor_ensure_gpu(px->output);
        float* d_Wr = nt_tensor_ensure_gpu(pwr->output);
        float* d_V  = nt_tensor_ensure_gpu(pv->output);
        float* d_O  = nt_tensor_ensure_gpu(out);
        /* Slot map (re-using free slots beyond MH/CE backward use):
         *   slot 1: forward-only, used by mh_attn forward — rrpram_lr never coexists.
         *           Reuse slot 1 for d_scores [H, T, T] of rrpram.
         *   slot 12: rrpram U buffer [H, T, R] — persisted to backward via tape.
         *   slot 13/14: rrpram backward d_attn / d_score scratch [H, T, T].
         * NOTE: forward U/scores must live in DEVICE buffers persisted across
         * forward→backward boundary. tape_clear frees activation tensor d_data.
         * Approach: cudaMalloc per-call into nt_tape entry's grad ptr is dirty.
         * Cleaner: alloc dedicated GPU scratch and snapshot it into a dedicated
         * tape slot. For now: backward will RECOMPUTE U and scores on GPU since
         * they are O(T·R·H) + O(T·T·H) ≈ 8·512·512 = 2M floats — cheap recompute. */
        int n_h = nr_heads;
        float* d_U      = gpu_scratch(12, n_h * T * rank);
        float* d_scores = gpu_scratch(1,  n_h * T * T);
        if (d_X && d_Wr && d_V && d_O && d_U && d_scores) {
            gpu_rrpram_lr_forward(d_X, d_Wr, d_V, d_O, d_U, d_scores,
                                  T, n_embd, n_h, rank, head_dim);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record4(out, NT_OP_RRPRAM_LR, wr_combined_idx, x_idx, v_idx,
                                      (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
            nt_tensor_free(out);
            return idx;
        }
    }
    /* CPU fallback — ensure inputs synced. */
    nt_tensor_ensure_cpu(pwr->output);
    nt_tensor_ensure_cpu(px->output);
    nt_tensor_ensure_cpu(pv->output);
#endif
    (void)rrlr_done_gpu;

    float* u_buf      = (float*)malloc(rank * sizeof(float));
    float* scores_buf = (float*)malloc(T_r  * sizeof(float));
    if (!u_buf || !scores_buf) { free(u_buf); free(scores_buf); nt_tensor_free(out); return -1; }

    for (int h = 0; h < nr_heads; h++) {
        long wr_a_base = (long)h * n_embd * rank;             /* Wr_a[h] */
        long wr_b_base = wra_total + (long)h * rank * T_r;     /* Wr_b[h] inside same buffer */
        int  v_off     = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* xi = px->output->data + i * n_embd;
            /* u[r] = Σ_d xi[d] · Wr_a[h, d, r] */
            for (int r = 0; r < rank; r++) u_buf[r] = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                float xd = xi[d];
                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                for (int r = 0; r < rank; r++) u_buf[r] += xd * wa_row[r];
            }
            /* scores[j] = Σ_r u[r] · Wr_b[h, r, j] for j ≤ i */
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float s = 0.0f;
                for (int r = 0; r < rank; r++) {
                    s += u_buf[r] * pwr->output->data[wr_b_base + (long)r * T_r + j];
                }
                scores_buf[j] = s;
                if (s > mx) mx = s;
            }
            /* softmax */
            float sm = 0.0f;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sm += scores_buf[j]; }
            if (sm > 0.0f) for (int j = 0; j <= i; j++) scores_buf[j] /= sm;
            /* out[d] = Σ_j attn[j] · v[j, h_off+d] */
            float* oi = out->data + i * out_dim + v_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0.0f;
            for (int j = 0; j <= i; j++) {
                const float* vj = pv->output->data + j * out_dim + v_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(u_buf); free(scores_buf);

    int idx = nt_tape_record4(out, NT_OP_RRPRAM_LR, wr_combined_idx, x_idx, v_idx,
                              (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

/* ════════════════════════════════════════════════════════════════════════
 * nt_rrpram_broadcast_attention
 *
 * Canonical Janus broadcast pattern (per dario/infer_v4.c:218-249).
 *
 *   mid[h, r]   = Σ_t Σ_e x[t, e] · Wr_a[h, e, r]                  (one mid per head, layer-broadcast)
 *   score[h, j] = Σ_r mid[h, r] · Wr_b[h, r, j]                    (one set of scores, broadcast across i)
 *   attn[h,i,j] = softmax(scores[h])[0..i] for j ≤ i               (causal softmax per i)
 *   out[i, h_off+d] = Σ_{j≤i} attn[h, i, j] · v[j, h_off+d]
 * ════════════════════════════════════════════════════════════════════════ */
int nt_rrpram_broadcast_attention(int wr_combined_idx, int x_idx, int v_idx,
                                   int T, int n_embd, int nr_heads, int head_dim, int rank) {
    if (wr_combined_idx < 0 || x_idx < 0 || v_idx < 0 ||
        wr_combined_idx >= g_tape.count || x_idx >= g_tape.count || v_idx >= g_tape.count) return -1;
    if (T < 1 || rank < 1 || nr_heads < 1 || head_dim < 1 || n_embd < 1) return -1;
    if (nr_heads * head_dim != n_embd) return -1;  /* invariant: H*D=E */
    int out_dim = nr_heads * head_dim;
    nt_tensor* out = nt_tensor_new((size_t)T * out_dim);
    if (!out) return -1;
    nt_tape_entry* pwr = &g_tape.entries[wr_combined_idx];
    nt_tape_entry* px  = &g_tape.entries[x_idx];
    nt_tape_entry* pv  = &g_tape.entries[v_idx];
    if (!pwr->output || !px->output || !pv->output) { nt_tensor_free(out); return -1; }
    if (px->output->len != (long)T * n_embd ||
        pv->output->len != (long)T * out_dim) {
        nt_tensor_free(out);
        return -1;
    }

    /* Packed weight shape: H*E*R + H*R*ctx_T = H*R*(E+ctx_T).
     * Derive ctx_T from combined_len / (H*R) - E (rank passed by caller). */
    long combined_len = pwr->output->len;
    long denom = (long)nr_heads * rank;
    if (combined_len <= 0 || (combined_len % denom) != 0) {
        nt_tensor_free(out);
        return -1;
    }
    int ctx_T = (int)(combined_len / denom - n_embd);
    if (ctx_T < T) { nt_tensor_free(out); return -1; }  /* runtime T must fit ctx */
    if ((long)nr_heads * rank * ((long)n_embd + ctx_T) != combined_len) {
        nt_tensor_free(out); return -1;  /* shape mismatch */
    }
    long wra_total = (long)nr_heads * n_embd * rank;
    /* Canonical Janus attention scale: 1/sqrt(D) per dario/infer_v4.c:239-244 */
    float sc = 1.0f / sqrtf((float)head_dim);

#ifdef USE_CUDA
    nt_tensor_ensure_cpu(pwr->output);
    nt_tensor_ensure_cpu(px->output);
    nt_tensor_ensure_cpu(pv->output);
#endif

    float* mid_buf    = (float*)malloc(rank * sizeof(float));
    float* all_scores = (float*)malloc(T  * sizeof(float));
    float* attn_buf   = (float*)malloc(T  * sizeof(float));
    if (!mid_buf || !all_scores || !attn_buf) {
        free(mid_buf); free(all_scores); free(attn_buf); nt_tensor_free(out); return -1;
    }

    for (int h = 0; h < nr_heads; h++) {
        long wr_a_base = (long)h * n_embd * rank;
        long wr_b_base = wra_total + (long)h * rank * ctx_T;
        int  v_off     = h * head_dim;

        for (int r = 0; r < rank; r++) mid_buf[r] = 0.0f;
        for (int t = 0; t < T; t++) {
            const float* xt = px->output->data + (long)t * n_embd;
            for (int e = 0; e < n_embd; e++) {
                float xe = xt[e];
                const float* wa_row = pwr->output->data + wr_a_base + (long)e * rank;
                for (int r = 0; r < rank; r++) mid_buf[r] += xe * wa_row[r];
            }
        }

        for (int j = 0; j < T; j++) {
            float s = 0.0f;
            for (int r = 0; r < rank; r++) {
                s += mid_buf[r] * pwr->output->data[wr_b_base + (long)r * ctx_T + j];
            }
            all_scores[j] = s * sc;
        }

        for (int i = 0; i < T; i++) {
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                attn_buf[j] = all_scores[j];
                if (attn_buf[j] > mx) mx = attn_buf[j];
            }
            float sm = 0.0f;
            for (int j = 0; j <= i; j++) {
                attn_buf[j] = expf(attn_buf[j] - mx);
                sm += attn_buf[j];
            }
            if (sm > 0.0f) for (int j = 0; j <= i; j++) attn_buf[j] /= sm;

            float* oi = out->data + (long)i * out_dim + v_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0.0f;
            for (int j = 0; j <= i; j++) {
                const float* vj = pv->output->data + (long)j * out_dim + v_off;
                for (int d = 0; d < head_dim; d++) oi[d] += attn_buf[j] * vj[d];
            }
        }
    }

    free(mid_buf); free(all_scores); free(attn_buf);

    /* aux4 stores RANK (not head_dim) — head_dim derivable at backward as E/H.
     * ctx_T is derivable from combined_len / (H*rank) - n_embd. */
    int idx = nt_tape_record4(out, NT_OP_RRPRAM_BCAST, wr_combined_idx, x_idx, v_idx,
                              (float)T, (float)n_embd, (float)nr_heads, (float)rank);
    nt_tensor_free(out);
    return idx;
}

int nt_concat(int a_idx, int b_idx, int T) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int Da = pa->output->len / T;
    int Db = pb->output->len / T;
    int Dc = Da + Db;
    nt_tensor* out = nt_tensor_new((size_t)T * Dc);
    if (!out) return -1;
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < Da; d++) out->data[t * Dc + d] = pa->output->data[t * Da + d];
        for (int d = 0; d < Db; d++) out->data[t * Dc + Da + d] = pb->output->data[t * Db + d];
    }
    int idx = nt_tape_record(out, NT_OP_CONCAT, a_idx, b_idx, (float)T);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWIGLU — y = SiLU(gate) * up (element-wise, pre-computed tensors)
// ═══════════════════════════════════════════════════════════════════════════════
int nt_swiglu(int gate_idx, int up_idx) {
    if (gate_idx < 0 || up_idx < 0) return -1;
    nt_tape_entry* pg = &g_tape.entries[gate_idx];
    nt_tape_entry* pu = &g_tape.entries[up_idx];
    int n = pg->output->len;
    if (pu->output->len != n) return -1;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    if (pg->output->ndim > 0)
        nt_tensor_reshape(out, pg->output->shape, pg->output->ndim);

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_G = nt_tensor_ensure_gpu(pg->output);
        float* d_U = nt_tensor_ensure_gpu(pu->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_G && d_U && d_Y) {
            gpu_swiglu(d_Y, d_G, d_U, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int i = 0; i < n; i++) {
            float g = pg->output->data[i];
            float s = 1.0f / (1.0f + expf(-g));
            out->data[i] = (g * s) * pu->output->data[i];  // silu(g) * u
        }
    }
    int idx = nt_tape_record(out, NT_OP_SWIGLU, gate_idx, up_idx, 0);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BITLINEAR — BitNet b1.58 (ternary W, int8 x, STE backward)
// Forward: Wq = clamp(round(W/γ_W), -1, +1), γ_W = mean|W|
//          xq = clamp(round(x * 127/γ_x), -128, +127), γ_x = max|x|
//          y = (γ_W γ_x / 127) × (Wq @ xq)
// Backward: STE — treats quant as identity, dW = dout ⊗ x, dx = W^T @ dout (full-precision W)
// ═══════════════════════════════════════════════════════════════════════════════
static inline float nt_bit_absmean(const float* w, int n) {
    if (n <= 0) return 1.0f;
    float s = 0; for (int i = 0; i < n; i++) s += fabsf(w[i]);
    float g = s / n;
    return g > 1e-8f ? g : 1e-8f;
}

static inline signed char nt_bit_ternary(float w, float inv_gamma) {
    int q = (int)lrintf(w * inv_gamma);
    if (q > 1) q = 1; else if (q < -1) q = -1;
    return (signed char)q;
}

static inline float nt_bit_int8_absmax(const float* x, int n) {
    float xmax = 0;
    for (int j = 0; j < n; j++) { float v = fabsf(x[j]); if (v > xmax) xmax = v; }
    return xmax > 1e-8f ? xmax : 1e-8f;
}

int nt_bit_linear(int w_idx, int x_idx) {
    if (w_idx < 0 || x_idx < 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
    if (rows <= 0 || cols <= 0) return -1;
    nt_tensor* out = nt_tensor_new(rows);
    if (!out) return -1;

    float gamma_w = nt_bit_absmean(pw->output->data, rows * cols);
    float inv_gw = 1.0f / gamma_w;
    float gamma_x = nt_bit_int8_absmax(px->output->data, cols);
    float inv_sx = 127.0f / gamma_x;
    float output_scale = gamma_w * gamma_x / 127.0f;

    signed char* x_q = (signed char*)calloc(cols, sizeof(signed char));
    if (!x_q) { nt_tensor_free(out); return -1; }
    for (int j = 0; j < cols; j++) {
        int q = (int)lrintf(px->output->data[j] * inv_sx);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        x_q[j] = (signed char)q;
    }

    const float* W = pw->output->data;
    for (int i = 0; i < rows; i++) {
        long long acc = 0;
        const float* W_row = W + i * cols;
        for (int j = 0; j < cols; j++)
            acc += (long long)nt_bit_ternary(W_row[j], inv_gw) * x_q[j];
        out->data[i] = output_scale * (float)acc;
    }
    free(x_q);

    int idx = nt_tape_record(out, NT_OP_BIT_LINEAR, w_idx, x_idx, gamma_w);
    nt_tensor_free(out);
    return idx;
}

int nt_bit_seq_linear(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
    if (rows <= 0 || cols <= 0) return -1;

    nt_tensor* out = nt_tensor_new((size_t)T * rows);
    if (!out) return -1;

    float gamma_w = nt_bit_absmean(pw->output->data, rows * cols);
    float inv_gw = 1.0f / gamma_w;

    /* Pre-quantize W to ternary stored as FLOAT (so cblas_sgemm can consume it) */
    float* Wq_f = (float*)malloc((size_t)rows * cols * sizeof(float));
    if (!Wq_f) { nt_tensor_free(out); return -1; }
    for (int i = 0; i < rows * cols; i++) {
        int q = (int)lrintf(pw->output->data[i] * inv_gw);
        if (q > 1) q = 1; else if (q < -1) q = -1;
        Wq_f[i] = (float)q;
    }

    /* Pre-quantize full X per-position to int8-range FLOAT, store per-position scale */
    float* Xq_f = (float*)malloc((size_t)T * cols * sizeof(float));
    float* gamma_x_per_t = (float*)malloc(T * sizeof(float));
    if (!Xq_f || !gamma_x_per_t) {
        free(Wq_f); free(Xq_f); free(gamma_x_per_t); nt_tensor_free(out); return -1;
    }
    for (int t = 0; t < T; t++) {
        const float* x_row = px->output->data + t * cols;
        float gamma_x = nt_bit_int8_absmax(x_row, cols);
        gamma_x_per_t[t] = gamma_x;
        float inv_sx = 127.0f / gamma_x;
        float* xq_row = Xq_f + t * cols;
        for (int j = 0; j < cols; j++) {
            float q = lrintf(x_row[j] * inv_sx);
            if (q > 127.0f) q = 127.0f; else if (q < -128.0f) q = -128.0f;
            xq_row[j] = q;
        }
    }

#ifdef USE_BLAS
    /* Single BLAS matmul: Y[T,rows] = Xq[T,cols] @ Wq^T[cols,rows]
     * Wq stored row-major as [rows, cols] so CblasTrans gives Wq^T. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, rows, cols,
                1.0f, Xq_f, cols, Wq_f, cols,
                0.0f, out->data, rows);
    /* Apply per-position output scale (gamma_w * gamma_x / 127) */
    float base = gamma_w / 127.0f;
    for (int t = 0; t < T; t++) {
        float s = base * gamma_x_per_t[t];
        float* y_row = out->data + t * rows;
        for (int i = 0; i < rows; i++) y_row[i] *= s;
    }
#else
    for (int t = 0; t < T; t++) {
        float output_scale = gamma_w * gamma_x_per_t[t] / 127.0f;
        const float* xq_row = Xq_f + t * cols;
        float* y_row = out->data + t * rows;
        for (int i = 0; i < rows; i++) {
            float acc = 0;
            const float* Wq_row = Wq_f + i * cols;
            for (int j = 0; j < cols; j++) acc += Wq_row[j] * xq_row[j];
            y_row[i] = output_scale * acc;
        }
    }
#endif

    free(Wq_f);
    free(Xq_f);
    free(gamma_x_per_t);

    int idx = nt_tape_record3(out, NT_OP_BIT_SEQ_LINEAR, w_idx, x_idx, -1, (float)T, gamma_w);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPA — Sentence Phonon Attention (inference-time; pure helpers, no tape)
// ═══════════════════════════════════════════════════════════════════════════════
void nt_spa_embed_sentence(const int* tokens, int n_tokens,
                           const float* W_embed, int vocab_size, int dim,
                           float alpha, float* out_emb) {
    if (!tokens || !W_embed || !out_emb || n_tokens <= 0 || dim <= 0 || vocab_size <= 0) return;
    if (alpha < 0 || alpha > 1) alpha = 0.85f;

    for (int d = 0; d < dim; d++) out_emb[d] = 0;

    float total_weight = 0;
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= vocab_size) continue;
        float w = powf(alpha, (float)(n_tokens - 1 - i));
        total_weight += w;
        const float* row = W_embed + (size_t)tok * dim;
        for (int d = 0; d < dim; d++) out_emb[d] += w * row[d];
    }
    if (total_weight > 0)
        for (int d = 0; d < dim; d++) out_emb[d] /= total_weight;
}

float nt_spa_connectedness(const float* query_emb, int dim,
                           const float* sentence_embeddings, int n_sentences) {
    if (!query_emb || !sentence_embeddings || dim <= 0 || n_sentences <= 0) return 0;
    float scale = 1.0f / sqrtf((float)dim);

    float* scores = (float*)calloc(n_sentences, sizeof(float));
    if (!scores) return 0;

    float max_s = -1e30f;
    for (int i = 0; i < n_sentences; i++) {
        float s = 0;
        const float* emb = sentence_embeddings + (size_t)i * dim;
        for (int d = 0; d < dim; d++) s += query_emb[d] * emb[d];
        s *= scale;
        scores[i] = s;
        if (s > max_s) max_s = s;
    }
    float sum = 0;
    for (int i = 0; i < n_sentences; i++) { scores[i] = expf(scores[i] - max_s); sum += scores[i]; }
    float max_attn = 0;
    if (sum > 0) {
        for (int i = 0; i < n_sentences; i++) {
            float w = scores[i] / sum;
            if (w > max_attn) max_attn = w;
        }
    }
    free(scores);
    return max_attn;
}

void nt_spa_modulate_logits(float* logits, int V, float connectedness, float strength) {
    if (!logits || V <= 0) return;
    if (connectedness < 0) connectedness = 0;
    if (connectedness > 1) connectedness = 1;
    float spa_temp = 1.0f - strength * connectedness;
    if (spa_temp < 1e-3f) spa_temp = 1e-3f;
    float inv = 1.0f / spa_temp;
    for (int i = 0; i < V; i++) logits[i] *= inv;
}

int nt_cross_entropy(int logits_idx, int target) {
    if (logits_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    int n = pl->output->len;
    if (target < 0 || target >= n) return -1;
    float mx = pl->output->data[0];
    for (int i = 1; i < n; i++) if (pl->output->data[i] > mx) mx = pl->output->data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) sum += expf(pl->output->data[i] - mx);
    float log_sm = pl->output->data[target] - mx - logf(sum);
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;
    out->data[0] = -log_sm;
    int idx = nt_tape_record(out, NT_OP_CROSS_ENT, logits_idx, -1, (float)target);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_cross_entropy(int logits_idx, int targets_idx, int T, int V) {
    if (logits_idx < 0 || targets_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    nt_tape_entry* pt = &g_tape.entries[targets_idx];
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_L = nt_tensor_ensure_gpu(pl->output);
        float* d_T = nt_tensor_ensure_gpu(pt->output);
        /* per-position losses scratch. gpu_cross_entropy reads it back to
         * compute the mean — the value is a host float. */
        float* d_losses = gpu_scratch(2, T);
        if (d_L && d_T && d_losses) {
            float mean = gpu_cross_entropy(d_L, d_T, d_losses, T, V);
            out->data[0] = mean;
            /* loss is a 1-element CPU value — mark GPU mirror invalid in case
             * something later tries to consume it on GPU. */
            out->gpu_valid = 0;
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float total_loss = 0;
        for (int t = 0; t < T; t++) {
            float* logits_t = pl->output->data + t * V;
            int target = (int)pt->output->data[t];
            if (target < 0 || target >= V) target = 0;
            float mx = logits_t[0];
            for (int j = 1; j < V; j++) if (logits_t[j] > mx) mx = logits_t[j];
            float sum = 0;
            for (int j = 0; j < V; j++) sum += expf(logits_t[j] - mx);
            total_loss += -(logits_t[target] - mx - logf(sum));
        }
        out->data[0] = total_loss / T;
    }
    int idx = nt_tape_record3(out, NT_OP_SEQ_CROSSENT, logits_idx, targets_idx, -1, (float)T, (float)V);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_cross_entropy_masked(int logits_idx, int targets_idx, int mask_idx, int T, int V) {
    if (logits_idx < 0 || targets_idx < 0 || mask_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    nt_tape_entry* pt = &g_tape.entries[targets_idx];
    nt_tape_entry* pm = &g_tape.entries[mask_idx];
#ifdef USE_CUDA
    /* CPU op — pull GPU mirrors back. Without these calls, when callers
     * use GPU mode, logits arrive GPU-fresh / CPU-stale (zeros) →
     * softmax(zeros) = uniform → loss = ln(V) every step regardless of
     * input. Caught during heart.c phase 1 cal: loss exactly 9.7041 =
     * ln(16384) on Resonance until this fix landed. */
    nt_tensor_ensure_cpu(pl->output);
    nt_tensor_ensure_cpu(pt->output);
    nt_tensor_ensure_cpu(pm->output);
#endif
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;
    float total_loss = 0;
    float n_active = 0;
    for (int t = 0; t < T; t++) {
        float m = pm->output->data[t];
        if (m == 0.0f) continue;
        float* logits_t = pl->output->data + t * V;
        int target = (int)pt->output->data[t];
        if (target < 0 || target >= V) target = 0;
        float mx = logits_t[0];
        for (int j = 1; j < V; j++) if (logits_t[j] > mx) mx = logits_t[j];
        float sum = 0;
        for (int j = 0; j < V; j++) sum += expf(logits_t[j] - mx);
        total_loss += m * -(logits_t[target] - mx - logf(sum));
        n_active += m;
    }
    out->data[0] = (n_active > 0) ? total_loss / n_active : 0.0f;
    int idx = nt_tape_record3(out, NT_OP_SEQ_CROSSENT_MASKED, logits_idx, targets_idx, mask_idx, (float)T, (float)V);
    nt_tensor_free(out);
    return idx;
}

int nt_add(int a_idx, int b_idx) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int n = pa->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    /* GPU add requires equal-length operands (no broadcast). Skip when
     * shapes mismatch — fall back to CPU broadcast loop. */
    if (g_use_gpu && pb->output->len == n) {
        float* d_A = nt_tensor_ensure_gpu(pa->output);
        float* d_B = nt_tensor_ensure_gpu(pb->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_A && d_B && d_Y) {
            gpu_add(d_Y, d_A, d_B, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(pa->output);
        nt_tensor_ensure_cpu(pb->output);
#endif
        for (int i = 0; i < n; i++)
            out->data[i] = pa->output->data[i] + pb->output->data[i % pb->output->len];
    }
    int idx = nt_tape_record(out, NT_OP_ADD, a_idx, b_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_mul(int a_idx, int b_idx) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int n = pa->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu && pb->output->len == n) {
        float* d_A = nt_tensor_ensure_gpu(pa->output);
        float* d_B = nt_tensor_ensure_gpu(pb->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_A && d_B && d_Y) {
            gpu_mul(d_Y, d_A, d_B, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(pa->output);
        nt_tensor_ensure_cpu(pb->output);
#endif
        for (int i = 0; i < n; i++)
            out->data[i] = pa->output->data[i] * pb->output->data[i % pb->output->len];
    }
    int idx = nt_tape_record(out, NT_OP_MUL, a_idx, b_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_scale(int x_idx, float s) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_scale(d_Y, d_X, n, s);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(px->output);
#endif
        for (int i = 0; i < n; i++) out->data[i] = px->output->data[i] * s;
    }
    int idx = nt_tape_record(out, NT_OP_SCALE, x_idx, -1, s);
    nt_tensor_free(out);
    return idx;
}

int nt_rope_freq(int x_idx, int T, int head_dim, float freq_base) {
    if (x_idx < 0 || T <= 0 || head_dim <= 0) return -1;
    if (freq_base <= 0.0f) freq_base = 10000.0f;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int total = px->output->len;
    int D = total / T;
    int n_heads = D / head_dim;
    if (n_heads <= 0) return -1;

    nt_tensor* out = nt_tensor_new(total);
    if (!out) return -1;
    if (px->output->ndim > 0) nt_tensor_reshape(out, px->output->shape, px->output->ndim);

    int done_gpu = 0;
#ifdef USE_CUDA
    int rope_gpu_disabled = getenv("NT_DISABLE_ROPE_GPU") != NULL;
    if (g_use_gpu && !rope_gpu_disabled) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_rope_forward(d_Y, d_X, T, D, n_heads, head_dim, freq_base);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
    if (!done_gpu) nt_tensor_ensure_cpu(px->output);
#endif

    if (!done_gpu) {
        memcpy(out->data, px->output->data, total * sizeof(float));
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                int base = t * D + h * head_dim;
                for (int i = 0; i < head_dim / 2; i++) {
                    float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
                    float angle = t * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float x0 = out->data[base + 2 * i];
                    float x1 = out->data[base + 2 * i + 1];
                    out->data[base + 2 * i] = x0 * cos_a - x1 * sin_a;
                    out->data[base + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }

    int idx = nt_tape_record4(out, NT_OP_ROPE, x_idx, -1, -1, (float)T, (float)head_dim, freq_base, 0.0f);
    nt_tensor_free(out);
    return idx;
}

int nt_rope(int x_idx, int T, int head_dim) {
    return nt_rope_freq(x_idx, T, head_dim, 10000.0f);
}

int nt_rope_split_half_freq(int x_idx, int T, int head_dim, float freq_base) {
    /* Split-half RoPE: pairs (i, i+head_dim/2) instead of even/odd
     * (2i, 2i+1). Used by nanochat / Janus v4 (infer_v4.c:35-49).
     * Sign convention matches canonical Janus rope_pos:
     *   q[i]      =  q0*cos + q1*sin
     *   q[i+half] = -q0*sin + q1*cos
     * (notorch's even/odd nt_rope_freq uses the inverse rotation.)
     * CPU-only forward; dispatches via NT_OP_ROPE with aux4=1.0
     * — backward case branches on aux4 for split-half formulas. */
    if (x_idx < 0 || T <= 0 || head_dim <= 0) return -1;
    if (freq_base <= 0.0f) freq_base = 10000.0f;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int total = px->output->len;
    int D = total / T;
    int n_heads = D / head_dim;
    int half = head_dim / 2;
    if (n_heads <= 0 || half <= 0) return -1;

    nt_tensor* out = nt_tensor_new(total);
    if (!out) return -1;
    if (px->output->ndim > 0) nt_tensor_reshape(out, px->output->shape, px->output->ndim);

#ifdef USE_CUDA
    nt_tensor_ensure_cpu(px->output);
#endif
    memcpy(out->data, px->output->data, total * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_heads; h++) {
            int base = t * D + h * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
                float angle = t * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = out->data[base + i];
                float x1 = out->data[base + half + i];
                out->data[base + i]        =  x0 * cos_a + x1 * sin_a;
                out->data[base + half + i] = -x0 * sin_a + x1 * cos_a;
            }
        }
    }
#ifdef USE_CUDA
    out->cpu_dirty = 0; out->gpu_valid = 0;
#endif
    int idx = nt_tape_record4(out, NT_OP_ROPE, x_idx, -1, -1,
                              (float)T, (float)head_dim, freq_base, 1.0f /* split-half */);
    nt_tensor_free(out);
    return idx;
}

int nt_dropout(int x_idx, float p) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    if (g_training_mode && p > 0.0f && p < 1.0f) {
        float scale = 1.0f / (1.0f - p);  // inverted dropout
        for (int i = 0; i < n; i++) {
            float r = rand_uniform();
            out->data[i] = (r >= p) ? px->output->data[i] * scale : 0.0f;
        }
    } else {
        memcpy(out->data, px->output->data, n * sizeof(float));
    }

    // Store the dropout mask in output for backward (mask encoded as: 0 = dropped, scale = kept)
    int idx = nt_tape_record(out, NT_OP_DROPOUT, x_idx, -1, p);
    nt_tensor_free(out);
    return idx;
}

int nt_gelu(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    for (int i = 0; i < n; i++) {
        float x = px->output->data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    int idx = nt_tape_record(out, NT_OP_GELU, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_layernorm(int x_idx, int gamma_idx, int beta_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    // Compute mean and variance
    float mean = 0;
    for (int i = 0; i < n; i++) mean += px->output->data[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) {
        float d = px->output->data[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < n; i++)
        out->data[i] = (px->output->data[i] - mean) * inv_std;

    // Apply affine: gamma * normalized + beta
    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int i = 0; i < n && i < pg->output->len; i++)
            out->data[i] *= pg->output->data[i];
    }
    if (beta_idx >= 0 && beta_idx < g_tape.count) {
        nt_tape_entry* pb = &g_tape.entries[beta_idx];
        for (int i = 0; i < n && i < pb->output->len; i++)
            out->data[i] += pb->output->data[i];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int b_idx = (beta_idx >= 0 && beta_idx < g_tape.count) ? beta_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_LAYERNORM, x_idx, g_idx, b_idx, 0, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_layernorm(int x_idx, int gamma_idx, int beta_idx, int T, int D) {
    if (x_idx < 0 || T <= 0 || D <= 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tensor* out = nt_tensor_new((size_t)T * D);
    if (!out) return -1;

    for (int t = 0; t < T; t++) {
        float* x_t = px->output->data + t * D;
        float* o_t = out->data + t * D;
        float mean = 0;
        for (int d = 0; d < D; d++) mean += x_t[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) { float dd = x_t[d] - mean; var += dd * dd; }
        var /= D;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D; d++) o_t[d] = (x_t[d] - mean) * inv_std;
    }

    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D && d < pg->output->len; d++)
                out->data[t * D + d] *= pg->output->data[d];
    }
    if (beta_idx >= 0 && beta_idx < g_tape.count) {
        nt_tape_entry* pb = &g_tape.entries[beta_idx];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D && d < pb->output->len; d++)
                out->data[t * D + d] += pb->output->data[d];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int b_idx = (beta_idx >= 0 && beta_idx < g_tape.count) ? beta_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_SEQ_LAYERNORM, x_idx, g_idx, b_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER
// ═══════════════════════════════════════════════════════════════════════════════

static void bpe_build_decode_table(nt_bpe* bpe) {
    for (int i = 0; i < 256; i++) {
        bpe->tokens[i][0] = (unsigned char)i;
        bpe->token_len[i] = 1;
    }
    for (int m = 0; m < bpe->n_merges; m++) {
        int new_id = 256 + m;
        int a = bpe->merges[m][0];
        int b = bpe->merges[m][1];
        int la = bpe->token_len[a];
        int lb = bpe->token_len[b];
        if (la + lb < NT_BPE_MAX_TOKEN_LEN) {
            memcpy(bpe->tokens[new_id], bpe->tokens[a], la);
            memcpy(bpe->tokens[new_id] + la, bpe->tokens[b], lb);
            bpe->token_len[new_id] = la + lb;
        }
    }
}

void nt_bpe_init(nt_bpe* bpe, const int merges[][2], int n_merges) {
    memset(bpe, 0, sizeof(nt_bpe));
    bpe->n_merges = n_merges;
    bpe->vocab_size = 256 + n_merges;
    for (int i = 0; i < n_merges; i++) {
        bpe->merges[i][0] = merges[i][0];
        bpe->merges[i][1] = merges[i][1];
    }
    bpe_build_decode_table(bpe);
}

int nt_bpe_load(nt_bpe* bpe, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    memset(bpe, 0, sizeof(nt_bpe));
    int a, b, n = 0;
    while (fscanf(f, "%d %d", &a, &b) == 2 && n < NT_BPE_MAX_MERGES) {
        bpe->merges[n][0] = a;
        bpe->merges[n][1] = b;
        n++;
    }
    fclose(f);
    bpe->n_merges = n;
    bpe->vocab_size = 256 + n;
    bpe_build_decode_table(bpe);
    return n;
}

int nt_bpe_encode(const nt_bpe* bpe, const char* text, int text_len, int* out, int max_tokens) {
    if (!text || text_len <= 0 || !out || max_tokens <= 0) return 0;
    int n = 0;
    for (int i = 0; i < text_len && n < max_tokens; i++)
        out[n++] = (unsigned char)text[i];
    /* Two-pointer write — O(n) per merge instead of O(n²).
     * Old shift-on-match was catastrophic on multi-MB corpora. */
    for (int m = 0; m < bpe->n_merges; m++) {
        int a = bpe->merges[m][0];
        int b = bpe->merges[m][1];
        int new_id = 256 + m;
        int w = 0, r = 0;
        while (r < n) {
            if (r + 1 < n && out[r] == a && out[r + 1] == b) {
                out[w++] = new_id;
                r += 2;
            } else {
                out[w++] = out[r++];
            }
        }
        n = w;
    }
    return n;
}

int nt_bpe_decode(const nt_bpe* bpe, const int* tokens, int n_tokens, char* out, int max_bytes) {
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        int id = tokens[i];
        if (id < 0 || id >= bpe->vocab_size) continue;
        int len = bpe->token_len[id];
        if (pos + len >= max_bytes) break;
        memcpy(out + pos, bpe->tokens[id], len);
        pos += len;
    }
    out[pos] = '\0';
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATALOADER
// ═══════════════════════════════════════════════════════════════════════════════

nt_dataloader* nt_dataloader_create(const char* text_file, nt_bpe* bpe,
                                     int seq_len, int batch_size) {
    if (!text_file || !bpe || seq_len <= 0 || batch_size <= 0) return NULL;

    // Read entire file
    FILE* f = fopen(text_file, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = (char*)malloc(fsize + 1);
    if (!text) { fclose(f); return NULL; }
    if (fread(text, 1, (size_t)fsize, f) != (size_t)fsize) { free(text); fclose(f); return NULL; }
    text[fsize] = 0;
    fclose(f);

    // Tokenize
    int* tokens = (int*)malloc(fsize * sizeof(int)); // worst case: 1 token per char
    if (!tokens) { free(text); return NULL; }
    int n_tokens = nt_bpe_encode(bpe, text, (int)fsize, tokens, (int)fsize);
    free(text);

    if (n_tokens < seq_len + 1) { free(tokens); return NULL; }

    // Shrink tokens array
    int* shrunk = (int*)realloc(tokens, n_tokens * sizeof(int));
    if (shrunk) tokens = shrunk;

    nt_dataloader* dl = (nt_dataloader*)calloc(1, sizeof(nt_dataloader));
    if (!dl) { free(tokens); return NULL; }
    dl->tokens = tokens;
    dl->n_tokens = n_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->n_batches = (n_tokens - 1) / (seq_len * batch_size);
    if (dl->n_batches <= 0) dl->n_batches = 1;

    // Create shuffle indices
    dl->shuffle_indices = (int*)malloc(dl->n_batches * sizeof(int));
    for (int i = 0; i < dl->n_batches; i++) dl->shuffle_indices[i] = i;

    return dl;
}

nt_dataloader* nt_dataloader_from_tokens(const char* token_file,
                                          int seq_len, int batch_size) {
    if (!token_file || seq_len <= 0 || batch_size <= 0) return NULL;
    FILE* f = fopen(token_file, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n_tokens = (int)(fsize / sizeof(int));
    if (n_tokens < seq_len + 1) { fclose(f); return NULL; }
    int* tokens = (int*)malloc(n_tokens * sizeof(int));
    if (!tokens) { fclose(f); return NULL; }
    if (fread(tokens, sizeof(int), (size_t)n_tokens, f) != (size_t)n_tokens) {
        free(tokens); fclose(f); return NULL;
    }
    fclose(f);

    nt_dataloader* dl = (nt_dataloader*)calloc(1, sizeof(nt_dataloader));
    if (!dl) { free(tokens); return NULL; }
    dl->tokens = tokens;
    dl->n_tokens = n_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->n_batches = (n_tokens - 1) / (seq_len * batch_size);
    if (dl->n_batches <= 0) dl->n_batches = 1;
    dl->shuffle_indices = (int*)malloc(dl->n_batches * sizeof(int));
    for (int i = 0; i < dl->n_batches; i++) dl->shuffle_indices[i] = i;
    return dl;
}

int nt_dataloader_next(nt_dataloader* dl, int* input, int* target) {
    if (!dl || !input || !target) return -1;
    if (dl->batch_idx >= dl->n_batches) {
        dl->epoch++;
        dl->batch_idx = 0;
        nt_dataloader_shuffle(dl);
        return -1;
    }

    int batch_start = dl->shuffle_indices[dl->batch_idx] * dl->seq_len * dl->batch_size;
    for (int b = 0; b < dl->batch_size; b++) {
        int offset = batch_start + b * dl->seq_len;
        for (int s = 0; s < dl->seq_len; s++) {
            int pos = offset + s;
            if (pos + 1 >= dl->n_tokens) pos = dl->n_tokens - 2;
            input[b * dl->seq_len + s] = dl->tokens[pos];
            target[b * dl->seq_len + s] = dl->tokens[pos + 1];
        }
    }
    dl->batch_idx++;
    return 0;
}

void nt_dataloader_reset(nt_dataloader* dl) {
    if (!dl) return;
    dl->batch_idx = 0;
    dl->pos = 0;
}

void nt_dataloader_shuffle(nt_dataloader* dl) {
    if (!dl || !dl->shuffle_indices) return;
    for (int i = dl->n_batches - 1; i > 0; i--) {
        int j = xorshift32() % (i + 1);
        int tmp = dl->shuffle_indices[i];
        dl->shuffle_indices[i] = dl->shuffle_indices[j];
        dl->shuffle_indices[j] = tmp;
    }
}

void nt_dataloader_free(nt_dataloader* dl) {
    if (!dl) return;
    free(dl->tokens);
    free(dl->shuffle_indices);
    free(dl);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAVE / LOAD
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_MAGIC 0x4E544F52  // "NTOR"

int nt_save(const char* path, nt_tensor** params, int n_params) {
    if (!path || !params || n_params <= 0) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = NT_MAGIC;
    int32_t n = n_params;
    fwrite(&magic, 4, 1, f);
    fwrite(&n, 4, 1, f);
    for (int i = 0; i < n_params; i++) {
        nt_tensor* t = params[i];
        int32_t ndim = t->ndim;
        fwrite(&ndim, 4, 1, f);
        for (int d = 0; d < ndim; d++) {
            int32_t s = t->shape[d];
            fwrite(&s, 4, 1, f);
        }
        fwrite(t->data, sizeof(float), t->len, f);
    }
    fclose(f);
    return 0;
}

nt_tensor** nt_load(const char* path, int* n_params) {
    if (!path || !n_params) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    int32_t n;
    if (fread(&magic, 4, 1, f) != 1 || magic != NT_MAGIC) { fclose(f); return NULL; }
    if (fread(&n, 4, 1, f) != 1 || n <= 0 || n > NT_TAPE_MAX_PARAMS) { fclose(f); return NULL; }

    nt_tensor** params = (nt_tensor**)calloc(n, sizeof(nt_tensor*));
    if (!params) { fclose(f); return NULL; }

    for (int i = 0; i < n; i++) {
        int32_t ndim;
        if (fread(&ndim, 4, 1, f) != 1 ||
            ndim < 0 || ndim > NT_MAX_DIMS) { fclose(f); *n_params = i; return params; }
        int shape[NT_MAX_DIMS];
        for (int d = 0; d < ndim; d++) {
            int32_t s;
            if (fread(&s, 4, 1, f) != 1) { fclose(f); *n_params = i; return params; }
            shape[d] = s;
        }
        params[i] = nt_tensor_new_shape(shape, ndim);
        if (!params[i]) { fclose(f); *n_params = i; return params; }
        /* A truncated payload leaves a tensor of calloc-zeros that looks valid to
         * the caller — drop it and report the count that actually loaded. */
        if (fread(params[i]->data, sizeof(float), (size_t)params[i]->len, f)
                != (size_t)params[i]->len) {
            nt_tensor_free(params[i]);
            params[i] = NULL;
            fclose(f); *n_params = i; return params;
        }
    }
    fclose(f);
    *n_params = n;
    return params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEBBIAN MICROLEARNING
// ═══════════════════════════════════════════════════════════════════════════════

void nt_hebbian_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal,
                     float lr, float decay) {
    if (!A || !B || !x || !dy) return;
    // A: [in_dim × rank], B: [rank × out_dim]
    // Hebbian: A += lr * signal * x ⊗ (B^T @ dy), B += lr * signal * (A^T @ x) ⊗ dy
    float* proj = (float*)calloc(rank, sizeof(float));
    if (!proj) return;

    // proj = B^T @ dy (rank vector)
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rank, out_dim,
                1.0f, B, out_dim, dy, 1, 0.0f, proj, 1);
#else
    for (int r = 0; r < rank; r++) {
        float s = 0;
        for (int j = 0; j < out_dim; j++) s += B[r * out_dim + j] * dy[j];
        proj[r] = s;
    }
#endif

    // A update: A[i*rank+r] += lr * signal * x[i] * proj[r]
    float alpha = lr * signal;
#ifdef USE_BLAS
    cblas_sger(CblasRowMajor, in_dim, rank,
               alpha, x, 1, proj, 1, A, rank);
#else
    for (int i = 0; i < in_dim; i++)
        for (int r = 0; r < rank; r++)
            A[i * rank + r] += alpha * x[i] * proj[r];
#endif

    // proj2 = A^T @ x (rank vector)
    float* proj2 = (float*)calloc(rank, sizeof(float));
    if (proj2) {
#ifdef USE_BLAS
        cblas_sgemv(CblasRowMajor, CblasTrans, in_dim, rank,
                    1.0f, A, rank, x, 1, 0.0f, proj2, 1);
#else
        for (int r = 0; r < rank; r++) {
            float s = 0;
            for (int i = 0; i < in_dim; i++) s += A[i * rank + r] * x[i];
            proj2[r] = s;
        }
#endif
        // B update: B[r*out_dim+j] += lr * signal * proj2[r] * dy[j]
#ifdef USE_BLAS
        cblas_sger(CblasRowMajor, rank, out_dim,
                   alpha, proj2, 1, dy, 1, B, out_dim);
#else
        for (int r = 0; r < rank; r++)
            for (int j = 0; j < out_dim; j++)
                B[r * out_dim + j] += alpha * proj2[r] * dy[j];
#endif
        free(proj2);
    }

    // Weight decay
    if (decay > 0.0f && decay < 1.0f) {
        for (int i = 0; i < in_dim * rank; i++) A[i] *= decay;
        for (int i = 0; i < rank * out_dim; i++) B[i] *= decay;
    }
    free(proj);
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

long nt_count_params(nt_tensor** params, int n) {
    long total = 0;
    for (int i = 0; i < n; i++)
        if (params[i]) total += params[i]->len;
    return total;
}

void nt_print_params(nt_tensor** params, int n, const char** names) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        if (!params[i]) continue;
        const char* name = (names && names[i]) ? names[i] : "param";
        nt_tensor_print(params[i], name);
        total += params[i]->len;
    }
    printf("Total: %ld parameters (%.2f MB)\n", total, (float)total * 4.0f / 1048576.0f);
}

/* BPE implementation is above, near dataloader */

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS — direct matmul API for inference engines
// ═══════════════════════════════════════════════════════════════════════════════

void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, BT, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * BT[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
#endif
}

void nt_blas_matvec(float *out, const float *W, const float *x, int m, int n) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                m, n, 1.0f, W, n, x, 1, 0.0f, out, 1);
#else
    for (int i = 0; i < m; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) s += W[i*n + j] * x[j];
        out[i] = s;
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// PACKED QUANTIZED MATVEC — out[m] = Wq[m,k] @ x[k], weights stay packed
// ═══════════════════════════════════════════════════════════════════════════════
//
// The CPU/BLAS path dequantizes a whole GGUF tensor to dense f32 (×6-8 RAM) before
// cblas_sgemv. nt_qmatvec keeps the weights packed in RAM and dequantizes each block
// inline in registers — same math as gguf_dequant -> nt_blas_matvec, a fraction of
// the memory and weight bandwidth. dtype = GGUF type code. Phase 1: Q4_0,
// single-threaded. Mirrors the packed q6k_rows pattern in
// examples/infer_gguf_metal.c and dequant_q4_0 in gguf.c.

// IEEE half -> float (GGUF block scales are stored as f16).
/* Every packed kernel converts one f16 scale per block, so this sits in the innermost loop
 * of every matvec in the library: a 0.5B decode calls it about eleven million times per
 * token. The portable version below is a branch, a shift chain and a loop for subnormals —
 * twenty-odd instructions next to the two SDOTs it accompanies. aarch64 has had the
 * conversion in hardware since armv8: one FCVT, same IEEE result. */
#if defined(__aarch64__)
static inline float nt_f16_to_f32(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, sizeof(v));
    return (float)v;
}
#else
static float nt_f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
               bits = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1F) bits = (s << 31) | (0xFFu << 23) | (m << 13);
    else bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
#endif

// Q4_0: 18 B/block, 32 vals — f16 scale + 16 bytes of (lo,hi) nibbles, each (-8).
static void nt_q4_0_rows(float *out, const uint8_t *W, const float *x,
                         int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 18;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 18;
            float d = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            const float *xb = x + (long)blk * 32;
            for (int i = 0; i < 16; i++) {
                int lo = (int)(b[2 + i] & 0x0F) - 8;
                int hi = (int)(b[2 + i] >> 4)   - 8;
                acc += d * (float)lo * xb[i];
                acc += d * (float)hi * xb[i + 16];
            }
        }
        out[row] = acc;
    }
}

// Q8_0: 34 B/block, 32 vals — f16 scale + 32 int8.
static void nt_q8_0_rows(float *out, const uint8_t *W, const float *x,
                         int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 34;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 34;
            float d = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            const float *xb = x + (long)blk * 32;
            for (int i = 0; i < 32; i++)
                acc += d * (float)(int8_t)b[2 + i] * xb[i];
        }
        out[row] = acc;
    }
}

// Q5_0: 22 B/block, 32 vals — f16 scale + 4 B high-bit word + 16 nibble bytes
// (the 5th bit of each value comes from the high-bit word).
static void nt_q5_0_rows(float *out, const uint8_t *W, const float *x,
                         int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 22;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 22;
            float d = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) |
                          ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
            const uint8_t *qs = b + 6;
            const float *xb = x + (long)blk * 32;
            for (int j = 0; j < 16; j++) {
                int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
                int hb0 = (qh >> j) & 1, hb1 = (qh >> (j + 16)) & 1;
                acc += d * (float)((lo | (hb0 << 4)) - 16) * xb[j];
                acc += d * (float)((hi | (hb1 << 4)) - 16) * xb[j + 16];
            }
        }
        out[row] = acc;
    }
}

// ── super-block formats (256 vals/block) ────────────────────────────────────
// Q4_K 6-bit packed scale/min unpack (matches gguf.c:get_scale_min_k4).
static void nt_get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *mn) {
    if (j < 4) { *s = sc[j] & 63; *mn = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *mn = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4); }
}

/* One Q4_K sub-block's contribution to the running float, in one place so that every kernel
 * that computes it — per token, batched, SMMLA — emits the same instruction sequence. Spelled
 * out at each site instead, the compiler is free to contract d*ls*dot - dmin*lm*asum into a
 * multiply-subtract in one kernel and leave it as two operations in another; both are correct
 * and they differ in the last bit, which is exactly what a test comparing bits will report. */
/* Q6_K's sub-block term, fused for the same reason as Q4_K's: the product chain and the
 * accumulate that follows it are contractible, so if they are left as expressions the kernels
 * are free to disagree in the last bit. */
static inline float nt_q6k_acc(float acc, float d, float sc, float da, int32_t dot) {
    return __builtin_fmaf(d * sc * da, (float)dot, acc);
}

static inline float nt_q4k_acc(float acc, float da, float d, float ls, int32_t dot,
                               float dmin, float lm, int32_t asum) {
    /* Both fused operations are spelled out rather than left to the compiler, and the
     * accumulate is inside the helper for the same reason as the subtract. Written as
     * ordinary expressions, `d*ls*dot - dmin*lm*asum` and the `acc +=` that consumes it
     * contract into fused instructions at some call sites and not at others — that alone
     * made the SDOT and SMMLA kernels disagree in the last bit, and a test that compares
     * bits is right to call that a failure. Naming the operations removes the choice:
     * every kernel rounds where these two lines say it rounds. */
    return __builtin_fmaf(da, __builtin_fmaf(d * ls, (float)dot,
                                             -(dmin * lm * (float)asum)), acc);
}

// Q4_K: 144 B/block, 256 vals — d, dmin (f16) + 12 B packed scales/mins + 128 nibbles.
static void nt_q4_k_rows(float *out, const uint8_t *W, const float *x,
                         int r0, int r1, int k) {
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            const float *xb = x + (long)blk * 256;
            int is = 0, qi = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                nt_get_scale_min_k4(is,     sc, &sc0, &m0);
                nt_get_scale_min_k4(is + 1, sc, &sc1, &m1);
                float d1 = d * sc0, mm1 = dmin * m0, d2 = d * sc1, mm2 = dmin * m1;
                for (int l = 0; l < 32; l++)
                    acc += (d1 * (float)(qs[qi + l] & 0x0F) - mm1) * xb[j + l];
                for (int l = 0; l < 32; l++)
                    acc += (d2 * (float)(qs[qi + l] >> 4)   - mm2) * xb[j + 32 + l];
                qi += 32; is += 2;
            }
        }
        out[row] = acc;
    }
}

// Q6_K: 210 B/block, 256 vals — ql[128] qh[64] int8 scales[16] + f16 d.
// Lifted from the proven packed q6k_rows in examples/infer_gguf_metal.c.
//
// AVX2/FMA path (Colibri T5c): the scalar unpack costs ~3.3 cycles/weight, which on a
// 151936x2048 lm_head is ~82 ms/token on 6 cores — the head, not memory, becomes the
// bottleneck (measured: 4.65 t/s with the scalar kernel vs 7.53 with an AVX2 int8 head,
// same box, same input, CPU 553%). Eight values per lane, four sub-groups accumulated
// separately and folded by their int8 sub-scale once per 16-value group, so the scale
// application is identical to the scalar order. The scalar body is kept verbatim under
// #else for ARM and non-AVX2 x86.
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline float nt_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo); lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);  lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
#endif

static void nt_q6_k_rows(float *out, const uint8_t *W, const float *x,
                         int r0, int r1, int k) {
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t *)(b + 192);
            float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
            const float *xb = x + (long)blk * 256;
#if defined(__AVX2__) && defined(__FMA__)
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                const int8_t *sch = sc + (n / 128) * 8;
                const __m256i m4 = _mm256_set1_epi32(0x0F), m3 = _mm256_set1_epi32(3),
                              b32 = _mm256_set1_epi32(32);
                for (int is = 0; is < 2; is++) {          /* is = l/16, 16 values per sub-scale */
                    __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
                           a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps();
                    for (int l = is * 16; l < is * 16 + 16; l += 8) {
                        __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qlh + l)));
                        __m256i hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qlh + l + 32)));
                        __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qhh + l)));
                        __m256i q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(lo, m4),
                                     _mm256_slli_epi32(_mm256_and_si256(hb, m3), 4)), b32);
                        __m256i q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(hi, m4),
                                     _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2), m3), 4)), b32);
                        __m256i q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(lo, 4),
                                     _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 4), m3), 4)), b32);
                        __m256i q4 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(hi, 4),
                                     _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 6), m3), 4)), b32);
                        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1), _mm256_loadu_ps(xb + n + l),      a1);
                        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q2), _mm256_loadu_ps(xb + n + l + 32), a2);
                        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q3), _mm256_loadu_ps(xb + n + l + 64), a3);
                        a4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q4), _mm256_loadu_ps(xb + n + l + 96), a4);
                    }
                    acc += d * ((float)sch[is + 0] * nt_hsum256_ps(a1)
                              + (float)sch[is + 2] * nt_hsum256_ps(a2)
                              + (float)sch[is + 4] * nt_hsum256_ps(a3)
                              + (float)sch[is + 6] * nt_hsum256_ps(a4));
                }
            }
#else
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                const int8_t *sch = sc + (n / 128) * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * sch[is + 0] * q1 * xb[n + l];
                    acc += d * sch[is + 2] * q2 * xb[n + l + 32];
                    acc += d * sch[is + 4] * q3 * xb[n + l + 64];
                    acc += d * sch[is + 6] * q4 * xb[n + l + 96];
                }
            }
#endif
        }
        out[row] = acc;
    }
}

// F16: contiguous half weights — converted per element. Keeps weights at 2 B/param
// (half the RAM of dense f32) without ever materializing a full f32 tensor.
static void nt_f16_rows(float *out, const uint8_t *W, const float *x,
                        int r0, int r1, int k) {
    const uint16_t *Wh = (const uint16_t *)W;
    for (int row = r0; row < r1; row++) {
        const uint16_t *r = Wh + (long)row * k;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += nt_f16_to_f32(r[j]) * x[j];
        out[row] = acc;
    }
}

// F32 dense dot as a range kernel, so the agnostic entry threads like the rest.
static void nt_f32_rows(float *out, const uint8_t *W, const float *x,
                        int r0, int r1, int k) {
    const float *Wf = (const float *)W;
    for (int row = r0; row < r1; row++) {
        const float *r = Wf + (long)row * k;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += r[j] * x[j];
        out[row] = acc;
    }
}

typedef void (*nt_qrows_fn)(float *, const uint8_t *, const float *, int, int, int);

// Map a GGUF dtype to its packed row-kernel, or NULL if unsupported / bad shape.
static nt_qrows_fn nt_qrows_for(int dtype, int k) {
    switch (dtype) {
    case 0:  return nt_f32_rows;                          /* F32  */
    case 1:  return nt_f16_rows;                          /* F16  */
    case 2:  return (k % 32)  ? NULL : nt_q4_0_rows;      /* Q4_0 */
    case 6:  return (k % 32)  ? NULL : nt_q5_0_rows;      /* Q5_0 */
    case 8:  return (k % 32)  ? NULL : nt_q8_0_rows;      /* Q8_0 */
    case 12: return (k % 256) ? NULL : nt_q4_k_rows;      /* Q4_K */
    case 14: return (k % 256) ? NULL : nt_q6_k_rows;      /* Q6_K */
    default: return NULL;
    }
}

#define NT_QMV_MAX_THREADS 16
#define NT_QMV_ASUM_MAX   2048   /* k <= 65536: activation-sum scratch stays on the stack */

/* Threading floor, shared by both packed matvecs. The API wins over the environment: a
 * consumer that knows its own shapes should not have to export a variable to be fast. */
/* 64K elements, not the 4M this used to be. The old number was right for the pool it was
 * measured against: a dispatch cost tens of microseconds, so anything under a few million
 * elements lost more to the fan-out than it gained. With the atomic, spin-first pool a
 * dispatch costs about five microseconds and the arithmetic inverts — an Exynos 1580 decode
 * of Qwen2.5-0.5B reads 27.3 t/s at the old floor and 35.0 at this one, and everything below
 * 64K measures the same, so this is where the curve flattens rather than a guess. Callers
 * that know their shapes still override through the API, and NT_QMV_THREAD_MIN through the
 * environment. */
#define NT_QMV_THREAD_MIN_DEFAULT (64L << 10)
static long g_qmv_thread_min = -1;
void nt_qmv_set_thread_min(long elems) {
    __atomic_store_n(&g_qmv_thread_min, (elems > 0) ? elems : NT_QMV_THREAD_MIN_DEFAULT,
                     __ATOMIC_RELAXED);
}
/* The plan is built further down, after the cpufreq helpers it needs; this is the one field
 * of it a function up here has to read. */
static long nt_qmv_plan_thread_min(void);

/* Atomic rather than a plain lazy store: this one has a public setter, so a caller can write
 * it while another thread is reading it, and the default has to arrive without a race either.
 * Compare-and-exchange so an explicit nt_qmv_set_thread_min still wins over the environment
 * regardless of which happens first. */
static long nt_qmv_thread_floor(void) {
    long v = __atomic_load_n(&g_qmv_thread_min, __ATOMIC_RELAXED);
    if (v < 0) {
        long expect = -1, from_env = nt_qmv_plan_thread_min();
        __atomic_compare_exchange_n(&g_qmv_thread_min, &expect, from_env, 0,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        v = __atomic_load_n(&g_qmv_thread_min, __ATOMIC_RELAXED);
    }
    return v;
}

/* _SC_NPROCESSORS_ONLN counts the cores the KERNEL has online, not the cores THIS process
 * is allowed to run on. Every big.LITTLE benchmark pins to the fast cluster — `taskset 0xF0`
 * on an 8-core phone — and there the old count returned 8 while four cores were usable, so
 * the pool oversubscribed 2:1 and each matvec ended up waiting on a context switch instead
 * of on memory. The affinity mask is the honest number. NT_QMV_THREADS overrides both, which
 * is what an A/B run needs and what a caller that already owns a thread budget wants. */
#if defined(__linux__)
/* Reported peak kHz of one core, or 0 where cpufreq is not exported. */
static long nt_cpu_peak_khz(int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long v = 0;
    if (fscanf(f, "%ld", &v) != 1) v = 0;
    fclose(f);
    return v;
}

/* Everything in `in` except its slowest class, and how many that is. Returns 0 when the
 * machine is uniform, when cpufreq says nothing, or when fewer than two cores would survive
 * — all three mean there is no useful choice to make here.
 *
 * sched_getaffinity answers "how many cores may I use", which on a phone is the wrong
 * question. Gemma 4 E2B on this SoC, decode, by core set: cpu7 alone 6.0 t/s, cpu6-7 5.2,
 * cpu5-7 8.3, cpu4-7 10.2, all eight 7.9, the four small ones 3.4. Adding a single small
 * core to the big four costs a fifth of the throughput (10.2 -> 8.3), because decode runs at
 * the memory ceiling already and the extra core brings contention and a long tail rather
 * than arithmetic.
 *
 * Note it is the slowest class that goes, not everything below the fastest. Peak clocks here
 * are 1.95, 2.6 and 2.91 GHz across three classes; keeping only the fastest leaves the one
 * prime core and measures 6.0, while dropping only the slowest leaves cpu4-7 and measures
 * 10.2. Written after that mistake was built and benchmarked, not before. */
static int nt_cpu_perf_set(const cpu_set_t *in, cpu_set_t *out) {
    long slowest = 0, fastest = 0;
    int seen = 0, n = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, in)) continue;
        long f = nt_cpu_peak_khz(c);
        if (f <= 0) return 0;
        if (!seen || f < slowest) slowest = f;
        if (f > fastest) fastest = f;
        seen++;
    }
    if (!seen || slowest == fastest) return 0;
    CPU_ZERO(out);
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, in) && nt_cpu_peak_khz(c) > slowest) { CPU_SET(c, out); n++; }
    /* One core is not a fan-out, and on this hardware it measures worse than the whole
     * machine. A shape that lopsided is more likely an unfamiliar topology than an
     * opportunity, so leave it to the scheduler. */
    return n >= 2 ? n : 0;
}

/* One thread per core, in order, rather than one mask shared by all of them.
 *
 * Leaving placement inside the mask to the scheduler is not the neutral choice it looks like.
 * A thread that alternates spinning with parking on a condvar reads as half idle, so two of
 * them fit on one core by the scheduler's arithmetic, and the decision is taken once and kept
 * for the whole run. Measured on taskset -c 5,6, six runs of Gemma 4 decode: 8.7, 5.1, 5.8,
 * 4.6, 6.9, 8.8 t/s — two plateaus, not a spread. Sampling /proc/<pid>/task through each run
 * showed the fast ones with the two threads accumulating time on cpu5 and cpu6, and the slow
 * ones with both on cpu5, the second thread getting a quarter of the first one's cycles. The
 * ratio between the plateaus is 1.9, which is what one core doing the work of two looks like.
 *
 * Honouring somebody else's mask means honouring which cores they gave us. It does not mean
 * declining to use all of them. NT_QMV_PIN=0 turns this off. */
#endif  /* __linux__ — the cpufreq helpers above */

/* Everything the fan-out decides once: how many threads, which cores, whether to pin, how
 * finely to divide the rows. It used to be four functions each caching into its own
 * function-scope static with a lazy `if (x < 0)` — which is a data race, not a shortcut. This
 * library has two pools behind two separate pthread_once guards, so a program calling the
 * float matvec and the integer one from different threads runs both initialisers at the same
 * time and both of them touch those statics. One pthread_once over one struct removes the
 * whole class rather than the four instances of it. */
typedef struct {
    int  threads;           /* fan-out width before the per-call clamp */
    int  chunks;            /* chunks handed to each worker */
    int  pin;               /* place threads on cores ourselves */
    int  pool;              /* reuse persistent workers instead of spawning per call */
    int  spin;              /* how long a worker spins before parking */
    long thread_min;        /* elements below which the fan-out is not worth it */
#if defined(__linux__)
    int  ncpu;              /* cores in `cpus`, 0 when there is nothing to place on */
    cpu_set_t cpus;
#endif
} nt_qmv_plan;

static nt_qmv_plan g_nt_qmv_plan;
static pthread_once_t g_nt_qmv_plan_once = PTHREAD_ONCE_INIT;
/* Published with release ordering when the plan is built, read with acquire on every access.
 * pthread_once is correct but it is a call into libpthread, and the plan is read four times
 * per matvec — of which there are a couple of hundred per token. Measured with the once on
 * the hot path: 10.5-10.6 t/s against 10.7 without, plus an occasional 9.3. One pointer load
 * costs nothing and the ordering is what makes it safe rather than merely fast. */
static const nt_qmv_plan *g_nt_qmv_plan_ready = NULL;

static void nt_qmv_plan_init(void) {
    nt_qmv_plan *p = &g_nt_qmv_plan;
    const char *e;

    e = getenv("NT_QMV_PIN");
    p->pin = !(e && e[0] == '0');

    /* Granularity trades load balance against per-chunk cost, and the balance point moves
     * whenever either side does. It moved when the pool started pinning one thread per core:
     * with placement settled, what remains is the tail, because the big cores are not equally
     * fast either — a thread on this SoC's prime core runs at 1.6-2.0 GHz while its
     * neighbours hold 2.6, the governor's energy model rather than a thermal cap, and nothing
     * in cpuinfo_max_freq says so. Coarse chunks leave that core holding the last one while
     * everybody waits. Gemma 4 E2B decode on four cores, chunks per worker against t/s:
     * 1 -> 9.9, 2 -> 10.3, 4 -> 10.5, 8 -> 10.6, 16 -> 10.7, 32 -> 10.7. Qwen 2.5 0.5B: 55.9,
     * 52.9, 54.4, 55.1, 55.9, 55.3. Sixteen is best or tied on both, and at sixteen four cores
     * match the three that exclude the prime core, so the fix is granularity rather than
     * discarding a core. The float pool reached sixteen independently. */
    e = getenv("NT_QMV_CHUNKS");
    long v = e ? atol(e) : 0;
    p->chunks = (v >= 1 && v <= 64) ? (int)v : 16;

    e = getenv("NT_QMV_POOL");
    p->pool = !(e && (!strcmp(e, "0") || !strcmp(e, "false") ||
                      !strcmp(e, "off") || !strcmp(e, "no")));

    /* How long a worker keeps looking before it parks on the condvar. Between two matvecs of
     * one token the gap is the scalar work — norms, rope, softmax, quantizing the activation
     * — and some of those gaps are longer than a short budget covers, so the worker parks and
     * has to be woken through a futex, once per worker per gap. Raising the budget past the
     * long gaps removes those wakes. Gemma 4 E2B and Qwen 2.5 0.5B decode, four cores:
     * 20000 -> 10.5 / 52.7 t/s, 200000 -> 10.9 / 56.1, 500000 -> 11.1 / 56.5,
     * 1000000 -> 11.1 / 56.1, 4000000 -> 11.1 / 57.6. Five to seven percent, flat past the
     * knee at half a million.
     *
     * Spinning does not cost what it looks like it costs. Measured with the CPU seconds beside
     * the wall clock: 11.6 cpu-s at 20000 against 11.3 at 500000 for the same work, because
     * parking and waking spends more of them than looking does. What it does cost is a core
     * held for roughly ten milliseconds after the last dispatch before the worker gives up —
     * free for continuous decoding, a small drain for a process that runs one matvec and
     * waits. NT_QMV_SPIN=0 parks immediately, which is also the only way the condvar path
     * gets exercised. */
    e = getenv("NT_QMV_SPIN");
    p->spin = (e && atoi(e) >= 0) ? atoi(e) : 500000;

    e = getenv("NT_QMV_THREAD_MIN");
    p->thread_min = (e && atol(e) > 0) ? atol(e) : NT_QMV_THREAD_MIN_DEFAULT;

    e = getenv("NT_QMV_THREADS");
    long want = e ? atol(e) : 0;
    p->threads = want > 0 ? (int)want : 0;

#if defined(__linux__)
    p->ncpu = 0;
    CPU_ZERO(&p->cpus);
    cpu_set_t mine;
    CPU_ZERO(&mine);
    if (sched_getaffinity(0, sizeof(mine), &mine) == 0) {
        int n = 0;
        /* Narrowing to a core class is off when somebody already decided: an explicit thread
         * count, a mask narrower than the machine, or NT_QMV_BIG_ONLY=0. */
        const char *off = getenv("NT_QMV_BIG_ONLY");
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (!(off && off[0] == '0') && !p->threads &&
            online > 0 && CPU_COUNT(&mine) == (int)online)
            n = nt_cpu_perf_set(&mine, &p->cpus);
        if (n <= 0) { p->cpus = mine; n = CPU_COUNT(&mine); }
        p->ncpu = n;
        if (!p->threads) p->threads = n;
    }
#endif
    if (p->threads < 1) p->threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (p->threads < 1) p->threads = 1;
    __atomic_store_n(&g_nt_qmv_plan_ready, p, __ATOMIC_RELEASE);
}

/* Split so the fast path is a load and a branch the compiler will inline. Leaving the
 * pthread_once call inside the accessor makes the whole thing uninlinable, and the accessor
 * sits in nt_qmatvec_i8 next to the shape checks: keeping it out of line cost 10.5-10.6 t/s
 * against 10.7 on Gemma, twelve runs each, which is small and was not noise. */
static const nt_qmv_plan *nt_qmv_plan_slow(void) {
    pthread_once(&g_nt_qmv_plan_once, nt_qmv_plan_init);
    return __atomic_load_n(&g_nt_qmv_plan_ready, __ATOMIC_ACQUIRE);
}

static inline const nt_qmv_plan *nt_qmv_get_plan(void) {
    const nt_qmv_plan *p = __atomic_load_n(&g_nt_qmv_plan_ready, __ATOMIC_ACQUIRE);
    return p ? p : nt_qmv_plan_slow();
}

static long nt_qmv_plan_thread_min(void) { return nt_qmv_get_plan()->thread_min; }

static void nt_qmv_pin_nth(pthread_t t, int idx) {
#if defined(__linux__)
    const nt_qmv_plan *p = nt_qmv_get_plan();
    if (!p->pin || p->ncpu <= 0) return;
    int want = idx % p->ncpu, seen = 0;
    cpu_set_t one;
    CPU_ZERO(&one);
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &p->cpus) && seen++ == want) { CPU_SET(c, &one); break; }
    if (CPU_COUNT(&one)) pthread_setaffinity_np(t, sizeof(one), &one);
#else
    (void)t; (void)idx;
#endif
}

static int nt_qmv_host_threads(int m) {
    int nt = nt_qmv_get_plan()->threads;
    if (nt > NT_QMV_MAX_THREADS) nt = NT_QMV_MAX_THREADS;
    if (nt > m) nt = m;
    return nt;
}

typedef struct {
    nt_qrows_fn fn; float *out; const uint8_t *Wq; const float *x;
    int r0, r1, k;
} nt_qjob;

#ifndef _OPENMP   /* only the pthread fan-out uses a worker entry point */
static int nt_qmv_pool_enabled(void) { return nt_qmv_get_plan()->pool; }

static void *nt_qworker(void *p) {
    nt_qjob *j = (nt_qjob *)p;
    j->fn(j->out, j->Wq, j->x, j->r0, j->r1, j->k);
    return NULL;
}

// Persistent qmatvec workers remove pthread_create/join from every decode matvec.
// The caller computes the last shard inline; workers handle the earlier shards.
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    pthread_t threads[NT_QMV_MAX_THREADS];
    int ids[NT_QMV_MAX_THREADS];
    int nthreads;
    int ready;
    int shutdown;
    long generation;
    int active;
    int done;
    nt_qjob jobs[NT_QMV_MAX_THREADS];
    nt_qjob shared;              /* fn/out/Wq/x/k common to every chunk */
    int lo, hi, chunk, next;     /* the range the workers drain */
} nt_qpool;

static nt_qpool g_nt_qpool = {
    PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_COND_INITIALIZER,
    PTHREAD_COND_INITIALIZER,
    {0},
    {0},
    0,
    0,
    0,
    0,
    0,
    0,
    {{0}},
    {0},              /* shared           */
    0, 0, 0, 0,       /* lo hi chunk next */
};
static pthread_once_t g_nt_qpool_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_nt_qpool_dispatch_mu = PTHREAD_MUTEX_INITIALIZER;

// Rows are handed out on demand rather than split up front. The split assumed every
// worker retires its share in the same wall time, which is false on any asymmetric CPU:
// on an Exynos 1580 a Cortex-A520 spends about three times as long per row as the prime
// A720, so an equal share leaves the fast cores idle waiting for the slow one. Measured
// on a 32000x2048 Q4_K head, 8.29 ms split evenly across all eight cores against 3.41 ms
// once the rows were claimed in chunks.
//
// The jobs array still arrives as a static split, and is still used verbatim by the
// pthread fallback below. The pool takes the RANGE it describes and ignores the division:
// [jobs[0].r0, jobs[n-1].r1) with a chunk size, and every worker plus the caller drains
// from a shared cursor. Rows are disjoint and each row's accumulation is self-contained,
// so which worker claims which chunk cannot move a bit of the result — a symmetric machine
// sees the same ranges it always did, one cursor step apart.
//
// About 16 chunks per worker. Coarser loses the balance this exists for; finer was swept
// on the same device and lost to lock traffic.
static void nt_qpool_drain(void) {
    for (;;) {
        pthread_mutex_lock(&g_nt_qpool.mu);
        int r0 = g_nt_qpool.next, hi = g_nt_qpool.hi, ch = g_nt_qpool.chunk;
        nt_qjob j = g_nt_qpool.shared;
        if (r0 < hi) g_nt_qpool.next = r0 + ch;
        pthread_mutex_unlock(&g_nt_qpool.mu);
        if (r0 >= hi) return;
        int r1 = r0 + ch; if (r1 > hi) r1 = hi;
        j.fn(j.out, j.Wq, j.x, r0, r1, j.k);
    }
}

static void *nt_qpool_loop(void *p) {
    int id = *(int *)p;
    long seen = 0;
    pthread_mutex_lock(&g_nt_qpool.mu);
    for (;;) {
        while (!g_nt_qpool.shutdown && g_nt_qpool.generation == seen)
            pthread_cond_wait(&g_nt_qpool.cv_work, &g_nt_qpool.mu);
        if (g_nt_qpool.shutdown) break;

        seen = g_nt_qpool.generation;
        int has_job = id < g_nt_qpool.active;
        pthread_mutex_unlock(&g_nt_qpool.mu);

        if (has_job) nt_qpool_drain();

        pthread_mutex_lock(&g_nt_qpool.mu);
        if (has_job) {
            g_nt_qpool.done++;
            if (g_nt_qpool.done >= g_nt_qpool.active)
                pthread_cond_signal(&g_nt_qpool.cv_done);
        }
    }
    pthread_mutex_unlock(&g_nt_qpool.mu);
    return NULL;
}

static void nt_qpool_shutdown(void) {
    pthread_mutex_lock(&g_nt_qpool.mu);
    g_nt_qpool.shutdown = 1;
    g_nt_qpool.generation++;
    pthread_cond_broadcast(&g_nt_qpool.cv_work);
    pthread_mutex_unlock(&g_nt_qpool.mu);
    for (int i = 0; i < g_nt_qpool.nthreads; i++)
        pthread_join(g_nt_qpool.threads[i], NULL);
}

static void nt_qpool_init_once(void) {
    int nt = nt_qmv_host_threads(NT_QMV_MAX_THREADS);
    nt_qmv_pin_nth(pthread_self(), 0);
    for (int i = 0; i < nt; i++) {
        g_nt_qpool.ids[i] = i;
        if (pthread_create(&g_nt_qpool.threads[i], NULL, nt_qpool_loop, &g_nt_qpool.ids[i]) != 0)
            break;
        nt_qmv_pin_nth(g_nt_qpool.threads[i], i + 1);
        g_nt_qpool.nthreads++;
    }
    g_nt_qpool.ready = g_nt_qpool.nthreads > 0;
    if (g_nt_qpool.ready) atexit(nt_qpool_shutdown);
}

static int nt_qpool_run(const nt_qjob *jobs, int nt) {
    if (!nt_qmv_pool_enabled()) return -1;
    pthread_once(&g_nt_qpool_once, nt_qpool_init_once);
    if (!g_nt_qpool.ready) return -1;
    int worker_nt = nt - 1;
    if (worker_nt <= 0 || worker_nt > g_nt_qpool.nthreads) return -1;

    int lo = jobs[0].r0, hi = jobs[worker_nt].r1;
    int chunk = (hi - lo) / (nt * 16); if (chunk < 1) chunk = 1;

    pthread_mutex_lock(&g_nt_qpool_dispatch_mu);
    pthread_mutex_lock(&g_nt_qpool.mu);
    g_nt_qpool.shared = jobs[0];
    g_nt_qpool.lo = lo; g_nt_qpool.hi = hi; g_nt_qpool.chunk = chunk;
    g_nt_qpool.next = lo;
    g_nt_qpool.active = worker_nt;
    g_nt_qpool.done = 0;
    g_nt_qpool.generation++;
    pthread_cond_broadcast(&g_nt_qpool.cv_work);
    pthread_mutex_unlock(&g_nt_qpool.mu);

    nt_qpool_drain();                    /* the caller is a worker too */

    pthread_mutex_lock(&g_nt_qpool.mu);
    while (g_nt_qpool.done < g_nt_qpool.active)
        pthread_cond_wait(&g_nt_qpool.cv_done, &g_nt_qpool.mu);
    pthread_mutex_unlock(&g_nt_qpool.mu);
    pthread_mutex_unlock(&g_nt_qpool_dispatch_mu);
    return 0;
}
#endif

// Packed quantized matvec, parallelized across rows (rows are independent and
// write disjoint out[]). dtype = GGUF type code. Returns 0 ok, -1 if the dtype
// has no packed kernel yet (caller falls back to gguf_dequant -> nt_blas_matvec).
int nt_qmatvec(float *out, const uint8_t *Wq, int dtype,
               const float *x, int m, int k) {
    nt_qrows_fn fn = nt_qrows_for(dtype, k);
    if (!fn) return -1;

    int nt = nt_qmv_host_threads(m);
    // Thread fan-out and the 2P+4E asymmetry of Apple-Silicon-class CPUs make small
    // single-token decode matvecs counterproductive even when workers are persistent.
    // Gate it high: only large matvecs (big models / batched work) thread; small
    // decode stays single-thread.
    /* The 4M floor was measured on a 360M-class decoder, where fan-out was noise.
     * Other shapes exist: a 500M decoder's matrices are 2.46M and sit just under
     * it, so its whole decode stays single-threaded. Default is unchanged;
     * NT_QMV_THREAD_MIN lets a consumer set the floor for its own shape after
     * measuring (the eye engine runs at 256K: 3.7 -> 7.3 tok/s, same output). */
    if (nt <= 1 || (long)m * k < nt_qmv_thread_floor()) { fn(out, Wq, x, 0, m, k); return 0; }

#ifdef _OPENMP
    /* When the consumer is an OpenMP program, private pthreads are actively harmful, not
     * merely redundant: libgomp parks its idle team in a SPIN wait by default, so six
     * spinning OpenMP threads and six pthreads land on six cores and fight. Measured on a
     * 151936x2048 head inside such an engine: 23.19 ms/tok with private pthreads against
     * 12.47 ms once the spinning stopped — the kernel was never the problem, the
     * oversubscription was. Reusing the caller's team removes the cause instead of asking
     * every consumer to remember OMP_WAIT_POLICY=passive. Row ranges are identical to the
     * pthread split, so results are bit-identical. */
    int per_omp = (m + nt - 1) / nt;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int r0 = t * per_omp, r1 = (r0 + per_omp > m) ? m : r0 + per_omp;
        if (r0 < m) fn(out, Wq, x, r0, r1, k);
    }
    return 0;
#else
    pthread_t th[NT_QMV_MAX_THREADS];
    nt_qjob   jobs[NT_QMV_MAX_THREADS];
    int per = (m + nt - 1) / nt, launched = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per, r1 = (r0 + per > m) ? m : r0 + per;
        if (r0 >= m) break;
        jobs[t] = (nt_qjob){ fn, out, Wq, x, r0, r1, k };
        launched++;
    }
    if (nt_qpool_run(jobs, launched) == 0) return 0;

    launched = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per;
        if (r0 >= m) break;
        if (pthread_create(&th[t], NULL, nt_qworker, &jobs[t]) != 0) {
            fn(out, Wq, x, r0, m, k);   // create failed: run the rest inline
            break;
        }
        launched++;
    }
    for (int t = 0; t < launched; t++) pthread_join(th[t], NULL);
    return 0;
#endif
}

// ── int8 dynamic-activation-quant matvec (the llama.cpp / MNN fast path) ─────────
// Quantize the activation to per-32-block symmetric int8 once, then dot it against
// the packed int4/int8 weights with INTEGER accumulation. APPROXIMATE: int8
// activation quant trades a little accuracy for speed; nt_qmatvec (f32 dequant) is
// the exact reference. Phase 2b: Q4_0, scalar (SDOT/VNNI + more dtypes next).

// x[k] -> per-32-block symmetric int8: qa[k] (int8) + da[k/32] (block scales).
// ── Weight quantization — f32 rows into packed blocks ───────────────────────────
// The inverse of the dequantizers above, and the step notorch did not have: a model came
// out of training as f32 and had to be handed to llama.cpp to become a file this library
// could run fast. The arithmetic below is llama.cpp's reference quantizer to the bit, so
// that what comes out is not "our format" but the same GGUF everything else reads.
//
// Two details are easy to get subtly wrong and both change every block. The 4- and 5-bit
// formats scale by the SIGNED extreme — the element with the largest magnitude, keeping its
// sign — and divide by a negative bound (-8, -16), which is what makes the unsigned nibble
// land biased rather than centred. And the rounding is a +8.5 / +16.5 offset with a
// truncating cast, not a round-to-nearest: for the negative half those differ.
static float nt_f32_to_f16_round(float f, uint16_t *out) {
#if defined(__aarch64__)
    __fp16 h = (__fp16)f;                    /* round-to-nearest-even in hardware */
    memcpy(out, &h, sizeof(h));
    return (float)h;
#else
    /* Round-to-nearest-even in software, matching ggml's fallback. */
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    uint16_t h;
    if (exp >= 0x1F) h = (uint16_t)(sign | 0x7C00);
    else if (exp <= 0) h = (uint16_t)sign;
    else {
        h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
        uint32_t rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    }
    *out = h;
    return nt_f16_to_f32(h);
#endif
}

// ── K-quants: the block formats' quantizers do not generalise ──────────────────
// Q4_0 and its relatives take an absmax and are done. A K-quant does not: each 16- or
// 32-value sub-block gets its own scale found by SEARCH — try a spread of candidate scales,
// keep the one with the lowest weighted error — and then those per-sub-block scales are
// themselves quantized to six bits against a super-block scale. That search is the format;
// an absmax approximation of it produces a legal file that is a different model.
//
// The algorithms below are ggml's reference quantizers (llama.cpp, MIT, Georgi Gerganov and
// contributors), ported rather than reinvented, because bit-compatibility with what everyone
// else writes is the entire point of doing this at all. The gate in tests/test_quantize.c is
// a byte comparison against llama-quantize's output; nothing here is a judgement call.
//
// nearest_int is ggml's, and it is not roundf: adding 2^23 + 2^22 forces the mantissa to
// drop everything below the units place with round-to-nearest-even, then the exponent bits
// are masked away. roundf rounds halves away from zero, which differs on exactly the ties
// this search lands on.
static inline int nt_nearest_int(float fval) {
    float val = fval + 12582912.0f;
    int32_t i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007FFFFF) - 0x00400000;
}

/* Every float product here is guarded the same way the block quantizers are: the compiler
 * may fuse a multiply into the add that follows it, which moves the last bit, which moves a
 * nearest_int across a tie, which picks a different level. Cold code, once per model. */
static inline float nt_noc(float v) { volatile float t = v; return t; }

#define NT_GROUP_MAX_EPS 1e-15f

/* Q6_K's per-sub-block scale: start from -nmax/max, then walk 18 nearby scales and keep the
 * one whose weighted reconstruction wins. rmse_type 1 means the weight is x*x. */
static float nt_make_qx_quants(int n, int nmax, const float *x, int8_t *L) {
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < NT_GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = -(float)nmax / max;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nt_nearest_int(nt_noc(iscale * x[i]));
        if (l < -nmax) l = -nmax;
        if (l > nmax - 1) l = nmax - 1;
        L[i] = (int8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += nt_noc(w * x[i] * (float)l);
        suml2 += nt_noc(w * (float)l * (float)l);
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        float isc = -((float)nmax + 0.1f * (float)is) / max;
        float slx = 0.0f, sl2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nt_nearest_int(nt_noc(isc * x[i]));
            if (l < -nmax) l = -nmax;
            if (l > nmax - 1) l = nmax - 1;
            float w = x[i] * x[i];
            slx += nt_noc(w * x[i] * (float)l);
            sl2 += nt_noc(w * (float)l * (float)l);
        }
        if (sl2 > 0.0f && slx * slx > best * sl2) {
            for (int i = 0; i < n; ++i) {
                int l = nt_nearest_int(nt_noc(isc * x[i]));
                if (l < -nmax) l = -nmax;
                if (l > nmax - 1) l = nmax - 1;
                L[i] = (int8_t)(l + nmax);
            }
            scale = slx / sl2;
            best = scale * slx;
        }
    }
    return scale;
}

/* Q4_K's per-sub-block affine fit: a scale AND a minimum, found by stepping the scale over
 * nstep candidates and solving the weighted least squares for each. */
static float nt_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                                 uint8_t *L, float *the_min, uint8_t *Laux,
                                 float rmin, float rdelta, int nstep) {
    float min = x[0], max = x[0];
    float sum_w = weights[0];
    float sum_x = nt_noc(sum_w * x[0]);
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += nt_noc(w * x[i]);
    }
    if (min > 0.0f) min = 0.0f;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.0f;
    }
    float iscale = (float)nmax / (max - min);
    float scale = 1.0f / iscale;
    float best_mad = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nt_nearest_int(nt_noc(iscale * (x[i] - min)));
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        float diff = nt_noc(scale * (float)L[i]) + min - x[i];
        best_mad += nt_noc(weights[i] * diff * diff);
    }
    for (int is = 0; is <= nstep; ++is) {
        float isc = (rmin + rdelta * (float)is + (float)nmax) / (max - min);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nt_nearest_int(nt_noc(isc * (x[i] - min)));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l  += nt_noc(w * (float)l);
            sum_l2 += nt_noc(w * (float)l * (float)l);
            sum_xl += nt_noc(w * (float)l * x[i]);
        }
        float D = nt_noc(sum_w * sum_l2) - nt_noc(sum_l * sum_l);
        if (D > 0.0f) {
            float this_scale = (nt_noc(sum_w * sum_xl) - nt_noc(sum_x * sum_l)) / D;
            float this_min   = (nt_noc(sum_l2 * sum_x) - nt_noc(sum_l * sum_xl)) / D;
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0.0f;
            for (int i = 0; i < n; ++i) {
                float diff = nt_noc(this_scale * (float)Laux[i]) + this_min - x[i];
                mad += nt_noc(weights[i] * diff * diff);
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static void nt_quantize_row_q6_k(const float *x, uint8_t *out, int k) {
    int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const float *xb = x + (long)i * 256;
        uint8_t *blk = out + (long)i * 210;
        int8_t L[256];
        float scales[16];
        float max_scale = 0.0f, max_abs_scale = 0.0f;

        for (int ib = 0; ib < 16; ++ib) {
            float s = nt_make_qx_quants(16, 32, xb + 16 * ib, L + 16 * ib);
            scales[ib] = s;
            float as = fabsf(s);
            if (as > max_abs_scale) { max_abs_scale = as; max_scale = s; }
        }

        if (max_abs_scale < NT_GROUP_MAX_EPS) { memset(blk, 0, 210); continue; }

        float iscale = -128.0f / max_scale;
        uint16_t dh; nt_f32_to_f16_round(1.0f / iscale, &dh);
        int8_t sc[16];
        for (int ib = 0; ib < 16; ++ib) {
            int v = nt_nearest_int(nt_noc(iscale * scales[ib]));
            sc[ib] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < 16; ++j) {
            float d = nt_f16_to_f32(dh) * (float)sc[j];
            if (d == 0.0f) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nt_nearest_int(nt_noc(xb[16 * j + ii] / d));
                if (l < -32) l = -32;
                if (l > 31) l = 31;
                L[16 * j + ii] = (int8_t)(l + 32);
            }
        }

        uint8_t *ql = blk, *qh = blk + 128;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                uint8_t q1 = (uint8_t)L[j + l +  0] & 0xF;
                uint8_t q2 = (uint8_t)L[j + l + 32] & 0xF;
                uint8_t q3 = (uint8_t)L[j + l + 64] & 0xF;
                uint8_t q4 = (uint8_t)L[j + l + 96] & 0xF;
                ql[l +  0] = (uint8_t)(q1 | (q3 << 4));
                ql[l + 32] = (uint8_t)(q2 | (q4 << 4));
                qh[l] = (uint8_t)(((uint8_t)L[j + l] >> 4)
                                | (((uint8_t)L[j + l + 32] >> 4) << 2)
                                | (((uint8_t)L[j + l + 64] >> 4) << 4)
                                | (((uint8_t)L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
        memcpy(blk + 192, sc, 16);
        blk[208] = (uint8_t)(dh & 0xFF);
        blk[209] = (uint8_t)(dh >> 8);
    }
}

static void nt_quantize_row_q4_k(const float *x, uint8_t *out, int k) {
    int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const float *xb = x + (long)i * 256;
        uint8_t *blk = out + (long)i * 144;
        uint8_t L[256], Laux[32];
        float weights[32], mins[8], scales[8];
        float max_scale = 0.0f, max_min = 0.0f;

        for (int j = 0; j < 8; ++j) {
            float sum_x2 = 0.0f;
            for (int l = 0; l < 32; ++l) sum_x2 += nt_noc(xb[32 * j + l] * xb[32 * j + l]);
            float av_x = sqrtf(sum_x2 / 32.0f);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(xb[32 * j + l]);
            scales[j] = nt_make_qkx2_quants(32, 15, xb + 32 * j, weights, L + 32 * j,
                                            &mins[j], Laux, -1.0f, 0.1f, 20);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        uint8_t *scbytes = blk + 4;
        memset(scbytes, 0, 12);
        float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        float inv_min   = max_min   > 0.0f ? 63.0f / max_min   : 0.0f;
        for (int j = 0; j < 8; ++j) {
            int lsv = nt_nearest_int(nt_noc(inv_scale * scales[j]));
            int lmv = nt_nearest_int(nt_noc(inv_min * mins[j]));
            uint8_t ls = (uint8_t)(lsv < 63 ? lsv : 63);
            uint8_t lm = (uint8_t)(lmv < 63 ? lmv : 63);
            if (j < 4) {
                scbytes[j] = ls;
                scbytes[j + 4] = lm;
            } else {
                scbytes[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
                scbytes[j - 4] |= (uint8_t)((ls >> 4) << 6);
                scbytes[j - 0] |= (uint8_t)((lm >> 4) << 6);
            }
        }
        uint16_t dh, dmh;
        nt_f32_to_f16_round(max_scale / 63.0f, &dh);
        nt_f32_to_f16_round(max_min / 63.0f, &dmh);
        blk[0] = (uint8_t)(dh & 0xFF);  blk[1] = (uint8_t)(dh >> 8);
        blk[2] = (uint8_t)(dmh & 0xFF); blk[3] = (uint8_t)(dmh >> 8);

        for (int j = 0; j < 8; ++j) {
            uint8_t s6, m6;
            nt_get_scale_min_k4(j, scbytes, &s6, &m6);
            float d = nt_f16_to_f32(dh) * (float)s6;
            if (d == 0.0f) continue;
            float dm = nt_f16_to_f32(dmh) * (float)m6;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nt_nearest_int(nt_noc((xb[32 * j + ii] + dm) / d));
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                L[32 * j + ii] = (uint8_t)l;
            }
        }

        uint8_t *q = blk + 16;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
    }
}

int nt_quantize_row(const float *x, void *dst, int k, int dtype) {
    if (!x || !dst || k <= 0 || (k % 32)) return -1;
    if (dtype == 12 || dtype == 14) {
        if (k % 256) return -1;
        if (dtype == 12) nt_quantize_row_q4_k(x, (uint8_t *)dst, k);
        else             nt_quantize_row_q6_k(x, (uint8_t *)dst, k);
        return 0;
    }
    int nb = k / 32;
    uint8_t *out = (uint8_t *)dst;

    for (int b = 0; b < nb; b++) {
        const float *xb = x + (long)b * 32;
        if (dtype == 8) {                                    /* Q8_0: 2 + 32 bytes */
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
            float d = amax / 127.0f;
            uint16_t dh; nt_f32_to_f16_round(d, &dh);
            float id = (d != 0.0f) ? 1.0f / d : 0.0f;
            uint8_t *blk = out + (long)b * 34;
            blk[0] = (uint8_t)(dh & 0xFF); blk[1] = (uint8_t)(dh >> 8);
            for (int i = 0; i < 32; i++) {
                int q = (int)roundf(xb[i] * id);
                if (q > 127) q = 127; else if (q < -128) q = -128;
                blk[2 + i] = (uint8_t)(int8_t)q;
            }
        } else if (dtype == 2 || dtype == 6) {               /* Q4_0: 18 B, Q5_0: 22 B */
            float amax = 0.0f, vmax = 0.0f;
            for (int i = 0; i < 32; i++) {
                float v = xb[i], a = fabsf(v);
                if (a > amax) { amax = a; vmax = v; }
            }
            int bound = (dtype == 2) ? 8 : 16;
            float d = vmax / (float)(-bound);
            uint16_t dh; nt_f32_to_f16_round(d, &dh);
            float id = (d != 0.0f) ? 1.0f / d : 0.0f;
            float half = (float)bound + 0.5f;
            if (dtype == 2) {
                uint8_t *blk = out + (long)b * 18;
                blk[0] = (uint8_t)(dh & 0xFF); blk[1] = (uint8_t)(dh >> 8);
                for (int j = 0; j < 16; j++) {
                    /* volatile, and the reason is compatibility rather than paranoia. The
                     * reference computes the scaled value, rounds it to a float, and only
                     * then adds the offset and truncates. Left as one expression the
                     * compiler fuses the multiply and the add, which changes the last bit,
                     * which flips the truncation for any value sitting near an integer —
                     * 143 tensors of 291 differed from llama-quantize's output for exactly
                     * this reason, by one level, in one nibble. Quantization runs once per
                     * model, so a store and a load per value is not a price worth arguing
                     * about; agreeing with every other GGUF reader is. */
                    volatile float p0 = xb[j] * id, p1 = xb[j + 16] * id;
                    float v0 = p0 + half, v1 = p1 + half;
                    int q0 = (int)v0, q1 = (int)v1;
                    if (q0 > 15) q0 = 15;
                    if (q0 < 0)  q0 = 0;
                    if (q1 > 15) q1 = 15;
                    if (q1 < 0)  q1 = 0;
                    blk[2 + j] = (uint8_t)(q0 | (q1 << 4));
                }
            } else {
                uint8_t *blk = out + (long)b * 22;
                blk[0] = (uint8_t)(dh & 0xFF); blk[1] = (uint8_t)(dh >> 8);
                uint32_t qh = 0;
                for (int j = 0; j < 16; j++) {
                    volatile float p0 = xb[j] * id, p1 = xb[j + 16] * id;   /* see Q4_0 above */
                    float v0 = p0 + half, v1 = p1 + half;
                    int q0 = (int)v0, q1 = (int)v1;
                    if (q0 > 31) q0 = 31;
                    if (q0 < 0)  q0 = 0;
                    if (q1 > 31) q1 = 31;
                    if (q1 < 0)  q1 = 0;
                    blk[6 + j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    qh |= ((uint32_t)(q0 & 0x10) >> 4) << j;
                    qh |= ((uint32_t)(q1 & 0x10) >> 4) << (j + 16);
                }
                blk[2] = (uint8_t)(qh & 0xFF);         blk[3] = (uint8_t)((qh >> 8) & 0xFF);
                blk[4] = (uint8_t)((qh >> 16) & 0xFF); blk[5] = (uint8_t)((qh >> 24) & 0xFF);
            }
        } else {
            return -1;
        }
    }
    return 0;
}

/* SUM(qa) per 32-value block, which is what Q5_0's -16 bias and Q4_K's affine minimum lift
 * out of the integer dot. NEON where SDOT exists — a dot against a vector of ones is exactly
 * the sum, and the pairwise adds come free — and a plain loop elsewhere. Either way the
 * result is an exact integer, so the two agree bit for bit and the kernels do not care which
 * one ran. */
static void nt_act_block_sums(const int8_t *qa, int k, int32_t *asum) {
    int nb = k / 32;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t one8 = vdupq_n_s8(1);
    for (int b = 0; b < nb; b++) {
        const int8_t *p = qa + (long)b * 32;
        int32x4_t t = vdotq_s32(vdupq_n_s32(0), one8, vld1q_s8(p));
        t = vdotq_s32(t, one8, vld1q_s8(p + 16));
        asum[b] = vaddvq_s32(t);
    }
#else
    for (int b = 0; b < nb; b++) {
        const int8_t *p = qa + (long)b * 32;
        int32_t t = 0;
        for (int i = 0; i < 32; i++) t += p[i];
        asum[b] = t;
    }
#endif
}

static void nt_quant_act_q8(const float *x, int k, int8_t *qa, float *da) {
    int nb = k / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (long)b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        float d  = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        da[b] = d;
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(xb[i] * id);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            qa[(long)b * 32 + i] = (int8_t)q;
        }
    }
}

// Q4_0 int8-dot rows: packed weights (18 B/32) × pre-quantized int8 activation.
// Block layout (per dequant_q4_0): byte i holds elem i (low nibble) and elem i+16
// (high nibble), each value = nibble - 8. So lo nibbles pair with qa[0..15], hi with
// qa[16..31]. Integer accumulation; per-block result scaled by d_w * d_a.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
static void nt_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 32;
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const int8x16_t  eight  = vdupq_n_s8(8);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 18;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 18;
            float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int8_t *qab = qa + (long)b * 32;
            uint8x16_t packed = vld1q_u8(blk + 2);                        // 16 nibble-bytes
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask0f)), eight);  // elems 0..15
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), eight);     // elems 16..31
            int8x16_t qlo = vld1q_s8(qab);                                // qa[0..15]
            int8x16_t qhi = vld1q_s8(qab + 16);                           // qa[16..31]
            int32x4_t s4 = vdupq_n_s32(0);
            s4 = vdotq_s32(s4, lo, qlo);                                  // 16 int8-MAC
            s4 = vdotq_s32(s4, hi, qhi);                                  // 16 int8-MAC
            acc += d_w * da[b] * (float)vaddvq_s32(s4);                   // horizontal sum
        }
        out[row] = acc;
    }
}
#else
static void nt_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 18;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 18;
            float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int8_t *qab = qa + (long)b * 32;
            int32_t s = 0;
            for (int i = 0; i < 16; i++) {
                int lo = (int)(blk[2 + i] & 0x0F) - 8;
                int hi = (int)(blk[2 + i] >> 4)   - 8;
                s += lo * qab[i];
                s += hi * qab[i + 16];
            }
            acc += d_w * da[b] * (float)s;
        }
        out[row] = acc;
    }
}
#endif

// Q8_0 int8-dot rows: packed weights (34 B/32) × pre-quantized int8 activation.
// Block layout (per dequant_q8_0): 2 B f16 scale, then 32 raw int8 weights — the
// weights are already integers, so unlike Q4_0 there is nothing to unpack: the
// dot is int8 x int8 straight through, per-block result scaled by d_w * d_a.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void nt_q8_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 34;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 34;
            float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int8_t *wq  = (const int8_t *)(blk + 2);
            const int8_t *qab = qa + (long)b * 32;
            int32x4_t s4 = vdupq_n_s32(0);
            s4 = vdotq_s32(s4, vld1q_s8(wq),      vld1q_s8(qab));         // elems 0..15
            s4 = vdotq_s32(s4, vld1q_s8(wq + 16), vld1q_s8(qab + 16));    // elems 16..31
            acc += d_w * da[b] * (float)vaddvq_s32(s4);
        }
        out[row] = acc;
    }
}
#else
static void nt_q8_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 34;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 34;
            float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int8_t *wq  = (const int8_t *)(blk + 2);
            const int8_t *qab = qa + (long)b * 32;
            int32_t s = 0;
            for (int i = 0; i < 32; i++) s += (int32_t)wq[i] * (int32_t)qab[i];
            acc += d_w * da[b] * (float)s;
        }
        out[row] = acc;
    }
}
#endif

// Q5_0 int8-dot rows: 22 B/32 vals — an f16 scale, a 32-bit mask carrying one high bit
// per value, then 16 nibble-bytes. The value is (nibble | high<<4) - 16.
//
// Two properties keep this cheap. The reconstructed q is [0,31], representable as signed
// int8, so SDOT applies with no sign handling — unlike the Q4_0 path there is nothing to
// bias into range first. And the -16 lifts out of the dot the way Q4_K's minimum does:
// SUM((q-16)*x) is SUM(q*x) - 16*SUM(x), so the per-block activation sum is computed once
// per call and the subtraction never touches a vector lane.
//
// The high bits need no lookup table, which is the usual approach. Broadcasting a mask
// byte across eight lanes and testing it against the powers of two expands one bit per
// lane in a single vtstq_u8; masking with 0x10 puts each bit straight into position four,
// where the nibble expects it. Block bytes 2-3 cover lanes 0-15 and bytes 4-5 cover 16-31,
// the same split the nibble halves already use.
// byte -> eight lanes, each 0x10 where the bit is set. Two loads from this replace the
// vdup/vcombine/vtst/and quartet the high-bit expansion used to cost per half-block. The
// table is 2 KB and stays resident; entries are pre-shifted to position four, which is
// where the nibble expects the bit, so no shift is needed after the load.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static const uint64_t nt_q5_hi[256] = {
    0x0000000000000000ULL, 0x0000000000000010ULL, 0x0000000000001000ULL, 0x0000000000001010ULL,
    0x0000000000100000ULL, 0x0000000000100010ULL, 0x0000000000101000ULL, 0x0000000000101010ULL,
    0x0000000010000000ULL, 0x0000000010000010ULL, 0x0000000010001000ULL, 0x0000000010001010ULL,
    0x0000000010100000ULL, 0x0000000010100010ULL, 0x0000000010101000ULL, 0x0000000010101010ULL,
    0x0000001000000000ULL, 0x0000001000000010ULL, 0x0000001000001000ULL, 0x0000001000001010ULL,
    0x0000001000100000ULL, 0x0000001000100010ULL, 0x0000001000101000ULL, 0x0000001000101010ULL,
    0x0000001010000000ULL, 0x0000001010000010ULL, 0x0000001010001000ULL, 0x0000001010001010ULL,
    0x0000001010100000ULL, 0x0000001010100010ULL, 0x0000001010101000ULL, 0x0000001010101010ULL,
    0x0000100000000000ULL, 0x0000100000000010ULL, 0x0000100000001000ULL, 0x0000100000001010ULL,
    0x0000100000100000ULL, 0x0000100000100010ULL, 0x0000100000101000ULL, 0x0000100000101010ULL,
    0x0000100010000000ULL, 0x0000100010000010ULL, 0x0000100010001000ULL, 0x0000100010001010ULL,
    0x0000100010100000ULL, 0x0000100010100010ULL, 0x0000100010101000ULL, 0x0000100010101010ULL,
    0x0000101000000000ULL, 0x0000101000000010ULL, 0x0000101000001000ULL, 0x0000101000001010ULL,
    0x0000101000100000ULL, 0x0000101000100010ULL, 0x0000101000101000ULL, 0x0000101000101010ULL,
    0x0000101010000000ULL, 0x0000101010000010ULL, 0x0000101010001000ULL, 0x0000101010001010ULL,
    0x0000101010100000ULL, 0x0000101010100010ULL, 0x0000101010101000ULL, 0x0000101010101010ULL,
    0x0010000000000000ULL, 0x0010000000000010ULL, 0x0010000000001000ULL, 0x0010000000001010ULL,
    0x0010000000100000ULL, 0x0010000000100010ULL, 0x0010000000101000ULL, 0x0010000000101010ULL,
    0x0010000010000000ULL, 0x0010000010000010ULL, 0x0010000010001000ULL, 0x0010000010001010ULL,
    0x0010000010100000ULL, 0x0010000010100010ULL, 0x0010000010101000ULL, 0x0010000010101010ULL,
    0x0010001000000000ULL, 0x0010001000000010ULL, 0x0010001000001000ULL, 0x0010001000001010ULL,
    0x0010001000100000ULL, 0x0010001000100010ULL, 0x0010001000101000ULL, 0x0010001000101010ULL,
    0x0010001010000000ULL, 0x0010001010000010ULL, 0x0010001010001000ULL, 0x0010001010001010ULL,
    0x0010001010100000ULL, 0x0010001010100010ULL, 0x0010001010101000ULL, 0x0010001010101010ULL,
    0x0010100000000000ULL, 0x0010100000000010ULL, 0x0010100000001000ULL, 0x0010100000001010ULL,
    0x0010100000100000ULL, 0x0010100000100010ULL, 0x0010100000101000ULL, 0x0010100000101010ULL,
    0x0010100010000000ULL, 0x0010100010000010ULL, 0x0010100010001000ULL, 0x0010100010001010ULL,
    0x0010100010100000ULL, 0x0010100010100010ULL, 0x0010100010101000ULL, 0x0010100010101010ULL,
    0x0010101000000000ULL, 0x0010101000000010ULL, 0x0010101000001000ULL, 0x0010101000001010ULL,
    0x0010101000100000ULL, 0x0010101000100010ULL, 0x0010101000101000ULL, 0x0010101000101010ULL,
    0x0010101010000000ULL, 0x0010101010000010ULL, 0x0010101010001000ULL, 0x0010101010001010ULL,
    0x0010101010100000ULL, 0x0010101010100010ULL, 0x0010101010101000ULL, 0x0010101010101010ULL,
    0x1000000000000000ULL, 0x1000000000000010ULL, 0x1000000000001000ULL, 0x1000000000001010ULL,
    0x1000000000100000ULL, 0x1000000000100010ULL, 0x1000000000101000ULL, 0x1000000000101010ULL,
    0x1000000010000000ULL, 0x1000000010000010ULL, 0x1000000010001000ULL, 0x1000000010001010ULL,
    0x1000000010100000ULL, 0x1000000010100010ULL, 0x1000000010101000ULL, 0x1000000010101010ULL,
    0x1000001000000000ULL, 0x1000001000000010ULL, 0x1000001000001000ULL, 0x1000001000001010ULL,
    0x1000001000100000ULL, 0x1000001000100010ULL, 0x1000001000101000ULL, 0x1000001000101010ULL,
    0x1000001010000000ULL, 0x1000001010000010ULL, 0x1000001010001000ULL, 0x1000001010001010ULL,
    0x1000001010100000ULL, 0x1000001010100010ULL, 0x1000001010101000ULL, 0x1000001010101010ULL,
    0x1000100000000000ULL, 0x1000100000000010ULL, 0x1000100000001000ULL, 0x1000100000001010ULL,
    0x1000100000100000ULL, 0x1000100000100010ULL, 0x1000100000101000ULL, 0x1000100000101010ULL,
    0x1000100010000000ULL, 0x1000100010000010ULL, 0x1000100010001000ULL, 0x1000100010001010ULL,
    0x1000100010100000ULL, 0x1000100010100010ULL, 0x1000100010101000ULL, 0x1000100010101010ULL,
    0x1000101000000000ULL, 0x1000101000000010ULL, 0x1000101000001000ULL, 0x1000101000001010ULL,
    0x1000101000100000ULL, 0x1000101000100010ULL, 0x1000101000101000ULL, 0x1000101000101010ULL,
    0x1000101010000000ULL, 0x1000101010000010ULL, 0x1000101010001000ULL, 0x1000101010001010ULL,
    0x1000101010100000ULL, 0x1000101010100010ULL, 0x1000101010101000ULL, 0x1000101010101010ULL,
    0x1010000000000000ULL, 0x1010000000000010ULL, 0x1010000000001000ULL, 0x1010000000001010ULL,
    0x1010000000100000ULL, 0x1010000000100010ULL, 0x1010000000101000ULL, 0x1010000000101010ULL,
    0x1010000010000000ULL, 0x1010000010000010ULL, 0x1010000010001000ULL, 0x1010000010001010ULL,
    0x1010000010100000ULL, 0x1010000010100010ULL, 0x1010000010101000ULL, 0x1010000010101010ULL,
    0x1010001000000000ULL, 0x1010001000000010ULL, 0x1010001000001000ULL, 0x1010001000001010ULL,
    0x1010001000100000ULL, 0x1010001000100010ULL, 0x1010001000101000ULL, 0x1010001000101010ULL,
    0x1010001010000000ULL, 0x1010001010000010ULL, 0x1010001010001000ULL, 0x1010001010001010ULL,
    0x1010001010100000ULL, 0x1010001010100010ULL, 0x1010001010101000ULL, 0x1010001010101010ULL,
    0x1010100000000000ULL, 0x1010100000000010ULL, 0x1010100000001000ULL, 0x1010100000001010ULL,
    0x1010100000100000ULL, 0x1010100000100010ULL, 0x1010100000101000ULL, 0x1010100000101010ULL,
    0x1010100010000000ULL, 0x1010100010000010ULL, 0x1010100010001000ULL, 0x1010100010001010ULL,
    0x1010100010100000ULL, 0x1010100010100010ULL, 0x1010100010101000ULL, 0x1010100010101010ULL,
    0x1010101000000000ULL, 0x1010101000000010ULL, 0x1010101000001000ULL, 0x1010101000001010ULL,
    0x1010101000100000ULL, 0x1010101000100010ULL, 0x1010101000101000ULL, 0x1010101000101010ULL,
    0x1010101010000000ULL, 0x1010101010000010ULL, 0x1010101010001000ULL, 0x1010101010001010ULL,
    0x1010101010100000ULL, 0x1010101010100010ULL, 0x1010101010101000ULL, 0x1010101010101010ULL,
};
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void nt_q5_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    int nb = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    // Four blocks retired together: vaddvq_s32 is a full horizontal reduction in the
    // dependency chain, and four of them collapse into two pairwise vpaddq_s32. The float
    // accumulation order is unchanged, block by block ascending, so this stays exact.
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 22;
        float acc = 0.0f; int b = 0;
        for (; b + 4 <= nb; b += 4) {
            int32x4_t s0, s1, s2, s3; float dv[4]; int32x4_t *sp[4] = { &s0, &s1, &s2, &s3 };
            for (int j = 0; j < 4; j++) {
                const uint8_t *blk = rb + (long)(b + j) * 22;
                dv[j] = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                uint8x16_t h0 = vreinterpretq_u8_u64(vcombine_u64(
                    vcreate_u64(nt_q5_hi[blk[2]]), vcreate_u64(nt_q5_hi[blk[3]])));
                uint8x16_t h1 = vreinterpretq_u8_u64(vcombine_u64(
                    vcreate_u64(nt_q5_hi[blk[4]]), vcreate_u64(nt_q5_hi[blk[5]])));
                uint8x16_t pk = vld1q_u8(blk + 6);
                int8x16_t lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(pk, m4), h0));
                int8x16_t hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(pk, 4), h1));
                const int8_t *qab = qa + (long)(b + j) * 32;
                int32x4_t t = vdupq_n_s32(0);
                t = vdotq_s32(t, lo, vld1q_s8(qab));
                t = vdotq_s32(t, hi, vld1q_s8(qab + 16));
                *sp[j] = t;
            }
            int32_t sums[4]; vst1q_s32(sums, vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3)));
            acc += dv[0] * da[b+0] * (float)(sums[0] - 16 * asum[b+0]);
            acc += dv[1] * da[b+1] * (float)(sums[1] - 16 * asum[b+1]);
            acc += dv[2] * da[b+2] * (float)(sums[2] - 16 * asum[b+2]);
            acc += dv[3] * da[b+3] * (float)(sums[3] - 16 * asum[b+3]);
        }
        for (; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 22;
            float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            uint8x16_t h0 = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(nt_q5_hi[blk[2]]), vcreate_u64(nt_q5_hi[blk[3]])));
            uint8x16_t h1 = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(nt_q5_hi[blk[4]]), vcreate_u64(nt_q5_hi[blk[5]])));
            uint8x16_t pk = vld1q_u8(blk + 6);
            int8x16_t lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(pk, m4), h0));
            int8x16_t hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(pk, 4), h1));
            const int8_t *qab = qa + (long)b * 32;
            int32x4_t t = vdupq_n_s32(0);
            t = vdotq_s32(t, lo, vld1q_s8(qab));
            t = vdotq_s32(t, hi, vld1q_s8(qab + 16));
            acc += d * da[b] * (float)(vaddvq_s32(t) - 16 * asum[b]);
        }
        out[row] = acc;
    }
}
#else
static void nt_q5_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 22;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 22;
            float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                        | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
            const uint8_t *qs = blk + 6;
            const int8_t *qab = qa + (long)b * 32;
            int32_t s = 0;
            for (int j = 0; j < 16; j++) {
                int q0 = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
                int q1 = (qs[j] >> 4)   | (((qh >> (j + 16)) & 1) << 4);
                s += q0 * qab[j]; s += q1 * qab[j + 16];
            }
            acc += d * da[b] * (float)(s - 16 * asum[b]);
        }
        out[row] = acc;
    }
}
#endif

// Q6_K int8-dot rows: 210 B/256 vals against the per-32 int8 activation.
// The two block grids line up, which is what makes this path exact-by-subblock: a weight
// sub-scale covers 16 values, an activation block covers 32, and 16j..16j+15 always sits
// inside activation block j/2 — never straddles. So the integer accumulator is per weight
// sub-block, and d * sc[j] * da[j/2] is applied once after it, never per value.
// A group of 32 consecutive positions therefore shares ONE activation scale and spans
// exactly TWO sub-scales; _mm256_maddubs_epi16 splits along 128-bit lanes, i.e. exactly on
// that 16/16 boundary, so one instruction covers the group and its halves fall out already
// separated. Sign trick as in the Q4_0 path: |w| is unsigned-safe because Q6 lands in
// [-32,31], and |w|*|x| <= 32*127, two of them still clear int16.
#if defined(__AVX2__) && defined(__FMA__)
static void nt_q6_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 256;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3),
                  b32 = _mm256_set1_epi8(32), ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t *)(b + 192);
            float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
            const int8_t *qab = qa + (long)blk * 256;
            const float  *dab = da + (long)blk * 8;
            /* Same lane treatment the Q4_K path got. This kernel drained TWICE per group —
             * four dependent hadds and two extracts, sixteen drains per 256-value block —
             * because each group carries two 16-value sub-blocks, one per 128-bit half.
             * That is exactly what a hadd tree resolves for free: hadd works within halves,
             * so folding four groups yields the four lower sub-block sums in lanes 0-3 and
             * the four upper ones in lanes 4-7 of a single vector. Two trees cover a block.
             * Accumulation order is preserved group by group, lower sub-block then upper,
             * so this is an integer re-order and the greedy vector must not move. */
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                __m256i qhv = _mm256_loadu_si256((const __m256i *)qhh);
                __m256i s[4];
                for (int g = 0; g < 4; g++) {
                    __m256i qlv = _mm256_loadu_si256((const __m256i *)(qlh + (g & 1) * 32));
                    __m256i lo  = (g < 2) ? _mm256_and_si256(qlv, m4)
                                          : _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4);
                    __m256i hi2 = _mm256_and_si256(_mm256_srli_epi16(qhv, 2 * g), m3);
                    __m256i w   = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi2, 4)), b32);
                    __m256i xv  = _mm256_loadu_si256((const __m256i *)(qab + n + g * 32));
                    s[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w),
                                                                  _mm256_sign_epi8(xv, w)), ones);
                }
                /* lanes 0-3: lower sub-block of groups 0..3; lanes 4-7: their upper one */
                __m256i T = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]),
                                              _mm256_hadd_epi32(s[2], s[3]));
                int32_t t[8];
                _mm256_storeu_si256((__m256i *)t, T);
                for (int g = 0; g < 4; g++) {
                    int j0 = n / 16 + g * 2;
                    acc += d * dab[(n + g * 32) / 32]
                         * ((float)sc[j0]     * (float)t[g]
                          + (float)sc[j0 + 1] * (float)t[g + 4]);
                }
            }
        }
        out[row] = acc;
    }
}
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
// The 16/32 seam that costs AVX2 a lane argument costs NEON nothing: a Q6 sub-scale
// covers 16 values and SDOT consumes exactly 16 int8 lanes, so one sub-block is one
// vdotq_s32 and ssum[j] is written once rather than accumulated. Q6 lands in [-32,31]
// after the -32 bias, which is signed int8, so SDOT applies directly — none of the
// unsigned/sign-flip choreography _mm256_maddubs_epi16 forces on the x86 path.
// The float tail below is character-for-character the scalar one: the integer dots are
// exact, so keeping the same expression and the same ascending j order makes this kernel
// bit-identical to the fallback it replaces, and that is what the micro-bench asserts.
static void nt_q6_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3);
    const int8x16_t  b32 = vdupq_n_s8(32);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t *)(b + 192);
            float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
            const int8_t *qab = qa + (long)blk * 256;
            const float  *dab = da + (long)blk * 8;
            int32_t ssum[16];
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                int base = (n / 128) * 8;
                for (int is = 0; is < 2; is++) {
                    uint8x16_t la = vld1q_u8(qlh + is * 16);        /* elems l      */
                    uint8x16_t lb = vld1q_u8(qlh + 32 + is * 16);   /* elems l + 32 */
                    uint8x16_t hv = vld1q_u8(qhh + is * 16);        /* four 2-bit tops */
                    int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(la, m4), vshlq_n_u8(vandq_u8(hv, m3), 4))), b32);
                    int8x16_t w2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(lb, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv, 2), m3), 4))), b32);
                    int8x16_t w3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(la, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv, 4), m3), 4))), b32);
                    int8x16_t w4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(lb, 4), vshlq_n_u8(vshrq_n_u8(hv, 6), 4))), b32);
                    const int8_t *x = qab + n + is * 16;
                    const int32x4_t z = vdupq_n_s32(0);
                    ssum[base + is + 0] = vaddvq_s32(vdotq_s32(z, w1, vld1q_s8(x)));
                    ssum[base + is + 2] = vaddvq_s32(vdotq_s32(z, w2, vld1q_s8(x + 32)));
                    ssum[base + is + 4] = vaddvq_s32(vdotq_s32(z, w3, vld1q_s8(x + 64)));
                    ssum[base + is + 6] = vaddvq_s32(vdotq_s32(z, w4, vld1q_s8(x + 96)));
                }
            }
            for (int j = 0; j < 16; j++)
                acc = nt_q6k_acc(acc, d, (float)sc[j], dab[j / 2], ssum[j]);
        }
        out[row] = acc;
    }
}
#else
static void nt_q6_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    (void)asum;                                  /* no bias term to lift for this format */
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t *)(b + 192);
            float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
            const int8_t *qab = qa + (long)blk * 256;
            const float  *dab = da + (long)blk * 8;
            int32_t ssum[16];
            for (int j = 0; j < 16; j++) ssum[j] = 0;
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                int base = (n / 128) * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                    ssum[base + is + 0] += q1 * (int)qab[n + l];
                    ssum[base + is + 2] += q2 * (int)qab[n + l + 32];
                    ssum[base + is + 4] += q3 * (int)qab[n + l + 64];
                    ssum[base + is + 6] += q4 * (int)qab[n + l + 96];
                }
            }
            for (int j = 0; j < 16; j++)
                acc = nt_q6k_acc(acc, d, (float)sc[j], dab[j / 2], ssum[j]);
        }
        out[row] = acc;
    }
}
#endif

/* Rows are independent and write disjoint out[], exactly as in nt_qmatvec, so the i8 path
 * fans out the same way and under the same NT_QMV_THREAD_MIN floor. The activation is
 * quantized ONCE before the fan-out and shared read-only by every worker — it is per-call
 * state, not per-row. Without this a 151936-row head ran single-threaded and the integer
 * kernel lost to the f32 one it was meant to replace. */
typedef void (*nt_qrows_i8_fn)(float *, const uint8_t *, const int8_t *, const float *,
                               const int32_t *, int, int, int);
typedef struct {
    nt_qrows_i8_fn fn; float *out; const uint8_t *Wq;
    const int8_t *qa; const float *da; const int32_t *asum; int r0, r1, k;
} nt_qjob_i8;

#ifndef _OPENMP   /* only the pthread fan-out uses a worker entry point */
static void *nt_qworker_i8(void *p) {
    nt_qjob_i8 *j = (nt_qjob_i8 *)p;
    j->fn(j->out, j->Wq, j->qa, j->da, j->asum, j->r0, j->r1, j->k);
    return NULL;
}

// Separate i8 pool keeps the shared per-call activation quant buffers typed and
// avoids a tagged union in the hot dispatch path.
//
// The handshake is atomic and spin-first, and on a phone that is the difference between a
// fan-out being worth doing and not. Measured on an Exynos 1580 before this: a 128x896
// matvec took 35 us on one core and 83 us on four — the dispatch cost more than the work —
// and the 4864x896 FFN shapes returned 1.45x from four cores instead of something near
// four. Two things were paying for that. Every chunk claim took a mutex, so four workers
// serialised sixty-four times per dispatch; and every dispatch woke the workers through a
// condvar, which is a futex wake and a scheduler round-trip each, paid seventy-three times
// per decoded token. Now a claim is one fetch_add, and workers spin on the generation
// counter before sleeping — between two matvecs of the same token the gap is microseconds,
// so the spin almost always catches the next job, and the condvar remains underneath so an
// idle phone is not held awake. NT_QMV_SPIN overrides the budget for measurement.
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_t threads[NT_QMV_MAX_THREADS];
    int nthreads;
    int ready;
    int shutdown;          /* atomic */
    int generation;        /* atomic: bumped once per dispatch */
    int busy;              /* atomic: workers still draining */
    int next;              /* atomic: next unclaimed row */
    nt_qjob_i8 shared;
    int hi, chunk;
} nt_qpool_i8;

static nt_qpool_i8 g_nt_qpool_i8 = {
    PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_COND_INITIALIZER,
    {0},
    0, 0, 0, 0, 0, 0,
    {0},              /* shared    */
    0, 0,             /* hi chunk  */
};

#if defined(__aarch64__) || defined(__arm__)
#define NT_QMV_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define NT_QMV_PAUSE() __asm__ __volatile__("pause" ::: "memory")
#else
#define NT_QMV_PAUSE() ((void)0)
#endif

static int nt_qmv_spin(void) { return nt_qmv_get_plan()->spin; }
static pthread_once_t g_nt_qpool_i8_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_nt_qpool_i8_dispatch_mu = PTHREAD_MUTEX_INITIALIZER;

/* Same on-demand hand-out as the f32 pool above; see the note there for why. */
static void nt_qpool_i8_drain(void) {
    /* The job is published before the generation bump that let anyone in here, so it can be
     * read without a lock; only the row cursor is contended, and that is one atomic. */
    nt_qjob_i8 j = g_nt_qpool_i8.shared;
    int hi = g_nt_qpool_i8.hi, ch = g_nt_qpool_i8.chunk;
    for (;;) {
        int r0 = __atomic_fetch_add(&g_nt_qpool_i8.next, ch, __ATOMIC_RELAXED);
        if (r0 >= hi) return;
        int r1 = r0 + ch; if (r1 > hi) r1 = hi;
        j.fn(j.out, j.Wq, j.qa, j.da, j.asum, r0, r1, j.k);
    }
}

static void *nt_qpool_i8_loop(void *p) {
    (void)p;
    int seen = 0;
    /* Read once per thread, not once per spin. The plan lives behind a pthread_once, and a
     * call into that from inside the innermost wait loop is a libc round trip per iteration
     * — measured as roughly a tenth of decode before it was hoisted out. */
    const int spin_budget = nt_qmv_spin();
    for (;;) {
        int spins = 0;
        for (;;) {
            if (__atomic_load_n(&g_nt_qpool_i8.shutdown, __ATOMIC_RELAXED)) return NULL;
            if (__atomic_load_n(&g_nt_qpool_i8.generation, __ATOMIC_ACQUIRE) != seen) break;
            if (++spins < spin_budget) { NT_QMV_PAUSE(); continue; }
            /* Budget spent: park. The generation is re-read under the lock the dispatcher
             * broadcasts under, so a job arriving in this window cannot be missed. */
            pthread_mutex_lock(&g_nt_qpool_i8.mu);
            if (__atomic_load_n(&g_nt_qpool_i8.generation, __ATOMIC_ACQUIRE) == seen &&
                !__atomic_load_n(&g_nt_qpool_i8.shutdown, __ATOMIC_RELAXED))
                pthread_cond_wait(&g_nt_qpool_i8.cv_work, &g_nt_qpool_i8.mu);
            pthread_mutex_unlock(&g_nt_qpool_i8.mu);
            spins = 0;
        }
        seen = __atomic_load_n(&g_nt_qpool_i8.generation, __ATOMIC_ACQUIRE);
        nt_qpool_i8_drain();
        __atomic_fetch_sub(&g_nt_qpool_i8.busy, 1, __ATOMIC_RELEASE);
    }
}

static void nt_qpool_i8_shutdown(void) {
    pthread_mutex_lock(&g_nt_qpool_i8.mu);
    __atomic_store_n(&g_nt_qpool_i8.shutdown, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_nt_qpool_i8.generation, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&g_nt_qpool_i8.cv_work);
    pthread_mutex_unlock(&g_nt_qpool_i8.mu);
    for (int i = 0; i < g_nt_qpool_i8.nthreads; i++)
        pthread_join(g_nt_qpool_i8.threads[i], NULL);
}

static void nt_qpool_i8_init_once(void) {
    /* One fewer than the cores we may use: the dispatching thread drains alongside them, so
     * a pool sized to the core count puts one thread too many on the cluster and every
     * matvec pays for the context switch. */
    int nt = nt_qmv_host_threads(NT_QMV_MAX_THREADS) - 1;
    nt_qmv_pin_nth(pthread_self(), 0);   /* the dispatcher takes a chunk too */
    for (int i = 0; i < nt; i++) {
        if (pthread_create(&g_nt_qpool_i8.threads[i], NULL, nt_qpool_i8_loop, NULL) != 0)
            break;
        nt_qmv_pin_nth(g_nt_qpool_i8.threads[i], i + 1);
        g_nt_qpool_i8.nthreads++;
    }
    g_nt_qpool_i8.ready = g_nt_qpool_i8.nthreads > 0;
    if (g_nt_qpool_i8.ready) atexit(nt_qpool_i8_shutdown);
}

static int nt_qpool_i8_run(const nt_qjob_i8 *jobs, int nt) {
    if (!nt_qmv_pool_enabled()) return -1;
    pthread_once(&g_nt_qpool_i8_once, nt_qpool_i8_init_once);
    if (!g_nt_qpool_i8.ready) return -1;
    int worker_nt = nt - 1;
    if (worker_nt <= 0 || worker_nt > g_nt_qpool_i8.nthreads) return -1;

    int lo = jobs[0].r0, hi = jobs[worker_nt].r1;
    /* Granularity: see nt_qmv_plan_init for what it costs and how it was measured. */
    int chunk = (hi - lo) / (nt * nt_qmv_get_plan()->chunks); if (chunk < 1) chunk = 1;

    pthread_mutex_lock(&g_nt_qpool_i8_dispatch_mu);   /* one dispatch in flight at a time */
    g_nt_qpool_i8.shared = jobs[0];
    g_nt_qpool_i8.hi = hi; g_nt_qpool_i8.chunk = chunk;
    __atomic_store_n(&g_nt_qpool_i8.next, lo, __ATOMIC_RELAXED);
    /* Every pool thread answers a dispatch, so every pool thread decrements — counting only
     * the ones this call asked for leaves the extras subtracting from the NEXT dispatch's
     * total, which lets a caller believe the job is done while a worker is still inside the
     * kernel reading the activation buffer it is about to free. A row range with no chunks
     * left costs one atomic and returns. */
    __atomic_store_n(&g_nt_qpool_i8.busy, g_nt_qpool_i8.nthreads, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_nt_qpool_i8.generation, 1, __ATOMIC_RELEASE);
    pthread_mutex_lock(&g_nt_qpool_i8.mu);            /* uncontended; wakes anyone parked */
    pthread_cond_broadcast(&g_nt_qpool_i8.cv_work);
    pthread_mutex_unlock(&g_nt_qpool_i8.mu);

    nt_qpool_i8_drain();                              /* the caller is a worker too */

    int spins = 0, spin_budget = nt_qmv_spin();      /* hoisted: see nt_qpool_i8_loop */
    while (__atomic_load_n(&g_nt_qpool_i8.busy, __ATOMIC_ACQUIRE) > 0) {
        if (++spins < spin_budget) { NT_QMV_PAUSE(); continue; }
        sched_yield(); spins = 0;
    }
    pthread_mutex_unlock(&g_nt_qpool_i8_dispatch_mu);
    return 0;
}
#endif

// Q4_K int8-dot rows: 144 B / 256 values against the per-32 int8 activation.
// The two grids COINCIDE here — a Q4_K sub-block is 32 values and so is an activation
// block — which makes the split exact and the bookkeeping simpler than Q6_K's 16/32 seam.
// Per sub-block s the affine format gives w = d*ls*q - dmin*lm, so
//     sum_p w*x  =  da[s] * ( d*ls * SUM(q*qa)  -  dmin*lm * SUM(qa) )
// The minus term depends only on the activation, so SUM(qa) is precomputed once per call
// and lifted straight out of the integer dot — no per-weight subtraction anywhere.
// q is already unsigned [0,15], so _mm256_maddubs_epi16 applies directly with no sign
// trick, and 15*127*2 = 3810 clears int16 with room to spare.
#if defined(__AVX2__) && defined(__FMA__)
static void nt_q4_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    int nb = k / 256, nsub = k / 32;
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            /* Four sub-blocks per pass. A per-sub-block horizontal reduction costs two
             * dependent hadds plus an extract, and at 32 values per sub-block that drain
             * dominates on the small matrices this body is made of (an expert is 768x2048;
             * measured 41% of stream bandwidth against 59% for the 151936-row head, where
             * the same drain amortises). One hadd cascade retires four sub-blocks instead
             * of one, so the drain is paid 3 times per 4 instead of 8.
             * The float accumulation ORDER is deliberately unchanged — contributions are
             * still added sub-block by sub-block ascending — so this stays a pure integer
             * re-order and the consumer's greedy vector must not move. */
            /* Whole block in one pass. Two things the four-at-a-time form still paid for:
             * qs was loaded EIGHT times though sub-blocks 2p and 2p+1 share one 32-byte load
             * (low nibbles feed the even sub-block, high nibbles the odd one), and the 6-bit
             * (scale,min) unpack sat inside the hot loop. Now: four loads, one unpack pass,
             * and the eight sub-block dots land in the lanes of a single vector through one
             * hadd tree instead of a drain per sub-block.
             * The float accumulation order is unchanged — still ascending by sub-block — so
             * this remains an integer re-order and the consumer's greedy vector must not move. */
            uint8_t ls[8], lm[8];
            for (int j = 0; j < 8; j++) nt_get_scale_min_k4(j, sc, &ls[j], &lm[j]);
            __m256i s[8];
            for (int p = 0; p < 4; p++) {
                __m256i qsv = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                __m256i lo  = _mm256_and_si256(qsv, m4);
                __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(qsv, 4), m4);
                __m256i a0  = _mm256_loadu_si256((const __m256i *)(qa + (long)(blk * 8 + 2*p) * 32));
                __m256i a1  = _mm256_loadu_si256((const __m256i *)(qa + (long)(blk * 8 + 2*p + 1) * 32));
                s[2*p]     = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a0), ones);
                s[2*p + 1] = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a1), ones);
            }
            __m256i A = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]),
                                          _mm256_hadd_epi32(s[2], s[3]));
            __m256i B = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]),
                                          _mm256_hadd_epi32(s[6], s[7]));
            __m256i sums = _mm256_add_epi32(_mm256_permute2x128_si256(A, B, 0x20),
                                            _mm256_permute2x128_si256(A, B, 0x31));
            /* The scalar tail stays scalar on purpose. GCC contracts
             * d * scale * dot - dmin * min * asum into an FMA under -march=native, and that
             * single rounding is part of every number this kernel has ever been accepted on.
             * Intrinsics cannot express the compiler's contraction choice: a hand-vectorised
             * tail was measured against this one and differed on 521 of 768 rows, worst 4.8e-4
             * relative, which moved the perplexity of an untouched container in the fourth
             * decimal while every argmax and every six-digit probe stayed put. The dots are
             * vectorised because they are integers and exact; the float tail is not. */
            int32_t dots[8];
            _mm256_storeu_si256((__m256i *)dots, sums);
            for (int j = 0; j < 8; j++) {
                int sub = blk * 8 + j;
                acc = nt_q4k_acc(acc, da[sub], d, (float)ls[j], dots[j], dmin, (float)lm[j], asum[sub]);
            }
        }
        out[row] = acc;
    }
}
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
// Q4_K on NEON needs no sign trick either, for a different reason than Q6_K: the nibble is
// already unsigned [0,15], and 15 is representable in int8, so reinterpreting it as signed
// is the identity and plain SDOT is exact. USDOT would also serve, but it is an i8mm
// instruction and this kernel has no reason to demand the wider baseline.
// SUM(qa) is a dot against a vector of ones — the same lift the x86 path takes, minus the
// maddubs detour. Two 16-lane SDOTs cover a 32-value sub-block, so a whole 256-value block
// is eight of them plus eight drains, against 256 scalar multiply-adds in the fallback.
// The float tail is the scalar one verbatim, ascending by sub-block: integer dots are exact,
// so this kernel and the fallback must agree bit for bit.
static void nt_q4_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    int nb = k / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            /* One 32-byte weight load feeds two sub-blocks: low nibbles the even one, high
             * nibbles the odd one — the same pairing the fallback expresses as (j >> 1). */
            int32_t dots[8];
            for (int p = 0; p < 4; p++) {
                uint8x16_t q0 = vld1q_u8(qs + p * 32);
                uint8x16_t q1 = vld1q_u8(qs + p * 32 + 16);
                const int8_t *a0 = qa + (long)(blk * 8 + 2 * p) * 32;
                const int8_t *a1 = qa + (long)(blk * 8 + 2 * p + 1) * 32;
                const int32x4_t z = vdupq_n_s32(0);
                int32x4_t e = vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(q0, m4)), vld1q_s8(a0));
                e = vdotq_s32(e, vreinterpretq_s8_u8(vandq_u8(q1, m4)), vld1q_s8(a0 + 16));
                int32x4_t o = vdotq_s32(z, vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), vld1q_s8(a1));
                o = vdotq_s32(o, vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), vld1q_s8(a1 + 16));
                dots[2 * p]     = vaddvq_s32(e);
                dots[2 * p + 1] = vaddvq_s32(o);
            }
            for (int j = 0; j < 8; j++) {
                uint8_t s6, m6; nt_get_scale_min_k4(j, sc, &s6, &m6);
                int sub = blk * 8 + j;
                acc = nt_q4k_acc(acc, da[sub], d, (float)s6, dots[j], dmin, (float)m6, asum[sub]);
            }
        }
        out[row] = acc;
    }
}
#else
static void nt_q4_k_rows_i8(float *out, const uint8_t *W, const int8_t *qa,
                            const float *da, const int32_t *asum,
                            int r0, int r1, int k) {
    int nb = k / 256, nsub = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            for (int j = 0; j < 8; j++) {
                uint8_t s6, m6; nt_get_scale_min_k4(j, sc, &s6, &m6);
                int sub = blk * 8 + j;
                const uint8_t *qsp = qs + (j >> 1) * 32;
                const int8_t  *qab = qa + (long)sub * 32;
                int32_t dot = 0;
                if (j & 1) for (int l = 0; l < 32; l++) dot += (int32_t)(qsp[l] >> 4)   * qab[l];
                else       for (int l = 0; l < 32; l++) dot += (int32_t)(qsp[l] & 0x0F) * qab[l];
                acc = nt_q4k_acc(acc, da[sub], d, (float)s6, dot, dmin, (float)m6, asum[sub]);
            }
        }
        out[row] = acc;
    }
}
#endif

/* Pick the i8 row kernel for a dtype, or NULL. Shared by the fan-out entry and the
 * caller-parallel one so the two can never drift apart. */
static nt_qrows_i8_fn nt_qrows_i8_for(int dtype, int k) {
    if (k % 32) return NULL;
    switch (dtype) {
    case 2:  return nt_q4_0_rows_i8;
    case 6:  return nt_q5_0_rows_i8;
    case 8:  return nt_q8_0_rows_i8;
    case 12: return (k % 256) ? NULL : nt_q4_k_rows_i8;
    case 14: return (k % 256) ? NULL : nt_q6_k_rows_i8;
    default: return NULL;
    }
}

void nt_quant_act(const float *x, int k, int8_t *qa, float *da) {
    nt_quant_act_q8(x, k, qa, da);
}

int nt_qmatvec_i8_rows(float *out, const uint8_t *Wq, int dtype,
                       const int8_t *qa, const float *da, int r0, int r1, int k) {
    nt_qrows_i8_fn fn = nt_qrows_i8_for(dtype, k);
    if (!fn) return -1;
    if (k / 32 > NT_QMV_ASUM_MAX) return -1;
    /* This entry takes the activation already quantized and knows nothing about how many
     * times the caller will hand it another row range, so the block sums are rebuilt here.
     * The pooled entry below builds them once beside the quantization, which is where they
     * belong when one call owns the whole matrix. */
    int32_t asum[NT_QMV_ASUM_MAX];
    nt_act_block_sums(qa, k, asum);
    if (r1 > r0) fn(out, Wq, qa, da, asum, r0, r1, k);
    return 0;
}

int nt_qmv_planned_threads(void) { return nt_qmv_host_threads(NT_QMV_MAX_THREADS); }

int nt_qmatvec_i8(float *out, const uint8_t *Wq, int dtype,
                  const float *x, int m, int k) {
    /* The dtype and shape were checked here from the start and the pointers were not, so a
     * caller whose weight failed to load reached the kernel and crashed there instead of
     * getting the -1 it was testing for. Callers do test for it: harness/arch_gemma4.c
     * zeroes its per-layer section on a non-zero return. */
    if (!out || !Wq || !x || m <= 0 || k <= 0) return -1;
    if (dtype != 2 && dtype != 6 && dtype != 8 && dtype != 12 && dtype != 14)
        return -1;                                       /* Q4_0/Q5_0/Q8_0/Q4_K/Q6_K */
    if (k % 32) return -1;
    if ((dtype == 12 || dtype == 14) && (k % 256)) return -1;
    int nb = k / 32;
    int8_t *qa = (int8_t *)malloc((size_t)k);
    float  *da = (float *)malloc((size_t)nb * sizeof(float));
    /* Q5_0's -16 bias and Q4_K's affine minimum both lift out of the integer dot as a
     * multiple of SUM(qa) per block, and the sum depends on the activation alone. The
     * kernels used to rebuild it on entry, which meant once per ROW CHUNK — with eight
     * chunks a worker that is thirty-two rebuilds of the same numbers per matvec, and about
     * four percent of a decode. It is built here now, once, beside the quantization that
     * produces the bytes it sums. */
    int32_t *asum = (int32_t *)malloc((size_t)nb * sizeof(int32_t));
    if (!qa || !da || !asum) { free(qa); free(da); free(asum); return -1; }
    nt_quant_act_q8(x, k, qa, da);
    nt_act_block_sums(qa, k, asum);

    /* Selected by table, not by a ternary chain whose last arm is a default. That shape
     * is why widening the guard above without touching this line sent Q5_0 into the Q6_K
     * kernel and read 210-byte blocks out of a 22-byte-block buffer. */
    nt_qrows_i8_fn fn = nt_qrows_i8_for(dtype, k);
    if (!fn) { free(qa); free(da); free(asum); return -1; }
    int nt = nt_qmv_host_threads(m);
    if (nt <= 1 || (long)m * k < nt_qmv_thread_floor()) {
        fn(out, Wq, qa, da, asum, 0, m, k);
        free(qa); free(da); free(asum);
        return 0;
    }
#ifdef _OPENMP
    /* Same reasoning as nt_qmatvec: reuse the caller's OpenMP team rather than opening a
     * second, competing set of threads. The activation stays quantized once, before the
     * region, and is read-only inside it. */
    int per_omp = (m + nt - 1) / nt;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int r0 = t * per_omp, r1 = (r0 + per_omp > m) ? m : r0 + per_omp;
        if (r0 < m) fn(out, Wq, qa, da, asum, r0, r1, k);
    }
#else
    pthread_t th[NT_QMV_MAX_THREADS];
    nt_qjob_i8 jobs[NT_QMV_MAX_THREADS];
    int per = (m + nt - 1) / nt, launched = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per, r1 = (r0 + per > m) ? m : r0 + per;
        if (r0 >= m) break;
        jobs[t] = (nt_qjob_i8){ fn, out, Wq, qa, da, asum, r0, r1, k };
        launched++;
    }
    if (nt_qpool_i8_run(jobs, launched) == 0) {
        free(qa);
        free(da);
        free(asum);
        return 0;
    }

    launched = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per;
        if (r0 >= m) break;
        if (pthread_create(&th[t], NULL, nt_qworker_i8, &jobs[t]) != 0) {
            fn(out, Wq, qa, da, asum, r0, m, k);   /* create failed: run the rest inline */
            break;
        }
        launched++;
    }
    for (int t = 0; t < launched; t++) pthread_join(th[t], NULL);
#endif
    free(qa); free(da); free(asum);
    return 0;
}

// ── batched int8 matmul — one pass over the weights, many activations ───────────
// Prefill pushes n token vectors through the same weight matrix, and the per-token
// entry above re-reads every packed byte for each of them: a 0.5B Qwen streams 373 MiB
// of weights per token, so a 241-token prompt streams them 241 times and prefill runs
// at decode speed. Here a row is unpacked once and dotted against a tile of activations,
// which divides the weight traffic by the tile width. The activation side is bit for bit
// the per-32-block int8 of nt_qmatvec_i8 and the accumulation order per row is the same,
// so a batched prefill and a token-by-token one produce identical floats, not merely
// close ones — the test asserts equality, not a tolerance.
#define NT_QMM_TILE 32   /* activations carried through one pass over the weights */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
/* One row range against one range of activations, SDOT. Split out because the i8mm path
 * below covers rows and activations two at a time and has to hand the odd one back. */
static void nt_q4_0_sdot_range(float *out, int m, const uint8_t *W, const int8_t *qa,
                               const float *da, int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const int8x16_t  eight  = vdupq_n_s8(8);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 18;
        float acc[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) acc[j] = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 18;
            float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            uint8x16_t packed = vld1q_u8(blk + 2);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask0f)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), eight);
            for (int j = 0; j < jn; j++) {
                const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                int32x4_t s4 = vdupq_n_s32(0);
                s4 = vdotq_s32(s4, lo, vld1q_s8(qab));
                s4 = vdotq_s32(s4, hi, vld1q_s8(qab + 16));
                acc[j] += d_w * da[(long)(j0 + j) * nb + b] * (float)vaddvq_s32(s4);
            }
        }
        for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* SMMLA takes two 2x8 int8 operands and returns their 2x2 product in one instruction:
 * put two WEIGHT rows in one and two ACTIVATION rows in the other and a single instruction
 * retires four dot products of eight. That is 32 multiply-accumulates against SDOT's 16,
 * but the reason it is here is the other half of the ledger. Doubling the SDOTs at constant
 * loads measured free on this core — 7.7 ms against 7.9 for the same shape — so the kernel
 * was never arithmetic-bound; it was bound by feeding, four 16-byte activation loads per 64
 * MACs. Under SMMLA each activation half-vector is read once and serves two rows, so the
 * bytes per MAC halve. Nothing about the arithmetic changes: integer dots are exact and the
 * float tail still walks blocks ascending per (row, activation), so the result is the same
 * bits as the SDOT path, which is what tests/test_qmatmul.c asserts.
 *
 * i8mm is unused on this phone by anything else — the llama.cpp shipped for Termux carries
 * neither SMMLA nor SDOT in its CPU backend — so this is the one lane where the Method is
 * not catching up with anybody. */
static void nt_q4_0_rows_i8mm(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const int8x16_t  eight  = vdupq_n_s8(8);
    int rpair = r0 + ((r1 - r0) & ~1);
    int jpair = jn & ~1;

    for (int row = r0; row < rpair; row += 2) {
        const uint8_t *rb0 = W + (long)row * nb * 18;
        const uint8_t *rb1 = rb0 + (long)nb * 18;
        float acc0[NT_QMM_TILE], acc1[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk0 = rb0 + (long)b * 18, *blk1 = rb1 + (long)b * 18;
            float d0 = nt_f16_to_f32((uint16_t)(blk0[0] | (blk0[1] << 8)));
            float d1 = nt_f16_to_f32((uint16_t)(blk1[0] | (blk1[1] << 8)));
            uint8x16_t p0 = vld1q_u8(blk0 + 2), p1 = vld1q_u8(blk1 + 2);
            int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask0f)), eight);
            int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p0, 4)), eight);
            int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask0f)), eight);
            int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p1, 4)), eight);
            /* Four 2x8 weight tiles: elements 0-7, 8-15 (low nibbles), 16-23, 24-31 (high),
             * each carrying row and row+1 stacked, built once for the whole activation tile. */
            int8x16_t A0 = vcombine_s8(vget_low_s8(lo0),  vget_low_s8(lo1));
            int8x16_t A1 = vcombine_s8(vget_high_s8(lo0), vget_high_s8(lo1));
            int8x16_t A2 = vcombine_s8(vget_low_s8(hi0),  vget_low_s8(hi1));
            int8x16_t A3 = vcombine_s8(vget_high_s8(hi0), vget_high_s8(hi1));
            for (int j = 0; j < jpair; j += 2) {
                const int8_t *x0 = qa + (long)(j0 + j) * k + (long)b * 32;
                const int8_t *x1 = x0 + k;
                int8x16_t B0 = vcombine_s8(vld1_s8(x0),      vld1_s8(x1));
                int8x16_t B1 = vcombine_s8(vld1_s8(x0 + 8),  vld1_s8(x1 + 8));
                int8x16_t B2 = vcombine_s8(vld1_s8(x0 + 16), vld1_s8(x1 + 16));
                int8x16_t B3 = vcombine_s8(vld1_s8(x0 + 24), vld1_s8(x1 + 24));
                int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A0, B0);
                s = vmmlaq_s32(s, A1, B1);
                s = vmmlaq_s32(s, A2, B2);
                s = vmmlaq_s32(s, A3, B3);
                /* lanes: row.j, row.j+1, row+1.j, row+1.j+1 */
                float da0 = da[(long)(j0 + j) * nb + b], da1 = da[(long)(j0 + j + 1) * nb + b];
                acc0[j]     += d0 * da0 * (float)vgetq_lane_s32(s, 0);
                acc0[j + 1] += d0 * da1 * (float)vgetq_lane_s32(s, 1);
                acc1[j]     += d1 * da0 * (float)vgetq_lane_s32(s, 2);
                acc1[j + 1] += d1 * da1 * (float)vgetq_lane_s32(s, 3);
            }
            if (jpair < jn) {                    /* odd activation: SDOT against both rows */
                const int8_t *x = qa + (long)(j0 + jpair) * k + (long)b * 32;
                int8x16_t X0 = vld1q_s8(x), X1 = vld1q_s8(x + 16);
                int32x4_t t0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, X0), hi0, X1);
                int32x4_t t1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo1, X0), hi1, X1);
                float daj = da[(long)(j0 + jpair) * nb + b];
                acc0[jpair] += d0 * daj * (float)vaddvq_s32(t0);
                acc1[jpair] += d1 * daj * (float)vaddvq_s32(t1);
            }
        }
        for (int j = 0; j < jn; j++) {
            out[(long)(j0 + j) * m + row]     = acc0[j];
            out[(long)(j0 + j) * m + row + 1] = acc1[j];
        }
    }
    if (rpair < r1)                              /* odd row: the plain path takes it */
        nt_q4_0_sdot_range(out, m, W, qa, da, rpair, r1, k, j0, jn);
}
#endif

static void nt_q4_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;                                  /* Q4_0 has no min term to lift */
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        if (jn >= 2 && (r1 - r0) >= 2) {
            nt_q4_0_rows_i8mm(out, m, W, qa, da, r0, r1, k, j0, jn);
            continue;
        }
#endif
        nt_q4_0_sdot_range(out, m, W, qa, da, r0, r1, k, j0, jn);
    }
}
#else
static void nt_q4_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;                                  /* Q4_0 has no min term to lift */
    int nb = k / 32;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
        for (int row = r0; row < r1; row++) {
            const uint8_t *rb = W + (long)row * nb * 18;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = rb + (long)b * 18;
                float d_w = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                int8_t wv[32];
                for (int i = 0; i < 16; i++) {
                    wv[i]      = (int8_t)((int)(blk[2 + i] & 0x0F) - 8);
                    wv[i + 16] = (int8_t)((int)(blk[2 + i] >> 4)   - 8);
                }
                for (int j = 0; j < jn; j++) {
                    const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                    int32_t s = 0;
                    for (int i = 0; i < 32; i++) s += wv[i] * qab[i];
                    acc[j] += d_w * da[(long)(j0 + j) * nb + b] * (float)s;
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}
#endif

// Q4_K batched: 144 B / 256 values, eight sub-blocks whose grid coincides with the
// activation's. Per (row, block) the affine format costs an unpack of eight 6-bit
// (scale, min) pairs and four 32-byte nibble loads before a single dot happens; per token
// that overhead is paid again for the same bytes. Here it is paid once for the tile, and
// SUM(qa) — the activation half of the min term — is precomputed once for the whole call
// rather than once per row range. The float tail keeps the per-token order, ascending by
// sub-block, so the batched and the per-token result are the same bits.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void nt_q4_k_rows_i8n_sdot(float *out, int m, const uint8_t *W, const int8_t *qa,
                                  const float *da, const int32_t *asum,
                                  int r0, int r1, int k, int j0, int jn) {
    int nb = k / 256, nsub = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) acc[j] = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            uint8_t ls[8], lm[8];
            for (int s = 0; s < 8; s++) nt_get_scale_min_k4(s, sc, &ls[s], &lm[s]);
            for (int p = 0; p < 4; p++) {
                uint8x16_t q0 = vld1q_u8(qs + p * 32);
                uint8x16_t q1 = vld1q_u8(qs + p * 32 + 16);
                int8x16_t e0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
                int8x16_t e1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
                int8x16_t o0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                int8x16_t o1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                int sub_e = blk * 8 + 2 * p, sub_o = sub_e + 1;
                for (int j = 0; j < jn; j++) {
                    const int8_t *a0 = qa + (long)(j0 + j) * k + (long)sub_e * 32;
                    const int8_t *a1 = a0 + 32;
                    const float *daj = da + (long)(j0 + j) * nsub;
                    const int32_t *asj = asum + (long)(j0 + j) * nsub;
                    const int32x4_t z = vdupq_n_s32(0);
                    int32x4_t e = vdotq_s32(z, e0, vld1q_s8(a0));
                    e = vdotq_s32(e, e1, vld1q_s8(a0 + 16));
                    int32x4_t o = vdotq_s32(z, o0, vld1q_s8(a1));
                    o = vdotq_s32(o, o1, vld1q_s8(a1 + 16));
                    acc[j] = nt_q4k_acc(acc[j], daj[sub_e], d, (float)ls[2*p],   vaddvq_s32(e), dmin, (float)lm[2*p],   asj[sub_e]);
                    acc[j] = nt_q4k_acc(acc[j], daj[sub_o], d, (float)ls[2*p+1], vaddvq_s32(o), dmin, (float)lm[2*p+1], asj[sub_o]);
                }
            }
        }
        for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* Q4_K under SMMLA. A super-block is eight 32-value sub-blocks, and each of those is two
 * 16-value halves in the packing, so one sub-block is four 2x8 tiles once a row pair is
 * stacked. Everything else — the affine minimum lifted as dmin*lm*SUM(qa), the ascending
 * sub-block order in the float tail — is the SDOT kernel's, unchanged, because changing it
 * would change the last bit and the test compares bits. */
static void nt_q4_k_rows_i8mm(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, const int32_t *asum,
                              int r0, int r1, int k, int j0, int jn) {
    int nb = k / 256, nsub = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    int rpair = r0 + ((r1 - r0) & ~1);
    int jpair = jn & ~1;
    for (int row = r0; row < rpair; row += 2) {
        const uint8_t *rb0 = W + (long)row * nb * 144;
        const uint8_t *rb1 = rb0 + (long)nb * 144;
        float acc0[NT_QMM_TILE], acc1[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b0 = rb0 + (long)blk * 144, *b1 = rb1 + (long)blk * 144;
            float d0    = nt_f16_to_f32((uint16_t)(b0[0] | (b0[1] << 8)));
            float dmin0 = nt_f16_to_f32((uint16_t)(b0[2] | (b0[3] << 8)));
            float d1    = nt_f16_to_f32((uint16_t)(b1[0] | (b1[1] << 8)));
            float dmin1 = nt_f16_to_f32((uint16_t)(b1[2] | (b1[3] << 8)));
            const uint8_t *sc0 = b0 + 4, *qs0 = b0 + 16;
            const uint8_t *sc1 = b1 + 4, *qs1 = b1 + 16;
            uint8_t ls0[8], lm0[8], ls1[8], lm1[8];
            for (int s = 0; s < 8; s++) {
                nt_get_scale_min_k4(s, sc0, &ls0[s], &lm0[s]);
                nt_get_scale_min_k4(s, sc1, &ls1[s], &lm1[s]);
            }
            for (int p = 0; p < 4; p++) {
                uint8x16_t w00 = vld1q_u8(qs0 + p * 32), w01 = vld1q_u8(qs0 + p * 32 + 16);
                uint8x16_t w10 = vld1q_u8(qs1 + p * 32), w11 = vld1q_u8(qs1 + p * 32 + 16);
                int8x16_t e00 = vreinterpretq_s8_u8(vandq_u8(w00, m4));
                int8x16_t e01 = vreinterpretq_s8_u8(vandq_u8(w01, m4));
                int8x16_t o00 = vreinterpretq_s8_u8(vshrq_n_u8(w00, 4));
                int8x16_t o01 = vreinterpretq_s8_u8(vshrq_n_u8(w01, 4));
                int8x16_t e10 = vreinterpretq_s8_u8(vandq_u8(w10, m4));
                int8x16_t e11 = vreinterpretq_s8_u8(vandq_u8(w11, m4));
                int8x16_t o10 = vreinterpretq_s8_u8(vshrq_n_u8(w10, 4));
                int8x16_t o11 = vreinterpretq_s8_u8(vshrq_n_u8(w11, 4));
                int8x16_t E0 = vcombine_s8(vget_low_s8(e00),  vget_low_s8(e10));
                int8x16_t E1 = vcombine_s8(vget_high_s8(e00), vget_high_s8(e10));
                int8x16_t E2 = vcombine_s8(vget_low_s8(e01),  vget_low_s8(e11));
                int8x16_t E3 = vcombine_s8(vget_high_s8(e01), vget_high_s8(e11));
                int8x16_t O0 = vcombine_s8(vget_low_s8(o00),  vget_low_s8(o10));
                int8x16_t O1 = vcombine_s8(vget_high_s8(o00), vget_high_s8(o10));
                int8x16_t O2 = vcombine_s8(vget_low_s8(o01),  vget_low_s8(o11));
                int8x16_t O3 = vcombine_s8(vget_high_s8(o01), vget_high_s8(o11));
                int sub_e = blk * 8 + 2 * p, sub_o = sub_e + 1;
                for (int j = 0; j < jpair; j += 2) {
                    const int8_t *x0 = qa + (long)(j0 + j) * k + (long)sub_e * 32;
                    const int8_t *x1 = x0 + k;
                    int32x4_t se = vmmlaq_s32(vdupq_n_s32(0), E0,
                                              vcombine_s8(vld1_s8(x0),      vld1_s8(x1)));
                    se = vmmlaq_s32(se, E1, vcombine_s8(vld1_s8(x0 + 8),  vld1_s8(x1 + 8)));
                    se = vmmlaq_s32(se, E2, vcombine_s8(vld1_s8(x0 + 16), vld1_s8(x1 + 16)));
                    se = vmmlaq_s32(se, E3, vcombine_s8(vld1_s8(x0 + 24), vld1_s8(x1 + 24)));
                    int32x4_t so = vmmlaq_s32(vdupq_n_s32(0), O0,
                                              vcombine_s8(vld1_s8(x0 + 32), vld1_s8(x1 + 32)));
                    so = vmmlaq_s32(so, O1, vcombine_s8(vld1_s8(x0 + 40), vld1_s8(x1 + 40)));
                    so = vmmlaq_s32(so, O2, vcombine_s8(vld1_s8(x0 + 48), vld1_s8(x1 + 48)));
                    so = vmmlaq_s32(so, O3, vcombine_s8(vld1_s8(x0 + 56), vld1_s8(x1 + 56)));
                    const float *daA = da + (long)(j0 + j) * nsub;
                    const float *daB = da + (long)(j0 + j + 1) * nsub;
                    const int32_t *asA = asum + (long)(j0 + j) * nsub;
                    const int32_t *asB = asum + (long)(j0 + j + 1) * nsub;
                    acc0[j] = nt_q4k_acc(acc0[j], daA[sub_e], d0, (float)ls0[2*p],
                                               vgetq_lane_s32(se, 0), dmin0, (float)lm0[2*p],   asA[sub_e]);
                    acc0[j] = nt_q4k_acc(acc0[j], daA[sub_o], d0, (float)ls0[2*p+1],
                                               vgetq_lane_s32(so, 0), dmin0, (float)lm0[2*p+1], asA[sub_o]);
                    acc0[j + 1] = nt_q4k_acc(acc0[j + 1], daB[sub_e], d0, (float)ls0[2*p],
                                               vgetq_lane_s32(se, 1), dmin0, (float)lm0[2*p],   asB[sub_e]);
                    acc0[j + 1] = nt_q4k_acc(acc0[j + 1], daB[sub_o], d0, (float)ls0[2*p+1],
                                               vgetq_lane_s32(so, 1), dmin0, (float)lm0[2*p+1], asB[sub_o]);
                    acc1[j] = nt_q4k_acc(acc1[j], daA[sub_e], d1, (float)ls1[2*p],
                                               vgetq_lane_s32(se, 2), dmin1, (float)lm1[2*p],   asA[sub_e]);
                    acc1[j] = nt_q4k_acc(acc1[j], daA[sub_o], d1, (float)ls1[2*p+1],
                                               vgetq_lane_s32(so, 2), dmin1, (float)lm1[2*p+1], asA[sub_o]);
                    acc1[j + 1] = nt_q4k_acc(acc1[j + 1], daB[sub_e], d1, (float)ls1[2*p],
                                               vgetq_lane_s32(se, 3), dmin1, (float)lm1[2*p],   asB[sub_e]);
                    acc1[j + 1] = nt_q4k_acc(acc1[j + 1], daB[sub_o], d1, (float)ls1[2*p+1],
                                               vgetq_lane_s32(so, 3), dmin1, (float)lm1[2*p+1], asB[sub_o]);
                }
                if (jpair < jn) {
                    const int8_t *x = qa + (long)(j0 + jpair) * k + (long)sub_e * 32;
                    const float *daj = da + (long)(j0 + jpair) * nsub;
                    const int32_t *asj = asum + (long)(j0 + jpair) * nsub;
                    int8x16_t X0 = vld1q_s8(x),      X1 = vld1q_s8(x + 16);
                    int8x16_t X2 = vld1q_s8(x + 32), X3 = vld1q_s8(x + 48);
                    const int32x4_t z = vdupq_n_s32(0);
                    int32_t e0v = vaddvq_s32(vdotq_s32(vdotq_s32(z, e00, X0), e01, X1));
                    int32_t o0v = vaddvq_s32(vdotq_s32(vdotq_s32(z, o00, X2), o01, X3));
                    int32_t e1v = vaddvq_s32(vdotq_s32(vdotq_s32(z, e10, X0), e11, X1));
                    int32_t o1v = vaddvq_s32(vdotq_s32(vdotq_s32(z, o10, X2), o11, X3));
                    acc0[jpair] = nt_q4k_acc(acc0[jpair], daj[sub_e], d0, (float)ls0[2*p],
                                               e0v, dmin0, (float)lm0[2*p],   asj[sub_e]);
                    acc0[jpair] = nt_q4k_acc(acc0[jpair], daj[sub_o], d0, (float)ls0[2*p+1],
                                               o0v, dmin0, (float)lm0[2*p+1], asj[sub_o]);
                    acc1[jpair] = nt_q4k_acc(acc1[jpair], daj[sub_e], d1, (float)ls1[2*p],
                                               e1v, dmin1, (float)lm1[2*p],   asj[sub_e]);
                    acc1[jpair] = nt_q4k_acc(acc1[jpair], daj[sub_o], d1, (float)ls1[2*p+1],
                                               o1v, dmin1, (float)lm1[2*p+1], asj[sub_o]);
                }
            }
        }
        for (int j = 0; j < jn; j++) {
            out[(long)(j0 + j) * m + row]     = acc0[j];
            out[(long)(j0 + j) * m + row + 1] = acc1[j];
        }
    }
    if (rpair < r1)
        nt_q4_k_rows_i8n_sdot(out, m, W, qa, da, asum, rpair, r1, k, j0, jn);
}
#endif

static void nt_q4_k_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        if (jn >= 2 && (r1 - r0) >= 2) {
            nt_q4_k_rows_i8mm(out, m, W, qa, da, asum, r0, r1, k, j0, jn);
            continue;
        }
#endif
        nt_q4_k_rows_i8n_sdot(out, m, W, qa, da, asum, r0, r1, k, j0, jn);
    }
}
#else
static void nt_q4_k_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    int nb = k / 256, nsub = k / 32;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
        for (int row = r0; row < r1; row++) {
            const uint8_t *rb = W + (long)row * nb * 144;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int blk = 0; blk < nb; blk++) {
                const uint8_t *b = rb + (long)blk * 144;
                float d    = nt_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
                float dmin = nt_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
                const uint8_t *sc = b + 4, *qs = b + 16;
                uint8_t ls[8], lm[8];
                for (int s = 0; s < 8; s++) nt_get_scale_min_k4(s, sc, &ls[s], &lm[s]);
                for (int s = 0; s < 8; s++) {
                    int sub = blk * 8 + s;
                    const uint8_t *qsb = qs + (s >> 1) * 32;
                    int shift = (s & 1) * 4;
                    for (int j = 0; j < jn; j++) {
                        const int8_t *a = qa + (long)(j0 + j) * k + (long)sub * 32;
                        int32_t dot = 0;
                        for (int i = 0; i < 32; i++)
                            dot += (int32_t)((qsb[i] >> shift) & 0x0F) * a[i];
                        acc[j] = nt_q4k_acc(acc[j], da[(long)(j0 + j) * nsub + sub], d, (float)ls[s], dot,
                                              dmin, (float)lm[s], asum[(long)(j0 + j) * nsub + sub]);
                    }
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}
#endif

// Q5_0 batched: 22 B / 32 values — an f16 scale, a 32-bit high-bit mask, 16 nibble bytes.
// The unpack is the most expensive of the block formats here (two table loads, an AND, a
// shift and two ORs before any dot happens), which makes it the one that gains most from
// being done once for a tile of activations instead of once per token. The -16 bias still
// lifts out as 16*SUM(qa), and SUM(qa) now comes precomputed per activation.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
/* One block of Q5_0 unpacked: elements 0-15 and 16-31, the -16 bias still lifted out. */
#define NT_Q5_0_UNPACK(blk, lo, hi)                                                        \
    uint8x16_t h0_ = vreinterpretq_u8_u64(vcombine_u64(                                    \
        vcreate_u64(nt_q5_hi[(blk)[2]]), vcreate_u64(nt_q5_hi[(blk)[3]])));                \
    uint8x16_t h1_ = vreinterpretq_u8_u64(vcombine_u64(                                    \
        vcreate_u64(nt_q5_hi[(blk)[4]]), vcreate_u64(nt_q5_hi[(blk)[5]])));                \
    uint8x16_t pk_ = vld1q_u8((blk) + 6);                                                  \
    int8x16_t lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(pk_, m4), h0_));                  \
    int8x16_t hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(pk_, 4), h1_))

static void nt_q5_0_sdot_range(float *out, int m, const uint8_t *W, const int8_t *qa,
                               const float *da, const int32_t *asum,
                               int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 22;
        float acc[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) acc[j] = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 22;
            float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            NT_Q5_0_UNPACK(blk, lo, hi);
            for (int j = 0; j < jn; j++) {
                const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                int32x4_t t = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(qab));
                t = vdotq_s32(t, hi, vld1q_s8(qab + 16));
                acc[j] += d * da[(long)(j0 + j) * nb + b]
                        * (float)(vaddvq_s32(t) - 16 * asum[(long)(j0 + j) * nb + b]);
            }
        }
        for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void nt_q5_0_rows_i8mm(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, const int32_t *asum,
                              int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    int rpair = r0 + ((r1 - r0) & ~1);
    int jpair = jn & ~1;
    for (int row = r0; row < rpair; row += 2) {
        const uint8_t *rb0 = W + (long)row * nb * 22;
        const uint8_t *rb1 = rb0 + (long)nb * 22;
        float acc0[NT_QMM_TILE], acc1[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk0 = rb0 + (long)b * 22, *blk1 = rb1 + (long)b * 22;
            float d0 = nt_f16_to_f32((uint16_t)(blk0[0] | (blk0[1] << 8)));
            float d1 = nt_f16_to_f32((uint16_t)(blk1[0] | (blk1[1] << 8)));
            int8x16_t lo0, hi0, lo1, hi1;
            { NT_Q5_0_UNPACK(blk0, l_, h_); lo0 = l_; hi0 = h_; }
            { NT_Q5_0_UNPACK(blk1, l_, h_); lo1 = l_; hi1 = h_; }
            int8x16_t A0 = vcombine_s8(vget_low_s8(lo0),  vget_low_s8(lo1));
            int8x16_t A1 = vcombine_s8(vget_high_s8(lo0), vget_high_s8(lo1));
            int8x16_t A2 = vcombine_s8(vget_low_s8(hi0),  vget_low_s8(hi1));
            int8x16_t A3 = vcombine_s8(vget_high_s8(hi0), vget_high_s8(hi1));
            for (int j = 0; j < jpair; j += 2) {
                const int8_t *x0 = qa + (long)(j0 + j) * k + (long)b * 32;
                const int8_t *x1 = x0 + k;
                int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A0,
                                         vcombine_s8(vld1_s8(x0),      vld1_s8(x1)));
                s = vmmlaq_s32(s, A1, vcombine_s8(vld1_s8(x0 + 8),  vld1_s8(x1 + 8)));
                s = vmmlaq_s32(s, A2, vcombine_s8(vld1_s8(x0 + 16), vld1_s8(x1 + 16)));
                s = vmmlaq_s32(s, A3, vcombine_s8(vld1_s8(x0 + 24), vld1_s8(x1 + 24)));
                long o0 = (long)(j0 + j) * nb + b, o1 = (long)(j0 + j + 1) * nb + b;
                acc0[j]     += d0 * da[o0] * (float)(vgetq_lane_s32(s, 0) - 16 * asum[o0]);
                acc0[j + 1] += d0 * da[o1] * (float)(vgetq_lane_s32(s, 1) - 16 * asum[o1]);
                acc1[j]     += d1 * da[o0] * (float)(vgetq_lane_s32(s, 2) - 16 * asum[o0]);
                acc1[j + 1] += d1 * da[o1] * (float)(vgetq_lane_s32(s, 3) - 16 * asum[o1]);
            }
            if (jpair < jn) {
                const int8_t *x = qa + (long)(j0 + jpair) * k + (long)b * 32;
                int8x16_t X0 = vld1q_s8(x), X1 = vld1q_s8(x + 16);
                int32x4_t t0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, X0), hi0, X1);
                int32x4_t t1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo1, X0), hi1, X1);
                long oj = (long)(j0 + jpair) * nb + b;
                acc0[jpair] += d0 * da[oj] * (float)(vaddvq_s32(t0) - 16 * asum[oj]);
                acc1[jpair] += d1 * da[oj] * (float)(vaddvq_s32(t1) - 16 * asum[oj]);
            }
        }
        for (int j = 0; j < jn; j++) {
            out[(long)(j0 + j) * m + row]     = acc0[j];
            out[(long)(j0 + j) * m + row + 1] = acc1[j];
        }
    }
    if (rpair < r1)
        nt_q5_0_sdot_range(out, m, W, qa, da, asum, rpair, r1, k, j0, jn);
}
#endif

static void nt_q5_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        if (jn >= 2 && (r1 - r0) >= 2) {
            nt_q5_0_rows_i8mm(out, m, W, qa, da, asum, r0, r1, k, j0, jn);
            continue;
        }
#endif
        nt_q5_0_sdot_range(out, m, W, qa, da, asum, r0, r1, k, j0, jn);
    }
}

// Q6_K batched: 210 B / 256 values as sixteen 16-value sub-blocks, each with its own int8
// scale. The unpack is the heaviest here — four vectors built from a nibble, a 2-bit top
// and a -32 bias, per quarter of the block — so it is the one that most wants doing once.
// The sixteen integer sums are kept per activation and drained in ascending sub-block
// order afterwards, which is the order the per-token kernel adds them in; accumulating
// them as they are produced would be the same arithmetic in a different order, and a
// different order is a different float.
#define NT_Q6_K_UNPACK(qlh, qhh, is, w1, w2, w3, w4)                                       \
    uint8x16_t la_ = vld1q_u8((qlh) + (is) * 16);                                          \
    uint8x16_t lb_ = vld1q_u8((qlh) + 32 + (is) * 16);                                     \
    uint8x16_t hv_ = vld1q_u8((qhh) + (is) * 16);                                          \
    int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(                                  \
        vandq_u8(la_, m4), vshlq_n_u8(vandq_u8(hv_, m3), 4))), b32);                       \
    int8x16_t w2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(                                  \
        vandq_u8(lb_, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv_, 2), m3), 4))), b32);        \
    int8x16_t w3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(                                  \
        vshrq_n_u8(la_, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv_, 4), m3), 4))), b32);       \
    int8x16_t w4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(                                  \
        vshrq_n_u8(lb_, 4), vshlq_n_u8(vshrq_n_u8(hv_, 6), 4))), b32)

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* Q6_K under SMMLA. Its sub-blocks are sixteen values, so each is two 2x8 tiles once a row
 * pair is stacked, and the four quarters of a half-block give eight tiles per (half, is).
 * The sixteen sums stay per row per activation and are drained ascending afterwards, which
 * is the per-token kernel's order and therefore the same float. */
static void nt_q6_k_rows_i8mm(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, int r0, int r1, int k, int j0, int jn) {
    int nb = k / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3);
    const int8x16_t  b32 = vdupq_n_s8(32);
    int rpair = r0 + ((r1 - r0) & ~1);
    int jpair = jn & ~1;
    for (int row = r0; row < rpair; row += 2) {
        const uint8_t *rb0 = W + (long)row * nb * 210;
        const uint8_t *rb1 = rb0 + (long)nb * 210;
        float acc0[NT_QMM_TILE], acc1[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b0 = rb0 + (long)blk * 210, *b1 = rb1 + (long)blk * 210;
            const int8_t *sc0 = (const int8_t *)(b0 + 192), *sc1 = (const int8_t *)(b1 + 192);
            float d0 = nt_f16_to_f32((uint16_t)(b0[208] | (b0[209] << 8)));
            float d1 = nt_f16_to_f32((uint16_t)(b1[208] | (b1[209] << 8)));
            int32_t ss0[NT_QMM_TILE][16], ss1[NT_QMM_TILE][16];
            for (int nn = 0; nn < 256; nn += 128) {
                const uint8_t *ql0 = b0 + (nn / 128) * 64, *qh0 = b0 + 128 + (nn / 128) * 32;
                const uint8_t *ql1 = b1 + (nn / 128) * 64, *qh1 = b1 + 128 + (nn / 128) * 32;
                int base = (nn / 128) * 8;
                for (int is = 0; is < 2; is++) {
                    int8x16_t u1, u2, u3, u4, v1, v2, v3, v4;
                    { NT_Q6_K_UNPACK(ql0, qh0, is, a1_, a2_, a3_, a4_);
                      u1 = a1_; u2 = a2_; u3 = a3_; u4 = a4_; }
                    { NT_Q6_K_UNPACK(ql1, qh1, is, a1_, a2_, a3_, a4_);
                      v1 = a1_; v2 = a2_; v3 = a3_; v4 = a4_; }
                    int8x16_t A[8] = {
                        vcombine_s8(vget_low_s8(u1),  vget_low_s8(v1)),
                        vcombine_s8(vget_high_s8(u1), vget_high_s8(v1)),
                        vcombine_s8(vget_low_s8(u2),  vget_low_s8(v2)),
                        vcombine_s8(vget_high_s8(u2), vget_high_s8(v2)),
                        vcombine_s8(vget_low_s8(u3),  vget_low_s8(v3)),
                        vcombine_s8(vget_high_s8(u3), vget_high_s8(v3)),
                        vcombine_s8(vget_low_s8(u4),  vget_low_s8(v4)),
                        vcombine_s8(vget_high_s8(u4), vget_high_s8(v4)),
                    };
                    for (int j = 0; j < jpair; j += 2) {
                        const int8_t *x0 = qa + (long)(j0 + j) * k + (long)blk * 256
                                         + nn + is * 16;
                        const int8_t *x1 = x0 + k;
                        for (int q = 0; q < 4; q++) {
                            const int8_t *y0 = x0 + q * 32, *y1 = x1 + q * 32;
                            int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A[2 * q],
                                                     vcombine_s8(vld1_s8(y0), vld1_s8(y1)));
                            s = vmmlaq_s32(s, A[2 * q + 1],
                                           vcombine_s8(vld1_s8(y0 + 8), vld1_s8(y1 + 8)));
                            int slot = base + is + 2 * q;
                            ss0[j][slot]     = vgetq_lane_s32(s, 0);
                            ss0[j + 1][slot] = vgetq_lane_s32(s, 1);
                            ss1[j][slot]     = vgetq_lane_s32(s, 2);
                            ss1[j + 1][slot] = vgetq_lane_s32(s, 3);
                        }
                    }
                    if (jpair < jn) {
                        const int8_t *x = qa + (long)(j0 + jpair) * k + (long)blk * 256
                                        + nn + is * 16;
                        const int32x4_t z = vdupq_n_s32(0);
                        int8x16_t U[4] = { u1, u2, u3, u4 }, V[4] = { v1, v2, v3, v4 };
                        for (int q = 0; q < 4; q++) {
                            int8x16_t xv = vld1q_s8(x + q * 32);
                            ss0[jpair][base + is + 2 * q] = vaddvq_s32(vdotq_s32(z, U[q], xv));
                            ss1[jpair][base + is + 2 * q] = vaddvq_s32(vdotq_s32(z, V[q], xv));
                        }
                    }
                }
            }
            for (int j = 0; j < jn; j++) {
                const float *dab = da + (long)(j0 + j) * (k / 32) + (long)blk * 8;
                for (int s = 0; s < 16; s++) {
                    acc0[j] = nt_q6k_acc(acc0[j], d0, (float)sc0[s], dab[s / 2], ss0[j][s]);
                    acc1[j] = nt_q6k_acc(acc1[j], d1, (float)sc1[s], dab[s / 2], ss1[j][s]);
                }
            }
        }
        for (int j = 0; j < jn; j++) {
            out[(long)(j0 + j) * m + row]     = acc0[j];
            out[(long)(j0 + j) * m + row + 1] = acc1[j];
        }
    }
    return;
}
#endif

static void nt_q6_k_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;
    int nb = k / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3);
    const int8x16_t  b32 = vdupq_n_s8(32);
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        int rp = r0 + ((r1 - r0) & ~1);
        if (jn >= 2 && rp > r0) {
            nt_q6_k_rows_i8mm(out, m, W, qa, da, r0, rp, k, j0, jn);
            if (rp >= r1) continue;
        }
        for (int row = (jn >= 2 ? r0 + ((r1 - r0) & ~1) : r0); row < r1; row++) {
#else
        for (int row = r0; row < r1; row++) {
#endif
            const uint8_t *rb = W + (long)row * nb * 210;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int blk = 0; blk < nb; blk++) {
                const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
                const int8_t *sc = (const int8_t *)(b + 192);
                float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
                int32_t ssum[NT_QMM_TILE][16];
                for (int nn = 0; nn < 256; nn += 128) {
                    const uint8_t *qlh = ql + (nn / 128) * 64, *qhh = qh + (nn / 128) * 32;
                    int base = (nn / 128) * 8;
                    for (int is = 0; is < 2; is++) {
                        uint8x16_t la = vld1q_u8(qlh + is * 16);
                        uint8x16_t lb = vld1q_u8(qlh + 32 + is * 16);
                        uint8x16_t hv = vld1q_u8(qhh + is * 16);
                        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(la, m4), vshlq_n_u8(vandq_u8(hv, m3), 4))), b32);
                        int8x16_t w2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(lb, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv, 2), m3), 4))), b32);
                        int8x16_t w3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(la, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hv, 4), m3), 4))), b32);
                        int8x16_t w4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(lb, 4), vshlq_n_u8(vshrq_n_u8(hv, 6), 4))), b32);
                        for (int j = 0; j < jn; j++) {
                            const int8_t *x = qa + (long)(j0 + j) * k + (long)blk * 256
                                            + nn + is * 16;
                            const int32x4_t z = vdupq_n_s32(0);
                            ssum[j][base + is + 0] = vaddvq_s32(vdotq_s32(z, w1, vld1q_s8(x)));
                            ssum[j][base + is + 2] = vaddvq_s32(vdotq_s32(z, w2, vld1q_s8(x + 32)));
                            ssum[j][base + is + 4] = vaddvq_s32(vdotq_s32(z, w3, vld1q_s8(x + 64)));
                            ssum[j][base + is + 6] = vaddvq_s32(vdotq_s32(z, w4, vld1q_s8(x + 96)));
                        }
                    }
                }
                for (int j = 0; j < jn; j++) {
                    const float *dab = da + (long)(j0 + j) * (k / 32) + (long)blk * 8;
                    for (int s = 0; s < 16; s++)
                        acc[j] = nt_q6k_acc(acc[j], d, (float)sc[s], dab[s / 2], ssum[j][s]);
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}

// Q8_0 batched: 34 B / 32 values, the weights already int8. Nothing to unpack, so what is
// saved here is the weight traffic itself — one pass over the row per tile instead of per
// token — and that is the whole reason prefill was reading 373 MiB per position.
static void nt_q8_0_sdot_range(float *out, int m, const uint8_t *W, const int8_t *qa,
                               const float *da, int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 34;
        float acc[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) acc[j] = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rb + (long)b * 34;
            float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            int8x16_t w0 = vld1q_s8((const int8_t *)(blk + 2));
            int8x16_t w1 = vld1q_s8((const int8_t *)(blk + 18));
            for (int j = 0; j < jn; j++) {
                const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                int32x4_t t = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(qab));
                t = vdotq_s32(t, w1, vld1q_s8(qab + 16));
                acc[j] += d * da[(long)(j0 + j) * nb + b] * (float)vaddvq_s32(t);
            }
        }
        for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void nt_q8_0_rows_i8mm(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, int r0, int r1, int k, int j0, int jn) {
    int nb = k / 32;
    int rpair = r0 + ((r1 - r0) & ~1);
    int jpair = jn & ~1;
    for (int row = r0; row < rpair; row += 2) {
        const uint8_t *rb0 = W + (long)row * nb * 34;
        const uint8_t *rb1 = rb0 + (long)nb * 34;
        float acc0[NT_QMM_TILE], acc1[NT_QMM_TILE];
        for (int j = 0; j < jn; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk0 = rb0 + (long)b * 34, *blk1 = rb1 + (long)b * 34;
            float d0 = nt_f16_to_f32((uint16_t)(blk0[0] | (blk0[1] << 8)));
            float d1 = nt_f16_to_f32((uint16_t)(blk1[0] | (blk1[1] << 8)));
            int8x16_t w00 = vld1q_s8((const int8_t *)(blk0 + 2));
            int8x16_t w01 = vld1q_s8((const int8_t *)(blk0 + 18));
            int8x16_t w10 = vld1q_s8((const int8_t *)(blk1 + 2));
            int8x16_t w11 = vld1q_s8((const int8_t *)(blk1 + 18));
            int8x16_t A0 = vcombine_s8(vget_low_s8(w00),  vget_low_s8(w10));
            int8x16_t A1 = vcombine_s8(vget_high_s8(w00), vget_high_s8(w10));
            int8x16_t A2 = vcombine_s8(vget_low_s8(w01),  vget_low_s8(w11));
            int8x16_t A3 = vcombine_s8(vget_high_s8(w01), vget_high_s8(w11));
            for (int j = 0; j < jpair; j += 2) {
                const int8_t *x0 = qa + (long)(j0 + j) * k + (long)b * 32;
                const int8_t *x1 = x0 + k;
                int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A0,
                                         vcombine_s8(vld1_s8(x0),      vld1_s8(x1)));
                s = vmmlaq_s32(s, A1, vcombine_s8(vld1_s8(x0 + 8),  vld1_s8(x1 + 8)));
                s = vmmlaq_s32(s, A2, vcombine_s8(vld1_s8(x0 + 16), vld1_s8(x1 + 16)));
                s = vmmlaq_s32(s, A3, vcombine_s8(vld1_s8(x0 + 24), vld1_s8(x1 + 24)));
                float da0 = da[(long)(j0 + j) * nb + b], da1 = da[(long)(j0 + j + 1) * nb + b];
                acc0[j]     += d0 * da0 * (float)vgetq_lane_s32(s, 0);
                acc0[j + 1] += d0 * da1 * (float)vgetq_lane_s32(s, 1);
                acc1[j]     += d1 * da0 * (float)vgetq_lane_s32(s, 2);
                acc1[j + 1] += d1 * da1 * (float)vgetq_lane_s32(s, 3);
            }
            if (jpair < jn) {
                const int8_t *x = qa + (long)(j0 + jpair) * k + (long)b * 32;
                int8x16_t X0 = vld1q_s8(x), X1 = vld1q_s8(x + 16);
                int32x4_t t0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w00, X0), w01, X1);
                int32x4_t t1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w10, X0), w11, X1);
                float daj = da[(long)(j0 + jpair) * nb + b];
                acc0[jpair] += d0 * daj * (float)vaddvq_s32(t0);
                acc1[jpair] += d1 * daj * (float)vaddvq_s32(t1);
            }
        }
        for (int j = 0; j < jn; j++) {
            out[(long)(j0 + j) * m + row]     = acc0[j];
            out[(long)(j0 + j) * m + row + 1] = acc1[j];
        }
    }
    if (rpair < r1)
        nt_q8_0_sdot_range(out, m, W, qa, da, rpair, r1, k, j0, jn);
}
#endif

static void nt_q8_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        if (jn >= 2 && (r1 - r0) >= 2) {
            nt_q8_0_rows_i8mm(out, m, W, qa, da, r0, r1, k, j0, jn);
            continue;
        }
#endif
        nt_q8_0_sdot_range(out, m, W, qa, da, r0, r1, k, j0, jn);
    }
}
#else
static void nt_q5_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    int nb = k / 32;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
        for (int row = r0; row < r1; row++) {
            const uint8_t *rb = W + (long)row * nb * 22;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = rb + (long)b * 22;
                float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                            | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
                const uint8_t *qs = blk + 6;
                int8_t wv[32];
                for (int i = 0; i < 16; i++) {
                    wv[i]      = (int8_t)((qs[i] & 0x0F) | (((qh >> i) & 1) << 4));
                    wv[i + 16] = (int8_t)((qs[i] >> 4)   | (((qh >> (i + 16)) & 1) << 4));
                }
                for (int j = 0; j < jn; j++) {
                    const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                    int32_t s = 0;
                    for (int i = 0; i < 32; i++) s += (int32_t)wv[i] * (int32_t)qab[i];
                    acc[j] += d * da[(long)(j0 + j) * nb + b]
                            * (float)(s - 16 * asum[(long)(j0 + j) * nb + b]);
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}

static void nt_q8_0_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;
    int nb = k / 32;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
        for (int row = r0; row < r1; row++) {
            const uint8_t *rb = W + (long)row * nb * 34;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = rb + (long)b * 34;
                float d = nt_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                const int8_t *wq = (const int8_t *)(blk + 2);
                for (int j = 0; j < jn; j++) {
                    const int8_t *qab = qa + (long)(j0 + j) * k + (long)b * 32;
                    int32_t s = 0;
                    for (int i = 0; i < 32; i++) s += (int32_t)wq[i] * (int32_t)qab[i];
                    acc[j] += d * da[(long)(j0 + j) * nb + b] * (float)s;
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}

static void nt_q6_k_rows_i8n(float *out, int m, const uint8_t *W, const int8_t *qa,
                             const float *da, const int32_t *asum,
                             int r0, int r1, int k, int n) {
    (void)asum;
    int nb = k / 256;
    for (int j0 = 0; j0 < n; j0 += NT_QMM_TILE) {
        int jn = n - j0; if (jn > NT_QMM_TILE) jn = NT_QMM_TILE;
        for (int row = r0; row < r1; row++) {
            const uint8_t *rb = W + (long)row * nb * 210;
            float acc[NT_QMM_TILE];
            for (int j = 0; j < jn; j++) acc[j] = 0.0f;
            for (int blk = 0; blk < nb; blk++) {
                const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
                const int8_t *sc = (const int8_t *)(b + 192);
                float d = nt_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
                int32_t ssum[NT_QMM_TILE][16];
                for (int j = 0; j < jn; j++)
                    for (int s = 0; s < 16; s++) ssum[j][s] = 0;
                for (int nn = 0; nn < 256; nn += 128) {
                    const uint8_t *qlh = ql + (nn / 128) * 64, *qhh = qh + (nn / 128) * 32;
                    int base = (nn / 128) * 8;
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                        for (int j = 0; j < jn; j++) {
                            const int8_t *qab = qa + (long)(j0 + j) * k + (long)blk * 256;
                            ssum[j][base + is + 0] += q1 * (int)qab[nn + l];
                            ssum[j][base + is + 2] += q2 * (int)qab[nn + l + 32];
                            ssum[j][base + is + 4] += q3 * (int)qab[nn + l + 64];
                            ssum[j][base + is + 6] += q4 * (int)qab[nn + l + 96];
                        }
                    }
                }
                for (int j = 0; j < jn; j++) {
                    const float *dab = da + (long)(j0 + j) * (k / 32) + (long)blk * 8;
                    for (int s = 0; s < 16; s++)
                        acc[j] = nt_q6k_acc(acc[j], d, (float)sc[s], dab[s / 2], ssum[j][s]);
                }
            }
            for (int j = 0; j < jn; j++) out[(long)(j0 + j) * m + row] = acc[j];
        }
    }
}
#endif

typedef void (*nt_qmmrows_fn)(float *out, int m, const uint8_t *W, const int8_t *qa,
                              const float *da, const int32_t *asum,
                              int r0, int r1, int k, int n);

/* Row chunks are pulled, not dealt out: on big.LITTLE a static split makes every matmul
 * wait for the slow cluster's share, and the pool above learned that the same way. */
typedef struct {
    nt_qmmrows_fn fn;
    float *out; const uint8_t *W; const int8_t *qa; const float *da; const int32_t *asum;
    int m, k, n, hi, chunk;
    int next;
} nt_qmm_job;

static void nt_qmm_drain(nt_qmm_job *j) {
    for (;;) {
        int r0 = __atomic_fetch_add(&j->next, j->chunk, __ATOMIC_RELAXED);
        if (r0 >= j->hi) return;
        int r1 = r0 + j->chunk; if (r1 > j->hi) r1 = j->hi;
        j->fn(j->out, j->m, j->W, j->qa, j->da, j->asum, r0, r1, j->k, j->n);
    }
}

static void *nt_qmm_worker(void *p) { nt_qmm_drain((nt_qmm_job *)p); return NULL; }

int nt_qmatmul_i8(float *out, const uint8_t *Wq, int dtype,
                  const float *X, int m, int k, int n) {
    if (m <= 0 || k <= 0 || n <= 0) return -1;
    if (n == 1) return nt_qmatvec_i8(out, Wq, dtype, X, m, k);
    if (k % 32) return -1;
    if ((dtype == 12 || dtype == 14) && (k % 256)) return -1;
    nt_qmmrows_fn fn;
    switch (dtype) {                     /* dtypes without a batched kernel go per token */
    case 2:  fn = nt_q4_0_rows_i8n; break;
    case 6:  fn = nt_q5_0_rows_i8n; break;
    case 8:  fn = nt_q8_0_rows_i8n; break;
    case 12: fn = nt_q4_k_rows_i8n; break;
    case 14: fn = nt_q6_k_rows_i8n; break;
    default: return -1;
    }

    int nsub = k / 32;
    int8_t *qa = (int8_t *)malloc((size_t)k * (size_t)n);
    float  *da = (float *)malloc((size_t)nsub * (size_t)n * sizeof(float));
    int32_t *asum = NULL;
    if (!qa || !da) { free(qa); free(da); return -1; }
    for (int j = 0; j < n; j++)
        nt_quant_act_q8(X + (long)j * k, k, qa + (long)j * k, da + (long)j * nsub);

    /* Q4_K's affine minimum and Q5_0's -16 bias both lift out of the integer dot as a
     * multiple of SUM(qa) per block. It depends on the activation alone, so every row
     * range and every tile reads the same numbers — computed once, here. */
    if (dtype == 12 || dtype == 6) {
        asum = (int32_t *)malloc((size_t)nsub * (size_t)n * sizeof(int32_t));
        if (!asum) { free(qa); free(da); return -1; }
        for (int j = 0; j < n; j++)
            for (int s = 0; s < nsub; s++) {
                const int8_t *p = qa + (long)j * k + (long)s * 32;
                int32_t t = 0;
                for (int i = 0; i < 32; i++) t += p[i];
                asum[(long)j * nsub + s] = t;
            }
    }

    /* The gate counts the work, and a batched call does n times the work of one matvec:
     * a 0.5B Qwen's query projection is 896x896, under the 4M floor and therefore
     * single-threaded per token, while the same projection over a chunk of 32 positions is
     * 25M and belongs on every core. Leaving n out of this comparison left most of a
     * prefill on one core and hid the batched kernel's own speedup behind it. */
    int nt = nt_qmv_host_threads(m);
    if (nt <= 1 || (long)m * k * (long)n < nt_qmv_thread_floor()) {
        fn(out, m, Wq, qa, da, asum, 0, m, k, n);
        free(qa); free(da); free(asum);
        return 0;
    }

    /* One fan-out per matmul, not per token: a 24-layer prefill opens seven of these per
     * layer instead of seven per layer per token, so pthread_create lands in the noise. */
    nt_qmm_job job = { fn, out, Wq, qa, da, asum, m, k, n, m, 0, 0 };
    job.chunk = m / (nt * 8); if (job.chunk < 1) job.chunk = 1;
    pthread_t th[NT_QMV_MAX_THREADS];
    int launched = 0;
    for (int t = 0; t + 1 < nt; t++) {
        if (pthread_create(&th[t], NULL, nt_qmm_worker, &job) != 0) break;
        launched++;
    }
    nt_qmm_drain(&job);                   /* the caller is a worker too */
    for (int t = 0; t < launched; t++) pthread_join(th[t], NULL);

    free(qa); free(da); free(asum);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE OPS — conv2d (im2col + GEMM) + group norm — forward-only inference ops
// for diffusion engines (Stable-Diffusion UNet/VAE). Companions to nt_qmatvec:
// pre-trained weights, no tape. The image-NN ops notorch lacked.
// ═══════════════════════════════════════════════════════════════════════════════

// nt_im2col — unfold [Cin,Hin,Win] into columns [Cin*kH*kW, Hout*Wout] so a
// convolution becomes a single GEMM. Out-of-range taps are zero (padding).
void nt_im2col(float *col, const float *in, int Cin, int Hin, int Win,
               int kH, int kW, int stride, int padding) {
    int Hout = (Hin + 2 * padding - kH) / stride + 1;
    int Wout = (Win + 2 * padding - kW) / stride + 1;
    int col_cols = Hout * Wout;
    for (int c = 0; c < Cin; c++)
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                size_t col_base = (size_t)row * col_cols;
                for (int oh = 0; oh < Hout; oh++)
                    for (int ow = 0; ow < Wout; ow++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;
                        float val = 0.0f;
                        if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win)
                            val = in[((size_t)c * Hin + ih) * Win + iw];
                        col[col_base + (size_t)oh * Wout + ow] = val;
                    }
            }
}

// nt_conv2d — out[Cout,Hout,Wout] = weight[Cout, Cin*kH*kW] @ im2col(in) + bias.
// weight is the standard [Cout,Cin,kH,kW] tensor row-major (== [Cout, Cin*kH*kW]).
// bias may be NULL. Returns 0, or -1 on bad geometry / allocation failure.
int nt_conv2d(float *out, const float *in, const float *weight, const float *bias,
              int Cin, int Hin, int Win, int Cout, int kH, int kW, int stride, int padding) {
    int Hout = (Hin + 2 * padding - kH) / stride + 1;
    int Wout = (Win + 2 * padding - kW) / stride + 1;
    if (Hout <= 0 || Wout <= 0) return -1;
    /* K and N are matmul dims for nt_blas_mm and must stay int; validate the
     * geometry products in a wide type first, so a wrapped int32 can't mis-size
     * the im2col buffer (Cin*kH*kW or Hout*Wout beyond INT_MAX is rejected). */
    long K_l = (long)Cin * kH * kW;
    long N_l = (long)Hout * Wout;
    if (Cin <= 0 || kH <= 0 || kW <= 0 || K_l > INT_MAX || N_l > INT_MAX) return -1;
    int K = (int)K_l;
    int N = (int)N_l;
    float *col = (float *)malloc((size_t)K * N * sizeof(float));
    if (!col) return -1;
    nt_im2col(col, in, Cin, Hin, Win, kH, kW, stride, padding);
    nt_blas_mm(out, weight, col, Cout, K, N);   /* [Cout,K] @ [K,N] -> [Cout,N] */
    if (bias) {
        for (int co = 0; co < Cout; co++) {
            float b = bias[co];
            float *op = out + (size_t)co * N;
            for (int n = 0; n < N; n++) op[n] += b;
        }
    }
    free(col);
    return 0;
}

// nt_group_norm — GroupNorm over [C,H,W]: split C into num_groups, normalize each
// group over (C/num_groups)*H*W, then per-channel affine (gamma/beta may be NULL).
// out may alias in. Returns 0, or -1 on bad args.
int nt_group_norm(float *out, const float *in, const float *gamma, const float *beta,
                  int C, int H, int W, int num_groups, float eps) {
    if (num_groups <= 0 || C % num_groups != 0) return -1;
    int gc = C / num_groups;
    int spatial = H * W;
    long count = (long)gc * spatial;
    if (count <= 0) return -1;
    for (int g = 0; g < num_groups; g++) {
        int c0 = g * gc;
        const float *base = in + (size_t)c0 * spatial;
        double sum = 0.0, sumsq = 0.0;
        for (long i = 0; i < count; i++) { double v = base[i]; sum += v; sumsq += v * v; }
        float mean = (float)(sum / count);
        float var = (float)(sumsq / count - (double)mean * mean);
        if (var < 0.0f) var = 0.0f;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = c0; c < c0 + gc; c++) {
            float wsc = (gamma ? gamma[c] : 1.0f) * inv;
            float wsh = (beta ? beta[c] : 0.0f) - mean * wsc;
            const float *ip = in + (size_t)c * spatial;
            float *op = out + (size_t)c * spatial;
            for (int s = 0; s < spatial; s++) op[s] = ip[s] * wsc + wsh;
        }
    }
    return 0;
}

// nt_upsample_nearest — nearest-neighbour upsample of [C,H,W] -> [C,H*scale,W*scale].
// The UNet decoder / VAE up-blocks upsample then convolve.
void nt_upsample_nearest(float *out, const float *in, int C, int H, int W, int scale) {
    int Ho = H * scale, Wo = W * scale;
    for (int c = 0; c < C; c++) {
        const float *ip = in + (size_t)c * H * W;
        float *op = out + (size_t)c * Ho * Wo;
        for (int oh = 0; oh < Ho; oh++) {
            const float *irow = ip + (size_t)(oh / scale) * W;
            float *orow = op + (size_t)oh * Wo;
            for (int ow = 0; ow < Wo; ow++) orow[ow] = irow[ow / scale];
        }
    }
}

// nt_attention — scaled dot-product attention (single head), forward inference.
// Q[T,d], K[S,d], V[S,d] -> out[T,d] = softmax(Q @ K^T / sqrt(d)) @ V. Self-attention:
// S == T (K,V from the same features). Cross-attention: S = context length (e.g. CLIP
// tokens) — the conditioning path of a diffusion UNet. -1 on bad args / alloc failure.
int nt_attention(float *out, const float *Q, const float *K, const float *V, int T, int S, int d) {
    if (T <= 0 || S <= 0 || d <= 0) return -1;
    float *scores = (float *)malloc((size_t)T * S * sizeof(float));
    if (!scores) return -1;
    nt_blas_mmT(scores, Q, K, T, d, S);          /* scores[T,S] = Q[T,d] @ K[S,d]^T */
    float scale = 1.0f / sqrtf((float)d);
    for (int t = 0; t < T; t++) {
        float *row = scores + (size_t)t * S;
        float mx = row[0] * scale;
        for (int s = 1; s < S; s++) { float v = row[s] * scale; if (v > mx) mx = v; }
        float sum = 0.0f;
        for (int s = 0; s < S; s++) { float e = expf(row[s] * scale - mx); row[s] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int s = 0; s < S; s++) row[s] *= inv;
    }
    nt_blas_mm(out, scores, V, T, S, d);          /* out[T,d] = scores[T,S] @ V[S,d] */
    free(scores);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH LoRA — low-rank adapter implementation
// ═══════════════════════════════════════════════════════════════════════════════
//
// Forward:  y = W @ x + (alpha/rank) * B @ (A @ x)
// A: [rank, in_dim]  → kaiming_uniform_(fan_in=in_dim) init
// B: [out_dim, rank] → zeros init  (so initial Δ output = 0)
// W is supplied externally (via existing tape entry), frozen via nt_tape_param_frozen().

#define NT_LORA_MAGIC   0x4C4F5241u  // 'LORA'
#define NT_LORA_VERSION 1u

int nt_lora_init(nt_lora_pair* lora, int in_dim, int out_dim, int rank, float alpha) {
    if (!lora || in_dim <= 0 || out_dim <= 0 || rank <= 0) return -1;
    int a_shape[2] = { rank, in_dim };
    int b_shape[2] = { out_dim, rank };
    lora->A = nt_tensor_new_shape(a_shape, 2);
    lora->B = nt_tensor_new_shape(b_shape, 2);
    if (!lora->A || !lora->B) {
        if (lora->A) nt_tensor_free(lora->A);
        if (lora->B) nt_tensor_free(lora->B);
        lora->A = NULL; lora->B = NULL;
        return -1;
    }
    nt_kaiming_uniform_init(lora->A, in_dim);
    // B already zero from nt_tensor_new_shape (calloc-backed), but be explicit:
    for (int i = 0; i < lora->B->len; i++) lora->B->data[i] = 0.0f;
    lora->rank    = rank;
    lora->alpha   = alpha;
    lora->scaling = alpha / (float)rank;
    lora->in_dim  = in_dim;
    lora->out_dim = out_dim;
    return 0;
}

void nt_lora_free(nt_lora_pair* lora) {
    if (!lora) return;
    if (lora->A) { nt_tensor_free(lora->A); lora->A = NULL; }
    if (lora->B) { nt_tensor_free(lora->B); lora->B = NULL; }
}

int nt_lora_forward(int w_idx, nt_lora_pair* lora, int x_idx, int T) {
    if (!lora || !lora->A || !lora->B) return -1;
    if (w_idx < 0 || x_idx < 0) return -1;
    // Register persistent A,B as trainable in this step's tape — Chuck slot
    // allocation for them happens here, AFTER any base nt_tape_param_frozen()
    // calls, so Chuck slot indices stay clean for the optimizer.
    int a_idx = nt_tape_param(lora->A);
    int b_idx = nt_tape_param(lora->B);
    if (a_idx < 0 || b_idx < 0) return -1;

    // Compose y = nt_seq_linear(w, x, T) + scaling * nt_seq_linear(b, nt_seq_linear(a, x, T), T)
    int wx_idx = nt_seq_linear(w_idx, x_idx, T);     // base: W @ x
    if (wx_idx < 0) return -1;
    int ax_idx = nt_seq_linear(a_idx, x_idx, T);     // A @ x  → [T, rank]
    if (ax_idx < 0) return -1;
    int bax_idx = nt_seq_linear(b_idx, ax_idx, T);   // B @ (A @ x)  → [T, out_dim]
    if (bax_idx < 0) return -1;
    int scaled_idx = nt_scale(bax_idx, lora->scaling);
    if (scaled_idx < 0) return -1;
    int y_idx = nt_add(wx_idx, scaled_idx);
    return y_idx;
}

// ── LoRA file I/O ──
//
// Format (little-endian, packed):
//   [u32 magic 0x4C4F5241 'LORA']
//   [u32 version=1]
//   [u32 num_targets]
//   [for each target T in [0,num_targets): u8 namelen, namelen × ascii bytes]
//   [u32 num_layers][u32 rank][u32 alpha_int (= (uint32_t)(alpha*1000))]
//   [u32 in_dim][u32 out_dim]
//   [for each layer L in [0,num_layers), for each target T in [0,num_targets):
//       A floats (rank × in_dim), B floats (out_dim × rank)]

int nt_lora_save(const nt_lora_pair* pairs, int num_layers, int num_targets,
                 const char* const* target_names, const char* path) {
    if (!pairs || !target_names || !path || num_layers <= 0 || num_targets <= 0) return -1;

    // Validate ALL pairs FIRST (per Codex notorch-pass-1 P2 #2 — fopen 'wb' truncates,
    // so checking dimensions before opening keeps any pre-existing checkpoint safe).
    int rank = pairs[0].rank;
    int in_dim = pairs[0].in_dim;
    int out_dim = pairs[0].out_dim;
    float alpha = pairs[0].alpha;
    for (int i = 0; i < num_layers * num_targets; i++) {
        if (pairs[i].rank != rank || pairs[i].in_dim != in_dim ||
            pairs[i].out_dim != out_dim || pairs[i].alpha != alpha) {
            return -1;  // fail before touching the destination file
        }
        if (!pairs[i].A || !pairs[i].B) return -1;
    }

    // Write to a temp file first, then atomically rename — guards against partial
    // writes leaving a corrupt checkpoint if the process is killed mid-save.
    char tmp_path[2048];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (n < 0 || n >= (int)sizeof(tmp_path)) return -1;
    FILE* f = fopen(tmp_path, "wb");
    if (!f) return -1;

    uint32_t magic = NT_LORA_MAGIC, version = NT_LORA_VERSION;
    uint32_t nt_targets = (uint32_t)num_targets;
    uint32_t nt_layers = (uint32_t)num_layers;
    uint32_t nt_rank = (uint32_t)rank;
    // Store alpha as raw float bytes (per Codex notorch-pass-1 P3 #4 — int-milli
    // is lossy for non-rounded alpha values like 16.5 / 13.7 / etc).
    union { float f; uint32_t u; } alpha_bits = { .f = alpha };
    uint32_t nt_alpha_bits = alpha_bits.u;
    uint32_t nt_in = (uint32_t)in_dim;
    uint32_t nt_out = (uint32_t)out_dim;

    if (fwrite(&magic, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&version, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_targets, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }

    for (int t = 0; t < num_targets; t++) {
        const char* name = target_names[t];
        size_t nl = name ? strlen(name) : 0;
        if (nl > 255) nl = 255;
        uint8_t bnl = (uint8_t)nl;
        if (fwrite(&bnl, 1, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
        if (nl > 0 && fwrite(name, 1, nl, f) != nl) { fclose(f); remove(tmp_path); return -1; }
    }

    if (fwrite(&nt_layers, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_rank, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_alpha_bits, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_in, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_out, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }

    int a_n = rank * in_dim;
    int b_n = out_dim * rank;
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < num_targets; T++) {
            const nt_lora_pair* p = &pairs[L * num_targets + T];
            // Sync GPU mirror to CPU before reading host buffer (per Codex Pass 3 P1 #5)
#ifdef USE_CUDA
            nt_tensor_ensure_cpu(p->A);
            nt_tensor_ensure_cpu(p->B);
#endif
            if ((int)fwrite(p->A->data, sizeof(float), a_n, f) != a_n) { fclose(f); remove(tmp_path); return -1; }
            if ((int)fwrite(p->B->data, sizeof(float), b_n, f) != b_n) { fclose(f); remove(tmp_path); return -1; }
        }
    }
    if (fflush(f) != 0) { fclose(f); remove(tmp_path); return -1; }
    fclose(f);
    // Atomic rename: only on Unix; on Windows rename(2) fails if dest exists.
    // For our pod-side flow this is Linux/macOS, so rename(2) is atomic.
    if (rename(tmp_path, path) != 0) { remove(tmp_path); return -1; }
    return 0;
}

int nt_lora_load(nt_lora_pair* pairs, int num_layers, int num_targets,
                 const char* const* target_names, const char* path) {
    if (!pairs || !target_names || !path || num_layers <= 0 || num_targets <= 0) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic, version, nt_targets, nt_layers, nt_rank, nt_in, nt_out;
    if (fread(&magic, 4, 1, f) != 1 || magic != NT_LORA_MAGIC) { fclose(f); return -1; }
    if (fread(&version, 4, 1, f) != 1 || version != NT_LORA_VERSION) { fclose(f); return -1; }
    if (fread(&nt_targets, 4, 1, f) != 1 || (int)nt_targets != num_targets) { fclose(f); return -1; }

    char buf[256];
    for (int t = 0; t < num_targets; t++) {
        uint8_t bnl;
        if (fread(&bnl, 1, 1, f) != 1) { fclose(f); return -1; }
        if (bnl > 0 && fread(buf, 1, bnl, f) != bnl) { fclose(f); return -1; }
        buf[bnl] = '\0';
        if (target_names[t] && strcmp(buf, target_names[t]) != 0) { fclose(f); return -1; }
    }

    if (fread(&nt_layers, 4, 1, f) != 1 || (int)nt_layers != num_layers) { fclose(f); return -1; }
    if (fread(&nt_rank, 4, 1, f) != 1) { fclose(f); return -1; }
    uint32_t nt_alpha_bits;
    if (fread(&nt_alpha_bits, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&nt_in, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&nt_out, 4, 1, f) != 1) { fclose(f); return -1; }

    int rank = (int)nt_rank, in_dim = (int)nt_in, out_dim = (int)nt_out;
    union { uint32_t u; float f; } alpha_bits = { .u = nt_alpha_bits };
    float alpha = alpha_bits.f;
    // Compare alpha with tolerance — float exact equality is brittle even when
    // load round-trips raw bytes (compiler / fp env may diverge).
    for (int i = 0; i < num_layers * num_targets; i++) {
        if (pairs[i].rank != rank || pairs[i].in_dim != in_dim ||
            pairs[i].out_dim != out_dim) {
            fclose(f); return -1;
        }
        float diff = pairs[i].alpha - alpha;
        if (diff < 0) diff = -diff;
        if (diff > 1e-4f) { fclose(f); return -1; }
    }

    int a_n = rank * in_dim, b_n = out_dim * rank;
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < num_targets; T++) {
            nt_lora_pair* p = &pairs[L * num_targets + T];
            if ((int)fread(p->A->data, sizeof(float), a_n, f) != a_n) { fclose(f); return -1; }
            if ((int)fread(p->B->data, sizeof(float), b_n, f) != b_n) { fclose(f); return -1; }
#ifdef USE_CUDA
            // Mark CPU mirror authoritative; next nt_tensor_ensure_gpu uploads.
            p->A->gpu_valid = 0;
            p->B->gpu_valid = 0;
#endif
        }
    }
    fclose(f);
    return 0;
}

void nt_lora_merge_into(float* W_dst, const float* W_frozen,
                        const nt_lora_pair* lora, int in_dim, int out_dim) {
    if (!W_dst || !W_frozen || !lora || !lora->A || !lora->B) return;
    if (in_dim <= 0 || out_dim <= 0) return;
    if (lora->in_dim != in_dim || lora->out_dim != out_dim) return;
#ifdef USE_CUDA
    nt_tensor_ensure_cpu(lora->A);
    nt_tensor_ensure_cpu(lora->B);
#endif
    int rank = lora->rank;
    float scale = lora->scaling;
    // W_dst[i,j] = W_frozen[i,j] + scale * sum_k B[i,k] * A[k,j]
    // Compute Δ = B @ A first (out × in), then add to W_frozen → W_dst.
    // Use existing nt_blas_mm: C[m,n] = A[m,k] @ B[k,n].
    float* delta = (float*)malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!delta) return;
    nt_blas_mm(delta, lora->B->data, lora->A->data, out_dim, rank, in_dim);
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            W_dst[i * in_dim + j] = W_frozen[i * in_dim + j] + scale * delta[i * in_dim + j];
        }
    }
    free(delta);
}
