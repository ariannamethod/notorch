/* notorch_metal.mm — Apple Silicon Metal/MSL backend for notorch.
 *
 * Implements the public C ABI from notorch_metal.h. Pure Obj-C++ — the
 * .mm extension is what triggers Obj-C++ compilation. We use an Obj-C++
 * raw string literal for the MSL kernel so the shader source lives
 * inline with the host code (one file, one read).
 *
 * Q4_K layout reference (GGML, identical to gguf.c:dequant_q4_k and
 * doe.c lines 941-973):
 *
 *   block = 144 bytes per 256 weights
 *     [0:2]    d     fp16   super-block scale
 *     [2:4]    dmin  fp16   super-block min
 *     [4:16]   sc    12B    packed 6-bit per-subblock scales+mins (8+8)
 *     [16:144] qs    128B   4-bit quants (256 nibbles, low-then-high)
 *
 * by Claude (Arianna Method)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "notorch_metal.h"

/* ── MSL kernel source ───────────────────────────────────────────────── */

static NSString * const kMetalKernelSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

/* Unpack the j-th 6-bit (scale,min) pair from the 12-byte `sc` table.
 * Mirrors gguf.c:get_scale_min_k4 byte-for-byte. */
inline void get_scale_min_k4(int j,
                             device const uchar *sc,
                             thread uchar &s,
                             thread uchar &m)
{
    if (j < 4) {
        s = sc[j]     & 63u;
        m = sc[j + 4] & 63u;
    } else {
        s = (sc[j + 4] & 0x0Fu) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4)    | ((sc[j]     >> 6) << 4);
    }
}

/* One thread per output row. Streams Q4_K blocks, dequants inline,
 * accumulates a single fp32 dot. */
kernel void q4k_matvec(
    device const uchar *W   [[buffer(0)]],   /* [m * (k/256) * 144] */
    device const float *x   [[buffer(1)]],   /* [k]                 */
    device       float *out [[buffer(2)]],   /* [m]                 */
    constant     uint  &k   [[buffer(3)]],
    uint i                  [[thread_position_in_grid]])
{
    uint nblocks   = k / 256u;
    uint row_bytes = nblocks * 144u;
    device const uchar *w_row = W + i * row_bytes;

    float acc = 0.0f;

    for (uint bi = 0; bi < nblocks; bi++) {
        device const uchar *b = w_row + bi * 144u;
        ushort dbits    = ushort(b[0]) | (ushort(b[1]) << 8);
        ushort dminbits = ushort(b[2]) | (ushort(b[3]) << 8);
        float  d        = float(as_type<half>(dbits));
        float  dmin     = float(as_type<half>(dminbits));
        device const uchar *sc = b + 4;
        device const uchar *qs = b + 16;
        device const float *xb = x + bi * 256u;

        int is = 0;
        int qi = 0;
        for (int jj = 0; jj < 256; jj += 64) {
            uchar sc0, m0, sc1, m1v;
            get_scale_min_k4(is,     sc, sc0, m0);
            get_scale_min_k4(is + 1, sc, sc1, m1v);
            float d1 = d * float(sc0); float mm1 = dmin * float(m0);
            float d2 = d * float(sc1); float mm2 = dmin * float(m1v);

            for (int l = 0; l < 32; l++) {
                float w_lo = d1 * float(qs[qi + l] & 0x0Fu) - mm1;
                acc += w_lo * xb[jj + l];
            }
            for (int l = 0; l < 32; l++) {
                float w_hi = d2 * float(qs[qi + l] >> 4) - mm2;
                acc += w_hi * xb[jj + 32 + l];
            }
            qi += 32;
            is += 2;
        }
    }

    out[i] = acc;
}

/* Q6_K: block = 210 bytes per 256 weights — 128 ql + 64 qh + 16 int8 scales + 2 d(fp16).
 * Mirrors doe.c:pq_q6_k_rows / dequant_q6_k byte-for-byte. One thread per output row. */
kernel void q6k_matvec(
    device const uchar *W   [[buffer(0)]],   /* [m * (k/256) * 210] */
    device const float *x   [[buffer(1)]],   /* [k]                 */
    device       float *out [[buffer(2)]],   /* [m]                 */
    constant     uint  &k   [[buffer(3)]],
    uint i                  [[thread_position_in_grid]])
{
    uint nblocks   = k / 256u;
    uint row_bytes = nblocks * 210u;
    device const uchar *w_row = W + i * row_bytes;
    float acc = 0.0f;

    for (uint bi = 0; bi < nblocks; bi++) {
        device const uchar *bl = w_row + bi * 210u;
        device const uchar *ql = bl;
        device const uchar *qh = bl + 128u;
        device const char  *sc = (device const char *)(bl + 192u);   /* int8 scales */
        ushort dbits = ushort(bl[208]) | (ushort(bl[209]) << 8);
        float  d  = float(as_type<half>(dbits));
        device const float *xb = x + bi * 256u;

        for (int nn = 0; nn < 256; nn += 128) {
            device const uchar *qlh = ql + (nn / 128) * 64;
            device const uchar *qhh = qh + (nn / 128) * 32;
            device const char  *sch = sc + (nn / 128) * 8;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((qlh[l]      & 0x0Fu) | (((qhh[l] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0Fu) | (((qhh[l] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((qlh[l]      >> 4)    | (((qhh[l] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4)    | (((qhh[l] >> 6) & 3u) << 4)) - 32;
                acc += d * float(sch[is + 0]) * float(q1) * xb[nn + l];
                acc += d * float(sch[is + 2]) * float(q2) * xb[nn + l + 32];
                acc += d * float(sch[is + 4]) * float(q3) * xb[nn + l + 64];
                acc += d * float(sch[is + 6]) * float(q4) * xb[nn + l + 96];
            }
        }
    }
    out[i] = acc;
}

/* ── M3 — simdgroup-cooperative matvecs (llama.cpp-class geometry) ──────
 * One SIMDGROUP (32 lanes) per output row; the lanes split WITHIN each
 * block (256 weights / 32 lanes = 8 weights per lane), all lanes walk the
 * blocks together — full lane utilization at any k, coalesced reads. A
 * simd_sum folds the 32 partials. Dispatch: grid (32, m), threadgroup
 * (32, NSG) — each y-line of the threadgroup is exactly one simdgroup.
 * The simd_sum tree is fixed for a fixed geometry, so runs are
 * bit-identical run-to-run; vs the naive kernel the reduction ORDER
 * differs, so agreement is tolerance-level (~1e-5 rel), not bitwise. */

kernel void q4k_matvec_sg(
    device const uchar *W   [[buffer(0)]],
    device const float *x   [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &k   [[buffer(3)]],
    uint2 tpig              [[thread_position_in_grid]],
    uint  lane              [[thread_index_in_simdgroup]])
{
    uint nblocks   = k / 256u;
    uint row_bytes = nblocks * 144u;
    device const uchar *w_row = W + tpig.y * row_bytes;

    float acc = 0.0f;
    for (uint bi = 0; bi < nblocks; bi++) {
        device const uchar *b = w_row + bi * 144u;
        ushort dbits    = ushort(b[0]) | (ushort(b[1]) << 8);
        ushort dminbits = ushort(b[2]) | (ushort(b[3]) << 8);
        float  d        = float(as_type<half>(dbits));
        float  dmin     = float(as_type<half>(dminbits));
        device const uchar *sc = b + 4;
        device const uchar *qs = b + 16;
        device const float *xb = x + bi * 256u;

        /* lane-indexed slice of the naive chunk loop: this lane owns byte
         * `lane` of every 32-byte chunk — w_lo at jj+lane, w_hi at
         * jj+32+lane; 4 chunks x 2 weights = 8 weights per lane. */
        int is = 0;
        int qi = 0;
        for (int jj = 0; jj < 256; jj += 64) {
            uchar sc0, m0, sc1, m1v;
            get_scale_min_k4(is,     sc, sc0, m0);
            get_scale_min_k4(is + 1, sc, sc1, m1v);
            float d1 = d * float(sc0); float mm1 = dmin * float(m0);
            float d2 = d * float(sc1); float mm2 = dmin * float(m1v);
            float w_lo = d1 * float(qs[qi + lane] & 0x0Fu) - mm1;
            acc += w_lo * xb[jj + lane];
            float w_hi = d2 * float(qs[qi + lane] >> 4) - mm2;
            acc += w_hi * xb[jj + 32 + lane];
            qi += 32;
            is += 2;
        }
    }
    float total = simd_sum(acc);
    if (lane == 0) out[tpig.y] = total;
}

kernel void q6k_matvec_sg(
    device const uchar *W   [[buffer(0)]],
    device const float *x   [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &k   [[buffer(3)]],
    uint2 tpig              [[thread_position_in_grid]],
    uint  lane              [[thread_index_in_simdgroup]])
{
    uint nblocks   = k / 256u;
    uint row_bytes = nblocks * 210u;
    device const uchar *w_row = W + tpig.y * row_bytes;

    float acc = 0.0f;
    for (uint bi = 0; bi < nblocks; bi++) {
        device const uchar *bl = w_row + bi * 210u;
        device const uchar *ql = bl;
        device const uchar *qh = bl + 128u;
        device const char  *sc = (device const char *)(bl + 192u);
        ushort dbits = ushort(bl[208]) | (ushort(bl[209]) << 8);
        float  d  = float(as_type<half>(dbits));
        device const float *xb = x + bi * 256u;

        /* lane-indexed slice of the naive l-loop: lane == l, 4 weights per
         * half (q1..q4), two halves = 8 weights per lane. */
        for (int nn = 0; nn < 256; nn += 128) {
            device const uchar *qlh = ql + (nn / 128) * 64;
            device const uchar *qhh = qh + (nn / 128) * 32;
            device const char  *sch = sc + (nn / 128) * 8;
            int is = (int)(lane / 16u);
            int q1 = (int)((qlh[lane]      & 0x0Fu) | (((qhh[lane] >> 0) & 3u) << 4)) - 32;
            int q2 = (int)((qlh[lane + 32] & 0x0Fu) | (((qhh[lane] >> 2) & 3u) << 4)) - 32;
            int q3 = (int)((qlh[lane]      >> 4)    | (((qhh[lane] >> 4) & 3u) << 4)) - 32;
            int q4 = (int)((qlh[lane + 32] >> 4)    | (((qhh[lane] >> 6) & 3u) << 4)) - 32;
            acc += d * float(sch[is + 0]) * float(q1) * xb[nn + lane];
            acc += d * float(sch[is + 2]) * float(q2) * xb[nn + lane + 32];
            acc += d * float(sch[is + 4]) * float(q3) * xb[nn + lane + 64];
            acc += d * float(sch[is + 6]) * float(q4) * xb[nn + lane + 96];
        }
    }
    float total = simd_sum(acc);
    if (lane == 0) out[tpig.y] = total;
}

/* ── M4 — layer ops: the CPU fence-posts move onto the GPU ──────────────
 * rmsnorm / rope / silu·mul / residual add / single-token attention over
 * the KV cache. With these plus the matvecs, a whole decode layer encodes
 * into one command buffer — the CPU stops being a fence-post between
 * every GPU op. All reductions use fixed trees (simd_sum + fixed
 * threadgroup ladders): bit-identical run-to-run. */

kernel void rmsnorm_f32(
    device const float *src [[buffer(0)]],
    device       float *dst [[buffer(1)]],
    device const float *w   [[buffer(2)]],
    constant     uint  &n   [[buffer(3)]],
    constant     float &eps [[buffer(4)]],
    uint tid                [[thread_position_in_threadgroup]],
    uint tgsz               [[threads_per_threadgroup]],
    uint sgid               [[simdgroup_index_in_threadgroup]],
    uint lane               [[thread_index_in_simdgroup]])
{
    threadgroup float partials[32];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += tgsz) acc += src[i] * src[i];
    float s = simd_sum(acc);
    if (lane == 0) partials[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (tgsz + 31u) / 32u;
    float total = 0.0f;
    if (sgid == 0) {
        float p = (lane < nsg) ? partials[lane] : 0.0f;
        total = simd_sum(p);
        if (lane == 0) partials[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rinv = rsqrt(partials[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tgsz) dst[i] = src[i] * rinv * w[i];
}

/* llama-style rotary: head h, pair (i, i+hd/2), angle = pos * theta^(-2i/hd).
 * One thread per pair, in place. */
kernel void rope_f32(
    device       float *v    [[buffer(0)]],
    constant     uint  &nh   [[buffer(1)]],
    constant     uint  &hd   [[buffer(2)]],
    constant     uint  &pos  [[buffer(3)]],
    constant     float &theta[[buffer(4)]],
    uint gid                 [[thread_position_in_grid]])
{
    uint half_hd = hd / 2u;
    if (gid >= nh * half_hd) return;
    uint h = gid / half_hd;
    uint i = gid % half_hd;
    float freq  = pow(theta, -2.0f * float(i) / float(hd));
    float angle = float(pos) * freq;
    float c = cos(angle), s = sin(angle);
    device float *p = v + h * hd;
    float x0 = p[i], x1 = p[i + half_hd];
    p[i]           = x0 * c - x1 * s;
    p[i + half_hd] = x0 * s + x1 * c;
}

kernel void silu_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device       float *dst  [[buffer(2)]],
    constant     uint  &n    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    dst[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

kernel void add_f32(
    device const float *a   [[buffer(0)]],
    device const float *b   [[buffer(1)]],
    device       float *dst [[buffer(2)]],
    constant     uint  &n   [[buffer(3)]],
    uint gid                [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = a[gid] + b[gid];
}

/* Single-token attention over the KV cache, one threadgroup per q-head.
 * Stage 1: 128 threads stride the positions — scores into threadgroup
 * memory; fixed-ladder max + expsum; normalize. Stage 2: thread d
 * accumulates sum_p P[p] * V[p][d]. GQA via gqa = n_q_heads / n_kv_heads.
 * Strides are in FLOATS. t_len <= 4096 (host-checked). */
typedef struct {
    uint  t_len, hd, gqa;
    uint  k_pos_stride, k_head_stride;
    uint  v_pos_stride, v_head_stride;
    float scale;
} AttnParams;

kernel void attn_decode_f32(
    device const float *q   [[buffer(0)]],   /* [n_q_heads][hd]  */
    device const float *K   [[buffer(1)]],
    device const float *V   [[buffer(2)]],
    device       float *out [[buffer(3)]],   /* [n_q_heads][hd]  */
    constant AttnParams &P  [[buffer(4)]],
    uint head               [[threadgroup_position_in_grid]],
    uint tid                [[thread_position_in_threadgroup]],
    uint tgsz               [[threads_per_threadgroup]],
    uint sgid               [[simdgroup_index_in_threadgroup]],
    uint lane               [[thread_index_in_simdgroup]])
{
    threadgroup float scores[4096];
    threadgroup float red[32];

    uint kvh = head / P.gqa;
    device const float *qh = q + head * P.hd;
    device const float *Kh = K + kvh * P.k_head_stride;
    device const float *Vh = V + kvh * P.v_head_stride;

    /* stage 1a: raw scores */
    for (uint p = tid; p < P.t_len; p += tgsz) {
        device const float *kp = Kh + p * P.k_pos_stride;
        float dot = 0.0f;
        for (uint d = 0; d < P.hd; d++) dot += qh[d] * kp[d];
        scores[p] = dot * P.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* stage 1b: max (fixed ladder: lanes -> simdgroups -> first sg) */
    float lmax = -3.4e38f;
    for (uint p = tid; p < P.t_len; p += tgsz) lmax = max(lmax, scores[p]);
    lmax = simd_max(lmax);
    if (lane == 0) red[sgid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (tgsz + 31u) / 32u;
    if (sgid == 0) {
        float m = (lane < nsg) ? red[lane] : -3.4e38f;
        m = simd_max(m);
        if (lane == 0) red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = red[0];

    /* stage 1c: exp + sum */
    float lsum = 0.0f;
    for (uint p = tid; p < P.t_len; p += tgsz) {
        float e = exp(scores[p] - gmax);
        scores[p] = e;
        lsum += e;
    }
    lsum = simd_sum(lsum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sgid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float s = (lane < nsg) ? red[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsum = 1.0f / red[0];

    /* stage 2: out[d] = sum_p P[p] * V[p][d] */
    for (uint d = tid; d < P.hd; d += tgsz) {
        float acc = 0.0f;
        for (uint p = 0; p < P.t_len; p++)
            acc += scores[p] * Vh[p * P.v_pos_stride + d];
        out[head * P.hd + d] = acc * rsum;
    }
}


kernel void copy_f32(
    device const float *src [[buffer(0)]],
    device       float *dst [[buffer(1)]],
    constant     uint  &n   [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid];
}

)MSL";

/* ── State (ARC-managed) ─────────────────────────────────────────────── */

static id<MTLDevice>               g_device      = nil;
static id<MTLCommandQueue>         g_queue       = nil;
static id<MTLComputePipelineState> g_q4k_pipe    = nil;
static id<MTLComputePipelineState> g_q6k_pipe    = nil;
static id<MTLComputePipelineState> g_q4k_sg_pipe = nil;   /* M3 simdgroup path */
static id<MTLComputePipelineState> g_q6k_sg_pipe = nil;
static int                         g_use_sg      = 1;     /* NT_METAL_NAIVE=1 -> 0 */
static id<MTLComputePipelineState> g_rms_pipe    = nil;   /* M4 layer ops */
static id<MTLComputePipelineState> g_rope_pipe   = nil;
static id<MTLComputePipelineState> g_silu_pipe   = nil;
static id<MTLComputePipelineState> g_add_pipe    = nil;
static id<MTLComputePipelineState> g_attn_pipe   = nil;
static id<MTLComputePipelineState> g_copy_pipe   = nil;

/* M4 — device-resident activation slots: fixed regions in a persistent
 * GPU arena. Ops read/write slots without host roundtrips, so a whole
 * decode layer chains inside one command buffer. Slots survive batch
 * commits; upload/download are the only host crossings. */
#define NT_SLOT_MAX 64
static id<MTLBuffer> g_slot_buf  = nil;
static NSUInteger    g_slot_cap  = 0, g_slot_used = 0;
static struct { NSUInteger off, bytes; int live; } g_slot_tab[NT_SLOT_MAX];
static int                         g_initialised = 0;

/* Phase 2: resident zero-copy buffers wrapping the packed GGUF data block.
 * Segmented because a single MTLBuffer is capped at device.maxBufferLength
 * (~0.6x RAM) — below a 14GB+ 24B weight block. */
#define NT_MAX_SEG 16
static id<MTLBuffer>  g_seg_buf[NT_MAX_SEG] = { nil };
static const uint8_t *g_seg_ptr[NT_MAX_SEG] = { NULL };
static uint64_t       g_seg_len[NT_MAX_SEG] = { 0 };
static int            g_nseg = 0;

/* M1 — persistent Shared arenas, bump-allocated: `in` holds x uploads (GPU
 * reads), `out` holds matvec results (GPU writes, host copies back after
 * the wait). Kills the per-call newBufferWithBytes / newBufferWithLength
 * churn. Suballocations are 256-byte aligned — a safe setBuffer:offset:
 * on every GPU family. */
static id<MTLBuffer> g_arena_in  = nil;
static NSUInteger    g_in_cap  = 0, g_in_off  = 0;
static id<MTLBuffer> g_arena_out = nil;
static NSUInteger    g_out_cap = 0, g_out_off = 0;

/* M2 — token-graph batch: one command buffer collects many matvec
 * dispatches — ONE commit + ONE waitUntilCompleted per batch instead of
 * one per call. Results land in arena regions, copied to their host
 * destinations at commit (or at a transparent mid-batch flush when an
 * arena or the pending table fills). */
#define NT_BATCH_MAX 256
typedef struct { float *dst; NSUInteger off, bytes; } NTPendingOut;
static int                          g_batch_active = 0;
static id<MTLCommandBuffer>         g_batch_cb     = nil;
static id<MTLComputeCommandEncoder> g_batch_enc    = nil;
static NTPendingOut                 g_pending[NT_BATCH_MAX];
static int                          g_npending     = 0;

/* ── API ─────────────────────────────────────────────────────────────── */

int nt_metal_available(void)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil ? 1 : 0;
    }
}

int nt_metal_init(void)
{
    if (g_initialised) return 0;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "nt_metal_init: no Metal device on this host\n");
            return 1;
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "nt_metal_init: failed to create command queue\n");
            return 2;
        }

        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kMetalKernelSrc
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            fprintf(stderr, "nt_metal_init: kernel compile failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error)");
            return 3;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4k_matvec"];
        if (!fn) {
            fprintf(stderr, "nt_metal_init: kernel q4k_matvec missing\n");
            return 4;
        }
        g_q4k_pipe = [g_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_q4k_pipe) {
            fprintf(stderr, "nt_metal_init: pipeline state failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error)");
            return 5;
        }
        id<MTLFunction> fn6 = [lib newFunctionWithName:@"q6k_matvec"];
        if (!fn6) {
            fprintf(stderr, "nt_metal_init: kernel q6k_matvec missing\n");
            return 4;
        }
        g_q6k_pipe = [g_device newComputePipelineStateWithFunction:fn6 error:&err];
        if (!g_q6k_pipe) {
            fprintf(stderr, "nt_metal_init: q6k pipeline state failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error)");
            return 5;
        }
        id<MTLFunction> fn4s = [lib newFunctionWithName:@"q4k_matvec_sg"];
        id<MTLFunction> fn6s = [lib newFunctionWithName:@"q6k_matvec_sg"];
        if (!fn4s || !fn6s) {
            fprintf(stderr, "nt_metal_init: simdgroup kernels missing\n");
            return 4;
        }
        g_q4k_sg_pipe = [g_device newComputePipelineStateWithFunction:fn4s error:&err];
        g_q6k_sg_pipe = [g_device newComputePipelineStateWithFunction:fn6s error:&err];
        if (!g_q4k_sg_pipe || !g_q6k_sg_pipe) {
            fprintf(stderr, "nt_metal_init: simdgroup pipeline state failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error)");
            return 5;
        }
        /* M4 layer-op pipelines */
        struct { NSString *name; id<MTLComputePipelineState> __strong *slot; } m4ops[] = {
            { @"rmsnorm_f32",     &g_rms_pipe  },
            { @"rope_f32",        &g_rope_pipe },
            { @"silu_mul_f32",    &g_silu_pipe },
            { @"add_f32",         &g_add_pipe  },
            { @"attn_decode_f32", &g_attn_pipe },
            { @"copy_f32",        &g_copy_pipe },
        };
        for (size_t oi = 0; oi < sizeof(m4ops)/sizeof(m4ops[0]); oi++) {
            id<MTLFunction> f = [lib newFunctionWithName:m4ops[oi].name];
            if (!f) { fprintf(stderr, "nt_metal_init: kernel %s missing\n", m4ops[oi].name.UTF8String); return 4; }
            *m4ops[oi].slot = [g_device newComputePipelineStateWithFunction:f error:&err];
            if (!*m4ops[oi].slot) {
                fprintf(stderr, "nt_metal_init: %s pipeline failed: %s\n", m4ops[oi].name.UTF8String,
                        err ? err.localizedDescription.UTF8String : "(no error)");
                return 5;
            }
        }
        /* Default = naive one-thread-per-row. The 24B doe A/B on M4 Pro
         * (2026-06-12) measured sg at -23% on the real mixed-shape decode
         * stream (280 matvecs/token, attn k/v down to 1024x5120) despite
         * x1.81 on the square resident microbench. NT_METAL_SG=1 opts in
         * for geometry tuning; NT_METAL_NAIVE=1 still forces naive. */
        g_use_sg = getenv("NT_METAL_SG") ? 1 : 0;
        if (getenv("NT_METAL_NAIVE")) g_use_sg = 0;
    }

    g_initialised = 1;
    return 0;
}

void nt_metal_shutdown(void)
{
    if (g_batch_enc) [g_batch_enc endEncoding];   /* abort any open batch */
    g_batch_cb     = nil;
    g_batch_enc    = nil;
    g_batch_active = 0;
    g_npending     = 0;
    g_arena_in  = nil; g_in_cap  = 0; g_in_off  = 0;
    g_arena_out = nil; g_out_cap = 0; g_out_off = 0;
    for (int s = 0; s < g_nseg; s++) g_seg_buf[s] = nil;
    g_nseg        = 0;
    g_q4k_pipe    = nil;
    g_q6k_pipe    = nil;
    g_q4k_sg_pipe = nil;
    g_q6k_sg_pipe = nil;
    g_rms_pipe = nil; g_rope_pipe = nil; g_silu_pipe = nil;
    g_add_pipe = nil; g_attn_pipe = nil; g_copy_pipe = nil;
    g_slot_buf = nil; g_slot_cap = 0; g_slot_used = 0;
    memset(g_slot_tab, 0, sizeof(g_slot_tab));
    g_queue       = nil;
    g_device      = nil;
    g_initialised = 0;
}

int nt_metal_register_base(const void *base, uint64_t nbytes)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    uint64_t pg    = (uint64_t)getpagesize();
    uint64_t chunk = (uint64_t)g_device.maxBufferLength & ~(pg - 1);  /* page-floored cap */
    if (chunk == 0) return 12;
    g_nseg = 0;
    @autoreleasepool {
        uint64_t off = 0;
        while (off < nbytes && g_nseg < NT_MAX_SEG) {
            uint64_t len = nbytes - off;
            if (len > chunk) len = chunk;   /* len stays a page multiple: nbytes,off,chunk all are */
            id<MTLBuffer> b = [g_device newBufferWithBytesNoCopy:(void *)((const uint8_t *)base + off)
                                                         length:(NSUInteger)len
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
            if (!b) {
                fprintf(stderr, "nt_metal_register_base: NoCopy seg failed "
                                "(off=%llu len=%llu maxBufferLength=%llu)\n",
                        (unsigned long long)off, (unsigned long long)len,
                        (unsigned long long)g_device.maxBufferLength);
                g_nseg = 0;
                return 12;
            }
            g_seg_buf[g_nseg] = b;
            g_seg_ptr[g_nseg] = (const uint8_t *)base + off;
            g_seg_len[g_nseg] = len;
            g_nseg++;
            off += len;
        }
        if (off < nbytes) { g_nseg = 0; return 13; }  /* exceeded NT_MAX_SEG */
    }
    return 0;
}

/* ── M1/M2 — scratch arenas + token-graph batch (state above) ────────── */

static int arena_grow(id<MTLBuffer> __strong *buf, NSUInteger *cap, NSUInteger need)
{
    NSUInteger c = *cap ? *cap : (NSUInteger)1 << 20;
    while (c < need) c <<= 1;
    id<MTLBuffer> nb = [g_device newBufferWithLength:c
                                             options:MTLResourceStorageModeShared];
    if (!nb) { fprintf(stderr, "nt_metal: arena grow to %lu failed\n", (unsigned long)c); return 11; }
    *buf = nb;
    *cap = c;
    return 0;
}

static int batch_open_cb(void)
{
    g_batch_cb  = [g_queue commandBuffer];
    g_batch_enc = g_batch_cb ? [g_batch_cb computeCommandEncoder] : nil;
    if (!g_batch_cb || !g_batch_enc) {
        fprintf(stderr, "nt_metal: batch encoder alloc failed\n");
        g_batch_cb = nil; g_batch_enc = nil;
        return 11;
    }
    return 0;
}

/* Commit the in-flight batch encoder, wait once, drain every pending out
 * region to its host destination, reset the arenas. */
static int batch_drain(void)
{
    if (!g_batch_enc) return 0;
    [g_batch_enc endEncoding];
    [g_batch_cb commit];
    [g_batch_cb waitUntilCompleted];
    int rc = 0;
    if (g_batch_cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "nt_metal: batch command buffer not completed status=%ld error=%s\n",
                (long)g_batch_cb.status,
                g_batch_cb.error ? [g_batch_cb.error.localizedDescription UTF8String] : "(none)");
        rc = 14;
    } else {
        const uint8_t *ob = (const uint8_t *)[g_arena_out contents];
        for (int i = 0; i < g_npending; i++)
            memcpy(g_pending[i].dst, ob + g_pending[i].off, (size_t)g_pending[i].bytes);
    }
    g_batch_cb = nil; g_batch_enc = nil;
    g_npending = 0;
    g_in_off = 0; g_out_off = 0;
    return rc;
}

/* Shared encode path for both quant kernels, solo and batch modes. The
 * kernels and dispatch geometry are UNTOUCHED relative to the per-call
 * path they replace — results stay bit-identical. */
static int encode_matvec(id<MTLComputePipelineState> pipe, NSUInteger block_bytes,
                         float *out, const uint8_t *W, const float *x, int m, int k)
{
    const NSUInteger nblocks   = (NSUInteger)k / 256u;
    const NSUInteger row_bytes = nblocks * block_bytes;
    const NSUInteger W_bytes   = (NSUInteger)m * row_bytes;
    const NSUInteger x_bytes   = (NSUInteger)k * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)m * sizeof(float);

    /* Resident weight: bind by offset inside a registered segment (zero
     * upload). Unregistered W uploads for this call (tests, small tensors). */
    id<MTLBuffer> bW = nil; NSUInteger W_off = 0;
    for (int s = 0; s < g_nseg; s++) {
        if (W >= g_seg_ptr[s] &&
            (uint64_t)(W - g_seg_ptr[s]) + W_bytes <= g_seg_len[s]) {
            bW = g_seg_buf[s];
            W_off = (NSUInteger)(W - g_seg_ptr[s]);
            break;
        }
    }
    if (!bW) {
        bW = [g_device newBufferWithBytes:W length:W_bytes
                                  options:MTLResourceStorageModeShared];
        if (!bW) { fprintf(stderr, "nt_metal: W upload alloc failed\n"); return 11; }
    }

    /* Arena capacity. Growing reallocates the MTLBuffer, which is only
     * safe with no encoded-but-uncommitted work referencing it — a live
     * batch is drained (one extra sync) before any grow or reset. */
    NSUInteger in_need  = ((g_in_off  + 255u) & ~(NSUInteger)255u) + x_bytes;
    NSUInteger out_need = ((g_out_off + 255u) & ~(NSUInteger)255u) + out_bytes;
    if (in_need > g_in_cap || out_need > g_out_cap ||
        (g_batch_active && g_npending >= NT_BATCH_MAX)) {
        if (g_batch_active) {
            int rc = batch_drain(); if (rc) return rc;
            rc = batch_open_cb();   if (rc) return rc;
        } else { g_in_off = 0; g_out_off = 0; }
        if (x_bytes   > g_in_cap  && arena_grow(&g_arena_in,  &g_in_cap,  x_bytes))   return 11;
        if (out_bytes > g_out_cap && arena_grow(&g_arena_out, &g_out_cap, out_bytes)) return 11;
        if (!g_arena_in  && arena_grow(&g_arena_in,  &g_in_cap,  x_bytes))   return 11;
        if (!g_arena_out && arena_grow(&g_arena_out, &g_out_cap, out_bytes)) return 11;
    }

    NSUInteger x_off = (g_in_off  + 255u) & ~(NSUInteger)255u;
    NSUInteger o_off = (g_out_off + 255u) & ~(NSUInteger)255u;
    g_in_off  = x_off + x_bytes;
    g_out_off = o_off + out_bytes;
    memcpy((uint8_t *)[g_arena_in contents] + x_off, x, (size_t)x_bytes);

    id<MTLCommandBuffer>         cb  = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    if (g_batch_active) {
        if (!g_batch_enc) { int rc = batch_open_cb(); if (rc) return rc; }
        enc = g_batch_enc;
    } else {
        cb  = [g_queue commandBuffer];
        enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) { fprintf(stderr, "nt_metal: encoder alloc failed\n"); return 11; }
    }

    /* M3: simdgroup path by default — one 32-lane simdgroup per row, grid
     * (32, m), each threadgroup y-line is one simdgroup. NT_METAL_NAIVE=1
     * keeps the one-thread-per-row reference geometry. */
    id<MTLComputePipelineState> sg_pipe =
        (block_bytes == 144u) ? g_q4k_sg_pipe : g_q6k_sg_pipe;
    uint32_t k_u32 = (uint32_t)k;
    if (g_use_sg && sg_pipe) {
        [enc setComputePipelineState:sg_pipe];
        [enc setBuffer:bW          offset:W_off atIndex:0];
        [enc setBuffer:g_arena_in  offset:x_off atIndex:1];
        [enc setBuffer:g_arena_out offset:o_off atIndex:2];
        [enc setBytes:&k_u32 length:sizeof(uint32_t) atIndex:3];
        NSUInteger nsg = sg_pipe.maxTotalThreadsPerThreadgroup / 32u;
        if (nsg > 8) nsg = 8;
        if (nsg < 1) nsg = 1;
        if (nsg > (NSUInteger)m) nsg = (NSUInteger)m;
        MTLSize grid = MTLSizeMake(32, (NSUInteger)m, 1);
        MTLSize tg   = MTLSizeMake(32, nsg, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    } else {
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bW          offset:W_off atIndex:0];
        [enc setBuffer:g_arena_in  offset:x_off atIndex:1];
        [enc setBuffer:g_arena_out offset:o_off atIndex:2];
        [enc setBytes:&k_u32 length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg_size = pipe.maxTotalThreadsPerThreadgroup;
        if (tg_size > 64) tg_size = 64;
        if (tg_size > (NSUInteger)m) tg_size = (NSUInteger)m;
        MTLSize grid = MTLSizeMake((NSUInteger)m, 1, 1);
        MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    if (g_batch_active) {
        g_pending[g_npending].dst   = out;
        g_pending[g_npending].off   = o_off;
        g_pending[g_npending].bytes = out_bytes;
        g_npending++;
        return 0;
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "nt_metal: command buffer not completed status=%ld error=%s\n",
                (long)cb.status,
                cb.error ? [cb.error.localizedDescription UTF8String] : "(none)");
        return 14;
    }
    memcpy(out, (const uint8_t *)[g_arena_out contents] + o_off, (size_t)out_bytes);
    g_in_off = 0; g_out_off = 0;   /* solo call complete — arenas fully reusable */
    return 0;
}

int nt_metal_q4k_matvec(float *out,
                        const uint8_t *W_q4k,
                        const float *x,
                        int m, int k)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    if (k <= 0 || (k % 256) != 0) {
        fprintf(stderr, "nt_metal_q4k_matvec: k=%d not a positive multiple of 256\n", k);
        return 10;
    }
    if (m <= 0) {
        fprintf(stderr, "nt_metal_q4k_matvec: m=%d must be positive\n", m);
        return 10;
    }
    @autoreleasepool {
        return encode_matvec(g_q4k_pipe, 144u, out, W_q4k, x, m, k);
    }
}

int nt_metal_q6k_matvec(float *out,
                        const uint8_t *W_q6k,
                        const float *x,
                        int m, int k)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    if (k <= 0 || (k % 256) != 0) {
        fprintf(stderr, "nt_metal_q6k_matvec: k=%d not a positive multiple of 256\n", k);
        return 10;
    }
    if (m <= 0) {
        fprintf(stderr, "nt_metal_q6k_matvec: m=%d must be positive\n", m);
        return 10;
    }
    @autoreleasepool {
        return encode_matvec(g_q6k_pipe, 210u, out, W_q6k, x, m, k);
    }
}

int nt_metal_batch_begin(void)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    if (g_batch_active) return 0;          /* idempotent */
    @autoreleasepool {
        g_batch_active = 1;
        g_npending = 0;
        g_in_off = 0; g_out_off = 0;
        int rc = batch_open_cb();
        if (rc) g_batch_active = 0;
        return rc;
    }
}

int nt_metal_batch_commit(void)
{
    if (!g_batch_active) return 0;         /* commit without begin: no-op */
    int rc;
    @autoreleasepool {
        rc = batch_drain();
    }
    g_batch_active = 0;
    return rc;
}

int nt_metal_batch_active(void)
{
    return g_batch_active;
}

/* ── M4 — slots + layer ops ──────────────────────────────────────────── */

/* Append a region to the registered-segment table WITHOUT resetting it
 * (nt_metal_register_base resets — weights block; this appends — KV cache
 * and friends). base must be page-aligned, nbytes a page multiple. */
int nt_metal_register_region(const void *base, uint64_t nbytes)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    uint64_t pg = (uint64_t)getpagesize();
    if (((uintptr_t)base & (pg - 1)) || (nbytes & (pg - 1))) {
        fprintf(stderr, "nt_metal_register_region: base/len not page-aligned\n");
        return 12;
    }
    uint64_t chunk = (uint64_t)g_device.maxBufferLength & ~(pg - 1);
    @autoreleasepool {
        uint64_t off = 0;
        while (off < nbytes && g_nseg < NT_MAX_SEG) {
            uint64_t len = nbytes - off;
            if (len > chunk) len = chunk;
            id<MTLBuffer> b = [g_device newBufferWithBytesNoCopy:(void *)((const uint8_t *)base + off)
                                                         length:(NSUInteger)len
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
            if (!b) { fprintf(stderr, "nt_metal_register_region: NoCopy failed\n"); return 12; }
            g_seg_buf[g_nseg] = b;
            g_seg_ptr[g_nseg] = (const uint8_t *)base + off;
            g_seg_len[g_nseg] = len;
            g_nseg++;
            off += len;
        }
        if (off < nbytes) return 13;
    }
    return 0;
}

/* Resolve a host pointer to a registered (buffer, offset). 0 = found. */
static int resolve_region(const void *p, uint64_t bytes,
                          id<MTLBuffer> __strong *buf, NSUInteger *off)
{
    for (int s = 0; s < g_nseg; s++) {
        if ((const uint8_t *)p >= g_seg_ptr[s] &&
            (uint64_t)((const uint8_t *)p - g_seg_ptr[s]) + bytes <= g_seg_len[s]) {
            *buf = g_seg_buf[s];
            *off = (NSUInteger)((const uint8_t *)p - g_seg_ptr[s]);
            return 0;
        }
    }
    return 1;
}

int nt_metal_slot_alloc(int slot, uint64_t bytes)
{
    if (!g_initialised) {
        int rc = nt_metal_init();
        if (rc != 0) return rc;
    }
    if (slot < 0 || slot >= NT_SLOT_MAX) return 20;
    if (g_slot_tab[slot].live && g_slot_tab[slot].bytes >= bytes) return 0;  /* idempotent */
    if (g_slot_tab[slot].live) return 21;     /* grow of a live slot: not supported */
    @autoreleasepool {
        NSUInteger need = ((g_slot_used + 255u) & ~(NSUInteger)255u) + (NSUInteger)bytes;
        if (need > g_slot_cap) {
            /* growing reallocates: drain any batch, then copy live contents */
            if (g_batch_active) { int rc = batch_drain(); if (rc) return rc; rc = batch_open_cb(); if (rc) return rc; }
            NSUInteger cap = g_slot_cap ? g_slot_cap : (NSUInteger)1 << 20;
            while (cap < need) cap <<= 1;
            id<MTLBuffer> nb = [g_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            if (!nb) return 11;
            if (g_slot_buf && g_slot_used)
                memcpy([nb contents], [g_slot_buf contents], (size_t)g_slot_used);
            g_slot_buf = nb;
            g_slot_cap = cap;
        }
        NSUInteger off = (g_slot_used + 255u) & ~(NSUInteger)255u;
        g_slot_tab[slot].off   = off;
        g_slot_tab[slot].bytes = (NSUInteger)bytes;
        g_slot_tab[slot].live  = 1;
        g_slot_used = off + (NSUInteger)bytes;
    }
    return 0;
}

int nt_metal_slot_upload(int slot, const void *src, uint64_t bytes)
{
    if (slot < 0 || slot >= NT_SLOT_MAX || !g_slot_tab[slot].live ||
        bytes > g_slot_tab[slot].bytes) return 20;
    memcpy((uint8_t *)[g_slot_buf contents] + g_slot_tab[slot].off, src, (size_t)bytes);
    return 0;
}

/* Read a slot back. Call OUTSIDE an open batch (commit first) — pending
 * GPU writes to the slot land only at commit. */
int nt_metal_slot_download(int slot, void *dst, uint64_t bytes)
{
    if (slot < 0 || slot >= NT_SLOT_MAX || !g_slot_tab[slot].live ||
        bytes > g_slot_tab[slot].bytes) return 20;
    memcpy(dst, (const uint8_t *)[g_slot_buf contents] + g_slot_tab[slot].off, (size_t)bytes);
    return 0;
}

/* Open an encoder for one op: the live batch one, or a fresh solo cb. */
static int op_enc(id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *enc)
{
    if (g_batch_active) {
        if (!g_batch_enc) { int rc = batch_open_cb(); if (rc) return rc; }
        *cb = nil; *enc = g_batch_enc;
        return 0;
    }
    *cb  = [g_queue commandBuffer];
    *enc = *cb ? [*cb computeCommandEncoder] : nil;
    if (!*cb || !*enc) { fprintf(stderr, "nt_metal: op encoder alloc failed\n"); return 11; }
    return 0;
}

/* Finish one op: solo waits + checks status; batch returns immediately. */
static int op_fin(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc)
{
    if (g_batch_active) return 0;
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "nt_metal: op command buffer not completed status=%ld error=%s\n",
                (long)cb.status, cb.error ? [cb.error.localizedDescription UTF8String] : "(none)");
        return 14;
    }
    return 0;
}

static int slot_ok(int s) { return s >= 0 && s < NT_SLOT_MAX && g_slot_tab[s].live; }

int nt_metal_rmsnorm(int dst_slot, int src_slot, const float *w, int n, float eps)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(dst_slot) || !slot_ok(src_slot) || n <= 0) return 20;
    @autoreleasepool {
        id<MTLBuffer> bw = nil; NSUInteger w_off = 0;
        if (resolve_region(w, (uint64_t)n * sizeof(float), &bw, &w_off) != 0) {
            bw = [g_device newBufferWithBytes:w length:(NSUInteger)n * sizeof(float)
                                      options:MTLResourceStorageModeShared];
            if (!bw) return 11;
        }
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        uint32_t n_u = (uint32_t)n;
        [enc setComputePipelineState:g_rms_pipe];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[src_slot].off atIndex:0];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[dst_slot].off atIndex:1];
        [enc setBuffer:bw offset:w_off atIndex:2];
        [enc setBytes:&n_u length:4 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        MTLSize tg = MTLSizeMake(1024, 1, 1);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tg];
        return op_fin(cb, enc);
    }
}

int nt_metal_rope(int slot, int n_heads, int head_dim, int pos, float theta)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(slot) || n_heads <= 0 || head_dim <= 0 || (head_dim & 1)) return 20;
    @autoreleasepool {
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        uint32_t nh = (uint32_t)n_heads, hd = (uint32_t)head_dim, ps = (uint32_t)pos;
        [enc setComputePipelineState:g_rope_pipe];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[slot].off atIndex:0];
        [enc setBytes:&nh length:4 atIndex:1];
        [enc setBytes:&hd length:4 atIndex:2];
        [enc setBytes:&ps length:4 atIndex:3];
        [enc setBytes:&theta length:4 atIndex:4];
        NSUInteger total = (NSUInteger)n_heads * (NSUInteger)(head_dim / 2);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(total < 256 ? total : 256, 1, 1)];
        return op_fin(cb, enc);
    }
}

static int elementwise2(id<MTLComputePipelineState> pipe, int dst, int a, int b, int n)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(dst) || !slot_ok(a) || !slot_ok(b) || n <= 0) return 20;
    @autoreleasepool {
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        uint32_t n_u = (uint32_t)n;
        [enc setComputePipelineState:pipe];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[a].off atIndex:0];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[b].off atIndex:1];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[dst].off atIndex:2];
        [enc setBytes:&n_u length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(n < 256 ? (NSUInteger)n : 256, 1, 1)];
        return op_fin(cb, enc);
    }
}

int nt_metal_silu_mul(int dst_slot, int gate_slot, int up_slot, int n)
{ return elementwise2(g_silu_pipe, dst_slot, gate_slot, up_slot, n); }

int nt_metal_add(int dst_slot, int a_slot, int b_slot, int n)
{ return elementwise2(g_add_pipe, dst_slot, a_slot, b_slot, n); }

int nt_metal_attn_decode(int dst_slot, int q_slot, const float *K, const float *V,
                         int t_len, int n_q_heads, int n_kv_heads, int head_dim,
                         uint32_t k_pos_stride, uint32_t k_head_stride,
                         uint32_t v_pos_stride, uint32_t v_head_stride, float scale)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(dst_slot) || !slot_ok(q_slot)) return 20;
    if (t_len <= 0 || t_len > 4096) {
        fprintf(stderr, "nt_metal_attn_decode: t_len=%d out of range (1..4096)\n", t_len);
        return 20;
    }
    if (n_kv_heads <= 0 || n_q_heads % n_kv_heads) return 20;
    @autoreleasepool {
        /* K/V must live in registered regions — too big to upload per call */
        id<MTLBuffer> bK = nil, bV = nil; NSUInteger K_off = 0, V_off = 0;
        uint64_t k_span = ((uint64_t)(n_kv_heads - 1) * k_head_stride +
                           (uint64_t)(t_len - 1) * k_pos_stride + head_dim) * sizeof(float);
        uint64_t v_span = ((uint64_t)(n_kv_heads - 1) * v_head_stride +
                           (uint64_t)(t_len - 1) * v_pos_stride + head_dim) * sizeof(float);
        if (resolve_region(K, k_span, &bK, &K_off) != 0 ||
            resolve_region(V, v_span, &bV, &V_off) != 0) {
            fprintf(stderr, "nt_metal_attn_decode: K/V not in a registered region\n");
            return 22;
        }
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        struct { uint32_t t_len, hd, gqa, kps, khs, vps, vhs; float scale; } P = {
            (uint32_t)t_len, (uint32_t)head_dim,
            (uint32_t)(n_q_heads / n_kv_heads),
            k_pos_stride, k_head_stride, v_pos_stride, v_head_stride, scale
        };
        [enc setComputePipelineState:g_attn_pipe];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[q_slot].off  atIndex:0];
        [enc setBuffer:bK offset:K_off atIndex:1];
        [enc setBuffer:bV offset:V_off atIndex:2];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[dst_slot].off atIndex:3];
        [enc setBytes:&P length:sizeof(P) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_q_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return op_fin(cb, enc);
    }
}

/* Copy a slot into a registered host region GPU-side (KV append inside a
 * batch). dst must be float-aligned inside a registered region. */
int nt_metal_copy_to_region(void *dst, int src_slot, uint64_t bytes)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(src_slot) || bytes == 0 || (bytes & 3)) return 20;
    @autoreleasepool {
        id<MTLBuffer> bD = nil; NSUInteger D_off = 0;
        if (resolve_region(dst, bytes, &bD, &D_off) != 0) {
            fprintf(stderr, "nt_metal_copy_to_region: dst not registered\n");
            return 22;
        }
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        uint32_t n_u = (uint32_t)(bytes / 4);
        [enc setComputePipelineState:g_copy_pipe];
        [enc setBuffer:g_slot_buf offset:g_slot_tab[src_slot].off atIndex:0];
        [enc setBuffer:bD offset:D_off atIndex:1];
        [enc setBytes:&n_u length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_u, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(n_u < 256 ? (NSUInteger)n_u : 256, 1, 1)];
        return op_fin(cb, enc);
    }
}

/* Slot-resident matvecs: x from a slot, out to a slot — chain links for
 * the layer graph. Same kernels and geometry as the host-pointer path. */
static int matvec_slot(id<MTLComputePipelineState> naive_pipe,
                       id<MTLComputePipelineState> sg_pipe,
                       NSUInteger block_bytes,
                       int dst_slot, const uint8_t *W, int src_slot, int m, int k)
{
    if (!g_initialised) { int rc = nt_metal_init(); if (rc) return rc; }
    if (!slot_ok(dst_slot) || !slot_ok(src_slot)) return 20;
    if (k <= 0 || (k % 256) != 0 || m <= 0) return 10;
    @autoreleasepool {
        const NSUInteger W_bytes = (NSUInteger)m * ((NSUInteger)k / 256u) * block_bytes;
        id<MTLBuffer> bW = nil; NSUInteger W_off = 0;
        if (resolve_region(W, W_bytes, &bW, &W_off) != 0) {
            bW = [g_device newBufferWithBytes:W length:W_bytes
                                      options:MTLResourceStorageModeShared];
            if (!bW) return 11;
        }
        id<MTLCommandBuffer> cb; id<MTLComputeCommandEncoder> enc;
        int rc = op_enc(&cb, &enc); if (rc) return rc;
        uint32_t k_u32 = (uint32_t)k;
        if (g_use_sg && sg_pipe) {
            [enc setComputePipelineState:sg_pipe];
            [enc setBuffer:bW offset:W_off atIndex:0];
            [enc setBuffer:g_slot_buf offset:g_slot_tab[src_slot].off atIndex:1];
            [enc setBuffer:g_slot_buf offset:g_slot_tab[dst_slot].off atIndex:2];
            [enc setBytes:&k_u32 length:4 atIndex:3];
            NSUInteger nsg = sg_pipe.maxTotalThreadsPerThreadgroup / 32u;
            if (nsg > 8) nsg = 8;
            if (nsg < 1) nsg = 1;
            if (nsg > (NSUInteger)m) nsg = (NSUInteger)m;
            [enc dispatchThreads:MTLSizeMake(32, (NSUInteger)m, 1)
           threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        } else {
            [enc setComputePipelineState:naive_pipe];
            [enc setBuffer:bW offset:W_off atIndex:0];
            [enc setBuffer:g_slot_buf offset:g_slot_tab[src_slot].off atIndex:1];
            [enc setBuffer:g_slot_buf offset:g_slot_tab[dst_slot].off atIndex:2];
            [enc setBytes:&k_u32 length:4 atIndex:3];
            NSUInteger tg = naive_pipe.maxTotalThreadsPerThreadgroup;
            if (tg > 64) tg = 64;
            if (tg > (NSUInteger)m) tg = (NSUInteger)m;
            [enc dispatchThreads:MTLSizeMake((NSUInteger)m, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        return op_fin(cb, enc);
    }
}

int nt_metal_q4k_matvec_slot(int dst_slot, const uint8_t *W, int src_slot, int m, int k)
{ return matvec_slot(g_q4k_pipe, g_q4k_sg_pipe, 144u, dst_slot, W, src_slot, m, k); }

int nt_metal_q6k_matvec_slot(int dst_slot, const uint8_t *W, int src_slot, int m, int k)
{ return matvec_slot(g_q6k_pipe, g_q6k_sg_pipe, 210u, dst_slot, W, src_slot, m, k); }
