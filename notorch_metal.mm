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
)MSL";

/* ── State (ARC-managed) ─────────────────────────────────────────────── */

static id<MTLDevice>               g_device      = nil;
static id<MTLCommandQueue>         g_queue       = nil;
static id<MTLComputePipelineState> g_q4k_pipe    = nil;
static id<MTLComputePipelineState> g_q6k_pipe    = nil;
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

    uint32_t k_u32 = (uint32_t)k;
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
