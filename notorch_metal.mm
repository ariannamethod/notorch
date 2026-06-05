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
)MSL";

/* ── State (ARC-managed) ─────────────────────────────────────────────── */

static id<MTLDevice>               g_device      = nil;
static id<MTLCommandQueue>         g_queue       = nil;
static id<MTLComputePipelineState> g_q4k_pipe    = nil;
static int                         g_initialised = 0;

/* Phase 2: resident zero-copy buffers wrapping the packed GGUF data block.
 * Segmented because a single MTLBuffer is capped at device.maxBufferLength
 * (~0.6x RAM) — below a 14GB+ 24B weight block. */
#define NT_MAX_SEG 16
static id<MTLBuffer>  g_seg_buf[NT_MAX_SEG] = { nil };
static const uint8_t *g_seg_ptr[NT_MAX_SEG] = { NULL };
static uint64_t       g_seg_len[NT_MAX_SEG] = { 0 };
static int            g_nseg = 0;

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
    }

    g_initialised = 1;
    return 0;
}

void nt_metal_shutdown(void)
{
    for (int s = 0; s < g_nseg; s++) g_seg_buf[s] = nil;
    g_nseg        = 0;
    g_q4k_pipe    = nil;
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
        const NSUInteger nblocks   = (NSUInteger)k / 256u;
        const NSUInteger row_bytes = nblocks * 144u;
        const NSUInteger W_bytes   = (NSUInteger)m * row_bytes;
        const NSUInteger x_bytes   = (NSUInteger)k * sizeof(float);
        const NSUInteger out_bytes = (NSUInteger)m * sizeof(float);

        /* Phase 2: if this weight lives inside the registered resident base,
         * bind it by offset (zero upload). Otherwise upload it for this call. */
        id<MTLBuffer> bW = nil; NSUInteger W_off = 0;
        for (int s = 0; s < g_nseg; s++) {
            if (W_q4k >= g_seg_ptr[s] &&
                (uint64_t)(W_q4k - g_seg_ptr[s]) + W_bytes <= g_seg_len[s]) {
                bW = g_seg_buf[s];
                W_off = (NSUInteger)(W_q4k - g_seg_ptr[s]);
                break;
            }
        }
        if (!bW) {   /* unregistered or straddles a segment boundary -> upload */
            bW = [g_device newBufferWithBytes:W_q4k
                                       length:W_bytes
                                      options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> bx = [g_device newBufferWithBytes:x
                                                 length:x_bytes
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [g_device newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
        uint32_t k_u32 = (uint32_t)k;
        id<MTLBuffer> bk = [g_device newBufferWithBytes:&k_u32
                                                 length:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        if (!bW || !bx || !bout || !bk) {
            fprintf(stderr, "nt_metal_q4k_matvec: buffer alloc failed\n");
            return 11;
        }

        id<MTLCommandBuffer>        cb  = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_q4k_pipe];
        [enc setBuffer:bW   offset:W_off atIndex:0];
        [enc setBuffer:bx   offset:0 atIndex:1];
        [enc setBuffer:bout offset:0 atIndex:2];
        [enc setBuffer:bk   offset:0 atIndex:3];

        NSUInteger tg_size = g_q4k_pipe.maxTotalThreadsPerThreadgroup;
        if (tg_size > 64) tg_size = 64;
        if (tg_size > (NSUInteger)m) tg_size = (NSUInteger)m;
        MTLSize grid = MTLSizeMake((NSUInteger)m, 1, 1);
        MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(out, [bout contents], (size_t)out_bytes);
    }

    return 0;
}
