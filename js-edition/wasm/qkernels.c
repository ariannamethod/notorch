/* qkernels.c — the packed matvec with real SIMD, for the JS edition.
 *
 * JS inference is compute-bound: the same kernel costs 0.556 ns per element on
 * a 306 KB working set and 0.569 ns on 19 MB, so memory tricks buy nothing and
 * the only levers are fewer operations and more of them at once. Plain JS has
 * no way to issue sixteen products in one instruction. WebAssembly does, and
 * that instruction — i16x8.extmul over i8x16, folded by extadd_pairwise — is
 * the whole reason this file exists.
 *
 * The activation is quantized to per-32-block int8 exactly as C does
 * (nt_quant_act_q8, notorch.c:5505), including the round-half-to-even that
 * lrintf performs and that Math.round does not. Approximate by construction,
 * like its C counterpart: qmatvec stays the exact reference.
 *
 * Freestanding on purpose — no libc, no allocator, no imports. Everything it
 * touches is a pointer into the one linear memory the host owns, so the host
 * can make that memory shared and hand the same bytes to a worker pool.
 *
 * Build: see build.sh in this directory.
 */
#include <wasm_simd128.h>

typedef unsigned char u8;
typedef signed char   i8;
typedef unsigned short u16;
typedef unsigned int  u32;

/* ── f16 → f32, by bit reconstruction (no libm here, and none wanted) ── */
static inline float h2f(u16 h) {
    unsigned sign = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0) {
        if (m == 0) bits = sign << 31;
        else {
            e = 113;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            bits = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        bits = (sign << 31) | (0xFFu << 23) | (m << 13);
    } else {
        bits = (sign << 31) | ((e + 112) << 23) | (m << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

/* Round half to even, matching lrintf under the default rounding mode. */
static inline int rint_even(float v) {
    float f = __builtin_floorf(v);
    float d = v - f;
    int fi = (int)f;
    if (d > 0.5f) return fi + 1;
    if (d < 0.5f) return fi;
    return (fi & 1) ? fi + 1 : fi;
}

/* ── activation quant: per-32 symmetric int8, absmax scale ── */
__attribute__((export_name("quant_act")))
void quant_act(const float *x, int k, i8 *qa, float *da) {
    int nb = k / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = __builtin_fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        da[b] = d;
        for (int i = 0; i < 32; i++) {
            int q = rint_even(xb[i] * id);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            qa[b * 32 + i] = (i8)q;
        }
    }
}

/* Sum of sixteen i8 products, in one pass through the SIMD unit. */
static inline v128_t dot16(v128_t w, v128_t a) {
    v128_t lo = wasm_i16x8_extmul_low_i8x16(w, a);
    v128_t hi = wasm_i16x8_extmul_high_i8x16(w, a);
    return wasm_i32x4_add(wasm_i32x4_extadd_pairwise_i16x8(lo),
                          wasm_i32x4_extadd_pairwise_i16x8(hi));
}

static inline int hsum(v128_t v) {
    return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1)
         + wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

/* Q8_0: 34 B / 32 values — f16 scale then 32 raw int8. Nothing to unpack, so
 * the whole block is two SIMD loads and one dot16 pair. */
static void q8_0_rows(float *out, const u8 *W, const i8 *qa, const float *da,
                      int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const u8 *rb = W + (long)row * nb * 34;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const u8 *blk = rb + b * 34;
            float dw = h2f((u16)(blk[0] | (blk[1] << 8)));
            v128_t w0 = wasm_v128_load(blk + 2), w1 = wasm_v128_load(blk + 18);
            v128_t a0 = wasm_v128_load(qa + b * 32), a1 = wasm_v128_load(qa + b * 32 + 16);
            v128_t s = wasm_i32x4_add(dot16(w0, a0), dot16(w1, a1));
            acc += dw * da[b] * (float)hsum(s);
        }
        out[row] = acc;
    }
}

/* Q4_0: 18 B / 32 values. Byte i carries element i in the low nibble and i+16
 * in the high one, each biased by -8. Unpacking is a mask, a shift and a
 * subtract — all of it vector work. */
static void q4_0_rows(float *out, const u8 *W, const i8 *qa, const float *da,
                      int r0, int r1, int k) {
    int nb = k / 32;
    const v128_t mask = wasm_i8x16_splat(0x0F);
    const v128_t bias = wasm_i8x16_splat(8);
    for (int row = r0; row < r1; row++) {
        const u8 *rb = W + (long)row * nb * 18;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const u8 *blk = rb + b * 18;
            float dw = h2f((u16)(blk[0] | (blk[1] << 8)));
            v128_t packed = wasm_v128_load(blk + 2);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(packed, mask), bias);
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(packed, 4), bias);
            v128_t a0 = wasm_v128_load(qa + b * 32), a1 = wasm_v128_load(qa + b * 32 + 16);
            v128_t s = wasm_i32x4_add(dot16(lo, a0), dot16(hi, a1));
            acc += dw * da[b] * (float)hsum(s);
        }
        out[row] = acc;
    }
}

/* Q5_0: 22 B / 32 values — the fifth bit of each value lives in a 32-bit mask.
 * The reconstructed q lands in [0,31], which is int8-safe, so the -16 lifts out
 * of the dot as 16*SUM(activation) exactly as it does in C. */
static void q5_0_rows(float *out, const u8 *W, const i8 *qa, const float *da,
                      int r0, int r1, int k, int *asum) {
    int nb = k / 32;
    for (int b = 0; b < nb; b++) {
        const i8 *p = qa + b * 32;
        int t = 0;
        for (int i = 0; i < 32; i++) t += p[i];
        asum[b] = t;
    }
    for (int row = r0; row < r1; row++) {
        const u8 *rb = W + (long)row * nb * 22;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const u8 *blk = rb + b * 22;
            float dw = h2f((u16)(blk[0] | (blk[1] << 8)));
            u32 qh = (u32)blk[2] | ((u32)blk[3] << 8) | ((u32)blk[4] << 16) | ((u32)blk[5] << 24);
            i8 lo[16], hi[16];
            const u8 *qs = blk + 6;
            for (int j = 0; j < 16; j++) {
                lo[j] = (i8)((qs[j] & 0x0F) | (((qh >> j) & 1u) << 4));
                hi[j] = (i8)((qs[j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
            }
            v128_t a0 = wasm_v128_load(qa + b * 32), a1 = wasm_v128_load(qa + b * 32 + 16);
            v128_t s = wasm_i32x4_add(dot16(wasm_v128_load(lo), a0),
                                      dot16(wasm_v128_load(hi), a1));
            acc += dw * da[b] * (float)(hsum(s) - 16 * asum[b]);
        }
        out[row] = acc;
    }
}

/* The 6-bit (scale, min) pair for one Q4_K sub-block, packed across 12 bytes
 * (nt_get_scale_min_k4, notorch.c:5105). */
static inline void scale_min_k4(int j, const u8 *sc, int *s, int *mn) {
    if (j < 4) { *s = sc[j] & 63; *mn = sc[j + 4] & 63; }
    else { *s  = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *mn = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4); }
}

/* Q4_K: 144 B / 256 values — d, dmin (f16), 12 B of packed 6-bit scales and
 * mins, then 128 nibble bytes. Sub-blocks 2p and 2p+1 share one 32-byte span:
 * the low nibbles feed the even one, the high nibbles the odd one, which is
 * why the unpack is a mask or a shift and never a table.
 *
 * The affine minimum lifts out of the dot the way Q5_0's -16 does. A value is
 * d*s6*q - dmin*m6, so the sum over a sub-block is d*s6*SUM(q*a) - dmin*m6*
 * SUM(a), and the integer loop only ever sees raw nibbles in [0,15]. */
static void q4_k_rows(float *out, const u8 *W, const i8 *qa, const float *da,
                      int r0, int r1, int k, int *asum) {
    int nb = k / 256, nsub = k / 32;
    for (int s = 0; s < nsub; s++) {
        const i8 *p = qa + s * 32;
        int t = 0;
        for (int i = 0; i < 32; i++) t += p[i];
        asum[s] = t;
    }
    const v128_t mask = wasm_i8x16_splat(0x0F);
    for (int row = r0; row < r1; row++) {
        const u8 *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const u8 *b = rb + (long)blk * 144;
            float d    = h2f((u16)(b[0] | (b[1] << 8)));
            float dmin = h2f((u16)(b[2] | (b[3] << 8)));
            const u8 *sc = b + 4, *qs = b + 16;
            for (int j = 0; j < 8; j++) {
                int s6, m6;
                scale_min_k4(j, sc, &s6, &m6);
                const u8 *qsp = qs + (j >> 1) * 32;
                int sub = blk * 8 + j;
                const i8 *ap = qa + (long)sub * 32;
                v128_t p0 = wasm_v128_load(qsp), p1 = wasm_v128_load(qsp + 16);
                v128_t w0, w1;
                if (j & 1) { w0 = wasm_u8x16_shr(p0, 4);       w1 = wasm_u8x16_shr(p1, 4); }
                else       { w0 = wasm_v128_and(p0, mask);     w1 = wasm_v128_and(p1, mask); }
                v128_t s = wasm_i32x4_add(dot16(w0, wasm_v128_load(ap)),
                                          dot16(w1, wasm_v128_load(ap + 16)));
                acc += da[sub] * (d * (float)s6 * (float)hsum(s)
                                  - dmin * (float)m6 * (float)asum[sub]);
            }
        }
        out[row] = acc;
    }
}

/* Q6_K: 210 B / 256 values — 128 B of low nibbles, 64 B holding two high bits
 * per value, 16 signed sub-scales, then d. A value is (ql | qh<<4) - 32, which
 * lands in [-32,31] and stays int8-safe, so the whole reconstruction is vector
 * work and the dot never leaves the integer unit.
 *
 * A sub-scale covers 16 values and an activation block covers 32, so 16j..16j+15
 * always sits inside activation block j/2 and never straddles one. That is what
 * lets the integer accumulator be per weight sub-block, with d*sc[j]*da[j/2]
 * applied once at the end instead of per value. */
static void q6_k_rows(float *out, const u8 *W, const i8 *qa, const float *da,
                      int r0, int r1, int k) {
    int nb = k / 256;
    const v128_t m4 = wasm_i8x16_splat(0x0F);
    const v128_t m3 = wasm_i8x16_splat(0x03);
    const v128_t b32 = wasm_i8x16_splat(32);
    for (int row = r0; row < r1; row++) {
        const u8 *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const u8 *b = rb + (long)blk * 210;
            const u8 *ql = b, *qh = b + 128, *sc = b + 192;
            float d = h2f((u16)(b[208] | (b[209] << 8)));
            const i8 *ab = qa + (long)blk * 256;
            const float *dab = da + blk * 8;
            int ssum[16];
            for (int i = 0; i < 16; i++) ssum[i] = 0;
            for (int half = 0; half < 2; half++) {
                const u8 *qlh = ql + half * 64, *qhh = qh + half * 32;
                int base = half * 8, n = half * 128;
                for (int is = 0; is < 2; is++) {
                    int h16 = is * 16;
                    v128_t lo   = wasm_v128_load(qlh + h16);
                    v128_t lo32 = wasm_v128_load(qlh + h16 + 32);
                    v128_t hb   = wasm_v128_load(qhh + h16);
                    v128_t w1 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(lo, m4),
                        wasm_i8x16_shl(wasm_v128_and(hb, m3), 4)), b32);
                    v128_t w2 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(lo32, m4),
                        wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(hb, 2), m3), 4)), b32);
                    v128_t w3 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(lo, 4),
                        wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(hb, 4), m3), 4)), b32);
                    v128_t w4 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(lo32, 4),
                        wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(hb, 6), m3), 4)), b32);
                    const i8 *ap = ab + n + h16;
                    ssum[base + is]     += hsum(dot16(w1, wasm_v128_load(ap)));
                    ssum[base + is + 2] += hsum(dot16(w2, wasm_v128_load(ap + 32)));
                    ssum[base + is + 4] += hsum(dot16(w3, wasm_v128_load(ap + 64)));
                    ssum[base + is + 6] += hsum(dot16(w4, wasm_v128_load(ap + 96)));
                }
            }
            /* Ascending sub-block, the order the per-token kernel adds them in.
             * Folding each into the accumulator as it appeared would be the same
             * arithmetic in a different order, and a different order is a
             * different float. */
            for (int j = 0; j < 16; j++)
                acc += d * (float)(i8)sc[j] * dab[j >> 1] * (float)ssum[j];
        }
        out[row] = acc;
    }
}

/* Scratch the host reserves for us: quantized activation, its scales, and the
 * per-block activation sums Q5_0 and Q4_K need. Offsets are handed in per call
 * so the host owns the layout and nothing here allocates. */
__attribute__((export_name("qmatvec_i8")))
int qmatvec_i8(float *out, const u8 *W, int dtype, const float *x,
               int m, int k, i8 *qa, float *da, int *asum) {
    if (k % 32) return -1;
    if ((dtype == 12 || dtype == 14) && (k % 256)) return -1;
    quant_act(x, k, qa, da);
    switch (dtype) {
    case 2:  q4_0_rows(out, W, qa, da, 0, m, k); return 0;
    case 6:  q5_0_rows(out, W, qa, da, 0, m, k, asum); return 0;
    case 8:  q8_0_rows(out, W, qa, da, 0, m, k); return 0;
    case 12: q4_k_rows(out, W, qa, da, 0, m, k, asum); return 0;
    case 14: q6_k_rows(out, W, qa, da, 0, m, k); return 0;
    default: return -1;
    }
}

/* Row range, for a caller that owns its own parallel region. */
__attribute__((export_name("qmatvec_i8_rows")))
int qmatvec_i8_rows(float *out, const u8 *W, int dtype, const i8 *qa,
                    const float *da, int r0, int r1, int k, int *asum) {
    if (k % 32) return -1;
    if ((dtype == 12 || dtype == 14) && (k % 256)) return -1;
    switch (dtype) {
    case 2:  q4_0_rows(out, W, qa, da, r0, r1, k); return 0;
    case 6:  q5_0_rows(out, W, qa, da, r0, r1, k, asum); return 0;
    case 8:  q8_0_rows(out, W, qa, da, r0, r1, k); return 0;
    case 12: q4_k_rows(out, W, qa, da, r0, r1, k, asum); return 0;
    case 14: q6_k_rows(out, W, qa, da, r0, r1, k); return 0;
    default: return -1;
    }
}
