// gguf.c — GGUF file parser for notorch
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ── Reading primitives ───────────────────────────────────────────────────────

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, 4, 1, f) == 1; }

static int read_string(FILE* f, char* buf, int max) {
    uint64_t len;
    if (!read_u64(f, &len)) return 0;
    if (len >= (uint64_t)max) {
        // String too long: read and discard
        for (uint64_t i = 0; i < len; i++) fgetc(f);
        buf[0] = 0;
        return 1;
    }
    if (fread(buf, 1, len, f) != len) return 0;
    buf[len] = 0;
    return 1;
}

static int skip_value(FILE* f, uint32_t type);

static int skip_array(FILE* f) {
    uint32_t atype;
    uint64_t alen;
    if (!read_u32(f, &atype) || !read_u64(f, &alen)) return 0;
    for (uint64_t i = 0; i < alen; i++)
        if (!skip_value(f, atype)) return 0;
    return 1;
}

static int skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case 4: fseek(f, 4, SEEK_CUR); return 1;  // uint32
        case 5: fseek(f, 4, SEEK_CUR); return 1;  // int32
        case 6: fseek(f, 4, SEEK_CUR); return 1;  // float32
        case 7: fseek(f, 1, SEEK_CUR); return 1;  // bool
        case 8: { char buf[4096]; return read_string(f, buf, sizeof(buf)); } // string
        case 9: return skip_array(f);               // array
        case 10: fseek(f, 8, SEEK_CUR); return 1;  // uint64
        case 12: fseek(f, 8, SEEK_CUR); return 1;  // uint64
        default: return 0;
    }
}

// ── GGUF Open ────────────────────────────────────────────────────────────────

gguf_file* gguf_open(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }

    // Header
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic (got 0x%08x)\n", magic);
        fclose(f); return NULL;
    }

    gguf_file* gf = (gguf_file*)calloc(1, sizeof(gguf_file));
    if (!gf) { fclose(f); return NULL; }

    read_u32(f, &gf->version);
    read_u64(f, &gf->n_tensors);
    read_u64(f, &gf->n_kv);

    // Parse metadata
    gf->n_kv_parsed = 0;
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        char key[512] = {0};
        uint32_t vtype;
        read_string(f, key, sizeof(key));
        read_u32(f, &vtype);

        // Store simple types, skip arrays
        if (gf->n_kv_parsed < GGUF_MAX_KV && vtype != 9) {
            gguf_kv* kv = &gf->kv[gf->n_kv_parsed];
            strncpy(kv->key, key, GGUF_MAX_NAME - 1);
            kv->type = vtype;
            switch (vtype) {
                case 4: read_u32(f, &kv->val.u32); break;
                case 5: { int32_t v; fread(&v, 4, 1, f); kv->val.i32 = v; break; }
                case 6: read_f32(f, &kv->val.f32); break;
                case 7: { uint8_t v; fread(&v, 1, 1, f); kv->val.b = v; break; }
                case 8: read_string(f, kv->val.str, sizeof(kv->val.str)); break;
                case 10: case 12: read_u64(f, &kv->val.u64); break;
                default: skip_value(f, vtype); break;
            }
            gf->n_kv_parsed++;
        } else {
            skip_value(f, vtype);
        }
    }

    // Extract architecture params
    for (int i = 0; i < gf->n_kv_parsed; i++) {
        gguf_kv* kv = &gf->kv[i];
        if (strcmp(kv->key, "general.architecture") == 0)
            strncpy(gf->arch, kv->val.str, sizeof(gf->arch) - 1);
        else if (strstr(kv->key, ".block_count"))
            gf->n_layers = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count") && !strstr(kv->key, "kv"))
            gf->n_heads = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count_kv"))
            gf->n_kv_heads = kv->val.u32;
        else if (strstr(kv->key, ".embedding_length"))
            gf->embed_dim = kv->val.u32;
        else if (strstr(kv->key, ".feed_forward_length"))
            gf->ffn_dim = kv->val.u32;
        else if (strstr(kv->key, ".vocab_size"))
            gf->vocab_size = kv->val.u32;
        else if (strstr(kv->key, ".context_length"))
            gf->ctx_len = kv->val.u32;
        else if (strstr(kv->key, ".rope.freq_base"))
            gf->rope_freq_base = kv->val.f32;
        else if (strstr(kv->key, "rms_epsilon"))
            gf->rms_eps = kv->val.f32;
    }
    if (gf->n_kv_heads == 0) gf->n_kv_heads = gf->n_heads;
    if (gf->rms_eps == 0) gf->rms_eps = 1e-5f;
    if (gf->rope_freq_base == 0) gf->rope_freq_base = 10000.0f;

    // Parse tensor infos
    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++) {
        gguf_tensor_info* ti = &gf->tensors[i];
        read_string(f, ti->name, GGUF_MAX_NAME);
        read_u32(f, &ti->ndim);
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->ndim && d < 4; d++) {
            read_u64(f, &ti->shape[d]);
            ti->n_elements *= ti->shape[d];
        }
        read_u32(f, &ti->dtype);
        read_u64(f, &ti->offset);
    }

    // Data section starts at aligned offset after tensor infos
    long pos = ftell(f);
    gf->data_offset = (pos + 31) & ~31UL;  // align to 32 bytes

    // Load tensor data (offsets are relative to data section start)
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    long data_size = fsize - gf->data_offset;
    gf->data = (uint8_t*)malloc(data_size);
    if (!gf->data) { fclose(f); free(gf); return NULL; }
    fseek(f, gf->data_offset, SEEK_SET);
    fread(gf->data, 1, data_size, f);
    fclose(f);

    return gf;
}

void gguf_close(gguf_file* gf) {
    if (!gf) return;
    free(gf->data);
    free(gf);
}

int gguf_find_tensor(const gguf_file* gf, const char* name) {
    if (!gf || !name) return -1;
    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++)
        if (strcmp(gf->tensors[i].name, name) == 0) return (int)i;
    return -1;
}

const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key) {
    if (!gf || !key) return NULL;
    for (int i = 0; i < gf->n_kv_parsed; i++)
        if (strcmp(gf->kv[i].key, key) == 0) return &gf->kv[i];
    return NULL;
}

// ── Dequantization ───────────────────────────────────────────────────────────

// F16 → F32 conversion
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t r = (sign << 31) | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    exp = exp + 127 - 15;
    uint32_t r = (sign << 31) | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4);
    return f;
}

// Q4_0 block: 2 bytes scale (f16) + 16 bytes data (32 nibbles) = 18 bytes per 32 elements
static void dequant_q4_0(const uint8_t* src, float* dst, uint64_t n_elements) {
    uint64_t n_blocks = n_elements / 32;
    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = src + b * 18;
        uint16_t sh; memcpy(&sh, block, 2);
        float scale = f16_to_f32(sh);
        for (int i = 0; i < 16; i++) {
            uint8_t byte = block[2 + i];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4) - 8;
            dst[b * 32 + i] = lo * scale;
            dst[b * 32 + i + 16] = hi * scale;
        }
    }
}

// Q8_0 block: 2 bytes scale (f16) + 32 bytes data (32 int8) = 34 bytes per 32 elements
static void dequant_q8_0(const uint8_t* src, float* dst, uint64_t n_elements) {
    uint64_t n_blocks = n_elements / 32;
    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = src + b * 34;
        uint16_t sh; memcpy(&sh, block, 2);
        float scale = f16_to_f32(sh);
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = (float)(int8_t)block[2 + i] * scale;
        }
    }
}

// Q4_K: block = 2+2 bytes f16 (d, dmin) + 12 bytes scales + 128 nibbles = 144 bytes, 256 values
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}

static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * 144;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        float dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0, oi = (int)(i * 256);
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v);
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                out[oi + j + l] = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                out[oi + j + 32 + l] = d2 * (float)(qs[qi+l] >> 4) - mm2;
            qi += 32; is += 2;
        }
    }
}

// Q6_K: block = 128 ql + 64 qh + 16 scales + 2 d = 210 bytes, 256 values
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / 256;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * 210;
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t*)(b + 192);
        float d = f16_to_f32(b[208] | (b[209] << 8));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            for (int l = 0; l < 32; l++) {
                int is_ = n_/128*2;
                uint8_t q0 = ql[n_/2+l] & 0xF, q1 = ql[n_/2+l] >> 4;
                uint8_t q2 = ql[n_/2+l+32] & 0xF, q3 = ql[n_/2+l+32] >> 4;
                uint8_t h0 = (qh[n_/4+l] >> 0) & 3, h1 = (qh[n_/4+l] >> 2) & 3;
                uint8_t h2 = (qh[n_/4+l] >> 4) & 3, h3 = (qh[n_/4+l] >> 6) & 3;
                out[i*256 + n_ + l]      = d * sc[is_+0] * ((int)(q0 | (h0<<4)) - 32);
                out[i*256 + n_ + l + 32] = d * sc[is_+1] * ((int)(q1 | (h1<<4)) - 32);
                out[i*256 + n_ + l + 64] = d * sc[is_+2] * ((int)(q2 | (h2<<4)) - 32);
                out[i*256 + n_ + l + 96] = d * sc[is_+3] * ((int)(q3 | (h3<<4)) - 32);
            }
        }
    }
}

// Q5_0: block = 2 bytes f16 + 4 bytes high bits + 16 bytes nibbles = 22 bytes, 32 values
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / 32;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * 22;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int hbit0 = (qh >> j) & 1, hbit1 = (qh >> (j+16)) & 1;
            out[i*32 + j] = (float)((lo | (hbit0<<4)) - 16) * d;
            out[i*32 + j + 16] = (float)((hi | (hbit1<<4)) - 16) * d;
        }
    }
}

float* gguf_dequant(const gguf_file* gf, int tensor_idx) {
    if (!gf || tensor_idx < 0 || tensor_idx >= (int)gf->n_tensors) return NULL;
    const gguf_tensor_info* ti = &gf->tensors[tensor_idx];
    const uint8_t* src = gf->data + ti->offset;

    float* dst = (float*)malloc(ti->n_elements * sizeof(float));
    if (!dst) return NULL;

    switch (ti->dtype) {
    case GGUF_TYPE_F32:
        memcpy(dst, src, ti->n_elements * sizeof(float));
        break;
    case GGUF_TYPE_F16: {
        const uint16_t* f16 = (const uint16_t*)src;
        for (uint64_t i = 0; i < ti->n_elements; i++)
            dst[i] = f16_to_f32(f16[i]);
        break;
    }
    case GGUF_TYPE_Q4_0:
        dequant_q4_0(src, dst, ti->n_elements);
        break;
    case GGUF_TYPE_Q5_0:
        dequant_q5_0(src, dst, ti->n_elements);
        break;
    case GGUF_TYPE_Q8_0:
        dequant_q8_0(src, dst, ti->n_elements);
        break;
    case GGUF_TYPE_Q4_K:
        dequant_q4_k(src, dst, ti->n_elements);
        break;
    case GGUF_TYPE_Q6_K:
        dequant_q6_k(src, dst, ti->n_elements);
        break;
    default:
        fprintf(stderr, "gguf: unsupported dtype %d for tensor '%s'\n", ti->dtype, ti->name);
        free(dst);
        return NULL;
    }
    return dst;
}

void gguf_print_info(const gguf_file* gf) {
    if (!gf) return;
    printf("GGUF v%d: %llu tensors, %llu metadata\n", gf->version, gf->n_tensors, gf->n_kv);
    printf("  arch: %s\n", gf->arch);
    printf("  layers=%d heads=%d kv_heads=%d embed=%d ffn=%d vocab=%d ctx=%d\n",
           gf->n_layers, gf->n_heads, gf->n_kv_heads,
           gf->embed_dim, gf->ffn_dim, gf->vocab_size, gf->ctx_len);
    printf("  rope_base=%.0f rms_eps=%.1e\n", gf->rope_freq_base, gf->rms_eps);

    // List tensors
    const char* dtype_names[] = {"F32", "F16", "Q4_0", "Q4_1", "?", "?", "Q5_0", "?", "Q8_0", "?", "?", "?", "Q4_K", "?", "Q6_K"};
    uint64_t total_params = 0;
    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++) {
        const gguf_tensor_info* ti = &gf->tensors[i];
        const char* dn = ti->dtype <= 14 ? dtype_names[ti->dtype] : "?";
        printf("  [%2llu] %-40s %s  [", i, ti->name, dn);
        for (uint32_t d = 0; d < ti->ndim; d++)
            printf("%llu%s", ti->shape[d], d < ti->ndim - 1 ? "," : "");
        printf("]\n");
        total_params += ti->n_elements;
    }
    printf("  total: %llu elements\n", total_params);
}
