// gguf.c — GGUF file parser for notorch
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* Mapping the tensor block instead of reading it. A model is the largest thing
 * this library ever holds, and reading it into a buffer costs twice: the file
 * passes through the page cache on its way into our copy, so a 2.6 GB model
 * occupies 5.2 GB at the moment of load, and two processes running the same
 * model repeat all of it. A mapping is the page cache, so neither happens —
 * measured on this phone as 1.9-2.1 s to first output against 3.6-4.1, two runs of each.
 *
 * What this does NOT fix, stated because the first version of this comment
 * claimed it did: decode speed on one Gemma file measured 10.2 t/s on one day
 * and 8.6-9.0 the next from an unchanged commit, and mapping does not move it.
 * The swap explanation was checked and refuted — under 3 GiB of deliberate
 * pressure the process reported no swapped pages on either path, because
 * weights touched every token stay hot enough that the kernel evicts something
 * else. Generation length, prompt, temperature, BLAS threads and huge pages
 * were eliminated too. That number is still unexplained.
 *
 * Metal keeps the read path. `nt_metal_register_base` wraps `data` as a NoCopy
 * MTLBuffer, which requires a page-aligned pointer and a page-rounded length,
 * and a mapped view starts wherever the file's 32-byte-aligned data section
 * starts. `NT_GGUF_MMAP=0` forces the read path anywhere else, and
 * `NT_GGUF_MMAP=lazy` maps without populating — half the resident memory, at
 * the cost of faulting the model in during the first forward pass. */
#if !defined(USE_METAL) && !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#define NOTORCH_GGUF_MMAP 1
#include <sys/mman.h>
#else
#define NOTORCH_GGUF_MMAP 0
#endif

// ── Reading primitives ───────────────────────────────────────────────────────

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, 4, 1, f) == 1; }

static int read_string(FILE* f, char* buf, int max) {
    uint64_t len;
    if (!read_u64(f, &len)) return 0;
    if (len >= (uint64_t)max) {
        // String too long: read and discard, stopping at EOF so a crafted
        // huge len can't spin fgetc billions of times past end-of-file.
        for (uint64_t i = 0; i < len; i++) if (fgetc(f) == EOF) break;
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

    if (!read_u32(f, &gf->version) || !read_u64(f, &gf->n_tensors) || !read_u64(f, &gf->n_kv)) {
        fprintf(stderr, "gguf: truncated header (%s)\n", path);
        fclose(f); free(gf); return NULL;
    }

    // The tensor-info table is fixed-size. With more tensors than it holds, the
    // info loop stops early and data_offset (computed from ftell after the loop)
    // lands mid-header, silently corrupting every tensor read. Fail loud instead.
    if (gf->n_tensors > GGUF_MAX_TENSORS) {
        fprintf(stderr, "gguf: %llu tensors exceeds GGUF_MAX_TENSORS=%d (%s); refusing to load\n",
                (unsigned long long)gf->n_tensors, GGUF_MAX_TENSORS, path);
        fclose(f); free(gf); return NULL;
    }

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

    /* Where the key-value section ends is where a rewriter has to resume: the metadata can
     * be copied byte for byte, but every tensor info after it carries a type and an offset
     * that quantization changes. */
    gf->kv_end = (uint64_t)ftell(f);

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
    if (fsize < 0 || gf->data_offset > (uint64_t)fsize) {
        fprintf(stderr, "gguf: truncated before tensor data (%s)\n", path);
        fclose(f); free(gf); return NULL;
    }
    long data_size = fsize - (long)gf->data_offset;
    size_t pg = (size_t)getpagesize();
    gf->data = NULL;
    gf->map_base = NULL;
    gf->map_len = 0;

#if NOTORCH_GGUF_MMAP
    {
        const char* mode = getenv("NT_GGUF_MMAP");
        if (!(mode && mode[0] == '0')) {
            // The mapping has to start on a page boundary; the data section is
            // only 32-byte aligned, so map from the page below it and hand out
            // the offset view. Everything readable through `data` stays inside
            // the mapping because the kernel rounds `map_len` up, never down.
            size_t delta = (size_t)(gf->data_offset & (uint64_t)(pg - 1));
            size_t len   = delta + (size_t)data_size;
            int    flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
            // Faulted lazily, the whole model arrives a page at a time inside the
            // first forward pass, and on a 2.6 GB file that costs half a second of
            // prefill — measured, 7.5 t/s against 12.3 on an 11-token prompt, and
            // invisible on a 577-token one because the cost is fixed rather than
            // per token. Populating pays it at load as one sequential stream, which
            // is what the read path did, and beats the read path on wall clock anyway
            // because nothing is copied — see the header comment for the timings. Where
            // MAP_POPULATE does not exist the mapping is lazy and prefill pays the faults.
            // NT_GGUF_MMAP=lazy asks for that deliberately, to halve resident memory when
            // the model is larger than RAM.
            if (!(mode && strcmp(mode, "lazy") == 0)) flags |= MAP_POPULATE;
#endif
            void*  base  = mmap(NULL, len, PROT_READ, flags, fileno(f), (off_t)(gf->data_offset - delta));
            if (base != MAP_FAILED) {
                gf->map_base  = base;
                gf->map_len   = (uint64_t)len;
                gf->data      = (uint8_t*)base + delta;
                gf->data_size = (uint64_t)data_size;
                fclose(f);
                return gf;
            }
        }
    }
#endif

    // Read path: mapping is off, or the kernel refused. Page-align the tensor
    // block so the Metal backend can wrap it as one zero-copy NoCopy MTLBuffer
    // (resident weights). free() stays valid.
    size_t alloc = ((size_t)data_size + pg - 1) & ~(pg - 1);
    if (posix_memalign((void**)&gf->data, pg, alloc) != 0 || !gf->data) { fclose(f); free(gf); return NULL; }
    gf->data_size = (uint64_t)alloc;
    fseek(f, gf->data_offset, SEEK_SET);
    if (fread(gf->data, 1, (size_t)data_size, f) != (size_t)data_size) {
        fprintf(stderr, "gguf: short read of tensor data (%s)\n", path);
        fclose(f); free(gf->data); free(gf); return NULL;
    }
    fclose(f);

    return gf;
}

uint64_t gguf_type_size(uint32_t dtype, uint64_t n) {
    switch (dtype) {
    case GGUF_TYPE_F32:  return n * 4;
    case GGUF_TYPE_F16:  return n * 2;
    case GGUF_TYPE_BF16: return n * 2;
    case GGUF_TYPE_Q4_0: return n / 32 * 18;
    case GGUF_TYPE_Q5_0: return n / 32 * 22;
    case GGUF_TYPE_Q8_0: return n / 32 * 34;
    case GGUF_TYPE_Q4_K: return n / 256 * 144;
    case GGUF_TYPE_Q6_K: return n / 256 * 210;
    default:             return 0;
    }
}

void gguf_close(gguf_file* gf) {
    if (!gf) return;
#if NOTORCH_GGUF_MMAP
    if (gf->map_base) munmap(gf->map_base, (size_t)gf->map_len);
    else
#endif
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

// Read a GGUF type-9 array of strings (e.g. tokenizer.ggml.tokens / .merges) by key.
// Arrays are skipped during gguf_open, so this re-scans the file. Returns a malloc'd
// char** of *out_n strdup'd strings, or NULL if the key/array is absent. Caller frees
// each string and the array.
/* One string value by key, without opening the tensor data. gguf_open reads the whole data
 * section, which for a 2.6 GB model is a strange price for one word, and the tokenizer needs
 * exactly one: which scheme the file was written in. Scans the metadata and stops. */
int gguf_read_str_kv(const char* path, const char* key, char* out, int cap) {
    if (!out || cap <= 0) return -1;
    out[0] = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return -1; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    int found = -1;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0};
        uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 8) {
            char buf[512] = {0};
            if (read_string(f, buf, sizeof(buf))) {
                snprintf(out, (size_t)cap, "%s", buf);
                found = 0;
            }
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return found;
}

/* One integer-ish value by key — u32, i32, bool or u64 — on the same terms as the string
 * reader above: metadata only, no tensor bytes. The tokenizer needs its BOS id and whether
 * to add one, and both are scalars sitting next to a 262144-entry vocabulary. */
int gguf_read_uint_kv(const char* path, const char* key, uint64_t* out) {
    if (!out) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return -1; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    int found = -1;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0};
        uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0) {
            if (vtype == 4)      { uint32_t v; if (read_u32(f, &v)) { *out = v; found = 0; } }
            else if (vtype == 5) { int32_t v; if (fread(&v, 4, 1, f) == 1) { *out = (uint64_t)v; found = 0; } }
            else if (vtype == 7) { uint8_t v; if (fread(&v, 1, 1, f) == 1) { *out = v; found = 0; } }
            else if (vtype == 10 || vtype == 12) { uint64_t v; if (read_u64(f, &v)) { *out = v; found = 0; } }
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return found;
}

char** gguf_read_str_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    char** result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0};
        uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 8) break;
            if (alen > GGUF_MAX_STR_ARRAY) {
                fprintf(stderr, "gguf: str-array '%s' len %llu exceeds GGUF_MAX_STR_ARRAY=%u; refusing\n",
                        key, (unsigned long long)alen, (unsigned)GGUF_MAX_STR_ARRAY);
                break;
            }
            result = (char**)calloc(alen ? alen : 1, sizeof(char*));
            if (!result) break;
            uint64_t j = 0;
            for (; j < alen; j++) {
                char buf[2048] = {0};
                if (!read_string(f, buf, sizeof(buf))) break;
                result[j] = strdup(buf);
                if (!result[j]) break;
            }
            if (out_n) *out_n = (int)j;   // actually-read count, not claimed alen
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return result;
}

/* Same walk for an INT32 array. tokenizer.ggml.token_type is one, and without it there is
 * no way to tell an added token — a literal run of spaces, say — from an ordinary merge. */
int32_t* gguf_read_i32_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    int32_t* result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0};
        uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen)) break;
            if (atype != 5 && atype != 4) break;          /* INT32 or UINT32 */
            if (alen > GGUF_MAX_STR_ARRAY) {
                fprintf(stderr, "gguf: i32-array '%s' len %llu exceeds GGUF_MAX_STR_ARRAY=%u; refusing\n",
                        key, (unsigned long long)alen, (unsigned)GGUF_MAX_STR_ARRAY);
                break;
            }
            result = (int32_t*)calloc(alen ? alen : 1, sizeof(int32_t));
            if (!result) break;
            uint64_t got = fread(result, sizeof(int32_t), alen, f);
            if (out_n) *out_n = (int)got;
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return result;
}

float* gguf_read_f32_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    float* result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0};
        uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 6) break;  // 6 = FLOAT32
            if (alen > GGUF_MAX_STR_ARRAY) {
                fprintf(stderr, "gguf: f32-array '%s' len %llu exceeds GGUF_MAX_STR_ARRAY=%u; refusing\n",
                        key, (unsigned long long)alen, (unsigned)GGUF_MAX_STR_ARRAY);
                break;
            }
            result = (float*)calloc(alen ? alen : 1, sizeof(float));
            if (!result) break;
            uint64_t got = fread(result, sizeof(float), alen, f);
            if (out_n) *out_n = (int)got;   // actually-read count, not claimed alen
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return result;
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
        // Per ggml dequantize_row_q6_K: two 128-elem halves; per half ql+=64, qh+=32, sc+=8.
        for (int n_ = 0; n_ < 256; n_ += 128) {
            const uint8_t *qlh = ql + (n_/128)*64;
            const uint8_t *qhh = qh + (n_/128)*32;
            const int8_t  *sch = sc + (n_/128)*8;
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                out[i*256 + n_ + l]      = d * sch[is + 0] * q1;
                out[i*256 + n_ + l + 32] = d * sch[is + 2] * q2;
                out[i*256 + n_ + l + 64] = d * sch[is + 4] * q3;
                out[i*256 + n_ + l + 96] = d * sch[is + 6] * q4;
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

// On-disk byte size of a tensor with the given dtype and element count, with
// uint64 overflow detection. n_elements comes straight from the file, so a
// crafted GGUF could otherwise overflow the multiply and wrap to a tiny value
// that slips through the bounds check below. Returns 0 to signal a HARD REJECT:
// unknown dtype, n too small for a full quantized block, or a multiply that
// would overflow. Strides match the dequant_* block layouts above.
/* Values per block for the packed formats, 1 for the ones stored element by element. A row
 * that is not a whole number of blocks cannot be decoded on its own — the last block's scale
 * lives in bytes this row does not own. */
static uint64_t gguf_dtype_block(uint32_t dtype) {
    switch (dtype) {
    case GGUF_TYPE_Q4_0: case GGUF_TYPE_Q5_0: case GGUF_TYPE_Q8_0: return 32;
    case GGUF_TYPE_Q4_K: case GGUF_TYPE_Q6_K:                      return 256;
    default:                                                       return 1;
    }
}

static uint64_t gguf_dtype_nbytes(uint32_t dtype, uint64_t n) {
    uint64_t blocks, per;
    switch (dtype) {
    case GGUF_TYPE_F32:  return (n > UINT64_MAX / 4) ? 0 : n * 4;
    case GGUF_TYPE_F16:  return (n > UINT64_MAX / 2) ? 0 : n * 2;
    case GGUF_TYPE_BF16: return (n > UINT64_MAX / 2) ? 0 : n * 2;
    case GGUF_TYPE_Q4_0: blocks = n / 32;  per = 18;  break;
    case GGUF_TYPE_Q5_0: blocks = n / 32;  per = 22;  break;
    case GGUF_TYPE_Q8_0: blocks = n / 32;  per = 34;  break;
    case GGUF_TYPE_Q4_K: blocks = n / 256; per = 144; break;
    case GGUF_TYPE_Q6_K: blocks = n / 256; per = 210; break;
    default: return 0;
    }
    if (blocks == 0 || blocks > UINT64_MAX / per) return 0;  // empty or overflow
    return blocks * per;
}

float* gguf_dequant(const gguf_file* gf, int tensor_idx) {
    if (!gf || tensor_idx < 0 || tensor_idx >= (int)gf->n_tensors) return NULL;
    const gguf_tensor_info* ti = &gf->tensors[tensor_idx];
    // ti->offset + on-disk byte size must fit in the data buffer, so a
    // malformed/oversized GGUF can't drive an out-of-bounds read from here.
    // (M-4: the offset-only guard missed a tensor starting just below the end.)
    // nbytes == 0 is a hard reject (unknown dtype / overflow / sub-block n) — no
    // escape hatch, so the dequant switch default is not the only guard.
    uint64_t nbytes = gguf_dtype_nbytes(ti->dtype, ti->n_elements);
    if (nbytes == 0 || ti->offset >= gf->data_size ||
        nbytes > gf->data_size - ti->offset) {
        fprintf(stderr, "gguf: tensor '%s' out of bounds/invalid (off %llu + %llu bytes, data_size %llu)\n",
                ti->name, (unsigned long long)ti->offset, (unsigned long long)nbytes,
                (unsigned long long)gf->data_size);
        return NULL;
    }
    const uint8_t* src = gf->data + ti->offset;

    float* dst = (float*)calloc(ti->n_elements, sizeof(float));
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
    /* bfloat16 is f32 with the low mantissa cut off, so widening it is a shift and
     * nothing else — no exponent rebias, no subnormal case. Gemma-4 stores its
     * per-layer projection this way. */
    case GGUF_TYPE_BF16: {
        const uint16_t* bf = (const uint16_t*)src;
        for (uint64_t i = 0; i < ti->n_elements; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&dst[i], &bits, 4);
        }
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

/* One row of a packed tensor, without materialising the rest.
 *
 * An embedding table is read one row per token, and in a GGUF it is usually the largest
 * tensor in the file: Qwen2.5-1.5B keeps 151936 x 1536 there, which is 933 MB expanded to
 * f32 against 131 MB left packed. A caller that only ever needs one row at a time should
 * not have to pay for the other 151935.
 *
 * Rows are contiguous and a row length is a whole number of blocks, so a row decodes with
 * exactly the format code gguf_dequant uses, pointed at an offset — same values, same
 * order, no second implementation to keep in step.
 *
 * dst needs shape[0] floats. Returns 0, or -1 on a bad tensor index, a row past the end,
 * a row length that is not a whole number of blocks, or a dtype with no decoder. */
int gguf_dequant_row(const gguf_file* gf, int tensor_idx, uint64_t row, float* dst) {
    if (!gf || !dst || tensor_idx < 0 || tensor_idx >= (int)gf->n_tensors) return -1;
    const gguf_tensor_info* ti = &gf->tensors[tensor_idx];

    uint64_t cols = ti->shape[0];
    if (cols == 0 || ti->n_elements % cols) return -1;
    uint64_t rows = ti->n_elements / cols;
    if (row >= rows) return -1;

    /* A partial block cannot be decoded on its own, and reading the neighbouring row's bytes
     * would be the wrong answer rather than a slow one. The size alone does not catch it:
     * gguf_dtype_nbytes divides, so 48 values of Q8_0 come back as the 34 bytes of one whole
     * block and the remaining sixteen are read out of whatever follows. Seen on a Mamba
     * checkpoint whose ssm_dt rows are 48 wide — current llama.cpp refuses that file outright,
     * this reader used to accept it and return uninitialised floats with a success code. */
    uint64_t blk = gguf_dtype_block(ti->dtype);
    if (blk > 1 && (cols % blk) != 0) return -1;
    uint64_t rb = gguf_dtype_nbytes(ti->dtype, cols);
    if (rb == 0) return -1;

    /* Same bounds discipline as gguf_dequant: a malformed file must not drive a read past
     * the mapped data, and the multiply must not wrap on the way. */
    if (rb > (UINT64_MAX - ti->offset) / (rows ? rows : 1)) return -1;
    uint64_t off = ti->offset + row * rb;
    if (off >= gf->data_size || rb > gf->data_size - off) return -1;

    const uint8_t* src = gf->data + off;
    switch (ti->dtype) {
    case GGUF_TYPE_F32: memcpy(dst, src, cols * sizeof(float)); break;
    case GGUF_TYPE_F16: {
        const uint16_t* f16 = (const uint16_t*)src;
        for (uint64_t i = 0; i < cols; i++) dst[i] = f16_to_f32(f16[i]);
        break;
    }
    case GGUF_TYPE_BF16: {
        const uint16_t* bf = (const uint16_t*)src;
        for (uint64_t i = 0; i < cols; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&dst[i], &bits, 4);
        }
        break;
    }
    case GGUF_TYPE_Q4_0: dequant_q4_0(src, dst, cols); break;
    case GGUF_TYPE_Q5_0: dequant_q5_0(src, dst, cols); break;
    case GGUF_TYPE_Q8_0: dequant_q8_0(src, dst, cols); break;
    case GGUF_TYPE_Q4_K: dequant_q4_k(src, dst, cols); break;
    case GGUF_TYPE_Q6_K: dequant_q6_k(src, dst, cols); break;
    default: return -1;
    }
    return 0;
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

// ── Writing ──────────────────────────────────────────────────────────────────
/* A GGUF puts its entire tensor directory — name, shape, dtype, and the offset of the
 * bytes — in front of the bytes, so no offset is known until every tensor has been
 * declared. That is the whole reason this is two phases rather than one call: declare,
 * then deliver in declaration order. See gguf.h for the contract.
 *
 * Nothing holds tensor data. The metadata and the directory are buffered because they
 * have to be written before offsets are known and they are kilobytes; the tensor bytes
 * go from the caller's pointer into fwrite. A 91 MB file therefore costs whatever the
 * caller's own chunk costs and nothing else, which on a phone is the difference between
 * a checkpoint that writes and one that wakes the low-memory killer. */

#define GGUF_WRITE_ALIGN   32
#define GGUF_WRITE_VERSION 3

typedef struct {
    char     name[GGUF_MAX_NAME];
    uint32_t ndim;
    uint64_t shape[4];
    uint32_t dtype;
    uint64_t offset;      /* from the start of the data section */
    uint64_t nbytes;      /* packed size on disk */
} gguf_wtensor;

struct gguf_writer {
    FILE*         f;
    char*         path;
    uint8_t*      kv;            /* metadata bytes, appended as they arrive */
    size_t        kv_len, kv_cap;
    uint64_t      n_kv;
    gguf_wtensor* t;
    uint64_t      n_t, t_cap;
    uint64_t      dcursor;       /* next free offset inside the data section */
    int           phase;         /* 0 declaring, 1 past the header */
    int           failed;
    int           in_tensor;
    uint64_t      data_start;    /* absolute file offset of the data section */
    uint64_t      next;          /* index of the tensor expected next */
    uint64_t      cur_written;   /* bytes of the open tensor written so far */
    uint64_t      pos;           /* absolute file position, tracked rather than asked */
};

static uint64_t gguf_walign(uint64_t v) {
    return (v + (GGUF_WRITE_ALIGN - 1)) & ~(uint64_t)(GGUF_WRITE_ALIGN - 1);
}

/* Every refusal goes through here, so a writer that has said no once stays no: the file
 * is going to be removed at close, and continuing to append to it only makes the wreck
 * bigger. The message names the tensor or key, because "gguf write failed" on a file with
 * three hundred tensors is not a diagnosis. */
static int gw_fail(gguf_writer* w, const char* what, const char* detail) {
    if (w) w->failed = 1;
    fprintf(stderr, "gguf write: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return -1;
}

/* f32 -> f16, round to nearest even — the inverse of f16_to_f32 above, and the same
 * rounding llama.cpp applies, so a tensor written here and one written there from the
 * same floats are the same bytes. Overflow goes to infinity rather than to the largest
 * finite value: a weight that large is a bug upstream and clamping would hide it. */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e32  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (e32 == 0xFF)                                   /* inf, or a NaN kept noisy */
        return (uint16_t)(sign | 0x7C00u | (mant ? (0x200u | (mant >> 13)) : 0u));
    int32_t exp = (int32_t)e32 - 127 + 15;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {                                    /* subnormal half, or under it */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;                             /* the implicit one */
        uint32_t shift = (uint32_t)(14 - exp);         /* 14..24 */
        uint32_t half  = 1u << (shift - 1);
        uint32_t r     = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = mant >> 13, rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        r++;
        if (r == 0x400u) { r = 0; exp++; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | r);
}

// ── Metadata buffer ──────────────────────────────────────────────────────────

static int kvbuf(gguf_writer* w, const void* p, size_t n) {
    if (n == 0) return 0;
    if (w->kv_len + n > w->kv_cap) {
        size_t cap = w->kv_cap ? w->kv_cap : 1024;
        while (cap < w->kv_len + n) cap *= 2;
        uint8_t* nb = (uint8_t*)realloc(w->kv, cap);
        if (!nb) return gw_fail(w, "out of memory for metadata", NULL);
        w->kv = nb;
        w->kv_cap = cap;
    }
    memcpy(w->kv + w->kv_len, p, n);
    w->kv_len += n;
    return 0;
}

static int kvb_u32(gguf_writer* w, uint32_t v) { return kvbuf(w, &v, 4); }
static int kvb_u64(gguf_writer* w, uint64_t v) { return kvbuf(w, &v, 8); }

static int kvb_str(gguf_writer* w, const char* s) {
    uint64_t n = strlen(s);
    if (kvb_u64(w, n)) return -1;
    return kvbuf(w, s, (size_t)n);
}

/* A key longer than GGUF_MAX_NAME is refused rather than written, because gguf_open
 * strncpy's it into a 128-byte field: the file would be valid and the key would come
 * back a different key. Same argument as the duplicate tensor name below. */
static int kv_begin(gguf_writer* w, const char* key, uint32_t type) {
    if (!w || w->failed) return -1;
    if (w->phase != 0)   return gw_fail(w, "metadata added after the first tensor byte", key);
    if (!key || !*key)   return gw_fail(w, "empty metadata key", NULL);
    if (strlen(key) >= GGUF_MAX_NAME)
        return gw_fail(w, "metadata key longer than GGUF_MAX_NAME, which the reader truncates", key);
    if (kvb_str(w, key) || kvb_u32(w, type)) return -1;
    w->n_kv++;
    return 0;
}

static int kv_array_begin(gguf_writer* w, const char* key, uint32_t etype, uint64_t n) {
    if (kv_begin(w, key, 9)) return -1;
    if (kvb_u32(w, etype) || kvb_u64(w, n)) return -1;
    return 0;
}

int gguf_write_kv_u32(gguf_writer* w, const char* key, uint32_t v) {
    if (kv_begin(w, key, 4)) return -1;
    return kvb_u32(w, v);
}

int gguf_write_kv_i32(gguf_writer* w, const char* key, int32_t v) {
    if (kv_begin(w, key, 5)) return -1;
    return kvbuf(w, &v, 4);
}

int gguf_write_kv_f32(gguf_writer* w, const char* key, float v) {
    if (kv_begin(w, key, 6)) return -1;
    return kvbuf(w, &v, 4);
}

int gguf_write_kv_bool(gguf_writer* w, const char* key, int v) {
    uint8_t b = v ? 1 : 0;
    if (kv_begin(w, key, 7)) return -1;
    return kvbuf(w, &b, 1);
}

int gguf_write_kv_str(gguf_writer* w, const char* key, const char* val) {
    if (!val) return gw_fail(w, "null string value", key);
    if (kv_begin(w, key, 8)) return -1;
    return kvb_str(w, val);
}

int gguf_write_kv_u64(gguf_writer* w, const char* key, uint64_t v) {
    if (kv_begin(w, key, 10)) return -1;
    return kvb_u64(w, v);
}

int gguf_write_kv_str_array(gguf_writer* w, const char* key, const char* const* vals, uint64_t n) {
    if (n && !vals) return gw_fail(w, "null string array", key);
    if (kv_array_begin(w, key, 8, n)) return -1;
    for (uint64_t i = 0; i < n; i++) {
        if (!vals[i]) return gw_fail(w, "null string in array", key);
        if (kvb_str(w, vals[i])) return -1;
    }
    return 0;
}

int gguf_write_kv_i32_array(gguf_writer* w, const char* key, const int32_t* vals, uint64_t n) {
    if (n && !vals) return gw_fail(w, "null int32 array", key);
    if (kv_array_begin(w, key, 5, n)) return -1;
    return kvbuf(w, vals, (size_t)n * sizeof(int32_t));
}

int gguf_write_kv_f32_array(gguf_writer* w, const char* key, const float* vals, uint64_t n) {
    if (n && !vals) return gw_fail(w, "null float array", key);
    if (kv_array_begin(w, key, 6, n)) return -1;
    return kvbuf(w, vals, (size_t)n * sizeof(float));
}

// ── File primitives ──────────────────────────────────────────────────────────

static int gw_raw(gguf_writer* w, const void* p, uint64_t n) {
    if (n == 0) return 0;
    if (fwrite(p, 1, (size_t)n, w->f) != (size_t)n)
        return gw_fail(w, "short write", w->path);
    w->pos += n;
    return 0;
}

static int gw_u32(gguf_writer* w, uint32_t v) { return gw_raw(w, &v, 4); }
static int gw_u64(gguf_writer* w, uint64_t v) { return gw_raw(w, &v, 8); }

static int gw_str(gguf_writer* w, const char* s) {
    uint64_t n = strlen(s);
    if (gw_u64(w, n)) return -1;
    return gw_raw(w, s, n);
}

static int gw_pad_to(gguf_writer* w, uint64_t target) {
    static const uint8_t zeros[GGUF_WRITE_ALIGN] = {0};
    if (w->pos > target) return gw_fail(w, "internal: file past its own alignment target", w->path);
    while (w->pos < target) {
        uint64_t want = target - w->pos;
        if (want > sizeof(zeros)) want = sizeof(zeros);
        if (gw_raw(w, zeros, want)) return -1;
    }
    return 0;
}

/* Header, metadata and directory, written once, at the moment the first tensor byte is
 * asked for or at close if there never is one. After this the metadata buffer is freed:
 * it is the only thing in the writer with any size, and nothing can add to it now. */
static int gw_flush_header(gguf_writer* w) {
    uint32_t magic = GGUF_MAGIC;
    if (gw_raw(w, &magic, 4) || gw_u32(w, GGUF_WRITE_VERSION) ||
        gw_u64(w, w->n_t) || gw_u64(w, w->n_kv)) return -1;
    if (gw_raw(w, w->kv, w->kv_len)) return -1;
    free(w->kv);
    w->kv = NULL;
    w->kv_len = w->kv_cap = 0;

    for (uint64_t i = 0; i < w->n_t; i++) {
        const gguf_wtensor* t = &w->t[i];
        if (gw_str(w, t->name) || gw_u32(w, t->ndim)) return -1;
        for (uint32_t d = 0; d < t->ndim; d++)
            if (gw_u64(w, t->shape[d])) return -1;
        if (gw_u32(w, t->dtype) || gw_u64(w, t->offset)) return -1;
    }

    w->data_start = gguf_walign(w->pos);
    if (gw_pad_to(w, w->data_start)) return -1;
    w->phase = 1;
    return 0;
}

// ── Open, declare, deliver, close ────────────────────────────────────────────

gguf_writer* gguf_write_open(const char* path) {
    if (!path || !*path) return NULL;
    gguf_writer* w = (gguf_writer*)calloc(1, sizeof(gguf_writer));
    if (!w) return NULL;
    w->path = strdup(path);
    if (!w->path) { free(w); return NULL; }
    w->f = fopen(path, "wb");
    if (!w->f) {
        fprintf(stderr, "gguf write: cannot create %s\n", path);
        free(w->path); free(w);
        return NULL;
    }
    return w;
}

int gguf_write_tensor_decl(gguf_writer* w, const char* name, uint32_t ndim,
                           const uint64_t* shape, uint32_t dtype) {
    if (!w || w->failed) return -1;
    if (w->phase != 0) return gw_fail(w, "tensor declared after the first tensor byte", name);
    if (!name || !*name) return gw_fail(w, "empty tensor name", NULL);
    if (strlen(name) >= GGUF_MAX_NAME)
        return gw_fail(w, "tensor name longer than GGUF_MAX_NAME, which the reader drops", name);
    if (ndim < 1 || ndim > 4 || !shape)
        return gw_fail(w, "tensor needs 1 to 4 dimensions", name);
    if (w->n_t >= GGUF_MAX_TENSORS)
        return gw_fail(w, "more tensors than GGUF_MAX_TENSORS, which gguf_open refuses to load", name);

    /* Linear, because the reader resolves names linearly too and a file it cannot
     * unambiguously index is not worth writing. */
    for (uint64_t i = 0; i < w->n_t; i++)
        if (strcmp(w->t[i].name, name) == 0)
            return gw_fail(w, "duplicate tensor name", name);

    uint64_t n = 1;
    for (uint32_t d = 0; d < ndim; d++) {
        if (shape[d] == 0) return gw_fail(w, "tensor dimension of zero", name);
        if (shape[d] > UINT64_MAX / n) return gw_fail(w, "tensor element count overflows", name);
        n *= shape[d];
    }
    uint64_t blk = gguf_dtype_block(dtype);
    if (blk > 1 && (n % blk) != 0)
        return gw_fail(w, "packed tensor whose element count is not a whole number of blocks", name);
    uint64_t nbytes = gguf_dtype_nbytes(dtype, n);
    if (nbytes == 0) return gw_fail(w, "dtype this library cannot size", name);

    if (w->n_t == w->t_cap) {
        uint64_t cap = w->t_cap ? w->t_cap * 2 : 64;
        gguf_wtensor* nt = (gguf_wtensor*)realloc(w->t, (size_t)cap * sizeof(gguf_wtensor));
        if (!nt) return gw_fail(w, "out of memory for the tensor directory", name);
        w->t = nt;
        w->t_cap = cap;
    }
    gguf_wtensor* t = &w->t[w->n_t++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->ndim  = ndim;
    for (uint32_t d = 0; d < ndim; d++) t->shape[d] = shape[d];
    t->dtype  = dtype;
    t->nbytes = nbytes;
    t->offset = w->dcursor;
    w->dcursor = gguf_walign(w->dcursor + nbytes);
    return 0;
}

int gguf_write_tensor_begin(gguf_writer* w, const char* name) {
    if (!w || w->failed) return -1;
    if (w->in_tensor)
        return gw_fail(w, "a tensor is already open", w->t[w->next].name);
    if (w->phase == 0 && gw_flush_header(w)) return -1;
    if (w->next >= w->n_t)
        return gw_fail(w, "tensor data for a tensor that was never declared", name);
    const gguf_wtensor* t = &w->t[w->next];
    if (!name || strcmp(name, t->name) != 0) {
        char msg[2 * GGUF_MAX_NAME + 32];
        snprintf(msg, sizeof(msg), "expected '%s', got '%s'", t->name, name ? name : "(null)");
        return gw_fail(w, "tensor data out of declaration order", msg);
    }
    if (gw_pad_to(w, w->data_start + t->offset)) return -1;
    w->in_tensor   = 1;
    w->cur_written = 0;
    return 0;
}

int gguf_write_tensor_chunk(gguf_writer* w, const void* bytes, uint64_t nbytes) {
    if (!w || w->failed) return -1;
    if (!w->in_tensor) return gw_fail(w, "tensor chunk outside begin/end", NULL);
    const gguf_wtensor* t = &w->t[w->next];
    if (nbytes > t->nbytes - w->cur_written) {
        char msg[GGUF_MAX_NAME + 96];
        snprintf(msg, sizeof(msg), "%s: %llu bytes past the declared %llu", t->name,
                 (unsigned long long)(w->cur_written + nbytes), (unsigned long long)t->nbytes);
        return gw_fail(w, "tensor data longer than declared", msg);
    }
    if (nbytes && !bytes) return gw_fail(w, "null tensor chunk", t->name);
    if (gw_raw(w, bytes, nbytes)) return -1;
    w->cur_written += nbytes;
    return 0;
}

/* f32 in, whatever the tensor was declared as out. F32 goes straight through with no
 * intermediate buffer at all; F16 rounds through a fixed window, so converting a tensor
 * costs 8 KB rather than half the tensor. */
int gguf_write_tensor_chunk_f32(gguf_writer* w, const float* src, uint64_t n) {
    if (!w || w->failed) return -1;
    if (!w->in_tensor) return gw_fail(w, "tensor chunk outside begin/end", NULL);
    if (n && !src) return gw_fail(w, "null float chunk", w->t[w->next].name);
    uint32_t dtype = w->t[w->next].dtype;
    if (dtype == GGUF_TYPE_F32)
        return gguf_write_tensor_chunk(w, src, n * 4);
    if (dtype != GGUF_TYPE_F16)
        return gw_fail(w, "float source for a tensor that is neither F32 nor F16 — quantize it first",
                       w->t[w->next].name);
    uint16_t half[4096];
    uint64_t done = 0;
    while (done < n) {
        uint64_t take = n - done;
        if (take > 4096) take = 4096;
        for (uint64_t i = 0; i < take; i++) half[i] = f32_to_f16(src[done + i]);
        if (gguf_write_tensor_chunk(w, half, take * 2)) return -1;
        done += take;
    }
    return 0;
}

int gguf_write_tensor_end(gguf_writer* w) {
    if (!w || w->failed) return -1;
    if (!w->in_tensor) return gw_fail(w, "tensor end without a begin", NULL);
    const gguf_wtensor* t = &w->t[w->next];
    if (w->cur_written != t->nbytes) {
        char msg[GGUF_MAX_NAME + 96];
        snprintf(msg, sizeof(msg), "%s: %llu bytes of a declared %llu", t->name,
                 (unsigned long long)w->cur_written, (unsigned long long)t->nbytes);
        return gw_fail(w, "tensor short of its declared size", msg);
    }
    w->in_tensor = 0;
    w->next++;
    return 0;
}

int gguf_write_tensor(gguf_writer* w, const char* name, const void* bytes, uint64_t nbytes) {
    if (gguf_write_tensor_begin(w, name)) return -1;
    if (gguf_write_tensor_chunk(w, bytes, nbytes)) return -1;
    return gguf_write_tensor_end(w);
}

int gguf_write_tensor_f32(gguf_writer* w, const char* name, const float* src, uint64_t n) {
    if (gguf_write_tensor_begin(w, name)) return -1;
    if (gguf_write_tensor_chunk_f32(w, src, n)) return -1;
    return gguf_write_tensor_end(w);
}

int gguf_write_close(gguf_writer* w) {
    if (!w) return -1;
    int rc = w->failed ? -1 : 0;
    if (!rc && w->in_tensor) rc = gw_fail(w, "file closed with a tensor open", w->t[w->next].name);
    if (!rc && w->phase == 0 && gw_flush_header(w)) rc = -1;
    if (!rc && w->next != w->n_t) {
        char msg[GGUF_MAX_NAME + 64];
        snprintf(msg, sizeof(msg), "%llu of %llu delivered, next is '%s'",
                 (unsigned long long)w->next, (unsigned long long)w->n_t, w->t[w->next].name);
        rc = gw_fail(w, "declared tensors never written", msg);
    }
    /* The tail of the last tensor is padded like every other, which is what ggml's writer
     * does; a reader that trusts the directory never looks at those bytes, and one that
     * measures the data section against the alignment finds what it expects. */
    if (!rc && gw_pad_to(w, gguf_walign(w->pos))) rc = -1;
    if (fclose(w->f) != 0 && !rc) rc = gw_fail(w, "close failed", w->path);
    if (rc) remove(w->path);
    free(w->kv);
    free(w->t);
    free(w->path);
    free(w);
    return rc;
}

void gguf_write_abort(gguf_writer* w) {
    if (!w) return;
    fclose(w->f);
    remove(w->path);
    free(w->kv);
    free(w->t);
    free(w->path);
    free(w);
}
