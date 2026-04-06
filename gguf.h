// gguf.h — GGUF file parser for notorch
// Reads llama.cpp-compatible GGUF files
// Supports: F32, F16, Q8_0, Q4_0 tensor types

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>

#define GGUF_MAGIC 0x46554747  // "GGUF"

// Tensor data types
#define GGUF_TYPE_F32   0
#define GGUF_TYPE_F16   1
#define GGUF_TYPE_Q4_0  2
#define GGUF_TYPE_Q4_1  3
#define GGUF_TYPE_Q5_0  6
#define GGUF_TYPE_Q8_0  8
#define GGUF_TYPE_Q4_K 12
#define GGUF_TYPE_Q6_K 14

#define GGUF_MAX_TENSORS 512
#define GGUF_MAX_NAME    128
#define GGUF_MAX_KV      128

typedef struct {
    char     name[GGUF_MAX_NAME];
    uint32_t ndim;
    uint64_t shape[4];
    uint32_t dtype;
    uint64_t offset;     // offset from data section start
    uint64_t n_elements;
} gguf_tensor_info;

typedef struct {
    char     key[GGUF_MAX_NAME];
    uint32_t type;
    union {
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint8_t  b;
        char     str[256];
        uint64_t u64;
    } val;
} gguf_kv;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    // Parsed metadata
    gguf_kv        kv[GGUF_MAX_KV];
    int            n_kv_parsed;

    // Tensor directory
    gguf_tensor_info tensors[GGUF_MAX_TENSORS];

    // Data section
    uint8_t*       data;          // mmap'd or loaded raw bytes
    uint64_t       data_offset;   // file offset where tensor data starts

    // Architecture params (extracted from metadata)
    int  n_layers;
    int  n_heads;
    int  n_kv_heads;
    int  embed_dim;
    int  ffn_dim;
    int  vocab_size;
    int  ctx_len;
    float rope_freq_base;
    float rms_eps;
    char arch[64];
} gguf_file;

// Open and parse a GGUF file. Returns NULL on failure.
gguf_file* gguf_open(const char* path);

// Free GGUF file
void gguf_close(gguf_file* gf);

// Find tensor by name. Returns index or -1.
int gguf_find_tensor(const gguf_file* gf, const char* name);

// Dequantize tensor to float32 array. Caller must free returned pointer.
// Handles: F32 (copy), F16 (convert), Q8_0 (dequant), Q4_0 (dequant).
float* gguf_dequant(const gguf_file* gf, int tensor_idx);

// Get metadata value by key. Returns NULL if not found.
const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key);

// Print GGUF summary
void gguf_print_info(const gguf_file* gf);

#endif // GGUF_H
