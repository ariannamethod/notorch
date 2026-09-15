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
#define GGUF_TYPE_BF16 30

#define GGUF_MAX_TENSORS 2048   // covers Llama-70B-class (~723 tensors); loader fails loud beyond this
#define GGUF_MAX_NAME    128
#define GGUF_MAX_KV      128
#define GGUF_MAX_STR_ARRAY (1u<<21)  // 2M — covers 256K-vocab tokenizers; reject crafted huge alen

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
    uint64_t       kv_end;        // file offset just past the metadata section
    uint8_t*       data;          // raw tensor bytes: a read-only mapped view of the
                                  // file, or a posix_memalign copy when mapping is off
    uint64_t       data_offset;   // file offset where tensor data starts
    uint64_t       data_size;     // bytes readable through `data`; page-rounded on the
                                  // copy path, where Metal NoCopy needs the rounding
    void*          map_base;      // mmap base, NULL when `data` is a heap copy
    uint64_t       map_len;       // length to hand back to munmap

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
/* Packed bytes for n elements of a dtype, or 0 for a dtype this reader cannot size.
 * A 3-D expert tensor is stored as one matrix, so slicing an expert out of it needs to
 * know how far a row is — the only reason this is public. */
uint64_t gguf_type_size(uint32_t dtype, uint64_t n_elements);

float* gguf_dequant(const gguf_file* gf, int tensor_idx);

// One row of a packed tensor, decoded in place of the whole thing. dst needs shape[0]
// floats. Meant for embedding lookups, where the table is often the largest tensor in
// the file and exactly one of its rows is wanted per token. Returns 0, or -1 on a bad
// index, a row past the end, a row that is not a whole number of blocks, or a dtype with
// no decoder.
int gguf_dequant_row(const gguf_file* gf, int tensor_idx, uint64_t row, float* dst);

// Get metadata value by key. Returns NULL if not found.
const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key);

// Read a GGUF type-9 string array (e.g. "tokenizer.ggml.tokens" / ".merges") by key.
// Re-scans the file (arrays are skipped during gguf_open). Returns malloc'd char**
// of *out_n strdup'd strings, or NULL if absent. Caller frees each string + the array.
char** gguf_read_str_array(const char* path, const char* key, int* out_n);

// One string metadata value by key, read without loading tensor data. Returns 0 on success.
int gguf_read_str_kv(const char* path, const char* key, char* out, int cap);

// One integer metadata value by key (u32, i32, bool or u64). Returns 0 on success.
int gguf_read_uint_kv(const char* path, const char* key, uint64_t* out);

// Same for a type-9 INT32/UINT32 array (e.g. "tokenizer.ggml.token_type").
int32_t* gguf_read_i32_array(const char* path, const char* key, int* out_n);

// Same for a type-9 FLOAT32 array (e.g. "tokenizer.ggml.scores"). Returns a
// malloc'd float* of *out_n values, or NULL if the key is absent or not f32.
float* gguf_read_f32_array(const char* path, const char* key, int* out_n);

// Print GGUF summary
void gguf_print_info(const gguf_file* gf);

// ── Writing ──────────────────────────────────────────────────────────────────
/* The other direction. Everything above reads a GGUF that something else produced —
 * llama.cpp's converter, or gguf_quantize, which rewrites a file it was given. Neither
 * makes one out of weights that were never in a GGUF, and an organism that trains here
 * and wants its checkpoint mappable has nowhere to put it: nt_save writes notorch's own
 * [magic][n][ndim,shape,data] format, which nothing else reads and which cannot be
 * mapped tensor by tensor. This writes the format this file already parses.
 *
 * Two phases, because the format demands it. A GGUF carries its whole tensor directory —
 * name, shape, dtype and the offset of the bytes — before any of the bytes, so no offset
 * is known until every tensor has been declared. So: declare everything, then deliver the
 * data in declaration order. The first tensor-data call ends the declaration phase, writes
 * the header, the metadata and the directory, and pads to the data section; after that a
 * gguf_write_kv_* is refused.
 *
 *     gguf_writer *w = gguf_write_open("weights.gguf");
 *     gguf_write_kv_str(w, "general.architecture", "molequla");
 *     gguf_write_kv_u32(w, "molequla.block_count", 5);
 *     gguf_write_tensor_decl(w, "token_embd.weight", 2, (uint64_t[]){224, 750}, GGUF_TYPE_F32);
 *     ...
 *     gguf_write_tensor_f32(w, "token_embd.weight", tok_embd, 750 * 224);
 *     ...
 *     if (gguf_write_close(w)) { ... }                 // close is the last gate
 *
 * Nothing is buffered except the metadata and the directory, which are kilobytes: tensor
 * bytes go from the caller's pointer to the file with no copy in between, and the chunked
 * entry points let a tensor larger than memory be produced a piece at a time. Writing the
 * 19 337 632-byte molequla stage-4 set costs 5.9 MB of resident set, which is the test's
 * own largest tensor and not the file.
 *
 * Alignment is 32 and is not an option, deliberately. gguf_open computes the data offset
 * as (pos + 31) & ~31 without consulting general.alignment (gguf.c, "Data section starts
 * at aligned offset"), so a writer that honoured a different alignment would produce files
 * this library cannot read. 32 is also the GGUF default, so the key is left unwritten.
 *
 * What the reader on the other side can give back, and therefore what a round trip should
 * expect: keys are truncated to GGUF_MAX_NAME, so a longer one is refused here rather than
 * silently shortened there; a string value longer than 255 bytes is dropped by gguf_open's
 * kv union and readable only through gguf_read_str_kv; arrays are skipped by gguf_open
 * entirely and come back through gguf_read_str_array / _i32_array / _f32_array.
 *
 * Every integer goes to the file little-endian, byte for byte as this machine holds it,
 * which is what the reader's fread assumes on the way back in. Neither side is portable
 * to a big-endian host and saying so is cheaper than pretending.
 *
 * Returns: 0 on success, -1 on failure, from every call. A failed writer stays failed —
 * later calls return -1 without touching the file — so a caller may check once at close.
 * gguf_write_close returns -1 and removes the file if a declared tensor was never
 * delivered; a partial GGUF is worse than no GGUF, because it loads. */
typedef struct gguf_writer gguf_writer;

// Create (truncating) `path`. NULL if it cannot be opened.
gguf_writer* gguf_write_open(const char* path);

// Metadata, declaration phase only. `key` must be shorter than GGUF_MAX_NAME.
int gguf_write_kv_str (gguf_writer* w, const char* key, const char* val);
int gguf_write_kv_u32 (gguf_writer* w, const char* key, uint32_t val);
int gguf_write_kv_i32 (gguf_writer* w, const char* key, int32_t val);
int gguf_write_kv_u64 (gguf_writer* w, const char* key, uint64_t val);
int gguf_write_kv_f32 (gguf_writer* w, const char* key, float val);
int gguf_write_kv_bool(gguf_writer* w, const char* key, int val);
int gguf_write_kv_str_array(gguf_writer* w, const char* key, const char* const* vals, uint64_t n);
int gguf_write_kv_i32_array(gguf_writer* w, const char* key, const int32_t* vals, uint64_t n);
int gguf_write_kv_f32_array(gguf_writer* w, const char* key, const float* vals, uint64_t n);

/* Declare one tensor: 1 to 4 dimensions, shape[0] fastest, any dtype gguf_type_size can
 * size. A duplicate name is refused — the reader resolves a name by linear search and
 * would hand back the first of them forever, which is a tensor silently missing from a
 * model that loads. So is an element count that is not a whole number of blocks for a
 * packed dtype, for the reason gguf_dequant_row gives. */
int gguf_write_tensor_decl(gguf_writer* w, const char* name, uint32_t ndim,
                           const uint64_t* shape, uint32_t dtype);

/* Data phase. `name` must be the next undelivered tensor in declaration order: the file
 * is written forward, and a caller who reorders the data would be writing one tensor's
 * bytes at another's offset. */
int gguf_write_tensor(gguf_writer* w, const char* name, const void* bytes, uint64_t nbytes);

/* f32 in, declared dtype out: a straight write for GGUF_TYPE_F32, rounded to nearest
 * even for GGUF_TYPE_F16. Packed dtypes are refused — quantize with nt_quantize_row and
 * hand the blocks to gguf_write_tensor, which is what gguf_quantize does. */
int gguf_write_tensor_f32(gguf_writer* w, const char* name, const float* src, uint64_t n);

/* The same tensor a piece at a time, for data that is produced rather than held. Between
 * begin and end the writer accepts any number of chunks; end refuses a total that is not
 * the declared size, so a truncated producer fails at the tensor rather than at the file. */
int gguf_write_tensor_begin(gguf_writer* w, const char* name);
int gguf_write_tensor_chunk(gguf_writer* w, const void* bytes, uint64_t nbytes);
int gguf_write_tensor_chunk_f32(gguf_writer* w, const float* src, uint64_t n);
int gguf_write_tensor_end(gguf_writer* w);

// Finish the file and free the writer. Returns 0, or -1 having removed a file that
// would have been incomplete. The writer is freed either way.
int gguf_write_close(gguf_writer* w);

// Give up: free the writer and remove the partial file.
void gguf_write_abort(gguf_writer* w);

#endif // GGUF_H
