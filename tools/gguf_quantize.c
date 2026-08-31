/*
 * gguf_quantize.c — f32/f16 GGUF in, packed GGUF out, without leaving this tree.
 *
 * notorch could read every quantized format it runs and produce none of them: a model came
 * out of training as floats and had to be handed to llama-quantize before this library
 * could run it at speed. That is a strange shape for a stack whose point is not depending on
 * the other one. This closes it — train here, quantize here, run here.
 *
 * What it does NOT do is invent a format. The blocks it writes are llama.cpp's, bit for bit,
 * because a file only earns the name GGUF if everything else can read it too; the gate in
 * tests/test_quantize.c is a byte comparison against a file llama-quantize produced from the
 * same weights, not a tolerance.
 *
 * Policy, matching llama-quantize --pure: every 2-D tensor whose row length divides 32 goes
 * to the target type, everything else — norms, 1-D biases, anything that does not divide —
 * is copied through untouched. The metadata section is copied verbatim, so the tokenizer,
 * the chat template and every architecture key survive exactly as they were; only the tensor
 * directory is rewritten, because types and offsets are the two things quantization moves.
 *
 * Build: make gguf_quantize
 * Run:   ./gguf_quantize <in.gguf> <out.gguf> [q4_0|q5_0|q8_0]
 */
#include "gguf.h"
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_ALIGN 32

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static int type_from_name(const char *s) {
    if (!strcmp(s, "q4_0") || !strcmp(s, "Q4_0")) return GGUF_TYPE_Q4_0;
    if (!strcmp(s, "q5_0") || !strcmp(s, "Q5_0")) return GGUF_TYPE_Q5_0;
    if (!strcmp(s, "q8_0") || !strcmp(s, "Q8_0")) return GGUF_TYPE_Q8_0;
    if (!strcmp(s, "q4_k") || !strcmp(s, "Q4_K")) return GGUF_TYPE_Q4_K;
    if (!strcmp(s, "q6_k") || !strcmp(s, "Q6_K")) return GGUF_TYPE_Q6_K;
    return -1;
}

static const char *type_name(uint32_t t) {
    switch (t) {
    case GGUF_TYPE_F32:  return "f32";
    case GGUF_TYPE_F16:  return "f16";
    case GGUF_TYPE_Q4_0: return "q4_0";
    case GGUF_TYPE_Q5_0: return "q5_0";
    case GGUF_TYPE_Q8_0: return "q8_0";
    case GGUF_TYPE_Q4_K: return "q4_K";
    case GGUF_TYPE_Q6_K: return "q6_K";
    default:             return "?";
    }
}

/* Packed size of n elements. Only the formats this tool emits plus the ones it may copy. */
static uint64_t type_size(uint32_t t, uint64_t n) {
    switch (t) {
    case GGUF_TYPE_F32:  return n * 4;
    case GGUF_TYPE_F16:  return n * 2;
    case GGUF_TYPE_Q4_0: return n / 32 * 18;
    case GGUF_TYPE_Q5_0: return n / 32 * 22;
    case GGUF_TYPE_Q8_0: return n / 32 * 34;
    case GGUF_TYPE_Q4_K: return n / 256 * 144;
    case GGUF_TYPE_Q6_K: return n / 256 * 210;
    default:             return 0;
    }
}

static int copy_range(FILE *in, FILE *out, uint64_t off, uint64_t len) {
    if (fseek(in, (long)off, SEEK_SET) != 0) return -1;
    char buf[1 << 16];
    while (len) {
        size_t want = len < sizeof(buf) ? (size_t)len : sizeof(buf);
        if (fread(buf, 1, want, in) != want) return -1;
        if (fwrite(buf, 1, want, out) != want) return -1;
        len -= want;
    }
    return 0;
}

static int write_u32(FILE *f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1 ? 0 : -1; }
static int write_u64(FILE *f, uint64_t v) { return fwrite(&v, 8, 1, f) == 1 ? 0 : -1; }
static int write_str(FILE *f, const char *s) {
    uint64_t n = strlen(s);
    if (write_u64(f, n)) return -1;
    return fwrite(s, 1, (size_t)n, f) == n ? 0 : -1;
}

static int pad_to(FILE *f, uint64_t target) {
    long pos = ftell(f);
    if (pos < 0 || (uint64_t)pos > target) return -1;
    static const char zeros[GGUF_ALIGN] = {0};
    uint64_t need = target - (uint64_t)pos;
    while (need) {
        size_t want = need < sizeof(zeros) ? (size_t)need : sizeof(zeros);
        if (fwrite(zeros, 1, want, f) != want) return -1;
        need -= want;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: %s <in.gguf> <out.gguf> [q4_0|q5_0|q8_0]\n", argv[0]);
        return 1;
    }
    int target = argc > 3 ? type_from_name(argv[3]) : GGUF_TYPE_Q4_0;
    if (target < 0) { fprintf(stderr, "unknown type %s\n", argv[3]); return 1; }

    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;
    printf("in : %s — %llu tensors, metadata ends at %llu, data at %llu\n",
           argv[1], (unsigned long long)gf->n_tensors,
           (unsigned long long)gf->kv_end, (unsigned long long)gf->data_offset);

    /* Decide each tensor's fate first: the directory has to carry final offsets, and those
     * are only known once every size is. */
    uint32_t *out_type = (uint32_t *)calloc(gf->n_tensors, sizeof(uint32_t));
    uint64_t *out_off  = (uint64_t *)calloc(gf->n_tensors, sizeof(uint64_t));
    if (!out_type || !out_off) { gguf_close(gf); return 1; }

    uint64_t cursor = 0, n_quantized = 0;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const gguf_tensor_info *t = &gf->tensors[i];
        /* A K-quant needs 256 to divide the row. llama-quantize does not give up on those
         * tensors, it drops them to a block format — observed on this model, not assumed:
         * --pure Q4_K writes 24 tensors as q4_K and the other 146 as q5_0, --pure Q6_K writes
         * 24 as q6_K and 146 as q8_0. Same rule here, so the two tools produce files of the
         * same shape rather than merely comparable ones. */
        int is_k = (target == GGUF_TYPE_Q4_K || target == GGUF_TYPE_Q6_K);
        int floatsrc = (t->dtype == GGUF_TYPE_F32 || t->dtype == GGUF_TYPE_F16);
        int quantizable = (t->ndim == 2) && (t->shape[0] % 32 == 0) && floatsrc;
        uint32_t chosen = (uint32_t)target;
        if (quantizable && is_k && (t->shape[0] % 256))
            chosen = (target == GGUF_TYPE_Q4_K) ? GGUF_TYPE_Q5_0 : GGUF_TYPE_Q8_0;
        out_type[i] = quantizable ? chosen : t->dtype;
        if (quantizable) n_quantized++;
        out_off[i] = cursor;
        uint64_t sz = type_size(out_type[i], t->n_elements);
        if (!sz) {
            fprintf(stderr, "tensor %s: cannot size type %u\n", t->name, out_type[i]);
            gguf_close(gf); return 1;
        }
        cursor = align_up(cursor + sz, GGUF_ALIGN);
    }
    printf("out: %llu of %llu tensors -> %s, %.1f MiB of tensor data\n",
           (unsigned long long)n_quantized, (unsigned long long)gf->n_tensors,
           type_name((uint32_t)target), (double)cursor / (1024.0 * 1024.0));

    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); gguf_close(gf); return 1; }

    /* Header and metadata verbatim: same key count, same tensor count, same everything the
     * reader on the other side will look for. */
    if (copy_range(in, out, 0, gf->kv_end)) { fprintf(stderr, "metadata copy failed\n"); return 1; }

    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const gguf_tensor_info *t = &gf->tensors[i];
        if (write_str(out, t->name) || write_u32(out, t->ndim)) return 1;
        for (uint32_t d = 0; d < t->ndim; d++) if (write_u64(out, t->shape[d])) return 1;
        if (write_u32(out, out_type[i]) || write_u64(out, out_off[i])) return 1;
    }

    long dir_end = ftell(out);
    if (dir_end < 0) return 1;
    uint64_t data_start = align_up((uint64_t)dir_end, GGUF_ALIGN);
    if (pad_to(out, data_start)) return 1;

    float *row = NULL;
    uint64_t row_cap = 0;
    uint8_t *packed = NULL;
    uint64_t packed_cap = 0;

    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const gguf_tensor_info *t = &gf->tensors[i];
        if (pad_to(out, data_start + out_off[i])) return 1;

        if (out_type[i] == t->dtype) {                     /* untouched: raw bytes across */
            uint64_t sz = type_size(t->dtype, t->n_elements);
            if (fwrite(gf->data + t->offset, 1, (size_t)sz, out) != sz) return 1;
            continue;
        }

        uint64_t k = t->shape[0], rows = t->n_elements / k;
        if (k > row_cap) {
            free(row); row = (float *)malloc((size_t)k * sizeof(float));
            if (!row) return 1;
            row_cap = k;
        }
        uint64_t rsz = type_size(out_type[i], k);
        if (rsz > packed_cap) {
            free(packed); packed = (uint8_t *)malloc((size_t)rsz);
            if (!packed) return 1;
            packed_cap = rsz;
        }
        for (uint64_t r = 0; r < rows; r++) {
            if (gguf_dequant_row(gf, (int)i, r, row) != 0) {
                fprintf(stderr, "tensor %s: row %llu would not decode\n",
                        t->name, (unsigned long long)r);
                return 1;
            }
            if (nt_quantize_row(row, packed, (int)k, (int)out_type[i]) != 0) {
                fprintf(stderr, "tensor %s: row %llu would not quantize\n",
                        t->name, (unsigned long long)r);
                return 1;
            }
            if (fwrite(packed, 1, (size_t)rsz, out) != rsz) return 1;
        }
    }

    long total = ftell(out);
    fclose(out); fclose(in);
    free(row); free(packed); free(out_type); free(out_off);
    printf("wrote %s — %.1f MiB\n", argv[2], (double)total / (1024.0 * 1024.0));
    gguf_close(gf);
    return 0;
}
