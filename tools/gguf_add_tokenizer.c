/* gguf_add_tokenizer.c — put a byte-level BPE vocabulary into a GGUF that has none.
 *
 * Some files carry weights and nothing else. Resonance 200M is one: 243 tensors,
 * eleven architecture keys, and a vocabulary that lives in its trainer's source
 * as integer merge pairs. A file like that cannot be read by anything but the
 * program it was written for — not this harness, not llama.cpp — so the fix is
 * to write the vocabulary into the file rather than teach every reader where
 * else to look.
 *
 * The merge list is integers because that is the id space the weights were
 * trained in: id i is byte i for i < 256, and id 256+k is the k-th merge. GGUF
 * wants strings, so each id is spelled in the GPT-2 byte-level alphabet — the
 * same table examples/bpe.c builds — and the merges are those spellings in
 * pairs. The order is the weights' order and not the alphabet's: a tokenizer
 * whose ids are sorted differently addresses different rows of the embedding
 * and produces fluent nonsense.
 *
 *   gguf_add_tokenizer in.gguf out.gguf merges.txt
 *
 * merges.txt is any text holding the pairs as integers, two per merge, in
 * order — a plain list or the C header the trainer emits both parse.
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define GGUF_ALIGN 32
#define KV_STRING  8
#define KV_ARRAY   9

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

static int write_kv_str(FILE *f, const char *key, const char *val) {
    if (write_str(f, key) || write_u32(f, KV_STRING)) return -1;
    return write_str(f, val);
}

static int write_kv_str_array(FILE *f, const char *key, char **vals, uint64_t n) {
    if (write_str(f, key) || write_u32(f, KV_ARRAY)) return -1;
    if (write_u32(f, KV_STRING) || write_u64(f, n)) return -1;
    for (uint64_t i = 0; i < n; i++) if (write_str(f, vals[i])) return -1;
    return 0;
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

/* GPT-2 byte <-> unicode, the same construction as examples/bpe.c. Bytes that
 * are not printable get a codepoint above 255 so that every token is a string
 * with no whitespace or control characters in it. */
static void byte_table(int cp[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp[b] = printable ? b : (256 + n);
        if (!printable) n++;
    }
}

static int utf8_enc(int cp, char *out) {
    if (cp < 0x80)  { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* The merge integers, in order.
 *
 * "Every number in the file" is the obvious reading and the wrong one: the
 * trainer's C header also states its vocabulary size, its merge count and the
 * model's size in its comments, and those four extra pairs make a vocabulary
 * four rows longer than the model has. So when the file brackets its pairs —
 * a C initialiser does — only what is inside a bracket counts, and a bare list
 * with no brackets is read whole. */
static int *read_ints(const char *path, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    int braced = 0;
    for (int c; (c = fgetc(f)) != EOF; ) if (c == '{') { braced = 1; break; }
    rewind(f);

    int cap = 4096, n = 0, depth = 0;
    int *v = (int *)malloc((size_t)cap * sizeof(int));
    if (!v) { fclose(f); return NULL; }
    int c, cur = 0, in_num = 0;
    while ((c = fgetc(f)) != EOF) {
        if (c == '{') { depth++; continue; }
        if (c == '}') { if (depth) depth--; }
        if (isdigit(c) && (!braced || depth)) { cur = cur * 10 + (c - '0'); in_num = 1; continue; }
        if (in_num) {
            if (n == cap) {
                cap *= 2;
                int *g = (int *)realloc(v, (size_t)cap * sizeof(int));
                if (!g) { free(v); fclose(f); return NULL; }
                v = g;
            }
            v[n++] = cur; cur = 0; in_num = 0;
        }
    }
    if (in_num && n < cap) v[n++] = cur;
    fclose(f);
    *out_n = n;
    return v;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <merges>\n", argv[0]);
        return 1;
    }
    gguf_file *gf = gguf_open(argv[1]);
    if (!gf) return 1;

    if (gguf_get_kv(gf, "tokenizer.ggml.tokens")) {
        fprintf(stderr, "%s already carries a tokenizer — refusing to write a second one\n", argv[1]);
        gguf_close(gf);
        return 1;
    }

    int nint = 0;
    int *ints = read_ints(argv[3], &nint);
    if (!ints) { gguf_close(gf); return 1; }
    int n_merges = nint / 2;
    if (nint % 2) {
        fprintf(stderr, "%s holds %d integers, which is not whole merge pairs\n", argv[3], nint);
        return 1;
    }
    int n_vocab = 256 + n_merges;
    printf("merges: %d  vocab: %d\n", n_merges, n_vocab);

    /* If the file says how large its vocabulary is, that is the check on the
     * merge list: the wrong list produces a tokenizer that indexes rows the
     * model does not have. */
    for (int i = 0; i < gf->n_kv_parsed; i++) {
        if (strstr(gf->kv[i].key, ".vocab_size")) {
            int want = (int)gf->kv[i].val.u32;
            if (want != n_vocab) {
                fprintf(stderr, "%s says vocab_size=%d, these merges make %d\n",
                        gf->kv[i].key, want, n_vocab);
                return 1;
            }
            printf("checked against %s = %d\n", gf->kv[i].key, want);
        }
    }

    int cp[256];
    byte_table(cp);
    char **tok = (char **)calloc((size_t)n_vocab, sizeof(char *));
    if (!tok) return 1;
    for (int b = 0; b < 256; b++) {
        char buf[8];
        int n = utf8_enc(cp[b], buf);
        buf[n] = 0;
        tok[b] = strdup(buf);
        if (!tok[b]) return 1;
    }
    char **merge_str = (char **)calloc((size_t)(n_merges ? n_merges : 1), sizeof(char *));
    if (!merge_str) return 1;
    for (int k = 0; k < n_merges; k++) {
        int a = ints[2 * k], b = ints[2 * k + 1];
        if (a < 0 || b < 0 || a >= 256 + k || b >= 256 + k) {
            fprintf(stderr, "merge %d refers to id %d,%d which does not exist yet\n", k, a, b);
            return 1;
        }
        size_t la = strlen(tok[a]), lb = strlen(tok[b]);
        tok[256 + k] = (char *)malloc(la + lb + 1);
        merge_str[k] = (char *)malloc(la + lb + 2);
        if (!tok[256 + k] || !merge_str[k]) return 1;
        memcpy(tok[256 + k], tok[a], la);
        memcpy(tok[256 + k] + la, tok[b], lb + 1);
        memcpy(merge_str[k], tok[a], la);
        merge_str[k][la] = ' ';
        memcpy(merge_str[k] + la + 1, tok[b], lb + 1);
    }

    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 1; }

    /* Header by hand because the key count changes; the keys themselves are
     * copied as bytes, since gguf_open skips array-valued ones and cannot give
     * them back. */
    if (write_u32(out, GGUF_MAGIC) || write_u32(out, gf->version) ||
        write_u64(out, gf->n_tensors) || write_u64(out, gf->n_kv + 3)) return 1;
    const uint64_t HDR = 24;
    if (copy_range(in, out, HDR, gf->kv_end - HDR)) {
        fprintf(stderr, "metadata copy failed\n"); return 1;
    }
    if (write_kv_str(out, "tokenizer.ggml.model", "gpt2") ||
        write_kv_str_array(out, "tokenizer.ggml.tokens", tok, (uint64_t)n_vocab) ||
        write_kv_str_array(out, "tokenizer.ggml.merges", merge_str, (uint64_t)n_merges)) {
        fprintf(stderr, "tokenizer write failed\n"); return 1;
    }

    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const gguf_tensor_info *t = &gf->tensors[i];
        if (write_str(out, t->name) || write_u32(out, t->ndim)) return 1;
        for (uint32_t d = 0; d < t->ndim; d++) if (write_u64(out, t->shape[d])) return 1;
        if (write_u32(out, t->dtype) || write_u64(out, t->offset)) return 1;
    }

    long dir_end = ftell(out);
    if (dir_end < 0) return 1;
    uint64_t data_start = ((uint64_t)dir_end + GGUF_ALIGN - 1) & ~(uint64_t)(GGUF_ALIGN - 1);
    if (pad_to(out, data_start)) return 1;

    /* Tensor offsets are relative to the data section, so the section moving
     * changes nothing inside it and it goes across whole. */
    uint64_t data_len = gf->data_size;
    if (fwrite(gf->data, 1, (size_t)data_len, out) != (size_t)data_len) {
        fprintf(stderr, "data copy failed\n"); return 1;
    }

    fclose(in);
    fclose(out);
    printf("wrote %s — tokenizer.ggml.{model,tokens,merges} added, %llu tensors untouched\n",
           argv[2], (unsigned long long)gf->n_tensors);
    gguf_close(gf);
    return 0;
}
