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
 *   gguf_add_tokenizer in.gguf out.gguf --chat "32759,32760|32761,32762|32763"
 *
 * merges.txt is any text holding the pairs as integers, two per merge, in
 * order — a plain list or the C header the trainer emits both parse.
 *
 * --chat writes the turn wrapping the family was trained with into the file, as
 * notorch.chat.{before,after,stop}, so that the harness finds it there instead of
 * being told through NT_CHAT. Ids and not strings, for the same reason the
 * harness reads them that way: the strings that spell them may not exist.
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define GGUF_ALIGN 32
#define KV_INT32   5
#define KV_STRING  8
#define KV_ARRAY   9
#define CHAT_MAX   32

/* The wrapping a family expects around a turn, written into the file as ids so
 * that nobody has to know them or type them. Ids and not strings for the same
 * reason the harness reads them that way: the strings may not exist. Janus
 * carries nine special ids above what its merge list reconstructs and only five
 * are named anywhere, but all nine are perfectly good numbers. */
typedef struct { int32_t v[CHAT_MAX]; int n; } chat_list;

static int parse_chat_ids(const char *s, chat_list *out) {
    out->n = 0;
    for (const char *p = s; *p && *p != '|' && out->n < CHAT_MAX; ) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p || *p == '|') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) return -1;
        out->v[out->n++] = (int32_t)v;
        p = end;
    }
    return 0;
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

static int write_kv_i32_array(FILE *f, const char *key, const int32_t *v, uint64_t n) {
    if (write_str(f, key) || write_u32(f, KV_ARRAY)) return -1;
    if (write_u32(f, KV_INT32) || write_u64(f, n)) return -1;
    return fwrite(v, sizeof(int32_t), (size_t)n, f) == n ? 0 : -1;
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
    const char *in = NULL, *out = NULL, *merges_path = NULL, *chat_spec = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chat") == 0 && i + 1 < argc) { chat_spec = argv[++i]; continue; }
        if (!in) in = argv[i];
        else if (!out) out = argv[i];
        else if (!merges_path) merges_path = argv[i];
    }
    if (!in || !out || (!merges_path && !chat_spec)) {
        fprintf(stderr,
            "usage: %s <in.gguf> <out.gguf> [merges] [--chat \"before|after|stop\"]\n"
            "  merges  a text holding the pairs as integers, two per merge, in order\n"
            "  --chat  the turn wrapping as id lists: what goes before the user's\n"
            "          text, what goes after it, and what ends the turn\n", argv[0]);
        return 1;
    }

    chat_list c_before = {{0},0}, c_after = {{0},0}, c_stop = {{0},0};
    if (chat_spec) {
        const char *a = chat_spec;
        const char *b = strchr(a, '|');
        const char *c = b ? strchr(b + 1, '|') : NULL;
        if (parse_chat_ids(a, &c_before) ||
            (b && parse_chat_ids(b + 1, &c_after)) ||
            (c && parse_chat_ids(c + 1, &c_stop))) {
            fprintf(stderr, "--chat: expected id lists separated by '|'\n");
            return 1;
        }
        if (!c_before.n && !c_after.n && !c_stop.n) {
            fprintf(stderr, "--chat: no ids in any of the three lists\n");
            return 1;
        }
    }

    gguf_file *gf = gguf_open(in);
    if (!gf) return 1;

    /* Through the path readers and not gguf_get_kv: the parsed table holds only
     * scalars, so asking it about an array-valued key answers "absent" for a key
     * that is right there, and the write would produce a file with the key
     * twice — with no rule about which copy a reader takes. */
    int have = 0;
    if (merges_path) {
        char **t = gguf_read_str_array(in, "tokenizer.ggml.tokens", &have);
        if (t) {
            fprintf(stderr, "%s already carries a tokenizer of %d tokens — refusing to write a second one\n", in, have);
            return 1;
        }
    }
    if (chat_spec) {
        int32_t *w = gguf_read_i32_array(in, "notorch.chat.before", &have);
        if (!w) w = gguf_read_i32_array(in, "notorch.chat.after", &have);
        if (!w) w = gguf_read_i32_array(in, "notorch.chat.stop", &have);
        if (w) {
            fprintf(stderr, "%s already carries a chat wrapping — refusing to write a second one\n", in);
            free(w);
            return 1;
        }
    }

    int n_merges = 0, n_vocab = 0;
    char **tok = NULL, **merge_str = NULL;
  if (merges_path) {
    int nint = 0;
    int *ints = read_ints(merges_path, &nint);
    if (!ints) { gguf_close(gf); return 1; }
    n_merges = nint / 2;
    if (nint % 2) {
        fprintf(stderr, "%s holds %d integers, which is not whole merge pairs\n", merges_path, nint);
        return 1;
    }
    n_vocab = 256 + n_merges;
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
    tok = (char **)calloc((size_t)n_vocab, sizeof(char *));
    if (!tok) return 1;
    for (int b = 0; b < 256; b++) {
        char buf[8];
        int n = utf8_enc(cp[b], buf);
        buf[n] = 0;
        tok[b] = strdup(buf);
        if (!tok[b]) return 1;
    }
    merge_str = (char **)calloc((size_t)(n_merges ? n_merges : 1), sizeof(char *));
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
  }

    /* An id the model has no row for would read as garbage in the embedding, so
     * a wrapping is checked against the vocabulary the file declares before it
     * is written into the file. */
    for (int i = 0; i < gf->n_kv_parsed && chat_spec; i++) {
        if (!strstr(gf->kv[i].key, ".vocab_size")) continue;
        int v = (int)gf->kv[i].val.u32;
        const chat_list *lists[3] = { &c_before, &c_after, &c_stop };
        for (int L = 0; L < 3; L++)
            for (int j = 0; j < lists[L]->n; j++)
                if (lists[L]->v[j] < 0 || lists[L]->v[j] >= v) {
                    fprintf(stderr, "--chat: id %d is outside a %d-token vocabulary\n",
                            lists[L]->v[j], v);
                    return 1;
                }
        printf("chat: %d before, %d after, %d stop — all inside %s = %d\n",
               c_before.n, c_after.n, c_stop.n, gf->kv[i].key, v);
    }

    uint64_t extra = (merges_path ? 3 : 0)
                   + (c_before.n ? 1 : 0) + (c_after.n ? 1 : 0) + (c_stop.n ? 1 : 0);

    FILE *fin = fopen(in, "rb");
    FILE *fout = fopen(out, "wb");
    if (!fin || !fout) { fprintf(stderr, "cannot open files\n"); return 1; }

    /* Header by hand because the key count changes; the keys themselves are
     * copied as bytes, since gguf_open skips array-valued ones and cannot give
     * them back. */
    if (write_u32(fout, GGUF_MAGIC) || write_u32(fout, gf->version) ||
        write_u64(fout, gf->n_tensors) || write_u64(fout, gf->n_kv + extra)) return 1;
    const uint64_t HDR = 24;
    if (copy_range(fin, fout, HDR, gf->kv_end - HDR)) {
        fprintf(stderr, "metadata copy failed\n"); return 1;
    }
    if (merges_path &&
        (write_kv_str(fout, "tokenizer.ggml.model", "gpt2") ||
         write_kv_str_array(fout, "tokenizer.ggml.tokens", tok, (uint64_t)n_vocab) ||
         write_kv_str_array(fout, "tokenizer.ggml.merges", merge_str, (uint64_t)n_merges))) {
        fprintf(stderr, "tokenizer write failed\n"); return 1;
    }
    if ((c_before.n && write_kv_i32_array(fout, "notorch.chat.before", c_before.v, (uint64_t)c_before.n)) ||
        (c_after.n  && write_kv_i32_array(fout, "notorch.chat.after",  c_after.v,  (uint64_t)c_after.n)) ||
        (c_stop.n   && write_kv_i32_array(fout, "notorch.chat.stop",   c_stop.v,   (uint64_t)c_stop.n))) {
        fprintf(stderr, "chat write failed\n"); return 1;
    }

    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const gguf_tensor_info *t = &gf->tensors[i];
        if (write_str(fout, t->name) || write_u32(fout, t->ndim)) return 1;
        for (uint32_t d = 0; d < t->ndim; d++) if (write_u64(fout, t->shape[d])) return 1;
        if (write_u32(fout, t->dtype) || write_u64(fout, t->offset)) return 1;
    }

    long dir_end = ftell(fout);
    if (dir_end < 0) return 1;
    uint64_t data_start = ((uint64_t)dir_end + GGUF_ALIGN - 1) & ~(uint64_t)(GGUF_ALIGN - 1);
    if (pad_to(fout, data_start)) return 1;

    /* Tensor offsets are relative to the data section, so the section moving
     * changes nothing inside it and it goes across whole. */
    uint64_t data_len = gf->data_size;
    if (fwrite(gf->data, 1, (size_t)data_len, fout) != (size_t)data_len) {
        fprintf(stderr, "data copy failed\n"); return 1;
    }

    fclose(fin);
    fclose(fout);
    printf("wrote %s — %s%s%llu tensors untouched\n", out,
           merges_path ? "tokenizer.ggml.{model,tokens,merges} added, " : "",
           chat_spec ? "notorch.chat.* added, " : "",
           (unsigned long long)gf->n_tensors);
    gguf_close(gf);
    return 0;
}
