/* Qwen3-Embedding checkpoint to NoTorch GGUF, two files from two calls:
 *
 *   qwen3embedding_to_gguf SNAPSHOT MODEL.gguf [--max-tokens N]
 *   qwen3embedding_to_gguf --tokenizer tokenizer.json TOKENIZER.gguf
 *
 * The model file holds the source tensors with their BF16/F32 bytes unchanged,
 * sorted by name, and the architecture metadata the encoder reads. The tokenizer
 * is a file of its own, so it does not move the model's identity. Both outputs
 * are create-only: a complete, synced .partial file is linked to the final name,
 * and the printed SHA-256 is the binding. */
#include "gguf.h"
#include "sha256.h"
#include "tools/narrow_json.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int fail(const char *what, const char *detail) {
    fprintf(stderr, "qwen3embedding_to_gguf: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return 1;
}
static int u32_at(const char *obj, const char *key, uint32_t *out) {
    long long v;
    if (nj_int(nj_get(obj, key), &v) || v < 1 || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}
static int literal(const char *v, const char *word) { return v && !strncmp(v, word, strlen(word)); }
static gguf_writer *start(const char *out, char *partial, size_t cap) {
    struct stat st;
    if (snprintf(partial, cap, "%s.partial", out) >= (int)cap) return NULL;
    if (!stat(out, &st)) { fail("output exists", out); return NULL; }
    int fd = open(partial, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { perror(partial); return NULL; }
    close(fd);
    gguf_writer *w = gguf_write_open(partial);
    if (!w) unlink(partial);
    return w;
}
static int publish(const char *partial, const char *out, int tensors) {
    int fd = open(partial, O_RDONLY), ok = 0;
    struct stat st;
    unsigned char d[32];
    if (fd >= 0 && !fsync(fd) && !fstat(fd, &st) && st.st_size > 0) {
        void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ok = map != MAP_FAILED && !nt_sha256(map, (size_t)st.st_size, d);
        if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    }
    if (fd >= 0) close(fd);
    if (!ok || link(partial, out)) { perror(out); unlink(partial); return 1; }
    unlink(partial);
    printf("{\"tensors\": %d, \"sha256\": \"", tensors);
    for (int i = 0; i < 32; i++) printf("%02x", d[i]);
    printf("\"}\n");
    return 0;
}

typedef struct { char *name; uint64_t lo, hi, shape[2]; uint32_t ndim, dtype; } tensor;
static int by_name(const void *a, const void *b) {
    return strcmp(((const tensor *)a)->name, ((const tensor *)b)->name);
}
static int convert_model(const char *snapshot, const char *out, uint32_t max_tokens) {
    char path[4096], partial[4200];
    char *config = NULL, *header = NULL;
    static char buf[1 << 20];
    tensor *t = NULL;
    int nt = 0, rc = 1, bad = 0;
    FILE *src = NULL;
    gguf_writer *w = NULL;
    uint32_t L, E, F, V, H, KV, HD, positions;
    double eps, theta;
    snprintf(path, sizeof(path), "%s/config.json", snapshot);
    if (!(config = nj_load(path, NULL))) goto done;
    const char *c = config;
    if (!nj_is(nj_get(c, "model_type"), "qwen3") || !nj_is(nj_get(c, "hidden_act"), "silu") ||
        !literal(nj_get(c, "attention_bias"), "false") || !literal(nj_get(c, "rope_scaling"), "null") ||
        !literal(nj_get(c, "use_sliding_window"), "false") || !literal(nj_get(c, "tie_word_embeddings"), "true")) {
        fail("expected a bias-free SiLU qwen3 checkpoint with unscaled RoPE, no sliding window, tied embeddings", path);
        goto done;
    }
    if (u32_at(c, "num_hidden_layers", &L) || u32_at(c, "hidden_size", &E) || u32_at(c, "intermediate_size", &F) ||
        u32_at(c, "vocab_size", &V) || u32_at(c, "num_attention_heads", &H) ||
        u32_at(c, "num_key_value_heads", &KV) || u32_at(c, "head_dim", &HD) ||
        u32_at(c, "max_position_embeddings", &positions) || nj_number(nj_get(c, "rms_norm_eps"), &eps) ||
        nj_number(nj_get(c, "rope_theta"), &theta) || !(eps > 0) || !(theta > 0) || !isfinite(eps) || !isfinite(theta)) {
        fail("config.json lacks a field this converter writes", path); goto done;
    }
    if (max_tokens > positions) { fail("max-tokens exceeds the model's positions", NULL); goto done; }

    snprintf(path, sizeof(path), "%s/model.safetensors", snapshot);
    uint64_t hsize = 0;
    struct stat st;
    if (!(src = fopen(path, "rb")) || fstat(fileno(src), &st) || st.st_size < 8 || fread(&hsize, 8, 1, src) != 1 ||
        hsize > (64u << 20) || hsize > (uint64_t)st.st_size - 8 || !(header = malloc(hsize + 1)) ||
        fread(header, 1, hsize, src) != hsize || memchr(header, 0, hsize)) {
        fail("unreadable safetensors header", path); goto done;
    }
    header[hsize] = 0;
    for (const char *k = nj_first(header, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        if (!strncmp(k, "\"__metadata__\"", 14)) continue;
        tensor *grown = realloc(t, (size_t)(nt + 1) * sizeof(*t));
        if (!grown) goto done;
        t = grown;
        tensor *x = &t[nt++];
        memset(x, 0, sizeof(*x));
        long long lo = -1, hi = -1, dim;
        uint64_t count = 1, width = 0;
        int b = 0;
        const char *dtype = nj_get(v, "dtype"), *offsets = nj_first(nj_get(v, "data_offsets"), &b), *end;
        if (!(x->name = nj_string(k))) goto done;
        if (nj_is(dtype, "F32")) { x->dtype = GGUF_TYPE_F32; width = 4; }
        if (nj_is(dtype, "BF16")) { x->dtype = GGUF_TYPE_BF16; width = 2; }
        if (!width) { fail("unsupported dtype", x->name); goto done; }
        for (const char *s = nj_first(nj_get(v, "shape"), &b); s && !b; s = nj_next(s, ']', &b)) {
            if (x->ndim == 2 || nj_int(s, &dim) || dim <= 0) { b = 1; break; }
            x->shape[x->ndim++] = (uint64_t)dim;
            count *= (uint64_t)dim;
        }
        if (b || !x->ndim) { fail("unsupported shape", x->name); goto done; }
        if (!offsets || nj_int(offsets, &lo) || !(end = nj_next(offsets, ']', &b)) || nj_int(end, &hi) ||
            nj_next(end, ']', &b) || b || lo < 0 || hi < lo || (uint64_t)(hi - lo) != count * width ||
            8 + hsize + (uint64_t)hi > (uint64_t)st.st_size) {
            fail("invalid byte span", x->name); goto done;
        }
        x->lo = (uint64_t)lo; x->hi = (uint64_t)hi;
    }
    if (bad) { fail("malformed safetensors header", path); goto done; }
    /* embed_tokens, norm and eleven per layer; a separate lm_head would contradict the tie */
    if (nt != 2 + 11 * (int)L) { fail("unexpected number of tensors", path); goto done; }
    qsort(t, (size_t)nt, sizeof(*t), by_name);
    if (!(w = start(out, partial, sizeof(partial)))) goto done;
    gguf_write_kv_str(w, "general.architecture", "qwen3embedding");
    gguf_write_kv_str(w, "general.name", "Qwen3-Embedding text, unchanged source weights");
    const char *names[] = {"block_count", "embedding_length", "feed_forward_length", "vocab_size", "context_length",
                           "attention.head_count", "attention.head_count_kv", "attention.key_length"};
    uint32_t values[] = {L, E, F, V, max_tokens, H, KV, HD};
    char key[GGUF_MAX_NAME];
    for (int i = 0; i < 8; i++) {
        snprintf(key, sizeof(key), "qwen3embedding.%s", names[i]);
        gguf_write_kv_u32(w, key, values[i]);
    }
    gguf_write_kv_f32(w, "qwen3embedding.rms_epsilon", (float)eps);
    gguf_write_kv_f32(w, "qwen3embedding.rope.freq_base", (float)theta);
    for (int i = 0; i < nt; i++) {
        uint64_t dims[2] = {t[i].shape[t[i].ndim - 1], t[i].shape[0]};
        gguf_write_tensor_decl(w, t[i].name, t[i].ndim, dims, t[i].dtype);
    }
    for (int i = 0; i < nt; i++) {
        uint64_t left = t[i].hi - t[i].lo;
        if (gguf_write_tensor_begin(w, t[i].name) || fseeko(src, (off_t)(8 + hsize + t[i].lo), SEEK_SET)) goto done;
        while (left) {
            size_t n = fread(buf, 1, left < sizeof(buf) ? (size_t)left : sizeof(buf), src);
            if (!n) { fail("truncated source tensor", t[i].name); goto done; }
            if (gguf_write_tensor_chunk(w, buf, n)) goto done;
            left -= n;
        }
        if (gguf_write_tensor_end(w)) goto done;
    }
    int closed = gguf_write_close(w);
    w = NULL;
    rc = closed ? 1 : publish(partial, out, nt);
done:
    if (w) gguf_write_abort(w);
    if (src) fclose(src);
    for (int i = 0; i < nt; i++) free(t[i].name);
    free(t); free(header); free(config);
    return rc;
}

/* The tokenizer.json of Qwen3: NFC, the regex split below, byte-level BPE without
 * fallback, and a post-processor that appends one special token after the text. */
static const char *const QWEN3_SPLIT =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static int byte_level(const char *v) {
    return nj_is(nj_get(v, "type"), "ByteLevel") && literal(nj_get(v, "add_prefix_space"), "false") &&
           literal(nj_get(v, "use_regex"), "false");
}
static int empty_or_null(const char *v) { return v && (nj_is(v, "") || literal(v, "null")); }
static int convert_tokenizer(const char *json_path, const char *out) {
    size_t size = 0;
    char *j = nj_load(json_path, &size), **tokens = NULL, **merges = NULL, *regex = NULL, *eos_name = NULL;
    char partial[4200], hex[65];
    int32_t *types = NULL;
    long long nv = 0, nm = 0, eos = -1;
    int rc = 1, bad = 0;
    gguf_writer *w = NULL;
    unsigned char d[32];
    if (!j) return 1;
    const char *model = nj_get(j, "model"), *pre = nj_get(j, "pre_tokenizer"), *pp = nj_get(j, "post_processor");
    const char *p1 = nj_first(nj_get(pre, "pretokenizers"), &bad), *p2 = p1 ? nj_next(p1, ']', &bad) : NULL;
    const char *q1 = nj_first(nj_get(pp, "processors"), &bad), *q2 = q1 ? nj_next(q1, ']', &bad) : NULL;
    regex = nj_string(nj_get(nj_get(p1, "pattern"), "Regex"));
    if (bad || !p2 || nj_next(p2, ']', &bad) || !q2 || nj_next(q2, ']', &bad) || bad || !regex || strcmp(regex, QWEN3_SPLIT) ||
        !nj_is(nj_get(nj_get(j, "normalizer"), "type"), "NFC") || !nj_is(nj_get(pre, "type"), "Sequence") ||
        !nj_is(nj_get(p1, "type"), "Split") || !nj_is(nj_get(p1, "behavior"), "Isolated") ||
        !literal(nj_get(p1, "invert"), "false") || !byte_level(p2) || !nj_is(nj_get(pp, "type"), "Sequence") ||
        !byte_level(q1) || !nj_is(nj_get(q2, "type"), "TemplateProcessing") ||
        !nj_is(nj_get(model, "type"), "BPE") || !literal(nj_get(model, "byte_fallback"), "false") ||
        !literal(nj_get(model, "ignore_merges"), "false") || !empty_or_null(nj_get(model, "continuing_subword_prefix")) ||
        !empty_or_null(nj_get(model, "end_of_word_suffix"))) {
        fail("not the Qwen3 scheme: NFC, the Qwen3 split, byte-level BPE without fallback, a template post-processor", json_path);
        goto done;
    }
    const char *vocab = nj_get(model, "vocab"), *list = nj_get(model, "merges"), *added = nj_get(j, "added_tokens");
    for (const char *k = nj_first(vocab, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        long long id;
        if (nj_int(v, &id) || id < 0 || id >= GGUF_MAX_STR_ARRAY) { bad = 1; break; }
        if (id >= nv) nv = id + 1;
    }
    for (const char *v = nj_first(added, &bad); v && !bad; v = nj_next(v, ']', &bad)) {
        long long id;
        if (nj_int(nj_get(v, "id"), &id) || id < 0 || id >= GGUF_MAX_STR_ARRAY) { bad = 1; break; }
        if (id >= nv) nv = id + 1;
    }
    for (const char *v = nj_first(list, &bad); v; v = nj_next(v, ']', &bad)) nm++;
    if (bad || nv < 1 || nm < 1 || nm > GGUF_MAX_STR_ARRAY || !(tokens = calloc((size_t)nv, sizeof(*tokens))) ||
        !(types = calloc((size_t)nv, sizeof(*types))) || !(merges = calloc((size_t)nm, sizeof(*merges)))) {
        fail("vocabulary or merge list missing or too large", json_path); goto done;
    }
    for (const char *k = nj_first(vocab, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        long long id;
        if (nj_int(v, &id) || tokens[id] || !(tokens[id] = nj_string(k))) { fail("vocabulary is not a map of distinct IDs", json_path); goto done; }
        types[id] = 1;
    }
    for (const char *v = nj_first(added, &bad); v; v = nj_next(v, ']', &bad)) {
        long long id;
        char *s = nj_string(nj_get(v, "content"));
        const char *special = nj_get(v, "special");
        int ok = s && !nj_int(nj_get(v, "id"), &id) && (!tokens[id] || !strcmp(tokens[id], s)) &&
                 literal(nj_get(v, "lstrip"), "false") && literal(nj_get(v, "rstrip"), "false") &&
                 literal(nj_get(v, "single_word"), "false") && literal(nj_get(v, "normalized"), "false") &&
                 (literal(special, "true") || literal(special, "false"));
        if (!ok) { free(s); fail("an added token is not a plain literal", json_path); goto done; }
        free(tokens[id]);
        tokens[id] = s;
        types[id] = literal(special, "true") ? 3 : 4; /* CONTROL or USER_DEFINED; text mode refuses both literals */
    }
    for (long long i = 0; i < nv; i++) if (!tokens[i]) { fail("the vocabulary has a hole", json_path); goto done; }
    long long i = 0;
    for (const char *v = nj_first(list, &bad); v; v = nj_next(v, ']', &bad), i++) {
        int b = 0;
        const char *x = nj_first(v, &b), *y = x ? nj_next(x, ']', &b) : NULL;
        char *a = nj_string(x), *z = y ? nj_string(y) : NULL;
        if (!a || !z || nj_next(y, ']', &b) || b || strchr(a, ' ') || strchr(z, ' ') ||
            !(merges[i] = malloc(strlen(a) + strlen(z) + 2))) {
            free(a); free(z); fail("a merge is not a pair of space-free strings", json_path); goto done;
        }
        sprintf(merges[i], "%s %s", a, z);
        free(a); free(z);
    }
    /* single: [A, one special token] — the token every sequence ends with */
    const char *s0 = nj_first(nj_get(q2, "single"), &bad), *s1 = s0 ? nj_next(s0, ']', &bad) : NULL;
    eos_name = nj_string(nj_get(nj_get(s1, "SpecialToken"), "id"));
    const char *eid = nj_first(nj_get(nj_get(nj_get(q2, "special_tokens"), eos_name ? eos_name : ""), "ids"), &bad);
    if (bad || !s1 || nj_next(s1, ']', &bad) || bad || !nj_is(nj_get(nj_get(s0, "Sequence"), "id"), "A") || !eos_name ||
        !eid || nj_int(eid, &eos) || nj_next(eid, ']', &bad) || bad || eos < 0 || eos >= nv ||
        strcmp(tokens[eos], eos_name) || types[eos] != 3) {
        fail("expected the single-sequence template A <special>", json_path); goto done;
    }
    if (nt_sha256(j, size, d)) goto done;
    for (int k = 0; k < 32; k++) snprintf(hex + 2 * k, 3, "%02x", d[k]);
    if (!(w = start(out, partial, sizeof(partial)))) goto done;
    gguf_write_kv_str(w, "general.architecture", "qwen3embedding");
    gguf_write_kv_str(w, "general.name", "Qwen3-Embedding tokenizer");
    gguf_write_kv_str(w, "general.source.sha256", hex);
    gguf_write_kv_str(w, "tokenizer.ggml.model", "gpt2");
    gguf_write_kv_str(w, "tokenizer.ggml.pre", "qwen3");
    gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", (const char *const *)tokens, (uint64_t)nv);
    gguf_write_kv_i32_array(w, "tokenizer.ggml.token_type", types, (uint64_t)nv);
    gguf_write_kv_str_array(w, "tokenizer.ggml.merges", (const char *const *)merges, (uint64_t)nm);
    gguf_write_kv_u32(w, "tokenizer.ggml.eos_token_id", (uint32_t)eos);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_bos_token", 0);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_eos_token", 1);
    int closed = gguf_write_close(w);
    w = NULL;
    rc = closed ? 1 : publish(partial, out, 0);
done:
    if (w) gguf_write_abort(w);
    for (long long k = 0; tokens && k < nv; k++) free(tokens[k]);
    for (long long k = 0; merges && k < nm; k++) free(merges[k]);
    free(tokens); free(merges); free(types); free(regex); free(eos_name); free(j);
    return rc;
}

int main(int argc, char **argv) {
    uint16_t endian = 1;
    char *end;
    if (*(unsigned char *)&endian != 1) return fail("little-endian host required", NULL) + 1;
    if (argc == 4 && !strcmp(argv[1], "--tokenizer")) return convert_tokenizer(argv[2], argv[3]);
    long max_tokens = 8192;
    if (argc == 5 && !strcmp(argv[3], "--max-tokens")) {
        errno = 0;
        max_tokens = strtol(argv[4], &end, 10);
        if (errno || *end || max_tokens < 1 || max_tokens > 8192) return fail("max-tokens must be within 1..8192", NULL) + 1;
    } else if (argc != 3) {
        fprintf(stderr, "usage: %s SNAPSHOT MODEL.gguf [--max-tokens N]\n"
                        "       %s --tokenizer tokenizer.json TOKENIZER.gguf\n", argv[0], argv[0]);
        return 2;
    }
    return convert_model(argv[1], argv[2], (uint32_t)max_tokens);
}
