/* EmbeddingGemma 2 checkpoint to NoTorch GGUF, two files from two calls:
 *
 *   embeddinggemma2_to_gguf SNAPSHOT MODEL.gguf [--max-tokens N]
 *   embeddinggemma2_to_gguf --tokenizer tokenizer.json TOKENIZER.gguf
 *
 * The model file is the one tools/embeddinggemma2_to_gguf.py writes: the text
 * tensors with their BF16/F32 source bytes unchanged, sorted by name, the same
 * metadata in the same order and the same alignment, so for the target
 * checkpoint both converters produce the same bytes. There is no per-tensor
 * receipt; the printed output hash is the binding. The tokenizer is a file of its
 * own, so adding it does not move the model's identity. Both outputs are
 * create-only: a complete, synced .partial file is linked to the final name. */
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
    fprintf(stderr, "embeddinggemma2_to_gguf: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return 1;
}
static int u32_at(const char *obj, const char *key, uint32_t *out) {
    long long v;
    if (nj_int(nj_get(obj, key), &v) || v < 0 || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}
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
    char path[4096], partial[4200], key[GGUF_MAX_NAME];
    char *config = NULL, *revision = NULL, *header = NULL;
    static char buf[1 << 20];
    tensor *t = NULL;
    int nt = 0, rc = 1, bad = 0;
    FILE *src = NULL;
    gguf_writer *w = NULL;
    uint32_t L, E, P, D, V, H, F, positions, window, head_dim, kv_heads, l = 0;
    double eps;
    snprintf(path, sizeof(path), "%s/config.json", snapshot);
    if (!(config = nj_load(path, NULL))) goto done;
    const char *c = nj_get(config, "text_config"), *bias = nj_get(c, "attention_bias");
    if (!nj_is(nj_get(c, "model_type"), "embedding_gemma2_text")) { fail("expected embedding_gemma2_text", path); goto done; }
    if (!nj_is(nj_get(c, "hidden_activation"), "gelu_pytorch_tanh") || !bias || strncmp(bias, "false", 5)) {
        fail("only bias-free GELU-tanh text checkpoints are supported", path); goto done;
    }
    if (u32_at(c, "num_hidden_layers", &L) || u32_at(c, "hidden_size", &E) ||
        u32_at(c, "hidden_size_per_layer_input", &P) || u32_at(c, "embedding_dim", &D) ||
        u32_at(c, "vocab_size", &V) || u32_at(c, "num_attention_heads", &H) ||
        u32_at(c, "intermediate_size", &F) || u32_at(c, "max_position_embeddings", &positions) ||
        u32_at(c, "sliding_window", &window) || u32_at(c, "head_dim", &head_dim) ||
        u32_at(c, "num_key_value_heads", &kv_heads) || nj_number(nj_get(c, "rms_norm_eps"), &eps) || !E) {
        fail("text_config lacks a field this converter writes", path); goto done;
    }
    if (max_tokens < 1 || max_tokens > 8192 || max_tokens > positions) { fail("max-tokens must be within 1..8192 and model positions", NULL); goto done; }
    if (3 + 11 + 4 * (uint64_t)L > GGUF_MAX_KV) { fail("metadata exceeds NoTorch parser limit", NULL); goto done; }
    if (!(w = start(out, partial, sizeof(partial)))) goto done;
    gguf_write_kv_str(w, "general.architecture", "embeddinggemma2");
    gguf_write_kv_str(w, "general.name", "EmbeddingGemma 2 text, unchanged source weights");
    snprintf(path, sizeof(path), "%s/SOURCE_REVISION", snapshot);
    if (!access(path, F_OK)) {
        if (!(revision = nj_load(path, NULL))) goto done;
        char *s = revision, *e = s + strlen(s);
        while (*s && strchr(" \t\n\r\v\f", *s)) s++;
        while (e > s && strchr(" \t\n\r\v\f", e[-1])) *--e = 0;
        gguf_write_kv_str(w, "general.source.revision", s);
    }
    const char *names[] = {"block_count", "embedding_length", "output_length", "vocab_size", "context_length",
                           "attention.head_count", "feed_forward_length", "ple_length"};
    uint32_t values[] = {L, E, D, V, max_tokens, H, F, P};
    for (int i = 0; i < 8; i++) {
        snprintf(key, sizeof(key), "embeddinggemma2.%s", names[i]);
        gguf_write_kv_u32(w, key, values[i]);
    }
    gguf_write_kv_f32(w, "embeddinggemma2.rms_epsilon", (float)eps);
    gguf_write_kv_f32(w, "embeddinggemma2.embed_scale", (float)sqrt((double)E));
    gguf_write_kv_f32(w, "embeddinggemma2.ple_scale", (float)pow((double)E, -0.5));
    const char *overrides = nj_get(c, "per_layer_config"), *ropes = nj_get(c, "rope_parameters");
    for (const char *v = nj_first(nj_get(c, "layer_types"), &bad); v; v = nj_next(v, ']', &bad), l++) {
        int sliding = nj_is(v, "sliding_attention"), b = 0;
        char tag[16];
        uint32_t hd = head_dim, kv = kv_heads;
        double theta;
        snprintf(tag, sizeof(tag), "%02u", l);
        const char *o = nj_get(overrides, tag), *rope = nj_get(ropes, sliding ? "sliding_attention" : "full_attention");
        for (const char *k = o ? nj_first(o, &b) : NULL, *x; k && (x = nj_value(k, &b)); k = nj_next(x, '}', &b))
            if (!nj_is(k, "head_dim") && !nj_is(k, "num_key_value_heads")) b = 1;
        if (b || (nj_get(o, "head_dim") && u32_at(o, "head_dim", &hd)) ||
            (nj_get(o, "num_key_value_heads") && u32_at(o, "num_key_value_heads", &kv))) {
            fail("unsupported layer override", tag); goto done;
        }
        if (!sliding && !nj_is(v, "full_attention")) { fail("unsupported attention type", tag); goto done; }
        for (const char *k = nj_first(rope, &b), *x; k && (x = nj_value(k, &b)); k = nj_next(x, '}', &b))
            if (!nj_is(k, "rope_type") && !nj_is(k, "rope_theta")) b = 1;
        if (b || !nj_is(nj_get(rope, "rope_type"), "default") || nj_number(nj_get(rope, "rope_theta"), &theta)) {
            fail("only unscaled default RoPE is supported", tag); goto done;
        }
        const char *lk[] = {"head_dim", "kv_heads", "window"};
        uint32_t lv[] = {hd, kv, sliding ? window : 0};
        for (int i = 0; i < 3; i++) {
            snprintf(key, sizeof(key), "embeddinggemma2.layer.%u.%s", l, lk[i]);
            gguf_write_kv_u32(w, key, lv[i]);
        }
        snprintf(key, sizeof(key), "embeddinggemma2.layer.%u.rope_base", l);
        gguf_write_kv_f32(w, key, (float)theta);
    }
    if (bad || l != L) { fail("layer_types length mismatch", NULL); goto done; }

    snprintf(path, sizeof(path), "%s/model.safetensors", snapshot);
    uint64_t hsize = 0;
    struct stat st;
    if (!(src = fopen(path, "rb")) || fstat(fileno(src), &st) || st.st_size < 8 || fread(&hsize, 8, 1, src) != 1 ||
        hsize > (uint64_t)st.st_size - 8 || !(header = malloc(hsize + 1)) || fread(header, 1, hsize, src) != hsize ||
        memchr(header, 0, hsize)) {
        fail("unreadable safetensors header", path); goto done;
    }
    header[hsize] = 0;
    for (const char *k = nj_first(header, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        if (strncmp(k + 1, "language_model.", 15)) continue;
        tensor *nt_ = realloc(t, (size_t)(nt + 1) * sizeof(*t));
        if (!nt_) goto done;
        t = nt_;
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
    if (nt != 5 + (int)L * 17) { fail("unexpected number of text tensors", NULL); goto done; }
    qsort(t, (size_t)nt, sizeof(*t), by_name);
    for (int i = 0; i < nt; i++) {
        uint64_t dims[2] = {t[i].shape[t[i].ndim - 1], t[i].shape[0]};
        gguf_write_tensor_decl(w, t[i].name + 15, t[i].ndim, dims, t[i].dtype);
    }
    for (int i = 0; i < nt; i++) {
        uint64_t left = t[i].hi - t[i].lo;
        if (gguf_write_tensor_begin(w, t[i].name + 15) || fseeko(src, (off_t)(8 + hsize + t[i].lo), SEEK_SET)) goto done;
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
    free(t); free(header); free(revision); free(config);
    return rc;
}

static int special_id(const char *specials, const char *name, long long n, long long *id) {
    int bad = 0;
    const char *x = nj_first(nj_get(nj_get(specials, name), "ids"), &bad);
    return !x || nj_int(x, id) || *id < 0 || *id >= n || nj_next(x, ']', &bad) || bad ? -1 : 0;
}
static int convert_tokenizer(const char *json_path, const char *out) {
    size_t size = 0;
    char *j = nj_load(json_path, &size), **tokens = NULL, **merges = NULL, *bos_name = NULL, *eos_name = NULL;
    char partial[4200], hex[65];
    int32_t *types = NULL;
    long long nv = 0, nm = 0, bos = -1, eos = -1;
    int rc = 1, bad = 0;
    gguf_writer *w = NULL;
    unsigned char d[32];
    if (!j) return 1;
    const char *model = nj_get(j, "model"), *norm = nj_get(j, "normalizer"), *pre = nj_get(j, "pre_tokenizer");
    const char *fallback = nj_get(model, "byte_fallback");
    if (!nj_is(nj_get(model, "type"), "BPE") || !fallback || strncmp(fallback, "true", 4) ||
        !nj_is(nj_get(norm, "type"), "Replace") || !nj_is(nj_get(nj_get(norm, "pattern"), "String"), " ") ||
        !nj_is(nj_get(norm, "content"), "\xE2\x96\x81") ||
        (pre && strncmp(pre, "null", 4) && (!nj_is(nj_get(pre, "type"), "Split") || !nj_is(nj_get(nj_get(pre, "pattern"), "String"), " ")))) {
        fail("not the Gemma 4 scheme: BPE with byte fallback, spaces replaced by U+2581, no other split", json_path); goto done;
    }
    const char *vocab = nj_get(model, "vocab"), *list = nj_get(model, "merges");
    for (const char *k = nj_first(vocab, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) nv++;
    for (const char *v = nj_first(list, &bad); v; v = nj_next(v, ']', &bad)) nm++;
    if (bad || nv < 1 || nv > GGUF_MAX_STR_ARRAY || nm < 1 || nm > GGUF_MAX_STR_ARRAY ||
        !(tokens = calloc((size_t)nv, sizeof(*tokens))) || !(types = malloc((size_t)nv * sizeof(*types))) ||
        !(merges = calloc((size_t)nm, sizeof(*merges)))) {
        fail("vocabulary or merge list missing or too large", json_path); goto done;
    }
    for (const char *k = nj_first(vocab, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        long long id;
        char *s = nj_string(k);
        if (!s || nj_int(v, &id) || id < 0 || id >= nv || tokens[id]) { free(s); fail("vocabulary is not a dense map of distinct IDs", json_path); goto done; }
        tokens[id] = s;
        types[id] = 1;
    }
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
    for (const char *v = nj_first(nj_get(j, "added_tokens"), &bad); v; v = nj_next(v, ']', &bad)) {
        long long id;
        char *s = nj_string(nj_get(v, "content"));
        const char *special = nj_get(v, "special");
        int ok = s && !nj_int(nj_get(v, "id"), &id) && id >= 0 && id < nv && !strcmp(s, tokens[id]) &&
                 special && !strncmp(special, "true", 4);
        free(s);
        if (!ok) { fail("an added token is not a special token of the vocabulary", json_path); goto done; }
        types[id] = 3; /* CONTROL: bpe.c never matches it in text; text mode refuses its literal */
    }
    const char *pp = nj_get(j, "post_processor"), *specials = nj_get(pp, "special_tokens"), *e0, *e1 = NULL, *e2 = NULL;
    if ((e0 = nj_first(nj_get(pp, "single"), &bad)) && (e1 = nj_next(e0, ']', &bad))) e2 = nj_next(e1, ']', &bad);
    bos_name = nj_string(nj_get(nj_get(e0, "SpecialToken"), "id"));
    eos_name = nj_string(nj_get(nj_get(e2, "SpecialToken"), "id"));
    if (bad || !nj_is(nj_get(pp, "type"), "TemplateProcessing") || !e2 || nj_next(e2, ']', &bad) || bad ||
        !nj_is(nj_get(nj_get(e1, "Sequence"), "id"), "A") || !bos_name || !eos_name ||
        special_id(specials, bos_name, nv, &bos) || special_id(specials, eos_name, nv, &eos) ||
        strcmp(tokens[bos], bos_name) || strcmp(tokens[eos], eos_name)) {
        fail("expected the single-sequence template <bos> A <eos>", json_path); goto done;
    }
    if (nt_sha256(j, size, d)) goto done;
    for (int k = 0; k < 32; k++) snprintf(hex + 2 * k, 3, "%02x", d[k]);
    if (!(w = start(out, partial, sizeof(partial)))) goto done;
    gguf_write_kv_str(w, "general.architecture", "embeddinggemma2");
    gguf_write_kv_str(w, "general.name", "EmbeddingGemma 2 tokenizer");
    gguf_write_kv_str(w, "general.source.sha256", hex);
    gguf_write_kv_str(w, "tokenizer.ggml.model", "gemma4");
    gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", (const char *const *)tokens, (uint64_t)nv);
    gguf_write_kv_i32_array(w, "tokenizer.ggml.token_type", types, (uint64_t)nv);
    gguf_write_kv_str_array(w, "tokenizer.ggml.merges", (const char *const *)merges, (uint64_t)nm);
    gguf_write_kv_u32(w, "tokenizer.ggml.bos_token_id", (uint32_t)bos);
    gguf_write_kv_u32(w, "tokenizer.ggml.eos_token_id", (uint32_t)eos);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_bos_token", 1);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_eos_token", 1);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_space_prefix", 0);
    int closed = gguf_write_close(w);
    w = NULL;
    rc = closed ? 1 : publish(partial, out, 0);
done:
    if (w) gguf_write_abort(w);
    for (long long k = 0; tokens && k < nv; k++) free(tokens[k]);
    for (long long k = 0; merges && k < nm; k++) free(merges[k]);
    free(tokens); free(merges); free(types); free(bos_name); free(eos_name); free(j);
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
