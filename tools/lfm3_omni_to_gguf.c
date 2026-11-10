/* lfm3_omni_to_gguf — LiquidAI's LFM3-Omni-2B checkpoint, every tower, into one GGUF.
 *
 *   lfm3_omni_to_gguf [--q8_0] SNAPSHOT OUT.gguf
 *
 * SNAPSHOT is the official repository as published: config.json, model.safetensors,
 * tokenizer.json, processor_config.json, generation_config.json and audio_tokenizer/
 * with the codec. One file comes out: the text decoder under the names harness/arch_lfm3.c
 * reads (general.architecture = lfm3), and beside it the vision tower and projector, the
 * audio downsampler, encoder and projector, the depth decoder, the codebook embedding and
 * the codec decoder, under the names the official mmproj GGUF gives them, so a tensor here
 * and the same tensor there can be compared byte for byte. Matrices are F16 (BF16 widened
 * to F32 exactly, then rounded to nearest even); norms, biases, convolution kernels and the
 * tables computed here are F32. Two tensors the official files do not carry are added: the
 * Slaney mel bank the feature extractor builds at run time, and the codec's synthesis
 * window, which the codec checkpoint stores.
 *
 * Every source tensor is either written or named on stderr with the reason it is not; an
 * unnamed leftover refuses the conversion. The codec's encoder side is the one skipped
 * part: it turns 24 kHz speech into codes for a prompt that replays earlier speech, and
 * the harness carries the codes it generated instead of re-encoding audio. */
#include "gguf.h"
#include "notorch.h"
#include "sha256.h"
#include "tools/narrow_json.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAXT 1024

typedef struct { char *name; const uint8_t *data; int bf16; uint64_t shape[4]; int nd; uint64_t n; int used; } src_t;
typedef struct { src_t t[MAXT]; int n; const uint8_t *map; size_t size; } st_file;

static int fail(const char *what, const char *detail) {
    fprintf(stderr, "lfm3_omni_to_gguf: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return 1;
}

static int st_open(st_file *s, const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size < 8) { if (fd >= 0) close(fd); return fail("cannot read", path); }
    s->size = (size_t)st.st_size;
    s->map = mmap(NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (s->map == MAP_FAILED) { s->map = NULL; return fail("cannot map", path); }
    uint64_t hs;
    memcpy(&hs, s->map, 8);
    if (hs > (64u << 20) || hs > s->size - 8) return fail("bad safetensors header", path);
    char *h = malloc(hs + 1);
    if (!h) return fail("out of memory", path);
    memcpy(h, s->map + 8, hs);
    h[hs] = 0;
    int bad = 0;
    const uint8_t *base = s->map + 8 + hs;
    for (const char *k = nj_first(h, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        char *name = nj_string(k);
        if (!name) { bad = 1; break; }
        if (!strcmp(name, "__metadata__")) { free(name); continue; }
        if (s->n == MAXT) { free(name); bad = 1; break; }
        src_t *t = &s->t[s->n++];
        memset(t, 0, sizeof *t);
        t->name = name;
        const char *dt = nj_get(v, "dtype");
        if (nj_is(dt, "BF16")) t->bf16 = 1;
        else if (!nj_is(dt, "F32")) { bad = 1; break; }
        uint64_t n = 1;
        int b = 0;
        long long d, lo, hi;
        for (const char *e = nj_first(nj_get(v, "shape"), &b); e; e = nj_next(e, ']', &b)) {
            if (t->nd == 4 || nj_int(e, &d) || d < 1) { bad = 1; break; }
            t->shape[t->nd++] = (uint64_t)d;
            n *= (uint64_t)d;
        }
        const char *o = nj_first(nj_get(v, "data_offsets"), &b), *end = o ? nj_next(o, ']', &b) : NULL;
        if (bad || b || !t->nd || !end || nj_int(o, &lo) || nj_int(end, &hi) || lo < 0 || hi < lo ||
            (uint64_t)(hi - lo) != n * (t->bf16 ? 2 : 4) || (uint64_t)hi > s->size - 8 - hs) { bad = 1; break; }
        t->n = n;
        t->data = base + lo;
    }
    free(h);
    return bad ? fail("malformed safetensors header", path) : 0;
}

static src_t *find(st_file *s, const char *name) {
    for (int i = 0; i < s->n; i++) if (!strcmp(s->t[i].name, name)) return &s->t[i];
    return NULL;
}

static float val(const src_t *t, uint64_t i) {
    if (t->bf16) {
        uint16_t b; memcpy(&b, t->data + 2 * i, 2);
        uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
    }
    float f; memcpy(&f, t->data + 4 * i, 4); return f;
}

/* ── the plan: every output tensor, declared first and then written in the same order ── */

enum { COPY, TEMPORAL0, TEMPORAL1, STACK_EMBD, STACK_NORM, ROPE_TABLE, FSQ_TABLE, MEL_BANK };
typedef struct {
    char dst[GGUF_MAX_NAME];
    st_file *f;
    char src[160];
    int kind, f16, nd;
    uint64_t shape[4];
} out_t;

static out_t plan[MAXT];
static int nplan;
static int bad_plan;

typedef struct { int n_text_layers, rope_rot, n_cb, cb_size, depth_width; int lv[8], nlv; double *mel; } extra_t;
static extra_t X;

/* shape given in GGUF order (fastest first); 0 means "the source shape reversed" */
static out_t *add(const char *dst, st_file *f, const char *src, int kind, int f16, int nd, const uint64_t *shape) {
    if (nplan == MAXT) { bad_plan = 1; return NULL; }
    out_t *o = &plan[nplan++];
    memset(o, 0, sizeof *o);
    snprintf(o->dst, sizeof o->dst, "%s", dst);
    o->f = f; o->kind = kind; o->f16 = f16;
    if (src) snprintf(o->src, sizeof o->src, "%s", src);
    if (kind == COPY) {
        src_t *t = find(f, src);
        if (!t) { fprintf(stderr, "lfm3_omni_to_gguf: missing source tensor %s\n", src); bad_plan = 1; return o; }
        t->used = 1;
        if (nd) {
            uint64_t n = 1;
            for (int i = 0; i < nd; i++) n *= shape[i];
            if (n != t->n) { fprintf(stderr, "lfm3_omni_to_gguf: %s holds %llu values, not %llu\n", src,
                                     (unsigned long long)t->n, (unsigned long long)n); bad_plan = 1; }
            o->nd = nd; memcpy(o->shape, shape, nd * sizeof *shape);
        } else {
            o->nd = t->nd;
            for (int i = 0; i < t->nd; i++) o->shape[i] = t->shape[t->nd - 1 - i];
        }
    } else {
        o->nd = nd; memcpy(o->shape, shape, nd * sizeof *shape);
    }
    return o;
}

#define S1(a) ((const uint64_t[]){a})
#define S2(a, b) ((const uint64_t[]){a, b})
#define S3(a, b, c) ((const uint64_t[]){a, b, c})
#define S4(a, b, c, d) ((const uint64_t[]){a, b, c, d})

static void lin(const char *dst, st_file *f, const char *src) { add(dst, f, src, COPY, 1, 0, NULL); }
static void vec(const char *dst, st_file *f, const char *src) {
    src_t *t = find(f, src);
    add(dst, f, src, COPY, 0, 1, S1(t ? t->n : 0));
}

/* An LFM-family block: operator norm, ffn norm, optional sandwich norms, and either a short
 * convolution or attention (gate optional). Destination names follow the mmproj towers. */
static void lfm_block(const char *dp, st_file *f, const char *sp, int width, int conv, int lfm2_names) {
    char d[160], s[200];
#define D(x) (snprintf(d, sizeof d, "%s.%s", dp, x), d)
#define S(x) (snprintf(s, sizeof s, "%s.%s", sp, x), s)
    vec(D("ln1.weight"), f, S("operator_norm.weight"));
    vec(D("ln2.weight"), f, S("ffn_norm.weight"));
    if (find(f, S("post_operator_norm.weight"))) {
        vec(D("attn_post_norm.weight"), f, S("post_operator_norm.weight"));
        vec(D("ffn_post_norm.weight"), f, S("post_ffn_norm.weight"));
    }
    lin(D("ffn_gate.weight"), f, S("feed_forward.w1.weight"));
    lin(D("ffn_up.weight"), f, S("feed_forward.w3.weight"));
    lin(D("ffn_down.weight"), f, S("feed_forward.w2.weight"));
    if (conv) {
        src_t *cw = find(f, S("conv.conv.weight"));
        add(D("conv_dw.weight"), f, s, COPY, 0, 2, S2(cw ? cw->shape[2] : 0, (uint64_t)width));
        lin(D("conv_pw1.weight"), f, S("conv.in_proj.weight"));
        lin(D("conv_pw2.weight"), f, S("conv.out_proj.weight"));
    } else {
        lin(D("attn_q.weight"), f, S("self_attn.q_proj.weight"));
        lin(D("attn_k.weight"), f, S("self_attn.k_proj.weight"));
        lin(D("attn_v.weight"), f, S("self_attn.v_proj.weight"));
        lin(D("attn_out.weight"), f, S(lfm2_names ? "self_attn.out_proj.weight" : "self_attn.o_proj.weight"));
        vec(D("attn_q_norm.weight"), f, S(lfm2_names ? "self_attn.q_layernorm.weight" : "self_attn.q_norm.weight"));
        vec(D("attn_k_norm.weight"), f, S(lfm2_names ? "self_attn.k_layernorm.weight" : "self_attn.k_norm.weight"));
        if (find(f, S("self_attn.gate_proj.weight"))) lin(D("attn_gate.weight"), f, S("self_attn.gate_proj.weight"));
    }
#undef D
#undef S
}

/* transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney"), in double. */
static double hz_to_mel(double f) {
    double m = 3.0 * f / 200.0;
    return f >= 1000.0 ? 15.0 + log(f / 1000.0) * (27.0 / log(6.4)) : m;
}
static double mel_to_hz(double m) {
    double f = 200.0 * m / 3.0;
    return m >= 15.0 ? 1000.0 * exp((log(6.4) / 27.0) * (m - 15.0)) : f;
}
/* out holds n_mel * n_freq values; the bank is sized by the geometry it is given. */
static int mel_bank(double *out, int n_mel, int n_freq, double sr) {
    double *fft = malloc((size_t)n_freq * sizeof(double)), *filt = malloc((size_t)(n_mel + 2) * sizeof(double));
    if (!fft || !filt) { free(fft); free(filt); return -1; }
    for (int i = 0; i < n_freq; i++) fft[i] = (double)(int)(sr / 2) * i / (n_freq - 1);
    double lo = hz_to_mel(0.0), hi = hz_to_mel(sr / 2);
    for (int i = 0; i < n_mel + 2; i++) filt[i] = mel_to_hz(lo + (hi - lo) * i / (n_mel + 1));
    for (int f = 0; f < n_freq; f++)
        for (int m = 0; m < n_mel; m++) {
            double down = -(filt[m] - fft[f]) / (filt[m + 1] - filt[m]);
            double up = (filt[m + 2] - fft[f]) / (filt[m + 2] - filt[m + 1]);
            double v = down < up ? down : up;
            if (v < 0) v = 0;
            out[(size_t)m * n_freq + f] = v * (2.0 / (filt[m + 2] - filt[m]));
        }
    free(fft); free(filt);
    return 0;
}

/* --q8_0: the policy of the official Q8_0 files: every F16 matrix (the stacked depth
 * embeddings included) whose rows are whole 32-value blocks, quantized here from the BF16
 * values. Going through F16 first would round the BF16 values below F16's normal range and
 * move some blocks (measured on blk.0.shortconv.in_proj and v.blk.13.conv2d_in_proj). */
static int g_q8;
static int q8_0(const out_t *o) { return g_q8 && o->f16 && o->nd >= 2 && o->shape[0] % 32 == 0; }

static int emit(gguf_writer *w, const out_t *o) {
    if (gguf_write_tensor_begin(w, o->dst)) return -1;
    float buf[4096];
    uint64_t total = 1;
    for (int i = 0; i < o->nd; i++) total *= o->shape[i];
    uint64_t done = 0;
    src_t *t = (o->kind == COPY || o->kind == TEMPORAL0 || o->kind == TEMPORAL1) ? find(o->f, o->src) : NULL;
    while (done < total) {
        uint64_t take = total - done < 4096 ? total - done : 4096;
        for (uint64_t k = 0; k < take; k++) {
            uint64_t i = done + k;
            float v = 0;
            switch (o->kind) {
            case COPY: v = val(t, i); break;
            case TEMPORAL0: case TEMPORAL1: {
                /* patchifier [out][c][t][py][px] -> one temporal slice [out][c][py][px] */
                uint64_t px = i % 16, py = (i / 16) % 16, c = (i / 256) % 3, out = i / 768;
                uint64_t tt = o->kind == TEMPORAL1;
                v = val(t, out * 1536 + ((c * 2 + tt) * 16 + py) * 16 + px);
                break;
            }
            case STACK_EMBD: case STACK_NORM: {
                /* one codebook's slice: its embedding table, or its norm of depth_width values */
                uint64_t per = (uint64_t)X.depth_width * (o->kind == STACK_EMBD ? (uint64_t)X.cb_size : 1);
                char nm[200];
                snprintf(nm, sizeof nm, o->kind == STACK_EMBD ? "%s.%llu.embedding.weight" : "%s.%llu.embedding_norm.weight",
                         o->src, (unsigned long long)(i / per));
                src_t *s = find(o->f, nm);
                v = val(s, i % per);
                break;
            }
            case ROPE_TABLE: v = (int)i < X.rope_rot / 2 ? 1.0f : 1e30f; break;
            case FSQ_TABLE: {
                uint64_t code = i / (uint64_t)X.nlv, d = i % (uint64_t)X.nlv, basis = 1;
                for (uint64_t j = 0; j < d; j++) basis *= (uint64_t)X.lv[j];
                int h = X.lv[d] / 2;
                v = (float)((int)((code / basis) % (uint64_t)X.lv[d]) - h) / (float)h;
                break;
            }
            case MEL_BANK: v = (float)X.mel[i]; break;
            }
            buf[k] = v;
        }
        if (q8_0(o)) {
            /* whole 32-value blocks: 4096 and every row length here are multiples of 32 */
            uint8_t q[4096 / 32 * 34];
            if (take % 32 || nt_quantize_row(buf, q, (int)take, GGUF_TYPE_Q8_0) ||
                gguf_write_tensor_chunk(w, q, take / 32 * 34)) return -1;
        } else if (gguf_write_tensor_chunk_f32(w, buf, take)) return -1;
        done += take;
    }
    return gguf_write_tensor_end(w);
}

static int u32_of(const char *obj, const char *key, uint32_t *out) {
    long long v;
    if (nj_int(nj_get(obj, key), &v) || v < 0 || v > UINT32_MAX) {
        fprintf(stderr, "lfm3_omni_to_gguf: config key %s missing or not a count\n", key);
        return -1;
    }
    *out = (uint32_t)v;
    return 0;
}
static int f64_of(const char *obj, const char *key, double *out) {
    if (nj_number(nj_get(obj, key), out) || !isfinite(*out)) {
        fprintf(stderr, "lfm3_omni_to_gguf: config key %s missing or not a number\n", key);
        return -1;
    }
    return 0;
}

static int special_id(const char *tok_json, const char *content) {
    int bad = 0;
    for (const char *v = nj_first(nj_get(tok_json, "added_tokens"), &bad); v; v = nj_next(v, ']', &bad))
        if (nj_is(nj_get(v, "content"), content)) {
            long long id;
            return nj_int(nj_get(v, "id"), &id) ? -1 : (int)id;
        }
    return -1;
}

/* tokenizer.json -> the arrays llama.cpp's lfm3 GGUFs carry: tokens padded to the model's
 * vocabulary with [PADn], type 1 normal, 3 special added, 4 other added, 5 padding. */
static int write_tokenizer(gguf_writer *w, const char *j, uint32_t vocab, char ***tok_out, int32_t **typ_out, char ***mer_out, long long *nm_out) {
    const char *model = nj_get(j, "model"), *pre = nj_get(j, "pre_tokenizer");
    const char *regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
    int bad = 0;
    const char *p1 = nj_first(nj_get(pre, "pretokenizers"), &bad), *p2 = p1 ? nj_next(p1, ']', &bad) : NULL;
    const char *rx = nj_get(nj_get(p1, "pattern"), "Regex"), *im = nj_get(model, "ignore_merges"), *norm = nj_get(j, "normalizer");
    if (bad || !p2 || !nj_is(nj_get(model, "type"), "BPE") || !im || strncmp(im, "true", 4) ||
        !nj_is(nj_get(p1, "type"), "Split") || !nj_is(nj_get(p2, "type"), "ByteLevel") || !rx || strncmp(rx + 1, regex, strlen(regex)) ||
        rx[1 + strlen(regex)] != '"' || !norm || strncmp(norm, "null", 4))
        return fail("tokenizer.json is not the LFM2 byte-level BPE this converter knows", NULL);
    char **tok = calloc(vocab, sizeof *tok);
    int32_t *typ = calloc(vocab, sizeof *typ);
    if (!tok || !typ) return fail("out of memory", NULL);
    *tok_out = tok; *typ_out = typ;
    for (const char *k = nj_first(nj_get(model, "vocab"), &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad)) {
        long long id;
        if (nj_int(v, &id) || id < 0 || id >= vocab || tok[id]) return fail("vocab id out of range or repeated", NULL);
        if (!(tok[id] = nj_string(k))) return fail("vocab entry unreadable", NULL);
        typ[id] = 1;
    }
    for (const char *v = nj_first(nj_get(j, "added_tokens"), &bad); v; v = nj_next(v, ']', &bad)) {
        long long id;
        char *word = nj_string(nj_get(v, "content"));
        if (!word || nj_int(nj_get(v, "id"), &id) || id < 0 || id >= vocab) { free(word); return fail("added token unreadable", NULL); }
        const char *sp = nj_get(v, "special");
        if (!sp) { free(word); return fail("added token without a special flag", NULL); }
        free(tok[id]);
        tok[id] = word;
        /* A token spelled <|...|> is control even when tokenizer.json calls it not special
         * (<|tool_call_start|>, <|tool_call_end|>), as in the official GGUF. */
        size_t wl = strlen(word);
        int looks = wl >= 4 && !strncmp(word, "<|", 2) && !strcmp(word + wl - 2, "|>");
        typ[id] = (!strncmp(sp, "true", 4) || looks) ? 3 : 4;
    }
    char pad[32];
    for (uint32_t i = 0; i < vocab; i++)
        if (!tok[i]) { snprintf(pad, sizeof pad, "[PAD%u]", i); tok[i] = strdup(pad); typ[i] = 5; }
    long long nm = 0;
    const char *list = nj_get(model, "merges");
    for (const char *v = nj_first(list, &bad); v; v = nj_next(v, ']', &bad)) nm++;
    char **mer = calloc((size_t)nm, sizeof *mer);
    if (!mer || bad) return fail("merges unreadable", NULL);
    *mer_out = mer; *nm_out = nm;
    long long i = 0;
    for (const char *v = nj_first(list, &bad); v; v = nj_next(v, ']', &bad), i++) {
        int b = 0;
        const char *a = nj_first(v, &b), *z = a ? nj_next(a, ']', &b) : NULL;
        char *x = nj_string(a), *y = nj_string(z);
        if (b || !x || !y || !(mer[i] = malloc(strlen(x) + strlen(y) + 2))) { free(x); free(y); return fail("merge pair unreadable", NULL); }
        sprintf(mer[i], "%s %s", x, y);
        free(x); free(y);
    }
    gguf_write_kv_str(w, "tokenizer.ggml.model", "gpt2");
    gguf_write_kv_str(w, "tokenizer.ggml.pre", "lfm2");
    gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", (const char *const *)tok, vocab);
    gguf_write_kv_i32_array(w, "tokenizer.ggml.token_type", typ, vocab);
    gguf_write_kv_str_array(w, "tokenizer.ggml.merges", (const char *const *)mer, (uint64_t)nm);
    return 0;
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

static st_file M, C;

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--q8_0")) { g_q8 = 1; argv++; argc--; }
    if (argc != 3) { fprintf(stderr, "usage: %s [--q8_0] SNAPSHOT OUT.gguf\n", argv[0]); return 2; }
    const char *dir = argv[1], *out = argv[2];
    char path[4096], partial[4200];
    struct stat st;
    if (!stat(out, &st)) return fail("output exists", out);
#define LOAD(var, name) (snprintf(path, sizeof path, "%s/%s", dir, name), (var = nj_load(path, NULL)))
    char *cfg, *tok, *proc, *gen, *ccfg;
    if (!LOAD(cfg, "config.json") || !LOAD(tok, "tokenizer.json") || !LOAD(proc, "processor_config.json") ||
        !LOAD(gen, "generation_config.json") || !LOAD(ccfg, "audio_tokenizer/config.json"))
        return fail("snapshot incomplete", dir);
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    if (st_open(&M, path)) return 1;
    snprintf(path, sizeof path, "%s/audio_tokenizer/model.safetensors", dir);
    if (st_open(&C, path)) return 1;

    const char *tc = nj_get(cfg, "text_config"), *vc = nj_get(cfg, "vision_config"), *ac = nj_get(cfg, "audio_encoder_config");
    const char *alc = nj_get(ac, "text_config"), *dc = nj_get(cfg, "depth_config");
    const char *fe = nj_get(proc, "feature_extractor"), *ip = nj_get(proc, "image_processor");
    if (!nj_is(nj_get(cfg, "model_type"), "lfm3_omni") || !tc || !vc || !ac || !alc || !dc || !fe || !ip ||
        !nj_is(nj_get(tc, "model_type"), "lfm3") || !nj_is(nj_get(alc, "model_type"), "lfm2") ||
        !nj_is(nj_get(ccfg, "model_type"), "liquid_codec"))
        return fail("not an LFM3-Omni snapshot", dir);

    uint32_t L, E, FF, H, KVH, HD, CTX, SW, LC, V;
    double eps, theta_full, theta_swa, rot;
    if (u32_of(tc, "num_hidden_layers", &L) || u32_of(tc, "hidden_size", &E) || u32_of(tc, "intermediate_size", &FF) ||
        u32_of(tc, "num_attention_heads", &H) || u32_of(tc, "num_key_value_heads", &KVH) || u32_of(tc, "head_dim", &HD) ||
        u32_of(tc, "max_position_embeddings", &CTX) || u32_of(tc, "sliding_window", &SW) || u32_of(tc, "conv_L_cache", &LC) ||
        u32_of(tc, "vocab_size", &V) || f64_of(tc, "norm_eps", &eps) ||
        f64_of(nj_get(nj_get(tc, "rope_parameters"), "full_attention"), "rope_theta", &theta_full) ||
        f64_of(nj_get(nj_get(tc, "rope_parameters"), "full_attention"), "partial_rotary_factor", &rot) ||
        f64_of(nj_get(nj_get(tc, "rope_parameters"), "sliding_attention"), "rope_theta", &theta_swa))
        return fail("text_config incomplete", NULL);
    if (L > 256 || HD == 0 || HD > 4096) return fail("text_config out of range", NULL);
    X.rope_rot = (int)(rot * HD) / 2 * 2;

    /* layer types: conv / sliding_attention / full_attention, one per layer */
    int is_conv[256] = {0};
    uint8_t is_swa[256] = {0};
    int32_t kvh[256];
    {
        int bad = 0, l = 0;
        for (const char *v = nj_first(nj_get(tc, "layer_types"), &bad); v; v = nj_next(v, ']', &bad), l++) {
            if (l >= (int)L) return fail("more layer_types than layers", NULL);
            is_conv[l] = nj_is(v, "conv");
            is_swa[l] = nj_is(v, "sliding_attention");
            if (!is_conv[l] && !is_swa[l] && !nj_is(v, "full_attention")) return fail("unknown layer type", NULL);
            kvh[l] = is_conv[l] ? 0 : (int32_t)KVH;
        }
        if (bad || l != (int)L) return fail("layer_types does not match num_hidden_layers", NULL);
    }

    char nm[256], sp[256];
    /* text decoder, under arch_lfm3's names */
    lin("token_embd.weight", &M, "model.language_model.embed_tokens.weight");
    vec("output_norm.weight", &M, "model.language_model.embedding_norm.weight");
    add("rope_freqs.weight", NULL, NULL, ROPE_TABLE, 0, 1, S1(HD / 2));
    for (uint32_t l = 0; l < L; l++) {
#define T(d, s) (snprintf(nm, sizeof nm, "blk.%u.%s", l, d), snprintf(sp, sizeof sp, "model.language_model.layers.%u.%s", l, s))
        T("attn_norm.weight", "operator_norm.weight"); vec(nm, &M, sp);
        T("ffn_norm.weight", "ffn_norm.weight"); vec(nm, &M, sp);
        T("post_attention_norm.weight", "post_operator_norm.weight"); vec(nm, &M, sp);
        T("post_ffw_norm.weight", "post_ffn_norm.weight"); vec(nm, &M, sp);
        T("ffn_gate.weight", "feed_forward.w1.weight"); lin(nm, &M, sp);
        T("ffn_up.weight", "feed_forward.w3.weight"); lin(nm, &M, sp);
        T("ffn_down.weight", "feed_forward.w2.weight"); lin(nm, &M, sp);
        if (is_conv[l]) {
            T("shortconv.conv.weight", "conv.conv.weight"); add(nm, &M, sp, COPY, 0, 2, S2(LC, E));
            T("shortconv.in_proj.weight", "conv.in_proj.weight"); lin(nm, &M, sp);
            T("shortconv.out_proj.weight", "conv.out_proj.weight"); lin(nm, &M, sp);
        } else {
            T("attn_q.weight", "self_attn.q_proj.weight"); lin(nm, &M, sp);
            T("attn_k.weight", "self_attn.k_proj.weight"); lin(nm, &M, sp);
            T("attn_v.weight", "self_attn.v_proj.weight"); lin(nm, &M, sp);
            T("attn_output.weight", "self_attn.o_proj.weight"); lin(nm, &M, sp);
            T("attn_gate.weight", "self_attn.gate_proj.weight"); lin(nm, &M, sp);
            T("attn_q_norm.weight", "self_attn.q_norm.weight"); vec(nm, &M, sp);
            T("attn_k_norm.weight", "self_attn.k_norm.weight"); vec(nm, &M, sp);
        }
#undef T
    }

    /* vision tower and projector */
    uint32_t VL, VE, VH, VKVH, VP, VM, VT;
    double veps, vtheta, peps;
    if (u32_of(vc, "num_hidden_layers", &VL) || u32_of(vc, "hidden_size", &VE) || u32_of(vc, "num_attention_heads", &VH) ||
        u32_of(vc, "num_key_value_heads", &VKVH) || u32_of(vc, "patch_size", &VP) || u32_of(vc, "spatial_merge_size", &VM) ||
        u32_of(vc, "temporal_patch_size", &VT) || f64_of(vc, "norm_eps", &veps) ||
        f64_of(nj_get(vc, "rope_parameters"), "rope_theta", &vtheta) || f64_of(cfg, "projector_layer_norm_eps", &peps))
        return fail("vision_config incomplete", NULL);
    if (VP != 16 || VT != 2 || VL > 256) return fail("this converter splits a 3x2x16x16 patchifier", NULL);
    add("v.patch_embd.weight", &M, "model.vision_tower.patchifier.weight", TEMPORAL0, 1, 4, S4(16, 16, 3, VE));
    add("v.patch_embd.weight.1", &M, "model.vision_tower.patchifier.weight", TEMPORAL1, 1, 4, S4(16, 16, 3, VE));
    find(&M, "model.vision_tower.patchifier.weight")->used = 1;
    vec("v.embedding_norm.weight", &M, "model.vision_tower.embedding_norm.weight");
    for (uint32_t l = 0; l < VL; l++) {
        snprintf(sp, sizeof sp, "model.vision_tower.layers.%u", l);
        snprintf(nm, sizeof nm, "%s.conv.in_proj.weight", sp);
        int conv = find(&M, nm) != NULL;
        char dp[64]; snprintf(dp, sizeof dp, "v.blk.%u", l);
        char d[160], s[300];
#define VD(x) (snprintf(d, sizeof d, "%s.%s", dp, x), d)
#define VS(x) (snprintf(s, sizeof s, "%s.%s", sp, x), s)
        vec(VD("ln1.weight"), &M, VS("operator_norm.weight"));
        vec(VD("ln2.weight"), &M, VS("ffn_norm.weight"));
        lin(VD("ffn_gate.weight"), &M, VS("feed_forward.w1.weight"));
        lin(VD("ffn_up.weight"), &M, VS("feed_forward.w3.weight"));
        lin(VD("ffn_down.weight"), &M, VS("feed_forward.w2.weight"));
        if (conv) {
            lin(VD("conv2d_in_proj.weight"), &M, VS("conv.in_proj.weight"));
            lin(VD("conv2d_out_proj.weight"), &M, VS("conv.out_proj.weight"));
            add(VD("conv2d_spatial.weight"), &M, VS("conv.spatial_conv.weight"), COPY, 0, 4, S4(3, 3, 1, VE));
        } else {
            lin(VD("attn_q.weight"), &M, VS("self_attn.q_proj.weight"));
            lin(VD("attn_k.weight"), &M, VS("self_attn.k_proj.weight"));
            lin(VD("attn_v.weight"), &M, VS("self_attn.v_proj.weight"));
            lin(VD("attn_out.weight"), &M, VS("self_attn.out_proj.weight"));
            vec(VD("attn_q_norm.weight"), &M, VS("self_attn.q_layernorm.weight"));
            vec(VD("attn_k_norm.weight"), &M, VS("self_attn.k_layernorm.weight"));
        }
#undef VD
#undef VS
    }
    vec("mm.input_norm.weight", &M, "model.multi_modal_projector.layer_norm.weight");
    vec("mm.input_norm.bias", &M, "model.multi_modal_projector.layer_norm.bias");
    lin("mm.0.weight", &M, "model.multi_modal_projector.linear_1.weight");
    vec("mm.0.bias", &M, "model.multi_modal_projector.linear_1.bias");
    lin("mm.2.weight", &M, "model.multi_modal_projector.linear_2.weight");
    vec("mm.2.bias", &M, "model.multi_modal_projector.linear_2.bias");

    /* audio downsampler, encoder and projector */
    uint32_t AL, AE, AH, AKVH, ACH, AMEL, AWIN, ALC;
    double aeps, atheta;
    if (u32_of(alc, "num_hidden_layers", &AL) || u32_of(alc, "hidden_size", &AE) || u32_of(alc, "num_attention_heads", &AH) ||
        u32_of(alc, "num_key_value_heads", &AKVH) || u32_of(ac, "conv_channels", &ACH) || u32_of(ac, "num_mel_bins", &AMEL) ||
        u32_of(ac, "sliding_window", &AWIN) || u32_of(alc, "conv_L_cache", &ALC) || f64_of(alc, "norm_eps", &aeps) ||
        f64_of(nj_get(alc, "rope_parameters"), "rope_theta", &atheta))
        return fail("audio_encoder_config incomplete", NULL);
    const char *ds = "model.audio_encoder.downsample";
    const struct { const char *d, *s; int pw; } dsl[] = {
        {"a.conv1d.0", "conv_in", 0}, {"a.conv1d.2", "conv_mid", 0}, {"a.conv1d.3", "lin_mid", 1},
        {"a.conv1d.5", "conv_end", 0}, {"a.conv1d.6", "lin_end", 1}};
    for (int i = 0; i < 5; i++) {
        snprintf(nm, sizeof nm, "%s.weight", dsl[i].d); snprintf(sp, sizeof sp, "%s.%s.weight", ds, dsl[i].s);
        if (dsl[i].pw) add(nm, &M, sp, COPY, 0, 4, S4(1, 1, ACH, ACH));
        else add(nm, &M, sp, COPY, 0, 4, S4(3, 3, 1, ACH));
        snprintf(nm, sizeof nm, "%s.bias", dsl[i].d); snprintf(sp, sizeof sp, "%s.%s.bias", ds, dsl[i].s);
        add(nm, &M, sp, COPY, 0, 3, S3(1, 1, ACH));
    }
    add("a.pre_encode.out.weight", &M, "model.audio_encoder.downsample.lin_out.weight", COPY, 1, 2, S2(AMEL * ACH / 8, AE));
    vec("a.pre_encode.out.bias", &M, "model.audio_encoder.downsample.lin_out.bias");
    for (uint32_t l = 0; l < AL; l++) {
        char dp[64]; snprintf(dp, sizeof dp, "a.blk.%u", l);
        snprintf(sp, sizeof sp, "model.audio_encoder.lfm.layers.%u", l);
        snprintf(nm, sizeof nm, "%s.conv.in_proj.weight", sp);
        lfm_block(dp, &M, sp, (int)AE, find(&M, nm) != NULL, 1);
    }
    vec("a.post_ln.weight", &M, "model.audio_encoder.lfm.embedding_norm.weight");
    vec("mm.a.mlp.0.weight", &M, "model.audio_projector.layer_norm.weight");
    vec("mm.a.mlp.0.bias", &M, "model.audio_projector.layer_norm.bias");
    lin("mm.a.mlp.1.weight", &M, "model.audio_projector.linear_1.weight");
    vec("mm.a.mlp.1.bias", &M, "model.audio_projector.linear_1.bias");
    lin("mm.a.mlp.3.weight", &M, "model.audio_projector.linear_2.weight");
    vec("mm.a.mlp.3.bias", &M, "model.audio_projector.linear_2.bias");

    /* depth decoder and the codebook embedding that feeds speech frames back */
    uint32_t DL, DE, DH, DKVH, DHD, NCB, CBS, EOSC, NTXT, NAUD;
    double deps, dtheta;
    if (u32_of(dc, "num_hidden_layers", &DL) || u32_of(dc, "hidden_size", &DE) || u32_of(dc, "num_attention_heads", &DH) ||
        u32_of(dc, "num_key_value_heads", &DKVH) || u32_of(dc, "head_dim", &DHD) || u32_of(cfg, "num_codebooks", &NCB) ||
        u32_of(cfg, "codebook_size", &CBS) || u32_of(cfg, "audio_eos_code", &EOSC) || u32_of(cfg, "interleaved_n_text", &NTXT) ||
        u32_of(cfg, "interleaved_n_audio", &NAUD) || f64_of(dc, "norm_eps", &deps) ||
        f64_of(nj_get(nj_get(dc, "rope_parameters"), "full_attention"), "rope_theta", &dtheta))
        return fail("depth_config incomplete", NULL);
    if (NCB > 64) return fail("num_codebooks out of range", NULL);
    X.n_cb = (int)NCB; X.cb_size = (int)CBS; X.depth_width = (int)DE;
    lin("a.gen.code.proj_in.weight", &M, "model.depth_decoder.depth_linear.weight");
    vec("a.gen.code.proj_in.bias", &M, "model.depth_decoder.depth_linear.bias");
    vec("a.gen.code.mask_embd", &M, "model.depth_decoder.mask_embedding");
    add("a.gen.code.embd.weight", &M, "model.depth_decoder.depth_embeddings", STACK_EMBD, 1, 3, S3(DE, CBS, NCB));
    /* one flat vector of the twelve norms: gguf_quantize leaves 1-D tensors alone, and a
     * norm table quantized to Q8_0 is precision thrown away (the official files keep it F32) */
    add("a.gen.code.head_norm.weight", &M, "model.depth_decoder.depth_embeddings", STACK_NORM, 0, 1, S1(DE * NCB));
    for (uint32_t k = 0; k < NCB; k++) {
        snprintf(sp, sizeof sp, "model.depth_decoder.depth_embeddings.%u.embedding.weight", k);
        src_t *t = find(&M, sp);
        if (!t || t->n != (uint64_t)CBS * DE) return fail("depth embedding missing or misshapen", sp);
        t->used = 1;
        snprintf(sp, sizeof sp, "model.depth_decoder.depth_embeddings.%u.embedding_norm.weight", k);
        if (!(t = find(&M, sp)) || t->n != DE) return fail("depth embedding norm missing", sp);
        t->used = 1;
    }
    vec("a.gen.code.output_norm.weight", &M, "model.depth_decoder.model.embedding_norm.weight");
    for (uint32_t l = 0; l < DL; l++) {
        char dp[64]; snprintf(dp, sizeof dp, "a.gen.code.blk.%u", l);
        snprintf(sp, sizeof sp, "model.depth_decoder.model.layers.%u", l);
        lfm_block(dp, &M, sp, (int)DE, 0, 0);
    }
    lin("a.gen.code.out_embd.weight", &M, "model.codebook_embedding.weight");

    /* codec decoder: FSQ table, projections, the LFM3 stack, the head and the window */
    uint32_t CE, CWIN, CFS, CNFFT, CHOP, CWINL, CSR, CQD;
    if (u32_of(ccfg, "encoder_dim", &CE) || u32_of(ccfg, "decoder_window", &CWIN) || u32_of(ccfg, "stft_frame_stack", &CFS) ||
        u32_of(ccfg, "stft_n_fft", &CNFFT) || u32_of(ccfg, "stft_hop_length", &CHOP) || u32_of(ccfg, "stft_win_length", &CWINL) ||
        u32_of(ccfg, "sampling_rate", &CSR) || u32_of(ccfg, "quantizer_dim", &CQD))
        return fail("audio_tokenizer/config.json incomplete", NULL);
    {
        int bad = 0;
        uint64_t prod = 1;
        for (const char *v = nj_first(nj_get(ccfg, "fsq_levels"), &bad); v; v = nj_next(v, ']', &bad)) {
            long long q;
            if (X.nlv == 8 || nj_int(v, &q) || q < 2 || q > 64) return fail("fsq_levels unreadable", NULL);
            X.lv[X.nlv++] = (int)q; prod *= (uint64_t)q;
        }
        if (bad || !X.nlv || prod + 1 != CBS || prod != (uint64_t)EOSC) return fail("fsq_levels do not give codebook_size - 1", NULL);
    }
    add("a.gen.wav.quant.first.codebook.weight", NULL, NULL, FSQ_TABLE, 0, 2, S2(X.nlv, CBS - 1));
    add("a.gen.wav.quant.first.out_proj.weight", &C, "quantizer.project_out.weight", COPY, 0, 0, NULL);
    vec("a.gen.wav.quant.first.out_proj.bias", &C, "quantizer.project_out.bias");
    lin("a.gen.wav.tfm.in_proj.weight", &C, "from_quantizer.weight");
    vec("a.gen.wav.tfm.in_proj.bias", &C, "from_quantizer.bias");
    int CL = 0;
    for (;; CL++) {
        snprintf(sp, sizeof sp, "decoder.lfm.layers.%d.operator_norm.weight", CL);
        if (!find(&C, sp)) break;
        char dp[64]; snprintf(dp, sizeof dp, "a.gen.wav.tfm.blk.%d", CL);
        snprintf(sp, sizeof sp, "decoder.lfm.layers.%d", CL);
        snprintf(nm, sizeof nm, "%s.conv.in_proj.weight", sp);
        lfm_block(dp, &C, sp, (int)CE, find(&C, nm) != NULL, 0);
    }
    vec("a.gen.wav.tfm.output_norm.weight", &C, "decoder.lfm.embedding_norm.weight");
    lin("a.gen.wav.tfm.out_proj.weight", &C, "decoder.lin.weight");
    vec("a.gen.wav.tfm.out_proj.bias", &C, "decoder.lin.bias");
    vec("a.gen.wav.istft.window", &C, "decoder.istft.window");

    /* the log-mel front end's filter bank */
    uint32_t FSR, FNFFT, FHOP, FWIN, FSTRIDE;
    double ffloor, foff, fdiv;
    if (u32_of(fe, "sampling_rate", &FSR) || u32_of(fe, "n_fft", &FNFFT) || u32_of(fe, "hop_length", &FHOP) ||
        u32_of(fe, "win_length", &FWIN) || u32_of(fe, "encoder_stride", &FSTRIDE) || f64_of(fe, "log_floor", &ffloor) ||
        f64_of(fe, "normalization_offset", &foff) || f64_of(fe, "normalization_divisor", &fdiv))
        return fail("feature_extractor config incomplete", NULL);
    /* The bank is allocated for the geometry the files declare, within bounds that keep the
     * product, and every index into it, far inside int. */
    if (AMEL < 8 || AMEL > 1024 || AMEL % 8 || FNFFT < 4 || FNFFT > 65536 || FNFFT % 4 || FWIN != FNFFT || FSR < 1 ||
        FHOP < 1 || FHOP > FNFFT)
        return fail("mel geometry out of range", NULL);
    if (!(X.mel = malloc((size_t)AMEL * (FNFFT / 2 + 1) * sizeof(double))) ||
        mel_bank(X.mel, (int)AMEL, (int)(FNFFT / 2 + 1), (double)FSR))
        return fail("out of memory", "mel bank");
    add("a.mel_filters", NULL, NULL, MEL_BANK, 0, 2, S2(FNFFT / 2 + 1, AMEL));

    if (bad_plan) return fail("the plan does not match the checkpoint", NULL);
    int stray = 0;
    for (st_file *f = &M; f; f = f == &M ? &C : NULL)
        for (int i = 0; i < f->n; i++) {
            if (f->t[i].used) continue;
            const char *n = f->t[i].name;
            if (f == &C && (!strncmp(n, "encoder.", 8) || !strncmp(n, "frontend.", 9) || !strncmp(n, "to_quantizer.", 13) ||
                            !strncmp(n, "quantizer.project_in.", 21))) {
                fprintf(stderr, "skip %s: codec encoder, 24 kHz speech into codes for a replayed prompt\n", n);
                continue;
            }
            fprintf(stderr, "lfm3_omni_to_gguf: source tensor %s has no place in the plan\n", n);
            stray = 1;
        }
    if (stray) return fail("unplanned source tensors", NULL);

    /* image processing, prompt and generation constants the harness reads */
    long long minpx = 0, maxpx = 0;
    if (nj_int(nj_get(nj_get(ip, "size"), "shortest_edge"), &minpx) || nj_int(nj_get(nj_get(ip, "size"), "longest_edge"), &maxpx))
        return fail("image_processor size unreadable", NULL);
    double atemp, arep;
    uint32_t awin_rep;
    if (f64_of(gen, "audio_temperature", &atemp) || f64_of(gen, "audio_repetition_penalty", &arep) ||
        u32_of(gen, "audio_repetition_window", &awin_rep))
        return fail("generation_config incomplete", NULL);
    int32_t sched[16]; int nsched = 0;
    {
        int bad = 0;
        for (const char *v = nj_first(nj_get(cfg, "maskgit_schedule"), &bad); v; v = nj_next(v, ']', &bad)) {
            long long q;
            if (nsched == 16 || nj_int(v, &q) || q < 1) return fail("maskgit_schedule unreadable", NULL);
            sched[nsched++] = (int32_t)q;
        }
        int sum = 0;
        for (int i = 0; i < nsched; i++) sum += sched[i];
        if (bad || sum != (int)NCB) return fail("maskgit_schedule does not cover the codebooks", NULL);
    }

    if (snprintf(partial, sizeof partial, "%s.partial", out) >= (int)sizeof partial) return fail("path too long", out);
    int fd = open(partial, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { perror(partial); return 1; }
    close(fd);
    gguf_writer *w = gguf_write_open(partial);
    if (!w) { unlink(partial); return fail("cannot write", partial); }

    gguf_write_kv_str(w, "general.architecture", "lfm3");
    gguf_write_kv_str(w, "general.name", "LFM3-Omni-2B, every tower, notorch layout");
    gguf_write_kv_u32(w, "lfm3.block_count", L);
    gguf_write_kv_u32(w, "lfm3.context_length", CTX);
    gguf_write_kv_u32(w, "lfm3.embedding_length", E);
    gguf_write_kv_u32(w, "lfm3.feed_forward_length", FF);
    gguf_write_kv_u32(w, "lfm3.attention.head_count", H);
    gguf_write_kv_i32_array(w, "lfm3.attention.head_count_kv", kvh, L);
    gguf_write_kv_f32(w, "lfm3.rope.freq_base", (float)theta_full);
    gguf_write_kv_f32(w, "lfm3.rope.freq_base_swa", (float)theta_swa);
    gguf_write_kv_f32(w, "lfm3.attention.layer_norm_rms_epsilon", (float)eps);
    gguf_write_kv_u32(w, "lfm3.attention.key_length", HD);
    gguf_write_kv_u32(w, "lfm3.attention.value_length", HD);
    gguf_write_kv_u32(w, "lfm3.rope.dimension_count", HD);
    gguf_write_kv_bool_array(w, "lfm3.attention.sliding_window_pattern", is_swa, L);
    gguf_write_kv_u32(w, "lfm3.attention.sliding_window", SW);
    gguf_write_kv_u32(w, "lfm3.shortconv.l_cache", LC);

    char **tokens = NULL, **merges = NULL; int32_t *types = NULL; long long nmerge = 0;
    long long bos, eos, pad;
    if (nj_int(nj_get(tc, "bos_token_id"), &bos) || nj_int(nj_get(tc, "eos_token_id"), &eos) || nj_int(nj_get(tc, "pad_token_id"), &pad) ||
        write_tokenizer(w, tok, V, &tokens, &types, &merges, &nmerge)) { gguf_write_abort(w); return fail("tokenizer", NULL); }
    gguf_write_kv_u32(w, "tokenizer.ggml.bos_token_id", (uint32_t)bos);
    gguf_write_kv_u32(w, "tokenizer.ggml.eos_token_id", (uint32_t)eos);
    gguf_write_kv_u32(w, "tokenizer.ggml.padding_token_id", (uint32_t)pad);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_bos_token", 0);
    gguf_write_kv_bool(w, "tokenizer.ggml.add_eos_token", 0);

    /* Tower keys avoid the suffixes gguf.c scans for the decoder (block_count,
     * embedding_length, rope.freq_base, rms_epsilon, ...), which a later key would win. */
    const struct { const char *k; const char *tok; } ids[] = {
        {"lfm3_omni.token.image", "<|image|>"}, {"lfm3_omni.token.video", "<|video|>"},
        {"lfm3_omni.token.audio", "<|audio|>"}, {"lfm3_omni.token.audio_out", "<|audio_out|>"},
        {"lfm3_omni.token.audio_start", "<|audio_start|>"}, {"lfm3_omni.token.mixed_start", "<|mixed_start|>"},
        {"lfm3_omni.token.text_end", "<|text_end|>"}, {"lfm3_omni.token.media_start", "<|vision_start|>"},
        {"lfm3_omni.token.media_end", "<|vision_end|>"}, {"lfm3_omni.token.im_start", "<|im_start|>"},
        {"lfm3_omni.token.im_end", "<|im_end|>"}, {"lfm3_omni.token.think", "<think>"},
        {"lfm3_omni.token.think_end", "</think>"}};
    for (size_t i = 0; i < sizeof ids / sizeof ids[0]; i++) {
        int id = special_id(tok, ids[i].tok);
        if (id < 0) { gguf_write_abort(w); return fail("special token not in tokenizer.json", ids[i].tok); }
        gguf_write_kv_u32(w, ids[i].k, (uint32_t)id);
    }
    long long cfg_ids[4];
    if (nj_int(nj_get(cfg, "image_token_id"), &cfg_ids[0]) || nj_int(nj_get(cfg, "audio_token_id"), &cfg_ids[1]) ||
        nj_int(nj_get(cfg, "audio_out_token_id"), &cfg_ids[2]) || nj_int(nj_get(cfg, "video_token_id"), &cfg_ids[3]) ||
        cfg_ids[0] != special_id(tok, "<|image|>") || cfg_ids[1] != special_id(tok, "<|audio|>") ||
        cfg_ids[2] != special_id(tok, "<|audio_out|>") || cfg_ids[3] != special_id(tok, "<|video|>")) {
        gguf_write_abort(w); return fail("config.json token ids disagree with tokenizer.json", NULL);
    }
    gguf_write_kv_u32(w, "lfm3_omni.vision.layers", VL);
    gguf_write_kv_u32(w, "lfm3_omni.vision.heads", VH);
    gguf_write_kv_u32(w, "lfm3_omni.vision.kv_heads", VKVH);
    gguf_write_kv_u32(w, "lfm3_omni.vision.patch", VP);
    gguf_write_kv_u32(w, "lfm3_omni.vision.merge", VM);
    gguf_write_kv_u32(w, "lfm3_omni.vision.temporal_patch", VT);
    gguf_write_kv_f32(w, "lfm3_omni.vision.theta", (float)vtheta);
    gguf_write_kv_f32(w, "lfm3_omni.vision.eps", (float)veps);
    gguf_write_kv_f32(w, "lfm3_omni.vision.projector_eps", (float)peps);
    gguf_write_kv_u32(w, "lfm3_omni.vision.min_pixels", (uint32_t)minpx);
    gguf_write_kv_u32(w, "lfm3_omni.vision.max_pixels", (uint32_t)maxpx);
    gguf_write_kv_u32(w, "lfm3_omni.audio.layers", AL);
    gguf_write_kv_u32(w, "lfm3_omni.audio.heads", AH);
    gguf_write_kv_u32(w, "lfm3_omni.audio.kv_heads", AKVH);
    gguf_write_kv_u32(w, "lfm3_omni.audio.window", AWIN);
    gguf_write_kv_f32(w, "lfm3_omni.audio.theta", (float)atheta);
    gguf_write_kv_f32(w, "lfm3_omni.audio.eps", (float)aeps);
    gguf_write_kv_u32(w, "lfm3_omni.audio.sample_rate", FSR);
    gguf_write_kv_u32(w, "lfm3_omni.audio.n_fft", FNFFT);
    gguf_write_kv_u32(w, "lfm3_omni.audio.hop", FHOP);
    gguf_write_kv_u32(w, "lfm3_omni.audio.stride", FSTRIDE);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_floor", (float)ffloor);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_offset", (float)foff);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_divisor", (float)fdiv);
    gguf_write_kv_u32(w, "lfm3_omni.depth.layers", DL);
    gguf_write_kv_u32(w, "lfm3_omni.depth.heads", DH);
    gguf_write_kv_u32(w, "lfm3_omni.depth.kv_heads", DKVH);
    gguf_write_kv_f32(w, "lfm3_omni.depth.theta", (float)dtheta);
    gguf_write_kv_f32(w, "lfm3_omni.depth.eps", (float)deps);
    gguf_write_kv_u32(w, "lfm3_omni.depth.codebooks", NCB);
    gguf_write_kv_u32(w, "lfm3_omni.depth.codebook_size", CBS);
    gguf_write_kv_u32(w, "lfm3_omni.depth.eos_code", EOSC);
    gguf_write_kv_i32_array(w, "lfm3_omni.depth.schedule", sched, (uint64_t)nsched);
    gguf_write_kv_u32(w, "lfm3_omni.gen.interleave_text", NTXT);
    gguf_write_kv_u32(w, "lfm3_omni.gen.interleave_audio", NAUD);
    gguf_write_kv_f32(w, "lfm3_omni.gen.audio_temperature", (float)atemp);
    gguf_write_kv_f32(w, "lfm3_omni.gen.audio_repetition_penalty", (float)arep);
    gguf_write_kv_u32(w, "lfm3_omni.gen.audio_repetition_window", awin_rep);
    gguf_write_kv_u32(w, "lfm3_omni.codec.layers", (uint32_t)CL);
    gguf_write_kv_u32(w, "lfm3_omni.codec.heads", 8);
    gguf_write_kv_u32(w, "lfm3_omni.codec.kv_heads", 2);
    gguf_write_kv_u32(w, "lfm3_omni.codec.window", CWIN);
    gguf_write_kv_f32(w, "lfm3_omni.codec.theta", 1000000.0f);
    gguf_write_kv_f32(w, "lfm3_omni.codec.eps", 1e-5f);
    gguf_write_kv_u32(w, "lfm3_omni.codec.frame_stack", CFS);
    gguf_write_kv_u32(w, "lfm3_omni.codec.n_fft", CNFFT);
    gguf_write_kv_u32(w, "lfm3_omni.codec.hop", CHOP);
    gguf_write_kv_u32(w, "lfm3_omni.codec.sample_rate", CSR);

    for (int i = 0; i < nplan; i++)
        gguf_write_tensor_decl(w, plan[i].dst, (uint32_t)plan[i].nd, plan[i].shape,
                               q8_0(&plan[i]) ? GGUF_TYPE_Q8_0 : plan[i].f16 ? GGUF_TYPE_F16 : GGUF_TYPE_F32);
    for (int i = 0; i < nplan; i++)
        if (emit(w, &plan[i])) { gguf_write_abort(w); return fail("write failed at", plan[i].dst); }
    if (gguf_write_close(w)) { unlink(partial); return fail("close failed", partial); }
    return publish(partial, out, nplan);
}
