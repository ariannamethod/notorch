/* EmbeddingGemma 2 text plumbing: tokenizer.json -> C converter -> tokenizer GGUF
 * -> bpe.c Gemma 4 path -> prefix, <bos>, <eos> -> CLI text mode.
 *
 * Without arguments: a synthetic vocabulary with two merges, an escaped token and
 * a surrogate-pair token, converted by the real binary and then encoded by the CLI
 * over an analytic model whose vocabulary covers it. Expected IDs are worked by
 * hand in the table below.
 *
 * With TOKENIZER.gguf ENCODED.jsonl GOLDEN.json FIXTURE_TOKENS.tsv: the frozen
 * gate. Each formatted text of the reference record is split back into its
 * condition and its text, tokenized here, and compared with the official IDs. */
#include "gguf.h"
#include "tools/embeddinggemma2_text.h"
#include "tools/narrow_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(c, what) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s (line %d)\n", what, __LINE__); } } while (0)

static int same(const int *a, int n, const int *b, int m) { return n == m && n > 0 && !memcmp(a, b, (size_t)n * sizeof(int)); }

/* ── frozen gate ── */
static int identical, total, per[EG2_CONDITIONS];
static void compare(const eg2_tokenizer *t, const char *id, const char *text, const int *want, int n) {
    static int got[8192];
    int c = EG2_CONDITIONS - 1;
    for (int k = 0; k < EG2_CONDITIONS - 1; k++)
        if (!strncmp(text, eg2_conditions[k][1], strlen(eg2_conditions[k][1]))) c = k;
    const char *prefix = eg2_conditions[c][1];
    int m = eg2_text_ids(t, prefix, text + strlen(prefix), got, 8192, NULL), at = 0;
    total++;
    per[c]++;
    if (same(got, m, want, n)) { identical++; return; }
    while (at < m && at < n && got[at] == want[at]) at++;
    fprintf(stderr, "MISMATCH %s: %d IDs, expected %d, first difference at %d\n", id, m, n, at);
}
static int gate(char **argv) {
    static int want[8192];
    eg2_tokenizer t;
    int loaded = !eg2_tokenizer_load(&t, argv[1]);
    char *golden = nj_load(argv[3], NULL), *fixtures = nj_load(argv[4], NULL), *line = NULL;
    FILE *f = fopen(argv[2], "r");
    size_t cap = 0;
    int bad = 0;
    if (!loaded || !golden || !fixtures || !f) {
        fprintf(stderr, "gate inputs unreadable\n");
        return 1;
    }
    while (getline(&line, &cap, f) >= 0) {
        char *id = nj_string(nj_get(line, "id")), *text = nj_string(nj_get(line, "formatted_text"));
        int n = 0;
        long long v;
        for (const char *x = nj_first(nj_get(line, "input_ids"), &bad); x && n < 8192; x = nj_next(x, ']', &bad)) {
            if (nj_int(x, &v)) { bad = 1; break; }
            want[n++] = (int)v;
        }
        if (!id || !text || bad) { fprintf(stderr, "%s: malformed record\n", argv[2]); return 1; }
        compare(&t, id, text, want, n);
        free(id); free(text);
    }
    for (const char *g = nj_first(golden, &bad); g; g = nj_next(g, ']', &bad)) {
        char *id = nj_string(nj_get(g, "id")), *text = nj_string(nj_get(g, "text")), *e;
        const char *row = NULL;
        int n = 0;
        if (!id || !text) { fprintf(stderr, "%s: malformed fixture\n", argv[3]); return 1; }
        size_t len = strlen(id);
        for (const char *p = fixtures; p && *p && !row; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL)
            if (!strncmp(p, id, len) && p[len] == '\t') row = p + len + 1;
        for (const char *p = row; p && *p && *p != '\n' && *p != '\t' && n < 8192; p = e) { /* IDs, then an optional mask */
            want[n++] = (int)strtol(p, &e, 10);
            if (e == p) { n = 0; break; }
        }
        compare(&t, id, text, want, n);
        free(id); free(text);
    }
    printf("tokenizer gate (%d added-token literals refused):", t.n_added);
    for (int k = 0; k < EG2_CONDITIONS; k++) printf(" %s %d", eg2_conditions[k][0], per[k]);
    printf("; %d/%d sequences identical\n", identical, total);
    fclose(f); free(line); free(golden); free(fixtures); eg2_tokenizer_free(&t);
    return !bad && total > 0 && identical == total ? 0 : 1;
}

/* ── synthetic plumbing ── */
enum { VALID, REPEATED_ID, NOT_SPECIAL };
static void write_tokenizer(const char *path, int variant) {
    FILE *f = fopen(path, "w");
    fprintf(f, "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":0,\"content\":\"<pad>\",\"special\":%s},"
               "{\"id\":1,\"content\":\"<eos>\",\"special\":true},{\"id\":2,\"content\":\"<bos>\",\"special\":true}],\n", variant == NOT_SPECIAL ? "false" : "true");
    fprintf(f,
               "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"\xE2\x96\x81\"},\"pre_tokenizer\":null,\n"
               "\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"<bos>\",\"type_id\":0}},"
               "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"SpecialToken\":{\"id\":\"<eos>\",\"type_id\":0}}],"
               "\"special_tokens\":{\"<bos>\":{\"id\":\"<bos>\",\"ids\":[2],\"tokens\":[\"<bos>\"]},"
               "\"<eos>\":{\"id\":\"<eos>\",\"ids\":[1],\"tokens\":[\"<eos>\"]}}},\n"
               "\"model\":{\"type\":\"BPE\",\"byte_fallback\":true,\"vocab\":{\"<pad>\":0,\"<eos>\":1,\"<bos>\":2,"
               "\"a\":3,\"b\":4,\"\\u0063\":5,\"ab\":6,\"abc\":7,\"\xE2\x96\x81\":8,\"\\ud83c\\udf66\":%d", variant == REPEATED_ID ? 8 : 9);
    for (int b = 0; b < 255; b++) fprintf(f, ",\"<0x%02X>\":%d", b, 10 + b); /* no <0xFF>: one byte nothing carries */
    fprintf(f, "},\n\"merges\":[[\"a\",\"b\"],[\"ab\",\"\\u0063\"]]}}\n");
    fclose(f);
}
#define V 265
static void write_model(const char *path) {
    static float embed[V * 2];
    static const float ones[] = {1, 1}, zero[] = {0, 0, 0, 0}, identity[] = {1, 0, 0, 1}, scalar[] = {0.75f};
    for (int i = 0; i < V; i++) { embed[2 * i] = (float)(1 + i % 3); embed[2 * i + 1] = (float)(i % 5) - 2; }
    struct { const char *name; int rows, cols, rank; const float *data; } t[] = {
        {"embed_tokens.weight", V, 2, 2, embed}, {"embedding_projection.weight", 2, 2, 2, identity},
        {"norm.weight", 1, 2, 1, ones}, {"ple.per_layer_model_projection.weight", 2, 2, 2, zero},
        {"ple.per_layer_projection_norm.weight", 1, 2, 1, ones},
        {"layers.0.input_layernorm.weight", 1, 2, 1, ones}, {"layers.0.post_attention_layernorm.weight", 1, 2, 1, ones},
        {"layers.0.pre_feedforward_layernorm.weight", 1, 2, 1, ones}, {"layers.0.post_feedforward_layernorm.weight", 1, 2, 1, ones},
        {"layers.0.self_attn.q_norm.weight", 1, 2, 1, ones}, {"layers.0.self_attn.k_norm.weight", 1, 2, 1, ones},
        {"layers.0.self_attn.q_proj.weight", 2, 2, 2, zero}, {"layers.0.self_attn.k_proj.weight", 2, 2, 2, zero},
        {"layers.0.self_attn.v_proj.weight", 2, 2, 2, identity}, {"layers.0.self_attn.o_proj.weight", 2, 2, 2, identity},
        {"layers.0.mlp.gate_proj.weight", 2, 2, 2, zero}, {"layers.0.mlp.up_proj.weight", 2, 2, 2, zero},
        {"layers.0.mlp.down_proj.weight", 2, 2, 2, zero}, {"layers.0.ple_block.per_layer_input_gate.weight", 2, 2, 2, identity},
        {"layers.0.ple_block.per_layer_projection.weight", 2, 2, 2, identity},
        {"layers.0.ple_block.post_per_layer_input_norm.weight", 1, 2, 1, ones}, {"layers.0.layer_scalar", 1, 1, 1, scalar},
    };
    const char *keys[] = {"block_count", "embedding_length", "output_length", "vocab_size", "context_length",
                          "attention.head_count", "feed_forward_length", "ple_length", "layer.0.head_dim",
                          "layer.0.kv_heads", "layer.0.window"};
    int vals[] = {1, 2, 2, V, 8, 1, 2, 2, 2, 1, 0};
    char key[128];
    gguf_writer *w = gguf_write_open(path);
    gguf_write_kv_str(w, "general.architecture", "embeddinggemma2");
    for (int i = 0; i < 11; i++) { snprintf(key, sizeof(key), "embeddinggemma2.%s", keys[i]); gguf_write_kv_u32(w, key, (uint32_t)vals[i]); }
    gguf_write_kv_f32(w, "embeddinggemma2.rms_epsilon", 1e-6f);
    gguf_write_kv_f32(w, "embeddinggemma2.embed_scale", 1);
    gguf_write_kv_f32(w, "embeddinggemma2.ple_scale", 1);
    gguf_write_kv_f32(w, "embeddinggemma2.layer.0.rope_base", 10000);
    for (size_t i = 0; i < sizeof(t) / sizeof(*t); i++) {
        uint64_t dims[] = {(uint64_t)t[i].cols, (uint64_t)t[i].rows};
        gguf_write_tensor_decl(w, t[i].name, (uint32_t)t[i].rank, dims, GGUF_TYPE_F32);
    }
    for (size_t i = 0; i < sizeof(t) / sizeof(*t); i++)
        gguf_write_tensor_f32(w, t[i].name, t[i].data, (uint64_t)t[i].rows * (uint64_t)t[i].cols);
    if (gguf_write_close(w)) { fprintf(stderr, "analytic model write failed\n"); exit(1); }
}
static int sh(const char *cmd) {
    int s = system(cmd);
    return WIFEXITED(s) ? WEXITSTATUS(s) : 128;
}
static void put(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f || fputs(text, f) < 0 || fclose(f)) { perror(path); exit(1); }
}
/* The raw JSON of one key in one line of a JSONL buffer. */
static char *field(const char *jsonl, int row, const char *key) {
    const char *p = jsonl;
    for (int i = 0; i < row && p; i++) if ((p = strchr(p, '\n'))) p++;
    const char *v = p ? nj_get(p, key) : NULL, *e = v ? nj_skip(v, 0) : NULL;
    char *s = e ? malloc((size_t)(e - v) + 1) : NULL;
    if (s) { memcpy(s, v, (size_t)(e - v)); s[e - v] = 0; }
    return s;
}

int main(int argc, char **argv) {
    if (argc == 5) return gate(argv);
    if (argc != 1) { fprintf(stderr, "usage: %s [TOKENIZER.gguf ENCODED.jsonl GOLDEN.json FIXTURE_TOKENS.tsv]\n", argv[0]); return 2; }
    char dir[] = "/tmp/notorch-eg2-text-XXXXXX", path[256], cmd[1024], convert[512], tsv[256];
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    snprintf(convert, sizeof(convert), "./embeddinggemma2_to_gguf --tokenizer %s %s/tok.gguf >/dev/null 2>&1", path, dir);
    write_tokenizer(path, REPEATED_ID);
    CHECK(sh(convert) != 0, "a vocabulary with a repeated ID is refused");
    write_tokenizer(path, NOT_SPECIAL);
    CHECK(sh(convert) != 0, "an added token that is not special is refused");
    write_tokenizer(path, VALID);
    CHECK(sh(convert) == 0, "converter writes the tokenizer GGUF");
    CHECK(sh(convert) != 0, "converter refuses an existing output");

    snprintf(path, sizeof(path), "%s/tok.gguf", dir);
    eg2_tokenizer t;
    if (eg2_tokenizer_load(&t, path)) return 1;
    CHECK(bpe_n_vocab(t.bpe) == V && bpe_bos_id(t.bpe) == 2 && t.eos == 1 && t.n_added == 3, "tokenizer loads with its framing");
    int ids[64], whole[64];
    struct { const char *text; int n, ids[8]; } cases[] = {
        {"abc", 3, {2, 7, 1}},                     /* both merges, in rank order */
        {"ab c", 5, {2, 6, 8, 5, 1}},              /* a space is U+2581 and blocks the second merge */
        {"cab", 4, {2, 5, 6, 1}},                  /* "c ab" is not a merge */
        {"\xF0\x9F\x8D\xA6" "a", 4, {2, 9, 3, 1}}, /* a surrogate-pair escape is one 4-byte token */
        {"az", 4, {2, 3, 10 + 'z', 1}},            /* byte fallback */
        {"a<eo", 6, {2, 3, 10 + '<', 10 + 'e', 10 + 'o', 1}}, /* part of a literal is ordinary text */
        {"", 2, {2, 1}},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        int n = eg2_text_ids(&t, "", cases[i].text, ids, 64, NULL);
        if (!same(ids, n, cases[i].ids, cases[i].n)) fprintf(stderr, "case \"%s\": %d IDs\n", cases[i].text, n);
        CHECK(same(ids, n, cases[i].ids, cases[i].n), "hand-worked IDs");
    }
    const char *literal = NULL;
    CHECK(eg2_text_ids(&t, "", "a<eos>b", ids, 64, &literal) == -4 && literal && !strcmp(literal, "<eos>"),
          "an added-token literal is refused and named");
    CHECK(eg2_text_ids(&t, "", "a\xFF", ids, 64, NULL) == -2, "a byte nothing carries is refused, not dropped");
    CHECK(eg2_text_ids(&t, "", "abc", ids, 3, NULL) == 3 && eg2_text_ids(&t, "", "abc", ids, 2, NULL) == -1, "cap refuses, never truncates");
    int a = eg2_text_ids(&t, eg2_prefix("query"), "ab c", ids, 64, NULL);
    int b = eg2_text_ids(&t, "", "task: search result | query: ab c", whole, 64, NULL);
    CHECK(a > 5 && same(ids, a, whole, b), "prefix and text are one string");
    CHECK(!strcmp(eg2_prefix("document"), "title: none | text: ") && !eg2_prefix("Document"), "upstream condition names");
    eg2_tokenizer_free(&t);

    snprintf(path, sizeof(path), "%s/model.gguf", dir);
    write_model(path);
    snprintf(tsv, sizeof(tsv), "%s/in.tsv", dir);
    struct { const char *rows; int ok; } rows[] = {
        {"t1\tabc\nt2\tab c\n", 1}, {"t3\ta\tb\n", 0}, {"t4\tabc\r\n", 0}, {"t5\tab ab ab ab\n", 0}, {"t6\ta\xFF\n", 0},
    };
    for (size_t i = 0; i < sizeof(rows) / sizeof(*rows); i++) {
        put(tsv, rows[i].rows);
        snprintf(cmd, sizeof(cmd), "./embeddinggemma2_encode %s/model.gguf %s %s/out%zu.jsonl "
                 "--tokenizer %s/tok.gguf --condition none 2>/dev/null", dir, tsv, dir, i, dir);
        CHECK((sh(cmd) == 0) == rows[i].ok, rows[i].ok ? "CLI text mode runs" : "CLI refuses TAB, CR, over-long and lossy rows");
    }
    snprintf(cmd, sizeof(cmd), "./embeddinggemma2_encode %s/model.gguf %s %s/bad.jsonl --tokenizer %s/tok.gguf "
             "--condition Document 2>/dev/null", dir, tsv, dir, dir);
    CHECK(sh(cmd) != 0, "unknown condition refused");
    put(tsv, "t7\ta<eos>b\n");
    snprintf(cmd, sizeof(cmd), "./embeddinggemma2_encode %s/model.gguf %s %s/literal.jsonl --tokenizer %s/tok.gguf "
             "--condition none 2>%s/literal.err", dir, tsv, dir, dir, dir);
    int literal_rc = sh(cmd);
    snprintf(path, sizeof(path), "%s/literal.err", dir);
    char *err = nj_load(path, NULL);
    CHECK(literal_rc != 0 && err && strstr(err, "row 1: holds the added-token literal \"<eos>\""), "CLI refuses an added-token literal by name");
    free(err);
    FILE *nul = fopen(tsv, "wb");
    if (!nul || fwrite("t8\tab\0c\n", 1, 8, nul) != 8 || fclose(nul)) { perror(tsv); return 1; }
    snprintf(cmd, sizeof(cmd), "./embeddinggemma2_encode %s/model.gguf %s %s/nul.jsonl --tokenizer %s/tok.gguf "
             "--condition none 2>%s/nul.err", dir, tsv, dir, dir, dir);
    int nul_rc = sh(cmd);
    snprintf(path, sizeof(path), "%s/nul.err", dir);
    err = nj_load(path, NULL);
    CHECK(nul_rc != 0 && err && strstr(err, "row 1: NUL byte inside the text"), "CLI refuses a NUL byte instead of encoding the prefix");
    free(err);
    put(tsv, "t1\t2 7 1\n");
    snprintf(cmd, sizeof(cmd), "./embeddinggemma2_encode %s/model.gguf %s %s/ids.jsonl 2>/dev/null", dir, tsv, dir);
    CHECK(sh(cmd) == 0, "CLI ID mode runs");
    snprintf(path, sizeof(path), "%s/out0.jsonl", dir);
    char *text_out = nj_load(path, NULL);
    snprintf(path, sizeof(path), "%s/ids.jsonl", dir);
    char *id_out = nj_load(path, NULL);
    char *i1 = text_out ? field(text_out, 0, "input_ids") : NULL, *i2 = text_out ? field(text_out, 1, "input_ids") : NULL;
    char *e1 = text_out ? field(text_out, 0, "embedding") : NULL, *e0 = id_out ? field(id_out, 0, "embedding") : NULL;
    CHECK(i1 && !strcmp(i1, "[2,7,1]") && i2 && !strcmp(i2, "[2,6,8,5,1]"), "IDs written next to each vector");
    CHECK(e1 && e0 && !strcmp(e1, e0), "text mode encodes what ID mode encodes for the same IDs");
    CHECK(id_out && !strstr(id_out, "input_ids"), "ID mode output unchanged");
    free(i1); free(i2); free(e1); free(e0); free(text_out); free(id_out);
    const char *made[] = {"tokenizer.json", "tok.gguf", "model.gguf", "in.tsv", "ids.jsonl", "out0.jsonl", "out1.jsonl",
                          "out2.jsonl", "out3.jsonl", "out4.jsonl", "literal.jsonl", "literal.err", "nul.jsonl", "nul.err"};
    for (size_t i = 0; i < sizeof(made) / sizeof(*made); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, made[i]);
        unlink(path);
    }
    CHECK(rmdir(dir) == 0, "no stray output left behind");
    if (failures) { fprintf(stderr, "EmbeddingGemma2 text: %d FAILED\n", failures); return 1; }
    puts("EmbeddingGemma2 text: converter, two-merge vocabulary, framing, refusals and CLI text mode PASS");
    return 0;
}
