/* Qwen3-Embedding text plumbing: tokenizer.json -> C converter -> tokenizer GGUF
 * -> bpe.c "qwen3" split -> prompt, <|endoftext|> -> CLI text mode.
 *
 * Without arguments: a synthetic byte-level vocabulary whose merges straddle the
 * boundaries of the Qwen3 split, converted by the real binary, so a piece cut in
 * the wrong place shows as a merge made or missed; expected IDs are worked by
 * hand below. Then the CLI over an analytic model: IDs beside each vector, text
 * mode equal to ID mode, and every refusal.
 *
 * With TOKENIZER.gguf FIXTURES.tsv: our IDs for each fixture (id TAB condition
 * TAB text) as JSONL on stdout, for comparison. With a third argument
 * REFERENCE.jsonl ({"id", "input_ids"} per fixture, from the official tokenizer):
 * the gate, every fixture's IDs identical to the reference. */
#include "gguf.h"
#include "tools/qwen3embedding_text.h"
#include "tools/narrow_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures, checks;
#define CHECK(c, what) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL %s (line %d)\n", what, __LINE__); } } while (0)
static int same(const int *a, int n, const int *b, int m) { return n == m && n > 0 && !memcmp(a, b, (size_t)n * sizeof(int)); }

/* ── fixtures against a reference ── */
static int fixtures(char **argv, int argc) {
    static int got[8192], want[8192];
    qe_tokenizer t;
    char *rows = qe_tokenizer_load(&t, argv[1]) ? NULL : nj_load(argv[2], NULL);
    char *ref = rows && argc == 4 ? nj_load(argv[3], NULL) : NULL;
    if (!rows || (argc == 4 && !ref)) { fprintf(stderr, "inputs unreadable\n"); return 1; }
    int total = 0, identical = 0, bad = 0;
    for (char *line = rows, *next; line && *line; line = next) {
        next = strchr(line, '\n');
        if (next) *next++ = 0;
        char *c = strchr(line, '\t'), *text = c ? strchr(c + 1, '\t') : NULL;
        if (!text) { fprintf(stderr, "%s: row is not id TAB condition TAB text\n", argv[2]); return 1; }
        *c++ = 0; *text++ = 0;
        const char *prefix = qe_prefix(c), *literal = NULL;
        if (!prefix) { fprintf(stderr, "%s: unknown condition %s\n", line, c); return 1; }
        int n = qe_text_ids(&t, prefix, text, got, 8192, &literal);
        total++;
        if (argc == 3) {
            printf("{\"id\":\"%s\",\"condition\":\"%s\",\"input_ids\":[", line, c);
            for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", got[i]);
            printf("]%s}\n", n < 0 ? ",\"refused\":true" : "");
            bad |= n < 0;
            continue;
        }
        const char *r = NULL;
        size_t len = strlen(line);
        for (const char *p = ref; p && *p && !r; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL) {
            const char *id = nj_get(p, "id");
            if (id && *id == '"' && !strncmp(id + 1, line, len) && id[len + 1] == '"') r = p;
        }
        int m = 0, b = 0;
        long long v;
        for (const char *x = r ? nj_first(nj_get(r, "input_ids"), &b) : NULL; x && m < 8192; x = nj_next(x, ']', &b)) {
            if (nj_int(x, &v)) { b = 1; break; }
            want[m++] = (int)v;
        }
        if (!r || b) { fprintf(stderr, "MISSING %s: no reference row\n", line); continue; }
        if (same(got, n, want, m)) { identical++; continue; }
        int at = 0;
        while (at < n && at < m && got[at] == want[at]) at++;
        fprintf(stderr, "MISMATCH %s: %d IDs, reference %d, first difference at %d\n", line, n, m, at);
    }
    qe_tokenizer_free(&t); free(rows); free(ref);
    if (argc == 3) return bad;
    printf("Qwen3-Embedding tokenizer gate: %d/%d sequences identical to the reference\n", identical, total);
    return total >= 20 && identical == total ? 0 : 1;
}

/* ── synthetic plumbing ── */
static char sym[256][4];   /* the byte-level spelling of each byte */
static const unsigned char pairs[][2] = {
    {'1', '2'},  {'a', '1'},  {'a', '!'},  {'!', 'a'},  {'\'', 's'}, {'s', 'a'},  {' ', ' '},  {0, 0} /* "  " + a */,
    {' ', 'a'},  {'a', '\n'}, {'a', 0xD6}, {0xB7, 'a'}, {'!', 0xD6}, {'\t', 'a'}, {0xA0, 'a'}, {0xBF, '\n'},
    {0x90, '\n'}, {0xA1, 0xD9}, {0xBF, 'a'}, {'!', '\n'},
};
#define NP (int)(sizeof(pairs) / sizeof(pairs[0]))
#define NV (256 + NP + 2)
#define EOT (256 + NP)
static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) { if (*s == '"' || *s == '\\') fputc('\\', f); fputc(*s, f); }
    fputc('"', f);
}
enum { VALID, MARKS_REGEX, FALLBACK, WITH_BOS, LSTRIP };
static void write_tokenizer(const char *path, int variant) {
    const char *split = variant == MARKS_REGEX
        ? "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"
        : "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
    FILE *f = fopen(path, "w");
    fprintf(f, "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":%d,\"content\":\"<|endoftext|>\",\"single_word\":false,"
               "\"lstrip\":%s,\"rstrip\":false,\"normalized\":false,\"special\":true},{\"id\":%d,\"content\":\"<think>\","
               "\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false}],\n",
            EOT, variant == LSTRIP ? "true" : "false", EOT + 1);
    fprintf(f, "\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
               "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"%s\"},\"behavior\":\"Isolated\",\"invert\":false},"
               "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]},\n", split);
    fprintf(f, "\"post_processor\":{\"type\":\"Sequence\",\"processors\":[{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
               "\"trim_offsets\":false,\"use_regex\":false},{\"type\":\"TemplateProcessing\",\"single\":[%s"
               "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"SpecialToken\":{\"id\":\"<|endoftext|>\",\"type_id\":0}}],"
               "\"special_tokens\":{\"<|endoftext|>\":{\"id\":\"<|endoftext|>\",\"ids\":[%d],\"tokens\":[\"<|endoftext|>\"]}}}]},\n",
            variant == WITH_BOS ? "{\"SpecialToken\":{\"id\":\"<|endoftext|>\",\"type_id\":0}}," : "", EOT);
    fprintf(f, "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\","
               "\"end_of_word_suffix\":\"\",\"fuse_unk\":false,\"byte_fallback\":%s,\"ignore_merges\":false,\"vocab\":{",
            variant == FALLBACK ? "true" : "false");
    char left[NP][16], right[NP][16];
    for (int b = 0; b < 256; b++) { fputs(b ? "," : "", f); json_string(f, sym[b]); fprintf(f, ":%d", b); }
    for (int i = 0; i < NP; i++) {
        if (i == 7) { strcpy(left[i], "\xC4\xA0\xC4\xA0"); strcpy(right[i], sym['a']); }   /* the token "  " + a */
        else { strcpy(left[i], sym[pairs[i][0]]); strcpy(right[i], sym[pairs[i][1]]); }
        char merged[32];
        snprintf(merged, sizeof(merged), "%s%s", left[i], right[i]);
        fputc(',', f); json_string(f, merged); fprintf(f, ":%d", 256 + i);
    }
    fputs("},\"merges\":[", f);
    for (int i = 0; i < NP; i++) { fputs(i ? ",[" : "[", f); json_string(f, left[i]); fputc(',', f); json_string(f, right[i]); fputc(']', f); }
    fputs("]}}\n", f);
    fclose(f);
}
static void write_model(const char *path, int max_tokens) {
    enum { E = 4, H = 2, HD = 2, F = 3 };
    static float embed[NV * E], ones[E], q[H * HD * E], k[HD * E], v[HD * E], o[E * H * HD], g[F * E], u[F * E], d[E * F];
    for (int i = 0; i < NV * E; i++) embed[i] = (float)((i * 7) % 11) / 11.0f - 0.4f;
    for (int i = 0; i < E; i++) ones[i] = 1;
    float *mats[] = {q, k, v, o, g, u, d};
    int sizes[] = {H * HD * E, HD * E, HD * E, E * H * HD, F * E, F * E, E * F};
    for (int m = 0; m < 7; m++) for (int i = 0; i < sizes[m]; i++) mats[m][i] = (float)((i * 5 + m * 3) % 13) / 13.0f - 0.5f;
    struct { const char *name; int rows, cols, rank; const float *data; } t[] = {
        {"embed_tokens.weight", NV, E, 2, embed}, {"norm.weight", 1, E, 1, ones},
        {"layers.0.input_layernorm.weight", 1, E, 1, ones}, {"layers.0.post_attention_layernorm.weight", 1, E, 1, ones},
        {"layers.0.self_attn.q_norm.weight", 1, HD, 1, ones}, {"layers.0.self_attn.k_norm.weight", 1, HD, 1, ones},
        {"layers.0.self_attn.q_proj.weight", H * HD, E, 2, q}, {"layers.0.self_attn.k_proj.weight", HD, E, 2, k},
        {"layers.0.self_attn.v_proj.weight", HD, E, 2, v}, {"layers.0.self_attn.o_proj.weight", E, H * HD, 2, o},
        {"layers.0.mlp.gate_proj.weight", F, E, 2, g}, {"layers.0.mlp.up_proj.weight", F, E, 2, u},
        {"layers.0.mlp.down_proj.weight", E, F, 2, d},
    };
    const char *keys[] = {"block_count", "embedding_length", "vocab_size", "context_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length", "feed_forward_length"};
    int vals[] = {1, E, NV, max_tokens, H, 1, HD, F};
    char key[128];
    gguf_writer *w = gguf_write_open(path);
    gguf_write_kv_str(w, "general.architecture", "qwen3embedding");
    for (int i = 0; i < 8; i++) { snprintf(key, sizeof(key), "qwen3embedding.%s", keys[i]); gguf_write_kv_u32(w, key, (uint32_t)vals[i]); }
    gguf_write_kv_f32(w, "qwen3embedding.rms_epsilon", 1e-6f);
    gguf_write_kv_f32(w, "qwen3embedding.rope.freq_base", 100);
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
static char *field(const char *jsonl, int row, const char *key) {
    const char *p = jsonl;
    for (int i = 0; i < row && p; i++) if ((p = strchr(p, '\n'))) p++;
    const char *v = p ? nj_get(p, key) : NULL, *e = v ? nj_skip(v, 0) : NULL;
    char *s = e ? malloc((size_t)(e - v) + 1) : NULL;
    if (s) { memcpy(s, v, (size_t)(e - v)); s[e - v] = 0; }
    return s;
}

int main(int argc, char **argv) {
    if (argc == 3 || argc == 4) return fixtures(argv, argc);
    if (argc != 1) { fprintf(stderr, "usage: %s [TOKENIZER.gguf FIXTURES.tsv [REFERENCE.jsonl]]\n", argv[0]); return 2; }
    for (int b = 0, next = 0; b < 256; b++) {
        int literal = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        int cp = literal ? b : 256 + next++;
        if (cp < 128) sym[b][0] = (char)cp;
        else { sym[b][0] = (char)(0xc0 | (cp >> 6)); sym[b][1] = (char)(0x80 | (cp & 63)); }
    }
    char dir[] = "/tmp/notorch-qe-text-XXXXXX", path[256], cmd[1024], convert[512], tsv[256];
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    snprintf(convert, sizeof(convert), "./qwen3embedding_to_gguf --tokenizer %s %s/tok.gguf >/dev/null 2>&1", path, dir);
    const int refusals[] = {MARKS_REGEX, FALLBACK, WITH_BOS, LSTRIP};
    for (int i = 0; i < 4; i++) { write_tokenizer(path, refusals[i]); CHECK(sh(convert) != 0, "converter refuses a tokenizer of another scheme"); }
    write_tokenizer(path, VALID);
    CHECK(sh(convert) == 0, "converter writes the tokenizer GGUF");
    CHECK(sh(convert) != 0, "converter refuses an existing output");

    snprintf(path, sizeof(path), "%s/tok.gguf", dir);
    qe_tokenizer t;
    if (qe_tokenizer_load(&t, path)) return 1;
    CHECK(bpe_n_vocab(t.bpe) == NV && t.eos == EOT && t.n_added == 2 && !bpe_add_bos(t.bpe), "tokenizer loads with its framing");
    int ids[256], whole[256], n;
    /* The split, through bpe_encode. Merge IDs are 256 + their row in pairs[]; byte
     * IDs are the byte. A cut in the wrong place makes or misses a merge. */
    struct { const char *text; int n, ids[10]; } cases[] = {
        {"12", 2, {'1', '2'}},                         /* one number per piece */
        {"a1", 2, {'a', '1'}},                         /* letters | number */
        {"a!", 2, {'a', '!'}},                         /* letters | punctuation */
        {"!a", 1, {256 + 3}},                          /* punctuation prefixes a word */
        {"'sa", 2, {256 + 4, 'a'}},                    /* contraction, then the word */
        {"'\xC5\xBF" "a", 4, {'\'', 0xC5, 0xBF, 'a'}}, /* 'ſ folds to 's: ſ does not join a */
        {"  a", 2, {' ', 256 + 8}},                    /* the run gives its last space to the word */
        {"a\n", 2, {'a', '\n'}},                       /* word | newline */
        {"\ta", 1, {256 + 13}},                        /* a tab prefixes a word */
        {"\xC2\xA0" "a", 2, {0xC2, 256 + 14}},         /* so does NBSP */
        {"a\xD6\xB7", 3, {'a', 0xD6, 0xB7}},           /* a mark is not part of a word */
        {"!\xD6\xB7" "a", 3, {256 + 12, 0xB7, 'a'}},   /* ! and the mark are one punctuation piece */
        {"\xC2\xBF\n", 2, {0xC2, 256 + 15}},           /* punctuation keeps its newline */
        {"!\n", 1, {256 + 19}},
        {"\xD7\x90\n", 3, {0xD7, 0x90, '\n'}},         /* letter | newline */
        {"\xD9\xA1\xD9\xA2", 4, {0xD9, 0xA1, 0xD9, 0xA2}}, /* Arabic-Indic digits: one each */
        {"\xD7\x91\xD6\xB7\xD6\xBC", 6, {0xD7, 0x91, 0xD6, 0xB7, 0xD6, 0xBC}}, /* bet patah dagesh, NFC order */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        n = bpe_encode(t.bpe, cases[i].text, ids, 64);
        if (!same(ids, n, cases[i].ids, cases[i].n)) {
            fprintf(stderr, "case %zu: %d IDs:", i, n);
            for (int j = 0; j < n; j++) fprintf(stderr, " %d", ids[j]);
            fputc('\n', stderr);
        }
        CHECK(same(ids, n, cases[i].ids, cases[i].n), "hand-worked IDs");
    }
    CHECK(same(ids, qe_text_ids(&t, "", "!a", ids, 64, NULL), (int[]){256 + 3, EOT}, 2), "<|endoftext|> appended, no BOS");
    CHECK(same(ids, qe_text_ids(&t, "", "", ids, 64, NULL), (int[]){EOT}, 1), "an empty document is <|endoftext|> alone");
    /* bet dagesh patah typed out of canonical order: bpe.c sorts it as NFC does, and the
     * text layer refuses it, because the IDs no longer spell the string it was given. */
    n = bpe_encode(t.bpe, "\xD7\x91\xD6\xBC\xD6\xB7", ids, 64);
    CHECK(same(ids, n, (int[]){0xD7, 0x91, 0xD6, 0xB7, 0xD6, 0xBC}, 6), "canonical reordering of marks");
    CHECK(qe_text_ids(&t, "", "\xD7\x91\xD6\xBC\xD6\xB7", ids, 64, NULL) == -2, "text NFC would change is refused");
    const char *refused[] = {"\xD0\xB8\xCC\x86", "\xE2\x84\xAB", "\xFF", "a\xC3"};
    for (int i = 0; i < 4; i++) CHECK(qe_text_ids(&t, "", refused[i], ids, 64, NULL) == -2, "NFC quick check and invalid UTF-8 refused");
    const char *literal = NULL;
    CHECK(qe_text_ids(&t, "", "a<|endoftext|>", ids, 64, &literal) == -4 && literal && !strcmp(literal, "<|endoftext|>"), "special literal refused and named");
    CHECK(qe_text_ids(&t, "", "x<think>", ids, 64, &literal) == -4 && !strcmp(literal, "<think>"), "non-special added literal refused too");
    const char *spaced[] = {" a", "a ", "a\xC2\xA0", "a\xE3\x80\x80", "a\x1F", ("\xE2\x80\xA8" "a")};
    for (int i = 0; i < 6; i++) CHECK(qe_text_ids(&t, "", spaced[i], ids, 64, NULL) == -5, "leading or trailing whitespace refused");
    CHECK(qe_text_ids(&t, "", "a  a", ids, 64, NULL) > 0, "inner whitespace accepted");
    CHECK(qe_text_ids(&t, "", "12", ids, 3, NULL) == 3 && qe_text_ids(&t, "", "12", ids, 2, NULL) == -1, "cap refuses, never truncates");
    int a = qe_text_ids(&t, qe_prefix("query"), "!a", ids, 256, NULL);
    int b = qe_text_ids(&t, "", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:!a", whole, 256, NULL);
    CHECK(a > 20 && same(ids, a, whole, b), "prompt and text are one string");
    CHECK(!strcmp(qe_prefix("document"), "") && !qe_prefix("Query") && !qe_prefix("none"), "upstream condition names only");
    qe_tokenizer_free(&t);

    snprintf(path, sizeof(path), "%s/model.gguf", dir);
    write_model(path, 6);
    snprintf(tsv, sizeof(tsv), "%s/in.tsv", dir);
    struct { const char *rows; int ok; } rows[] = {
        {"t1\t!a\nt2\t12\n", 1}, {"t3\ta\tb\n", 0}, {"t4\ta\r\n", 0}, {"t5\t1 2 3 4 5 6\n", 0}, {"t6\ta\xFF\n", 0},
        {"t7\ta \n", 0},
    };
    for (size_t i = 0; i < sizeof(rows) / sizeof(*rows); i++) {
        put(tsv, rows[i].rows);
        snprintf(cmd, sizeof(cmd), "./qwen3embedding_encode %s/model.gguf %s %s/out%zu.jsonl "
                 "--tokenizer %s/tok.gguf --condition document 2>/dev/null", dir, tsv, dir, i, dir);
        CHECK((sh(cmd) == 0) == rows[i].ok, rows[i].ok ? "CLI text mode runs" : "CLI refuses TAB, CR, over-long, lossy and padded rows");
    }
    snprintf(cmd, sizeof(cmd), "./qwen3embedding_encode %s/model.gguf %s %s/bad.jsonl --tokenizer %s/tok.gguf "
             "--condition none 2>/dev/null", dir, tsv, dir, dir);
    CHECK(sh(cmd) != 0, "unknown condition refused");
    put(tsv, "t8\ta<|endoftext|>\n");
    snprintf(cmd, sizeof(cmd), "./qwen3embedding_encode %s/model.gguf %s %s/literal.jsonl --tokenizer %s/tok.gguf "
             "--condition document 2>%s/literal.err", dir, tsv, dir, dir, dir);
    int literal_rc = sh(cmd);
    snprintf(path, sizeof(path), "%s/literal.err", dir);
    char *err = nj_load(path, NULL);
    CHECK(literal_rc != 0 && err && strstr(err, "row 1: holds the added-token literal \"<|endoftext|>\""), "CLI refuses an added-token literal by name");
    free(err);
    FILE *nul = fopen(tsv, "wb");
    if (!nul || fwrite("t9\t!a\0b\n", 1, 8, nul) != 8 || fclose(nul)) { perror(tsv); return 1; }
    snprintf(cmd, sizeof(cmd), "./qwen3embedding_encode %s/model.gguf %s %s/nul.jsonl --tokenizer %s/tok.gguf "
             "--condition document 2>%s/nul.err", dir, tsv, dir, dir, dir);
    int nul_rc = sh(cmd);
    snprintf(path, sizeof(path), "%s/nul.err", dir);
    err = nj_load(path, NULL);
    CHECK(nul_rc != 0 && err && strstr(err, "row 1: NUL byte inside the row"), "CLI refuses a NUL byte");
    free(err);
    snprintf(tsv, sizeof(tsv), "%s/ids.tsv", dir);
    char id_rows[64];
    snprintf(id_rows, sizeof(id_rows), "t1\t%d %d\n", 256 + 3, EOT);
    put(tsv, id_rows);
    snprintf(cmd, sizeof(cmd), "./qwen3embedding_encode %s/model.gguf %s %s/ids.jsonl 2>/dev/null", dir, tsv, dir);
    CHECK(sh(cmd) == 0, "CLI ID mode runs");
    snprintf(path, sizeof(path), "%s/out0.jsonl", dir);
    char *text_out = nj_load(path, NULL);
    snprintf(path, sizeof(path), "%s/ids.jsonl", dir);
    char *id_out = nj_load(path, NULL);
    char want1[32], want2[32];
    snprintf(want1, sizeof(want1), "[%d,%d]", 256 + 3, EOT);
    snprintf(want2, sizeof(want2), "[49,50,%d]", EOT);
    char *i1 = text_out ? field(text_out, 0, "input_ids") : NULL, *i2 = text_out ? field(text_out, 1, "input_ids") : NULL;
    char *e1 = text_out ? field(text_out, 0, "embedding") : NULL, *e0 = id_out ? field(id_out, 0, "embedding") : NULL;
    CHECK(i1 && !strcmp(i1, want1) && i2 && !strcmp(i2, want2), "IDs written next to each vector");
    CHECK(e1 && e0 && !strcmp(e1, e0), "text mode encodes what ID mode encodes for the same IDs");
    CHECK(id_out && !strstr(id_out, "input_ids"), "ID mode output carries no input_ids");
    free(i1); free(i2); free(e1); free(e0); free(text_out); free(id_out);
    const char *made[] = {"tokenizer.json", "tok.gguf", "model.gguf", "in.tsv", "ids.tsv", "ids.jsonl", "out0.jsonl",
                          "out1.jsonl", "out2.jsonl", "out3.jsonl", "out4.jsonl", "out5.jsonl", "literal.jsonl",
                          "literal.err", "nul.jsonl", "nul.err"};
    for (size_t i = 0; i < sizeof(made) / sizeof(*made); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, made[i]);
        unlink(path);
    }
    CHECK(rmdir(dir) == 0, "no stray output left behind");
    if (failures) { fprintf(stderr, "Qwen3-Embedding text: %d of %d checks FAILED\n", failures, checks); return 1; }
    printf("Qwen3-Embedding text: %d checks: converter refusals, Qwen3 split, NFC, framing, refusals and CLI text mode PASS\n", checks);
    return 0;
}
