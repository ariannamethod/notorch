/* Text to Qwen3-Embedding input IDs. The condition's prompt and the text are one
 * string, tokenized whole (NFC, the Qwen3 split, byte-level BPE), then the
 * post-processor's <|endoftext|> is appended; there is no BOS. The prompts are
 * the upstream sentence-transformers prompts under their upstream names
 * (config_sentence_transformers.json of Qwen/Qwen3-Embedding-0.6B): "query"
 * carries the web-search instruction, "document" carries none.
 *
 * Refused, never altered: a text holding the literal of an added token (the
 * upstream tokenizer would turn it into that token, a path not verified here);
 * a prompt+text that begins or ends with whitespace (sentence-transformers strips
 * it, the Transformers recipe does not, so the two official paths disagree on
 * it); text whose IDs do not decode back to it byte for byte (invalid UTF-8, or
 * text NFC would change). */
#ifndef QWEN3EMBEDDING_TEXT_H
#define QWEN3EMBEDDING_TEXT_H
#include "examples/bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const qe_conditions[][2] = {
    {"query", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"},
    {"document", ""},
};
#define QE_CONDITIONS (int)(sizeof(qe_conditions) / sizeof(qe_conditions[0]))

typedef struct { bpe_tokenizer *bpe; int eos; char **added; int n_added; } qe_tokenizer;

static inline const char *qe_prefix(const char *condition) {
    for (int i = 0; i < QE_CONDITIONS; i++)
        if (!strcmp(condition, qe_conditions[i][0])) return qe_conditions[i][1];
    return NULL;
}
static inline void qe_tokenizer_free(qe_tokenizer *t) {
    for (int i = 0; i < t->n_added; i++) free(t->added[i]);
    free(t->added);
    bpe_free(t->bpe);
    memset(t, 0, sizeof(*t));
}
/* The tokenizer GGUF of qwen3embedding_to_gguf: pre-type "qwen3", no BOS, EOS
 * appended; the strings of its CONTROL and USER_DEFINED tokens are refused. */
static inline int qe_tokenizer_load(qe_tokenizer *t, const char *path) {
    uint64_t eos = 0, add_eos = 0;
    int nt = 0, ntype = 0;
    char pre[16] = {0};
    memset(t, 0, sizeof(*t));
    t->bpe = bpe_load(path);
    char **tokens = t->bpe ? gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt) : NULL;
    int32_t *types = tokens ? gguf_read_i32_array(path, "tokenizer.ggml.token_type", &ntype) : NULL;
    int ok = types && ntype == nt && nt == bpe_n_vocab(t->bpe) && bpe_bos_declared(t->bpe) && !bpe_add_bos(t->bpe) &&
             !gguf_read_str_kv(path, "tokenizer.ggml.pre", pre, sizeof(pre)) && !strcmp(pre, "qwen3") &&
             !gguf_read_uint_kv(path, "tokenizer.ggml.add_eos_token", &add_eos) && add_eos == 1 &&
             !gguf_read_uint_kv(path, "tokenizer.ggml.eos_token_id", &eos) && eos < (uint64_t)nt &&
             (t->added = calloc((size_t)nt, sizeof(char *)));
    for (int i = 0; ok && i < nt; i++)
        if ((types[i] == 3 || types[i] == 4) && tokens[i] && tokens[i][0]) { t->added[t->n_added++] = tokens[i]; tokens[i] = NULL; }
    for (int i = 0; tokens && i < nt; i++) free(tokens[i]);
    free(tokens);
    free(types);
    t->eos = (int)eos;
    if (!ok) {
        fprintf(stderr, "%s: not a qwen3 tokenizer with declared EOS framing and token types\n", path);
        qe_tokenizer_free(t);
        return -1;
    }
    return 0;
}
/* What Python's str.strip() removes: Unicode whitespace (Zs, and bidi B, S, WS). */
static inline int qe_strip_space(unsigned int c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static inline unsigned int qe_codepoint(const unsigned char *s, int len) {
    unsigned int c = s[0];
    if (len == 1) return c;
    c &= 0x3Fu >> (len - 1);
    for (int i = 1; i < len; i++) c = (c << 6) | (s[i] & 0x3Fu);
    return c;
}
/* IDs of prefix+text into ids[0..cap). Returns their count; -1 when the framed
 * sequence does not fit in cap (refused, never truncated); -2 when the IDs do not
 * decode back to the input; -3 out of memory; -4 when the input holds an
 * added-token literal, named in *literal; -5 when prefix+text begins or ends with
 * whitespace. */
static inline int qe_text_ids(const qe_tokenizer *t, const char *prefix, const char *text,
                              int *ids, int cap, const char **literal) {
    size_t a = strlen(prefix), b = strlen(text), at = 0, len = a + b;
    char *s = malloc(len + 1), piece[1024];
    if (!s) return -3;
    memcpy(s, prefix, a);
    memcpy(s + a, text, b + 1);
    for (int i = 0; i < t->n_added; i++)
        if (strstr(s, t->added[i])) { if (literal) *literal = t->added[i]; free(s); return -4; }
    int n = len ? bpe_encode(t->bpe, s, ids, cap) : 0, rc = n;
    if (n >= cap) rc = -1;
    else if (len && n < 1) rc = -2;
    for (int i = 0; rc >= 0 && i < n; i++) {
        int c = bpe_decode_token_capacity(t->bpe, ids[i]);
        int k = c > 0 && c <= (int)sizeof(piece) ? bpe_decode_token(t->bpe, ids[i], piece, sizeof(piece)) : -1;
        if (k < 0 || at + (size_t)k > len || memcmp(s + at, piece, (size_t)k)) rc = -2;
        else at += (size_t)k;
    }
    if (rc >= 0 && at != len) rc = -2;
    if (rc >= 0 && len) {   /* valid UTF-8 from here on: it decoded back exactly */
        const unsigned char *u = (const unsigned char *)s;
        int first = u[0] < 0x80 ? 1 : u[0] < 0xE0 ? 2 : u[0] < 0xF0 ? 3 : 4, last = 1;
        while (last < 4 && (size_t)last < len && (u[len - last] & 0xC0) == 0x80) last++;
        if (qe_strip_space(qe_codepoint(u, first)) || qe_strip_space(qe_codepoint(u + len - last, last))) rc = -5;
    }
    if (rc >= 0) { ids[n] = t->eos; rc = n + 1; }
    free(s);
    return rc;
}
#endif
