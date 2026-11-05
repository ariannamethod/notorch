/* Text to EmbeddingGemma 2 input IDs. The condition's task prefix and the text are
 * one string and are tokenized whole, because a merge may cross the place where
 * they meet; the sequence is framed <bos> ... <eos>. The prefixes are the upstream
 * sentence-transformers prompts under their upstream names
 * (config_sentence_transformers.json of google/embeddinggemma-2).
 *
 * A text containing the literal of an added token (<eos>, <|turn>, ...) is
 * refused: the upstream tokenizer would turn it into that token, and that
 * behaviour has not been verified against the reference, so it stays out. */
#ifndef EMBEDDINGGEMMA2_TEXT_H
#define EMBEDDINGGEMMA2_TEXT_H
#include "examples/bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const eg2_conditions[][2] = {
    {"query", "task: search result | query: "},
    {"document", "title: none | text: "},
    {"QuestionAnswering", "task: question answering | query: "},
    {"SentenceSimilarity", "task: sentence similarity | query: "},
    {"none", ""},
};
#define EG2_CONDITIONS (int)(sizeof(eg2_conditions) / sizeof(eg2_conditions[0]))

typedef struct { bpe_tokenizer *bpe; int eos; char **added; int n_added; } eg2_tokenizer;

static inline const char *eg2_prefix(const char *condition) {
    for (int i = 0; i < EG2_CONDITIONS; i++)
        if (!strcmp(condition, eg2_conditions[i][0])) return eg2_conditions[i][1];
    return NULL;
}
static inline void eg2_tokenizer_free(eg2_tokenizer *t) {
    for (int i = 0; i < t->n_added; i++) free(t->added[i]);
    free(t->added);
    bpe_free(t->bpe);
    memset(t, 0, sizeof(*t));
}
/* The tokenizer GGUF of embeddinggemma2_to_gguf: it must declare BOS and EOS
 * framing; the strings of its CONTROL (type 3) tokens become the refused literals. */
static inline int eg2_tokenizer_load(eg2_tokenizer *t, const char *path) {
    uint64_t eos = 0, add_eos = 0;
    int nt = 0, ntype = 0;
    memset(t, 0, sizeof(*t));
    t->bpe = bpe_load(path);
    char **tokens = t->bpe ? gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt) : NULL;
    int32_t *types = tokens ? gguf_read_i32_array(path, "tokenizer.ggml.token_type", &ntype) : NULL;
    int ok = types && ntype == nt && nt == bpe_n_vocab(t->bpe) && bpe_add_bos(t->bpe) && bpe_bos_declared(t->bpe) &&
             !gguf_read_uint_kv(path, "tokenizer.ggml.add_eos_token", &add_eos) && add_eos == 1 &&
             !gguf_read_uint_kv(path, "tokenizer.ggml.eos_token_id", &eos) && eos < (uint64_t)nt &&
             (t->added = calloc((size_t)nt, sizeof(char *)));
    for (int i = 0; ok && i < nt; i++)
        if (types[i] == 3 && tokens[i] && tokens[i][0]) { t->added[t->n_added++] = tokens[i]; tokens[i] = NULL; }
    for (int i = 0; tokens && i < nt; i++) free(tokens[i]);
    free(tokens);
    free(types);
    t->eos = (int)eos;
    if (!ok) {
        fprintf(stderr, "%s: not a tokenizer with declared <bos>/<eos> framing and token types\n", path);
        eg2_tokenizer_free(t);
        return -1;
    }
    return 0;
}
/* IDs of prefix+text into ids[0..cap). Returns their count; -1 when the framed
 * sequence does not fit in cap (refused, never truncated); -2 when the IDs do not
 * decode back to the input, which is how a character that neither a token nor a
 * byte fallback carries shows up (the tokenizer drops it); -3 out of memory;
 * -4 when the input holds an added-token literal, named in *literal. */
static inline int eg2_text_ids(const eg2_tokenizer *t, const char *prefix, const char *text,
                               int *ids, int cap, const char **literal) {
    size_t a = strlen(prefix), b = strlen(text), at = 0;
    char *s = malloc(a + b + 1), piece[1024];
    if (!s) return -3;
    memcpy(s, prefix, a);
    memcpy(s + a, text, b + 1);
    for (int i = 0; i < t->n_added; i++)
        if (strstr(s, t->added[i])) { if (literal) *literal = t->added[i]; free(s); return -4; }
    int n = bpe_encode(t->bpe, s, ids, cap), rc = n;
    if (n >= cap) rc = -1;
    else if (n < 1 || ids[0] != bpe_bos_id(t->bpe)) rc = -2;
    for (int i = 1; rc > 0 && i < n; i++) {
        int c = bpe_decode_token_capacity(t->bpe, ids[i]);
        int k = c > 0 && c <= (int)sizeof(piece) ? bpe_decode_token(t->bpe, ids[i], piece, sizeof(piece)) : -1;
        if (k < 0 || at + (size_t)k > a + b || memcmp(s + at, piece, (size_t)k)) rc = -2;
        else at += (size_t)k;
    }
    if (rc > 0 && at != a + b) rc = -2;
    if (rc > 0) { ids[n] = t->eos; rc = n + 1; }
    free(s);
    return rc;
}
#endif
