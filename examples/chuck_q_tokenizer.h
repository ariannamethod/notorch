/* Q's byte BPE: ordered merge passes, compatible with q.merges and QPTQ.
 * Reference: ariannamethod/q f5d00a36, postgpt_q.c bpe_load/bpe_encode.
 * Store only the bounded merge graph; decoded bytes are reconstructed on demand. */
#ifndef NOTORCH_CHUCK_Q_TOKENIZER_H
#define NOTORCH_CHUCK_Q_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q_TOKENIZER_MAX_VOCAB 4096
#define Q_TOKENIZER_MAX_MERGES (Q_TOKENIZER_MAX_VOCAB - 256)
#define Q_TOKENIZER_TOKEN_BYTES 64

typedef struct {
    uint32_t vocab_size, merge_count;
    uint32_t pairs[Q_TOKENIZER_MAX_MERGES][2];
} q_tokenizer;

static int q_tokenizer_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

static int q_tokenizer_validate(const q_tokenizer *q, char *error, size_t size) {
    if (error && size) error[0] = 0;
    if (!q || q->merge_count > Q_TOKENIZER_MAX_MERGES ||
        q->vocab_size != 256 + q->merge_count)
        return q_tokenizer_error(error, size, "invalid Q vocabulary size");
    unsigned char lengths[Q_TOKENIZER_MAX_VOCAB];
    memset(lengths, 1, 256);
    for (uint32_t i = 0; i < q->merge_count; ++i) {
        uint32_t id = 256 + i, a = q->pairs[i][0], b = q->pairs[i][1];
        if (a >= id || b >= id)
            return q_tokenizer_error(error, size, "Q merge references an unavailable token");
        unsigned length = (unsigned)lengths[a] + lengths[b];
        /* The Q runtime's vocab_bytes[id][64] reserves one byte for NUL. */
        if (length >= Q_TOKENIZER_TOKEN_BYTES)
            return q_tokenizer_error(error, size, "Q token exceeds runtime byte limit");
        lengths[id] = (unsigned char)length;
    }
    return 0;
}

static int q_tokenizer_read_u32(FILE *f, uint32_t *value) {
    unsigned char b[4];
    if (fread(b, 1, sizeof(b), f) != sizeof(b)) return -1;
    *value = (uint32_t)b[0] | (uint32_t)b[1] << 8 |
             (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 0;
}

static int q_tokenizer_load(q_tokenizer *out, const char *path,
                            char *error, size_t size) {
    if (error && size) error[0] = 0;
    if (!out || !path || !*path)
        return q_tokenizer_error(error, size, "invalid Q tokenizer arguments");
    FILE *f = fopen(path, "rb");
    if (!f) return q_tokenizer_error(error, size, "cannot open Q merge file");
    q_tokenizer next = {0};
    int failed = q_tokenizer_read_u32(f, &next.merge_count) ||
                 next.merge_count > Q_TOKENIZER_MAX_MERGES;
    next.vocab_size = 256 + next.merge_count;
    for (uint32_t i = 0; !failed && i < next.merge_count; ++i) {
        uint32_t id;
        failed = q_tokenizer_read_u32(f, &next.pairs[i][0]) ||
                 q_tokenizer_read_u32(f, &next.pairs[i][1]) ||
                 q_tokenizer_read_u32(f, &id) || id != 256 + i;
    }
    if (!failed && (fgetc(f) != EOF || ferror(f))) failed = 1;
    if (fclose(f)) failed = 1;
    if (failed) return q_tokenizer_error(error, size, "malformed Q merge file");
    if (q_tokenizer_validate(&next, error, size)) return -1;
    *out = next;
    return 0;
}

static int q_tokenizer_encode(const q_tokenizer *q, const unsigned char *bytes,
                              size_t length, uint32_t **ids, size_t *count,
                              char *error, size_t size) {
    if (!ids || !count || (!bytes && length) || length > SIZE_MAX / sizeof(uint32_t))
        return q_tokenizer_error(error, size, "invalid Q encoding arguments");
    *ids = NULL; *count = 0;
    if (q_tokenizer_validate(q, error, size)) return -1;
    if (!length) return 0;
    uint32_t *tokens = malloc(length * sizeof(*tokens));
    if (!tokens) return q_tokenizer_error(error, size, "Q token allocation failed");
    for (size_t i = 0; i < length; ++i) tokens[i] = bytes[i];
    size_t n = length;
    /* Q applies each merge once over the whole current sequence, left to right.
     * A match consumes both inputs; overlapping pairs never merge together. */
    for (uint32_t m = 0; m < q->merge_count; ++m) {
        uint32_t a = q->pairs[m][0], b = q->pairs[m][1];
        size_t written = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i + 1 < n && tokens[i] == a && tokens[i + 1] == b) {
                tokens[written++] = 256 + m;
                ++i;
            } else tokens[written++] = tokens[i];
        }
        n = written;
    }
    *ids = tokens; *count = n;
    return 0;
}

static int q_tokenizer_decode(const q_tokenizer *q, uint32_t id,
                              unsigned char out[Q_TOKENIZER_TOKEN_BYTES], size_t *length,
                              char *error, size_t size) {
    if (!out || !length) return q_tokenizer_error(error, size, "invalid Q decoding arguments");
    *length = 0;
    if (q_tokenizer_validate(q, error, size)) return -1;
    if (id >= q->vocab_size) return q_tokenizer_error(error, size, "Q token out of range");
    uint32_t stack[Q_TOKENIZER_TOKEN_BYTES];
    size_t pending = 1, written = 0;
    stack[0] = id;
    while (pending) {
        uint32_t token = stack[--pending];
        if (token < 256) out[written++] = (unsigned char)token;
        else {
            if (pending + 2 > Q_TOKENIZER_TOKEN_BYTES)
                return q_tokenizer_error(error, size, "Q decoding stack overflow");
            stack[pending++] = q->pairs[token - 256][1];
            stack[pending++] = q->pairs[token - 256][0];
        }
    }
    *length = written;
    return 0;
}

static int q_tokenizer_emit(const q_tokenizer *q, uint32_t id, FILE *output,
                            char *error, size_t size) {
    unsigned char bytes[Q_TOKENIZER_TOKEN_BYTES];
    size_t count;
    if (!output) return q_tokenizer_error(error, size, "missing Q output stream");
    if (q_tokenizer_decode(q, id, bytes, &count, error, size)) return -1;
    if (fwrite(bytes, 1, count, output) != count)
        return q_tokenizer_error(error, size, "Q output write failed");
    return 0;
}

#endif
