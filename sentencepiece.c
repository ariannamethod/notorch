/* SentencePiece Unigram inference, owned immutable models and per-call scratch.
 *
 * Normalization and Viterbi semantics follow SentencePiece v0.2.2,
 * Copyright 2016 Google Inc., Apache-2.0. Darts unit decoding follows
 * Darts-clone, Copyright 2008-2011 Susumu Yata, BSD-3-Clause.
 * These algorithms are adapted to bounded C ownership and checked parsing.
 * See THIRD_PARTY_NOTICES.md and LICENSES/ for the full notices.
 */
#include "sentencepiece.h"
#include "sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "SentencePiece scores require IEEE binary32");

#ifndef NT_SPM_MALLOC
#define NT_SPM_MALLOC malloc
#define NT_SPM_CALLOC calloc
#define NT_SPM_FREE free
#endif

typedef struct { const unsigned char *p; size_t n; } sp_view;
typedef struct {
    unsigned number, wire;
    uint64_t integer;
    sp_view data;
} sp_field;
typedef struct { sp_view text; float score; int type; } sp_token;
typedef struct { int child, next, id; unsigned char byte; } sp_node;

struct nt_spm_model {
    unsigned char *data;
    size_t data_bytes;
    char identity[65];
    sp_token *tokens;
    int n_tokens, unk_id;
    float min_score;
    sp_node *nodes;
    size_t n_nodes, node_cap;
    sp_view charsmap, replacements;
    size_t map_units;
    int add_dummy, remove_extra, escape_ws, suffix;
};

static void sp_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
}

static uint32_t sp_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int sp_varint(sp_view *v, uint64_t *out) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 70; shift += 7) {
        if (!v->n) return -1;
        unsigned byte = *v->p++;
        v->n--;
        if (shift == 63 && byte > 1) return -1;
        value |= (uint64_t)(byte & 127) << shift;
        if (!(byte & 128)) { *out = value; return 0; }
    }
    return -1;
}

/* Protobuf scalar/message fields: unknown fields are skipped, groups rejected.
 * Every length is checked against its containing message before pointer math. */
static int sp_next(sp_view *v, sp_field *f) {
    uint64_t key, n;
    memset(f, 0, sizeof(*f));
    if (sp_varint(v, &key) || key >> 3 == 0 || key >> 3 > 0x1fffffff) return -1;
    f->number = (unsigned)(key >> 3);
    f->wire = (unsigned)(key & 7);
    if (f->wire == 0) return sp_varint(v, &f->integer);
    if (f->wire == 2) {
        if (sp_varint(v, &n) || n > v->n) return -1;
    } else if (f->wire == 1) n = 8;
    else if (f->wire == 5) n = 4;
    else return -1;
    if (n > v->n) return -1;
    f->data.p = v->p; f->data.n = (size_t)n;
    if (f->wire == 5) f->integer = sp_u32(v->p);
    v->p += (size_t)n; v->n -= (size_t)n;
    return 0;
}

/* Returns the valid scalar width, or zero for malformed/truncated UTF-8. */
static size_t sp_utf8(const unsigned char *s, size_t n) {
    if (!n) return 0;
    unsigned c = s[0];
    if (c < 0x80) return 1;
    if (c < 0xc2 || c > 0xf4) return 0;
    size_t width = c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    if (n < width) return 0;
    for (size_t i = 1; i < width; i++) if ((s[i] & 0xc0) != 0x80) return 0;
    if ((c == 0xe0 && s[1] < 0xa0) || (c == 0xed && s[1] >= 0xa0) ||
        (c == 0xf0 && s[1] < 0x90) || (c == 0xf4 && s[1] >= 0x90)) return 0;
    return width;
}

static int sp_valid_text(sp_view v) {
    while (v.n) {
        size_t n = sp_utf8(v.p, v.n);
        if (!n || !*v.p) return 0;
        v.p += n; v.n -= n;
    }
    return 1;
}

static int sp_parse_token(sp_view v, sp_token *t) {
    t->type = 1;
    while (v.n) {
        sp_field f;
        if (sp_next(&v, &f)) return -1;
        if (f.number == 1) {
            if (f.wire != 2) return -1;
            t->text = f.data;
        } else if (f.number == 2) {
            if (f.wire != 5) return -1;
            uint32_t bits = (uint32_t)f.integer;
            memcpy(&t->score, &bits, sizeof(bits));
        } else if (f.number == 3) {
            if (f.wire != 0 || f.integer < 1 || f.integer > 6) return -1;
            t->type = (int)f.integer;
        }
    }
    return t->text.n && t->text.n <= NT_SPM_MAX_PIECE_BYTES &&
        sp_valid_text(t->text) && isfinite(t->score) ? 0 : -1;
}

static int sp_parse_trainer(sp_view v, nt_spm_model *m) {
    while (v.n) {
        sp_field f;
        if (sp_next(&v, &f)) return -1;
        if (f.number == 3) {             /* ModelType UNIGRAM */
            if (f.wire != 0 || f.integer != 1) return -2;
        } else if (f.number == 35) {     /* byte_fallback */
            if (f.wire != 0 || f.integer != 0) return -2;
        } else if (f.number == 24) {     /* treat_whitespace_as_suffix */
            if (f.wire != 0 || f.integer > 1) return -1;
            m->suffix = (int)f.integer;
        }
    }
    return 0;
}

static int sp_parse_normalizer(sp_view v, nt_spm_model *m) {
    while (v.n) {
        sp_field f;
        if (sp_next(&v, &f)) return -1;
        if (f.number == 2) {
            if (f.wire != 2) return -1;
            m->charsmap = f.data;
        } else if (f.number >= 3 && f.number <= 5) {
            if (f.wire != 0 || f.integer > 1) return -1;
            if (f.number == 3) m->add_dummy = (int)f.integer;
            if (f.number == 4) m->remove_extra = (int)f.integer;
            if (f.number == 5) m->escape_ws = (int)f.integer;
        }
    }
    return 0;
}

static uint32_t sp_offset(uint32_t unit) {
    return (unit >> 10) << ((unit & 512) >> 6);
}

static int sp_validate_map(nt_spm_model *m) {
    if (!m->charsmap.n) return 0;       /* identity normalization */
    if (m->charsmap.n <= 4) return -1;
    size_t trie_bytes = sp_u32(m->charsmap.p);
    if (trie_bytes < 1024 || (trie_bytes & 1023) || trie_bytes >= m->charsmap.n - 4) return -1;
    m->replacements.p = m->charsmap.p + 4 + trie_bytes;
    m->replacements.n = m->charsmap.n - 4 - trie_bytes;
    m->charsmap.p += 4; m->charsmap.n = trie_bytes;
    m->map_units = trie_bytes / 4;
    if (m->replacements.p[m->replacements.n - 1] != 0) return -1;
    /* Every replacement is UTF-8; every leaf points to a scalar boundary. */
    size_t p = 0;
    while (p < m->replacements.n) {
        if (!m->replacements.p[p]) { p++; continue; }
        size_t n = sp_utf8(m->replacements.p + p, m->replacements.n - p);
        if (!n) return -1;
        p += n;
    }
    uint32_t root = sp_u32(m->charsmap.p);
    if ((root & 0x800000ffU) || (root & 256) || !sp_offset(root)) return -1;
    for (size_t i = 0; i < m->map_units; i++) {
        uint32_t u = sp_u32(m->charsmap.p + i * 4);
        if (!(u & 0x80000000U)) {
            size_t base = i ^ sp_offset(u);
            if ((base | 255) >= m->map_units) return -1;
            if (u & 256) {
                uint32_t leaf = sp_u32(m->charsmap.p + base * 4);
                if (!(leaf & 0x80000000U)) return -1;
            }
        } else {
            size_t value = u & 0x7fffffffU;
            if (value >= m->replacements.n || (m->replacements.p[value] & 0xc0) == 0x80) return -1;
        }
    }
    return 0;
}

static int sp_child(const nt_spm_model *m, int node, unsigned char byte) {
    for (int c = m->nodes[node].child; c >= 0; c = m->nodes[c].next)
        if (m->nodes[c].byte == byte) return c;
    return -1;
}

static int sp_insert(nt_spm_model *m, int id) {
    const sp_token *t = m->tokens + id;
    int node = 0;
    for (size_t i = 0; i < t->text.n; i++) {
        int child = sp_child(m, node, t->text.p[i]);
        if (child < 0) {
            if (m->n_nodes == m->node_cap) return -1;
            child = (int)m->n_nodes++;
            m->nodes[child].byte = t->text.p[i];
            m->nodes[child].child = -1; m->nodes[child].id = -1;
            m->nodes[child].next = m->nodes[node].child;
            m->nodes[node].child = child;
        }
        node = child;
    }
    if (m->nodes[node].id >= 0) return -1;
    m->nodes[node].id = id;
    return 0;
}

static int sp_token_compare(const void *a, const void *b) {
    const sp_token *x = *(const sp_token *const *)a;
    const sp_token *y = *(const sp_token *const *)b;
    size_t n = x->text.n < y->text.n ? x->text.n : y->text.n;
    int cmp = memcmp(x->text.p, y->text.p, n);
    if (cmp) return cmp;
    return x->text.n < y->text.n ? -1 : x->text.n > y->text.n;
}

void nt_spm_free(nt_spm_model *m) {
    if (!m) return;
    NT_SPM_FREE(m->nodes); NT_SPM_FREE(m->tokens);
    NT_SPM_FREE(m->data); NT_SPM_FREE(m);
}

/* Adopts data on every path, including malformed input. */
static nt_spm_model *sp_load_owned(unsigned char *data, size_t bytes,
                                  char *error, size_t error_cap) {
    const char *reason = "malformed SentencePiece ModelProto";
    nt_spm_model *m = NT_SPM_CALLOC(1, sizeof(*m));
    if (!m) { NT_SPM_FREE(data); sp_error(error, error_cap, "out of memory loading tokenizer"); return NULL; }
    m->data = data; m->data_bytes = bytes; m->unk_id = -1;
    m->add_dummy = m->remove_extra = m->escape_ws = 1;
    m->min_score = FLT_MAX;
    sp_view v = {data, bytes};
    while (v.n) {
        sp_field f;
        if (sp_next(&v, &f)) goto fail;
        if (f.number == 1) {
            if (f.wire != 2 || m->n_tokens == NT_SPM_MAX_VOCAB) goto fail;
            m->n_tokens++;
        }
    }
    if (!m->n_tokens) goto fail;
    m->tokens = NT_SPM_CALLOC((size_t)m->n_tokens, sizeof(*m->tokens));
    if (!m->tokens) { reason = "out of memory loading tokenizer pieces"; goto fail; }
    v.p = data; v.n = bytes;
    int id = 0, n_normal = 0;
    m->node_cap = 1;
    while (v.n) {
        sp_field f;
        if (sp_next(&v, &f)) goto fail;
        if (f.number == 1) {
            sp_token *t = m->tokens + id;
            if (sp_parse_token(f.data, t)) goto fail;
            if (t->type == 6) { reason = "SentencePiece byte fallback is unsupported"; goto fail; }
            if (t->type == 2) {
                if (m->unk_id >= 0) goto fail;
                m->unk_id = id;
            }
            if (t->type == 1) {
                n_normal++;
                if (t->score < m->min_score) m->min_score = t->score;
            }
            if (t->type == 1 || t->type == 4 || t->type == 5) {
                if (t->text.n > (size_t)INT_MAX - m->node_cap) goto fail;
                m->node_cap += t->text.n;
            }
            id++;
        } else if (f.number == 2) {
            if (f.wire != 2) goto fail;
            int rc = sp_parse_trainer(f.data, m);
            if (rc == -2) { reason = "tokenizer requires UNIGRAM without byte fallback"; goto fail; }
            if (rc) goto fail;
        } else if (f.number == 3) {
            if (f.wire != 2 || sp_parse_normalizer(f.data, m)) goto fail;
        } else if (f.number == 5) {
            /* Encoding ignores denormalization; no decode API is exposed. */
            if (f.wire != 2) goto fail;
        }
    }
    if (!n_normal) { reason = "tokenizer requires at least one NORMAL piece"; goto fail; }
    if (m->unk_id < 0 || !isfinite(m->min_score - 10.0f)) goto fail;
    if (sp_validate_map(m)) { reason = "malformed SentencePiece normalization charsmap"; goto fail; }
    /* Reserved and ordinary vocabulary maps have separate duplicate domains. */
    const sp_token **order = NT_SPM_MALLOC((size_t)m->n_tokens * sizeof(*order));
    if (!order) { reason = "out of memory validating tokenizer pieces"; goto fail; }
    for (int i = 0; i < m->n_tokens; i++) order[i] = m->tokens + i;
    qsort(order, (size_t)m->n_tokens, sizeof(*order), sp_token_compare);
    int duplicate = 0;
    for (int i = 0; i < m->n_tokens;) {
        int j = i + 1, ordinary = 0, reserved = 0;
        while (j < m->n_tokens && sp_token_compare(order + i, order + j) == 0) j++;
        for (int k = i; k < j; k++) {
            int type = order[k]->type;
            if (type == 1 || type == 4 || type == 5) ordinary++; else reserved++;
        }
        if (ordinary > 1 || reserved > 1) duplicate = 1;
        i = j;
    }
    NT_SPM_FREE(order);
    if (duplicate) { reason = "duplicate SentencePiece vocabulary entry"; goto fail; }
    if (m->node_cap > SIZE_MAX / sizeof(*m->nodes)) goto fail;
    m->nodes = NT_SPM_MALLOC(m->node_cap * sizeof(*m->nodes));
    if (!m->nodes) { reason = "out of memory indexing tokenizer pieces"; goto fail; }
    m->n_nodes = 1;
    m->nodes[0].child = m->nodes[0].next = m->nodes[0].id = -1;
    for (int i = 0; i < m->n_tokens; i++) {
        int type = m->tokens[i].type;
        if ((type == 1 || type == 4 || type == 5) && sp_insert(m, i)) goto fail;
    }
    unsigned char digest[32];
    if (nt_sha256(m->data, m->data_bytes, digest)) goto fail;
    static const char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < 32; i++) {
        m->identity[i * 2] = hex[digest[i] >> 4];
        m->identity[i * 2 + 1] = hex[digest[i] & 15];
    }
    m->identity[64] = 0;
    sp_error(error, error_cap, "");
    return m;
fail:
    sp_error(error, error_cap, reason);
    nt_spm_free(m);
    return NULL;
}

nt_spm_model *nt_spm_load_memory(const void *data, size_t size,
                                 char *error, size_t error_cap) {
    if (!data || !size || size > NT_SPM_MAX_MODEL_BYTES) {
        sp_error(error, error_cap, "tokenizer model must contain 1 byte..64 MiB"); return NULL;
    }
    unsigned char *copy = NT_SPM_MALLOC(size);
    if (!copy) { sp_error(error, error_cap, "out of memory copying tokenizer model"); return NULL; }
    memcpy(copy, data, size);
    return sp_load_owned(copy, size, error, error_cap);
}

nt_spm_model *nt_spm_load(const char *path, char *error, size_t error_cap) {
    if (!path || !*path) { sp_error(error, error_cap, "tokenizer path is empty"); return NULL; }
    FILE *file = fopen(path, "rb");
    if (!file) { sp_error(error, error_cap, "cannot open tokenizer model"); return NULL; }
    unsigned char *data = NULL;
    long length;
    const char *reason = "cannot read tokenizer model";
    if (fseek(file, 0, SEEK_END) || (length = ftell(file)) <= 0 ||
        (unsigned long)length > NT_SPM_MAX_MODEL_BYTES || fseek(file, 0, SEEK_SET)) goto fail;
    data = NT_SPM_MALLOC((size_t)length);
    if (!data) { reason = "out of memory reading tokenizer model"; goto fail; }
    if (fread(data, 1, (size_t)length, file) != (size_t)length) goto fail;
    if (fgetc(file) != EOF || ferror(file)) goto fail;
    if (fclose(file)) { file = NULL; goto fail; }
    return sp_load_owned(data, (size_t)length, error, error_cap);
fail:
    if (file) fclose(file);
    NT_SPM_FREE(data);
    sp_error(error, error_cap, reason);
    return NULL;
}

int nt_spm_n_vocab(const nt_spm_model *model) { return model ? model->n_tokens : 0; }
const char *nt_spm_identity(const nt_spm_model *model) { return model ? model->identity : NULL; }

/* USER_DEFINED strings bypass the compiled normalizer, longest match first. */
static sp_view sp_prefix(const nt_spm_model *m, sp_view input, size_t *consumed) {
    int node = 0;
    size_t user_length = 0;
    for (size_t i = 0; i < input.n; i++) {
        node = sp_child(m, node, input.p[i]);
        if (node < 0) break;
        int id = m->nodes[node].id;
        if (id >= 0 && m->tokens[id].type == 4) user_length = i + 1;
    }
    if (user_length) { *consumed = user_length; return (sp_view){input.p, user_length}; }
    size_t longest = 0, value = 0;
    if (m->map_units) {
        size_t pos = sp_offset(sp_u32(m->charsmap.p));
        for (size_t i = 0; i < input.n; i++) {
            pos ^= input.p[i];
            uint32_t u = sp_u32(m->charsmap.p + pos * 4);
            if ((u & 0x800000ffU) != input.p[i]) break;
            pos ^= sp_offset(u);
            if (u & 256) {
                longest = i + 1;
                value = sp_u32(m->charsmap.p + pos * 4) & 0x7fffffffU;
            }
        }
    }
    if (longest) {
        const unsigned char *p = m->replacements.p + value;
        *consumed = longest;
        return (sp_view){p, strlen((const char *)p)};
    }
    size_t n = sp_utf8(input.p, input.n);
    if (n) { *consumed = n; return (sp_view){input.p, n}; }
    *consumed = 1;
    return (sp_view){(const unsigned char *)"\xef\xbf\xbd", 3};
}

/* Two normalization passes share this exact emitter. It tracks a trailing
 * run of the whitespace symbol even when that symbol occurs literally in input. */
typedef struct {
    char *data;
    size_t size, trailing, partial, capacity;
    const unsigned char *space;
    size_t space_size;
    int trim;
} sp_output;

static int sp_emit(sp_output *o, const unsigned char *p, size_t n) {
    if (n > SIZE_MAX - o->size) return -1;
    for (size_t i = 0; i < n; i++) {
        /* The first pass knows the final size. The second writes only that
         * prefix; a larger trailing run is removed without allocating it. */
        if (o->data && o->size < o->capacity) o->data[o->size] = (char)p[i];
        o->size++;
        if (p[i] == o->space[o->partial]) {
            o->partial++;
            if (o->partial == o->space_size) { o->trailing += o->space_size; o->partial = 0; }
        } else {
            o->trailing = 0;
            o->partial = p[i] == o->space[0] ? 1 : 0;
            if (o->partial == o->space_size) { o->trailing = o->space_size; o->partial = 0; }
        }
        size_t retained = o->trim ? o->size - o->trailing - o->partial : o->size;
        if (retained > NT_SPM_MAX_TEXT_BYTES) return -1;
    }
    return 0;
}

static int sp_normalize(const nt_spm_model *m, sp_view input, char *out,
                         size_t capacity, size_t *length) {
    sp_output o = {0};
    o.data = out; o.capacity = capacity; o.trim = m->remove_extra;
    o.space = (const unsigned char *)(m->escape_ws ? "\xe2\x96\x81" : " ");
    o.space_size = m->escape_ws ? 3 : 1;
    if (m->remove_extra) while (input.n) {
        size_t consumed;
        sp_view p = sp_prefix(m, input, &consumed);
        if (p.n != 1 || p.p[0] != ' ') break;
        input.p += consumed; input.n -= consumed;
    }
    if (!input.n) {
        if (out) out[0] = 0;
        *length = 0;
        return 0;
    }
    if (!m->suffix && m->add_dummy && sp_emit(&o, o.space, o.space_size)) return -1;
    int prev_space = m->remove_extra;
    while (input.n) {
        size_t consumed;
        sp_view p = sp_prefix(m, input, &consumed);
        while (prev_space && p.n && *p.p == ' ') { p.p++; p.n--; }
        if (p.n) {
            for (size_t i = 0; i < p.n; i++) {
                if (p.p[i] == ' ') { if (sp_emit(&o, o.space, o.space_size)) return -1; }
                else if (sp_emit(&o, p.p + i, 1)) return -1;
            }
            prev_space = p.p[p.n - 1] == ' ';
        }
        input.p += consumed; input.n -= consumed;
        if (!m->remove_extra) prev_space = 0;
    }
    if (m->remove_extra && !o.partial) o.size -= o.trailing;
    if (o.size > NT_SPM_MAX_TEXT_BYTES) return -1;
    o.trim = 0;
    if (m->suffix && m->add_dummy && sp_emit(&o, o.space, o.space_size)) return -1;
    if (out) out[o.size] = 0;
    *length = o.size;
    return 0;
}

void nt_spm_result_free(nt_spm_result *result) {
    if (!result) return;
    NT_SPM_FREE(result->normalized); NT_SPM_FREE(result->pieces);
    memset(result, 0, sizeof(*result));
}

typedef struct { int id, start; float score; } sp_path;

int nt_spm_encode(const nt_spm_model *m, const char *text, size_t bytes,
                  nt_spm_result *out, char *error, size_t error_cap) {
    const char *reason = "tokenizer input or normalized output exceeds 1 MiB";
    if (!m || !out || (!text && bytes)) { sp_error(error, error_cap, "invalid tokenizer encode arguments"); return -1; }
    if (bytes > NT_SPM_MAX_TEXT_BYTES) { sp_error(error, error_cap, reason); return -1; }
    if (bytes && memchr(text, 0, bytes)) { sp_error(error, error_cap, "tokenizer text contains NUL"); return -1; }
    nt_spm_result r = {0};
    sp_path *path = NULL;
    sp_view input = {(const unsigned char *)text, bytes};
    if (sp_normalize(m, input, NULL, 0, &r.normalized_bytes)) goto fail;
    r.normalized = NT_SPM_MALLOC(r.normalized_bytes + 1);
    if (!r.normalized) { reason = "out of memory normalizing tokenizer input"; goto fail; }
    if (sp_normalize(m, input, r.normalized, r.normalized_bytes, &r.normalized_bytes)) goto fail;
    if (!r.normalized_bytes) { *out = r; sp_error(error, error_cap, ""); return 0; }
    path = NT_SPM_CALLOC(r.normalized_bytes + 1, sizeof(*path));
    if (!path) { reason = "out of memory finding tokenizer path"; goto fail; }
    for (size_t i = 0; i <= r.normalized_bytes; i++) path[i].start = -1;
    float unknown_score = m->min_score - 10.0f;
    size_t frontier = 0;
    for (size_t start = 0; start < r.normalized_bytes;) {
        float before = path[start].score;
        if (before < -100000.0f || before > 100000.0f) {
            for (size_t i = start; i <= frontier; i++)
                if (i == start || path[i].start != -1) path[i].score -= before;
            before = 0;
        }
        size_t width = sp_utf8((const unsigned char *)r.normalized + start, r.normalized_bytes - start);
        if (!width) { reason = "normalizer emitted invalid UTF-8"; goto fail; }
        int node = 0, has_single = 0;
        for (size_t end = start; end < r.normalized_bytes;) {
            node = sp_child(m, node, (unsigned char)r.normalized[end]);
            if (node < 0) break;
            end++;
            int id = m->nodes[node].id;
            if (id < 0 || m->tokens[id].type == 5) continue;
            size_t length = end - start;
            float score = m->tokens[id].type == 4 ? (float)(0.1 * (length - 1)) : m->tokens[id].score;
            float candidate = score + before;
            if (!isfinite(candidate)) { reason = "tokenizer path score overflow"; goto fail; }
            if (path[end].start == -1 || candidate > path[end].score) {
                path[end].start = (int)start; path[end].id = id; path[end].score = candidate;
            }
            if (end > frontier) frontier = end;
            if (length == width) has_single = 1;
        }
        if (!has_single) {
            size_t end = start + width;
            float candidate = unknown_score + before;
            if (!isfinite(candidate)) { reason = "tokenizer unknown score overflow"; goto fail; }
            if (path[end].start == -1 || candidate > path[end].score) {
                path[end].start = (int)start; path[end].id = m->unk_id; path[end].score = candidate;
            }
            if (end > frontier) frontier = end;
        }
        start += width;
    }
    size_t count = 0;
    for (size_t end = r.normalized_bytes; end;) {
        if (path[end].start < 0 || (size_t)path[end].start >= end) { reason = "invalid tokenizer path"; goto fail; }
        end = (size_t)path[end].start; count++;
    }
    r.pieces = NT_SPM_MALLOC(count * sizeof(*r.pieces));
    if (!r.pieces) { reason = "out of memory publishing tokenizer pieces"; goto fail; }
    size_t index = count;
    for (size_t end = r.normalized_bytes; end;) {
        size_t start = (size_t)path[end].start;
        r.pieces[--index] = (nt_spm_piece){path[end].id, start, end - start};
        end = start;
    }
    for (size_t i = 0; i < count; i++) {
        if (r.count && r.pieces[r.count - 1].id == m->unk_id && r.pieces[i].id == m->unk_id)
            r.pieces[r.count - 1].length += r.pieces[i].length;
        else r.pieces[r.count++] = r.pieces[i];
    }
    NT_SPM_FREE(path);
    *out = r;
    sp_error(error, error_cap, "");
    return 0;
fail:
    NT_SPM_FREE(path); nt_spm_result_free(&r);
    sp_error(error, error_cap, reason);
    return -1;
}
