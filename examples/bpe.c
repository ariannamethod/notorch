/* bpe.c — byte-level BPE (GPT-2 / Tekken style) over a GGUF tokenizer. See bpe.h. */
#include "bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ── string -> int open-addressing hashmap ─────────────────────────────────── */
typedef struct { char **keys; int *vals; int cap; int n; } smap;

static unsigned long fnv1a(const char *s) {
    unsigned long h = 1469598103934665603UL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; }
    return h;
}
static void smap_init(smap *m, int cap) {
    if (cap < 8) cap = 8;
    m->cap = cap; m->n = 0;
    m->keys = (char**)calloc(cap, sizeof(char*));
    m->vals = (int*)calloc(cap, sizeof(int));
}
static void smap_put(smap *m, const char *k, int v) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], k) == 0) { m->vals[h] = v; return; }
        h = (h + 1) % m->cap;
    }
    m->keys[h] = strdup(k); m->vals[h] = v; m->n++;
}
static int smap_get(const smap *m, const char *k) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], k) == 0) return m->vals[h];
        h = (h + 1) % m->cap;
    }
    return -1;
}
static void smap_free(smap *m) {
    for (int i = 0; i < m->cap; i++) free(m->keys[i]);
    free(m->keys); free(m->vals);
}

/* ── GPT-2 bytes<->unicode ─────────────────────────────────────────────────── */
static void build_byte_table(int cp[256], int cp2byte[512]) {
    for (int i = 0; i < 512; i++) cp2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp[b] = printable ? b : (256 + n);
        if (!printable) n++;
    }
    for (int b = 0; b < 256; b++) if (cp[b] < 512) cp2byte[cp[b]] = b;
}
static int utf8_enc(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
}
/* Four-byte sequences are not an edge case, they are every emoji. Without this arm a lead
 * byte of 0xF0 was taken as one character and the three continuation bytes as three more,
 * none of which is a legal token, so a vocabulary that has 🍦 whole fell back to four byte
 * tokens for it — correct text out, wrong ids, and the ids are the model's input. */
static int utf8_dec(const char *s, int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 0x6) { *cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F); return 2; }
    if ((c >> 4) == 0xE) { *cp = ((c & 0xF) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F); return 3; }
    if ((c >> 3) == 0x1E) {
        *cp = ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12)
            | (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
        return 4;
    }
    *cp = c; return 1;
}

/* ── tokenizer ─────────────────────────────────────────────────────────────── */
struct bpe_tokenizer {
    char **tokens; int n_tokens;   /* id -> token string */
    smap vocab;                    /* token string -> id */
    smap merges;                   /* "A B" -> rank (byte-level only) */
    int byte_cp[256];
    int cp2byte[512];
    /* SentencePiece: no merge list, a score per token, space written U+2581,
     * and every byte available as a "<0xHH>" token to fall back on. */
    int spm;
    float *scores;
    int byte_id[256];              /* byte -> id of "<0xHH>", or -1 */
    /* Gemma 4 is the third shape and it is not deducible from the file's contents: it
     * carries BOTH a merge list and a score per token, so the "merges means byte-level,
     * scores means SentencePiece" rule sends it to the wrong one. It is SPM-style BPE —
     * spaces written U+2581 and byte fallback like SentencePiece, merges by rank on raw
     * UTF-8 like BPE, and no word-level pre-split at all, only a break at newlines. The
     * only signal is the name in tokenizer.ggml.model, which is what llama.cpp reads too. */
    int spm_bpe;
    /* Tokens the file marks USER_DEFINED. They are stored as literal text rather than in the
     * byte-level encoding — a run of four real spaces, not four 'Ġ' — so the merge path can
     * never produce them, and the reference matches them against the raw text before it runs.
     * Skipping this step is quiet on prose and wrong on indented code: OLMoE's vocabulary
     * carries twenty-five of these and they are all whitespace runs. */
    int *added_id;      /* ids, longest string first */
    int n_added;
    int add_bos, bos_id;
};

/* U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's space. */
#define SPM_SPACE "\xE2\x96\x81"

/* The value of a "<0xHH>" token, or -1 for anything else. */
static int byte_token_value(const char *s) {
    if (!s || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>' || s[6]) return -1;
    int hi = -1, lo = -1;
    for (int k = 0; k < 2; k++) {
        char c = s[3 + k];
        int v = (c >= '0' && c <= '9') ? c - '0'
              : (c >= 'A' && c <= 'F') ? c - 'A' + 10
              : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : -1;
        if (v < 0) return -1;
        if (k == 0) hi = v; else lo = v;
    }
    return hi * 16 + lo;
}

bpe_tokenizer *bpe_load(const char *path) {
    int nt = 0;
    char **toks = gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt);
    if (!toks || nt <= 0) return NULL;
    int nm = 0;
    char **mg = gguf_read_str_array(path, "tokenizer.ggml.merges", &nm);

    bpe_tokenizer *t = (bpe_tokenizer*)calloc(1, sizeof(*t));
    t->tokens = toks; t->n_tokens = nt;
    build_byte_table(t->byte_cp, t->cp2byte);
    smap_init(&t->vocab, nt * 2);
    for (int i = 0; i < nt; i++) if (toks[i]) smap_put(&t->vocab, toks[i], i);
    smap_init(&t->merges, (nm > 0 ? nm : 1) * 2);
    for (int i = 0; i < nm; i++) if (mg[i]) smap_put(&t->merges, mg[i], i); /* rank = line index */
    for (int i = 0; i < nm; i++) free(mg[i]);
    free(mg);

    /* Which scheme this file carries is decided by what it gives us to work
     * with, not by the name it goes under: a merge list means byte-level BPE
     * and merge ranks; a score per token and no merges means SentencePiece,
     * where the same two operations do not exist. Getting this wrong is quiet
     * — byte-level encode over a SentencePiece vocabulary finds no token for a
     * space and used to drop it, which reads as text with the spaces missing. */
    /* USER_DEFINED is type 4 in tokenizer.ggml.token_type. CONTROL (3) is deliberately left
     * out: the reference only splits on those when asked to parse specials, and a prompt that
     * happens to spell one should not become that token by accident. */
    int ntt = 0;
    int32_t *ttypes = gguf_read_i32_array(path, "tokenizer.ggml.token_type", &ntt);
    if (ttypes && ntt == nt) {
        for (int i = 0; i < nt; i++)
            if (ttypes[i] == 4 && toks[i] && toks[i][0]) t->n_added++;
        if (t->n_added) {
            t->added_id = (int*)calloc((size_t)t->n_added, sizeof(int));
            int k = 0;
            for (int i = 0; i < nt; i++)
                if (ttypes[i] == 4 && toks[i] && toks[i][0]) t->added_id[k++] = i;
            /* Longest first, so the scan takes the longest match at each position the way a
             * longest-match scan must; insertion sort over a couple of dozen entries. */
            for (int a = 1; a < t->n_added; a++) {
                int v = t->added_id[a];
                size_t vl = strlen(toks[v]);
                int b = a - 1;
                while (b >= 0 && strlen(toks[t->added_id[b]]) < vl) {
                    t->added_id[b + 1] = t->added_id[b];
                    b--;
                }
                t->added_id[b + 1] = v;
            }
        }
    }
    free(ttypes);

    int ns = 0;
    float *scores = gguf_read_f32_array(path, "tokenizer.ggml.scores", &ns);
    if (nm <= 0 && scores && ns == nt) {
        t->spm = 1;
        t->scores = scores;
        for (int b = 0; b < 256; b++) {
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", b);
            t->byte_id[b] = smap_get(&t->vocab, name);
        }
    } else {
        free(scores);
        for (int b = 0; b < 256; b++) t->byte_id[b] = -1;
    }

    /* The name settles what the contents cannot. Gemma 4 hands us merges and scores both,
     * and it wants neither of the two paths above. */
    char model_name[64] = {0};
    if (gguf_read_str_kv(path, "tokenizer.ggml.model", model_name, sizeof(model_name)) == 0 &&
        strcmp(model_name, "gemma4") == 0 && nm > 0) {
        t->spm = 0;
        t->spm_bpe = 1;
        for (int b = 0; b < 256; b++) {
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", b);
            t->byte_id[b] = smap_get(&t->vocab, name);
        }
        /* This family always opens with <bos>, and the file says so rather than the code
         * assuming it. Getting it wrong costs more than a token: Gemma without its opening
         * marker answers a different question. */
        uint64_t v = 0;
        t->bos_id = (gguf_read_uint_kv(path, "tokenizer.ggml.bos_token_id", &v) == 0) ? (int)v : 2;
        t->add_bos = (gguf_read_uint_kv(path, "tokenizer.ggml.add_bos_token", &v) == 0) ? (int)v : 1;
    }
    return t;
}

void bpe_free(bpe_tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_tokens; i++) free(t->tokens[i]);
    free(t->tokens);
    smap_free(&t->vocab); smap_free(&t->merges);
    free(t->scores);
    free(t->added_id);
    free(t);
}

int bpe_n_vocab(const bpe_tokenizer *t) { return t ? t->n_tokens : 0; }
int bpe_token_id(const bpe_tokenizer *t, const char *token) { return t ? smap_get(&t->vocab, token) : -1; }

/* A symbol that no token covers and no byte token can carry. Said once per
 * process, because the alternative — the old behaviour — was saying nothing
 * and returning a sentence with holes in it. */
static void warn_dropped(const char *what) {
    static int said = 0;
    if (said) return;
    said = 1;
    fprintf(stderr, "bpe: no token and no byte fallback for %s — input is being lost\n", what);
}

/* ── SentencePiece ──────────────────────────────────────────────────────────
 * Spaces become U+2581 and one is prepended, which is what these vocabularies
 * were built over: 22965 of nano_arianna's 32000 tokens begin with it. Symbols
 * start as UTF-8 characters and the adjacent pair with the highest score is
 * merged until no adjacent pair is a token, which is SentencePiece's rule and
 * not BPE's rank order.
 *
 * The pair scores are kept beside the symbols and only the two around a merge
 * are recomputed, so the vocabulary is consulted O(n) times for the whole
 * string rather than once per scan. Finding the best pair is still a walk, and
 * at the length of a chat line that walk is cheaper than a heap. */
static int spm_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    size_t L = strlen(text);
    char *buf = (char*)malloc((L + 1) * 3 + 1);
    if (!buf) return 0;
    size_t bl = 0;
    memcpy(buf + bl, SPM_SPACE, 3); bl += 3;            /* add_dummy_prefix */
    for (size_t i = 0; i < L; i++) {
        if (text[i] == ' ') { memcpy(buf + bl, SPM_SPACE, 3); bl += 3; }
        else buf[bl++] = text[i];
    }
    buf[bl] = 0;

    int *off = (int*)malloc((bl + 1) * sizeof(int));
    int *len = (int*)malloc((bl + 1) * sizeof(int));
    int *prv = (int*)malloc((bl + 1) * sizeof(int));
    int *nxt = (int*)malloc((bl + 1) * sizeof(int));
    float *ps = (float*)malloc((bl + 1) * sizeof(float));
    char *tmp = (char*)malloc(bl + 1);
    if (!off || !len || !prv || !nxt || !ps || !tmp) {
        free(buf); free(off); free(len); free(prv); free(nxt); free(ps); free(tmp);
        return 0;
    }

    int n = 0;
    for (size_t i = 0; i < bl; ) {
        unsigned char c = (unsigned char)buf[i];
        int adv = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        if (i + (size_t)adv > bl) adv = 1;
        off[n] = (int)i; len[n] = adv;
        prv[n] = n - 1; nxt[n] = -1;
        if (n > 0) nxt[n - 1] = n;
        n++; i += adv;
    }

    /* score of merging symbol i with the one after it, or "not a token" */
    #define PAIR(i) do { \
        int r_ = nxt[i]; \
        if (r_ < 0) { ps[i] = -1e30f; break; } \
        memcpy(tmp, buf + off[i], (size_t)(len[i] + len[r_])); \
        tmp[len[i] + len[r_]] = 0; \
        int id_ = smap_get(&t->vocab, tmp); \
        ps[i] = (id_ >= 0) ? t->scores[id_] : -1e30f; \
    } while (0)

    for (int i = 0; i < n; i++) PAIR(i);

    for (;;) {
        int bi = -1; float best = -1e30f;
        for (int i = 0; i >= 0; i = nxt[i]) if (ps[i] > best) { best = ps[i]; bi = i; }
        if (bi < 0) break;                       /* nothing adjacent is a token */
        int r = nxt[bi];
        len[bi] += len[r];
        nxt[bi] = nxt[r];
        if (nxt[r] >= 0) prv[nxt[r]] = bi;
        PAIR(bi);
        if (prv[bi] >= 0) PAIR(prv[bi]);
    }
    #undef PAIR

    int no = 0;
    for (int i = 0; i >= 0 && no < cap; i = nxt[i]) {
        memcpy(tmp, buf + off[i], (size_t)len[i]);
        tmp[len[i]] = 0;
        int id = smap_get(&t->vocab, tmp);
        if (id >= 0) { out[no++] = id; continue; }
        /* One character the vocabulary does not carry. Its bytes do. */
        for (int b = 0; b < len[i] && no < cap; b++) {
            int bid = t->byte_id[(unsigned char)buf[off[i] + b]];
            if (bid >= 0) out[no++] = bid;
            else warn_dropped(tmp);
        }
    }

    free(buf); free(off); free(len); free(prv); free(nxt); free(ps); free(tmp);
    return no;
}

/* Gemma 4: spaces become U+2581, the text is broken only where newlines are, and each piece
 * is merged by rank over its UTF-8 characters. No word-level pre-split, which is the part
 * that separates this from byte-level BPE — a merge is allowed to cross what other families
 * treat as a word boundary, and forbidding that is what produced nine tokens where the
 * reference produces five. */
static int spm_bpe_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    int no = 0;
    if (t->add_bos && no < cap) out[no++] = t->bos_id;

    /* Escape first: one pass, ' ' -> three bytes. */
    int L = (int)strlen(text);
    char *esc = (char*)malloc((size_t)L * 3 + 1);
    if (!esc) return no;
    int el = 0;
    for (int i = 0; i < L; i++) {
        if (text[i] == ' ') { memcpy(esc + el, SPM_SPACE, 3); el += 3; }
        else esc[el++] = text[i];
    }
    esc[el] = 0;

    int i = 0;
    while (i < el) {
        /* One chunk is a run of newlines or a run of everything else. */
        int nl = (esc[i] == '\n');
        int j = i;
        while (j < el && ((esc[j] == '\n') == nl)) j++;

        /* A pure run of newlines that the vocabulary already has stays whole — the merge
         * table cannot be consulted for it, and llama.cpp special-cases it the same way. */
        char whole[64];
        if (nl && j - i < (int)sizeof(whole)) {
            memcpy(whole, esc + i, (size_t)(j - i));
            whole[j - i] = 0;
            int id = smap_get(&t->vocab, whole);
            if (id >= 0) {
                if (no < cap) out[no++] = id;
                i = j;
                continue;
            }
        }

        /* Symbols start as UTF-8 characters, then merge by lowest rank. */
        int nsym = 0;
        for (int p = i; p < j; ) { int cp; p += utf8_dec(esc + p, &cp); nsym++; }
        char **sym = (char**)malloc((size_t)nsym * sizeof(char*));
        int s = 0;
        for (int p = i; p < j; ) {
            int cp; int n = utf8_dec(esc + p, &cp);
            sym[s] = (char*)malloc((size_t)n + 1);
            memcpy(sym[s], esc + p, (size_t)n); sym[s][n] = 0;
            p += n; s++;
        }
        int ns = nsym;
        while (ns > 1) {
            int best_rank = INT_MAX, bi = -1;
            char key[512];
            for (int b = 0; b < ns - 1; b++) {
                if (strlen(sym[b]) + strlen(sym[b + 1]) + 2 > sizeof(key)) continue;
                snprintf(key, sizeof(key), "%s %s", sym[b], sym[b + 1]);
                int r = smap_get(&t->merges, key);
                if (r >= 0 && r < best_rank) { best_rank = r; bi = b; }
            }
            if (bi < 0) break;
            char *merged = (char*)malloc(strlen(sym[bi]) + strlen(sym[bi + 1]) + 1);
            strcpy(merged, sym[bi]); strcat(merged, sym[bi + 1]);
            free(sym[bi]); free(sym[bi + 1]); sym[bi] = merged;
            for (int b = bi + 1; b < ns - 1; b++) sym[b] = sym[b + 1];
            ns--;
        }
        for (int b = 0; b < ns; b++) {
            int id = smap_get(&t->vocab, sym[b]);
            if (id >= 0) { if (no < cap) out[no++] = id; }
            else {
                /* Byte fallback, one <0xHH> per byte of whatever did not resolve. */
                for (const unsigned char *p = (const unsigned char*)sym[b]; *p; p++) {
                    int bid = t->byte_id[*p];
                    if (bid >= 0) { if (no < cap) out[no++] = bid; }
                    else warn_dropped(sym[b]);
                }
            }
            free(sym[b]);
        }
        free(sym);
        i = j;
    }
    free(esc);
    return no;
}

/* Byte-level BPE over one stretch of text with no added token in it. */
static int bpe_encode_span(const bpe_tokenizer *t, const char *text, int len, int *out, int cap);

int bpe_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    if (t && t->spm_bpe) return spm_bpe_encode(t, text, out, cap);
    if (t && t->spm) return spm_encode(t, text, out, cap);
    if (!t || t->n_added <= 0) return bpe_encode_span(t, text, (int)strlen(text), out, cap);

    /* Added tokens are matched against the raw text first, longest at each position, and the
     * merge path only ever sees what lies between them. Doing it the other way round cannot
     * work: these tokens are stored as literal bytes, so no sequence of merges over the
     * byte-level alphabet will ever spell one. */
    int no = 0, L = (int)strlen(text), i = 0, gap = 0;
    while (i < L) {
        int hit = -1, hlen = 0;
        for (int a = 0; a < t->n_added; a++) {
            const char *s = t->tokens[t->added_id[a]];
            int sl = (int)strlen(s);
            if (sl && sl <= L - i && memcmp(text + i, s, (size_t)sl) == 0) {
                hit = t->added_id[a]; hlen = sl; break;   /* the list is longest-first */
            }
        }
        if (hit < 0) { i++; continue; }
        if (i > gap) no += bpe_encode_span(t, text + gap, i - gap, out + no, cap - no);
        if (no < cap) out[no++] = hit;
        i += hlen;
        gap = i;
    }
    if (L > gap) no += bpe_encode_span(t, text + gap, L - gap, out + no, cap - no);
    return no;
}

static int bpe_encode_span(const bpe_tokenizer *t, const char *span, int L, int *out, int cap) {
    char *text = (char*)malloc((size_t)L + 1);
    if (!text) return 0;
    memcpy(text, span, (size_t)L);
    text[L] = 0;
    int no = 0, i = 0;
    while (i < L) {
        /* pre-tok piece: [i, j). One space belongs to the run that follows it, but a RUN of
         * spaces does not — GPT-2 splits `\s+(?!\S)` off first, so four spaces before a word
         * are three spaces and then " word". Taking each space as its own piece instead is
         * quiet on prose and loud on code: an indented line came out as three separate
         * space tokens where the reference emits one, and every position after it shifted. */
        int j;
        int run = 0;
        while (i + run < L && text[i + run] == ' ') run++;
        if (run > 1) {
            /* A run of spaces: all but the last are one piece, and the last one goes with
             * the word after it. At end of text there is no word, so the run stays whole. */
            j = (i + run < L) ? i + run - 1 : L;
        } else {
            j = i + 1;
            while (j < L && text[j] != ' ') j++;
        }
        int nsym = j - i;
        char **sym = (char**)malloc(nsym * sizeof(char*));
        for (int b = 0; b < nsym; b++) {
            char buf[8];
            int cp = t->byte_cp[(unsigned char)text[i + b]];
            int n = utf8_enc(cp, buf); buf[n] = 0;
            sym[b] = strdup(buf);
        }
        int ns = nsym;
        while (ns > 1) {
            int best_rank = INT_MAX, bi = -1;
            char key[512];
            for (int b = 0; b < ns - 1; b++) {
                snprintf(key, sizeof(key), "%s %s", sym[b], sym[b + 1]);
                int r = smap_get(&t->merges, key);
                if (r >= 0 && r < best_rank) { best_rank = r; bi = b; }
            }
            if (bi < 0) break;
            char *merged = (char*)malloc(strlen(sym[bi]) + strlen(sym[bi + 1]) + 1);
            strcpy(merged, sym[bi]); strcat(merged, sym[bi + 1]);
            free(sym[bi]); free(sym[bi + 1]); sym[bi] = merged;
            for (int b = bi + 1; b < ns - 1; b++) sym[b] = sym[b + 1];
            ns--;
        }
        for (int b = 0; b < ns; b++) {
            int id = smap_get(&t->vocab, sym[b]);
            if (id >= 0) { if (no < cap) out[no++] = id; }
            else warn_dropped(sym[b]);
            free(sym[b]);
        }
        free(sym);
        i = j;
    }
    free(text);            /* the span copy this function made, not the caller's */
    return no;
}

int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap) {
    if (!t || id < 0 || id >= t->n_tokens || !t->tokens[id]) return 0;
    const char *s = t->tokens[id];
    if (t->spm || t->spm_bpe) {
        /* A byte token carries one byte. Everything else is text in which
         * U+2581 stands for a space — three bytes wide, so the table the
         * byte-level path reads (512 entries) could never have mapped it. */
        int bv = byte_token_value(s);
        if (bv >= 0) {
            if (cap < 2) { if (cap > 0) buf[0] = 0; return 0; }
            buf[0] = (char)bv; buf[1] = 0; return 1;
        }
        int n = 0;
        for (int i = 0; s[i] && n < cap - 1; ) {
            if ((unsigned char)s[i] == 0xE2 && (unsigned char)s[i+1] == 0x96
                                            && (unsigned char)s[i+2] == 0x81) {
                buf[n++] = ' '; i += 3;
            } else buf[n++] = s[i++];
        }
        buf[n] = 0;
        return n;
    }
    /* An added token is literal text and was never in the byte-level alphabet, so running it
     * through that table drops every byte it does not recognise — a run of real spaces comes
     * out empty, which is how indentation disappeared from generated code while the ids were
     * right all along. Copy it as it stands. */
    for (int a = 0; a < t->n_added; a++) {
        if (t->added_id[a] != id) continue;
        int len = (int)strlen(s);
        if (len > cap - 1) len = cap - 1;
        if (len < 0) len = 0;
        memcpy(buf, s, (size_t)len);
        buf[len] = 0;
        return len;
    }

    int L = (int)strlen(s), i = 0, n = 0;
    while (i < L && n < cap - 1) {
        int cp; int adv = utf8_dec(s + i, &cp); i += adv;
        int byte = (cp >= 0 && cp < 512) ? t->cp2byte[cp] : -1;
        if (byte >= 0) buf[n++] = (char)byte;
    }
    buf[n] = 0;
    return n;
}

#ifdef BPE_TEST
/* The gate this file had all along, pointed at both schemes instead of one.
 *
 * What it asserts is the property a tokenizer has to have and the one that was
 * broken: text goes in and the same text comes back. SentencePiece prepends a
 * space to what it encodes — add_dummy_prefix, which is what these vocabularies
 * were built over — so the expected answer there carries one leading space and
 * says so rather than being trimmed into looking tidy.
 *
 *   cc -O2 -I. -DBPE_TEST -o bpe_test examples/bpe.c gguf.c notorch.c -lm
 *   ./bpe_test model.gguf   → BPE_OK
 */
int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <gguf> [text ...]\n", argv[0]); return 1; }
    bpe_tokenizer *t = bpe_load(argv[1]);
    if (!t) { printf("bpe_load failed\n"); return 1; }
    printf("bpe  vocab=%d scheme=%s\n", bpe_n_vocab(t),
           t->spm ? "sentencepiece (scores, U+2581, byte fallback)"
                  : "byte-level BPE (merge ranks)");

    static const char *DEFAULT_CASES[] = {
        "The capital of France is Paris.",
        "Privet, mir! Hello world.",
        "  leading and trailing  ",
        "1234567890 !@#$%^&*()",
        "resonance",
        "\xF0\x9F\x94\xA5 fire",          /* not in any of these vocabularies */
    };
    const char **cases = DEFAULT_CASES;
    int ncase = (int)(sizeof(DEFAULT_CASES) / sizeof(DEFAULT_CASES[0]));
    if (argc > 2) { cases = (const char**)&argv[2]; ncase = argc - 2; }

    int fails = 0, ids[4096];
    char out[8192], want[8192];
    for (int c = 0; c < ncase; c++) {
        const char *text = cases[c];
        int n = bpe_encode(t, text, ids, 4096);
        int on = 0;
        for (int i = 0; i < n; i++)
            on += bpe_decode_token(t, ids[i], out + on, (int)sizeof(out) - on);
        out[on] = 0;
        snprintf(want, sizeof(want), "%s%s", t->spm ? " " : "", text);
        int ok = (strcmp(out, want) == 0);
        if (!ok) fails++;
        printf("bpe  [%d tok / %d bytes] '%s' -> '%s'  %s\n",
               n, (int)strlen(text), text, out, ok ? "PASS" : "FAIL");
    }

    /* Merges have to be doing something: a sentence of words must cost far
     * fewer tokens than it has bytes. Character-at-a-time encoding passes a
     * round-trip and is still wrong, which is exactly how this hid. */
    int n = bpe_encode(t, "The capital of France is Paris.", ids, 4096);
    int merged = (n * 3 <= 31);
    if (!merged) fails++;
    printf("bpe  [merges] 31 bytes -> %d tokens  %s\n", n, merged ? "PASS" : "FAIL");

    printf("%s\n", fails ? "BPE_FAIL" : "BPE_OK");
    bpe_free(t);
    return fails ? 1 : 0;
}
#endif

/* End of generation is not always one id. Gemma 4 closes a turn with <turn|> and answers a
 * tool call with <|tool_response>, and a run that only watches <eos> prints those markers as
 * text and keeps going — which is what this harness did on its first gemma run. llama.cpp
 * carries the same list for this family; nothing is assumed for the others, where the eos id
 * from the file is the whole answer. */
int bpe_is_eog(const bpe_tokenizer *t, int id) {
    if (!t || id < 0 || id >= t->n_tokens || !t->tokens[id]) return 0;
    if (!t->spm_bpe) return 0;
    const char *s = t->tokens[id];
    return strcmp(s, "<eos>") == 0 || strcmp(s, "<turn|>") == 0 ||
           strcmp(s, "<|tool_response>") == 0;
}
