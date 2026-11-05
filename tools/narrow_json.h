/* narrow_json.h — the part of JSON the EmbeddingGemma 2 tools read, and no more:
 * skip a value, find a key, walk the members of an array or object, decode a
 * string, read a number. Input is one NUL-terminated buffer. Every reader stops
 * at the terminator and reports malformed input as NULL or -1. It walks what it
 * can walk and is not a validator; callers check the values they use. Keys are
 * matched by their raw bytes, so an escaped spelling of a key is not found and
 * the caller refuses the file instead of guessing. */
#ifndef NARROW_JSON_H
#define NARROW_JSON_H
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline const char *nj_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}
/* p at '"': the byte after the closing quote. */
static inline const char *nj_str_end(const char *p) {
    if (*p != '"') return NULL;
    for (p++; *p != '"'; p++) {
        if (!*p) return NULL;
        if (*p == '\\' && !*++p) return NULL;
    }
    return p + 1;
}
static inline const char *nj_skip(const char *p, int depth) {
    p = nj_ws(p);
    if (*p == '"') return nj_str_end(p);
    if (*p == '{' || *p == '[') {
        char close = *p == '{' ? '}' : ']';
        if (depth > 64) return NULL;
        p = nj_ws(p + 1);
        if (*p == close) return p + 1;
        for (;;) {
            if (close == '}') {
                if (!(p = nj_str_end(p))) return NULL;
                p = nj_ws(p);
                if (*p++ != ':') return NULL;
            }
            if (!(p = nj_skip(p, depth + 1))) return NULL;
            p = nj_ws(p);
            if (*p == close) return p + 1;
            if (*p++ != ',') return NULL;
            p = nj_ws(p);
        }
    }
    const char *s = p;
    while (*p && strchr("+-0123456789.eEtrufalsn", *p)) p++;
    return p > s ? p : NULL;
}
/* Walking a container: first member (for an object, its key) or NULL when empty;
 * nj_value turns a key into its value; nj_next steps past a value to the next
 * member, NULL at the closing bracket. Malformed input sets *bad. */
static inline const char *nj_first(const char *p, int *bad) {
    if (p) p = nj_ws(p);
    if (!p || (*p != '[' && *p != '{')) { *bad = 1; return NULL; }
    char close = *p == '[' ? ']' : '}';
    p = nj_ws(p + 1);
    return *p == close ? NULL : p;
}
static inline const char *nj_value(const char *key, int *bad) {
    const char *p = nj_str_end(key);
    if (!p || *(p = nj_ws(p)) != ':') { *bad = 1; return NULL; }
    return nj_ws(p + 1);
}
static inline const char *nj_next(const char *value, char close, int *bad) {
    const char *p = nj_skip(value, 0);
    if (!p) { *bad = 1; return NULL; }
    p = nj_ws(p);
    if (*p == close) return NULL;
    if (*p != ',') { *bad = 1; return NULL; }
    return nj_ws(p + 1);
}
static inline const char *nj_get(const char *obj, const char *key) {
    int bad = 0;
    size_t n = strlen(key);
    for (const char *k = nj_first(obj, &bad), *v; k && (v = nj_value(k, &bad)); k = nj_next(v, '}', &bad))
        if (!strncmp(k + 1, key, n) && k[n + 1] == '"') return v;
    return NULL;
}
/* Whether the value at v is exactly the string s (raw bytes, no escapes). */
static inline int nj_is(const char *v, const char *s) {
    size_t n = strlen(s);
    return v && *v == '"' && !strncmp(v + 1, s, n) && v[n + 1] == '"';
}
static inline int nj_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return -1;
        v = v * 16 + d;
    }
    return v;
}
/* The string at p, unescaped into malloc'd UTF-8. NULL on malformed input, on
 * \u0000 (the result is a C string) and on an unpaired surrogate. */
static inline char *nj_string(const char *p) {
    const char *end = p ? nj_str_end(p) : NULL;
    char *out = end ? malloc((size_t)(end - p)) : NULL, *o = out;
    if (!out) return NULL;
    for (p++; *p != '"'; p++) {
        if (*p != '\\') { *o++ = *p; continue; }
        int c = *++p;
        if (c == 'b') c = '\b'; else if (c == 'f') c = '\f'; else if (c == 'n') c = '\n';
        else if (c == 'r') c = '\r'; else if (c == 't') c = '\t';
        else if (c == 'u') {
            if ((c = nj_hex4(p + 1)) < 0) { free(out); return NULL; }
            p += 4;
            if (c >= 0xD800 && c < 0xDC00) {
                int lo = p[1] == '\\' && p[2] == 'u' ? nj_hex4(p + 3) : -1;
                if (lo < 0xDC00 || lo >= 0xE000) { free(out); return NULL; }
                c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
                p += 6;
            }
            if (c == 0 || (c >= 0xD800 && c < 0xE000)) { free(out); return NULL; }
            if (c < 0x80) *o++ = (char)c;
            else if (c < 0x800) { *o++ = (char)(0xC0 | c >> 6); *o++ = (char)(0x80 | (c & 0x3F)); }
            else if (c < 0x10000) { *o++ = (char)(0xE0 | c >> 12); *o++ = (char)(0x80 | (c >> 6 & 0x3F)); *o++ = (char)(0x80 | (c & 0x3F)); }
            else { *o++ = (char)(0xF0 | c >> 18); *o++ = (char)(0x80 | (c >> 12 & 0x3F)); *o++ = (char)(0x80 | (c >> 6 & 0x3F)); *o++ = (char)(0x80 | (c & 0x3F)); }
            continue;
        } else if (c != '"' && c != '\\' && c != '/') { free(out); return NULL; }
        *o++ = (char)c;
    }
    *o = 0;
    return out;
}
static inline int nj_number(const char *p, double *out) {
    char *e; errno = 0;
    double v = p ? strtod(p, &e) : 0;
    if (!p || e == p || errno) return -1;
    *out = v;
    return 0;
}
static inline int nj_int(const char *p, long long *out) {
    char *e; errno = 0;
    long long v = p ? strtoll(p, &e, 10) : 0;
    if (!p || e == p || errno || *e == '.' || *e == 'e' || *e == 'E') return -1;
    *out = v;
    return 0;
}
/* A whole file as one C string; refused if it holds a NUL byte. */
static inline char *nj_load(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    char *buf = NULL;
    long n = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    if (n >= 0 && !fseek(f, 0, SEEK_SET) && (buf = malloc((size_t)n + 1))) {
        if (fread(buf, 1, (size_t)n, f) != (size_t)n || memchr(buf, 0, (size_t)n)) { free(buf); buf = NULL; }
        else buf[n] = 0;
    }
    fclose(f);
    if (!buf) fprintf(stderr, "%s: unreadable, or contains a NUL byte\n", path);
    else if (size) *size = (size_t)n;
    return buf;
}
#endif
