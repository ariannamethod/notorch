/* A captured prompt is an immutable sensory episode. Its digest covers the
 * exact host, token IDs, sparse projected rows and the output-token policy.
 * Disk files are sealed GGUF v3 containers, episode schema version 1. */
#include "harness/replay_internal.h"
#include "sha256.h"
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(int) == 4, "episode IDs use 32-bit int");
_Static_assert(sizeof(float) == 4, "episode rows use F32");

/* Bound metadata independently of tensor storage. A million prompt positions
 * covers long contexts while keeping each typed ID array within four MiB. */
enum { EPISODE_MAX_IDS = 1 << 20, EPISODE_FIELDS = 14 };
static const char episode_arch[] = "notorch-doe-episode";
static const char media_name[] = "doe.episode.media";

static int little_endian(void) {
    const uint32_t one = 1;
    return *(const unsigned char *)&one == 1;
}

static int digest_string(const char s[65]) {
    if (!s || s[64]) return 0;
    for (int i = 0; i < 64; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return 0;
    return 1;
}

static int counts_valid(const nt_replay_episode *e) {
    if (!e || !little_endian() || e->width < 1 || e->width > NT_MAX_ELEMENTS ||
        e->vocab < 1 || e->vocab > NT_MAX_ELEMENTS || e->n < 1 ||
        e->n > EPISODE_MAX_IDS || e->n > NT_MAX_ELEMENTS ||
        e->n_media < 0 || e->n_media > e->n ||
        e->n_suppressed < 0 || e->n_suppressed > e->vocab ||
        e->n_suppressed > EPISODE_MAX_IDS ||
        e->n_stops < 0 || e->n_stops > e->vocab || e->n_stops > EPISODE_MAX_IDS)
        return 0;
    uint64_t values = (uint64_t)e->n_media * (uint64_t)e->width;
    return values <= NT_MAX_ELEMENTS && values <= SIZE_MAX / sizeof(float) &&
           (uint64_t)e->n <= SIZE_MAX / sizeof(int) &&
           (uint64_t)e->n_suppressed <= SIZE_MAX / sizeof(int) &&
           (uint64_t)e->n_stops <= SIZE_MAX / sizeof(int);
}

static int body_valid(const nt_experts *p) {
    return p && p->body.model && p->body.tokenizer && p->decoder &&
           p->decoder->doe && p->decoder->doe->forward &&
           p->dim > 0 && p->dim <= NT_MAX_ELEMENTS &&
           p->dims.vocab > 0 && p->dims.vocab <= NT_MAX_ELEMENTS &&
           digest_string(p->host);
}

static int body_matches(const nt_experts *p, const nt_replay_episode *e) {
    return body_valid(p) && counts_valid(e) &&
           e->width == p->dim && e->vocab == p->dims.vocab &&
           !memcmp(e->host, p->host, sizeof(e->host)) &&
           (!e->n_media || p->decoder->doe->forward_embd) &&
           (!p->frame_host || p->frame_host->ctx_len <= 0 ||
            e->n <= p->frame_host->ctx_len);
}

/* The arrays passed here may be unaligned views into a GGUF metadata section.
 * memcpy also preserves the bit pattern of each projected media value. */
static int array_valid(const void *data, int n, int limit, int increasing) {
    if (n && !data) return 0;
    const unsigned char *bytes = data;
    int previous = -1;
    for (int i = 0; i < n; i++) {
        int32_t v;
        memcpy(&v, bytes + (size_t)i * sizeof(v), sizeof(v));
        if (v < 0 || v >= limit || (increasing && v <= previous)) return 0;
        previous = v;
    }
    return 1;
}

static void episode_arrays(const nt_replay_episode *e, const void *arrays[5]) {
    arrays[0] = e->ids;
    arrays[1] = e->positions;
    arrays[2] = e->suppressed;
    arrays[3] = e->stops;
    arrays[4] = e->media;
}

static int payload_valid(const nt_replay_episode *e, const void *const arrays[5]) {
    if (!counts_valid(e) || !digest_string(e->host) ||
        !array_valid(arrays[0], e->n, e->vocab, 0) ||
        !array_valid(arrays[1], e->n_media, e->n, 1) ||
        !array_valid(arrays[2], e->n_suppressed, e->vocab, 1) ||
        !array_valid(arrays[3], e->n_stops, e->vocab, 1) ||
        (e->n_media && !arrays[4])) return 0;
    size_t n = (size_t)e->n_media * e->width;
    const unsigned char *bytes = arrays[4];
    for (size_t i = 0; i < n; i++) {
        float value;
        memcpy(&value, bytes + i * sizeof(value), sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

static int contents_valid(const nt_replay_episode *e) {
    if (!e) return 0;
    const void *arrays[5];
    episode_arrays(e, arrays);
    return payload_valid(e, arrays);
}

static void encode_u32(unsigned char *out, uint32_t value) {
    for (int i = 0; i < 4; i++) out[i] = (unsigned char)(value >> (8*i));
}

/* Fixed-length manifest: domain, host hex, six LE u32 counts, then the SHA-256
 * of ids, positions, suppression, stops and F32 rows in that order. Empty
 * arrays use SHA-256(empty). This needs no second copy of the media tensor. */
static int payload_digest(const nt_replay_episode *e, const void *const arrays[5], char out[65]) {
    static const char domain[] = "notorch-doe-episode-v1";
    unsigned char manifest[sizeof(domain)-1 + 64 + 6*4 + 5*32];
    unsigned char *at = manifest;
    memcpy(at, domain, sizeof(domain)-1); at += sizeof(domain)-1;
    memcpy(at, e->host, 64); at += 64;
    const int counts[6] = {e->width, e->vocab, e->n, e->n_media,
                           e->n_suppressed, e->n_stops};
    for (int i = 0; i < 6; i++, at += 4) encode_u32(at, (uint32_t)counts[i]);
    const size_t sizes[5] = {(size_t)e->n * sizeof(int),
                            (size_t)e->n_media * sizeof(int),
                            (size_t)e->n_suppressed * sizeof(int),
                            (size_t)e->n_stops * sizeof(int),
                            (size_t)e->n_media * e->width * sizeof(float)};
    for (int i = 0; i < 5; i++, at += 32)
        if (nt_sha256(arrays[i], sizes[i], at)) return -1;
    unsigned char digest[32];
    if (nt_sha256(manifest, sizeof(manifest), digest)) return -1;
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2*i] = hex[digest[i] >> 4];
        out[2*i+1] = hex[digest[i] & 15];
    }
    out[64] = 0;
    return 0;
}

static int episode_digest(const nt_replay_episode *e, char out[65]) {
    const void *arrays[5];
    episode_arrays(e, arrays);
    return payload_digest(e, arrays, out);
}

static int episode_valid(const nt_replay_episode *e) {
    char digest[65];
    return contents_valid(e) && digest_string(e->id) &&
           !episode_digest(e, digest) && !memcmp(digest, e->id, sizeof(digest));
}

void nt_replay_free(nt_replay_episode *e) {
    if (!e) return;
    free(e->ids);
    free(e->positions);
    free(e->suppressed);
    free(e->stops);
    free(e->media);
    free(e);
}

/* Counts are checked before this allocator; all allocations belong to the
 * new episode, even while a source episode or mapped file remains alive. */
static nt_replay_episode *allocate_episode(const nt_replay_episode *src) {
    if (!counts_valid(src)) return NULL;
    nt_replay_episode *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    *e = *src;
    e->ids = NULL; e->positions = NULL; e->suppressed = NULL;
    e->stops = NULL; e->media = NULL;
    e->ids = malloc((size_t)e->n * sizeof(int));
    if (e->n_media) {
        e->positions = malloc((size_t)e->n_media * sizeof(int));
        e->media = malloc((size_t)e->n_media * e->width * sizeof(float));
    }
    if (e->n_suppressed) e->suppressed = malloc((size_t)e->n_suppressed * sizeof(int));
    if (e->n_stops) e->stops = malloc((size_t)e->n_stops * sizeof(int));
    if (!e->ids || (e->n_media && (!e->positions || !e->media)) ||
        (e->n_suppressed && !e->suppressed) || (e->n_stops && !e->stops)) {
        nt_replay_free(e);
        return NULL;
    }
    return e;
}

static nt_replay_episode *copy_payload(const nt_replay_episode *src, const void *const arrays[5]) {
    nt_replay_episode *e = allocate_episode(src);
    if (!e) return NULL;
    memcpy(e->ids, arrays[0], (size_t)e->n * sizeof(int));
    if (e->n_media) {
        memcpy(e->positions, arrays[1], (size_t)e->n_media * sizeof(int));
        memcpy(e->media, arrays[4], (size_t)e->n_media * e->width * sizeof(float));
    }
    if (e->n_suppressed)
        memcpy(e->suppressed, arrays[2], (size_t)e->n_suppressed * sizeof(int));
    if (e->n_stops) memcpy(e->stops, arrays[3], (size_t)e->n_stops * sizeof(int));
    return e;
}

static nt_replay_episode *copy_episode(const nt_replay_episode *src) {
    const void *arrays[5];
    episode_arrays(src, arrays);
    return copy_payload(src, arrays);
}

static int compare_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

nt_replay_episode *nt_replay_capture(const nt_experts *p, const int *ids,
                                    const float *const *rows, int n,
                                    const int *suppressed, int n_suppressed,
                                    const int *stops, int n_stops) {
    if (!body_valid(p)) return NULL;
    nt_replay_episode shape = {.width=p->dim, .vocab=p->dims.vocab, .n=n,
                               .n_suppressed=n_suppressed, .n_stops=n_stops};
    memcpy(shape.host, p->host, sizeof(shape.host));
    if (!body_matches(p, &shape) || !array_valid(ids, n, shape.vocab, 0) ||
        !array_valid(suppressed, n_suppressed, shape.vocab, 0) ||
        !array_valid(stops, n_stops, shape.vocab, 0)) return NULL;
    if (rows) {
        for (int i = 0; i < n; i++) if (rows[i]) {
            if (++shape.n_media > NT_MAX_ELEMENTS / shape.width) return NULL;
            for (int j = 0; j < shape.width; j++)
                if (!isfinite(rows[i][j])) return NULL;
        }
    }
    if (!body_matches(p, &shape)) return NULL;
    nt_replay_episode *e = allocate_episode(&shape);
    if (!e) return NULL;
    memcpy(e->ids, ids, (size_t)n * sizeof(int));
    int media = 0;
    for (int i = 0; rows && i < n; i++) if (rows[i]) {
        e->positions[media] = i;
        memcpy(e->media + (size_t)media * e->width, rows[i], (size_t)e->width * sizeof(float));
        media++;
    }
    if (n_suppressed) {
        memcpy(e->suppressed, suppressed, (size_t)n_suppressed * sizeof(int));
        qsort(e->suppressed, (size_t)n_suppressed, sizeof(int), compare_int);
    }
    if (n_stops) {
        memcpy(e->stops, stops, (size_t)n_stops * sizeof(int));
        qsort(e->stops, (size_t)n_stops, sizeof(int), compare_int);
    }
    if (!contents_valid(e) || episode_digest(e, e->id)) {
        nt_replay_free(e);
        return NULL;
    }
    return e;
}

nt_replay_episode *nt_replay_clone(const nt_replay_episode *e) {
    return episode_valid(e) ? copy_episode(e) : NULL;
}

int nt_replay_matches(const nt_experts *p, const nt_replay_episode *e) {
    return body_matches(p, e) && episode_valid(e);
}

const char *nt_replay_id(const nt_replay_episode *e) {
    return e ? e->id : NULL;
}

int nt_replay_save(const nt_replay_episode *e, const char *path) {
    if (!path || !*path || !episode_valid(e)) return -1;
    size_t length = strlen(path);
    if (length > SIZE_MAX - sizeof(".XXXXXX")) return -1;
    char *tmp = malloc(length + sizeof(".XXXXXX"));
    if (!tmp) return -1;
    memcpy(tmp, path, length);
    memcpy(tmp + length, ".XXXXXX", sizeof(".XXXXXX"));
    int fd = mkstemp(tmp), rc = -1;
    if (fd < 0) { free(tmp); return -1; }
    gguf_writer *w = gguf_write_open(tmp);
    if (!w) goto done;
    gguf_write_kv_str(w, "general.architecture", episode_arch);
    gguf_write_kv_u32(w, "doe.episode.version", 1);
    gguf_write_kv_str(w, "doe.episode.host_sha256", e->host);
    gguf_write_kv_str(w, "doe.episode.sha256", e->id);
    gguf_write_kv_u32(w, "doe.episode.width", (uint32_t)e->width);
    gguf_write_kv_u32(w, "doe.episode.vocab", (uint32_t)e->vocab);
    gguf_write_kv_u32(w, "doe.episode.tokens", (uint32_t)e->n);
    gguf_write_kv_u32(w, "doe.episode.media_rows", (uint32_t)e->n_media);
    gguf_write_kv_u32(w, "doe.episode.suppressed_count", (uint32_t)e->n_suppressed);
    gguf_write_kv_u32(w, "doe.episode.stop_count", (uint32_t)e->n_stops);
    gguf_write_kv_i32_array(w, "doe.episode.ids", (const int32_t *)e->ids, (uint64_t)e->n);
    gguf_write_kv_i32_array(w, "doe.episode.positions", (const int32_t *)e->positions, (uint64_t)e->n_media);
    gguf_write_kv_i32_array(w, "doe.episode.suppressed", (const int32_t *)e->suppressed, (uint64_t)e->n_suppressed);
    gguf_write_kv_i32_array(w, "doe.episode.stops", (const int32_t *)e->stops, (uint64_t)e->n_stops);
    if (e->n_media) {
        gguf_write_tensor_decl(w, media_name, 2,
                               (uint64_t[]){(uint64_t)e->width, (uint64_t)e->n_media}, GGUF_TYPE_F32);
        gguf_write_tensor_f32(w, media_name, e->media, (uint64_t)e->width * e->n_media);
    }
    /* close also flushes a zero-tensor text episode. link preserves every
     * pre-existing destination, including a host file or another episode. */
    if (!gguf_write_close(w) && !fsync(fd)) rc = link(tmp, path);
done:
    close(fd);
    unlink(tmp);
    free(tmp);
    return rc;
}

/* A bounded reader for the sealed episode schema. In contrast to model GGUF
 * arrays, every array's declared count, complete bytes and values are checked
 * before any episode allocation, using one immutable file mapping. */
typedef struct { const unsigned char *p, *end; } reader;

static const unsigned char *take(reader *r, size_t bytes) {
    if (bytes > (size_t)(r->end - r->p)) return NULL;
    const unsigned char *result = r->p;
    r->p += bytes;
    return result;
}

static int read_u32(reader *r, uint32_t *out) {
    const unsigned char *p = take(r, 4);
    if (!p) return 0;
    *out = (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
    return 1;
}

static int read_u64(reader *r, uint64_t *out) {
    uint32_t lo, hi;
    if (!read_u32(r, &lo) || !read_u32(r, &hi)) return 0;
    *out = (uint64_t)lo | (uint64_t)hi << 32;
    return 1;
}

static int read_string(reader *r, char *out, size_t capacity) {
    uint64_t n;
    if (!read_u64(r, &n) || n >= capacity) return 0;
    const unsigned char *p = take(r, (size_t)n);
    if (!p || memchr(p, 0, (size_t)n)) return 0;
    memcpy(out, p, (size_t)n);
    out[n] = 0;
    return 1;
}

static int read_episode(const nt_experts *p, const unsigned char *data, size_t size,
                         nt_replay_episode *view, const void *payload[5]) {
    static const char *const keys[EPISODE_FIELDS] = {
        "general.architecture", "doe.episode.version", "doe.episode.host_sha256",
        "doe.episode.sha256", "doe.episode.width", "doe.episode.vocab",
        "doe.episode.tokens", "doe.episode.media_rows", "doe.episode.suppressed_count",
        "doe.episode.stop_count", "doe.episode.ids", "doe.episode.positions",
        "doe.episode.suppressed", "doe.episode.stops"
    };
    reader r = {data, data + size};
    uint32_t magic, version, present = 0;
    uint64_t tensors, fields, lengths[4] = {0};
    const unsigned char *arrays[4] = {NULL};
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC ||
        !read_u32(&r, &version) || version != 3 ||
        !read_u64(&r, &tensors) || tensors > 1 ||
        !read_u64(&r, &fields) || fields != EPISODE_FIELDS) return 0;
    for (unsigned i = 0; i < EPISODE_FIELDS; i++) {
        char name[GGUF_MAX_NAME];
        uint32_t type;
        if (!read_string(&r, name, sizeof(name)) || !read_u32(&r, &type)) return 0;
        unsigned key = 0;
        while (key < EPISODE_FIELDS && strcmp(name, keys[key])) key++;
        if (key == EPISODE_FIELDS || (present & (UINT32_C(1) << key))) return 0;
        present |= UINT32_C(1) << key;
        if (key == 0) {
            char arch[64];
            if (type != 8 || !read_string(&r, arch, sizeof(arch)) || strcmp(arch, episode_arch)) return 0;
        } else if (key == 2 || key == 3) {
            char *digest = key == 2 ? view->host : view->id;
            if (type != 8 || !read_string(&r, digest, 65) || !digest_string(digest)) return 0;
        } else if (key < 10) {
            uint32_t value;
            if (type != 4 || !read_u32(&r, &value) || value > INT_MAX) return 0;
            switch (key) {
                case 1: if (value != 1) return 0; break;
                case 4: view->width = (int)value; break;
                case 5: view->vocab = (int)value; break;
                case 6: view->n = (int)value; break;
                case 7: view->n_media = (int)value; break;
                case 8: view->n_suppressed = (int)value; break;
                case 9: view->n_stops = (int)value; break;
            }
        } else {
            unsigned a = key - 10;
            uint32_t element_type;
            if (type != 9 || !read_u32(&r, &element_type) || element_type != 5 ||
                !read_u64(&r, &lengths[a]) || lengths[a] > EPISODE_MAX_IDS ||
                lengths[a] > SIZE_MAX / sizeof(int)) return 0;
            arrays[a] = take(&r, (size_t)lengths[a] * sizeof(int));
            if (!arrays[a]) return 0;
        }
    }
    if (present != (UINT32_C(1) << EPISODE_FIELDS)-1 || !body_matches(p, view) ||
        lengths[0] != (uint64_t)view->n || lengths[1] != (uint64_t)view->n_media ||
        lengths[2] != (uint64_t)view->n_suppressed || lengths[3] != (uint64_t)view->n_stops ||
        tensors != (uint64_t)(view->n_media != 0)) return 0;
    if (tensors) {
        char name[GGUF_MAX_NAME];
        uint32_t ndim, dtype;
        uint64_t width, rows, offset;
        if (!read_string(&r, name, sizeof(name)) || strcmp(name, media_name) ||
            !read_u32(&r, &ndim) || ndim != 2 || !read_u64(&r, &width) ||
            width != (uint64_t)view->width || !read_u64(&r, &rows) ||
            rows != (uint64_t)view->n_media || !read_u32(&r, &dtype) ||
            dtype != GGUF_TYPE_F32 || !read_u64(&r, &offset) || offset) return 0;
    }
    size_t header = (size_t)(r.p - data);
    if (header > SIZE_MAX - 31) return 0;
    size_t offset = (header + 31) & ~(size_t)31;
    size_t bytes = (size_t)view->n_media * view->width * sizeof(float);
    if (bytes > SIZE_MAX - 31) return 0;
    size_t padded = (bytes + 31) & ~(size_t)31;
    if (offset > size || padded != size - offset) return 0;
    for (size_t i = header; i < offset; i++) if (data[i]) return 0;
    for (size_t i = bytes; i < padded; i++) if (data[offset+i]) return 0;
    /* Metadata arrays may be unaligned. Keep byte views until validation and
     * digesting finish; only copy_payload publishes aligned owned arrays. */
    for (int i = 0; i < 4; i++) payload[i] = arrays[i];
    payload[4] = view->n_media ? data + offset : NULL;
    char digest[65];
    return payload_valid(view, payload) && !payload_digest(view, payload, digest) &&
           !memcmp(digest, view->id, sizeof(digest));
}

nt_replay_episode *nt_replay_load(const nt_experts *p, const char *path) {
    if (!body_valid(p) || !little_endian() || !path || !*path) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    const uint64_t maximum = (uint64_t)NT_MAX_ELEMENTS * sizeof(float) +
                             (uint64_t)4 * EPISODE_MAX_IDS * sizeof(int) + 65536;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 24 ||
        (uintmax_t)st.st_size > SIZE_MAX || (uintmax_t)st.st_size > maximum) {
        close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return NULL;
    nt_replay_episode view = {0};
    const void *payload[5] = {NULL};
    nt_replay_episode *e = read_episode(p, map, size, &view, payload) ? copy_payload(&view, payload) : NULL;
    munmap(map, size);
    return e;
}
