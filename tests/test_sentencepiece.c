#define _POSIX_C_SOURCE 200809L
#include "sentencepiece.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sentencepiece_reference.h"

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static void compare(const nt_spm_model *m, const sp_reference *ref) {
    nt_spm_result r = {0};
    char error[128];
    int rc = nt_spm_encode(m, ref->text, strlen(ref->text), &r, error, sizeof(error));
    if (rc) fprintf(stderr, "case %s: %s\n", ref->name, error);
    CHECK(rc == 0);
    CHECK(r.normalized_bytes == strlen(ref->normalized));
    CHECK(strcmp(r.normalized, ref->normalized) == 0);
    CHECK(r.count == ref->count);
    size_t offset = 0;
    for (size_t i = 0; i < r.count; i++) {
        CHECK(r.pieces[i].id == ref->ids[i]);
        CHECK(r.pieces[i].offset == offset);
        CHECK(r.pieces[i].length == strlen(ref->pieces[i]));
        CHECK(memcmp(r.normalized + offset, ref->pieces[i], r.pieces[i].length) == 0);
        offset += r.pieces[i].length;
    }
    CHECK(offset == r.normalized_bytes);
    nt_spm_result_free(&r);
    CHECK(r.normalized == NULL && r.pieces == NULL && r.count == 0 && r.normalized_bytes == 0);
}

static void reference_cases(const char *haiku_path) {
    char error[128];
    nt_spm_model *haiku = haiku_path ? nt_spm_load(haiku_path, error, sizeof(error)) : NULL;
    if (haiku_path && !haiku) fprintf(stderr, "Haiku model: %s\n", error);
    CHECK(!haiku_path || haiku != NULL);
    if (haiku) CHECK(nt_spm_n_vocab(haiku) == 650);
    unsigned embedded = 0, original = 0;
    for (size_t i = 0; i < COUNT(sp_references); i++) {
        const sp_reference *ref = sp_references + i;
        if (!ref->model && !haiku) continue;
        nt_spm_model *m = haiku;
        if (ref->model) {
            unsigned char *copy = malloc(ref->model_bytes);
            CHECK(copy != NULL);
            memcpy(copy, ref->model, ref->model_bytes);
            m = nt_spm_load_memory(copy, ref->model_bytes, error, sizeof(error));
            CHECK(m != NULL);
            memset(copy, 0, ref->model_bytes); free(copy); /* model owns its bytes */
            embedded++;
        } else original++;
        compare(m, ref);
        if (ref->model) nt_spm_free(m);
    }
    nt_spm_free(haiku);
    printf("PASS %u embedded oracle cases", embedded);
    if (haiku_path) printf(" and %u original Haiku model cases\n", original);
    else puts("; SKIP original Haiku model corpus (set SPM_MODEL=path)");
}

static nt_spm_model *tiny_with_flags(void) {
    /* Proto message merging: replace the existing normalizer's three flags. */
    unsigned char bytes[sizeof(sp_model_0) + 8];
    const unsigned char flags[] = {0x1a, 6, 0x18, 1, 0x20, 1, 0x28, 1};
    memcpy(bytes, sp_model_0, sizeof(sp_model_0));
    memcpy(bytes + sizeof(sp_model_0), flags, sizeof(flags));
    return nt_spm_load_memory(bytes, sizeof(bytes), NULL, 0);
}

static void boundaries(void) {
    nt_spm_model *m = tiny_with_flags();
    CHECK(m != NULL);
    size_t n = NT_SPM_MAX_TEXT_BYTES;
    char *text = malloc(n + 1);
    CHECK(text != NULL);
    memset(text, 'a', n); text[n] = 0;
    memcpy(text + n - 3, "\xe2\x96\x81", 3);
    nt_spm_result r = {0};
    /* Dummy prefix plus a removed literal trailing marker: intermediate
     * normalization is larger than the final permitted one-MiB result. */
    CHECK(nt_spm_encode(m, text, n, &r, NULL, 0) == 0);
    CHECK(r.normalized_bytes == n && r.normalized[n] == 0);
    CHECK(memcmp(r.normalized, "\xe2\x96\x81", 3) == 0);
    CHECK(r.normalized[n - 1] == 'a');
    nt_spm_result_free(&r);
    memset(text, 'a', n);
    nt_spm_result sentinel = {(char *)(uintptr_t)1, 123, (nt_spm_piece *)(uintptr_t)2, 456};
    r = sentinel;
    CHECK(nt_spm_encode(m, text, n, &r, NULL, 0) == -1);
    CHECK(memcmp(&r, &sentinel, sizeof(r)) == 0);
    CHECK(nt_spm_encode(m, text, n + 1, &r, NULL, 0) == -1);
    CHECK(memcmp(&r, &sentinel, sizeof(r)) == 0);
    for (size_t i = 0; i + 3 <= n; i += 3) memcpy(text + i, "\xe2\x96\x81", 3);
    memset(&r, 0, sizeof(r));
    CHECK(nt_spm_encode(m, text, n / 3 * 3, &r, NULL, 0) == 0);
    CHECK(r.normalized_bytes == 0 && r.count == 0);
    nt_spm_result_free(&r); free(text); nt_spm_free(m);

    m = nt_spm_load_memory(sp_model_0, sizeof(sp_model_0), NULL, 0);
    CHECK(m != NULL);
    const char invalid[] = {'a', (char)0xf0, (char)0x80, (char)0x80, 'a'};
    CHECK(nt_spm_encode(m, invalid, sizeof(invalid), &r, NULL, 0) == 0);
    CHECK(strcmp(r.normalized, "a\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" "a") == 0);
    CHECK(r.count == 3 && r.pieces[1].id == 0 && r.pieces[1].length == 9);
    nt_spm_result_free(&r);
    CHECK(nt_spm_encode(m, NULL, 0, &r, NULL, 0) == 0 && r.count == 0);
    nt_spm_result_free(&r);
    r = sentinel;
    CHECK(nt_spm_encode(m, "a\0b", 3, &r, NULL, 0) == -1);
    CHECK(nt_spm_encode(NULL, "a", 1, &r, NULL, 0) == -1);
    CHECK(nt_spm_encode(m, NULL, 1, &r, NULL, 0) == -1);
    CHECK(nt_spm_encode(m, "a", 1, NULL, NULL, 0) == -1);
    CHECK(memcmp(&r, &sentinel, sizeof(r)) == 0);
    nt_spm_free(m); nt_spm_free(NULL); nt_spm_result_free(NULL);
    puts("PASS exact final-output cap, malformed UTF-8, NUL and unchanged failure outputs");
}

static void malformed_models(void) {
    char error[128];
    CHECK(nt_spm_load_memory(NULL, 1, error, sizeof(error)) == NULL);
    CHECK(nt_spm_load_memory(sp_model_0, 0, error, sizeof(error)) == NULL);
    CHECK(nt_spm_load_memory(sp_model_0, NT_SPM_MAX_MODEL_BYTES + 1, error, sizeof(error)) == NULL);
    CHECK(nt_spm_load(NULL, error, sizeof(error)) == NULL);
    CHECK(nt_spm_load("", error, sizeof(error)) == NULL);
    CHECK(nt_spm_load("/no/such/sentencepiece-model", error, sizeof(error)) == NULL);
    const unsigned char bad[][11] = {
        {0}, {0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 2},
        {0x0b}, {0xff, 0xff, 0xff, 0xff, 0x1f}, {0x0a, 0x7f, 0x01}
    };
    for (size_t i = 0; i < COUNT(bad); i++) CHECK(nt_spm_load_memory(bad[i], sizeof(bad[i]), NULL, 0) == NULL);
    unsigned char bytes[sizeof(sp_model_0) + 16];
    memcpy(bytes, sp_model_0, sizeof(sp_model_0));
    /* Unknown fields are legal and retained model bytes remain owned. */
    const unsigned char extension[] = {0xa0, 6, 7};
    memcpy(bytes + sizeof(sp_model_0), extension, sizeof(extension));
    nt_spm_model *m = nt_spm_load_memory(bytes, sizeof(sp_model_0) + sizeof(extension), NULL, 0);
    CHECK(m != NULL); nt_spm_free(m);
    const unsigned char unsupported[] = {0x12, 2, 0x18, 2};
    memcpy(bytes + sizeof(sp_model_0), unsupported, sizeof(unsupported));
    CHECK(nt_spm_load_memory(bytes, sizeof(sp_model_0) + sizeof(unsupported), error, sizeof(error)) == NULL);
    CHECK(strstr(error, "UNIGRAM") != NULL);
    const unsigned char fallback[] = {0x12, 3, 0x98, 2, 1};
    memcpy(bytes + sizeof(sp_model_0), fallback, sizeof(fallback));
    CHECK(nt_spm_load_memory(bytes, sizeof(sp_model_0) + sizeof(fallback), error, sizeof(error)) == NULL);
    CHECK(strstr(error, "byte fallback") != NULL);
    memcpy(bytes, sp_model_0, sizeof(sp_model_0));
    for (size_t i = 0; i + 5 < sizeof(sp_model_0); i++) if (bytes[i] == 0x15) {
        bytes[i + 1] = 0; bytes[i + 2] = 0; bytes[i + 3] = 0xc0; bytes[i + 4] = 0x7f;
        break;
    }
    CHECK(nt_spm_load_memory(bytes, sizeof(sp_model_0), NULL, 0) == NULL);
    error[0] = error[1] = 'x';
    CHECK(nt_spm_load_memory(NULL, 0, error, 1) == NULL && error[0] == 0 && error[1] == 'x');

    /* All truncated prefixes and 2,048 deterministic corruptions exercise
     * parser ownership. Valid modified models may still load and encode. */
    unsigned rejected = 0;
    const sp_reference *mapped = NULL;
    for (size_t i = 0; i < COUNT(sp_references); i++) if (strstr(sp_references[i].name, "flags=")) { mapped = sp_references + i; break; }
    CHECK(mapped != NULL);
    for (size_t n = 1; n < sizeof(sp_model_0); n++) {
        m = nt_spm_load_memory(sp_model_0, n, NULL, 0);
        rejected += m == NULL; nt_spm_free(m);
    }
    unsigned char *copy = malloc(mapped->model_bytes);
    CHECK(copy != NULL);
    uint32_t rng = 0x472192ab;
    for (unsigned i = 0; i < 2048; i++) {
        memcpy(copy, mapped->model, mapped->model_bytes);
        rng = rng * 1664525U + 1013904223U;
        size_t offset = rng % mapped->model_bytes;
        copy[offset] ^= (unsigned char)(1U << ((rng >> 24) & 7));
        m = nt_spm_load_memory(copy, mapped->model_bytes, NULL, 0);
        if (!m) rejected++;
        else {
            nt_spm_result r = {0};
            int rc = nt_spm_encode(m, "X XY é Σ <tag>", strlen("X XY é Σ <tag>"), &r, NULL, 0);
            CHECK(rc == 0 || rc == -1);
            nt_spm_result_free(&r); nt_spm_free(m);
        }
    }
    free(copy); CHECK(rejected > 500);
    printf("PASS malformed protobuf/options and %u rejected truncated/corrupted models\n", rejected);
}

typedef struct { const nt_spm_model *model; int failed; } thread_case;
static void *worker(void *arg) {
    thread_case *c = arg;
    for (unsigned i = 0; i < 1000; i++) {
        nt_spm_result r = {0};
        if (nt_spm_encode(c->model, "aaa👀aaa", strlen("aaa👀aaa"), &r, NULL, 0) ||
            r.count != 5 || strcmp(r.normalized, "aaa👀aaa") ||
            r.pieces[0].id != 3 || r.pieces[1].id != 4 ||
            r.pieces[2].id != 0 || r.pieces[2].length != 4 ||
            r.pieces[3].id != 3 || r.pieces[4].id != 4) c->failed = 1;
        nt_spm_result_free(&r);
    }
    return NULL;
}

static void concurrent_and_file(void) {
    nt_spm_model *m = nt_spm_load_memory(sp_model_0, sizeof(sp_model_0), NULL, 0);
    CHECK(m != NULL);
    pthread_t threads[8]; thread_case cases[8];
    for (unsigned i = 0; i < 8; i++) {
        cases[i] = (thread_case){m, 0};
        CHECK(pthread_create(threads + i, NULL, worker, cases + i) == 0);
    }
    for (unsigned i = 0; i < 8; i++) {
        CHECK(pthread_join(threads[i], NULL) == 0); CHECK(!cases[i].failed);
    }
    nt_spm_free(m);
    char path[] = "/tmp/notorch-spm-XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0);
    CHECK(write(fd, sp_model_0, sizeof(sp_model_0)) == sizeof(sp_model_0));
    CHECK(close(fd) == 0);
    m = nt_spm_load(path, NULL, 0); CHECK(m != NULL);
    CHECK(unlink(path) == 0);
    nt_spm_result r = {0};
    CHECK(nt_spm_encode(m, "aaa", 3, &r, NULL, 0) == 0);
    CHECK(r.count == 2 && r.pieces[0].id == 3 && r.pieces[1].id == 4);
    nt_spm_result_free(&r); nt_spm_free(m);
    puts("PASS eight concurrent readers / 8,000 calls and ownership after file removal");
}

int main(int argc, char **argv) {
    reference_cases(argc > 1 && argv[1][0] ? argv[1] : NULL);
    boundaries(); malformed_models(); concurrent_and_file();
    printf("SentencePiece native: %u checks passed\n", checks);
    return 0;
}
