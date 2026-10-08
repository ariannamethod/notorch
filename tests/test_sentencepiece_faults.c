#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t calls, fail_at, live;
static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
typedef struct { max_align_t alignment; size_t bytes; } allocation;
static void *checked_malloc(size_t bytes) {
    calls++;
    if (calls == fail_at) return NULL;
    allocation *a = malloc(sizeof(*a) + bytes + 16);
    if (!a) return NULL;
    a->bytes = bytes; live++;
    memset((unsigned char *)(a + 1) + bytes, 0xa7, 16);
    return a + 1;
}
static void *checked_calloc(size_t n, size_t size) {
    if (size && n > SIZE_MAX / size) return NULL;
    void *p = checked_malloc(n * size);
    if (p) memset(p, 0, n * size);
    return p;
}
static void checked_free(void *ptr) {
    if (!ptr) return;
    allocation *a = (allocation *)ptr - 1;
    for (unsigned i = 0; i < 16; i++) CHECK(((unsigned char *)ptr)[a->bytes + i] == 0xa7);
    CHECK(live > 0); live--; free(a);
}
#define NT_SPM_MALLOC checked_malloc
#define NT_SPM_CALLOC checked_calloc
#define NT_SPM_FREE checked_free
#include "../sentencepiece.c"
#include "sentencepiece_reference.h"

int main(void) {
    char error[128];
    calls = 0;
    nt_spm_model *m = nt_spm_load_memory(sp_model_0, sizeof(sp_model_0), error, sizeof(error));
    CHECK(m != NULL);
    size_t load_calls = calls;
    nt_spm_free(m); CHECK(live == 0);
    for (size_t at = 1; at <= load_calls; at++) {
        calls = 0; fail_at = at;
        m = nt_spm_load_memory(sp_model_0, sizeof(sp_model_0), error, sizeof(error));
        CHECK(m == NULL && live == 0 && error[0]);
    }
    fail_at = 0;
    m = nt_spm_load_memory(sp_model_0, sizeof(sp_model_0), error, sizeof(error));
    CHECK(m != NULL);
    size_t model_live = live;
    nt_spm_result r = {0};
    calls = 0;
    CHECK(nt_spm_encode(m, "aaa👀aaa", strlen("aaa👀aaa"), &r, error, sizeof(error)) == 0);
    size_t encode_calls = calls;
    nt_spm_result_free(&r); CHECK(live == model_live);
    nt_spm_result sentinel = {(char *)(uintptr_t)1, 123, (nt_spm_piece *)(uintptr_t)2, 456};
    for (size_t at = 1; at <= encode_calls; at++) {
        calls = 0; fail_at = at; r = sentinel;
        CHECK(nt_spm_encode(m, "aaa👀aaa", strlen("aaa👀aaa"), &r, error, sizeof(error)) == -1);
        CHECK(memcmp(&r, &sentinel, sizeof(r)) == 0);
        CHECK(live == model_live && error[0]);
    }
    fail_at = 0;
    nt_spm_free(m); CHECK(live == 0);
    printf("SentencePiece faults: %u checks, %zu loader + %zu encoder allocation sites\n", checks, load_calls, encode_calls);
    return 0;
}
