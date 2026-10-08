#include "sha256.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256_reference.h"

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void compare(const void *data, size_t n, const char *expected) {
    unsigned char digest[32];
    char hex[65];
    CHECK(nt_sha256(data, n, digest) == 0);
    for (unsigned i = 0; i < 32; i++) snprintf(hex + i * 2, 3, "%02x", digest[i]);
    CHECK(strcmp(hex, expected) == 0);
}

int main(void) {
    /* NIST SHA256 example document and RFC 6234 section 8.5. */
    compare("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char *two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    compare(two_blocks, strlen(two_blocks), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    compare(NULL, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    unsigned char *data = malloc(1000001);
    CHECK(data != NULL);
    memset(data, 'a', 1000000);
    compare(data, 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    for (size_t i = 0; i < sizeof(sha256_references) / sizeof(sha256_references[0]); i++) {
        size_t n = sha256_references[i].bytes;
        for (size_t j = 0; j < n; j++) data[j + 1] = (unsigned char)(j * 53 + j / 7 + 173);
        compare(data + 1, n, sha256_references[i].hex); /* deliberately unaligned */
    }
    unsigned char expected[32], overlap[96];
    for (unsigned i = 0; i < sizeof(overlap); i++) overlap[i] = (unsigned char)i;
    CHECK(nt_sha256(overlap, sizeof(overlap), expected) == 0);
    CHECK(nt_sha256(overlap, sizeof(overlap), overlap + 5) == 0);
    CHECK(memcmp(overlap + 5, expected, 32) == 0);
    unsigned char sentinel[32]; memset(sentinel, 0xa5, sizeof(sentinel));
    memcpy(expected, sentinel, sizeof(expected));
    CHECK(nt_sha256(NULL, 1, sentinel) == -1);
    CHECK(memcmp(sentinel, expected, sizeof(sentinel)) == 0);
    CHECK(nt_sha256("a", 1, NULL) == -1);
#if SIZE_MAX > UINT64_MAX / 8
    CHECK(nt_sha256("a", (size_t)(UINT64_MAX / 8 + 1), sentinel) == -1);
    CHECK(memcmp(sentinel, expected, sizeof(sentinel)) == 0);
#endif
    free(data);
    printf("SHA-256 native: %u checks; NIST/RFC vectors and 264 hashlib cases\n", checks);
    return 0;
}
