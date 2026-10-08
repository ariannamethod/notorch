/* Allocation-free SHA-256 over caller-owned bytes (FIPS 180-4). */
#ifndef NT_SHA256_H
#define NT_SHA256_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 and writes exactly 32 digest bytes, or -1 with output unchanged.
 * NULL data is valid only for an empty message. Length must fit in the SHA-256
 * 64-bit bit count. All scratch is local; calls can run concurrently.
 * The digest may overlap input: input is consumed before output is published. */
int nt_sha256(const void *data, size_t bytes, unsigned char digest[32]);

#ifdef __cplusplus
}
#endif
#endif
