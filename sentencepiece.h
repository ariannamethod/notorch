/* Canonical deterministic SentencePiece Unigram inference in portable C.
 * ModelProto supplies its vocabulary and compiled normalization rules.
 * No SentencePiece/protobuf library is required at runtime. */
#ifndef NT_SENTENCEPIECE_H
#define NT_SENTENCEPIECE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NT_SPM_MAX_MODEL_BYTES ((size_t)64 * 1024 * 1024)
#define NT_SPM_MAX_TEXT_BYTES ((size_t)1024 * 1024)
#define NT_SPM_MAX_VOCAB 1048576
#define NT_SPM_MAX_PIECE_BYTES 7999

typedef struct nt_spm_model nt_spm_model;

typedef struct {
    int id;
    size_t offset;                 /* bytes in result.normalized */
    size_t length;
} nt_spm_piece;

typedef struct {
    char *normalized;              /* owned, NUL-terminated UTF-8 */
    size_t normalized_bytes;
    nt_spm_piece *pieces;           /* owned; unknown runs are fused */
    size_t count;
} nt_spm_result;

/* Models are immutable. Encode may run concurrently on the same model.
 * The caller keeps the model alive through every concurrent call.
 * Both loaders own their complete data; load_memory copies borrowed bytes.
 * NULL means failure. Optional error buffers receive a NUL-terminated reason.
 * Supports deterministic UNIGRAM with NORMAL, UNKNOWN, CONTROL, USER_DEFINED,
 * UNUSED pieces, at least one NORMAL piece, embedded compiled charsmap and
 * normalizer whitespace flags.
 * Byte fallback and other model algorithms are rejected explicitly. */
nt_spm_model *nt_spm_load(const char *path, char *error, size_t error_cap);
nt_spm_model *nt_spm_load_memory(const void *data, size_t size,
                                 char *error, size_t error_cap);
void nt_spm_free(nt_spm_model *model);
int nt_spm_n_vocab(const nt_spm_model *model);

/* Lowercase SHA-256 hex of the exact owned ModelProto bytes parsed at load.
 * Borrowed 64-character string, immutable until model destruction; NULL for
 * a NULL model. File replacement after load cannot alter this identity. */
const char *nt_spm_identity(const nt_spm_model *model);

/* Input is borrowed for this call only. Embedded NUL is rejected. Malformed
 * UTF-8 bytes become U+FFFD, as in SentencePiece normalization. Input and
 * normalized output each have a 1 MiB cap. No BOS/EOS tokens are inserted.
 * On success returns 0 and publishes a complete owned result to *out.
 * On failure returns -1 and leaves *out unchanged. The caller supplies an
 * empty destination, or frees its prior result before a successful overwrite.
 * A known piece's span is its normalized spelling; an unknown piece's span
 * preserves the complete normalized surface of the consecutive unknown run.
 * Empty text succeeds with an empty result. */
int nt_spm_encode(const nt_spm_model *model, const char *text, size_t bytes,
                  nt_spm_result *out, char *error, size_t error_cap);
void nt_spm_result_free(nt_spm_result *result);

#ifdef __cplusplus
}
#endif
#endif
