#ifndef NT_HARNESS_TOKENIZER_H
#define NT_HARNESS_TOKENIZER_H
#include "gguf.h"
#include "examples/bpe.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Prefer an explicit tokenizer, then the host vocabulary, then tokenizer.gguf
 * in the host's directory. A model directory owns its companion. With no
 * vocabulary, success returns NULL for callers taking token IDs directly.
 * vocab is the decoder's output width, or 0 if unknown in a tokenize-only run.
 * On success the caller owns *tok; an error string has static storage. */
const char *nt_model_tokenizer_open(const gguf_file *host, const char *host_path,
                                    int vocab, const char *override_path,
                                    bpe_tokenizer **tok, int *eos);

/* Draw from decodable IDs. Without a tokenizer all model rows are available.
 * As in sample(), positive temperature uses the logits as scratch. */
int nt_sample_model_token(float *logits, int vocab, const bpe_tokenizer *tok, float temp);
#ifdef __cplusplus
}
#endif
#endif
