/* A small native word language model for the SPA text host.
 * Copyright (C) 2026 Arianna Method contributors; LGPL-3.0-or-later. */
#ifndef SPA_TEXT_BODY_H
#define SPA_TEXT_BODY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct spa_text_body spa_text_body;
/* 16-dimensional embedding -> tanh -> vocabulary projection.
 * Training uses the upstream notorch tape; callers serialize training with
 * other tape users. Inference and sampling never enter the global tape.
 * Every body is owned by one host; calls on one body must be serialized. */
spa_text_body *spa_text_create(uint32_t vocabulary, uint64_t seed);
void spa_text_free(spa_text_body *body);
uint32_t spa_text_vocabulary(const spa_text_body *body);
uint32_t spa_text_dimension(void);
/* Full-batch next-word cross entropy, native SGD. count 2..4096, epochs
 * 1..10000, rate (0,4]. losses[0/1] are pre/post-training corpus losses.
 * Training commits weights only on success; generator RNG is preserved. */
int spa_text_train(spa_text_body *body, const uint32_t *tokens, size_t count,
                   uint32_t epochs, float rate, float losses[2]);
int spa_text_embeddings(const spa_text_body *body, float *out, size_t capacity);
/* Generate at most capacity tokens (1..64), seeded by one context word.
 * Include the end token when selected. Successful generation advances the
 * native PCG stream; failure preserves the body. Returns token count or -1. */
int spa_text_generate(spa_text_body *body, uint32_t context, uint32_t end,
                      float temperature, uint32_t *out, size_t capacity);
/* Canonical little-endian checkpoint: weights, generator RNG, checksum.
 * Restore returns a new owned body after complete validation. */
size_t spa_text_snapshot_size(const spa_text_body *body);
int spa_text_snapshot(const spa_text_body *body, void *out, size_t capacity);
spa_text_body *spa_text_restore(const void *data, size_t size);
#ifdef __cplusplus
}
#endif
#endif
