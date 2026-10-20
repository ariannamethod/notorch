/* Shared state of the standalone SimpleLLM / Chuck training life. */
#ifndef NOTORCH_CHUCK_TRAINER_STATE_H
#define NOTORCH_CHUCK_TRAINER_STATE_H

#define TRAINER_MAX_VOCAB 4096
#ifdef NT_CHUCK_Q_TRAINER
#include "chuck_q_tokenizer.h"
#endif

typedef struct {
    body model;
    nt_chuck_architect architect;
    uint32_t *vocab;              /* sorted Unicode scalar values, owned */
    uint32_t vocab_size;
#ifdef NT_CHUCK_Q_TRAINER
    q_tokenizer tokenizer;
#endif
    unsigned char corpus_sha256[32];
    uint64_t token_count, split;
    uint64_t step, windows;
    uint32_t seed, acquire_every;
    float lr;
    uint64_t acquisitions, fit_updates, branch_updates;
    uint64_t action_counts[4];    /* legacy, hold, brake, push */
} trainer_state;

#endif
