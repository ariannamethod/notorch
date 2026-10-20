/* Shared state of the standalone SimpleLLM / Chuck training life. */
#ifndef NOTORCH_CHUCK_TRAINER_STATE_H
#define NOTORCH_CHUCK_TRAINER_STATE_H

#define TRAINER_MAX_VOCAB 4096

typedef struct {
    body model;
    nt_chuck_architect architect;
    uint32_t *vocab;              /* sorted Unicode scalar values, owned */
    uint32_t vocab_size;
    unsigned char corpus_sha256[32];
    uint64_t token_count, split;
    uint64_t step, windows;
    uint32_t seed, acquire_every;
    float lr;
    uint64_t acquisitions, fit_updates, branch_updates;
    uint64_t action_counts[4];    /* legacy, hold, brake, push */
} trainer_state;

#endif
