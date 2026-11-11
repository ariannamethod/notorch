/* Immutable prompt snapshots, owned separately from a living parliament. */
#ifndef NT_REPLAY_INTERNAL_H
#define NT_REPLAY_INTERNAL_H

#include "harness/experts_internal.h"

struct nt_replay_episode {
    char host[65], id[65];
    int width, vocab, n, n_media, n_suppressed, n_stops;
    int *ids, *positions, *suppressed, *stops;
    float *media; /* [n_media][width], in ascending prompt-position order */
};

/* The caller owns the result. A worker clones before accepting ownership. */
nt_replay_episode *nt_replay_clone(const nt_replay_episode *e);
/* Boolean: content and shape are valid for this exact resident body. */
int nt_replay_matches(const nt_experts *p, const nt_replay_episode *e);

#endif
