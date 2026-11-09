/* Private state shared by waking consolidation and the dream worker. */
#ifndef NT_EXPERTS_INTERNAL_H
#define NT_EXPERTS_INTERNAL_H
#include "harness/experts.h"

enum { SEATS = 6, FOUNDERS = 4, RANK = 4, WINDOW = 2, HISTORY = 16 };
#define TAU 6.28318530718f

struct nt_experts {
    int dim, layer, depth;
    char host[65], arch[64], norm_sha[65];
    nt_expert_body body;
    const gguf_file *frame_host;
    float *frame;
    FILE *trace;
    uint32_t rng;
    uint64_t turns, elections;
    struct {
        uint64_t cycles, choices, elections;
        uint32_t seen;
    } sleep;
    int dreaming;  /* set only in the private dream episode */
    float consensus;
    float *a, *b, *vote;
    float alive[SEATS], vitality[SEATS], frequency[SEATS], cursor[SEATS];
    float age[SEATS], low[SEATS], history[HISTORY];
    /* The history is oldest first and zero-padded at birth. */
    int count, observed, choices, hits[SEATS];
    double signal;
    float *sum_in, *sum_out, *injected, *update;
};

int nt_experts_calibrate(nt_experts *p);
int nt_experts_update_factors(nt_experts *p, float learning_rate, float signal);
#endif
