/* Private state shared by waking consolidation and the dream worker. */
#ifndef NT_EXPERTS_INTERNAL_H
#define NT_EXPERTS_INTERNAL_H
#include "harness/experts.h"

enum { SEATS = NT_EXPERT_SEATS, FOUNDERS = 4, RANK = 4, WINDOW = 2, HISTORY = 16 };
#define TAU 6.28318530718f

typedef struct {
    char id[65], source_host[65];
    uint64_t turn, death_turn;
    uint32_t parent; /* one-based earlier ledger entry, 0 for a root */
    int kind;
} nt_lineage_entry;

struct nt_experts {
    int dim, layer, depth;
    char host[65], arch[64], norm_sha[65];
    nt_expert_body body;
    const gguf_file *frame_host;
    const nt_arch *decoder;
    nt_dims dims;
    float *frame;
    FILE *trace;
    nt_moe_route_fn dream_route;
    void *dream_route_user;
    uint32_t rng;
    uint64_t turns, elections;
    nt_lineage_entry *ancestors;
    int ancestor_count, ancestor_capacity;
    int32_t seat_ancestor[SEATS]; /* one-based ledger addresses; zero is vacant */
    struct {
        uint64_t cycles, choices, elections;
        uint32_t seen;
    } sleep;
    struct {
        uint64_t cycles, choices, elections, prefill, stops;
        char last[65];
    } replay;
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
int nt_lineage_founders(nt_experts *p);
int nt_lineage_birth(nt_experts *p, int parent, int child);
void nt_lineage_retire(nt_experts *p, int seat);
int nt_lineage_write(const nt_experts *p, gguf_writer *w);
int nt_lineage_read(nt_experts *p, const gguf_file *g, uint64_t turns);
int nt_lineage_adopt(nt_experts *p, const char *spore_hash, const char *source_host,
                     uint64_t turns, int width, const float *a, const float *b);
#endif
