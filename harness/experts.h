/* A parliament for one residual stream. One owner per conversation. */
#ifndef NT_HARNESS_EXPERTS_H
#define NT_HARNESS_EXPERTS_H

#include "harness/arch.h"
#include "examples/bpe.h"
#include <stdio.h>

typedef struct nt_experts nt_experts;

/* A fresh parliament starts at the middle decoder layer. A spore restores a
 * completed turn for these exact host bytes. Destroy the session after any
 * nonzero hook/finish result; a failed turn must never be saved or continued. */
nt_experts *nt_experts_open(const gguf_file *host, const char *spore, FILE *trace);
/* Borrow the host, its loaded DOE-capable CPU decoder and tokenizer through
 * close. Export and foreign-host import sample the body with private KV.
 * The original open handles fresh and exact-host state. */
typedef struct {
    void *model;
    const bpe_tokenizer *tokenizer;
} nt_expert_body;
nt_experts *nt_experts_open_on_body(const gguf_file *host, const char *spore, FILE *trace,
                                    const nt_expert_body *body);
void nt_experts_free(nt_experts *p);
int nt_experts_attend(void *user, int layer, int pos, int n, int width, float *rows);
int nt_experts_observe(void *user, int layer, int pos, int n, int width, float *rows);
void nt_experts_choice(nt_experts *p, float best_logit, float chosen_logit);
int nt_experts_finish(nt_experts *p, int learn);
/* One outstanding consolidation per session. Successful submission transfers
 * the sole owner to a joinable worker and clears *owner. Await before the next
 * forward or spore export. A rejected numerical update destroys the state.
 * Initialize *job to NULL and keep the trace stream open through await. */
typedef struct nt_consolidation nt_consolidation;
int nt_experts_submit(nt_experts **owner, nt_consolidation **job);
/* From a completed generation opened with open_on_body, let the body
 * generate 1..32 greedy choices on private KV. Pass its EOS id, or -1 when
 * absent. With learn=0 the same episode can be inspected through the trace.
 * On failure no dream learning or counters are committed. */
enum { NT_DREAM_MAX_TOKENS = 32 };
int nt_experts_dream(nt_experts *p, int tokens, int eos, int learn);
/* Transfer sole ownership to the same worker as submit. It first consolidates
 * a pending waking turn, if present, then learns from one dream. The borrowed
 * model, tokenizer, host and trace must outlive await. */
int nt_experts_submit_dream(nt_experts **owner, nt_consolidation **job, int tokens, int eos);
int nt_experts_await(nt_experts **owner, nt_consolidation **job);
/* Replace only one living seat's A and B from the same host and slot. Both
 * parliaments must be at a completed turn boundary. Returns NT_OK or a refusal
 * leaving both states untouched. Self-copy is a successful no-op. */
int nt_experts_copy_factors(nt_experts *recipient, const nt_experts *donor, int slot);
/* Publish a completed turn as a new GGUF. An existing path is preserved. */
int nt_experts_save(nt_experts *p, const char *path);
/* Also store the source body's calibrated frame for a later model switch. */
int nt_experts_save_portable(nt_experts *p, const char *path);
#endif
