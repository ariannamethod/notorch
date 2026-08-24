/* runtime.h — the parts of running a transformer that no architecture owns.
 *
 * A weight as it sits in the file, a KV cache, the scalar pieces every family
 * uses, and the two matvec entry points that decide packed vs int8 vs BLAS.
 * Architectures live next door in arch_*.c and are the only thing that changes
 * when a new model family arrives.
 *
 * Lifted from examples/infer_llama.c, which stays where it is as the reference
 * this was measured against. */
#ifndef NT_HARNESS_RUNTIME_H
#define NT_HARNESS_RUNTIME_H

#include "gguf.h"
#include "notorch.h"

/* Prompt positions carried through the weights together. Wider amortizes the
 * weight read further; past the caches it starts paying it back in misses, and
 * 32 is where the batched kernel's own tile sits. */
#ifndef NT_PREFILL_CHUNK
#define NT_PREFILL_CHUNK 32
#endif

/* A weight as it sits in the file, plus the one thing worth deciding once.
 *
 * Expanding every tensor to f32 at load is 6 GB for a 1.5B Q4_0 whose packed
 * form is 1.1 GB, and on an 8 GB phone that is not a slowdown but a reboot.
 * The packed bytes are already resident — gguf_open reads the file into one
 * buffer — so a weight here is a pointer into it and nothing is copied.
 *
 * f32 is the escape hatch: a dtype with no packed kernel gets expanded, alone,
 * and the matvec falls through to BLAS for that one tensor. Probing at load
 * rather than per call keeps the decision off the hot path. */
typedef struct {
    const uint8_t *q;   /* into gf->data; NULL only if the tensor is absent */
    float *f32;         /* expanded copy, allocated only when q has no kernel */
    int dtype, rows, cols;
    int use_i8;         /* the integer path applies to this dtype and shape */
} wt;

int  wt_load(wt *w, gguf_file *gf, const char *name);

typedef struct {
    float *k, *v;
    int max_seq, n_layers, kv_dim;
} kv_cache;

kv_cache *kv_new(int n_layers, int max_seq, int kv_dim);
void      kv_free(kv_cache *kv);

/* out[rows] = W[rows,cols] @ x[cols] — the only matmul shape decode asks for. */
void qmv(float *out, const wt *w, const float *x);
/* n activation rows through one weight matrix, weights read once for the group
 * where the batched kernel has the dtype and row-at-a-time where it does not. */
void qmm(float *out, const wt *w, const float *X, int n);

void  mm_t(float *C, const float *A, const float *B, int m, int k, int n);
float dot_f32(const float *a, const float *b, int n);
void  axpy_f32(float *y, float alpha, const float *x, int n);
void  rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void  softmax(float *x, int n);
void  add_bias(float *x, const float *bias, int n);

/* Two rotation conventions share the name RoPE and differ only in which pairs
 * of lanes rotate together, which makes a mismatch quiet rather than fatal: the
 * model still emits fluent text, it just answers the wrong question. NORM pairs
 * adjacent lanes (2i, 2i+1) and is what llama-architecture GGUFs carry; NEOX
 * pairs a lane with its opposite half (i, i + hd/2) and is what qwen2 and
 * friends carry. The architecture picks, because one binary reads both. */
void rope(float *x, int pos, int head_dim, float freq_base, int neox);

int    sample(float *logits, int n, float temp);
double now_ms(void);

/* ── Section timing ──────────────────────────────────────────────────────────
 * Off unless NT_PROFILE is set, because the answer to "where does prefill go"
 * stopped being obvious once the weights were read once per chunk instead of
 * once per token. Two timestamps per section per chunk, so at 24 layers the
 * clock reads cost nothing next to the work they bracket, and with the flag
 * unset each call is one predicted branch. Reports go to stderr with the rest
 * of the diagnostics. */
enum { PF_EMBED, PF_NORM, PF_QKV, PF_ROPE, PF_ATTN, PF_PROJ, PF_FFN, PF_SILU,
       PF_RESID, PF_HEAD, PF_N };

extern int pf_on;

double pf_mark(void);
void   pf_add(int slot, double t0);
void   pf_reset(void);
void   pf_report(const char *phase, double wall_ms);

#endif
