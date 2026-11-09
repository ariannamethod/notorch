/* A private greedy episode on the resident decoder. The waking KV and turn
 * statistics never enter this episode. Only elected factors can learn from it. */
#include "harness/experts_internal.h"
#include "harness/spore_frame.h"
#include "harness/tokenizer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Energy of this seat's low-rank residual A(Bq) on one calibrated axis. */
static double response(const nt_experts *p, int seat, const float *q) {
    size_t offset = (size_t)seat * p->dim * RANK;
    const float *a = p->a + offset, *b = p->b + offset;
    double z[RANK] = {0}, energy = 0;
    for (int r = 0; r < RANK; r++)
        for (int i = 0; i < p->dim; i++) z[r] += (double)b[(size_t)r*p->dim+i] * q[i];
    for (int i = 0; i < p->dim; i++) {
        double y = 0;
        for (int r = 0; r < RANK; r++) y += a[(size_t)i*RANK+r] * z[r];
        energy += y*y;
    }
    return p->vitality[seat] * energy;
}

static int select_cue(const nt_experts *p, uint32_t seen, int *seat) {
    double best = -1;
    int selected = -1;
    for (int k = 0; k < NT_SPORE_ANCHORS; k++) {
        if (seen & (UINT32_C(1) << k)) continue;
        const float *q = p->frame + (size_t)k * p->dim;
        double sum = 0, largest = -1;
        int contributor = -1;
        for (int e = 0; e < SEATS; e++) {
            if (!p->alive[e]) continue;
            double energy = response(p, e, q);
            sum += energy;
            if (energy > largest) { largest = energy; contributor = e; }
        }
        if (!isfinite(sum) || contributor < 0) return -1;
        if (sum > best) { best = sum; selected = k; *seat = contributor; }
    }
    return selected;
}

static int trace_piece(FILE *out, const bpe_tokenizer *tok, int step, int id, int stop) {
    int cap = bpe_decode_token_capacity(tok, id);
    char *piece = cap > 0 ? malloc((size_t)cap) : NULL;
    if (!piece) return cap > 0 ? NT_E_MEMORY : NT_E_TOKEN;
    int n = bpe_decode_token(tok, id, piece, cap);
    flockfile(out);
    fprintf(out, "doe dream-choice step=%d id=%d stop=%d piece=\"", step, id, stop);
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c == '"' || c == '\\') fprintf(out, "\\%c", c);
        else if (c == '\n') fputs("\\n", out);
        else if (c >= 32 && c < 127) fputc(c, out);
        else fprintf(out, "\\x%02x", c);
    }
    fputs("\"\n", out);
    funlockfile(out);
    free(piece);
    return NT_OK;
}

static int generate(nt_experts *d, int cue, int limit, int eos) {
    const nt_arch *arch = d->decoder;
    const bpe_tokenizer *tok = d->body.tokenizer;
    int ids[NT_SPORE_CUE_CAP];
    int n = nt_spore_cue_ids(tok, cue, ids);
    if (!n) return NT_E_TOKEN;
    int capacity = n + limit;
    if (d->frame_host->ctx_len > 0 && capacity > d->frame_host->ctx_len) return NT_E_CAPACITY;
    kv_cache *kv = nt_model_kv_new(arch, d->body.model, d->dims, capacity);
    float *logits = malloc((size_t)d->dims.vocab * sizeof(float));
    int rc = NT_E_MEMORY;
    if (!kv || !logits) goto done;
    rc = arch->doe->forward(d->body.model, kv, ids, n, 0, logits,
                                nt_experts_attend, nt_experts_observe, d);
    for (int t = 0; !rc && t < limit; t++) {
        for (int i = 0; i < d->dims.vocab; i++)
            if (!isfinite(logits[i])) { rc = NT_E_STATE; goto done; }
        int id = nt_sample_model_token(logits, d->dims.vocab, tok, 0);
        int stop = id == eos || bpe_is_eog(tok, id);
        nt_experts_choice(d, logits[id], logits[id]);
        if (d->trace && (rc = trace_piece(d->trace, tok, t, id, stop))) break;
        if (stop) break;
        rc = arch->doe->forward(d->body.model, kv, &id, 1, n+t,
                                    t+1 == limit ? NULL : logits,
                                    nt_experts_attend, nt_experts_observe, d);
    }
done:
    free(logits);
    nt_model_kv_free(arch, d->body.model, kv);
    return rc;
}

/* Stage the joint update. A late rejection on one seat must not publish the
 * factors, cursors or RNG of an earlier seat. */
static int record_episode(nt_experts *p, nt_experts *d, uint32_t seen) {
    size_t n = (size_t)SEATS * d->dim * RANK;
    float *staged = malloc(2*n*sizeof(float));
    if (!staged) return NT_E_MEMORY;
    d->a = staged;
    d->b = staged + n;
    memcpy(d->a, p->a, n*sizeof(float));
    memcpy(d->b, p->b, n*sizeof(float));
    int rc = nt_experts_update_factors(d, 0.0025f, (float)(d->signal/d->choices));
    if (!rc) {
        memcpy(p->a, d->a, n*sizeof(float));
        memcpy(p->b, d->b, n*sizeof(float));
        memcpy(p->cursor, d->cursor, sizeof(p->cursor));
        p->rng = d->rng;
        p->sleep.cycles++;
        p->sleep.choices += d->choices;
        p->sleep.elections += d->elections;
        p->sleep.seen = seen;
    }
    free(staged);
    return rc;
}

int nt_experts_dream(nt_experts *p, int tokens, int eos, int learn) {
    if (!p || tokens < 1 || tokens > NT_DREAM_MAX_TOKENS || eos < -1) return NT_E_ARG;
    if (!p->turns || p->count || p->observed || p->choices || p->sleep.cycles == UINT64_MAX ||
        !p->body.tokenizer || eos >= bpe_n_vocab(p->body.tokenizer) || nt_experts_calibrate(p))
        return NT_E_STATE;
    uint32_t seen = p->sleep.seen == UINT32_MAX ? 0 : p->sleep.seen;
    int seat = -1, cue = select_cue(p, seen, &seat);
    if (cue < 0) return NT_E_STATE;
    /* Scalar clocks, oscillator and per-seat arrays are copied by value.
     * The owner lends its read-only factors and its disposable kernel scratch. */
    nt_experts d = *p;
    d.dreaming = 1;
    d.elections = 0;
    d.sum_in = calloc((size_t)2*d.dim, sizeof(float));
    if (!d.sum_in) return NT_E_MEMORY;
    d.sum_out = d.sum_in + d.dim;
    if (d.trace) fprintf(d.trace, "doe dream-start cycle=%llu cue=%d literal=\"%s\" seat=%d limit=%d\n",
                         (unsigned long long)d.sleep.cycles+1, cue, nt_spore_cue_text(cue), seat, tokens);
    int rc = generate(&d, cue, tokens, eos);
    if (!rc && (!d.choices || !d.count || d.observed != d.count ||
        (uint64_t)d.choices > UINT64_MAX - p->sleep.choices ||
        d.elections > UINT64_MAX - p->sleep.elections)) rc = NT_E_STATE;
    if (!rc && learn) rc = record_episode(p, &d, seen | (UINT32_C(1) << cue));
    if (!rc && d.trace) fprintf(d.trace, "doe dream-end cycle=%llu choices=%d elections=%llu learn=%d\n",
                               (unsigned long long)d.sleep.cycles+1, d.choices,
                               (unsigned long long)d.elections, !!learn);
    free(d.sum_in);
    return rc;
}
