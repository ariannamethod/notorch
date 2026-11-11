/* Observe a sealed prompt on private decoder state. Selection policy and its
 * round ledger belong to the caller; this reader changes no living state. */
#include "harness/replay_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    nt_experts view;
    nt_replay_measurement stats;
    double *proposals, *center;
    int living;
} measurement;

/* As in dream.c's anchor response, evaluate A(Bx) in double precision. The
 * unweighted living population supplies the center. Gate/vitality/injection
 * weights do not hide an expert's potential response from this measurement.
 * q==0 has an exact zero response; otherwise divide by q without a floor. */
static int response(measurement *m, const float *x) {
    const nt_experts *p = &m->view;
    size_t width = (size_t)p->dim, dr = width * RANK;
    double input_energy = 0, energy[SEATS] = {0};
    for (size_t i = 0; i < width; i++) {
        if (!isfinite(x[i])) return NT_E_STATE;
        input_energy += (double)x[i] * x[i];
    }
    if (!isfinite(input_energy)) return NT_E_STATE;
    memset(m->center, 0, width * sizeof(double));
    for (int e = 0; e < SEATS; e++) {
        if (!p->alive[e]) continue;
        const float *a = p->a + (size_t)e * dr, *b = p->b + (size_t)e * dr;
        double z[RANK] = {0}, *proposal = m->proposals + (size_t)e * width;
        for (int r = 0; r < RANK; r++)
            for (size_t i = 0; i < width; i++) z[r] += (double)b[(size_t)r * width + i] * x[i];
        for (size_t i = 0; i < width; i++) {
            double y = 0;
            for (int r = 0; r < RANK; r++) y += (double)a[i * RANK + r] * z[r];
            if (!isfinite(y)) return NT_E_STATE;
            proposal[i] = y;
            energy[e] += y * y;
            m->center[i] += y / m->living;
        }
        if (!isfinite(energy[e])) return NT_E_STATE;
    }
    if (input_energy == 0) return NT_OK;
    double total = 0, variance = 0;
    for (int e = 0; e < SEATS; e++) {
        if (!p->alive[e]) continue;
        double normalized = energy[e] / input_energy;
        m->stats.response[e] += normalized;
        total += normalized / m->living;
        const double *proposal = m->proposals + (size_t)e * width;
        double deviation = 0;
        for (size_t i = 0; i < width; i++) {
            double delta = proposal[i] - m->center[i];
            deviation += delta * delta;
        }
        variance += (deviation / input_energy) / m->living;
        if (!isfinite(m->stats.response[e])) return NT_E_STATE;
    }
    m->stats.mean_response += total;
    m->stats.mean_disagreement += variance;
    return isfinite(m->stats.mean_response) && isfinite(m->stats.mean_disagreement) ? NT_OK : NT_E_STATE;
}

static int attend(void *user, int layer, int pos, int n, int width, float *rows) {
    measurement *m = user;
    if (layer != m->view.layer) return NT_OK;
    if (width != m->view.dim || n < 1 || !rows) return NT_E_STATE;
    for (int j = 0; j < n; j++) {
        int rc = response(m, rows + (size_t)j * width);
        if (rc) return rc;
    }
    return nt_experts_attend_measured(&m->view, layer, pos, n, width, rows, &m->stats);
}

static int observe(void *user, int layer, int pos, int n, int width, float *rows) {
    measurement *m = user;
    return nt_experts_observe(&m->view, layer, pos, n, width, rows);
}

static int finalize(nt_replay_measurement *s) {
    if (!s->positions) return NT_E_STATE;
    for (int e = 0; e < SEATS; e++) {
        s->gate_mass[e] /= s->positions;
        s->response[e] /= s->positions;
        if (!isfinite(s->gate_mass[e]) || !isfinite(s->response[e])) return NT_E_STATE;
    }
    double *scalars[] = {&s->mean_max_gate, &s->mean_entropy, &s->mean_effective_seats,
        &s->mean_selected, &s->mean_nonzero, &s->mean_response, &s->mean_disagreement};
    for (size_t i = 0; i < sizeof(scalars) / sizeof(*scalars); i++) {
        *scalars[i] /= s->positions;
        if (!isfinite(*scalars[i])) return NT_E_STATE;
    }
    return NT_OK;
}

int nt_experts_measure_replay(const nt_experts *p, const nt_replay_episode *e,
                              nt_replay_measurement *out) {
    if (!p || !e || !out) return NT_E_ARG;
    if (!nt_replay_matches(p, e) || !p->turns || p->count || p->observed || p->choices ||
        !p->a || !p->b || !p->vote) return NT_E_STATE;
    measurement m;
    memset(&m, 0, sizeof(m));
    memcpy(&m.view, p, sizeof(*p));
    memcpy(m.stats.episode, nt_replay_id(e), sizeof(m.stats.episode));
    for (int seat = 0; seat < SEATS; seat++) {
        if (p->alive[seat] != 0 && p->alive[seat] != 1) return NT_E_STATE;
        if (!p->alive[seat]) continue;
        nt_expert_identity id;
        if (nt_experts_seat_identity(p, seat, &id) || !id.id[0]) return NT_E_STATE;
        memcpy(m.stats.identities[seat], id.id, sizeof(id.id));
        m.stats.alive_mask |= UINT32_C(1) << seat;
        m.living++;
    }
    if (!m.living) return NT_E_STATE;
    size_t width = (size_t)p->dim;
    if (width > SIZE_MAX / (3 * sizeof(float)) || width > SIZE_MAX / ((SEATS + 1) * sizeof(double)))
        return NT_E_CAPACITY;
    float *scratch = calloc(3 * width, sizeof(float));
    m.proposals = calloc((SEATS + 1) * width, sizeof(double));
    kv_cache *kv = nt_model_kv_new(p->decoder, p->body.model, p->dims, e->n);
    int rc = NT_E_MEMORY;
    if (!scratch || !m.proposals || !kv) goto done;
    m.center = m.proposals + SEATS * width;
    m.view.sum_in = scratch;
    m.view.sum_out = scratch + width;
    m.view.injected = scratch + 2 * width;
    m.view.update = NULL;
    m.view.trace = NULL;
    m.view.dream_route = NULL;
    m.view.dream_route_user = NULL;
    m.view.dreaming = 2;
    m.view.elections = 0;
    memset(m.view.hits, 0, sizeof(m.view.hits));
    m.view.signal = 0;
    rc = NT_OK;
    int media = 0;
    for (int pos = 0; !rc && pos < e->n;) {
        int n = e->n - pos < NT_PREFILL_CHUNK ? e->n - pos : NT_PREFILL_CHUNK;
        if (e->n_media) {
            const float *rows[NT_PREFILL_CHUNK] = {0};
            while (media < e->n_media && e->positions[media] < pos + n) {
                rows[e->positions[media] - pos] = e->media + (size_t)media * width;
                media++;
            }
            rc = p->decoder->doe->forward_embd(p->body.model, kv, e->ids + pos, rows,
                                               n, pos, NULL, NULL, attend, observe, &m);
        } else {
            rc = p->decoder->doe->forward(p->body.model, kv, e->ids + pos, n, pos,
                                          NULL, attend, observe, &m);
        }
        pos += n;
    }
    if (!rc && (m.stats.positions != (uint64_t)e->n || m.view.count != e->n ||
        m.view.observed != e->n || m.view.elections != (uint64_t)e->n || m.view.choices)) rc = NT_E_STATE;
    if (!rc) rc = finalize(&m.stats);
    if (!rc) memcpy(out, &m.stats, sizeof(*out));
done:
    nt_model_kv_free(p->decoder, p->body.model, kv);
    free(m.proposals);
    free(scratch);
    return rc;
}
