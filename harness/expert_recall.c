/* Keep the incoming shared coordinates and recall this body's earlier
 * orthogonal component. Birth addresses select experts independently of slots. */
#include "harness/experts_internal.h"
#include "harness/spore_frame.h"
#include <stdlib.h>
#include <string.h>

int nt_experts_recall_body(nt_experts *incoming, const nt_experts *local, int *matched) {
    if (matched) *matched = 0;
    if (!incoming || !local) return NT_E_ARG;
    if (!incoming->turns || !local->turns ||
        incoming->count || incoming->observed || incoming->choices ||
        local->count || local->observed || local->choices ||
        incoming->dim != local->dim || incoming->layer != local->layer ||
        strcmp(incoming->host, local->host)) return NT_E_STATE;

    int from[SEATS], count = 0;
    for (int e = 0; e < SEATS; e++) {
        from[e] = -1;
        if (!incoming->alive[e]) continue;
        nt_expert_identity id;
        if (nt_experts_seat_identity(incoming, e, &id)) return NT_E_STATE;
        from[e] = nt_experts_find_identity(local, id.id);
        count += from[e] >= 0;
    }
    if (!count || incoming == local) {
        if (matched) *matched = count;
        return NT_OK;
    }
    if (nt_experts_calibrate(incoming)) return NT_E_STATE;

    size_t dr = (size_t)incoming->dim * RANK;
    size_t factors = SEATS * dr, votes = (size_t)SEATS * incoming->dim;
    float *staged = malloc((2 * factors + votes) * sizeof(float));
    if (!staged) return NT_E_MEMORY;
    float *a = staged, *b = a + factors, *vote = b + factors;
    memcpy(a, incoming->a, factors * sizeof(float));
    memcpy(b, incoming->b, factors * sizeof(float));
    memcpy(vote, incoming->vote, votes * sizeof(float));
    const float *frame = incoming->frame;
    int rc = NT_OK;
    for (int e = 0; e < SEATS; e++) {
        if (from[e] < 0) continue;
        size_t dst = (size_t)e * dr, src = (size_t)from[e] * dr;
        size_t dv = (size_t)e * incoming->dim, sv = (size_t)from[e] * incoming->dim;
        memcpy(a + dst, local->a + src, dr * sizeof(float));
        memcpy(b + dst, local->b + src, dr * sizeof(float));
        memcpy(vote + dv, local->vote + sv, (size_t)incoming->dim * sizeof(float));
        if (nt_spore_project(frame, incoming->dim, incoming->vote + dv, 1,
                             frame, incoming->dim, vote + dv, 1)) { rc = NT_E_STATE; break; }
        for (int r = 0; r < RANK; r++) {
            if (nt_spore_project(frame, incoming->dim, incoming->a + dst + r, RANK,
                                 frame, incoming->dim, a + dst + r, RANK) ||
                nt_spore_project(frame, incoming->dim, incoming->b + dst + (size_t)r * incoming->dim, 1,
                                 frame, incoming->dim, b + dst + (size_t)r * incoming->dim, 1)) {
                rc = NT_E_STATE;
                break;
            }
        }
        if (rc) break;
    }
    if (!rc) {
        memcpy(incoming->a, a, factors * sizeof(float));
        memcpy(incoming->b, b, factors * sizeof(float));
        memcpy(incoming->vote, vote, votes * sizeof(float));
        if (matched) *matched = count;
        if (incoming->trace) fprintf(incoming->trace,
            "doe body-recall matched=%d anchors=%d host=%s\n", count, NT_SPORE_ANCHORS, incoming->host);
    }
    free(staged);
    return rc;
}
