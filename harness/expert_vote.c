/* Calibrate one body's vote scale without changing its population or clocks.
 * The raw anchor residuals preserve the scale that the QR frame discards. */
#include "harness/experts_internal.h"
#include "harness/spore_frame.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int spread(const nt_experts *p, const float *votes, const float *rows,
                   int living, double *rms) {
    double ss = 0;
    for (int k = 0; k < NT_SPORE_ANCHORS; k++) {
        float dots[SEATS];
        if (nt_linear_values(votes, (float[SEATS]){0}, rows + (size_t)k*p->dim,
                              SEATS, p->dim, dots)) return NT_E_STATE;
        double mean = 0;
        for (int e = 0; e < SEATS; e++) if (p->alive[e]) mean += dots[e];
        mean /= living;
        for (int e = 0; e < SEATS; e++) if (p->alive[e]) {
            double d = dots[e] - mean;
            ss += d*d;
        }
    }
    *rms = sqrt(ss / (NT_SPORE_ANCHORS * living));
    return isfinite(*rms) && *rms > 0 ? NT_OK : NT_E_STATE;
}

int nt_experts_calibrate_votes(nt_experts *p, double target_rms, nt_vote_calibration *out) {
    if (out) *out = (nt_vote_calibration){0};
    if (!p || !isfinite(target_rms) || target_rms <= 0) return NT_E_ARG;
    if (!p->turns || p->count || p->observed || p->choices ||
        !p->body.model || !p->body.tokenizer || !p->frame_host ||
        !p->decoder || !p->decoder->doe || p->dim < 1 ||
        p->dim != p->frame_host->embed_dim || p->depth != p->dims.n_layers ||
        p->layer != p->depth / 2 ||
        p->dim > NT_MAX_ELEMENTS / NT_SPORE_ANCHORS) return NT_E_STATE;
    int living = 0;
    for (int e = 0; e < SEATS; e++) {
        if (p->alive[e] != 0 && p->alive[e] != 1) return NT_E_STATE;
        living += p->alive[e] != 0;
    }
    if (living < 2) return NT_E_STATE;
    size_t count = (size_t)SEATS * p->dim;
    for (size_t i = 0; i < count; i++)
        if (!isfinite(p->vote[i]) || fabsf(p->vote[i]) > 1e4f) return NT_E_STATE;
    float *rows = malloc((size_t)NT_SPORE_ANCHORS * p->dim * sizeof(float));
    float *staged = malloc(count * sizeof(float));
    if (!rows || !staged) { free(rows); free(staged); return NT_E_MEMORY; }
    int rc = NT_E_STATE;
    nt_vote_calibration report = {.anchors=NT_SPORE_ANCHORS, .living=living};
    if (nt_spore_anchor_rows(&p->body, p->frame_host, p->decoder, p->dims, rows) ||
        spread(p, p->vote, rows, living, &report.before_rms)) goto done;
    report.scale = target_rms / report.before_rms;
    if (!isfinite(report.scale) || report.scale <= 0) goto done;
    if (fabs(report.scale - 1) <= 8 * FLT_EPSILON) {
        report.scale = 1;
        report.after_rms = report.before_rms;
    } else {
        for (size_t i = 0; i < count; i++) {
            double v = (double)p->vote[i] * report.scale;
            if (!isfinite(v) || fabs(v) > 1e4) goto done;
            staged[i] = (float)v;
        }
        if (spread(p, staged, rows, living, &report.after_rms)) goto done;
        memcpy(p->vote, staged, count * sizeof(float));
    }
    if (out) *out = report;
    rc = NT_OK;
done:
    free(staged); free(rows);
    return rc;
}
