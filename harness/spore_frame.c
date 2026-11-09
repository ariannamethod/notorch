#include "harness/spore_frame.h"
#include "harness/arch_models.h"
#include "sha256.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The order, bytes, tokenizer mode, hook and QR sign convention are part of
 * the file format. Each literal begins a fresh raw sequence at position zero.
 * The exposure and test vocabulary are independent of these calibration words. */
static const char *const anchors[NT_SPORE_ANCHORS] = {
    " river", " glass", " seven", " green", " bread", " music", " north", " sleep",
    " iron", " cloud", " circle", " paper", " sweet", " walk", " winter", " clock",
    " horse", " salt", " red", " light", " open", " hand", " stone", " leaf",
    " silver", " quiet", " ocean", " round", " fire", " bird", " small", " wood"
};

typedef struct { int layer, width; float *row; } capture;
enum { CAPTURED = -1 };

static int capture_last(void *user, int layer, int pos, int n, int width, float *rows) {
    capture *c = user;
    if (layer != c->layer) return NT_OK;
    if (pos != 0 || n < 1 || width != c->width) return NT_E_STATE;
    memcpy(c->row, rows + (size_t)(n-1)*width, (size_t)width * sizeof(float));
    return CAPTURED;
}

static double dot(const float *a, const float *b, int stride, int n) {
    double v = 0;
    for (int i = 0; i < n; i++) v += (double)a[i] * b[(size_t)i*stride];
    return v;
}

static int orthogonalize(float *frame, int index, int width) {
    float *v = frame + (size_t)index * width;
    double original = dot(v, v, 1, width);
    /* Two modified Gram-Schmidt passes keep near-parallel residuals orthogonal. */
    for (int pass = 0; pass < 2; pass++) {
        for (int j = 0; j < index; j++) {
            const float *q = frame + (size_t)j * width;
            double c = dot(q, v, 1, width);
            for (int i = 0; i < width; i++) v[i] -= c * q[i];
        }
    }
    double norm = sqrt(dot(v, v, 1, width));
    if (!isfinite(norm) || !(norm > sqrt(original) * 1e-5)) return -1;
    for (int i = 0; i < width; i++) v[i] /= norm;
    return 0;
}

const char *nt_spore_cue_text(int index) {
    return index >= 0 && index < NT_SPORE_ANCHORS ? anchors[index] : NULL;
}

int nt_spore_cue_ids(const bpe_tokenizer *tok, int index, int ids[NT_SPORE_CUE_CAP]) {
    const char *literal = nt_spore_cue_text(index);
    if (!literal || !tok || !ids) return 0;
    int n = bpe_encode_raw(tok, literal, ids, NT_SPORE_CUE_CAP);
    if (n < 1 || n >= NT_SPORE_CUE_CAP) return 0;
    char bytes[64];
    int used = 0;
    for (int i = 0; i < n; i++) {
        int k = bpe_decode_token(tok, ids[i], bytes + used, (int)sizeof(bytes)-used);
        if (!k || used+k >= (int)sizeof(bytes)-1) return 0;
        used += k;
    }
    return (size_t)used == strlen(literal) && !memcmp(bytes, literal, used) ? n : 0;
}

/* The ordered per-channel RMSNorm weights survive requantization of the linear
 * maps. Hash the actual dense decoder's f32 vectors, including its final norm. */
static int hash_norm(unsigned char chain[64], const llama_model *m, const char *name,
                     const float *norm) {
    int index = gguf_find_tensor(m->gf, name);
    if (!norm || index < 0) return -1;
    const gguf_tensor_info *t = &m->gf->tensors[index];
    if (t->ndim != 1 || t->shape[0] != (unsigned)m->embed) return -1;
    for (int i = 0; i < m->embed; i++) if (!isfinite(norm[i])) return -1;
    return nt_sha256(norm, (size_t)m->embed*sizeof(float), chain+32) ||
           nt_sha256(chain, 64, chain);
}

static int norm_fingerprint(const llama_model *m, char out[65]) {
    unsigned char chain[64] = {0};
    char name[64];
    for (int l = 0; l < m->n_layers; l++) {
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        if (hash_norm(chain, m, name, m->layers[l].attn_norm)) return -1;
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        if (hash_norm(chain, m, name, m->layers[l].ffn_norm)) return -1;
    }
    if (hash_norm(chain, m, "output_norm.weight", m->out_norm)) return -1;
    for (int i = 0; i < 32; i++) snprintf(out+2*i, 3, "%02x", chain[i]);
    return 0;
}

int nt_spore_frame_build(const nt_expert_body *body, const gguf_file *host, float *frame,
                         char norm_sha[65]) {
    if (!host || !body || !body->model || !body->tokenizer) return -1;
    const llama_model *m = body->model;
    int width = host->embed_dim, layer = host->n_layers / 2;
    if (m->gf != host || width < NT_SPORE_ANCHORS ||
        bpe_n_vocab(body->tokenizer) > m->vocab || norm_fingerprint(m, norm_sha))
        return -1;
    kv_cache *kv = kv_new(m->n_layers, NT_SPORE_CUE_CAP, m->kv_dim);
    if (!kv) return -1;
    int rc = -1;
    for (int k = 0; k < NT_SPORE_ANCHORS; k++) {
        int ids[NT_SPORE_CUE_CAP];
        int n = nt_spore_cue_ids(body->tokenizer, k, ids);
        capture c = {layer, width, frame + (size_t)k * width};
        if (!n || (host->ctx_len > 0 && n > host->ctx_len) ||
            nt_llama_forward_hooks(body->model, kv, ids, n, 0, NULL,
                                   capture_last, NULL, &c) != CAPTURED ||
            orthogonalize(frame, k, width)) goto done;
    }
    rc = nt_spore_frame_valid(frame, width) ? 0 : -1;
done:
    kv_free(kv);
    return rc;
}

int nt_spore_frame_valid(const float *frame, int width) {
    for (int i = 0; i < NT_SPORE_ANCHORS; i++)
        for (int j = 0; j <= i; j++) {
            double v = dot(frame + (size_t)i*width, frame + (size_t)j*width, 1, width);
            if (!isfinite(v) || fabs(v - (i == j)) > 1e-4) return 0;
        }
    return 1;
}

double nt_spore_frame_min_cos(const float *left, const float *right, int width) {
    double minimum = 1;
    for (int i = 0; i < NT_SPORE_ANCHORS; i++) {
        size_t offset = (size_t)i * width;
        minimum = fmin(minimum, dot(left + offset, right + offset, 1, width));
    }
    return minimum;
}

int nt_spore_project(const float *source_frame, int source_width,
                     const float *source, int source_stride,
                     const float *target_frame, int target_width,
                     float *target, int target_stride) {
    double change[NT_SPORE_ANCHORS];
    for (int k = 0; k < NT_SPORE_ANCHORS; k++)
        change[k] = dot(source_frame + (size_t)k*source_width, source, source_stride, source_width)
                  - dot(target_frame + (size_t)k*target_width, target, target_stride, target_width);
    for (int i = 0; i < target_width; i++) {
        size_t offset = (size_t)i * target_stride;
        double v = target[offset];
        for (int k = 0; k < NT_SPORE_ANCHORS; k++)
            v += change[k] * target_frame[(size_t)k*target_width + i];
        if (!isfinite(v) || fabs(v) > 1e4) return -1;
        target[offset] = (float)v;
    }
    return 0;
}
