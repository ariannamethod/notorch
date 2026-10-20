/* Q's rrpram3_janus3 substrate on the upstream tape.
 * Reference: ariannamethod/q@f5d00a36ecfcdb5e655e1576770f03d06d900e04,
 * train_q_notorch.py, ariannamethod/notorch_nn.py and postgpt_q.c.
 * Token output is tied; normalization has no learned parameters. */
#ifndef NOTORCH_CHUCK_Q_BODY_H
#define NOTORCH_CHUCK_Q_BODY_H

#include <float.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef NT_CHUCK_ARCHITECT_TEST
#define Q_LAYERS 1
#define Q_R_HEADS 1
#define Q_J_HEADS 1
#define Q_MAX_CONTEXT 8
#else
#define Q_LAYERS 3
#define Q_R_HEADS 3
#define Q_J_HEADS 3
#define Q_MAX_CONTEXT 128
#endif
#define Q_MECHANISMS 2
#define Q_LAYER_PARAMS 8

static int q_body_params(void) { return 2 + Q_LAYER_PARAMS * Q_LAYERS; }

static int q_body_shape(int slot, int vocab, int shape[2], int *ndim) {
    if (slot < 0 || slot >= q_body_params() || vocab < 2 || vocab > 1280) return 0;
    *ndim = 2; shape[1] = DIM;
    if (slot == 0) shape[0] = vocab;
    else if (slot == 1) shape[0] = Q_MAX_CONTEXT;
    else switch ((slot - 2) % Q_LAYER_PARAMS) {
        case 0: *ndim = 1; shape[0] = Q_R_HEADS * DIM * Q_MAX_CONTEXT; shape[1] = 0; break;
        case 1: shape[0] = Q_R_HEADS * HD; break;
        case 2: case 3: shape[0] = Q_J_HEADS * HD; break;
        case 4: shape[0] = Q_MECHANISMS; shape[1] = DIM + 1; break;
        case 5: shape[0] = DIM; break;
        case 6: shape[0] = 4 * DIM; break;
        case 7: shape[0] = DIM; shape[1] = 4 * DIM; break;
    }
    return 1;
}

static void body_init_vocab(body *m, const char *name, int vocab) {
    (void)name;
    memset(m, 0, sizeof(*m));
    m->name = "q-rrpram3-janus3";
    m->vocab = vocab > 0 ? vocab : 1280;
    m->layers = Q_LAYERS; m->hidden = 4 * DIM; m->positional = 1;
    for (int slot = 0; slot < q_body_params(); ++slot) {
        int shape[2], ndim;
        if (!q_body_shape(slot, m->vocab, shape, &ndim)) die("invalid Q body shape");
        nt_tensor *p = nt_tensor_new_shape(shape, ndim);
        if (!p || m->count == MAX_PARAMS) die("Q parameter allocation failed");
        int part = slot < 2 ? -1 : (slot - 2) % Q_LAYER_PARAMS;
        if (slot < 2 || part == 0 || part == 4) nt_tensor_rand(p, .02f);
        else nt_tensor_xavier(p, shape[1], shape[0]);
        m->param[m->count++] = p; m->elements += p->len;
    }
}

static int q_body_constant(int size, float fill) {
    nt_tensor *tensor = nt_tensor_new(size);
    if (!tensor) die("Q constant allocation failed");
    nt_tensor_fill(tensor, fill);
    int index = nt_tape_record(tensor, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tensor);
    if (index < 0) die("Q constant tape failed");
    return index;
}

static int body_forward(body *m, const uint32_t *data, size_t offset, int train) {
    nt_tape_start(); nt_train_mode(train);
    int p[MAX_PARAMS];
    for (int i = 0; i < m->count; ++i) p[i] = nt_tape_param(m->param[i]);
    int gamma_d = q_body_constant(DIM, 1), gamma_j = q_body_constant(Q_J_HEADS * HD, 1);
    int ones = q_body_constant(CTX, 1);
    nt_tensor *tokens = nt_tensor_new(CTX), *targets = nt_tensor_new(CTX);
    if (!tokens || !targets) die("Q token allocation failed");
    for (int t = 0; t < CTX; ++t) {
        tokens->data[t] = (float)data[offset + t];
        targets->data[t] = (float)data[offset + t + 1];
    }
    int ti = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
    int yi = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    int h = nt_seq_embedding(p[0], p[1], ti, CTX, DIM), pi = 2;
    for (int layer = 0; layer < Q_LAYERS; ++layer) {
        int x = nt_seq_rmsnorm(h, gamma_d, CTX, DIM);
        int wr = p[pi++], vr = nt_seq_linear(p[pi++], x, CTX);
        int rr = nt_rrpram_attention(wr, x, vr, CTX, DIM, Q_R_HEADS, HD);
        int wj = nt_seq_linear(p[pi++], x, CTX), vj = nt_seq_linear(p[pi++], x, CTX);
        /* Native Q uses j / sqrt(sum(j*j) + 1e-8). The tape RMSNorm
         * epsilon is 1e-6: scaling its input by sqrt(B*100), then dividing
         * its output by sqrt(B), preserves Q's native epsilon and gradient. */
        int scaled = nt_scale(wj, sqrtf((float)(Q_J_HEADS * HD) * 100.0f));
        int normalized = nt_seq_rmsnorm(scaled, gamma_j, CTX, Q_J_HEADS * HD);
        normalized = nt_scale(normalized, 1.0f / sqrtf((float)(Q_J_HEADS * HD)));
        int janus = nt_mul(vj, normalized);
        int augmented = nt_concat(x, ones, CTX);
        int gates = nt_sigmoid(nt_seq_linear(p[pi++], augmented, CTX));
        int r = nt_seq_gate(rr, gates, CTX, Q_MECHANISMS, 0);
        int j = nt_seq_gate(janus, gates, CTX, Q_MECHANISMS, 1);
        h = nt_add(h, nt_seq_linear(p[pi++], nt_concat(r, j, CTX), CTX));
        x = nt_seq_rmsnorm(h, gamma_d, CTX, DIM);
        int up = nt_relu(nt_seq_linear(p[pi++], x, CTX));
        h = nt_add(h, nt_seq_linear(p[pi++], up, CTX));
    }
    int normalized = nt_seq_rmsnorm(h, gamma_d, CTX, DIM);
    int logits = nt_seq_linear(p[0], normalized, CTX);
    int loss = nt_seq_cross_entropy(logits, yi, CTX, m->vocab);
    if (pi != m->count || loss < 0) die("Q forward tape failed");
    return loss;
}

static int q_export_word(FILE *file, uint32_t word) {
    unsigned char bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = (unsigned char)(word >> (8 * i));
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int q_export_float(FILE *file, float value) {
    uint32_t word;
    if (!isfinite(value)) return -1;
    memcpy(&word, &value, sizeof(word));
    return q_export_word(file, word);
}

/* QPTQ v1 consumed by Q's native, Python and browser engines. The gate's
 * augmented final column is exported as gbs after all ordinary gws columns. */
static int q_body_export(const body *m, const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!m || !path || !*path || m->count != q_body_params() || sizeof(float) != 4 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) goto invalid;
    for (int i = 0; i < m->count; ++i) {
        int shape[2], ndim;
        nt_tensor *p = m->param[i];
        if (!p || !q_body_shape(i, m->vocab, shape, &ndim) || p->ndim != ndim) goto invalid;
        for (int d = 0; d < ndim; ++d) if (p->shape[d] != shape[d]) goto invalid;
        for (int k = 0; k < p->len; ++k) if (!isfinite(p->data[k])) goto invalid;
    }
    size_t length = strlen(path);
    if (length > SIZE_MAX - 16) goto invalid;
    char *temporary = malloc(length + 16), *parent = strdup(path);
    if (!temporary || !parent) { free(temporary); free(parent); goto invalid; }
    snprintf(temporary, length + 16, "%s.tmp.XXXXXX", path);
    char *slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) parent[1] = 0;
    else *slash = 0;
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
    int directory = open(parent, flags), fd = -1, ok = 0, renamed = 0;
    FILE *file = NULL;
    if (directory < 0 || (fd = mkstemp(temporary)) < 0 || !(file = fdopen(fd, "wb"))) goto done;
    uint32_t header[] = {UINT32_C(0x51505451), 1, (uint32_t)m->vocab, DIM,
                        Q_R_HEADS + Q_J_HEADS, Q_LAYERS, Q_MAX_CONTEXT,
                        0, Q_R_HEADS, Q_J_HEADS, HD};
    for (size_t i = 0; i < sizeof(header) / sizeof(header[0]); ++i)
        if (q_export_word(file, header[i])) goto done;
    for (int slot = 0; slot < m->count; ++slot) {
        nt_tensor *p = m->param[slot];
        if (slot >= 2 && (slot - 2) % Q_LAYER_PARAMS == 4) {
            for (int row = 0; row < Q_MECHANISMS; ++row)
                for (int d = 0; d < DIM; ++d)
                    if (q_export_float(file, p->data[row * (DIM + 1) + d])) goto done;
            for (int row = 0; row < Q_MECHANISMS; ++row)
                if (q_export_float(file, p->data[row * (DIM + 1) + DIM])) goto done;
        } else for (int k = 0; k < p->len; ++k)
            if (q_export_float(file, p->data[k])) goto done;
    }
    if (fflush(file) || fsync(fd)) goto done;
    if (fclose(file)) { file = NULL; fd = -1; goto done; }
    file = NULL; fd = -1;
    if (rename(temporary, path)) goto done;
    renamed = 1;
    if (fsync(directory)) goto done;
    ok = 1;
done:
    if (file) { if (fclose(file)) ok = 0; fd = -1; }
    if (fd >= 0 && close(fd)) ok = 0;
    if (!renamed) unlink(temporary);
    if (directory >= 0 && close(directory)) ok = 0;
    free(temporary); free(parent);
    if (ok) return 0;
invalid:
    if (error && error_size) snprintf(error, error_size, "Q runtime export failed");
    return -1;
}

#endif
