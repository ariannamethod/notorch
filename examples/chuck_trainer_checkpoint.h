/* Complete standalone trainer checkpoints. All integer/IEEE-F32 fields are
 * explicitly little-endian; SHA-256 covers every byte preceding the digest.
 * Transient forward/backward tape entries are rebuilt at the next step. */
#ifndef NOTORCH_CHUCK_TRAINER_CHECKPOINT_H
#define NOTORCH_CHUCK_TRAINER_CHECKPOINT_H

#include "sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <sys/stat.h>
#include <unistd.h>

#define TRAINER_CP_LIMIT ((size_t)64 * 1024 * 1024)
#define TRAINER_CP_LIFE_LIMIT 16384
#ifdef NT_CHUCK_Q_TRAINER
#define TRAINER_CP_MAGIC "NTCHTRQ1"
#define TRAINER_CP_BODY "q"
#else
#define TRAINER_CP_MAGIC "NTCHTRN1"
#define TRAINER_CP_BODY "simple"
#endif

typedef struct {
    unsigned char *bytes;
    size_t size, at, cap;
    int reading, failed;
} trainer_cp_codec;

typedef struct {
    int t;
    nt_chuck_param_state chuck;
    size_t data, first, second;
} trainer_cp_slot;

static int trainer_cp_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

static void trainer_cp_bytes(trainer_cp_codec *c, void *data, size_t size) {
    if (c->failed || size > TRAINER_CP_LIMIT - c->at) { c->failed = 1; return; }
    if (c->reading) {
        if (size > c->size - c->at) { c->failed = 1; return; }
        if (data) memcpy(data, c->bytes + c->at, size);
    } else {
        if (c->at + size > c->cap) {
            size_t capacity = c->cap ? c->cap : 4096;
            while (capacity < c->at + size && capacity < TRAINER_CP_LIMIT / 2) capacity *= 2;
            if (capacity < c->at + size) capacity = c->at + size;
            unsigned char *grown = realloc(c->bytes, capacity);
            if (!grown) { c->failed = 1; return; }
            c->bytes = grown; c->cap = capacity;
        }
        memcpy(c->bytes + c->at, data, size);
    }
    c->at += size;
}

static uint32_t trainer_cp_u32(trainer_cp_codec *c, uint32_t value) {
    unsigned char b[4] = {0};
    if (!c->reading) for (int i = 0; i < 4; ++i) b[i] = (unsigned char)(value >> (8 * i));
    trainer_cp_bytes(c, b, 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static uint64_t trainer_cp_u64(trainer_cp_codec *c, uint64_t value) {
    uint32_t low = trainer_cp_u32(c, (uint32_t)value);
    uint32_t high = trainer_cp_u32(c, (uint32_t)(value >> 32));
    return (uint64_t)low | (uint64_t)high << 32;
}

static int trainer_cp_int(trainer_cp_codec *c, int value) {
    uint32_t raw = trainer_cp_u32(c, (uint32_t)value);
    if (raw > INT_MAX) { c->failed = 1; return 0; }
    return (int)raw;
}

static float trainer_cp_float(trainer_cp_codec *c, float value) {
    uint32_t raw = 0;
    memcpy(&raw, &value, 4);
    raw = trainer_cp_u32(c, raw);
    memcpy(&value, &raw, 4);
    if (!isfinite(value)) c->failed = 1;
    return value;
}

static void trainer_cp_array(trainer_cp_codec *c, float *data, size_t count, int nonnegative) {
    if (count > (TRAINER_CP_LIMIT - c->at) / 4) { c->failed = 1; return; }
    for (size_t i = 0; i < count && !c->failed; ++i) {
        float value = trainer_cp_float(c, c->reading ? 0 : data[i]);
        if (nonnegative && value < 0) c->failed = 1;
        if (c->reading && data) data[i] = value;
    }
}

static int trainer_cp_shape(int slot, int vocab, int shape[2], int *ndim) {
#ifdef NT_CHUCK_Q_TRAINER
    *ndim = 0;
    if (!q_body_shape(slot, vocab, shape, ndim)) return 0;
    return *ndim == 1 ? shape[0] : shape[0] * shape[1];
#else
#ifdef NT_CHUCK_ARCHITECT_TEST
    const int layers = 1, hidden = 16;
#else
    const int layers = 2, hidden = 384;
#endif
    *ndim = 2;
    if (slot == 0 || slot == 2 + 9 * layers) { shape[0] = vocab; shape[1] = DIM; }
    else if (slot == 1 + 9 * layers) { *ndim = 1; shape[0] = DIM; }
    else if (slot >= 1 && slot <= 9 * layers) {
        int part = (slot - 1) % 9;
        if (part == 0 || part == 5) { *ndim = 1; shape[0] = DIM; }
        else if (part == 6 || part == 7) { shape[0] = hidden; shape[1] = DIM; }
        else if (part == 8) { shape[0] = DIM; shape[1] = hidden; }
        else { shape[0] = DIM; shape[1] = DIM; }
    } else return 0;
    return *ndim == 1 ? shape[0] : shape[0] * shape[1];
#endif
}

static int trainer_cp_params(void) {
#ifdef NT_CHUCK_Q_TRAINER
    return q_body_params();
#elif defined(NT_CHUCK_ARCHITECT_TEST)
    return 12;
#else
    return 21;
#endif
}

static void trainer_cp_global(trainer_cp_codec *c, nt_chuck_state *s) {
    trainer_cp_array(c, s->loss_hist, NT_CHUCK_WINDOW, 0);
    s->dampen = trainer_cp_float(c, s->dampen);
    s->noise = trainer_cp_float(c, s->noise);
    s->loss_ema = trainer_cp_float(c, s->loss_ema);
    s->macro_ema = trainer_cp_float(c, s->macro_ema);
    s->best_macro = trainer_cp_float(c, s->best_macro);
    s->lr_scale = trainer_cp_float(c, s->lr_scale);
    s->macro_stag = trainer_cp_int(c, s->macro_stag);
    s->global_step = trainer_cp_int(c, s->global_step);
    s->pos = trainer_cp_int(c, s->pos);
    s->full = trainer_cp_int(c, s->full);
    s->stag = trainer_cp_int(c, s->stag);
    s->initialized = trainer_cp_int(c, s->initialized);
    /* A loaded state must still admit a next action. Saturated counters are
     * refused by nt_tape_chuck_step_action; initialized controls must fit the
     * core envelope. The all-zero, pre-step state intentionally remains valid. */
    if (s->pos >= NT_CHUCK_WINDOW || s->full > 1 || s->initialized > 1 || s->noise < 0 ||
        s->global_step == INT_MAX || s->stag == INT_MAX || s->macro_stag == INT_MAX ||
        (s->initialized &&
         (s->dampen < NT_CHUCK_DAMP_LO || s->dampen > NT_CHUCK_DAMP_HI ||
          s->lr_scale < NT_CHUCK_LR_SCALE_LO || s->lr_scale > NT_CHUCK_LR_SCALE_HI ||
          s->noise > NT_CHUCK_NOISE_HI)))
        c->failed = 1;
}

static void trainer_cp_local(trainer_cp_codec *c, nt_chuck_param_state *s) {
    trainer_cp_array(c, s->grad_hist, NT_CHUCK_WINDOW, 1);
    s->dampen = trainer_cp_float(c, s->dampen);
    s->frozen = trainer_cp_int(c, s->frozen);
    s->pos = trainer_cp_int(c, s->pos);
    s->full = trainer_cp_int(c, s->full);
    s->stag = trainer_cp_int(c, s->stag);
    if (s->pos >= NT_CHUCK_WINDOW || s->full > 1 || s->frozen > 1 || s->stag == INT_MAX ||
        (s->dampen != 0 && (s->dampen < NT_CHUCK_DAMP_LO || s->dampen > NT_CHUCK_DAMP_HI)))
        c->failed = 1;
}

/* Use the core life codec and validation, without maintaining a second copy of
 * its field layout. The private temporary directory is always removed. */
static int trainer_cp_life(nt_chuck_architect *a, unsigned char *bytes, size_t *size, int reading) {
    char directory[4096], path[4096];
    const char *base = getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    int n = snprintf(directory, sizeof(directory), "%s/chuck-life-XXXXXX", base);
    if (n < 0 || (size_t)n >= sizeof(directory) - 16 || !mkdtemp(directory)) return -1;
    snprintf(path, sizeof(path), "%s/life.bin", directory);
    int result = -1;
    FILE *f = NULL;
    if (reading) {
        if (*size > TRAINER_CP_LIFE_LIMIT || !(f = fopen(path, "wb"))) goto done;
        int ok = fwrite(bytes, 1, *size, f) == *size;
        if (fclose(f)) ok = 0;
        f = NULL;
        if (ok && !nt_chuck_architect_load(a, path)) result = 0;
    } else {
        if (nt_chuck_architect_save(a, path) || !(f = fopen(path, "rb"))) goto done;
        *size = fread(bytes, 1, TRAINER_CP_LIFE_LIMIT, f);
        int extra = fgetc(f), bad = ferror(f);
        if (fclose(f)) bad = 1;
        f = NULL;
        if (!bad && extra == EOF && *size) result = 0;
    }
done:
    if (f) fclose(f);
    if (unlink(path) && errno != ENOENT) result = -1;
    if (rmdir(directory)) result = -1;
    return result;
}

static void trainer_cp_metadata(trainer_cp_codec *c, trainer_state *s, int *training, uint32_t *noise) {
    if (trainer_cp_u32(c, CTX) != CTX || trainer_cp_u32(c, DIM) != DIM || trainer_cp_u32(c, HD) != HD)
        c->failed = 1;
    s->vocab_size = trainer_cp_u32(c, s->vocab_size);
    s->token_count = trainer_cp_u64(c, s->token_count);
    s->split = trainer_cp_u64(c, s->split);
    s->step = trainer_cp_u64(c, s->step);
    s->windows = trainer_cp_u64(c, s->windows);
    s->seed = trainer_cp_u32(c, s->seed);
    s->acquire_every = trainer_cp_u32(c, s->acquire_every);
    s->lr = trainer_cp_float(c, s->lr);
    s->acquisitions = trainer_cp_u64(c, s->acquisitions);
    s->fit_updates = trainer_cp_u64(c, s->fit_updates);
    s->branch_updates = trainer_cp_u64(c, s->branch_updates);
    uint64_t actions = 0;
    for (int i = 0; i < 4; ++i) {
        s->action_counts[i] = trainer_cp_u64(c, s->action_counts[i]);
        if (s->action_counts[i] > s->step) c->failed = 1;
        actions += s->action_counts[i];
    }
    *training = trainer_cp_int(c, *training);
    *noise = trainer_cp_u32(c, *noise);
    trainer_cp_bytes(c, s->corpus_sha256, sizeof(s->corpus_sha256));
    uint64_t expected = s->acquire_every && s->step ? 1 + (s->step - 1) / s->acquire_every : 0;
    if (s->vocab_size < 2 || s->vocab_size > TRAINER_MAX_VOCAB || !s->seed || !*noise ||
        !s->acquire_every || s->step >= INT_MAX || s->token_count > SIZE_MAX / sizeof(uint32_t) ||
        s->split <= CTX || s->split >= s->token_count || s->token_count - s->split <= CTX ||
        s->split != (s->token_count / 10 * 9 + s->token_count % 10 * 9 / 10) ||
        s->lr <= 0 || s->lr > 1 || *training > 1 || actions != s->step ||
        s->acquisitions != expected || s->fit_updates != expected * 32 || s->branch_updates != expected * 48)
        c->failed = 1;
    if (c->failed) return;
#ifdef NT_CHUCK_Q_TRAINER
    if (s->vocab || s->vocab_size < 256 || s->vocab_size > 1280) {
        c->failed = 1; return;
    }
    s->tokenizer.merge_count = trainer_cp_u32(c, s->tokenizer.merge_count);
    if (c->failed || s->tokenizer.merge_count >
        sizeof(s->tokenizer.pairs) / sizeof(s->tokenizer.pairs[0])) {
        c->failed = 1; return;
    }
    if (c->reading) s->tokenizer.vocab_size = 256 + s->tokenizer.merge_count;
    if (s->vocab_size != s->tokenizer.vocab_size ||
        s->vocab_size != 256 + s->tokenizer.merge_count) {
        c->failed = 1; return;
    }
    for (uint32_t i = 0; i < s->tokenizer.merge_count && !c->failed; ++i)
        for (int j = 0; j < 2; ++j)
            s->tokenizer.pairs[i][j] = trainer_cp_u32(c, s->tokenizer.pairs[i][j]);
    char tokenizer_error[160];
    if (!c->failed && q_tokenizer_validate(&s->tokenizer, tokenizer_error, sizeof(tokenizer_error)))
        c->failed = 1;
#else
    if (c->reading) {
        s->vocab = malloc((size_t)s->vocab_size * sizeof(*s->vocab));
        if (!s->vocab) { c->failed = 1; return; }
    } else if (!s->vocab) { c->failed = 1; return; }
    for (uint32_t i = 0; i < s->vocab_size; ++i) {
        uint32_t scalar = trainer_cp_u32(c, c->reading ? 0 : s->vocab[i]);
        if (c->reading) s->vocab[i] = scalar;
        if (!scalar || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff) ||
            (i && scalar <= s->vocab[i - 1])) c->failed = 1;
    }
#endif
}

static int trainer_cp_atomic(const unsigned char *bytes, size_t size, const char *path) {
    size_t length = strlen(path);
    if (!length || path[length - 1] == '/' || length > SIZE_MAX - 16) return -1;
    char *temporary = malloc(length + 16), *parent = strdup(path);
    if (!temporary || !parent) { free(temporary); free(parent); return -1; }
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
    FILE *f = NULL;
    struct stat st;
    if (directory < 0 || fstat(directory, &st) || !S_ISDIR(st.st_mode)) goto done;
    fd = mkstemp(temporary);
    if (fd < 0 || !(f = fdopen(fd, "wb"))) goto done;
    if (fwrite(bytes, 1, size, f) != size || fflush(f) || fsync(fd)) goto done;
    if (fclose(f)) { f = NULL; fd = -1; goto done; }
    f = NULL; fd = -1;
    if (rename(temporary, path)) goto done;
    renamed = 1;
    if (fsync(directory)) goto done;
    ok = 1;
done:
    if (f) { if (fclose(f)) ok = 0; fd = -1; }
    if (fd >= 0 && close(fd)) ok = 0;
    if (!renamed) unlink(temporary);
    if (directory >= 0 && close(directory)) ok = 0;
    free(temporary); free(parent);
    return ok ? 0 : -1;
}

static int trainer_checkpoint_save(const trainer_state *state, const char *path,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!state || !path || !*path || sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 || nt_get_gpu_mode())
        return trainer_cp_error(error, error_size, "checkpoint arguments, float format or CPU mode invalid");
    nt_tape *tape = nt_tape_get();
    if (state->model.count != trainer_cp_params() || tape->n_params != state->model.count ||
        state->architect.pending || state->architect.config.mode != NT_CHUCK_ARCHITECT_LEARNED ||
        state->architect.decisions != state->step || state->architect.updates != state->step ||
        tape->chuck.global_step != (int)state->step)
        return trainer_cp_error(error, error_size, "checkpoint requires a completed registered training step");
    unsigned char life[TRAINER_CP_LIFE_LIMIT];
    size_t life_size = sizeof(life);
    trainer_state s = *state;
    if (trainer_cp_life(&s.architect, life, &life_size, 0))
        return trainer_cp_error(error, error_size, "cannot encode Architect life");
    trainer_cp_codec c = {0};
    char magic[] = TRAINER_CP_MAGIC;
    trainer_cp_bytes(&c, magic, 8);
    trainer_cp_u32(&c, 1); trainer_cp_u64(&c, 0);
    int training = nt_is_training();
    uint32_t noise = nt_chuck_rng_get();
    trainer_cp_metadata(&c, &s, &training, &noise);
    nt_chuck_state chuck = tape->chuck;
    trainer_cp_global(&c, &chuck);
    trainer_cp_u32(&c, (uint32_t)life_size); trainer_cp_bytes(&c, life, life_size);
    trainer_cp_u32(&c, (uint32_t)state->model.count);
    for (int i = 0; i < state->model.count && !c.failed; ++i) {
        const nt_tensor *p = state->model.param[i];
        const nt_adam_state *a = &tape->adam[i];
        int shape[2] = {0}, ndim, length = trainer_cp_shape(i, (int)state->vocab_size, shape, &ndim);
        if (length <= 0 || !p || p->ndim != ndim || p->len != length ||
            memcmp(p->shape, shape, (size_t)ndim * sizeof(int)) || !a->m || !a->v ||
            a->m->len != length || a->v->len != length || a->acc_grad || a->t < 0 ||
            (uint64_t)a->t > state->step) { c.failed = 1; break; }
        trainer_cp_u32(&c, (uint32_t)ndim);
        for (int j = 0; j < ndim; ++j) trainer_cp_u32(&c, (uint32_t)shape[j]);
        trainer_cp_u32(&c, (uint32_t)length); trainer_cp_int(&c, a->t);
        nt_chuck_param_state local = tape->chuck_params[i];
        trainer_cp_local(&c, &local);
        trainer_cp_array(&c, p->data, (size_t)length, 0);
        trainer_cp_array(&c, a->m->data, (size_t)length, 0);
        trainer_cp_array(&c, a->v->data, (size_t)length, 1);
    }
    int result = -1;
    if (!c.failed) {
        uint64_t total = c.at + 32;
        for (int i = 0; i < 8; ++i) c.bytes[12 + i] = (unsigned char)(total >> (i * 8));
        unsigned char digest[32];
        if (!nt_sha256(c.bytes, c.at, digest)) {
            trainer_cp_bytes(&c, digest, sizeof(digest));
            if (!c.failed && !trainer_cp_atomic(c.bytes, c.at, path)) result = 0;
        }
    }
    free(c.bytes);
    return result ? trainer_cp_error(error, error_size, "checkpoint state invalid or atomic write failed") : 0;
}

static int trainer_checkpoint_load(trainer_state *out, const char *path,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!out || !path || !*path || out->vocab || out->model.count || nt_get_gpu_mode() ||
        sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128)
        return trainer_cp_error(error, error_size, "checkpoint load requires an empty CPU trainer");
    nt_tape *live = nt_tape_get();
    if (live->count || live->n_params) return trainer_cp_error(error, error_size, "checkpoint load requires an empty tape");
    for (int i = 0; i < NT_TAPE_MAX_PARAMS; ++i)
        if (live->adam[i].m || live->adam[i].v || live->adam[i].acc_grad)
            return trainer_cp_error(error, error_size, "checkpoint load requires empty optimizer slots");
    FILE *f = fopen(path, "rb");
    if (!f) return trainer_cp_error(error, error_size, "cannot open checkpoint");
    struct stat st;
    if (fstat(fileno(f), &st) || !S_ISREG(st.st_mode) || st.st_size < 64 ||
        (uint64_t)st.st_size > TRAINER_CP_LIMIT) {
        fclose(f); return trainer_cp_error(error, error_size, "checkpoint size or file type invalid");
    }
    size_t size = (size_t)st.st_size;
    unsigned char *bytes = malloc(size);
    if (!bytes) { fclose(f); return trainer_cp_error(error, error_size, "checkpoint allocation failed"); }
    int ok = fread(bytes, 1, size, f) == size;
    int extra = fgetc(f);
    if (ferror(f) || extra != EOF) ok = 0;
    if (fclose(f)) ok = 0;
    unsigned char digest[32];
    if (!ok || nt_sha256(bytes, size - 32, digest) || memcmp(digest, bytes + size - 32, 32)) {
        free(bytes); return trainer_cp_error(error, error_size, "checkpoint checksum or read failed");
    }
    trainer_state s = {0};
    trainer_cp_codec c = {bytes, size - 32, 0, size - 32, 1, 0};
    char magic[8]; trainer_cp_bytes(&c, magic, sizeof(magic));
    if (memcmp(magic, TRAINER_CP_MAGIC, 8) || trainer_cp_u32(&c, 0) != 1 || trainer_cp_u64(&c, 0) != size)
        c.failed = 1;
    int training = 0;
    uint32_t noise = 0;
    trainer_cp_metadata(&c, &s, &training, &noise);
    nt_chuck_state chuck = {0};
    trainer_cp_global(&c, &chuck);
    uint32_t life_size = trainer_cp_u32(&c, 0);
    if (life_size > TRAINER_CP_LIFE_LIMIT || life_size > c.size - c.at) c.failed = 1;
    if (!c.failed) {
        size_t n = life_size;
        if (trainer_cp_life(&s.architect, c.bytes + c.at, &n, 1) || s.architect.pending ||
            s.architect.config.mode != NT_CHUCK_ARCHITECT_LEARNED ||
            s.architect.decisions != s.step || s.architect.updates != s.step ||
            chuck.global_step != (int)s.step) c.failed = 1;
        trainer_cp_bytes(&c, NULL, life_size);
    }
    uint32_t parameters = trainer_cp_u32(&c, 0);
    if (parameters != (uint32_t)trainer_cp_params()) c.failed = 1;
    trainer_cp_slot slots[MAX_PARAMS] = {{0}};
    for (uint32_t i = 0; i < parameters && !c.failed; ++i) {
        int shape[2] = {0}, ndim, length = trainer_cp_shape((int)i, (int)s.vocab_size, shape, &ndim);
        if (length <= 0) { c.failed = 1; break; }
        if (trainer_cp_u32(&c, 0) != (uint32_t)ndim) c.failed = 1;
        for (int j = 0; j < ndim; ++j)
            if (trainer_cp_u32(&c, 0) != (uint32_t)shape[j]) c.failed = 1;
        if (trainer_cp_u32(&c, 0) != (uint32_t)length) c.failed = 1;
        slots[i].t = trainer_cp_int(&c, 0);
        if ((uint64_t)slots[i].t > s.step) c.failed = 1;
        trainer_cp_local(&c, &slots[i].chuck);
        slots[i].data = c.at; trainer_cp_array(&c, NULL, (size_t)length, 0);
        slots[i].first = c.at; trainer_cp_array(&c, NULL, (size_t)length, 0);
        slots[i].second = c.at; trainer_cp_array(&c, NULL, (size_t)length, 1);
    }
    if (c.failed || c.at != c.size) {
        free(s.vocab); free(bytes);
        return trainer_cp_error(error, error_size, "checkpoint version, dimensions, fields or payload invalid");
    }
    /* Only fully authenticated and validated bytes can allocate/install a body. */
    body_init_vocab(&s.model, TRAINER_CP_BODY, (int)s.vocab_size);
    nt_tape_start();
    for (int i = 0; i < s.model.count; ++i) {
        nt_tensor *p = s.model.param[i];
        if (nt_tape_param(p) < 0) die("checkpoint parameter registration failed");
        nt_adam_state *a = &live->adam[i];
        c.at = slots[i].data; trainer_cp_array(&c, p->data, (size_t)p->len, 0);
        c.at = slots[i].first; trainer_cp_array(&c, a->m->data, (size_t)p->len, 0);
        c.at = slots[i].second; trainer_cp_array(&c, a->v->data, (size_t)p->len, 1);
        a->t = slots[i].t;
        live->chuck_params[i] = slots[i].chuck;
    }
    live->chuck = chuck;
    if (nt_chuck_rng_set(noise)) die("validated checkpoint RNG refused");
    nt_train_mode(training);
    *out = s;
    free(bytes);
    return 0;
}

#endif
