/*
 * Chuck: Loss Architect — two real training bodies on the upstream tape.
 *
 * SimpleLLM: ariannamethod/notorch-simple-llm, train_dracula.py (80b3bd6).
 * HeVLM: ariannamethod/notorch-diffusion, train_hevlm.c (1bc6450).
 * Both architectures retain their initialization, attention and FFN recipes.
 * Corpus tokenization/provenance and repeated runs live in
 * experiments/chuck_loss_architect/run.py. No vendored numerical engine.
 *
 * Build/run: see the experiment README. Every process owns one arm/seed.
 */
#include "notorch.h"
#include "chuck_architect.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define MAX_LAYERS 4
#define MAX_PARAMS (2 + MAX_LAYERS * 9 + 2)
#define CTX 64
#define DIM 128
#define HD 32
#define EVAL_WINDOWS 8

typedef struct {
    int layers, hidden, vocab, rope, positional;
    const char *name;
    nt_tensor *param[MAX_PARAMS];
    int count;
    long elements;
} body;

static void die(const char *message) {
    fprintf(stderr, "chuck_architect_train: %s\n", message);
    exit(1);
}

static double seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) die("clock_gettime failed");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* The window stream is independent of notorch's initialization/noise streams. */
static uint64_t next_window(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static nt_tensor *matrix(body *m, int rows, int cols, int fan_in, int fan_out) {
    nt_tensor *p = nt_tensor_new2d(rows, cols);
    if (!p || m->count == MAX_PARAMS) die("parameter allocation failed");
    nt_tensor_xavier(p, fan_in, fan_out);
    m->param[m->count++] = p;
    m->elements += p->len;
    return p;
}

static void norm(body *m) {
    nt_tensor *p = nt_tensor_new(DIM);
    if (!p || m->count == MAX_PARAMS) die("norm allocation failed");
    nt_tensor_fill(p, 1.0f);
    m->param[m->count++] = p;
    m->elements += p->len;
}

static void body_init(body *m, const char *name) {
    memset(m, 0, sizeof(*m));
    m->name = name;
    if (!strcmp(name, "simple")) {
        m->layers = 2; m->hidden = 384; m->vocab = 94; m->rope = 1;
    } else if (!strcmp(name, "hevlm")) {
        m->layers = 4; m->hidden = 512; m->vocab = 256; m->positional = 1;
    } else die("body must be simple or hevlm");
    matrix(m, m->vocab, DIM, m->vocab, DIM);
    if (m->positional) matrix(m, CTX, DIM, CTX, DIM);
    float residual_scale = m->positional ? 0.02f / sqrtf(2.0f * m->layers) / 0.1f : 1.0f;
    for (int layer = 0; layer < m->layers; layer++) {
        norm(m);
        for (int j = 0; j < 4; j++) {
            nt_tensor *p = matrix(m, DIM, DIM, DIM, DIM);
            if (j == 3 && m->positional)
                for (int k = 0; k < p->len; k++) p->data[k] *= residual_scale;
        }
        norm(m);
        matrix(m, m->hidden, DIM, DIM, m->hidden);
        matrix(m, m->hidden, DIM, DIM, m->hidden);
        nt_tensor *p = matrix(m, DIM, m->hidden, m->hidden, DIM);
        if (m->positional)
            for (int k = 0; k < p->len; k++) p->data[k] *= residual_scale;
    }
    norm(m);
    matrix(m, m->vocab, DIM, DIM, m->vocab);
}

static int body_forward(body *m, const uint32_t *data, size_t offset, int train) {
    nt_tape_start();
    nt_train_mode(train);
    int p[MAX_PARAMS];
    for (int i = 0; i < m->count; i++) p[i] = nt_tape_param(m->param[i]);
    nt_tape_no_decay(p[0]);
    if (m->positional) nt_tape_no_decay(p[1]);
    nt_tensor *tokens = nt_tensor_new(CTX), *targets = nt_tensor_new(CTX);
    if (!tokens || !targets) die("token allocation failed");
    for (int i = 0; i < CTX; i++) {
        tokens->data[i] = (float)data[offset + i];
        targets->data[i] = (float)data[offset + i + 1];
    }
    int ti = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
    int yi = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    int pi = 1 + m->positional;
    int h = nt_seq_embedding(p[0], m->positional ? p[1] : -1, ti, CTX, DIM);
    for (int layer = 0; layer < m->layers; layer++) {
        int r1 = p[pi++], wq = p[pi++], wk = p[pi++], wv = p[pi++], wo = p[pi++];
        int r2 = p[pi++], wg = p[pi++], wu = p[pi++], wd = p[pi++];
        int x = nt_seq_rmsnorm(h, r1, CTX, DIM);
        int q = nt_seq_linear(wq, x, CTX), k = nt_seq_linear(wk, x, CTX);
        if (m->rope) { q = nt_rope(q, CTX, HD); k = nt_rope(k, CTX, HD); }
        int v = nt_seq_linear(wv, x, CTX);
        int attention = nt_mh_causal_attention(q, k, v, CTX, HD);
        h = nt_add(h, nt_seq_linear(wo, attention, CTX));
        x = nt_seq_rmsnorm(h, r2, CTX, DIM);
        int gate = nt_silu(nt_seq_linear(wg, x, CTX));
        int up = nt_seq_linear(wu, x, CTX);
        h = nt_add(h, nt_seq_linear(wd, nt_mul(gate, up), CTX));
    }
    int hf = nt_seq_rmsnorm(h, p[pi], CTX, DIM);
    int logits = nt_seq_linear(p[pi + 1], hf, CTX);
    int loss = nt_seq_cross_entropy(logits, yi, CTX, m->vocab);
    if (loss < 0) die("forward tape failed");
    return loss;
}

static float read_loss(int index) {
    float loss = nt_tape_get()->entries[index].output->data[0];
    if (!isfinite(loss)) die("non-finite loss");
    return loss;
}

static float evaluate(body *m, const uint32_t *data, size_t split, size_t count) {
    double total = 0;
    size_t width = count - split - CTX;
    for (int i = 0; i < EVAL_WINDOWS; i++) {
        size_t offset = split + width * (size_t)i / EVAL_WINDOWS;
        total += read_loss(body_forward(m, data, offset, 0));
    }
    return (float)(total / EVAL_WINDOWS);
}

static uint32_t *read_tokens(const char *path, size_t *count, int vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open token file");
    if (fseek(f, 0, SEEK_END)) die("cannot seek token file");
    long bytes = ftell(f);
    if (bytes < 0 || bytes % 4 || bytes < CTX * 40) die("invalid token file length");
    rewind(f);
    *count = (size_t)bytes / 4;
    uint32_t *data = malloc(*count * sizeof(*data));
    if (!data) die("cannot allocate token corpus");
    for (size_t i = 0; i < *count; i++) {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) die("truncated token file");
        data[i] = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
        if (data[i] >= (uint32_t)vocab) die("token outside body vocabulary");
    }
    if (fclose(f)) die("cannot close token file");
    return data;
}

static void write_chuck(FILE *f, const nt_chuck_state *s) {
    fprintf(f, "{\"initialized\":%d,\"global_step\":%d,\"dampen\":%.9g,"
            "\"lr_scale\":%.9g,\"noise\":%.9g,\"loss_ema\":%.9g,\"macro_ema\":%.9g,"
            "\"best_macro\":%.9g,\"stag\":%d,\"macro_stag\":%d,\"pos\":%d,\"full\":%d,\"loss_hist\":[",
            s->initialized, s->global_step, s->dampen, s->lr_scale, s->noise,
            s->loss_ema, s->macro_ema, s->best_macro, s->stag, s->macro_stag, s->pos, s->full);
    for (int i = 0; i < NT_CHUCK_WINDOW; i++) fprintf(f, "%s%.9g", i ? "," : "", s->loss_hist[i]);
    fputs("]}", f);
}

static void path_for(char *path, size_t size, const char *prefix, const char *suffix) {
    int n = snprintf(path, size, "%s%s", prefix, suffix);
    if (n < 0 || (size_t)n >= size) die("output path too long");
}

static void read_config(nt_chuck_architect_config *config, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open Architect configuration");
    char text[16385], error[256];
    size_t n = fread(text, 1, sizeof(text), f);
    if (ferror(f) || n == sizeof(text)) die("Architect configuration read/size error");
    if (memchr(text, 0, n)) die("embedded NUL in Architect configuration");
    text[n] = 0;
    if (fclose(f)) die("cannot close Architect configuration");
    if (nt_chuck_architect_config_parse_json(config, text, error, sizeof(error))) die(error);
}

static void save_optimizer(const char *prefix, uint64_t window_rng, int steps) {
    nt_tape *tape = nt_tape_get();
    char path[4096];
    nt_tensor *moments[2 * MAX_PARAMS];
    for (int i = 0; i < tape->n_params; i++) {
        moments[2 * i] = tape->adam[i].m;
        moments[2 * i + 1] = tape->adam[i].v;
    }
    path_for(path, sizeof(path), prefix, ".moments.final.bin");
    if (nt_save(path, moments, 2 * tape->n_params)) die("cannot save optimizer moments");
    path_for(path, sizeof(path), prefix, ".optimizer.final.json");
    FILE *f = fopen(path, "w");
    if (!f) die("cannot save optimizer controls");
    fprintf(f, "{\"schema\":1,\"steps\":%d,\"window_rng\":\"%016" PRIx64 "\",\"chuck_rng\":%" PRIu32 ",\"chuck\":",
            steps, window_rng, nt_chuck_rng_get());
    write_chuck(f, &tape->chuck);
    fputs(",\"parameters\":[", f);
    for (int i = 0; i < tape->n_params; i++) {
        const nt_chuck_param_state *p = &tape->chuck_params[i];
        fprintf(f, "%s{\"slot\":%d,\"adam_t\":%d,\"dampen\":%.9g,\"frozen\":%d,\"pos\":%d,"
                "\"full\":%d,\"stag\":%d,\"grad_hist\":[", i ? "," : "", i, tape->adam[i].t,
                p->dampen, p->frozen, p->pos, p->full, p->stag);
        for (int j = 0; j < NT_CHUCK_WINDOW; j++) fprintf(f, "%s%.9g", j ? "," : "", p->grad_hist[j]);
        fputs("]}", f);
    }
    fputs("]}\n", f);
    if (ferror(f) || fclose(f)) die("optimizer control write failed");
}

static void write_architect(FILE *f, const nt_chuck_architect_decision *d,
        const nt_chuck_architect_receipt *r, uint64_t pre, uint64_t pending, uint64_t post,
        const nt_chuck_action *initial_weights_action) {
    static const char *names[] = {"legacy", "hold", "brake", "push", "set_dampen", "set_lr_scale", "set_noise"};
    const nt_chuck_observation *o = &d->observation;
    fprintf(f, "{\"sequence\":%" PRIu64 ",\"explored\":%d,\"action\":{\"type\":\"%s\",\"kind\":%d,\"value\":%.9g},"
            "\"policy_pre\":\"%016" PRIx64 "\",\"policy_pending\":\"%016" PRIx64 "\",\"policy_post\":\"%016" PRIx64 "\","
            "\"observation\":{\"loss\":%.9g,\"loss_ema\":%.9g,\"loss_trend\":%.9g,\"macro_ema\":%.9g,"
            "\"best_macro\":%.9g,\"dampen\":%.9g,\"lr_scale\":%.9g,\"noise\":%.9g,"
            "\"grad_norm\":%.9g,\"grad_trend\":%.9g,\"frozen_fraction\":%.9g,"
            "\"step\":%d,\"stag\":%d,\"macro_stag\":%d,\"history_len\":%d},\"features\":[",
            d->sequence, d->explored, names[d->action.kind], d->action.kind, d->action.value, pre, pending, post,
            o->loss, o->loss_ema, o->loss_trend, o->macro_ema, o->best_macro, o->dampen, o->lr_scale,
            o->noise, o->grad_norm, o->grad_trend, o->frozen_fraction, o->step, o->stag, o->macro_stag, o->history_len);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; i++) fprintf(f, "%s%.9g", i ? "," : "", d->features[i]);
    fputs("],\"scores\":[", f);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_HEADS; i++) fprintf(f, "%s%.9g", i ? "," : "", d->scores[i]);
    fprintf(f, "],\"consequence\":{\"before_loss\":%.9g,\"after_loss\":%.9g,\"loss_delta\":%.17g,"
            "\"reward\":%.9g,\"predicted\":%.9g,\"error\":%.9g,\"decision\":%" PRIu64 ",\"learned\":%d,\"nonfinite\":%d}",
            r->before_loss, r->after_loss, r->loss_delta, r->reward, r->predicted, r->error,
            r->decision, r->learned, r->nonfinite);
    if (initial_weights_action) {
        fprintf(f, ",\"initial_weights_action\":{\"type\":\"%s\",\"kind\":%d,\"value\":%.9g},\"weight_dependent_choice\":%s",
                names[initial_weights_action->kind], initial_weights_action->kind, initial_weights_action->value,
                initial_weights_action->kind != d->action.kind || initial_weights_action->value != d->action.value ? "true" : "false");
    }
    fputs("}", f);
}

static long parse_integer(const char *text, long low, long high) {
    char *end = NULL;
    errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || !text[0] || *end || v < low || v > high) die("invalid integer argument");
    return v;
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s simple|hevlm adam|chuck|legacy|learned TOKENS OUT_PREFIX STEPS SEED LR [ARCHITECT_JSON]\n", argv[0]);
        return 2;
    }
    const char *arm = argv[2], *prefix = argv[4];
    int adam = !strcmp(arm, "adam");
    int legacy = !strcmp(arm, "legacy"), learned = !strcmp(arm, "learned");
    int has_architect = legacy || learned;
    if (!adam && !has_architect && strcmp(arm, "chuck")) die("unknown arm");
    if (argc == 9 && !learned) die("custom configuration belongs to the learned arm");
    int steps = (int)parse_integer(argv[5], 1, 1000000);
    unsigned seed = (unsigned)parse_integer(argv[6], 1, UINT_MAX);
    char *end = NULL;
    errno = 0;
    float lr = strtof(argv[7], &end);
    if (errno || *end || !isfinite(lr) || lr <= 0 || lr > 1) die("invalid learning rate");
    nt_seed(seed);
    nt_chuck_architect architect = {0}, initial_architect = {0};
    if (has_architect) {
        nt_chuck_architect_config config;
        nt_chuck_architect_config_default(&config);
        if (argc == 9) {
            read_config(&config, argv[8]);
            if (config.mode != NT_CHUCK_ARCHITECT_LEARNED) die("learned arm needs learned configuration");
        } else {
            config.mode = learned ? NT_CHUCK_ARCHITECT_LEARNED : NT_CHUCK_ARCHITECT_LEGACY;
            snprintf(config.life_id, sizeof(config.life_id), "chuck.%s", argv[1]);
        }
        config.seed = seed;
        if (nt_chuck_architect_init(&architect, &config)) die("cannot initialize Architect");
        initial_architect = architect;
    }
    body m;
    body_init(&m, argv[1]);
    size_t count;
    uint32_t *data = read_tokens(argv[3], &count, m.vocab);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) die("corpus split too small");
    char path[4096];
    path_for(path, sizeof(path), prefix, ".initial.bin");
    if (nt_save(path, m.param, m.count)) die("cannot save initial weights");
    if (has_architect) {
        path_for(path, sizeof(path), prefix, ".policy.initial.bin");
        if (nt_chuck_architect_save(&architect, path)) die("cannot save initial Architect life");
    }
    path_for(path, sizeof(path), prefix, ".jsonl");
    FILE *trace = fopen(path, "w");
    if (!trace) die("cannot open receipt file");
    fprintf(trace, "{\"type\":\"run\",\"body\":\"%s\",\"arm\":\"%s\",\"seed\":%u,\"steps\":%d,"
            "\"parameters\":%ld,\"tensors\":%d,\"vocab\":%d,\"context\":%d,\"lr\":%.9g,"
            "\"tokens\":%zu,\"train_tokens\":%zu,\"eval_windows\":%d",
            m.name, arm, seed, steps, m.elements, m.count, m.vocab, CTX, lr, count, split, EVAL_WINDOWS);
    if (has_architect) {
        const nt_chuck_architect_config *c = &architect.config;
        fprintf(trace, ",\"architect_config\":{\"mode\":%d,\"life_id\":\"%s\",\"seed\":%" PRIu32 ","
                "\"learning_rate\":%.9g,\"exploration\":%.9g,\"parameters\":%d,\"initial_hash\":\"%016" PRIx64 "\","
                "\"enabled_actions\":%" PRIu32 ",\"dampen_min\":%.9g,\"dampen_max\":%.9g,"
                "\"lr_scale_min\":%.9g,\"lr_scale_max\":%.9g,\"noise_min\":%.9g,\"noise_max\":%.9g}",
                c->mode, c->life_id, c->seed, c->learning_rate, c->exploration, NT_CHUCK_ARCHITECT_PARAMETERS,
                nt_chuck_architect_hash(&architect), c->limits.enabled_actions, c->limits.dampen_min, c->limits.dampen_max,
                c->limits.lr_scale_min, c->limits.lr_scale_max, c->limits.noise_min, c->limits.noise_max);
    }
    fputs("}\n", trace);
    double start = seconds();
    float eval_initial = evaluate(&m, data, split, count);
    fprintf(trace, "{\"type\":\"evaluation\",\"step\":0,\"heldout_loss\":%.9g}\n", eval_initial);
    uint64_t windows = seed;
    double first_sum = 0, last_sum = 0, improvement = 0;
    int average_count = steps < 16 ? steps : 16;
    float first_loss = 0, last_loss = 0;
    int weight_dependent_choices = 0;
    for (int step = 0; step < steps; step++) {
        size_t offset = (size_t)(next_window(&windows) % (split - CTX));
        int loss_idx = body_forward(&m, data, offset, 1);
        float before = read_loss(loss_idx);
        nt_tape_backward(loss_idx);
        float grad_norm = nt_tape_clip_grads(1.0f);
        if (!isfinite(grad_norm)) die("non-finite gradient norm");
        nt_chuck_state pre = nt_tape_get()->chuck;
        nt_chuck_architect_decision decision = {0};
        nt_chuck_architect_receipt receipt = {0};
        nt_chuck_action initial_weights_action = {0};
        uint64_t policy_pre = 0, policy_pending = 0, policy_post = 0;
        if (has_architect) policy_pre = nt_chuck_architect_hash(&architect);
        if (adam) nt_tape_adam_step(lr);
        else if (has_architect) {
            nt_chuck_architect counterfactual = architect;
            if (learned) {
                memcpy(counterfactual.w1, initial_architect.w1, sizeof(counterfactual.w1));
                memcpy(counterfactual.b1, initial_architect.b1, sizeof(counterfactual.b1));
                memcpy(counterfactual.w2, initial_architect.w2, sizeof(counterfactual.w2));
                memcpy(counterfactual.b2, initial_architect.b2, sizeof(counterfactual.b2));
            }
            if (nt_chuck_architect_step(&architect, lr, before, &decision)) die("Architect action refused");
            policy_pending = nt_chuck_architect_hash(&architect);
            if (learned) {
                if (nt_chuck_architect_select(&counterfactual, &decision.observation, &initial_weights_action))
                    die("initial-weights action readout refused");
                weight_dependent_choices += initial_weights_action.kind != decision.action.kind ||
                                            initial_weights_action.value != decision.action.value;
            }
        } else nt_tape_chuck_step(lr, before);
        nt_chuck_state post = nt_tape_get()->chuck;
        int frozen = 0;
        for (int i = 0; i < m.count; i++) frozen += nt_tape_get()->chuck_params[i].frozen;
        float after = read_loss(body_forward(&m, data, offset, 0));
        if (learned) {
            if (nt_chuck_architect_feedback(&architect, after, &receipt)) die("Architect consequence refused");
        } else {
            receipt.before_loss = before; receipt.after_loss = after;
            receipt.loss_delta = (double)before - after;
            receipt.action = decision.action; receipt.decision = decision.sequence;
        }
        if (has_architect) policy_post = nt_chuck_architect_hash(&architect);
        fprintf(trace, "{\"type\":\"step\",\"step\":%d,\"offset\":%zu,\"window_rng\":\"%016" PRIx64 "\","
                "\"loss_before\":%.9g,\"loss_after_same_window\":%.9g,\"improvement\":%.9g,"
                "\"gradient_norm_before_clip\":%.9g,\"frozen_tensors\":%d,\"pre_chuck\":",
                step + 1, offset, windows, before, after, before - after, grad_norm, frozen);
        write_chuck(trace, &pre);
        fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
        if (has_architect) {
            fputs(",\"architect\":", trace);
            write_architect(trace, &decision, &receipt, policy_pre, policy_pending, policy_post,
                            learned ? &initial_weights_action : NULL);
        }
        fputs("}\n", trace);
        if (!step) first_loss = before;
        last_loss = before;
        if (step < average_count) first_sum += before;
        if (step >= steps - average_count) last_sum += before;
        improvement += before - after;
        if ((step + 1) % 32 == 0 || !step || step + 1 == steps) {
            printf("%s %s seed=%u step=%d/%d loss=%.6f after=%.6f elapsed=%.2fs\n",
                    m.name, arm, seed, step + 1, steps, before, after, seconds() - start);
            fflush(stdout); fflush(trace);
        }
    }
    float eval_final = evaluate(&m, data, split, count);
    double elapsed = seconds() - start;
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) die("getrusage failed");
    long max_rss_kib = usage.ru_maxrss;
#ifdef __APPLE__
    max_rss_kib /= 1024;
#endif
    double user_seconds = (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec * 1e-6;
    double system_seconds = (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec * 1e-6;
    fprintf(trace, "{\"type\":\"evaluation\",\"step\":%d,\"heldout_loss\":%.9g}\n", steps, eval_final);
    fprintf(trace, "{\"type\":\"summary\",\"first_loss\":%.9g,\"last_loss\":%.9g,\"first_mean\":%.9g,"
            "\"last_mean\":%.9g,\"mean_count\":%d,\"mean_improvement\":%.9g,\"eval_initial\":%.9g,"
            "\"eval_final\":%.9g,\"weight_dependent_choices\":%d,\"seconds\":%.9g,\"max_rss_kib\":%ld,"
            "\"user_seconds\":%.9g,\"system_seconds\":%.9g}\n", first_loss, last_loss, first_sum / average_count,
            last_sum / average_count, average_count, improvement / steps, eval_initial, eval_final, weight_dependent_choices,
            elapsed, max_rss_kib, user_seconds, system_seconds);
    if (ferror(trace) || fclose(trace)) die("receipt write failed");
    path_for(path, sizeof(path), prefix, ".final.bin");
    if (nt_save(path, m.param, m.count)) die("cannot save final weights");
    save_optimizer(prefix, windows, steps);
    if (has_architect) {
        path_for(path, sizeof(path), prefix, ".policy.final.bin");
        if (nt_chuck_architect_save(&architect, path)) die("cannot save final Architect life");
    }
    printf("DONE %s %s seed=%u parameters=%ld heldout %.6f -> %.6f elapsed %.2fs\n",
            m.name, arm, seed, m.elements, eval_initial, eval_final, elapsed);
    nt_tape_destroy();
    for (int i = 0; i < m.count; i++) nt_tensor_free(m.param[i]);
    free(data);
    return 0;
}
