/* Fixed replay of measured common-state consequences. Neural arithmetic stays
 * in notorch; this executable owns experiment ordering and provenance receipts.
 * Sample table: NTCAFT01 (8 bytes), u32 count, little-endian records
 * described by experiments/chuck_loss_architect/future/run.py:write_samples.
 */
#include "chuck_architect.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SAMPLES 64
#define PATH_BYTES 4096

typedef struct {
    uint32_t body, seed, checkpoint;
    uint64_t state_hash, policy_hash;
    nt_chuck_observation observation;
    nt_chuck_architect_comparison comparison;
    char policy_path[PATH_BYTES];
} sample;

static void fail(const char *message) {
    fprintf(stderr, "chuck_architect_future: %s\n", message); exit(1);
}
static uint32_t read_u32(FILE *f) {
    unsigned char b[4]; if (fread(b, 1, 4, f) != 4) fail("truncated sample table");
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static uint64_t read_u64(FILE *f) {
    uint64_t low = read_u32(f), high = read_u32(f); return low | high << 32;
}
static float read_float(FILE *f) {
    uint32_t bits = read_u32(f); float value; memcpy(&value, &bits, 4);
    if (!isfinite(value)) fail("nonfinite sample field");
    return value;
}
static long integer(const char *s, long low, long high) {
    char *end; errno = 0; long v = strtol(s, &end, 10);
    if (errno || !*s || *end || v < low || v > high) fail("invalid integer argument");
    return v;
}
static void floats(FILE *f, const float *v, int n) {
    fputc('[', f); for (int i = 0; i < n; ++i) fprintf(f, "%s%.9g", i ? "," : "", v[i]); fputc(']', f);
}
static void set_identity(nt_chuck_architect *a, const char *identity) {
    size_t n = strlen(identity); if (!n || n >= sizeof(a->config.life_id)) fail("invalid life identity");
    for (size_t i = 0; i < n; ++i)
        if (!strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-", identity[i])) fail("invalid life identity");
    memset(a->config.life_id, 0, sizeof(a->config.life_id)); memcpy(a->config.life_id, identity, n);
}
static void copy_weights(nt_chuck_architect *dest, const nt_chuck_architect *source) {
    memcpy(dest->w1, source->w1, sizeof(dest->w1)); memcpy(dest->b1, source->b1, sizeof(dest->b1));
    memcpy(dest->w2, source->w2, sizeof(dest->w2)); memcpy(dest->b2, source->b2, sizeof(dest->b2));
}
static int read_samples(const char *path, sample *samples) {
    FILE *f = fopen(path, "rb"); if (!f) fail("cannot open samples");
    char magic[8]; if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "NTCAFT01", 8)) fail("invalid sample version");
    uint32_t count = read_u32(f); if (!count || count > MAX_SAMPLES) fail("invalid sample count");
    char base[PATH_BYTES]; const char *slash = strrchr(path, '/'); size_t base_len = slash ? (size_t)(slash - path + 1) : 0;
    if (base_len >= sizeof(base)) fail("sample directory too long");
    memcpy(base, path, base_len); base[base_len] = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sample *s = &samples[i]; memset(s, 0, sizeof(*s));
        s->body = read_u32(f); s->seed = read_u32(f); s->checkpoint = read_u32(f);
        s->state_hash = read_u64(f); s->policy_hash = read_u64(f);
        if ((s->body != 1 && s->body != 2) || !s->seed || !s->checkpoint) fail("invalid sample identity");
        nt_chuck_observation *o = &s->observation;
        float *fields[] = {&o->loss,&o->loss_ema,&o->loss_trend,&o->macro_ema,&o->best_macro,
            &o->dampen,&o->lr_scale,&o->noise,&o->grad_norm,&o->grad_trend,&o->frozen_fraction};
        for (int j = 0; j < 11; ++j) *fields[j] = read_float(f);
        int *ints[] = {&o->step,&o->stag,&o->macro_stag,&o->history_len};
        for (int j = 0; j < 4; ++j) { uint32_t v = read_u32(f); if (v > INT_MAX) fail("invalid observation integer"); *ints[j] = (int)v; }
        for (int j = 0; j < NT_CHUCK_ARCHITECT_FEATURES; ++j) s->comparison.features[j] = read_float(f);
        for (int j = 0; j < NT_CHUCK_ARCHITECT_HEADS; ++j) s->comparison.future_loss[j] = read_float(f);
        uint32_t length = read_u32(f); char relative[1024];
        if (!length || length >= sizeof(relative) || fread(relative, 1, length, f) != length) fail("invalid source-life path");
        relative[length] = 0;
        if (relative[0] == '/' || strstr(relative, "..")) fail("source-life path must stay relative");
        for (uint32_t j = 0; j < length; ++j)
            if (!strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-", relative[j])) fail("invalid source-life path");
        int n = snprintf(s->policy_path, sizeof(s->policy_path), "%s%s", base, relative);
        if (n < 0 || (size_t)n >= sizeof(s->policy_path)) fail("source-life path too long");
    }
    if (fgetc(f) != EOF || ferror(f) || fclose(f)) fail("trailing sample data or read error");
    return (int)count;
}
static void source_life(const sample *s, nt_chuck_architect *a) {
    if (nt_chuck_architect_load(a, s->policy_path) || nt_chuck_architect_hash(a) != s->policy_hash)
        fail("source policy identity mismatch");
    float captured[NT_CHUCK_ARCHITECT_FEATURES];
    if (nt_chuck_architect_capture(a, &s->observation, captured) ||
        memcmp(captured, s->comparison.features, sizeof(captured))) fail("source observation/features mismatch");
}
static void sample_identity(FILE *f, const sample *s) {
    fprintf(f, "\"body\":\"%s\",\"seed\":%" PRIu32 ",\"checkpoint\":%" PRIu32 ",\"state_hash\":\"%016" PRIx64 "\",\"source_policy_hash\":\"%016" PRIx64 "\"",
        s->body == 1 ? "simple" : "hevlm", s->seed, s->checkpoint, s->state_hash, s->policy_hash);
}
static int init_main(int argc, char **argv) {
    if (argc != 5) fail("usage: future init SEED OUTPUT_LIFE LIFE_ID");
    nt_chuck_architect_config c; nt_chuck_architect_config_default(&c);
    c.mode = NT_CHUCK_ARCHITECT_LEARNED; c.seed = (uint32_t)integer(argv[2],1,UINT32_MAX);
    c.learning_rate = 0.03f;
    nt_chuck_architect a; if (nt_chuck_architect_init(&a, &c)) fail("initialization refused");
    set_identity(&a, argv[4]);
    if (nt_chuck_architect_save(&a, argv[3])) fail("initial life save failed");
    printf("{\"type\":\"initialized\",\"hash\":\"%016" PRIx64 "\",\"parameters\":%d,\"fit_steps\":0}\n", nt_chuck_architect_hash(&a), NT_CHUCK_ARCHITECT_PARAMETERS);
    return 0;
}
static int fit_main(int argc, char **argv, int conditioned) {
    if (argc != 8) fail("usage: future fit INPUT_LIFE SAMPLES OUTPUT_LIFE TRACE EPOCHS LIFE_ID");
    nt_chuck_architect a; if (nt_chuck_architect_load(&a, argv[2])) fail("input life load failed");
    set_identity(&a, argv[7]); nt_chuck_architect initial = a;
    sample *samples = calloc(MAX_SAMPLES, sizeof(*samples)); if (!samples) fail("sample allocation failed");
    int count = read_samples(argv[3], samples), epochs = (int)integer(argv[6],1,100000);
    for (int i = 0; i < count; ++i) { nt_chuck_architect source; source_life(&samples[i], &source); }
    FILE *trace = fopen(argv[5], "w"); if (!trace) fail("fit trace open failed");
    uint64_t fit_steps = 0;
    fprintf(trace, "{\"type\":\"fit_run\",\"samples\":%d,\"epochs\":%d,\"learning_rate\":%.9g,\"initial_hash\":\"%016" PRIx64 "\",\"life_id\":\"%s\",\"objective\":\"%s\"}\n", count, epochs, a.config.learning_rate, nt_chuck_architect_hash(&a), a.config.life_id, conditioned ? "state-span-huber-v1" : "hold-relative-huber-v1");
    for (int epoch = 0; epoch < epochs; ++epoch) for (int i = 0; i < count; ++i) {
        nt_chuck_architect_comparison_receipt receipt;
        double scale = 0;
        if (conditioned) {
            nt_chuck_architect_conditioned_receipt normalized;
            if (nt_chuck_architect_fit_conditioned(&a, &samples[i].comparison, &normalized))
                fail("conditioned comparison fit refused");
            receipt = normalized.comparison;
            scale = normalized.scale;
        } else if (nt_chuck_architect_fit_comparison(&a, &samples[i].comparison, &receipt)) fail("comparison fit refused");
        ++fit_steps;
        fprintf(trace, "{\"type\":\"fit\",\"fit_step\":%" PRIu64 ",\"epoch\":%d,\"sample_index\":%d,", fit_steps, epoch + 1, i);
        sample_identity(trace, &samples[i]);
        fprintf(trace, ",\"hash_before\":\"%016" PRIx64 "\",\"hash_after\":\"%016" PRIx64 "\",\"future_loss\":", receipt.hash_before, receipt.hash_after);
        floats(trace, receipt.future_loss, 3); fputs(",\"target\":", trace); floats(trace, receipt.target, 3);
        if (conditioned) fprintf(trace, ",\"scale\":%.17g", scale);
        fputs(",\"predicted_before\":", trace); floats(trace, receipt.predicted_before, 3);
        fputs(",\"predicted_after\":", trace); floats(trace, receipt.predicted_after, 3);
        fputs(",\"error_before\":", trace); floats(trace, receipt.error_before, 3);
        fprintf(trace, ",\"huber_before\":%.9g,\"huber_after\":%.9g,\"online_decisions\":%" PRIu64 ",\"online_updates\":%" PRIu64 "}\n",
                receipt.huber_before, receipt.huber_after, receipt.decisions, receipt.updates);
    }
    nt_chuck_architect unchanged = a; copy_weights(&unchanged, &initial);
    if (memcmp(&unchanged, &initial, sizeof(initial))) fail("fit changed non-weight life fields");
    if (nt_chuck_architect_save(&a, argv[4])) fail("fitted life save failed");
    fprintf(trace, "{\"type\":\"fit_summary\",\"fit_steps\":%" PRIu64 ",\"final_hash\":\"%016" PRIx64 "\",\"non_weight_fields_unchanged\":true}\n", fit_steps, nt_chuck_architect_hash(&a));
    if (ferror(trace) || fclose(trace)) fail("fit trace write failed");
    printf("{\"type\":\"fit_complete\",\"fit_steps\":%" PRIu64 ",\"hash\":\"%016" PRIx64 "\"}\n", fit_steps, nt_chuck_architect_hash(&a));
    free(samples); return 0;
}
static int eval_main(int argc, char **argv) {
    if (argc != 6) fail("usage: future eval SAMPLES MODEL_LIFE LABEL TRACE (MODEL_LIFE=- for source/hold/brake/push)");
    sample *samples = calloc(MAX_SAMPLES, sizeof(*samples)); if (!samples) fail("sample allocation failed");
    int count = read_samples(argv[2], samples), forced = -1;
    const char *label = argv[4];
    if (!strcmp(label, "hold")) forced = 0; else if (!strcmp(label, "brake")) forced = 1; else if (!strcmp(label, "push")) forced = 2;
    int use_source = !strcmp(argv[3], "-");
    if (use_source && forced < 0 && strcmp(label, "source")) fail("invalid source readout label");
    nt_chuck_architect model; if (!use_source && nt_chuck_architect_load(&model, argv[3])) fail("model life load failed");
    FILE *trace = fopen(argv[5], "w"); if (!trace) fail("readout trace open failed");
    for (int i = 0; i < count; ++i) {
        sample *s = &samples[i]; nt_chuck_architect source; source_life(s, &source);
        nt_chuck_architect query = source;
        if (!use_source) copy_weights(&query, &model);
        query.config.exploration = 0;
        float features[NT_CHUCK_ARCHITECT_FEATURES], scores[3]; nt_chuck_action action;
        if (nt_chuck_architect_capture(&query, &s->observation, features) || memcmp(features, s->comparison.features, sizeof(features)) ||
            nt_chuck_architect_scores(&query, features, scores) || nt_chuck_architect_select(&query, &s->observation, &action))
            fail("readout refused source context");
        int selected = forced >= 0 ? forced : (int)action.kind - (int)NT_CHUCK_ACTION_HOLD;
        if (selected < 0 || selected > 2) fail("readout selected unavailable action");
        if (nt_chuck_architect_hash(&source) != s->policy_hash) fail("readout changed source life");
        const float *loss = s->comparison.future_loss;
        float best = fminf(loss[0], fminf(loss[1],loss[2]));
        fprintf(trace, "{\"type\":\"readout\",\"label\":\"%s\",", label); sample_identity(trace, s);
        fprintf(trace, ",\"model_hash\":\"%016" PRIx64 "\",\"query_hash\":\"%016" PRIx64 "\",\"scores\":", use_source ? s->policy_hash : nt_chuck_architect_hash(&model), nt_chuck_architect_hash(&query));
        floats(trace, scores, 3); fprintf(trace, ",\"action\":%d,\"forced\":%s,\"future_loss\":%.9g,\"hold_loss\":%.9g,\"advantage\":%.17g,\"relative_advantage\":%.17g,\"regret\":%.17g,\"optimal\":%s}\n",
                selected + NT_CHUCK_ACTION_HOLD, forced >= 0 ? "true" : "false", loss[selected], loss[0],
                (double)loss[0]-loss[selected], ((double)loss[0]-loss[selected])/(fabs((double)loss[0])+1e-6),
                (double)loss[selected]-best, loss[selected] == best ? "true" : "false");
    }
    if (ferror(trace) || fclose(trace)) fail("readout trace write failed");
    free(samples); return 0;
}
int main(int argc, char **argv) {
    if (argc < 2) fail("expected init, fit, fit-conditioned or eval");
    if (!strcmp(argv[1], "init")) return init_main(argc,argv);
    if (!strcmp(argv[1], "fit")) return fit_main(argc,argv,0);
    if (!strcmp(argv[1], "fit-conditioned")) return fit_main(argc,argv,1);
    if (!strcmp(argv[1], "eval")) return eval_main(argc,argv);
    fail("unknown command"); return 1;
}
