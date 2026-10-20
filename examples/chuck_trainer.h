/* One native route from a UTF-8 corpus to a complete, resumable Chuck life. */
#ifndef NOTORCH_CHUCK_TRAINER_H
#define NOTORCH_CHUCK_TRAINER_H

#include "chuck_trainer_state.h"
#include "chuck_trainer_checkpoint.h"
#include "sha256.h"
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

static volatile sig_atomic_t trainer_stop;

static void trainer_signal(int signal_number) { trainer_stop = signal_number; }

static void trainer_usage(FILE *out) {
#ifdef NT_CHUCK_Q_TRAINER
    fputs("Usage:\n"
          "  train_chuck_q train --corpus TEXT --merges Q.MERGES --checkpoint FILE --steps N\n"
          "      [--seed N --lr F --save-every N --acquire-every N --trace FILE]\n"
          "  train_chuck_q resume --corpus TEXT --checkpoint INPUT --output FILE --steps TOTAL\n"
          "      [--save-every N --trace FILE]\n"
          "  train_chuck_q generate --checkpoint FILE [--prompt TEXT --tokens N]\n"
          "      [--seed N --temperature F]\n"
          "  train_chuck_q export --checkpoint FILE --output QPTQ.BIN\n\n"
          "Q rrpram3_janus3, sequence64, learned positions128, byte BPE stored in checkpoint.\n"
          "Defaults: seed=42, lr=.0003, save-every=128, acquire-every=64.\n"
          "Resume --steps is TOTAL completed steps. SIGINT/SIGTERM saves a completed step.\n"
          "Generation: 256 BPE tokens, temperature=.8; 0 is greedy. CPU training.\n", out);
#else
    fputs("Usage:\n"
          "  train_chuck train --corpus TEXT --checkpoint FILE --steps N\n"
          "      [--seed N --lr F --save-every N --acquire-every N --trace FILE]\n"
          "  train_chuck resume --corpus TEXT --checkpoint INPUT --output FILE --steps TOTAL\n"
          "      [--save-every N --trace FILE]\n"
          "  train_chuck generate --checkpoint FILE [--prompt TEXT --tokens N]\n"
          "      [--seed N --temperature F]\n\n"
          "Training defaults: seed=42, lr=.0003, save-every=128, acquire-every=64.\n"
          "Resume --steps is the total completed step, not additional steps; train --steps 0 saves initialization.\n"
          "Generation defaults: 256 Unicode characters, seed=42, temperature=.8; 0 is greedy.\n"
          "UTF-8 corpus: 2..4096 distinct characters, at most 128 MiB; final 10% held out.\n"
          "SIGINT/SIGTERM saves the last completed step before exiting. CPU training.\n", out);
#endif
}

static void trainer_error(const char *message) {
    fprintf(stderr, "train_chuck: %s\n", message);
    exit(1);
}

static uint64_t trainer_uint(const char *text, uint64_t low, uint64_t high) {
    char *end;
    if (!text[0] || text[0] == '-' || text[0] == '+') trainer_error("invalid integer argument");
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value < low || value > high) trainer_error("integer argument out of range");
    return (uint64_t)value;
}

static float trainer_float(const char *text, float low, float high, int zero_ok) {
    char *end;
    errno = 0;
    float value = strtof(text, &end);
    if (!text[0] || errno || *end || !isfinite(value) || value < low || value > high ||
        (!zero_ok && value == 0)) trainer_error("floating-point argument out of range");
    return value;
}

/* Real paths also identify aliases of a destination which does not exist yet. */
static char *trainer_target(const char *path) {
    char *resolved = realpath(path, NULL);
    if (resolved) return resolved;
    char *copy = strdup(path);
    if (!copy) trainer_error("path allocation failed");
    char *slash = strrchr(copy, '/');
    const char *base = slash ? slash + 1 : copy;
    if (!base[0] || !strcmp(base, ".") || !strcmp(base, "..")) trainer_error("invalid file destination");
    char *leaf = strdup(base);
    if (!leaf) trainer_error("path allocation failed");
    if (slash) *slash = 0;
    const char *parent = slash ? (copy[0] ? copy : "/") : ".";
    char *directory = realpath(parent, NULL);
    if (!directory) trainer_error("file parent directory does not exist");
    size_t size = strlen(directory) + strlen(leaf) + 2;
    resolved = malloc(size);
    if (!resolved) trainer_error("path allocation failed");
    snprintf(resolved, size, "%s/%s", directory, leaf);
    free(copy); free(leaf); free(directory);
    return resolved;
}

static int trainer_same_file(const char *a, const char *b) {
    if (!a || !b) return 0;
    struct stat left, right;
    if (!stat(a, &left) && !stat(b, &right) && left.st_dev == right.st_dev && left.st_ino == right.st_ino)
        return 1;
    char *x = trainer_target(a), *y = trainer_target(b);
    int same = !strcmp(x, y);
    free(x); free(y);
    return same;
}

static uint32_t trainer_utf8(const unsigned char *bytes, size_t size, size_t *at) {
    size_t p = *at;
    if (p >= size) trainer_error("truncated UTF-8");
    uint32_t value = bytes[p++], minimum = 0;
    unsigned remaining = 0;
    if (value < 0x80) { /* ASCII. */ }
    else if (value >= 0xc2 && value <= 0xdf) { value &= 31; remaining = 1; minimum = 0x80; }
    else if (value >= 0xe0 && value <= 0xef) { value &= 15; remaining = 2; minimum = 0x800; }
    else if (value >= 0xf0 && value <= 0xf4) { value &= 7; remaining = 3; minimum = 0x10000; }
    else trainer_error("invalid UTF-8 text");
    if (remaining > size - p) trainer_error("truncated UTF-8 text");
    while (remaining--) {
        unsigned byte = bytes[p++];
        if ((byte & 0xc0) != 0x80) trainer_error("invalid UTF-8 continuation");
        value = (value << 6) | (byte & 63);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff) || !value)
        trainer_error("text must contain Unicode scalar values without NUL");
    *at = p;
    return value;
}

static int trainer_u32_compare(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int trainer_token(const trainer_state *state, uint32_t codepoint) {
    size_t low = 0, high = state->vocab_size;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (state->vocab[mid] < codepoint) low = mid + 1;
        else high = mid;
    }
    return low < state->vocab_size && state->vocab[low] == codepoint ? (int)low : -1;
}

static uint32_t *trainer_corpus(trainer_state *state, const char *path, int resume, const char *merges) {
    FILE *f = fopen(path, "rb");
    if (!f) trainer_error("cannot open corpus");
    if (fseek(f, 0, SEEK_END)) trainer_error("cannot seek corpus");
    long length = ftell(f);
    if (length <= 0 || length > 128L * 1024 * 1024) trainer_error("corpus must be 1 byte to 128 MiB");
    rewind(f);
    size_t bytes = (size_t)length;
    unsigned char *raw = malloc(bytes);
    uint32_t *tokens = NULL;
    if (!raw) trainer_error("corpus allocation failed");
    if (fread(raw, 1, bytes, f) != bytes || ferror(f) || fclose(f)) trainer_error("cannot read corpus");
    unsigned char hash[32];
    if (nt_sha256(raw, bytes, hash)) trainer_error("corpus SHA-256 failed");
    if (resume && memcmp(hash, state->corpus_sha256, 32)) trainer_error("corpus SHA-256 differs from checkpoint");
    size_t count = 0;
#ifdef NT_CHUCK_Q_TRAINER
    char error[256];
    if (!resume) {
        if (q_tokenizer_load(&state->tokenizer, merges, error, sizeof(error))) trainer_error(error);
        state->vocab_size = state->tokenizer.vocab_size;
    }
    if (q_tokenizer_encode(&state->tokenizer, raw, bytes, &tokens, &count, error, sizeof(error)))
        trainer_error(error);
#else
    (void)merges;
    tokens = malloc(bytes * sizeof(*tokens));
    if (!tokens) trainer_error("corpus allocation failed");
    size_t at = 0;
    while (at < bytes) tokens[count++] = trainer_utf8(raw, bytes, &at);
#endif
    free(raw);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) trainer_error("corpus train and held-out splits must both exceed context length");
    if (resume) {
        if (state->token_count != count || state->split != split) trainer_error("corpus token layout differs from checkpoint");
    } else {
#ifndef NT_CHUCK_Q_TRAINER
        uint32_t *sorted = malloc(count * sizeof(*sorted));
        if (!sorted) trainer_error("vocabulary allocation failed");
        memcpy(sorted, tokens, count * sizeof(*sorted));
        qsort(sorted, count, sizeof(*sorted), trainer_u32_compare);
        state->vocab = malloc(TRAINER_MAX_VOCAB * sizeof(*state->vocab));
        if (!state->vocab) trainer_error("vocabulary allocation failed");
        for (size_t i = 0; i < count; ++i) if (!i || sorted[i] != sorted[i - 1]) {
            if (state->vocab_size == TRAINER_MAX_VOCAB) trainer_error("corpus vocabulary exceeds 4096 characters");
            state->vocab[state->vocab_size++] = sorted[i];
        }
        free(sorted);
        if (state->vocab_size < 2) trainer_error("corpus needs at least two distinct characters");
#endif
        memcpy(state->corpus_sha256, hash, 32);
        state->token_count = count; state->split = split;
    }
#ifndef NT_CHUCK_Q_TRAINER
    for (size_t i = 0; i < count; ++i) {
        int id = trainer_token(state, tokens[i]);
        if (id < 0) trainer_error("corpus character missing from checkpoint vocabulary");
        tokens[i] = (uint32_t)id;
    }
#endif
    return tokens;
}

static void trainer_save(const trainer_state *state, const char *path) {
    char error[256];
    if (trainer_checkpoint_save(state, path, error, sizeof(error))) trainer_error(error);
    fprintf(stderr, "checkpoint: step=%" PRIu64 " %s\n", state->step, path);
}

static void trainer_acquire(trainer_state *state, const uint32_t *data, size_t offset,
                            float before, FILE *trace) {
    body *m = &state->model;
    nt_chuck_architect *a = &state->architect;
    scenario_snapshot *saved = scenario_capture(m, a, state->windows);
    nt_chuck_observation observation;
    nt_chuck_action selected_before, selected_after;
    nt_chuck_architect_comparison sample;
    float scores_before[NT_CHUCK_ARCHITECT_HEADS], scores_after[NT_CHUCK_ARCHITECT_HEADS];
    if (nt_tape_chuck_observe(before, &observation) ||
        nt_chuck_architect_capture(a, &observation, sample.features) ||
        nt_chuck_architect_scores(a, sample.features, scores_before) ||
        nt_chuck_architect_select(a, &observation, &selected_before)) trainer_error("acquisition observation refused");
    uint64_t old_policy = nt_chuck_architect_hash(a);
    size_t offsets[SCENARIO_HORIZON + SCENARIO_PROBES];
    uint64_t windows[SCENARIO_HORIZON + SCENARIO_PROBES], lookahead = state->windows;
    offsets[0] = offset; windows[0] = lookahead;
    for (int i = 1; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) {
        offsets[i] = (size_t)(next_window(&lookahead) % (state->split - CTX));
        windows[i] = lookahead;
    }
    for (int ai = 0; ai < NT_CHUCK_ARCHITECT_HEADS; ++ai) {
        scenario_restore(saved, m, a, &state->windows);
        for (int h = 0; h < SCENARIO_HORIZON; ++h) {
            float loss = before;
            if (h) {
                int index = body_forward(m, data, offsets[h], 1);
                loss = read_loss(index);
                nt_tape_backward(index);
                if (!isfinite(nt_tape_clip_grads(1.0f))) trainer_error("nonfinite branch gradient");
                state->windows = windows[h];
            }
            lived_window window = {m, data, offsets[h]};
            nt_chuck_architect_decision decision;
            nt_chuck_architect_receipt receipt;
            if (!h) {
                nt_chuck_action action = {(nt_chuck_action_kind)(NT_CHUCK_ACTION_HOLD + ai), 0};
                if (nt_chuck_architect_intervene(a, state->lr, loss, &action, lived_after,
                        &window, &decision, &receipt)) trainer_error("acquisition intervention refused");
            } else {
                if (nt_chuck_architect_step(a, state->lr, loss, &decision) ||
                    nt_chuck_architect_feedback_frozen(a, lived_after(&window), &receipt))
                    trainer_error("acquisition continuation refused");
            }
            if (!isfinite(receipt.after_loss)) trainer_error("nonfinite acquisition consequence");
            ++state->branch_updates;
        }
        sample.future_loss[ai] = lived_probe(m, data, offsets + SCENARIO_HORIZON, SCENARIO_PROBES);
        if (!isfinite(sample.future_loss[ai])) trainer_error("nonfinite acquisition probe");
    }
    scenario_restore(saved, m, a, &state->windows);
    if (!scenario_matches(saved, m, a, state->windows)) trainer_error("acquisition source restore differs");
    nt_chuck_architect_conditioned_receipt fitted;
    for (int i = 0; i < LIVING_FITS; ++i) {
        if (nt_chuck_architect_fit_conditioned(a, &sample, &fitted)) trainer_error("acquisition fitting refused");
        ++state->fit_updates;
    }
    nt_chuck_architect non_weights = *a;
    rollout_weights(&non_weights, &saved->architect);
    if (memcmp(&non_weights, &saved->architect, sizeof(non_weights)) ||
        !scenario_matches(saved, m, &non_weights, state->windows) ||
        nt_chuck_architect_scores(a, sample.features, scores_after) ||
        nt_chuck_architect_select(a, &observation, &selected_after)) trainer_error("acquisition changed source world");
    ++state->acquisitions;
    if (trace) {
        fprintf(trace, "{\"type\":\"acquisition\",\"step\":%" PRIu64 ",\"horizon\":%d,\"fits\":%d,"
                "\"source_restored\":true,\"action_before\":%d,\"action_after\":%d,"
                "\"policy_before\":\"%016" PRIx64 "\",\"policy_after\":\"%016" PRIx64 "\","
                "\"measured_future_loss\":[%.9g,%.9g,%.9g],\"scale\":%.17g,\"features\":",
                state->step + 1, SCENARIO_HORIZON, LIVING_FITS, selected_before.kind, selected_after.kind,
                old_policy, nt_chuck_architect_hash(a), sample.future_loss[0], sample.future_loss[1],
                sample.future_loss[2], fitted.scale);
        living_floats(trace, sample.features, NT_CHUCK_ARCHITECT_FEATURES);
        fputs(",\"scores_before\":", trace); living_floats(trace, scores_before, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"scores_after\":", trace); living_floats(trace, scores_after, NT_CHUCK_ARCHITECT_HEADS);
        fputs("}\n", trace);
    }
    scenario_release(saved);
}

static void trainer_emit(uint32_t c) {
    unsigned char bytes[4];
    size_t n;
    if (c < 0x80) { bytes[0] = (unsigned char)c; n = 1; }
    else if (c < 0x800) { bytes[0] = (unsigned char)(0xc0 | (c >> 6)); bytes[1] = (unsigned char)(0x80 | (c & 63)); n = 2; }
    else if (c < 0x10000) {
        bytes[0] = (unsigned char)(0xe0 | (c >> 12)); bytes[1] = (unsigned char)(0x80 | ((c >> 6) & 63));
        bytes[2] = (unsigned char)(0x80 | (c & 63)); n = 3;
    } else {
        bytes[0] = (unsigned char)(0xf0 | (c >> 18)); bytes[1] = (unsigned char)(0x80 | ((c >> 12) & 63));
        bytes[2] = (unsigned char)(0x80 | ((c >> 6) & 63)); bytes[3] = (unsigned char)(0x80 | (c & 63)); n = 4;
    }
    if (fwrite(bytes, 1, n, stdout) != n) trainer_error("generation output failed");
}

static void trainer_generate(trainer_state *state, const char *prompt, uint64_t count,
                              uint32_t seed, float temperature) {
    uint32_t context[CTX + 1] = {0};
    size_t used = 0, at = 0, size = strlen(prompt);
#ifdef NT_CHUCK_Q_TRAINER
    uint32_t *ids = NULL;
    size_t count_ids = 0;
    char error[256];
    if (q_tokenizer_encode(&state->tokenizer, (const unsigned char *)prompt, size,
                           &ids, &count_ids, error, sizeof(error))) trainer_error(error);
    used = count_ids < CTX ? count_ids : CTX;
    if (used) memcpy(context, ids + count_ids - used, used * sizeof(*context));
    free(ids);
    (void)at;
    if (!used) context[used++] = '\n';
#else
    while (at < size) {
        uint32_t cp = trainer_utf8((const unsigned char *)prompt, size, &at);
        int id = trainer_token(state, cp);
        if (id < 0) trainer_error("prompt contains a character outside the saved vocabulary");
        if (used == CTX) { memmove(context, context + 1, (CTX - 1) * sizeof(*context)); --used; }
        context[used++] = (uint32_t)id;
    }
    if (!used) {
        int newline = trainer_token(state, '\n');
        context[used++] = newline < 0 ? 0 : (uint32_t)newline;
    }
#endif
    float *weights = malloc(state->vocab_size * sizeof(*weights));
    if (!weights) trainer_error("generation allocation failed");
    uint64_t rng;
    nt_rng_seed(&rng, seed);
    if (fwrite(prompt, 1, size, stdout) != size) trainer_error("generation output failed");
    for (uint64_t i = 0; i < count; ++i) {
        for (size_t j = used; j <= CTX; ++j) context[j] = 0;
        int loss = body_forward(&state->model, context, 0, 0);
        nt_tape *tape = nt_tape_get();
        int logits_index = tape->entries[loss].parent1;
        if (logits_index < 0 || logits_index >= tape->count) trainer_error("generation logits missing");
        const float *logits = tape->entries[logits_index].output->data + (used - 1) * state->vocab_size;
        int token = 0;
        for (uint32_t k = 0; k < state->vocab_size; ++k) {
            if (!isfinite(logits[k])) trainer_error("nonfinite generation logits");
            if (logits[k] > logits[token]) token = (int)k;
        }
        if (temperature > 0) {
            float maximum = logits[token];
            for (uint32_t k = 0; k < state->vocab_size; ++k) weights[k] = expf((logits[k] - maximum) / temperature);
            if (nt_rng_categorical(&rng, weights, (int)state->vocab_size, 1, &token)) trainer_error("sampling refused");
        }
#ifdef NT_CHUCK_Q_TRAINER
        if (q_tokenizer_emit(&state->tokenizer, (uint32_t)token, stdout, error, sizeof(error))) trainer_error(error);
#else
        trainer_emit(state->vocab[token]);
#endif
        if (used == CTX) { memmove(context, context + 1, (CTX - 1) * sizeof(*context)); --used; }
        context[used++] = (uint32_t)token;
    }
    if (fflush(stdout)) trainer_error("generation output flush failed");
    free(weights);
}

static int chuck_trainer_main(int argc, char **argv) {
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) { trainer_usage(stdout); return 0; }
    if (argc < 2) { trainer_usage(stderr); return 2; }
    int fresh = !strcmp(argv[1], "train"), resume = !strcmp(argv[1], "resume"), generate = !strcmp(argv[1], "generate");
    int export_weights = 0;
#ifdef NT_CHUCK_Q_TRAINER
    export_weights = !strcmp(argv[1], "export");
#endif
    int read_only = generate || export_weights;
    if (!fresh && !resume && !read_only) { trainer_usage(stderr); return 2; }
    const char *corpus = NULL, *checkpoint = NULL, *output = NULL, *trace_path = NULL, *prompt = "";
    const char *merges = NULL;
    uint64_t steps = 0, tokens = 256, save_every = 128;
    uint32_t seed = 42, acquire_every = 64;
    float lr = .0003f, temperature = .8f;
    unsigned seen = 0;
    for (int i = 2; i < argc; i += 2) {
        if (!strcmp(argv[i], "--help")) { trainer_usage(stdout); return 0; }
        if (i + 1 == argc) trainer_error("option is missing its value");
        const char *name = argv[i], *value = argv[i + 1];
        unsigned flag = 0;
        if (!strcmp(name, "--corpus") && !read_only) { flag = 1u << 0; corpus = value; }
        else if (!strcmp(name, "--checkpoint")) { flag = 1u << 1; checkpoint = value; }
        else if (!strcmp(name, "--output") && (resume || export_weights)) { flag = 1u << 2; output = value; }
        else if (!strcmp(name, "--steps") && !read_only) { flag = 1u << 3; steps = trainer_uint(value, 0, 1000000); }
        else if (!strcmp(name, "--seed") && (fresh || generate)) { flag = 1u << 4; seed = (uint32_t)trainer_uint(value, 1, UINT32_MAX); }
        else if (!strcmp(name, "--lr") && fresh) { flag = 1u << 5; lr = trainer_float(value, 0, 1, 0); }
        else if (!strcmp(name, "--save-every") && !read_only) { flag = 1u << 6; save_every = trainer_uint(value, 1, 1000000); }
        else if (!strcmp(name, "--acquire-every") && fresh) { flag = 1u << 7; acquire_every = (uint32_t)trainer_uint(value, 1, 1000000); }
        else if (!strcmp(name, "--trace") && !read_only) { flag = 1u << 8; trace_path = value; }
        else if (!strcmp(name, "--prompt") && generate) { flag = 1u << 9; prompt = value; }
        else if (!strcmp(name, "--tokens") && generate) { flag = 1u << 10; tokens = trainer_uint(value, 1, 1000000); }
        else if (!strcmp(name, "--temperature") && generate) { flag = 1u << 11; temperature = trainer_float(value, 0, 10, 1); }
#ifdef NT_CHUCK_Q_TRAINER
        else if (!strcmp(name, "--merges") && fresh) { flag = 1u << 12; merges = value; }
#endif
        else trainer_error("unknown option for this command");
        if (seen & flag) trainer_error("duplicate option");
        seen |= flag;
    }
    if (!checkpoint || (!read_only && (!corpus || !(seen & (1u << 3)))) || ((resume || export_weights) && !output)) trainer_error("missing required option; use --help");
#ifdef NT_CHUCK_Q_TRAINER
    if (fresh && !merges) trainer_error("Q training requires --merges");
#endif
    if (nt_get_gpu_mode()) trainer_error("this trainer requires the CPU backend");
    if (fresh) output = checkpoint;
    if (!read_only && trainer_same_file(corpus, output)) trainer_error("checkpoint would overwrite corpus");
    if (export_weights && trainer_same_file(checkpoint, output)) trainer_error("export would overwrite checkpoint");
    if (trace_path && (trainer_same_file(trace_path, corpus) || trainer_same_file(trace_path, checkpoint) ||
                       trainer_same_file(trace_path, output) || trainer_same_file(trace_path, merges))) trainer_error("trace aliases an input or checkpoint");
    if (fresh || export_weights) {
        struct stat existing;
        if (!lstat(output, &existing)) trainer_error("checkpoint already exists; use resume or a new path");
        if (errno != ENOENT) trainer_error("cannot inspect checkpoint destination");
    }
    trainer_state state = {0};
    uint32_t *data = NULL;
    char error[256];
    if (resume || read_only) {
        if (trainer_checkpoint_load(&state, checkpoint, error, sizeof(error))) trainer_error(error);
    } else {
        state.seed = seed; state.windows = seed; state.lr = lr; state.acquire_every = acquire_every;
        data = trainer_corpus(&state, corpus, 0, merges);
        nt_seed(seed);
        if (nt_chuck_rng_set(UINT32_C(2463534242))) trainer_error("noise RNG initialization refused");
        body_init_vocab(&state.model, "simple", (int)state.vocab_size);
        nt_chuck_architect_config config;
        if (nt_chuck_architect_config_parse_json(&config,
#ifdef NT_CHUCK_Q_TRAINER
                "{\"mode\":\"learned\",\"life_id\":\"chuck-q-h16\",\"exploration\":0,"
#else
                "{\"mode\":\"learned\",\"life_id\":\"chuck-trainer-h16\",\"exploration\":0,"
#endif
                "\"enabled_actions\":[\"hold\",\"brake\",\"push\"]}", error, sizeof(error))) trainer_error(error);
        config.seed = seed;
        if (nt_chuck_architect_init(&state.architect, &config)) trainer_error("policy initialization refused");
    }
    if (generate) trainer_generate(&state, prompt, tokens, seed, temperature);
#ifdef NT_CHUCK_Q_TRAINER
    else if (export_weights) {
        if (q_body_export(&state.model, output, error, sizeof(error))) trainer_error(error);
        fprintf(stderr, "QPTQ: %s\n", output);
    }
#endif
    else {
        if (resume) data = trainer_corpus(&state, corpus, 1, NULL);
        if (steps < state.step) trainer_error("--steps precedes checkpoint's completed step");
        FILE *trace = trace_path ? fopen(trace_path, "wx") : NULL;
        if (trace_path && !trace) trainer_error("trace must be a new writable file");
        if (trace) setvbuf(trace, NULL, _IOLBF, 0);
        fprintf(stderr,
#ifdef NT_CHUCK_Q_TRAINER
                "Q rrpram3_janus3: "
#else
                "SimpleLLM: "
#endif
                "%ld parameters, vocabulary=%u, context=%d, step=%" PRIu64 "/%" PRIu64
                ", Chuck=163 parameters\n", state.model.elements, state.vocab_size, CTX, state.step, steps);
        float heldout = evaluate(&state.model, data, (size_t)state.split, (size_t)state.token_count);
        fprintf(stderr, "step=%" PRIu64 " heldout=%.6f\n", state.step, heldout);
        if (trace) fprintf(trace, "{\"type\":\"start\",\"step\":%" PRIu64 ",\"target\":%" PRIu64
                ",\"parameters\":%ld,\"vocabulary\":%u,\"context\":%d,\"heldout\":%.9g,"
                "\"policy_hash\":\"%016" PRIx64 "\"}\n", state.step, steps, state.model.elements,
                state.vocab_size, CTX, heldout, nt_chuck_architect_hash(&state.architect));
        struct sigaction handler;
        memset(&handler, 0, sizeof(handler)); handler.sa_handler = trainer_signal;
        sigemptyset(&handler.sa_mask);
        trainer_stop = 0;
        if (sigaction(SIGINT, &handler, NULL) || sigaction(SIGTERM, &handler, NULL)) trainer_error("cannot install checkpoint signal handlers");
        trainer_save(&state, output);
        uint64_t saved_step = state.step;
        while (state.step < steps && !trainer_stop) {
            size_t offset = (size_t)(next_window(&state.windows) % (state.split - CTX));
            int index = body_forward(&state.model, data, offset, 1);
            float before = read_loss(index);
            nt_tape_backward(index);
            float gradient_norm = nt_tape_clip_grads(1.0f);
            if (!isfinite(gradient_norm)) trainer_error("nonfinite training gradient");
            if (state.step % state.acquire_every == 0) trainer_acquire(&state, data, offset, before, trace);
            nt_chuck_architect_decision decision;
            nt_chuck_architect_receipt receipt;
            if (nt_chuck_architect_step(&state.architect, state.lr, before, &decision)) trainer_error("training action refused");
            float after = read_loss(body_forward(&state.model, data, offset, 0));
            if (nt_chuck_architect_feedback_frozen(&state.architect, after, &receipt)) trainer_error("training consequence refused");
            ++state.step;
            if (decision.action.kind < NT_CHUCK_ACTION_HOLD || decision.action.kind > NT_CHUCK_ACTION_PUSH)
                trainer_error("unexpected training action");
            ++state.action_counts[decision.action.kind];
            if (state.architect.pending || state.architect.decisions != state.step || state.architect.updates != state.step)
                trainer_error("training chronology differs");
            if (trace) fprintf(trace, "{\"type\":\"step\",\"step\":%" PRIu64 ",\"offset\":%zu,"
                "\"before\":%.9g,\"after\":%.9g,\"gradient_norm\":%.9g,\"action\":%d,"
                "\"window_rng\":\"%016" PRIx64 "\",\"noise_rng\":%" PRIu32 ","
                "\"policy_hash\":\"%016" PRIx64 "\"}\n", state.step, offset, before, after,
                gradient_norm, decision.action.kind, state.windows, nt_chuck_rng_get(), nt_chuck_architect_hash(&state.architect));
            if (state.step % 128 == 0 || state.step == steps || trainer_stop) {
                heldout = evaluate(&state.model, data, (size_t)state.split, (size_t)state.token_count);
                fprintf(stderr, "step=%" PRIu64 "/%" PRIu64 " loss=%.6f heldout=%.6f acquisitions=%" PRIu64
                        "\n", state.step, steps, before, heldout, state.acquisitions);
                if (trace) fprintf(trace, "{\"type\":\"evaluation\",\"step\":%" PRIu64 ",\"heldout\":%.9g}\n", state.step, heldout);
            }
            if (state.step % save_every == 0 || state.step == steps || trainer_stop) {
                trainer_save(&state, output); saved_step = state.step;
            }
            if (trace && ferror(trace)) trainer_error("training trace write failed");
        }
        /* Zero-step initialization/resave and signals before the first update. */
        if (saved_step != state.step) trainer_save(&state, output);
        if (trace) {
            fprintf(trace, "{\"type\":\"complete\",\"step\":%" PRIu64 ",\"interrupted\":%s,"
                    "\"acquisitions\":%" PRIu64 ",\"fit_updates\":%" PRIu64 ",\"branch_updates\":%" PRIu64
                    ",\"action_counts\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]}\n",
                    state.step, trainer_stop ? "true" : "false", state.acquisitions, state.fit_updates,
                    state.branch_updates, state.action_counts[0], state.action_counts[1], state.action_counts[2], state.action_counts[3]);
            if (ferror(trace) || fclose(trace)) trainer_error("training trace close failed");
        }
    }
    nt_tape_destroy();
    for (int i = 0; i < state.model.count; ++i) nt_tensor_free(state.model.param[i]);
    free(state.vocab); free(data);
    return trainer_stop ? 128 + trainer_stop : 0;
}

#endif
