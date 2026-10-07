/* SPA future credit from retained sentence-action consequences.
 * Python joins and verifies source records; this executable validates their
 * typed import, fits the native 267-parameter policy, and records every fit and
 * readout. No sentence generation or online agent operation occurs here.
 * Protocol: experiments/spa_agent/future/protocol.json.
 */
#include "spa_agent.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SAMPLES 256u
#define TRAIN_SAMPLES 48u
#define EPOCHS 512u
#define FIT_RATE 0.03f
#define PATH_BYTES 4096u
#define HASH_BYTES 65u

typedef struct {
    FILE *file;
    uint64_t hash;
} reader;

typedef struct {
    uint32_t ordinal, seed, episode, step, agent_rng;
    uint64_t body_hash, host_rng, feature_witness;
    char snapshot_sha[HASH_BYTES], outcome_sha[2][NT_SPA_AGENT_ACTIONS][HASH_BYTES];
    nt_spa_experience experience;
    nt_spa_comparison comparison[2];
} sample;

typedef struct {
    char protocol_sha[HASH_BYTES], archive_sha[HASH_BYTES];
    uint32_t count;
    uint64_t file_hash;
    sample samples[MAX_SAMPLES];
} dataset;

typedef struct {
    char protocol_sha[HASH_BYTES], archive_sha[HASH_BYTES];
    uint64_t dataset_hash, life_hash[4];
} seal;

static const char *const model_names[4] = {"initial", "h4", "h0", "shuffled_h4"};
static const char *const arm_names[3] = {"h4", "h0", "shuffled_h4"};

static void fail(const char *message) {
    fprintf(stderr, "spa_agent_future: %s\n", message);
    exit(1);
}

static void check(int status, const char *message) {
    if (status != NT_SPA_OK) {
        fprintf(stderr, "spa_agent_future: %s (%d)\n", message, status);
        exit(1);
    }
}

static uint64_t hash_byte(uint64_t hash, unsigned byte) {
    return (hash ^ byte) * UINT64_C(1099511628211);
}

static uint64_t hash_integer(uint64_t hash, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) hash = hash_byte(hash, (unsigned)((value >> (8 * i)) & 255u));
    return hash;
}

static uint64_t feature_witness(const nt_spa_experience *experience) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = hash_integer(hash, experience->source_life_hash, 8);
    hash = hash_integer(hash, experience->sentence_index, 4);
    hash = hash_integer(hash, experience->sentence_count, 4);
    for (int i = 0; i < NT_SPA_AGENT_FEATURES; i++) {
        uint32_t bits;
        memcpy(&bits, &experience->features[i], sizeof(bits));
        hash = hash_integer(hash, bits, 4);
    }
    return hash;
}

static int read_byte(reader *in) {
    int byte = fgetc(in->file);
    if (byte != EOF) in->hash = hash_byte(in->hash, (unsigned)byte);
    return byte;
}

static int token(reader *in, char out[128], int required) {
    int byte;
    do { byte = read_byte(in); } while (byte != EOF && isspace((unsigned char)byte));
    if (byte == EOF) {
        if (ferror(in->file)) fail("dataset read failed");
        if (required) fail("truncated dataset");
        return 0;
    }
    unsigned length = 0;
    do {
        if (byte < 33 || byte > 126 || length == 127) fail("invalid or overlong dataset token");
        out[length++] = (char)byte;
        byte = read_byte(in);
    } while (byte != EOF && !isspace((unsigned char)byte));
    out[length] = 0;
    return 1;
}

static void expect(reader *in, const char *label) {
    char value[128]; token(in, value, 1);
    if (strcmp(value, label)) fail("unexpected dataset label");
}

static uint32_t read_u32(reader *in) {
    char value[128], *end;
    token(in, value, 1);
    for (const char *p = value; *p; p++) if (*p < '0' || *p > '9') fail("invalid unsigned dataset integer");
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || *end || parsed > UINT32_MAX) fail("dataset integer outside uint32");
    return (uint32_t)parsed;
}

static void read_hex(reader *in, char *out, unsigned length) {
    char value[128]; token(in, value, 1);
    if (strlen(value) != length) fail("invalid identity length");
    for (unsigned i = 0; i < length; i++)
        if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f')))
            fail("identity must use lowercase hexadecimal");
    memcpy(out, value, length + 1);
}

static uint64_t read_hex64(reader *in) {
    char value[17]; read_hex(in, value, 16);
    uint64_t result = 0;
    for (unsigned i = 0; i < 16; i++)
        result = (result << 4) | (uint64_t)(value[i] <= '9' ? value[i] - '0' : value[i] - 'a' + 10);
    return result;
}

static float read_float(reader *in) {
    char value[128], *end;
    token(in, value, 1);
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno || end == value || *end || !isfinite(parsed)) fail("invalid finite dataset float");
    return parsed;
}

static nt_spa_metrics read_metrics(reader *in) {
    nt_spa_metrics metrics;
    float *fields[] = {&metrics.local_connectedness, &metrics.global_connectedness,
        &metrics.coherence, &metrics.novelty, &metrics.repetition, &metrics.collapse, &metrics.continuity};
    for (unsigned i = 0; i < 7; i++) {
        *fields[i] = read_float(in);
        if (*fields[i] < 0 || *fields[i] > 1) fail("raw metric outside [0,1]");
    }
    return metrics;
}

static int same_metrics(const nt_spa_metrics *a, const nt_spa_metrics *b) {
    return a->local_connectedness == b->local_connectedness &&
        a->global_connectedness == b->global_connectedness && a->coherence == b->coherence &&
        a->novelty == b->novelty && a->repetition == b->repetition &&
        a->collapse == b->collapse && a->continuity == b->continuity;
}

static nt_spa_action action_at(unsigned kind, const nt_spa_experience *experience) {
    nt_spa_action action = {(nt_spa_action_kind)kind, experience->sentence_index, NT_SPA_AGENT_NO_SOURCE};
    if (kind == NT_SPA_RESEED_LEFT) action.source = action.target ? action.target - 1 : NT_SPA_AGENT_NO_SOURCE;
    if (kind == NT_SPA_RESEED_RIGHT) action.source = action.target + 1;
    return action;
}

static uint32_t available_mask(const nt_spa_experience *experience) {
    uint32_t mask = 1u << NT_SPA_KEEP;
    if (experience->sentence_index > 0) mask |= 1u << NT_SPA_RESEED_LEFT;
    if (experience->sentence_index + 1 < experience->sentence_count) mask |= 1u << NT_SPA_RESEED_RIGHT;
    return mask;
}

static int same_action(const nt_spa_action *a, const nt_spa_action *b) {
    return a->kind == b->kind && a->target == b->target && a->source == b->source;
}

static void check_experience(const sample *source, const nt_spa_experience *experience) {
    check(nt_spa_experience_validate(experience), "invalid frozen experience");
    if (feature_witness(experience) != source->feature_witness)
        fail("source-feature witness mismatch");
}

static dataset *read_dataset(const char *path) {
    dataset *data = calloc(1, sizeof(*data));
    if (!data) fail("dataset allocation failed");
    reader in = {fopen(path, "rb"), UINT64_C(14695981039346656037)};
    if (!in.file) fail("cannot open dataset");
    expect(&in, "NT_SPA_FUTURE_V1");
    expect(&in, "PROTOCOL"); read_hex(&in, data->protocol_sha, 64);
    expect(&in, "ARCHIVE"); read_hex(&in, data->archive_sha, 64);
    expect(&in, "COUNT"); data->count = read_u32(&in);
    if (!data->count || data->count > MAX_SAMPLES) fail("dataset count outside bounds");
    for (uint32_t i = 0; i < data->count; i++) {
        sample *s = &data->samples[i];
        nt_spa_experience *experience = &s->experience;
        experience->version = NT_SPA_EXPERIENCE_VERSION;
        expect(&in, "SNAPSHOT");
        s->ordinal = read_u32(&in); s->seed = read_u32(&in);
        s->episode = read_u32(&in); s->step = read_u32(&in);
        experience->sentence_index = read_u32(&in); experience->sentence_count = read_u32(&in);
        experience->source_life_hash = read_hex64(&in); s->body_hash = read_hex64(&in);
        s->agent_rng = read_u32(&in); s->host_rng = read_hex64(&in);
        read_hex(&in, s->snapshot_sha, 64);
        if (s->ordinal != i || !s->seed || s->episode >= 6 || s->step >= 4 || !s->body_hash ||
            !s->agent_rng || experience->sentence_count != 4 ||
            experience->sentence_index != (s->episode + s->step) % 4)
            fail("invalid snapshot coordinates or identity");
        if (i) {
            const sample *previous = &data->samples[i - 1];
            if (s->seed < previous->seed || (s->seed == previous->seed &&
                s->episode * 4 + s->step <= previous->episode * 4 + previous->step))
                fail("snapshot ordering is not canonical");
            if (s->body_hash != data->samples[0].body_hash) fail("mixed body identities");
        }
        expect(&in, "FEATURES");
        for (int j = 0; j < NT_SPA_AGENT_FEATURES; j++) experience->features[j] = read_float(&in);
        expect(&in, "FEATURE_WITNESS"); s->feature_witness = read_hex64(&in);
        check_experience(s, experience);
        uint32_t expected_mask = available_mask(experience);
        for (unsigned hi = 0; hi < 2; hi++) {
            nt_spa_comparison *comparison = &s->comparison[hi];
            comparison->source_life_hash = experience->source_life_hash;
            expect(&in, "HORIZON"); comparison->horizon = read_u32(&in); comparison->action_mask = read_u32(&in);
            if (comparison->horizon != (hi ? 4u : 0u) || comparison->action_mask != expected_mask)
                fail("invalid horizon or missing/extra action");
            for (unsigned kind = 0; kind < NT_SPA_AGENT_ACTIONS; kind++) if (expected_mask & (1u << kind)) {
                nt_spa_alternative *alternative = &comparison->alternatives[kind];
                expect(&in, "OUTCOME");
                uint32_t declared_kind = read_u32(&in);
                if (declared_kind >= NT_SPA_AGENT_ACTIONS) fail("invalid typed action kind");
                alternative->action.kind = (nt_spa_action_kind)declared_kind;
                alternative->action.target = read_u32(&in); alternative->action.source = read_u32(&in);
                nt_spa_action expected_action = action_at(kind, experience);
                if (!same_action(&alternative->action, &expected_action)) fail("outcome action or bounds mismatch");
                read_hex(&in, s->outcome_sha[hi][kind], 64);
                expect(&in, "BEFORE"); alternative->consequence.before = read_metrics(&in);
                expect(&in, "AFTER"); alternative->consequence.after = read_metrics(&in);
                expect(&in, "COST"); alternative->consequence.regeneration_cost = read_float(&in);
                if (alternative->consequence.regeneration_cost < 0 || alternative->consequence.regeneration_cost > 1)
                    fail("normalized cost outside [0,1]");
                const nt_spa_metrics *before = &s->comparison[0].alternatives[0].consequence.before;
                if (!same_metrics(before, &alternative->consequence.before)) fail("outcomes do not share snapshot metrics");
                if (!hi && !kind && (!same_metrics(before, &alternative->consequence.after) ||
                    alternative->consequence.regeneration_cost != 0)) fail("immediate KEEP changed its field");
            }
            expect(&in, "END_HORIZON");
        }
        expect(&in, "END_SNAPSHOT");
    }
    expect(&in, "END");
    char extra[128]; if (token(&in, extra, 0)) fail("trailing dataset data");
    data->file_hash = in.hash;
    if (fclose(in.file)) fail("dataset close failed");
    return data;
}

static void floats(FILE *file, const float *values, unsigned count) {
    fputc('[', file);
    for (unsigned i = 0; i < count; i++) fprintf(file, "%s%.9g", i ? "," : "", values[i]);
    fputc(']', file);
}

static void metrics(FILE *file, const nt_spa_metrics *m) {
    fprintf(file, "{\"local_connectedness\":%.9g,\"global_connectedness\":%.9g,\"coherence\":%.9g,"
        "\"novelty\":%.9g,\"repetition\":%.9g,\"collapse\":%.9g,\"continuity\":%.9g}",
        m->local_connectedness, m->global_connectedness, m->coherence, m->novelty,
        m->repetition, m->collapse, m->continuity);
}

static void action_json(FILE *file, const nt_spa_action *action) {
    fprintf(file, "{\"kind\":%d,\"target\":%u,\"source\":", action->kind, action->target);
    if (action->source == NT_SPA_AGENT_NO_SOURCE) fputs("null}", file);
    else fprintf(file, "%u}", action->source);
}

static void source_json(FILE *file, const sample *s) {
    fprintf(file, "{\"index\":%u,\"seed\":%u,\"episode\":%u,\"step\":%u,\"target\":%u,\"count\":%u,"
        "\"life_hash\":\"%016" PRIx64 "\",\"body_hash\":\"%016" PRIx64 "\",\"agent_rng\":%u,"
        "\"host_rng\":\"%016" PRIx64 "\",\"snapshot_sha256\":\"%s\",\"feature_witness\":\"%016" PRIx64 "\"}",
        s->ordinal, s->seed, s->episode, s->step, s->experience.sentence_index, s->experience.sentence_count,
        s->experience.source_life_hash, s->body_hash, s->agent_rng, s->host_rng, s->snapshot_sha, s->feature_witness);
}

static void outcome_ids(FILE *file, const sample *s, unsigned hi) {
    fputc('[', file);
    for (unsigned kind = 0; kind < NT_SPA_AGENT_ACTIONS; kind++) {
        if (kind) fputc(',', file);
        if (s->comparison[hi].action_mask & (1u << kind)) fprintf(file, "\"%s\"", s->outcome_sha[hi][kind]);
        else fputs("null", file);
    }
    fputc(']', file);
}

static void receipt_json(FILE *file, const nt_spa_comparison_receipt *receipt) {
    fprintf(file, "\"horizon\":%u,\"action_mask\":%u,\"learning_rate\":%.9g,\"rewards\":",
        receipt->horizon, receipt->action_mask, receipt->learning_rate);
    floats(file, receipt->rewards, NT_SPA_AGENT_ACTIONS);
    fputs(",\"targets\":", file); floats(file, receipt->targets, NT_SPA_AGENT_ACTIONS);
    fputs(",\"scores_before\":", file); floats(file, receipt->scores_before, NT_SPA_AGENT_ACTIONS);
    fputs(",\"scores_after\":", file); floats(file, receipt->scores_after, NT_SPA_AGENT_ACTIONS);
    fprintf(file, ",\"loss_before\":%.17g,\"loss_after\":%.17g", receipt->loss_before, receipt->loss_after);
}

static void path_for(char out[PATH_BYTES], const char *prefix, const char *suffix) {
    int length = snprintf(out, PATH_BYTES, "%s.%s", prefix, suffix);
    if (length < 0 || (unsigned)length >= PATH_BYTES) fail("output path too long");
}

static FILE *open_new(const char *path) {
    FILE *file = fopen(path, "wx");
    if (!file) fail("cannot create new output; destination must not exist");
    return file;
}

static void close_output(FILE *file) {
    int failed = ferror(file);
    if (fflush(file)) failed = 1;
    if (fclose(file)) failed = 1;
    if (failed) fail("output flush/close failed");
}

static uint64_t life_hash(const nt_spa_agent *life) {
    uint64_t hash = nt_spa_agent_hash(life);
    if (!hash) fail("invalid canonical life");
    return hash;
}

static void only_policy_changed(const nt_spa_agent *before, const nt_spa_agent *after) {
    nt_spa_agent restored;
    memcpy(&restored, after, sizeof(restored));
    memcpy(&restored.policy, &before->policy, sizeof(restored.policy));
    if (memcmp(&restored, before, sizeof(restored))) fail("comparison changed non-policy state");
}

static uint64_t save_life(const nt_spa_agent *life, const char *prefix, unsigned model,
                          const dataset *data, FILE *trace) {
    char suffix[64], path[PATH_BYTES];
    int n = snprintf(suffix, sizeof(suffix), "%s.life.bin", model_names[model]);
    if (n < 0 || (size_t)n >= sizeof(suffix)) fail("invalid model path");
    path_for(path, prefix, suffix);
    close_output(open_new(path)); // Reserve this new path; save replaces only our placeholder.
    uint64_t hash = life_hash(life);
    check(nt_spa_agent_save(life, path), "life save failed");
    nt_spa_agent loaded;
    check(nt_spa_agent_load(&loaded, path), "saved life load failed");
    if (life_hash(&loaded) != hash) fail("save/resume canonical hash mismatch");
    for (uint32_t i = 0; i < data->count; i++) {
        nt_spa_readout before, after;
        check(nt_spa_agent_score_experience(life, &data->samples[i].experience, &before), "pre-save readout failed");
        check(nt_spa_agent_score_experience(&loaded, &data->samples[i].experience, &after), "resumed readout failed");
        if (!same_action(&before.action, &after.action) || before.action_mask != after.action_mask ||
            memcmp(before.scores, after.scores, sizeof(before.scores))) fail("save/resume readout changed");
    }
    if (life_hash(life) != hash || life_hash(&loaded) != hash) fail("save/resume mutated a life");
    fprintf(trace, "{\"type\":\"saved_life\",\"model\":\"%s\",\"hash\":\"%016" PRIx64
        "\",\"resume_readouts\":%u,\"same_hash\":true,\"same_actions_scores\":true}\n",
        model_names[model], hash, data->count);
    return hash;
}

static unsigned donor_index(const dataset *data, unsigned receiver) {
    const nt_spa_experience *source = &data->samples[receiver].experience;
    for (unsigned offset = 1; offset < data->count; offset++) {
        unsigned candidate = (receiver + offset) % data->count;
        const nt_spa_experience *other = &data->samples[candidate].experience;
        if (source->sentence_count == other->sentence_count && source->sentence_index == other->sentence_index)
            return candidate;
    }
    fail("shuffle has no distinct coordinate-compatible donor");
    return 0;
}

static void validate_training_order(const dataset *data) {
    if (data->count != TRAIN_SAMPLES) fail("training requires all 48 registered snapshots");
    for (unsigned i = 0; i < TRAIN_SAMPLES; i++) {
        const sample *s = &data->samples[i];
        if (s->seed != (i < 24 ? 42u : 73u) || s->episode != (i % 24) / 4 || s->step != i % 4)
            fail("training sample order differs from frozen protocol");
    }
}

static void write_import(FILE *trace, const dataset *data) {
    fprintf(trace, "{\"type\":\"dataset\",\"protocol_sha256\":\"%s\",\"archive_sha256\":\"%s\","
        "\"dataset_fnv1a\":\"%016" PRIx64 "\",\"samples\":%u}\n",
        data->protocol_sha, data->archive_sha, data->file_hash, data->count);
    for (unsigned i = 0; i < data->count; i++) {
        const sample *s = &data->samples[i];
        fputs("{\"type\":\"import\",\"source\":", trace); source_json(trace, s);
        fputs(",\"features\":", trace); floats(trace, s->experience.features, NT_SPA_AGENT_FEATURES);
        fputs(",\"h0_outcome_sha256\":", trace); outcome_ids(trace, s, 0);
        fputs(",\"h4_outcome_sha256\":", trace); outcome_ids(trace, s, 1);
        fputs(",\"feature_witness_valid\":true}\n", trace);
    }
}

static void write_seal(const char *prefix, const seal *saved) {
    char path[PATH_BYTES]; path_for(path, prefix, "seal");
    FILE *file = open_new(path);
    fprintf(file, "NT_SPA_FUTURE_SEAL_V1\nPROTOCOL %s\nARCHIVE %s\nDATASET %016" PRIx64 "\n",
        saved->protocol_sha, saved->archive_sha, saved->dataset_hash);
    for (unsigned i = 0; i < 4; i++) fprintf(file, "LIFE %s %016" PRIx64 "\n", model_names[i], saved->life_hash[i]);
    fputs("END\n", file);
    close_output(file);
}

static seal read_seal(const char *prefix) {
    char path[PATH_BYTES]; path_for(path, prefix, "seal");
    reader in = {fopen(path, "rb"), UINT64_C(14695981039346656037)};
    if (!in.file) fail("cannot open completed fit seal");
    seal saved = {0};
    expect(&in, "NT_SPA_FUTURE_SEAL_V1");
    expect(&in, "PROTOCOL"); read_hex(&in, saved.protocol_sha, 64);
    expect(&in, "ARCHIVE"); read_hex(&in, saved.archive_sha, 64);
    expect(&in, "DATASET"); saved.dataset_hash = read_hex64(&in);
    for (unsigned i = 0; i < 4; i++) {
        expect(&in, "LIFE"); expect(&in, model_names[i]); saved.life_hash[i] = read_hex64(&in);
        if (!saved.life_hash[i]) fail("invalid sealed life hash");
    }
    expect(&in, "END");
    char extra[128]; if (token(&in, extra, 0)) fail("trailing fit seal data");
    if (fclose(in.file)) fail("fit seal close failed");
    return saved;
}

static int train_main(int argc, char **argv) {
    if (argc != 4) fail("usage: spa_agent_future train DATASET OUTPUT_PREFIX");
    dataset *data = read_dataset(argv[2]);
    validate_training_order(data);
    char path[PATH_BYTES]; path_for(path, argv[3], "fit.jsonl");
    FILE *trace = open_new(path);
    write_import(trace, data);
    nt_spa_agent_config config;
    nt_spa_agent_config_default(&config);
    config.mode = NT_SPA_AGENT_LEARNED; config.seed = 1;
    config.learning_rate = FIT_RATE; config.exploration = 0;
    nt_spa_agent initial;
    check(nt_spa_agent_init(&initial, &config), "initial life refused");
    seal saved = {0};
    memcpy(saved.protocol_sha, data->protocol_sha, HASH_BYTES);
    memcpy(saved.archive_sha, data->archive_sha, HASH_BYTES);
    saved.dataset_hash = data->file_hash;
    fprintf(trace, "{\"type\":\"fit_run\",\"samples\":%u,\"epochs\":%u,\"learning_rate\":%.9g,"
        "\"seed\":1,\"exploration\":0,\"parameters\":%d,\"initial_hash\":\"%016" PRIx64 "\"}\n",
        data->count, EPOCHS, FIT_RATE, NT_SPA_AGENT_PARAMETERS, life_hash(&initial));
    saved.life_hash[0] = save_life(&initial, argv[3], 0, data, trace);
    uint64_t total = 0;
    for (unsigned arm = 0; arm < 3; arm++) {
        nt_spa_agent life; memcpy(&life, &initial, sizeof(life));
        uint64_t arm_steps = 0;
        for (unsigned epoch = 0; epoch < EPOCHS; epoch++) for (unsigned i = 0; i < data->count; i++) {
            const sample *source = &data->samples[i];
            unsigned donor = arm == 2 ? donor_index(data, i) : i;
            const sample *outcome_source = &data->samples[donor];
            // SPA_FUTURE_MUTATE_HORIZON: the arm selects its registered outcome horizon here.
            unsigned horizon_index = arm == 1 ? 0u : 1u;
            nt_spa_experience experience = source->experience;
            // SPA_FUTURE_MUTATE_SOURCE_FEATURES: provenance must follow the features actually used.
            check_experience(source, &experience);
            nt_spa_comparison comparison = outcome_source->comparison[horizon_index];
            // The shuffled control deliberately binds donor outcomes to receiver features.
            // Original source identity is retained in every receipt below.
            comparison.source_life_hash = experience.source_life_hash;
            // SPA_FUTURE_MUTATE_ACTION_TARGETS: independent gates join each typed outcome to its source.
            nt_spa_comparison comparison_before; memcpy(&comparison_before, &comparison, sizeof(comparison_before));
            nt_spa_agent before; memcpy(&before, &life, sizeof(before));
            uint64_t hash_before = life_hash(&life);
            nt_spa_comparison_receipt receipt;
            check(nt_spa_agent_fit_comparison(&life, &experience, &comparison, FIT_RATE, &receipt), "comparison fit refused");
            only_policy_changed(&before, &life);
            if (memcmp(&experience, &source->experience, sizeof(experience))) fail("fit mutated captured experience");
            if (memcmp(&comparison, &comparison_before, sizeof(comparison))) fail("fit mutated its outcome input");
            ++arm_steps; ++total;
            fprintf(trace, "{\"type\":\"fit\",\"arm\":\"%s\",\"fit_step\":%" PRIu64
                ",\"total_step\":%" PRIu64 ",\"epoch\":%u,\"sample_index\":%u,\"donor_index\":%u,\"feature_source\":",
                arm_names[arm], arm_steps, total, epoch + 1, i, donor);
            source_json(trace, source); fputs(",\"outcome_source\":", trace); source_json(trace, outcome_source);
            fprintf(trace, ",\"comparison_feature_hash\":\"%016" PRIx64 "\",\"hash_before\":\"%016" PRIx64
                "\",\"hash_after\":\"%016" PRIx64 "\",\"outcome_sha256\":",
                receipt.source_life_hash, hash_before, life_hash(&life));
            outcome_ids(trace, outcome_source, horizon_index); fputc(',', trace); receipt_json(trace, &receipt);
            fputs(",\"non_policy_state_unchanged\":true,\"feature_witness_valid\":true}\n", trace);
            if (i + 1 == data->count && (ferror(trace) || fflush(trace))) fail("fit trace flush failed");
        }
        only_policy_changed(&initial, &life);
        saved.life_hash[arm + 1] = save_life(&life, argv[3], arm + 1, data, trace);
        fprintf(trace, "{\"type\":\"arm_summary\",\"arm\":\"%s\",\"fit_steps\":%" PRIu64
            ",\"hash\":\"%016" PRIx64 "\",\"non_policy_state_unchanged\":true}\n",
            arm_names[arm], arm_steps, saved.life_hash[arm + 1]);
    }
    fprintf(trace, "{\"type\":\"fit_summary\",\"fit_steps\":%" PRIu64 ",\"lives\":[", total);
    for (unsigned i = 0; i < 4; i++) fprintf(trace, "%s{\"model\":\"%s\",\"hash\":\"%016" PRIx64 "\"}",
        i ? "," : "", model_names[i], saved.life_hash[i]);
    fputs("],\"all_lives_saved_resumed\":true}\n", trace);
    close_output(trace);
    write_seal(argv[3], &saved);
    printf("{\"type\":\"fit_complete\",\"fit_steps\":%" PRIu64 ",\"lives_sealed\":4}\n", total);
    free(data);
    return 0;
}

static int read_main(int argc, char **argv) {
    if (argc != 5) fail("usage: spa_agent_future read DATASET FIT_PREFIX TRACE");
    dataset *data = read_dataset(argv[2]);
    seal saved = read_seal(argv[3]);
    if (strcmp(data->protocol_sha, saved.protocol_sha)) fail("readout protocol differs from sealed fit");
    nt_spa_agent lives[4];
    for (unsigned model = 0; model < 4; model++) {
        char suffix[64], path[PATH_BYTES];
        int n = snprintf(suffix, sizeof(suffix), "%s.life.bin", model_names[model]);
        if (n < 0 || (size_t)n >= sizeof(suffix)) fail("invalid life suffix");
        path_for(path, argv[3], suffix);
        check(nt_spa_agent_load(&lives[model], path), "sealed life load failed");
        if (life_hash(&lives[model]) != saved.life_hash[model]) fail("saved life differs from fit seal");
        if (model) only_policy_changed(&lives[0], &lives[model]);
    }
    FILE *trace = open_new(argv[4]);
    write_import(trace, data);
    fprintf(trace, "{\"type\":\"read_run\",\"fit_dataset_fnv1a\":\"%016" PRIx64
        "\",\"fit_archive_sha256\":\"%s\",\"models\":7,\"horizons\":[0,4],\"learning_rate\":0}\n",
        saved.dataset_hash, saved.archive_sha);
    static const char *const labels[7] = {"initial", "h4", "h0", "shuffled_h4", "keep", "left", "right"};
    uint64_t rows = 0;
    for (unsigned i = 0; i < data->count; i++) for (unsigned label = 0; label < 7; label++) {
        const sample *source = &data->samples[i];
        unsigned model = label < 4 ? label : 0;
        nt_spa_agent *life = &lives[model];
        nt_spa_agent before; memcpy(&before, life, sizeof(before));
        nt_spa_experience experience = source->experience;
        check_experience(source, &experience);
        nt_spa_readout readout;
        check(nt_spa_agent_score_experience(life, &experience, &readout), "captured readout refused");
        nt_spa_action selected = readout.action;
        unsigned requested_kind = label >= 4 ? label - 4 : (unsigned)selected.kind;
        int fallback = label >= 4 && !(readout.action_mask & (1u << requested_kind));
        if (label >= 4) selected = action_at(fallback ? NT_SPA_KEEP : requested_kind, &experience);
        if (!(readout.action_mask & (1u << selected.kind))) fail("readout selected unavailable action");
        for (unsigned hi = 0; hi < 2; hi++) {
            const nt_spa_comparison *comparison = &source->comparison[hi];
            nt_spa_comparison comparison_before; memcpy(&comparison_before, comparison, sizeof(comparison_before));
            nt_spa_comparison_receipt receipt;
            check(nt_spa_agent_fit_comparison(life, &experience, comparison, 0, &receipt), "zero-rate outcome readout refused");
            if (memcmp(life, &before, sizeof(before)) || life_hash(life) != saved.life_hash[model] ||
                memcmp(&experience, &source->experience, sizeof(experience))) fail("readout changed policy, history or input");
            if (memcmp(comparison, &comparison_before, sizeof(comparison_before))) fail("readout changed source outcomes");
            if (readout.action_mask != receipt.action_mask || memcmp(readout.scores, receipt.scores_before, sizeof(readout.scores)) ||
                memcmp(readout.scores, receipt.scores_after, sizeof(readout.scores))) fail("score and comparison readout differ");
            const nt_spa_alternative *outcome = &comparison->alternatives[selected.kind];
            if (!same_action(&outcome->action, &selected)) fail("selected action is not its executed branch");
            float best = receipt.rewards[NT_SPA_KEEP];
            for (unsigned kind = 1; kind < NT_SPA_AGENT_ACTIONS; kind++)
                if ((receipt.action_mask & (1u << kind)) && receipt.rewards[kind] > best) best = receipt.rewards[kind];
            double regret = (double)best - receipt.rewards[selected.kind];
            fprintf(trace, "{\"type\":\"readout\",\"label\":\"%s\",\"source\":", labels[label]); source_json(trace, source);
            fprintf(trace, ",\"model_hash\":\"%016" PRIx64 "\",\"forced\":%s,\"requested_kind\":%u,\"boundary_fallback\":%s,"
                "\"action\":", saved.life_hash[model], label >= 4 ? "true" : "false", requested_kind, fallback ? "true" : "false");
            action_json(trace, &selected);
            fprintf(trace, ",\"selected_outcome_sha256\":\"%s\",\"outcome_sha256\":", source->outcome_sha[hi][selected.kind]);
            outcome_ids(trace, source, hi); fputs(",\"before\":", trace); metrics(trace, &outcome->consequence.before);
            fputs(",\"after\":", trace); metrics(trace, &outcome->consequence.after);
            fprintf(trace, ",\"cost\":%.9g,", outcome->consequence.regeneration_cost); receipt_json(trace, &receipt);
            fprintf(trace, ",\"selected_reward\":%.9g,\"keep_reward\":%.9g,\"advantage_over_keep\":%.17g,\"regret\":%.17g,"
                "\"optimal\":%s,\"life_unchanged\":true,\"features_unchanged\":true,\"feature_witness_valid\":true}\n",
                receipt.rewards[selected.kind], receipt.rewards[0], (double)receipt.rewards[selected.kind] - receipt.rewards[0],
                regret, regret <= 1e-7 ? "true" : "false");
            rows++;
        }
    }
    for (unsigned model = 0; model < 4; model++) if (life_hash(&lives[model]) != saved.life_hash[model]) fail("readout changed final life hash");
    fprintf(trace, "{\"type\":\"read_summary\",\"samples\":%u,\"readouts\":%" PRIu64
        ",\"sealed_lives_unchanged\":true,\"source_features_unchanged\":true}\n", data->count, rows);
    close_output(trace);
    printf("{\"type\":\"read_complete\",\"readouts\":%" PRIu64 ",\"sealed_lives_unchanged\":true}\n", rows);
    free(data);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) fail("expected train or read command");
    if (!strcmp(argv[1], "train")) return train_main(argc, argv);
    if (!strcmp(argv[1], "read")) return read_main(argc, argv);
    fail("unknown command");
    return 1;
}
