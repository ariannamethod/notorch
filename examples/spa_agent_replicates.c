/* Paired sentence-action RNG replicates from fixed native sentence fields.
 * Reuse the registered body, generation, perception, execution and measurement
 * writer directly. The reference executable remains an unchanged source file.
 */
#define main spa_agent_reference_demo_main
#include "spa_agent_demo.c"
#undef main
#include <ctype.h>

#define REP_COUNT 8u
#define REP_MAX_SOURCES 96u
#define REP_HASH_BYTES 65u

typedef struct {
    uint32_t ordinal, seed, episode, step, agent_rng;
    uint64_t body_identity, host_rng, feature_witness, snapshot_witness;
    char archive_sha[REP_HASH_BYTES], snapshot_sha[REP_HASH_BYTES];
    nt_spa_experience experience;
    nt_spa_metrics before;
    unsigned reseeds[SENTENCES];
    sentence chain[SENTENCES];
} rep_source;

typedef struct {
    char protocol_sha[REP_HASH_BYTES];
    uint64_t vocabulary_hash, file_hash;
    uint32_t count;
    rep_source sources[REP_MAX_SOURCES];
} rep_dataset;

typedef struct {
    FILE *file;
    uint64_t hash;
} rep_reader;

static uint64_t rep_hash_integer(uint64_t hash, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) {
        unsigned char byte = (unsigned char)((value >> (8 * i)) & 255u);
        hash = hash_bytes(hash, &byte, 1);
    }
    return hash;
}

static uint64_t rep_hash_float(uint64_t hash, float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    return rep_hash_integer(hash, bits, 4);
}

static uint64_t rep_feature_witness(const nt_spa_experience *experience) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = rep_hash_integer(hash, experience->source_life_hash, 8);
    hash = rep_hash_integer(hash, experience->sentence_index, 4);
    hash = rep_hash_integer(hash, experience->sentence_count, 4);
    for (int i = 0; i < NT_SPA_AGENT_FEATURES; i++) hash = rep_hash_float(hash, experience->features[i]);
    return hash;
}

/* Canonical little-endian source witness, independent of C structure padding.
 * u32(seed,episode,step,target,count), u64(life,body,host_rng), u32(agent_rng),
 * 29 f32 features, seven f32 metrics, four u32 reseeds, then each sentence's
 * u32(length,terminated,tokens[length]). The Python exporter names this layout.
 */
static uint64_t rep_snapshot_witness(const rep_source *source) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const nt_spa_experience *experience = &source->experience;
    const uint32_t coordinates[] = {source->seed, source->episode, source->step,
        experience->sentence_index, experience->sentence_count};
    for (unsigned i = 0; i < 5; i++) hash = rep_hash_integer(hash, coordinates[i], 4);
    hash = rep_hash_integer(hash, experience->source_life_hash, 8);
    hash = rep_hash_integer(hash, source->body_identity, 8);
    hash = rep_hash_integer(hash, source->host_rng, 8);
    hash = rep_hash_integer(hash, source->agent_rng, 4);
    for (int i = 0; i < NT_SPA_AGENT_FEATURES; i++) hash = rep_hash_float(hash, experience->features[i]);
    const nt_spa_metrics *m = &source->before;
    const float axes[] = {m->local_connectedness, m->global_connectedness, m->coherence,
        m->novelty, m->repetition, m->collapse, m->continuity};
    for (unsigned i = 0; i < 7; i++) hash = rep_hash_float(hash, axes[i]);
    for (unsigned i = 0; i < SENTENCES; i++) hash = rep_hash_integer(hash, source->reseeds[i], 4);
    for (unsigned i = 0; i < SENTENCES; i++) {
        const sentence *s = &source->chain[i];
        hash = rep_hash_integer(hash, (uint32_t)s->length, 4);
        hash = rep_hash_integer(hash, (uint32_t)s->stopped, 4);
        for (int j = 0; j < s->length; j++) hash = rep_hash_integer(hash, (uint32_t)s->ids[j], 4);
    }
    return hash;
}

static uint64_t rep_vocabulary_hash(const sentence_body *body) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < VOCAB; i++) {
        uint32_t cp = body->vocabulary[i];
        if (cp > UINT32_C(0x10ffff) || (cp >= UINT32_C(0xd800) && cp <= UINT32_C(0xdfff)) ||
            (i && cp <= body->vocabulary[i - 1])) fail("replicate vocabulary must contain sorted unique Unicode scalars");
        hash = rep_hash_integer(hash, cp, 4);
    }
    return hash;
}

static int rep_byte(rep_reader *in) {
    int value = fgetc(in->file);
    if (value != EOF) {
        unsigned char byte = (unsigned char)value;
        in->hash = hash_bytes(in->hash, &byte, 1);
    }
    return value;
}

static int rep_token(rep_reader *in, char out[128], int required) {
    int value;
    do { value = rep_byte(in); } while (value != EOF && isspace((unsigned char)value));
    if (value == EOF) {
        if (ferror(in->file)) fail("replicate source read failed");
        if (required) fail("truncated replicate source");
        return 0;
    }
    unsigned n = 0;
    do {
        if (value < 33 || value > 126 || n == 127) fail("invalid or overlong replicate source token");
        out[n++] = (char)value;
        value = rep_byte(in);
    } while (value != EOF && !isspace((unsigned char)value));
    out[n] = 0;
    return 1;
}

static void rep_expect(rep_reader *in, const char *label) {
    char token[128]; rep_token(in, token, 1);
    if (strcmp(token, label)) fail("unexpected replicate source label");
}

static uint32_t rep_u32(rep_reader *in) {
    char token[128], *end; rep_token(in, token, 1);
    for (const char *p = token; *p; p++) if (*p < '0' || *p > '9') fail("invalid replicate unsigned integer");
    errno = 0;
    unsigned long long value = strtoull(token, &end, 10);
    if (errno || *end || value > UINT32_MAX) fail("replicate integer outside uint32");
    return (uint32_t)value;
}

static void rep_hex(rep_reader *in, char *out, unsigned digits) {
    char token[128]; rep_token(in, token, 1);
    if (strlen(token) != digits) fail("invalid replicate identity length");
    for (unsigned i = 0; i < digits; i++)
        if (!((token[i] >= '0' && token[i] <= '9') || (token[i] >= 'a' && token[i] <= 'f')))
            fail("replicate identity requires lowercase hexadecimal");
    memcpy(out, token, digits + 1);
}

static uint64_t rep_hex64(rep_reader *in) {
    char token[17]; rep_hex(in, token, 16);
    uint64_t result = 0;
    for (unsigned i = 0; i < 16; i++)
        result = (result << 4) | (uint64_t)(token[i] <= '9' ? token[i] - '0' : token[i] - 'a' + 10);
    return result;
}

static float rep_float(rep_reader *in) {
    char token[128], *end; rep_token(in, token, 1);
    errno = 0;
    float value = strtof(token, &end);
    if ((errno && !(errno == ERANGE && isfinite(value) && value != 0)) ||
        end == token || *end || !isfinite(value)) fail("invalid replicate float");
    return value;
}

static nt_spa_metrics rep_metrics(rep_reader *in) {
    nt_spa_metrics result;
    float *axes[] = {&result.local_connectedness, &result.global_connectedness, &result.coherence,
        &result.novelty, &result.repetition, &result.collapse, &result.continuity};
    for (unsigned i = 0; i < 7; i++) {
        *axes[i] = rep_float(in);
        if (*axes[i] < 0 || *axes[i] > 1) fail("replicate metric outside [0,1]");
    }
    return result;
}

static int rep_stop(uint32_t codepoint) {
    return codepoint == '.' || codepoint == '!' || codepoint == '?';
}

static void rep_validate_sentence(const sentence_body *body, const sentence *s) {
    if (s->length < MIN_GENERATED || s->length > MAX_GENERATED || (s->stopped != 0 && s->stopped != 1))
        fail("replicate source sentence length or termination invalid");
    for (int i = 0; i < s->length; i++) {
        if (s->ids[i] < 0 || s->ids[i] >= VOCAB) fail("replicate source token outside vocabulary");
        if (i >= MIN_GENERATED - 1 && i + 1 < s->length && rep_stop(body->vocabulary[s->ids[i]]))
            fail("replicate sentence passed an eligible stop character");
    }
    if (rep_stop(body->vocabulary[s->ids[s->length - 1]]) != s->stopped ||
        (!s->stopped && s->length != MAX_GENERATED)) fail("replicate termination does not match its tokens");
}

static void rep_validate_source(const rep_source *source, sentence_body *body,
                                uint64_t expected_body, const nt_spa_agent *witness_life) {
    const nt_spa_experience *experience = &source->experience;
    check(nt_spa_experience_validate(experience), "replicate experience refused");
    if (source->body_identity != expected_body || !source->agent_rng || !source->seed ||
        source->episode >= EPISODES || source->step >= SENTENCES || experience->sentence_count != SENTENCES ||
        experience->sentence_index != (source->episode + source->step) % SENTENCES ||
        source->ordinal % (EPISODES * SENTENCES) != source->episode * SENTENCES + source->step)
        fail("replicate source coordinates or body identity mismatch");
    if (source->host_rng != stream(source->seed, UINT64_C(0x7370615f6163746e), source->episode, source->step))
        fail("replicate source host RNG differs from original stream");
    for (unsigned i = 0; i < SENTENCES; i++) {
        if (source->reseeds[i] > EPISODES * SENTENCES) fail("replicate source reseed count outside bounds");
        rep_validate_sentence(body, &source->chain[i]);
    }
    if (rep_feature_witness(experience) != source->feature_witness ||
        rep_snapshot_witness(source) != source->snapshot_witness) fail("replicate source witness mismatch");
    nt_spa_observation observed;
    nt_spa_metrics before;
    observation(body, source->chain, experience->sentence_index,
                source->reseeds[experience->sentence_index], &observed, &before);
    if (!scenario_metrics_equal(&before, &source->before)) fail("replicate source metrics differ from its sentence field");
    nt_spa_experience captured;
    check(nt_spa_agent_capture_experience(witness_life, &observed, &captured), "replicate sensory witness refused");
    // These nineteen inputs depend on the sentence observation only. The other
    // ten are the imported source's history; this empty witness is not that life.
    if (memcmp(captured.features, experience->features, 19 * sizeof(float)))
        fail("replicate sensory features differ from its sentence field");
    scenario_check_tape();
}

static rep_dataset *rep_read_dataset(const char *path, sentence_body *body) {
    rep_dataset *data = calloc(1, sizeof(*data));
    if (!data) fail("replicate source allocation failed");
    rep_reader in = {fopen(path, "rb"), UINT64_C(14695981039346656037)};
    if (!in.file) fail("cannot open replicate source");
    rep_expect(&in, "NT_SPA_REPLICATES_V1");
    rep_expect(&in, "PROTOCOL"); rep_hex(&in, data->protocol_sha, 64);
    rep_expect(&in, "VOCAB_FNV1A"); data->vocabulary_hash = rep_hex64(&in);
    if (data->vocabulary_hash != rep_vocabulary_hash(body)) fail("replicate vocabulary witness mismatch");
    rep_expect(&in, "COUNT"); data->count = rep_u32(&in);
    if (!data->count || data->count > REP_MAX_SOURCES) fail("replicate source count outside bounds");
    nt_spa_agent_config config; nt_spa_agent_config_default(&config);
    config.mode = NT_SPA_AGENT_LEARNED;
    nt_spa_agent witness_life;
    check(nt_spa_agent_init(&witness_life, &config), "replicate sensory witness initialization failed");
    const uint64_t expected_body = body_hash(body), witness_hash = nt_spa_agent_hash(&witness_life);
    const uint64_t forwards = body->forwards;
    int mode = nt_is_training();
    for (unsigned i = 0; i < data->count; i++) {
        rep_source *source = &data->sources[i];
        source->experience.version = NT_SPA_EXPERIENCE_VERSION;
        rep_expect(&in, "SNAPSHOT");
        source->ordinal = rep_u32(&in); source->seed = rep_u32(&in);
        source->episode = rep_u32(&in); source->step = rep_u32(&in);
        source->experience.sentence_index = rep_u32(&in); source->experience.sentence_count = rep_u32(&in);
        source->experience.source_life_hash = rep_hex64(&in); source->body_identity = rep_hex64(&in);
        source->agent_rng = rep_u32(&in); source->host_rng = rep_hex64(&in);
        rep_hex(&in, source->archive_sha, 64); rep_hex(&in, source->snapshot_sha, 64);
        if (source->ordinal >= REP_MAX_SOURCES || (i && source->ordinal <= data->sources[i - 1].ordinal))
            fail("replicate source ordinals not strictly ordered");
        if (i) {
            const rep_source *previous = &data->sources[i - 1];
            if (source->seed < previous->seed || (source->seed == previous->seed &&
                source->episode * SENTENCES + source->step <= previous->episode * SENTENCES + previous->step))
                fail("replicate source coordinates not canonically ordered");
        }
        rep_expect(&in, "FEATURES");
        for (int j = 0; j < NT_SPA_AGENT_FEATURES; j++) source->experience.features[j] = rep_float(&in);
        rep_expect(&in, "BEFORE"); source->before = rep_metrics(&in);
        rep_expect(&in, "RESEEDS");
        for (unsigned j = 0; j < SENTENCES; j++) source->reseeds[j] = rep_u32(&in);
        for (unsigned j = 0; j < SENTENCES; j++) {
            rep_expect(&in, "SENTENCE");
            if (rep_u32(&in) != j) fail("replicate source sentence ordering mismatch");
            uint32_t length = rep_u32(&in), stopped = rep_u32(&in);
            if (length < MIN_GENERATED || length > MAX_GENERATED || stopped > 1)
                fail("replicate source sentence dimensions invalid");
            source->chain[j].length = (int)length; source->chain[j].stopped = (int)stopped;
            for (unsigned k = 0; k < length; k++) {
                uint32_t token = rep_u32(&in);
                if (token >= VOCAB) fail("replicate source token outside vocabulary");
                source->chain[j].ids[k] = (int)token;
            }
        }
        rep_expect(&in, "FEATURE_WITNESS"); source->feature_witness = rep_hex64(&in);
        rep_expect(&in, "SNAPSHOT_WITNESS"); source->snapshot_witness = rep_hex64(&in);
        rep_expect(&in, "END_SNAPSHOT");
        rep_validate_source(source, body, expected_body, &witness_life);
    }
    rep_expect(&in, "END");
    char extra[128]; if (rep_token(&in, extra, 0)) fail("trailing replicate source data");
    data->file_hash = in.hash;
    if (fclose(in.file)) fail("replicate source close failed");
    if (body->forwards != forwards || body_hash(body) != expected_body || nt_is_training() != mode ||
        nt_spa_agent_hash(&witness_life) != witness_hash) fail("replicate import changed its witnesses");
    return data;
}

static uint64_t rep_rng(const rep_source *source, unsigned replicate, unsigned hop) {
    if (replicate >= REP_COUNT || hop > 4) fail("invalid replicate RNG coordinates");
    if (!replicate) {
        if (!hop) return stream(source->seed, UINT64_C(0x7370615f6163746e), source->episode, source->step);
        return stream(source->seed, UINT64_C(0x7370615f66757472), source->episode * SENTENCES + source->step, hop);
    }
    unsigned packed = (source->episode * SENTENCES + source->step) * REP_COUNT + replicate;
    return stream(source->seed, hop ? UINT64_C(0x7370615f72667574) : UINT64_C(0x7370615f72696e69), packed, hop);
}

static void rep_context(FILE *file, const rep_source *source, unsigned replicate) {
    fprintf(file, "\"source_index\":%u,\"seed\":%u,\"episode\":%u,\"step\":%u,\"snapshot\":%u,\"replicate\":%u",
        source->ordinal, source->seed, source->episode, source->step, source->episode * SENTENCES + source->step, replicate);
}

static void rep_write_source(FILE *file, const rep_source *source) {
    fprintf(file, "{\"type\":\"source\",\"source_index\":%u,\"seed\":%u,\"episode\":%u,\"step\":%u,"
        "\"snapshot\":%u,\"target\":%u,\"sentence_count\":%u,\"life_hash\":\"%016" PRIx64
        "\",\"body_hash\":\"%016" PRIx64 "\",\"agent_rng\":%u,\"host_rng_before\":\"%016" PRIx64
        "\",\"source_archive_sha256\":\"%s\",\"snapshot_sha256\":\"%s\",\"feature_witness\":\"%016" PRIx64
        "\",\"snapshot_witness\":\"%016" PRIx64 "\",\"features\":",
        source->ordinal, source->seed, source->episode, source->step, source->episode * SENTENCES + source->step,
        source->experience.sentence_index, source->experience.sentence_count, source->experience.source_life_hash,
        source->body_identity, source->agent_rng, source->host_rng, source->archive_sha, source->snapshot_sha,
        source->feature_witness, source->snapshot_witness);
    write_floats(file, source->experience.features, NT_SPA_AGENT_FEATURES);
    fputs(",\"before\":", file); write_metrics(file, &source->before);
    fputs(",\"chain\":", file); write_chain(file, source->chain);
    fputs(",\"reseeds\":", file); scenario_write_reseeds(file, source->reseeds);
    fputs(",\"sensory_features_checked\":19,\"metrics_match_field\":true}\n", file);
}

static void rep_run(FILE *file, sentence_body *body, const rep_dataset *data) {
    uint64_t original_body = body_hash(body), original_forwards = body->forwards;
    int original_mode = nt_is_training();
    scenario_check_tape();
    scenario_sink sink = {0}; sink.file = file;
    uint64_t total_forwards = 0, alternatives = 0, replicates = 0;
    fprintf(file, "{\"type\":\"replicate_run\",\"protocol_sha256\":\"%s\",\"dataset_fnv1a\":\"%016" PRIx64
        "\",\"vocabulary_fnv1a\":\"%016" PRIx64 "\",\"body_hash\":\"%016" PRIx64
        "\",\"parameters\":%ld,\"sources\":%u,\"replicates_per_source\":%u,\"horizons\":[0,1,4]}\n",
        data->protocol_sha, data->file_hash, data->vocabulary_hash, original_body, body->count, data->count, REP_COUNT);
    for (unsigned si = 0; si < data->count; si++) {
        const rep_source *source = &data->sources[si];
        rep_source saved; memcpy(&saved, source, sizeof(saved));
        const unsigned target = source->experience.sentence_index;
        sink.seed = source->seed;
        rep_write_source(file, source);
        nt_spa_observation bounds = {0};
        bounds.sentence_count = SENTENCES; bounds.sentence_index = target; bounds.temperature = 0.8f;
        for (unsigned replicate = 0; replicate < REP_COUNT; replicate++) {
            uint64_t starts[5];
            for (unsigned hop = 0; hop <= 4; hop++) starts[hop] = rep_rng(source, replicate, hop);
            fprintf(file, "{\"type\":\"replicate_begin\","); rep_context(file, source, replicate);
            fprintf(file, ",\"feature_witness\":\"%016" PRIx64 "\",\"snapshot_witness\":\"%016" PRIx64
                "\",\"life_hash\":\"%016" PRIx64 "\",\"agent_rng\":%u,\"rng_starts\":[",
                source->feature_witness, source->snapshot_witness, source->experience.source_life_hash, source->agent_rng);
            for (unsigned hop = 0; hop <= 4; hop++) fprintf(file, "%s\"%016" PRIx64 "\"", hop ? "," : "", starts[hop]);
            fputs("]}\n", file);
            uint64_t replicate_forwards = 0, measurements_before = sink.measurements;
            unsigned valid = 0;
            for (unsigned kind = NT_SPA_KEEP; kind <= NT_SPA_RESEED_RIGHT; kind++) {
                nt_spa_action action = {(nt_spa_action_kind)kind, target, NT_SPA_AGENT_NO_SOURCE};
                if (kind == NT_SPA_RESEED_LEFT) action.source = target ? target - 1 : NT_SPA_AGENT_NO_SOURCE;
                if (kind == NT_SPA_RESEED_RIGHT) action.source = target + 1;
                if (nt_spa_action_validate(&action, &bounds) != NT_SPA_OK) continue;
                sentence branch[SENTENCES]; memcpy(branch, source->chain, sizeof(branch));
                unsigned counts[SENTENCES]; memcpy(counts, source->reseeds, sizeof(counts));
                uint64_t initial_rng = starts[0];
                // SPA_REPLICATES_MUTATE_PAIRING: every action receives the same precomputed start.
                if (initial_rng != rep_rng(source, replicate, 0)) fail("replicate initial RNG pairing mismatch");
                scenario_execution initial = scenario_execute(body, branch, counts, action, initial_rng, 0);
                for (unsigned i = 0; i < SENTENCES; i++) rep_validate_sentence(body, &branch[i]);
                scenario_execution continuation[4] = {{0}};
                unsigned cumulative = initial.generated;
                nt_spa_observation observed;
                nt_spa_metrics after;
                observation(body, branch, target, counts[target], &observed, &after);
                scenario_write_measurement(&sink, source->episode, source->step, target, 0,
                    source->experience.source_life_hash, source->body_identity, source->agent_rng,
                    &initial, continuation, cumulative, &source->before, &after, branch, counts,
                    body->forwards - original_forwards);
                for (unsigned hop = 1; hop <= 4; hop++) {
                    unsigned future_target = (target + hop) % SENTENCES;
                    nt_spa_action future = {future_target ? NT_SPA_RESEED_LEFT : NT_SPA_RESEED_RIGHT,
                        future_target, future_target ? future_target - 1 : 1};
                    uint64_t future_rng = starts[hop];
                    // SPA_REPLICATES_MUTATE_FUTURE_PAIRING: continuation starts do not inherit action draw counts.
                    if (future_rng != rep_rng(source, replicate, hop)) fail("replicate future RNG pairing mismatch");
                    continuation[hop - 1] = scenario_execute(body, branch, counts, future, future_rng, hop);
                    rep_validate_sentence(body, &branch[future_target]);
                    cumulative += continuation[hop - 1].generated;
                    if (hop == 1 || hop == 4) {
                        // SPA_REPLICATES_MUTATE_TARGET: measure the original intervention target.
                        observation(body, branch, target, counts[target], &observed, &after);
                        scenario_write_measurement(&sink, source->episode, source->step, target, hop,
                            source->experience.source_life_hash, source->body_identity, source->agent_rng,
                            &initial, continuation, cumulative, &source->before, &after, branch, counts,
                            body->forwards - original_forwards);
                    }
                }
                uint64_t branch_forwards = body->forwards - original_forwards;
                if (branch_forwards != cumulative) fail("replicate forward/character count mismatch");
                replicate_forwards += branch_forwards;
                body->forwards = original_forwards;
                nt_train_mode(original_mode);
                // SPA_REPLICATES_MUTATE_SOURCE: branches must leave the imported field and history untouched.
                if (memcmp(source, &saved, sizeof(saved)) || rep_snapshot_witness(source) != source->snapshot_witness ||
                    rep_feature_witness(&source->experience) != source->feature_witness)
                    fail("replicate branch leaked into its fixed source");
                if (body_hash(body) != original_body || body->forwards != original_forwards || nt_is_training() != original_mode)
                    fail("replicate branch changed body or host state");
                scenario_check_tape();
                fputs("{\"type\":\"replicate_restoration\",", file); rep_context(file, source, replicate);
                fputs(",\"action\":", file); write_action(file, &action);
                fprintf(file, ",\"body_hash_before\":\"%016" PRIx64 "\",\"body_hash_after\":\"%016" PRIx64
                    "\",\"snapshot_witness_before\":\"%016" PRIx64 "\",\"snapshot_witness_after\":\"%016" PRIx64
                    "\",\"host_forwards\":%" PRIu64 ",\"diagnostic_forwards\":%" PRIu64
                    ",\"source_unchanged\":true,\"features_unchanged\":true,\"rng_witness_unchanged\":true,"
                    "\"body_unchanged\":true,\"forwards_restored\":true,\"training_mode_restored\":true,\"tape_empty\":true}\n",
                    original_body, body_hash(body), saved.snapshot_witness, rep_snapshot_witness(source), original_forwards, branch_forwards);
                valid++; alternatives++;
            }
            if (sink.measurements - measurements_before != (uint64_t)valid * 3) fail("replicate measurement count mismatch");
            fputs("{\"type\":\"replicate_end\",", file); rep_context(file, source, replicate);
            fprintf(file, ",\"alternatives\":%u,\"measurements\":%" PRIu64 ",\"diagnostic_forwards\":%" PRIu64
                ",\"source_unchanged\":true}\n", valid, sink.measurements - measurements_before, replicate_forwards);
            total_forwards += replicate_forwards; replicates++;
        }
        scenario_flush(&sink);
        fprintf(stderr, "spa_agent_replicates: source %u/%u complete (seed %u, episode %u, step %u; forwards %" PRIu64 ")\n",
            si + 1, data->count, source->seed, source->episode, source->step, total_forwards);
    }
    if (body_hash(body) != original_body || body->forwards != original_forwards || nt_is_training() != original_mode)
        fail("replicate run changed its body state");
    fprintf(file, "{\"type\":\"replicate_summary\",\"sources\":%u,\"replicates\":%" PRIu64
        ",\"alternatives\":%" PRIu64 ",\"measurements\":%" PRIu64 ",\"diagnostic_forwards\":%" PRIu64
        ",\"body_hash\":\"%016" PRIx64 "\",\"host_forwards\":%" PRIu64
        ",\"source_fields_unchanged\":true,\"body_unchanged\":true}\n",
        data->count, replicates, alternatives, sink.measurements, total_forwards, original_body, original_forwards);
    scenario_flush(&sink);
}

static FILE *rep_open_output(const char *path) {
    // SPA_REPLICATES_MUTATE_OPEN_MODE: an existing input or output must survive.
    FILE *file = fopen(path, "wx");
    if (!file) fail("cannot create new replicate output; destination must not exist");
    return file;
}

static void rep_close_output(FILE *file) {
    int failed = ferror(file);
    if (fflush(file)) failed = 1;
    if (fclose(file)) failed = 1;
    if (failed) fail("replicate output flush/close failed");
}

#ifndef SPA_REPLICATES_NO_MAIN
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s CHECKPOINT VOCAB_U32 SNAPSHOTS OUTPUT_JSONL\n", argv[0]);
        return 2;
    }
    sentence_body body;
    load_body(&body, argv[1], argv[2]);
    nt_train_mode(0);
    rep_dataset *data = rep_read_dataset(argv[3], &body);
    FILE *file = rep_open_output(argv[4]);
    rep_run(file, &body, data);
    rep_close_output(file);
    free(data);
    nt_tape_destroy();
    for (int i = 0; i < PARAMS; i++) nt_tensor_free(body.parameters[i]);
    free(body.parameters);
    return 0;
}
#endif
