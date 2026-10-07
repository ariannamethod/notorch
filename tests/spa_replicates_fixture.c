/* Synthetic gate for the production fixed-source replicate host.
 * The body has deterministic token embeddings and a cheap generation callback;
 * real-model weights and generation are not required by this fixture.
 */
#define SPA_REPLICATES_NO_MAIN
#define SPA_SCENARIO_GENERATOR fixture_generate
#include "../examples/spa_agent_replicates.c"
#include <sys/wait.h>
#include <unistd.h>

static void fixture_generate(sentence_body *body, const int *prompt, int length,
                             uint64_t *rng, sentence *out) {
    if (length < 1 || length > CONTEXT) fail("fixture prompt invalid");
    memset(out, 0, sizeof(*out));
    unsigned count = MIN_GENERATED + (unsigned)((*rng >> 9) % (MAX_GENERATED - MIN_GENERATED + 1));
    unsigned pattern = ((unsigned)prompt[length - 1] + (unsigned)(*rng >> 32)) % 4;
    for (unsigned i = 0; i < count; i++) {
        uint64_t draw = next_random(rng);
        unsigned letter = pattern == 0 ? 0 : pattern == 1 ? i % 3 : (unsigned)(draw % 26);
        out->ids[i] = 33 + (int)letter; // ASCII A..Z in the fixture's sorted vocabulary.
        body->forwards++;
    }
    out->ids[count - 1] = 14; // ASCII full stop, exactly the first eligible stop.
    out->length = (int)count; out->stopped = 1;
}

static void fixture_body(sentence_body *body) {
    memset(body, 0, sizeof(*body));
    body->parameters = calloc(PARAMS, sizeof(*body->parameters));
    if (!body->parameters) fail("fixture body allocation failed");
    for (unsigned i = 0; i < VOCAB; i++) body->vocabulary[i] = 32 + i;
    for (int i = 0; i < PARAMS; i++) {
        body->parameters[i] = nt_tensor_new(i ? 1 : VOCAB * DIM);
        if (!body->parameters[i]) fail("fixture tensor allocation failed");
        body->count += body->parameters[i]->len;
        for (int j = 0; j < body->parameters[i]->len; j++)
            body->parameters[i]->data[j] = (float)(((j + 7) * (i + 11)) % 97 - 48) / 64.0f;
    }
    nt_train_mode(0);
}

static void fixture_body_free(sentence_body *body) {
    nt_tape_destroy();
    for (int i = 0; i < PARAMS; i++) nt_tensor_free(body->parameters[i]);
    free(body->parameters);
}

static void fixture_path(char out[4096], const char *prefix, const char *suffix) {
    int n = snprintf(out, 4096, "%s.%s", prefix, suffix);
    if (n < 0 || n >= 4096) fail("fixture path too long");
}

static void fixture_checksums(rep_source *source) {
    source->feature_witness = rep_feature_witness(&source->experience);
    source->snapshot_witness = rep_snapshot_witness(source);
}

static rep_dataset *fixture_sources(sentence_body *body) {
    rep_dataset *data = calloc(1, sizeof(*data));
    if (!data) fail("fixture dataset allocation failed");
    strcpy(data->protocol_sha, "c53b00f02ebbc8d7bd1144839719741887f0a0e3ac411767a3fde7359a59db75");
    data->vocabulary_hash = rep_vocabulary_hash(body);
    data->count = 4;
    nt_spa_agent_config config; nt_spa_agent_config_default(&config);
    config.mode = NT_SPA_AGENT_LEARNED;
    nt_spa_agent life;
    check(nt_spa_agent_init(&life, &config), "fixture life initialization failed");
    for (unsigned i = 0; i < data->count; i++) {
        rep_source *source = &data->sources[i];
        source->ordinal = i; source->seed = 42; source->step = i;
        source->body_identity = body_hash(body); source->agent_rng = life.rng;
        source->host_rng = stream(42, UINT64_C(0x7370615f6163746e), 0, i);
        memset(source->archive_sha, '1', 64);
        memset(source->snapshot_sha, '2', 64); source->snapshot_sha[63] = (char)('0' + i);
        for (unsigned j = 0; j < SENTENCES; j++) {
            sentence *s = &source->chain[j]; s->length = MAX_GENERATED;
            for (unsigned k = 0; k < MAX_GENERATED; k++) {
                unsigned letter = j == 0 ? i : j == 1 ? k % 3 : j == 2 ? (k * 7 + i) % 23 : (k * k + 3 * k + i) % 26;
                s->ids[k] = 33 + (int)letter;
            }
        }
        nt_spa_observation observed;
        observation(body, source->chain, i, 0, &observed, &source->before);
        check(nt_spa_agent_capture_experience(&life, &observed, &source->experience), "fixture capture failed");
        fixture_checksums(source);
    }
    return data;
}

static void fixture_write_metrics(FILE *file, const nt_spa_metrics *m) {
    fprintf(file, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", m->local_connectedness, m->global_connectedness,
        m->coherence, m->novelty, m->repetition, m->collapse, m->continuity);
}

static void fixture_write_sources(const char *path, const rep_dataset *data, int ending) {
    FILE *file = rep_open_output(path);
    fprintf(file, "NT_SPA_REPLICATES_V1\nPROTOCOL %s\nVOCAB_FNV1A %016" PRIx64 "\nCOUNT %u\n",
        data->protocol_sha, data->vocabulary_hash, data->count);
    unsigned written = data->count > REP_MAX_SOURCES ? 0 : data->count;
    for (unsigned i = 0; i < written; i++) {
        const rep_source *source = &data->sources[i];
        const nt_spa_experience *experience = &source->experience;
        fprintf(file, "SNAPSHOT %u %u %u %u %u %u %016" PRIx64 " %016" PRIx64 " %u %016" PRIx64 " %s %s\nFEATURES",
            source->ordinal, source->seed, source->episode, source->step, experience->sentence_index,
            experience->sentence_count, experience->source_life_hash, source->body_identity, source->agent_rng,
            source->host_rng, source->archive_sha, source->snapshot_sha);
        for (int j = 0; j < NT_SPA_AGENT_FEATURES; j++) fprintf(file, " %.9g", experience->features[j]);
        fputs("\nBEFORE ", file); fixture_write_metrics(file, &source->before);
        fputs("RESEEDS", file);
        for (unsigned j = 0; j < SENTENCES; j++) fprintf(file, " %u", source->reseeds[j]);
        fputc('\n', file);
        for (unsigned j = 0; j < SENTENCES; j++) {
            const sentence *s = &source->chain[j];
            fprintf(file, "SENTENCE %u %d %d", j, s->length, s->stopped);
            int length = s->length > MAX_GENERATED ? MAX_GENERATED : s->length;
            for (int k = 0; k < length; k++) fprintf(file, " %d", s->ids[k]);
            fputc('\n', file);
        }
        fprintf(file, "FEATURE_WITNESS %016" PRIx64 "\nSNAPSHOT_WITNESS %016" PRIx64 "\nEND_SNAPSHOT\n",
            source->feature_witness, source->snapshot_witness);
    }
    if (ending) fputs("END\n", file);
    if (ending == 2) fputs("EXTRA\n", file);
    rep_close_output(file);
}

static void fixture_write_r0(const char *path, sentence_body *body, const rep_dataset *data) {
    FILE *out = rep_open_output(path);
    nt_spa_agent_config config; nt_spa_agent_config_default(&config); config.mode = NT_SPA_AGENT_LEARNED;
    nt_spa_agent life; check(nt_spa_agent_init(&life, &config), "fixture legacy life initialization failed");
    for (unsigned i = 0; i < data->count; i++) {
        const rep_source *source = &data->sources[i];
        FILE *temporary = tmpfile(); if (!temporary) fail("fixture temporary trace failed");
        scenario_sink sink = {0}; sink.file = temporary; sink.seed = source->seed;
        experiment_arm arm = {0}; arm.name = "fixture"; arm.life = life; arm.initial = life.policy;
        sentence chain[SENTENCES]; memcpy(chain, source->chain, sizeof(chain));
        unsigned reseeds[SENTENCES]; memcpy(reseeds, source->reseeds, sizeof(reseeds));
        nt_spa_observation observed; nt_spa_metrics before;
        observation(body, chain, source->experience.sentence_index, 0, &observed, &before);
        // This is the original scenario fork implementation, not the new host.
        scenario_snapshot(&sink, body, &arm, chain, reseeds, source->seed, source->episode,
                          source->step, source->experience.sentence_index, &observed, &before);
        rewind(temporary);
        char line[16384];
        while (fgets(line, sizeof(line), temporary)) {
            if (!strchr(line, '\n')) fail("fixture golden line overflow");
            if (!strncmp(line, "{\"type\":\"measurement\",", 22)) fputs(line, out);
        }
        if (ferror(temporary) || fclose(temporary)) fail("fixture golden temporary close failed");
    }
    rep_close_output(out);
}

static int fixture_write(const char *prefix) {
    sentence_body body; fixture_body(&body);
    rep_dataset *data = fixture_sources(&body);
    char path[4096]; fixture_path(path, prefix, "sources.txt"); fixture_write_sources(path, data, 1);
    rep_dataset *loaded = rep_read_dataset(path, &body);
    fixture_path(path, prefix, "sources.jsonl"); FILE *sources = rep_open_output(path);
    for (unsigned i = 0; i < loaded->count; i++) rep_write_source(sources, &loaded->sources[i]);
    rep_close_output(sources);
    fixture_path(path, prefix, "expected_r0.jsonl"); fixture_write_r0(path, &body, data);
    printf("{\"type\":\"fixture_written\",\"sources\":4,\"r0_measurements\":30}\n");
    free(loaded); free(data); fixture_body_free(&body);
    return 0;
}

static int fixture_run(const char *source, const char *output, int body_fault) {
    sentence_body body; fixture_body(&body);
    if (body_fault == 1) body.vocabulary[5] = body.vocabulary[4];
    if (body_fault == 2) body.vocabulary[VOCAB - 1] = UINT32_C(0x110000);
    rep_dataset *data = rep_read_dataset(source, &body);
    FILE *file = rep_open_output(output);
    rep_run(file, &body, data);
    rep_close_output(file);
    free(data); fixture_body_free(&body);
    return 0;
}

static uint64_t fixture_file_hash(const char *path) {
    FILE *file = fopen(path, "rb"); if (!file) fail("fixture cannot hash file");
    uint64_t hash = UINT64_C(14695981039346656037);
    unsigned char buffer[4096]; size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file))) hash = hash_bytes(hash, buffer, count);
    if (ferror(file) || fclose(file)) fail("fixture file hash read failed");
    return hash;
}

static void fixture_refusal(const char *source, const char *output, int body_fault,
                            const char *name, const char *expected) {
    int errors[2]; if (pipe(errors)) fail("fixture pipe failed");
    if (fflush(NULL)) fail("fixture pre-fork flush failed");
    pid_t pid = fork(); if (pid < 0) fail("fixture fork failed");
    if (pid == 0) {
        close(errors[0]);
        if (dup2(errors[1], STDERR_FILENO) < 0) _exit(3);
        close(errors[1]);
        exit(fixture_run(source, output, body_fault));
    }
    close(errors[1]);
    char message[4096]; size_t used = 0; char part[512]; ssize_t n;
    while ((n = read(errors[0], part, sizeof(part))) > 0) {
        size_t take = (size_t)n;
        if (take > sizeof(message) - 1 - used) take = sizeof(message) - 1 - used;
        memcpy(message + used, part, take); used += take;
    }
    message[used] = 0; close(errors[0]);
    int status;
    if (n < 0 || waitpid(pid, &status, 0) != pid) fail("fixture child wait failed");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 1 || !strstr(message, expected)) {
        fprintf(stderr, "fixture refusal gate %s failed: status=%d stderr=%s\n", name, status, message);
        fail("fixture expected a named normal-exit refusal");
    }
    printf("{\"type\":\"fixture_gate\",\"name\":\"%s\",\"exit_code\":1,\"expected_diagnostic\":\"%s\",\"pass\":true}\n", name, expected);
}

static int fixture_cases(const char *prefix) {
    sentence_body body; fixture_body(&body);
    rep_dataset *base = fixture_sources(&body), *changed = malloc(sizeof(*changed));
    if (!changed) fail("fixture mutated dataset allocation failed");
    char input[4096], output[4096], base_path[4096];
    fixture_path(base_path, prefix, "valid.txt"); fixture_write_sources(base_path, base, 1);
    uint64_t original_hash = fixture_file_hash(base_path);
    struct fault { const char *name, *message; } faults[] = {
        {"count_zero", "source count outside bounds"}, {"count_overflow", "source count outside bounds"},
        {"bad_coordinate", "source coordinates or body identity mismatch"}, {"duplicate_ordinal", "ordinals not strictly ordered"},
        {"bad_length", "sentence dimensions invalid"}, {"bad_token", "token outside vocabulary"},
        {"bad_termination_flag", "sentence dimensions invalid"}, {"missed_stop", "passed an eligible stop"},
        {"bad_feature_witness", "source witness mismatch"}, {"bad_snapshot_witness", "source witness mismatch"},
        {"bad_host_rng", "host RNG differs"}, {"bad_body_identity", "body identity mismatch"},
        {"bad_vocabulary_hash", "vocabulary witness mismatch"}, {"false_raw_metrics", "metrics differ"},
        {"false_sensory_features", "sensory features differ"}, {"truncated_input", "truncated replicate source"},
        {"trailing_input", "trailing replicate source data"}, {"bad_reseed_bound", "reseed count outside bounds"}
    };
    unsigned count = (unsigned)(sizeof(faults) / sizeof(faults[0]));
    for (unsigned f = 0; f < count; f++) {
        memcpy(changed, base, sizeof(*changed));
        rep_source *s = &changed->sources[0]; int ending = 1;
        switch (f) {
            case 0: changed->count = 0; break;
            case 1: changed->count = REP_MAX_SOURCES + 1; break;
            case 2: s->seed = 0; break;
            case 3: changed->sources[1].ordinal = 0; break;
            case 4: s->chain[0].length = 0; break;
            case 5: s->chain[0].ids[0] = VOCAB; break;
            case 6: s->chain[0].stopped = 2; break;
            case 7: s->chain[0].ids[MIN_GENERATED - 1] = 14; break;
            case 8: s->feature_witness ^= 1; break;
            case 9: s->snapshot_witness ^= 1; break;
            case 10: s->host_rng ^= 1; break;
            case 11: s->body_identity ^= 1; break;
            case 12: changed->vocabulary_hash ^= 1; break;
            case 13: s->before.local_connectedness = 0.125f; fixture_checksums(s); break;
            case 14: s->experience.features[0] += 0.125f; fixture_checksums(s); break;
            case 15: ending = 0; break;
            case 16: ending = 2; break;
            case 17: s->reseeds[0] = EPISODES * SENTENCES + 1; break;
            default: fail("unknown fixture fault");
        }
        char suffix[128]; snprintf(suffix, sizeof(suffix), "%s.txt", faults[f].name);
        fixture_path(input, prefix, suffix); fixture_write_sources(input, changed, ending);
        snprintf(suffix, sizeof(suffix), "%s.output.jsonl", faults[f].name); fixture_path(output, prefix, suffix);
        fixture_refusal(input, output, 0, faults[f].name, faults[f].message);
        struct stat info; if (stat(output, &info) == 0) fail("malformed import created an output");
    }
    fixture_path(output, prefix, "bad_vocabulary_order.output.jsonl");
    fixture_refusal(base_path, output, 1, "bad_vocabulary_order", "sorted unique Unicode scalars");
    fixture_path(output, prefix, "bad_unicode.output.jsonl");
    fixture_refusal(base_path, output, 2, "bad_unicode", "sorted unique Unicode scalars");
    fixture_refusal(base_path, base_path, 0, "input_as_output", "destination must not exist");
    fixture_path(output, prefix, "existing.output.jsonl");
    FILE *existing = rep_open_output(output); fputs("fixture existing output\n", existing); rep_close_output(existing);
    uint64_t existing_hash = fixture_file_hash(output);
    fixture_refusal(base_path, output, 0, "existing_output", "destination must not exist");
    if (fixture_file_hash(output) != existing_hash) fail("existing output was changed");
    fixture_path(output, prefix, "input_symlink");
    const char *base_name = strrchr(base_path, '/'); base_name = base_name ? base_name + 1 : base_path;
    if (symlink(base_name, output)) fail("fixture source symlink failed");
    fixture_refusal(base_path, output, 0, "input_symlink_output", "destination must not exist");
    if (fixture_file_hash(base_path) != original_hash || fixture_file_hash(output) != original_hash)
        fail("fixture source bytes were changed");
    memcpy(changed, base, sizeof(*changed));
    changed->sources[0].experience.features[19] = nextafterf(0.0f, 1.0f);
    changed->sources[0].experience.features[22] = 1;
    changed->sources[0].experience.features[25] = 1;
    fixture_checksums(&changed->sources[0]);
    fixture_path(input, prefix, "finite_subnormal.txt"); fixture_write_sources(input, changed, 1);
    rep_dataset *subnormal = rep_read_dataset(input, &body);
    if (subnormal->sources[0].experience.features[19] != changed->sources[0].experience.features[19])
        fail("finite subnormal source feature did not round-trip");
    free(subnormal);
    printf("{\"type\":\"fixture_cases\",\"cases\":%u,\"named_refusals\":%u,\"finite_subnormal_roundtrip\":true,"
        "\"inputs_preserved\":true,\"pass\":true}\n", count + 6, count + 5);
    free(changed); free(base); fixture_body_free(&body);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "write")) return fixture_write(argv[2]);
    if (argc == 4 && !strcmp(argv[1], "run")) return fixture_run(argv[2], argv[3], 0);
    if (argc == 3 && !strcmp(argv[1], "cases")) return fixture_cases(argv[2]);
    fprintf(stderr, "usage: %s write PREFIX | run SNAPSHOTS OUTPUT | cases PREFIX\n", argv[0]);
    return 2;
}
