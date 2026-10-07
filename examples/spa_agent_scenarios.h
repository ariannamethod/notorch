/* Fixed common-state SPA action interventions.
 * Included by spa_agent_demo.c after its native body/measurement functions.
 * Protocol: experiments/spa_agent/scenarios/protocol.json. No policy learning
 * occurs inside a diagnostic branch; every generated character is accounted.
 */
#ifndef NOTORCH_SPA_AGENT_SCENARIOS_H
#define NOTORCH_SPA_AGENT_SCENARIOS_H
#include <sys/stat.h>

/* A deterministic cheap generator can be supplied by the independent fixture.
 * Production builds use the exact native generate() implementation above. */
#ifdef SPA_SCENARIO_GENERATOR
static void SPA_SCENARIO_GENERATOR(sentence_body *, const int *, int,
                                   uint64_t *, sentence *);
#else
#define SPA_SCENARIO_GENERATOR generate
#endif

typedef struct {
    unsigned hop, generated;
    nt_spa_action action;
    uint64_t rng_before, rng_after;
    float cost;
} scenario_execution;

typedef struct {
    int valid;
    sentence chain[SENTENCES];
    unsigned reseeds[SENTENCES];
    nt_spa_metrics after;
    float cost;
    uint64_t rng_after;
} scenario_immediate;

typedef struct {
    FILE *file;
    uint32_t seed;
    uint64_t snapshots, alternatives, measurements, diagnostic_forwards;
    uint64_t matched_actual_actions;
    int pending;
    unsigned pending_episode, pending_step;
    nt_spa_decision preview;
    scenario_immediate immediate[NT_SPA_AGENT_ACTIONS];
} scenario_sink;

static void scenario_open(scenario_sink *sink, const char *path, uint32_t seed) {
    memset(sink, 0, sizeof(*sink));
    if (!path || !*path) fail("empty scenario path");
    sink->file = fopen(path, "wx");
    if (!sink->file) fail("cannot create new scenario trace");
    sink->seed = seed;
}

static void scenario_require_separate(const scenario_sink *sink, const char *host_path) {
    struct stat scenario_info, host_info;
    if (fstat(fileno(sink->file), &scenario_info)) fail("cannot identify scenario output");
    if (stat(host_path, &host_info) == 0) {
        if (scenario_info.st_dev == host_info.st_dev && scenario_info.st_ino == host_info.st_ino)
            fail("scenario trace aliases a host output");
    } else if (errno != ENOENT && errno != ENOTDIR) fail("cannot inspect host output path");
}

static void scenario_flush(scenario_sink *sink) {
    if (ferror(sink->file) || fflush(sink->file)) fail("scenario trace write/flush failed");
}

static void scenario_write_reseeds(FILE *file, const unsigned reseeds[SENTENCES]) {
    fputc('[', file);
    for (int i = 0; i < SENTENCES; i++) fprintf(file, "%s%u", i ? "," : "", reseeds[i]);
    fputc(']', file);
}

static int scenario_metrics_equal(const nt_spa_metrics *a, const nt_spa_metrics *b) {
    return a->local_connectedness == b->local_connectedness &&
           a->global_connectedness == b->global_connectedness &&
           a->coherence == b->coherence && a->novelty == b->novelty &&
           a->repetition == b->repetition && a->collapse == b->collapse &&
           a->continuity == b->continuity;
}

static float scenario_reward(const nt_spa_metrics *before, const nt_spa_metrics *after,
                             float cost) {
    double value = 0;
    value += (double)0.15f * (after->local_connectedness - before->local_connectedness);
    value += (double)0.15f * (after->global_connectedness - before->global_connectedness);
    value += (double)0.20f * (after->coherence - before->coherence);
    value += (double)0.20f * (after->novelty - before->novelty);
    value += (double)0.15f * (before->repetition - after->repetition);
    value += (double)0.10f * (before->collapse - after->collapse);
    value += (double)0.05f * (after->continuity - before->continuity);
    value -= (double)0.05f * cost;
    float reward = (float)value;
    // SPA_SCENARIO_MUTATE_REWARD: independent gates recompute every raw axis.
    return reward < -1 ? -1 : reward > 1 ? 1 : reward;
}

static void scenario_check_tape(void) {
    const nt_tape *tape = nt_tape_get();
    if (tape->active || tape->count || tape->n_params) fail("scenario requires empty inference tape");
}

static scenario_execution scenario_execute(sentence_body *body, sentence chain[SENTENCES],
                                           unsigned reseeds[SENTENCES], nt_spa_action action,
                                           uint64_t rng, unsigned hop) {
    scenario_execution executed = {0};
    executed.hop = hop;
    executed.action = action;
    executed.rng_before = rng;
    nt_spa_observation bounds = {0};
    bounds.temperature = 0.8f;
    bounds.sentence_count = SENTENCES;
    bounds.sentence_index = action.target;
    check(nt_spa_action_validate(&action, &bounds), "invalid diagnostic host action");
    uint64_t forwards_before = body->forwards;
    if (action.kind != NT_SPA_KEEP) {
        const sentence *neighbor = &chain[action.source];
        int length = neighbor->length > 3 ? 3 : neighbor->length;
        if (length < 1) fail("diagnostic host refused empty neighbor");
        sentence candidate;
        SPA_SCENARIO_GENERATOR(body, neighbor->ids + neighbor->length - length,
                               length, &rng, &candidate);
        if (candidate.length < MIN_GENERATED || candidate.length > MAX_GENERATED)
            fail("diagnostic generator length outside protocol");
        for (int i = 0; i < candidate.length; i++)
            if (candidate.ids[i] < 0 || candidate.ids[i] >= VOCAB) fail("diagnostic generated invalid token");
        chain[action.target] = candidate;
        reseeds[action.target]++;
        executed.generated = (unsigned)candidate.length;
        executed.cost = (float)candidate.length / MAX_GENERATED;
    }
    executed.rng_after = rng;
    uint64_t expected_rng = executed.rng_before;
    for (unsigned i = 0; i < executed.generated; i++) next_random(&expected_rng);
    if (expected_rng != rng || body->forwards - forwards_before != executed.generated)
        fail("diagnostic generation RNG/forward accounting differs");
    scenario_check_tape();
    return executed;
}

static void scenario_write_execution(FILE *file, const scenario_execution *execution) {
    fprintf(file, "{\"hop\":%u,\"action\":", execution->hop);
    write_action(file, &execution->action);
    fprintf(file, ",\"rng_before\":\"%016" PRIx64 "\",\"rng_after\":\"%016" PRIx64
                  "\",\"generated\":%u,\"cost\":%.9g}", execution->rng_before,
            execution->rng_after, execution->generated, execution->cost);
}

static void scenario_write_measurement(scenario_sink *sink, unsigned episode, unsigned step,
    unsigned target, unsigned horizon, uint64_t life_hash, uint64_t weights_hash,
    uint32_t agent_rng, const scenario_execution *initial,
    const scenario_execution continuation[4], unsigned cumulative,
    const nt_spa_metrics *before, const nt_spa_metrics *after,
    const sentence chain[SENTENCES], const unsigned reseeds[SENTENCES], uint64_t forwards) {
    float cost = (float)cumulative / (MAX_GENERATED * (horizon + 1));
    float reward = scenario_reward(before, after, cost);
    if (!isfinite(cost) || cost < 0 || cost > 1 || !isfinite(reward)) fail("invalid horizon consequence");
    FILE *file = sink->file;
    fprintf(file, "{\"type\":\"measurement\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"snapshot\":%u,"
                  "\"target\":%u,\"horizon\":%u,\"life_hash\":\"%016" PRIx64 "\",\"body_hash\":\"%016" PRIx64
                  "\",\"agent_rng\":%u,\"host_rng_before\":\"%016" PRIx64 "\",\"action\":",
            sink->seed, episode, step, episode * SENTENCES + step, target, horizon,
            life_hash, weights_hash, agent_rng, initial->rng_before);
    write_action(file, &initial->action);
    fprintf(file, ",\"initial_rng_before\":\"%016" PRIx64 "\",\"initial_rng_after\":\"%016" PRIx64
                  "\",\"initial_generated\":%u,\"initial_cost\":%.9g,\"cumulative_generated\":%u,"
                  "\"cost_denominator\":%u,\"cost_normalized\":%.9g,\"reward\":%.9g,\"branch_forwards\":%" PRIu64,
            initial->rng_before, initial->rng_after, initial->generated, initial->cost,
            cumulative, MAX_GENERATED * (horizon + 1), cost, reward, forwards);
    fputs(",\"before\":", file); write_metrics(file, before);
    fputs(",\"after\":", file); write_metrics(file, after);
    fputs(",\"chain\":", file); write_chain(file, chain);
    fputs(",\"reseeds\":", file); scenario_write_reseeds(file, reseeds);
    fputs(",\"continuation\":[", file);
    for (unsigned i = 0; i < horizon; i++) {
        if (i) fputc(',', file);
        scenario_write_execution(file, &continuation[i]);
    }
    fputs("]}\n", file);
    sink->measurements++;
}

static void scenario_snapshot(scenario_sink *sink, sentence_body *body, experiment_arm *arm,
    sentence chain[SENTENCES], unsigned reseeds[SENTENCES], uint32_t seed,
    unsigned episode, unsigned step, unsigned target, const nt_spa_observation *obs,
    const nt_spa_metrics *before) {
    if (!sink->file) return;
    if (sink->pending || seed != sink->seed || arm->life.pending) fail("invalid scenario snapshot boundary");
    scenario_check_tape();
    experiment_arm saved_arm;
    memcpy(&saved_arm, arm, sizeof(saved_arm));
    sentence saved_chain[SENTENCES]; memcpy(saved_chain, chain, sizeof(saved_chain));
    unsigned saved_reseeds[SENTENCES]; memcpy(saved_reseeds, reseeds, sizeof(saved_reseeds));
    uint64_t host_forwards = body->forwards, weights_hash = body_hash(body);
    uint64_t life_hash = nt_spa_agent_hash(&arm->life);
    if (!life_hash) fail("invalid diagnostic agent snapshot");
    int training_before = nt_is_training();
    uint64_t initial_rng = stream(seed, UINT64_C(0x7370615f6163746e), episode, step);
    uint64_t future_rng[4];
    for (unsigned hop = 1; hop <= 4; hop++)
        future_rng[hop - 1] = stream(seed, UINT64_C(0x7370615f66757472), episode * SENTENCES + step, hop);
    check(nt_spa_agent_select(&arm->life, obs, &sink->preview), "scenario policy preview failed");
    memset(sink->immediate, 0, sizeof(sink->immediate));
    sink->pending_episode = episode; sink->pending_step = step;
    FILE *file = sink->file;
    fprintf(file, "{\"type\":\"snapshot\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"snapshot\":%u,"
                  "\"target\":%u,\"life_hash\":\"%016" PRIx64 "\",\"body_hash\":\"%016" PRIx64
                  "\",\"agent_rng\":%u,\"host_rng_before\":\"%016" PRIx64 "\",\"host_forwards\":%" PRIu64
                  ",\"training_mode\":%d,\"policy_rng_before\":%u,\"policy_action\":",
            seed, episode, step, episode * SENTENCES + step, target, life_hash, weights_hash,
            arm->life.rng, initial_rng, host_forwards, training_before, sink->preview.rng_before);
    write_action(file, &sink->preview.action);
    fputs(",\"features\":", file); write_floats(file, sink->preview.features, NT_SPA_AGENT_FEATURES);
    fputs(",\"scores\":", file); write_floats(file, sink->preview.scores, NT_SPA_AGENT_ACTIONS);
    fputs(",\"before\":", file); write_metrics(file, before);
    fputs(",\"chain\":", file); write_chain(file, chain);
    fputs(",\"reseeds\":", file); scenario_write_reseeds(file, reseeds); fputs("}\n", file);
    unsigned alternatives = 0;
    uint64_t snapshot_forwards = 0;
    for (int kind = NT_SPA_KEEP; kind <= NT_SPA_RESEED_RIGHT; kind++) {
        nt_spa_action action = {(nt_spa_action_kind)kind, target, NT_SPA_AGENT_NO_SOURCE};
        if (kind == NT_SPA_RESEED_LEFT) action.source = target ? target - 1 : NT_SPA_AGENT_NO_SOURCE;
        if (kind == NT_SPA_RESEED_RIGHT) action.source = target + 1;
        if (nt_spa_action_validate(&action, obs) != NT_SPA_OK) continue;
        sentence branch[SENTENCES]; memcpy(branch, saved_chain, sizeof(branch));
        unsigned counts[SENTENCES]; memcpy(counts, saved_reseeds, sizeof(counts));
        scenario_execution initial = scenario_execute(body, branch, counts, action, initial_rng, 0);
        scenario_execution continuation[4] = {{0}};
        unsigned cumulative = initial.generated;
        nt_spa_observation measured_observation;
        nt_spa_metrics after;
        observation(body, branch, target, counts[target], &measured_observation, &after);
        scenario_immediate *immediate = &sink->immediate[kind];
        immediate->valid = 1;
        memcpy(immediate->chain, branch, sizeof(branch));
        memcpy(immediate->reseeds, counts, sizeof(counts));
        immediate->after = after; immediate->cost = initial.cost; immediate->rng_after = initial.rng_after;
        scenario_write_measurement(sink, episode, step, target, 0, life_hash, weights_hash,
            saved_arm.life.rng, &initial, continuation, cumulative, before, &after,
            branch, counts, body->forwards - host_forwards);
        for (unsigned hop = 1; hop <= 4; hop++) {
            unsigned future_target = (target + hop) % SENTENCES;
            nt_spa_action future = {future_target ? NT_SPA_RESEED_LEFT : NT_SPA_RESEED_RIGHT,
                                   future_target, future_target ? future_target - 1 : 1};
            uint64_t paired_rng = future_rng[hop - 1];
            // SPA_SCENARIO_MUTATE_FUTURE_RNG: paired starts are action-independent.
            if (paired_rng != future_rng[hop - 1]) fail("diagnostic future RNG pairing mismatch");
            continuation[hop - 1] = scenario_execute(body, branch, counts, future, paired_rng, hop);
            cumulative += continuation[hop - 1].generated;
            if (hop == 1 || hop == 4) {
                observation(body, branch, target, counts[target], &measured_observation, &after);
                scenario_write_measurement(sink, episode, step, target, hop, life_hash, weights_hash,
                    saved_arm.life.rng, &initial, continuation, cumulative, before, &after,
                    branch, counts, body->forwards - host_forwards);
            }
        }
        uint64_t branch_forwards = body->forwards - host_forwards;
        if (branch_forwards != cumulative) fail("diagnostic forward count differs from generated characters");
        snapshot_forwards += branch_forwards;
        body->forwards = host_forwards;
        nt_train_mode(training_before);
        // SPA_SCENARIO_MUTATE_HOST_CHAIN: a leaked branch must fail before join.
        if (memcmp(chain, saved_chain, sizeof(saved_chain))) fail("scenario leaked host chain");
        if (memcmp(reseeds, saved_reseeds, sizeof(saved_reseeds))) fail("scenario leaked host reseed counters");
        if (memcmp(arm, &saved_arm, sizeof(saved_arm)) || nt_spa_agent_hash(&arm->life) != life_hash)
            fail("scenario leaked persistent agent state");
        if (body_hash(body) != weights_hash) fail("scenario changed body weights");
        if (arm->life.rng != saved_arm.life.rng ||
            stream(seed, UINT64_C(0x7370615f6163746e), episode, step) != initial_rng)
            fail("scenario changed host or agent RNG");
        if (body->forwards != host_forwards || nt_is_training() != training_before)
            fail("scenario failed to restore host counters/mode");
        scenario_check_tape();
        alternatives++;
    }
    if (!sink->immediate[sink->preview.action.kind].valid) fail("policy chose missing diagnostic alternative");
    fprintf(file, "{\"type\":\"restoration\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"snapshot\":%u,"
                  "\"life_hash_before\":\"%016" PRIx64 "\",\"life_hash_after\":\"%016" PRIx64
                  "\",\"body_hash_before\":\"%016" PRIx64 "\",\"body_hash_after\":\"%016" PRIx64
                  "\",\"agent_rng\":%u,\"host_rng_before\":\"%016" PRIx64 "\",\"host_forwards\":%" PRIu64
                  ",\"diagnostic_forwards\":%" PRIu64 ",\"alternatives\":%u,"
                  "\"chain_unchanged\":true,\"reseeds_unchanged\":true,\"life_unchanged\":true,"
                  "\"body_unchanged\":true,\"rng_unchanged\":true,\"forwards_restored\":true,"
                  "\"tape_empty\":true,\"training_mode_restored\":true}\n",
            seed, episode, step, episode * SENTENCES + step, life_hash, nt_spa_agent_hash(&arm->life),
            weights_hash, body_hash(body), arm->life.rng, initial_rng, host_forwards, snapshot_forwards, alternatives);
    sink->snapshots++; sink->alternatives += alternatives;
    sink->diagnostic_forwards += snapshot_forwards;
    sink->pending = 1;
    scenario_flush(sink);
}

static void scenario_match_actual(scenario_sink *sink, const nt_spa_decision *decision,
    const sentence chain[SENTENCES], const unsigned reseeds[SENTENCES], float cost,
    uint64_t host_rng, const nt_spa_metrics *after) {
    if (!sink->file) return;
    if (!sink->pending || !same_action(&decision->action, &sink->preview.action) ||
        decision->rng_before != sink->preview.rng_before ||
        memcmp(decision->features, sink->preview.features, sizeof(decision->features)) ||
        memcmp(decision->scores, sink->preview.scores, sizeof(decision->scores)))
        fail("actual policy choice differs from common-state preview");
    const scenario_immediate *expected = &sink->immediate[decision->action.kind];
    if (!expected->valid || memcmp(chain, expected->chain, sizeof(expected->chain)) ||
        memcmp(reseeds, expected->reseeds, sizeof(expected->reseeds)) || expected->cost != cost ||
        expected->rng_after != host_rng || !scenario_metrics_equal(&expected->after, after))
        fail("selected diagnostic initial action differs from actual host consequence");
    fprintf(sink->file, "{\"type\":\"selected_initial\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"snapshot\":%u,"
                        "\"action\":", sink->seed, sink->pending_episode, sink->pending_step,
            sink->pending_episode * SENTENCES + sink->pending_step);
    write_action(sink->file, &decision->action);
    fputs(",\"chain_equal\":true,\"reseeds_equal\":true,\"cost_equal\":true,"
          "\"rng_equal\":true,\"metrics_equal\":true,\"policy_input_equal\":true}\n", sink->file);
    sink->pending = 0; sink->matched_actual_actions++;
    scenario_flush(sink);
}

static void scenario_close(scenario_sink *sink) {
    if (!sink->file) return;
    if (sink->pending || sink->snapshots != sink->matched_actual_actions)
        fail("incomplete scenario action confirmation");
    fprintf(sink->file, "{\"type\":\"summary\",\"seed\":%u,\"snapshots\":%" PRIu64
                        ",\"alternatives\":%" PRIu64 ",\"measurements\":%" PRIu64
                        ",\"diagnostic_forwards\":%" PRIu64 ",\"matched_actual_actions\":%" PRIu64 "}\n",
            sink->seed, sink->snapshots, sink->alternatives, sink->measurements,
            sink->diagnostic_forwards, sink->matched_actual_actions);
    scenario_flush(sink);
    if (fclose(sink->file)) fail("scenario trace close failed");
    sink->file = NULL;
}
#endif
