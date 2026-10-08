/* Measured action futures from states reached by a sealed, living parent.
 * Each branch advances native Architect history through its actual choices. */
#ifndef NOTORCH_CHUCK_ARCHITECT_LIVED_H
#define NOTORCH_CHUCK_ARCHITECT_LIVED_H

typedef struct {
    body *model;
    const uint32_t *data;
    size_t offset;
} lived_window;

typedef struct {
    int step;
    nt_chuck_action_kind action;
    float before, after;
    uint64_t windows, policy_pre, policy_post, state_hash;
} lived_transition;

static float lived_after(void *context) {
    lived_window *w = context;
    return read_loss_raw(body_forward(w->model, w->data, w->offset, 0));
}

static int lived_checkpoint(int step) {
    return step == 1 || step == 64 || step == 128 || step == 192 ||
           step == 256 || step == 320 || step == 384 || step == 448;
}

static int trajectory_checkpoint(int step) {
    return step == 1 || step == 17 || step == 33 || step == 49 ||
           step == 65 || step == 97 || step == 129 || step == 193;
}

static void trajectory_continuations(FILE *trace, int alias) {
    fprintf(trace, ",\"executed_continuations\":[\"policy\"%s],\"continuation_aliases\":%s",
            alias ? "" : ",\"student\"", alias ? "{\"student\":\"policy\"}" : "{}");
}

static float lived_probe(body *m, const uint32_t *data, const size_t *offsets, int n) {
    double total = 0;
    for (int i = 0; i < n; ++i) {
        float loss = read_loss_raw(body_forward(m, data, offsets[i], 0));
        if (!isfinite(loss)) return loss;
        total += loss;
    }
    return (float)(total / n);
}

static float lived_heldout(body *m, const uint32_t *data, size_t split, size_t count) {
    size_t offsets[EVAL_WINDOWS], width = count - split - CTX;
    for (int i = 0; i < EVAL_WINDOWS; ++i) offsets[i] = split + width * (size_t)i / EVAL_WINDOWS;
    return lived_probe(m, data, offsets, EVAL_WINDOWS);
}

static const char *lived_number_status(float value) {
    return isfinite(value) ? "finite" : isnan(value) ? "nan" :
           value < 0 ? "negative_infinity" : "positive_infinity";
}

static void lived_branch_failure(FILE *trace, int checkpoint, const char *continuation,
        const char *intervention, int update, const char *reason, float loss,
        float gradient_norm, int gradient_measured) {
    char loss_json[64], norm_json[64];
    json_number(loss_json, sizeof(loss_json), loss, 9);
    json_number(norm_json, sizeof(norm_json), gradient_measured ? gradient_norm : NAN, 9);
    fprintf(trace, "{\"type\":\"branch_failure\",\"checkpoint\":%d,\"continuation\":\"%s\","
            "\"intervention\":\"%s\",\"update\":%d,\"reason\":\"%s\",\"before\":%s,"
            "\"before_status\":\"%s\",\"gradient_norm\":%s,\"gradient_status\":\"%s\",\"stopped\":true}\n",
            checkpoint, continuation, intervention, update, reason, loss_json, lived_number_status(loss),
            norm_json, gradient_measured ? lived_number_status(gradient_norm) : "not_measured");
}

static void lived_write_architect(FILE *f, const nt_chuck_architect_decision *d,
        const nt_chuck_architect_receipt *r, uint64_t pre, uint64_t pending, uint64_t post, int forced) {
    const nt_chuck_observation *o = &d->observation;
    static const char *names[] = {"legacy", "hold", "brake", "push"};
    char after_json[64], delta_json[64];
    json_number(after_json, sizeof(after_json), r->after_loss, 9);
    json_number(delta_json, sizeof(delta_json), r->loss_delta, 17);
    fprintf(f, "{\"sequence\":%" PRIu64 ",\"explored\":%d,\"action\":{\"type\":\"%s\",\"kind\":%d,\"value\":%.9g},"
            "\"policy_pre\":\"%016" PRIx64 "\",\"policy_pending\":", d->sequence, d->explored,
            names[d->action.kind], d->action.kind, d->action.value, pre);
    if (forced) fputs("null", f); else fprintf(f, "\"%016" PRIx64 "\"", pending);
    fprintf(f, ",\"policy_post\":\"%016" PRIx64 "\",\"intervention\":%s,\"observation\":{"
            "\"loss\":%.9g,\"loss_ema\":%.9g,\"loss_trend\":%.9g,\"macro_ema\":%.9g,"
            "\"best_macro\":%.9g,\"dampen\":%.9g,\"lr_scale\":%.9g,\"noise\":%.9g,"
            "\"grad_norm\":%.9g,\"grad_trend\":%.9g,\"frozen_fraction\":%.9g,"
            "\"step\":%d,\"stag\":%d,\"macro_stag\":%d,\"history_len\":%d},\"features\":[",
            post, forced ? "true" : "false", o->loss, o->loss_ema, o->loss_trend, o->macro_ema,
            o->best_macro, o->dampen, o->lr_scale, o->noise, o->grad_norm, o->grad_trend,
            o->frozen_fraction, o->step, o->stag, o->macro_stag, o->history_len);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i) fprintf(f, "%s%.9g", i ? "," : "", d->features[i]);
    fputs("],\"scores\":[", f);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_HEADS; ++i) fprintf(f, "%s%.9g", i ? "," : "", d->scores[i]);
    fprintf(f, "],\"consequence\":{\"before_loss\":%.9g,\"after_loss\":%s,\"loss_delta\":%s,"
            "\"reward\":%.9g,\"predicted\":%.9g,\"error\":%.9g,\"decision\":%" PRIu64 ",\"learned\":%d,\"nonfinite\":%d}}",
            r->before_loss, after_json, delta_json, r->reward, r->predicted, r->error,
            r->decision, r->learned, r->nonfinite);
}

static void lived_save_source(body *m, const nt_chuck_architect *a, uint64_t windows,
                              int step, const char *prefix) {
    char fork_prefix[4096], suffix[64], path[4096];
    snprintf(suffix, sizeof(suffix), ".fork-%d", step);
    path_for(fork_prefix, sizeof(fork_prefix), prefix, suffix);
    path_for(path, sizeof(path), fork_prefix, ".body.bin");
    if (nt_save(path, m->param, m->count)) die("lived source body save failed");
    path_for(path, sizeof(path), fork_prefix, ".policy.bin");
    if (nt_chuck_architect_save(a, path)) die("lived source life save failed");
    save_optimizer(fork_prefix, windows, step - 1);
    nt_tensor *gradients[MAX_PARAMS] = {0};
    const nt_tape *tape = nt_tape_get();
    for (int i = 0; i < tape->count; ++i)
        if (tape->entries[i].is_param) gradients[tape->entries[i].slot] = tape->entries[i].grad;
    for (int i = 0; i < m->count; ++i) if (!gradients[i]) die("lived source gradient missing");
    path_for(path, sizeof(path), fork_prefix, ".gradients.bin");
    if (nt_save(path, gradients, m->count)) die("lived source gradient save failed");
}

/* Returns zero after retaining a failed branch and restoring the host exactly.
 * expected receives the source-selected policy continuation's 16 actual states. */
static int lived_diagnose(body *m, const uint32_t *data, size_t count, size_t split,
        nt_chuck_architect *architect, uint64_t *windows, size_t offset, float before, float gradient_norm, float lr,
        int step, const char *prefix, FILE *trace, lived_transition expected[SCENARIO_HORIZON],
        const nt_chuck_architect *continuation) {
    static const char *names[] = {"hold", "brake", "push"};
    const char *continuations[] = {continuation ? "policy" : "hold", continuation ? "student" : "policy"};
    scenario_snapshot *saved = scenario_capture(m, architect, *windows);
    uint64_t saved_hash = scenario_state_hash(m, architect, *windows);
    int alias = continuation && rollout_weights_equal(architect, continuation);
    int continuation_count = alias ? 1 : 2;
    nt_chuck_observation observation;
    nt_chuck_action source_action;
    float features[NT_CHUCK_ARCHITECT_FEATURES];
    if (architect->config.exploration != 0 ||
        nt_tape_chuck_observe(before, &observation) ||
        nt_chuck_architect_capture(architect, &observation, features) ||
        nt_chuck_architect_select(architect, &observation, &source_action))
        die("lived source capture/selection failed");
    size_t offsets[SCENARIO_HORIZON + SCENARIO_PROBES];
    uint64_t states[SCENARIO_HORIZON + SCENARIO_PROBES];
    offsets[0] = offset; states[0] = *windows;
    uint64_t lookahead = *windows;
    for (int i = 1; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) {
        offsets[i] = (size_t)(next_window(&lookahead) % (split - CTX)); states[i] = lookahead;
    }
    if (scenario_state_hash(m, architect, *windows) != saved_hash) die("lived source observation changed state");
    lived_save_source(m, architect, *windows, step, prefix);
    uint64_t student_policy_hash = 0, student_state_hash = 0;
    if (continuation) {
        nt_chuck_architect student = *architect;
        rollout_weights(&student, continuation);
        student_policy_hash = nt_chuck_architect_hash(&student);
        student_state_hash = scenario_state_hash(m, &student, *windows);
        char suffix[64], path[4096];
        snprintf(suffix, sizeof(suffix), ".fork-%d.student.policy.bin", step);
        path_for(path, sizeof(path), prefix, suffix);
        if (nt_chuck_architect_save(&student, path)) die("trajectory student source life save failed");
    }
    float future_before = lived_probe(m, data, offsets + SCENARIO_HORIZON, SCENARIO_PROBES);
    float heldout_before = lived_heldout(m, data, split, count);
    int ok = isfinite(future_before) && isfinite(heldout_before);
    char future_json[64], heldout_json[64];
    json_number(future_json, sizeof(future_json), future_before, 9);
    json_number(heldout_json, sizeof(heldout_json), heldout_before, 9);
    fprintf(trace, "{\"type\":\"fork\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"loss_before\":%.9g,"
            "\"future_probe_before\":%s,\"heldout_before\":%s,\"source_action\":%d,\"offsets\":[",
            step, saved_hash, before, future_json, heldout_json, source_action.kind);
    for (int i = 0; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) fprintf(trace, "%s%zu", i ? "," : "", offsets[i]);
    fprintf(trace, "],\"policy_hash\":\"%016" PRIx64 "\",\"observation\":{"
            "\"loss\":%.9g,\"loss_ema\":%.9g,\"loss_trend\":%.9g,\"macro_ema\":%.9g,"
            "\"best_macro\":%.9g,\"dampen\":%.9g,\"lr_scale\":%.9g,\"noise\":%.9g,"
            "\"grad_norm\":%.9g,\"grad_trend\":%.9g,\"frozen_fraction\":%.9g,"
            "\"step\":%d,\"stag\":%d,\"macro_stag\":%d,\"history_len\":%d},\"features\":[",
            nt_chuck_architect_hash(&saved->architect), observation.loss, observation.loss_ema, observation.loss_trend,
            observation.macro_ema, observation.best_macro, observation.dampen, observation.lr_scale,
            observation.noise, observation.grad_norm, observation.grad_trend, observation.frozen_fraction,
            observation.step, observation.stag, observation.macro_stag, observation.history_len);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i) fprintf(trace, "%s%.9g", i ? "," : "", features[i]);
    fputc(']', trace);
    if (continuation) {
        fprintf(trace, ",\"student_policy_hash\":\"%016" PRIx64 "\",\"student_state_hash\":\"%016" PRIx64 "\"",
                student_policy_hash, student_state_hash);
        trajectory_continuations(trace, alias);
    }
    fputs("}\n", trace);
    if (!ok) fprintf(trace, "{\"type\":\"branch_failure\",\"checkpoint\":%d,\"reason\":\"nonfinite_source_probe\",\"stopped\":true}\n", step);
    for (int ci = 0; ci < continuation_count && ok; ++ci) for (int ai = 0; ai < 3 && ok; ++ai) {
        scenario_restore(saved, m, architect, windows);
        if (scenario_state_hash(m, architect, *windows) != saved_hash) die("lived branch source identity differs");
        const nt_chuck_architect *weights = &saved->architect;
        if (continuation && ci == 1) {
            rollout_weights(architect, continuation); // NT_TRAJECTORY_STUDENT_WEIGHTS
            weights = continuation;
        }
        uint64_t branch_initial_hash = saved_hash, initial_policy_hash = nt_chuck_architect_hash(architect);
        if (continuation) {
            nt_chuck_architect non_weights;
            memcpy(&non_weights, architect, sizeof(non_weights));
            rollout_weights(&non_weights, &saved->architect);
            if (!rollout_weights_equal(architect, weights)) die("trajectory branch weights differ from named continuation");
            if (memcmp(&non_weights, &saved->architect, sizeof(non_weights))) die("trajectory weight graft changed acquired state");
            float captured[NT_CHUCK_ARCHITECT_FEATURES];
            if (nt_chuck_architect_capture(architect, &observation, captured) || memcmp(captured, features, sizeof(features)))
                die("trajectory weight graft changed source features");
            branch_initial_hash = scenario_state_hash(m, architect, *windows);
            if (ci == 1 && (initial_policy_hash != student_policy_hash || branch_initial_hash != student_state_hash))
                die("trajectory branch differs from saved student source");
        }
        float immediate_after = 0;
        uint64_t branch_hash = branch_initial_hash;
        for (int h = 1; h <= SCENARIO_HORIZON; ++h) {
            float loss = before, norm = gradient_norm;
            if (h > 1) {
                int loss_idx = body_forward(m, data, offsets[h - 1], 1);
                loss = read_loss_raw(loss_idx);
                if (!isfinite(loss)) {
                    lived_branch_failure(trace, step, continuations[ci], names[ai], h,
                                         "nonfinite_pre_action_loss", loss, 0, 0);
                    ok = 0; break;
                }
                nt_tape_backward(loss_idx); norm = nt_tape_clip_grads(1.0f);
                if (!isfinite(norm)) {
                    lived_branch_failure(trace, step, continuations[ci], names[ai], h,
                                         "nonfinite_gradient", loss, norm, 1);
                    ok = 0; break;
                }
                *windows = states[h - 1];
            }
            nt_chuck_state pre = nt_tape_get()->chuck;
            nt_chuck_architect_decision decision;
            nt_chuck_architect_receipt receipt;
            uint64_t policy_pre = nt_chuck_architect_hash(architect), pending_hash = 0;
            int forced = h == 1 || ci == 0; // NT_LIVED_CONTINUATION: parent acts after the first policy-branch intervention.
            if (continuation) forced = h == 1;
            lived_window window = {m, data, offsets[h - 1]};
            if (forced) {
                nt_chuck_action action = {h == 1 ? (nt_chuck_action_kind)(ai + NT_CHUCK_ACTION_HOLD) : NT_CHUCK_ACTION_HOLD, 0};
                if (nt_chuck_architect_intervene(architect, lr, loss, &action, lived_after, &window, &decision, &receipt))
                    die("lived intervention refused");
            } else {
                if (nt_chuck_architect_step(architect, lr, loss, &decision)) die("lived policy step refused");
                pending_hash = nt_chuck_architect_hash(architect);
                float after = lived_after(&window);
                if (nt_chuck_architect_feedback_frozen(architect, after, &receipt)) die("lived policy history refused");
            }
            float after = receipt.after_loss;
            nt_chuck_state post = nt_tape_get()->chuck;
            uint64_t policy_post = nt_chuck_architect_hash(architect);
            branch_hash = scenario_state_hash(m, architect, *windows); // Before separate probe evaluation changes the tape.
            if (!rollout_weights_equal(architect, weights) || architect->pending ||
                architect->decisions != saved->architect.decisions + (uint64_t)h ||
                architect->updates != saved->architect.updates + (uint64_t)h || receipt.learned)
                die("lived frozen history chronology failed");
            if (continuation && h == 1 && memcmp(decision.features, features, sizeof(features)))
                die("trajectory branch first features differ from source");
            if (h == 1) immediate_after = after;
            char after_json[64]; json_number(after_json, sizeof(after_json), after, 9);
            fprintf(trace, "{\"type\":\"branch_step\",\"checkpoint\":%d,\"continuation\":\"%s\",\"intervention\":\"%s\","
                    "\"update\":%d,\"forced\":%s,\"executed_action\":\"%s\",\"action\":%d,\"offset\":%zu,"
                    "\"before\":%.9g,\"after_same_window\":%s,\"gradient_norm\":%.9g,\"gradient_norm_stage\":\"before_clip\",\"window_rng\":\"%016" PRIx64 "\","
                    "\"noise_rng\":%" PRIu32 ",\"source_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\",\"pre_chuck\":",
                    step, continuations[ci], names[ai], h, forced ? "true" : "false", names[decision.action.kind - NT_CHUCK_ACTION_HOLD],
                    decision.action.kind, offsets[h - 1], loss, after_json, norm, *windows, nt_chuck_rng_get(), saved_hash, branch_hash);
            write_chuck(trace, &pre); fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
            fputs(",\"architect\":", trace);
            lived_write_architect(trace, &decision, &receipt, policy_pre, pending_hash, policy_post, forced);
            if (continuation)
                fprintf(trace, ",\"initial_policy_hash\":\"%016" PRIx64 "\",\"branch_initial_hash\":\"%016" PRIx64 "\"",
                        initial_policy_hash, branch_initial_hash);
            fputs("}\n", trace);
            if (ci == (continuation ? 0 : 1) && (nt_chuck_action_kind)(ai + NT_CHUCK_ACTION_HOLD) == source_action.kind && expected)
                expected[h - 1] = (lived_transition){step + h - 1, decision.action.kind, loss, after,
                                                   *windows, policy_pre, policy_post, branch_hash};
            if (!isfinite(after)) {
                lived_branch_failure(trace, step, continuations[ci], names[ai], h,
                                     "nonfinite_post_action_loss", loss, norm, 1);
                ok = 0; break;
            }
            if (scenario_horizon(h)) {
                float origin_after = read_loss_raw(body_forward(m, data, offsets[0], 0));
                float future_after = lived_probe(m, data, offsets + SCENARIO_HORIZON, SCENARIO_PROBES);
                float heldout_after = lived_heldout(m, data, split, count);
                ok = isfinite(origin_after) && isfinite(future_after) && isfinite(heldout_after);
                char origin_json[64], future_after_json[64], heldout_after_json[64], future_delta[64], heldout_delta[64];
                json_number(origin_json, sizeof(origin_json), origin_after, 9);
                json_number(future_after_json, sizeof(future_after_json), future_after, 9);
                json_number(heldout_after_json, sizeof(heldout_after_json), heldout_after, 9);
                json_number(future_delta, sizeof(future_delta), (double)future_before - future_after, 9);
                json_number(heldout_delta, sizeof(heldout_delta), (double)heldout_before - heldout_after, 9);
                fprintf(trace, "{\"type\":\"comparison\",\"checkpoint\":%d,\"continuation\":\"%s\",\"action\":\"%s\",\"horizon\":%d,"
                        "\"initial_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\",\"finite\":%s,"
                        "\"immediate_before\":%.9g,\"immediate_after\":%.9g,\"immediate_improvement\":%.9g,"
                        "\"origin_after_horizon\":%s,\"future_before\":%.9g,\"future_after\":%s,\"future_improvement\":%s,"
                        "\"heldout_before\":%.9g,\"heldout_after\":%s,\"heldout_improvement\":%s,"
                        "\"dampen\":%.9g,\"lr_scale\":%.9g,\"noise\":%.9g",
                        step, continuations[ci], names[ai], h, saved_hash, branch_hash, ok ? "true" : "false", before, // NT_LIVED_COMPARISON_ACTION
                        immediate_after, before - immediate_after, origin_json, future_before, future_after_json, future_delta,
                        heldout_before, heldout_after_json, heldout_delta, nt_tape_get()->chuck.dampen,
                        nt_tape_get()->chuck.lr_scale, nt_tape_get()->chuck.noise);
                if (continuation)
                    fprintf(trace, ",\"initial_policy_hash\":\"%016" PRIx64 "\",\"branch_initial_hash\":\"%016" PRIx64 "\"",
                            initial_policy_hash, branch_initial_hash);
                fputs("}\n", trace);
                if (!ok) {
                    lived_branch_failure(trace, step, continuations[ci], names[ai], h,
                                         "nonfinite_branch_probe", loss, norm, 1);
                    break;
                }
            }
        }
    }
    scenario_restore(saved, m, architect, windows); // NT_LIVED_HOST_RESTORE: retain the complete source world.
    if (scenario_state_hash(m, architect, *windows) != saved_hash) die("lived host restore identity differs");
    fprintf(trace, "{\"type\":\"restore\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"exact\":true}\n", step, saved_hash);
    if (fflush(trace)) die("lived diagnostic flush failed");
    scenario_release(saved);
    return ok;
}

static int lived_main(int argc, char **argv) {
    int trajectory = !strcmp(argv[1], "--trajectory");
    if (argc != (trajectory ? 12 : 11)) {
        if (trajectory)
            fprintf(stderr, "usage: %s --trajectory BODY TOKENS PREFIX STEPS SEED LR CONFIG SOURCE_LIFE PROBES(0|1) CONTINUATION_LIFE\n", argv[0]);
        else
            fprintf(stderr, "usage: %s --lived BODY TOKENS PREFIX STEPS SEED LR CONFIG PARENT_LIFE PROBES(0|1)\n", argv[0]);
        return 2;
    }
    if (nt_get_gpu_mode()) die("lived protocol requires CPU mode");
    const char *prefix = argv[4];
    int steps = (int)parse_integer(argv[5], 1, 1000000), probes = (int)parse_integer(argv[10], 0, 1);
    unsigned seed = (unsigned)parse_integer(argv[6], 1, UINT_MAX);
    char *end = NULL; errno = 0;
    float lr = strtof(argv[7], &end);
    if (errno || end == argv[7] || *end || !isfinite(lr) || lr <= 0 || lr > 1) die("invalid lived learning rate");
    nt_chuck_architect_config config; read_config(&config, argv[8]);
    if (config.mode != NT_CHUCK_ARCHITECT_LEARNED) die("lived requires learned configuration");
    config.seed = seed; config.exploration = 0;
    nt_chuck_architect architect, source, continuation;
    if (nt_chuck_architect_init(&architect, &config) || nt_chuck_architect_load(&source, argv[9]) ||
        source.config.mode != NT_CHUCK_ARCHITECT_LEARNED || source.pending) die("lived parent initialization failed");
    uint64_t source_hash = nt_chuck_architect_hash(&source);
    if (trajectory && (nt_chuck_architect_load(&continuation, argv[11]) ||
        continuation.config.mode != NT_CHUCK_ARCHITECT_LEARNED || continuation.pending))
        die("trajectory continuation initialization failed");
    rollout_weights(&architect, &source);
    const nt_chuck_architect initial = architect;
    nt_seed(seed);
    if (nt_chuck_rng_set(UINT32_C(2463534242))) die("lived initial noise RNG failed");
    body m; body_init(&m, argv[2]);
    size_t count; uint32_t *data = read_tokens(argv[3], &count, m.vocab);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) die("lived corpus split too small");
    uint64_t windows = seed;
    char path[4096]; path_for(path, sizeof(path), prefix, ".initial.bin");
    if (nt_save(path, m.param, m.count)) die("lived initial body save failed");
    path_for(path, sizeof(path), prefix, ".policy.initial.bin");
    if (nt_chuck_architect_save(&architect, path)) die("lived initial policy save failed");
    path_for(path, sizeof(path), prefix, ".jsonl");
    FILE *trace = fopen(path, "w"); if (!trace) die("lived trace open failed");
    fprintf(trace, "{\"type\":\"rollout_run\",\"body\":\"%s\",\"arm\":\"parent\",\"seed\":%u,\"steps\":%d,"
            "\"parameters\":%ld,\"lr\":%.9g,\"resume_step\":0,\"sealed_policy_hash\":\"%016" PRIx64 "\","
            "\"initial_policy_hash\":\"%016" PRIx64 "\",\"frozen_weights\":true,\"score_objective\":\"H16 within-state-span advantage\","
            "\"feedback_target\":\"same-window relative improvement for temporal history\",\"feedback_regression\":false,"
            "\"diagnostics\":\"%s\",\"probes\":%s", m.name, seed, steps, m.elements, lr, source_hash,
            nt_chuck_architect_hash(&architect), trajectory ? "trajectory" : "lived", probes ? "true" : "false");
    if (trajectory) {
        fprintf(trace, ",\"source_saved_policy_hash\":\"%016" PRIx64 "\",\"continuation_saved_policy_hash\":\"%016" PRIx64 "\"",
                source_hash, nt_chuck_architect_hash(&continuation));
        trajectory_continuations(trace, rollout_weights_equal(&source, &continuation));
    }
    fputs("}\n", trace);
    float initial_eval = evaluate(&m, data, split, count);
    fprintf(trace, "{\"type\":\"evaluation\",\"step\":0,\"heldout_loss\":%.9g}\n", initial_eval);
    double start = seconds();
    int action_count[4] = {0}, checkpoints = 0, continuation_checks = 0, completed = 0, failed = 0;
    lived_transition expected[SCENARIO_HORIZON] = {{0}};
    for (int step = 1; step <= steps; ++step) {
        size_t offset = (size_t)(next_window(&windows) % (split - CTX));
        int loss_idx = body_forward(&m, data, offset, 1);
        float before = read_loss(loss_idx); nt_tape_backward(loss_idx);
        float norm = nt_tape_clip_grads(1.0f);
        if (!isfinite(norm)) die("lived host nonfinite gradient");
        if (probes && (trajectory ? trajectory_checkpoint(step) : lived_checkpoint(step))) {
            if (!lived_diagnose(&m, data, count, split, &architect, &windows, offset, before, norm, lr,
                                step, prefix, trace, expected, trajectory ? &continuation : NULL)) {
                fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_diagnostic_branch\",\"stopped\":true}\n", step);
                failed = 1; break;
            }
            ++checkpoints;
        }
        nt_chuck_state pre = nt_tape_get()->chuck;
        nt_chuck_architect_decision decision;
        nt_chuck_architect_receipt receipt;
        uint64_t pre_hash = nt_chuck_architect_hash(&architect);
        if (nt_chuck_architect_step(&architect, lr, before, &decision)) die("lived host policy action refused");
        uint64_t pending_hash = nt_chuck_architect_hash(&architect);
        nt_chuck_state post = nt_tape_get()->chuck;
        float after = read_loss_raw(body_forward(&m, data, offset, 0));
        if (nt_chuck_architect_feedback_frozen(&architect, after, &receipt)) die("lived host feedback refused");
        if (!rollout_weights_equal(&architect, &initial) || architect.pending ||
            architect.decisions != (uint64_t)step || architect.updates != (uint64_t)step || receipt.learned)
            die("lived host frozen history chronology failed");
        ++action_count[decision.action.kind]; completed = step;
        uint64_t post_hash = nt_chuck_architect_hash(&architect), state_hash = scenario_state_hash(&m, &architect, windows);
        char after_json[64]; json_number(after_json, sizeof(after_json), after, 9);
        fprintf(trace, "{\"type\":\"rollout_step\",\"step\":%d,\"offset\":%zu,\"window_rng\":\"%016" PRIx64 "\","
                "\"before\":%.9g,\"after\":%s,\"gradient_norm\":%.9g,\"action\":%d,\"pre_chuck\":",
                step, offset, windows, before, after_json, norm, decision.action.kind);
        write_chuck(trace, &pre); fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
        fputs(",\"architect\":", trace); write_architect(trace, &decision, &receipt, pre_hash, pending_hash, post_hash, NULL);
        fprintf(trace, ",\"state_hash\":\"%016" PRIx64 "\"}\n", state_hash);
        if (!isfinite(after)) {
            fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_post_action_loss\",\"stopped\":true}\n", step);
            failed = 1; break;
        }
        for (int i = 0; i < SCENARIO_HORIZON; ++i) if (expected[i].step == step) {
            const lived_transition *e = &expected[i];
            int exact = e->action == decision.action.kind && e->windows == windows &&
                !memcmp(&e->before, &before, sizeof(float)) && !memcmp(&e->after, &after, sizeof(float)) &&
                e->policy_pre == pre_hash && e->policy_post == post_hash && e->state_hash == state_hash;
            fprintf(trace, "{\"type\":\"continuation_check\",\"checkpoint\":%d,\"step\":%d,\"update\":%d,"
                    "\"state_hash\":\"%016" PRIx64 "\",\"expected_hash\":\"%016" PRIx64 "\",\"exact\":%s}\n",
                    expected[0].step, step, i + 1, state_hash, e->state_hash, exact ? "true" : "false");
            if (!exact) die("lived selected continuation differs from host");
            ++continuation_checks;
        }
        if (step % 128 == 0 || step == steps) {
            float heldout = evaluate(&m, data, split, count);
            fprintf(trace, "{\"type\":\"evaluation\",\"step\":%d,\"heldout_loss\":%.9g}\n", step, heldout);
            printf("lived %s seed=%u step=%d/%d loss=%.6f elapsed=%.2fs\n", m.name, seed, step, steps, before, seconds() - start);
            fflush(stdout); fflush(trace);
        }
    }
    struct rusage usage; if (getrusage(RUSAGE_SELF, &usage)) die("lived getrusage failed");
    long rss = usage.ru_maxrss;
#ifdef __APPLE__
    rss /= 1024;
#endif
    fprintf(trace, "{\"type\":\"rollout_summary\",\"seconds\":%.9g,\"max_rss_kib\":%ld,\"steps\":%d,"
            "\"action_counts\":[%d,%d,%d,%d],\"frozen_weights_unchanged\":true,\"history_updates\":%" PRIu64 ","
            "\"policy_hash\":\"%016" PRIx64 "\",\"diagnostic_checkpoints\":%d,\"continuation_checks\":%d,\"failed\":%s}\n",
            seconds() - start, rss, completed, action_count[0], action_count[1], action_count[2], action_count[3],
            architect.updates, nt_chuck_architect_hash(&architect), checkpoints, continuation_checks, failed ? "true" : "false");
    if (ferror(trace) || fclose(trace)) die("lived trace close failed");
    rollout_save(prefix, &m, &architect, windows, completed, 1);
    nt_tape_destroy();
    for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
    free(data);
    return failed ? 3 : 0;
}
#endif
