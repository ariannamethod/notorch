/* Chuck acquires measured action credit while its training world continues.
 * Branches freeze the current life; only explicit comparison fits acquire
 * weights. Every branch restores the live body before fitting or acting. */
#ifndef NOTORCH_CHUCK_ARCHITECT_LIVING_H
#define NOTORCH_CHUCK_ARCHITECT_LIVING_H

#define LIVING_FITS 32

static int living_checkpoint(int step) {
    return step <= 449 && (step - 1) % 64 == 0;
}

static void living_floats(FILE *f, const float *values, int n) {
    fputc('[', f);
    for (int i = 0; i < n; ++i) {
        char number[64]; json_number(number, sizeof(number), values[i], 9);
        fprintf(f, "%s%s", i ? "," : "", number);
    }
    fputc(']', f);
}

static void living_observation(FILE *f, const nt_chuck_observation *o) {
    fprintf(f, "{\"loss\":%.9g,\"loss_ema\":%.9g,\"loss_trend\":%.9g,"
            "\"macro_ema\":%.9g,\"best_macro\":%.9g,\"dampen\":%.9g,"
            "\"lr_scale\":%.9g,\"noise\":%.9g,\"grad_norm\":%.9g,"
            "\"grad_trend\":%.9g,\"frozen_fraction\":%.9g,\"step\":%d,"
            "\"stag\":%d,\"macro_stag\":%d,\"history_len\":%d}",
            o->loss, o->loss_ema, o->loss_trend, o->macro_ema, o->best_macro,
            o->dampen, o->lr_scale, o->noise, o->grad_norm, o->grad_trend,
            o->frozen_fraction, o->step, o->stag, o->macro_stag, o->history_len);
}

static void living_save_policy(const nt_chuck_architect *a, const char *prefix,
                                int checkpoint, const char *phase) {
    char path[4096], suffix[96];
    if (checkpoint) snprintf(suffix, sizeof(suffix), ".acquire-%d.%s.policy.bin", checkpoint, phase);
    else snprintf(suffix, sizeof(suffix), ".policy.%s.bin", phase);
    path_for(path, sizeof(path), prefix, suffix);
    if (nt_chuck_architect_save(a, path)) die("living policy save failed");
}

static int living_acquire(body *m, const uint32_t *data, size_t count, size_t split,
        nt_chuck_architect *architect, uint64_t *windows, size_t offset,
        float before, float gradient_norm, float lr, int step, const char *arm,
        const char *prefix, FILE *trace, lived_transition expected[SCENARIO_HORIZON],
        int *branch_updates, int *fit_updates) {
    static const char *names[] = {"hold", "brake", "push"};
    int frozen = !strcmp(arm, "frozen"), rotated = !strcmp(arm, "rotated");
    scenario_snapshot *saved = scenario_capture(m, architect, *windows);
    uint64_t source_hash = scenario_state_hash(m, architect, *windows);
    uint64_t source_policy_hash = nt_chuck_architect_hash(architect);
    nt_chuck_observation observation;
    nt_chuck_action action_before;
    nt_chuck_architect_comparison sample;
    float scores_before[NT_CHUCK_ARCHITECT_HEADS];
    if (nt_tape_chuck_observe(before, &observation) ||
        nt_chuck_architect_capture(architect, &observation, sample.features) ||
        nt_chuck_architect_scores(architect, sample.features, scores_before) ||
        nt_chuck_architect_select(architect, &observation, &action_before))
        die("living source observation refused");
    size_t offsets[SCENARIO_HORIZON + SCENARIO_PROBES];
    uint64_t states[SCENARIO_HORIZON + SCENARIO_PROBES];
    offsets[0] = offset; states[0] = *windows;
    uint64_t lookahead = *windows;
    for (int i = 1; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) {
        offsets[i] = (size_t)(next_window(&lookahead) % (split - CTX));
        states[i] = lookahead;
    }
    if (!scenario_matches(saved, m, architect, *windows) ||
        scenario_state_hash(m, architect, *windows) != source_hash)
        die("living source observation changed world");
    living_save_policy(architect, prefix, step, "pre");
    fprintf(trace, "{\"type\":\"acquisition\",\"checkpoint\":%d,\"arm\":\"%s\","
            "\"source_hash\":\"%016" PRIx64 "\",\"policy_hash\":\"%016" PRIx64 "\","
            "\"before\":%.9g,\"gradient_norm\":%.9g,\"action_before\":%d,\"observation\":",
            step, arm, source_hash, source_policy_hash, before, gradient_norm, action_before.kind);
    living_observation(trace, &observation);
    fputs(",\"features\":", trace); living_floats(trace, sample.features, NT_CHUCK_ARCHITECT_FEATURES);
    fputs(",\"scores_before\":", trace); living_floats(trace, scores_before, NT_CHUCK_ARCHITECT_HEADS);
    fputs(",\"offsets\":[", trace);
    for (int i = 0; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i)
        fprintf(trace, "%s%zu", i ? "," : "", offsets[i]);
    fputs("]}\n", trace);

    float measured[NT_CHUCK_ARCHITECT_HEADS] = {NAN, NAN, NAN};
    int ok = 1;
    for (int ai = 0; ai < NT_CHUCK_ARCHITECT_HEADS && ok; ++ai) {
        scenario_restore(saved, m, architect, windows);
        if (scenario_state_hash(m, architect, *windows) != source_hash)
            die("living branch source identity differs");
        uint64_t branch_hash = source_hash;
        float immediate_after = NAN;
        for (int h = 1; h <= SCENARIO_HORIZON; ++h) {
            float loss = before, norm = gradient_norm;
            if (h > 1) {
                int index = body_forward(m, data, offsets[h - 1], 1);
                loss = read_loss_raw(index);
                if (!isfinite(loss)) {
                    lived_branch_failure(trace, step, "current", names[ai], h,
                                         "nonfinite_pre_action_loss", loss, 0, 0);
                    ok = 0; break;
                }
                nt_tape_backward(index); norm = nt_tape_clip_grads(1.0f);
                if (!isfinite(norm)) {
                    lived_branch_failure(trace, step, "current", names[ai], h,
                                         "nonfinite_gradient", loss, norm, 1);
                    ok = 0; break;
                }
                *windows = states[h - 1];
            }
            nt_chuck_state pre = nt_tape_get()->chuck;
            nt_chuck_architect_decision decision;
            nt_chuck_architect_receipt receipt;
            uint64_t policy_pre = nt_chuck_architect_hash(architect), pending_hash = 0;
            lived_window window = {m, data, offsets[h - 1]};
            if (h == 1) {
                nt_chuck_action action = {(nt_chuck_action_kind)(ai + NT_CHUCK_ACTION_HOLD), 0};
                if (nt_chuck_architect_intervene(architect, lr, loss, &action,
                        lived_after, &window, &decision, &receipt))
                    die("living branch intervention refused");
            } else {
                if (nt_chuck_architect_step(architect, lr, loss, &decision))
                    die("living branch policy action refused");
                pending_hash = nt_chuck_architect_hash(architect);
                if (nt_chuck_architect_feedback_frozen(architect, lived_after(&window), &receipt))
                    die("living branch feedback refused");
            }
            ++*branch_updates;
            nt_chuck_state post = nt_tape_get()->chuck;
            uint64_t policy_post = nt_chuck_architect_hash(architect);
            branch_hash = scenario_state_hash(m, architect, *windows);
            if (!rollout_weights_equal(architect, &saved->architect) || architect->pending ||
                architect->decisions != saved->architect.decisions + (uint64_t)h ||
                architect->updates != saved->architect.updates + (uint64_t)h || receipt.learned)
                die("living branch frozen chronology failed");
            if (h == 1 && memcmp(decision.features, sample.features, sizeof(sample.features)))
                die("living branch first features differ from source");
            if (h == 1) immediate_after = receipt.after_loss;
            char after_json[64]; json_number(after_json, sizeof(after_json), receipt.after_loss, 9);
            fprintf(trace, "{\"type\":\"branch_step\",\"checkpoint\":%d,\"continuation\":\"current\","
                    "\"intervention\":\"%s\",\"update\":%d,\"forced\":%s,\"action\":%d,"
                    "\"offset\":%zu,\"before\":%.9g,\"after_same_window\":%s,\"gradient_norm\":%.9g,"
                    "\"gradient_norm_stage\":\"before_clip\",\"window_rng\":\"%016" PRIx64 "\","
                    "\"noise_rng\":%" PRIu32 ",\"source_hash\":\"%016" PRIx64 "\","
                    "\"state_hash\":\"%016" PRIx64 "\",\"pre_chuck\":",
                    step, names[ai], h, h == 1 ? "true" : "false", decision.action.kind,
                    offsets[h - 1], loss, after_json, norm, *windows, nt_chuck_rng_get(), source_hash, branch_hash);
            write_chuck(trace, &pre); fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
            fputs(",\"architect\":", trace);
            lived_write_architect(trace, &decision, &receipt, policy_pre, pending_hash, policy_post, h == 1);
            fputs("}\n", trace);
            if (frozen && (nt_chuck_action_kind)(ai + NT_CHUCK_ACTION_HOLD) == action_before.kind)
                expected[h - 1] = (lived_transition){step + h - 1, decision.action.kind, loss,
                    receipt.after_loss, *windows, policy_pre, policy_post, branch_hash};
            if (!isfinite(receipt.after_loss)) {
                lived_branch_failure(trace, step, "current", names[ai], h,
                                     "nonfinite_post_action_loss", receipt.after_loss, norm, 1);
                ok = 0; break;
            }
        }
        if (ok) {
            measured[ai] = lived_probe(m, data, offsets + SCENARIO_HORIZON, SCENARIO_PROBES);
            float heldout = lived_heldout(m, data, split, count);
            ok = isfinite(measured[ai]) && isfinite(heldout);
            char future_json[64], heldout_json[64];
            json_number(future_json, sizeof(future_json), measured[ai], 9);
            json_number(heldout_json, sizeof(heldout_json), heldout, 9);
            fprintf(trace, "{\"type\":\"comparison\",\"checkpoint\":%d,\"action\":\"%s\","
                    "\"action_kind\":%d,\"horizon\":%d,\"source_hash\":\"%016" PRIx64 "\","
                    "\"policy_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\","
                    "\"immediate_after\":%.9g,\"future_after\":%s,\"heldout_after\":%s,\"finite\":%s}\n",
                    step, names[ai], ai + NT_CHUCK_ACTION_HOLD, SCENARIO_HORIZON,
                    source_hash, source_policy_hash, branch_hash, immediate_after,
                    future_json, heldout_json, ok ? "true" : "false");
            if (!ok) lived_branch_failure(trace, step, "current", names[ai], SCENARIO_HORIZON,
                                          "nonfinite_branch_probe", measured[ai], 0, 0);
        }
    }

    scenario_restore(saved, m, architect, windows); // NT_LIVING_RESTORE: measured worlds never replace the live one.
    uint64_t restored_hash = scenario_state_hash(m, architect, *windows);
    if (restored_hash != source_hash) die("living source restore identity differs");
    fprintf(trace, "{\"type\":\"restore\",\"checkpoint\":%d,\"source_hash\":\"%016" PRIx64 "\","
            "\"restored_hash\":\"%016" PRIx64 "\",\"exact\":true}\n", step, source_hash, restored_hash);
    if (!ok) {
        if (fflush(trace)) die("living failed acquisition flush failed");
        scenario_release(saved);
        return 0;
    }
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k)
        sample.future_loss[k] = measured[rotated ? (k + 2) % NT_CHUCK_ARCHITECT_HEADS : k]; // NT_LIVING_ASSIGNED_OUTCOMES
    int fits = frozen ? 0 : LIVING_FITS;
    for (int iteration = 1; iteration <= fits; ++iteration) {
        nt_chuck_architect_conditioned_receipt fitted;
        if (nt_chuck_architect_fit_conditioned(architect, &sample, &fitted))
            die("living conditioned fit refused");
        ++*fit_updates;
        const nt_chuck_architect_comparison_receipt *r = &fitted.comparison;
        fprintf(trace, "{\"type\":\"fit\",\"checkpoint\":%d,\"iteration\":%d,\"fit_step\":%d,"
                "\"source_hash\":\"%016" PRIx64 "\",\"hash_before\":\"%016" PRIx64 "\","
                "\"hash_after\":\"%016" PRIx64 "\",\"measured_future_loss\":",
                step, iteration, *fit_updates, source_hash, r->hash_before, r->hash_after);
        living_floats(trace, measured, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"assigned_future_loss\":", trace); living_floats(trace, r->future_loss, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"target\":", trace); living_floats(trace, r->target, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"predicted_before\":", trace); living_floats(trace, r->predicted_before, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"predicted_after\":", trace); living_floats(trace, r->predicted_after, NT_CHUCK_ARCHITECT_HEADS);
        fputs(",\"error_before\":", trace); living_floats(trace, r->error_before, NT_CHUCK_ARCHITECT_HEADS);
        fprintf(trace, ",\"scale\":%.17g,\"huber_before\":%.9g,\"huber_after\":%.9g,"
                "\"online_decisions\":%" PRIu64 ",\"online_updates\":%" PRIu64 "}\n",
                fitted.scale, r->huber_before, r->huber_after, r->decisions, r->updates);
    }
    nt_chuck_architect non_weights;
    memcpy(&non_weights, architect, sizeof(non_weights));
    rollout_weights(&non_weights, &saved->architect);
    if (memcmp(&non_weights, &saved->architect, sizeof(non_weights)) ||
        !scenario_matches(saved, m, &non_weights, *windows) ||
        scenario_state_hash(m, &non_weights, *windows) != source_hash)
        die("living fit changed source world or non-weight life state");
    nt_chuck_observation after_observation;
    nt_chuck_action action_after;
    float after_features[NT_CHUCK_ARCHITECT_FEATURES], scores_after[NT_CHUCK_ARCHITECT_HEADS];
    if (nt_tape_chuck_observe(before, &after_observation) ||
        memcmp(&observation, &after_observation, sizeof(observation)) ||
        nt_chuck_architect_capture(architect, &after_observation, after_features) ||
        memcmp(after_features, sample.features, sizeof(after_features)) ||
        nt_chuck_architect_scores(architect, after_features, scores_after) ||
        nt_chuck_architect_select(architect, &after_observation, &action_after))
        die("living fit changed observation or feature capture");
    living_save_policy(architect, prefix, step, "post");
    fprintf(trace, "{\"type\":\"acquisition_result\",\"checkpoint\":%d,\"arm\":\"%s\","
            "\"source_hash\":\"%016" PRIx64 "\",\"restored_hash\":\"%016" PRIx64 "\","
            "\"post_fit_state_hash\":\"%016" PRIx64 "\",\"policy_before\":\"%016" PRIx64 "\","
            "\"policy_after\":\"%016" PRIx64 "\",\"fit_iterations\":%d,\"fit_updates\":%d,"
            "\"assignment\":\"%s\",\"non_weight_fields_unchanged\":true,\"source_world_unchanged\":true,"
            "\"action_before\":%d,\"action_after\":%d,\"measured_future_loss\":",
            step, arm, source_hash, restored_hash, scenario_state_hash(m, architect, *windows),
            source_policy_hash, nt_chuck_architect_hash(architect), fits, *fit_updates,
            rotated ? "rotated-PHB" : "measured-HBP", action_before.kind, action_after.kind);
    living_floats(trace, measured, NT_CHUCK_ARCHITECT_HEADS);
    fputs(",\"assigned_future_loss\":", trace); living_floats(trace, sample.future_loss, NT_CHUCK_ARCHITECT_HEADS);
    fputs(",\"features\":", trace); living_floats(trace, sample.features, NT_CHUCK_ARCHITECT_FEATURES);
    fputs(",\"scores_before\":", trace); living_floats(trace, scores_before, NT_CHUCK_ARCHITECT_HEADS);
    fputs(",\"scores_after\":", trace); living_floats(trace, scores_after, NT_CHUCK_ARCHITECT_HEADS);
    fputs("}\n", trace);
    if (fflush(trace)) die("living acquisition flush failed");
    scenario_release(saved);
    return 1;
}

static int living_main(int argc, char **argv) {
    if (argc != 13) {
        fprintf(stderr, "usage: %s --living BODY TOKENS PREFIX STEPS SEED LR CONFIG ARM LIFE RESUME_STEP PROBES(0|1)\n", argv[0]);
        return 2;
    }
    if (nt_get_gpu_mode()) die("living protocol requires CPU mode");
    const char *prefix = argv[4], *arm = argv[9];
    int canonical = !strcmp(arm, "canonical"), frozen = !strcmp(arm, "frozen");
    int living = !strcmp(arm, "living"), rotated = !strcmp(arm, "rotated");
    if (!canonical && !frozen && !living && !rotated) die("unknown living arm");
    int learned = !canonical;
    int steps = (int)parse_integer(argv[5], 1, 512);
    unsigned seed = (unsigned)parse_integer(argv[6], 1, UINT_MAX);
    int resume_step = (int)parse_integer(argv[11], 0, steps - 1);
    int probes = (int)parse_integer(argv[12], 0, 1);
    if ((canonical && (probes || resume_step)) || ((living || rotated) && !probes))
        die("living arm conflicts with probe or resume mode");
    char *end = NULL; errno = 0;
    float lr = strtof(argv[7], &end);
    if (errno || end == argv[7] || *end || !isfinite(lr) || lr <= 0 || lr > 1)
        die("invalid living body learning rate");
    nt_chuck_architect_config config; read_config(&config, argv[8]);
    uint32_t required = NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD) |
        NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_BRAKE) | NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_PUSH);
    if (config.mode != NT_CHUCK_ARCHITECT_LEARNED || config.learning_rate != 0.03f ||
        (config.limits.enabled_actions & required) != required)
        die("living requires learned H/B/P configuration with policy learning rate .03");
    config.seed = seed; config.exploration = 0;
    nt_chuck_architect architect, source;
    if (nt_chuck_architect_init(&architect, &config)) die("living host life initialization refused");
    uint64_t sealed_hash = 0;
    if (learned || strcmp(argv[10], "-")) {
        if (nt_chuck_architect_load(&source, argv[10]) ||
            source.config.mode != NT_CHUCK_ARCHITECT_LEARNED || source.pending)
            die("living initial life refused");
        sealed_hash = nt_chuck_architect_hash(&source);
        rollout_weights(&architect, &source);
    }
    const nt_chuck_architect initial = architect;
    nt_seed(seed);
    if (nt_chuck_rng_set(UINT32_C(2463534242))) die("living initial noise RNG failed");
    body m; body_init(&m, argv[2]);
    size_t count; uint32_t *data = read_tokens(argv[3], &count, m.vocab);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) die("living corpus split too small");
    uint64_t windows = seed;
    living_save_policy(&architect, prefix, 0, "initial");
    char path[4096]; path_for(path, sizeof(path), prefix, ".jsonl");
    FILE *trace = fopen(path, "w"); if (!trace) die("living trace open failed");
#ifdef NT_CHUCK_ARCHITECT_TEST
    const char *fixture = "true";
#else
    const char *fixture = "false";
#endif
    fprintf(trace, "{\"type\":\"living_run\",\"body\":\"%s\",\"arm\":\"%s\",\"seed\":%u,"
            "\"steps\":%d,\"parameters\":%ld,\"context\":%d,\"dtype\":\"F32\",\"fixture\":%s,"
            "\"lr\":%.9g,\"policy_learning_rate\":%.9g,\"resume_step\":%d,\"probes\":%s,"
            "\"sealed_policy_hash\":\"%016" PRIx64 "\",\"initial_policy_hash\":\"%016" PRIx64 "\","
            "\"fits_per_acquisition\":%d,\"horizon\":%d,\"feedback_regression\":false,"
            "\"score_objective\":\"H16 within-state-span advantage\"}\n",
            m.name, arm, seed, steps, m.elements, CTX, fixture, lr, config.learning_rate,
            resume_step, probes ? "true" : "false", sealed_hash, nt_chuck_architect_hash(&architect),
            living || rotated ? LIVING_FITS : 0, SCENARIO_HORIZON);
    float initial_eval = evaluate(&m, data, split, count);
    fprintf(trace, "{\"type\":\"evaluation\",\"step\":0,\"heldout_loss\":%.9g}\n", initial_eval);
    double start = seconds();
    int action_count[4] = {0}, branch_updates = 0, fit_updates = 0, checkpoints = 0;
    int continuation_checks = 0, completed = 0, failed = 0, brake_changed = 0, brake_saturated = 0;
    lived_transition expected[SCENARIO_HORIZON] = {{0}};
    for (int step = 1; step <= steps; ++step) {
        size_t offset = (size_t)(next_window(&windows) % (split - CTX));
        int loss_index = body_forward(&m, data, offset, 1);
        float before = read_loss_raw(loss_index);
        if (!isfinite(before)) {
            fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_pre_action_loss\",\"stopped\":true}\n", step);
            failed = 1; break;
        }
        nt_tape_backward(loss_index); float norm = nt_tape_clip_grads(1.0f);
        if (!isfinite(norm)) {
            fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_gradient\",\"stopped\":true}\n", step);
            failed = 1; break;
        }
        if (probes && living_checkpoint(step)) {
            if (!living_acquire(&m, data, count, split, &architect, &windows, offset, before, norm, lr,
                    step, arm, prefix, trace, expected, &branch_updates, &fit_updates)) {
                fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"failed_acquisition\",\"stopped\":true}\n", step);
                failed = 1; break;
            }
            ++checkpoints;
        }
        nt_chuck_state pre = nt_tape_get()->chuck;
        nt_chuck_architect_decision decision = {0};
        nt_chuck_architect_receipt receipt = {0};
        nt_chuck_architect pre_life = architect;
        uint64_t policy_pre = nt_chuck_architect_hash(&architect), pending_hash = 0;
        if (learned) {
            if (nt_chuck_architect_step(&architect, lr, before, &decision)) die("living host action refused");
            pending_hash = nt_chuck_architect_hash(&architect);
        } else nt_tape_chuck_step(lr, before);
        nt_chuck_state post = nt_tape_get()->chuck;
        float after = read_loss_raw(body_forward(&m, data, offset, 0));
        if (learned) {
            if (nt_chuck_architect_feedback_frozen(&architect, after, &receipt)) die("living host feedback refused");
            if (!rollout_weights_equal(&architect, &pre_life) || architect.pending ||
                architect.decisions != (uint64_t)step || architect.updates != (uint64_t)step || receipt.learned)
                die("living host frozen feedback chronology failed");
        }
        if (frozen && !rollout_weights_equal(&architect, &initial)) die("living frozen arm changed weights");
        if (decision.action.kind < NT_CHUCK_ACTION_LEGACY || decision.action.kind > NT_CHUCK_ACTION_PUSH)
            die("living host action outside protocol");
        ++action_count[decision.action.kind]; completed = step;
        if (decision.action.kind == NT_CHUCK_ACTION_BRAKE) {
            float before_dampen = pre.initialized ? pre.dampen : 1.0f;
            if (post.dampen < before_dampen) ++brake_changed; else ++brake_saturated;
        }
        uint64_t policy_post = nt_chuck_architect_hash(&architect);
        uint64_t state_hash = scenario_state_hash(&m, &architect, windows);
        char after_json[64]; json_number(after_json, sizeof(after_json), after, 9);
        fprintf(trace, "{\"type\":\"rollout_step\",\"step\":%d,\"offset\":%zu,\"window_rng\":\"%016" PRIx64 "\","
                "\"before\":%.9g,\"after\":%s,\"gradient_norm\":%.9g,\"action\":%d,\"pre_chuck\":",
                step, offset, windows, before, after_json, norm, decision.action.kind);
        write_chuck(trace, &pre); fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
        if (learned) {
            fputs(",\"architect\":", trace);
            write_architect(trace, &decision, &receipt, policy_pre, pending_hash, policy_post, NULL);
        }
        fprintf(trace, ",\"state_hash\":\"%016" PRIx64 "\"}\n", state_hash);
        if (!isfinite(after)) {
            fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_post_action_loss\",\"stopped\":true}\n", step);
            failed = 1; break;
        }
        if (frozen) for (int i = 0; i < SCENARIO_HORIZON; ++i) if (expected[i].step == step) {
            const lived_transition *e = &expected[i];
            int exact = e->action == decision.action.kind && e->windows == windows &&
                !memcmp(&e->before, &before, sizeof(float)) && !memcmp(&e->after, &after, sizeof(float)) &&
                e->policy_pre == policy_pre && e->policy_post == policy_post && e->state_hash == state_hash;
            fprintf(trace, "{\"type\":\"continuation_check\",\"checkpoint\":%d,\"step\":%d,\"update\":%d,"
                    "\"state_hash\":\"%016" PRIx64 "\",\"expected_hash\":\"%016" PRIx64 "\",\"exact\":%s}\n",
                    expected[0].step, step, i + 1, state_hash, e->state_hash, exact ? "true" : "false");
            if (!exact) die("living frozen continuation differs from host");
            ++continuation_checks;
        }
        if (step == resume_step) {
            living_save_policy(&architect, prefix, 0, "resume");
            path_for(path, sizeof(path), prefix, ".policy.resume.bin");
            memset(&architect, 0, sizeof(architect));
            if (nt_chuck_architect_load(&architect, path)) die("living policy resume refused");
            if (scenario_state_hash(&m, &architect, windows) != state_hash)
                die("living policy resume changed world");
            fprintf(trace, "{\"type\":\"policy_resume\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"exact\":true}\n", step, state_hash);
        }
        if (step % 128 == 0 || step == steps) {
            float heldout = evaluate(&m, data, split, count);
            fprintf(trace, "{\"type\":\"evaluation\",\"step\":%d,\"heldout_loss\":%.9g}\n", step, heldout);
            printf("living %s %s seed=%u step=%d/%d loss=%.6f elapsed=%.2fs\n",
                   m.name, arm, seed, step, steps, before, seconds() - start);
            fflush(stdout); if (fflush(trace)) die("living trace flush failed");
        }
    }
    struct rusage usage; if (getrusage(RUSAGE_SELF, &usage)) die("living getrusage failed");
    long rss = usage.ru_maxrss;
#ifdef __APPLE__
    rss /= 1024;
#endif
    living_save_policy(&architect, prefix, 0, "final");
    fprintf(trace, "{\"type\":\"living_summary\",\"seconds\":%.9g,\"max_rss_kib\":%ld,\"steps\":%d,"
            "\"action_counts\":[%d,%d,%d,%d],\"brake_changed_dampen\":%d,\"brake_at_bound\":%d,"
            "\"acquisitions\":%d,\"branch_updates\":%d,\"fit_updates\":%d,\"history_updates\":%" PRIu64 ","
            "\"policy_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\","
            "\"continuation_checks\":%d,\"failed\":%s}\n",
            seconds() - start, rss, completed, action_count[0], action_count[1], action_count[2], action_count[3],
            brake_changed, brake_saturated, checkpoints, branch_updates, fit_updates, architect.updates,
            nt_chuck_architect_hash(&architect), scenario_state_hash(&m, &architect, windows),
            continuation_checks, failed ? "true" : "false");
    if (ferror(trace) || fclose(trace)) die("living trace close failed");
#ifdef NT_CHUCK_ARCHITECT_TEST
    /* The tiny boundary gate compares final tensors byte for byte. Production
     * retains complete state identities and small policy lives, without dumps. */
    rollout_save(prefix, &m, &architect, windows, completed, learned);
#endif
    nt_tape_destroy();
    for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
    free(data);
    return failed ? 3 : 0;
}
#endif
