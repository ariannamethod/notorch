/* Closed-loop CPU deployment of a sealed Chuck policy. The body supplies each
 * observation; the selected action changes its next state. Frozen feedback
 * acquires temporal history while preserving every fitted weight. */
#ifndef NOTORCH_CHUCK_ARCHITECT_ROLLOUT_H
#define NOTORCH_CHUCK_ARCHITECT_ROLLOUT_H

static void rollout_weights(nt_chuck_architect *dst, const nt_chuck_architect *src) {
    memcpy(dst->w1, src->w1, sizeof(dst->w1));
    memcpy(dst->b1, src->b1, sizeof(dst->b1));
    memcpy(dst->w2, src->w2, sizeof(dst->w2));
    memcpy(dst->b2, src->b2, sizeof(dst->b2));
}

static int rollout_weights_equal(const nt_chuck_architect *a, const nt_chuck_architect *b) {
    return !memcmp(a->w1, b->w1, sizeof(a->w1)) && !memcmp(a->b1, b->b1, sizeof(a->b1)) &&
           !memcmp(a->w2, b->w2, sizeof(a->w2)) && !memcmp(a->b2, b->b2, sizeof(a->b2));
}

static void rollout_save(const char *prefix, body *m, nt_chuck_architect *a,
                         uint64_t windows, int steps, int learned) {
    char path[4096];
    path_for(path, sizeof(path), prefix, ".final.bin");
    if (nt_save(path, m->param, m->count)) die("rollout final body save failed");
    save_optimizer(prefix, windows, steps);
    if (learned) {
        path_for(path, sizeof(path), prefix, ".policy.final.bin");
        if (nt_chuck_architect_save(a, path)) die("rollout final life save failed");
    }
}

static int rollout_main(int argc, char **argv) {
    if (argc != 12) {
        fprintf(stderr, "usage: %s --rollout BODY TOKENS PREFIX STEPS SEED LR CONFIG ARM LIFE RESUME_STEP\n", argv[0]);
        return 2;
    }
    if (nt_get_gpu_mode()) die("rollout protocol requires CPU mode");
    const char *prefix = argv[4], *arm = argv[9];
    int canonical = !strcmp(arm, "canonical");
    int forced = !strcmp(arm, "hold") ? NT_CHUCK_ACTION_HOLD :
                 !strcmp(arm, "push") ? NT_CHUCK_ACTION_PUSH : -1;
    int learned = !strcmp(arm, "oldfuture-simple") || !strcmp(arm, "conditional-simple") ||
                  !strcmp(arm, "conditional-adapted") || !strcmp(arm, "parent") ||
                  !strcmp(arm, "lived-hold") || !strcmp(arm, "lived-policy") ||
                  !strcmp(arm, "student") || !strcmp(arm, "refit-parent") || !strcmp(arm, "refit-self");
    if (!canonical && forced < 0 && !learned) die("unknown rollout arm");
    if (!learned && strcmp(argv[10], "-")) die("fixed rollout arm must use LIFE=-");
    int steps = (int)parse_integer(argv[5], 1, 1000000);
    unsigned seed = (unsigned)parse_integer(argv[6], 1, UINT_MAX);
    int resume_step = (int)parse_integer(argv[11], 0, steps - 1);
    if (resume_step && !learned) die("policy continuation needs a learned rollout arm");
    char *end = NULL; errno = 0;
    float lr = strtof(argv[7], &end);
    if (errno || end == argv[7] || *end || !isfinite(lr) || lr <= 0 || lr > 1)
        die("invalid rollout learning rate");
    nt_chuck_architect_config config;
    read_config(&config, argv[8]);
    if (config.mode != NT_CHUCK_ARCHITECT_LEARNED) die("rollout requires learned configuration");
    config.seed = seed; config.exploration = 0;
    nt_chuck_architect architect;
    if (nt_chuck_architect_init(&architect, &config)) die("rollout host life init failed");
    uint64_t source_hash = 0;
    if (learned) {
        nt_chuck_architect source;
        if (nt_chuck_architect_load(&source, argv[10]) || source.config.mode != NT_CHUCK_ARCHITECT_LEARNED ||
            source.pending) die("rollout sealed life load failed");
        source_hash = nt_chuck_architect_hash(&source);
        rollout_weights(&architect, &source);
    }
    const nt_chuck_architect initial = architect;
    nt_seed(seed);
    if (nt_chuck_rng_set(UINT32_C(2463534242))) die("rollout initial noise RNG failed");
    body m; body_init(&m, argv[2]);
    size_t count; uint32_t *data = read_tokens(argv[3], &count, m.vocab);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) die("rollout corpus split too small");
    uint64_t windows = seed;
    char path[4096];
    path_for(path, sizeof(path), prefix, ".initial.bin");
    if (nt_save(path, m.param, m.count)) die("rollout initial body save failed");
    if (learned) {
        path_for(path, sizeof(path), prefix, ".policy.initial.bin");
        if (nt_chuck_architect_save(&architect, path)) die("rollout initial life save failed");
    }
    path_for(path, sizeof(path), prefix, ".jsonl");
    FILE *trace = fopen(path, "w"); if (!trace) die("rollout trace open failed");
    fprintf(trace, "{\"type\":\"rollout_run\",\"body\":\"%s\",\"arm\":\"%s\",\"seed\":%u,\"steps\":%d,"
            "\"parameters\":%ld,\"lr\":%.9g,\"resume_step\":%d,\"sealed_policy_hash\":\"%016" PRIx64 "\","
            "\"initial_policy_hash\":\"%016" PRIx64 "\",\"frozen_weights\":%s,\"score_objective\":\"%s\","
            "\"feedback_target\":\"same-window relative improvement for temporal history\",\"feedback_regression\":false}\n",
            m.name, arm, seed, steps, m.elements, lr, resume_step, source_hash,
            learned ? nt_chuck_architect_hash(&architect) : 0, learned ? "true" : "false",
            !learned ? "none" : !strcmp(arm, "oldfuture-simple") ? "H16 HOLD-loss-relative advantage" :
            "H16 within-state-span advantage");
    float initial_eval = evaluate(&m, data, split, count);
    fprintf(trace, "{\"type\":\"evaluation\",\"step\":0,\"heldout_loss\":%.9g}\n", initial_eval);
    double start = seconds();
    int action_count[4] = {0};
    for (int step = 1; step <= steps; ++step) {
        size_t offset = (size_t)(next_window(&windows) % (split - CTX));
        int loss_idx = body_forward(&m, data, offset, 1);
        float before = read_loss(loss_idx); nt_tape_backward(loss_idx);
        float norm = nt_tape_clip_grads(1.0f);
        if (!isfinite(norm)) die("rollout nonfinite gradient");
        nt_chuck_state pre = nt_tape_get()->chuck;
        nt_chuck_architect_decision decision = {0};
        uint64_t pre_hash = 0, pending_hash = 0;
        if (learned) {
            pre_hash = nt_chuck_architect_hash(&architect);
            if (nt_chuck_architect_step(&architect, lr, before, &decision)) die("rollout selected action refused");
            pending_hash = nt_chuck_architect_hash(&architect);
        } else if (canonical) nt_tape_chuck_step(lr, before);
        else {
            decision.action.kind = (nt_chuck_action_kind)forced;
            if (nt_tape_chuck_step_action(lr, before, &decision.action, &config.limits)) die("rollout fixed action refused");
        }
        nt_chuck_state post = nt_tape_get()->chuck;
        float after = read_loss_raw(body_forward(&m, data, offset, 0));
        nt_chuck_architect_receipt receipt = {0};
        if (learned) {
            if (nt_chuck_architect_feedback_frozen(&architect, after, &receipt)) die("rollout frozen feedback refused");
            if (!rollout_weights_equal(&architect, &initial)) die("rollout feedback changed frozen weights");
            if (architect.decisions != (uint64_t)step || architect.updates != (uint64_t)step ||
                architect.pending || !architect.has_history || receipt.learned)
                die("rollout history chronology mismatch");
        }
        if (decision.action.kind < NT_CHUCK_ACTION_LEGACY || decision.action.kind > NT_CHUCK_ACTION_PUSH)
            die("rollout action outside protocol");
        ++action_count[decision.action.kind];
        char after_json[64]; json_number(after_json, sizeof(after_json), after, 9);
        fprintf(trace, "{\"type\":\"rollout_step\",\"step\":%d,\"offset\":%zu,\"window_rng\":\"%016" PRIx64 "\","
                "\"before\":%.9g,\"after\":%s,\"gradient_norm\":%.9g,\"action\":%d,\"pre_chuck\":",
                step, offset, windows, before, after_json, norm, decision.action.kind);
        write_chuck(trace, &pre); fputs(",\"post_chuck\":", trace); write_chuck(trace, &post);
        if (learned) {
            fputs(",\"architect\":", trace);
            write_architect(trace, &decision, &receipt, pre_hash, pending_hash, nt_chuck_architect_hash(&architect), NULL);
        }
        fprintf(trace, ",\"state_hash\":\"%016" PRIx64 "\"}\n", scenario_state_hash(&m, &architect, windows));
        if (!isfinite(after)) {
            fprintf(trace, "{\"type\":\"failure\",\"step\":%d,\"reason\":\"nonfinite_post_action_loss\",\"stopped\":true}\n", step);
            if (ferror(trace) || fclose(trace)) die("rollout failure trace write failed");
            rollout_save(prefix, &m, &architect, windows, step, learned);
            nt_tape_destroy();
            for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
            free(data); return 3;
        }
        if (step == resume_step) {
            uint64_t state_hash = scenario_state_hash(&m, &architect, windows);
            path_for(path, sizeof(path), prefix, ".policy.resume.bin");
            if (nt_chuck_architect_save(&architect, path)) die("rollout continuation save failed");
            memset(&architect, 0, sizeof(architect));
            if (nt_chuck_architect_load(&architect, path)) die("rollout continuation load failed");
            if (state_hash != scenario_state_hash(&m, &architect, windows)) die("rollout continuation state differs");
            fprintf(trace, "{\"type\":\"policy_resume\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"exact\":true}\n", step, state_hash);
        }
        if (step % 128 == 0 || step == steps) {
            float heldout = evaluate(&m, data, split, count);
            fprintf(trace, "{\"type\":\"evaluation\",\"step\":%d,\"heldout_loss\":%.9g}\n", step, heldout);
            printf("rollout %s %s seed=%u step=%d/%d loss=%.6f elapsed=%.2fs\n",
                   m.name, arm, seed, step, steps, before, seconds() - start);
            fflush(stdout); fflush(trace);
        }
    }
    struct rusage usage; if (getrusage(RUSAGE_SELF, &usage)) die("rollout getrusage failed");
    long rss = usage.ru_maxrss;
#ifdef __APPLE__
    rss /= 1024;
#endif
    fprintf(trace, "{\"type\":\"rollout_summary\",\"seconds\":%.9g,\"max_rss_kib\":%ld,"
            "\"steps\":%d,\"action_counts\":[%d,%d,%d,%d],\"frozen_weights_unchanged\":%s,"
            "\"history_updates\":%" PRIu64 ",\"policy_hash\":\"%016" PRIx64 "\"}\n",
            seconds() - start, rss, steps, action_count[0], action_count[1], action_count[2], action_count[3],
            learned ? "true" : "false", learned ? architect.updates : 0,
            learned ? nt_chuck_architect_hash(&architect) : 0);
    if (ferror(trace) || fclose(trace)) die("rollout trace write failed");
    rollout_save(prefix, &m, &architect, windows, steps, learned);
    nt_tape_destroy();
    for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
    free(data); return 0;
}
#endif
