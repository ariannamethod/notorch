/* CPU fork diagnostics for chuck_architect_train.c.
 * A fork restores the entire pending tape, then applies one action followed by
 * HOLD on identical windows. No branch consequence trains the host policy.
 */
#ifndef NOTORCH_CHUCK_ARCHITECT_SCENARIOS_H
#define NOTORCH_CHUCK_ARCHITECT_SCENARIOS_H

#define SCENARIO_HORIZON 16
#define SCENARIO_PROBES 4

typedef struct {
    nt_tensor *parameters[MAX_PARAMS];
    nt_tape *tape;
    nt_chuck_architect architect;
    uint64_t windows;
    uint32_t noise_rng;
    int training, count;
} scenario_snapshot;

static nt_tensor *scenario_clone(const nt_tensor *source) {
    nt_tensor *copy = source ? nt_tensor_clone(source) : NULL;
    if (source && !copy) die("scenario tensor clone failed");
    return copy;
}

static int scenario_tensor_equal(const nt_tensor *a, const nt_tensor *b) {
    if (!a || !b) return a == b;
    return a->len == b->len && a->ndim == b->ndim &&
        !memcmp(a->shape, b->shape, (size_t)a->ndim * sizeof(int)) &&
        !memcmp(a->stride, b->stride, (size_t)a->ndim * sizeof(int)) &&
        !memcmp(a->data, b->data, (size_t)a->len * sizeof(float));
}

static uint64_t scenario_bytes(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

static uint64_t scenario_word(uint64_t h, uint64_t word) {
    for (int i = 0; i < 8; ++i) { h = (h ^ (unsigned char)word) * UINT64_C(1099511628211); word >>= 8; }
    return h;
}

static uint64_t scenario_tensor_hash(uint64_t h, const nt_tensor *tensor) {
    h = scenario_word(h, tensor != NULL);
    if (!tensor) return h;
    h = scenario_word(h, tensor->len);
    h = scenario_word(h, tensor->ndim);
    for (int i = 0; i < tensor->ndim; ++i) {
        h = scenario_word(h, tensor->shape[i]);
        h = scenario_word(h, tensor->stride[i]);
    }
    return scenario_bytes(h, tensor->data, (size_t)tensor->len * sizeof(float));
}

/* Identities omit pointers/refcounts. Raw float bytes are labelled by platform
 * in the run manifest; on the measured x86_64 body they are little-endian F32. */
static uint64_t scenario_state_hash(const body *m, const nt_chuck_architect *a, uint64_t windows) {
    const nt_tape *t = nt_tape_get();
    uint64_t h = UINT64_C(14695981039346656037);
    h = scenario_word(h, windows);
    h = scenario_word(h, nt_chuck_rng_get());
    h = scenario_word(h, nt_is_training());
    h = scenario_word(h, nt_chuck_architect_hash(a));
    h = scenario_word(h, m->count);
    for (int i = 0; i < m->count; ++i) h = scenario_tensor_hash(h, m->param[i]);
    h = scenario_word(h, t->count); h = scenario_word(h, t->active); h = scenario_word(h, t->n_params);
    h = scenario_bytes(h, &t->chuck, sizeof(t->chuck));
    h = scenario_bytes(h, t->chuck_params, sizeof(t->chuck_params));
    for (int i = 0; i < t->n_params; ++i) {
        h = scenario_word(h, t->adam[i].t);
        h = scenario_tensor_hash(h, t->adam[i].m);
        h = scenario_tensor_hash(h, t->adam[i].v);
        h = scenario_tensor_hash(h, t->adam[i].acc_grad);
    }
    for (int i = 0; i < t->count; ++i) {
        const nt_tape_entry *e = &t->entries[i];
        h = scenario_word(h, e->op); h = scenario_word(h, e->parent1);
        h = scenario_word(h, e->parent2); h = scenario_word(h, e->parent3);
        h = scenario_bytes(h, &e->aux, sizeof(float)); h = scenario_bytes(h, &e->aux2, sizeof(float));
        h = scenario_bytes(h, &e->aux3, sizeof(float)); h = scenario_bytes(h, &e->aux4, sizeof(float));
        h = scenario_word(h, e->is_param); h = scenario_word(h, e->slot);
        h = scenario_word(h, e->no_decay); h = scenario_word(h, e->frozen);
        if (!e->is_param) h = scenario_tensor_hash(h, e->output);
        h = scenario_tensor_hash(h, e->grad);
    }
    return h;
}

static scenario_snapshot *scenario_capture(const body *m, const nt_chuck_architect *a, uint64_t windows) {
    if (nt_get_gpu_mode()) die("scenario snapshots require CPU mode");
    const nt_tape *live = nt_tape_get();
    if (live->n_params != m->count) die("scenario body/tape parameter mismatch");
    scenario_snapshot *s = calloc(1, sizeof(*s));
    if (!s || !(s->tape = malloc(sizeof(*s->tape)))) die("scenario snapshot allocation failed");
    s->count = m->count; s->windows = windows;
    s->noise_rng = nt_chuck_rng_get(); s->training = nt_is_training();
    memcpy(&s->architect, a, sizeof(*a));
    memcpy(s->tape, live, sizeof(*live));
    for (int i = 0; i < m->count; ++i) s->parameters[i] = scenario_clone(m->param[i]);
    for (int i = 0; i < live->count; ++i) {
        const nt_tape_entry *source = &live->entries[i];
        nt_tape_entry *dest = &s->tape->entries[i];
        if (source->is_param) {
            if (source->slot < 0 || source->slot >= m->count || source->output != m->param[source->slot])
                die("scenario snapshot requires one registered entry per body parameter");
            dest->output = nt_tensor_ref(s->parameters[source->slot]);
        } else dest->output = scenario_clone(source->output);
        dest->grad = scenario_clone(source->grad);
    }
    for (int i = 0; i < live->n_params; ++i) {
        s->tape->adam[i].m = scenario_clone(live->adam[i].m);
        s->tape->adam[i].v = scenario_clone(live->adam[i].v);
        s->tape->adam[i].acc_grad = scenario_clone(live->adam[i].acc_grad);
    }
    return s;
}

static int scenario_matches(const scenario_snapshot *s, const body *m,
                            const nt_chuck_architect *a, uint64_t windows) {
    const nt_tape *t = nt_tape_get(), *saved = s->tape;
    if (s->count != m->count || s->windows != windows || s->noise_rng != nt_chuck_rng_get() ||
        s->training != nt_is_training() || nt_chuck_architect_hash(&s->architect) != nt_chuck_architect_hash(a) ||
        t->count != saved->count || t->n_params != saved->n_params || t->active != saved->active ||
        memcmp(&t->chuck, &saved->chuck, sizeof(t->chuck)) ||
        memcmp(t->chuck_params, saved->chuck_params, sizeof(t->chuck_params))) return 0;
    for (int i = 0; i < m->count; ++i)
        if (!scenario_tensor_equal(m->param[i], s->parameters[i])) return 0;
    for (int i = 0; i < t->n_params; ++i) {
        const nt_adam_state *x = &t->adam[i], *y = &saved->adam[i];
        if (x->t != y->t || !scenario_tensor_equal(x->m, y->m) || !scenario_tensor_equal(x->v, y->v) ||
            !scenario_tensor_equal(x->acc_grad, y->acc_grad)) return 0;
    }
    for (int i = 0; i < t->count; ++i) {
        const nt_tape_entry *x = &t->entries[i], *y = &saved->entries[i];
        if (x->op != y->op || x->parent1 != y->parent1 || x->parent2 != y->parent2 || x->parent3 != y->parent3 ||
            memcmp(&x->aux, &y->aux, sizeof(float)) || memcmp(&x->aux2, &y->aux2, sizeof(float)) ||
            memcmp(&x->aux3, &y->aux3, sizeof(float)) || memcmp(&x->aux4, &y->aux4, sizeof(float)) ||
            x->is_param != y->is_param || x->no_decay != y->no_decay || x->frozen != y->frozen || x->slot != y->slot ||
            !scenario_tensor_equal(x->output, y->output) || !scenario_tensor_equal(x->grad, y->grad)) return 0;
        if (x->is_param && x->output != m->param[x->slot]) return 0;
    }
    return 1;
}

static void scenario_restore(const scenario_snapshot *s, body *m, nt_chuck_architect *a, uint64_t *windows) {
    if (nt_get_gpu_mode()) die("scenario restore requires CPU mode");
    nt_tape_destroy();
    for (int i = 0; i < m->count; ++i)
        memcpy(m->param[i]->data, s->parameters[i]->data, (size_t)m->param[i]->len * sizeof(float));
    nt_tape *live = nt_tape_get();
    memcpy(live, s->tape, sizeof(*live));
    for (int i = 0; i < live->count; ++i) {
        nt_tape_entry *e = &live->entries[i];
        e->output = e->is_param ? nt_tensor_ref(m->param[e->slot]) : scenario_clone(s->tape->entries[i].output);
        e->grad = scenario_clone(s->tape->entries[i].grad);
    }
    for (int i = 0; i < live->n_params; ++i) {
        nt_adam_state *d = &live->adam[i];
        const nt_adam_state *source = &s->tape->adam[i];
        d->m = scenario_clone(source->m); d->v = scenario_clone(source->v);
        d->acc_grad = scenario_clone(source->acc_grad);
    }
    memcpy(a, &s->architect, sizeof(*a));
    *windows = s->windows;
    if (nt_chuck_rng_set(s->noise_rng)) die("scenario noise RNG restore failed");
    nt_train_mode(s->training);
    if (!scenario_matches(s, m, a, *windows)) die("scenario snapshot_restore_exact failed");
}

static void scenario_release(scenario_snapshot *s) {
    for (int i = 0; i < s->tape->count; ++i) {
        nt_tensor_free(s->tape->entries[i].output); nt_tensor_free(s->tape->entries[i].grad);
    }
    for (int i = 0; i < s->tape->n_params; ++i) {
        nt_tensor_free(s->tape->adam[i].m); nt_tensor_free(s->tape->adam[i].v);
        nt_tensor_free(s->tape->adam[i].acc_grad);
    }
    for (int i = 0; i < s->count; ++i) nt_tensor_free(s->parameters[i]);
    free(s->tape); free(s);
}

static float scenario_probe_loss(body *m, const uint32_t *data, const size_t *offsets) {
    double total = 0;
    for (int i = 0; i < SCENARIO_PROBES; ++i) total += read_loss(body_forward(m, data, offsets[i], 0));
    return (float)(total / SCENARIO_PROBES);
}

static int scenario_checkpoint(int step) { return step == 1 || step == 32 || step == 128 || step == 384; }
static int scenario_horizon(int step) { return step == 1 || step == 4 || step == 16; }

static void scenario_diagnose(body *m, const uint32_t *data, size_t count, size_t split,
        nt_chuck_architect *architect, uint64_t *windows, size_t offset, float before, float lr,
        int step, const char *prefix, FILE *trace) {
    static const nt_chuck_action_kind actions[] = {NT_CHUCK_ACTION_HOLD, NT_CHUCK_ACTION_BRAKE, NT_CHUCK_ACTION_PUSH};
    static const char *names[] = {"hold", "brake", "push"};
    scenario_snapshot *saved = scenario_capture(m, architect, *windows);
    uint64_t saved_hash = scenario_state_hash(m, architect, *windows);
    size_t offsets[SCENARIO_HORIZON + SCENARIO_PROBES];
    uint64_t states[SCENARIO_HORIZON + SCENARIO_PROBES];
    offsets[0] = offset; states[0] = *windows;
    uint64_t lookahead = *windows;
    for (int i = 1; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) {
        offsets[i] = (size_t)(next_window(&lookahead) % (split - CTX)); states[i] = lookahead;
    }
    char fork_prefix[4096], suffix[64], path[4096];
    snprintf(suffix, sizeof(suffix), ".fork-%d", step); path_for(fork_prefix, sizeof(fork_prefix), prefix, suffix);
    path_for(path, sizeof(path), fork_prefix, ".body.bin");
    if (nt_save(path, m->param, m->count)) die("scenario body snapshot save failed");
    path_for(path, sizeof(path), fork_prefix, ".policy.bin");
    if (nt_chuck_architect_save(architect, path)) die("scenario policy snapshot save failed");
    save_optimizer(fork_prefix, *windows, step - 1);
    nt_tensor *gradients[MAX_PARAMS] = {0};
    const nt_tape *tape = nt_tape_get();
    for (int i = 0; i < tape->count; ++i)
        if (tape->entries[i].is_param) gradients[tape->entries[i].slot] = tape->entries[i].grad;
    for (int i = 0; i < m->count; ++i) if (!gradients[i]) die("scenario missing body gradient");
    path_for(path, sizeof(path), fork_prefix, ".gradients.bin");
    if (nt_save(path, gradients, m->count)) die("scenario gradient snapshot save failed");
    float future_before = scenario_probe_loss(m, data, offsets + SCENARIO_HORIZON);
    float heldout_before = evaluate(m, data, split, count);
    fprintf(trace, "{\"type\":\"fork\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"loss_before\":%.9g,"
            "\"future_probe_before\":%.9g,\"heldout_before\":%.9g,\"offsets\":[", step, saved_hash, before, future_before, heldout_before);
    for (int i = 0; i < SCENARIO_HORIZON + SCENARIO_PROBES; ++i) fprintf(trace, "%s%zu", i ? "," : "", offsets[i]);
    fputs("]}\n", trace);
    for (int ai = 0; ai < 3; ++ai) {
        scenario_restore(saved, m, architect, windows);
        if (scenario_state_hash(m, architect, *windows) != saved_hash) die("scenario restored identity differs");
        float immediate_after = 0;
        for (int h = 1; h <= SCENARIO_HORIZON; ++h) {
            float loss = before;
            if (h > 1) {
                int loss_idx = body_forward(m, data, offsets[h - 1], 1);
                loss = read_loss(loss_idx); nt_tape_backward(loss_idx);
                if (!isfinite(nt_tape_clip_grads(1.0f))) die("scenario nonfinite gradient");
                *windows = states[h - 1];
            }
            nt_chuck_action action = {h == 1 ? actions[ai] : NT_CHUCK_ACTION_HOLD, 0};
            if (nt_tape_chuck_step_action(lr, loss, &action, &architect->config.limits)) die("scenario action refused");
            float after = read_loss(body_forward(m, data, offsets[h - 1], 0));
            if (h == 1) immediate_after = after;
            fprintf(trace, "{\"type\":\"branch_step\",\"checkpoint\":%d,\"intervention\":\"%s\",\"update\":%d,"
                    "\"executed_action\":\"%s\",\"offset\":%zu,\"before\":%.9g,\"after_same_window\":%.9g,"
                    "\"window_rng\":\"%016" PRIx64 "\",\"noise_rng\":%" PRIu32 "}\n",
                    step, names[ai], h, h == 1 ? names[ai] : "hold", offsets[h - 1], loss, after, *windows, nt_chuck_rng_get());
            if (scenario_horizon(h)) {
                float origin_after = read_loss(body_forward(m, data, offsets[0], 0));
                float future_after = scenario_probe_loss(m, data, offsets + SCENARIO_HORIZON);
                float heldout_after = evaluate(m, data, split, count);
                fprintf(trace, "{\"type\":\"comparison\",\"checkpoint\":%d,\"action\":\"%s\",\"horizon\":%d,"
                        "\"initial_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\","
                        "\"immediate_before\":%.9g,\"immediate_after\":%.9g,\"immediate_improvement\":%.9g,"
                        "\"origin_after_horizon\":%.9g,\"future_before\":%.9g,\"future_after\":%.9g,\"future_improvement\":%.9g,"
                        "\"heldout_before\":%.9g,\"heldout_after\":%.9g,\"heldout_improvement\":%.9g,"
                        "\"dampen\":%.9g,\"lr_scale\":%.9g,\"noise\":%.9g}\n",
                        step, names[ai], h, saved_hash, scenario_state_hash(m, architect, *windows), before, immediate_after,
                        before - immediate_after, origin_after, future_before, future_after, future_before - future_after,
                        heldout_before, heldout_after, heldout_before - heldout_after,
                        nt_tape_get()->chuck.dampen, nt_tape_get()->chuck.lr_scale, nt_tape_get()->chuck.noise);
            }
        }
    }
    scenario_restore(saved, m, architect, windows);
    if (scenario_state_hash(m, architect, *windows) != saved_hash) die("scenario host continuation restore differs");
    fprintf(trace, "{\"type\":\"restore\",\"step\":%d,\"state_hash\":\"%016" PRIx64 "\",\"exact\":true}\n", step, saved_hash);
    fflush(trace);
    scenario_release(saved);
}

static int scenario_main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s --scenarios BODY TOKENS PREFIX STEPS SEED LR CONFIG PROBES(0|1)\n", argv[0]);
        return 2;
    }
    if (nt_get_gpu_mode()) die("scenario diagnostics require CPU mode");
    const char *prefix = argv[4];
    int steps = (int)parse_integer(argv[5], 1, 1000000), probes = (int)parse_integer(argv[9], 0, 1);
    unsigned seed = (unsigned)parse_integer(argv[6], 1, UINT_MAX);
    char *end = NULL; errno = 0;
    float lr = strtof(argv[7], &end);
    if (errno || *end || !isfinite(lr) || lr <= 0 || lr > 1) die("invalid scenario learning rate");
    nt_seed(seed); if (nt_chuck_rng_set(UINT32_C(2463534242))) die("scenario initial noise RNG failed");
    body m; body_init(&m, argv[2]);
    size_t count; uint32_t *data = read_tokens(argv[3], &count, m.vocab);
    size_t split = count * 9 / 10;
    if (split <= CTX || count - split <= CTX) die("corpus split too small");
    nt_chuck_architect_config config; read_config(&config, argv[8]);
    if (config.mode != NT_CHUCK_ARCHITECT_LEARNED) die("scenario host requires learned mode");
    config.seed = seed;
    nt_chuck_architect architect;
    if (nt_chuck_architect_init(&architect, &config)) die("scenario host init failed");
    uint64_t windows = seed;
    char path[4096]; path_for(path, sizeof(path), prefix, ".jsonl");
    FILE *trace = fopen(path, "w"); if (!trace) die("scenario receipt open failed");
    fprintf(trace, "{\"type\":\"scenario_run\",\"body\":\"%s\",\"seed\":%u,\"steps\":%d,"
            "\"probes\":%s,\"parameters\":%ld,\"lr\":%.9g,\"policy_initial\":\"%016" PRIx64 "\"}\n",
            m.name, seed, steps, probes ? "true" : "false", m.elements, lr, nt_chuck_architect_hash(&architect));
    float initial_eval = evaluate(&m, data, split, count);
    double start = seconds();
    for (int step = 1; step <= steps; ++step) {
        size_t offset = (size_t)(next_window(&windows) % (split - CTX));
        int loss_idx = body_forward(&m, data, offset, 1);
        float before = read_loss(loss_idx); nt_tape_backward(loss_idx);
        float norm = nt_tape_clip_grads(1.0f); if (!isfinite(norm)) die("scenario host nonfinite gradient");
        if (probes && scenario_checkpoint(step)) scenario_diagnose(&m, data, count, split, &architect, &windows,
                                                                  offset, before, lr, step, prefix, trace);
        nt_chuck_architect_decision decision;
        if (nt_chuck_architect_step(&architect, lr, before, &decision)) die("scenario host action refused");
        float after = read_loss(body_forward(&m, data, offset, 0));
        nt_chuck_architect_receipt consequence;
        if (nt_chuck_architect_feedback(&architect, after, &consequence)) die("scenario host feedback refused");
        fprintf(trace, "{\"type\":\"host_step\",\"step\":%d,\"offset\":%zu,\"before\":%.9g,\"after\":%.9g,"
                "\"gradient_norm\":%.9g,\"action\":%d,\"reward\":%.9g,\"policy_hash\":\"%016" PRIx64 "\","
                "\"state_hash\":\"%016" PRIx64 "\"}\n", step, offset, before, after, norm, decision.action.kind,
                consequence.reward, nt_chuck_architect_hash(&architect), scenario_state_hash(&m, &architect, windows));
        if (step % 128 == 0 || step == steps) {
            printf("scenario %s seed=%u probes=%d step=%d/%d loss=%.6f elapsed=%.2fs\n",
                    m.name, seed, probes, step, steps, before, seconds() - start);
            fflush(stdout); fflush(trace);
        }
    }
    float final_eval = evaluate(&m, data, split, count);
    struct rusage usage; if (getrusage(RUSAGE_SELF, &usage)) die("scenario getrusage failed");
    long rss = usage.ru_maxrss;
#ifdef __APPLE__
    rss /= 1024;
#endif
    fprintf(trace, "{\"type\":\"scenario_summary\",\"initial_heldout\":%.9g,\"final_heldout\":%.9g,"
            "\"seconds\":%.9g,\"max_rss_kib\":%ld,\"policy_hash\":\"%016" PRIx64 "\"}\n",
            initial_eval, final_eval, seconds() - start, rss, nt_chuck_architect_hash(&architect));
    if (ferror(trace) || fclose(trace)) die("scenario receipt write failed");
    path_for(path, sizeof(path), prefix, ".final.bin");
    if (nt_save(path, m.param, m.count)) die("scenario final body save failed");
    path_for(path, sizeof(path), prefix, ".policy.final.bin");
    if (nt_chuck_architect_save(&architect, path)) die("scenario final policy save failed");
    save_optimizer(prefix, windows, steps);
    nt_tape_destroy();
    for (int i = 0; i < m.count; ++i) nt_tensor_free(m.param[i]);
    free(data);
    return 0;
}
#endif
