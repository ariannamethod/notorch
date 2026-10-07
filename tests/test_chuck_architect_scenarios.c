// Chuck temporal scenarios: interrupted lives, interleaving and mode boundaries.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "chuck_architect.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SLOTS 3
#define WIDTH 11
#define LENGTH 96
static const int sizes[SLOTS] = {5, 7, 11};
static const char *scenario;
static nt_tensor *weights[SLOTS];
static unsigned long transitions;
#define REQUIRE(c, message) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", scenario, __LINE__, message); return 0; } } while (0)

// The host's complete fixed-size training body. The policy life deliberately
// remains a separate object, as it is in the public checkpoint contract.
typedef struct {
    float p[SLOTS][WIDTH], g[SLOTS][WIDTH], m[SLOTS][WIDTH], v[SLOTS][WIDTH];
    int moment_step[SLOTS];
    nt_chuck_state global;
    nt_chuck_param_state local[SLOTS];
    uint32_t noise_rng;
} body_state;

typedef struct {
    body_state body;
    nt_chuck_architect life;
} world;

static void close_body(void) {
    nt_tape_destroy();
    for (int s = 0; s < SLOTS; ++s) { nt_tensor_free(weights[s]); weights[s] = NULL; }
}

static void new_body(int identity) {
    close_body();
    nt_tape_start();
    nt_chuck_rng_set((uint32_t)(7103 + identity * 23));
    for (int s = 0; s < SLOTS; ++s) {
        weights[s] = nt_tensor_new(sizes[s]);
        for (int j = 0; j < sizes[s]; ++j)
            weights[s]->data[j] = .037f * (1 + s + j) + .01f * identity;
        int index = nt_tape_param(weights[s]);
        nt_tape_get()->entries[index].grad = nt_tensor_new(sizes[s]);
    }
}

static body_state capture_body(void) {
    body_state image;
    memset(&image, 0, sizeof image);
    nt_tape *t = nt_tape_get();
    image.global = t->chuck;
    memcpy(image.local, t->chuck_params, sizeof image.local);
    image.noise_rng = nt_chuck_rng_get();
    for (int s = 0; s < SLOTS; ++s) {
        size_t bytes = (size_t)sizes[s] * sizeof(float);
        memcpy(image.p[s], weights[s]->data, bytes);
        memcpy(image.g[s], t->entries[s].grad->data, bytes);
        memcpy(image.m[s], t->adam[s].m->data, bytes);
        memcpy(image.v[s], t->adam[s].v->data, bytes);
        image.moment_step[s] = t->adam[s].t;
    }
    return image;
}

static void restore_body(const body_state *image) {
    new_body(0);
    nt_tape *t = nt_tape_get();
    t->chuck = image->global;
    memcpy(t->chuck_params, image->local, sizeof image->local);
    nt_chuck_rng_set(image->noise_rng);
    for (int s = 0; s < SLOTS; ++s) {
        size_t bytes = (size_t)sizes[s] * sizeof(float);
        memcpy(weights[s]->data, image->p[s], bytes);
        memcpy(t->entries[s].grad->data, image->g[s], bytes);
        memcpy(t->adam[s].m->data, image->m[s], bytes);
        memcpy(t->adam[s].v->data, image->v[s], bytes);
        t->adam[s].t = image->moment_step[s];
    }
}

static int same_body(const body_state *expected) {
    body_state actual = capture_body();
    return memcmp(&actual, expected, sizeof actual) == 0;
}

// A changing, exactly specified quadratic world; the after measurement uses
// the same identity and phase as the before measurement. Gradients are exact
// analytic derivatives, so these gates isolate optimizer/policy chronology.
static float environment(int step, int identity, int gradients) {
    float loss = 0;
    for (int s = 0; s < SLOTS; ++s)
        for (int j = 0; j < sizes[s]; ++j) {
            float target = .023f * ((step / 13 + identity * 3 + s * 5 + j) % 19 - 9);
            float error = weights[s]->data[j] - target;
            loss += error * error;
            if (gradients) nt_tape_get()->entries[s].grad->data[j] = 2 * error;
        }
    return loss;
}

static int new_life(nt_chuck_architect *a, int identity, uint32_t mask, float exploration) {
    nt_chuck_architect_config config;
    nt_chuck_architect_config_default(&config);
    config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    config.seed = (uint32_t)(19 + identity * 31);
    config.exploration = exploration;
    config.learning_rate = .03f;
    config.limits.enabled_actions = mask;
    snprintf(config.life_id, sizeof config.life_id, "scenario.%d", identity);
    return nt_chuck_architect_init(a, &config) == NT_CHUCK_OK;
}

static int start_world(world *w, int identity, uint32_t mask, float exploration) {
    new_body(identity);
    if (!new_life(&w->life, identity, mask, exploration)) return 0;
    nt_chuck_action noise = {NT_CHUCK_ACTION_SET_NOISE, .0007f * (identity + 1)};
    if (nt_tape_chuck_step_action(0, environment(0, identity, 1), &noise, NULL)) return 0;
    w->body = capture_body();
    return 1;
}

static int act(nt_chuck_architect *a, int step, int identity, nt_chuck_architect_decision *d) {
    ++transitions;
    return nt_chuck_architect_step(a, .008f, environment(step, identity, 1), d) == NT_CHUCK_OK;
}

static int outcome(nt_chuck_architect *a, int step, int identity) {
    nt_chuck_architect_receipt receipt;
    int rc = nt_chuck_architect_feedback(a, environment(step, identity, 0), &receipt);
    return rc == NT_CHUCK_OK && receipt.learned && !receipt.nonfinite
        && receipt.decision == a->decisions && a->updates == a->decisions && !a->pending;
}

static int advance(nt_chuck_architect *a, int step, int identity) {
    return act(a, step, identity, NULL) && outcome(a, step, identity);
}

static int test_mixed_replay(void) {
    enum {STEPS = 1200, CUT = 613};
    body_state expected[STEPS], checkpoint;
    unsigned counts[NT_CHUCK_ACTION_COUNT] = {0};
    int noise_seen = 0;
    new_body(0);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) restore_body(&checkpoint);
        for (int step = pass ? CUT : 0; step < STEPS; ++step) {
            if (!pass && step == CUT) checkpoint = capture_body();
            nt_chuck_action a = {(nt_chuck_action_kind)((step * 5 + 3) % NT_CHUCK_ACTION_COUNT), 0};
            if (a.kind == NT_CHUCK_ACTION_SET_DAMPEN) a.value = .3f + .17f * (step % 11);
            if (a.kind == NT_CHUCK_ACTION_SET_LR_SCALE) a.value = .1f + .1f * (step % 12);
            if (a.kind == NT_CHUCK_ACTION_SET_NOISE) a.value = .0005f * (step % 9);
            REQUIRE(nt_tape_chuck_step_action(.003f, environment(step, 0, 1), &a, NULL) == 0,
                    "mixed action accepted");
            ++transitions;
            if (pass) REQUIRE(same_body(&expected[step]), "mixed body/history/moments/RNG replay is exact at every step");
            else {
                expected[step] = capture_body();
                ++counts[a.kind];
                noise_seen |= expected[step].global.noise > 0;
            }
        }
    }
    for (int k = 0; k < NT_CHUCK_ACTION_COUNT; ++k) REQUIRE(counts[k] > 100, "each declared primitive participates");
    REQUIRE(noise_seen && expected[STEPS - 1].global.global_step == STEPS, "mixed trace crosses noise and macro phases");
    printf("  mixed: %d steps, split=%d, every action >= %u executions\n", STEPS, CUT, counts[0]);
    return 1;
}

static int test_interruptions(void) {
    static const int cuts[] = {0, 1, 7, 15, 16, 31, 47, 63};
    world checkpoints[sizeof cuts / sizeof *cuts][3], baseline;
    body_state expected[LENGTH];
    uint64_t policy_hash[LENGTH];
    REQUIRE(start_world(&baseline, 0, NT_CHUCK_ALL_ACTIONS, .75f), "baseline training world");
    for (int step = 0; step < LENGTH; ++step) {
        int cut = -1;
        for (size_t j = 0; j < sizeof cuts / sizeof *cuts; ++j) if (cuts[j] == step) cut = (int)j;
        if (cut >= 0) { checkpoints[cut][0].body = capture_body(); checkpoints[cut][0].life = baseline.life; }
        REQUIRE(act(&baseline.life, step, 0, NULL), "uninterrupted action");
        if (cut >= 0) { checkpoints[cut][1].body = capture_body(); checkpoints[cut][1].life = baseline.life; }
        REQUIRE(outcome(&baseline.life, step, 0), "uninterrupted consequence");
        if (cut >= 0) { checkpoints[cut][2].body = capture_body(); checkpoints[cut][2].life = baseline.life; }
        expected[step] = capture_body();
        policy_hash[step] = nt_chuck_architect_hash(&baseline.life);
    }
    char directory[] = "/tmp/chuck-scenario-stage-XXXXXX";
    REQUIRE(mkdtemp(directory), "temporary lives directory");
    char path[256]; snprintf(path, sizeof path, "%s/life.bin", directory);
    for (size_t cut = 0; cut < sizeof cuts / sizeof *cuts; ++cut)
        for (int phase = 0; phase < 3; ++phase) {
            world *saved = &checkpoints[cut][phase];
            nt_chuck_architect resumed;
            REQUIRE(nt_chuck_architect_save(&saved->life, path) == 0, "persist each interruption stage");
            restore_body(&saved->body);
            REQUIRE(nt_chuck_architect_load(&resumed, path) == 0, "wake each interruption stage");
            if (phase == 1) {
                // Inspection may happen while the environment result is outstanding.
                uint64_t waiting_hash = nt_chuck_architect_hash(&resumed);
                nt_chuck_observation inspection;
                for (int look = 0; look < 5; ++look)
                    REQUIRE(nt_tape_chuck_observe(10 + look, &inspection) == 0, "pending read-only inspection");
                REQUIRE(nt_chuck_architect_hash(&resumed) == waiting_hash && same_body(&saved->body),
                        "waiting for a consequence does not consume state");
                REQUIRE(outcome(&resumed, cuts[cut], 0), "delayed same-window consequence after wake");
            }
            if (phase) REQUIRE(same_body(&expected[cuts[cut]])
                               && nt_chuck_architect_hash(&resumed) == policy_hash[cuts[cut]],
                               "interrupted boundary rejoins uninterrupted life exactly");
            int next = cuts[cut] + (phase != 0);
            for (int step = next; step < LENGTH; ++step) {
                REQUIRE(advance(&resumed, step, 0), "resumed training action/consequence");
                REQUIRE(same_body(&expected[step]) && nt_chuck_architect_hash(&resumed) == policy_hash[step],
                        "resumed continuation equals uninterrupted continuation at every step");
            }
        }
    unlink(path); rmdir(directory);
    printf("  interruption: %zu cuts x 3 stages, %d-step reference\n", sizeof cuts / sizeof *cuts, LENGTH);
    return 1;
}

static int test_interleaved_lives(void) {
    world initial[2], live[2];
    body_state expected[2][LENGTH];
    uint64_t life_hash[2][LENGTH];
    for (int who = 0; who < 2; ++who) {
        REQUIRE(start_world(&initial[who], who, NT_CHUCK_ALL_ACTIONS, .8f), "independent named life");
        live[who] = initial[who];
        for (int step = 0; step < LENGTH; ++step) {
            REQUIRE(advance(&live[who].life, step, who), "serial reference life");
            expected[who][step] = capture_body();
            life_hash[who][step] = nt_chuck_architect_hash(&live[who].life);
        }
        live[who] = initial[who];
    }
    char directory[] = "/tmp/chuck-scenario-lives-XXXXXX";
    REQUIRE(mkdtemp(directory), "temporary independent lives");
    char paths[2][256];
    for (int who = 0; who < 2; ++who) snprintf(paths[who], sizeof paths[who], "%s/%d.bin", directory, who);
    for (int step = 0; step < LENGTH; ++step)
        for (int turn = 0; turn < 2; ++turn) {
            int who = turn ^ (step % 2);
            restore_body(&live[who].body);
            REQUIRE(nt_chuck_architect_save(&live[who].life, paths[who]) == 0, "put named life to sleep");
            nt_chuck_architect awake = live[1 - who].life;
            REQUIRE(nt_chuck_architect_load(&awake, paths[who]) == 0
                    && !strcmp(awake.config.life_id, live[who].life.config.life_id), "wake requested saved identity");
            REQUIRE(advance(&awake, step, who), "interleaved step");
            REQUIRE(same_body(&expected[who][step]) && nt_chuck_architect_hash(&awake) == life_hash[who][step],
                    "other life cannot alter this body's continuation");
            live[who].life = awake;
            live[who].body = capture_body();
        }
    for (int who = 0; who < 2; ++who) unlink(paths[who]);
    rmdir(directory);
    REQUIRE(life_hash[0][LENGTH - 1] != life_hash[1][LENGTH - 1], "distinct lives retain distinct acquired state");
    printf("  lives: two independent identities, %d interleaved saved wakes\n", 2 * LENGTH);
    return 1;
}

static int test_mode_transitions(void) {
    world live;
    REQUIRE(start_world(&live, 0, NT_CHUCK_ALL_ACTIONS, .6f), "mode-switch training life");
    for (int step = 0; step < 16; ++step) REQUIRE(advance(&live.life, step, 0), "initial learned history");
    nt_chuck_architect acquired = live.life;
    body_state boundary = capture_body();
    uint64_t acquired_hash = nt_chuck_architect_hash(&acquired);
    for (int phase = 0; phase < 2; ++phase) {
        live.life.config.mode = phase ? NT_CHUCK_ARCHITECT_DISABLED : NT_CHUCK_ARCHITECT_LEGACY;
        uint64_t sleeping = nt_chuck_architect_hash(&live.life);
        for (int step = 16 + phase * 20; step < 36 + phase * 20; ++step) {
            REQUIRE(nt_chuck_architect_step(&live.life, .008f, environment(step, 0, 1), NULL) == 0,
                    "compatibility phase executes");
            REQUIRE(nt_chuck_architect_hash(&live.life) == sleeping, "compatibility phase preserves acquired policy exactly");
        }
    }
    body_state compatibility = capture_body();
    live.life.config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    REQUIRE(nt_chuck_architect_hash(&live.life) == acquired_hash, "returning to learned mode wakes same acquired state");
    restore_body(&boundary);
    for (int step = 16; step < 56; ++step) nt_tape_chuck_step(.008f, environment(step, 0, 1));
    REQUIRE(same_body(&compatibility), "legacy and disabled phases equal ordinary Chuck on the same body");
    for (int step = 56; step < 72; ++step) REQUIRE(advance(&live.life, step, 0), "learned life resumes after compatibility");
    body_state continued = capture_body();
    uint64_t continued_hash = nt_chuck_architect_hash(&live.life);
    restore_body(&compatibility);
    for (int step = 56; step < 72; ++step) REQUIRE(advance(&acquired, step, 0), "reference resumed acquired life");
    REQUIRE(same_body(&continued) && nt_chuck_architect_hash(&acquired) == continued_hash,
            "mode changes preserve future learned behavior");
    REQUIRE(act(&acquired, 72, 0, NULL), "pending mode boundary");
    for (int mode = NT_CHUCK_ARCHITECT_DISABLED; mode <= NT_CHUCK_ARCHITECT_LEGACY; ++mode) {
        acquired.config.mode = (nt_chuck_architect_mode)mode;
        nt_chuck_architect before = acquired;
        body_state untouched = capture_body();
        REQUIRE(nt_chuck_architect_step(&acquired, .008f, 1, NULL) != 0
                && !memcmp(&before, &acquired, sizeof before) && same_body(&untouched),
                "pending consequence cannot be skipped by switching mode");
    }
    acquired.config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    REQUIRE(outcome(&acquired, 72, 0), "return and finish original pending consequence");
    uint64_t finished = nt_chuck_architect_hash(&acquired);
    REQUIRE(nt_chuck_architect_feedback(&acquired, -100, NULL) != 0
            && nt_chuck_architect_hash(&acquired) == finished, "duplicate outcome refuses credit after transition");
    return 1;
}

static void copy_policy_weights(nt_chuck_architect *to, const nt_chuck_architect *from) {
    memcpy(to->w1, from->w1, sizeof to->w1); memcpy(to->b1, from->b1, sizeof to->b1);
    memcpy(to->w2, from->w2, sizeof to->w2); memcpy(to->b2, from->b2, sizeof to->b2);
}

static int test_temporal_readout(void) {
    // Named question: at identical current observation and acquired weights,
    // does the preceding measured consequence reach the next outcome readout?
    nt_chuck_architect memories[2];
    body_state common;
    for (int arm = 0; arm < 2; ++arm) {
        new_body(0);
        REQUIRE(new_life(&memories[arm], 0, NT_CHUCK_ALL_ACTIONS, 0), "paired temporal life");
        environment(0, 0, 1);
        REQUIRE(nt_chuck_architect_step(&memories[arm], .001f, 1, NULL) == 0, "paired executed action");
        REQUIRE(nt_chuck_architect_feedback(&memories[arm], arm ? 1.5f : .5f, NULL) == 0, "opposite recorded consequence");
        if (!arm) common = capture_body();
        else REQUIRE(same_body(&common), "paired action leaves identical physical body");
    }
    copy_policy_weights(&memories[1], &memories[0]);
    nt_chuck_architect_decision actual[2];
    nt_chuck_architect retained[2] = {memories[0], memories[1]};
    for (int arm = 0; arm < 2; ++arm) {
        restore_body(&common);
        REQUIRE(nt_chuck_architect_step(&memories[arm], .001f, 1, &actual[arm]) == 0, "same present observation");
    }
    REQUIRE(!memcmp(&actual[0].observation, &actual[1].observation, sizeof actual[0].observation),
            "temporal comparison fixes present observation");
    REQUIRE(memcmp(actual[0].features, actual[1].features, sizeof actual[0].features)
            && memcmp(actual[0].scores, actual[1].scores, sizeof actual[0].scores),
            "prior consequence changes features and predicted action outcomes at fixed weights");
    retained[1].prev_reward = retained[0].prev_reward;
    REQUIRE(nt_chuck_architect_hash(&retained[1]) == nt_chuck_architect_hash(&retained[0]),
            "equalizing the sole differing history value rejoins identical state");
    printf("  temporal: identical observation/weights, hold scores %.9g / %.9g\n", actual[0].scores[0], actual[1].scores[0]);
    return 1;
}

static int test_masked_exploration(void) {
    uint32_t hold = NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD);
    world w;
    body_state expected[LENGTH];
    REQUIRE(start_world(&w, 0, hold, 1), "hold-only learned world");
    body_state initial = capture_body();
    uint32_t initial_rng = w.life.rng;
    for (int step = 0; step < LENGTH; ++step) {
        nt_chuck_architect_decision d;
        REQUIRE(act(&w.life, step, 0, &d) && d.action.kind == NT_CHUCK_ACTION_HOLD && !d.explored,
                "single available action cannot explore unavailable alternatives");
        REQUIRE(outcome(&w.life, step, 0), "hold consequence");
        expected[step] = capture_body();
    }
    REQUIRE(w.life.rng == initial_rng, "singleton action vocabulary consumes no exploration RNG");
    restore_body(&initial);
    nt_chuck_action action = {NT_CHUCK_ACTION_HOLD, 0};
    for (int step = 0; step < LENGTH; ++step) {
        REQUIRE(nt_tape_chuck_step_action(.008f, environment(step, 0, 1), &action, NULL) == 0,
                "direct hold reference");
        REQUIRE(same_body(&expected[step]), "learned singleton action equals direct executor over whole trajectory");
    }
    REQUIRE(start_world(&w, 1, hold | NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_BRAKE), 1), "two-action exploration world");
    unsigned held = 0, braked = 0;
    for (int step = 0; step < 512; ++step) {
        nt_chuck_architect_decision d;
        REQUIRE(act(&w.life, step, 1, &d), "bounded exploratory action");
        REQUIRE(d.action.kind == NT_CHUCK_ACTION_HOLD || d.action.kind == NT_CHUCK_ACTION_BRAKE,
                "disabled PUSH cannot execute during sustained exploration");
        held += d.action.kind == NT_CHUCK_ACTION_HOLD;
        braked += d.action.kind == NT_CHUCK_ACTION_BRAKE;
        REQUIRE(outcome(&w.life, step, 1), "exploration consequence");
    }
    REQUIRE(held > 0 && braked > 0, "both available actions are exercised");
    printf("  masked exploration: hold=%u brake=%u push=0\n", held, braked);
    return 1;
}

int main(int argc, char **argv) {
    const struct {const char *name; int (*run)(void);} cases[] = {
        {"mixed_replay", test_mixed_replay}, {"interruptions", test_interruptions},
        {"interleaved_lives", test_interleaved_lives}, {"mode_transitions", test_mode_transitions},
        {"temporal_readout", test_temporal_readout}, {"masked_exploration", test_masked_exploration}
    };
    int passed = 0, failed = 0;
    for (size_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
        if (argc == 2 && strcmp(argv[1], cases[i].name)) continue;
        scenario = cases[i].name;
        if (cases[i].run()) { ++passed; printf("PASS %s\n", scenario); } else ++failed;
        close_body();
    }
    if (!passed && !failed) return 2;
    printf("Chuck scenarios: %d passed, %d failed, %lu selected/executed transitions\n", passed, failed, transitions);
    return failed ? 1 : 0;
}
