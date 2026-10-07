// Checked Chuck actions: empty/frozen bodies, numeric state and transition edges.
#include "notorch.h"
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static const char *group;
static int assertions;
#define CHECK(c, why) do { assertions++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", group, __LINE__, why); return 0; \
} } while (0)

static nt_tensor *weights[3];
static int indices[3];
static void close_body(void) {
    nt_tape_destroy();
    for (int i = 0; i < 3; i++) { nt_tensor_free(weights[i]); weights[i] = NULL; }
}
static void open_body(int count) {
    close_body();
    nt_tape_start();
    nt_chuck_rng_set(1234567);
    for (int i = 0; i < count; i++) {
        weights[i] = nt_tensor_new(2);
        weights[i]->data[0] = 1.0f + i;
        weights[i]->data[1] = -2.0f - i;
        indices[i] = nt_tape_param(weights[i]);
        nt_tape_entry *e = &nt_tape_get()->entries[indices[i]];
        e->grad = nt_tensor_new(2);
        e->grad->data[0] = 0.1f;
        e->grad->data[1] = -0.2f;
    }
}
typedef struct {
    nt_chuck_state global;
    nt_chuck_param_state local[3];
    float weight[3][2], moment[3][2], variance[3][2];
    int time[3];
    uint32_t rng;
} snapshot;
static snapshot capture(int count) {
    snapshot s;
    memset(&s, 0, sizeof s);
    nt_tape *t = nt_tape_get();
    s.global = t->chuck;
    s.rng = nt_chuck_rng_get();
    for (int i = 0; i < count; i++) {
        s.local[i] = t->chuck_params[i];
        s.time[i] = t->adam[i].t;
        memcpy(s.weight[i], weights[i]->data, sizeof s.weight[i]);
        memcpy(s.moment[i], t->adam[i].m->data, sizeof s.moment[i]);
        memcpy(s.variance[i], t->adam[i].v->data, sizeof s.variance[i]);
    }
    return s;
}
static int unchanged(snapshot *s, int count) {
    snapshot now = capture(count);
    return memcmp(s, &now, sizeof now) == 0;
}
static int action(nt_chuck_action_kind kind, float value, float loss) {
    nt_chuck_action a = {kind, value};
    return nt_tape_chuck_step_action(.01f, loss, &a, NULL);
}

static int empty_and_frozen(void) {
    open_body(0);
    nt_chuck_observation o;
    CHECK(nt_tape_chuck_observe(1, &o) == 0 && o.grad_norm == 0 &&
          o.grad_trend == 0 && o.frozen_fraction == 0, "empty tape observation");
    uint32_t rng = nt_chuck_rng_get();
    CHECK(action(NT_CHUCK_ACTION_SET_NOISE, .01f, 1) == 0,
          "empty body can carry an action history");
    CHECK(nt_chuck_rng_get() == rng && nt_tape_get()->chuck.global_step == 1,
          "empty body consumes no parameter noise");
    open_body(3);
    nt_tape *t = nt_tape_get();
    nt_tape_freeze_param(indices[0]);
    nt_tensor_free(t->entries[indices[1]].grad);
    t->entries[indices[1]].grad = NULL;
    CHECK(action(NT_CHUCK_ACTION_SET_NOISE, .001f, 1) == 0, "mixed body step");
    CHECK(weights[0]->data[0] == 1 && weights[1]->data[0] == 2 &&
          weights[2]->data[0] != 3, "frozen and missing-gradient slots stay fixed");
    CHECK(t->adam[0].t == 0 && t->adam[1].t == 0 && t->adam[2].t == 1,
          "only updated slot advances its Adam counter");
    CHECK(nt_tape_chuck_observe(1, &o) == 0 &&
          fabsf(o.frozen_fraction - 1.0f / 3) < 1e-6f,
          "frozen fraction counts registered optimizer slots");
    nt_tape_freeze_param(indices[1]); nt_tape_freeze_param(indices[2]);
    float before[3][2];
    for (int i = 0; i < 3; i++) memcpy(before[i], weights[i]->data, sizeof before[i]);
    rng = nt_chuck_rng_get();
    for (int i = 0; i < 40; i++) CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0,
                                     "all-frozen body keeps sensing");
    for (int i = 0; i < 3; i++) CHECK(memcmp(before[i], weights[i]->data, sizeof before[i]) == 0,
                                     "all-frozen weights unchanged");
    CHECK(nt_chuck_rng_get() == rng, "all-frozen body consumes no noise");
    close_body();
    // A permanently frozen base has no optimizer slot.
    nt_tape_start(); weights[0] = nt_tensor_new(2);
    indices[0] = nt_tape_param_frozen(weights[0]);
    CHECK(nt_tape_chuck_observe(1, &o) == 0 && o.frozen_fraction == 0 && o.grad_norm == 0,
          "slot-free base is absent from optimizer summaries");
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0, "slot-free base step");
    close_body(); return 1;
}

static int gradient_refusals(void) {
    const float bad[] = {NAN, INFINITY, -INFINITY, FLT_MAX / 4};
    for (unsigned i = 0; i < sizeof bad / sizeof *bad; i++) {
        open_body(2);
        nt_tape *t = nt_tape_get();
        t->entries[indices[1]].grad->data[1] = bad[i];
        snapshot before = capture(2);
        CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == NT_CHUCK_E_STATE,
              "non-finite or overflowing gradient norm is refused");
        CHECK(unchanged(&before, 2), "late bad gradient leaves earlier slot and RNG unchanged");
        nt_chuck_observation o, prior;
        memset(&o, 0x3a, sizeof o); prior = o;
        CHECK(nt_tape_chuck_observe(1, &o) == NT_CHUCK_E_STATE &&
              memcmp(&o, &prior, sizeof o) == 0, "bad gradient cannot become an observation");
    }
    open_body(1);
    nt_tape *t = nt_tape_get();
    nt_tensor_free(t->entries[0].grad); t->entries[0].grad = nt_tensor_new(1);
    snapshot before = capture(1);
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == NT_CHUCK_E_STATE && unchanged(&before, 1),
          "short gradient is refused before any out-of-bounds access");
    close_body(); return 1;
}

static int local_state_refusals(void) {
    for (int which = 0; which < 8; which++) {
        open_body(1); nt_tape *t = nt_tape_get();
        if (which == 0) t->chuck_params[0].pos = NT_CHUCK_WINDOW;
        if (which == 1) t->chuck_params[0].pos = -1;
        if (which == 2) t->chuck_params[0].full = 2;
        if (which == 3) t->chuck_params[0].dampen = NAN;
        if (which == 4) t->chuck_params[0].grad_hist[0] = NAN;
        if (which == 5) t->adam[0].t = INT_MAX;
        if (which == 6) t->chuck_params[0].stag = INT_MAX;
        if (which == 7) t->chuck_params[0].frozen = -1;
        snapshot before = capture(1);
        CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == NT_CHUCK_E_STATE,
              "invalid local state is refused");
        CHECK(unchanged(&before, 1), "invalid local state refuses transactionally");
    }
    open_body(1); nt_tape *t = nt_tape_get();
    t->chuck_params[0].pos = 1; t->chuck_params[0].grad_hist[0] = NAN;
    nt_chuck_observation o, prior;
    memset(&o, 0x3a, sizeof o); prior = o;
    CHECK(nt_tape_chuck_observe(1, &o) == NT_CHUCK_E_STATE &&
          memcmp(&o, &prior, sizeof o) == 0, "NaN history is refused, not erased into zero trend");
    close_body(); return 1;
}

static int controls_and_noise(void) {
    open_body(1);
    nt_chuck_action_limits l;
    nt_chuck_action_limits_default(&l);
    l.dampen_min = .9f; l.dampen_max = 1.1f;
    nt_chuck_action a = {NT_CHUCK_ACTION_PUSH, 0};
    for (int i = 0; i < 20; i++) CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &l) == 0,
                                     "bounded push");
    CHECK(nt_tape_get()->chuck.dampen == 1.1f, "push saturation exact");
    a.kind = NT_CHUCK_ACTION_BRAKE;
    for (int i = 0; i < 20; i++) CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &l) == 0,
                                     "bounded brake");
    CHECK(nt_tape_get()->chuck.dampen == .9f, "brake saturation exact");
    l.lr_scale_max = .5f;
    snapshot before = capture(1);
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &l) == NT_CHUCK_E_BOUNDS && unchanged(&before, 1),
          "an action cannot leave another control outside its envelope");
    a.kind = NT_CHUCK_ACTION_SET_LR_SCALE; a.value = .5f;
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &l) == 0, "setter enters the new envelope");
    l.enabled_actions = 0; before = capture(1);
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &l) == NT_CHUCK_E_ACTION && unchanged(&before, 1),
          "empty vocabulary refuses execution");
    uint32_t rng = nt_chuck_rng_get();
    CHECK(action(NT_CHUCK_ACTION_SET_NOISE, .001f, 1) == 0 && nt_chuck_rng_get() != rng,
          "noise action consumes its independent stream");
    CHECK(action(NT_CHUCK_ACTION_SET_NOISE, 0, 1) == 0, "noise can be turned off");
    rng = nt_chuck_rng_get();
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0 && nt_chuck_rng_get() == rng,
          "hold with zero noise consumes no stream");
    CHECK(action(NT_CHUCK_ACTION_LEGACY, 0, 1) == 0, "custom life can return to legacy");
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0, "legacy can return to custom");
    close_body(); return 1;
}

static int boundary_counters(void) {
    open_body(1); nt_tape *t = nt_tape_get();
    for (int i = 0; i < 4000; i++) CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0,
                                       "long constant trajectory");
    CHECK(t->chuck.global_step == 4000 && t->chuck.full == 1 && t->chuck.pos == 0 &&
          t->chuck.stag == 3993 && t->chuck.macro_stag == 3 && t->chuck.lr_scale == 1,
          "custom sensing crosses rings/macros without executing legacy controls");
    CHECK(action(NT_CHUCK_ACTION_LEGACY, 0, 1) == 0 && t->chuck.stag == 0 &&
          t->chuck.noise == NT_CHUCK_NOISE_MAG, "legacy consumes accumulated stagnation");
    for (int i = 0; i < 999; i++) CHECK(action(NT_CHUCK_ACTION_LEGACY, 0, 1) == 0,
                                      "legacy next macro window");
    CHECK(t->chuck.global_step == 5000 && t->chuck.lr_scale == .5f && t->chuck.macro_stag == 0,
          "macro patience survives the mode boundary");
    t->chuck.global_step = INT_MAX;
    snapshot before = capture(1); nt_chuck_observation o;
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == NT_CHUCK_E_STATE && unchanged(&before, 1),
          "global counter overflow refused");
    CHECK(nt_tape_chuck_observe(1, &o) == NT_CHUCK_E_STATE,
          "overflowing pending observation refused");
    open_body(1); t = nt_tape_get();
    t->entries[0].grad->data[0] = t->entries[0].grad->data[1] = .0001f;
    for (int i = 0; i < 15; i++) CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0,
                                     "small-gradient freeze trajectory");
    CHECK(t->chuck_params[0].frozen == 1 && t->adam[0].t == 15,
          "freezing occurs at the canonical local boundary");
    before = capture(1);
    CHECK(action(NT_CHUCK_ACTION_HOLD, 0, 1) == 0 &&
          memcmp(before.weight[0], weights[0]->data, sizeof before.weight[0]) == 0 &&
          t->adam[0].t == 15, "the next frozen step leaves weights and moments fixed");
    close_body(); return 1;
}

int main(void) {
    struct { const char *name; int (*run)(void); } cases[] = {
        {"empty-and-frozen", empty_and_frozen}, {"gradient-refusals", gradient_refusals},
        {"local-state-refusals", local_state_refusals}, {"controls-and-noise", controls_and_noise},
        {"boundary-counters", boundary_counters}
    };
    int passed = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        group = cases[i].name;
        int ok = cases[i].run();
        printf("%s %s\n", ok ? "PASS" : "FAIL", group);
        passed += ok; close_body();
    }
    printf("CHUCK_ACTIONS_EDGE groups=%d/%zu assertions=%d\n", passed,
           sizeof cases / sizeof *cases, assertions);
    return passed == (int)(sizeof cases / sizeof *cases) ? 0 : 1;
}
