// Chuck conditioned future credit and weight-frozen completion contracts.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
#include "chuck_architect.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *group;
static int checks;
#define CHECK(c, why) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", group, __LINE__, why); return 0; \
} } while (0)
#define NEAR(a, b) (isfinite(a) && fabs((double)(a) - (double)(b)) <= 1e-7 + 2e-6 * fabs((double)(b)))

static int new_life(nt_chuck_architect *a) {
    nt_chuck_architect_config c;
    nt_chuck_architect_config_default(&c);
    c.mode = NT_CHUCK_ARCHITECT_LEARNED; c.seed = 73;
    c.learning_rate = .05f; c.exploration = 0;
    strcpy(c.life_id, "conditioned-credit-gate");
    return nt_chuck_architect_init(a, &c);
}

static void clear_weights(nt_chuck_architect *a) {
    memset(a->w1, 0, sizeof a->w1); memset(a->b1, 0, sizeof a->b1);
    memset(a->w2, 0, sizeof a->w2); memset(a->b2, 0, sizeof a->b2);
}

static int only_weights_changed(const nt_chuck_architect *before, const nt_chuck_architect *after) {
    nt_chuck_architect a = *before, b = *after;
    clear_weights(&a); clear_weights(&b);
    return memcmp(&a, &b, sizeof a) == 0;
}

static int same_weights(const nt_chuck_architect *a, const nt_chuck_architect *b) {
    return !memcmp(a->w1, b->w1, sizeof a->w1) && !memcmp(a->b1, b->b1, sizeof a->b1) &&
           !memcmp(a->w2, b->w2, sizeof a->w2) && !memcmp(a->b2, b->b2, sizeof a->b2);
}

static nt_chuck_architect_comparison sample(float hold, float brake, float push) {
    nt_chuck_architect_comparison s;
    memset(&s, 0, sizeof s);
    s.features[0] = .5f; s.features[3] = -.25f;
    s.future_loss[0] = hold; s.future_loss[1] = brake; s.future_loss[2] = push;
    return s;
}

static int conditioned_scale(void) {
    nt_chuck_architect a, original;
    CHECK(new_life(&a) == 0, "initialize scale life"); original = a;
    nt_chuck_architect_comparison s = sample(2, 1.75f, 2.5f);
    nt_chuck_architect_conditioned_receipt r;
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0, "fit unequal action effects");
    CHECK(r.scale == .5 && r.comparison.target[0] == 0 && r.comparison.target[1] == .5f &&
          r.comparison.target[2] == -1, "scale is the largest absolute finite action consequence");
    CHECK(r.comparison.loss_delta[1] == .25 && r.comparison.loss_delta[2] == -.5 &&
          !memcmp(r.comparison.future_loss, s.future_loss, sizeof s.future_loss),
          "raw consequences survive target scaling");
    CHECK(r.comparison.fitted && r.comparison.hash_before == nt_chuck_architect_hash(&original) &&
          r.comparison.hash_after == nt_chuck_architect_hash(&a), "receipt binds actual before and after lives");
    CHECK(only_weights_changed(&original, &a) && r.comparison.decisions == original.decisions &&
          r.comparison.updates == original.updates, "fit leaves online chronology intact");

    CHECK(new_life(&a) == 0, "reset scale invariance life");
    nt_chuck_architect b = a;
    nt_chuck_architect_comparison scaled = sample(16, 14, 20);
    nt_chuck_architect_conditioned_receipt q;
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 &&
          nt_chuck_architect_fit_conditioned(&b, &scaled, &q) == 0, "fit exact eightfold loss rescaling");
    CHECK(q.scale == 8 * r.scale && !memcmp(q.comparison.target, r.comparison.target, sizeof r.comparison.target) &&
          !memcmp(&a, &b, sizeof a), "above floor, common positive loss scaling preserves the exact update");

    s = sample(1, nextafterf(1, 0), nextafterf(1, 2));
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0, "fit sub-floor distinct outcomes");
    CHECK(r.scale == 2e-6 && NEAR(r.comparison.target[1], (1.0 - s.future_loss[1]) / 2e-6) &&
          NEAR(r.comparison.target[2], (1.0 - s.future_loss[2]) / 2e-6), "floor preserves tiny effect magnitude");
    s = sample(20, 20, 20);
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 && r.scale == 21e-6 &&
          r.comparison.target[0] == 0 && r.comparison.target[1] == 0 && r.comparison.target[2] == 0,
          "finite ties have zero targets and a positive documented floor");
    s = sample(FLT_MAX, -FLT_MAX, 0);
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 && isfinite(r.scale) &&
          r.scale == 2.0 * FLT_MAX && r.comparison.target[1] == 1 && r.comparison.target[2] == .5f,
          "double differences and scale retain finite float extremes");

    nt_chuck_architect_comparison_receipt legacy;
    s = sample(2, 1.75f, 2.5f);
    CHECK(new_life(&a) == 0 && nt_chuck_architect_fit_comparison(&a, &s, &legacy) == 0,
          "original comparison API remains available");
    CHECK(NEAR(legacy.target[1], .25 / 2.000001) && NEAR(legacy.target[2], -.5 / 2.000001),
          "original comparison retains its original relative-HOLD target");
    return 1;
}

static int conditioned_choice(void) {
    // Measured development H16 future-window losses: SimpleLLM, seed 42,
    // snapshots 1 (6e32e6eba49d00cb) and 128 (e6f78cd924d80727), from
    // experiments/chuck_loss_architect/future/receipts.json. Feature vectors
    // below are explicitly synthetic, opposite unit coordinates. Real captured
    // features and unseen-seed performance belong to the integration experiment.
    nt_chuck_architect_comparison early = sample(3.48037958f, 3.49617457f, 3.46477795f);
    nt_chuck_architect_comparison late = sample(2.87523842f, 2.87194562f, 2.87865996f);
    memset(early.features, 0, sizeof early.features); early.features[0] = 1;
    memset(late.features, 0, sizeof late.features); late.features[0] = -1;
    nt_chuck_architect a;
    CHECK(new_life(&a) == 0, "initialize conditional life");
    nt_chuck_architect before = a;
    nt_chuck_architect_conditioned_receipt r;
    for (int epoch = 0; epoch < 512; epoch++) {
        CHECK(nt_chuck_architect_fit_conditioned(&a, &early, &r) == 0,
              "fit measured early PUSH preference");
        CHECK(r.comparison.target[2] > 0 && r.comparison.target[1] == -1,
              "early measured targets favor PUSH and penalize BRAKE");
        CHECK(nt_chuck_architect_fit_conditioned(&a, &late, &r) == 0,
              "fit measured late BRAKE preference");
        CHECK(r.comparison.target[1] > 0 && r.comparison.target[2] == -1,
              "late measured targets favor BRAKE and penalize PUSH");
    }
    float x[3], y[3];
    CHECK(nt_chuck_architect_scores(&a, early.features, x) == 0 &&
          nt_chuck_architect_scores(&a, late.features, y) == 0, "read both conditioned states");
    CHECK(x[2] > x[0] && x[0] > x[1] && y[1] > y[0] && y[0] > y[2],
          "one acquired life preserves opposite future preferences in distinct states");
    CHECK(only_weights_changed(&before, &a), "conditional fitting changes only weights");
    printf("CONDITIONED_CHOICES early=%.9g,%.9g,%.9g late=%.9g,%.9g,%.9g\n",
           x[0], x[1], x[2], y[0], y[1], y[2]);
    return 1;
}

static int conditioned_refused(nt_chuck_architect *a, const nt_chuck_architect_comparison *s, int expected) {
    nt_chuck_architect before = *a;
    nt_chuck_architect_conditioned_receipt r, sentinel;
    memset(&r, 0x3a, sizeof r); sentinel = r;
    int status = nt_chuck_architect_fit_conditioned(a, s, &r);
    return status == expected && !memcmp(a, &before, sizeof before) && !memcmp(&r, &sentinel, sizeof r);
}

static int conditioned_refusals(void) {
    nt_chuck_architect a;
    CHECK(new_life(&a) == 0, "initialize refusal life");
    nt_chuck_architect_comparison s = sample(2, 1, 3);
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        s.future_loss[0] = bad[i];
        CHECK(conditioned_refused(&a, &s, NT_CHUCK_E_BASELINE), "nonfinite baseline refuses transactionally");
    }
    s = sample(2, 1, 3);
    for (int i = 0; i < 5; i++) {
        s.features[2] = i < 3 ? bad[i] : i == 3 ? 1.01f : -1.01f;
        CHECK(conditioned_refused(&a, &s, NT_CHUCK_E_ARGUMENT), "invalid feature refuses transactionally");
    }
    s = sample(2, 1, 3);
    for (int k = 0; k < 3; k++) {
        CHECK(new_life(&a) == 0, "reset action mask");
        a.config.limits.enabled_actions &= ~NT_CHUCK_ACTION_BIT(k + NT_CHUCK_ACTION_HOLD);
        // HOLD is required by the life schema itself; BRAKE/PUSH by this fit.
        CHECK(conditioned_refused(&a, &s, k ? NT_CHUCK_E_ACTION : NT_CHUCK_E_STATE),
              "unavailable action cannot receive conditioned credit");
    }
    for (int mode = NT_CHUCK_ARCHITECT_DISABLED; mode <= NT_CHUCK_ARCHITECT_LEGACY; mode++) {
        CHECK(new_life(&a) == 0, "reset mode"); a.config.mode = mode;
        CHECK(conditioned_refused(&a, &s, NT_CHUCK_E_STATE), "nonlearned mode refuses fit transactionally");
    }
    CHECK(new_life(&a) == 0, "reset null sample life");
    CHECK(conditioned_refused(&a, NULL, NT_CHUCK_E_ARGUMENT), "null sample refuses transactionally");
    a.version = 0;
    CHECK(conditioned_refused(&a, &s, NT_CHUCK_E_STATE), "invalid saved-life schema refuses transactionally");
    return 1;
}

static int nonfinite_alternatives(void) {
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int k = 1; k < 3; k++) for (int i = 0; i < 3; i++) {
        nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize failed-alternative life");
        nt_chuck_architect_comparison s = sample(2, 1.75f, 2.5f);
        s.future_loss[k] = bad[i];
        nt_chuck_architect_conditioned_receipt r;
        CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0,
              "nonfinite alternative supplies explicit failure credit");
        CHECK(r.comparison.nonfinite[k] && r.comparison.target[k] == -1 &&
              !isfinite(r.comparison.future_loss[k]) && isfinite(r.scale) && r.scale == (k == 1 ? .5 : .25),
              "failed branch is recorded and excluded from finite scale");
        CHECK(r.comparison.predicted_after[k] < r.comparison.predicted_before[k],
              "failed alternative loses preference");
    }
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize two-failure life");
    nt_chuck_architect_comparison s = sample(2, NAN, INFINITY);
    nt_chuck_architect_conditioned_receipt r;
    CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 && r.scale == 3e-6 &&
          r.comparison.target[0] == 0 && r.comparison.target[1] == -1 && r.comparison.target[2] == -1,
          "two failed alternatives leave a finite floor and two explicit negative targets");
    return 1;
}

typedef struct {
    float weights[2], gradient[2], m[2], v[2], accumulated[2];
    nt_chuck_state global;
    nt_chuck_param_state local;
    int adam_t, count, active, n_params, frozen, no_decay;
    uint32_t noise_rng;
} body_image;
static nt_tensor *body;
static int body_index;

static void body_close(void) { nt_tape_destroy(); nt_tensor_free(body); body = NULL; }
static void body_open(void) {
    body_close(); nt_seed(42); nt_chuck_rng_set(2463534242u); nt_tape_start();
    body = nt_tensor_new(2); body->data[0] = .75f; body->data[1] = -1.25f;
    body_index = nt_tape_param(body);
    nt_tape_get()->entries[body_index].grad = nt_tensor_new(2);
}
static float body_loss(void) { return .5f * (body->data[0] * body->data[0] + body->data[1] * body->data[1]); }
static void body_gradient(void) {
    memcpy(nt_tape_get()->entries[body_index].grad->data, body->data, 2 * sizeof(float));
}
static body_image body_capture(void) {
    body_image im; memset(&im, 0, sizeof im);
    nt_tape *t = nt_tape_get(); nt_adam_state *as = &t->adam[0];
    memcpy(im.weights, body->data, sizeof im.weights);
    memcpy(im.gradient, t->entries[body_index].grad->data, sizeof im.gradient);
    memcpy(im.m, as->m->data, sizeof im.m); memcpy(im.v, as->v->data, sizeof im.v);
    if (as->acc_grad) memcpy(im.accumulated, as->acc_grad->data, sizeof im.accumulated);
    im.global = t->chuck; im.local = t->chuck_params[0]; im.adam_t = as->t;
    im.count = t->count; im.active = t->active; im.n_params = t->n_params;
    im.frozen = t->entries[body_index].frozen; im.no_decay = t->entries[body_index].no_decay;
    im.noise_rng = nt_chuck_rng_get(); return im;
}
static void body_restore(const body_image *im) {
    body_open(); nt_tape *t = nt_tape_get(); nt_adam_state *as = &t->adam[0];
    memcpy(body->data, im->weights, sizeof im->weights);
    memcpy(t->entries[body_index].grad->data, im->gradient, sizeof im->gradient);
    memcpy(as->m->data, im->m, sizeof im->m); memcpy(as->v->data, im->v, sizeof im->v);
    if (as->acc_grad) memcpy(as->acc_grad->data, im->accumulated, sizeof im->accumulated);
    t->chuck = im->global; t->chuck_params[0] = im->local; as->t = im->adam_t;
    t->count = im->count; t->active = im->active; t->n_params = im->n_params;
    t->entries[body_index].frozen = im->frozen; t->entries[body_index].no_decay = im->no_decay;
    nt_chuck_rng_set(im->noise_rng);
}

static int frozen_feedback(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize frozen execution life");
    nt_chuck_architect initial = a, resumed = a;
    char path[] = "/tmp/notorch-conditioned-XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0, "allocate checkpoint path"); close(fd);
    body_open();
    for (int step = 0; step < 32; step++) {
        body_gradient(); body_image before = body_capture(); float loss = body_loss();
        nt_chuck_architect_decision d, e;
        CHECK(nt_chuck_architect_step(&a, .01f, loss, &d) == 0, "execute frozen life's real training step");
        body_image after = body_capture(); float following = body_loss();
        body_restore(&before);
        CHECK(nt_chuck_architect_step(&resumed, .01f, loss, &e) == 0, "execute continuation from identical body state");
        body_image resumed_body = body_capture();
        CHECK(!memcmp(&after, &resumed_body, sizeof after) && !memcmp(&d, &e, sizeof d) &&
              !memcmp(&a, &resumed, sizeof a), "saved-life continuation reproduces action and full body state");
        if (step == 5) {
            CHECK(nt_chuck_architect_save(&a, path) == 0 && nt_chuck_architect_load(&resumed, path) == 0 &&
                  !memcmp(&a, &resumed, sizeof a), "pending frozen experience survives v1 save/load");
        }
        nt_chuck_architect ordinary = a;
        nt_chuck_architect_receipt r, q, learned;
        CHECK(nt_chuck_architect_feedback(&ordinary, following, &learned) == 0 && learned.learned == 1,
              "ordinary same-window feedback still learns");
        CHECK(nt_chuck_architect_feedback_frozen(&a, following, &r) == 0 &&
              nt_chuck_architect_feedback_frozen(&resumed, following, &q) == 0,
              "frozen completion accepts actual same-window consequence");
        CHECK(same_weights(&initial, &a) && r.learned == 0 && r.error == 0,
              "frozen feedback preserves all 163 weights and reports no learning or regression error");
        CHECK(only_weights_changed(&a, &ordinary) && !same_weights(&a, &ordinary),
              "history and counters match ordinary feedback while ordinary weights actually change");
        learned.learned = 0; learned.error = 0;
        CHECK(!memcmp(&r, &learned, sizeof r) && !memcmp(&r, &q, sizeof r) &&
              !memcmp(&a, &resumed, sizeof a), "raw consequence fields match ordinary feedback and deterministic continuation");
        CHECK(a.updates == (uint64_t)step + 1 && a.decisions == a.updates && !a.pending && a.has_history &&
              a.prev_loss == loss && a.prev_reward == r.reward && a.prev_trend == d.observation.loss_trend,
              "frozen experience advances complete temporal state");
        if (step == 12) {
            CHECK(nt_chuck_architect_save(&a, path) == 0 && nt_chuck_architect_load(&resumed, path) == 0 &&
                  !memcmp(&a, &resumed, sizeof a), "completed frozen history survives v1 save/load");
        }
    }
    unlink(path); body_close();
    CHECK(a.prev_loss != initial.prev_loss && a.updates == 32, "frozen life accumulates real training history");
    return 1;
}

static int frozen_refused(nt_chuck_architect *a, float loss) {
    nt_chuck_architect before = *a;
    nt_chuck_architect_receipt r, sentinel;
    memset(&r, 0x3a, sizeof r); sentinel = r;
    int result = nt_chuck_architect_feedback_frozen(a, loss, &r);
    return result == NT_CHUCK_E_STATE && !memcmp(a, &before, sizeof before) && !memcmp(&r, &sentinel, sizeof r);
}

static int frozen_refusals_and_nonfinite(void) {
    nt_chuck_architect a;
    CHECK(new_life(&a) == 0, "initialize frozen refusal life");
    CHECK(frozen_refused(&a, 1), "no pending execution means no eligible completion");
    body_open(); body_gradient();
    CHECK(nt_chuck_architect_step(&a, .01f, body_loss(), NULL) == 0, "create actual pending credit");
    nt_chuck_architect_comparison s = sample(2, 1, 3);
    CHECK(conditioned_refused(&a, &s, NT_CHUCK_E_STATE), "pending immediate consequence cannot be overwritten by replay fitting");
    nt_chuck_architect broken = a; broken.pending_decision.scores[0] += 1;
    CHECK(frozen_refused(&broken, 1), "incoherent pending cache refuses transactionally");
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        nt_chuck_architect b = a, ordinary = a;
        nt_chuck_architect_receipt r, q;
        CHECK(nt_chuck_architect_feedback_frozen(&b, bad[i], &r) == 0 &&
              nt_chuck_architect_feedback(&ordinary, bad[i], &q) == 0, "complete measured nonfinite outcome");
        CHECK(r.nonfinite && r.reward == -1 && !r.learned && b.prev_reward == -1 &&
              same_weights(&a, &b) && only_weights_changed(&b, &ordinary),
              "nonfinite outcome advances failure history without touching weights");
        CHECK(frozen_refused(&b, 1), "completed failure cannot be credited twice");
    }
    CHECK(nt_chuck_architect_feedback_frozen(&a, body_loss(), NULL) == 0,
          "receipt is optional on successful completion");
    body_close(); return 1;
}

static int deterministic_fit_resume(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize deterministic replay life");
    nt_chuck_architect b = a;
    nt_chuck_architect_comparison s = sample(2, 1.75f, 2.5f);
    for (int i = 0; i < 16; i++) {
        nt_chuck_architect_conditioned_receipt r, q;
        CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 &&
              nt_chuck_architect_fit_conditioned(&b, &s, &q) == 0 &&
              !memcmp(&a, &b, sizeof a) && !memcmp(&r, &q, sizeof r), "deterministic complete fit and receipt bytes");
    }
    char path[] = "/tmp/notorch-conditioned-fit-XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0, "allocate fit checkpoint path"); close(fd);
    CHECK(nt_chuck_architect_save(&a, path) == 0 && nt_chuck_architect_load(&b, path) == 0 &&
          !memcmp(&a, &b, sizeof a), "conditioned weights preserve the existing v1 life schema");
    unlink(path);
    for (int i = 0; i < 32; i++) {
        nt_chuck_architect_conditioned_receipt r, q;
        s.features[1] = (i % 3 - 1) * .25f;
        CHECK(nt_chuck_architect_fit_conditioned(&a, &s, &r) == 0 &&
              nt_chuck_architect_fit_conditioned(&b, &s, &q) == 0 &&
              !memcmp(&a, &b, sizeof a) && !memcmp(&r, &q, sizeof r), "save/resume reproduces changing-state replay continuation");
    }
    return 1;
}

int main(int argc, char **argv) {
    struct { const char *name; int (*run)(void); } cases[] = {
        {"conditioned_scale", conditioned_scale}, {"conditioned_choice", conditioned_choice},
        {"conditioned_refusals", conditioned_refusals}, {"nonfinite_alternatives", nonfinite_alternatives},
        {"frozen_feedback", frozen_feedback}, {"frozen_refusals_and_nonfinite", frozen_refusals_and_nonfinite},
        {"deterministic_fit_resume", deterministic_fit_resume}
    };
    int passed = 0, ran = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (argc > 1 && strcmp(argv[1], cases[i].name)) continue;
        group = cases[i].name; ran++;
        int ok = cases[i].run(); passed += ok;
        printf("%s %s\n", ok ? "PASS" : "FAIL", group);
    }
    printf("CHUCK_CONDITIONED groups=%d/%d checks=%d\n", passed, ran, checks);
    return ran && passed == ran ? 0 : 1;
}
