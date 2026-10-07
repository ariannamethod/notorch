// A measured intervention joins Chuck's actual history as a completed action.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
#include "chuck_architect.h"
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

static nt_tensor *p;
static int param;
typedef struct {
    float weights[2], gradient[2], m[2], v[2], accumulated[2];
    nt_chuck_state global;
    nt_chuck_param_state local;
    int moment_step, count, active, n_params, frozen, no_decay, training;
    uint32_t noise_rng;
} body_image;

static void body_close(void) { nt_tape_destroy(); nt_tensor_free(p); p = NULL; }
static void body_open(void) {
    body_close(); nt_seed(42); nt_chuck_rng_set(2463534242u); nt_tape_start(); nt_train_mode(1);
    p = nt_tensor_new(2); p->data[0] = .75f; p->data[1] = -1.25f;
    param = nt_tape_param(p); nt_tape_get()->entries[param].grad = nt_tensor_new(2);
}
static float loss(void) { return .5f * (p->data[0] * p->data[0] + p->data[1] * p->data[1]); }
static void gradient(void) { memcpy(nt_tape_get()->entries[param].grad->data, p->data, 2 * sizeof(float)); }
static body_image body_capture(void) {
    body_image im; memset(&im, 0, sizeof im);
    nt_tape *t = nt_tape_get(); nt_adam_state *m = &t->adam[0];
    memcpy(im.weights, p->data, sizeof im.weights);
    memcpy(im.gradient, t->entries[param].grad->data, sizeof im.gradient);
    memcpy(im.m, m->m->data, sizeof im.m); memcpy(im.v, m->v->data, sizeof im.v);
    if (m->acc_grad) memcpy(im.accumulated, m->acc_grad->data, sizeof im.accumulated);
    im.global = t->chuck; im.local = t->chuck_params[0]; im.moment_step = m->t;
    im.count = t->count; im.active = t->active; im.n_params = t->n_params;
    im.frozen = t->entries[param].frozen; im.no_decay = t->entries[param].no_decay;
    im.noise_rng = nt_chuck_rng_get(); im.training = nt_is_training(); return im;
}
static void body_restore(const body_image *im) {
    body_open(); nt_tape *t = nt_tape_get(); nt_adam_state *m = &t->adam[0];
    memcpy(p->data, im->weights, sizeof im->weights);
    memcpy(t->entries[param].grad->data, im->gradient, sizeof im->gradient);
    memcpy(m->m->data, im->m, sizeof im->m); memcpy(m->v->data, im->v, sizeof im->v);
    if (m->acc_grad) memcpy(m->acc_grad->data, im->accumulated, sizeof im->accumulated);
    t->chuck = im->global; t->chuck_params[0] = im->local; m->t = im->moment_step;
    t->count = im->count; t->active = im->active; t->n_params = im->n_params;
    t->entries[param].frozen = im->frozen; t->entries[param].no_decay = im->no_decay;
    nt_chuck_rng_set(im->noise_rng); nt_train_mode(im->training);
}
static int same_weights(const nt_chuck_architect *a, const nt_chuck_architect *b) {
    return !memcmp(a->w1, b->w1, sizeof a->w1) && !memcmp(a->b1, b->b1, sizeof a->b1) &&
           !memcmp(a->w2, b->w2, sizeof a->w2) && !memcmp(a->b2, b->b2, sizeof a->b2);
}
static int new_life(nt_chuck_architect *a) {
    nt_chuck_architect_config c; nt_chuck_architect_config_default(&c);
    c.mode = NT_CHUCK_ARCHITECT_LEARNED; c.seed = 73; c.exploration = 0;
    strcpy(c.life_id, "lived-intervention-gate");
    int rc = nt_chuck_architect_init(a, &c);
    if (!rc) a->b2[2] = .5f;
    return rc;
}

typedef struct {
    int calls, override;
    float outcome;
    const nt_chuck_architect *life;
    uint64_t source_hash;
    int source_unchanged;
} callback_state;
static float after(void *context) {
    callback_state *c = context; ++c->calls;
    if (c->life) c->source_unchanged = !c->life->pending && nt_chuck_architect_hash(c->life) == c->source_hash;
    return c->override ? c->outcome : loss();
}

static int selected_equivalence(void) {
    nt_chuck_architect ordinary;
    CHECK(new_life(&ordinary) == 0, "initialize selected policy");
    nt_chuck_architect initial = ordinary, forced = ordinary;
    body_open();
    for (int step = 0; step < 32; step++) {
        gradient(); body_image source = body_capture(); float before = loss();
        nt_chuck_architect_decision chosen, intervention;
        nt_chuck_architect_receipt a, b;
        CHECK(nt_chuck_architect_step(&ordinary, .01f, before, &chosen) == 0 &&
              nt_chuck_architect_feedback_frozen(&ordinary, loss(), &a) == 0,
              "ordinary learned step and frozen consequence execute");
        body_image expected = body_capture();
        body_restore(&source);
        callback_state c = {.life = &forced, .source_hash = nt_chuck_architect_hash(&forced)};
        CHECK(nt_chuck_architect_intervene(&forced, .01f, before, &chosen.action, after, &c,
                                          &intervention, &b) == 0 && c.calls == 1,
              "same selected action executes one measured callback");
        CHECK(c.source_unchanged, "observer sees the original completed life, never an off-greedy pending cache");
        body_image actual = body_capture();
        CHECK(!memcmp(&expected, &actual, sizeof actual), "selected intervention reproduces complete optimizer/body/RNG state");
        CHECK(!memcmp(&ordinary, &forced, sizeof forced) && !memcmp(&a, &b, sizeof a) &&
              !memcmp(&chosen, &intervention, sizeof chosen), "selected intervention reproduces native decision, consequence and life bytes");
        CHECK(same_weights(&initial, &forced) && !forced.pending && !b.learned && b.error == 0 &&
              forced.decisions == (uint64_t)step + 1 && forced.updates == forced.decisions,
              "completed experience advances chronology with all 163 weights frozen");
    }
    body_close(); return 1;
}

static int unselected_resume(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize off-greedy intervention life");
    nt_chuck_architect initial = a;
    body_open(); gradient(); float before = loss();
    nt_chuck_observation observation;
    float features[NT_CHUCK_ARCHITECT_FEATURES], scores[NT_CHUCK_ARCHITECT_HEADS];
    CHECK(nt_tape_chuck_observe(before, &observation) == 0 &&
          nt_chuck_architect_capture(&a, &observation, features) == 0 &&
          nt_chuck_architect_scores(&a, features, scores) == 0, "capture native source observation and acquired scores");
    nt_chuck_action action = {NT_CHUCK_ACTION_BRAKE, 0}, greedy;
    CHECK(nt_chuck_architect_select(&a, &observation, &greedy) == 0 && greedy.kind == NT_CHUCK_ACTION_PUSH,
          "fixture intervenes against a known greedy PUSH choice");
    callback_state c = {.life = &a, .source_hash = nt_chuck_architect_hash(&a)};
    nt_chuck_architect_decision d; nt_chuck_architect_receipt r;
    CHECK(nt_chuck_architect_intervene(&a, .01f, before, &action, after, &c, &d, &r) == 0 && c.calls == 1,
          "explicit BRAKE intervention obtains one actual consequence");
    CHECK(c.source_unchanged, "off-greedy action does not publish fabricated pending state to its observer");
    CHECK(d.action.kind == NT_CHUCK_ACTION_BRAKE && !d.explored &&
          !memcmp(d.features, features, sizeof features) && !memcmp(d.scores, scores, sizeof scores),
          "forced action retains real pre-action scores and features without fabricated exploration");
    CHECK(!a.pending && a.updates == 1 && a.decisions == 1 && a.has_history && same_weights(&initial, &a) &&
          a.prev_loss == before && a.prev_reward == r.reward && a.prev_trend == observation.loss_trend &&
          a.pending_decision.action.kind == NT_CHUCK_ACTION_BRAKE && r.predicted == scores[1],
          "completed cache names the actual action and temporal history receives its actual consequence");
    char path[] = "/tmp/notorch-lived-life-XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0, "allocate persisted intervention life"); close(fd);
    nt_chuck_architect resumed;
    CHECK(nt_chuck_architect_save(&a, path) == 0 && nt_chuck_architect_load(&resumed, path) == 0 &&
          !memcmp(&a, &resumed, sizeof a), "off-greedy completed cache persists in the unchanged v1 format");
    unlink(path);
    for (int i = 0; i < 16; i++) {
        gradient(); body_image source = body_capture(); before = loss();
        nt_chuck_architect_decision x, y; nt_chuck_architect_receipt rx, ry;
        CHECK(nt_chuck_architect_step(&a, .01f, before, &x) == 0 &&
              nt_chuck_architect_feedback_frozen(&a, loss(), &rx) == 0, "ordinary learned continuation after an intervention");
        body_image expected = body_capture(); body_restore(&source);
        CHECK(nt_chuck_architect_step(&resumed, .01f, before, &y) == 0 &&
              nt_chuck_architect_feedback_frozen(&resumed, loss(), &ry) == 0,
              "restored intervention joins the same ordinary continuation");
        body_image actual = body_capture();
        CHECK(!memcmp(&a, &resumed, sizeof a) && !memcmp(&x, &y, sizeof x) && !memcmp(&rx, &ry, sizeof rx) &&
              !memcmp(&expected, &actual, sizeof actual), "all subsequent decisions and body/history bytes resume exactly");
    }
    body_close(); return 1;
}

static int refused(nt_chuck_architect *a, float lr, float before, const nt_chuck_action *action,
                   nt_chuck_architect_after_fn callback) {
    nt_chuck_architect prior = *a;
    body_image image = body_capture(); callback_state c = {0};
    nt_chuck_architect_decision d, d0; nt_chuck_architect_receipt r, r0;
    memset(&d, 0x3a, sizeof d); d0 = d; memset(&r, 0x5c, sizeof r); r0 = r;
    int rc = nt_chuck_architect_intervene(a, lr, before, action, callback, &c, &d, &r);
    body_image now = body_capture();
    return rc != 0 && !c.calls && !memcmp(a, &prior, sizeof prior) && !memcmp(&image, &now, sizeof now) &&
           !memcmp(&d, &d0, sizeof d) && !memcmp(&r, &r0, sizeof r);
}

static int refusals(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize refusal life");
    body_open(); gradient();
    nt_chuck_action action = {NT_CHUCK_ACTION_PUSH, 0};
    CHECK(refused(&a, .01f, loss(), NULL, after), "missing action refuses before action and callback");
    CHECK(refused(&a, .01f, loss(), &action, NULL), "missing observer refuses before action");
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        CHECK(refused(&a, bad[i], loss(), &action, after), "nonfinite learning rate refuses transactionally");
        CHECK(refused(&a, .01f, bad[i], &action, after), "nonfinite source observation refuses transactionally");
        action.value = bad[i];
        CHECK(refused(&a, .01f, loss(), &action, after), "nonfinite action value refuses transactionally");
    }
    action.value = 1;
    CHECK(refused(&a, .01f, loss(), &action, after), "primitive intervention requires canonical zero value");
    action.value = 0;
    const int kinds[] = {NT_CHUCK_ACTION_LEGACY, NT_CHUCK_ACTION_SET_DAMPEN, -1, NT_CHUCK_ACTION_COUNT};
    for (unsigned i = 0; i < sizeof kinds / sizeof *kinds; i++) {
        action.kind = (nt_chuck_action_kind)kinds[i];
        CHECK(refused(&a, .01f, loss(), &action, after), "unavailable intervention vocabulary refuses transactionally");
    }
    action.kind = NT_CHUCK_ACTION_PUSH;
    a.config.limits.enabled_actions &= ~NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_PUSH);
    CHECK(refused(&a, .01f, loss(), &action, after), "disabled action never executes");
    CHECK(new_life(&a) == 0, "reset life after mask case");
    for (int mode = NT_CHUCK_ARCHITECT_DISABLED; mode <= NT_CHUCK_ARCHITECT_LEGACY; mode++) {
        a.config.mode = mode;
        CHECK(refused(&a, .01f, loss(), &action, after), "synchronous intervention requires a learned life");
    }
    CHECK(new_life(&a) == 0, "reset life after modes");
    a.version = 0;
    CHECK(refused(&a, .01f, loss(), &action, after), "malformed life refuses transactionally");
    CHECK(new_life(&a) == 0, "reset life after schema");
    a.decisions = a.updates = UINT64_MAX; a.has_history = 1;
    CHECK(refused(&a, .01f, loss(), &action, after), "chronology overflow refuses before callback");
    CHECK(new_life(&a) == 0, "reset life after counter limit");
    CHECK(nt_chuck_architect_step(&a, .01f, loss(), NULL) == 0, "create actual pending native decision");
    CHECK(refused(&a, .01f, loss(), &action, after), "intervention cannot replace outstanding native credit");
    CHECK(nt_chuck_architect_feedback_frozen(&a, loss(), NULL) == 0, "complete original pending consequence");
    nt_tape_get()->entries[param].grad->data[0] = NAN;
    CHECK(refused(&a, .01f, loss(), &action, after), "nonfinite gradient refuses before callback");
    body_close(); return 1;
}

static int nonfinite_consequences(void) {
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize failed-outcome life");
        nt_chuck_architect initial = a; body_open(); gradient(); float before = loss();
        nt_chuck_action action = {NT_CHUCK_ACTION_HOLD, 0};
        callback_state c = {.override = 1, .outcome = bad[i]}; nt_chuck_architect_receipt r;
        CHECK(nt_chuck_architect_intervene(&a, .01f, before, &action, after, &c, NULL, &r) == 0 && c.calls == 1,
              "actual nonfinite outcome still completes the executed intervention once");
        CHECK(r.nonfinite && r.reward == -1 && !r.learned && r.error == 0 && !a.pending &&
              a.prev_reward == -1 && a.prev_loss == before && a.decisions == 1 && a.updates == 1 &&
              same_weights(&initial, &a), "failure receives native negative history while weights stay frozen");
        CHECK(nt_chuck_architect_hash(&a) != 0, "completed failed consequence remains a valid saved life");
    }
    body_close(); return 1;
}

// Explicitly synthetic history-sensitive policy for real-body diagnostic gates.
// Positive measured previous reward changes the BRAKE/PUSH score ordering.
static int write_life(const char *path) {
    nt_chuck_architect a; if (new_life(&a)) return 0;
    memset(a.w1, 0, sizeof a.w1); memset(a.b1, 0, sizeof a.b1);
    memset(a.w2, 0, sizeof a.w2); memset(a.b2, 0, sizeof a.b2);
    a.w1[0][15] = 16; a.w2[1][0] = -1; a.w2[2][0] = 1;
    a.b2[0] = -.2f; a.b2[1] = .01f; a.b2[2] = -.01f;
    return nt_chuck_architect_save(&a, path) == 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--write-life")) return write_life(argv[2]) ? 0 : 1;
    struct { const char *name; int (*run)(void); } cases[] = {
        {"selected_equivalence", selected_equivalence}, {"unselected_resume", unselected_resume},
        {"refusals", refusals}, {"nonfinite_consequences", nonfinite_consequences}
    };
    int passed = 0, ran = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (argc > 1 && strcmp(argv[1], cases[i].name)) continue;
        group = cases[i].name; ran++;
        int ok = cases[i].run(); passed += ok;
        printf("%s %s\n", ok ? "PASS" : "FAIL", group);
    }
    printf("CHUCK_LIVED groups=%d/%d checks=%d\n", passed, ran, checks);
    return ran && passed == ran ? 0 : 1;
}
