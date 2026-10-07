// Future comparison credit: measured reversal, fixed-state replay and v1 continuity.
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

static int new_life(nt_chuck_architect *a) {
    nt_chuck_architect_config config;
    nt_chuck_architect_config_default(&config);
    config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    config.seed = 73;
    config.learning_rate = .05f;
    config.exploration = 0;
    strcpy(config.life_id, "future-credit-gate");
    return nt_chuck_architect_init(a, &config);
}

// Uses only the pre-comparison API so this exact emitter also compiles against
// pinned 48512d8. Three complete v1 files cover initialization, pending credit,
// and completed online feedback; the mutation runner compares their bytes.
static int write_v1(const char *directory) {
    nt_chuck_architect a;
    char path[1024];
    if (new_life(&a)) return 0;
    snprintf(path, sizeof path, "%s/init.bin", directory);
    if (nt_chuck_architect_save(&a, path)) return 0;
    nt_tape_destroy(); nt_tape_start(); nt_chuck_rng_set(12345);
    nt_tensor *p = nt_tensor_new(2);
    p->data[0] = .75f; p->data[1] = -1.25f;
    int index = nt_tape_param(p);
    nt_tape_get()->entries[index].grad = nt_tensor_new(2);
    int ok = 1;
    for (int i = 0; i < 24; i++) {
        nt_tensor *g = nt_tape_get()->entries[index].grad;
        g->data[0] = p->data[0]; g->data[1] = p->data[1];
        float loss = .5f * (p->data[0] * p->data[0] + p->data[1] * p->data[1]);
        if (nt_chuck_architect_step(&a, .01f, loss, NULL)) { ok = 0; break; }
        if (i == 23) {
            snprintf(path, sizeof path, "%s/pending.bin", directory);
            if (nt_chuck_architect_save(&a, path)) { ok = 0; break; }
        }
        float after = .5f * (p->data[0] * p->data[0] + p->data[1] * p->data[1]);
        if (nt_chuck_architect_feedback(&a, after, NULL)) { ok = 0; break; }
    }
    snprintf(path, sizeof path, "%s/complete.bin", directory);
    if (ok && nt_chuck_architect_save(&a, path)) ok = 0;
    nt_tape_destroy(); nt_tensor_free(p);
    return ok;
}

#ifndef NT_CHUCK_FUTURE_V1_COMPAT_ONLY
// Losses are the recorded SimpleLLM seed42, snapshot128, horizon16 reversal:
// experiments/chuck_loss_architect/scenarios/receipts.json, state e6f78cd924d80727.
// Features here are an explicitly synthetic fixed normalized vector. The
// integration runner captures its real trajectory features separately.
static nt_chuck_architect_comparison reversal(void) {
    nt_chuck_architect_comparison s;
    memset(&s, 0, sizeof s);
    for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; i++)
        s.features[i] = (float)(i % 5 - 2) * .2f;
    s.future_loss[0] = 2.87523842f;
    s.future_loss[1] = 2.87194562f;
    s.future_loss[2] = 2.87865996f;
    return s;
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

static int measured_reversal(void) {
    const float immediate[] = {2.35505104f, 2.356493f, 2.35361862f};
    nt_chuck_architect a;
    CHECK(new_life(&a) == 0, "initialize future life");
    nt_chuck_architect_comparison s = reversal();
    CHECK(immediate[2] < immediate[0] && immediate[0] < immediate[1] &&
          s.future_loss[1] < s.future_loss[0] && s.future_loss[0] < s.future_loss[2],
          "recorded immediate PUSH winner reverses to future BRAKE winner");
    nt_chuck_architect before = a;
    nt_chuck_architect_comparison_receipt r;
    uint64_t hash = nt_chuck_architect_hash(&a);
    for (int i = 0; i < 128; i++) {
        CHECK(nt_chuck_architect_fit_comparison(&a, &s, &r) == NT_CHUCK_OK,
              "replay the measured future comparison");
        CHECK(r.target[0] == 0 && r.target[1] > 0 && r.target[2] < 0,
              "future credit favors BRAKE and penalizes PUSH");
        CHECK(r.loss_delta[1] > 0 && r.loss_delta[2] < 0 && r.fitted &&
              !r.nonfinite[0] && !r.nonfinite[1] && !r.nonfinite[2],
              "receipt separates measured consequences and fit status");
    }
    float scores[3];
    CHECK(nt_chuck_architect_scores(&a, s.features, scores) == 0,
          "read learned action scores on the same fixed state");
    CHECK(scores[1] > scores[0] && scores[0] > scores[2],
          "acquired action preference follows future BRAKE > HOLD > PUSH");
    CHECK(r.huber_after < r.huber_before && nt_chuck_architect_hash(&a) != hash,
          "learning changes weights and reduces prediction error");
    CHECK(only_weights_changed(&before, &a), "comparison fitting changes only policy weights");
    CHECK(r.hash_after == nt_chuck_architect_hash(&a) && r.decisions == a.decisions &&
          r.updates == a.updates, "receipt identifies its resulting life and unchanged online counters");
    printf("MEASURED_REVERSAL scores=%.9g,%.9g,%.9g\n", scores[0], scores[1], scores[2]);
    return 1;
}

static int refused_unchanged(nt_chuck_architect *a, nt_chuck_architect_comparison *s, int exact) {
    nt_chuck_architect before = *a;
    nt_chuck_architect_comparison_receipt r, sentinel;
    memset(&r, 0x3a, sizeof r); sentinel = r;
    int status = nt_chuck_architect_fit_comparison(a, s, &r);
    return (exact ? status == exact : status != 0) &&
        memcmp(&before, a, sizeof before) == 0 && memcmp(&r, &sentinel, sizeof r) == 0;
}

static int refusals(void) {
    nt_chuck_architect a;
    CHECK(new_life(&a) == 0, "initialize refusal life");
    nt_chuck_architect_comparison s = reversal();
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        s.future_loss[0] = bad[i];
        CHECK(refused_unchanged(&a, &s, NT_CHUCK_E_BASELINE),
              "nonfinite HOLD has no comparison baseline and leaves state/output unchanged");
    }
    for (int i = 0; i < 5; i++) {
        s = reversal();
        s.features[7] = i < 3 ? bad[i] : i == 3 ? 1.01f : -1.01f;
        CHECK(refused_unchanged(&a, &s, 0), "nonfinite or out-of-range feature rejected transactionally");
    }
    s = reversal();
    for (int k = 0; k < 3; k++) {
        CHECK(new_life(&a) == 0, "reset action-mask life");
        a.config.limits.enabled_actions &= ~NT_CHUCK_ACTION_BIT(k + NT_CHUCK_ACTION_HOLD);
        CHECK(refused_unchanged(&a, &s, 0), "disabled comparison action cannot be fitted");
    }
    for (int mode = NT_CHUCK_ARCHITECT_DISABLED; mode <= NT_CHUCK_ARCHITECT_LEGACY; mode++) {
        CHECK(new_life(&a) == 0, "reset nonlearned life"); a.config.mode = mode;
        CHECK(refused_unchanged(&a, &s, 0), "fit is explicit to learned lives");
    }
    CHECK(new_life(&a) == 0, "reset malformed life"); a.version = 0;
    CHECK(refused_unchanged(&a, &s, 0), "malformed policy state refused");
    CHECK(new_life(&a) == 0, "reset null sample life");
    CHECK(refused_unchanged(&a, NULL, 0), "null sample refused");
    return 1;
}

static int nonfinite_alternatives(void) {
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (int k = 1; k < 3; k++) for (int i = 0; i < 3; i++) {
        nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize alternative failure life");
        nt_chuck_architect_comparison s = reversal(); s.future_loss[k] = bad[i];
        nt_chuck_architect_comparison_receipt r;
        CHECK(nt_chuck_architect_fit_comparison(&a, &s, &r) == 0,
              "actual nonfinite alternative still supplies failure credit");
        CHECK(r.nonfinite[k] == 1 && r.target[k] == -1 && r.target[0] == 0 && r.nonfinite[0] == 0,
              "nonfinite alternative target and flag are explicit");
        CHECK(r.predicted_after[k] < r.predicted_before[k], "failed alternative loses preference");
    }
    return 1;
}

static double independent_objective(const nt_chuck_architect *a, const nt_chuck_architect_comparison *s) {
    float scores[3];
    if (nt_chuck_architect_scores(a, s->features, scores)) return NAN;
    double total = 0;
    for (int k = 0; k < 3; k++) {
        double target = ((double)s->future_loss[0] - s->future_loss[k]) /
                        (fabs((double)s->future_loss[0]) + 1e-6);
        if (target > 1) target = 1;
        if (target < -1) target = -1;
        double error = scores[k] - target, magnitude = fabs(error);
        total += magnitude <= 1 ? .5 * error * error : magnitude - .5;
    }
    return total / 3;
}

static float *coordinate(nt_chuck_architect *a, int i) {
    switch (i) {
    case 0: return &a->w1[2][3];
    case 1: return &a->w1[5][7];
    case 2: return &a->b1[3];
    case 3: return &a->w2[0][5];
    case 4: return &a->w2[1][4];
    case 5: return &a->w2[2][2];
    default: return &a->b2[2];
    }
}
static void nonzero_heads(nt_chuck_architect *a) {
    for (int k = 0; k < 3; k++) {
        a->b2[k] = (k - 1) * .2f;
        for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; h++)
            a->w2[k][h] = (float)((k + 1) * (h % 3 - 1)) * .15f;
    }
}
static void swap_alternatives(nt_chuck_architect *a) {
    float tmp[NT_CHUCK_ARCHITECT_HIDDEN], b = a->b2[1];
    memcpy(tmp, a->w2[1], sizeof tmp); memcpy(a->w2[1], a->w2[2], sizeof tmp);
    memcpy(a->w2[2], tmp, sizeof tmp); a->b2[1] = a->b2[2]; a->b2[2] = b;
}
static int gradient_and_determinism(void) {
    nt_chuck_architect before; CHECK(new_life(&before) == 0, "initialize derivative life");
    nonzero_heads(&before);
    nt_chuck_architect_comparison s = reversal();
    s.future_loss[0] = 2; s.future_loss[1] = 1.7f; s.future_loss[2] = 2.6f;
    nt_chuck_architect a = before, duplicate = before;
    nt_chuck_architect_comparison_receipt r, same;
    CHECK(nt_chuck_architect_fit_comparison(&a, &s, &r) == 0 &&
          nt_chuck_architect_fit_comparison(&duplicate, &s, &same) == 0 &&
          memcmp(&a, &duplicate, sizeof a) == 0 && memcmp(&r, &same, sizeof r) == 0,
          "identical complete state produces byte-identical fit and receipt");
    CHECK(independent_objective(&a, &s) < independent_objective(&before, &s),
          "one fit descends independently evaluated three-head Huber loss");
    for (int i = 0; i < 7; i++) {
        nt_chuck_architect plus = before, minus = before;
        const float epsilon = .002f;
        *coordinate(&plus, i) += epsilon; *coordinate(&minus, i) -= epsilon;
        double numerical = (independent_objective(&plus, &s) - independent_objective(&minus, &s)) / (2 * epsilon);
        double applied = (*coordinate(&before, i) - *coordinate(&a, i)) / before.config.learning_rate;
        if (fabs(numerical - applied) > 2e-4) {
            fprintf(stderr, "coordinate=%d numerical=%.9g applied=%.9g\n", i, numerical, applied);
            CHECK(0, "simultaneous gradient agrees with finite differences of the full objective");
        }
        CHECK(isfinite(numerical) && isfinite(applied), "finite derivative");
    }
    nt_chuck_architect permuted = before; swap_alternatives(&permuted);
    nt_chuck_architect_comparison reordered = s;
    reordered.future_loss[1] = s.future_loss[2]; reordered.future_loss[2] = s.future_loss[1];
    CHECK(nt_chuck_architect_fit_comparison(&permuted, &reordered, NULL) == 0, "fit renamed alternatives");
    swap_alternatives(&permuted);
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; h++) {
        CHECK(fabsf(permuted.b1[h] - a.b1[h]) < 1e-6f, "hidden bias independent of alternative enumeration");
        for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; i++)
            CHECK(fabsf(permuted.w1[h][i] - a.w1[h][i]) < 1e-6f,
                  "simultaneous hidden gradient independent of alternative enumeration");
    }
    return 1;
}

static int pending_and_online(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize pending life");
    nt_chuck_architect_comparison s = reversal();
    nt_tape_destroy(); nt_tape_start(); nt_chuck_rng_set(12345);
    nt_tensor *p = nt_tensor_new(2); p->data[0] = .75f; p->data[1] = -1.25f;
    int index = nt_tape_param(p);
    nt_tape_get()->entries[index].grad = nt_tensor_new(2);
    nt_tape_get()->entries[index].grad->data[0] = .1f;
    nt_tape_get()->entries[index].grad->data[1] = -.2f;
    nt_chuck_observation o; CHECK(nt_tape_chuck_observe(2, &o) == 0, "capture pending-step observation");
    float features[16]; nt_chuck_architect saved = a;
    CHECK(nt_chuck_architect_capture(&a, &o, features) == 0 && memcmp(&a, &saved, sizeof a) == 0,
          "feature capture is read-only");
    nt_chuck_architect_decision d;
    CHECK(nt_chuck_architect_step(&a, .01f, 2, &d) == 0 &&
          memcmp(features, d.features, sizeof features) == 0, "captured features match online decision features");
    CHECK(refused_unchanged(&a, &s, 0), "pending online action cannot receive off-policy comparison credit");
    float sentinel[16], output[16];
    for (int i = 0; i < 16; i++) sentinel[i] = output[i] = 100 + i;
    CHECK(nt_chuck_architect_capture(&a, &o, output) != 0 &&
          memcmp(output, sentinel, sizeof output) == 0, "pending feature capture refuses transactionally");
    nt_chuck_architect_receipt old;
    CHECK(nt_chuck_architect_feedback(&a, 1, &old) == 0 && old.reward > .49f &&
          old.reward < .51f && a.decisions == 1 && a.updates == 1 && a.pending == 0,
          "ordinary online feedback keeps its original objective and counters");
    nt_chuck_architect completed = a;
    CHECK(nt_chuck_architect_fit_comparison(&a, &s, NULL) == 0 && only_weights_changed(&completed, &a),
          "fitting preserves completed online history and its historical cache");
    nt_tape_destroy(); nt_tensor_free(p); return 1;
}

static int save_resume(void) {
    nt_chuck_architect a; CHECK(new_life(&a) == 0, "initialize resume life");
    nt_chuck_architect_comparison s = reversal();
    for (int i = 0; i < 19; i++) CHECK(nt_chuck_architect_fit_comparison(&a, &s, NULL) == 0,
                                     "fit before checkpoint");
    char path[] = "/tmp/notorch-future-life-XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0, "allocate checkpoint path"); close(fd);
    CHECK(nt_chuck_architect_save(&a, path) == 0, "save fitted life in v1 format");
    nt_chuck_architect restored;
    CHECK(nt_chuck_architect_load(&restored, path) == 0 && memcmp(&restored, &a, sizeof a) == 0,
          "fitted life loads exactly");
    unlink(path);
    for (int i = 0; i < 31; i++) {
        nt_chuck_architect_comparison_receipt x, y;
        CHECK(nt_chuck_architect_fit_comparison(&a, &s, &x) == 0 &&
              nt_chuck_architect_fit_comparison(&restored, &s, &y) == 0 &&
              memcmp(&a, &restored, sizeof a) == 0 && memcmp(&x, &y, sizeof x) == 0,
              "save/resume reproduces every subsequent fit byte");
    }
    return 1;
}
#endif

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--write-v1") == 0) return write_v1(argv[2]) ? 0 : 1;
#ifdef NT_CHUCK_FUTURE_V1_COMPAT_ONLY
    fprintf(stderr, "usage: %s --write-v1 directory\n", argv[0]); return 2;
#else
    struct { const char *name; int (*run)(void); } cases[] = {
        {"measured_reversal", measured_reversal}, {"refusals", refusals},
        {"nonfinite_alternatives", nonfinite_alternatives},
        {"gradient_and_determinism", gradient_and_determinism},
        {"pending_and_online", pending_and_online}, {"save_resume", save_resume}
    };
    int passed = 0, ran = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (argc > 1 && strcmp(argv[1], cases[i].name)) continue;
        group = cases[i].name; ran++;
        int ok = cases[i].run(); passed += ok;
        printf("%s %s\n", ok ? "PASS" : "FAIL", group);
    }
    printf("CHUCK_FUTURE groups=%d/%d checks=%d\n", passed, ran, checks);
    return ran && passed == ran ? 0 : 1;
#endif
}
