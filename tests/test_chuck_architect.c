// Chuck: Loss Architect — executable state/action/consequence contracts.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
#include "chuck_architect.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *case_name;
#define CHECK(c, message) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", case_name, __LINE__, message); \
    return 0; } } while (0)
#define NEAR(a, b) (isfinite(a) && fabsf((a) - (b)) <= 1e-7f + 2e-6f * fabsf(b))

typedef struct {
    float weights[2], gradient[2], m[2], v[2], accumulated[2];
    nt_chuck_state global;
    nt_chuck_param_state local;
    int adam_t, count, active, n_params, frozen, no_decay;
    uint32_t noise_rng;
} body_image;

static nt_tensor *body;
static int body_index;

static void body_close(void) {
    nt_tape_destroy();
    nt_tensor_free(body);
    body = NULL;
}

static void body_open(void) {
    body_close();
    nt_seed(UINT64_C(42));
    nt_chuck_rng_set(UINT32_C(2463534242));
    nt_tape_start();
    body = nt_tensor_new(2);
    body->data[0] = .75f;
    body->data[1] = -1.25f;
    body_index = nt_tape_param(body);
    nt_tape_get()->entries[body_index].grad = nt_tensor_new(2);
    nt_tape_get()->entries[body_index].grad->data[0] = 1.0f;
    nt_tape_get()->entries[body_index].grad->data[1] = -2.0f;
}

static body_image body_capture(void) {
    body_image image;
    memset(&image, 0, sizeof image);
    nt_tape *t = nt_tape_get();
    nt_adam_state *as = &t->adam[0];
    memcpy(image.weights, body->data, sizeof image.weights);
    memcpy(image.gradient, t->entries[body_index].grad->data, sizeof image.gradient);
    memcpy(image.m, as->m->data, sizeof image.m);
    memcpy(image.v, as->v->data, sizeof image.v);
    if (as->acc_grad) memcpy(image.accumulated, as->acc_grad->data, sizeof image.accumulated);
    image.global = t->chuck;
    image.local = t->chuck_params[0];
    image.adam_t = as->t;
    image.count = t->count;
    image.active = t->active;
    image.n_params = t->n_params;
    image.frozen = t->entries[body_index].frozen;
    image.no_decay = t->entries[body_index].no_decay;
    image.noise_rng = nt_chuck_rng_get();
    return image;
}

static void body_restore(const body_image *image) {
    body_open();
    nt_tape *t = nt_tape_get();
    nt_adam_state *as = &t->adam[0];
    memcpy(body->data, image->weights, sizeof image->weights);
    memcpy(t->entries[body_index].grad->data, image->gradient, sizeof image->gradient);
    memcpy(as->m->data, image->m, sizeof image->m);
    memcpy(as->v->data, image->v, sizeof image->v);
    if (as->acc_grad) memcpy(as->acc_grad->data, image->accumulated, sizeof image->accumulated);
    t->chuck = image->global;
    t->chuck_params[0] = image->local;
    as->t = image->adam_t;
    t->count = image->count;
    t->active = image->active;
    t->n_params = image->n_params;
    t->entries[body_index].frozen = image->frozen;
    t->entries[body_index].no_decay = image->no_decay;
    nt_chuck_rng_set(image->noise_rng);
}

static int same_body(const body_image *before) {
    body_image after = body_capture();
    return memcmp(before, &after, sizeof after) == 0;
}

// Immutable canonical fixture captured before Architect existed at b14dd3633b.
// Same coordinates as tests/test_notorch.c:test_chuck_golden_vector: two
// weights, two first moments, two second moments, global/local dampen, two EMAs.
static const float golden[25][10] = {
        { 0.74000001f, -1.24000001f, 0.012500003f, -0.025000006f, 1.56247988e-05f, 6.24991953e-05f, 1.0f, 1.0f, 1.0f, 1.0f }, /* 1 */
        { 0.730348229f, -1.23034823f, 0.0362500101f, -0.0725000203f, 7.81083654e-05f, 0.000312433462f, 1.0f, 1.0f, 1.08000004f, 1.00800002f }, /* 2 */
        { 0.720768213f, -1.22076821f, 0.0701250136f, -0.140250027f, 0.00021865344f, 0.000874613761f, 1.0f, 1.0f, 1.2392f, 1.02399194f }, /* 3 */
        { 0.711164117f, -1.21116412f, 0.113112524f, -0.226225048f, 0.000468431565f, 0.00187372626f, 1.0f, 1.0f, 1.47680795f, 1.04796791f }, /* 4 */
        { 0.701491773f, -1.20149171f, 0.164301276f, -0.328602552f, 0.000858583138f, 0.00343433255f, 1.0f, 1.0f, 1.79203987f, 1.07991993f }, /* 5 */
        { 0.691727459f, -1.1917274f, 0.222871169f, -0.445742339f, 0.00142021733f, 0.00568086933f, 1.0f, 1.0f, 2.18411946f, 1.11984003f }, /* 6 */
        { 0.681857347f, -1.18185723f, 0.28808409f, -0.57616818f, 0.00218441244f, 0.00873764977f, 1.0f, 1.0f, 2.65227818f, 1.1677202f }, /* 7 */
        { 0.671881795f, -1.17188168f, 0.359275699f, -0.718551397f, 0.00318221515f, 0.0127288606f, 0.97003001f, 1.02996993f, 3.19575524f, 1.22355258f }, /* 8 */
        { 0.66177839f, -1.16177821f, 0.423348159f, -0.846696317f, 0.00417902041f, 0.0167160816f, 0.940988183f, 1.06080818f, 3.16479754f, 1.22242904f }, /* 9 */
        { 0.651594877f, -1.15159464f, 0.46851337f, -0.937026739f, 0.00494045671f, 0.0197618268f, 0.91284585f, 1.09253979f, 3.13414955f, 1.22130668f }, /* 10 */
        { 0.641418934f, -1.1414187f, 0.49666205f, -0.993324101f, 0.00549800927f, 0.0219920371f, 0.885575056f, 1.1251905f, 3.10380793f, 1.2201854f }, /* 11 */
        { 0.631351054f, -1.13135087f, 0.509495854f, -1.01899171f, 0.00588313118f, 0.0235325247f, 0.8591488f, 1.15878725f, 3.07376981f, 1.21906519f }, /* 12 */
        { 0.621501148f, -1.12150097f, 0.508546233f, -1.01709247f, 0.00612724479f, 0.0245089792f, 0.833540976f, 1.19335723f, 3.0440321f, 1.21794617f }, /* 13 */
        { 0.611990273f, -1.11199009f, 0.495191634f, -0.990383267f, 0.00626174081f, 0.0250469632f, 0.808726192f, 1.2289288f, 3.01459169f, 1.21682823f }, /* 14 */
        { 0.602952957f, -1.10295284f, 0.470672458f, -0.941344917f, 0.00631797826f, 0.025271913f, 0.784679949f, 1.26553082f, 2.98544574f, 1.21571147f }, /* 15 */
        { 0.594783425f, -1.09478331f, 0.436105192f, -0.872210383f, 0.00632728497f, 0.0253091399f, 0.761378407f, 1.26526523f, 2.95659113f, 1.21459579f }, /* 16 */
        { 0.587806344f, -1.08780622f, 0.392594635f, -0.785189271f, 0.00632095849f, 0.025283834f, 0.738798559f, 1.22708011f, 2.92802525f, 1.21348119f }, /* 17 */
        { 0.581841469f, -1.08184135f, 0.353435159f, -0.706870317f, 0.00631463854f, 0.0252585541f, 0.716917992f, 1.1900773f, 2.89974499f, 1.21236777f }, /* 18 */
        { 0.576737583f, -1.07673752f, 0.318191618f, -0.636383235f, 0.0063083251f, 0.0252333004f, 0.69571507f, 1.15422082f, 2.87174749f, 1.21125543f }, /* 19 */
        { 0.572367191f, -1.07236719f, 0.28647244f, -0.57294488f, 0.00630201772f, 0.0252080709f, 0.675168753f, 1.11947465f, 2.8440299f, 1.21014416f }, /* 20 */
        { 0.568622649f, -1.06862259f, 0.257925183f, -0.515850365f, 0.00629571686f, 0.0251828674f, 0.655258834f, 1.08580446f, 2.81658959f, 1.20903409f }, /* 21 */
        { 0.565214396f, -1.06521428f, 0.23223266f, -0.46446532f, 0.00628942205f, 0.0251576882f, 0.675241649f, 1.05317712f, 2.7894237f, 1.20792508f }, /* 22 */
        { 0.562111139f, -1.06211102f, 0.209109396f, -0.418218791f, 0.00628313376f, 0.025132535f, 0.695803404f, 1.02156019f, 2.76252937f, 1.20681715f }, /* 23 */
        { 0.559284806f, -1.05928469f, 0.188298449f, -0.376596898f, 0.00627685152f, 0.0251074061f, 0.716960788f, 0.990922451f, 2.73590398f, 1.20571041f }, /* 24 */
        { 0.559284806f, -1.05928469f, 0.188298449f, -0.376596898f, 0.00627685152f, 0.0251074061f, 0.738731146f, 0.990922451f, 2.7095449f, 1.20460474f }, /* 25 */
};

static int test_legacy(void) {
    for (int mode = 0; mode < 9; ++mode) {
        body_open();
        nt_chuck_architect architect;
        nt_chuck_architect_config config;
        nt_chuck_architect_config_default(&config);
        if (mode == 3) config.mode = NT_CHUCK_ARCHITECT_DISABLED;
        if (mode == 4) config.mode = NT_CHUCK_ARCHITECT_LEGACY;
        if (mode == 5 || mode == 7 || mode == 8) {
            char error[128];
            const char *json = mode == 5 ? "{}" : mode == 7 ? " \n\t " : NULL;
            CHECK(nt_chuck_architect_config_parse_json(&config, json, error, sizeof error) == 0,
                  "empty configuration accepted");
        }
        CHECK(nt_chuck_architect_init(&architect, &config) == 0, "legacy life initialized");
        for (int step = 1; step <= 25; ++step) {
            nt_tape *t = nt_tape_get();
            float g = step <= 8 ? .125f * step : step <= 16 ? .125f * (17 - step) : .001f;
            t->entries[body_index].grad->data[0] = g;
            t->entries[body_index].grad->data[1] = -2.0f * g;
            float loss = step <= 8 ? 1.0f + 8.0f * (step - 1) : .1f;
            nt_chuck_action action = {NT_CHUCK_ACTION_LEGACY, 0};
            if (mode == 0) nt_tape_chuck_step(.01f, loss);
            else if (mode >= 3) {
                nt_chuck_architect_decision decision;
                CHECK(nt_chuck_architect_step(mode == 6 ? NULL : &architect, .01f, loss, &decision) == 0,
                      "execute disabled/absent/default legacy life");
                CHECK(decision.action.kind == NT_CHUCK_ACTION_LEGACY, "default preserves legacy action");
            } else {
                CHECK(nt_tape_chuck_step_action(.01f, loss, mode == 1 ? NULL : &action, NULL) == 0,
                      "execute legacy path");
            }
            nt_chuck_state *cs = &t->chuck;
            nt_chuck_param_state *cp = &t->chuck_params[0];
            nt_adam_state *as = &t->adam[0];
            float actual[] = {body->data[0], body->data[1], as->m->data[0], as->m->data[1],
                as->v->data[0], as->v->data[1], cs->dampen, cp->dampen, cs->loss_ema, cs->macro_ema};
            for (int j = 0; j < 10; ++j) {
                if (!NEAR(actual[j], golden[step - 1][j])) {
                    fprintf(stderr, "mode=%d step=%d field=%d got=%.9g expected=%.9g\n",
                            mode, step, j, actual[j], golden[step - 1][j]);
                    CHECK(0, "canonical golden trajectory");
                }
            }
            int n = step < 25 ? step : 24;
            CHECK(cs->global_step == step && cs->pos == step % 16 && cs->full == (step >= 16)
                  && cs->initialized == 1 && cs->stag == 0 && cs->noise == 0.0f
                  && cs->lr_scale == 1.0f && cs->macro_stag == 0 && cs->best_macro == 1e9f
                  && cp->pos == n % 16 && cp->full == (step >= 16)
                  && cp->stag == (step <= 16 ? 0 : n - 16) && cp->frozen == (step >= 24)
                  && as->t == n, "canonical counters and freeze");
            for (int h = 0; h < 16; ++h) {
                int last = h + 1;
                if (last + 16 <= step) last += 16;
                float expected = last <= step ? golden[last - 1][8] : 0;
                CHECK(NEAR(cs->loss_hist[h], expected), "canonical full loss-history ring");
            }
        }
        CHECK(nt_chuck_rng_get() == UINT32_C(2463534242), "noise-free parity consumes no random draw");
    }
    body_close();
    return 1;
}

static int test_observation(void) {
    body_open();
    body_image before = body_capture();
    nt_chuck_observation obs;
    CHECK(nt_tape_chuck_observe(2.0f, &obs) == 0, "cold observation");
    CHECK(same_body(&before), "observation is read-only including noise RNG");
    CHECK(obs.loss == 2 && obs.loss_ema == 2 && obs.macro_ema == 2
          && obs.dampen == 1 && obs.lr_scale == 1 && obs.noise == 0
          && NEAR(obs.grad_norm, sqrtf(5.0f)), "independent cold observation values");
    CHECK(nt_tape_chuck_observe(NAN, &obs) != 0 && same_body(&before), "NaN observation rejected transactionally");
    nt_chuck_observation histories[2];
    for (int run = 0; run < 2; ++run) {
        body_open();
        for (int step = 0; step < 10; ++step) nt_tape_chuck_step(.001f, run ? 20.0f - step : 1.0f + step);
        CHECK(nt_tape_chuck_observe(5.0f, &histories[run]) == 0, "history observation");
    }
    CHECK(histories[0].loss == histories[1].loss && histories[0].loss_trend != histories[1].loss_trend,
          "identical present loss retains distinct trajectory");
    body_close();
    return 1;
}

static int test_actions(void) {
    static const nt_chuck_action_kind actions[] = {NT_CHUCK_ACTION_HOLD, NT_CHUCK_ACTION_BRAKE, NT_CHUCK_ACTION_PUSH};
    static const float expected_dampen[] = {1.0f, .97f, 1.03f};
    for (int i = 0; i < 3; ++i) {
        body_open();
        nt_chuck_action a = {actions[i], 0};
        CHECK(nt_tape_chuck_step_action(.01f, 1, &a, NULL) == 0, "primitive accepted");
        CHECK(NEAR(nt_tape_get()->chuck.dampen, expected_dampen[i]), "primitive has its declared direction");
        CHECK(NEAR(body->data[0], .75f - .01f * expected_dampen[i])
              && NEAR(body->data[1], -1.25f + .01f * expected_dampen[i]), "primitive changes known first-step weights");
    }
    body_open();
    nt_chuck_action_limits limits;
    nt_chuck_action_limits_default(&limits);
    limits.dampen_min = .8f; limits.dampen_max = 1.1f;
    nt_chuck_action a = {NT_CHUCK_ACTION_SET_DAMPEN, .8f};
    CHECK(nt_tape_chuck_step_action(0, 1, &a, &limits) == 0, "absolute dampen bound accepted");
    a.kind = NT_CHUCK_ACTION_BRAKE; a.value = 0;
    CHECK(nt_tape_chuck_step_action(0, 1, &a, &limits) == 0 && nt_tape_get()->chuck.dampen == .8f,
          "brake saturates lower configured bound");
    a.kind = NT_CHUCK_ACTION_SET_DAMPEN; a.value = 1.1f;
    CHECK(nt_tape_chuck_step_action(0, 1, &a, &limits) == 0, "absolute upper bound accepted");
    a.kind = NT_CHUCK_ACTION_PUSH; a.value = 0;
    CHECK(nt_tape_chuck_step_action(0, 1, &a, &limits) == 0 && nt_tape_get()->chuck.dampen == 1.1f,
          "push saturates upper configured bound");
    a.kind = NT_CHUCK_ACTION_SET_LR_SCALE; a.value = .5f;
    CHECK(nt_tape_chuck_step_action(0, 1, &a, &limits) == 0 && nt_tape_get()->chuck.lr_scale == .5f,
          "absolute LR scale");
    body_close();
    return 1;
}

static int test_action_rejection(void) {
    body_open();
    nt_chuck_action_limits limits;
    nt_chuck_action_limits_default(&limits);
    const nt_chuck_action invalid[] = {
        {(nt_chuck_action_kind)-1, 0}, {NT_CHUCK_ACTION_COUNT, 0},
        {NT_CHUCK_ACTION_HOLD, 1}, {NT_CHUCK_ACTION_PUSH, NAN},
        {NT_CHUCK_ACTION_SET_DAMPEN, .299f}, {NT_CHUCK_ACTION_SET_DAMPEN, 2.001f},
        {NT_CHUCK_ACTION_SET_LR_SCALE, .049f}, {NT_CHUCK_ACTION_SET_LR_SCALE, 2.001f},
        {NT_CHUCK_ACTION_SET_NOISE, -.001f}, {NT_CHUCK_ACTION_SET_NOISE, .011f},
        {NT_CHUCK_ACTION_SET_NOISE, INFINITY}
    };
    body_image before = body_capture();
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        CHECK(nt_tape_chuck_step_action(.01f, 1, &invalid[i], &limits) != 0, "invalid action refused");
        CHECK(same_body(&before), "rejected action leaves full body and RNG unchanged");
    }
    nt_chuck_action a = {NT_CHUCK_ACTION_HOLD, 0};
    limits.enabled_actions &= ~NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD);
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &limits) != 0 && same_body(&before), "unavailable action refused transactionally");
    nt_chuck_action_limits_default(&limits);
    limits.dampen_min = 1.5f; limits.dampen_max = 1.0f;
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &limits) != 0 && same_body(&before), "inverted bounds refused");
    nt_chuck_action_limits_default(&limits);
    limits.noise_max = NAN;
    CHECK(nt_tape_chuck_step_action(.01f, 1, &a, &limits) != 0 && same_body(&before), "nonfinite bounds refused");
    CHECK(nt_tape_chuck_step_action(NAN, 1, &a, NULL) != 0 && same_body(&before), "NaN LR refused");
    CHECK(nt_tape_chuck_step_action(.01f, INFINITY, &a, NULL) != 0 && same_body(&before), "infinite loss refused");
    CHECK(nt_chuck_rng_set(0) != 0 && same_body(&before), "zero xorshift state refused");
    body_close();
    return 1;
}

/* Policy/config/persistence gates below exercise the same public interface as a training body. */

static int test_configuration(void) {
    const char *invalid[] = {
        "{", "[]", "true", "{} trailing", "{\"version\":2}",
        "{\"mode\":\"unknown\"}", "{\"mode\":false}",
        "{\"mode\":\"legacy\",\"enabled_actions\":[\"hold\"]}",
        "{\"mode\":\"disabled\",\"enabled_actions\":[\"hold\"]}",
        "{\"mode\":\"disabled\",\"enabled_actions\":[]}",
        "{\"mode\":\"legacy\",\"dampen_max\":0.5}",
        "{\"mode\":\"disabled\",\"dampen_max\":0.5}",
        "{\"life_id\":\"" "\\", "{\"life_id\":\"\\u",
        "{\"life_id\":\"\\u0", "{\"life_id\":\"\\u00", "{\"life_id\":\"\\u000",
        "{\"mode\":\"legacy\",\"mode\":\"learned\"}",
        "{\"mode\":\"legacy\",\"\\u006dode\":\"legacy\"}",
        "{\"unknown\":1}", "{\"seed\":0}", "{\"seed\":-1}",
        "{\"seed\":1.5}", "{\"seed\":4294967296}", "{\"seed\":01}", "{\"seed\":+1}",
        "{\"learning_rate\":0.}", "{\"learning_rate\":1e}", "{\"learning_rate\":0x1}",
        "{\"life_id\":\"\"}", "{\"life_id\":\"bad life\"}",
        "{\"life_id\":\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678901\"}",
        "{\"learning_rate\":-0.1}", "{\"exploration\":1.001}",
        "{\"exploration\":NaN}", "{\"learning_rate\":1e999}",
        "{\"dampen_min\":0.2}", "{\"dampen_max\":2.1}",
        "{\"dampen_min\":1.5,\"dampen_max\":1.0}",
        "{\"lr_scale_min\":0.01}", "{\"noise_max\":0.1}",
        "{\"enabled_actions\":[\"hold\",\"hold\"]}",
        "{\"enabled_actions\":[\"hold\",\"\\u0068old\"]}",
        "{\"enabled_actions\":[\"invented\"]}",
        "{\"enabled_actions\":\"hold\"}",
        "{\"mode\":\"learned\",\"enabled_actions\":[\"push\"]}"
    };
    nt_chuck_architect_config config;
    nt_chuck_architect_config_default(&config);
    nt_chuck_architect_config before = config;
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        char error[128] = {0};
        int result = nt_chuck_architect_config_parse_json(&config, invalid[i], error, sizeof error);
        if (result == 0) fprintf(stderr, "accepted invalid JSON: %s\n", invalid[i]);
        CHECK(result != 0, "strict configuration refusal");
        CHECK(memcmp(&config, &before, sizeof config) == 0, "bad config leaves destination unchanged");
        CHECK(error[0], "bad config identifies refusal");
    }
    char error[128];
    CHECK(nt_chuck_architect_config_parse_json(&config,
          "{\"mode\":\"learned\",\"life_id\":\"fixture.v1\",\"enabled_actions\":[\"hold\",\"brake\",\"push\"],\"seed\":17,\"exploration\":0}",
          error, sizeof error) == 0, "inspectable learned config");
    CHECK(config.mode == NT_CHUCK_ARCHITECT_LEARNED && config.seed == 17 && config.exploration == 0
          && !strcmp(config.life_id, "fixture.v1")
          && config.limits.enabled_actions == (NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD)
             | NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_BRAKE) | NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_PUSH)),
          "declared values survive parser");
    return 1;
}

static int learned_init(nt_chuck_architect *a, float exploration) {
    nt_chuck_architect_config config;
    nt_chuck_architect_config_default(&config);
    config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    config.seed = 71;
    config.learning_rate = .1f;
    config.exploration = exploration;
    return nt_chuck_architect_init(a, &config) == 0;
}

static int test_policy_transactions(void) {
    body_open();
    nt_chuck_architect a;
    CHECK(learned_init(&a, .75f), "learned init");
    uint64_t before = nt_chuck_architect_hash(&a);
    nt_chuck_architect_receipt receipt;
    CHECK(nt_chuck_architect_feedback(&a, .5f, &receipt) != 0
          && nt_chuck_architect_hash(&a) == before, "unexecuted prediction cannot receive credit");
    nt_chuck_observation obs;
    nt_chuck_action one, two;
    CHECK(nt_tape_chuck_observe(1, &obs) == 0, "policy observation");
    CHECK(nt_chuck_architect_select(&a, &obs, &one) == 0
          && nt_chuck_architect_select(&a, &obs, &two) == 0
          && one.kind == two.kind && one.value == two.value
          && nt_chuck_architect_hash(&a) == before, "select is deterministic read-only preview including RNG");
    body_image body_before = body_capture();
    nt_chuck_architect_decision decision, untouched;
    memset(&decision, 0xa5, sizeof decision); untouched = decision;
    for (int mode = NT_CHUCK_ARCHITECT_DISABLED; mode <= NT_CHUCK_ARCHITECT_LEGACY; ++mode) {
        nt_chuck_architect_config invalid_config;
        nt_chuck_architect_config_default(&invalid_config);
        invalid_config.mode = (nt_chuck_architect_mode)mode;
        nt_chuck_architect legacy, original;
        CHECK(nt_chuck_architect_init(&legacy, &invalid_config) == 0, "default compatibility life");
        original = legacy;
        invalid_config.limits.enabled_actions = NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD);
        CHECK(nt_chuck_architect_init(&legacy, &invalid_config) != 0
              && memcmp(&legacy, &original, sizeof legacy) == 0,
              "contradictory compatibility config rejected before replacing life");
        legacy.config = invalid_config;
        original = legacy;
        nt_chuck_action output = {NT_CHUCK_ACTION_SET_LR_SCALE, .5f};
        CHECK(nt_chuck_architect_select(&legacy, &obs, &output) != 0
              && output.kind == NT_CHUCK_ACTION_SET_LR_SCALE && output.value == .5f
              && memcmp(&legacy, &original, sizeof legacy) == 0,
              "post-init invalid compatibility config refuses preview transactionally");
        CHECK(nt_chuck_architect_step(&legacy, .01f, 1, &decision) != 0
              && memcmp(&legacy, &original, sizeof legacy) == 0 && same_body(&body_before)
              && memcmp(&decision, &untouched, sizeof decision) == 0,
              "post-init invalid compatibility config refuses execution transactionally");
    }
    CHECK(nt_chuck_architect_step(&a, NAN, 1, &decision) != 0
          && nt_chuck_architect_hash(&a) == before && same_body(&body_before)
          && memcmp(&decision, &untouched, sizeof decision) == 0,
          "rejected step preserves policy, body, RNG and decision destination");
    nt_chuck_architect rejected;
    nt_chuck_architect_config narrow = a.config;
    narrow.exploration = 1;
    narrow.limits.noise_min = .001f;
    CHECK(nt_chuck_architect_init(&rejected, &narrow) == 0, "valid limits for an already noisy training life");
    uint64_t rejected_before = nt_chuck_architect_hash(&rejected);
    CHECK(nt_chuck_architect_step(&rejected, .01f, 1, &decision) != 0
          && nt_chuck_architect_hash(&rejected) == rejected_before && same_body(&body_before)
          && memcmp(&decision, &untouched, sizeof decision) == 0,
          "post-selection core refusal rolls back exploration RNG and pending credit");
    CHECK(nt_chuck_architect_step(&a, .01f, 1, &decision) == 0 && a.pending,
          "successful action creates pending transition");
    CHECK(decision.action.kind == one.kind, "preview equals executed action");
    before = nt_chuck_architect_hash(&a);
    body_before = body_capture();
    CHECK(nt_chuck_architect_step(&a, .01f, 1, NULL) != 0
          && nt_chuck_architect_hash(&a) == before && same_body(&body_before),
          "pending action must receive its own consequence first");
    CHECK(nt_chuck_architect_feedback(&a, INFINITY, &receipt) == 0
          && receipt.nonfinite == 1 && receipt.reward == -1 && receipt.learned == 1
          && !a.pending && a.updates == 1, "nonfinite consequence receives explicit negative target");
    before = nt_chuck_architect_hash(&a);
    CHECK(nt_chuck_architect_feedback(&a, .5f, &receipt) != 0
          && nt_chuck_architect_hash(&a) == before, "a consequence cannot be credited twice");
    body_close();
    return 1;
}

static int test_credit(void) {
    nt_chuck_architect lives[2];
    nt_chuck_action initial[2], next[2];
    float selected_bias[2];
    for (int arm = 0; arm < 2; ++arm) {
        body_open();
        CHECK(learned_init(&lives[arm], 0), "matched life initialization");
        nt_chuck_architect_decision decision;
        nt_chuck_architect_receipt receipt;
        CHECK(nt_chuck_architect_step(&lives[arm], .01f, 1, &decision) == 0, "execute selected action before credit");
        initial[arm] = decision.action;
        CHECK(NEAR(body->data[0], .74f) && NEAR(body->data[1], -1.24f),
              "initial learned action follows pinned HOLD weight trajectory");
        CHECK(initial[arm].kind == NT_CHUCK_ACTION_HOLD, "zero outcome heads resolve tie to hold");
        float consequence = arm == 0 ? .5f : 1.5f;
        CHECK(nt_chuck_architect_feedback(&lives[arm], consequence, &receipt) == 0, "actual supplied consequence accepted");
        float expected = arm == 0 ? .5f / 1.000001f : -.5f / 1.000001f;
        CHECK(NEAR(receipt.reward, expected) && receipt.before_loss == 1 && receipt.after_loss == consequence
              && receipt.loss_delta == (double)1 - consequence && receipt.action.kind == initial[arm].kind,
              "receipt binds measured loss difference to executed action");
        selected_bias[arm] = lives[arm].b2[0];
        CHECK((arm == 0 && selected_bias[arm] > 0) || (arm == 1 && selected_bias[arm] < 0),
              "credit sign reaches the selected learned head");
        CHECK(lives[arm].b2[1] == 0 && lives[arm].b2[2] == 0, "unselected action heads receive no fabricated target");
        nt_chuck_observation obs;
        CHECK(nt_tape_chuck_observe(1, &obs) == 0
              && nt_chuck_architect_select(&lives[arm], &obs, &next[arm]) == 0, "future action after consequence");
    }
    CHECK(next[0].kind == NT_CHUCK_ACTION_HOLD && next[1].kind != NT_CHUCK_ACTION_HOLD,
          "opposite consequences alter the next selected action");
    CHECK(selected_bias[0] > selected_bias[1], "learned preference orders outcomes correctly");
    body_close();
    return 1;
}

static int files_equal(const char *left, const char *right) {
    FILE *a = fopen(left, "rb"), *b = fopen(right, "rb");
    if (!a || !b) { if (a) fclose(a); if (b) fclose(b); return 0; }
    int x, y, same = 1;
    do { x = fgetc(a); y = fgetc(b); if (x != y) same = 0; } while (same && x != EOF);
    fclose(a); fclose(b);
    return same;
}

static float quadratic_loss(void) {
    float x = body->data[0] - .1f, y = body->data[1] + .3f;
    return x * x + y * y;
}

static void quadratic_gradient(void) {
    nt_tensor *g = nt_tape_get()->entries[body_index].grad;
    g->data[0] = 2 * (body->data[0] - .1f);
    g->data[1] = 2 * (body->data[1] + .3f);
}

static int train_step(nt_chuck_architect *a) {
    quadratic_gradient();
    nt_chuck_architect_decision decision;
    nt_chuck_architect_receipt receipt;
    if (nt_chuck_architect_step(a, .03f, quadratic_loss(), &decision) != 0) return 0;
    if (nt_chuck_architect_feedback(a, quadratic_loss(), &receipt) != 0) return 0;
    return isfinite(receipt.after_loss) && receipt.learned;
}

static int test_resume(void) {
    char directory[] = "/tmp/notorch-architect-test-XXXXXX";
    CHECK(mkdtemp(directory) != NULL, "temporary checkpoint directory");
    char pending_path[256], baseline_path[256], resumed_path[256], bad_path[256];
    snprintf(pending_path, sizeof pending_path, "%s/pending.bin", directory);
    snprintf(baseline_path, sizeof baseline_path, "%s/baseline.bin", directory);
    snprintf(resumed_path, sizeof resumed_path, "%s/resumed.bin", directory);
    snprintf(bad_path, sizeof bad_path, "%s/bad.bin", directory);
    body_open();
    nt_chuck_architect a, resumed;
    CHECK(learned_init(&a, .6f), "resumable learned life");
    nt_chuck_action noise = {NT_CHUCK_ACTION_SET_NOISE, .003f};
    CHECK(nt_tape_chuck_step_action(0, quadratic_loss(), &noise, NULL) == 0, "nonzero independent Chuck noise stream");
    for (int i = 0; i < 5; ++i) CHECK(train_step(&a), "training before checkpoint");
    quadratic_gradient();
    nt_chuck_architect_decision pending;
    CHECK(nt_chuck_architect_step(&a, .03f, quadratic_loss(), &pending) == 0 && a.pending,
          "checkpoint after execution before consequence");
    body_image checkpoint = body_capture();
    float after_loss = quadratic_loss();
    uint64_t pending_hash = nt_chuck_architect_hash(&a);
    CHECK(nt_chuck_architect_save(&a, pending_path) == 0, "save pending life");
    nt_chuck_architect_receipt first, second;
    CHECK(nt_chuck_architect_feedback(&a, after_loss, &first) == 0, "baseline pending credit");
    for (int i = 0; i < 16; ++i) CHECK(train_step(&a), "uninterrupted continuation");
    body_image final = body_capture();
    uint64_t final_hash = nt_chuck_architect_hash(&a);
    CHECK(nt_chuck_architect_save(&a, baseline_path) == 0, "save baseline continuation");
    body_restore(&checkpoint);
    memset(&resumed, 0, sizeof resumed);
    CHECK(nt_chuck_architect_load(&resumed, pending_path) == 0
          && nt_chuck_architect_hash(&resumed) == pending_hash && resumed.pending,
          "restored pending state is exact");
    CHECK(nt_chuck_architect_feedback(&resumed, after_loss, &second) == 0
          && first.reward == second.reward && first.predicted == second.predicted
          && first.error == second.error && first.action.kind == second.action.kind
          && first.decision == second.decision, "pending consequence is credited identically");
    for (int i = 0; i < 16; ++i) CHECK(train_step(&resumed), "resumed continuation");
    CHECK(same_body(&final), "CPU target weights, moments, Chuck history and noise RNG resume bit-exactly");
    CHECK(nt_chuck_architect_hash(&resumed) == final_hash, "policy weights, exploration RNG and credit resume exactly");
    CHECK(nt_chuck_architect_save(&resumed, resumed_path) == 0 && files_equal(baseline_path, resumed_path),
          "canonical saved continuation bytes identical");
    nt_chuck_architect invalid = resumed;
    invalid.w1[0][0] = NAN;
    CHECK(nt_chuck_architect_save(&invalid, baseline_path) != 0
          && files_equal(baseline_path, resumed_path), "refused save preserves existing checkpoint bytes");
    CHECK(final.noise_rng != checkpoint.noise_rng, "resume comparison exercised actual noise draws");
    FILE *src = fopen(pending_path, "rb"), *bad = fopen(bad_path, "wb");
    CHECK(src && bad, "corruption fixture files");
    int ch, offset = 0;
    while ((ch = fgetc(src)) != EOF) { if (offset++ == 24) ch ^= 1; fputc(ch, bad); }
    fclose(src); fclose(bad);
    CHECK(nt_chuck_architect_load(&resumed, bad_path) != 0
          && nt_chuck_architect_hash(&resumed) == final_hash, "corrupt life refused without changing resident life");
    bad = fopen(bad_path, "wb"); CHECK(bad != NULL, "truncated fixture");
    fputs("bad", bad); fclose(bad);
    CHECK(nt_chuck_architect_load(&resumed, bad_path) != 0
          && nt_chuck_architect_hash(&resumed) == final_hash, "truncated life refused transactionally");
    unlink(pending_path); unlink(baseline_path); unlink(resumed_path); unlink(bad_path); rmdir(directory);
    body_close();
    return 1;
}

typedef int (*test_function)(void);
int main(int argc, char **argv) {
    const struct {const char *name; test_function run;} tests[] = {
        {"legacy", test_legacy}, {"observation", test_observation},
        {"actions", test_actions}, {"action_rejection", test_action_rejection},
        {"configuration", test_configuration}, {"policy_transactions", test_policy_transactions},
        {"credit", test_credit}, {"resume", test_resume}
    };
    int passed = 0, failed = 0, selected = 0;
    for (size_t i = 0; i < sizeof tests / sizeof *tests; ++i) {
        if (argc == 2 && strcmp(argv[1], tests[i].name)) continue;
        ++selected;
        case_name = tests[i].name;
        if (tests[i].run()) { ++passed; printf("PASS %s\n", case_name); }
        else ++failed;
        body_close();
    }
    if (!selected || argc > 2) { fprintf(stderr, "unknown test selection\n"); return 2; }
    printf("Chuck Architect: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
