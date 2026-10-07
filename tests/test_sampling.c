// Explicit sampling streams and categorical laws, without a model or weights file.
#include "notorch.h"
#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "sampling_reference.h"

static unsigned checks;
#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void vectors_and_reference(void) {
    // Published pcg32-demo vector, initstate 42 and initseq 54.
    const uint32_t published[] = {
        0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e
    };
    uint64_t state;
    nt_rng_seed(&state, 42);
    for (unsigned i = 0; i < sizeof(published) / sizeof(*published); i++)
        CHECK(nt_rng_u32(&state) == published[i]);

    const float weights[] = {0, 1, 2, 5, 13};
    for (unsigned i = 0; i < sizeof(sampling_reference) / sizeof(*sampling_reference); i++) {
        const sampling_reference_case* r = &sampling_reference[i];
        if (r->reset) nt_rng_seed(&state, r->seed);
        if (r->op == 0) CHECK(nt_rng_u32(&state) == r->integer);
        else if (r->op == 1) {
            float value = nt_rng_uniform(&state);
            CHECK(value == r->uniform);
            CHECK(value >= 0.0f && value < 1.0f);
        } else if (r->op == 2) {
            uint32_t value = UINT32_MAX;
            CHECK(nt_rng_index(&state, r->bound, &value) == 0);
            CHECK(value == r->integer && value < r->bound);
        } else {
            int value = -1;
            CHECK(nt_rng_categorical(&state, weights, 5, r->temperature, &value) == 0);
            CHECK(value == (int)r->integer);
        }
        CHECK(state == r->state);
    }
    puts("PASS published PCG vector and 384 Python integer/Decimal reference cases");
}

static void bounds_and_cdf(void) {
    // State zero emits zero, which bound 3 must reject (2^32 mod 3 == 1).
    // State 109 also emits zero. The third word is the first accepted one.
    uint64_t state = 0, expected = 0;
    uint32_t index = UINT32_MAX;
    CHECK(nt_rng_u32(&expected) == 0);
    CHECK(nt_rng_u32(&expected) == 0);
    uint32_t accepted = nt_rng_u32(&expected);
    CHECK(accepted > 0);
    CHECK(nt_rng_index(&state, 3, &index) == 0);
    CHECK(index == accepted % 3 && state == expected);

    nt_rng_seed(&state, 42);
    expected = state;
    (void)nt_rng_u32(&expected);
    CHECK(nt_rng_index(&state, 1, &index) == 0);
    CHECK(index == 0 && state == expected);

    // Every u32 result must become a float strictly below one. This state
    // emits UINT32_MAX under XSH-RR, exercising float rounding at the top.
    state = UINT64_C(0x07fffe0000000000);
    expected = state;
    CHECK(nt_rng_u32(&expected) == UINT32_MAX);
    CHECK(nt_rng_uniform(&state) == 0x1.fffffep-1f);
    CHECK(state == expected);

    const float holes[] = {0, 1, 0, 1, 0};
    int chosen = -1;
    CHECK(nt_categorical_index(holes, 5, 1, 0, &chosen) == 0 && chosen == 1);
    CHECK(nt_categorical_index(holes, 5, 1, nextafter(0.5, 0), &chosen) == 0 && chosen == 1);
    CHECK(nt_categorical_index(holes, 5, 1, 0.5, &chosen) == 0 && chosen == 3);
    CHECK(nt_categorical_index(holes, 5, 1, nextafter(1, 0), &chosen) == 0 && chosen == 3);
    const float singleton[] = {0, 0, 1, 0};
    state = 0; expected = state;
    (void)nt_rng_u32(&expected);
    CHECK(nt_rng_categorical(&state, singleton, 4, 1, &chosen) == 0);
    CHECK(chosen == 2 && state == expected);

    // Exact powers provide CDF boundaries for the temperature law.
    const float powers[] = {1, 4};
    CHECK(nt_categorical_index(powers, 2, 0.5f, 0.1, &chosen) == 0 && chosen == 1);
    CHECK(nt_categorical_index(powers, 2, 1, 0.1, &chosen) == 0 && chosen == 0);
    CHECK(nt_categorical_index(powers, 2, 2, 0.25, &chosen) == 0 && chosen == 0);
    CHECK(nt_categorical_index(powers, 2, 1, 0.25, &chosen) == 0 && chosen == 1);

    const float extreme[] = {FLT_TRUE_MIN, FLT_MAX, 0};
    const float temperatures[] = {FLT_TRUE_MIN, FLT_MIN, 0.1f, 1, FLT_MAX};
    const double draws[] = {0, 0.1, 0.5, nextafter(1, 0)};
    for (unsigned t = 0; t < sizeof(temperatures) / sizeof(*temperatures); t++) {
        for (unsigned d = 0; d < sizeof(draws) / sizeof(*draws); d++) {
            CHECK(nt_categorical_index(extreme, 3, temperatures[t], draws[d], &chosen) == 0);
            CHECK(chosen == 0 || chosen == 1);
            if (temperatures[t] == FLT_MAX)
                CHECK(chosen == (draws[d] < 0.5 ? 0 : 1));
            else if (draws[d] > 0)
                CHECK(chosen == 1);
        }
    }

    const float base[] = {0, 1, 2, 5, 13};
    const float scaled[] = {0, 1024, 2048, 5120, 13312};
    for (int d = 0; d < 1000; d++) {
        int a, b;
        CHECK(nt_categorical_index(base, 5, 0.75f, d / 1000.0, &a) == 0);
        CHECK(nt_categorical_index(scaled, 5, 0.75f, d / 1000.0, &b) == 0);
        CHECK(a == b);
    }
    puts("PASS rejection, singleton consumption, strict CDF, temperature/extreme ranges and scale invariance");
}

static void invalid_calls(void) {
    const float valid[] = {0, 1, 2};
    const float bad[][3] = {{0, 0, -0.0f}, {1, -1, 2}, {1, NAN, 2}, {1, INFINITY, 2}, {1, -INFINITY, 2}};
    const float bad_t[] = {0, -0.0f, -1, NAN, INFINITY, -INFINITY};
    const double bad_draw[] = {-1, NAN, INFINITY, -INFINITY, 1, nextafter(1, 2)};
    uint64_t state, before;
    nt_rng_seed(&state, 7);
    before = state;
    int out = -17;
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        CHECK(nt_categorical_index(bad[i], 3, 1, 0.5, &out) == -1);
        CHECK(out == -17);
        CHECK(nt_rng_categorical(&state, bad[i], 3, 1, &out) == -1);
        CHECK(out == -17 && state == before);
    }
    for (unsigned i = 0; i < sizeof(bad_t) / sizeof(*bad_t); i++) {
        CHECK(nt_categorical_index(valid, 3, bad_t[i], 0.5, &out) == -1);
        CHECK(out == -17);
        CHECK(nt_rng_categorical(&state, valid, 3, bad_t[i], &out) == -1);
        CHECK(out == -17 && state == before);
    }
    for (unsigned i = 0; i < sizeof(bad_draw) / sizeof(*bad_draw); i++) {
        CHECK(nt_categorical_index(valid, 3, 1, bad_draw[i], &out) == -1);
        CHECK(out == -17);
    }
    CHECK(nt_categorical_index(NULL, 3, 1, 0.5, &out) == -1);
    CHECK(nt_categorical_index(valid, 0, 1, 0.5, &out) == -1);
    CHECK(nt_categorical_index(valid, -1, 1, 0.5, &out) == -1);
    CHECK(nt_categorical_index(valid, 3, 1, 0.5, NULL) == -1);
    CHECK(out == -17);
    CHECK(nt_rng_categorical(NULL, valid, 3, 1, &out) == -1);
    CHECK(nt_rng_categorical(&state, NULL, 3, 1, &out) == -1);
    CHECK(nt_rng_categorical(&state, valid, 0, 1, &out) == -1);
    CHECK(nt_rng_categorical(&state, valid, -1, 1, &out) == -1);
    CHECK(nt_rng_categorical(&state, valid, 3, 1, NULL) == -1);
    CHECK(out == -17 && state == before);
    uint32_t index = 987;
    CHECK(nt_rng_index(NULL, 3, &index) == -1);
    CHECK(nt_rng_index(&state, 0, &index) == -1);
    CHECK(nt_rng_index(&state, 3, NULL) == -1);
    CHECK(state == before && index == 987);
    nt_rng_seed(NULL, 42);
    CHECK(nt_rng_u32(NULL) == 0);
    CHECK(nt_rng_uniform(NULL) == 0);
    puts("PASS invalid arguments preserve state/output and safe NULL raw calls");
}

static void ownership_and_isolation(void) {
    uint64_t a, b;
    nt_rng_seed(&a, 42); b = a;
    nt_tensor* legacy = nt_tensor_new(8);
    nt_tensor* legacy_reference = nt_tensor_new(8);
    CHECK(legacy && legacy_reference);
    for (int i = 0; i < 1000; i++) {
        uint32_t expected = nt_rng_u32(&b);
        srand((unsigned)i);
        (void)rand();
        nt_seed((uint64_t)i);
        nt_tensor_rand(legacy, 1);
        CHECK(nt_rng_u32(&a) == expected && a == b);
    }
    nt_seed(73); nt_tensor_rand(legacy_reference, 1);
    nt_seed(73);
    srand(31); int libc_expected = rand(); srand(31);
    const float weights[] = {1, 2, 3};
    for (int i = 0; i < 1000; i++) {
        int chosen;
        uint32_t index;
        nt_rng_seed(&a, (uint64_t)i);
        (void)nt_rng_u32(&a);
        (void)nt_rng_uniform(&a);
        CHECK(nt_rng_index(&a, 17, &index) == 0);
        CHECK(nt_rng_categorical(&a, weights, 3, 0.8f, &chosen) == 0);
    }
    CHECK(rand() == libc_expected);
    nt_tensor_rand(legacy, 1);
    CHECK(memcmp(legacy->data, legacy_reference->data, 8 * sizeof(float)) == 0);
    CHECK(weights[0] == 1 && weights[1] == 2 && weights[2] == 3);
    nt_tensor_free(legacy); nt_tensor_free(legacy_reference);
    puts("PASS owned snapshots and bidirectional libc/legacy nt_seed isolation");
}

int main(void) {
    vectors_and_reference();
    bounds_and_cdf();
    invalid_calls();
    ownership_and_isolation();
    printf("PASS owned sampling: %u checks\n", checks);
    return 0;
}
