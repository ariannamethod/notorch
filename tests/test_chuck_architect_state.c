// Chuck lives: locale-independent JSON and internally consistent checkpoints.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "chuck_architect.h"
#include <errno.h>
#include <locale.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

static unsigned checks, skips;
static const char *case_name;
#define CHECK(expr, why) do { ++checks; if (!(expr)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", case_name, __LINE__, why); return 0; \
} } while (0)

static const char *const numeric_json[] = {
    "{\"mode\":\"learned\",\"learning_rate\":0.03,\"exploration\":0.15,\"dampen_min\":0.3,\"lr_scale_min\":0.05,\"noise_max\":0.01}",
    "{\"mode\":\"learned\",\"learning_rate\":3e-2,\"exploration\":15E-2,\"seed\":4294967295}",
    "{\"mode\":\"learned\",\"learning_rate\":1.401298464324817e-45,\"exploration\":-0.0}",
    "{\"mode\":\"learned\",\"learning_rate\":1.17549435082228750797e-38}",
    "{\"mode\":\"learned\",\"learning_rate\":0.9999999701976776123046875}"
};
#define NUMERIC_CASES (sizeof numeric_json / sizeof *numeric_json)
static nt_chuck_architect_config numeric_expected[NUMERIC_CASES];

typedef struct { locale_t locale; const char *decimal; int failed; } locale_thread;

static void *parse_in_thread(void *arg) {
    locale_thread *t = (locale_thread *)arg;
    locale_t previous = uselocale(t->locale);
    if (!previous) { t->failed = 1; return NULL; }
    for (int repeat = 0; repeat < 200 && !t->failed; ++repeat) {
        for (size_t k = 0; k < NUMERIC_CASES; ++k) {
            nt_chuck_architect_config result;
            char *end;
            // localeconv uses shared libc storage on some targets; strtod
            // directly checks this reader's locale without that shared buffer.
            double marker = strtod(t->decimal[0] == ',' ? "0,25" : "0.25", &end);
            if (nt_chuck_architect_config_parse_json(&result, numeric_json[k], NULL, 0) ||
                memcmp(&result, &numeric_expected[k], sizeof result) ||
                uselocale((locale_t)0) != t->locale ||
                marker != .25 || *end) {
                t->failed = 1;
                break;
            }
        }
    }
    if (!uselocale(previous)) t->failed = 1;
    return NULL;
}

static int test_numeric_locales(void) {
    const char *initial = setlocale(LC_NUMERIC, NULL);
    char *saved = initial ? strdup(initial) : NULL;
    CHECK(saved != NULL && setlocale(LC_NUMERIC, "C"), "save caller locale and select C reference");
    for (size_t k = 0; k < NUMERIC_CASES; ++k) {
        char error[128];
        CHECK(nt_chuck_architect_config_parse_json(&numeric_expected[k], numeric_json[k], error, sizeof error) == 0,
              "C numeric reference accepted, including float/subnormal and integer boundaries");
    }
    CHECK(numeric_expected[0].learning_rate == (float)strtod("0.03", NULL) &&
          numeric_expected[1].seed == UINT32_MAX &&
          numeric_expected[2].learning_rate == nextafterf(0, 1) &&
          signbit(numeric_expected[2].exploration) &&
          numeric_expected[4].learning_rate == (float)strtod("0.9999999701976776123046875", NULL),
          "libc C rounding, uint32 boundary, minimum subnormal and signed zero preserved");

    const char *requested = getenv("NT_CHUCK_TEST_LOCALE");
    const char *candidates[] = {requested, "de_DE.UTF-8", "de_DE.utf8", "de_DE",
        "fr_FR.UTF-8", "fr_FR.utf8", "fr_FR", "ru_RU.UTF-8", NULL};
    const char *chosen = NULL;
    for (size_t i = requested ? 0 : 1; i < sizeof candidates / sizeof *candidates; ++i) {
        if (candidates[i] && setlocale(LC_NUMERIC, candidates[i]) &&
            !strcmp(localeconv()->decimal_point, ",")) { chosen = candidates[i]; break; }
        if (requested) break;
    }
    if (!chosen) {
        CHECK(!requested, "explicitly requested comma locale must be available");
        puts("SKIPPED numeric comma/thread locale: none installed; use NT_CHUCK_TEST_LOCALE and LOCPATH");
        CHECK(setlocale(LC_NUMERIC, saved) != NULL, "restore caller locale");
        free(saved);
        return 2;
    }
    printf("numeric locale fixture: %s (decimal comma)\n", chosen);
    for (size_t k = 0; k < NUMERIC_CASES; ++k) {
        nt_chuck_architect_config result;
        CHECK(nt_chuck_architect_config_parse_json(&result, numeric_json[k], NULL, 0) == 0 &&
              memcmp(&result, &numeric_expected[k], sizeof result) == 0,
              "dot-decimal JSON is byte-identical under comma LC_NUMERIC");
        CHECK(!strcmp(localeconv()->decimal_point, ","), "parser preserves caller's process locale");
    }
    nt_chuck_architect_config unchanged = numeric_expected[0], invalid = unchanged;
    CHECK(nt_chuck_architect_config_parse_json(&invalid,
          "{\"mode\":\"learned\",\"learning_rate\":0,03}", NULL, 0) != 0 &&
          memcmp(&invalid, &unchanged, sizeof invalid) == 0,
          "comma-decimal JSON remains invalid and refusal remains transactional");

    locale_thread contexts[2] = {{newlocale(LC_NUMERIC_MASK, "C", (locale_t)0), ".", 0},
        {newlocale(LC_NUMERIC_MASK, chosen, (locale_t)0), ",", 0}};
    CHECK(contexts[0].locale && contexts[1].locale, "create simultaneous C/comma thread locales");
    pthread_t threads[2];
    CHECK(pthread_create(&threads[0], NULL, parse_in_thread, &contexts[0]) == 0,
          "start C-locale reader");
    CHECK(pthread_create(&threads[1], NULL, parse_in_thread, &contexts[1]) == 0,
          "start comma-locale reader");
    CHECK(pthread_join(threads[0], NULL) == 0 && pthread_join(threads[1], NULL) == 0,
          "join simultaneous readers");
    CHECK(!contexts[0].failed && !contexts[1].failed,
          "2000 concurrent conversions retain exact values and each thread's locale");
    freelocale(contexts[0].locale); freelocale(contexts[1].locale);
    CHECK(!strcmp(localeconv()->decimal_point, ","), "thread parsing preserves process locale");
    CHECK(setlocale(LC_NUMERIC, saved) != NULL, "restore original locale");
    free(saved);
    return 1;
}

enum { LIFE_HEADER = 24, LIFE_PAYLOAD = 992, LIFE_BYTES = 1016 };
static uint64_t checksum(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static uint32_t float_bits(float value) { uint32_t u; memcpy(&u, &value, 4); return u; }
static void put_integer(unsigned char *p, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = (unsigned char)(value >> (8 * i));
}
static int write_life(const char *path, unsigned char *bytes) {
    put_integer(bytes + 16, checksum(bytes + LIFE_HEADER, LIFE_PAYLOAD), 8);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(bytes, 1, LIFE_BYTES, f) == LIFE_BYTES;
    if (fclose(f)) ok = 0;
    return ok;
}

static nt_tensor *body;
static int pending_life(nt_chuck_architect *a) {
    nt_chuck_architect_config c;
    nt_chuck_architect_config_default(&c);
    c.mode = NT_CHUCK_ARCHITECT_LEARNED;
    c.exploration = 0;
    if (nt_chuck_architect_init(a, &c)) return 0;
    nt_tape_start();
    body = nt_tensor_new(1);
    if (!body) return 0;
    body->data[0] = 1;
    int slot = nt_tape_param(body);
    if (slot < 0) return 0;
    nt_tape_get()->entries[slot].grad = nt_tensor_new(1);
    if (!nt_tape_get()->entries[slot].grad) return 0;
    nt_tape_get()->entries[slot].grad->data[0] = 1;
    return nt_chuck_architect_step(a, .01f, 1, NULL) == 0;
}

static int test_hostile_lives(void) {
    char directory[] = "/tmp/notorch-architect-state-XXXXXX";
    CHECK(mkdtemp(directory), "create temporary life directory");
    char original[256], forged[256];
    snprintf(original, sizeof original, "%s/original.bin", directory);
    snprintf(forged, sizeof forged, "%s/forged.bin", directory);
    nt_chuck_architect pending, resident;
    CHECK(pending_life(&pending), "execute a real pending training action");
    CHECK(nt_chuck_architect_save(&pending, original) == 0, "save pending life");
    unsigned char bytes[LIFE_BYTES];
    FILE *f = fopen(original, "rb");
    CHECK(f && fread(bytes, 1, sizeof bytes, f) == sizeof bytes && fgetc(f) == EOF,
          "read exact v1 canonical life");
    CHECK(fclose(f) == 0, "close life");
    CHECK(nt_chuck_architect_load(&resident, original) == 0, "load valid pending life");
    uint64_t valid_hash = nt_chuck_architect_hash(&resident);
    struct mutation { const char *name; size_t offset; unsigned width; uint64_t value; } cases[] = {
        {"payload version", 0, 4, 2}, {"mode", 4, 4, 99}, {"empty identity", 8, 1, 0},
        {"unavailable hold", 72, 4, NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_PUSH)},
        {"invalid dampen limit", 76, 4, float_bits(.2f)},
        {"nonfinite learning rate", 100, 4, float_bits(NAN)},
        {"zero configured seed", 108, 4, 0}, {"zero live RNG", 112, 4, 0},
        {"missing pending decision count", 116, 8, 0}, {"updates exceed decisions", 124, 8, 2},
        {"pending flag", 132, 4, 2}, {"history without update", 136, 4, 1},
        {"nonfinite weight", 140, 4, float_bits(NAN)}, {"unbounded weight", 140, 4, float_bits(17)},
        {"unbounded prior reward", 800, 4, float_bits(2)},
        {"nonfinite observation", 804, 4, float_bits(INFINITY)},
        {"unrepresentable step", 848, 4, UINT32_MAX}, {"history ring length", 860, 4, 17},
        {"nonfinite feature", 864, 4, float_bits(NAN)}, {"unbounded feature", 864, 4, float_bits(2)},
        {"nonfinite prediction", 928, 4, float_bits(NAN)}, {"unbounded prediction", 928, 4, float_bits(145)},
        {"invalid action", 940, 4, NT_CHUCK_ACTION_COUNT}, {"nonzero primitive argument", 944, 4, float_bits(.1f)},
        {"wrong pending sequence", 948, 8, 2}, {"explored flag", 956, 4, 2},
        {"nonfinite hidden", 960, 4, float_bits(NAN)}, {"unbounded hidden", 960, 4, float_bits(1.1f)},
        {"prediction contradicts live network", 928, 4, float_bits(.75f)},
        {"hidden contradicts live network", 960, 4, float_bits(.875f)},
        {"feature contradicts observation", 864, 4, float_bits(.875f)},
        {"loss contradicts captured features", 804, 4, float_bits(2)},
        {"unexplored action contradicts argmax", 940, 4, NT_CHUCK_ACTION_BRAKE},
        {"exploration with zero probability", 956, 4, 1}
    };
    int all_rejected = 1;
    for (size_t k = 0; k < sizeof cases / sizeof *cases; ++k) {
        unsigned char changed[LIFE_BYTES];
        memcpy(changed, bytes, sizeof changed);
        put_integer(changed + LIFE_HEADER + cases[k].offset, cases[k].value, cases[k].width);
        CHECK(write_life(forged, changed), "write hostile payload with its correct checksum");
        resident = pending;
        int rc = nt_chuck_architect_load(&resident, forged);
        ++checks;
        if (rc == 0 || nt_chuck_architect_hash(&resident) != valid_hash) {
            fprintf(stderr, "FAIL hostile life accepted or destination changed: %s (rc=%d)\n", cases[k].name, rc);
            all_rejected = 0;
        }
    }

    unsigned char rounded[LIFE_BYTES];
    memcpy(rounded, bytes, sizeof rounded);
    put_integer(rounded + LIFE_HEADER + 960,
                float_bits(nextafterf(pending.pending_hidden[0], 1)), 4);
    CHECK(write_life(forged, rounded) && nt_chuck_architect_load(&resident, forged) == 0 &&
          nt_chuck_architect_hash(&resident) == checksum(rounded + LIFE_HEADER, LIFE_PAYLOAD),
          "one-ULP hidden variation within documented tolerance preserves recorded bits");

    // After feedback, historical caches describe the prior network. Such a
    // completed life remains valid and round-trips without recomputing them.
    nt_chuck_architect_receipt receipt;
    CHECK(nt_chuck_architect_feedback(&pending, .5f, &receipt) == 0 && pending.b2[0] > 0,
          "actual positive outcome reaches the pending selected head");
    CHECK(nt_chuck_architect_save(&pending, original) == 0 &&
          nt_chuck_architect_load(&resident, original) == 0 &&
          nt_chuck_architect_hash(&pending) == nt_chuck_architect_hash(&resident),
          "completed life with historical predictions round-trips exactly");
    nt_tape_destroy(); nt_tensor_free(body); body = NULL;
    unlink(original); unlink(forged); rmdir(directory);
    CHECK(all_rejected, "all malformed, correctly checksummed lives are refused transactionally");
    return 1;
}

static int test_pending_feedback(void) {
    nt_chuck_architect a, broken;
    CHECK(pending_life(&a), "execute pending action for feedback validation");
    broken = a;
    broken.pending_decision.scores[0] = .75f;
    nt_chuck_architect before = broken;
    nt_chuck_architect_receipt receipt, previous;
    memset(&receipt, 0xa5, sizeof receipt); previous = receipt;
    CHECK(nt_chuck_architect_feedback(&broken, .5f, &receipt) != 0 &&
          memcmp(&broken, &before, sizeof broken) == 0 &&
          memcmp(&receipt, &previous, sizeof receipt) == 0,
          "inconsistent pending prediction cannot reverse measured credit or mutate outputs");
    CHECK(nt_chuck_architect_feedback(&a, .5f, &receipt) == 0 &&
          receipt.reward > 0 && a.b2[0] > 0,
          "unchanged executed action acquires the positive consequence");
    nt_tape_destroy(); nt_tensor_free(body); body = NULL;
    return 1;
}

int main(void) {
    unsigned passed = 0, failed = 0;
    struct { const char *name; int (*run)(void); } cases[] = {
        {"numeric_locales", test_numeric_locales},
        {"hostile_lives", test_hostile_lives},
        {"pending_feedback", test_pending_feedback}
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; ++i) {
        case_name = cases[i].name;
        int result = cases[i].run();
        if (result == 2) ++skips;
        else if (result) { ++passed; printf("PASS %s\n", case_name); }
        else ++failed;
    }
    printf("Chuck state: %u passed, %u failed, %u checks, %u locale skips\n", passed, failed, checks, skips);
    return failed ? 1 : 0;
}
