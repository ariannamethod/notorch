// Included once by notorch.c. Public contract: chuck_architect.h.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef NOTORCH_CHUCK_ARCHITECT_IMPL_H
#define NOTORCH_CHUCK_ARCHITECT_IMPL_H

#include "chuck_architect.h"
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

#define NT_CA_VERSION UINT32_C(1)
#define NT_CA_PAYLOAD_BYTES 992
#define NT_CA_FILE_BYTES (24 + NT_CA_PAYLOAD_BYTES)

static float nt_ca_clip(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int nt_ca_life_valid(const char *s) {
    size_t n;
    for (n = 0; n < NT_CHUCK_ARCHITECT_LIFE_ID; ++n) {
        unsigned char c = (unsigned char)s[n];
        if (!c) return n != 0;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
            return 0;
    }
    return 0;
}

static int nt_ca_config_valid(const nt_chuck_architect_config *c) {
    if (!c || c->mode < NT_CHUCK_ARCHITECT_DISABLED ||
        c->mode > NT_CHUCK_ARCHITECT_LEARNED || !nt_ca_life_valid(c->life_id) ||
        !isfinite(c->learning_rate) || c->learning_rate <= 0 ||
        c->learning_rate > 1 || !isfinite(c->exploration) ||
        c->exploration < 0 || c->exploration > 1 || !c->seed)
        return 0;
    if (nt_chuck_action_limits_validate(&c->limits) != NT_CHUCK_OK) return 0;
    if (c->mode == NT_CHUCK_ARCHITECT_LEARNED &&
        !(c->limits.enabled_actions & NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_HOLD)))
        return 0;
    if (c->mode != NT_CHUCK_ARCHITECT_LEARNED &&
        (!(c->limits.enabled_actions & NT_CHUCK_ACTION_BIT(NT_CHUCK_ACTION_LEGACY)) ||
         c->limits.dampen_min != NT_CHUCK_DAMP_LO || c->limits.dampen_max != NT_CHUCK_DAMP_HI ||
         c->limits.lr_scale_min != NT_CHUCK_LR_SCALE_LO || c->limits.lr_scale_max != NT_CHUCK_LR_SCALE_HI ||
         c->limits.noise_min != NT_CHUCK_NOISE_LO || c->limits.noise_max != NT_CHUCK_NOISE_HI))
        return 0;
    return 1;
}

void nt_chuck_architect_config_default(nt_chuck_architect_config *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->mode = NT_CHUCK_ARCHITECT_LEGACY;
    memcpy(out->life_id, "chuck", 6);
    nt_chuck_action_limits_default(&out->limits);
    out->learning_rate = 0.03f;
    out->exploration = 0.15f;
    out->seed = 1;
}

typedef struct {
    const char *start, *at;
    char *error;
    size_t error_size;
    int failed;
} nt_ca_json;

static int nt_ca_json_error(nt_ca_json *j, const char *why) {
    if (!j->failed && j->error && j->error_size)
        snprintf(j->error, j->error_size, "%s at byte %lu", why,
                 (unsigned long)(j->at - j->start));
    j->failed = 1;
    return 0;
}

static void nt_ca_json_space(nt_ca_json *j) {
    while (*j->at == ' ' || *j->at == '\n' || *j->at == '\r' || *j->at == '\t')
        ++j->at;
}

static int nt_ca_json_take(nt_ca_json *j, char token) {
    nt_ca_json_space(j);
    if (*j->at != token) return nt_ca_json_error(j, "unexpected JSON token");
    ++j->at;
    return 1;
}

static int nt_ca_hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int nt_ca_json_string(nt_ca_json *j, char *out, size_t cap) {
    size_t n = 0;
    if (!nt_ca_json_take(j, '"')) return 0;
    while (*j->at && *j->at != '"') {
        unsigned char c = (unsigned char)*j->at++;
        if (c < 32) return nt_ca_json_error(j, "control character in string");
        if (c == '\\') {
            if (!*j->at) return nt_ca_json_error(j, "unfinished string escape");
            c = (unsigned char)*j->at++;
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    unsigned u = 0;
                    for (int k = 0; k < 4; ++k) {
                        int h = nt_ca_hex((unsigned char)*j->at);
                        if (h < 0) return nt_ca_json_error(j, "bad unicode escape");
                        u = u * 16 + (unsigned)h;
                        ++j->at;
                    }
                    if (!u || u > 127) return nt_ca_json_error(j, "ASCII identifier required");
                    c = (unsigned char)u;
                    break;
                }
                default: return nt_ca_json_error(j, "bad string escape");
            }
        }
        if (!c || n + 1 >= cap) return nt_ca_json_error(j, "string too long");
        out[n++] = (char)c;
    }
    if (!nt_ca_json_take(j, '"')) return 0;
    out[n] = 0;
    return 1;
}

// libc keeps decimal conversion/rounding; the explicit C locale keeps JSON's
// decimal point independent of the caller's process or thread locale. This
// immutable locale is shared after pthread_once and lives with the library.
static locale_t nt_ca_numeric_locale;
static pthread_once_t nt_ca_numeric_once = PTHREAD_ONCE_INIT;

static void nt_ca_numeric_init(void) {
    nt_ca_numeric_locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
}

// Recognize the JSON grammar before strtod_l: NaN, infinity, hex, leading zeros,
// missing fraction digits and incomplete exponents never reach the schema.
static int nt_ca_json_number(nt_ca_json *j, double *out) {
    const char *begin, *end;
    char *parsed;
    nt_ca_json_space(j);
    begin = j->at;
    if (*j->at == '-') ++j->at;
    if (*j->at == '0') {
        ++j->at;
        if (*j->at >= '0' && *j->at <= '9')
            return nt_ca_json_error(j, "leading zero in number");
    } else if (*j->at >= '1' && *j->at <= '9') {
        do { ++j->at; } while (*j->at >= '0' && *j->at <= '9');
    } else return nt_ca_json_error(j, "number required");
    if (*j->at == '.') {
        ++j->at;
        if (*j->at < '0' || *j->at > '9')
            return nt_ca_json_error(j, "fraction digit required");
        do { ++j->at; } while (*j->at >= '0' && *j->at <= '9');
    }
    if (*j->at == 'e' || *j->at == 'E') {
        ++j->at;
        if (*j->at == '+' || *j->at == '-') ++j->at;
        if (*j->at < '0' || *j->at > '9')
            return nt_ca_json_error(j, "exponent digit required");
        do { ++j->at; } while (*j->at >= '0' && *j->at <= '9');
    }
    end = j->at;
    if (pthread_once(&nt_ca_numeric_once, nt_ca_numeric_init) || !nt_ca_numeric_locale)
        return nt_ca_json_error(j, "C numeric locale unavailable");
    errno = 0;
    *out = strtod_l(begin, &parsed, nt_ca_numeric_locale);
    if (parsed != end || errno == ERANGE || !isfinite(*out))
        return nt_ca_json_error(j, "number out of range");
    return 1;
}

static int nt_ca_action_name(const char *name) {
    static const char *const names[] = {
        "legacy", "hold", "brake", "push", "set_global_dampen", "set_lr_scale", "set_noise"
    };
    for (int k = 0; k < NT_CHUCK_ACTION_COUNT; ++k)
        if (!strcmp(name, names[k])) return k;
    return -1;
}

static int nt_ca_json_actions(nt_ca_json *j, uint32_t *mask) {
    char name[40];
    uint32_t bits = 0;
    if (!nt_ca_json_take(j, '[')) return 0;
    nt_ca_json_space(j);
    if (*j->at == ']') { ++j->at; *mask = 0; return 1; }
    for (;;) {
        if (!nt_ca_json_string(j, name, sizeof(name))) return 0;
        int k = nt_ca_action_name(name);
        if (k < 0) return nt_ca_json_error(j, "unknown action");
        uint32_t bit = NT_CHUCK_ACTION_BIT(k);
        if (bits & bit) return nt_ca_json_error(j, "duplicate action");
        bits |= bit;
        nt_ca_json_space(j);
        if (*j->at == ']') { ++j->at; *mask = bits; return 1; }
        if (!nt_ca_json_take(j, ',')) return 0;
    }
}

int nt_chuck_architect_config_parse_json(nt_chuck_architect_config *out,
    const char *json, char *error, size_t error_size) {
    static const char *const keys[] = {
        "mode", "life_id", "enabled_actions", "dampen_min", "dampen_max",
        "lr_scale_min", "lr_scale_max", "noise_min", "noise_max",
        "learning_rate", "exploration", "seed", "version"
    };
    nt_chuck_architect_config c;
    nt_ca_json j;
    uint32_t seen = 0;
    char key[64], value[64];
    if (error && error_size) error[0] = 0;
    if (!out) return NT_CHUCK_E_ARGUMENT;
    nt_chuck_architect_config_default(&c);
    if (!json) { *out = c; return NT_CHUCK_OK; }
    j.start = j.at = json; j.error = error; j.error_size = error_size; j.failed = 0;
    nt_ca_json_space(&j);
    if (!*j.at) { *out = c; return NT_CHUCK_OK; }
    if (!nt_ca_json_take(&j, '{')) return NT_CHUCK_E_CONFIG;
    nt_ca_json_space(&j);
    if (*j.at != '}') for (;;) {
        int k = -1;
        double v = 0;
        if (!nt_ca_json_string(&j, key, sizeof(key))) break;
        for (int i = 0; i < 13; ++i) if (!strcmp(keys[i], key)) { k = i; break; }
        if (k < 0) { nt_ca_json_error(&j, "unknown field"); break; }
        if (seen & (UINT32_C(1) << k)) { nt_ca_json_error(&j, "duplicate field"); break; }
        seen |= UINT32_C(1) << k;
        if (!nt_ca_json_take(&j, ':')) break;
        if (k == 0 || k == 1) {
            if (!nt_ca_json_string(&j, value, sizeof(value))) break;
            if (k == 1) {
                memset(c.life_id, 0, sizeof(c.life_id));
                memcpy(c.life_id, value, strlen(value) + 1);
            } else if (!strcmp(value, "disabled")) c.mode = NT_CHUCK_ARCHITECT_DISABLED;
            else if (!strcmp(value, "legacy")) c.mode = NT_CHUCK_ARCHITECT_LEGACY;
            else if (!strcmp(value, "learned")) c.mode = NT_CHUCK_ARCHITECT_LEARNED;
            else { nt_ca_json_error(&j, "unknown mode"); break; }
        } else if (k == 2) {
            if (!nt_ca_json_actions(&j, &c.limits.enabled_actions)) break;
        } else {
            if (!nt_ca_json_number(&j, &v)) break;
            if (k == 11 || k == 12) {
                if (v < 1 || v > UINT32_MAX || floor(v) != v || (k == 12 && v != 1)) {
                    nt_ca_json_error(&j, "invalid seed or version"); break;
                }
                if (k == 11) c.seed = (uint32_t)v;
            } else {
                if (v > FLT_MAX || v < -FLT_MAX || (v != 0 && (float)v == 0)) {
                    nt_ca_json_error(&j, "float out of range"); break;
                }
                float f = (float)v;
                switch (k) {
                    case 3: c.limits.dampen_min = f; break;
                    case 4: c.limits.dampen_max = f; break;
                    case 5: c.limits.lr_scale_min = f; break;
                    case 6: c.limits.lr_scale_max = f; break;
                    case 7: c.limits.noise_min = f; break;
                    case 8: c.limits.noise_max = f; break;
                    case 9: c.learning_rate = f; break;
                    case 10: c.exploration = f; break;
                }
            }
        }
        nt_ca_json_space(&j);
        if (*j.at == '}') break;
        if (!nt_ca_json_take(&j, ',')) break;
    }
    if (!j.failed && !nt_ca_json_take(&j, '}')) j.failed = 1;
    nt_ca_json_space(&j);
    if (!j.failed && *j.at) nt_ca_json_error(&j, "trailing JSON data");
    if (!j.failed && !nt_ca_config_valid(&c)) nt_ca_json_error(&j, "invalid configuration, action availability or bounds");
    if (j.failed) return NT_CHUCK_E_CONFIG;
    *out = c;
    return NT_CHUCK_OK;
}

static uint32_t nt_ca_random(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static float nt_ca_uniform(uint32_t *state) {
    return (float)(nt_ca_random(state) >> 8) * (1.0f / 16777216.0f);
}

int nt_chuck_architect_init(nt_chuck_architect *out,
    const nt_chuck_architect_config *config) {
    nt_chuck_architect a;
    nt_chuck_architect_config defaults;
    if (!out) return NT_CHUCK_E_ARGUMENT;
    if (!config) { nt_chuck_architect_config_default(&defaults); config = &defaults; }
    if (!nt_ca_config_valid(config)) return NT_CHUCK_E_CONFIG;
    memset(&a, 0, sizeof(a));
    a.version = NT_CA_VERSION;
    a.config = *config;
    // Canonicalize unused identity bytes for stable hashes across callers.
    size_t id_len = strlen(a.config.life_id);
    memset(a.config.life_id + id_len + 1, 0, sizeof(a.config.life_id) - id_len - 1);
    a.rng = config->seed;
    if (config->mode == NT_CHUCK_ARCHITECT_LEARNED)
        for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h)
            for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i)
                a.w1[h][i] = (nt_ca_uniform(&a.rng) * 2 - 1) * 0.15f;
    *out = a;
    return NT_CHUCK_OK;
}

static int nt_ca_observation_finite(const nt_chuck_observation *o) {
    if (!o) return 0;
    const float fields[] = {o->loss, o->loss_ema, o->loss_trend, o->macro_ema,
        o->best_macro, o->dampen, o->lr_scale, o->noise, o->grad_norm,
        o->grad_trend, o->frozen_fraction};
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
        if (!isfinite(fields[i])) return 0;
    return 1;
}

static int nt_ca_observation_valid(const nt_chuck_observation *o) {
    if (!nt_ca_observation_finite(o)) return 0;
    return o->dampen >= NT_CHUCK_DAMP_LO && o->dampen <= NT_CHUCK_DAMP_HI &&
        o->lr_scale >= NT_CHUCK_LR_SCALE_LO && o->lr_scale <= NT_CHUCK_LR_SCALE_HI &&
        o->noise >= NT_CHUCK_NOISE_LO && o->noise <= NT_CHUCK_NOISE_HI &&
        o->grad_norm >= 0 && o->frozen_fraction >= 0 && o->frozen_fraction <= 1 &&
        o->step >= 0 && o->stag >= 0 && o->macro_stag >= 0 &&
        o->history_len >= 0 && o->history_len <= NT_CHUCK_WINDOW;
}

static int nt_ca_floats_valid(const float *p, size_t n, float bound) {
    for (size_t i = 0; i < n; ++i)
        if (!isfinite(p[i]) || fabsf(p[i]) > bound) return 0;
    return 1;
}

static int nt_ca_pending_coherent(const nt_chuck_architect *a);

static int nt_ca_state_valid(const nt_chuck_architect *a) {
    if (!a || a->version != NT_CA_VERSION || !nt_ca_config_valid(&a->config) ||
        !a->rng || (a->pending != 0 && a->pending != 1) ||
        (a->has_history != 0 && a->has_history != 1) || a->updates > a->decisions ||
        a->decisions - a->updates != (uint64_t)a->pending ||
        a->has_history != (a->updates != 0) ||
        !isfinite(a->prev_loss) || !isfinite(a->prev_trend) ||
        !isfinite(a->prev_reward) || fabsf(a->prev_reward) > 1)
        return 0;
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h)
        if (!nt_ca_floats_valid(a->w1[h], NT_CHUCK_ARCHITECT_FEATURES, 16)) return 0;
    if (!nt_ca_floats_valid(a->b1, NT_CHUCK_ARCHITECT_HIDDEN, 16) ||
        !nt_ca_floats_valid(a->b2, NT_CHUCK_ARCHITECT_HEADS, 16)) return 0;
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k)
        if (!nt_ca_floats_valid(a->w2[k], NT_CHUCK_ARCHITECT_HIDDEN, 16)) return 0;
    const nt_chuck_architect_decision *d = &a->pending_decision;
    if (!nt_ca_observation_finite(&d->observation) ||
        !nt_ca_floats_valid(d->features, NT_CHUCK_ARCHITECT_FEATURES, 1) ||
        !nt_ca_floats_valid(d->scores, NT_CHUCK_ARCHITECT_HEADS, 144) ||
        !nt_ca_floats_valid(a->pending_hidden, NT_CHUCK_ARCHITECT_HIDDEN, 1) ||
        !isfinite(d->action.value) || d->action.value != 0 ||
        (d->explored != 0 && d->explored != 1)) return 0;
    if (a->decisions) {
        if (!nt_ca_observation_valid(&d->observation) ||
            d->sequence != a->decisions || d->action.kind < NT_CHUCK_ACTION_HOLD ||
            d->action.kind > NT_CHUCK_ACTION_PUSH) return 0;
    } else if (d->sequence || d->action.kind != NT_CHUCK_ACTION_LEGACY) return 0;
    if (a->pending && (a->config.mode != NT_CHUCK_ARCHITECT_LEARNED ||
        !(a->config.limits.enabled_actions & NT_CHUCK_ACTION_BIT(d->action.kind)))) return 0;
    if (a->pending && !nt_ca_pending_coherent(a)) return 0;
    return 1;
}

static float nt_ca_ratio(double n, double d) {
    return (float)fmax(-1.0, fmin(1.0, n / d));
}

static void nt_ca_features(const nt_chuck_architect *a,
    const nt_chuck_observation *o, float *f) {
    f[0] = nt_ca_ratio(o->loss, 1.0 + fabs((double)o->loss));
    f[1] = nt_ca_clip(o->loss_trend, -1, 1);
    f[2] = nt_ca_ratio((double)o->loss - o->loss_ema, 1.0 + fabs((double)o->loss_ema));
    f[3] = nt_ca_ratio((double)o->macro_ema - o->best_macro, 1.0 + fabs((double)o->best_macro));
    f[4] = o->dampen / NT_CHUCK_DAMP_HI;
    f[5] = o->lr_scale / NT_CHUCK_LR_SCALE_HI;
    f[6] = o->noise / NT_CHUCK_NOISE_HI;
    f[7] = nt_ca_ratio(o->grad_norm, 1.0 + o->grad_norm);
    f[8] = nt_ca_clip(o->grad_trend, -1, 1);
    f[9] = o->frozen_fraction;
    f[10] = (float)((double)o->stag / ((double)o->stag + NT_CHUCK_STAG_STEPS));
    f[11] = (float)((double)o->macro_stag / ((double)o->macro_stag + NT_CHUCK_MACRO_PAT));
    f[12] = (float)((double)o->step / ((double)o->step + NT_CHUCK_MACRO_INT));
    f[13] = a->has_history ? nt_ca_ratio((double)o->loss - a->prev_loss,
                                       1.0 + fabs((double)a->prev_loss)) : 0;
    f[14] = a->has_history ? nt_ca_clip(a->prev_trend, -1, 1) : 0;
    f[15] = a->has_history ? a->prev_reward : 0;
}

static void nt_ca_forward(const nt_chuck_architect *a, const float *f,
    float *hidden, float *scores) {
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h) {
        float sum = a->b1[h];
        for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i) sum += a->w1[h][i] * f[i];
        hidden[h] = tanhf(sum);
    }
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k) {
        float sum = a->b2[k];
        for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h) sum += a->w2[k][h] * hidden[h];
        scores[k] = sum;
    }
}

static int nt_ca_cache_near(const float *cached, const float *expected, size_t n) {
    // Preserve recorded values for exact continuation. Recomputed caches may
    // differ across CPU/libm implementations by this absolute+relative bound.
    for (size_t i = 0; i < n; ++i)
        if (fabsf(cached[i] - expected[i]) > 1e-6f + 1e-5f * fabsf(expected[i])) return 0;
    return 1;
}

static int nt_ca_pending_coherent(const nt_chuck_architect *a) {
    const nt_chuck_architect_decision *d = &a->pending_decision;
    float features[NT_CHUCK_ARCHITECT_FEATURES], hidden[NT_CHUCK_ARCHITECT_HIDDEN];
    float scores[NT_CHUCK_ARCHITECT_HEADS];
    // Before feedback, the acquired weights and temporal history are exactly
    // those which made this pending decision. Completed caches are historical.
    nt_ca_features(a, &d->observation, features);
    if (!nt_ca_cache_near(d->features, features, NT_CHUCK_ARCHITECT_FEATURES)) return 0;
    nt_ca_forward(a, d->features, hidden, scores);
    if (!nt_ca_cache_near(a->pending_hidden, hidden, NT_CHUCK_ARCHITECT_HIDDEN) ||
        !nt_ca_cache_near(d->scores, scores, NT_CHUCK_ARCHITECT_HEADS)) return 0;
    int best = -1, allowed = 0;
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k) {
        if (!(a->config.limits.enabled_actions & NT_CHUCK_ACTION_BIT(k + NT_CHUCK_ACTION_HOLD))) continue;
        ++allowed;
        if (best < 0 || d->scores[k] > d->scores[best]) best = k;
    }
    if (d->explored) return a->config.exploration > 0 && allowed > 1;
    return best >= 0 && d->action.kind == (nt_chuck_action_kind)(best + NT_CHUCK_ACTION_HOLD);
}

static int nt_ca_choose(nt_chuck_architect *a, const nt_chuck_observation *o,
    nt_chuck_architect_decision *d, float *hidden) {
    int allowed[NT_CHUCK_ARCHITECT_HEADS], n = 0, best = -1;
    if (!nt_ca_observation_valid(o)) return NT_CHUCK_E_STATE;
    memset(d, 0, sizeof(*d));
    d->observation = *o;
    nt_ca_features(a, o, d->features);
    nt_ca_forward(a, d->features, hidden, d->scores);
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k) {
        if (!(a->config.limits.enabled_actions & NT_CHUCK_ACTION_BIT(k + NT_CHUCK_ACTION_HOLD)))
            continue;
        allowed[n++] = k;
        if (best < 0 || d->scores[k] > d->scores[best]) best = k;
    }
    if (best < 0) return NT_CHUCK_E_ACTION;
    if (n > 1 && a->config.exploration > 0 && nt_ca_uniform(&a->rng) < a->config.exploration) {
        best = allowed[nt_ca_random(&a->rng) % (uint32_t)n];
        d->explored = 1;
    }
    d->action.kind = (nt_chuck_action_kind)(best + NT_CHUCK_ACTION_HOLD);
    d->action.value = 0;
    d->sequence = a->decisions + 1;
    return NT_CHUCK_OK;
}

int nt_chuck_architect_select(const nt_chuck_architect *architect,
    const nt_chuck_observation *observation, nt_chuck_action *action) {
    if (!action) return NT_CHUCK_E_ARGUMENT;
    if (architect && !nt_ca_state_valid(architect)) return NT_CHUCK_E_STATE;
    if (!architect || architect->config.mode == NT_CHUCK_ARCHITECT_DISABLED ||
        architect->config.mode == NT_CHUCK_ARCHITECT_LEGACY) {
        nt_chuck_action legacy = {NT_CHUCK_ACTION_LEGACY, 0};
        *action = legacy;
        return NT_CHUCK_OK;
    }
    if (architect->pending || architect->decisions == UINT64_MAX)
        return NT_CHUCK_E_STATE;
    nt_chuck_architect copy = *architect;
    nt_chuck_architect_decision d;
    float hidden[NT_CHUCK_ARCHITECT_HIDDEN];
    int rc = nt_ca_choose(&copy, observation, &d, hidden);
    if (rc == NT_CHUCK_OK) *action = d.action;
    return rc;
}

int nt_chuck_architect_step(nt_chuck_architect *architect, float lr, float loss,
    nt_chuck_architect_decision *decision) {
    nt_chuck_architect_decision d;
    if (architect && !nt_ca_state_valid(architect)) return NT_CHUCK_E_STATE;
    memset(&d, 0, sizeof(d));
    if (!architect || architect->config.mode == NT_CHUCK_ARCHITECT_DISABLED ||
        architect->config.mode == NT_CHUCK_ARCHITECT_LEGACY) {
        if (decision) {
            int rc = nt_tape_chuck_observe(loss, &d.observation);
            if (rc != NT_CHUCK_OK) return rc;
        }
        int rc = nt_tape_chuck_step_action(lr, loss, NULL, NULL);
        if (rc == NT_CHUCK_OK && decision) *decision = d;
        return rc;
    }
    if (architect->pending || architect->decisions == UINT64_MAX)
        return NT_CHUCK_E_STATE;
    nt_chuck_architect copy = *architect;
    nt_chuck_observation o;
    int rc = nt_tape_chuck_observe(loss, &o);
    if (rc != NT_CHUCK_OK) return rc;
    rc = nt_ca_choose(&copy, &o, &d, copy.pending_hidden);
    if (rc != NT_CHUCK_OK) return rc;
    rc = nt_tape_chuck_step_action(lr, loss, &d.action, &copy.config.limits);
    if (rc != NT_CHUCK_OK) return rc;
    copy.pending_decision = d;
    copy.pending = 1;
    ++copy.decisions;
    *architect = copy;
    if (decision) *decision = d;
    return NT_CHUCK_OK;
}

int nt_chuck_architect_feedback(nt_chuck_architect *architect, float after_loss,
    nt_chuck_architect_receipt *receipt) {
    if (!nt_ca_state_valid(architect) || !architect->pending) return NT_CHUCK_E_STATE;
    nt_chuck_architect a = *architect;
    const nt_chuck_architect_decision *d = &a.pending_decision;
    nt_chuck_architect_receipt r;
    memset(&r, 0, sizeof(r));
    r.before_loss = d->observation.loss;
    r.after_loss = after_loss;
    r.loss_delta = (double)r.before_loss - (double)r.after_loss;
    r.nonfinite = !isfinite(after_loss);
    r.reward = r.nonfinite ? -1 : nt_ca_ratio(r.loss_delta, fabs((double)r.before_loss) + 1e-6);
    int head = d->action.kind - NT_CHUCK_ACTION_HOLD;
    r.predicted = d->scores[head];
    r.error = r.predicted - r.reward; // NT_CA_CREDIT_SIGN: measured consequence sets the target.
    r.action = d->action;
    r.decision = d->sequence;
    r.learned = 1;
    float g = nt_ca_clip(r.error, -1, 1); // Huber derivative, delta=1.
    float rate = a.config.learning_rate;
    float old_w2[NT_CHUCK_ARCHITECT_HIDDEN];
    memcpy(old_w2, a.w2[head], sizeof(old_w2));
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h)
        a.w2[head][h] = nt_ca_clip(a.w2[head][h] - rate * g * a.pending_hidden[h], -16, 16);
    a.b2[head] = nt_ca_clip(a.b2[head] - rate * g, -16, 16);
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h) {
        float gh = g * old_w2[h] * (1 - a.pending_hidden[h] * a.pending_hidden[h]);
        for (int i = 0; i < NT_CHUCK_ARCHITECT_FEATURES; ++i)
            a.w1[h][i] = nt_ca_clip(a.w1[h][i] - rate * gh * d->features[i], -16, 16);
        a.b1[h] = nt_ca_clip(a.b1[h] - rate * gh, -16, 16);
    }
    a.prev_loss = d->observation.loss;
    a.prev_trend = d->observation.loss_trend;
    a.prev_reward = r.reward;
    a.has_history = 1;
    a.pending = 0;
    ++a.updates;
    *architect = a;
    if (receipt) *receipt = r;
    return NT_CHUCK_OK;
}

// The same explicit field walk writes, reads and hashes the life. No struct
// padding, native byte order, pointer or compiler enum width enters the file.
typedef struct {
    unsigned char *bytes;
    size_t at, cap;
    int read, failed;
} nt_ca_codec;

static void nt_ca_bytes(nt_ca_codec *c, void *value, size_t n) {
    if (c->failed || n > c->cap - c->at) { c->failed = 1; return; }
    if (c->read) memcpy(value, c->bytes + c->at, n);
    else memcpy(c->bytes + c->at, value, n);
    c->at += n;
}

static void nt_ca_u32(nt_ca_codec *c, uint32_t *value) {
    unsigned char b[4];
    if (!c->read) for (int i = 0; i < 4; ++i) b[i] = (unsigned char)(*value >> (i * 8));
    nt_ca_bytes(c, b, 4);
    if (c->read && !c->failed)
        *value = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static void nt_ca_u64(nt_ca_codec *c, uint64_t *value) {
    uint32_t lo = 0, hi = 0;
    if (!c->read) { lo = (uint32_t)*value; hi = (uint32_t)(*value >> 32); }
    nt_ca_u32(c, &lo); nt_ca_u32(c, &hi);
    if (c->read && !c->failed) *value = (uint64_t)lo | (uint64_t)hi << 32;
}

static void nt_ca_int(nt_ca_codec *c, int *value) {
    uint32_t u = c->read ? 0 : (uint32_t)*value;
    nt_ca_u32(c, &u);
    if (c->read) {
        if (u > INT_MAX) c->failed = 1;
        else *value = (int)u;
    }
}

static void nt_ca_float(nt_ca_codec *c, float *value) {
    uint32_t u = 0;
    if (!c->read) memcpy(&u, value, 4);
    nt_ca_u32(c, &u);
    if (c->read && !c->failed) memcpy(value, &u, 4);
}

static void nt_ca_floats(nt_ca_codec *c, float *v, size_t n) {
    for (size_t i = 0; i < n; ++i) nt_ca_float(c, &v[i]);
}

static void nt_ca_observation_codec(nt_ca_codec *c, nt_chuck_observation *o) {
    nt_ca_float(c, &o->loss); nt_ca_float(c, &o->loss_ema); nt_ca_float(c, &o->loss_trend);
    nt_ca_float(c, &o->macro_ema); nt_ca_float(c, &o->best_macro);
    nt_ca_float(c, &o->dampen); nt_ca_float(c, &o->lr_scale); nt_ca_float(c, &o->noise);
    nt_ca_float(c, &o->grad_norm); nt_ca_float(c, &o->grad_trend); nt_ca_float(c, &o->frozen_fraction);
    nt_ca_int(c, &o->step); nt_ca_int(c, &o->stag); nt_ca_int(c, &o->macro_stag); nt_ca_int(c, &o->history_len);
}

static void nt_ca_state_codec(nt_ca_codec *c, nt_chuck_architect *a) {
    int mode = (int)a->config.mode;
    int kind = (int)a->pending_decision.action.kind;
    nt_ca_u32(c, &a->version);
    nt_ca_int(c, &mode);
    if (c->read) a->config.mode = (nt_chuck_architect_mode)mode;
    nt_ca_bytes(c, a->config.life_id, sizeof(a->config.life_id));
    nt_ca_u32(c, &a->config.limits.enabled_actions);
    nt_ca_float(c, &a->config.limits.dampen_min); nt_ca_float(c, &a->config.limits.dampen_max);
    nt_ca_float(c, &a->config.limits.lr_scale_min); nt_ca_float(c, &a->config.limits.lr_scale_max);
    nt_ca_float(c, &a->config.limits.noise_min); nt_ca_float(c, &a->config.limits.noise_max);
    nt_ca_float(c, &a->config.learning_rate); nt_ca_float(c, &a->config.exploration);
    nt_ca_u32(c, &a->config.seed); nt_ca_u32(c, &a->rng);
    nt_ca_u64(c, &a->decisions); nt_ca_u64(c, &a->updates);
    nt_ca_int(c, &a->pending); nt_ca_int(c, &a->has_history);
    for (int h = 0; h < NT_CHUCK_ARCHITECT_HIDDEN; ++h)
        nt_ca_floats(c, a->w1[h], NT_CHUCK_ARCHITECT_FEATURES);
    nt_ca_floats(c, a->b1, NT_CHUCK_ARCHITECT_HIDDEN);
    for (int k = 0; k < NT_CHUCK_ARCHITECT_HEADS; ++k)
        nt_ca_floats(c, a->w2[k], NT_CHUCK_ARCHITECT_HIDDEN);
    nt_ca_floats(c, a->b2, NT_CHUCK_ARCHITECT_HEADS);
    nt_ca_float(c, &a->prev_loss); nt_ca_float(c, &a->prev_trend); nt_ca_float(c, &a->prev_reward);
    nt_ca_observation_codec(c, &a->pending_decision.observation);
    nt_ca_floats(c, a->pending_decision.features, NT_CHUCK_ARCHITECT_FEATURES);
    nt_ca_floats(c, a->pending_decision.scores, NT_CHUCK_ARCHITECT_HEADS);
    nt_ca_int(c, &kind);
    if (c->read) a->pending_decision.action.kind = (nt_chuck_action_kind)kind;
    nt_ca_float(c, &a->pending_decision.action.value);
    nt_ca_u64(c, &a->pending_decision.sequence); nt_ca_int(c, &a->pending_decision.explored);
    nt_ca_floats(c, a->pending_hidden, NT_CHUCK_ARCHITECT_HIDDEN);
}

static uint64_t nt_ca_checksum(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static int nt_ca_encode(const nt_chuck_architect *a, unsigned char *payload) {
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128)
        return NT_CHUCK_E_FORMAT;
    if (!nt_ca_state_valid(a)) return NT_CHUCK_E_STATE;
    nt_chuck_architect copy = *a;
    nt_ca_codec c = {payload, 0, NT_CA_PAYLOAD_BYTES, 0, 0};
    nt_ca_state_codec(&c, &copy);
    return c.failed || c.at != NT_CA_PAYLOAD_BYTES ? NT_CHUCK_E_FORMAT : NT_CHUCK_OK;
}

uint64_t nt_chuck_architect_hash(const nt_chuck_architect *architect) {
    unsigned char payload[NT_CA_PAYLOAD_BYTES];
    if (nt_ca_encode(architect, payload) != NT_CHUCK_OK) return 0;
    return nt_ca_checksum(payload, sizeof(payload));
}

int nt_chuck_architect_save(const nt_chuck_architect *architect, const char *path) {
    unsigned char file[NT_CA_FILE_BYTES];
    if (!path || !*path) return NT_CHUCK_E_ARGUMENT;
    int rc = nt_ca_encode(architect, file + 24);
    if (rc != NT_CHUCK_OK) return rc;
    memcpy(file, "NTCALIFE", 8);
    uint32_t version = NT_CA_VERSION, length = NT_CA_PAYLOAD_BYTES;
    uint64_t checksum = nt_ca_checksum(file + 24, NT_CA_PAYLOAD_BYTES);
    nt_ca_codec c = {file, 8, 24, 0, 0};
    nt_ca_u32(&c, &version); nt_ca_u32(&c, &length); nt_ca_u64(&c, &checksum);
    // A failed write leaves the preceding life intact. The unique temporary
    // file lives beside its destination so rename is an atomic replacement.
    size_t path_len = strlen(path);
    if (path_len > SIZE_MAX - 12) return NT_CHUCK_E_ARGUMENT;
    char *temporary = (char *)malloc(path_len + 12);
    if (!temporary) return NT_CHUCK_E_IO;
    memcpy(temporary, path, path_len);
    memcpy(temporary + path_len, ".tmp.XXXXXX", 12);
    int fd = mkstemp(temporary);
    if (fd < 0) { free(temporary); return NT_CHUCK_E_IO; }
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(temporary); free(temporary); return NT_CHUCK_E_IO; }
    int failed = fwrite(file, 1, sizeof(file), f) != sizeof(file);
    if (fflush(f) != 0) failed = 1;
    if (!failed && fsync(fd) != 0) failed = 1;
    if (fclose(f) != 0) failed = 1;
    if (!failed && rename(temporary, path) != 0) failed = 1;
    if (failed) unlink(temporary);
    free(temporary);
    return failed ? NT_CHUCK_E_IO : NT_CHUCK_OK;
}

int nt_chuck_architect_load(nt_chuck_architect *architect, const char *path) {
    unsigned char file[NT_CA_FILE_BYTES];
    nt_chuck_architect a;
    if (!architect || !path || !*path) return NT_CHUCK_E_ARGUMENT;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128)
        return NT_CHUCK_E_FORMAT;
    FILE *f = fopen(path, "rb");
    if (!f) return NT_CHUCK_E_IO;
    size_t size = fread(file, 1, sizeof(file), f);
    int extra = fgetc(f), io_error = ferror(f);
    if (fclose(f) != 0) io_error = 1;
    if (io_error) return NT_CHUCK_E_IO;
    if (size != sizeof(file) || extra != EOF || memcmp(file, "NTCALIFE", 8))
        return NT_CHUCK_E_FORMAT;
    uint32_t version = 0, length = 0;
    uint64_t checksum = 0;
    nt_ca_codec head = {file, 8, 24, 1, 0};
    nt_ca_u32(&head, &version); nt_ca_u32(&head, &length); nt_ca_u64(&head, &checksum);
    if (head.failed || version != NT_CA_VERSION || length != NT_CA_PAYLOAD_BYTES ||
        checksum != nt_ca_checksum(file + 24, NT_CA_PAYLOAD_BYTES)) return NT_CHUCK_E_FORMAT;
    memset(&a, 0, sizeof(a));
    nt_ca_codec payload = {file + 24, 0, NT_CA_PAYLOAD_BYTES, 1, 0};
    nt_ca_state_codec(&payload, &a);
    if (payload.failed || payload.at != NT_CA_PAYLOAD_BYTES || !nt_ca_state_valid(&a))
        return NT_CHUCK_E_FORMAT;
    *architect = a;
    return NT_CHUCK_OK;
}

#endif
