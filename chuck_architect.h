// Chuck: Loss Architect — a small action language, acquired training experience.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef NOTORCH_CHUCK_ARCHITECT_H
#define NOTORCH_CHUCK_ARCHITECT_H

#include "notorch.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NT_CHUCK_ARCHITECT_FEATURES 16
#define NT_CHUCK_ARCHITECT_HIDDEN 8
#define NT_CHUCK_ARCHITECT_HEADS 3
#define NT_CHUCK_ARCHITECT_PARAMETERS 163
#define NT_CHUCK_ARCHITECT_LIFE_ID 64

typedef enum {
    NT_CHUCK_ARCHITECT_DISABLED = 0,
    NT_CHUCK_ARCHITECT_LEGACY = 1,
    NT_CHUCK_ARCHITECT_LEARNED = 2
} nt_chuck_architect_mode;

enum {
    NT_CHUCK_E_CONFIG = -10,
    NT_CHUCK_E_IO = -11,
    NT_CHUCK_E_FORMAT = -12,
    NT_CHUCK_E_BASELINE = -13
};

typedef struct {
    nt_chuck_architect_mode mode;
    char life_id[NT_CHUCK_ARCHITECT_LIFE_ID];
    nt_chuck_action_limits limits;
    float learning_rate, exploration;
    uint32_t seed;
} nt_chuck_architect_config;

typedef struct {
    nt_chuck_observation observation;
    float features[NT_CHUCK_ARCHITECT_FEATURES];
    float scores[NT_CHUCK_ARCHITECT_HEADS]; // predicted relative improvement
    nt_chuck_action action;
    uint64_t sequence;
    int explored;
} nt_chuck_architect_decision;

typedef struct {
    float before_loss, after_loss;
    double loss_delta; // before - after, in the SAME evaluation window
    float reward, predicted, error;
    nt_chuck_action action;
    uint64_t decision;
    int learned, nonfinite;
} nt_chuck_architect_receipt;

typedef struct {
    uint32_t version;
    nt_chuck_architect_config config;
    uint32_t rng;
    uint64_t decisions, updates;
    int pending, has_history;
    float w1[NT_CHUCK_ARCHITECT_HIDDEN][NT_CHUCK_ARCHITECT_FEATURES];
    float b1[NT_CHUCK_ARCHITECT_HIDDEN];
    float w2[NT_CHUCK_ARCHITECT_HEADS][NT_CHUCK_ARCHITECT_HIDDEN];
    float b2[NT_CHUCK_ARCHITECT_HEADS];
    float prev_loss, prev_trend, prev_reward;
    nt_chuck_architect_decision pending_decision;
    float pending_hidden[NT_CHUCK_ARCHITECT_HIDDEN];
} nt_chuck_architect;

// A measured common-state comparison; array order is HOLD, BRAKE, PUSH.
// The caller retains the executed branch receipts, horizon/window identity,
// and the protocol which binds these outcomes to the captured features.
typedef struct {
    float features[NT_CHUCK_ARCHITECT_FEATURES];
    float future_loss[NT_CHUCK_ARCHITECT_HEADS];
} nt_chuck_architect_comparison;

typedef struct {
    float future_loss[NT_CHUCK_ARCHITECT_HEADS];
    double loss_delta[NT_CHUCK_ARCHITECT_HEADS]; // HOLD loss minus action loss
    float target[NT_CHUCK_ARCHITECT_HEADS];
    float predicted_before[NT_CHUCK_ARCHITECT_HEADS];
    float predicted_after[NT_CHUCK_ARCHITECT_HEADS];
    float error_before[NT_CHUCK_ARCHITECT_HEADS];
    float huber_before, huber_after; // mean over the three measured actions
    int nonfinite[NT_CHUCK_ARCHITECT_HEADS];
    int fitted;
    uint64_t hash_before, hash_after;
    uint64_t decisions, updates; // unchanged online chronology
} nt_chuck_architect_comparison_receipt;

// Defaults: legacy, life_id="chuck", seed=1, learning_rate=.03, exploration=.15.
// Strict flat JSON fields: mode ("disabled", "legacy", "learned"), life_id
// ([A-Za-z0-9_.-], 1..63 bytes), enabled_actions (array of action names),
// dampen_min/max, lr_scale_min/max, noise_min/max, learning_rate, exploration,
// seed (nonzero uint32), version (1). Unknown/duplicate fields/actions, bad
// bounds and non-finite numbers are refused. Learned mode requires HOLD.
// Legacy/disabled modes require LEGACY and the default numerical envelope;
// contradictory masks or custom bounds are refused before canonical execution.
// JSON numbers use a private C numeric locale; caller locales remain unchanged.
// NULL, whitespace, and {} select defaults. Refusal leaves *out unchanged.
void nt_chuck_architect_config_default(nt_chuck_architect_config *out);
int nt_chuck_architect_config_parse_json(nt_chuck_architect_config *out,
    const char *json, char *error, size_t error_size);
int nt_chuck_architect_init(nt_chuck_architect *out,
    const nt_chuck_architect_config *config);

// Read-only preview, including a copy of the exploration RNG. It does not
// execute an action or create a transition eligible for learning.
int nt_chuck_architect_select(const nt_chuck_architect *architect,
    const nt_chuck_observation *observation, nt_chuck_action *action);

// Observe -> select -> execute -> pending experience. A refused core action
// leaves the policy and decision output unchanged. Only a successful learned
// step creates pending credit; finish it before taking another learned step.
// NULL architect and valid disabled/legacy lives call canonical Chuck. A
// non-NULL life is validated before dispatch, including its current config.
int nt_chuck_architect_step(nt_chuck_architect *architect, float lr, float loss,
    nt_chuck_architect_decision *decision);

// Caller evaluates the SAME batch/window immediately after the successful
// step. Reward is clamp((before-after)/(abs(before)+1e-6), -1, 1); non-finite
// after_loss receives -1 and sets receipt.nonfinite. Selected head + hidden
// layer learn the Huber regression target (delta=1), with bounded weights.
// Loss delta, reward, prediction and regression error are separate receipts.
int nt_chuck_architect_feedback(nt_chuck_architect *architect, float after_loss,
    nt_chuck_architect_receipt *receipt);

// Read-only replay interfaces. Capture uses the observation and this life's
// actual history; scores reads the supplied finite features in [-1,1]. Both
// require no pending action. Scores additionally requires learned mode and
// returns all three heads in HOLD/BRAKE/PUSH order without exploration or RNG.
// Refusal leaves output buffers unchanged.
int nt_chuck_architect_capture(const nt_chuck_architect *architect,
    const nt_chuck_observation *observation,
    float features[NT_CHUCK_ARCHITECT_FEATURES]);
int nt_chuck_architect_scores(const nt_chuck_architect *architect,
    const float features[NT_CHUCK_ARCHITECT_FEATURES],
    float scores[NT_CHUCK_ARCHITECT_HEADS]);

// Replay fitting requires learned mode, all three actions enabled, no pending
// action, and finite captured features in [-1,1]. A non-finite HOLD baseline
// returns NT_CHUCK_E_BASELINE. Finite targets are
// clamp((HOLD_loss-action_loss)/(abs(HOLD_loss)+1e-6),-1,1); non-finite
// alternatives receive -1 with their separate raw outcome/status in receipt.
// One mean-Huber update (delta=1) uses a single pre-update weight snapshot for
// all heads and hidden gradients, then applies the existing [-16,16] bounds.
// Only weights change. RNG, configuration, online counters/history and cached
// decisions remain intact. The caller records replay fit_step and binds life_id
// to its fixed external protocol. Existing feedback and life format are intact.
// Any refusal leaves both the life and optional receipt unchanged.
int nt_chuck_architect_fit_comparison(nt_chuck_architect *architect,
    const nt_chuck_architect_comparison *sample,
    nt_chuck_architect_comparison_receipt *receipt);

// Versioned, little-endian IEEE-754 policy life with canonical field encoding,
// FNV-1a checksum, strict size/range validation and transactional load. Includes
// configuration, weights, RNG, counters, temporal features and pending credit.
// Pending caches must agree with their observation/history/network within
// 1e-6 + 1e-5*abs(expected); recorded values are preserved during validation.
// The training body separately checkpoints its parameters, optimizer and RNG.
int nt_chuck_architect_save(const nt_chuck_architect *architect, const char *path);
int nt_chuck_architect_load(nt_chuck_architect *architect, const char *path);
uint64_t nt_chuck_architect_hash(const nt_chuck_architect *architect);

#ifdef __cplusplus
}
#endif
#endif
