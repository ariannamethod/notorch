// SPA: Sentence Phonon Agent — sentence perception, typed action, acquired experience.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef NOTORCH_SPA_AGENT_H
#define NOTORCH_SPA_AGENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NT_SPA_AGENT_VERSION 1u
#define NT_SPA_AGENT_PERCEPTION_VERSION 1u
#define NT_SPA_AGENT_REWARD_VERSION 1u
#define NT_SPA_AGENT_EMBED 4
#define NT_SPA_AGENT_FEATURES 29
#define NT_SPA_AGENT_HIDDEN 8
#define NT_SPA_AGENT_ACTIONS 3
#define NT_SPA_AGENT_PARAMETERS 267
#define NT_SPA_AGENT_HISTORY 8
#define NT_SPA_AGENT_MAX_SENTENCES 4096u
#define NT_SPA_AGENT_MAX_DIM 4096u
#define NT_SPA_AGENT_NO_SOURCE UINT32_MAX

typedef enum {
    NT_SPA_AGENT_DISABLED = 0,
    NT_SPA_AGENT_LEGACY = 1,
    NT_SPA_AGENT_LEARNED = 2
} nt_spa_agent_mode;

typedef enum {
    NT_SPA_KEEP = 0,
    NT_SPA_RESEED_LEFT = 1,
    NT_SPA_RESEED_RIGHT = 2
} nt_spa_action_kind;

enum {
    NT_SPA_OK = 0,
    NT_SPA_E_CONFIG = -20,
    NT_SPA_E_STATE = -21,
    NT_SPA_E_OBSERVATION = -22,
    NT_SPA_E_ACTION = -23,
    NT_SPA_E_PENDING = -24,
    NT_SPA_E_CONSEQUENCE = -25,
    NT_SPA_E_SEQUENCE = -26,
    NT_SPA_E_IO = -27,
    NT_SPA_E_FORMAT = -28,
    NT_SPA_E_MEMORY = -29
};

typedef struct {
    nt_spa_action_kind kind;
    uint32_t target, source; // KEEP source is NT_SPA_AGENT_NO_SOURCE.
} nt_spa_action;

// Host measurements are distinct axes in [0,1]. Their operational definitions
// belong to the host's frozen protocol. The agent retains every component.
typedef struct {
    float local_connectedness, global_connectedness, coherence;
    float novelty, repetition, collapse, continuity;
} nt_spa_metrics;

typedef struct {
    nt_spa_metrics before, after;
    float regeneration_cost; // Host-normalized cost in [0,1].
} nt_spa_consequence;

typedef struct {
    nt_spa_agent_mode mode;
    uint32_t seed; // Nonzero private xorshift32 seed; never uses global rand().
    float learning_rate, imitation_rate, exploration, memory_decay;
    // Reward = clamp(sum(w_i * oriented delta_i) - cost_weight * cost,-1,1).
    // Positive deltas: local/global connectedness, coherence, novelty, continuity.
    // Negative deltas: repetition, collapse. All weights are nonnegative.
    nt_spa_metrics reward_weights;
    float cost_weight;
} nt_spa_agent_config;

typedef struct {
    float embedding[NT_SPA_AGENT_EMBED]; // Compact coordinates, each [-1,1].
    float connectedness, left_similarity, right_similarity; // Missing neighbours are masked.
    float coherence, novelty, repetition;
    float phase_lock; // Host phase-gate input; Q derives it from its persistent state.
    // Supplied legacy scores: nonnegative, <=1e6. For Q's registered rule the
    // host supplies its own scores and selects the earliest weakest sentence.
    // nt_spa_agent_perceive uses upstream SPA row connectedness as its scores.
    float sentence_score, mean_sentence_score;
    float temperature; // (0,16].
    uint32_t sentence_index, sentence_count, reseed_count;
} nt_spa_observation;

typedef struct {
    float w1[NT_SPA_AGENT_HIDDEN][NT_SPA_AGENT_FEATURES];
    float b1[NT_SPA_AGENT_HIDDEN];
    float w2[NT_SPA_AGENT_ACTIONS][NT_SPA_AGENT_HIDDEN];
    float b2[NT_SPA_AGENT_ACTIONS];
} nt_spa_policy;

typedef struct {
    nt_spa_observation observation;
    float features[NT_SPA_AGENT_FEATURES];
    float hidden[NT_SPA_AGENT_HIDDEN], scores[NT_SPA_AGENT_ACTIONS];
    nt_spa_action action;
    uint64_t sequence;
    uint32_t rng_before;
    int explored;
} nt_spa_decision;

typedef struct {
    uint64_t sequence;
    nt_spa_action action;
    uint32_t sentence_count;
    nt_spa_consequence consequence;
    float reward, predicted, error;
    int learned;
} nt_spa_receipt;

// Public value type: a complete in-memory snapshot may be copied by assignment.
// Config is frozen at init and checked against config_hash on every operation.
// Use set_policy for acquired-weight counterfactuals; it changes policy only.
typedef struct {
    uint32_t version, perception_version, reward_version;
    nt_spa_agent_config config;
    uint64_t config_hash;
    nt_spa_policy policy;
    uint32_t rng;
    uint64_t decisions, observations, updates, imitation_updates, cancelled;
    uint64_t memory_observations;
    uint32_t history_count, history_head;
    nt_spa_receipt history[NT_SPA_AGENT_HISTORY];
    float ema_reward, ema_connectedness, ema_novelty;
    int pending;
    nt_spa_decision pending_decision;
} nt_spa_agent;

// Defaults: LEGACY; seed1; learning_rate=.03; imitation_rate=.05;
// exploration=.1; memory_decay=.8; reward weights .15,.15,.2,.2,.15,.1,.05;
// cost_weight=.05. Every refused mutating call leaves state/output unchanged.
// Decision/receipt/loss output buffers must not overlap the agent; overlap is
// refused. init(&agent,&agent.config) and set_policy(&agent,&agent.policy) work.
void nt_spa_agent_config_default(nt_spa_agent_config *out);
int nt_spa_agent_init(nt_spa_agent *out, const nt_spa_agent_config *config);
int nt_spa_agent_validate(const nt_spa_agent *agent);
int nt_spa_observation_validate(const nt_spa_observation *observation);
int nt_spa_action_validate(const nt_spa_action *action,
                          const nt_spa_observation *observation);

// Optional native perception over contiguous [count][dim] sentence embeddings.
// Calls unchanged nt_spa_connectedness on all OTHER sentences. Scores are
// these row connectedness values, mean_sentence_score their arithmetic mean.
// Compact coordinates fold dimensions modulo4, average, then tanh. Similarity
// is cosine; missing-neighbour similarity is zero. Coherence/continuity is the
// mean adjacent cosine mapped to [0,1]; novelty=1-max positive other cosine;
// repetition=max positive other cosine. Host may supply its own observation.
int nt_spa_agent_perceive(const float *embeddings, uint32_t count, uint32_t dim,
    uint32_t target, float phase_lock, float temperature, uint32_t reseeds,
    nt_spa_observation *out);

// Registered legacy: score < mean*(.52+.18*(1-phase_lock)) means RESEED_LEFT
// when index>0, else RESEED_RIGHT when count>1. Strict comparison; otherwise
// KEEP. Q chooses the earliest weakest target before applying this rule.
int nt_spa_agent_legacy(const nt_spa_observation *observation,
                       nt_spa_action *action);

// select is a pure preview including copied exploration RNG. choose reserves
// exactly one decision; host validates/executes its typed action, then observe
// credits that sequence once. Credit requires the exact executed action witness;
// host owns the before/after metric window and its measurement definitions.
// Pending lives refuse another select/choose.
// NULL or valid DISABLED agent returns KEEP, sequence0 and does not mutate.
int nt_spa_agent_select(const nt_spa_agent *agent,
    const nt_spa_observation *observation, nt_spa_decision *decision);
int nt_spa_agent_choose(nt_spa_agent *agent,
    const nt_spa_observation *observation, nt_spa_decision *decision);
int nt_spa_agent_observe(nt_spa_agent *agent, uint64_t sequence,
    const nt_spa_action *executed_action,
    const nt_spa_consequence *consequence, nt_spa_receipt *receipt);
// Cancel a rejected/unexecuted host action. Counters record cancellation;
// no experience/history/weight update. Selection RNG remains consumed.
int nt_spa_agent_cancel(nt_spa_agent *agent, uint64_t sequence);

// Supervised softmax cross-entropy SGD, with action bounds and boundary masks.
// Fits the label through the policy; no legacy shortcut in learned selection.
// LEARNED idle life required. Returns pre-update loss when loss!=NULL.
int nt_spa_agent_imitate(nt_spa_agent *agent,
    const nt_spa_observation *observation, nt_spa_action_kind label, float *loss);
// Temporal reset preserves config, weights, RNG and lifetime counters.
// set_policy preserves all other state; both refuse a pending decision.
int nt_spa_agent_reset_memory(nt_spa_agent *agent);
int nt_spa_agent_set_policy(nt_spa_agent *agent, const nt_spa_policy *policy);

// Versioned canonical little-endian IEEE binary32 with checksum, exact length,
// strict structural/range/cache checks, transactional load and atomic save
// through a unique same-directory temporary file, fsync and rename. Includes pending
// decision, raw outcomes, frozen reward/perception versions, private RNG.
// The host separately saves model/generation state. Same-platform continuation
// is exact; no raw C structure dump. hash covers this canonical representation.
int nt_spa_agent_save(const nt_spa_agent *agent, const char *path);
int nt_spa_agent_load(nt_spa_agent *agent, const char *path);
uint64_t nt_spa_agent_hash(const nt_spa_agent *agent);

#ifdef __cplusplus
}
#endif
#endif
