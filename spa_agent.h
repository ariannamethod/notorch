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
#define NT_SPA_EXPERIENCE_VERSION 1u
#define NT_SPA_COMPARISON_MAX_HORIZON 4096u
#define NT_SPA_COMPARISON_MAX_REPEATS 64u

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
    NT_SPA_E_MEMORY = -29,
    NT_SPA_E_EXPERIENCE = -30,
    NT_SPA_E_COMPARISON = -31
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

// A frozen policy input, captured before choosing an action. Raw scenario
// records may construct this value directly using their original 29 features.
// The host records source_life_hash as provenance and retains the source-row
// association; this module checks compact feature geometry and action bounds.
typedef struct {
    uint32_t version, sentence_index, sentence_count;
    uint64_t source_life_hash;
    float features[NT_SPA_AGENT_FEATURES];
} nt_spa_experience;

typedef struct {
    nt_spa_action action;
    nt_spa_consequence consequence;
} nt_spa_alternative;

typedef struct {
    uint64_t source_life_hash; // Identifies the associated FEATURE source.
    uint32_t horizon; // Host-named continuation horizon, 0..4096 inclusive.
    uint32_t action_mask; // Exactly all valid actions at experience coordinates.
    nt_spa_alternative alternatives[NT_SPA_AGENT_ACTIONS]; // Indexed by kind.
} nt_spa_comparison;

typedef struct {
    nt_spa_action action; // Greedy valid argmax, deterministic KEEP-first ties.
    uint32_t action_mask;
    float scores[NT_SPA_AGENT_ACTIONS]; // All raw heads; mask governs selection.
} nt_spa_readout;

typedef struct {
    uint64_t source_life_hash;
    uint32_t horizon, action_mask;
    float learning_rate;
    float rewards[NT_SPA_AGENT_ACTIONS], targets[NT_SPA_AGENT_ACTIONS];
    float scores_before[NT_SPA_AGENT_ACTIONS], scores_after[NT_SPA_AGENT_ACTIONS];
    double loss_before, loss_after; // Mean Huber loss over valid actions, delta1.
} nt_spa_comparison_receipt;

// Conditioned replay retains the measured mean rewards beside its explicit
// training scale. The nested targets/losses use the conditioned objective.
// This transient receipt adds no field to the saved Agent life.
typedef struct {
    nt_spa_comparison_receipt comparison;
    float scale_floor;
    double scale;
} nt_spa_conditioned_receipt;

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

// Comparison learning reuses the 267 policy parameters. Capture and readout
// are pure; replay changes only policy bytes. Config, RNG, online counters,
// temporal memory, checkpoint schema and ordinary choose/observe stay intact.
// Agent-facing calls require a validated idle LEARNED life. Readout uses the captured
// features directly and consumes no exploration RNG. Output buffers must not
// overlap their agent or inputs; replay inputs must not overlap the agent.
int nt_spa_experience_validate(const nt_spa_experience *experience);
int nt_spa_agent_capture_experience(const nt_spa_agent *agent,
    const nt_spa_observation *observation, nt_spa_experience *experience);
int nt_spa_agent_score_experience(const nt_spa_agent *agent,
    const nt_spa_experience *experience, nt_spa_readout *readout);
// Every available action occurs exactly once, at its indexed slot, with the
// same before metrics. Inactive slots are zero. Rewards use the life's frozen
// reward coefficients; target[a]=reward[a]-reward[KEEP]. All valid heads fit
// their mean Huber loss in one simultaneous step from the same old weights.
// Rate is explicit [0,1]; zero evaluates without changing any policy byte.
// Receipt is optional. All refusals leave state, inputs and outputs unchanged.
int nt_spa_agent_fit_comparison(nt_spa_agent *agent,
    const nt_spa_experience *experience, const nt_spa_comparison *comparison,
    float learning_rate, nt_spa_comparison_receipt *receipt);
// Repeated same-state comparisons: count1..64, identical feature source,
// action mask, horizon and before metrics across every repetition. Compute
// each native clipped reward first, accumulate in double, then round its mean
// to float once. Targets are mean_reward[a]-mean_reward[KEEP]. One simultaneous
// mean-Huber policy update uses those targets; receipt rewards are the means.
// count1 is byte-identical to fit_comparison. No persistent field is added.
// All source comparisons stay unchanged. Overlapping inputs/output/life and
// malformed members are refused before any update, including later members.
int nt_spa_agent_fit_repeated(nt_spa_agent *agent,
    const nt_spa_experience *experience, const nt_spa_comparison *comparisons,
    uint32_t count, float learning_rate, nt_spa_comparison_receipt *receipt);
// Condition the same repeated mean-reward objective by its action-effect span.
// Raw delta[a] is the existing float subtraction mean[a]-mean[KEEP]. Scale is
// max((double)scale_floor, max_valid abs((double)delta[a])); normalized targets
// are (float)((double)delta[a]/scale). The floor must be finite and positive.
// Clipping and averaging precede scaling, including count1. One simultaneous
// mean-Huber update changes policy only; rate0 preserves every Agent byte.
// The nested receipt retains raw mean rewards and normalized targets. The
// caller's protocol identifies this objective; v1 weights do not encode it.
// All repeated-input and full receipt overlap/refusal contracts apply.
int nt_spa_agent_fit_conditioned(nt_spa_agent *agent,
    const nt_spa_experience *experience, const nt_spa_comparison *comparisons,
    uint32_t count, float learning_rate, float scale_floor,
    nt_spa_conditioned_receipt *receipt);

// Versioned canonical little-endian IEEE binary32 with checksum, exact length,
// strict structural/range/cache checks, transactional load and atomic save
// through a unique same-directory temporary file, file fsync, rename and parent
// directory fsync. Pre-rename failures preserve the old checkpoint. A directory
// sync/close failure after rename returns E_IO with the new file already installed.
// Includes pending
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
