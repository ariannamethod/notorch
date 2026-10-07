// Native oracle for Python/C sensory, action, credit and checkpoint parity.
#include "notorch.h"
#include "spa_agent.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(call) do { int status = (call); if (status != NT_SPA_OK) { \
    fprintf(stderr, "%s: %d\n", #call, status); return 1; } } while (0)

static const float field[12] = {1,.2f,0,-.1f, -.4f,.8f,.3f,0, .6f,.1f,-.5f,.9f};

static nt_spa_consequence outcome(unsigned kind) {
    const float local[3] = {.3f,.8f,.2f};
    const float novelty[3] = {.5f,.6f,.4f};
    nt_spa_consequence c;
    memset(&c, 0, sizeof(c));
    c.before = (nt_spa_metrics){.5f,.5f,.5f,.5f,.5f,.5f,.5f};
    c.after = c.before;
    c.after.local_connectedness = local[kind];
    c.after.novelty = novelty[kind];
    c.regeneration_cost = kind ? .25f : 0;
    return c;
}

static void hex(const char *name, const void *data, size_t len) {
    const unsigned char *p = data;
    size_t i;
    printf("%s ", name);
    for (i = 0; i < len; ++i) printf("%02x", p[i]);
    putchar('\n');
}

static int save(const nt_spa_agent *agent, const char *directory, const char *name) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (n < 0 || (size_t)n >= sizeof(path)) return NT_SPA_E_IO;
    return nt_spa_agent_save(agent, path);
}

int main(int argc, char **argv) {
    nt_spa_agent_config config;
    nt_spa_agent agent;
    nt_spa_observation observation;
    nt_spa_experience experience;
    nt_spa_comparison comparison;
    nt_spa_comparison_receipt fit;
    nt_spa_readout readout;
    int i, ids[3] = {2,0,1};
    float embedding[4], logits[3] = {1,-2,.5f}, connectedness;
    if (argc != 2) return 2;
    nt_spa_embed_sentence(ids, 3, field, 3, 4, .85f, embedding);
    hex("embedding", embedding, sizeof(embedding));
    connectedness = nt_spa_connectedness(embedding, 4, field, 3);
    hex("connectedness", &connectedness, sizeof(connectedness));
    nt_spa_modulate_logits(logits, 3, connectedness, .3f);
    hex("logits", logits, sizeof(logits));
    nt_spa_agent_config_default(&config);
    config.mode = NT_SPA_AGENT_LEARNED;
    config.seed = 123;
    config.exploration = .35f;
    OK(nt_spa_agent_init(&agent, &config));
    OK(nt_spa_agent_perceive(field, 3, 4, 1, .2f, .8f, 0, &observation));
    hex("observation", &observation, sizeof(observation));
    for (i = 0; i < 8; ++i) {
        float loss;
        OK(nt_spa_agent_imitate(&agent, &observation, NT_SPA_RESEED_LEFT, &loss));
        hex("imitation_loss", &loss, sizeof(loss));
    }
    OK(nt_spa_agent_capture_experience(&agent, &observation, &experience));
    memset(&comparison, 0, sizeof(comparison));
    comparison.source_life_hash = experience.source_life_hash;
    comparison.horizon = 4;
    comparison.action_mask = 7;
    for (i = 0; i < 3; ++i) {
        comparison.alternatives[i].action = (nt_spa_action){(nt_spa_action_kind)i,1,
            i == 0 ? NT_SPA_AGENT_NO_SOURCE : i == 1 ? 0u : 2u};
        comparison.alternatives[i].consequence = outcome((unsigned)i);
    }
    for (i = 0; i < 16; ++i) {
        OK(nt_spa_agent_fit_comparison(&agent, &experience, &comparison, .025f, &fit));
        hex("fit", &fit, sizeof(fit));
    }
    OK(nt_spa_agent_score_experience(&agent, &experience, &readout));
    hex("readout", &readout, sizeof(readout));
    for (i = 0; i < 10; ++i) {
        nt_spa_decision decision;
        nt_spa_receipt receipt;
        nt_spa_consequence consequence;
        OK(nt_spa_agent_perceive(field, 3, 4, (uint32_t)(i%3), .2f, .8f, (uint32_t)i, &observation));
        OK(nt_spa_agent_choose(&agent, &observation, &decision));
        hex("decision", &decision, sizeof(decision));
        if (i == 4) OK(save(&agent, argv[1], "native-pending.life"));
        consequence = outcome((unsigned)decision.action.kind);
        OK(nt_spa_agent_observe(&agent, decision.sequence, &decision.action, &consequence, &receipt));
        hex("receipt", &receipt, sizeof(receipt));
        printf("hash %" PRIu64 "\n", nt_spa_agent_hash(&agent));
    }
    OK(save(&agent, argv[1], "native-final.life"));
    return 0;
}
