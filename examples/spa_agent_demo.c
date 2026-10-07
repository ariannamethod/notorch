/* Sentence Phonon Agent: a public pretrained SimpleLLM body on upstream notorch.
 *
 * Architecture: ariannamethod/notorch-simple-llm, train_dracula.py, 80b3bd6.
 * Experiment: experiments/spa_agent/protocol.json (frozen before measurements).
 * Generation, sentence sensing and policy arithmetic link the upstream C engine.
 * Every output record carries the executed action and its separate consequences.
 */
#include "notorch.h"
#include "spa_agent.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 128
#define HEAD_DIM 32
#define HIDDEN 384
#define VOCAB 94
#define PARAMS 21
#define CONTEXT 64
#define SENTENCES 4
#define EPISODES 6
#define MAX_GENERATED 64
#define MIN_GENERATED 12
#define PROMPT_LENGTH 16
#define ARMS 5

typedef struct {
    nt_tensor **parameters;
    long count;
    uint32_t vocabulary[VOCAB];
    uint64_t forwards;
} sentence_body;

typedef struct {
    int ids[MAX_GENERATED];
    int length, stopped;
} sentence;

typedef struct {
    const char *name;
    nt_spa_agent life;
    nt_spa_policy initial;
    unsigned actions[NT_SPA_AGENT_ACTIONS];
    unsigned weight_changed_choices, verified_resumes;
    double reward;
} experiment_arm;

static void fail(const char *message) {
    fprintf(stderr, "spa_agent_demo: %s\n", message);
    exit(1);
}

static void check(int status, const char *message) {
    if (status != NT_SPA_OK) {
        fprintf(stderr, "spa_agent_demo: %s (%d)\n", message, status);
        exit(1);
    }
}

static uint64_t next_random(uint64_t *state) {
    uint64_t x = (*state += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t stream(uint32_t seed, uint64_t domain, unsigned episode, unsigned slot) {
    uint64_t state = domain ^ (uint64_t)seed ^ ((uint64_t)episode << 32) ^ ((uint64_t)slot << 48);
    return next_random(&state);
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) { hash ^= p[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

static uint64_t body_hash(const sentence_body *body) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < PARAMS; i++)
        hash = hash_bytes(hash, body->parameters[i]->data,
                          body->parameters[i]->len * sizeof(float));
    return hash;
}

static uint32_t *read_u32(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) fail("cannot open token/vocabulary input");
    if (fseek(f, 0, SEEK_END)) fail("cannot seek token input");
    long bytes = ftell(f);
    if (bytes <= 0 || bytes % 4) fail("invalid token input size");
    rewind(f);
    *count = (size_t)bytes / 4;
    uint32_t *out = malloc(*count * sizeof(*out));
    if (!out) fail("token input allocation failed");
    for (size_t i = 0; i < *count; i++) {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) fail("truncated token input");
        out[i] = (uint32_t)b[0] | (uint32_t)b[1] << 8 |
                 (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    }
    if (fclose(f)) fail("token input close failed");
    return out;
}

static void load_body(sentence_body *body, const char *weights, const char *vocabulary) {
    memset(body, 0, sizeof(*body));
    int count = 0;
    body->parameters = nt_load(weights, &count);
    if (!body->parameters || count != PARAMS) fail("expected 21 complete SimpleLLM parameters");
    const int shapes[PARAMS][2] = {
        {VOCAB,DIM}, {DIM,0}, {DIM,DIM}, {DIM,DIM}, {DIM,DIM}, {DIM,DIM},
        {DIM,0}, {HIDDEN,DIM}, {HIDDEN,DIM}, {DIM,HIDDEN},
        {DIM,0}, {DIM,DIM}, {DIM,DIM}, {DIM,DIM}, {DIM,DIM},
        {DIM,0}, {HIDDEN,DIM}, {HIDDEN,DIM}, {DIM,HIDDEN}, {DIM,0}, {VOCAB,DIM}
    };
    for (int i = 0; i < PARAMS; i++) {
        nt_tensor *p = body->parameters[i];
        if (p->ndim != (shapes[i][1] ? 2 : 1) || p->shape[0] != shapes[i][0] ||
            (shapes[i][1] && p->shape[1] != shapes[i][1])) fail("SimpleLLM checkpoint shape mismatch");
        for (int j = 0; j < p->len; j++) if (!isfinite(p->data[j])) fail("non-finite body weight");
        body->count += (long)p->len;
    }
    if (body->count != 450688) fail("SimpleLLM parameter count mismatch");
    size_t vocab_count = 0;
    uint32_t *codepoints = read_u32(vocabulary, &vocab_count);
    if (vocab_count != VOCAB) fail("SimpleLLM vocabulary count mismatch");
    memcpy(body->vocabulary, codepoints, sizeof(body->vocabulary));
    free(codepoints);
}

/* Exact SimpleLLM forward recipe; a rolling context resets RoPE positions 0..T-1,
 * as in train_dracula.py::generate. Parameter registration is inference-only. */
static void next_logits(sentence_body *body, const int *context, int length, float *out) {
    if (length < 1 || length > CONTEXT) fail("invalid generation context");
    nt_train_mode(0);
    nt_tape_start();
    int p[PARAMS];
    for (int i = 0; i < PARAMS; i++) {
        p[i] = nt_tape_param_frozen(body->parameters[i]);
        if (p[i] < 0) fail("cannot register frozen body parameter");
    }
    nt_tensor *tokens = nt_tensor_new((size_t)length);
    if (!tokens) fail("cannot allocate context");
    for (int i = 0; i < length; i++) {
        if (context[i] < 0 || context[i] >= VOCAB) fail("context token out of range");
        tokens->data[i] = (float)context[i];
    }
    int ids = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tokens);
    int h = nt_seq_embedding(p[0], -1, ids, length, DIM), pi = 1;
    for (int layer = 0; layer < 2; layer++) {
        int norm1 = p[pi++], wq = p[pi++], wk = p[pi++], wv = p[pi++], wo = p[pi++];
        int norm2 = p[pi++], gate_w = p[pi++], up_w = p[pi++], down_w = p[pi++];
        int x = nt_seq_rmsnorm(h, norm1, length, DIM);
        int q = nt_rope(nt_seq_linear(wq, x, length), length, HEAD_DIM);
        int k = nt_rope(nt_seq_linear(wk, x, length), length, HEAD_DIM);
        int v = nt_seq_linear(wv, x, length);
        int attention = nt_mh_causal_attention(q, k, v, length, HEAD_DIM);
        h = nt_add(h, nt_seq_linear(wo, attention, length));
        x = nt_seq_rmsnorm(h, norm2, length, DIM);
        int gate = nt_silu(nt_seq_linear(gate_w, x, length));
        int up = nt_seq_linear(up_w, x, length);
        h = nt_add(h, nt_seq_linear(down_w, nt_mul(gate, up), length));
    }
    int normalized = nt_seq_rmsnorm(h, p[pi], length, DIM);
    int logits = nt_seq_linear(p[pi + 1], normalized, length);
    if (logits < 0) fail("body forward refused");
    const nt_tensor *values = nt_tape_get()->entries[logits].output;
    if (!values || values->len != length * VOCAB) fail("invalid body output");
    memcpy(out, values->data + (length - 1) * VOCAB, VOCAB * sizeof(float));
    for (int i = 0; i < VOCAB; i++) if (!isfinite(out[i])) fail("non-finite body logit");
    nt_tape_clear();
    body->forwards++;
}

static void generate(sentence_body *body, const int *prompt, int prompt_length,
                     uint64_t *rng, sentence *out) {
    if (prompt_length < 1 || prompt_length > CONTEXT) fail("invalid generation prompt");
    memset(out, 0, sizeof(*out));
    int context[CONTEXT], length = prompt_length;
    memcpy(context, prompt, (size_t)length * sizeof(*context));
    for (int step = 0; step < MAX_GENERATED; step++) {
        float logits[VOCAB];
        next_logits(body, context, length, logits);
        double maximum = logits[0] / 0.8, sum = 0, probabilities[VOCAB];
        for (int i = 1; i < VOCAB; i++) if (logits[i] / 0.8 > maximum) maximum = logits[i] / 0.8;
        for (int i = 0; i < VOCAB; i++) { probabilities[i] = exp(logits[i] / 0.8 - maximum); sum += probabilities[i]; }
        double draw = (double)(next_random(rng) >> 11) * 0x1.0p-53 * sum;
        int selected = VOCAB - 1;
        for (int i = 0; i < VOCAB; i++) { draw -= probabilities[i]; if (draw < 0) { selected = i; break; } }
        out->ids[out->length++] = selected;
        if (length == CONTEXT) { memmove(context, context + 1, (CONTEXT - 1) * sizeof(*context)); length--; }
        context[length++] = selected;
        uint32_t codepoint = body->vocabulary[selected];
        if (out->length >= MIN_GENERATED && (codepoint == '.' || codepoint == '!' || codepoint == '?')) {
            out->stopped = 1;
            break;
        }
    }
}

static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static float cosine(const float *a, const float *b) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < DIM; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    if (na == 0 || nb == 0) return 0;
    double c = dot / sqrt(na * nb);
    return (float)(c < -1 ? -1 : c > 1 ? 1 : c);
}

static float repetition(const sentence *s) {
    if (s->length < 3) return 0;
    int repeated = 0;
    for (int i = 2; i < s->length; i++) {
        for (int j = 2; j < i; j++) {
            if (s->ids[i - 2] == s->ids[j - 2] && s->ids[i - 1] == s->ids[j - 1] && s->ids[i] == s->ids[j]) {
                repeated++; break;
            }
        }
    }
    return (float)repeated / (float)(s->length - 2);
}

static void embeddings(const sentence_body *body, const sentence chain[SENTENCES], float out[SENTENCES][DIM]) {
    for (int i = 0; i < SENTENCES; i++)
        nt_spa_embed_sentence(chain[i].ids, chain[i].length, body->parameters[0]->data,
                              VOCAB, DIM, 0.85f, out[i]);
}

static nt_spa_metrics measure(const sentence chain[SENTENCES], float embs[SENTENCES][DIM],
                             unsigned target, float global_connectedness) {
    nt_spa_metrics m = {0};
    float maximum = 0, minimum = 1, local = 0;
    int neighbors = 0, collapsed = 0;
    for (int i = 0; i < SENTENCES; i++) {
        if ((unsigned)i != target) {
            float c = cosine(embs[target], embs[i]);
            if (c > maximum) maximum = c;
        }
        if (i + 1 < SENTENCES) {
            float similarity = clamp01(0.5f * (cosine(embs[i], embs[i + 1]) + 1));
            m.coherence += similarity / (SENTENCES - 1);
            if (similarity < minimum) minimum = similarity;
            if ((unsigned)i == target || (unsigned)(i + 1) == target) { local += similarity; neighbors++; }
        }
        for (int j = i + 1; j < SENTENCES; j++) if (cosine(embs[i], embs[j]) >= 0.98f) collapsed++;
    }
    m.local_connectedness = local / neighbors;
    m.global_connectedness = global_connectedness;
    m.coherence = clamp01(m.coherence);
    m.novelty = clamp01(1 - maximum);
    m.repetition = repetition(&chain[target]);
    m.collapse = (float)collapsed / ((SENTENCES * (SENTENCES - 1)) / 2);
    m.continuity = minimum;
    return m;
}

static void write_sentence(FILE *file, const sentence *s) {
    fprintf(file, "{\"length\":%d,\"terminated\":%s,\"tokens\":[", s->length, s->stopped ? "true" : "false");
    for (int i = 0; i < s->length; i++) fprintf(file, "%s%d", i ? "," : "", s->ids[i]);
    fputs("]}", file);
}

static void write_chain(FILE *file, const sentence chain[SENTENCES]) {
    fputc('[', file);
    for (int i = 0; i < SENTENCES; i++) { if (i) fputc(',', file); write_sentence(file, &chain[i]); }
    fputc(']', file);
}

static void write_metrics(FILE *file, const nt_spa_metrics *m) {
    fprintf(file, "{\"local_connectedness\":%.9g,\"global_connectedness\":%.9g,\"coherence\":%.9g,"
                  "\"novelty\":%.9g,\"repetition\":%.9g,\"collapse\":%.9g,\"continuity\":%.9g}",
            m->local_connectedness, m->global_connectedness, m->coherence, m->novelty,
            m->repetition, m->collapse, m->continuity);
}

static void write_action(FILE *file, const nt_spa_action *a) {
    fprintf(file, "{\"kind\":%d,\"target\":%u,\"source\":", a->kind, a->target);
    if (a->source == NT_SPA_AGENT_NO_SOURCE) fputs("null", file); else fprintf(file, "%u", a->source);
    fputc('}', file);
}

static int same_action(const nt_spa_action *a, const nt_spa_action *b) {
    return a->kind == b->kind && a->target == b->target && a->source == b->source;
}

static void write_floats(FILE *file, const float *values, int count) {
    fputc('[', file);
    for (int i = 0; i < count; i++) fprintf(file, "%s%.9g", i ? "," : "", values[i]);
    fputc(']', file);
}

static void arm_init(experiment_arm *arm, int index, uint32_t seed) {
    static const char *names[ARMS] = {"disabled", "legacy", "random", "frozen", "learned"};
    memset(arm, 0, sizeof(*arm));
    arm->name = names[index];
    nt_spa_agent_config cfg;
    nt_spa_agent_config_default(&cfg);
    cfg.seed = seed;
    cfg.mode = index == 0 ? NT_SPA_AGENT_DISABLED : index == 1 ? NT_SPA_AGENT_LEGACY : NT_SPA_AGENT_LEARNED;
    cfg.learning_rate = index == 4 ? 0.03f : 0;
    cfg.exploration = index == 2 ? 1 : index >= 3 ? 0.2f : 0;
    cfg.memory_decay = 0.8f;
    cfg.reward_weights = (nt_spa_metrics){0.15f,0.15f,0.20f,0.20f,0.15f,0.10f,0.05f};
    cfg.cost_weight = 0.05f;
    check(nt_spa_agent_init(&arm->life, &cfg), "arm initialization failed");
    arm->initial = arm->life.policy;
}

static int whitespace(uint32_t c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

static size_t *sentence_starts(const uint32_t *tokens, size_t count, const uint32_t *vocab, size_t *n_starts) {
    size_t *starts = malloc(count * sizeof(*starts));
    if (!starts) fail("cannot allocate prompt starts");
    *n_starts = 0;
    starts[(*n_starts)++] = 0;
    for (size_t i = 1; i + PROMPT_LENGTH < count; i++) {
        uint32_t prev = vocab[tokens[i - 1]];
        if ((prev == '.' || prev == '!' || prev == '?') && whitespace(vocab[tokens[i]])) {
            size_t next = i;
            while (next < count && whitespace(vocab[tokens[next]])) next++;
            if (next + PROMPT_LENGTH < count) starts[(*n_starts)++] = next;
        }
    }
    if (*n_starts < 100) fail("corpus has too few sentence starts");
    return starts;
}

static void observation(const sentence_body *body, const sentence chain[SENTENCES],
                        unsigned target, unsigned reseeds, nt_spa_observation *out,
                        nt_spa_metrics *metrics) {
    float embs[SENTENCES][DIM];
    embeddings(body, chain, embs);
    check(nt_spa_agent_perceive(&embs[0][0], SENTENCES, DIM, target, 0.5f, 0.8f, reseeds, out), "sentence perception failed");
    out->repetition = repetition(&chain[target]);
    // Reuse the validated upstream SPA result. A second unobserved allocation
    // failure in the legacy helper must never become a zero-valued consequence.
    if (!isfinite(out->connectedness) || out->connectedness <= 0) fail("invalid sentence connectedness");
    *metrics = measure(chain, embs, target, out->connectedness);
}

static void save_and_resume(experiment_arm *arm, const char *prefix, const nt_spa_observation *obs) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s.%s.life.bin", prefix, arm->name);
    if (n < 0 || (size_t)n >= sizeof(path)) fail("output path too long");
    uint64_t before = nt_spa_agent_hash(&arm->life);
    if (!before) fail("invalid canonical life hash");
    check(nt_spa_agent_save(&arm->life, path), "life save failed");
    nt_spa_agent loaded;
    check(nt_spa_agent_load(&loaded, path), "life load failed");
    if (nt_spa_agent_hash(&loaded) != before) fail("save/resume changed canonical life hash");
    nt_spa_decision a, b;
    check(nt_spa_agent_select(&arm->life, obs, &a), "pre-save continuation preview failed");
    check(nt_spa_agent_select(&loaded, obs, &b), "loaded continuation preview failed");
    if (!same_action(&a.action, &b.action) || a.rng_before != b.rng_before ||
        memcmp(a.features, b.features, sizeof(a.features)) || memcmp(a.scores, b.scores, sizeof(a.scores)))
        fail("save/resume changed continuation");
    arm->life = loaded;
    arm->verified_resumes++;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s CHECKPOINT TOKENS_U32 VOCAB_U32 OUTPUT_PREFIX SEED\n", argv[0]);
        return 2;
    }
    errno = 0; char *end = NULL;
    unsigned long parsed = strtoul(argv[5], &end, 10);
    if (errno || !argv[5][0] || *end || parsed == 0 || parsed > UINT32_MAX) fail("invalid seed");
    uint32_t seed = (uint32_t)parsed;
    sentence_body body;
    load_body(&body, argv[1], argv[3]);
    uint64_t original_weights = body_hash(&body);
    size_t token_count = 0, n_starts = 0;
    uint32_t *tokens = read_u32(argv[2], &token_count);
    if (token_count < 1000) fail("corpus too short");
    for (size_t i = 0; i < token_count; i++) if (tokens[i] >= VOCAB) fail("corpus token outside vocabulary");
    size_t *starts = sentence_starts(tokens, token_count, body.vocabulary, &n_starts);
    experiment_arm arms[ARMS];
    for (int i = 0; i < ARMS; i++) arm_init(&arms[i], i, seed);
    char path[4096];
    int pn = snprintf(path, sizeof(path), "%s.jsonl", argv[4]);
    if (pn < 0 || (size_t)pn >= sizeof(path)) fail("trace output path too long");
    FILE *trace = fopen(path, "w");
    if (!trace) fail("cannot open trace");
    fprintf(trace, "{\"type\":\"body\",\"seed\":%u,\"parameters\":%ld,\"weights_fnv1a\":\"%016" PRIx64 "\"}\n", seed, body.count, original_weights);
    for (unsigned episode = 0; episode < EPISODES; episode++) {
        sentence base[SENTENCES];
        size_t offsets[SENTENCES];
        for (unsigned slot = 0; slot < SENTENCES; slot++) {
            uint64_t rng = stream(seed, UINT64_C(0x7370615f62617365), episode, slot);
            offsets[slot] = starts[next_random(&rng) % n_starts];
            int prompt[PROMPT_LENGTH];
            for (int i = 0; i < PROMPT_LENGTH; i++) prompt[i] = (int)tokens[offsets[slot] + (size_t)i];
            generate(&body, prompt, PROMPT_LENGTH, &rng, &base[slot]);
        }
        fprintf(trace, "{\"type\":\"base\",\"seed\":%u,\"episode\":%u,\"prompt_offsets\":[", seed, episode);
        for (int i = 0; i < SENTENCES; i++) fprintf(trace, "%s%zu", i ? "," : "", offsets[i]);
        fputs("],\"chain\":", trace); write_chain(trace, base); fputs("}\n", trace);
        for (int ai = 0; ai < ARMS; ai++) {
            experiment_arm *arm = &arms[ai];
            sentence chain[SENTENCES]; memcpy(chain, base, sizeof(chain));
            unsigned reseeds[SENTENCES] = {0};
            for (unsigned step = 0; step < SENTENCES; step++) {
                unsigned target = (episode + step) % SENTENCES;
                nt_spa_observation obs;
                nt_spa_metrics before, after;
                observation(&body, chain, target, reseeds[target], &obs, &before);
                nt_spa_decision decision, counterfactual;
                int has_counterfactual = ai == 4;
                if (has_counterfactual) {
                    nt_spa_agent initial = arm->life;
                    check(nt_spa_agent_set_policy(&initial, &arm->initial), "initial-weight counterfactual failed");
                    check(nt_spa_agent_select(&initial, &obs, &counterfactual), "counterfactual selection failed");
                }
                uint64_t life_before = nt_spa_agent_hash(&arm->life);
                check(nt_spa_agent_choose(&arm->life, &obs, &decision), "agent choice failed");
                check(nt_spa_action_validate(&decision.action, &obs), "host action validation failed");
                if (has_counterfactual && (decision.rng_before != counterfactual.rng_before ||
                    memcmp(decision.features, counterfactual.features, sizeof(decision.features))))
                    fail("weight-only counterfactual changed observation/history/RNG input");
                int changed = has_counterfactual && !same_action(&decision.action, &counterfactual.action);
                if (changed) arm->weight_changed_choices++;
                arm->actions[decision.action.kind]++;
                uint64_t host_rng = stream(seed, UINT64_C(0x7370615f6163746e), episode, step), host_rng_before = host_rng;
                float cost = 0;
                if (decision.action.kind != NT_SPA_KEEP) {
                    const sentence *source = &chain[decision.action.source];
                    int length = source->length > 3 ? 3 : source->length;
                    if (!length) {
                        check(nt_spa_agent_cancel(&arm->life, decision.sequence), "cannot cancel empty-neighbor action");
                        fail("host refused empty neighbor");
                    }
                    sentence candidate;
                    generate(&body, source->ids + source->length - length, length, &host_rng, &candidate);
                    chain[target] = candidate;
                    reseeds[target]++;
                    cost = (float)candidate.length / MAX_GENERATED;
                } else if (host_rng != host_rng_before) fail("KEEP consumed host RNG");
                nt_spa_observation next_obs;
                observation(&body, chain, target, reseeds[target], &next_obs, &after);
                nt_spa_consequence consequence = {before, after, cost};
                nt_spa_receipt receipt = {0};
                if (ai == 0) {
                    if (decision.sequence || decision.action.kind != NT_SPA_KEEP ||
                        memcmp(chain, base, sizeof(chain)) || host_rng != host_rng_before ||
                        nt_spa_agent_hash(&arm->life) != life_before) fail("disabled mode changed host/life");
                } else {
                    check(nt_spa_agent_observe(&arm->life, decision.sequence, &decision.action,
                                             &consequence, &receipt), "consequence credit failed");
                    arm->reward += receipt.reward;
                }
                fprintf(trace, "{\"type\":\"decision\",\"seed\":%u,\"episode\":%u,\"step\":%u,\"arm\":\"%s\",\"action\":",
                        seed, episode, step, arm->name);
                write_action(trace, &decision.action);
                fprintf(trace, ",\"sequence\":%" PRIu64 ",\"explored\":%s,\"policy_rng_before\":%u,\"host_rng_before\":\"%016" PRIx64
                               "\",\"host_rng_after\":\"%016" PRIx64 "\",\"scores\":", decision.sequence,
                        decision.explored ? "true" : "false", decision.rng_before, host_rng_before, host_rng);
                write_floats(trace, decision.scores, NT_SPA_AGENT_ACTIONS);
                fputs(",\"features\":", trace); write_floats(trace, decision.features, NT_SPA_AGENT_FEATURES);
                fputs(",\"before\":", trace); write_metrics(trace, &before);
                fputs(",\"after\":", trace); write_metrics(trace, &after);
                fprintf(trace, ",\"cost\":%.9g,\"reward\":%.9g,\"learned\":%s,\"weights_changed_choice\":%s",
                        cost, receipt.reward, receipt.learned ? "true" : "false", changed ? "true" : "false");
                if (has_counterfactual) {
                    fputs(",\"initial_weights_action\":", trace); write_action(trace, &counterfactual.action);
                    fputs(",\"initial_weights_scores\":", trace); write_floats(trace, counterfactual.scores, NT_SPA_AGENT_ACTIONS);
                    fputs(",\"counterfactual_inputs_equal\":true", trace);
                }
                fputs(",\"chain\":", trace); write_chain(trace, chain); fputs("}\n", trace);
                if (ferror(trace)) fail("trace write failed");
            }
            nt_spa_observation resume_obs;
            nt_spa_metrics ignored;
            observation(&body, chain, 0, reseeds[0], &resume_obs, &ignored);
            save_and_resume(arm, argv[4], &resume_obs);
        }
        if (fflush(trace)) fail("trace flush failed");
        fprintf(stderr, "seed=%u episode=%u/%d forwards=%" PRIu64 "\n", seed, episode + 1, EPISODES, body.forwards);
    }
    if (body_hash(&body) != original_weights) fail("SPA altered body parameters");
    fprintf(trace, "{\"type\":\"summary\",\"seed\":%u,\"forwards\":%" PRIu64 ",\"body_unchanged\":true,\"arms\":[", seed, body.forwards);
    for (int i = 0; i < ARMS; i++) {
        experiment_arm *a = &arms[i];
        fprintf(trace, "%s{\"name\":\"%s\",\"actions\":[%u,%u,%u],\"reward_sum\":%.17g,"
                      "\"weights_changed_choices\":%u,\"verified_resumes\":%u,\"decisions\":%" PRIu64 ",\"observations\":%" PRIu64
                      ",\"updates\":%" PRIu64 ",\"life_hash\":\"%016" PRIx64 "\"}", i ? "," : "", a->name,
                a->actions[0], a->actions[1], a->actions[2], a->reward, a->weight_changed_choices, a->verified_resumes,
                a->life.decisions, a->life.observations, a->life.updates, nt_spa_agent_hash(&a->life));
    }
    fputs("]}\n", trace);
    if (ferror(trace) || fclose(trace)) fail("trace close failed");
    free(tokens); free(starts);
    nt_tape_destroy();
    for (int i = 0; i < PARAMS; i++) nt_tensor_free(body.parameters[i]);
    free(body.parameters);
    return 0;
}
