/* Frozen arithmetic examples plus a binary trajectory for cross-revision SPA.
 * This fixture links only legacy perception, so an unused Agent has no route
 * into ordinary inference. Expected values below are independent closed forms.
 */
#include "notorch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks;
static int failures;

static void near_value(const char *name, float got, double expected, double tol) {
    checks++;
    if (!isfinite(got) || fabs((double)got - expected) > tol) {
        fprintf(stderr, "FAIL legacy: %s got=%.9g expected=%.12g\n", name, got, expected);
        failures++;
    }
}

static void closed_forms(void) {
    const float table[12] = {1, 0, -2, 4, 0, 2, 4, -2, 3, -1, 0, 1};
    const int ids[3] = {0, 1, 2};
    const double weighted[4] = {13.0/7, 0, 6.0/7, 4.0/7};
    const double uniform[4] = {4.0/3, 1.0/3, 2.0/3, 1};
    float out[4] = {0};
    nt_spa_embed_sentence(ids, 3, table, 3, 4, 0.5f, out);
    for (int d = 0; d < 4; d++) near_value("weighted embedding", out[d], weighted[d], 2e-7);
    nt_spa_embed_sentence(ids, 3, table, 3, 4, 1, out);
    for (int d = 0; d < 4; d++) near_value("uniform embedding", out[d], uniform[d], 2e-7);
    nt_spa_embed_sentence(ids, 3, table, 3, 4, 0, out);
    for (int d = 0; d < 4; d++) near_value("last token embedding", out[d], table[8+d], 0);

    const int skipped[5] = {0, -1, 2, 3, 1};
    const double skip_expected[4] = {13.0/21, 4.0/3, 62.0/21, -8.0/7};
    nt_spa_embed_sentence(skipped, 5, table, 3, 4, 0.5f, out);
    for (int d = 0; d < 4; d++) near_value("invalid IDs preserve position", out[d], skip_expected[d], 3e-7);
    float def[4];
    nt_spa_embed_sentence(ids, 3, table, 3, 4, 0.85f, def);
    nt_spa_embed_sentence(ids, 3, table, 3, 4, -1, out);
    for (int d = 0; d < 4; d++) near_value("default recency", out[d], def[d], 0);
    nt_spa_embed_sentence(NULL, 3, table, 3, 4, 0.5f, out);
    for (int d = 0; d < 4; d++) near_value("null input unchanged", out[d], def[d], 0);

    const float query[4] = {2, 0, 0, 0};
    const float history[12] = {0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0};
    const float shifted[12] = {1000, 0, 0, 0, 1001, 0, 0, 0, 1002, 0, 0, 0};
    const float repeated[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    near_value("softmax scores 0,1,2", nt_spa_connectedness(query, 4, history, 3), 0.6652409557748219, 8e-8);
    near_value("softmax large shift", nt_spa_connectedness(query, 4, shifted, 3), 0.6652409557748219, 8e-8);
    near_value("equal history", nt_spa_connectedness(query, 4, repeated, 3), 1.0/3, 2e-8);
    near_value("single history", nt_spa_connectedness(query, 4, history, 1), 1, 0);
    near_value("empty history", nt_spa_connectedness(query, 4, history, 0), 0, 0);

    const float base[4] = {-2, 0, 1, 4};
    float logits[4];
    memcpy(logits, base, sizeof logits);
    nt_spa_modulate_logits(logits, 4, 0.75f, 0.5f);
    for (int d = 0; d < 4; d++) near_value("temperature 5/8", logits[d], (double)base[d]*1.6, 5e-7);
    memcpy(logits, base, sizeof logits);
    nt_spa_modulate_logits(logits, 4, -2, 0.5f);
    for (int d = 0; d < 4; d++) near_value("lower connectedness bound", logits[d], base[d], 0);
    nt_spa_modulate_logits(logits, 4, 2, 0.25f);
    for (int d = 0; d < 4; d++) near_value("upper connectedness bound", logits[d], (double)base[d]*4/3, 3e-7);
    memcpy(logits, base, sizeof logits);
    nt_spa_modulate_logits(logits, 4, 1, 1);
    for (int d = 0; d < 4; d++) near_value("temperature floor", logits[d], (double)base[d]*1000, 0.0003);
}

static uint32_t rng_state = 0x6138ce79u;
static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float fixture_float(void) {
    return ((int)(next_u32() % 8193) - 4096) / 1024.0f;
}
static int write_floats(const float *data, size_t count) {
    return fwrite(data, sizeof(float), count, stdout) == count ? 0 : 1;
}

static int trace(void) {
    enum {D=8, V=19, H=9};
    float table[V*D], history[H*D] = {0};
    for (int i=0; i<V*D; i++) table[i]=fixture_float();
    for (int step=0; step<1024; step++) {
        int ids[17], count=1+step%17;
        for (int i=0; i<count; i++) ids[i]=(int)(next_u32()%(V+2))-1;
        float embedding[D], logits[V];
        float alpha=(step%7 == 0) ? 0 : (step%7 == 1) ? 1 : 0.85f;
        nt_spa_embed_sentence(ids, count, table, V, D, alpha, embedding);
        int filled=step < H ? step : H;
        float conn=nt_spa_connectedness(embedding, D, history, filled);
        for (int i=0; i<V; i++) logits[i]=fixture_float();
        nt_spa_modulate_logits(logits, V, conn, 0.3f);
        if (write_floats(embedding,D) || write_floats(&conn,1) || write_floats(logits,V)) return 2;
        memcpy(history+(step%H)*D,embedding,sizeof embedding);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--trace")) return trace();
    closed_forms();
    if (failures) return 1;
    printf("SPA_LEGACY_FIXED_OK checks=%u\n", checks);
    return 0;
}
