/* Independent spore/trace oracle and public-API failure checks for a private episode. */
#include "harness/experts.h"
#include "harness/archs.h"
#include "harness/arch_models.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { SEATS = 6, RANK = 4, AXES = 32 };

static const float *tensor(const gguf_file *g, const char *name) {
    int i = gguf_find_tensor(g, name);
    assert(i >= 0 && g->tensors[i].dtype == GGUF_TYPE_F32);
    return (const float *)(g->data + g->tensors[i].offset);
}

static uint64_t counter(const gguf_file *g, const char *name) {
    const gguf_kv *v = gguf_get_kv(g, name);
    assert(v && (v->type == 4 || v->type == 10));
    return v->type == 4 ? v->val.u32 : v->val.u64;
}

/* On the source fixture the calibration frame is the identity on its first
 * 32 coordinates. Bq is exactly one column of B. */
static void expect_cue(const gguf_file *g, int width, int cue, int seat) {
    const float *a = tensor(g, "doe.lora_a"), *b = tensor(g, "doe.lora_b");
    const float *alive = tensor(g, "doe.alive"), *vitality = tensor(g, "doe.vitality");
    int expected = -1, expected_seat = -1;
    double best = -1;
    for (int k = 0; k < AXES; k++) {
        double total = 0, largest = -1;
        int contributor = -1;
        for (int e = 0; e < SEATS; e++) {
            if (!alive[e]) continue;
            double energy = 0;
            for (int i = 0; i < width; i++) {
                double y = 0;
                for (int r = 0; r < RANK; r++)
                    y += (double)a[(e*width+i)*RANK+r] * b[(e*RANK+r)*width+k];
                energy += y*y;
            }
            energy *= vitality[e];
            total += energy;
            if (energy > largest) { largest = energy; contributor = e; }
        }
        if (total > best) { best = total; expected = k; expected_seat = contributor; }
    }
    assert(cue == expected && seat == expected_seat);
}

static void check_episode(const char *parent, const char *child, const char *trace) {
    gguf_file *p = gguf_open(parent), *d = gguf_open(child);
    FILE *f = fopen(trace, "r");
    assert(p && d && f && p->n_tensors == d->n_tensors);
    int hits[SEATS] = {0}, choices = 0, votes = 0, cue = -1, seat = -1;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "doe dream-start ")) {
            assert(sscanf(strstr(line, " cue="), " cue=%d", &cue) == 1);
            assert(sscanf(strstr(line, " seat="), " seat=%d", &seat) == 1);
        }
        choices += strstr(line, "doe dream-choice ") != NULL;
        if (!strstr(line, "doe dream-vote ")) continue;
        votes++;
        char *gates = strstr(line, "gates=");
        assert(gates);
        gates += 6;
        for (int e = 0; e < SEATS; e++) {
            char *end;
            double g = strtod(gates, &end);
            assert(end != gates && (e == SEATS-1 ? *end == '\n' : *end == ','));
            gates = end + 1;
            hits[e] |= g > 0;
        }
    }
    assert(!fclose(f) && choices == 6 && votes == 7 && cue >= 0 && cue < AXES);
    assert(counter(d, "doe.version") == 6 && counter(d, "doe.dream.cycles") == 1);
    assert(counter(d, "doe.dream.seen") == (UINT32_C(1) << cue));
    assert(counter(d, "doe.dream.choices") == (unsigned)choices &&
           counter(d, "doe.dream.elections") == (unsigned)votes);
    assert(counter(p, "doe.turns") == counter(d, "doe.turns"));
    assert(counter(p, "doe.elections") == counter(d, "doe.elections"));
    assert(gguf_get_kv(p, "doe.consensus")->val.f32 == gguf_get_kv(d, "doe.consensus")->val.f32);
    assert(counter(p, "doe.rng") != counter(d, "doe.rng"));
    int np=0,nd=0;
    char **lp=gguf_read_str_array(parent,"doe.lineage.entries",&np);
    char **ld=gguf_read_str_array(child,"doe.lineage.entries",&nd);
    assert(lp && ld && np==nd);
    for(int i=0;i<np;i++) { assert(!strcmp(lp[i],ld[i])); free(lp[i]); free(ld[i]); }
    free(lp); free(ld);
    int32_t *sp=gguf_read_i32_array(parent,"doe.lineage.seats",&np);
    int32_t *sd=gguf_read_i32_array(child,"doe.lineage.seats",&nd);
    assert(sp && sd && np==SEATS && nd==SEATS && !memcmp(sp,sd,SEATS*sizeof(*sp)));
    free(sp); free(sd);
    int width = (int)counter(p, "doe.width");
    expect_cue(p, width, cue, seat);
    for (uint64_t t = 0; t < p->n_tensors; t++) {
        const char *name = p->tensors[t].name;
        const float *before = tensor(p, name), *after = tensor(d, name);
        size_t n = p->tensors[t].n_elements;
        if (!strcmp(name, "doe.lora_a") || !strcmp(name, "doe.lora_b") || !strcmp(name, "doe.cursor")) {
            size_t count = n/SEATS;
            for (int e = 0; e < SEATS; e++) {
                int changed = memcmp(before+e*count, after+e*count, count*sizeof(float)) != 0;
                assert(changed == hits[e]);
            }
        } else assert(!memcmp(before, after, n*sizeof(float)));
    }
    gguf_close(d); gguf_close(p);
}

/* Mutate one byte of the small fixture's metadata, preserving field lengths
 * so each case reaches the reader's clock/type validation. */
static void reject_field(gguf_file *host, const char *field, int relative, unsigned char value) {
    FILE *f = fopen("one", "rb");
    assert(f && !fseek(f, 0, SEEK_END));
    long n = ftell(f);
    assert(n > 0);
    rewind(f);
    unsigned char *bytes = malloc((size_t)n);
    assert(bytes && fread(bytes, 1, n, f) == (size_t)n && !fclose(f));
    size_t len = strlen(field);
    long i = 0;
    while (i <= n-(long)len && memcmp(bytes+i, field, len)) i++;
    assert(i <= n-(long)len && i+(long)len+relative < n);
    bytes[i+len+relative] = value;
    f = fopen("malformed", "wb");
    assert(f && fwrite(bytes, 1, n, f) == (size_t)n && !fclose(f));
    free(bytes);
    assert(!nt_experts_open(host, "malformed", NULL));
}

static void api(void) {
    gguf_file *host = gguf_open("source");
    nt_dims dims;
    assert(host);
    void *model = nt_arch_llama.load(host, &dims);
    bpe_tokenizer *tok = bpe_load("source");
    nt_expert_body body = {model, tok};
    assert(model && tok);
    nt_experts *p = nt_experts_open_on_body(host, "awake", NULL, &body);
    assert(p);
    assert(nt_experts_dream(p, 0, -1, 1) == NT_E_ARG);
    assert(nt_experts_dream(p, 33, -1, 1) == NT_E_ARG);
    assert(nt_experts_dream(p, 1, -2, 1) == NT_E_ARG);
    int ctx = host->ctx_len;
    host->ctx_len = 1;
    assert(nt_experts_dream(p, 6, -1, 1) == NT_E_CAPACITY);
    host->ctx_len = ctx;
    llama_model *lm = model;
    float scale = lm->out_norm[0];
    lm->out_norm[0] = NAN;
    assert(nt_experts_dream(p, 6, -1, 1) == NT_E_STATE);
    lm->out_norm[0] = scale;
    assert(!nt_experts_save_portable(p, "failure-copy"));
    /* The toy's tied head greedily repeats the cue ID; EOS ends its first choice. */
    int cue = -1;
    FILE *trace = fopen("one.trace", "r");
    char line[1024];
    assert(trace);
    while (fgets(line, sizeof(line), trace))
        if (strstr(line, "doe dream-start ")) assert(sscanf(strstr(line, " cue="), " cue=%d", &cue) == 1);
    assert(!fclose(trace) && cue >= 0);
    assert(!nt_experts_dream(p, 6, cue, 1));
    assert(!nt_experts_save(p, "eos-state"));
    gguf_file *eos = gguf_open("eos-state");
    assert(eos && counter(eos, "doe.dream.choices") == 1 && counter(eos, "doe.dream.elections") == 1);
    gguf_close(eos);
    nt_consolidation *job = NULL;
    assert(nt_experts_submit_dream(&p, &job, 0, -1) == NT_E_ARG && p && !job);
    nt_experts_free(p);
    p = nt_experts_open(host, "awake", NULL);
    assert(p && !nt_experts_submit_dream(&p, &job, 2, -1) && !p && job);
    assert(nt_experts_await(&p, &job) == NT_E_STATE && !p && !job);
    /* Signed clocks, zero cycles, impossible counts, missing mask, old schema. */
    reject_field(host, "doe.dream.cycles", 0, 11);
    reject_field(host, "doe.dream.cycles", 4, 0);
    reject_field(host, "doe.dream.choices", 4, 33);
    reject_field(host, "doe.dream.elections", 4, 5);
    reject_field(host, "doe.dream.seen", -1, 'x');
    reject_field(host, "doe.version", 4, 2);
    bpe_free(tok); nt_arch_llama.free(model); gguf_close(host);
}

int main(int argc, char **argv) {
    if (argc != 4) return 1;
    check_episode(argv[1], argv[2], argv[3]);
    api();
    return 0;
}
