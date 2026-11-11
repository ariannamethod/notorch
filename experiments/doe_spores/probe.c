/* Native operations for the spore experiments. The protocol, interventions and
 * statistics belong to AML. Scoring returns the full pre-sampling logits. */
#include "ariannamethod.h"
#include "harness/archs.h"
#include "harness/experts.h"
#include "harness/experts_internal.h"
#include "harness/tokenizer.h"
#include "examples/bpe.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int transplant(gguf_file *host, const char *recipient, const char *donor,
                      const char *out, int n, char **slots, int by_id) {
    nt_experts *dst = nt_experts_open(host, recipient, NULL);
    nt_experts *src = nt_experts_open(host, donor, NULL);
    int ok = dst && src;
    for (int i = 0; ok && i < n; i++) {
        if (by_id) { ok = nt_experts_copy_factors_by_identity(dst,src,slots[i]) == NT_OK; continue; }
        if (slots[i][0] < '0' || slots[i][0] > '9' || slots[i][1]) { ok = 0; break; }
        ok = nt_experts_copy_factors(dst, src, slots[i][0] - '0') == NT_OK;
    }
    if (ok) ok = nt_experts_save(dst, out) == 0;
    nt_experts_free(src);
    nt_experts_free(dst);
    return ok;
}

static int lineage(const char *host_path,const char *spore) {
    gguf_file *host=gguf_open(host_path);
    nt_experts *p=host ? nt_experts_open(host,spore,NULL) : NULL;
    if(!p) { gguf_close(host); return 0; }
    puts("identity\tparent\tsource_host\tturn\tdeath_turn\tkind\tliving_slot");
    for(int i=0;i<nt_experts_ancestor_count(p);i++) {
        nt_expert_identity e;
        if(nt_experts_ancestor(p,i,&e)) { nt_experts_free(p); gguf_close(host); return 0; }
        printf("%s\t%s\t%s\t%llu\t%llu\t%d\t%d\n",e.id,e.parent,e.source_host,
            (unsigned long long)e.turn,(unsigned long long)e.death_turn,e.kind,nt_experts_find_identity(p,e.id));
    }
    nt_experts_free(p); gguf_close(host); return 1;
}

/* Both target surfaces must extend the literal prompt by exactly their own
 * single token. Refuse truncation and tokenizer merges across the boundary. */
enum { CAP = 512 };
static int target_follows(bpe_tokenizer *tok, const char *prefix, const int *ids, int n,
                          const char *target, int id) {
    size_t size = strlen(prefix) + strlen(target) + 1;
    char *joined = malloc(size);
    if (!joined) return 0;
    snprintf(joined, size, "%s%s", prefix, target);
    int both[CAP+2];
    int count = bpe_encode(tok, joined, both, CAP+2);
    free(joined);
    return count == n+1 && both[n] == id && !memcmp(ids, both, (size_t)n*sizeof(int));
}

static int field(AM_Record *rec, const char *name, AML_Var value) {
    AM_String *key = am_string_new(name);
    int ok = key && am_record_set(rec, key, &value) == 0;
    am_string_free(key);
    return ok;
}

static int save_record(AM_Record *rec, const char *out) {
    char error[256] = {0};
    char *tmp = malloc(strlen(out) + sizeof(".XXXXXX"));
    int ok = 0;
    if (tmp) {
        sprintf(tmp, "%s.XXXXXX", out);
        int fd = mkstemp(tmp);
        if (fd >= 0) {
            close(fd);
            ok = am_checkpoint_save(rec, tmp, error, sizeof error) == 1 && link(tmp, out) == 0;
            unlink(tmp);
        }
        free(tmp);
    }
    if (!ok) fprintf(stderr, "could not publish new AML checkpoint %s: %s\n", out, error);
    return ok;
}

static int text_field(AM_Record *rec, const char *name, const char *text) {
    AM_String *s = am_string_new(text);
    int ok = s && field(rec, name, (AML_Var){.type=AML_TYPE_STRING, .string=s});
    am_string_free(s);
    return ok;
}

/* Counters are decimal text: an AML scalar is F32, and long-lived clocks
 * must not round after 2^24. The experiment parses only its bounded schedule. */
static int count_field(AM_Record *rec, const char *name, uint64_t value) {
    char text[32];
    snprintf(text, sizeof(text), "%llu", (unsigned long long)value);
    return text_field(rec, name, text);
}

/* Capture the same raw text input and stop policy as the resident driver,
 * without waking or learning. The numerical operation lives in the Harness. */
static int capture_text(gguf_file *host, const nt_arch *arch, const char *tokenizer,
                        const char *spore, const char *out, const char *prompt) {
    nt_dims dims = {0};
    void *model = arch->load(host, &dims);
    bpe_tokenizer *tok = NULL;
    int eos = -1, ok = 0;
    nt_experts *p = NULL;
    nt_replay_episode *episode = NULL;
    if (!model || nt_model_tokenizer_open(host, host->path, dims.vocab, tokenizer, &tok, &eos)) goto done;
    nt_expert_body body = {model, tok};
    p = nt_experts_open_on_body(host, spore, NULL, &body);
    int ids[CAP+1];
    int n = tok ? bpe_encode(tok, prompt, ids, CAP+1) : 0;
    if (!p || n < 1 || n > CAP) goto done;
    episode = nt_replay_capture(p, ids, NULL, n, NULL, 0, &eos, eos >= 0 ? 1 : 0);
    ok = episode && !nt_replay_save(episode, out);
    if (ok) ok = printf("%s\n", nt_replay_id(episode)) >= 0 && fflush(stdout) == 0;
done:
    nt_replay_free(episode); nt_experts_free(p); bpe_free(tok);
    if (model) arch->free(model);
    return ok;
}

static int metric_field(AM_Record *rec, int index, const char *suffix, double value) {
    char key[80];
    snprintf(key, sizeof(key), "e%d_%s", index, suffix);
    return isfinite(value) && fabs(value) <= FLT_MAX &&
        field(rec, key, (AML_Var){.type=AML_TYPE_FLOAT, .value=(float)value});
}

/* The checkpoint exposes F32 selection scores to AML. The TSV preserves the
 * native double measurements; equal AML scores use the episode address. */
static int measure_replays(gguf_file *host, const nt_arch *arch, const char *tokenizer,
                           const char *spore, const char *out, int count, char **paths) {
    if (count < 1 || count > 32) return 0;
    nt_dims dims = {0};
    void *model = arch->load(host, &dims);
    bpe_tokenizer *tok = NULL;
    int eos = -1, ok = 0;
    nt_experts *p = NULL;
    AM_Record *rec = am_record_new();
    if (!model || !rec || nt_model_tokenizer_open(host, host->path, dims.vocab, tokenizer, &tok, &eos)) goto done;
    nt_expert_body body = {model, tok};
    p = nt_experts_open_on_body(host, spore, NULL, &body);
    if (!p || !field(rec, "count", (AML_Var){.type=AML_TYPE_FLOAT, .value=(float)count})) goto done;
    puts("index\tepisode\tpositions\talive_mask\tmax_gate\tentropy\teffective_seats\tselected\tnonzero\tresponse\tdisagreement\tgate0\tgate1\tgate2\tgate3\tgate4\tgate5\tresponse0\tresponse1\tresponse2\tresponse3\tresponse4\tresponse5");
    ok = 1;
    for (int i = 0; ok && i < count; i++) {
        nt_replay_episode *e = nt_replay_load(p, paths[i]);
        nt_replay_measurement m;
        ok = e && !nt_experts_measure_replay(p, e, &m);
        nt_replay_free(e);
        if (!ok) break;
        char key[80];
        snprintf(key, sizeof(key), "e%d_id", i);
        ok = text_field(rec, key, m.episode) &&
             metric_field(rec, i, "positions", (double)m.positions) &&
             metric_field(rec, i, "alive", m.alive_mask) &&
             metric_field(rec, i, "max_gate", m.mean_max_gate) &&
             metric_field(rec, i, "entropy", m.mean_entropy) &&
             metric_field(rec, i, "effective", m.mean_effective_seats) &&
             metric_field(rec, i, "selected", m.mean_selected) &&
             metric_field(rec, i, "nonzero", m.mean_nonzero) &&
             metric_field(rec, i, "response", m.mean_response) &&
             metric_field(rec, i, "disagreement", m.mean_disagreement);
        printf("%d\t%s\t%llu\t%u\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",
               i, m.episode, (unsigned long long)m.positions, m.alive_mask, m.mean_max_gate,
               m.mean_entropy, m.mean_effective_seats, m.mean_selected, m.mean_nonzero,
               m.mean_response, m.mean_disagreement);
        for (int seat = 0; ok && seat < SEATS; seat++) {
            char suffix[32];
            snprintf(suffix, sizeof(suffix), "gate%d", seat);
            ok = metric_field(rec, i, suffix, m.gate_mass[seat]);
            snprintf(suffix, sizeof(suffix), "response%d", seat);
            ok = ok && metric_field(rec, i, suffix, m.response[seat]);
            snprintf(key, sizeof(key), "e%d_identity%d", i, seat);
            ok = ok && text_field(rec, key, m.identities[seat]);
        }
        for (int seat = 0; seat < SEATS; seat++) printf("\t%.17g", m.gate_mass[seat]);
        for (int seat = 0; seat < SEATS; seat++) printf("\t%.17g", m.response[seat]);
        putchar('\n');
    }
    if (ok) ok = fflush(stdout) == 0 && !ferror(stdout) && save_record(rec, out);
done:
    am_record_free(rec); nt_experts_free(p); bpe_free(tok);
    if (model) arch->free(model);
    return ok;
}

static int inspect(const char *host_path, const char *spore, const char *out) {
    gguf_file *host = gguf_open(host_path);
    nt_experts *p = host ? nt_experts_open(host, spore, NULL) : NULL;
    AM_Record *rec = am_record_new();
    AM_Array *row = am_array_new(SEATS);
    int ok = p && rec && row;
    if (ok) ok = text_field(rec,"host",p->host) &&
        count_field(rec,"turns",p->turns) && count_field(rec,"elections",p->elections) &&
        count_field(rec,"dream_cycles",p->sleep.cycles) && count_field(rec,"dream_choices",p->sleep.choices) &&
        count_field(rec,"dream_elections",p->sleep.elections) && count_field(rec,"dream_seen",p->sleep.seen) &&
        count_field(rec,"replay_cycles",p->replay.cycles) && count_field(rec,"replay_choices",p->replay.choices) &&
        count_field(rec,"replay_elections",p->replay.elections) && count_field(rec,"replay_prefill",p->replay.prefill) &&
        count_field(rec,"replay_stops",p->replay.stops) && text_field(rec,"replay_last",p->replay.last) &&
        count_field(rec,"rng",p->rng) && count_field(rec,"ancestors",(uint64_t)p->ancestor_count);
    if (ok) {
        const char *names[]={"alive","vitality","frequency","age","low","cursor"};
        const float *values[]={p->alive,p->vitality,p->frequency,p->age,p->low,p->cursor};
        for (int k=0;ok && k<6;k++) {
            memcpy(row->data,values[k],SEATS*sizeof(float));
            ok=field(rec,names[k],(AML_Var){.type=AML_TYPE_ARRAY,.array=row});
        }
        for (int e=0;ok && e<SEATS;e++) {
            nt_expert_identity id;
            char key[24];
            snprintf(key,sizeof(key),"identity%d",e);
            int found=nt_experts_seat_identity(p,e,&id)==0;
            ok=text_field(rec,key,found?id.id:"");
        }
    }
    if(ok) ok=save_record(rec,out);
    am_array_free(row); am_record_free(rec); nt_experts_free(p); gguf_close(host);
    if(!ok) fprintf(stderr,"doe_probe: could not inspect exact-host spore\n");
    return ok;
}

/* notorch_omni writes the first raw vocabulary row before modality suppression.
 * This adapter transports it to AML without replaying a different text template. */
static int pack_logits(const char *host, const char *raw, const char *out, char **targets) {
    bpe_tokenizer *tok = bpe_load(host);
    int n = tok ? bpe_n_vocab(tok) : 0;
    FILE *f = fopen(raw,"rb");
    AM_Array *a = n > 0 ? am_array_new(n) : NULL;
    AM_Record *rec = am_record_new();
    int ok = f && a && rec;
    if (ok) ok = fread(a->data,sizeof(float),(size_t)n,f)==(size_t)n && fgetc(f)==EOF && !ferror(f);
    for (int i=0; ok && i<n; i++) ok = isfinite(a->data[i]);
    int cand[2] = {-1,-1};
    for (int i=0; ok && i<2; i++) {
        int ids[2];
        ok = bpe_encode_raw(tok,targets[i],ids,2)==1 && ids[0]>=0 && ids[0]<n;
        if (ok) {
            cand[i]=ids[0];
            ok=field(rec,i ? "right_id" : "left_id",(AML_Var){.type=AML_TYPE_FLOAT,.value=(float)cand[i]});
        }
    }
    if (ok) ok = cand[0]!=cand[1] && field(rec,"probe0",(AML_Var){.type=AML_TYPE_ARRAY,.array=a}) && save_record(rec,out);
    if (f && fclose(f)) ok=0;
    am_array_free(a); am_record_free(rec); bpe_free(tok);
    if (!ok) fprintf(stderr,"doe_probe: invalid full vocabulary row or candidate tokens\n");
    return ok;
}

static int score_one(gguf_file *host, const nt_arch *arch, nt_expert_body body,
                     nt_dims dims, const char *spore,
                     const int *ids, int n, float *logits) {
    int plain = strcmp(spore, "-") == 0;
    nt_experts *p = plain ? NULL : nt_experts_open_on_body(host, spore, stderr, &body);
    kv_cache *kv = nt_model_kv_new(arch, body.model, dims, n);
    int ok = kv && (plain || p);
    for (int start = 0; ok && start < n; start += NT_PREFILL_CHUNK) {
        int count = n-start < NT_PREFILL_CHUNK ? n-start : NT_PREFILL_CHUNK;
        float *out = start+count == n ? logits : NULL;
        int rc = arch->doe->forward(body.model, kv, ids+start, count, start, out,
                                    p ? nt_experts_attend : NULL,
                                    p ? nt_experts_observe : NULL, p);
        ok = rc == NT_OK;
    }
    for (int i = 0; ok && i < dims.vocab; i++) ok = isfinite(logits[i]);
    nt_model_kv_free(arch, body.model, kv);
    nt_experts_free(p);
    return ok;
}

static int scores(gguf_file *host, const nt_arch *arch, const char *tokenizer,
                  const char *spore, const char *out,
                  char **targets, int probes, char **prompts) {
    nt_dims dims = {0};
    if (tokenizer) {
        char family[64];
        if (gguf_read_str_kv(tokenizer, "general.architecture", family, sizeof(family)) ||
            strcmp(family, host->arch)) return 0;
    }
    void *model = arch->load(host, &dims);
    bpe_tokenizer *tok = bpe_load(tokenizer ? tokenizer : host->path);
    AM_Record *rec = am_record_new();
    AM_Array *logits = model ? am_array_new(dims.vocab) : NULL;
    int cand[2] = {-1,-1};
    int ok = tok && logits && rec && (tokenizer ? bpe_n_vocab(tok) <= dims.vocab :
                                      bpe_n_vocab(tok) == dims.vocab);
    for (int i = 0; ok && i < 2; i++) {
        int ids[2];
        ok = bpe_encode_raw(tok, targets[i], ids, 2) == 1;
        if (ok) {
            cand[i] = ids[0];
            ok = cand[i] >= 0 && cand[i] < dims.vocab &&
                 field(rec, i ? "right_id" : "left_id", (AML_Var){.type=AML_TYPE_FLOAT, .value=(float)cand[i]});
        }
    }
    ok = ok && cand[0] != cand[1];
    for (int i = 0; ok && i < probes; i++) {
        int ids[CAP+1];
        int n = bpe_encode(tok, prompts[i], ids, CAP+1);
        ok = n > 0 && n <= CAP && (!host->ctx_len || n <= host->ctx_len);
        for (int j = 0; ok && j < 2; j++)
            ok = target_follows(tok, prompts[i], ids, n, targets[j], cand[j]);
        fprintf(stderr, "probe=%d tokens=%d\n", i, n);
        if (ok) ok = score_one(host, arch, (nt_expert_body){model, tok},
                               dims, spore, ids, n, logits->data);
        if (ok) {
            char name[32];
            snprintf(name, sizeof(name), "probe%d", i);
            ok = field(rec, name, (AML_Var){.type=AML_TYPE_ARRAY, .array=logits});
        }
    }
    if (ok) ok = save_record(rec,out);
    am_array_free(logits);
    am_record_free(rec);
    bpe_free(tok);
    if (model) arch->free(model);
    return ok;
}

/* Compare named, equally shaped F32 tensors on disk. An aligned spore can
 * preserve the learned factors while acquiring a different calibration frame. */
static int compare_f32(const char *left, const char *right, int n, char **names) {
    gguf_file *a = gguf_open(left), *b = gguf_open(right);
    int ok = a && b;
    puts("tensor\tfloat_count\tbytes\tidentical");
    for (int i = 0; ok && i < n; i++) {
        int ia = gguf_find_tensor(a, names[i]), ib = gguf_find_tensor(b, names[i]);
        const gguf_tensor_info *x = ia >= 0 ? &a->tensors[ia] : NULL;
        const gguf_tensor_info *y = ib >= 0 ? &b->tensors[ib] : NULL;
        ok = x && y && x->dtype == GGUF_TYPE_F32 && y->dtype == x->dtype &&
             x->ndim > 0 && x->ndim <= 4 && x->ndim == y->ndim &&
             !memcmp(x->shape, y->shape, x->ndim * sizeof(x->shape[0])) &&
             x->n_elements == y->n_elements;
        if (ok) {
            uint64_t bytes = gguf_type_size(x->dtype, x->n_elements);
            ok = bytes && bytes <= SIZE_MAX && x->offset <= a->data_size && y->offset <= b->data_size &&
                 bytes <= a->data_size - x->offset && bytes <= b->data_size - y->offset;
            if (ok) printf("%s\t%llu\t%llu\t%d\n", names[i], (unsigned long long)x->n_elements,
                           (unsigned long long)bytes,
                           !memcmp(a->data+x->offset, b->data+y->offset, (size_t)bytes));
        }
        if (!ok) fprintf(stderr, "doe_probe: missing or incompatible F32 tensor %s\n", names[i]);
    }
    gguf_close(b);
    gguf_close(a);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1],"lineage")) return !lineage(argv[2],argv[3]);
    if (argc == 5 && !strcmp(argv[1],"inspect")) return !inspect(argv[2],argv[3],argv[4]);
    if (argc == 7 && !strcmp(argv[1],"pack-logits"))
        return !pack_logits(argv[2],argv[3],argv[4],argv+5);
    if (argc >= 5 && !strcmp(argv[1], "compare-f32"))
        return !compare_f32(argv[2], argv[3], argc-4, argv+4);
    const char *tokenizer = NULL;
    if (argc > 3 && !strcmp(argv[1], "--tokenizer")) {
        tokenizer = argv[2]; argv += 2; argc -= 2;
    }
    if (argc >= 6 && (!strcmp(argv[1], "capture") || !strcmp(argv[1], "measure-replay"))) {
        gguf_file *host = gguf_open(argv[2]);
        const nt_arch *arch = host ? nt_pick_arch(host->arch) : NULL;
        int ok = arch && arch->doe;
        if (ok && !strcmp(argv[1], "capture"))
            ok = argc == 6 && capture_text(host, arch, tokenizer, argv[3], argv[4], argv[5]);
        else if (ok)
            ok = measure_replays(host, arch, tokenizer, argv[3], argv[4], argc-5, argv+5);
        gguf_close(host);
        if (!ok) fprintf(stderr, "doe_probe: %s failed\n", argv[1]);
        return !ok;
    }
    if (argc < 7 || (strcmp(argv[1], "transplant") && strcmp(argv[1], "transplant-id") && strcmp(argv[1], "score"))) {
        fprintf(stderr, "doe_probe compare-f32 FILE1 FILE2 TENSOR...\n"
                        "doe_probe pack-logits HOST RAW_F32 CHECKPOINT LEFT RIGHT\n"
                        "doe_probe transplant HOST RECIPIENT DONOR NEW_SPORE SLOT...\n"
                        "doe_probe transplant-id HOST RECIPIENT DONOR NEW_SPORE ID...\n"
                        "doe_probe lineage HOST SPORE\n"
                        "doe_probe inspect HOST SPORE CHECKPOINT\n"
                        "doe_probe [--tokenizer FILE] capture HOST SPORE NEW_EPISODE PROMPT\n"
                        "doe_probe [--tokenizer FILE] measure-replay HOST SPORE CHECKPOINT EPISODE...\n"
                        "doe_probe [--tokenizer FILE] score HOST SPORE|- CHECKPOINT LEFT RIGHT PROMPT...\n");
        return 1;
    }
    gguf_file *host = gguf_open(argv[2]);
    if (!host) return 1;
    const nt_arch *arch = nt_pick_arch(host->arch);
    int ok = arch && arch->doe;
    if (ok && (!strcmp(argv[1], "transplant") || !strcmp(argv[1], "transplant-id")))
        ok = !tokenizer && transplant(host, argv[3], argv[4], argv[5], argc-6, argv+6,!strcmp(argv[1],"transplant-id"));
    else if (ok)
        ok = argc >= 8 && scores(host, arch, tokenizer, argv[3], argv[4], argv+5, argc-7, argv+7);
    if (!ok) fprintf(stderr, "doe_probe: %s failed for %s\n", argv[1], argv[2]);
    gguf_close(host);
    return !ok;
}
