/* Native operations for the spore experiments. The protocol, interventions and
 * statistics belong to AML. Scoring returns the full pre-sampling logits. */
#include "ariannamethod.h"
#include "harness/archs.h"
#include "harness/experts.h"
#include "examples/bpe.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int transplant(gguf_file *host, const char *recipient, const char *donor,
                      const char *out, int n, char **slots) {
    nt_experts *dst = nt_experts_open(host, recipient, NULL);
    nt_experts *src = nt_experts_open(host, donor, NULL);
    int ok = dst && src;
    for (int i = 0; ok && i < n; i++) {
        if (slots[i][0] < '0' || slots[i][0] > '9' || slots[i][1]) { ok = 0; break; }
        ok = nt_experts_copy_factors(dst, src, slots[i][0] - '0') == NT_OK;
    }
    if (ok) ok = nt_experts_save(dst, out) == 0;
    nt_experts_free(src);
    nt_experts_free(dst);
    return ok;
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
    if (argc == 7 && !strcmp(argv[1],"pack-logits"))
        return !pack_logits(argv[2],argv[3],argv[4],argv+5);
    if (argc >= 5 && !strcmp(argv[1], "compare-f32"))
        return !compare_f32(argv[2], argv[3], argc-4, argv+4);
    const char *tokenizer = NULL;
    if (argc > 3 && !strcmp(argv[1], "--tokenizer")) {
        tokenizer = argv[2]; argv += 2; argc -= 2;
    }
    if (argc < 7 || (strcmp(argv[1], "transplant") && strcmp(argv[1], "score"))) {
        fprintf(stderr, "doe_probe compare-f32 FILE1 FILE2 TENSOR...\n"
                        "doe_probe pack-logits HOST RAW_F32 CHECKPOINT LEFT RIGHT\n"
                        "doe_probe transplant HOST RECIPIENT DONOR NEW_SPORE SLOT...\n"
                        "doe_probe [--tokenizer FILE] score HOST SPORE|- CHECKPOINT LEFT RIGHT PROMPT...\n");
        return 1;
    }
    gguf_file *host = gguf_open(argv[2]);
    if (!host) return 1;
    const nt_arch *arch = nt_pick_arch(host->arch);
    int ok = arch && arch->doe;
    if (ok && !strcmp(argv[1], "transplant"))
        ok = !tokenizer && transplant(host, argv[3], argv[4], argv[5], argc-6, argv+6);
    else if (ok)
        ok = argc >= 8 && scores(host, arch, tokenizer, argv[3], argv[4], argv+5, argc-7, argv+7);
    if (!ok) fprintf(stderr, "doe_probe: %s failed for %s\n", argv[1], argv[2]);
    gguf_close(host);
    return !ok;
}
