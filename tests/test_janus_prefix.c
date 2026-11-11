#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "janus_prefix_fixture.h"

static int checks, failures;
#define CHECK(expr, label) do { checks++; if (!(expr)) { failures++; fprintf(stderr, "FAIL: %s (line %d)\n", label, __LINE__); } } while (0)

static int fixture(const char *path, int graph, float smear, float eps) {
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w, "general.architecture", "janus");
    const char *keys[] = {"vocab_size", "embedding_length", "attention.head_count", "attention.head_dim",
                         "block_count", "feed_forward_length", "context_length", "rrpram.rank"};
    int vals[] = {JP_V, JP_E, JP_H, JP_D, JP_B, JP_M, JP_T, JP_R};
    char key[96];
    for (int i = 0; i < 8; i++) {
        snprintf(key, sizeof(key), "janus.%s", keys[i]); gguf_write_kv_u32(w, key, vals[i]);
    }
    if (graph == 3) gguf_write_kv_u32(w, "janus.forward_graph", 1);
    else if (graph) gguf_write_kv_str(w, "janus.forward_graph", graph == 1 ? "pytorch-v4-prefix" : "unknown");
    if (eps) gguf_write_kv_f32(w, "janus.rms_norm_epsilon", eps);
    int count = (int)(sizeof(jp_spec) / sizeof(jp_spec[0]));
    for (int i = 0; i < count; i++)
        gguf_write_tensor_decl(w, jp_spec[i].name, jp_spec[i].nd, jp_spec[i].shape, GGUF_TYPE_F32);
    for (int i = 0; i < count; i++) {
        size_t n = 1;
        for (int d = 0; d < jp_spec[i].nd; d++) n *= jp_spec[i].shape[d];
        float *v = malloc(n * sizeof(float)); if (!v) return -1;
        for (size_t j = 0; j < n; j++) {
            int a = (int)((j * 17 + jp_spec[i].salt * 29 + (j / 7) * 3) % 97) - 48;
            v[j] = a / 64.0f * jp_spec[i].factor;
            if (jp_spec[i].kind == 1) v[j] = 1 + (j + 1) / 64.0f;
            if (jp_spec[i].kind == 2) v[j] = (3 - (int)j) / 32.0f;
            if (jp_spec[i].kind == 3) v[j] = smear;
            if (jp_spec[i].kind == 4) v[j] = .1875f;
        }
        int rc = gguf_write_tensor_f32(w, jp_spec[i].name, v, n); free(v);
        if (rc) return rc;
    }
    return gguf_write_close(w);
}

static float worst(const float *a, const float *b) {
    float d = 0;
    for (int i = 0; i < JP_V; i++) {
        if (!isfinite(a[i])) return INFINITY;
        float e = fabsf(a[i] - b[i]); if (e > d) d = e;
    }
    return d;
}

int main(void) {
    char path[] = "/tmp/nt-janus-prefix-XXXXXX";
    int fd = mkstemp(path); if (fd < 0) return 1; close(fd);
    CHECK(fixture(path, 1, .375f, 0) == 0, "fixture written");
    gguf_file *gf = gguf_open(path); if (!gf) return 1;
    nt_dims dims; const nt_arch *a = &nt_arch_janus;
    void *m = a->load(gf, &dims); if (!m) return 1;
    kv_cache *one = nt_model_kv_new(a, m, dims, JP_T), *two = nt_model_kv_new(a, m, dims, JP_T);
    CHECK(one && two, "separate caches"); if (!one || !two) return 1;
    const int ids[] = {1,8,3,11,5,2,20}, other[] = {9,4,17,6,21};
    float out[JP_V], full[JP_V], reference[JP_V];
    CHECK(a->forward(m, one, ids, 7, 0, full) == NT_OK, "full prefix");
    CHECK(worst(full, jp_gold[6]) < 2e-5f, "full prefix vs PyTorch");
    float maxdiff = worst(full, jp_gold[6]);
    for (int i = 0; i < 7; i++) {
        CHECK(a->forward(m, two, ids+i, 1, i, out) == NT_OK, "one-token append");
        float d = worst(out, jp_gold[i]); if (d > maxdiff) maxdiff = d;
        CHECK(d < 2e-5f, "every prefix vs PyTorch");
    }
    CHECK(!memcmp(full, out, sizeof(full)), "single/full identical");
    CHECK(a->forward(m, one, ids, 2, 0, NULL) == NT_OK, "deferred first chunk");
    CHECK(a->forward(m, one, ids+2, 3, 2, NULL) == NT_OK, "deferred second chunk");
    CHECK(a->forward(m, one, ids+5, 2, 5, out) == NT_OK, "final chunk");
    CHECK(!memcmp(full, out, sizeof(full)), "chunk/full identical");
    CHECK(a->forward(m, one, ids, 3, 0, NULL) == NT_OK, "interleave A begins");
    CHECK(a->forward(m, two, other, 5, 0, out) == NT_OK, "interleave B");
    CHECK(worst(out, jp_gold[7]) < 2e-5f, "B vs PyTorch");
    CHECK(a->forward(m, one, ids+3, 4, 3, out) == NT_OK, "interleave A continues");
    CHECK(!memcmp(full, out, sizeof(full)), "A independent of B");
    CHECK(a->forward(m, one, ids, 7, 0, out) == NT_OK && !memcmp(full,out,sizeof(full)), "reset identical");
    CHECK(a->forward(m, one, other, 5, 0, out) == NT_OK, "shorter reset");
    CHECK(worst(out, jp_gold[7]) < 2e-5f, "shorter reset vs PyTorch");
    memcpy(reference,out,sizeof(out));
    int bad=JP_V;
    CHECK(a->forward(m, one, &bad, 1, 5, out) == NT_E_TOKEN, "bad token refused");
    CHECK(a->forward(m, one, ids, 1, 6, out) == NT_E_STATE, "gap refused");
    CHECK(a->forward(m, one, ids, 1, 4, out) == NT_E_STATE, "nonzero rewind refused");
    CHECK(a->forward(m, one, ids, 1, JP_T, out) == NT_E_CAPACITY, "overflow refused");
    CHECK(a->forward(m, one, ids, 0, 0, out) == NT_E_ARG, "empty call refused");
    CHECK(!memcmp(out,reference,sizeof(out)), "errors preserve output");
    kv_cache *plain = kv_new(JP_B,JP_T,JP_E);
    CHECK(a->forward(m, plain, ids, 7, 0, out) == NT_E_CACHE, "plain cache refused safely"); kv_free(plain);
    CHECK(!nt_model_kv_new(a,m,dims,0), "zero capacity refused");
    CHECK(!nt_model_kv_new(a,m,dims,JP_T+1), "past-context allocation refused");
    nt_dims dims2; void *m2 = a->load(gf,&dims2);
    CHECK(m2 && a->forward(m2,one,ids,7,0,out)==NT_E_CACHE, "foreign model cache refused");
    a->free(m2);
    nt_model_kv_free(a,m,one); nt_model_kv_free(a,m,two); a->free(m); gguf_close(gf);

    for (int mode=0;mode<5;mode++) {
        int graph=mode<3?1:(mode==3?2:3);
        float smear=mode==0?-.25f:.375f, eps=mode==1?.0078125f:(mode==2?-1:0);
        CHECK(fixture(path,graph,smear,eps)==0,"variant fixture"); gf=gguf_open(path);m=a->load(gf,&dims);
        if (mode>=2) CHECK(!m,"invalid metadata refused");
        else {
            CHECK(m!=NULL,"valid variant loaded"); if (!m) return 1;
            one=nt_model_kv_new(a,m,dims,JP_T);
            CHECK(a->forward(m,one,ids,7,0,out)==NT_OK,"variant forward");
            CHECK(worst(out,jp_gold[mode+8])<2e-5f,"negative smear / explicit epsilon vs PyTorch");
            nt_model_kv_free(a,m,one);a->free(m);
        }
        gguf_close(gf);
    }
    CHECK(fixture(path,0,.375f,0)==0,"legacy fixture");gf=gguf_open(path);m=a->load(gf,&dims);
    one=nt_model_kv_new(a,m,dims,JP_T);
    CHECK(a->forward(m,one,ids,7,0,out)==NT_OK,"legacy still runs");
    CHECK(worst(out,jp_gold[6])>1e-3f,"oracle detects old graph");
    printf("janus prefix: %d checks, %d failures, PyTorch max abs %.9g; legacy delta %.9g\n",checks,failures,maxdiff,worst(out,jp_gold[6]));
    nt_model_kv_free(a,m,one);a->free(m);gguf_close(gf);unlink(path);
    return failures?1:0;
}
