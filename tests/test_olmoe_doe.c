/* Two-layer sparse decoders with an analytically soluble attention, router and FFN.
 * Tokens have constant signed embeddings. Layer 0 preserves them. In layer 1,
 * uniform causal attention adds the mean signed value to channel 2. Each expert
 * writes its own multiple of SwiGLU to channel 1. Flip channel 0 just before
 * the FFN to select the other two experts while preserving the input RMS. */
#include "harness/archs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { WIDTH = 32, HEAD = 16, LAYERS = 2, EXPERTS = 4, USED = 2, VOCAB = 4, COUNT = 4 };
static const float eps = 0.25f;
static const int tokens[COUNT] = {2,0,3,1};
static const float coefficients[EXPERTS] = {0.75f,0.25f,-0.25f,-0.75f};
static int checks, failures;
#define CHECK(c,msg) do { ++checks; if (!(c)) { ++failures; fprintf(stderr,"FAIL: %s\n",msg); } } while(0)

static int sign_of(int token) { return (token & 1) ? -1 : 1; }

typedef struct {
    const char *name;
    uint64_t cols, rows, slices;
    const float *values;
} weight;

static uint64_t element_count(const uint64_t *shape, uint32_t rank) {
    uint64_t count = 1;
    for (uint32_t i = 0; i < rank; i++) {
        if (!shape[i] || shape[i] > UINT64_MAX / count) return 0;
        count *= shape[i];
    }
    return count;
}

static int write_weight(gguf_writer *w, int pass, int layer, const weight *a) {
    char name[96];
    if (layer < 0) snprintf(name,sizeof(name),"%s.weight",a->name);
    else snprintf(name,sizeof(name),"blk.%d.%s.weight",layer,a->name);
    uint64_t shape[] = {a->cols,a->rows,a->slices};
    uint32_t rank = a->slices ? 3 : a->rows ? 2 : 1;
    uint64_t count = element_count(shape, rank);
    if (!count) return -1;
    if (pass == 0) return gguf_write_tensor_decl(w,name,rank,shape,GGUF_TYPE_F32);
    return gguf_write_tensor_f32(w,name,a->values,count);
}

enum defect { VALID, SHORT_NORM, WIDE_HEADER, SHALLOW_HEADER };
static int fixture(const char *path, const char *family, enum defect defect) {
    float zero[EXPERTS*WIDTH*WIDTH] = {0}, ones[WIDTH], norms[2*LAYERS+1][WIDTH];
    float embed[VOCAB*WIDTH], output[VOCAB*WIDTH] = {0}, router[EXPERTS*WIDTH] = {0};
    float value[HEAD*WIDTH] = {0}, project[WIDTH*WIDTH] = {0};
    float gate[EXPERTS*WIDTH*WIDTH] = {0}, down[EXPERTS*WIDTH*WIDTH] = {0};
    for (int j=0; j<WIDTH; j++) ones[j] = 1;
    for (int i=0; i<=2*LAYERS; i++) {
        memcpy(norms[i],ones,sizeof(ones));
        norms[i][WIDTH-1] = 2+i;  /* distinguish the ordered fingerprint vectors */
    }
    for (int t=0; t<VOCAB; t++) {
        for (int j=0; j<WIDTH; j++) embed[t*WIDTH+j] = sign_of(t);
        output[t*WIDTH+t] = 1;
    }
    for (int e=0; e<EXPERTS; e++) {
        router[e*WIDTH] = coefficients[e];
        gate[e*WIDTH*WIDTH+1] = 1;
        down[e*WIDTH*WIDTH+WIDTH] = (e+1)*0.125f;
    }
    value[1] = 1; project[2*WIDTH] = 1;
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w,"general.architecture",family);
    char key[96];
    const struct { const char *field; int value; } metadata[] = {
        {"embedding_length",WIDTH},{"block_count",LAYERS},{"context_length",COUNT},
        {"attention.head_count",2},{"attention.head_count_kv",1},
        {"feed_forward_length",WIDTH},{"expert_count",EXPERTS},{"expert_used_count",USED}
    };
    for (size_t i=0; i<sizeof(metadata)/sizeof(*metadata); i++) {
        snprintf(key,sizeof(key),"%s.%s",family,metadata[i].field);
        gguf_write_kv_u32(w,key,metadata[i].value);
    }
    snprintf(key,sizeof(key),"%s.attention.layer_norm_rms_epsilon",family);
    gguf_write_kv_f32(w,key,eps);
    if (defect==WIDE_HEADER) gguf_write_kv_u32(w,"stray.embedding_length",WIDTH+1);
    if (defect==SHALLOW_HEADER) gguf_write_kv_u32(w,"stray.block_count",LAYERS-1);
    const weight global[] = {
        {"token_embd",WIDTH,VOCAB,0,embed},{"output",WIDTH,VOCAB,0,output},
        {"output_norm",WIDTH-(defect==SHORT_NORM),0,0,norms[2*LAYERS]}
    };
    int qnorm = !strcmp(family,"qwen3moe") ? HEAD : WIDTH;
    int rc = 0;
    for (int pass=0; pass<2; pass++) {
        for (size_t i=0; i<sizeof(global)/sizeof(*global); i++) rc |= write_weight(w,pass,-1,&global[i]);
        for (int l=0; l<LAYERS; l++) {
            const weight local[] = {
                {"attn_norm",WIDTH,0,0,norms[2*l]},{"attn_q",WIDTH,WIDTH,0,zero},
                {"attn_k",WIDTH,HEAD,0,zero},{"attn_v",WIDTH,HEAD,0,l ? value : zero},
                {"attn_output",WIDTH,WIDTH,0,l ? project : zero},
                {"attn_q_norm",qnorm,0,0,ones},{"attn_k_norm",HEAD,0,0,ones},
                {"ffn_norm",WIDTH,0,0,norms[2*l+1]},{"ffn_gate_inp",WIDTH,EXPERTS,0,router},
                {"ffn_gate_exps",WIDTH,WIDTH,EXPERTS,gate},{"ffn_up_exps",WIDTH,WIDTH,EXPERTS,gate},
                {"ffn_down_exps",WIDTH,WIDTH,EXPERTS,l ? down : zero}
            };
            for (size_t i=0; i<sizeof(local)/sizeof(*local); i++) rc |= write_weight(w,pass,l,&local[i]);
        }
    }
    return gguf_write_close(w) || rc;
}

typedef struct {
    double attended, gate[USED], delta;
    int selected[USED];
} reference;

static reference expected(int layer, int pos, int flip, int renorm) {
    reference r = {0};
    int s = sign_of(tokens[pos]);
    int sum = 0;
    for (int i=0; i<=pos; i++) sum += sign_of(tokens[i]);
    r.attended = s + (layer ? sum/((pos+1)*sqrt(1+eps)) : 0);
    double scale = 1/sqrt((WIDTH-1+r.attended*r.attended)/WIDTH+eps);
    int direction = layer && flip ? -s : s;
    double denominator = 0;
    for (int k=0; k<USED; k++) r.selected[k] = direction>0 ? k : EXPERTS-1-k;
    for (int e=0; e<EXPERTS; e++) {
        if (!renorm || e==r.selected[0] || e==r.selected[1])
            denominator += exp(coefficients[e]*direction*scale);
    }
    for (int k=0; k<USED; k++) {
        int e = r.selected[k];
        r.gate[k] = exp(coefficients[e]*direction*scale)/denominator;
        if (layer) r.delta += r.gate[k]*(e+1)*0.125*scale*scale/(1+exp(-s*scale));
    }
    return r;
}

typedef struct {
    int flip, renorm, start, n, stop_pre, stop_post;
    int before[LAYERS], after[LAYERS], routed[LAYERS][COUNT];
} observation;

static int before_ffn(void *arg,int layer,int pos,int n,int width,float *x) {
    observation *o = arg;
    if (layer<0 || layer>=LAYERS || pos!=o->start || n!=o->n || width!=WIDTH) return NT_E_ARG;
    o->before[layer]++;
    if (o->stop_pre && layer==1) return 71;
    for (int j=0; j<n; j++) {
        reference r = expected(layer,pos+j,o->flip,o->renorm);
        float *row = x+j*width;
        CHECK(row[0]==sign_of(tokens[pos+j]) && fabs(row[2]-r.attended)<1e-5,
              "pre-FFN hook sees the attention residual at each position");
        if (layer && o->flip) row[0] = -row[0];
    }
    return NT_OK;
}

static int after_ffn(void *arg,int layer,int pos,int n,int width,float *x) {
    observation *o = arg;
    if (layer<0 || layer>=LAYERS || pos!=o->start || n!=o->n || width!=WIDTH) return NT_E_ARG;
    o->after[layer]++;
    if (o->stop_post && layer==1) return 72;
    for (int j=0; j<n; j++) {
        reference r = expected(layer,pos+j,o->flip,o->renorm);
        int s = sign_of(tokens[pos+j]);
        CHECK(fabs(x[j*width+1]-(s+r.delta))<1e-5,
              "post-block residual includes exactly the selected weighted expert outputs");
    }
    return NT_OK;
}

/* Distinct cookies let a forward with active hooks also carry an independent observer. */
typedef struct { observation *obs; } route_cookie;
static void on_route(void *cookie,int layer,int pos,int token,int k,const int *ids,const float *weights) {
    observation *o = ((route_cookie *)cookie)->obs;
    int valid = layer>=0 && layer<LAYERS && pos>=o->start && pos<o->start+o->n &&
                token==tokens[pos] && k==USED && ids && weights;
    CHECK(valid,"route identifies the absolute position, input token, layer and top-k");
    if (!valid) return;
    o->routed[layer][pos]++;
    reference r = expected(layer,pos,o->flip,o->renorm);
    for (int i=0; i<k; i++) {
        CHECK(ids[i]==r.selected[i],"pre-FFN intervention selects the predicted ordered experts");
        CHECK(fabs(weights[i]-r.gate[i])<1e-6,"route reports each expert's applied weight");
    }
    double total = weights[0]+weights[1];
    CHECK(o->renorm ? fabs(total-1)<1e-6 : total<0.9,"chosen mass follows the GGUF family's policy");
}

static void run_case(const nt_arch *a, void *m, kv_cache *kv, int renorm, int flip, int chunk) {
    observation o = {.flip=flip,.renorm=renorm};
    route_cookie cookie = {&o};
    float output[VOCAB], plain[VOCAB], hooked[VOCAB];
    for (int p=0; p<COUNT; p+=chunk) {
        o.start = p; o.n = chunk;
        CHECK(a->forward_routes(m,kv,tokens+p,chunk,p,output,before_ffn,after_ffn,&o,on_route,&cookie)==NT_OK,
              "joint DOE and native routing forward succeeds");
    }
    for (int l=0; l<LAYERS; l++) {
        CHECK(o.before[l]==COUNT/chunk && o.after[l]==COUNT/chunk,"both hooks fire once per layer and chunk");
        for (int p=0; p<COUNT; p++) CHECK(o.routed[l][p]==1,"one native route per layer and position");
    }
    int last = COUNT-1, s = sign_of(tokens[last]);
    reference r = expected(1,last,flip,renorm);
    double final_channel[] = {flip ? -s : s,s+r.delta,r.attended,s};
    double ss = (WIDTH-2) + final_channel[1]*final_channel[1] + r.attended*r.attended;
    for (int i=0; i<VOCAB; i++)
        CHECK(fabs(output[i]-final_channel[i]/sqrt(ss/WIDTH+eps))<1e-5,
              "head logits agree with the independently weighted sparse FFN");
    for (int p=0; p<COUNT; p+=chunk) {
        o.start = p;
        CHECK(a->doe->forward(m,kv,tokens+p,chunk,p,hooked,before_ffn,after_ffn,&o)==NT_OK,
              "registered parliament uses both hooks");
    }
    CHECK(!memcmp(output,hooked,sizeof(output)),"observer leaves active-hook logits byte-identical");
    for (int p=0; p<COUNT; p+=chunk) CHECK(a->forward(m,kv,tokens+p,chunk,p,plain)==NT_OK,"plain forward");
    CHECK((memcmp(output,plain,sizeof(output))!=0)==flip,
          "passive hooks reproduce plain bytes and active intervention changes logits");
    if (!flip) {
        memset(o.routed,0,sizeof(o.routed));
        for (int p=0; p<COUNT; p+=chunk) {
            o.start = p;
            CHECK(a->forward_routes(m,kv,tokens+p,chunk,p,output,NULL,NULL,NULL,on_route,&cookie)==NT_OK,
                  "plain route-only forward");
        }
        for (int l=0; l<LAYERS; l++) for (int p=0; p<COUNT; p++)
            CHECK(o.routed[l][p]==1,"plain observer sees every native choice");
        CHECK(!memcmp(output,plain,sizeof(output)),"route-only preserves every plain logit");
    }
}

static void abort_case(const nt_arch *a,void *m,kv_cache *kv,int renorm,int pre) {
    observation o = {.renorm=renorm,.n=COUNT,.stop_pre=pre,.stop_post=!pre};
    route_cookie cookie = {&o};
    float plain[VOCAB], output[VOCAB], untouched[VOCAB];
    CHECK(a->forward(m,kv,tokens,COUNT,0,plain)==NT_OK,"baseline before interrupted run");
    for (int i=0; i<VOCAB; i++) untouched[i] = output[i] = -77.0f;
    int rc = a->forward_routes(m,kv,tokens,COUNT,0,output,before_ffn,after_ffn,&o,on_route,&cookie);
    CHECK(rc==(pre ? 71 : 72) && !memcmp(output,untouched,sizeof(output)),
          "pre- and post-FFN abort codes propagate with untouched logits");
    CHECK(o.routed[1][0]==!pre && o.after[1]==!pre,"aborting hook stops subsequent native routing");
    CHECK(a->forward(m,kv,tokens,COUNT,0,output)==NT_OK && !memcmp(output,plain,sizeof(plain)),
          "position-zero restart after abort matches the plain model");
}

static void check_stack_layout(const nt_arch *a, gguf_file *g) {
    int ti = gguf_find_tensor(g, "blk.1.ffn_gate_exps.weight");
    CHECK(ti >= 0, "expert stack is present");
    if (ti < 0) return;
    gguf_tensor_info *t = &g->tensors[ti], original = *t;
    nt_dims dims = {0};
    t->shape[1] = EXPERTS;
    t->shape[2] = WIDTH;
    void *model = a->load(g, &dims);
    CHECK(!model, "stack with swapped expert and hidden axes is refused");
    a->free(model);
    *t = original;
    t->offset = g->data_size;
    model = a->load(g, &dims);
    CHECK(!model, "stack outside the GGUF data is refused before its packed row is probed");
    a->free(model);
    *t = original;
    t->dtype = GGUF_TYPE_Q4_K;
    model = a->load(g, &dims);
    CHECK(!model, "a 32-wide row cannot store a 256-value Q4_K block");
    a->free(model);
    *t = original;
    t->ndim = 2;
    t->shape[1] *= t->shape[2];
    model = a->load(g, &dims);
    CHECK(model, "flattened stack retains its contiguous expert slices");
    a->free(model);
    *t = original;
}

static void family_test(const char *path, const char *family) {
    const nt_arch *a = nt_pick_arch(family);
    CHECK(a==&nt_arch_olmoe && a->doe && a->forward_routes,
          "registered sparse decoder offers DOE and route observation");
    if (!a || !a->doe || !a->forward_routes) return;
    for (int defect=VALID; defect<=SHALLOW_HEADER; defect++) {
        CHECK(fixture(path,family,defect)==0,"write sparse GGUF");
        gguf_file *g = gguf_open(path);
        CHECK(g && g->embed_dim==WIDTH+(defect==WIDE_HEADER),"fixture exposes its declared header width");
        nt_dims dims = {0}, described = {0};
        void *m = g ? a->load(g,&dims) : NULL;
        if (defect==VALID) {
            CHECK(m && a->doe->describe(m,&described)==g && described.n_layers==LAYERS &&
                  described.kv_dim==HEAD && described.vocab==VOCAB,"descriptor reproduces loaded geometry");
            if (m) {
                check_stack_layout(a,g);
                for (int i=0; i<=2*LAYERS; i++) {
                    const float *norm = a->doe->norm(m,i);
                    CHECK(norm && norm[WIDTH-1]==2+i,"portable frame obtains each ordered full-width norm");
                }
                CHECK(!a->doe->norm(m,-1) && !a->doe->norm(m,2*LAYERS+1),"invalid norm indices refused");
                kv_cache *kv = nt_model_kv_new(a,m,dims,COUNT);
                CHECK(kv,"allocate native KV");
                if (kv) {
                    int renorm = !strcmp(family,"qwen3moe");
                    for (int flip=0; flip<=1; flip++) {
                        run_case(a,m,kv,renorm,flip,2);
                        run_case(a,m,kv,renorm,flip,1);
                    }
                    abort_case(a,m,kv,renorm,0);
                    abort_case(a,m,kv,renorm,1);
                    nt_model_kv_free(a,m,kv);
                }
            }
        } else {
            int rejected = !m || !a->doe->describe(m,&described);
            if (!rejected && defect==SHORT_NORM) rejected = !a->doe->norm(m,2*LAYERS);
            CHECK(rejected,"malformed width, depth or norm cannot enter portable calibration");
        }
        if (m) a->free(m);
        if (g) gguf_close(g);
    }
}

int main(void) {
    CHECK(element_count((uint64_t[]){65536,65536,2},3) == (UINT64_C(1) << 33),
          "large fixture counts use all 64 bits");
    CHECK(element_count((uint64_t[]){UINT64_MAX,2},2) == 0,
          "fixture rejects an unrepresentable element count");
    char path[] = "/tmp/nt_olmoe_doe_XXXXXX";
    int fd = mkstemp(path);
    if (fd<0) return 1;
    close(fd);
    family_test(path,"olmoe");
    family_test(path,"qwen3moe");
    unlink(path);
    printf("OLMOE_DOE_%s (%d checks, %d failures)\n",failures ? "FAIL" : "OK",checks,failures);
    return failures!=0;
}
