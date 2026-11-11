/* Zero decoder matrices leave known embedding axes at the DOE hook. Llama's
 * anchors are e_k; Gemma's are 8*e_k. The independent variance oracle below
 * uses those coordinates directly, including the raw scale lost by QR. */
#include "harness/experts_internal.h"
#include "harness/spore_frame.h"
#include "harness/archs.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define CHECK(c, what) do { checks++; if (!(c)) { \
    fprintf(stderr,"FAIL vote calibration: %s (line %d)\n",what,__LINE__); failures++; \
} } while (0)

typedef struct {
    gguf_file *g;
    const nt_arch *arch;
    void *model;
    bpe_tokenizer *tok;
    nt_expert_body body;
    nt_dims dims;
} body;

typedef struct {
    nt_experts state;
    float *weights, *frame;
    nt_lineage_entry *ledger;
} snapshot;

static body open_body(const char *path) {
    body b={0};
    b.g=gguf_open(path); assert(b.g);
    b.arch=nt_pick_arch(b.g->arch); assert(b.arch && b.arch->doe);
    b.model=b.arch->load(b.g,&b.dims); b.tok=bpe_load(path);
    assert(b.model && b.tok);
    b.body=(nt_expert_body){b.model,b.tok}; return b;
}
static void close_body(body *b) { bpe_free(b->tok); b->arch->free(b->model); gguf_close(b->g); }

static nt_experts *open_state(body *b,const char *file) {
    nt_experts *p=nt_experts_open_on_body(b->g,file,NULL,&b->body); assert(p); return p;
}
static snapshot remember(const nt_experts *p) {
    snapshot s={.state=*p};
    size_t factors=(size_t)SEATS*p->dim*RANK, votes=(size_t)SEATS*p->dim;
    s.weights=malloc((2*factors+votes)*sizeof(float));
    s.ledger=malloc((size_t)p->ancestor_count*sizeof(*s.ledger)); assert(s.weights && s.ledger);
    memcpy(s.weights,p->a,factors*sizeof(float)); memcpy(s.weights+factors,p->b,factors*sizeof(float));
    memcpy(s.weights+2*factors,p->vote,votes*sizeof(float));
    memcpy(s.ledger,p->ancestors,(size_t)p->ancestor_count*sizeof(*s.ledger));
    if(p->frame) { s.frame=malloc((size_t)NT_SPORE_ANCHORS*p->dim*sizeof(float)); assert(s.frame);
        memcpy(s.frame,p->frame,(size_t)NT_SPORE_ANCHORS*p->dim*sizeof(float)); }
    return s;
}
static int state_same(const nt_experts *p,const snapshot *s) {
    size_t factors=(size_t)SEATS*p->dim*RANK;
    return !memcmp(p,&s->state,sizeof(*p)) &&
        !memcmp(p->a,s->weights,factors*sizeof(float)) &&
        !memcmp(p->b,s->weights+factors,factors*sizeof(float)) &&
        !memcmp(p->ancestors,s->ledger,(size_t)p->ancestor_count*sizeof(*s->ledger)) &&
        (!p->frame || !memcmp(p->frame,s->frame,(size_t)NT_SPORE_ANCHORS*p->dim*sizeof(float)));
}
static int votes_same(const nt_experts *p,const snapshot *s) {
    return !memcmp(p->vote,s->weights+2*(size_t)SEATS*p->dim*RANK,(size_t)SEATS*p->dim*sizeof(float));
}
static void forget(snapshot *s) { free(s->weights);free(s->frame);free(s->ledger); }

static int same_file(const char *left,const char *right) {
    FILE *a=fopen(left,"rb"),*b=fopen(right,"rb"); assert(a && b);
    int x,y,ok=1;
    do {x=fgetc(a);y=fgetc(b);if(x!=y)ok=0;}while(x!=EOF && y!=EOF);
    ok=ok && !ferror(a) && !ferror(b); assert(!fclose(a) && !fclose(b)); return ok;
}

static void complete(nt_experts *p) {
    float *x=calloc(p->dim,sizeof(float)); assert(x);
    x[0]=1;
    assert(!nt_experts_attend(p,p->layer,0,1,p->dim,x));
    assert(!nt_experts_observe(p,p->layer,0,1,p->dim,x));
    nt_experts_choice(p,1,1); assert(!nt_experts_finish(p,0)); free(x);
}
static void pattern(nt_experts *p) {
    /* Shared input-dependent offsets must not enter the centered RMS. Dead
     * rows are deliberately different: they scale but do not set the RMS. */
    for(int e=0;e<SEATS;e++) for(int i=0;i<p->dim;i++)
        p->vote[(size_t)e*p->dim+i]=(e<FOUNDERS ? e+1 : 100+e)+i/8.0f;
}
static double oracle(const nt_experts *p,double amplitude) {
    double ss=0; int living=0;
    for(int e=0;e<SEATS;e++) living+=p->alive[e]!=0;
    for(int k=0;k<NT_SPORE_ANCHORS;k++) {
        double mean=0;
        for(int e=0;e<SEATS;e++) if(p->alive[e]) mean+=amplitude*p->vote[(size_t)e*p->dim+k];
        mean/=living;
        for(int e=0;e<SEATS;e++) if(p->alive[e]) {
            double d=amplitude*p->vote[(size_t)e*p->dim+k]-mean;ss+=d*d;
        }
    }
    return sqrt(ss/(NT_SPORE_ANCHORS*living));
}
static void rows_oracle(body *b,double amplitude) {
    float *rows=malloc((size_t)NT_SPORE_ANCHORS*b->g->embed_dim*sizeof(float));assert(rows);
    CHECK(!nt_spore_anchor_rows(&b->body,b->g,b->arch,b->dims,rows),"raw anchors sampled");
    int exact=1;
    for(int k=0;k<NT_SPORE_ANCHORS;k++) for(int i=0;i<b->g->embed_dim;i++)
        if(rows[(size_t)k*b->g->embed_dim+i]!=(i==k ? (float)amplitude : 0))exact=0;
    CHECK(exact,"raw anchor rows match independent embedding axes and amplitude"); free(rows);
}

static void calibrate_and_restart(body *b,double amplitude,const char *label) {
    char before[80],portable[80],default_copy[80],after[80],portable_after[80],restart[80];
    snprintf(before,sizeof(before),"%s-before",label);snprintf(portable,sizeof(portable),"%s-portable",label);
    snprintf(default_copy,sizeof(default_copy),"%s-default",label);snprintf(after,sizeof(after),"%s-after",label);
    snprintf(portable_after,sizeof(portable_after),"%s-portable-after",label);snprintf(restart,sizeof(restart),"%s-restart",label);
    nt_experts *p=open_state(b,NULL);complete(p);pattern(p);
    assert(!nt_experts_save(p,before) && !nt_experts_save_portable(p,portable));
    nt_experts *copy=nt_experts_open(b->g,before,NULL);assert(copy && !nt_experts_save(copy,default_copy));
    CHECK(same_file(before,default_copy),"default exact serialization has no implicit calibration");nt_experts_free(copy);
    snapshot prior=remember(p);
    double before_rms=oracle(p,amplitude),target=.5;
    CHECK(fabs(before_rms-amplitude*sqrt(1.25))<1e-12,"independent centered four-live-expert RMS");
    nt_vote_calibration report;
    CHECK(!nt_experts_calibrate_votes(p,target,&report),"explicit calibration succeeds");
    CHECK(report.anchors==NT_SPORE_ANCHORS && report.living==FOUNDERS &&
        fabs(report.before_rms-before_rms)<1e-12 &&
        fabs(report.scale-target/before_rms)<1e-12 &&
        fabs(report.after_rms-oracle(p,amplitude))<1e-12 &&
        fabs(report.after_rms-target)<8*FLT_EPSILON*target,"measured report matches independent oracle");
    const float *old=prior.weights+2*(size_t)SEATS*p->dim*RANK;
    int exact=1;
    for(size_t i=0;i<(size_t)SEATS*p->dim;i++) if(p->vote[i]!=(float)(old[i]*(target/before_rms)))exact=0;
    CHECK(exact,"one common scale reaches every stored row including dead slots and non-anchor dimensions");
    CHECK(state_same(p,&prior),"A/B, frame, ledger, lifecycle, RNG, consensus and all clocks unchanged");
    assert(!nt_experts_save(p,after) && !nt_experts_save_portable(p,portable_after));
    snapshot calibrated=remember(p);
    for(int i=0;i<32;i++) {
        CHECK(!nt_experts_calibrate_votes(p,target,&report) && report.scale==1 &&
            state_same(p,&calibrated) && votes_same(p,&calibrated),"repeated stable calibration is bit-identical");
    }
    CHECK(!nt_experts_calibrate_votes(p,report.after_rms*(1+4*FLT_EPSILON),NULL) &&
        state_same(p,&calibrated) && votes_same(p,&calibrated),"epsilon no-op and optional report");
    copy=open_state(b,after);
    CHECK(!nt_experts_calibrate_votes(copy,target,&report) && report.scale==1 &&
        !memcmp(copy->vote,p->vote,(size_t)SEATS*p->dim*sizeof(float)),"same-host restart retains calibrated votes");
    assert(!nt_experts_save(copy,restart));CHECK(same_file(after,restart),"same-host restart remains exact bytes");nt_experts_free(copy);
    copy=nt_experts_open(b->g,portable_after,NULL);assert(copy);
    CHECK(!memcmp(copy->vote,p->vote,(size_t)SEATS*p->dim*sizeof(float)),"portable schema retains the calibrated rows without a new field");
    gguf_file *file=gguf_open(portable_after);assert(file);
    CHECK(gguf_get_kv(file,"doe.version")->val.u32==6 &&
          !gguf_get_kv(file,"doe.vote_scale"),"portable schema remains six");gguf_close(file);
    nt_experts_free(copy);forget(&calibrated);forget(&prior);nt_experts_free(p);
}

static void refusals(body *b) {
    nt_experts *p=open_state(b,NULL);nt_vote_calibration report={0};
    CHECK(nt_experts_calibrate_votes(p,1,&report)==NT_E_STATE,"uncompleted founder state refused");
    complete(p);pattern(p);snapshot prior=remember(p);
    double bad[]={0,-1,NAN,INFINITY,-INFINITY,DBL_MAX,DBL_MIN};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        report=(nt_vote_calibration){1,1,1,1,1};
        CHECK(nt_experts_calibrate_votes(p,bad[i],&report)!=NT_OK && report.anchors==0 &&
            report.living==0 && report.before_rms==0 && report.after_rms==0 && report.scale==0 &&
            state_same(p,&prior) && votes_same(p,&prior),"invalid or unrepresentable target refuses atomically");
    }
    CHECK(nt_experts_calibrate_votes(NULL,1,NULL)==NT_E_ARG,"null owner refused");
    int *pending[]={&p->count,&p->observed,&p->choices};
    for(size_t i=0;i<sizeof(pending)/sizeof(*pending);i++) {
        *pending[i]=1;CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE,"partial boundary refused");*pending[i]=0;
    }
    const bpe_tokenizer *tok=p->body.tokenizer;p->body.tokenizer=NULL;
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE,"missing tokenizer refused");p->body.tokenizer=tok;
    void *model=p->body.model;p->body.model=NULL;
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE,"missing decoder refused");p->body.model=model;
    float alive[SEATS];memcpy(alive,p->alive,sizeof(alive));memset(p->alive,0,sizeof(p->alive));p->alive[0]=1;
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE,"one living expert refused");memcpy(p->alive,alive,sizeof(alive));
    CHECK(state_same(p,&prior) && votes_same(p,&prior),"boundary refusals preserve all state");
    forget(&prior);
    size_t last=(size_t)SEATS*p->dim-1;
    p->vote[last]=9999;prior=remember(p);
    CHECK(nt_experts_calibrate_votes(p,2*oracle(p,1),&report)==NT_E_STATE &&
        state_same(p,&prior) && votes_same(p,&prior),"late dead-row bound failure rolls back earlier staged rows");forget(&prior);
    p->vote[last]=NAN;prior=remember(p);
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE &&
        state_same(p,&prior) && votes_same(p,&prior),"nonfinite stored row is refused without writes");forget(&prior);
    pattern(p);
    for(int e=0;e<FOUNDERS;e++)for(int i=0;i<p->dim;i++)p->vote[(size_t)e*p->dim+i]=1;
    prior=remember(p);
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE &&
        state_same(p,&prior) && votes_same(p,&prior),"zero centered spread is refused unchanged");
    forget(&prior);nt_experts_free(p);
    p=nt_experts_open(b->g,"source-before",NULL);assert(p);prior=remember(p);
    CHECK(nt_experts_calibrate_votes(p,1,NULL)==NT_E_STATE && state_same(p,&prior) && votes_same(p,&prior),
        "exact spore without a borrowed body cannot calibrate");forget(&prior);nt_experts_free(p);
}

int main(void) {
    body source=open_body("source"),gemma=open_body("gemma");
    rows_oracle(&source,1);rows_oracle(&gemma,8);
    calibrate_and_restart(&source,1,"source");calibrate_and_restart(&gemma,8,"gemma");
    refusals(&source);close_body(&gemma);close_body(&source);
    printf("Expert vote calibration: %d checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
