/* Native conditioned consequence credit: exact means, one scaled update. */
#include "spa_agent.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char *gate;
static unsigned checks;
#define CHECK(c,msg) do { ++checks; if(!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",gate,__LINE__,msg); return 0; } } while(0)
#define SAME(a,b) (memcmp(&(a),&(b),sizeof(a))==0)
#define NEAR(a,b,t) (isfinite(a) && fabs((double)(a)-(b))<(t))

static int initialize(nt_spa_agent *a,int four_axes) {
    nt_spa_agent_config c;
    nt_spa_agent_config_default(&c); c.mode=NT_SPA_AGENT_LEARNED;
    c.exploration=0; c.learning_rate=0;
    if(four_axes) {
        c.reward_weights=(nt_spa_metrics){1,1,1,1,0,0,0}; c.cost_weight=0;
    }
    return nt_spa_agent_init(a,&c)==NT_SPA_OK;
}
static nt_spa_observation observation(unsigned target,unsigned count) {
    nt_spa_observation o;
    memset(&o,0,sizeof(o));
    o.embedding[0]=.25f; o.embedding[1]=-.5f; o.embedding[2]=.75f; o.embedding[3]=-.125f;
    o.connectedness=.4f; o.left_similarity=target ? .2f:0;
    o.right_similarity=target+1<count ? .8f:0;
    o.coherence=.5f; o.novelty=.3f; o.repetition=.1f; o.phase_lock=.5f;
    o.sentence_score=.7f; o.mean_sentence_score=1; o.temperature=.8f;
    o.sentence_index=target; o.sentence_count=count; o.reseed_count=2;
    return o;
}
static nt_spa_comparison comparison(const nt_spa_experience *e,unsigned repetition) {
    nt_spa_comparison c;
    unsigned i;
    memset(&c,0,sizeof(c)); c.source_life_hash=e->source_life_hash; c.horizon=4;
    c.action_mask=1u|(e->sentence_index ? 2u:0u)|(e->sentence_index+1<e->sentence_count ? 4u:0u);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(c.action_mask&(1u<<i)) {
        nt_spa_alternative *alt=&c.alternatives[i];
        alt->action.kind=(nt_spa_action_kind)i; alt->action.target=e->sentence_index;
        alt->action.source=i==0 ? NT_SPA_AGENT_NO_SOURCE : i==1 ? e->sentence_index-1 : e->sentence_index+1;
        alt->consequence.before=(nt_spa_metrics){.4f,.3f,.5f,.4f,.2f,.1f,.6f};
        alt->consequence.after=alt->consequence.before;
        if(i==1) {
            alt->consequence.after.coherence=.7f-.03f*repetition;
            alt->consequence.after.novelty=.6f-.02f*repetition;
            alt->consequence.regeneration_cost=.25f;
        } else if(i==2) {
            alt->consequence.after.coherence=.3f+.03f*repetition;
            alt->consequence.after.repetition=.5f-.02f*repetition;
            alt->consequence.regeneration_cost=.75f;
        }
    }
    return c;
}
static void neutralize(nt_spa_comparison *c) {
    unsigned i;
    for(i=0;i<3;++i) if(c->action_mask&(1u<<i)) {
        c->alternatives[i].consequence.before=(nt_spa_metrics){.5f,.5f,.5f,.5f,.5f,.5f,.5f};
        c->alternatives[i].consequence.after=c->alternatives[i].consequence.before;
        c->alternatives[i].consequence.regeneration_cost=0;
    }
}
static int only_policy(const nt_spa_agent *before,const nt_spa_agent *after) {
    nt_spa_agent expected=*before; expected.policy=after->policy;
    return !memcmp(&expected,after,sizeof(expected));
}

static int test_parity(void) {
    static const unsigned targets[]={0,1,3,0},counts[]={4,3,4,1};
    static const float rates[]={0,.03125f,1};
    unsigned k,rate,n,i;
    for(k=0;k<4;++k) {
        nt_spa_agent initial,a,b;
        nt_spa_observation o=observation(targets[k],counts[k]);
        nt_spa_experience e;
        nt_spa_comparison repeated[64];
        nt_spa_comparison_receipt old;
        nt_spa_conditioned_receipt scaled;
        CHECK(initialize(&initial,0),"parity init");
        for(i=0;i<3;++i) { initial.policy.w2[i][0]=-0.0f; initial.policy.b2[i]=-0.0f; }
        CHECK(nt_spa_agent_capture_experience(&initial,&o,&e)==NT_SPA_OK,"parity capture");
        for(i=0;i<64;++i) repeated[i]=comparison(&e,i%3);
        for(n=1;n<=64;++n) for(rate=0;rate<3;++rate) {
            a=initial; b=initial;
            CHECK(nt_spa_agent_fit_repeated(&a,&e,repeated,n,rates[rate],&old)==NT_SPA_OK,
                "old repeated objective evaluates");
            CHECK(nt_spa_agent_fit_conditioned(&b,&e,repeated,n,rates[rate],1,&scaled)==NT_SPA_OK,
                "all repeat counts and action boundaries condition");
            CHECK(scaled.scale==1 && scaled.scale_floor==1 && SAME(old,scaled.comparison) && SAME(a,b),
                "unit scale reproduces complete old receipt and life bytes");
            CHECK(only_policy(&initial,&b),"conditioned fit changes policy only");
            if(rate==0) CHECK(SAME(b,initial),"zero rate preserves complete life");
            if(n==1) {
                b=initial;
                CHECK(nt_spa_agent_fit_comparison(&b,&e,repeated,rates[rate],NULL)==NT_SPA_OK && SAME(a,b),
                    "single-count unit scale retains original comparison API");
            }
            for(i=0;i<3;++i) if(!(old.action_mask&(1u<<i)))
                CHECK(!memcmp(b.policy.w2[i],initial.policy.w2[i],sizeof(b.policy.w2[i])) &&
                    !memcmp(&b.policy.b2[i],&initial.policy.b2[i],sizeof(float)),
                    "inactive head bytes including signed zero survive");
        }
        a=initial; b=initial;
        CHECK(nt_spa_agent_fit_conditioned(&a,&e,repeated,8,.03f,1e-6f,NULL)==NT_SPA_OK &&
            nt_spa_agent_fit_conditioned(&b,&e,repeated,8,.03f,1e-6f,&scaled)==NT_SPA_OK && SAME(a,b),
            "optional receipt preserves identical update");
    }
    return 1;
}

static int refused(nt_spa_agent *a,const nt_spa_experience *e,nt_spa_comparison *c,
    uint32_t count,float rate,float floor) {
    nt_spa_agent before=*a;
    nt_spa_experience saved=*e;
    nt_spa_comparison originals[64];
    nt_spa_conditioned_receipt r,sentinel;
    unsigned extent=count<=64 ? count:64;
    memcpy(originals,c,extent*sizeof(*c)); memset(&sentinel,0xa5,sizeof(sentinel)); r=sentinel;
    CHECK(nt_spa_agent_fit_conditioned(a,e,c,count,rate,floor,&r)!=NT_SPA_OK,"malformed fit refused");
    CHECK(SAME(*a,before) && SAME(*e,saved) && SAME(r,sentinel) &&
        !memcmp(c,originals,extent*sizeof(*c)),"refusal preserves life, receipt and every input byte");
    return 1;
}
static int test_validation(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e,bad_e;
    nt_spa_comparison good[64],bad[64];
    nt_spa_conditioned_receipt r;
    nt_spa_decision d;
    unsigned i;
    static const float bad_floor[]={0,-0.0f,-.1f,NAN,INFINITY,-INFINITY};
    CHECK(initialize(&a,0),"refusal init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"refusal capture");
    for(i=0;i<64;++i) good[i]=comparison(&e,i%3);
    before=a;
    CHECK(refused(&a,&e,good,0,.03f,1e-6f) && refused(&a,&e,good,65,.03f,1e-6f) &&
        refused(&a,&e,good,UINT32_MAX,.03f,1e-6f),"count bounds precede size multiplication");
    for(i=0;i<sizeof(bad_floor)/sizeof(*bad_floor);++i)
        CHECK(refused(&a,&e,good,2,.03f,bad_floor[i]),"invalid scale floor refused");
    CHECK(refused(&a,&e,good,2,-.1f,1e-6f) && refused(&a,&e,good,2,1.1f,1e-6f) &&
        refused(&a,&e,good,2,NAN,1e-6f) && refused(&a,&e,good,2,INFINITY,1e-6f),"invalid rates refused");
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,NULL,2,.03f,1e-6f,&r)!=NT_SPA_OK && SAME(a,before),
        "null comparisons refused");
    CHECK(nt_spa_agent_fit_conditioned(&a,NULL,good,2,.03f,1e-6f,&r)!=NT_SPA_OK && SAME(a,before),
        "null experience refused");
    CHECK(nt_spa_agent_fit_conditioned(NULL,&e,good,2,.03f,1e-6f,&r)!=NT_SPA_OK,"null life refused");
    bad_e=e; bad_e.features[0]=NAN;
    CHECK(refused(&a,&bad_e,good,2,.03f,1e-6f),"malformed experience refused");
#define BAD_LAST(field,value) do { memcpy(bad,good,sizeof(bad)); bad[63].field=(value); \
    CHECK(refused(&a,&e,bad,64,.03f,1e-6f),"invalid last member " #field); } while(0)
    BAD_LAST(source_life_hash,e.source_life_hash^UINT64_C(1));
    BAD_LAST(action_mask,3); BAD_LAST(horizon,5); BAD_LAST(horizon,4097);
    BAD_LAST(alternatives[0].action.source,0);
    BAD_LAST(alternatives[1].action.kind,NT_SPA_RESEED_RIGHT);
    BAD_LAST(alternatives[2].action.source,0); BAD_LAST(alternatives[2].action.target,0);
    BAD_LAST(alternatives[2].consequence.after.novelty,NAN);
    BAD_LAST(alternatives[2].consequence.after.novelty,1.01f);
    BAD_LAST(alternatives[2].consequence.regeneration_cost,-.01f);
    BAD_LAST(alternatives[2].consequence.regeneration_cost,INFINITY);
    BAD_LAST(alternatives[0].consequence.before.coherence,.7f);
    BAD_LAST(alternatives[2].consequence.before.coherence,.7f);
#undef BAD_LAST
    memcpy(bad,good,sizeof(bad));
    for(i=0;i<3;++i) bad[63].alternatives[i].consequence.before.novelty=.2f;
    CHECK(refused(&a,&e,bad,64,.03f,1e-6f),"same before axes required across repetitions");
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,good,64,.03f,1e-6f,(nt_spa_conditioned_receipt*)&a)!=NT_SPA_OK &&
        SAME(a,before),"receipt cannot overlap life");
    bad_e=e;
    CHECK(nt_spa_agent_fit_conditioned(&a,&bad_e,good,64,.03f,1e-6f,(nt_spa_conditioned_receipt*)&bad_e)!=NT_SPA_OK &&
        SAME(a,before) && SAME(bad_e,e),"receipt cannot overlap experience");
    memcpy(bad,good,sizeof(bad));
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,bad,64,.03f,1e-6f,(nt_spa_conditioned_receipt*)&bad[63])!=NT_SPA_OK &&
        SAME(a,before) && !memcmp(bad,good,sizeof(bad)),"receipt cannot overlap last array member");
    {
        union { nt_spa_comparison comparisons[2]; nt_spa_experience experience; } overlap,saved;
        memset(&overlap,0,sizeof(overlap)); overlap.experience=e; saved=overlap;
        CHECK(nt_spa_agent_fit_conditioned(&a,&overlap.experience,overlap.comparisons,2,.03f,1e-6f,&r)!=NT_SPA_OK &&
            SAME(a,before) && SAME(overlap,saved),"input arrays cannot overlap each other");
    }
    {
        union { nt_spa_agent aligned; unsigned char bytes[sizeof(nt_spa_conditioned_receipt)+sizeof(nt_spa_agent)]; } storage,saved;
        nt_spa_agent *embedded=(nt_spa_agent*)(void*)(storage.bytes+offsetof(nt_spa_conditioned_receipt,scale));
        memset(&storage,0xa5,sizeof(storage)); memcpy(embedded,&a,sizeof(a)); saved=storage;
        CHECK(nt_spa_agent_fit_conditioned(embedded,&e,good,2,.03f,1e-6f,
            (nt_spa_conditioned_receipt*)(void*)storage.bytes)!=NT_SPA_OK && SAME(storage,saved),
            "new scale tail overlapping life is checked beyond old receipt size");
    }
    {
        union { nt_spa_agent aligned; unsigned char bytes[sizeof(nt_spa_conditioned_receipt)+sizeof(nt_spa_experience)]; } storage,saved;
        nt_spa_experience *embedded=(nt_spa_experience*)(void*)(storage.bytes+offsetof(nt_spa_conditioned_receipt,scale));
        memset(&storage,0xa5,sizeof(storage)); memcpy(embedded,&e,sizeof(e)); saved=storage;
        CHECK(nt_spa_agent_fit_conditioned(&a,embedded,good,2,.03f,1e-6f,
            (nt_spa_conditioned_receipt*)(void*)storage.bytes)!=NT_SPA_OK && SAME(storage,saved) && SAME(a,before),
            "new scale tail overlapping experience is checked beyond old receipt size");
    }
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"create pending credit");
    CHECK(refused(&a,&e,good,2,.03f,1e-6f),"pending life refused");
    CHECK(nt_spa_agent_cancel(&a,d.sequence)==NT_SPA_OK,"cancel pending decision");
    a.version=0;
    CHECK(refused(&a,&e,good,2,.03f,1e-6f),"malformed persistent life refused");
    {
        nt_spa_agent_config config;
        nt_spa_agent_config_default(&config);
        for(i=0;i<2;++i) {
            config.mode=(nt_spa_agent_mode)i;
            CHECK(nt_spa_agent_init(&a,&config)==NT_SPA_OK && refused(&a,&e,good,2,.03f,1e-6f),
                "disabled and legacy lives refuse replay fitting");
        }
    }
    o=observation(0,4); CHECK(initialize(&a,0),"inactive slot init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"inactive slot capture");
    good[0]=comparison(&e,0); good[0].alternatives[1].consequence.after.coherence=.1f;
    CHECK(refused(&a,&e,good,1,.03f,1e-6f),"inactive alternative must retain zero bytes");
    return 1;
}

static int test_aggregate(void) {
    nt_spa_agent a,initial,b;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[3],saved[3];
    nt_spa_conditioned_receipt r,scaled;
    unsigned i;
    CHECK(initialize(&a,1),"aggregation init"); initial=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"aggregation capture");
    for(i=0;i<3;++i) { c[i]=comparison(&e,0); neutralize(&c[i]); }
    c[0].alternatives[1].consequence.after.local_connectedness=1;
    c[1].alternatives[1].consequence.after.local_connectedness=.125f;
    c[0].alternatives[2].consequence.after.local_connectedness=.5625f;
    c[1].alternatives[2].consequence.after.local_connectedness=.5625f;
    memcpy(saved,c,sizeof(c));
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,2,0,1e-6f,&r)==NT_SPA_OK && SAME(a,initial),
        "evaluate opposite-sign paired futures without updating");
    CHECK(r.comparison.rewards[0]==0 && r.comparison.rewards[1]==.0625f && r.comparison.rewards[2]==.0625f,
        "exact mean of plus1/2 and minus3/8 is plus1/16");
    CHECK(r.scale==.0625 && r.comparison.targets[0]==0 && r.comparison.targets[1]==1 && r.comparison.targets[2]==1,
        "normalize the completed mean, never separately normalized repetitions");
    CHECK(NEAR(r.comparison.loss_before,1.0/3,1e-12),"independent mean Huber at zero heads");
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,2,.12f,1e-6f,&r)==NT_SPA_OK &&
        NEAR(a.policy.b2[1],.04,1e-8) && NEAR(a.policy.b2[2],.04,1e-8),
        "one simultaneous mean-Huber update credits both positive heads");
    CHECK(r.comparison.loss_after<r.comparison.loss_before && only_policy(&initial,&a) && !memcmp(c,saved,sizeof(c)),
        "scaled learning lowers loss and preserves inputs and nonpolicy state");
    b=initial;
    CHECK(nt_spa_agent_fit_conditioned(&b,&e,c,1,.12f,1e-6f,NULL)==NT_SPA_OK &&
        nt_spa_agent_fit_conditioned(&b,&e,c+1,1,.12f,1e-6f,NULL)==NT_SPA_OK && !SAME(a.policy,b.policy),
        "one mean update differs from sequential single-future updates");
    a=initial;
    c[0].alternatives[1].consequence.after=(nt_spa_metrics){1,1,1,1,.5f,.5f,.5f};
    c[0].alternatives[2].consequence.after=(nt_spa_metrics){0,0,0,0,.5f,.5f,.5f};
    neutralize(&c[1]); neutralize(&c[2]);
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,3,0,1e-6f,&r)==NT_SPA_OK &&
        r.comparison.rewards[1]==(float)(1.0/3) && r.comparison.rewards[2]==(float)(-1.0/3) &&
        r.scale==(double)(float)(1.0/3) && r.comparison.targets[1]==1 && r.comparison.targets[2]==-1,
        "native clipping precedes double mean, one float rounding, then scaling");
    neutralize(&c[0]);
    c[0].alternatives[0].consequence.after.local_connectedness=.375f;
    c[0].alternatives[1].consequence.after.local_connectedness=.625f;
    c[0].alternatives[2].consequence.after.local_connectedness=.25f;
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,1,0,1e-6f,&r)==NT_SPA_OK && r.scale==.25 &&
        r.comparison.rewards[0]==-.125f && r.comparison.rewards[1]==.125f &&
        r.comparison.rewards[2]==-.25f && r.comparison.targets[0]==0 &&
        r.comparison.targets[1]==1 && r.comparison.targets[2]==-.5f,
        "paired KEEP subtraction precedes maximum absolute advantage scale");
    c[1]=c[0];
    for(i=0;i<3;++i) {
        float effect=c[0].alternatives[i].consequence.after.local_connectedness-.5f;
        c[1].alternatives[i].consequence.after.local_connectedness=.5f+effect*.5f;
    }
    a=initial; b=initial;
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,1,.03f,1e-6f,&r)==NT_SPA_OK &&
        nt_spa_agent_fit_conditioned(&b,&e,c+1,1,.03f,1e-6f,&scaled)==NT_SPA_OK &&
        scaled.scale*2==r.scale && SAME(a,b) && !memcmp(r.comparison.targets,scaled.comparison.targets,sizeof(r.comparison.targets)),
        "exact positive common effect rescaling preserves update above floor");
    return 1;
}

static int test_floor(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c;
    nt_spa_conditioned_receipt r;
    float tiny=nextafterf(0,1),subfloor=ldexpf(1,-20),floor=ldexpf(1,-16);
    CHECK(initialize(&a,1),"floor init"); before=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"floor capture");
    c=comparison(&e,0); neutralize(&c);
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,&c,1,.03f,1e-6f,&r)==NT_SPA_OK && SAME(a,before) &&
        r.scale==(double)1e-6f && r.scale_floor==1e-6f && r.comparison.targets[0]==0 &&
        r.comparison.targets[1]==0 && r.comparison.targets[2]==0,"equal rewards keep zero targets and explicit positive floor");
    c.alternatives[1].consequence.after.local_connectedness=.5f+subfloor;
    c.alternatives[2].consequence.after.local_connectedness=.5f-2*subfloor;
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,&c,1,0,floor,&r)==NT_SPA_OK && r.scale==floor &&
        r.comparison.targets[1]==.0625f && r.comparison.targets[2]==-.125f,
        "sub-floor measured effects retain their proportion to declared floor");
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,&c,1,0,tiny,&r)==NT_SPA_OK && r.scale==2*subfloor &&
        r.scale_floor==tiny && r.comparison.targets[1]==.5f && r.comparison.targets[2]==-1,
        "smallest positive floor accepted and largest absolute negative effect sets scale");
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,&c,1,0,FLT_MAX,&r)==NT_SPA_OK && r.scale==FLT_MAX &&
        isfinite(r.comparison.targets[1]) && isfinite(r.comparison.targets[2]),
        "largest finite positive floor accepted without invalid arithmetic");
    neutralize(&c);
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,&c,1,0,tiny,&r)==NT_SPA_OK && r.scale==tiny &&
        r.comparison.targets[0]==0 && r.comparison.targets[1]==0 && r.comparison.targets[2]==0,
        "tied outcomes with subnormal floor remain zero");
    return 1;
}

static float *parameter(nt_spa_policy *p,unsigned index) {
    if(index<232) return &p->w1[index/29][index%29];
    index-=232;
    if(index<8) return &p->b1[index];
    index-=8;
    if(index<24) return &p->w2[index/8][index%8];
    return &p->b2[index-24];
}
static double independent_loss(const nt_spa_policy *p,const float *features,const float *targets) {
    double hidden[8],result=0;
    unsigned i,j;
    for(i=0;i<8;++i) {
        double value=p->b1[i];
        for(j=0;j<29;++j) value+=(double)p->w1[i][j]*features[j];
        hidden[i]=tanh(value);
    }
    for(i=0;i<3;++i) {
        double score=p->b2[i],error;
        for(j=0;j<8;++j) score+=(double)p->w2[i][j]*hidden[j];
        error=fabs(score-targets[i]); result+=(error<=1 ? .5*error*error:error-.5)/3;
    }
    return result;
}
static int test_gradient(void) {
    nt_spa_agent a,next;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[3];
    nt_spa_conditioned_receipt update;
    unsigned i,last,regime,k,nonzero[267]={0},covered=0,total=0;
    double maximum_error=0;
    const float rate=.03125f,epsilon=.002f;
    CHECK(initialize(&a,0),"gradient init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"gradient capture");
    e.features[19]=.17f; e.features[20]=.42f; e.features[21]=.31f;
    e.features[22]=e.features[23]=e.features[24]=1.0f/3; e.features[28]=-.23f;
    for(i=0;i<3;++i) c[i]=comparison(&e,i);
    for(regime=0;regime<2;++regime) for(last=0;last<3;++last) {
        nt_spa_policy p;
        e.features[25]=e.features[26]=e.features[27]=0; e.features[25+last]=1;
        CHECK(nt_spa_experience_validate(&e)==NT_SPA_OK,"valid temporal-input fixture");
        memset(&p,0,sizeof(p));
        for(k=0;k<267;++k) *parameter(&p,k)=.011f*((int)((k*7u+3u)%17u)-8);
        for(i=0;i<8;++i) {
            p.w2[0][i]=.031f+.004f*i; p.w2[1][i]=-.057f+.003f*i; p.w2[2][i]=.087f-.002f*i;
        }
        p.b2[0]=regime ? 2:.13f; p.b2[1]=regime ? -2:.17f; p.b2[2]=regime ? 2:-.31f;
        CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"install gradient fixture"); next=a;
        CHECK(nt_spa_agent_fit_conditioned(&next,&e,c,3,rate,1e-6f,&update)==NT_SPA_OK,
            "conditioned derivative step");
        CHECK(update.comparison.loss_after<update.comparison.loss_before && only_policy(&a,&next),
            "derivative step lowers measured objective");
        CHECK((fabsf(update.comparison.scores_before[0]-update.comparison.targets[0])>1)==(regime!=0),
            "quadratic and linear Huber regimes exercised");
        CHECK(NEAR(update.comparison.loss_before,independent_loss(&a.policy,e.features,update.comparison.targets),2e-7),
            "independent double forward reproduces native mean loss");
        for(k=0;k<267;++k) {
            nt_spa_policy positive=a.policy,negative=a.policy;
            double numeric,observed,error;
            *parameter(&positive,k)+=epsilon; *parameter(&negative,k)-=epsilon;
            numeric=(independent_loss(&positive,e.features,update.comparison.targets)-
                independent_loss(&negative,e.features,update.comparison.targets))/
                ((double)*parameter(&positive,k)-*parameter(&negative,k));
            observed=((double)*parameter(&next.policy,k)-*parameter(&a.policy,k))/rate;
            error=fabs(numeric+observed); if(error>maximum_error) maximum_error=error;
            if(fabs(numeric)>1e-6) nonzero[k]=1;
            if(error>3e-5+.003*fabs(numeric)) fprintf(stderr,"gradient detail %u/%u/%u %.9g vs %.9g\n",regime,last,k,numeric,observed);
            CHECK(error<=3e-5+.003*fabs(numeric),"all267 simultaneous gradients match independent finite differences");
            ++total;
        }
    }
    for(k=0;k<267;++k) covered+=nonzero[k];
    CHECK(total==1602 && covered==267,"all267 parameters have nonzero independently checked gradients");
    printf("gradient: comparisons=%u nonzero_parameters=%u maximum_absolute_error=%.9g\n",total,covered,maximum_error);
    return 1;
}

static int test_persistence(void) {
    nt_spa_agent a,b,before,initial_policy;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[3];
    nt_spa_conditioned_receipt r1,r2;
    nt_spa_decision decision;
    nt_spa_readout initial,acquired;
    nt_spa_consequence consequence;
    char path[128];
    unsigned i;
    CHECK(initialize(&a,0),"persistence init");
    memset(&consequence,0,sizeof(consequence)); consequence.after.coherence=.5f;
    for(i=0;i<12;++i)
        CHECK(nt_spa_agent_choose(&a,&o,&decision)==NT_SPA_OK &&
            nt_spa_agent_observe(&a,decision.sequence,&decision.action,&consequence,NULL)==NT_SPA_OK,
            "populate real action/outcome ring and temporal statistics");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"capture complete temporal input");
    for(i=0;i<3;++i) c[i]=comparison(&e,i);
    before=a;
    CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,3,.03f,1e-6f,&r1)==NT_SPA_OK && only_policy(&before,&a),
        "conditioned credit preserves populated ring, EMA, RNG and counters");
    CHECK(nt_spa_agent_score_experience(&before,&e,&initial)==NT_SPA_OK &&
        nt_spa_agent_score_experience(&a,&e,&acquired)==NT_SPA_OK &&
        initial.action.kind==NT_SPA_KEEP && acquired.action.kind==NT_SPA_RESEED_LEFT,
        "identical observation/history/RNG select a different action under acquired weights");
    initial_policy=a;
    CHECK(nt_spa_agent_set_policy(&initial_policy,&before.policy)==NT_SPA_OK && SAME(initial_policy,before),
        "reset acquired weights alone restores entire original life");
    (void)snprintf(path,sizeof(path),"/tmp/notorch-spa-conditioned-%ld.bin",(long)getpid());
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK && nt_spa_agent_load(&b,path)==NT_SPA_OK && SAME(a,b),
        "canonical v1 save/resume preserves acquired conditioned policy");
    for(i=0;i<16;++i)
        CHECK(nt_spa_agent_fit_conditioned(&a,&e,c,3,.03f,1e-6f,&r1)==NT_SPA_OK &&
            nt_spa_agent_fit_conditioned(&b,&e,c,3,.03f,1e-6f,&r2)==NT_SPA_OK && SAME(a,b) && SAME(r1,r2),
            "resumed continuation and complete receipts remain byte-identical");
    CHECK(only_policy(&before,&a) && nt_spa_agent_validate(&a)==NT_SPA_OK,"continued replay preserves every v1 chronology field");
    CHECK(unlink(path)==0,"remove temporary life");
    return 1;
}

int main(int argc,char **argv) {
    static const struct { const char *name; int (*run)(void); } groups[]={
        {"parity",test_parity},{"validation",test_validation},{"aggregate",test_aggregate},
        {"floor",test_floor},{"gradient",test_gradient},{"persistence",test_persistence}};
    unsigned i,ran=0;
    if(argc>2) return 2;
    for(i=0;i<sizeof(groups)/sizeof(groups[0]);++i) {
        if(argc==2 && strcmp(argv[1],groups[i].name)) continue;
        gate=groups[i].name; if(!groups[i].run()) return 1;
        printf("PASS %s\n",gate); ++ran;
    }
    if(!ran) return 2;
    printf("PASS SPA conditioned: %u groups, %u checks\n",ran,checks); return 0;
}
