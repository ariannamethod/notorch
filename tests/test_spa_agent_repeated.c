/* Same-state repeated consequences, one native policy update. */
#include "spa_agent.h"
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

static int initialize(nt_spa_agent *a,int clip_fixture) {
    nt_spa_agent_config c;
    nt_spa_agent_config_default(&c); c.mode=NT_SPA_AGENT_LEARNED;
    c.exploration=0; c.learning_rate=0;
    if(clip_fixture) {
        c.reward_weights=(nt_spa_metrics){1,1,1,1,0,0,0}; c.cost_weight=0;
    }
    return nt_spa_agent_init(a,&c)==NT_SPA_OK;
}
static nt_spa_observation observation(unsigned target,unsigned count) {
    nt_spa_observation o;
    memset(&o,0,sizeof(o));
    o.embedding[0]=.25f; o.embedding[1]=-.5f; o.embedding[2]=.75f; o.embedding[3]=-.125f;
    o.connectedness=.4f; o.left_similarity=.2f; o.right_similarity=.8f;
    o.coherence=.5f; o.novelty=.3f; o.repetition=.1f; o.phase_lock=.5f;
    o.sentence_score=.7f; o.mean_sentence_score=1; o.temperature=.8f;
    o.sentence_index=target; o.sentence_count=count; o.reseed_count=2;
    return o;
}
static nt_spa_comparison comparison(const nt_spa_experience *e,unsigned horizon,unsigned repetition) {
    nt_spa_comparison c;
    unsigned i;
    memset(&c,0,sizeof(c)); c.source_life_hash=e->source_life_hash; c.horizon=horizon;
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
static int only_policy(const nt_spa_agent *before,const nt_spa_agent *after) {
    nt_spa_agent expected=*before; expected.policy=after->policy;
    return !memcmp(&expected,after,sizeof(expected));
}
static int test_parity(void) {
    static const unsigned targets[]={0,1,3,0},counts[]={4,3,4,1};
    static const float rates[]={0,.03125f,1};
    unsigned k,rate,i;
    for(k=0;k<4;++k) {
        nt_spa_agent initial,a,b;
        nt_spa_observation o=observation(targets[k],counts[k]);
        nt_spa_experience e;
        nt_spa_comparison c,repeated[NT_SPA_COMPARISON_MAX_REPEATS];
        nt_spa_comparison_receipt old,one,many;
        CHECK(initialize(&initial,0),"parity init");
        {
            nt_spa_policy p=initial.policy;
            for(i=0;i<3;++i) { p.w2[i][0]=-0.0f; p.b2[i]=-0.0f; }
            CHECK(nt_spa_agent_set_policy(&initial,&p)==NT_SPA_OK,"signed-zero action head fixture");
        }
        CHECK(nt_spa_agent_capture_experience(&initial,&o,&e)==NT_SPA_OK,"parity capture");
        c=comparison(&e,4096,0);
        for(i=0;i<NT_SPA_COMPARISON_MAX_REPEATS;++i) repeated[i]=c;
        for(rate=0;rate<3;++rate) {
            a=initial; b=initial;
            CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,rates[rate],&old)==NT_SPA_OK,"old comparison fit");
            CHECK(nt_spa_agent_fit_repeated(&b,&e,&c,1,rates[rate],&one)==NT_SPA_OK,"single repeated fit");
            CHECK(SAME(a,b) && SAME(old,one),"count1 preserves exact old life and receipt bytes");
            b=initial;
            CHECK(nt_spa_agent_fit_repeated(&b,&e,repeated,64,rates[rate],&many)==NT_SPA_OK,"maximum64 repetitions accepted");
            CHECK(SAME(a,b) && SAME(old,many),"64 identical outcomes equal one update exactly");
            CHECK(only_policy(&initial,&a),"comparison replay changes policy only");
            if(rate==0) CHECK(SAME(a,initial),"zero rate leaves whole state unchanged");
        }
        a=initial; b=initial;
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,.03f,NULL)==NT_SPA_OK &&
            nt_spa_agent_fit_repeated(&b,&e,&c,1,.03f,NULL)==NT_SPA_OK && SAME(a,b),
            "optional receipt preserves count1 parity");
    }
    return 1;
}

static int refused(nt_spa_agent *a,const nt_spa_experience *e,nt_spa_comparison *c,uint32_t count,float rate) {
    nt_spa_agent before=*a;
    nt_spa_experience saved=*e;
    nt_spa_comparison originals[NT_SPA_COMPARISON_MAX_REPEATS];
    nt_spa_comparison_receipt r,sentinel;
    unsigned extent=count<=NT_SPA_COMPARISON_MAX_REPEATS ? count : NT_SPA_COMPARISON_MAX_REPEATS;
    memcpy(originals,c,extent*sizeof(*c)); memset(&sentinel,0xa5,sizeof(sentinel)); r=sentinel;
    CHECK(nt_spa_agent_fit_repeated(a,e,c,count,rate,&r)!=NT_SPA_OK,"malformed repeated fit refused");
    CHECK(SAME(*a,before) && SAME(*e,saved) && SAME(r,sentinel) &&
        !memcmp(c,originals,extent*sizeof(*c)),"all-or-nothing refusal preserves every byte");
    return 1;
}
static int test_validation(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e,bad_e;
    nt_spa_comparison good[64],bad[64];
    nt_spa_comparison_receipt r;
    nt_spa_decision d;
    unsigned i;
    CHECK(initialize(&a,0),"validation init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"validation capture");
    for(i=0;i<64;++i) good[i]=comparison(&e,4,i%3);
    before=a;
    CHECK(refused(&a,&e,good,0,.03f),"zero repetitions refused");
    CHECK(refused(&a,&e,good,65,.03f),"too many repetitions refused");
    CHECK(refused(&a,&e,good,UINT32_MAX,.03f),"huge count refused before multiplication/access");
    CHECK(refused(&a,&e,good,2,-.1f) && refused(&a,&e,good,2,1.1f) &&
        refused(&a,&e,good,2,NAN),"invalid rates refused");
    CHECK(nt_spa_agent_fit_repeated(&a,&e,NULL,2,.03f,&r)!=NT_SPA_OK && SAME(a,before),"null array refused");
    CHECK(nt_spa_agent_fit_repeated(&a,NULL,good,2,.03f,&r)!=NT_SPA_OK && SAME(a,before),"null input refused");
    CHECK(nt_spa_agent_fit_repeated(NULL,&e,good,2,.03f,&r)!=NT_SPA_OK,"null agent refused");
    bad_e=e; bad_e.features[0]=NAN;
    CHECK(refused(&a,&bad_e,good,2,.03f),"malformed captured input refused");
#define BAD_LAST(field,value) do { memcpy(bad,good,sizeof(bad)); bad[63].field=(value); \
    CHECK(refused(&a,&e,bad,64,.03f),"invalid last comparison " #field); } while(0)
    BAD_LAST(source_life_hash,e.source_life_hash^UINT64_C(1));
    BAD_LAST(action_mask,3);
    BAD_LAST(horizon,5);
    BAD_LAST(horizon,NT_SPA_COMPARISON_MAX_HORIZON+1);
    BAD_LAST(alternatives[0].action.source,0);
    BAD_LAST(alternatives[1].action.kind,NT_SPA_RESEED_RIGHT);
    BAD_LAST(alternatives[2].action.source,0);
    BAD_LAST(alternatives[2].action.target,0);
    BAD_LAST(alternatives[2].consequence.after.novelty,NAN);
    BAD_LAST(alternatives[2].consequence.after.novelty,1.01f);
    BAD_LAST(alternatives[2].consequence.regeneration_cost,-.01f);
    BAD_LAST(alternatives[2].consequence.regeneration_cost,INFINITY);
    BAD_LAST(alternatives[0].consequence.before.coherence,.7f);
    BAD_LAST(alternatives[2].consequence.before.coherence,.7f);
#undef BAD_LAST
    memcpy(bad,good,sizeof(bad));
    for(i=0;i<3;++i) bad[63].alternatives[i].consequence.before.novelty=.2f;
    CHECK(refused(&a,&e,bad,64,.03f),"internally shared before axes must also match other repetitions");
    CHECK(nt_spa_agent_fit_repeated(&a,&e,good,64,.03f,(nt_spa_comparison_receipt*)&a)!=NT_SPA_OK &&
        SAME(a,before),"output cannot overlap agent");
    bad_e=e;
    CHECK(nt_spa_agent_fit_repeated(&a,&bad_e,good,64,.03f,(nt_spa_comparison_receipt*)&bad_e)!=NT_SPA_OK &&
        SAME(a,before) && SAME(bad_e,e),"output cannot overlap captured experience");
    memcpy(bad,good,sizeof(bad));
    CHECK(nt_spa_agent_fit_repeated(&a,&e,bad,64,.03f,(nt_spa_comparison_receipt*)&bad[63])!=NT_SPA_OK &&
        SAME(a,before) && !memcmp(bad,good,sizeof(bad)),"output overlap checked across entire repeated array");
    /* A valid experience embedded in a comparison-array allocation must still be refused. */
    {
        union { nt_spa_comparison comparisons[2]; nt_spa_experience experience; } overlap,saved;
        memset(&overlap,0,sizeof(overlap)); overlap.experience=e; saved=overlap;
        CHECK(nt_spa_agent_fit_repeated(&a,&overlap.experience,overlap.comparisons,2,.03f,&r)!=NT_SPA_OK &&
            SAME(a,before) && SAME(overlap,saved),"input arrays cannot overlap each other");
    }
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"create pending credit");
    CHECK(refused(&a,&e,good,2,.03f),"pending replay refused");
    CHECK(nt_spa_agent_cancel(&a,d.sequence)==NT_SPA_OK,"cancel pending action");
    a.version=0;
    CHECK(refused(&a,&e,good,2,.03f),"malformed persistent state refused");
    return 1;
}

static int test_aggregate(void) {
    nt_spa_agent a,initial,b;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[2],saved[2];
    nt_spa_comparison_receipt r,single[2];
    nt_spa_metrics middle={.5f,.5f,.5f,.5f,.5f,.5f,.5f};
    unsigned repetition,kind;
    CHECK(initialize(&a,1),"clipping fixture init"); initial=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"clipping fixture capture");
    for(repetition=0;repetition<2;++repetition) {
        c[repetition]=comparison(&e,4,0);
        for(kind=0;kind<3;++kind) {
            c[repetition].alternatives[kind].consequence.before=middle;
            c[repetition].alternatives[kind].consequence.after=middle;
            c[repetition].alternatives[kind].consequence.regeneration_cost=0;
        }
    }
    c[0].alternatives[1].consequence.after=(nt_spa_metrics){1,1,1,1,.5f,.5f,.5f};
    c[0].alternatives[2].consequence.after=(nt_spa_metrics){0,0,0,0,.5f,.5f,.5f};
    memcpy(saved,c,sizeof(saved));
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&c[0],0,&single[0])==NT_SPA_OK &&
        nt_spa_agent_fit_comparison(&a,&e,&c[1],0,&single[1])==NT_SPA_OK,"native per-draw rewards");
    CHECK(single[0].rewards[1]==1 && single[0].rewards[2]==-1 && single[1].rewards[1]==0,
        "native reward clips both raw plus2 and minus2");
    CHECK(nt_spa_agent_fit_repeated(&a,&e,c,2,0,&r)==NT_SPA_OK && SAME(a,initial),"zero-rate repeated aggregation");
    CHECK(r.rewards[0]==0 && r.rewards[1]==.5f && r.rewards[2]==-.5f,
        "clip before average yields plus/minus half, not plus/minus one");
    CHECK(r.targets[0]==0 && r.targets[1]==.5f && r.targets[2]==-.5f,
        "positive and negative comparison targets retain their signs");
    CHECK(NEAR(r.loss_before,1.0/12.0,1e-10),"mean Huber over three heads uses averaged targets");
    CHECK(nt_spa_agent_fit_repeated(&a,&e,c,2,.12f,&r)==NT_SPA_OK,"single simultaneous repeated update");
    CHECK(NEAR(a.policy.b2[1],.02,2e-8) && NEAR(a.policy.b2[2],-.02,2e-8),
        "one update credits positive and negative mean consequences exactly once");
    CHECK(r.loss_after<r.loss_before && only_policy(&initial,&a) && !memcmp(c,saved,sizeof(c)),
        "repeated fit improves loss and changes only policy");
    b=initial;
    CHECK(nt_spa_agent_fit_comparison(&b,&e,&c[0],.12f,NULL)==NT_SPA_OK &&
        nt_spa_agent_fit_comparison(&b,&e,&c[1],.12f,NULL)==NT_SPA_OK && !SAME(a.policy,b.policy),
        "repeated fit is one mean-target step, distinct from sequential draw updates");
    c[0].alternatives[0].consequence.after=(nt_spa_metrics){0,0,0,0,.5f,.5f,.5f};
    a=initial;
    CHECK(nt_spa_agent_fit_repeated(&a,&e,c,2,0,&r)==NT_SPA_OK &&
        r.rewards[0]==-.5f && r.rewards[1]==.5f && r.targets[1]==1 && r.targets[2]==0,
        "subtract separately averaged KEEP reward after native clipping");
    printf("aggregate: native rewards [+1,0]/[-1,0] -> means +0.5/-0.5; one update\n");
    return 1;
}

static float *parameter(nt_spa_policy *p,unsigned index) {
    if(index<232) return &p->w1[index/29][index%29];
    index-=232;
    if(index<8) return &p->b1[index];
    index-=8;
    if(index<24) return &p->w2[index/8][index%8];
    index-=24;
    return &p->b2[index];
}
static int test_gradient(void) {
    nt_spa_agent a,next,positive,negative;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[3];
    nt_spa_comparison_receipt update,hi,lo;
    unsigned i,last,regime,k,nonzero[267]={0},covered=0,total=0;
    double maximum_error=0;
    const float rate=.03125f,epsilon=.002f;
    CHECK(initialize(&a,0),"gradient init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"gradient capture");
    e.features[19]=.17f; e.features[20]=.42f; e.features[21]=.31f;
    e.features[22]=e.features[23]=e.features[24]=1.0f/3; e.features[28]=-.23f;
    for(i=0;i<3;++i) c[i]=comparison(&e,4,i);
    for(regime=0;regime<2;++regime) for(last=0;last<3;++last) {
        nt_spa_policy p;
        e.features[25]=e.features[26]=e.features[27]=0; e.features[25+last]=1;
        CHECK(nt_spa_experience_validate(&e)==NT_SPA_OK,"valid last-action fixture");
        memset(&p,0,sizeof(p));
        for(k=0;k<267;++k) *parameter(&p,k)=.011f*((int)((k*7u+3u)%17u)-8);
        for(i=0;i<8;++i) {
            p.w2[0][i]=.031f+.004f*i; p.w2[1][i]=-.057f+.003f*i; p.w2[2][i]=.087f-.002f*i;
        }
        p.b2[0]=regime ? 2:.13f; p.b2[1]=regime ? -2:-.17f; p.b2[2]=.31f;
        CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"install derivative fixture"); next=a;
        CHECK(nt_spa_agent_fit_repeated(&next,&e,c,3,rate,&update)==NT_SPA_OK,"repeated derivative step");
        CHECK(update.loss_after<update.loss_before && only_policy(&a,&next),"gradient step lowers mean-target loss");
        CHECK((fabsf(update.scores_before[0]-update.targets[0])>1)==(regime!=0),"both Huber regimes exercised");
        for(k=0;k<267;++k) {
            double numeric,observed,error;
            positive=a; negative=a;
            *parameter(&positive.policy,k)+=epsilon; *parameter(&negative.policy,k)-=epsilon;
            CHECK(nt_spa_agent_fit_repeated(&positive,&e,c,3,0,&hi)==NT_SPA_OK &&
                nt_spa_agent_fit_repeated(&negative,&e,c,3,0,&lo)==NT_SPA_OK,"finite differences");
            numeric=(hi.loss_before-lo.loss_before)/
                ((double)*parameter(&positive.policy,k)-*parameter(&negative.policy,k));
            observed=((double)*parameter(&next.policy,k)-*parameter(&a.policy,k))/rate;
            error=fabs(numeric+observed); if(error>maximum_error) maximum_error=error;
            if(fabs(numeric)>1e-6) nonzero[k]=1;
            if(error>3e-5+.003*fabs(numeric)) fprintf(stderr,"gradient detail %u/%u/%u: %.9g vs %.9g\n",regime,last,k,numeric,observed);
            CHECK(error<=3e-5+.003*fabs(numeric),"all267 simultaneous gradients match finite differences");
            ++total;
        }
    }
    for(k=0;k<267;++k) covered+=nonzero[k];
    CHECK(total==1602 && covered==267,"all267 gradients nonzero across three history and two Huber fixtures");
    printf("gradient: comparisons=%u nonzero_parameters=%u maximum_absolute_error=%.9g\n",total,covered,maximum_error);
    return 1;
}

static int test_persistence(void) {
    nt_spa_agent a,b,before;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c[3];
    nt_spa_comparison_receipt r1,r2;
    nt_spa_decision decision;
    nt_spa_consequence consequence;
    char path[128];
    unsigned i;
    CHECK(initialize(&a,0),"persistence init");
    memset(&consequence,0,sizeof(consequence)); consequence.after.coherence=.5f;
    for(i=0;i<12;++i) {
        CHECK(nt_spa_agent_choose(&a,&o,&decision)==NT_SPA_OK &&
            nt_spa_agent_observe(&a,decision.sequence,&decision.action,&consequence,NULL)==NT_SPA_OK,
            "create temporal history before replay");
    }
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"capture complete temporal input");
    for(i=0;i<3;++i) c[i]=comparison(&e,4,i);
    before=a;
    CHECK(nt_spa_agent_fit_repeated(&a,&e,c,3,.03f,&r1)==NT_SPA_OK && only_policy(&before,&a),
        "repeated update preserves populated ring, EMA, RNG and all counters");
    (void)snprintf(path,sizeof(path),"/tmp/notorch-spa-repeated-%ld.bin",(long)getpid());
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK && nt_spa_agent_load(&b,path)==NT_SPA_OK && SAME(a,b),
        "existing canonical save/resume retains acquired repeated policy exactly");
    for(i=0;i<16;++i) {
        CHECK(nt_spa_agent_fit_repeated(&a,&e,c,3,.03f,&r1)==NT_SPA_OK &&
            nt_spa_agent_fit_repeated(&b,&e,c,3,.03f,&r2)==NT_SPA_OK && SAME(a,b) && SAME(r1,r2),
            "resumed repeated continuation and receipts are byte-identical");
    }
    CHECK(only_policy(&before,&a) && nt_spa_agent_validate(&a)==NT_SPA_OK,"continued replay preserves v1 life semantics");
    CHECK(unlink(path)==0,"remove temporary life");
    return 1;
}

int main(int argc,char **argv) {
    static const struct { const char *name; int (*run)(void); } groups[]={
        {"parity",test_parity},{"validation",test_validation},{"aggregate",test_aggregate},
        {"gradient",test_gradient},{"persistence",test_persistence}};
    unsigned i,ran=0;
    if(argc>2) return 2;
    for(i=0;i<sizeof(groups)/sizeof(groups[0]);++i) {
        if(argc==2 && strcmp(argv[1],groups[i].name)) continue;
        gate=groups[i].name; if(!groups[i].run()) return 1;
        printf("PASS %s\n",gate); ++ran;
    }
    if(!ran) return 2;
    printf("PASS SPA repeated: %u groups, %u checks\n",ran,checks); return 0;
}
