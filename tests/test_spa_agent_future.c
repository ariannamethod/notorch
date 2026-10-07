/* Native comparison credit: captured inputs, raw outcomes, policy-only SGD. */
#include "spa_agent.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *gate;
static unsigned checks;
#define CHECK(c,msg) do { ++checks; if(!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",gate,__LINE__,msg); return 0; } } while(0)
#define SAME(a,b) (memcmp(&(a),&(b),sizeof(a))==0)
#define NEAR(a,b,t) (isfinite(a) && fabs((double)(a)-(double)(b))<(t))

static int initialized(nt_spa_agent *a,float online_rate) {
    nt_spa_agent_config config;
    nt_spa_agent_config_default(&config);
    config.mode=NT_SPA_AGENT_LEARNED; config.exploration=0;
    config.learning_rate=online_rate;
    return nt_spa_agent_init(a,&config)==NT_SPA_OK;
}
static nt_spa_observation observation(uint32_t target,uint32_t count) {
    nt_spa_observation o;
    memset(&o,0,sizeof(o));
    o.embedding[0]=.25f; o.embedding[1]=-.5f;
    o.embedding[2]=.75f; o.embedding[3]=-.125f;
    o.connectedness=.4f; o.left_similarity=.2f; o.right_similarity=.8f;
    o.coherence=.5f; o.novelty=.3f; o.repetition=.1f; o.phase_lock=.5f;
    o.sentence_score=.7f; o.mean_sentence_score=1;
    o.temperature=.8f; o.sentence_index=target; o.sentence_count=count;
    o.reseed_count=2;
    return o;
}
static uint32_t action_mask(const nt_spa_experience *e) {
    return 1u | (e->sentence_index>0 ? 2u : 0u) |
        (e->sentence_index+1<e->sentence_count ? 4u : 0u);
}
static nt_spa_action action(unsigned kind,uint32_t target) {
    nt_spa_action a;
    memset(&a,0,sizeof(a)); a.kind=(nt_spa_action_kind)kind; a.target=target;
    a.source=kind==0 ? NT_SPA_AGENT_NO_SOURCE : kind==1 ? target-1 : target+1;
    return a;
}
static nt_spa_consequence consequence(void) {
    nt_spa_consequence c;
    memset(&c,0,sizeof(c));
    c.before=(nt_spa_metrics){.4f,.3f,.5f,.4f,.2f,.1f,.6f};
    c.after=(nt_spa_metrics){.6f,.5f,.7f,.6f,.1f,.05f,.7f};
    c.regeneration_cost=.25f;
    return c;
}
static nt_spa_comparison comparison(const nt_spa_experience *e,uint32_t horizon) {
    nt_spa_comparison c;
    nt_spa_consequence result=consequence();
    unsigned i;
    memset(&c,0,sizeof(c)); c.source_life_hash=e->source_life_hash;
    c.action_mask=action_mask(e); c.horizon=horizon;
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(c.action_mask&(1u<<i)) {
        c.alternatives[i].action=action(i,e->sentence_index);
        c.alternatives[i].consequence=result;
    }
    c.alternatives[0].consequence.after=result.before;
    c.alternatives[0].consequence.regeneration_cost=0;
    if(c.action_mask&4u) {
        c.alternatives[2].consequence.after.coherence=.35f;
        c.alternatives[2].consequence.after.repetition=.5f;
        c.alternatives[2].consequence.after.novelty=.3f;
        c.alternatives[2].consequence.regeneration_cost=.75f;
    }
    return c;
}
static int only_policy_changed(const nt_spa_agent *before,const nt_spa_agent *after) {
    nt_spa_agent expected=*before;
    expected.policy=after->policy;
    return memcmp(&expected,after,sizeof(expected))==0;
}

static int test_capture(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e,original,recent;
    nt_spa_decision d;
    nt_spa_readout r,read_before;
    nt_spa_consequence c=consequence();
    unsigned i;
    CHECK(initialized(&a,0),"init learned frozen online policy");
    before=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"capture idle life");
    original=e;
    CHECK(SAME(a,before),"capture preserves complete source life");
    CHECK(e.version==1 && e.sentence_index==1 && e.sentence_count==3 &&
        e.source_life_hash==nt_spa_agent_hash(&a),"capture coordinates and canonical source hash");
    CHECK(nt_spa_experience_validate(&e)==NT_SPA_OK,"captured input validates");
    CHECK(nt_spa_agent_select(&a,&o,&d)==NT_SPA_OK,"ordinary pure selection");
    CHECK(!memcmp(e.features,d.features,sizeof(e.features)),"capture uses exact v1 policy inputs");
    CHECK(nt_spa_agent_score_experience(&a,&e,&r)==NT_SPA_OK,"score captured input");
    CHECK(SAME(r.action,d.action) && !memcmp(r.scores,d.scores,sizeof(r.scores)),
        "captured readout equals ordinary zero-exploration policy");
    CHECK(r.action_mask==7 && r.action.kind==NT_SPA_KEEP,"initial valid KEEP-first tie");
    CHECK(SAME(a,before) && SAME(e,original),"pure readout preserves life and input");
    read_before=r;
    for(i=0;i<17;++i) {
        CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"choose for temporal ring");
        CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,NULL)==NT_SPA_OK,
            "record real action outcome without online learning");
        CHECK(nt_spa_agent_capture_experience(&a,&o,&recent)==NT_SPA_OK,
            "capture validates every ring length and rollover");
        CHECK(recent.features[22]==1 && recent.features[25]==1,
            "captured action frequencies and latest action follow recorded KEEP history");
    }
    CHECK(a.observations==17 && a.history_count==8 && a.memory_observations==17,
        "live temporal memory advanced");
    CHECK(!SAME(a,before) && SAME(a.policy,before.policy),"history changes while weights stay fixed");
    CHECK(nt_spa_agent_score_experience(&a,&e,&r)==NT_SPA_OK && SAME(r,read_before),
        "captured readout uses captured history instead of current live history");
    before=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&recent)==NT_SPA_OK &&
        recent.source_life_hash!=e.source_life_hash,"updated source life identity");
    CHECK(SAME(a,before) && SAME(e,original),"later capture preserves original experience");
    return 1;
}

static int refused(nt_spa_agent *a,const nt_spa_experience *e,const nt_spa_comparison *c,
    float rate) {
    nt_spa_agent before=*a;
    nt_spa_experience saved_e=*e;
    nt_spa_comparison saved_c=*c;
    nt_spa_comparison_receipt r,sentinel;
    memset(&sentinel,0xa5,sizeof(sentinel)); r=sentinel;
    CHECK(nt_spa_agent_fit_comparison(a,e,c,rate,&r)!=NT_SPA_OK,"malformed fit refused");
    CHECK(SAME(*a,before) && SAME(*e,saved_e) && SAME(*c,saved_c) && SAME(r,sentinel),
        "refusal preserves every state/input/output byte");
    return 1;
}
static int test_validation(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation(1,3),bad_o;
    nt_spa_experience good,e,sentinel;
    nt_spa_comparison valid,c;
    nt_spa_comparison_receipt receipt;
    nt_spa_readout read,read_sentinel;
    nt_spa_decision decision;
    nt_spa_agent_config config;
    CHECK(initialized(&a,0),"init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&good)==NT_SPA_OK,"valid experience");
    valid=comparison(&good,4); before=a;
#define BAD_EXPERIENCE(field,value) do { e=good; e.field=(value); \
    CHECK(nt_spa_experience_validate(&e)==NT_SPA_E_EXPERIENCE,"invalid experience " #field); \
    CHECK(refused(&a,&e,&valid,.03f),"refused experience " #field); } while(0)
    BAD_EXPERIENCE(version,0);
    BAD_EXPERIENCE(source_life_hash,0);
    BAD_EXPERIENCE(sentence_count,0);
    BAD_EXPERIENCE(sentence_count,NT_SPA_AGENT_MAX_SENTENCES+1);
    BAD_EXPERIENCE(sentence_index,3);
    BAD_EXPERIENCE(features[0],NAN);
    BAD_EXPERIENCE(features[3],1.01f);
    BAD_EXPERIENCE(features[4],-.1f);
    BAD_EXPERIENCE(features[7],INFINITY);
    BAD_EXPERIENCE(features[12],.75f);
    BAD_EXPERIENCE(features[13],1);
    BAD_EXPERIENCE(features[15],0);
    BAD_EXPERIENCE(features[16],0);
    BAD_EXPERIENCE(features[17],1);
    BAD_EXPERIENCE(features[18],1);
    BAD_EXPERIENCE(features[19],.1f);
    BAD_EXPERIENCE(features[20],.1f);
    BAD_EXPERIENCE(features[21],.1f);
    BAD_EXPERIENCE(features[22],.2f);
    BAD_EXPERIENCE(features[25],.5f);
    BAD_EXPERIENCE(features[25],1);
    BAD_EXPERIENCE(features[28],.1f);
#undef BAD_EXPERIENCE
    e=good; e.features[22]=.1f; e.features[23]=.9f; e.features[25]=1;
    CHECK(refused(&a,&e,&valid,.03f),"frequency cannot arise from one-to-eight ring");
    e=good; e.features[22]=1; e.features[25]=1; e.features[26]=1;
    CHECK(refused(&a,&e,&valid,.03f),"multiple last actions refused");
#define BAD_COMPARISON(field,value) do { c=valid; c.field=(value); \
    CHECK(refused(&a,&good,&c,.03f),"invalid comparison " #field); } while(0)
    BAD_COMPARISON(source_life_hash,0);
    BAD_COMPARISON(source_life_hash,good.source_life_hash^UINT64_C(1));
    BAD_COMPARISON(action_mask,0);
    BAD_COMPARISON(action_mask,3);
    BAD_COMPARISON(action_mask,15);
    BAD_COMPARISON(horizon,NT_SPA_COMPARISON_MAX_HORIZON+1);
    BAD_COMPARISON(alternatives[1].action.kind,NT_SPA_RESEED_RIGHT);
    BAD_COMPARISON(alternatives[1].action.target,0);
    BAD_COMPARISON(alternatives[1].action.source,2);
    BAD_COMPARISON(alternatives[0].action.source,0);
    BAD_COMPARISON(alternatives[2].consequence.before.coherence,.75f);
    BAD_COMPARISON(alternatives[2].consequence.after.coherence,NAN);
    BAD_COMPARISON(alternatives[1].consequence.after.novelty,1.01f);
    BAD_COMPARISON(alternatives[0].consequence.before.repetition,-.01f);
    BAD_COMPARISON(alternatives[1].consequence.regeneration_cost,1.01f);
    BAD_COMPARISON(alternatives[1].consequence.regeneration_cost,NAN);
#undef BAD_COMPARISON
    CHECK(refused(&a,&good,&valid,-.01f),"negative rate refused");
    CHECK(refused(&a,&good,&valid,1.01f),"rate above one refused");
    CHECK(refused(&a,&good,&valid,NAN),"nonfinite rate refused");
    CHECK(nt_spa_agent_fit_comparison(&a,&good,&valid,1,NULL)==NT_SPA_OK,"rate one accepted");
    a=before; c=valid; c.horizon=NT_SPA_COMPARISON_MAX_HORIZON;
    CHECK(nt_spa_agent_fit_comparison(&a,&good,&c,0,&receipt)==NT_SPA_OK && SAME(a,before),
        "generic maximum horizon and zero rate accepted");
    memset(&sentinel,0x5a,sizeof(sentinel)); e=sentinel;
    bad_o=o; bad_o.connectedness=NAN;
    CHECK(nt_spa_agent_capture_experience(&a,&bad_o,&e)==NT_SPA_E_OBSERVATION && SAME(e,sentinel),
        "invalid capture output unchanged");
    CHECK(nt_spa_agent_capture_experience(&a,&o,NULL)!=NT_SPA_OK,"capture requires output");
    CHECK(nt_spa_agent_capture_experience(&a,&o,(nt_spa_experience*)&a)!=NT_SPA_OK && SAME(a,before),
        "capture rejects output alias to life");
    {
        union { nt_spa_observation observation; nt_spa_experience experience; } alias,saved;
        memset(&alias,0,sizeof(alias)); alias.observation=o; saved=alias;
        CHECK(nt_spa_agent_capture_experience(&a,&alias.observation,&alias.experience)!=NT_SPA_OK &&
            SAME(alias,saved),"capture rejects observation/output overlap");
    }
    memset(&read_sentinel,0x5a,sizeof(read_sentinel)); read=read_sentinel;
    e=good; e.features[0]=NAN;
    CHECK(nt_spa_agent_score_experience(&a,&e,&read)!=NT_SPA_OK && SAME(read,read_sentinel),
        "malformed readout output unchanged");
    CHECK(nt_spa_agent_score_experience(&a,&good,NULL)!=NT_SPA_OK,"readout requires output");
    CHECK(nt_spa_agent_score_experience(&a,&good,(nt_spa_readout*)&a)!=NT_SPA_OK && SAME(a,before),
        "readout rejects output alias to life");
    e=good;
    CHECK(nt_spa_agent_score_experience(&a,&e,(nt_spa_readout*)&e)!=NT_SPA_OK && SAME(e,good),
        "readout rejects output alias to experience");
    CHECK(nt_spa_agent_fit_comparison(&a,&good,&valid,.03f,(nt_spa_comparison_receipt*)&a)!=NT_SPA_OK &&
        SAME(a,before),"fit rejects output alias to life");
    e=good;
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&valid,.03f,(nt_spa_comparison_receipt*)&e)!=NT_SPA_OK &&
        SAME(e,good) && SAME(a,before),"fit rejects output alias to experience");
    c=valid;
    CHECK(nt_spa_agent_fit_comparison(&a,&good,&c,.03f,(nt_spa_comparison_receipt*)&c)!=NT_SPA_OK &&
        SAME(c,valid) && SAME(a,before),"fit rejects output alias to comparison");
    CHECK(nt_spa_agent_fit_comparison(&a,NULL,&valid,.03f,&receipt)!=NT_SPA_OK && SAME(a,before),
        "null experience refused");
    CHECK(nt_spa_agent_fit_comparison(&a,&good,NULL,.03f,&receipt)!=NT_SPA_OK && SAME(a,before),
        "null comparison refused");
    CHECK(nt_spa_agent_score_experience(NULL,&good,&read)!=NT_SPA_OK,"null life refused");
    CHECK(nt_spa_agent_choose(&a,&o,&decision)==NT_SPA_OK,"create pending decision");
    before=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_E_PENDING,"pending capture refused");
    CHECK(nt_spa_agent_score_experience(&a,&good,&read)==NT_SPA_E_PENDING,"pending readout refused");
    CHECK(refused(&a,&good,&valid,.03f) && SAME(a,before),"pending replay refused transactionally");
    CHECK(nt_spa_agent_cancel(&a,decision.sequence)==NT_SPA_OK,"cancel pending action");
    nt_spa_agent_config_default(&config);
    CHECK(nt_spa_agent_init(&a,&config)==NT_SPA_OK,"legacy init");
    before=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_E_STATE,"legacy capture refused");
    CHECK(nt_spa_agent_score_experience(&a,&good,&read)==NT_SPA_E_STATE,"legacy readout refused");
    CHECK(refused(&a,&good,&valid,.03f) && SAME(a,before),"legacy replay refused");
    return 1;
}

static float *parameter(nt_spa_policy *p,unsigned index) {
    if(index<NT_SPA_AGENT_HIDDEN*NT_SPA_AGENT_FEATURES)
        return &p->w1[index/NT_SPA_AGENT_FEATURES][index%NT_SPA_AGENT_FEATURES];
    index-=NT_SPA_AGENT_HIDDEN*NT_SPA_AGENT_FEATURES;
    if(index<NT_SPA_AGENT_HIDDEN) return &p->b1[index];
    index-=NT_SPA_AGENT_HIDDEN;
    if(index<NT_SPA_AGENT_ACTIONS*NT_SPA_AGENT_HIDDEN)
        return &p->w2[index/NT_SPA_AGENT_HIDDEN][index%NT_SPA_AGENT_HIDDEN];
    index-=NT_SPA_AGENT_ACTIONS*NT_SPA_AGENT_HIDDEN;
    return &p->b2[index];
}
static void gradient_policy(nt_spa_policy *p,int saturated) {
    unsigned i;
    memset(p,0,sizeof(*p));
    for(i=0;i<NT_SPA_AGENT_PARAMETERS;++i)
        *parameter(p,i)=.011f*(float)((int)((i*7u+3u)%17u)-8);
    /* Distinct nonzero rows make hidden gradients depend on all output heads. */
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        p->w2[0][i]=.031f+.004f*i;
        p->w2[1][i]=-.057f+.003f*i;
        p->w2[2][i]=.087f-.002f*i;
    }
    p->b2[0]=saturated ? 2.0f : .13f;
    p->b2[1]=saturated ? -2.0f : -.17f;
    p->b2[2]=.31f;
}
static int test_gradient(void) {
    nt_spa_agent a,updated,plus,minus;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c;
    nt_spa_comparison_receipt step,hi,lo;
    unsigned k,last,regime,nonzero[NT_SPA_AGENT_PARAMETERS]={0},covered=0,comparisons=0;
    double maximum_error=0;
    const float epsilon=.002f,rate=.03125f;
    CHECK(initialized(&a,0),"gradient init");
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"capture gradient input");
    e.features[19]=.17f; e.features[20]=.42f; e.features[21]=.31f;
    e.features[22]=e.features[23]=e.features[24]=1.0f/3.0f; e.features[28]=-.23f;
    c=comparison(&e,4);
    for(regime=0;regime<2;++regime) for(last=0;last<NT_SPA_AGENT_ACTIONS;++last) {
        nt_spa_policy p;
        e.features[25]=e.features[26]=e.features[27]=0; e.features[25+last]=1;
        CHECK(nt_spa_experience_validate(&e)==NT_SPA_OK,"valid recorded last-action fixture");
        gradient_policy(&p,(int)regime);
        CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"install smooth interior policy");
        updated=a;
        CHECK(nt_spa_agent_fit_comparison(&updated,&e,&c,rate,&step)==NT_SPA_OK,"one comparison step");
        CHECK(only_policy_changed(&a,&updated),"all gradients change only policy");
        CHECK(step.loss_after<step.loss_before,"Huber step decreases fixture loss");
        if(regime) CHECK(fabsf(step.scores_before[0]-step.targets[0])>1 &&
            fabsf(step.scores_before[1]-step.targets[1])>1,"linear Huber branches exercised");
        else CHECK(fabsf(step.scores_before[0]-step.targets[0])<1 &&
            fabsf(step.scores_before[1]-step.targets[1])<1,"quadratic Huber branches exercised");
        for(k=0;k<NT_SPA_AGENT_PARAMETERS;++k) {
            double numeric,observed,error,tolerance;
            float high,low;
            plus=a; minus=a;
            *parameter(&plus.policy,k)+=epsilon; *parameter(&minus.policy,k)-=epsilon;
            high=*parameter(&plus.policy,k); low=*parameter(&minus.policy,k);
            CHECK(nt_spa_agent_fit_comparison(&plus,&e,&c,0,&hi)==NT_SPA_OK,"positive finite difference");
            CHECK(nt_spa_agent_fit_comparison(&minus,&e,&c,0,&lo)==NT_SPA_OK,"negative finite difference");
            numeric=(hi.loss_before-lo.loss_before)/((double)high-low);
            observed=((double)*parameter(&updated.policy,k)-*parameter(&a.policy,k))/rate;
            error=fabs(observed+numeric); tolerance=3e-5+.003*fabs(numeric);
            if(error>maximum_error) maximum_error=error;
            if(fabs(numeric)>1e-6) nonzero[k]=1;
            if(error>tolerance) fprintf(stderr,
                "gradient detail: regime=%u last=%u parameter=%u numeric=%.10g step=%.10g error=%.10g\n",
                regime,last,k,numeric,observed,error);
            CHECK(error<=tolerance,"all-parameter finite difference agrees with simultaneous SGD");
            ++comparisons;
        }
    }
    for(k=0;k<NT_SPA_AGENT_PARAMETERS;++k) covered+=nonzero[k];
    CHECK(comparisons==1602,"267 parameters by three history inputs by two Huber regimes");
    CHECK(covered==NT_SPA_AGENT_PARAMETERS,"every parameter has a nonzero measured gradient");
    printf("gradient: comparisons=%u nonzero_parameters=%u maximum_absolute_error=%.9g\n",
        comparisons,covered,maximum_error);
    return 1;
}

static int test_masks(void) {
    static const uint32_t targets[]={0,3,1,0},counts[]={4,4,3,1},masks[]={5,3,7,1};
    unsigned case_index,i;
    for(case_index=0;case_index<4;++case_index) {
        nt_spa_agent a,before;
        nt_spa_observation o=observation(targets[case_index],counts[case_index]);
        nt_spa_experience e;
        nt_spa_comparison c,bad;
        nt_spa_policy p;
        nt_spa_readout read;
        nt_spa_comparison_receipt r;
        CHECK(initialized(&a,0),"boundary init");
        memset(&p,0,sizeof(p)); p.b2[0]=.25f;
        for(i=1;i<NT_SPA_AGENT_ACTIONS;++i) {
            p.b2[i]=(masks[case_index]&(1u<<i)) ? -.25f : 7;
            p.w2[i][0]=-0.0f;
        }
        CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"install boundary preferences");
        CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"boundary experience");
        c=comparison(&e,17); before=a;
        CHECK(nt_spa_agent_score_experience(&a,&e,&read)==NT_SPA_OK &&
            read.action_mask==masks[case_index] && read.action.kind==NT_SPA_KEEP,
            "unavailable high-scoring heads cannot select invalid action");
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,0,&r)==NT_SPA_OK && SAME(a,before),
            "zero-rate fit is exact whole-state no-op");
        CHECK(!memcmp(r.scores_before,read.scores,sizeof(read.scores)) &&
            !memcmp(r.scores_after,read.scores,sizeof(read.scores)),"receipt retains every raw head");
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,.03f,&r)==NT_SPA_OK &&
            r.action_mask==masks[case_index] && r.horizon==17,"all valid heads fitted at generic horizon");
        CHECK(only_policy_changed(&before,&a) && r.loss_after<r.loss_before,"bounded fit learns policy only");
        for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(!(masks[case_index]&(1u<<i))) {
            CHECK(!memcmp(a.policy.w2[i],before.policy.w2[i],sizeof(a.policy.w2[i])) &&
                !memcmp(&a.policy.b2[i],&before.policy.b2[i],sizeof(float)),
                "inactive parameters retain exact bytes including negative zero");
            CHECK(r.rewards[i]==0 && r.targets[i]==0,"inactive measurements have no target");
            bad=c; bad.alternatives[i].action=action(i,e.sentence_index);
            CHECK(refused(&a,&e,&bad,.03f),"inactive alternative storage must be empty");
        }
        bad=c; bad.action_mask^=1u;
        CHECK(refused(&a,&e,&bad,.03f),"paired KEEP must be present");
        if(case_index==3) CHECK(r.targets[0]==0,"single sentence has one KEEP zero target");
    }
    return 1;
}

/* Retained real body: scenarios/raw_traces.jsonl.gz, seed42 snapshot0, target0.
   SHA256 dcee2f733f8a7a4063a44f0654b574dad55ce6b57ea258ee81a7582d39370d22.
   H0 and H4 are separate measured comparisons with identical captured input. */
static nt_spa_experience measured_experience(void) {
    static const float x[NT_SPA_AGENT_FEATURES]={
        -.00639171572f,-.00175221951f,-.0116951521f,-.00439821463f,
        .33455649f,0,.541619003f,.805849493f,.198651791f,.0483870953f,
        .5f,.916208446f,0,0,.0500000007f,0,1,.250457585f,.250687391f,
        0,0,0,0,0,0,0,0,0,0};
    nt_spa_experience e;
    memset(&e,0,sizeof(e)); e.version=NT_SPA_EXPERIENCE_VERSION;
    e.sentence_index=0; e.sentence_count=4; e.source_life_hash=UINT64_C(0x659aba85064eba3c);
    memcpy(e.features,x,sizeof(x)); return e;
}
static nt_spa_comparison measured_comparison(const nt_spa_experience *e,uint32_t horizon) {
    nt_spa_metrics before={.770809531f,.33455649f,.805849552f,.198651791f,
        .0483870953f,0,.770809531f};
    nt_spa_comparison c;
    memset(&c,0,sizeof(c)); c.source_life_hash=e->source_life_hash; c.horizon=horizon;
    c.action_mask=5; c.alternatives[0].action=action(0,0); c.alternatives[2].action=action(2,0);
    c.alternatives[0].consequence.before=before; c.alternatives[2].consequence.before=before;
    if(horizon==0) {
        c.alternatives[0].consequence.after=before;
        c.alternatives[2].consequence.after=(nt_spa_metrics){.749121428f,.333635777f,
            .798620105f,.501757085f,.0535714291f,0,.749121428f};
        c.alternatives[2].consequence.regeneration_cost=.90625f;
    } else {
        c.alternatives[0].consequence.after=(nt_spa_metrics){.765494525f,.333503932f,
            .773790121f,.436454892f,.112903222f,0,.765494525f};
        c.alternatives[0].consequence.regeneration_cost=.756250024f;
        c.alternatives[2].consequence.after=(nt_spa_metrics){.55556339f,.334254235f,
            .630421758f,.368925571f,.193548381f,0,.544640839f};
        c.alternatives[2].consequence.regeneration_cost=.834375024f;
    }
    return c;
}
static int test_credit(void) {
    nt_spa_agent a,initial,h0_life,counterfactual;
    nt_spa_experience e=measured_experience(),saved_e=e;
    nt_spa_comparison immediate=measured_comparison(&e,0),future=measured_comparison(&e,4);
    nt_spa_comparison saved_immediate=immediate,saved_future=future;
    nt_spa_comparison_receipt h0,h4,last;
    nt_spa_readout r;
    unsigned i;
    CHECK(initialized(&a,0),"real fixture init"); initial=a;
    CHECK(nt_spa_experience_validate(&e)==NT_SPA_OK,"retained body feature geometry");
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&immediate,0,&h0)==NT_SPA_OK,"measure H0 targets");
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&future,0,&h4)==NT_SPA_OK,"measure H4 targets");
    CHECK(NEAR(h0.rewards[0],0,1e-9) && NEAR(h0.rewards[2],.0086092921,2e-8),
        "raw H0 reward agrees with retained body receipt");
    CHECK(NEAR(h4.rewards[0],-.00756207155,2e-8) && NEAR(h4.rewards[2],-.108164445,2e-8),
        "raw H4 rewards agree with retained body receipts");
    CHECK(h0.targets[0]==0 && h4.targets[0]==0 && h0.targets[2]>.008 && h4.targets[2]<-.10,
        "future credit changes measured action advantage sign");
    for(i=0;i<512;++i)
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&immediate,.03f,&last)==NT_SPA_OK,"fit immediate measured outcomes");
    CHECK(last.loss_after<h0.loss_before*.001,"H0 fixture loss decreases");
    CHECK(nt_spa_agent_score_experience(&a,&e,&r)==NT_SPA_OK && r.action.kind==NT_SPA_RESEED_RIGHT,
        "measured immediate experience acquires RIGHT choice");
    h0_life=a;
    CHECK(only_policy_changed(&initial,&a),"immediate replay preserves all non-policy state");
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&future,0,&h4)==NT_SPA_OK,"H4 loss at acquired H0 policy");
    for(i=0;i<512;++i)
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&future,.03f,&last)==NT_SPA_OK,"fit future measured outcomes");
    CHECK(last.loss_after<h4.loss_before*.001,"H4 fixture loss decreases");
    CHECK(nt_spa_agent_score_experience(&a,&e,&r)==NT_SPA_OK && r.action.kind==NT_SPA_KEEP,
        "future experience reverses RIGHT to KEEP");
    CHECK(only_policy_changed(&h0_life,&a),"future replay changes only policy weights");
    counterfactual=a;
    CHECK(nt_spa_agent_set_policy(&counterfactual,&h0_life.policy)==NT_SPA_OK,"restore H0 weights only");
    CHECK(nt_spa_agent_score_experience(&counterfactual,&e,&r)==NT_SPA_OK && r.action.kind==NT_SPA_RESEED_RIGHT,
        "identical captured input/history/RNG with H0 weights selects RIGHT");
    CHECK(only_policy_changed(&a,&counterfactual) && SAME(e,saved_e) &&
        SAME(immediate,saved_immediate) && SAME(future,saved_future),"causal comparison preserves source inputs");
    printf("credit: H0 RIGHT advantage=%.9g; H4 RIGHT advantage=%.9g; acquired RIGHT -> KEEP\n",
        h0.targets[2],h4.targets[2]);
    return 1;
}

static int test_persistence(void) {
    nt_spa_agent a,b,initial;
    nt_spa_observation o=observation(1,3);
    nt_spa_experience e;
    nt_spa_comparison c;
    nt_spa_comparison_receipt first,second;
    nt_spa_decision d1,d2;
    nt_spa_receipt r1,r2;
    nt_spa_consequence outcome=consequence();
    char path[160];
    FILE *f;
    unsigned char magic[8];
    long bytes;
    unsigned i;
    CHECK(initialized(&a,.03f),"continuation init"); initial=a;
    CHECK(nt_spa_agent_capture_experience(&a,&o,&e)==NT_SPA_OK,"continuation capture");
    c=comparison(&e,4);
    CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,.03f,&first)==NT_SPA_OK,"fit before save");
    CHECK(only_policy_changed(&initial,&a),"comparison adds no schema counters");
    (void)snprintf(path,sizeof(path),"/tmp/notorch-spa-future-%ld.bin",(long)getpid());
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"save acquired life");
    f=fopen(path,"rb"); CHECK(f!=NULL,"open canonical file");
    CHECK(fread(magic,1,sizeof(magic),f)==sizeof(magic) && !memcmp(magic,"NTSPA001",8),
        "v1 format magic remains exact");
    CHECK(fseek(f,0,SEEK_END)==0,"seek saved life"); bytes=ftell(f);
    CHECK(fclose(f)==0 && bytes==2296,"canonical file retains v1 byte count");
    memset(&b,0x5a,sizeof(b));
    CHECK(nt_spa_agent_load(&b,path)==NT_SPA_OK && SAME(a,b),"reload preserves every acquired state byte");
    CHECK(nt_spa_agent_hash(&a)==nt_spa_agent_hash(&b),"canonical hash agrees after reload");
    for(i=0;i<20;++i) {
        CHECK(nt_spa_agent_fit_comparison(&a,&e,&c,.03f,&first)==NT_SPA_OK &&
            nt_spa_agent_fit_comparison(&b,&e,&c,.03f,&second)==NT_SPA_OK,
            "continue comparison fitting after reload");
        CHECK(SAME(a,b) && SAME(first,second),"replay continuation and receipt bytes exact");
    }
    CHECK(nt_spa_agent_choose(&a,&o,&d1)==NT_SPA_OK && nt_spa_agent_choose(&b,&o,&d2)==NT_SPA_OK &&
        SAME(d1,d2) && SAME(a,b),"old choose continues identically after replay");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK && nt_spa_agent_load(&b,path)==NT_SPA_OK && SAME(a,b),
        "pending old decision survives same v1 encoding");
    CHECK(refused(&a,&e,&c,.03f) && SAME(a,b),"replay cannot rewrite pending decision weights");
    CHECK(nt_spa_agent_observe(&a,d1.sequence,&d1.action,&outcome,&r1)==NT_SPA_OK &&
        nt_spa_agent_observe(&b,d2.sequence,&d2.action,&outcome,&r2)==NT_SPA_OK && SAME(a,b) && SAME(r1,r2),
        "v1 action witness and online credit continue exactly after replay/save");
    CHECK(nt_spa_agent_validate(&a)==NT_SPA_OK && a.observations==1 && a.updates==1,
        "only online credit advances v1 learning counters");
    CHECK(unlink(path)==0,"remove temporary saved life");
    return 1;
}

int main(int argc,char **argv) {
    static const struct { const char *name; int (*run)(void); } groups[]={
        {"capture",test_capture},{"validation",test_validation},{"gradient",test_gradient},
        {"masks",test_masks},{"credit",test_credit},{"persistence",test_persistence}};
    unsigned i,ran=0;
    if(argc>2) return 2;
    for(i=0;i<sizeof(groups)/sizeof(groups[0]);++i) {
        if(argc==2 && strcmp(argv[1],groups[i].name)) continue;
        gate=groups[i].name;
        if(!groups[i].run()) return 1;
        printf("PASS %s\n",gate); ++ran;
    }
    if(!ran) { fprintf(stderr,"unknown future gate\n"); return 2; }
    printf("PASS SPA future: %u groups, %u checks\n",ran,checks);
    return 0;
}
