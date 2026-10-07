/* Independent host/API gates for Sentence Phonon Agent. */
#include "spa_agent.h"
#include "notorch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *gate;
static unsigned checks;
#define CHECK(c, msg) do { checks++; if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",gate,__LINE__,msg); return 0; } } while(0)
#define SAME(a,b) (memcmp(&(a), &(b), sizeof(a)) == 0)
#define CLOSE(a,b) (isfinite(a) && fabsf((a)-(b)) < 1e-6f)

static nt_spa_observation observation(unsigned target, unsigned count, float score) {
    nt_spa_observation o;
    memset(&o,0,sizeof o);
    o.embedding[0]=0.25f; o.embedding[1]=-0.5f;
    o.embedding[2]=0.75f; o.embedding[3]=-0.125f;
    o.connectedness=.4f;
    o.left_similarity=target ? .2f : 0;
    o.right_similarity=target+1<count ? .8f : 0;
    o.coherence=.5f; o.novelty=.3f; o.repetition=.1f; o.phase_lock=.5f;
    o.sentence_score=score; o.mean_sentence_score=1;
    o.temperature=.85f; o.sentence_index=target; o.sentence_count=count;
    return o;
}

static nt_spa_consequence outcome(float before, float after) {
    nt_spa_consequence c;
    memset(&c,0,sizeof c);
    c.before.local_connectedness=c.after.local_connectedness=.5f;
    c.before.global_connectedness=c.after.global_connectedness=.5f;
    c.before.novelty=c.after.novelty=.25f;
    c.before.repetition=c.after.repetition=.25f;
    c.before.collapse=c.after.collapse=.25f;
    c.before.continuity=c.after.continuity=.5f;
    c.before.coherence=before; c.after.coherence=after;
    return c;
}

static int initialized(nt_spa_agent *a, nt_spa_agent_mode mode, float exploration) {
    nt_spa_agent_config c;
    nt_spa_agent_config_default(&c);
    c.mode=mode; c.exploration=exploration;
    return nt_spa_agent_init(a,&c)==NT_SPA_OK;
}

static int zero_policy(nt_spa_agent *a) {
    nt_spa_policy zero;
    memset(&zero,0,sizeof zero);
    return nt_spa_agent_set_policy(a,&zero)==NT_SPA_OK;
}

static int test_legacy(void) {
    nt_spa_agent a, before;
    nt_spa_decision d;
    nt_spa_action action;
    nt_spa_observation o=observation(1,3,.1f);
    CHECK(initialized(&a,NT_SPA_AGENT_DISABLED,0),"disabled init");
    before=a;
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"disabled choose");
    CHECK(SAME(a,before),"disabled leaves every agent byte unchanged");
    CHECK(d.action.kind==NT_SPA_KEEP && d.sequence==0,"disabled selects KEEP without credit");
    CHECK(nt_spa_agent_choose(NULL,&o,&d)==NT_SPA_OK && d.action.kind==NT_SPA_KEEP,
          "absent agent selects KEEP");
    const float table[12]={1,0,-2,4,0,2,4,-2,3,-1,0,1};
    const int ids[3]={0,1,2};
    float emb[4], copied[4], logits[4]={-2,0,1,4}, reference[4]={-2,0,1,4};
    nt_spa_embed_sentence(ids,3,table,3,4,.5f,reference);
    nt_spa_embed_sentence(ids,3,table,3,4,.5f,emb);
    memcpy(copied,emb,sizeof copied);
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"disabled host call");
    CHECK(!memcmp(emb,copied,sizeof emb) && !memcmp(emb,reference,sizeof emb),
          "disabled host preserves sentence embedding");
    float conn=nt_spa_connectedness(emb,4,table,3);
    memcpy(reference,logits,sizeof logits);
    nt_spa_modulate_logits(logits,4,conn,.3f);
    nt_spa_modulate_logits(reference,4,conn,.3f);
    CHECK(!memcmp(reference,logits,sizeof logits),"disabled host preserves logits");

    o.phase_lock=0; o.sentence_score=.7f;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_KEEP,
          "strict legacy threshold at phase0");
    o.sentence_score=.69f;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_RESEED_LEFT
          && action.target==1 && action.source==0,"legacy prefers left despite stronger right");
    o.sentence_index=0; o.left_similarity=0;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_RESEED_RIGHT
          && action.source==1,"legacy right at first sentence");
    o.phase_lock=1; o.sentence_score=.52f;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_KEEP,
          "strict legacy threshold at phase1");
    o.sentence_score=.51f;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_RESEED_RIGHT,
          "legacy below phase1 threshold");
    o.sentence_count=1; o.right_similarity=0;
    CHECK(nt_spa_agent_legacy(&o,&action)==NT_SPA_OK && action.kind==NT_SPA_KEEP,
          "single sentence keeps");
    CHECK(initialized(&a,NT_SPA_AGENT_LEGACY,0),"legacy init");
    o=observation(2,3,.1f);
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK && d.action.kind==NT_SPA_RESEED_LEFT,
          "explicit legacy mode uses registered rule");
    nt_spa_policy policy=a.policy;
    nt_spa_consequence c=outcome(.1f,.9f);
    nt_spa_receipt r;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)==NT_SPA_OK,"legacy consequence recorded");
    CHECK(SAME(policy,a.policy) && !r.learned,"legacy policy weights retained");
    return 1;
}

static int test_config(void) {
    nt_spa_agent a,before;
    nt_spa_agent_config good,bad;
    nt_spa_agent_config_default(&good);
    CHECK(nt_spa_agent_init(&a,&good)==NT_SPA_OK,"default valid");
    CHECK(nt_spa_agent_validate(&a)==NT_SPA_OK,"new state valid");
    before=a;
#define BAD_CONFIG(field,value) do { bad=good; bad.field=(value); \
    CHECK(nt_spa_agent_init(&a,&bad)!=NT_SPA_OK,"invalid config " #field); \
    CHECK(SAME(a,before),"config refusal transactional " #field); } while(0)
    BAD_CONFIG(mode,(nt_spa_agent_mode)99);
    BAD_CONFIG(seed,0);
    BAD_CONFIG(learning_rate,-.1f);
    BAD_CONFIG(learning_rate,NAN);
    BAD_CONFIG(imitation_rate,-1);
    BAD_CONFIG(exploration,-.1f);
    BAD_CONFIG(exploration,1.01f);
    BAD_CONFIG(memory_decay,-.1f);
    BAD_CONFIG(memory_decay,1.01f);
    BAD_CONFIG(reward_weights.novelty,-.1f);
    BAD_CONFIG(reward_weights.coherence,INFINITY);
    BAD_CONFIG(cost_weight,-.1f);
#undef BAD_CONFIG
    nt_spa_observation o=observation(1,3,.1f);
    nt_spa_decision d,untouched;
    memset(&untouched,0x5a,sizeof untouched);
#define BAD_STATE(field,value) do { a=before; a.field=(value); nt_spa_agent saved=a; d=untouched; \
    CHECK(nt_spa_agent_validate(&a)!=NT_SPA_OK,"invalid state " #field); \
    CHECK(nt_spa_agent_choose(&a,&o,&d)!=NT_SPA_OK,"invalid state choose " #field); \
    CHECK(SAME(a,saved) && SAME(d,untouched),"state refusal transactional " #field); } while(0)
    BAD_STATE(version,0);
    BAD_STATE(perception_version,0);
    BAD_STATE(reward_version,0);
    BAD_STATE(config_hash,0);
    BAD_STATE(config.exploration,.987f);
    BAD_STATE(rng,0);
    BAD_STATE(history_count,NT_SPA_AGENT_HISTORY+1);
    BAD_STATE(history_head,NT_SPA_AGENT_HISTORY);
    BAD_STATE(ema_reward,NAN);
    BAD_STATE(policy.b2[0],INFINITY);
    BAD_STATE(pending,2);
#undef BAD_STATE
    return 1;
}

static int test_actions(void) {
    nt_spa_observation good=observation(1,3,.1f), bad;
    nt_spa_agent a,before;
    nt_spa_decision d,untouched;
    memset(&untouched,0x5a,sizeof untouched);
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,0),"learned init");
    before=a;
#define BAD_OBS(field,value) do { bad=good; bad.field=(value); d=untouched; \
    CHECK(nt_spa_observation_validate(&bad)!=NT_SPA_OK,"invalid observation " #field); \
    CHECK(nt_spa_agent_choose(&a,&bad,&d)!=NT_SPA_OK,"observation choose rejected " #field); \
    CHECK(SAME(a,before) && SAME(d,untouched),"observation refusal transactional " #field); } while(0)
    BAD_OBS(embedding[0],NAN);
    BAD_OBS(embedding[1],1.01f);
    BAD_OBS(connectedness,-.01f);
    BAD_OBS(coherence,1.01f);
    BAD_OBS(phase_lock,NAN);
    BAD_OBS(novelty,1.01f);
    BAD_OBS(repetition,-.01f);
    BAD_OBS(left_similarity,1.01f);
    BAD_OBS(right_similarity,-1.01f);
    BAD_OBS(sentence_score,-1);
    BAD_OBS(mean_sentence_score,1000001);
    BAD_OBS(temperature,0);
    BAD_OBS(temperature,16.01f);
    BAD_OBS(sentence_index,3);
    BAD_OBS(sentence_count,0);
    BAD_OBS(sentence_count,NT_SPA_AGENT_MAX_SENTENCES+1);
#undef BAD_OBS
    nt_spa_action x={NT_SPA_KEEP,1,NT_SPA_AGENT_NO_SOURCE};
    CHECK(nt_spa_action_validate(&x,&good)==NT_SPA_OK,"valid KEEP");
    x.source=0;
    CHECK(nt_spa_action_validate(&x,&good)!=NT_SPA_OK,"KEEP cannot carry source");
    x.kind=NT_SPA_RESEED_LEFT;
    CHECK(nt_spa_action_validate(&x,&good)==NT_SPA_OK,"valid left");
    x.source=2;
    CHECK(nt_spa_action_validate(&x,&good)!=NT_SPA_OK,"left source must be immediate neighbor");
    x.kind=NT_SPA_RESEED_RIGHT;
    CHECK(nt_spa_action_validate(&x,&good)==NT_SPA_OK,"valid right");
    x.target=3;
    CHECK(nt_spa_action_validate(&x,&good)!=NT_SPA_OK,"out of range target");
    x.target=1; x.kind=(nt_spa_action_kind)3;
    CHECK(nt_spa_action_validate(&x,&good)!=NT_SPA_OK,"unknown action");
    nt_spa_policy p;
    memset(&p,0,sizeof p); p.b2[NT_SPA_RESEED_LEFT]=5;
    CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"fixed preferences installed");
    CHECK(nt_spa_agent_select(&a,&good,&d)==NT_SPA_OK && d.action.kind==NT_SPA_RESEED_LEFT,
          "selection reads learned left preference");
    good.sentence_index=0; good.left_similarity=0;
    CHECK(nt_spa_agent_select(&a,&good,&d)==NT_SPA_OK && d.action.kind==NT_SPA_KEEP,
          "first sentence masks unavailable left");
    p.b2[NT_SPA_RESEED_RIGHT]=6;
    CHECK(nt_spa_agent_set_policy(&a,&p)==NT_SPA_OK,"fixed right preference installed");
    good.sentence_index=2; good.left_similarity=.2f; good.right_similarity=0;
    CHECK(nt_spa_agent_select(&a,&good,&d)==NT_SPA_OK && d.action.kind==NT_SPA_RESEED_LEFT,
          "last sentence masks unavailable right");
    good.sentence_count=1; good.sentence_index=0; good.left_similarity=0;
    CHECK(nt_spa_agent_select(&a,&good,&d)==NT_SPA_OK && d.action.kind==NT_SPA_KEEP,
          "single sentence masks both reseeds");
    before=a; p.b1[0]=NAN;
    CHECK(nt_spa_agent_set_policy(&a,&p)!=NT_SPA_OK && SAME(a,before),"invalid replacement transactional");
    return 1;
}

static int test_perception(void) {
    const float e[12]={1,0,0,0,0,1,0,0,-1,0,0,0};
    nt_spa_observation o,saved;
    CHECK(nt_spa_agent_perceive(e,3,4,0,.5f,.8f,0,&o)==NT_SPA_OK,"native perception");
    CHECK(CLOSE(o.embedding[0],tanhf(1)) && o.embedding[1]==0,"compact representation from native vectors");
    CHECK(o.left_similarity==0 && o.right_similarity==0,"orthogonal adjacent sentence");
    CHECK(CLOSE(o.novelty,1) && CLOSE(o.repetition,0),"orthogonal/opposed history novelty");
    CHECK(CLOSE(o.coherence,.5f),"orthogonal trajectory coherence");
    const float others[8]={0,1,0,0,-1,0,0,0};
    float legacy=nt_spa_connectedness(e,4,others,2);
    CHECK(o.connectedness==legacy,"native perception calls existing SPA on other sentences");
    saved=o;
    CHECK(nt_spa_agent_perceive(e,3,4,3,.5f,.8f,0,&o)!=NT_SPA_OK && SAME(o,saved),
          "perception target refusal transactional");
    CHECK(nt_spa_agent_perceive(e,0,4,0,.5f,.8f,0,&o)!=NT_SPA_OK && SAME(o,saved),
          "empty field rejected");
    CHECK(nt_spa_agent_perceive(e,3,0,0,.5f,.8f,0,&o)!=NT_SPA_OK && SAME(o,saved),
          "zero dimension rejected");
    CHECK(nt_spa_agent_perceive(e,3,NT_SPA_AGENT_MAX_DIM+1,0,.5f,.8f,0,&o)!=NT_SPA_OK,
          "dimension limit checked before read");
    float broken[12]; memcpy(broken,e,sizeof e); broken[11]=NAN;
    CHECK(nt_spa_agent_perceive(broken,3,4,0,.5f,.8f,0,&o)!=NT_SPA_OK && SAME(o,saved),
          "nonfinite sentence rejected transactionally");
    return 1;
}

static int test_deterministic(void) {
    nt_spa_agent a,b,snapshot;
    srand(314159);
    int expected_global=rand();
    srand(314159);
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,.8f),"exploring life"); b=a;
    for (unsigned i=0;i<128;i++) {
        nt_spa_observation o=observation(i%3,3,.1f+.1f*(i%7));
        nt_spa_decision da,db,preview;
        snapshot=a;
        CHECK(nt_spa_agent_select(&a,&o,&preview)==NT_SPA_OK,"pure preview");
        CHECK(SAME(a,snapshot),"preview leaves RNG and all state unchanged");
        CHECK(nt_spa_agent_choose(&a,&o,&da)==NT_SPA_OK,"choose first life");
        CHECK(nt_spa_agent_choose(&b,&o,&db)==NT_SPA_OK,"choose repeated life");
        CHECK(SAME(da,db) && SAME(da,preview) && SAME(a,b),"repeatable private RNG policy and preview");
        nt_spa_consequence c=outcome(.5f,.1f+.1f*(i%9));
        nt_spa_receipt ra,rb;
        CHECK(nt_spa_agent_observe(&a,da.sequence,&da.action,&c,&ra)==NT_SPA_OK,"first credit");
        CHECK(nt_spa_agent_observe(&b,db.sequence,&db.action,&c,&rb)==NT_SPA_OK,"second credit");
        CHECK(SAME(ra,rb) && SAME(a,b),"repeated acquired state identical");
    }
    CHECK(rand()==expected_global,"private RNG leaves host C RNG unchanged");
    return 1;
}

static int test_credit(void) {
    nt_spa_agent_config cfg;
    nt_spa_agent_config_default(&cfg); cfg.mode=NT_SPA_AGENT_LEARNED; cfg.exploration=0;
    memset(&cfg.reward_weights,0,sizeof cfg.reward_weights);
    cfg.reward_weights.coherence=.5f; cfg.reward_weights.novelty=.25f;
    cfg.reward_weights.repetition=.125f; cfg.reward_weights.collapse=.0625f;
    cfg.cost_weight=.25f;
    nt_spa_agent a;
    CHECK(nt_spa_agent_init(&a,&cfg)==NT_SPA_OK && zero_policy(&a),"zero policy credit fixture");
    nt_spa_observation o=observation(1,3,.1f);
    nt_spa_decision d;
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK && d.action.kind==NT_SPA_KEEP,"initial tie chooses KEEP");
    nt_spa_consequence c=outcome(.5f,.75f);
    c.before.novelty=.5f; c.after.novelty=.875f;
    c.before.repetition=.5f; c.after.repetition=.125f;
    c.before.collapse=.5f; c.after.collapse=.25f; c.regeneration_cost=.5f;
    nt_spa_receipt r;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)==NT_SPA_OK,"measured consequence");
    CHECK(r.reward==.15625f,"declared independent reward arithmetic = 5/32");
    CHECK(SAME(r.consequence,c),"receipt retains every raw measurement");
    CHECK(r.predicted==0 && r.learned && a.updates==1,"initial zero prediction trained once");
    CHECK(a.policy.b2[NT_SPA_KEEP]>0,"positive outcome increases selected prediction");
    nt_spa_agent positive=a;
    CHECK(nt_spa_agent_init(&a,&cfg)==NT_SPA_OK && zero_policy(&a),"negative fixture");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"negative selection");
    c=outcome(.75f,.25f);
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)==NT_SPA_OK && r.reward==-.25f,"negative reward");
    CHECK(a.policy.b2[NT_SPA_KEEP]<0,"negative outcome decreases selected prediction");
    CHECK(a.policy.b2[NT_SPA_RESEED_LEFT]==0 && a.policy.b2[NT_SPA_RESEED_RIGHT]==0,
          "unchosen readout biases receive no outcome credit");
    CHECK(!SAME(a.policy,positive.policy),"opposite consequences acquire different weights");

    nt_spa_agent acquired=a, initial=a;
    nt_spa_policy learned=acquired.policy, zero;
    memset(&zero,0,sizeof zero);
    CHECK(nt_spa_agent_set_policy(&initial,&zero)==NT_SPA_OK,"reset only counterfactual weights");
    nt_spa_agent expected=acquired; expected.policy=zero;
    CHECK(SAME(initial,expected),"counterfactual observation memory history counters RNG identical");
    nt_spa_decision di,da;
    CHECK(nt_spa_agent_select(&initial,&o,&di)==NT_SPA_OK,"initial counterfactual selection");
    CHECK(nt_spa_agent_select(&acquired,&o,&da)==NT_SPA_OK,"acquired counterfactual selection");
    CHECK(!memcmp(di.features,da.features,sizeof di.features) && di.rng_before==da.rng_before,
          "fixed policy inputs and RNG");
    CHECK(di.action.kind==NT_SPA_KEEP && da.action.kind==NT_SPA_RESEED_LEFT,
          "acquired experience alone changes action KEEP to LEFT");
    CHECK(SAME(learned,acquired.policy),"select leaves acquired policy fixed");
    return 1;
}

static int test_sequence(void) {
    nt_spa_agent a,before;
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,0),"sequence life");
    nt_spa_observation o=observation(1,3,.1f);
    nt_spa_decision d,spare,untouched;
    nt_spa_consequence c=outcome(.5f,.75f),broken;
    nt_spa_receipt r,untouched_receipt;
    memset(&untouched,0x5a,sizeof untouched); memset(&untouched_receipt,0x5a,sizeof untouched_receipt);
    before=a; r=untouched_receipt;
    nt_spa_action unexecuted={NT_SPA_KEEP,1,NT_SPA_AGENT_NO_SOURCE};
    CHECK(nt_spa_agent_observe(&a,1,&unexecuted,&c,&r)!=NT_SPA_OK && SAME(a,before) && SAME(r,untouched_receipt),
          "unsolicited credit refused");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"reserve credit");
    before=a; spare=untouched;
    CHECK(nt_spa_agent_choose(&a,&o,&spare)!=NT_SPA_OK && SAME(a,before) && SAME(spare,untouched),
          "second choice refused before credit");
    CHECK(nt_spa_agent_select(&a,&o,&spare)!=NT_SPA_OK && SAME(a,before),"pending preview refused");
    r=untouched_receipt;
    CHECK(nt_spa_agent_observe(&a,d.sequence+1,&d.action,&c,&r)!=NT_SPA_OK && SAME(a,before) && SAME(r,untouched_receipt),
          "out of order credit refused transactionally");
    nt_spa_action wrong={NT_SPA_RESEED_LEFT,1,0};
    if(d.action.kind==NT_SPA_RESEED_LEFT) {
        wrong.kind=NT_SPA_KEEP; wrong.source=NT_SPA_AGENT_NO_SOURCE;
    }
    CHECK(nt_spa_agent_observe(&a,d.sequence,&wrong,&c,&r)==NT_SPA_E_ACTION
          && SAME(a,before) && SAME(r,untouched_receipt),"different valid executed action refused");
    CHECK(nt_spa_agent_observe(&a,d.sequence,NULL,&c,&r)==NT_SPA_E_ACTION && SAME(a,before),
          "executed-action witness required");
    broken=c; broken.after.novelty=NAN;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&broken,&r)!=NT_SPA_OK && SAME(a,before),"nonfinite credit refused");
    broken=c; broken.regeneration_cost=1.01f;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&broken,&r)!=NT_SPA_OK && SAME(a,before),"cost bounds refused");
    CHECK(nt_spa_agent_reset_memory(&a)!=NT_SPA_OK && SAME(a,before),"pending reset refused");
    CHECK(nt_spa_agent_set_policy(&a,&a.policy)!=NT_SPA_OK && SAME(a,before),"pending weight replacement refused");
    float loss=123;
    CHECK(nt_spa_agent_imitate(&a,&o,NT_SPA_KEEP,&loss)!=NT_SPA_OK && SAME(a,before) && loss==123,
          "pending imitation refused");
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)==NT_SPA_OK,"correct credit accepted");
    before=a;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)!=NT_SPA_OK && SAME(a,before),"duplicate credit refused");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"second decision");
    nt_spa_policy policy=a.policy;
    uint64_t prior_updates=a.updates,prior_observations=a.observations;
    uint32_t prior_rng=a.rng;
    CHECK(nt_spa_agent_cancel(&a,d.sequence)==NT_SPA_OK,"unexecuted action cancelled");
    CHECK(!a.pending && SAME(a.policy,policy) && a.updates==prior_updates && a.observations==prior_observations
          && a.rng==prior_rng && a.cancelled==1,"cancel consumes no consequence or acquired weights");
    before=a;
    CHECK(nt_spa_agent_cancel(&a,d.sequence)!=NT_SPA_OK && SAME(a,before),"duplicate cancellation rejected");
    return 1;
}

static int test_imitation(void) {
    nt_spa_agent a;
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,0),"imitation life");
    nt_spa_observation fixture[9];
    nt_spa_action labels[9];
    for (unsigned i=0;i<9;i++) {
        unsigned target=i%3;
        fixture[i]=observation(target,3,i<3 ? .9f : .1f);
        fixture[i].phase_lock=i>=6 ? .8f : .2f;
        CHECK(nt_spa_agent_legacy(&fixture[i],&labels[i])==NT_SPA_OK,"registered teacher label");
    }
    double first=0,last=0;
    for (int epoch=0;epoch<1200;epoch++) {
        double loss_sum=0;
        for (int j=0;j<9;j++) {
            int i=(j+epoch)%9;
            float loss=-1;
            CHECK(nt_spa_agent_imitate(&a,&fixture[i],labels[i].kind,&loss)==NT_SPA_OK && isfinite(loss) && loss>=0,
                  "finite supervised training");
            loss_sum+=loss;
        }
        if (!epoch) first=loss_sum;
        if (epoch==1199) last=loss_sum;
    }
    CHECK(last<first*.1,"imitation loss reduced tenfold");
    unsigned matches=0;
    for (int i=0;i<9;i++) {
        nt_spa_decision d;
        CHECK(nt_spa_agent_select(&a,&fixture[i],&d)==NT_SPA_OK,"learned imitation selection");
        matches += d.action.kind==labels[i].kind;
    }
    CHECK(matches==9,"learned policy reproduces all controlled legacy labels");
    CHECK(a.imitation_updates==10800 && a.decisions==0 && a.observations==0,"imitation isolated from life history");
    unsigned heldout=0,heldout_matches=0;
    const unsigned counts[3]={3,5,7};
    for(unsigned ci=0;ci<3;ci++) for(unsigned ti=0;ti<3;ti++)
        for(unsigned ph=0;ph<3;ph++) for(unsigned m=0;m<2;m++) for(unsigned side=0;side<2;side++) {
            unsigned n=counts[ci],target=ti==0 ? 0 : ti==1 ? n/2 : n-1;
            float phase=.5f*ph,mean=m ? 1.5f : .5f;
            float ratio=.52f+.18f*(1-phase)+(side ? .2f : -.2f);
            nt_spa_observation fresh=observation(target,n,ratio*mean);
            fresh.mean_sentence_score=mean; fresh.phase_lock=phase;
            nt_spa_action label; nt_spa_decision decision;
            CHECK(nt_spa_agent_legacy(&fresh,&label)==NT_SPA_OK,"heldout teacher label");
            CHECK(nt_spa_agent_select(&a,&fresh,&decision)==NT_SPA_OK,"heldout learned selection");
            heldout++; heldout_matches+=label.kind==decision.action.kind;
        }
    printf("SPA_IMITATION_HELDOUT matched=%u total=%u score_ratio_margin=.2\n",heldout_matches,heldout);
    CHECK(heldout_matches==heldout,"imitation generalizes to heldout counts phases and score scales");
    nt_spa_agent before=a;
    float loss=123;
    CHECK(nt_spa_agent_imitate(&a,&fixture[0],NT_SPA_RESEED_LEFT,&loss)!=NT_SPA_OK && SAME(a,before) && loss==123,
          "imitation cannot fit unavailable action");
    printf("SPA_IMITATION labels=9 matched=%u updates=10800 first_loss=%.9g last_loss=%.9g\n",matches,first/9,last/9);
    return 1;
}

static int test_memory(void) {
    nt_spa_agent a;
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,.2f),"memory life");
    for (unsigned i=0;i<19;i++) {
        nt_spa_observation o=observation(i%3,3,.1f);
        nt_spa_decision d; nt_spa_receipt r; nt_spa_consequence c=outcome(.25f,.75f);
        CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK && nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&r)==NT_SPA_OK,
              "fill persistent history");
    }
    CHECK(a.history_count==NT_SPA_AGENT_HISTORY && a.memory_observations==19,"history wraps while memory counts");
    nt_spa_agent before=a;
    CHECK(nt_spa_agent_reset_memory(&a)==NT_SPA_OK,"temporal reset");
    CHECK(SAME(a.config,before.config) && a.config_hash==before.config_hash && SAME(a.policy,before.policy),
          "reset preserves config and acquired policy");
    CHECK(a.rng==before.rng && a.decisions==before.decisions && a.observations==before.observations
          && a.updates==before.updates && a.imitation_updates==before.imitation_updates && a.cancelled==before.cancelled,
          "reset preserves RNG and lifetime counters");
    CHECK(a.history_count==0 && a.history_head==0 && a.memory_observations==0
          && a.ema_reward==0 && a.ema_connectedness==0 && a.ema_novelty==0,"reset clears temporal statistics");
    nt_spa_receipt empty[NT_SPA_AGENT_HISTORY]; memset(empty,0,sizeof empty);
    CHECK(!memcmp(a.history,empty,sizeof empty),"reset clears recorded outcome memory");
    nt_spa_agent expected=before;
    expected.history_count=0; expected.history_head=0; expected.memory_observations=0;
    expected.ema_reward=0; expected.ema_connectedness=0; expected.ema_novelty=0;
    memset(expected.history,0,sizeof expected.history);
    CHECK(SAME(a,expected),"reset changes only the explicitly named temporal fields");
    CHECK(nt_spa_agent_validate(&a)==NT_SPA_OK,"reset state remains valid");
    return 1;
}

static int equal_files(const char *x,const char *y) {
    FILE *a=fopen(x,"rb"),*b=fopen(y,"rb");
    if (!a || !b) { if(a)fclose(a); if(b)fclose(b); return 0; }
    int ca,cb,ok=1;
    do { ca=fgetc(a); cb=fgetc(b); if(ca!=cb){ok=0;break;} } while(ca!=EOF);
    if(ferror(a)||ferror(b))ok=0;
    fclose(a); fclose(b); return ok;
}

static int test_persistence(void) {
    char path[]="/tmp/notorch-spa-life-XXXXXX",other[]="/tmp/notorch-spa-copy-XXXXXX";
    int fd=mkstemp(path),fd2=mkstemp(other);
    CHECK(fd>=0 && fd2>=0,"temporary lives created"); close(fd); close(fd2);
    nt_spa_agent a,b;
    CHECK(initialized(&a,NT_SPA_AGENT_LEARNED,.5f),"saved life");
    for (unsigned step=0;step<40;step++) {
        nt_spa_observation o=observation(step%3,3,.1f+.1f*(step%7));
        nt_spa_decision d;
        CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"decision before interruption");
        CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"pending save");
        memset(&b,0x5a,sizeof b);
        CHECK(nt_spa_agent_load(&b,path)==NT_SPA_OK,"pending restore");
        CHECK(SAME(a,b) && nt_spa_agent_hash(&a)==nt_spa_agent_hash(&b),"pending state exact");
        CHECK(nt_spa_agent_save(&b,other)==NT_SPA_OK && equal_files(path,other),"canonical save exact");
        nt_spa_consequence c=outcome(.5f,.1f+.1f*(step%9)); nt_spa_receipt ra,rb;
        CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&ra)==NT_SPA_OK,"original pending credit");
        CHECK(nt_spa_agent_observe(&b,d.sequence,&d.action,&c,&rb)==NT_SPA_OK,"restored pending credit");
        CHECK(SAME(a,b) && SAME(ra,rb),"continuation acquired state exact");
    }
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"idle save");
    nt_spa_agent before=b;
    FILE *f=fopen(path,"r+b"); CHECK(f!=NULL,"open corruption fixture");
    CHECK(fseek(f,64,SEEK_SET)==0,"seek corrupt byte"); int byte=fgetc(f);
    CHECK(byte!=EOF && fseek(f,64,SEEK_SET)==0 && fputc(byte^1,f)!=EOF,"flip stored byte"); fclose(f);
    CHECK(nt_spa_agent_load(&b,path)!=NT_SPA_OK && SAME(b,before),"corrupted life refused transactionally");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"restore clean file");
    f=fopen(path,"ab"); CHECK(f && fputc(0,f)!=EOF,"append unexpected byte"); fclose(f);
    CHECK(nt_spa_agent_load(&b,path)!=NT_SPA_OK && SAME(b,before),"trailing byte refused transactionally");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"restore clean file for truncation");
    CHECK(truncate(path,19)==0,"truncate life");
    CHECK(nt_spa_agent_load(&b,path)!=NT_SPA_OK && SAME(b,before),"truncated life refused transactionally");
    CHECK(nt_spa_agent_load(&b,"/this/path/does/not/exist")!=NT_SPA_OK && SAME(b,before),"missing file refused transactionally");
    unlink(path); unlink(other);
    return 1;
}

int main(int argc,char **argv) {
    struct {const char *name; int (*run)(void);} gates[]={
        {"legacy",test_legacy},{"config",test_config},{"actions",test_actions},
        {"perception",test_perception},{"deterministic",test_deterministic},
        {"credit",test_credit},{"sequence",test_sequence},{"imitation",test_imitation},
        {"memory",test_memory},{"persistence",test_persistence}};
    unsigned ran=0;
    for (unsigned i=0;i<sizeof gates/sizeof gates[0];i++) {
        if(argc>1 && strcmp(argv[1],gates[i].name))continue;
        gate=gates[i].name;
        if(!gates[i].run())return 1;
        ran++; printf("PASS %s\n",gate);
    }
    if(!ran){fprintf(stderr,"FAIL unknown SPA gate\n");return 2;}
    printf("SPA_AGENT_OK gates=%u checks=%u\n",ran,checks);
    return 0;
}
