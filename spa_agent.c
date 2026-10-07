// SPA: Sentence Phonon Agent. Native policy and persistent acquired experience.
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "spa_agent.h"
#include "notorch.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SPA_WEIGHT_LIMIT 8.0f
#define SPA_FILE_CAPACITY 8192u

static int in_range(float x, float lo, float hi) {
    return isfinite(x) && x >= lo && x <= hi;
}
static float bounded(float x, float lo, float hi) {
    return x < lo ? lo : x > hi ? hi : x;
}
static int overlaps_agent(const nt_spa_agent *a,const void *p,size_t n) {
    uintptr_t x,y;
    if(!a || !p) return 0;
    x=(uintptr_t)a; y=(uintptr_t)p;
    return x<=y ? y-x<sizeof(*a) : x-y<n;
}
static uint32_t next_rng(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}
static float random_unit(uint32_t *state) {
    return (float)(next_rng(state) >> 8) * (1.0f / 16777216.0f);
}
static uint64_t fnv_byte(uint64_t h, unsigned char b) {
    return (h ^ b) * UINT64_C(1099511628211);
}
static uint64_t fnv_u32(uint64_t h, uint32_t v) {
    int k;
    for (k = 0; k < 4; ++k) h = fnv_byte(h, (unsigned char)(v >> (8*k)));
    return h;
}
static uint64_t fnv_float(uint64_t h, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return fnv_u32(h, bits);
}
static uint64_t config_fingerprint(const nt_spa_agent_config *c) {
    uint64_t h = UINT64_C(14695981039346656037);
    h = fnv_u32(h, (uint32_t)c->mode); h = fnv_u32(h, c->seed);
#define HASH_FLOAT(field) h = fnv_float(h, c->field)
    HASH_FLOAT(learning_rate); HASH_FLOAT(imitation_rate);
    HASH_FLOAT(exploration); HASH_FLOAT(memory_decay);
    HASH_FLOAT(reward_weights.local_connectedness);
    HASH_FLOAT(reward_weights.global_connectedness);
    HASH_FLOAT(reward_weights.coherence); HASH_FLOAT(reward_weights.novelty);
    HASH_FLOAT(reward_weights.repetition); HASH_FLOAT(reward_weights.collapse);
    HASH_FLOAT(reward_weights.continuity); HASH_FLOAT(cost_weight);
#undef HASH_FLOAT
    return h;
}

static int metrics_valid(const nt_spa_metrics *m) {
    return in_range(m->local_connectedness,0,1) &&
        in_range(m->global_connectedness,0,1) && in_range(m->coherence,0,1) &&
        in_range(m->novelty,0,1) && in_range(m->repetition,0,1) &&
        in_range(m->collapse,0,1) && in_range(m->continuity,0,1);
}
static int consequence_valid(const nt_spa_consequence *c) {
    return c && metrics_valid(&c->before) && metrics_valid(&c->after) &&
        in_range(c->regeneration_cost,0,1);
}
static int config_valid(const nt_spa_agent_config *c) {
    if (!c || c->mode < NT_SPA_AGENT_DISABLED || c->mode > NT_SPA_AGENT_LEARNED ||
        !c->seed || !in_range(c->learning_rate,0,1) ||
        !in_range(c->imitation_rate,0,1) || !in_range(c->exploration,0,1) ||
        !in_range(c->memory_decay,0,1) || c->memory_decay == 1 ||
        !metrics_valid(&c->reward_weights) || !in_range(c->cost_weight,0,1))
        return 0;
    return 1;
}
static int policy_valid(const nt_spa_policy *p) {
    int i,j;
    if (!p) return 0;
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        if(!in_range(p->b1[i],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT)) return 0;
        for(j=0;j<NT_SPA_AGENT_FEATURES;++j)
            if(!in_range(p->w1[i][j],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT)) return 0;
    }
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        if(!in_range(p->b2[i],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT)) return 0;
        for(j=0;j<NT_SPA_AGENT_HIDDEN;++j)
            if(!in_range(p->w2[i][j],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT)) return 0;
    }
    return 1;
}

void nt_spa_agent_config_default(nt_spa_agent_config *out) {
    nt_spa_agent_config c;
    if (!out) return;
    memset(&c,0,sizeof(c));
    c.mode=NT_SPA_AGENT_LEGACY; c.seed=1;
    c.learning_rate=.03f; c.imitation_rate=.05f;
    c.exploration=.1f; c.memory_decay=.8f;
    c.reward_weights.local_connectedness=.15f;
    c.reward_weights.global_connectedness=.15f;
    c.reward_weights.coherence=.2f; c.reward_weights.novelty=.2f;
    c.reward_weights.repetition=.15f; c.reward_weights.collapse=.1f;
    c.reward_weights.continuity=.05f; c.cost_weight=.05f;
    *out=c;
}
int nt_spa_agent_init(nt_spa_agent *out, const nt_spa_agent_config *config) {
    nt_spa_agent a;
    nt_spa_agent_config defaults;
    int i,j;
    if(!out) return NT_SPA_E_STATE;
    if(!config) { nt_spa_agent_config_default(&defaults); config=&defaults; }
    if(!config_valid(config)) return NT_SPA_E_CONFIG;
    memset(&a,0,sizeof(a));
    a.version=NT_SPA_AGENT_VERSION;
    a.perception_version=NT_SPA_AGENT_PERCEPTION_VERSION;
    a.reward_version=NT_SPA_AGENT_REWARD_VERSION;
    a.config=*config; a.config_hash=config_fingerprint(config); a.rng=config->seed;
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i)
        for(j=0;j<NT_SPA_AGENT_FEATURES;++j)
            a.policy.w1[i][j]=.2f*(random_unit(&a.rng)-.5f);
    // Zero action heads give an initial KEEP tie while seeded hidden features
    // supply a reproducible basis for imitation and outcome learning.
    *out=a;
    return NT_SPA_OK;
}

int nt_spa_observation_validate(const nt_spa_observation *o) {
    int i;
    if(!o || !o->sentence_count || o->sentence_count>NT_SPA_AGENT_MAX_SENTENCES ||
        o->sentence_index>=o->sentence_count || o->reseed_count>1000000u ||
        !in_range(o->connectedness,0,1) || !in_range(o->left_similarity,-1,1) ||
        !in_range(o->right_similarity,-1,1) || !in_range(o->coherence,0,1) ||
        !in_range(o->novelty,0,1) || !in_range(o->repetition,0,1) ||
        !in_range(o->phase_lock,0,1) || !in_range(o->sentence_score,0,1e6f) ||
        !in_range(o->mean_sentence_score,0,1e6f) ||
        !in_range(o->temperature,0,16) || o->temperature==0)
        return NT_SPA_E_OBSERVATION;
    for(i=0;i<NT_SPA_AGENT_EMBED;++i)
        if(!in_range(o->embedding[i],-1,1)) return NT_SPA_E_OBSERVATION;
    return NT_SPA_OK;
}
int nt_spa_action_validate(const nt_spa_action *a,const nt_spa_observation *o) {
    if(nt_spa_observation_validate(o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    if(!a || a->target!=o->sentence_index) return NT_SPA_E_ACTION;
    if(a->kind==NT_SPA_KEEP)
        return a->source==NT_SPA_AGENT_NO_SOURCE ? NT_SPA_OK : NT_SPA_E_ACTION;
    if(a->kind==NT_SPA_RESEED_LEFT)
        return a->target>0 && a->source==a->target-1 ? NT_SPA_OK : NT_SPA_E_ACTION;
    if(a->kind==NT_SPA_RESEED_RIGHT)
        return a->target+1<o->sentence_count && a->source==a->target+1 ?
            NT_SPA_OK : NT_SPA_E_ACTION;
    return NT_SPA_E_ACTION;
}
static nt_spa_action make_action(nt_spa_action_kind kind,uint32_t target) {
    nt_spa_action a;
    memset(&a,0,sizeof(a)); a.kind=kind; a.target=target;
    a.source=kind==NT_SPA_KEEP ? NT_SPA_AGENT_NO_SOURCE :
        kind==NT_SPA_RESEED_LEFT ? target-1 : target+1;
    return a;
}
int nt_spa_agent_legacy(const nt_spa_observation *o,nt_spa_action *action) {
    nt_spa_action_kind kind=NT_SPA_KEEP;
    if(!action) return NT_SPA_E_ACTION;
    if(nt_spa_observation_validate(o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    if(o->sentence_score < o->mean_sentence_score*(.52f+.18f*(1.0f-o->phase_lock))) {
        if(o->sentence_index>0) kind=NT_SPA_RESEED_LEFT;
        else if(o->sentence_count>1) kind=NT_SPA_RESEED_RIGHT;
    }
    *action=make_action(kind,o->sentence_index);
    return NT_SPA_OK;
}

static float cosine(const float *a,const float *b,uint32_t dim) {
    double dot=0,na=0,nb=0;
    uint32_t d;
    for(d=0;d<dim;++d) {
        dot+=(double)a[d]*b[d]; na+=(double)a[d]*a[d]; nb+=(double)b[d]*b[d];
    }
    if(na==0 || nb==0) return 0;
    return bounded((float)(dot/sqrt(na*nb)),-1,1);
}
int nt_spa_agent_perceive(const float *e,uint32_t count,uint32_t dim,
    uint32_t target,float phase_lock,float temperature,uint32_t reseeds,
    nt_spa_observation *out) {
    nt_spa_observation o;
    float *others=NULL;
    double sum_score=0,adjacent=0,projected[NT_SPA_AGENT_EMBED]={0};
    uint32_t i,j,d,bucket_count[NT_SPA_AGENT_EMBED]={0};
    float max_cos=0;
    if(!out || !e || !count || count>NT_SPA_AGENT_MAX_SENTENCES || !dim ||
        dim>NT_SPA_AGENT_MAX_DIM || target>=count || !in_range(phase_lock,0,1) ||
        !in_range(temperature,0,16) || temperature==0 || reseeds>1000000u)
        return NT_SPA_E_OBSERVATION;
    for(i=0;i<count;++i) for(d=0;d<dim;++d)
        if(!in_range(e[(size_t)i*dim+d],-1e6f,1e6f)) return NT_SPA_E_OBSERVATION;
    memset(&o,0,sizeof(o));
    o.sentence_count=count; o.sentence_index=target; o.phase_lock=phase_lock;
    o.temperature=temperature; o.reseed_count=reseeds;
    for(d=0;d<dim;++d) {
        projected[d%NT_SPA_AGENT_EMBED]+=e[(size_t)target*dim+d];
        ++bucket_count[d%NT_SPA_AGENT_EMBED];
    }
    for(d=0;d<NT_SPA_AGENT_EMBED;++d)
        o.embedding[d]=bucket_count[d] ? tanhf((float)(projected[d]/bucket_count[d])) : 0;
    if(count>1) {
        others=(float*)malloc((size_t)(count-1)*dim*sizeof(float));
        if(!others) return NT_SPA_E_MEMORY;
        for(i=0;i<count;++i) {
            uint32_t row=0;
            float score;
            for(j=0;j<count;++j) if(j!=i) {
                memcpy(others+(size_t)row*dim,e+(size_t)j*dim,(size_t)dim*sizeof(float));
                ++row;
            }
            score=nt_spa_connectedness(e+(size_t)i*dim,(int)dim,others,(int)(count-1));
            // Valid bounded inputs always give a positive softmax maximum.
            // The unchanged helper uses zero for its allocation refusal.
            if(score==0) { free(others); return NT_SPA_E_MEMORY; }
            if(!in_range(score,0,1)) { free(others); return NT_SPA_E_OBSERVATION; }
            sum_score+=score;
            if(i==target) o.connectedness=o.sentence_score=score;
            if(i>0) adjacent+=(cosine(e+(size_t)(i-1)*dim,e+(size_t)i*dim,dim)+1)*.5;
            if(i!=target) {
                float c=cosine(e+(size_t)target*dim,e+(size_t)i*dim,dim);
                if(c>max_cos) max_cos=c;
            }
        }
        free(others);
        o.coherence=(float)(adjacent/(count-1));
    } else o.coherence=1;
    o.mean_sentence_score=(float)(sum_score/count);
    o.novelty=1-max_cos; o.repetition=max_cos;
    if(target>0) o.left_similarity=cosine(e+(size_t)target*dim,e+(size_t)(target-1)*dim,dim);
    if(target+1<count) o.right_similarity=cosine(e+(size_t)target*dim,e+(size_t)(target+1)*dim,dim);
    if(nt_spa_observation_validate(&o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    *out=o;
    return NT_SPA_OK;
}

static void features(const nt_spa_agent *a,const nt_spa_observation *o,float *x) {
    uint32_t i;
    float frequency[NT_SPA_AGENT_ACTIONS]={0};
    const nt_spa_receipt *last=NULL;
    for(i=0;i<NT_SPA_AGENT_EMBED;++i) x[i]=o->embedding[i];
    x[4]=o->connectedness;
    x[5]=o->sentence_index>0 ? o->left_similarity : 0;
    x[6]=o->sentence_index+1<o->sentence_count ? o->right_similarity : 0;
    x[7]=o->coherence; x[8]=o->novelty; x[9]=o->repetition; x[10]=o->phase_lock;
    x[11]=tanhf(4*(o->sentence_score/(o->mean_sentence_score+1e-6f)-
        (.52f+.18f*(1-o->phase_lock))));
    x[12]=o->sentence_count>1 ? (float)o->sentence_index/(o->sentence_count-1) : 0;
    x[13]=(float)o->reseed_count/(o->reseed_count+1.0f);
    x[14]=o->temperature/16;
    x[15]=o->sentence_index>0 ? 1.0f : 0.0f;
    x[16]=o->sentence_index+1<o->sentence_count ? 1.0f : 0.0f;
    x[17]=o->mean_sentence_score/(o->mean_sentence_score+1.0f);
    x[18]=o->sentence_score/(o->sentence_score+1.0f);
    x[19]=a->ema_reward; x[20]=a->ema_connectedness; x[21]=a->ema_novelty;
    for(i=0;i<a->history_count;++i) frequency[a->history[i].action.kind]+=1;
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i)
        x[22+i]=a->history_count ? frequency[i]/a->history_count : 0;
    if(a->history_count) last=&a->history[(a->history_head+NT_SPA_AGENT_HISTORY-1)%NT_SPA_AGENT_HISTORY];
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) x[25+i]=last && (uint32_t)last->action.kind==i ? 1.0f : 0.0f;
    x[28]=last ? last->reward : 0;
}
static void forward(const nt_spa_policy *p,const float *x,float *hidden,float *scores) {
    int i,j;
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        float z=p->b1[i];
        for(j=0;j<NT_SPA_AGENT_FEATURES;++j) z+=p->w1[i][j]*x[j];
        hidden[i]=tanhf(z);
    }
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        float z=p->b2[i];
        for(j=0;j<NT_SPA_AGENT_HIDDEN;++j) z+=p->w2[i][j]*hidden[j];
        scores[i]=z;
    }
}
static int available(int kind,const nt_spa_observation *o) {
    return kind==NT_SPA_KEEP || (kind==NT_SPA_RESEED_LEFT && o->sentence_index>0) ||
        (kind==NT_SPA_RESEED_RIGHT && o->sentence_index+1<o->sentence_count);
}
static void select_unchecked(const nt_spa_agent *a,const nt_spa_observation *o,
    uint32_t rng_before,nt_spa_decision *d,uint32_t *rng_after) {
    uint32_t rng=rng_before;
    int k,best=NT_SPA_KEEP;
    memset(d,0,sizeof(*d)); d->observation=*o;
    d->rng_before=rng_before;
    d->action=make_action(NT_SPA_KEEP,o->sentence_index);
    if(!a || a->config.mode==NT_SPA_AGENT_DISABLED) {
        *rng_after=rng; return;
    }
    d->sequence=a->decisions+1;
    features(a,o,d->features);
    forward(&a->policy,d->features,d->hidden,d->scores);
    if(a->config.mode==NT_SPA_AGENT_LEGACY) nt_spa_agent_legacy(o,&d->action);
    else {
        // NT_SPA_SELECTION: strict greater-than makes equal predictions KEEP.
        for(k=1;k<NT_SPA_AGENT_ACTIONS;++k)
            if(available(k,o) && d->scores[k]>d->scores[best]) best=k;
        if(a->config.exploration>0 && random_unit(&rng)<a->config.exploration) {
            int valid[NT_SPA_AGENT_ACTIONS],n=0;
            for(k=0;k<NT_SPA_AGENT_ACTIONS;++k) if(available(k,o)) valid[n++]=k;
            best=valid[next_rng(&rng)%(uint32_t)n]; d->explored=1;
        }
        d->action=make_action((nt_spa_action_kind)best,o->sentence_index);
    }
    *rng_after=rng;
}

static float reward_for(const nt_spa_agent_config *c,const nt_spa_consequence *v) {
    double reward=0;
#define REWARD_PLUS(field) reward+=(double)c->reward_weights.field*(v->after.field-v->before.field)
#define REWARD_MINUS(field) reward+=(double)c->reward_weights.field*(v->before.field-v->after.field)
    // NT_SPA_CREDIT_SIGN: improvements earn positive credit, repetition costs it.
    REWARD_PLUS(local_connectedness); REWARD_PLUS(global_connectedness);
    REWARD_PLUS(coherence); REWARD_PLUS(novelty); REWARD_MINUS(repetition);
    REWARD_MINUS(collapse); REWARD_PLUS(continuity);
#undef REWARD_PLUS
#undef REWARD_MINUS
    reward-=(double)c->cost_weight*v->regeneration_cost;
    return bounded((float)reward,-1,1);
}
static int same_float(float a,float b) {
    return isfinite(a) && isfinite(b) && a==b;
}
static int receipt_valid(const nt_spa_receipt *r,const nt_spa_agent_config *c) {
    nt_spa_observation o;
    memset(&o,0,sizeof(o)); o.sentence_count=r->sentence_count;
    o.sentence_index=r->action.target; o.temperature=1;
    if(!r->sequence || nt_spa_action_validate(&r->action,&o)!=NT_SPA_OK ||
        !consequence_valid(&r->consequence) || !in_range(r->reward,-1,1) ||
        !in_range(r->predicted,-72,72) || !in_range(r->error,-73,73) ||
        (r->learned!=0 && r->learned!=1) ||
        !same_float(r->reward,reward_for(c,&r->consequence)) ||
        !same_float(r->error,r->reward-r->predicted)) return 0;
    if(r->learned!=(c->mode==NT_SPA_AGENT_LEARNED && c->learning_rate>0)) return 0;
    return 1;
}

int nt_spa_agent_validate(const nt_spa_agent *a) {
    uint32_t i,k;
    uint64_t accounted,previous=0;
    float exact_reward=0,exact_connectedness=0,exact_novelty=0;
    float reward_min=-1,reward_max=1,connectedness_min=0,connectedness_max=1;
    float novelty_min=0,novelty_max=1;
    nt_spa_receipt zero_receipt;
    nt_spa_decision zero_decision;
    if(!a || a->version!=NT_SPA_AGENT_VERSION ||
        a->perception_version!=NT_SPA_AGENT_PERCEPTION_VERSION ||
        a->reward_version!=NT_SPA_AGENT_REWARD_VERSION || !config_valid(&a->config) ||
        a->config_hash!=config_fingerprint(&a->config) || !policy_valid(&a->policy) ||
        !a->rng || (a->pending!=0 && a->pending!=1) ||
        a->observations>UINT64_MAX-a->cancelled) return NT_SPA_E_STATE;
    accounted=a->observations+a->cancelled;
    if((a->pending && accounted==UINT64_MAX) || a->decisions!=accounted+(uint64_t)a->pending ||
        a->updates!=(a->config.mode==NT_SPA_AGENT_LEARNED && a->config.learning_rate>0 ?
            a->observations : 0) || a->memory_observations>a->observations ||
        a->history_count!=(a->memory_observations<NT_SPA_AGENT_HISTORY ?
            (uint32_t)a->memory_observations : NT_SPA_AGENT_HISTORY) ||
        a->history_head!=a->memory_observations%NT_SPA_AGENT_HISTORY ||
        !in_range(a->ema_reward,-1,1) || !in_range(a->ema_connectedness,0,1) ||
        !in_range(a->ema_novelty,0,1)) return NT_SPA_E_STATE;
    if(!a->memory_observations && (a->ema_reward!=0 || a->ema_connectedness!=0 || a->ema_novelty!=0))
        return NT_SPA_E_STATE;
    if(a->config.mode==NT_SPA_AGENT_DISABLED && (a->decisions || a->imitation_updates))
        return NT_SPA_E_STATE;
    if(a->config.mode!=NT_SPA_AGENT_LEARNED && (a->updates || a->imitation_updates))
        return NT_SPA_E_STATE;
    memset(&zero_receipt,0,sizeof(zero_receipt));
    for(i=0;i<NT_SPA_AGENT_HISTORY;++i) {
        if(i>=a->history_count && a->history_count<NT_SPA_AGENT_HISTORY &&
            memcmp(&a->history[i],&zero_receipt,sizeof(zero_receipt))) return NT_SPA_E_STATE;
    }
    for(i=0;i<a->history_count;++i) {
        uint32_t slot=(a->history_head+NT_SPA_AGENT_HISTORY-a->history_count+i)%NT_SPA_AGENT_HISTORY;
        const nt_spa_receipt *r=&a->history[slot];
        if(!receipt_valid(r,&a->config) || r->sequence<=previous ||
            r->sequence>a->decisions-(uint64_t)a->pending) return NT_SPA_E_STATE;
        previous=r->sequence;
        if(a->memory_observations<=NT_SPA_AGENT_HISTORY) {
            if(i==0) {
                exact_reward=r->reward;
                exact_connectedness=r->consequence.after.local_connectedness;
                exact_novelty=r->consequence.after.novelty;
            } else {
                float d=a->config.memory_decay;
                exact_reward=bounded(d*exact_reward+(1-d)*r->reward,-1,1);
                exact_connectedness=bounded(d*exact_connectedness+(1-d)*r->consequence.after.local_connectedness,0,1);
                exact_novelty=bounded(d*exact_novelty+(1-d)*r->consequence.after.novelty,0,1);
            }
        } else {
            float d=a->config.memory_decay;
            reward_min=bounded(d*reward_min+(1-d)*r->reward,-1,1);
            reward_max=bounded(d*reward_max+(1-d)*r->reward,-1,1);
            connectedness_min=bounded(d*connectedness_min+(1-d)*r->consequence.after.local_connectedness,0,1);
            connectedness_max=bounded(d*connectedness_max+(1-d)*r->consequence.after.local_connectedness,0,1);
            novelty_min=bounded(d*novelty_min+(1-d)*r->consequence.after.novelty,0,1);
            novelty_max=bounded(d*novelty_max+(1-d)*r->consequence.after.novelty,0,1);
        }
    }
    if(a->memory_observations<=NT_SPA_AGENT_HISTORY) {
        if(!same_float(a->ema_reward,exact_reward) ||
            !same_float(a->ema_connectedness,exact_connectedness) ||
            !same_float(a->ema_novelty,exact_novelty)) return NT_SPA_E_STATE;
    } else if(!in_range(a->ema_reward,reward_min,reward_max) ||
        !in_range(a->ema_connectedness,connectedness_min,connectedness_max) ||
        !in_range(a->ema_novelty,novelty_min,novelty_max)) return NT_SPA_E_STATE;
    if(a->pending) {
        nt_spa_agent base;
        nt_spa_decision expected;
        uint32_t rng_after;
        const nt_spa_decision *p=&a->pending_decision;
        if(!p->rng_before || p->sequence!=a->decisions ||
            nt_spa_observation_validate(&p->observation)!=NT_SPA_OK ||
            nt_spa_action_validate(&p->action,&p->observation)!=NT_SPA_OK ||
            (p->explored!=0 && p->explored!=1)) return NT_SPA_E_STATE;
        base=*a; --base.decisions;
        select_unchecked(&base,&p->observation,p->rng_before,&expected,&rng_after);
        if(rng_after!=a->rng || expected.action.kind!=p->action.kind ||
            expected.action.source!=p->action.source || expected.explored!=p->explored)
            return NT_SPA_E_STATE;
        for(k=0;k<NT_SPA_AGENT_FEATURES;++k)
            if(!same_float(p->features[k],expected.features[k])) return NT_SPA_E_STATE;
        for(k=0;k<NT_SPA_AGENT_HIDDEN;++k)
            if(!same_float(p->hidden[k],expected.hidden[k])) return NT_SPA_E_STATE;
        for(k=0;k<NT_SPA_AGENT_ACTIONS;++k)
            if(!same_float(p->scores[k],expected.scores[k])) return NT_SPA_E_STATE;
    } else {
        memset(&zero_decision,0,sizeof(zero_decision));
        if(memcmp(&a->pending_decision,&zero_decision,sizeof(zero_decision))) return NT_SPA_E_STATE;
    }
    return NT_SPA_OK;
}

int nt_spa_agent_select(const nt_spa_agent *a,const nt_spa_observation *o,
    nt_spa_decision *decision) {
    nt_spa_decision d;
    uint32_t rng_after;
    if(!decision) return NT_SPA_E_ACTION;
    if(overlaps_agent(a,decision,sizeof(*decision))) return NT_SPA_E_STATE;
    if(nt_spa_observation_validate(o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    if(a && nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(a && a->pending) return NT_SPA_E_PENDING;
    if(a && a->config.mode!=NT_SPA_AGENT_DISABLED && a->decisions==UINT64_MAX) return NT_SPA_E_STATE;
    select_unchecked(a,o,a?a->rng:0,&d,&rng_after);
    *decision=d;
    return NT_SPA_OK;
}
int nt_spa_agent_choose(nt_spa_agent *a,const nt_spa_observation *o,
    nt_spa_decision *decision) {
    nt_spa_decision d;
    uint32_t rng_after;
    int status;
    if(!decision) return NT_SPA_E_ACTION;
    if(overlaps_agent(a,decision,sizeof(*decision))) return NT_SPA_E_STATE;
    status=nt_spa_agent_select(a,o,&d);
    if(status!=NT_SPA_OK) return status;
    if(a && a->config.mode!=NT_SPA_AGENT_DISABLED) {
        select_unchecked(a,o,a->rng,&d,&rng_after);
        a->rng=rng_after; ++a->decisions; a->pending=1; a->pending_decision=d;
    }
    *decision=d;
    return NT_SPA_OK;
}

static void train_policy(nt_spa_policy *p,const float *x,const float *hidden,
    const float *output_gradient,float rate) {
    float hidden_gradient[NT_SPA_AGENT_HIDDEN];
    int i,j;
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        float g=0;
        for(j=0;j<NT_SPA_AGENT_ACTIONS;++j) g+=output_gradient[j]*p->w2[j][i];
        hidden_gradient[i]=g*(1-hidden[i]*hidden[i]);
    }
    // NT_SPA_LEARN_RATE: ascent along target-minus-prediction gradient.
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        p->b2[i]=bounded(p->b2[i]+rate*output_gradient[i],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT);
        for(j=0;j<NT_SPA_AGENT_HIDDEN;++j)
            p->w2[i][j]=bounded(p->w2[i][j]+rate*output_gradient[i]*hidden[j],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT);
    }
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        p->b1[i]=bounded(p->b1[i]+rate*hidden_gradient[i],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT);
        for(j=0;j<NT_SPA_AGENT_FEATURES;++j)
            p->w1[i][j]=bounded(p->w1[i][j]+rate*hidden_gradient[i]*x[j],-SPA_WEIGHT_LIMIT,SPA_WEIGHT_LIMIT);
    }
}
int nt_spa_agent_observe(nt_spa_agent *a,uint64_t sequence,
    const nt_spa_action *executed_action,
    const nt_spa_consequence *c,nt_spa_receipt *receipt) {
    nt_spa_agent next;
    nt_spa_receipt r;
    const nt_spa_decision *p;
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(overlaps_agent(a,receipt,sizeof(*receipt))) return NT_SPA_E_STATE;
    if(!a->pending) return NT_SPA_E_PENDING;
    if(sequence!=a->pending_decision.sequence) return NT_SPA_E_SEQUENCE;
    if(nt_spa_action_validate(executed_action,&a->pending_decision.observation)!=NT_SPA_OK ||
        executed_action->kind!=a->pending_decision.action.kind ||
        executed_action->target!=a->pending_decision.action.target ||
        executed_action->source!=a->pending_decision.action.source) return NT_SPA_E_ACTION;
    if(!consequence_valid(c)) return NT_SPA_E_CONSEQUENCE;
    if(a->observations==UINT64_MAX || a->memory_observations==UINT64_MAX) return NT_SPA_E_STATE;
    next=*a; p=&a->pending_decision;
    memset(&r,0,sizeof(r)); r.sequence=sequence; r.action=p->action;
    r.sentence_count=p->observation.sentence_count; r.consequence=*c;
    r.reward=reward_for(&a->config,c); r.predicted=p->scores[p->action.kind];
    r.error=r.reward-r.predicted;
    if(a->config.mode==NT_SPA_AGENT_LEARNED && a->config.learning_rate>0) {
        float gradient[NT_SPA_AGENT_ACTIONS]={0};
        gradient[p->action.kind]=bounded(r.error,-1,1);
        train_policy(&next.policy,p->features,p->hidden,gradient,a->config.learning_rate);
        ++next.updates; r.learned=1;
    }
    next.history[next.history_head]=r;
    next.history_head=(next.history_head+1)%NT_SPA_AGENT_HISTORY;
    if(next.history_count<NT_SPA_AGENT_HISTORY) ++next.history_count;
    if(!next.memory_observations) {
        next.ema_reward=r.reward;
        next.ema_connectedness=c->after.local_connectedness;
        next.ema_novelty=c->after.novelty;
    } else {
        float d=a->config.memory_decay;
        next.ema_reward=bounded(d*next.ema_reward+(1-d)*r.reward,-1,1);
        next.ema_connectedness=bounded(d*next.ema_connectedness+(1-d)*c->after.local_connectedness,0,1);
        next.ema_novelty=bounded(d*next.ema_novelty+(1-d)*c->after.novelty,0,1);
    }
    ++next.observations; ++next.memory_observations;
    next.pending=0; memset(&next.pending_decision,0,sizeof(next.pending_decision));
    if(nt_spa_agent_validate(&next)!=NT_SPA_OK) return NT_SPA_E_STATE;
    *a=next;
    if(receipt) *receipt=r;
    return NT_SPA_OK;
}
int nt_spa_agent_cancel(nt_spa_agent *a,uint64_t sequence) {
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(!a->pending) return NT_SPA_E_PENDING;
    if(sequence!=a->pending_decision.sequence) return NT_SPA_E_SEQUENCE;
    a->pending=0; ++a->cancelled;
    memset(&a->pending_decision,0,sizeof(a->pending_decision));
    return NT_SPA_OK;
}
int nt_spa_agent_imitate(nt_spa_agent *a,const nt_spa_observation *o,
    nt_spa_action_kind label,float *loss) {
    nt_spa_agent next;
    nt_spa_action action;
    float x[NT_SPA_AGENT_FEATURES],h[NT_SPA_AGENT_HIDDEN],s[NT_SPA_AGENT_ACTIONS];
    float grad[NT_SPA_AGENT_ACTIONS]={0},mx=-FLT_MAX,sum=0,value;
    int i;
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(overlaps_agent(a,loss,sizeof(*loss))) return NT_SPA_E_STATE;
    if(a->pending) return NT_SPA_E_PENDING;
    if(a->config.mode!=NT_SPA_AGENT_LEARNED || a->imitation_updates==UINT64_MAX) return NT_SPA_E_STATE;
    if(nt_spa_observation_validate(o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    if(label<NT_SPA_KEEP || label>NT_SPA_RESEED_RIGHT) return NT_SPA_E_ACTION;
    action=make_action(label,o->sentence_index);
    if(nt_spa_action_validate(&action,o)!=NT_SPA_OK) return NT_SPA_E_ACTION;
    features(a,o,x); forward(&a->policy,x,h,s);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(available(i,o) && s[i]>mx) mx=s[i];
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(available(i,o)) { grad[i]=expf(s[i]-mx); sum+=grad[i]; }
    value=(mx-s[label])+logf(sum);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) grad[i]=(i==(int)label?1.0f:0.0f)-grad[i]/sum;
    next=*a; train_policy(&next.policy,x,h,grad,a->config.imitation_rate);
    ++next.imitation_updates;
    if(nt_spa_agent_validate(&next)!=NT_SPA_OK) return NT_SPA_E_STATE;
    *a=next; if(loss) *loss=value;
    return NT_SPA_OK;
}
int nt_spa_agent_reset_memory(nt_spa_agent *a) {
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(a->pending) return NT_SPA_E_PENDING;
    a->memory_observations=0; a->history_count=0; a->history_head=0;
    memset(a->history,0,sizeof(a->history));
    a->ema_reward=0; a->ema_connectedness=0; a->ema_novelty=0;
    return NT_SPA_OK;
}
int nt_spa_agent_set_policy(nt_spa_agent *a,const nt_spa_policy *policy) {
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(a->pending) return NT_SPA_E_PENDING;
    if(!policy_valid(policy)) return NT_SPA_E_STATE;
    a->policy=*policy;
    return NT_SPA_OK;
}

static int buffer_overlap(const void *a,size_t a_size,const void *b,size_t b_size) {
    uintptr_t x,y;
    if(!a || !b) return 0;
    x=(uintptr_t)a; y=(uintptr_t)b;
    return x<=y ? y-x<a_size : x-y<b_size;
}
static uint32_t experience_mask(const nt_spa_experience *e) {
    uint32_t mask=1u<<NT_SPA_KEEP;
    if(e->sentence_index>0) mask|=1u<<NT_SPA_RESEED_LEFT;
    if(e->sentence_index+1<e->sentence_count) mask|=1u<<NT_SPA_RESEED_RIGHT;
    return mask;
}
static int learned_idle(const nt_spa_agent *a) {
    if(nt_spa_agent_validate(a)!=NT_SPA_OK || a->config.mode!=NT_SPA_AGENT_LEARNED)
        return NT_SPA_E_STATE;
    return a->pending ? NT_SPA_E_PENDING : NT_SPA_OK;
}
int nt_spa_experience_validate(const nt_spa_experience *e) {
    const float *x;
    uint32_t i,n;
    int last=-1,has_history=0,valid_frequency=0;
    float maximum_fraction=1000000.0f/1000001.0f;
    if(!e || e->version!=NT_SPA_EXPERIENCE_VERSION || !e->source_life_hash ||
        !e->sentence_count || e->sentence_count>NT_SPA_AGENT_MAX_SENTENCES ||
        e->sentence_index>=e->sentence_count) return NT_SPA_E_EXPERIENCE;
    x=e->features;
    for(i=0;i<NT_SPA_AGENT_FEATURES;++i)
        if(!in_range(x[i],-1,1)) return NT_SPA_E_EXPERIENCE;
    for(i=4;i<NT_SPA_AGENT_FEATURES;++i)
        if(i!=5 && i!=6 && i!=11 && i!=19 && i!=28 && x[i]<0)
            return NT_SPA_E_EXPERIENCE;
    if(x[12]!=(e->sentence_count>1 ? (float)e->sentence_index/(e->sentence_count-1) : 0) ||
        x[15]!=(e->sentence_index>0 ? 1.0f : 0.0f) ||
        x[16]!=(e->sentence_index+1<e->sentence_count ? 1.0f : 0.0f) ||
        (e->sentence_index==0 && x[5]!=0) ||
        (e->sentence_index+1==e->sentence_count && x[6]!=0) ||
        x[13]>maximum_fraction || x[17]>maximum_fraction || x[18]>maximum_fraction)
        return NT_SPA_E_EXPERIENCE;
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        if(x[25+i]!=0 && x[25+i]!=1) return NT_SPA_E_EXPERIENCE;
        if(x[25+i]==1) { if(last>=0) return NT_SPA_E_EXPERIENCE; last=(int)i; }
        if(x[22+i]!=0) has_history=1;
    }
    if(!has_history) {
        if(last>=0 || x[19]!=0 || x[20]!=0 || x[21]!=0 || x[28]!=0)
            return NT_SPA_E_EXPERIENCE;
    } else {
        if(last<0 || x[22+last]==0) return NT_SPA_E_EXPERIENCE;
        // Every frequency vector comes from the bounded one-to-eight ring.
        for(n=1;n<=NT_SPA_AGENT_HISTORY && !valid_frequency;++n) {
            unsigned total=0;
            int matches=1;
            for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
                unsigned count=(unsigned)floorf(x[22+i]*n+.5f);
                if(count>n || (float)count/n!=x[22+i]) matches=0;
                total+=count;
            }
            if(matches && total==n) valid_frequency=1;
        }
        if(!valid_frequency) return NT_SPA_E_EXPERIENCE;
    }
    return NT_SPA_OK;
}
int nt_spa_agent_capture_experience(const nt_spa_agent *a,const nt_spa_observation *o,
    nt_spa_experience *out) {
    nt_spa_experience e;
    int status=learned_idle(a);
    if(status!=NT_SPA_OK) return status;
    if(!out || overlaps_agent(a,out,sizeof(*out)) ||
        buffer_overlap(o,sizeof(*o),out,sizeof(*out))) return NT_SPA_E_EXPERIENCE;
    if(nt_spa_observation_validate(o)!=NT_SPA_OK) return NT_SPA_E_OBSERVATION;
    memset(&e,0,sizeof(e));
    e.version=NT_SPA_EXPERIENCE_VERSION;
    e.sentence_index=o->sentence_index; e.sentence_count=o->sentence_count;
    e.source_life_hash=nt_spa_agent_hash(a);
    features(a,o,e.features);
    status=nt_spa_experience_validate(&e);
    if(status!=NT_SPA_OK) return status;
    *out=e;
    return NT_SPA_OK;
}
int nt_spa_agent_score_experience(const nt_spa_agent *a,const nt_spa_experience *e,
    nt_spa_readout *out) {
    nt_spa_readout readout;
    float hidden[NT_SPA_AGENT_HIDDEN];
    int best=NT_SPA_KEEP,i,status=learned_idle(a);
    if(status!=NT_SPA_OK) return status;
    if(nt_spa_experience_validate(e)!=NT_SPA_OK) return NT_SPA_E_EXPERIENCE;
    if(!out || overlaps_agent(a,out,sizeof(*out)) ||
        buffer_overlap(e,sizeof(*e),out,sizeof(*out))) return NT_SPA_E_ACTION;
    memset(&readout,0,sizeof(readout));
    readout.action_mask=experience_mask(e);
    forward(&a->policy,e->features,hidden,readout.scores);
    for(i=1;i<NT_SPA_AGENT_ACTIONS;++i)
        if((readout.action_mask&(1u<<i)) && readout.scores[i]>readout.scores[best]) best=i;
    readout.action=make_action((nt_spa_action_kind)best,e->sentence_index);
    *out=readout;
    return NT_SPA_OK;
}
static int metrics_equal(const nt_spa_metrics *a,const nt_spa_metrics *b) {
    return a->local_connectedness==b->local_connectedness &&
        a->global_connectedness==b->global_connectedness && a->coherence==b->coherence &&
        a->novelty==b->novelty && a->repetition==b->repetition &&
        a->collapse==b->collapse && a->continuity==b->continuity;
}
static double comparison_loss(const float *scores,const float *targets,uint32_t mask) {
    double loss=0;
    unsigned count=0,i;
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(mask&(1u<<i)) {
        double error=fabs((double)scores[i]-targets[i]);
        loss+=error<=1 ? .5*error*error : error-.5;
        ++count;
    }
    return loss/count;
}
int nt_spa_agent_fit_comparison(nt_spa_agent *a,const nt_spa_experience *e,
    const nt_spa_comparison *comparison,float rate,nt_spa_comparison_receipt *out) {
    nt_spa_comparison_receipt r;
    nt_spa_policy next;
    nt_spa_observation bounds;
    nt_spa_alternative zero;
    float hidden[NT_SPA_AGENT_HIDDEN],post_hidden[NT_SPA_AGENT_HIDDEN];
    float gradient[NT_SPA_AGENT_ACTIONS]={0};
    unsigned count=0,i;
    uint32_t mask;
    int status=learned_idle(a);
    if(status!=NT_SPA_OK) return status;
    if(nt_spa_experience_validate(e)!=NT_SPA_OK) return NT_SPA_E_EXPERIENCE;
    if(!comparison || !in_range(rate,0,1)) return NT_SPA_E_COMPARISON;
    if(overlaps_agent(a,e,sizeof(*e)) || overlaps_agent(a,comparison,sizeof(*comparison)) ||
        overlaps_agent(a,out,sizeof(*out)) || buffer_overlap(e,sizeof(*e),comparison,sizeof(*comparison)) ||
        buffer_overlap(e,sizeof(*e),out,sizeof(*out)) ||
        buffer_overlap(comparison,sizeof(*comparison),out,sizeof(*out))) return NT_SPA_E_COMPARISON;
    mask=experience_mask(e);
    if(comparison->source_life_hash!=e->source_life_hash || comparison->action_mask!=mask ||
        comparison->horizon>NT_SPA_COMPARISON_MAX_HORIZON) return NT_SPA_E_COMPARISON;
    memset(&bounds,0,sizeof(bounds)); bounds.temperature=1;
    bounds.sentence_index=e->sentence_index; bounds.sentence_count=e->sentence_count;
    memset(&zero,0,sizeof(zero)); memset(&r,0,sizeof(r));
    r.source_life_hash=e->source_life_hash; r.action_mask=mask;
    r.horizon=comparison->horizon; r.learning_rate=rate;
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        const nt_spa_alternative *alternative=&comparison->alternatives[i];
        if(!(mask&(1u<<i))) {
            if(memcmp(alternative,&zero,sizeof(zero))) return NT_SPA_E_COMPARISON;
            continue;
        }
        if(alternative->action.kind!=(nt_spa_action_kind)i ||
            nt_spa_action_validate(&alternative->action,&bounds)!=NT_SPA_OK)
            return NT_SPA_E_ACTION;
        if(!consequence_valid(&alternative->consequence) ||
            !metrics_equal(&alternative->consequence.before,&comparison->alternatives[0].consequence.before))
            return NT_SPA_E_CONSEQUENCE;
        r.rewards[i]=reward_for(&a->config,&alternative->consequence);
        ++count;
    }
    forward(&a->policy,e->features,hidden,r.scores_before);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(mask&(1u<<i)) {
        // NT_SPA_COMPARISON_TARGET_SIGN: measured advantage over paired KEEP.
        r.targets[i]=r.rewards[i]-r.rewards[NT_SPA_KEEP];
        gradient[i]=bounded(r.targets[i]-r.scores_before[i],-1,1)/(float)count;
    }
    r.loss_before=comparison_loss(r.scores_before,r.targets,mask);
    next=a->policy;
    // NT_SPA_COMPARISON_UPDATE: all heads use the same pre-update weights.
    if(rate>0) train_policy(&next,e->features,hidden,gradient,rate);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) if(!(mask&(1u<<i))) {
        memcpy(next.w2[i],a->policy.w2[i],sizeof(next.w2[i]));
        next.b2[i]=a->policy.b2[i];
    }
    if(!policy_valid(&next)) return NT_SPA_E_STATE;
    forward(&next,e->features,post_hidden,r.scores_after);
    r.loss_after=comparison_loss(r.scores_after,r.targets,mask);
    a->policy=next;
    if(out) *out=r;
    return NT_SPA_OK;
}

// Canonical fields are explicitly encoded. No padding, enum representation or
// host byte order enters the file. One codec defines both directions.
typedef struct {
    unsigned char *data;
    size_t size,pos;
    int reading,failed;
} spa_codec;
static void codec_u32(spa_codec *c,uint32_t *v) {
    int k;
    if(c->failed || c->size-c->pos<4) { c->failed=1; return; }
    if(c->reading) {
        *v=0;
        for(k=0;k<4;++k) *v|=(uint32_t)c->data[c->pos+k]<<(8*k);
    } else for(k=0;k<4;++k) c->data[c->pos+k]=(unsigned char)(*v>>(8*k));
    c->pos+=4;
}
static void codec_u64(spa_codec *c,uint64_t *v) {
    uint32_t lo=(uint32_t)*v,hi=(uint32_t)(*v>>32);
    codec_u32(c,&lo); codec_u32(c,&hi);
    if(c->reading) *v=(uint64_t)lo|((uint64_t)hi<<32);
}
static void codec_float(spa_codec *c,float *f) {
    uint32_t v=0;
    if(!c->reading) memcpy(&v,f,sizeof(v));
    codec_u32(c,&v);
    if(c->reading) memcpy(f,&v,sizeof(v));
}
static void codec_int(spa_codec *c,int *v) {
    uint32_t x=(uint32_t)*v;
    codec_u32(c,&x);
    if(c->reading) {
        if(x>1) { c->failed=1; return; }
        *v=(int)x;
    }
}
static void codec_metrics(spa_codec *c,nt_spa_metrics *m) {
    codec_float(c,&m->local_connectedness); codec_float(c,&m->global_connectedness);
    codec_float(c,&m->coherence); codec_float(c,&m->novelty);
    codec_float(c,&m->repetition); codec_float(c,&m->collapse); codec_float(c,&m->continuity);
}
static void codec_config(spa_codec *c,nt_spa_agent_config *v) {
    uint32_t mode=(uint32_t)v->mode;
    codec_u32(c,&mode);
    if(c->reading) {
        if(mode>NT_SPA_AGENT_LEARNED) { c->failed=1; return; }
        v->mode=(nt_spa_agent_mode)mode;
    }
    codec_u32(c,&v->seed); codec_float(c,&v->learning_rate); codec_float(c,&v->imitation_rate);
    codec_float(c,&v->exploration); codec_float(c,&v->memory_decay);
    codec_metrics(c,&v->reward_weights); codec_float(c,&v->cost_weight);
}
static void codec_action(spa_codec *c,nt_spa_action *a) {
    uint32_t kind=(uint32_t)a->kind;
    codec_u32(c,&kind);
    if(c->reading) {
        if(kind>=NT_SPA_AGENT_ACTIONS) { c->failed=1; return; }
        a->kind=(nt_spa_action_kind)kind;
    }
    codec_u32(c,&a->target); codec_u32(c,&a->source);
}
static void codec_observation(spa_codec *c,nt_spa_observation *o) {
    int i;
    for(i=0;i<NT_SPA_AGENT_EMBED;++i) codec_float(c,&o->embedding[i]);
    codec_float(c,&o->connectedness); codec_float(c,&o->left_similarity);
    codec_float(c,&o->right_similarity); codec_float(c,&o->coherence);
    codec_float(c,&o->novelty); codec_float(c,&o->repetition); codec_float(c,&o->phase_lock);
    codec_float(c,&o->sentence_score); codec_float(c,&o->mean_sentence_score); codec_float(c,&o->temperature);
    codec_u32(c,&o->sentence_index); codec_u32(c,&o->sentence_count); codec_u32(c,&o->reseed_count);
}
static void codec_consequence(spa_codec *c,nt_spa_consequence *v) {
    codec_metrics(c,&v->before); codec_metrics(c,&v->after); codec_float(c,&v->regeneration_cost);
}
static void codec_receipt(spa_codec *c,nt_spa_receipt *v) {
    codec_u64(c,&v->sequence); codec_action(c,&v->action); codec_u32(c,&v->sentence_count);
    codec_consequence(c,&v->consequence); codec_float(c,&v->reward);
    codec_float(c,&v->predicted); codec_float(c,&v->error); codec_int(c,&v->learned);
}
static void codec_decision(spa_codec *c,nt_spa_decision *d) {
    int i;
    codec_observation(c,&d->observation);
    for(i=0;i<NT_SPA_AGENT_FEATURES;++i) codec_float(c,&d->features[i]);
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) codec_float(c,&d->hidden[i]);
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) codec_float(c,&d->scores[i]);
    codec_action(c,&d->action); codec_u64(c,&d->sequence);
    codec_u32(c,&d->rng_before); codec_int(c,&d->explored);
}
static void codec_agent(spa_codec *c,nt_spa_agent *a) {
    int i,j;
    codec_u32(c,&a->version); codec_u32(c,&a->perception_version); codec_u32(c,&a->reward_version);
    codec_config(c,&a->config); codec_u64(c,&a->config_hash);
    for(i=0;i<NT_SPA_AGENT_HIDDEN;++i) {
        for(j=0;j<NT_SPA_AGENT_FEATURES;++j) codec_float(c,&a->policy.w1[i][j]);
        codec_float(c,&a->policy.b1[i]);
    }
    for(i=0;i<NT_SPA_AGENT_ACTIONS;++i) {
        for(j=0;j<NT_SPA_AGENT_HIDDEN;++j) codec_float(c,&a->policy.w2[i][j]);
        codec_float(c,&a->policy.b2[i]);
    }
    codec_u32(c,&a->rng); codec_u64(c,&a->decisions); codec_u64(c,&a->observations);
    codec_u64(c,&a->updates); codec_u64(c,&a->imitation_updates); codec_u64(c,&a->cancelled);
    codec_u64(c,&a->memory_observations); codec_u32(c,&a->history_count); codec_u32(c,&a->history_head);
    for(i=0;i<NT_SPA_AGENT_HISTORY;++i) codec_receipt(c,&a->history[i]);
    codec_float(c,&a->ema_reward); codec_float(c,&a->ema_connectedness); codec_float(c,&a->ema_novelty);
    codec_int(c,&a->pending); codec_decision(c,&a->pending_decision);
}
static int representation_supported(void) {
    return sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128;
}
static uint64_t bytes_hash(const unsigned char *bytes,size_t n) {
    size_t i;
    uint64_t h=UINT64_C(14695981039346656037);
    for(i=0;i<n;++i) h=fnv_byte(h,bytes[i]);
    return h;
}
static int encode(const nt_spa_agent *a,unsigned char *bytes,size_t capacity,size_t *size) {
    nt_spa_agent copy;
    spa_codec c;
    uint32_t length=0;
    uint64_t checksum;
    if(!representation_supported()) return NT_SPA_E_FORMAT;
    if(nt_spa_agent_validate(a)!=NT_SPA_OK) return NT_SPA_E_STATE;
    if(capacity<12) return NT_SPA_E_FORMAT;
    memcpy(bytes,"NTSPA001",8);
    c.data=bytes; c.size=capacity; c.pos=8; c.reading=0; c.failed=0;
    codec_u32(&c,&length); copy=*a; codec_agent(&c,&copy);
    if(c.failed || c.pos>UINT32_MAX-8) return NT_SPA_E_FORMAT;
    length=(uint32_t)(c.pos+8);
    bytes[8]=(unsigned char)length; bytes[9]=(unsigned char)(length>>8);
    bytes[10]=(unsigned char)(length>>16); bytes[11]=(unsigned char)(length>>24);
    checksum=bytes_hash(bytes,c.pos); codec_u64(&c,&checksum);
    if(c.failed) return NT_SPA_E_FORMAT;
    *size=c.pos;
    return NT_SPA_OK;
}
int nt_spa_agent_save(const nt_spa_agent *a,const char *path) {
    unsigned char bytes[SPA_FILE_CAPACITY];
    size_t size,path_length,parent_length;
    FILE *f=NULL;
    char *temporary=NULL,*parent=NULL;
    const char *slash;
    struct stat parent_stat;
    int fd=-1,directory_fd=-1,created=0,renamed=0,saved_errno=0;
    int status=encode(a,bytes,sizeof(bytes),&size),directory_flags=O_RDONLY;
    if(status!=NT_SPA_OK) return status;
    if(!path || !*path) return NT_SPA_E_IO;
    path_length=strlen(path);
    if(path_length>SIZE_MAX-sizeof(".tmp.XXXXXX")) return NT_SPA_E_IO;
    slash=strrchr(path,'/');
    if(slash && slash[1]=='\0') { errno=EISDIR; return NT_SPA_E_IO; }
    parent_length=slash ? (slash==path ? 1 : (size_t)(slash-path)) : 1;
    temporary=(char*)malloc(path_length+sizeof(".tmp.XXXXXX"));
    parent=(char*)malloc(parent_length+1);
    if(!temporary || !parent) { free(temporary); free(parent); return NT_SPA_E_MEMORY; }
    memcpy(temporary,path,path_length);
    memcpy(temporary+path_length,".tmp.XXXXXX",sizeof(".tmp.XXXXXX"));
    if(slash) memcpy(parent,path,parent_length); else parent[0]='.';
    parent[parent_length]='\0';
#ifdef O_DIRECTORY
    directory_flags|=O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    directory_flags|=O_CLOEXEC;
#endif
    status=NT_SPA_E_IO;
    // Open and validate the directory before creating or replacing any file.
    directory_fd=open(parent,directory_flags);
    if(directory_fd<0) { saved_errno=errno; goto cleanup; }
    if(fstat(directory_fd,&parent_stat)!=0) { saved_errno=errno; goto cleanup; }
    if(!S_ISDIR(parent_stat.st_mode)) { saved_errno=ENOTDIR; goto cleanup; }
    fd=mkstemp(temporary);
    if(fd<0) { saved_errno=errno; goto cleanup; }
    created=1;
    f=fdopen(fd,"wb");
    if(!f) { saved_errno=errno; goto cleanup; }
    if(fwrite(bytes,1,size,f)!=size) { saved_errno=errno?errno:EIO; goto cleanup; }
    if(fflush(f)!=0) { saved_errno=errno; goto cleanup; }
    if(fsync(fd)!=0) { saved_errno=errno; goto cleanup; }
    if(fclose(f)!=0) { f=NULL; fd=-1; saved_errno=errno; goto cleanup; }
    f=NULL; fd=-1;
    if(rename(temporary,path)!=0) { saved_errno=errno; goto cleanup; }
    renamed=1;
    // NT_SPA_DIRECTORY_DURABILITY: persist the installed directory entry.
    if(fsync(directory_fd)!=0) { saved_errno=errno; goto cleanup; }
    status=NT_SPA_OK;
cleanup:
    if(f) {
        if(fclose(f)!=0 && !saved_errno) saved_errno=errno;
        fd=-1;
    }
    if(fd>=0 && close(fd)!=0 && !saved_errno) saved_errno=errno;
    if(created && !renamed) {
        if(unlink(temporary)!=0 && !saved_errno) saved_errno=errno;
    }
    if(directory_fd>=0 && close(directory_fd)!=0) {
        if(!saved_errno) saved_errno=errno;
        status=NT_SPA_E_IO;
    }
    free(temporary); free(parent);
    if(status!=NT_SPA_OK) errno=saved_errno?saved_errno:EIO;
    return status;
}
int nt_spa_agent_load(nt_spa_agent *a,const char *path) {
    unsigned char bytes[SPA_FILE_CAPACITY];
    nt_spa_agent next;
    spa_codec c;
    FILE *f;
    size_t n;
    uint32_t length=0;
    uint64_t stored=0,expected;
    int extra,bad;
    if(!a) return NT_SPA_E_STATE;
    if(!representation_supported()) return NT_SPA_E_FORMAT;
    if(!path || !*path) return NT_SPA_E_IO;
    f=fopen(path,"rb");
    if(!f) return NT_SPA_E_IO;
    n=fread(bytes,1,sizeof(bytes),f); extra=fgetc(f); bad=ferror(f);
    if(fclose(f)!=0) bad=1;
    if(bad) return NT_SPA_E_IO;
    if(extra!=EOF || n<20 || memcmp(bytes,"NTSPA001",8)) return NT_SPA_E_FORMAT;
    c.data=bytes; c.size=n; c.pos=8; c.reading=1; c.failed=0;
    codec_u32(&c,&length);
    if(length!=n) return NT_SPA_E_FORMAT;
    memset(&next,0,sizeof(next)); codec_agent(&c,&next);
    if(c.failed || c.pos+8!=n) return NT_SPA_E_FORMAT;
    expected=bytes_hash(bytes,c.pos); codec_u64(&c,&stored);
    if(c.failed || stored!=expected || nt_spa_agent_validate(&next)!=NT_SPA_OK)
        return NT_SPA_E_FORMAT;
    *a=next;
    return NT_SPA_OK;
}
uint64_t nt_spa_agent_hash(const nt_spa_agent *a) {
    unsigned char bytes[SPA_FILE_CAPACITY];
    size_t size;
    if(encode(a,bytes,sizeof(bytes),&size)!=NT_SPA_OK) return 0;
    return bytes_hash(bytes,size);
}
