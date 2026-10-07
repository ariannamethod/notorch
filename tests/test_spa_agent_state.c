/* Independent malformed-life, allocation and atomic-checkpoint gates.
 * Compile notorch.c separately with -Dcalloc=spa_test_calloc for the injected
 * allocation gate. The policy and this test use the ordinary C allocator. */
#define _POSIX_C_SOURCE 200809L
#include "spa_agent.h"
#include <dirent.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *gate;
static unsigned checks;
static char directory[]="/tmp/notorch-spa-state-XXXXXX";
static int fail_next_calloc, intercepted_failures;
#define CHECK(c,msg) do { ++checks; if(!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",gate,__LINE__,msg); return 0; } } while(0)
#define SAME(a,b) (memcmp(&(a),&(b),sizeof(a))==0)

void *spa_test_calloc(size_t count,size_t width) {
    if(fail_next_calloc) {
        fail_next_calloc=0;
        ++intercepted_failures;
        return NULL;
    }
    return calloc(count,width);
}

static nt_spa_observation observation(void) {
    nt_spa_observation o;
    memset(&o,0,sizeof o);
    o.sentence_count=3; o.sentence_index=1; o.temperature=1;
    o.connectedness=.6f; o.left_similarity=.4f; o.right_similarity=.5f;
    o.sentence_score=.6f; o.mean_sentence_score=.7f; o.coherence=.7f;
    o.novelty=.2f; o.phase_lock=.5f;
    return o;
}
static int init(nt_spa_agent *a) {
    nt_spa_agent_config c;
    nt_spa_agent_config_default(&c);
    c.mode=NT_SPA_AGENT_LEARNED; c.exploration=.2f;
    return nt_spa_agent_init(a,&c)==NT_SPA_OK;
}
static int step(nt_spa_agent *a) {
    nt_spa_observation o=observation();
    nt_spa_consequence c;
    nt_spa_decision d;
    memset(&c,0,sizeof c);
    c.after.coherence=.8f; c.after.novelty=.8f;
    return nt_spa_agent_choose(a,&o,&d)==NT_SPA_OK &&
        nt_spa_agent_observe(a,d.sequence,&d.action,&c,NULL)==NT_SPA_OK;
}
static void path_for(char *path,size_t size,const char *name) {
    (void)snprintf(path,size,"%s/%s",directory,name);
}
static size_t read_bytes(const char *path,unsigned char *bytes,size_t capacity) {
    FILE *f=fopen(path,"rb");
    size_t size;
    int extra,bad;
    if(!f) return 0;
    size=fread(bytes,1,capacity,f); extra=fgetc(f); bad=ferror(f);
    if(fclose(f)!=0) bad=1;
    return bad || extra!=EOF ? 0 : size;
}
static int write_bytes(const char *path,const unsigned char *bytes,size_t size) {
    FILE *f=fopen(path,"wb");
    int bad;
    if(!f) return 0;
    bad=fwrite(bytes,1,size,f)!=size;
    if(fclose(f)!=0) bad=1;
    return !bad;
}
static void put32(unsigned char *bytes,uint32_t value) {
    unsigned i;
    for(i=0;i<4;++i) bytes[i]=(unsigned char)(value>>(8*i));
}
static void put64(unsigned char *bytes,uint64_t value) {
    unsigned i;
    for(i=0;i<8;++i) bytes[i]=(unsigned char)(value>>(8*i));
}
static uint64_t checksum(const unsigned char *bytes,size_t count) {
    uint64_t hash=UINT64_C(14695981039346656037);
    size_t i;
    for(i=0;i<count;++i) hash=(hash^bytes[i])*UINT64_C(1099511628211);
    return hash;
}
static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits,&value,sizeof bits);
    return bits;
}

/* Version 1's documented field codec, independent of C structure padding. */
enum {
    FILE_POLICY=12+12+56+8,
    FILE_RNG=FILE_POLICY+4*NT_SPA_AGENT_PARAMETERS,
    FILE_DECISIONS=FILE_RNG+4,
    FILE_OBSERVATIONS=FILE_DECISIONS+8,
    FILE_UPDATES=FILE_OBSERVATIONS+8,
    FILE_CANCELLED=FILE_UPDATES+16,
    FILE_MEMORY_OBSERVATIONS=FILE_CANCELLED+8,
    FILE_HISTORY_COUNT=FILE_MEMORY_OBSERVATIONS+8,
    FILE_HISTORY_HEAD=FILE_HISTORY_COUNT+4,
    FILE_HISTORY=FILE_HISTORY_HEAD+4,
    FILE_RECEIPT_BYTES=100,
    FILE_EMA=FILE_HISTORY+NT_SPA_AGENT_HISTORY*FILE_RECEIPT_BYTES,
    FILE_PENDING=FILE_EMA+12,
    FILE_DECISION=FILE_PENDING+4,
    FILE_OBSERVATION_BYTES=68,
    FILE_FEATURES=FILE_DECISION+FILE_OBSERVATION_BYTES,
    FILE_HIDDEN=FILE_FEATURES+4*NT_SPA_AGENT_FEATURES,
    FILE_SCORES=FILE_HIDDEN+4*NT_SPA_AGENT_HIDDEN,
    FILE_ACTION=FILE_SCORES+4*NT_SPA_AGENT_ACTIONS,
    FILE_RNG_BEFORE=FILE_ACTION+12+8,
    FILE_LENGTH=FILE_RNG_BEFORE+4+4+8
};

static int load_mutation(const unsigned char *source,size_t size,size_t offset,
                         uint64_t value,unsigned width,const char *name,
                         nt_spa_agent *resident) {
    unsigned char bytes[8192];
    nt_spa_agent before=*resident;
    char path[512];
    if(size>sizeof bytes || offset+width>size-8 || (width!=4 && width!=8)) return 0;
    memcpy(bytes,source,size);
    if(width==4) put32(bytes+offset,(uint32_t)value); else put64(bytes+offset,value);
    put64(bytes+size-8,checksum(bytes,size-8));
    path_for(path,sizeof path,"malformed.bin");
    if(!write_bytes(path,bytes,size)) return 0;
    if(nt_spa_agent_load(resident,path)!=NT_SPA_E_FORMAT || !SAME(*resident,before)) {
        fprintf(stderr,"malformed checksum-valid life accepted/mutated: %s\n",name);
        return 0;
    }
    return 1;
}

static int test_hostile_serialized_states(void) {
    nt_spa_agent a,resident;
    nt_spa_decision d;
    nt_spa_observation o=observation();
    unsigned char bytes[8192];
    char path[512];
    size_t size;
    unsigned i;
    CHECK(init(&a) && step(&a),"one actual learned transition");
    resident=a;
    path_for(path,sizeof path,"valid.bin");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"save valid source");
    size=read_bytes(path,bytes,sizeof bytes);
    CHECK(size==FILE_LENGTH,"versioned canonical layout has expected length");
    CHECK(load_mutation(bytes,size,FILE_UPDATES,0,8,"erased acquired-update count",&resident),"updates equal fixed-policy learned observations");
    CHECK(load_mutation(bytes,size,FILE_HISTORY+96,0,4,"erased learned flag",&resident),"receipt flag agrees with frozen learning configuration");
    CHECK(load_mutation(bytes,size,FILE_EMA,float_bits(-1),4,"impossible first reward EMA",&resident),"first reward memory reconstructs from sole receipt");
    CHECK(load_mutation(bytes,size,FILE_EMA+4,float_bits(.5f),4,"impossible first connectedness EMA",&resident),"first connectedness memory reconstructs from sole receipt");
    CHECK(load_mutation(bytes,size,FILE_EMA+8,float_bits(.5f),4,"impossible first novelty EMA",&resident),"first novelty memory reconstructs from sole receipt");
    CHECK(load_mutation(bytes,size,FILE_HISTORY_COUNT,9,4,"oversized ring",&resident),"oversized ring refused");
    CHECK(load_mutation(bytes,size,FILE_HISTORY_HEAD,9,4,"invalid ring head",&resident),"ring head refused before indexing");
    CHECK(load_mutation(bytes,size,FILE_CANCELLED,UINT64_MAX,8,"counter sum overflow",&resident),"counter addition overflow refused");
    CHECK(load_mutation(bytes,size,FILE_POLICY,float_bits(INFINITY),4,"infinite policy weight",&resident),"nonfinite weight refused");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"open pending decision");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"save pending source");
    size=read_bytes(path,bytes,sizeof bytes);
    CHECK(size==FILE_LENGTH,"pending preserves versioned layout");
    CHECK(load_mutation(bytes,size,FILE_FEATURES+19*4,float_bits(nextafterf(d.features[19],INFINITY)),4,"one ULP cached feature",&resident),"pending features exactly reproduce");
    CHECK(load_mutation(bytes,size,FILE_HIDDEN,float_bits(nextafterf(d.hidden[0],INFINITY)),4,"one ULP hidden cache",&resident),"pending hidden exactly reproduces");
    CHECK(load_mutation(bytes,size,FILE_SCORES,float_bits(nextafterf(d.scores[0],INFINITY)),4,"one ULP score cache",&resident),"pending scores exactly reproduce");
    CHECK(load_mutation(bytes,size,FILE_RNG_BEFORE,0,4,"zero prior RNG",&resident),"pending RNG refused");
    CHECK(load_mutation(bytes,size,FILE_ACTION+8,99,4,"foreign source index",&resident),"pending action identity refused");
    CHECK(nt_spa_agent_cancel(&a,d.sequence)==NT_SPA_OK,"close pending fixture");
    for(i=0;i<10;++i) CHECK(step(&a),"history rollover with real consequences");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"save rolled history");
    size=read_bytes(path,bytes,sizeof bytes);
    CHECK(load_mutation(bytes,size,FILE_EMA,float_bits(-1),4,"EMA outside retained-tail possible interval",&resident),"rolled EMA bounded by retained outcomes");
    return 1;
}

static int test_aliases_and_reset(void) {
    nt_spa_agent a,before,expected;
    nt_spa_agent_config config;
    nt_spa_observation o=observation();
    nt_spa_decision d;
    nt_spa_consequence c;
    unsigned i;
    CHECK(init(&a),"init alias fixture"); before=a;
    CHECK(nt_spa_agent_select(&a,&o,&a.pending_decision)==NT_SPA_E_STATE && SAME(a,before),"pure preview refuses internal output without mutation");
    CHECK(nt_spa_agent_choose(&a,&o,&a.pending_decision)==NT_SPA_E_STATE && SAME(a,before),"choose refuses internal output without mutation");
    CHECK(nt_spa_agent_imitate(&a,&o,NT_SPA_KEEP,&a.policy.b2[0])==NT_SPA_E_STATE && SAME(a,before),"imitation loss cannot overwrite live policy");
    config=a.config; config.mode=NT_SPA_AGENT_DISABLED;
    CHECK(nt_spa_agent_init(&a,&config)==NT_SPA_OK,"init disabled alias fixture"); before=a;
    CHECK(nt_spa_agent_choose(&a,&o,&a.pending_decision)==NT_SPA_E_STATE && SAME(a,before),"disabled choose also refuses a state-mutating internal output");
    CHECK(init(&a),"restore learned alias fixture");
    config=a.config;
    CHECK(nt_spa_agent_init(&a,&a.config)==NT_SPA_OK && SAME(a.config,config),"init accepts its own config input");
    before=a;
    CHECK(nt_spa_agent_set_policy(&a,&a.policy)==NT_SPA_OK && SAME(a,before),"self policy replacement is a no-op");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_OK,"reserve alias consequence");
    memset(&c,0,sizeof c); c.after.coherence=.5f; before=a;
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,&a.history[7])==NT_SPA_E_STATE && SAME(a,before),"receipt cannot overwrite unused live-history slot");
    CHECK(nt_spa_agent_observe(&a,d.sequence,&d.action,&c,NULL)==NT_SPA_OK,"normal consequence after refused alias");
    for(i=0;i<10;++i) CHECK(step(&a),"acquire memory before reset");
    expected=a;
    expected.memory_observations=0; expected.history_count=0; expected.history_head=0;
    memset(expected.history,0,sizeof expected.history);
    expected.ema_reward=expected.ema_connectedness=expected.ema_novelty=0;
    CHECK(nt_spa_agent_reset_memory(&a)==NT_SPA_OK && SAME(a,expected),"reset changes exactly temporal memory bytes");
    CHECK(nt_spa_agent_validate(&a)==NT_SPA_OK && step(&a),"reset life continues acquiring outcomes");
    return 1;
}

static int test_counter_limits(void) {
    nt_spa_agent a,before;
    nt_spa_observation o=observation();
    nt_spa_decision d,untouched;
    CHECK(init(&a),"init counter fixture");
    a.decisions=a.cancelled=UINT64_MAX;
    CHECK(nt_spa_agent_validate(&a)==NT_SPA_OK,"maximum cancelled-only lifetime has consistent counters");
    before=a; memset(&d,0xa5,sizeof d); untouched=d;
    CHECK(nt_spa_agent_select(&a,&o,&d)==NT_SPA_E_STATE && SAME(a,before) && SAME(d,untouched),"preview refuses overflowing sequence without mutation");
    CHECK(nt_spa_agent_choose(&a,&o,&d)==NT_SPA_E_STATE && SAME(a,before) && SAME(d,untouched),"choose refuses overflowing sequence without mutation");
    CHECK(init(&a),"reset counter fixture");
    a.imitation_updates=UINT64_MAX; before=a;
    CHECK(nt_spa_agent_imitate(&a,&o,NT_SPA_KEEP,NULL)==NT_SPA_E_STATE && SAME(a,before),"imitation counter overflow refused");
    return 1;
}

static int test_native_allocation_failure(void) {
    const float embeddings[6]={1,0,0,1,1,1};
    nt_spa_observation normal,output,sentinel;
    CHECK(nt_spa_agent_perceive(embeddings,3,2,1,.5f,1,0,&normal)==NT_SPA_OK,"ordinary native perception");
    CHECK(normal.connectedness>.66f && normal.connectedness<.68f,"native allocation baseline is a real connectedness calculation");
    memset(&output,0xa5,sizeof output); sentinel=output;
    fail_next_calloc=1;
    CHECK(nt_spa_agent_perceive(embeddings,3,2,1,.5f,1,0,&output)==NT_SPA_E_MEMORY,"legacy internal allocation failure propagates to agent");
    CHECK(intercepted_failures==1 && fail_next_calloc==0,"exactly one real legacy calloc was refused");
    CHECK(SAME(output,sentinel),"failed native perception leaves observation output unchanged");
    CHECK(nt_spa_agent_perceive(embeddings,3,2,1,.5f,1,0,&output)==NT_SPA_OK && SAME(output,normal),"perception recovers exactly after one-shot failure");
    return 1;
}

static int test_failed_save_preserves_checkpoint(void) {
    nt_spa_agent a,resumed;
    char path[512];
    unsigned char original[8192],after[8192];
    size_t size,after_size;
    pid_t child;
    int status;
    struct rlimit original_limit,after_limit;
    CHECK(init(&a) && step(&a),"init persisted life");
    path_for(path,sizeof path,"atomic.bin");
    CHECK(nt_spa_agent_save(&a,path)==NT_SPA_OK,"create valid previous checkpoint");
    size=read_bytes(path,original,sizeof original);
    CHECK(size==FILE_LENGTH && getrlimit(RLIMIT_FSIZE,&original_limit)==0,"read checkpoint and parent resource limit");
    child=fork();
    CHECK(child>=0,"fork isolated write-failure host");
    if(child==0) {
        struct rlimit limit={32,32};
        if(signal(SIGXFSZ,SIG_IGN)==SIG_ERR || setrlimit(RLIMIT_FSIZE,&limit)!=0) _exit(77);
        _exit(nt_spa_agent_save(&a,path)==NT_SPA_E_IO ? 0 : 1);
    }
    CHECK(waitpid(child,&status,0)==child && WIFEXITED(status),"write-failure child completed");
    if(WEXITSTATUS(status)==77) {
        puts("SKIPPED atomic failed save: child cannot install RLIMIT_FSIZE/SIGXFSZ fixture");
        return -1;
    }
    CHECK(WEXITSTATUS(status)==0,"forced partial write reports IO failure in child");
    CHECK(getrlimit(RLIMIT_FSIZE,&after_limit)==0 && original_limit.rlim_cur==after_limit.rlim_cur && original_limit.rlim_max==after_limit.rlim_max,"parent resource limits remain unchanged");
    after_size=read_bytes(path,after,sizeof after);
    CHECK(after_size==size && memcmp(original,after,size)==0,"failed replacement preserves every previous checkpoint byte");
    CHECK(nt_spa_agent_load(&resumed,path)==NT_SPA_OK && nt_spa_agent_hash(&resumed)==nt_spa_agent_hash(&a),"previous life remains resumable after failed save");
    return 1;
}

int main(void) {
    struct { const char *name; int (*run)(void); } groups[]={
        {"checksum-valid malformed lives",test_hostile_serialized_states},
        {"output aliases and exact reset",test_aliases_and_reset},
        {"counter limits",test_counter_limits},
        {"native allocation failure",test_native_allocation_failure},
        {"atomic failed save",test_failed_save_preserves_checkpoint}
    };
    unsigned i,passed=0,skipped=0;
    DIR *dir;
    struct dirent *entry;
    char path[512];
    if(!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    for(i=0;i<sizeof groups/sizeof groups[0];++i) {
        int result;
        gate=groups[i].name;
        result=groups[i].run();
        if(result>0) { ++passed; printf("PASS %s\n",gate); }
        else if(result<0) ++skipped;
    }
    dir=opendir(directory);
    if(dir) {
        while((entry=readdir(dir))) if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) {
            path_for(path,sizeof path,entry->d_name); (void)unlink(path);
        }
        closedir(dir);
    }
    (void)rmdir(directory);
    printf("SPA_AGENT_STATE %u/%zu groups, %u checks, %u skipped\n",passed,sizeof groups/sizeof groups[0],checks,skipped);
    if(passed+skipped!=sizeof groups/sizeof groups[0]) return 1;
    return skipped ? 77 : 0;
}
