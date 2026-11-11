/* Read the saved parliament independently of the running experts. All six
 * seats are inspected, including seats born during the long experience. */
#include "gguf.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEATS=6, RANK=4 };
static const float *tensor(const gguf_file *g, const char *name, uint64_t *count) {
    int i=gguf_find_tensor(g,name);
    assert(i>=0);
    const gguf_tensor_info *t=&g->tensors[i];
    assert(t->dtype==GGUF_TYPE_F32 && t->n_elements && t->offset<=g->data_size &&
           t->n_elements<=(g->data_size-t->offset)/sizeof(float));
    const float *v=(const float *)(g->data+t->offset);
    for(uint64_t j=0;j<t->n_elements;j++) assert(isfinite(v[j]));
    if(count) *count=t->n_elements;
    return v;
}
static uint64_t clock_value(const gguf_file *g,const char *name) {
    const gguf_kv *k=gguf_get_kv(g,name);
    if(!k) return 0;
    assert(k->type==4 || k->type==10);
    return k->type==4 ? k->val.u32 : k->val.u64;
}
static const float *fixed(const gguf_file *g,const char *name,uint64_t n) {
    uint64_t got;
    const float *p=tensor(g,name,&got);
    assert(got==n);
    return p;
}
static double rms(const float *x,uint64_t n) {
    double sum=0;
    for(uint64_t i=0;i<n;i++) sum+=(double)x[i]*x[i];
    return sqrt(sum/n);
}
static void summary(const char *path,const char *label,int slots) {
    gguf_file *g=gguf_open(path);
    assert(g);
    uint64_t n;
    const float *a=tensor(g,"doe.lora_a",&n);
    const float *b=fixed(g,"doe.lora_b",n);
    assert(n%SEATS==0);
    uint64_t width=n/SEATS;
    const float *alive=fixed(g,"doe.alive",SEATS), *age=fixed(g,"doe.age",SEATS);
    const float *v=fixed(g,"doe.vitality",SEATS), *c=fixed(g,"doe.cursor",SEATS);
    const float *low=fixed(g,"doe.low_vitality",SEATS);
    for(int e=0;e<SEATS;e++) {
        if(slots) { if(alive[e]) printf("%d ",e); continue; }
        printf("%s\t%llu\t%llu\t%llu\t%llu\t%d\t%.0f\t%.0f\t%.9g\t%.0f\t%.0f\t%.12g\t%.12g\n",
               label,(unsigned long long)clock_value(g,"doe.turns"),
               (unsigned long long)clock_value(g,"doe.elections"),
               (unsigned long long)clock_value(g,"doe.dream.cycles"),
               (unsigned long long)clock_value(g,"doe.rng"),e,
               alive[e],age[e],v[e],low[e],c[e],rms(a+e*width,width),rms(b+e*width,width));
    }
    if(slots) putchar('\n');
    gguf_close(g);
}
static void night(const char *before,const char *after,const char *trace,const char *label,int replay) {
    gguf_file *a=gguf_open(before), *b=gguf_open(after);
    FILE *f=fopen(trace,"r");
    assert(a && b && f && a->n_tensors==b->n_tensors);
    int hits[SEATS]={0}, elections=0, choices=0;
    char line[2048];
    while(fgets(line,sizeof line,f)) {
        choices+=strstr(line,replay ? "doe replay-choice " : "doe dream-choice ")!=NULL;
        if(!strstr(line,replay ? "doe replay-vote " : "doe dream-vote ")) continue;
        char *g=strstr(line,"gates=");
        assert(g); g+=6; elections++;
        for(int e=0;e<SEATS;e++) {
            char *end;
            double gate=strtod(g,&end);
            assert(end!=g && isfinite(gate));
            hits[e]+=gate>0; g=end+1;
        }
    }
    assert(!ferror(f) && !fclose(f) && choices>0 && elections>0);
    assert(clock_value(a,"doe.turns")==clock_value(b,"doe.turns") &&
           clock_value(a,"doe.elections")==clock_value(b,"doe.elections"));
    const char *cycles=replay ? "doe.replay.cycles" : "doe.dream.cycles";
    const char *decisions=replay ? "doe.replay.choices" : "doe.dream.choices";
    const char *ballots=replay ? "doe.replay.elections" : "doe.dream.elections";
    assert(clock_value(a,cycles)+1==clock_value(b,cycles));
    assert(clock_value(a,decisions)+(unsigned)choices==clock_value(b,decisions));
    assert(clock_value(a,ballots)+(unsigned)elections==clock_value(b,ballots));
    if(replay) {
        assert(clock_value(a,"doe.dream.cycles")==clock_value(b,"doe.dream.cycles") &&
               clock_value(a,"doe.dream.seen")==clock_value(b,"doe.dream.seen"));
        int na=0,nb=0;
        char **la=gguf_read_fixed_str_array(before,"doe.lineage.entries",174,&na);
        char **lb=gguf_read_fixed_str_array(after,"doe.lineage.entries",174,&nb);
        assert(la && lb && na==nb);
        for(int i=0;i<na;i++) { assert(!strcmp(la[i],lb[i])); free(la[i]); free(lb[i]); }
        free(la); free(lb);
    }
    assert(gguf_get_kv(a,"doe.consensus")->val.f32==gguf_get_kv(b,"doe.consensus")->val.f32);
    for(uint64_t i=0;i<a->n_tensors;i++) {
        const char *name=a->tensors[i].name;
        uint64_t n,m;
        const float *x=tensor(a,name,&n), *y=tensor(b,name,&m);
        assert(n==m);
        if(strcmp(name,"doe.lora_a") && strcmp(name,"doe.lora_b") && strcmp(name,"doe.cursor")) {
            assert(!memcmp(x,y,(size_t)n*sizeof(float)));
            continue;
        }
        assert(n%SEATS==0);
        uint64_t width=n/SEATS;
        for(int e=0;e<SEATS;e++) {
            uint64_t changed=0;
            double delta=0;
            for(uint64_t j=0;j<width;j++) {
                size_t at=e*width+j;
                changed+=x[at]!=y[at];
                double d=(double)y[at]-x[at]; delta+=d*d;
            }
            assert((changed>0)==(hits[e]>0));
            if(!strcmp(name,"doe.cursor")) assert(y[e]==(hits[e] ? ((int)x[e]+2)%RANK : x[e]));
            printf("%s\t%s\t%d\t%d\t%llu\t%.12g\n",label,name,e,hits[e],
                   (unsigned long long)changed,sqrt(delta));
        }
    }
    gguf_close(b); gguf_close(a);
}

static void factor_distance(const char *left,const char *right,const char *label) {
    gguf_file *a=gguf_open(left), *b=gguf_open(right);
    assert(a && b);
    const char *names[]={"doe.lora_a","doe.lora_b"};
    double ss[2][SEATS]={{0}};
    uint64_t changes[2][SEATS]={{0}};
    for(int k=0;k<2;k++) {
        uint64_t n;
        const float *x=tensor(a,names[k],&n), *y=fixed(b,names[k],n);
        assert(n%SEATS==0);
        uint64_t width=n/SEATS;
        for(int e=0;e<SEATS;e++) for(uint64_t j=0;j<width;j++) {
            uint64_t at=e*width+j;
            double d=(double)x[at]-y[at];
            ss[k][e]+=d*d;
            changes[k][e]+=x[at]!=y[at];
        }
    }
    for(int e=0;e<SEATS;e++) printf("%s\t%d\t%llu\t%llu\t%.12g\t%.12g\t%.12g\n",
        label,e,(unsigned long long)changes[0][e],(unsigned long long)changes[1][e],
        sqrt(ss[0][e]),sqrt(ss[1][e]),sqrt(ss[0][e]+ss[1][e]));
    gguf_close(b); gguf_close(a);
}

static float *read_row(const char *path,size_t *n) {
    FILE *f=fopen(path,"rb");
    assert(f && !fseek(f,0,SEEK_END));
    long bytes=ftell(f);
    assert(bytes>0 && bytes%4==0 && bytes/4<=1<<24);
    *n=(size_t)bytes/4;
    float *a=malloc((size_t)bytes);
    rewind(f);
    assert(a && fread(a,4,*n,f)==*n && !ferror(f) && !fclose(f));
    for(size_t i=0;i<*n;i++) assert(isfinite(a[i]));
    return a;
}
static void compare_logits(const char *left,const char *right,const char *label) {
    size_t n,m;
    float *a=read_row(left,&n), *b=read_row(right,&m);
    assert(n==m);
    size_t changed=0, first=0, second=0;
    double max=0, ss=0, za=0, zb=0, kl=0;
    for(size_t i=0;i<n;i++) {
        changed+=memcmp(a+i,b+i,sizeof(float))!=0;
        double d=fabs((double)b[i]-a[i]);
        if(d>max) max=d;
        ss+=d*d;
        if(a[i]>a[first]) first=i;
        if(b[i]>b[second]) second=i;
    }
    for(size_t i=0;i<n;i++) { za+=exp((double)a[i]-a[first]); zb+=exp((double)b[i]-b[second]); }
    for(size_t i=0;i<n;i++) {
        double p=exp((double)a[i]-a[first])/za;
        kl+=p*((double)a[i]-a[first]-b[i]+b[second]+log(zb)-log(za));
    }
    printf("%s\t%zu\t%zu\t%.12g\t%.12g\t%.12g\t%zu\t%zu\n",
           label,n,changed,max,sqrt(ss),kl,first,second);
    free(b); free(a);
}
static void clocks(const char *path,const char *label) {
    gguf_file *g=gguf_open(path); assert(g);
    const char *keys[]={"doe.version","doe.turns","doe.elections","doe.dream.cycles",
        "doe.dream.choices","doe.dream.elections","doe.dream.seen","doe.replay.cycles",
        "doe.replay.choices","doe.replay.elections","doe.replay.prefill","doe.replay.stops","doe.rng"};
    printf("%s",label);
    for(size_t i=0;i<sizeof keys/sizeof keys[0];i++) printf("\t%llu",(unsigned long long)clock_value(g,keys[i]));
    int count=0;
    char **ledger=gguf_read_fixed_str_array(path,"doe.lineage.entries",174,&count);
    assert(ledger); for(int i=0;i<count;i++) free(ledger[i]); free(ledger);
    const gguf_kv *last=gguf_get_kv(g,"doe.replay.last");
    printf("\t%d\t%s\n",count,last ? last->val.str : "");
    gguf_close(g);
}
int main(int argc,char **argv) {
    if(argc==4 && !strcmp(argv[1],"summary")) summary(argv[2],argv[3],0);
    else if(argc==4 && !strcmp(argv[1],"clocks")) clocks(argv[2],argv[3]);
    else if(argc==3 && !strcmp(argv[1],"slots")) summary(argv[2],"",1);
    else if(argc==6 && !strcmp(argv[1],"night")) night(argv[2],argv[3],argv[4],argv[5],0);
    else if(argc==6 && !strcmp(argv[1],"replay")) night(argv[2],argv[3],argv[4],argv[5],1);
    else if(argc==5 && !strcmp(argv[1],"distance")) factor_distance(argv[2],argv[3],argv[4]);
    else if(argc==5 && !strcmp(argv[1],"logits")) compare_logits(argv[2],argv[3],argv[4]);
    else { fprintf(stderr,"usage: %s summary SPORE LABEL | slots SPORE | night BEFORE AFTER TRACE LABEL | distance LEFT RIGHT LABEL | logits BEFORE_F32 AFTER_F32 LABEL\n",argv[0]); return 2; }
    return 0;
}
