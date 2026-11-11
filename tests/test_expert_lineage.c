/* Durable identities on the actual consolidation/serialization path. The
 * caller runs test_portable_spore fixtures in a fresh directory first. */
#include "harness/experts_internal.h"
#include "harness/archs.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
#define CHECK(c, what) do { checks++; if (!(c)) { \
    fprintf(stderr,"FAIL lineage: %s (line %d)\n",what,__LINE__); failures++; \
} } while (0)

typedef struct {
    gguf_file *g;
    const nt_arch *arch;
    void *model;
    bpe_tokenizer *tok;
    nt_expert_body body;
} body;

static body open_body(const char *path) {
    body b={0};
    b.g=gguf_open(path);
    assert(b.g);
    b.arch=nt_pick_arch(b.g->arch);
    nt_dims dims;
    b.model=b.arch ? b.arch->load(b.g,&dims) : NULL;
    b.tok=bpe_load(path);
    assert(b.model && b.tok);
    b.body=(nt_expert_body){b.model,b.tok};
    return b;
}

static void close_body(body *b) {
    bpe_free(b->tok); b->arch->free(b->model); gguf_close(b->g);
}

static nt_experts *open_state(body *b,const char *path) {
    nt_experts *p=nt_experts_open_on_body(b->g,path,NULL,&b->body);
    assert(p);
    return p;
}

static nt_expert_identity seat(const nt_experts *p,int slot) {
    nt_expert_identity e;
    assert(nt_experts_seat_identity(p,slot,&e)==NT_OK);
    return e;
}

static int same_identity(const nt_expert_identity *a,const nt_expert_identity *b) {
    return !strcmp(a->id,b->id) && !strcmp(a->parent,b->parent) &&
        !strcmp(a->source_host,b->source_host) && a->turn==b->turn &&
        a->death_turn==b->death_turn && a->kind==b->kind;
}

static void same_ledger(const nt_experts *a,const nt_experts *b) {
    CHECK(a->ancestor_count==b->ancestor_count,"all ancestry records survive");
    int n=a->ancestor_count<b->ancestor_count ? a->ancestor_count : b->ancestor_count;
    for(int i=0;i<n;i++) {
        nt_expert_identity x,y;
        assert(nt_experts_ancestor(a,i,&x)==NT_OK && nt_experts_ancestor(b,i,&y)==NT_OK);
        CHECK(same_identity(&x,&y),"birth address, parent, source body and dates survive");
    }
    for(int i=0;i<SEATS;i++) {
        nt_expert_identity x=seat(a,i),y=seat(b,i);
        CHECK(same_identity(&x,&y),"slot-to-identity map survives");
    }
}

static void same_numeric(const nt_experts *a,const nt_experts *b) {
    assert(a->dim==b->dim);
    size_t dr=(size_t)SEATS*a->dim*RANK, dw=(size_t)SEATS*a->dim;
    CHECK(!memcmp(a->a,b->a,dr*sizeof(float)) && !memcmp(a->b,b->b,dr*sizeof(float)),"migration preserves all A/B bits");
    CHECK(!memcmp(a->vote,b->vote,dw*sizeof(float)),"migration preserves voting factors");
    CHECK(!memcmp(a->alive,b->alive,sizeof a->alive) && !memcmp(a->vitality,b->vitality,sizeof a->vitality) &&
          !memcmp(a->frequency,b->frequency,sizeof a->frequency) && !memcmp(a->cursor,b->cursor,sizeof a->cursor) &&
          !memcmp(a->age,b->age,sizeof a->age) && !memcmp(a->low,b->low,sizeof a->low) &&
          !memcmp(a->history,b->history,sizeof a->history),"migration preserves lifecycle and oscillator state");
    CHECK(a->rng==b->rng && a->turns==b->turns && a->elections==b->elections &&
          a->consensus==b->consensus && a->sleep.cycles==b->sleep.cycles &&
          a->sleep.choices==b->sleep.choices && a->sleep.elections==b->sleep.elections &&
          a->sleep.seen==b->sleep.seen,"migration preserves scalar state and both clocks");
}

static void same_file(const char *a,const char *b) {
    FILE *x=fopen(a,"rb"),*y=fopen(b,"rb");
    assert(x && y);
    int c,d,equal=1;
    do { c=fgetc(x); d=fgetc(y); if(c!=d) equal=0; } while(c!=EOF && d!=EOF);
    CHECK(equal && !ferror(x) && !ferror(y),"complete saved generations are byte-identical");
    assert(!fclose(x) && !fclose(y));
}

/* Real hooks accumulate the pending turn. Only lifecycle thresholds/hit
 * counts are forced, to exercise birth and death without a long stochastic run. */
static void observe(nt_experts *p,float value,int parent,int retire) {
    float *row=malloc((size_t)p->dim*sizeof(float));
    assert(row);
    for(int t=0;t<3;t++) {
        for(int i=0;i<p->dim;i++) row[i]=value+(float)((i+3*t)%17)/32;
        assert(nt_experts_attend(p,p->layer,t,1,p->dim,row)==NT_OK);
        assert(nt_experts_observe(p,p->layer,t,1,p->dim,row)==NT_OK);
    }
    nt_experts_choice(p,1,1);
    if(parent>=0 || retire>=0) {
        for(int e=0;e<SEATS;e++) {
            p->age[e]=0;
            p->low[e]=0;
            if(p->alive[e]) p->vitality[e]=0.5f;
            p->hits[e]=0;
        }
        if(parent>=0) { p->age[parent]=21; p->vitality[parent]=0.99f; p->hits[parent]=p->count; }
        if(retire>=0) { p->vitality[retire]=0; p->low[retire]=7; }
    }
    free(row);
}

static void complete(nt_experts **p,int worker) {
    if(worker) {
        nt_consolidation *job=NULL;
        assert(nt_experts_submit(p,&job)==NT_OK && !*p && job);
        assert(nt_experts_await(p,&job)==NT_OK && *p && !job);
    } else assert(nt_experts_finish(*p,1)==NT_OK);
}

static void founders_and_births(body *b) {
    nt_experts *a=open_state(b,NULL), *same=open_state(b,NULL);
    CHECK(nt_experts_ancestor_count(a)==FOUNDERS,"four founder records");
    same_ledger(a,same);
    for(int i=0;i<SEATS;i++) {
        nt_expert_identity e=seat(a,i);
        if(i<FOUNDERS) {
            CHECK(strlen(e.id)==64 && !*e.parent && e.kind==NT_LINEAGE_FOUNDER &&
                  !e.turn && !e.death_turn && !strcmp(e.source_host,a->host),"founder address and provenance");
            CHECK(nt_experts_find_identity(a,e.id)==i,"find a living identity");
            for(int j=0;j<i;j++) { nt_expert_identity other=seat(a,j); CHECK(strcmp(e.id,other.id),"founder IDs are unique"); }
        } else CHECK(!*e.id && !*e.parent && !e.kind,"unused slot has no invented ancestor");
    }
    nt_expert_identity poison,copy;
    memset(&poison,0x55,sizeof poison); copy=poison;
    CHECK(nt_experts_ancestor(a,-1,&poison)==NT_E_ARG && !memcmp(&poison,&copy,sizeof poison),"invalid read preserves caller output");
    CHECK(nt_experts_seat_identity(a,SEATS,&poison)==NT_E_ARG &&
          nt_experts_find_identity(a,"bad")==-1 && nt_experts_ancestor_count(NULL)==0,"public identity argument bounds");
    observe(a,0.2f,-1,-1); complete(&a,0);
    same_ledger(a,same);
    assert(!nt_experts_save_portable(a,"lineage-seed"));
    nt_experts_free(same);

    /* Same ancestor/turn/ordinal: different observations produce different
     * birth addresses, while serial/worker paths preserve the same address. */
    nt_experts *branch=open_state(b,"lineage-seed"), *worker=open_state(b,"lineage-seed");
    observe(a,0.3f,0,-1); complete(&a,0);
    observe(branch,0.8f,0,-1); complete(&branch,0);
    observe(worker,0.3f,0,-1); complete(&worker,1);
    nt_expert_identity root=seat(a,0), child=seat(a,4), other=seat(branch,4), repeated=seat(worker,4);
    CHECK(a->ancestor_count==5 && child.kind==NT_LINEAGE_BIRTH && child.turn==2 &&
          !strcmp(child.parent,root.id),"new birth records its actual parent and completed turn");
    CHECK(strcmp(child.id,other.id) && !strcmp(child.parent,other.parent),"divergent branches do not collide on birth ordinal");
    CHECK(same_identity(&child,&repeated),"worker birth address matches serial");
    assert(!nt_experts_save_portable(a,"lineage-birth") && !nt_experts_save_portable(worker,"lineage-worker"));
    same_file("lineage-birth","lineage-worker");
    nt_experts_free(branch); nt_experts_free(worker);

    observe(a,0.4f,-1,0); complete(&a,0);
    nt_expert_identity retired=seat(a,0);
    CHECK(!a->alive[0] && !strcmp(root.id,retired.id) && retired.death_turn==3 &&
          nt_experts_find_identity(a,root.id)==-1,"retirement preserves the old address and records its date");
    assert(!nt_experts_save_portable(a,"lineage-retired"));
    observe(a,0.5f,4,-1); complete(&a,0);
    nt_expert_identity replacement=seat(a,0), old;
    assert(nt_experts_ancestor(a,0,&old)==NT_OK);
    CHECK(a->ancestor_count==6 && a->alive[0] && strcmp(root.id,replacement.id) &&
          !strcmp(replacement.parent,child.id) && replacement.turn==4,"reusing a seat creates a fresh identity");
    CHECK(same_identity(&old,&retired) && nt_experts_find_identity(a,root.id)==-1 &&
          nt_experts_find_identity(a,replacement.id)==0,"retired ancestor remains addressable in the ledger after reuse");
    assert(!nt_experts_save_portable(a,"lineage-reused"));
    nt_experts *restored=open_state(b,"lineage-reused");
    same_ledger(a,restored); same_numeric(a,restored);
    assert(!nt_experts_save_portable(restored,"lineage-restored"));
    same_file("lineage-reused","lineage-restored");
    nt_experts_free(restored); nt_experts_free(a);
}

static int portable_key(const char *key) {
    return !strcmp(key,"doe.frame") || !strcmp(key,"doe.source_arch") ||
        !strcmp(key,"doe.norm_sha256") || !strcmp(key,"doe.width") || !strcmp(key,"doe.depth");
}

/* Independent writer: preserve tensor bytes, replace only named metadata.
 * Version 1..4 output omits lineage and uses the original tensor directory. */
static void rewrite(const gguf_file *g,const char *path,unsigned version,
                    const char *const *entries,int n,const int32_t *slots,int ns,
                    const char *drop,int lineage_version) {
    int legacy=version<=4, portable=(version&1)==0;
    gguf_writer *w=gguf_write_open(path);
    assert(w);
    for(int i=0;i<g->n_kv_parsed;i++) {
        const gguf_kv *k=g->kv+i;
        if((drop && !strcmp(k->key,drop)) ||
           (legacy && !strncmp(k->key,"doe.lineage.",12)) ||
           (!portable && portable_key(k->key)) ||
           (version<=2 && !strncmp(k->key,"doe.dream.",10))) continue;
        int rc=-1;
        if(!strcmp(k->key,"doe.version")) rc=gguf_write_kv_u32(w,k->key,version);
        else if(!strcmp(k->key,"doe.lineage.version")) rc=gguf_write_kv_u32(w,k->key,(uint32_t)lineage_version);
        else if(!strcmp(k->key,"doe.lineage.entries")) rc=gguf_write_kv_str_array(w,k->key,entries,(uint64_t)n);
        else if(!strcmp(k->key,"doe.lineage.seats")) rc=gguf_write_kv_i32_array(w,k->key,slots,(uint64_t)ns);
        else if(k->type==4) rc=gguf_write_kv_u32(w,k->key,k->val.u32);
        else if(k->type==6) rc=gguf_write_kv_f32(w,k->key,k->val.f32);
        else if(k->type==8) rc=gguf_write_kv_str(w,k->key,k->val.str);
        else if(k->type==10) rc=gguf_write_kv_u64(w,k->key,k->val.u64);
        assert(!rc);
    }
    for(int pass=0;pass<2;pass++) for(uint64_t i=0;i<g->n_tensors;i++) {
        const gguf_tensor_info *t=g->tensors+i;
        if(!portable && !strcmp(t->name,"doe.anchor_frame")) continue;
        assert(t->dtype==GGUF_TYPE_F32);
        assert(!(pass ? gguf_write_tensor_f32(w,t->name,(const float *)(g->data+t->offset),t->n_elements)
                     : gguf_write_tensor_decl(w,t->name,t->ndim,t->shape,t->dtype)));
    }
    assert(!gguf_write_close(w));
}

static void transfer_and_legacy(body *source,body *aligned,body *wide) {
    nt_experts *p=open_state(source,"lineage-reused");
    nt_experts *a=open_state(aligned,"lineage-reused"), *w=open_state(wide,"lineage-reused");
    same_ledger(p,a); same_ledger(p,w);
    assert(!nt_experts_save_portable(w,"lineage-wide"));
    nt_experts *returned=open_state(source,"lineage-wide");
    same_ledger(p,returned);
    nt_experts_free(returned); nt_experts_free(w); nt_experts_free(a); nt_experts_free(p);

    p=open_state(source,"lineage-seed");
    assert(!nt_experts_dream(p,3,-1,1) && !nt_experts_save_portable(p,"lineage-slept"));
    nt_experts_free(p);
    for(unsigned version=1;version<=4;version++) {
        const char *base=version<=2 ? "lineage-retired" : "lineage-slept";
        p=open_state(source,base);
        gguf_file *g=gguf_open(base);
        assert(g);
        char legacy[40],first[40],second[40];
        snprintf(legacy,sizeof legacy,"legacy-%u",version);
        snprintf(first,sizeof first,"adopted-%u",version);
        snprintf(second,sizeof second,"adopted-again-%u",version);
        rewrite(g,legacy,version,NULL,0,NULL,0,NULL,0);
        gguf_close(g);
        nt_experts *adopted=open_state(source,legacy), *again=open_state(source,legacy);
        same_numeric(p,adopted); same_ledger(adopted,again);
        CHECK(adopted->ancestor_count==p->ancestor_count,"legacy retains occupied and retired records");
        for(int slot=0;slot<SEATS;slot++) {
            nt_expert_identity e=seat(adopted,slot);
            if(p->seat_ancestor[slot]) CHECK(e.kind==NT_LINEAGE_ADOPTED && !*e.parent &&
                e.turn==p->turns && !strcmp(e.source_host,p->host) &&
                e.death_turn==(p->alive[slot] ? 0 : UINT64_MAX),"legacy adoption declares its first known observation and unknown ancestry");
        }
        if(!(version&1)) {
            nt_experts *foreign=open_state(wide,legacy);
            same_ledger(adopted,foreign);
            nt_experts_free(foreign);
        }
        assert(!nt_experts_save_portable(adopted,first));
        nt_experts *round=open_state(source,first);
        same_ledger(adopted,round); same_numeric(adopted,round);
        assert(!nt_experts_save_portable(round,second)); same_file(first,second);
        nt_experts_free(round); nt_experts_free(again); nt_experts_free(adopted); nt_experts_free(p);
    }
}

static void identity_copy(body *b) {
    nt_experts *src=open_state(b,"lineage-birth"), *dst=open_state(b,"lineage-birth");
    size_t n=(size_t)dst->dim*RANK;
    /* Relocate the first two living identities together with their factors.
     * The API must resolve addresses on each side, not assume matching slots. */
    float *tmp=malloc(n*sizeof(float)); assert(tmp);
    memcpy(tmp,dst->a,n*sizeof(float)); memcpy(dst->a,dst->a+n,n*sizeof(float)); memcpy(dst->a+n,tmp,n*sizeof(float));
    memcpy(tmp,dst->b,n*sizeof(float)); memcpy(dst->b,dst->b+n,n*sizeof(float)); memcpy(dst->b+n,tmp,n*sizeof(float));
    size_t vw=(size_t)dst->dim*sizeof(float);
    memcpy(tmp,dst->vote,vw); memcpy(dst->vote,dst->vote+dst->dim,vw); memcpy(dst->vote+dst->dim,tmp,vw);
    float *fields[]={dst->alive,dst->vitality,dst->frequency,dst->cursor,dst->age,dst->low};
    for(size_t i=0;i<sizeof fields/sizeof *fields;i++) { float x=fields[i][0]; fields[i][0]=fields[i][1]; fields[i][1]=x; }
    int32_t swapped=dst->seat_ancestor[0]; dst->seat_ancestor[0]=dst->seat_ancestor[1]; dst->seat_ancestor[1]=swapped;
    nt_expert_identity id=seat(src,0),before=seat(dst,1);
    src->a[0]+=0.25f; src->b[0]-=0.125f;
    CHECK(nt_experts_find_identity(dst,id.id)==1 && nt_experts_find_identity(src,id.id)==0,"identity resolution follows relocated seats");
    memcpy(tmp,dst->a,n*sizeof(float));
    CHECK(nt_experts_copy_factors_by_identity(dst,src,id.id)==NT_OK &&
          !memcmp(dst->a+n,src->a,n*sizeof(float)) && !memcmp(dst->b+n,src->b,n*sizeof(float)) &&
          !memcmp(tmp,dst->a,n*sizeof(float)),"addressed copy replaces only the resolved recipient factors");
    nt_expert_identity after=seat(dst,1);
    CHECK(same_identity(&before,&after),"factor intervention retains recipient identity");
    CHECK(nt_experts_copy_factors_by_identity(dst,src,"invalid")==NT_E_ARG &&
          nt_experts_copy_factors_by_identity(dst,dst,id.id)==NT_OK,"addressed copy rejects missing identity and allows self copy");
    free(tmp); nt_experts_free(dst); nt_experts_free(src);
}

static void patch_file(const char *path,const void *needle,size_t length,
                       size_t relative,const void *value,size_t bytes) {
    FILE *f=fopen(path,"r+b"); assert(f && !fseek(f,0,SEEK_END));
    long size=ftell(f); assert(size>0); rewind(f);
    unsigned char *data=malloc((size_t)size); assert(data);
    assert(fread(data,1,(size_t)size,f)==(size_t)size);
    size_t at=0;
    assert(length<=(size_t)size && bytes<=(size_t)size);
    while(at<=(size_t)size-length && memcmp(data+at,needle,length)) at++;
    assert(at<=(size_t)size-length && relative<=(size_t)size-at && bytes<=(size_t)size-at-relative);
    assert(!fseek(f,(long)(at+relative),SEEK_SET) && fwrite(value,1,bytes,f)==bytes && !fclose(f));
    free(data);
}

static void fixed_array_reader(void) {
    const char *key="doe.lineage.entries";
    char record[175]; memset(record,'a',174); record[174]=0;
    const char *records[2]={record,record};
    const char *names[]={"complete fixed-width array", "declared count exceeds remaining bytes",
                         "declared count exceeds allocation bound", "later declared length differs",
                         "embedded NUL inside fixed record"};
    for(int test=0;test<5;test++) {
        char path[48]; snprintf(path,sizeof path,"lineage-array-%d",test);
        gguf_writer *w=gguf_write_open(path); assert(w);
        assert(!gguf_write_kv_u32(w,"before.records",1) &&
               !gguf_write_kv_str_array(w,key,records,2) && !gguf_write_close(w));
        size_t end=strlen(key);
        uint64_t changed=test==1 ? 3 : test==2 ? (uint64_t)GGUF_MAX_STR_ARRAY+1 : 175;
        unsigned char zero=0;
        if(test==1 || test==2) patch_file(path,key,end,end+8,&changed,8);
        if(test==3) patch_file(path,key,end,end+16+8+174,&changed,8);
        if(test==4) patch_file(path,key,end,end+16+8,&zero,1);
        int n=99;
        char **read=gguf_read_fixed_str_array(path,key,174,&n);
        CHECK(test ? !read && n==0 : read && n==2 && !strcmp(read[0],record) && !strcmp(read[1],record),names[test]);
        for(int i=0;read && i<n;i++) free(read[i]);
        free(read);
    }
}

static void malformed(body *b) {
    gguf_file *g=gguf_open("lineage-reused"); assert(g);
    int n,ns;
    char **records=gguf_read_str_array(g->path,"doe.lineage.entries",&n);
    int32_t *slots=gguf_read_i32_array(g->path,"doe.lineage.seats",&ns);
    assert(records && n==6 && slots && ns==SEATS);
    static const char *const names[]={
        "duplicate identity", "non-hex identity", "self parent", "future birth", "death before birth",
        "future death", "unseated living ancestor", "duplicate slot address", "missing living slot address",
        "dead identity in living seat", "short record", "short slot map", "missing entries", "missing slots",
        "missing schema", "unknown schema", "bad source host", "parent younger than child",
        "embedded NUL after canonical record"
    };
    for(size_t test=0;test<sizeof names/sizeof *names;test++) {
        char *changed[6]; int32_t map[SEATS];
        for(int i=0;i<n;i++) { changed[i]=strdup(records[i]); assert(changed[i]); }
        memcpy(map,slots,sizeof map);
        const char *drop=NULL; int count=ns,schema=1;
        switch(test) {
            case 0: memcpy(changed[1]+45,changed[0]+45,64); break;
            case 1: changed[1][45]='z'; break;
            case 2: memcpy(changed[5]+36,"00000006",8); break;
            case 3: memcpy(changed[5]+2,"ffffffffffffffff",16); break;
            case 4: memcpy(changed[5]+19,"0000000000000003",16); break;
            case 5: memcpy(changed[0]+19,"0000000000000005",16); break;
            case 6: memcpy(changed[0]+19,"0000000000000000",16); break;
            case 7: map[1]=map[0]; break;
            case 8: map[0]=0; break;
            case 9: map[0]=1; break;
            case 10: changed[2][173]=0; break;
            case 11: count=SEATS-1; break;
            case 12: drop="doe.lineage.entries"; break;
            case 13: drop="doe.lineage.seats"; break;
            case 14: drop="doe.lineage.version"; break;
            case 15: schema=2; break;
            case 16: changed[5][110]='Z'; break;
            case 17: memcpy(changed[5]+2,"0000000000000001",16); break;
            case 18:
                changed[0]=realloc(changed[0],176); assert(changed[0]);
                changed[0][174]='X'; changed[0][175]=0;
                break;
        }
        char path[48]; snprintf(path,sizeof path,"lineage-malformed-%zu",test);
        rewrite(g,path,6,(const char *const *)changed,n,map,count,drop,schema);
        if(test==18) {
            /* Keep the declared 175 bytes; a C-string reader must not erase
             * this suffix and recover a seemingly canonical 174-byte record. */
            unsigned char zero=0;
            patch_file(path,changed[0],175,174,&zero,1);
        }
        nt_experts *bad=nt_experts_open_on_body(b->g,path,NULL,&b->body);
        CHECK(!bad,names[test]); nt_experts_free(bad);
        for(int i=0;i<n;i++) free(changed[i]);
    }
    for(int i=0;i<n;i++) free(records[i]);
    free(records); free(slots); gguf_close(g);
}

int main(void) {
    body source=open_body("source"), aligned=open_body("aligned"), wide=open_body("wider");
    founders_and_births(&source);
    transfer_and_legacy(&source,&aligned,&wide);
    identity_copy(&source);
    fixed_array_reader();
    malformed(&source);
    close_body(&wide); close_body(&aligned); close_body(&source);
    printf("Expert lineage: %d checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
