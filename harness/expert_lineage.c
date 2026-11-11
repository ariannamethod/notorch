/* Birth addresses and a durable ancestry ledger. No random draws: adding
 * provenance must preserve the numerical trajectory of an existing parliament. */
#include "harness/experts_internal.h"
#include "sha256.h"
#include <stdlib.h>
#include <string.h>

enum { RECORD_BYTES = 175 };

static int hex_string(const char *s, size_t n) {
    return s && strlen(s)==n && strspn(s,"0123456789abcdef")==n;
}

static void hex_digest(const void *p,size_t n,char out[65]) {
    unsigned char d[32];
    nt_sha256(p,n,d);
    for(int i=0;i<32;i++) snprintf(out+2*i,3,"%02x",d[i]);
}

static int reserve(nt_experts *p) {
    if(p->ancestor_count >= (int)GGUF_MAX_STR_ARRAY) return NT_E_CAPACITY;
    if(p->ancestor_count < p->ancestor_capacity) return NT_OK;
    int cap=p->ancestor_capacity ? p->ancestor_capacity*2 : 8;
    if(cap>(int)GGUF_MAX_STR_ARRAY) cap=GGUF_MAX_STR_ARRAY;
    nt_lineage_entry *next=realloc(p->ancestors,(size_t)cap*sizeof(*next));
    if(!next) return NT_E_MEMORY;
    p->ancestors=next; p->ancestor_capacity=cap;
    return NT_OK;
}

static int birth_address(nt_experts *p,int slot,int parent,int kind) {
    int rc=reserve(p);
    if(rc) return rc;
    nt_lineage_entry e={0};
    e.kind=kind; e.turn=kind==NT_LINEAGE_FOUNDER ? 0 : p->turns+1;
    e.parent=parent<0 ? 0 : (uint32_t)p->seat_ancestor[parent];
    if(parent>=0 && (!e.parent || e.parent>(uint32_t)p->ancestor_count)) return NT_E_STATE;
    memcpy(e.source_host,p->host,65);
    unsigned char material[512], hashes[96];
    size_t dr=(size_t)p->dim*RANK;
    nt_sha256(p->a+slot*dr,dr*sizeof(float),hashes);
    nt_sha256(p->b+slot*dr,dr*sizeof(float),hashes+32);
    nt_sha256(p->vote+(size_t)slot*p->dim,(size_t)p->dim*sizeof(float),hashes+64);
    int n=snprintf((char *)material,sizeof(material),"notorch-doe-lineage-v1:%d:%s:%s:%d:%d:%llu:%d:",
        kind,p->host,e.parent ? p->ancestors[e.parent-1].id : "root",slot,p->ancestor_count,
        (unsigned long long)e.turn,p->dim);
    if(n<0 || (size_t)n+sizeof hashes>sizeof material) return NT_E_STATE;
    memcpy(material+n,hashes,sizeof hashes);
    hex_digest(material,(size_t)n+sizeof hashes,e.id);
    p->ancestors[p->ancestor_count++]=e;
    p->seat_ancestor[slot]=p->ancestor_count;
    return NT_OK;
}

int nt_lineage_founders(nt_experts *p) {
    for(int i=0;i<FOUNDERS;i++) {
        int rc=birth_address(p,i,-1,NT_LINEAGE_FOUNDER);
        if(rc) return rc;
    }
    return NT_OK;
}

int nt_lineage_birth(nt_experts *p,int parent,int child) {
    int rc=birth_address(p,child,parent,NT_LINEAGE_BIRTH);
    if(!rc && p->trace) fprintf(p->trace,"doe lineage-birth id=%s parent=%s turn=%llu slot=%d\n",
        p->ancestors[p->seat_ancestor[child]-1].id,
        p->ancestors[p->seat_ancestor[parent]-1].id,
        (unsigned long long)p->turns+1,child);
    return rc;
}

void nt_lineage_retire(nt_experts *p,int seat) {
    int n=p->seat_ancestor[seat];
    if(n>0 && n<=p->ancestor_count) {
        p->ancestors[n-1].death_turn=p->turns+1;
        if(p->trace) fprintf(p->trace,"doe lineage-death id=%s turn=%llu slot=%d\n",
            p->ancestors[n-1].id,(unsigned long long)p->turns+1,seat);
    }
}

int nt_experts_ancestor_count(const nt_experts *p) { return p ? p->ancestor_count : 0; }

int nt_experts_ancestor(const nt_experts *p,int index,nt_expert_identity *out) {
    if(!p || !out || index<0 || index>=p->ancestor_count) return NT_E_ARG;
    const nt_lineage_entry *e=p->ancestors+index;
    nt_expert_identity value={0};
    memcpy(value.id,e->id,65); memcpy(value.source_host,e->source_host,65);
    if(e->parent) memcpy(value.parent,p->ancestors[e->parent-1].id,65);
    value.turn=e->turn; value.death_turn=e->death_turn; value.kind=e->kind;
    *out=value; return NT_OK;
}

int nt_experts_seat_identity(const nt_experts *p,int slot,nt_expert_identity *out) {
    if(!p || slot<0 || slot>=SEATS || !out) return NT_E_ARG;
    if(!p->seat_ancestor[slot]) { memset(out,0,sizeof(*out)); return NT_OK; }
    return nt_experts_ancestor(p,p->seat_ancestor[slot]-1,out);
}

int nt_experts_find_identity(const nt_experts *p,const char *id) {
    if(!p || !hex_string(id,64)) return -1;
    for(int e=0;e<SEATS;e++) if(p->alive[e] && p->seat_ancestor[e] &&
        !strcmp(p->ancestors[p->seat_ancestor[e]-1].id,id)) return e;
    return -1;
}

int nt_experts_copy_factors_by_identity(nt_experts *dst,const nt_experts *src,const char *id) {
    int a=nt_experts_find_identity(dst,id), b=nt_experts_find_identity(src,id);
    if(a<0 || b<0) return NT_E_ARG;
    if(!dst->turns || !src->turns || dst->count || src->count || dst->choices || src->choices ||
       dst->dim!=src->dim || dst->layer!=src->layer || strcmp(dst->host,src->host)) return NT_E_STATE;
    size_t n=(size_t)dst->dim*RANK;
    memmove(dst->a+a*n,src->a+b*n,n*sizeof(float));
    memmove(dst->b+a*n,src->b+b*n,n*sizeof(float));
    return NT_OK;
}

int nt_lineage_write(const nt_experts *p,gguf_writer *w) {
    int n=p->ancestor_count;
    char **records=calloc((size_t)n,sizeof(*records));
    if(!records) return -1;
    int rc=0;
    for(int i=0;i<n;i++) {
        const nt_lineage_entry *e=p->ancestors+i;
        records[i]=malloc(RECORD_BYTES);
        if(!records[i]) { rc=-1; break; }
        int wrote=snprintf(records[i],RECORD_BYTES,"%c %016llx %016llx %08x %s %s",
            e->kind==NT_LINEAGE_FOUNDER ? 'F' : e->kind==NT_LINEAGE_BIRTH ? 'B' : 'A',
            (unsigned long long)e->turn,(unsigned long long)e->death_turn,
            e->parent,e->id,e->source_host);
        if(wrote!=RECORD_BYTES-1) { rc=-1; break; }
    }
    if(!rc) rc=gguf_write_kv_u32(w,"doe.lineage.version",1) ||
        gguf_write_kv_str_array(w,"doe.lineage.entries",(const char *const *)records,(uint64_t)n) ||
        gguf_write_kv_i32_array(w,"doe.lineage.seats",p->seat_ancestor,SEATS);
    for(int i=0;i<n;i++) free(records[i]);
    free(records); return rc ? -1 : 0;
}

static int parse_hex(const char *s,int n,uint64_t *out) {
    static const char hex[]="0123456789abcdef";
    uint64_t v=0;
    for(int i=0;i<n;i++) {
        const char *at=strchr(hex,s[i]);
        if(!s[i] || !at) return -1;
        v=(v<<4)|(uint64_t)(at-hex);
    }
    *out=v; return 0;
}

static int compare_ids(const void *a,const void *b) {
    const nt_lineage_entry *const *x=a,*const *y=b;
    return strcmp((*x)->id,(*y)->id);
}

int nt_lineage_read(nt_experts *p,const gguf_file *g,uint64_t turns) {
    const gguf_kv *v=gguf_get_kv(g,"doe.lineage.version");
    if(!v || v->type!=4 || v->val.u32!=1) return -1;
    int n=0,ns=0;
    char **records=gguf_read_fixed_str_array(g->path,"doe.lineage.entries",RECORD_BYTES-1,&n);
    int32_t *slots=gguf_read_i32_array(g->path,"doe.lineage.seats",&ns);
    nt_lineage_entry *entries=NULL;
    nt_lineage_entry **sorted=NULL;
    int rc=-1;
    if(!records || n<2 || !slots || ns!=SEATS) goto done;
    entries=calloc((size_t)n,sizeof(*entries)); sorted=malloc((size_t)n*sizeof(*sorted));
    if(!entries || !sorted) goto done;
    for(int i=0;i<n;i++) {
        const char *s=records[i]; nt_lineage_entry *e=entries+i;
        uint64_t parent;
        if(strlen(s)!=RECORD_BYTES-1 || s[1]!=' ' || s[18]!=' ' || s[35]!=' ' || s[44]!=' ' || s[109]!=' ' ||
           parse_hex(s+2,16,&e->turn) || parse_hex(s+19,16,&e->death_turn) || parse_hex(s+36,8,&parent)) goto done;
        e->kind=s[0]=='F' ? NT_LINEAGE_FOUNDER : s[0]=='B' ? NT_LINEAGE_BIRTH : s[0]=='A' ? NT_LINEAGE_ADOPTED : 0;
        e->parent=(uint32_t)parent;
        memcpy(e->id,s+45,64); memcpy(e->source_host,s+110,64);
        if(!e->kind || !hex_string(e->id,64) || !hex_string(e->source_host,64) || e->turn>turns ||
           (e->death_turn && e->death_turn!=UINT64_MAX && (e->death_turn<e->turn || e->death_turn>turns)) ||
           (e->death_turn==UINT64_MAX && e->kind!=NT_LINEAGE_ADOPTED)) goto done;
        if(e->kind==NT_LINEAGE_BIRTH) {
            if(!e->turn || !e->parent || e->parent>(uint32_t)i || entries[e->parent-1].turn>e->turn ||
               entries[e->parent-1].death_turn==UINT64_MAX ||
               (entries[e->parent-1].death_turn && entries[e->parent-1].death_turn<e->turn)) goto done;
        } else if(e->parent || (e->kind==NT_LINEAGE_FOUNDER && e->turn)) goto done;
        sorted[i]=e;
    }
    qsort(sorted,(size_t)n,sizeof(*sorted),compare_ids);
    for(int i=1;i<n;i++) if(!strcmp(sorted[i-1]->id,sorted[i]->id)) goto done;
    for(int e=0;e<SEATS;e++) {
        int idx=slots[e];
        if(idx<0 || idx>n || (p->alive[e] && (!idx || entries[idx-1].death_turn)) ||
           (!p->alive[e] && idx && !entries[idx-1].death_turn)) goto done;
        for(int j=0;j<e;j++) if(idx && idx==slots[j]) goto done;
    }
    for(int i=0;i<n;i++) if(!entries[i].death_turn) {
        int found=0;
        for(int e=0;e<SEATS;e++) found+=slots[e]==i+1 && p->alive[e];
        if(found!=1) goto done;
    }
    p->ancestors=entries; entries=NULL; p->ancestor_count=p->ancestor_capacity=n;
    memcpy(p->seat_ancestor,slots,sizeof(p->seat_ancestor)); rc=0;
done:
    for(int i=0;i<n;i++) free(records[i]);
    free(records); free(slots); free(entries); free(sorted); return rc;
}

int nt_lineage_adopt(nt_experts *p,const char *spore_hash,const char *source_host,
                     uint64_t turns,int width,const float *a,const float *b) {
    size_t dr=(size_t)width*RANK;
    for(int slot=0;slot<SEATS;slot++) {
        int used=p->alive[slot]!=0 || p->age[slot]!=0;
        for(size_t i=0;!used && i<dr;i++) used=a[slot*dr+i]!=0 || b[slot*dr+i]!=0;
        if(!used) continue;
        int rc=reserve(p); if(rc) return rc;
        nt_lineage_entry e={0}; e.kind=NT_LINEAGE_ADOPTED; e.turn=turns;
        e.death_turn=p->alive[slot] ? 0 : UINT64_MAX;
        memcpy(e.source_host,source_host,65);
        char material[160];
        int n=snprintf(material,sizeof material,"notorch-doe-lineage-v1:adopt:%s:%d",spore_hash,slot);
        hex_digest(material,(size_t)n,e.id);
        p->ancestors[p->ancestor_count++]=e; p->seat_ancestor[slot]=p->ancestor_count;
    }
    return NT_OK;
}
