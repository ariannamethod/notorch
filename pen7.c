/* Snapshot ownership and tokenizer policy for the PEN7 inference family.
 * Family arithmetic stays in harness/arch_penelope.c; this file owns loading,
 * identity, immutable handles, bounded workspaces and atomic publication. */
#include "pen7.h"
#include "harness/penelope_internal.h"
#include "sha256.h"
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PEN7_FILE_LIMIT ((size_t)512*1024*1024)
#define PEN7_WORK_LIMIT ((size_t)256*1024*1024)
#define PEN7_PIECE_LIMIT ((size_t)32*1024*1024)
#define PEN7_INPUT_LIMIT 8191
struct nt_pen7_model {
    atomic_uint refs;
    nt_pen7_info info;
    unsigned char *snapshot;
    gguf_file *gf;
    pen_model *body;
    int32_t *pairs;
    unsigned char **pieces;
    size_t *piece_lengths;
};
struct nt_pen7_workspace {
    nt_pen7_model *owner;
    size_t max_tokens, bytes;
    kv_cache kv;
    float *memory, *scratch, *mix, *logits, *repr;
};

typedef struct { const unsigned char *p; size_t n; } span;
typedef struct { const unsigned char *p; size_t n, at; } cursor;
static int take(cursor *r, void *out, size_t n) {
    if (n > r->n-r->at) return 0;
    if (out) memcpy(out, r->p+r->at, n);
    r->at += n;
    return 1;
}
static int u32(cursor *r, uint32_t *v) { return take(r,v,4); }
static int u64(cursor *r, uint64_t *v) { return take(r,v,8); }
static int string(cursor *r, span *s) {
    uint64_t n;
    if (!u64(r,&n) || n > r->n-r->at) return 0;
    *s = (span){r->p+r->at,(size_t)n};
    return take(r,NULL,(size_t)n);
}
static int copied(span s, char *out, size_t cap) {
    if (s.n >= cap || memchr(s.p,0,s.n)) return 0;
    memcpy(out,s.p,s.n); out[s.n]=0; return 1;
}
static int skip(cursor *r, uint32_t type) {
    static const unsigned char sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type == 8) { span s; return string(r,&s); }
    return type < sizeof(sizes) && sizes[type] && take(r,NULL,sizes[type]);
}
static int digest(const void *data, size_t bytes, char out[65]) {
    unsigned char d[32]; static const char h[]="0123456789abcdef";
    if (nt_sha256(data,bytes,d)) return 0;
    for (int i=0;i<32;++i) { out[2*i]=h[d[i]>>4]; out[2*i+1]=h[d[i]&15]; }
    out[64]=0; return 1;
}
static int keystr(const gguf_file *gf, const char *key, const char *value) {
    const gguf_kv *v=gguf_get_kv(gf,key);
    return v && v->type==8 && !strcmp(v->val.str,value);
}
static int keybool(const gguf_file *gf, const char *key, int value) {
    const gguf_kv *v=gguf_get_kv(gf,key);
    return v && ((v->type==7 && v->val.b==value) ||
                 (v->type==4 && v->val.u32==(uint32_t)value));
}
/* Strict bounded container reader over the same owned bytes that are hashed.
 * It neither reopens the path for arrays nor changes shared GGUF loader policy. */
static int parse(nt_pen7_model *m, size_t bytes, span *tokens, span *merges, span *pairs) {
    gguf_file *g=m->gf=calloc(1,sizeof(*g));
    if (!g) return NT_PEN7_E_MEMORY;
    cursor r={m->snapshot,bytes,0}; uint32_t magic;
    if (!u32(&r,&magic) || magic!=GGUF_MAGIC || !u32(&r,&g->version) || g->version!=3 ||
        !u64(&r,&g->n_tensors) || !u64(&r,&g->n_kv) || !g->n_tensors ||
        g->n_tensors>4+11*PEN_MAX_BLOCKS || g->n_kv>GGUF_MAX_KV) return NT_PEN7_E_FORMAT;
    for (uint64_t i=0;i<g->n_kv;++i) {
        gguf_kv *k=&g->kv[g->n_kv_parsed]; span key;
        if (!string(&r,&key) || !copied(key,k->key,sizeof(k->key)) || !u32(&r,&k->type)) return NT_PEN7_E_FORMAT;
        for (int j=0;j<g->n_kv_parsed;++j) if (!strcmp(g->kv[j].key,k->key)) return NT_PEN7_E_FORMAT;
        if (k->type==9) {
            size_t start=r.at; uint32_t type; uint64_t count;
            if (!u32(&r,&type) || !u64(&r,&count) || count>2*65536 || type==9) return NT_PEN7_E_FORMAT;
            for (uint64_t j=0;j<count;++j) if (!skip(&r,type)) return NT_PEN7_E_FORMAT;
            span array={r.p+start,r.at-start};
            if (!strcmp(k->key,"tokenizer.ggml.tokens")) *tokens=array;
            else if (!strcmp(k->key,"tokenizer.ggml.merges")) *merges=array;
            else if (!strcmp(k->key,"tokenizer.penelope.merge_pairs")) *pairs=array;
            else if (!strcmp(k->key,"tokenizer.ggml.token_type")) return NT_PEN7_E_FORMAT;
        } else if (k->type==8) {
            span s; if (!string(&r,&s) || !copied(s,k->val.str,sizeof(k->val.str))) return NT_PEN7_E_FORMAT;
        } else {
            static const unsigned char sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (k->type>=sizeof(sizes) || !sizes[k->type] || !take(&r,&k->val,sizes[k->type])) return NT_PEN7_E_FORMAT;
        }
        g->n_kv_parsed++;
    }
    g->kv_end=r.at;
    const gguf_kv *arch=gguf_get_kv(g,"general.architecture");
    if (!arch || arch->type!=8 || strcmp(arch->val.str,"penelope")) return NT_PEN7_E_FORMAT;
    strcpy(g->arch,"penelope");
    for (uint64_t i=0;i<g->n_tensors;++i) {
        gguf_tensor_info *t=&g->tensors[i]; span name;
        if (!string(&r,&name) || !copied(name,t->name,sizeof(t->name)) ||
            !u32(&r,&t->ndim) || t->ndim<1 || t->ndim>2) return NT_PEN7_E_FORMAT;
        for (uint64_t j=0;j<i;++j) if (!strcmp(g->tensors[j].name,t->name)) return NT_PEN7_E_FORMAT;
        t->n_elements=1;
        for (uint32_t j=0;j<t->ndim;++j) {
            if (!u64(&r,&t->shape[j]) || !t->shape[j] || t->shape[j]>UINT64_MAX/t->n_elements) return NT_PEN7_E_FORMAT;
            t->n_elements*=t->shape[j];
        }
        if (!u32(&r,&t->dtype) || !u64(&r,&t->offset) ||
            (t->dtype!=GGUF_TYPE_F32 && t->dtype!=GGUF_TYPE_Q8_0) ||
            t->n_elements>PEN7_FILE_LIMIT || (t->dtype==GGUF_TYPE_Q8_0 && t->shape[0]%32)) return NT_PEN7_E_FORMAT;
    }
    if (r.at>SIZE_MAX-31) return NT_PEN7_E_FORMAT;
    g->data_offset=(r.at+31)&~(size_t)31;
    if (g->data_offset>bytes) return NT_PEN7_E_FORMAT;
    g->data=m->snapshot+(size_t)g->data_offset;
    g->data_size=bytes-(size_t)g->data_offset;
    uint64_t params=0;
    for (uint64_t i=0;i<g->n_tensors;++i) {
        const gguf_tensor_info *t=&g->tensors[i]; uint64_t n=gguf_type_size(t->dtype,t->n_elements);
        if (!n || t->offset%32 || t->offset>g->data_size || n>g->data_size-t->offset) return NT_PEN7_E_FORMAT;
        params+=t->n_elements;
        if (params>PEN7_FILE_LIMIT/sizeof(float)) return NT_PEN7_E_FORMAT;
        for (uint64_t j=0;j<i;++j) {
            const gguf_tensor_info *s=&g->tensors[j]; uint64_t z=gguf_type_size(s->dtype,s->n_elements);
            if (t->offset<s->offset+z && s->offset<t->offset+n) return NT_PEN7_E_FORMAT;
        }
    }
    return NT_PEN7_OK;
}
static int encoded_piece(span encoded, const unsigned char *piece, size_t bytes) {
    size_t at=0;
    for (size_t j=0;j<bytes;++j) {
        int b=piece[j], cp=b;
        if (!((b>=33&&b<=126)||(b>=161&&b<=172)||b>=174)) {
            int extra=0;
            for (int k=0;k<b;++k) if (!((k>=33&&k<=126)||(k>=161&&k<=172)||k>=174)) extra++;
            cp=256+extra;
        }
        unsigned char s[2]; size_t n;
        if (cp<128) { s[0]=(unsigned char)cp; n=1; }
        else { s[0]=(unsigned char)(0xc0|(cp>>6)); s[1]=(unsigned char)(0x80|(cp&63)); n=2; }
        if (n>encoded.n-at || memcmp(encoded.p+at,s,n)) return 0;
        at+=n;
    }
    return at==encoded.n;
}
static int tokenizer(nt_pen7_model *m, span toks, span merges, span pairs) {
    const gguf_file *g=m->gf; int v=m->body->vocab;
    const gguf_kv *limit=gguf_get_kv(g,"tokenizer.penelope.max_input_bytes");
    if (v<256 || v>65536 || !toks.p || !merges.p || !pairs.p ||
        !keystr(g,"tokenizer.ggml.model","penelope") ||
        !keystr(g,"tokenizer.penelope.algorithm","byte-rank-pass-v1") ||
        !keybool(g,"tokenizer.ggml.add_bos_token",0) || !keybool(g,"tokenizer.ggml.add_eos_token",0) ||
        !keybool(g,"tokenizer.penelope.lowercase_ascii",1) ||
        !limit || limit->type!=4 || limit->val.u32!=PEN7_INPUT_LIMIT) return NT_PEN7_E_FORMAT;
    cursor r={pairs.p,pairs.n,0}; uint32_t type; uint64_t n;
    if (!u32(&r,&type) || type!=5 || !u64(&r,&n) || n!=(uint64_t)2*(v-256)) return NT_PEN7_E_FORMAT;
    m->pairs=malloc(n ? (size_t)n*4 : 1);
    if (!m->pairs) return NT_PEN7_E_MEMORY;
    if (!take(&r,m->pairs,(size_t)n*4) || !digest(pairs.p+12,(size_t)n*4,m->info.tokenizer_identity) ||
        !keystr(g,"tokenizer.penelope.sha256",m->info.tokenizer_identity)) return NT_PEN7_E_FORMAT;
    m->pieces=calloc((size_t)v,sizeof(*m->pieces));
    m->piece_lengths=calloc((size_t)v,sizeof(*m->piece_lengths));
    if (!m->pieces || !m->piece_lengths) return NT_PEN7_E_MEMORY;
    r=(cursor){toks.p,toks.n,0};
    if (!u32(&r,&type) || type!=8 || !u64(&r,&n) || n!=(uint64_t)v) return NT_PEN7_E_FORMAT;
    cursor mr={merges.p,merges.n,0};
    if (!u32(&mr,&type) || type!=8 || !u64(&mr,&n) || n!=(uint64_t)v-256) return NT_PEN7_E_FORMAT;
    size_t total=0;
    for (int i=0;i<v;++i) {
        int l=0,rr=0; size_t len=1;
        if (i>=256) {
            l=m->pairs[2*(i-256)]; rr=m->pairs[2*(i-256)+1];
            if (l<0 || rr<0 || l>=i || rr>=i) return NT_PEN7_E_FORMAT;
            len=m->piece_lengths[l]+m->piece_lengths[rr];
        }
        if (len>PEN7_INPUT_LIMIT || len>PEN7_PIECE_LIMIT-total) return NT_PEN7_E_FORMAT;
        total+=len;
        m->pieces[i]=malloc(len); m->piece_lengths[i]=len;
        if (!m->pieces[i]) return NT_PEN7_E_MEMORY;
        if (i<256) m->pieces[i][0]=(unsigned char)i;
        else {
            memcpy(m->pieces[i],m->pieces[l],m->piece_lengths[l]);
            memcpy(m->pieces[i]+m->piece_lengths[l],m->pieces[rr],m->piece_lengths[rr]);
        }
        span surface;
        if (!string(&r,&surface) || !encoded_piece(surface,m->pieces[i],len)) return NT_PEN7_E_FORMAT;
        if (i>=256) {
            span merge; if (!string(&mr,&merge)) return NT_PEN7_E_FORMAT;
            const unsigned char *space=memchr(merge.p,' ',merge.n);
            if (!space || !encoded_piece((span){merge.p,(size_t)(space-merge.p)},m->pieces[l],m->piece_lengths[l]) ||
                !encoded_piece((span){space+1,merge.n-(size_t)(space+1-merge.p)},m->pieces[rr],m->piece_lengths[rr])) return NT_PEN7_E_FORMAT;
        }
    }
    return NT_PEN7_OK;
}
const char *nt_pen7_error_string(int s) {
    static const char *const msg[]={"ok","invalid argument","cannot read model",
        "invalid PEN7 model or tokenizer","allocation failed","capacity exceeded",
        "nonfinite inference arithmetic","incompatible model or workspace","invalid token id"};
    return s>=0 && (size_t)s<sizeof(msg)/sizeof(*msg) ? msg[s] : "unknown PEN7 error";
}
void nt_pen7_release(nt_pen7_model *m) {
    if (!m || atomic_fetch_sub_explicit(&m->refs,1,memory_order_acq_rel)!=1) return;
    if (m->pieces) for (int i=0;i<m->info.vocab;++i) free(m->pieces[i]);
    free(m->pieces); free(m->piece_lengths); free(m->pairs);
    nt_penelope_drop(m->body); free(m->gf); free(m->snapshot); free(m);
}
nt_pen7_model *nt_pen7_retain(nt_pen7_model *m) {
    if (!m) return NULL;
    unsigned n=atomic_load_explicit(&m->refs,memory_order_relaxed);
    do { if (n==UINT_MAX) return NULL; }
    while (!atomic_compare_exchange_weak_explicit(&m->refs,&n,n+1,memory_order_relaxed,memory_order_relaxed));
    return m;
}
const nt_pen7_info *nt_pen7_get_info(const nt_pen7_model *m) { return m ? &m->info : NULL; }
int nt_pen7_load(const char *path, nt_pen7_model **out) {
    uint32_t endian=1;
    if (!path || !*path || !out) return NT_PEN7_E_ARG;
    if (*(unsigned char *)&endian!=1 || sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24) return NT_PEN7_E_FORMAT;
    /* A FIFO is not a model and must not wait for a writer before its type can
     * be refused. Nonblocking has no effect on regular-file snapshot reads. */
    int fd=open(path,O_RDONLY|O_NONBLOCK); if (fd<0) return NT_PEN7_E_IO;
    struct stat st;
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<24 || (uint64_t)st.st_size>PEN7_FILE_LIMIT) {
        close(fd); return NT_PEN7_E_FORMAT;
    }
    FILE *f=fdopen(fd,"rb"); if (!f) { close(fd); return NT_PEN7_E_IO; }
    nt_pen7_model *m=calloc(1,sizeof(*m));
    if (!m) { fclose(f); return NT_PEN7_E_MEMORY; }
    atomic_init(&m->refs,1);
    size_t bytes=(size_t)st.st_size;
    m->snapshot=malloc(bytes);
    int rc=NT_PEN7_E_MEMORY;
    if (!m->snapshot) { fclose(f); goto fail; }
    int read_ok=fread(m->snapshot,1,bytes,f)==bytes && fgetc(f)==EOF && !ferror(f);
    fclose(f);
    if (!read_ok) { rc=NT_PEN7_E_IO; goto fail; }
    if (!digest(m->snapshot,bytes,m->info.identity)) { rc=NT_PEN7_E_FORMAT; goto fail; }
    span toks={0},merges={0},pairs={0};
    rc=parse(m,bytes,&toks,&merges,&pairs); if (rc) goto fail;
    nt_dims dims;
    m->body=nt_penelope_load_quiet(m->gf,&dims);
    if (!m->body) { rc=NT_PEN7_E_FORMAT; goto fail; }
    const pen_model *b=m->body;
    m->info.vocab=b->vocab; m->info.width=b->embed; m->info.context=b->ctx;
    m->info.layers=b->layers; m->info.ffn=b->ffn; m->info.heads=b->heads;
    rc=tokenizer(m,toks,merges,pairs); if (rc) goto fail;
    *out=m; return NT_PEN7_OK;
fail:
    nt_pen7_release(m); return rc;
}
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    if (!a || !b || !an || !bn) return 0;
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y ? y-x<an : x-y<bn;
}
int nt_pen7_encode(const nt_pen7_model *m, const char *text, size_t bytes,
                   int *ids, size_t capacity, size_t *count) {
    if (!m || (!text && bytes) || !count || (!ids && capacity)) return NT_PEN7_E_ARG;
    if (bytes>PEN7_INPUT_LIMIT) return NT_PEN7_E_CAPACITY;
    int seq[PEN7_INPUT_LIMIT]; size_t len=bytes;
    for (size_t i=0;i<bytes;++i) { unsigned char c=(unsigned char)text[i]; seq[i]=c>='A'&&c<='Z' ? c-'A'+'a' : c; }
    for (int r=0;r<m->info.vocab-256;++r) {
        int left=m->pairs[2*r],right=m->pairs[2*r+1]; size_t used=0;
        for (size_t i=0;i<len;++i) {
            if (i+1<len && seq[i]==left && seq[i+1]==right) { seq[used++]=256+r; i++; }
            else seq[used++]=seq[i];
        }
        len=used;
    }
    if (len>capacity) return NT_PEN7_E_CAPACITY;
    if (overlap(ids,len*sizeof(int),count,sizeof(*count))) return NT_PEN7_E_ARG;
    if (len) memcpy(ids,seq,len*sizeof(int));
    *count=len; return NT_PEN7_OK;
}
const unsigned char *nt_pen7_token_piece(const nt_pen7_model *m, int id, size_t *bytes) {
    if (!m || !bytes || id<0 || id>=m->info.vocab) return NULL;
    *bytes=m->piece_lengths[id]; return m->pieces[id];
}
static int compatible(const nt_pen7_model *a, const nt_pen7_model *b) {
    return a && b && a->info.vocab==b->info.vocab && a->info.width==b->info.width &&
        a->info.context==b->info.context && a->info.layers==b->info.layers &&
        a->info.ffn==b->info.ffn && a->info.heads==b->info.heads &&
        !strcmp(a->info.tokenizer_identity,b->info.tokenizer_identity);
}
int nt_pen7_workspace_new(nt_pen7_model *m, size_t n, nt_pen7_workspace **out) {
    if (!m || !out || !n) return NT_PEN7_E_ARG;
    if (n>(size_t)m->info.context) return NT_PEN7_E_CAPACITY;
    const pen_model *b=m->body;
    uint64_t kv=(uint64_t)n*b->layers*b->embed;
    uint64_t scratch=6*(uint64_t)n*b->embed+2*(uint64_t)n*b->ffn+2*(uint64_t)n*b->hd;
    uint64_t rows=b->vocab>b->ffn ? (uint64_t)b->vocab : (uint64_t)b->ffn;
    if (rows<(uint64_t)b->embed) rows=b->embed;
    uint64_t mix=rows*b->embed, floats=2*kv+scratch+mix+b->vocab+b->embed;
    if (floats>(PEN7_WORK_LIMIT-sizeof(nt_pen7_workspace))/sizeof(float)) return NT_PEN7_E_CAPACITY;
    nt_pen7_workspace *w=calloc(1,sizeof(*w));
    if (!w) return NT_PEN7_E_MEMORY;
    w->memory=malloc((size_t)floats*sizeof(float));
    if (!w->memory || !nt_pen7_retain(m)) { free(w->memory); free(w); return NT_PEN7_E_MEMORY; }
    w->owner=m; w->max_tokens=n; w->bytes=sizeof(*w)+(size_t)floats*sizeof(float);
    w->kv=(kv_cache){.k=w->memory,.v=w->memory+kv,.max_seq=(int)n,.n_layers=b->layers,.kv_dim=b->embed};
    w->scratch=w->memory+2*kv; w->mix=w->scratch+scratch;
    w->logits=w->mix+mix; w->repr=w->logits+b->vocab;
    *out=w; return NT_PEN7_OK;
}
void nt_pen7_workspace_free(nt_pen7_workspace *w) {
    if (!w) return;
    nt_pen7_release(w->owner); free(w->memory); free(w);
}
size_t nt_pen7_workspace_bytes(const nt_pen7_workspace *w) { return w ? w->bytes : 0; }
int nt_pen7_mix_forward(const nt_pen7_model *a, const nt_pen7_model *b, float alpha,
                        nt_pen7_workspace *w, const int *ids, size_t n,
                        float *logits, size_t lc, float *repr, size_t hc) {
    if (!a || !b || !w || !ids || !n || (!logits&&!repr) ||
        (!logits&&lc) || (!repr&&hc) || !isfinite(alpha) || alpha<0 || alpha>1) return NT_PEN7_E_ARG;
    if (!compatible(a,b) || !compatible(a,w->owner)) return NT_PEN7_E_COMPAT;
    size_t v=(size_t)a->info.vocab,e=(size_t)a->info.width;
    if (n>w->max_tokens || (logits&&lc<v) || (repr&&hc<e)) return NT_PEN7_E_CAPACITY;
    if (overlap(logits,v*sizeof(float),repr,e*sizeof(float))) return NT_PEN7_E_ARG;
    for (size_t i=0;i<n;++i) if (ids[i]<0 || ids[i]>=a->info.vocab) return NT_PEN7_E_TOKEN;
    const pen_model *body=a->body,*other=b->body;
    if (alpha==0.f) { body=b->body; other=NULL; }
    else if (alpha==1.f) other=NULL;
    int rc=nt_penelope_eval(body,other,alpha,&w->kv,ids,(int)n,0,
                            w->logits,w->repr,w->scratch,w->mix,NULL,NULL);
    if (rc) return rc==NT_E_MEMORY ? NT_PEN7_E_MEMORY : rc==NT_E_STATE ? NT_PEN7_E_NUMERIC : NT_PEN7_E_ARG;
    for (size_t i=0;i<v;++i) if (!isfinite(w->logits[i])) return NT_PEN7_E_NUMERIC;
    for (size_t i=0;i<e;++i) if (!isfinite(w->repr[i])) return NT_PEN7_E_NUMERIC;
    if (logits) memcpy(logits,w->logits,v*sizeof(float));
    if (repr) memcpy(repr,w->repr,e*sizeof(float));
    return NT_PEN7_OK;
}
int nt_pen7_forward(const nt_pen7_model *m, nt_pen7_workspace *w,
                    const int *ids, size_t n, float *logits, size_t lc, float *repr, size_t hc) {
    return nt_pen7_mix_forward(m,m,1.f,w,ids,n,logits,lc,repr,hc);
}
