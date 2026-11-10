/* Qwen3.5 dense text decoder: gated attention + recurrent GatedDeltaNet.
 * Equations/layout: Qwen and Transformers modeling_qwen3_5.py (Apache-2.0).
 * The source-layout GGUF made by tools/qwen35_to_gguf.c retains BF16/F32
 * bytes. CPU arithmetic is F32; vision and quantized GGUF layouts are separate
 * integration boundaries. No external model runtime is called here. */
#include "harness/arch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int full, index;
    wt q, k, v, o, z, a, b, gate, up, down;
    float *in_norm, *ffn_norm, *q_norm, *k_norm, *conv, *dt, *decay, *norm;
} q35_layer;
typedef struct {
    gguf_file *gf;
    int L,E,F,V,H,KV,HD,R,KH,VH,KD,VD,CK,ctx,full_count,linear_count,emb;
    float eps,rope_base;
    wt head;
    float *norm;
    q35_layer *layers;
} q35_model;
typedef struct {
    kv_cache base; /* family-owned; k/v hold only full-attention layers */
    const q35_model *owner;
    int next;
    float *conv,*state;
} q35_cache;

static void *cells(size_t rows,size_t cols) {
    if(!rows || !cols || rows>SIZE_MAX/cols/sizeof(float)) return NULL;
    return calloc(rows*cols,sizeof(float));
}
static int mi(gguf_file *g,const char *key,int *out) {
    char name[128];snprintf(name,sizeof(name),"qwen35.%s",key);
    const gguf_kv *v=gguf_get_kv(g,name);
    if(!v || v->type!=4 || v->val.u32>2147483647u)return 0;
    *out=(int)v->val.u32;return 1;
}
static int mf(gguf_file *g,const char *key,float *out) {
    char name[128];snprintf(name,sizeof(name),"qwen35.%s",key);
    const gguf_kv *v=gguf_get_kv(g,name);
    if(!v || v->type!=6 || !isfinite(v->val.f32))return 0;
    *out=v->val.f32;return 1;
}
static int shape(gguf_file *g,const char *name,int rows,int cols,int ndim) {
    int i=gguf_find_tensor(g,name);if(i<0)return 0;
    gguf_tensor_info *t=&g->tensors[i];
    return (t->dtype==GGUF_TYPE_F32 || t->dtype==GGUF_TYPE_BF16) &&
        t->ndim==(uint32_t)ndim && t->shape[0]==(uint64_t)cols &&
        (ndim==1 || t->shape[1]==(uint64_t)rows);
}
static int mat(gguf_file *g,wt *w,const char *name,int rows,int cols) {
    if(!shape(g,name,rows,cols,2)) {
        fprintf(stderr,"qwen35: invalid matrix %s [%d,%d]\n",name,rows,cols);return 0;
    }
    w->rows=rows;w->cols=cols;
    /* Explicit F32 CPU path; embedding rows alone stay mapped. */
    w->f32=gguf_dequant(g,gguf_find_tensor(g,name));return w->f32!=NULL;
}
static float *vec(gguf_file *g,const char *name,int n,int offset) {
    if(!shape(g,name,1,n,1))return NULL;
    float *p=gguf_dequant(g,gguf_find_tensor(g,name));
    if(p && offset)for(int i=0;i<n;i++)p[i]+=1.0f;
    return p;
}
static void q35_free(void *ptr) {
    q35_model *m=ptr;if(!m)return;
    for(int l=0;m->layers && l<m->L;l++) {
        q35_layer *a=&m->layers[l];
        wt *ws[]={&a->q,&a->k,&a->v,&a->o,&a->z,&a->a,&a->b,&a->gate,&a->up,&a->down};
        for(int i=0;i<10;i++)free(ws[i]->f32);
        free(a->in_norm);free(a->ffn_norm);free(a->q_norm);free(a->k_norm);
        free(a->conv);free(a->dt);free(a->decay);free(a->norm);
    }
    free(m->head.f32);free(m->norm);free(m->layers);free(m);
}
static void *q35_load(gguf_file *g,nt_dims *dims) {
    if(!g || !dims || strcmp(g->arch,"qwen35"))return NULL;
    q35_model *m=calloc(1,sizeof(*m));if(!m)return NULL;m->gf=g;
    int version;
    #define I(key,field) if(!mi(g,key,&m->field))goto bad
    if(!mi(g,"source_layout_version",&version)) {
        fprintf(stderr,"qwen35: no qwen35.source_layout_version; this family reads what tools/qwen35_to_gguf "
                       "writes from the original checkpoint, not the llama.cpp qwen35 layout\n");
        q35_free(m);return NULL;
    }
    if(version!=1)goto bad;
    I("block_count",L);I("embedding_length",E);I("feed_forward_length",F);
    I("vocab_size",V);I("attention.head_count",H);I("attention.head_count_kv",KV);
    I("attention.head_dim",HD);I("rope.dimension_count",R);I("context_length",ctx);
    I("linear.key_heads",KH);I("linear.value_heads",VH);I("linear.key_dim",KD);
    I("linear.value_dim",VD);I("linear.conv_kernel",CK);
    #undef I
    if(!mf(g,"rms_epsilon",&m->eps) || !mf(g,"rope.freq_base",&m->rope_base))goto bad;
    if(m->L<1 || m->L>128 || m->E<1 || m->E>8192 || m->F<1 || m->F>65536 ||
       m->V<1 || m->V>1048576 || m->H<1 || m->H>128 || m->KV<1 || m->KV>m->H ||
       m->H%m->KV || m->HD<2 || m->HD>512 || m->R<2 || m->R>m->HD || m->R%2 ||
       m->KH<1 || m->KH>128 || m->VH<1 || m->VH>128 || m->VH%m->KH ||
       m->KD<1 || m->KD>512 || m->VD<1 || m->VD>512 || m->CK<1 || m->CK>32 ||
       m->ctx<1 || m->ctx>1048576 || m->eps<=0 || m->rope_base<=0)goto bad;
    if(!shape(g,"embed_tokens.weight",m->V,m->E,2))goto bad;
    m->emb=gguf_find_tensor(g,"embed_tokens.weight");
    m->norm=vec(g,"norm.weight",m->E,1);if(!m->norm)goto bad;
    const char *head=gguf_find_tensor(g,"lm_head.weight")>=0?"lm_head.weight":"embed_tokens.weight";
    if(!mat(g,&m->head,head,m->V,m->E))goto bad;
    m->layers=calloc((size_t)m->L,sizeof(*m->layers));if(!m->layers)goto bad;
    int Q=m->H*m->HD,K=m->KV*m->HD,KD=m->KH*m->KD,VD=m->VH*m->VD;
    for(int l=0;l<m->L;l++) {
        q35_layer *a=&m->layers[l];char key[128],name[128];
        snprintf(key,sizeof(key),"layer.%d.full_attention",l);
        if(!mi(g,key,&a->full) || (a->full!=0 && a->full!=1))goto bad;
        a->index=a->full?m->full_count++:m->linear_count++;
        #define W(field,s,r,c) do{snprintf(name,sizeof(name),"layers.%d." s,l);if(!mat(g,&a->field,name,r,c))goto bad;}while(0)
        #define N(field,s,n,off) do{snprintf(name,sizeof(name),"layers.%d." s,l);a->field=vec(g,name,n,off);if(!a->field)goto bad;}while(0)
        N(in_norm,"input_layernorm.weight",m->E,1);N(ffn_norm,"post_attention_layernorm.weight",m->E,1);
        W(gate,"mlp.gate_proj.weight",m->F,m->E);W(up,"mlp.up_proj.weight",m->F,m->E);
        W(down,"mlp.down_proj.weight",m->E,m->F);
        if(a->full) {
            W(q,"self_attn.q_proj.weight",2*Q,m->E);W(k,"self_attn.k_proj.weight",K,m->E);
            W(v,"self_attn.v_proj.weight",K,m->E);W(o,"self_attn.o_proj.weight",m->E,Q);
            N(q_norm,"self_attn.q_norm.weight",m->HD,1);N(k_norm,"self_attn.k_norm.weight",m->HD,1);
        } else {
            W(q,"linear_attn.in_proj_qkv.weight",2*KD+VD,m->E);
            W(z,"linear_attn.in_proj_z.weight",VD,m->E);
            W(a,"linear_attn.in_proj_a.weight",m->VH,m->E);W(b,"linear_attn.in_proj_b.weight",m->VH,m->E);
            W(o,"linear_attn.out_proj.weight",m->E,VD);
            N(dt,"linear_attn.dt_bias",m->VH,0);N(decay,"linear_attn.A_log",m->VH,0);
            N(norm,"linear_attn.norm.weight",m->VD,0);
            snprintf(name,sizeof(name),"layers.%d.linear_attn.conv1d.weight",l);
            int ti=gguf_find_tensor(g,name);
            if(ti<0)goto bad;
            gguf_tensor_info *t=&g->tensors[ti];
            if(t->ndim!=3 || t->shape[0]!=(uint64_t)m->CK || t->shape[1]!=1 ||
               t->shape[2]!=(uint64_t)(2*KD+VD) || (t->dtype!=GGUF_TYPE_F32 && t->dtype!=GGUF_TYPE_BF16))goto bad;
            a->conv=gguf_dequant(g,ti);if(!a->conv)goto bad;
        }
        #undef W
        #undef N
    }
    dims->n_layers=m->L;dims->kv_dim=0;dims->vocab=m->V;
    fprintf(stderr,"qwen35: E=%d L=%d full=%d delta=%d V=%d (F32 CPU)\n",m->E,m->L,m->full_count,m->linear_count,m->V);
    return m;
bad:
    fprintf(stderr,"qwen35: unsupported or malformed source-layout text GGUF\n");q35_free(m);return NULL;
}

static void q35_kv_free(void *model,kv_cache *kv) {
    (void)model;if(!kv)return;q35_cache *c=(q35_cache*)kv;
    free(c->base.k);free(c->base.v);free(c->conv);free(c->state);free(c);
}
static kv_cache *q35_kv_new(void *ptr,int max_seq) {
    q35_model *m=ptr;if(!m || max_seq<1 || max_seq>m->ctx)return NULL;
    q35_cache *c=calloc(1,sizeof(*c));if(!c)return NULL;
    c->owner=m;c->base.max_seq=max_seq;c->base.n_layers=m->L;c->base.kv_dim=-35;
    size_t full=(size_t)(m->full_count?m->full_count:1)*max_seq;
    c->base.k=cells(full,(size_t)m->KV*m->HD);c->base.v=cells(full,(size_t)m->KV*m->HD);
    size_t linear=m->linear_count?m->linear_count:1;
    c->conv=cells(linear*(2*(size_t)m->KH*m->KD+(size_t)m->VH*m->VD),(size_t)m->CK);
    c->state=cells(linear*(size_t)m->VH,(size_t)m->KD*m->VD);
    if(!c->base.k || !c->base.v || !c->conv || !c->state){q35_kv_free(m,&c->base);return NULL;}
    return &c->base;
}
static float sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
static float silu(float x){return x*sigmoid(x);}
static float softplus(float x){return x>20?x:log1pf(expf(x));}
static void mm(float *y,const wt *w,const float *x,int n){mm_t(y,x,w->f32,n,w->cols,w->rows);}

typedef struct {
    q35_model *m;q35_layer *a;q35_cache *cache;
    float *q,*z,*aa,*bb,*out;int n;
} delta_context;
static void delta_heads(void *ptr,int h0,int h1) {
    delta_context *c=ptr;q35_model *m=c->m;q35_layer *a=c->a;
    int KD=m->KH*m->KD,VD=m->VH*m->VD,C=2*KD+VD;
    for(int h=h0;h<h1;h++) {
        int kh=h/(m->VH/m->KH);
        float *S=c->cache->state+((size_t)a->index*m->VH+h)*m->KD*m->VD;
        for(int t=0;t<c->n;t++) {
            const float *q=c->q+(size_t)t*C+kh*m->KD,*k=q+KD,*v=c->q+(size_t)t*C+2*KD+h*m->VD;
            float qs=1.0f/sqrtf(dot_f32(q,q,m->KD)+1e-6f)/sqrtf((float)m->KD);
            float ks=1.0f/sqrtf(dot_f32(k,k,m->KD)+1e-6f);
            float decay=expf(-expf(a->decay[h])*softplus(c->aa[(size_t)t*m->VH+h]+a->dt[h]));
            float beta=sigmoid(c->bb[(size_t)t*m->VH+h]);
            float *out=c->out+(size_t)t*VD+h*m->VD;
            float delta[512];memset(out,0,(size_t)m->VD*sizeof(float));
            for(int j=0;j<m->VD;j++)delta[j]=0;
            for(int i=0;i<m->KD;i++) {
                float *row=S+(size_t)i*m->VD;
                for(int j=0;j<m->VD;j++){row[j]*=decay;delta[j]+=row[j]*(k[i]*ks);}
            }
            for(int j=0;j<m->VD;j++)delta[j]=(v[j]-delta[j])*beta;
            for(int i=0;i<m->KD;i++) {
                float *row=S+(size_t)i*m->VD;
                for(int j=0;j<m->VD;j++){row[j]+=(k[i]*ks)*delta[j];out[j]+=row[j]*(q[i]*qs);}
            }
            rmsnorm(out,out,a->norm,m->VD,m->eps);
            for(int j=0;j<m->VD;j++)out[j]*=silu(c->z[(size_t)t*VD+(size_t)h*m->VD+j]);
        }
    }
}
typedef struct {
    q35_model *m;q35_layer *a;q35_cache *cache;
    float *q,*out,*scores;int n,pos;
} attention_context;
static void attention_heads(void *ptr,int h0,int h1) {
    attention_context *c=ptr;q35_model *m=c->m;int Q=m->H*m->HD,K=m->KV*m->HD;
    for(int h=h0;h<h1;h++) {
        float *scores=c->scores+(size_t)h*c->cache->base.max_seq;
        size_t off=(size_t)c->a->index*c->cache->base.max_seq*K+(size_t)(h/(m->H/m->KV))*m->HD;
        for(int t=0;t<c->n;t++) {
            const float *q=c->q+(size_t)t*2*Q+h*2*m->HD;
            float *out=c->out+(size_t)t*Q+h*m->HD;
            int end=c->pos+t+1;memset(out,0,(size_t)m->HD*sizeof(float));
            for(int j=0;j<end;j++)scores[j]=dot_f32(q,c->cache->base.k+off+(size_t)j*K,m->HD)/sqrtf((float)m->HD);
            softmax(scores,end);
            for(int j=0;j<end;j++)axpy_f32(out,scores[j],c->cache->base.v+off+(size_t)j*K,m->HD);
            for(int j=0;j<m->HD;j++)out[j]*=sigmoid(q[m->HD+j]);
        }
    }
}
static int q35_forward_hooks(void *ptr,kv_cache *kv,const int *tokens,int n,int pos,
                             float *logits,nt_residual_fn post_attention,nt_residual_fn post_block,void *user) {
    q35_model *m=ptr;if(!m)return NT_E_ARG;
    int rc=nt_check_call(kv,tokens,n,pos,m->V,m->L,0);if(rc)return rc;
    if(kv->kv_dim!=-35)return NT_E_CACHE;
    q35_cache *cache=(q35_cache*)kv;if(cache->owner!=m)return NT_E_CACHE;
    if(pos && pos!=cache->next)return NT_E_STATE;
    int E=m->E,Q=m->H*m->HD,K=m->KV*m->HD,KD=m->KH*m->KD,VD=m->VH*m->VD,C=2*KD+VD;
    int qw=C>2*Q?C:2*Q,ow=VD>Q?VD:Q,zw=VD>K?VD:K;
    float *x=cells(n,E),*norm=cells(n,E),*q=cells(n,qw),*z=cells(n,zw),*k=cells(n,K);
    float *aa=cells(n,m->VH),*bb=cells(n,m->VH),*out=cells(n,ow),*tmp=cells(n,E);
    float *gate=cells(n,m->F),*up=cells(n,m->F),*scores=cells(m->H,kv->max_seq);
    if(!x || !norm || !q || !z || !k || !aa || !bb || !out || !tmp || !gate || !up || !scores){rc=NT_E_MEMORY;goto done;}
    if(!pos) {
        memset(cache->conv,0,(size_t)m->linear_count*C*m->CK*sizeof(float));
        memset(cache->state,0,(size_t)m->linear_count*m->VH*m->KD*m->VD*sizeof(float));
    }
    cache->next=-1;
    for(int t=0;t<n;t++)if(gguf_dequant_row(m->gf,m->emb,tokens[t],x+(size_t)t*E)){rc=NT_E_STATE;goto done;}
    for(int l=0;l<m->L;l++) {
        q35_layer *a=&m->layers[l];
        for(int t=0;t<n;t++)rmsnorm(norm+(size_t)t*E,x+(size_t)t*E,a->in_norm,E,m->eps);
        mm(q,&a->q,norm,n);
        if(a->full) {
            mm(k,&a->k,norm,n);mm(z,&a->v,norm,n);
            for(int t=0;t<n;t++) {
                for(int h=0;h<m->H;h++) {
                    float *qh=q+(size_t)t*2*Q+h*2*m->HD;
                    rmsnorm(qh,qh,a->q_norm,m->HD,m->eps);rope(qh,pos+t,m->R,m->rope_base,1);
                }
                for(int h=0;h<m->KV;h++) {
                    float *kh=k+(size_t)t*K+h*m->HD;
                    rmsnorm(kh,kh,a->k_norm,m->HD,m->eps);rope(kh,pos+t,m->R,m->rope_base,1);
                }
                size_t off=((size_t)a->index*kv->max_seq+pos+t)*K;
                memcpy(kv->k+off,k+(size_t)t*K,(size_t)K*sizeof(float));
                memcpy(kv->v+off,z+(size_t)t*K,(size_t)K*sizeof(float));
            }
            attention_context ac={m,a,cache,q,out,scores,n,pos};nt_par_for(attention_heads,&ac,m->H,2);
        } else {
            mm(z,&a->z,norm,n);mm(aa,&a->a,norm,n);mm(bb,&a->b,norm,n);
            float *hist=cache->conv+(size_t)a->index*C*m->CK;
            for(int t=0;t<n;t++)for(int c=0;c<C;c++) {
                float *h=hist+(size_t)c*m->CK;
                memmove(h,h+1,(size_t)(m->CK-1)*sizeof(float));h[m->CK-1]=q[(size_t)t*C+c];
                q[(size_t)t*C+c]=silu(dot_f32(h,a->conv+(size_t)c*m->CK,m->CK));
            }
            delta_context dc={m,a,cache,q,z,aa,bb,out,n};nt_par_for(delta_heads,&dc,m->VH,2);
        }
        mm(tmp,&a->o,out,n);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=tmp[i];
        if(post_attention && (rc=post_attention(user,l,pos,n,E,x)))goto done;
        for(int t=0;t<n;t++)rmsnorm(norm+(size_t)t*E,x+(size_t)t*E,a->ffn_norm,E,m->eps);
        mm(gate,&a->gate,norm,n);mm(up,&a->up,norm,n);
        for(size_t i=0;i<(size_t)n*m->F;i++)gate[i]=silu(gate[i])*up[i];
        mm(tmp,&a->down,gate,n);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=tmp[i];
        if(post_block && (rc=post_block(user,l,pos,n,E,x)))goto done;
    }
    if(logits){rmsnorm(norm,x+(size_t)(n-1)*E,m->norm,E,m->eps);mm(logits,&m->head,norm,1);}
    cache->next=pos+n;
done:
    free(x);free(norm);free(q);free(z);free(k);free(aa);free(bb);free(out);free(tmp);free(gate);free(up);free(scores);
    return rc;
}
static int q35_forward_residual(void *m,kv_cache *kv,const int *tokens,int n,int pos,
                                float *logits,nt_residual_fn callback,void *user) {
    return q35_forward_hooks(m,kv,tokens,n,pos,logits,NULL,callback,user);
}
static const gguf_file *q35_describe(const void *model,nt_dims *dims) {
    const q35_model *m=model;
    *dims=(nt_dims){m->L,0,m->V};
    /* The generic header scan may disagree with the family keys on a crafted
     * file; DOE sizes norm reads by the header, so a mismatch is refused. */
    return m->E==m->gf->embed_dim && m->L==m->gf->n_layers ? m->gf : NULL;
}
static const float *q35_doe_norm(const void *model,int index) {
    const q35_model *m=model;
    if(index<0 || index>2*m->L)return NULL;
    if(index==2*m->L)return m->norm;
    return index%2 ? m->layers[index/2].ffn_norm : m->layers[index/2].in_norm;
}
static const nt_doe_decoder q35_doe={q35_forward_hooks,q35_describe,q35_doe_norm,2,0};
static int q35_forward(void *m,kv_cache *kv,const int *tokens,int n,int pos,float *logits) {
    return q35_forward_residual(m,kv,tokens,n,pos,logits,NULL,NULL);
}
static const char *const names[]={"qwen35",NULL};
const nt_arch nt_arch_qwen35={.names=names,.load=q35_load,.free=q35_free,
    .forward=q35_forward,.forward_residual=q35_forward_residual,
    .kv_new=q35_kv_new,.kv_free=q35_kv_free,.doe=&q35_doe};
