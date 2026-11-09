/* Qwen3-Embedding text encoder, native F32 CPU arithmetic.
 * Architecture reference: Qwen/Qwen3-Embedding-0.6B (config.json, 1_Pooling,
 * 2_Normalize) and Hugging Face Transformers modeling_qwen3.py (Apache-2.0).
 * A causal Qwen3 decoder read up to its final RMSNorm: per-head Q/K RMSNorm
 * before NEOX RoPE, grouped KV heads, SiLU gate. The pooled vector is the final
 * normalized state of the last unmasked position; the embedding is its L2
 * normalization. RoPE positions are sequence indices, masked ones included, as
 * the reference forms them when no position IDs are passed. The GGUF keeps the
 * source BF16 bytes; no quantization and no KV cache. */
#include "harness/arch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    wt q, k, v, o, gate, up, down;
    float *in_norm, *q_norm, *k_norm, *post_attn;
} qe_layer;

typedef struct {
    gguf_file *gf; /* borrowed; caller keeps it open */
    nt_encode_dims dims;
    int heads, kv_heads, hd, ffn, emb_ti;
    float eps, rope_base;
    float *out_norm;
    qe_layer *layers;
} qe_model;

static int meta_int(gguf_file *gf, const char *key, int *out) {
    const gguf_kv *v=gguf_get_kv(gf,key);
    if(!v || v->type!=4 || v->val.u32>2147483647u) return 0;
    *out=(int)v->val.u32; return 1;
}
static int meta_float(gguf_file *gf, const char *key, float *out) {
    const gguf_kv *v=gguf_get_kv(gf,key);
    if(!v || v->type!=6 || !isfinite(v->val.f32)) return 0;
    *out=v->val.f32; return 1;
}
static int shape(gguf_file *gf,const char *name,int rows,int cols,int matrix) {
    int i=gguf_find_tensor(gf,name); if(i<0) return 0;
    gguf_tensor_info *t=&gf->tensors[i];
    return (t->dtype==GGUF_TYPE_F32 || t->dtype==GGUF_TYPE_BF16) &&
        t->ndim==(matrix?2u:1u) && t->shape[0]==(uint64_t)cols &&
        (!matrix || t->shape[1]==(uint64_t)rows);
}
static int matrix(gguf_file *gf,wt *w,const char *name,int rows,int cols) {
    if(!shape(gf,name,rows,cols,1)) {
        fprintf(stderr,"qwen3embedding: missing or invalid matrix %s [%d,%d]\n",name,rows,cols);
        return 0;
    }
    memset(w,0,sizeof(*w)); w->rows=rows; w->cols=cols;
    w->f32=gguf_dequant(gf,gguf_find_tensor(gf,name));
    return w->f32!=NULL;
}
static float *vector(gguf_file *gf,const char *name,int n) {
    return shape(gf,name,1,n,0)?gguf_dequant(gf,gguf_find_tensor(gf,name)):NULL;
}
static void qe_free(void *ptr) {
    qe_model *m=ptr; if(!m) return;
    free(m->out_norm);
    for(int l=0;m->layers && l<m->dims.n_layers;l++) {
        qe_layer *a=&m->layers[l];
        wt *ws[]={&a->q,&a->k,&a->v,&a->o,&a->gate,&a->up,&a->down};
        for(int i=0;i<7;i++) free(ws[i]->f32);
        free(a->in_norm);free(a->q_norm);free(a->k_norm);free(a->post_attn);
    }
    free(m->layers);free(m);
}
static void *qe_load(gguf_file *gf,nt_encode_dims *dims) {
    if(!gf || !dims || strcmp(gf->arch,"qwen3embedding")) return NULL;
    qe_model *m=calloc(1,sizeof(*m)); if(!m) return NULL; m->gf=gf;
    #define MI(key,dst) if(!meta_int(gf,"qwen3embedding." key,&(dst))) goto bad
    #define MF(key,dst) if(!meta_float(gf,"qwen3embedding." key,&(dst))) goto bad
    MI("block_count",m->dims.n_layers); MI("embedding_length",m->dims.width);
    MI("vocab_size",m->dims.vocab); MI("context_length",m->dims.max_tokens);
    MI("attention.head_count",m->heads); MI("attention.head_count_kv",m->kv_heads);
    MI("attention.key_length",m->hd); MI("feed_forward_length",m->ffn);
    MF("rms_epsilon",m->eps); MF("rope.freq_base",m->rope_base);
    #undef MI
    #undef MF
    m->dims.output_dim=m->dims.width;
    int E=m->dims.width,L=m->dims.n_layers;
    if(E<1 || E>8192 || L<1 || L>128 || m->heads<1 || m->heads>128 || m->kv_heads<1 ||
       m->heads%m->kv_heads || m->hd<2 || m->hd>1024 || m->hd%2 || m->ffn<1 || m->ffn>65536 ||
       m->eps<=0 || m->rope_base<=0 || m->dims.vocab<1 || m->dims.max_tokens<1 ||
       m->dims.max_tokens>8192) goto bad;
    if(!shape(gf,"embed_tokens.weight",m->dims.vocab,E,1)) goto bad;
    m->emb_ti=gguf_find_tensor(gf,"embed_tokens.weight");
    if(!(m->out_norm=vector(gf,"norm.weight",E))) goto bad;
    m->layers=calloc((size_t)L,sizeof(*m->layers));if(!m->layers) goto bad;
    int Q=m->heads*m->hd,K=m->kv_heads*m->hd;
    for(int l=0;l<L;l++) {
        qe_layer *a=&m->layers[l];char name[128];
        #define V(dst,s,n) do{snprintf(name,sizeof(name),"layers.%d." s,l);a->dst=vector(gf,name,n);if(!a->dst)goto bad;}while(0)
        V(in_norm,"input_layernorm.weight",E); V(post_attn,"post_attention_layernorm.weight",E);
        V(q_norm,"self_attn.q_norm.weight",m->hd); V(k_norm,"self_attn.k_norm.weight",m->hd);
        #undef V
        #define W(dst,s,r,c) do{snprintf(name,sizeof(name),"layers.%d." s,l);if(!matrix(gf,&a->dst,name,r,c))goto bad;}while(0)
        W(q,"self_attn.q_proj.weight",Q,E); W(k,"self_attn.k_proj.weight",K,E);
        W(v,"self_attn.v_proj.weight",K,E); W(o,"self_attn.o_proj.weight",E,Q);
        W(gate,"mlp.gate_proj.weight",m->ffn,E); W(up,"mlp.up_proj.weight",m->ffn,E);
        W(down,"mlp.down_proj.weight",E,m->ffn);
        #undef W
    }
    *dims=m->dims;return m;
bad:
    fprintf(stderr,"qwen3embedding: invalid or unsupported model contract\n");qe_free(m);return NULL;
}

static void mm(float *y,const wt *w,const float *x,int n) {
    nt_blas_mmT(y,x,w->f32,n,w->cols,w->rows);
}
/* x * rsqrt(mean(x^2) + eps) * w, the reference's float32 RMSNorm; the sum of
 * squares is accumulated in double. */
static void norms(float *y,const float *x,const float *w,int rows,int width,float eps) {
    for(int i=0;i<rows;i++) {
        const float *a=x+(size_t)i*width;float *b=y+(size_t)i*width;double sum=0;
        for(int j=0;j<width;j++)sum+=(double)a[j]*a[j];
        float scale=1.0f/sqrtf((float)(sum/width)+eps);
        for(int j=0;j<width;j++)b[j]=a[j]*scale*w[j];
    }
}
static int finite_rows(const float *x,size_t n) {for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return 0;return 1;}
static int observe(nt_encode_fn cb,void *user,nt_encode_site site,int l,int n,int width,float *x) {
    if(cb){int rc=cb(user,site,l,n,width,x);if(rc)return rc;}
    return finite_rows(x,(size_t)n*width)?NT_OK:NT_E_STATE;
}

static int qe_encode(void *ptr,const int *ids,const unsigned char *mask,int n,
                     float *pooled,float *output,nt_encode_fn cb,void *user) {
    qe_model *m=ptr;
    if(!m || !ids || !output || n<1)return NT_E_ARG;
    if(n>m->dims.max_tokens)return NT_E_CAPACITY;
    int last=-1;
    for(int t=0;t<n;t++) {
        if(ids[t]<0 || ids[t]>=m->dims.vocab)return NT_E_TOKEN;
        if(mask && mask[t]>1)return NT_E_ARG;
        if(!mask || mask[t])last=t;
    }
    if(last<0)return NT_E_ARG;
    int E=m->dims.width,L=m->dims.n_layers,H=m->heads,hd=m->hd,half=hd/2,Q=H*hd,K=m->kv_heads*hd,group=H/m->kv_heads;
    float *x=NULL,*a=NULL,*b=NULL,*q=NULL,*k=NULL,*v=NULL,*attn=NULL,*g=NULL,*u=NULL,*cosv=NULL,*sinv=NULL,*normalized=NULL;
    double *scores=NULL;
    int rc=NT_E_MEMORY;
    #define A(p,w) do{p=malloc((size_t)n*(w)*sizeof(*p));if(!p)goto done;}while(0)
    A(x,E);A(a,E);A(b,E);A(q,Q);A(k,K);A(v,K);A(attn,Q);A(g,m->ffn);A(u,m->ffn);
    A(cosv,half);A(sinv,half);A(scores,1);
    #undef A
    if(!(normalized=malloc((size_t)E*sizeof(float))))goto done;
    /* inv_freq = 1/base^(2i/hd) and angle = position*inv_freq, both in float32. */
    for(int i=0;i<half;i++) {
        float inv=1.0f/powf(m->rope_base,(float)(2*i)/(float)hd);
        for(int t=0;t<n;t++){float f=(float)t*inv;cosv[(size_t)t*half+i]=cosf(f);sinv[(size_t)t*half+i]=sinf(f);}
    }
    const double scale=1.0/sqrt((double)hd);
    #define OBS(site,layer,width,data) do{rc=observe(cb,user,site,layer,n,width,data);if(rc)goto done;}while(0)
    for(int t=0;t<n;t++)
        if(gguf_dequant_row(m->gf,m->emb_ti,(uint64_t)ids[t],x+(size_t)t*E)){rc=NT_E_STATE;goto done;}
    OBS(NT_ENCODE_INPUT,-1,E,x);
    for(int l=0;l<L;l++) {
        qe_layer *ly=&m->layers[l];
        norms(a,x,ly->in_norm,n,E,m->eps);
        mm(q,&ly->q,a,n);mm(k,&ly->k,a,n);mm(v,&ly->v,a,n);
        norms(q,q,ly->q_norm,n*H,hd,m->eps);
        norms(k,k,ly->k_norm,n*m->kv_heads,hd,m->eps);
        for(int t=0;t<n;t++) {
            const float *c=cosv+(size_t)t*half,*s=sinv+(size_t)t*half;
            for(int h=0;h<H+m->kv_heads;h++) {
                float *z=h<H?q+(size_t)t*Q+(size_t)h*hd:k+(size_t)t*K+(size_t)(h-H)*hd;
                for(int i=0;i<half;i++){float x1=z[i],x2=z[i+half];z[i]=x1*c[i]-x2*s[i];z[i+half]=x2*c[i]+x1*s[i];}
            }
        }
        for(int t=0;t<n;t++)for(int h=0;h<H;h++) {
            const float *qr=q+(size_t)t*Q+(size_t)h*hd;int kh=h/group;double max=-INFINITY,total=0;
            for(int j=0;j<=t;j++) {
                double sum=-INFINITY;
                if(!mask || mask[j]){const float *kr=k+(size_t)j*K+(size_t)kh*hd;sum=0;for(int z=0;z<hd;z++)sum+=(double)qr[z]*kr[z];sum*=scale;}
                scores[j]=sum;if(sum>max)max=sum;
            }
            for(int j=0;j<=t;j++){scores[j]=isfinite(scores[j])?exp(scores[j]-max):0;total+=scores[j];}
            float *row=attn+(size_t)t*Q+(size_t)h*hd;
            for(int z=0;z<hd;z++) {
                double acc=0;
                if(total>0)for(int j=0;j<=t;j++)if(scores[j]>0)acc+=scores[j]*v[(size_t)j*K+(size_t)kh*hd+z];
                row[z]=total>0?(float)(acc/total):0.0f;
            }
        }
        mm(b,&ly->o,attn,n);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=b[i];
        OBS(NT_ENCODE_POST_ATTENTION,l,E,x);
        norms(a,x,ly->post_attn,n,E,m->eps);
        mm(g,&ly->gate,a,n);mm(u,&ly->up,a,n);
        for(size_t i=0;i<(size_t)n*m->ffn;i++)g[i]=g[i]/(1.0f+expf(-g[i]))*u[i];
        mm(b,&ly->down,g,n);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=b[i];
        OBS(NT_ENCODE_BLOCK,l,E,x);
    }
    norms(x,x,m->out_norm,n,E,m->eps);OBS(NT_ENCODE_FINAL_NORM,-1,E,x);
    const float *pool=x+(size_t)last*E;double squared=0;
    for(int j=0;j<E;j++)squared+=(double)pool[j]*pool[j];
    float length=(float)sqrt(squared);if(length<1e-12f)length=1e-12f;
    for(int j=0;j<E;j++)normalized[j]=pool[j]/length;
    if(!finite_rows(normalized,E)){rc=NT_E_STATE;goto done;}
    if(pooled)memcpy(pooled,pool,(size_t)E*sizeof(float));
    memcpy(output,normalized,(size_t)E*sizeof(float));rc=NT_OK;
done:
    free(x);free(a);free(b);free(q);free(k);free(v);free(attn);free(g);free(u);free(cosv);free(sinv);free(scores);free(normalized);
    return rc;
    #undef OBS
}
const nt_encoder nt_encoder_qwen3embedding={"qwen3embedding",qe_load,qe_free,qe_encode};
