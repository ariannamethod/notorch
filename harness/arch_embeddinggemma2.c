/* EmbeddingGemma 2 text encoder, native F32 CPU arithmetic.
 * Architecture reference: Google embeddinggemma-2 and Hugging Face Transformers
 * modeling_embedding_gemma2.py (Apache-2.0). The GGUF conversion preserves the
 * original BF16/F32 tensor bytes; no inferred causal mask, quantization or KV
 * cache is used. A full sequence is one call, including both sides of the
 * symmetric sliding window. */
#include "harness/arch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    wt q, k, v, o, gate, up, down, ple_gate, ple_out;
    float *in_norm, *q_norm, *k_norm, *post_attn, *ffn_norm, *post_ffn, *ple_norm;
    int hd, kv_heads, window;
    float rope_base, scalar;
} eg2_layer;

typedef struct {
    gguf_file *gf; /* borrowed; caller keeps it open */
    nt_encode_dims dims;
    int heads, ffn, ple_width, emb_ti;
    float eps, embed_scale, ple_scale;
    wt ple, projection;
    float *ple_norm, *out_norm;
    eg2_layer *layers;
} eg2_model;

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
        fprintf(stderr,"embeddinggemma2: missing or invalid matrix %s [%d,%d]\n",name,rows,cols);
        return 0;
    }
    /* Exact dequantized F32 copy for arithmetic. The much larger input table
     * stays mapped and only requested rows are expanded. */
    memset(w,0,sizeof(*w)); w->rows=rows; w->cols=cols;
    w->f32=gguf_dequant(gf,gguf_find_tensor(gf,name));
    return w->f32!=NULL;
}
static float *vector(gguf_file *gf,const char *name,int n) {
    return shape(gf,name,1,n,0)?gguf_dequant(gf,gguf_find_tensor(gf,name)):NULL;
}
static void eg2_free(void *ptr) {
    eg2_model *m=ptr; if(!m) return;
    free(m->ple.f32);free(m->projection.f32);free(m->ple_norm);free(m->out_norm);
    for(int l=0;m->layers && l<m->dims.n_layers;l++) {
        eg2_layer *a=&m->layers[l];
        wt *ws[]={&a->q,&a->k,&a->v,&a->o,&a->gate,&a->up,&a->down,&a->ple_gate,&a->ple_out};
        for(int i=0;i<9;i++) free(ws[i]->f32);
        free(a->in_norm);free(a->q_norm);free(a->k_norm);free(a->post_attn);
        free(a->ffn_norm);free(a->post_ffn);free(a->ple_norm);
    }
    free(m->layers);free(m);
}
static void *eg2_load(gguf_file *gf,nt_encode_dims *dims) {
    if(!gf || !dims || strcmp(gf->arch,"embeddinggemma2")) return NULL;
    eg2_model *m=calloc(1,sizeof(*m)); if(!m) return NULL; m->gf=gf;
    #define MI(key,dst) if(!meta_int(gf,"embeddinggemma2." key,&(dst))) goto bad
    #define MF(key,dst) if(!meta_float(gf,"embeddinggemma2." key,&(dst))) goto bad
    MI("block_count",m->dims.n_layers); MI("embedding_length",m->dims.width);
    MI("output_length",m->dims.output_dim); MI("vocab_size",m->dims.vocab);
    MI("context_length",m->dims.max_tokens); MI("attention.head_count",m->heads);
    MI("feed_forward_length",m->ffn); MI("ple_length",m->ple_width);
    MF("rms_epsilon",m->eps); MF("embed_scale",m->embed_scale); MF("ple_scale",m->ple_scale);
    #undef MI
    #undef MF
    int E=m->dims.width,L=m->dims.n_layers,P=m->ple_width,O=m->dims.output_dim;
    if(E<1 || E>8192 || L<1 || L>128 || P<1 || P>8192 || O<1 || O>8192 ||
       m->heads<1 || m->heads>128 || m->ffn<1 || m->ffn>65536 || m->eps<=0 ||
       m->dims.vocab<1 || m->dims.max_tokens<1 || m->dims.max_tokens>8192) goto bad;
    if(!shape(gf,"embed_tokens.weight",m->dims.vocab,E,1)) goto bad;
    m->emb_ti=gguf_find_tensor(gf,"embed_tokens.weight");
    if(!matrix(gf,&m->ple,"ple.per_layer_model_projection.weight",L*P,E) ||
       !matrix(gf,&m->projection,"embedding_projection.weight",O,E)) goto bad;
    m->ple_norm=vector(gf,"ple.per_layer_projection_norm.weight",P);
    m->out_norm=vector(gf,"norm.weight",E);
    if(!m->ple_norm || !m->out_norm) goto bad;
    m->layers=calloc((size_t)L,sizeof(*m->layers));if(!m->layers) goto bad;
    for(int l=0;l<L;l++) {
        eg2_layer *a=&m->layers[l];char key[128],name[128];
        #define LI(s,dst) do{snprintf(key,sizeof(key),"embeddinggemma2.layer.%d." s,l);if(!meta_int(gf,key,&(dst)))goto bad;}while(0)
        LI("head_dim",a->hd); LI("kv_heads",a->kv_heads); LI("window",a->window);
        #undef LI
        snprintf(key,sizeof(key),"embeddinggemma2.layer.%d.rope_base",l);
        if(!meta_float(gf,key,&a->rope_base) || a->rope_base<=0 || a->hd<2 || a->hd>1024 || a->hd%2 ||
           a->kv_heads<1 || a->kv_heads>m->heads || m->heads%a->kv_heads || a->window<0)goto bad;
        #define V(dst,s,n) do{snprintf(name,sizeof(name),"layers.%d." s,l);a->dst=vector(gf,name,n);if(!a->dst)goto bad;}while(0)
        V(in_norm,"input_layernorm.weight",E); V(q_norm,"self_attn.q_norm.weight",a->hd);
        V(k_norm,"self_attn.k_norm.weight",a->hd); V(post_attn,"post_attention_layernorm.weight",E);
        V(ffn_norm,"pre_feedforward_layernorm.weight",E); V(post_ffn,"post_feedforward_layernorm.weight",E);
        V(ple_norm,"ple_block.post_per_layer_input_norm.weight",E);
        #undef V
        #define W(dst,s,r,c) do{snprintf(name,sizeof(name),"layers.%d." s,l);if(!matrix(gf,&a->dst,name,r,c))goto bad;}while(0)
        int Q=m->heads*a->hd,K=a->kv_heads*a->hd;
        W(q,"self_attn.q_proj.weight",Q,E); W(k,"self_attn.k_proj.weight",K,E);
        W(v,"self_attn.v_proj.weight",K,E); W(o,"self_attn.o_proj.weight",E,Q);
        W(gate,"mlp.gate_proj.weight",m->ffn,E); W(up,"mlp.up_proj.weight",m->ffn,E);
        W(down,"mlp.down_proj.weight",E,m->ffn);W(ple_gate,"ple_block.per_layer_input_gate.weight",P,E);
        W(ple_out,"ple_block.per_layer_projection.weight",E,P);
        #undef W
        snprintf(name,sizeof(name),"layers.%d.layer_scalar",l);
        float *sc=vector(gf,name,1);if(!sc)goto bad;a->scalar=*sc;free(sc);
        if(!isfinite(a->scalar))goto bad;
    }
    *dims=m->dims;return m;
bad:
    fprintf(stderr,"embeddinggemma2: invalid or unsupported model contract\n");eg2_free(m);return NULL;
}

/* The F32 GEMM uses notorch's own row kernels (or the selected BLAS backend).
 * No int8 activation quantization enters this encoder path. */
static void mm(float *y,const wt *w,const float *x,int n) {
    nt_blas_mmT(y,x,w->f32,n,w->cols,w->rows);
}
static void norm(float *y,const float *x,const float *w,int n,float eps) {
    float sum=0;for(int i=0;i<n;i++)sum+=x[i]*x[i];
    float scale=powf(sum/(float)n+eps,-0.5f);
    for(int i=0;i<n;i++)y[i]=x[i]*scale*(w?w[i]:1.0f);
}
static void norms(float *y,const float *x,const float *w,int rows,int width,float eps) {
    for(int i=0;i<rows;i++)norm(y+(size_t)i*width,x+(size_t)i*width,w,width,eps);
}
static float gelu(float x) {
    return 0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));
}
static int finite_rows(const float *x,size_t n) {for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return 0;return 1;}
static int observe(nt_encode_fn cb,void *user,nt_encode_site site,int l,int n,int width,float *x) {
    if(cb){int rc=cb(user,site,l,n,width,x);if(rc)return rc;}
    return finite_rows(x,(size_t)n*width)?NT_OK:NT_E_STATE;
}
static void rotate(float *x,int pos,int hd,float base) {
    for(int i=0;i<hd/2;i++) {
        float inv=1.0f/powf(base,(float)(2*i)/(float)hd);
        float f=(float)pos*inv,c=cosf(f),s=sinf(f),a=x[i],b=x[i+hd/2];
        x[i]=a*c-b*s;x[i+hd/2]=b*c+a*s;
    }
}

static int eg2_encode(void *ptr,const int *ids,const unsigned char *mask,int n,
                      float *pooled,float *output,nt_encode_fn cb,void *user) {
    eg2_model *m=ptr;
    if(!m || !ids || !output || n<1)return NT_E_ARG;
    if(n>m->dims.max_tokens)return NT_E_CAPACITY;
    int count=0;
    for(int t=0;t<n;t++) {
        if(ids[t]<0 || ids[t]>=m->dims.vocab)return NT_E_TOKEN;
        if(mask && mask[t]>1)return NT_E_ARG;
        count+=!mask || mask[t];
    }
    if(!count)return NT_E_ARG;
    int E=m->dims.width,L=m->dims.n_layers,P=m->ple_width,O=m->dims.output_dim;
    int maxq=0,maxk=0;for(int l=0;l<L;l++) {
        int q=m->heads*m->layers[l].hd,k=m->layers[l].kv_heads*m->layers[l].hd;
        if(q>maxq)maxq=q;
        if(k>maxk)maxk=k;
    }
    float *x=NULL,*ple=NULL,*a=NULL,*b=NULL,*q=NULL,*k=NULL,*v=NULL,*attn=NULL,*g=NULL,*u=NULL,*pg=NULL,*proj=NULL,*scores=NULL,*pool=NULL,*normalized=NULL;
    int rc=NT_E_MEMORY;
    #define A(p,w) do{p=malloc((size_t)n*(w)*sizeof(float));if(!p)goto done;}while(0)
    A(x,E);A(ple,L*P);A(a,E);A(b,E);A(q,maxq);A(k,maxk);A(v,maxk);A(attn,maxq);
    A(g,m->ffn);A(u,m->ffn);A(pg,P);A(proj,O);A(scores,1);
    pool=calloc((size_t)O,sizeof(float));normalized=malloc((size_t)O*sizeof(float));
    if(!pool || !normalized)goto done;
    #undef A
    #define OBS(site,layer,width,data) do{rc=observe(cb,user,site,layer,n,width,data);if(rc)goto done;}while(0)
    for(int t=0;t<n;t++) {
        if(gguf_dequant_row(m->gf,m->emb_ti,(uint64_t)ids[t],x+(size_t)t*E)) {rc=NT_E_STATE;goto done;}
        for(int j=0;j<E;j++)x[(size_t)t*E+j]*=m->embed_scale;
    }
    OBS(NT_ENCODE_INPUT,-1,E,x);
    mm(ple,&m->ple,x,n);
    for(size_t i=0;i<(size_t)n*L*P;i++)ple[i]*=m->ple_scale;
    norms(ple,ple,m->ple_norm,n*L,P,m->eps);
    OBS(NT_ENCODE_PLE,-1,L*P,ple);
    for(int l=0;l<L;l++) {
        eg2_layer *ly=&m->layers[l];int hd=ly->hd,Q=m->heads*hd,K=ly->kv_heads*hd;
        norms(a,x,ly->in_norm,n,E,m->eps);
        mm(q,&ly->q,a,n);mm(k,&ly->k,a,n);mm(v,&ly->v,a,n);
        norms(q,q,ly->q_norm,n*m->heads,hd,m->eps);
        norms(k,k,ly->k_norm,n*ly->kv_heads,hd,m->eps);
        norms(v,v,NULL,n*ly->kv_heads,hd,m->eps);
        for(int t=0;t<n;t++) {
            for(int h=0;h<m->heads;h++)rotate(q+(size_t)t*Q+h*hd,t,hd,ly->rope_base);
            for(int h=0;h<ly->kv_heads;h++)rotate(k+(size_t)t*K+h*hd,t,hd,ly->rope_base);
        }
        for(int t=0;t<n;t++)for(int h=0;h<m->heads;h++) {
            int kh=h/(m->heads/ly->kv_heads),lo=0,hi=n;
            if(ly->window){lo=t-ly->window;if(lo<0)lo=0;hi=t+ly->window+1;if(hi>n)hi=n;}
            float max=-INFINITY;
            for(int j=lo;j<hi;j++) {
                float sum=0;
                if(!mask || mask[j])for(int z=0;z<hd;z++)sum+=q[(size_t)t*Q+h*hd+z]*k[(size_t)j*K+kh*hd+z];
                else sum=-INFINITY;
                scores[j]=sum;if(sum>max)max=sum;
            }
            float total=0;
            for(int j=lo;j<hi;j++){scores[j]=isfinite(scores[j])?expf(scores[j]-max):0;total+=scores[j];}
            float *row=attn+(size_t)t*Q+h*hd;memset(row,0,(size_t)hd*sizeof(float));
            if(total>0)for(int j=lo;j<hi;j++) {
                float p=scores[j]/total;
                for(int z=0;z<hd;z++)row[z]+=p*v[(size_t)j*K+kh*hd+z];
            }
        }
        mm(b,&ly->o,attn,n);norms(b,b,ly->post_attn,n,E,m->eps);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=b[i];
        OBS(NT_ENCODE_POST_ATTENTION,l,E,x);
        norms(a,x,ly->ffn_norm,n,E,m->eps);
        mm(g,&ly->gate,a,n);mm(u,&ly->up,a,n);
        for(size_t i=0;i<(size_t)n*m->ffn;i++)g[i]=gelu(g[i])*u[i];
        mm(b,&ly->down,g,n);norms(b,b,ly->post_ffn,n,E,m->eps);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]+=b[i];
        OBS(NT_ENCODE_POST_FFN,l,E,x);
        mm(pg,&ly->ple_gate,x,n);
        for(int t=0;t<n;t++)for(int j=0;j<P;j++)pg[(size_t)t*P+j]=gelu(pg[(size_t)t*P+j])*ple[((size_t)t*L+l)*P+j];
        mm(b,&ly->ple_out,pg,n);norms(b,b,ly->ple_norm,n,E,m->eps);
        for(size_t i=0;i<(size_t)n*E;i++)x[i]=(x[i]+b[i])*ly->scalar;
        OBS(NT_ENCODE_BLOCK,l,E,x);
    }
    norms(x,x,m->out_norm,n,E,m->eps);OBS(NT_ENCODE_FINAL_NORM,-1,E,x);
    mm(proj,&m->projection,x,n);OBS(NT_ENCODE_PROJECTED,-1,O,proj);
    for(int t=0;t<n;t++)if(!mask || mask[t])for(int j=0;j<O;j++)pool[j]+=proj[(size_t)t*O+j];
    float squared=0;for(int j=0;j<O;j++){pool[j]/=(float)count;squared+=pool[j]*pool[j];}
    float length=sqrtf(squared);if(length<1e-12f)length=1e-12f;
    for(int j=0;j<O;j++)normalized[j]=pool[j]/length;
    if(!finite_rows(pool,O) || !finite_rows(normalized,O)){rc=NT_E_STATE;goto done;}
    if(pooled)memcpy(pooled,pool,(size_t)O*sizeof(float));
    memcpy(output,normalized,(size_t)O*sizeof(float));rc=NT_OK;
done:
    free(x);free(ple);free(a);free(b);free(q);free(k);free(v);free(attn);free(g);free(u);free(pg);free(proj);free(scores);free(pool);free(normalized);
    return rc;
    #undef OBS
}
const nt_encoder nt_encoder_embeddinggemma2={"embeddinggemma2",eg2_load,eg2_free,eg2_encode};
