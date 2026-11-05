/* Analytic two-channel encoder: zero Q/K gives uniform bidirectional attention.
 * Identity V/O/projection and zero FFN/PLE leave a closed-form output. This
 * catches causal masks, window boundaries, padding in attention/pooling, layer
 * scalar mistakes, hook placement and failure/output atomicity independently
 * of a downloaded checkpoint or Python numerical runtime. */
#include "harness/arch.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {const char *name;int rows,cols,rank;const float *data;} tensor;
static const float ones[]={1,1},zero[]={0,0,0,0},identity[]={1,0,0,1};
static const float embeddings[]={1,0,0,2,1,1,-1,1};
static const float scalar[]={0.75f};
static void write_model(const char *path,int window) {
    tensor tensors[]={
        {"embed_tokens.weight",4,2,2,embeddings},
        {"embedding_projection.weight",2,2,2,identity},
        {"norm.weight",1,2,1,ones},
        {"ple.per_layer_model_projection.weight",2,2,2,zero},
        {"ple.per_layer_projection_norm.weight",1,2,1,ones},
        {"layers.0.input_layernorm.weight",1,2,1,ones},
        {"layers.0.post_attention_layernorm.weight",1,2,1,ones},
        {"layers.0.pre_feedforward_layernorm.weight",1,2,1,ones},
        {"layers.0.post_feedforward_layernorm.weight",1,2,1,ones},
        {"layers.0.self_attn.q_norm.weight",1,2,1,ones},
        {"layers.0.self_attn.k_norm.weight",1,2,1,ones},
        {"layers.0.self_attn.q_proj.weight",2,2,2,zero},
        {"layers.0.self_attn.k_proj.weight",2,2,2,zero},
        {"layers.0.self_attn.v_proj.weight",2,2,2,identity},
        {"layers.0.self_attn.o_proj.weight",2,2,2,identity},
        {"layers.0.mlp.gate_proj.weight",2,2,2,zero},
        {"layers.0.mlp.up_proj.weight",2,2,2,zero},
        {"layers.0.mlp.down_proj.weight",2,2,2,zero},
        {"layers.0.ple_block.per_layer_input_gate.weight",2,2,2,identity},
        {"layers.0.ple_block.per_layer_projection.weight",2,2,2,identity},
        {"layers.0.ple_block.post_per_layer_input_norm.weight",1,2,1,ones},
        {"layers.0.layer_scalar",1,1,1,scalar},
    };
    gguf_writer *w=gguf_write_open(path);assert(w);
    assert(!gguf_write_kv_str(w,"general.architecture","embeddinggemma2"));
    const char *keys[]={"block_count","embedding_length","output_length","vocab_size","context_length","attention.head_count","feed_forward_length","ple_length","layer.0.head_dim","layer.0.kv_heads","layer.0.window"};
    int vals[]={1,2,2,4,8,1,2,2,2,1,window};
    char key[128];
    for(int i=0;i<11;i++){snprintf(key,sizeof(key),"embeddinggemma2.%s",keys[i]);assert(!gguf_write_kv_u32(w,key,vals[i]));}
    assert(!gguf_write_kv_f32(w,"embeddinggemma2.rms_epsilon",1e-6f));
    assert(!gguf_write_kv_f32(w,"embeddinggemma2.embed_scale",1));
    assert(!gguf_write_kv_f32(w,"embeddinggemma2.ple_scale",1));
    assert(!gguf_write_kv_f32(w,"embeddinggemma2.layer.0.rope_base",10000));
    for(size_t i=0;i<sizeof(tensors)/sizeof(*tensors);i++) {
        tensor *t=&tensors[i];uint64_t dims[]={t->cols,t->rows};
        assert(!gguf_write_tensor_decl(w,t->name,t->rank,dims,GGUF_TYPE_F32));
    }
    for(size_t i=0;i<sizeof(tensors)/sizeof(*tensors);i++) {
        tensor *t=&tensors[i];assert(!gguf_write_tensor_f32(w,t->name,t->data,(uint64_t)t->rows*t->cols));
    }
    assert(!gguf_write_close(w));
}
static void rms(double *x) {double d=sqrt((x[0]*x[0]+x[1]*x[1])*0.5+1e-6);x[0]/=d;x[1]/=d;}
static void analytic(const int *ids,const unsigned char *mask,int n,int window,double *pool,double *out,double *blocks) {
    pool[0]=pool[1]=0;int count=0;
    for(int t=0;t<n;t++) {
        double avg[]={0,0};int keys=0;
        for(int j=0;j<n;j++)if((!mask||mask[j]) && (!window||abs(t-j)<=window)) {
            double v[]={embeddings[ids[j]*2],embeddings[ids[j]*2+1]};rms(v);rms(v);
            avg[0]+=v[0];avg[1]+=v[1];keys++;
        }
        if(keys){avg[0]/=keys;avg[1]/=keys;}rms(avg);
        double x[]={(embeddings[ids[t]*2]+avg[0])*scalar[0],(embeddings[ids[t]*2+1]+avg[1])*scalar[0]};
        blocks[t*2]=x[0];blocks[t*2+1]=x[1];rms(x);
        if(!mask||mask[t]){pool[0]+=x[0];pool[1]+=x[1];count++;}
    }
    pool[0]/=count;pool[1]/=count;double len=hypot(pool[0],pool[1]);out[0]=pool[0]/len;out[1]=pool[1]/len;
}
typedef struct {float block[16];int calls,action;} capture;
static int cb(void *user,nt_encode_site site,int layer,int n,int width,float *x) {
    capture *c=user;c->calls++;assert(width==2);
    if(site==NT_ENCODE_BLOCK){assert(layer==0);memcpy(c->block,x,(size_t)n*2*sizeof(float));
        if(c->action==1)for(int t=0;t<n;t++)x[t*2]+=0.5f;
        if(c->action==2)return -77;
        if(c->action==3)x[0]=NAN;
    }
    return 0;
}
static void close_to(double a,double b){if(fabs(a-b)>3e-6){fprintf(stderr,"%.9g != %.9g\n",a,b);abort();}}
int main(void) {
    char path[]="/tmp/notorch-eg2-XXXXXX";int fd=mkstemp(path);assert(fd>=0);close(fd);
    for(int window=0;window<=1;window++) {
        write_model(path,window);gguf_file *gf=gguf_open(path);assert(gf);nt_encode_dims d;
        void *m=nt_encoder_embeddinggemma2.load(gf,&d);assert(m);assert(d.width==2&&d.output_dim==2&&d.n_layers==1);
        const int ids[]={0,1,2};unsigned char mask[]={1,1,0};
        for(int padded=0;padded<2;padded++) {
            const unsigned char *mk=padded?mask:NULL;float p[2],e[2],again[2];capture c={0};double ep[2],ee[2],blocks[6];
            analytic(ids,mk,3,window,ep,ee,blocks);
            assert(!nt_encoder_embeddinggemma2.encode(m,ids,mk,3,p,e,cb,&c));assert(c.calls==7);
            for(int j=0;j<2;j++){close_to(p[j],ep[j]);close_to(e[j],ee[j]);}
            for(int j=0;j<6;j++)close_to(c.block[j],blocks[j]);
            assert(!nt_encoder_embeddinggemma2.encode(m,ids,mk,3,NULL,again,NULL,NULL));assert(!memcmp(e,again,sizeof(e)));
            c.action=1;assert(!nt_encoder_embeddinggemma2.encode(m,ids,mk,3,NULL,again,cb,&c));assert(fabsf(again[0]-e[0])>0.01f);
            for(int action=2;action<=3;action++) {
                c.action=action;p[0]=p[1]=e[0]=e[1]=42;
                assert(nt_encoder_embeddinggemma2.encode(m,ids,mk,3,p,e,cb,&c)==(action==2?-77:NT_E_STATE));
                assert(p[0]==42&&p[1]==42&&e[0]==42&&e[1]==42);
            }
        }
        float out[]={42,42};int bad_ids[]={0,4};unsigned char badmask[]={0,0,0};
        assert(nt_encoder_embeddinggemma2.encode(m,bad_ids,NULL,2,NULL,out,NULL,NULL)==NT_E_TOKEN);
        assert(nt_encoder_embeddinggemma2.encode(m,ids,badmask,3,NULL,out,NULL,NULL)==NT_E_ARG);
        badmask[0]=2;assert(nt_encoder_embeddinggemma2.encode(m,ids,badmask,3,NULL,out,NULL,NULL)==NT_E_ARG);
        assert(nt_encoder_embeddinggemma2.encode(m,ids,NULL,9,NULL,out,NULL,NULL)==NT_E_CAPACITY);
        assert(out[0]==42&&out[1]==42);
        nt_encoder_embeddinggemma2.free(m);gguf_close(gf);
    }
    unlink(path);puts("EmbeddingGemma2: analytic attention, masks, boundaries, hooks and failure atomicity PASS");return 0;
}
