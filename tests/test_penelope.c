/* PEN7 contract gate: small independent double-equation goldens, complete
 * learned-position/RoPE geometry, cache/callback ownership and explicit refusal. */
#include "harness/archs.h"
#include "examples/bpe.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
enum { E=8,V=7,L=2,F=16,T=40,NT=4+11*L };
static int checks,failed;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
typedef struct {char name[80];int rows,cols;float data[T*E];} tensor;
static void put(tensor*t,const char*name,int rows,int cols,int salt){
    snprintf(t->name,sizeof(t->name),"%s",name);t->rows=rows;t->cols=cols;
    for(int i=0;i<(rows?rows:1)*cols;i++)
        t->data[i]=rows?((i*7+salt*3)%23-11)*.0375f:.8f+.03f*((i+salt)%7);
}
static int fixture(const char*path,int mode){
    tensor ts[NT+1];int count=0;
    put(&ts[count++],"token_embd.weight",V,E,1);put(&ts[count++],"position_embd.weight",T,E,2);
    put(&ts[count++],"output_norm.weight",0,E,3);put(&ts[count++],"output.weight",V,E,4);
    const char*names[]={"attn_norm","attn_q","attn_k","attn_v","attn_output","resonance",
        "resonance_gate","ffn_norm","ffn_gate","ffn_up","ffn_down"};
    int rows[]={0,E,E,E,E,E,0,0,F,F,E},cols[]={E,E,E,E,E,E,2,E,E,E,F};
    for(int l=0;l<L;l++)for(int k=0;k<11;k++){
        char name[80];snprintf(name,sizeof(name),"blk.%d.%s.weight",l,names[k]);
        put(&ts[count++],name,rows[k],cols[k],5+l*11+k);
    }
    if(mode==3){ts[12].rows=E;ts[12].cols=F;}
    if(mode==5)ts[0].data[0]=NAN;
    if(mode==6)count--;
    if(mode==7)put(&ts[count++],"unhandled.bias",0,E,1);
    gguf_writer*w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"general.architecture","penelope");
#define U(k,v) gguf_write_kv_u32(w,"penelope."k,v)
#define K(k,v) gguf_write_kv_f32(w,"penelope."k,v)
    U("vocab_size",V);U("block_count",L);U("embedding_length",E);U("context_length",mode==4?T-1:T);
    U("feed_forward_length",F);U("attention.head_count",2);U("attention.head_dim",4);
    if(mode!=9)U("attention.head_count_kv",mode==10?1:2);
    U("rope.dimension_count",4);K("attention.layer_norm_rms_epsilon",mode==2?1e-6f:1e-5f);
    K("rope.freq_base",10000);K("silu_cutoff",-20);
    gguf_write_kv_str(w,"penelope.rope.layout",mode==1?"adjacent":"split-half");
    gguf_write_kv_str(w,"penelope.resonance","dense-gated");
    gguf_write_kv_bool(w,"penelope.position_embedding",1);
    gguf_write_kv_str(w,"tokenizer.ggml.model",mode==8?"gpt2":"penelope");
    gguf_write_kv_str(w,"tokenizer.penelope.algorithm","byte-rank-pass-v1");
#undef U
#undef K
    for(int i=0;i<count;i++){
        uint64_t shape[]={(uint64_t)ts[i].cols,(uint64_t)ts[i].rows};
        gguf_write_tensor_decl(w,ts[i].name,ts[i].rows?2:1,shape,GGUF_TYPE_F32);
    }
    for(int i=0;i<count;i++)gguf_write_tensor_f32(w,ts[i].name,ts[i].data,
        (uint64_t)(ts[i].rows?ts[i].rows:1)*(uint64_t)ts[i].cols);
    return gguf_write_close(w);
}
/* Evaluated independently with Python math (double arithmetic); the fixture's
 * float32 weights are the shared inputs. No NoTorch kernel produced these. */
static const float golden[3][V]={
 { .0998828216f,-.837708552f,.314154978f,.0722493627f,.441433671f,-.620180654f,.926130145f },
 { -.334164547f,.596025406f,.499898778f,.794052585f,-.890895293f,.697693886f,.0473859654f },
 { .204421005f,-.712378934f,.37633021f,.0704035637f,.283622313f,-.138293524f,.784634294f }
};
static int close_values(const float*a,const float*b,int n){
    for(int i=0;i<n;i++)if(!isfinite(a[i])||fabsf(a[i]-b[i])>3e-6f)return 0;
    return 1;
}
typedef struct{int calls,mode;} observer;
static int observe(void*user,int l,int pos,int n,int width,float*x){
    observer*p=user;p->calls++;
    if(width!=E||l<0||l>=L||pos<0||n<1)return 78;
    if(p->mode==2)return 77;
    if(p->mode==1&&l==0)for(int i=0;i<n;i++)x[i*E+1]+=.5f;
    return 0;
}
typedef struct{void*m;int ids[33],rc;float out[V];} task;
static void*run(void*arg){
    task*t=arg;kv_cache*kv=kv_new(L,T,E);if(!kv){t->rc=-1;return NULL;}
    t->rc=nt_arch_penelope.forward(t->m,kv,t->ids,33,0,t->out);kv_free(kv);return NULL;
}
static int u8(int cp,char*out){if(cp<128){out[0]=(char)cp;out[1]=0;return 1;}
    out[0]=(char)(0xc0|(cp>>6));out[1]=(char)(0x80|(cp&63));out[2]=0;return 2;}
static int tokenizer_fixture(const char*path,int mode){
    char surfaces[260][24];const char*tokens[260],*merges[4];char joins[4][50];int extra=0;
    int32_t pairs[]={97,97,256,97,32,257,258,32};
    for(int b=0;b<256;b++){int printable=(b>=33&&b<=126)||(b>=161&&b<=172)||b>=174;
        u8(printable?b:256+extra++,surfaces[b]);tokens[b]=surfaces[b];}
    for(int i=0;i<4;i++){
        int left=pairs[2*i],right=pairs[2*i+1];
        snprintf(surfaces[256+i],24,"%s%s",surfaces[left],surfaces[right]);tokens[256+i]=surfaces[256+i];
        snprintf(joins[i],50,"%s %s",surfaces[left],surfaces[right]);merges[i]=joins[i];
    }
    if(mode==1)pairs[2]=257;
    if(mode==2)surfaces[259][0]='x';
    gguf_writer*w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"general.architecture","penelope");
    gguf_write_kv_u32(w,"penelope.vocab_size",mode==3?261:260);
    gguf_write_kv_str(w,"tokenizer.ggml.model","penelope");
    gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",mode==4);
    gguf_write_kv_i32(w,"tokenizer.ggml.bos_token_id",1);
    gguf_write_kv_bool(w,"tokenizer.ggml.add_eos_token",mode==5);
    gguf_write_kv_str(w,"tokenizer.penelope.algorithm",mode==6?"gpt2":"byte-rank-pass-v1");
    gguf_write_kv_bool(w,"tokenizer.penelope.lowercase_ascii",mode!=7);
    gguf_write_kv_u32(w,"tokenizer.penelope.max_input_bytes",mode==8?8192:8191);
    gguf_write_kv_i32_array(w,"tokenizer.penelope.merge_pairs",pairs,mode==9?6:8);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,260);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,4);
    return gguf_write_close(w);
}
int main(void){
    char path[]="/tmp/nt_penelope_XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    CHECK(fixture(path,0)==0,"write PEN7 fixture");gguf_file*gf=gguf_open(path);nt_dims dims;
    const nt_arch*a=nt_pick_arch("penelope");CHECK(a==&nt_arch_penelope,"registered family");
    void*m=gf?a->load(gf,&dims):NULL;CHECK(m!=NULL,"load finite declared model");if(!m)return 1;
    CHECK(dims.n_layers==L&&dims.kv_dim==E&&dims.vocab==V,"loaded geometry");
    kv_cache*kv=kv_new(L,T,E);if(!kv)return 1;
    int ids[33];for(int i=0;i<33;i++)ids[i]=(i*3+1)%V;
    float base[V],out[V];int sizes[]={1,9,33};
    for(int c=0;c<3;c++){
        CHECK(a->forward(m,kv,ids,sizes[c],0,out)==NT_OK,"full forward");
        CHECK(close_values(out,golden[c],V),"independent double golden logits");
    }
    memcpy(base,out,sizeof(base));
    for(int i=0;i<33;i++)CHECK(a->forward(m,kv,ids+i,1,i,out)==NT_OK,"single-step decode");
    CHECK(!memcmp(base,out,sizeof(base)),"decode and prefill identical");
    observer obs={0};CHECK(a->forward_residual(m,kv,ids,33,0,out,observe,&obs)==0&&obs.calls==L&&
        !memcmp(out,base,sizeof(base)),"observation leaves logits identical");
    obs=(observer){.mode=1};CHECK(a->forward_residual(m,kv,ids,33,0,out,observe,&obs)==0&&
        memcmp(out,base,sizeof(base)),"residual intervention changes downstream logits");
    for(int i=0;i<V;i++)out[i]=-123;
    obs=(observer){.mode=2};CHECK(a->forward_residual(m,kv,ids,33,0,out,observe,&obs)==77&&obs.calls==1,"callback refusal propagates");
    for(int i=0;i<V;i++)CHECK(out[i]==-123,"failed callback keeps output unchanged");
    CHECK(a->forward(m,kv,ids,33,0,out)==0&&!memcmp(out,base,sizeof(base)),"restart has no callback residue");
    obs=(observer){0};CHECK(a->forward_residual(m,kv,ids,33,0,NULL,observe,&obs)==0&&obs.calls==L,"KV-only observation");
    int bad=V;CHECK(a->forward(m,kv,&bad,1,0,out)==NT_E_TOKEN,"bad token refused");
    CHECK(a->forward(m,kv,ids,0,0,out)==NT_E_ARG,"empty call refused");
    CHECK(a->forward(m,kv,ids,33,8,out)==NT_E_CAPACITY,"context bound");
    kv_cache*large=kv_new(L,T+10,E);CHECK(a->forward(m,large,ids,1,T,out)==NT_E_CAPACITY,"learned position bound exceeds cache capacity");kv_free(large);
    int saved=kv->kv_dim;kv->kv_dim--;CHECK(a->forward(m,kv,ids,1,0,out)==NT_E_CACHE,"mismatched cache refused");kv->kv_dim=saved;
    task jobs[2]={{.m=m},{.m=m}};pthread_t workers[2];
    for(int i=0;i<2;i++){memcpy(jobs[i].ids,ids,sizeof(ids));CHECK(!pthread_create(&workers[i],NULL,run,&jobs[i]),"launch owned cache");}
    for(int i=0;i<2;i++){pthread_join(workers[i],NULL);CHECK(!jobs[i].rc&&!memcmp(base,jobs[i].out,sizeof(base)),"shared immutable model with independent concurrent caches");}
    uint32_t dtype=gf->tensors[0].dtype;gf->tensors[0].dtype=GGUF_TYPE_Q4_0;
    void*badmodel=a->load(gf,&dims);CHECK(!badmodel,"unverified quantized type refused");if(badmodel)a->free(badmodel);gf->tensors[0].dtype=dtype;
    a->free(m);gguf_close(gf);kv_free(kv);
    for(int mode=1;mode<=10;mode++){
        CHECK(fixture(path,mode)==0,"write malformed model");gf=gguf_open(path);
        dims=(nt_dims){7,8,9};m=gf?a->load(gf,&dims):NULL;
        CHECK(!m,"unsupported metadata, shape, missing tensor or NaN refused");if(m)a->free(m);gguf_close(gf);
        CHECK(dims.n_layers==7&&dims.kv_dim==8&&dims.vocab==9,"failed load preserves caller dimensions");
    }
    CHECK(tokenizer_fixture(path,0)==0,"write tokenizer");bpe_tokenizer*t=bpe_load(path);CHECK(t!=NULL,"load native byte tokenizer");
    if(t){int tok[16];CHECK(bpe_encode(t," AAA ",tok,16)==1&&tok[0]==259,"whole-input merges cross spaces and lowercase ASCII");
        CHECK(bpe_encode(t,"AAAA",tok,16)==2&&tok[0]==256&&tok[1]==256,"one ranked pass is left to right");
        char bytes[2];CHECK(bpe_decode_token(t,0,bytes,2)==1&&bytes[0]==0,"NUL byte survives decode length");
        CHECK(bpe_encode(t,"\xff",tok,16)==1&&tok[0]==255,"high byte survives input");
        CHECK(bpe_decode_token(t,0,bytes,0)==0,"zero decode capacity");bpe_free(t);}
    for(int mode=1;mode<=9;mode++){CHECK(tokenizer_fixture(path,mode)==0,"write malformed tokenizer");t=bpe_load(path);
        CHECK(!t,"tokenizer metadata/pairs/surface mismatch refused");bpe_free(t);}
    unlink(path);printf("PENELOPE_%s (%d checks)\n",failed?"FAIL":"OK",checks);return !!failed;
}
