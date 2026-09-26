/* Independent scalar-double golden values from the Gemma3 decoder equations.
 * F32 fixture: E4, H2, HD4 (!=E/H), KV1, FF8, six layers, five tokens, SWA2.
 * Layers 0..4 slide, layer5 is global. Nonuniform norms already include +1.
 * No model download or Python dependency in this gate. */
#include "harness/archs.h"
#include "examples/bpe.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
enum {E=4,V=5,L=6,FF=8,N=5};
static int checks,failed;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
typedef struct{char name[80];int rows,cols;float data[32];}tensor;
static void put(tensor*t,const char*name,int rows,int cols,int salt){
    snprintf(t->name,sizeof(t->name),"%s",name);t->rows=rows;t->cols=cols;
    for(int i=0;i<(rows?rows:1)*cols;i++)t->data[i]=rows?((i*7+salt*3)%23-11)*.0375f:.8f+.03f*((i+salt)%7);
}
/* mode1 malformed postnorm; mode2 illegal residual bias; mode3 none+factor2. */
static int fixture(const char*path,int mode){
    tensor ts[82];int nt=0;
    put(&ts[nt++],"token_embd.weight",V,E,1);put(&ts[nt++],"output_norm.weight",0,E,2);
    const char*fields[]={"attn_norm","attn_q","attn_k","attn_v","attn_output","attn_q_norm",
        "attn_k_norm","post_attention_norm","ffn_norm","ffn_gate","ffn_up","ffn_down","post_ffw_norm"};
    int rows[]={0,8,4,4,4,0,0,0,0,8,8,4,0},cols[]={4,4,4,4,8,4,4,4,4,4,4,8,4};
    for(int l=0;l<L;l++)for(int k=0;k<13;k++){
        char name[80];snprintf(name,sizeof(name),"blk.%d.%s.weight",l,fields[k]);
        put(&ts[nt++],name,rows[k],mode==1&&l==0&&k==12?3:cols[k],3+l*13+k);
    }
    if(mode==2)put(&ts[nt++],"blk.0.ffn_down.bias",0,E,0);
    gguf_writer*w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"general.architecture","gemma3");
#define U(k,v) gguf_write_kv_u32(w,"gemma3."k,v)
#define F(k,v) gguf_write_kv_f32(w,"gemma3."k,v)
    U("block_count",L);U("embedding_length",E);U("feed_forward_length",FF);
    U("attention.head_count",2);U("attention.head_count_kv",1);
    U("attention.key_length",4);U("attention.value_length",4);U("attention.sliding_window",2);
    U("context_length",8);F("attention.layer_norm_rms_epsilon",1e-5f);
    F("rope.freq_base",10000);F("rope.freq_base_swa",100);
    if(mode==4){int32_t pattern[]={1,1,1,1,1,0};gguf_write_kv_i32_array(w,"gemma3.attention.sliding_window_pattern",pattern,L);}
    if(mode==5)U("rope.dimension_count",2);
    if(mode==3){gguf_write_kv_str(w,"gemma3.rope.scaling.type","none");F("rope.scaling.factor",2);}
#undef U
#undef F
    for(int i=0;i<nt;i++){
        uint64_t shape[]={(uint64_t)ts[i].cols,(uint64_t)ts[i].rows};
        gguf_write_tensor_decl(w,ts[i].name,ts[i].rows?2:1,shape,GGUF_TYPE_F32);
    }
    for(int i=0;i<nt;i++)gguf_write_tensor_f32(w,ts[i].name,ts[i].data,
        (uint64_t)(ts[i].rows?ts[i].rows:1)*(uint64_t)ts[i].cols);
    return gguf_write_close(w);
}
static const float golden[6][20] = {
    {0.0284815219f, 2.0770653f, 0.840257327f, 2.14495301f, 1.01165678f, -0.848720263f, 1.85917908f, -1.78981462f, -1.40687608f, -1.98990512f, 1.18628431f, -0.916063865f, -1.45213821f, -0.0245823992f, -0.16720756f, 0.169604259f, -1.9959579f, 1.07541506f, 0.276609638f, 0.360120796f},
    {-0.570720189f, 3.41368277f, 0.506229127f, 3.35326499f, 1.50574507f, -0.229900408f, 1.43876167f, -2.25001649f, -0.693035316f, -1.36250533f, 0.937027711f, -2.57858554f, 0.105093516f, -0.560016325f, 0.81652585f, -2.06830554f, -0.422612787f, -0.943290616f, 1.276241f, -0.735169925f},
    {0.295257479f, 2.43676516f, 2.23295267f, 3.09836103f, 0.214323941f, 1.47703401f, 0.628363728f, -1.82866254f, -1.24438706f, 1.22137316f, 0.437334472f, -0.602960798f, -0.0922575993f, 1.95856526f, 0.660139634f, 0.173496685f, 0.928299618f, 0.764130905f, 0.300113091f, 0.115759867f},
    {-0.639322458f, 1.91085537f, 2.26022598f, 3.32538474f, 0.230779915f, 1.52351152f, -0.439636243f, -1.14275224f, -1.03960675f, 1.56232013f, -0.92676057f, 0.494309351f, 0.295724881f, 2.57815733f, -0.398870789f, 1.65524659f, 0.680390169f, 0.167903562f, 2.18643326f, 0.232517283f},
    {-0.942967415f, 1.56763507f, 1.89774331f, 3.3587374f, 0.9580811f, 1.43461954f, -1.50485312f, -0.720717596f, -0.831504236f, 0.927900376f, -1.84364999f, -0.357137074f, 1.02973416f, 3.37366723f, -1.39991437f, 1.50594611f, 1.18928318f, 0.531966144f, 0.832257138f, -0.0240553715f},
    {1.10110751f, 1.94546621f, 2.6842727f, 5.01301386f, 2.98090966f, 0.0187691071f, -0.227503995f, -3.06417761f, 1.35803079f, 1.09222285f, -0.36719951f, -1.23514149f, 3.11425688f, 1.85084835f, -0.0643194543f, -0.705746921f, 2.36306036f, -1.31010234f, 1.74993547f, -2.25602572f},
};
static const float golden_logits[5] = {0.30526997f, 0.33703298f, -0.33768609f, 0.20574287f, -0.65428814f};
static void tokenizer_checks(const char *path) {
    for (int prefix=0; prefix<=1; prefix++) {
        const char *tokens[]={"<unk>","<bos>","▁","a","b","▁a","ab"};
        float scores[]={-10,-10,-5,-4,-4,-2,-1};
        gguf_writer *w=gguf_write_open(path);
        if(!w){CHECK(0,"open tokenizer fixture");return;}
        gguf_write_kv_str(w,"general.architecture","gemma3");
        gguf_write_kv_str(w,"tokenizer.ggml.model","llama");
        gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,7);
        gguf_write_kv_f32_array(w,"tokenizer.ggml.scores",scores,7);
        gguf_write_kv_u32(w,"tokenizer.ggml.add_bos_token",1);
        gguf_write_kv_u32(w,"tokenizer.ggml.bos_token_id",1);
        gguf_write_kv_u32(w,"tokenizer.ggml.add_space_prefix",prefix);
        CHECK(gguf_write_close(w)==0,"write tokenizer fixture");
        bpe_tokenizer *t=bpe_load(path);int ids[16];
        CHECK(t!=NULL,"load SentencePiece fixture");if(!t)continue;
        CHECK(bpe_encode(t,"a",ids,16)==2&&ids[0]==1&&ids[1]==(prefix?5:3),"metadata controls dummy prefix and BOS");
        CHECK(bpe_encode_raw(t,"ab",ids,16)==1&&ids[0]==6,"raw span has no dummy prefix or BOS");
        CHECK(bpe_encode(t,"",ids,16)==1&&ids[0]==1,"empty sequence returns only requested BOS");
        CHECK(bpe_encode_raw(t,"",ids,16)==0,"empty raw span produces no tokens");
        CHECK(bpe_encode(t,"a",ids,16)==2&&ids[0]==1&&ids[1]==(prefix?5:3),"raw span does not mutate tokenizer");
        bpe_free(t);
    }
}
static int close_values(const float*a,const float*b,int n,float tol){
    for(int i=0;i<n;i++)if(!isfinite(a[i])||fabsf(a[i]-b[i])>tol)return 0;
    return 1;
}
typedef struct{int calls,bad,mutate,fail;float last[E];}probe;
static int observe(void*user,int layer,int pos,int n,int width,float*x){
    probe*p=user;p->calls++;
    if(width!=E||layer<0||layer>=L||pos<0||pos+n>N)p->bad=1;
    else if(!p->mutate&&!close_values(x,golden[layer]+pos*E,n*E,2e-5f))p->bad=1;
    if(p->fail)return 77;
    if(p->mutate&&layer==2)for(int j=0;j<n;j++)x[j*E+1]+=.125f;
    if(layer==L-1)memcpy(p->last,x+(n-1)*E,sizeof(p->last));
    return 0;
}
int main(void){
    char path[]="/tmp/nt_gemma3_XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    CHECK(fixture(path,0)==0,"write tiny Gemma3 GGUF");
    gguf_file*gf=gguf_open(path);nt_dims dims;
    const nt_arch*a=nt_pick_arch("gemma3");
    CHECK(a==&nt_arch_gemma3,"architecture registry selects Gemma3");
    void*m=gf?a->load(gf,&dims):NULL;if(!m){unlink(path);return 1;}
    CHECK(dims.n_layers==L&&dims.kv_dim==4&&dims.vocab==V,"independent head geometry");
    kv_cache*kv=kv_new(L,8,4);if(!kv)return 1;
    int ids[N]={1,2,3,4,0};float base[V],out[V];probe p={0};
    CHECK(a->forward_residual(m,kv,ids,N,0,base,observe,&p)==0&&!p.bad&&p.calls==L,
        "all six layer residuals match independent equations across sliding boundary");
    CHECK(close_values(base,golden_logits,V,2e-5f),"tied output logits match independent equations");
    CHECK(a->forward(m,kv,ids,N,0,out)==0&&!memcmp(base,out,sizeof(out)),"plain forward byte equality");
    CHECK(a->forward_residual(m,kv,ids,N,0,out,NULL,NULL)==0&&!memcmp(base,out,sizeof(out)),"NULL callback byte equality");
    p=(probe){0};int rc=0;
    for(int j=0;j<N;j++)rc|=a->forward_residual(m,kv,ids+j,1,j,out,observe,&p);
    CHECK(!rc&&!p.bad&&p.calls==N*L&&close_values(base,out,V,2e-5f),"decode cache and absolute positions match prefill");
    p=(probe){.mutate=1};
    CHECK(a->forward_residual(m,kv,ids,N,0,out,observe,&p)==0&&!p.bad&&memcmp(base,out,sizeof(out)),"post-block mutation changes downstream logits");
    p=(probe){.fail=1};for(int i=0;i<V;i++)out[i]=-123;
    CHECK(a->forward_residual(m,kv,ids,N,0,out,observe,&p)==77&&p.calls==1,"callback error propagates immediately");
    int same=1;for(int i=0;i<V;i++)same&=out[i]==-123;
    CHECK(same,"callback error leaves logits untouched");
    CHECK(a->forward(m,kv,ids,N,0,out)==0&&!memcmp(base,out,sizeof(out)),"restart retains no callback state");
    p=(probe){0};CHECK(a->forward_residual(m,kv,ids,N,0,NULL,observe,&p)==0&&p.calls==L&&!p.bad,"residual capture without logits");
    int bad=V;CHECK(a->forward(m,kv,&bad,1,0,out)==NT_E_TOKEN,"invalid token refused");
    CHECK(a->forward(m,kv,ids,N,5,out)==NT_E_CAPACITY,"cache overflow refused");
    a->free(m);gguf_close(gf);
    for(int mode=1;mode<=5;mode++){
        CHECK(fixture(path,mode)==0,"write variant GGUF");gf=gguf_open(path);m=gf?a->load(gf,&dims):NULL;
        if(mode!=3)CHECK(!m,"malformed or unsupported tensor/metadata refused");
        else CHECK(m&&a->forward(m,kv,ids,N,0,out)==0&&!memcmp(base,out,sizeof(out)),"rope scaling none ignores factor");
        if(m)a->free(m);
        gguf_close(gf);
    }
    kv_free(kv);tokenizer_checks(path);unlink(path);
    printf("GEMMA3_%s (%d checks)\n",failed?"FAIL":"OK",checks);return failed?1:0;
}
