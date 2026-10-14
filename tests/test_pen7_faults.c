/* Exercise the real loader with each allocation failing in turn. The allocator
 * records owned blocks and red zones; the oracle is the public MEMORY status,
 * atomic output, quiet failure, and an empty ownership ledger after cleanup. */
#include "pen7.h"
#include "harness/arch.h"
#include "gguf.h"
#include "sha256.h"
#include "tests/pen7_fault_alloc.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static size_t calls, fail_at, live, fired, guard_errors;
static int armed, checks, failed;
static struct { void *ptr; size_t bytes; } blocks[1024];
#define CHECK(c,s) do { ++checks; if (!(c)) { fprintf(stderr,"FAIL: %s\n",s); ++failed; } } while (0)
void *nt_pen7_fault_malloc(size_t bytes) {
    if (armed && ++calls==fail_at) { fired++; return NULL; }
    if (bytes>SIZE_MAX-32) return NULL;
    unsigned char *p=malloc(bytes+32); if (!p) return NULL;
    size_t slot=0;while(slot<1024&&blocks[slot].ptr)slot++;
    if(slot==1024){free(p);return NULL;}
    blocks[slot].ptr=p;blocks[slot].bytes=bytes;live++;
    memset(p+bytes,0xa7,32);return p;
}
void *nt_pen7_fault_calloc(size_t n,size_t bytes) {
    if(bytes&&n>SIZE_MAX/bytes)return NULL;
    void*p=nt_pen7_fault_malloc(n*bytes);if(p)memset(p,0,n*bytes);return p;
}
void nt_pen7_fault_free(void *ptr) {
    if(!ptr)return;
    for(size_t i=0;i<1024;i++)if(blocks[i].ptr==ptr){
        for(size_t j=0;j<32;j++)if(((unsigned char*)ptr)[blocks[i].bytes+j]!=0xa7)guard_errors++;
        blocks[i].ptr=NULL;live--;break;
    }
    free(ptr);
}
static void hash_pairs(const int32_t*p,size_t bytes,char out[65]) {
    unsigned char d[32];const char*h="0123456789abcdef";nt_sha256(p,bytes,d);
    for(int i=0;i<32;i++){out[2*i]=h[d[i]>>4];out[2*i+1]=h[d[i]&15];}out[64]=0;
}
/* An independent, small but complete family contract. Width 32 permits Q8_0
 * on every matrix, including both learned embedding tables; two blocks expose
 * cleanup after early and late matrix allocations. */
enum { E=32,V=257,F=64,L=2,T=16,N=4+11*L };
typedef struct{char name[80];int rows,cols;} tensor;
static int fixture(const char*path,int q8) {
    gguf_writer*w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"general.architecture","penelope");
#define U(k,v) gguf_write_kv_u32(w,"penelope."k,v)
#define K(k,v) gguf_write_kv_f32(w,"penelope."k,v)
    U("vocab_size",V);U("block_count",L);U("embedding_length",E);U("context_length",T);
    U("feed_forward_length",F);U("attention.head_count",2);U("attention.head_count_kv",2);
    U("attention.head_dim",16);U("rope.dimension_count",16);
    K("attention.layer_norm_rms_epsilon",1e-5f);K("rope.freq_base",10000);K("silu_cutoff",-20);
#undef U
#undef K
    gguf_write_kv_str(w,"penelope.rope.layout","split-half");gguf_write_kv_str(w,"penelope.resonance","dense-gated");
    gguf_write_kv_bool(w,"penelope.position_embedding",1);
    gguf_write_kv_str(w,"tokenizer.ggml.model","penelope");
    gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",0);gguf_write_kv_bool(w,"tokenizer.ggml.add_eos_token",0);
    gguf_write_kv_str(w,"tokenizer.penelope.algorithm","byte-rank-pass-v1");
    gguf_write_kv_bool(w,"tokenizer.penelope.lowercase_ascii",1);gguf_write_kv_u32(w,"tokenizer.penelope.max_input_bytes",8191);
    char spellings[V][8],identity[65];const char*tokens[V],*merges[]={"a a"};int32_t pairs[]={97,97};int extra=0;
    for(int b=0;b<256;b++){
        int printable=(b>=33&&b<=126)||(b>=161&&b<=172)||b>=174,cp=printable?b:256+extra++;
        if(cp<128){spellings[b][0]=(char)cp;spellings[b][1]=0;}
        else{spellings[b][0]=(char)(0xc0|(cp>>6));spellings[b][1]=(char)(0x80|(cp&63));spellings[b][2]=0;}
        tokens[b]=spellings[b];
    }
    tokens[256]="aa";hash_pairs(pairs,sizeof(pairs),identity);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,V);gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,1);
    gguf_write_kv_i32_array(w,"tokenizer.penelope.merge_pairs",pairs,2);gguf_write_kv_str(w,"tokenizer.penelope.sha256",identity);
    tensor ts[N]={{"token_embd.weight",V,E},{"position_embd.weight",T,E},
        {"output_norm.weight",0,E},{"output.weight",V,E}};
    const char*names[]={"attn_norm","attn_q","attn_k","attn_v","attn_output","resonance","resonance_gate","ffn_norm","ffn_gate","ffn_up","ffn_down"};
    int rows[]={0,E,E,E,E,E,0,0,F,F,E},cols[]={E,E,E,E,E,E,2,E,E,E,F},at=4;
    for(int l=0;l<L;l++)for(int k=0;k<11;k++,at++){
        snprintf(ts[at].name,sizeof(ts[at].name),"blk.%d.%s.weight",l,names[k]);ts[at].rows=rows[k];ts[at].cols=cols[k];
    }
    for(int i=0;i<N;i++){uint64_t shape[]={(uint64_t)ts[i].cols,(uint64_t)ts[i].rows};
        gguf_write_tensor_decl(w,ts[i].name,ts[i].rows?2:1,shape,q8&&ts[i].rows?GGUF_TYPE_Q8_0:GGUF_TYPE_F32);}
    float values[V*E];unsigned char packed[V*E/32*34];
    int bad=0;
    for(int i=0;i<N;i++){
        uint64_t count=(uint64_t)(ts[i].rows?ts[i].rows:1)*(uint64_t)ts[i].cols;
        for(uint64_t j=0;j<count;j++)values[j]=ts[i].rows?((int)((j*7+(unsigned)i*3)%23)-11)*.01875f:1.f;
        if(q8&&ts[i].rows){
            size_t row_bytes=(size_t)gguf_type_size(GGUF_TYPE_Q8_0,(uint64_t)ts[i].cols);
            for(int r=0;r<ts[i].rows;r++)bad|=nt_quantize_row(values+(size_t)r*ts[i].cols,packed+(size_t)r*row_bytes,ts[i].cols,GGUF_TYPE_Q8_0);
            bad|=gguf_write_tensor(w,ts[i].name,packed,gguf_type_size(GGUF_TYPE_Q8_0,count));
        }else bad|=gguf_write_tensor_f32(w,ts[i].name,values,count);
    }
    return gguf_write_close(w)||bad;
}
static void sweep(const char*path,int q8){
    CHECK(fixture(path,q8)==0,"write real F32/Q8 fixture");CHECK(live==0,"fixture writer released owned blocks");
    nt_pen7_model*model=NULL;calls=0;fail_at=0;armed=1;
    int rc=nt_pen7_load(path,&model);armed=0;size_t allocations=calls;
    CHECK(rc==NT_PEN7_OK&&model!=NULL,"baseline fixture loads");if(rc)return;
    int ids[8];size_t n=0;nt_pen7_workspace*work=NULL;float logits[V],repr[E];
    CHECK(nt_pen7_encode(model,"AAA",3,ids,8,&n)==NT_PEN7_OK&&n==2&&ids[0]==256,"baseline canonical tokenizer works");
    CHECK(nt_pen7_workspace_new(model,8,&work)==NT_PEN7_OK,"baseline workspace");
    CHECK(nt_pen7_forward(model,work,ids,n,logits,V,repr,E)==NT_PEN7_OK,"baseline body executes finite forward");
    nt_pen7_workspace_free(work);nt_pen7_release(model);CHECK(live==0,"baseline release empties owned ledger");
    FILE*capture=tmpfile();int saved_out=dup(STDOUT_FILENO),saved_err=dup(STDERR_FILENO);
    int capturing=capture&&saved_out>=0&&saved_err>=0;
    fflush(stdout);fflush(stderr);
    if(capturing)capturing=dup2(fileno(capture),STDOUT_FILENO)>=0&&dup2(fileno(capture),STDERR_FILENO)>=0;
    if(!capturing){
        if(saved_out>=0){dup2(saved_out,STDOUT_FILENO);close(saved_out);}
        if(saved_err>=0){dup2(saved_err,STDERR_FILENO);close(saved_err);}
        if(capture)fclose(capture);
        CHECK(0,"quiet capture opened");return;
    }
    CHECK(capturing,"quiet capture opened");
    size_t wrong_status=0,changed=0,leaked=0,not_fired=0,first_bad=0;int first_status=0;
    for(size_t at=1;at<=allocations;at++){
        calls=0;fail_at=at;fired=0;model=(nt_pen7_model*)(uintptr_t)1;armed=1;
        rc=nt_pen7_load(path,&model);armed=0;
        if(rc!=NT_PEN7_E_MEMORY){wrong_status++;if(!first_bad){first_bad=at;first_status=rc;}}
        if(model!=(nt_pen7_model*)(uintptr_t)1){changed++;if(model)nt_pen7_release(model);}
        if(live)leaked++;
        if(fired!=1)not_fired++;
    }
    fflush(stdout);fflush(stderr);long emitted=ftell(capture);
    int restored_out=dup2(saved_out,STDOUT_FILENO),restored_err=dup2(saved_err,STDERR_FILENO);
    close(saved_out);close(saved_err);fclose(capture);
    CHECK(restored_out>=0&&restored_err>=0,"quiet capture restored");
    fail_at=0;
    if(wrong_status)fprintf(stderr,"allocation refusal: %s first=%zu status=%d wrong=%zu/%zu\n",q8?"Q8":"F32",first_bad,first_status,wrong_status,allocations);
    CHECK(wrong_status==0,"every allocation refusal is MEMORY");CHECK(changed==0,"every failed load preserves output handle");
    CHECK(leaked==0&&live==0,"every failed load releases every owned allocation");CHECK(not_fired==0,"each scheduled failure actually fired exactly once");
    CHECK(emitted==0,"failed loads never write stdout or stderr");CHECK(guard_errors==0,"all allocation red zones intact");
    model=NULL;CHECK(nt_pen7_load(path,&model)==NT_PEN7_OK,"load recovers after failure sweep");nt_pen7_release(model);CHECK(live==0,"recovered body releases cleanly");
    printf("PEN7_FAULT_SWEEP %s %zu allocation sites\n",q8?"Q8":"F32",allocations);
}
int main(void){
    char path[]="/tmp/nt_pen7_fault_XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    sweep(path,0);sweep(path,1);unlink(path);
    printf("PEN7_FAULTS_%s (%d checks)\n",failed?"FAIL":"OK",checks);return !!failed;
}
