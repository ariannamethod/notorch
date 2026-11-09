#include "harness/arch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "qwen35_fixture.h"
#include "qwen35_gqa_fixture.h"
static int checks,failed;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
static int inspect(void *user,int layer,int pos,int n,int width,float *x) {
    const float (*golden)[56]=user;CHECK(width==8,"residual width");
    for(int t=0;t<n;t++)for(int j=0;j<width;j++)
        CHECK(fabsf(x[t*width+j]-golden[layer][(pos+t)*width+j])<2e-5f,"independent layer golden");
    return 0;
}
static int stop(void *user,int l,int p,int n,int e,float *x){(void)user;(void)l;(void)p;(void)n;(void)e;(void)x;return 73;}
int main(void) {
    char path[]="/tmp/notorch-qwen35-XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    CHECK(!fixture(path,0),"fixture written");gguf_file *g=gguf_open(path);nt_dims d={0};
    void *m=g?nt_arch_qwen35.load(g,&d):NULL;CHECK(m!=NULL,"fixture loads");if(!m){unlink(path);return 1;}
    int ids[]={0,3,5,12,2,6,7};float out[40];
    kv_cache *kv=nt_arch_qwen35.kv_new(m,8);CHECK(kv!=NULL,"family cache");
    for(int chunk=1;chunk<=7;chunk+=2) {
        for(int pos=0;pos<7;pos+=chunk) {
            int n=7-pos<chunk?7-pos:chunk;
            CHECK(!nt_arch_qwen35.forward_residual(m,kv,ids+pos,n,pos,pos+n==7?out:NULL,inspect,(void*)golden_residual),"chunk/decode pass");
        }
        for(int i=0;i<40;i++)CHECK(fabsf(out[i]-golden_logits[i])<2e-5f,"independent final logits");
    }
    for(int i=0;i<40;i++)out[i]=1234;
    int invalid=40;
    CHECK(nt_arch_qwen35.forward(m,kv,&invalid,1,0,out)==NT_E_TOKEN,"invalid ID");
    CHECK(nt_arch_qwen35.forward(m,kv,ids,7,3,out)==NT_E_CAPACITY,"capacity");
    CHECK(nt_arch_qwen35.forward(m,kv,ids,1,3,out)==NT_E_STATE,"nonconsecutive continuation");
    CHECK(nt_arch_qwen35.forward_residual(m,kv,ids,7,0,out,stop,NULL)==73,"callback failure");
    CHECK(nt_arch_qwen35.forward(m,kv,ids,1,1,out)==NT_E_STATE,"failed cache cannot continue");
    for(int i=0;i<40;i++)CHECK(out[i]==1234,"failure preserves output");
    CHECK(!nt_arch_qwen35.forward(m,kv,ids,7,0,out),"restart from zero resets recurrent state");
    for(int i=0;i<40;i++)CHECK(fabsf(out[i]-golden_logits[i])<2e-5f,"restarted logits");
    kv_cache *wrong=kv_new(4,8,4);
    CHECK(nt_arch_qwen35.forward(m,wrong,ids,1,0,out)==NT_E_CACHE,"ordinary cache refused");kv_free(wrong);
    CHECK(nt_arch_qwen35.kv_new(m,129)==NULL,"model context bound");
    nt_arch_qwen35.kv_free(m,kv);nt_arch_qwen35.free(m);gguf_close(g);
    CHECK(!fixture(path,1),"malformed fixture written");g=gguf_open(path);m=g?nt_arch_qwen35.load(g,&d):NULL;
    CHECK(m==NULL,"missing convolution refused");nt_arch_qwen35.free(m);gguf_close(g);
    printf("Qwen3.5: %d checks, %d failures\n",checks,failed);
    /* Two KV heads, two key heads and two full-attention layers: a head read from
     * the wrong group, or a full layer read at another layer's cache slot, moves
     * these numbers where the fixture above, with one of each, cannot see it. */
    int first=checks,before=failed;
    CHECK(!gqa_fixture(path,0),"GQA fixture written");g=gguf_open(path);
    m=g?nt_arch_qwen35.load(g,&d):NULL;CHECK(m!=NULL,"GQA fixture loads");
    if(m) {
        kv=nt_arch_qwen35.kv_new(m,8);CHECK(kv!=NULL,"GQA family cache");
        for(int chunk=1;kv && chunk<=7;chunk+=2) {
            for(int pos=0;pos<7;pos+=chunk) {
                int n=7-pos<chunk?7-pos:chunk;
                CHECK(!nt_arch_qwen35.forward_residual(m,kv,ids+pos,n,pos,pos+n==7?out:NULL,inspect,(void*)gqa_golden_residual),"GQA chunk/decode pass");
            }
            for(int i=0;i<40;i++)CHECK(fabsf(out[i]-gqa_golden_logits[i])<2e-5f,"GQA independent final logits");
        }
        nt_arch_qwen35.kv_free(m,kv);nt_arch_qwen35.free(m);
    }
    gguf_close(g);unlink(path);
    printf("Qwen3.5 GQA: %d checks, %d failures\n",checks-first,failed-before);return failed?1:0;
}
