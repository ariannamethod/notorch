#include "harness/qwen35_vision.h"
#include "sha256.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "qwen35_vision_fixture.h"
#include "qwen35_visual_text_fixture.h"
static int checks,failed;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
static float difference(const float *a,const float *b,int n){float d=0;for(int j=0;j<n;j++)if(fabsf(a[j]-b[j])>d)d=fabsf(a[j]-b[j]);return d;}
static int stop(void *u,int l,int p,int n,int e,float *x){(void)u;(void)l;(void)p;(void)n;(void)e;(void)x;return 37;}
int main(void) {
    char path[]="/tmp/notorch-vision-XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    CHECK(!vision_fixture(path,0),"write vision fixture");gguf_file *g=gguf_open(path);
    nt_qwen35_vision *v=nt_qwen35_vision_load(g);CHECK(v!=NULL,"vision load");if(!v){gguf_close(g);unlink(path);return 1;}
    uint8_t rgb[7*11*3];for(int j=0;j<(int)sizeof(rgb);j++)rgb[j]=(uint8_t)(j*7+3);
    nt_qwen35_image im={0};CHECK(!nt_qwen35_image_prepare(v,rgb,7,11,256,&im),"prepare non-square image");
    CHECK(im.height==8 && im.width==12 && im.tokens==6,"patch and merger geometry");
    unsigned char digest[32];char hex[65];
    CHECK(!nt_sha256(im.patches,24*24*sizeof(float),digest),"hash prepared patches");
    for(int j=0;j<32;j++)snprintf(hex+2*j,3,"%02x",digest[j]);
    CHECK(!strcmp(hex,vision_patches_sha),"PIL patch bytes, channel/time/merge order");
    float *out=NULL;CHECK(!nt_qwen35_vision_encode(v,&im,&out),"encode image");
    CHECK(out && difference(out,vision_golden,48)<2e-5f,"independent full encoder features");
    int ids[]={2,101,100,100,100,100,100,100,102,3},positions[10][3];const float *rows[10];
    CHECK(nt_qwen35_image_positions(v,&im,out,ids,10,positions,rows)==7,"rotary continuation position differs from cache offset");
    CHECK(positions[7][0]==2 && positions[7][1]==3 && positions[7][2]==4 && positions[9][0]==6,"T/H/W coordinates");
    CHECK(rows[1]==NULL && rows[2]==out && rows[7]==out+5*8 && rows[8]==NULL,"sparse embedding rows");
    positions[0][0]=999;ids[4]=3;
    CHECK(nt_qwen35_image_positions(v,&im,out,ids,10,positions,rows)<0 && positions[0][0]==999,"bad placeholders preserve outputs");
    ids[4]=100;ids[9]=101;
    CHECK(nt_qwen35_image_positions(v,&im,out,ids,10,positions,rows)<0,"duplicate image frame refused");
    nt_qwen35_image untouched={.tokens=999};
    CHECK(nt_qwen35_image_prepare(v,rgb,0,11,256,&untouched)==NT_E_ARG && untouched.tokens==999,"bad dimensions preserve output");
    CHECK(nt_qwen35_image_prepare(v,rgb,7,11,15,&untouched)==NT_E_ARG,"below-model pixel budget refused");
    float *sentinel=(float*)&im;float saved=im.patches[0];im.patches[0]=NAN;
    CHECK(nt_qwen35_vision_encode(v,&im,&sentinel)==NT_E_ARG && sentinel==(float*)&im,"NaN patches rejected before output");im.patches[0]=saved;
    free(out);nt_qwen35_image_free(&im);nt_qwen35_vision_free(v);gguf_close(g);
    CHECK(!vision_fixture(path,1),"incomplete vision fixture");g=gguf_open(path);v=nt_qwen35_vision_load(g);
    CHECK(v==NULL,"missing tensor refused");nt_qwen35_vision_free(v);gguf_close(g);
    CHECK(!visual_text_fixture(path,0),"write 3D decoder fixture");g=gguf_open(path);nt_dims dims={0};
    void *m=nt_arch_qwen35.load(g,&dims);CHECK(m!=NULL,"3D decoder load");if(!m){gguf_close(g);unlink(path);return 1;}
    int text[]={0,3,5,12,2,6,7},p[7][3]={{0,0,0},{1,1,1},{1,1,2},{1,2,1},{1,2,2},{3,3,3},{4,4,4}},sections[]={2,1,1};
    float image_rows[32],logits[40],plain[40];const float *emb[7]={0};
    for(int j=0;j<32;j++)image_rows[j]=(float)(j-16)/32;
    for(int t=1;t<5;t++)emb[t]=image_rows+(t-1)*8;
    kv_cache *kv=nt_arch_qwen35.kv_new(m,8);
    for(int chunk=1;chunk<=7;chunk+=2) {
        for(int pos=0;pos<7;pos+=chunk){int take=7-pos<chunk?7-pos:chunk;
            CHECK(!nt_qwen35_forward_3d(m,kv,text+pos,emb+pos,(const int(*)[3])(p+pos),sections,take,pos,pos+take==7?logits:NULL,NULL,NULL),"3D decoder chunk");}
        CHECK(difference(logits,visual_text_golden_logits,40)<2e-5f,"independent M-RoPE and image-row logits");
    }
    CHECK(!nt_qwen35_forward_3d(m,kv,text,NULL,(const int(*)[3])p,sections,7,0,plain,NULL,NULL),"red hand disconnect rows");
    CHECK(difference(plain,visual_text_golden_logits,40)>2e-5f,"gate detects disconnected image");
    CHECK(!nt_qwen35_forward_3d(m,kv,text,emb,NULL,NULL,7,0,plain,NULL,NULL),"red hand replace M-RoPE with 1D");
    CHECK(difference(plain,visual_text_golden_logits,40)>2e-5f,"gate detects lost spatial positions");
    CHECK(!nt_arch_qwen35.forward(m,kv,text,7,0,plain),"ordinary text forward");
    CHECK(!nt_qwen35_forward_3d(m,kv,text,NULL,NULL,NULL,7,0,logits,NULL,NULL) && !memcmp(plain,logits,sizeof(logits)),"NULL media is bit-identical text");
    for(int j=0;j<40;j++)logits[j]=1234;
    int bad_sections[]={2,2,1};CHECK(nt_qwen35_forward_3d(m,kv,text,emb,(const int(*)[3])p,bad_sections,7,0,logits,NULL,NULL)==NT_E_ARG,"invalid axis layout");
    image_rows[0]=NAN;CHECK(nt_qwen35_forward_3d(m,kv,text,emb,(const int(*)[3])p,sections,7,0,logits,NULL,NULL)==NT_E_ARG,"nonfinite embedding");image_rows[0]=-.5f;
    p[0][0]=-1;CHECK(nt_qwen35_forward_3d(m,kv,text,emb,(const int(*)[3])p,sections,7,0,logits,NULL,NULL)==NT_E_ARG,"negative position");p[0][0]=0;
    CHECK(nt_qwen35_forward_3d(m,kv,text,emb,(const int(*)[3])p,sections,7,0,logits,stop,NULL)==37,"callback error");
    for(int j=0;j<40;j++)CHECK(logits[j]==1234,"failure preserves logits");
    CHECK(nt_arch_qwen35.forward(m,kv,text,1,1,logits)==NT_E_STATE,"failed cache requires restart");
    CHECK(!nt_qwen35_forward_3d(m,kv,text,emb,(const int(*)[3])p,sections,7,0,logits,NULL,NULL) && difference(logits,visual_text_golden_logits,40)<2e-5f,"restart restores image result");
    nt_arch_qwen35.kv_free(m,kv);nt_arch_qwen35.free(m);gguf_close(g);unlink(path);
    printf("Qwen3.5 vision: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
