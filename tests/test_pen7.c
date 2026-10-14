/* The immutable boundary links with libnotorch alone. Fixtures give every
 * parameter two distinct values; a materialized mixed GGUF is the oracle for
 * tensor-by-tensor mixing. Existing family tests own the independent equations. */
#include "pen7.h"
#include "harness/arch.h"
#include "gguf.h"
#include "sha256.h"
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { E=8, V=260, L=2, F=16, T=40, NT=4+11*L };
static int checks,failed;
#define CHECK(c,s) do { ++checks; if(!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,s); ++failed; } } while(0)
static void hex(const void *data,size_t bytes,char out[65]) {
    unsigned char d[32];const char*h="0123456789abcdef";nt_sha256(data,bytes,d);
    for(int i=0;i<32;i++){out[2*i]=h[d[i]>>4];out[2*i+1]=h[d[i]&15];}out[64]=0;
}
static int hashfile(const char*path,char out[65]) {
    FILE*f=fopen(path,"rb");if(!f)return 0;fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
    void*p=malloc(n);if(!p){fclose(f);return 0;}int ok=fread(p,1,n,f)==(size_t)n;
    fclose(f);if(ok)hex(p,n,out);free(p);return ok;
}
static int u8(int cp,char*out){if(cp<128){out[0]=(char)cp;out[1]=0;return 1;}
    out[0]=(char)(0xc0|(cp>>6));out[1]=(char)(0x80|(cp&63));out[2]=0;return 2;}
static int tokmeta(gguf_writer*w,int mode) {
    char surfaces[V][24],joins[4][50],sha[65];const char*tokens[V],*merges[4];int extra=0;
    int32_t pairs[]={97,97,256,97,32,257,258,32};
    if(mode==1)pairs[0]=98;
    for(int b=0;b<256;b++){int printable=(b>=33&&b<=126)||(b>=161&&b<=172)||b>=174;
        u8(printable?b:256+extra++,surfaces[b]);tokens[b]=surfaces[b];}
    for(int i=0;i<4;i++){int left=pairs[2*i],right=pairs[2*i+1];
        snprintf(surfaces[256+i],24,"%s%s",surfaces[left],surfaces[right]);tokens[256+i]=surfaces[256+i];
        snprintf(joins[i],50,"%s %s",surfaces[left],surfaces[right]);merges[i]=joins[i];}
    hex(pairs,sizeof(pairs),sha);
    if(mode==3)surfaces[259][0]='x';
    if(mode==4)joins[3][0]='x';
    if(mode==5)pairs[0]=256;
    if(mode==6)sha[0]=sha[0]=='f'?'e':'f';
    gguf_write_kv_str(w,"tokenizer.ggml.model","penelope");
    gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",mode==7);
    gguf_write_kv_bool(w,"tokenizer.ggml.add_eos_token",0);
    gguf_write_kv_str(w,"tokenizer.penelope.algorithm","byte-rank-pass-v1");
    gguf_write_kv_bool(w,"tokenizer.penelope.lowercase_ascii",1);
    gguf_write_kv_u32(w,"tokenizer.penelope.max_input_bytes",8191);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,V);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,4);
    gguf_write_kv_i32_array(w,"tokenizer.penelope.merge_pairs",pairs,8);
    return gguf_write_kv_str(w,"tokenizer.penelope.sha256",sha);
}
typedef struct {char name[80];int rows,cols;float data[V*F];} tensor;
static void put(tensor*t,const char*name,int rows,int cols,int salt,float alpha){
    snprintf(t->name,sizeof(t->name),"%s",name);t->rows=rows;t->cols=cols;
    for(int i=0;i<(rows?rows:1)*cols;i++){
        float a=rows?((i*7+salt*3)%23-11)*.0375f:.8f+.03f*((i+salt)%7);
        float b=rows?((i*3+salt*5)%19-9)*.028f:.7f+.047f*((i+salt*3)%9);
        t->data[i]=alpha==1?a:alpha==0?b:alpha*a+(1.f-alpha)*b;
    }
}
static int fixture(const char*path,float alpha,int mode){
    tensor*ts=calloc(NT,sizeof(*ts));if(!ts)return -1;int count=0;
    put(&ts[count++],"token_embd.weight",V,E,1,alpha);put(&ts[count++],"position_embd.weight",T,E,2,alpha);
    put(&ts[count++],"output_norm.weight",0,E,3,alpha);put(&ts[count++],"output.weight",V,E,4,alpha);
    const char*names[]={"attn_norm","attn_q","attn_k","attn_v","attn_output","resonance",
        "resonance_gate","ffn_norm","ffn_gate","ffn_up","ffn_down"};
    int rows[]={0,E,E,E,E,E,0,0,F,F,E},cols[]={E,E,E,E,E,E,2,E,E,E,F};
    for(int l=0;l<L;l++)for(int k=0;k<11;k++){char name[80];snprintf(name,sizeof(name),"blk.%d.%s.weight",l,names[k]);
        put(&ts[count++],name,rows[k],cols[k],5+l*11+k,alpha);}
    if(mode==8)ts[0].data[0]=NAN;
    if(mode==9)for(int i=0;i<V*E;i++)ts[3].data[i]=FLT_MAX;
    if(mode==10)for(int i=0;i<V*E;i++)ts[0].data[i]=FLT_MAX/4;
    gguf_writer*w=gguf_write_open(path);if(!w){free(ts);return -1;}
    gguf_write_kv_str(w,"general.architecture","penelope");
#define U(k,v) gguf_write_kv_u32(w,"penelope."k,v)
#define K(k,v) gguf_write_kv_f32(w,"penelope."k,v)
    U("vocab_size",V);U("block_count",L);U("embedding_length",E);U("context_length",T);
    U("feed_forward_length",F);U("attention.head_count",mode==2?1:2);U("attention.head_dim",mode==2?8:4);
    U("attention.head_count_kv",mode==2?1:2);U("rope.dimension_count",mode==2?8:4);
    K("attention.layer_norm_rms_epsilon",1e-5f);K("rope.freq_base",10000);K("silu_cutoff",-20);
    gguf_write_kv_str(w,"penelope.rope.layout","split-half");gguf_write_kv_str(w,"penelope.resonance","dense-gated");
    gguf_write_kv_bool(w,"penelope.position_embedding",1);tokmeta(w,mode);
    if(mode==11)gguf_write_kv_u32(w,"penelope.vocab_size",V);
#undef U
#undef K
    for(int i=0;i<count;i++){uint64_t shape[]={(uint64_t)ts[i].cols,(uint64_t)ts[i].rows};
        gguf_write_tensor_decl(w,ts[i].name,ts[i].rows?2:1,shape,GGUF_TYPE_F32);}
    for(int i=0;i<count;i++)gguf_write_tensor_f32(w,ts[i].name,ts[i].data,(ts[i].rows?ts[i].rows:1)*ts[i].cols);
    free(ts);return gguf_write_close(w);
}
static int constant(const float*x,int n,float value){for(int i=0;i<n;i++)if(x[i]!=value)return 0;return 1;}
static void fill(float*x,int n){for(int i=0;i<n;i++)x[i]=-123.75f;}
typedef struct{nt_pen7_model*a,*b;int rc;float alpha,out[V],h[E];} job;
static void*run(void*arg){job*j=arg;nt_pen7_workspace*w=NULL;
    if((j->rc=nt_pen7_workspace_new(j->a,16,&w)))return NULL;
    int ids[]={258,3,99,256,17};
    for(int i=0;i<3&&!j->rc;i++)j->rc=nt_pen7_mix_forward(j->a,j->b,j->alpha,w,ids,5,j->out,V,j->h,E);
    nt_pen7_workspace_free(w);return NULL;
}

static void malformed_containers(const char*source,const char*dest,nt_pen7_model*owner){
    FILE*f=fopen(source,"rb");fseek(f,0,SEEK_END);size_t bytes=(size_t)ftell(f);rewind(f);
    unsigned char*data=malloc(bytes);if(!data||fread(data,1,bytes,f)!=bytes){fclose(f);free(data);CHECK(0,"container fixture read");return;}fclose(f);
    gguf_file*g=gguf_open(source);if(!g){free(data);CHECK(0,"container fixture geometry");return;}
    size_t dim=(size_t)g->kv_end+8+strlen(g->tensors[0].name);
    struct{size_t offset,width;uint64_t value;} cases[]={
        {0,4,0},{4,4,2},{8,8,UINT64_MAX},{16,8,GGUF_MAX_KV+1},
        {dim,4,3},{dim+4,8,UINT64_MAX},{dim+24,8,UINT64_MAX}
    };
    for(size_t i=0;i<sizeof(cases)/sizeof(*cases)+1;i++){
        unsigned char old[8];size_t len=bytes;
        if(i<sizeof(cases)/sizeof(*cases)){
            memcpy(old,data+cases[i].offset,cases[i].width);memcpy(data+cases[i].offset,&cases[i].value,cases[i].width);
        }else len=20;
        f=fopen(dest,"wb");int written=f&&fwrite(data,1,len,f)==len;if(f)fclose(f);
        nt_pen7_model*out=owner;int rc=written?nt_pen7_load(dest,&out):0;
        CHECK(written&&rc==NT_PEN7_E_FORMAT&&out==owner,"malformed magic/version/count/dimension/overflow/offset/truncation preserves handle");
        if(out!=owner)nt_pen7_release(out);
        if(i<sizeof(cases)/sizeof(*cases))memcpy(data+cases[i].offset,old,cases[i].width);
    }
    gguf_close(g);free(data);
}
static void small(const char*pa,const char*pb,const char*pm,const char*bad){
    CHECK(!fixture(pa,1,0)&&!fixture(pb,0,0)&&!fixture(pm,.37f,0),"fixtures");
    nt_pen7_model*a=NULL,*b=NULL,*m=NULL,*c=NULL;nt_pen7_workspace*w=NULL,*wm=NULL;
    CHECK(!nt_pen7_load(pa,&a)&&!nt_pen7_load(pb,&b)&&!nt_pen7_load(pm,&m),"load three declared bodies");
    if(!a||!b||!m)return;
    const nt_pen7_info*info=nt_pen7_get_info(a);char sha[65];
    malformed_containers(pa,bad,a);
    CHECK(hashfile(pa,sha)&&!strcmp(sha,info->identity),"identity hashes exact complete consumed GGUF");
    CHECK(info->vocab==V&&info->width==E&&info->layers==L&&info->context==T,"geometry");
    CHECK(strcmp(info->identity,nt_pen7_get_info(b)->identity)&&!strcmp(info->tokenizer_identity,nt_pen7_get_info(b)->tokenizer_identity),"distinct model identity shared tokenizer");
    CHECK(!nt_pen7_workspace_new(a,16,&w)&&!nt_pen7_workspace_new(m,16,&wm),"owned bounded workspaces");
    CHECK(nt_pen7_workspace_bytes(w)<100000,"workspace bounded below full model body");
    int ids[]={258,3,99,256,17},enc[16];size_t n=88;
    CHECK(!nt_pen7_encode(a," AAA ",5,enc,16,&n)&&n==1&&enc[0]==259,"whole-input lowercase canonical merge passes");
    CHECK(!nt_pen7_encode(a,"AAAA",4,enc,16,&n)&&n==2&&enc[0]==256&&enc[1]==256,"ranked left-to-right nonrecursive pass");
    char raw[]={0,(char)255,'A'};CHECK(!nt_pen7_encode(a,raw,3,enc,16,&n)&&n==3&&enc[0]==0&&enc[1]==255&&enc[2]==97,"explicit bytes preserve NUL and high byte");
    const unsigned char*p=nt_pen7_token_piece(a,259,&n);CHECK(p&&n==5&&!memcmp(p," aaa ",5),"owned complete token surface");
    p=nt_pen7_token_piece(a,0,&n);CHECK(p&&n==1&&p[0]==0,"byte zero surface");
    n=88;CHECK(!nt_pen7_token_piece(a,V,&n)&&n==88,"bad surface preserves length");
    enc[0]=989;n=88;CHECK(nt_pen7_encode(a,"abc",3,enc,1,&n)==NT_PEN7_E_CAPACITY&&enc[0]==989&&n==88,"encode refuses truncation atomically");
    CHECK(nt_pen7_encode(a,"x",8192,enc,16,&n)==NT_PEN7_E_CAPACITY&&n==88,"overlength checked before reading input");
    CHECK(!nt_pen7_encode(a,NULL,0,NULL,0,&n)&&n==0,"empty encoder");
    float av[V],bv[V],mv[V],out[V],ah[E],bh[E],mh[E],h[E];
    CHECK(!nt_pen7_forward(a,w,ids,5,av,V,ah,E)&&!nt_pen7_forward(b,w,ids,5,bv,V,bh,E),"shared workspace has no sequence history");
    CHECK(!nt_pen7_forward(m,wm,ids,5,mv,V,mh,E),"materialized mixed model reference");
    CHECK(!nt_pen7_mix_forward(a,b,.37f,w,ids,5,out,V,h,E)&&!memcmp(out,mv,sizeof(out))&&!memcmp(h,mh,sizeof(h)),"all-parameter mixture equals materialized GGUF exactly");
    CHECK(memcmp(out,av,sizeof(out))&&memcmp(out,bv,sizeof(out))&&memcmp(h,ah,sizeof(h))&&memcmp(h,bh,sizeof(h)),"mixture changes both logits and contextual representation");
    CHECK(!nt_pen7_mix_forward(a,b,1,w,ids,5,out,V,h,E)&&!memcmp(out,av,sizeof(out))&&!memcmp(h,ah,sizeof(h)),"alpha one exact source A");
    CHECK(!nt_pen7_mix_forward(a,b,0,w,ids,5,out,V,h,E)&&!memcmp(out,bv,sizeof(out))&&!memcmp(h,bh,sizeof(h)),"alpha zero exact source B");
    CHECK(!nt_pen7_forward(a,w,ids,5,NULL,0,h,E)&&!memcmp(h,ah,sizeof(h)),"representation only");
    CHECK(!nt_pen7_forward(a,w,ids,5,out,V,NULL,0)&&!memcmp(out,av,sizeof(out)),"logits only");
    int other[]={99,3,258,256,17};
    CHECK(!nt_pen7_forward(a,w,other,5,out,V,h,E)&&memcmp(h,ah,sizeof(h)),"context changes last-token representation");
    fill(out,V);fill(h,E);
#define REFUSE(expr,msg) CHECK((expr)!=0&&constant(out,V,-123.75f)&&constant(h,E,-123.75f),msg)
    REFUSE(nt_pen7_mix_forward(a,b,NAN,w,ids,5,out,V,h,E),"NaN alpha preserves outputs");
    REFUSE(nt_pen7_mix_forward(a,b,INFINITY,w,ids,5,out,V,h,E),"infinite alpha preserves outputs");
    REFUSE(nt_pen7_mix_forward(a,b,-.1f,w,ids,5,out,V,h,E),"negative alpha");
    REFUSE(nt_pen7_mix_forward(a,b,1.01f,w,ids,5,out,V,h,E),"alpha beyond one");
    REFUSE(nt_pen7_forward(a,w,ids,17,out,V,h,E),"capacity checked before ids");
    REFUSE(nt_pen7_forward(a,w,ids,0,out,V,h,E),"empty forward");
    REFUSE(nt_pen7_forward(a,w,ids,5,out,V-1,h,E),"short logits output");
    REFUSE(nt_pen7_forward(a,w,ids,5,out,V,h,E-1),"short representation output");
    REFUSE(nt_pen7_forward(a,w,ids,5,out,V,out,E),"overlapping output ranges");
    int invalid[]={V};REFUSE(nt_pen7_forward(a,w,invalid,1,out,V,h,E),"bad id");
    for(int mode=1;mode<=2;mode++){
        CHECK(!fixture(bad,1,mode)&&!nt_pen7_load(bad,&c),"well-formed incompatible body");
        REFUSE(nt_pen7_mix_forward(a,c,.5f,w,ids,5,out,V,h,E),"incompatible tokenizer or attention geometry");
        nt_pen7_release(c);c=NULL;
    }
    for(int mode=3;mode<=8;mode++){
        CHECK(!fixture(bad,1,mode),"malformed fixture");c=a;
        CHECK(nt_pen7_load(bad,&c)==NT_PEN7_E_FORMAT&&c==a,"bad token/merge/pair/hash/BOS/nonfinite preserves output handle");
    }
    CHECK(!fixture(bad,1,11),"duplicate metadata fixture");c=a;
    CHECK(nt_pen7_load(bad,&c)==NT_PEN7_E_FORMAT&&c==a,"duplicate metadata refused before publication");
    c=a;CHECK(nt_pen7_load("/this-path-does-not-exist-pen7",&c)==NT_PEN7_E_IO&&c==a,"missing file quiet failed load");
    unlink(bad);CHECK(!mkfifo(bad,0600),"FIFO boundary fixture");
    CHECK(nt_pen7_load(bad,&c)==NT_PEN7_E_FORMAT&&c==a,"nonregular file refused without blocking");unlink(bad);
    CHECK(!fixture(bad,1,9)&&!nt_pen7_load(bad,&c),"finite weights with overflowing head");
    REFUSE(nt_pen7_forward(c,w,ids,5,out,V,h,E),"late overflow keeps both outputs untouched");nt_pen7_release(c);
    CHECK(!fixture(bad,1,10)&&!nt_pen7_load(bad,&c),"finite embeddings with overflowing RMS accumulator");
    REFUSE(nt_pen7_forward(c,w,ids,5,out,V,h,E),"intermediate RMS overflow cannot masquerade as finite zero result");nt_pen7_release(c);
#undef REFUSE
    job jobs[4]={{.a=a,.b=b,.alpha=1},{.a=a,.b=b,.alpha=0},{.a=a,.b=b,.alpha=.37f},{.a=a,.b=b,.alpha=.37f}};pthread_t threads[4];
    for(int i=0;i<4;i++)CHECK(!pthread_create(&threads[i],NULL,run,&jobs[i]),"concurrent shared immutable bodies");
    for(int i=0;i<4;i++){pthread_join(threads[i],NULL);const float*reference=i==0?av:i==1?bv:mv;const float*rh=i==0?ah:i==1?bh:mh;
        CHECK(!jobs[i].rc&&!memcmp(jobs[i].out,reference,sizeof(av))&&!memcmp(jobs[i].h,rh,sizeof(ah)),"each owner's workspace remains independent");}
    CHECK(hashfile(pa,sha)&&!strcmp(sha,info->identity),"forwards never alter source model bytes");
    FILE*f=fopen(pa,"r+b");fputc(0,f);fclose(f);unlink(pa);
    CHECK(!nt_pen7_forward(a,w,ids,5,out,V,h,E)&&!memcmp(out,av,sizeof(out))&&!strcmp(sha,info->identity),"source overwrite and unlink preserve loaded body and identity");
    nt_pen7_model*kept=nt_pen7_retain(a);nt_pen7_release(a);
    CHECK(kept==a&&!nt_pen7_forward(kept,w,ids,5,out,V,h,E),"explicit retained ownership");
    nt_pen7_release(kept);nt_pen7_workspace_free(w);nt_pen7_workspace_free(wm);nt_pen7_release(b);nt_pen7_release(m);
}
static int grab(void*user,int layer,int pos,int n,int width,float*x){
    (void)layer;(void)pos;memcpy(user,x+(size_t)(n-1)*width,(size_t)width*sizeof(float));return 0;
}
static void real(const char*path){
    nt_pen7_model*m=NULL;CHECK(!nt_pen7_load(path,&m),"real model quiet load");if(!m)return;
    const nt_pen7_info*i=nt_pen7_get_info(m);char sha[65];CHECK(hashfile(path,sha)&&!strcmp(sha,i->identity),"real exact full SHA256");
    gguf_file*g=gguf_open(path);nt_dims dims;void*body=g?nt_arch_penelope.load(g,&dims):NULL;
    CHECK(body!=NULL,"same real file in harness");if(!body){nt_pen7_release(m);return;}
    nt_pen7_workspace*w=NULL;CHECK(!nt_pen7_workspace_new(m,256,&w),"bounded real full context");
    kv_cache*kv=kv_new(dims.n_layers,256,dims.kv_dim);
    float *a=malloc((size_t)i->vocab*sizeof(float)),*b=malloc((size_t)i->vocab*sizeof(float));
    float *h=malloc((size_t)i->width*sizeof(float)),*res=malloc((size_t)i->width*sizeof(float)),*expected=malloc((size_t)i->width*sizeof(float));
    int ids[256];for(int n=0;n<256;n++)ids[n]=(n*173+11)%i->vocab;
    int norm=gguf_find_tensor(g,"output_norm.weight");float*scale=gguf_dequant(g,norm);
    const int sizes[]={1,9,33,256};
    for(int s=0;s<4;s++){
        int n=sizes[s];CHECK(!nt_pen7_forward(m,w,ids,n,a,i->vocab,h,i->width),"real immutable forward");
        CHECK(!nt_arch_penelope.forward_residual(body,kv,ids,n,0,b,grab,res),"real harness forward");
        CHECK(!memcmp(a,b,(size_t)i->vocab*sizeof(float)),"real logits exact harness parity");
        rmsnorm(expected,res,scale,i->width,1e-5f);
        CHECK(!memcmp(h,expected,(size_t)i->width*sizeof(float)),"real representation equals normalized final contextual residual");
    }
    CHECK(hashfile(path,sha)&&!strcmp(sha,i->identity),"real source hash unchanged");
    printf("PEN7_REAL %s vocab=%d width=%d workspace256=%zu bytes\n",sha,i->vocab,i->width,nt_pen7_workspace_bytes(w));
    free(scale);free(a);free(b);free(h);free(res);free(expected);kv_free(kv);nt_arch_penelope.free(body);gguf_close(g);nt_pen7_workspace_free(w);nt_pen7_release(m);
}

static void real_pair(const char*pa,const char*pb){
    nt_pen7_model*a=NULL,*b=NULL;nt_pen7_workspace*w=NULL;
    CHECK(!nt_pen7_load(pa,&a)&&!nt_pen7_load(pb,&b),"both real immutable sources");
    if(!a||!b){nt_pen7_release(a);nt_pen7_release(b);return;}
    const nt_pen7_info*i=nt_pen7_get_info(a);int ids[32];size_t n=0;
    const char*text="The trees remember the rain.";
    CHECK(!nt_pen7_encode(a,text,strlen(text),ids,32,&n)&&n>0,"real pair canonical context");
    CHECK(!nt_pen7_workspace_new(a,32,&w),"real pair bounded shared scratch");
    size_t vb=(size_t)i->vocab*sizeof(float),eb=(size_t)i->width*sizeof(float);
    float*av=malloc(vb),*bv=malloc(vb),*out=malloc(vb),*ah=malloc(eb),*bh=malloc(eb),*h=malloc(eb);
    CHECK(!nt_pen7_forward(a,w,ids,n,av,i->vocab,ah,i->width)&&!nt_pen7_forward(b,w,ids,n,bv,i->vocab,bh,i->width),"two real baseline contexts");
    CHECK(!nt_pen7_mix_forward(a,b,1,w,ids,n,out,i->vocab,h,i->width)&&!memcmp(out,av,vb)&&!memcmp(h,ah,eb),"real alpha one endpoint exact");
    CHECK(!nt_pen7_mix_forward(a,b,0,w,ids,n,out,i->vocab,h,i->width)&&!memcmp(out,bv,vb)&&!memcmp(h,bh,eb),"real alpha zero endpoint exact");
    CHECK(!nt_pen7_mix_forward(a,b,.37f,w,ids,n,out,i->vocab,h,i->width)&&memcmp(out,av,vb)&&memcmp(out,bv,vb)&&memcmp(h,ah,eb)&&memcmp(h,bh,eb),"real interior mixture differs from both sources in logits and representation");
    float ld=0,hd=0;for(int j=0;j<i->vocab;j++)if(fabsf(out[j]-av[j])>ld)ld=fabsf(out[j]-av[j]);
    for(int j=0;j<i->width;j++)if(fabsf(h[j]-ah[j])>hd)hd=fabsf(h[j]-ah[j]);
    printf("PEN7_REAL_MIX alpha=.37 tokens=%zu max_abs_vs_A_logits=%.9g repr=%.9g workspace32=%zu bytes\n",n,ld,hd,nt_pen7_workspace_bytes(w));
    CHECK(!nt_pen7_forward(a,w,ids,n,out,i->vocab,h,i->width)&&!memcmp(out,av,vb)&&!memcmp(h,ah,eb),"real mixture leaves source A unchanged");
    CHECK(!nt_pen7_forward(b,w,ids,n,out,i->vocab,h,i->width)&&!memcmp(out,bv,vb)&&!memcmp(h,bh,eb),"real mixture leaves source B unchanged");
    free(av);free(bv);free(out);free(ah);free(bh);free(h);nt_pen7_workspace_free(w);nt_pen7_release(a);nt_pen7_release(b);
}
int main(int argc,char**argv){
    if(argc==3&&!strcmp(argv[1],"--fixtures")){
        char a[1024],b[1024],m[1024];
        if(snprintf(a,sizeof(a),"%s/a.gguf",argv[2])>=(int)sizeof(a) ||
           snprintf(b,sizeof(b),"%s/b.gguf",argv[2])>=(int)sizeof(b) ||
           snprintf(m,sizeof(m),"%s/m.gguf",argv[2])>=(int)sizeof(m))return 1;
        return !!(fixture(a,1,0)||fixture(b,0,0)||fixture(m,.37f,0));
    }
    char dir[]="/tmp/nt_pen7_XXXXXX";if(!mkdtemp(dir))return 1;
    char a[128],b[128],m[128],bad[128];snprintf(a,sizeof(a),"%s/a.gguf",dir);snprintf(b,sizeof(b),"%s/b.gguf",dir);snprintf(m,sizeof(m),"%s/m.gguf",dir);snprintf(bad,sizeof(bad),"%s/bad.gguf",dir);
    small(a,b,m,bad);for(int i=1;i<argc;i++)real(argv[i]);
    if(argc==3)real_pair(argv[1],argv[2]);
    unlink(a);unlink(b);unlink(m);unlink(bad);rmdir(dir);
    printf("PEN7_%s (%d checks)\n",failed?"FAIL":"OK",checks);return !!failed;
}
