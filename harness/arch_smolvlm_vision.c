/* SmolVLM / Idefics3 vision, carried over from ocelli's vision.c (GPL-3.0, same authors) with
 * its arithmetic kept so the two can be compared stage by stage: bilinear tiles and a global
 * frame, SigLIP pre-norm blocks with the f16-rounded tanh GELU of llama.cpp's clip, the
 * idefics3 pixel shuffle and the connector projection without bias. The sizes ocelli wrote
 * as constants (tile 512, scale 4, 64 rows a frame) are read from the projector here. */
#include "harness/smolvlm_vision.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

typedef struct { float *w,*b; } lin;
typedef struct { float *n1w,*n1b,*n2w,*n2b; lin q,k,v,o,fc1,fc2; } sblock;
struct nt_smolvlm_vision {
    nt_smolvlm_vision_info info;
    float *patch_w,*patch_b,*pos,*post_w,*post_b,*fc;
    sblock *blocks;
};
static float *alloc_f(size_t n){return n && n<=SIZE_MAX/sizeof(float)?malloc(n*sizeof(float)):NULL;}
static int key_u32(gguf_file *g,const char *s,int *out) {
    char key[128];snprintf(key,sizeof(key),"clip.vision.%s",s);
    const gguf_kv *k=gguf_get_kv(g,key);
    if(!k || k->type!=4 || k->val.u32<1 || k->val.u32>65536)return -1;
    *out=(int)k->val.u32;return 0;
}
/* Dimensions in GGUF order, ne0 first; any dtype gguf_dequant decodes. */
static float *tensor(gguf_file *g,const char *name,int rank,int n0,int n1,int n2,int n3) {
    int i=gguf_find_tensor(g,name);if(i<0)return NULL;
    const gguf_tensor_info *t=&g->tensors[i];const int want[4]={n0,n1,n2,n3};
    if(t->ndim!=(unsigned)rank)return NULL;
    for(int d=0;d<rank;d++)if(t->shape[d]!=(uint64_t)want[d])return NULL;
    return gguf_dequant(g,i);
}
static void lin_free(lin *l){free(l->w);free(l->b);}
void nt_smolvlm_vision_free(nt_smolvlm_vision *v) {
    if(!v)return;
    for(int l=0;v->blocks && l<v->info.layers;l++) {
        sblock *b=&v->blocks[l];free(b->n1w);free(b->n1b);free(b->n2w);free(b->n2b);
        lin_free(&b->q);lin_free(&b->k);lin_free(&b->v);lin_free(&b->o);lin_free(&b->fc1);lin_free(&b->fc2);
    }
    free(v->patch_w);free(v->patch_b);free(v->pos);free(v->post_w);free(v->post_b);free(v->fc);free(v->blocks);free(v);
}
nt_smolvlm_vision *nt_smolvlm_vision_load(gguf_file *g) {
    if(!g || strcmp(g->arch,"clip"))return NULL;
    const gguf_kv *pt=gguf_get_kv(g,"clip.projector_type");
    if(!pt || pt->type!=8 || strcmp(pt->val.str,"idefics3"))return NULL;
    nt_smolvlm_vision *v=calloc(1,sizeof(*v));if(!v)return NULL;
    nt_smolvlm_vision_info *i=&v->info;
    #define M(key,field) if(key_u32(g,key,&i->field))goto bad
    M("embedding_length",width);M("block_count",layers);M("attention.head_count",heads);
    M("feed_forward_length",ffn);M("image_size",image_size);M("patch_size",patch);
    M("projector.scale_factor",scale);M("projection_dim",text_width);
    #undef M
    if(gguf_get_kv(g,"clip.vision.preproc_image_size") && key_u32(g,"preproc_image_size",&i->longest))goto bad;
    const gguf_kv *e=gguf_get_kv(g,"clip.vision.attention.layer_norm_epsilon");
    if(!e || e->type!=6 || !isfinite(e->val.f32) || e->val.f32<=0)goto bad;
    i->eps=e->val.f32;
    for(int s=0;s<2;s++) {
        int n=0;float *a=gguf_read_f32_array(g->path,s?"clip.vision.image_std":"clip.vision.image_mean",&n);
        int ok=a && n==3;
        for(int c=0;ok && c<3;c++){ok=isfinite(a[c]) && (!s || a[c]>0);(s?i->std:i->mean)[c]=a[c];}
        free(a);if(!ok)goto bad;
    }
    int grid=i->image_size/i->patch;
    /* fc1 is told from fc2 by shape below, which needs ffn != width */
    if(i->width%i->heads || i->image_size%i->patch || grid%i->scale || i->ffn==i->width ||
       i->layers>128 || (i->longest && i->longest<i->image_size))goto bad;
    i->patches=grid*grid;i->tokens=(grid/i->scale)*(grid/i->scale);
    int D=i->width,F=i->ffn,P=i->patch,merged=D*i->scale*i->scale;
    v->patch_w=tensor(g,"v.patch_embd.weight",4,P,P,3,D);v->patch_b=tensor(g,"v.patch_embd.bias",1,D,0,0,0);
    v->pos=tensor(g,"v.position_embd.weight",2,D,i->patches,0,0);
    v->post_w=tensor(g,"v.post_ln.weight",1,D,0,0,0);v->post_b=tensor(g,"v.post_ln.bias",1,D,0,0,0);
    v->fc=tensor(g,"mm.model.fc.weight",2,merged,i->text_width,0,0);
    if(!v->patch_w || !v->patch_b || !v->pos || !v->post_w || !v->post_b || !v->fc)goto bad;
    v->blocks=calloc((size_t)i->layers,sizeof(sblock));if(!v->blocks)goto bad;
    for(int l=0;l<i->layers;l++) {
        sblock *b=&v->blocks[l];char key[128];
        #define T(dst,name,rank,n0,n1) snprintf(key,sizeof(key),"v.blk.%d." name,l);if(!(dst=tensor(g,key,rank,n0,n1,0,0)))goto bad
        T(b->n1w,"ln1.weight",1,D,0);T(b->n1b,"ln1.bias",1,D,0);T(b->n2w,"ln2.weight",1,D,0);T(b->n2b,"ln2.bias",1,D,0);
        T(b->q.w,"attn_q.weight",2,D,D);T(b->q.b,"attn_q.bias",1,D,0);T(b->k.w,"attn_k.weight",2,D,D);T(b->k.b,"attn_k.bias",1,D,0);
        T(b->v.w,"attn_v.weight",2,D,D);T(b->v.b,"attn_v.bias",1,D,0);T(b->o.w,"attn_out.weight",2,D,D);T(b->o.b,"attn_out.bias",1,D,0);
        #undef T
        /* The MLP by role, as ocelli resolves it: converters disagree on whether ffn_up or
         * ffn_down holds fc1 (SmolVLM-256M mirrors SmolVLM2-500M); fc1's bias has ffn rows. */
        snprintf(key,sizeof(key),"v.blk.%d.ffn_up.bias",l);int ub=gguf_find_tensor(g,key);if(ub<0)goto bad;
        const char *f1=g->tensors[ub].n_elements==(uint64_t)F?"ffn_up":"ffn_down",*f2=f1[4]=='u'?"ffn_down":"ffn_up";
        snprintf(key,sizeof(key),"v.blk.%d.%s.weight",l,f1);if(!(b->fc1.w=tensor(g,key,2,D,F,0,0)))goto bad;
        snprintf(key,sizeof(key),"v.blk.%d.%s.bias",l,f1);if(!(b->fc1.b=tensor(g,key,1,F,0,0,0)))goto bad;
        snprintf(key,sizeof(key),"v.blk.%d.%s.weight",l,f2);if(!(b->fc2.w=tensor(g,key,2,F,D,0,0)))goto bad;
        snprintf(key,sizeof(key),"v.blk.%d.%s.bias",l,f2);if(!(b->fc2.b=tensor(g,key,1,D,0,0,0)))goto bad;
    }
    return v;
bad: fprintf(stderr,"smolvlm_vision: unsupported or malformed idefics3 mmproj\n");nt_smolvlm_vision_free(v);return NULL;
}
const nt_smolvlm_vision_info *nt_smolvlm_vision_get_info(const nt_smolvlm_vision *v){return v?&v->info:NULL;}
void nt_smolvlm_image_free(nt_smolvlm_image *im){if(im){free(im->pixels);memset(im,0,sizeof(*im));}}

/* The resize ocelli tiles with, notorch_vision.h:92-128 operation for operation. That header
 * cannot join a harness unit: it compiles its own stb_image next to image.c's at link time. */
static float *bilinear(const float *src,int w,int h,int tw,int th) {
    float *dst=alloc_f((size_t)3*th*tw);if(!dst)return NULL;
    float sx=(float)w/tw;
    float sy=(float)h/th;
    for(int ch=0;ch<3;ch++) {
        const float *sc=src+(size_t)ch*h*w;float *dc=dst+(size_t)ch*th*tw;
        for(int y=0;y<th;y++) {
            float fy=(y+0.5f)*sy-0.5f;
            int y0=(int)floorf(fy);
            int y1=y0+1;
            float wy=fy-y0;
            if(y0<0)y0=0;
            if(y1>=h)y1=h-1;
            for(int x=0;x<tw;x++) {
                float fx=(x+0.5f)*sx-0.5f;
                int x0=(int)floorf(fx);
                int x1=x0+1;
                float wx=fx-x0;
                if(x0<0)x0=0;
                if(x1>=w)x1=w-1;
                float val=sc[(size_t)y0*w+x0]*(1-wx)*(1-wy)+sc[(size_t)y0*w+x1]*wx*(1-wy)
                         +sc[(size_t)y1*w+x0]*(1-wx)*wy+sc[(size_t)y1*w+x1]*wx*wy;
                dc[(size_t)y*tw+x]=val;
            }
        }
    }
    return dst;
}
int nt_smolvlm_image_prepare(const nt_smolvlm_vision *v,const uint8_t *rgb,int h,int w,int longest,nt_smolvlm_image *out) {
    if(!v || !rgb || !out || h<1 || w<1 || h>65536 || w>65536 || (size_t)h*w>(1u<<26))return NT_E_ARG;
    const nt_smolvlm_vision_info *i=&v->info;int S=i->image_size;
    if(!longest)longest=i->longest;
    if(longest<S || longest>65536)return NT_E_ARG;
    /* bytes to channel-first [0,1], as notorch_vision.h:49-52 converts them */
    float *img=alloc_f((size_t)3*h*w);if(!img)return NT_E_MEMORY;
    for(int c=0;c<3;c++)for(int y=0;y<h;y++)for(int x=0;x<w;x++)
        img[(size_t)c*h*w+(size_t)y*w+x]=rgb[((size_t)y*w+x)*3+c]/255.0f;
    /* The longest edge goes TO `longest`, up or down, then a grid of image_size tiles. */
    float s=(float)longest/(w>h?w:h);
    int nw=(int)(w*s+0.5f),nh=(int)(h*s+0.5f);
    if(nw<1)nw=1;
    if(nh<1)nh=1;
    if(nw!=w || nh!=h){float *r=bilinear(img,w,h,nw,nh);free(img);if(!r)return NT_E_MEMORY;img=r;w=nw;h=nh;}
    int cols=(w+S-1)/S,rows=(h+S-1)/S,split=rows*cols>1,frames=split?rows*cols+1:1;
    size_t fs=(size_t)3*S*S,plane=(size_t)S*S;
    float *px=alloc_f((size_t)frames*fs),*grid=split?bilinear(img,w,h,cols*S,rows*S):NULL,*glob=bilinear(img,w,h,S,S);
    free(img);
    if(!px || (split && !grid) || !glob){free(px);free(grid);free(glob);return NT_E_MEMORY;}
    size_t f=0,gw=(size_t)cols*S,gh=(size_t)rows*S;
    for(int r=0;split && r<rows;r++)for(int c=0;c<cols;c++,f++)
        for(int ch=0;ch<3;ch++)for(int y=0;y<S;y++)
            memcpy(px+f*fs+ch*plane+(size_t)y*S,grid+ch*gh*gw+((size_t)r*S+y)*gw+(size_t)c*S,(size_t)S*sizeof(float));
    memcpy(px+f*fs,glob,fs*sizeof(float));
    free(grid);free(glob);
    for(size_t k=0;k<(size_t)frames;k++)for(int ch=0;ch<3;ch++)
        for(size_t j=0;j<plane;j++){float *p=px+k*fs+ch*plane+j;*p=(*p-i->mean[ch])/i->std[ch];}
    *out=(nt_smolvlm_image){.frames=frames,.rows=split?rows:1,.cols=split?cols:1,.pixels=px};
    return NT_OK;
}

/* C[m,n] = A[m,k] @ B[k,n], the scores @ V product ocelli runs as a plain sgemm */
static void mm_nn(float *C,const float *A,const float *B,int m,int k,int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,n,0.0f,C,n);
#else
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){float s=0;for(int p=0;p<k;p++)s+=A[(size_t)i*k+p]*B[(size_t)p*n+j];C[(size_t)i*n+j]=s;}
#endif
}
static void affine(float *out,const float *x,const lin *l,int n,int in,int width) {
    mm_t(out,x,l->w,n,in,width);
    for(int t=0;t<n;t++)for(int j=0;j<width;j++)out[(size_t)t*width+j]+=l->b[j];
}
static void layernorm(float *x,const float *g,const float *b,int m,int n,float eps) {
    for(int i=0;i<m;i++) {
        float *r=x+(size_t)i*n;
        double mu=0;for(int j=0;j<n;j++)mu+=r[j];mu/=n;
        double var=0;for(int j=0;j<n;j++){double d=r[j]-mu;var+=d*d;}var/=n;
        float inv=1.0f/sqrtf((float)var+eps);
        for(int j=0;j<n;j++)r[j]=((r[j]-(float)mu)*inv)*g[j]+b[j];
    }
}
/* (float)(__fp16)x where that type exists: IEEE half, nearest, ties to even. Finite input;
 * the GELU below never hands it a value at or past 10 in magnitude. */
static float half(float x) {
    uint32_t u;memcpy(&u,&x,sizeof(u));
    uint32_t a=u&0x7fffffffu;
    if(a<0x38800000u) { /* below 2^-14: half subnormals, one step is 2^-24 */
        float r=nearbyintf(fabsf(x)*16777216.0f)/16777216.0f;
        return u>>31?-r:r;
    }
    a=(a+0xfffu+((a>>13)&1u))&~0x1fffu;
    u=(u&0x80000000u)|a;memcpy(&x,&u,sizeof(x));return x;
}
/* gelu_pytorch_tanh through ggml's f16 table, as ocelli has it: the input rounded to half,
 * the exact tanh GELU of that, the result rounded to half; clamped outside (-10, 10). */
static void gelu16(float *x,size_t n) {
    const float c=0.79788456080286535588f,a=0.044715f;
    for(size_t i=0;i<n;i++) {
        float v=x[i];
        if(v<=-10.0f){x[i]=0.0f;continue;}
        if(v>=10.0f)continue;
        float xr=half(v);
        float g=0.5f*xr*(1.0f+tanhf(c*(xr+a*xr*xr*xr)));
        x[i]=half(g);
    }
}
int nt_smolvlm_vision_tower(const nt_smolvlm_vision *v,const float *frame,float *hidden) {
    if(!v || !frame || !hidden)return NT_E_ARG;
    const nt_smolvlm_vision_info *i=&v->info;
    int S=i->image_size,P=i->patch,G=S/P,N=i->patches,D=i->width,F=i->ffn,HD=D/i->heads,in=3*P*P,rc=NT_E_MEMORY;
    for(size_t j=0;j<(size_t)3*S*S;j++)if(!isfinite(frame[j]))return NT_E_ARG;
    float scale=1.0f/sqrtf((float)HD);
    float *X=alloc_f((size_t)N*in),*x=alloc_f((size_t)N*D),*t=alloc_f((size_t)N*D),*q=alloc_f((size_t)N*D);
    float *k=alloc_f((size_t)N*D),*vv=alloc_f((size_t)N*D),*att=alloc_f((size_t)N*D),*pr=alloc_f((size_t)N*D);
    float *up=alloc_f((size_t)N*F),*qh=alloc_f((size_t)N*HD),*kh=alloc_f((size_t)N*HD),*vh=alloc_f((size_t)N*HD);
    float *sc=alloc_f((size_t)N*N),*oh=alloc_f((size_t)N*HD);
    if(!X || !x || !t || !q || !k || !vv || !att || !pr || !up || !qh || !kh || !vh || !sc || !oh)goto done;
    /* patches row-major (row outer), each read in (channel, row, column) order */
    for(int ph=0;ph<G;ph++)for(int pw=0;pw<G;pw++) {
        float *xr=X+(size_t)(ph*G+pw)*in;int z=0;
        for(int c=0;c<3;c++)for(int kh_=0;kh_<P;kh_++)for(int kw=0;kw<P;kw++)
            xr[z++]=frame[(size_t)c*S*S+(size_t)(ph*P+kh_)*S+(size_t)(pw*P+kw)];
    }
    affine(x,X,&(lin){v->patch_w,v->patch_b},N,in,D);
    for(size_t j=0;j<(size_t)N*D;j++)x[j]+=v->pos[j];
    for(int l=0;l<i->layers;l++) {
        const sblock *b=&v->blocks[l];
        memcpy(t,x,(size_t)N*D*sizeof(float));layernorm(t,b->n1w,b->n1b,N,D,i->eps);
        affine(q,t,&b->q,N,D,D);affine(k,t,&b->k,N,D,D);affine(vv,t,&b->v,N,D,D);
        for(int h=0;h<i->heads;h++) {
            for(int r=0;r<N;r++) {
                memcpy(qh+(size_t)r*HD,q+(size_t)r*D+h*HD,(size_t)HD*sizeof(float));
                memcpy(kh+(size_t)r*HD,k+(size_t)r*D+h*HD,(size_t)HD*sizeof(float));
                memcpy(vh+(size_t)r*HD,vv+(size_t)r*D+h*HD,(size_t)HD*sizeof(float));
            }
            mm_t(sc,qh,kh,N,HD,N);
            for(size_t j=0;j<(size_t)N*N;j++)sc[j]*=scale;
            for(int r=0;r<N;r++)softmax(sc+(size_t)r*N,N);
            mm_nn(oh,sc,vh,N,N,HD);
            for(int r=0;r<N;r++)memcpy(att+(size_t)r*D+h*HD,oh+(size_t)r*HD,(size_t)HD*sizeof(float));
        }
        affine(pr,att,&b->o,N,D,D);for(size_t j=0;j<(size_t)N*D;j++)x[j]+=pr[j];
        memcpy(t,x,(size_t)N*D*sizeof(float));layernorm(t,b->n2w,b->n2b,N,D,i->eps);
        affine(up,t,&b->fc1,N,D,F);gelu16(up,(size_t)N*F);
        affine(pr,up,&b->fc2,N,F,D);for(size_t j=0;j<(size_t)N*D;j++)x[j]+=pr[j];
    }
    layernorm(x,v->post_w,v->post_b,N,D,i->eps);
    rc=NT_E_STATE;
    for(size_t j=0;j<(size_t)N*D;j++)if(!isfinite(x[j]))goto done;
    memcpy(hidden,x,(size_t)N*D*sizeof(float));rc=NT_OK;
done:
    free(X);free(x);free(t);free(q);free(k);free(vv);free(att);free(pr);free(up);free(qh);free(kh);free(vh);free(sc);free(oh);
    return rc;
}
/* Output row hg*g+wg gathers the scale x scale block below it, slot hin*scale+win, from patch
 * (hg*scale+hin, wg*scale+win): the index map ocelli checked against clip.cpp. */
int nt_smolvlm_vision_connect(const nt_smolvlm_vision *v,const float *hidden,float *rows) {
    if(!v || !hidden || !rows)return NT_E_ARG;
    const nt_smolvlm_vision_info *i=&v->info;
    int D=i->width,s=i->scale,grid=i->image_size/i->patch,g=grid/s,merged=D*s*s;
    float *sh=alloc_f((size_t)i->tokens*merged);if(!sh)return NT_E_MEMORY;
    for(int hg=0;hg<g;hg++)for(int wg=0;wg<g;wg++)for(int hin=0;hin<s;hin++)for(int win=0;win<s;win++)
        memcpy(sh+(size_t)(hg*g+wg)*merged+(size_t)(hin*s+win)*D,
               hidden+(size_t)((hg*s+hin)*grid+wg*s+win)*D,(size_t)D*sizeof(float));
    mm_t(rows,sh,v->fc,i->tokens,merged,i->text_width);
    free(sh);return NT_OK;
}
static int image_valid(const nt_smolvlm_image *im) {
    return im && im->pixels && im->rows>=1 && im->cols>=1 && im->frames==(im->rows*im->cols>1?im->rows*im->cols+1:1);
}
int nt_smolvlm_vision_encode(const nt_smolvlm_vision *v,const nt_smolvlm_image *im,float **out) {
    if(!v || !out || !image_valid(im))return NT_E_ARG;
    const nt_smolvlm_vision_info *i=&v->info;
    size_t fs=(size_t)3*i->image_size*i->image_size,rs=(size_t)i->tokens*i->text_width;
    float *hidden=alloc_f((size_t)i->patches*i->width),*res=alloc_f((size_t)im->frames*rs);int rc=NT_E_MEMORY;
    if(hidden && res)for(int f=0;f<im->frames;f++) {
        if((rc=nt_smolvlm_vision_tower(v,im->pixels+f*fs,hidden)) ||
           (rc=nt_smolvlm_vision_connect(v,hidden,res+f*rs)))break;
    }
    if(!rc){*out=res;res=NULL;}
    free(hidden);free(res);return rc;
}

/* The layout ocelli renders (ocelli.c:622-641), which is idefics3's split image: per tile
 * "<fake_token_around_image><row_r_col_c>" and the frame's <image> slots, "\n" after each
 * tile row, then "\n<fake_token_around_image><global-img>", the global frame's slots and a
 * closing "<fake_token_around_image>". The global-only form keeps ocelli's newline after
 * the closing marker. */
int nt_smolvlm_prompt(const nt_smolvlm_vision *v,const nt_smolvlm_image *im,const bpe_tokenizer *tok,
                      const char *instruction,int *ids,int cap) {
    if(!v || !image_valid(im) || !tok || !instruction || !ids || cap<2 || bpe_add_bos(tok))return -1;
    const char *fixed[]={"<|im_start|>","<fake_token_around_image>","<global-img>","<image>","<end_of_utterance>"};
    for(int k=0;k<5;k++)if(bpe_token_id(tok,fixed[k])<0)return -1;
    /* An instruction spelling a control token would become that token. */
    size_t len=strlen(instruction);
    if(len>(1u<<20))return -1;
    int *a=malloc((len+2)*sizeof(int)),*b=malloc((len+2)*sizeof(int)),same=0;
    if(a && b) {
        int na=bpe_encode_raw(tok,instruction,a,(int)len+2),nb=bpe_encode_special(tok,instruction,b,(int)len+2);
        same=na==nb && na<(int)len+2 && !memcmp(a,b,(size_t)na*sizeof(int));
    }
    free(a);free(b);if(!same)return -1;
    int T=v->info.tokens,split=im->frames>1;
    size_t size=(size_t)im->frames*((size_t)T*7+48)+len+128,o=0;
    char *s=malloc(size);if(!s)return -1;
    #define PUT(...) do{if(o<size)o+=(size_t)snprintf(s+o,size-o,__VA_ARGS__);}while(0)
    PUT("<|im_start|>User:");
    for(int r=0;split && r<im->rows;r++) {
        for(int c=0;c<im->cols;c++) {
            char m[48];snprintf(m,sizeof(m),"<row_%d_col_%d>",r+1,c+1);
            if(bpe_token_id(tok,m)<0){free(s);return -1;}
            PUT("<fake_token_around_image>%s",m);
            for(int k=0;k<T;k++)PUT("<image>");
        }
        PUT("\n");
    }
    PUT(split?"\n<fake_token_around_image><global-img>":"<fake_token_around_image><global-img>");
    for(int k=0;k<T;k++)PUT("<image>");
    PUT(split?"<fake_token_around_image>%s<end_of_utterance>\nAssistant:":"<fake_token_around_image>\n%s<end_of_utterance>\nAssistant:",instruction);
    #undef PUT
    if(o>=size){free(s);return -1;}
    int n=bpe_encode_special(tok,s,ids,cap),image=bpe_token_id(tok,"<image>");free(s);
    long count=0;for(int t=0;t<n;t++)count+=ids[t]==image;
    return n<cap && count==(long)im->frames*T?n:-1;
}
int nt_smolvlm_image_rows(const nt_smolvlm_vision *v,const nt_smolvlm_image *im,const float *features,
                          const bpe_tokenizer *tok,const int *ids,int n,const float **rows) {
    if(!v || !image_valid(im) || !features || !tok || !ids || !rows || n<1)return NT_E_ARG;
    int image=bpe_token_id(tok,"<image>");if(image<0)return NT_E_ARG;
    long count=0;for(int t=0;t<n;t++)count+=ids[t]==image;
    if(count!=(long)im->frames*v->info.tokens)return NT_E_ARG;
    size_t k=0;
    for(int t=0;t<n;t++)rows[t]=ids[t]==image?features+(k++)*(size_t)v->info.text_width:NULL;
    return NT_OK;
}
