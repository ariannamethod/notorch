/* Qwen3.5 native vision transformer. Equations/layout follow Qwen's Apache-2.0
 * Transformers implementation: 3D patch projection, learned bilinear positions,
 * axial RoPE, full image attention, pre-shuffle LayerNorm and 2x2 merger. */
#include "harness/qwen35_vision.h"
#include "harness/image.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct { float *w,*b; int in,out; } linear;
typedef struct { float *nw[2],*nb[2]; linear qkv,proj,fc1,fc2; } block;
struct nt_qwen35_vision {
    nt_qwen35_vision_info info;
    int ffn,side; float theta;
    linear patch,fc1,fc2;
    float *pos,*norm_w,*norm_b;
    block *blocks;
};
static void *alloc_f(size_t n,size_t d) {
    return n && d && n<=SIZE_MAX/d/sizeof(float) ? malloc(n*d*sizeof(float)) : NULL;
}
static int meta(gguf_file *g,const char *s,int *out) {
    char key[128];snprintf(key,sizeof(key),"qwen35_vision.%s",s);
    const gguf_kv *k=gguf_get_kv(g,key);
    if(!k || k->type!=4 || k->val.u32>2147483647u)return -1;
    *out=(int)k->val.u32;return 0;
}
static float *tensor(gguf_file *g,const char *s,int rows,int cols,int rank) {
    int i=gguf_find_tensor(g,s);if(i<0)return NULL;
    const gguf_tensor_info *t=&g->tensors[i];
    if(t->ndim!=(unsigned)rank || t->shape[0]!=(unsigned)cols ||
       (rank==2 && t->shape[1]!=(unsigned)rows) ||
       (t->dtype!=GGUF_TYPE_F32 && t->dtype!=GGUF_TYPE_BF16))return NULL;
    return gguf_dequant(g,i);
}
static int weights(gguf_file *g,linear *l,const char *name,int in,int out) {
    char key[160]; l->in=in;l->out=out;
    snprintf(key,sizeof(key),"%s.weight",name);l->w=tensor(g,key,out,in,2);
    snprintf(key,sizeof(key),"%s.bias",name);l->b=tensor(g,key,1,out,1);
    return !l->w || !l->b;
}
static void linear_free(linear *l){free(l->w);free(l->b);}
void nt_qwen35_vision_free(nt_qwen35_vision *v) {
    if(!v)return;
    for(int i=0;v->blocks && i<v->info.layers;i++) {
        block *b=&v->blocks[i];for(int j=0;j<2;j++){free(b->nw[j]);free(b->nb[j]);}
        linear_free(&b->qkv);linear_free(&b->proj);linear_free(&b->fc1);linear_free(&b->fc2);
    }
    linear_free(&v->patch);linear_free(&v->fc1);linear_free(&v->fc2);
    free(v->pos);free(v->norm_w);free(v->norm_b);free(v->blocks);free(v);
}
nt_qwen35_vision *nt_qwen35_vision_load(gguf_file *g) {
    if(!g || strcmp(g->arch,"qwen35_vision"))return NULL;
    nt_qwen35_vision *v=calloc(1,sizeof(*v));if(!v)return NULL;
    nt_qwen35_vision_info *i=&v->info;int version,channels;
    #define M(key,field) if(meta(g,key,&field))goto bad
    M("source_layout_version",version);M("channels",channels);
    M("embedding_length",i->width);M("output_length",i->output_width);M("block_count",i->layers);
    M("attention.head_count",i->heads);M("feed_forward_length",v->ffn);M("patch_size",i->patch);
    M("temporal_patch_size",i->temporal);M("merge_size",i->merge);M("position_count",i->positions);
    M("min_pixels",i->min_pixels);M("max_pixels",i->max_pixels);M("text_rope_dim",i->rope_dim);
    M("image_token_id",i->image_id);M("vision_start_token_id",i->start_id);M("vision_end_token_id",i->end_id);
    M("mrope_section.0",i->section[0]);M("mrope_section.1",i->section[1]);M("mrope_section.2",i->section[2]);
    #undef M
    const gguf_kv *theta=gguf_get_kv(g,"qwen35_vision.rope.freq_base");
    if(!theta || theta->type!=6 || !isfinite(theta->val.f32) || theta->val.f32<=0)goto bad;
    v->theta=theta->val.f32;
    if(version!=1 || channels!=3 || i->temporal!=2 || i->merge!=2 || i->patch<1 || i->patch>64 ||
       i->width<4 || i->width>4096 || i->output_width<1 || i->output_width>8192 ||
       i->heads<1 || i->heads>128 || i->width%i->heads || (i->width/i->heads)%4 ||
       i->layers<1 || i->layers>128 || v->ffn<1 || v->ffn>65536 || i->positions<4 || i->positions>65536 ||
       i->min_pixels<i->patch*i->patch*4 || i->max_pixels<i->min_pixels || i->max_pixels>(1<<26) ||
       i->rope_dim<2 || i->rope_dim>512 || i->rope_dim%2 ||
       i->section[0]+i->section[1]+i->section[2]!=i->rope_dim/2 ||
       i->section[1]>(i->rope_dim/2+1)/3 || i->section[2]>i->rope_dim/6 ||
       i->image_id==i->start_id || i->image_id==i->end_id || i->start_id==i->end_id)goto bad;
    v->side=(int)sqrt((double)i->positions);if(v->side*v->side!=i->positions)goto bad;
    if(weights(g,&v->patch,"patch_embed.proj",3*i->temporal*i->patch*i->patch,i->width) ||
       weights(g,&v->fc1,"merger.linear_fc1",4*i->width,4*i->width) ||
       weights(g,&v->fc2,"merger.linear_fc2",4*i->width,i->output_width))goto bad;
    v->pos=tensor(g,"pos_embed.weight",i->positions,i->width,2);
    v->norm_w=tensor(g,"merger.norm.weight",1,i->width,1);v->norm_b=tensor(g,"merger.norm.bias",1,i->width,1);
    if(!v->pos || !v->norm_w || !v->norm_b)goto bad;
    v->blocks=calloc((size_t)i->layers,sizeof(block));if(!v->blocks)goto bad;
    for(int l=0;l<i->layers;l++) {
        block *b=&v->blocks[l];char key[128];
        for(int n=0;n<2;n++) {
            snprintf(key,sizeof(key),"blocks.%d.norm%d.weight",l,n+1);b->nw[n]=tensor(g,key,1,i->width,1);
            snprintf(key,sizeof(key),"blocks.%d.norm%d.bias",l,n+1);b->nb[n]=tensor(g,key,1,i->width,1);
            if(!b->nw[n] || !b->nb[n])goto bad;
        }
        #define W(field,name,in,out) snprintf(key,sizeof(key),"blocks.%d." name,l);if(weights(g,&b->field,key,in,out))goto bad
        W(qkv,"attn.qkv",i->width,3*i->width);W(proj,"attn.proj",i->width,i->width);
        W(fc1,"mlp.linear_fc1",i->width,v->ffn);W(fc2,"mlp.linear_fc2",v->ffn,i->width);
        #undef W
    }
    return v;
bad: fprintf(stderr,"qwen35_vision: unsupported or malformed source-layout GGUF\n");nt_qwen35_vision_free(v);return NULL;
}
const nt_qwen35_vision_info *nt_qwen35_vision_get_info(const nt_qwen35_vision *v){return v?&v->info:NULL;}
void nt_qwen35_image_free(nt_qwen35_image *im){if(im){free(im->patches);memset(im,0,sizeof(*im));}}
int nt_qwen35_image_prepare(const nt_qwen35_vision *v,const uint8_t *rgb,int h,int w,int cap,nt_qwen35_image *out) {
    if(!v || !rgb || !out || h<1 || w<1 || h>65536 || w>65536 ||
       (double)(h>w?h:w)/(h>w?w:h)>200)return NT_E_ARG;
    const nt_qwen35_vision_info *i=&v->info;int f=i->patch*i->merge;
    if(!cap)cap=i->max_pixels;
    if(cap<i->min_pixels || cap>i->max_pixels)return NT_E_ARG;
    double H=nearbyint((double)h/f)*f,W=nearbyint((double)w/f)*f;
    if(H*W>cap) { double beta=sqrt((double)h*w/cap);H=fmax(f,floor(h/beta/f)*f);W=fmax(f,floor(w/beta/f)*f); }
    else if(H*W<i->min_pixels){double beta=sqrt((double)i->min_pixels/((double)h*w));H=ceil(h*beta/f)*f;W=ceil(w*beta/f)*f;}
    if(H<f || W<f || H>65536 || W>65536 || H*W>(1<<26))return NT_E_ARG;
    uint8_t *res=NULL;if(nt_image_resize_pil(rgb,h,w,(int)H,(int)W,&res))return NT_E_MEMORY;
    nt_qwen35_image im={.height=(int)H,.width=(int)W,.grid_h=(int)H/i->patch,.grid_w=(int)W/i->patch};
    int N=im.grid_h*im.grid_w,P=i->patch;im.tokens=N/4;
    im.patches=alloc_f(N,6*(size_t)P*P);if(!im.patches){free(res);return NT_E_MEMORY;}
    size_t z=0;
    for(int br=0;br<im.grid_h/2;br++)for(int bc=0;bc<im.grid_w/2;bc++)
        for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++)
            for(int c=0;c<3;c++)for(int t=0;t<2;t++)for(int y=0;y<P;y++)for(int x=0;x<P;x++) {
                int yy=(2*br+dr)*P+y,xx=(2*bc+dc)*P+x;
                im.patches[z++]=((float)((double)res[((size_t)yy*im.width+xx)*3+c]/255.0)-0.5f)/0.5f;
            }
    free(res);*out=im;return NT_OK;
}
static void norm(float *out,const float *x,const float *w,const float *b,int n,int d) {
    for(int t=0;t<n;t++) {
        const float *row=x+(size_t)t*d;double sum=0,var=0;
        for(int j=0;j<d;j++)sum+=row[j];
        float mean=(float)(sum/d);
        for(int j=0;j<d;j++){double z=(double)row[j]-mean;var+=z*z;}
        float inv=1.0f/sqrtf((float)(var/d)+1e-6f);
        for(int j=0;j<d;j++)out[(size_t)t*d+j]=(row[j]-mean)*inv*w[j]+b[j];
    }
}
static void affine(float *out,const float *x,const linear *l,int n) {
    mm_t(out,x,l->w,n,l->in,l->out);
    for(int t=0;t<n;t++)for(int j=0;j<l->out;j++)out[(size_t)t*l->out+j]+=l->b[j];
}
static int image_valid(const nt_qwen35_vision *v,const nt_qwen35_image *im) {
    return v && im && im->grid_h>=2 && im->grid_w>=2 && im->grid_h%2==0 && im->grid_w%2==0 &&
        im->grid_h<=65536/v->info.patch && im->grid_w<=65536/v->info.patch &&
        im->height==im->grid_h*v->info.patch && im->width==im->grid_w*v->info.patch &&
        (size_t)im->height*im->width<=(1<<26) && (size_t)im->tokens==(size_t)im->grid_h*im->grid_w/4;
}
int nt_qwen35_vision_encode(const nt_qwen35_vision *v,const nt_qwen35_image *im,float **out) {
    if(!out || !image_valid(v,im) || !im->patches)return NT_E_ARG;
    int N=im->grid_h*im->grid_w,D=v->info.width,HD=D/v->info.heads,F=v->ffn,rc=NT_E_MEMORY;
    int tile=N<128?N:128;
    for(size_t j=0;j<(size_t)N*v->patch.in;j++)if(!isfinite(im->patches[j]))return NT_E_ARG;
    float *x=alloc_f(N,D),*xn=alloc_f(N,D),*qkv=alloc_f(N,3*D),*a=alloc_f(N,D),*tmp=alloc_f(N,D);
    float *ff=alloc_f(N,F),*q=alloc_f(N,HD),*k=alloc_f(N,HD),*vt=alloc_f(HD,N),*att=alloc_f(tile,N);
    float *ho=alloc_f(N,HD),*merged=alloc_f(N,D),*result=alloc_f(im->tokens,v->info.output_width);
    if(!x || !xn || !qkv || !a || !tmp || !ff || !q || !k || !vt || !att || !ho || !merged || !result)goto done;
    affine(x,im->patches,&v->patch,N);
    for(int t=0;t<N;t++) {
        int row=(t/(4*(im->grid_w/2)))*2+(t/2)%2,col=(t/4)%(im->grid_w/2)*2+t%2;
        float yy=(float)(row*(v->side-1))/(im->grid_h-1),xx=(float)(col*(v->side-1))/(im->grid_w-1);
        int y0=(int)floorf(yy),x0=(int)floorf(xx),y1=y0+1<v->side?y0+1:y0,x1=x0+1<v->side?x0+1:x0;
        float wy=yy-y0,wx=xx-x0,ws[4]={(1-wy)*(1-wx),(1-wy)*wx,wy*(1-wx),wy*wx};
        int ix[4]={y0*v->side+x0,y0*v->side+x1,y1*v->side+x0,y1*v->side+x1};
        for(int j=0;j<D;j++){float p=0;for(int z=0;z<4;z++)p+=v->pos[(size_t)ix[z]*D+j]*ws[z];x[(size_t)t*D+j]+=p;}
    }
    for(int l=0;l<v->info.layers;l++) {
        const block *b=&v->blocks[l];norm(xn,x,b->nw[0],b->nb[0],N,D);affine(qkv,xn,&b->qkv,N);
        for(int h=0;h<v->info.heads;h++) {
            for(int t=0;t<N;t++) {
                size_t offset=(size_t)t*3*(size_t)D+(size_t)h*(size_t)HD;
                memcpy(q+(size_t)t*HD,qkv+offset,(size_t)HD*sizeof(float));
                memcpy(k+(size_t)t*HD,qkv+offset+(size_t)D,(size_t)HD*sizeof(float));
                for(int j=0;j<HD;j++)vt[(size_t)j*N+t]=qkv[offset+2*(size_t)D+(size_t)j];
                int row=t/(4*(im->grid_w/2))*2+(t/2)%2,col=(t/4)%(im->grid_w/2)*2+t%2;
                for(int j=0;j<HD/2;j++) {
                    float inv=1.0f/powf(v->theta,(float)(2*(j%(HD/4)))/(HD/2));
                    float angle=(j<HD/4?row:col)*inv,c=cosf(angle),s=sinf(angle);
                    float *qs=q+(size_t)t*HD,*ks=k+(size_t)t*HD,qa=qs[j],qb=qs[j+HD/2],ka=ks[j],kb=ks[j+HD/2];
                    qs[j]=qa*c-qb*s;qs[j+HD/2]=qb*c+qa*s;ks[j]=ka*c-kb*s;ks[j+HD/2]=kb*c+ka*s;
                }
            }
            for(int first=0;first<N;first+=tile) {
                int count=N-first<tile?N-first:tile;
                mm_t(att,q+(size_t)first*HD,k,count,HD,N);
                float scale=1.0f/sqrtf((float)HD);
                /* Long image rows lose low probabilities in a sequential F32
                 * sum. Only the normalizer accumulates in double. */
                for(int t=0;t<count;t++) {
                    float *r=att+(size_t)t*N,mx=-INFINITY;double sum=0;
                    for(int j=0;j<N;j++){r[j]*=scale;if(r[j]>mx)mx=r[j];}
                    for(int j=0;j<N;j++){r[j]=expf(r[j]-mx);sum+=r[j];}
                    for(int j=0;j<N;j++)r[j]/=(float)sum;
                }
                mm_t(ho,att,vt,count,N,HD);
                for(int t=0;t<count;t++)memcpy(a+(size_t)(first+t)*D+h*HD,ho+(size_t)t*HD,HD*sizeof(float));
            }
        }
        affine(tmp,a,&b->proj,N);for(size_t j=0;j<(size_t)N*D;j++)x[j]+=tmp[j];
        norm(xn,x,b->nw[1],b->nb[1],N,D);affine(ff,xn,&b->fc1,N);
        for(size_t j=0;j<(size_t)N*F;j++){float z=ff[j];ff[j]=0.5f*z*(1+tanhf(0.7978845608028654f*(z+0.044715f*z*z*z)));}
        affine(tmp,ff,&b->fc2,N);for(size_t j=0;j<(size_t)N*D;j++)x[j]+=tmp[j];
    }
    norm(xn,x,v->norm_w,v->norm_b,N,D);affine(merged,xn,&v->fc1,im->tokens);
    for(size_t j=0;j<(size_t)N*D;j++){float z=merged[j];merged[j]=0.5f*z*(1+erff(z*0.7071067811865475f));}
    affine(result,merged,&v->fc2,im->tokens);
    for(size_t j=0;j<(size_t)im->tokens*v->info.output_width;j++)if(!isfinite(result[j])){rc=NT_E_STATE;goto done;}
    *out=result;result=NULL;rc=NT_OK;
done:
    free(x);free(xn);free(qkv);free(a);free(tmp);free(ff);free(q);free(k);free(vt);free(att);free(ho);free(merged);free(result);return rc;
}
int nt_qwen35_image_positions(const nt_qwen35_vision *v,const nt_qwen35_image *im,const float *features,
    const int *ids,int n,int (*positions)[3],const float **rows) {
    if(!image_valid(v,im) || !features || !ids || !positions || !rows || n<im->tokens+2)return -1;
    const nt_qwen35_vision_info *i=&v->info;int start=-1;
    for(int t=0;t<n;t++)if(ids[t]==i->start_id){if(start>=0)return -1;start=t+1;}
    if(start<1 || start>n-im->tokens-1 || ids[start+im->tokens]!=i->end_id)return -1;
    for(int t=0;t<n;t++) {
        int visual=t>=start && t<start+im->tokens;
        if((ids[t]==i->image_id)!=visual || (ids[t]==i->end_id && t!=start+im->tokens))return -1;
    }
    int pos=0,H=im->grid_h/2,W=im->grid_w/2;
    for(int t=0;t<n;t++) {
        rows[t]=NULL;
        if(t==start) {
            for(int r=0;r<H;r++)for(int c=0;c<W;c++,t++) {
                positions[t][0]=pos;positions[t][1]=pos+r;positions[t][2]=pos+c;
                rows[t]=features+(size_t)(r*W+c)*i->output_width;
            }
            pos+=H>W?H:W;t--;
        } else {for(int a=0;a<3;a++)positions[t][a]=pos;pos++;}
    }
    return pos;
}
