/* Diagnostic SmolVLM / Idefics3 probe for staged parity. No Python in this path.
 * Usage: MMPROJ IMAGE LONGEST PREFIX [TEXT INSTRUCTION STEPS [FORCE.i32]]
 * LONGEST 0 takes the projector's preproc_image_size. Writes PREFIX.pixels.f32, .tower.f32
 * (every frame), .rows.f32; with TEXT also .ids.i32, .logits.f32 (the prefill distribution,
 * then one per step) and .json. FORCE feeds its ids at each step in place of the greedy
 * ones, which are recorded either way. Files are test artifacts only. */
#include "harness/smolvlm_vision.h"
#include "harness/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static FILE *output(const char *prefix,const char *suffix){char p[4096];if(snprintf(p,sizeof(p),"%s.%s",prefix,suffix)>=(int)sizeof(p))return NULL;return fopen(p,"wb");}
static int dump(const char *p,const char *s,const void *v,size_t n){FILE *f=output(p,s);if(!f)return 1;int bad=fwrite(v,1,n,f)!=n;return fclose(f)||bad;}
static int argmax(const float *x,int n){int b=0;for(int j=1;j<n;j++)if(x[j]>x[b])b=j;return b;}
int main(int argc,char **argv) {
    if(argc!=5 && argc!=8 && argc!=9){fprintf(stderr,"usage: %s MMPROJ IMAGE LONGEST PREFIX [TEXT INSTRUCTION STEPS [FORCE.i32]]\n",argv[0]);return 2;}
    gguf_file *vg=gguf_open(argv[1]),*tg=NULL;nt_smolvlm_vision *v=vg?nt_smolvlm_vision_load(vg):NULL;
    nt_smolvlm_image im={0};float *hidden=NULL,*features=NULL,*logits=NULL;uint8_t *rgb=NULL;
    int *ids=NULL,*force=NULL,*greedy=NULL,*fed=NULL,rc=1,n=0,V=0,steps=0;const float **rows=NULL;
    void *model=NULL;kv_cache *kv=NULL;bpe_tokenizer *tok=NULL;FILE *f=NULL,*lf=NULL;
    if(!v)goto done;
    const nt_smolvlm_vision_info *info=nt_smolvlm_vision_get_info(v);
    int h,w;rgb=nt_image_load(argv[2],&h,&w);
    if(!rgb || nt_smolvlm_image_prepare(v,rgb,h,w,atoi(argv[3]),&im))goto done;
    size_t fs=(size_t)3*info->image_size*info->image_size,hs=(size_t)info->patches*info->width,rs=(size_t)info->tokens*info->text_width;
    hidden=malloc((size_t)im.frames*hs*sizeof(float));features=malloc((size_t)im.frames*rs*sizeof(float));
    if(!hidden || !features)goto done;
    for(int k=0;k<im.frames;k++)
        if(nt_smolvlm_vision_tower(v,im.pixels+k*fs,hidden+k*hs) || nt_smolvlm_vision_connect(v,hidden+k*hs,features+k*rs))goto done;
    if(dump(argv[4],"pixels.f32",im.pixels,(size_t)im.frames*fs*sizeof(float)) ||
       dump(argv[4],"tower.f32",hidden,(size_t)im.frames*hs*sizeof(float)) ||
       dump(argv[4],"rows.f32",features,(size_t)im.frames*rs*sizeof(float)))goto done;
    if(argc>5) {
        steps=atoi(argv[7]);if(steps<0 || steps>512)goto done;
        tg=gguf_open(argv[5]);nt_dims dims={0};model=tg?nt_arch_llama.load(tg,&dims):NULL;tok=bpe_load(argv[5]);
        if(!model || !tok || tg->embed_dim!=info->text_width)goto done;
        int cap=1<<16;ids=malloc((size_t)cap*sizeof(int));
        if(!ids || (n=nt_smolvlm_prompt(v,&im,tok,argv[6],ids,cap))<0)goto done;
        rows=calloc((size_t)n,sizeof(*rows));
        if(!rows || nt_smolvlm_image_rows(v,&im,features,tok,ids,n,rows) || dump(argv[4],"ids.i32",ids,(size_t)n*sizeof(int)))goto done;
        if(argc==9) {
            FILE *ff=fopen(argv[8],"rb");force=malloc((size_t)(steps?steps:1)*sizeof(int));
            int ok=ff && force && fread(force,sizeof(int),(size_t)steps,ff)==(size_t)steps;
            if(ff)fclose(ff);
            if(!ok)goto done;
        }
        V=dims.vocab;logits=malloc((size_t)V*sizeof(float));greedy=calloc((size_t)steps+1,sizeof(int));fed=calloc((size_t)steps+1,sizeof(int));
        kv=nt_model_kv_new(&nt_arch_llama,model,dims,n+steps+1);
        if(!logits || !greedy || !fed || !kv || !(lf=output(argv[4],"logits.f32")))goto done;
        int err=nt_arch_llama.forward_embd(model,kv,ids,rows,n,0,logits,NULL);
        if(err){fprintf(stderr,"prefill: %s\n",nt_strerror(err));goto done;}
        if(fwrite(logits,sizeof(float),(size_t)V,lf)!=(size_t)V)goto done;
        for(int s=0;s<steps;s++) {
            greedy[s]=argmax(logits,V);fed[s]=force?force[s]:greedy[s];
            if(fed[s]<0 || fed[s]>=V)goto done;
            err=nt_arch_llama.forward(model,kv,&fed[s],1,n+s,logits);
            if(err || fwrite(logits,sizeof(float),(size_t)V,lf)!=(size_t)V)goto done;
        }
        if(fclose(lf)){lf=NULL;goto done;}lf=NULL;
    }
    f=output(argv[4],"json");if(!f)goto done;
    fprintf(f,"{\"frames\":%d,\"rows\":%d,\"cols\":%d,\"tokens_per_frame\":%d,\"text_width\":%d,\"input_tokens\":%d,\"vocab\":%d,\"forced\":%d,\"greedy\":[",
            im.frames,im.rows,im.cols,info->tokens,info->text_width,n,V,force!=NULL);
    for(int j=0;j<steps;j++)fprintf(f,j?",%d":"%d",greedy[j]);
    fputs("],\"fed\":[",f);
    for(int j=0;j<steps;j++)fprintf(f,j?",%d":"%d",fed[j]);
    fputs("]}\n",f);
    if(fclose(f)){f=NULL;goto done;}f=NULL;rc=0;
done:
    if(f)fclose(f);
    if(lf)fclose(lf);
    if(kv)nt_model_kv_free(&nt_arch_llama,model,kv);
    if(model)nt_arch_llama.free(model);
    bpe_free(tok);gguf_close(tg);free(ids);free(rows);free(force);free(greedy);free(fed);free(logits);free(rgb);
    free(hidden);free(features);nt_smolvlm_image_free(&im);nt_smolvlm_vision_free(v);gguf_close(vg);return rc;
}
