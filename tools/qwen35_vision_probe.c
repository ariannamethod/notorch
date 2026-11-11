/* Diagnostic native encoder / full decoder probe. No Python in this path.
 * Usage: VISION IMAGE MAX_PIXELS PREFIX [TEXT IDS.i32 CHUNK STEPS]
 * IMAGE '-' runs the unchanged text path. Files are test artifacts only. */
#include "harness/qwen35_vision.h"
#include "harness/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static FILE *output(const char *prefix,const char *suffix){char p[4096];if(snprintf(p,sizeof(p),"%s.%s",prefix,suffix)>=(int)sizeof(p))return NULL;return fopen(p,"wb");}
static int dump(const char *p,const char *s,const void *v,size_t n){FILE *f=output(p,s);if(!f)return 1;int bad=fwrite(v,1,n,f)!=n;return fclose(f)||bad;}
int main(int argc,char **argv) {
    if(argc!=5 && argc!=9){fprintf(stderr,"usage: %s VISION IMAGE MAX_PIXELS PREFIX [TEXT IDS.i32 CHUNK STEPS]\n",argv[0]);return 2;}
    gguf_file *vg=gguf_open(argv[1]),*tg=NULL;nt_qwen35_vision *v=vg?nt_qwen35_vision_load(vg):NULL;
    nt_qwen35_image im={0};float *features=NULL,*logits=NULL;uint8_t *rgb=NULL;int *ids=NULL,(*pos)[3]=NULL,rc=1;
    const float **rows=NULL;void *model=NULL;kv_cache *cache=NULL;FILE *f=NULL,*lf=NULL;
    if(!v)goto done;
    const nt_qwen35_vision_info *info=nt_qwen35_vision_get_info(v);
    if(strcmp(argv[2],"-")) {
        int h,w;rgb=nt_image_load(argv[2],&h,&w);
        if(!rgb || nt_qwen35_image_prepare(v,rgb,h,w,atoi(argv[3]),&im) || nt_qwen35_vision_encode(v,&im,&features))goto done;
        if(dump(argv[4],"patches.f32",im.patches,(size_t)im.grid_h*im.grid_w*6*info->patch*info->patch*sizeof(float)) ||
           dump(argv[4],"features.f32",features,(size_t)im.tokens*info->output_width*sizeof(float)))goto done;
    }
    int n=0,V=0,next=0,steps=0,greedy[64];
    if(argc==9) {
        int chunk=atoi(argv[7]);steps=atoi(argv[8]);
        if(chunk<1 || chunk>8192 || steps<0 || steps>64)goto done;
        f=fopen(argv[6],"rb");if(!f || fseek(f,0,SEEK_END))goto done;
        long size=ftell(f);if(size<=0 || size%4 || size/4>8192-steps || fseek(f,0,SEEK_SET))goto done;n=(int)(size/4);
        ids=malloc(size);pos=malloc((size_t)n*sizeof(*pos));rows=calloc(n,sizeof(*rows));
        if(!ids || !pos || !rows || fread(ids,1,size,f)!=(size_t)size)goto done;
        fclose(f);f=NULL;
        if(features){next=nt_qwen35_image_positions(v,&im,features,ids,n,pos,rows);if(next<0)goto done;}
        tg=gguf_open(argv[5]);nt_dims dims={0};model=tg?nt_arch_qwen35.load(tg,&dims):NULL;
        if(!model || tg->embed_dim!=info->output_width)goto done;
        V=dims.vocab;logits=malloc((size_t)V*sizeof(float));cache=nt_arch_qwen35.kv_new(model,n+steps);
        if(!logits || !cache || !(lf=output(argv[4],"logits.f32")))goto done;
        for(int off=0;off<n;off+=chunk) {
            int take=n-off<chunk?n-off:chunk;
            int err=features?nt_qwen35_forward_3d(model,cache,ids+off,rows+off,(const int(*)[3])(pos+off),info->section,take,off,off+take==n?logits:NULL,NULL,NULL):
                nt_arch_qwen35.forward(model,cache,ids+off,take,off,off+take==n?logits:NULL);
            if(err){fprintf(stderr,"forward: %s\n",nt_strerror(err));goto done;}
        }
        if(fwrite(logits,sizeof(float),V,lf)!=(size_t)V)goto done;
        for(int s=0;s<steps;s++) {
            int id=0;for(int j=1;j<V;j++)if(logits[j]>logits[id])id=j;greedy[s]=id;
            int p[1][3]={{next+s,next+s,next+s}};
            int err=features?nt_qwen35_forward_3d(model,cache,&id,NULL,(const int(*)[3])p,info->section,1,n+s,logits,NULL,NULL):
                nt_arch_qwen35.forward(model,cache,&id,1,n+s,logits);
            if(err || fwrite(logits,sizeof(float),V,lf)!=(size_t)V)goto done;
        }
        if(fclose(lf)){lf=NULL;goto done;}lf=NULL;
    }
    f=output(argv[4],"json");if(!f)goto done;
    fprintf(f,"{\"height\":%d,\"width\":%d,\"grid_h\":%d,\"grid_w\":%d,\"image_tokens\":%d,\"feature_width\":%d,\"input_tokens\":%d,\"vocab\":%d,\"next_position\":%d,\"greedy\":[",im.height,im.width,im.grid_h,im.grid_w,im.tokens,info->output_width,n,V,next);
    for(int j=0;j<steps;j++)fprintf(f,j?",%d":"%d",greedy[j]);
    fputs("]}\n",f);
    if(fclose(f)){f=NULL;goto done;}f=NULL;rc=0;
done:
    if(f)fclose(f);
    if(lf)fclose(lf);
    if(cache)nt_arch_qwen35.kv_free(model,cache);
    if(model)nt_arch_qwen35.free(model);
    gguf_close(tg);free(ids);free(pos);free(rows);free(logits);free(rgb);free(features);
    nt_qwen35_image_free(&im);nt_qwen35_vision_free(v);gguf_close(vg);return rc;
}
