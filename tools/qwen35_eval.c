/* Diagnostic text-ID runner: id<TAB>space-separated IDs on stdin.
 * Output: final A/B logits, their softmax and the supplied IDs. Optional
 * residual/full-logit traces support independent numerical comparison. */
#include "harness/arch.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {const char *dir,*id;} trace_context;
static int trace(void *ptr,int layer,int pos,int n,int width,float *x) {
    trace_context *t=ptr;char path[4096];
    if(snprintf(path,sizeof(path),"%s/%s.layer%02d.f32",t->dir,t->id,layer)>=(int)sizeof(path))return NT_E_ARG;
    FILE *f=fopen(path,pos?"ab":"wb");if(!f)return NT_E_STATE;
    int ok=fwrite(x,sizeof(float),(size_t)n*width,f)==(size_t)n*width;
    if(fclose(f))ok=0;
    return ok?NT_OK:NT_E_STATE;
}
static int positive(const char *s) {
    char *end;errno=0;long n=strtol(s,&end,10);
    return errno || !*s || *end || n<1 || n>8192 ? -1:(int)n;
}
int main(int argc,char **argv) {
    if(argc<2){fprintf(stderr,"Usage: %s MODEL [--chunk N] [--trace DIR]\n",argv[0]);return 2;}
    int chunk=8192;const char *dir=NULL;
    for(int i=2;i<argc;i++) {
        if(!strcmp(argv[i],"--chunk") && i+1<argc)chunk=positive(argv[++i]);
        else if(!strcmp(argv[i],"--trace") && i+1<argc)dir=argv[++i];
        else {fprintf(stderr,"Invalid option\n");return 2;}
    }
    if(chunk<1)return 2;
    gguf_file *g=gguf_open(argv[1]);nt_dims dims={0};
    void *m=g?nt_arch_qwen35.load(g,&dims):NULL;
    if(!m){gguf_close(g);return 1;}
    if(dims.vocab<=33){fprintf(stderr,"A/B output slots are absent\n");nt_arch_qwen35.free(m);gguf_close(g);return 1;}
    float *logits=malloc((size_t)dims.vocab*sizeof(float));char *line=NULL;size_t cap=0;
    int rc=0,seen=0;ssize_t bytes;
    if(!logits){rc=1;goto done;}
    while((bytes=getline(&line,&cap,stdin))>=0) {
        if(memchr(line,0,(size_t)bytes)){fprintf(stderr,"NUL inside input row\n");rc=1;break;}
        char *tab=strchr(line,'\t');
        if(!tab || tab==line){fprintf(stderr,"Expected id<TAB>token IDs\n");rc=1;break;}
        *tab=0;
        for(char *p=line;*p;p++)if(!isalnum((unsigned char)*p) && *p!='-' && *p!='_'){rc=1;break;}
        if(rc){fprintf(stderr,"Invalid case ID\n");break;}
        int ids[8192],n=0;char *p=tab+1;
        while(*p) {
            while(isspace((unsigned char)*p))p++;
            if(!*p)break;
            char *end;errno=0;long v=strtol(p,&end,10);
            if(end==p || errno || v<0 || v>=dims.vocab || n==8192 || (*end && !isspace((unsigned char)*end))){rc=1;break;}
            ids[n++]=(int)v;p=end;
        }
        if(rc || !n){fprintf(stderr,"Invalid or oversized token row\n");rc=1;break;}
        kv_cache *kv=nt_arch_qwen35.kv_new(m,n);
        if(!kv){rc=1;break;}
        trace_context tc={dir,line};
        for(int pos=0;pos<n;pos+=chunk) {
            int count=n-pos<chunk?n-pos:chunk;
            int status=nt_arch_qwen35.forward_residual(m,kv,ids+pos,count,pos,pos+count==n?logits:NULL,dir?trace:NULL,&tc);
            if(status){fprintf(stderr,"Forward: %s\n",nt_strerror(status));rc=1;break;}
        }
        nt_arch_qwen35.kv_free(m,kv);if(rc)break;
        if(dir) {
            char path[4096];
            if(snprintf(path,sizeof(path),"%s/%s.logits.f32",dir,line)>=(int)sizeof(path)){rc=1;break;}
            FILE *f=fopen(path,"wb");if(!f){rc=1;break;}
            if(fwrite(logits,sizeof(float),dims.vocab,f)!=(size_t)dims.vocab)rc=1;
            if(fclose(f))rc=1;
            if(rc)break;
        }
        double prob=1/(1+exp((double)logits[33]-logits[32]));
        printf("{\"id\":\"%s\",\"slot_logits\":[%.9g,%.9g],\"probabilities\":[%.17g,%.17g],\"input_ids\":[",line,logits[32],logits[33],prob,1-prob);
        for(int i=0;i<n;i++)printf("%s%d",i?",":"",ids[i]);
        puts("]}");if(fflush(stdout)){rc=1;break;}seen++;
    }
    if(ferror(stdin) || !seen)rc=1;
done:
    free(line);free(logits);nt_arch_qwen35.free(m);gguf_close(g);return rc;
}
