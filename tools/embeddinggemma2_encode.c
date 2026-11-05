/* Native text-encoder batch runner. Each TSV row is id<TAB>integer IDs<TAB>
 * optional 0/1 mask, or, with --tokenizer and --condition, id<TAB>text: the
 * condition's task prefix and the text are tokenized in C and the IDs used are
 * written next to the vectors. A text holding TAB, CR, LF or NUL is refused, as is a
 * sequence longer than the model's limit or one the tokenizer cannot carry whole.
 * Output JSONL includes both pooled and normalized vectors. Optional trace files
 * are little-endian host F32 (this runner refuses big-endian hosts).
 * All outputs are create-only; an interrupted batch remains visibly partial. */
#include "harness/arch.h"
#include "tools/embeddinggemma2_text.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    const char *dir, *id;
    FILE *index;
    const unsigned char *mask;
    float *offset, scale;
    int layer, nonpad;
} runner;

static const char *site_name(nt_encode_site site) {
    static const char *names[]={"input_embeddings","ple","post_attention","post_ffn","block","final_norm","projected"};
    return names[site];
}
static int hook(void *user,nt_encode_site site,int layer,int n,int width,float *states) {
    runner *r=user;
    if(r->offset && site==NT_ENCODE_BLOCK && layer==r->layer)
        for(int t=0;t<n;t++)if(!r->nonpad || !r->mask || r->mask[t])
            for(int j=0;j<width;j++)states[(size_t)t*width+j]+=r->scale*r->offset[j];
    if(!r->dir)return NT_OK;
    char path[4096];
    if(snprintf(path,sizeof(path),"%s/%s.%s.%d.f32",r->dir,r->id,site_name(site),layer)>=(int)sizeof(path))return NT_E_ARG;
    FILE *f=fopen(path,"wbx");if(!f){perror(path);return NT_E_STATE;}
    size_t count=(size_t)n*width;
    int bad=fwrite(states,sizeof(float),count,f)!=count;
    bad|=fclose(f)!=0;
    if(bad || fprintf(r->index,"%s\t%s\t%d\t%d\t%d\t%s\n",r->id,site_name(site),layer,n,width,path)<0)return NT_E_STATE;
    return NT_OK;
}
static double seconds(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+t.tv_nsec*1e-9;
}
static int parse_ints(char *s,int *out,int cap,int maximum) {
    int n=0;
    while(*s) {
        while(isspace((unsigned char)*s))s++;
        if(!*s)break;
        char *end;errno=0;long v=strtol(s,&end,10);
        if(end==s || errno || v<0 || v>maximum || n==cap || (*end && !isspace((unsigned char)*end)))return -1;
        out[n++]=(int)v;s=end;
    }
    return n;
}
static int safe_id(const char *s) {
    if(!*s || strlen(s)>128)return 0;
    for(;*s;s++)if(!isalnum((unsigned char)*s) && *s!='_' && *s!='-' && *s!='.')return 0;
    return 1;
}
static void array(FILE *f,const char *key,const float *values,int n) {
    fprintf(f,"\"%s\":[",key);
    for(int i=0;i<n;i++)fprintf(f,"%s%.9g",i?",":"",values[i]);
    fputc(']',f);
}
static void usage(const char *prog) {
    fprintf(stderr,"usage: %s MODEL.gguf TOKENS.tsv RESULTS.jsonl [--trace DIR]\n"
        "       [--offset vector.f32 --layer N --scale ALPHA] [--nonpad]\n"
        "       [--tokenizer TOKENIZER.gguf --condition NAME]  (rows: id<TAB>text)\n"
        "offset applies after complete block N, before the next block/final norm;\n"
        "all tokens by default, only unmasked tokens with --nonpad.\nconditions:",prog);
    for(int i=0;i<EG2_CONDITIONS;i++)fprintf(stderr," %s",eg2_conditions[i][0]);
    fputc('\n',stderr);
}
int main(int argc,char **argv) {
    if(argc<4){usage(argv[0]);return 2;}
    uint16_t endian=1;if(*(unsigned char*)&endian!=1){fprintf(stderr,"little-endian host required\n");return 2;}
    runner r={0};r.layer=-1;r.scale=1;
    const char *offset_path=NULL,*tokenizer_path=NULL,*condition=NULL,*prefix=NULL;
    for(int i=4;i<argc;i++) {
        if(!strcmp(argv[i],"--nonpad")){r.nonpad=1;continue;}
        if(i+1>=argc){usage(argv[0]);return 2;}
        const char *key=argv[i],*value=argv[++i];char *end;
        if(!strcmp(key,"--trace"))r.dir=value;
        else if(!strcmp(key,"--offset"))offset_path=value;
        else if(!strcmp(key,"--tokenizer"))tokenizer_path=value;
        else if(!strcmp(key,"--condition")){condition=value;if(!(prefix=eg2_prefix(value))){usage(argv[0]);return 2;}}
        else if(!strcmp(key,"--layer")){errno=0;long v=strtol(value,&end,10);if(errno || *end || v<0 || v>127)return 2;r.layer=(int)v;}
        else if(!strcmp(key,"--scale")){errno=0;r.scale=strtof(value,&end);if(errno || *end || !isfinite(r.scale))return 2;}
        else{usage(argv[0]);return 2;}
    }
    if((offset_path!=NULL)!=(r.layer>=0)){fprintf(stderr,"--offset and --layer must be paired\n");return 2;}
    if((tokenizer_path!=NULL)!=(condition!=NULL)){fprintf(stderr,"--tokenizer and --condition must be paired\n");return 2;}
    int status=1;gguf_file *gf=NULL;void *model=NULL;FILE *in=NULL,*out=NULL;eg2_tokenizer tok={0};
    char *line=NULL;size_t line_cap=0;int *ids=NULL,*masks=NULL;unsigned char *mask=NULL;
    float *pooled=NULL,*embedding=NULL;
    gf=gguf_open(argv[1]);if(!gf)goto done;
    nt_encode_dims d;model=nt_encoder_embeddinggemma2.load(gf,&d);if(!model)goto done;
    if(r.layer>=d.n_layers){fprintf(stderr,"layer out of bounds\n");goto done;}
    if(tokenizer_path) {
        if(eg2_tokenizer_load(&tok,tokenizer_path))goto done;
        if(bpe_n_vocab(tok.bpe)!=d.vocab){fprintf(stderr,"tokenizer vocabulary %d differs from the model's %d\n",bpe_n_vocab(tok.bpe),d.vocab);goto done;}
    }
    if(offset_path) {
        r.offset=malloc((size_t)d.width*sizeof(float));if(!r.offset)goto done;
        FILE *f=fopen(offset_path,"rb");if(!f){perror(offset_path);goto done;}
        int ok=fread(r.offset,sizeof(float),d.width,f)==(size_t)d.width && fgetc(f)==EOF;
        fclose(f);if(!ok){fprintf(stderr,"offset must contain exactly %d F32 values\n",d.width);goto done;}
        for(int j=0;j<d.width;j++)if(!isfinite(r.offset[j])){fprintf(stderr,"nonfinite offset\n");goto done;}
    }
    if(r.dir) {
        if(mkdir(r.dir,0777)){perror(r.dir);goto done;}
        char path[4096];if(snprintf(path,sizeof(path),"%s/index.tsv",r.dir)>=(int)sizeof(path))goto done;
        r.index=fopen(path,"wx");if(!r.index){perror(path);goto done;}
        fprintf(r.index,"id\tsite\tlayer\ttokens\twidth\tpath\n");
    }
    in=fopen(argv[2],"r");if(!in){perror(argv[2]);goto done;}
    out=fopen(argv[3],"wx");if(!out){perror(argv[3]);goto done;}
    ids=malloc((size_t)d.max_tokens*sizeof(int));masks=malloc((size_t)d.max_tokens*sizeof(int));
    mask=malloc((size_t)d.max_tokens);pooled=malloc((size_t)d.output_dim*sizeof(float));embedding=malloc((size_t)d.output_dim*sizeof(float));
    if(!ids || !masks || !mask || !pooled || !embedding)goto done;
    unsigned row=0;
    ssize_t got;
    while((got=getline(&line,&line_cap,in))>=0) {
        row++;char *tab=strchr(line,'\t');
        if(!tab){fprintf(stderr,"row %u: expected id TAB IDs [TAB mask]\n",row);goto done;}
        *tab++=0;r.id=line;
        if(!safe_id(r.id)){fprintf(stderr,"row %u: invalid ID\n",row);goto done;}
        r.mask=NULL;int n;
        if(tok.bpe) {
            if(memchr(tab,0,(size_t)(line+got-tab))){fprintf(stderr,"row %u: NUL byte inside the text, refused\n",row);goto done;}
            size_t len=strlen(tab);if(len && tab[len-1]=='\n')tab[--len]=0;
            if(strpbrk(tab,"\t\r\n")){fprintf(stderr,"row %u: TAB, CR or LF inside the text\n",row);goto done;}
            const char *literal=NULL;
            n=eg2_text_ids(&tok,prefix,tab,ids,d.max_tokens,&literal);
            if(n==-4){fprintf(stderr,"row %u: holds the added-token literal \"%s\", refused\n",row,literal);goto done;}
            if(n<0){fprintf(stderr,"row %u: %s\n",row,n==-1?"longer than the model's token limit, refused":
                n==-2?"the tokenizer cannot carry this text whole":"out of memory");goto done;}
        } else {
            char *mask_text=strchr(tab,'\t');if(mask_text)*mask_text++=0;
            n=parse_ints(tab,ids,d.max_tokens,d.vocab-1);
            if(n<1){fprintf(stderr,"row %u: invalid token IDs\n",row);goto done;}
            if(mask_text) {
                if(parse_ints(mask_text,masks,d.max_tokens,1)!=n){fprintf(stderr,"row %u: invalid mask\n",row);goto done;}
                for(int j=0;j<n;j++)mask[j]=(unsigned char)masks[j];
                r.mask=mask;
            }
        }
        double start=seconds();
        int rc=nt_encoder_embeddinggemma2.encode(model,ids,r.mask,n,pooled,embedding,(r.dir||r.offset)?hook:NULL,&r);
        double elapsed=seconds()-start;
        if(rc){fprintf(stderr,"row %u (%s): encode failed %d\n",row,r.id,rc);goto done;}
        fprintf(out,"{\"id\":\"%s\",\"tokens\":%d,",r.id,n);
        if(tok.bpe){fputs("\"input_ids\":[",out);for(int j=0;j<n;j++)fprintf(out,"%s%d",j?",":"",ids[j]);fputs("],",out);}
        fprintf(out,"\"seconds\":%.9g,",elapsed);
        array(out,"pooled",pooled,d.output_dim);fputc(',',out);array(out,"embedding",embedding,d.output_dim);fputs("}\n",out);
        if(fflush(out) || (r.index && fflush(r.index))){perror("output");goto done;}
        fprintf(stderr,"%s\t%d tokens\t%.3f s\n",r.id,n,elapsed);
    }
    if(ferror(in)){perror(argv[2]);goto done;}
    status=0;
done:
    free(line);free(ids);free(masks);free(mask);free(pooled);free(embedding);free(r.offset);
    if(in)fclose(in);
    if(out && fclose(out))status=1;
    if(r.index && fclose(r.index))status=1;
    if(model)nt_encoder_embeddinggemma2.free(model);
    if(gf)gguf_close(gf);
    eg2_tokenizer_free(&tok);
    return status;
}
