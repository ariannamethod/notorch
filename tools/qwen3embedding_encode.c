/* Native Qwen3-Embedding batch runner, the sibling of embeddinggemma2_encode.
 * Each TSV row is id<TAB>integer IDs<TAB>optional 0/1 mask, or, with --tokenizer
 * and --condition, id<TAB>text: the condition's prompt and the text are tokenized
 * in C, <|endoftext|> is appended, and the IDs used are written next to the
 * vectors. A row holding NUL in either mode is refused, as is a text holding TAB,
 * CR or LF, a sequence longer than the model's limit, or a text the tokenizer
 * refuses (tools/qwen3embedding_text.h).
 * Output JSONL: id, tokens, [input_ids], seconds, pooled (the final normalized
 * state of the last unmasked position) and embedding (its L2 normalization), the
 * keys embeddinggemma2_score reads. Trace files are little-endian host F32.
 * All outputs are create-only; an interrupted batch remains visibly partial. */
#include "harness/arch.h"
#include "tools/qwen3embedding_text.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct { const char *dir, *id; FILE *index; } runner;

static const char *site_name(nt_encode_site site) {
    static const char *names[]={"input_embeddings","ple","post_attention","post_ffn","block","final_norm","projected"};
    return names[site];
}
static int trace(void *user,nt_encode_site site,int layer,int n,int width,float *states) {
    runner *r=user;char path[4096];
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
        "       [--tokenizer TOKENIZER.gguf --condition NAME]  (rows: id<TAB>text)\nconditions:",prog);
    for(int i=0;i<QE_CONDITIONS;i++)fprintf(stderr," %s",qe_conditions[i][0]);
    fputc('\n',stderr);
}
int main(int argc,char **argv) {
    if(argc<4){usage(argv[0]);return 2;}
    uint16_t endian=1;if(*(unsigned char*)&endian!=1){fprintf(stderr,"little-endian host required\n");return 2;}
    runner r={0};
    const char *tokenizer_path=NULL,*condition=NULL,*prefix=NULL;
    for(int i=4;i<argc;i++) {
        if(i+1>=argc){usage(argv[0]);return 2;}
        const char *key=argv[i],*value=argv[++i];
        if(!strcmp(key,"--trace"))r.dir=value;
        else if(!strcmp(key,"--tokenizer"))tokenizer_path=value;
        else if(!strcmp(key,"--condition")){condition=value;if(!(prefix=qe_prefix(value))){usage(argv[0]);return 2;}}
        else{usage(argv[0]);return 2;}
    }
    if((tokenizer_path!=NULL)!=(condition!=NULL)){fprintf(stderr,"--tokenizer and --condition must be paired\n");return 2;}
    int status=1;gguf_file *gf=NULL;void *model=NULL;FILE *in=NULL,*out=NULL;qe_tokenizer tok={0};
    char *line=NULL;size_t line_cap=0;int *ids=NULL,*masks=NULL;unsigned char *mask=NULL;
    float *pooled=NULL,*embedding=NULL;const unsigned char *use_mask;
    gf=gguf_open(argv[1]);if(!gf)goto done;
    nt_encode_dims d;model=nt_encoder_qwen3embedding.load(gf,&d);if(!model)goto done;
    if(tokenizer_path) {
        if(qe_tokenizer_load(&tok,tokenizer_path))goto done;
        if(bpe_n_vocab(tok.bpe)!=d.vocab){fprintf(stderr,"tokenizer vocabulary %d differs from the model's %d\n",bpe_n_vocab(tok.bpe),d.vocab);goto done;}
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
        row++;
        if(memchr(line,0,(size_t)got)){fprintf(stderr,"row %u: NUL byte inside the row, refused\n",row);goto done;}
        char *tab=strchr(line,'\t');
        if(!tab){fprintf(stderr,"row %u: expected id TAB IDs [TAB mask]\n",row);goto done;}
        *tab++=0;r.id=line;
        if(!safe_id(r.id)){fprintf(stderr,"row %u: invalid ID\n",row);goto done;}
        use_mask=NULL;int n;
        if(tok.bpe) {
            size_t len=strlen(tab);if(len && tab[len-1]=='\n')tab[--len]=0;
            if(strpbrk(tab,"\t\r\n")){fprintf(stderr,"row %u: TAB, CR or LF inside the text\n",row);goto done;}
            const char *literal=NULL;
            n=qe_text_ids(&tok,prefix,tab,ids,d.max_tokens,&literal);
            if(n==-4){fprintf(stderr,"row %u: holds the added-token literal \"%s\", refused\n",row,literal);goto done;}
            if(n<0){fprintf(stderr,"row %u: %s\n",row,n==-1?"longer than the model's token limit, refused":
                n==-2?"the IDs do not decode back to this text (invalid UTF-8, or text NFC would change), refused":
                n==-5?"begins or ends with whitespace, which the two official paths treat differently, refused":"out of memory");goto done;}
        } else {
            char *mask_text=strchr(tab,'\t');if(mask_text)*mask_text++=0;
            n=parse_ints(tab,ids,d.max_tokens,d.vocab-1);
            if(n<1){fprintf(stderr,"row %u: invalid token IDs\n",row);goto done;}
            if(mask_text) {
                if(parse_ints(mask_text,masks,d.max_tokens,1)!=n){fprintf(stderr,"row %u: invalid mask\n",row);goto done;}
                for(int j=0;j<n;j++)mask[j]=(unsigned char)masks[j];
                use_mask=mask;
            }
        }
        double start=seconds();
        int rc=nt_encoder_qwen3embedding.encode(model,ids,use_mask,n,pooled,embedding,r.dir?trace:NULL,&r);
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
    free(line);free(ids);free(masks);free(mask);free(pooled);free(embedding);
    if(in)fclose(in);
    if(out && fclose(out))status=1;
    if(r.index && fclose(r.index))status=1;
    if(model)nt_encoder_qwen3embedding.free(model);
    if(gf)gguf_close(gf);
    qe_tokenizer_free(&tok);
    return status;
}
