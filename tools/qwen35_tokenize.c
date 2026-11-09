/* Explicit chat-prompt tokenization; JSONL {id,prompt} -> id<TAB>IDs. Text the
 * pre-type refuses (non-ASCII for qwen35-ascii, the NFC quick check for qwen35)
 * encodes to nothing and fails the row. */
#include "examples/bpe.h"
#include "tools/narrow_json.h"
#include <ctype.h>
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"Usage: %s TOKENIZER.gguf < prompts.jsonl\n",argv[0]);return 2;}
    bpe_tokenizer *tok=bpe_load(argv[1]);if(!tok)return 1;
    char *line=NULL;size_t cap=0;ssize_t bytes;int rc=0,seen=0;
    while((bytes=getline(&line,&cap,stdin))>=0) {
        if(memchr(line,0,(size_t)bytes)){rc=1;break;}
        char *id=nj_string(nj_get(line,"id")),*prompt=nj_string(nj_get(line,"prompt"));
        if(!id || !*id || !prompt || !*prompt){free(id);free(prompt);rc=1;break;}
        for(char *p=id;*p;p++)if(!isalnum((unsigned char)*p) && *p!='-' && *p!='_')rc=1;
        size_t n=strlen(prompt);if(n>1048576)rc=1;
        int *ids=rc?NULL:malloc((n+1)*sizeof(int));
        int count=ids?bpe_encode_special(tok,prompt,ids,(int)n+1):0;
        if(count<1)rc=1;
        if(!rc){printf("%s\t",id);for(int i=0;i<count;i++)printf("%s%d",i?" ":"",ids[i]);puts("");seen++;}
        free(ids);free(id);free(prompt);if(rc)break;
    }
    if(ferror(stdin) || fflush(stdout) || !seen)rc=1;
    if(rc)fprintf(stderr,"Invalid prompt row; this text path requires nonempty text without NUL that the tokenizer accepts\n");
    free(line);bpe_free(tok);return rc;
}
