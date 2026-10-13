/* Exercise the actual CLI emitter, including allocation and stdout writing.
 * Optional argv[1] adds the ten long surfaces from either published PEN7 GGUF. */
#define main notorch_cli_main
#include "../harness/main.c"
#undef main
#include "tools/pen7_bpe.h"
#include <unistd.h>

static int checks, failed;
#define CHECK(c,s) do { checks++; if (!(c)) { fprintf(stderr,"FAIL: %s\n",s); failed++; } } while (0)
static void expect_emit(const session *s, int id, const void *expected, size_t len) {
    FILE *capture = tmpfile(); int saved = dup(STDOUT_FILENO);
    if (!capture || saved < 0) { CHECK(0,"capture allocation"); if(capture)fclose(capture);if(saved>=0)close(saved);return; }
    fflush(stdout);
    if (dup2(fileno(capture),STDOUT_FILENO)<0) { CHECK(0,"capture redirect");close(saved);fclose(capture);return; }
    int ok = emit(s,id); fflush(stdout);
    int restored = dup2(saved,STDOUT_FILENO); close(saved);
    CHECK(restored>=0,"restore stdout"); CHECK(ok,"complete token emission succeeds");
    rewind(capture); unsigned char *bytes = malloc(len+1);
    if (!bytes) {CHECK(0,"capture buffer");fclose(capture);return;}
    size_t n=fread(bytes,1,len+1,capture);
    CHECK(n==len && !memcmp(bytes,expected,len),"actual emitter preserves every decoded byte");
    free(bytes);fclose(capture);
}
static int fixture(const char *path, char *long_utf8) {
    const char *tokens[]={"x",long_utf8,"A\xc4\x80" "B","\xc4\xa0" "fire"};
    int32_t types[]={1,4,1,1};
    gguf_writer *w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"general.architecture","llama");
    gguf_write_kv_str(w,"tokenizer.ggml.model","gpt2");
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,4);
    gguf_write_kv_i32_array(w,"tokenizer.ggml.token_type",types,4);
    gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",0);
    return gguf_write_close(w);
}
static void actual_pen7(const char *path) {
    bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load actual PEN7 tokenizer");if(!t)return;
    session s={.tok=t};unsigned char *raw[2048]={0};size_t sizes[2048]={0},max=0;int count=0;
    for(int id=0;id<2048;id++){
        if(id<256){sizes[id]=1;raw[id]=malloc(1);if(raw[id])raw[id][0]=(unsigned char)id;}
        else {int left=pen7_merge_pairs[2*(id-256)],right=pen7_merge_pairs[2*(id-256)+1];
            sizes[id]=sizes[left]+sizes[right];raw[id]=malloc(sizes[id]);
            if(raw[id]){memcpy(raw[id],raw[left],sizes[left]);memcpy(raw[id]+sizes[left],raw[right],sizes[right]);}}
        if(!raw[id]){CHECK(0,"oracle surface allocation");break;}
        if(sizes[id]>255){count++;expect_emit(&s,id,raw[id],sizes[id]);}
        if(sizes[id]>max)max=sizes[id];
    }
    CHECK(count==10 && max==394 && sizes[1750]==394,"all ten native long surfaces including ID1750");
    for(int id=0;id<2048;id++)free(raw[id]);
    bpe_free(t);
}
int main(int argc,char **argv) {
    char path[]="/tmp/nt_emit_XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    char long_utf8[901];for(int i=0;i<150;i++)memcpy(long_utf8+6*i,"\xe4\xb8\x96\xe7\x95\x8c",6);long_utf8[900]=0;
    CHECK(fixture(path,long_utf8)==0,"write long UTF8 tokenizer fixture");
    bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load long UTF8 tokenizer");
    if(t){session s={.tok=t};
        CHECK(bpe_decode_token_capacity(t,1)>=901,"complete capacity exceeds old stack bound");
        CHECK(bpe_decode_token_capacity(t,-1)==0&&bpe_decode_token_capacity(t,4)==0,"invalid IDs have no capacity");
        expect_emit(&s,0,"x",1);expect_emit(&s,1,long_utf8,900);
        const unsigned char zero[]={65,0,66};expect_emit(&s,2,zero,3);
        expect_emit(&s,3," fire",5);bpe_free(t);
    }
    session bytes={0};expect_emit(&bytes,65,"A",1);expect_emit(&bytes,10,"\n",1);expect_emit(&bytes,1,"[1]",3);
    if(argc>1)actual_pen7(argv[1]);
    unlink(path);
    printf("HARNESS_EMIT_%s (%d checks)\n",failed?"FAIL":"OK",checks);return !!failed;
}
