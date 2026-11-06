#include "examples/bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int failed,checks;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
static void check(bpe_tokenizer *t,const char *s,const int *expected,int count) {
    int ids[100];int n=bpe_encode(t,s,ids,100);
    CHECK(n==count && !memcmp(ids,expected,(size_t)count*sizeof(int)),s);
}
int main(void) {
    char path[]="/tmp/notorch-qwen35-text-XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    char words[271][32]={{0}};const char *tokens[271];int32_t types[271];int next=0;
    for(int b=0;b<256;b++) {
        int literal=(b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
        int cp=literal?b:256+next++;
        if(cp<128)words[b][0]=(char)cp;
        else{words[b][0]=(char)(0xc0|(cp>>6));words[b][1]=(char)(0x80|(cp&63));}
    }
    const char *merges[]={"1 2","a 1","a !","! a","' s","s a","Ġ a","a Ċ","Ġ Ċ","Ċ Ċ","' R","'R e","Ġ Ġ","ĠĠ a"};
    const char *joined[]={"12","a1","a!","!a","'s","sa","Ġa","aĊ","ĠĊ","ĊĊ","'R","'Re","ĠĠ","ĠĠa"};
    for(int i=0;i<14;i++)strcpy(words[256+i],joined[i]);
    strcpy(words[270],"<|im_start|>");
    for(int i=0;i<271;i++){tokens[i]=words[i];types[i]=i==270?3:1;}
    gguf_writer *w=gguf_write_open(path);if(!w)return 1;
    gguf_write_kv_str(w,"tokenizer.ggml.pre","qwen35-ascii");
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,271);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,14);
    gguf_write_kv_i32_array(w,"tokenizer.ggml.token_type",types,271);
    CHECK(!gguf_write_close(w),"fixture write");bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load");
    if(!t){unlink(path);return 1;}
    check(t,"12",(int[]){49,50},2);check(t,"a1",(int[]){97,49},2);check(t,"a!",(int[]){97,33},2);
    check(t,"!a",(int[]){259},1);check(t,"'sa",(int[]){260,97},2);check(t,"'Re",(int[]){267},1);
    check(t,"  a",(int[]){32,262},2);check(t," a\n",(int[]){262,10},2);
    check(t," \n",(int[]){264},1);check(t,"!\n\n",(int[]){33,265},2);
    int ids[100];CHECK(bpe_encode_special(t,"<|im_start|>",ids,100)==1 && ids[0]==270,"explicit control token");
    CHECK(bpe_encode(t,"<|im_start|>",ids,100)>1,"ordinary path stays literal");
    ids[0]=1234;CHECK(bpe_encode_special(t,"caf\xc3\xa9",ids,100)==0 && ids[0]==1234,"NFC boundary refused before output");
    bpe_free(t);unlink(path);printf("Qwen3.5 text: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
