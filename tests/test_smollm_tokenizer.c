/* Synthetic byte vocabulary with deliberately tempting cross-number merges.
 * No downloads: checks the segmentation, rather than relying on a real vocab
 * accidentally lacking the forbidden merge. */
#include "examples/bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int checks, failed;
#define CHECK(c,s) do { checks++; if (!(c)) { fprintf(stderr,"FAIL: %s\n",s); failed++; } } while (0)
static int enc(int cp, char *s) {
    if(cp<128){s[0]=(char)cp;s[1]=0;return 1;}
    s[0]=(char)(0xC0|(cp>>6));s[1]=(char)(0x80|(cp&63));s[2]=0;return 2;
}
static int fixture(const char *path, const char *pre) {
    char storage[512][16], ranks[256][24], tab_word[24], tab_rank[24], tab_def[24];
    const char *tokens[519],*merges[262];
    int n=0;
    for(int b=0;b<256;b++) {
        int printable=(b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174&&b<=255);
        enc(printable?b:256+n,storage[b]);if(!printable)n++;
        tokens[b]=storage[b];
    }
    /* Every possible first UTF-8 byte can merge with a space. */
    for(int b=0;b<256;b++) {
        snprintf(storage[256+b],16,"%s%s",storage[32],storage[b]);tokens[256+b]=storage[256+b];
        snprintf(ranks[b],24,"%s %s",storage[32],storage[b]);merges[b]=ranks[b];
    }
    tokens[512]="12";merges[256]="1 2";
    snprintf(tab_word,sizeof(tab_word),"%sx",storage[9]);tokens[513]=tab_word;
    snprintf(tab_rank,sizeof(tab_rank),"%s x",storage[9]);merges[257]=tab_rank;
    /* Digit merges that cross a group of three, and a whole tab-word with no merge to it. */
    tokens[514]="14";tokens[515]="141";tokens[516]="1415";tokens[517]="59";
    merges[258]="1 4";merges[259]="14 1";merges[260]="141 5";merges[261]="5 9";
    snprintf(tab_def,sizeof(tab_def),"%sdef",storage[9]);tokens[518]=tab_def;
    gguf_writer *w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"tokenizer.ggml.model","gpt2");
    gguf_write_kv_str(w,"tokenizer.ggml.pre",pre);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,519);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,262);
    gguf_write_kv_u32(w,"tokenizer.ggml.add_bos_token",0);
    return gguf_write_close(w);
}
int main(void) {
    char path[]="/tmp/notorch-smollm-XXXXXX";int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    const char *numbers[]={"0","٠","१","９","Ⅷ","²","½","𐄇","𝟡","𞥐"};
    const char *pre[]={"smollm","qwen2","gpt2"};
    for(int family=0;family<3;family++) {
        CHECK(fixture(path,pre[family])==0,"write fixture");
        bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load fixture");if(!t)continue;
        for(size_t k=0;k<sizeof(numbers)/sizeof(numbers[0]);k++) {
            char text[32];snprintf(text,sizeof(text)," %s%sx",numbers[k],numbers[k]);
            int ids[64],n=bpe_encode_raw(t,text,ids,64),L=(int)strlen(text);
            CHECK(n==L-(family!=0),"number boundary length / unchanged old family");
            CHECK(ids[0]==(family==0?32:256+(unsigned char)numbers[k][0]),"space and numeric lead byte boundary");
            if(family==0) { int equal=n==L;for(int i=0;i<n;i++)equal&=ids[i]==(unsigned char)text[i];CHECK(equal,"Nd/Nl/No retain every UTF-8 byte and adjacent number"); }
        }
        int ids[16];
        CHECK(bpe_encode_raw(t,"12",ids,16)==(family==0?2:1)&&ids[0]==(family==0?'1':512),"adjacent ASCII numbers cannot merge in Smol only");
        CHECK(bpe_encode_raw(t," ax",ids,16)==2&&ids[0]==256+'a',"ordinary text keeps existing merge");
        CHECK(bpe_encode_raw(t,"",ids,16)==0,"empty raw input");
        if(family==1) {
            CHECK(bpe_encode_raw(t," \tx",ids,16)==2&&ids[0]==32&&ids[1]==513,"Qwen mixed whitespace leaves the final tab with the word");
            CHECK(bpe_encode_raw(t,"\t \tx",ids,16)==3&&ids[0]==9&&ids[1]==32&&ids[2]==513,"Qwen longer mixed prefix preserves the final tab-word merge");
            CHECK(bpe_encode_raw(t," \t",ids,16)==1&&ids[0]==256+9,"Qwen trailing whitespace remains a complete run");
        }
        if(family==0) {
            CHECK(bpe_encode_raw(t,"  ٠",ids,16)==3&&ids[0]==256+32&&ids[1]==0xD9&&ids[2]==0xA0,"whole whitespace run before isolated numeral");
            CHECK(bpe_encode_raw(t," \xF0\x9D",ids,16)==2&&ids[0]==256+0xF0,"truncated UTF-8 stays bytes");
            CHECK(bpe_encode_raw(t," \xB2",ids,16)==1&&ids[0]==256+0xB2,"invalid UTF-8 byte is not Unicode superscript two");
            CHECK(bpe_encode_raw(t," ٠",ids,1)==1&&ids[0]==32,"capacity bound");
        }
        if(family==2)
            CHECK(bpe_encode_raw(t,"14159",ids,16)==2&&ids[0]==516&&ids[1]=='9',"gpt2 path unchanged: its merges cross digit groups");
        bpe_free(t);
    }
    /* lfm2 is llama.cpp's llama3 split with ignore_merges: numbers in groups of three
     * however the merges tempt, and a piece the vocabulary holds whole is one token. */
    CHECK(fixture(path,"lfm2")==0,"write lfm2 fixture");
    bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load lfm2 fixture");
    if(t) {
        int ids[16];
        CHECK(bpe_encode_raw(t,"14159",ids,16)==2&&ids[0]==515&&ids[1]==517,"lfm2 numbers split 141|59");
        CHECK(bpe_encode_raw(t,"\tdef",ids,16)==1&&ids[0]==518,"lfm2 emits a whole-vocabulary piece without merges");
        CHECK(bpe_encode_raw(t," \tx",ids,16)==2&&ids[0]==32&&ids[1]==513,"lfm2 mixed whitespace hands the tab to the word");
        CHECK(bpe_encode_raw(t,"x 12",ids,16)==3&&ids[0]=='x'&&ids[1]==32&&ids[2]==512,"lfm2 a space does not join a number");
        bpe_free(t);
    }
    unlink(path);printf("SmolLM tokenizer: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
