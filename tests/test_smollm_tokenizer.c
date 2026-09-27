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
    char storage[512][16], ranks[256][24];const char *tokens[513],*merges[257];
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
    gguf_writer *w=gguf_write_open(path);if(!w)return -1;
    gguf_write_kv_str(w,"tokenizer.ggml.model","gpt2");
    gguf_write_kv_str(w,"tokenizer.ggml.pre",pre);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,513);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,257);
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
        if(family==0) {
            CHECK(bpe_encode_raw(t,"  ٠",ids,16)==3&&ids[0]==256+32&&ids[1]==0xD9&&ids[2]==0xA0,"whole whitespace run before isolated numeral");
            CHECK(bpe_encode_raw(t," \xF0\x9D",ids,16)==2&&ids[0]==256+0xF0,"truncated UTF-8 stays bytes");
            CHECK(bpe_encode_raw(t," \xB2",ids,16)==1&&ids[0]==256+0xB2,"invalid UTF-8 byte is not Unicode superscript two");
            CHECK(bpe_encode_raw(t," ٠",ids,1)==1&&ids[0]==32,"capacity bound");
        }
        bpe_free(t);
    }
    unlink(path);printf("SmolLM tokenizer: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
