#include "examples/bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int failed,checks;
static char sym[256][4];   /* the byte-level spelling of each byte */
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
static void check(bpe_tokenizer *t,const char *s,const int *expected,int count) {
    int ids[100];int n=bpe_encode(t,s,ids,100);
    CHECK(n==count && !memcmp(ids,expected,(size_t)count*sizeof(int)),s);
}
static void refused(bpe_tokenizer *t,const char *s) {
    int ids[4]={1234};CHECK(bpe_encode(t,s,ids,4)==0 && ids[0]==1234,s);
}
/* Byte tokens 0..255, one token per merge from 256, then <|im_start|> as CONTROL. */
static bpe_tokenizer *fixture(const char *pre,char merges[][16],int nm) {
    char path[]="/tmp/notorch-qwen35-text-XXXXXX";int fd=mkstemp(path);if(fd<0)return NULL;close(fd);
    static char words[512][16];const char *tokens[512],*list[256];int32_t types[512];
    for(int b=0;b<256;b++)strcpy(words[b],sym[b]);
    for(int i=0;i<nm;i++){char *o=words[256+i];for(const char *m=merges[i];*m;m++)if(*m!=' ')*o++=*m;*o=0;list[i]=merges[i];}
    strcpy(words[256+nm],"<|im_start|>");
    for(int i=0;i<=256+nm;i++){tokens[i]=words[i];types[i]=i==256+nm?3:1;}
    gguf_writer *w=gguf_write_open(path);if(!w){unlink(path);return NULL;}
    gguf_write_kv_str(w,"tokenizer.ggml.pre",pre);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,257+nm);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",list,nm);
    gguf_write_kv_i32_array(w,"tokenizer.ggml.token_type",types,257+nm);
    CHECK(!gguf_write_close(w),"fixture write");bpe_tokenizer *t=bpe_load(path);CHECK(t!=NULL,"load");
    unlink(path);return t;
}
/* The ASCII cases both Qwen3.5 pre-types answer alike. */
static void ascii_cases(bpe_tokenizer *t) {
    check(t,"12",(int[]){49,50},2);check(t,"a1",(int[]){97,49},2);check(t,"a!",(int[]){97,33},2);
    check(t,"!a",(int[]){259},1);check(t,"'sa",(int[]){260,97},2);check(t,"'Re",(int[]){267},1);
    check(t,"  a",(int[]){32,262},2);check(t," a\n",(int[]){262,10},2);
    check(t," \n",(int[]){264},1);check(t,"!\n\n",(int[]){33,265},2);
    int ids[100];CHECK(bpe_encode_special(t,"<|im_start|>",ids,100)==1 && ids[0]==bpe_token_id(t,"<|im_start|>"),"explicit control token");
    CHECK(bpe_encode(t,"<|im_start|>",ids,100)>1,"ordinary path stays literal");
}
int main(void) {
    for(int b=0,next=0;b<256;b++) {
        int literal=(b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
        int cp=literal?b:256+next++;
        if(cp<128)sym[b][0]=(char)cp;
        else{sym[b][0]=(char)(0xc0|(cp>>6));sym[b][1]=(char)(0x80|(cp&63));}
    }
    char merges[128][16]={"1 2","a 1","a !","! a","' s","s a","Ġ a","a Ċ","Ġ Ċ","Ċ Ċ","' R","'R e","Ġ Ġ","ĠĠ a"};
    bpe_tokenizer *t=fixture("qwen35-ascii",merges,14);if(!t)return 1;
    ascii_cases(t);
    int ids[100];ids[0]=1234;CHECK(bpe_encode_special(t,"caf\xc3\xa9",ids,100)==0 && ids[0]==1234,"NFC boundary refused before output");
    bpe_free(t);

    /* "qwen35": the same merges, then byte pairs that straddle each boundary under test,
     * ids 270.., so a piece cut in the wrong place shows as a merge made or missed. */
    static const unsigned char pairs[][2]={{'a',0xD6},{0xB7,'a'},{0xA1,0xD9},{'a',0xD9},{'\t','a'},
        {0xA0,'a'},{0x90,'\n'},{0xBF,'\n'},{0xBF,'a'},{0x8F,'!'}};
    int np=(int)(sizeof(pairs)/sizeof(pairs[0]));
    for(int i=0;i<np;i++)snprintf(merges[14+i],16,"%s %s",sym[pairs[i][0]],sym[pairs[i][1]]);
    bpe_tokenizer *q=fixture("qwen35",merges,14+np);if(!q)return 1;
    ascii_cases(q);
    check(q,"a\xd6\xb7",(int[]){270,183},2);                       /* a + patah: one word */
    check(q,"!\xd6\xb7""a",(int[]){33,214,271},3);                 /* prefix, mark, letter */
    check(q,"\xd9\xa1\xd9\xa2",(int[]){217,161,217,162},4);        /* ١٢: one number each */
    check(q,"a\xd9\xa1",(int[]){97,217,161},3);
    check(q,"\ta",(int[]){274},1);                                 /* a tab prefixes a word */
    check(q,"\xc2\xa0""a",(int[]){194,275},2);                     /* so does NBSP */
    check(q,"\xc2\xa0\xc2\xa0""a",(int[]){194,160,194,275},4);     /* the run gives up its last */
    check(q,"\xd7\x90\n",(int[]){215,144,10},3);                   /* א, newline apart */
    check(q,"\xc2\xbf\n",(int[]){194,277},2);                      /* ¿ keeps its newline */
    check(q,"'\xc5\xbf""a",(int[]){39,197,191,97},4);              /* 'ſ folds to 's */
    refused(q,"'\xe2\x84\xaa""a");         /* 'K (Kelvin) is NFC_QC No: the reference reads 'K */
    check(q,"\xe2\x9d\xa4\xef\xb8\x8f!",(int[]){226,157,164,239,184,143,33},7); /* ❤ + VS16 (a mark) | ! */
    /* בּ with dagesh (ccc 21) typed before patah (17): the encoder sees patah first. */
    check(q,"\xd7\x91\xd6\xbc\xd6\xb7",(int[]){215,145,214,183,214,188},6);
    check(q,"\xd7\x91\xd6\xb7\xd6\xbc",(int[]){215,145,214,183,214,188},6);
    check(q,"\xd7\x91\xd6\xbc\xd7\x91\xd6\xb7",(int[]){215,145,214,188,215,145,214,183},8);
    refused(q,"\xd0\xb8\xcc\x86");          /* и + breve: NFC_QC Maybe, composes to й */
    refused(q,"\xe2\x84\xab");              /* Å ANGSTROM SIGN: NFC_QC No */
    refused(q,"\xe1\x84\x80\xe1\x85\xa1");  /* Hangul L + V */
    refused(q,"\xe0\xb4\xbb");              /* U+0D3B: ccc 9, a starter to the reference */
    refused(q,"\xe0\xa2\x97");              /* U+0897: unassigned in 15.0, a mark to the reference */
    refused(q,"\xc3");refused(q,"\xff");refused(q,"\xc0\xaf");refused(q,"\xed\xa0\x80");refused(q,"a\x80");
    ids[0]=1234;CHECK(bpe_encode_special(q,"\xe2\x84\xab<|im_start|>",ids,100)==0 && ids[0]==1234,"special path refuses too");
    bpe_free(q);

    /* Every ASCII string of up to four characters over an alphabet with every pair merged
     * splits under "qwen35" exactly as under "qwen35-ascii". */
    const char *alpha="a1's Re\t\n!";char dense[100][16];int nd=0,same=0,total=0;
    for(const char *x=alpha;*x;x++)for(const char *y=alpha;*y;y++)
        snprintf(dense[nd++],16,"%s %s",sym[(unsigned char)*x],sym[(unsigned char)*y]);
    bpe_tokenizer *da=fixture("qwen35-ascii",dense,nd),*dq=fixture("qwen35",dense,nd);if(!da || !dq)return 1;
    for(int len=1;len<=4;len++)for(int idx[4]={0};;) {
        char s[5];int a[8],b[8];
        for(int k=0;k<len;k++)s[k]=alpha[idx[k]];
        s[len]=0;
        int na=bpe_encode(da,s,a,8),nb=bpe_encode(dq,s,b,8);
        total++;same+=na==nb && !memcmp(a,b,(size_t)na*sizeof(int));
        int k=0;while(k<len && ++idx[k]==10)idx[k++]=0;
        if(k==len)break;
    }
    CHECK(total==11110 && same==total,"ASCII: qwen35 splits as qwen35-ascii");
    bpe_free(da);bpe_free(dq);
    printf("Qwen3.5 text: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
