/* SmolVLM / Idefics3 vision on fixtures: geometry from the projector's keys, pixels, tower and
 * connector against an independent numpy reference, the prompt layout from the body's own
 * vocabulary (<image> at 264, not SmolVLM's 49190), and the rows arch_llama's forward_embd
 * puts in place of token embeddings. No real body is read. */
#include "harness/smolvlm_vision.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "smolvlm_vision_fixture.h"
static int checks,failed;
#define CHECK(c,s) do{checks++;if(!(c)){fprintf(stderr,"FAIL: %s\n",s);failed++;}}while(0)
static float difference(const float *a,const float *b,size_t n){float d=0;for(size_t j=0;j<n;j++)if(fabsf(a[j]-b[j])>d)d=fabsf(a[j]-b[j]);return d;}

enum { E=8, VOCAB=265, TFFN=12, IMAGE_ID=264 };
static const char *markers[]={"<|im_start|>","<end_of_utterance>","<fake_token_around_image>","<global-img>",
    "<row_1_col_1>","<row_1_col_2>","<row_2_col_1>","<row_2_col_2>","<image>"};
/* A one-layer llama over the 256 byte symbols and the markers above; drop renames one marker. */
static int text_fixture(const char *path,const char *drop) {
    static char words[VOCAB][32];const char *tokens[VOCAB];int32_t types[VOCAB];int next=0;
    for(int b=0;b<256;b++) {
        int literal=(b>=33 && b<=126) || (b>=161 && b<=172) || b>=174,cp=literal?b:256+next++;
        memset(words[b],0,sizeof(words[b]));
        if(cp<128)words[b][0]=(char)cp;
        else{words[b][0]=(char)(0xc0|(cp>>6));words[b][1]=(char)(0x80|(cp&63));}
        types[b]=1;
    }
    for(int i=256;i<VOCAB;i++){snprintf(words[i],sizeof(words[i]),"%s",strcmp(markers[i-256],drop)?markers[i-256]:"<|unused|>");types[i]=3;}
    for(int i=0;i<VOCAB;i++)tokens[i]=words[i];
    const char *merges[]={"x y"};
    gguf_writer *w=gguf_write_open(path);if(!w)return 1;
    gguf_write_kv_str(w,"general.architecture","llama");
    gguf_write_kv_u32(w,"llama.block_count",1);gguf_write_kv_u32(w,"llama.context_length",128);
    gguf_write_kv_u32(w,"llama.embedding_length",E);gguf_write_kv_u32(w,"llama.feed_forward_length",TFFN);
    gguf_write_kv_u32(w,"llama.attention.head_count",2);gguf_write_kv_u32(w,"llama.attention.head_count_kv",1);
    gguf_write_kv_f32(w,"llama.rope.freq_base",10000.0f);gguf_write_kv_f32(w,"llama.attention.layer_norm_rms_epsilon",1e-5f);
    gguf_write_kv_str(w,"tokenizer.ggml.model","gpt2");
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",tokens,VOCAB);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",merges,1);
    gguf_write_kv_i32_array(w,"tokenizer.ggml.token_type",types,VOCAB);
    struct { const char *name; int norm; uint64_t shape[2]; } ts[]={
        {"token_embd.weight",0,{E,VOCAB}},{"output_norm.weight",1,{E,1}},{"blk.0.attn_norm.weight",1,{E,1}},
        {"blk.0.attn_q.weight",0,{E,E}},{"blk.0.attn_k.weight",0,{E,4}},{"blk.0.attn_v.weight",0,{E,4}},
        {"blk.0.attn_output.weight",0,{E,E}},{"blk.0.ffn_norm.weight",1,{E,1}},{"blk.0.ffn_gate.weight",0,{E,TFFN}},
        {"blk.0.ffn_up.weight",0,{E,TFFN}},{"blk.0.ffn_down.weight",0,{TFFN,E}}};
    int nt=(int)(sizeof(ts)/sizeof(ts[0]));
    for(int t=0;t<nt;t++)gguf_write_tensor_decl(w,ts[t].name,ts[t].norm?1:2,ts[t].shape,GGUF_TYPE_F32);
    for(int t=0;t<nt;t++) {
        size_t n=ts[t].shape[0]*ts[t].shape[1];float x[E*VOCAB];
        for(size_t j=0;j<n;j++)x[j]=(float)((int)((j*5+(size_t)t*11)%19)-9)/32.0f+(float)ts[t].norm;
        if(gguf_write_tensor_f32(w,ts[t].name,x,n)){gguf_write_abort(w);return 1;}
    }
    return gguf_write_close(w);
}

int main(void) {
    char path[]="/tmp/notorch-smolvlm-XXXXXX",tpath[]="/tmp/notorch-smolvlm-text-XXXXXX";
    int fd=mkstemp(path);if(fd<0)return 1;close(fd);
    fd=mkstemp(tpath);if(fd<0){unlink(path);return 1;}close(fd);
    CHECK(!smolvlm_vision_fixture(path,0),"write vision fixture");gguf_file *g=gguf_open(path);
    nt_smolvlm_vision *v=nt_smolvlm_vision_load(g);CHECK(v!=NULL,"vision load");
    if(!v){gguf_close(g);unlink(path);unlink(tpath);return 1;}
    const nt_smolvlm_vision_info *i=nt_smolvlm_vision_get_info(v);
    CHECK(i->width==8 && i->layers==2 && i->heads==2 && i->ffn==4 && i->image_size==8 && i->patch==2 &&
          i->scale==2 && i->text_width==8 && i->patches==16 && i->tokens==4 && i->longest==16,"geometry from the projector keys");
    CHECK(i->eps==1e-5f && i->mean[0]==0.4f && i->mean[2]==0.6f && i->std[0]==0.3f && i->std[2]==0.7f,"eps, mean and std from the file");

    uint8_t rgb[7*11*3];for(int j=0;j<(int)sizeof(rgb);j++)rgb[j]=(uint8_t)(j*7+3);
    nt_smolvlm_image im={0};CHECK(!nt_smolvlm_image_prepare(v,rgb,7,11,0,&im),"prepare 7x11 to longest edge 16");
    CHECK(im.rows==smolvlm_rows && im.cols==smolvlm_cols && im.frames==smolvlm_frames,"2x2 tiles plus the global frame");
    size_t fs=3*8*8;
    CHECK(im.pixels && difference(im.pixels,smolvlm_pixels,smolvlm_frames*fs)<1e-6f,"bilinear tiles, global frame last, metadata mean/std");
    float hidden[16*8],rows_g[4*8];
    CHECK(!nt_smolvlm_vision_tower(v,im.pixels+(smolvlm_frames-1)*fs,hidden) &&
          difference(hidden,smolvlm_global_hidden,16*8)<2e-5f,"SigLIP tower with f16 GELU, both fc1 namings");
    CHECK(!nt_smolvlm_vision_connect(v,hidden,rows_g) && difference(rows_g,smolvlm_global_rows,4*8)<2e-5f,"pixel shuffle and connector");
    float *features=NULL;CHECK(!nt_smolvlm_vision_encode(v,&im,&features) && features,"encode every frame");
    if(features)CHECK(!memcmp(features+(smolvlm_frames-1)*4*8,rows_g,sizeof(rows_g)),"encode keeps frame order");

    CHECK(!text_fixture(tpath,""),"write text fixture");
    bpe_tokenizer *tok=bpe_load(tpath);CHECK(tok && bpe_token_id(tok,"<image>")==IMAGE_ID,"marker found by text, away from 49190");
    int ids[128],n=tok?nt_smolvlm_prompt(v,&im,tok,smolvlm_instruction,ids,128):-1,np=(int)(sizeof(smolvlm_prompt_ids)/sizeof(int));
    CHECK(n==np && !memcmp(ids,smolvlm_prompt_ids,sizeof(smolvlm_prompt_ids)),"idefics3 layout, ids from the body's vocabulary");
    CHECK(!tok || nt_smolvlm_prompt(v,&im,tok,"<image>",ids+64,64)<0,"an instruction spelling a marker is refused");
    CHECK(!tok || nt_smolvlm_prompt(v,&im,tok,smolvlm_instruction,ids+64,np)<0,"short capacity is refused");
    const float *rows[128]={0};
    CHECK(features && n==np && !nt_smolvlm_image_rows(v,&im,features,tok,ids,n,rows),"rows at the <image> ids");
    int first=-1,last=-1;for(int t=0;t<n;t++)if(ids[t]==IMAGE_ID){if(first<0)first=t;last=t;}
    CHECK(first>0 && rows[first-1]==NULL && rows[first]==features && rows[last]==features+19*8 && rows[n-1]==NULL,"k-th <image> reads the k-th row");
    int dropped[128];memcpy(dropped,ids,sizeof(ids));dropped[first]='A';
    const float *sentinel[128];for(int t=0;t<128;t++)sentinel[t]=(const float*)rows;
    CHECK(nt_smolvlm_image_rows(v,&im,features,tok,dropped,n,sentinel)==NT_E_ARG && sentinel[first+1]==(const float*)rows,"a missing slot refused, rows untouched");

    gguf_file *tg=gguf_open(tpath);nt_dims dims={0};void *m=tg?nt_arch_llama.load(tg,&dims):NULL;
    CHECK(m && nt_arch_llama.forward_embd,"llama decoder takes media rows");
    if(m && n==np) {
        kv_cache *kv=kv_new(dims.n_layers,64,dims.kv_dim);
        float plain[VOCAB],logits[VOCAB],ref[VOCAB],h[E];
        CHECK(!nt_arch_llama.forward(m,kv,ids,n,0,plain),"plain text forward");
        CHECK(!nt_arch_llama.forward_embd(m,kv,ids,NULL,n,0,logits,NULL) && !memcmp(plain,logits,sizeof(plain)),"NULL rows is forward bit for bit");
        /* Each <image> position gets the embedding of a different byte token: the spliced run
         * must equal plain text holding those tokens, so a row that lands one position off fails. */
        float *emb=gguf_dequant(tg,gguf_find_tensor(tg,"token_embd.weight"));int alt[128];const float *own[128];
        for(int t=0;t<n;t++){alt[t]=ids[t]==IMAGE_ID?65+t%23:ids[t];own[t]=ids[t]==IMAGE_ID&&emb?emb+(size_t)alt[t]*E:NULL;}
        CHECK(emb && !nt_arch_llama.forward(m,kv,alt,n,0,ref),"forward with distinct tokens in the slots");
        CHECK(!nt_arch_llama.forward_embd(m,kv,ids,own,n,0,logits,NULL) && !memcmp(ref,logits,sizeof(ref)),"a row stands exactly at its own position");
        CHECK(!nt_arch_llama.forward_embd(m,kv,ids,rows,n,0,logits,h) && difference(logits,plain,VOCAB)>1e-4f,"vision rows reach the logits");
        float worst=0;for(int t=0;emb && t<VOCAB;t++){float dot=0;for(int e=0;e<E;e++)dot+=h[e]*emb[(size_t)t*E+e];if(fabsf(dot-logits[t])>worst)worst=fabsf(dot-logits[t]);}
        CHECK(emb && worst<1e-4f,"hidden is the vector the tied head reads");
        for(int j=0;j<VOCAB;j++)logits[j]=1234;
        float bad[E];memcpy(bad,features,sizeof(bad));bad[3]=NAN;const float *saved=rows[first];rows[first]=bad;
        CHECK(nt_arch_llama.forward_embd(m,kv,ids,rows,n,0,logits,NULL)==NT_E_ARG && logits[0]==1234,"nonfinite row refused, logits untouched");
        rows[first]=saved;free(emb);kv_free(kv);
    }
    if(m)nt_arch_llama.free(m);gguf_close(tg);bpe_free(tok);

    CHECK(!text_fixture(tpath,"<row_2_col_2>"),"write text fixture without a tile marker");
    tok=bpe_load(tpath);CHECK(tok && nt_smolvlm_prompt(v,&im,tok,smolvlm_instruction,ids,128)<0,"layout refuses a marker the vocabulary lacks");
    bpe_free(tok);

    float keep=hidden[0],frame[3*8*8];memcpy(frame,im.pixels,sizeof(frame));frame[5]=NAN;
    CHECK(nt_smolvlm_vision_tower(v,frame,hidden)==NT_E_ARG && hidden[0]==keep,"NaN frame refused before output");
    nt_smolvlm_image untouched={.frames=99};
    CHECK(nt_smolvlm_image_prepare(v,rgb,0,11,0,&untouched)==NT_E_ARG && untouched.frames==99,"bad dimensions preserve output");
    CHECK(nt_smolvlm_image_prepare(v,rgb,7,11,4,&untouched)==NT_E_ARG,"longest edge below the tile refused");
    free(features);nt_smolvlm_vision_free(v);gguf_close(g);

    CHECK(!smolvlm_vision_fixture(path,2),"write projector without preproc_image_size");
    g=gguf_open(path);v=nt_smolvlm_vision_load(g);CHECK(v && nt_smolvlm_vision_get_info(v)->longest==0,"absent longest edge loads as 0");
    nt_smolvlm_image im2={0};
    CHECK(v && nt_smolvlm_image_prepare(v,rgb,7,11,0,&im2)==NT_E_ARG,"no longest edge in file or call: refused");
    CHECK(v && !nt_smolvlm_image_prepare(v,rgb,7,11,16,&im2) && !memcmp(im2.pixels,im.pixels,smolvlm_frames*fs*sizeof(float)),"explicit longest edge gives the same pixels");
    nt_smolvlm_image_free(&im2);nt_smolvlm_vision_free(v);gguf_close(g);
    const char *names[]={"","missing tensor","","missing scale_factor","non-idefics3 projector"};
    for(int variant=1;variant<=4;variant+=variant==1?2:1) {
        CHECK(!smolvlm_vision_fixture(path,variant),"write malformed projector");
        g=gguf_open(path);v=nt_smolvlm_vision_load(g);CHECK(v==NULL,names[variant]);nt_smolvlm_vision_free(v);gguf_close(g);
    }
    nt_smolvlm_image_free(&im);unlink(path);unlink(tpath);
    printf("SmolVLM vision: %d checks, %d failures\n",checks,failed);return failed?1:0;
}
