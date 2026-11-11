/* Private replay consumes the saved rows, survives disk restart and owns its
 * worker input. These use the mixed LFM3 convolution/attention fixture. */
#include "harness/experts_internal.h"
#include "harness/archs.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void equal(const char *a,const char *b) {
    FILE *x=fopen(a,"rb"),*y=fopen(b,"rb"); assert(x && y);
    int c,d; do { c=fgetc(x); d=fgetc(y); assert(c==d); } while(c!=EOF);
    assert(!ferror(x) && !ferror(y)); assert(!fclose(x) && !fclose(y));
}
static void wake(nt_experts *p,const int *ids,const float *const *rows,int n) {
    float *logits=malloc((size_t)p->dims.vocab*sizeof(float));
    kv_cache *kv=nt_model_kv_new(p->decoder,p->body.model,p->dims,n);
    assert(logits && kv);
    assert(!p->decoder->doe->forward_embd(p->body.model,kv,ids,rows,n,0,logits,NULL,
                                         nt_experts_attend,nt_experts_observe,p));
    int best=0; for(int i=1;i<p->dims.vocab;i++) if(logits[i]>logits[best]) best=i;
    nt_experts_choice(p,logits[best],logits[best]);
    nt_model_kv_free(p->decoder,p->body.model,kv); free(logits);
}

static const nt_doe_decoder *real_doe;
static int abort_late(void *m,kv_cache *kv,const int *ids,int n,int pos,float *logits,
                      nt_residual_fn a,nt_residual_fn b,void *u) {
    if(pos>=8) return NT_E_STATE;
    return real_doe->forward(m,kv,ids,n,pos,logits,a,b,u);
}

static void reject_file(nt_experts *p,const char *key,int relative,unsigned char byte) {
    FILE *f=fopen("episode.gguf","rb"); assert(f && !fseek(f,0,SEEK_END));
    long n=ftell(f); rewind(f); assert(n>0);
    unsigned char *data=malloc((size_t)n); assert(data && fread(data,1,n,f)==(size_t)n && !fclose(f));
    long i=0; size_t k=strlen(key);
    while(i<=n-(long)k && memcmp(data+i,key,k)) i++;
    assert(i<=n-(long)k && i+(long)k+relative<n);
    data[i+k+relative]=byte;
    f=fopen("bad-episode","wb"); assert(f && fwrite(data,1,n,f)==(size_t)n && !fclose(f)); free(data);
    assert(!nt_replay_load(p,"bad-episode"));
}

static void reject_receipt(nt_experts *p,const char *key,int relative,unsigned char byte) {
    FILE *f=fopen("sync-first","rb"); assert(f && !fseek(f,0,SEEK_END));
    long n=ftell(f); rewind(f); assert(n>0);
    unsigned char *data=malloc((size_t)n); assert(data && fread(data,1,n,f)==(size_t)n && !fclose(f));
    size_t length=strlen(key); long i=0;
    while(i<=n-(long)length && memcmp(data+i,key,length)) i++;
    assert(i<=n-(long)length && i+(long)length+relative<n);
    data[i+length+relative]=byte;
    f=fopen("bad-receipt","wb"); assert(f && fwrite(data,1,n,f)==(size_t)n && !fclose(f)); free(data);
    assert(!nt_experts_open_on_body(p->frame_host,"bad-receipt",NULL,&p->body));
}

int main(void) {
    alarm(90);
    gguf_file *host=gguf_open("lfm-mixed");
    const nt_arch *arch=nt_pick_arch("lfm3"); nt_dims dims={0};
    void *model=host && arch ? arch->load(host,&dims) : NULL;
    bpe_tokenizer *tok=bpe_load("tokenizer.gguf"); assert(model && tok);
    nt_expert_body body={model,tok};
    nt_experts *p=nt_experts_open_on_body(host,NULL,NULL,&body); assert(p);
    int ids[6]={2,3,0,0,4,5};
    float media[128]; assert(p->dim==64);
    for(int i=0;i<128;i++) media[i]=sinf(i*0.17f);
    const float *rows[6]={NULL,NULL,media,media+64,NULL,NULL};
    wake(p,ids,rows,6); assert(!nt_experts_finish(p,1));
    assert(!nt_experts_save_portable(p,"awake"));
    int suppressed[2]={33,32};
    nt_replay_episode *e=nt_replay_capture(p,ids,rows,6,suppressed,2,NULL,0); assert(e);
    assert(strlen(nt_replay_id(e))==64 && !nt_replay_save(e,"episode.gguf"));
    assert(nt_replay_save(e,"episode.gguf"));
    int reverse[2]={32,33},duplicates[2]={32,32};
    nt_replay_episode *ordered=nt_replay_capture(p,ids,rows,6,reverse,2,NULL,0); assert(ordered);
    assert(!strcmp(nt_replay_id(e),nt_replay_id(ordered)) && !nt_replay_save(ordered,"ordered"));
    equal("episode.gguf","ordered"); nt_replay_free(ordered);
    assert(!nt_replay_capture(p,ids,rows,6,duplicates,2,NULL,0));
    assert(!nt_replay_capture(p,ids,rows,6,NULL,0,duplicates,2));
    ids[2]=-1; media[0]=NAN;
    assert(!nt_replay_capture(p,ids,rows,6,NULL,0,NULL,0));
    ids[2]=0; assert(!nt_replay_capture(p,ids,rows,6,NULL,0,NULL,0));
    media[0]=0;
    nt_replay_episode *loaded=nt_replay_load(p,"episode.gguf"); assert(loaded);
    assert(!strcmp(nt_replay_id(loaded),nt_replay_id(e)));
    assert(!nt_replay_save(loaded,"episode-copy")); equal("episode.gguf","episode-copy");
    nt_replay_free(loaded);
    nt_replay_episode *text=nt_replay_capture(p,ids,NULL,6,NULL,0,NULL,0); assert(text);
    assert(!nt_replay_save(text,"text-episode"));
    loaded=nt_replay_load(p,"text-episode"); assert(loaded);
    assert(!strcmp(nt_replay_id(loaded),nt_replay_id(text)));
    nt_replay_free(loaded); nt_replay_free(text);
    assert(!nt_experts_replay(p,e,6,0)); assert(!nt_experts_save_portable(p,"preview")); equal("awake","preview");
    assert(nt_experts_replay(p,e,0,1)==NT_E_ARG && nt_experts_replay(p,e,33,1)==NT_E_ARG);
    char saved=p->host[0]; p->host[0]=saved=='0'?'1':'0';
    assert(nt_experts_replay(p,e,6,1)==NT_E_STATE && !nt_replay_load(p,"episode.gguf")); p->host[0]=saved;
    int ctx=host->ctx_len; host->ctx_len=7;
    assert(nt_experts_replay(p,e,6,1)==NT_E_CAPACITY); host->ctx_len=ctx;
    uint64_t cycles=p->replay.cycles; p->replay.cycles=UINT64_MAX;
    assert(nt_experts_replay(p,e,6,1)==NT_E_STATE); p->replay.cycles=cycles;
    p->replay.elections=UINT64_MAX;
    assert(nt_experts_replay(p,e,6,1)==NT_E_STATE); p->replay.elections=0;
    nt_doe_decoder faulty=*arch->doe; nt_arch broken=*arch; real_doe=arch->doe;
    faulty.forward=abort_late; broken.doe=&faulty; p->decoder=&broken;
    assert(nt_experts_replay(p,e,6,1)==NT_E_STATE); p->decoder=arch;
    assert(!nt_experts_save_portable(p,"after-failure")); equal("awake","after-failure");
    reject_file(p,"doe.episode.sha256",12,'x');
    reject_file(p,"doe.episode.width",4,63);
    reject_file(p,"doe.episode.ids",16,255);
    reject_file(p,"doe.episode.positions",16,255);
    reject_file(p,"doe.episode.tokens",4,255);
    FILE *f=fopen("episode.gguf","rb"),*g=fopen("short-episode","wb"); assert(f && g);
    for(int i=0;i<40;i++) { int c=fgetc(f); assert(c!=EOF && fputc(c,g)!=EOF); }
    assert(!fclose(f) && !fclose(g) && !nt_replay_load(p,"short-episode"));
    /* A valid finite row with its old content address must also be rejected. */
    gguf_file *eg=gguf_open("episode.gguf"); assert(eg && eg->n_tensors==1);
    f=fopen("episode.gguf","rb"); assert(f && !fseek(f,0,SEEK_END));
    long file_size=ftell(f); rewind(f); assert(file_size>0);
    unsigned char *bytes=malloc((size_t)file_size+1); assert(bytes);
    assert(fread(bytes,1,file_size,f)==(size_t)file_size && !fclose(f));
    size_t at=eg->data_offset+eg->tensors[0].offset;
    float changed_value=1; memcpy(bytes+at,&changed_value,sizeof changed_value);
    g=fopen("corrupt-row","wb"); assert(g && fwrite(bytes,1,file_size,g)==(size_t)file_size && !fclose(g));
    assert(!nt_replay_load(p,"corrupt-row"));
    /* Restore row, then test exact physical extent and duplicate metadata. */
    memcpy(bytes+at,eg->data+eg->tensors[0].offset,sizeof(float)); gguf_close(eg);
    for(int extra=-1;extra<=1;extra+=2) {
        bytes[file_size]=0;
        g=fopen("wrong-size","wb"); assert(g && fwrite(bytes,1,file_size+extra,g)==(size_t)(file_size+extra) && !fclose(g));
        assert(!nt_replay_load(p,"wrong-size"));
    }
    const char key[]="doe.episode.vocab";
    size_t at_key=0;
    while(at_key+sizeof(key)-1<(size_t)file_size && memcmp(bytes+at_key,key,sizeof(key)-1)) at_key++;
    assert(at_key+sizeof(key)-1<(size_t)file_size);
    memcpy(bytes+at_key,"doe.episode.width",sizeof(key)-1);
    g=fopen("duplicate-key","wb"); assert(g && fwrite(bytes,1,file_size,g)==(size_t)file_size && !fclose(g));
    assert(!nt_replay_load(p,"duplicate-key")); free(bytes);

    /* Sparse rows on both sides of two prefill chunk boundaries. Changing
     * only a row in the final chunk must reach the elected experts. */
    enum { LONG_N=2*NT_PREFILL_CHUNK+6 };
    int long_ids[LONG_N]; const float *long_rows[LONG_N]={0};
    for(int i=0;i<LONG_N;i++) long_ids[i]=i%dims.vocab;
    float late[64]; memcpy(late,media,sizeof late);
    long_rows[NT_PREFILL_CHUNK-1]=media;
    long_rows[NT_PREFILL_CHUNK]=media+64;
    long_rows[2*NT_PREFILL_CHUNK-1]=media;
    long_rows[2*NT_PREFILL_CHUNK]=late;
    nt_experts *cross[2];
    for(int flip=0;flip<2;flip++) {
        cross[flip]=nt_experts_open_on_body(host,"awake",NULL,&body); assert(cross[flip]);
        late[0]=flip ? 2 : 0;
        ordered=nt_replay_capture(cross[flip],long_ids,long_rows,LONG_N,NULL,0,NULL,0); assert(ordered);
        assert(!nt_experts_replay(cross[flip],ordered,2,1)); nt_replay_free(ordered);
    }
    assert(memcmp(cross[0]->a,cross[1]->a,SEATS*(size_t)p->dim*RANK*sizeof(float)));
    nt_experts_free(cross[0]); nt_experts_free(cross[1]);

    /* Changed media under identical IDs must change learned factors. */
    assert(!nt_experts_replay(p,e,6,1));
    nt_experts *q=nt_experts_open_on_body(host,"awake",NULL,&body); assert(q);
    media[0]=2; nt_replay_episode *changed=nt_replay_capture(q,ids,rows,6,suppressed,2,NULL,0); assert(changed);
    assert(strcmp(nt_replay_id(changed),nt_replay_id(e)) && !nt_experts_replay(q,changed,6,1));
    assert(memcmp(q->a,p->a,SEATS*(size_t)p->dim*RANK*sizeof(float)));
    nt_experts_free(q); nt_replay_free(changed);
    assert(!nt_experts_save_portable(p,"sync-first")); nt_experts_free(p);

    /* A locked trace proves submit returns before the worker can replay.
     * Free its source immediately; the worker must own a deep copy. */
    FILE *trace=tmpfile(); assert(trace); flockfile(trace);
    p=nt_experts_open_on_body(host,"awake",trace,&body); assert(p);
    nt_consolidation *job=NULL;
    assert(!nt_experts_submit_replay(&p,&job,e,6) && !p && job);
    nt_replay_free(e); e=NULL;
    funlockfile(trace);
    assert(!nt_experts_await(&p,&job) && p && !job);
    assert(!nt_experts_save_portable(p,"worker-first")); equal("sync-first","worker-first");
    reject_receipt(p,"doe.replay.cycles",4,0);
    reject_receipt(p,"doe.replay.cycles",0,11);
    reject_receipt(p,"doe.replay.choices",4,0);
    reject_receipt(p,"doe.replay.elections",4,0);
    reject_receipt(p,"doe.replay.prefill",4,0);
    reject_receipt(p,"doe.replay.stops",4,2);
    reject_receipt(p,"doe.replay.last",12,'x');
    reject_receipt(p,"doe.replay.last",-1,'x');
    nt_experts_free(p); assert(!fclose(trace));

    /* 96 further waking turns, 96 replay episodes and 32 anchor dreams.
     * Every turn crosses disk and every worker result equals its serial twin. */
    char previous[2][64]={"sync-first","worker-first"},out[2][64];
    for(int turn=1;turn<=96;turn++) {
        for(int worker=0;worker<2;worker++) {
            p=nt_experts_open_on_body(host,previous[worker],NULL,&body); assert(p);
            wake(p,ids,rows,6);
            e=nt_replay_load(p,"episode.gguf"); assert(e);
            if(worker) {
                assert(!nt_experts_submit_replay(&p,&job,e,6) && !p && job);
                nt_replay_free(e); e=NULL;
                assert(!nt_experts_await(&p,&job) && p && !job);
            } else {
                assert(!nt_experts_finish(p,1) && !nt_experts_replay(p,e,6,1)); nt_replay_free(e); e=NULL;
            }
            if(turn%3==0) assert(!nt_experts_dream(p,2,-1,1));
            assert(p->turns==(uint64_t)turn+1 && p->replay.cycles==(uint64_t)turn+1);
            assert(p->replay.elections==p->replay.prefill+p->replay.choices-p->replay.stops);
            assert(p->sleep.cycles==(uint64_t)turn/3);
            snprintf(out[worker],sizeof out[worker],"long-%d-%d",worker,turn);
            assert(!nt_experts_save_portable(p,out[worker]));
            if(turn>1) assert(!unlink(previous[worker]));
            strcpy(previous[worker],out[worker]); nt_experts_free(p);
        }
        equal(out[0],out[1]);
    }
    p=nt_experts_open_on_body(host,previous[0],NULL,&body); assert(p);
    assert(p->sleep.seen==UINT32_MAX && nt_experts_ancestor_count(p)>=6);
    /* Every valid token stopping on its first choice exercises exact clocks. */
    int *all=malloc((size_t)dims.vocab*sizeof(int)); assert(all);
    for(int i=0;i<dims.vocab;i++) all[i]=i;
    e=nt_replay_capture(p,ids,rows,6,NULL,0,all,dims.vocab); assert(e);
    uint64_t choices=p->replay.choices,stops=p->replay.stops,elections=p->replay.elections;
    assert(!nt_experts_replay(p,e,6,1));
    assert(p->replay.choices==choices+1 && p->replay.stops==stops+1 && p->replay.elections==elections+6);
    nt_replay_free(e);
    e=nt_replay_capture(p,ids,rows,6,all,dims.vocab,NULL,0); assert(e);
    assert(nt_experts_replay(p,e,6,1)==NT_E_STATE);
    assert(!nt_experts_dream(p,2,-1,1));
    assert(p->sleep.cycles==33 && p->sleep.seen && (p->sleep.seen&(p->sleep.seen-1))==0);
    nt_replay_free(e); free(all); nt_experts_free(p);
    bpe_free(tok); arch->free(model); gguf_close(host);
    puts("Media replay: 97 persisted episodes, 96 worker restarts, 33 text dreams, owned inputs and rollback passed");
    return 0;
}
