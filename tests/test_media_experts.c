/* Living experts on mixed text/media rows, including disk restart and a worker.
 * The fixture's convolution and both attention kinds run for every episode. */
#include "harness/experts.h"
#include "harness/archs.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void equal_files(const char *a, const char *b) {
    FILE *x=fopen(a,"rb"), *y=fopen(b,"rb");
    assert(x && y);
    int c,d;
    do { c=fgetc(x); d=fgetc(y); assert(c==d); } while(c!=EOF);
    assert(!ferror(x) && !ferror(y));
    assert(!fclose(x) && !fclose(y));
}

int main(void) {
    gguf_file *host=gguf_open("lfm-mixed");
    const nt_arch *arch=nt_pick_arch("lfm3");
    nt_dims dims={0};
    void *model=host && arch ? arch->load(host,&dims) : NULL;
    bpe_tokenizer *tok=bpe_load("tokenizer.gguf");
    assert(model && tok && arch->doe && arch->doe->forward_embd);
    nt_expert_body body={model,tok};
    enum { N=6 };
    int ids[N]={2,3,0,0,4,5};
    int width=host->embed_dim;
    float *media=malloc(2*(size_t)width*sizeof(float));
    float *lg=malloc((size_t)dims.vocab*sizeof(float));
    float *other=malloc((size_t)dims.vocab*sizeof(float));
    assert(media && lg && other);
    const float *rows[N]={NULL,NULL,media,media+width,NULL,NULL};
    char prev[2][64]={{0}}, path[2][64];
    for (int turn=1; turn<=24; turn++) {
        for (int i=0; i<2*width; i++) media[i]=sinf((float)(i+turn)*0.17f);
        for (int worker=0; worker<2; worker++) {
            nt_experts *p=nt_experts_open_on_body(host,turn==1 ? NULL : prev[worker],NULL,&body);
            kv_cache *kv=nt_model_kv_new(arch,model,dims,N);
            assert(p && kv);
            assert(arch->doe->forward_embd(model,kv,ids,rows,N,0,lg,NULL,
                                          nt_experts_attend,nt_experts_observe,p)==NT_OK);
            if (!worker) memcpy(other,lg,(size_t)dims.vocab*sizeof(float));
            else assert(!memcmp(other,lg,(size_t)dims.vocab*sizeof(float)));
            int best=0;
            for(int i=1;i<dims.vocab;i++) if(lg[i]>lg[best]) best=i;
            nt_experts_choice(p,lg[best],lg[best]);
            nt_model_kv_free(arch,model,kv);
            if (worker) {
                nt_consolidation *job=NULL;
                assert((turn%4 ? nt_experts_submit(&p,&job) : nt_experts_submit_dream(&p,&job,6,-1))==NT_OK);
                assert(!p && job && nt_experts_await(&p,&job)==NT_OK && p && !job);
            } else {
                assert(nt_experts_finish(p,1)==NT_OK);
                if (turn%4==0) assert(nt_experts_dream(p,6,-1,1)==NT_OK);
            }
            snprintf(path[worker],sizeof path[worker],"media-%d-%d.gguf",worker,turn);
            assert(!nt_experts_save_portable(p,path[worker]));
            nt_experts_free(p);
            strcpy(prev[worker],path[worker]);
        }
        equal_files(path[0],path[1]);
        gguf_file *g=gguf_open(path[0]);
        assert(g && gguf_get_kv(g,"doe.turns")->val.u64==(uint64_t)turn &&
               gguf_get_kv(g,"doe.elections")->val.u64==(uint64_t)turn*N);
        if(turn>=4) assert(gguf_get_kv(g,"doe.dream.cycles")->val.u64==(uint64_t)turn/4);
        gguf_close(g);
    }
    /* Same placeholder ids, different media. All stages must actually read the rows. */
    for(int flip=0;flip<2;flip++) {
        nt_experts *p=nt_experts_open_on_body(host,prev[0],NULL,&body);
        kv_cache *kv=nt_model_kv_new(arch,model,dims,N);
        assert(p && kv);
        media[0]+=flip ? 2.0f : 0.0f;
        assert(arch->doe->forward_embd(model,kv,ids,rows,N,0,lg,NULL,
                                      nt_experts_attend,nt_experts_observe,p)==NT_OK);
        if (!flip) memcpy(other,lg,(size_t)dims.vocab*sizeof(float));
        else assert(memcmp(other,lg,(size_t)dims.vocab*sizeof(float)));
        nt_model_kv_free(arch,model,kv); nt_experts_free(p);
    }
    free(other); free(lg); free(media); bpe_free(tok); arch->free(model); gguf_close(host);
    puts("Media experts: 24 restarted turns, 6 private dreams, exact worker generations, changed-row control passed");
    return 0;
}
