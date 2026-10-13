#!/usr/bin/env python3
"""Compare the PEN7 harness against the pinned, unmodified native C forward.

No package dependencies. Supply the reference checkout and original/GGUF pair(s).
The optional --red-hand builds a wrong-RoPE family in a temporary directory and
requires the same full-vector gate to reject it. It never edits product sources.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True, help='1984/penelope.c')
parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('PEN7', 'GGUF'))
parser.add_argument('--red-hand', action='store_true')
parser.add_argument('--q8', action='store_true', help='Q8->F32 exact native oracle, plus measured original-F32 differences')
parser.add_argument('--sanitize', action='store_true')
parser.add_argument('--ownership', action='store_true', help='GNU linker allocation-failure/ownership gate')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = r'''
#define main pen_reference_cli
#define rmsnorm pen_reference_norm
#define bpe_encode pen_reference_encode
#include "REFERENCE_SOURCE"
#undef bpe_encode
#undef rmsnorm
#undef main
#include "harness/arch.h"
#include "examples/bpe.h"
#include <stdint.h>

static int failed, checks;
#define CHECK(c,s) do { checks++; if (!(c)) { fprintf(stderr,"FAIL: %s\n",s); failed++; } } while(0)
#ifdef PEN_TEST_OWNERSHIP
static long live_allocations;static int allocation_budget=-1;
void *__real_malloc(size_t);void *__real_calloc(size_t,size_t);void __real_free(void*);
void *__wrap_malloc(size_t n){if(allocation_budget==0)return NULL;if(allocation_budget>0)allocation_budget--;
    void*p=__real_malloc(n);if(p)live_allocations++;return p;}
void *__wrap_calloc(size_t n,size_t size){if(allocation_budget==0)return NULL;if(allocation_budget>0)allocation_budget--;
    void*p=__real_calloc(n,size);if(p)live_allocations++;return p;}
void __wrap_free(void*p){if(p)live_allocations--;__real_free(p);}
#endif
static Model load_reference(const char *path, void **owned) {
    Model m={0}; FILE *f=fopen(path,"rb");
    int32_t header[8], expected[8]={0x50454E37,BPE_VOCAB,NWORDS,DIM,HDIM,N_HEADS,N_LAYERS,MAX_SEQ};
    if(!f || fread(header,4,8,f)!=8 || memcmp(header,expected,32)) exit(2);
    size_t count=(size_t)total_param_count(); float *p=malloc(count*sizeof(float));
    if(!p || fread(p,sizeof(float),count,f)!=count || fgetc(f)!=EOF) exit(2);
    fclose(f); *owned=p;
#define TAKE(field,n) do {m.field=p;p+=(n);}while(0)
    TAKE(tok_emb,BPE_VOCAB*DIM);TAKE(pos_emb,MAX_SEQ*DIM);TAKE(final_norm,DIM);TAKE(lm_head,BPE_VOCAB*DIM);
    for(int l=0;l<N_LAYERS;l++) {
        TAKE(layers[l].attn_norm,DIM);TAKE(layers[l].wq,DIM*DIM);TAKE(layers[l].wk,DIM*DIM);
        TAKE(layers[l].wv,DIM*DIM);TAKE(layers[l].wo,DIM*DIM);TAKE(layers[l].wr,DIM*DIM);
        memcpy(m.layers[l].gate,p,2*sizeof(float));p+=2;TAKE(layers[l].ffn_norm,DIM);
        TAKE(layers[l].w_gate,HDIM*DIM);TAKE(layers[l].w_up,HDIM*DIM);TAKE(layers[l].w_down,DIM*HDIM);
    }
#undef TAKE
    return m;
}
static uint64_t tape_hash(void) {
    const unsigned char *p=(const unsigned char*)nt_tape_get(); uint64_t h=14695981039346656037ULL;
    for(size_t i=0;i<sizeof(nt_tape);i++){h^=p[i];h*=1099511628211ULL;}return h;
}
static void token_case(bpe_tokenizer *t, const char *s, int cap) {
    int a[8192],b[8192];int na=pen_reference_encode(s,a,cap),nb=bpe_encode(t,s,b,cap);
    CHECK(na==nb && !memcmp(a,b,(size_t)na*sizeof(int)),"native token ID vector");
    nb=bpe_encode_raw(t,s,b,cap);
    CHECK(na==nb && !memcmp(a,b,(size_t)na*sizeof(int)),"raw native token ID vector");
}
static int vector_equal(const float *a,const float *b,int n,const char *label) {
    double max=0;int diff=0,bad=0;
    for(int i=0;i<n;i++){double d=fabs((double)a[i]-b[i]);if(d>max)max=d;
        diff+=memcmp(a+i,b+i,sizeof(float))!=0;bad+=!isfinite(a[i])||!isfinite(b[i]);}
    printf("%s: values=%d differing_f32=%d max_abs=%.9g\n",label,n,diff,max);
    CHECK(!bad && !diff,label);return !bad && !diff;
}
static void quantify(const float *a,const float *b,int n,const char *label) {
    double max=0,sum=0,aa=0,bb=0,ab=0;int ia=0,ib=0;
    for(int i=0;i<n;i++){double d=(double)a[i]-b[i];if(fabs(d)>max)max=fabs(d);sum+=d*d;
        aa+=(double)a[i]*a[i];bb+=(double)b[i]*b[i];ab+=(double)a[i]*b[i];
        if(a[i]>a[ia])ia=i;if(b[i]>b[ib])ib=i;CHECK(isfinite(a[i])&&isfinite(b[i]),"finite quantized logits");}
    printf("%s: max_abs=%.9g rms=%.9g cosine=%.12g original_argmax=%d q8_argmax=%d\n",label,max,sqrt(sum/n),ab/sqrt(aa*bb),ia,ib);
}
static Model dequant_reference(gguf_file *gf, float **owned) {
    Model m={0};char name[128];int count=0;
#define GET(field,suffix) do{int ti=gguf_find_tensor(gf,suffix);if(ti<0)exit(2);m.field=gguf_dequant(gf,ti);if(!m.field)exit(2);owned[count++]=m.field;}while(0)
    GET(tok_emb,"token_embd.weight");GET(pos_emb,"position_embd.weight");GET(final_norm,"output_norm.weight");GET(lm_head,"output.weight");
    for(int l=0;l<N_LAYERS;l++){
#define B(field,suffix) do{snprintf(name,sizeof(name),"blk.%d." suffix ".weight",l);GET(layers[l].field,name);}while(0)
        B(attn_norm,"attn_norm");B(wq,"attn_q");B(wk,"attn_k");B(wv,"attn_v");B(wo,"attn_output");B(wr,"resonance");
        snprintf(name,sizeof(name),"blk.%d.resonance_gate.weight",l);int ti=gguf_find_tensor(gf,name);
        float *gate=gguf_dequant(gf,ti);if(!gate)exit(2);memcpy(m.layers[l].gate,gate,2*sizeof(float));free(gate);
        B(ffn_norm,"ffn_norm");B(w_gate,"ffn_gate");B(w_up,"ffn_up");B(w_down,"ffn_down");
#undef B
    }
#undef GET
    owned[count]=NULL;return m;
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    void *bytes=NULL;Model ref=load_reference(argv[1],&bytes);
    gguf_file *gf=gguf_open(argv[2]);nt_dims dims;const nt_arch *a=&nt_arch_penelope;
    void *m=gf?a->load(gf,&dims):NULL;bpe_tokenizer *t=bpe_load(argv[2]);
    if(!m||!t)return 2;
#ifdef PEN_TEST_OWNERSHIP
    int budgets[]={0,1,2,8,32,66};
    for(size_t i=0;i<sizeof(budgets)/sizeof(*budgets);i++){
        long before=live_allocations;allocation_budget=budgets[i];nt_dims untouched={7,8,9};
        void*bad=a->load(gf,&untouched);allocation_budget=-1;
        CHECK(!bad,"failed allocation refuses model");if(bad)a->free(bad);
        CHECK(live_allocations==before,"failed allocation releases every owned buffer");
        CHECK(untouched.n_layers==7&&untouched.kv_dim==8&&untouched.vocab==9,"failed load leaves dimensions untouched");
    }
    long prior=live_allocations;void*extra=a->load(gf,&dims);CHECK(extra!=NULL,"ordinary owned reload");
    a->free(extra);CHECK(live_allocations==prior,"successful model destruction releases every owned buffer");
#endif
    float *dequant_owned[100]={0};Model dq={0};int is_q8=0;
#ifdef PEN_TEST_Q8
    dq=dequant_reference(gf,dequant_owned);is_q8=1;
#endif
    CHECK(dims.vocab==BPE_VOCAB&&dims.n_layers==N_LAYERS&&dims.kv_dim==DIM,"native geometry");
    const char *cases[]={"","memory turns through the silence","Hello WORLD!  fire\nwater",
        "existential dread's present decision-making","Привет, мир. 海 火 🔥", "\xff\xfe\x80"};
    for(size_t i=0;i<sizeof(cases)/sizeof(*cases);i++)token_case(t,cases[i],8192);
    char all[256];for(int i=1;i<256;i++)all[i-1]=(char)i;all[255]=0;token_case(t,all,8192);
    char longtext[9001];for(int i=0;i<9000;i++)longtext[i]="Water FIRE 19\n"[i%14];longtext[9000]=0;
    token_case(t,longtext,8192);token_case(t,longtext,3);
    init_bpe_decode();
    /* Native C's 64-byte table drops long pieces and damages descendants.
     * GGUF retains their full recursive bytes. Build those independently from
     * the source pair IDs, preserving NUL by explicit lengths. */
    unsigned char *full[BPE_VOCAB]={0};size_t sizes[BPE_VOCAB]={0};int native_short=0;
    for(int i=0;i<BPE_VOCAB;i++){
        if(i<256){sizes[i]=1;full[i]=malloc(1);full[i][0]=(unsigned char)i;}
        else{int left=BPE_TABLE[i-256][0],right=BPE_TABLE[i-256][1];sizes[i]=sizes[left]+sizes[right];
            full[i]=malloc(sizes[i]);memcpy(full[i],full[left],sizes[left]);memcpy(full[i]+sizes[left],full[right],sizes[right]);}
        char *out=malloc(sizes[i]+1);int n=bpe_decode_token(t,i,out,(int)sizes[i]+1);
        CHECK(n==(int)sizes[i]&&!memcmp(out,full[i],sizes[i]),"complete recursive decoded token bytes");
        if(sizes[i]!=(size_t)bpe_str_len[i])native_short++;
        else CHECK(!memcmp(out,bpe_strs[i],sizes[i]),"untruncated native decoder bytes");
        free(out);
    }
    for(int i=0;i<BPE_VOCAB;i++)free(full[i]);
    printf("native fixed-buffer decoder truncated %d token surfaces; GGUF retained complete bytes\n",native_short);
    int ids[MAX_SEQ];for(int i=0;i<MAX_SEQ;i++)ids[i]=(i*97+31)%BPE_VOCAB;
    int prompt_n=pen_reference_encode(cases[1],ids,MAX_SEQ);
    float expected[BPE_VOCAB],actual[BPE_VOCAB],chunked[BPE_VOCAB],original[BPE_VOCAB];
    kv_cache *kv=kv_new(N_LAYERS,MAX_SEQ,DIM);if(!kv)return 2;
    uint64_t before=tape_hash();int lengths[]={1,prompt_n,33,MAX_SEQ};
    for(size_t k=0;k<sizeof(lengths)/sizeof(*lengths);k++){
        int n=lengths[k];forward(is_q8?&dq:&ref,ids,n,expected);
        if(is_q8){forward(&ref,ids,n,original);char label[80];snprintf(label,sizeof(label),"original versus Q8 context=%d",n);quantify(original,expected,BPE_VOCAB,label);}
        CHECK(a->forward(m,kv,ids,n,0,actual)==NT_OK,"full context execution");
        vector_equal(expected,actual,BPE_VOCAB,"native full logits");
        int pos=0;while(pos<n){int take=pos==0?7:31;if(take>n-pos)take=n-pos;
            CHECK(a->forward(m,kv,ids+pos,take,pos,chunked)==NT_OK,"chunk execution");pos+=take;}
        vector_equal(expected,chunked,BPE_VOCAB,"native chunked logits");
    }
    /* Fixed greedy continuation is exact against the same stored values.
     * Q8's original-F32 disagreements are observations recorded separately. */
    int n=prompt_n, original_disagreements=0;
    for(int step=0;step<16;step++){
        forward(is_q8?&dq:&ref,ids,n,expected);CHECK(a->forward(m,kv,ids,n,0,actual)==NT_OK,"greedy forward");
        int ia=0,ib=0;for(int j=1;j<BPE_VOCAB;j++){if(expected[j]>expected[ia])ia=j;if(actual[j]>actual[ib])ib=j;}
        if(is_q8){forward(&ref,ids,n,original);int io=0;for(int j=1;j<BPE_VOCAB;j++)if(original[j]>original[io])io=j;
            original_disagreements+=io!=ia;printf("teacher_forced_greedy step=%d original=%d q8=%d\n",step,io,ia);}
        CHECK(ia==ib,"native greedy ID");ids[n++]=ia;
    }
    if(is_q8)printf("Q8 original-F32 teacher-forced greedy disagreements=%d/16\n",original_disagreements);
    CHECK(tape_hash()==before,"complete global tape unchanged");
    CHECK(nt_tape_get()->n_params==0&&!nt_tape_get()->active&&!nt_tape_get()->count,"zero optimizer and tape entries");
    int last=gguf_find_tensor(gf,"blk.7.ffn_down.weight");uint32_t ndim=gf->tensors[last].ndim;
    gf->tensors[last].ndim=1;void *invalid=a->load(gf,&dims);
    CHECK(!invalid,"late invalid tensor refuses and releases partial model");if(invalid)a->free(invalid);
    gf->tensors[last].ndim=ndim;
    kv_free(kv);bpe_free(t);a->free(m);gguf_close(gf);free(bytes);
    for(int i=0;dequant_owned[i];i++)free(dequant_owned[i]);
    printf("PENELOPE_REFERENCE_%s (%d checks)\n",failed?"FAIL":"OK",checks);return !!failed;
}
'''
with tempfile.TemporaryDirectory(prefix='notorch-penelope-') as tmp:
    tmp = Path(tmp)
    c = tmp / 'reference.c'
    c.write_text(source.replace('REFERENCE_SOURCE', str(args.reference.resolve())))
    family = root / 'harness/arch_penelope.c'
    if args.red_hand:
        modified = family.read_text().replace('D, m->base, 1);', 'D, m->base, 0);')
        assert modified != family.read_text()
        family = tmp / 'wrong_rope.c'
        family.write_text(modified)
    binary = tmp / 'reference'
    cmd = [os.environ.get('CC', 'cc'), '-O2', '-std=gnu11', '-ffp-contract=off', '-pthread',
           '-I', str(root), str(c), str(family), str(root / 'harness/runtime.c'),
           str(root / 'examples/bpe.c'), str(root / 'gguf.c'), str(root / 'notorch.c'), '-lm', '-o', str(binary)]
    if args.q8:
        cmd.insert(1, '-DPEN_TEST_Q8')
    if args.sanitize:
        cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    if args.ownership:
        if not args.q8:
            parser.error('--ownership uses Q8 owned matrices; add --q8')
        cmd[1:1] = ['-DPEN_TEST_OWNERSHIP', '-Wl,--wrap=malloc', '-Wl,--wrap=calloc', '-Wl,--wrap=free']
    subprocess.run(cmd, check=True)
    for pen7, gguf in args.pair:
        result = subprocess.run([str(binary), str(Path(pen7).resolve()), str(Path(gguf).resolve())])
        expected = 1 if args.red_hand else 0
        if result.returncode != expected:
            raise SystemExit(f'expected exit {expected}, got {result.returncode}')
    if args.red_hand:
        print('PENELOPE_RED_HAND_CAUGHT')
