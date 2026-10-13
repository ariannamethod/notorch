/* PEN7 -> GGUF v3. F32 bits retained by default; explicit --q8_0 matrices.
 * cc -O2 -std=gnu11 -ffp-contract=off -pthread -I. tools/pen7_to_gguf.c \
 *    gguf.c sha256.c notorch.c -lm -o pen7_to_gguf
 * ./pen7_to_gguf [--q8_0] source.bin destination.gguf
 *
 * The source file carries 8 little-endian int32 values and 92 float tensors.
 * Its tokenizer lives in source: tools/pen7_bpe.h pins both original tables.
 * GGUF supplies explicit geometry, numerical conventions and native byte BPE.
 * An existing destination is never replaced. A temporary sibling is validated
 * tensor-for-tensor before a no-replace hard link publishes the complete file.
 */
#include "gguf.h"
#include "notorch.h"
#include "sha256.h"
#include "pen7_bpe.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { BV=2048, WV=1984, D=448, H=896, NH=7, NL=8, CTX=256, NT=92 };
#define PEN7_PARAMS UINT64_C(19619280)
#define PEN7_BYTES (UINT64_C(32) + PEN7_PARAMS * 4)

typedef struct {
    char name[80];
    uint32_t ndim, dtype;
    uint64_t shape[2], count, source_offset;
} tensor;

static void hex_digest(const void *p, size_t n, char out[65]) {
    unsigned char digest[32];
    static const char hex[] = "0123456789abcdef";
    if (nt_sha256(p, n, digest)) { out[0] = 0; return; }
    for (int i=0; i<32; ++i) {
        out[2*i] = hex[digest[i]>>4]; out[2*i+1] = hex[digest[i]&15];
    }
    out[64] = 0;
}

static void add_tensor(tensor ts[NT], int *n, uint64_t *off,
                       const char *name, uint64_t a, uint64_t b) {
    tensor *t = &ts[(*n)++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->ndim = b ? 2 : 1; t->shape[0] = a; t->shape[1] = b;
    t->count = a*(b ? b : 1); t->source_offset = *off;
    *off += t->count*4;
}

static int make_directory(tensor ts[NT], int q8) {
    int n=0;
    uint64_t off=32;
    add_tensor(ts,&n,&off,"token_embd.weight",D,BV);
    add_tensor(ts,&n,&off,"position_embd.weight",D,CTX);
    add_tensor(ts,&n,&off,"output_norm.weight",D,0);
    add_tensor(ts,&n,&off,"output.weight",D,BV);
    static const char *suffix[] = {
        "attn_norm.weight","attn_q.weight","attn_k.weight","attn_v.weight",
        "attn_output.weight","resonance.weight","resonance_gate.weight",
        "ffn_norm.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"
    };
    const uint64_t a[] = {D,D,D,D,D,D,2,D,D,D,H};
    const uint64_t b[] = {0,D,D,D,D,D,0,0,H,H,D};
    for (int layer=0; layer<NL; ++layer) for (int i=0; i<11; ++i) {
        char name[80];
        snprintf(name,sizeof(name),"blk.%d.%s",layer,suffix[i]);
        add_tensor(ts,&n,&off,name,a[i],b[i]);
    }
    for (int i=0; i<n; ++i)
        ts[i].dtype=q8 && ts[i].ndim==2 && strcmp(ts[i].name,"position_embd.weight")
                    ? GGUF_TYPE_Q8_0 : GGUF_TYPE_F32;
    return n==NT && off==PEN7_BYTES ? 0 : -1;
}

/* GPT-2's byte alphabet is only a lossless spelling of bytes in GGUF strings.
 * The declared encoder is PEN7's rank-pass byte BPE, with no GPT-2 regex split. */
static int make_tokenizer(char *tokens[BV], char *merges[PEN7_MERGES]) {
    int extra=0;
    for (int byte=0; byte<256; ++byte) {
        int printable=(byte>=33 && byte<=126)||(byte>=161 && byte<=172)||byte>=174;
        int cp=printable ? byte : 256+extra++;
        tokens[byte]=calloc(3,1);
        if (!tokens[byte]) return -1;
        if (cp<128) tokens[byte][0]=(char)cp;
        else { tokens[byte][0]=(char)(0xc0|(cp>>6)); tokens[byte][1]=(char)(0x80|(cp&63)); }
    }
    for (int i=0; i<PEN7_MERGES; ++i) {
        int l=pen7_merge_pairs[2*i], r=pen7_merge_pairs[2*i+1];
        if (l<0 || r<0 || l>=256+i || r>=256+i) return -1;
        size_t a=strlen(tokens[l]), b=strlen(tokens[r]);
        if (a+b>512) return -1;
        tokens[256+i]=malloc(a+b+1); merges[i]=malloc(a+b+2);
        if (!tokens[256+i] || !merges[i]) return -1;
        memcpy(tokens[256+i],tokens[l],a); memcpy(tokens[256+i]+a,tokens[r],b+1);
        memcpy(merges[i],tokens[l],a); merges[i][a]=' ';
        memcpy(merges[i]+a+1,tokens[r],b+1);
    }
    return 0;
}

static void free_tokenizer(char *tokens[BV], char *merges[PEN7_MERGES]) {
    for (int i=0; i<BV; ++i) free(tokens[i]);
    for (int i=0; i<PEN7_MERGES; ++i) free(merges[i]);
}

static int metadata(gguf_writer *w, const char *sha, int q8,
                    char *tokens[BV], char *merges[PEN7_MERGES]) {
    const char *name="PEN7", *repo=NULL, *revision=NULL;
    if (!strcmp(sha,"afff67c780a7c78fdc1f76e521dfda30d65e68b95fe5bd324bb392b84ba3cd38")) {
        name="Haiku 1 / Penelope PEN7";
        repo="https://github.com/ariannamethod/1984";
        revision="a0d20019587ff14e5516eb050ba0876d6edd0baa";
    } else if (!strcmp(sha,"d675df2ee3a150d63bc73935a84060280d0f6b0aa84283e4b78ab905180b576c")) {
        name="Haiku 2 / NanoJanus PEN7";
        repo="https://github.com/ariannamethod/janus";
        revision="5a945a9d7c1dc9abfa217f685f0a81455f9ac493";
    }
    int rc=0;
    rc |= gguf_write_kv_str(w,"general.architecture","penelope");
    rc |= gguf_write_kv_str(w,"general.name",name);
    rc |= gguf_write_kv_u32(w,"general.file_type",q8 ? 7 : 0);
    if (q8) rc |= gguf_write_kv_u32(w,"general.quantization_version",2);
    rc |= gguf_write_kv_str(w,"penelope.source.format","PEN7");
    rc |= gguf_write_kv_str(w,"penelope.source.sha256",sha);
    rc |= gguf_write_kv_u64(w,"penelope.source.bytes",PEN7_BYTES);
    rc |= gguf_write_kv_u64(w,"penelope.parameter_count",PEN7_PARAMS);
    rc |= gguf_write_kv_str(w,"penelope.conversion",q8 ? "q8_0-matrices-v1" : "f32-lossless-v1");
    if (repo) rc |= gguf_write_kv_str(w,"penelope.source.repository",repo);
    if (revision) rc |= gguf_write_kv_str(w,"penelope.source.revision",revision);
    rc |= gguf_write_kv_u32(w,"penelope.embedding_length",D);
    rc |= gguf_write_kv_u32(w,"penelope.context_length",CTX);
    rc |= gguf_write_kv_u32(w,"penelope.vocab_size",BV);
    rc |= gguf_write_kv_u32(w,"penelope.word_vocab_size",WV);
    rc |= gguf_write_kv_u32(w,"penelope.block_count",NL);
    rc |= gguf_write_kv_u32(w,"penelope.feed_forward_length",H);
    rc |= gguf_write_kv_u32(w,"penelope.attention.head_count",NH);
    rc |= gguf_write_kv_u32(w,"penelope.attention.head_count_kv",NH);
    rc |= gguf_write_kv_u32(w,"penelope.attention.head_dim",D/NH);
    rc |= gguf_write_kv_f32(w,"penelope.attention.layer_norm_rms_epsilon",1e-5f);
    rc |= gguf_write_kv_f32(w,"penelope.rope.freq_base",10000.0f);
    rc |= gguf_write_kv_u32(w,"penelope.rope.dimension_count",D/NH);
    rc |= gguf_write_kv_str(w,"penelope.rope.layout","split-half");
    rc |= gguf_write_kv_bool(w,"penelope.position_embedding",1);
    rc |= gguf_write_kv_str(w,"penelope.resonance","dense-gated");
    rc |= gguf_write_kv_f32(w,"penelope.silu_cutoff",-20.0f);
    rc |= gguf_write_kv_str(w,"tokenizer.ggml.model","penelope");
    rc |= gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",0);
    rc |= gguf_write_kv_bool(w,"tokenizer.ggml.add_eos_token",0);
    rc |= gguf_write_kv_i32(w,"tokenizer.ggml.bos_token_id",-1);
    rc |= gguf_write_kv_i32(w,"tokenizer.ggml.eos_token_id",-1);
    rc |= gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",(const char *const *)tokens,BV);
    rc |= gguf_write_kv_str_array(w,"tokenizer.ggml.merges",(const char *const *)merges,PEN7_MERGES);
    rc |= gguf_write_kv_str(w,"tokenizer.penelope.algorithm","byte-rank-pass-v1");
    rc |= gguf_write_kv_str(w,"tokenizer.penelope.source_contract",
                           "ariannamethod/1984:penelope.c@a0d20019587ff14e5516eb050ba0876d6edd0baa");
    rc |= gguf_write_kv_bool(w,"tokenizer.penelope.lowercase_ascii",1);
    rc |= gguf_write_kv_u32(w,"tokenizer.penelope.max_input_bytes",8191);
    rc |= gguf_write_kv_i32_array(w,"tokenizer.penelope.merge_pairs",pen7_merge_pairs,PEN7_MERGES*2);
    rc |= gguf_write_kv_str(w,"tokenizer.penelope.sha256",PEN7_BPE_SHA256);
    return rc;
}

/* Packed Q8_0 tensors use canonical NoTorch arithmetic. A row-sized buffer
 * bounds conversion scratch independently of the complete checkpoint. */
static int write_quantized(gguf_writer *w, const tensor *t, const float *src) {
    unsigned char packed[H/32*34];
    if (t->shape[0]>H || t->shape[0]%32) return -1;
    uint64_t row_bytes=t->shape[0]/32*34;
    if (gguf_write_tensor_begin(w,t->name)) return -1;
    for (uint64_t row=0; row<t->shape[1]; ++row) {
        const float *x=src+row*t->shape[0];
        /* A finite reciprocal and finite half scale are required before the
         * canonical quantizer converts scaled floats to integer quants. */
        for (uint64_t b=0; b<t->shape[0]; b+=32) {
            float amax=0;
            for (int i=0; i<32; ++i) if (fabsf(x[b+i])>amax) amax=fabsf(x[b+i]);
            float d=amax/127.0f;
            if (d>=65520.0f || (d!=0 && !isfinite(1.0f/d))) {
                fprintf(stderr,"pen7: Q8_0 scale/reciprocal outside finite range in %s row %llu block %llu\n",
                        t->name,(unsigned long long)row,(unsigned long long)(b/32));
                return -1;
            }
        }
        if (nt_quantize_row(x,packed,(int)t->shape[0],GGUF_TYPE_Q8_0) ||
            gguf_write_tensor_chunk(w,packed,row_bytes)) return -1;
    }
    return gguf_write_tensor_end(w);
}

static int verify(const char *path, const unsigned char *source,
                  const tensor ts[NT], char *tokens[BV],
                  double *max_error, double *sum_sq) {
    gguf_file *g=gguf_open(path);
    if (!g) return -1;
    int bad=g->version!=3 || g->n_tensors!=NT || strcmp(g->arch,"penelope");
    for (int i=0; !bad && i<NT; ++i) {
        const gguf_tensor_info *t=&g->tensors[i];
        uint64_t bytes=gguf_type_size(ts[i].dtype,ts[i].count);
        bad=strcmp(t->name,ts[i].name) || t->dtype!=ts[i].dtype ||
            t->ndim!=ts[i].ndim || t->shape[0]!=ts[i].shape[0] ||
            (t->ndim==2 && t->shape[1]!=ts[i].shape[1]) ||
            t->n_elements!=ts[i].count || t->offset>g->data_size ||
            bytes>g->data_size-t->offset;
        if (!bad && ts[i].dtype==GGUF_TYPE_F32)
            bad=memcmp(g->data+t->offset,source+ts[i].source_offset,(size_t)bytes)!=0;
        if (!bad && ts[i].dtype==GGUF_TYPE_Q8_0) {
            unsigned char expected[H/32*34];
            uint64_t row_bytes=t->shape[0]/32*34;
            for (uint64_t row=0; !bad && row<t->shape[1]; ++row) {
                const float *x=(const float *)(source+ts[i].source_offset+row*t->shape[0]*4);
                bad=nt_quantize_row(x,expected,(int)t->shape[0],GGUF_TYPE_Q8_0) ||
                    memcmp(expected,g->data+t->offset+row*row_bytes,(size_t)row_bytes);
            }
            float *values=gguf_dequant(g,i);
            if (!values) bad=1;
            for (uint64_t j=0; !bad && j<ts[i].count; ++j) {
                float original; memcpy(&original,source+ts[i].source_offset+4*j,4);
                if (!isfinite(values[j])) { bad=1; break; }
                double error=fabs((double)values[j]-original);
                if (error>*max_error) *max_error=error;
                *sum_sq+=error*error;
            }
            free(values);
        }
    }
    gguf_close(g);
    int n=0;
    int32_t *pairs=gguf_read_i32_array(path,"tokenizer.penelope.merge_pairs",&n);
    bad |= !pairs || n!=PEN7_MERGES*2;
    if (pairs && n==PEN7_MERGES*2) bad |= memcmp(pairs,pen7_merge_pairs,sizeof(pen7_merge_pairs))!=0;
    free(pairs);
    char **v=gguf_read_str_array(path,"tokenizer.ggml.tokens",&n);
    bad |= !v || n!=BV;
    if (v) { for (int i=0; i<n; ++i) {
        if (i<BV && strcmp(v[i],tokens[i])) bad=1;
        free(v[i]);
    } free(v); }
    return bad ? -1 : 0;
}

static int hash_file(const char *path, char out[65], uint64_t *size) {
    int fd=open(path,O_RDONLY);
    struct stat st;
    if (fd<0) return -1;
    if (fstat(fd,&st) || st.st_size<=0) { close(fd); return -1; }
    void *p=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if (p==MAP_FAILED) return -1;
    hex_digest(p,(size_t)st.st_size,out);
    munmap(p,(size_t)st.st_size); *size=(uint64_t)st.st_size;
    return out[0] ? 0 : -1;
}

int main(int argc, char **argv) {
    int q8=argc==4 && !strcmp(argv[1],"--q8_0");
    if ((!q8 && argc!=3) || (argc>1 && argv[1][0]=='-' && !q8)) {
        fprintf(stderr,"usage: %s [--q8_0] source.bin destination.gguf\n",argv[0]); return 2;
    }
    const char *input=argv[1+q8], *output=argv[2+q8];
    const uint32_t host=1;
    if (*(const unsigned char *)&host!=1 || sizeof(float)!=4 || FLT_RADIX!=2 ||
        FLT_MANT_DIG!=24 || FLT_MAX_EXP!=128) {
        fprintf(stderr,"pen7: conversion requires little-endian IEEE binary32\n"); return 1;
    }
    struct stat st;
    if (!lstat(output,&st) || errno!=ENOENT) {
        fprintf(stderr,"pen7: destination already exists or cannot be inspected\n"); return 1;
    }
    int fd=open(input,O_RDONLY);
    if (fd<0) { perror("pen7: input"); return 1; }
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size!=PEN7_BYTES) {
        fprintf(stderr,"pen7: source must be a regular file of exactly %llu bytes\n",
                (unsigned long long)PEN7_BYTES); close(fd); return 1;
    }
    const unsigned char *source=mmap(NULL,(size_t)PEN7_BYTES,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if (source==MAP_FAILED) { perror("pen7: mmap"); return 1; }
    const uint32_t expected[8]={0x50454e37,BV,WV,D,H,NH,NL,CTX};
    if (memcmp(source,expected,sizeof(expected))) {
        fprintf(stderr,"pen7: unsupported PEN7 magic or geometry\n");
        munmap((void *)source,(size_t)PEN7_BYTES); return 1;
    }
    for (uint64_t off=32; off<PEN7_BYTES; off+=4) {
        uint32_t bits; memcpy(&bits,source+off,4);
        if ((bits&UINT32_C(0x7f800000))==UINT32_C(0x7f800000)) {
            fprintf(stderr,"pen7: non-finite parameter at index %llu\n",(unsigned long long)((off-32)/4));
            munmap((void *)source,(size_t)PEN7_BYTES); return 1;
        }
    }
    char source_sha[65], bpe_sha[65];
    hex_digest(source,(size_t)PEN7_BYTES,source_sha);
    hex_digest(pen7_merge_pairs,sizeof(pen7_merge_pairs),bpe_sha);
    if (!source_sha[0] || strcmp(bpe_sha,PEN7_BPE_SHA256)) {
        fprintf(stderr,"pen7: SHA256 or pinned tokenizer identity failed\n");
        munmap((void *)source,(size_t)PEN7_BYTES); return 1;
    }
    tensor ts[NT]; char *tokens[BV]={0}, *merges[PEN7_MERGES]={0};
    int rc=1; char *temp=NULL; gguf_writer *w=NULL;
    if (make_directory(ts,q8) || make_tokenizer(tokens,merges)) goto done;
    size_t cap=strlen(output)+32;
    temp=malloc(cap); if (!temp) goto done;
    snprintf(temp,cap,"%s.partial.XXXXXX",output);
    fd=mkstemp(temp);
    if (fd<0) { perror("pen7: temporary output"); goto done; }
    close(fd);
    w=gguf_write_open(temp); if (!w) goto done;
    if (metadata(w,source_sha,q8,tokens,merges)) goto done;
    for (int i=0; i<NT; ++i)
        if (gguf_write_tensor_decl(w,ts[i].name,ts[i].ndim,ts[i].shape,ts[i].dtype)) goto done;
    for (int i=0; i<NT; ++i) {
        const float *src=(const float *)(source+ts[i].source_offset);
        int err=ts[i].dtype==GGUF_TYPE_Q8_0 ? write_quantized(w,&ts[i],src) :
                gguf_write_tensor(w,ts[i].name,src,ts[i].count*4);
        if (err) goto done;
    }
    { int err=gguf_write_close(w); w=NULL; if (err) goto done; }
    double max_error=0, sum_sq=0;
    if (verify(temp,source,ts,tokens,&max_error,&sum_sq)) { fprintf(stderr,"pen7: GGUF round-trip/rounding failed\n"); goto done; }
    char recheck[65], output_sha[65]; uint64_t bytes=0;
    hex_digest(source,(size_t)PEN7_BYTES,recheck);
    if (strcmp(recheck,source_sha)) { fprintf(stderr,"pen7: source changed during conversion\n"); goto done; }
    if (hash_file(temp,output_sha,&bytes)) goto done;
    if (chmod(temp,0644)) { perror("pen7: output permissions"); goto done; }
    if (link(temp,output)) { perror("pen7: publish without replacement"); goto done; }
    printf("{\"source_sha256\":\"%s\",\"gguf_sha256\":\"%s\",\"gguf_bytes\":%llu,"
           "\"tensors\":%d,\"parameters\":%llu,\"dtype\":\"%s\",\"bit_exact\":%s,"
           "\"storage_verified\":true,\"max_parameter_abs_error\":%.17g,\"rms_parameter_error\":%.17g,"
           "\"tokenizer_sha256\":\"%s\"}\n",source_sha,output_sha,(unsigned long long)bytes,
           NT,(unsigned long long)PEN7_PARAMS,q8 ? "Q8_0 matrices; F32 positions/norms/gates" : "F32",
           q8 ? "false" : "true",max_error,sqrt(sum_sq/PEN7_PARAMS),bpe_sha);
    rc=0;
done:
    if (w) gguf_write_abort(w);
    if (temp) { unlink(temp); free(temp); }
    free_tokenizer(tokens,merges);
    munmap((void *)source,(size_t)PEN7_BYTES);
    if (rc) fprintf(stderr,"pen7: no output published\n");
    return rc;
}
