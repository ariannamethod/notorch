/* Native source-layout Qwen3.5 converter. BF16/F32 bytes stay unchanged. */
#include "gguf.h"
#include "sha256.h"
#include "tools/narrow_json.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int fail(const char *what, const char *detail) {
    fprintf(stderr, "qwen35_to_gguf: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return 1;
}
static int u32_at(const char *obj, const char *key, uint32_t *out) {
    long long v;
    if (nj_int(nj_get(obj, key), &v) || v < 0 || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}
static gguf_writer *start(const char *out, char *partial, size_t cap) {
    struct stat st;
    if (snprintf(partial, cap, "%s.partial", out) >= (int)cap) return NULL;
    if (!stat(out, &st)) { fail("output exists", out); return NULL; }
    int fd = open(partial, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { perror(partial); return NULL; }
    close(fd);
    gguf_writer *w = gguf_write_open(partial);
    if (!w) unlink(partial);
    return w;
}
static int publish(const char *partial, const char *out, int tensors) {
    int fd = open(partial, O_RDONLY), ok = 0;
    struct stat st;
    unsigned char d[32];
    if (fd >= 0 && !fsync(fd) && !fstat(fd, &st) && st.st_size > 0) {
        void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ok = map != MAP_FAILED && !nt_sha256(map, (size_t)st.st_size, d);
        if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    }
    if (fd >= 0) close(fd);
    if (!ok || link(partial, out)) { perror(out); unlink(partial); return 1; }
    unlink(partial);
    printf("{\"tensors\": %d, \"sha256\": \"", tensors);
    for (int i = 0; i < 32; i++) printf("%02x", d[i]);
    printf("\"}\n");
    return 0;
}

typedef struct {char *name;const char *logical;uint64_t lo,hi,shape[5];uint32_t ndim,dtype;int shard;} tensor;
#define MAX_SHARDS 64
static int by_name(const void *a,const void *b){return strcmp(((const tensor*)a)->name,((const tensor*)b)->name);}
static int convert_model(const char *snapshot,const char *out,int vision) {
    char path[4096],partial[4200],key[128];char *config=NULL,*header=NULL,*index=NULL,*shards[MAX_SHARDS]={0};
    gguf_writer *w=NULL;FILE *src[MAX_SHARDS]={0};uint64_t data_at[MAX_SHARDS]={0};tensor *ts=NULL;
    int nt=0,bad=0,rc=1,head=0,expected=2,ns=0;
    snprintf(path,sizeof(path),"%s/config.json",snapshot);config=nj_load(path,NULL);if(!config)goto done;
    const char *c=nj_get(config,"text_config");if(!c)c=config;
    if(vision) {
        const char *v=nj_get(config,"vision_config"),*r=nj_get(c,"rope_parameters");
        const char *from[]={"depth","hidden_size","intermediate_size","num_heads","out_hidden_size","patch_size",
            "temporal_patch_size","spatial_merge_size","num_position_embeddings","in_channels"};
        const char *to[]={"block_count","embedding_length","feed_forward_length","attention.head_count","output_length","patch_size",
            "temporal_patch_size","merge_size","position_count","channels"};
        uint32_t vals[10],width,hd,ids[3]; double frac;
        if(!nj_is(nj_get(config,"model_type"),"qwen3_5") || !nj_is(nj_get(v,"hidden_act"),"gelu_pytorch_tanh") ||
           !nj_is(nj_get(r,"rope_type"),"default") || !nj_get(r,"mrope_interleaved") ||
           strncmp(nj_get(r,"mrope_interleaved"),"true",4))goto done;
        for(int i=0;i<10;i++)if(u32_at(v,from[i],&vals[i]) || !vals[i])goto done;
        if(vals[0]>128 || vals[9]!=3 || vals[6]!=2 || vals[7]!=2 ||
           u32_at(c,"hidden_size",&width) || width!=vals[4] || u32_at(c,"head_dim",&hd) ||
           nj_number(nj_get(r,"partial_rotary_factor"),&frac) || !isfinite(frac) || frac<=0 || frac>1 ||
           u32_at(config,"image_token_id",&ids[0]) || u32_at(config,"vision_start_token_id",&ids[1]) ||
           u32_at(config,"vision_end_token_id",&ids[2]))goto done;
        const char *ds=nj_get(v,"deepstack_visual_indexes");
        if(ds && (nj_first(ds,&bad) || bad))goto done;
        snprintf(path,sizeof(path),"%s/processor_config.json",snapshot);
        char *processor=access(path,F_OK)?NULL:nj_load(path,NULL);
        if(!processor){snprintf(path,sizeof(path),"%s/preprocessor_config.json",snapshot);processor=nj_load(path,NULL);}
        const char *ip=nj_get(processor,"image_processor");if(!ip)ip=processor;
        uint32_t minpx,maxpx,patch,merge,temp,resample; double scale;
        int valid=ip && !u32_at(nj_get(ip,"size"),"shortest_edge",&minpx) &&
            !u32_at(nj_get(ip,"size"),"longest_edge",&maxpx) && !u32_at(ip,"patch_size",&patch) &&
            !u32_at(ip,"merge_size",&merge) && !u32_at(ip,"temporal_patch_size",&temp) &&
            !u32_at(ip,"resample",&resample) && resample==3 && patch==vals[5] && merge==2 && temp==2 &&
            minpx>0 && maxpx>=minpx && !nj_number(nj_get(ip,"rescale_factor"),&scale) && fabs(scale-1.0/255.0)<1e-12;
        const char *flags[]={"do_resize","do_rescale","do_normalize","do_convert_rgb"};
        for(int i=0;i<4;i++) { const char *f=nj_get(ip,flags[i]); if(!f || strncmp(f,"true",4))valid=0; }
        const char *stats[]={"image_mean","image_std"};
        for(int i=0;i<2;i++) { int n=0,b=0; double x;
            for(const char *a=nj_first(nj_get(ip,stats[i]),&b);a;a=nj_next(a,']',&b),n++)
                if(nj_number(a,&x) || x!=0.5)valid=0;
            if(n!=3 || b)valid=0;
        }
        free(processor); if(!valid || !(w=start(out,partial,sizeof(partial))))goto done;
        gguf_write_kv_str(w,"general.architecture","qwen35_vision");
        gguf_write_kv_str(w,"general.name","Qwen3.5 vision, source tensor layout");
        gguf_write_kv_u32(w,"qwen35_vision.source_layout_version",1);
        for(int i=0;i<10;i++){snprintf(key,sizeof(key),"qwen35_vision.%s",to[i]);gguf_write_kv_u32(w,key,vals[i]);}
        gguf_write_kv_u32(w,"qwen35_vision.min_pixels",minpx);gguf_write_kv_u32(w,"qwen35_vision.max_pixels",maxpx);
        gguf_write_kv_u32(w,"qwen35_vision.text_rope_dim",(uint32_t)(hd*frac));
        const char *keys[]={"image_token_id","vision_start_token_id","vision_end_token_id"};
        for(int i=0;i<3;i++){snprintf(key,sizeof(key),"qwen35_vision.%s",keys[i]);gguf_write_kv_u32(w,key,ids[i]);}
        int count=0; uint32_t section;
        for(const char *a=nj_first(nj_get(r,"mrope_section"),&bad);a;a=nj_next(a,']',&bad),count++) {
            long long x;if(count>=3 || nj_int(a,&x) || x<0 || x>256)goto done;section=(uint32_t)x;
            snprintf(key,sizeof(key),"qwen35_vision.mrope_section.%d",count);gguf_write_kv_u32(w,key,section);
        }
        if(bad || count!=3)goto done;
        double theta=10000; const char *vr=nj_get(v,"rope_parameters");
        if(vr && (!nj_is(nj_get(vr,"rope_type"),"axial") || nj_number(nj_get(vr,"rope_theta"),&theta)))goto done;
        if(!isfinite(theta) || theta<=0)goto done;
        gguf_write_kv_f32(w,"qwen35_vision.rope.freq_base",(float)theta);
        expected=9+12*(int)vals[0];
    } else {
    const char *bias=nj_get(c,"attention_bias"),*rope=nj_get(c,"rope_parameters");
    if(!nj_is(nj_get(c,"model_type"),"qwen3_5_text") || !nj_is(nj_get(c,"hidden_act"),"silu") ||
       !bias || strncmp(bias,"false",5) || !nj_is(nj_get(rope,"rope_type"),"default"))goto done;
    const char *from[]={"num_hidden_layers","hidden_size","intermediate_size","vocab_size","max_position_embeddings",
        "num_attention_heads","num_key_value_heads","head_dim","linear_num_key_heads","linear_num_value_heads",
        "linear_key_head_dim","linear_value_head_dim","linear_conv_kernel_dim"};
    const char *to[]={"block_count","embedding_length","feed_forward_length","vocab_size","context_length",
        "attention.head_count","attention.head_count_kv","attention.head_dim","linear.key_heads","linear.value_heads",
        "linear.key_dim","linear.value_dim","linear.conv_kernel"};
    uint32_t vals[13];double eps,base,fraction;
    for(int i=0;i<13;i++)if(u32_at(c,from[i],&vals[i]) || !vals[i])goto done;
    if(vals[0]+20>GGUF_MAX_KV || nj_number(nj_get(c,"rms_norm_eps"),&eps) ||
       nj_number(nj_get(rope,"rope_theta"),&base) || nj_number(nj_get(rope,"partial_rotary_factor"),&fraction) ||
       !isfinite(eps) || eps<=0 || !isfinite(base) || base<=0 || !isfinite(fraction) || fraction<=0 || fraction>1)goto done;
    if(!(w=start(out,partial,sizeof(partial))))goto done;
    gguf_write_kv_str(w,"general.architecture","qwen35");
    gguf_write_kv_str(w,"general.name","Qwen3.5 dense text, source tensor layout");
    gguf_write_kv_u32(w,"qwen35.source_layout_version",1);
    /* Keep metadata order identical to the initial independent reference pack. */
    for(int i=0;i<13;i++) {
        snprintf(key,sizeof(key),"qwen35.%s",to[i]);gguf_write_kv_u32(w,key,vals[i]);
        if(i==7)gguf_write_kv_u32(w,"qwen35.rope.dimension_count",(uint32_t)(vals[7]*fraction));
    }
    int l=0;
    for(const char *v=nj_first(nj_get(c,"layer_types"),&bad);v;v=nj_next(v,']',&bad),l++) {
        int full=nj_is(v,"full_attention");if(!full && !nj_is(v,"linear_attention"))goto done;
        expected+=full?11:14;snprintf(key,sizeof(key),"qwen35.layer.%d.full_attention",l);gguf_write_kv_u32(w,key,full);
    }
    if(bad || l!=(int)vals[0])goto done;
    gguf_write_kv_f32(w,"qwen35.rms_epsilon",(float)eps);gguf_write_kv_f32(w,"qwen35.rope.freq_base",(float)base);
    }
    /* model.safetensors, or the shards model.safetensors.index.json names: Qwen's own
     * 0.8B and 2B ship one shard called -00001-of-00001, 4B and 9B several. With an
     * index, every text tensor must be found in the shard the index puts it in. */
    const char *map=NULL;
    snprintf(path,sizeof(path),"%s/model.safetensors",snapshot);
    if(access(path,F_OK)) {
        snprintf(path,sizeof(path),"%s/model.safetensors.index.json",snapshot);
        if(!(index=nj_load(path,NULL)) || !(map=nj_get(index,"weight_map")))goto done;
        for(const char *k=nj_first(map,&bad),*v;k && (v=nj_value(k,&bad));k=nj_next(v,'}',&bad)) {
            char *file=nj_string(v);size_t len=file?strlen(file):0;int seen=0;
            if(!file || len<13 || strchr(file,'/') || strcmp(file+len-12,".safetensors")){free(file);goto done;}
            for(int s=0;s<ns;s++)if(!strcmp(shards[s],file))seen=1;
            if(seen){free(file);continue;}
            if(ns==MAX_SHARDS){free(file);goto done;}
            shards[ns++]=file;
        }
        if(bad || !ns)goto done;
    } else if(!(shards[ns++]=strdup("model.safetensors")))goto done;
    for(int s=0;s<ns;s++) {
    uint64_t hs=0;struct stat st;
    snprintf(path,sizeof(path),"%s/%s",snapshot,shards[s]);
    free(header);header=NULL;
    src[s]=fopen(path,"rb");
    if(!src[s] || fstat(fileno(src[s]),&st) || st.st_size<8 || fread(&hs,8,1,src[s])!=1 || hs>(64u<<20) ||
       hs>(uint64_t)st.st_size-8 || !(header=malloc((size_t)hs+1)) || fread(header,1,hs,src[s])!=hs || memchr(header,0,hs))goto done;
    header[hs]=0;data_at[s]=8+hs;
    for(const char *k=nj_first(header,&bad),*v;k && (v=nj_value(k,&bad));k=nj_next(v,'}',&bad)) {
        char *name=nj_string(k);if(!name)goto done;
        int skip=vision ? (!strncmp(name,"model.visual.",13)?13:0) : !strncmp(name,"model.language_model.",21)?21:
            (nj_is(nj_get(config,"model_type"),"qwen3_5_text") && !strncmp(name,"model.",6)?6:0);
        if(!skip && (vision || strcmp(name,"lm_head.weight"))){free(name);continue;}
        if(map){char *where=nj_string(nj_get(map,name));int here=where && !strcmp(where,shards[s]);free(where);if(!here){free(name);goto done;}}
        if(!strcmp(name,"lm_head.weight"))head++;
        tensor *next=realloc(ts,(size_t)(nt+1)*sizeof(*ts));if(!next){free(name);goto done;}ts=next;
        tensor *t=&ts[nt++];memset(t,0,sizeof(*t));t->name=name;t->logical=name+skip;t->shard=s;
        const char *dtype=nj_get(v,"dtype");uint64_t width=0,count=1;
        if(nj_is(dtype,"F32")){t->dtype=GGUF_TYPE_F32;width=4;}
        if(nj_is(dtype,"BF16")){t->dtype=GGUF_TYPE_BF16;width=2;}
        if(!width)goto done;
        int b=0;long long dim,lo,hi;
        for(const char *s=nj_first(nj_get(v,"shape"),&b);s;s=nj_next(s,']',&b)) {
            if(t->ndim==(vision?5u:3u) || nj_int(s,&dim) || dim<1 || count>UINT64_MAX/(uint64_t)dim)goto done;
            t->shape[t->ndim++]=(uint64_t)dim;count*=dim;
        }
        const char *o=nj_first(nj_get(v,"data_offsets"),&b),*end=o?nj_next(o,']',&b):NULL;
        if(b || !t->ndim || !end || nj_int(o,&lo) || nj_int(end,&hi) || nj_next(end,']',&b) || b || lo<0 || hi<lo ||
           count>UINT64_MAX/width || (uint64_t)(hi-lo)!=count*width || (uint64_t)hi>(uint64_t)st.st_size-8-hs)goto done;
        if(vision && !strcmp(t->logical,"patch_embed.proj.weight")) {
            uint32_t patch,temp,channels; const char *v=nj_get(config,"vision_config");
            if(u32_at(v,"patch_size",&patch) || u32_at(v,"temporal_patch_size",&temp) ||
               u32_at(v,"in_channels",&channels) || t->ndim!=5 || t->shape[1]!=channels ||
               t->shape[2]!=temp || t->shape[3]!=patch || t->shape[4]!=patch)goto done;
            t->shape[1]=count/t->shape[0];t->ndim=2;
        }
        if(t->ndim>4)goto done;
        t->lo=lo;t->hi=hi;
    }
    if(bad)goto done;
    }
    if(head>1 || nt!=expected+head)goto done;
    if(!vision && !head){const char *tied=nj_get(config,"tie_word_embeddings");if(!tied)tied=nj_get(c,"tie_word_embeddings");if(!tied || strncmp(tied,"true",4))goto done;}
    qsort(ts,(size_t)nt,sizeof(*ts),by_name);
    for(int i=0;i<nt;i++) {
        uint64_t shape[4];for(uint32_t d=0;d<ts[i].ndim;d++)shape[d]=ts[i].shape[ts[i].ndim-1-d];
        if(i && !strcmp(ts[i-1].logical,ts[i].logical))goto done;
        gguf_write_tensor_decl(w,ts[i].logical,ts[i].ndim,shape,ts[i].dtype);
    }
    char buf[65536];
    for(int i=0;i<nt;i++) {
        uint64_t left=ts[i].hi-ts[i].lo;
        FILE *f=src[ts[i].shard];
        if(gguf_write_tensor_begin(w,ts[i].logical) || fseeko(f,(off_t)(data_at[ts[i].shard]+ts[i].lo),SEEK_SET))goto done;
        while(left){size_t n=fread(buf,1,left<sizeof(buf)?(size_t)left:sizeof(buf),f);if(!n || gguf_write_tensor_chunk(w,buf,n))goto done;left-=n;}
        if(gguf_write_tensor_end(w))goto done;
    }
    {int closed=gguf_write_close(w);w=NULL;if(!closed)rc=publish(partial,out,nt);}
done:
    if(rc)fail("conversion failed; expected a supported Qwen3.5 source checkpoint (model.safetensors, or an index and its shards)",snapshot);
    if(w)gguf_write_abort(w);
    for(int s=0;s<MAX_SHARDS;s++){if(src[s])fclose(src[s]);free(shards[s]);}
    for(int i=0;i<nt;i++)free(ts[i].name);
    free(ts);free(header);free(index);free(config);return rc;
}

static int is_false(const char *v){return v && !strncmp(v,"false",5);}
static int convert_tokenizer(const char *input,const char *out) {
    const char *pattern="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    /* The official Qwen3.5 regex, marks inside words: pre-type "qwen35". */
    const char *marks="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    char *j=nj_load(input,NULL),**tokens=NULL,**merges=NULL,*regex=NULL;int32_t *types=NULL;
    int bad=0,rc=1;long long nv=0,nm=0;char partial[4200];gguf_writer *w=NULL;
    if(!j)goto done;
    const char *model=nj_get(j,"model"),*pre=nj_get(j,"pre_tokenizer");
    const char *p1=nj_first(nj_get(pre,"pretokenizers"),&bad),*p2=p1?nj_next(p1,']',&bad):NULL;
    regex=nj_string(nj_get(nj_get(p1,"pattern"),"Regex"));
    if(bad || !p2 || nj_next(p2,']',&bad) || bad || !regex || (strcmp(regex,pattern) && strcmp(regex,marks)) ||
       !nj_is(nj_get(model,"type"),"BPE") || !nj_is(nj_get(nj_get(j,"normalizer"),"type"),"NFC") ||
       !nj_is(nj_get(pre,"type"),"Sequence") || !nj_is(nj_get(p1,"type"),"Split") ||
       !nj_is(nj_get(p1,"behavior"),"Isolated") || !is_false(nj_get(p1,"invert")) ||
       !nj_is(nj_get(p2,"type"),"ByteLevel") || !is_false(nj_get(p2,"add_prefix_space")) || !is_false(nj_get(p2,"use_regex")))goto done;
    const char *vocab=nj_get(model,"vocab"),*list=nj_get(model,"merges"),*added=nj_get(j,"added_tokens");
    for(const char *k=nj_first(vocab,&bad),*v;k && (v=nj_value(k,&bad));k=nj_next(v,'}',&bad)) {
        long long id;if(nj_int(v,&id) || id<0 || id>=GGUF_MAX_STR_ARRAY)goto done;if(id>=nv)nv=id+1;
    }
    for(const char *v=nj_first(added,&bad);v;v=nj_next(v,']',&bad)) {
        long long id;if(nj_int(nj_get(v,"id"),&id) || id<0 || id>=GGUF_MAX_STR_ARRAY)goto done;if(id>=nv)nv=id+1;
    }
    for(const char *v=nj_first(list,&bad);v;v=nj_next(v,']',&bad))nm++;
    if(bad || !nv || !nm || nm>GGUF_MAX_STR_ARRAY)goto done;
    tokens=calloc(nv,sizeof(*tokens));types=calloc(nv,sizeof(*types));merges=calloc(nm,sizeof(*merges));
    if(!tokens || !types || !merges)goto done;
    for(const char *k=nj_first(vocab,&bad),*v;k && (v=nj_value(k,&bad));k=nj_next(v,'}',&bad)) {
        long long id;if(nj_int(v,&id) || tokens[id] || !(tokens[id]=nj_string(k)))goto done;types[id]=1;
    }
    for(const char *v=nj_first(added,&bad);v;v=nj_next(v,']',&bad)) {
        long long id;char *word=nj_string(nj_get(v,"content"));
        if(!word || nj_int(nj_get(v,"id"),&id)){free(word);goto done;}
        if(!is_false(nj_get(v,"lstrip")) || !is_false(nj_get(v,"rstrip")) || !is_false(nj_get(v,"single_word")) ||
           (tokens[id] && strcmp(tokens[id],word))){free(word);goto done;}
        free(tokens[id]);tokens[id]=word;const char *special=nj_get(v,"special");
        if(!special)goto done;
        types[id]=!strncmp(special,"true",4)?3:4;
    }
    for(long long i=0;i<nv;i++)if(!tokens[i])goto done;
    long long i=0;
    for(const char *v=nj_first(list,&bad);v;v=nj_next(v,']',&bad),i++) {
        if(*v=='"') {   /* "a b": the same merge, exactly one space between the halves */
            char *s=nj_string(v),*sp=s?strchr(s,' '):NULL;
            if(!sp || strchr(sp+1,' ')){free(s);goto done;}
            merges[i]=s;continue;
        }
        int b=0;const char *a=nj_first(v,&b),*z=a?nj_next(a,']',&b):NULL;
        char *x=nj_string(a),*y=nj_string(z);
        if(b || !x || !y || nj_next(z,']',&b) || b || strchr(x,' ') || strchr(y,' ') || !(merges[i]=malloc(strlen(x)+strlen(y)+2))){free(x);free(y);goto done;}
        sprintf(merges[i],"%s %s",x,y);free(x);free(y);
    }
    if(bad || !(w=start(out,partial,sizeof(partial))))goto done;
    gguf_write_kv_str(w,"general.architecture","qwen35");gguf_write_kv_str(w,"tokenizer.ggml.model","gpt2");
    gguf_write_kv_str(w,"tokenizer.ggml.pre",strcmp(regex,pattern)?"qwen35":"qwen35-ascii");gguf_write_kv_bool(w,"tokenizer.ggml.add_bos_token",0);
    gguf_write_kv_str_array(w,"tokenizer.ggml.tokens",(const char *const*)tokens,nv);
    gguf_write_kv_str_array(w,"tokenizer.ggml.merges",(const char *const*)merges,nm);
    gguf_write_kv_i32_array(w,"tokenizer.ggml.token_type",types,nv);
    {int closed=gguf_write_close(w);w=NULL;if(!closed)rc=publish(partial,out,0);}
done:
    if(rc)fail("unsupported tokenizer or output failure",input);
    if(w)gguf_write_abort(w);
    for(long long i=0;tokens && i<nv;i++)free(tokens[i]);
    for(long long i=0;merges && i<nm;i++)free(merges[i]);
    free(tokens);free(types);free(merges);free(regex);free(j);return rc;
}
int main(int argc,char **argv) {
    if(argc==4 && !strcmp(argv[1],"--tokenizer"))return convert_tokenizer(argv[2],argv[3]);
    if(argc==4 && !strcmp(argv[1],"--vision"))return convert_model(argv[2],argv[3],1);
    if(argc==3)return convert_model(argv[1],argv[2],0);
    fprintf(stderr,"Usage: %s SNAPSHOT MODEL.gguf | --vision SNAPSHOT VISION.gguf | --tokenizer tokenizer.json TOKENIZER.gguf\n",argv[0]);return 2;
}
