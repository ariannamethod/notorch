/* Gemma 3 text decoder. Arithmetic follows llama.cpp src/models/gemma3.cpp:
 * scaled embeddings; per-head QK RMSNorm; NEOX local/global RoPE; GELU;
 * post-attention and post-FFN RMSNorm before each residual addition.
 * GGUF conversion has already folded +1 into every norm weight.
 * Unlike a llama block, ffn_down.bias is NOT an additive residual direction. */
#include "harness/arch.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm;
    wt q, k, v, o, gate, up, down;
} gemma3_layer;
typedef struct {
    int E, H, KV, HD, QD, KD, FF, V, L, window, pattern, emb_ti;
    float eps, base, local_base, rope_scale, query_scale, softcap;
    gguf_file *gf;
    wt emb, output;
    float *norm;
    gemma3_layer layers[];
} gemma3_model;

static int meta_u(const gguf_file *gf, const char *name, int fallback) {
    const gguf_kv *v = gguf_get_kv(gf, name);
    if (!v) return fallback;
    if (v->type != 4 || v->val.u32 > INT_MAX) return -1;
    return (int)v->val.u32;
}
static float meta_f(const gguf_file *gf, const char *name, float fallback) {
    const gguf_kv *v = gguf_get_kv(gf, name);
    return v ? (v->type == 6 ? v->val.f32 : NAN) : fallback;
}
static int shape(gguf_file *gf, const char *name, int rows, int cols) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return -1;
    const gguf_tensor_info *t = &gf->tensors[ti];
    if (t->ndim != (rows ? 2u : 1u) || t->shape[0] != (uint64_t)cols ||
        (rows && t->shape[1] != (uint64_t)rows)) return -1;
    return ti;
}
static float *norm_load(gguf_file *gf, const char *name, int width) {
    int ti = shape(gf, name, 0, width);
    return ti < 0 ? NULL : gguf_dequant(gf, ti);
}
static int matrix_load(wt *w, gguf_file *gf, const char *name, int rows, int cols) {
    return shape(gf, name, rows, cols) >= 0 && wt_load(w, gf, name);
}
static void gemma3_free(void *ptr) {
    gemma3_model *m = ptr;
    if (!m) return;
    free(m->emb.f32); free(m->output.f32); free(m->norm);
    for (int i = 0; i < m->L; i++) {
        gemma3_layer *l = &m->layers[i];
        free(l->attn_norm); free(l->q_norm); free(l->k_norm);
        free(l->post_attn_norm); free(l->ffn_norm); free(l->post_ffw_norm);
        free(l->q.f32); free(l->k.f32); free(l->v.f32); free(l->o.f32);
        free(l->gate.f32); free(l->up.f32); free(l->down.f32);
    }
    free(m);
}
static void *gemma3_load(gguf_file *gf, nt_dims *dims) {
    if (!gf || !dims || strcmp(gf->arch, "gemma3")) return NULL;
    int L = meta_u(gf, "gemma3.block_count", 0);
    if (L <= 0 || L > GGUF_MAX_TENSORS / 13) return NULL;
    gemma3_model *m = calloc(1, sizeof(*m) + (size_t)L * sizeof(m->layers[0]));
    if (!m) return NULL;
    m->L = L; m->gf = gf;
    m->E = meta_u(gf, "gemma3.embedding_length", 0);
    m->H = meta_u(gf, "gemma3.attention.head_count", 0);
    m->KV = meta_u(gf, "gemma3.attention.head_count_kv", m->H);
    m->HD = meta_u(gf, "gemma3.attention.key_length", m->H > 0 ? m->E / m->H : 0);
    int rot = meta_u(gf, "gemma3.rope.dimension_count", m->HD);
    int vd = meta_u(gf, "gemma3.attention.value_length", m->HD);
    m->FF = meta_u(gf, "gemma3.feed_forward_length", 0);
    m->window = meta_u(gf, "gemma3.attention.sliding_window", 0);
    m->pattern = meta_u(gf, "gemma3.attention.sliding_window_pattern", 6);
    m->eps = meta_f(gf, "gemma3.attention.layer_norm_rms_epsilon", NAN);
    m->base = meta_f(gf, "gemma3.rope.freq_base", 10000.0f);
    m->local_base = meta_f(gf, "gemma3.rope.freq_base_swa", 10000.0f);
    m->softcap = meta_f(gf, "gemma3.final_logit_softcapping", 0.0f);
    float factor = meta_f(gf, "gemma3.rope.scaling.factor", 1.0f);
    const gguf_kv *scaling = gguf_get_kv(gf, "gemma3.rope.scaling.type");
    if (scaling && (scaling->type != 8 ||
        (strcmp(scaling->val.str, "none") && strcmp(scaling->val.str, "linear")))) goto bad;
    if (m->E <= 0 || m->H <= 0 || m->KV <= 0 || m->H % m->KV ||
        m->HD <= 0 || (m->HD & 1) || vd != m->HD || rot != m->HD || m->FF <= 0 ||
        m->H > INT_MAX / m->HD || m->window < 0 || m->pattern <= 0 ||
        !isfinite(m->eps) || m->eps <= 0 || !isfinite(m->base) || m->base <= 0 ||
        !isfinite(m->local_base) || m->local_base <= 0 || !isfinite(factor) || factor <= 0 ||
        !isfinite(m->softcap) || m->softcap < 0) goto bad;
    m->QD = m->H * m->HD; m->KD = m->KV * m->HD;
    m->rope_scale = scaling && !strcmp(scaling->val.str, "none") ? 1.0f : 1.0f / factor;
    /* Gemma 3 27B uses embed / heads; 270M, 1B, 4B, 12B use head_dim. */
    m->query_scale = 1.0f / sqrtf((float)(L == 62 ? m->E / m->H : m->HD));
    if (!isfinite(m->query_scale)) goto bad;
    m->emb_ti = gguf_find_tensor(gf, "token_embd.weight");
    if (m->emb_ti < 0 || gf->tensors[m->emb_ti].ndim != 2 ||
        gf->tensors[m->emb_ti].shape[1] > INT_MAX) goto bad;
    m->V = (int)gf->tensors[m->emb_ti].shape[1];
    if (m->V <= 0 || !matrix_load(&m->emb, gf, "token_embd.weight", m->V, m->E) ||
        !(m->norm = norm_load(gf, "output_norm.weight", m->E))) goto bad;
    if (gguf_find_tensor(gf, "output.weight") >= 0 &&
        !matrix_load(&m->output, gf, "output.weight", m->V, m->E)) goto bad;
    for (int i = 0; i < L; i++) {
        gemma3_layer *l = &m->layers[i]; char name[128];
#define N(field, suffix, width) do { \
    snprintf(name, sizeof(name), "blk.%d." suffix ".weight", i); \
    if (!(l->field = norm_load(gf, name, width))) goto bad; \
} while (0)
#define W(field, suffix, rows, cols) do { \
    snprintf(name, sizeof(name), "blk.%d." suffix ".weight", i); \
    if (!matrix_load(&l->field, gf, name, rows, cols)) goto bad; \
} while (0)
        N(attn_norm, "attn_norm", m->E); N(q_norm, "attn_q_norm", m->HD);
        N(k_norm, "attn_k_norm", m->HD); N(post_attn_norm, "post_attention_norm", m->E);
        N(ffn_norm, "ffn_norm", m->E); N(post_ffw_norm, "post_ffw_norm", m->E);
        W(q, "attn_q", m->QD, m->E); W(k, "attn_k", m->KD, m->E);
        W(v, "attn_v", m->KD, m->E); W(o, "attn_output", m->E, m->QD);
        W(gate, "ffn_gate", m->FF, m->E); W(up, "ffn_up", m->FF, m->E);
        W(down, "ffn_down", m->E, m->FF);
#undef N
#undef W
        /* A llama-style saved residual shift must never load and disappear silently. */
        snprintf(name, sizeof(name), "blk.%d.ffn_down.bias", i);
        if (gguf_find_tensor(gf, name) >= 0) goto bad;
    }
    dims->n_layers = L; dims->kv_dim = m->KD; dims->vocab = m->V;
    fprintf(stderr, "gemma3: E=%d H=%d KV=%d HD=%d FF=%d V=%d L=%d window=%d pattern=%d\n",
            m->E, m->H, m->KV, m->HD, m->FF, m->V, L, m->window, m->pattern);
    return m;
bad:
    fprintf(stderr, "gemma3: invalid tensor shape or unsupported model metadata\n");
    gemma3_free(m); return NULL;
}

/* Linear scaling affects global layers only. Gemma3 GGUF Q/K are not permuted:
 * the rotation pairs opposite halves, not adjacent lanes. */
static void gemma3_rope(float *x, int pos, int hd, float base, float scale) {
    for (int i = 0; i < hd / 2; i++) {
        float angle = (float)pos * scale / powf(base, 2.0f * (float)i / (float)hd);
        float c = cosf(angle), s = sinf(angle), a = x[i], b = x[i + hd/2];
        x[i] = a*c - b*s; x[i + hd/2] = a*s + b*c;
    }
}
/* Row j is ids[j] at position pos[j] of the cache kvs[j]: one cache and consecutive
 * positions for a prefill or a single decode, one row per sequence for a multi-sequence
 * decode. `head_rows` trailing rows get logits, [head_rows, vocab]. With `users` NULL the
 * hook sees the whole block once per layer; with `users` it runs per row, n = 1, with that
 * row's position and user. Callers have validated every row. */
static int gemma3_rows(gemma3_model *m, int n, const int *ids, kv_cache *const *kvs,
                       const int *pos, float *logits, int head_rows,
                       nt_residual_fn hook, void *user, void *const *users) {
    int rc=NT_OK;
    int E=m->E, QD=m->QD, KD=m->KD, HD=m->HD, FF=m->FF;
    int stride=0;
    for (int j=0;j<n;j++) if (kvs[j]->max_seq>stride) stride=kvs[j]->max_seq;
    float *x=calloc((size_t)n*E,sizeof(float)), *norm=calloc((size_t)n*E,sizeof(float));
    float *q=calloc((size_t)n*QD,sizeof(float)), *k=calloc((size_t)n*KD,sizeof(float));
    float *v=calloc((size_t)n*KD,sizeof(float)), *att=calloc((size_t)n*QD,sizeof(float));
    float *out=calloc((size_t)n*E,sizeof(float)), *gate=calloc((size_t)n*FF,sizeof(float));
    float *up=calloc((size_t)n*FF,sizeof(float)), *scores=calloc((size_t)stride,sizeof(float));
    if (!x||!norm||!q||!k||!v||!att||!out||!gate||!up||!scores) {rc=NT_E_MEMORY;goto done;}
    for (int j=0;j<n;j++) {
        float *row=x+(size_t)j*E;
        if (m->emb.f32) memcpy(row,m->emb.f32+(size_t)ids[j]*E,(size_t)E*sizeof(float));
        else if (gguf_dequant_row(m->gf,m->emb_ti,(uint64_t)ids[j],row)) {rc=NT_E_STATE;goto done;}
        for (int i=0;i<E;i++) row[i]*=sqrtf((float)E);
    }
    for (int b=0;b<m->L;b++) {
        gemma3_layer *l=&m->layers[b];
        int local=m->window>0 && (b+1)%m->pattern!=0;
        float base=local?m->local_base:m->base, scale=local?1.0f:m->rope_scale;
        for (int j=0;j<n;j++) rmsnorm(norm+(size_t)j*E,x+(size_t)j*E,l->attn_norm,E,m->eps);
        qmm(q,&l->q,norm,n); qmm(k,&l->k,norm,n); qmm(v,&l->v,norm,n);
        for (int j=0;j<n;j++) {
            size_t cache_base=(size_t)b*kvs[j]->max_seq*KD;
            for (int h=0;h<m->H;h++) {
                float *row=q+(size_t)j*QD+h*HD;
                rmsnorm(row,row,l->q_norm,HD,m->eps);
                gemma3_rope(row,pos[j],HD,base,scale);
                for (int i=0;i<HD;i++) row[i]*=m->query_scale;
            }
            for (int h=0;h<m->KV;h++) {
                float *row=k+(size_t)j*KD+h*HD;
                rmsnorm(row,row,l->k_norm,HD,m->eps);
                gemma3_rope(row,pos[j],HD,base,scale);
            }
            memcpy(kvs[j]->k+cache_base+(size_t)pos[j]*KD,k+(size_t)j*KD,(size_t)KD*sizeof(float));
            memcpy(kvs[j]->v+cache_base+(size_t)pos[j]*KD,v+(size_t)j*KD,(size_t)KD*sizeof(float));
        }
        memset(att,0,(size_t)n*QD*sizeof(float));
        for (int j=0;j<n;j++) {
            const kv_cache *kv=kvs[j];
            size_t cache_base=(size_t)b*kv->max_seq*KD;
            int p=pos[j], first=local && p>=m->window ? p-m->window+1 : 0;
            int span=p-first+1;
            for (int h=0;h<m->H;h++) {
                int kh=h/(m->H/m->KV);
                for (int t=0;t<span;t++) scores[t]=dot_f32(q+(size_t)j*QD+h*HD,
                    kv->k+cache_base+(size_t)(first+t)*KD+kh*HD,HD);
                softmax(scores,span);
                for (int t=0;t<span;t++) axpy_f32(att+(size_t)j*QD+h*HD,scores[t],
                    kv->v+cache_base+(size_t)(first+t)*KD+kh*HD,HD);
            }
        }
        qmm(out,&l->o,att,n);
        for (int j=0;j<n;j++) rmsnorm(out+(size_t)j*E,out+(size_t)j*E,l->post_attn_norm,E,m->eps);
        for (size_t i=0;i<(size_t)n*E;i++) x[i]+=out[i];
        for (int j=0;j<n;j++) rmsnorm(norm+(size_t)j*E,x+(size_t)j*E,l->ffn_norm,E,m->eps);
        qmm(gate,&l->gate,norm,n); qmm(up,&l->up,norm,n);
        for (size_t i=0;i<(size_t)n*FF;i++) {
            float g=gate[i];
            gate[i]=0.5f*g*(1.0f+tanhf(0.7978845608028654f*g*(1.0f+0.044715f*g*g)))*up[i];
        }
        qmm(out,&l->down,gate,n);
        for (int j=0;j<n;j++) rmsnorm(out+(size_t)j*E,out+(size_t)j*E,l->post_ffw_norm,E,m->eps);
        for (size_t i=0;i<(size_t)n*E;i++) x[i]+=out[i];
        if (hook && !users && (rc=hook(user,b,pos[0],n,E,x))!=NT_OK) goto done;
        if (hook && users)
            for (int j=0;j<n;j++)
                if ((rc=hook(users[j],b,pos[j],1,E,x+(size_t)j*E))!=NT_OK) goto done;
    }
    if (logits) {
        const wt *head=m->output.q||m->output.f32?&m->output:&m->emb;
        int first=n-head_rows;
        for (int j=0;j<head_rows;j++) rmsnorm(norm+(size_t)j*E,x+(size_t)(first+j)*E,m->norm,E,m->eps);
        if (head_rows==1) qmv(logits,head,norm);
        else qmm(logits,head,norm,head_rows);
        if (m->softcap>0) for(size_t i=0;i<(size_t)head_rows*m->V;i++) logits[i]=m->softcap*tanhf(logits[i]/m->softcap);
    }
done:
    free(x);free(norm);free(q);free(k);free(v);free(att);free(out);free(gate);free(up);free(scores);
    return rc;
}
static int gemma3_forward_residual(void *ptr, kv_cache *kv, const int *ids, int n,
                                   int pos0, float *logits, nt_residual_fn hook, void *user) {
    gemma3_model *m = ptr;
    if (!m) return NT_E_ARG;
    int rc = nt_check_call(kv, ids, n, pos0, m->V, m->L, m->KD);
    if (rc) return rc;
    int *pos=malloc((size_t)n*sizeof(int)); kv_cache **kvs=malloc((size_t)n*sizeof(kv_cache*));
    if (!pos||!kvs) { free(pos); free(kvs); return NT_E_MEMORY; }
    for (int j=0;j<n;j++) { pos[j]=pos0+j; kvs[j]=kv; }
    rc=gemma3_rows(m,n,ids,kvs,pos,logits,1,hook,user,NULL);
    free(pos); free(kvs);
    return rc;
}
static int gemma3_forward(void *m,kv_cache *kv,const int *ids,int n,int pos,float *logits) {
    return gemma3_forward_residual(m,kv,ids,n,pos,logits,NULL,NULL);
}
/* One decode step for n independent sequences; see forward_multi in arch.h. */
static int gemma3_forward_multi(void *ptr, kv_cache *const *kvs, const int *ids, const int *pos,
                                int n, float *logits, nt_residual_fn hook, void *const *users) {
    gemma3_model *m = ptr;
    if (!m || !kvs || !ids || !pos || n < 1 || (hook && !users)) return NT_E_ARG;
    for (int j=0;j<n;j++) {
        if (!kvs[j]) return NT_E_ARG;
        for (int i=0;i<j;i++) if (kvs[i]==kvs[j]) return NT_E_ARG;
        int rc=nt_check_call(kvs[j],&ids[j],1,pos[j],m->V,m->L,m->KD);
        if (rc) return rc;
    }
    return gemma3_rows(m,n,ids,kvs,pos,logits,n,hook,NULL,users);
}
static const char *const names[]={"gemma3",NULL};
const nt_arch nt_arch_gemma3={.names=names,.load=gemma3_load,.free=gemma3_free,
    .forward=gemma3_forward,.forward_residual=gemma3_forward_residual,
    .forward_multi=gemma3_forward_multi};
