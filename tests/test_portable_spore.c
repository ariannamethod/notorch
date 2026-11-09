/* A decoder with zero linear maps exposes its embedding rows at the DOE hook.
 * Give each calibration literal a known signed axis and permute token IDs on
 * the second body. The expected cross-width A/B and vote are then analytic. */
#include "gguf.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 32, V = 34, R = 4, E = 6 };
static const char *const words[V] = {
    " river"," glass"," seven"," green"," bread"," music"," north"," sleep",
    " iron"," cloud"," circle"," paper"," sweet"," walk"," winter"," clock",
    " horse"," salt"," red"," light"," open"," hand"," stone"," leaf",
    " silver"," quiet"," ocean"," round"," fire"," bird"," small"," wood",
    " exposure"," probe"
};

static void body(const char *path, const char *arch, int width, int shift, int flat, int norm_salt) {
    int depth = width == 64 ? 2 : 4;
    const char *vocab[V];
    int32_t types[V];
    for (int i = 0; i < V; i++) { vocab[(i+shift)%V] = words[i]; types[i] = 4; }
    gguf_writer *w = gguf_write_open(path);
    assert(w);
    gguf_write_kv_str(w, "general.architecture", arch);
    gguf_write_kv_str(w, "general.name", path);
    const char *keys[] = {"embedding_length","block_count","feed_forward_length",
                          "attention.head_count","attention.head_count_kv","context_length"};
    int values[] = {width,depth,width,1,1,64};
    char key[80];
    for (size_t i = 0; i < sizeof(keys)/sizeof(*keys); i++) {
        snprintf(key, sizeof(key), "%s.%s", arch, keys[i]);
        gguf_write_kv_u32(w, key, values[i]);
    }
    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", arch);
    gguf_write_kv_f32(w, key, 1e-5f);
    gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", vocab, V);
    gguf_write_kv_i32_array(w, "tokenizer.ggml.token_type", types, V);
    float *matrix = calloc((size_t)width*width, sizeof(float));
    float *embedding = calloc((size_t)width*V, sizeof(float));
    float *norm = malloc((size_t)width*sizeof(float));
    assert(matrix && embedding && norm);
    for (int i = 0; i < width; i++) norm[i] = 1+(i+norm_salt)*0.001f;
    for (int i = 0; i < V; i++) {
        int axis = flat ? 0 : (i+shift)%width;
        embedding[(size_t)((i+shift)%V)*width+axis] = shift && i%2 ? -1 : 1;
    }
    const char *names[] = {"attn_norm","attn_q","attn_k","attn_v","attn_output",
                           "ffn_norm","ffn_gate","ffn_up","ffn_down",
                           "attn_q_norm","attn_k_norm","post_attention_norm","post_ffw_norm"};
    int fields = !strcmp(arch, "gemma3") ? 13 : 9;
    uint64_t output_width = norm_salt == -1 ? width-1 : width;
    uint64_t shape[] = {(unsigned)width, (unsigned)width};
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0) {
            gguf_write_tensor_decl(w, "token_embd.weight", 2, (uint64_t[]){(unsigned)width,V}, GGUF_TYPE_F32);
            gguf_write_tensor_decl(w, "output_norm.weight", 1, &output_width, GGUF_TYPE_F32);
        } else {
            gguf_write_tensor_f32(w, "token_embd.weight", embedding, (uint64_t)width*V);
            gguf_write_tensor_f32(w, "output_norm.weight", norm, output_width);
        }
        for (int l = 0; l < depth; l++) for (int k = 0; k < fields; k++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, names[k]);
            int vector = strstr(names[k], "norm") != NULL;
            if (pass == 0) gguf_write_tensor_decl(w, name, vector ? 1 : 2, shape, GGUF_TYPE_F32);
            else gguf_write_tensor_f32(w, name, vector ? norm : matrix,
                                       vector ? (unsigned)width : (uint64_t)width*width);
        }
    }
    assert(!gguf_write_close(w));
    free(norm); free(matrix); free(embedding);
}

static void check(const char *src, const char *fresh, const char *dst) {
    gguf_file *s = gguf_open(src), *f = gguf_open(fresh), *d = gguf_open(dst);
    assert(s && f && d);
    assert(!strcmp(gguf_get_kv(f, "doe.host_sha256")->val.str,
                   gguf_get_kv(d, "doe.host_sha256")->val.str));
    assert(strcmp(gguf_get_kv(s, "doe.host_sha256")->val.str,
                  gguf_get_kv(d, "doe.host_sha256")->val.str));
    for (int i = 0; i < 10; i++) {
        const char *name = s->tensors[i].name;
        float *sv = gguf_dequant(s, gguf_find_tensor(s, name));
        float *fv = gguf_dequant(f, gguf_find_tensor(f, name));
        float *dv = gguf_dequant(d, gguf_find_tensor(d, name));
        assert(sv && fv && dv);
        if (i < 3) {
            int ranks = i == 2 ? 1 : R;
            for (int e = 0; e < E; e++) for (int r = 0; r < ranks; r++) {
                for (int x = 0; x < 96; x++) {
                    int j = x-7;
                    int di = e*96*ranks + (i == 0 ? x*ranks+r : r*96+x);
                    float expected = fv[di];
                    if (j >= 0 && j < N) {
                        int si = e*64*ranks + (i == 0 ? j*ranks+r : r*64+j);
                        expected = sv[si] * (j%2 ? -1 : 1);
                    }
                    assert(fabsf(dv[di]-expected) < 1e-7f);
                }
            }
        } else assert(!memcmp(sv, dv, s->tensors[i].n_elements*sizeof(float)));
        free(sv); free(fv); free(dv);
    }
    const char *keys[] = {"doe.rng", "doe.turns", "doe.elections", "doe.consensus"};
    for (int i = 0; i < 4; i++) {
        const gguf_kv *a = gguf_get_kv(s, keys[i]), *b = gguf_get_kv(d, keys[i]);
        assert(a && b && !memcmp(&a->val, &b->val, a->type == 10 ? 8 : 4));
    }
    gguf_close(d); gguf_close(f); gguf_close(s);
}

static void set_first(const char *path, const char *tensor) {
    gguf_file *g = gguf_open(path);
    assert(g);
    int ti = gguf_find_tensor(g, tensor);
    assert(ti >= 0);
    long pos = (long)(g->data_offset + g->tensors[ti].offset);
    gguf_close(g);
    FILE *file = fopen(path, "r+b");
    float two = 2;
    assert(file && !fseek(file, pos, SEEK_SET) && fwrite(&two, sizeof(two), 1, file) == 1);
    assert(!fclose(file));
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "fixtures")) {
        body("source", "llama", 64, 0, 0, 0);
        body("aligned", "llama", 64, 0, 0, 0);
        body("rotated", "llama", 64, 7, 0, 0);
        body("norm-change", "llama", 64, 0, 0, 1);
        body("wider", "llama", 96, 7, 0, 0);
        body("flat", "llama", 64, 0, 1, 0);
        body("short-norm", "llama", 64, 0, 0, -1);
        body("gemma", "gemma3", 64, 0, 0, 0);
        body("gemma-aligned", "gemma3", 64, 0, 0, 0);
    } else if (argc == 5 && !strcmp(argv[1], "check")) check(argv[2], argv[3], argv[4]);
    else if (argc == 3 && !strcmp(argv[1], "break-frame")) set_first(argv[2], "doe.anchor_frame");
    else if (argc == 4 && !strcmp(argv[1], "set-first")) set_first(argv[2], argv[3]);
    else return 1;
    return 0;
}
