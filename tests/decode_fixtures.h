/* decode_fixtures.h — the small Q8_0 GGUF bodies the decode tests run on.
 *
 * A qwen2 body (attention biases), a qwen3 body (per-head q/k norms) and a gemma3 body
 * (sliding-window layers), 64 wide so every matrix goes through the packed kernels. Shared
 * by test_multi_decode, which checks a multi-sequence step against single decodes, and
 * test_cuda_decode, which checks the CUDA decoder against the CPU one. */
#ifndef NT_TESTS_DECODE_FIXTURES_H
#define NT_TESTS_DECODE_FIXTURES_H

#include "harness/arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { E = 64, V = 40, L = 2, FF = 128, HD = 32, SEQ = 3, STEPS = 6, CAP = 16 };

static float value(long i, int salt) { return (float)(((i * 7 + 3 + salt) % 17) - 8) * 0.03125f; }

static int put_q8(gguf_writer *w, const char *name, int rows, int cols, int salt) {
    float *row = malloc((size_t)cols * sizeof(float));
    uint8_t *packed = malloc((size_t)rows * (cols / 32) * 34);
    if (!row || !packed) { free(row); free(packed); return -1; }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) row[c] = value((long)r * cols + c, salt);
        if (nt_quantize_row(row, packed + (size_t)r * (cols / 32) * 34, cols, GGUF_TYPE_Q8_0)) return -1;
    }
    int rc = gguf_write_tensor(w, name, packed, (uint64_t)rows * (cols / 32) * 34);
    free(row); free(packed); return rc;
}

static int fixture(const char *path, int qwen3) {
    const char *arch = qwen3 ? "qwen3" : "qwen2";
    char key[96], name[96];
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w, "general.architecture", arch);
    #define KEY(k) (snprintf(key, sizeof(key), "%s." k, arch), key)
    gguf_write_kv_u32(w, KEY("block_count"), L);
    gguf_write_kv_u32(w, KEY("embedding_length"), E);
    gguf_write_kv_u32(w, KEY("feed_forward_length"), FF);
    gguf_write_kv_u32(w, KEY("attention.head_count"), 2);
    gguf_write_kv_u32(w, KEY("attention.head_count_kv"), 1);
    gguf_write_kv_u32(w, KEY("context_length"), CAP);
    gguf_write_kv_f32(w, KEY("attention.layer_norm_rms_epsilon"), 1e-5f);
    /* Declarations: name, rows, cols (0 rows = 1-D), packed. */
    struct { const char *field; int rows, cols, packed; } per_layer[] = {
        {"attn_norm.weight", 0, E, 0}, {"attn_q.weight", E, E, 1}, {"attn_k.weight", HD, E, 1},
        {"attn_v.weight", HD, E, 1}, {"attn_output.weight", E, E, 1}, {"ffn_norm.weight", 0, E, 0},
        {"ffn_gate.weight", FF, E, 1}, {"ffn_up.weight", FF, E, 1}, {"ffn_down.weight", E, FF, 1},
        {"attn_q.bias", 0, E, 0}, {"attn_k.bias", 0, HD, 0}, {"attn_v.bias", 0, HD, 0},
        {"attn_q_norm.weight", 0, HD, 0}, {"attn_k_norm.weight", 0, HD, 0},
    };
    int fields = sizeof(per_layer) / sizeof(per_layer[0]);
    #define WANTED(f) (qwen3 ? !strstr(per_layer[f].field, ".bias") : !strstr(per_layer[f].field, "_norm.weight") || !strncmp(per_layer[f].field, "attn_norm", 9) || !strncmp(per_layer[f].field, "ffn_norm", 8))
    gguf_write_tensor_decl(w, "token_embd.weight", 2, (uint64_t[]){E, V}, GGUF_TYPE_Q8_0);
    gguf_write_tensor_decl(w, "output_norm.weight", 1, (uint64_t[]){E}, GGUF_TYPE_F32);
    for (int l = 0; l < L; l++)
        for (int f = 0; f < fields; f++) {
            if (!WANTED(f)) continue;
            snprintf(name, sizeof(name), "blk.%d.%s", l, per_layer[f].field);
            if (per_layer[f].rows)
                gguf_write_tensor_decl(w, name, 2, (uint64_t[]){(uint64_t)per_layer[f].cols,
                                       (uint64_t)per_layer[f].rows}, GGUF_TYPE_Q8_0);
            else
                gguf_write_tensor_decl(w, name, 1, (uint64_t[]){(uint64_t)per_layer[f].cols}, GGUF_TYPE_F32);
        }
    if (put_q8(w, "token_embd.weight", V, E, 1)) return -1;
    float ones[FF];
    for (int i = 0; i < FF; i++) ones[i] = 1.0f;
    gguf_write_tensor_f32(w, "output_norm.weight", ones, E);
    for (int l = 0; l < L; l++)
        for (int f = 0; f < fields; f++) {
            if (!WANTED(f)) continue;
            snprintf(name, sizeof(name), "blk.%d.%s", l, per_layer[f].field);
            if (per_layer[f].rows) {
                if (put_q8(w, name, per_layer[f].rows, per_layer[f].cols, l * 31 + f * 5)) return -1;
            } else if (strstr(per_layer[f].field, ".bias")) {
                float b[E];
                for (int i = 0; i < per_layer[f].cols; i++) b[i] = value(i, f) * 0.5f;
                gguf_write_tensor_f32(w, name, b, (uint64_t)per_layer[f].cols);
            } else {
                gguf_write_tensor_f32(w, name, ones, (uint64_t)per_layer[f].cols);
            }
        }
    return gguf_write_close(w);
}

/* Gemma 3: six layers, five of them sliding with a window of 4, so the positions this test
 * reaches (up to 10) cut the window for most rows. Norms near one, Q8_0 matrices. */
enum { GL = 6 };
static int fixture_gemma3(const char *path) {
    const char *fields[] = {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm",
        "attn_k_norm", "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm"};
    int rows[] = {0, E, HD, HD, E, 0, 0, 0, 0, FF, FF, E, 0};
    int cols[] = {E, E, E, E, E, HD, HD, E, E, E, E, FF, E};
    char name[96];
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w, "general.architecture", "gemma3");
    gguf_write_kv_u32(w, "gemma3.block_count", GL);
    gguf_write_kv_u32(w, "gemma3.embedding_length", E);
    gguf_write_kv_u32(w, "gemma3.feed_forward_length", FF);
    gguf_write_kv_u32(w, "gemma3.attention.head_count", 2);
    gguf_write_kv_u32(w, "gemma3.attention.head_count_kv", 1);
    gguf_write_kv_u32(w, "gemma3.attention.key_length", HD);
    gguf_write_kv_u32(w, "gemma3.attention.value_length", HD);
    gguf_write_kv_u32(w, "gemma3.attention.sliding_window", 4);
    gguf_write_kv_u32(w, "gemma3.context_length", CAP);
    gguf_write_kv_f32(w, "gemma3.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_write_kv_f32(w, "gemma3.rope.freq_base", 10000);
    gguf_write_kv_f32(w, "gemma3.rope.freq_base_swa", 100);
    gguf_write_tensor_decl(w, "token_embd.weight", 2, (uint64_t[]){E, V}, GGUF_TYPE_Q8_0);
    gguf_write_tensor_decl(w, "output_norm.weight", 1, (uint64_t[]){E}, GGUF_TYPE_F32);
    for (int l = 0; l < GL; l++)
        for (int f = 0; f < 13; f++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, fields[f]);
            if (rows[f]) gguf_write_tensor_decl(w, name, 2, (uint64_t[]){(uint64_t)cols[f], (uint64_t)rows[f]}, GGUF_TYPE_Q8_0);
            else gguf_write_tensor_decl(w, name, 1, (uint64_t[]){(uint64_t)cols[f]}, GGUF_TYPE_F32);
        }
    if (put_q8(w, "token_embd.weight", V, E, 2)) return -1;
    float norm[FF];
    for (int i = 0; i < FF; i++) norm[i] = 0.9f + 0.02f * (float)(i % 7);
    gguf_write_tensor_f32(w, "output_norm.weight", norm, E);
    for (int l = 0; l < GL; l++)
        for (int f = 0; f < 13; f++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, fields[f]);
            if (rows[f]) { if (put_q8(w, name, rows[f], cols[f], l * 13 + f)) return -1; }
            else gguf_write_tensor_f32(w, name, norm, (uint64_t)cols[f]);
        }
    return gguf_write_close(w);
}

#endif
