/* test_gguf_keys.c — every metadata value type the GGUF format defines, read in place.
 *
 * gguf_open skipped only the types it happened to meet first. For uint8, int8, uint16,
 * int16 and int64 it neither read nor skipped: the position stayed put, and every key,
 * tensor and offset after it came from the wrong bytes. llama-gguf-split writes
 * split.no and split.count as uint16, so each merged model failed to load at its last key
 * while llama.cpp read it. This file is written by hand, byte by byte, because the writer
 * in gguf.c has no small-integer keys — a fixture built with it could not contain the case.
 *
 * The file carries one key of each missing type, an array of uint16, a float64, and after
 * all of them an ordinary uint32 and a tensor: if any value is skipped by the wrong width,
 * the block count, the tensor's name or its data come back wrong. A value type the format
 * does not define must make gguf_open refuse the file. */
#include "gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failed, checks;
#define CHECK(c, s) do { checks++; if (!(c)) { fprintf(stderr, "FAIL: %s\n", s); failed++; } } while (0)

static void u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void str(FILE *f, const char *s) { u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }
static void key(FILE *f, const char *k, uint32_t type) { str(f, k); u32(f, type); }

static int write_fixture(const char *path, int bad_type) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    u32(f, 0x46554747u); u32(f, 3); u64(f, 1); u64(f, bad_type ? 3 : 10);
    key(f, "general.architecture", 8); str(f, "qwen2");
    if (bad_type) {
        key(f, "odd.key", 13); u32(f, 1);
        key(f, "qwen2.block_count", 4); u32(f, 7);
    } else {
        uint8_t a = 200; int8_t b = -5; uint16_t c = 60000; int16_t d = -30000;
        int64_t e = -123456789012LL; double g = 2.5;
        key(f, "t.u8", 0); fwrite(&a, 1, 1, f);
        key(f, "t.i8", 1); fwrite(&b, 1, 1, f);
        key(f, "split.no", 2); fwrite(&c, 2, 1, f);
        key(f, "t.i16", 3); fwrite(&d, 2, 1, f);
        key(f, "t.i64", 11); fwrite(&e, 8, 1, f);
        key(f, "t.f64", 12); fwrite(&g, 8, 1, f);
        key(f, "t.u16s", 9); u32(f, 2); u64(f, 3);
        uint16_t arr[3] = {1, 2, 3}; fwrite(arr, 2, 3, f);
        key(f, "split.count", 2); fwrite(&c, 2, 1, f);
        key(f, "qwen2.block_count", 4); u32(f, 7);
    }
    /* One F32 tensor [4, 3], data after 32-byte alignment. */
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 4); u64(f, 3); u32(f, 0); u64(f, 0);
    long here = ftell(f);
    for (long pad = (32 - here % 32) % 32; pad > 0; pad--) fputc(0, f);
    for (int i = 0; i < 12; i++) { float v = 0.25f * (float)i - 1.0f; fwrite(&v, 4, 1, f); }
    return fclose(f);
}

int main(void) {
    char path[] = "/tmp/nt_gguf_keys_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    close(fd);
    CHECK(write_fixture(path, 0) == 0, "write fixture");
    gguf_file *gf = gguf_open(path);
    CHECK(gf != NULL, "a file with every defined value type opens");
    if (gf) {
        CHECK(!strcmp(gf->arch, "qwen2"), "architecture read");
        CHECK(gf->n_layers == 7, "a uint32 after all the small types reads its own bytes");
        const gguf_kv *kv;
        CHECK((kv = gguf_get_kv(gf, "t.u8")) && kv->type == 0 && kv->val.u32 == 200, "uint8");
        CHECK((kv = gguf_get_kv(gf, "t.i8")) && kv->type == 1 && kv->val.i32 == -5, "int8");
        CHECK((kv = gguf_get_kv(gf, "split.no")) && kv->type == 2 && kv->val.u32 == 60000, "uint16");
        CHECK((kv = gguf_get_kv(gf, "t.i16")) && kv->type == 3 && kv->val.i32 == -30000, "int16");
        CHECK((kv = gguf_get_kv(gf, "t.i64")) && kv->type == 11 &&
              (int64_t)kv->val.u64 == -123456789012LL, "int64");
        double g = 0; if ((kv = gguf_get_kv(gf, "t.f64"))) memcpy(&g, &kv->val.u64, 8);
        CHECK(kv && kv->type == 12 && g == 2.5, "float64");
        CHECK((kv = gguf_get_kv(gf, "t.u16s")) && kv->type == 9, "array of uint16 kept as an array key");
        CHECK((kv = gguf_get_kv(gf, "split.count")) && kv->val.u32 == 60000, "uint16 after an array");
        int ti = gguf_find_tensor(gf, "token_embd.weight");
        CHECK(ti >= 0, "tensor found after the metadata");
        float *data = ti >= 0 ? gguf_dequant(gf, ti) : NULL;
        int same = data != NULL;
        for (int i = 0; same && i < 12; i++) same = data[i] == 0.25f * (float)i - 1.0f;
        CHECK(same, "tensor data read at its own offset");
        free(data);
        gguf_close(gf);
    }
    CHECK(write_fixture(path, 1) == 0, "write fixture with an undefined type");
    gf = gguf_open(path);
    CHECK(gf == NULL, "an undefined value type is refused, not skipped by guess");
    if (gf) gguf_close(gf);
    unlink(path);
    if (failed) { fprintf(stderr, "GGUF_KEYS FAILED (%d of %d)\n", failed, checks); return 1; }
    printf("GGUF_KEYS_OK (%d checks)\n", checks);
    return 0;
}
