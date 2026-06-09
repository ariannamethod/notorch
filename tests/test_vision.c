/*
 * test_vision.c — tests for notorch_vision.h
 *
 * Build: make test_vision
 * Run:   ./test_vision
 */

#include "notorch_vision.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabsf((a) - (b)) < (tol), msg)

/* Write a minimal 4x4 BMP for testing */
static void write_test_bmp(const char* path) {
    FILE* f = fopen(path, "wb");
    /* BMP header: 54 bytes + 4x4x3=48 bytes + padding */
    unsigned char header[54] = {
        0x42,0x4D,        /* BM */
        102,0,0,0,        /* file size: 54 + 48 = 102 */
        0,0,0,0,          /* reserved */
        54,0,0,0,         /* data offset */
        40,0,0,0,         /* header size */
        4,0,0,0,          /* width=4 */
        4,0,0,0,          /* height=4 */
        1,0,              /* planes */
        24,0,             /* bits per pixel */
        0,0,0,0,          /* compression */
        48,0,0,0,         /* image size */
        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
    };
    fwrite(header, 1, 54, f);
    /* BMP rows are bottom-up, BGR, no padding needed for width=4 (4*3=12, divisible by 4) */
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            /* Checkerboard: red (255,0,0) and blue (0,0,255) */
            if ((x + y) % 2 == 0) {
                unsigned char bgr[] = {0, 0, 255}; /* red in BGR */
                fwrite(bgr, 1, 3, f);
            } else {
                unsigned char bgr[] = {255, 0, 0}; /* blue in BGR */
                fwrite(bgr, 1, 3, f);
            }
        }
    }
    fclose(f);
}

/* ── Test: image loading ── */
static void test_load(void) {
    printf("\n── image loading ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");

    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);
    ASSERT(img != NULL, "load BMP");
    ASSERT(img->width == 4, "width == 4");
    ASSERT(img->height == 4, "height == 4");
    ASSERT(img->channels == 3, "channels == 3");

    /* Check pixel values are in [0, 1] */
    int all_valid = 1;
    for (int i = 0; i < 3 * 4 * 4; i++)
        if (img->data[i] < 0 || img->data[i] > 1.0f) all_valid = 0;
    ASSERT(all_valid, "all pixels in [0, 1]");

    /* Grayscale load */
    nt_image* gray = nt_image_load("/tmp/notorch_test.bmp", 1);
    ASSERT(gray != NULL, "load grayscale");
    ASSERT(gray->channels == 1, "gray channels == 1");
    ASSERT(gray->width == 4 && gray->height == 4, "gray size 4x4");

    /* Load nonexistent file */
    nt_image* bad = nt_image_load("/tmp/this_does_not_exist.xyz", 3);
    ASSERT(bad == NULL, "nonexistent file returns NULL");

    nt_image_free(img);
    nt_image_free(gray);
}

/* ── Test: resize ── */
static void test_resize(void) {
    printf("\n── resize ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);

    nt_image* big = nt_image_resize(img, 16, 16);
    ASSERT(big->width == 16, "resize width == 16");
    ASSERT(big->height == 16, "resize height == 16");
    ASSERT(big->channels == 3, "resize keeps 3 channels");

    /* All pixels should still be in [0, 1] after bilinear */
    int valid = 1;
    for (int i = 0; i < 3 * 16 * 16; i++)
        if (big->data[i] < -0.01f || big->data[i] > 1.01f) valid = 0;
    ASSERT(valid, "resized pixels in [0, 1]");

    nt_image* small = nt_image_resize(img, 2, 2);
    ASSERT(small->width == 2, "downscale width == 2");
    ASSERT(small->height == 2, "downscale height == 2");

    /* Resize to same size should be ~identical */
    nt_image* same = nt_image_resize(img, 4, 4);
    float max_diff = 0;
    for (int i = 0; i < 3 * 4 * 4; i++) {
        float d = fabsf(same->data[i] - img->data[i]);
        if (d > max_diff) max_diff = d;
    }
    ASSERT(max_diff < 0.1f, "resize to same size preserves values");

    nt_image_free(img);
    nt_image_free(big);
    nt_image_free(small);
    nt_image_free(same);
}

/* ── Test: center crop ── */
static void test_crop(void) {
    printf("\n── center crop ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);

    nt_image* cropped = nt_image_center_crop(img, 2, 2);
    ASSERT(cropped->width == 2, "crop width == 2");
    ASSERT(cropped->height == 2, "crop height == 2");
    ASSERT(cropped->channels == 3, "crop keeps channels");

    /* Crop larger than image clamps */
    nt_image* big_crop = nt_image_center_crop(img, 8, 8);
    ASSERT(big_crop->width == 4, "overcrop clamped to 4");
    ASSERT(big_crop->height == 4, "overcrop clamped to 4");

    nt_image_free(img);
    nt_image_free(cropped);
    nt_image_free(big_crop);
}

/* ── Test: normalize ── */
static void test_normalize(void) {
    printf("\n── normalize ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);

    float mean[] = {0.5f, 0.5f, 0.5f};
    float std[]  = {0.5f, 0.5f, 0.5f};
    nt_image_normalize(img, mean, std);

    /* After normalize with mean=0.5 std=0.5: [0,1] → [-1, 1] */
    int in_range = 1;
    for (int i = 0; i < 3 * 4 * 4; i++)
        if (img->data[i] < -1.1f || img->data[i] > 1.1f) in_range = 0;
    ASSERT(in_range, "normalized pixels in [-1, 1]");

    /* 0.0 → (0-0.5)/0.5 = -1.0, 1.0 → (1-0.5)/0.5 = 1.0 */
    int has_neg = 0, has_pos = 0;
    for (int i = 0; i < 3 * 4 * 4; i++) {
        if (img->data[i] < -0.5f) has_neg = 1;
        if (img->data[i] > 0.5f) has_pos = 1;
    }
    ASSERT(has_neg && has_pos, "normalize produces both negative and positive");

    nt_image_free(img);
}

/* ── Test: horizontal flip ── */
static void test_hflip(void) {
    printf("\n── horizontal flip ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);

    /* Save original top-left pixel */
    float orig_00_r = img->data[0 * 4 * 4 + 0 * 4 + 0]; /* R[0,0] */
    float orig_03_r = img->data[0 * 4 * 4 + 0 * 4 + 3]; /* R[0,3] */

    nt_image_hflip(img);

    /* After flip: pixel[0,0] should be old pixel[0,3] and vice versa */
    ASSERT_NEAR(img->data[0 * 16 + 0], orig_03_r, 0.01f, "hflip swaps corners (R)");
    ASSERT_NEAR(img->data[0 * 16 + 3], orig_00_r, 0.01f, "hflip swaps corners (R) reverse");

    /* Double flip = original */
    nt_image_hflip(img);
    ASSERT_NEAR(img->data[0 * 16 + 0], orig_00_r, 0.01f, "double hflip = identity");

    nt_image_free(img);
}

/* ── Test: grayscale conversion ── */
static void test_grayscale(void) {
    printf("\n── grayscale ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);
    ASSERT(img->channels == 3, "starts as RGB");

    nt_image_to_gray(img);
    ASSERT(img->channels == 1, "converted to 1 channel");

    /* Gray values should be in [0, 1] */
    int valid = 1;
    for (int i = 0; i < 4 * 4; i++)
        if (img->data[i] < 0 || img->data[i] > 1.01f) valid = 0;
    ASSERT(valid, "gray pixels in [0, 1]");

    nt_image_free(img);
}

/* ── Test: patch extraction ── */
static void test_patches(void) {
    printf("\n── patch extraction ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");
    nt_image* img = nt_image_load("/tmp/notorch_test.bmp", 3);

    /* 4x4 image, 2x2 patches → 4 patches, each 3*2*2=12 dim */
    nt_tensor* p = nt_image_to_patches(img, 2, 2);
    ASSERT(p->ndim == 2, "patches ndim == 2");
    ASSERT(p->shape[0] == 4, "4 patches from 4x4 / 2x2");
    ASSERT(p->shape[1] == 12, "patch dim = 3*2*2 = 12");

    /* All patch values in [0, 1] */
    int valid = 1;
    for (int i = 0; i < p->len; i++)
        if (p->data[i] < 0 || p->data[i] > 1.01f) valid = 0;
    ASSERT(valid, "patch values in [0, 1]");

    /* Single patch covering whole image */
    nt_tensor* p2 = nt_image_to_patches(img, 4, 4);
    ASSERT(p2->shape[0] == 1, "1 patch for full image");
    ASSERT(p2->shape[1] == 48, "full patch dim = 3*4*4 = 48");

    nt_tensor_free(p);
    nt_tensor_free(p2);
    nt_image_free(img);
}

/* ── Test: ViT preprocess pipeline ── */
static void test_vit_preprocess(void) {
    printf("\n── ViT preprocess ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");

    /* 4x4 BMP → resize to 8 → center crop 8x8 → patches 4x4 → [4, 48] */
    nt_tensor* patches = nt_vit_preprocess("/tmp/notorch_test.bmp", 8, 4);
    ASSERT(patches != NULL, "vit_preprocess returns non-NULL");
    ASSERT(patches->shape[0] == 4, "vit: 4 patches from 8x8 / 4x4");
    ASSERT(patches->shape[1] == 48, "vit: patch_dim = 3*4*4 = 48");
    nt_tensor_free(patches);

    /* Nonexistent file */
    nt_tensor* bad = nt_vit_preprocess("/tmp/nope.xyz", 8, 4);
    ASSERT(bad == NULL, "vit_preprocess NULL on bad file");
}

/* ── Test: gray preprocess ── */
static void test_gray_preprocess(void) {
    printf("\n── gray preprocess ──\n");
    write_test_bmp("/tmp/notorch_test.bmp");

    nt_tensor* t = nt_gray_preprocess("/tmp/notorch_test.bmp", 8);
    ASSERT(t != NULL, "gray_preprocess returns non-NULL");
    ASSERT(t->len == 64, "gray 8x8 = 64 values");

    int valid = 1;
    for (int i = 0; i < t->len; i++)
        if (t->data[i] < 0 || t->data[i] > 1.01f) valid = 0;
    ASSERT(valid, "gray preprocess values in [0, 1]");

    nt_tensor_free(t);
}

/* ── Test: BPE encode/decode roundtrip ── */
static void test_bpe(void) {
    printf("\n── BPE tokenizer ──\n");

    /* Test with inline merges */
    const int merges[][2] = {{101, 32}, {116, 104}, {116, 32}};
    nt_bpe bpe;
    nt_bpe_init(&bpe, merges, 3);
    ASSERT(bpe.vocab_size == 259, "bpe vocab = 256 + 3");
    ASSERT(bpe.n_merges == 3, "bpe 3 merges");

    /* Encode/decode */
    const char* text = "the cat";
    int tokens[64];
    int n = nt_bpe_encode(&bpe, text, 7, tokens, 64);
    ASSERT(n > 0 && n < 7, "bpe compresses text");

    char decoded[64];
    int db = nt_bpe_decode(&bpe, tokens, n, decoded, 64);
    ASSERT(db == 7, "bpe decode length matches");
    ASSERT(strcmp(decoded, text) == 0, "bpe roundtrip exact match");

    /* Empty input */
    int n2 = nt_bpe_encode(&bpe, "", 0, tokens, 64);
    ASSERT(n2 == 0, "bpe encode empty = 0 tokens");
}

static void test_conv2d(void) {
    printf("\n-- conv2d (im2col + GEMM) --\n");
    // 1ch 3x3 input, one 2x2 all-ones kernel, stride 1, pad 0 -> 2x2 window sums.
    float in[9]  = {1,2,3, 4,5,6, 7,8,9};
    float w[4]   = {1,1,1,1};
    float out[4] = {0};
    int rc = nt_conv2d(out, in, w, NULL, 1, 3, 3, 1, 2, 2, 1, 0);
    ASSERT(rc == 0, "nt_conv2d returns 0");
    ASSERT_NEAR(out[0], 12.0f, 1e-4f, "conv out[0,0] = 1+2+4+5 = 12");
    ASSERT_NEAR(out[1], 16.0f, 1e-4f, "conv out[0,1] = 2+3+5+6 = 16");
    ASSERT_NEAR(out[2], 24.0f, 1e-4f, "conv out[1,0] = 4+5+7+8 = 24");
    ASSERT_NEAR(out[3], 28.0f, 1e-4f, "conv out[1,1] = 5+6+8+9 = 28");
    float bias[1] = {10.0f};
    nt_conv2d(out, in, w, bias, 1, 3, 3, 1, 2, 2, 1, 0);
    ASSERT_NEAR(out[0], 22.0f, 1e-4f, "conv + bias 10 -> out[0,0] = 22");
}

static void test_group_norm(void) {
    printf("\n-- group_norm --\n");
    // 2 channels (1x2 each), 2 groups -> each channel normalised to {-1,+1}.
    float in[4]  = {1,2, 3,4};   // ch0 = [1,2], ch1 = [3,4]
    float out[4] = {0};
    int rc = nt_group_norm(out, in, NULL, NULL, 2, 1, 2, 2, 1e-5f);
    ASSERT(rc == 0, "nt_group_norm returns 0");
    ASSERT_NEAR(out[0], -1.0f, 1e-3f, "gn 2grp ch0[0] ~ -1");
    ASSERT_NEAR(out[1],  1.0f, 1e-3f, "gn 2grp ch0[1] ~ +1");
    ASSERT_NEAR(out[2], -1.0f, 1e-3f, "gn 2grp ch1[0] ~ -1");
    ASSERT_NEAR(out[3],  1.0f, 1e-3f, "gn 2grp ch1[1] ~ +1");
    // one group over [1,2,3,4] (mean 2.5, std ~1.118), affine gamma=2 beta=1.
    float gamma[2] = {2.0f, 2.0f}, beta[2] = {1.0f, 1.0f};
    nt_group_norm(out, in, gamma, beta, 2, 1, 2, 1, 1e-5f);
    ASSERT_NEAR(out[0], -1.6833f, 2e-3f, "gn 1grp affine [0] = -1.683");
    ASSERT_NEAR(out[3],  3.6833f, 2e-3f, "gn 1grp affine [3] =  3.683");
}

static void test_upsample(void) {
    printf("\n-- upsample (nearest) --\n");
    float in[4]  = {1,2, 3,4};        // 1ch 2x2
    float out[16] = {0};
    nt_upsample_nearest(out, in, 1, 2, 2, 2);  // -> 4x4, each pixel duplicated 2x2
    ASSERT_NEAR(out[0],  1.0f, 1e-6f, "up[0,0] = 1");
    ASSERT_NEAR(out[3],  2.0f, 1e-6f, "up[0,3] = 2");
    ASSERT_NEAR(out[5],  1.0f, 1e-6f, "up[1,1] = 1");
    ASSERT_NEAR(out[8],  3.0f, 1e-6f, "up[2,0] = 3");
    ASSERT_NEAR(out[15], 4.0f, 1e-6f, "up[3,3] = 4");
}

static void test_attention(void) {
    printf("\n-- attention (scaled dot-product) --\n");
    // self-attn T=S=2, d=2; Q=K=identity -> softmax(I/sqrt2) @ V.
    float Q[4] = {1,0, 0,1}, K[4] = {1,0, 0,1}, V[4] = {1,2, 3,4};
    float out[4] = {0};
    int rc = nt_attention(out, Q, K, V, 2, 2, 2);
    ASSERT(rc == 0, "nt_attention returns 0");
    ASSERT_NEAR(out[0], 1.6605f, 5e-3f, "attn out[0,0]");
    ASSERT_NEAR(out[1], 2.6605f, 5e-3f, "attn out[0,1]");
    ASSERT_NEAR(out[2], 2.3395f, 5e-3f, "attn out[1,0]");
    ASSERT_NEAR(out[3], 3.3395f, 5e-3f, "attn out[1,1]");
    // cross-attn T=1, S=2 (context) -> the conditioning path.
    float Qc[2] = {1,1}, Kc[4] = {1,1, 0,0}, Vc[4] = {10,20, 30,40};
    float outc[2] = {0};
    nt_attention(outc, Qc, Kc, Vc, 1, 2, 2);
    ASSERT_NEAR(outc[0], 13.914f, 5e-2f, "cross-attn out[0]");
    ASSERT_NEAR(outc[1], 23.914f, 5e-2f, "cross-attn out[1]");
}

int main(void) {
    printf("═══════════════════════════════════════════\n");
    printf("  notorch vision + BPE tests\n");
    printf("═══════════════════════════════════════════\n");

    test_load();
    test_resize();
    test_crop();
    test_normalize();
    test_hflip();
    test_grayscale();
    test_patches();
    test_vit_preprocess();
    test_gray_preprocess();
    test_bpe();
    test_conv2d();
    test_group_norm();
    test_upsample();
    test_attention();

    printf("\n═══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════════\n");
    return n_fail > 0 ? 1 : 0;
}
