/* lfm3_omni_probe — run one tower of LFM3-Omni on a fixture and dump what it computed.
 *
 *   lfm3_omni_probe fixtures DIR                   write tone.wav (1 s, 440 Hz, 16 kHz) and grad.ppm (256x256)
 *   lfm3_omni_probe vision MODEL.gguf IMAGE OUT    pixels, every layer, projected rows
 *   lfm3_omni_probe audio  MODEL.gguf WAV   OUT    mel, downsampler, every layer, projected rows
 *   lfm3_omni_probe depth  MODEL.gguf SEED  OUT    logits for a seeded hidden state and three masks
 *   lfm3_omni_probe codec  MODEL.gguf SEED FRAMES OUT   seeded codes -> waveform
 *   lfm3_omni_compare OURS ORACLE                  (tools/lfm3_omni_compare.c) reads two dumps
 *
 * A dump is "NTD1" then records: u32 tag length, tag, i32 layer, u32 rows, u32 cols,
 * u32 dtype (0 f32, 1 f64), data. The oracle writes the same records from the official
 * PyTorch modules, so the two can be compared tag by tag. Seeds drive a 64-bit LCG whose
 * draws the oracle reproduces, so no random tensor has to cross between machines. */
#include "harness/lfm3_omni.h"
#include "harness/lfm3_omni_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *dump;

static void put(const char *tag, int layer, const float *x, int rows, int cols) {
    uint32_t tl = (uint32_t)strlen(tag), r = (uint32_t)rows, c = (uint32_t)cols, dt = 0;
    int32_t l = layer;
    fwrite(&tl, 4, 1, dump); fwrite(tag, 1, tl, dump); fwrite(&l, 4, 1, dump);
    fwrite(&r, 4, 1, dump); fwrite(&c, 4, 1, dump); fwrite(&dt, 4, 1, dump);
    fwrite(x, sizeof(float), (size_t)rows * cols, dump);
}
static void on_probe(void *user, const char *tag, int layer, const float *x, int n, int w) {
    (void)user;
    put(tag, layer, x, n, w);
}

/* x_{k+1} = x_k * 6364136223846793005 + 1442695040888963407 (mod 2^64); a draw is the
 * top 24 bits scaled to [-1, 1). */
static uint64_t lcg;
static float draw(void) {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)(lcg >> 40) / 8388608.0 - 1.0);
}
static uint32_t draw_int(uint32_t n) {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)((lcg >> 33) % n);
}

static int fixtures(const char *dir) {
    char path[4096];
    float pcm[16000];
    for (int i = 0; i < 16000; i++) pcm[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / 16000.0f);
    snprintf(path, sizeof path, "%s/tone.wav", dir);
    if (lfm3_wav_write(path, pcm, 16000, 16000)) return 1;
    snprintf(path, sizeof path, "%s/grad.ppm", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    fprintf(f, "P6\n256 256\n255\n");
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            unsigned char px[3] = {(unsigned char)x, (unsigned char)y, (unsigned char)((x * 3 + y * 5) & 255)};
            fwrite(px, 1, 3, f);
        }
    return fclose(f) != 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "fixtures")) return fixtures(argv[2]);
    if (argc < 5) {
        fprintf(stderr, "usage: %s fixtures DIR | vision|audio MODEL INPUT OUT | depth MODEL SEED OUT | codec MODEL SEED FRAMES OUT\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    gguf_file *gf = gguf_open(argv[2]);
    if (!gf) return 1;
    /* the decoder is not loaded here; its vocabulary is token_embd's row count */
    int te = gguf_find_tensor(gf, "token_embd.weight");
    int vocab = te >= 0 && gf->tensors[te].ndim == 2 ? (int)gf->tensors[te].shape[1] : 0;
    lfm3_omni *o = lfm3_omni_load(gf, vocab);
    if (!o) { gguf_close(gf); return 1; }
    const lfm3_omni_info *I = lfm3_omni_get_info(o);
    const char *outp = argv[argc - 1];
    dump = fopen(outp, "wb");
    if (!dump) { perror(outp); return 1; }
    fwrite("NTD1", 1, 4, dump);
    lfm3_omni_set_probe(o, on_probe, NULL);
    int rc = 1;
    double t0 = now_ms();
    if (!strcmp(mode, "vision")) {
        int h, w;
        uint8_t *rgb = lfm3_image_load(argv[3], &h, &w);
        float *px = NULL;
        int H, W;
        if (rgb && !lfm3_omni_image_prepare(o, rgb, h, w, &px, &H, &W)) {
            put("vision.pixels", -1, px, 3 * H, W);
            int rows = (H / 32) * (W / 32);
            float *out = malloc((size_t)rows * I->width * sizeof(float));
            const float *pair[1][2] = {{px, px}};
            fprintf(stderr, "vision: %dx%d -> %dx%d, %d rows\n", w, h, W, H, rows);
            if (out && !lfm3_omni_vision_encode(o, pair, 1, H, W, out)) rc = 0;
            free(out);
        }
        free(rgb); free(px);
    } else if (!strcmp(mode, "audio")) {
        float *pcm = NULL, *mel = NULL;
        int n = 0, rate = 0, frames = 0;
        if (!lfm3_wav_read(argv[3], &pcm, &n, &rate) && rate == I->sample_rate &&
            !lfm3_omni_audio_mel(o, pcm, n, &mel, &frames)) {
            put("audio.mel", -1, mel, I->n_mel, frames);
            int rows = (frames + I->stride - 1) / I->stride;
            float *out = malloc((size_t)rows * I->width * sizeof(float));
            fprintf(stderr, "audio: %d samples -> %d mel frames -> %d rows\n", n, frames, rows);
            if (out && !lfm3_omni_audio_encode(o, mel, frames, out)) rc = 0;
            free(out);
        } else fprintf(stderr, "audio: %s is not a readable %d Hz WAV\n", argv[3], I->sample_rate);
        free(pcm); free(mel);
    } else if (!strcmp(mode, "depth")) {
        lcg = strtoull(argv[3], NULL, 10);
        int NC = I->n_codebooks, CS = I->codebook_size;
        float *hid = malloc((size_t)I->width * sizeof(float)), *lg = malloc((size_t)NC * CS * sizeof(float));
        int codes[64];
        uint8_t mask[64];
        if (hid && lg) {
            for (int i = 0; i < I->width; i++) hid[i] = draw();
            for (int k = 0; k < NC; k++) codes[k] = (int)draw_int((uint32_t)CS - 1);
            put("depth.hidden", -1, hid, 1, I->width);
            float cf[64];
            for (int k = 0; k < NC; k++) cf[k] = (float)codes[k];
            put("depth.codes", -1, cf, 1, NC);
            rc = 0;
            for (int pass = 0; pass < 3 && !rc; pass++) {
                for (int k = 0; k < NC; k++) mask[k] = pass == 0 ? 1 : pass == 1 ? (k % 3 != 0) : 0;
                if (lfm3_omni_depth_logits(o, hid, codes, mask, lg)) rc = 1;
                else put("depth.logits", pass, lg, NC, CS);
            }
            float *fe = malloc((size_t)I->width * sizeof(float));
            if (!rc && fe && !lfm3_omni_frame_embed(o, codes, fe)) put("depth.frame_embed", -1, fe, 1, I->width);
            else rc = 1;
            free(fe);
        }
        free(hid); free(lg);
    } else if (!strcmp(mode, "codec") && argc == 6) {
        lcg = strtoull(argv[3], NULL, 10);
        int T = atoi(argv[4]), NC = I->n_codebooks;
        int *codes = malloc((size_t)T * NC * sizeof(int));
        float *pcm = malloc((size_t)T * I->codec_samples_per_frame * sizeof(float));
        float *cf = malloc((size_t)T * NC * sizeof(float));
        if (T > 0 && codes && pcm && cf) {
            for (int i = 0; i < T * NC; i++) { codes[i] = (int)draw_int((uint32_t)I->codebook_size - 1); cf[i] = (float)codes[i]; }
            put("codec.codes", -1, cf, T, NC);
            if (!lfm3_omni_codec_decode(o, codes, T, pcm)) {
                put("codec.pcm", -1, pcm, 1, T * I->codec_samples_per_frame);
                rc = 0;
            }
        }
        free(codes); free(pcm); free(cf);
    } else fprintf(stderr, "unknown mode %s\n", mode);
    fprintf(stderr, "%s: %s in %.0f ms\n", mode, rc ? "FAILED" : "done", now_ms() - t0);
    if (fclose(dump)) rc = 1;
    lfm3_omni_free(o);
    gguf_close(gf);
    return rc;
}
