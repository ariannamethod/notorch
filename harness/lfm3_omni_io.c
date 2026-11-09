/* lfm3_omni_io.c — see lfm3_omni_io.h. */
#include "harness/lfm3_omni_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "stb_image.h"

uint8_t *lfm3_image_load(const char *path, int *h, int *w) {
    int c = 0;
    uint8_t *px = stbi_load(path, w, h, &c, 3);
    if (!px) { fprintf(stderr, "image: cannot decode %s\n", path); return NULL; }
    /* stb's buffer goes back through free() like every other buffer here */
    size_t n = (size_t)*w * *h * 3;
    uint8_t *own = malloc(n);
    if (own) memcpy(own, px, n);
    stbi_image_free(px);
    return own;
}

static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

int lfm3_wav_read(const char *path, float **pcm, int *n, int *rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    uint8_t hdr[12], ch[8];
    int fmt = 0, chans = 0, bits = 0, rc = -1;
    uint32_t sr = 0;
    uint8_t *data = NULL;
    uint32_t dlen = 0;
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) goto done;
    while (fread(ch, 1, 8, f) == 8) {
        uint32_t len = le32(ch + 4);
        if (!memcmp(ch, "fmt ", 4)) {
            uint8_t b[40] = {0};
            if (len < 16 || len > sizeof b || fread(b, 1, len, f) != len) goto done;
            fmt = le16(b); chans = le16(b + 2); sr = le32(b + 4); bits = le16(b + 14);
            if (fmt == 0xFFFE && len >= 26) fmt = le16(b + 24);
            if (len & 1) fgetc(f);
        } else if (!memcmp(ch, "data", 4)) {
            if (len > (1u << 30) || !(data = malloc(len ? len : 1)) || fread(data, 1, len, f) != len) goto done;
            dlen = len;
            break;
        } else if (fseek(f, (long)len + (len & 1), SEEK_CUR)) goto done;
    }
    if (!data || chans < 1 || chans > 32 || !sr || !(fmt == 1 || fmt == 3) ||
        (fmt == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32) || (fmt == 3 && bits != 32 && bits != 64)) {
        fprintf(stderr, "wav: %s: format %d, %d bits, %d channels is not read here\n", path, fmt, bits, chans);
        goto done;
    }
    int bps = bits / 8, frames = (int)(dlen / (uint32_t)(bps * chans));
    float *out = malloc((size_t)(frames ? frames : 1) * sizeof(float));
    if (!out) goto done;
    for (int i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < chans; c++) {
            const uint8_t *p = data + ((size_t)i * chans + c) * bps;
            double v;
            if (fmt == 3 && bits == 32) { float x; memcpy(&x, p, 4); v = x; }
            else if (fmt == 3) { double x; memcpy(&x, p, 8); v = x; }
            else if (bits == 8) v = (p[0] - 128) / 128.0;
            else if (bits == 16) v = (int16_t)le16(p) / 32768.0;
            else if (bits == 24) v = ((int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24) >> 8) / 8388608.0;
            else v = (int32_t)le32(p) / 2147483648.0;
            acc += v;
        }
        out[i] = (float)(acc / chans);
    }
    *pcm = out; *n = frames; *rate = (int)sr;
    rc = 0;
done:
    if (rc) fprintf(stderr, "wav: cannot read %s\n", path);
    free(data);
    fclose(f);
    return rc;
}

static void w32(FILE *f, uint32_t v) { uint8_t b[4] = {v, v >> 8, v >> 16, v >> 24}; fwrite(b, 1, 4, f); }
static void w16(FILE *f, uint16_t v) { uint8_t b[2] = {v, v >> 8}; fwrite(b, 1, 2, f); }

int lfm3_wav_write(const char *path, const float *pcm, int n, int rate) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    fwrite("RIFF", 1, 4, f); w32(f, 36 + (uint32_t)n * 2); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(f, 16); w16(f, 1); w16(f, 1); w32(f, (uint32_t)rate); w32(f, (uint32_t)rate * 2);
    w16(f, 2); w16(f, 16);
    fwrite("data", 1, 4, f); w32(f, (uint32_t)n * 2);
    for (int i = 0; i < n; i++) {
        float v = pcm[i] > 1 ? 1 : pcm[i] < -1 ? -1 : pcm[i];
        long s = lrintf(v * 32767.0f);
        w16(f, (uint16_t)(int16_t)s);
    }
    return fclose(f) ? -1 : 0;
}
