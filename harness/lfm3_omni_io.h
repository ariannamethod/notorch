/* lfm3_omni_io.h — the files the omni towers read and write: images through stb_image
 * (PNG, JPEG, BMP, PNM), and RIFF WAV. */
#ifndef NT_HARNESS_LFM3_OMNI_IO_H
#define NT_HARNESS_LFM3_OMNI_IO_H

#include <stdint.h>

/* RGB bytes, h x w x 3; NULL on failure. Caller frees. */
uint8_t *lfm3_image_load(const char *path, int *h, int *w);
/* PCM 8/16/24/32-bit or IEEE float 32/64, any channel count (averaged to mono), samples
 * scaled to [-1, 1) as soundfile scales them. No resampling: *rate says what the file is. */
int lfm3_wav_read(const char *path, float **pcm, int *n, int *rate);
/* Mono 16-bit PCM, samples clipped to [-1, 1]. */
int lfm3_wav_write(const char *path, const float *pcm, int n, int rate);

#endif
