/* lfm3_omni.h — the parts of LFM3-Omni-2B beside the text decoder.
 *
 * The decoder itself is harness/arch_lfm3.c; it reads embedding rows through
 * forward_embd. Everything that turns a picture, a sound or a speech frame into such a
 * row, and a speech frame back into sound, is here, loaded from the same GGUF
 * tools/lfm3_omni_to_gguf writes (tensor names of the official mmproj file):
 *
 *   vision   frames of normalized pixels -> one 2048-wide row per 2x2 patch block
 *   audio    16 kHz mono -> log-mel -> one row per eight mel frames
 *   depth    the decoder's final hidden state -> logits of the twelve codebooks
 *   codec    codes -> 24 kHz waveform
 *
 * Every function returns 0 or -1 and writes nothing on -1. Prints go to stderr. */
#ifndef NT_HARNESS_LFM3_OMNI_H
#define NT_HARNESS_LFM3_OMNI_H

#include "harness/runtime.h"
#include <stdint.h>

typedef struct lfm3_omni lfm3_omni;

typedef struct {
    int width;                      /* text decoder width, the size of every row produced here */
    int patch, merge, min_pixels, max_pixels;
    int sample_rate, n_mel, hop, stride;
    int n_codebooks, codebook_size, eos_code, schedule[16], n_schedule;
    int interleave_text, interleave_audio, audio_rep_window;
    float audio_temperature, audio_rep_penalty;
    int codec_rate, codec_samples_per_frame;
    int tok_image, tok_video, tok_audio, tok_audio_out, tok_audio_start, tok_mixed_start,
        tok_text_end, tok_media_start, tok_media_end, tok_im_start, tok_im_end;
} lfm3_omni_info;

lfm3_omni *lfm3_omni_load(gguf_file *gf);
void lfm3_omni_free(lfm3_omni *o);
const lfm3_omni_info *lfm3_omni_get_info(const lfm3_omni *o);

/* The size the official image processor resizes an h x w picture to: both sides a
 * multiple of patch*merge, the pixel count inside [min_pixels, max_pixels]. */
int lfm3_omni_image_size(const lfm3_omni *o, int h, int w, int *out_h, int *out_w);
/* RGB bytes h x w -> [3][oh][ow] floats, resized (antialiased bicubic on bytes, as the
 * torchvision processor does) and normalized to [-1, 1]. Caller frees *out. */
int lfm3_omni_image_prepare(const lfm3_omni *o, const uint8_t *rgb, int h, int w,
                            float **out, int *oh, int *ow);
/* n_groups temporal groups of two frames each, frames[g][0..1] pointing at [3][h][w]
 * planes (a still image passes the same plane twice). out receives
 * n_groups * (h/32) * (w/32) rows of `width`, group by group, block rows in raster order. */
int lfm3_omni_vision_encode(lfm3_omni *o, const float *const (*frames)[2], int n_groups,
                            int h, int w, float *out);

/* 16 kHz mono -> [n_mel][*n_frames] log-mel, the official extractor's padding and scale. */
int lfm3_omni_audio_mel(const lfm3_omni *o, const float *pcm, int n, float **mel, int *n_frames);
/* mel [n_mel][n_frames] -> ceil(n_frames/8) rows of `width`. */
int lfm3_omni_audio_encode(lfm3_omni *o, const float *mel, int n_frames, float *out);

/* One MaskGIT pass: hidden (width) and the frame's codes, masked[k] != 0 for codebooks
 * still hidden; logits [n_codebooks][codebook_size]. */
int lfm3_omni_depth_logits(lfm3_omni *o, const float *hidden, const int *codes,
                           const uint8_t *masked, float *logits);
/* A speech frame fed back to the decoder: the sum of its codebook embeddings. */
int lfm3_omni_frame_embed(const lfm3_omni *o, const int *codes, float *out);
/* codes [n_frames][n_codebooks] (no end-of-audio frames) -> n_frames * samples_per_frame. */
int lfm3_omni_codec_decode(lfm3_omni *o, const int *codes, int n_frames, float *pcm);

/* Intermediate states for parity checks, NULL to skip: called with a tag and the rows. */
typedef void (*lfm3_omni_probe)(void *user, const char *tag, int layer, const float *x, int n, int width);
void lfm3_omni_set_probe(lfm3_omni *o, lfm3_omni_probe fn, void *user);

#endif
