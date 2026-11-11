/* Native Qwen3.5 still-image path. Source-layout vision GGUF, F32 arithmetic.
 * Geometry is explicit: physical cache offsets and 3-axis rotary positions differ.
 * Caller owns GGUF lifetime; all returned image/feature buffers use free(). */
#ifndef NT_QWEN35_VISION_H
#define NT_QWEN35_VISION_H
#include "harness/arch.h"
#include <stdint.h>
typedef struct nt_qwen35_vision nt_qwen35_vision;
typedef struct {
    int width, output_width, layers, heads, patch, temporal, merge, positions;
    int min_pixels, max_pixels, rope_dim, section[3], image_id, start_id, end_id;
} nt_qwen35_vision_info;
typedef struct {
    int height, width, grid_h, grid_w, tokens;
    float *patches; /* [grid_h*grid_w, 3*temporal*patch*patch], merge-block order */
} nt_qwen35_image;
nt_qwen35_vision *nt_qwen35_vision_load(gguf_file *g);
void nt_qwen35_vision_free(nt_qwen35_vision *v);
const nt_qwen35_vision_info *nt_qwen35_vision_get_info(const nt_qwen35_vision *v);
/* max_pixels=0 uses the checkpoint limit; a positive cap must be >= min_pixels. */
int nt_qwen35_image_prepare(const nt_qwen35_vision *v, const uint8_t *rgb, int h, int w,
                           int max_pixels, nt_qwen35_image *out);
void nt_qwen35_image_free(nt_qwen35_image *im);
/* Allocate [image.tokens, output_width]. Output pointer unchanged on failure. */
int nt_qwen35_vision_encode(const nt_qwen35_vision *v, const nt_qwen35_image *im, float **out);
/* Insert one image run into an already-tokenized sequence; placeholders must occur
 * exactly once between vision_start/end, with no other visual markers. Writes
 * positions/rows only after validation. Returns next rotary position, or <0. */
int nt_qwen35_image_positions(const nt_qwen35_vision *v, const nt_qwen35_image *im,
    const float *features, const int *ids, int n, int (*positions)[3], const float **rows);
/* Supplied positions are [n,3] T/H/W, physical pos indexes KV and recurrent state.
 * NULL positions keeps ordinary 1D RoPE. sections required with 3D positions.
 * Same callback semantics as forward_residual; logits untouched on failure. */
int nt_qwen35_forward_3d(void *model, kv_cache *kv, const int *tokens, const float *const *rows,
    const int (*positions)[3], const int sections[3], int n, int pos, float *logits,
    nt_residual_fn callback, void *user);
#endif
