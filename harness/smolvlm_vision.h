/* SmolVLM / Idefics3 still-image path from a "clip" mmproj GGUF (projector_type idefics3):
 * the tile grid, the SigLIP tower, the pixel-shuffle connector, the prompt layout and the
 * rows a decoder's forward_embd reads at the <image> positions. Every size comes from the
 * projector's keys and every marker is found by its text in the body's vocabulary.
 * F32 arithmetic. Caller owns GGUF lifetime; all returned buffers use free(). */
#ifndef NT_SMOLVLM_VISION_H
#define NT_SMOLVLM_VISION_H
#include "harness/arch.h"
#include "examples/bpe.h"
#include <stdint.h>
typedef struct nt_smolvlm_vision nt_smolvlm_vision;
typedef struct {
    int width, layers, heads, ffn, image_size, patch, scale, text_width;
    int patches, tokens; /* per frame: tower rows (image_size/patch)^2, rows out patches/scale^2 */
    int longest;         /* clip.vision.preproc_image_size, 0 when the file has none */
    float eps, mean[3], std[3];
} nt_smolvlm_vision_info;
typedef struct {
    int frames, rows, cols; /* rows x cols tiles, then the global frame; 1x1 is the global alone */
    float *pixels;          /* [frames, 3, image_size, image_size], normalized */
} nt_smolvlm_image;
nt_smolvlm_vision *nt_smolvlm_vision_load(gguf_file *g);
void nt_smolvlm_vision_free(nt_smolvlm_vision *v);
const nt_smolvlm_vision_info *nt_smolvlm_vision_get_info(const nt_smolvlm_vision *v);
/* rgb is [h, w, 3] bytes. longest=0 takes preproc_image_size and is refused without it. */
int nt_smolvlm_image_prepare(const nt_smolvlm_vision *v, const uint8_t *rgb, int h, int w,
                             int longest, nt_smolvlm_image *out);
void nt_smolvlm_image_free(nt_smolvlm_image *im);
/* One frame [3, image_size, image_size] -> hidden [patches, width] after the post LayerNorm.
 * hidden is written only on success. */
int nt_smolvlm_vision_tower(const nt_smolvlm_vision *v, const float *frame, float *hidden);
/* hidden [patches, width] -> rows [tokens, text_width]: pixel shuffle, then mm.model.fc. */
int nt_smolvlm_vision_connect(const nt_smolvlm_vision *v, const float *hidden, float *rows);
/* Every frame in order into a new [frames*tokens, text_width]. Output unchanged on failure. */
int nt_smolvlm_vision_encode(const nt_smolvlm_vision *v, const nt_smolvlm_image *im, float **out);
/* The user turn for one image and an instruction through the generation prompt, encoded with
 * the vocabulary's control tokens. Returns the id count (< cap), or <0 when a marker is not a
 * token, the instruction spells one, the tokenizer prepends BOS, or cap is short. */
int nt_smolvlm_prompt(const nt_smolvlm_vision *v, const nt_smolvlm_image *im,
                      const bpe_tokenizer *tok, const char *instruction, int *ids, int cap);
/* rows[t] = the k-th feature row at the k-th <image> id, NULL elsewhere. The <image> count
 * must equal frames*tokens; rows are written only after that check. */
int nt_smolvlm_image_rows(const nt_smolvlm_vision *v, const nt_smolvlm_image *im,
                          const float *features, const bpe_tokenizer *tok,
                          const int *ids, int n, const float **rows);
#endif
