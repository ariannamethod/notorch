/* Shared RGB file decoding and antialiased byte resize. Buffers use free(). */
#ifndef NT_IMAGE_H
#define NT_IMAGE_H
#include <stdint.h>
uint8_t *nt_image_load(const char *path, int *h, int *w);
int nt_image_resize(const uint8_t *rgb, int h, int w, int oh, int ow, uint8_t **out);
/* PIL coefficient precision, used by the declared Qwen PIL processor. */
int nt_image_resize_pil(const uint8_t *rgb, int h, int w, int oh, int ow, uint8_t **out);
#endif
