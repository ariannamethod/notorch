/* Version 1 of the portable spore's ordered residual frame. */
#ifndef NT_SPORE_FRAME_H
#define NT_SPORE_FRAME_H

#include "harness/experts.h"

enum { NT_SPORE_ANCHORS = 32 };
#define NT_SPORE_FRAME_ID "mid-attention-literals-qr-v1"

int nt_spore_frame_build(const nt_expert_body *body, const gguf_file *host, float *frame,
                         char norm_sha[65]);
int nt_spore_frame_valid(const float *frame, int width);
double nt_spore_frame_min_cos(const float *left, const float *right, int width);
/* Replace the target vector's coordinates along its frame with the corresponding
 * source coordinates. Strides address columns of A or contiguous B/vote rows. */
int nt_spore_project(const float *source_frame, int source_width,
                     const float *source, int source_stride,
                     const float *target_frame, int target_width,
                     float *target, int target_stride);
#endif
