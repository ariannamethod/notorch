/* Host mirror emulator for the Chuck device contract gate.
 * Device buffers are independent heap allocations; no CUDA kernel executes.
 * Only the narrow allocation/norm/Chuck surface is provided. Link with section
 * garbage collection so an accidentally introduced GPU operation fails to link.
 */
#include "notorch_cuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long dispatches, downloads;
long long chuck_host_download_count(void) { return downloads; }
int gpu_init(void) { return 0; }
void gpu_shutdown(void) { }
long long nt_gpu_dispatch_count(void) { return dispatches; }
void nt_gpu_dispatch_reset(void) { dispatches = 0; }
float *gpu_alloc(int n) { return calloc((size_t)n, sizeof(float)); }
void gpu_free(float *p) { free(p); }
void gpu_upload(float *d, const float *h, int n) { memcpy(d, h, (size_t)n * sizeof(float)); }
void gpu_download(float *h, const float *d, int n) {
    memcpy(h, d, (size_t)n * sizeof(float));
    downloads++;
}
void gpu_zero(float *p, int n) { memset(p, 0, (size_t)n * sizeof(float)); }
void gpu_mark_all_dirty(void) { }
void gpu_sgemm_nt(int rows, int cols, int inner, const float *a, const float *b, float *out) {
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        float sum = 0;
        for (int k = 0; k < inner; ++k) sum += a[i * inner + k] * b[j * inner + k];
        out[i * cols + j] = sum;
    }
    dispatches++;
}
float gpu_nrm2(const float *p, int n) {
    float sum = 0;
    for (int i = 0; i < n; ++i) sum += p[i] * p[i];
    dispatches++;
    return sqrtf(sum);
}
void gpu_nrm2_batch(const float **p, const int *n, int count, float *out) {
    for (int i = 0; i < count; ++i) out[i] = p[i] && n[i] ? gpu_nrm2(p[i], n[i]) : 0;
}
void gpu_chuck_inner(float *p, float *m, float *v, const float *g, int n,
                     float b1, float b2, float bc1, float bc2, float lr, float eps) {
    for (int i = 0; i < n; ++i) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        float mh = m[i] / bc1, vh = v[i] / bc2;
        p[i] -= lr * mh / (sqrtf(vh) + eps);
    }
    dispatches++;
}
