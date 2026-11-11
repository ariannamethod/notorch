#include "harness/image.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "stb_image.h"

uint8_t *nt_image_load(const char *path, int *h, int *w) {
    int c = 0,ih,iw;
    if(!path || !h || !w || !stbi_info(path,&iw,&ih,&c) || iw<1 || ih<1 ||
       iw>65536 || ih>65536 || (size_t)iw*ih>(1u<<26))return NULL;
    uint8_t *px = stbi_load(path, w, h, &c, 3);
    if (!px) { fprintf(stderr, "image: cannot decode %s\n", path); return NULL; }
    /* stb's buffer goes back through free() like every other buffer here */
    size_t n = (size_t)*w * *h * 3;
    uint8_t *own = malloc(n);
    if (own) memcpy(own, px, n);
    stbi_image_free(px);
    return own;
}

/* Antialiased bicubic (a = -0.5) on bytes, one axis at a time, width first — the native
 * uint8 kernel torch runs for torchvision's BICUBIC resize on every CPU (PIL's algorithm):
 * normalized double weights per output, then the whole table rounded to int16 at the
 * largest precision its biggest weight allows, then integer sums rounded and shifted back
 * to bytes after each pass. */
static double cubic(double x) {
    const double a = -0.5;
    x = fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}
static int resample_axis(const uint8_t *src, uint8_t *dst, int n_in, int n_out, int lines, int stride_in,
                         int stride_line_in, int stride_out, int stride_line_out, int pil) {
    double scale = (double)n_in / n_out, support = 2.0 * (scale >= 1 ? scale : 1), inv = scale >= 1 ? 1.0 / scale : 1.0;
    int ksize = (int)ceil(support) * 2 + 1;
    double *k = malloc((size_t)n_out * ksize * sizeof(double));
    int32_t *k16 = malloc((size_t)n_out * ksize * sizeof(int32_t));
    int *lo = malloc((size_t)n_out * sizeof(int)), *len = malloc((size_t)n_out * sizeof(int));
    if (!k || !k16 || !lo || !len) { free(k); free(k16); free(lo); free(len); return -1; }
    double wmax = 0;
    for (int i = 0; i < n_out; i++) {
        double center = scale * (i + 0.5);
        long a = (long)(center - support + 0.5); if (a < 0) a = 0;
        long b = (long)(center + support + 0.5); if (b > n_in) b = n_in;
        lo[i] = (int)a; len[i] = (int)(b - a);
        if (len[i] > ksize) len[i] = ksize;
        double *w = k + (size_t)i * ksize, total = 0;
        for (int j = 0; j < len[i]; j++) { w[j] = cubic((j + a - center + 0.5) * inv); total += w[j]; }
        for (int j = 0; j < len[i]; j++) { if (total != 0) w[j] /= total; if (w[j] > wmax) wmax = w[j]; }
    }
    int prec = 0;
    for (prec = 0; prec < 22; prec++)
        if ((int)(0.5 + wmax * (1 << (prec + 1))) >= (1 << 15)) break;
    if(pil)prec=22; /* PIL uses 22 fractional bits and int32 coefficients. */
    for (int i = 0; i < n_out; i++)
        for (int j = 0; j < len[i]; j++) {
            double v = k[(size_t)i * ksize + j] * (1 << prec);
            k16[(size_t)i * ksize + j] = (int32_t)(v < 0 ? v - 0.5 : v + 0.5);
        }
    for (int i = 0; i < n_out; i++) {
        const int32_t *w = k16 + (size_t)i * ksize;
        for (int l = 0; l < lines; l++) {
            long long t = prec ? 1LL << (prec - 1) : 0;
            const uint8_t *s = src + (size_t)l * stride_line_in + (size_t)lo[i] * stride_in;
            for (int j = 0; j < len[i]; j++) t += (long long)s[(size_t)j * stride_in] * w[j];
            long long v = t >> prec;
            dst[(size_t)l * stride_line_out + (size_t)i * stride_out] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    free(k); free(k16); free(lo); free(len);
    return 0;
}

static int resize_rgb(const uint8_t *rgb, int h, int w, int H, int W, uint8_t **out, int pil) {
    if (!rgb || !out || h<1 || w<1 || H<1 || W<1 ||
        h>65536 || w>65536 || H>65536 || W>65536 ||
        (size_t)h*w > (1u<<28) || (size_t)H*W > (1u<<28) || (size_t)h*W > (1u<<28)) return -1;
    uint8_t *tmp=malloc((size_t)h*W*3), *res=malloc((size_t)H*W*3);
    if (!tmp || !res) { free(tmp); free(res); return -1; }
    for (int c=0; c<3; c++) {
        if(W==w) { for(int y=0;y<h;y++) for(int x=0;x<w;x++) tmp[((size_t)y*W+x)*3+c]=rgb[((size_t)y*w+x)*3+c]; }
        else if(resample_axis(rgb+c,tmp+c,w,W,h,3,w*3,3,W*3,pil)) goto bad;
        if(H==h) { for(int y=0;y<H;y++) for(int x=0;x<W;x++) res[((size_t)y*W+x)*3+c]=tmp[((size_t)y*W+x)*3+c]; }
        else if(resample_axis(tmp+c,res+c,h,H,W,W*3,3,W*3,3,pil)) goto bad;
    }
    free(tmp); *out=res; return 0;
bad: free(tmp); free(res); return -1;
}

int nt_image_resize(const uint8_t *rgb,int h,int w,int H,int W,uint8_t **out){return resize_rgb(rgb,h,w,H,W,out,0);}
int nt_image_resize_pil(const uint8_t *rgb,int h,int w,int H,int W,uint8_t **out){return resize_rgb(rgb,h,w,H,W,out,1);}
