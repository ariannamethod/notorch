/* Partial SIMD tiles, transpose/stride handling, and output bounds.
 * cc -O2 -mavx2 -mfma -DUSE_SIMD -I. tests/test_simd_tails.c -lm -pthread -o test_simd_tails
 * Run each pool size in a fresh process: NT_SIMD_THREADS=4 ./test_simd_tails --exact
 * --exact requires bitwise agreement with the shim's ordered FMA arithmetic.
 */
#include "notorch_simd.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GUARD = 16 };
typedef struct {
    float *allocation, *data, *before;
    size_t length;
    int rows, cols, ld;
} matrix;
typedef struct {
    int m, n, k, ta, tb, pad;
    float alpha, beta;
} shape;

static size_t cases, cells, different_bits;
static float max_error;
static double max_double_error;
static int exact;

static void fail(const shape *s, const char *what, size_t at, float got, float want) {
    fprintf(stderr,"FAIL M=%d N=%d K=%d %c%c pad=%d alpha=%g beta=%g: %s "
            "at %zu, got=%.9g want=%.9g\n",s->m,s->n,s->k,
            s->ta?'T':'N',s->tb?'T':'N',s->pad,s->alpha,s->beta,what,at,got,want);
    exit(1);
}

static float sample(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (float)((*seed >> 8) & 65535u) / 65536.0f - .5f;
}

static matrix new_matrix(int rows, int cols, int padding, int guard_rows, uint32_t *seed) {
    matrix m = {.rows=rows, .cols=cols, .ld=cols+padding};
    m.length = 2*GUARD + (size_t)(rows+guard_rows)*m.ld;
    m.allocation = malloc(m.length*sizeof(float));
    m.before = malloc(m.length*sizeof(float));
    if (!m.allocation || !m.before) { fputs("allocation failed\n",stderr); exit(1); }
    m.data = m.allocation + GUARD;
    for (size_t i = 0; i < m.length; i++) m.allocation[i] = -12345.25f;
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++)
        m.data[(size_t)i*m.ld+j] = sample(seed);
    memcpy(m.before,m.allocation,m.length*sizeof(float));
    return m;
}

static void check_storage(const shape *s, const matrix *m, int output, const char *name) {
    for (size_t i = 0; i < m->length; i++) {
        size_t p = i >= GUARD ? i-GUARD : SIZE_MAX;
        if (output && p < (size_t)m->rows*m->ld && p%m->ld < (size_t)m->cols) continue;
        if (memcmp(m->allocation+i,m->before+i,sizeof(float)))
            fail(s,name,i,m->allocation[i],m->before[i]);
    }
}

static void free_matrix(matrix *m) {
    free(m->allocation);
    free(m->before);
}

static void run_case(shape s) {
    uint32_t seed = 0x7184au + (uint32_t)cases*37u;
    matrix a = new_matrix(s.ta?s.k:s.m,s.ta?s.m:s.k,s.pad?3:0,0,&seed);
    matrix b = new_matrix(s.tb?s.n:s.k,s.tb?s.k:s.n,s.pad?5:0,0,&seed);
    /* Five extra guard rows catch an accidental copy of a complete six-row tile. */
    matrix c = new_matrix(s.m,s.n,s.pad?7:0,5,&seed);
    if (s.beta == 0) {
        for (int i = 0; i < s.m; i++) for (int j = 0; j < s.n; j++)
            c.data[(size_t)i*c.ld+j] = NAN;
        memcpy(c.before,c.allocation,c.length*sizeof(float));
    }
    cblas_sgemm(CblasRowMajor,s.ta?CblasTrans:CblasNoTrans,s.tb?CblasTrans:CblasNoTrans,
                s.m,s.n,s.k,s.alpha,a.data,a.ld,b.data,b.ld,s.beta,c.data,c.ld);
    check_storage(&s,&a,0,"A modified");
    check_storage(&s,&b,0,"B modified");
    check_storage(&s,&c,1,"C canary modified");
    for (int i = 0; i < s.m; i++) for (int j = 0; j < s.n; j++) {
        size_t at = (size_t)i*c.ld+j;
        /* beta==0 must not read old C. The shim rounds alpha*A before its GEMM. */
        float want = s.beta == 0 ? 0.0f : s.beta*c.before[GUARD+at];
        double precise = s.beta == 0 ? 0.0 : (double)s.beta*c.before[GUARD+at];
        double magnitude = fabs(precise);
        for (int p = 0; p < s.k; p++) {
            float av = a.data[s.ta?(size_t)p*a.ld+i:(size_t)i*a.ld+p];
            float bv = b.data[s.tb?(size_t)j*b.ld+p:(size_t)p*b.ld+j];
            float scaled = s.alpha*av;
            want = fmaf(scaled,bv,want);
            double term = (double)s.alpha*av*bv;
            precise += term;
            magnitude += fabs(term);
        }
        float got = c.data[at], error = fabsf(got-want);
        if (!isfinite(got) || !isfinite(want) || error > 8e-6f*(1+fabsf(want)))
            fail(&s,"numerical result",at,got,want);
        double double_error = fabs((double)got-precise);
        /* Cover rounded input scaling, beta scaling, and K float accumulations. */
        if (double_error > (s.k+4)*FLT_EPSILON*(1+magnitude))
            fail(&s,"double reference bound",at,got,(float)precise);
        if (memcmp(&got,&want,sizeof(float))) {
            different_bits++;
            if (exact) fail(&s,"ordered FMA bits",at,got,want);
        }
        max_error = fmaxf(max_error,error);
        max_double_error = fmax(max_double_error,double_error);
        cells++;
    }
    free_matrix(&a); free_matrix(&b); free_matrix(&c);
    cases++;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1],"--exact")) exact=1;
    else if (argc != 1) { fprintf(stderr,"usage: %s [--exact]\n",argv[0]); return 2; }
    const int ms[] = {1,2,3,4,5,7,11,16,17};
    const int ns[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,33,65,129};
    const int ks[] = {1,7,128,129,257};
    const float scales[][2] = {{1,1},{.5f,-.25f},{-.7f,.3f},{0,-.5f},{1,0}};
    for (size_t mi = 0; mi < sizeof(ms)/sizeof(*ms); mi++)
        for (size_t ni = 0; ni < sizeof(ns)/sizeof(*ns); ni++)
            for (size_t ki = 0; ki < sizeof(ks)/sizeof(*ks); ki++)
                for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) {
                    shape s = {ms[mi],ns[ni],ks[ki],ta,tb,0,1,0};
                    run_case(s);
                    int variant = (int)(mi+ni+ki+2*ta+tb)%5;
                    s.pad=1; s.alpha=scales[variant][0]; s.beta=scales[variant][1];
                    run_case(s);
                }
    /* Cross both packing-panel boundaries and the multithread dispatch threshold. */
    const int panels[][3] = {{16,257,257},{17,259,129},{97,17,257},{101,259,129}};
    for (size_t q = 0; q < sizeof(panels)/sizeof(*panels); q++)
        for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++)
            for (int v = 0; v < 5; v++)
                run_case((shape){panels[q][0],panels[q][1],panels[q][2],ta,tb,1,
                                 scales[v][0],scales[v][1]});
    printf("SIMD tails: %zu cases, %zu cells, threads=%s, max_error=%g, double_error=%g, differing_bits=%zu%s; "
           "input/output canaries intact\n",cases,cells,
           getenv("NT_SIMD_THREADS")?getenv("NT_SIMD_THREADS"):"default",
           max_error,max_double_error,different_bits,exact?" (exact required)":"");
    return 0;
}
