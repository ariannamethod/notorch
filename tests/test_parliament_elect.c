/* test_parliament_elect.c — isolated gate for the GPU parliament election.
 *  A) parliament_votes: vdot[e]=dot(w_vote_e,x) vs CPU dot, bar 1e-3.
 *  B) parliament_elect fed the CPU vdot (isolates the election logic from the
 *     dot reduction-order noise): EMA consensus, variable k, hard top-k, softmax
 *     must match parliament_elect() (doe.c) — same selected SET (exact), same k,
 *     consensus and gate weights within 1e-5.
 *
 * build (neo): cc -DUSE_METAL -O2 tests/test_parliament_elect.c notorch_metal.o \
 *   -framework Metal -framework Foundation -lc++ -lm -o test_parliament_elect
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../notorch_metal.h"

#define NE   8
#define D    256
#define MINE 2

static float wval(int e, int j) { return 0.02f * sinf(0.011f*(float)(e*9173 + j*53)); }

int main(void) {
    if (!nt_metal_available()) { fprintf(stderr, "metal unavailable\n"); return 2; }

    size_t pg = (size_t)getpagesize();
    size_t wbytes = (size_t)NE * D * sizeof(float);
    size_t bytes  = (wbytes + pg - 1) / pg * pg;
    float *arena = NULL;
    if (posix_memalign((void**)&arena, pg, bytes) != 0) { perror("posix_memalign"); return 2; }
    memset(arena, 0, bytes);
    for (int e = 0; e < NE; e++) for (int j = 0; j < D; j++) arena[(size_t)e*D + j] = wval(e, j);

    float x[D];      for (int j = 0; j < D; j++) x[j] = 0.1f * cosf(0.07f*(float)j);
    float alive[NE] = {1,1,1,0,1,1,0,1};                 /* 6 alive, experts 3 & 6 dead */
    float res[NE];   for (int e = 0; e < NE; e++) res[e] = 0.1f * (0.3f*sinf(0.9f*(float)e));
    float cons0 = 0.5f;

    /* ---- CPU reference: dot, then parliament_elect() math verbatim ---- */
    float vdot_cpu[NE];
    for (int e = 0; e < NE; e++) { float s=0; for (int j=0;j<D;j++) s += arena[(size_t)e*D+j]*x[j]; vdot_cpu[e]=s; }
    float votes[NE]; int n_alive = 0;
    for (int e = 0; e < NE; e++) { if (alive[e]!=0.0f){ votes[e]=vdot_cpu[e]+res[e]; n_alive++; } else votes[e]=-1e30f; }
    float mean=0; for (int e=0;e<NE;e++) if (alive[e]!=0.0f) mean+=votes[e]; mean/=n_alive;
    float var=0;  for (int e=0;e<NE;e++) if (alive[e]!=0.0f){ float d=votes[e]-mean; var+=d*d; } var/=n_alive;
    float cnew=sqrtf(var+1e-8f)/(fabsf(mean)+1.0f); if (cnew>1.0f) cnew=1.0f;
    float c_cpu=0.9f*cons0+0.1f*cnew;
    int k_cpu=(int)(n_alive*(1.0f-c_cpu)); if (k_cpu<MINE) k_cpu=MINE; if (k_cpu>n_alive) k_cpu=n_alive;
    int used[NE]={0}; int sel[NE]; float selv[NE];
    for (int ki=0;ki<k_cpu;ki++){ float bv=-1e30f;int bi=0; for(int e=0;e<NE;e++) if(alive[e]!=0.0f&&!used[e]&&votes[e]>bv){bv=votes[e];bi=e;} sel[ki]=bi;selv[ki]=votes[bi];used[bi]=1; }
    float mx=selv[0]; for(int ki=1;ki<k_cpu;ki++) if(selv[ki]>mx)mx=selv[ki];
    float sum=0; for(int ki=0;ki<k_cpu;ki++){ selv[ki]=expf(selv[ki]-mx); sum+=selv[ki]; }
    float gate_cpu[NE]={0}; for(int ki=0;ki<k_cpu;ki++) gate_cpu[sel[ki]]=selv[ki]/sum;

    /* ---- GPU ---- */
    if (nt_metal_register_region(arena, bytes) != 0) { fprintf(stderr, "register failed\n"); return 2; }
    enum { S_X=0, S_VDOT=1, S_RES=2, S_ALIVE=3, S_CONS=4, S_GATE=5 };
    if (nt_metal_slot_alloc(S_X,(uint64_t)D*4)||nt_metal_slot_alloc(S_VDOT,(uint64_t)NE*4)||
        nt_metal_slot_alloc(S_RES,(uint64_t)NE*4)||nt_metal_slot_alloc(S_ALIVE,(uint64_t)NE*4)||
        nt_metal_slot_alloc(S_CONS,(uint64_t)4*4)||nt_metal_slot_alloc(S_GATE,(uint64_t)NE*4)) { fprintf(stderr,"alloc failed\n"); return 2; }
    nt_metal_slot_upload(S_X, x, (uint64_t)D*4);
    nt_metal_slot_upload(S_RES, res, (uint64_t)NE*4);
    nt_metal_slot_upload(S_ALIVE, alive, (uint64_t)NE*4);

    /* A) votes kernel vs CPU dot */
    int rc = nt_metal_parliament_votes(arena, S_X, S_VDOT, D, NE);
    if (rc) { fprintf(stderr,"votes rc=%d\n", rc); return 2; }
    float vdot_gpu[NE]; nt_metal_slot_download(S_VDOT, vdot_gpu, (uint64_t)NE*4);
    double vmax=0; for(int e=0;e<NE;e++){ double d=fabs((double)vdot_gpu[e]-(double)vdot_cpu[e]); if(d>vmax)vmax=d; }
    printf("[votes] max|vdot_gpu - vdot_cpu| = %.3e\n", vmax);

    /* B) elect fed the CPU vdot — isolate election logic */
    float cons4[4]={cons0,0,0,0};
    nt_metal_slot_upload(S_VDOT, vdot_cpu, (uint64_t)NE*4);
    nt_metal_slot_upload(S_CONS, cons4, (uint64_t)4*4);
    rc = nt_metal_parliament_elect(S_VDOT, S_RES, S_ALIVE, S_CONS, S_GATE, NE, 0, MINE);
    if (rc) { fprintf(stderr,"elect rc=%d\n", rc); return 2; }
    float gate_gpu[NE]; nt_metal_slot_download(S_GATE, gate_gpu, (uint64_t)NE*4);
    float cons_gpu[4]; nt_metal_slot_download(S_CONS, cons_gpu, (uint64_t)4*4);

    int k_gpu=0; for(int e=0;e<NE;e++) if(gate_gpu[e]!=0.0f) k_gpu++;
    int set_match=1; for(int e=0;e<NE;e++) if((gate_cpu[e]!=0.0f)!=(gate_gpu[e]!=0.0f)) set_match=0;
    double gmax=0; for(int e=0;e<NE;e++){ double d=fabs((double)gate_gpu[e]-(double)gate_cpu[e]); if(d>gmax)gmax=d; }
    double cdiff=fabs((double)cons_gpu[0]-(double)c_cpu);
    printf("[elect] k_cpu=%d k_gpu=%d  set_match=%d  max|gate|diff=%.3e  cons diff=%.3e (c_cpu=%.6f)\n",
           k_cpu, k_gpu, set_match, gmax, cdiff, c_cpu);

    int pass = (vmax<1e-3) && (k_gpu==k_cpu) && set_match && (gmax<1e-5) && (cdiff<1e-5);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
