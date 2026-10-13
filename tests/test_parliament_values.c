/* Borrowed-buffer DoE arithmetic: numerical witnesses, failure atomicity,
 * private rank cursors and parallel owners under an unchanged live tape.
 * cc -O2 -I. tests/test_parliament_values.c notorch.c -lm -pthread -o test_parliament_values
 */
#include "notorch.h"
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { \
    fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static void near_value(double actual, double expected, double tolerance) {
    CHECK(isfinite(actual) && fabs(actual - expected) <= tolerance);
}
static void fill(float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = 314.25f + (float)i;
}

static void election(void) {
    float votes[4] = {-3,-1,1,3}, res[4] = {0}, alive[4] = {1,1,1,1};
    float out[6], before[6];
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,.5f,out)==0);
    near_value(out[4],.55,1e-7); CHECK(out[5]==2 && out[0]==0 && out[1]==0);
    near_value(out[3],1/(1+exp(-2)),1e-7); near_value(out[2],1-out[3],1e-7);
    res[0] = 10;
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,.5f,out)==0);
    CHECK(out[0]>out[3] && out[1]==0 && out[2]==0);
    alive[0]=0; res[0]=0;
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,.5f,out)==0);
    CHECK(out[0]==0 && out[3]>out[2] && out[5]==2);
    alive[1]=alive[2]=0;
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,.37f,out)==0);
    CHECK(out[0]==0 && out[1]==0 && out[2]==0 && out[3]==0 && out[4]==.37f && out[5]==0);
    for (int e=0;e<4;e++) { alive[e]=1; votes[e]=0; }
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,0,out)==0);
    CHECK(out[5]==3 && out[3]==0);
    for(int e=0;e<3;e++) near_value(out[e],1.0/3,2e-8);
    // Very negative finite ties still select live slots; no sentinel wins.
    for(int e=0;e<4;e++) votes[e]=-FLT_MAX;
    alive[0]=alive[1]=alive[2]=0;
    CHECK(nt_parliament_elect_values(votes,res,alive,4,1,0,out)==0);
    CHECK(out[3]==1 && out[5]==1);
    fill(out,6); memcpy(before,out,sizeof out);
    CHECK(nt_parliament_elect_values(votes,res,alive,17,2,0,out)==-1);
    CHECK(!memcmp(out,before,sizeof out));
    CHECK(nt_parliament_elect_values(votes,res,alive,4,0,0,out)==-1);
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,NAN,out)==-1);
    CHECK(nt_parliament_elect_values(NULL,res,alive,4,2,0,out)==-1);
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,0,NULL)==-1);
    alive[0]=NAN;
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,0,out)==-1);
    for(int e=0;e<4;e++) { alive[e]=1; votes[e]=FLT_MAX; }
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,.5f,out)==-1);
    CHECK(!memcmp(out,before,sizeof out)); // late mean overflow never publishes
    CHECK(nt_parliament_elect_values(votes,res,alive,4,2,0,votes)==-1);
    CHECK(votes[0]==FLT_MAX);
    puts("PASS election: resonance, variable count, dead slots, ties and atomic refusal");
}

static void injection(void) {
    enum { E=3,D=5,R=2,N=E*D*R };
    float a[N],b[N],saved_a[N],saved_b[N],gate[E]={.25f,0,.75f},x[D],out[D];
    for(int i=0;i<N;i++) { a[i]=(float)((i*7)%13-6)*.1f; b[i]=(float)((i*11)%17-8)*.07f; }
    for(int i=0;i<D;i++) x[i]=(float)(i-2)*.3f;
    memcpy(saved_a,a,sizeof a); memcpy(saved_b,b,sizeof b);
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,.35f,out)==0);
    for(int i=0;i<D;i++) {
        double expected=x[i];
        for(int e=0;e<E;e++) for(int r=0;r<R;r++) {
            double bx=0;
            for(int j=0;j<D;j++) bx+=(double)b[(e*R+r)*D+j]*x[j];
            expected+=(double).35f*gate[e]*a[(e*D+i)*R+r]*bx;
        }
        near_value(out[i],expected,2e-7);
    }
    CHECK(!memcmp(a,saved_a,sizeof a) && !memcmp(b,saved_b,sizeof b));
    // Inactive experts may contain numerical poison without contaminating x.
    a[D*R+3]=NAN; b[D*R+5]=INFINITY;
    float repeat[D];
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,.35f,repeat)==0);
    CHECK(!memcmp(out,repeat,sizeof out));
    x[0]=-0.0f;
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,0,out)==0);
    CHECK(!memcmp(out,x,sizeof out));
    float zero[E]={0};
    CHECK(nt_parliament_inject_values(a,b,zero,x,E,D,R,1,out)==0);
    CHECK(!memcmp(out,x,sizeof out));
    float snapshot[D]; fill(out,D); memcpy(snapshot,out,sizeof out);
    gate[1]=.5f;
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,.35f,out)==-1);
    CHECK(!memcmp(out,snapshot,sizeof out));
    gate[1]=0;
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,INT_MAX,R,1,out)==-1);
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,17,1,out)==-1);
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,NAN,out)==-1);
    CHECK(nt_parliament_inject_values(NULL,b,gate,x,E,D,R,1,out)==-1);
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,1,a+1)==-1);
    CHECK(nt_parliament_inject_values(a,b,gate,x,E,D,R,1,x)==-1);
    CHECK(!memcmp(out,snapshot,sizeof out));
    float late_a[3]={1,1,FLT_MAX},late_b[3]={1,1,1},ones[3]={1,1,1},g[1]={1};
    CHECK(nt_parliament_inject_values(late_a,late_b,g,ones,1,3,1,1,out)==-1);
    CHECK(!memcmp(out,snapshot,sizeof out)); // output row2 overflows after valid rows0/1
    puts("PASS low-rank mixture: independent dense oracle, inactive/zero-alpha identity, atomic overflow");
}

static void oja(void) {
    enum { D=3,R=4,N=D*R };
    float a[N],b[N],x[D]={.2f,-.4f,.7f},dy[D]={-.1f,.5f,.3f},noise[2]={.6f,-.2f};
    float out[2*N+1],clamped[2*N+1],saved_a[N],saved_b[N];
    for(int i=0;i<N;i++) { a[i]=(float)(i-5)*.03f; b[i]=(float)(7-i)*.02f; }
    memcpy(saved_a,a,sizeof a); memcpy(saved_b,b,sizeof b);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,3,2,.03f,20,out)==0);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,3,2,.03f,2,clamped)==0);
    CHECK(!memcmp(out,clamped,sizeof out));
    for(int r=0;r<R;r++) for(int i=0;i<D;i++) {
        int ai=i*R+r,bi=r*D+i;
        if(r==1 || r==2) { CHECK(out[ai]==a[ai] && out[N+bi]==b[bi]); continue; }
        int j=(r==3?0:1);
        double u=(double).01f*noise[j];
        for(int k=0;k<D;k++) u+=(double)b[r*D+k]*dy[k];
        double eta=(double).03f*2;
        near_value(out[ai],a[ai]+eta*u*(x[i]-u*a[ai]),2e-8);
        near_value(out[N+bi],b[bi]+eta*u*(dy[i]-u*b[bi]),2e-8);
    }
    CHECK(out[2*N]==1 && !memcmp(a,saved_a,sizeof a) && !memcmp(b,saved_b,sizeof b));
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,3,2,.03f,0,out)==0);
    CHECK(!memcmp(out,a,sizeof a) && !memcmp(out+N,b,sizeof b) && out[2*N]==3);
    fill(out,2*N+1); float saved[2*N+1]; memcpy(saved,out,sizeof out);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,4,2,.03f,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,0,0,.03f,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,0,R+1,.03f,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,INT_MAX,R,0,2,.03f,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,0,2,-1,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,0,2,1,NAN,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,NULL,D,R,0,2,1,1,out)==-1);
    CHECK(nt_oja_values(a,b,x,dy,noise,D,R,0,2,1,1,a)==-1);
    CHECK(!memcmp(out,saved,sizeof out) && !memcmp(a,saved_a,sizeof a));
    float late_a[3]={.1f,.1f,FLT_MAX},late_b[3]={1,1,1},ones[3]={1,1,1},nn[1]={0};
    CHECK(nt_oja_values(late_a,late_b,ones,ones,nn,3,1,0,1,1,1,out)==-1);
    CHECK(!memcmp(out,saved,sizeof out));
    puts("PASS Oja: independent equation, signal cap, wrapping window, frozen inputs and atomic refusal");
}

static void private_cursors(void) {
    enum{E=4,D=2,R=16,N=D*R};
    float a[E][N],b[E][N],x[D]={1,1},noise[4]={0}; int cursor[E]={0};
    for(int e=0;e<E;e++) for(int i=0;i<N;i++) a[e][i]=b[e][i]=.02f;
    for(int turn=0;turn<4;turn++) for(int e=0;e<E;e++) {
        float next[2*N+1];
        CHECK(nt_oja_values(a[e],b[e],x,x,noise,D,R,cursor[e],4,.005f,.5f,next)==0);
        memcpy(a[e],next,sizeof a[e]); memcpy(b[e],next+N,sizeof b[e]); cursor[e]=(int)next[2*N];
    }
    for(int e=0;e<E;e++) {
        CHECK(cursor[e]==0);
        for(int r=0;r<R;r++) CHECK(a[e][r]!=.02f && b[e][r*D]!=.02f);
        CHECK(!memcmp(a[0],a[e],sizeof a[e]) && !memcmp(b[0],b[e],sizeof b[e]));
    }
    puts("PASS four private cursors visit every rank component independently");
}

typedef struct { float state[4*4*2*2]; float election[6]; int status; } work;
static void *worker(void *opaque) {
    work *w=opaque;
    enum{E=4,D=4,R=2,N=E*D*R};
    float *a=w->state,*b=w->state+N;
    float x[D]={.2f,.3f,-.1f,.7f},votes[E]={.1f,.2f,.3f,.4f},res[E]={0},alive[E]={1,1,1,1};
    float noise[R]={.02f,-.01f};
    for(int i=0;i<2*N;i++) w->state[i]=.01f*(float)(i%7+1);
    float cons=.5f; w->status=0;
    for(int turn=0;turn<64;turn++) {
        float proposed[D];
        if(nt_parliament_elect_values(votes,res,alive,E,2,cons,w->election) ||
           nt_parliament_inject_values(a,b,w->election,x,E,D,R,.2f,proposed)) { w->status=-1;return NULL; }
        cons=w->election[E];
        for(int e=0;e<E;e++) {
            float next[2*D*R+1];
            if(nt_oja_values(a+e*D*R,b+e*D*R,x,proposed,noise,D,R,0,R,.001f,.1f,next)) { w->status=-1;return NULL; }
            memcpy(a+e*D*R,next,D*R*sizeof(float)); memcpy(b+e*D*R,next+D*R,D*R*sizeof(float));
        }
    }
    return NULL;
}
static void owners_and_tape(void) {
    nt_tape_start(); nt_tensor *p=nt_tensor_new(1); CHECK(p!=NULL); p->data[0]=.4f;
    int leaf=nt_tape_param(p),y=nt_tanh(leaf); CHECK(leaf>=0 && y>=0); nt_tape_backward(y);
    nt_tape *before=malloc(sizeof(*before)); CHECK(before!=NULL);
    memcpy(before,nt_tape_get(),sizeof(*before));
    float leafgrad=before->entries[leaf].grad->data[0];
    work expected,actual[8]; worker(&expected); pthread_t threads[8];
    for(int i=0;i<8;i++) CHECK(pthread_create(&threads[i],NULL,worker,&actual[i])==0);
    for(int i=0;i<8;i++) {
        CHECK(pthread_join(threads[i],NULL)==0);
        CHECK(expected.status==0 && actual[i].status==0);
        CHECK(!memcmp(expected.state,actual[i].state,sizeof expected.state));
        CHECK(!memcmp(expected.election,actual[i].election,sizeof expected.election));
    }
    CHECK(!memcmp(before,nt_tape_get(),sizeof(*before)) && p->data[0]==.4f);
    CHECK(before->entries[leaf].grad->data[0]==leafgrad);
    free(before); nt_tape_destroy(); nt_tensor_free(p);
    puts("PASS eight concurrent owners equal serial continuation; active tape untouched");
}

int main(void) {
    election(); injection(); oja(); private_cursors(); owners_and_tape();
    printf("PARLIAMENT_VALUES_OK %u checks\n",checks);
    return 0;
}
