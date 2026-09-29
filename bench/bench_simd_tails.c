/* Frozen Qwen 0.5B vocabulary projection and its input gradient.
 * make bench/bench_simd_tails
 * NT_SIMD_THREADS=4 ./bench/bench_simd_tails 16
 * One warm-up and two measured iterations; optional argument: token batch 1..48.
 * Reports native tape projection time and process peak RSS; backward uses unit gradients.
 */
#include "notorch.h"
#include <assert.h>
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1000.0+(double)t.tv_nsec/1000000.0;
}
int main(int argc,char **argv) {
    const int E=896,V=151936;
    int T=argc>1?atoi(argv[1]):16;
    if(T<1||T>48)return 2;
    nt_tensor *head=nt_tensor_new2d(V,E),*input=nt_tensor_new2d(T,E);
    assert(head&&input);
    for(int i=0;i<head->len;i++)head->data[i]=.0001f*(float)((i*13u)%127u-63.0f);
    for(int i=0;i<input->len;i++)input->data[i]=.01f*(float)((i*7u)%31u-15.0f);
    double fwd[3],bwd[3],check_forward=0,check_backward=0;
    for(int trial=0;trial<3;trial++) {
        nt_tape_start();int w=nt_tape_param_frozen(head),x=nt_tape_param(input);
        double start=milliseconds();int y=nt_seq_linear(w,x,T);fwd[trial]=milliseconds()-start;
        assert(y>=0);nt_tape *t=nt_tape_get();
        check_forward=t->entries[y].output->data[0]+t->entries[y].output->data[T*V-1];
        start=milliseconds();nt_tape_backward(y);bwd[trial]=milliseconds()-start;
        assert(!t->entries[w].grad&&t->entries[x].grad);
        check_backward=t->entries[x].grad->data[0]+t->entries[x].grad->data[T*E-1];
        assert(isfinite(check_forward)&&isfinite(check_backward));
        nt_tape_clear();
    }
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    long peak_rss_kib=usage.ru_maxrss;
#ifdef __APPLE__
    peak_rss_kib/=1024;
#endif
    printf("{\"threads\":%d,\"T\":%d,\"E\":%d,\"V\":%d,\"fwd_ms\":[%.3f,%.3f],\"bwd_ms\":[%.3f,%.3f],\"mean_ms_per_token\":%.4f,\"peak_rss_kib\":%ld,\"forward_check\":%.9g,\"backward_check\":%.9g}\n",
        getenv("NT_SIMD_THREADS")?atoi(getenv("NT_SIMD_THREADS")):0,T,E,V,
        fwd[1],fwd[2],bwd[1],bwd[2],(fwd[1]+fwd[2]+bwd[1]+bwd[2])/(2*T),peak_rss_kib,check_forward,check_backward);
    nt_tape_start();nt_tape_param(input);nt_tape_destroy();nt_tensor_free(input);nt_tensor_free(head);
    return 0;
}
