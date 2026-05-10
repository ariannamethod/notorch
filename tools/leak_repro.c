#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "notorch.h"

extern int gpu_init(void);
extern void gpu_shutdown(void);

static size_t gpu_used_mb(void) {
    FILE* f = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", "r");
    if (!f) return 0;
    char buf[64]; if (!fgets(buf, sizeof(buf), f)) { pclose(f); return 0; }
    pclose(f);
    return (size_t)atol(buf);
}

int main(void) {
    if (gpu_init() != 0) { fprintf(stderr, "gpu_init failed\n"); return 1; }
    nt_set_gpu_mode(1);
    nt_tape_start();

    int D=64, T=16, V=128, RANK=4;
    nt_tensor* W1 = nt_tensor_new(D*D);
    for (int i=0;i<D*D;i++) W1->data[i]=((float)rand()/RAND_MAX - 0.5f)*0.02f;
    nt_tensor* W2 = nt_tensor_new(V*D);
    for (int i=0;i<V*D;i++) W2->data[i]=((float)rand()/RAND_MAX - 0.5f)*0.02f;
    nt_tensor* X  = nt_tensor_new(T*D);
    for (int i=0;i<T*D;i++) X->data[i]=((float)rand()/RAND_MAX - 0.5f);

    /* Targets tensor: T int targets (use 7 for all positions, mask at last) */
    nt_tensor* targets = nt_tensor_new(T);
    for (int i=0;i<T;i++) targets->data[i] = 7.0f;
    nt_tensor* mask = nt_tensor_new(T);
    for (int i=0;i<T-1;i++) mask->data[i] = 0.0f;
    mask->data[T-1] = 1.0f;

    nt_lora_pair lora1; nt_lora_init(&lora1, D, D, RANK, 8.0f);
    nt_lora_pair lora2; nt_lora_init(&lora2, D, V, RANK, 8.0f);

    printf("step,gpu_mb,delta_mb\n");
    size_t prev = gpu_used_mb();
    printf("0,%zu,0\n", prev);

    for (int step=1; step<=200; step++) {
        nt_tape_start();
        int x_idx = nt_tape_param(X);
        int w1_idx = nt_tape_param_frozen(W1);
        int w2_idx = nt_tape_param_frozen(W2);
        int t_idx = nt_tape_param(targets);
        int m_idx = nt_tape_param(mask);

        int h_idx = nt_lora_forward(w1_idx, &lora1, x_idx, T);
        if (h_idx < 0) { fprintf(stderr,"step %d: lora1 fwd failed\n", step); break; }
        int log_idx = nt_lora_forward(w2_idx, &lora2, h_idx, T);
        if (log_idx < 0) { fprintf(stderr,"step %d: lora2 fwd failed\n", step); break; }
        int loss_idx = nt_seq_cross_entropy_masked(log_idx, t_idx, m_idx, T, V);
        if (loss_idx < 0) { fprintf(stderr,"step %d: ce failed\n", step); break; }
        nt_tape_backward(loss_idx);

        size_t now = gpu_used_mb();
        printf("%d,%zu,%ld\n", step, now, (long)now - (long)prev);
        prev = now;
    }
    return 0;
}
