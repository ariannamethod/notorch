/* Chuck CPU/device boundaries. The same source runs with scalar CPU, CUDA,
 * or a separately labelled HOST_EMULATION of the mirror protocol.
 * Synthetic trajectories isolate optimizer coherence from GEMM rounding;
 * the CUDA build also executes a real linear/CE forward/backward training body.
 */
#include "chuck_architect.h"
#ifdef USE_CUDA
#include "notorch_cuda.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { N = 257, STEPS = 40 };
static int checks, failures;
static const char *scenario;
static int at_step;
#define REQUIRE(c, msg) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s step=%d line=%d: %s\n", scenario, at_step, __LINE__, msg); \
    failures++; return 0; } } while (0)

typedef struct {
    float p[N], g[N], m[N], v[N];
    nt_chuck_state cs;
    nt_chuck_param_state cp;
    int t;
    uint32_t rng;
    uint64_t policy_hash;
} state_image;
static nt_tensor *body;
static int body_index;
static float max_abs_error;
#ifdef CHUCK_HOST_EMULATION
extern long long chuck_host_download_count(void);
#endif

static int near(float a, float b, float tolerance) {
    float delta = fabsf(a - b);
    if (delta > max_abs_error) max_abs_error = delta;
    return isfinite(a) && isfinite(b) && delta <= tolerance * (1 + fabsf(b));
}

static void close_body(void) {
    nt_tape_destroy();
    nt_tensor_free(body);
    body = NULL;
}

static int open_body(void) {
    close_body();
    nt_seed(29);
    nt_chuck_rng_set(UINT32_C(1234567));
    nt_tape_start();
    body = nt_tensor_new(N);
    REQUIRE(body != NULL, "body allocation");
    body_index = nt_tape_param(body);
    REQUIRE(body_index >= 0, "parameter registration");
    nt_tape *t = nt_tape_get();
    t->entries[body_index].grad = nt_tensor_new(N);
    REQUIRE(t->entries[body_index].grad != NULL, "gradient allocation");
    for (int j = 0; j < N; ++j) {
        body->data[j] = (float)(j % 19 - 9) / 32;
        t->adam[0].m->data[j] = (float)(j % 5 - 2) / 64;
        t->adam[0].v->data[j] = .02f + (float)(j % 3) / 128;
    }
    t->adam[0].t = 3;
    return 1;
}

/* Read back to a SEPARATE buffer. Syncing the tensor itself between steps
 * would hide the very stale-mirror transition this gate must catch. */
static void read_tensor(float *dst, nt_tensor *src) {
#ifdef USE_CUDA
    if (src->cpu_dirty && src->d_data) {
        gpu_download(dst, src->d_data, src->len);
        return;
    }
#endif
    memcpy(dst, src->data, (size_t)src->len * sizeof(float));
}

static void capture(state_image *s) {
    nt_tape *t = nt_tape_get();
    memset(s, 0, sizeof(*s));
    read_tensor(s->p, body);
    read_tensor(s->g, t->entries[body_index].grad);
    read_tensor(s->m, t->adam[0].m);
    read_tensor(s->v, t->adam[0].v);
    s->cs = t->chuck;
    s->cp = t->chuck_params[0];
    s->t = t->adam[0].t;
    s->rng = nt_chuck_rng_get();
}

static int restore(const state_image *s) {
    if (!open_body()) return 0;
    nt_tape *t = nt_tape_get();
    memcpy(body->data, s->p, sizeof(s->p));
    memcpy(t->entries[body_index].grad->data, s->g, sizeof(s->g));
    memcpy(t->adam[0].m->data, s->m, sizeof(s->m));
    memcpy(t->adam[0].v->data, s->v, sizeof(s->v));
    t->chuck = s->cs;
    t->chuck_params[0] = s->cp;
    t->adam[0].t = s->t;
    REQUIRE(nt_chuck_rng_set(s->rng) == 0, "restore noise RNG");
    return 1;
}

static int same_state(const state_image *a, const state_image *b) {
    const float *arrays_a[] = {a->p, a->g, a->m, a->v};
    const float *arrays_b[] = {b->p, b->g, b->m, b->v};
    const char *names[] = {"parameter", "gradient", "first moment", "second moment"};
    for (int k = 0; k < 4; ++k) for (int j = 0; j < N; ++j) {
        if (!near(arrays_a[k][j], arrays_b[k][j], 3e-6f)) {
            fprintf(stderr, "%s[%d]: %.9g != %.9g\n", names[k], j, arrays_a[k][j], arrays_b[k][j]);
            REQUIRE(0, "CPU/device tensor parity");
        }
    }
    REQUIRE(a->t == b->t && a->rng == b->rng, "Adam timestep and noise RNG");
    REQUIRE(!memcmp(&a->cs, &b->cs, sizeof a->cs), "global state exact parity");
    REQUIRE(a->cp.pos == b->cp.pos && a->cp.full == b->cp.full &&
            a->cp.stag == b->cp.stag && a->cp.frozen == b->cp.frozen,
            "per-parameter counters parity");
    REQUIRE(near(a->cp.dampen, b->cp.dampen, 3e-6f), "per-parameter dampening");
    for (int i = 0; i < NT_CHUCK_WINDOW; ++i)
        REQUIRE(near(a->cp.grad_hist[i], b->cp.grad_hist[i], 3e-6f), "gradient history parity");
    return 1;
}

static int write_tensor(nt_tensor *t, const float *values, int device) {
#ifdef USE_CUDA
    if (device) {
        if (!t->d_data) t->d_data = gpu_alloc(t->len);
        REQUIRE(t->d_data != NULL, "device allocation");
        gpu_upload(t->d_data, values, t->len);
        t->gpu_valid = 1;
        t->cpu_dirty = 1;
        /* Poison a valid but stale mirror; every required download is visible. */
        for (int j = 0; j < t->len; ++j) t->data[j] = 7.0f + j * .03125f;
        return 1;
    }
    t->gpu_valid = 0;
    t->cpu_dirty = 0;
#else
    (void)device;
#endif
    memcpy(t->data, values, (size_t)t->len * sizeof(float));
    return 1;
}

static int prime_device(void) {
    nt_tape *t = nt_tape_get();
    float tmp[N];
    nt_tensor *arrays[] = {body, t->adam[0].m, t->adam[0].v};
    for (int k = 0; k < 3; ++k) {
        read_tensor(tmp, arrays[k]);
        if (!write_tensor(arrays[k], tmp, 1)) return 0;
    }
    return 1;
}

static float gradient_at(int step, int j) {
    return (float)(((j * 7 + step * 3) % 17) - 8) * .0078125f + .001f * step;
}

/* Four paths: all typed controls; ordinary canonical noise; mode switches with
 * the canonical API; acquired policy. The first custom noise step follows a
 * device step, and the next zero-noise step must upload changed moments. */
static int trajectory(int kind, int device, int resume, state_image out[STEPS]) {
    if (!open_body()) return 0;
    if (device && !prime_device()) return 0;
    nt_chuck_architect a;
    nt_chuck_architect_config config;
    nt_chuck_architect_config_default(&config);
    config.mode = NT_CHUCK_ARCHITECT_LEARNED;
    config.seed = 73;
    REQUIRE(nt_chuck_architect_init(&a, &config) == 0, "life initialization");
    for (int step = 0; step < STEPS; ++step) {
        at_step = step;
        int active_device = device && (kind != 2 || step % 3 != 1);
        nt_set_gpu_mode(active_device);
        nt_tape *t = nt_tape_get();
        float grad[N];
        for (int j = 0; j < N; ++j) grad[j] = gradient_at(step, j);
        if (!write_tensor(t->entries[body_index].grad, grad, device)) return 0;
        float loss = kind == 1 ? 1.0f : 1.0f + (float)(step % 7) / 64;
        if (kind == 0 || kind == 3) {
            nt_chuck_observation obs;
            REQUIRE(nt_tape_chuck_observe(loss, &obs) == 0, "observe device gradient");
            double sum = 0;
            for (int j = 0; j < N; ++j) sum += (double)grad[j] * grad[j];
            REQUIRE(near(obs.grad_norm, (float)sqrt(sum), 1e-6f), "observed norm uses current device values");
        }
        if (kind == 0) {
            nt_chuck_action actions[] = {{NT_CHUCK_ACTION_HOLD, 0}, {NT_CHUCK_ACTION_BRAKE, 0},
                {NT_CHUCK_ACTION_PUSH, 0}, {NT_CHUCK_ACTION_SET_NOISE, .002f},
                {NT_CHUCK_ACTION_HOLD, 0}, {NT_CHUCK_ACTION_SET_NOISE, 0},
                {NT_CHUCK_ACTION_SET_LR_SCALE, .6f}, {NT_CHUCK_ACTION_SET_DAMPEN, 1.1f}};
            REQUIRE(nt_tape_chuck_step_action(.003f, loss, &actions[step % 8], NULL) == 0,
                    "bounded typed action");
        } else if (kind == 3) {
            nt_chuck_architect_decision decision;
            nt_chuck_architect_receipt receipt;
            REQUIRE(nt_chuck_architect_step(&a, .003f, loss, &decision) == 0, "learned action executes");
            /* Synthetic measured environment for the device contract: this
             * receipt is deterministic test data, not a model-training result. */
            REQUIRE(nt_chuck_architect_feedback(&a, loss - .01f, &receipt) == 0, "feedback consumed");
        } else {
            nt_tape_chuck_step(.003f, loss);
        }
        capture(&out[step]);
        out[step].policy_hash = nt_chuck_architect_hash(&a);
        if (resume && step == 17) {
            char path[] = "/tmp/notorch-device-life-XXXXXX";
            int fd = mkstemp(path);
            REQUIRE(fd >= 0, "resume temporary path");
            close(fd);
            int saved = nt_chuck_architect_save(&a, path);
            if (!restore(&out[step])) { unlink(path); return 0; }
            memset(&a, 0, sizeof a);
            int loaded = nt_chuck_architect_load(&a, path);
            unlink(path);
            REQUIRE(saved == 0 && loaded == 0, "saved life resume");
        }
    }
    if (kind == 1) REQUIRE(out[STEPS - 1].cs.noise > 0, "canonical stagnation reached CPU noise fallback");
    if (kind == 3) REQUIRE(a.updates == STEPS, "each device consequence learned");
    close_body();
    return 1;
}

/* A next CPU forward must consume the weights the preceding device step left
 * behind. Both weights and activations here carry deliberately stale mirrors. */
static int forward_mode_switch(int device) {
    close_body();
    nt_tape_start();
    nt_tensor *w = nt_tensor_new2d(4, 3), *x = nt_tensor_new2d(2, 3);
    REQUIRE(w && x, "mode-switch body allocation");
    float w_values[12], x_values[6], expected[8], actual[8];
    for (int i = 0; i < 12; ++i) w_values[i] = (float)(i % 5 - 2) / 8;
    for (int i = 0; i < 6; ++i) x_values[i] = (float)(i - 3) / 4;
    if (!write_tensor(w, w_values, device) || !write_tensor(x, x_values, device)) return 0;
    int wi = nt_tape_record(w, NT_OP_NONE, -1, -1, 0);
    int xi = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    nt_set_gpu_mode(device);
    int zi = nt_seq_linear(wi, xi, 2);
    REQUIRE(zi >= 0, "device forward before mode switch");
    read_tensor(expected, nt_tape_get()->entries[zi].output);
#ifdef CHUCK_HOST_EMULATION
    long long before = chuck_host_download_count();
#endif
    nt_set_gpu_mode(0);
    zi = nt_seq_linear(wi, xi, 2);
    REQUIRE(zi >= 0, "CPU forward after mode switch");
    read_tensor(actual, nt_tape_get()->entries[zi].output);
    for (int j = 0; j < 8; ++j) {
        if (!near(actual[j], expected[j], 1e-6f)) {
            fprintf(stderr, "mode-switch output[%d] actual=%.9g expected=%.9g\n", j, actual[j], expected[j]);
            REQUIRE(0, "CPU forward reads current device weights and inputs");
        }
    }
#ifdef CHUCK_HOST_EMULATION
    REQUIRE(chuck_host_download_count() - before == (device ? 2 : 0),
            "exactly two stale inputs downloaded at CPU boundary");
#endif
    zi = nt_seq_linear(wi, xi, 2);
    REQUIRE(zi >= 0, "repeated CPU forward");
#ifdef CHUCK_HOST_EMULATION
    REQUIRE(chuck_host_download_count() - before == (device ? 2 : 0),
            "already-current inputs avoid redundant downloads");
#endif
    nt_tape_destroy();
    nt_tensor_free(w); nt_tensor_free(x);
    return 1;
}

#if !defined(CHUCK_HOST_EMULATION)
typedef struct {
    float weights[512], losses[8], gradient_norms[8], gradients[8 * 512];
} linear_image;
/* Actual small training body: 16x32 weights; four 32-dimensional inputs.
 * A CUDA build must register cuBLAS dispatch and GPU-resident output/gradient.
 * CPU and CUDA compare observations and final parameters after mixed actions.
 */
static int linear_training(int device, linear_image *result) {
    enum { T = 4, D = 32, V = 16 };
    close_body();
    nt_set_gpu_mode(device);
    nt_tensor *w = nt_tensor_new2d(V, D), *x = nt_tensor_new2d(T, D), *y = nt_tensor_new(T);
    REQUIRE(w && x && y, "linear body allocation");
    for (int j = 0; j < V * D; ++j) w->data[j] = (float)(j % 11 - 5) / 64;
    for (int j = 0; j < T * D; ++j) x->data[j] = (float)(j % 7 - 3) / 16;
    for (int j = 0; j < T; ++j) y->data[j] = (float)(j * 3);
    nt_chuck_rng_set(1234567);
#ifdef USE_CUDA
    nt_gpu_dispatch_reset();
#endif
    for (int step = 0; step < 8; ++step) {
        at_step = step;
        nt_tape_start();
        int wi = nt_tape_param(w);
        int xi = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
        int yi = nt_tape_record(y, NT_OP_NONE, -1, -1, 0);
        int z = nt_seq_linear(wi, xi, T);
        REQUIRE(z >= 0, "linear forward");
#ifdef USE_CUDA
        if (device) REQUIRE(nt_tape_get()->entries[z].output->gpu_valid, "linear output device residency");
#endif
        int li = nt_seq_cross_entropy(z, yi, T, V);
        REQUIRE(li >= 0, "cross entropy forward");
        nt_tensor_sync_cpu(nt_tape_get()->entries[li].output);
        float loss = nt_tape_get()->entries[li].output->data[0];
        REQUIRE(isfinite(loss), "finite body loss");
        nt_tape_backward(li);
        nt_chuck_observation obs;
        REQUIRE(nt_tape_chuck_observe(loss, &obs) == 0 && obs.grad_norm > 0, "actual backward gradient observed");
        result->losses[step] = loss;
        result->gradient_norms[step] = obs.grad_norm;
        read_tensor(result->gradients + step * 512, nt_tape_get()->entries[wi].grad);
        nt_chuck_action action = {step % 2 ? NT_CHUCK_ACTION_PUSH : NT_CHUCK_ACTION_BRAKE, 0};
        REQUIRE(nt_tape_chuck_step_action(.001f, loss, &action, NULL) == 0, "body training action");
    }
    nt_tensor_sync_cpu(w);
    memcpy(result->weights, w->data, sizeof result->weights);
#ifdef USE_CUDA
    if (device) {
        REQUIRE(nt_gpu_dispatch_count() > 0, "actual training dispatched cuBLAS operations");
        printf("CHUCK_DEVICE_TRAINING dispatches=%lld parameters=512 steps=8 loss=%.9g\n",
               nt_gpu_dispatch_count(), result->losses[7]);
    }
#endif
    nt_tape_destroy();
    nt_tensor_free(w); nt_tensor_free(x); nt_tensor_free(y);
    return 1;
}
#endif

int main(void) {
    const char *backend = "CPU";
    int device = 0;
#ifdef USE_CUDA
    if (gpu_init() != 0) {
        puts("CHUCK_DEVICE_SKIPPED backend=CUDA reason=no_usable_CUDA_device");
        return 77;
    }
    device = 1;
#ifdef CHUCK_HOST_EMULATION
    backend = "HOST_EMULATION";
    puts("CHUCK_DEVICE_SCOPE host mirror emulation; no CUDA kernels execute");
#else
    backend = "CUDA";
#endif
#endif
    static state_image cpu[STEPS], tested[STEPS], resumed[STEPS];
    const char *names[] = {"typed-actions-noise", "legacy-noise", "mode-alternation", "learned-policy"};
    for (int kind = 0; kind < 4; ++kind) {
        scenario = names[kind];
        int before = failures;
        if (trajectory(kind, 0, 0, cpu) && trajectory(kind, device, 0, tested) &&
            trajectory(kind, device, 1, resumed)) {
            for (int step = 0; step < STEPS; ++step) {
                at_step = step;
                if (!same_state(&cpu[step], &tested[step]) || !same_state(&tested[step], &resumed[step])) break;
                if (tested[step].policy_hash != resumed[step].policy_hash) {
                    fprintf(stderr, "FAIL %s step=%d: resumed acquired policy hash\n", scenario, step);
                    failures++;
                    break;
                }
                checks++;
            }
        }
        printf("CHUCK_DEVICE_CASE backend=%s scenario=%s status=%s steps=%d\n",
               backend, scenario, failures == before ? "PASS" : "FAIL", STEPS);
    }
    scenario = "forward-mode-switch";
    {
        int before = failures;
        forward_mode_switch(device);
        printf("CHUCK_DEVICE_CASE backend=%s scenario=%s status=%s\n",
               backend, scenario, failures == before ? "PASS" : "FAIL");
    }
#if !defined(CHUCK_HOST_EMULATION)
    scenario = "linear-autograd";
    linear_image reference, candidate;
    int before = failures;
    if (linear_training(0, &reference) && linear_training(device, &candidate)) {
        const float *expected_arrays[] = {reference.weights, reference.losses, reference.gradient_norms,
                                         reference.gradients};
        const float *actual_arrays[] = {candidate.weights, candidate.losses, candidate.gradient_norms,
                                       candidate.gradients};
        const char *field_names[] = {"weights", "losses", "gradient_norms", "gradients"};
        const int sizes[] = {512, 8, 8, 8 * 512};
        for (int field = 0; field < 4; ++field) {
            for (int j = 0; j < sizes[field]; ++j) {
                if (!near(expected_arrays[field][j], actual_arrays[field][j], 3e-4f)) {
                    fprintf(stderr, "FAIL linear-autograd %s[%d] cpu=%.9g candidate=%.9g\n",
                            field_names[field], j, expected_arrays[field][j], actual_arrays[field][j]);
                    failures++;
                    break;
                }
                checks++;
            }
        }
    }
    printf("CHUCK_DEVICE_CASE backend=%s scenario=linear-autograd status=%s\n",
           backend, failures == before ? "PASS" : "FAIL");
#endif
    close_body();
    nt_set_gpu_mode(0);
#ifdef USE_CUDA
    gpu_shutdown();
#endif
    printf("CHUCK_DEVICE_%s backend=%s checks=%d failures=%d max_abs_error=%.9g\n",
           failures ? "FAIL" : "OK", backend, checks, failures, max_abs_error);
    return failures ? 1 : 0;
}
