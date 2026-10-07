// Explicit arrays own the whole learning step; canonical kernels own arithmetic.
#include "notorch.h"
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include "numerical_values_reference.h"

static unsigned checks;
#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static int near(float actual, double expected, double tolerance) {
    return isfinite(actual) && fabs((double)actual - expected) <=
        tolerance * (1.0 + fabs(expected));
}

static void primitive_values(void) {
    const float w[] = {1, 2, -3, 4, -5, 6}, b[] = {0.25f, -0.5f};
    const float x[] = {0.5f, -2, 3}, dy[] = {0.4f, -0.25f};
    float y[2], vjp[11];
    CHECK(nt_linear_values(w, b, x, 2, 3, y) == 0);
    CHECK(y[0] == -12.25f && y[1] == 29.5f);
    CHECK(nt_linear_vjp_values(w, x, dy, 2, 3, vjp) == 0);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) CHECK(vjp[r * 3 + c] == dy[r] * x[c]);
        CHECK(vjp[6 + r] == dy[r]);
    }
    for (int c = 0; c < 3; c++)
        CHECK(near(vjp[8 + c], (double)w[c] * dy[0] + (double)w[3 + c] * dy[1], 1e-7));

    // Bias-first accumulation is observable: (1e8 - 1e8) + 1 = 1.
    const float order_w[] = {-1e8f, 1}, order_b[] = {1e8f}, ones[] = {1, 1};
    CHECK(nt_linear_values(order_w, order_b, ones, 1, 2, y) == 0);
    CHECK(y[0] == 1.0f);

    const float tx[] = {-FLT_MAX, -4, -0.25f, -0.0f, 0.25f, 4, FLT_MAX};
    const float td[] = {1, -1, 0.25f, -2, 3, 0, -1};
    float activation[7], derivative[7];
    CHECK(nt_tanh_values(tx, 7, activation) == 0);
    CHECK(nt_tanh_vjp_values(activation, td, 7, derivative) == 0);
    for (int i = 0; i < 7; i++) {
        CHECK(near(activation[i], tanh((double)tx[i]), 1e-7));
        CHECK(near(derivative[i], td[i] * (1.0 - (double)activation[i] * activation[i]), 1e-7));
    }
    CHECK(signbit(activation[3]));
    CHECK(derivative[0] == 0 && derivative[6] == 0);

    const float p[] = {1, -2, 0.5f}, t[] = {0, 1, -0.5f};
    float loss[4], step[3];
    CHECK(nt_mse_grad_values(p, t, 3, loss) == 0);
    CHECK(near(loss[0], 11.0 / 3.0, 1e-7));
    CHECK(near(loss[1], 2.0 / 3.0, 1e-7));
    CHECK(loss[2] == -2 && near(loss[3], 2.0 / 3.0, 1e-7));
    CHECK(nt_sgd_values(p, loss + 1, 3, 0.125f, step) == 0);
    for (int i = 0; i < 3; i++) CHECK(step[i] == p[i] - 0.125f * loss[i + 1]);
    CHECK(nt_sgd_values(p, loss + 1, 3, 0, step) == 0);
    CHECK(memcmp(p, step, sizeof(p)) == 0);
    puts("PASS linear order, packed VJP, tanh, mean loss and functional SGD");
}

static int mlp(const float* p, const float* x, float target,
               float* prediction, float* loss, float* gradient) {
    float w1[40], b1[8], z1[8], h[8], z2[1], y[1], mse[2];
    float dz2[1], back2[17], dz1[8], back1[53];
    for (int r = 0; r < 8; r++) {
        memcpy(w1 + r * 5, p + r * 6, 5 * sizeof(float));
        b1[r] = p[r * 6 + 5];
    }
    if (nt_linear_values(w1, b1, x, 8, 5, z1) || nt_tanh_values(z1, 8, h) ||
        nt_linear_values(p + 48, p + 56, h, 1, 8, z2) || nt_tanh_values(z2, 1, y) ||
        nt_mse_grad_values(y, &target, 1, mse) || nt_tanh_vjp_values(y, mse + 1, 1, dz2) ||
        nt_linear_vjp_values(p + 48, h, dz2, 1, 8, back2) ||
        nt_tanh_vjp_values(h, back2 + 9, 8, dz1) ||
        nt_linear_vjp_values(w1, x, dz1, 8, 5, back1)) return -1;
    for (int r = 0; r < 8; r++) {
        memcpy(gradient + r * 6, back1 + r * 5, 5 * sizeof(float));
        gradient[r * 6 + 5] = back1[40 + r];
        gradient[48 + r] = back2[r];
    }
    gradient[56] = back2[8];
    *prediction = y[0];
    *loss = mse[0];
    return 0;
}

static double double_loss(const double* p, float target) {
    double h[8];
    for (int r = 0; r < 8; r++) {
        double z = p[r * 6 + 5];
        for (int c = 0; c < 5; c++) z += p[r * 6 + c] * value_mlp_input[c];
        h[r] = tanh(z);
    }
    double z = p[56];
    for (int r = 0; r < 8; r++) z += p[48 + r] * h[r];
    double diff = tanh(z) - target;
    return diff * diff;
}

static void gradient_and_trajectory(void) {
    float p[57], before[57], gradient[57], next[57], prediction, loss;
    double dp[57];
    memcpy(p, value_mlp_initial, sizeof(p));
    memcpy(before, p, sizeof(p));
    CHECK(mlp(p, value_mlp_input, value_mlp_targets[0], &prediction, &loss, gradient) == 0);
    CHECK(memcmp(before, p, sizeof(p)) == 0);
    for (int i = 0; i < 57; i++) dp[i] = p[i];
    for (int i = 0; i < 57; i++) {
        double v = dp[i];
        dp[i] = v + 1e-5;
        double plus = double_loss(dp, value_mlp_targets[0]);
        dp[i] = v - 1e-5;
        double minus = double_loss(dp, value_mlp_targets[0]);
        dp[i] = v;
        CHECK(near(gradient[i], (plus - minus) / 2e-5, 3e-6));
    }
    for (int s = 0; s < 32; s++) {
        const mlp_value_step* ref = &value_mlp_reference[s];
        CHECK(mlp(p, value_mlp_input, value_mlp_targets[s % 8], &prediction, &loss, gradient) == 0);
        CHECK(near(prediction, ref->prediction, 3e-6));
        CHECK(near(loss, ref->loss, 3e-6));
        memcpy(before, p, sizeof(p));
        CHECK(nt_sgd_values(p, gradient, 57, 0.01f, next) == 0);
        CHECK(memcmp(before, p, sizeof(p)) == 0);
        for (int i = 0; i < 57; i++) CHECK(near(next[i], ref->params[i], 3e-6));
        memcpy(p, next, sizeof(p));
    }
    puts("PASS all 57 finite-difference gradients and 32 independent Python SGD steps");
}

static void invalid_values(void) {
    float good[] = {0.25f, -0.5f, 0.75f, 1}, out[12], sentinel[12];
    for (int i = 0; i < 12; i++) sentinel[i] = -91.0f;
    const int bad_n[] = {0, -1, INT_MIN, NT_MAX_ELEMENTS + 1, INT_MAX};
    for (unsigned k = 0; k < sizeof(bad_n) / sizeof(*bad_n); k++) {
        int n = bad_n[k];
        memcpy(out, sentinel, sizeof(out));
        CHECK(nt_linear_values(good, good, good, n, 1, out) == -1);
        CHECK(nt_linear_values(good, good, good, 1, n, out) == -1);
        CHECK(nt_linear_vjp_values(good, good, good, n, 1, out) == -1);
        CHECK(nt_linear_vjp_values(good, good, good, 1, n, out) == -1);
        CHECK(nt_tanh_values(good, n, out) == -1);
        CHECK(nt_tanh_vjp_values(good, good, n, out) == -1);
        CHECK(nt_mse_grad_values(good, good, n, out) == -1);
        CHECK(nt_sgd_values(good, good, n, 0.1f, out) == -1);
        uint64_t state = 12;
        CHECK(nt_rng_normal_values(&state, n, out) == -1 && state == 12);
        CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
    }
    CHECK(nt_linear_values(good, good, good, NT_MAX_ELEMENTS, NT_MAX_ELEMENTS, out) == -1);
    CHECK(nt_linear_values(good, good, good, 65536, 65536, out) == -1);
    CHECK(nt_linear_vjp_values(good, good, good, 65536, 65536, out) == -1);
    CHECK(nt_linear_vjp_values(good, good, good, NT_MAX_ELEMENTS, 1, out) == -1);
    CHECK(nt_mse_grad_values(good, good, NT_MAX_ELEMENTS, out) == -1);

    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (unsigned k = 0; k < sizeof(bad) / sizeof(*bad); k++) {
        float broken[] = {0.25f, bad[k]};
        memcpy(out, sentinel, sizeof(out));
        CHECK(nt_linear_values(broken, good, good, 1, 2, out) == -1);
        CHECK(nt_linear_values(good, broken, good, 2, 1, out) == -1);
        CHECK(nt_linear_values(good, good, broken, 1, 2, out) == -1);
        CHECK(nt_linear_vjp_values(broken, good, good, 1, 2, out) == -1);
        CHECK(nt_linear_vjp_values(good, broken, good, 1, 2, out) == -1);
        CHECK(nt_linear_vjp_values(good, good, broken, 2, 1, out) == -1);
        CHECK(nt_tanh_values(broken, 2, out) == -1);
        CHECK(nt_tanh_vjp_values(broken, good, 2, out) == -1);
        CHECK(nt_tanh_vjp_values(good, broken, 2, out) == -1);
        CHECK(nt_mse_grad_values(broken, good, 2, out) == -1);
        CHECK(nt_mse_grad_values(good, broken, 2, out) == -1);
        CHECK(nt_sgd_values(broken, good, 2, 0.1f, out) == -1);
        CHECK(nt_sgd_values(good, broken, 2, 0.1f, out) == -1);
        CHECK(nt_sgd_values(good, good, 2, bad[k], out) == -1);
        CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
    }
    CHECK(nt_sgd_values(good, good, 1, -0.1f, out) == -1);
    float outside = nextafterf(1.0f, 2.0f);
    CHECK(nt_tanh_vjp_values(&outside, good, 1, out) == -1);
    outside = -outside;
    CHECK(nt_tanh_vjp_values(&outside, good, 1, out) == -1);

    CHECK(nt_linear_values(NULL, good, good, 1, 1, out) == -1);
    CHECK(nt_linear_values(good, NULL, good, 1, 1, out) == -1);
    CHECK(nt_linear_values(good, good, NULL, 1, 1, out) == -1);
    CHECK(nt_linear_values(good, good, good, 1, 1, NULL) == -1);
    CHECK(nt_linear_vjp_values(NULL, good, good, 1, 1, out) == -1);
    CHECK(nt_linear_vjp_values(good, NULL, good, 1, 1, out) == -1);
    CHECK(nt_linear_vjp_values(good, good, NULL, 1, 1, out) == -1);
    CHECK(nt_linear_vjp_values(good, good, good, 1, 1, NULL) == -1);
    CHECK(nt_tanh_values(NULL, 1, out) == -1);
    CHECK(nt_tanh_values(good, 1, NULL) == -1);
    CHECK(nt_tanh_vjp_values(NULL, good, 1, out) == -1);
    CHECK(nt_tanh_vjp_values(good, NULL, 1, out) == -1);
    CHECK(nt_tanh_vjp_values(good, good, 1, NULL) == -1);
    CHECK(nt_mse_grad_values(NULL, good, 1, out) == -1);
    CHECK(nt_mse_grad_values(good, NULL, 1, out) == -1);
    CHECK(nt_mse_grad_values(good, good, 1, NULL) == -1);
    CHECK(nt_sgd_values(NULL, good, 1, 1, out) == -1);
    CHECK(nt_sgd_values(good, NULL, 1, 1, out) == -1);
    CHECK(nt_sgd_values(good, good, 1, 1, NULL) == -1);
    CHECK(nt_rng_normal_values(NULL, 1, out) == -1);
    uint64_t state = 88;
    CHECK(nt_rng_normal_values(&state, 1, NULL) == -1 && state == 88);

    const float large[] = {FLT_MAX, FLT_MAX}, zero[] = {0, 0}, two[] = {2, 2};
    CHECK(nt_linear_values(large, zero, two, 1, 2, out) == -1);
    CHECK(nt_linear_vjp_values(good, large, two, 1, 2, out) == -1);
    CHECK(nt_linear_vjp_values(large, zero, two, 2, 1, out) == -1);
    CHECK(nt_mse_grad_values(large, zero, 2, out) == -1);
    CHECK(nt_sgd_values(zero, large, 2, 2, out) == -1);
    puts("PASS dimensions, nulls, nonfinite inputs, activation domains and overflow rejection");
}

static void normal_values(void) {
    for (unsigned i = 0; i < sizeof(normal_value_reference) / sizeof(*normal_value_reference); i++) {
        const normal_value_case* ref = &normal_value_reference[i];
        uint64_t state = ref->before;
        float value;
        CHECK(nt_rng_normal_values(&state, 1, &value) == 0);
        CHECK(state == ref->after);
        CHECK(near(value, ref->value, 3e-7));
    }
    float together[17], split[17];
    uint64_t a, b;
    nt_rng_seed(&a, 73);
    b = a;
    CHECK(nt_rng_normal_values(&a, 17, together) == 0);
    CHECK(nt_rng_normal_values(&b, 3, split) == 0);
    CHECK(nt_rng_normal_values(&b, 14, split + 3) == 0);
    CHECK(a == b && memcmp(together, split, sizeof(split)) == 0);
    nt_tensor* legacy = nt_tensor_new(3);
    CHECK(legacy != NULL);
    a = b;
    nt_seed(919);
    nt_tensor_rand(legacy, 3);
    srand(193);
    (void)rand();
    CHECK(nt_rng_normal_values(&a, 17, together) == 0);
    CHECK(nt_rng_normal_values(&b, 17, split) == 0);
    CHECK(a == b && memcmp(together, split, sizeof(split)) == 0);
    nt_tensor_free(legacy);

    // Fixed seed, fixed 65,536-sample population; expected moments 0 and 1.
    double sum = 0, squares = 0;
    nt_rng_seed(&a, 42);
    for (int batch = 0; batch < 4096; batch++) {
        float values[16];
        CHECK(nt_rng_normal_values(&a, 16, values) == 0);
        for (int i = 0; i < 16; i++) {
            sum += values[i];
            squares += (double)values[i] * values[i];
        }
    }
    CHECK(fabs(sum / 65536) < 0.015);
    CHECK(fabs(squares / 65536 - 1.0) < 0.025);
    puts("PASS 256 Python normal/state vectors, chunk invariance, legacy isolation and moments");
}

typedef struct { int id, status; float params[57]; uint64_t state; } worker_data;

static void* worker(void* opaque) {
    worker_data* data = (worker_data*)opaque;
    memcpy(data->params, value_mlp_initial, sizeof(data->params));
    nt_rng_seed(&data->state, (uint64_t)(100 + data->id));
    data->status = 0;
    for (int i = 0; i < 64; i++) {
        float noise[2], prediction, loss, grad[57], next[57];
        if (nt_rng_normal_values(&data->state, 2, noise) ||
            mlp(data->params, value_mlp_input, value_mlp_targets[(i + data->id) % 8],
                &prediction, &loss, grad) ||
            nt_sgd_values(data->params, grad, 57, 0.01f, next)) {
            data->status = -1;
            return NULL;
        }
        memcpy(data->params, next, sizeof(next));
    }
    return NULL;
}

static void tape_and_workers(void) {
    nt_tape_start();
    nt_tensor* parameter = nt_tensor_new(1);
    CHECK(parameter != NULL);
    parameter->data[0] = 0.3f;
    int leaf = nt_tape_param(parameter);
    int activation = nt_tanh(leaf);
    int loss = nt_mul(activation, activation);
    CHECK(leaf >= 0 && activation >= 0 && loss >= 0);
    nt_tape_backward(loss);
    nt_tape* before = (nt_tape*)malloc(sizeof(nt_tape));
    CHECK(before != NULL);
    memcpy(before, nt_tape_get(), sizeof(*before));
    float output_before[3], grad_before[3];
    for (int i = 0; i < 3; i++) {
        output_before[i] = before->entries[i].output->data[0];
        grad_before[i] = before->entries[i].grad->data[0];
    }
    worker_data expected[8], actual[8];
    pthread_t threads[8];
    for (int i = 0; i < 8; i++) {
        expected[i].id = i;
        worker(&expected[i]);
        actual[i].id = i;
        CHECK(pthread_create(&threads[i], NULL, worker, &actual[i]) == 0);
    }
    for (int i = 0; i < 8; i++) {
        CHECK(pthread_join(threads[i], NULL) == 0);
        CHECK(expected[i].status == 0 && actual[i].status == 0);
        CHECK(actual[i].state == expected[i].state);
        CHECK(memcmp(actual[i].params, expected[i].params, sizeof(actual[i].params)) == 0);
    }
    CHECK(memcmp(before, nt_tape_get(), sizeof(*before)) == 0);
    for (int i = 0; i < 3; i++) {
        CHECK(before->entries[i].output->data[0] == output_before[i]);
        CHECK(before->entries[i].grad->data[0] == grad_before[i]);
    }
    free(before);
    nt_tape_destroy();
    nt_tensor_free(parameter);
    puts("PASS eight independent training workers and byte-unchanged active legacy tape");
}

int main(void) {
    primitive_values();
    gradient_and_trajectory();
    invalid_values();
    normal_values();
    tape_and_workers();
    printf("NUMERICAL_VALUES_OK %u checks\n", checks);
    return 0;
}
