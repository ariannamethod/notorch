/* Haiku MathBrain/RAE: tanh at both layers, scalar MSE, plain SGD.
 * Reference arithmetic is double precision and uses the analytic chain rule;
 * finite differences check every parameter at the initial state.
 * Build needs C and libm only. */
#include "notorch.h"
#include <stdio.h>
#include <string.h>

static int checks;

#define CHECK(ok) do { \
    checks++; \
    if (!(ok)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #ok); \
        exit(1); \
    } \
} while (0)

static void close_to(double got, double expected, double tolerance) {
    checks++;
    if (!(fabs(got - expected) <= tolerance)) {
        fprintf(stderr, "FAIL: %.12g != %.12g (tolerance %.3g)\n",
                got, expected, tolerance);
        exit(1);
    }
}

static int input(nt_tensor *t) {
    int idx = nt_tape_record(t, NT_OP_NONE, -1, -1, 0.0f);
    CHECK(idx >= 0);
    return idx;
}

static void test_tanh(void) {
    const float values[] = {-100.0f, -2.0f, -0.5f, 0.0f,
                            0.5f, 2.0f, 100.0f, 1e-6f};
    nt_tensor *x = nt_tensor_new2d(2, 4);
    CHECK(x != NULL);
    memcpy(x->data, values, sizeof(values));
    nt_tape_start();
    int xi = nt_tape_param(x);
    CHECK(xi >= 0);
    int count = nt_tape_get()->count;
    CHECK(nt_tanh(-1) == -1);
    CHECK(nt_tanh(count) == -1);
    CHECK(nt_tanh(NT_TAPE_MAX_ENTRIES) == -1);
    CHECK(nt_tape_get()->count == count);
    int yi = nt_tanh(xi);
    CHECK(yi >= 0);
    nt_tape *t = nt_tape_get();
    nt_tensor *y = t->entries[yi].output;
    CHECK(t->entries[yi].op == NT_OP_TANH);
    CHECK(y->ndim == 2 && y->shape[0] == 2 && y->shape[1] == 4);
    CHECK(y->stride[0] == 4 && y->stride[1] == 1);
    for (int j = 0; j < x->len; j++)
        close_to(y->data[j], tanh((double)values[j]), 8e-8);

    /* Multiple consumers accumulate upstream: sum(y*y + 0.3*y). */
    int square = nt_mul(yi, yi);
    int scaled = nt_scale(yi, 0.3f);
    int loss = nt_add(square, scaled);
    CHECK(loss >= 0);
    nt_tape_backward(loss);
    CHECK(t->entries[xi].grad != NULL);
    for (int j = 0; j < x->len; j++) {
        double yy = tanh((double)values[j]);
        double derivative = (2.0 * yy + (double)0.3f) * (1.0 - yy * yy);
        close_to(t->entries[xi].grad->data[j], derivative, 3e-7);
        const double eps = 1e-4;
        double plus = tanh((double)values[j] + eps);
        double minus = tanh((double)values[j] - eps);
        double numerical = (plus * plus + (double)0.3f * plus
                          - minus * minus - (double)0.3f * minus) / (2 * eps);
        close_to(t->entries[xi].grad->data[j], numerical, 3e-7);
    }
    nt_tape_destroy();

    nt_tape_start();
    xi = nt_tape_param_frozen(x);
    yi = nt_tanh(xi);
    CHECK(yi >= 0);
    nt_tape_backward(yi);
    CHECK(nt_tape_get()->entries[xi].grad == NULL);
    nt_tape_destroy();
    CHECK(nt_tanh(0) == -1);
    nt_tensor_free(x);
    puts("PASS tanh: shape, saturated/central values, branching gradient, frozen input");
}

static void test_sgd_contract(void) {
    nt_tensor *p = nt_tensor_new(3);
    nt_tensor *frozen = nt_tensor_new(3);
    nt_tensor *unused = nt_tensor_new(3);
    nt_tensor *x = nt_tensor_new(3);
    nt_tensor *base = nt_tensor_new(3);
    CHECK(p && frozen && unused && x && base);
    const float start[] = {4.99f, -4.99f, 0.25f};
    const float direction[] = {-2.0f, 3.0f, -0.5f};
    memcpy(p->data, start, sizeof(start));
    memcpy(x->data, direction, sizeof(direction));
    nt_tensor_fill(frozen, 0.75f);
    nt_tensor_fill(unused, 0.125f);
    nt_tensor_fill(base, -0.625f);
    nt_tape_start();
    int pi = nt_tape_param(p);
    int xi = input(x);             /* Slot and entry indices differ. */
    int fi = nt_tape_param(frozen);
    int ui = nt_tape_param(unused);
    int bi = nt_tape_param_frozen(base);
    CHECK(pi >= 0 && fi >= 0 && ui >= 0 && bi >= 0);
    int product = nt_mul(pi, xi);
    int fixed_sum = nt_add(fi, bi);
    int loss = nt_add(product, fixed_sum);
    CHECK(loss >= 0);
    nt_tape_backward(loss);
    nt_tape *t = nt_tape_get();
    CHECK(t->entries[fi].grad != NULL);
    CHECK(t->entries[xi].grad != NULL);
    CHECK(t->entries[ui].grad == NULL);
    CHECK(t->entries[bi].grad == NULL);
    /* Freezing after backward must protect an already-present gradient. */
    nt_tape_freeze_param(fi);
    nt_tape_no_decay(pi);

    float moments[3][2][3];
    for (int k = 0; k < 3; k++) {
        t->adam[k].t = k + 7;
        for (int j = 0; j < 3; j++) {
            moments[k][0][j] = t->adam[k].m->data[j] = (k + j + 1) * 0.125f;
            moments[k][1][j] = t->adam[k].v->data[j] = (k + j + 1) * 0.0625f;
        }
    }
    nt_chuck_state chuck = t->chuck;
    nt_chuck_param_state chuck_params[NT_TAPE_MAX_PARAMS];
    memcpy(chuck_params, t->chuck_params, sizeof(chuck_params));
    nt_tape_sgd_step(0.1f);
    for (int j = 0; j < 3; j++) {
        close_to(p->data[j], start[j] - 0.1f * direction[j], 0);
        close_to(t->entries[pi].grad->data[j], direction[j], 0);
        close_to(frozen->data[j], 0.75f, 0);
        close_to(unused->data[j], 0.125f, 0);
        close_to(base->data[j], -0.625f, 0);
        close_to(x->data[j], direction[j], 0);
    }
    CHECK(p->data[0] > 5.0f && p->data[1] < -5.0f);
    /* Haiku owns its [-5,5] clamp, after the numerical optimizer step. */
    for (int j = 0; j < 3; j++) p->data[j] = fmaxf(-5, fminf(5, p->data[j]));
    CHECK(p->data[0] == 5.0f && p->data[1] == -5.0f);
    for (int k = 0; k < 3; k++) {
        CHECK(t->adam[k].t == k + 7);
        CHECK(memcmp(t->adam[k].m->data, moments[k][0], sizeof(moments[k][0])) == 0);
        CHECK(memcmp(t->adam[k].v->data, moments[k][1], sizeof(moments[k][1])) == 0);
    }
    CHECK(memcmp(&t->chuck, &chuck, sizeof(chuck)) == 0);
    CHECK(memcmp(t->chuck_params, chuck_params, sizeof(chuck_params)) == 0);
    nt_tape_destroy();
    nt_tensor_free(p); nt_tensor_free(frozen); nt_tensor_free(unused);
    nt_tensor_free(x); nt_tensor_free(base);
    puts("PASS SGD: frozen/unused/non-parameter entries, external clamp, optimizer state");
}

#define MAX_WIDTH 8
#define MAX_WEIGHTS (MAX_WIDTH * MAX_WIDTH)

/* Four parameter groups: W1[hidden,in], b1[hidden], W2[1,hidden], b2[1]. */
static double reference(int nin, int hidden, double p[4][MAX_WEIGHTS],
                        const float *x, double target,
                        double grad[4][MAX_WEIGHTS], double *score) {
    double h[MAX_WIDTH];
    double z = p[3][0];
    for (int j = 0; j < hidden; j++) {
        double a = p[1][j];
        for (int k = 0; k < nin; k++) a += p[0][j * nin + k] * (double)x[k];
        h[j] = tanh(a);
        z += p[2][j] * h[j];
    }
    double y = tanh(z);
    double error = y - target;
    *score = y;
    if (grad) {
        double dz = 2 * error * (1 - y * y);
        grad[3][0] = dz;
        for (int j = 0; j < hidden; j++) {
            grad[2][j] = dz * h[j];
            grad[1][j] = dz * p[2][j] * (1 - h[j] * h[j]);
            for (int k = 0; k < nin; k++)
                grad[0][j * nin + k] = grad[1][j] * (double)x[k];
        }
    }
    return error * error;
}

static void test_network(int nin, int hidden) {
    nt_tensor *p[] = {nt_tensor_new2d(hidden, nin), nt_tensor_new(hidden),
                      nt_tensor_new2d(1, hidden), nt_tensor_new(1)};
    nt_tensor *x = nt_tensor_new(nin);
    nt_tensor *target = nt_tensor_new(1);
    double rp[4][MAX_WEIGHTS] = {{0}};
    double gradient[4][MAX_WEIGHTS] = {{0}};
    CHECK(x && target);
    for (int group = 0; group < 4; group++) {
        CHECK(p[group] != NULL);
        for (int j = 0; j < p[group]->len; j++)
            rp[group][j] = p[group]->data[j] = ((j * 7 + group * 3) % 17 - 8) * 0.05f;
    }

    for (int step = 0; step < 24; step++) {
        for (int j = 0; j < nin; j++) x->data[j] = ((step * 3 + j * 7) % 19 - 9) * 0.1f;
        target->data[0] = (step % 5) * 0.2f;
        double expected_score;
        double expected_loss = reference(nin, hidden, rp, x->data, target->data[0],
                                         gradient, &expected_score);
        nt_tape_start();
        int xi = input(x);
        int ti = input(target);
        int ids[4];
        for (int group = 0; group < 4; group++) {
            ids[group] = nt_tape_param(p[group]);
            CHECK(ids[group] >= 0);
        }
        int a = nt_linear(ids[0], xi, ids[1]);
        int h = nt_tanh(a);
        int z = nt_linear(ids[2], h, ids[3]);
        int y = nt_tanh(z);
        int negative_target = nt_scale(ti, -1.0f);
        int diff = nt_add(y, negative_target);
        int loss = nt_mul(diff, diff);
        CHECK(loss >= 0);
        nt_tape *t = nt_tape_get();
        close_to(t->entries[y].output->data[0], expected_score, 2e-6);
        close_to(t->entries[loss].output->data[0], expected_loss, 3e-6);
        nt_tape_backward(loss);
        for (int group = 0; group < 4; group++) {
            CHECK(t->entries[ids[group]].grad != NULL);
            for (int j = 0; j < p[group]->len; j++) {
                close_to(t->entries[ids[group]].grad->data[j], gradient[group][j], 3e-6);
                if (step == 0) {
                    double saved = rp[group][j], ignored;
                    const double eps = 1e-6;
                    rp[group][j] = saved + eps;
                    double plus = reference(nin, hidden, rp, x->data, target->data[0], NULL, &ignored);
                    rp[group][j] = saved - eps;
                    double minus = reference(nin, hidden, rp, x->data, target->data[0], NULL, &ignored);
                    rp[group][j] = saved;
                    close_to(t->entries[ids[group]].grad->data[j], (plus - minus) / (2 * eps), 3e-6);
                }
            }
        }
        const float lr = 0.01f;
        nt_tape_sgd_step(lr);
        for (int group = 0; group < 4; group++) {
            for (int j = 0; j < p[group]->len; j++) {
                rp[group][j] = fmax(-5, fmin(5, rp[group][j] - (double)lr * gradient[group][j]));
                p[group]->data[j] = fmaxf(-5, fminf(5, p[group]->data[j]));
                close_to(p[group]->data[j], rp[group][j], 2e-6);
            }
        }
    }
    nt_tape_destroy();
    for (int group = 0; group < 4; group++) nt_tensor_free(p[group]);
    nt_tensor_free(x); nt_tensor_free(target);
    printf("PASS %d->%d->1: double-reference forward/gradients and 24 SGD steps\n", nin, hidden);
}

int main(void) {
    test_tanh();
    test_sgd_contract();
    test_network(5, 8);  /* Python MathBrain and default RecursiveRAESelector. */
    test_network(6, 4);  /* RAE exposes configurable feature/hidden dimensions. */
    printf("TANH_SGD_OK: %d checks\n", checks);
    return 0;
}
