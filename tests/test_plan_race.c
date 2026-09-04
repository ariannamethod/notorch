/* test_plan_race.c — two pools, two threads, one plan.
 *
 * The fan-out settles its thread count, its core set, its pinning and its chunk size once.
 * That used to be four functions each caching into a function-scope static behind a lazy
 * `if (x < 0)`, which is a data race rather than a shortcut: this library has two worker
 * pools behind two separate pthread_once guards, so a program that calls the float matvec
 * from one thread and the integer matvec from another runs both initialisers at the same
 * moment and both of them reach into those statics.
 *
 * Nothing here is a timing trick. The two calls are released together so the initialisers
 * overlap, through a mutex and a condition variable rather than pthread_barrier — the barrier
 * API is optional in POSIX and absent on macOS, which this repo builds on. The run is meant to
 * be read by ThreadSanitizer:
 *
 *     cc -fsanitize=thread -O1 -pthread -I. -o test_plan_race tests/test_plan_race.c notorch.c -lm
 *     setarch -R ./test_plan_race          # Android's ASLR is wider than TSan expects
 *
 * Without a sanitizer it still checks the thing a race would break: both paths must agree on
 * the plan, and the results must match a single-threaded reference bit for bit.
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define K_COLS 1024
#define M_ROWS 257            /* prime: no core count divides it */

static uint8_t *W;            /* Q4_0 rows, shared by both callers */
static float *x;
static float *out_f, *out_i;
/* The starting gate: both callers wait until the second one arrives, then go together. */
static pthread_mutex_t gate_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cv = PTHREAD_COND_INITIALIZER;
static int gate_arrived = 0;
static int seen_threads_f, seen_threads_i;

static void gate_wait(void) {
    pthread_mutex_lock(&gate_mu);
    if (++gate_arrived == 2) pthread_cond_broadcast(&gate_cv);
    else while (gate_arrived < 2) pthread_cond_wait(&gate_cv, &gate_mu);
    pthread_mutex_unlock(&gate_mu);
}

static void *call_float(void *p) {
    (void)p;
    gate_wait();
    nt_qmatvec(out_f, W, 2, x, M_ROWS, K_COLS);
    seen_threads_f = nt_qmv_planned_threads();
    return NULL;
}

static void *call_int(void *p) {
    (void)p;
    gate_wait();
    nt_qmatvec_i8(out_i, W, 2, x, M_ROWS, K_COLS);
    seen_threads_i = nt_qmv_planned_threads();
    return NULL;
}

int main(void) {
    printf("plan under concurrent first use\n");

    size_t rowsz = (size_t)(K_COLS / 32) * 18;
    W = (uint8_t *)malloc(rowsz * (size_t)M_ROWS);
    x = (float *)malloc((size_t)K_COLS * sizeof(float));
    out_f = (float *)malloc((size_t)M_ROWS * sizeof(float));
    out_i = (float *)malloc((size_t)M_ROWS * sizeof(float));
    if (!W || !x || !out_f || !out_i) { printf("  FAIL out of memory\n"); return 1; }
    for (size_t i = 0; i < rowsz * (size_t)M_ROWS; i++) W[i] = (uint8_t)(i * 37u + 11u);
    for (int i = 0; i < K_COLS; i++) x[i] = (float)((i % 23) - 11) * 0.0625f;

    pthread_t a, b;
    pthread_create(&a, NULL, call_float, NULL);
    pthread_create(&b, NULL, call_int, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);

    int fails = 0;
    if (seen_threads_f != seen_threads_i) {
        printf("  FAIL the two pools disagree on the plan: %d against %d\n",
               seen_threads_f, seen_threads_i);
        fails++;
    } else {
        printf("  PASS both pools agree on the plan — %d threads\n", seen_threads_f);
    }

    /* Same weights through both paths: the integer kernel quantizes the activation, so this
     * is not a bit comparison between them. Each is compared against itself run alone, which
     * is what a corrupted plan or a half-built core set would disturb. */
    float *ref_f = (float *)malloc((size_t)M_ROWS * sizeof(float));
    float *ref_i = (float *)malloc((size_t)M_ROWS * sizeof(float));
    if (!ref_f || !ref_i) { printf("  FAIL out of memory\n"); return 1; }
    nt_qmatvec(ref_f, W, 2, x, M_ROWS, K_COLS);
    nt_qmatvec_i8(ref_i, W, 2, x, M_ROWS, K_COLS);
    int bad_f = memcmp(ref_f, out_f, (size_t)M_ROWS * sizeof(float)) != 0;
    int bad_i = memcmp(ref_i, out_i, (size_t)M_ROWS * sizeof(float)) != 0;
    printf("  %s float path matches its own serial run\n", bad_f ? "FAIL" : "PASS");
    printf("  %s integer path matches its own serial run\n", bad_i ? "FAIL" : "PASS");
    fails += bad_f + bad_i;

    free(W); free(x); free(out_f); free(out_i); free(ref_f); free(ref_i);
    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
