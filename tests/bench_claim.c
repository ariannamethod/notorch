/* claimbench.c — what the row claim costs when its counter shares a line, and when it does not.
 *
 * The pool coordinates through three counters: workers spin reading `generation`, claim rows
 * with an atomic add on `next`, and report completion by decrementing `busy`. Adjacent, those
 * three sit in one 64-byte line, so every claim invalidates the line the others are reading.
 * Decode did not measurably care on four cores; this asks whether the effect exists at all and
 * how it grows with thread count, which is the part four cores cannot answer.
 *
 * Build: make bench_claim
 * Run:   taskset -c 4-7 ./bench_claim [threads] [claims-per-thread]
 *
 * Exynos 1580, four big cores, 2M claims per thread: one line 51.9 Mclaims/s against 121.3
 * separated at four threads, 2.34x; 1.25x at one thread, 1.95x at eight. Decode does not
 * move, because a matvec makes 68 claims — 1.3 us shared against 0.6 — and a token runs a
 * few hundred matvecs against ninety milliseconds of arithmetic. Real, and small where we
 * stand; the layout is separated anyway because the cost of doing so is 192 bytes in one
 * global and the ratio grows with anything that claims more often.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <stdint.h>

#define LINE 64

/* The two layouts, side by side, so nothing else differs between them. */
typedef struct { int generation; int busy; int next; } shared_layout;
typedef struct {
    _Alignas(LINE) int generation;
    _Alignas(LINE) int busy;
    _Alignas(LINE) int next;
} split_layout;

static shared_layout g_shared;
static split_layout  g_split;
static int nthreads, claims, use_split;
static int cpus[64];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* One worker: read the generation the way the spin loop does, then claim a row. The read is
 * what makes the sharing bite — without it a lone atomic add on a private line is just an
 * atomic add. */
static void *worker(void *p) {
    long id = (long)p;
    cpu_set_t one;
    CPU_ZERO(&one); CPU_SET(cpus[id % nthreads], &one);
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one);

    int sink = 0;
    for (int i = 0; i < claims; i++) {
        if (use_split) {
            sink += __atomic_load_n(&g_split.generation, __ATOMIC_ACQUIRE);
            __atomic_fetch_add(&g_split.next, 1, __ATOMIC_RELAXED);
        } else {
            sink += __atomic_load_n(&g_shared.generation, __ATOMIC_ACQUIRE);
            __atomic_fetch_add(&g_shared.next, 1, __ATOMIC_RELAXED);
        }
    }
    return (void *)(long)sink;
}

static double run(int split) {
    use_split = split;
    g_shared.next = 0; g_split.next = 0;
    pthread_t th[64];
    double t0 = now();
    for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    return now() - t0;
}

int main(int argc, char **argv) {
    nthreads = argc > 1 ? atoi(argv[1]) : 4;
    claims   = argc > 2 ? atoi(argv[2]) : 2000000;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE && n < 64; c++) if (CPU_ISSET(c, &mask)) cpus[n++] = c;
    if (n == 0) return 1;
    if (nthreads > n) for (int i = n; i < nthreads && i < 64; i++) cpus[i] = cpus[i % n];

    run(0); run(1);                                    /* warm both paths */
    double a = run(0), b = run(1);
    double total = (double)nthreads * (double)claims;
    printf("%d threads x %d claims: one line %.2f Mclaims/s, separated %.2f  (%.2fx)\n",
           nthreads, claims, total / a / 1e6, total / b / 1e6, a / b);
    return 0;
}
