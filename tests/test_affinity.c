/* test_affinity.c — notorch must choose its cores, and must stop choosing when told.
 *
 * On a machine whose cores are not all the same speed, counting the affinity mask answers the
 * wrong question: the matvec splits into equal chunks and waits for whichever chunk landed on
 * a small core. Measured on Exynos 1580, Gemma 4 E2B decode: 10.2 t/s on the four big cores,
 * 8.3 with one small core added, 7.9 across all eight, 6.0 on the prime core alone.
 *
 * The expected core set here is computed from sysfs by this file, not asked of the library.
 * A test that calls the same function it is checking agrees with itself no matter what the
 * function does.
 *
 * Four modes, four processes, because the decision is taken once per process:
 *   test_affinity            — the default: narrow on a mixed machine, no-op on a uniform one
 *   NT_QMV_BIG_ONLY=0 ... off — the class opt-out must be honoured
 *   test_affinity narrowed    — a mask somebody already narrowed picks the cores, and we
 *                               still spread across every core they gave us
 *   NT_QMV_PIN=0 ... nopin    — the affinity opt-out must be honoured completely
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

static int fails = 0;

static void check(const char *what, int ok, const char *detail) {
    printf("  %s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail ? " — " : "", detail ? detail : "");
    if (!ok) fails++;
}

#if defined(__linux__)
static long peak_khz(int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long v = 0;
    if (fscanf(f, "%ld", &v) != 1) v = 0;
    fclose(f);
    return v;
}

/* The set this test expects notorch to settle on, derived independently: everything in `in`
 * whose peak clock is above the slowest present, or `in` itself when that is not a choice. */
static int expected_set(const cpu_set_t *in, cpu_set_t *out) {
    long slow = 0, fast = 0;
    int seen = 0, n = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, in)) continue;
        long f = peak_khz(c);
        if (f <= 0) { *out = *in; return CPU_COUNT(in); }
        if (!seen || f < slow) slow = f;
        if (f > fast) fast = f;
        seen++;
    }
    if (!seen || slow == fast) { *out = *in; return CPU_COUNT(in); }
    CPU_ZERO(out);
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, in) && peak_khz(c) > slow) { CPU_SET(c, out); n++; }
    if (n < 2) { *out = *in; return CPU_COUNT(in); }
    return n;
}

static void describe(const cpu_set_t *s, char *buf, size_t n) {
    size_t off = 0;
    buf[0] = 0;
    for (int c = 0; c < CPU_SETSIZE && off + 8 < n; c++)
        if (CPU_ISSET(c, s)) off += (size_t)snprintf(buf + off, n - off, "%d ", c);
}

/* One real matvec, big enough to take the threaded path, so the pool exists and has pinned
 * whatever it is going to pin. */
static int run_one_matvec(void) {
    const int k = 1536, m = 512;
    size_t rowsz = (size_t)(k / 32) * 18;                 /* Q4_0 */
    uint8_t *W = (uint8_t *)malloc(rowsz * (size_t)m);
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *out = (float *)malloc((size_t)m * sizeof(float));
    if (!W || !x || !out) return -1;
    memset(W, 0x42, rowsz * (size_t)m);
    for (int i = 0; i < k; i++) x[i] = 0.5f;
    int rc = nt_qmatvec_i8(out, W, 2, x, m, k);
    free(W); free(x); free(out);
    return rc;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "default";
    printf("core selection [%s]\n", mode);

    cpu_set_t start;
    CPU_ZERO(&start);
    if (sched_getaffinity(0, sizeof(start), &start) != 0) {
        printf("  SKIP sched_getaffinity unavailable\n");
        return 0;
    }

    if (!strcmp(mode, "narrowed")) {
        /* Two cores, chosen from the ones we already have, standing in for taskset or a
         * cgroup. Nothing notorch does may widen this or narrow it further. */
        cpu_set_t two;
        CPU_ZERO(&two);
        int taken = 0;
        for (int c = 0; c < CPU_SETSIZE && taken < 2; c++)
            if (CPU_ISSET(c, &start)) { CPU_SET(c, &two); taken++; }
        if (taken < 2) { printf("  SKIP fewer than two CPUs to work with\n"); return 0; }
        if (sched_setaffinity(0, sizeof(two), &two) != 0) {
            printf("  SKIP sched_setaffinity refused\n");
            return 0;
        }
        start = two;
    }

    cpu_set_t want;
    int want_n = expected_set(&start, &want);
    if (!strcmp(mode, "off") || !strcmp(mode, "narrowed")) {
        want = start;                       /* opted out, or already somebody's decision */
        want_n = CPU_COUNT(&start);
    }
    /* NT_QMV_PIN=0 is about affinity alone. How many threads to run is a separate question
     * and still follows the core classes, so only the placement assertion changes below. */

    char a[256], b[256];
    describe(&want, a, sizeof(a));

    int planned = nt_qmv_planned_threads();
    char detail[128];
    snprintf(detail, sizeof(detail), "planned %d, expected %d over cpus %s", planned, want_n, a);
    check("thread count matches the core plan", planned == want_n, detail);

    if (run_one_matvec() != 0) { check("matvec runs", 0, "returned non-zero"); return 1; }

    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(0, sizeof(now), &now) != 0) {
        check("affinity readable after the matvec", 0, NULL);
        return 1;
    }
    describe(&now, b, sizeof(b));
    /* One thread per core, in order, so the thread that drove the matvec holds the first CPU
     * of the plan and nothing else. A shared mask is not enough: the scheduler will put two
     * spin-then-park threads on one core and keep them there. */
    cpu_set_t first;
    CPU_ZERO(&first);
    if (!strcmp(mode, "nopin")) first = start;        /* nothing may have touched it */
    else
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &want)) { CPU_SET(c, &first); break; }
    char fbuf[64];
    describe(&first, fbuf, sizeof(fbuf));
    /* The message has to name what this mode actually expects. With NT_QMV_PIN=0 the mask
     * must come back untouched, and calling that "the plan's first core" would send whoever
     * reads a failure here looking in the wrong place. */
    if (!strcmp(mode, "nopin")) {
        snprintf(detail, sizeof(detail), "on cpus %s, expected the mask untouched: %s", b, fbuf);
        check("nothing touched the affinity mask", CPU_EQUAL(&now, &first), detail);
    } else {
        snprintf(detail, sizeof(detail), "on cpus %s, expected the plan's first core %s", b, fbuf);
        check("the driving thread holds one core of its own", CPU_EQUAL(&now, &first), detail);
    }

    printf("\nResults: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
#else
int main(void) {
    printf("core selection\n  SKIP not Linux — nothing here selects cores\n");
    return 0;
}
#endif
