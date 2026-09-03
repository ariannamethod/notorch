/* Threading must not move a single bit of a packed matvec.
 *
 * Rows are disjoint and each row's accumulation is self-contained, so how the rows are
 * divided among workers cannot reach the result. That is a real invariant, not a hope:
 * the pools hand rows out on demand, which means the division differs run to run with
 * core speed, and any kernel that accumulated across rows — or any dispatch that skipped
 * or truncated a range — would show up here as a changed byte.
 *
 * The single-threaded run is the reference. nt_qmv_set_thread_min lifts the floor above
 * the shape to force it, then drops it to force fan-out, so one process covers both.
 * m is deliberately not a multiple of any plausible core count, to exercise the tail.
 * How finely the rows are divided is read once per process, so the Makefile runs this
 * under several NT_QMV_CHUNKS values: coarse and fine must agree to the bit with each
 * other and with the single-threaded reference. It also runs once at NT_QMV_SPIN=0, which
 * is the only way the park-and-wake path is exercised at all: at the default budget a worker
 * almost never reaches the condvar, so that code would otherwise go years without executing.
 * Stated exactly, because it was tried: that run caught nothing the default run missed. A
 * lost wakeup is a race and no bit comparison finds it, and corrupting the busy count — the
 * hazard the dispatch comment warns about — fails both runs. It is coverage before the next
 * refactor touches that path, not a second detector.
 *
 * Two failure modes were injected while writing this to confirm it can fail: a cursor
 * that skips a row per chunk, and a range one row short. Both were caught. An overlapping
 * cursor was NOT caught, and correctly so — the kernels assign out[row] rather than
 * accumulate into it, so a row computed twice is written twice with the same value. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../notorch.h"

static uint64_t xs = 88172645463325252ULL;
static uint32_t rnd(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return (uint32_t)(xs >> 32); }

#define M_ROWS 4099          /* prime: no core count divides it evenly */
#define K_COLS 2048

static uint8_t *make_weights(int blk_bytes, int blk_vals) {
    int nb = K_COLS / blk_vals;
    size_t bytes = (size_t)M_ROWS * nb * blk_bytes;
    uint8_t *W = malloc(bytes);
    if (!W) return NULL;
    for (size_t i = 0; i < bytes; i++) W[i] = (uint8_t)(rnd() & 0xFF);
    /* Clamp the f16 scale exponents so random bytes cannot produce inf/NaN, which would
     * compare equal-or-not for reasons that have nothing to do with the dispatch. */
    for (int r = 0; r < M_ROWS; r++)
        for (int b = 0; b < nb; b++) {
            uint8_t *bl = W + ((size_t)r * nb + b) * blk_bytes;
            bl[1] = (uint8_t)(0x2C | (bl[1] & 0x80));
            if (blk_bytes == 144 || blk_bytes == 210)
                bl[3] = (uint8_t)(0x2C | (bl[3] & 0x80));
        }
    return W;
}

static int check(const char *name, int dtype, int blk_bytes, int blk_vals, const float *x) {
    uint8_t *W = make_weights(blk_bytes, blk_vals);
    if (!W) { printf("%-5s  SKIP (out of memory)\n", name); return 0; }

    float *one_f = calloc(M_ROWS, sizeof(float)), *many_f = calloc(M_ROWS, sizeof(float));
    float *one_i = calloc(M_ROWS, sizeof(float)), *many_i = calloc(M_ROWS, sizeof(float));
    int fails = 0;

    nt_qmv_set_thread_min(1L << 40);            /* above any shape: stay on one core */
    int rf = nt_qmatvec(one_f, W, dtype, x, M_ROWS, K_COLS);
    int ri = nt_qmatvec_i8(one_i, W, dtype, x, M_ROWS, K_COLS);

    nt_qmv_set_thread_min(1);                   /* below any shape: fan out */
    nt_qmatvec(many_f, W, dtype, x, M_ROWS, K_COLS);
    nt_qmatvec_i8(many_i, W, dtype, x, M_ROWS, K_COLS);

    size_t nbytes = (size_t)M_ROWS * sizeof(float);
    if (rf == 0) {
        int bad = memcmp(one_f, many_f, nbytes) != 0;
        printf("%-5s f32  threaded vs single  %s\n", name, bad ? "DIFFERS" : "identical");
        fails += bad;
    }
    if (ri == 0) {
        int bad = memcmp(one_i, many_i, nbytes) != 0;
        printf("%-5s i8   threaded vs single  %s\n", name, bad ? "DIFFERS" : "identical");
        fails += bad;
    }
    free(W); free(one_f); free(many_f); free(one_i); free(many_i);
    return fails;
}

int main(void) {
    float *x = malloc((size_t)K_COLS * sizeof(float));
    for (int i = 0; i < K_COLS; i++)
        x[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;

    int fails = 0;
    fails += check("Q4_0", 2,  18,  32,  x);
    fails += check("Q5_0", 6,  22,  32,  x);
    fails += check("Q8_0", 8,  34,  32,  x);
    fails += check("Q4_K", 12, 144, 256, x);
    fails += check("Q6_K", 14, 210, 256, x);
    free(x);

    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("%d dispatch(es) FAILED\n", fails);
    return 1;
}
