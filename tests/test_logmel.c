/*
 * test_logmel.c — the audio front end: window, transform, filter bank, normalise.
 *
 * The transform is the part that can be subtly wrong forever, so it is checked
 * against a direct DFT computed from the definition in double precision — an
 * O(n^2) implementation that shares no code and no table with the recursion it is
 * judging. Both non-power-of-two lengths (400 = 2^4 * 25, which makes the recursion
 * fall through to its odd-length DFT) and clean powers of two are covered, because
 * those are different paths through nt_fft_rec.
 *
 * The log-mel is checked on what survives its own normalisation: the clamp sets the
 * floor exactly eight decades under the peak and the affine divides by four, so the
 * output range is exactly 2.0 whenever any frame reaches the floor, and a silence
 * tail guarantees one does. That is an invariant of the arithmetic rather than a
 * number copied out of a run, which is the only kind worth asserting here.
 *
 * Bit-equality with whisper.cpp's own front end is not assertable in this repo — it
 * needs a model file and a wav — and is gated downstream in ears (tests/gate_mel.sh),
 * where it reads max|d| = 0 over 328000 values.
 *
 * NT_LOGMEL_THREADS sets the fan-out; the result must not depend on it.
 *
 * Build: cc -O2 -I. tests/test_logmel.c notorch.c -lm [-DUSE_BLAS -lopenblas]
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;

static void ok(int cond, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf(cond ? "  PASS " : "  FAIL ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (cond) n_pass++; else n_fail++;
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The definition of a real-input DFT, in double, with no table and no recursion.
 * power[k] = |sum_n x[n] e^{-2 pi i k n / N}|^2 for k in [0, N/2]. */
static void dft_power_ref(const float *x, int N, double *power) {
    for (int k = 0; k <= N / 2; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; n++) {
            double th = 2.0 * M_PI * (double)k * (double)n / (double)N;
            re += (double)x[n] * cos(th);
            im -= (double)x[n] * sin(th);
        }
        power[k] = re * re + im * im;
    }
}

/* One STFT shape against the reference. Compared relative to the largest power in
 * the frame, because a float transform of length N carries ~N ulps of the peak and
 * an absolute limit would just be a disguised statement about the signal's scale. */
static void check_stft(int n_fft, int hop, int n_frames, int n_signal, int use_window,
                       unsigned seed, double rel_limit) {
    const int n_bins = 1 + n_fft / 2;
    float *sig = malloc((size_t)n_signal * sizeof(float));
    float *win = malloc((size_t)n_fft * sizeof(float));
    float *pw  = malloc((size_t)n_frames * n_bins * sizeof(float));
    double *ref = malloc((size_t)n_bins * sizeof(double));
    float *frame = malloc((size_t)n_fft * sizeof(float));
    if (!sig || !win || !pw || !ref || !frame) { ok(0, "allocation"); goto done; }

    srand(seed);
    for (int i = 0; i < n_signal; i++) sig[i] = frand();
    nt_hann_window(win, n_fft);

    if (nt_stft(pw, sig, n_signal, n_fft, hop, n_frames, use_window ? win : NULL, 4) != 0) {
        ok(0, "nt_stft n_fft=%d returned -1", n_fft); goto done;
    }

    double worst = 0.0; int worst_f = -1, worst_k = -1;
    for (int f = 0; f < n_frames; f++) {
        for (int i = 0; i < n_fft; i++) {
            int t = f * hop + i;
            float v = (t < n_signal) ? sig[t] : 0.0f;
            frame[i] = use_window ? win[i] * v : v;
        }
        dft_power_ref(frame, n_fft, ref);
        double peak = 0.0;
        for (int k = 0; k < n_bins; k++) if (ref[k] > peak) peak = ref[k];
        if (peak < 1e-12) peak = 1e-12;
        for (int k = 0; k < n_bins; k++) {
            double d = fabs((double)pw[(size_t)f * n_bins + k] - ref[k]) / peak;
            if (d > worst) { worst = d; worst_f = f; worst_k = k; }
        }
    }
    ok(worst <= rel_limit,
       "nt_stft n_fft=%d hop=%d frames=%d window=%d vs a direct DFT — worst relative "
       "%.3e at frame %d bin %d, limit %.0e",
       n_fft, hop, n_frames, use_window, worst, worst_f, worst_k, rel_limit);

done:
    free(sig); free(win); free(pw); free(ref); free(frame);
}

static void check_hann(void) {
    const int n = 400;
    float w[400];
    nt_hann_window(w, n);
    ok(w[0] == 0.0f, "periodic Hann starts at zero");
    ok(w[n / 2] > 0.999f && w[n / 2] <= 1.0f, "periodic Hann peaks at 1 in the middle — w[200] = %.6f", w[n / 2]);
    /* Periodic, not symmetric: w[i] == w[n-i] for i >= 1, and w[n-1] != w[0]. */
    int sym = 1;
    for (int i = 1; i < n; i++) if (fabsf(w[i] - w[n - i]) > 1e-6f) sym = 0;
    ok(sym, "periodic Hann is symmetric about n/2 — w[i] == w[n-i]");
    ok(w[n - 1] > 0.0f,
       "periodic Hann does NOT return to zero at n-1 (w[%d] = %.6e); the symmetric "
       "window does, and is a different window", n - 1, w[n - 1]);
}

/* A deterministic stand-in for a model's filter bank: overlapping triangles across
 * the bins. Not whisper's bank — the real one comes out of the weight file — but the
 * same shape of object, which is what the matmul and the normalisation need. */
static void make_filters(float *f, int n_mel, int n_bins) {
    memset(f, 0, (size_t)n_mel * n_bins * sizeof(float));
    for (int b = 0; b < n_mel; b++) {
        double lo = (double)b * (n_bins - 1) / (n_mel + 1);
        double mid = (double)(b + 1) * (n_bins - 1) / (n_mel + 1);
        double hi = (double)(b + 2) * (n_bins - 1) / (n_mel + 1);
        for (int k = 0; k < n_bins; k++) {
            double v = 0.0;
            if (k >= lo && k <= mid && mid > lo) v = (k - lo) / (mid - lo);
            else if (k > mid && k <= hi && hi > mid) v = (hi - k) / (hi - mid);
            f[(size_t)b * n_bins + k] = (float)v;
        }
    }
}

static void check_logmel(int n_threads) {
    const int n_fft = 400, hop = 160, n_mel = 80, n_bins = 1 + n_fft / 2;
    const int n_samples = 16000, pad_tail = 16000 * 2;   /* 1 s of signal, 2 s of silence */
    float *pcm = malloc((size_t)n_samples * sizeof(float));
    float *filt = malloc((size_t)n_mel * n_bins * sizeof(float));
    nt_mel mel; memset(&mel, 0, sizeof(mel));
    if (!pcm || !filt) { ok(0, "allocation"); free(pcm); free(filt); return; }

    srand(7);
    for (int i = 0; i < n_samples; i++)
        pcm[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 16000.0f) + 0.02f * frand();
    make_filters(filt, n_mel, n_bins);

    if (nt_logmel(&mel, pcm, n_samples, filt, n_mel, n_bins, n_fft, hop, pad_tail, n_threads) != 0) {
        ok(0, "nt_logmel returned -1"); free(pcm); free(filt); return;
    }

    /* Geometry, from the padding the header describes. */
    size_t n_padded = (size_t)n_samples + pad_tail + 2 * (size_t)(n_fft / 2);
    int want_len = (int)((n_padded - n_fft) / hop);
    int want_org = 1 + (n_samples + n_fft / 2 - n_fft) / hop;
    ok(mel.n_len == want_len, "n_len over the padded signal — %d, expected %d", mel.n_len, want_len);
    ok(mel.n_len_org == want_org, "n_len_org counts only frames carrying audio — %d, expected %d",
       mel.n_len_org, want_org);
    ok(mel.n_len_org < mel.n_len, "the silence tail adds frames — %d real of %d", mel.n_len_org, mel.n_len);

    /* The normalisation's own invariant: floor at peak-8 decades, then /4. */
    size_t n = (size_t)mel.n_mel * mel.n_len;
    float mx = mel.data[0], mn = mel.data[0];
    for (size_t i = 0; i < n; i++) { if (mel.data[i] > mx) mx = mel.data[i]; if (mel.data[i] < mn) mn = mel.data[i]; }
    ok(fabsf((mx - mn) - 2.0f) < 1e-5f,
       "clamp eight decades under the peak then (x+4)/4 gives a range of exactly 2 — %.7f", mx - mn);

    /* The silence tail must sit on the floor, and the audio must not. */
    int tail_on_floor = 1;
    for (int b = 0; b < n_mel; b++)
        if (fabsf(mel.data[(size_t)b * mel.n_len + (mel.n_len - 1)] - mn) > 1e-6f) tail_on_floor = 0;
    ok(tail_on_floor, "every bin of the last silent frame sits on the floor");

    int finite = 1;
    for (size_t i = 0; i < n; i++) if (!isfinite(mel.data[i])) finite = 0;
    ok(finite, "no NaN or inf anywhere in %zu values", n);

    nt_mel_free(&mel);
    ok(mel.data == NULL && mel.n_len == 0, "nt_mel_free clears the struct it frees");
    free(pcm); free(filt);
}

/* The fan-out must not touch the numbers: one thread and many must agree bit for bit,
 * which is what striping frames (rather than splitting a sum) buys. */
static void check_thread_invariance(void) {
    const int n_fft = 400, hop = 160, n_mel = 80, n_bins = 1 + n_fft / 2;
    const int n_samples = 8000, pad_tail = 16000;
    float *pcm = malloc((size_t)n_samples * sizeof(float));
    float *filt = malloc((size_t)n_mel * n_bins * sizeof(float));
    nt_mel a, b; memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    srand(11);
    for (int i = 0; i < n_samples; i++) pcm[i] = frand();
    make_filters(filt, n_mel, n_bins);

    int ra = nt_logmel(&a, pcm, n_samples, filt, n_mel, n_bins, n_fft, hop, pad_tail, 1);
    int rb = nt_logmel(&b, pcm, n_samples, filt, n_mel, n_bins, n_fft, hop, pad_tail, 6);
    if (ra || rb) { ok(0, "nt_logmel failed (%d, %d)", ra, rb); goto done; }

    int same = (a.n_len == b.n_len && a.n_mel == b.n_mel);
    size_t diff = 0;
    if (same)
        for (size_t i = 0; i < (size_t)a.n_mel * a.n_len; i++)
            if (memcmp(&a.data[i], &b.data[i], sizeof(float)) != 0) diff++;
    ok(same && diff == 0, "1 thread and 6 threads agree bit for bit over %zu values (%zu differ)",
       (size_t)a.n_mel * a.n_len, diff);

    /* And the same for the plain STFT. */
    const int n_frames = 64;
    float *p1 = malloc((size_t)n_frames * n_bins * sizeof(float));
    float *p6 = malloc((size_t)n_frames * n_bins * sizeof(float));
    float win[400]; nt_hann_window(win, n_fft);
    nt_stft(p1, pcm, n_samples, n_fft, hop, n_frames, win, 1);
    nt_stft(p6, pcm, n_samples, n_fft, hop, n_frames, win, 6);
    ok(memcmp(p1, p6, (size_t)n_frames * n_bins * sizeof(float)) == 0,
       "nt_stft is bit-identical at 1 and 6 threads over %d values", n_frames * n_bins);
    free(p1); free(p6);

done:
    nt_mel_free(&a); nt_mel_free(&b);
    free(pcm); free(filt);
}

static void check_rejects(void) {
    float x[16] = {0}, f[16] = {0}, p[16] = {0};
    nt_mel m;
    ok(nt_logmel(&m, x, 16, f, 8, 100, 400, 160, 0, 4) == -1,
       "a filter bank that is not 1 + n_fft/2 wide is refused");
    ok(nt_logmel(&m, NULL, 16, f, 8, 201, 400, 160, 0, 4) == -1, "a NULL signal is refused");
    ok(nt_logmel(&m, x, 16, f, 8, 201, 400, 0, 0, 4) == -1, "hop 0 is refused");
    ok(nt_stft(p, x, 16, 0, 160, 4, NULL, 1) == -1, "nt_stft refuses n_fft 0");
    ok(nt_stft(p, x, 16, 16, 160, 0, NULL, 1) == -1, "nt_stft refuses a zero frame count");
}

int main(void) {
    const char *env = getenv("NT_LOGMEL_THREADS");
    int nth = env ? atoi(env) : 4;
    if (nth < 1) nth = 1;
    printf("nt_stft / nt_logmel — the front end, at %d thread(s)\n\n", nth);

    check_hann();

    /* 400 is whisper's length and the interesting one: 2^4 * 25, so the recursion
     * splits four times and finishes 25 with the direct DFT. 512 and 256 stay on the
     * radix-2 path the whole way down; 480 splits to an odd 15. */
    check_stft(400, 160, 12, 4000, 1, 21, 1e-5);
    check_stft(400, 160, 12, 4000, 0, 22, 1e-5);
    check_stft(512, 256,  8, 4000, 1, 23, 1e-5);
    check_stft(256, 128,  8, 3000, 1, 24, 1e-5);
    check_stft(480, 160,  6, 3000, 1, 25, 1e-5);
    check_stft( 64,  32,  4,  500, 1, 26, 1e-5);
    /* Frames running off the end of the signal are zero-padded, not garbage. */
    check_stft(400, 160,  8,  900, 1, 27, 1e-5);

    check_logmel(nth);
    check_thread_invariance();
    check_rejects();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
