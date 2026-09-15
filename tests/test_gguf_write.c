/*
 * test_gguf_write.c — the writer against the reader that has to live with it.
 *
 * A writer is only correct with respect to a reader, so every gate here is a round trip:
 * what gguf_write_* put in the file is read back through gguf_open, gguf_dequant,
 * gguf_get_kv and the gguf_read_*_array family, and compared to what went in. Byte
 * equality where the format is exact — F32 tensors, every scalar key, every array — and
 * a stated half-ULP budget where it is not, which is F16 and nothing else.
 *
 * Three things beyond the round trip, because a round trip alone can be passed by a
 * writer and reader that are wrong in the same direction:
 *
 *   - alignment is checked against the format's rule (32 bytes) rather than against the
 *     reader, at element counts that are odd, prime and not multiples of anything, since
 *     a padding bug hides completely behind shapes that divide;
 *   - the F16 path is checked for INequality against its own input, because a conversion
 *     that quietly copied f32 would satisfy every tolerance in this file;
 *   - the refusals are checked for actually refusing — duplicate name, data out of
 *     declaration order, short tensor, metadata after the header, a file closed with a
 *     tensor undelivered — and each one is also checked for leaving no file behind. A
 *     partial GGUF is worse than none, because it loads.
 *
 * Last, a fake molequla stage-4 checkpoint: 53 tensors, 4 834 408 parameters, exactly
 * the 19 337 632 bytes of f32 the resonator design counts on, written a tensor at a time
 * and read back mapped. Tensor values are generated from a hash of (tensor, index), so
 * the comparison regenerates them rather than keeping a second copy — which is also what
 * lets the peak resident set of the write mean anything.
 *
 *   ./test_gguf_write           the gates
 *   ./test_gguf_write rss19     write the stage-4 set alone, report VmHWM
 *   ./test_gguf_write rss91     write 91 MB through the chunked entry points, report VmHWM
 *
 * The two rss modes are separate processes on purpose: VmHWM is a high-water mark for the
 * life of a process, and reading a 91 MB file back through a populated mapping in the same
 * process would put the file in the number.
 *
 * Build: make test_gguf_write
 */
#include "gguf.h"
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

static const char *outdir(void) {
    const char *d = getenv("NT_TEST_DIR");
    return d && *d ? d : "/tmp";
}

static const char *scratch(const char *leaf) {
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s/%s", outdir(), leaf);
    return buf;
}

/* Values from a hash of (tensor, index) rather than from an array: 19 MB of weights can
 * then be compared without a 19 MB copy of what they were supposed to be. */
static uint32_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (uint32_t)x;
}

static float gen(uint64_t stream, uint64_t i) {
    return (float)(int32_t)mix64(stream * 1000003ull + i) / 2147483648.0f;   /* [-1, 1) */
}

/* The most an f16 may differ from the float it was rounded from: half a ULP inside the
 * normal range, half the subnormal step of 2^-24 below it. A flat relative tolerance is
 * wrong here and wrong in the direction that matters — a weight of 1e-6 rounds to a
 * subnormal half and is off by a hundredth of itself, which is the format being used
 * correctly, and a flat 4.9e-4 would call it a bug. */
static double f16_budget(double want) {
    double a = fabs(want);
    if (a < 6.103515625e-05) return 2.9802322387695312e-08;   /* half of 2^-24 */
    int e; frexp(a, &e);                                      /* a is in [2^(e-1), 2^e) */
    return ldexp(1.0, e - 12);                                /* 2^(e-1) * 2^-11 */
}

static long vm_hwm_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    fclose(f);
    return kb;
}

static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

/* ── 1. Every metadata type, out and back ────────────────────────────────────── */

static void check_metadata(void) {
    const char *path = scratch("nt_gw_kv.gguf");
    const char *toks[7] = { "<s>", "</s>", "\xd0\xbc", "and", "  ", "a longer token piece", "" };
    const int32_t types[7] = { 1, 1, 6, 1, 4, 1, 1 };
    const float scores[5] = { -0.0f, 1.5f, -3.25e-8f, 65504.0f, 3.4028235e38f };
    float probe[4] = { 1.0f, -2.5f, 3.125f, -0.0f };

    gguf_writer *w = gguf_write_open(path);
    if (!w) { ok(0, "gguf_write_open(%s)", path); return; }
    int rc = 0;
    rc |= gguf_write_kv_str (w, "general.architecture", "molequla");
    rc |= gguf_write_kv_str (w, "general.name", "fake stage-4, written by test_gguf_write");
    rc |= gguf_write_kv_u32 (w, "molequla.block_count", 5);
    rc |= gguf_write_kv_u32 (w, "molequla.embedding_length", 224);
    rc |= gguf_write_kv_i32 (w, "molequla.growth_step_offset", -12345);
    rc |= gguf_write_kv_u64 (w, "molequla.corpus_ingested_total", 0x0123456789ABCDEFull);
    rc |= gguf_write_kv_f32 (w, "molequla.attention.layer_norm_rms_epsilon", 1.25e-5f);
    rc |= gguf_write_kv_bool(w, "molequla.warmup_done", 1);
    rc |= gguf_write_kv_bool(w, "molequla.asleep", 0);
    rc |= gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", toks, 7);
    rc |= gguf_write_kv_i32_array(w, "tokenizer.ggml.token_type", types, 7);
    rc |= gguf_write_kv_f32_array(w, "tokenizer.ggml.scores", scores, 5);
    ok(rc == 0, "twelve metadata keys of nine types accepted");

    uint64_t s4[1] = { 4 };
    rc = gguf_write_tensor_decl(w, "probe", 1, s4, GGUF_TYPE_F32);
    rc |= gguf_write_tensor_f32(w, "probe", probe, 4);
    rc |= gguf_write_close(w);
    ok(rc == 0, "file closed clean");

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of our own file"); return; }
    ok(gf->version == 3, "version 3 (got %u)", gf->version);
    ok(gf->n_tensors == 1 && gf->n_kv == 12,
       "header counts survive — %llu tensors, %llu keys",
       (unsigned long long)gf->n_tensors, (unsigned long long)gf->n_kv);
    ok(gf->data_offset % 32 == 0, "data section is 32-byte aligned at %llu",
       (unsigned long long)gf->data_offset);

    /* The reader's own architecture extraction has to recognise the file, or a molequla
     * checkpoint is only readable by code that knows it is one. */
    ok(strcmp(gf->arch, "molequla") == 0, "general.architecture -> gf->arch = '%s'", gf->arch);
    ok(gf->n_layers == 5 && gf->embed_dim == 224,
       "block_count and embedding_length reach gf->n_layers=%d gf->embed_dim=%d",
       gf->n_layers, gf->embed_dim);
    ok(fabsf(gf->rms_eps - 1.25e-5f) == 0.0f, "rms_epsilon f32 exact: %.9g", (double)gf->rms_eps);

    const gguf_kv *kv = gguf_get_kv(gf, "general.name");
    ok(kv && kv->type == 8 && strcmp(kv->val.str, "fake stage-4, written by test_gguf_write") == 0,
       "string value byte-equal");
    kv = gguf_get_kv(gf, "molequla.growth_step_offset");
    ok(kv && kv->type == 5 && kv->val.i32 == -12345, "int32 keeps its sign: %d",
       kv ? kv->val.i32 : 0);
    kv = gguf_get_kv(gf, "molequla.corpus_ingested_total");
    ok(kv && kv->type == 10 && kv->val.u64 == 0x0123456789ABCDEFull,
       "uint64 keeps all 64 bits: 0x%llx", kv ? (unsigned long long)kv->val.u64 : 0ull);
    kv = gguf_get_kv(gf, "molequla.warmup_done");
    ok(kv && kv->type == 7 && kv->val.b == 1, "bool true");
    kv = gguf_get_kv(gf, "molequla.asleep");
    ok(kv && kv->type == 7 && kv->val.b == 0, "bool false is a written 0, not an absent key");

    int idx = gguf_find_tensor(gf, "probe");
    float *back = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
    ok(back && memcmp(back, probe, sizeof(probe)) == 0, "the f32 tensor comes back byte-equal");
    free(back);
    gguf_close(gf);

    /* The metadata-only readers walk the same bytes without the tensor section. */
    char buf[64] = {0};
    ok(gguf_read_str_kv(path, "general.architecture", buf, sizeof(buf)) == 0 &&
       strcmp(buf, "molequla") == 0, "gguf_read_str_kv finds it without opening the data");
    uint64_t u = 0;
    ok(gguf_read_uint_kv(path, "molequla.corpus_ingested_total", &u) == 0 &&
       u == 0x0123456789ABCDEFull, "gguf_read_uint_kv agrees");

    int n = 0;
    char **ts = gguf_read_str_array(path, "tokenizer.ggml.tokens", &n);
    int same = (ts && n == 7);
    for (int i = 0; same && i < 7; i++) same = (strcmp(ts[i], toks[i]) == 0);
    ok(same, "string array of 7 round-trips, empty string and multibyte included (n=%d)", n);
    if (ts) { for (int i = 0; i < n; i++) free(ts[i]); free(ts); }

    n = 0;
    int32_t *ti = gguf_read_i32_array(path, "tokenizer.ggml.token_type", &n);
    ok(ti && n == 7 && memcmp(ti, types, sizeof(types)) == 0, "int32 array round-trips (n=%d)", n);
    free(ti);

    n = 0;
    float *sc = gguf_read_f32_array(path, "tokenizer.ggml.scores", &n);
    ok(sc && n == 5 && memcmp(sc, scores, sizeof(scores)) == 0,
       "float array round-trips bit-exact, -0.0 and FLT_MAX included (n=%d)", n);
    free(sc);
    if (getenv("NT_TEST_KEEP")) printf("  kept: %s\n", path); else remove(path);
}

/* ── 2. Alignment, at sizes that do not divide ───────────────────────────────── */

static void check_alignment_odd(void) {
    const char *path = scratch("nt_gw_odd.gguf");
    /* Deliberately awkward: 1, a prime, one byte over an alignment unit, one under. */
    const uint64_t counts[8] = { 1, 3, 7, 13, 17, 31, 33, 255 };
    gguf_writer *w = gguf_write_open(path);
    if (!w) { ok(0, "gguf_write_open(%s)", path); return; }

    char name[32];
    int rc = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t shape[1] = { counts[i] };
        snprintf(name, sizeof(name), "odd_f32_%d", i);
        rc |= gguf_write_tensor_decl(w, name, 1, shape, GGUF_TYPE_F32);
        snprintf(name, sizeof(name), "odd_f16_%d", i);
        rc |= gguf_write_tensor_decl(w, name, 1, shape, GGUF_TYPE_F16);
    }
    /* Ranks 2 to 4 as well: ndim is written per tensor and a reader that guessed it from
     * the element count would still pass a file of vectors. */
    uint64_t s2[2] = { 3, 5 }, s3[3] = { 3, 5, 7 }, s4[4] = { 3, 5, 7, 11 };
    rc |= gguf_write_tensor_decl(w, "rank2", 2, s2, GGUF_TYPE_F32);
    rc |= gguf_write_tensor_decl(w, "rank3", 3, s3, GGUF_TYPE_F32);
    rc |= gguf_write_tensor_decl(w, "rank4", 4, s4, GGUF_TYPE_F32);
    ok(rc == 0, "19 tensors of awkward shape declared");

    float vals[1536];
    for (int i = 0; i < 8; i++) {
        for (uint64_t k = 0; k < counts[i]; k++) vals[k] = gen((uint64_t)i, k);
        snprintf(name, sizeof(name), "odd_f32_%d", i);
        rc |= gguf_write_tensor_f32(w, name, vals, counts[i]);
        snprintf(name, sizeof(name), "odd_f16_%d", i);
        rc |= gguf_write_tensor_f32(w, name, vals, counts[i]);
    }
    for (int k = 0; k < 1536; k++) vals[k] = gen(100, (uint64_t)k);
    rc |= gguf_write_tensor_f32(w, "rank2", vals, 15);
    rc |= gguf_write_tensor_f32(w, "rank3", vals, 105);
    rc |= gguf_write_tensor_f32(w, "rank4", vals, 1155);
    rc |= gguf_write_close(w);
    ok(rc == 0, "19 tensors written and closed");

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of the odd-shape file"); return; }
    int aligned = 1, bad = -1;
    for (uint64_t i = 0; i < gf->n_tensors; i++)
        if (gf->tensors[i].offset % 32) { aligned = 0; if (bad < 0) bad = (int)i; }
    ok(aligned, "every one of %llu tensor offsets is a multiple of 32 (first bad: %d)",
       (unsigned long long)gf->n_tensors, bad);

    int f32_equal = 1, f16_within = 1;
    double worst_f16 = 0.0;
    for (int i = 0; i < 8; i++) {
        for (uint64_t k = 0; k < counts[i]; k++) vals[k] = gen((uint64_t)i, k);
        snprintf(name, sizeof(name), "odd_f32_%d", i);
        int idx = gguf_find_tensor(gf, name);
        float *back = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
        if (!back || memcmp(back, vals, (size_t)counts[i] * sizeof(float)) != 0) f32_equal = 0;
        free(back);

        snprintf(name, sizeof(name), "odd_f16_%d", i);
        idx = gguf_find_tensor(gf, name);
        back = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
        if (!back) { f16_within = 0; continue; }
        for (uint64_t k = 0; k < counts[i]; k++) {
            double rel = fabs((double)back[k] - (double)vals[k]) / f16_budget(vals[k]);
            if (rel > worst_f16) worst_f16 = rel;
            if (rel > 1.0) f16_within = 0;
        }
        free(back);
    }
    ok(f32_equal, "every odd-length F32 tensor is byte-equal after the padding it needed");
    ok(f16_within, "every odd-length F16 tensor is inside its half-ULP budget — worst %.3f of it", worst_f16);

    int idx = gguf_find_tensor(gf, "rank4");
    ok(idx >= 0 && gf->tensors[idx].ndim == 4 &&
       gf->tensors[idx].shape[0] == 3 && gf->tensors[idx].shape[3] == 11 &&
       gf->tensors[idx].n_elements == 1155,
       "rank and shape survive: rank4 comes back %u-D with %llu elements",
       idx >= 0 ? gf->tensors[idx].ndim : 0,
       idx >= 0 ? (unsigned long long)gf->tensors[idx].n_elements : 0ull);
    gguf_close(gf);
    remove(path);
}

/* ── 3. F16 rounds, and rounds the way the format says ───────────────────────── */

static void check_f16_rounding(void) {
    const char *path = scratch("nt_gw_f16.gguf");
    /* Exactly representable, then a value that is not, then the edges of the format. */
    /* Exactly representable, then values that are not, then the edges of the format.
     * Two of these exist only to catch truncation, which is the plausible wrong f16: a
     * mantissa that rounds UP in the normal range and one that rounds up among the
     * subnormals. Without them a truncating converter passes this gate and is caught only
     * by the statistical budget further down, which is a worse place to learn it. */
    const float in[18] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 3.140625f, 65504.0f, -65504.0f,
        6.103515625e-05f,          /* smallest normal half */
        5.960464477539063e-08f,    /* smallest subnormal half */
        2.980232238769531e-08f,    /* half of it: ties to even, so zero */
        1.0001234f, 0.1f, -1234.5f, 1e-9f, 1e9f,
        1.000732421875f,           /* 1 + 3*2^-12: three quarters of a ULP, rounds up */
        1.043081283569336e-07f     /* 1.75 * 2^-24: subnormal, rounds up to 2^-23 */
    };
    const float want[18] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 3.140625f, 65504.0f, -65504.0f,
        6.103515625e-05f, 5.960464477539063e-08f, 0.0f,
        1.0f, 0.0999755859375f, -1234.0f, 0.0f, INFINITY,
        1.0009765625f, 1.1920928955078125e-07f
    };
    uint64_t shape[1] = { 18 };
    gguf_writer *w = gguf_write_open(path);
    if (!w) { ok(0, "gguf_write_open(%s)", path); return; }
    int rc = gguf_write_tensor_decl(w, "half", 1, shape, GGUF_TYPE_F16);
    rc |= gguf_write_tensor_f32(w, "half", in, 18);
    rc |= gguf_write_close(w);
    ok(rc == 0, "18 f16 edge cases written");

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of the f16 file"); return; }
    int idx = gguf_find_tensor(gf, "half");
    ok(idx >= 0 && gguf_type_size(GGUF_TYPE_F16, 18) == 36 &&
       gf->tensors[idx].dtype == GGUF_TYPE_F16, "declared F16 is stored as F16, two bytes each");
    float *back = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
    if (!back) { ok(0, "dequant of the f16 tensor"); gguf_close(gf); return; }

    int exact = 1;
    for (int i = 0; i < 18; i++) {
        int eq = (back[i] == want[i]);
        if (i == 1) {                         /* -0.0: the sign is the point */
            eq = (back[i] == 0.0f && signbit(back[i]));
        }
        if (!eq) { ok(0, "f16[%d]: %.17g -> %.17g, wanted %.17g",
                      i, (double)in[i], (double)back[i], (double)want[i]); exact = 0; }
    }
    ok(exact, "all 18 land on the half the format would choose, ties to even and both "
              "round-ups included");

    /* A conversion that quietly copied f32 would pass every tolerance above. */
    ok(back[11] != in[11], "1.0001234 changed on the way through f16 (%.9g -> %.9g); "
                           "equality here would mean the rounding never ran",
       (double)in[11], (double)back[11]);
    free(back);
    gguf_close(gf);
    remove(path);
}

/* ── 4. A packed tensor as an opaque blob ────────────────────────────────────── */

static void check_packed_blob(void) {
    const char *path = scratch("nt_gw_q8.gguf");
    const int cols = 64, rows = 3, n = cols * rows;
    float src[192];
    for (int i = 0; i < n; i++) src[i] = gen(7, (uint64_t)i) * 3.0f;

    uint64_t packed_bytes = gguf_type_size(GGUF_TYPE_Q8_0, (uint64_t)n);
    uint8_t *packed = (uint8_t *)malloc((size_t)packed_bytes);
    if (!packed) { ok(0, "allocation"); return; }
    int qrc = 0;
    for (int r = 0; r < rows; r++)
        qrc |= nt_quantize_row(src + r * cols, packed + (size_t)r * (cols / 32 * 34), cols, GGUF_TYPE_Q8_0);
    ok(qrc == 0 && packed_bytes == 204, "nt_quantize_row produced %llu bytes of Q8_0 for %d values",
       (unsigned long long)packed_bytes, n);

    uint64_t shape[2] = { (uint64_t)cols, (uint64_t)rows };
    gguf_writer *w = gguf_write_open(path);
    if (!w) { ok(0, "gguf_write_open(%s)", path); free(packed); return; }
    int rc = gguf_write_tensor_decl(w, "packed.weight", 2, shape, GGUF_TYPE_Q8_0);
    rc |= gguf_write_tensor(w, "packed.weight", packed, packed_bytes);
    rc |= gguf_write_close(w);
    ok(rc == 0, "a dtype the writer does not understand goes through as bytes of the right size");

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of the packed file"); free(packed); return; }
    int idx = gguf_find_tensor(gf, "packed.weight");
    ok(idx >= 0 && memcmp(gf->data + gf->tensors[idx].offset, packed, (size_t)packed_bytes) == 0,
       "the blocks are in the file byte for byte");
    float *back = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
    double worst = 0.0;
    if (back) for (int i = 0; i < n; i++) {
        double d = fabs((double)back[i] - (double)src[i]);
        if (d > worst) worst = d;
    }
    ok(back && worst < 0.02, "the reader dequantizes what the writer stored — max|d| %.4f", worst);
    free(back);

    float row1[64];
    int rrc = idx >= 0 ? gguf_dequant_row(gf, idx, 1, row1) : -1;
    double rworst = 0.0;
    for (int i = 0; i < cols && rrc == 0; i++) {
        double d = fabs((double)row1[i] - (double)src[cols + i]);
        if (d > rworst) rworst = d;
    }
    ok(rrc == 0 && rworst < 0.02, "gguf_dequant_row finds row 1 at the offset we wrote — max|d| %.4f",
       rworst);
    gguf_close(gf);
    free(packed);
    remove(path);
}

/* ── 5. The refusals ─────────────────────────────────────────────────────────── */

/* One writer per refusal, because a refused writer stays refused: the contract is that a
 * caller may ignore every return value and still be told at close, which means the first
 * no has to poison the rest. That is checked here too. */
static gguf_writer *fresh(const char *path) {
    remove(path);
    return gguf_write_open(path);
}

static void check_refusals(void) {
    const char *path = scratch("nt_gw_bad.gguf");
    uint64_t s[2] = { 4, 4 };
    uint64_t one[1] = { 64 };
    float data[17] = {0};
    gguf_writer *w;

    printf("\n  (the writer's own diagnostics follow — each line is a gate being exercised)\n");

    w = fresh(path);
    gguf_write_tensor_decl(w, "dup", 2, s, GGUF_TYPE_F32);
    ok(gguf_write_tensor_decl(w, "dup", 2, s, GGUF_TYPE_F32) == -1, "a duplicate tensor name is refused");
    ok(gguf_write_close(w) == -1, "the refusal poisons the writer: close says no as well");
    ok(file_size(path) < 0, "and removes the file rather than leaving a partial GGUF to load");

    w = fresh(path);
    gguf_write_tensor_decl(w, "a", 2, s, GGUF_TYPE_F32);
    gguf_write_tensor_decl(w, "b", 2, s, GGUF_TYPE_F32);
    ok(gguf_write_tensor(w, "b", data, 64) == -1, "tensor data out of declaration order is refused");
    gguf_write_abort(w);
    ok(file_size(path) < 0, "gguf_write_abort leaves no file behind");

    w = fresh(path);
    gguf_write_tensor_decl(w, "a", 2, s, GGUF_TYPE_F32);
    ok(gguf_write_tensor(w, "a", data, 60) == -1, "a tensor short of its declared size is refused");
    gguf_write_abort(w);

    w = fresh(path);
    gguf_write_tensor_decl(w, "a", 2, s, GGUF_TYPE_F32);
    ok(gguf_write_tensor(w, "a", data, 68) == -1, "a tensor longer than declared is refused");
    gguf_write_abort(w);

    w = fresh(path);
    gguf_write_tensor_decl(w, "a", 2, s, GGUF_TYPE_F32);
    ok(gguf_write_tensor(w, "a", data, 64) == 0, "the same tensor at its declared size is accepted");
    ok(gguf_write_kv_u32(w, "too.late", 1) == -1, "metadata after the header is refused");
    gguf_write_abort(w);

    w = fresh(path);
    gguf_write_tensor_decl(w, "a", 2, s, GGUF_TYPE_F32);
    gguf_write_tensor_decl(w, "b", 2, s, GGUF_TYPE_F32);
    gguf_write_tensor(w, "a", data, 64);
    ok(gguf_write_close(w) == -1, "closing with a declared tensor undelivered is refused");
    ok(file_size(path) < 0, "and that file is removed too");

    w = fresh(path);
    uint64_t bad[1] = { 48 };
    ok(gguf_write_tensor_decl(w, "q", 1, bad, GGUF_TYPE_Q8_0) == -1,
       "48 values of Q8_0 — not a whole number of blocks — is refused at declaration");
    gguf_write_abort(w);

    w = fresh(path);
    ok(gguf_write_tensor_decl(w, "q", 1, one, GGUF_TYPE_Q8_0) == 0,
       "64 values of Q8_0 is two whole blocks and is accepted");
    gguf_write_abort(w);

    w = fresh(path);
    ok(gguf_write_tensor_decl(w, "unknown", 1, one, 99) == -1, "a dtype nothing can size is refused");
    gguf_write_abort(w);

    w = fresh(path);
    uint64_t zero[1] = { 0 };
    ok(gguf_write_tensor_decl(w, "zero", 1, zero, GGUF_TYPE_F32) == -1, "a zero dimension is refused");
    gguf_write_abort(w);

    w = fresh(path);
    ok(gguf_write_tensor_decl(w, "many", 5, s, GGUF_TYPE_F32) == -1, "a fifth dimension is refused");
    gguf_write_abort(w);

    char toolong[GGUF_MAX_NAME + 8];
    memset(toolong, 'x', sizeof(toolong) - 1);
    toolong[sizeof(toolong) - 1] = 0;

    w = fresh(path);
    ok(gguf_write_tensor_decl(w, toolong, 1, one, GGUF_TYPE_F32) == -1,
       "a tensor name the reader would drop is refused here instead");
    gguf_write_abort(w);

    w = fresh(path);
    ok(gguf_write_kv_u32(w, toolong, 1) == -1, "a key the reader would truncate is refused too");
    gguf_write_abort(w);
    ok(file_size(path) < 0, "nothing of any of that is left on disk");
}

/* ── 6. The fake molequla stage-4 checkpoint ─────────────────────────────────── */

typedef struct { char name[GGUF_MAX_NAME]; uint32_t ndim; uint64_t shape[4]; } spec;

#define STAGE4_D   224
#define STAGE4_V   750
#define STAGE4_H  1495
#define STAGE4_R    64
#define STAGE4_L     5
#define STAGE4_PARAMS 4834408ull
#define STAGE4_BYTES  (STAGE4_PARAMS * 4)

/* The shapes of §1.2 of the resonator design: T=96, D=224, L=5, V=750, and the 4 834 408
 * parameters that make 19 337 632 bytes of f32. The MLP width is what closes that sum,
 * and 1495 is not a multiple of 32 on purpose — the padding is exercised by the largest
 * tensors in the file rather than only by the toy ones above. */
static uint64_t stage4_specs(spec *out) {
    uint64_t n = 0;
    spec *s = &out[n++];
    snprintf(s->name, sizeof(s->name), "token_embd.weight");
    s->ndim = 2; s->shape[0] = STAGE4_D; s->shape[1] = STAGE4_V;
    for (int l = 0; l < STAGE4_L; l++) {
        const char *mats[4] = { "attn_q", "attn_k", "attn_v", "attn_output" };
        for (int m = 0; m < 4; m++) {
            s = &out[n++];
            snprintf(s->name, sizeof(s->name), "blk.%d.%s.weight", l, mats[m]);
            s->ndim = 2; s->shape[0] = STAGE4_D; s->shape[1] = STAGE4_D;
        }
        const char *norms[2] = { "attn_norm", "ffn_norm" };
        for (int m = 0; m < 2; m++) {
            s = &out[n++];
            snprintf(s->name, sizeof(s->name), "blk.%d.%s.weight", l, norms[m]);
            s->ndim = 1; s->shape[0] = STAGE4_D;
        }
        s = &out[n++];
        snprintf(s->name, sizeof(s->name), "blk.%d.ffn_up.weight", l);
        s->ndim = 2; s->shape[0] = STAGE4_D; s->shape[1] = STAGE4_H;
        s = &out[n++];
        snprintf(s->name, sizeof(s->name), "blk.%d.ffn_down.weight", l);
        s->ndim = 2; s->shape[0] = STAGE4_H; s->shape[1] = STAGE4_D;
        s = &out[n++];
        snprintf(s->name, sizeof(s->name), "blk.%d.rrpram_a.weight", l);
        s->ndim = 2; s->shape[0] = STAGE4_D; s->shape[1] = STAGE4_R;
        s = &out[n++];
        snprintf(s->name, sizeof(s->name), "blk.%d.rrpram_b.weight", l);
        s->ndim = 2; s->shape[0] = STAGE4_R; s->shape[1] = STAGE4_D;
    }
    s = &out[n++];
    snprintf(s->name, sizeof(s->name), "output.weight");
    s->ndim = 2; s->shape[0] = STAGE4_D; s->shape[1] = STAGE4_V;
    s = &out[n++];
    snprintf(s->name, sizeof(s->name), "molequla.growth_gate");
    s->ndim = 1; s->shape[0] = 488;
    return n;
}

static uint64_t spec_elements(const spec *s) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < s->ndim; d++) n *= s->shape[d];
    return n;
}

/* Written a tensor at a time, from a buffer the size of that tensor and no larger: the
 * largest here is ffn_up at 334 880 floats, so the peak resident set of this function is
 * 1.3 MB against a 19 MB file. That is the measurement the two rss modes report. */
static int stage4_write(const char *path, int verbose) {
    spec specs[64];
    uint64_t n = stage4_specs(specs), params = 0;
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    if (gguf_write_kv_str(w, "general.architecture", "molequla") ||
        gguf_write_kv_u32(w, "molequla.block_count", STAGE4_L) ||
        gguf_write_kv_u32(w, "molequla.embedding_length", STAGE4_D) ||
        gguf_write_kv_u32(w, "molequla.context_length", 96) ||
        gguf_write_kv_u32(w, "molequla.vocab_size", STAGE4_V) ||
        gguf_write_kv_u64(w, "molequla.global_step", 41216) ||
        gguf_write_kv_u32(w, "molequla.growth_stage", 4)) { gguf_write_abort(w); return -1; }

    for (uint64_t i = 0; i < n; i++) {
        if (gguf_write_tensor_decl(w, specs[i].name, specs[i].ndim, specs[i].shape, GGUF_TYPE_F32)) {
            gguf_write_abort(w); return -1;
        }
        params += spec_elements(&specs[i]);
    }
    if (verbose)
        printf("  stage-4: %llu tensors, %llu parameters, %llu bytes of f32\n",
               (unsigned long long)n, (unsigned long long)params,
               (unsigned long long)params * 4);

    for (uint64_t i = 0; i < n; i++) {
        uint64_t ne = spec_elements(&specs[i]);
        float *buf = (float *)malloc((size_t)ne * sizeof(float));
        if (!buf) { gguf_write_abort(w); return -1; }
        for (uint64_t k = 0; k < ne; k++) buf[k] = gen(i, k);
        int rc = gguf_write_tensor_f32(w, specs[i].name, buf, ne);
        free(buf);
        if (rc) { gguf_write_abort(w); return -1; }
    }
    return gguf_write_close(w);
}

static void check_stage4(void) {
    const char *path = scratch("nt_gw_stage4.gguf");
    spec specs[64];
    uint64_t n = stage4_specs(specs), params = 0;
    for (uint64_t i = 0; i < n; i++) params += spec_elements(&specs[i]);
    ok(params == STAGE4_PARAMS && params * 4 == STAGE4_BYTES,
       "the fake stage-4 set is %llu parameters, %llu bytes of f32 — the design's numbers",
       (unsigned long long)params, (unsigned long long)params * 4);

    ok(stage4_write(path, 1) == 0, "19.3 MB written a tensor at a time");
    long sz = file_size(path);
    ok(sz > 0 && (uint64_t)sz >= STAGE4_BYTES && (uint64_t)sz < STAGE4_BYTES + 65536,
       "the file is %ld bytes — %lld of tensor data plus metadata, directory and padding",
       sz, (long long)STAGE4_BYTES);

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of the stage-4 file"); return; }
    ok(gf->n_tensors == n, "%llu tensors read back", (unsigned long long)gf->n_tensors);
    ok(gf->n_layers == STAGE4_L && gf->vocab_size == STAGE4_V && gf->embed_dim == STAGE4_D,
       "the reader recovers L=%d V=%d D=%d from the metadata",
       gf->n_layers, gf->vocab_size, gf->embed_dim);

    /* Byte equality, not tolerance: f32 in, f32 out, through a mapping. Regenerated
     * tensor by tensor so the comparison costs one tensor rather than a second file. */
    int equal = 1, aligned = 1;
    const char *first_bad = NULL;
    uint64_t compared = 0;
    for (uint64_t i = 0; i < n && equal; i++) {
        int idx = gguf_find_tensor(gf, specs[i].name);
        if (idx < 0) { equal = 0; first_bad = specs[i].name; break; }
        const gguf_tensor_info *t = &gf->tensors[idx];
        if (t->offset % 32) aligned = 0;
        uint64_t ne = spec_elements(&specs[i]);
        if (t->n_elements != ne || t->ndim != specs[i].ndim) { equal = 0; first_bad = specs[i].name; break; }
        float *want = (float *)malloc((size_t)ne * sizeof(float));
        if (!want) { equal = 0; break; }
        for (uint64_t k = 0; k < ne; k++) want[k] = gen(i, k);
        if (memcmp(gf->data + t->offset, want, (size_t)ne * sizeof(float)) != 0) {
            equal = 0; first_bad = specs[i].name;
        }
        compared += ne;
        free(want);
    }
    ok(equal, "all %llu parameters are byte-equal through the mapping%s%s",
       (unsigned long long)compared, first_bad ? ", first mismatch at " : "",
       first_bad ? first_bad : "");
    ok(aligned, "every one of the 53 tensor offsets is 32-byte aligned");

    /* One row of the embedding table, which is how a sleeping organism reads it. */
    int idx = gguf_find_tensor(gf, "token_embd.weight");
    float row[STAGE4_D];
    int rrc = idx >= 0 ? gguf_dequant_row(gf, idx, 749, row) : -1;
    int row_ok = (rrc == 0);
    for (int k = 0; k < STAGE4_D && row_ok; k++)
        if (row[k] != gen(0, 749ull * STAGE4_D + (uint64_t)k)) row_ok = 0;
    ok(row_ok, "the last row of the embedding table reads back exactly, in place");
    gguf_close(gf);
    remove(path);
}

/* ── 7. 91 MB through the chunked entry points ───────────────────────────────── */

#define BIG_TENSORS  20
#define BIG_ELEMS    1137500ull          /* 4 550 000 B each, 91 000 000 B in all */
#define BIG_CHUNK    65536

static int big_write(const char *path, int verbose) {
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    char name[64];
    if (gguf_write_kv_str(w, "general.architecture", "molequla") ||
        gguf_write_kv_str(w, "general.name", "moments, m and v, in registration order")) {
        gguf_write_abort(w); return -1;
    }
    uint64_t shape[1] = { BIG_ELEMS };
    for (int i = 0; i < BIG_TENSORS; i++) {
        snprintf(name, sizeof(name), "moment.%02d", i);
        if (gguf_write_tensor_decl(w, name, 1, shape, GGUF_TYPE_F32)) { gguf_write_abort(w); return -1; }
    }
    uint64_t hshape[1] = { 100000 };
    if (gguf_write_tensor_decl(w, "moment.half", 1, hshape, GGUF_TYPE_F16)) { gguf_write_abort(w); return -1; }

    /* One 256 KB window for the whole file. Nothing else is held. */
    float *chunk = (float *)malloc(BIG_CHUNK * sizeof(float));
    if (!chunk) { gguf_write_abort(w); return -1; }
    for (int i = 0; i < BIG_TENSORS; i++) {
        snprintf(name, sizeof(name), "moment.%02d", i);
        if (gguf_write_tensor_begin(w, name)) { free(chunk); gguf_write_abort(w); return -1; }
        uint64_t done = 0;
        while (done < BIG_ELEMS) {
            uint64_t take = BIG_ELEMS - done;
            if (take > BIG_CHUNK) take = BIG_CHUNK;
            for (uint64_t k = 0; k < take; k++) chunk[k] = gen((uint64_t)(200 + i), done + k);
            if (gguf_write_tensor_chunk(w, chunk, take * 4)) { free(chunk); gguf_write_abort(w); return -1; }
            done += take;
        }
        if (gguf_write_tensor_end(w)) { free(chunk); gguf_write_abort(w); return -1; }
    }
    /* The same path with rounding on the way out, one window at a time. */
    if (gguf_write_tensor_begin(w, "moment.half")) { free(chunk); gguf_write_abort(w); return -1; }
    for (uint64_t done = 0; done < 100000; done += BIG_CHUNK) {
        uint64_t take = 100000 - done;
        if (take > BIG_CHUNK) take = BIG_CHUNK;
        for (uint64_t k = 0; k < take; k++) chunk[k] = gen(300, done + k);
        if (gguf_write_tensor_chunk_f32(w, chunk, take)) { free(chunk); gguf_write_abort(w); return -1; }
    }
    if (gguf_write_tensor_end(w)) { free(chunk); gguf_write_abort(w); return -1; }
    free(chunk);
    if (verbose)
        printf("  streamed: %d tensors of %llu f32 plus one of 100000 f16, %d-float window\n",
               BIG_TENSORS, (unsigned long long)BIG_ELEMS, BIG_CHUNK);
    return gguf_write_close(w);
}

static void check_big(void) {
    const char *path = scratch("nt_gw_big.gguf");
    ok(big_write(path, 1) == 0, "91 MB written through begin/chunk/end");
    long sz = file_size(path);
    ok(sz > 91000000L && sz < 91400000L, "the file is %ld bytes", sz);

    gguf_file *gf = gguf_open(path);
    if (!gf) { ok(0, "gguf_open of the 91 MB file"); return; }
    int equal = 1, aligned = 1;
    char name[64];
    for (int i = 0; i < BIG_TENSORS && equal; i++) {
        snprintf(name, sizeof(name), "moment.%02d", i);
        int idx = gguf_find_tensor(gf, name);
        if (idx < 0) { equal = 0; break; }
        if (gf->tensors[idx].offset % 32) aligned = 0;
        const float *got = (const float *)(gf->data + gf->tensors[idx].offset);
        /* Every chunk boundary, plus the ends: a window written twice or skipped once
         * shows up exactly there and nowhere else. */
        for (uint64_t k = 0; k < BIG_ELEMS && equal; k += BIG_CHUNK) {
            if (got[k] != gen((uint64_t)(200 + i), k)) equal = 0;
            uint64_t last = k + BIG_CHUNK - 1;
            if (last < BIG_ELEMS && got[last] != gen((uint64_t)(200 + i), last)) equal = 0;
        }
        if (got[BIG_ELEMS - 1] != gen((uint64_t)(200 + i), BIG_ELEMS - 1)) equal = 0;
    }
    ok(equal, "every chunk boundary of every streamed tensor holds the value written there");
    ok(aligned, "streamed tensors are aligned too");

    int idx = gguf_find_tensor(gf, "moment.half");
    float *half = idx >= 0 ? gguf_dequant(gf, idx) : NULL;
    int half_ok = (half != NULL);
    double worst = 0.0;
    for (uint64_t k = 0; k < 100000 && half_ok; k++) {
        float want = gen(300, k);
        double rel = fabs((double)half[k] - (double)want) / f16_budget(want);
        if (rel > worst) worst = rel;
        if (rel > 1.0) half_ok = 0;
    }
    ok(half_ok, "the streamed F16 tensor is inside its half-ULP budget across 100000 values — worst %.3f of it",
       worst);
    free(half);
    gguf_close(gf);
    remove(path);
}

/* ── main ────────────────────────────────────────────────────────────────────── */

static int rss_mode(const char *which) {
    const char *path = scratch(strcmp(which, "rss19") == 0 ? "nt_gw_rss19.gguf" : "nt_gw_rss91.gguf");
    long before = vm_hwm_kb();
    int rc = strcmp(which, "rss19") == 0 ? stage4_write(path, 0) : big_write(path, 0);
    long after = vm_hwm_kb();
    long sz = file_size(path);
    ok(rc == 0 && sz > 0, "%s: wrote %ld bytes", which, sz);
    if (after < 0)
        printf("  VmHWM unavailable on this system\n");
    else
        printf("  peak RSS: %ld kB after the write (%ld kB before it), file %.1f MB — "
               "the writer holds the caller's buffer and nothing else\n",
               after, before, (double)sz / 1048576.0);
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    /* Kept on request: the file is the artifact another implementation has to agree with,
     * and llama.cpp's reader is the one that matters. */
    if (getenv("NT_TEST_KEEP")) printf("  kept: %s\n", path); else remove(path);
    return n_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && (strcmp(argv[1], "rss19") == 0 || strcmp(argv[1], "rss91") == 0))
        return rss_mode(argv[1]);

    printf("gguf writer — everything written here is read back by gguf.c's own reader\n\n");
    check_metadata();
    check_alignment_odd();
    check_f16_rounding();
    check_packed_blob();
    check_stage4();
    check_big();
    check_refusals();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
