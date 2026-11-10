/* test_lfm3_omni.c — the LFM3-Omni towers on a file written here, so make test runs them.
 *
 * A tiny omni GGUF (every tower, widths of 4 to 32) is written through the library's own
 * writer. The honest file must load and every tower must return finite rows of the right
 * count. Then each crafted variant breaks one shape or one key the forward sizes a buffer
 * or an index from, and the loader must refuse it by name: a crafted file dies at load,
 * not inside a matmul. The arithmetic itself is measured against the official modules on
 * the real weights (tools/lfm3_omni_probe, tools/lfm3_omni_compare). */
#include "harness/lfm3_omni.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failed;
#define CHECK(c, s) do { checks++; if (!(c)) { fprintf(stderr, "FAIL: %s\n", s); failed++; } } while (0)

enum { WD = 16, VW = 8, VH = 2, VFF = 12, MMH = 20, AC = 4, NMEL = 16, NFFT = 16, AW = 8, AFF = 12,
       NC = 3, CS = 5, DW = 8, DFF = 12, CW = 8, CFF = 12, CNFFT = 8, VOCAB = 128 };

typedef struct {
    int v_rows, v_kv3, a_ffn, c_ck, lin_mid, hop_gt, hop_half, d_heads3, d_heads_huge;
    int tok_out, sched_over, sched_neg, tok_alias;
} breaks;

static uint32_t rng;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) / 16777216.0f) - 0.5f; }

typedef struct { char name[96]; uint32_t nd; uint64_t shape[4]; float base, scale; } T;
static T ts[256];
static int nts;
static void add(const char *name, uint32_t nd, uint64_t a, uint64_t b, uint64_t c, uint64_t d, float base, float scale) {
    T *t = &ts[nts++];
    snprintf(t->name, sizeof t->name, "%s", name);
    t->nd = nd; t->shape[0] = a; t->shape[1] = b; t->shape[2] = c; t->shape[3] = d;
    t->base = base; t->scale = scale;
}
static void lin(const char *name, uint64_t in, uint64_t out) { add(name, 2, in, out, 0, 0, 0, 0.6f); }
static void nrm(const char *name, uint64_t n) { add(name, 1, n, 0, 0, 0, 1, 0.1f); }

static void block(const char *p, int l, int width, int ffn, int conv, int ck, int heads, int kvh, int hd, int gate, int sandwich) {
    char n[96];
#define B(x) (snprintf(n, sizeof n, "%s.blk.%d.%s", p, l, x), n)
    nrm(B("ln1.weight"), width); nrm(B("ln2.weight"), width);
    if (sandwich) { nrm(B("attn_post_norm.weight"), width); nrm(B("ffn_post_norm.weight"), width); }
    lin(B("ffn_gate.weight"), width, ffn); lin(B("ffn_up.weight"), width, ffn); lin(B("ffn_down.weight"), ffn, width);
    if (conv) {
        add(B("conv_dw.weight"), 2, ck, width, 0, 0, 0, 0.6f);
        lin(B("conv_pw1.weight"), width, (uint64_t)3 * width); lin(B("conv_pw2.weight"), width, width);
    } else {
        lin(B("attn_q.weight"), width, (uint64_t)heads * hd); lin(B("attn_k.weight"), width, (uint64_t)kvh * hd);
        lin(B("attn_v.weight"), width, (uint64_t)kvh * hd); lin(B("attn_out.weight"), (uint64_t)heads * hd, width);
        nrm(B("attn_q_norm.weight"), hd); nrm(B("attn_k_norm.weight"), hd);
        if (gate) lin(B("attn_gate.weight"), width, heads);
    }
#undef B
}

static int write_file(const char *path, breaks b) {
    nts = 0; rng = 4242;
    char n[96];
    /* vision: a conv layer, then an attention layer */
    add("v.patch_embd.weight", 4, 16, 16, 3, VW, 0, 0.05f);
    add("v.patch_embd.weight.1", 4, 16, 16, 3, VW, 0, 0.05f);
    nrm("v.embedding_norm.weight", VW);
    nrm("v.blk.0.ln1.weight", VW); nrm("v.blk.0.ln2.weight", VW);
    lin("v.blk.0.ffn_gate.weight", VW, VFF); lin("v.blk.0.ffn_up.weight", VW, VFF); lin("v.blk.0.ffn_down.weight", VFF, VW);
    lin("v.blk.0.conv2d_in_proj.weight", VW, 3 * VW); lin("v.blk.0.conv2d_out_proj.weight", VW, VW);
    add("v.blk.0.conv2d_spatial.weight", 4, 3, 3, 1, VW, 0, 0.6f);
    nrm("v.blk.1.ln1.weight", VW); nrm("v.blk.1.ln2.weight", VW);
    lin("v.blk.1.ffn_gate.weight", VW, VFF); lin("v.blk.1.ffn_up.weight", VW, VFF); lin("v.blk.1.ffn_down.weight", VFF, VW);
    lin("v.blk.1.attn_q.weight", VW, VW); lin("v.blk.1.attn_k.weight", VW, 4);
    lin("v.blk.1.attn_v.weight", VW, b.v_rows ? 8 : 4); lin("v.blk.1.attn_out.weight", VW, VW);
    nrm("v.blk.1.attn_q_norm.weight", 4); nrm("v.blk.1.attn_k_norm.weight", 4);
    nrm("mm.input_norm.weight", VW); add("mm.input_norm.bias", 1, VW, 0, 0, 0, 0, 0.1f);
    lin("mm.0.weight", 4 * VW, MMH); add("mm.0.bias", 1, MMH, 0, 0, 0, 0, 0.1f);
    lin("mm.2.weight", MMH, WD); add("mm.2.bias", 1, WD, 0, 0, 0, 0, 0.1f);
    /* audio: downsampler, a conv layer, an attention layer, projector */
    const char *cv[5] = {"a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6"};
    for (int i = 0; i < 5; i++) {
        snprintf(n, sizeof n, "%s.weight", cv[i]);
        if (i == 2 || i == 4) add(n, 4, 1, 1, AC, (i == 2 && b.lin_mid) ? AC + 1 : AC, 0, 0.6f);
        else add(n, 4, 3, 3, 1, AC, 0, 0.6f);
        snprintf(n, sizeof n, "%s.bias", cv[i]);
        add(n, 3, 1, 1, AC, 0, 0, 0.1f);
    }
    add("a.mel_filters", 2, NFFT / 2 + 1, NMEL, 0, 0, 0.5f, 0.5f);
    lin("a.pre_encode.out.weight", AC * NMEL / 8, AW); add("a.pre_encode.out.bias", 1, AW, 0, 0, 0, 0, 0.1f);
    block("a", 0, AW, AFF, 1, 3, 0, 0, 0, 0, 0);
    block("a", 1, AW, b.a_ffn ? AFF + 4 : AFF, 0, 0, 2, 1, 4, 0, 0);
    nrm("a.post_ln.weight", AW);
    nrm("mm.a.mlp.0.weight", AW); add("mm.a.mlp.0.bias", 1, AW, 0, 0, 0, 0, 0.1f);
    lin("mm.a.mlp.1.weight", AW, WD); add("mm.a.mlp.1.bias", 1, WD, 0, 0, 0, 0, 0.1f);
    lin("mm.a.mlp.3.weight", WD, WD); add("mm.a.mlp.3.bias", 1, WD, 0, 0, 0, 0, 0.1f);
    /* depth decoder and the codebook embedding */
    lin("a.gen.code.proj_in.weight", WD, NC * DW); add("a.gen.code.proj_in.bias", 1, NC * DW, 0, 0, 0, 0, 0.1f);
    add("a.gen.code.mask_embd", 1, DW, 0, 0, 0, 0, 0.5f);
    add("a.gen.code.embd.weight", 3, DW, CS, NC, 0, 0, 0.6f);
    add("a.gen.code.head_norm.weight", 2, DW, NC, 0, 0, 1, 0.1f);
    nrm("a.gen.code.output_norm.weight", DW);
    /* d_heads3: three heads of 4 whose own projections agree (q 12 rows, out 12 -> 8), so
     * only heads x head width against the stack width can refuse it */
    block("a.gen.code", 0, DW, DFF, 0, 0, b.d_heads3 ? 3 : 2, 1, 4, 1, 0);
    lin("a.gen.code.out_embd.weight", WD, NC * CS);
    /* codec: FSQ table of 2x2 levels, two conv layers and an attention layer, the head */
    add("a.gen.wav.quant.first.codebook.weight", 2, 2, CS - 1, 0, 0, 0, 1);
    add("a.gen.wav.quant.first.out_proj.weight", 2, NC * 2, 4, 0, 0, 0, 0.6f);
    add("a.gen.wav.quant.first.out_proj.bias", 1, 4, 0, 0, 0, 0, 0.1f);
    lin("a.gen.wav.tfm.in_proj.weight", 4, CW); add("a.gen.wav.tfm.in_proj.bias", 1, CW, 0, 0, 0, 0, 0.1f);
    block("a.gen.wav.tfm", 0, CW, CFF, 1, 3, 0, 0, 0, 0, 1);
    block("a.gen.wav.tfm", 1, CW, CFF, 1, b.c_ck ? 5 : 3, 0, 0, 0, 0, 1);
    block("a.gen.wav.tfm", 2, CW, CFF, 0, 0, 2, 1, 4, 1, 1);
    nrm("a.gen.wav.tfm.output_norm.weight", CW);
    lin("a.gen.wav.tfm.out_proj.weight", CW, 2 * (CNFFT / 2 + 1)); add("a.gen.wav.tfm.out_proj.bias", 1, 2 * (CNFFT / 2 + 1), 0, 0, 0, 0, 0.1f);
    add("a.gen.wav.istft.window", 1, CNFFT, 0, 0, 0, 0.6f, 0.2f);

    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    gguf_write_kv_str(w, "general.architecture", "lfm3");
    gguf_write_kv_u32(w, "lfm3.embedding_length", WD);
    const char *tok[] = {"image", "video", "audio", "audio_out", "audio_start", "mixed_start", "text_end",
                         "media_start", "media_end", "im_start", "im_end"};
    for (int i = 0; i < 11; i++) {
        uint32_t id = 100 + i;
        if (i == 0 && b.tok_out) id = VOCAB + 5;               /* image */
        if (i == 1 && b.tok_alias) id = 100;                   /* video == image */
        snprintf(n, sizeof n, "lfm3_omni.token.%s", tok[i]);
        gguf_write_kv_u32(w, n, id);
    }
    gguf_write_kv_u32(w, "lfm3_omni.vision.layers", 2);
    gguf_write_kv_u32(w, "lfm3_omni.vision.heads", VH);
    gguf_write_kv_u32(w, "lfm3_omni.vision.kv_heads", b.v_kv3 ? 3 : 1);
    gguf_write_kv_u32(w, "lfm3_omni.vision.patch", 16);
    gguf_write_kv_u32(w, "lfm3_omni.vision.merge", 2);
    gguf_write_kv_u32(w, "lfm3_omni.vision.min_pixels", 1024);
    gguf_write_kv_u32(w, "lfm3_omni.vision.max_pixels", 4096);
    gguf_write_kv_f32(w, "lfm3_omni.vision.theta", 1e6f);
    gguf_write_kv_f32(w, "lfm3_omni.vision.eps", 1e-5f);
    gguf_write_kv_f32(w, "lfm3_omni.vision.projector_eps", 1e-6f);
    gguf_write_kv_u32(w, "lfm3_omni.audio.layers", 2);
    gguf_write_kv_u32(w, "lfm3_omni.audio.heads", 2);
    gguf_write_kv_u32(w, "lfm3_omni.audio.kv_heads", 1);
    gguf_write_kv_u32(w, "lfm3_omni.audio.window", 4);
    gguf_write_kv_f32(w, "lfm3_omni.audio.theta", 1e6f);
    gguf_write_kv_f32(w, "lfm3_omni.audio.eps", 1e-5f);
    gguf_write_kv_u32(w, "lfm3_omni.audio.sample_rate", 16000);
    gguf_write_kv_u32(w, "lfm3_omni.audio.n_fft", NFFT);
    gguf_write_kv_u32(w, "lfm3_omni.audio.hop", 4);
    gguf_write_kv_u32(w, "lfm3_omni.audio.stride", 8);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_floor", -12);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_offset", 4);
    gguf_write_kv_f32(w, "lfm3_omni.audio.log_divisor", 4);
    gguf_write_kv_u32(w, "lfm3_omni.depth.layers", 1);
    gguf_write_kv_u32(w, "lfm3_omni.depth.heads", b.d_heads3 ? 3 : b.d_heads_huge ? 65536 : 2);
    gguf_write_kv_u32(w, "lfm3_omni.depth.kv_heads", 1);
    gguf_write_kv_f32(w, "lfm3_omni.depth.theta", 1e6f);
    gguf_write_kv_f32(w, "lfm3_omni.depth.eps", 1e-5f);
    gguf_write_kv_u32(w, "lfm3_omni.depth.codebooks", NC);
    gguf_write_kv_u32(w, "lfm3_omni.depth.codebook_size", CS);
    gguf_write_kv_u32(w, "lfm3_omni.depth.eos_code", CS - 1);
    if (b.sched_over) gguf_write_kv_i32_array(w, "lfm3_omni.depth.schedule", (const int32_t[]){2, 2}, 2);
    else if (b.sched_neg) gguf_write_kv_i32_array(w, "lfm3_omni.depth.schedule", (const int32_t[]){-1, 4}, 2);
    else gguf_write_kv_i32_array(w, "lfm3_omni.depth.schedule", (const int32_t[]){2, 1}, 2);
    gguf_write_kv_u32(w, "lfm3_omni.gen.interleave_text", 6);
    gguf_write_kv_u32(w, "lfm3_omni.gen.interleave_audio", 12);
    gguf_write_kv_f32(w, "lfm3_omni.gen.audio_temperature", 0.4f);
    gguf_write_kv_f32(w, "lfm3_omni.gen.audio_repetition_penalty", 1.2f);
    gguf_write_kv_u32(w, "lfm3_omni.gen.audio_repetition_window", 64);
    gguf_write_kv_u32(w, "lfm3_omni.codec.layers", 3);
    gguf_write_kv_u32(w, "lfm3_omni.codec.heads", 2);
    gguf_write_kv_u32(w, "lfm3_omni.codec.kv_heads", 1);
    gguf_write_kv_u32(w, "lfm3_omni.codec.window", 3);
    gguf_write_kv_f32(w, "lfm3_omni.codec.theta", 1e6f);
    gguf_write_kv_f32(w, "lfm3_omni.codec.eps", 1e-5f);
    gguf_write_kv_u32(w, "lfm3_omni.codec.frame_stack", 2);
    gguf_write_kv_u32(w, "lfm3_omni.codec.n_fft", CNFFT);
    gguf_write_kv_u32(w, "lfm3_omni.codec.hop", b.hop_gt ? CNFFT + 2 : b.hop_half ? CNFFT / 2 + 1 : 2);
    gguf_write_kv_u32(w, "lfm3_omni.codec.sample_rate", 24000);
    for (int i = 0; i < nts; i++) gguf_write_tensor_decl(w, ts[i].name, ts[i].nd, ts[i].shape, GGUF_TYPE_F32);
    for (int i = 0; i < nts; i++) {
        uint64_t cnt = 1;
        for (uint32_t d = 0; d < ts[i].nd; d++) cnt *= ts[i].shape[d];
        float *v = malloc(cnt * sizeof(float));
        if (!v) { gguf_write_abort(w); return -1; }
        for (uint64_t k = 0; k < cnt; k++) v[k] = ts[i].base + ts[i].scale * rnd();
        if (!strcmp(ts[i].name, "a.gen.wav.quant.first.codebook.weight"))
            for (uint64_t k = 0; k < cnt; k++) v[k] = ((k / 2) >> (k % 2) & 1) ? 1.0f : -1.0f;
        if (gguf_write_tensor_f32(w, ts[i].name, v, cnt)) { free(v); gguf_write_abort(w); return -1; }
        free(v);
    }
    return gguf_write_close(w);
}

static lfm3_omni *open_omni(breaks b, gguf_file **gf) {
    char path[] = "/tmp/notorch-lfm3omni-XXXXXX";
    int fd = mkstemp(path);
    *gf = NULL;
    if (fd < 0) return NULL;
    close(fd);
    lfm3_omni *o = NULL;
    if (write_file(path, b) == 0 && (*gf = gguf_open(path))) o = lfm3_omni_load(*gf, VOCAB);
    unlink(path);
    return o;
}

static int all_finite(const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) if (!isfinite(x[i])) return 0;
    return 1;
}

int main(void) {
    gguf_file *gf;
    lfm3_omni *o = open_omni((breaks){0}, &gf);
    CHECK(o != NULL, "the honest synthetic omni file loads");
    if (o) {
        const lfm3_omni_info *I = lfm3_omni_get_info(o);
        /* vision: one 32x32 frame is one 2x2 block of patches, one row */
        float px[3 * 32 * 32], vout[WD];
        for (size_t i = 0; i < sizeof px / sizeof px[0]; i++) px[i] = rnd();
        const float *pair[1][2] = {{px, px}};
        CHECK(lfm3_omni_vision_encode(o, pair, 1, 32, 32, vout) == 0 && all_finite(vout, WD), "vision returns one finite row");
        CHECK(lfm3_omni_vision_encode(o, pair, 1, 32, 48, vout) != 0, "a width that is not a whole merge block is refused");
        /* audio: 200 samples -> 47 mel frames -> 6 rows */
        float pcm[200], *mel = NULL;
        int frames = 0;
        for (int i = 0; i < 200; i++) pcm[i] = 0.3f * sinf(0.2f * i);
        CHECK(lfm3_omni_audio_mel(o, pcm, 200, &mel, &frames) == 0 && frames == 1 + (200 - NFFT) / 4 + ((200 - NFFT) % 4 != 0),
              "mel frame count follows the extractor's padding");
        float aout[8 * WD];
        int rows = (frames + 7) / 8;
        CHECK(mel && rows <= 8 && lfm3_omni_audio_encode(o, mel, frames, aout) == 0 && all_finite(aout, (size_t)rows * WD),
              "audio returns finite rows");
        free(mel);
        /* depth, feedback embedding, codec */
        float hid[WD], lg[NC * CS], fe[WD], wav[3 * 4];
        int codes[NC] = {1, 3, 0}, frames3[3 * NC] = {0, 1, 2, 3, 2, 1, 0, 0, 3};
        uint8_t mask[NC] = {1, 0, 1};
        for (int i = 0; i < WD; i++) hid[i] = rnd();
        CHECK(lfm3_omni_depth_logits(o, hid, codes, mask, lg) == 0 && all_finite(lg, NC * CS), "depth logits are finite");
        codes[1] = CS;
        CHECK(lfm3_omni_depth_logits(o, hid, codes, mask, lg) != 0, "a revealed code outside the codebook is refused");
        codes[1] = 3;
        CHECK(lfm3_omni_frame_embed(o, codes, fe) == 0 && all_finite(fe, WD), "frame feedback is finite");
        CHECK(I->codec_samples_per_frame == 4 && lfm3_omni_codec_decode(o, frames3, 3, wav) == 0 && all_finite(wav, 12),
              "codec returns frame_stack x hop samples a frame");
        frames3[4] = CS - 1;
        CHECK(lfm3_omni_codec_decode(o, frames3, 3, wav) != 0, "the end-of-audio code is refused by the codec");
        lfm3_omni_free(o);
    }
    gguf_close(gf);

    const struct { breaks b; const char *what; } bad[] = {
        {{.v_rows = 1},   "vision attn_v rows differing from attn_k is refused"},
        {{.v_kv3 = 1},    "vision kv_heads that do not divide heads are refused"},
        {{.a_ffn = 1},    "an audio layer whose FFN differs from layer 0 is refused"},
        {{.c_ck = 1},     "a codec conv layer with another window than the first is refused"},
        {{.lin_mid = 1},  "a downsampler 1x1 convolution that is not channels x channels is refused"},
        {{.hop_gt = 1},   "a codec hop beyond n_fft is refused"},
        {{.hop_half = 1}, "a codec hop beyond n_fft / 2 is refused"},
        {{.d_heads3 = 1}, "a depth stack whose heads x head width is not its width is refused"},
        {{.d_heads_huge = 1}, "a depth stack of 65536 heads is refused before any product is formed"},
        {{.tok_out = 1},    "an image id past the vocabulary is refused"},
        {{.sched_over = 1}, "a MaskGIT schedule revealing more codebooks than exist is refused"},
        {{.sched_neg = 1},  "a MaskGIT schedule with a pass of -1 is refused although it sums right"},
        {{.tok_alias = 1},  "a video id equal to the image id is refused"},
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        lfm3_omni *x = open_omni(bad[i].b, &gf);
        CHECK(gf != NULL && x == NULL, bad[i].what);
        lfm3_omni_free(x);
        gguf_close(gf);
    }
    printf("LFM3-Omni: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
