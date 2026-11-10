/* omni_main.c — notorch_omni: LFM3-Omni-2B with every modality, from one GGUF.
 *
 *   notorch_omni [options] MODEL.gguf ["prompt"]
 *     --system TEXT        system turn ("Perform ASR in English.", "Perform TTS in ...")
 *     --image FILE         an image in the user turn (PNG, JPEG, BMP, PNM); repeatable
 *     --audio FILE.wav     a 16 kHz clip in the user turn; repeatable
 *     --video F1,F2,...    frames of one video sampled at --fps (default 2), in pairs
 *     --mode text|audio|interleaved   force the answer's modality (default: the model picks)
 *     --speech-out FILE    where generated speech goes (24 kHz WAV, default omni_speech.wav)
 *     -n N                 generated slots, a speech frame counting as one (default 256)
 *     -t T                 text temperature (default 0, greedy)
 *     --audio-temp T       speech temperature (default from the file: 0.4)
 *     --seed S             seed of the speech sampler and the codebook order (default 1)
 *     --ids                print the prompt ids on stderr and stop
 *     NT_ECHO_IDS=1        print every generated text id on stderr
 *
 * The prompt is the checkpoint's chat template rendered as ids: <|startoftext|>, an
 * optional system turn, the user turn with each image, video and clip wrapped in
 * <|vision_start|> ... <|vision_end|> and expanded to one placeholder per row, the text,
 * then <|im_start|>assistant\n<think></think>. Placeholders are real ids; the rows the
 * towers made take their place through the family's forward_embd.
 *
 * The answer follows the official generate: text by default; <|audio_start|> switches to
 * speech until a frame carries the end-of-audio code; <|mixed_start|> alternates 6 text
 * tokens and 12 speech frames until <|text_end|>, the remaining speech ending with an
 * end-of-audio frame. A frame is twelve codes the depth decoder reveals in a seeded random
 * order over the passes of the MaskGIT schedule. stdout is the text; speech goes to the
 * WAV; everything else to stderr. */
#include "harness/archs.h"
#include "harness/tokenizer.h"
#include "harness/lfm3_omni.h"
#include "harness/lfm3_omni_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MEDIA 16

typedef struct { int n; int *ids; const float **rows; int cap; } prompt_t;

static int push(prompt_t *p, int id, const float *row) {
    if (p->n == p->cap) {
        int cap = p->cap ? p->cap * 2 : 1024;
        int *ids = realloc(p->ids, (size_t)cap * sizeof(int));
        if (!ids) return -1;
        p->ids = ids;
        const float **rows = realloc(p->rows, (size_t)cap * sizeof(float *));
        if (!rows) return -1;
        p->rows = rows;
        p->cap = cap;
    }
    p->ids[p->n] = id;
    p->rows[p->n] = row;
    p->n++;
    return 0;
}

static int push_text(prompt_t *p, const bpe_tokenizer *tok, const char *text) {
    size_t len = strlen(text);
    int cap = (int)len * 4 + 16, *ids = malloc((size_t)cap * sizeof(int));
    if (!ids) return -1;
    int n = len ? bpe_encode_raw(tok, text, ids, cap) : 0;
    if (len && n <= 0) { free(ids); fprintf(stderr, "omni: the tokenizer refused '%s'\n", text); return -1; }
    for (int i = 0; i < n; i++) if (push(p, ids[i], NULL)) { free(ids); return -1; }
    free(ids);
    return 0;
}

/* seeded draws: 64-bit LCG, top 53 bits as a double in [0, 1) */
static uint64_t g_rng = 1;
static double uniform(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

static int draw(const float *logits, int n, float temp) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    if (temp <= 0) return best;
    double sum = 0;
    double *p = malloc((size_t)n * sizeof(double));
    if (!p) return best;
    for (int i = 0; i < n; i++) { p[i] = exp((double)(logits[i] - logits[best]) / temp); sum += p[i]; }
    double r = uniform() * sum, acc = 0;
    int pick = n - 1;
    for (int i = 0; i < n; i++) { acc += p[i]; if (acc >= r) { pick = i; break; } }
    free(p);
    return pick;
}

/* The loader has checked every special id against the vocabulary; this checks again where
 * the write happens. */
static void forbid(float *logits, int vocab, int id) {
    if (id >= 0 && id < vocab) logits[id] = -INFINITY;
}

/* The penalty of transformers' RepetitionPenaltyLogitsProcessor, once per distinct code. */
static void penalize(float *l, int n, const int *hist, int nh, float pen) {
    if (pen == 1.0f || nh <= 0) return;
    uint8_t *seen = calloc((size_t)n, 1);
    if (!seen) return;
    for (int i = 0; i < nh; i++) {
        int c = hist[i];
        if (c < 0 || c >= n || seen[c]) continue;
        seen[c] = 1;
        l[c] = l[c] < 0 ? l[c] * pen : l[c] / pen;
    }
    free(seen);
}

typedef struct {
    lfm3_omni *o;
    const lfm3_omni_info *I;
    float temp, penalty;
    int window;
    int *hist;          /* [window][n_codebooks] ring of the last non-eos frames */
    int nhist, hpos;
    double ms;
} speech_t;

static int sample_frame(speech_t *s, const float *hidden, int *codes) {
    int NC = s->I->n_codebooks, CS = s->I->codebook_size;
    int order[64];
    uint8_t mask[64];
    float *lg = malloc((size_t)NC * CS * sizeof(float));
    int *col = malloc((size_t)(s->window + 1) * sizeof(int));
    if (!lg || !col) { free(lg); free(col); return -1; }
    double t0 = now_ms();
    for (int k = 0; k < NC; k++) { order[k] = k; codes[k] = 0; mask[k] = 1; }
    for (int k = NC - 1; k > 0; k--) {       /* Fisher-Yates */
        int j = (int)(uniform() * (k + 1));
        int t = order[k]; order[k] = order[j]; order[j] = t;
    }
    int at = 0, rc = 0;
    for (int g = 0; g < s->I->n_schedule && !rc; g++) {
        if (s->I->schedule[g] < 1 || at + s->I->schedule[g] > NC) { rc = -1; break; }
        if (lfm3_omni_depth_logits(s->o, hidden, codes, mask, lg)) { rc = -1; break; }
        for (int q = 0; q < s->I->schedule[g]; q++) {
            int k = order[at + q];
            float *l = lg + (size_t)k * CS;
            for (int i = 0; i < s->nhist; i++) col[i] = s->hist[(size_t)i * NC + k];
            penalize(l, CS, col, s->nhist, s->penalty);
            codes[k] = draw(l, CS, s->temp);
        }
        for (int q = 0; q < s->I->schedule[g]; q++) mask[order[at + q]] = 0;
        at += s->I->schedule[g];
    }
    s->ms += now_ms() - t0;
    free(lg); free(col);
    return rc;
}

static void remember(speech_t *s, const int *codes) {
    int NC = s->I->n_codebooks;
    if (s->window < 1) return;
    if (s->nhist < s->window) { memcpy(s->hist + (size_t)s->nhist * NC, codes, NC * sizeof(int)); s->nhist++; return; }
    memmove(s->hist, s->hist + NC, (size_t)(s->window - 1) * NC * sizeof(int));
    memcpy(s->hist + (size_t)(s->window - 1) * NC, codes, NC * sizeof(int));
}

static void usage(const char *self) {
    fprintf(stderr, "usage: %s [--system T] [--image F]... [--audio F.wav]... [--video F1,F2,...] [--fps N]\n"
                    "          [--mode text|audio|interleaved] [--speech-out F.wav] [-n N] [-t T] [--audio-temp T]\n"
                    "          [--seed S] [--ids] MODEL.gguf [prompt]\n", self);
}

int main(int argc, char **argv) {
    const char *system = NULL, *images[MAX_MEDIA], *audios[MAX_MEDIA], *video = NULL, *mode_s = NULL;
    const char *speech_out = "omni_speech.wav";
    int n_img = 0, n_aud = 0, max_new = 256, print_ids = 0;
    float temp = 0, audio_temp = -1, fps = 2;
    int ai = 1;
    while (ai < argc && argv[ai][0] == '-') {
        const char *a = argv[ai];
        int has = ai + 1 < argc;
        if (!strcmp(a, "--ids")) { print_ids = 1; ai++; continue; }
        if (!has) { usage(argv[0]); return 2; }
        if (!strcmp(a, "--system")) system = argv[ai + 1];
        else if (!strcmp(a, "--image") && n_img < MAX_MEDIA) images[n_img++] = argv[ai + 1];
        else if (!strcmp(a, "--audio") && n_aud < MAX_MEDIA) audios[n_aud++] = argv[ai + 1];
        else if (!strcmp(a, "--video")) video = argv[ai + 1];
        else if (!strcmp(a, "--fps")) fps = (float)atof(argv[ai + 1]);
        else if (!strcmp(a, "--mode")) mode_s = argv[ai + 1];
        else if (!strcmp(a, "--speech-out")) speech_out = argv[ai + 1];
        else if (!strcmp(a, "-n")) max_new = atoi(argv[ai + 1]);
        else if (!strcmp(a, "-t")) temp = (float)atof(argv[ai + 1]);
        else if (!strcmp(a, "--audio-temp")) audio_temp = (float)atof(argv[ai + 1]);
        else if (!strcmp(a, "--seed")) g_rng = strtoull(argv[ai + 1], NULL, 10);
        else { usage(argv[0]); return 2; }
        ai += 2;
    }
    if (ai >= argc) { usage(argv[0]); return 2; }
    const char *path = argv[ai], *text = ai + 1 < argc ? argv[ai + 1] : "";
    enum { M_TEXT, M_AUDIO, M_MIXED } mode = M_TEXT;
    int forced = 0;
    if (mode_s) {
        forced = 1;
        if (!strcmp(mode_s, "audio")) mode = M_AUDIO;
        else if (!strcmp(mode_s, "interleaved")) mode = M_MIXED;
        else if (strcmp(mode_s, "text")) { usage(argv[0]); return 2; }
    }
    if (max_new < 1 || fps <= 0) { usage(argv[0]); return 2; }

    double t_load = now_ms();
    gguf_file *gf = gguf_open(path);
    if (!gf) return 1;
    const nt_arch *arch = nt_pick_arch(gf->arch);
    if (!arch || !arch->forward_embd) {
        fprintf(stderr, "omni: %s has no family that reads embedding rows (architecture '%s')\n", path, gf->arch);
        gguf_close(gf);
        return 1;
    }
    nt_dims dims = {0};
    void *model = arch->load(gf, &dims);
    lfm3_omni *o = model ? lfm3_omni_load(gf, dims.vocab) : NULL;
    bpe_tokenizer *tok = NULL;
    int eos = -1;
    const char *terr = o ? nt_model_tokenizer_open(gf, path, dims.vocab, NULL, &tok, &eos) : "model did not load";
    if (!o || terr || !tok) {
        fprintf(stderr, "omni: %s\n", terr ? terr : "no tokenizer in the file");
        if (o) lfm3_omni_free(o);
        if (model) arch->free(model);
        gguf_close(gf);
        return 1;
    }
    const lfm3_omni_info *I = lfm3_omni_get_info(o);
    int W = I->width, V = dims.vocab, rc = 1;
    fprintf(stderr, "omni: loaded in %.0f ms\n", now_ms() - t_load);

    /* media first, in the order given: rows the towers make */
    float *media[3 * MAX_MEDIA] = {0};
    int media_rows[3 * MAX_MEDIA] = {0}, n_media = 0;
    enum { K_IMAGE, K_VIDEO, K_AUDIO } media_kind[3 * MAX_MEDIA];   /* token ids enter only at the template */
    double t_vis = 0, t_aud = 0;
    prompt_t p = {0};
    float *hidden = malloc((size_t)W * sizeof(float)), *logits = malloc((size_t)V * sizeof(float));
    float *frame_row = malloc((size_t)W * sizeof(float));
    int *frames = NULL, n_frames = 0, cap_frames = 0;
    kv_cache *kv = NULL;
    speech_t sp = {0};
    if (!hidden || !logits || !frame_row) goto done;
    for (int i = 0; i < n_img; i++) {
        int h, w, H, Wd;
        float *px = NULL;
        uint8_t *rgb = lfm3_image_load(images[i], &h, &w);
        double t0 = now_ms();
        if (!rgb || lfm3_omni_image_prepare(o, rgb, h, w, &px, &H, &Wd)) { free(rgb); goto done; }
        int rows = (H / 32) * (Wd / 32);
        float *out = malloc((size_t)rows * W * sizeof(float));
        const float *pair[1][2] = {{px, px}};
        if (!out || lfm3_omni_vision_encode(o, pair, 1, H, Wd, out)) { free(rgb); free(px); free(out); goto done; }
        t_vis += now_ms() - t0;
        fprintf(stderr, "omni: image %s %dx%d -> %dx%d, %d rows\n", images[i], w, h, Wd, H, rows);
        media[n_media] = out; media_rows[n_media] = rows; media_kind[n_media++] = K_IMAGE;
        free(rgb); free(px);
    }
    int vid_groups = 0, vid_rows_each = 0;
    if (video) {
        /* frames in pairs, each pair one temporal patch; an odd count repeats its last frame */
        char *list = strdup(video), *save = NULL;
        float *planes[768] = {0};
        int nf = 0, H = 0, Wd = 0;
        double t0 = now_ms();
        for (char *f = strtok_r(list, ",", &save); f && nf < 768; f = strtok_r(NULL, ",", &save)) {
            int h, w, h2, w2;
            uint8_t *rgb = lfm3_image_load(f, &h, &w);
            float *px = NULL;
            if (!rgb || lfm3_omni_image_prepare(o, rgb, h, w, &px, &h2, &w2) || (nf && (h2 != H || w2 != Wd))) {
                fprintf(stderr, "omni: video frame %s unreadable or of another size\n", f);
                free(rgb); free(px); nf = -1; break;
            }
            H = h2; Wd = w2;
            planes[nf++] = px;
            free(rgb);
        }
        free(list);
        if (nf > 0) {
            vid_groups = (nf + 1) / 2;
            vid_rows_each = (H / 32) * (Wd / 32);
            const float *(*pairs)[2] = malloc((size_t)vid_groups * sizeof *pairs);
            float *out = malloc((size_t)vid_groups * vid_rows_each * W * sizeof(float));
            for (int g = 0; pairs && g < vid_groups; g++) {
                pairs[g][0] = planes[2 * g];
                pairs[g][1] = planes[2 * g + 1 < nf ? 2 * g + 1 : 2 * g];
            }
            if (pairs && out && !lfm3_omni_vision_encode(o, (const float *const (*)[2])pairs, vid_groups, H, Wd, out)) {
                media[n_media] = out; media_rows[n_media] = vid_groups * vid_rows_each; media_kind[n_media++] = K_VIDEO;
                fprintf(stderr, "omni: video %d frames %dx%d -> %d groups of %d rows\n", nf, Wd, H, vid_groups, vid_rows_each);
            } else { free(out); nf = -1; }
            free(pairs);
        }
        for (int i = 0; i < 768; i++) free(planes[i]);
        t_vis += now_ms() - t0;
        if (nf <= 0) goto done;
    }
    for (int i = 0; i < n_aud; i++) {
        float *pcm = NULL, *mel = NULL;
        int n = 0, rate = 0, fr = 0;
        double t0 = now_ms();
        if (lfm3_wav_read(audios[i], &pcm, &n, &rate) || rate != I->sample_rate) {
            fprintf(stderr, "omni: %s is not a %d Hz WAV (this harness does not resample)\n", audios[i], I->sample_rate);
            free(pcm); goto done;
        }
        if (lfm3_omni_audio_mel(o, pcm, n, &mel, &fr)) { free(pcm); goto done; }
        int rows = (fr + I->stride - 1) / I->stride;
        float *out = malloc((size_t)rows * W * sizeof(float));
        if (!out || lfm3_omni_audio_encode(o, mel, fr, out)) { free(pcm); free(mel); free(out); goto done; }
        t_aud += now_ms() - t0;
        fprintf(stderr, "omni: audio %s %.2f s -> %d rows\n", audios[i], (double)n / rate, rows);
        media[n_media] = out; media_rows[n_media] = rows; media_kind[n_media++] = K_AUDIO;
        free(pcm); free(mel);
    }

    /* the template, as ids */
    int bos = bpe_bos_id(tok) >= 0 ? bpe_bos_id(tok) : 124894;
    int think = bpe_token_id(tok, "<think>"), think_end = bpe_token_id(tok, "</think>");
    if (think < 0 || think_end < 0) { fprintf(stderr, "omni: <think> tokens missing\n"); goto done; }
    int bad = push(&p, bos, NULL);
    if (system) {
        bad |= push(&p, I->tok_im_start, NULL) || push_text(&p, tok, "system\n") || push_text(&p, tok, system) ||
               push(&p, I->tok_im_end, NULL) || push_text(&p, tok, "\n");
    }
    bad |= push(&p, I->tok_im_start, NULL) || push_text(&p, tok, "user\n");
    for (int m = 0; m < n_media && !bad; m++) {
        bad |= push(&p, I->tok_media_start, NULL);
        if (media_kind[m] == K_VIDEO) {
            for (int g = 0; g < vid_groups && !bad; g++) {
                char ts[64];
                double t = (2.0 * g + 0.5) / fps;      /* mean of the pair's frame indices over fps */
                snprintf(ts, sizeof ts, "<%.1f seconds>", t);
                bad |= push_text(&p, tok, ts) || push(&p, I->tok_media_start, NULL);
                for (int r = 0; r < vid_rows_each && !bad; r++)
                    bad |= push(&p, I->tok_video, media[m] + ((size_t)g * vid_rows_each + r) * W);
                bad |= push(&p, I->tok_media_end, NULL);
            }
        } else {
            int id = media_kind[m] == K_IMAGE ? I->tok_image : I->tok_audio;
            for (int r = 0; r < media_rows[m] && !bad; r++) bad |= push(&p, id, media[m] + (size_t)r * W);
        }
        bad |= push(&p, I->tok_media_end, NULL);
    }
    bad |= push_text(&p, tok, text) || push(&p, I->tok_im_end, NULL) || push_text(&p, tok, "\n") ||
           push(&p, I->tok_im_start, NULL) || push_text(&p, tok, "assistant\n") || push(&p, think, NULL) ||
           push(&p, think_end, NULL);
    if (forced && mode == M_AUDIO) bad |= push(&p, I->tok_audio_start, NULL);
    if (forced && mode == M_MIXED) bad |= push(&p, I->tok_mixed_start, NULL);
    if (bad) goto done;
    if (print_ids) {
        for (int i = 0; i < p.n; i++) fprintf(stderr, "%d%s", p.ids[i], i + 1 < p.n ? "," : "\n");
        rc = 0;
        goto done;
    }

    kv = nt_model_kv_new(arch, model, dims, p.n + max_new + 1);
    if (!kv) { fprintf(stderr, "omni: no cache for %d positions\n", p.n + max_new + 1); goto done; }
    double t_pre = now_ms();
    for (int i = 0; i < p.n; i += NT_PREFILL_CHUNK) {
        int cn = p.n - i < NT_PREFILL_CHUNK ? p.n - i : NT_PREFILL_CHUNK;
        int last = i + cn == p.n;
        int r = arch->forward_embd(model, kv, p.ids + i, p.rows + i, cn, i, last ? logits : NULL, last ? hidden : NULL);
        if (r != NT_OK) { fprintf(stderr, "omni: prefill refused at %d: %s\n", i, nt_strerror(r)); goto done; }
    }
    t_pre = now_ms() - t_pre;

    sp.o = o; sp.I = I;
    sp.temp = audio_temp >= 0 ? audio_temp : I->audio_temperature;
    sp.penalty = I->audio_rep_penalty;
    sp.window = I->audio_rep_window;
    sp.hist = malloc((size_t)(sp.window > 0 ? sp.window : 1) * I->n_codebooks * sizeof(int));
    if (!sp.hist) goto done;
    /* a prompt that ends on a mode marker starts in that mode, as the reference infers */
    int lastid = p.ids[p.n - 1];
    if (!forced) mode = lastid == I->tok_audio_start ? M_AUDIO : lastid == I->tok_mixed_start ? M_MIXED : M_TEXT;
    int text_only = forced && mode == M_TEXT, speaking = 0, text_done = 0, slots = I->interleave_text;
    int echo_ids = getenv("NT_ECHO_IDS") != NULL;   /* the generated text ids on stderr, as notorch does */
    int n_text = 0, pos = p.n;
    double t_dec = now_ms();
    for (int step = 0; step < max_new; step++) {
        int codes[64], id = -1, finished = 0;
        const float *row = NULL;
        if (mode == M_AUDIO || (mode == M_MIXED && speaking)) {
            if (sample_frame(&sp, hidden, codes)) goto done;
            int is_eos = 0;
            for (int k = 0; k < I->n_codebooks; k++) is_eos |= codes[k] == I->eos_code;
            if (is_eos) {
                for (int k = 0; k < I->n_codebooks; k++) codes[k] = I->eos_code;
                mode = M_TEXT;
            } else {
                remember(&sp, codes);
                if (n_frames == cap_frames) {
                    cap_frames = cap_frames ? 2 * cap_frames : 64;
                    int *nf = realloc(frames, (size_t)cap_frames * I->n_codebooks * sizeof(int));
                    if (!nf) goto done;
                    frames = nf;
                }
                memcpy(frames + (size_t)n_frames * I->n_codebooks, codes, I->n_codebooks * sizeof(int));
                n_frames++;
            }
            if (lfm3_omni_frame_embed(o, codes, frame_row)) goto done;
            id = I->tok_audio_out;
            row = frame_row;
        } else {
            const int sup[] = {I->tok_audio, I->tok_audio_out, I->tok_image, I->tok_video};
            for (size_t k = 0; k < 4; k++) forbid(logits, V, sup[k]);
            if (text_only) { forbid(logits, V, I->tok_mixed_start); forbid(logits, V, I->tok_audio_start); }
            id = draw(logits, V, temp);
            n_text++;
            if (echo_ids) fprintf(stderr, "[id %d]", id);
            if (id == I->tok_text_end) {
                if (mode == M_MIXED) { text_done = 1; speaking = 1; }
                else finished = 1;
            } else if (id == I->tok_im_end || id == eos) finished = 1;
            else if (mode != M_MIXED && id == I->tok_audio_start) mode = M_AUDIO;
            else if (mode != M_MIXED && id == I->tok_mixed_start) {
                mode = M_MIXED; slots = I->interleave_text; speaking = 0; text_done = 0;
            } else {
                char buf[256];
                int cap = bpe_decode_token_capacity(tok, id);
                if (cap > 0 && cap <= (int)sizeof buf) {
                    int nb = bpe_decode_token(tok, id, buf, cap);
                    fwrite(buf, 1, (size_t)nb, stdout);
                    fflush(stdout);
                }
            }
        }
        if (mode == M_MIXED && !text_done) {
            if (--slots == 0) { speaking = !speaking; slots = speaking ? I->interleave_audio : I->interleave_text; }
        }
        if (finished || step == max_new - 1) break;
        int r = arch->forward_embd(model, kv, &id, &row, 1, pos, logits, hidden);
        if (r != NT_OK) { fprintf(stderr, "\nomni: decode refused at %d: %s\n", pos, nt_strerror(r)); goto done; }
        pos++;
    }
    t_dec = now_ms() - t_dec;
    putchar('\n');
    fflush(stdout);
    int slots_done = n_text + n_frames;
    fprintf(stderr, "omni: vision %.0f ms, audio %.0f ms | prefill %d tok %.0f ms (%.1f t/s) | "
                    "decode %d text + %d frames %.0f ms (%.2f slots/s, depth %.0f ms)\n",
            t_vis, t_aud, p.n, t_pre, p.n * 1000.0 / t_pre, n_text, n_frames, t_dec,
            slots_done * 1000.0 / (t_dec > 0 ? t_dec : 1), sp.ms);
    if (n_frames) {
        size_t ns = (size_t)n_frames * I->codec_samples_per_frame;
        float *pcm = malloc(ns * sizeof(float));
        double t0 = now_ms();
        if (!pcm || lfm3_omni_codec_decode(o, frames, n_frames, pcm) || lfm3_wav_write(speech_out, pcm, (int)ns, I->codec_rate)) {
            free(pcm);
            fprintf(stderr, "omni: speech could not be written\n");
            goto done;
        }
        fprintf(stderr, "omni: speech %d frames -> %.2f s at %d Hz in %.0f ms -> %s\n", n_frames,
                (double)ns / I->codec_rate, I->codec_rate, now_ms() - t0, speech_out);
        free(pcm);
    }
    rc = 0;
done:
    for (int m = 0; m < n_media; m++) free(media[m]);
    free(p.ids); free(p.rows); free(hidden); free(logits); free(frame_row); free(frames); free(sp.hist);
    if (kv) nt_model_kv_free(arch, model, kv);
    bpe_free(tok);
    lfm3_omni_free(o);
    arch->free(model);
    gguf_close(gf);
    return rc;
}
