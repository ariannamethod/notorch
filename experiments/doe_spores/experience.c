/* Resident long-experience driver. The Harness owns every numerical operation.
 * Each wake uses raw tokenizer input, one greedy choice and fresh decoder state.
 * Exact checkpoints are new files; a caller publishes its run receipt last. */
#include "harness/archs.h"
#include "harness/experts.h"
#include "harness/tokenizer.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { INPUT_TOKENS = 512, INPUT_BYTES = 16 * 1024 * 1024, LAST_STEP = 4096 };
enum { AWAKE, DREAM, REPLAY };

static int ordinal(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long n = strtol(text, &end, 10);
    if (errno || end == text || *end || n < 1 || n > LAST_STEP) return -1;
    *value = (int)n;
    return 0;
}

static int utf8(const unsigned char *text, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned v = text[i++], need, min;
        if (v < 0x80) { if (!v) return 0; continue; }
        if (v >= 0xc2 && v <= 0xdf) { need = 1; min = 0x80; v &= 31; }
        else if (v >= 0xe0 && v <= 0xef) { need = 2; min = 0x800; v &= 15; }
        else if (v >= 0xf0 && v <= 0xf4) { need = 3; min = 0x10000; v &= 7; }
        else return 0;
        if (need > n - i) return 0;
        while (need--) {
            unsigned c = text[i++];
            if ((c & 0xc0) != 0x80) return 0;
            v = (v << 6) | (c & 63);
        }
        if (v < min || v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff)) return 0;
    }
    return 1;
}

/* Validate the complete input tape before any output is published. */
static char *input_lines(const char *path, char **lines, int expected) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *text = NULL;
    if (fseek(f, 0, SEEK_END)) goto done;
    long bytes = ftell(f);
    if (bytes < 1 || bytes > INPUT_BYTES || fseek(f, 0, SEEK_SET)) goto done;
    text = malloc((size_t)bytes + 1);
    if (!text || fread(text, 1, (size_t)bytes, f) != (size_t)bytes || fgetc(f) != EOF ||
        ferror(f) || !utf8((const unsigned char *)text, (size_t)bytes)) goto fail;
    text[bytes] = 0;
    int count = 0;
    char *next = text, *limit = text + bytes;
    while (next < limit) {
        if (count == expected) goto fail;
        char *end = memchr(next, '\n', (size_t)(limit - next));
        char *line_end = end ? end : limit;
        if (line_end > next && line_end[-1] == '\r') line_end--;
        *line_end = 0;
        lines[count++] = next;
        next = end ? end + 1 : limit;
    }
    if (count != expected) goto fail;
    if (fclose(f)) { free(text); return NULL; }
    return text;
fail:
    free(text); text = NULL;
done:
    fclose(f);
    return text;
}

static int empty_directory(const char *path) {
    DIR *d = opendir(path);
    if (!d) return 0;
    int empty = 1;
    struct dirent *e;
    errno = 0;
    while ((e = readdir(d))) {
        if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) { empty = 0; break; }
    }
    int error = errno;
    if (closedir(d)) error = 1;
    return empty && !error;
}

static char *output_path(const char *dir, const char *kind, int step) {
    size_t n = strlen(dir);
    if (n > SIZE_MAX - 80) return NULL;
    char *path = malloc(n + 80);
    if (path) {
        if (step) snprintf(path, n + 80, "%s/%s-%d.gguf", dir, kind, step);
        else snprintf(path, n + 80, "%s/%s.gguf", dir, kind);
    }
    return path;
}

static int save_state(nt_experts *p, const char *dir, const char *kind, int step) {
    char *path = output_path(dir, kind, step);
    int rc = path ? nt_experts_save(p, path) : -1;
    if (rc) fprintf(stderr, "experience: could not publish %s at ordinal %d\n", kind, step);
    free(path);
    return rc;
}

static int wake(const nt_arch *arch, void *model, nt_dims dims, const bpe_tokenizer *tok,
                int eos, nt_experts *p, const int *ids, int n, int *choice) {
    kv_cache *kv = nt_model_kv_new(arch, model, dims, n + 2);
    float *logits = malloc((size_t)dims.vocab * sizeof(float));
    int rc = NT_E_MEMORY;
    if (!kv || !logits) goto done;
    for (int pos = 0; pos < n; pos += NT_PREFILL_CHUNK) {
        int count = n - pos < NT_PREFILL_CHUNK ? n - pos : NT_PREFILL_CHUNK;
        rc = arch->doe->forward(model, kv, ids + pos, count, pos,
                               pos + count == n ? logits : NULL,
                               nt_experts_attend, nt_experts_observe, p);
        if (rc) goto done;
    }
    for (int i = 0; i < dims.vocab; i++) if (!isfinite(logits[i])) { rc = NT_E_STATE; goto done; }
    int id = nt_sample_model_token(logits, dims.vocab, tok, 0);
    if (id < 0 || id >= dims.vocab || !isfinite(logits[id])) { rc = NT_E_TOKEN; goto done; }
    nt_experts_choice(p, logits[id], logits[id]);
    fprintf(stderr, "doe choice step=0 id=%d logit=%.9g best=%d best_logit=%.9g\n",
            id, logits[id], id, logits[id]);
    *choice = id;
    /* The ordinary one-shot Harness feeds back its last non-stop choice. */
    if (id != eos && !bpe_is_eog(tok, id))
        rc = arch->doe->forward(model, kv, &id, 1, n, logits,
                               nt_experts_attend, nt_experts_observe, p);
done:
    free(logits);
    nt_model_kv_free(arch, model, kv);
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 11) {
        fputs("usage: doe_experience HOST TOKENIZER|- PARENT EPISODE|- awake|dream|replay "
              "sync|worker START END OUTDIR INPUTS\n", stderr);
        return 1;
    }
    int start, end, mode, worker;
    if (ordinal(argv[7], &start) || ordinal(argv[8], &end) || end < start) return 1;
    mode = !strcmp(argv[5], "awake") ? AWAKE : !strcmp(argv[5], "dream") ? DREAM :
           !strcmp(argv[5], "replay") ? REPLAY : -1;
    worker = !strcmp(argv[6], "sync") ? 0 : !strcmp(argv[6], "worker") ? 1 : -1;
    const char *chat = getenv("NT_CHAT"), *raw = getenv("NT_TOKENS");
    if (mode < 0 || worker < 0 || (chat && *chat && strcmp(chat, "||")) || (raw && *raw)) {
        fputs("experience: choose a mode/worker and raw tokenizer input without chat wrapping\n", stderr);
        return 1;
    }
    const char *outdir = argv[9];
    if (!empty_directory(outdir)) {
        fputs("experience: output directory must already exist and be empty\n", stderr);
        return 1;
    }
    char **lines = calloc((size_t)(end - start + 1), sizeof(*lines));
    char *input = lines ? input_lines(argv[10], lines, end - start + 1) : NULL;
    if (!input) {
        fputs("experience: input must be bounded UTF-8 with exactly one line per ordinal\n", stderr);
        free(lines); return 1;
    }
    gguf_file *host = gguf_open(argv[1]);
    const nt_arch *arch = host ? nt_pick_arch(host->arch) : NULL;
    nt_dims dims = {0};
    void *model = arch && arch->doe ? arch->load(host, &dims) : NULL;
    bpe_tokenizer *tok = NULL;
    int eos = -1, rc = 1;
    nt_experts *p = NULL;
    nt_replay_episode *episode = NULL;
    nt_consolidation *job = NULL;
    if (!model) { fputs("experience: no loaded DoE body\n", stderr); goto done; }
    const char *error = nt_model_tokenizer_open(host, argv[1], dims.vocab,
        strcmp(argv[2], "-") ? argv[2] : NULL, &tok, &eos);
    if (error || !tok) {
        fprintf(stderr, "experience: %s\n", error ? error : "a tokenizer is required"); goto done;
    }
    nt_expert_body body = {model, tok};
    p = nt_experts_open_on_body(host, argv[3], stderr, &body);
    if (!p) { fputs("experience: parent spore refused\n", stderr); goto done; }
    if (strcmp(argv[4], "-")) {
        episode = nt_replay_load(p, argv[4]);
        if (!episode) { fputs("experience: episode refused for this body\n", stderr); goto done; }
    }
    for (int step = start; step <= end; step++) {
        int ids[INPUT_TOKENS + 1];
        int n = bpe_encode(tok, lines[step - start], ids, INPUT_TOKENS + 1), choice = -1;
        if (n < 1 || n > INPUT_TOKENS ||
            (host->ctx_len > 0 && n + 2 > host->ctx_len)) {
            fprintf(stderr, "experience: ordinal %d exceeds the input/cache bound or has no IDs\n", step); goto done;
        }
        if (!episode) {
            episode = nt_replay_capture(p, ids, NULL, n, NULL, 0, &eos, eos >= 0 ? 1 : 0);
            char *path = output_path(outdir, "episode", 0);
            int failed = !episode || !path || nt_replay_save(episode, path);
            free(path);
            if (failed) { fputs("experience: could not publish the captured input\n", stderr); goto done; }
        }
        fprintf(stderr, "experience wake ordinal=%d mode=%s worker=%d\n", step, argv[5], worker);
        int r = wake(arch, model, dims, tok, eos, p, ids, n, &choice);
        if (!r) r = worker ? nt_experts_submit(&p, &job) : nt_experts_finish(p, 1);
        if (!r && job) r = nt_experts_await(&p, &job);
        if (r) { fprintf(stderr, "experience: waking ordinal %d failed: %s\n", step, nt_strerror(r)); goto done; }
        if (step % 8 == 0) {
            if (save_state(p, outdir, "wake", step)) goto done;
            if (mode == DREAM)
                r = worker ? nt_experts_submit_dream(&p, &job, 6, eos) : nt_experts_dream(p, 6, eos, 1);
            else if (mode == REPLAY)
                r = worker ? nt_experts_submit_replay(&p, &job, episode, 6) : nt_experts_replay(p, episode, 6, 1);
            if (!r && job) r = nt_experts_await(&p, &job);
            if (r) { fprintf(stderr, "experience: sleeping ordinal %d failed: %s\n", step, nt_strerror(r)); goto done; }
        }
        if (save_state(p, outdir, "state", step)) goto done;
        if (printf("%d\tstate-%d.gguf\t%d\n", step, step, choice) < 0 || fflush(stdout)) goto done;
    }
    rc = 0;
done:
    if (job && nt_experts_await(&p, &job)) rc = 1;
    nt_replay_free(episode); nt_experts_free(p);
    if (tok) bpe_free(tok);
    if (model) arch->free(model);
    if (host) gguf_close(host);
    free(input); free(lines);
    return rc;
}
