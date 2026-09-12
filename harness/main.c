/* main.c — notorch, the harness.
 *
 *   notorch model.gguf                    chat in the terminal
 *   notorch model.gguf "prompt"           one shot
 *   notorch model.gguf "prompt" 64 0.8    tokens, temperature
 *
 * One rule about output: stdout carries what the model said, stderr carries
 * everything else — the banner, the shape of the model, the prompt you typed,
 * timings, the profile. A run redirected to a file is text, not a transcript
 * of the tool.
 *
 * Architectures are a table, and the table lives in archs.c so that a body can
 * link it without linking this file. Adding a family is adding a file next to
 * arch_llama.c and one line to nt_archs. */
#include "harness/archs.h"
#include "harness/logo.h"
#include "examples/bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Everything a turn needs, so one-shot and chat run the same code. */
typedef struct {
    const nt_arch *arch;
    void *model;
    kv_cache *kv;
    bpe_tokenizer *tok;
    int eos, vocab, max_seq;
    float *logits;
} session;

/* GGUF-embedded BPE where the file has one; bytes where it does not, which is
 * how the char-level models in this tree are read. */
/* A chat template, as ids rather than as text.
 *
 * Some families were trained with the prompt wrapped in tokens of their own —
 * a marker to open, one around the user's turn, one to hand over to the model —
 * and fed a bare prompt they do not fail, they drift. Janus v4 answers "The
 * Method of the Method of the Method of" without its wrapping and "I sense the
 * resonance of the field: the field" with it, on the same weights and the same
 * question.
 *
 * The wrapping is a sequence of ids, and deliberately not a sequence of strings.
 * The ids are what the model was trained on; the strings that spell them may not
 * even be recoverable — Janus has nine special ids above what its merge list
 * reconstructs, and five of them are named anywhere at all. Ids sidestep that
 * entirely, and they sidestep a template language with them.
 *
 *   NT_CHAT="<before>|<after>|<stop>"   each a comma-separated id list
 *   NT_CHAT="32759,32760|32761,32762|32763"
 *
 * Fields may be empty. A file that carries its own wrapping needs none of this:
 * notorch.chat.{before,after,stop} are read from the GGUF when NT_CHAT is unset,
 * and tools/gguf_add_tokenizer --chat writes them. The environment still wins,
 * because trying a different wrapping is how the right one gets found. */
#define NT_CHAT_MAX 32

typedef struct {
    int before[NT_CHAT_MAX], n_before;
    int after[NT_CHAT_MAX],  n_after;
    int stop[NT_CHAT_MAX],   n_stop;
    int active;
} chat_wrap;

static int parse_ids(const char *s, int *out, int max) {
    int n = 0;
    for (const char *p = s; *p && *p != '|' && n < max; ) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p || *p == '|') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

static int chat_from_file(const char *path, const char *key, int *out, int cap) {
    int n = 0;
    int32_t *v = gguf_read_i32_array(path, key, &n);
    if (!v) return 0;
    if (n > cap) n = cap;
    for (int i = 0; i < n; i++) out[i] = (int)v[i];
    free(v);
    return n;
}

static void chat_wrap_init(chat_wrap *w, int vocab, const char *path) {
    memset(w, 0, sizeof(*w));
    const char *spec = getenv("NT_CHAT");
    const char *src;
    if (spec && *spec) {
        const char *a = spec;
        const char *b = strchr(a, '|');
        const char *c = b ? strchr(b + 1, '|') : NULL;
        w->n_before = parse_ids(a, w->before, NT_CHAT_MAX);
        if (b) w->n_after = parse_ids(b + 1, w->after, NT_CHAT_MAX);
        if (c) w->n_stop  = parse_ids(c + 1, w->stop,  NT_CHAT_MAX);
        src = "NT_CHAT";
    } else {
        /* A family that was trained with a wrapping can carry it, and then
         * nobody has to know the ids. The environment still wins, because
         * trying a different wrapping is how the right one gets found. */
        w->n_before = chat_from_file(path, "notorch.chat.before", w->before, NT_CHAT_MAX);
        w->n_after  = chat_from_file(path, "notorch.chat.after",  w->after,  NT_CHAT_MAX);
        w->n_stop   = chat_from_file(path, "notorch.chat.stop",   w->stop,   NT_CHAT_MAX);
        src = "file";
    }
    /* An id outside the vocabulary would index a row the model does not have,
     * so it is refused here rather than read as garbage in the embedding. */
    for (int i = 0; i < w->n_before; i++)
        if (w->before[i] < 0 || w->before[i] >= vocab) { w->n_before = 0; break; }
    for (int i = 0; i < w->n_after; i++)
        if (w->after[i] < 0 || w->after[i] >= vocab) { w->n_after = 0; break; }
    w->active = (w->n_before || w->n_after || w->n_stop);
    if (w->active)
        fprintf(stderr, "chat: %d ids before, %d after, %d stop (%s)\n",
                w->n_before, w->n_after, w->n_stop, src);
}

/* Process-wide because it is read once from the environment and never varies
 * per turn — passing it through every signature would say otherwise. */
static chat_wrap g_chat;

static int chat_is_stop(int id) {
    for (int i = 0; i < g_chat.n_stop; i++) if (g_chat.stop[i] == id) return 1;
    return 0;
}

static int encode_prompt(const session *s, const char *text, int *tokens, int cap) {
    /* NT_TOKENS bypasses the tokenizer with a comma-separated list of ids. Bringing a new
     * family up has two independent failure modes — the tokenizer disagrees, or the forward
     * disagrees — and debugging them together is debugging neither. With the reference's own
     * ids in hand the forward can be compared on its own terms. */
    const char *raw = getenv("NT_TOKENS");
    if (raw && *raw) {
        int n = 0;
        for (const char *p = raw; *p && n < cap; ) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            tokens[n++] = (int)strtol(p, (char **)&p, 10);
        }
        fprintf(stderr, "tokens: %d supplied through NT_TOKENS\n", n);
        return n;
    }
    /* The wrapping goes around whatever the tokenizer produces, so the family's
     * own opening marker sits ahead of the text and the hand-over to the model
     * behind it — the order the weights were trained to read. */
    int n = 0;
    for (int i = 0; i < g_chat.n_before && n < cap; i++) tokens[n++] = g_chat.before[i];

    if (s->tok) n += bpe_encode(s->tok, text, tokens + n, cap - n);
    else {
        if (!g_chat.active && n < cap) tokens[n++] = 1;   /* byte-level BOS */
        for (int i = 0; text[i] && n < cap; i++) tokens[n++] = (unsigned char)text[i];
    }

    for (int i = 0; i < g_chat.n_after && n < cap; i++) tokens[n++] = g_chat.after[i];
    return n;
}

static void emit(const session *s, int id) {
    char piece[256];
    if (s->tok) {
        bpe_decode_token(s->tok, id, piece, sizeof(piece));
        fputs(piece, stdout);
    } else if (id >= 32 && id < 127) putchar((char)id);
    else if (id == 10) putchar('\n');
    else printf("[%d]", id);
    fflush(stdout);
}

/* How much of this process is actually in memory, and how much the machine has left.
 *
 * A model larger than comfort on a phone makes every timing a statement about page residency
 * rather than about arithmetic: the same binary on the same file measured 6.7 t/s and 18.8 on
 * one afternoon, the difference being whether 1.2 GB happened to be free. A benchmark that
 * does not say which of those it caught is not reporting a speed. Both numbers come from the
 * kernel; where it does not offer them, the line is omitted rather than guessed. */
static void residency(const char *when) {
    long rss = -1, avail = -1, memfree = -1;
    char line[256];
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (f) {
        while (fgets(line, sizeof(line), f))
            if (sscanf(line, "Rss: %ld kB", &rss) == 1) break;
        fclose(f);
    }
    /* MemAvailable rather than MemFree: most of what a phone can hand a process is page cache
     * it would drop on request, and MemFree counts none of that. Reading MemFree here would
     * report a machine with nothing left at the exact moment it has room — which is the
     * question this line exists to answer. MemFree is the fallback for a kernel too old to
     * publish the better number. */
    f = fopen("/proc/meminfo", "r");
    if (f) {
        while (fgets(line, sizeof(line), f)) {
            if (avail < 0) sscanf(line, "MemAvailable: %ld kB", &avail);
            if (memfree < 0) sscanf(line, "MemFree: %ld kB", &memfree);
            if (avail >= 0 && memfree >= 0) break;
        }
        fclose(f);
    }
    long room = avail >= 0 ? avail : memfree;
    if (rss < 0 && room < 0) return;
    fprintf(stderr, "  [%s]", when);
    if (rss >= 0) fprintf(stderr, " resident %.2f GiB", rss / 1048576.0);
    if (room >= 0) fprintf(stderr, " | machine can give %.2f GiB", room / 1048576.0);
    fputc('\n', stderr);
}

/* Prompt in, text out, cache advanced. Returns the position after the last
 * token written, so a chat turn can hand it to the next one.
 *
 * Prefill is chunked rather than one call for the whole prompt: the batch
 * buffers are n × FFN floats, and a chunk of 32 is where the weight traffic is
 * already amortized while the working set still fits the caches. The KV cache
 * carries context across chunks, so a chunk sees every position before it
 * exactly as a single pass would. */
static int run_turn(session *s, const int *tokens, int n_tok, int pos0,
                    int max_tokens, float temp, int show_stats) {
    double gen0 = now_ms();
    for (int i = 0; i < n_tok; i += NT_PREFILL_CHUNK) {
        int cn = n_tok - i; if (cn > NT_PREFILL_CHUNK) cn = NT_PREFILL_CHUNK;
        int rc = s->arch->forward(s->model, s->kv, tokens + i, cn, pos0 + i,
                                  (i + cn == n_tok) ? s->logits : NULL);
        if (rc != NT_OK) {
            fprintf(stderr, "\nprefill refused at position %d: %s\n", pos0 + i, nt_strerror(rc));
            putchar(10); fflush(stdout);
            return 0;
        }
    }
    double prefill_ms = now_ms() - gen0;
    pf_report("prefill", prefill_ms);
    pf_reset();

    int pos = pos0 + n_tok, gen = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(s->logits, s->vocab, temp);
        if (chat_is_stop(next)) break;
        if (s->tok ? (next == s->eos || bpe_is_eog(s->tok, next)) : (!g_chat.active && next <= 2)) break;
        emit(s, next);
        gen++;
        if (pos >= s->max_seq - 1) break;
        int rc = s->arch->forward(s->model, s->kv, &next, 1, pos, s->logits);
        if (rc != NT_OK) {
            fprintf(stderr, "\ndecode stopped at position %d: %s\n", pos, nt_strerror(rc));
            break;
        }
        pos++;
    }
    putchar('\n');
    fflush(stdout);

    double total_ms = now_ms() - gen0;
    if (show_stats)
        fprintf(stderr, "\n── prefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s) ──\n",
                n_tok, prefill_ms, n_tok * 1000.0 / prefill_ms,
                gen, total_ms - prefill_ms,
                gen > 0 ? gen * 1000.0 / (total_ms - prefill_ms) : 0);
    pf_report("decode", total_ms - prefill_ms);
    return pos;
}

/* Chat: the cache is the conversation. Each turn appends to it, so the model
 * sees everything said so far without re-reading a transcript. /reset drops
 * it, /exit and end-of-input leave. When the context fills, say so rather than
 * silently answering from a truncated past. */
static void chat(session *s, int max_tokens, float temp) {
    int *tokens = (int*)malloc((size_t)s->max_seq * sizeof(int));
    if (!tokens) return;
    char line[4096];
    int pos = 0;
    fprintf(stderr, "chat — /reset clears the context, /exit leaves (ctx %d)\n\n", s->max_seq);
    for (;;) {
        fprintf(stderr, "> ");
        fflush(stderr);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = '\0';
        if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) break;
        if (strcmp(line, "/reset") == 0) {
            memset(s->kv->k, 0, (size_t)s->kv->n_layers * s->kv->max_seq * s->kv->kv_dim * sizeof(float));
            memset(s->kv->v, 0, (size_t)s->kv->n_layers * s->kv->max_seq * s->kv->kv_dim * sizeof(float));
            pos = 0;
            fprintf(stderr, "  (context cleared)\n");
            continue;
        }
        if (!line[0]) continue;

        int room = s->max_seq - pos - max_tokens - 1;
        if (room <= 0) {
            fprintf(stderr, "  (context full at %d tokens — /reset to start over)\n", pos);
            continue;
        }
        int n = encode_prompt(s, line, tokens, room);
        if (n <= 0) continue;
        pos = run_turn(s, tokens, n, pos, max_tokens, temp, 0);
    }
    free(tokens);
}

static void usage(const char *self) {
    fprintf(stderr,
        "usage: %s [-q] [-n tokens] [-t temp] [-r runs] <model.gguf> [prompt] [max_tokens] [temp]\n"
        "  no prompt        chat in the terminal\n"
        "  -q               no banner\n"
        "  -n, -t           tokens and temperature, for chat as well as one shot\n"
        "  NT_CTX=N         context length for chat (default 2048)\n"
        "  NT_PROFILE=1     per-section timings\n", self);
}

/* BLAS gets one thread, and the reason is that it has its own pool.
 *
 * Everything heavy here runs through notorch's matvec pool, whose workers spin on a
 * generation counter before they sleep. OpenBLAS ships a second pool that spins the same
 * way, and on four cores the two of them take turns evicting each other: a Gemma-4 decode
 * that reaches 9.5 t/s with OPENBLAS_NUM_THREADS=1 does 7.5 with the default. The library is
 * used here for one f32 projection per token; it does not need four cores for that, and it
 * certainly does not need them while the other pool is trying to stream weights.
 *
 * Weak symbol so a build against Accelerate or a BLAS without this call links unchanged. */
#if defined(USE_BLAS) && !defined(ACCELERATE)
extern void openblas_set_num_threads(int) __attribute__((weak));
#endif
static void blas_single_thread(void) {
#if defined(USE_BLAS) && !defined(ACCELERATE)
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif
}

int main(int argc, char **argv) {
    blas_single_thread();
    int quiet = 0, ai = 1;
    int flag_n = -1; float flag_t = -1.0f;
    /* The positional form is what examples/infer_llama.c takes and what the
     * parity gate drives; the flags are for chat, which has no prompt to hang
     * positional arguments behind. */
    int tokenize_only = 0, repeats = 1;
    while (ai < argc && argv[ai][0] == '-' && argv[ai][1] && !argv[ai][2]) {
        char f = argv[ai][1];
        if (f == 'q') { quiet = 1; ai++; continue; }
        /* -r runs the same prompt N times in one process. The load happens once, so what is
         * timed is the model running rather than the model arriving: a fresh process pays for
         * faulting in gigabytes of weights, and the reference benchmark this is compared
         * against does not, which made every such comparison a comparison of two different
         * things. Later iterations are the ones to read. */
        /* A flag that wants a value and has none used to fall out of this loop and become the
         * model path, so `notorch -r` tried to open "-r" as a GGUF and complained about that
         * instead of about the missing count. The same held for -n and -t before it. */
        if (f == 'r' || f == 'n' || f == 't') {
            if (ai + 1 >= argc) {
                fprintf(stderr, "notorch: -%c needs a value\n", f);
                usage(argv[0]);
                return 1;
            }
            if (f == 'r') repeats = atoi(argv[ai + 1]);
            else if (f == 'n') flag_n = atoi(argv[ai + 1]);
            else flag_t = (float)atof(argv[ai + 1]);
            ai += 2; continue;
        }
        /* -T prints the ids and stops. A family arrives with two ways to be wrong and this
         * separates them: the tokenizer can be diffed against another implementation without
         * loading a single weight, and the forward can be fed ids through NT_TOKENS. */
        if (f == 'T') { tokenize_only = 1; quiet = 1; ai++; continue; }
        break;
    }
    if (ai >= argc) { nt_logo(quiet); usage(argv[0]); return 1; }

    nt_logo(quiet);
    const char *path = argv[ai];

    if (tokenize_only) {
        bpe_tokenizer *tok = bpe_load(path);
        if (!tok) { fprintf(stderr, "notorch: no tokenizer in %s\n", path); return 1; }
        const char *text = (ai + 1 < argc) ? argv[ai + 1] : "";
        int ids[8192];
        int n = bpe_encode(tok, text, ids, (int)(sizeof(ids) / sizeof(ids[0])));
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
        if (n == 0) printf("\n");
        /* On stderr, so the ids on stdout stay a bare list: whether this file asked for an
         * opening token. A file that does not declare one is where we part from the
         * reference on purpose, and a gate comparing the two needs to be told which case
         * it is looking at rather than guessing from a length difference. */
        if (bpe_bos_declared(tok))
            fprintf(stderr, "bos: %d (the file asks for %s)\n",
                    bpe_bos_id(tok), bpe_add_bos(tok) ? "one" : "none");
        else
            fprintf(stderr, "bos: undeclared (the file does not say; nothing prepended)\n");
        bpe_free(tok);
        return 0;
    }

    double t0 = now_ms();
    gguf_file *gf = gguf_open(path);
    if (!gf) return 1;

    const nt_arch *arch = nt_pick_arch(gf->arch);
    if (!arch) {
        fprintf(stderr, "notorch: no architecture handles '%s'\n", gf->arch);
        gguf_close(gf);
        return 1;
    }

    nt_dims dims = {0};
    void *model = arch->load(gf, &dims);
    if (!model) { gguf_close(gf); return 1; }
    fprintf(stderr, "loaded in %.0f ms\n", now_ms() - t0);

    session s = { .arch = arch, .model = model, .vocab = dims.vocab, .eos = -1 };
    chat_wrap_init(&g_chat, dims.vocab, path);
    s.tok = bpe_load(path);
    if (s.tok) {
        const gguf_kv *e = gguf_get_kv(gf, "tokenizer.ggml.eos_token_id");
        if (e) s.eos = (int)e->val.u32;
        fprintf(stderr, "tokenizer: GGUF BPE (vocab=%d eos=%d)\n", bpe_n_vocab(s.tok), s.eos);
    } else {
        fprintf(stderr, "tokenizer: byte-level fallback (no GGUF BPE vocab)\n");
    }

    const char *prompt = (ai + 1 < argc) ? argv[ai + 1] : NULL;
    int max_tokens = (ai + 2 < argc) ? atoi(argv[ai + 2]) : 50;
    float temp = (ai + 3 < argc) ? (float)atof(argv[ai + 3]) : 0.8f;
    if (flag_n > 0) max_tokens = flag_n;
    if (flag_t >= 0.0f) temp = flag_t;
    if (max_tokens < 1) max_tokens = 1;

    pf_on = getenv("NT_PROFILE") != NULL;
    s.logits = (float*)calloc(dims.vocab, sizeof(float));
    int rc = 0;

    if (prompt) {
        /* Size the cache to the job. The reference example carried a fixed 256
         * and clamped the prompt into what was left, which turns a long prompt
         * into a quietly truncated one. */
        int cap = 8192;
        int *tokens = (int*)malloc((size_t)cap * sizeof(int));
        int n_tok = tokens ? encode_prompt(&s, prompt, tokens, cap - max_tokens - 1) : 0;
        if (n_tok <= 0) { fprintf(stderr, "notorch: empty prompt\n"); rc = 1; }
        else {
            s.max_seq = n_tok + max_tokens + 1;
            s.kv = kv_new(dims.n_layers, s.max_seq, dims.kv_dim);
            fprintf(stderr, "\nprompt: \"%s\" (%d tokens, temp=%.2f)\n", prompt, n_tok, temp);
            if (repeats < 1) repeats = 1;
            for (int rep = 0; rep < repeats; rep++) {
                if (repeats > 1)
                    fprintf(stderr, "\n── run %d of %d ──\n", rep + 1, repeats);
                /* Printed for a single run too. One timing without it says as little as the
                 * first of six does, and this is the line that tells them apart. */
                residency(rep == 0 ? "before the first run" : "before this run");
                /* The cache is written from position zero every time and attention reads only
                 * up to the current position, so what an earlier run left behind is never
                 * looked at. Reallocating it would only add a page-fault storm to the thing
                 * being measured. */
                fputs(prompt, stdout);
                fflush(stdout);
                run_turn(&s, tokens, n_tok, 0, max_tokens, temp, 1);
            }
        }
        free(tokens);
    } else {
        const char *ctx = getenv("NT_CTX");
        s.max_seq = ctx ? atoi(ctx) : 2048;
        if (s.max_seq < max_tokens + 2) s.max_seq = max_tokens + 2;
        s.kv = kv_new(dims.n_layers, s.max_seq, dims.kv_dim);
        chat(&s, max_tokens, temp);
    }

    if (s.tok) bpe_free(s.tok);
    free(s.logits); kv_free(s.kv); arch->free(model); gguf_close(gf);
    return rc;
}
