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
#include "harness/tokenizer.h"
#include "harness/logo.h"
#include "harness/experts.h"
#include "examples/bpe.h"
#include <errno.h>
#include <math.h>
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
    nt_experts *experts;
    float *sampling_copy; /* sampling at T>0 consumes its input */
    nt_consolidation *pending;
    int learn, trace, moe_trace, async, dream_tokens;
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

/* Capture the same assembled prompt and stopping rules as the waking turn.
 * EOG is already supplied by the tokenizer in the replay kernel. */
static nt_replay_episode *capture_text(const session *s, const int *ids, int n) {
    int stops[NT_CHAT_MAX + 1], count = 0;
    if (s->eos >= 0 && s->eos < s->vocab) stops[count++] = s->eos;
    for (int i = 0; i < g_chat.n_stop; i++) {
        int id = g_chat.stop[i], seen = 0;
        if (id < 0 || id >= s->vocab) return NULL;
        for (int j = 0; j < count; j++) seen |= stops[j] == id;
        if (!seen) stops[count++] = id;
    }
    return nt_replay_capture(s->experts, ids, NULL, n, NULL, 0, stops, count);
}

static int emit(const session *s, int id) {
    if (s->tok) {
        char local[256];
        int cap = bpe_decode_token_capacity(s->tok, id);
        if (!cap) return 0;
        char *piece = (size_t)cap <= sizeof(local) ? local : malloc((size_t)cap);
        if (!piece) return 0;
        int n = bpe_decode_token(s->tok, id, piece, cap);
        int ok = fwrite(piece, 1, (size_t)n, stdout) == (size_t)n;
        if (piece != local) free(piece);
        if (!ok) return 0;
    } else if (id >= 32 && id < 127) { if (putchar((char)id) == EOF) return 0; }
    else if (id == 10) { if (putchar('\n') == EOF) return 0; }
    else if (printf("[%d]", id) < 0) return 0;
    return fflush(stdout) == 0;
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

static void print_moe_route(void *user, int layer, int pos, int token, int k,
                            const int *ids, const float *weights) {
    const char *event = user;
    FILE *out = stderr;
    flockfile(out);
    fprintf(out, "moe %s layer=%d pos=%d token=%d ids=", event, layer, pos, token);
    for (int i = 0; i < k; i++) fprintf(out, "%s%d", i ? "," : "", ids[i]);
    fputs(" weights=", out);
    for (int i = 0; i < k; i++) fprintf(out, "%s%.9g", i ? "," : "", weights[i]);
    fputc('\n', out);
    funlockfile(out);
}

static int session_forward(session *s, const int *tokens, int n, int pos, float *logits) {
    if (s->moe_trace)
        return s->arch->forward_routes(s->model, s->kv, tokens, n, pos, logits,
                                       s->experts ? nt_experts_attend : NULL,
                                       s->experts ? nt_experts_observe : NULL, s->experts,
                                       print_moe_route, "route");
    if (s->experts)
        return s->arch->doe->forward(s->model, s->kv, tokens, n, pos, logits,
                                      nt_experts_attend, nt_experts_observe, s->experts);
    return s->arch->forward(s->model, s->kv, tokens, n, pos, logits);
}

/* A private episode can follow a waking turn or resume a completed spore. */
static int consolidate(session *s, int wake) {
    if (s->async) {
        if (s->dream_tokens)
            return nt_experts_submit_dream(&s->experts, &s->pending, s->dream_tokens, s->eos);
        return nt_experts_submit(&s->experts, &s->pending);
    }
    if (wake && s->experts) {
        int rc = nt_experts_finish(s->experts, s->learn);
        if (rc) return rc;
    }
    return s->dream_tokens ? nt_experts_dream(s->experts, s->dream_tokens, s->eos, s->learn) : NT_OK;
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
    if (s->pending && nt_experts_await(&s->experts, &s->pending)) {
        fprintf(stderr, "notorch: previous expert consolidation failed\n");
        return 0;
    }
    if (pf_on) pf_reset();
    double gen0 = now_ms();
    for (int i = 0; i < n_tok; i += NT_PREFILL_CHUNK) {
        int cn = n_tok - i; if (cn > NT_PREFILL_CHUNK) cn = NT_PREFILL_CHUNK;
        int rc = session_forward(s, tokens + i, cn, pos0 + i,
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
        float *sampling = s->logits;
        if (s->sampling_copy) {
            memcpy(s->sampling_copy, s->logits, (size_t)s->vocab*sizeof(float));
            sampling = s->sampling_copy;
        }
        int next = nt_sample_model_token(sampling, s->vocab, s->tok, temp);
        if (s->experts || s->trace) {
            int best = nt_sample_model_token(s->logits, s->vocab, s->tok, 0);
            if (s->experts) nt_experts_choice(s->experts, s->logits[best], s->logits[next]);
            if (s->trace) fprintf(stderr, "doe choice step=%d id=%d logit=%.9g best=%d best_logit=%.9g\n",
                                  step, next, s->logits[next], best, s->logits[best]);
        }
        if (getenv("NT_ECHO_IDS")) fprintf(stderr, "[id %d]", next);
        if (chat_is_stop(next)) break;
        if (s->tok ? (next == s->eos || bpe_is_eog(s->tok, next)) : (!g_chat.active && next <= 2)) break;
        if (!emit(s, next)) {
            fprintf(stderr, "\ndecode stopped: could not emit complete token %d\n", next);
            return 0;
        }
        gen++;
        if (pos >= s->max_seq - 1) break;
        int rc = session_forward(s, &next, 1, pos, s->logits);
        if (rc != NT_OK) {
            fprintf(stderr, "\ndecode stopped at position %d: %s\n", pos, nt_strerror(rc));
            return 0;
        }
        pos++;
    }
    if (putchar('\n') == EOF || fflush(stdout)) return 0;

    double total_ms = now_ms() - gen0;
    if (show_stats)
        fprintf(stderr, "\n── prefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s) ──\n",
                n_tok, prefill_ms, n_tok * 1000.0 / prefill_ms,
                gen, total_ms - prefill_ms,
                gen > 0 ? gen * 1000.0 / (total_ms - prefill_ms) : 0);
    pf_report("decode", total_ms - prefill_ms);
    int rc = consolidate(s, 1);
    if (rc != NT_OK) {
        fprintf(stderr, "notorch: expert turn could not be committed\n");
        return 0;
    }
    return pos;
}

/* /reset. A family that owns its cache owns its layout too: its kv_dim can be a
 * sentinel and its state lives beside k and v, so it is dropped and made again
 * through the family's own hooks. A host cache is the plain
 * [n_layers][max_seq][kv_dim] pair and is zeroed in place. */
static int reset_cache(session *s) {
    if (s->arch->kv_new) {
        if (s->arch->kv_free) s->arch->kv_free(s->model, s->kv); else kv_free(s->kv);
        s->kv = s->arch->kv_new(s->model, s->max_seq);
        return s->kv ? 0 : 1;
    }
    memset(s->kv->k, 0, (size_t)s->kv->n_layers * s->kv->max_seq * s->kv->kv_dim * sizeof(float));
    memset(s->kv->v, 0, (size_t)s->kv->n_layers * s->kv->max_seq * s->kv->kv_dim * sizeof(float));
    return 0;
}

/* Chat: the cache is the conversation. Each turn appends to it, so the model
 * sees everything said so far without re-reading a transcript. /reset drops
 * it, /exit and end-of-input leave. When the context fills, say so rather than
 * silently answering from a truncated past. */
static int chat(session *s, int max_tokens, float temp) {
    int *tokens = (int*)malloc((size_t)s->max_seq * sizeof(int));
    if (!tokens || !s->kv) { free(tokens); return 1; }
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
            if (reset_cache(s)) { fprintf(stderr, "  (could not make a new cache; leaving)\n"); break; }
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
        if (!pos) { free(tokens); return 1; }
    }
    free(tokens);
    return ferror(stdin) != 0;
}

static void usage(const char *self) {
    fprintf(stderr,
        "usage: %s [-q] [-n tokens] [-t temp] [-r runs] [--tokenizer file] <model.gguf> [prompt] [max_tokens] [temp]\n"
        "  no prompt        chat in the terminal\n"
        "  --experts        seat a native parliament on a supported CPU decoder\n"
        "  --learn-experts  update elected experts after each turn (requires --experts)\n"
        "  --async-consolidation  consolidate in the background (requires --learn-experts)\n"
        "  --dream N        generate 1..32 private choices after each learned turn\n"
        "  --dream-only N   dream from --spore-in without a user turn (preview unless learning)\n"
        "  --spore-in FILE  restore an exact or portable parliament (requires --experts)\n"
        "  --body-memory FILE  recall this body's saved factors by identity after --spore-in\n"
        "  --spore-only     convert --spore-in to --spore-out before the first turn\n"
        "  --portable-spore include a residual frame in --spore-out for switching bodies\n"
        "  --spore-out FILE write a new GGUF spore on success (requires --experts)\n"
        "  --replay-out NEW.gguf  capture a one-shot prompt and its stop policy\n"
        "  --replay-only FILE  private replay from --spore-in; -n 1..32 (default 6)\n"
        "  --expert-trace   print elections, waking logits and private dream pieces on stderr\n"
        "  --moe-trace      print native sparse FFN choices and weights for waking and dream tokens\n"
        "  -q               no banner\n"
        "  -n, -t           tokens and temperature, for chat as well as one shot\n"
        "  --tokenizer F    choose F instead of the embedded or adjacent vocabulary\n"
        "  NT_CTX=N         context length for chat (default 2048)\n"
        "  NT_BLAS_THREADS=N  OpenBLAS threads for this run (default 1)\n"
        "  NT_PROFILE=1     per-section timings\n", self);
}

/* The shared loader selects the vocabulary. The CLI also accepts raw token IDs
 * or byte input when the decoder has a byte-sized vocabulary. */
static int open_tokenizer(session *s, const gguf_file *gf, const char *model_path, const char *tok_path) {
    const char *error = nt_model_tokenizer_open(gf, model_path, s->vocab, tok_path, &s->tok, &s->eos);
    if (error) { fprintf(stderr, "notorch: %s (%s)\n", error, model_path); return 1; }
    if (s->tok) {
        fprintf(stderr, "tokenizer: GGUF BPE (vocab=%d eos=%d)\n", bpe_n_vocab(s->tok), s->eos);
        return 0;
    }
    const char *raw = getenv("NT_TOKENS");
    if (s->vocab > 256 && !(raw && *raw)) {
        fprintf(stderr, "notorch: %s carries no tokenizer and its %d ids are not bytes; "
                        "pass --tokenizer TOKENIZER.gguf, or ids through NT_TOKENS\n", model_path, s->vocab);
        return 1;
    }
    fprintf(stderr, "tokenizer: byte-level fallback (no GGUF BPE vocab)\n");
    return 0;
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

/* NT_BLAS_THREADS=N gives OpenBLAS N threads for this run, and only when asked: the
 * one thread above stays the default for every family. It is for a family whose
 * matrices are plain f32 through cblas_sgemm with no matvec pool beside them, which
 * is how Qwen3.5 decodes. Accelerate schedules its own threads and has no such call. */
static int blas_threads_from_env(void) {
    const char *env = getenv("NT_BLAS_THREADS");
    if (!env || !*env) return 0;
    char *end;
    long n = strtol(env, &end, 10);
    if (*end || n < 1 || n > 256) {
        fprintf(stderr, "notorch: NT_BLAS_THREADS wants 1..256, not '%s'\n", env);
        return 1;
    }
#if defined(USE_BLAS) && !defined(ACCELERATE)
    if (openblas_set_num_threads) {
        openblas_set_num_threads((int)n);
        fprintf(stderr, "blas: %ld threads (NT_BLAS_THREADS)\n", n);
        return 0;
    }
#endif
    fprintf(stderr, "blas: NT_BLAS_THREADS ignored, this build has no OpenBLAS thread control\n");
    return 0;
}

int main(int argc, char **argv) {
    blas_single_thread();
    if (blas_threads_from_env()) return 1;
    int quiet = 0, ai = 1;
    int flag_n = -1; float flag_t = -1.0f;
    const char *flag_n_text = NULL, *flag_t_text = NULL;
    /* The positional form is what examples/infer_llama.c takes and what the
     * parity gate drives; the flags are for chat, which has no prompt to hang
     * positional arguments behind. */
    int tokenize_only = 0, arch_only = 0, repeats = 1;
    int experts = 0, learn = 0, trace = 0, moe_trace = 0, async = 0, portable = 0, spore_only = 0;
    int dream_tokens = 0, dream_only = 0;
    const char *spore_in = NULL, *spore_out = NULL, *tok_path = NULL, *body_memory = NULL;
    const char *replay_out = NULL, *replay_in = NULL;
    int replay_tokens = 6;
    while (ai < argc && argv[ai][0] == '-' && argv[ai][1]) {
        const char *arg = argv[ai];
        if (!strcmp(arg, "--experts")) { experts = 1; ai++; continue; }
        if (!strcmp(arg, "--learn-experts")) { learn = 1; ai++; continue; }
        if (!strcmp(arg, "--async-consolidation")) { async = 1; ai++; continue; }
        if (!strcmp(arg, "--dream") || !strcmp(arg, "--dream-only")) {
            char *end = NULL;
            long n = ai+1 < argc ? strtol(argv[ai+1], &end, 10) : 0;
            if (dream_tokens || !end || *end || n < 1 || n > NT_DREAM_MAX_TOKENS) {
                fprintf(stderr, "notorch: choose one dream mode with a budget of 1..%d choices\n", NT_DREAM_MAX_TOKENS);
                return 1;
            }
            dream_tokens = (int)n;
            dream_only = !strcmp(arg, "--dream-only");
            ai += 2; continue;
        }
        if (!strcmp(arg, "--spore-only")) { spore_only = 1; ai++; continue; }
        if (!strcmp(arg, "--portable-spore")) { portable = 1; ai++; continue; }
        if (!strcmp(arg, "--expert-trace")) { trace = 1; ai++; continue; }
        if (!strcmp(arg, "--moe-trace")) { moe_trace = 1; ai++; continue; }
        if (!strcmp(arg, "--replay-out") || !strcmp(arg, "--replay-only") ||
            !strcmp(arg, "--body-memory")) {
            const char **value = !strcmp(arg, "--replay-out") ? &replay_out :
                                 !strcmp(arg, "--replay-only") ? &replay_in : &body_memory;
            if (*value || ai+1 >= argc || !*argv[ai+1]) {
                fprintf(stderr, "notorch: %s needs one file\n", arg);
                return 1;
            }
            *value = argv[ai+1]; ai += 2; continue;
        }
        if (!strcmp(arg, "--spore-in") || !strcmp(arg, "--spore-out")) {
            if (++ai >= argc) { usage(argv[0]); return 1; }
            if (!strcmp(arg, "--spore-in")) spore_in = argv[ai++];
            else spore_out = argv[ai++];
            continue;
        }
        if (!strcmp(arg, "--tokenizer")) {
            if (ai + 1 >= argc) {
                fprintf(stderr, "notorch: --tokenizer needs a file\n");
                usage(argv[0]);
                return 1;
            }
            tok_path = argv[ai + 1];
            ai += 2; continue;
        }
        if (arg[2]) break;
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
            else if (f == 'n') { flag_n_text = argv[ai+1]; flag_n = atoi(flag_n_text); }
            else { flag_t_text = argv[ai+1]; flag_t = (float)atof(flag_t_text); }
            ai += 2; continue;
        }
        /* -T prints the ids and stops. A family arrives with two ways to be wrong and this
         * separates them: the tokenizer can be diffed against another implementation without
         * loading a single weight, and the forward can be fed ids through NT_TOKENS. */
        if (f == 'T') { tokenize_only = 1; quiet = 1; ai++; continue; }
        /* -A prints the file's own general.architecture and stops. A gate needs to
         * know which family it is looking at before it decides whether its
         * comparison partner implements that family at all — test_parity.sh read
         * three FAILs off a Qwen3 file because examples/infer_llama.c predates
         * qwen3 and has no QK norm. Cheaper than a tensor probe and it is the
         * same string nt_pick_arch matches on. */
        if (f == 'A') { arch_only = 1; quiet = 1; ai++; continue; }
        break;
    }
    if (ai >= argc) { nt_logo(quiet); usage(argv[0]); return 1; }

    if (replay_in) {
        char *end = NULL;
        errno = 0;
        long n = flag_n_text ? strtol(flag_n_text, &end, 10) : 6;
        int bad_n = flag_n_text && (errno || end == flag_n_text || *end);
        errno = 0;
        double t = flag_t_text ? strtod(flag_t_text, &end) : 0;
        int bad_t = flag_t_text && (errno || end == flag_t_text || *end);
        const char *raw = getenv("NT_TOKENS");
        if (!experts || !learn || !spore_in || replay_out || dream_tokens || spore_only ||
            ai+1 != argc || repeats != 1 || bad_n || n<1 || n>NT_DREAM_MAX_TOKENS ||
            bad_t || !isfinite(t) || t!=0 || (raw && *raw)) {
            fprintf(stderr, "notorch: --replay-only requires --experts --learn-experts --spore-in, "
                            "greedy -n 1..32, and no new input or dream\n");
            return 1;
        }
        replay_tokens = (int)n;
    }
    if (replay_out && (!experts || spore_only || dream_only || ai+1 >= argc || ai+4 < argc)) {
        fprintf(stderr, "notorch: --replay-out requires --experts and one prompt\n");
        return 1;
    }
    if (replay_out) {
        const char *ns = flag_n_text ? flag_n_text : ai+2 < argc ? argv[ai+2] : NULL;
        const char *ts = flag_t_text ? flag_t_text : ai+3 < argc ? argv[ai+3] : NULL;
        char *end = NULL;
        errno = 0;
        long n = ns ? strtol(ns, &end, 10) : 50;
        int bad_n = ns && (errno || end == ns || *end);
        errno = 0;
        double t = ts ? strtod(ts, &end) : 0.8;
        int bad_t = ts && (errno || end == ts || *end);
        if (bad_n || n<1 || n>=8191 || bad_t || !isfinite(t) || t<0) {
            fprintf(stderr, "notorch: captured text requires -n 1..8190 and a finite nonnegative temperature\n");
            return 1;
        }
    }
    if (body_memory && (!experts || !spore_in)) {
        fprintf(stderr, "notorch: --body-memory requires --experts --spore-in\n");
        return 1;
    }

    if (dream_tokens && (!experts || spore_only || (!dream_only && !learn) ||
        (dream_only && (!spore_in || ai+1 != argc || (learn && !spore_out))))) {
        fprintf(stderr, "notorch: --dream requires --experts --learn-experts; --dream-only takes "
                        "--experts --spore-in and one host (add --spore-out when learning)\n");
        return 1;
    }
    if (async && !learn) {
        fprintf(stderr, "notorch: --async-consolidation requires --learn-experts\n");
        return 1;
    }
    if (spore_only && (!spore_in || !spore_out || learn || ai+1 != argc)) {
        fprintf(stderr, "notorch: --spore-only takes --spore-in, --spore-out and one host without a training turn\n");
        return 1;
    }
    if (portable && !spore_out) {
        fprintf(stderr, "notorch: --portable-spore requires --spore-out\n");
        return 1;
    }
    if ((learn || spore_in || spore_out || replay_in || replay_out || body_memory) && !experts) {
        fprintf(stderr, "notorch: expert learning and spore I/O require --experts\n");
        return 1;
    }
    if (moe_trace && (tokenize_only || arch_only || spore_only)) {
        fprintf(stderr, "notorch: --moe-trace needs a waking or dream forward\n");
        return 1;
    }
    if (experts && (tokenize_only || arch_only || repeats != 1)) {
        fprintf(stderr, "notorch: --experts needs one inference session\n");
        return 1;
    }
    nt_logo(quiet);
    const char *path = argv[ai];

    if (arch_only) {
        gguf_file *g = gguf_open(path);
        if (!g) return 1;
        printf("%s\n", g->arch);
        gguf_close(g);
        return 0;
    }

    if (tokenize_only) {
        gguf_file *host = gguf_open_metadata(path);
        if (!host) return 1;
        bpe_tokenizer *tok = NULL;
        int eos;
        const char *error = nt_model_tokenizer_open(host, path, host->vocab_size, tok_path, &tok, &eos);
        gguf_close(host);
        if (error) { fprintf(stderr, "notorch: %s\n", error); return 1; }
        if (!tok) { fprintf(stderr, "notorch: no tokenizer in %s\n", tok_path ? tok_path : path); return 1; }
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

    if (experts && !arch->doe) {
        fprintf(stderr, "notorch: --experts is unavailable on %s\n", gf->arch);
        gguf_close(gf);
        return 1;
    }
    if (moe_trace && !arch->forward_routes) {
        fprintf(stderr, "notorch: --moe-trace is unavailable on %s\n", gf->arch);
        gguf_close(gf);
        return 1;
    }
    nt_dims dims = {0};
    void *model = arch->load(gf, &dims);
    if (!model) { gguf_close(gf); return 1; }
    fprintf(stderr, "loaded in %.0f ms\n", now_ms() - t0);

    session s = { .arch = arch, .model = model, .vocab = dims.vocab, .eos = -1, .learn = learn,
                  .trace = trace, .moe_trace = moe_trace, .async = async, .dream_tokens = dream_tokens };
    chat_wrap_init(&g_chat, dims.vocab, path);
    if (open_tokenizer(&s, gf, path, tok_path)) { arch->free(model); gguf_close(gf); return 1; }

    const char *prompt = (ai + 1 < argc) ? argv[ai + 1] : NULL;
    int max_tokens = (ai + 2 < argc) ? atoi(argv[ai + 2]) : 50;
    float temp = (ai + 3 < argc) ? (float)atof(argv[ai + 3]) : 0.8f;
    if (flag_n > 0) max_tokens = flag_n;
    if (flag_t >= 0.0f) temp = flag_t;
    if (max_tokens < 1) max_tokens = 1;

    pf_on = getenv("NT_PROFILE") != NULL;
    int rc = 1;
    nt_replay_episode *episode = NULL;
    if (experts) {
        nt_expert_body body = {model, s.tok};
        s.experts = nt_experts_open_on_body(gf, spore_in, trace ? stderr : NULL, &body);
        if (!s.experts) {
            fprintf(stderr, "notorch: could not open expert state%s%s\n",
                    spore_in ? " from " : "", spore_in ? spore_in : "");
            goto done;
        }
        if (body_memory) {
            nt_experts *local = nt_experts_open(gf, body_memory, NULL);
            int matched = 0;
            int result = local ? nt_experts_recall_body(s.experts, local, &matched) : NT_E_STATE;
            nt_experts_free(local);
            if (result) {
                fprintf(stderr, "notorch: body memory refused: %s\n", nt_strerror(result));
                goto done;
            }
            fprintf(stderr, "doe body-memory matched=%d\n", matched);
        }
        if (moe_trace && (dream_tokens || replay_in) &&
            nt_experts_set_dream_route(s.experts, print_moe_route, replay_in ? "replay-route" : "dream-route")) {
            fprintf(stderr, "notorch: could not observe native routes on the private dream body\n");
            goto done;
        }
    }
    if (spore_only) { rc = 0; goto done; }
    if (replay_in) {
        episode = nt_replay_load(s.experts, replay_in);
        if (!episode) {
            fprintf(stderr, "notorch: replay episode refused for this body\n");
            goto done;
        }
        rc = async ? nt_experts_submit_replay(&s.experts, &s.pending, episode, replay_tokens)
                   : nt_experts_replay(s.experts, episode, replay_tokens, 1);
        if (rc) fprintf(stderr, "notorch: private replay failed: %s\n", nt_strerror(rc));
        goto done;
    }
    if (dream_only) {
        rc = consolidate(&s, 0);
        if (rc) fprintf(stderr, "notorch: private dream failed: %s\n", nt_strerror(rc));
        goto done;
    }
    s.logits = (float*)calloc(dims.vocab, sizeof(float));
    if (!s.logits) goto done;
    if ((experts || trace) && !isfinite(temp)) {
        fprintf(stderr, "notorch: expert sampling requires a finite temperature\n");
        goto done;
    }
    if ((experts || trace) && temp > 0) {
        s.sampling_copy = malloc((size_t)dims.vocab * sizeof(float));
        if (!s.sampling_copy) goto done;
    }
    rc = 0;
    if (prompt) {
        /* Size the cache to the job. The reference example carried a fixed 256
         * and clamped the prompt into what was left, which turns a long prompt
         * into a quietly truncated one. */
        int cap = 8192;
        int *tokens = (int*)malloc((size_t)cap * sizeof(int));
        int n_tok = tokens ? encode_prompt(&s, prompt, tokens, cap - max_tokens - 1) : 0;
        if (n_tok <= 0) { fprintf(stderr, "notorch: no ids for this prompt (empty, or refused by the tokenizer)\n"); rc = 1; }
        else {
            s.max_seq = n_tok + max_tokens + 1;
            s.kv = nt_model_kv_new(arch, model, dims, s.max_seq);
            if (!s.kv) { fprintf(stderr, "notorch: could not make a cache for %d positions\n", s.max_seq); rc = 1; }
        }
        if (s.kv && replay_out) {
            episode = capture_text(&s, tokens, n_tok);
            if (!episode || nt_replay_save(episode, replay_out)) {
                fprintf(stderr, "notorch: could not publish a new replay episode at %s\n", replay_out);
                rc = 1;
            }
        }
        if (s.kv && !rc) {
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
                if (!s.kv || !run_turn(&s, tokens, n_tok, 0, max_tokens, temp, 1)) {
                    rc = 1;
                    break;
                }
            }
        }
        free(tokens);
    } else {
        const char *ctx = getenv("NT_CTX");
        s.max_seq = ctx ? atoi(ctx) : 2048;
        if (s.max_seq < max_tokens + 2) s.max_seq = max_tokens + 2;
        s.kv = nt_model_kv_new(arch, model, dims, s.max_seq);
        if (!s.kv) { fprintf(stderr, "notorch: could not make a cache for %d positions\n", s.max_seq); rc = 1; }
        else rc = chat(&s, max_tokens, temp);
    }

done:
    if (s.pending && nt_experts_await(&s.experts, &s.pending)) {
        fprintf(stderr, "notorch: final expert consolidation failed\n");
        rc = 1;
    }
    if (!rc && spore_out && (portable ? nt_experts_save_portable(s.experts, spore_out)
                                     : nt_experts_save(s.experts, spore_out))) {
        fprintf(stderr, "notorch: could not publish a new spore at %s\n", spore_out);
        rc = 1;
    }
    nt_experts_free(s.experts);
    nt_replay_free(episode);
    free(s.sampling_copy);
    if (s.tok) bpe_free(s.tok);
    free(s.logits);
    if (s.kv) {
        nt_model_kv_free(arch, model, s.kv);
    }
    arch->free(model); gguf_close(gf);
    return rc;
}
