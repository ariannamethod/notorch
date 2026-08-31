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
 * Architectures are a table. Adding a family is adding a file next to
 * arch_llama.c and one line to ARCHS. */
#include "harness/arch.h"
#include "harness/logo.h"
#include "examples/bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const nt_arch *const ARCHS[] = {
    &nt_arch_gemma4,
    &nt_arch_llama,
};

/* Exact name first, the unnamed fallback second. The reference example ran any
 * architecture it had never heard of through the llama forward, and losing
 * that on the way here would have been a regression dressed as a refactor. */
static const nt_arch *pick_arch(const char *arch) {
    const nt_arch *fallback = NULL;
    for (unsigned i = 0; i < sizeof(ARCHS) / sizeof(ARCHS[0]); i++) {
        const nt_arch *a = ARCHS[i];
        if (!a->names) { fallback = a; continue; }
        for (const char *const *n = a->names; *n; n++)
            if (strcmp(*n, arch) == 0) return a;
    }
    return fallback;
}

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
    if (s->tok) return bpe_encode(s->tok, text, tokens, cap);
    int n = 0;
    if (cap > 0) tokens[n++] = 1;                  /* BOS */
    for (int i = 0; text[i] && n < cap; i++) tokens[n++] = (unsigned char)text[i];
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
        s->arch->forward(s->model, s->kv, tokens + i, cn, pos0 + i,
                         (i + cn == n_tok) ? s->logits : NULL);
    }
    double prefill_ms = now_ms() - gen0;
    pf_report("prefill", prefill_ms);
    pf_reset();

    int pos = pos0 + n_tok, gen = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(s->logits, s->vocab, temp);
        if (s->tok ? (next == s->eos || bpe_is_eog(s->tok, next)) : (next <= 2)) break;
        emit(s, next);
        gen++;
        if (pos >= s->max_seq - 1) break;
        s->arch->forward(s->model, s->kv, &next, 1, pos, s->logits);
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
        "usage: %s [-q] [-n tokens] [-t temp] <model.gguf> [prompt] [max_tokens] [temp]\n"
        "  no prompt        chat in the terminal\n"
        "  -q               no banner\n"
        "  -n, -t           tokens and temperature, for chat as well as one shot\n"
        "  NT_CTX=N         context length for chat (default 2048)\n"
        "  NT_PROFILE=1     per-section timings\n", self);
}

int main(int argc, char **argv) {
    int quiet = 0, ai = 1;
    int flag_n = -1; float flag_t = -1.0f;
    /* The positional form is what examples/infer_llama.c takes and what the
     * parity gate drives; the flags are for chat, which has no prompt to hang
     * positional arguments behind. */
    int tokenize_only = 0;
    while (ai < argc && argv[ai][0] == '-' && argv[ai][1] && !argv[ai][2]) {
        char f = argv[ai][1];
        if (f == 'q') { quiet = 1; ai++; continue; }
        /* -T prints the ids and stops. A family arrives with two ways to be wrong and this
         * separates them: the tokenizer can be diffed against another implementation without
         * loading a single weight, and the forward can be fed ids through NT_TOKENS. */
        if (f == 'T') { tokenize_only = 1; quiet = 1; ai++; continue; }
        if ((f == 'n' || f == 't') && ai + 1 < argc) {
            if (f == 'n') flag_n = atoi(argv[ai + 1]);
            else flag_t = (float)atof(argv[ai + 1]);
            ai += 2; continue;
        }
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
        bpe_free(tok);
        return 0;
    }

    double t0 = now_ms();
    gguf_file *gf = gguf_open(path);
    if (!gf) return 1;

    const nt_arch *arch = pick_arch(gf->arch);
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
            fputs(prompt, stdout);
            fflush(stdout);
            run_turn(&s, tokens, n_tok, 0, max_tokens, temp, 1);
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
