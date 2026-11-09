/* The CLI's text side and its /reset, through the real harness/main.c: which
 * tokenizer a run gets, when bytes are refused as ids, and that a family cache is
 * made again by its own hooks instead of being zeroed through its kv_dim sentinel. */
#define main notorch_cli_main
#include "../harness/main.c"
#undef main
#include "qwen35_fixture.h"
#include <math.h>
#include <unistd.h>

static int checks, failed;
#define CHECK(c,s) do { checks++; if (!(c)) { fprintf(stderr,"FAIL: %s\n",s); failed++; } } while (0)

/* A byte-level vocabulary of n tokens: the 256 byte symbols, then special fillers. */
static int tokenizer_file(const char *path, const char *arch, int n) {
    static char words[300][16];
    const char *tokens[300];
    if (n > 300) return -1;
    int next = 0;
    for (int b = 0; b < 256; b++) {
        int literal = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        int cp = literal ? b : 256 + next++;
        memset(words[b], 0, sizeof(words[b]));
        if (cp < 128) words[b][0] = (char)cp;
        else { words[b][0] = (char)(0xc0 | (cp >> 6)); words[b][1] = (char)(0x80 | (cp & 63)); }
    }
    for (int i = 256; i < n; i++) snprintf(words[i], sizeof(words[i]), "<|x%d|>", i);
    for (int i = 0; i < n; i++) tokens[i] = words[i];
    const char *merges[] = {"x y"};
    gguf_writer *w = gguf_write_open(path);
    if (!w) return -1;
    if (arch) gguf_write_kv_str(w, "general.architecture", arch);
    gguf_write_kv_str(w, "tokenizer.ggml.model", "gpt2");
    gguf_write_kv_str_array(w, "tokenizer.ggml.tokens", tokens, n);
    gguf_write_kv_str_array(w, "tokenizer.ggml.merges", merges, 1);
    gguf_write_kv_u32(w, "tokenizer.ggml.eos_token_id", (uint32_t)(n - 1));
    return gguf_write_close(w);
}

int main(void) {
    char model[] = "/tmp/nt_cli_model_XXXXXX", tok[] = "/tmp/nt_cli_tok_XXXXXX";
    int fd = mkstemp(model); if (fd < 0) return 1; close(fd);
    fd = mkstemp(tok); if (fd < 0) { unlink(model); return 1; } close(fd);
    gguf_file gf; memset(&gf, 0, sizeof(gf)); strcpy(gf.arch, "qwen35");
    { gguf_writer *w = gguf_write_open(model); CHECK(w && gguf_write_kv_str(w, "general.architecture", "qwen35") == 0 && gguf_write_close(w) == 0, "model without tokenizer written"); }

    unsetenv("NT_TOKENS");
    session s = { .vocab = 248320, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, NULL) == 1 && !s.tok, "no tokenizer and non-byte ids: refused");
    setenv("NT_TOKENS", "1,2,3", 1);
    s = (session){ .vocab = 248320, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, NULL) == 0 && !s.tok, "NT_TOKENS still reaches the forward");
    unsetenv("NT_TOKENS");
    s = (session){ .vocab = 256, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, NULL) == 0 && !s.tok, "a byte-id model keeps the byte scheme");

    CHECK(tokenizer_file(tok, "qwen35", 271) == 0, "tokenizer file written");
    s = (session){ .vocab = 300, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, tok) == 0 && s.tok && s.eos == 270, "named tokenizer loads with its own eos");
    if (s.tok) { int ids[8]; CHECK(bpe_encode(s.tok, "ab", ids, 8) == 2, "named tokenizer encodes"); bpe_free(s.tok); }
    s = (session){ .vocab = 40, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, tok) == 1, "tokenizer larger than the model: refused");
    bpe_free(s.tok);
    CHECK(tokenizer_file(tok, "llama", 271) == 0, "foreign tokenizer written");
    s = (session){ .vocab = 300, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, tok) == 1, "tokenizer of another architecture: refused");
    bpe_free(s.tok);
    s = (session){ .vocab = 300, .eos = -1 };
    CHECK(open_tokenizer(&s, &gf, model, "/nonexistent/tok.gguf") == 1 && !s.tok, "missing tokenizer file: refused");

    /* /reset on a family-owned cache: Qwen3.5 stores -35 in kv_dim. */
    CHECK(!fixture(model, 0), "Qwen3.5 fixture written");
    gguf_file *g = gguf_open(model); nt_dims d = {0};
    void *m = g ? nt_arch_qwen35.load(g, &d) : NULL;
    CHECK(m != NULL, "Qwen3.5 fixture loads");
    if (m) {
        int ids[] = {0, 3, 5, 12, 2, 6, 7}; float out[40];
        session q = { .arch = &nt_arch_qwen35, .model = m, .max_seq = 8, .vocab = d.vocab };
        q.kv = nt_arch_qwen35.kv_new(m, 8);
        CHECK(q.kv && q.kv->kv_dim < 0, "family cache carries its sentinel");
        CHECK(q.kv && !nt_arch_qwen35.forward(m, q.kv, ids, 7, 0, out), "forward before reset");
        CHECK(reset_cache(&q) == 0 && q.kv, "reset through the family hooks");
        CHECK(q.kv && nt_arch_qwen35.forward(m, q.kv, ids, 1, 7, out) == NT_E_STATE, "reset cache does not continue the old sequence");
        CHECK(q.kv && !nt_arch_qwen35.forward(m, q.kv, ids, 7, 0, out), "forward after reset");
        for (int i = 0; i < 40; i++) CHECK(fabsf(out[i] - golden_logits[i]) < 2e-5f, "logits after reset");
        if (q.kv) nt_arch_qwen35.kv_free(m, q.kv);
        nt_arch_qwen35.free(m);
    }
    gguf_close(g);
    unlink(model); unlink(tok);
    printf("HARNESS_TOKENIZER_%s (%d checks)\n", failed ? "FAIL" : "OK", checks);
    return !!failed;
}
