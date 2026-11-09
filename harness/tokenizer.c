#include "harness/tokenizer.h"
#include "harness/runtime.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

const char *nt_model_tokenizer_open(const gguf_file *host, const char *host_path,
                                    int vocab, const char *override_path,
                                    bpe_tokenizer **out, int *eos) {
    if (!host || !host_path || !out || !eos || vocab < 0) return "invalid tokenizer request";
    const char *path = override_path, *error = NULL;
    char *companion = NULL;
    bpe_tokenizer *tok = NULL;
    if (!path) {
        tok = bpe_load(host_path);
        if (!tok && gguf_get_kv(host, "tokenizer.ggml.tokens")) return "invalid embedded tokenizer";
        if (!tok) {
            const char *slash = strrchr(host_path, '/');
            size_t dir = slash ? (size_t)(slash + 1 - host_path) : 0;
            companion = malloc(dir + sizeof("tokenizer.gguf"));
            if (!companion) return "tokenizer path allocation failed";
            memcpy(companion, host_path, dir);
            memcpy(companion + dir, "tokenizer.gguf", sizeof("tokenizer.gguf"));
            struct stat st;
            if (stat(companion, &st) == 0) {
                if (S_ISREG(st.st_mode)) path = companion;
                else error = "tokenizer.gguf is not a regular file";
            } else if (errno != ENOENT) error = "cannot inspect tokenizer.gguf";
        }
    }
    if (path) {
        char arch[sizeof(host->arch)];
        if (!gguf_read_str_kv(path, "general.architecture", arch, sizeof(arch)) &&
            *host->arch && strcmp(arch, host->arch)) error = "tokenizer architecture differs from host";
        else if (!(tok = bpe_load(path))) error = "cannot load model tokenizer";
    }
    int stop = -1;
    uint64_t id;
    if (!error && tok && vocab && bpe_n_vocab(tok) > vocab) error = "tokenizer exceeds model vocabulary";
    if (!error && ((path && !gguf_read_uint_kv(path, "tokenizer.ggml.eos_token_id", &id)) ||
                   !gguf_read_uint_kv(host_path, "tokenizer.ggml.eos_token_id", &id))) {
        if (id > INT_MAX || (vocab && id >= (uint64_t)vocab)) error = "EOS exceeds model vocabulary";
        else if (tok && id >= (uint64_t)bpe_n_vocab(tok)) error = "EOS outside tokenizer vocabulary";
        else stop = (int)id;
    }
    free(companion);
    if (error) { bpe_free(tok); return error; }
    *out = tok;
    *eos = stop;
    return NULL;
}

int nt_sample_model_token(float *logits, int vocab, const bpe_tokenizer *tok, float temp) {
    int tokens = bpe_n_vocab(tok);
    return sample(logits, tokens > 0 && tokens < vocab ? tokens : vocab, temp);
}
