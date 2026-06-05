/* test_gguf_tokens.c — verify gguf_read_str_array reads the tokenizer arrays
 * (tokens + merges) from a real GGUF. Step 2 of the notorch-C packed inference.
 *
 * Build: cc -O2 -I. tests/test_gguf_tokens.c gguf.c -lm -o tests/test_gguf_tokens
 * Run:   ./tests/test_gguf_tokens <model.gguf>
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model.gguf>\n", argv[0]); return 1; }

    int n = 0;
    char **toks = gguf_read_str_array(argv[1], "tokenizer.ggml.tokens", &n);
    printf("tokenizer.ggml.tokens: n=%d\n", n);
    if (toks && n > 0)
        for (int i = 0; i < 3 && i < n; i++) printf("  tok[%d] = '%s'\n", i, toks[i] ? toks[i] : "(null)");

    int nm = 0;
    char **mg = gguf_read_str_array(argv[1], "tokenizer.ggml.merges", &nm);
    printf("tokenizer.ggml.merges: n=%d\n", nm);
    if (mg && nm > 0)
        for (int i = 0; i < 3 && i < nm; i++) printf("  merge[%d] = '%s'\n", i, mg[i] ? mg[i] : "(null)");

    int ok = (n > 0);
    printf("%s\n", ok ? "TOKENS_READ_OK" : "FAIL");
    return ok ? 0 : 1;
}
