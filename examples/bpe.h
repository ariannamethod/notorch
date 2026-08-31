/* bpe.h — the GGUF tokenizer, in both schemes GGUF files come in.
 *
 * Byte-level BPE (GPT-2 / Tekken): tokenizer.ggml.tokens + .merges, merged by
 * rank. Qwen3, SmolLM2, Mistral/Tekken.
 *
 * SentencePiece: tokenizer.ggml.tokens + .scores and no merges, space written
 * U+2581, merged by score, with "<0xHH>" tokens to fall back on. LLaMA,
 * Mistral's older vocabularies, and every nanollama in this tree.
 *
 * Which one a file carries is decided by what it provides — a merge list, or
 * scores without one — not by the name in tokenizer.ggml.model. Running the
 * wrong one is quiet: byte-level encode over a SentencePiece vocabulary finds
 * no token for a space and used to drop it, and the model read a sentence with
 * the spaces missing.
 */
#ifndef BPE_H
#define BPE_H

typedef struct bpe_tokenizer bpe_tokenizer;

/* Load tokens+merges from a GGUF file. NULL on failure. */
bpe_tokenizer *bpe_load(const char *gguf_path);
void bpe_free(bpe_tokenizer *t);

int bpe_n_vocab(const bpe_tokenizer *t);

/* Encode UTF-8 text -> token ids. Writes up to cap ids into out_ids,
 * returns the number written (may be < needed if cap is hit). */
int bpe_encode(const bpe_tokenizer *t, const char *text, int *out_ids, int cap);

/* Decode one token id -> its UTF-8 bytes, appended to buf (cap incl. NUL).
 * Returns bytes appended. */
int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap);

/* id of an exact token string (e.g. "<|im_start|>"), or -1 if absent. */
int bpe_token_id(const bpe_tokenizer *t, const char *token);

/* Whether an id ends the generation for this file, beyond the eos id in the metadata.
 * Some families close a turn with a marker of their own. */
int bpe_is_eog(const bpe_tokenizer *t, int id);

#endif /* BPE_H */
