# Qwen3-Embedding text encoder

`nt_encoder_qwen3embedding` is the second implementation of `nt_encoder`
(`harness/arch.h`), the sibling of [EmbeddingGemma 2](EMBEDDINGGEMMA2.md) and
the control encoder of the SUBLITERATUS EG line. One encode call processes one
complete sequence. The API takes token IDs and an optional byte mask, and
returns the pooled state and its L2 normalization. Model parameters stay
immutable, and nothing persists between calls.

The target is `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, `model.safetensors` sha256
`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`; every pin
is in [research/qwen3-embedding-pins.json](../research/qwen3-embedding-pins.json).
It is a causal Qwen3 decoder read up to its final RMSNorm: 28 blocks, width
1024, 16 query heads over 8 KV heads of width 128, FFN 3072 with a SiLU gate,
RMS epsilon 1e-6. Q and K get a per-head RMSNorm before NEOX RoPE with theta
1e6; scores are scaled by 1/√128. The pooled vector is the final state of the
last unmasked position. With the tokenizer's framing that position is the
`<|endoftext|>` the post-processor appends. The embedding is the pooled vector
divided by its norm (`1_Pooling` last token, `2_Normalize`). A masked position
is never a key. RoPE positions are sequence indices, masked ones included, as
the reference forms them without position IDs. Left padding therefore changes
only rounding, and right padding leaves the pooled state as it was; the analytic
test checks both.

## Build and convert

```sh
make qwen3embedding_encode qwen3embedding_to_gguf     # scalar F32; BLAS=1 or SIMD=1 as for EG2
./qwen3embedding_to_gguf SNAPSHOT model.gguf
./qwen3embedding_to_gguf --tokenizer SNAPSHOT/tokenizer.json tokenizer.gguf
./qwen3embedding_encode model.gguf queries.tsv vectors.jsonl \
  --tokenizer tokenizer.gguf --condition query
```

The model converter checks `config.json` (a bias-free SiLU `qwen3`, unscaled
RoPE, no sliding window, tied embeddings). It copies the 310 BF16 tensors
unchanged, sorted by name, and writes the architecture metadata the encoder
reads (`qwen3embedding.*`). It refuses a tensor count other than 2 + 11 per
block. For the target revision it writes 1,191,575,136 bytes with sha256
`a4bfbbbd49850f82b7eb8e8b4ba4514648685ae49578ca2ae99cd011656490a2`, the same
on a repeat conversion. `--max-tokens N` (default and maximum 8192) sets the
encoder's input cap. The input embedding table stays mapped BF16, and only the
requested rows are expanded. The other weights are expanded once to F32.

`--tokenizer` writes a separate GGUF from `tokenizer.json`, so the model file's
hash does not depend on it. It refuses a file that is not NFC, the exact Qwen3
split regex, byte-level BPE without byte fallback or `ignore_merges`, and a
`[A, <special>]` template. It writes the 151,669 tokens, the 151,387 merges,
special added tokens as CONTROL and the others as USER_DEFINED, the template's
`<|endoftext|>` as EOS (`add_eos_token`, no BOS), pre-type `qwen3`, and the
source file's SHA-256. For the target revision that is 5,922,016 bytes with
sha256 `710d5b0ce98fc2c3f53f0185951054e46ee750ad3d3e227a1072a684e72c6a25`.

The `qwen3` pre-type in `examples/bpe.c` is the tokenizer.json split
`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
ported alternative by alternative over code points. It runs over the same NFC
quick check as `qwen35`, whose split differs only in counting `\p{M}` as part
of a word. Here a combining mark is punctuation. Text the quick check cannot
normalize exactly is refused. The existing `qwen2` pre-type breaks pieces at
whitespace only and is not used by this encoder.

## Text mode

Each row is `id<TAB>text`. The condition's prompt and the text are one string,
tokenized whole, then `<|endoftext|>` (151643) is appended. The conditions are
the upstream sentence-transformers prompts under their upstream names.
`query` carries
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`
(no space after `Query:`, as `config_sentence_transformers.json` and the model
card's Transformers recipe write it), and `document` carries none. A row is
refused, and the batch fails, when its text holds TAB, CR, LF or NUL, or the
literal of any of the 26 added tokens. It is also refused when the prompt+text
begins or ends with whitespace: sentence-transformers strips it and the
Transformers recipe does not, so the two official paths disagree there. Other
refusals are an over-long framed sequence (never truncated) and IDs that do not
decode back to the text byte for byte, which is how invalid UTF-8 and text that
NFC would reorder show up. Without `--tokenizer`, rows are
`id<TAB>IDs<TAB>optional mask` as for EG2.

The JSONL output holds `id`, `tokens`, `input_ids` (text mode), `seconds`,
`pooled` and `embedding`. `embeddinggemma2_score` reads it unchanged, Matryoshka
`--dim` included. `--trace DIR` writes every hook state as little-endian F32
plus `index.tsv`. The hook sites are `input_embeddings`, `post_attention`,
`block` (the post-FFN residual) and `final_norm`.

## Gates

`make test_qwen3embedding` (in `make test`) runs an independent
double-precision forward written from `modeling_qwen3.py`. It covers two
layers, four query heads over two KV heads, per-head Q/K norm, NEOX RoPE with a
base that turns every pair, SiLU, final norm and last-token pooling over a BF16
embedding table, with a masked key and a masked tail. The required agreement is
max |Δ| ≤ 1e-5. The test also checks right and left padding, writable hooks,
callback failure, nonfinite states, every refusal and untouched outputs on
failure. Six deliberate mutants turn it red: interleaved RoPE, no causal mask,
KV group by modulo, no Q norm, pooling the last position regardless of the
mask, and no score scale.

`make test_qwen3embedding_text` (in `make test`) converts a synthetic
tokenizer.json whose merges straddle every boundary of the split. Wrong cuts
show as merges made or missed. It checks hand-worked IDs (one number per piece,
punctuation prefixing a word, a mark outside a word, `'ſ` folding to `'s`, tab
and NBSP prefixes, whitespace runs, canonical mark order), the converter's four
refusals, framing, and every text-mode refusal. It then runs the CLI over an
analytic model: IDs beside each vector, text mode equal to ID mode, and each
refused row. Six deliberate mutants turn it red: marks inside words, digit runs,
no NFC, no `ſ` fold, no EOS, no whitespace refusal.

`make check_qwen3embedding_model QE_MODEL=… QE_TOKENIZER=…` needs the
converted checkpoint. Two processes must give byte-identical vectors, every
embedding must be 1024 wide with |norm − 1| ≤ 2e-6, and the model card's four
published Transformers-recipe scores must agree within 1e-4, with the card's
prompt and texts.

`make check_qwen3embedding_tokenizer QE_TOKENIZER=… QE_REFERENCE=…` compares
the 28 fixtures of `tests/qwen3embedding_fixtures.tsv` (`id TAB condition TAB
text`) with official IDs. The fixtures cover English queries and documents,
contractions, upper case, numbers, code, identifiers, Russian, Hebrew with and
without niqqud, Arabic with Arabic-Indic digits, Hindi, CJK, Korean, emoji with
ZWJ, flags and keycaps, number-like symbols, whitespace runs, punctuation,
mixed scripts, French, German, a single character and a 3,379-byte paragraph.
The reference file has one line per fixture, `{"id": "...", "input_ids": [...]}`.
Its IDs must come from the pinned `tokenizer.json` (sha256 `def76fb0…b50a`)
applied as sentence-transformers applies it: the condition's prompt, then the
text, encoded with special tokens added, so the post-processor's 151643 closes
it. Without `QE_REFERENCE`, `./test_qwen3embedding_text TOKENIZER.gguf
tests/qwen3embedding_fixtures.tsv` prints the native IDs in the same format.

`tests/check_qwen3embedding_parity.sh MODEL.gguf REFERENCE.jsonl` is the vector
gate. Each line holds `{"id", "input_ids", "embedding"}`: the official model's
normalized output for exactly those IDs. Every coordinate is float32 written
with at least 9 significant digits, from a float32 forward of the unchanged
BF16 weights at batch size 1. The native encoder encodes the reference's own IDs
in ID mode, so the tokenizer boundary stays outside this gate. The thresholds
are EG2's RG3 thresholds, fixed before any reference exists: max |Δ| ≤ 2e-5,
cosine ≥ 0.999999 and native |norm − 1| ≤ 2e-6 for every vector.

## Measured on neo — 2026-11-09

MacBook Neo (A18 Pro), Apple clang 21, portable scalar build (`make`, no
BLAS/SIMD). `test_qwen3embedding`: max |Δ| 2.83e-7 against the double reference;
6/6 mutants red. `test_qwen3embedding_text`: 58/58 checks; 6/6 mutants red.
`check_qwen3embedding_model` PASS. Two processes gave byte-identical vectors for
all four card texts. The widths were 1024, with |norm − 1| at most 2.41e-8. The
published scores agree to at most 2.99e-7:

| pair | native | model card (Transformers recipe) | abs. difference |
|---|---|---|---|
| q1, d1 | 0.76455697903996556 | 0.7645568251609802 | 1.54e-7 |
| q1, d2 | 0.1414252313374984 | 0.14142508804798126 | 1.43e-7 |
| q2, d1 | 0.13549754337802583 | 0.13549736142158508 | 1.82e-7 |
| q2, d2 | 0.59995466367794126 | 0.5999549627304077 | 2.99e-7 |

The native IDs for the card's four texts end in 151643 and carry no BOS.
All 28 fixtures are accepted by the text mode. Their official IDs and the
official vectors are the two open references below.

## Open items

- **Official token IDs for the 28 fixtures.** No approved tool in this tree
  produces them: the GGUF metadata carries the vocabulary and merges, not an
  independent tokenizer. The repo's precedent oracle
  (`tests/qwen35_tokenizer_parity.py`, Python `tokenizers`) and llama.cpp are
  outside this mandate. The reference needs a hand Oleg names, with the tool and
  its version recorded beside the file.
- **Official vectors for the parity gate.** They need a float32 forward of the
  pinned checkpoint by sentence-transformers or Transformers. The four model-card
  scores above are the only official numbers this gate has seen.
