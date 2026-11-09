"""Qwen3.5 tokenizer parity: the "qwen35" pre-type against the reference.

    python3 -I tests/qwen35_tokenizer_parity.py tokenizer.json TOKENIZER.gguf ./qwen35_tokenize \
        tests/qwen35_unicode_corpus.jsonl [llama-tokenize VOCAB.gguf]

Each corpus row, and 300 seeded random rows, is encoded by tokenizers
(add_special_tokens=False; special tokens in the text are parsed, as
qwen35_tokenize parses them) and by qwen35_tokenize. The gate: every row notorch
accepts has the reference's ids, and every row it refuses, limit-* rows aside,
is one the reference's NFC changes. limit-* rows are text NFC leaves alone that
the quick check still refuses; they are counted, not gated. With
llama-tokenize, every row is also encoded by llama.cpp and its differences from
the reference are listed. Oracle only.
"""
import json, os, random, subprocess, sys, tempfile
from tokenizers import Tokenizer

tj, gguf, tool, corpus = sys.argv[1:5]
llama = sys.argv[5:7]
hf = Tokenizer.from_file(tj)
rows = [json.loads(line) for line in open(corpus, encoding="utf-8") if line.strip()]
assert len({r["id"] for r in rows}) == len(rows)

ATOMS = ("hello World it s d re LL 123 x 'S 'RE 'ſ " "'s 't 're 've 'm 'll 'd ' ! ? . , ... — « » ( ) \" - # @ € ₪ ✓ "
         "привет Мир ёж щ ש ל ו ם בית سلام ب ٣ ۴ ५ ５ ² ½ Ⅻ ① 𝟙 नमस्ते क ि 中 文 日本 한국 "
         "😀 👍🏽 ❤️ 👨‍👩‍👧 🇮🇱 1️⃣").split(" ")
ATOMS += [" ", "  ", "\t", "\n", "\r\n", "\r", " ", "　", " ", "\u000b", "‍", "️",
          "्", "⃝", "⃣", "َ", "ّ", "ْ", "ً"]
ATOMS += [chr(c) for c in range(0x05B0, 0x05BD)] + [chr(c) for c in range(0x0591, 0x05AE, 3)]
ATOMS += ["̲", "͆", "̖", "̵", "͒", "͓", "̕", "̚", "͇"]
ATOMS += ["Å", "豈"]   # NFC_QC No: refused, and NFC changes them
rng = random.Random(35)
for k in range(300):
    rows.append({"id": "fuzz-%d" % k,
                 "prompt": "".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 10)))})

got, i = {}, 0
while i < len(rows):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:]).encode()
    p = subprocess.run([tool, gguf], input=data, capture_output=True)
    lines = p.stdout.decode().splitlines()
    for line in lines:
        rid, ids = line.split("\t")
        got[rid] = [int(x) for x in ids.split()]
    i += len(lines)
    if p.returncode:
        assert b"Invalid prompt row" in p.stderr and i < len(rows), p.stderr
        got[rows[i]["id"]] = None
        i += 1

n = dict(rows=len(rows), accepted=0, match=0, refused=0, refused_nfc_changes=0, limit_refused=0)
bad = []
for r in rows:
    ref = hf.encode(r["prompt"], add_special_tokens=False).ids
    mine, limit = got[r["id"]], r["id"].startswith("limit-")
    changes = hf.normalizer.normalize_str(r["prompt"]) != r["prompt"]
    if mine is None:
        n["limit_refused" if limit else "refused"] += 1
        n["refused_nfc_changes"] += changes and not limit
        if not limit and not changes: bad.append(("refused, NFC leaves it", r, ref, None))
    else:
        n["accepted"] += 1
        n["match"] += mine == ref
        if mine != ref: bad.append(("ids differ", r, ref, mine))
for why, r, ref, mine in bad:
    print("FAIL %s: %s %r\n  ref %s\n  got %s" % (why, r["id"], r["prompt"], ref, mine))
print(json.dumps(n))

if llama:
    diff = {"accepted": 0, "refused": 0}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prompt")
        for r in rows:
            with open(path, "w", encoding="utf-8", newline="") as f: f.write(r["prompt"])
            out = subprocess.run([llama[0], "-m", llama[1], "-f", path, "--ids", "--log-disable",
                                  "--no-escape"], capture_output=True, text=True).stdout
            ids = json.loads(out.strip().splitlines()[-1])
            ref = hf.encode(r["prompt"], add_special_tokens=False).ids
            if ids != ref:
                side = "refused" if got[r["id"]] is None else "accepted"
                diff[side] += 1
                print("llama.cpp != reference (%s by notorch): %s %r" % (side, r["id"], r["prompt"]))
    print(json.dumps({"llama_cpp_differs_from_reference": diff}))
sys.exit(1 if bad else 0)
