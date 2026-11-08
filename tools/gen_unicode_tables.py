"""Regenerates examples/unicode_marks.h and examples/unicode_qwen35.h.

    python3 -I tools/gen_unicode_tables.py LLAMA_CPP_CHECKOUT examples

\\p{M} is the ACCENT_MARK flag of llama.cpp's unicode_ranges_flags, read the way
unicode_letters.h (LETTER) and unicode_numbers.h (NUMBER) were; the run checks
that it reproduces both. ccc and the NFC quick-check refusals come from this
Python's unicodedata, which must be 15.0.0, the rest of the refusals from the
tokenizers NFC normalizer and split regex the "qwen35" pre-type is held to.
Generation tool only: nothing here runs at inference.
"""
import random, re, subprocess, sys, unicodedata
import tokenizers
from tokenizers import Regex
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Split

llama, out = sys.argv[1], sys.argv[2]
assert unicodedata.unidata_version == "15.0.0", unicodedata.unidata_version
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True,
                        text=True, check=True).stdout.strip()
PAIR = r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}"
QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
          r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
SPACE = {*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
         0x2028, 0x2029, 0x202F, 0x205F, 0x3000}   # lfm2_space in examples/bpe.c


def flag_ranges(text, flag):
    body = text.split("unicode_ranges_flags = {", 1)[1].split("};", 1)[0]
    pairs = [(int(a, 16), int(b, 16)) for a, b in re.findall(PAIR, body)]
    ranges = []
    for (start, f), (nxt, _) in zip(pairs, pairs[1:] + [(0x110000, 0)]):
        if f & flag:
            if ranges and ranges[-1][1] + 1 == start: ranges[-1][1] = nxt - 1
            else: ranges.append([start, nxt - 1])
    return ranges


def header_ranges(path):
    return [[int(a, 16), int(b, 16)] for a, b in re.findall(PAIR, open(path).read())]


def runs(cps):
    r = []
    for c in sorted(cps):
        if r and r[-1][1] + 1 == c: r[-1][1] = c
        else: r.append([c, c])
    return r


def emit(path, comment, decl, rows):
    with open(path, "w") as f:
        f.write(comment + decl + " = {\n")
        for row in rows:
            f.write("    {" + ", ".join("0x%X" % v for v in row[:2]) +
                    "".join(", %d" % v for v in row[2:]) + "},\n")
        f.write("};\n")


data = open(llama + "/src/unicode-data.cpp").read()
letters, numbers = flag_ranges(data, 0x0004), flag_ranges(data, 0x0002)
assert letters == header_ranges(out + "/unicode_letters.h")
assert numbers == header_ranges(out + "/unicode_numbers.h")
marks = flag_ranges(data, 0x0010)

cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
word = {c for lo, hi in letters + marks for c in range(lo, hi + 1)}
number = {c for lo, hi in numbers for c in range(lo, hi + 1)}
split = Split(Regex(QWEN35), "isolated")
shape = {(4,): "word", (1, 1, 1, 1): "number", (1, 1, 2): "space", (1, 2, 1): "other"}
unread = set()
for c in cps:
    if c in (0x0A, 0x0D): continue    # alternative 5 owns them; both sides call them \s
    got = shape[tuple(b - a for _, (a, b) in split.pre_tokenize_str("a" + chr(c) * 2 + "a"))]
    if got != ("word" if c in word else "number" if c in number else
               "space" if c in SPACE else "other"):
        unread.add(c)
assert all(unicodedata.category(chr(c)) == "Cn" for c in unread)

ccc = {c: unicodedata.combining(chr(c)) for c in cps}
no = {c for c in cps if unicodedata.normalize("NFC", chr(c)) != chr(c)}
first = {}
for c in cps:
    d = unicodedata.decomposition(chr(c)).split()
    if len(d) == 2 and not d[0].startswith("<") and unicodedata.normalize("NFC", chr(c)) == chr(c):
        first.setdefault(int(d[1], 16), chr(int(d[0], 16)))
for v in range(0x1161, 0x1176): first[v] = "\u1100"   # Hangul V after L
for t in range(0x11A8, 0x11C3): first[t] = "\uAC00"   # Hangul T after LV
maybe = set(first) - no

assert all((c in no) == (not unicodedata.is_normalized("NFC", chr(c))) for c in cps)
assert all(not unicodedata.is_normalized("NFC", first[c] + chr(c)) for c in maybe)

hf = NFC()
assert all(hf.normalize_str(chr(c)) == chr(c) or c in no for c in cps)
divergent = {c for c in cps if ccc[c] and any(
    hf.normalize_str(s) != unicodedata.normalize("NFC", s)
    for s in ("q" + chr(c) + "\u0334", "q\u0301" + chr(c)))}
refuse = no | maybe | divergent | unread


def reorder(s):
    o, i = list(s), 0
    while i < len(o):
        j = i
        while j < len(o) and ccc[ord(o[j])]: j += 1
        o[i:j] = sorted(o[i:j], key=lambda x: ccc[ord(x)])
        i = j + 1
    return "".join(o)


pool = [chr(c) for c in cps if ccc[c] and c not in refuse]
pool += [x for x in set(first.values()) if ord(x) not in refuse]
pool += list("aqzAZ \u0430\u044f\u05d0\u05ea\uac00\u4e00!1")
rng = random.Random(15)
for _ in range(200000):
    s = "".join(rng.choice(pool) for _ in range(rng.randint(1, 8)))
    assert unicodedata.normalize("NFC", s) == reorder(s) == hf.normalize_str(s), s

emit(out + "/unicode_marks.h",
     "/* Unicode general category M (Mn, Mc, Me), inclusive ranges extracted from\n"
     " * llama.cpp %s/src/unicode-data.cpp (the\n"
     " * ACCENT_MARK flag of unicode_ranges_flags), the same source and form as\n"
     " * unicode_letters.h. Used by the \"qwen35\" pre-tokenizer for \\p{M}.\n"
     " * Regenerate with tools/gen_unicode_tables.py. */\n" % commit,
     "static const unsigned int bpe_marks[][2]", marks)

cr = []
for c in cps:
    if not ccc[c]: continue
    if cr and cr[-1][1] + 1 == c and cr[-1][2] == ccc[c]: cr[-1][1] = c
    else: cr.append([c, c, ccc[c]])
comment = (
    "/* Tables for the \"qwen35\" pre-type, from Python unicodedata %s (UAX #15) and\n"
    " * the reference it is held to, tokenizers %s.\n"
    " * nfc_ccc: every nonzero canonical combining class, {first, last, ccc}.\n"
    " * qwen35_refuse: codepoints refused rather than tokenized, because the NFC quick\n"
    " * check cannot settle them or the reference would read them otherwise:\n"
    " * NFC_QC=No (%d), NFC_QC=Maybe (%d: second halves of primary composites, Hangul\n"
    " * V and T jamo), %d marks with ccc > 0 that the reference normalizer orders as\n"
    " * starters, its ccc table predating them, and %d codepoints unassigned in %s\n"
    " * that the reference regex reads as \\p{L}, \\p{M} or \\p{N} and the llama.cpp\n"
    " * tables do not. Text free of all of them is NFC once each run of non-starters\n"
    " * is stably sorted by ccc. Regenerate with tools/gen_unicode_tables.py. */\n"
    % (unicodedata.unidata_version, tokenizers.__version__, len(no), len(maybe),
       len(divergent), len(unread), unicodedata.unidata_version))
emit(out + "/unicode_qwen35.h", comment, "static const unsigned int nfc_ccc[][3]", cr)
with open(out + "/unicode_qwen35.h", "a") as f:
    f.write("static const unsigned int qwen35_refuse[][2] = {\n")
    f.writelines("    {0x%X, 0x%X},\n" % (a, b) for a, b in runs(refuse))
    f.write("};\n")
print("marks %d ranges, ccc %d ranges, refuse %d ranges (No %d, Maybe %d, ccc %d, class %d)"
      % (len(marks), len(cr), len(runs(refuse)), len(no), len(maybe), len(divergent), len(unread)))
