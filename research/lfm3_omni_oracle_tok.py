# reference oracle, not an inference path; deps: transformers 5.22.2, torch 2.15.0, torchvision 0.29.2
"""Official tokenizer ids for the fixture lines (\\n and \\t unescaped). Not a deliverable.
    python research/lfm3_omni_oracle_tok.py tokenizer.json tokfix.txt"""
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
for line in open(sys.argv[2], encoding="utf-8").read().split("\n")[:-1]:
    s = line.replace("\\n", "\n").replace("\\t", "\t")
    print(",".join(map(str, tok.encode(s, add_special_tokens=False).ids)))
