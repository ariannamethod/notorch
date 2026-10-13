#!/usr/bin/env python3
"""Bounded red controls for CLI emission and PEN7 KV metadata review fixes.

Build scalar archives and the two gates first; no model forward is run here.
An optional --model adds the ten long surfaces from an actual PEN7 tokenizer.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = [str(args.model.resolve())] if args.model else []
for name, extra in [('test_harness_emit', model), ('test_penelope', [])]:
    subprocess.run([str(root / name), *extra], check=True)

with tempfile.TemporaryDirectory(prefix='notorch-penelope-review-') as tmp:
    tmp = Path(tmp)
    compiler = [os.environ.get('CC', 'cc'), '-O2', '-std=gnu11', '-pthread', '-I', str(root)]
    archives = [str(root / 'libnotorch_harness.a'), str(root / 'libnotorch.a'), '-lm']
    main = (root / 'harness/main.c').read_text()
    needle = 'int cap = bpe_decode_token_capacity(s->tok, id);'
    assert main.count(needle) == 1
    main = main.replace(needle, 'int cap = 256; /* restored truncation defect */')
    wrong_main = tmp / 'wrong_main.c'
    wrong_main.write_text(main)
    emitter = (root / 'tests/test_harness_emit.c').read_text()
    emitter = emitter.replace('../harness/main.c', str(wrong_main))
    wrong_emit = tmp / 'wrong_emit.c'
    wrong_emit.write_text(emitter)
    emit_binary = tmp / 'wrong_emit'
    subprocess.run([*compiler, str(wrong_emit), *archives, '-o', str(emit_binary)], check=True)
    result = subprocess.run([str(emit_binary), *model], capture_output=True, text=True)
    if result.returncode != 1 or 'HARNESS_EMIT_FAIL' not in result.stdout:
        raise SystemExit('fixed-capacity defect was not caught by the actual emitter gate')
    print('PENELOPE_EMIT_RED_CAUGHT')

    family = (root / 'harness/arch_penelope.c').read_text()
    needle = '        !pen_u32(gf, "penelope.attention.head_count_kv", &kv_heads, 128) || kv_heads != m->heads ||\n'
    assert family.count(needle) == 1
    wrong_family = tmp / 'wrong_family.c'
    wrong_family.write_text(family.replace(needle, ''))
    family_binary = tmp / 'wrong_family'
    subprocess.run([*compiler, str(root / 'tests/test_penelope.c'), str(wrong_family),
                    *archives, '-o', str(family_binary)], check=True)
    result = subprocess.run([str(family_binary)], capture_output=True, text=True)
    if result.returncode != 1 or 'PENELOPE_FAIL' not in result.stdout:
        raise SystemExit('ignored KV-head metadata defect was not caught')
    print('PENELOPE_KV_HEADS_RED_CAUGHT')
