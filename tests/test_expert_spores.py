"""Drive the ordinary harness in fresh processes and inspect its GGUF tensors."""
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

BINARY = Path(sys.argv[1]).resolve()
FIXTURES = Path(sys.argv[2]).resolve()


def read_spore(path):
    """Decode this schema independently of the C reader (all tensors are F32)."""
    body = path.read_bytes()
    pos = 0

    def unpack(fmt):
        nonlocal pos
        vals = struct.unpack_from("<" + fmt, body, pos)
        pos += struct.calcsize("<" + fmt)
        return vals

    def string():
        nonlocal pos
        n, = unpack("Q")
        data = body[pos:pos+n].decode()
        pos += n
        return data

    magic, version, nt, nk = unpack("4sIQQ")
    assert (magic, version) == (b"GGUF", 3)
    meta = {}
    for _ in range(nk):
        name = string()
        kind, = unpack("I")
        meta[name] = string() if kind == 8 else unpack({4: "I", 6: "f", 10: "Q"}[kind])[0]
    directory = {}
    for _ in range(nt):
        name = string()
        nd, = unpack("I")
        shape = unpack("Q"*nd)
        dtype, offset = unpack("IQ")
        assert dtype == 0
        directory[name] = (offset, math.prod(shape))
    start = (pos + 31) & ~31
    tensors = {key: body[start+off:start+off+4*n] for key, (off, n) in directory.items()}
    return meta, tensors, {key: start+off for key, (off, _) in directory.items()}


def test(work):
    host = work / "qwen.gguf"
    gemma = work / "gemma.gguf"
    subprocess.run([FIXTURES, host, gemma], check=True)
    original = host.read_bytes()
    source = "!#$"
    probe = "#! "
    opts = ["--experts", "--expert-trace"]
    train = [*opts, "--learn-experts"]

    def run(flags, prompt=probe, model=host, chat=None, ok=True, env=None):
        argv = [BINARY, "-q", "-n", "3", "-t", "0", *flags, model]
        if chat is None:
            argv.append(prompt)
        result = subprocess.run(argv, cwd=work, input=chat, capture_output=True,
                                text=True, env={**os.environ, **(env or {})})
        assert (result.returncode == 0) == ok, (argv, result.stderr)
        return result

    def trace(result):
        return re.findall(r"doe (?:choice|vote) [^\n]*", result.stderr)

    assert not trace(run([]))
    assert set(work.iterdir()) == {host, gemma}
    run(["--spore-out", "implicit.gguf"], ok=False)
    assert not (work / "implicit.gguf").exists()
    run(opts, model=gemma, ok=False)
    a, static = work / "learn.gguf", work / "static.gguf"
    first = run([*train, "--spore-out", a], source)
    frozen = run([*opts, "--spore-out", static], source)
    assert trace(first) == trace(frozen) and trace(first)
    meta, learned, offsets = read_spore(a)
    smeta, fixed, _ = read_spore(static)
    assert meta["doe.host_sha256"] == hashlib.sha256(original).hexdigest()
    assert meta["doe.turns"] == 1 and meta["doe.elections"] == sum("vote " in x for x in trace(first))
    assert meta["doe.consensus"] == smeta["doe.consensus"]
    assert learned["doe.lora_a"] != fixed["doe.lora_a"]
    assert learned["doe.lora_b"] != fixed["doe.lora_b"]
    assert learned["doe.vote"] == fixed["doe.vote"]
    hits = [False] * 6
    for line in trace(first):
        if "gates=" in line:
            gates = [float(x) for x in line.split("gates=")[1].split(",")]
            assert abs(sum(gates)-1) < 1e-6
            hits = [hit or gate > 0 for hit, gate in zip(hits, gates)]
    for tensor in ("doe.lora_a", "doe.lora_b"):
        width = len(learned[tensor]) // len(hits)
        for i, elected in enumerate(hits):
            part = slice(i*width, (i+1)*width)
            assert (learned[tensor][part] != fixed[tensor][part]) == elected

    # Sampling at positive temperature consumes a scratch copy. The Oja
    # signal comes from raw pre-softmax logits, including off-argmax choices.
    sampled = run([*train, "-t", "5"], source)
    gaps = [float(best)-float(chosen) for chosen, best in
            re.findall(r"logit=([^ ]+) best=\d+ best_logit=([^\n]+)", sampled.stderr)]
    assert gaps and max(gaps) > 0
    debts = [g / (g+1) for g in gaps]
    expected = sum(-d if d > 0.3 else 0.1*(1-d) for d in debts)/len(debts)
    signal = float(re.search(r"doe turn=1 elections=\d+ signal=([^\n]+)", sampled.stderr)[1])
    assert abs(expected-signal) < 1e-6

    # Compare both outputs of the next turn: tokens/votes and the entire child
    # spore. /reset clears the live process's KV between the same two prompts.
    child, live_child = work / "child.gguf", work / "live.gguf"
    second = run([*train, "--spore-in", a, "--spore-out", child])
    live = run([*train, "--spore-out", live_child], chat=f"{source}\n/reset\n{probe}\n/exit\n")
    assert trace(live) == trace(first) + trace(second)
    assert child.read_bytes() == live_child.read_bytes()
    assert child.read_bytes() != a.read_bytes()
    assert read_spore(child)[0]["doe.turns"] == 2
    before = a.read_bytes()
    run([*train, "--spore-in", a, "--spore-out", a], ok=False)
    assert a.read_bytes() == before
    run([*train, "--spore-out", host], ok=False)
    assert host.read_bytes() == original
    other = work / "other.gguf"
    other.write_bytes(original + bytes(32))
    run([*opts, "--spore-in", a], model=other, ok=False)
    bad = work / "bad.gguf"
    bad.write_bytes(before[:offsets["doe.rms_history"] + 1])
    for mmap in ("0", "1"):
        run([*opts, "--spore-in", bad], ok=False, env={"NT_GGUF_MMAP": mmap})
    corrupt = bytearray(before)
    struct.pack_into("<f", corrupt, offsets["doe.lora_a"], math.nan)
    bad.write_bytes(corrupt)
    run([*opts, "--spore-in", bad], ok=False)
    # A parliament with equal votes elects lower slots first. Give the fourth
    # slot seven completed low-vitality turns and let it reach DOE's threshold.
    seeded = bytearray(static.read_bytes())
    for key in ("doe.vote", "doe.frequency"):
        start = offsets[key]
        seeded[start:start+len(fixed[key])] = bytes(len(fixed[key]))
    struct.pack_into("<f", seeded, offsets["doe.vitality"] + 3*4, 0)
    struct.pack_into("<f", seeded, offsets["doe.low_vitality"] + 3*4, 7)
    seed_path, retired = work / "seeded.gguf", work / "retired.gguf"
    seed_path.write_bytes(seeded)
    death = run([*train, "--spore-in", seed_path, "--spore-out", retired])
    assert "doe death slot=3" in death.stderr
    assert struct.unpack("<6f", read_spore(retired)[1]["doe.alive"]) == (1, 1, 1, 0, 0, 0)
    assert trace(run([*opts, "--spore-in", retired]))
    # An invalid turn cannot publish a partly accumulated checkpoint.
    partial = work / "partial.gguf"
    run([*train, "--spore-out", partial], env={"NT_TOKENS": "500"}, ok=False)
    assert not partial.exists()
    # Exercise the actual birth threshold, with the same body across KV resets.
    long = run([*train, "--spore-out", work / "long.gguf"],
               chat=(source + "\n/reset\n")*25 + "/exit\n")
    assert "doe birth parent=" in long.stderr
    assert read_spore(work / "long.gguf")[0]["doe.turns"] == 25


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        test(Path(temp))
    print("EXPERT_SPORES_OK")
