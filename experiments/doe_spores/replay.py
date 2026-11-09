"""Run the SmolLM2 expert-memory experiment. Usage: replay.py MODEL NEW_DIRECTORY."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

EXPOSURE = "The lighthouse keeper is named Miro. Every evening Miro carries a copper key to the harbor."
ALTERNATE = "The lighthouse keeper is named Nela. Every evening Nela carries a silver bell to the harbor."
CYCLE = ["A robin builds its nest in", "The color of the sea is", "The lighthouse keeper carries a",
         "The capital of France is", "Once upon a time"]
PROBE = CYCLE[2]
VOTE = ["--experts"]
LEARN = [*VOTE, "--learn-experts"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def experiment(model, folder):
    binary = Path(__file__).resolve().parents[2] / "notorch"
    env = os.environ.copy()
    for key in ("NT_TOKENS", "NT_CHAT", "NT_ECHO_IDS"):
        env.pop(key, None)
    env["NT_ATTN_THREADS"] = "1"
    host_hash = sha256(model)
    folder.mkdir()
    runs = {}

    def run(name, flags, turns, tokens, parent=None, save=False):
        args = [binary, "-q", "--expert-trace", *flags, "-n", str(tokens), "-t", "0"]
        if parent:
            args.extend(["--spore-in", parent + ".gguf"])
        if save:
            args.extend(["--spore-out", name + ".gguf"])
        # A fresh KV for every probe, including probes in the same live process.
        args.append(model)
        chat = None
        if isinstance(turns, str):
            args.append(turns)
        else:
            chat = "\n/reset\n".join(turns) + "\n/exit\n"
        result = subprocess.run(args, input=chat, text=True, capture_output=True, cwd=folder, env=env)
        (folder / (name + ".trace")).write_text(result.stderr)
        (folder / (name + ".txt")).write_text(result.stdout)
        result.check_returncode()
        choices = re.findall(r"doe choice step=(\d+) id=(\d+) logit=([^ ]+) best=(\d+)", result.stderr)
        summaries = re.findall(r"doe turn=(\d+) elections=(\d+) signal=([^\n]+)", result.stderr)
        if not choices or (flags and not summaries):
            raise ValueError(f"{name}: missing token or parliament trace")
        runs[name] = {"last_text": result.stdout.rstrip().splitlines()[-1],
                      "first_logit": float(choices[0][2]), "first_id": int(choices[0][1])}
        if summaries:
            runs[name].update(turns=int(summaries[-1][0]), elections=int(summaries[-1][1]))
        if save:
            spore = folder / (name + ".gguf")
            runs[name].update(spore_sha256=sha256(spore), bytes=spore.stat().st_size)
        return re.findall(r"doe (?:vote|choice) [^\n]*", result.stderr)

    run("plain", [], PROBE, 8)
    run("fresh", VOTE, PROBE, 8)
    for name, flags, exposure in [("frozen", VOTE, EXPOSURE), ("learned", LEARN, EXPOSURE),
                                  ("alternate", LEARN, ALTERNATE)]:
        run(name, flags, exposure, 8, save=True)
        run("probe-" + name, VOTE, PROBE, 8, parent=name)
    sequence = CYCLE * 6
    run("frozen30", VOTE, sequence, 4, save=True)
    prior = run("learned30", LEARN, sequence, 4, save=True)
    run("control31", LEARN, PROBE, 4, parent="frozen30")
    resumed = run("resumed31", LEARN, PROBE, 4, parent="learned30", save=True)
    live = run("live31", LEARN, [*sequence, PROBE], 4, save=True)
    if prior + resumed != live or runs["resumed31"]["spore_sha256"] != runs["live31"]["spore_sha256"]:
        raise ValueError("fresh-process continuation diverged from the live parliament")
    if sha256(model) != host_hash:
        raise ValueError("host bytes changed")
    report = {"model": model.name, "host_sha256": host_hash, "runs": runs}
    (folder / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DOE_REPLAY_OK", folder / "receipt.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: replay.py MODEL NEW_DIRECTORY")
    experiment(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
