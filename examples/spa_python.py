"""Executable Python host fixture for native SPA: three token sentences, nine words.

    make shared BLAS_FLAGS= BLAS_LIBS=
    PYTHONPATH=python python3 examples/spa_python.py

The fixture's host replaces a sentence from its neighbour's context. SPA owns
perception, action selection, learning and saved life; Python owns the field.
"""

import argparse
import json
from pathlib import Path
import tempfile

import SPA


WORDS = ("river", "crosses", "stone", "wind", "carries", "song", "light", "returns", ".")
EMBEDDINGS = ((1, .1, 0, .2), (.3, .6, .1, 0), (.8, -.1, .2, .1),
              (.1, .7, .3, 0), (.2, .6, .2, .1), (0, .4, .8, .2),
              (.2, 0, .3, .9), (.3, .2, .1, .6), (0, 0, 0, 0))


def measure(native, sentences, target, reseeds):
    field = [native.embed_sentence(tokens, EMBEDDINGS) for tokens in sentences]
    observation = native.perceive(field, target, temperature=.8, reseeds=reseeds)
    # This fixture defines collapse as the fraction of duplicate complete token
    # sentences. Continuity uses native adjacent-cosine coherence. The other
    # five axes come directly from the native perception output.
    collapse = 1 - len({tuple(tokens) for tokens in sentences}) / len(sentences)
    metrics = SPA.Metrics(observation.connectedness, observation.mean_sentence_score,
                          observation.coherence, observation.novelty, observation.repetition,
                          collapse, observation.coherence)
    return observation, metrics


def execute(action, sentences, step):
    if action.kind == SPA.ActionKind.KEEP:
        return 0
    # Four-token host regeneration. The seed is the neighbour's last content
    # token, then a fixture continuation. Generation cost is tokens / cap(8).
    context = sentences[action.source][-2]
    sentences[action.target] = [context, 4, (0, 5, 6)[step % 3], 8]
    return 4 / 8


def run(native, checkpoint, steps):
    sentences = [[0, 1, 2, 8], [3, 4, 5, 8], [6, 7, 0, 8]]
    reseeds = [0, 0, 0]
    agent = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED,
                                        seed=42, exploration=.6), native=native)
    for step in range(steps):
        target = step % len(sentences)
        observation, before = measure(native, sentences, target, reseeds[target])
        decision = agent.choose(observation)
        native.validate_action(decision.action, observation)
        cost = execute(decision.action, sentences, step)
        reseeds[target] += decision.action.kind != SPA.ActionKind.KEEP
        _, after = measure(native, sentences, target, reseeds[target])
        receipt = agent.observe(decision.sequence, decision.action, SPA.Consequence(before, after, cost))
        print(json.dumps({"fixture": "three-sentence-token-host", "step": step,
                          "action": SPA.ActionKind(decision.action.kind).name,
                          "before": before.as_dict(), "after": after.as_dict(),
                          "cost": cost, "reward": receipt.reward,
                          "sentences": [" ".join(WORDS[t] for t in tokens) for tokens in sentences]}))
        if step == steps // 2:
            agent.save(checkpoint)
            old_hash = agent.hash
            agent = SPA.Agent.from_file(checkpoint, native=native)
            assert agent.hash == old_hash
            print(json.dumps({"save_resume": "same acquired life", "hash": old_hash}))
    agent.save(checkpoint)
    return agent.hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--save", type=Path, help="retain the final native Agent life at this path")
    parser.add_argument("--steps", type=int, default=9)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    native = SPA.Native(args.library)
    if args.save:
        run(native, args.save, args.steps)
    else:
        with tempfile.TemporaryDirectory(prefix="spa-python-example-") as directory:
            run(native, Path(directory) / "agent.life", args.steps)
