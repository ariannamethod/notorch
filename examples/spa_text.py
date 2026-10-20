"""Train a native word model and let SPA live in its sentence field.

    make spa_text BLAS_FLAGS= BLAS_LIBS=
    PYTHONPATH=python python3 examples/spa_text.py --steps 20 --save text.spa
    PYTHONPATH=python python3 examples/spa_text.py --resume text.spa --steps 20

The body learns next-word cross entropy from --corpus, or the original small
corpus below. Native notorch owns training, generation, sentence perception,
policy and consequence learning. KEEP retains the target; reseeding generates
one replacement from a neighbour's first content word. The complete body,
sentence field, counters and both RNG streams resume with the Agent.
"""
from __future__ import annotations

import argparse
import base64
import ctypes as C
import json
import math
from pathlib import Path
import re
import sys
import threading

import SPA

ROOT = Path(__file__).resolve().parents[1]
CORPUS = """The river carries light through the valley. The wind carries a song
through the trees. A traveller follows the river toward the sea. The river
returns to the sea and the rain returns to the valley. A traveller hears a
song among the trees. The wind moves the leaves and the light moves across
the water. A small boat crosses the river. The boat carries a traveller
toward the shore. The traveller returns with a story. A story carries the
sound of the river. The trees hold the rain and the valley holds the river.
The evening light falls across the water. The morning wind returns to the
trees. A traveller remembers the shore and follows the light. The river
carries the boat and the boat carries the story. The song returns with the
wind. The water reflects the trees and the trees shelter the traveller.
The valley opens toward the sea. The traveller listens to the river and
remembers the song. The light returns and the story continues."""


def _int(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return value


class BodyLibrary:
    """ctypes boundary to the example body, linked to upstream libnotorch."""
    _training_lock = threading.Lock()

    def __init__(self, path=None):
        if path is None:
            path = ROOT / ("spa_text_body.dylib" if sys.platform == "darwin" else "spa_text_body.so")
        self.path = Path(path)
        try:
            self.lib = C.CDLL(str(self.path.resolve()))
        except OSError as exc:
            raise RuntimeError("Build the native text body with make spa_text") from exc
        signatures = {
            "create": ([C.c_uint32, C.c_uint64], C.c_void_p),
            "free": ([C.c_void_p], None),
            "vocabulary": ([C.c_void_p], C.c_uint32),
            "dimension": ([], C.c_uint32),
            "train": ([C.c_void_p, C.POINTER(C.c_uint32), C.c_size_t, C.c_uint32,
                       C.c_float, C.POINTER(C.c_float)], C.c_int),
            "embeddings": ([C.c_void_p, C.POINTER(C.c_float), C.c_size_t], C.c_int),
            "generate": ([C.c_void_p, C.c_uint32, C.c_uint32, C.c_float,
                          C.POINTER(C.c_uint32), C.c_size_t], C.c_int),
            "snapshot_size": ([C.c_void_p], C.c_size_t),
            "snapshot": ([C.c_void_p, C.c_void_p, C.c_size_t], C.c_int),
            "restore": ([C.c_void_p, C.c_size_t], C.c_void_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, "spa_text_" + name)
            function.argtypes, function.restype = arguments, result
        if self.lib.spa_text_dimension() != 16:
            raise RuntimeError("incompatible SPA text body dimension")

    def create(self, vocabulary, seed=42):
        pointer = self.lib.spa_text_create(_int(vocabulary, "vocabulary", 2, 512),
                                            _int(seed, "seed", 0, 2**64 - 1))
        if not pointer:
            raise RuntimeError("native text model allocation failed")
        return Body(self, pointer)

    def restore(self, data):
        if not isinstance(data, bytes) or not 32 <= len(data) <= 65568:
            raise ValueError("invalid native text checkpoint size")
        pointer = self.lib.spa_text_restore(data, len(data))
        if not pointer:
            raise ValueError("native text checkpoint refused")
        return Body(self, pointer)


class Body:
    def __init__(self, library, pointer):
        self.library, self.pointer = library, pointer

    def close(self):
        if self.pointer:
            self.library.lib.spa_text_free(self.pointer)
            self.pointer = None

    def __del__(self):
        self.close()

    @property
    def vocabulary(self):
        if not self.pointer:
            raise ValueError("text body is closed")
        return self.library.lib.spa_text_vocabulary(self.pointer)

    def snapshot(self):
        size = self.library.lib.spa_text_snapshot_size(self.pointer)
        if not size:
            raise ValueError("text body is closed or invalid")
        out = C.create_string_buffer(size)
        if self.library.lib.spa_text_snapshot(self.pointer, out, size):
            raise RuntimeError("native text snapshot failed")
        return bytes(out)

    def train(self, tokens, epochs=300, learning_rate=2.0):
        tokens = [_int(t, "token", 0, self.vocabulary - 1) for t in tokens]
        _int(len(tokens), "token count", 2, 4096)
        _int(epochs, "epochs", 1, 10000)
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 4:
            raise ValueError("learning rate must be in (0,4]")
        array = (C.c_uint32 * len(tokens))(*tokens)
        losses = (C.c_float * 2)()
        with self.library._training_lock:
            if self.library.lib.spa_text_train(self.pointer, array, len(array), epochs,
                                               learning_rate, losses):
                raise RuntimeError("native training refused (requires an idle notorch tape)")
        return tuple(losses)

    def embeddings(self):
        out = (C.c_float * (self.vocabulary * 16))()
        if self.library.lib.spa_text_embeddings(self.pointer, out, len(out)):
            raise RuntimeError("native embedding read failed")
        return out

    def generate(self, context, end, temperature=.8, capacity=23):
        _int(context, "context", 0, self.vocabulary - 1)
        _int(end, "end", 0, self.vocabulary - 1)
        _int(capacity, "capacity", 1, 64)
        if not math.isfinite(temperature) or not 0 < temperature <= 16:
            raise ValueError("temperature must be in (0,16]")
        out = (C.c_uint32 * capacity)()
        count = self.library.lib.spa_text_generate(self.pointer, context, end,
                                                  temperature, out, capacity)
        if count < 1:
            raise RuntimeError("native generation refused")
        return list(out[:count])


class TextHost:
    """Four generated sentences; neighbour actions rewrite exactly one sentence.

    All seven metrics are explicit: SPA local/mean connectedness, native
    adjacent-cosine coherence, native novelty and repetition, exact duplicate
    sentence fraction, and continuity equal to adjacent coherence. Rewrite
    cost is generated-token count divided by the fixed generation cap.
    """
    host_id = "notorch.spa-text.v1"

    def __init__(self, native=None, library=None):
        self.native = native or SPA.Native()
        self.library = library or BodyLibrary()
        self.body = self.library.create(2)
        self.words = [".", "start"]
        self.sentences = [[1, 0] for _ in range(4)]
        self.reseeds = [0] * 4
        self.cursor = 0
        self.temperature = .8
        self.max_tokens = 24

    @classmethod
    def trained(cls, corpus=CORPUS, *, native=None, library=None, seed=42,
                epochs=300, learning_rate=2.0, temperature=.8):
        if not isinstance(corpus, str):
            raise TypeError("corpus must be text")
        tokens = re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*|[.!?]", corpus.lower())
        tokens = ["." if t in ".!?" else t for t in tokens]
        if not tokens or tokens[-1] != ".":
            tokens.append(".")
        _int(len(tokens), "corpus tokens", 2, 4096)
        words = ["."] + sorted(set(tokens) - {"."})
        _int(len(words), "corpus vocabulary", 2, 512)
        if any(len(word) > 256 for word in words):
            raise ValueError("corpus words must contain at most 256 characters")
        if not math.isfinite(temperature) or not 0 < temperature <= 16:
            raise ValueError("temperature must be in (0,16]")
        ids = {word: index for index, word in enumerate(words)}
        sequence = [ids[t] for t in tokens]
        host = cls(native, library)
        body = host.library.create(len(words), seed)
        try:
            losses = body.train(sequence, epochs, learning_rate)
            starts = [t for i, t in enumerate(sequence) if t != 0 and (i == 0 or sequence[i - 1] == 0)]
            if not starts:
                raise ValueError("corpus needs a sentence containing words")
            host.sentences = [[starts[i % len(starts)]] + body.generate(
                starts[i % len(starts)], 0, temperature, host.max_tokens - 1) for i in range(4)]
        except BaseException:
            body.close()
            host.close()
            raise
        host.body.close()
        host.body, host.words, host.temperature = body, words, temperature
        return host, losses

    def close(self):
        self.body.close()

    def text(self):
        return [" ".join(self.words[t] for t in sentence).replace(" .", ".")
                for sentence in self.sentences]

    def observe(self, target):
        _int(target, "target", 0, len(self.sentences) - 1)
        weights = self.body.embeddings()
        field = [self.native.embed_sentence(tokens, weights, dim=16)
                 for tokens in self.sentences]
        observation = self.native.perceive(field, target, temperature=self.temperature,
                                           reseeds=self.reseeds[target])
        collapse = 1 - len({tuple(s) for s in self.sentences}) / len(self.sentences)
        metrics = SPA.Metrics(observation.connectedness, observation.mean_sentence_score,
                              observation.coherence, observation.novelty, observation.repetition,
                              collapse, observation.coherence)
        return observation, metrics

    def execute(self, action):
        observation, _ = self.observe(action.target)
        self.native.validate_action(action, observation)
        if self.cursor == 2**64 - 1:
            raise ValueError("text host cursor exhausted")
        cost = 0.0
        if action.kind != SPA.ActionKind.KEEP:
            if self.reseeds[action.target] == 1000000:
                raise ValueError("sentence reseed count exhausted")
            source = self.sentences[action.source]
            # Start from the neighbour's beginning. A one-word model seeded
            # with its final word would primarily learn to emit the terminator.
            context = next((t for t in source if t != 0), 0)
            generated = self.body.generate(context, 0, self.temperature, self.max_tokens - 1)
            self.sentences[action.target] = [context] + generated
            self.reseeds[action.target] += 1
            cost = len(generated) / (self.max_tokens - 1)
        self.cursor += 1
        return cost

    def snapshot(self):
        return json.dumps({"version": 1, "body": base64.b64encode(self.body.snapshot()).decode("ascii"),
                           "words": self.words, "sentences": self.sentences, "reseeds": self.reseeds,
                           "cursor": self.cursor, "temperature": self.temperature,
                           "max_tokens": self.max_tokens}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    def restore(self, data):
        if not isinstance(data, bytes) or len(data) > 1000000:
            raise ValueError("invalid text host snapshot")
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate text host key")
                result[name] = value
            return result
        state = json.loads(data, object_pairs_hook=unique)
        if not isinstance(state, dict) or set(state) != {
                "version", "body", "words", "sentences", "reseeds", "cursor", "temperature", "max_tokens"}:
            raise ValueError("invalid text host schema")
        _int(state["version"], "version", 1, 1)
        _int(state["cursor"], "cursor", 0, 2**64 - 1)
        _int(state["max_tokens"], "max_tokens", 2, 64)
        words, sentences, reseeds = state["words"], state["sentences"], state["reseeds"]
        if (not isinstance(words, list) or not 2 <= len(words) <= 512 or
                any(not isinstance(w, str) or not w or len(w) > 256 for w in words) or
                words[0] != "." or len(set(words)) != len(words)):
            raise ValueError("invalid text vocabulary")
        if not isinstance(sentences, list) or len(sentences) != 4:
            raise ValueError("text host requires four sentences")
        for sentence in sentences:
            if not isinstance(sentence, list) or not 1 <= len(sentence) <= state["max_tokens"]:
                raise ValueError("invalid sentence length")
            for token in sentence:
                _int(token, "sentence token", 0, len(words) - 1)
        if not isinstance(reseeds, list) or len(reseeds) != 4:
            raise ValueError("invalid text reseeds")
        for count in reseeds:
            _int(count, "reseed count", 0, 1000000)
        temperature = state["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 < temperature <= 16:
            raise ValueError("invalid text temperature")
        if not isinstance(state["body"], str):
            raise ValueError("invalid text body encoding")
        candidate = self.library.restore(base64.b64decode(state["body"], validate=True))
        if candidate.vocabulary != len(words):
            candidate.close()
            raise ValueError("body/vocabulary mismatch")
        old = self.body
        self.body, self.words, self.sentences, self.reseeds = candidate, words, sentences, reseeds
        self.cursor, self.temperature, self.max_tokens = state["cursor"], temperature, state["max_tokens"]
        old.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="UTF-8 training text; at most 4096 words/punctuation")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--exploration", type=float, default=.25)
    parser.add_argument("--mode", choices=("learned", "legacy", "disabled"), default="learned")
    parser.add_argument("--save", type=Path, help="atomically save the whole body and Agent")
    parser.add_argument("--resume", type=Path, help="resume a saved body and Agent; no retraining")
    parser.add_argument("--library", type=Path, help="upstream libnotorch")
    parser.add_argument("--body-library", type=Path, help="compiled spa_text_body library")
    args = parser.parse_args(argv)
    if args.steps < 0 or args.steps > 1000000:
        parser.error("--steps must be in [0,1000000]")
    if args.resume and args.corpus:
        parser.error("--resume contains its model and vocabulary; omit --corpus")
    if not 1 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be in [1,4294967295]")
    if not math.isfinite(args.exploration) or not 0 <= args.exploration <= 1:
        parser.error("--exploration must be in [0,1]")
    native_path = args.library or ROOT / ("libnotorch.dylib" if sys.platform == "darwin" else "libnotorch.so")
    native, library = SPA.Native(native_path), BodyLibrary(args.body_library)
    if args.resume:
        host = TextHost(native, library)
        engine = SPA.Engine(host, native=native)
        engine.load(args.resume)
        print(f"Resumed at step {host.cursor}; agent has {engine.agent.state.updates} learned updates.")
    else:
        corpus = args.corpus.read_text(encoding="utf-8") if args.corpus else CORPUS
        host, losses = TextHost.trained(corpus, native=native, library=library,
                                      seed=args.seed, epochs=args.epochs, temperature=args.temperature)
        config = SPA.Config.default(native=native, mode=SPA.Mode[args.mode.upper()],
                                    seed=args.seed, exploration=args.exploration)
        engine = SPA.Engine(host, native=native, config=config)
        print(f"Native word model: {len(host.words)} words, {len(host.words) * 32} parameters; "
              f"cross entropy {losses[0]:.4f} -> {losses[1]:.4f}.")
    try:
        for index, sentence in enumerate(host.text()):
            print(f"[{index}] {sentence}")
        for _ in range(args.steps):
            target = host.cursor % len(host.sentences)
            result = engine.step(target)
            action = SPA.ActionKind(result.action.kind).name
            if result.receipt is None:
                print(f"SPA disabled: {action}; body unchanged.")
                break
            print(f"{host.cursor:04d} {action} [{target}] reward={result.receipt.reward:+.5f} "
                  f"coherence={result.after.coherence:.3f} novelty={result.after.novelty:.3f} "
                  f"repetition={result.after.repetition:.3f}")
            if result.action.kind != SPA.ActionKind.KEEP:
                print("    " + host.text()[target])
        destination = args.save or args.resume
        if destination:
            engine.save(destination)
            print(f"Saved complete SPA session: {destination}")
    finally:
        host.close()


if __name__ == "__main__":
    main()
