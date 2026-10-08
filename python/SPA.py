"""Sentence Phonon Attention + Agent, through the native notorch C API.

Put this directory on PYTHONPATH and ``import SPA``. Build ``make shared``;
Native(path), NOTORCH_LIB, or the repository's shared library selects the body.
All perception, policy, learning, and checkpoint arithmetic stays in C.
"""

from __future__ import annotations

import ctypes as C
from enum import IntEnum
import math
import operator
import os
import threading

from notorch import find_library

NO_SOURCE = 0xFFFFFFFF
EMBED, FEATURES, HIDDEN, ACTIONS, HISTORY = 4, 29, 8, 3, 8
MAX_REPEATS = 64


class Mode(IntEnum):
    DISABLED = 0
    LEGACY = 1
    LEARNED = 2


class ActionKind(IntEnum):
    KEEP = 0
    RESEED_LEFT = 1
    RESEED_RIGHT = 2


class Status(IntEnum):
    OK = 0
    CONFIG = -20
    STATE = -21
    OBSERVATION = -22
    ACTION = -23
    PENDING = -24
    CONSEQUENCE = -25
    SEQUENCE = -26
    IO = -27
    FORMAT = -28
    MEMORY = -29
    EXPERIENCE = -30
    COMPARISON = -31


class Error(RuntimeError):
    """A refused native operation. ``status`` retains the exact C status."""

    def __init__(self, operation, status):
        self.operation = operation
        self.status = Status(status)
        super().__init__(f"{operation}: NT_SPA_E_{self.status.name} ({status})")


class ABIError(RuntimeError):
    """The loaded library and Python declarations have incompatible layouts."""


def _check(operation, status):
    if status != Status.OK:
        raise Error(operation, status)


def _integer(value, name, minimum=0, maximum=NO_SOURCE):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


class _Value(C.Structure):
    def copy(self):
        return type(self).from_buffer_copy(bytes(self))

    def as_dict(self):
        def unpack(value):
            if isinstance(value, _Value):
                return value.as_dict()
            if isinstance(value, C.Array):
                return [unpack(v) for v in value]
            return value
        return {name: unpack(getattr(self, name)) for name, _ in self._fields_}

    def __setattr__(self, name, value):
        # ctypes otherwise silently wraps integers before C can validate them.
        for field, typ in getattr(type(self), "_fields_", ()):
            if field == name and typ in (C.c_int, C.c_uint32, C.c_uint64):
                bits = C.sizeof(typ) * 8
                signed = typ is C.c_int
                value = _integer(value, name, -(1 << (bits - 1)) if signed else 0,
                                 (1 << (bits - (1 if signed else 0))) - 1)
                break
        super().__setattr__(name, value)


class Action(_Value):
    _fields_ = [("kind", C.c_int), ("target", C.c_uint32), ("source", C.c_uint32)]

    @classmethod
    def at(cls, kind, target):
        kind, target = ActionKind(kind), _integer(target, "target")
        source = (NO_SOURCE if kind == ActionKind.KEEP else
                  target - 1 if kind == ActionKind.RESEED_LEFT else target + 1)
        return cls(kind, target, source)


class Metrics(_Value):
    _fields_ = [(name, C.c_float) for name in (
        "local_connectedness", "global_connectedness", "coherence", "novelty",
        "repetition", "collapse", "continuity")]


class Consequence(_Value):
    _fields_ = [("before", Metrics), ("after", Metrics), ("regeneration_cost", C.c_float)]


class Config(_Value):
    _fields_ = [("mode", C.c_int), ("seed", C.c_uint32),
                ("learning_rate", C.c_float), ("imitation_rate", C.c_float),
                ("exploration", C.c_float), ("memory_decay", C.c_float),
                ("reward_weights", Metrics), ("cost_weight", C.c_float)]

    @classmethod
    def default(cls, *, native=None, **changes):
        config = (native or default_native()).config()
        fields = {name for name, _ in cls._fields_}
        for name, value in changes.items():
            if name not in fields:
                raise TypeError(f"unknown config field: {name}")
            setattr(config, name, value)
        return config


class Observation(_Value):
    _fields_ = [("embedding", C.c_float * EMBED)] + [
        (name, C.c_float) for name in ("connectedness", "left_similarity", "right_similarity",
        "coherence", "novelty", "repetition", "phase_lock", "sentence_score",
        "mean_sentence_score", "temperature")] + [
        (name, C.c_uint32) for name in ("sentence_index", "sentence_count", "reseed_count")]


class Policy(_Value):
    _fields_ = [("w1", (C.c_float * FEATURES) * HIDDEN), ("b1", C.c_float * HIDDEN),
                ("w2", (C.c_float * HIDDEN) * ACTIONS), ("b2", C.c_float * ACTIONS)]


class Decision(_Value):
    _fields_ = [("observation", Observation), ("features", C.c_float * FEATURES),
                ("hidden", C.c_float * HIDDEN), ("scores", C.c_float * ACTIONS),
                ("action", Action), ("sequence", C.c_uint64),
                ("rng_before", C.c_uint32), ("explored", C.c_int)]


class Receipt(_Value):
    _fields_ = [("sequence", C.c_uint64), ("action", Action),
                ("sentence_count", C.c_uint32), ("consequence", Consequence),
                ("reward", C.c_float), ("predicted", C.c_float),
                ("error", C.c_float), ("learned", C.c_int)]


class AgentState(_Value):
    _fields_ = [("version", C.c_uint32), ("perception_version", C.c_uint32),
                ("reward_version", C.c_uint32), ("config", Config),
                ("config_hash", C.c_uint64), ("policy", Policy), ("rng", C.c_uint32)] + [
        (name, C.c_uint64) for name in ("decisions", "observations", "updates",
        "imitation_updates", "cancelled", "memory_observations")] + [
        ("history_count", C.c_uint32), ("history_head", C.c_uint32),
        ("history", Receipt * HISTORY), ("ema_reward", C.c_float),
        ("ema_connectedness", C.c_float), ("ema_novelty", C.c_float),
        ("pending", C.c_int), ("pending_decision", Decision)]


class Experience(_Value):
    _fields_ = [("version", C.c_uint32), ("sentence_index", C.c_uint32),
                ("sentence_count", C.c_uint32), ("source_life_hash", C.c_uint64),
                ("features", C.c_float * FEATURES)]


class Alternative(_Value):
    _fields_ = [("action", Action), ("consequence", Consequence)]


class Comparison(_Value):
    _fields_ = [("source_life_hash", C.c_uint64), ("horizon", C.c_uint32),
                ("action_mask", C.c_uint32), ("alternatives", Alternative * ACTIONS)]

    @classmethod
    def from_outcomes(cls, experience, outcomes, *, horizon=0):
        """Associate each measured action's consequence with one frozen input."""
        _require(experience, Experience)
        result = cls(source_life_hash=experience.source_life_hash,
                     horizon=_integer(horizon, "horizon", 0, 4096))
        for kind, consequence in outcomes.items():
            kind = ActionKind(kind)
            _require(consequence, Consequence)
            result.alternatives[kind] = Alternative(
                Action.at(kind, experience.sentence_index), consequence)
            result.action_mask |= 1 << kind
        return result


class Readout(_Value):
    _fields_ = [("action", Action), ("action_mask", C.c_uint32),
                ("scores", C.c_float * ACTIONS)]


class ComparisonReceipt(_Value):
    _fields_ = [("source_life_hash", C.c_uint64), ("horizon", C.c_uint32),
                ("action_mask", C.c_uint32), ("learning_rate", C.c_float),
                ("rewards", C.c_float * ACTIONS), ("targets", C.c_float * ACTIONS),
                ("scores_before", C.c_float * ACTIONS), ("scores_after", C.c_float * ACTIONS),
                ("loss_before", C.c_double), ("loss_after", C.c_double)]


class ConditionedReceipt(_Value):
    _fields_ = [("comparison", ComparisonReceipt), ("scale_floor", C.c_float),
                ("scale", C.c_double)]


_STRUCTS = {
    "nt_spa_action": Action, "nt_spa_metrics": Metrics,
    "nt_spa_consequence": Consequence, "nt_spa_agent_config": Config,
    "nt_spa_observation": Observation, "nt_spa_policy": Policy,
    "nt_spa_decision": Decision, "nt_spa_receipt": Receipt,
    "nt_spa_agent": AgentState, "nt_spa_experience": Experience,
    "nt_spa_alternative": Alternative, "nt_spa_comparison": Comparison,
    "nt_spa_readout": Readout, "nt_spa_comparison_receipt": ComparisonReceipt,
    "nt_spa_conditioned_receipt": ConditionedReceipt,
}


def _layout():
    result = {"NT_SPA_AGENT_VERSION": 1, "NT_SPA_AGENT_PERCEPTION_VERSION": 1,
              "NT_SPA_AGENT_REWARD_VERSION": 1, "NT_SPA_EXPERIENCE_VERSION": 1,
              "NT_SPA_AGENT_EMBED": EMBED, "NT_SPA_AGENT_FEATURES": FEATURES,
              "NT_SPA_AGENT_HIDDEN": HIDDEN, "NT_SPA_AGENT_ACTIONS": ACTIONS,
              "NT_SPA_AGENT_PARAMETERS": 267, "NT_SPA_AGENT_HISTORY": HISTORY,
              "NT_SPA_AGENT_MAX_SENTENCES": 4096, "NT_SPA_AGENT_MAX_DIM": 4096,
              "NT_SPA_AGENT_NO_SOURCE": NO_SOURCE, "NT_SPA_COMPARISON_MAX_HORIZON": 4096,
              "NT_SPA_COMPARISON_MAX_REPEATS": MAX_REPEATS,
              "sizeof.float": C.sizeof(C.c_float), "sizeof.double": C.sizeof(C.c_double),
              "sizeof.nt_spa_action_kind": C.sizeof(C.c_int),
              "sizeof.nt_spa_agent_mode": C.sizeof(C.c_int)}
    result.update({"NT_SPA_AGENT_" + mode.name: int(mode) for mode in Mode})
    result.update({"NT_SPA_" + action.name: int(action) for action in ActionKind})
    for name, struct in _STRUCTS.items():
        result[name + ".sizeof"] = C.sizeof(struct)
        result[name + ".alignof"] = C.alignment(struct)
        for field, typ in struct._fields_:
            result[f"{name}.{field}.offset"] = getattr(struct, field).offset
            result[f"{name}.{field}.sizeof"] = C.sizeof(typ)
    return result


def _require(value, typ):
    if not isinstance(value, typ):
        raise TypeError(f"expected {typ.__name__}, got {type(value).__name__}")


def _float(value, name):
    value = float(value)
    if not math.isfinite(value) or not math.isfinite(C.c_float(value).value):
        raise ValueError(f"{name} must be finite float32")
    return value


def _vector(values, name, *, allow_empty=False):
    values = list(values)
    if not allow_empty and not values:
        raise ValueError(f"{name} must be nonempty")
    _integer(len(values), name + " length", 0, 0x7FFFFFFF)
    return (C.c_float * len(values))(*(_float(v, name) for v in values))


def _matrix(values, name, *, dim=None, empty_dim=None):
    values = list(values)
    if dim is not None:
        dim = _integer(dim, "dim", 1, 0x7FFFFFFF)
        if len(values) % dim:
            raise ValueError(f"{name}: flat length must be a multiple of dim")
        count, flat = len(values) // dim, values
    elif values:
        rows = [list(row) for row in values]
        dim = len(rows[0])
        if not dim or any(len(row) != dim for row in rows):
            raise ValueError(f"{name}: nonempty equal-width rows required")
        count, flat = len(rows), [x for row in rows for x in row]
    else:
        count, dim, flat = 0, empty_dim, []
    if not count and empty_dim is None:
        raise ValueError(f"{name} must be nonempty")
    if empty_dim is not None and dim != empty_dim:
        raise ValueError(f"{name}: row width must equal query width {empty_dim}")
    _integer(count, name + " rows", 0, 0x7FFFFFFF)
    return _vector(flat, name, allow_empty=True), count, dim


def _path(path):
    path = os.fsencode(os.fspath(path))
    if not path or b"\0" in path:
        raise ValueError("checkpoint path must be nonempty and contain no NUL")
    return path


class Native:
    """An ABI-checked libnotorch handle. Construction performs no training."""

    def __init__(self, path=None):
        self.path = find_library(os.fspath(path) if path is not None else None)
        self.lib = C.CDLL(self.path)
        try:
            version = self.lib.nt_spa_binding_version
            version.argtypes, version.restype = [], C.c_uint32
            layout = self.lib.nt_spa_binding_layout
            layout.argtypes, layout.restype = [C.c_char_p], C.c_size_t
        except AttributeError as exc:
            raise ABIError("libnotorch lacks the SPA ABI manifest; rebuild make shared") from exc
        if version() != 1:
            raise ABIError("unsupported SPA binding ABI version")
        for key, expected in _layout().items():
            actual = layout(key.encode("ascii"))
            if actual != expected:
                raise ABIError(f"SPA ABI mismatch for {key}: C={actual}, Python={expected}")
        signatures = {
            "config_default": ([C.POINTER(Config)], None),
            "init": ([C.POINTER(AgentState), C.POINTER(Config)], C.c_int),
            "validate": ([C.POINTER(AgentState)], C.c_int),
            "perceive": ([C.POINTER(C.c_float), C.c_uint32, C.c_uint32, C.c_uint32,
                           C.c_float, C.c_float, C.c_uint32, C.POINTER(Observation)], C.c_int),
            "legacy": ([C.POINTER(Observation), C.POINTER(Action)], C.c_int),
            "select": ([C.POINTER(AgentState), C.POINTER(Observation), C.POINTER(Decision)], C.c_int),
            "choose": ([C.POINTER(AgentState), C.POINTER(Observation), C.POINTER(Decision)], C.c_int),
            "observe": ([C.POINTER(AgentState), C.c_uint64, C.POINTER(Action),
                         C.POINTER(Consequence), C.POINTER(Receipt)], C.c_int),
            "cancel": ([C.POINTER(AgentState), C.c_uint64], C.c_int),
            "imitate": ([C.POINTER(AgentState), C.POINTER(Observation), C.c_int,
                         C.POINTER(C.c_float)], C.c_int),
            "reset_memory": ([C.POINTER(AgentState)], C.c_int),
            "set_policy": ([C.POINTER(AgentState), C.POINTER(Policy)], C.c_int),
            "capture_experience": ([C.POINTER(AgentState), C.POINTER(Observation),
                                    C.POINTER(Experience)], C.c_int),
            "score_experience": ([C.POINTER(AgentState), C.POINTER(Experience),
                                  C.POINTER(Readout)], C.c_int),
            "fit_comparison": ([C.POINTER(AgentState), C.POINTER(Experience),
                                C.POINTER(Comparison), C.c_float,
                                C.POINTER(ComparisonReceipt)], C.c_int),
            "fit_repeated": ([C.POINTER(AgentState), C.POINTER(Experience),
                              C.POINTER(Comparison), C.c_uint32, C.c_float,
                              C.POINTER(ComparisonReceipt)], C.c_int),
            "fit_conditioned": ([C.POINTER(AgentState), C.POINTER(Experience),
                                 C.POINTER(Comparison), C.c_uint32, C.c_float,
                                 C.c_float, C.POINTER(ConditionedReceipt)], C.c_int),
            "save": ([C.POINTER(AgentState), C.c_char_p], C.c_int),
            "load": ([C.POINTER(AgentState), C.c_char_p], C.c_int),
            "hash": ([C.POINTER(AgentState)], C.c_uint64),
        }
        extra = {
            "nt_spa_observation_validate": ([C.POINTER(Observation)], C.c_int),
            "nt_spa_action_validate": ([C.POINTER(Action), C.POINTER(Observation)], C.c_int),
            "nt_spa_experience_validate": ([C.POINTER(Experience)], C.c_int),
            "nt_spa_embed_sentence": ([C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_float),
                                       C.c_int, C.c_int, C.c_float, C.POINTER(C.c_float)], None),
            "nt_spa_connectedness": ([C.POINTER(C.c_float), C.c_int,
                                      C.POINTER(C.c_float), C.c_int], C.c_float),
            "nt_spa_modulate_logits": ([C.POINTER(C.c_float), C.c_int, C.c_float, C.c_float], None),
        }
        extra.update({"nt_spa_agent_" + name: signature for name, signature in signatures.items()})
        try:
            for name, (args, result) in extra.items():
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = args, result
        except AttributeError as exc:
            raise ABIError(f"libnotorch lacks a required SPA function: {exc}") from exc

    def config(self):
        value = Config()
        self.lib.nt_spa_agent_config_default(C.byref(value))
        return value

    def embed_sentence(self, tokens, embeddings, *, dim=None, alpha=0.85):
        matrix, vocab, width = _matrix(embeddings, "embeddings", dim=dim)
        tokens = [_integer(t, "token", 0, vocab - 1) for t in tokens]
        _integer(len(tokens), "token count", 0, 0x7FFFFFFF)
        ids = (C.c_int * len(tokens))(*tokens)
        out = (C.c_float * width)()
        self.lib.nt_spa_embed_sentence(ids, len(ids), matrix, vocab, width,
                                      _float(alpha, "alpha"), out)
        return out

    def connectedness(self, query, history, *, dim=None):
        query = _vector(query, "query")
        history, count, _ = _matrix(history, "history", dim=dim, empty_dim=len(query))
        return self.lib.nt_spa_connectedness(query, len(query), history, count)

    def modulate_logits(self, logits, connectedness, strength=0.3):
        """Return a new native float32 array; the supplied logits remain owned by the host."""
        out = _vector(logits, "logits", allow_empty=True)
        self.lib.nt_spa_modulate_logits(out, len(out), _float(connectedness, "connectedness"),
                                       _float(strength, "strength"))
        return out

    def perceive(self, embeddings, target, *, dim=None, phase_lock=0.0,
                 temperature=1.0, reseeds=0):
        matrix, count, width = _matrix(embeddings, "embeddings", dim=dim)
        _integer(count, "sentence count", 1, 4096)
        _integer(width, "dim", 1, 4096)
        target = _integer(target, "target", 0, count - 1)
        out = Observation()
        _check("perceive", self.lib.nt_spa_agent_perceive(
            matrix, count, width, target, _float(phase_lock, "phase_lock"),
            _float(temperature, "temperature"), _integer(reseeds, "reseeds", 0, 1000000), C.byref(out)))
        return out

    def legacy(self, observation):
        _require(observation, Observation)
        action = Action()
        _check("legacy", self.lib.nt_spa_agent_legacy(C.byref(observation), C.byref(action)))
        return action

    def validate_action(self, action, observation):
        _require(action, Action)
        _require(observation, Observation)
        _check("validate_action", self.lib.nt_spa_action_validate(C.byref(action), C.byref(observation)))


_default = None
_default_lock = threading.Lock()


def default_native():
    global _default
    with _default_lock:
        if _default is None:
            _default = Native()
        return _default


class Agent:
    """One persistent native life. Methods serialize access to this life."""

    def __init__(self, config=None, *, native=None):
        self.native = native or default_native()
        self._lock = threading.RLock()
        self._state = AgentState()
        if config is not None:
            _require(config, Config)
        _check("init", self.native.lib.nt_spa_agent_init(
            C.byref(self._state), C.byref(config) if config is not None else None))

    @property
    def state(self):
        """A detached snapshot; mutation here cannot change the running life."""
        with self._lock:
            return self._state.copy()

    @property
    def hash(self):
        with self._lock:
            return self.native.lib.nt_spa_agent_hash(C.byref(self._state))

    def _call(self, name, *args):
        with self._lock:
            _check(name, getattr(self.native.lib, "nt_spa_agent_" + name)(C.byref(self._state), *args))

    def select(self, observation):
        _require(observation, Observation)
        out = Decision()
        self._call("select", C.byref(observation), C.byref(out))
        return out

    def choose(self, observation):
        _require(observation, Observation)
        out = Decision()
        self._call("choose", C.byref(observation), C.byref(out))
        return out

    def observe(self, sequence, executed_action, consequence):
        _require(executed_action, Action)
        _require(consequence, Consequence)
        out = Receipt()
        self._call("observe", _integer(sequence, "sequence", 0, 0xFFFFFFFFFFFFFFFF),
                   C.byref(executed_action), C.byref(consequence), C.byref(out))
        return out

    def cancel(self, sequence):
        self._call("cancel", _integer(sequence, "sequence", 0, 0xFFFFFFFFFFFFFFFF))

    def imitate(self, observation, label):
        _require(observation, Observation)
        label = ActionKind(label)
        loss = C.c_float()
        self._call("imitate", C.byref(observation), label, C.byref(loss))
        return loss.value

    def reset_memory(self):
        self._call("reset_memory")

    def set_policy(self, policy):
        _require(policy, Policy)
        self._call("set_policy", C.byref(policy))

    def capture(self, observation):
        _require(observation, Observation)
        out = Experience()
        self._call("capture_experience", C.byref(observation), C.byref(out))
        return out

    def score(self, experience):
        _require(experience, Experience)
        out = Readout()
        self._call("score_experience", C.byref(experience), C.byref(out))
        return out

    def fit_comparison(self, experience, comparison, learning_rate):
        _require(experience, Experience)
        _require(comparison, Comparison)
        out = ComparisonReceipt()
        self._call("fit_comparison", C.byref(experience), C.byref(comparison),
                   _float(learning_rate, "learning_rate"), C.byref(out))
        return out

    def fit_repeated(self, experience, comparisons, learning_rate):
        """Fit mean native rewards from 1..64 paired outcomes of one state.

        Each repeat is a complete Comparison. Clipping, averaging, target
        construction and gradients all execute in C.
        """
        _require(experience, Experience)
        comparisons = list(comparisons)
        _integer(len(comparisons), "repeat count", 1, MAX_REPEATS)
        for comparison in comparisons:
            _require(comparison, Comparison)
        array = (Comparison * len(comparisons))(*comparisons)
        out = ComparisonReceipt()
        self._call("fit_repeated", C.byref(experience), array, len(array),
                   _float(learning_rate, "learning_rate"), C.byref(out))
        return out

    def fit_conditioned(self, experience, comparisons, learning_rate, scale_floor):
        """Fit KEEP-relative repeated credit divided by its action-effect span.

        Native C averages clipped rewards, subtracts KEEP, and divides each
        target by max(scale_floor, largest absolute valid target). The nested
        comparison receipt retains raw mean rewards and normalized targets;
        the enclosing receipt records the actual double-precision scale.
        """
        _require(experience, Experience)
        comparisons = list(comparisons)
        _integer(len(comparisons), "repeat count", 1, MAX_REPEATS)
        for comparison in comparisons:
            _require(comparison, Comparison)
        array = (Comparison * len(comparisons))(*comparisons)
        out = ConditionedReceipt()
        self._call("fit_conditioned", C.byref(experience), array, len(array),
                   _float(learning_rate, "learning_rate"),
                   _float(scale_floor, "scale_floor"), C.byref(out))
        return out

    def validate(self):
        self._call("validate")

    def save(self, path):
        self._call("save", _path(path))

    def load(self, path):
        """Replace this life transactionally from a native canonical checkpoint."""
        self._call("load", _path(path))
        return self

    @classmethod
    def from_file(cls, path, *, native=None):
        return cls(native=native).load(path)


__all__ = ["Native", "Agent", "Config", "Mode", "ActionKind", "Action", "Metrics",
           "Consequence", "Observation", "Policy", "Decision", "Receipt", "AgentState",
           "Experience", "Alternative", "Comparison", "Readout", "ComparisonReceipt",
           "ConditionedReceipt",
           "Status", "Error", "ABIError", "NO_SOURCE", "MAX_REPEATS", "default_native"]
