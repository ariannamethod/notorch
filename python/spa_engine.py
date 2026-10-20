"""Transactional host runtime for the native Sentence Phonon Agent.

A host supplies ``host_id``, ``snapshot() -> bytes``, ``restore(bytes)``,
``observe(target) -> (Observation, Metrics)`` and ``execute(Action) -> cost``.
Snapshots contain its complete mutable sentence field, model state and RNG.
Observe is pure; restore validates before replacing state. Execute may advance
the trajectory even for KEEP. Host side effects must be reversible by restore.
Engine owns access to that host for the duration of each operation.

The single-file checkpoint contains a native canonical Agent life and the host
snapshot, with explicit lengths and a SHA-256 over the complete payload. It
contains no executable serialization. Native C owns all policy arithmetic.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import SPA


_MAGIC = b"NTSPAENG"
_HEADER = struct.Struct("<8sIIQQ")  # magic, version, metadata, native life, host
_VERSION = 1
_MAX_META = 4096
_MAX_LIFE = 1024 * 1024
_DEFAULT_HOST_LIMIT = 256 * 1024 * 1024


class CheckpointError(ValueError):
    """The complete engine checkpoint failed structural or identity checks."""


class RollbackError(RuntimeError):
    """Host restore failed; load a valid complete checkpoint before continuing."""

    def __init__(self, operation_error, restore_error):
        self.operation_error = operation_error
        self.restore_error = restore_error
        super().__init__(f"host rollback failed after {type(operation_error).__name__}: "
                         f"{restore_error}")


@dataclass(frozen=True)
class StepResult:
    """Detached native values from one committed action/consequence step."""

    decision: SPA.Decision
    receipt: SPA.Receipt | None
    before: SPA.Metrics
    after: SPA.Metrics

    @property
    def action(self):
        return self.decision.action


class Engine:
    """Own a host loop, native learning, failure rollback and complete resume.

    With no Agent/config supplied the native policy runs in LEARNED mode.
    ``max_host_bytes`` bounds snapshots and checkpoint loading; increase it
    explicitly for bodies whose complete mutable state exceeds 256 MiB.
    Idle Agent lives are required. A pending low-level action belongs to its
    original host and cannot be adopted by this runtime.
    """

    def __init__(self, host, agent=None, *, native=None, config=None,
                 max_host_bytes=_DEFAULT_HOST_LIMIT):
        import SPA
        self._spa = SPA
        if agent is not None:
            if not isinstance(agent, SPA.Agent):
                raise TypeError("agent must be SPA.Agent")
            if config is not None:
                raise ValueError("config cannot be supplied with an existing agent")
            if native is not None and native is not agent.native:
                raise ValueError("native must be the existing agent's Native instance")
        else:
            native = native or SPA.default_native()
            config = config if config is not None else SPA.Config.default(
                native=native, mode=SPA.Mode.LEARNED)
            agent = SPA.Agent(config, native=native)
        self.agent = agent
        self.native = agent.native
        self.host = host
        self._host_id = self._get_host_id()
        for name in ("snapshot", "restore", "observe", "execute"):
            if not callable(getattr(host, name, None)):
                raise TypeError(f"host.{name} must be callable")
        self.max_host_bytes = SPA._integer(max_host_bytes, "max_host_bytes", 1, 1 << 40)
        self._lock = threading.RLock()
        self._busy = False
        self._broken = False
        self._steps = 0
        with self.agent._lock:
            self._idle()

    @property
    def steps(self):
        """Committed active steps; disabled previews leave this count unchanged."""
        with self._lock:
            return self._steps

    def _get_host_id(self):
        value = getattr(self.host, "host_id", None)
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
            raise ValueError("host.host_id must be a nonempty UTF-8 string of at most 512 bytes")
        return value

    def _idle(self):
        self.agent.validate()
        if self.agent._state.pending:
            raise self._spa.Error("engine requires an idle life", self._spa.Status.PENDING)

    @contextmanager
    def _operation(self, *, recovering=False):
        with self._lock, self.agent._lock:
            if self._busy:
                raise RuntimeError("Engine operations cannot be called from host callbacks")
            if self._broken and not recovering:
                raise RuntimeError("host rollback failed; load a complete checkpoint to recover")
            self._idle()
            if self._get_host_id() != self._host_id:
                raise ValueError("host_id changed while the Engine was running")
            self._busy = True
            try:
                yield
            finally:
                self._busy = False

    def _snapshot(self):
        snapshot = self.host.snapshot()
        if not isinstance(snapshot, bytes):
            raise TypeError("host.snapshot() must return bytes")
        if len(snapshot) > self.max_host_bytes:
            raise ValueError("host snapshot exceeds max_host_bytes")
        return snapshot

    def _restore_after_error(self, snapshot, state, error):
        # The native copy is always restored, even if the host refuses rollback.
        try:
            self.host.restore(snapshot)
        except BaseException as restore_error:
            self._broken = True
            raise RollbackError(error, restore_error) from restore_error
        finally:
            self.agent._state = state

    def _measure(self, target):
        observation, metrics = self.host.observe(target)
        SPA = self._spa
        SPA._require(observation, SPA.Observation)
        SPA._require(metrics, SPA.Metrics)
        # Copy before invoking another host callback: hosts may reuse buffers.
        observation, metrics = observation.copy(), metrics.copy()
        SPA._check("host observation", self.native.lib.nt_spa_observation_validate(
            C.byref(observation)))
        if observation.sentence_index != target:
            raise ValueError("host observation target differs from the requested target")
        for name, _ in SPA.Metrics._fields_:
            value = getattr(metrics, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"host metric {name} must be finite and in [0, 1]")
        return observation, metrics

    def step(self, target):
        """Execute one action and acquire its measured consequence atomically.

        Failure restores the complete host snapshot, Agent history, weights,
        counters and RNG. A successful disabled step is a pure KEEP preview:
        no execute callback, learning, history or step-count mutation occurs.
        """
        target = self._spa._integer(target, "target", 0, 4095)
        with self._operation():
            if self.agent._state.config.mode == self._spa.Mode.DISABLED:
                observation, before = self._measure(target)
                decision = self.agent.select(observation)
                return StepResult(decision, None, before, before.copy())
            snapshot, old_state = self._snapshot(), self.agent.state
            try:
                observation, before = self._measure(target)
                decision = self.agent.choose(observation)
                self.native.validate_action(decision.action, observation)
                cost = self.host.execute(decision.action.copy())
                if isinstance(cost, bool):
                    raise TypeError("host execution cost must be a number, not bool")
                cost = self._spa._float(cost, "host execution cost")
                if not 0 <= cost <= 1:
                    raise ValueError("host execution cost must be in [0, 1]")
                after_observation, after = self._measure(target)
                if after_observation.sentence_count != observation.sentence_count:
                    raise ValueError("host action changed sentence-field size during its credit window")
                consequence = self._spa.Consequence(before, after, cost)
                receipt = self.agent.observe(decision.sequence, decision.action, consequence)
            except BaseException as error:
                self._restore_after_error(snapshot, old_state, error)
                raise
            self._steps += 1
            return StepResult(decision, receipt, before, after)

    def run(self, targets):
        """Yield committed steps for an iterable of host-chosen sentence indices."""
        for target in targets:
            yield self.step(target)

    def save(self, path):
        """Atomically replace one file containing the Agent and complete host.

        File fsync precedes replacement, and parent-directory fsync follows it
        on POSIX. An error after replacement leaves the new file installed.
        """
        path = Path(os.fsdecode(self._spa._path(path)))
        with self._operation():
            host = self._snapshot()
            with tempfile.TemporaryDirectory(prefix="spa-engine-life-") as directory:
                life_path = Path(directory) / "agent.life"
                self.agent.save(life_path)
                life = life_path.read_bytes()
            if not 0 < len(life) <= _MAX_LIFE:
                raise CheckpointError("native life exceeds checkpoint format bounds")
            metadata = json.dumps({"host_id": self._host_id, "steps": self._steps},
                                  ensure_ascii=True, separators=(",", ":"),
                                  sort_keys=True).encode("utf-8")
            header = _HEADER.pack(_MAGIC, _VERSION, len(metadata), len(life), len(host))
            digest = hashlib.sha256()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", prefix=".spa-engine-",
                                                 dir=path.parent, delete=False) as file:
                    temporary = file.name
                    for part in (header, metadata, life, host):
                        file.write(part)
                        digest.update(part)
                    file.write(digest.digest())
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
                temporary = None
                if os.name == "posix":
                    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            finally:
                if temporary is not None:
                    os.unlink(temporary)

    def _read_checkpoint(self, path):
        with open(path, "rb") as file:
            header = file.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise CheckpointError("truncated engine checkpoint header")
            magic, version, meta_size, life_size, host_size = _HEADER.unpack(header)
            if magic != _MAGIC or version != _VERSION:
                raise CheckpointError("unsupported engine checkpoint format")
            if not 0 < meta_size <= _MAX_META or not 0 < life_size <= _MAX_LIFE:
                raise CheckpointError("engine checkpoint length outside format bounds")
            if host_size > self.max_host_bytes:
                raise CheckpointError("host checkpoint exceeds max_host_bytes")
            expected_size = _HEADER.size + meta_size + life_size + host_size + 32
            if os.fstat(file.fileno()).st_size != expected_size:
                raise CheckpointError("engine checkpoint length mismatch")
            metadata = file.read(meta_size)
            life = file.read(life_size)
            host = file.read(host_size)
            digest = hashlib.sha256()
            for part in (header, metadata, life, host):
                digest.update(part)
            if digest.digest() != file.read(32) or file.read(1):
                raise CheckpointError("engine checkpoint checksum mismatch")
        try:
            values = json.loads(metadata)
        except (ValueError, UnicodeError) as error:
            raise CheckpointError("invalid engine checkpoint metadata") from error
        if (not isinstance(values, dict) or set(values) != {"host_id", "steps"}
                or values["host_id"] != self._host_id
                or type(values["steps"]) is not int
                or not 0 <= values["steps"] <= 0xFFFFFFFFFFFFFFFF):
            raise CheckpointError("engine checkpoint host identity or counters mismatch")
        return life, host, values["steps"]

    def load(self, path):
        """Validate and resume both states, or leave the running pair unchanged.

        The host must restore its snapshot transactionally. If a broken host
        also refuses rollback, RollbackError blocks further steps/saves until
        a complete checkpoint is loaded successfully.
        """
        path = os.fsdecode(self._spa._path(path))
        with self._operation(recovering=True):
            life, host, steps = self._read_checkpoint(path)
            with tempfile.TemporaryDirectory(prefix="spa-engine-load-") as directory:
                life_path = Path(directory) / "agent.life"
                life_path.write_bytes(life)
                candidate = self._spa.Agent.from_file(life_path, native=self.native)
            if candidate.state.pending:
                raise CheckpointError("engine checkpoint contains an uncommitted decision")
            old_host = None if self._broken else self._snapshot()
            old_state = self.agent.state
            try:
                self.host.restore(host)
            except BaseException as error:
                if old_host is not None:
                    self._restore_after_error(old_host, old_state, error)
                raise
            self.agent._state = candidate.state
            self._steps = steps
            self._broken = False
        return self
