from dataclasses import dataclass
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    READY = "READY"
    RECORDING = "RECORDING"
    INVALID = "INVALID"
    FINALIZING = "FINALIZING"
    EXITED = "EXITED"


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionStatus:
    state: SessionState
    frames: int
    skipped: int
    reason: str | None


class RecordingSession:
    def __init__(
        self, sink: Any, synchronizer: Any, task: str, fps: int = 30,
        min_episode_sec: float = 1.0, max_episode_sec: float = 120.0
    ) -> None:
        self.sink, self.synchronizer, self.task = sink, synchronizer, task
        self.period_ns = round(1_000_000_000 / fps)
        self.min_episode_ns = round(min_episode_sec * 1_000_000_000)
        self.max_episode_ns = round(max_episode_sec * 1_000_000_000)
        self.state = SessionState.READY
        self.started_ns: int | None = None
        self.next_sample_ns: int | None = None
        self.frames = self.skipped = 0
        self.reason: str | None = None

    def mark_invalid(self, reason: str) -> None:
        if self.state is SessionState.RECORDING:
            self.state, self.reason = SessionState.INVALID, reason

    def handle_key(self, key: str, now_ns: int) -> None:
        if key not in "rsdq": return
        if key == "r":
            if self.state is not SessionState.READY: raise InvalidTransition("r requires READY")
            self.sink.begin_episode(self.task)
            self.state, self.started_ns, self.next_sample_ns = SessionState.RECORDING, now_ns, now_ns + self.period_ns
            self.frames = self.skipped = 0; self.reason = None
        elif key == "s":
            if self.state is not SessionState.RECORDING: raise InvalidTransition("s requires RECORDING")
            if now_ns - self.started_ns < self.min_episode_ns: raise InvalidTransition("episode is too short")
            self.sink.save_episode(); self.state = SessionState.READY
        elif key == "d":
            if self.state not in (SessionState.RECORDING, SessionState.INVALID): raise InvalidTransition("d requires active episode")
            self.sink.discard_episode(); self.state, self.reason = SessionState.READY, None
        elif key == "q":
            if self.state is not SessionState.READY: raise InvalidTransition("q requires READY")
            self.state = SessionState.FINALIZING; self.sink.finalize(); self.state = SessionState.EXITED

    def tick(self, now_ns: int) -> None:
        if self.state is not SessionState.RECORDING or now_ns < self.next_sample_ns: return
        if now_ns - self.started_ns > self.max_episode_ns:
            self.reason = "maximum episode duration reached"; return
        target = self.next_sample_ns
        self.next_sample_ns += self.period_ns
        if now_ns >= self.next_sample_ns:
            self.next_sample_ns = now_ns + self.period_ns
        sample = self.synchronizer.select(target)
        if sample is None:
            self.skipped += 1
            health = self.synchronizer.health(now_ns)
            if health.fatal: self.mark_invalid(health.reason or "data source timeout")
            return
        self.sink.add_sample(sample); self.frames += 1

    def status(self) -> SessionStatus:
        return SessionStatus(self.state, self.frames, self.skipped, self.reason)
