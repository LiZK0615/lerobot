from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class SessionState(str, Enum):
    READY = "READY"
    ARMING = "ARMING"
    RECORDING = "RECORDING"
    INVALID = "INVALID"
    FINALIZING = "FINALIZING"
    EXITED = "EXITED"


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionStatus:
    state: SessionState
    episode_index: int | None
    frames: int
    sync_skipped: int
    deadline_missed: int
    reason: str | None
    elapsed_sec: float
    effective_fps: float
    window_fps: float
    arming_elapsed_sec: float
    arming_successful: int
    arming_required: int
    sync_failures: dict[str, int]


class RecordingSession:
    def __init__(
        self, sink: Any, synchronizer: Any, task: str, fps: int = 30,
        min_episode_sec: float = 1.0, max_episode_sec: float = 120.0,
        arming_timeout_sec: float = 3.0, arming_stable_sec: float = 1.0,
        min_effective_fps_ratio: float = 0.90, fps_check_grace_sec: float = 3.0,
        fps_failure_duration_sec: float = 2.0, fps_window_sec: float = 1.0,
        sync_wait_grace_ms: float = 12.0,
    ) -> None:
        self.sink, self.synchronizer, self.task = sink, synchronizer, task
        self.fps = fps
        self.period_ns = round(1_000_000_000 / fps)
        self.min_episode_ns = round(min_episode_sec * 1_000_000_000)
        self.max_episode_ns = round(max_episode_sec * 1_000_000_000)
        self.arming_timeout_ns = round(arming_timeout_sec * 1_000_000_000)
        self.arming_stable_ns = round(arming_stable_sec * 1_000_000_000)
        self.minimum_fps = fps * min_effective_fps_ratio
        self.arming_required = math.ceil(fps * arming_stable_sec * min_effective_fps_ratio)
        self.fps_check_grace_ns = round(fps_check_grace_sec * 1_000_000_000)
        self.fps_failure_duration_ns = round(fps_failure_duration_sec * 1_000_000_000)
        self.fps_window_ns = round(fps_window_sec * 1_000_000_000)
        self.sync_wait_grace_ns = round(sync_wait_grace_ms * 1_000_000)
        self.state = SessionState.READY
        self.started_ns: int | None = None
        self.next_sample_ns: int | None = None
        self.arming_started_ns: int | None = None
        self.frames = self.sync_skipped = self.deadline_missed = 0
        self.reason: str | None = None
        self.episode_index: int | None = None
        self._recent_frames: deque[int] = deque()
        self._arming_successes: deque[int] = deque()
        self._low_fps_started_ns: int | None = None
        self._pending_sample_target_ns: int | None = None
        self._pending_sample_deadline_ns: int | None = None
        self._sync_failures: Counter[str] = Counter()

    def mark_invalid(self, reason: str) -> None:
        if self.state is SessionState.RECORDING:
            self.state, self.reason = SessionState.INVALID, reason

    def _reset_counters(self) -> None:
        self.frames = self.sync_skipped = self.deadline_missed = 0
        self._recent_frames.clear()
        self._arming_successes.clear()
        self._low_fps_started_ns = None
        self._pending_sample_target_ns = None
        self._pending_sample_deadline_ns = None
        self._sync_failures.clear()

    def _start_recording(self, now_ns: int) -> None:
        self.episode_index = self.sink.begin_episode(self.task)
        self.state = SessionState.RECORDING
        self.started_ns = now_ns
        self.next_sample_ns = now_ns + self.period_ns
        self.reason = None
        self._reset_counters()

    def _average_fps(self, now_ns: int) -> float:
        if self.started_ns is None or now_ns <= self.started_ns:
            return 0.0
        return self.frames / ((now_ns - self.started_ns) / 1_000_000_000)

    def _window_fps(self, now_ns: int) -> float:
        if self.fps_window_ns <= 0:
            return self._average_fps(now_ns)
        cutoff = now_ns - self.fps_window_ns
        while self._recent_frames and self._recent_frames[0] <= cutoff:
            self._recent_frames.popleft()
        return len(self._recent_frames) / (self.fps_window_ns / 1_000_000_000)

    def _fps_failure_reason(self, actual_fps: float) -> str:
        return (
            f"effective FPS too low: actual={actual_fps:.1f} required={self.minimum_fps:.1f} "
            f"target={self.fps:.1f}; sync_skipped={self.sync_skipped} "
            f"deadline_missed={self.deadline_missed}"
        )

    def handle_key(self, key: str, now_ns: int) -> None:
        if key not in "rsdq":
            return
        if key == "r":
            if self.state is not SessionState.READY:
                raise InvalidTransition("r requires READY")
            self.state = SessionState.ARMING
            self.arming_started_ns = now_ns
            self.next_sample_ns = now_ns
            self.reason = None
            self.episode_index = self.sink.total_episodes
            self._reset_counters()
        elif key == "s":
            if self.state is not SessionState.RECORDING:
                raise InvalidTransition("s requires RECORDING")
            if now_ns - self.started_ns < self.min_episode_ns:
                raise InvalidTransition("episode is too short")
            average_fps = self._average_fps(now_ns)
            if average_fps < self.minimum_fps:
                reason = self._fps_failure_reason(average_fps)
                self.mark_invalid(reason)
                raise InvalidTransition(reason)
            self.sink.save_episode()
            self.state = SessionState.READY
            self.episode_index = None
        elif key == "d":
            if self.state is SessionState.ARMING:
                self.state, self.reason = SessionState.READY, None
                self.episode_index = None
            elif self.state in (SessionState.RECORDING, SessionState.INVALID):
                self.sink.discard_episode()
                self.state, self.reason = SessionState.READY, None
                self.episode_index = None
            else:
                raise InvalidTransition("d requires active episode")
        elif key == "q":
            if self.state is not SessionState.READY:
                raise InvalidTransition("q requires READY")
            self.state = SessionState.FINALIZING
            self.sink.finalize()
            self.state = SessionState.EXITED

    def _tick_arming(self, now_ns: int) -> None:
        if self._pending_sample_target_ns is None and now_ns < self.next_sample_ns:
            return
        if self._pending_sample_target_ns is None:
            due_count = (now_ns - self.next_sample_ns) // self.period_ns + 1
            target = self.next_sample_ns + (due_count - 1) * self.period_ns
            self.next_sample_ns += due_count * self.period_ns
            self._pending_sample_target_ns = target
            self._pending_sample_deadline_ns = now_ns + self.sync_wait_grace_ns
        else:
            target = self._pending_sample_target_ns
        sample = self.synchronizer.select(target)
        if sample is None:
            health = self.synchronizer.health(now_ns)
            if now_ns < self._pending_sample_deadline_ns:
                return
            self._sync_failures[health.category or "unknown"] += 1
        else:
            self._arming_successes.append(now_ns)
        self._pending_sample_target_ns = None
        self._pending_sample_deadline_ns = None

        cutoff = now_ns - self.arming_stable_ns
        while self._arming_successes and self._arming_successes[0] < cutoff:
            self._arming_successes.popleft()
        arming_elapsed_ns = now_ns - self.arming_started_ns
        if arming_elapsed_ns >= self.arming_stable_ns and len(self._arming_successes) >= self.arming_required:
            self._start_recording(now_ns)
            return
        if now_ns - self.arming_started_ns >= self.arming_timeout_ns:
            self.state = SessionState.READY
            self.episode_index = None
            self.reason = (
                "arming timeout: synchronized sources below threshold: "
                f"successful={len(self._arming_successes)} required={self.arming_required} "
                f"sync_failures={dict(self._sync_failures)}"
            )

    def _tick_recording(self, now_ns: int) -> None:
        if self._pending_sample_target_ns is None and now_ns < self.next_sample_ns:
            return
        if now_ns - self.started_ns > self.max_episode_ns:
            self.mark_invalid("maximum episode duration reached")
            return
        if self._pending_sample_target_ns is None:
            due_count = (now_ns - self.next_sample_ns) // self.period_ns + 1
            self.deadline_missed += max(0, due_count - 1)
            target = self.next_sample_ns + (due_count - 1) * self.period_ns
            self.next_sample_ns += due_count * self.period_ns
            self._pending_sample_target_ns = target
            self._pending_sample_deadline_ns = now_ns + self.sync_wait_grace_ns
        else:
            target = self._pending_sample_target_ns
        sample = self.synchronizer.select(target)
        if sample is None:
            health = self.synchronizer.health(now_ns)
            if now_ns < self._pending_sample_deadline_ns:
                return
            self.sync_skipped += 1
            self._sync_failures[health.category or "unknown"] += 1
            self._pending_sample_target_ns = None
            self._pending_sample_deadline_ns = None
            if health.fatal:
                self.mark_invalid(health.reason or "data source timeout")
            return
        self._pending_sample_target_ns = None
        self._pending_sample_deadline_ns = None
        self.sink.add_sample(sample)
        self.frames += 1
        self._recent_frames.append(now_ns)

        elapsed_ns = now_ns - self.started_ns
        if elapsed_ns < self.fps_check_grace_ns or self.fps_window_ns <= 0:
            return
        window_fps = self._window_fps(now_ns)
        if window_fps < self.minimum_fps:
            if self._low_fps_started_ns is None:
                self._low_fps_started_ns = now_ns
            elif now_ns - self._low_fps_started_ns >= self.fps_failure_duration_ns:
                self.mark_invalid(self._fps_failure_reason(window_fps))
        else:
            self._low_fps_started_ns = None

    def tick(self, now_ns: int) -> None:
        if self.state is SessionState.ARMING:
            self._tick_arming(now_ns)
        elif self.state is SessionState.RECORDING:
            self._tick_recording(now_ns)

    def status(self, now_ns: int | None = None) -> SessionStatus:
        elapsed_sec = arming_elapsed_sec = 0.0
        if now_ns is not None and self.state in (SessionState.RECORDING, SessionState.INVALID):
            if self.started_ns is not None:
                elapsed_sec = max(0.0, (now_ns - self.started_ns) / 1_000_000_000)
        if now_ns is not None and self.state is SessionState.ARMING and self.arming_started_ns is not None:
            arming_elapsed_sec = max(0.0, (now_ns - self.arming_started_ns) / 1_000_000_000)
        effective_fps = self.frames / elapsed_sec if elapsed_sec > 0.0 else 0.0
        window_fps = self._window_fps(now_ns) if now_ns is not None else 0.0
        return SessionStatus(
            self.state, self.episode_index, self.frames, self.sync_skipped,
            self.deadline_missed, self.reason, elapsed_sec, effective_fps,
            window_fps, arming_elapsed_sec, len(self._arming_successes),
            self.arming_required, dict(self._sync_failures),
        )
