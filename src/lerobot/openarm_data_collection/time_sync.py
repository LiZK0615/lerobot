from collections import deque
from dataclasses import dataclass, replace
import statistics

from lerobot.cameras.orbbec import OrbbecFrame

from .types import RecordingSnapshot, SynchronizedSample, TimedVector


class ClockMappingError(ValueError):
    pass


class DeviceClockMapper:
    def __init__(self, window_size: int = 60, max_drift_ppm: float = 2000) -> None:
        self._samples: deque[tuple[int, int]] = deque(maxlen=window_size)
        self._max_drift = max_drift_ppm / 1_000_000
        self._last_device: int | None = None
        self._last_mapped: int | None = None

    def update(self, device_timestamp_us: int, received_monotonic_ns: int) -> int:
        device_ns = device_timestamp_us * 1000
        if self._last_device is not None and device_ns <= self._last_device:
            raise ClockMappingError("device timestamp is not strictly increasing")
        self._samples.append((device_ns, received_monotonic_ns))
        if len(self._samples) < 2:
            mapped = received_monotonic_ns
        else:
            first_x, first_y = self._samples[0]
            last_x, last_y = self._samples[-1]
            slope = (last_y - first_y) / (last_x - first_x)
            if not 1.0 - self._max_drift <= slope <= 1.0 + self._max_drift:
                raise ClockMappingError("device clock drift exceeds configured limit")
            offsets = [y - slope * x for x, y in self._samples]
            mapped = round(slope * device_ns + statistics.median(offsets))
        if self._last_mapped is not None and mapped <= self._last_mapped:
            raise ClockMappingError("mapped timestamp is not strictly increasing")
        self._last_device, self._last_mapped = device_ns, mapped
        return mapped


@dataclass(frozen=True)
class SyncHealth:
    fatal: bool
    reason: str | None
    failure_started_ns: int | None


class SampleSynchronizer:
    CAMERA_NAMES = ("head", "left_wrist", "right_wrist")

    def __init__(
        self,
        camera_skew_ns: int = 35_000_000,
        camera_age_ns: int = 100_000_000,
        state_age_ns: int = 50_000_000,
        action_age_ns: int = 50_000_000,
        fatal_after_ns: int = 500_000_000,
    ) -> None:
        self.camera_skew_ns = camera_skew_ns
        self.camera_age_ns = camera_age_ns
        self.state_age_ns = state_age_ns
        self.action_age_ns = action_age_ns
        self.fatal_after_ns = fatal_after_ns
        self._cameras = {name: deque(maxlen=120) for name in self.CAMERA_NAMES}
        self._snapshots: deque[RecordingSnapshot] = deque(maxlen=400)
        self._last_consumed = {name: -1 for name in self.CAMERA_NAMES}
        self._failure_started_ns: int | None = None
        self._failure_reason: str | None = None

    def push_camera(self, name: str, frame: OrbbecFrame) -> None:
        if name not in self._cameras:
            raise ValueError(f"unknown camera: {name}")
        self._cameras[name].append(frame)

    def push_snapshot(self, snapshot: RecordingSnapshot) -> None:
        self._snapshots.append(snapshot)

    @staticmethod
    def _nearest(items, target_ns, timestamp):
        return min(items, key=lambda item: abs(timestamp(item) - target_ns), default=None)

    def _fail(self, now_ns: int, reason: str) -> None:
        if self._failure_reason != reason:
            self._failure_started_ns = now_ns
            self._failure_reason = reason

    def select(self, sample_monotonic_ns: int) -> SynchronizedSample | None:
        selected: dict[str, OrbbecFrame] = {}
        for name in self.CAMERA_NAMES:
            candidates = [
                frame for frame in self._cameras[name]
                if frame.sequence > self._last_consumed[name] and frame.mapped_monotonic_ns is not None
            ]
            candidate = self._nearest(candidates, sample_monotonic_ns, lambda item: item.mapped_monotonic_ns)
            if candidate is None or abs(sample_monotonic_ns - candidate.mapped_monotonic_ns) > self.camera_age_ns:
                self._fail(sample_monotonic_ns, f"{name} missing or stale")
                return None
            selected[name] = candidate
        times = [frame.mapped_monotonic_ns for frame in selected.values()]
        if max(times) - min(times) > self.camera_skew_ns:
            self._fail(sample_monotonic_ns, "camera skew exceeded")
            return None
        reference_ns = sorted(times)[1]
        snapshot = self._nearest(self._snapshots, reference_ns, lambda item: item.sent_monotonic_ns)
        if snapshot is None or snapshot.state is None or snapshot.action is None:
            self._fail(sample_monotonic_ns, "state or action missing")
            return None
        state_age = abs(reference_ns - snapshot.state.received_monotonic_ns)
        action_age = abs(reference_ns - snapshot.action.received_monotonic_ns)
        if state_age > self.state_age_ns or action_age > self.action_age_ns:
            self._fail(sample_monotonic_ns, "state or action stale")
            return None
        for name, frame in selected.items():
            self._last_consumed[name] = frame.sequence
        self._failure_started_ns = None
        self._failure_reason = None
        return SynchronizedSample(
            sample_monotonic_ns,
            selected["head"], selected["left_wrist"], selected["right_wrist"],
            snapshot.state, snapshot.action, snapshot.command,
            max(times) - min(times), state_age, action_age,
        )

    def health(self, now_ns: int) -> SyncHealth:
        started = self._failure_started_ns
        return SyncHealth(
            fatal=started is not None and now_ns - started >= self.fatal_after_ns,
            reason=self._failure_reason,
            failure_started_ns=started,
        )
