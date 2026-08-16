import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from lerobot.openarm_data_collection.types import JOINT_NAMES

TABLE_READY = (
    0.72,
    0.0,
    0.0,
    1.15,
    0.0,
    0.0,
    -1.10,
    0.0,
    -0.72,
    0.0,
    0.0,
    1.15,
    0.0,
    0.0,
    1.10,
    0.0,
)


class PolicyRunState(str, Enum):
    STARTING = "STARTING"
    RETURNING = "RETURNING"
    READY = "READY"
    INFERENCE = "INFERENCE"
    HOLD = "HOLD"
    STOPPED = "STOPPED"


def quintic_scale(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _vector(values: Mapping[str, Real]) -> tuple[float, ...]:
    result = tuple(float(values[name]) for name in JOINT_NAMES)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("joint state must contain finite values")
    return result


@dataclass
class ReturnToReadyMotion:
    max_joint_velocity_rad_s: float = 0.2
    max_gripper_velocity_m_s: float = 0.01
    minimum_duration_sec: float = 3.0
    tolerance: float = 0.10

    def __post_init__(self) -> None:
        if (
            min(
                self.max_joint_velocity_rad_s,
                self.max_gripper_velocity_m_s,
                self.minimum_duration_sec,
                self.tolerance,
            )
            <= 0
        ):
            raise ValueError("motion parameters must be positive")
        self._start: tuple[float, ...] | None = None
        self._started_at: float | None = None
        self.duration_sec = 0.0

    def start(self, current: Mapping[str, Real], now: float) -> None:
        self._start = _vector(current)
        self._started_at = float(now)
        durations = []
        for index, (source, target) in enumerate(zip(self._start, TABLE_READY, strict=True)):
            velocity = self.max_gripper_velocity_m_s if index in (7, 15) else self.max_joint_velocity_rad_s
            durations.append(1.875 * abs(target - source) / velocity)
        self.duration_sec = max(self.minimum_duration_sec, *durations)

    def command(self, now: float) -> dict[str, float]:
        if self._start is None or self._started_at is None:
            raise RuntimeError("return motion has not started")
        scale = quintic_scale((float(now) - self._started_at) / self.duration_sec)
        values = tuple(
            source + scale * (target - source)
            for source, target in zip(self._start, TABLE_READY, strict=True)
        )
        return dict(zip(JOINT_NAMES, values, strict=True))

    def trajectory_complete(self, now: float) -> bool:
        if self._started_at is None:
            return False
        return float(now) - self._started_at >= self.duration_sec

    def target_reached(self, current: Mapping[str, Real]) -> bool:
        values = _vector(current)
        return all(
            abs(value - target) <= self.tolerance for value, target in zip(values, TABLE_READY, strict=True)
        )
