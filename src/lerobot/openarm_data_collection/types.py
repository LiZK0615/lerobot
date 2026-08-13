from dataclasses import dataclass

from lerobot.cameras.orbbec import OrbbecFrame

JOINT_NAMES = tuple(
    [f"openarm_left_joint{index}" for index in range(1, 8)]
    + ["openarm_left_finger_joint1"]
    + [f"openarm_right_joint{index}" for index in range(1, 8)]
    + ["openarm_right_finger_joint1"]
)


@dataclass(frozen=True)
class TimedVector:
    values: tuple[float, ...]
    received_monotonic_ns: int
    ros_stamp_ns: int | None


@dataclass(frozen=True)
class RecordingSnapshot:
    sequence: int
    sent_monotonic_ns: int
    state: TimedVector | None
    action: TimedVector | None
    leader: TimedVector | None


@dataclass(frozen=True)
class SynchronizedSample:
    sample_monotonic_ns: int
    head: OrbbecFrame
    left_wrist: OrbbecFrame
    right_wrist: OrbbecFrame
    state: TimedVector
    action: TimedVector
    leader: TimedVector | None
    camera_skew_ns: int
    state_age_ns: int
    action_age_ns: int
