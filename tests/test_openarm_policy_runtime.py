import pytest

from lerobot.openarm_data_collection.types import JOINT_NAMES
from lerobot.openarm_policy_runtime import TABLE_READY, ReturnToReadyMotion


def state(values):
    return dict(zip(JOINT_NAMES, values, strict=True))


def test_return_motion_starts_at_current_and_finishes_at_ready():
    motion = ReturnToReadyMotion()
    current = state([0.0] * 16)
    motion.start(current, 10.0)
    assert motion.command(10.0) == current
    assert motion.command(10.0 + motion.duration_sec) == pytest.approx(state(TABLE_READY))
    assert motion.trajectory_complete(10.0 + motion.duration_sec)


def test_return_motion_requires_feedback_within_tolerance():
    motion = ReturnToReadyMotion(tolerance=0.1)
    assert motion.target_reached(state(TABLE_READY))
    outside = list(TABLE_READY)
    outside[0] += 0.11
    assert not motion.target_reached(state(outside))


def test_gripper_velocity_can_set_duration():
    start = list(TABLE_READY)
    start[7] = 0.04
    motion = ReturnToReadyMotion(max_gripper_velocity_m_s=0.01)
    motion.start(state(start), 0.0)
    assert motion.duration_sec == pytest.approx(7.5)
