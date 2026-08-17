from argparse import Namespace

import pytest

from lerobot.openarm_data_collection.types import JOINT_NAMES
from lerobot.openarm_policy_runtime import TABLE_READY
from lerobot.scripts.lerobot_openarm_policy import (
    ReturnInterruptedError,
    _parser,
    _hold_current_position,
    _return_to_ready,
)


class FakeRobot:
    def __init__(self, state):
        self.state = dict(state)
        self.actions = []
        self.get_state_calls = 0
        self.fail_state_after = None

    def get_state(self):
        self.get_state_calls += 1
        if self.fail_state_after is not None and self.get_state_calls > self.fail_state_after:
            raise TimeoutError("ROS joint state is stale")
        return dict(self.state)

    def send_action(self, action):
        self.actions.append(dict(action))
        self.state = dict(action)


class FakeClient:
    def __init__(self, state):
        self.robot = FakeRobot(state)
        self.paused = 0

    def pause_session(self):
        self.paused += 1


class FakeKeyboard:
    def __init__(self, key):
        self.key = key

    def poll(self):
        key, self.key = self.key, None
        return key


def _args():
    return Namespace(
        max_joint_velocity_rad_s=10.0,
        max_gripper_velocity_m_s=10.0,
        return_minimum_duration_sec=0.001,
        ready_tolerance=0.10,
        return_timeout_sec=1.0,
        fps=1000,
    )


def test_parser_defaults_to_smolvla_policy():
    args = _parser().parse_args(["--camera-config=cameras.yaml", "--policy-path=checkpoint"])

    assert args.policy_type == "smolvla"


def test_parser_accepts_pi0_policy():
    args = _parser().parse_args(
        ["--camera-config=cameras.yaml", "--policy-type=pi0", "--policy-path=checkpoint"]
    )

    assert args.policy_type == "pi0"


def test_hold_sends_measured_state():
    state = dict(zip(JOINT_NAMES, [0.25] * 16, strict=True))
    client = FakeClient(state)

    _hold_current_position(client)

    assert client.paused == 1
    assert client.robot.actions == [state]


def test_return_to_ready_reaches_named_pose():
    state = dict(zip(JOINT_NAMES, [0.0] * 16, strict=True))
    client = FakeClient(state)

    _return_to_ready(client, _args())

    assert client.paused == 1
    assert tuple(client.robot.actions[-1][name] for name in JOINT_NAMES) == TABLE_READY
    assert client.robot.get_state_calls > 2


def test_return_to_ready_detects_stale_feedback_before_trajectory_finishes():
    state = dict(zip(JOINT_NAMES, [0.0] * 16, strict=True))
    client = FakeClient(state)
    client.robot.fail_state_after = 2

    with pytest.raises(TimeoutError, match="ROS joint state is stale"):
        _return_to_ready(client, _args())

    assert client.paused == 1
    assert len(client.robot.actions) == 1


def test_space_interrupts_return_and_holds_measured_position():
    state = dict(zip(JOINT_NAMES, [0.25] * 16, strict=True))
    client = FakeClient(state)

    with pytest.raises(ReturnInterruptedError) as raised:
        _return_to_ready(client, _args(), FakeKeyboard(" "))

    assert not raised.value.exit_requested
    assert client.paused == 2
    assert client.robot.actions == [state]
