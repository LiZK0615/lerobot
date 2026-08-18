"""Bimanual Mini clutch and preset workflow used by OpenArm data collection."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import queue
import socket
import time
from typing import Any

from lerobot.scripts.lerobot_openarm_mini_ros2_teleop import (
    MotionPhase,
    PresetMotion,
    build_bimanual_datagram,
    load_preset_config,
    starting_sequence,
)

SIDES = ("left", "right")
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
JOINT_ACTION_NAMES = tuple(f"{side}_{joint}.pos" for side in SIDES for joint in JOINT_NAMES)


class WorkflowState(str, Enum):
    CONNECTING = "CONNECTING"
    PREPARING = "PREPARING"
    READY = "READY"
    RECORDING_MANUAL = "RECORDING_MANUAL"
    AUTO_RETURNING = "AUTO_RETURNING"
    AWAITING_DECISION = "AWAITING_DECISION"
    RESETTING_DISCARD = "RESETTING_DISCARD"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"
    FAILED = "FAILED"


class WorkflowCommand(str, Enum):
    START_RECORDING = "START_RECORDING"
    RESET_DISCARD = "RESET_DISCARD"
    FINISH_DECISION = "FINISH_DECISION"
    SHUTDOWN = "SHUTDOWN"
    STOP = "STOP"


class TorqueRequest(str, Enum):
    UNCHANGED = "UNCHANGED"
    ENABLE = "ENABLE"
    DISABLE = "DISABLE"


@dataclass(frozen=True)
class WorkflowOutput:
    goal: dict[str, float] | None = None
    left_torque: TorqueRequest = TorqueRequest.UNCHANGED
    right_torque: TorqueRequest = TorqueRequest.UNCHANGED


@dataclass(frozen=True)
class TeleopWorkerConfig:
    left_port: str
    right_port: str
    teleop_id: str
    preset_config: str
    host: str = "127.0.0.1"
    udp_port: int = 15000
    fps: float = 30.0


class JointIdleDetector:
    def __init__(self, duration_sec: float, joint_delta_deg: float) -> None:
        if not math.isfinite(duration_sec) or duration_sec <= 0.0:
            raise ValueError("duration_sec must be a finite positive number")
        if not math.isfinite(joint_delta_deg) or joint_delta_deg <= 0.0:
            raise ValueError("joint_delta_deg must be a finite positive number")
        self.duration_sec = duration_sec
        self.joint_delta_deg = joint_delta_deg
        self._samples: deque[tuple[float, tuple[float, ...]]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, positions: Mapping[str, float], now: float) -> bool:
        values = tuple(float(positions[name]) for name in JOINT_ACTION_NAMES)
        self._samples.append((float(now), values))
        cutoff = float(now) - self.duration_sec
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if not self._samples or float(now) - self._samples[0][0] < self.duration_sec:
            return False
        columns = zip(*(sample[1] for sample in self._samples), strict=True)
        return all(max(column) - min(column) <= self.joint_delta_deg for column in columns)


class CollectionTeleopWorkflow:
    def __init__(self, preset_config: Any) -> None:
        self.config = preset_config
        self.motion = PresetMotion(preset_config)
        self.state = WorkflowState.CONNECTING
        self.operator_engaged = False
        self.gripper_open = dict.fromkeys(SIDES, False)
        self.idle_detector = JointIdleDetector(
            preset_config.auto_return_idle_duration_sec,
            preset_config.auto_return_idle_joint_delta_deg,
        )

    def initialize(self, current: Mapping[str, float], now: float) -> WorkflowOutput:
        if self.state is not WorkflowState.CONNECTING:
            raise RuntimeError("workflow is already initialized")
        self.motion.start_prepare(current, now)
        self.state = WorkflowState.PREPARING
        return WorkflowOutput(
            goal=dict(current),
            left_torque=TorqueRequest.ENABLE,
            right_torque=TorqueRequest.ENABLE,
        )

    def handle_command(
        self,
        command: WorkflowCommand,
        current: Mapping[str, float],
        now: float,
    ) -> WorkflowOutput:
        if command is WorkflowCommand.START_RECORDING:
            if self.state is not WorkflowState.READY:
                raise RuntimeError("START_RECORDING requires READY")
            self.operator_engaged = False
            self.gripper_open = dict.fromkeys(SIDES, False)
            self.idle_detector.reset()
            self.state = WorkflowState.RECORDING_MANUAL
            return WorkflowOutput(
                goal=self.motion.step(current, now),
                left_torque=TorqueRequest.ENABLE,
                right_torque=TorqueRequest.ENABLE,
            )

        if command is WorkflowCommand.RESET_DISCARD:
            if self.state not in {WorkflowState.READY, WorkflowState.RECORDING_MANUAL}:
                raise RuntimeError(f"RESET_DISCARD is not allowed from {self.state.value}")
            self.motion.start_prepare(current, now)
            self.operator_engaged = False
            self.gripper_open = dict.fromkeys(SIDES, False)
            self.idle_detector.reset()
            self.state = WorkflowState.RESETTING_DISCARD
            return WorkflowOutput(
                goal=dict(current),
                left_torque=TorqueRequest.ENABLE,
                right_torque=TorqueRequest.ENABLE,
            )

        if command is WorkflowCommand.FINISH_DECISION:
            if self.state is not WorkflowState.AWAITING_DECISION:
                raise RuntimeError("FINISH_DECISION requires AWAITING_DECISION")
            self.state = WorkflowState.READY
            return WorkflowOutput(self.motion.step(current, now))

        if command is WorkflowCommand.SHUTDOWN:
            if self.state is not WorkflowState.READY:
                raise RuntimeError("SHUTDOWN requires READY")
            self.motion.start_clearance_only(current, now)
            self.state = WorkflowState.SHUTTING_DOWN
            return WorkflowOutput(
                goal=dict(current),
                left_torque=TorqueRequest.ENABLE,
                right_torque=TorqueRequest.ENABLE,
            )

        if command is WorkflowCommand.STOP:
            self.state = WorkflowState.SHUTDOWN_COMPLETE
            return WorkflowOutput(
                left_torque=TorqueRequest.DISABLE,
                right_torque=TorqueRequest.DISABLE,
            )
        raise RuntimeError(f"unsupported workflow command {command}")

    def tick(self, current: Mapping[str, float], now: float) -> WorkflowOutput:
        if self.state in {
            WorkflowState.PREPARING,
            WorkflowState.RESETTING_DISCARD,
            WorkflowState.SHUTTING_DOWN,
        }:
            goal = self.motion.step(current, now)
            if self.motion.phase == MotionPhase.HOLDING:
                if self.state is WorkflowState.SHUTTING_DOWN:
                    self.state = WorkflowState.SHUTDOWN_COMPLETE
                    return WorkflowOutput(
                        goal=goal,
                        left_torque=TorqueRequest.DISABLE,
                        right_torque=TorqueRequest.DISABLE,
                    )
                self.state = WorkflowState.READY
            return WorkflowOutput(goal)

        if self.state is WorkflowState.READY:
            return WorkflowOutput(self.motion.step(current, now))

        if self.state is WorkflowState.RECORDING_MANUAL:
            current_gripper_open = {
                side: self._gripper_is_open(current, side) for side in SIDES
            }
            left_torque = TorqueRequest.UNCHANGED
            right_torque = TorqueRequest.UNCHANGED
            goal = None

            for side in SIDES:
                request = TorqueRequest.UNCHANGED
                if current_gripper_open[side] and not self.gripper_open[side]:
                    request = TorqueRequest.DISABLE
                    if not self.operator_engaged:
                        self.motion.release()
                    self.operator_engaged = True
                elif not current_gripper_open[side] and self.gripper_open[side]:
                    request = TorqueRequest.ENABLE
                    goal = dict(current)
                if side == "left":
                    left_torque = request
                else:
                    right_torque = request

            both_just_closed = any(self.gripper_open.values()) and not any(
                current_gripper_open.values()
            )
            self.gripper_open = current_gripper_open

            if any(current_gripper_open.values()):
                self.idle_detector.reset()
                return WorkflowOutput(
                    goal=goal,
                    left_torque=left_torque,
                    right_torque=right_torque,
                )
            if not self.operator_engaged:
                self.idle_detector.reset()
                return WorkflowOutput(
                    goal=self.motion.step(current, now),
                    left_torque=left_torque,
                    right_torque=right_torque,
                )
            if both_just_closed:
                self.idle_detector.reset()
            if self.idle_detector.update(current, now):
                self.motion.start_direct_ready(current, now)
                self.state = WorkflowState.AUTO_RETURNING
                return WorkflowOutput(
                    goal=dict(current),
                    left_torque=TorqueRequest.ENABLE,
                    right_torque=TorqueRequest.ENABLE,
                )
            return WorkflowOutput(
                goal=goal,
                left_torque=left_torque,
                right_torque=right_torque,
            )

        if self.state is WorkflowState.AUTO_RETURNING:
            goal = self.motion.step(current, now)
            if self.motion.phase == MotionPhase.HOLDING:
                self.state = WorkflowState.AWAITING_DECISION
            return WorkflowOutput(goal)

        if self.state is WorkflowState.AWAITING_DECISION:
            return WorkflowOutput(self.motion.step(current, now))
        return WorkflowOutput()

    def _gripper_is_open(self, current: Mapping[str, float], side: str) -> bool:
        threshold = self.config.gripper_closed_threshold_deg
        return float(current[f"{side}_gripper.pos"]) < threshold


def _joint_goals(action: Mapping[str, float], side: str) -> dict[str, float]:
    return {
        f"{joint}.pos": float(action[f"{side}_{joint}.pos"])
        for joint in JOINT_NAMES
    }


def run_teleop_worker(
    config: TeleopWorkerConfig,
    command_queue: Any,
    status_queue: Any,
) -> None:
    """Own both Mini serial buses and stream their measured actions to ROS 2."""
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.bi_openarm_mini.config_bi_openarm_mini import BiOpenArmMiniConfig
    from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfigBase
    from lerobot.utils.robot_utils import precise_sleep

    teleop_config = BiOpenArmMiniConfig(
        id=config.teleop_id,
        left_arm_config=OpenArmMiniConfigBase(port=config.left_port, side="left"),
        right_arm_config=OpenArmMiniConfigBase(port=config.right_port, side="right"),
    )
    teleop = make_teleoperator_from_config(teleop_config)
    preset = load_preset_config(config.preset_config)
    workflow = CollectionTeleopWorkflow(preset)
    sequence = starting_sequence()
    torque_enabled = dict.fromkeys(SIDES, False)
    last_request_id = 0
    last_status: tuple[str, int, str | None] | None = None
    last_status_time = 0.0
    stop_requested = False
    fatal_error: str | None = None

    def publish_status(error: str | None = None, force: bool = False) -> None:
        nonlocal last_status, last_status_time
        status = (workflow.state.value, last_request_id, error)
        now = time.monotonic()
        if force or status != last_status or now - last_status_time >= 0.2:
            status_queue.put(
                {"state": workflow.state.value, "request_id": last_request_id, "error": error}
            )
            last_status = status
            last_status_time = now

    def set_joint_torque(
        side: str,
        enabled: bool,
        seed: Mapping[str, float] | None = None,
    ) -> None:
        if enabled == torque_enabled[side]:
            return
        arm = teleop.left_arm if side == "left" else teleop.right_arm
        if enabled:
            if seed is None:
                raise RuntimeError("enabling leader torque requires a measured seed")
            arm.write_goal_positions(_joint_goals(seed, side))
            try:
                arm.bus.enable_torque(list(JOINT_NAMES))
            except Exception:
                arm.bus.disable_torque(list(JOINT_NAMES))
                raise
        else:
            arm.bus.disable_torque(list(JOINT_NAMES))
        torque_enabled[side] = enabled

    def apply_output(output: WorkflowOutput, current: Mapping[str, float]) -> None:
        requests = {
            "left": output.left_torque,
            "right": output.right_torque,
        }
        for side, request in requests.items():
            if request is TorqueRequest.DISABLE:
                set_joint_torque(side, False)
            elif request is TorqueRequest.ENABLE:
                set_joint_torque(side, True, output.goal or current)
        if output.goal is not None:
            for side in SIDES:
                if torque_enabled[side]:
                    arm = teleop.left_arm if side == "left" else teleop.right_arm
                    arm.write_goal_positions(_joint_goals(output.goal, side))

    status_queue.put({"state": WorkflowState.CONNECTING.value, "request_id": 0, "error": None})
    try:
        if not teleop.left_arm.calibration or not teleop.right_arm.calibration:
            raise RuntimeError("both openarms_mini_left/right calibration files are required")
        teleop.connect(calibrate=False)
        current = teleop.get_action()
        apply_output(workflow.initialize(current, time.monotonic()), current)
        publish_status(force=True)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            while not stop_requested:
                loop_started = time.perf_counter()
                current = teleop.get_action()
                while True:
                    try:
                        request = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    last_request_id = int(request["request_id"])
                    command = WorkflowCommand(request["command"])
                    try:
                        output = workflow.handle_command(command, current, time.monotonic())
                        apply_output(output, current)
                        if command is WorkflowCommand.STOP:
                            stop_requested = True
                    except Exception as exc:
                        fatal_error = str(exc)
                        workflow.state = WorkflowState.FAILED
                        stop_requested = True
                        publish_status(fatal_error, force=True)
                        break

                output = workflow.tick(current, time.monotonic())
                apply_output(output, current)
                udp_socket.sendto(
                    build_bimanual_datagram(current, sequence, time.monotonic_ns()),
                    (config.host, config.udp_port),
                )
                sequence += 1
                publish_status(fatal_error)
                if workflow.state is WorkflowState.SHUTDOWN_COMPLETE:
                    stop_requested = True
                precise_sleep(max(1.0 / config.fps - (time.perf_counter() - loop_started), 0.0))
        publish_status(fatal_error, force=True)
    except BaseException as exc:
        workflow.state = WorkflowState.FAILED
        publish_status(str(exc), force=True)
    finally:
        for arm in (teleop.left_arm, teleop.right_arm):
            if arm.is_connected:
                try:
                    arm.bus.disable_torque(list(JOINT_NAMES))
                finally:
                    arm.disconnect()
