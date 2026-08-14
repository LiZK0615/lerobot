#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stream mapped bimanual OpenArm Mini targets to the ROS 2 teleop bridge."""

import ipaddress
import json
import math
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

POSITION_NAMES = tuple([f"joint_{index}" for index in range(1, 8)] + ["gripper"])
ACTION_NAMES = tuple(
    f"{side}_{name}.pos" for side in ("left", "right") for name in POSITION_NAMES
)
DEFAULT_PRESET_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config/openarm_mini_joint_presets.yaml"


class MotionPhase(str):
    IDLE = "idle"
    MOVING = "moving"
    PAUSING = "pausing"
    HOLDING = "holding"


@dataclass(frozen=True)
class PresetConfig:
    max_joint_velocity_rad_s: float
    minimum_segment_duration_sec: float
    waypoint_pause_sec: float
    target_tolerance_rad: float
    gripper_closed_threshold_deg: float
    operator_idle_duration_sec: float
    operator_idle_joint_delta_deg: float
    prepare_sequence: tuple[str, ...]
    waypoints: dict[str, tuple[float, ...]]


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def load_preset_config(path: str | Path) -> PresetConfig:
    config_path = Path(path).expanduser()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("preset config schema_version must be 1")

    positive_fields = (
        "max_joint_velocity_rad_s",
        "minimum_segment_duration_sec",
        "target_tolerance_rad",
    )
    values = {field: _finite_float(payload.get(field), field) for field in positive_fields}
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("preset velocity, duration, and tolerance must be positive")
    pause = _finite_float(payload.get("waypoint_pause_sec"), "waypoint_pause_sec")
    if pause < 0.0:
        raise ValueError("waypoint_pause_sec must be non-negative")
    gripper_closed_threshold_deg = _finite_float(
        payload.get("gripper_closed_threshold_deg", -3.0),
        "gripper_closed_threshold_deg",
    )
    operator_idle_duration_sec = _finite_float(
        payload.get("operator_idle_duration_sec", 1.5),
        "operator_idle_duration_sec",
    )
    operator_idle_joint_delta_deg = _finite_float(
        payload.get("operator_idle_joint_delta_deg", 2.0),
        "operator_idle_joint_delta_deg",
    )
    if operator_idle_duration_sec <= 0.0 or operator_idle_joint_delta_deg <= 0.0:
        raise ValueError("operator idle duration and joint delta must be positive")
    if not -65.0 < gripper_closed_threshold_deg <= 0.0:
        raise ValueError("gripper_closed_threshold_deg must be in (-65, 0]")

    raw_waypoints = payload.get("waypoints")
    if not isinstance(raw_waypoints, dict) or not raw_waypoints:
        raise ValueError("waypoints must be a non-empty mapping")
    waypoints: dict[str, tuple[float, ...]] = {}
    for name, raw_waypoint in raw_waypoints.items():
        if not isinstance(name, str) or not name or not isinstance(raw_waypoint, dict):
            raise ValueError("waypoint names and values are invalid")
        positions: list[float] = []
        for side in ("left", "right"):
            raw_side = raw_waypoint.get(side)
            if not isinstance(raw_side, list) or len(raw_side) != len(POSITION_NAMES):
                raise ValueError(f"waypoints.{name}.{side} must contain 8 positions")
            positions.extend(
                _finite_float(value, f"waypoints.{name}.{side}[{index}]")
                for index, value in enumerate(raw_side)
            )
            gripper = positions[-1]
            if not 0.0 <= gripper <= 0.044:
                raise ValueError(f"waypoints.{name}.{side} gripper must be in [0, 0.044] m")
        waypoints[name] = tuple(positions)

    raw_sequence = payload.get("prepare_sequence")
    if not isinstance(raw_sequence, list) or not raw_sequence:
        raise ValueError("prepare_sequence must be a non-empty list")
    sequence = tuple(raw_sequence)
    if any(not isinstance(name, str) or name not in waypoints for name in sequence):
        raise ValueError("prepare_sequence references an unknown waypoint")
    if "table_clearance" not in waypoints or "table_ready" not in waypoints:
        raise ValueError("table_clearance and table_ready waypoints are required")

    return PresetConfig(
        max_joint_velocity_rad_s=values["max_joint_velocity_rad_s"],
        minimum_segment_duration_sec=values["minimum_segment_duration_sec"],
        waypoint_pause_sec=pause,
        target_tolerance_rad=values["target_tolerance_rad"],
        gripper_closed_threshold_deg=gripper_closed_threshold_deg,
        operator_idle_duration_sec=operator_idle_duration_sec,
        operator_idle_joint_delta_deg=operator_idle_joint_delta_deg,
        prepare_sequence=sequence,
        waypoints=waypoints,
    )


def ros_positions_to_mapped_action(positions: tuple[float, ...]) -> dict[str, float]:
    if len(positions) != len(ACTION_NAMES):
        raise ValueError("preset must contain 16 ROS positions")
    action: dict[str, float] = {}
    for index, (name, raw_value) in enumerate(zip(ACTION_NAMES, positions, strict=True)):
        value = _finite_float(raw_value, name)
        if index % len(POSITION_NAMES) == len(POSITION_NAMES) - 1:
            if not 0.0 <= value <= 0.044:
                raise ValueError(f"{name} must be in [0, 0.044] m")
            action[name] = -65.0 * value / 0.044
        else:
            action[name] = math.degrees(value)
    return action


def quintic_scale(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


class PresetMotion:
    def __init__(self, config: PresetConfig) -> None:
        self.config = config
        self.phase = MotionPhase.IDLE
        self.waypoint_name: str | None = None
        self.segment_duration_sec = 0.0
        self._segment_started_at = 0.0
        self._pause_started_at = 0.0
        self._segment_start: dict[str, float] | None = None
        self._target: dict[str, float] | None = None
        self._remaining: list[str] = []

    @property
    def is_active(self) -> bool:
        return self.phase != MotionPhase.IDLE

    def _mapped_waypoint(self, name: str) -> dict[str, float]:
        return ros_positions_to_mapped_action(self.config.waypoints[name])

    def _within_tolerance(self, current: Mapping[str, float], target: Mapping[str, float]) -> bool:
        tolerance_deg = math.degrees(self.config.target_tolerance_rad)
        return all(
            abs(float(current[name]) - target[name]) <= tolerance_deg
            for name in ACTION_NAMES
            if "gripper" not in name
        )

    def _duration(self, start: Mapping[str, float], target: Mapping[str, float]) -> float:
        joint_deltas_rad = [
            math.radians(abs(float(start[name]) - target[name]))
            for name in ACTION_NAMES
            if "gripper" not in name
        ]
        # A quintic scale reaches a peak derivative of 1.875 / duration.
        velocity_limited = 1.875 * max(joint_deltas_rad, default=0.0) / self.config.max_joint_velocity_rad_s
        return max(self.config.minimum_segment_duration_sec, velocity_limited)

    def _begin_segment(self, name: str, current: Mapping[str, float], now: float) -> None:
        self.waypoint_name = name
        self._segment_start = {key: float(current[key]) for key in ACTION_NAMES}
        self._target = self._mapped_waypoint(name)
        self.segment_duration_sec = self._duration(self._segment_start, self._target)
        self._segment_started_at = float(now)
        self.phase = MotionPhase.MOVING

    def _start_motion(self, names: tuple[str, ...], current: Mapping[str, float], now: float) -> None:
        if self.phase in (MotionPhase.MOVING, MotionPhase.PAUSING):
            raise ValueError("preset motion is already active")
        self._remaining = list(names[1:])
        self._begin_segment(names[0], current, now)

    def start_prepare(self, current: Mapping[str, float], now: float) -> None:
        self._start_motion(self.config.prepare_sequence, current, now)

    def start_clearance_only(self, current: Mapping[str, float], now: float) -> None:
        self._start_motion(("table_clearance",), current, now)

    def start_ready_only(self, current: Mapping[str, float], now: float) -> None:
        clearance = self._mapped_waypoint("table_clearance")
        if not self._within_tolerance(current, clearance):
            raise ValueError("table_ready requires the current Mini pose to match table_clearance")
        self._start_motion(("table_ready",), current, now)

    def abort(self, current: Mapping[str, float]) -> None:
        if self.phase == MotionPhase.IDLE:
            raise ValueError("no preset motion to abort")
        self._remaining.clear()
        self.waypoint_name = "aborted_hold"
        self._target = {key: float(current[key]) for key in ACTION_NAMES}
        self.phase = MotionPhase.HOLDING

    def release(self) -> None:
        self.phase = MotionPhase.IDLE
        self.waypoint_name = None
        self._segment_start = self._target = None
        self._remaining.clear()

    def step(self, current: Mapping[str, float], now: float) -> dict[str, float] | None:
        if self.phase == MotionPhase.IDLE:
            return None
        assert self._target is not None
        if self.phase == MotionPhase.HOLDING:
            return dict(self._target)
        if self.phase == MotionPhase.PAUSING:
            if float(now) - self._pause_started_at < self.config.waypoint_pause_sec:
                return dict(self._target)
            name = self._remaining.pop(0)
            self._begin_segment(name, current, now)
            return dict(self._segment_start)

        assert self._segment_start is not None
        elapsed = max(0.0, float(now) - self._segment_started_at)
        scale = quintic_scale(elapsed / self.segment_duration_sec)
        command = {
            name: self._segment_start[name]
            + scale * (self._target[name] - self._segment_start[name])
            for name in ACTION_NAMES
        }
        if elapsed >= self.segment_duration_sec and self._within_tolerance(current, self._target):
            if self._remaining:
                self.phase = MotionPhase.PAUSING
                self._pause_started_at = float(now)
            else:
                self.phase = MotionPhase.HOLDING
            return dict(self._target)
        return command


def starting_sequence() -> int:
    """Choose a restart-safe sequence origin on the shared host."""
    return time.monotonic_ns()


def _require_non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def build_bimanual_datagram(
    action: Mapping[str, Any],
    sequence: int,
    sent_monotonic_ns: int,
) -> bytes:
    """Build one atomic datagram from already-mapped left and right actions."""
    sequence = _require_non_negative_integer(sequence, "sequence")
    sent_monotonic_ns = _require_non_negative_integer(
        sent_monotonic_ns, "sent_monotonic_ns"
    )

    sides: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        positions: dict[str, float] = {}
        for name in POSITION_NAMES:
            key = f"{side}_{name}.pos"
            value = action.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{key} must be a finite number")
            positions[name] = float(value)
        sides[side] = positions

    return json.dumps(
        {
            "version": 1,
            "sequence": sequence,
            "sent_monotonic_ns": sent_monotonic_ns,
            **sides,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> None:
    # Imports stay inside main so protocol tests do not require hardware extras.
    import draccus

    from lerobot.openarm_data_collection.terminal import TerminalKeys
    from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
    from lerobot.teleoperators.bi_openarm_mini.config_bi_openarm_mini import (
        BiOpenArmMiniConfig,
    )
    from lerobot.utils.robot_utils import precise_sleep

    @dataclass
    class OpenArmMiniRos2TeleopConfig:
        teleop: TeleoperatorConfig
        host: str = "127.0.0.1"
        udp_port: int = 15000
        fps: float = 30.0
        preset_config: str = str(DEFAULT_PRESET_CONFIG_PATH)

    @draccus.wrap()
    def run(cfg: OpenArmMiniRos2TeleopConfig) -> None:
        if not isinstance(cfg.teleop, BiOpenArmMiniConfig):
            raise ValueError("--teleop.type must be bi_openarm_mini")

        address = ipaddress.ip_address(cfg.host)
        if address.version != 4 or not address.is_loopback:
            raise ValueError("--host must be an IPv4 loopback address")
        if not 1 <= cfg.udp_port <= 65535:
            raise ValueError("--udp_port must be between 1 and 65535")
        if not math.isfinite(cfg.fps) or cfg.fps <= 0.0:
            raise ValueError("--fps must be a finite positive number")

        preset_config = load_preset_config(cfg.preset_config)
        motion = PresetMotion(preset_config)
        teleop = make_teleoperator_from_config(cfg.teleop)
        destination = (cfg.host, cfg.udp_port)
        sequence = starting_sequence()
        torque_enabled = False
        last_motion_status: tuple[str, str | None] = (motion.phase, motion.waypoint_name)

        def enable_preset_torque(current: Mapping[str, float]) -> None:
            nonlocal torque_enabled
            if torque_enabled:
                return
            # Seed the servo goal registers before enabling torque to avoid a jump
            # towards stale goals left by an earlier process.
            teleop.send_feedback(dict(current))
            try:
                teleop.left_arm.enable_torque()
                teleop.right_arm.enable_torque()
            except Exception:
                teleop.left_arm.disable_torque()
                teleop.right_arm.disable_torque()
                raise
            torque_enabled = True

        def disable_preset_torque() -> None:
            nonlocal torque_enabled
            teleop.left_arm.disable_torque()
            teleop.right_arm.disable_torque()
            torque_enabled = False

        def report_motion_status() -> None:
            nonlocal last_motion_status
            status = (motion.phase, motion.waypoint_name)
            if status == last_motion_status:
                return
            if motion.phase == MotionPhase.MOVING:
                print(
                    f"[PRESET] moving_to={motion.waypoint_name} "
                    f"duration={motion.segment_duration_sec:.1f}s",
                    flush=True,
                )
            elif motion.phase == MotionPhase.PAUSING:
                print(
                    f"[PRESET] reached={motion.waypoint_name} "
                    f"pause={preset_config.waypoint_pause_sec:.1f}s",
                    flush=True,
                )
            elif motion.phase == MotionPhase.HOLDING:
                print(
                    f"[PRESET] holding={motion.waypoint_name}; "
                    "support both Minis and press SPACE to resume manual teleoperation",
                    flush=True,
                )
            elif motion.phase == MotionPhase.IDLE:
                print("[MANUAL] Mini torque released; live teleoperation active", flush=True)
            last_motion_status = status

        try:
            teleop.connect()
            print(
                "[MANUAL] keys: [p] clearance->ready [1] clearance [2] ready "
                "[SPACE] release [x] abort/hold [q] quit",
                flush=True,
            )
            with (
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket,
                TerminalKeys() as terminal,
            ):
                while True:
                    loop_started = time.perf_counter()
                    action = teleop.get_action()
                    key = terminal.poll()
                    try:
                        if key == "q":
                            break
                        if key in ("p", "1", "2"):
                            if motion.phase in (MotionPhase.MOVING, MotionPhase.PAUSING):
                                raise ValueError("preset motion is already active; press x to abort")
                            now = time.monotonic()
                            if key == "p":
                                motion.start_prepare(action, now)
                            elif key == "1":
                                motion.start_clearance_only(action, now)
                            else:
                                motion.start_ready_only(action, now)
                            try:
                                enable_preset_torque(action)
                            except Exception:
                                motion.release()
                                raise
                        elif key == " ":
                            if motion.phase in (MotionPhase.MOVING, MotionPhase.PAUSING):
                                raise ValueError("preset motion is not complete; press x to abort first")
                            if motion.phase != MotionPhase.HOLDING:
                                raise ValueError("SPACE requires a held preset pose")
                            disable_preset_torque()
                            motion.release()
                        elif key == "x":
                            motion.abort(action)
                    except Exception as error:
                        print(f"[FAILED] key={key!r} reason={error}", flush=True)

                    goal = motion.step(action, time.monotonic())
                    if goal is not None:
                        teleop.send_feedback(goal)
                    report_motion_status()
                    udp_socket.sendto(
                        build_bimanual_datagram(
                            action,
                            sequence,
                            time.monotonic_ns(),
                        ),
                        destination,
                    )
                    sequence += 1
                    precise_sleep(
                        max(
                            1.0 / cfg.fps - (time.perf_counter() - loop_started),
                            0.0,
                        )
                    )
        except KeyboardInterrupt:
            print("\nOpenArm Mini ROS 2 teleop streaming stopped.")
        finally:
            if teleop.is_connected:
                if torque_enabled:
                    disable_preset_torque()
                teleop.disconnect()

    run()


if __name__ == "__main__":
    main()
