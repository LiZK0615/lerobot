#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run a reversible, bimanual OpenArm Mini low-stiffness experiment."""

import math
import time
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

SIDES = ("left", "right")
JOINT_MOTORS = tuple(f"joint_{index}" for index in range(1, 8))
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config/openarm_mini_compliance.yaml"


class ComplianceMode(str, Enum):
    DISABLED = "disabled"
    HOLD = "hold"
    FOLLOW = "follow"


class ComplianceStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointComplianceSettings:
    p_coefficient: int
    torque_limit: int
    position_deadband_deg: float

    def validate(self, field: str) -> None:
        if (
            not isinstance(self.p_coefficient, int)
            or isinstance(self.p_coefficient, bool)
            or not 0 <= self.p_coefficient <= 254
        ):
            raise ValueError(f"{field}.p_coefficient must be an integer between 0 and 254")
        if (
            not isinstance(self.torque_limit, int)
            or isinstance(self.torque_limit, bool)
            or not 0 <= self.torque_limit <= 1000
        ):
            raise ValueError(f"{field}.torque_limit must be an integer between 0 and 1000")
        if (
            isinstance(self.position_deadband_deg, bool)
            or not isinstance(self.position_deadband_deg, Real)
            or not math.isfinite(float(self.position_deadband_deg))
            or self.position_deadband_deg <= 0.0
        ):
            raise ValueError(f"{field}.position_deadband_deg must be a finite positive number")


@dataclass(frozen=True)
class BimanualComplianceConfig:
    joints: dict[str, dict[str, JointComplianceSettings]]


@dataclass
class OriginalRegisters:
    p_coefficient: int
    torque_limit: int
    restore_p: bool = False
    restore_torque_limit: bool = False


def load_compliance_config(path: str | Path) -> BimanualComplianceConfig:
    config_path = Path(path).expanduser()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("compliance config schema_version must be 1")
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(SIDES):
        raise ValueError("compliance config arms must contain exactly left and right")

    configured: dict[str, dict[str, JointComplianceSettings]] = {}
    expected_joints = set(JOINT_MOTORS)
    for side in SIDES:
        raw_joints = raw_arms[side]
        if not isinstance(raw_joints, dict) or set(raw_joints) != expected_joints:
            raise ValueError(f"arms.{side} must contain exactly joint_1 through joint_7")
        configured[side] = {}
        for motor in JOINT_MOTORS:
            raw = raw_joints[motor]
            if not isinstance(raw, dict):
                raise ValueError(f"arms.{side}.{motor} must be a mapping")
            try:
                settings = JointComplianceSettings(
                    p_coefficient=raw["p_coefficient"],
                    torque_limit=raw["torque_limit"],
                    position_deadband_deg=raw["position_deadband_deg"],
                )
            except KeyError as exc:
                raise ValueError(f"arms.{side}.{motor} is missing {exc.args[0]}") from exc
            settings.validate(f"arms.{side}.{motor}")
            configured[side][motor] = settings
    return BimanualComplianceConfig(configured)


class BimanualComplianceExperiment:
    """Apply and restore low-stiffness settings on both Mini arms."""

    def __init__(self, teleop: Any, config: BimanualComplianceConfig) -> None:
        self.teleop = teleop
        self.config = config
        self.mode = ComplianceMode.DISABLED
        self.goals_deg: dict[str, float] = {}
        self._originals: dict[tuple[str, str], OriginalRegisters] = {}

    @staticmethod
    def action_key(side: str, motor: str) -> str:
        return f"{side}_{motor}.pos"

    def _arms(self) -> dict[str, Any]:
        return {"left": self.teleop.left_arm, "right": self.teleop.right_arm}

    def read_positions_deg(self) -> dict[str, float]:
        action = self.teleop.get_action()
        positions: dict[str, float] = {}
        for side in SIDES:
            for motor in JOINT_MOTORS:
                key = self.action_key(side, motor)
                value = action.get(key)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"{key} must be a finite number")
                positions[key] = float(value)
        return positions

    def enable_hold(self, current_deg: dict[str, float]) -> None:
        if self.mode is not ComplianceMode.DISABLED:
            self.hold(current_deg)
            return

        arms = self._arms()
        try:
            for arm in arms.values():
                arm.bus.disable_torque(list(JOINT_MOTORS))

            for side, arm in arms.items():
                for motor in JOINT_MOTORS:
                    original = OriginalRegisters(
                        p_coefficient=int(arm.bus.read("P_Coefficient", motor, normalize=False)),
                        torque_limit=int(arm.bus.read("Torque_Limit", motor, normalize=False)),
                    )
                    self._originals[(side, motor)] = original
                    settings = self.config.joints[side][motor]
                    if settings.p_coefficient != original.p_coefficient:
                        arm.bus.write("P_Coefficient", motor, settings.p_coefficient)
                        original.restore_p = True
                    if settings.torque_limit != original.torque_limit:
                        arm.bus.write("Torque_Limit", motor, settings.torque_limit)
                        original.restore_torque_limit = True

            self._write_positions(current_deg)
            for arm in arms.values():
                arm.bus.enable_torque(list(JOINT_MOTORS))
        except BaseException as cause:
            try:
                self._restore_all()
            except Exception as restore_error:
                raise RuntimeError(f"enable failed and register restoration also failed: {restore_error}") from cause
            raise

        self.goals_deg = dict(current_deg)
        self.mode = ComplianceMode.HOLD

    def hold(self, current_deg: dict[str, float]) -> None:
        self._require_enabled()
        self._write_positions(current_deg)
        self.goals_deg = dict(current_deg)
        self.mode = ComplianceMode.HOLD

    def follow(self) -> None:
        self._require_enabled()
        self.mode = ComplianceMode.FOLLOW

    def update(self, current_deg: dict[str, float]) -> tuple[str, ...]:
        if self.mode is not ComplianceMode.FOLLOW:
            return ()
        changed: dict[str, float] = {}
        for side in SIDES:
            for motor in JOINT_MOTORS:
                key = self.action_key(side, motor)
                deadband = self.config.joints[side][motor].position_deadband_deg
                if abs(current_deg[key] - self.goals_deg[key]) > deadband:
                    changed[key] = current_deg[key]
        if changed:
            self._write_positions(changed)
            self.goals_deg.update(changed)
        return tuple(changed)

    def disable(self) -> None:
        if self.mode is ComplianceMode.DISABLED and not self._originals:
            return
        self._restore_all()
        self.goals_deg.clear()
        self.mode = ComplianceMode.DISABLED

    def _write_positions(self, positions: dict[str, float]) -> None:
        for side, arm in self._arms().items():
            goals = {
                f"{motor}.pos": positions[key]
                for motor in JOINT_MOTORS
                if (key := self.action_key(side, motor)) in positions
            }
            if goals:
                arm.write_goal_positions(goals)

    def _restore_all(self) -> None:
        errors: list[str] = []
        disabled_sides: set[str] = set()
        for side, arm in self._arms().items():
            try:
                arm.bus.disable_torque(list(JOINT_MOTORS))
                disabled_sides.add(side)
            except Exception as exc:
                errors.append(f"{side} torque disable: {exc}")

        for (side, motor), original in tuple(self._originals.items()):
            if side not in disabled_sides:
                continue
            bus = self._arms()[side].bus
            try:
                if original.restore_p:
                    bus.write("P_Coefficient", motor, original.p_coefficient)
                    original.restore_p = False
                if original.restore_torque_limit:
                    bus.write("Torque_Limit", motor, original.torque_limit)
                    original.restore_torque_limit = False
                self._originals.pop((side, motor))
            except Exception as exc:
                errors.append(f"{side} {motor} restore: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))

    def _require_enabled(self) -> None:
        if self.mode is ComplianceMode.DISABLED:
            raise ComplianceStateError("press e to enable the bimanual experiment first")


def _peak_telemetry(teleop: Any) -> str:
    summaries: list[str] = []
    for side, arm in (("L", teleop.left_arm), ("R", teleop.right_arm)):
        loads = arm.bus.sync_read("Present_Load", list(JOINT_MOTORS), normalize=False)
        currents = arm.bus.sync_read("Present_Current", list(JOINT_MOTORS), normalize=False)
        load_joint = max(loads, key=lambda name: abs(loads[name]))
        current_joint = max(currents, key=lambda name: abs(currents[name]))
        summaries.append(
            f"{side}:load={load_joint}:{loads[load_joint]} current={current_joint}:{currents[current_joint]}"
        )
    return " ".join(summaries)


def main() -> None:
    # Hardware imports stay local so unit tests do not require the Feetech SDK.
    import draccus

    from lerobot.openarm_data_collection.terminal import TerminalKeys
    from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
    from lerobot.teleoperators.bi_openarm_mini.config_bi_openarm_mini import BiOpenArmMiniConfig
    from lerobot.utils.robot_utils import precise_sleep

    @dataclass
    class OpenArmMiniComplianceTestConfig:
        teleop: TeleoperatorConfig
        joint_config: str = str(DEFAULT_CONFIG_PATH)
        fps: float = 30.0
        status_interval_sec: float = 1.0

    @draccus.wrap()
    def run(cfg: OpenArmMiniComplianceTestConfig) -> None:
        if not isinstance(cfg.teleop, BiOpenArmMiniConfig):
            raise ValueError("--teleop.type must be bi_openarm_mini")
        if not math.isfinite(cfg.fps) or cfg.fps <= 0.0:
            raise ValueError("--fps must be a finite positive number")
        if not math.isfinite(cfg.status_interval_sec) or cfg.status_interval_sec <= 0.0:
            raise ValueError("--status_interval_sec must be a finite positive number")

        joint_config = load_compliance_config(cfg.joint_config)
        teleop = make_teleoperator_from_config(cfg.teleop)
        experiment = BimanualComplianceExperiment(teleop, joint_config)

        try:
            teleop.connect()
            print(
                "[DISABLED] both Mini arms J1-J7 are torque-off; support both arms by hand\n"
                "keys: [e] enable HOLD  [f] FOLLOW  [h] hold here "
                "[SPACE] disable/restore  [q] quit"
            )
            next_status = time.monotonic()
            with TerminalKeys() as terminal:
                while True:
                    loop_started = time.perf_counter()
                    current_deg = experiment.read_positions_deg()
                    key = terminal.poll()

                    try:
                        if key == "e":
                            experiment.enable_hold(current_deg)
                            print("[HOLD] both Mini arms enabled at their measured positions")
                        elif key == "f":
                            experiment.follow()
                            print("[FOLLOW] move both arms by hand; each joint follows outside its deadband")
                        elif key == "h":
                            experiment.hold(current_deg)
                            print("[HOLD] both arms now hold their measured positions")
                        elif key == " ":
                            experiment.disable()
                            print("[DISABLED] J1-J7 torque off; original registers restored")
                        elif key == "q":
                            break
                    except ComplianceStateError as exc:
                        print(f"[FAILED] key={key!r} reason={exc}")

                    changed = experiment.update(current_deg)
                    now = time.monotonic()
                    if now >= next_status:
                        try:
                            telemetry = _peak_telemetry(teleop)
                        except Exception as exc:
                            telemetry = f"telemetry_unavailable={type(exc).__name__}"
                        print(
                            f"[{experiment.mode.value.upper()}] updated={len(changed)} "
                            f"configured_joints=14 {telemetry}"
                        )
                        next_status = now + cfg.status_interval_sec

                    precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - loop_started), 0.0))
        except KeyboardInterrupt:
            print("\n[STOPPED] interrupted")
        except BaseException as exc:
            print(f"[FAILED] {type(exc).__name__}: {exc}")
            raise
        finally:
            connected_arms = [arm for arm in (teleop.left_arm, teleop.right_arm) if arm.is_connected]
            if connected_arms:
                try:
                    experiment.disable()
                finally:
                    for arm in connected_arms:
                        if arm.is_connected:
                            arm.disconnect()

    run()


if __name__ == "__main__":
    main()
