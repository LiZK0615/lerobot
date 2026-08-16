#!/usr/bin/env python
"""Keyboard-supervised OpenArm policy client for a remote LeRobot server."""

import argparse
import logging
import threading
import time
from pathlib import Path

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.openarm_data_collection.terminal import TerminalKeys
from lerobot.openarm_policy_runtime import PolicyRunState, ReturnToReadyMotion
from lerobot.robots.ros_openarm_bimanual.configuration_ros_openarm_bimanual import (
    RosOpenArmBimanualConfig,
)

DEFAULT_TASK = "将桌面上的螺丝收纳到盒子里"


class ReturnInterruptedError(RuntimeError):
    def __init__(self, exit_requested: bool):
        super().__init__("return-to-ready motion interrupted by keyboard")
        self.exit_requested = exit_requested


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--server-address", default="192.168.123.20:8080")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.4)
    parser.add_argument("--episode-duration-sec", type=float, default=60.0)
    parser.add_argument("--return-timeout-sec", type=float, default=30.0)
    parser.add_argument("--source-max-age-ms", type=int, default=200)
    parser.add_argument("--max-joint-velocity-rad-s", type=float, default=0.2)
    parser.add_argument("--max-gripper-velocity-m-s", type=float, default=0.01)
    parser.add_argument("--return-minimum-duration-sec", type=float, default=3.0)
    parser.add_argument("--ready-tolerance", type=float, default=0.10)
    return parser


def _hold_current_position(client: RobotClient) -> None:
    client.pause_session()
    state = client.robot.get_state()
    client.robot.send_action(state)


def _return_to_ready(client: RobotClient, args: argparse.Namespace, keyboard=None) -> None:
    client.pause_session()
    motion = ReturnToReadyMotion(
        max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
        max_gripper_velocity_m_s=args.max_gripper_velocity_m_s,
        minimum_duration_sec=args.return_minimum_duration_sec,
        tolerance=args.ready_tolerance,
    )
    motion.start(client.robot.get_state(), time.monotonic())
    deadline = time.monotonic() + args.return_timeout_sec
    period = 1.0 / args.fps

    while time.monotonic() < deadline:
        started = time.monotonic()
        key = keyboard.poll() if keyboard is not None else None
        if key in {" ", "\x1b"}:
            _hold_current_position(client)
            raise ReturnInterruptedError(exit_requested=key == "\x1b")
        client.robot.send_action(motion.command(started))
        if motion.trajectory_complete(started) and motion.target_reached(client.robot.get_state()):
            return
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    _hold_current_position(client)
    raise TimeoutError("机械臂未在限定时间内到达 table_ready，已保持当前位置")


def _print_controls(state: PolicyRunState) -> None:
    print(f"\n[{state.value}] s=开始推理  q=停止并回到table_ready  Space=立即保持  Esc=退出并保持")


def _run_supervisor(client: RobotClient, args: argparse.Namespace) -> None:
    client.pause_session()
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    client.start_barrier.wait()

    episode_started_at: float | None = None
    period = 1.0 / args.fps

    with TerminalKeys() as keyboard:
        state = PolicyRunState.RETURNING
        _print_controls(state)
        try:
            _return_to_ready(client, args, keyboard)
            state = PolicyRunState.READY
        except ReturnInterruptedError as interrupt:
            state = PolicyRunState.STOPPED if interrupt.exit_requested else PolicyRunState.HOLD
        _print_controls(state)

        while state is not PolicyRunState.STOPPED:
            loop_started = time.monotonic()
            key = keyboard.poll()

            if key == "\x1b":
                _hold_current_position(client)
                state = PolicyRunState.STOPPED
                break
            if key == " ":
                _hold_current_position(client)
                state = PolicyRunState.HOLD
                _print_controls(state)
            elif key == "q" and state in {PolicyRunState.INFERENCE, PolicyRunState.HOLD}:
                state = PolicyRunState.RETURNING
                _print_controls(state)
                try:
                    _return_to_ready(client, args, keyboard)
                    state = PolicyRunState.READY
                except ReturnInterruptedError as interrupt:
                    state = PolicyRunState.STOPPED if interrupt.exit_requested else PolicyRunState.HOLD
                episode_started_at = None
                _print_controls(state)
            elif key == "s" and state is PolicyRunState.READY:
                client.begin_session()
                episode_started_at = time.monotonic()
                state = PolicyRunState.INFERENCE
                _print_controls(state)

            if state is PolicyRunState.INFERENCE:
                if client.receiver_error.is_set():
                    raise RuntimeError("与推理服务器的动作流连接异常")
                if (
                    episode_started_at is not None
                    and loop_started - episode_started_at >= args.episode_duration_sec
                ):
                    print("\n[TIMEOUT] 单次任务达到时限，自动返回 table_ready")
                    state = PolicyRunState.RETURNING
                    try:
                        _return_to_ready(client, args, keyboard)
                        state = PolicyRunState.READY
                    except ReturnInterruptedError as interrupt:
                        state = PolicyRunState.STOPPED if interrupt.exit_requested else PolicyRunState.HOLD
                    episode_started_at = None
                    _print_controls(state)
                else:
                    if client.actions_available():
                        client.control_loop_action()
                    if client._ready_to_send_observation():
                        observation = client.control_loop_observation(args.task)
                        if observation is None:
                            raise RuntimeError("采集或发送观测失败")

            time.sleep(max(0.0, period - (time.monotonic() - loop_started)))

    client.stop()
    receiver.join(timeout=3.0)


def main() -> None:
    args = _parser().parse_args()
    if (
        min(
            args.fps,
            args.episode_duration_sec,
            args.return_timeout_sec,
            args.return_minimum_duration_sec,
        )
        <= 0
    ):
        raise ValueError("频率和所有时间参数必须为正数")
    camera_config = Path(args.camera_config).expanduser().resolve()
    if not camera_config.is_file():
        raise FileNotFoundError(f"camera config not found: {camera_config}")

    robot_config = RosOpenArmBimanualConfig(
        camera_config=str(camera_config),
        source_max_age_ms=args.source_max_age_ms,
    )
    config = RobotClientConfig(
        policy_type="smolvla",
        pretrained_name_or_path=args.policy_path,
        robot=robot_config,
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        server_address=args.server_address,
        policy_device="cuda",
        client_device="cpu",
        chunk_size_threshold=args.chunk_size_threshold,
        fps=args.fps,
    )

    logging.info("连接 Jetson 本地 ROS/相机与 Starpath 推理服务")
    client = RobotClient(config)
    if not client.start():
        client.stop()
        raise ConnectionError(f"无法连接推理服务器 {args.server_address}")
    try:
        _run_supervisor(client, args)
    except BaseException:
        try:
            _hold_current_position(client)
        except Exception:
            logging.exception("异常后的保持指令发送失败；请使用物理急停")
        client.stop()
        raise


if __name__ == "__main__":
    main()
