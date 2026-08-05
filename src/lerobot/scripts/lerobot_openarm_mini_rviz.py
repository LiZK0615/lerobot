#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stream mapped OpenArm Mini actions to the ROS 2 RViz bridge over UDP."""

import json
import math
import socket
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any

EXPECTED_ACTION_KEYS = tuple([f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"])


def starting_sequence() -> int:
    """Choose a restart-safe sequence origin on the shared host."""
    return time.monotonic_ns()


def build_datagram(
    action: Mapping[str, Any],
    sequence: int,
    sent_monotonic_ns: int,
    side: str,
) -> bytes:
    """Build one versioned datagram from an already-mapped teleoperator action."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if isinstance(sent_monotonic_ns, bool) or not isinstance(sent_monotonic_ns, int) or sent_monotonic_ns < 0:
        raise ValueError("sent_monotonic_ns must be a non-negative integer")

    positions: dict[str, float] = {}
    for key in EXPECTED_ACTION_KEYS:
        value = action.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be a finite number")
        positions[key.removesuffix(".pos")] = float(value)

    return json.dumps(
        {
            "version": 1,
            "sequence": sequence,
            "sent_monotonic_ns": sent_monotonic_ns,
            "side": side,
            "positions_deg": positions,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def main() -> None:
    # Imports stay inside main so protocol tests do not require hardware extras.
    from dataclasses import dataclass

    import draccus

    from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
    from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig
    from lerobot.utils.robot_utils import precise_sleep

    @dataclass
    class OpenArmMiniRvizConfig:
        teleop: TeleoperatorConfig
        host: str = "127.0.0.1"
        udp_port: int = 15000
        fps: float = 30.0

    @draccus.wrap()
    def run(cfg: OpenArmMiniRvizConfig) -> None:
        if not isinstance(cfg.teleop, OpenArmMiniConfig):
            raise ValueError("--teleop.type must be openarm_mini")
        if cfg.teleop.side not in ("left", "right"):
            raise ValueError("--teleop.side must be left or right")
        if not 1 <= cfg.udp_port <= 65535:
            raise ValueError("--udp_port must be between 1 and 65535")
        if not math.isfinite(cfg.fps) or cfg.fps <= 0.0:
            raise ValueError("--fps must be a finite positive number")

        teleop = make_teleoperator_from_config(cfg.teleop)
        destination = (cfg.host, cfg.udp_port)
        sequence = starting_sequence()

        try:
            teleop.connect()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
                while True:
                    loop_started = time.perf_counter()
                    action = teleop.get_action()
                    datagram = build_datagram(
                        action,
                        sequence,
                        time.monotonic_ns(),
                        cfg.teleop.side,
                    )
                    udp_socket.sendto(datagram, destination)
                    sequence += 1
                    precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - loop_started), 0.0))
        except KeyboardInterrupt:
            print("\nOpenArm Mini RViz streaming stopped.")
        finally:
            if teleop.is_connected:
                teleop.disconnect()

    run()


if __name__ == "__main__":
    main()
