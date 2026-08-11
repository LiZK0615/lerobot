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
from numbers import Real
from typing import Any

POSITION_NAMES = tuple([f"joint_{index}" for index in range(1, 8)] + ["gripper"])


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
    from dataclasses import dataclass

    import draccus

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

        teleop = make_teleoperator_from_config(cfg.teleop)
        destination = (cfg.host, cfg.udp_port)
        sequence = starting_sequence()

        try:
            teleop.connect()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
                while True:
                    loop_started = time.perf_counter()
                    action = teleop.get_action()
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
                teleop.disconnect()

    run()


if __name__ == "__main__":
    main()
