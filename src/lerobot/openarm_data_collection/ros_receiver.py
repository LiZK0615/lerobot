import ipaddress
import json
import math
from numbers import Real
import socket
from typing import Any, Callable

from .types import JOINT_NAMES, RecordingSnapshot, TimedVector


def _integer(value: Any, field: str, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _vector(value: Any, field: str) -> TimedVector | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    values = value.get("values")
    if not isinstance(values, list) or len(values) != len(JOINT_NAMES):
        raise ValueError(f"{field}.values must contain 16 values")
    if any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)) for item in values):
        raise ValueError(f"{field}.values must contain finite numbers")
    return TimedVector(
        tuple(float(item) for item in values),
        _integer(value.get("received_monotonic_ns"), f"{field}.received_monotonic_ns"),
        _integer(value.get("ros_stamp_ns"), f"{field}.ros_stamp_ns", optional=True),
    )


def decode_recording_snapshot(data: bytes, max_datagram_bytes: int = 4096) -> RecordingSnapshot:
    if len(data) > max_datagram_bytes:
        raise ValueError("datagram exceeds maximum size")
    try:
        payload = json.loads(data.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON datagram") from error
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise ValueError("unsupported protocol version")
    return RecordingSnapshot(
        _integer(payload.get("sequence"), "sequence"),
        _integer(payload.get("sent_monotonic_ns"), "sent_monotonic_ns"),
        _vector(payload.get("state"), "state"),
        _vector(payload.get("action"), "action"),
        _vector(payload.get("leader"), "leader"),
    )


class RosSnapshotReceiver:
    def __init__(
        self, address: str = "127.0.0.1", port: int = 15001,
        max_datagram_bytes: int = 4096, socket_factory: Callable[[], Any] | None = None
    ) -> None:
        if not ipaddress.ip_address(address).is_loopback:
            raise ValueError("address must be loopback")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.max_datagram_bytes = max_datagram_bytes
        self.socket = socket_factory() if socket_factory else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((address, port))
        self.last_sequence: int | None = None
        self.invalid_packets = 0

    def poll(self) -> RecordingSnapshot | None:
        newest = None
        while True:
            try:
                data, _ = self.socket.recvfrom(self.max_datagram_bytes + 1)
            except BlockingIOError:
                break
            try:
                candidate = decode_recording_snapshot(data, self.max_datagram_bytes)
                if self.last_sequence is not None and candidate.sequence <= self.last_sequence:
                    raise ValueError("sequence must increase")
                newest = candidate
                self.last_sequence = candidate.sequence
            except ValueError:
                self.invalid_packets += 1
        return newest

    def close(self) -> None:
        self.socket.close()
