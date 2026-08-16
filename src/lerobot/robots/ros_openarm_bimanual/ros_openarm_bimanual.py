import json
import math
import socket
import time
from pathlib import Path

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.openarm_data_collection.config import load_camera_rig
from lerobot.openarm_data_collection.ros_receiver import RosSnapshotReceiver
from lerobot.openarm_data_collection.types import JOINT_NAMES
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .configuration_ros_openarm_bimanual import RosOpenArmBimanualConfig


class RosOpenArmBimanual(Robot):
    config_class = RosOpenArmBimanualConfig
    name = "ros_openarm_bimanual"

    def __init__(self, config: RosOpenArmBimanualConfig):
        super().__init__(config)
        self.config = config
        self.camera_configs = load_camera_rig(Path(config.camera_config))
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._receiver: RosSnapshotReceiver | None = None
        self._command_socket: socket.socket | None = None
        self._sequence = time.monotonic_ns()
        self._latest_state = None

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **dict.fromkeys(JOINT_NAMES, float),
            **{name: (config.height, config.width, 3) for name, config in self.camera_configs.items()},
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(JOINT_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._receiver is not None and self._command_socket is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._receiver = RosSnapshotReceiver(
            address=self.config.ros_snapshot_host,
            port=self.config.ros_snapshot_port,
        )
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        connected = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
        except Exception:
            for camera in reversed(connected):
                camera.disconnect()
            self._receiver.close()
            self._receiver = None
            self._command_socket.close()
            self._command_socket = None
            raise

    def _fresh_state(self):
        assert self._receiver is not None
        snapshot = self._receiver.poll()
        if snapshot is not None and snapshot.state is not None:
            self._latest_state = snapshot.state
        if self._latest_state is None:
            raise RuntimeError("no ROS joint state snapshot received")
        age_ms = (time.monotonic_ns() - self._latest_state.received_monotonic_ns) / 1_000_000
        if age_ms > self.config.source_max_age_ms:
            raise TimeoutError(f"ROS joint state is {age_ms:.1f} ms old")
        return self._latest_state

    @check_if_not_connected
    def get_state(self) -> dict[str, float]:
        state = self._fresh_state()
        return dict(zip(JOINT_NAMES, state.values, strict=True))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = self.get_state()
        for name, camera in self.cameras.items():
            observation[name] = camera.read_latest(self.config.source_max_age_ms)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        values = []
        for name in JOINT_NAMES:
            value = action.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values.append(value)
        payload = json.dumps(
            {
                "version": 1,
                "source": "policy",
                "sequence": self._sequence,
                "sent_monotonic_ns": time.monotonic_ns(),
                "positions": values,
            },
            separators=(",", ":"),
        ).encode()
        assert self._command_socket is not None
        self._command_socket.sendto(
            payload,
            (self.config.policy_command_host, self.config.policy_command_port),
        )
        self._sequence += 1
        return dict(zip(JOINT_NAMES, values, strict=True))

    @check_if_not_connected
    def disconnect(self) -> None:
        for camera in reversed(tuple(self.cameras.values())):
            if camera.is_connected:
                camera.disconnect()
        assert self._receiver is not None and self._command_socket is not None
        self._receiver.close()
        self._command_socket.close()
        self._receiver = None
        self._command_socket = None
