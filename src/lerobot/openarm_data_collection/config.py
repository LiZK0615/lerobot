from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from lerobot.cameras.orbbec import OrbbecCameraConfig


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenArmRecordConfig:
    dataset_root: Path
    dataset_name: str
    task: str
    camera_config: Path
    expected_mount: Path | None = None
    fps: int = 30
    min_free_space_gb: float = 20.0
    min_episode_sec: float = 1.0
    max_episode_sec: float = 120.0
    record_command_diagnostics: bool = False
    ros_udp_port: int = 15001
    display_cameras: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.dataset_name):
            raise ValueError("dataset_name may contain only ASCII letters, digits, _ and -")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if self.fps != 30:
            raise ValueError("the initial implementation supports fps=30 only")
        if not 1 <= self.ros_udp_port <= 65535:
            raise ValueError("ros_udp_port must be between 1 and 65535")

    @property
    def dataset_path(self) -> Path:
        return self.dataset_root.expanduser().resolve() / self.dataset_name


def load_camera_rig(path: Path) -> dict[str, OrbbecCameraConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cameras = payload.get("cameras") if isinstance(payload, dict) else None
    expected = {"head", "left_wrist", "right_wrist"}
    if not isinstance(cameras, dict) or set(cameras) != expected:
        raise ValueError(f"camera config must contain exactly {sorted(expected)}")
    configs = {
        name: OrbbecCameraConfig(fps=30, width=640, height=480, **values)
        for name, values in cameras.items()
    }
    serials = [config.serial_number for config in configs.values()]
    if len(set(serials)) != len(serials):
        raise ValueError("camera serial numbers must be unique")
    return configs


def _mount_point(path: Path) -> Path:
    current = path.resolve()
    while not current.is_mount() and current != current.parent:
        current = current.parent
    return current


def validate_storage(path: Path, min_free_space_gb: float, expected_mount: Path | None = None) -> None:
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    actual_mount = _mount_point(root)
    if expected_mount is not None and actual_mount != expected_mount.expanduser().resolve():
        raise StorageError(f"{root} is not on the expected mounted filesystem {expected_mount}")
    free_gb = shutil.disk_usage(root).free / (1024**3)
    if free_gb < min_free_space_gb:
        raise StorageError(f"only {free_gb:.1f} GB free; {min_free_space_gb:.1f} GB required")
    try:
        fd, probe = tempfile.mkstemp(prefix=".openarm-write-probe-", dir=root)
        os.write(fd, b"ok")
        os.fsync(fd)
        os.close(fd)
        Path(probe).unlink()
    except OSError as error:
        raise StorageError(f"dataset root is not writable: {error}") from error
