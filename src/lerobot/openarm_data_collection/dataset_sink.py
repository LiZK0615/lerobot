import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .types import JOINT_NAMES, SynchronizedSample


class DatasetCompatibilityError(ValueError):
    pass


def build_features() -> dict[str, dict[str, Any]]:
    vector = {"dtype": "float32", "shape": (16,), "names": list(JOINT_NAMES)}
    video = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.head": dict(video),
        "observation.images.left_wrist": dict(video),
        "observation.images.right_wrist": dict(video),
        "observation.state": dict(vector),
        "action": dict(vector),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class DatasetSink:
    def __init__(
        self, root: Path, repo_id: str, dataset: Any | None = None, fps: int = 30,
        min_episode_sec: float = 1.0, max_episode_sec: float = 120.0
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.fps = fps
        self.min_episode_sec = min_episode_sec
        self.max_episode_sec = max_episode_sec
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset if dataset is not None else self._open_dataset()
        self._task: str | None = None
        self._diagnostics: list[dict[str, Any]] = []
        self._episode_index: int | None = None

    def _open_dataset(self) -> LeRobotDataset:
        info = self.root / "meta/info.json"
        if info.is_file():
            dataset = LeRobotDataset.resume(self.repo_id, root=self.root)
            expected = build_features()
            differences = []
            if dataset.meta.fps != self.fps:
                differences.append(f"fps: {dataset.meta.fps} != {self.fps}")
            for key, feature in expected.items():
                actual = dataset.meta.features.get(key)
                for field in ("dtype", "shape", "names"):
                    if actual is None:
                        differences.append(f"{key}.{field}")
                        continue
                    matches = (
                        actual.get(field) == feature[field]
                        if field == "dtype"
                        else tuple(actual.get(field, ())) == tuple(feature[field])
                    )
                    if not matches:
                        differences.append(f"{key}.{field}")
            if differences:
                raise DatasetCompatibilityError("incompatible dataset: " + ", ".join(differences))
            return dataset
        # LeRobot create requires the destination not to exist.
        self.root.rmdir()
        return LeRobotDataset.create(
            self.repo_id, self.fps, build_features(), root=self.root,
            robot_type="openarm_v1_bimanual", use_videos=True,
            image_writer_threads=4, video_backend="pyav"
        )

    @property
    def total_episodes(self) -> int:
        return int(self.dataset.meta.total_episodes)

    def begin_episode(self, task: str) -> int:
        if self._episode_index is not None:
            raise RuntimeError("an episode is already active")
        self._episode_index = self.total_episodes
        self._task = task
        self._diagnostics = []
        _atomic_json(self.root / ".openarm_recording/active.json", {"episode_index": self._episode_index, "task": task})
        return self._episode_index

    def add_sample(self, sample: SynchronizedSample) -> None:
        if self._episode_index is None or self._task is None:
            raise RuntimeError("no active episode")
        self.dataset.add_frame({
            "observation.images.head": sample.head.image,
            "observation.images.left_wrist": sample.left_wrist.image,
            "observation.images.right_wrist": sample.right_wrist.image,
            "observation.state": np.asarray(sample.state.values, dtype=np.float32),
            "action": np.asarray(sample.action.values, dtype=np.float32),
            "task": self._task,
        })
        self._diagnostics.append({
            "sample_monotonic_ns": sample.sample_monotonic_ns,
            "head_device_timestamp_us": sample.head.device_timestamp_us,
            "left_wrist_device_timestamp_us": sample.left_wrist.device_timestamp_us,
            "right_wrist_device_timestamp_us": sample.right_wrist.device_timestamp_us,
            "head_mapped_monotonic_ns": sample.head.mapped_monotonic_ns,
            "left_wrist_mapped_monotonic_ns": sample.left_wrist.mapped_monotonic_ns,
            "right_wrist_mapped_monotonic_ns": sample.right_wrist.mapped_monotonic_ns,
            "camera_skew_ns": sample.camera_skew_ns,
            "state_age_ns": sample.state_age_ns,
            "action_age_ns": sample.action_age_ns,
            "command_joint_states": list(sample.command.values) if sample.command is not None else None,
        })

    def _reset(self) -> None:
        self._task = None
        self._episode_index = None
        self._diagnostics = []
        (self.root / ".openarm_recording/active.json").unlink(missing_ok=True)
        (self.root / ".openarm_recording/commit.json").unlink(missing_ok=True)

    def save_episode(self) -> int:
        if self._episode_index is None or not self._diagnostics:
            raise RuntimeError("active episode has no frames")
        frame_count = len(self._diagnostics)
        duration = frame_count / self.fps
        if duration < self.min_episode_sec or duration > self.max_episode_sec:
            raise ValueError(f"episode duration {duration:.3f}s is outside allowed range")
        episode = self._episode_index
        diag_dir = self.root / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        temporary = diag_dir / f".episode_{episode:06d}.parquet.tmp"
        final = diag_dir / f"episode_{episode:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(self._diagnostics), temporary)
        _atomic_json(self.root / ".openarm_recording/commit.json", {"episode_index": episode, "state": "PREPARED"})
        self.dataset.save_episode(parallel_encoding=False)
        _atomic_json(self.root / ".openarm_recording/commit.json", {"episode_index": episode, "state": "DATASET_SAVED"})
        os.replace(temporary, final)
        self._reset()
        return episode

    def discard_episode(self) -> None:
        if self._episode_index is None:
            return
        self.dataset.clear_episode_buffer(delete_images=True)
        self._reset()

    def finalize(self) -> None:
        if self.dataset.has_pending_frames():
            self.discard_episode()
        self.dataset.finalize()
