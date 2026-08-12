from pathlib import Path

import numpy as np

from lerobot.openarm_data_collection.dataset_sink import DatasetSink, build_features
from lerobot.openarm_data_collection.types import JOINT_NAMES, SynchronizedSample, TimedVector
from lerobot.cameras.orbbec import OrbbecFrame


class FakeMeta:
    total_episodes = 0
    total_frames = 0
    fps = 30
    robot_type = "openarm_v1_bimanual"
    features = build_features()


class FakeDataset:
    def __init__(self): self.meta, self.frames, self.saved, self.cleared, self.finalized = FakeMeta(), [], 0, 0, 0
    def add_frame(self, frame): self.frames.append(frame)
    def save_episode(self): self.saved += 1; self.meta.total_episodes += 1; self.meta.total_frames += len(self.frames); self.frames = []
    def clear_episode_buffer(self, delete_images=True): self.cleared += 1; self.frames = []
    def has_pending_frames(self): return bool(self.frames)
    def finalize(self): self.finalized += 1


def sample(at=1_000_000_000):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    def camera(name): return OrbbecFrame(image, name, "color", 1, 2, at, at, 1)
    vector = TimedVector((0.0,) * 16, at, None)
    return SynchronizedSample(at, camera("head"), camera("left"), camera("right"), vector, vector, None, 0, 0, 0)


def test_features_match_training_contract():
    features = build_features()
    assert features["observation.state"]["names"] == list(JOINT_NAMES)
    assert features["action"]["shape"] == (16,)
    assert features["observation.images.head"]["shape"] == (480, 640, 3)


def test_save_and_discard_are_transactional(tmp_path):
    dataset = FakeDataset()
    sink = DatasetSink(tmp_path, "local/test", dataset=dataset, min_episode_sec=0.0)
    assert sink.begin_episode("任务") == 0
    sink.add_sample(sample())
    assert sink.save_episode() == 0
    assert dataset.saved == 1
    assert (tmp_path / "diagnostics/episode_000000.parquet").is_file()
    sink.begin_episode("任务")
    sink.add_sample(sample())
    sink.discard_episode()
    assert dataset.cleared == 1
    assert dataset.meta.total_episodes == 1


def test_finalize_discards_pending_frames(tmp_path):
    dataset = FakeDataset()
    sink = DatasetSink(tmp_path, "local/test", dataset=dataset, min_episode_sec=0.0)
    sink.begin_episode("任务")
    sink.add_sample(sample())
    sink.finalize()
    assert dataset.cleared == 1
    assert dataset.finalized == 1
