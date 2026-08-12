from dataclasses import replace

import numpy as np
import pytest

from lerobot.cameras.orbbec import OrbbecFrame
from lerobot.openarm_data_collection.time_sync import ClockMappingError, DeviceClockMapper, SampleSynchronizer
from lerobot.openarm_data_collection.types import RecordingSnapshot, TimedVector


def frame(name, mapped_ns, sequence):
    return OrbbecFrame(
        np.zeros((2, 2, 3), dtype=np.uint8), name, "color", sequence * 1000,
        None, mapped_ns, mapped_ns, sequence
    )


def snapshot(at_ns):
    vector = TimedVector((0.0,) * 16, at_ns, None)
    return RecordingSnapshot(1, at_ns, vector, vector, None)


def push_complete(sync, target, skew=10_000_000, base_sequence=1):
    sync.push_camera("head", frame("head", target - skew // 2, base_sequence))
    sync.push_camera("left_wrist", frame("left", target, base_sequence))
    sync.push_camera("right_wrist", frame("right", target + skew // 2, base_sequence))
    sync.push_snapshot(snapshot(target))


def test_mapper_tracks_offset_and_drift_monotonically():
    mapper = DeviceClockMapper(window_size=20, max_drift_ppm=2000)
    mapped = [mapper.update(i * 33_333, 2_000_000_000 + i * 33_340_000) for i in range(20)]
    assert mapped == sorted(mapped)
    with pytest.raises(ClockMappingError, match="strictly increasing"):
        mapper.update(19 * 33_333, 3_000_000_000)


def test_accepts_complete_set_and_never_reuses_camera_frames():
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000)
    sample = sync.select(1_000_000_000)
    assert sample is not None
    assert sample.camera_skew_ns == 10_000_000
    assert sync.select(1_001_000_000) is None


def test_rejects_skew_and_stale_sources_at_boundaries():
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000, skew=36_000_000)
    assert sync.select(1_000_000_000) is None
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000, skew=35_000_000)
    assert sync.select(1_000_000_000) is not None
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000)
    assert sync.select(1_105_000_001) is None


def test_health_becomes_fatal_after_half_second():
    sync = SampleSynchronizer()
    assert sync.select(1_000_000_000) is None
    assert not sync.health(1_499_999_999).fatal
    assert sync.health(1_500_000_000).fatal
