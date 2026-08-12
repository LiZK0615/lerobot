from dataclasses import replace

import numpy as np
import pytest

from lerobot.cameras.orbbec import OrbbecFrame
from lerobot.openarm_data_collection.time_sync import (
    ClockMappingError,
    DeviceClockMapper,
    SampleSynchronizer,
    SystemClockMapper,
)
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


def test_system_clock_mapper_converts_epoch_timestamp_to_monotonic_time():
    mapper = SystemClockMapper(realtime_minus_monotonic_ns=1_700_000_000_000_000_000)

    mapped = mapper.update(
        system_timestamp_us=1_700_000_002_000_000,
        received_monotonic_ns=2_012_000_000,
    )

    assert mapped == 2_000_000_000


def test_system_clock_mapper_rejects_non_increasing_timestamp():
    mapper = SystemClockMapper(realtime_minus_monotonic_ns=1_700_000_000_000_000_000)
    mapper.update(1_700_000_002_000_000, 2_010_000_000)

    with pytest.raises(ClockMappingError, match="system timestamp is not strictly increasing"):
        mapper.update(1_700_000_002_000_000, 2_020_000_000)


def test_system_clock_mapper_rejects_implausible_capture_latency():
    mapper = SystemClockMapper(
        realtime_minus_monotonic_ns=1_700_000_000_000_000_000,
        max_capture_latency_ns=250_000_000,
    )

    with pytest.raises(ClockMappingError, match="system clock mapping is inconsistent"):
        mapper.update(1_700_000_002_000_000, 2_500_000_001)


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
    assert sync.health(1_000_000_000).reason == "camera skew exceeded: actual=36.0ms limit=35.0ms"
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000, skew=35_000_000)
    assert sync.select(1_000_000_000) is not None
    sync = SampleSynchronizer()
    push_complete(sync, 1_000_000_000)
    assert sync.select(1_105_000_001) is None
    assert sync.health(1_105_000_001).reason == "head stale: age=110.0ms limit=100.0ms"


def test_reports_state_and_action_age_with_actual_limits():
    sync = SampleSynchronizer()
    target = 1_000_000_000
    push_complete(sync, target)
    old = TimedVector((0.0,) * 16, target - 60_000_000, None)
    sync._snapshots.clear()
    sync.push_snapshot(RecordingSnapshot(2, target, old, old, None))
    assert sync.select(target) is None
    assert sync.health(target).reason == "state stale: age=60.0ms limit=50.0ms"


def test_health_becomes_fatal_after_half_second():
    sync = SampleSynchronizer()
    assert sync.select(1_000_000_000) is None
    assert not sync.health(1_499_999_999).fatal
    assert sync.health(1_500_000_000).fatal
