from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lerobot.cameras.orbbec import OrbbecCameraConfig
from lerobot.cameras.orbbec.camera_orbbec import _PyOrbbecAdapter
from lerobot.openarm_data_collection.dataset_sink import DatasetSink, build_features
from tests.openarm_data_collection.test_dataset_sink import FakeDataset, sample


class Device:
    def __init__(self):
        self.values = {"laser": 0, "pattern": 1, "interleave": True}
    def is_property_supported(self, *args): return True
    def get_int_property(self, prop): return self.values[prop]
    get_bool_property = get_int_property
    def set_int_property(self, prop, value): self.values[prop] = value
    set_bool_property = set_int_property


class Pipeline:
    def __init__(self, device): self.stopped = False; self.synced = False
    def get_stream_profile_list(self, sensor):
        return NS(get_video_stream_profile=lambda *args: args)
    def enable_frame_sync(self): self.synced = True
    def start(self, config, callback): self.config = config; self.callback = callback
    def stop(self): self.stopped = True


class Config:
    def __init__(self): self.profiles = []
    def enable_stream(self, profile): self.profiles.append(profile)
    def set_frame_aggregate_output_mode(self, mode): self.mode = mode


def adapter():
    device = Device()
    ob = NS(Pipeline=Pipeline, Config=Config,
            OBSensorType=NS(COLOR_SENSOR="color", LEFT_COLOR_SENSOR="lc", RIGHT_COLOR_SENSOR="rc", RIGHT_IR_SENSOR="nir"),
            OBFrameType=NS(COLOR_FRAME="rgb", LEFT_COLOR_FRAME="lc", RIGHT_COLOR_FRAME="rc", RIGHT_IR_FRAME="nir"),
            OBFormat=NS(RGB="RGB", Y8="Y8"),
            OBFrameAggregateOutputMode=NS(FULL_FRAME_REQUIRE="full"),
            OBPropertyID=NS(OB_PROP_LASER_CONTROL_INT="laser", OB_PROP_LASER_ON_OFF_PATTERN_INT="pattern",
                            OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL="interleave"),
            OBPermissionType=NS(PERMISSION_READ_WRITE="rw"),
            OBFrameMetadataType=NS(EXPOSURE="exposure", GAIN="gain", LASER_STATUS="laser"))
    obj = _PyOrbbecAdapter.__new__(_PyOrbbecAdapter)
    obj._ob = ob
    obj._context = NS(query_devices=lambda: NS(get_device_by_serial_number=lambda serial: device))
    obj._controls = {}
    return obj, device


def frame(nir=False, timestamp=10000, laser=1):
    image = np.full((480, 640) if nir else (480, 640, 3), 120, dtype=np.uint8)
    metadata = {"exposure": 7582, "gain": 248, "laser": laser}
    obj = NS(get_timestamp_us=lambda: timestamp, get_system_timestamp_us=lambda: 20000,
             get_data=lambda: image.tobytes(), get_format=lambda: "Y8" if nir else "RGB",
             has_metadata=lambda key: key in metadata, get_metadata_value=lambda key: metadata[key])
    obj.as_video_frame = lambda: NS(get_height=lambda: 480, get_width=lambda: 640)
    return obj


def test_one_pipeline_rgb_nir_pairing_metadata_and_exit_restore():
    obj, device = adapter()
    cfg = OrbbecCameraConfig(serial_number="head", model="gemini_336", selected_color_stream="color",
                            fps=30, width=640, height=480, nir_side="right", ldm_enabled=True)
    received = []
    pipeline = obj.start(cfg, received.append)
    assert pipeline.synced and len(pipeline.config.profiles) == 2
    assert device.values == {"laser": 1, "pattern": 0, "interleave": False}
    rgb = frame()
    def emit(nir):
        pipeline.callback(NS(get_frame_by_type=lambda kind: rgb if kind == "rgb" else nir))
    emit(frame(nir=True, timestamp=10200))
    assert len(received) == 1
    assert received[0].nir_image.shape == (480, 640, 3)
    assert np.all(received[0].nir_image == 120)
    assert received[0].nir_timestamp_us == 10200
    assert received[0].nir_exposure == 7582
    emit(None)
    emit(frame(nir=True, timestamp=50000))
    emit(frame(nir=True, laser=0))
    assert len(received) == 1  # Missing, distant, and wrong-state NIR never enter dataset.
    obj.stop(pipeline)
    assert pipeline.stopped
    assert device.values == {"laser": 0, "pattern": 1, "interleave": True}


def test_nir_dataset_requires_nir_and_keeps_capture_diagnostics(tmp_path):
    dataset = FakeDataset()
    sink = DatasetSink(tmp_path, "local/nir", dataset=dataset, include_nir=True, min_episode_sec=0)
    sink.begin_episode("Pick up the block and place it on the plate.")
    original = sample()
    with pytest.raises(ValueError, match="NIR required"):
        sink.add_sample(original)
    assert dataset.frames == []
    nir = np.full((480, 640, 3), 100, dtype=np.uint8)
    bundle = replace(original.head, nir_image=nir, nir_timestamp_us=2, nir_laser_status=1)
    sink.add_sample(replace(original, head=bundle))
    assert np.array_equal(dataset.frames[0]["observation.images.head_nir"], nir)
    assert sink._diagnostics[0]["head_rgb_nir_delta_us"] == 1
    assert "observation.images.head_nir" in build_features(True)
    assert "observation.images.head_nir" not in build_features()
