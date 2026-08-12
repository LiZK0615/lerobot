import time

import numpy as np
import pytest

from lerobot.cameras.orbbec import (
    OrbbecCamera,
    OrbbecCameraConfig,
    OrbbecSdkRuntime,
    RawOrbbecFrame,
)
from lerobot.cameras.utils import make_cameras_from_configs


class FakeAdapter:
    def __init__(self):
        self.devices = {
            "336-head": {"model": "gemini_336", "firmware": "1.0"},
            "305-left": {"model": "gemini_305", "firmware": "1.0"},
            "305-right": {"model": "gemini_305", "firmware": "1.0"},
        }
        self.started = []
        self.callbacks = {}

    def list_devices(self):
        return self.devices

    def start(self, config, callback):
        self.started.append((config.serial_number, config.selected_color_stream))
        self.callbacks[config.serial_number] = callback
        return config.serial_number

    def stop(self, pipeline):
        self.callbacks.pop(pipeline, None)

    def emit(self, serial, image, pixel_format="RGB"):
        self.callbacks[serial](
            RawOrbbecFrame(image, pixel_format, 1000, 1010, time.monotonic_ns())
        )


def head_config():
    return OrbbecCameraConfig(
        serial_number="336-head",
        model="gemini_336",
        selected_color_stream="color",
        fps=30,
        width=640,
        height=480,
    )


def test_validates_model_stream_and_profile():
    with pytest.raises(ValueError, match="Dual Color Streams"):
        OrbbecCameraConfig(
            serial_number="305-left",
            model="gemini_305",
            selected_color_stream="left_color",
            fps=30,
            width=640,
            height=480,
        )
    with pytest.raises(ValueError, match="color stream"):
        OrbbecCameraConfig(
            serial_number="336-head",
            model="gemini_336",
            selected_color_stream="left_color",
            fps=30,
            width=640,
            height=480,
        )


def test_factory_builds_orbbec_camera(monkeypatch):
    monkeypatch.setattr(OrbbecCamera, "_default_runtime", OrbbecSdkRuntime(FakeAdapter()))
    assert isinstance(make_cameras_from_configs({"head": head_config()})["head"], OrbbecCamera)


def test_three_serials_use_independent_pipelines_and_selected_streams():
    adapter = FakeAdapter()
    runtime = OrbbecSdkRuntime(adapter)
    configs = [
        head_config(),
        OrbbecCameraConfig(
            serial_number="305-left", model="gemini_305", preset="Dual Color Streams",
            selected_color_stream="right_color", fps=30, width=640, height=480
        ),
        OrbbecCameraConfig(
            serial_number="305-right", model="gemini_305", preset="Dual Color Streams",
            selected_color_stream="left_color", fps=30, width=640, height=480
        ),
    ]
    cameras = [OrbbecCamera(config, runtime) for config in configs]
    for camera in cameras:
        camera.connect(warmup=False)
    assert adapter.started == [
        ("336-head", "color"),
        ("305-left", "right_color"),
        ("305-right", "left_color"),
    ]
    for camera in cameras:
        camera.disconnect()


def test_packet_is_contiguous_rgb_and_preserves_timestamps():
    adapter = FakeAdapter()
    camera = OrbbecCamera(head_config(), OrbbecSdkRuntime(adapter))
    camera.connect(warmup=False)
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr[0, 0] = [1, 2, 3]
    adapter.emit("336-head", bgr, "BGR")
    packet = camera.read_latest_packet()
    assert packet is not None
    assert packet.image[0, 0].tolist() == [3, 2, 1]
    assert packet.image.flags.c_contiguous
    assert packet.device_timestamp_us == 1000
    assert packet.system_timestamp_us == 1010
    assert packet.mapped_monotonic_ns is None
    assert camera.read_latest().shape == (480, 640, 3)
    camera.disconnect()


def test_runtime_rejects_duplicate_claim():
    runtime = OrbbecSdkRuntime(FakeAdapter())
    first = OrbbecCamera(head_config(), runtime)
    second = OrbbecCamera(head_config(), runtime)
    first.connect(warmup=False)
    with pytest.raises(RuntimeError, match="already claimed"):
        second.connect(warmup=False)
    first.disconnect()


def test_runtime_rejects_serial_with_wrong_model():
    adapter = FakeAdapter()
    adapter.devices["336-head"]["model"] = "orbbec gemini 305"
    camera = OrbbecCamera(head_config(), OrbbecSdkRuntime(adapter))
    with pytest.raises(RuntimeError, match="model mismatch"):
        camera.connect(warmup=False)
