"""Orbbec SDK v2 color camera backend with timestamp-preserving packets."""

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _pyorbbecsdk_available, require_package

from ..camera import Camera
from .configuration_orbbec import OrbbecCameraConfig


def _sdk_stream_type(stream_type_enum: Any, selected_color_stream: str) -> Any:
    """Map our stable stream names to pyorbbecsdk2's v2 enum members."""
    return {
        "color": stream_type_enum.COLOR_STREAM,
        "left_color": stream_type_enum.LEFT_COLOR_STREAM,
        "right_color": stream_type_enum.RIGHT_COLOR_STREAM,
    }[selected_color_stream]


def _sdk_sensor_type(sensor_type_enum: Any, selected_color_stream: str) -> Any:
    """Map stream selection to the sensor enum required for profile discovery."""
    return {
        "color": sensor_type_enum.COLOR_SENSOR,
        "left_color": sensor_type_enum.LEFT_COLOR_SENSOR,
        "right_color": sensor_type_enum.RIGHT_COLOR_SENSOR,
    }[selected_color_stream]


def _sdk_frame_type(frame_type_enum: Any, selected_color_stream: str) -> Any:
    """Map stream selection to the frame enum required by FrameSet."""
    return {
        "color": frame_type_enum.COLOR_FRAME,
        "left_color": frame_type_enum.LEFT_COLOR_FRAME,
        "right_color": frame_type_enum.RIGHT_COLOR_FRAME,
    }[selected_color_stream]


def _select_video_profile(profiles: Any, format_enum: Any, width: int, height: int, fps: int) -> Any:
    """Select the requested profile while preferring formats cheap to convert to RGB."""
    errors: list[str] = []
    for format_name in ("RGB", "BGR", "MJPG", "YUYV", "YUY2"):
        pixel_format = getattr(format_enum, format_name, None)
        if pixel_format is None:
            continue
        try:
            return profiles.get_video_stream_profile(width, height, pixel_format, fps)
        except RuntimeError as error:
            errors.append(f"{format_name}: {error}")
    raise RuntimeError(
        f"no supported Orbbec color profile for {width}x{height}@{fps}; " + "; ".join(errors)
    )


@dataclass(frozen=True)
class RawOrbbecFrame:
    image: NDArray[np.uint8]
    pixel_format: str
    device_timestamp_us: int
    system_timestamp_us: int | None
    received_monotonic_ns: int
    nir_image: NDArray[np.uint8] | None = None
    nir_timestamp_us: int | None = None
    nir_exposure: int | None = None
    nir_gain: int | None = None
    nir_laser_status: int | None = None


@dataclass(frozen=True)
class OrbbecFrame:
    image: NDArray[np.uint8]
    serial_number: str
    stream_name: str
    device_timestamp_us: int
    system_timestamp_us: int | None
    received_monotonic_ns: int
    mapped_monotonic_ns: int | None
    sequence: int
    nir_image: NDArray[np.uint8] | None = None
    nir_timestamp_us: int | None = None
    nir_exposure: int | None = None
    nir_gain: int | None = None
    nir_laser_status: int | None = None


class SdkAdapter(Protocol):
    def list_devices(self) -> dict[str, dict[str, str]]: ...
    def start(self, config: OrbbecCameraConfig, callback: Any) -> Any: ...
    def stop(self, pipeline: Any) -> None: ...




def _restore_controls(device: Any, ob: Any, controls: dict[str, Any]) -> None:
    errors = []
    for name, value in controls.items():
        prop = getattr(ob.OBPropertyID, name)
        setter = device.set_bool_property if name.endswith("BOOL") else device.set_int_property
        try:
            setter(prop, value)
        except Exception as error:
            errors.append(f"{name}: {error}")
    for name, value in controls.items():
        getter = device.get_bool_property if name.endswith("BOOL") else device.get_int_property
        try:
            if getter(getattr(ob.OBPropertyID, name)) != value:
                errors.append(f"{name}: restoration readback mismatch")
        except Exception as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError("camera settings restoration failed: " + "; ".join(errors))


class _PyOrbbecAdapter:
    """Thin dynamic adapter so unit tests never require the hardware SDK."""

    def __init__(self) -> None:
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        import pyorbbecsdk as ob  # type: ignore

        self._ob = ob
        self._context = ob.Context()
        self._controls: dict[int, tuple[Any, dict[str, Any]]] = {}

    def list_devices(self) -> dict[str, dict[str, str]]:
        devices: dict[str, dict[str, str]] = {}
        device_list = self._context.query_devices()
        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            serial = str(info.get_serial_number())
            devices[serial] = {
                "model": str(info.get_name()).lower(),
                "firmware": str(info.get_firmware_version()),
            }
        return devices

    def start(self, config: OrbbecCameraConfig, callback: Any) -> Any:
        ob = self._ob
        device = self._context.query_devices().get_device_by_serial_number(config.serial_number)
        if config.preset:
            device.load_preset(config.preset)
        pipeline = ob.Pipeline(device)
        pipeline_config = ob.Config()
        sensor_type = _sdk_sensor_type(ob.OBSensorType, config.selected_color_stream)
        frame_type = _sdk_frame_type(ob.OBFrameType, config.selected_color_stream)
        profiles = pipeline.get_stream_profile_list(sensor_type)
        profile = _select_video_profile(profiles, ob.OBFormat, config.width, config.height, config.fps)
        pipeline_config.enable_stream(profile)
        nir_frame_type = None
        if config.nir_side is not None:
            nir_sensor = getattr(ob.OBSensorType, f"{config.nir_side.upper()}_IR_SENSOR")
            nir_frame_type = getattr(ob.OBFrameType, f"{config.nir_side.upper()}_IR_FRAME")
            nir_profiles = pipeline.get_stream_profile_list(nir_sensor)
            nir_profile = nir_profiles.get_video_stream_profile(config.width, config.height, ob.OBFormat.Y8, config.fps)
            pipeline_config.enable_stream(nir_profile)
            pipeline.enable_frame_sync()
            pipeline_config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        controls: dict[str, Any] = {}
        # No preset changes for the head in the supplied recording config.
        if config.ldm_enabled is not None:
            for name in ("OB_PROP_LASER_CONTROL_INT", "OB_PROP_LASER_ON_OFF_PATTERN_INT", "OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL"):
                prop = getattr(ob.OBPropertyID, name, None)
                if prop is None or not device.is_property_supported(prop, ob.OBPermissionType.PERMISSION_READ_WRITE):
                    raise RuntimeError(f"required LDM control unavailable: {name}")
                getter = device.get_bool_property if name.endswith("BOOL") else device.get_int_property
                controls[name] = getter(prop)

        def on_frames(frames: Any) -> None:
            frame = frames.get_frame_by_type(frame_type)
            if frame is None:
                return
            nir_values: dict[str, Any] = {}
            if nir_frame_type is not None:
                nir = frames.get_frame_by_type(nir_frame_type)
                if nir is None:
                    return
                nir_ts = int(nir.get_timestamp_us())
                if abs(nir_ts - int(frame.get_timestamp_us())) > config.nir_pair_max_ms * 1000:
                    return  # Unpaired bundles cannot become valid training samples.
                nir_video = nir.as_video_frame()
                nir_image = np.frombuffer(nir.get_data(), dtype=np.uint8).copy().reshape(
                    int(nir_video.get_height()), int(nir_video.get_width())
                )
                nir_values = {"nir_image": np.repeat(nir_image[..., None], 3, axis=2), "nir_timestamp_us": nir_ts}
                for key, metadata_name in (("nir_exposure", "EXPOSURE"), ("nir_gain", "GAIN"), ("nir_laser_status", "LASER_STATUS")):
                    metadata = getattr(ob.OBFrameMetadataType, metadata_name, None)
                    nir_values[key] = int(nir.get_metadata_value(metadata)) if metadata is not None and nir.has_metadata(metadata) else None
                if config.ldm_enabled is not None and nir_values["nir_laser_status"] != int(config.ldm_enabled):
                    return  # Startup transients/mismatches are excluded, not written as NIR.
            video = frame.as_video_frame()
            data = np.frombuffer(frame.get_data(), dtype=np.uint8).copy()
            height = int(video.get_height())
            width = int(video.get_width())
            pixel_format = str(frame.get_format()).upper()
            if "MJPG" in pixel_format or "MJPEG" in pixel_format:
                image = data
            elif data.size == height * width * 3:
                image = data.reshape((height, width, 3))
            elif data.size == height * width * 2:
                image = data.reshape((height, width, 2))
            else:
                raise RuntimeError(
                    f"unexpected Orbbec frame size {data.size} for {width}x{height} {pixel_format}"
                )
            callback(
                RawOrbbecFrame(
                    image=image,
                    pixel_format=pixel_format,
                    device_timestamp_us=int(frame.get_timestamp_us()),
                    system_timestamp_us=int(frame.get_system_timestamp_us()),
                    received_monotonic_ns=time.monotonic_ns(),
                    **nir_values,
                )
            )

        try:
            if controls:
                device.set_bool_property(ob.OBPropertyID.OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL, False)
                device.set_int_property(ob.OBPropertyID.OB_PROP_LASER_ON_OFF_PATTERN_INT, 0)
                device.set_int_property(ob.OBPropertyID.OB_PROP_LASER_CONTROL_INT, int(config.ldm_enabled))
                if device.get_int_property(ob.OBPropertyID.OB_PROP_LASER_CONTROL_INT) != int(config.ldm_enabled):
                    raise RuntimeError("LDM readback mismatch")
                print(f"[CAMERA] {config.serial_number} LDM={config.ldm_enabled}; restore snapshot={controls}", flush=True)
            pipeline.start(pipeline_config, on_frames)
        except BaseException:
            try:
                pipeline.stop()
            finally:
                _restore_controls(device, ob, controls)
            raise
        self._controls[id(pipeline)] = (device, controls)
        return pipeline

    def stop(self, pipeline: Any) -> None:
        device, controls = self._controls.pop(id(pipeline))
        try:
            pipeline.stop()
        finally:
            _restore_controls(device, self._ob, controls)


class OrbbecSdkRuntime:
    def __init__(self, adapter: SdkAdapter | None = None) -> None:
        self.adapter = adapter if adapter is not None else _PyOrbbecAdapter()
        self._claimed: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, config: OrbbecCameraConfig, callback: Any) -> Any:
        with self._lock:
            if config.serial_number in self._claimed:
                raise RuntimeError(f"Orbbec serial {config.serial_number} is already claimed")
            device = self.adapter.list_devices().get(config.serial_number)
            if device is None:
                raise RuntimeError(f"Orbbec serial {config.serial_number} was not found")
            expected_model = config.model.replace("_", " ")
            actual_model = device.get("model", "").lower().replace("_", " ")
            if expected_model not in actual_model:
                raise RuntimeError(
                    f"Orbbec serial {config.serial_number} model mismatch: "
                    f"expected {config.model}, got {device.get('model', '')}"
                )
            self._claimed.add(config.serial_number)
        try:
            return self.adapter.start(config, callback)
        except Exception:
            self.release(config.serial_number)
            raise

    def release(self, serial_number: str) -> None:
        with self._lock:
            self._claimed.discard(serial_number)


class OrbbecCamera(Camera):
    _default_runtime: OrbbecSdkRuntime | None = None

    def __init__(self, config: OrbbecCameraConfig, runtime: OrbbecSdkRuntime | None = None):
        if runtime is None:
            if self.__class__._default_runtime is None:
                if not _pyorbbecsdk_available:
                    require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
                self.__class__._default_runtime = OrbbecSdkRuntime()
            runtime = self.__class__._default_runtime
        super().__init__(config)
        self.config = config
        self.serial_number = config.serial_number
        self.stream_name = config.selected_color_stream
        self._runtime = runtime
        self._pipeline: Any | None = None
        self._frames: deque[OrbbecFrame] = deque(maxlen=120)
        self._condition = threading.Condition()
        self._sequence = 0
        self._last_async_sequence = -1

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        runtime = OrbbecCamera._default_runtime or OrbbecSdkRuntime()
        return [dict(serial_number=serial, **info) for serial, info in runtime.adapter.list_devices().items()]

    @staticmethod
    def _convert(image: NDArray[np.uint8], pixel_format: str) -> NDArray[np.uint8]:
        fmt = pixel_format.upper()
        if "BGR" in fmt:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif "YUY" in fmt:
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)
        elif "MJPG" in fmt or "MJPEG" in fmt:
            decoded = cv2.imdecode(image.reshape(-1), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("failed to decode Orbbec MJPEG frame")
            image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif "RGB" not in fmt:
            raise RuntimeError(f"unsupported Orbbec color format: {pixel_format}")
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _on_frame(self, raw: RawOrbbecFrame) -> None:
        packet = OrbbecFrame(
            image=self._convert(raw.image, raw.pixel_format),
            serial_number=self.serial_number,
            stream_name=self.stream_name,
            device_timestamp_us=raw.device_timestamp_us,
            system_timestamp_us=raw.system_timestamp_us,
            received_monotonic_ns=raw.received_monotonic_ns,
            mapped_monotonic_ns=None,
            sequence=self._sequence,
            nir_image=raw.nir_image,
            nir_timestamp_us=raw.nir_timestamp_us,
            nir_exposure=raw.nir_exposure,
            nir_gain=raw.nir_gain,
            nir_laser_status=raw.nir_laser_status,
        )
        self._sequence += 1
        with self._condition:
            self._frames.append(packet)
            self._condition.notify_all()

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        self._pipeline = self._runtime.claim(self.config, self._on_frame)
        if warmup:
            self.async_read(timeout_ms=max(self.config.warmup_s * 1000, 200))

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        pipeline, self._pipeline = self._pipeline, None
        self._runtime.adapter.stop(pipeline)
        self._runtime.release(self.serial_number)
        with self._condition:
            self._frames.clear()

    def read_latest_packet(self, max_age_ms: int = 500) -> OrbbecFrame | None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._condition:
            packet = self._frames[-1] if self._frames else None
        if packet is None:
            return None
        age_ms = (time.monotonic_ns() - packet.received_monotonic_ns) / 1_000_000
        if age_ms > max_age_ms:
            raise TimeoutError(f"latest Orbbec frame is {age_ms:.1f} ms old")
        return packet

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        packet = self.read_latest_packet(max_age_ms)
        if packet is None:
            raise RuntimeError(f"{self} has not captured any frames")
        return packet.image

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while not self._frames or self._frames[-1].sequence == self._last_async_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(remaining):
                    raise TimeoutError("timed out waiting for an Orbbec frame")
            packet = self._frames[-1]
            self._last_async_sequence = packet.sequence
            return packet.image

    def read(self) -> NDArray[Any]:
        return self.async_read(timeout_ms=10_000)
