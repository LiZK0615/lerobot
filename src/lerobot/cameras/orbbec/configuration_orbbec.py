"""Configuration for Orbbec SDK v2 color cameras."""

from dataclasses import dataclass
from typing import Literal

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("orbbec")
@dataclass(kw_only=True)
class OrbbecCameraConfig(CameraConfig):
    serial_number: str
    model: Literal["gemini_336", "gemini_305"]
    selected_color_stream: Literal["color", "left_color", "right_color"]
    preset: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0
    nir_side: Literal["left", "right"] | None = None
    ldm_enabled: bool | None = None
    nir_pair_max_ms: float = 20.0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if self.nir_side not in (None, "left", "right"):
            raise ValueError("nir_side must be left or right")
        if self.model != "gemini_336" and (self.nir_side is not None or self.ldm_enabled is not None):
            raise ValueError("NIR/LDM controls require Gemini 336")
        if self.nir_pair_max_ms <= 0:
            raise ValueError("nir_pair_max_ms must be positive")
        if not self.serial_number.strip():
            raise ValueError("serial_number must not be empty")
        if self.model not in ("gemini_336", "gemini_305"):
            raise ValueError("unsupported Orbbec model")
        if self.model == "gemini_336" and self.selected_color_stream != "color":
            raise ValueError("Gemini 336 requires the color stream")
        if self.model == "gemini_305":
            if self.preset != "Dual Color Streams":
                raise ValueError("Gemini 305 requires the Dual Color Streams preset")
            if self.selected_color_stream not in ("left_color", "right_color"):
                raise ValueError("Gemini 305 requires a left or right color stream")
        if self.fps is None or self.width is None or self.height is None:
            raise ValueError("fps, width and height are required")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("fps, width and height must be positive")
        if self.warmup_s < 0:
            raise ValueError("warmup_s must be non-negative")
