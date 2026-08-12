"""Optional OpenCV preview for the three recording cameras."""

import os
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
QT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/truetype/freefont"),
)


def configure_qt_font_dir(environ=os.environ, candidates=QT_FONT_CANDIDATES) -> None:
    configured = environ.get("QT_QPA_FONTDIR")
    if configured and Path(configured).is_dir():
        return
    for candidate in candidates:
        if candidate.is_dir():
            environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def compose_preview(
    frames: dict[str, NDArray[np.uint8]],
    status_text: str,
    panel_width: int = 320,
    cv2_module: Any = cv2,
) -> NDArray[np.uint8]:
    panels = []
    for name in CAMERA_NAMES:
        image = frames.get(name)
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        height = max(1, round(image.shape[0] * panel_width / image.shape[1]))
        panel = cv2_module.resize(image, (panel_width, height), interpolation=getattr(cv2_module, "INTER_AREA", 3))
        panel = cv2_module.cvtColor(panel, getattr(cv2_module, "COLOR_RGB2BGR", 4))
        cv2_module.putText(panel, name, (10, 24), getattr(cv2_module, "FONT_HERSHEY_SIMPLEX", 0), 0.65, (0, 255, 0), 2)
        panels.append(panel)
    canvas = np.hstack(panels)
    cv2_module.putText(
        canvas, status_text, (10, canvas.shape[0] - 12),
        getattr(cv2_module, "FONT_HERSHEY_SIMPLEX", 0), 0.55, (0, 255, 255), 2,
    )
    return canvas


class CameraPreview:
    WINDOW_NAME = "OpenArm Data Collection"

    def __init__(self, enabled: bool, cv2_module: Any = cv2) -> None:
        if enabled:
            configure_qt_font_dir()
        self.enabled = enabled
        self.failure: str | None = None
        self._cv2 = cv2_module
        self._opened = False

    def poll(self, frames: dict[str, NDArray[np.uint8]], status_text: str) -> str | None:
        if not self.enabled:
            return None
        try:
            canvas = compose_preview(frames, status_text, cv2_module=self._cv2)
            self._cv2.imshow(self.WINDOW_NAME, canvas)
            self._opened = True
            key_code = self._cv2.waitKey(1)
            if self._cv2.getWindowProperty(self.WINDOW_NAME, self._cv2.WND_PROP_VISIBLE) < 1:
                self.close()
                return None
            if key_code < 0:
                return None
            key = chr(key_code & 0xFF).lower()
            return key if key in "rsdq" else None
        except Exception as error:
            self.failure = str(error)
            self.close()
            return None

    def close(self) -> None:
        if self._opened:
            try:
                self._cv2.destroyWindow(self.WINDOW_NAME)
            except Exception:
                pass
        self._opened = False
        self.enabled = False
