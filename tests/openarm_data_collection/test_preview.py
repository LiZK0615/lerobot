import numpy as np

from lerobot.openarm_data_collection.preview import CameraPreview, compose_preview


def test_compose_preview_builds_three_labeled_panels():
    frames = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("head", "left_wrist", "right_wrist")}
    canvas = compose_preview(frames, "RECORDING episode=0 elapsed=1.0s", panel_width=320)
    assert canvas.shape == (240, 960, 3)
    assert canvas.dtype == np.uint8


class FakeCv2:
    WND_PROP_VISIBLE = 1
    COLOR_RGB2BGR = 2
    INTER_AREA = 3

    def __init__(self, key=-1, visible=1.0, fail=False):
        self.key, self.visible, self.fail = key, visible, fail
        self.shown = 0

    def cvtColor(self, image, code): return image
    def resize(self, image, size, interpolation=None): return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    def putText(self, image, *args, **kwargs): return image
    def imshow(self, name, image):
        if self.fail: raise RuntimeError("no display")
        self.shown += 1
    def waitKey(self, delay): return self.key
    def getWindowProperty(self, name, prop): return self.visible
    def destroyWindow(self, name): pass


def test_preview_accepts_window_keys_and_close_only_disables_preview():
    frames = {name: np.zeros((10, 10, 3), dtype=np.uint8) for name in ("head", "left_wrist", "right_wrist")}
    preview = CameraPreview(True, cv2_module=FakeCv2(key=ord("s")))
    assert preview.poll(frames, "READY") == "s"
    preview = CameraPreview(True, cv2_module=FakeCv2(visible=0.0))
    assert preview.poll(frames, "READY") is None
    assert not preview.enabled


def test_preview_gui_failure_disables_preview_and_exposes_failure():
    frames = {name: np.zeros((10, 10, 3), dtype=np.uint8) for name in ("head", "left_wrist", "right_wrist")}
    preview = CameraPreview(True, cv2_module=FakeCv2(fail=True))
    assert preview.poll(frames, "READY") is None
    assert not preview.enabled
    assert "no display" in preview.failure
