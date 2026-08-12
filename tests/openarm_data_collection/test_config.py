from pathlib import Path

import pytest

from lerobot.openarm_data_collection.config import OpenArmRecordConfig, StorageError, load_camera_rig, validate_storage


@pytest.mark.parametrize("name", ["../escape", "中文名", "a/b", ""])
def test_dataset_name_rejects_unsafe_values(name):
    with pytest.raises(ValueError):
        OpenArmRecordConfig(Path("/media/data"), name, "抓取", Path("rig.yaml"))


def test_camera_rig_requires_three_unique_serials(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text("""cameras:
  head: {serial_number: same, model: gemini_336, selected_color_stream: color}
  left_wrist: {serial_number: same, model: gemini_305, preset: Dual Color Streams, selected_color_stream: right_color}
  right_wrist: {serial_number: right, model: gemini_305, preset: Dual Color Streams, selected_color_stream: left_color}
""")
    with pytest.raises(ValueError, match="unique"):
        load_camera_rig(path)


def test_storage_rejects_wrong_expected_mount(monkeypatch, tmp_path):
    monkeypatch.setattr("lerobot.openarm_data_collection.config._mount_point", lambda path: Path("/"))
    with pytest.raises(StorageError, match="mounted filesystem"):
        validate_storage(tmp_path, 0.0, expected_mount=tmp_path)
