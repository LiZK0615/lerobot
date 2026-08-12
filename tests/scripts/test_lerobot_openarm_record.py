from pathlib import Path

from lerobot.openarm_data_collection.config import OpenArmRecordConfig


def test_recording_defaults_match_design():
    config = OpenArmRecordConfig(Path("/media/data"), "red_cube", "抓取红块", Path("rig.yaml"))
    assert config.fps == 30
    assert config.min_free_space_gb == 20.0
    assert config.min_episode_sec == 1.0
    assert config.max_episode_sec == 120.0
    assert config.record_command_diagnostics is False
    assert config.ros_udp_port == 15001


def test_console_script_is_registered():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'lerobot-openarm-record="lerobot.scripts.lerobot_openarm_record:main"' in pyproject.read_text()


def test_script_exposes_main():
    from lerobot.scripts.lerobot_openarm_record import main
    assert callable(main)
