from pathlib import Path
import tomllib

from lerobot.openarm_data_collection.config import OpenArmRecordConfig


def test_recording_defaults_match_design():
    config = OpenArmRecordConfig(Path("/media/data"), "red_cube", "抓取红块", Path("rig.yaml"))
    assert config.fps == 30
    assert config.min_free_space_gb == 20.0
    assert config.min_episode_sec == 1.0
    assert config.max_episode_sec == 120.0
    assert config.ros_udp_port == 15001


def test_console_script_is_registered():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'lerobot-openarm-record="lerobot.scripts.lerobot_openarm_record:main"' in pyproject.read_text()


def test_script_exposes_main():
    from lerobot.scripts.lerobot_openarm_record import main
    assert callable(main)


def test_cli_defaults_to_no_camera_display():
    from lerobot.scripts.lerobot_openarm_record import OpenArmRecordCliConfig

    cfg = OpenArmRecordCliConfig("/tmp/data", "task", "任务", "rig.yaml")
    assert cfg.display_cameras is False
    assert cfg.core().display_cameras is False


def test_cli_passes_camera_display_to_core_config():
    from lerobot.scripts.lerobot_openarm_record import OpenArmRecordCliConfig

    cfg = OpenArmRecordCliConfig("/tmp/data", "task", "任务", "rig.yaml", display_cameras=True)
    assert cfg.core().display_cameras is True


def test_arm64_uses_gui_opencv_and_other_platforms_use_headless():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

    assert 'opencv-python>=4.9.0,<4.14.0; sys_platform == "linux" and (platform_machine == "aarch64" or platform_machine == "arm64")' in dependencies
    assert 'opencv-python-headless>=4.9.0,<4.14.0; sys_platform != "linux" or (platform_machine != "aarch64" and platform_machine != "arm64")' in dependencies


def test_camera_timestamp_prefers_sdk_system_timestamp():
    from lerobot.cameras.orbbec import OrbbecFrame
    from lerobot.scripts.lerobot_openarm_record import map_camera_timestamp

    packet = OrbbecFrame(None, "serial", "color", 123, 1_700_000_002_000_000, 2_012_000_000, None, 0)

    class Mapper:
        def update(self, timestamp, received):
            assert timestamp == 1_700_000_002_000_000
            assert received == 2_012_000_000
            return 2_000_000_000

    class DeviceMapper:
        def update(self, timestamp, received):
            raise AssertionError("device mapper must not be used when system timestamp exists")

    assert map_camera_timestamp(packet, Mapper(), DeviceMapper()) == (2_000_000_000, "system")


def test_camera_timestamp_falls_back_to_device_timestamp():
    from lerobot.cameras.orbbec import OrbbecFrame
    from lerobot.scripts.lerobot_openarm_record import map_camera_timestamp

    packet = OrbbecFrame(None, "serial", "color", 123, None, 2_012_000_000, None, 0)

    class SystemMapper:
        def update(self, timestamp, received):
            raise AssertionError("system mapper must not be used without system timestamp")

    class DeviceMapper:
        def update(self, timestamp, received):
            assert timestamp == 123
            assert received == 2_012_000_000
            return 2_000_000_000

    assert map_camera_timestamp(packet, SystemMapper(), DeviceMapper()) == (2_000_000_000, "device_fallback")


def test_feedback_reports_lifecycle_progress_and_failures(capsys):
    from lerobot.scripts.lerobot_openarm_record import RecorderFeedback
    from lerobot.openarm_data_collection.session import RecordingSession

    class Sink:
        total_episodes = 0
        def begin_episode(self, task): return 0
        def add_sample(self, sample): pass
        def save_episode(self): return 0
        def discard_episode(self): pass
        def finalize(self): pass

    class Sync:
        def select(self, now): return object()
        def health(self, now): return type("Health", (), {"fatal": False, "reason": None})()

    session = RecordingSession(Sink(), Sync(), "任务", min_episode_sec=0.0)
    feedback = RecorderFeedback("/tmp/data/task", progress_interval_sec=1.0)
    feedback.ready(session)
    feedback.handle_key(session, "r", 1_000_000_000)
    session.tick(1_033_333_333)
    feedback.observe(session, 2_000_000_000)
    session.mark_invalid("camera skew exceeded: actual=48.0ms limit=35.0ms")
    feedback.observe(session, 2_100_000_000)
    output = capsys.readouterr().out
    assert "[READY]" in output
    assert "[RECORDING] episode=0 started" in output
    assert "elapsed=1.0s" in output and "effective_fps=1.0" in output
    assert "[FAILED] episode=0 camera skew exceeded" in output


def test_feedback_reports_same_failure_again_in_a_new_episode(capsys):
    from lerobot.scripts.lerobot_openarm_record import RecorderFeedback
    from lerobot.openarm_data_collection.session import RecordingSession

    class Sink:
        total_episodes = 0
        def begin_episode(self, task): return 0
        def add_sample(self, sample): pass
        def save_episode(self): return 0
        def discard_episode(self): pass
        def finalize(self): pass

    class Sync:
        def select(self, now): return object()
        def health(self, now): return type("Health", (), {"fatal": False, "reason": None})()

    session = RecordingSession(Sink(), Sync(), "任务", min_episode_sec=0.0)
    feedback = RecorderFeedback("/tmp/data/task")
    reason = "camera skew exceeded: actual=48.0ms limit=35.0ms"

    feedback.handle_key(session, "r", 1_000_000_000)
    session.mark_invalid(reason)
    feedback.observe(session, 1_100_000_000)
    feedback.handle_key(session, "d", 1_200_000_000)
    feedback.handle_key(session, "r", 2_000_000_000)
    session.mark_invalid(reason)
    feedback.observe(session, 2_100_000_000)

    output = capsys.readouterr().out
    assert output.count("[FAILED] episode=0 camera skew exceeded") == 2
