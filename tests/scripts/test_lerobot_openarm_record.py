from pathlib import Path
import tomllib

import pytest

from lerobot.openarm_data_collection.config import OpenArmRecordConfig


def test_recording_defaults_match_design():
    config = OpenArmRecordConfig(Path("/media/data"), "red_cube", "抓取红块", Path("rig.yaml"))
    assert config.fps == 30
    assert config.min_free_space_gb == 20.0
    assert config.min_episode_sec == 1.0
    assert config.max_episode_sec == 120.0
    assert config.ros_udp_port == 15001
    assert config.image_writer_threads == 4
    assert config.arming_timeout_sec == 3.0
    assert config.arming_stable_sec == 1.0
    assert config.min_effective_fps_ratio == 0.9
    assert config.fps_window_sec == 1.0
    assert config.sync_wait_grace_ms == 12.0


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
    assert "5B61033187" in cfg.leader_left_port
    assert "5B61034924" in cfg.leader_right_port
    assert cfg.teleop_udp_port == 15000


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

    session = RecordingSession(
        Sink(), Sync(), "任务", min_episode_sec=0.0,
        arming_stable_sec=0.0, fps_window_sec=0.0,
    )
    feedback = RecorderFeedback("/tmp/data/task", progress_interval_sec=1.0)
    feedback.ready(session)
    feedback.handle_key(session, "r", 1_000_000_000)
    session.tick(1_000_000_000)
    feedback.observe(session, 1_000_000_000)
    session.tick(1_033_333_333)
    feedback.observe(session, 2_000_000_000)
    session.mark_invalid("camera skew exceeded: actual=48.0ms limit=35.0ms")
    feedback.observe(session, 2_100_000_000)
    output = capsys.readouterr().out
    assert "[READY]" in output
    assert "[ARMING]" in output
    assert "[RECORDING] episode=0 started" in output
    assert "elapsed=1.0s" in output and "effective_fps=1.0" in output
    assert "sync_skipped=0" in output and "deadline_missed=0" in output
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

    session = RecordingSession(
        Sink(), Sync(), "任务", min_episode_sec=0.0,
        arming_stable_sec=0.0, fps_window_sec=0.0,
    )
    feedback = RecorderFeedback("/tmp/data/task")
    reason = "camera skew exceeded: actual=48.0ms limit=35.0ms"

    feedback.handle_key(session, "r", 1_000_000_000)
    session.tick(1_000_000_000)
    feedback.observe(session, 1_000_000_000)
    session.mark_invalid(reason)
    feedback.observe(session, 1_100_000_000)
    feedback.handle_key(session, "d", 1_200_000_000)
    feedback.handle_key(session, "r", 2_000_000_000)
    session.tick(2_000_000_000)
    feedback.observe(session, 2_000_000_000)
    session.mark_invalid(reason)
    feedback.observe(session, 2_100_000_000)

    output = capsys.readouterr().out
    assert output.count("[FAILED] episode=0 camera skew exceeded") == 2


def test_camera_rate_tracker_uses_sdk_sequence_delta():
    from lerobot.scripts.lerobot_openarm_record import CameraRateTracker

    tracker = CameraRateTracker(window_sec=1.0)
    tracker.update("head", sequence=10, received_monotonic_ns=1_000_000_000)
    tracker.update("head", sequence=25, received_monotonic_ns=1_500_000_000)
    tracker.update("head", sequence=40, received_monotonic_ns=2_000_000_000)

    assert tracker.rates(2_000_000_000)["head"] == pytest.approx(30.0)


def test_follower_waypoint_gate_requires_fresh_actual_joint_feedback():
    from lerobot.openarm_data_collection.types import RecordingSnapshot, TimedVector
    from lerobot.scripts.lerobot_openarm_record import follower_matches_waypoint

    now_ns = 2_000_000_000
    target = tuple(float(index) for index in range(16))
    state = TimedVector(target, now_ns, None)
    snapshot = RecordingSnapshot(1, now_ns, state, None, None)

    assert follower_matches_waypoint(snapshot, target, 0.05, now_ns)
    changed = list(target)
    changed[4] += 0.1
    assert not follower_matches_waypoint(
        RecordingSnapshot(2, now_ns, TimedVector(tuple(changed), now_ns, None), None, None),
        target,
        0.05,
        now_ns,
    )
    stale = RecordingSnapshot(3, now_ns, TimedVector(target, 1_000_000_000, None), None, None)
    assert not follower_matches_waypoint(stale, target, 0.05, now_ns)


def test_follower_waypoint_error_reports_largest_joint_error():
    from lerobot.openarm_data_collection.types import RecordingSnapshot, TimedVector
    from lerobot.scripts.lerobot_openarm_record import follower_waypoint_error

    now_ns = 2_000_000_000
    target = tuple(0.0 for _ in range(16))
    actual = list(target)
    actual[11] = 0.12
    snapshot = RecordingSnapshot(
        1,
        now_ns,
        TimedVector(tuple(actual), now_ns, None),
        None,
        None,
    )

    message = follower_waypoint_error(snapshot, target, 0.05, now_ns)

    assert "openarm_right_joint4" in message
    assert "error=0.120000rad" in message
    assert "tolerance=0.050000rad" in message


def test_feedback_arming_failure_includes_camera_rates(capsys):
    from lerobot.openarm_data_collection.session import RecordingSession
    from lerobot.scripts.lerobot_openarm_record import CameraRateTracker, RecorderFeedback

    class Sink:
        total_episodes = 0
        def begin_episode(self, task): return 0
        def add_sample(self, sample): pass
        def save_episode(self): return 0
        def discard_episode(self): pass
        def finalize(self): pass

    class Sync:
        def select(self, now): return None
        def health(self, now):
            return type(
                "Health", (),
                {"fatal": False, "reason": "head missing", "category": "head_missing"},
            )()

    tracker = CameraRateTracker(window_sec=1.0)
    tracker.update("head", 1, 0)
    tracker.update("head", 31, 1_000_000_000)
    session = RecordingSession(
        Sink(), Sync(), "任务", fps=10, arming_timeout_sec=1.0,
        arming_stable_sec=1.0, sync_wait_grace_ms=0.0,
    )
    feedback = RecorderFeedback("/tmp/data/task", camera_rates=tracker)

    feedback.handle_key(session, "r", 0)
    for index in range(11):
        now_ns = index * 100_000_000
        session.tick(now_ns)
        feedback.observe(session, now_ns)

    output = capsys.readouterr().out
    assert "[FAILED] arming" in output
    assert "successful=0 required=9" in output
    assert "camera_fps={'head': 30.0}" in output
