import pytest

from lerobot.openarm_data_collection.session import InvalidTransition, RecordingSession, SessionState


class Sink:
    total_episodes = 0
    def __init__(self): self.saved = self.discarded = self.frames = 0
    def begin_episode(self, task): return 0
    def add_sample(self, sample): self.frames += 1
    def save_episode(self): self.saved += 1; return 0
    def discard_episode(self): self.discarded += 1
    def finalize(self): pass


class Sync:
    def __init__(self): self.value = object(); self.fatal = False
    def select(self, now): return self.value
    def health(self, now):
        return type(
            "Health", (),
            {"fatal": self.fatal, "reason": "right_wrist missing", "category": "right_wrist_missing"},
        )()


def armed_session(sink=None, sync=None, **kwargs):
    sink, sync = sink or Sink(), sync or Sync()
    kwargs.setdefault("arming_stable_sec", 0.0)
    kwargs.setdefault("fps_window_sec", 0.0)
    kwargs.setdefault("min_episode_sec", 0.0)
    session = RecordingSession(
        sink, sync, "任务", **kwargs,
    )
    session.handle_key("r", 0)
    session.tick(0)
    assert session.state is SessionState.RECORDING
    return session, sink, sync


def test_invalid_episode_can_only_be_discarded():
    session, sink, sync = armed_session()
    session.mark_invalid("camera timeout")
    with pytest.raises(InvalidTransition): session.handle_key("s", 1)
    session.handle_key("d", 1)
    assert session.state is SessionState.READY
    assert sink.discarded == 1


def test_sampling_and_save():
    session, sink, sync = armed_session()
    session.tick(33_333_333)
    session.pause_for_decision(33_333_333)
    session.handle_key("s", 33_333_333)
    assert sink.frames == 1 and sink.saved == 1


def test_pause_for_decision_stops_sampling_until_save_or_discard():
    session, sink, sync = armed_session(fps=10)
    session.next_sample_ns = 100_000_000
    session.tick(100_000_000)
    session.pause_for_decision(100_000_000)

    session.tick(1_000_000_000)

    assert session.state is SessionState.AWAITING_DECISION
    assert sink.frames == 1
    assert session.status(2_000_000_000).elapsed_sec == 0.1
    session.handle_key("d", 2_000_000_000)
    assert sink.discarded == 1


def test_q_only_from_ready():
    session = RecordingSession(Sink(), Sync(), "任务")
    session.handle_key("r", 0)
    with pytest.raises(InvalidTransition): session.handle_key("q", 1)


def test_status_exposes_episode_elapsed_and_effective_fps():
    session, _, _ = armed_session()
    session.started_ns = 1_000_000_000
    session.next_sample_ns = 1_033_333_333
    session.tick(1_033_333_333)
    status = session.status(3_000_000_000)
    assert status.episode_index == 0
    assert status.elapsed_sec == 2.0
    assert status.effective_fps == 0.5


def test_arming_does_not_create_or_write_episode_until_sources_are_stable():
    sink, sync = Sink(), Sync()
    session = RecordingSession(
        sink, sync, "任务", fps=10, arming_stable_sec=1.0,
        arming_timeout_sec=3.0, fps_window_sec=1.0,
    )

    session.handle_key("r", 0)
    assert session.state is SessionState.ARMING
    assert sink.frames == 0
    for index in range(11):
        session.tick(index * 100_000_000)

    assert session.state is SessionState.RECORDING
    assert sink.frames == 0
    assert session.started_ns == 1_000_000_000


def test_arming_wait_grace_accepts_a_late_synchronized_sample():
    sink, sync = Sink(), Sync()
    session = RecordingSession(
        sink, sync, "任务", fps=10, arming_stable_sec=0.0,
        arming_timeout_sec=1.0, fps_window_sec=0.0, sync_wait_grace_ms=12.0,
    )
    sync.value = None

    session.handle_key("r", 0)
    session.tick(0)
    assert session.state is SessionState.ARMING

    sync.value = object()
    session.tick(11_000_000)
    assert session.state is SessionState.RECORDING


def test_arming_uses_success_ratio_instead_of_requiring_zero_failures():
    sink, sync = Sink(), Sync()
    session = RecordingSession(
        sink, sync, "任务", fps=10, arming_stable_sec=1.0,
        arming_timeout_sec=3.0, min_effective_fps_ratio=0.9,
        fps_window_sec=1.0, sync_wait_grace_ms=0.0,
    )

    session.handle_key("r", 0)
    for index in range(11):
        sync.value = None if index in (3, 7) else object()
        session.tick(index * 100_000_000)

    assert session.state is SessionState.RECORDING


def test_arming_timeout_exposes_threshold_and_structured_failures():
    sink, sync = Sink(), Sync()
    session = RecordingSession(
        sink, sync, "任务", fps=10, arming_stable_sec=1.0,
        arming_timeout_sec=1.0, min_effective_fps_ratio=0.9,
        fps_window_sec=1.0, sync_wait_grace_ms=0.0,
    )
    sync.value = None

    session.handle_key("r", 0)
    for index in range(11):
        session.tick(index * 100_000_000)

    status = session.status(1_000_000_000)
    assert status.state is SessionState.READY
    assert status.arming_successful == 0
    assert status.arming_required == 9
    assert status.sync_failures == {"right_wrist_missing": 11}
    assert "successful=0 required=9" in status.reason


def test_deadline_misses_are_counted_separately_from_sync_skips():
    session, _, sync = armed_session(fps=10, sync_wait_grace_ms=0.0)
    session.next_sample_ns = 100_000_000
    session.tick(350_000_000)
    assert session.frames == 1
    assert session.deadline_missed == 2
    assert session.sync_skipped == 0

    sync.value = None
    session.tick(400_000_000)
    assert session.sync_skipped == 1


def test_sync_wait_grace_accepts_a_frame_that_arrives_after_target():
    session, sink, sync = armed_session(fps=10, sync_wait_grace_ms=12.0)
    session.next_sample_ns = 100_000_000
    sync.value = None

    session.tick(100_000_000)
    assert session.sync_skipped == 0
    assert sink.frames == 0

    sync.value = object()
    session.tick(111_000_000)
    assert session.sync_skipped == 0
    assert sink.frames == 1


def test_sync_wait_timeout_records_structured_failure_reason():
    session, sink, sync = armed_session(fps=10, sync_wait_grace_ms=12.0)
    session.next_sample_ns = 100_000_000
    sync.value = None

    session.tick(100_000_000)
    session.tick(111_999_999)
    assert session.sync_skipped == 0
    session.tick(112_000_000)

    status = session.status(112_000_000)
    assert status.sync_skipped == 1
    assert status.sync_failures == {"right_wrist_missing": 1}
    assert sink.frames == 0


def test_sustained_low_fps_becomes_invalid_and_cannot_be_saved():
    session, _, _ = armed_session(
        fps=10, min_effective_fps_ratio=0.9, fps_check_grace_sec=1.0,
        fps_failure_duration_sec=1.0, fps_window_sec=1.0,
    )
    session.next_sample_ns = 100_000_000

    for now_ns in (500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000):
        session.tick(now_ns)

    assert session.state is SessionState.INVALID
    assert "effective FPS too low" in session.reason
    with pytest.raises(InvalidTransition):
        session.handle_key("s", 2_100_000_000)


def test_final_save_rejects_low_average_fps():
    session, _, _ = armed_session(
        fps=10, min_effective_fps_ratio=0.9, fps_check_grace_sec=100.0,
    )
    session.next_sample_ns = 100_000_000
    session.tick(1_000_000_000)
    session.pause_for_decision(1_100_000_000)

    with pytest.raises(InvalidTransition, match="effective FPS too low"):
        session.handle_key("s", 1_100_000_000)
    assert session.state is SessionState.INVALID
