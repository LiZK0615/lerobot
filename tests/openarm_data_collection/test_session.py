import pytest

from lerobot.openarm_data_collection.session import InvalidTransition, RecordingSession, SessionState


class Sink:
    def __init__(self): self.saved = self.discarded = self.frames = 0
    def begin_episode(self, task): return 0
    def add_sample(self, sample): self.frames += 1
    def save_episode(self): self.saved += 1; return 0
    def discard_episode(self): self.discarded += 1
    def finalize(self): pass


class Sync:
    def __init__(self): self.value = object(); self.fatal = False
    def select(self, now): return self.value
    def health(self, now): return type("Health", (), {"fatal": self.fatal, "reason": "timeout"})()


def test_invalid_episode_can_only_be_discarded():
    sink, sync = Sink(), Sync()
    session = RecordingSession(sink, sync, "任务", min_episode_sec=0.0)
    session.handle_key("r", 0)
    session.mark_invalid("camera timeout")
    with pytest.raises(InvalidTransition): session.handle_key("s", 1)
    session.handle_key("d", 1)
    assert session.state is SessionState.READY
    assert sink.discarded == 1


def test_sampling_and_save():
    sink, sync = Sink(), Sync()
    session = RecordingSession(sink, sync, "任务", min_episode_sec=0.0)
    session.handle_key("r", 0)
    session.tick(33_333_333)
    session.handle_key("s", 33_333_333)
    assert sink.frames == 1 and sink.saved == 1


def test_q_only_from_ready():
    session = RecordingSession(Sink(), Sync(), "任务")
    session.handle_key("r", 0)
    with pytest.raises(InvalidTransition): session.handle_key("q", 1)


def test_status_exposes_episode_elapsed_and_effective_fps():
    session = RecordingSession(Sink(), Sync(), "任务", min_episode_sec=0.0)
    session.handle_key("r", 1_000_000_000)
    session.tick(1_033_333_333)
    status = session.status(3_000_000_000)
    assert status.episode_index == 0
    assert status.elapsed_sec == 2.0
    assert status.effective_fps == 0.5
