#!/usr/bin/env python
"""Interactive passive recorder for OpenArm bimanual LeRobot datasets."""

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import time

import draccus

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecSdkRuntime
from lerobot.openarm_data_collection.config import OpenArmRecordConfig, load_camera_rig, validate_storage
from lerobot.openarm_data_collection.dataset_sink import DatasetSink
from lerobot.openarm_data_collection.preview import CameraPreview
from lerobot.openarm_data_collection.ros_receiver import RosSnapshotReceiver
from lerobot.openarm_data_collection.session import RecordingSession, SessionState
from lerobot.openarm_data_collection.terminal import TerminalKeys
from lerobot.openarm_data_collection.time_sync import DeviceClockMapper, SampleSynchronizer, SystemClockMapper


@dataclass
class OpenArmRecordCliConfig:
    dataset_root: str
    dataset_name: str
    task: str
    camera_config: str
    expected_mount: str = ""
    fps: int = 30
    min_free_space_gb: float = 20.0
    min_episode_sec: float = 1.0
    max_episode_sec: float = 120.0
    image_writer_threads: int = 4
    arming_timeout_sec: float = 3.0
    arming_stable_sec: float = 1.0
    min_effective_fps_ratio: float = 0.90
    fps_check_grace_sec: float = 3.0
    fps_failure_duration_sec: float = 2.0
    fps_window_sec: float = 1.0
    sync_wait_grace_ms: float = 12.0
    ros_udp_port: int = 15001
    display_cameras: bool = False

    def core(self) -> OpenArmRecordConfig:
        return OpenArmRecordConfig(
            dataset_root=Path(self.dataset_root),
            dataset_name=self.dataset_name,
            task=self.task,
            camera_config=Path(self.camera_config),
            expected_mount=Path(self.expected_mount) if self.expected_mount else None,
            fps=self.fps,
            min_free_space_gb=self.min_free_space_gb,
            min_episode_sec=self.min_episode_sec,
            max_episode_sec=self.max_episode_sec,
            image_writer_threads=self.image_writer_threads,
            arming_timeout_sec=self.arming_timeout_sec,
            arming_stable_sec=self.arming_stable_sec,
            min_effective_fps_ratio=self.min_effective_fps_ratio,
            fps_check_grace_sec=self.fps_check_grace_sec,
            fps_failure_duration_sec=self.fps_failure_duration_sec,
            fps_window_sec=self.fps_window_sec,
            sync_wait_grace_ms=self.sync_wait_grace_ms,
            ros_udp_port=self.ros_udp_port,
            display_cameras=self.display_cameras,
        )


class RecorderFeedback:
    def __init__(self, dataset_path: str, progress_interval_sec: float = 1.0, camera_rates=None) -> None:
        self.dataset_path = dataset_path
        self.progress_interval_ns = round(progress_interval_sec * 1_000_000_000)
        self._last_state: SessionState | None = None
        self._last_progress_ns: int | None = None
        self._failed_reason: str | None = None
        self.camera_rates = camera_rates

    def ready(self, session: RecordingSession) -> None:
        print(f"[READY] next_episode={session.sink.total_episodes} [r] record [q] finalize", flush=True)
        self._last_state = session.state

    @staticmethod
    def status_text(session: RecordingSession, now_ns: int) -> str:
        status = session.status(now_ns)
        text = status.state.value
        if status.episode_index is not None:
            if status.state is SessionState.ARMING:
                text += f" episode={status.episode_index} warmup={status.arming_elapsed_sec:.1f}s"
            else:
                text += (
                    f" episode={status.episode_index} elapsed={status.elapsed_sec:.1f}s "
                    f"frames={status.frames} sync_skipped={status.sync_skipped} "
                    f"deadline_missed={status.deadline_missed}"
                )
        if status.reason:
            text += f" FAILED: {status.reason}"
        return text

    def handle_key(self, session: RecordingSession, key: str, now_ns: int) -> None:
        before = session.status(now_ns)
        try:
            if key == "s" and before.state is SessionState.RECORDING:
                print(
                    f"[SAVING] episode={before.episode_index} frames={before.frames} "
                    f"duration={before.elapsed_sec:.1f}s encoding videos",
                    flush=True,
                )
            session.handle_key(key, now_ns)
        except Exception as error:
            print(f"[FAILED] key={key} state={before.state.value} reason={error}", flush=True)
            return
        after = session.status(now_ns)
        if key == "r":
            self._failed_reason = None
            print(
                f"[ARMING] episode={after.episode_index} waiting for stable synchronized sources",
                flush=True,
            )
            self._last_progress_ns = now_ns
        elif key == "s":
            print(f"[SAVED] episode={before.episode_index} path={self.dataset_path}", flush=True)
            self.ready(session)
        elif key == "d":
            print(f"[DISCARDED] episode={before.episode_index}", flush=True)
            self.ready(session)
        elif key == "q":
            print(f"[FINALIZED] path={self.dataset_path}", flush=True)
        self._last_state = after.state

    def observe(self, session: RecordingSession, now_ns: int) -> None:
        status = session.status(now_ns)
        if self._last_state is SessionState.ARMING and status.state is SessionState.RECORDING:
            print(f"[RECORDING] episode={status.episode_index} started", flush=True)
            self._last_progress_ns = now_ns
        elif self._last_state is SessionState.ARMING and status.state is SessionState.READY and status.reason:
            print(
                f"[FAILED] arming reason={status.reason} "
                f"camera_fps={self.camera_rates.rates(now_ns) if self.camera_rates else {}}",
                flush=True,
            )
            self.ready(session)
        if status.state is SessionState.INVALID and status.reason != self._failed_reason:
            print(f"[FAILED] episode={status.episode_index} {status.reason}; press d to discard", flush=True)
            self._failed_reason = status.reason
        if status.state is SessionState.ARMING and (
            self._last_progress_ns is None or now_ns - self._last_progress_ns >= self.progress_interval_ns
        ):
            print(
                f"[ARMING] episode={status.episode_index} warmup={status.arming_elapsed_sec:.1f}s "
                f"synchronized={status.arming_successful}/{status.arming_required} "
                f"camera_fps={self.camera_rates.rates(now_ns) if self.camera_rates else {}} "
                f"sync_failures={status.sync_failures}",
                flush=True,
            )
            self._last_progress_ns = now_ns
        if status.state is SessionState.RECORDING and (
            self._last_progress_ns is None or now_ns - self._last_progress_ns >= self.progress_interval_ns
        ):
            print(
                f"[RECORDING] episode={status.episode_index} elapsed={status.elapsed_sec:.1f}s "
                f"frames={status.frames} sync_skipped={status.sync_skipped} "
                f"deadline_missed={status.deadline_missed} "
                f"effective_fps={status.effective_fps:.1f} window_fps={status.window_fps:.1f} "
                f"camera_fps={self.camera_rates.rates(now_ns) if self.camera_rates else {}} "
                f"sync_failures={status.sync_failures}",
                flush=True,
            )
            self._last_progress_ns = now_ns
        self._last_state = status.state


def map_camera_timestamp(
    packet, system_mapper: SystemClockMapper, device_mapper: DeviceClockMapper
) -> tuple[int, str]:
    if packet.system_timestamp_us is not None:
        return system_mapper.update(packet.system_timestamp_us, packet.received_monotonic_ns), "system"
    return device_mapper.update(packet.device_timestamp_us, packet.received_monotonic_ns), "device_fallback"


class CameraRateTracker:
    def __init__(self, window_sec: float = 1.0) -> None:
        self.window_ns = round(window_sec * 1_000_000_000)
        self._samples: dict[str, deque[tuple[int, int]]] = {}

    def update(self, name: str, sequence: int, received_monotonic_ns: int) -> None:
        samples = self._samples.setdefault(name, deque())
        samples.append((received_monotonic_ns, sequence))
        cutoff = received_monotonic_ns - self.window_ns
        while len(samples) > 2 and samples[1][0] <= cutoff:
            samples.popleft()

    def rates(self, now_ns: int) -> dict[str, float]:
        result = {}
        cutoff = now_ns - self.window_ns
        for name, samples in self._samples.items():
            while len(samples) > 2 and samples[1][0] <= cutoff:
                samples.popleft()
            if len(samples) < 2 or samples[-1][0] <= samples[0][0]:
                result[name] = 0.0
                continue
            elapsed = (samples[-1][0] - samples[0][0]) / 1_000_000_000
            result[name] = round((samples[-1][1] - samples[0][1]) / elapsed, 1)
        return result


def run(cfg: OpenArmRecordCliConfig) -> None:
    cfg = cfg.core()
    validate_storage(cfg.dataset_root, cfg.min_free_space_gb, cfg.expected_mount)
    camera_configs = load_camera_rig(cfg.camera_config)
    receiver = RosSnapshotReceiver(port=cfg.ros_udp_port)
    runtime = OrbbecSdkRuntime()
    cameras = {name: OrbbecCamera(config, runtime) for name, config in camera_configs.items()}
    system_clock_offset_ns = time.time_ns() - time.monotonic_ns()
    system_mappers = {
        name: SystemClockMapper(realtime_minus_monotonic_ns=system_clock_offset_ns) for name in cameras
    }
    device_mappers = {name: DeviceClockMapper() for name in cameras}
    synchronizer = SampleSynchronizer()
    repo_id = f"local/{cfg.dataset_name}"
    sink = DatasetSink(
        cfg.dataset_path, repo_id, fps=cfg.fps,
        min_episode_sec=cfg.min_episode_sec, max_episode_sec=cfg.max_episode_sec,
        image_writer_threads=cfg.image_writer_threads,
    )
    session = RecordingSession(
        sink, synchronizer, cfg.task, cfg.fps,
        cfg.min_episode_sec, cfg.max_episode_sec,
        cfg.arming_timeout_sec, cfg.arming_stable_sec,
        cfg.min_effective_fps_ratio, cfg.fps_check_grace_sec,
        cfg.fps_failure_duration_sec, cfg.fps_window_sec,
        cfg.sync_wait_grace_ms,
    )
    camera_rates = CameraRateTracker(cfg.fps_window_sec)
    feedback = RecorderFeedback(str(cfg.dataset_path), camera_rates=camera_rates)
    preview = CameraPreview(cfg.display_cameras)
    last_camera_sequence = {name: -1 for name in cameras}
    latest_images = {}
    closed = False

    try:
        for camera in cameras.values(): camera.connect()
        feedback.ready(session)
        with TerminalKeys() as keys:
            while session.state is not SessionState.EXITED:
                now_ns = time.monotonic_ns()
                snapshot = receiver.poll()
                if snapshot is not None: synchronizer.push_snapshot(snapshot)
                for name, camera in cameras.items():
                    packet = camera.read_latest_packet()
                    if packet is None or packet.sequence == last_camera_sequence[name]: continue
                    last_camera_sequence[name] = packet.sequence
                    latest_images[name] = packet.image
                    camera_rates.update(name, packet.sequence, packet.received_monotonic_ns)
                    try:
                        mapped, _time_source = map_camera_timestamp(
                            packet, system_mappers[name], device_mappers[name]
                        )
                        synchronizer.push_camera(name, replace(packet, mapped_monotonic_ns=mapped))
                    except ValueError as error:
                        source = "system" if packet.system_timestamp_us is not None else "device_fallback"
                        session.mark_invalid(f"{name} {source} timestamp mapping failed: {error}")
                terminal_key = keys.poll()
                window_key = preview.poll(latest_images, feedback.status_text(session, now_ns))
                if preview.failure:
                    print(f"[FAILED] camera_preview reason={preview.failure}; preview disabled", flush=True)
                    preview.failure = None
                key = terminal_key or window_key
                if key: feedback.handle_key(session, key, now_ns)
                session.tick(now_ns)
                feedback.observe(session, now_ns)
                time.sleep(0.002)
    except KeyboardInterrupt:
        if session.state in (SessionState.RECORDING, SessionState.INVALID): sink.discard_episode()
    finally:
        try: sink.finalize()
        finally:
            preview.close()
            receiver.close()
            for camera in cameras.values():
                if camera.is_connected: camera.disconnect()


@draccus.wrap()
def _main(cfg: OpenArmRecordCliConfig) -> None:
    run(cfg)


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
