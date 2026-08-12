#!/usr/bin/env python
"""Interactive passive recorder for OpenArm bimanual LeRobot datasets."""

from dataclasses import dataclass, replace
from pathlib import Path
import time

import draccus

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecSdkRuntime
from lerobot.openarm_data_collection.config import OpenArmRecordConfig, load_camera_rig, validate_storage
from lerobot.openarm_data_collection.dataset_sink import DatasetSink
from lerobot.openarm_data_collection.ros_receiver import RosSnapshotReceiver
from lerobot.openarm_data_collection.session import InvalidTransition, RecordingSession, SessionState
from lerobot.openarm_data_collection.terminal import TerminalKeys
from lerobot.openarm_data_collection.time_sync import DeviceClockMapper, SampleSynchronizer


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
    record_command_diagnostics: bool = False
    ros_udp_port: int = 15001

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
            record_command_diagnostics=self.record_command_diagnostics,
            ros_udp_port=self.ros_udp_port,
        )


def run(cfg: OpenArmRecordCliConfig) -> None:
    cfg = cfg.core()
    validate_storage(cfg.dataset_root, cfg.min_free_space_gb, cfg.expected_mount)
    camera_configs = load_camera_rig(cfg.camera_config)
    receiver = RosSnapshotReceiver(port=cfg.ros_udp_port)
    runtime = OrbbecSdkRuntime()
    cameras = {name: OrbbecCamera(config, runtime) for name, config in camera_configs.items()}
    mappers = {name: DeviceClockMapper() for name in cameras}
    synchronizer = SampleSynchronizer()
    repo_id = f"local/{cfg.dataset_name}"
    sink = DatasetSink(
        cfg.dataset_path, repo_id, fps=cfg.fps,
        min_episode_sec=cfg.min_episode_sec, max_episode_sec=cfg.max_episode_sec
    )
    session = RecordingSession(
        sink, synchronizer, cfg.task, cfg.fps,
        cfg.min_episode_sec, cfg.max_episode_sec
    )
    last_camera_sequence = {name: -1 for name in cameras}
    closed = False

    try:
        for camera in cameras.values(): camera.connect()
        print("READY  [r] record  [s] save  [d] discard  [q] finalize")
        with TerminalKeys() as keys:
            while session.state is not SessionState.EXITED:
                now_ns = time.monotonic_ns()
                snapshot = receiver.poll()
                if snapshot is not None: synchronizer.push_snapshot(snapshot)
                for name, camera in cameras.items():
                    packet = camera.read_latest_packet()
                    if packet is None or packet.sequence == last_camera_sequence[name]: continue
                    last_camera_sequence[name] = packet.sequence
                    try:
                        mapped = mappers[name].update(packet.device_timestamp_us, packet.received_monotonic_ns)
                        synchronizer.push_camera(name, replace(packet, mapped_monotonic_ns=mapped))
                    except ValueError:
                        session.mark_invalid(f"{name} timestamp mapping failed")
                key = keys.poll()
                if key:
                    try: session.handle_key(key, now_ns)
                    except InvalidTransition as error: print(f"Ignored key: {error}")
                session.tick(now_ns)
                time.sleep(0.002)
    except KeyboardInterrupt:
        if session.state in (SessionState.RECORDING, SessionState.INVALID): sink.discard_episode()
    finally:
        try: sink.finalize()
        finally:
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
