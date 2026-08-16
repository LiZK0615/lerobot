from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("ros_openarm_bimanual")
@dataclass(kw_only=True)
class RosOpenArmBimanualConfig(RobotConfig):
    """Jetson-local ROS bridge used by the remote VLA inference client."""

    id: str | None = "ros_openarm_bimanual"
    camera_config: str
    ros_snapshot_host: str = "127.0.0.1"
    ros_snapshot_port: int = 15001
    policy_command_host: str = "127.0.0.1"
    policy_command_port: int = 15002
    source_max_age_ms: int = 200

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("ros_snapshot_port", "policy_command_port"):
            value = getattr(self, name)
            if not 1 <= value <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
        if self.source_max_age_ms <= 0:
            raise ValueError("source_max_age_ms must be positive")
