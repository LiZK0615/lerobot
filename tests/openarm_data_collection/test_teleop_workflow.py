from lerobot.openarm_data_collection.teleop_workflow import (
    CollectionTeleopWorkflow,
    JOINT_ACTION_NAMES,
    TorqueRequest,
    WorkflowCommand,
    WorkflowState,
)
from lerobot.scripts.lerobot_openarm_mini_ros2_teleop import (
    DEFAULT_PRESET_CONFIG_PATH,
    load_preset_config,
    ros_positions_to_mapped_action,
)


def action(value=0.0):
    result = {name: float(value) for name in JOINT_ACTION_NAMES}
    result["left_gripper.pos"] = 0.0
    result["right_gripper.pos"] = 0.0
    return result


def drive_to_ready(workflow, now=0.0):
    current = action()
    output = workflow.initialize(current, now)
    assert output.torque is TorqueRequest.ENABLE

    clearance = ros_positions_to_mapped_action(workflow.config.waypoints["table_clearance"])
    now += workflow.motion.segment_duration_sec
    workflow.tick(clearance, now)
    now += workflow.config.waypoint_pause_sec
    workflow.tick(clearance, now)

    ready = ros_positions_to_mapped_action(workflow.config.waypoints["table_ready"])
    now += workflow.motion.segment_duration_sec
    workflow.tick(ready, now)
    assert workflow.state is WorkflowState.READY
    return ready, now


def test_default_clutch_thresholds_are_loaded_from_preset_yaml():
    config = load_preset_config(DEFAULT_PRESET_CONFIG_PATH)
    assert config.leader_target_tolerance_rad == 0.10
    assert config.follower_target_tolerance_rad == 0.05
    assert config.gripper_closed_threshold_deg == -3.0
    assert config.operator_idle_duration_sec == 1.0
    assert config.operator_idle_joint_delta_deg == 2.0


def test_opening_either_gripper_releases_both_leader_arms():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    opened = dict(ready)
    opened["left_gripper.pos"] = -10.0
    output = workflow.tick(opened, now + 0.1)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.torque is TorqueRequest.DISABLE


def test_both_closed_and_quiet_for_one_second_locks_at_measured_pose():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    opened = dict(ready)
    opened["right_gripper.pos"] = -20.0
    workflow.handle_command(WorkflowCommand.START_RECORDING, opened, now)
    assert workflow.state is WorkflowState.RECORDING_MANUAL

    closed = dict(ready)
    assert workflow.tick(closed, now + 0.1).torque is TorqueRequest.UNCHANGED
    assert workflow.tick(closed, now + 0.99).torque is TorqueRequest.UNCHANGED
    output = workflow.tick(closed, now + 1.1)

    assert workflow.state is WorkflowState.RECORDING_LOCKED
    assert output.torque is TorqueRequest.ENABLE
    assert output.goal == closed


def test_joint_motion_over_two_degrees_restarts_quiet_detection():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.handle_command(WorkflowCommand.START_RECORDING, opened, now)
    closed = dict(ready)
    workflow.tick(closed, now + 0.1)

    moved = dict(closed)
    moved["left_joint_1.pos"] += 2.1
    workflow.tick(moved, now + 1.0)
    assert workflow.tick(moved, now + 1.6).torque is TorqueRequest.UNCHANGED
    output = workflow.tick(moved, now + 2.1)

    assert output.torque is TorqueRequest.ENABLE
    assert workflow.state is WorkflowState.RECORDING_LOCKED


def test_save_reset_moves_directly_to_ready():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    output = workflow.handle_command(WorkflowCommand.RESET_SAVE, ready, now + 0.1)

    assert workflow.state is WorkflowState.RESETTING_SAVE
    assert output.torque is TorqueRequest.ENABLE
    assert workflow.motion.waypoint_name == "table_ready"


def test_discard_reset_still_returns_through_clearance_to_ready():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    output = workflow.handle_command(WorkflowCommand.RESET_DISCARD, ready, now + 0.1)

    assert workflow.state is WorkflowState.RESETTING_DISCARD
    assert output.torque is TorqueRequest.ENABLE
    assert workflow.motion.waypoint_name == "table_clearance"


def test_shutdown_targets_clearance_and_then_disables_leader_torque():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    output = workflow.handle_command(WorkflowCommand.SHUTDOWN, ready, now + 0.1)
    assert output.torque is TorqueRequest.ENABLE

    clearance = ros_positions_to_mapped_action(workflow.config.waypoints["table_clearance"])
    output = workflow.tick(clearance, now + 0.1 + workflow.motion.segment_duration_sec)

    assert workflow.state is WorkflowState.SHUTDOWN_COMPLETE
    assert output.torque is TorqueRequest.DISABLE


def test_gripper_difference_does_not_prevent_preset_completion():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    current = action()
    workflow.initialize(current, 0.0)
    clearance = ros_positions_to_mapped_action(workflow.config.waypoints["table_clearance"])
    clearance["left_gripper.pos"] = -65.0
    clearance["right_gripper.pos"] = -65.0

    workflow.tick(clearance, workflow.motion.segment_duration_sec)

    assert workflow.motion.phase == "pausing"
