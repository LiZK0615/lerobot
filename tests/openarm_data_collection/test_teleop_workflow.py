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
    assert config.target_tolerance_rad == 0.10
    assert config.gripper_closed_threshold_deg == -3.0
    assert config.auto_return_idle_duration_sec == 0.5
    assert config.auto_return_idle_joint_delta_deg == 2.0


def test_opening_either_gripper_releases_both_leader_arms():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    start_output = workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    assert start_output.torque is TorqueRequest.ENABLE
    assert start_output.goal == ready

    opened = dict(ready)
    opened["left_gripper.pos"] = -10.0
    output = workflow.tick(opened, now + 0.1)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.torque is TorqueRequest.DISABLE
    assert workflow.operator_engaged


def test_closed_grippers_do_not_trigger_until_operator_has_opened_one():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    workflow.tick(ready, now + 0.1)
    output = workflow.tick(ready, now + 1.0)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.torque is TorqueRequest.UNCHANGED
    assert output.goal == ready
    assert not workflow.operator_engaged


def test_closed_and_quiet_triggers_automatic_return_after_half_second():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    opened = dict(ready)
    opened["right_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)

    closed = dict(ready)
    assert workflow.tick(closed, now + 0.2).torque is TorqueRequest.UNCHANGED
    assert workflow.tick(closed, now + 0.69).torque is TorqueRequest.UNCHANGED
    output = workflow.tick(closed, now + 0.71)

    assert workflow.state is WorkflowState.AUTO_RETURNING
    assert output.torque is TorqueRequest.ENABLE
    assert output.goal == closed


def test_closed_and_quiet_far_from_ready_still_triggers_return():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)

    far = dict(ready)
    far["left_joint_1.pos"] += 18.0
    assert workflow.tick(far, now + 0.2).torque is TorqueRequest.UNCHANGED
    output = workflow.tick(far, now + 1.0)

    assert workflow.state is WorkflowState.AUTO_RETURNING
    assert output.torque is TorqueRequest.ENABLE
    assert output.goal == far


def test_joint_motion_over_two_degrees_restarts_auto_return_quiet_detection():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)
    workflow.tick(ready, now + 0.2)

    moved = dict(ready)
    moved["left_joint_1.pos"] += 2.1
    workflow.tick(moved, now + 0.4)
    assert workflow.tick(moved, now + 0.61).torque is TorqueRequest.UNCHANGED
    output = workflow.tick(moved, now + 0.91)

    assert output.torque is TorqueRequest.ENABLE
    assert workflow.state is WorkflowState.AUTO_RETURNING


def test_automatic_return_holds_ready_until_finish_decision():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)
    workflow.tick(ready, now + 0.2)
    workflow.tick(ready, now + 0.71)

    finish = now + 0.71 + workflow.motion.segment_duration_sec
    workflow.tick(ready, finish)
    assert workflow.state is WorkflowState.AWAITING_DECISION

    output = workflow.handle_command(WorkflowCommand.FINISH_DECISION, ready, finish + 0.1)
    assert workflow.state is WorkflowState.READY
    assert output.goal == ready


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
