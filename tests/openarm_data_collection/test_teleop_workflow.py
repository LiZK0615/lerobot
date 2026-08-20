import math

from lerobot.openarm_data_collection.teleop_workflow import (
    CollectionTeleopWorkflow,
    JOINT_ACTION_NAMES,
    TorqueRequest,
    WorkflowCommand,
    WorkflowState,
    filter_recording_action,
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
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE

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
    assert config.auto_return_idle_duration_sec == 1.0
    assert config.auto_return_idle_joint_delta_deg == 2.0
    assert config.auto_return_j1_tolerance_rad == 0.5


def test_opening_left_gripper_releases_only_left_leader_arm():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    start_output = workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    assert start_output.left_torque is TorqueRequest.ENABLE
    assert start_output.right_torque is TorqueRequest.ENABLE
    assert start_output.goal == ready

    opened = dict(ready)
    opened["left_gripper.pos"] = -10.0
    output = workflow.tick(opened, now + 0.1)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.left_torque is TorqueRequest.DISABLE
    assert output.right_torque is TorqueRequest.UNCHANGED
    assert workflow.operator_engaged


def test_each_gripper_releases_its_side_once_and_closing_does_not_reenable_it():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    left_open = dict(ready)
    left_open["left_gripper.pos"] = -10.0
    output = workflow.tick(left_open, now + 0.1)
    assert output.left_torque is TorqueRequest.DISABLE
    assert output.right_torque is TorqueRequest.UNCHANGED

    both_open = dict(left_open)
    both_open["right_gripper.pos"] = -10.0
    output = workflow.tick(both_open, now + 0.2)
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.DISABLE

    right_open = dict(ready)
    right_open["right_gripper.pos"] = -10.0
    output = workflow.tick(right_open, now + 0.3)
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.UNCHANGED
    assert output.goal is None
    assert workflow.side_engaged == {"left": True, "right": True}

    output = workflow.tick(ready, now + 0.4)
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.UNCHANGED
    assert output.goal is None


def test_closed_grippers_do_not_trigger_until_operator_has_opened_one():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    workflow.tick(ready, now + 0.1)
    output = workflow.tick(ready, now + 1.0)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.UNCHANGED
    assert output.goal == ready
    assert not workflow.operator_engaged


def test_idle_timer_starts_only_after_both_grippers_close():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    opened = dict(ready)
    opened["right_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)
    workflow.tick(opened, now + 10.0)

    closed = dict(ready)
    closed_output = workflow.tick(closed, now + 10.1)
    assert closed_output.right_torque is TorqueRequest.UNCHANGED
    before_timeout = workflow.tick(closed, now + 11.09)
    assert before_timeout.left_torque is TorqueRequest.UNCHANGED
    assert before_timeout.right_torque is TorqueRequest.UNCHANGED
    assert workflow.state is WorkflowState.RECORDING_MANUAL
    output = workflow.tick(closed, now + 11.11)

    assert workflow.state is WorkflowState.AUTO_RETURNING
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE
    assert output.goal == closed


def test_closed_and_quiet_far_j1_blocks_return_until_j1_is_near_ready():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)

    far = dict(ready)
    far["left_joint_1.pos"] += math.degrees(0.51)
    closed_output = workflow.tick(far, now + 0.2)
    assert closed_output.left_torque is TorqueRequest.UNCHANGED
    assert closed_output.right_torque is TorqueRequest.UNCHANGED
    output = workflow.tick(far, now + 1.3)

    assert workflow.state is WorkflowState.RECORDING_MANUAL
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.UNCHANGED

    workflow.tick(ready, now + 1.4)
    assert workflow.tick(ready, now + 2.39).left_torque is TorqueRequest.UNCHANGED
    output = workflow.tick(ready, now + 2.41)

    assert workflow.state is WorkflowState.AUTO_RETURNING
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE


def test_inactive_arm_vibration_does_not_block_active_arm_idle_detection():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)

    right_open = dict(ready)
    right_open["right_gripper.pos"] = -20.0
    workflow.tick(right_open, now + 0.1)
    workflow.tick(ready, now + 0.2)

    left_vibration = dict(ready)
    left_vibration["left_joint_2.pos"] += 15.0
    workflow.tick(left_vibration, now + 0.6)
    output = workflow.tick(ready, now + 1.21)

    assert workflow.side_engaged == {"left": False, "right": True}
    assert workflow.state is WorkflowState.AUTO_RETURNING
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE


def test_recording_action_freezes_inactive_joints_but_keeps_both_grippers_live():
    previous = action()
    current = action()
    for joint in range(1, 8):
        current[f"left_joint_{joint}.pos"] = 10.0 + joint
        current[f"right_joint_{joint}.pos"] = 20.0 + joint
    current["left_gripper.pos"] = -12.0
    current["right_gripper.pos"] = -34.0

    filtered = filter_recording_action(
        current,
        previous,
        {"left": False, "right": True},
    )

    for joint in range(1, 8):
        assert filtered[f"left_joint_{joint}.pos"] == 0.0
        assert filtered[f"right_joint_{joint}.pos"] == 20.0 + joint
    assert filtered["left_gripper.pos"] == -12.0
    assert filtered["right_gripper.pos"] == -34.0


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
    output = workflow.tick(moved, now + 1.39)
    assert output.left_torque is TorqueRequest.UNCHANGED
    assert output.right_torque is TorqueRequest.UNCHANGED
    output = workflow.tick(moved, now + 1.41)

    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE
    assert workflow.state is WorkflowState.AUTO_RETURNING


def test_automatic_return_holds_ready_until_finish_decision():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    workflow.handle_command(WorkflowCommand.START_RECORDING, ready, now)
    opened = dict(ready)
    opened["left_gripper.pos"] = -20.0
    workflow.tick(opened, now + 0.1)
    workflow.tick(ready, now + 0.2)
    workflow.tick(ready, now + 1.21)

    finish = now + 1.21 + workflow.motion.segment_duration_sec
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
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE
    assert workflow.motion.waypoint_name == "table_clearance"


def test_shutdown_targets_clearance_and_then_disables_leader_torque():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    ready, now = drive_to_ready(workflow)
    output = workflow.handle_command(WorkflowCommand.SHUTDOWN, ready, now + 0.1)
    assert output.left_torque is TorqueRequest.ENABLE
    assert output.right_torque is TorqueRequest.ENABLE

    clearance = ros_positions_to_mapped_action(workflow.config.waypoints["table_clearance"])
    output = workflow.tick(clearance, now + 0.1 + workflow.motion.segment_duration_sec)

    assert workflow.state is WorkflowState.SHUTDOWN_COMPLETE
    assert output.left_torque is TorqueRequest.DISABLE
    assert output.right_torque is TorqueRequest.DISABLE


def test_gripper_difference_does_not_prevent_preset_completion():
    workflow = CollectionTeleopWorkflow(load_preset_config(DEFAULT_PRESET_CONFIG_PATH))
    current = action()
    workflow.initialize(current, 0.0)
    clearance = ros_positions_to_mapped_action(workflow.config.waypoints["table_clearance"])
    clearance["left_gripper.pos"] = -65.0
    clearance["right_gripper.pos"] = -65.0

    workflow.tick(clearance, workflow.motion.segment_duration_sec)

    assert workflow.motion.phase == "pausing"
