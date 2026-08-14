import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

import yaml

from lerobot.scripts.lerobot_openarm_mini_compliance_test import (
    DEFAULT_CONFIG_PATH,
    JOINT_MOTORS,
    BimanualComplianceExperiment,
    ComplianceMode,
    ComplianceStateError,
    load_compliance_config,
)


def make_positions(value: float = 0.0) -> dict[str, float]:
    return {
        f"{side}_{motor}.pos": value
        for side in ("left", "right")
        for motor in JOINT_MOTORS
    }


class BimanualComplianceExperimentTest(unittest.TestCase):
    def setUp(self):
        self.config = load_compliance_config(DEFAULT_CONFIG_PATH)
        self.teleop = MagicMock()
        self.teleop.left_arm.bus.read.side_effect = lambda register, motor, normalize=False: (
            32 if register == "P_Coefficient" else 1000
        )
        self.teleop.right_arm.bus.read.side_effect = self.teleop.left_arm.bus.read.side_effect
        self.experiment = BimanualComplianceExperiment(self.teleop, self.config)

    def test_default_config_has_symmetric_settings_for_fourteen_joints(self):
        self.assertEqual(set(self.config.joints), {"left", "right"})
        for motor in JOINT_MOTORS:
            self.assertEqual(self.config.joints["left"][motor], self.config.joints["right"][motor])
            self.assertEqual(self.config.joints["left"][motor].p_coefficient, 8)
            self.assertEqual(self.config.joints["left"][motor].torque_limit, 50)

    def test_enable_configures_and_enables_j1_through_j7_without_gripper(self):
        positions = make_positions(12.5)
        self.experiment.enable_hold(positions)

        self.assertEqual(self.experiment.mode, ComplianceMode.HOLD)
        for arm in (self.teleop.left_arm, self.teleop.right_arm):
            arm.bus.disable_torque.assert_called_once_with(list(JOINT_MOTORS))
            arm.bus.enable_torque.assert_called_once_with(list(JOINT_MOTORS))
            self.assertEqual(arm.bus.write.call_count, 14)
            goals = arm.write_goal_positions.call_args.args[0]
            self.assertEqual(set(goals), {f"{motor}.pos" for motor in JOINT_MOTORS})
            self.assertNotIn("gripper.pos", goals)

    def test_follow_updates_only_joints_outside_their_deadbands(self):
        positions = make_positions(10.0)
        self.experiment.enable_hold(positions)
        self.experiment.follow()
        self.teleop.left_arm.write_goal_positions.reset_mock()
        self.teleop.right_arm.write_goal_positions.reset_mock()
        moved = dict(positions)
        moved["left_joint_2.pos"] = 10.4
        moved["left_joint_4.pos"] = 10.6
        moved["right_joint_7.pos"] = 11.0

        changed = self.experiment.update(moved)

        self.assertEqual(set(changed), {"left_joint_4.pos", "right_joint_7.pos"})
        self.teleop.left_arm.write_goal_positions.assert_called_once_with({"joint_4.pos": 10.6})
        self.teleop.right_arm.write_goal_positions.assert_called_once_with({"joint_7.pos": 11.0})

    def test_hold_reseeds_all_goals_and_stops_following(self):
        positions = make_positions(10.0)
        self.experiment.enable_hold(positions)
        self.experiment.follow()
        moved = make_positions(15.0)
        self.experiment.hold(moved)

        self.assertEqual(self.experiment.mode, ComplianceMode.HOLD)
        self.assertEqual(self.experiment.goals_deg, moved)
        self.assertEqual(self.experiment.update(make_positions(20.0)), ())

    def test_disable_restores_original_registers_on_both_arms(self):
        self.experiment.enable_hold(make_positions())
        for arm in (self.teleop.left_arm, self.teleop.right_arm):
            arm.bus.disable_torque.reset_mock()
            arm.bus.write.reset_mock()

        self.experiment.disable()

        self.assertEqual(self.experiment.mode, ComplianceMode.DISABLED)
        for arm in (self.teleop.left_arm, self.teleop.right_arm):
            arm.bus.disable_torque.assert_called_once_with(list(JOINT_MOTORS))
            expected = []
            for motor in JOINT_MOTORS:
                expected.extend(
                    [call("P_Coefficient", motor, 32), call("Torque_Limit", motor, 1000)]
                )
            self.assertEqual(arm.bus.write.call_args_list, expected)

    def test_partial_enable_failure_restores_both_arms(self):
        self.teleop.right_arm.bus.enable_torque.side_effect = RuntimeError("right enable failed")

        with self.assertRaisesRegex(RuntimeError, "right enable failed"):
            self.experiment.enable_hold(make_positions())

        self.assertEqual(self.experiment.mode, ComplianceMode.DISABLED)
        self.assertFalse(self.experiment._originals)
        for arm in (self.teleop.left_arm, self.teleop.right_arm):
            self.assertEqual(arm.bus.disable_torque.call_count, 2)

    def test_follow_before_enable_is_a_state_error(self):
        with self.assertRaises(ComplianceStateError):
            self.experiment.follow()


class ComplianceConfigTest(unittest.TestCase):
    def test_missing_joint_is_rejected(self):
        payload = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
        del payload["arms"]["right"]["joint_7"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(ValueError, "joint_1 through joint_7"):
                load_compliance_config(path)

    def test_invalid_register_and_deadband_values_are_rejected(self):
        base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
        for field, value in (
            ("p_coefficient", 255),
            ("p_coefficient", 8.0),
            ("torque_limit", 1001),
            ("torque_limit", 50.0),
            ("position_deadband_deg", 0.0),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                payload = copy.deepcopy(base)
                payload["arms"]["left"]["joint_1"][field] = value
                path = Path(directory) / "invalid.yaml"
                path.write_text(yaml.safe_dump(payload))
                with self.assertRaisesRegex(ValueError, field):
                    load_compliance_config(path)


if __name__ == "__main__":
    unittest.main()
