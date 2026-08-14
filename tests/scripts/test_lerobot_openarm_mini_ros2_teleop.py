import json
import math
import unittest
from unittest.mock import patch

from lerobot.scripts.lerobot_openarm_mini_ros2_teleop import (
    DEFAULT_PRESET_CONFIG_PATH,
    MotionPhase,
    PresetMotion,
    build_bimanual_datagram,
    load_preset_config,
    ros_positions_to_mapped_action,
    starting_sequence,
)


def make_bimanual_action() -> dict[str, float]:
    action: dict[str, float] = {}
    for index in range(1, 8):
        action[f"left_joint_{index}.pos"] = float(index)
        action[f"right_joint_{index}.pos"] = float(100 + index)
    action["left_gripper.pos"] = -32.5
    action["right_gripper.pos"] = -65.0
    return action


class BuildBimanualDatagramTest(unittest.TestCase):
    @patch(
        "lerobot.scripts.lerobot_openarm_mini_ros2_teleop.time.monotonic_ns",
        return_value=987654321,
    )
    def test_starting_sequence_uses_monotonic_clock(self, monotonic_ns):
        self.assertEqual(starting_sequence(), 987654321)
        monotonic_ns.assert_called_once_with()

    def test_serializes_both_already_mapped_actions_in_one_packet(self):
        payload = json.loads(build_bimanual_datagram(make_bimanual_action(), 7, 1234))

        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["sequence"], 7)
        self.assertEqual(payload["sent_monotonic_ns"], 1234)
        self.assertEqual(payload["left"]["joint_6"], 6.0)
        self.assertEqual(payload["left"]["joint_7"], 7.0)
        self.assertEqual(payload["right"]["joint_6"], 106.0)
        self.assertEqual(payload["right"]["joint_7"], 107.0)
        self.assertEqual(payload["left"]["gripper"], -32.5)
        self.assertEqual(payload["right"]["gripper"], -65.0)
        self.assertEqual(
            set(payload),
            {"version", "sequence", "sent_monotonic_ns", "left", "right"},
        )

    def test_rejects_a_missing_field_from_either_side(self):
        for key in ("left_joint_4.pos", "right_gripper.pos"):
            with self.subTest(key=key):
                action = make_bimanual_action()
                del action[key]
                with self.assertRaisesRegex(ValueError, key.replace(".", r"\.")):
                    build_bimanual_datagram(action, 0, 0)


class PresetMotionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_preset_config(DEFAULT_PRESET_CONFIG_PATH)

    def test_default_waypoints_are_clean_symmetric_ros_positions(self):
        clearance = self.config.waypoints["table_clearance"]
        ready = self.config.waypoints["table_ready"]

        self.assertEqual(
            clearance,
            (
                0.10, 0.0, 0.0, 0.0, 0.0, 0.0, -0.80, 0.0,
                -0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.0,
            ),
        )
        self.assertEqual(
            ready,
            (
                0.72, 0.0, 0.0, 1.15, 0.0, 0.0, -1.10, 0.0,
                -0.72, 0.0, 0.0, 1.15, 0.0, 0.0, 1.10, 0.0,
            ),
        )
        self.assertEqual(self.config.prepare_sequence, ("table_clearance", "table_ready"))

    def test_ros_positions_convert_to_existing_mapped_degree_protocol(self):
        action = ros_positions_to_mapped_action(self.config.waypoints["table_ready"])

        self.assertAlmostEqual(action["left_joint_1.pos"], math.degrees(0.72))
        self.assertAlmostEqual(action["left_joint_7.pos"], math.degrees(-1.10))
        self.assertAlmostEqual(action["right_joint_4.pos"], math.degrees(1.15))
        self.assertEqual(action["left_gripper.pos"], 0.0)
        self.assertEqual(action["right_gripper.pos"], 0.0)

    def test_prepare_runs_clearance_pause_ready_with_quintic_endpoints(self):
        motion = PresetMotion(self.config)
        current = ros_positions_to_mapped_action((0.0,) * 16)
        motion.start_prepare(current, now=0.0)

        self.assertEqual(motion.phase, MotionPhase.MOVING)
        self.assertEqual(motion.waypoint_name, "table_clearance")
        self.assertEqual(motion.step(current, 0.0), current)

        clearance = ros_positions_to_mapped_action(self.config.waypoints["table_clearance"])
        duration = motion.segment_duration_sec
        halfway = motion.step(current, duration / 2.0)
        self.assertAlmostEqual(halfway["left_joint_1.pos"], clearance["left_joint_1.pos"] / 2.0)
        self.assertAlmostEqual(halfway["left_joint_7.pos"], clearance["left_joint_7.pos"] / 2.0)

        motion.step(clearance, duration)
        self.assertEqual(motion.phase, MotionPhase.PAUSING)
        motion.step(clearance, duration + self.config.waypoint_pause_sec)
        self.assertEqual(motion.phase, MotionPhase.MOVING)
        self.assertEqual(motion.waypoint_name, "table_ready")

        ready = ros_positions_to_mapped_action(self.config.waypoints["table_ready"])
        finish_time = duration + self.config.waypoint_pause_sec + motion.segment_duration_sec
        motion.step(ready, finish_time)
        self.assertEqual(motion.phase, MotionPhase.HOLDING)
        self.assertEqual(motion.step(ready, finish_time + 1.0), ready)

    def test_second_waypoint_requires_clearance(self):
        motion = PresetMotion(self.config)
        zero = ros_positions_to_mapped_action((0.0,) * 16)
        with self.assertRaisesRegex(ValueError, "table_clearance"):
            motion.start_ready_only(zero, now=0.0)

        clearance = ros_positions_to_mapped_action(self.config.waypoints["table_clearance"])
        motion.start_ready_only(clearance, now=0.0)
        self.assertEqual(motion.waypoint_name, "table_ready")

    def test_abort_holds_measured_position_and_release_returns_idle(self):
        motion = PresetMotion(self.config)
        current = ros_positions_to_mapped_action((0.0,) * 16)
        motion.start_prepare(current, now=0.0)
        measured = dict(current)
        measured["left_joint_1.pos"] = 3.0

        motion.abort(measured)
        self.assertEqual(motion.phase, MotionPhase.HOLDING)
        self.assertEqual(motion.step(measured, 1.0), measured)
        motion.release()
        self.assertEqual(motion.phase, MotionPhase.IDLE)
        self.assertIsNone(motion.step(measured, 2.0))

    def test_rejects_invalid_sequence_metadata(self):
        for sequence in (-1, True, 1.5):
            with (
                self.subTest(sequence=sequence),
                self.assertRaisesRegex(ValueError, "sequence"),
            ):
                build_bimanual_datagram(make_bimanual_action(), sequence, 0)
        for timestamp in (-1, True, 1.5):
            with (
                self.subTest(timestamp=timestamp),
                self.assertRaisesRegex(ValueError, "sent_monotonic_ns"),
            ):
                build_bimanual_datagram(make_bimanual_action(), 0, timestamp)

    def test_rejects_non_finite_or_non_numeric_positions(self):
        for value in (math.nan, math.inf, -math.inf, True, "1.0"):
            with self.subTest(value=value):
                action = make_bimanual_action()
                action["right_joint_3.pos"] = value
                with self.assertRaisesRegex(ValueError, r"right_joint_3\.pos"):
                    build_bimanual_datagram(action, 0, 0)


if __name__ == "__main__":
    unittest.main()
