import json
import math
import unittest
from unittest.mock import patch

from lerobot.scripts.lerobot_openarm_mini_ros2_teleop import (
    build_bimanual_datagram,
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
