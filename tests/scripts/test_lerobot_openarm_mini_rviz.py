import json
import math
import unittest
from unittest.mock import patch

from lerobot.scripts.lerobot_openarm_mini_rviz import build_datagram, starting_sequence


def make_action() -> dict[str, float]:
    action = {f"joint_{index}.pos": float(index) for index in range(1, 8)}
    action["gripper.pos"] = -32.5
    return action


class BuildDatagramTest(unittest.TestCase):
    @patch(
        "lerobot.scripts.lerobot_openarm_mini_rviz.time.monotonic_ns",
        return_value=987654321,
    )
    def test_starting_sequence_uses_monotonic_clock(self, monotonic_ns):
        self.assertEqual(starting_sequence(), 987654321)
        monotonic_ns.assert_called_once_with()

    def test_preserves_mapped_joint_names(self):
        payload = json.loads(build_datagram(make_action(), 7, 1234, "left"))

        self.assertEqual(
            payload,
            {
                "version": 1,
                "sequence": 7,
                "sent_monotonic_ns": 1234,
                "side": "left",
                "positions_deg": {
                    "joint_1": 1.0,
                    "joint_2": 2.0,
                    "joint_3": 3.0,
                    "joint_4": 4.0,
                    "joint_5": 5.0,
                    "joint_6": 6.0,
                    "joint_7": 7.0,
                    "gripper": -32.5,
                },
            },
        )

    def test_rejects_invalid_side(self):
        for side in ("", "both", "LEFT"):
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "side"):
                build_datagram(make_action(), 0, 0, side)

    def test_rejects_missing_action_field(self):
        action = make_action()
        del action["joint_4.pos"]

        with self.assertRaisesRegex(ValueError, "joint_4.pos"):
            build_datagram(action, 0, 0, "left")

    def test_rejects_invalid_sequence_metadata(self):
        for sequence in (-1, True, 1.5):
            with (
                self.subTest(sequence=sequence),
                self.assertRaisesRegex(ValueError, "sequence"),
            ):
                build_datagram(make_action(), sequence, 0, "left")
        for timestamp in (-1, True, 1.5):
            with (
                self.subTest(timestamp=timestamp),
                self.assertRaisesRegex(ValueError, "sent_monotonic_ns"),
            ):
                build_datagram(make_action(), 0, timestamp, "left")

    def test_rejects_non_finite_or_non_numeric_value(self):
        for value in (math.nan, math.inf, -math.inf, True, "1.0"):
            action = make_action()
            action["joint_6.pos"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "joint_6.pos"),
            ):
                build_datagram(action, 0, 0, "left")
