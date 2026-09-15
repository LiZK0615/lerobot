"""Regression checks for checkpoint camera mappings in async inference."""
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lerobot.async_inference.configs import PolicyServerConfig
import numpy as np
import torch

from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer


class CameraMappingTest(unittest.TestCase):
    def check_mapping(self, mapping):
        server = PolicyServer(PolicyServerConfig(host="localhost", port=9999))
        server.shutdown_event.clear()
        specs = RemotePolicyConfig("smolvla", "checkpoint", {}, 10, "cpu", mapping)
        with (
            patch("lerobot.async_inference.policy_server.get_policy_class") as policy_class,
            patch("lerobot.async_inference.policy_server.make_pre_post_processors") as processors,
        ):
            policy_class.return_value.from_pretrained.return_value = MagicMock()
            processors.return_value = (MagicMock(), MagicMock())
            server.SendPolicyInstructions(SimpleNamespace(data=pickle.dumps(specs)), MagicMock())
        return processors.call_args.kwargs["preprocessor_overrides"]

    def test_empty_client_mapping_preserves_checkpoint(self):
        overrides = self.check_mapping({})
        self.assertNotIn("rename_observations_processor", overrides)
        self.assertEqual(overrides["device_processor"], {"device": "cpu"})

    def test_explicit_client_mapping_takes_precedence(self):
        mapping = {"observation.images.head": "observation.images.camera1"}
        overrides = self.check_mapping(mapping)
        self.assertEqual(overrides["rename_observations_processor"]["rename_map"], mapping)

    def test_raw_observation_reaches_policy_with_effective_camera_mapping(self):
        for raw_camera in ("head", "custom_head", "camera1"):
            with self.subTest(raw_camera=raw_camera):
                server = PolicyServer(PolicyServerConfig(host="localhost", port=9999))
                source = f"observation.images.{raw_camera}"
                target = "observation.images.camera1"
                features = {target: SimpleNamespace(shape=(3, 8, 8))}
                server.policy = MagicMock()
                server.policy.config.image_features = features
                server.policy.predict_action_chunk.return_value = torch.zeros(1, 2, 1)
                rename = RenameObservationsProcessorStep(rename_map={source: target})
                server.preprocessor = MagicMock(side_effect=rename.observation)
                server.preprocessor.steps = [rename]
                server.postprocessor = lambda action: action
                server.actions_per_chunk = 2
                server.lerobot_features = {
                    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
                    source: {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
                }
                raw = {"joint.pos": 0.5, raw_camera: np.full((16, 16, 3), 127, dtype=np.uint8),
                       "task": "Pick up the block and place it on the plate."}
                actions = server._predict_action_chunk(TimedObservation(1.0, 0, raw))
                observation = server.policy.predict_action_chunk.call_args.args[0]
                self.assertEqual(tuple(observation[target].shape), (1, 3, 8, 8))
                if source != target:
                    self.assertNotIn(source, observation)
                self.assertEqual(set(features), {target})
                self.assertEqual(len(actions), 2)
                self.assertTrue(all(torch.isfinite(action.get_action()).all() for action in actions))
