"""Regression checks for checkpoint camera mappings in async inference."""
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
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
