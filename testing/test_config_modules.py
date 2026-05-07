import unittest
import types
from unittest import mock

album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = mock.Mock()

prompt_utils_module = types.ModuleType("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = type("PromptEmbeds", (), {})

with mock.patch.dict(
    "sys.modules",
    {
        "toolkit.audio.album_artwork": album_artwork_module,
        "toolkit.prompt_utils": prompt_utils_module,
    },
):
    from toolkit.config_modules import NetworkConfig



class NetworkConfigTest(unittest.TestCase):
    def test_network_weights_alias_sets_pretrained_lora_path(self):
        config = NetworkConfig(network_weights="C:/models/example.safetensors")

        self.assertEqual(config.pretrained_lora_path, "C:/models/example.safetensors")

    def test_pretrained_lora_path_takes_precedence_over_network_weights(self):
        config = NetworkConfig(
            pretrained_lora_path="C:/models/canonical.safetensors",
            network_weights="C:/models/legacy.safetensors",
        )

        self.assertEqual(config.pretrained_lora_path, "C:/models/canonical.safetensors")


if __name__ == "__main__":
    unittest.main()
