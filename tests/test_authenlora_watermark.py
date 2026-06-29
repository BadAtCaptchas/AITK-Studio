import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image
from safetensors import safe_open

from toolkit.config_modules import WatermarkConfig
from toolkit.lora_special import LoRAModule
from toolkit.lycoris_special import LoConSpecialModule
from toolkit.models.lokr import LokrModule
from toolkit.watermarking.authenlora import (
    AuthenLoRAController,
    AuthenLoRACodec,
    MapperNet,
    bit_accuracy,
)
from toolkit.watermarking.codecs import resolve_codec_path
from scripts.check_authenlora_watermark import summarize_detection


class FakeNetwork:
    def __init__(self):
        self.network_type = "lora"
        self.is_active = True
        self.is_merged_in = False
        self.is_lorm = False
        self._multiplier = 1.0
        self.torch_multiplier = torch.ones(1)
        self.rank_scale = None

    def match_authenlora_rank_scale(self, rank_scale, rank, device, dtype):
        rank_scale = rank_scale.to(device=device, dtype=dtype)
        if rank_scale.dim() == 1:
            rank_scale = rank_scale.unsqueeze(0)
        if rank_scale.shape[-1] == rank:
            return rank_scale
        if rank_scale.shape[-1] > rank:
            return rank_scale[..., :rank]
        repeats = (rank + rank_scale.shape[-1] - 1) // rank_scale.shape[-1]
        return rank_scale.repeat(1, repeats)[..., :rank]

    def get_authenlora_rank_scale(self, rank, device, dtype):
        if self.rank_scale is None:
            return None
        return self.match_authenlora_rank_scale(self.rank_scale, rank, device, dtype)


class AuthenLoRATest(unittest.TestCase):
    def test_mapper_shape_and_determinism(self):
        torch.manual_seed(123)
        mapper_a = MapperNet(input_size=4, output_size=6)
        torch.manual_seed(123)
        mapper_b = MapperNet(input_size=4, output_size=6)
        bits = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.float32)

        output_a = mapper_a(bits)
        output_b = mapper_b(bits)

        self.assertEqual(tuple(output_a.shape), (2, 6))
        self.assertTrue(torch.allclose(output_a, output_b))

    def test_zero_message_bit_accuracy(self):
        logits = torch.tensor(
            [
                [[3.0, -1.0], [2.0, -0.5], [4.0, -2.0], [1.5, -3.0]],
                [[1.0, -1.0], [5.0, -0.5], [2.0, -2.0], [7.0, -3.0]],
            ]
        )
        bits = torch.zeros((1, 4), dtype=torch.long)

        self.assertEqual(bit_accuracy(logits, bits), 1.0)

    def test_zero_message_detection_summary(self):
        self.assertEqual(
            summarize_detection("0000"),
            {
                "zero_message": True,
                "watermark_detected": False,
                "watermark_status": "not_detected",
            },
        )
        self.assertEqual(summarize_detection("1000")["watermark_status"], "candidate")
        self.assertEqual(summarize_detection("1000", match=True, has_expected_secret=True)["watermark_status"], "verified")
        self.assertEqual(summarize_detection("1000", match=False, has_expected_secret=True)["watermark_status"], "mismatch")

    def test_lora_and_locon_rank_modulation(self):
        for module_cls in (LoRAModule, LoConSpecialModule):
            network = FakeNetwork()
            original = torch.nn.Linear(2, 2, bias=False)
            module = module_cls("test", original, lora_dim=2, alpha=2, network=network)
            with torch.no_grad():
                module.lora_down.weight.copy_(torch.eye(2))
                module.lora_up.weight.copy_(torch.eye(2))
                if hasattr(module, "scalar") and isinstance(module.scalar, torch.nn.Parameter):
                    module.scalar.data.fill_(1.0)

            x = torch.tensor([[1.0, 2.0]])
            unscaled = module._call_forward(x)
            network.rank_scale = torch.tensor([[2.0, 3.0]])
            scaled = module._call_forward(x)

            self.assertTrue(torch.allclose(unscaled, torch.tensor([[1.0, 2.0]]), atol=1e-6))
            self.assertTrue(torch.allclose(scaled, torch.tensor([[2.0, 6.0]]), atol=1e-6))

    def test_lokr_rank_modulation_changes_weight(self):
        network = FakeNetwork()
        original = torch.nn.Linear(4, 4, bias=False)
        module = LokrModule("lokr", original, lora_dim=2, alpha=2, network=network)
        with torch.no_grad():
            if hasattr(module, "lokr_w2_b"):
                module.lokr_w2_b.fill_(0.25)
            if hasattr(module, "lokr_w2"):
                module.lokr_w2.fill_(0.25)

        unscaled = module.get_weight(original.weight.shape)
        network.rank_scale = torch.tensor([[2.0, 3.0]])
        scaled = module.get_weight(original.weight.shape)

        self.assertEqual(tuple(unscaled.shape), tuple(scaled.shape))
        self.assertFalse(torch.allclose(unscaled, scaled))

    def test_mapper_and_private_sidecar_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codec_path = os.path.join(temp_dir, "codec.pt")
            codec = AuthenLoRACodec(msg_bits=4)
            torch.save(
                {
                    "sec_encoder": codec.encoder.state_dict(),
                    "sec_decoder": codec.decoder.state_dict(),
                },
                codec_path,
            )
            config = WatermarkConfig(
                enabled=True,
                codec_path=codec_path,
                msg_bits=4,
                mapper_rank=6,
                secret="1010",
                bake_on_save=True,
            )
            controller = AuthenLoRAController(
                config=config,
                device=torch.device("cpu"),
                dtype=torch.float32,
                save_root=temp_dir,
                run_name="run",
            )
            checkpoint_path = os.path.join(temp_dir, "run.safetensors")
            mapper_path = controller.save_mapper(checkpoint_path)
            sidecar_path = controller.save_private_sidecar(checkpoint_path)

            with safe_open(mapper_path, framework="pt") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["aitk_watermark_method"], "authenlora")
            self.assertEqual(metadata["aitk_watermark_msg_bits"], "4")

            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            self.assertEqual(sidecar["secret"], "1010")
            self.assertEqual(sidecar["verification_threshold"], 0.5)

    def test_builtin_codec_reference_resolves_to_local_file(self):
        path = resolve_codec_path("builtin:authenlora_48bits")
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("authenlora_codec_48bits.pth"))

    def test_check_image_cli_outputs_decoded_bits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codec_path = os.path.join(temp_dir, "codec.pt")
            image_path = os.path.join(temp_dir, "image.png")
            codec = AuthenLoRACodec(msg_bits=4)
            torch.save(
                {
                    "sec_encoder": codec.encoder.state_dict(),
                    "sec_decoder": codec.decoder.state_dict(),
                },
                codec_path,
            )
            Image.new("RGB", (32, 32), color=(32, 64, 96)).save(image_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    os.path.join("scripts", "check_authenlora_watermark.py"),
                    "--image",
                    image_path,
                    "--codec",
                    codec_path,
                    "--msg-bits",
                    "4",
                    "--expected-secret",
                    "0000",
                ],
                check=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["msg_bits"], 4)
            self.assertEqual(len(result["decoded_bits"]), 4)
            self.assertIn("confidence", result)
            self.assertIn("bit_accuracy", result)
            self.assertIn("zero_message", result)
            self.assertIn("watermark_detected", result)
            self.assertIn("watermark_status", result)


if __name__ == "__main__":
    unittest.main()
