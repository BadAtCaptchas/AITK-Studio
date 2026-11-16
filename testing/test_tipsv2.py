import unittest
import tempfile
from pathlib import Path

from safetensors.torch import save_file
from unittest import mock

import torch
from torch import nn

from toolkit.models import tipsv2


class TinyVisionEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.dummy = nn.Parameter(torch.empty(0))
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **_kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


def build_tiny_vision(**kwargs):
    return TinyVisionEncoder(**kwargs)


class LoaderOnlyTIPSv2Model(tipsv2.TIPSv2DPTModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.loaded_state = None
        self.to_kwargs = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = dict(state_dict)
        return [], []

    def to(self, *args, **kwargs):
        self.to_kwargs = kwargs
        return self


class TIPSv2LocalModelTest(unittest.TestCase):
    def test_instantiates_expected_dense_prediction_modules_without_download(self):
        config = {
            "vision_fn": "tiny_test",
            "embed_dim": 8,
            "channels": 4,
            "post_process_channels": (2, 4, 8, 8),
            "block_indices": (0, 1, 2, 3),
            "readout_type": "project",
            "num_depth_bins": 4,
            "min_depth": 1e-3,
            "max_depth": 10.0,
            "num_seg_classes": 3,
            "patch_size": 14,
            "img_size": 14,
            "init_values": 1.0,
            "ffn_layer": "mlp",
        }

        with mock.patch.dict(
            tipsv2._VISION_BUILDERS, {"tiny_test": build_tiny_vision}
        ):
            model = tipsv2.TIPSv2DPTModel(config)

        self.assertIsInstance(model.vision_encoder, TinyVisionEncoder)
        self.assertIsInstance(model.depth_head, tipsv2.DPTDepthHead)
        self.assertIsInstance(model.normals_head, tipsv2.DPTNormalsHead)
        self.assertIsInstance(model.segmentation_head, tipsv2.DPTSegmentationHead)
        self.assertEqual(model.config["backbone_repo"], tipsv2.TIPS2_BACKBONE_REPO)
        self.assertEqual(model.config["num_seg_classes"], 3)

    def test_from_pretrained_downloads_only_pinned_safetensors_weights(self):
        download_calls = []
        paths = {
            tipsv2.TIPS2_DPT_REPO: "dpt.safetensors",
            tipsv2.TIPS2_BACKBONE_REPO: "backbone.safetensors",
        }

        def fake_hf_hub_download(repo_id, filename, revision=None, cache_dir=None):
            download_calls.append(
                {
                    "repo_id": repo_id,
                    "filename": filename,
                    "revision": revision,
                    "cache_dir": cache_dir,
                }
            )
            return paths[repo_id]

        def fake_load_file(path):
            if path == "dpt.safetensors":
                return {"depth_head.value": torch.tensor([1.0])}
            if path == "backbone.safetensors":
                return {
                    "vision_encoder.value": torch.tensor([2.0]),
                    "text_encoder.value": torch.tensor([3.0]),
                }
            raise AssertionError(f"unexpected safetensors path: {path}")

        with mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download
        ), mock.patch("safetensors.torch.load_file", side_effect=fake_load_file):
            model = LoaderOnlyTIPSv2Model.from_pretrained(
                cache_dir="cache-dir", dtype=torch.float16, device="cpu"
            )

        self.assertEqual(
            download_calls,
            [
                {
                    "repo_id": tipsv2.TIPS2_DPT_REPO,
                    "filename": tipsv2.TIPS2_WEIGHT_FILENAME,
                    "revision": tipsv2.TIPS2_DPT_REVISION,
                    "cache_dir": "cache-dir",
                },
                {
                    "repo_id": tipsv2.TIPS2_BACKBONE_REPO,
                    "filename": tipsv2.TIPS2_WEIGHT_FILENAME,
                    "revision": tipsv2.TIPS2_BACKBONE_REVISION,
                    "cache_dir": "cache-dir",
                },
            ],
        )
        self.assertEqual(
            set(model.loaded_state.keys()), {"depth_head.value", "vision_encoder.value"}
        )
        self.assertEqual(model.to_kwargs, {"device": "cpu", "dtype": torch.float16})


class TIPSv2VisionLoaderTest(unittest.TestCase):
    @staticmethod
    def state(vision_fn="vit_base", patch_size=14, img_size=448):
        with torch.device("meta"):
            return tipsv2.TIPSv2VisionModel({
                "vision_fn": vision_fn, "patch_size": patch_size, "img_size": img_size,
            }).state_dict()

    def test_supported_architectures_infer_all_shapes_without_config_download(self):
        for name, width, depth in (("vit_base", 768, 12), ("vit_so400m", 1152, 27)):
            with self.subTest(name=name):
                state = self.state(name)
                # Real architectures on meta avoid allocating multi-GB test weights.
                state["text_encoder.unused"] = torch.empty(1, device="meta")
                with mock.patch("huggingface_hub.hf_hub_download", return_value="weights") as download, \
                     mock.patch("safetensors.torch.load_file", return_value=state), torch.device("meta"):
                    model = tipsv2.TIPSv2VisionModel.from_pretrained(
                        "fixture/repo", cache_dir="fixture-cache", device="meta", dtype=torch.float16,
                    )
                download.assert_called_once_with("fixture/repo", "model.safetensors", cache_dir="fixture-cache")
                self.assertEqual(model.config, {"vision_fn": name, "patch_size": 14, "img_size": 448})
                self.assertEqual(model.vision_config.hidden_size, width)
                self.assertEqual(model.vision_config.num_hidden_layers, depth)
                self.assertEqual(model.vision_config.num_register_tokens, 1)
                self.assertEqual(model.dtype, torch.float16)
                self.assertEqual(model.device.type, "meta")

    def test_local_cpu_safetensors_roundtrip_forward_and_input_gradient(self):
        torch.manual_seed(19)
        config = {"vision_fn": "vit_base", "patch_size": 2, "img_size": 4}
        model = tipsv2.TIPSv2VisionModel(config).eval().requires_grad_(False)
        pixels = torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            expected = model(pixels).last_hidden_state
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "model.safetensors")
            save_file(model.state_dict(), path)
            del model
            with mock.patch("huggingface_hub.hf_hub_download", return_value=path) as download:
                loaded = tipsv2.TIPSv2VisionModel.from_pretrained("fixture/local", cache_dir=folder)
            download.assert_called_once_with("fixture/local", "model.safetensors", cache_dir=folder)
            self.assertEqual(loaded.config, config)
            self.assertEqual(loaded.device.type, "cpu")
            self.assertEqual(loaded.dtype, torch.float32)
            loaded.eval().requires_grad_(False)
            loaded.gradient_checkpointing_enable()
            pixels.requires_grad_(True)
            result = loaded(pixels)
            torch.testing.assert_close(result.last_hidden_state, expected)
            self.assertEqual(len(result.hidden_states), 13)
            self.assertEqual(result.last_hidden_state.shape, (1, 6, 768))
            torch.testing.assert_close(result.pooler_output, result.last_hidden_state[:, 0])
            result.last_hidden_state[..., 0].sum().backward()
            self.assertTrue(torch.isfinite(pixels.grad).all())
            self.assertGreater(pixels.grad.abs().sum().item(), 0)
            loaded.gradient_checkpointing_disable()
            with torch.no_grad():
                self.assertIsNone(loaded(pixels, output_hidden_states=False).hidden_states)

    def test_invalid_architecture_shapes_are_rejected_before_allocation(self):
        good = self.state()
        replacements = {
            "cls_token": [(1, 768), (2, 1, 768), (1, 2, 768), (1, 1, 0)],
            "patch_embed.proj.weight": [(768, 3, 14), (767, 3, 14, 14),
                                        (768, 1, 14, 14), (768, 3, 14, 16), (768, 3, 0, 0)],
            "pos_embed": [(1025, 768), (2, 1025, 768), (1, 1025, 767), (1, 1, 768), (1, 8, 768)],
        }
        for suffix, shapes in replacements.items():
            for shape in shapes:
                with self.subTest(key=suffix, shape=shape):
                    state = {**good, f"vision_encoder.{suffix}": torch.empty(shape, device="meta")}
                    with self.assertRaises(ValueError):
                        tipsv2._infer_vision_config(state)
        for suffix in ("cls_token", "patch_embed.proj.weight", "pos_embed"):
            state = dict(good)
            del state[f"vision_encoder.{suffix}"]
            with self.subTest(missing=suffix), self.assertRaisesRegex(ValueError, "requires tensor"):
                tipsv2._infer_vision_config(state)

    def test_invalid_or_unsupported_block_layout_is_rejected(self):
        good = self.state()
        for prefix in ("vision_encoder.blocks.", "vision_encoder.blocks.0.", "vision_encoder.blocks.5."):
            state = {k: v for k, v in good.items() if not k.startswith(prefix)}
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "contiguous"):
                tipsv2._infer_vision_config(state)
        for index in ("-1", "x", "01", "١"):
            state = {**good, f"vision_encoder.blocks.{index}.norm1.weight": torch.empty(768, device="meta")}
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "block key"):
                tipsv2._infer_vision_config(state)
        state = {k: v for k, v in good.items() if not k.startswith("vision_encoder.blocks.11.")}
        with self.assertRaisesRegex(NotImplementedError, "width 768, depth 11"):
            tipsv2._infer_vision_config(state)

    def test_incomplete_extra_and_inconsistent_weights_fail_closed(self):
        good = self.state()
        missing = dict(good)
        del missing["vision_encoder.blocks.3.attn.qkv.weight"]
        invalid = [
            {}, {"text_encoder.only": torch.empty(1, device="meta")}, missing,
            {**good, "vision_encoder.extra": torch.empty(1, device="meta")},
        ]
        for key, shape in (("register_tokens", (1, 2, 768)),
                           ("blocks.3.attn.qkv.weight", (768, 768)),
                           ("blocks.3.mlp.fc1.weight", (3000, 768))):
            invalid.append({**good, f"vision_encoder.{key}": torch.empty(shape, device="meta")})
        invalid.append({**good, "vision_encoder.norm.weight": torch.empty(768, dtype=torch.int32, device="meta")})
        for index, state in enumerate(invalid):
            with self.subTest(index=index), \
                 mock.patch("huggingface_hub.hf_hub_download", return_value="weights"), \
                 mock.patch("safetensors.torch.load_file", return_value=state), \
                 self.assertRaises((ValueError, RuntimeError)):
                tipsv2.TIPSv2VisionModel.from_pretrained()


if __name__ == "__main__":
    unittest.main()
