import unittest
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


if __name__ == "__main__":
    unittest.main()
