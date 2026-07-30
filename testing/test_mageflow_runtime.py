import unittest

import torch

from extensions_built_in.diffusion_models import MageFlowEditModel, MageFlowModel
from extensions_built_in.diffusion_models.mageflow.src.pipeline import (
    build_shifted_sigmas,
    pack_text_features,
    predict_velocity,
)


class RecordingMageFlow:
    def __init__(self):
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return kwargs["img"]


class MageFlowRuntimeTests(unittest.TestCase):
    def test_optional_registry_imports_without_loading_weights(self):
        self.assertEqual(MageFlowModel.arch, "mageflow")
        self.assertEqual(MageFlowEditModel.arch, "mageflow_edit")

    def test_text_and_image_sequences_pack_without_padding(self):
        text = [torch.ones(3, 5), torch.full((1, 5), 2.0)]
        packed_text, text_cu = pack_text_features(text, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(packed_text.shape), (1, 4, 5))
        self.assertTrue(torch.equal(text_cu, torch.tensor([0, 3, 4], dtype=torch.int32)))

        latents = torch.arange(16, dtype=torch.float32).view(2, 2, 2, 2)
        refs = [
            [torch.full((2, 1, 2), 100.0)],
            [
                torch.full((2, 1, 1), 200.0),
                torch.full((2, 2, 1), 300.0),
            ],
        ]
        model = RecordingMageFlow()
        output = predict_velocity(
            model,
            latents,
            torch.tensor([0.25, 0.75]),
            text,
            ref_latents=refs,
        )

        self.assertTrue(torch.equal(output, latents))
        self.assertEqual(tuple(model.call["img"].shape), (1, 13, 2))
        self.assertTrue(
            torch.equal(
                model.call["img_cu_seqlens"],
                torch.tensor([0, 6, 13], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                model.call["txt_cu_seqlens"],
                torch.tensor([0, 3, 4], dtype=torch.int32),
            )
        )
        self.assertEqual(
            model.call["img_shapes"],
            [[(1, 2, 2), (1, 1, 2), (1, 2, 2), (1, 1, 1), (1, 2, 1)]],
        )

    def test_shifted_scheduler_is_monotonic_and_has_clean_terminal(self):
        sigmas = build_shifted_sigmas(4, shift=6.0)
        expected = torch.tensor([1.0, 18.0 / 19.0, 3.0 / 3.5, 1.5 / 2.25, 0.0])
        self.assertTrue(torch.allclose(sigmas, expected))
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        self.assertEqual(float(sigmas[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
