import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.flux2.flux2_model import Flux2Model
from extensions_built_in.diffusion_models.flux2.src.pipeline import Flux2Pipeline
from toolkit.prompt_utils import PromptEmbeds


class RecordingPipeline:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(images=["sample"])


class ProgressBar:
    def __init__(self, total):
        self.total = total
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount=1):
        self.updates += amount


class RecordingTransformer(torch.nn.Module):
    def __init__(self, expected_device):
        super().__init__()
        self.expected_device = torch.device(expected_device)
        self.calls = []

    def forward(self, **kwargs):
        tensor_inputs = {
            name: value
            for name, value in kwargs.items()
            if torch.is_tensor(value)
        }
        self.calls.append({name: value.device for name, value in tensor_inputs.items()})
        for name, value in tensor_inputs.items():
            self.assert_same_device(name, value)
        return torch.zeros_like(kwargs["x"])

    def assert_same_device(self, name, value):
        if value.device != self.expected_device:
            raise AssertionError(
                f"{name} is on {value.device}, expected {self.expected_device}"
            )


class Flux2SamplingDeviceTest(unittest.TestCase):
    def test_generate_single_image_passes_training_device_to_pipeline(self):
        model = object.__new__(Flux2Model)
        model.device_torch = torch.device("cuda")
        model.flux2_is_guidance_distilled = False
        model.get_bucket_divisibility = lambda: 16
        pipeline = RecordingPipeline()
        gen_config = SimpleNamespace(
            width=31,
            height=47,
            ctrl_img=None,
            ctrl_img_1=None,
            ctrl_img_2=None,
            ctrl_img_3=None,
            num_inference_steps=1,
            guidance_scale=7.0,
            latents=None,
        )
        conditional = PromptEmbeds(torch.ones(1, 1, 2))
        unconditional = PromptEmbeds(torch.zeros(1, 1, 2))

        image = model.generate_single_image(
            pipeline,
            gen_config,
            conditional,
            unconditional,
            torch.Generator(),
            {},
        )

        self.assertEqual(image, "sample")
        self.assertEqual(pipeline.kwargs["device"], torch.device("cuda"))
        self.assertEqual(pipeline.kwargs["width"], 16)
        self.assertEqual(pipeline.kwargs["height"], 32)
        self.assertIs(
            pipeline.kwargs["negative_prompt_embeds"],
            unconditional.text_embeds,
        )

    def test_encode_prompt_moves_precomputed_embeds_to_explicit_device(self):
        pipeline = object.__new__(Flux2Pipeline)
        prompt_embeds = torch.ones(1, 2, 3)
        prompt_embeds_mask = torch.ones(1, 2, dtype=torch.bool)

        encoded, encoded_mask = pipeline.encode_prompt(
            prompt=None,
            device=torch.device("meta"),
            num_images_per_prompt=2,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        self.assertEqual(encoded.device.type, "meta")
        self.assertEqual(encoded_mask.device.type, "meta")
        self.assertEqual(tuple(encoded.shape), (2, 2, 3))

    def test_call_uses_explicit_device_for_preencoded_denoising_inputs(self):
        pipeline = object.__new__(Flux2Pipeline)
        object.__setattr__(pipeline, "_internal_dict", {})
        expected_device = torch.device("cpu")
        transformer = RecordingTransformer(expected_device)
        pipeline.transformer = transformer
        pipeline.vae_scale_factor = 16
        pipeline.num_channels_latents = 2
        pipeline.default_sample_size = 1
        pipeline.is_guidance_distilled = False
        pipeline.progress_bar = lambda total: ProgressBar(total)
        pipeline.maybe_free_model_hooks = lambda: None

        output = pipeline(
            prompt_embeds=torch.ones(1, 1, 2),
            negative_prompt_embeds=torch.zeros(1, 1, 2),
            height=16,
            width=16,
            num_inference_steps=1,
            guidance_scale=7.0,
            latents=torch.ones(1, 2, 1, 1),
            output_type="latent",
            device=expected_device,
        )

        self.assertEqual(len(transformer.calls), 2)
        for call in transformer.calls:
            self.assertEqual(call["x"], expected_device)
            self.assertEqual(call["x_ids"], expected_device)
            self.assertEqual(call["timesteps"], expected_device)
            self.assertEqual(call["ctx"], expected_device)
            self.assertEqual(call["ctx_ids"], expected_device)
            self.assertEqual(call["guidance"], expected_device)
        self.assertEqual(tuple(output.images.shape), (1, 2, 1, 1))


if __name__ == "__main__":
    unittest.main()
