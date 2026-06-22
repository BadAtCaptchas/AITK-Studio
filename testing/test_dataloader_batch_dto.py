import types
import unittest

import torch

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


def make_file_item(unconditional_tensor=None):
    return types.SimpleNamespace(
        is_latent_cached=False,
        dataset_config=types.SimpleNamespace(load_image_when_caching_latents=False),
        tensor=torch.ones(1, 2, 2),
        extra_values=[],
        audio_data=None,
        num_frames=1,
        control_tensor=None,
        control_tensor_list=None,
        inpaint_tensor=None,
        loss_multiplier=1.0,
        clip_image_tensor=None,
        mask_tensor=None,
        unaugmented_tensor=None,
        unconditional_tensor=unconditional_tensor,
        clip_image_embeds=None,
        clip_image_embeds_unconditional=None,
        prompt_embeds=None,
        audio_tensor=None,
    )


class DataLoaderBatchDTOTest(unittest.TestCase):
    def test_mixed_unconditional_tensors_are_zero_filled(self):
        unconditional_tensor = torch.full((1, 2, 2), 3.0)
        batch = DataLoaderBatchDTO(
            file_items=[
                make_file_item(unconditional_tensor=unconditional_tensor),
                make_file_item(),
            ],
        )

        self.assertTrue(torch.equal(batch.unconditional_tensor[0], unconditional_tensor))
        self.assertTrue(torch.equal(batch.unconditional_tensor[1], torch.zeros_like(unconditional_tensor)))


if __name__ == "__main__":
    unittest.main()
