import types
import unittest

import torch

from toolkit.data_transfer_object.data_loader import (
    DataLoaderBatchDTO,
    snap_fixed_video_frame_count,
)


def make_file_item(
    unconditional_tensor=None,
    audio_data=None,
    audio_tensor=None,
    num_frames=1,
):
    return types.SimpleNamespace(
        is_latent_cached=False,
        dataset_config=types.SimpleNamespace(load_image_when_caching_latents=False),
        tensor=torch.ones(1, 2, 2),
        extra_values=[],
        audio_data=audio_data,
        num_frames=num_frames,
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
        audio_tensor=audio_tensor,
    )


class DataLoaderBatchDTOTest(unittest.TestCase):
    def test_fixed_video_frame_snap_happens_before_batch_audio_geometry(self):
        snapper = lambda frames: 39 if frames == 40 else frames

        self.assertEqual(
            snap_fixed_video_frame_count(
                40,
                is_video=True,
                auto_frame_count=False,
                frame_count_snapper=snapper,
            ),
            39,
        )
        self.assertEqual(
            snap_fixed_video_frame_count(
                1,
                is_video=False,
                auto_frame_count=False,
                frame_count_snapper=lambda _frames: 5,
            ),
            1,
        )
        self.assertEqual(
            snap_fixed_video_frame_count(
                40,
                is_video=True,
                auto_frame_count=True,
                frame_count_snapper=snapper,
            ),
            40,
        )

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

    def test_mixed_silent_and_audio_rows_use_zero_waveform_entries(self):
        waveform = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        audio_data = {"waveform": waveform, "sample_rate": 16_000}
        batch = DataLoaderBatchDTO(
            file_items=[
                make_file_item(num_frames=17),
                make_file_item(
                    audio_data=audio_data,
                    audio_tensor=waveform,
                    num_frames=17,
                ),
            ],
        )

        self.assertEqual(len(batch.audio_data), 2)
        self.assertEqual(batch.audio_data[0]["sample_rate"], 16_000)
        self.assertTrue(
            torch.equal(batch.audio_data[0]["waveform"], torch.zeros_like(waveform))
        )
        self.assertTrue(torch.equal(batch.audio_data[1]["waveform"], waveform))
        self.assertEqual(tuple(batch.audio_tensor.shape), (2, 2, 4))

    def test_audio_tensors_are_padded_to_the_longest_row(self):
        short = torch.ones((2, 3))
        long = torch.ones((2, 5))
        batch = DataLoaderBatchDTO(
            file_items=[
                make_file_item(
                    audio_data={"waveform": short, "sample_rate": 16_000},
                    audio_tensor=short,
                    num_frames=17,
                ),
                make_file_item(
                    audio_data={"waveform": long, "sample_rate": 16_000},
                    audio_tensor=long,
                    num_frames=17,
                ),
            ],
        )

        self.assertEqual(tuple(batch.audio_tensor.shape), (2, 2, 5))
        self.assertTrue(torch.equal(batch.audio_tensor[0, :, 3:], torch.zeros((2, 2))))


if __name__ == "__main__":
    unittest.main()
