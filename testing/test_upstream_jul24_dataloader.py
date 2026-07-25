import os
import sys
import tempfile
import unittest
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torchaudio_module = ModuleType("torchaudio")
torchaudio_module.__spec__ = ModuleSpec("torchaudio", loader=None)
torchaudio_module.save = mock.Mock()
album_artwork_module = ModuleType("toolkit.audio.album_artwork")
album_artwork_module.__spec__ = ModuleSpec("toolkit.audio.album_artwork", loader=None)
album_artwork_module.add_album_artwork = mock.Mock()

with mock.patch.dict(
    "sys.modules",
    {
        "torchaudio": torchaudio_module,
        "toolkit.audio.album_artwork": album_artwork_module,
    },
):
    import toolkit.dataloader_mixins as dataloader_mixins


class FakeVideoCapture:
    def __init__(self, frame_count, fps, fail_reads=None):
        self.frames = [
            np.full((2, 2, 3), frame_index * 20, dtype=np.uint8)
            for frame_index in range(frame_count)
        ]
        self.fps = float(fps)
        self.position = 0
        self.fail_reads = dict(fail_reads or {})
        self.set_calls = []
        self.grab_positions = []
        self.read_positions = []
        self.release_calls = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        self.set_calls.append(self.position)
        return True

    def grab(self):
        if self.position >= len(self.frames):
            return False
        self.grab_positions.append(self.position)
        self.position += 1
        return True

    def read(self):
        frame_index = self.position
        self.read_positions.append(frame_index)
        remaining_failures = self.fail_reads.get(frame_index, 0)
        if remaining_failures > 0:
            self.fail_reads[frame_index] = remaining_failures - 1
            return False, None
        if frame_index >= len(self.frames):
            return False, None
        self.position += 1
        return True, self.frames[frame_index].copy()

    def release(self):
        self.release_calls += 1


def _video_dto(num_frames, dataset_fps):
    return SimpleNamespace(
        augments=None,
        has_augmentations=False,
        dataset_config=SimpleNamespace(
            buckets=True,
            do_audio=False,
            fps=dataset_fps,
            shrink_video_to_frames=False,
            auto_frame_count=False,
            debug=False,
        ),
        path="fixture.mp4",
        num_frames=num_frames,
        temporal_compression=4,
        flip_x=False,
        flip_y=False,
        scale_to_width=2,
        scale_to_height=2,
        crop_x=0,
        crop_y=0,
        crop_width=2,
        crop_height=2,
    )


def _tensor_transform(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


class SequentialVideoLoadingTest(unittest.TestCase):
    def _load(self, dto, capture):
        with (
            mock.patch.object(dataloader_mixins.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(dataloader_mixins.random, "randint", return_value=0),
        ):
            dataloader_mixins.ImageProcessingDTOMixin.load_and_process_video(
                dto,
                _tensor_transform,
            )

    def test_sparse_frames_decode_forward_with_grab_and_release(self):
        dto = _video_dto(num_frames=3, dataset_fps=3)
        capture = FakeVideoCapture(frame_count=6, fps=6)

        self._load(dto, capture)

        self.assertEqual(capture.set_calls, [])
        self.assertEqual(capture.read_positions, [0, 2, 4])
        self.assertEqual(capture.grab_positions, [1, 3])
        self.assertEqual(dto.tensor[:, 0, 0, 0].tolist(), [0, 40, 80])
        self.assertEqual(capture.release_calls, 1)

    def test_stretched_video_reuses_duplicate_decodes_in_requested_order(self):
        dto = _video_dto(num_frames=4, dataset_fps=2)
        capture = FakeVideoCapture(frame_count=2, fps=2)

        self._load(dto, capture)

        self.assertEqual(capture.read_positions, [0, 1])
        self.assertEqual(dto.tensor[:, 0, 0, 0].tolist(), [0, 0, 20, 20])
        self.assertEqual(capture.release_calls, 1)

    def test_positive_fallback_reseeks_before_the_next_requested_frame(self):
        dto = _video_dto(num_frames=2, dataset_fps=3)
        capture = FakeVideoCapture(frame_count=3, fps=3, fail_reads={0: 1})

        self._load(dto, capture)

        self.assertEqual(capture.set_calls, [1, 1])
        self.assertEqual(dto.tensor[:, 0, 0, 0].tolist(), [20, 20])
        self.assertEqual(capture.release_calls, 1)

    def test_capture_is_released_when_video_validation_fails(self):
        dto = _video_dto(num_frames=2, dataset_fps=3)
        capture = FakeVideoCapture(frame_count=0, fps=3)

        with (
            mock.patch.object(dataloader_mixins.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(dataloader_mixins.traceback, "print_exc"),
            mock.patch.object(dataloader_mixins, "print_acc"),
        ):
            with self.assertRaisesRegex(Exception, "Invalid video frame count"):
                dataloader_mixins.ImageProcessingDTOMixin.load_and_process_video(
                    dto,
                    _tensor_transform,
                )

        self.assertEqual(capture.release_calls, 1)


class Uint8PixelCacheTest(unittest.TestCase):
    def test_pixel_cache_conversion_clamps_and_round_trips(self):
        values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

        encoded = dataloader_mixins._latent_to_uint8(values)
        decoded = dataloader_mixins._latent_from_uint8(encoded)

        self.assertEqual(encoded.dtype, torch.uint8)
        self.assertEqual(encoded.tolist(), [0, 0, 64, 128, 191, 255, 255])
        expected = values.clamp(-1, 1)
        self.assertLessEqual(torch.max(torch.abs(decoded - expected)).item(), (1.0 / 255.0) + 1e-6)

    def test_opt_in_compresses_only_disk_pixels_and_loads_transparently(self):
        source_frames = torch.linspace(-0.9, 0.9, steps=24).reshape(2, 3, 2, 2)
        source_audio = torch.tensor([0.125, -0.25], dtype=torch.float32)

        class FakePixelSD:
            torch_dtype = torch.float32
            device_torch = torch.device("cpu")
            cache_latents_as_uint8 = True

            @staticmethod
            def encode_images(images):
                return images

            @staticmethod
            def encode_audio(_audio):
                return source_audio.unsqueeze(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            latent_path = os.path.join(temp_dir, "pixel-cache.safetensors")
            file_item = SimpleNamespace(
                tensor=source_frames.clone(),
                audio_data={"waveform": torch.ones(1, 4), "sample_rate": 4},
                num_frames=2,
                path="fixture.mp4",
                load_and_process_image=lambda _transform, only_load_latents: None,
                get_latent_info_dict=lambda: {},
                cleanup=lambda: None,
            )
            dataset = SimpleNamespace(
                transform=None,
                sd=FakePixelSD(),
                dataset_config=SimpleNamespace(
                    auto_frame_count=False,
                    num_frames=2,
                    do_i2v=True,
                ),
                is_audio_model=False,
            )

            memory_state = dataloader_mixins.LatentCachingMixin._encode_latent_for_file_item(
                dataset,
                file_item,
                latent_path,
                to_disk=True,
            )
            disk_state = dataloader_mixins.load_file(latent_path, device="cpu")

            self.assertEqual(memory_state["latent"].dtype, torch.float32)
            self.assertTrue(torch.equal(memory_state["latent"], source_frames))
            self.assertEqual(memory_state["first_frame_latent"].dtype, torch.float32)
            self.assertTrue(torch.equal(memory_state["first_frame_latent"], source_frames[0]))
            self.assertEqual(disk_state["latent"].dtype, torch.uint8)
            self.assertEqual(disk_state["first_frame_latent"].dtype, torch.uint8)
            self.assertEqual(disk_state["audio_latent"].dtype, torch.float32)
            self.assertTrue(torch.equal(disk_state["audio_latent"], source_audio))

            native_memory = SimpleNamespace()
            dataloader_mixins.LatentCachingMixin._assign_latent_state_to_file_items(
                dataset,
                [native_memory],
                memory_state,
            )
            self.assertTrue(torch.equal(native_memory._encoded_latent, source_frames))
            self.assertTrue(torch.equal(native_memory._cached_first_frame_latent, source_frames[0]))

            assigned = SimpleNamespace()
            dataloader_mixins.LatentCachingMixin._assign_latent_state_to_file_items(
                dataset,
                [assigned],
                disk_state,
            )
            self.assertEqual(assigned._encoded_latent.dtype, torch.float32)
            self.assertTrue(torch.allclose(assigned._encoded_latent, source_frames, atol=1.0 / 255.0))
            self.assertTrue(
                torch.allclose(
                    assigned._cached_first_frame_latent,
                    source_frames[0],
                    atol=1.0 / 255.0,
                )
            )
            self.assertTrue(torch.equal(assigned._cached_audio_latent, source_audio))

            lazy_item = SimpleNamespace(
                is_latent_cached=True,
                _encoded_latent=None,
                _cached_first_frame_latent=None,
                _cached_audio_latent=None,
                get_latent_path=lambda: latent_path,
            )
            lazy_latent = dataloader_mixins.LatentCachingFileItemDTOMixin.get_latent(lazy_item)
            self.assertEqual(lazy_latent.dtype, torch.float32)
            self.assertTrue(torch.allclose(lazy_latent, source_frames, atol=1.0 / 255.0))
            self.assertTrue(
                torch.allclose(
                    lazy_item._cached_first_frame_latent,
                    source_frames[0],
                    atol=1.0 / 255.0,
                )
            )
            self.assertTrue(torch.equal(lazy_item._cached_audio_latent, source_audio))

    def test_flag_disabled_preserves_float_disk_cache(self):
        source = torch.tensor([[[[-0.5, 0.25]]]], dtype=torch.float32)
        sd = SimpleNamespace(
            torch_dtype=torch.float32,
            device_torch=torch.device("cpu"),
            cache_latents_as_uint8=False,
            encode_images=lambda images: images,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            latent_path = os.path.join(temp_dir, "float-cache.safetensors")
            file_item = SimpleNamespace(
                tensor=source.clone(),
                audio_data=None,
                num_frames=1,
                path="fixture.png",
                load_and_process_image=lambda _transform, only_load_latents: None,
                get_latent_info_dict=lambda: {},
                cleanup=lambda: None,
            )
            dataset = SimpleNamespace(
                transform=None,
                sd=sd,
                dataset_config=SimpleNamespace(
                    auto_frame_count=False,
                    num_frames=1,
                    do_i2v=False,
                ),
                is_audio_model=False,
            )

            dataloader_mixins.LatentCachingMixin._encode_latent_for_file_item(
                dataset,
                file_item,
                latent_path,
                to_disk=True,
            )
            disk_state = dataloader_mixins.load_file(latent_path, device="cpu")

            self.assertEqual(disk_state["latent"].dtype, torch.float32)
            self.assertTrue(torch.equal(disk_state["latent"], source))


if __name__ == "__main__":
    unittest.main()
