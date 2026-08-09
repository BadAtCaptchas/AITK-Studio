import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class MixedMediaContractTests(unittest.TestCase):
    def test_public_config_defaults_are_backward_compatible(self):
        source = _source("toolkit/config_modules.py")
        self.assertIn(
            "kwargs.get(\n            'include_images_in_video_dataset', False\n        )",
            source,
        )
        self.assertIn("'guidance_loss_schedule', 'constant'", source)
        self.assertIn("'minimax_h3'", source)

    def test_mixed_media_is_classified_per_item(self):
        source = _source("toolkit/data_transfer_object/data_loader.py")
        self.assertIn("self.encrypted_item.mediaKind == \"video\"", source)
        self.assertIn("os.path.splitext(self.path)[1].lower() in VIDEO_EXTENSIONS", source)
        self.assertIn("requested_num_frames = self.dataset_config.num_frames if self.is_video else 1", source)
        self.assertIn("snap_fixed_video_frame_count(", source)

    def test_discovery_and_bucketing_keep_media_separate(self):
        loader = _source("toolkit/data_loader.py")
        mixins = _source("toolkit/dataloader_mixins.py")
        self.assertIn("media_kinds.append(\"image\")", loader)
        self.assertIn("video_extensions + image_extensions", loader)
        self.assertIn("f'{bucket_key}x{file_item.num_frames}f'", mixins)
        self.assertIn("if self.is_video:\n            self.load_and_process_video", mixins)
        self.assertIn("is_video = file_item.is_video", mixins)

    def test_model_frame_snapper_covers_plain_and_encrypted_video(self):
        base_model = _source("toolkit/models/base_model.py")
        mixins = _source("toolkit/dataloader_mixins.py")
        encrypted = _source("toolkit/encrypted_dataset.py")
        self.assertIn("def get_frame_count_snapper(self):", base_model)
        self.assertIn("desired_num_frames = self.frame_count_snapper", mixins)
        self.assertIn("selected_num_frames = frame_count_snapper", encrypted)

    def test_silent_videos_are_probed_without_plaintext_sidecars(self):
        mixins = _source("toolkit/dataloader_mixins.py")
        encrypted = _source("toolkit/encrypted_dataset.py")
        self.assertIn("len(audio_probe.streams.audio) > 0", mixins)
        self.assertIn("def has_audio_stream", encrypted)
        self.assertIn("av.open(io.BytesIO(data))", encrypted)


class AudioTrainingContractTests(unittest.TestCase):
    def test_prediction_slots_are_typed_and_cleaned_up(self):
        source = _source("toolkit/data_transfer_object/data_loader.py")
        tree = ast.parse(source)
        methods = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("set_secondary_audio_pred", methods)
        for field in (
            "audio_noise",
            "audio_pred_uncond",
            "audio_pred_prior",
            "audio_pred_preservation",
            "audio_noisy",
            "audio_sigma",
        ):
            self.assertIn(field, source)
        self.assertIn("if any(x.audio_data is not None for x in self.file_items)", source)
        self.assertIn('"waveform": torch.zeros_like(base_audio_data["waveform"])', source)

    def test_ltx_reuses_noise_and_rejects_stale_image_audio(self):
        source = _source("extensions_built_in/diffusion_models/ltx2/ltx2.py")
        self.assertIn("batch.audio_noise.shape == raw_audio_latents.shape", source)
        self.assertIn("getattr(batch, \"num_frames\", 1) > 1", source)
        self.assertIn("batch.set_secondary_audio_pred(noise_pred_audio)", source)
        self.assertIn("batch.audio_pred = noise_pred_audio", source)

    def test_trainer_routes_all_secondary_audio_passes(self):
        source = _source("extensions_built_in/sd_trainer/SDTrainer.py")
        for slot in (
            '"audio_pred_uncond"',
            '"audio_pred_prior"',
            '"audio_pred_preservation"',
        ):
            self.assertIn(f"batch.audio_pred_slot = {slot}", source)
        self.assertIn("guidance_loss_schedule == \"sigma\"", source)
        self.assertIn("audio_preservation_loss", source)
        self.assertIn("not torch.isfinite(prior_loss).all()", source)
        self.assertIn("not torch.isfinite(loss).all()", source)

    def test_h3_reuses_joint_conditioning_and_rejects_stale_audio(self):
        source = _source(
            "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py"
        )
        self.assertIn(
            "batch.keyframe_conditioning_noise.shape == first_latents.shape",
            source,
        )
        self.assertIn("tuple(batch.audio_noise.shape) == silence_shape", source)
        self.assertIn("batch.audio_target = None", source)
        self.assertIn("def _uses_latent_keyframe_conditioning", source)
        self.assertIn('return self._partition().startswith("fl2va")', source)

    def test_h3_and_nvfp4_packed_modes_are_rejected_before_mutation(self):
        source = _source("toolkit/config_modules.py")
        self.assertIn("model_config.quantize\n        and str(model_config.qtype", source)
        self.assertIn("MiniMax H3 does not support model.base_lora_path", source)
        self.assertIn("MiniMax H3 does not support model.inference_lora_path", source)
        self.assertIn("ref2va I2V training is not supported yet", source)


if __name__ == "__main__":
    unittest.main()
