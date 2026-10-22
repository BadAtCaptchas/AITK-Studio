import tempfile
import unittest
from pathlib import Path
import torch
from extensions_built_in.captioner.BaseCaptioner import BaseCaptioner, CaptionConfig
from testing.test_shared_training_batch_math import make_batch, make_loss_trainer


class OctoberRuntimeTests(unittest.TestCase):
    def test_caption_builder_uses_tokens_and_keeps_json_contract(self):
        owner = BaseCaptioner.__new__(BaseCaptioner)
        owner.encrypted_reader = None
        owner.caption_config = CaptionConfig(model_name_or_path='test', path_to_caption='test', extensions=['png'], caption_prompt='Describe <caption.txt>')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            path.with_suffix('.txt').write_text('a red bird')
            self.assertEqual(owner.build_caption_prompt(str(path)), 'Describe a red bird')
            owner.caption_config.output_format = 'json'
            prompt = owner.build_caption_prompt(str(path))
            self.assertIn('Describe a red bird', prompt)
            self.assertIn('elements', prompt)
        owner.encrypted_reader = object()
        owner.get_source_caption_for_file = lambda _: 'encrypted caption'
        owner.caption_config.source_caption_extension = 'txt'
        self.assertEqual(owner.resolve_prompt('<caption.txt>|<caption.json>', 'not-a-plaintext-path'), 'encrypted caption|')

    def test_omni_messages_expand_tokens_for_every_media_kind(self):
        from extensions_built_in.captioner.Qwen25OmniCaptioner import Qwen25OmniCaptioner
        from extensions_built_in.captioner.Qwen3OmniCaptioner import Qwen3OmniCaptioner
        with tempfile.TemporaryDirectory() as folder:
            for captioner_class in (Qwen25OmniCaptioner, Qwen3OmniCaptioner):
                owner = captioner_class.__new__(captioner_class)
                owner.encrypted_reader = None
                owner.caption_config = CaptionConfig(model_name_or_path='test', path_to_caption=folder,
                    extensions=['png', 'wav', 'mp4'], caption_prompt='Describe <caption.txt>')
                for extension, media_kind in (('png', 'image'), ('wav', 'audio'), ('mp4', 'video')):
                    path = Path(folder) / f'media.{extension}'
                    path.with_suffix('.txt').write_text('a bird')
                    content = owner._build_messages(str(path))[0]['content']
                    self.assertEqual(content[0]['type'], media_kind)
                    self.assertEqual(content[1]['text'], 'Describe a bird')

    def test_velocity_equivalent_t0_and_fft_low_sigma_losses_and_gradients(self):
        for timestep in (0, 5, 10, 50, 500):
            for fft in (False, True):
                trainer = make_loss_trainer(flow=True, t0_loss_target=not fft,
                    t0_velocity_equiv_weight=not fft, do_fft_loss=fft, do_fft_velocity_equiv_weight=fft)
                batch = make_batch(1)
                parameter = torch.nn.Parameter(torch.tensor(1.))
                prediction = parameter.expand_as(batch.latents)
                noise = batch.test_noise
                loss = trainer.calculate_loss(prediction, noise, noise, torch.tensor([timestep]), batch,
                    mask_multiplier=torch.ones(1, 1, 2, 2))
                sigma = max(timestep / 1000, .001)
                weight = max(sigma, .01) ** -2
                expected = (2 - sigma) ** 2 * weight + (1 if fft else 0)
                self.assertAlmostEqual(loss.item() / expected, 1., places=5)
                loss.backward()
                derivative = -2 * sigma * (2 - sigma) * weight - (2 if fft else 0)
                self.assertAlmostEqual(parameter.grad.item() / derivative, 1., places=5)
