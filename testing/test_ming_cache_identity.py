"""Ming cache revisions track local weights and ordinary reference pixels."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from extensions_built_in.diffusion_models.ming_image.ming_image import (
    MingImageDesignModel, ming_component_identity,
)
from toolkit.config_modules import ModelConfig
from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin


class MingCacheIdentityTests(unittest.TestCase):
    def test_local_component_replacement_invalidates_relevant_cache_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('mllm', 'connector', 'mlp', 'vae', 'transformer'):
                (root / name).mkdir()
                (root / name / 'weights.safetensors').write_bytes(b'original')
            model = MingImageDesignModel('cpu', ModelConfig(name_or_path=directory,
                arch='ming_image_design', quantize=False), dtype='float32')

            def identities():
                model._component_identity = ming_component_identity(root, ('mllm', 'mlp', 'connector', 'vae'))
                model._vae_identity = ming_component_identity(root, ('vae',))
                return model.get_latent_space_version(), model.get_text_embedding_space_version()

            def replace(component):
                target = root / component / 'weights.safetensors'
                original = target.stat()
                replacement = root / component / 'new.safetensors'
                replacement.write_bytes(b'modified')
                # Even identical size and preserved mtime do not identify a new file.
                os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
                os.replace(replacement, target)

            first = identities()
            self.assertEqual(identities(), first)
            replace('mllm')
            text_changed = identities()
            self.assertEqual(first[0], text_changed[0])
            self.assertNotEqual(first[1], text_changed[1])
            replace('vae')
            vae_changed = identities()
            self.assertNotEqual(text_changed[0], vae_changed[0])
            self.assertNotEqual(text_changed[1], vae_changed[1])
            replace('transformer')
            self.assertEqual(identities(), vae_changed)

    def test_ming_reference_content_changes_cache_key_without_path_change(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference.png'
            Image.new('RGBA', (16, 16), 'red').save(reference)
            item = SimpleNamespace(caption='poster', text_embedding_space_version='ming_image_design_conditioning_pr8021_v1_test',
                text_embedding_version=1, is_layered=False, is_encrypted=False,
                preserve_image_alpha=True, scale_to_width=16, scale_to_height=16,
                crop_x=0, crop_y=0, crop_width=16, crop_height=16, flip_x=False, flip_y=False,
                control_path=str(reference), encode_control_in_text_embeddings=True)
            info = TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict
            before = info(item)
            Image.new('RGBA', (16, 16), 'blue').save(reference)
            after = info(item)
            self.assertEqual(before['control_path'], after['control_path'])
            self.assertNotEqual(before['ming_reference_contents'], after['ming_reference_contents'])
            self.assertNotIn('ming_reference_contents', info(item, text_only=True))
            item.text_embedding_space_version = 'unrelated-model'
            reference.unlink()
            self.assertNotIn('ming_reference_contents', info(item))
            item.text_embedding_space_version = 'ming_image_design_conditioning_pr8021_v1_test'
            item.is_encrypted = True
            self.assertNotIn('ming_reference_contents', info(item))


if __name__ == '__main__':
    unittest.main()
