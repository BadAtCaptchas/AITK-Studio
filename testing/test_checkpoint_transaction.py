import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.checkpoint_transaction import (
    InvalidCheckpointError, atomic_save_safetensors, begin_checkpoint, commit_checkpoint,
    committed_generations, publish_exports, prune_generations, read_generation,
    validate_legacy_checkpoint,
)
from toolkit.optimizer_checkpoint import atomic_save_optimizer_checkpoint, load_optimizer_checkpoint_for_resume


class CheckpointTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='checkpoint-transaction-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def process(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.save_root = str(self.root)
        process.job = SimpleNamespace(name='adapter')
        process.name = 'adapter'
        process.network_config = None
        process.sd = SimpleNamespace()
        return process

    def model(self, folder, step, suffix='', managed=True):
        path = Path(folder) / f'adapter_{step:09d}{suffix}.safetensors'
        metadata = {'training_info': json.dumps({'step': step})}
        if managed:
            metadata['aitk_checkpoint_generation'] = 'test'
        save_file({'weight': torch.ones(2)}, str(path), metadata=metadata)
        return path

    def generation(self, step, split=False):
        staging = begin_checkpoint(self.root)
        paths = ([self.model(staging, step, '_high_noise'), self.model(staging, step, '_low_noise')]
                 if split else [self.model(staging, step)])
        atomic_save_optimizer_checkpoint(str(Path(staging) / 'optimizer.pt'),
            {'state': {}, 'param_groups': []}, step, str(paths[0]), checkpoint_paths=paths)
        return commit_checkpoint(staging, paths[0], step)

    def test_interrupted_new_generation_preserves_previous_committed_set(self):
        original = self.generation(10)
        publish_exports(original, self.root)
        staging = begin_checkpoint(self.root)
        path = self.model(staging, 20)
        with mock.patch('toolkit.checkpoint_transaction.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                commit_checkpoint(staging, path, 20)
        self.assertEqual(self.process().get_latest_save_path(), original.primary)
        self.assertTrue(Path(original.optimizer).exists())

    def test_same_step_failed_serializer_keeps_previous_portable_file(self):
        path = self.model(self.root, 10, managed=False)
        original = path.read_bytes()
        with mock.patch('safetensors.torch.save_file', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                atomic_save_safetensors({'weight': torch.zeros(2)}, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob('.checkpoint-write-*')), [])

    def test_directory_checkpoint_commits_and_exports_complete_tree(self):
        staging = Path(begin_checkpoint(self.root))
        directory = staging / 'adapter_000000020'
        (directory / 'transformer').mkdir(parents=True)
        save_file({'weight': torch.ones(2)}, str(directory / 'transformer' / 'model.safetensors'))
        (directory / 'aitk_meta.yaml').write_text('training_info:\n  step: 20\naitk_checkpoint_generation: test\n')
        generation = commit_checkpoint(staging, directory, 20)
        publish_exports(generation, self.root)
        self.assertTrue((self.root / directory.name / 'transformer' / 'model.safetensors').exists())
        self.assertEqual(self.process().get_latest_save_path(), generation.primary)

    def test_directory_requires_real_weights_and_every_indexed_shard(self):
        staging = Path(begin_checkpoint(self.root))
        directory = staging / 'adapter_000000020'
        directory.mkdir()
        (directory / 'config.json').write_text('{}')
        with self.assertRaisesRegex(InvalidCheckpointError, 'no model weights'):
            commit_checkpoint(staging, directory, 20)
        torch.save({'weight': torch.ones(2)}, directory / 'pytorch_model.bin')
        validate_legacy_checkpoint(directory)
        index = directory / 'pytorch_model.bin.index.json'
        index.write_text(json.dumps({'weight_map': {'weight': 'missing.bin'}}))
        with self.assertRaisesRegex(InvalidCheckpointError, 'Missing checkpoint shard'):
            commit_checkpoint(staging, directory, 20)
        index.write_text(json.dumps({'weight_map': {'weight': 'pytorch_model.bin'}}))
        generation = commit_checkpoint(staging, directory, 20)
        self.assertTrue(Path(read_generation(generation.directory).primary).is_dir())

    def test_resume_rejects_linked_generation_root(self):
        self.generation(10)
        from toolkit import checkpoint_transaction
        original = checkpoint_transaction._is_link
        with mock.patch.object(checkpoint_transaction, '_is_link', side_effect=lambda path:
                Path(path).name == '.aitk-checkpoints' or original(path)):
            with self.assertRaisesRegex(InvalidCheckpointError, 'cannot be a link'):
                committed_generations(self.root)

    def test_public_save_failure_restores_live_state_without_advancing_progress(self):
        original = self.generation(10)
        process = self.process()
        process.accelerator = SimpleNamespace(is_main_process=True)
        process.step_num = 20
        process.last_save_step = 10
        process.network = SimpleNamespace(multiplier=0.75)
        process.ema = mock.Mock()
        process.post_save_hook = mock.Mock()
        def fail(_step):
            self.model(process.save_root, 20)
            process.network.multiplier = 1.0
            process.ema.eval()
            raise OSError('disk full')
        with mock.patch.object(process, '_write_checkpoint_files', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'disk full'):
                process.save(20)
        self.assertEqual(process.save_root, str(self.root))
        self.assertEqual(process.network.multiplier, 0.75)
        self.assertEqual(process.last_save_step, 10)
        self.assertEqual(process.get_latest_save_path(), original.primary)
        process.ema.train.assert_called_once()
        process.post_save_hook.assert_not_called()
        self.assertEqual(list((self.root / '.aitk-checkpoints').glob('.tmp-*')), [])

    def test_invalid_split_set_falls_back_and_cannot_use_its_portable_alias(self):
        old = self.generation(10, split=True)
        newer = self.generation(20, split=True)
        publish_exports(newer, self.root)
        Path(newer.model_files[1]).unlink()
        with mock.patch('jobs.process.BaseSDTrainProcess.print_acc'):
            self.assertIn(self.process().get_latest_save_path(), old.model_files)

    def test_all_invalid_commits_fail_instead_of_resetting_progress(self):
        generation = self.generation(10)
        publish_exports(generation, self.root)
        Path(generation.optimizer).write_bytes(b'broken')
        with mock.patch('jobs.process.BaseSDTrainProcess.print_acc'):
            with self.assertRaisesRegex(InvalidCheckpointError, 'No valid checkpoint'):
                self.process().get_latest_save_path()

    def test_legacy_selection_uses_training_step_instead_of_copy_time(self):
        older = self.model(self.root, 10, managed=False)
        newer = self.model(self.root, 20, managed=False)
        os.utime(older, (2_000_000_000, 2_000_000_000))
        self.assertEqual(self.process().get_latest_save_path(), str(newer))

    def test_malformed_legacy_directory_metadata_falls_back(self):
        original = self.model(self.root, 10, managed=False)
        directory = self.root / 'adapter_000000020'
        directory.mkdir()
        save_file({'weight': torch.ones(2)}, str(directory / 'model.safetensors'))
        (directory / 'aitk_meta.yaml').write_text('training_info: [invalid')
        with mock.patch('jobs.process.BaseSDTrainProcess.print_acc'):
            self.assertEqual(self.process().get_latest_save_path(), str(original))

    def test_explicit_corrupt_checkpoint_does_not_mutate_network(self):
        path = self.root / 'bad.safetensors'
        path.write_bytes(b'broken')
        process = self.process()
        process.network = mock.Mock()
        with self.assertRaises(InvalidCheckpointError):
            process.load_weights(str(path))
        process.network.load_weights.assert_not_called()

    def test_required_legacy_wan_half_must_exist(self):
        high = self.model(self.root, 10, '_high_noise', managed=False)
        validate_legacy_checkpoint(high, ('high',))
        with self.assertRaises(InvalidCheckpointError):
            validate_legacy_checkpoint(high, ('high', 'low'))

    def test_optimizer_digest_covers_both_split_members(self):
        generation = self.generation(10, split=True)
        load_optimizer_checkpoint_for_resume(generation.optimizer, 10, generation.primary,
            checkpoint_paths=generation.model_files)
        # A different, individually valid tensor file still cannot share the saved moments.
        save_file({'weight': torch.zeros(2)}, generation.model_files[1])
        with self.assertRaisesRegex(ValueError, 'complete checkpoint artifact set'):
            load_optimizer_checkpoint_for_resume(generation.optimizer, 10, generation.primary,
                checkpoint_paths=generation.model_files)

    def test_raw_adapter_pairing_accepts_either_verified_split_stage_only(self):
        staging = begin_checkpoint(self.root)
        paths = [self.model(staging, 10, suffix) for suffix in ('_high_noise', '_low_noise')]
        raw = {'format_version': 1, 'parameters': {'weight': torch.ones(2)}}
        atomic_save_optimizer_checkpoint(str(Path(staging) / 'optimizer.pt'),
            {'state': {}, 'param_groups': []}, 10, str(paths[0]),
            checkpoint_paths=paths, adapter_resume_state=raw)
        generation = commit_checkpoint(staging, paths[0], 10)
        selected = self.process().get_latest_save_path()
        _, restored = load_optimizer_checkpoint_for_resume(generation.optimizer, 10, selected,
            checkpoint_paths=generation.model_files)
        self.assertTrue(torch.equal(restored['parameters']['weight'], raw['parameters']['weight']))
        unrelated = self.model(self.root, 10, '_unrelated', managed=False)
        with self.assertRaisesRegex(ValueError, 'not a member'):
            load_optimizer_checkpoint_for_resume(generation.optimizer, 10, unrelated,
                checkpoint_paths=generation.model_files)

    def test_retention_removes_model_and_optimizer_generations_together(self):
        original = self.generation(10)
        latest = self.generation(20)
        prune_generations(self.root, 1)
        self.assertFalse(Path(original.directory).exists())
        self.assertTrue(Path(latest.optimizer).exists())
        self.assertEqual(len(committed_generations(self.root)[0]), 1)

    def test_export_failure_keeps_the_committed_resume_available(self):
        generation = self.generation(10)
        warnings = []
        with mock.patch('toolkit.checkpoint_transaction._link_or_copy', side_effect=OSError('locked')):
            publish_exports(generation, self.root, warnings.append)
        self.assertTrue(warnings)
        self.assertEqual(self.process().get_latest_save_path(), generation.primary)

    def test_retention_protects_new_commit_after_fallback_from_higher_corrupt_step(self):
        self.generation(10)
        corrupt = self.generation(20)
        original_bytes = Path(corrupt.optimizer).read_bytes()
        Path(corrupt.optimizer).write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
        latest = self.generation(15)
        prune_generations(self.root, 1, current_generation=latest)
        self.assertEqual(self.process().get_latest_save_path(), latest.primary)
        self.assertTrue(Path(latest.optimizer).is_file())
        self.assertFalse(Path(corrupt.directory).exists())

    def test_retention_access_failure_warns_but_containment_failure_surfaces(self):
        latest = self.generation(10)
        warnings = []
        with mock.patch('toolkit.checkpoint_transaction.committed_generations', side_effect=PermissionError('locked')):
            prune_generations(self.root, 1, warnings.append, current_generation=latest)
        self.assertTrue(warnings)
        self.assertTrue(Path(latest.optimizer).is_file())
        with mock.patch('toolkit.checkpoint_transaction.committed_generations', side_effect=InvalidCheckpointError('linked root')):
            with self.assertRaisesRegex(InvalidCheckpointError, 'linked root'):
                prune_generations(self.root, 1, current_generation=latest)

    def test_manifest_cannot_escape_generation(self):
        generation = self.generation(10)
        manifest = Path(generation.directory) / 'checkpoint.json'
        payload = json.loads(manifest.read_text())
        payload['files'][0]['path'] = '../../outside'
        manifest.write_text(json.dumps(payload))
        with self.assertRaises(InvalidCheckpointError):
            read_generation(generation.directory)

    def test_manifest_rejects_unverified_primary_and_unlisted_artifacts(self):
        generation = self.generation(10)
        manifest = Path(generation.directory) / 'checkpoint.json'
        original = manifest.read_text()
        payload = json.loads(original)
        payload['primary'] = 'optimizer.pt'
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(InvalidCheckpointError, 'not a verified model member'):
            read_generation(generation.directory)
        manifest.write_text(original)
        self.model(generation.directory, 11)
        with self.assertRaisesRegex(InvalidCheckpointError, 'unlisted'):
            read_generation(generation.directory)

    def test_wan_exporter_reports_real_files_and_loader_rejects_missing_stage(self):
        model_class = importlib.import_module('extensions_built_in.diffusion_models.wan22.wan22_14b_model').Wan2214bModel
        model = model_class.__new__(model_class)
        model.network = SimpleNamespace(network_config=SimpleNamespace(split_multistage_loras=True))
        model.train_high_noise = model.train_low_noise = True
        state = {'diffusion_model.transformer_1.layer.lora_down.weight': torch.ones(2, 2),
                 'diffusion_model.transformer_2.layer.lora_down.weight': torch.zeros(2, 2)}
        paths = model.save_lora(state, str(self.root / 'adapter.safetensors'))
        self.assertEqual(len(paths), 2)
        self.assertEqual(set(model.load_lora(paths[0])), set(state))
        Path(paths[1]).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing low-noise'):
            model.load_lora(paths[0])


if __name__ == '__main__':
    unittest.main()
