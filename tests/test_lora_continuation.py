"""Functional tests with real Krea modules; no downloaded model weights."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import save_file
from tests.test_kroma import tiny_config, holder
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT
from toolkit.config_modules import NetworkConfig, ModelConfig, TrainConfig, DatasetConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lora_continuation import (inspect_lora, load_strict, validate_targets,
    snapshot_file, source_teacher_context, preservation_loss, build_source_teacher,
    validate_continuation_config, sha256_file, native_module)


class ContinuationTests(unittest.TestCase):
    def test_metadata_publication_is_atomic_and_no_replace(self):
        from toolkit.continuation_assets import pin_json
        path = Path(self.tmp.name) / 'metadata.json'
        value = {'source': 'a' * 64}
        with patch('toolkit.continuation_assets.os.fsync', side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                pin_json(path, value, 'conflict')
        self.assertEqual(list(path.parent.iterdir()), [])
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: pin_json(path, value, 'conflict'), range(16)))
        self.assertEqual(json.loads(path.read_text()), value)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            pin_json(path, {'source': 'b' * 64}, 'conflict')
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_finite_weights_cannot_overflow_import_precision(self):
        model = SingleStreamDiT(tiny_config())
        weights = self.source(model)
        key = next(key for key in weights if '.lora_A.' in key)
        weights[key] = torch.full_like(weights[key], 1e300, dtype=torch.float64)
        save_file(weights, str(self.path))
        with self.assertRaisesRegex(ValueError, 'conversion overflow'):
            inspect_lora(self.path)
        self.source(model)
        inspected = inspect_lora(self.path)
        net = self.network(model, holder(), inspected).to(dtype=torch.float16)
        before = {key: value.clone() for key, value in net.state_dict().items()}
        inspected.state[next(iter(inspected.state))].fill_(1e30)
        with self.assertRaisesRegex(ValueError, 'adapter precision'):
            load_strict(net, inspected)
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_failed_teacher_installation_restores_shared_forwards(self):
        model = SingleStreamDiT(tiny_config()).requires_grad_(False)
        self.source(model)
        inspected = inspect_lora(self.path)
        base = holder()
        base.model = model
        student = self.network(model, base, inspected)
        process = SimpleNamespace(sd=base, network_config=NetworkConfig(type='lora'), network=student, device_torch='cpu')
        layer = model.blocks[0].attn.wq
        original_forward = layer.forward
        x = torch.randn(2, layer.in_features)
        prediction = layer(x).detach().clone()
        with patch.object(LoRASpecialNetwork, 'force_to', side_effect=RuntimeError('device transfer failed')):
            with self.assertRaisesRegex(RuntimeError, 'device transfer failed'):
                build_source_teacher(process, inspected)
        self.assertEqual(layer.forward, original_forward)
        torch.testing.assert_close(layer(x), prediction, rtol=0, atol=0)

    def test_curated_manifest_rejects_replacement_and_extra_images(self):
        from toolkit.preservation_manifest import validate_preservation_manifest
        root = Path(self.tmp.name)
        image, caption = root / '0000.png', root / '0000.txt'
        image.write_bytes(b'image fixture')
        caption.write_text('a candid photo', encoding='utf-8')
        manifest = dict(version=1, source_sha256='a' * 64, examples=[dict(image=image.name, caption=caption.name,
            image_sha256=sha256_file(image), caption_sha256=sha256_file(caption))])
        (root / '.aitk-preservation.json').write_text(json.dumps(manifest), encoding='utf-8')
        validate_preservation_manifest(root, 'a' * 64)
        with self.assertRaisesRegex(ValueError, 'source LoRA'):
            validate_preservation_manifest(root, 'b' * 64)
        extra = root / 'unreviewed.png'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'uncurated'):
            validate_preservation_manifest(root, 'a' * 64)
        extra.unlink()
        caption.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after curation'):
            validate_preservation_manifest(root, 'a' * 64)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'source.safetensors'
        torch.manual_seed(42)

    def source(self, model, alpha=False):
        weights = {}
        modules = dict(model.named_modules())
        for name, rank in [('blocks.0.attn.wq', 2), ('txtmlp.1', 3), ('txtfusion.projector', 1)]:
            out_features, in_features = modules[name].weight.shape
            weights[f'diffusion_model.{name}.lora_A.weight'] = torch.randn(rank, in_features) * .03
            weights[f'diffusion_model.{name}.lora_B.weight'] = torch.randn(out_features, rank) * .03
            if alpha:
                weights[f'diffusion_model.{name}.alpha'] = torch.tensor(float(rank * 2))
        save_file(weights, str(self.path))
        return weights

    def network(self, model, base, inspected):
        config = NetworkConfig(type='lora', linear=1)
        net = LoRASpecialNetwork(text_encoder=None, unet=model, lora_dim=1, alpha=1,
            train_unet=True, train_text_encoder=False, network_config=config,
            is_transformer=True, transformer_only=False, base_model=base,
            target_lin_modules=['SingleStreamDiT'], **inspected.network_kwargs())
        net.apply_to(None, model, False, True)
        load_strict(net, inspected)
        net.force_to('cpu', dtype=torch.float32)
        net.multiplier = 1
        net._update_torch_multiplier()
        net.is_active = True
        return net

    def test_mixed_ranks_scaling_targets_and_reload(self):
        model = SingleStreamDiT(tiny_config())
        source = self.source(model, alpha=True)
        inspected = inspect_lora(self.path)
        validate_targets(inspected, model)
        original = copy.deepcopy(model)
        base = holder()
        net = self.network(model, base, inspected)
        for spec in inspected.modules:
            name = spec['name']
            layer, plain = dict(model.named_modules())[name], dict(original.named_modules())[name]
            x = torch.randn(2, spec['input'])
            delta = 2 * source[f'diffusion_model.{name}.lora_B.weight'] @ source[f'diffusion_model.{name}.lora_A.weight']
            torch.testing.assert_close(layer(x), plain(x) + x @ delta.T, rtol=1e-5, atol=1e-6)
        exported = Path(self.tmp.name) / 'export.safetensors'
        save_file(net.get_state_dict(dtype=torch.float32), str(exported))
        reloaded = inspect_lora(exported)
        for key, value in inspected.state.items():
            torch.testing.assert_close(value, reloaded.state[key])

    def test_teacher_gradient_checkpointing_and_exception_recovery(self):
        model = SingleStreamDiT(tiny_config()).requires_grad_(False)
        self.source(model)
        inspected = inspect_lora(self.path)
        base = holder()
        base.model = model
        student = self.network(model, base, inspected)
        process = SimpleNamespace(sd=base, network_config=NetworkConfig(type='lora'), network=student, device_torch='cpu')
        teacher = build_source_teacher(process, inspected)
        before = {key: value.clone() for key, value in teacher.state_dict().items()}
        base_before = {key: value.clone() for key, value in model.state_dict().items()}
        layer = model.blocks[0].attn.wq
        x = torch.randn(2, layer.in_features)
        with source_teacher_context(student, teacher):
            target = layer(x).clone()
        torch.testing.assert_close(layer(x), target)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with source_teacher_context(student, teacher):
                raise RuntimeError('test')
        self.assertTrue(student.is_active)
        self.assertFalse(teacher.is_active)
        optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
        from torch.utils.checkpoint import checkpoint
        prediction = checkpoint(layer, x, use_reentrant=False)
        (prediction.square().mean() + preservation_loss(prediction, target, 'replay_teacher', 1)).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        optimizer.step()
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, base_before[key], rtol=0, atol=0)
        with source_teacher_context(student, teacher):
            torch.testing.assert_close(layer(x), target, rtol=0, atol=0)

    def test_rejects_incomplete_unknown_nonfinite_and_duplicate(self):
        model = SingleStreamDiT(tiny_config())
        good = self.source(model)
        variants = [dict(good, **{'unexpected.weight': torch.ones(1)}),
                    {key: value for key, value in good.items() if key != next(iter(good))},
                    dict(good, **{next(iter(good)): torch.full_like(next(iter(good.values())), float('nan'))}),
                    dict(good, **{next(iter(good)).replace('diffusion_model.', 'transformer.'): next(iter(good.values()))})]
        for invalid in variants:
            save_file({k: v.clone() for k, v in invalid.items()}, str(self.path))
            with self.assertRaises(ValueError):
                inspect_lora(self.path)

    def test_unknown_scale_and_diffusers_mapping(self):
        weights = {'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(2, 4),
                   'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(4, 2)}
        save_file(weights, str(self.path))
        with self.assertRaisesRegex(ValueError, 'Scaling is unknown'):
            inspect_lora(self.path)
        parsed = inspect_lora(self.path, scale=.5)
        self.assertEqual(parsed.modules[0]['name'], 'blocks.0.attn.wq')
        self.assertEqual(native_module('transformer.txt_in.linear_1'), 'txtmlp.1')
        torch.testing.assert_close(parsed.state['transformer.blocks.0.attn.wq.lora_B.weight'], torch.full((4, 2), .5))
        save_file(weights, str(self.path), metadata={'ss_network_alpha': '1'})
        torch.testing.assert_close(inspect_lora(self.path).state['transformer.blocks.0.attn.wq.lora_B.weight'], torch.full((4, 2), .5))
        save_file(weights, str(self.path), metadata={'use_rslora': 'true'})
        with self.assertRaisesRegex(ValueError, 'Unsupported adapter type'):
            inspect_lora(self.path, scale=1)

    def test_kohya_native_mapping_preserves_text_fusion(self):
        model = SingleStreamDiT(tiny_config())
        self.source(model, alpha=True)
        native = inspect_lora(self.path)
        weights = {}
        for name, pair in [('txtfusion.layerwise_blocks.0.attn.wq', 2), ('txtfusion.refiner_blocks.0.mlp.gate', 3)]:
            layer = dict(model.named_modules())[name]
            prefix = 'lora_unet_' + name.replace('.', '_')
            weights[prefix + '.lora_down.weight'] = torch.randn(pair, layer.in_features)
            weights[prefix + '.lora_up.weight'] = torch.randn(layer.out_features, pair)
            weights[prefix + '.alpha'] = torch.tensor(pair / 2)
        save_file(weights, str(self.path))
        mapped = inspect_lora(self.path)
        validate_targets(mapped, model)
        self.assertEqual({m['source_scale'] for m in mapped.modules}, {.5})
        self.assertEqual(len(mapped.modules), 2)
        self.assertEqual(native_module('lora_unet_txtmlp_3'), 'txtmlp.3')

    def test_preservation_mask_and_start_step(self):
        prediction = torch.tensor([1., 10.], requires_grad=True)
        loss = preservation_loss(prediction, torch.zeros(2), 'teacher', 2., torch.tensor([1., 0.]))
        loss.backward()
        self.assertEqual(loss.item(), 1.)
        torch.testing.assert_close(prediction.grad, torch.tensor([2., 0.]))
        network = NetworkConfig(type='lora', pretrained_lora_path='source.safetensors', continuation={
            'version': 1, 'source_sha256': 'a' * 64, 'training_base': 'raw'})
        train = TrainConfig(start_step=0)
        validate_continuation_config(network, ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw'), train, [DatasetConfig(folder_path='unused')])
        self.assertIsNone(train.start_step)

    def test_snapshot_and_checksum(self):
        self.source(SingleStreamDiT(tiny_config()))
        digest = sha256_file(self.path)
        target, actual = snapshot_file(self.path, Path(self.tmp.name) / 'assets', digest)
        self.assertEqual(actual, digest)
        self.assertEqual(sha256_file(target), digest)
        self.assertEqual(snapshot_file(self.path, Path(self.tmp.name) / 'assets')[0], target)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            inspect_lora(self.path, expected_sha256='0' * 64)
        self.assertEqual(sha256_file(self.path), digest)

    def test_preservation_rejects_missing_caption_conditioning(self):
        network = NetworkConfig(type='lora', pretrained_lora_path='source.safetensors', continuation={
            'version': 1, 'source_sha256': 'a' * 64, 'training_base': 'raw'})
        model = ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw')
        train = TrainConfig(source_preservation={'mode': 'teacher'})
        replay = DatasetConfig(folder_path='preserved', role='source_preservation', caption_dropout_rate=0)
        datasets = [DatasetConfig(folder_path='new'), replay]
        validate_continuation_config(network, model, train, datasets)
        for extension in ('', '.json', None):
            replay.caption_ext = extension
            with self.assertRaisesRegex(ValueError, 'curated .txt captions'):
                validate_continuation_config(network, model, train, datasets)
        replay.caption_ext = '.txt'
        train.unload_text_encoder = True
        with self.assertRaisesRegex(ValueError, 'Cache preservation text embeddings'):
            validate_continuation_config(network, model, train, datasets)
        replay.cache_text_embeddings = True
        validate_continuation_config(network, model, train, datasets)

    def test_continuation_start_snapshots_ordinary_preservation_data_for_all_modes(self):
        from toolkit.lora_continuation import prepare_continuation
        self.source(SingleStreamDiT(tiny_config()))
        root = Path(self.tmp.name)
        data = root / 'preserve'
        data.mkdir()
        (data / 'photo.png').write_bytes(b'image fixture')
        (data / 'photo.txt').write_text('source behavior', encoding='utf-8')
        for mode in ('replay', 'teacher', 'replay_teacher'):
            with self.subTest(mode=mode):
                process = SimpleNamespace(save_root=str(root / mode),
                    network_config=NetworkConfig(type='lora', pretrained_lora_path=str(self.path), continuation={
                        'version': 1, 'source_sha256': sha256_file(self.path), 'training_base': 'raw'}),
                    model_config=ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw'),
                    train_config=TrainConfig(source_preservation=dict(mode=mode, new_batches=3, teacher_weight=1)),
                    dataset_configs=[DatasetConfig(folder_path='new-examples'), DatasetConfig(folder_path=str(data),
                        role='source_preservation', caption_dropout_rate=0, cache_text_embeddings=True)])
                prepare_continuation(process)
                assets = Path(process.save_root) / '.continuation'
                manifest_hash = process._continuation_provenance['preservation_manifests'][0]
                pinned = assets / (manifest_hash + '.json')
                self.assertEqual(sha256_file(pinned), manifest_hash)
                self.assertFalse((data / '.aitk-preservation.json').exists())
                prepare_continuation(process)
                self.assertEqual(process._continuation_provenance['preservation_manifests'], [manifest_hash])
                # Losing both job-owned copies must not turn a resume into a
                # new selection with a different baseline.
                pinned.unlink()
                (assets / 'preservation-0000.json').unlink()
                with self.assertRaisesRegex(ValueError, 'manifest is missing'):
                    prepare_continuation(process)
                self.assertFalse(pinned.exists())

    def test_existing_saved_job_recovers_after_old_missing_manifest_failure(self):
        from toolkit.lora_continuation import prepare_continuation
        self.source(SingleStreamDiT(tiny_config()))
        root = Path(self.tmp.name)
        data = root / 'preserve-existing-job'
        data.mkdir()
        (data / 'photo.png').write_bytes(b'image fixture')
        (data / 'photo.txt').write_text('source behavior', encoding='utf-8')
        original_source = self.path.read_bytes()
        # This is the unchanged config saved by the old UI. There is no new
        # flag, manifest reference, config edit, or replacement job involved.
        saved_config = json.dumps(dict(
            save_root=str(root / 'already-created-job'),
            network=dict(type='lora', pretrained_lora_path=str(self.path), continuation=dict(
                version=1, source_sha256=sha256_file(self.path), training_base='raw')),
            model=dict(arch='krea2', name_or_path='krea/Krea-2-Raw'),
            train=dict(source_preservation=dict(mode='replay', new_batches=3, teacher_weight=1)),
            datasets=[dict(folder_path='new-examples'), dict(folder_path=str(data),
                role='source_preservation', caption_dropout_rate=0, cache_text_embeddings=True)]))

        def load_saved_job():
            config = json.loads(saved_config)
            return SimpleNamespace(save_root=config['save_root'], step_num=0,
                network_config=NetworkConfig(**config['network']), model_config=ModelConfig(**config['model']),
                train_config=TrainConfig(**config['train']), dataset_configs=[DatasetConfig(**item) for item in config['datasets']])

        failed = load_saved_job()
        with patch('toolkit.preservation_manifest.prepare_preservation_manifest',
                   side_effect=FileNotFoundError(str(data / '.aitk-preservation.json'))):
            with self.assertRaises(FileNotFoundError):
                prepare_continuation(failed)
        assets = Path(failed.save_root) / '.continuation'
        runtime_profile = (assets / 'runtime-profile.json').read_bytes()
        pinned_source = Path(failed.network_config.pretrained_lora_path)
        self.assertEqual(pinned_source.read_bytes(), original_source)
        self.assertFalse((assets / 'provenance.json').exists())
        self.path.unlink()  # The failed attempt's pinned source is sufficient.

        restarted = load_saved_job()
        prepare_continuation(restarted)
        self.assertEqual(restarted.save_root, failed.save_root)
        self.assertEqual(restarted.step_num, 0)
        self.assertEqual(Path(restarted.network_config.pretrained_lora_path), pinned_source)
        self.assertEqual(pinned_source.read_bytes(), original_source)
        self.assertEqual((assets / 'runtime-profile.json').read_bytes(), runtime_profile)
        self.assertFalse((data / '.aitk-preservation.json').exists())
        self.assertTrue((assets / 'preservation-0000.json').is_file())
        expected = restarted._continuation_provenance
        # A later process also loads the original saved config, without needing
        # the in-memory manifest fields added by the preceding attempt.
        later = load_saved_job()
        prepare_continuation(later)
        self.assertEqual(later._continuation_provenance, expected)

    def test_curated_resume_can_use_job_snapshot_when_dataset_sidecar_is_absent(self):
        from toolkit.lora_continuation import prepare_continuation
        self.source(SingleStreamDiT(tiny_config()))
        root = Path(self.tmp.name)
        data = root / 'curated'
        data.mkdir()
        image, caption = data / 'photo.png', data / 'photo.txt'
        image.write_bytes(b'image fixture')
        caption.write_text('source behavior', encoding='utf-8')
        manifest = data / '.aitk-preservation.json'
        manifest.write_text(json.dumps(dict(version=1, source_sha256=sha256_file(self.path), examples=[dict(
            image=image.name, caption=caption.name, image_sha256=sha256_file(image), caption_sha256=sha256_file(caption))])), encoding='utf-8')
        process = SimpleNamespace(save_root=str(root / 'job'),
            network_config=NetworkConfig(type='lora', pretrained_lora_path=str(self.path), continuation={
                'version': 1, 'source_sha256': sha256_file(self.path), 'training_base': 'raw'}),
            model_config=ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw'),
            train_config=TrainConfig(source_preservation=dict(mode='replay', new_batches=3, teacher_weight=1)),
            dataset_configs=[DatasetConfig(folder_path='new-examples'), DatasetConfig(folder_path=str(data),
                role='source_preservation', caption_dropout_rate=0, cache_text_embeddings=True)])
        prepare_continuation(process)
        expected = process._continuation_provenance['preservation_manifests']
        manifest.unlink()
        prepare_continuation(process)
        self.assertEqual(process._continuation_provenance['preservation_manifests'], expected)
        caption.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after curation'):
            prepare_continuation(process)

    def test_relocated_base_is_accepted_only_with_the_same_pinned_bytes(self):
        from toolkit.lora_continuation import prepare_continuation
        from toolkit.continuation_diagnostics import pin_base_checkpoint
        self.source(SingleStreamDiT(tiny_config()))
        root = Path(self.tmp.name)
        original, relocated = root / 'base-original.safetensors', root / 'base-relocated.safetensors'
        original.write_bytes(b'base fixture')
        relocated.write_bytes(original.read_bytes())
        process = SimpleNamespace(save_root=str(root / 'job'),
            network_config=NetworkConfig(type='lora', pretrained_lora_path=str(self.path), continuation={
                'version': 1, 'source_sha256': sha256_file(self.path), 'training_base': 'raw'}),
            model_config=ModelConfig(arch='krea2', name_or_path=str(original)),
            train_config=TrainConfig(), dataset_configs=[DatasetConfig(folder_path='new-examples')])
        prepare_continuation(process)
        process.model_config.name_or_path = str(relocated)
        with self.assertRaisesRegex(ValueError, 'pinned checkpoint identity'):
            prepare_continuation(process)
        pin_base_checkpoint(original, process.model_config._continuation_context)
        prepare_continuation(process)
        pin_base_checkpoint(relocated, process.model_config._continuation_context)
        self.assertEqual(process._continuation_provenance['base'], str(original))
        relocated.write_bytes(b'changed base fixture')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            pin_base_checkpoint(relocated, process.model_config._continuation_context)

    def test_resume_rejects_changed_declared_auxiliary_checksum(self):
        from toolkit.lora_continuation import prepare_continuation
        self.source(SingleStreamDiT(tiny_config()))
        root = Path(self.tmp.name)
        auxiliary = root / 'auxiliary.safetensors'
        auxiliary.write_bytes(b'auxiliary fixture')
        process = SimpleNamespace(save_root=str(root / 'job'),
            network_config=NetworkConfig(type='lora', pretrained_lora_path=str(self.path), continuation={
                'version': 1, 'source_sha256': sha256_file(self.path), 'training_base': 'turbo'}),
            model_config=ModelConfig(arch='krea2:turbo', name_or_path='krea/Krea-2-Turbo', assistant_lora_path=str(auxiliary)),
            train_config=TrainConfig(), dataset_configs=[DatasetConfig(folder_path='new-examples')])
        prepare_continuation(process)
        process.network_config.continuation['assistant_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'training-adapter checksum changed'):
            prepare_continuation(process)

    def test_job_owned_source_survives_original_removal_and_rejects_runtime_changes(self):
        from toolkit.lora_continuation import prepare_continuation
        from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
        self.source(SingleStreamDiT(tiny_config()))
        digest = sha256_file(self.path)
        config = dict(version=1, source_sha256=digest, training_base='raw', deployment_base='turbo')
        process = SimpleNamespace(save_root=str(Path(self.tmp.name) / 'job'), step_num=0,
            network_config=NetworkConfig(type='lora', pretrained_lora_path=str(self.path), continuation=config),
            model_config=ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw'),
            train_config=TrainConfig(start_step=0), dataset_configs=[DatasetConfig(folder_path='new-examples')])
        first = prepare_continuation(process)
        pinned = Path(process.network_config.pretrained_lora_path)
        self.assertNotEqual(pinned, self.path)
        self.assertEqual(sha256_file(self.path), digest)
        self.path.unlink()
        self.assertEqual(prepare_continuation(process).sha256, first.sha256)
        BaseSDTrainProcess.load_training_state_from_metadata(process, str(pinned))
        self.assertEqual(process.step_num, 0)
        process.model_config.qtype = 'changed'
        with self.assertRaisesRegex(ValueError, 'precision or offloading'):
            prepare_continuation(process)

    def test_diagnostic_source_attachment_is_reversible_and_base_is_pinned(self):
        from toolkit.continuation_diagnostics import source_probe, pin_base_checkpoint, drift_report
        model = SingleStreamDiT(tiny_config()).requires_grad_(False)
        self.source(model, alpha=True)
        base = holder()
        if hasattr(base, 'model'):
            del base.model  # The diagnostic runs before the first model is installed.
        context = dict(source=str(self.path), sha256=sha256_file(self.path), assets=self.tmp.name)
        before = model.blocks[0].attn.wq.forward
        rng = torch.get_rng_state()
        first = source_probe(base, model, context)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(model.blocks[0].attn.wq.forward, before)
        second = source_probe(base, model, context)
        self.assertEqual(drift_report(first, second)['rmse'], 0)
        pin_base_checkpoint(self.path, context)
        pin_base_checkpoint(self.path, context)
        self.path.write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, 'base checkpoint changed'):
            pin_base_checkpoint(self.path, context)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for quantization/offload validation')
    def test_cuda_shared_teacher_quantization_offload_and_checkpoint_backward(self):
        from toolkit.memory_management import MemoryManager
        from extensions_built_in.diffusion_models.krea2.src.pipeline import predict_velocity
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                model = SingleStreamDiT(tiny_config()).requires_grad_(False)
                self.source(model)
                inspected = inspect_lora(self.path)
                model.to(dtype=torch.bfloat16)
                model.aitk_post_load(qtype='float8' if quantized else None, offload=1.0,
                    device='cpu', quantize_device='cuda:0', dtype=torch.bfloat16)
                base = holder()
                base.model = model
                student = self.network(model, base, inspected)
                student.force_to('cuda:0', dtype=torch.float32)
                student._update_torch_multiplier()
                process = SimpleNamespace(sd=base, network_config=NetworkConfig(type='lora'), network=student, device_torch='cuda:0')
                teacher = build_source_teacher(process, inspected)
                frozen = {key: value.clone() for key, value in teacher.state_dict().items()}
                model.enable_gradient_checkpointing()
                args = (torch.randn(1, 2, 4, 4, device='cuda', dtype=torch.bfloat16), torch.tensor([.5], device='cuda'),
                        torch.randn(1, 3, 64, device='cuda', dtype=torch.bfloat16), torch.ones(1, 3, device='cuda', dtype=torch.bool))
                try:
                    with source_teacher_context(student, teacher):
                        target = predict_velocity(model, *args)
                        self.assertFalse(teacher.training)
                    prediction = predict_velocity(model, *args)
                    torch.testing.assert_close(prediction, target, rtol=0, atol=0)
                    optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
                    (prediction.float().square().mean() + preservation_loss(prediction, target, 'replay_teacher', 1)).backward()
                    self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in student.parameters()))
                    optimizer.step()
                    for key, value in teacher.state_dict().items():
                        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                finally:
                    MemoryManager.detach(model)

    def test_configuration_rejects_edit_and_wrong_base(self):
        network = NetworkConfig(type='lora', pretrained_lora_path='source.safetensors', continuation={
            'version': 1, 'source_sha256': 'a' * 64, 'training_base': 'raw'})
        train = TrainConfig()
        dataset = DatasetConfig(folder_path='unused')
        validate_continuation_config(network, ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Raw'), train, [dataset])
        for model in [ModelConfig(arch='krea2:o_edit', name_or_path='krea/Krea-2-Raw'),
                      ModelConfig(arch='krea2', name_or_path='krea/Krea-2-Turbo')]:
            with self.assertRaises(ValueError):
                validate_continuation_config(network, model, train, [dataset])


if __name__ == '__main__':
    unittest.main()
