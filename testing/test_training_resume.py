import copy
import io
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from toolkit.training_resume import (
    ResumableDataIterator, capture_rng, capture_training_resume, loader_resume_reason,
    preserve_rng, restore_rng, restore_training_resume, training_signature,
)


def randomized_collate(items):
    values = torch.stack([item[0] for item in items])
    return values + torch.rand_like(values) + random.random() + float(np.random.random())


class BucketFixtureDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset_config = SimpleNamespace(batch_load_threads=1, poi=None, caption_ext='txt')
        self.file_list = [SimpleNamespace(path=f'fixture-media-{index}', crop_x=index, crop_y=0) for index in range(4)]
        self.batch_indices = [[2, 0], [3, 1]]
        self.buckets = {'square': SimpleNamespace(file_list_idx=[2, 0, 3, 1])}
        self.epoch_num = 3

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, index):
        return torch.tensor([self.file_list[item].crop_x for item in self.batch_indices[index]])


class TrainingResumeTests(unittest.TestCase):
    def seed(self):
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)

    def process(self):
        parameter = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=0.001, foreach=False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return SimpleNamespace(parameter=parameter, optimizer=optimizer, lr_scheduler=scheduler,
            train_config=SimpleNamespace(steps=16, gradient_accumulation_steps=3),
            accelerator=SimpleNamespace(num_processes=1, step=0, scaler=None),
            phase_manager=SimpleNamespace(enabled=False), step_num=0, start_step=0, epoch_num=1,
            grad_accumulation_step=1, _accumulated_microbatches=0,
            _in_training_step=False, current_boundary_index=0, steps_this_boundary=0,
            _resume_approximate_reasons=[])

    def loader(self, changed=False):
        values = torch.arange(16, dtype=torch.float64).reshape(-1, 1) / 16
        if changed:
            values[0] = -1
        return DataLoader(TensorDataset(values), batch_size=2, shuffle=True,
                          num_workers=0, collate_fn=randomized_collate)

    def advance(self, process, iterator, count):
        batches = []
        for _ in range(count):
            batch = next(iterator)
            batches.append(batch.clone())
            prediction = process.parameter * batch + torch.randn_like(batch) * 0.01
            loss = (prediction - 0.5).square().mean()
            loss.backward()
            process._accumulated_microbatches += 1
            if process._accumulated_microbatches == 3:
                process.parameter.grad.div_(3)
                process.optimizer.step()
                process.optimizer.zero_grad(set_to_none=True)
                process._accumulated_microbatches = 0
            # Scheduler cadence intentionally follows the toolkit's microstep cadence.
            process.lr_scheduler.step()
            process.step_num += 1
            process.grad_accumulation_step += 1
            process.steps_this_boundary += 1
        return batches

    def test_interrupted_run_preserves_rng_order_accumulation_and_next_updates(self):
        self.seed()
        original = self.process()
        iterator = ResumableDataIterator(self.loader())
        original._training_data_iterators = {'train': iterator, 'regularization': None}
        self.advance(original, iterator, 4)
        self.assertEqual(original._accumulated_microbatches, 1)
        state = capture_training_resume(original)
        moments = copy.deepcopy(original.optimizer.state_dict())
        # Exercise the same restricted deserialization used by optimizer checkpoints.
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        expected_batches = self.advance(original, iterator, 4)
        next_epoch = ResumableDataIterator(self.loader())
        expected_batches += self.advance(original, next_epoch, 3)
        expected_parameter = original.parameter.detach().clone()
        expected_rng = capture_rng()

        random.random()
        np.random.random()
        torch.randn(50)
        resumed = self.process()
        resumed.optimizer.load_state_dict(moments)
        resumed_iterator = ResumableDataIterator(self.loader(), state['loaders']['train'])
        resumed._training_data_iterators = {'train': resumed_iterator, 'regularization': None}
        restore_training_resume(resumed, state)
        actual_batches = self.advance(resumed, resumed_iterator, 4)
        resumed_next_epoch = ResumableDataIterator(self.loader())
        actual_batches += self.advance(resumed, resumed_next_epoch, 3)
        for expected, actual in zip(expected_batches, actual_batches):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_parameter, resumed.parameter))
        self.assertEqual(original.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
        self.assertTrue(torch.equal(expected_rng['torch'], torch.get_rng_state()))
        self.assertEqual(expected_rng['python'], random.getstate())
        self.assertEqual(original._accumulated_microbatches, resumed._accumulated_microbatches)

    def test_changed_data_is_not_reported_as_exact(self):
        iterator = ResumableDataIterator(self.loader())
        next(iterator)
        with self.assertRaisesRegex(ValueError, 'Dataset files or configuration changed'):
            ResumableDataIterator(self.loader(changed=True), iterator.state_dict())

    def test_bucket_order_and_crop_geometry_restore_before_remaining_batch(self):
        original = ResumableDataIterator(DataLoader(BucketFixtureDataset(), batch_size=None))
        next(original)
        state = original.state_dict()
        expected = next(original)
        dataset = BucketFixtureDataset()
        dataset.batch_indices.reverse()
        dataset.buckets['square'].file_list_idx.reverse()
        dataset.file_list[3].crop_x = 10
        restored = ResumableDataIterator(DataLoader(dataset, batch_size=None), state)
        self.assertTrue(torch.equal(expected, next(restored)))
        self.assertEqual(dataset.buckets['square'].file_list_idx, [2, 0, 3, 1])
        self.assertEqual(dataset.epoch_num, 3)

    def test_invalid_bucket_snapshot_does_not_partially_mutate_dataset(self):
        dataset = BucketFixtureDataset()
        state = ResumableDataIterator(DataLoader(dataset, batch_size=None)).state_dict()
        state['datasets'][0]['batch_indices'].reverse()
        state['datasets'][0]['crops'][0]['crop_x'] = 9
        state['indices'] = [999]
        with self.assertRaisesRegex(ValueError, 'Invalid saved data order'):
            ResumableDataIterator(DataLoader(dataset, batch_size=None), state)
        self.assertEqual(dataset.batch_indices, [[2, 0], [3, 1]])
        self.assertEqual(dataset.file_list[0].crop_x, 0)

    def test_prefetch_and_manual_phases_have_explicit_approximate_capability(self):
        loader = DataLoader(TensorDataset(torch.ones(4, 1)), num_workers=1)
        self.assertIn('prefetch', loader_resume_reason(loader))
        process = self.process()
        process._training_data_iterators = {'train': None}
        process.phase_manager.enabled = True
        state = capture_training_resume(process)
        self.assertEqual(state['mode'], 'approximate')
        self.assertTrue(state['reasons'])
        self.assertNotIn('parameters', state)

    def test_rng_roundtrip_preserves_all_local_generators(self):
        self.seed()
        saved = capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        restore_rng(saved)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], float(np.random.random()))
        self.assertTrue(torch.equal(expected[2], torch.rand(4)))

    def test_sampling_restores_random_stream_even_when_it_fails(self):
        self.seed()
        saved = capture_rng()
        @preserve_rng
        def sample():
            random.random()
            np.random.random()
            torch.rand(5)
            raise RuntimeError('sample failed')
        with self.assertRaisesRegex(RuntimeError, 'sample failed'):
            sample()
        self.assertEqual(saved['python'], random.getstate())
        self.assertTrue(torch.equal(saved['torch'], torch.get_rng_state()))
        self.assertEqual(saved['numpy'][1], np.random.get_state()[1].tolist())

    def test_signature_tracks_model_and_regularization_cadence(self):
        process = self.process()
        process.model_config = SimpleNamespace(arch='model_a')
        original = training_signature(process)
        self.assertEqual(original, training_signature(copy.deepcopy(process)))
        process.model_config.arch = 'model_b'
        self.assertNotEqual(original, training_signature(process))
        process.model_config.arch = 'model_a'
        process.save_config = SimpleNamespace(save_every=4)
        self.assertNotEqual(original, training_signature(process))

    def test_saved_training_state_rejects_nonfinite_parameters_before_mutation(self):
        process = self.process()
        process._training_data_iterators = {'train': ResumableDataIterator(self.loader())}
        state = capture_training_resume(process)
        original = process.parameter.detach().clone()
        state['parameters'][0].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Invalid saved training parameters'):
            restore_training_resume(process, state)
        self.assertTrue(torch.equal(original, process.parameter))


if __name__ == '__main__':
    unittest.main()
