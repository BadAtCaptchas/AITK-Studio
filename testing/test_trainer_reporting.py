"""Both UI entry points flush saved/terminal progress and preserve fencing."""

import unittest
from types import SimpleNamespace
from unittest import mock

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.sd_trainer.UITrainer import UITrainer


class TrainerReportingTests(unittest.TestCase):
    def trainer(self, cls):
        trainer = cls.__new__(cls)
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.is_ui_trainer = True
        trainer.is_stopping = False
        trainer.step_num, trainer.last_save_step = 19, 12
        trainer.ui_job_store = SimpleNamespace(update_key=mock.Mock(), update_status=mock.Mock(), close=mock.Mock())
        trainer._init_ui_reporting()
        self.addCleanup(trainer._close_ui_reporting)
        return trainer

    def test_error_drains_rollback_progress_before_closing_store(self):
        for cls in (UITrainer, DiffusionTrainer):
            with self.subTest(cls=cls), mock.patch.object(SDTrainer, 'on_error'):
                trainer = self.trainer(cls)
                trainer.update_step()
                trainer.on_error(RuntimeError('Invalid gradients'))
                trainer.ui_job_store.update_key.assert_called_with('step', 12)
                trainer.ui_job_store.update_status.assert_called_with('error', 'Invalid gradients')
                trainer.ui_job_store.close.assert_called_once()
                self.assertIsNone(trainer._progress_reporter)

    def test_completion_and_save_events_are_flushed(self):
        for cls in (UITrainer, DiffusionTrainer):
            with self.subTest(cls=cls), mock.patch.object(SDTrainer, 'save'), mock.patch.object(SDTrainer, 'done_hook'):
                trainer = self.trainer(cls)
                trainer.maybe_stop = mock.Mock()
                trainer.save(12)
                trainer.ui_job_store.update_key.assert_called_with('step', 12)
                trainer.done_hook()
                trainer.ui_job_store.update_status.assert_called_with('completed', 'Training completed')
                self.assertIsNone(trainer._progress_reporter)

    def test_secondary_rank_has_no_reporting_thread(self):
        trainer = UITrainer.__new__(UITrainer)
        trainer.accelerator = SimpleNamespace(is_main_process=False)
        trainer.ui_job_store = mock.Mock()
        trainer._init_ui_reporting()
        trainer.update_db_key('step', 1)
        trainer.update_status('running', 'Training')
        trainer._close_ui_reporting()
        self.assertIsNone(trainer._progress_reporter)
        trainer.ui_job_store.update_key.assert_not_called()


if __name__ == '__main__':
    unittest.main()
