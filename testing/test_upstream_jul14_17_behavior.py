import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest import mock

import torch

from extensions_built_in.captioner.Qwen3VLCaptioner import ThinkingBudgetCriteria
from toolkit.config_modules import SampleConfig
from toolkit.optimizers.automagic3 import Automagic3
from toolkit.ui_database import UIJobStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeMongoJobs:
    def __init__(self):
        self.document = {"id": "job-1", "sample_now": True}
        self.lock = threading.Lock()

    def find_one_and_update(self, query, update, projection=None):
        del projection
        with self.lock:
            if not all(self.document.get(key) == value for key, value in query.items()):
                return None
            previous = dict(self.document)
            self.document.update(update["$set"])
            return previous


class UpstreamJuly1417BehaviorTest(unittest.TestCase):
    def test_thinking_budget_reserves_visible_answer_tokens(self):
        criteria = ThinkingBudgetCriteria(think_end_token_id=9, max_new_tokens=2)
        self.assertFalse(criteria(torch.tensor([[1, 9]]), None))
        self.assertFalse(criteria(torch.tensor([[1, 9, 2]]), None))
        self.assertTrue(criteria(torch.tensor([[1, 9, 2, 3]]), None))

    def test_sample_start_step_defaults_and_schedule_guard(self):
        self.assertEqual(SampleConfig().sample_start_step, 0)
        self.assertEqual(SampleConfig(sample_start_step=75).sample_start_step, 75)
        source = (PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.step_num >= self.sample_config.sample_start_step", source)

    def test_sqlite_sample_request_is_consumed_once_across_concurrent_readers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "jobs.db"
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute(
                    "CREATE TABLE Job (id TEXT PRIMARY KEY, sample_now INTEGER NOT NULL, updated_at TEXT)"
                )
                connection.execute(
                    "INSERT INTO Job (id, sample_now, updated_at) VALUES (?, 1, CURRENT_TIMESTAMP)",
                    ("job-1",),
                )
                connection.commit()

            def consume():
                return UIJobStore("job-1", str(db_path)).consume_sample_request()

            with mock.patch.dict(os.environ, {"AITK_DB_PROVIDER": "sqlite"}):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(lambda _: consume(), range(4)))

            self.assertEqual(sum(results), 1)
            with closing(sqlite3.connect(db_path)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT sample_now FROM Job WHERE id = ?", ("job-1",)
                    ).fetchone()[0],
                    0,
                )

    def test_mongodb_sample_request_uses_atomic_find_and_update(self):
        store = object.__new__(UIJobStore)
        store.available = True
        store.provider = "mongodb"
        store.job_id = "job-1"
        store._jobs = FakeMongoJobs()

        self.assertTrue(store.consume_sample_request())
        self.assertFalse(store.consume_sample_request())

    def test_automagic3_honors_and_validates_lr_rails(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = Automagic3(
            [parameter], lr=1e-7, min_lr=1e-5, max_lr=1e-3, fused=False
        )
        group = optimizer.param_groups[0]
        optimizer._init_state(parameter, group)
        self.assertAlmostEqual(float(optimizer.state[parameter]["lr"]), 1e-5)

        with self.assertRaisesRegex(ValueError, "must be <="):
            Automagic3([torch.nn.Parameter(torch.ones(1))], min_lr=1e-2, max_lr=1e-3)

    def test_krea_attention_fallback_order_is_cross_platform(self):
        source = (
            PROJECT_ROOT
            / "extensions_built_in"
            / "diffusion_models"
            / "krea2"
            / "src"
            / "mmdit.py"
        ).read_text(encoding="utf-8")
        backends = [
            "SDPBackend.CUDNN_ATTENTION",
            "SDPBackend.FLASH_ATTENTION",
            "SDPBackend.EFFICIENT_ATTENTION",
            "SDPBackend.MATH",
        ]
        positions = [source.index(backend) for backend in backends]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("set_priority=True", source)

    def test_qwen_vae_checkpoint_patch_is_idempotent(self):
        from toolkit.util import qwen_vae_gradient_checkpointing as patch_module

        patch_module.QwenImageEncoder3d.forward = patch_module._original_encoder_forward
        patch_module.QwenImageDecoder3d.forward = patch_module._original_decoder_forward
        patch_module._patched = False
        patch_module.patch_qwen_vae_gradient_checkpointing()
        encoder_forward = patch_module.QwenImageEncoder3d.forward
        decoder_forward = patch_module.QwenImageDecoder3d.forward
        patch_module.patch_qwen_vae_gradient_checkpointing()

        self.assertIs(encoder_forward, patch_module.QwenImageEncoder3d.forward)
        self.assertIs(decoder_forward, patch_module.QwenImageDecoder3d.forward)
        self.assertTrue(
            patch_module.AutoencoderKLQwenImage._supports_gradient_checkpointing
        )

    def test_caption_quantization_and_adapter_scalars_are_compile_safe(self):
        caption_source = (
            PROJECT_ROOT
            / "extensions_built_in"
            / "captioner"
            / "Qwen3VLCaptioner.py"
        ).read_text(encoding="utf-8")
        adapter_source = (PROJECT_ROOT / "toolkit" / "lora_special.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('exclude=["lm_head", "*.lm_head"]', caption_source)
        self.assertIn("quantize_device=", caption_source)
        self.assertIn("self.scale = float", adapter_source)


if __name__ == "__main__":
    unittest.main()
