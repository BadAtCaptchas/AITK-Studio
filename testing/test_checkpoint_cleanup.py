import errno
import importlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from toolkit import checkpoint_cleanup
from toolkit.checkpoint_cleanup import (
    check_save_permissions,
    load_checkpoint_names,
    remove_old_save,
    save_checkpoint_names,
)
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


train_module = importlib.import_module("jobs.process.BaseSDTrainProcess")


class CheckpointCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aitk-checkpoint-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.warnings = []
        self.sleep = mock.patch.object(checkpoint_cleanup.time, "sleep").start()
        self.addCleanup(mock.patch.stopall)

    def make_process(self, name="My LoRA", keep=10):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.name = "job-test-id"
        process.job = SimpleNamespace(name=name)
        process.save_root = str(self.root)
        process.save_config = SimpleNamespace(max_step_saves_to_keep=keep, dtype="float32")
        process.embed_config = None
        process.network_config = None
        process.sd = SimpleNamespace()
        process.accelerator = SimpleNamespace(is_main_process=True)
        return process

    def checkpoint(self, step, prefix="My LoRA", extension=".safetensors"):
        path = self.root / f"{prefix}_{step:09d}{extension}"
        path.write_bytes(b"checkpoint deletion fixture")
        return path

    def cleanup_with_order(self, process, paths):
        order = {str(path): index for index, path in enumerate(paths)}
        with mock.patch.object(train_module.os.path, "getctime", side_effect=lambda path: order[str(path)]), \
                mock.patch.object(train_module, "print_acc", side_effect=self.warnings.append):
            return process.clean_up_saves()

    def test_retention_keeps_newest_ten_across_legacy_and_named_saves(self):
        process = self.make_process()
        paths = [self.checkpoint(step, "job-test-id" if step <= 5 else "My LoRA") for step in range(1, 12)]
        self.assertEqual(self.cleanup_with_order(process, paths), str(paths[-1]))
        self.assertFalse(paths[0].exists())
        self.assertTrue(all(path.exists() for path in paths[1:]))

    def test_transient_delete_failure_is_retried(self):
        path = self.checkpoint(1)
        remove = os.remove
        calls = []

        def locked_once(target):
            calls.append(target)
            if len(calls) == 1:
                raise PermissionError(errno.EACCES, "locked", target)
            remove(target)

        with mock.patch.object(checkpoint_cleanup.os, "remove", side_effect=locked_once):
            self.assertTrue(remove_old_save(str(path), self.warnings.append))
        self.assertFalse(path.exists())
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.warnings, [])

    def test_locked_old_save_keeps_yaml_and_does_not_delete_a_newer_replacement(self):
        process = self.make_process()
        paths = [self.checkpoint(step) for step in range(1, 13)]
        yaml = paths[0].with_suffix(".yaml")
        yaml.write_text("metadata", encoding="utf-8")
        remove = os.remove

        def locked(target):
            if str(target) == str(paths[0]):
                raise PermissionError(errno.EACCES, "locked", target)
            remove(target)

        with mock.patch.object(checkpoint_cleanup.os, "remove", side_effect=locked):
            self.cleanup_with_order(process, paths)
        self.assertTrue(paths[0].exists())
        self.assertTrue(yaml.exists())
        self.assertFalse(paths[1].exists())
        self.assertTrue(all(path.exists() for path in paths[2:]))
        self.assertTrue(any("Training will continue" in message for message in self.warnings))
        self.assertEqual(self.sleep.call_count, 2)
        self.cleanup_with_order(process, paths)
        self.assertFalse(paths[0].exists())
        self.assertFalse(yaml.exists())

    def test_locked_directory_is_nonfatal_and_retains_sidecar(self):
        process = self.make_process(keep=1)
        old = self.root / "My LoRA_000000001"
        new = self.root / "My LoRA_000000002"
        old.mkdir()
        new.mkdir()
        yaml = old.with_suffix(".yaml")
        yaml.write_text("metadata", encoding="utf-8")
        with mock.patch.object(checkpoint_cleanup.shutil, "rmtree", side_effect=PermissionError("locked")) as remove:
            self.cleanup_with_order(process, [old, new])
        self.assertEqual(remove.call_count, 3)
        self.assertTrue(old.exists() and new.exists() and yaml.exists())

    def test_locked_yaml_does_not_abort_cleanup_of_other_old_saves(self):
        process = self.make_process(keep=1)
        paths = [self.checkpoint(step) for step in range(1, 4)]
        yaml = paths[0].with_suffix(".yaml")
        yaml.write_text("metadata", encoding="utf-8")
        remove = os.remove

        def locked(target):
            if str(target) == str(yaml):
                raise PermissionError(errno.EACCES, "locked", target)
            remove(target)

        with mock.patch.object(checkpoint_cleanup.os, "remove", side_effect=locked):
            self.cleanup_with_order(process, paths)
        self.assertFalse(paths[0].exists() or paths[1].exists())
        self.assertTrue(paths[2].exists() and yaml.exists())

    def test_already_removed_file_is_successful_without_retry(self):
        self.assertTrue(remove_old_save(str(self.root / "gone.safetensors"), self.warnings.append))
        self.sleep.assert_not_called()
        self.assertEqual(self.warnings, [])

    def test_non_main_process_never_removes_saves(self):
        process = self.make_process(keep=1)
        process.accelerator.is_main_process = False
        paths = [self.checkpoint(1), self.checkpoint(2)]
        process.clean_up_saves()
        self.assertTrue(all(path.exists() for path in paths))

    def test_save_continues_to_post_save_hook_after_retention_failure(self):
        process = self.make_process(keep=1)
        old = self.checkpoint(250, "job-test-id")
        new = self.root / "My LoRA_000001250.safetensors"
        process.ema = mock.Mock()
        process.is_fine_tuning = False
        process.train_config = SimpleNamespace(merge_network_on_save=False)
        process.network = mock.Mock(multiplier=1.0)
        process.network.save_weights.side_effect = lambda path, **kwargs: Path(path).write_bytes(b"new save")
        process.named_lora = False
        process.embedding = process.decorator = process.adapter = process.snr_gos = process.optimizer = None
        process.meta = {}
        process.step_num = 1250
        process.update_training_metadata = mock.Mock()
        process.post_save_hook = mock.Mock()
        with mock.patch.object(train_module, "flush"), \
                mock.patch.object(train_module, "print_acc", side_effect=self.warnings.append), \
                mock.patch.object(checkpoint_cleanup.os, "remove", side_effect=PermissionError("locked")), \
                mock.patch.object(train_module.os.path, "getctime", side_effect=lambda path: 0 if path == str(old) else 1):
            process.save(1250)
        self.assertTrue(new.exists() and old.exists())
        process.post_save_hook.assert_called_once_with(str(new))
        process.ema.train.assert_called_once()

    def test_permission_check_does_not_modify_existing_checkpoints(self):
        path = self.checkpoint(250)
        original = path.read_bytes()
        check_save_permissions(str(self.root), ["My LoRA"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_directory_allows_writes_but_denies_deletion_is_caught(self):
        with mock.patch.object(checkpoint_cleanup.os, "remove", side_effect=PermissionError("delete denied")):
            with self.assertRaisesRegex(PermissionError, "Training has not started"):
                check_save_permissions(str(self.root), ["My LoRA"])

    @unittest.skipUnless(os.name == "nt", "Windows read-only attribute")
    def test_read_only_existing_checkpoint_fails_preflight_and_survives_cleanup(self):
        path = self.checkpoint(250)
        os.chmod(path, stat.S_IREAD)
        try:
            with self.assertRaisesRegex(PermissionError, "read-only"):
                check_save_permissions(str(self.root), ["My LoRA"])
            self.assertFalse(remove_old_save(str(path), self.warnings.append))
            self.assertTrue(path.exists())
        finally:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)

    @unittest.skipUnless(os.name == "nt", "Windows sharing permissions")
    def test_existing_open_checkpoint_fails_delete_access_check(self):
        path = self.checkpoint(250)
        with path.open("rb"):
            with self.assertRaisesRegex(PermissionError, "Training has not started"):
                check_save_permissions(str(self.root), ["My LoRA"])
        check_save_permissions(str(self.root), ["My LoRA"])
        self.assertTrue(path.exists())

    @unittest.skipUnless(os.name == "nt", "Windows read-only attribute")
    def test_preflight_checks_nested_directory_checkpoint_files(self):
        folder = self.root / "My LoRA_000000250"
        folder.mkdir()
        path = folder / "weights.safetensors"
        path.write_bytes(b"weights")
        os.chmod(path, stat.S_IREAD)
        try:
            with self.assertRaisesRegex(PermissionError, "weights.safetensors"):
                check_save_permissions(str(self.root), ["My LoRA"])
        finally:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)

    def test_failed_startup_check_prevents_model_loading(self):
        process = self.make_process()
        process.check_checkpoint_access = mock.Mock(side_effect=PermissionError("no delete access"))
        process.hook_before_model_load = mock.Mock()
        with self.assertRaisesRegex(PermissionError, "no delete access"):
            process.run()
        process.hook_before_model_load.assert_not_called()

    def test_posix_sticky_folder_permissions_allow_only_root_or_owners(self):
        path = str(self.root / "checkpoint.safetensors")
        file_info = SimpleNamespace(st_mode=stat.S_IFREG, st_uid=1001)
        parent_info = SimpleNamespace(st_mode=stat.S_IFDIR | stat.S_ISVTX, st_uid=1002)
        with mock.patch.object(checkpoint_cleanup.os, "lstat", return_value=file_info), \
                mock.patch.object(checkpoint_cleanup.os, "stat", return_value=parent_info), \
                mock.patch.object(checkpoint_cleanup.os, "access", return_value=True), \
                mock.patch.object(checkpoint_cleanup.os, "geteuid", create=True) as get_uid:
            for uid in (0, 1001, 1002):
                get_uid.return_value = uid
                checkpoint_cleanup._check_existing_save(path)
            get_uid.return_value = 1003
            with self.assertRaisesRegex(PermissionError, "sticky folder"):
                checkpoint_cleanup._check_existing_save(path)

    def test_missing_user_id_api_does_not_raise_attribute_error(self):
        path = str(self.root / "checkpoint.safetensors")
        file_info = SimpleNamespace(st_mode=stat.S_IFREG, st_uid=1001)
        parent_info = SimpleNamespace(st_mode=stat.S_IFDIR | stat.S_ISVTX, st_uid=1002)
        with mock.patch.object(checkpoint_cleanup.os, "lstat", return_value=file_info), \
                mock.patch.object(checkpoint_cleanup.os, "stat", return_value=parent_info), \
                mock.patch.object(checkpoint_cleanup.os, "access", return_value=True), \
                mock.patch.object(checkpoint_cleanup.os, "geteuid", None, create=True):
            with self.assertRaisesRegex(PermissionError, "sticky folder"):
                checkpoint_cleanup._check_existing_save(path)

    def test_legacy_uuid_checkpoint_and_renamed_lora_both_remain_resumable(self):
        process = self.make_process()
        legacy = self.checkpoint(250, "job-test-id")
        process.check_checkpoint_access()
        self.assertEqual(process.get_latest_save_path(), str(legacy))
        named = self.checkpoint(1250)
        renamed = self.make_process(name="Renamed LoRA")
        renamed.check_checkpoint_access()
        with mock.patch.object(train_module.os.path, "getctime", side_effect=lambda path: 0 if path == str(legacy) else 1):
            self.assertEqual(renamed.get_latest_save_path(), str(named))
        self.assertEqual(renamed.checkpoint_names("Renamed LoRA_LoRA"), [
            "Renamed LoRA_LoRA", "job-test-id_LoRA", "My LoRA_LoRA",
        ])

    def test_name_history_rejects_paths_outside_the_job_folder(self):
        (self.root / ".checkpoint_names.json").write_text(json.dumps(["../other-job"]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint name history"):
            load_checkpoint_names(str(self.root), "My LoRA", "job-test-id")

    def test_rename_does_not_confuse_words_in_the_old_name_with_adapter_suffixes(self):
        original = self.make_process(name="Portrait_LoRA [v1]")
        original.check_checkpoint_access()
        checkpoint = self.checkpoint(250, original.job.name)
        self.checkpoint(500, original.job.name + "_refiner")
        self.checkpoint(500, original.job.name + "_LoRA")
        renamed = self.make_process(name="Portrait v2")
        renamed.check_checkpoint_access()
        self.assertEqual(renamed.get_latest_save_path(), str(checkpoint))
        self.assertEqual(renamed.get_latest_save_path("Portrait v2_LoRA"), str(
            self.root / "Portrait_LoRA [v1]_LoRA_000000500.safetensors",
        ))

    def test_name_history_is_replaced_without_leaving_temporary_files(self):
        save_checkpoint_names(str(self.root), ["My LoRA", "job-test-id"])
        save_checkpoint_names(str(self.root), ["New LoRA", "My LoRA", "job-test-id"])
        self.assertEqual(load_checkpoint_names(str(self.root), "New LoRA", "job-test-id"), [
            "New LoRA", "job-test-id", "My LoRA",
        ])
        self.assertEqual([path.name for path in self.root.iterdir()], [".checkpoint_names.json"])


if __name__ == "__main__":
    unittest.main()
