import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.hf_download_progress import HFDownloadProgressReporter, _TrackedProgressContext


class FakeProgress:
    def __init__(self):
        self.updated = 0

    def update(self, amount=1):
        self.updated += amount


@contextmanager
def fake_progress_context(progress):
    yield progress


class HFDownloadProgressReporterTest(unittest.TestCase):
    def read_progress(self, progress_path):
        with open(progress_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_writes_percent_and_parallel_active_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".hf_download_progress.json")
            reporter = HFDownloadProgressReporter(progress_path)

            first = reporter.start("model-00001.safetensors", 100, 0, "huggingface_hub.http_get")
            second = reporter.start("model-00002.safetensors", 300, 0, "huggingface_hub.http_get")
            reporter.update(first, 50)
            reporter.update(second, 150)
            reporter.write(force=True, status="downloading")

            payload = self.read_progress(progress_path)

            self.assertEqual(payload["status"], "downloading")
            self.assertEqual(payload["activeCount"], 2)
            self.assertEqual(payload["bytesDownloaded"], 200)
            self.assertEqual(payload["bytesTotal"], 400)
            self.assertEqual(payload["percent"], 50.0)
            self.assertEqual(payload["fileName"], "model-00002.safetensors")

    def test_completion_and_failure_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".hf_download_progress.json")
            reporter = HFDownloadProgressReporter(progress_path)

            transfer = reporter.start("done.safetensors", 100, 0, "huggingface_hub.xet_get")
            reporter.update(transfer, 100)
            reporter.finish(transfer)

            completed = self.read_progress(progress_path)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["percent"], 100.0)
            self.assertEqual(completed["fileName"], "done.safetensors")

            failed_transfer = reporter.start("failed.safetensors", 100, 0, "huggingface_hub.http_get")
            reporter.update(failed_transfer, 25)
            reporter.finish(failed_transfer, failed=True, error="network failed")

            failed = self.read_progress(progress_path)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"], "network failed")
            self.assertEqual(failed["fileName"], "failed.safetensors")

    def test_tracked_context_forwards_updates_to_wrapped_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".hf_download_progress.json")
            reporter = HFDownloadProgressReporter(progress_path)
            fake_progress = FakeProgress()

            with _TrackedProgressContext(
                fake_progress_context(fake_progress),
                reporter,
                "tracked.bin",
                10,
                0,
                "huggingface_hub.http_get",
            ) as progress:
                progress.update(4)
                reporter.write(force=True, status="downloading")

            payload = self.read_progress(progress_path)
            self.assertEqual(fake_progress.updated, 4)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["fileName"], "tracked.bin")


if __name__ == "__main__":
    unittest.main()
