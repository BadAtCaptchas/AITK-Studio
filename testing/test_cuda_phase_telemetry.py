import unittest

from toolkit.memory_management.cuda_telemetry import (
    CudaPhaseTelemetry,
    record_cuda_phase,
)


class _FakeCuda:
    def __init__(self):
        self.allocated = 100
        self.reserved = 200
        self.peak_allocated = 100
        self.peak_reserved = 200
        self.reset_calls = 0

    @staticmethod
    def is_available():
        return True

    def memory_allocated(self, _device):
        return self.allocated

    def memory_reserved(self, _device):
        return self.reserved

    def max_memory_allocated(self, _device):
        return self.peak_allocated

    def max_memory_reserved(self, _device):
        return self.peak_reserved

    def reset_peak_memory_stats(self, _device):
        self.reset_calls += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved


class CudaPhaseTelemetryTest(unittest.TestCase):
    def test_phase_records_allocated_reserved_and_peaks(self):
        cuda = _FakeCuda()
        telemetry = CudaPhaseTelemetry("cuda:0", cuda_api=cuda)

        with telemetry.phase("forward"):
            cuda.allocated = 150
            cuda.reserved = 260
            cuda.peak_allocated = 180
            cuda.peak_reserved = 300

        record = telemetry.latest("forward")
        self.assertIsNotNone(record)
        self.assertEqual(record.allocated_start_bytes, 100)
        self.assertEqual(record.allocated_end_bytes, 150)
        self.assertEqual(record.allocated_peak_bytes, 180)
        self.assertEqual(record.reserved_start_bytes, 200)
        self.assertEqual(record.reserved_end_bytes, 260)
        self.assertEqual(record.reserved_peak_bytes, 300)
        self.assertEqual(cuda.reset_calls, 1)

    def test_repeated_phase_keeps_high_water_marks_and_call_count(self):
        cuda = _FakeCuda()
        telemetry = CudaPhaseTelemetry("cuda", cuda_api=cuda)

        with telemetry.phase("backward"):
            cuda.peak_allocated = 500
            cuda.peak_reserved = 600
        cuda.allocated = 120
        cuda.reserved = 220
        with telemetry.phase("backward"):
            cuda.peak_allocated = 400
            cuda.peak_reserved = 450

        summary = telemetry.report()["backward"]
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["allocated_peak_bytes"], 500)
        self.assertEqual(summary["reserved_peak_bytes"], 600)

    def test_cpu_scope_is_a_noop(self):
        cuda = _FakeCuda()
        telemetry = CudaPhaseTelemetry("cpu", cuda_api=cuda)

        with telemetry.phase("optimizer"):
            cuda.allocated = 999

        self.assertEqual(telemetry.report(), {})
        self.assertEqual(telemetry.metrics(), {})
        self.assertEqual(cuda.reset_calls, 0)

    def test_unknown_phase_is_rejected(self):
        telemetry = CudaPhaseTelemetry("cpu", cuda_api=_FakeCuda())
        with self.assertRaisesRegex(ValueError, "Unknown CUDA memory phase"):
            with telemetry.phase("other"):
                pass

    def test_decorator_uses_reporter_and_preserves_result(self):
        cuda = _FakeCuda()

        class Worker:
            def __init__(self):
                self.cuda_phase_telemetry = CudaPhaseTelemetry(
                    "cuda", cuda_api=cuda
                )

            @record_cuda_phase("saving")
            def save(self, value):
                cuda.peak_allocated = 321
                return value + 1

        worker = Worker()
        self.assertEqual(worker.save(4), 5)
        self.assertEqual(
            worker.cuda_phase_telemetry.latest("saving").allocated_peak_bytes,
            321,
        )

    def test_metrics_report_allocated_and_reserved_separately(self):
        cuda = _FakeCuda()
        telemetry = CudaPhaseTelemetry("cuda", cuda_api=cuda)
        with telemetry.phase("sampling"):
            cuda.allocated = 1024**3
            cuda.reserved = 2 * 1024**3
            cuda.peak_allocated = 3 * 1024**3
            cuda.peak_reserved = 4 * 1024**3

        metrics = telemetry.metrics()
        self.assertEqual(metrics["train/vram/sampling/allocated_gb"], 1.0)
        self.assertEqual(metrics["train/vram/sampling/reserved_gb"], 2.0)
        self.assertEqual(metrics["train/vram/sampling/peak_allocated_gb"], 3.0)
        self.assertEqual(metrics["train/vram/sampling/peak_reserved_gb"], 4.0)


if __name__ == "__main__":
    unittest.main()
