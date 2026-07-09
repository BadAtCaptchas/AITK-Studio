import ast
import inspect
import textwrap
import unittest

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class TrainingOomStateTest(unittest.TestCase):
    def make_process(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.step_num = 17
        process.grad_accumulation_step = 4
        process.is_grad_accumulation_step = False
        process.num_consecutive_oom = 0
        process.current_boundary_index = 8
        process.steps_this_boundary = 12
        return process

    def test_oom_restarts_accumulation_without_advancing_global_step(self):
        process = self.make_process()

        should_abort = process._record_training_oom(boundary_index=3, boundary_steps=5)

        self.assertFalse(should_abort)
        self.assertEqual(process.step_num, 17)
        self.assertEqual(process.grad_accumulation_step, 1)
        self.assertTrue(process.is_grad_accumulation_step)
        self.assertEqual(process.num_consecutive_oom, 1)
        self.assertEqual(process.current_boundary_index, 3)
        self.assertEqual(process.steps_this_boundary, 5)

    def test_third_consecutive_oom_aborts(self):
        process = self.make_process()

        self.assertFalse(process._record_training_oom(0, 0))
        self.assertFalse(process._record_training_oom(0, 0))
        self.assertTrue(process._record_training_oom(0, 0))
        self.assertEqual(
            process.num_consecutive_oom,
            BaseSDTrainProcess.MAX_CONSECUTIVE_TRAINING_OOMS,
        )

    def test_oom_branch_retries_before_step_scheduling(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(BaseSDTrainProcess.run)))
        oom_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "did_oom"
        ]

        retry_branch = next(
            (node for node in oom_branches if node.body and isinstance(node.body[-1], ast.Continue)),
            None,
        )
        self.assertIsNotNone(retry_branch)


if __name__ == "__main__":
    unittest.main()
