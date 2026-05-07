import unittest

import torch

from toolkit.optimizers.adafactor import Adafactor


class AdafactorOptimizerTest(unittest.TestCase):
    def _build_optimizer_with_param(self):
        param = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = Adafactor(
            [param],
            lr=1e-3,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        return optimizer, param

    def _step_once(self, optimizer, param):
        param.grad = torch.ones_like(param)
        optimizer.step()

    def test_load_state_dict_backfills_missing_beta1(self):
        optimizer, param = self._build_optimizer_with_param()
        state_dict = optimizer.state_dict()
        for group in state_dict["param_groups"]:
            group.pop("beta1", None)

        resumed_optimizer, resumed_param = self._build_optimizer_with_param()

        resumed_optimizer.load_state_dict(state_dict)
        self._step_once(resumed_optimizer, resumed_param)

        self.assertIn("beta1", resumed_optimizer.param_groups[0])
        self.assertIsNone(resumed_optimizer.param_groups[0]["beta1"])

    def test_step_reinitializes_missing_factored_state(self):
        optimizer, param = self._build_optimizer_with_param()
        self._step_once(optimizer, param)
        state_dict = optimizer.state_dict()
        for state in state_dict["state"].values():
            state.pop("exp_avg_sq_row", None)

        resumed_optimizer, resumed_param = self._build_optimizer_with_param()
        resumed_optimizer.load_state_dict(state_dict)
        self._step_once(resumed_optimizer, resumed_param)

        resumed_state = resumed_optimizer.state[resumed_param]
        self.assertIn("exp_avg_sq_row", resumed_state)
        self.assertEqual(tuple(resumed_state["exp_avg_sq_row"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
