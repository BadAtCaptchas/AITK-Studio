import unittest

import torch

from toolkit.optimizers.adafactor import Adafactor


class AdafactorOptimizerTest(unittest.TestCase):
    def test_load_state_dict_backfills_missing_beta1(self):
        param = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = Adafactor(
            [param],
            lr=1e-3,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        state_dict = optimizer.state_dict()
        for group in state_dict["param_groups"]:
            group.pop("beta1", None)

        resumed_param = torch.nn.Parameter(torch.ones(2, 2))
        resumed_optimizer = Adafactor(
            [resumed_param],
            lr=1e-3,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

        resumed_optimizer.load_state_dict(state_dict)
        resumed_param.grad = torch.ones_like(resumed_param)
        resumed_optimizer.step()

        self.assertIn("beta1", resumed_optimizer.param_groups[0])
        self.assertIsNone(resumed_optimizer.param_groups[0]["beta1"])


if __name__ == "__main__":
    unittest.main()
