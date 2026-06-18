import types
import unittest

import torch

from toolkit.unloader import FakeTextEncoder, unload_text_encoder


class UnloaderTest(unittest.TestCase):
    def test_unload_text_encoder_handles_pipeline_mllm(self):
        mllm = torch.nn.Linear(1, 1)
        model = types.SimpleNamespace(
            device_torch=torch.device("cpu"),
            torch_dtype=torch.float32,
            text_encoder=[mllm],
            pipeline=types.SimpleNamespace(mllm=mllm),
            mllm=mllm,
        )

        unload_text_encoder(model)

        self.assertEqual(len(model.text_encoder), 1)
        self.assertIsInstance(model.text_encoder[0], FakeTextEncoder)
        self.assertIsInstance(model.pipeline.mllm, FakeTextEncoder)
        self.assertIs(model.mllm, model.pipeline.mllm)


if __name__ == "__main__":
    unittest.main()
