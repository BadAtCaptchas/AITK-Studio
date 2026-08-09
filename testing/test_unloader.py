import types
import unittest

import torch

from toolkit.unloader import FakeTextEncoder, unload_text_encoder


class UnloaderTest(unittest.TestCase):
    def test_unload_single_text_encoder_frees_original_storage(self):
        original = torch.nn.Linear(1, 1)
        model = types.SimpleNamespace(
            device_torch=torch.device("cpu"),
            torch_dtype=torch.float32,
            text_encoder=original,
        )

        unload_text_encoder(model)

        self.assertIsInstance(model.text_encoder, FakeTextEncoder)
        self.assertEqual(original.weight.device.type, "meta")

    def test_unload_text_encoder_skips_none_pipeline_slots(self):
        original = torch.nn.Linear(1, 1)
        model = types.SimpleNamespace(
            device_torch=torch.device("cpu"),
            torch_dtype=torch.float32,
            text_encoder=[original],
            pipeline=types.SimpleNamespace(
                text_encoder=None,
                text_encoder_2=original,
                text_encoder_3=None,
                mllm=None,
            ),
            mllm=None,
        )

        unload_text_encoder(model)

        self.assertIsNone(model.pipeline.text_encoder)
        self.assertIsInstance(model.pipeline.text_encoder_2, FakeTextEncoder)
        self.assertIsNone(model.pipeline.text_encoder_3)
        self.assertIsNone(model.pipeline.mllm)
        self.assertIsNone(model.mllm)
        self.assertEqual(len(model.text_encoder), 1)
        self.assertIs(model.text_encoder[0], model.pipeline.text_encoder_2)
        self.assertEqual(original.weight.device.type, "meta")

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
        self.assertEqual(mllm.weight.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
