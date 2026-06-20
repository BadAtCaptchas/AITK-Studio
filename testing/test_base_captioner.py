import sys
import types
import unittest
from unittest import mock


class BaseCaptionerTest(unittest.TestCase):
    def import_base_captioner(self):
        modules = {}

        torch_module = types.ModuleType("torch")
        torch_module.device = lambda value: value
        torch_module.no_grad = lambda: mock.MagicMock()
        modules["torch"] = torch_module
        modules["torchaudio"] = types.ModuleType("torchaudio")

        jobs_process = types.ModuleType("jobs.process")

        class FakeBaseExtensionProcess:
            pass

        jobs_process.BaseExtensionProcess = FakeBaseExtensionProcess
        modules["jobs.process"] = jobs_process

        exceptions = types.ModuleType("toolkit.exceptions")
        exceptions.JobStopRequested = type("JobStopRequested", (Exception,), {})
        modules["toolkit.exceptions"] = exceptions

        encrypted_dataset = types.ModuleType("toolkit.encrypted_dataset")
        encrypted_dataset.EncryptedDatasetReader = object
        encrypted_dataset.is_encrypted_dataset_path = lambda _path: False
        modules["toolkit.encrypted_dataset"] = encrypted_dataset

        image_io = types.ModuleType("toolkit.image_io")
        image_io.open_static_image = mock.Mock()
        modules["toolkit.image_io"] = image_io

        ideogram_caption = types.ModuleType("toolkit.ideogram_caption")
        for name in [
            "normalize_caption_dict",
            "normalize_element",
            "normalize_style",
            "sanitize_palette",
        ]:
            setattr(ideogram_caption, name, lambda value, *args, **kwargs: value)
        ideogram_caption.MAX_ELEMENT_PALETTE = 5
        ideogram_caption.MAX_IMAGE_PALETTE = 16
        modules["toolkit.ideogram_caption"] = ideogram_caption

        train_tools = types.ModuleType("toolkit.train_tools")
        train_tools.get_torch_dtype = lambda _value: "dtype"
        modules["toolkit.train_tools"] = train_tools

        ui_database = types.ModuleType("toolkit.ui_database")
        ui_database.UIJobStore = object
        modules["toolkit.ui_database"] = ui_database

        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
        modules["tqdm"] = tqdm_module

        sys.modules.pop("extensions_built_in.captioner.BaseCaptioner", None)
        with mock.patch.dict(sys.modules, modules):
            from extensions_built_in.captioner.BaseCaptioner import (
                BaseCaptioner,
                is_failed_caption,
                is_refusal_caption,
                sanitize_caption_text,
            )

        return BaseCaptioner, is_failed_caption, is_refusal_caption, sanitize_caption_text

    def make_captioner(self, caption):
        BaseCaptioner, _is_failed_caption, _is_refusal_caption, _sanitize_caption_text = self.import_base_captioner()
        captioner = object.__new__(BaseCaptioner)
        captioner.file_paths = ["image.png"]
        captioner.caption_success_count = 0
        captioner.caption_failure_count = 0
        captioner.is_ui_captioner = False
        captioner.get_preserved_ideogram_json_caption_for_file = mock.Mock(return_value=None)
        captioner.get_caption_for_file = mock.Mock(return_value=caption)
        captioner.save_caption_for_file = mock.Mock()
        return BaseCaptioner, captioner

    def test_refusal_caption_is_failed(self):
        _BaseCaptioner, is_failed_caption, is_refusal_caption, _sanitize_caption_text = self.import_base_captioner()
        refusals = [
            "I cannot fulfill this request.",
            "Sorry, but I can't help with that.",
            "I'm unable to help with that.",
            "I can\u2019t provide a caption for this image.",
            "I won't be able to caption this image.",
            "Apologies, but I can't caption this image.",
            "I apologize, but I cannot describe this content.",
            "I'm afraid I can't answer that request.",
            "As an AI language model, I cannot assist with that.",
            "This request violates my content policy.",
            "That request is outside my safety guidelines.",
            "Request denied.",
            "I must refuse this request.",
            "It would be inappropriate to provide that caption.",
            "I cannot access the image.",
            "No image was provided.",
            "The image is not accessible.",
            "I don't feel comfortable helping with this.",
        ]

        for refusal in refusals:
            with self.subTest(refusal=refusal):
                self.assertTrue(is_refusal_caption(refusal))
                self.assertTrue(is_failed_caption(refusal))

    def test_negative_wording_in_caption_is_not_failed(self):
        _BaseCaptioner, is_failed_caption, is_refusal_caption, _sanitize_caption_text = self.import_base_captioner()
        caption = "A person cannot reach the top shelf in a bright kitchen."
        self.assertFalse(is_refusal_caption(caption))
        self.assertFalse(is_failed_caption(caption))

    def test_caption_separator_is_sanitized(self):
        _BaseCaptioner, _is_failed_caption, _is_refusal_caption, sanitize_caption_text = self.import_base_captioner()

        self.assertEqual(
            sanitize_caption_text("---\nA black cat---on a red chair.\n---"),
            "A black cat on a red chair.",
        )

    def test_caption_loop_saves_sanitized_caption(self):
        BaseCaptioner, captioner = self.make_captioner("---\nA black cat---on a red chair.\n---")

        BaseCaptioner.run_caption_loop(captioner)

        self.assertEqual(captioner.caption_success_count, 1)
        captioner.save_caption_for_file.assert_called_once_with(
            "image.png",
            "A black cat on a red chair.",
        )

    def test_caption_loop_does_not_save_refusal_caption(self):
        BaseCaptioner, captioner = self.make_captioner("I cannot fulfill this request.")

        with self.assertRaisesRegex(RuntimeError, "no captions were generated"):
            BaseCaptioner.run_caption_loop(captioner)

        self.assertEqual(captioner.caption_success_count, 0)
        self.assertEqual(captioner.caption_failure_count, 1)
        captioner.save_caption_for_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
