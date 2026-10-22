import time
import unittest
from unittest import mock


class BaseCaptionerTest(unittest.TestCase):
    def import_base_captioner(self):
        # Exercise real runtime imports; stale framework stubs hid API drift.
        from extensions_built_in.captioner.BaseCaptioner import (
            BaseCaptioner, is_failed_caption, is_refusal_caption, sanitize_caption_text,
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

    def test_caption_separator_preserves_captions_without_triple_dash(self):
        _BaseCaptioner, _is_failed_caption, _is_refusal_caption, sanitize_caption_text = self.import_base_captioner()

        caption = "A black cat -- not a separator -- on a red chair.\t "
        self.assertEqual(sanitize_caption_text(caption), caption)

    def test_caption_separator_removes_consecutive_runs_and_padding(self):
        _BaseCaptioner, _is_failed_caption, _is_refusal_caption, sanitize_caption_text = self.import_base_captioner()

        self.assertEqual(
            sanitize_caption_text(
                "A black cat \t---\ton a red chair--- ---near a window."
            ),
            "A black cat on a red chair  near a window.",
        )

    def test_caption_separator_handles_redos_payload_in_bounded_time(self):
        _BaseCaptioner, _is_failed_caption, _is_refusal_caption, sanitize_caption_text = self.import_base_captioner()

        payload = "---x" + (" \t" * 8000)
        started_at = time.perf_counter()
        self.assertEqual(sanitize_caption_text(payload), "x")
        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 1.0)

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
