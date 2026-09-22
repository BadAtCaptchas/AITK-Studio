"""Checkpoint source selection without downloading or modifying model files."""

import os
import unittest
from unittest.mock import patch

from toolkit.models.v2 import resolver


class ComfyWeightResolverTests(unittest.TestCase):
    repo = "test/model"
    int8 = "diffusion_models/model_int8_convrot.safetensors"
    bf16 = "diffusion_models/model_bf16.safetensors"
    fp16 = "diffusion_models/model_fp16.safetensors"

    def resolve(self, candidates, cached, **kwargs):
        models_path = os.path.abspath("test-model-root")
        downloaded = []

        def download(*, repo_id, filename, token, local_dir):
            self.assertEqual(repo_id, self.repo)
            self.assertEqual(local_dir, models_path)
            downloaded.append(filename)
            return os.path.join(local_dir, filename)

        with (
            patch.object(resolver, "MODELS_PATH", models_path),
            patch.object(
                resolver, "resolve_comfy_file",
                side_effect=lambda rel, repo, local_only: cached.get(rel),
            ),
            patch("huggingface_hub.hf_hub_download", side_effect=download),
        ):
            result = resolver.resolve_comfy_candidates(candidates, self.repo, **kwargs)
        return result, downloaded

    def test_full_precision_downloads_bf16_instead_of_dequantizing_cached_int8(self):
        for qtype in (None, ""):
            with self.subTest(qtype=qtype):
                result, downloaded = self.resolve(
                    [self.int8, self.bf16], {self.int8: "cached-int8"}, qtype=qtype,
                )
                self.assertEqual(downloaded, [self.bf16])
                self.assertEqual(os.path.abspath(result), os.path.abspath(os.path.join("test-model-root", self.bf16)))

    def test_full_precision_reuses_bf16_when_both_variants_are_cached(self):
        result, downloaded = self.resolve(
            [self.int8, self.bf16],
            {self.int8: "cached-int8", self.bf16: "cached-bf16"},
        )
        self.assertEqual(result, "cached-bf16")
        self.assertEqual(downloaded, [])

    def test_full_precision_can_reuse_fp16_without_downloading_bf16(self):
        result, downloaded = self.resolve(
            [self.int8, self.bf16, self.fp16],
            {self.int8: "cached-int8", self.fp16: "cached-fp16"},
        )
        self.assertEqual(result, "cached-fp16")
        self.assertEqual(downloaded, [])

    def test_local_only_full_precision_does_not_fall_back_to_cached_int8(self):
        result, downloaded = self.resolve(
            [self.int8, self.bf16], {self.int8: "cached-int8"}, local_only=True,
        )
        self.assertIsNone(result)
        self.assertEqual(downloaded, [])

    def test_full_precision_recognizes_fp32_candidates(self):
        fp32 = "vae/model_fp32.safetensors"
        result, downloaded = self.resolve([self.int8, fp32], {self.int8: "cached-int8"})
        self.assertEqual(downloaded, [fp32])
        self.assertEqual(os.path.abspath(result), os.path.abspath(os.path.join("test-model-root", fp32)))

    def test_mixed_quantized_filenames_with_bf16_hint_are_not_full_precision(self):
        for quantization in ("int8_convrot", "int8", "fp8mixed", "nvfp4", "w4a8"):
            mixed = f"diffusion_models/model_{quantization}_bf16.safetensors"
            with self.subTest(quantization=quantization):
                result, downloaded = self.resolve([mixed, self.bf16], {mixed: "cached-mixed"})
                self.assertEqual(downloaded, [self.bf16])
                self.assertEqual(os.path.abspath(result), os.path.abspath(os.path.join("test-model-root", self.bf16)))

    def test_convrot_request_reuses_matching_quantized_checkpoint(self):
        result, downloaded = self.resolve(
            [self.int8, self.bf16],
            {self.int8: "cached-int8", self.bf16: "cached-bf16"},
            qtype="convrot8",
        )
        self.assertEqual(result, "cached-int8")
        self.assertEqual(downloaded, [])

    def test_convrot_request_reuses_cached_full_precision_for_fresh_quantization(self):
        result, downloaded = self.resolve(
            [self.int8, self.bf16], {self.bf16: "cached-bf16"}, qtype="convrot8",
        )
        self.assertEqual(result, "cached-bf16")
        self.assertEqual(downloaded, [])

    def test_convrot_request_downloads_matching_checkpoint_when_nothing_is_cached(self):
        result, downloaded = self.resolve([self.int8, self.bf16], {}, qtype="convrot8")
        self.assertEqual(downloaded, [self.int8])
        self.assertEqual(os.path.abspath(result), os.path.abspath(os.path.join("test-model-root", self.int8)))

    def test_registries_without_full_precision_candidates_keep_existing_behavior(self):
        for candidate in (self.int8, "diffusion_models/model.safetensors"):
            with self.subTest(candidate=candidate):
                result, downloaded = self.resolve([candidate], {candidate: "cached-model"})
                self.assertEqual(result, "cached-model")
                self.assertEqual(downloaded, [])

    def test_explicit_file_override_remains_authoritative(self):
        with (
            patch.object(resolver.os.path, "exists", return_value=True),
            patch("huggingface_hub.hf_hub_download") as download,
        ):
            result = resolver.resolve_comfy_file(
                self.bf16, self.repo, override_path="explicit-int8.safetensors",
            )
        self.assertEqual(result, "explicit-int8.safetensors")
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
