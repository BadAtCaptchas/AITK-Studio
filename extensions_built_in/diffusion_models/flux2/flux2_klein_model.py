import os

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3ForCausalLM, Qwen2TokenizerFast
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from toolkit.quantized_cache import quantized_cache_key
from .flux2_model import Flux2Model
from .src.model import Klein9BParams, Klein4BParams


def _qwen3_from_config(config):
    from_config = getattr(Qwen3ForCausalLM, "from_config", None)
    if from_config is not None:
        return from_config(config)
    return Qwen3ForCausalLM._from_config(config)


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False
    flux2_klein_te_subfolder: str = "text_encoder"
    flux2_klein_tokenizer_subfolder: str = "tokenizer"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def _get_qwen_source_candidates(self):
        candidates = []
        model_path = self.model_config.name_or_path
        if model_path is not None:
            candidates.append(
                {
                    "text_encoder_path": model_path,
                    "text_encoder_subfolder": self.flux2_klein_te_subfolder,
                    "tokenizer_path": model_path,
                    "tokenizer_subfolder": self.flux2_klein_tokenizer_subfolder,
                    "label": f"{model_path}/{self.flux2_klein_te_subfolder}",
                    "is_fallback": False,
                }
            )

        candidates.append(
            {
                "text_encoder_path": self.flux2_klein_te_path,
                "text_encoder_subfolder": None,
                "tokenizer_path": self.flux2_klein_te_path,
                "tokenizer_subfolder": None,
                "label": self.flux2_klein_te_path,
                "is_fallback": True,
            }
        )

        deduped = []
        seen = set()
        for candidate in candidates:
            key = (
                candidate["text_encoder_path"],
                candidate["text_encoder_subfolder"],
                candidate["tokenizer_path"],
                candidate["tokenizer_subfolder"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _get_qwen_source_fingerprint_path(self, source):
        path = source["text_encoder_path"]
        subfolder = source["text_encoder_subfolder"]
        if subfolder is not None:
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                return subfolder_path
        return path

    def _get_qwen_cache_key(self, source):
        return quantized_cache_key(
            "flux2_text_encoder",
            {
                "arch": self.arch,
                "base_model_version": self.get_base_model_version(),
                "class": self.__class__.__name__,
                "dtype": str(self.torch_dtype),
                "qtype_te": self.model_config.qtype_te,
                "text_encoder_path": source["text_encoder_path"],
                "text_encoder_subfolder": source["text_encoder_subfolder"],
                "tokenizer_path": source["tokenizer_path"],
                "tokenizer_subfolder": source["tokenizer_subfolder"],
            },
            sources=[self._get_qwen_source_fingerprint_path(source)],
        )

    def _load_qwen_quantized_cache(self, source):
        if not self._can_use_quantized_cache(
            self.model_config.qtype_te, "Qwen3 text encoder"
        ):
            return None

        cache_key, _ = self._get_qwen_cache_key(source)
        cache = self._get_quantized_cache()
        if not cache.has_entry("flux2_text_encoder", cache_key):
            return None

        try:
            self.print_and_status_update(
                f"Loading Qwen3 quantized cache from {source['label']}"
            )
            config_kwargs = {"local_files_only": True}
            if source["text_encoder_subfolder"] is not None:
                config_kwargs["subfolder"] = source["text_encoder_subfolder"]
            config = AutoConfig.from_pretrained(
                source["text_encoder_path"],
                **config_kwargs,
            )
            with init_empty_weights():
                text_encoder = _qwen3_from_config(config)
            cache.load(
                text_encoder,
                "flux2_text_encoder",
                cache_key,
                device=torch.device("cpu"),
            )
            text_encoder._aitk_loaded_from_quantized_cache = True
            text_encoder._aitk_qwen_source = source
            return text_encoder
        except Exception as e:
            self.print_and_status_update(
                f"Failed to load Qwen3 quantized cache, rebuilding: {e}"
            )
            return None

    def _save_qwen_quantized_cache(self, text_encoder, source):
        if not self._can_use_quantized_cache(
            self.model_config.qtype_te, "Qwen3 text encoder"
        ):
            return

        cache_key, key_payload = self._get_qwen_cache_key(source)
        try:
            self.print_and_status_update(
                f"Saving Qwen3 quantized cache for {source['label']}"
            )
            self._get_quantized_cache().save(
                text_encoder,
                "flux2_text_encoder",
                cache_key,
                key_payload,
                extra_metadata={
                    "source_path": source["text_encoder_path"],
                    "source_subfolder": source["text_encoder_subfolder"],
                },
            )
        except Exception as e:
            self.print_and_status_update(f"Failed to save Qwen3 quantized cache: {e}")

    def _load_qwen_text_encoder(self, source, dtype):
        text_encoder_kwargs = {"torch_dtype": dtype}
        if source["text_encoder_subfolder"] is not None:
            text_encoder_kwargs["subfolder"] = source["text_encoder_subfolder"]
        return Qwen3ForCausalLM.from_pretrained(
            source["text_encoder_path"],
            **text_encoder_kwargs,
        )

    def _load_qwen_tokenizer(self, source, local_files_only=False):
        tokenizer_kwargs = {}
        if source["tokenizer_subfolder"] is not None:
            tokenizer_kwargs["subfolder"] = source["tokenizer_subfolder"]
        try:
            return Qwen2TokenizerFast.from_pretrained(
                source["tokenizer_path"],
                local_files_only=local_files_only,
                **tokenizer_kwargs,
            )
        except Exception:
            if not local_files_only:
                raise
            return Qwen2TokenizerFast.from_pretrained(
                source["tokenizer_path"],
                **tokenizer_kwargs,
            )

    def _is_loaded_from_qwen_quantized_cache(self, text_encoder):
        return getattr(text_encoder, "_aitk_loaded_from_quantized_cache", False) is True

    def _finalize_qwen_text_encoder(self, text_encoder, dtype, move_to_device=True):
        if move_to_device and not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        source_candidates = self._get_qwen_source_candidates()
        last_error = None

        for source in source_candidates:
            text_encoder = None
            try:
                if self.model_config.quantize_te:
                    text_encoder = self._load_qwen_quantized_cache(source)

                if text_encoder is None:
                    self.print_and_status_update(f"Loading Qwen3 from {source['label']}")
                    text_encoder = self._load_qwen_text_encoder(source, dtype)

                tokenizer_from_cache = self._is_loaded_from_qwen_quantized_cache(
                    text_encoder
                )
                tokenizer = self._load_qwen_tokenizer(
                    source,
                    local_files_only=tokenizer_from_cache,
                )

                if self.model_config.quantize_te:
                    if not self._is_loaded_from_qwen_quantized_cache(text_encoder):
                        self.print_and_status_update("Quantizing Qwen3")
                        quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                        freeze(text_encoder)
                        self._save_qwen_quantized_cache(text_encoder, source)
                    flush()
                    self._finalize_qwen_text_encoder(
                        text_encoder,
                        dtype,
                        move_to_device=False,
                    )
                else:
                    self._finalize_qwen_text_encoder(text_encoder, dtype)

                return text_encoder, tokenizer
            except Exception as e:
                last_error = e
                if source.get("is_fallback", False):
                    raise
                self.print_and_status_update(
                    "Failed to load official FLUX.2 Klein Qwen3 subfolders "
                    f"from {self.model_config.name_or_path}; falling back to "
                    f"{self.flux2_klein_te_path}: {e}"
                )
                del text_encoder
                flush()

        raise RuntimeError("Failed to load FLUX.2 Klein text encoder") from last_error

    def convert_lora_weights_before_save(self, state_dict):
        return state_dict

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            if key.startswith("diffusion_model."):
                key = key.replace("diffusion_model.", "transformer.", 1)
            new_sd[key] = value
        return new_sd


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
