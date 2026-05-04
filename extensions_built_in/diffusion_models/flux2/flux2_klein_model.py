from .flux2_model import Flux2Model
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from toolkit.quantized_cache import quantized_cache_key
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

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

    def _get_qwen_cache_key(self):
        return quantized_cache_key(
            "flux2_text_encoder",
            {
                "arch": self.arch,
                "base_model_version": self.get_base_model_version(),
                "class": self.__class__.__name__,
                "dtype": str(self.torch_dtype),
                "qtype_te": self.model_config.qtype_te,
                "text_encoder_path": self.flux2_klein_te_path,
            },
            sources=[self.flux2_klein_te_path],
        )

    def _load_qwen_quantized_cache(self):
        if not self._can_use_quantized_cache(
            self.model_config.qtype_te, "Qwen3 text encoder"
        ):
            return None

        cache_key, _ = self._get_qwen_cache_key()
        cache = self._get_quantized_cache()
        if not cache.has_entry("flux2_text_encoder", cache_key):
            return None

        try:
            self.print_and_status_update("Loading Qwen3 quantized cache")
            config = AutoConfig.from_pretrained(
                self.flux2_klein_te_path,
                local_files_only=True,
            )
            with init_empty_weights():
                text_encoder = Qwen3ForCausalLM.from_config(config)
            cache.load(
                text_encoder,
                "flux2_text_encoder",
                cache_key,
                device=torch.device("cpu"),
            )
            text_encoder._aitk_loaded_from_quantized_cache = True
            return text_encoder
        except Exception as e:
            self.print_and_status_update(
                f"Failed to load Qwen3 quantized cache, rebuilding: {e}"
            )
            return None

    def _save_qwen_quantized_cache(self, text_encoder):
        if not self._can_use_quantized_cache(
            self.model_config.qtype_te, "Qwen3 text encoder"
        ):
            return

        cache_key, key_payload = self._get_qwen_cache_key()
        try:
            self.print_and_status_update("Saving Qwen3 quantized cache")
            self._get_quantized_cache().save(
                text_encoder,
                "flux2_text_encoder",
                cache_key,
                key_payload,
                extra_metadata={"source_path": self.flux2_klein_te_path},
            )
        except Exception as e:
            self.print_and_status_update(f"Failed to save Qwen3 quantized cache: {e}")

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        text_encoder = None
        if self.model_config.quantize_te:
            text_encoder = self._load_qwen_quantized_cache()

        if text_encoder is None:
            self.print_and_status_update("Loading Qwen3")
            text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
                self.flux2_klein_te_path,
                torch_dtype=dtype,
            )
        if self.model_config.quantize_te:
            if not getattr(text_encoder, "_aitk_loaded_from_quantized_cache", False):
                self.print_and_status_update("Quantizing Qwen3")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                self._save_qwen_quantized_cache(text_encoder)
            flush()
        elif not self.model_config.low_vram:
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

        tokenizer_from_cache = getattr(text_encoder, "_aitk_loaded_from_quantized_cache", False)
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                self.flux2_klein_te_path,
                local_files_only=tokenizer_from_cache,
            )
        except Exception:
            if not tokenizer_from_cache:
                raise
            tokenizer = Qwen2Tokenizer.from_pretrained(self.flux2_klein_te_path)
        return text_encoder, tokenizer


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
