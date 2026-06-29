import hashlib
import json
import math
import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torchvision.models import efficientnet_b1

from toolkit.config_modules import WatermarkConfig


def _torch_load(path: str):
    if path.endswith(".safetensors"):
        return load_file(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _secret_hash(secret: Optional[str]) -> Optional[str]:
    if secret is None:
        return None
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Repeat(nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.repeat(1, *self.sizes)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class SecretEncoder(nn.Module):
    def __init__(self, secret_len: int, base_res: int = 32, resolution: int = 64) -> None:
        super().__init__()
        log_resolution = int(math.log2(resolution))
        log_base = int(math.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            zero_module(nn.Conv2d(4, 4, 3, padding=1)),
        )

    def encode(self, x):
        return self.secret_scaler(x)

    def forward(self, x, c):
        c = self.encode(c)
        c = F.interpolate(c, size=(x.shape[2], x.shape[3]), mode="bilinear")
        return x + c, c


class SecretDecoder(nn.Module):
    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        self.model = efficientnet_b1(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, output_size * 2, bias=True)

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode="bilinear")
        return self.model(x).view(-1, self.output_size, 2)


class MapperNet(nn.Module):
    def __init__(self, input_size: int = 16, output_size: int = 64, std: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bit_embeddings = nn.Embedding(input_size, output_size)
        nn.init.orthogonal_(self.bit_embeddings.weight)
        self.bit_embeddings.weight.data = self.bit_embeddings.weight.data / self.bit_embeddings.weight.data.std(dim=1, keepdim=True)
        self.bit_embeddings.weight.data = self.bit_embeddings.weight.data * std

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pos_idx = torch.arange(self.input_size, device=x.device).long()
        encoded = self.bit_embeddings(pos_idx)
        encoded = encoded * x[:, :, None].clone()
        return encoded.sum(dim=1) / torch.sqrt(torch.tensor(self.input_size, device=x.device, dtype=x.dtype)) + 1.0


def bits_from_secret(secret: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    bits = [int(bit) for bit in secret]
    return torch.tensor(bits, device=device, dtype=dtype).view(1, -1)


def sample_bits(batch_size: int, msg_bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, msg_bits), device=device).to(dtype=dtype)


def bit_accuracy(decoded_logits: torch.Tensor, expected_bits: torch.Tensor) -> float:
    decoded_bits = torch.argmax(decoded_logits, dim=-1).to(expected_bits.device)
    expected = expected_bits.to(device=decoded_bits.device, dtype=decoded_bits.dtype)
    if decoded_bits.shape[0] != expected.shape[0]:
        expected = expected[:1].repeat(decoded_bits.shape[0], 1)
    return (decoded_bits == expected).float().mean().detach().item()


class AuthenLoRACodec(nn.Module):
    def __init__(self, msg_bits: int):
        super().__init__()
        self.encoder = SecretEncoder(secret_len=msg_bits, resolution=64)
        self.decoder = SecretDecoder(output_size=msg_bits)

    def load_codec_state(self, path: str):
        state = _torch_load(path)
        if not isinstance(state, dict):
            raise ValueError("AuthenLoRA codec must be a dict checkpoint")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if "sec_encoder" in state and "sec_decoder" in state:
            self.encoder.load_state_dict(state["sec_encoder"])
            self.decoder.load_state_dict(state["sec_decoder"])
            return
        if "encoder" in state and "decoder" in state:
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])
            return
        if "secret_encoder" in state and "secret_decoder" in state:
            self.encoder.load_state_dict(state["secret_encoder"])
            self.decoder.load_state_dict(state["secret_decoder"])
            return

        flat_prefixes = [
            ("sec_encoder.", "sec_decoder."),
            ("encoder.", "decoder."),
            ("secret_encoder.", "secret_decoder."),
        ]
        for encoder_prefix, decoder_prefix in flat_prefixes:
            encoder_state = {
                key[len(encoder_prefix):]: value
                for key, value in state.items()
                if key.startswith(encoder_prefix)
            }
            decoder_state = {
                key[len(decoder_prefix):]: value
                for key, value in state.items()
                if key.startswith(decoder_prefix)
            }
            if encoder_state and decoder_state:
                self.encoder.load_state_dict(encoder_state)
                self.decoder.load_state_dict(decoder_state)
                return

        raise ValueError(
            "AuthenLoRA codec checkpoint must contain sec_encoder/sec_decoder state dicts"
        )


class AuthenLoRAController(nn.Module):
    def __init__(
        self,
        config: WatermarkConfig,
        device: torch.device,
        dtype: torch.dtype,
        save_root: str,
        run_name: str,
    ):
        super().__init__()
        self.config = config
        self.device_ref = device
        self.dtype_ref = dtype
        self.save_root = save_root
        self.run_name = run_name
        self.codec_hash = _file_sha256(config.codec_path)
        self.mapper = MapperNet(input_size=config.msg_bits, output_size=config.mapper_rank)
        self.codec = AuthenLoRACodec(config.msg_bits)
        self.codec.load_codec_state(config.codec_path)
        self.codec.requires_grad_(False)
        self.codec.eval()
        self.last_bit_accuracy: Optional[float] = None
        self.last_zero_bit_accuracy: Optional[float] = None
        self.last_verification_pass: Optional[bool] = None
        self.current_secret: Optional[str] = config.secret

        latest_mapper = self.find_latest_mapper(save_root, run_name)
        if latest_mapper is not None:
            self.load_mapper(latest_mapper)

        self.to(device=device)

    @staticmethod
    def find_latest_mapper(save_root: str, run_name: str) -> Optional[str]:
        if not os.path.isdir(save_root):
            return None
        suffix = "_authenlora_mapper.safetensors"
        candidates = [
            os.path.join(save_root, name)
            for name in os.listdir(save_root)
            if name.startswith(run_name) and name.endswith(suffix)
        ]
        if not candidates:
            return None
        candidates.sort(key=os.path.getctime)
        return candidates[-1]

    def load_mapper(self, path: str):
        self.mapper.load_state_dict(load_file(path))

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        return self

    def get_public_metadata(self) -> dict:
        return {
            "method": "authenlora",
            "msg_bits": self.config.msg_bits,
            "mapper_rank": self.config.mapper_rank,
            "codec_sha256": self.codec_hash,
            "secret_sha256": _secret_hash(self.current_secret),
            "bake_on_save": self.config.bake_on_save,
        }

    def get_private_metadata(self) -> dict:
        return {
            "method": "authenlora",
            "msg_bits": self.config.msg_bits,
            "mapper_rank": self.config.mapper_rank,
            "codec_path": self.config.codec_path,
            "codec_sha256": self.codec_hash,
            "secret": self.current_secret,
            "secret_sha256": _secret_hash(self.current_secret),
            "verification_threshold": 0.5,
            "last_bit_accuracy": self.last_bit_accuracy,
            "last_zero_bit_accuracy": self.last_zero_bit_accuracy,
            "last_verification_pass": self.last_verification_pass,
        }

    def choose_bits(self, batch_size: int, force_zero: bool = False) -> Tuple[torch.Tensor, Optional[str]]:
        mapper_dtype = self.mapper.bit_embeddings.weight.dtype
        mapper_device = self.mapper.bit_embeddings.weight.device
        if force_zero:
            return torch.zeros((batch_size, self.config.msg_bits), device=mapper_device, dtype=mapper_dtype), "0" * self.config.msg_bits
        if self.config.secret is not None:
            bits = bits_from_secret(self.config.secret, mapper_device, mapper_dtype)
            return bits.repeat(batch_size, 1), self.config.secret
        bits = sample_bits(batch_size, self.config.msg_bits, mapper_device, mapper_dtype)
        secret = "".join(str(int(bit.item())) for bit in bits[0].detach().cpu())
        self.current_secret = secret
        return bits, secret

    def rank_scale_from_bits(self, bits: torch.Tensor) -> torch.Tensor:
        mapper_dtype = self.mapper.bit_embeddings.weight.dtype
        return self.mapper(bits.to(device=self.mapper.bit_embeddings.weight.device, dtype=mapper_dtype))

    def encode_latent_delta(self, latents: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        encoder = self.codec.encoder.to(device=latents.device)
        bits = bits.to(device=latents.device, dtype=torch.float32)
        with torch.no_grad():
            _, delta = encoder(latents.float(), bits.float())
        return delta.to(device=latents.device, dtype=latents.dtype)

    def decode_image_bits(self, images: torch.Tensor, bits: torch.Tensor) -> float:
        decoder = self.codec.decoder.to(device=images.device)
        with torch.no_grad():
            logits = decoder(images.float().clamp(0, 1))
        return bit_accuracy(logits, bits.to(images.device))

    def save_mapper(self, checkpoint_path: str) -> str:
        stem, _ = os.path.splitext(checkpoint_path)
        mapper_path = f"{stem}_authenlora_mapper.safetensors"
        state = {
            key: value.detach().clone().to("cpu")
            for key, value in self.mapper.state_dict().items()
        }
        save_file(state, mapper_path, metadata={
            "format": "pt",
            "aitk_watermark_method": "authenlora",
            "aitk_watermark_msg_bits": str(self.config.msg_bits),
            "aitk_watermark_codec_sha256": self.codec_hash,
        })
        return mapper_path

    def save_private_sidecar(self, checkpoint_path: str) -> str:
        stem, _ = os.path.splitext(checkpoint_path)
        sidecar_path = f"{stem}_authenlora.private.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(self.get_private_metadata(), f, indent=2)
        return sidecar_path

    @contextmanager
    def baked_weights(self, network, rank_scale: torch.Tensor):
        originals = []
        rank_scale = rank_scale.detach().to("cpu")
        try:
            for module in network.get_all_modules():
                if hasattr(module, "lora_up") and hasattr(module.lora_up, "weight"):
                    weight = module.lora_up.weight
                    originals.append((weight, weight.data.clone()))
                    scale = network.match_authenlora_rank_scale(rank_scale, getattr(module, "lora_dim", rank_scale.shape[-1]), weight.device, weight.dtype)
                    scale = scale[:1].squeeze(0)
                    if weight.dim() == 2:
                        weight.data = weight.data * scale.view(1, -1)
                    elif weight.dim() == 4:
                        weight.data = weight.data * scale.view(1, -1, 1, 1)
                    else:
                        weight.data = weight.data * scale.mean()
                elif module.__class__.__name__ == "LokrModule" and hasattr(module, "bake_authenlora_rank_scale"):
                    restore = module.bake_authenlora_rank_scale(network.match_authenlora_rank_scale(rank_scale, module.lora_dim, rank_scale.device, rank_scale.dtype))
                    originals.extend(restore)
            yield
        finally:
            for param, data in reversed(originals):
                param.data = data

    def save_baked_lora(self, network, checkpoint_path: str, metadata: Optional[dict], dtype: torch.dtype) -> Optional[str]:
        if self.current_secret is None:
            return None
        bits = bits_from_secret(self.current_secret, self.mapper.bit_embeddings.weight.device, self.mapper.bit_embeddings.weight.dtype)
        rank_scale = self.rank_scale_from_bits(bits)
        stem, ext = os.path.splitext(checkpoint_path)
        baked_path = f"{stem}_authenlora_baked{ext or '.safetensors'}"
        with self.baked_weights(network, rank_scale):
            network.save_weights(baked_path, dtype=dtype, metadata=metadata)
        return baked_path
