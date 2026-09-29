import torch
from toolkit.memory_management import MemoryManager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )

    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def to(self, *args, **kwargs):
        return self


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    replacements = {}

    def replace(encoder):
        if encoder is None or isinstance(encoder, FakeTextEncoder):
            return encoder
        identity = id(encoder)
        if identity not in replacements:
            MemoryManager.free(encoder)
            replacements[identity] = FakeTextEncoder(model.device_torch, model.torch_dtype)
        return replacements[identity]

    # The holder owns the encoders even when its pipeline is itself, has no
    # encoder attribute, or was constructed with text_encoder=None.
    encoders = model.text_encoder
    model.text_encoder = (
        [replace(encoder) for encoder in encoders]
        if isinstance(encoders, list) else replace(encoders)
    )
    pipe = getattr(model, "pipeline", None)
    if pipe is not None and pipe is not model:
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder = replace(pipe.text_encoder)
        i = 2
        while hasattr(pipe, f"text_encoder_{i}"):
            name = f"text_encoder_{i}"
            setattr(pipe, name, replace(getattr(pipe, name)))
            i += 1
    # HiDream and other Studio integrations expose the same encoder as mllm.
    for owner in (model, pipe):
        if owner is not None and getattr(owner, "mllm", None) is not None:
            owner.mllm = replace(owner.mllm)

    MemoryManager.release_cached_memory()
