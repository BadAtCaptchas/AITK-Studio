"""Explicit sampler contract for the unregistered example model template."""

import torch


def pad_prompt_embeds(embeds, device, dtype):
    """Pad variable-length (tokens, channels) embeddings and mark real tokens."""
    if not embeds:
        raise ValueError("At least one prompt embedding is required")
    if any(item.ndim != 2 or item.shape[1] != embeds[0].shape[1] for item in embeds):
        raise ValueError("Prompt embeddings must have shape (tokens, channels) with the same channel count")
    length = max(item.shape[0] for item in embeds)
    features = torch.zeros(len(embeds), length, embeds[0].shape[1], device=device, dtype=dtype)
    mask = torch.zeros(len(embeds), length, device=device, dtype=torch.bool)
    for index, item in enumerate(embeds):
        features[index, :item.shape[0]] = item.to(device=device, dtype=dtype)
        mask[index, :item.shape[0]] = True
    return features, mask


class ExamplePipeline:
    def __init__(self, model):
        self.model = model

    def __call__(self, conditional_embeds, unconditional_embeds=None, height=1024, width=1024,
                 num_inference_steps=20, guidance_scale=1.0, latents=None, generator=None):
        raise NotImplementedError(
            "The example model is an unregistered integration template. Implement its sampler "
            "using the model's timestep convention, guidance and VAE before generating images."
        )
