# PRXPixel (Photoroom PRX-7B, Pixel-Space Text-To-Image)

ai-toolkit integration for [`Photoroom/prxpixel-t2i`](https://huggingface.co/Photoroom/prxpixel-t2i),
a ~7B pixel-space diffusion transformer.

The transformer is vendored in [src/transformer_prx.py](src/transformer_prx.py)
and a minimal preview sampler lives in [src/pipeline.py](src/pipeline.py), so
ai-toolkit does not depend on the unreleased Diffusers `PRXPixelPipeline`.

## Model Details

PRXPixel differs from a typical latent flow-matching model in three ways:

| Property | What It Means | Handling |
|---|---|---|
| **Pixel space** | No VAE; the transformer denoises raw RGB (`in_channels=3`, `patch_size=16`) | `FakeVAE` makes encode/decode no-ops; latents are images in `[-1, 1]` |
| **x-prediction** | The model predicts clean image `x0`, not flow velocity | `get_noise_prediction` returns `x0`; `get_loss_target` returns clean latents |
| **noise_scale = 2.0** | Training and sampling use `randn * 2.0`, not unit noise | training noise and preview starting noise are scaled |

Text is encoded by the Qwen3-VL text tower (`Qwen3VLTextModel`, hidden size
2048 -> the transformer's `context_in_dim`) and padded to 256 tokens.

Released checkpoint architecture: `depth=24`, `hidden_size=3584`,
`num_heads=28`, `mlp_ratio=3.5`, `in_channels=3`, `patch_size=16`,
`context_in_dim=2048`, `bottleneck_size=768`, `axes_dim=[64, 64]`,
`resolution_embeds=True`, and flow-matching scheduler `shift=3.0`.

## Train It

```yaml
model:
  arch: "prx_pixel"
  name_or_path: "Photoroom/prxpixel-t2i"
  quantize: true
  quantize_te: true
train:
  gradient_checkpointing: true
  noise_scheduler: "flowmatch"
  timestep_type: "linear"
sample:
  width: 1024
  height: 1024
  guidance_scale: 5.0
  sample_steps: 28
```

Datasets bucket to multiples of 16px (`vae_scale_factor * patch_size`). The
model trains LoRA adapters against `PRXTransformer2DModel` transformer blocks.
