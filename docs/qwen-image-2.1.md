# Qwen Image 2.1 in Studio

Qwen Image 2.1 supports text-to-image generation and image editing through the `qwen_image_2` architecture. Studio uses the Comfy-Org weights with ConvRot int8 defaults. The [official model card](https://huggingface.co/Qwen/Qwen-Image-2.1) and [reference repository](https://github.com/QwenLM/Qwen-Image-2.1) describe the model's capabilities.

## Weight sources and precision

The transformer and text encoder support both BF16 and ConvRot int8 repacks. With `quantize: false` or `quantize_te: false`, the corresponding component selects its BF16 source rather than reusing cached int8 weights and dequantizing them. If only the int8 file is cached, Studio downloads the BF16 file; existing cached weights remain intact. With quantization enabled, source selection prefers an available cached variant, ranks ConvRot int8 first for `convrot8`, and applies the requested quantization after loading when needed. Explicit checkpoint file paths keep their existing source semantics.

Text encoder source selection respects `qtype_te` and `model_kwargs.use_comfy_weights`. To load the standard Hugging Face layout directly, use `name_or_path: Qwen/Qwen-Image-2.1` with `model_kwargs.use_comfy_weights: false`. The vision patch is installed before post-load quantization and device placement.

Original Diffusers transformer checkpoints now convert the separate MLP gate and up projections into Studio's fused projection during loading, including when the pair spans different shard files. This preserves the pretrained weights and the module names used by existing Studio adapters. BF16 repacks already use the fused layout.

## References and transparency

Training sample forms, Native Generate, and Live Generate accept up to 10 reference images in order. Use the Up and Remove controls to manage that order. Annotated images or a separate edit mask can be supplied as references alongside the original image; there is no separate mask-painting editor. CLI/YAML samples use an ordered list:

```yaml
ctrl_imgs:
  - /path/to/first.png
  - /path/to/second.png
```

When present, `ctrl_imgs` replaces the legacy `ctrl_img` / `ctrl_img_1` / `ctrl_img_2` / `ctrl_img_3` fields. An empty list explicitly clears references. Legacy configurations continue to work; the first-image alias is counted once. Repeated entries in an explicit list remain distinct references. Each reference is resized once to the 32-pixel grid for both vision encoding and the VAE, including small images.

Qwen training preserves the alpha channel in target and control images. RGB sources receive opaque alpha. This is independent of the output setting. Other models retain their previous RGB behavior. Alpha-aware latent caches and reference embedding caches use new identities, so older opaque caches are not reused. Existing cache files are not deleted. Cached reference embeddings use the training crop; disable `cache_text_embeddings` for randomized control selection, control augmentations, or moving point-of-interest crops. Qwen datasets require `standardize_images: false` because the RGBA VAE supplies its own normalization.

Training targets and reference images now use deterministic VAE posterior means. Latent cache identities include `_posterior_mean_v1`, including when a custom latent-space version is configured, so caches built from sampled posteriors are not reused. Restart training after updating to build the corrected caches in a fresh training process. Existing cache files remain intact, and this change does not invalidate text embedding caches.

Enable **Transparent output (RGBA)** to retain generated alpha. The prompt helper adds the official transparency instructions and enables RGBA output. Save as PNG, WebP, or JXL; JPEG cannot retain alpha. The VAE sees four channels, while the vision encoder sees the reference composited over white. The bundled Comfy bridge also carries the ordered list and alpha output.

## Quality and sampling

The Qwen controls offer a 1024x1024 / 20-step preview and seven native 2K / 40-step aspect-ratio presets. Choosing a preset changes the canvas and steps, keeping guidance unchanged. Studio defaults to guidance 1. Existing saved configurations keep their explicit values; set guidance to 1 manually for existing jobs if desired. Explicit guidance values above 1 remain supported.

Prefix KV caching is enabled for sampling when `causal_condition` is supported. Conditional and negative prompts have separate caches, created fresh for each generation. Training uses the ordinary differentiable path. Disable caching if you prefer to reduce cache memory.

```yaml
model_kwargs:
  rgba: true
  use_kv_cache: true
  rewrite_prompt: false
```

## Optional prompt expansion

**Expand prompts with the official Qwen rewriter** selects `Qwen/Qwen-Image-2.1-PE-T2I` without references and `Qwen/Qwen-Image-2.1-PE-I2I` with references. This is disabled by default. On first use it downloads the separate 9B checkpoint and its system prompt into the configured model root's Hugging Face cache. It runs on CPU to avoid competing with the diffusion model for GPU memory, requires substantial system RAM, and can be slow. Weights are released after rewriting.

The integration follows the task-specific sampling profiles, including the T2I presence penalty. Only the validated final rewritten prompt is used; reasoning text is not logged or saved. The selected output canvas is preserved rather than applying the rewriter's aspect-ratio suggestion. Training captions are unchanged. Live engine cancellation interrupts rewriting between generated tokens. Invalid rewrite output fails the request instead of sending reasoning text to image generation.

## Verification limits

Weight-free checks cover small-reference token alignment with the installed processor, RGBA data loading and saving, cache identities, reference order and limits, guided editing with and without KV caching, cancellation, and the rewriter request/parser integration. Full-size image generation, training, and 9B prompt rewriting still require validation with pretrained weights on the target machine.

Weight-free regression checks also cover timestep precision after dtype conversions, deterministic posterior means, and source-selection policy for BF16 and int8 loading. Tiny real checkpoints verify weight preservation, forward outputs, and training gradients through the original Diffusers loading path. Mocked loading checks verify text encoder policy propagation and vision-patch ordering; they do not load pretrained components. These corrections have not yet been shown to resolve the reported training collapse with the affected configuration and pretrained weights.
