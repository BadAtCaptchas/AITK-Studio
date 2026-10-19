# Kromoss adapter training

[Kromoss](https://huggingface.co/lodestones/Kromoss) is a MOSS-Image fine-tune.
Studio reuses its MOSS transformer, Qwen3.5-9B conditioning, flow-matching loss,
and packed FLUX.2 latents. The Kromoss presets select the publisher's checkpoint
and small-decoder VAE. Existing MOSS presets retain their defaults.

## Presets

| Studio choice / `model.arch` | Preview defaults | Training learning rate |
| --- | --- | --- |
| Kromoss / `moss_image:kromoss` | 1024px, 50 Euler steps, CFG 3, shift 5 | `1e-4` |
| Kromoss Turbo / `moss_image:kromoss_turbo` | 1024px, 8 Euler steps, CFG 1, shift 5 | `1e-5` |

Start with the [full example](../config/examples/train_lora_kromoss_16gb.yaml)
or [turbo example](../config/examples/train_lora_kromoss_turbo_16gb.yaml).
Replace the dataset and training-root placeholders with your configured locations.
Choose `network.type: lora` or `lokr`. Full transformer fine-tuning, text encoder
training, and edit conditioning are outside these presets' supported scope.

The 16 GB profile uses BF16 compute, ConvRot8 transformer and text encoder
weights, batch size 1, LoRA rank/alpha 16, gradient checkpointing, cached latents
and text embeddings, and an unloaded text encoder after caching. Legacy layer
offloading is enabled at 100% for both components and kept active for previews.
LoKr retains Studio's existing factorization defaults, including
`lokr_full_rank: false` and `lokr_factor: -1`. Finite-rank LoKr is required with
ConvRot8. The VAE runs in FP32.
These starting learning rates are not a quality-tuned recipe.

Use fixed captions and image transforms when caching text embeddings. The
examples disable caption dropout, token shuffle/dropout, and random image
augmentation. The full preset excludes transformer layer 22 from the user
adapter because the negative sampling branch skips that layer. Turbo uses only
the conditional branch and trains layer 22 normally.

## Checkpoints and precision

`model.name_or_path: lodestones/Kromoss` resolves Kromoss's own
`diffusion_models/kromoss_int8_convrot.safetensors` for ConvRot8, or its BF16
file for full-precision loading. It never substitutes MOSS Base or SFT weights.
Local files, complete local Diffusers directories, and vendor shard loading
remain available. Files use the configured `MODELS_PATH`; offline operation
requires all weights, tokenizer files, and encoder configuration to be cached.

The presets explicitly set:

```yaml
vae_path: lodestones/Kromoss/vae/flux2-vae_small.safetensors
vae_dtype: fp32
```

The small decoder differs from the full FLUX.2 decoder used by existing MOSS
defaults. Initial noise and Euler accumulation also remain FP32.

## Turbo training and export

Turbo adds this model option:

```yaml
model_kwargs:
  sample_shift: 5
  uncond_skip_layers: []
  distillation_lora_path: lodestones/Kromoss/8-steps-dmd-lora.safetensors
```

The supplied distillation adapter stays active at strength 1 during training,
previews, and generation. Its frozen FP32 weights are applied separately from
the transformer and the trainable user adapter. They are never merged into BF16
weights or included in the optimizer or exported adapter. This preserves the
publisher's FP32 adapter arithmetic, including independent q/k/v projections.
The auxiliary `assistant_lora_path`, `base_lora_path`, and
`inference_lora_path` options cannot be combined with this mode. Generation
can load your exported adapter through the library or `model.lora_path`;
its sample multiplier changes only that adapter, leaving distillation at 1.

Direct turbo training is experimental. A successful training step and export
round trip establish that the integration works; they do not establish
convergence or preservation of distilled image quality.

Exported user adapters retain the MOSS `diffusion_model.*` layout. LoRA exports
split fused QKV projections into q/k/v. LoKr exports retain fused QKV tensors and
their alpha, so use a loader that supports the fused MOSS checkpoint layout.
Compatibility with arbitrary split-QKV LoKr loaders is not claimed.

To use a turbo-trained adapter, load **Kromoss + the original eight-step
distillation LoRA at strength 1 + your exported adapter**, with 8 steps and CFG 1.
The exported metadata records the required distillation path/strength and base
model source. Selecting the Turbo preset restores that dependency in Studio;
metadata alone does not cause external applications to download or apply it.
For a resumed training job, retain the same model and distillation settings.

## Verification

Focused tests exercise checkpoint selection, malformed adapter rejection,
precision, gradient checkpointing, frozen-adapter arithmetic, LoRA/LoKr
export/reload, generation defaults, and pooled full/Turbo switching.

The real-checkpoint smoke harness creates synthetic images in a new `.tmp`
directory. It runs each full/Turbo and LoRA/LoKr combination, saves previews and
an adapter after one optimizer step, then resumes in a fresh process for a
second step. It records process memory, GPU allocation, logs, and export hashes.
Download the components first; the harness runs offline:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_kromoss tests.test_kromoss_distillation tests.test_kromoss_examples tests.test_model_preset_registry tests.test_flux3_moss -v
.\.venv\Scripts\python.exe testing/kromoss_smoke.py
# Also check fresh generation from the exported Turbo adapters:
.\.venv\Scripts\python.exe testing/kromoss_smoke.py --generation-from .tmp/<smoke-directory> --variants turbo_lora turbo_lokr
cd ui
node --test scripts/kromossOptions.test.mjs
npm.cmd run typecheck
```

A 16 GB profile is only a measured result for the tested resolution, captions,
and adapter settings; longer prompts and larger batches can require more memory.

### Validated checkpoint revisions

The October 19, 2026 checks use PyTorch 2.10.0+cu130 on an RTX 5080 (16 GB):

| Component | Hugging Face revision |
| --- | --- |
| Kromoss transformer, small VAE, distillation adapter, tokenizer | `722bb93f5cfe827fce124ae4d7c53eb8940e84db` |
| Comfy-Org/MOSS-Image ConvRot8 Qwen3.5-9B weights | `3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb` |
| Qwen/Qwen3.5-9B configuration | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

All four downloaded weight files were checked against the Hub's SHA-256 values.
Validation runs use only synthetic circles and captions, isolated output folders,
and offline loading after download.

### RTX 5080 results

All four combinations completed one optimizer step, exported a finite adapter,
then restored both adapter and optimizer in fresh processes for steps two and
three. The third step verified the final memory-residency changes. Each produced
four nonconstant 1024px previews: baseline and each training step. Gradients were
finite and nonzero; exported weights changed after each resume.
LoRA exports contain only the rank-16 user weights; LoKr exports retain fused
QKV alpha. Turbo exports contain the required distillation dependency metadata.

| Preset / adapter | Optimizer step 1 / 2 / 3 (seconds) | Peak GPU allocated / reserved (GiB) | Peak process-tree RAM (GiB) |
| --- | --- | --- | --- |
| Kromoss / LoRA | 4.53 / 4.50 / 4.27 | 14.57 / 15.14 | 14.84 |
| Kromoss / LoKr | 4.38 / 4.24 / 4.21 | 14.45 / 15.02 | 14.92 |
| Turbo / LoRA | 5.07 / 4.93 / 5.09 | 14.32 / 14.43 | 16.33 |
| Turbo / LoKr | 4.84 / 4.88 / 4.92 | 14.19 / 14.21 | 16.07 |

Memory peaks cover loading, caching, training, and previews across all three
processes; optimizer timings exclude those other phases. These are bounded
integration checks, not throughput benchmarks or an image-quality evaluation.
Full previews used 50 steps/CFG 3; Turbo previews used 8 steps/CFG 1.

Fresh-process standalone generation also passed with both exported Turbo
adapters at user strengths 0 and 1, using the same prompt, seed, and 8-step/CFG 1
settings. Strength 0 produced identical pixels for LoRA and LoKr, while strength
1 changed each output. The fixed distillation adapter remained active in both
cases. Peak GPU allocation/reservation was 9.05/12.04 GiB for LoRA and
8.99/12.00 GiB for LoKr; peak process-tree RAM was 17.91 and 19.65 GiB.
Phase transitions park the text encoder before activating the transformer.

Offline regressions also passed for projection arithmetic and input gradients,
BF16 autocast, checkpointing, ConvRot8/offload lifecycle, frozen base and DMD
tensors, non-default LoKr alpha, export/reload predictions, pooled model
switching, and positive-only eight-step sampling. Python runtime imports,
existing MOSS/LoKr tests, UI tests, TypeScript checks, targeted lint, and diff
whitespace checks passed.

Generation memory defaults and saved-adapter selection are covered by behavior
tests in `ui/scripts/generationConfig.test.mjs`, replacing two outdated Python
source-text assertions. The tests cover preset defaults, preservation of omitted
settings on same-architecture selection, explicit saved overrides, and reset to
the target defaults when switching architectures.
