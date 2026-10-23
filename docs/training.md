# Training reference

[← Back to AITK Studio](../README.md#documentation)

Use the [example configurations](../config/examples) as starting points. This guide covers CLI runs, adapters, phases, watermarking, and memory controls. Dataset preparation is covered in the [user guide](user-guide.md#datasets).

- [CLI training](#cli-training)
- [Batch loading threads](#batch-loading-threads)
- [LoRA Watermarking](#lora-watermarking)
- [Multi-step training phases](#multi-step-training-phases)
- [Layer Targeting](#layer-targeting)
- [LoKr Training](#lokr-training)
- [Quantized model cache](#quantized-model-cache)
- [AITK Orbit 4-bit adapter training](#aitk-orbit-4-bit-adapter-training)
- [Layer offloading](#layer-offloading)
- [Gradio UI](#gradio-ui)

## CLI training

1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

For Ideogram 4 starting points, use `config/examples/train_lora_ideogram4_48gb.yaml` for NF4 LoRA, `config/examples/train_lora_ideogram4_fp8_48gb.yaml` for FP8 LoRA, `config/examples/train_lora_ideogram4_nvfp4_48gb.yaml` for Comfy NVFP4 LoRA, or `config/examples/train_full_fine_tune_ideogram4.yaml` for full conditional-transformer fine-tuning. For Comfy NVFP4 Ideogram 4 LoRA, an H200 GPU is suggested; keep the example's `train.batch_size: 1` so the job fits in memory. Ideogram 4 dataset captions work as natural text or JSON objects serialized as text files; JSON is recommended for best prompt fidelity, and training does not call Ideogram magic-prompt, moderation, or any other hosted API.

When training starts, AITK Studio creates the configured training folder and writes checkpoints and samples there. You can stop training with `Ctrl+C`; resume selects the latest valid checkpoint by training step. New saves publish the complete model and optimizer set together, so interrupting a save leaves the previous committed checkpoint available.

Latent cache identities include `standardize_images`. Caches created before this field was tracked rebuild once, because their pixel transform cannot be identified reliably. Existing cache files are retained. Reference-conditioned text caches also track ordinary reference image contents, so replacing a reference at the same path triggers re-encoding.

See [checkpoint recovery and resume](checkpoint-recovery.md) for corrupt-checkpoint fallback, supported exact continuation, approximate resume modes, retention and backup requirements. Preserve the hidden `.aitk-checkpoints` directory when backing up a run for continued training.

## Batch loading threads

Each dataset accepts `batch_load_threads` in YAML. The default `1` (also `0`) loads samples serially. Set it to `2` or higher to decode and resize samples concurrently within a bucketed batch; the worker count is capped at that batch's item count. Unbucketed loading and single-item batches stay serial. Workers shut down after each batch, including on errors, and returned sample order is preserved.

This setting is separate from DataLoader `num_workers`. Start with a small value: additional threads can increase peak memory and change random augmentation ordering. Existing dataset grouping, encrypted-data handling and layered dataset support are unchanged; upstream dataset pooling is not enabled.

## LoRA Watermarking

AITK Studio includes optional AuthenLoRA-style watermarking for image LoRA training. This is useful when you want a LoRA release to carry an ownership or authenticity signal that can be checked later, while keeping the normal LoRA workflow intact. Watermarking is **off by default** and has no effect unless you enable the `watermark` block in a job config or turn on **Watermarking** in the simple-job UI.

AuthenLoRA trains the LoRA behavior and a secret-bit watermark together. A small mapper converts secret bits into rank-wise LoRA modulation, and the trainer adds an auxiliary watermark loss alongside the normal style/content loss. AITK Studio keeps the standard resumable LoRA checkpoint, then saves extra private/public watermark artifacts when watermarking is enabled.

Supported scope:

- Image LoRA jobs only.
- LoRA, LoCon/LyCORIS, and LoKr networks.
- Standard image LoRA loss path. Guided-loss and mean-flow jobs are currently rejected for watermarking.
- Audio/video adapters and non-image training jobs are not supported.

The UI exposes built-in codec choices for the official AuthenLoRA 48-bit, 80-bit, and 100-bit codecs. CLI configs can use those same packaged local defaults with `builtin:` IDs, or point at an explicit local codec checkpoint. AITK Studio does not download codecs automatically at training startup.

Example config:

```yaml
watermark:
  enabled: true
  method: authenlora
  codec_path: builtin:authenlora_48bits # or a local .pth/.safetensors codec path
  msg_bits: 48
  mapper_rank: 160
  mapper_lr: 0.0001
  watermark_loss_weight: 1.0
  style_loss_weight: 1.0
  zero_message_probability: 0.05
  verify_every: 100
  secret: null # optional binary string matching msg_bits
  bake_on_save: false
```

Available built-in codec IDs:

- `builtin:authenlora_48bits`
- `builtin:authenlora_80bits`
- `builtin:authenlora_100bits`

When enabled, saves include the normal resumable LoRA plus an AuthenLoRA mapper safetensors file and a private local sidecar with the secret, codec fingerprint, thresholds, and verification summary. Public safetensors metadata stores only non-secret watermark metadata such as method, bit count, codec hash, and secret hash. If `bake_on_save` is true, AITK Studio also emits a baked public LoRA with the selected/generated secret applied, which is the most portable export for runtimes that do not understand the dynamic mapper.

## Multi-step training phases

Training jobs can split one run into sequential phases with different runtime training settings. This is useful when you want to teach broad structure first, stabilize it, then refine details without rebuilding the model or changing the dataset.

Add `train.phases` to a config. Each phase needs a `name` and `steps`. The top-level `train.steps` value must equal the sum of all phase `steps`. The UI phase editor keeps this synchronized automatically.

```yaml
train:
  steps: 3000
  save_on_phase_change: true
  phases:
    - name: anatomy
      steps: 1200
      optimizer: adamw
      lr: 0.00003
      timestep_type: weighted
      content_or_style: content
      loss_type: mse
      optimizer_params:
        weight_decay: 0.0001

    - name: stabilize
      steps: 1000
      optimizer: adamw
      lr: 0.00001
      timestep_type: weighted
      content_or_style: balanced
      loss_type: mse

    - name: detail
      steps: 800
      optimizer: adamw
      lr: 0.000005
      timestep_type: weighted
      content_or_style: style
      loss_type: mse
```

Phase overrides inherit from the top-level `train` block. Supported phase-local overrides include learning rates, optimizer, optimizer params, LR scheduler params, timestep type/bias, loss type, denoising min/max, SNR settings, and prompt/noise multipliers. Model, network, dataset, save, sample, batch size, gradient accumulation, dtype, cache, LoRA rank, and LoKr rank/factor/advanced settings stay top-level only for the whole run.

At each phase boundary the trainer saves by default, rebuilds the optimizer and LR scheduler, clears gradients, and continues. Phase changes only happen after a completed optimizer update, so boundaries defer cleanly during gradient accumulation. Checkpoints store the current phase index/name and phase-local step so resumes return to the correct phase.

The UI loss graph shows phase boundary markers when phase metrics are present. Training always has a finite step count. Older configs that enable `auto_train`/`auto_learn` or contain phase `auto_advance` settings must remove those settings and provide explicit positive phase steps before they can run.

When EMA is enabled, exported checkpoints contain the averaged weights. Keep the complete checkpoint generation, or the matching `optimizer.pt` for legacy saves, to resume from the live training weights, optimizer moments, and EMA history. Resuming with EMA disabled still restores the live weights and optimizer moments from that state.

## Layer Targeting

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
it will be ignored.

### Optional Krea 2 text-fusion exclusion

For standard **Krea 2 (raw)** and **Krea 2 Turbo (w/ Training Adapter)** LoRA jobs, the
**LoRA target** section has an **Exclude text-fusion layers** checkbox. It is off by
default and is available only with the `lora` network type and edit conditioning disabled.
This experimental character/style option excludes attention and MLP adapters inside text
fusion, plus the text projection MLP, from the LoRA being trained.

The checkbox uses the existing configuration field. To enable it in YAML, add these
entries to your existing `network.network_kwargs.ignore_if_contains` list:

```yaml
network_kwargs:
  ignore_if_contains:
    - "txtfusion."
    - "txtmlp"
```

The checkbox also recognizes the equivalent six-entry list in the
[community training report](https://www.reddit.com/r/StableDiffusion/comments/1wgl3ss/for_the_sake_of_training_krea2/).
Toggling it on normalizes those entries to the compact list above. Toggling it off removes
only the recognized preset entries; other custom exclusions and inclusion filters remain
active. The checkbox represents this preset, so custom filters can still exclude text
fusion independently when it is off. Saved and imported jobs keep their explicit filters;
opening the form does not rewrite them. Model selection retains its existing default-handling
behavior. Use a new training run when changing adapter targets instead of resuming a
checkpoint with a different target set.

With the current architecture and default `transformer_only: true`, this reduces the LoRA
target count from 256 to 224 by excluding 32 text-fusion Linear layers. The text-fusion
projector and `txtmlp` already receive no adapters under that default, but the explicit
exclusions also cover configurations with `transformer_only: false`. All these base-model
layers still participate in the forward pass. Experimental edit presets and the Turbo
assistant adapter are unchanged.

The layer-selection behavior is verified; improvements in training quality or stability
are not established. The [Diffusers Krea 2 guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md#lora-rank-alpha-and-target-modules)
and [Musubi Krea 2 guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md#lora-target-layers--loraの対象レイヤー)
include text-fusion layers in their default target sets and describe narrower attention
targeting for longer runs. Compare matched runs for likeness, style fidelity, prompt
adherence, and instability before treating this option as a preferred default.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md) and the LyCORIS [network arguments](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Network-Args.md). To train a LoKr model, change the network type in the config file:

```yaml
      network:
        type: "lokr"
        linear: 16
        linear_alpha: 16
        lokr_factor: 8
        lokr_full_matrix: false
```

`linear` and `linear_alpha` are the LoKr dimension and alpha. `lokr_factor` maps to upstream AI Toolkit `factor`; use `-1` for automatic factorization. Plain LoKr configs default to upstream-compatible factor ordering. Set `lokr_legacy_factorization: false` only when you intentionally want this fork's newer balanced factor layout. `lokr_full_matrix` forces the second Kronecker block to be stored as a full matrix and is normally left off unless you explicitly want that larger model. The older `lokr_full_rank` key is still accepted for compatibility and also enables full-matrix mode.

Current LoKr options can be set either from the UI (`New Job` -> `Target Type: LoKr`) or in YAML:

```yaml
      network:
        type: "lokr"
        linear: 16
        linear_alpha: 16
        lokr_factor: 8
        lokr_use_tucker: true
        lokr_use_scalar: false
        lokr_decompose_both: false
        lokr_weight_decompose: false
        lokr_bypass_mode: false
        lokr_rs_lora: false
```

Supported advanced keys include `lokr_use_tucker`, `lokr_use_scalar`, `lokr_decompose_both`, `lokr_rank_dropout_scale`, `lokr_weight_decompose`, `lokr_wd_on_output`, `lokr_full_matrix`, `lokr_bypass_mode`, `lokr_rs_lora`, `lokr_unbalanced_factorization`, and `lokr_legacy_factorization`. The loader also accepts upstream-style aliases such as `factor`, `use_tucker`, `use_scalar`, `decompose_both`, `rank_dropout_scale`, `weight_decompose`, `dora_wd`, `wd_on_output`, `full_matrix`, `bypass_mode`, `rs_lora`, `unbalanced_factorization`, and `legacy_factorization`.

Everything else should work the same, including layer targeting with `network_kwargs`.

For scalar LoKr or rank-stabilized LoKr with two full factors, keep the matching `optimizer.pt` alongside the adapter checkpoint to resume the original training parameters and optimizer moments. New portable exports fold scalar and rank scaling into the factors; the optimizer checkpoint stores the original parameters separately and verifies the matching adapter file. If those original parameters are missing for scalar LoKr or a new full-factor RS export, incompatible optimizer moments are skipped with a warning, allowing training to continue from the portable adapter with a fresh optimizer. Older full-factor RS exports without a trainable scalar retain their original factors, so their matching optimizer state can still resume with the original rank and RS configuration.

New rank-stabilized LoKr exports with two full factors preserve their strength in inference loaders that cannot infer rank from those factors. Older exports need to be loaded with their original rank and `lokr_rs_lora` configuration, then re-exported to obtain this portable scaling.

## Quantized model cache

FLUX.2 quantized transformer loads and FLUX.2 Klein quantized Qwen3 text encoder loads cache supported `optimum.quanto` weights and packed Orbit manifests by default. The first quantized load still builds the quantized model, then later runs can reuse the cache instead of quantizing the same component again.

The default cache location is `MODELS_PATH/.aitk_quantized_cache`, which is usually `models/.aitk_quantized_cache` in this repo. The cache is keyed by the source model files, quantization type, dtype, model settings, and package versions so it is rebuilt when the inputs change.

Set `quantize_cache: false` to disable the cache, or set `quantize_cache_dir` to move it:

```yaml
model:
  quantize_cache: true
  quantize_cache_dir: /path/to/cache
```

The cache is used for `optimum.quanto` qtypes such as `qfloat8` and for Orbit qtypes such as `orbit4`. It is skipped for torchao qtypes and for FLUX.2 transformer loads that use an accuracy recovery adapter.

For offline runs, set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`. FLUX.2 Klein Qwen3 quantized cache entries store the text encoder config with the cached weights, so later cache hits do not need to contact Hugging Face. Older cache entries may need one online rebuild if neither the source model config nor the tokenizer is already cached locally.

## AITK Orbit 4-bit adapter training

`orbit4` is AITK Studio's first-party packed-weight backend for LoRA-family
training with a frozen base model. It stores eligible linear weights at roughly
4 bits per parameter, keeps activations and adapter gradients in FP16/BF16, and
propagates gradients through the base without training or retaining a dense base
weight. Full base-model fine-tuning is intentionally rejected.

```yaml
model:
  quantize: true
  qtype: orbit4
  quantize_te: true
  qtype_te: orbit4
  quantize_cache: true
  quantize_kwargs:
    kernel: auto          # auto, triton, or torch
    max_workspace_mb: 64
    include: []
    exclude: []
```

`orbit2`, `orbit3`, and the OrbitVQ formats remain experimental and are hidden from the normal UI selectors.

The `auto` kernel selects the fused Triton path when a compatible installation
is available and otherwise uses the memory-bounded PyTorch path. The fallback is
slower but does not reconstruct a complete dense layer. For 12 GB workflows,
combine Orbit with gradient checkpointing, cached latents and text embeddings,
and the `block` layer-offloading backend. The UI's low-VRAM profile starts with
70% transformer and 50% text-encoder block offload while preserving manual
overrides.

## Layer offloading

Layer offloading can reduce peak VRAM by keeping part of a supported model in CPU RAM. The default backend is `block`, which offloads deterministic whole transformer/text-encoder blocks and prefetches them back to CUDA as needed. The older per-Linear/Conv offloader remains available as `legacy` for fallback cases.

```yaml
model:
  layer_offloading: true
  layer_offloading_backend: block # block or legacy
  layer_offloading_transformer_percent: 0.7
  layer_offloading_text_encoder_percent: 0.5
```

The percentage values are fractions of whole blocks for the block backend. Lower values are faster; higher values save more VRAM. The UI fills conservative defaults for supported large CUDA models, but manually edited backend and percentage values are preserved.

Block offloading is currently CUDA-first and is intended for LoRA/network training plus standalone generation. For supported architectures, full base-model fine-tuning with `layer_offloading_backend: block` is rejected unless the base transformer/text encoder is frozen; use LoRA training or set `layer_offloading_backend: legacy` instead. `low_vram` is still separate: it unloads broader model components and can be combined with memory-saving workflows, but it is generally slower than block offloading alone.

## Gradio UI

To train locally with the legacy Gradio UI after installing AITK Studio:

```bash
cd AITK-Studio # in case you are not yet in the AITK-Studio folder
huggingface-cli login # provide a `write` token to publish your LoRA at the end
python flux_train_ui.py
```

This starts a UI for uploading images, captioning them, training a LoRA, and publishing the result.

![Legacy Gradio training UI](../assets/lora_ease_ui.png)

The opt-in `t0_velocity_equiv_weight` and `do_fft_velocity_equiv_weight` settings
use `1 / max(sigma, 0.01)^2`, computed in float32. This increases weighting below
sigma 0.1 compared with older Studio versions; both settings remain disabled by
default. Studio retains its existing clean-prediction and DFE handling.


### October 23 selective integrations

- **Fibo 2.0 Dev** uses `Comfy-Org/Fibo-2.0` or an explicit Comfy-layout safetensors checkpoint. It supports text-to-image and optional ordered reference images with target-aware sizing. Qwen3-VL conditioning stays frozen and available for text-only CFG negatives; text-encoder training/unloading is rejected. The FLUX.2 VAE is reusable, and decode remains differentiable for DFE losses. Caption `text_render` image entries are not encoded by this integration.
- **Automagic V4** is experimental. Its group controller is isolated from Studio's existing V3 behavior and checkpoints. The first step measures sensitivity without updating weights. Studio uses non-fused steps and ordinary visible PyTorch gradient accumulation to preserve clipping and recovery; `fused: true` is rejected. Factored second moments stay in FP32, including on resume, to prevent FP16 overflow (at an increased moment-state memory cost versus upstream).
- Cached preview prompts now retain frame count as well as dimensions. Compiled model exports remove compiler wrapper keys and reject collisions. MPS memory release skips the unsafe native host-cache call; rotary phases use FP32 on MPS.
