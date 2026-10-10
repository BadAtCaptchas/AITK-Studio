# Iris-3B (Sperid Labs, pixel-space text-to-image)

ai-toolkit integration for [`speridlabs/iris-3b`](https://huggingface.co/speridlabs/iris-3b)
([arXiv:2610.09450](https://arxiv.org/abs/2610.09450)), a 3B-parameter diffusion
transformer that generates directly in pixel space. Apache 2.0.

The architecture is vendored in [src/dit.py](src/dit.py) (from the reference
[speridlabs/iris-3b](https://github.com/speridlabs/iris-3b) repo), the prompt
encoder in [src/text_encoder.py](src/text_encoder.py), and the preview sampler
(the reference FlowDPM-Solver++) in [src/pipeline.py](src/pipeline.py).

## What is different about this model

| Property | What it means | How it's handled |
|---|---|---|
| **Pixel space** | No VAE — the transformer denoises raw RGB (`in_channels=3`, `patch_size=16`) | `FakeVAE` (identity); "latents" are the image in `[-1, 1]`; buckets snap to 16 px |
| **v-prediction, shift 4** | Rectified flow, velocity = `noise - clean`; the network is conditioned on the *shifted* noise level × 1000 | The Iris scheduler matches the reference shifted grid and integer-truncated model conditioning, while noise mixing retains continuous sigmas. Use `train.timestep_type: shift` (the UI default) |
| **Stacked text layers** | 12 Qwen3-VL-4B hidden layers per token, fixed 300-token chat template, pooled by a learned layer-attention adapter inside the DiT | Prompts are always padded to 300 tokens (the trunk attends to pad positions, so the pad count is part of the trained distribution); embeds are stored as `(300, 12*2560)` |
| **Hybrid trunk** | 8 dual-stream MM-DiT blocks + 16 single-stream blocks, GQA 20/5 heads, gated attention, sandwich RMSNorm, shared-bias adaLN | Shared adaLN cores live in `modulation_cores` and are evaluated once per forward |
| **Pixel head** | 4 PiT blocks decode each 16×16 patch back to pixels | Kept in full precision when quantizing |

## Text encoder

Qwen3-VL-4B-Instruct, frozen. Config and tokenizer come from
`Qwen/Qwen3-VL-4B-Instruct`; the weights come from the ComfyUI repack
`Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_bf16.safetensors` (the same file
Krea 2 uses). A copy already under `MODELS_PATH/text_encoders/` is used in
place; otherwise it is downloaded there. Point `model.te_name_or_path` at
another `.safetensors` file, a transformers folder, or a hub repo to override.

## Train it

```yaml
model:
  arch: "iris3b"
  name_or_path: "speridlabs/iris-3b"   # hub repo, or a dir with model.safetensors + config.yaml
  quantize_te: true                     # optional: quantize the 4B text encoder
  # te_name_or_path: "Qwen/Qwen3-VL-4B-Instruct"   # load the vendor repo instead of the repack
train:
  noise_scheduler: "flowmatch"
  timestep_type: "shift"                # reference training schedule (shift 4.0)
  gradient_checkpointing: true
sample:
  guidance_scale: 3.0
  sample_steps: 30                      # DPM-Solver++ order 2; the paper uses 100
```

`model_kwargs`: `flow_shift` (override the checkpoint's shift), `sample_order`
(DPM-Solver++ order, default 2), `cfg_interval` (default `[0, 1]`),
`checkpoint_filename` (file to pick inside a directory / repo),
`model_config` (architecture overrides merged over the checkpoint's `config.yaml`).

Full fine-tunes save as `model.safetensors` + `config.yaml`, the reference
export layout, so `scripts/sample.py --checkpoint <dir>` from the Iris repo
loads them directly. LoRA keys use the `diffusion_model.` prefix.

The `depth/` and `upscaler/` fine-tunes in the hub repo share this architecture
but are not text-to-image models; they load (`name_or_path:
speridlabs/iris-3b/depth/model.safetensors`) but their input conditioning is
not implemented here.

## Studio integration notes

Ported from [upstream Iris support](https://github.com/ostris/ai-toolkit/commit/2fa0bd87e6cd12f4d054346865687937b9f8df63), with a model-specific training scheduler matching the reference grid. Sampling handles x-prediction checkpoints directly, without the training velocity clamp. Hub subfolder paths use forward slashes on Windows. Text embedding cache identities include the encoder source, tokenizer/config source, token length and selected hidden layers. The shared Qwen loader accepts legacy Comfy key names while retaining native checkpoint names.

Weight-free tests cover the timestep grid, a tiny real forward/backward pass, checkpoint round-trip, cache identities, legacy text encoder keys and x-prediction sampling. Full-size pretrained generation and GPU training remain unverified.
