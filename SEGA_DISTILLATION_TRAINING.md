# SEGA Distillation Training

SEGA distillation is an opt-in LoRA training mode for Flux2, Flux2 Klein, Z-Image, and Z-Image Turbo. It is a training-time teacher signal only: saved LoRAs are normal LoRAs and do not need SEGA at inference.

## Supported Jobs

SEGA distillation is enabled when:

- `train.sega_distill` is `true`.
- `network.type` is `lora`.
- `model.arch` resolves to one of:
  - `flux2`
  - `flux2_klein_4b`
  - `flux2_klein_9b`
  - `zimage`

The UI has model variants such as `zimage:turbo`, `zimage:deturbo`, and `zimage:juggernaut_z`. Backend config loading strips the variant suffix, so those run through the supported `zimage` backend. The built-in `Z-Image SEGA Distill` profile is intended for both standard Z-Image and Z-Image Turbo style jobs.

V1 rejects these conflicting target features:

- Differential output preservation
- Blank prompt preservation
- Prior divergence
- Inverted-mask prior
- Guidance loss
- Differential guidance
- `mean_flow` loss

Regularization batches skip SEGA by default. Set `train.sega_distill_on_reg: true` only if you want regularization batches to receive the auxiliary SEGA teacher term too.

## Config Fields

```yaml
train:
  sega_distill: true
  sega_distill_weight: 1.0
  sega_distill_base_resolution: 1024
  sega_distill_strength: 1.0
  sega_distill_min_scale: 0.5
  sega_distill_max_scale: 2.0
  sega_distill_on_reg: false
```

`sega_distill_weight` controls how much the teacher matching term contributes after the normal dataset loss is computed.

`sega_distill_base_resolution` is the identity point. At or below this resolution, SEGA returns identity scaling.

`sega_distill_strength` controls how strongly FFT energy changes RoPE scaling. Set it to `0` for identity scaling.

`sega_distill_min_scale` and `sega_distill_max_scale` clamp the generated scale values.

## Built-In Profiles

`FLUX.2 SEGA Distill` enables SEGA for Flux2/Klein with conservative defaults:

```yaml
sega_distill_weight: 1.0
sega_distill_base_resolution: 1024
sega_distill_strength: 1.0
sega_distill_min_scale: 0.5
sega_distill_max_scale: 2.0
sega_distill_on_reg: false
```

`Z-Image SEGA Distill` uses the same scale defaults, but starts with a gentler auxiliary loss:

```yaml
sega_distill_weight: 0.25
```

Z-Image is more sensitive to the teacher term in early testing, so `0.25` is a safer starting point for concept learning.

## Per-Step Training Flow

For an eligible batch, the trainer prepares the batch normally:

1. Load latents, masks, prompts, timesteps, noise, adapters, controls, and CFG state.
2. Build the current noisy latent.
3. Build the normal model prediction kwargs.
4. Run a frozen SEGA teacher prediction.
5. Run the trainable student prediction.
6. Compute normal dataset loss.
7. Add the weighted SEGA teacher loss.

The teacher pass:

- Temporarily disables the active trainable LoRA network.
- Puts the transformer in eval mode.
- Runs under `torch.no_grad()`.
- Uses detached noisy latents and detached prompt embeddings.
- Uses the same timestep, prompt conditioning, CFG settings, adapters, and controls as the student.
- Enables SEGA RoPE scaling only for this teacher prediction.

The student pass:

- Restores the previous LoRA state.
- Runs with gradients enabled.
- Does not receive SEGA RoPE scaling.
- Learns through the normal LoRA training path.

## Loss

The normal dataset target remains the anchor:

```text
teacher_pred = frozen_base_model(noisy_latent, prompt, timestep, sega_rope)
student_pred = lora_model(noisy_latent, prompt, timestep)

supervised_loss = loss_fn(student_pred, dataset_target)
distill_loss = mse(student_pred, teacher_pred)
loss = supervised_loss + distill_loss * sega_distill_weight
```

For flow-matching models such as Z-Image, the dataset target is still the model's usual flow target. SEGA does not replace that target.

## Scale Construction

SEGA builds scale values from detached fp32 FFT statistics of the current noisy latent:

1. Convert the noisy latent to fp32 without gradients.
2. Compute a radial FFT energy profile.
3. Estimate current pixel resolution from latent size and VAE scale factor.
4. Return identity scaling at or below `sega_distill_base_resolution`.
5. Above the base resolution, use resolution gain and `sega_distill_strength` to move scale values away from one.
6. Give low-energy frequency bands stronger scaling and high-energy bands weaker scaling.
7. Clamp values between `sega_distill_min_scale` and `sega_distill_max_scale`.

Only primary image spatial RoPE axes receive non-identity values. Text, caption, timestep/sequence, reference/control, and padding paths stay identity.

## Flux2 Hook

Flux2/Klein use a local transformer forward path that accepts optional `sega_rope_scale`.

When SEGA is disabled, or when the scale is all ones, the forward pass preserves the previous behavior. When SEGA is active for the teacher pass, the transformer applies the scale to query/key tensors after RoPE has been applied.

## Z-Image Hook

Z-Image uses the Diffusers transformer implementation. During the teacher pass, the wrapper temporarily patches the transformer's `_prepare_sequence` method.

That patch scales only the RoPE tensor created for the image token sequence. Caption/text sequence RoPE calls still return identity behavior. The patch is scoped with a context manager and is removed immediately after the teacher prediction.

Z-Image Turbo uses the same `zimage` backend path. The Turbo training adapter remains part of the normal job setup; SEGA only changes the frozen teacher's spatial RoPE during the teacher prediction.

## UI Behavior

The simple new-job UI shows the `SEGA Distillation` section for supported Flux2/Klein and Z-Image selections. Imported configs with SEGA already enabled still show the section so the setting can be turned off even if the current model state is unsupported.

Enabling SEGA in the simple UI clears conflicting simple-mode target features such as DOP, BPP, and differential guidance. The Training Advisor also emits critical findings for unsupported architectures, non-LoRA targets, and conflicting target modes.

## Metrics

When SEGA runs, the backend records:

- `train/sega_supervised_loss`
- `train/sega_distill_loss`
- `train/sega_distill_weighted_loss`
- `train/sega_scale_mean`
- `train/sega_scale_min`
- `train/sega_scale_max`

The UI shows SEGA loss and scale stats in the existing job metrics/stat area when present.

## Practical Guidance

Start with the built-in profile defaults. If the concept is not learning strongly enough, lower `sega_distill_weight` before changing scale strength. For Z-Image, `0.25` is the safer starting value. For Flux2, `1.0` is the current profile default, but smaller values are reasonable for weak datasets or subtle subjects.

If scale values are frequently pinned at the min or max clamp, reduce `sega_distill_strength` or widen the clamp carefully. If `train/sega_distill_weighted_loss` is larger than the supervised loss for long periods, the teacher term is probably overpowering concept learning.
