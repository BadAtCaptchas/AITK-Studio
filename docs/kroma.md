# Kroma LoRA training

Kroma uses the existing Krea 2 model implementation. The three released
checkpoints have the same 430 tensor names and shapes as AITK's 12.82B-parameter
`SingleStreamDiT`. Conditioning uses Qwen3-VL-4B-Instruct's 12 selected hidden
layers and the Qwen-Image VAE. No new Python dependencies are required.

## Choose a checkpoint

| Studio choice / `model.arch` | File in `lodestones/Kroma` | Use |
| --- | --- | --- |
| Kroma Teacher / `krea2:kroma_teacher` | `kroma-sensei-booru-e6-teacher-velocity-v0.3.safetensors` | Recommended starting point; approximately 25.7 GB |
| Kroma Base / `krea2:kroma_base` | `kroma-v0.3-base.safetensors` | Normal training; approximately 51.3 GB, stored in FP32 |
| Kroma Turbo OPD / `krea2:kroma_turbo_opd` | `kroma-v0.3.1-turbo-opd.safetensors` | Experimental direct training; approximately 26.3 GB |

The [publisher recommends base or teacher training](https://huggingface.co/lodestones/Kroma)
and states that the resulting LoRAs can be loaded on Turbo OPD unchanged.
Teacher is recommended here for its smaller download, not a demonstrated quality
advantage. Direct Turbo training can degrade low-step generation and remains
experimental. The underlying Krea 2 license continues to apply; see the model card.

Select the preset in New Job, or copy one of these examples:

- [Teacher](../config/examples/train_lora_kroma_teacher_16gb.yaml)
- [Base](../config/examples/train_lora_kroma_base_16gb.yaml)
- [Turbo OPD](../config/examples/train_lora_kroma_turbo_opd_16gb.yaml)

Replace the example dataset and training-folder placeholders with your configured
workspace locations before running `python run.py <config.yaml>` in the project
environment. Studio resolves these locations through the existing workspace
settings. The examples do not change or create a separate training workspace.

`model.name_or_path` is `lodestones/Kroma`, with the exact filename in
`model.model_kwargs.checkpoint_filename`. A bare repo ID without that filename
would request `kroma.safetensors`, which does not exist. Local `.safetensors`
files also work; for a directory containing multiple checkpoints specify the
filename. Local text encoder and VAE paths can be supplied through
`model_kwargs.text_encoder_path` and `model_kwargs.vae_path`. Weights use the
existing Hugging Face cache. Offline execution requires cached weights,
tokenizers and configurations for all components.

## 16 GB starting profile

The presets use 512px buckets/previews, batch size 1, rank/alpha 16, BF16 compute,
Float8 transformer and text encoder weights, gradient checkpointing, latent/text
embedding caching, and text encoder unloading. Legacy layer offloading is enabled
at 100% for both components and remains enabled for previews. This trades speed
and host memory for VRAM. Disk checkpoint sizes are not VRAM requirements.

Training uses AdamW8bit and linear flow-matching timesteps, with a velocity target
of `noise - clean`. Starting learning rates are `1e-4` for base/teacher and `1e-5`
for Turbo. These are initial settings, not quality-tuned recipes.

Only denoiser LoRA weights are trained; the text encoder and VAE stay frozen.
The default exclusions are `txtfusion.` and `txtmlp`. Disable the existing
text-fusion exclusion control to include those denoiser projections. The example
field `network.transformer_only: false` permits all denoiser linear layers rather
than only repeated blocks; it does **not** enable text encoder training.

The transformer now supplies its block names and sensitive-layer exclusions to
the component loader. Input/output, timestep and text projection exclusions remain
in full precision while repeated blocks are quantized in sequence. Existing Krea
presets use this correction too.

## Preview guidance and the optional adapter

AITK's existing Krea sampler uses:

```text
velocity = conditional + guidance_scale * (conditional - unconditional)
```

Thus Krea guidance `0` is ComfyUI CFG `1`, `0.5` is CFG `1.5`, and `4` is CFG `5`.
At guidance zero only the conditional transformer pass runs. This convention is
unchanged for existing Krea jobs.

Base/teacher previews default to 25 steps and Krea guidance 4, with a
resolution-dependent schedule. Turbo previews use 10 steps, Krea guidance 0,
and fixed `model_kwargs.schedule_mu: 1.15`, consistent with the publisher's
[8–12 step sampling guidance](https://huggingface.co/lodestones/Kroma).
The fixed preview shift does not replace the training timestep schedule.

Turbo's **Experimental Turbo training adapter** selector defaults to None.
The opt-in [Krea 2 training adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter)
was developed for Krea 2 Turbo; its effectiveness on Kroma OPD is unverified.
The existing loader merges it before quantization, subtracts it during previews,
and exports only the newly trained LoRA. The YAML option is:

```yaml
assistant_lora_path: ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors
```

Put this under `model`, not `model_kwargs`. Training adapters and accuracy recovery
adapters serve different purposes. Switching away from the Turbo preset clears
the training adapter along with its checkpoint and sampling defaults.

## Export and validation

Exported adapters retain the Krea/Comfy `diffusion_model.*.lora_A.weight` and
`diffusion_model.*.lora_B.weight` layout. In ComfyUI, select a Kroma diffusion
checkpoint, use the Krea 2 text encoder loader and Qwen-Image VAE, and apply the
exported LoRA. For Turbo OPD, start with 10 steps, CFG 1 and mu 1.15. Compare
fixed-seed outputs at LoRA strengths zero and one. Cross-checkpoint key compatibility
does not by itself establish training quality.

Automated checks cover local/Hub checkpoint selection, offline errors, strict
shape loading, FP32/mixed precision, quantization exclusions, reduced-model
backward/checkpointing, frozen base weights, LoRA save/reload, assistant subtraction
and export separation, guidance arithmetic, schedules, preset transitions and YAML
round-trips. CUDA-capable environments also run Float8/offloading backward on a
reduced transformer.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_kroma -v
cd ui
npm.cmd run test:kroma
npm.cmd run typecheck
```

Full-checkpoint validation results are recorded below. Until a configuration is
listed as passing, the 16 GB profile is a target rather than a measured guarantee.
Long-run learning quality and ComfyUI image comparisons require separate
validation. This release does not add full-weight training,
edit conditioning, other adapter types, or new quantization backends.

### Local checks (2026-10-07)

Environment: Windows, RTX 5080 (16 GB), 191.2 GiB system RAM,
PyTorch 2.10.0+cu130, Diffusers 0.39.0.dev0, Transformers 5.5.3. The tests used
the existing project environment without dependency changes. Kroma checkpoints
were downloaded from revision `c28f8eadb4938445051dee698db76d08d1d374a6`.

The full-model smoke runs use four synthetic colored-circle images with short
captions, rank/alpha 16, the preset quantization/offloading settings, one optimizer
step and native previews. Each 512px variant also starts a fresh process, loads
the saved adapter and optimizer, resumes at step 1 and completes step 2.
All 458 exported teacher adapter tensor shapes match Turbo OPD's target layers.

GPU numbers below are PyTorch peak allocated/reserved memory, not total driver
usage. Host numbers are observed peak process-tree RSS, sampled once per second.
Step timings are single-step observations, not throughput benchmarks; loading,
caching and previews take additional time. Host memory use is substantial even
when the GPU has ample headroom. These measurements do not establish a minimum
system-RAM requirement or guarantees for longer captions and real datasets.

| Check | Result | GPU allocated / reserved (GiB) | Host RSS (GiB) | Training step (s) |
| --- | --- | --- | --- | --- |
| Teacher, 512px | Train, save, baseline/final previews passed | 3.16 / 4.56 | 50.60 | 4.62 |
| Base, 512px | Train, save, baseline/final previews passed | 3.16 / 4.56 | 72.57 | 4.16 |
| Turbo OPD, 512px | Train, save, baseline/final previews passed | 3.16 / 4.56 | 38.42 | 3.87 |
| Turbo OPD + adapter, 512px | Train, save, baseline/final previews passed | 3.16 / 4.49 | 38.91 | 4.09 |
| Teacher resume, 512px | Adapter/optimizer reload, next step, save and preview passed | 3.27 / 4.51 | 42.38 | 5.17 |
| Base resume, 512px | Adapter/optimizer reload, next step, save and preview passed | 3.27 / 4.51 | 72.58 | 4.77 |
| Turbo OPD resume, 512px | Adapter/optimizer reload, next step, save and preview passed | 3.27 / 4.51 | 37.48 | 4.79 |
| Turbo OPD + adapter resume, 512px | Adapter/optimizer reload, next step, save and preview passed | 3.27 / 4.48 | 38.96 | 4.93 |
| Teacher, 768px | Train, save and preview passed | 3.69 / 5.98 | 45.72 | 7.15 |
| Teacher, 1024px | Train, save and preview passed | 5.24 / 8.18 | 51.36 | 8.66 |

The 768/1024px checks use 1024px source fixtures and are teacher-only. They do not
change the conservative 512px defaults. One or two training steps cannot measure
concept learning, convergence, or the Turbo adapter's protection against loss of
distillation. Both direct Turbo modes therefore remain experimental.

In a separate fresh process, the teacher LoRA was loaded onto Turbo OPD with
all 458 tensors matching the saved values exactly and no missing keys. Native
512px inference at seed 42, 10 steps, Krea guidance 0 and mu 1.15 produced valid
images at LoRA strengths 0 and 1. Their mean absolute pixel difference was 0.841
on the 0–255 scale; 171,517 pixels changed. Peak GPU allocation/reservation was
3.16/3.95 GiB. This establishes working adapter application, not improved quality.

Nine focused Python tests passed, including the CUDA Float8/offloading test and
real registry/model imports. The 24 targeted UI/inference tests, app/worker
typecheck, focused ESLint and Python undefined-name checks also passed.

ComfyUI image compatibility has **not** been exercised: no ComfyUI installation
path or endpoint was configured for this validation. The export format and target
tensor shapes are checked, but this does not replace a ComfyUI runtime test.
