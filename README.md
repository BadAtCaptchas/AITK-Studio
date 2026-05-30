# OstrisAI-Toolkit Revamped

**Experimental fork notice:** OstrisAI-Toolkit Revamped is an experimental fork of the original [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit), not the upstream project. It contains fast-moving model integrations, UI changes, remote-worker support, and other changes that may be unstable or diverge from upstream behavior.

> This fork is not guaranteed to stay in sync with upstream. New upstream model support may arrive here after a delay, may be changed substantially, or may not be ported at all. Use the upstream repo if you need the canonical AI Toolkit release.

OstrisAI-Toolkit Revamped is an easy to use all in one training suite for diffusion models. It aims to support current models on consumer grade hardware, including image, video, and audio models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.

## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [Lakonik/AsymFLUX.2-klein-9B](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B) (AsymFLUX.2-klein-9B, experimental)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) (GLM-Image)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream-O1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [RunDiffusion/Juggernaut-Z-Image](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image) (Juggernaut Z)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)

HiDream-O1 training defaults to `train.t0_loss_target: true`, so the trainer compares the reconstructed timestep-0 prediction directly against the image latent target. That keeps O1 in its native x0 loss space instead of relying on velocity-space loss weighting to control small-timestep spikes.

GLM-Image is supported for text-to-image sampling and transformer LoRA training through upstream Diffusers `GlmImagePipeline` and `GlmImageTransformer2DModel`. The built-in `glm_image` preset defaults to `zai-org/GLM-Image`, flowmatch scheduling, 1024px samples, 50 sample steps, guidance `1.5`, quantization, and exposed low-VRAM controls. V1 trains transformer LoRA only with `target_lora_modules: ["GlmImageTransformer2DModel"]`.

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

## Installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/rmcc3/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install the older-GPU Torch stack first
pip3 install -r requirements_torch_legacy_cu128.txt
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/rmcc3/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements_torch_legacy_cu128.txt
pip install -r requirements.txt
```

### NVIDIA Blackwell / RTX 50-series GPUs

Blackwell GPUs such as the RTX 50-series require a PyTorch build with CUDA 12.8 or newer and `sm_120` kernels. If an older CUDA wheel is installed, PyTorch may still report that CUDA is available, but model loading or training can fail or run poorly once kernels are used.

Use the Blackwell Torch stack on Windows and standard Linux systems:

```bash
pip install -r requirements_torch_blackwell_cu128.txt
```

DGX OS users should use the CUDA 13.0 stack in `dgx_instructions.md`.

You can verify the active environment with:

```bash
python scripts/check_blackwell_cuda.py
```

OstrisAI-Toolkit Revamped also checks this at startup and will fail early with the recommended install command if it detects a Blackwell GPU with an incompatible PyTorch wheel. Non-Blackwell GPUs continue with a warning if the active Torch version is outside the older-GPU known-good stack. Set `AI_TOOLKIT_SKIP_CUDA_COMPAT_CHECK=1` only for a custom PyTorch build that you know includes Blackwell support.

### HiDream-O1 PyTorch note

The HiDream-O1 model card currently warns against PyTorch 2.9.x. For older GPUs such as L40, use `requirements_torch_legacy_cu128.txt` (`torch==2.8.0`, `torchcodec==0.7.0`). For Blackwell, use `requirements_torch_blackwell_cu128.txt` or the DGX CUDA 13.0 stack (`torch==2.10.0`, `torchcodec==0.10.0`). AI Toolkit warns at runtime when HiDream-O1 is started on PyTorch 2.9.x.

### Quantized model cache

FLUX.2 quantized transformer loads and FLUX.2 Klein quantized Qwen3 text encoder loads cache supported `optimum.quanto` weights by default. The first quantized load still builds the quantized model, then later runs can reuse the cache instead of quantizing the same component again.

The default cache location is `MODELS_PATH/.aitk_quantized_cache`, which is usually `models/.aitk_quantized_cache` in this repo. The cache is keyed by the source model files, quantization type, dtype, model settings, and package versions so it is rebuilt when the inputs change.

Set `quantize_cache: false` to disable the cache, or set `quantize_cache_dir` to move it:

```yaml
model:
  quantize_cache: true
  quantize_cache_dir: /path/to/cache
```

The cache is used for `optimum.quanto` qtypes such as `qfloat8`. It is skipped for torchao qtypes and for FLUX.2 transformer loads that use an accuracy recovery adapter.

MacOS:

Experimental support for Silicon Macs is available. I do not have a Mac with enough RAM to fully test this
so please let me know if there are issues. There is a convience script to install and run on MacOS 
locates at `./run_mac.zsh` that will install the dependencies locally and run the UI. To run this, 
do the following:

```bash
git clone https://github.com/rmcc3/ai-toolkit.git
cd ai-toolkit
chmod +x run_mac.zsh
./run_mac.zsh
```


# OstrisAI-Toolkit Revamped UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="OstrisAI-Toolkit Revamped UI" width="100%">

The OstrisAI-Toolkit Revamped UI is a web interface for OstrisAI-Toolkit Revamped. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js >=20.19.0

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Image generation UI

The **Generate** page can run image generation from a base model or a locally trained LoRA without creating a training job. A single requested image is generated inline and displayed on the same page by default. If the request would create more than one image, for example multiple prompts or `Images per Prompt` greater than `1`, the UI creates a normal `generate` job instead so it can run through the queue and be tracked from the jobs page.

Prompts can be typed directly, one prompt per line, or imported from a text file:

```txt
photo of a cinematic portrait, detailed lighting
wide shot of a futuristic city at sunrise
```

Prompt JSON files are also supported for per-image settings. The JSON can be an array, or an object with an `images`, `prompts`, or `samples` array. String entries use the page defaults; object entries can override settings for that image:

```json
{
  "images": [
    {
      "prompt": "photo of a cinematic portrait, detailed lighting",
      "width": 1024,
      "height": 1024,
      "seed": 1234,
      "guidance_scale": 4,
      "sample_steps": 20,
      "negative_prompt": "blurry, low quality"
    },
    {
      "prompt": "wide shot of a futuristic city at sunrise",
      "width": 1344,
      "height": 768,
      "sampler": "flowmatch",
      "format": "webp"
    }
  ]
}
```

Common per-image keys include `prompt`, `negative_prompt` or `neg`, `width`, `height`, `seed`, `guidance_scale`, `sample_steps`, `sampler`, `format` or `ext`, and `network_multiplier`.

## TensorBoard

TensorBoard is installed with the Python requirements. If `AITK_ENABLE_TENSORBOARD` is not set, the UI tries to auto-enable TensorBoard when the package is available in the active Python environment and silently skips it if the probe or startup fails.

You can force it on or off when starting the UI:

```bash
# Linux/macOS
AITK_ENABLE_TENSORBOARD=1 npm run build_and_start
AITK_ENABLE_TENSORBOARD=0 npm run build_and_start

# Windows Powershell
$env:AITK_ENABLE_TENSORBOARD="1"; npm run build_and_start
$env:AITK_ENABLE_TENSORBOARD="0"; npm run build_and_start
```

When TensorBoard is enabled, the UI starts it on port `6006`, writes a small `aitk_status` run so TensorBoard has data before the first training job, writes UI-launched training events to `<training folder>/.tensorboard`, and shows a TensorBoard link on the dashboard and train job overview.

TensorBoard is a separate service and is not protected by `AI_TOOLKIT_AUTH`; use localhost binding, a firewall, or proxy auth when exposing it outside a trusted network.

Optional environment variables:

- `AITK_TENSORBOARD_PORT=6006` changes the TensorBoard port.
- `AITK_TENSORBOARD_HOST=0.0.0.0` changes the bind host.
- `AITK_TENSORBOARD_LOG_DIR=/path/to/logs` changes the event log directory.
- `AITK_TENSORBOARD_PUBLIC_URL=http://host:6006` changes the link shown in the UI, useful behind proxies or custom Docker port mappings.
- `AITK_TENSORBOARD_STATUS_RUN=0` removes and stops writing the synthetic `aitk_status` run. Without another run, TensorBoard may show an empty dashboard until training writes events.

For Docker Compose, leave `AITK_ENABLE_TENSORBOARD` unset for auto-detection, or set it explicitly. The Compose file binds the published TensorBoard port to host `127.0.0.1` by default, even though TensorBoard listens on `0.0.0.0` inside the container so Docker port forwarding can reach it:

```bash
AITK_ENABLE_TENSORBOARD=1 docker compose up
AITK_ENABLE_TENSORBOARD=0 docker compose up
```

## UI database

The UI uses SQLite by default and stores its state in `aitk_db.db`. You can switch all UI-backed state to MongoDB at startup:

```bash
AITK_DB_PROVIDER=mongodb \
AITK_MONGODB_URI="mongodb://localhost:27017" \
AITK_MONGODB_DB=ai_toolkit \
npm run build_and_start
```

Supported database environment variables:

- `AITK_DB_PROVIDER=sqlite|mongodb` defaults to `sqlite`.
- `AITK_SQLITE_PATH` defaults to `../aitk_db.db` from the `ui` folder.
- `AITK_MONGODB_URI` is required when `AITK_DB_PROVIDER=mongodb`.
- `AITK_MONGODB_DB` defaults to `ai_toolkit`.

Run `npm run update_db` after changing database settings. SQLite mode prepares Prisma and the SQLite schema. MongoDB mode prepares the MongoDB indexes while still generating the Prisma client for SQLite fallback support.

To migrate existing SQLite UI data into MongoDB, leave `aitk_db.db` and the training output folders in place, set the MongoDB variables, then run:

```bash
cd ui
AITK_MONGODB_URI="mongodb://localhost:27017" npm run migrate_sqlite_to_mongo
```

The migration imports jobs, queues, settings, and existing per-job `loss_log.db` metrics. SQLite files are left untouched so you can switch back to SQLite.

## Remote workers and Cloudflare Tunnel

The UI can control remote AI Toolkit worker instances. Each worker runs the same UI/cron app with `AI_TOOLKIT_AUTH` set. Add the worker from **Settings > Remote Workers** using its public URL and bearer token. When you start a job assigned to a remote worker, the central UI creates a `.aitk.zip` job bundle with datasets, uploads it to the worker, starts the worker queue, and then proxies logs, metrics, samples, checkpoints, and exports back through the central UI.

Remote workers are authoritative after upload. The central UI mirrors status, step, speed, config, and error text from the worker. Base model files are not bundled; they must exist on the worker or the import will report warnings.

Encrypted dataset bundles include only ciphertext dataset folders. Starting an encrypted job on a remote worker requires supplying the dataset secret at start time unless durable encrypted resume was enabled for that job. YubiKey-protected encrypted datasets use the same central unlock path: the browser connected to the central UI prompts for the USB security key, unwraps the dataset key, and the central server forwards only the ephemeral dataset key to the HTTPS worker start request. Remote workers do not need a YubiKey, USB access, or a native FIDO helper. Remote encrypted starts require an `https://` worker URL unless `AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS=1` is set explicitly.

Optional managed `cloudflared` support is configured with environment variables on any instance you want to expose through Cloudflare Tunnel:

```bash
AITK_CLOUDFLARED_ENABLED=1
AITK_CLOUDFLARED_PUBLIC_URL=https://your-worker.example.com
AITK_CLOUDFLARED_TOKEN_FILE=/path/to/cloudflared-token
AITK_CLOUDFLARED_TARGET_URL=http://127.0.0.1:8675
AITK_CLOUDFLARED_METRICS_ADDR=127.0.0.1:60123
AITK_CLOUDFLARED_LOG_LEVEL=info
AITK_CLOUDFLARED_AUTO_DOWNLOAD=0
```

`AI_TOOLKIT_AUTH` is required when `AITK_CLOUDFLARED_ENABLED=1`. `AITK_CLOUDFLARED_TOKEN_FILE` is optional: when it is set, the app starts a named tunnel with that token; when it is not set, the app starts a Cloudflare quick tunnel with a random `trycloudflare.com` URL and shows the generated URL in Settings after Cloudflared reports it. `AITK_CLOUDFLARED_PUBLIC_URL` is optional metadata for named tunnels. The app can start, stop, download, and show tunnel status from the Settings page; Docker images include `cloudflared` for this workflow. If the binary is missing and no custom `AITK_CLOUDFLARED_BIN` is set, the Settings page can download the official Cloudflare GitHub release into `bin/cloudflared` (`bin/cloudflared.exe` on Windows). Set `AITK_CLOUDFLARED_AUTO_DOWNLOAD=1` or enable the Settings checkbox to download automatically before starting.

### Secure remote Ollama captioning

Settings includes a **Secure Remote Captioning** tool for image datasets. It starts a local UI caption job that streams one image at a time to a selected remote worker's UI, where the worker calls its local Ollama server. The dataset is not bundled or stored on the remote worker, and prompt/system-prompt/image/caption payloads are encrypted at the application layer in addition to the worker's HTTPS tunnel and bearer token. The optional system prompt is saved per dataset in the central UI and reused when that dataset is selected.

On the remote worker, run the UI with `AI_TOOLKIT_AUTH` and Cloudflared as above, and keep Ollama bound to localhost:

```bash
AITK_OLLAMA_ENABLED=1
AITK_OLLAMA_HOST=127.0.0.1:11434
AITK_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Docker and RunPod images include Ollama. When `AITK_OLLAMA_ENABLED=1`, the startup scripts launch `ollama serve` without exposing port `11434`; only the authenticated UI is exposed through Cloudflared. If the selected Ollama model is not installed, the remote worker pulls it automatically before captioning.

Threat model limit: the remote host and its Ollama process must decrypt each image and prompt in memory to run inference. This protects transport, logs, and remote disk persistence; it does not protect against a compromised remote machine.

## Training job import/export

The UI can export and import training jobs from the queue page. Use the action menu on a training job to export either:

- `Export Job State` for the training folder, job metadata, config, optimizer state when present, and checkpoints.
- `Export With Datasets` for the same job state plus local dataset paths referenced by the job config.

Exports are saved as `.aitk.zip` archives and include a manifest, `job.json`, and `job_config.json`. Base model files are not bundled; local model paths are recorded and checked on import so missing references can be reported as warnings.

Large exports run in the background with progress for files and bytes, success and failed status handling, warning alerts, and a cancel button. Before each export, you can choose whether to include only the latest checkpoint or all checkpoint files in the training folder.

Use `Import Training Job` on the queue page to upload a `.aitk` or `.zip` export. Imports rewrite runtime-local paths, copy included datasets into the configured datasets root, pick the target GPU, rename the job if there is a name conflict, and add the job back to the queue in a stopped state so it can be resumed.

Encrypted dataset exports do not decrypt files. Import/export copies encrypted manifests and `objects/*.bin` files as-is. There is no plaintext or decrypt-export mode.

Jobs launched from the UI are detached from the cron worker process, and the worker now waits for in-flight queue work and disconnects cleanly on shutdown signals.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, set `AI_TOOLKIT_AUTH` before starting the UI.
When this variable is set, API routes require a matching bearer token and unauthenticated calls are rejected with `401 Unauthorized`.
This includes job creation and job queue/start endpoints, which should never be exposed without authentication.

You can set the environment variable `AI_TOOLKIT_AUTH` to a strong secret token when starting the UI:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Multi-step training phases

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
      auto_advance:
        type: loss_plateau
        min_steps: 500

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

Phase overrides inherit from the top-level `train` block. Supported phase-local overrides include learning rates, optimizer, optimizer params, LR scheduler params, timestep type/bias, loss type, denoising min/max, SNR settings, and prompt/noise multipliers. Model, network, dataset, save, sample, batch size, gradient accumulation, dtype, cache, LoRA rank, and LoKr factor settings stay top-level only for the whole run.

At each phase boundary the trainer saves by default, rebuilds the optimizer and LR scheduler, clears gradients, and continues. Phase changes only happen after a completed optimizer update, so boundaries defer cleanly during gradient accumulation. Checkpoints store the current phase index/name and phase-local step so resumes return to the correct phase.

Phases can also advance early by logged metric plateau:

```yaml
auto_advance:
  type: loss_plateau
  metric: loss/loss
  mode: min
  min_steps: 500
  window: 100
  patience: 2
  min_delta_pct: 1.0
```

Defaults are `metric: loss/loss`, `mode: min`, `window: 100`, `patience: 2`, `min_steps: max(200, window * 2)`, and `min_delta_pct: 1.0`. Generated sample images are not scored directly; future evaluators can feed numeric metrics into the same logging system. The UI loss graph shows phase boundary markers when phase metrics are present.

### Auto learn / auto training

Auto learn lets a training job keep running until the configured metric stops improving, then move to the next training phase. When the final phase plateaus, the trainer stops the job. This is useful when the correct number of steps is not known up front.

In the UI, open `New Job`, go to `Training Phases`, and enable `Auto learn`. Fixed step inputs are hidden because auto learn does not know the total step count ahead of time. Profiles can be global or scoped to the selected model architecture; legacy profiles without a model scope remain visible for every model, and newly saved custom profiles record the active model architecture.

The profile dropdown includes an `Anatomy LoKr` preset with three open-ended stages:

- Teach: AdamW, `lr: 0.00002`, weighted high-noise timesteps, MSE loss, `weight_decay: 0.0001`, LoKr factor 8.
- Stabilize: AdamW, `lr: 0.00001`, weighted balanced timesteps, MSE loss.
- Fine detail cleanup: AdamW, `lr: 0.000005`, weighted low-noise timesteps, MSE loss.

For GLM-Image, the UI defaults Auto learn to `glm-image-balanced-lora` instead of the generic Anatomy profile. The GLM profiles are open-ended, use loss-plateau auto advance, save on each phase change, and do not set fixed phase step counts:

- `glm-image-balanced-lora`: LoRA rank/alpha `32`, `adamw8bit`, weighted timesteps, MSE loss, and LR phases `0.00005 -> 0.00003 -> 0.000015` for content, balanced, and style stages.
- `glm-image-low-vram-lora`: LoRA rank/alpha `16`, dropout `0.05`, `adamw8bit`, weighted timesteps, MSE loss, batch size `1`, gradient accumulation `2`, and LR phases `0.00003 -> 0.00002 -> 0.00001`.

You can also save the current auto-learn settings as a custom profile from the same editor. Custom profiles are stored in the browser's local storage.

For CLI configs, set `train.auto_train: true` and omit phase `steps`. Each phase must have plateau auto-advance settings, either explicitly or by relying on the defaults:

```yaml
train:
  auto_train: true
  save_on_phase_change: true
  optimizer: adamw
  lr: 0.00002
  timestep_type: weighted
  content_or_style: content
  loss_type: mse
  optimizer_params:
    weight_decay: 0.0001
  phases:
    - name: teach anatomy
      lr: 0.00002
      content_or_style: content
      auto_advance:
        type: loss_plateau
        metric: loss/loss
        mode: min
        window: 100
        patience: 2
        min_delta_pct: 1.0

    - name: stabilize
      lr: 0.00001
      content_or_style: balanced
      auto_advance:
        type: loss_plateau

    - name: fine detail cleanup
      lr: 0.000005
      content_or_style: style
      auto_advance:
        type: loss_plateau
```

Progress displays use the current step without a percentage bar while auto learn is active, because there is no planned final step. Resuming a checkpoint restores the current phase and continues plateau tracking from the saved training state.

For a GLM-Image auto-train starting point, see `config/examples/train_lora_glm_image_auto_24gb.yaml`.

### Need help or found a bug?

This fork tracks reproducible code bugs in this repository only. If you find a bug in `rmcc3/ai-toolkit`, open a bug report at [github.com/rmcc3/ai-toolkit/issues/new?template=bug_report.md](https://github.com/rmcc3/ai-toolkit/issues/new?template=bug_report.md) and include your reproduction steps, environment details, logs, and the commit or version you are running.

Please do not open bug reports here for upstream-only behavior, setup help, usage questions, or feature requests. If the behavior exists only in the original Ostris AI Toolkit, report it upstream.

## Gradio UI

To get started training locally with a with a custom UI, once you followed the steps above and `ai-toolkit` is installed:

```bash
cd ai-toolkit #in case you are not yet in the ai-toolkit folder
huggingface-cli login #provide a `write` token to publish your LoRA at the end
python flux_train_ui.py
```

You will instantiate a UI that will let you upload your images, caption them, train and publish your LoRA
![image](assets/lora_ease_ui.png)


## Training in RunPod

This fork includes a maintained private RunPod Pod template for the revamped UI. See [`runpod/README.md`](runpod/README.md) for the Blackwell-first template, persistent volume layout, required `AI_TOOLKIT_AUTH` secret, and access URL format: `https://<POD_ID>-8675.proxy.runpod.net`.

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/rmcc3/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install -r requirements_torch_legacy_cu128.txt
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Supported static image formats
are jpg, jpeg, png, and webp. Animated WebP files are not supported as image dataset inputs; use a video dataset
format or convert them to static WebP, PNG, or JPEG first. WebP transparency is supported for alpha mask and inpaint
workflows, while normal training images are loaded as RGB. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.webp` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 

### Encrypted datasets

The UI can create encrypted datasets for images, video, audio, and captions. Choose **Encrypted** when creating a dataset, then select either:

- `Password`: the browser derives an AES-256-GCM key with WebCrypto PBKDF2-SHA256 and a random salt.
- `Key File`: the browser hashes the selected key file and uses the resulting raw key material.
- `YubiKey`: the browser uses WebAuthn PRF with a cross-platform security key, such as a USB YubiKey, to unwrap a randomly generated dataset key.

Encrypted upload happens before files leave the browser. The dataset folder stores:

- `.aitk_encrypted_dataset.json`: clear crypto headers plus an encrypted catalog.
- `objects/<random-id>.bin`: AES-GCM encrypted media and caption payloads.

Original filenames, captions, media metadata, and logical paths live inside the encrypted catalog. The server never receives plaintext media or captions during encrypted upload, preview, caption editing, auto-caption saves, training, import, or export.

To preview, edit, upload more files, auto-caption, or train with an encrypted dataset, open the dataset page and unlock it with the password, key file, or YubiKey. The browser keeps the raw key in page memory only. Training and caption jobs require the secret again when they start; by default, secrets are sent with the start request, are not written into job configs, database rows, logs, or export bundles, and are removed from the Python environment after launch.

YubiKey mode requires a browser and origin that support the WebAuthn PRF extension in a secure context. The dataset key is wrapped to the WebAuthn relying-party ID used when the dataset was created, so unlock from the same hostname you used during creation. The manifest records USB-capable credential metadata and a planned native USB extension point, but direct server-side `libfido2` or `python-fido2` USB access is not implemented yet.

For queue durability, set `AITK_DURABLE_DATASET_KEY_SECRET` to a real secret of at least 32 characters, then enable **Allow durable encrypted resume** when starting an encrypted train or caption job. This stores a wrapped copy of the dataset key in the UI database so the cron launcher can start or resume the queued job after the app restarts. Database backups alone are not enough to recover the dataset key without the server-side wrapping secret, but a compromised server process can still unwrap it. Durable keys are cleared when the job completes successfully or is deleted, and are retained after stop/error states so the job can resume. Changing `AITK_DURABLE_DATASET_KEY_SECRET` invalidates existing queued durable keys and users must re-enter the dataset secret. Durable keys are not written into job configs, launch logs, Python logs, or export bundles.

Threat model limit: encrypted datasets protect against plaintext at rest on disk and accidental dataset export. A compromised training host can still read the key or plaintext from browser, Node, or Python process memory while the dataset is unlocked or training is running. File count and ciphertext sizes are also visible.

Disk caches and plaintext sidecars are disabled for encrypted datasets. Generated controls and external control/mask/inpaint paths are not supported for encrypted datasets yet.


## Training Specific Layers

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
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.
