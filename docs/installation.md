# Installation

[← Back to AITK Studio](../README.md#documentation)

Set up the Python runtime, then launch the web UI. Commands start from the repository root unless a `cd` command says otherwise. For Docker, RunPod, Modal, or remote workers, see [deployment](deployment.md).

- [Requirements](#requirements)
- [Linux](#linux)
- [Windows](#windows)
- [macOS](#macos)
- [Run the UI](#run-the-ui)
- [NVIDIA Blackwell / RTX 50-series GPUs](#nvidia-blackwell--rtx-50-series-gpus)
- [HiDream-O1 PyTorch note](#hidream-o1-pytorch-note)
- [Runtime diagnostics](#runtime-diagnostics)

## Requirements

- Python 3.12 is the reference interpreter; Python 3.11 is retained for DGX. These dependency profiles do not support Python 3.13 or newer.
- NVIDIA GPU with enough VRAM for the model and workflow you want to run.
- Python virtual environment support.
- Git.
- Node.js 22.12+ (22.x), 24.x, or 26.x if you plan to use the web UI, as specified in `ui/package.json`.

## Linux

```bash
git clone https://github.com/BadAtCaptchas/AITK-Studio.git
cd AITK-Studio
python3.12 -m venv venv
source venv/bin/activate
python scripts/install_runtime.py
```

For devices running **DGX OS** (including DGX Spark), follow the [DGX setup guide](../dgx_instructions.md).

## Windows

```powershell
git clone https://github.com/BadAtCaptchas/AITK-Studio.git
cd AITK-Studio
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python scripts/install_runtime.py
```

## macOS

Experimental support for Apple silicon Macs is available. It has not been fully tested on high-RAM Mac systems, so please report issues if you hit them. The convenience script at `./run_mac.zsh` installs dependencies locally and starts the UI:

```bash
git clone https://github.com/BadAtCaptchas/AITK-Studio.git
cd AITK-Studio
chmod +x run_mac.zsh
./run_mac.zsh
```

## Run the UI

The UI does not need to stay open for jobs to keep running. It is only needed to start, stop, and monitor jobs. The command below installs or updates the UI dependencies, builds the app, and starts it.

```bash
cd ui
npm run build_and_start
```

Open [http://localhost:8675](http://localhost:8675). Direct installations bind to localhost by default; see [network access and authentication](deployment.md#security) for LAN or server access.

## NVIDIA Blackwell / RTX 50-series GPUs

Blackwell GPUs such as the RTX 50-series require a PyTorch build with CUDA 12.8 or newer and `sm_120` kernels. If an older CUDA wheel is installed, PyTorch may still report that CUDA is available, but model loading or training can fail or run poorly once kernels are used.

The default installer uses CUDA 13.0 (`cu130`) on Windows and standard Linux, including Blackwell GPUs:

```bash
python scripts/install_runtime.py
```

DGX OS users should use the CUDA 13.0 stack in the [DGX setup guide](../dgx_instructions.md).

The default profile is `blackwell-cu130`: Torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0, and TorchCodec 0.10.0. It requires a CUDA 13-capable NVIDIA driver. For CUDA 12 compatibility, explicitly select `--profile blackwell-cu128`, or `--profile legacy-cu128` for the older-GPU Torch 2.8.0 stack. Blackwell still supports CUDA 12.8 wheels that include `sm_120` kernels. These profiles remain available as compatibility alternatives.

You can verify the active environment with:

```bash
python scripts/check_blackwell_cuda.py
```

AITK Studio also checks this at startup and will fail early with the recommended install command if it detects a Blackwell GPU with an incompatible PyTorch wheel. Non-Blackwell GPUs do not trigger a Torch downgrade recommendation solely because of their GPU generation. Set `AI_TOOLKIT_SKIP_CUDA_COMPAT_CHECK=1` only for a custom PyTorch build that you know includes Blackwell support.

## HiDream-O1 PyTorch note

AITK Studio warns when HiDream-O1 runs on PyTorch 2.9.x. The default CUDA 13.0 profile uses `torch==2.10.0` and `torchcodec==0.10.0`. The explicit `legacy-cu128` profile remains an alternative for compatible older GPUs (`torch==2.8.0`, `torchcodec==0.7.0`); `blackwell-cu128` provides a CUDA 12.8 alternative for Blackwell. DGX users should follow the [DGX CUDA 13.0 instructions](../dgx_instructions.md).

## Runtime diagnostics

Run these commands from the repository root with your virtual environment active:

```bash
python scripts/environment_doctor.py
python scripts/check_blackwell_cuda.py
```

The environment doctor checks the installed runtime without downloading model weights or starting training. See [execution and recovery](execution-safety.md#operator-setup) for interpreter selection, installation profiles, and deeper diagnostics.
