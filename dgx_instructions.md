# DGX OS and Linux ARM64 setup

DGX Spark and Grace systems use the standard Linux requirements and the managed installer. Run these commands from the repository root with a system Python 3.8 or newer:

```bash
python3 -m manager install
python3 -m manager launch
```

The manager chooses the NVIDIA runtime for the installed driver, installs Python 3.12 through uv, and provisions the UI dependencies locally. The UI opens at http://localhost:8675. See the [manager guide](manager/README.md) for the current runtime pins and `python3 -m manager doctor` for diagnostics.

Managed Python includes the C headers needed by Triton. If an existing Linux environment has the wrong interpreter or lacks `Python.h`, installation preserves it in a checkout-local backup before creating a replacement. A failed replacement restores the previous environment. No separate DGX requirements file or Conda environment is needed.

The manual `dgx-cu130` profile remains an alias for the standard CUDA 13.0 compatibility profile in [manual installation](docs/installation.md); it uses `requirements.txt`. Prefer the manager for new installations.

## Shared-memory behavior

On NVIDIA GB10, the GPU widget reports system RAM as shared memory when NVIDIA telemetry omits dedicated VRAM totals. Stale GPU readings still expire normally.

On Linux integrated NVIDIA GPUs, training suppresses module moves to CPU because CPU and GPU already share memory. Dtype conversions and tensor transfers remain active. Set `AITK_DISABLE_UNIFIED_MEMORY=1` before launching to disable this optimization.
