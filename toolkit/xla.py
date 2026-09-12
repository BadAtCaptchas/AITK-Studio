"""Optional XLA synchronization shared by sampling pipelines."""

from diffusers.utils import is_torch_xla_available


def mark_xla_step() -> None:
    """Submit pending XLA operations when the optional backend is installed."""
    if is_torch_xla_available():
        # Keep the TPU-only dependency out of ordinary CPU/CUDA startup.
        from torch_xla.core.xla_model import mark_step

        mark_step()
