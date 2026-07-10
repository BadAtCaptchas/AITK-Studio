from contextlib import contextmanager


@contextmanager
def temporary_vae_tiling(vae, enabled: bool = True):
    """Temporarily enable VAE tiling without changing a caller-owned setting."""
    if not enabled:
        yield vae
        return

    was_enabled = bool(getattr(vae, "use_tiling", False))
    if not was_enabled:
        vae.enable_tiling()
    try:
        yield vae
    finally:
        if not was_enabled:
            vae.disable_tiling()
