"""Validate engine requests before they enter the resident queue."""
import math


def validate_generation(body):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    model, sample = body.get("model"), body.get("sample", {})
    stream = body.get("stream") or {}
    if not isinstance(model, dict) or not isinstance(model.get("arch"), str) or not model["arch"]:
        raise ValueError("model.arch is required")
    if not isinstance(sample, dict) or not isinstance(stream, dict):
        raise ValueError("sample and stream must be objects")
    # The engine alone owns its output location; client paths must never override it.
    if any(k in sample for k in ("output_path", "output_folder", "output_tail", "output_name")):
        raise ValueError("Output paths are managed by the engine")
    if sample.get("output_ext", "png") not in ("png", "jpg", "jpeg", "webp", "mp4", "mp3", "wav", "flac", "txt"):
        raise ValueError("Unsupported output extension")
    for key in ("prompt", "negative_prompt"):
        if key in sample and (not isinstance(sample[key], str) or len(sample[key]) > 100_000):
            raise ValueError(f"Invalid {key}")
    for key, low, high in (("width", 1, 32768), ("height", 1, 32768), ("num_inference_steps", 1, 10000), ("num_frames", 1, 4097), ("fps", 0.001, 240), ("duration", 0.01, 3600), ("guidance_scale", 0, 1000), ("seed", -1, 2**63 - 1)):
        value = sample.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"Invalid {key}")
    loras = model.get("loras") or []
    if not isinstance(loras, list) or len(loras) > 32:
        raise ValueError("At most 32 adapters are supported")
    for lora in loras:
        if not isinstance(lora, dict) or not isinstance(lora.get("path"), str) or not lora["path"].strip():
            raise ValueError("Each adapter requires a path")
        strength = lora.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or abs(strength) > 16:
            raise ValueError("Invalid adapter strength")
    if model.get("lora_mode", "hook") not in ("hook", "merge"):
        raise ValueError("Invalid adapter mode")
    for key, maximum in (("every_n_steps", 10000), ("max_frames", 16)):
        value = stream.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"Invalid stream.{key}")
    return model, sample, stream
