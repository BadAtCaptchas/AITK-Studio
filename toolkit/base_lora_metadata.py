from typing import Any, Dict


def add_base_lora_metadata(meta: Dict[str, Any], model_config: Any) -> None:
    base_lora_path = getattr(model_config, "base_lora_path", None)
    if base_lora_path is None or not str(base_lora_path).strip():
        return
    meta["aitk_trained_on_adapted_base"] = True
    meta["aitk_base_lora_path"] = str(base_lora_path)
    meta["aitk_base_lora_strength"] = str(getattr(model_config, "base_lora_strength", 1.0))


def add_distillation_lora_metadata(meta: Dict[str, Any], model_config: Any) -> None:
    """Record the frozen dependency required to reproduce a Turbo adapter."""
    path = getattr(model_config, "model_kwargs", {}).get("distillation_lora_path")
    if not path:
        return
    meta["aitk_distillation_lora_path"] = str(path)
    meta["aitk_distillation_lora_strength"] = "1.0"
    meta["aitk_distillation_lora_required"] = True
    meta["aitk_base_model_name_or_path"] = str(model_config.name_or_path)
