from typing import Any, Dict


def add_base_lora_metadata(meta: Dict[str, Any], model_config: Any) -> None:
    base_lora_path = getattr(model_config, "base_lora_path", None)
    if base_lora_path is None or not str(base_lora_path).strip():
        return
    meta["aitk_trained_on_adapted_base"] = True
    meta["aitk_base_lora_path"] = str(base_lora_path)
    meta["aitk_base_lora_strength"] = str(getattr(model_config, "base_lora_strength", 1.0))
