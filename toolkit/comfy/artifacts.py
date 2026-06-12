import copy
import os
from typing import Any, Dict, Optional


def model_config_to_dict(process) -> Dict[str, Any]:
    raw = {}
    if hasattr(process, 'raw_process_config') and isinstance(process.raw_process_config, dict):
        raw = copy.deepcopy(process.raw_process_config.get('model', {}))
    if not raw and hasattr(process, 'model_config'):
        raw = {
            key: copy.deepcopy(value)
            for key, value in vars(process.model_config).items()
            if not key.startswith('_')
        }
    if hasattr(process, 'train_config'):
        raw['dtype'] = getattr(process.train_config, 'dtype', raw.get('dtype', 'float16'))
    return raw


def export_training_artifacts(process, step: Optional[int] = None) -> Dict[str, Any]:
    from toolkit.train_tools import get_torch_dtype

    artifact_root = os.path.join(process.save_root, '.comfy', 'artifacts')
    step_name = 'latest' if step is None else str(step).zfill(9)
    artifact_dir = os.path.join(artifact_root, step_name)
    os.makedirs(artifact_dir, exist_ok=True)

    artifact: Dict[str, Any] = {
        'artifact_dir': artifact_dir,
        'kind': 'none',
    }

    save_meta = copy.deepcopy(getattr(process, 'meta', {}))
    save_dtype = get_torch_dtype(getattr(process.save_config, 'dtype', 'float16'))

    if getattr(process, 'network', None) is not None:
        network_path = os.path.join(artifact_dir, 'network.safetensors')
        prev_multiplier = getattr(process.network, 'multiplier', 1.0)
        process.network.multiplier = 1.0
        try:
            embedding_dict = process.embedding.state_dict() if getattr(process, 'embedding', None) is not None else None
            process.network.save_weights(
                network_path,
                dtype=save_dtype,
                metadata=save_meta,
                extra_state_dict=embedding_dict,
            )
        finally:
            process.network.multiplier = prev_multiplier
        artifact.update({'kind': 'network', 'network_path': network_path})
        return artifact

    if getattr(process, 'sd', None) is not None and (
        getattr(process.train_config, 'train_unet', False) or getattr(process.train_config, 'train_text_encoder', False)
    ):
        model_path = os.path.join(artifact_dir, 'model.safetensors')
        process.sd.save(model_path, save_meta, save_dtype)
        artifact.update({'kind': 'model', 'model_path': model_path})
        return artifact

    return artifact
