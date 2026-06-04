import uuid
from typing import Any, Dict, List, Optional

from toolkit.comfy.artifacts import export_training_artifacts, model_config_to_dict
from toolkit.comfy.runtime import runtime_for_config
from toolkit.comfy.workflows import build_workflow, validate_workflow_nodes


def generate_images_with_comfy(
    image_configs: List[Any],
    sampler: str,
    comfy_config,
    model_config: Optional[Dict[str, Any]] = None,
    process_config: Optional[Dict[str, Any]] = None,
    training_process=None,
    step: Optional[int] = None,
    device: Optional[str] = None,
    progress_hook=None,
    cancel_check=None,
):
    if not image_configs:
        return []

    if model_config is None:
        if training_process is None:
            model_config = {}
        else:
            model_config = model_config_to_dict(training_process)

    artifact = export_training_artifacts(training_process, step=step) if training_process is not None else None
    runtime = runtime_for_config(comfy_config)
    client = runtime.start()
    saved_paths = []
    try:
        object_info = client.object_info()
        total = len(image_configs)
        for idx, gen_config in enumerate(image_configs):
            workflow = build_workflow(
                comfy_config=comfy_config,
                gen_config=gen_config,
                sampler=sampler,
                model_config=model_config,
                process_config=process_config,
                artifact=artifact,
                device=device,
            )
            validate_workflow_nodes(workflow, object_info)
            history = client.prompt_and_wait(workflow, client_id=str(uuid.uuid4()), cancel_check=cancel_check)
            saved_paths.extend(client.save_history_images(history, gen_config))
            if progress_hook is not None:
                progress_hook(idx, total)
            if comfy_config.free_memory_after_each:
                try:
                    client.free()
                except Exception:
                    pass
        return saved_paths
    finally:
        runtime.close()
