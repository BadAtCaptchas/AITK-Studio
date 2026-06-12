import copy
import json
import os
from typing import Any, Dict, Iterable

from toolkit.comfy.errors import ComfyConfigError, ComfyWorkflowError


AITK_NODE_CLASS = 'AITKGenerateImage'


def build_workflow(
    comfy_config,
    gen_config,
    sampler: str,
    model_config: Dict[str, Any],
    process_config: Dict[str, Any] | None = None,
    artifact: Dict[str, Any] | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    if comfy_config.workflow is not None:
        workflow = load_workflow(comfy_config.workflow)
        apply_bindings(workflow, comfy_config.bindings, gen_config, sampler)
        return workflow

    if comfy_config.workflow_name not in [None, '', 'auto']:
        raise ComfyWorkflowError(
            f"Unknown built-in ComfyUI workflow '{comfy_config.workflow_name}'. "
            "Use workflow_name: auto or provide comfy.workflow."
        )

    payload = {
        'model_config': _json_safe(model_config),
        'process_config': _json_safe(process_config or {}),
        'image_config': image_config_to_dict(gen_config),
        'sampler': sampler,
        'device': device,
        'artifact': _json_safe(artifact or {}),
    }
    return {
        '1': {
            'class_type': AITK_NODE_CLASS,
            'inputs': {
                'config_json': json.dumps(payload),
            },
        },
        '2': {
            'class_type': 'SaveImage',
            'inputs': {
                'images': ['1', 0],
                'filename_prefix': 'aitk_comfy/aitk',
            },
        },
    }


def load_workflow(workflow: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(workflow, dict):
        return copy.deepcopy(workflow)
    if not isinstance(workflow, str):
        raise ComfyConfigError("comfy.workflow must be a workflow object or JSON file path")
    if not os.path.exists(workflow):
        raise ComfyConfigError(f"ComfyUI workflow does not exist: {workflow}")
    with open(workflow, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ComfyConfigError("ComfyUI workflow JSON must be an object")
    return loaded


def image_config_to_dict(gen_config) -> Dict[str, Any]:
    return {
        'prompt': gen_config.prompt,
        'prompt_2': gen_config.prompt_2,
        'width': gen_config.width,
        'height': gen_config.height,
        'num_inference_steps': gen_config.num_inference_steps,
        'guidance_scale': gen_config.guidance_scale,
        'negative_prompt': gen_config.negative_prompt,
        'negative_prompt_2': gen_config.negative_prompt_2,
        'seed': gen_config.seed,
        'network_multiplier': gen_config.network_multiplier,
        'guidance_rescale': gen_config.guidance_rescale,
        'output_ext': gen_config.output_ext,
        'adapter_image_path': gen_config.adapter_image_path,
        'adapter_conditioning_scale': gen_config.adapter_conditioning_scale,
        'refiner_start_at': gen_config.refiner_start_at,
        'extra_values': _json_safe(gen_config.extra_values),
        'num_frames': gen_config.num_frames,
        'fps': gen_config.fps,
        'ctrl_img': gen_config.ctrl_img,
        'ctrl_idx': gen_config.ctrl_idx,
        'ctrl_img_1': gen_config.ctrl_img_1,
        'ctrl_img_2': gen_config.ctrl_img_2,
        'ctrl_img_3': gen_config.ctrl_img_3,
        'do_cfg_norm': gen_config.do_cfg_norm,
    }


def apply_bindings(workflow: Dict[str, Any], bindings: Dict[str, Any], gen_config, sampler: str):
    values = {
        'prompt': gen_config.prompt,
        'prompt_2': gen_config.prompt_2,
        'negative_prompt': gen_config.negative_prompt,
        'neg': gen_config.negative_prompt,
        'negative_prompt_2': gen_config.negative_prompt_2,
        'neg_2': gen_config.negative_prompt_2,
        'seed': gen_config.seed,
        'width': gen_config.width,
        'height': gen_config.height,
        'steps': gen_config.num_inference_steps,
        'sample_steps': gen_config.num_inference_steps,
        'num_inference_steps': gen_config.num_inference_steps,
        'guidance_scale': gen_config.guidance_scale,
        'cfg': gen_config.guidance_scale,
        'sampler': sampler,
    }
    for name, paths in (bindings or {}).items():
        if name not in values:
            raise ComfyWorkflowError(f"Unknown ComfyUI workflow binding '{name}'")
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise ComfyWorkflowError(f"ComfyUI binding '{name}' must be a path string or list of path strings")
        for path in paths:
            set_workflow_path(workflow, path, values[name])


def set_workflow_path(workflow: Dict[str, Any], path: str, value: Any):
    parts = path.split('.')
    if not parts:
        raise ComfyWorkflowError("ComfyUI binding path cannot be empty")
    target: Any = workflow
    for part in parts[:-1]:
        if isinstance(target, dict) and part in target:
            target = target[part]
        elif isinstance(target, list) and part.isdigit() and int(part) < len(target):
            target = target[int(part)]
        else:
            raise ComfyWorkflowError(f"ComfyUI binding path does not exist: {path}")
    last = parts[-1]
    if isinstance(target, dict):
        target[last] = value
    elif isinstance(target, list) and last.isdigit() and int(last) < len(target):
        target[int(last)] = value
    else:
        raise ComfyWorkflowError(f"ComfyUI binding path does not exist: {path}")


def required_class_types(workflow: Dict[str, Any]) -> Iterable[str]:
    for node in workflow.values():
        if isinstance(node, dict) and node.get('class_type'):
            yield node['class_type']


def validate_workflow_nodes(workflow: Dict[str, Any], object_info: Dict[str, Any]):
    if not object_info:
        return
    missing = sorted(set(required_class_types(workflow)) - set(object_info.keys()))
    if missing:
        raise ComfyWorkflowError(f"ComfyUI server is missing required workflow nodes: {', '.join(missing)}")


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)
