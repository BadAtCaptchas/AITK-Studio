from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Toolkit owns accumulation windows and loss normalization. An external
        # Accelerate setting must not divide losses again or skip our updates.
        global_accelerator = Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=1)
        )
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
