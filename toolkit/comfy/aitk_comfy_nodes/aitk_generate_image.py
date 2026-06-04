import json
import os
import sys
import tempfile


TOOLKIT_ROOT = os.environ.get('AITK_TOOLKIT_ROOT')
if TOOLKIT_ROOT and TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)


class AITKGenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'config_json': ('STRING', {'multiline': True}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'generate'
    CATEGORY = 'AI Toolkit'

    def generate(self, config_json):
        import numpy as np
        import torch
        from PIL import Image

        from toolkit.config_modules import GenerateImageConfig, ModelConfig
        from toolkit.sampler import get_sampler
        from toolkit.train_tools import get_torch_dtype
        from toolkit.util.get_model import get_model_class

        payload = json.loads(config_json)
        model_config_dict = dict(payload.get('model_config') or {})
        artifact = payload.get('artifact') or {}
        if artifact.get('kind') == 'network' and artifact.get('network_path'):
            model_config_dict['lora_path'] = artifact['network_path']
        elif artifact.get('kind') == 'model' and artifact.get('model_path'):
            model_config_dict['name_or_path'] = artifact['model_path']

        model_config = ModelConfig(**model_config_dict)
        model_class = get_model_class(model_config)
        if hasattr(model_class, 'get_train_scheduler'):
            noise_scheduler = model_class.get_train_scheduler()
        else:
            arch = 'sd'
            if model_config.is_pixart:
                arch = 'pixart'
            if model_config.is_flux:
                arch = 'flux'
            if model_config.is_lumina2:
                arch = 'lumina2'
            noise_scheduler = get_sampler(
                payload.get('sampler') or 'ddpm',
                {'prediction_type': 'v_prediction' if model_config.is_v_pred else 'epsilon'},
                arch=arch,
            )

        device = payload.get('device') or 'cuda'
        dtype_name = model_config_dict.get('dtype') or getattr(model_config, 'dtype', 'float16')
        sd = model_class(
            device=device,
            model_config=model_config,
            dtype=model_config.dtype,
            noise_scheduler=noise_scheduler,
        )
        sd.load_model()
        if not (model_config.low_vram or model_config.layer_offloading):
            sd.pipeline.to(device, get_torch_dtype(dtype_name))

        image_config = dict(payload.get('image_config') or {})
        output_ext = image_config.get('output_ext') or 'png'
        with tempfile.TemporaryDirectory(prefix='aitk_comfy_') as output_folder:
            image_config['output_folder'] = output_folder
            image_config.pop('output_path', None)
            image_config['output_ext'] = output_ext
            gen_config = GenerateImageConfig(**image_config)
            sd.generate_images([gen_config], sampler=payload.get('sampler'))
            image_path = gen_config.get_image_path(0, 0)
            image = Image.open(image_path).convert('RGB')

        array = np.asarray(image).astype(np.float32) / 255.0
        return (torch.from_numpy(array)[None,],)


NODE_CLASS_MAPPINGS = {
    'AITKGenerateImage': AITKGenerateImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'AITKGenerateImage': 'AI Toolkit Image Generate',
}
