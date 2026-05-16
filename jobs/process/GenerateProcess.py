import gc
import json
import os
from collections import OrderedDict
from typing import Any, Dict, ForwardRef, List, Optional, Union

import torch
from safetensors.torch import save_file, load_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_model_hash_to_meta, \
    add_base_model_info_to_meta
from toolkit.sampler import get_sampler
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
import random

from toolkit.util.get_model import get_model_class


class GenerateConfig:

    def __init__(self, **kwargs):
        self.prompts: List[str]
        self.images: List[Dict[str, Any]]
        self.sampler = kwargs.get('sampler', 'ddpm')
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)
        self.size_list: Union[List[int], None] = kwargs.get('size_list', None)
        self.neg = kwargs.get('neg', '')
        self.seed = kwargs.get('seed', -1)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.prompt_2 = kwargs.get('prompt_2', None)
        self.neg_2 = kwargs.get('neg_2', None)
        self.prompts = kwargs.get('prompts', None)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.compile = kwargs.get('compile', False)
        self.ext = kwargs.get('ext', 'png')
        self.prompt_file = kwargs.get('prompt_file', False)
        try:
            self.num_repeats = int(kwargs.get('num_repeats', 1))
        except (TypeError, ValueError):
            self.num_repeats = 1
        self.prompts_in_file = self.prompts
        raw_images = kwargs.get('images', None)
        if raw_images is None:
            raw_images = self.prompts
        if raw_images is None:
            raise ValueError("Prompts must be set")

        raw_images = self._load_prompt_source(raw_images)
        self.images = self._normalize_images(raw_images)
        if len(self.images) == 0:
            raise ValueError("Prompts must contain at least one prompt")
        self.prompts_in_file = [image['prompt'] for image in self.images]

        self.random_prompts = kwargs.get('random_prompts', False)
        self.max_random_per_prompt = kwargs.get('max_random_per_prompt', 1)
        self.max_images = kwargs.get('max_images', 10000)

        if self.random_prompts:
            self.images = []
            for i in range(self.max_images):
                num_prompts = random.randint(1, self.max_random_per_prompt)
                prompt_list = [random.choice(self.prompts_in_file) for _ in range(num_prompts)]
                self.images.append({'prompt': ", ".join(prompt_list)})

        if kwargs.get('shuffle', False):
            random.shuffle(self.images)

        self.prompts = [image['prompt'] for image in self.images]

    def _load_prompt_source(self, source: Union[str, List[Any], Dict[str, Any]]):
        if not isinstance(source, str):
            return source
        if not os.path.exists(source):
            raise ValueError("Prompts file does not exist, put in list if you want to use a list of prompts")

        if source.lower().endswith('.json'):
            with open(source, 'r', encoding='utf-8') as f:
                parsed = json.load(f)
            return self._extract_json_prompt_items(parsed)

        with open(source, 'r', encoding='utf-8') as f:
            return [p.strip() for p in f.read().splitlines() if len(p.strip()) > 0]

    def _extract_json_prompt_items(self, parsed: Any):
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ['images', 'prompts', 'samples']:
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
            if parsed.get('prompt') is not None:
                return [parsed]
        raise ValueError("Prompt JSON must be an array, or an object with images, prompts, samples, or prompt")

    def _normalize_images(self, items: Any):
        if not isinstance(items, list):
            items = [items]

        images: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                prompt = item.strip()
                if prompt:
                    images.append({'prompt': prompt})
                continue

            if isinstance(item, dict):
                prompt = item.get('prompt', item.get('text', item.get('caption', '')))
                prompt = str(prompt).strip()
                if prompt:
                    image = dict(item)
                    image['prompt'] = prompt
                    images.append(image)
                continue

            raise ValueError("Prompt entries must be strings or objects with a prompt")

        return images


class GenerateProcess(BaseProcess):
    process_id: int
    config: OrderedDict
    progress_bar: ForwardRef('tqdm') = None
    sd: StableDiffusion

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.model_config = ModelConfig(**self.get_conf('model', required=True))
        self.device = self.get_conf('device', self.job.device)
        self.generate_config = GenerateConfig(**self.get_conf('generate', required=True))
        self.torch_dtype = get_torch_dtype(self.get_conf('dtype', 'float16'))

        self.progress_bar = None
        
        ModelClass = get_model_class(self.model_config)
        # if the model class has get_train_scheduler static method
        if hasattr(ModelClass, 'get_train_scheduler'):
            sampler = ModelClass.get_train_scheduler()
        else:
            # get the noise scheduler
            arch = 'sd'
            if self.model_config.is_pixart:
                arch = 'pixart'
            if self.model_config.is_flux:
                arch = 'flux'
            if self.model_config.is_lumina2:
                arch = 'lumina2'
            sampler = get_sampler(
                self.generate_config.sampler,
                {
                    "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
                },
                arch=arch,
            )
        self.sd = ModelClass(
            device=self.device,
            model_config=self.model_config,
            dtype=self.model_config.dtype,
            noise_scheduler=sampler,
        )

        print(f"Using device {self.device}")

    def clean_prompt(self, prompt: str):
        # remove any non alpha numeric characters or ,'" from prompt
        return ''.join(e for e in prompt if e.isalnum() or e in ", '\"")

    def get_image_value(self, image_config: Dict[str, Any], keys: Union[str, List[str]], fallback: Any):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            value = image_config.get(key, None)
            if value is not None and value != '':
                return value
        return fallback

    def get_image_int(self, image_config: Dict[str, Any], keys: Union[str, List[str]], fallback: int):
        value = self.get_image_value(image_config, keys, fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def get_image_float(self, image_config: Dict[str, Any], keys: Union[str, List[str]], fallback: float):
        value = self.get_image_value(image_config, keys, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def get_image_bool(self, image_config: Dict[str, Any], keys: Union[str, List[str]], fallback: bool):
        value = self.get_image_value(image_config, keys, fallback)
        if isinstance(value, str):
            return value.strip().lower() not in ['0', 'false', 'no', 'off', '']
        return bool(value)

    def run(self):
        with torch.no_grad():
            super().run()
            print("Loading model...")
            self.sd.load_model()
            self.sd.pipeline.to(self.device, self.torch_dtype)

            print("Compiling model...")
            # self.sd.unet = torch.compile(self.sd.unet, mode="reduce-overhead", fullgraph=True)
            if self.generate_config.compile:
                self.sd.unet = torch.compile(self.sd.unet, mode="reduce-overhead")

            total_images = len(self.generate_config.images) * self.generate_config.num_repeats
            print(f"Generating {total_images} images")
            # build prompt image configs
            sampler_groups = OrderedDict()
            for repeat_idx in range(self.generate_config.num_repeats):
                for prompt_idx, image_config in enumerate(self.generate_config.images):
                    prompt = image_config['prompt'].strip()
                    width = self.get_image_int(image_config, 'width', self.generate_config.width)
                    height = self.get_image_int(image_config, 'height', self.generate_config.height)
                    # prompt = self.clean_prompt(prompt)

                    if self.generate_config.size_list is not None and 'width' not in image_config and 'height' not in image_config:
                        # randomly select a size
                        width, height = random.choice(self.generate_config.size_list)

                    output_ext = str(self.get_image_value(image_config, ['ext', 'format'], self.generate_config.ext)).lstrip('.')
                    output_index = repeat_idx * len(self.generate_config.images) + prompt_idx
                    output_path = os.path.join(
                        self.output_folder,
                        f"[time]_000000000_{output_index}.{output_ext}"
                    )

                    sampler_name = str(self.get_image_value(image_config, 'sampler', self.generate_config.sampler))
                    prompt_image_config = GenerateImageConfig(
                        prompt=prompt,
                        prompt_2=self.get_image_value(image_config, 'prompt_2', self.generate_config.prompt_2),
                        width=width,
                        height=height,
                        num_inference_steps=self.get_image_int(
                            image_config,
                            ['sample_steps', 'steps', 'num_inference_steps'],
                            self.generate_config.sample_steps
                        ),
                        guidance_scale=self.get_image_float(
                            image_config,
                            ['guidance_scale', 'guidance'],
                            self.generate_config.guidance_scale
                        ),
                        negative_prompt=self.get_image_value(
                            image_config,
                            ['neg', 'negative_prompt'],
                            self.generate_config.neg
                        ),
                        negative_prompt_2=self.get_image_value(
                            image_config,
                            ['neg_2', 'negative_prompt_2'],
                            self.generate_config.neg_2
                        ),
                        seed=self.get_image_int(image_config, 'seed', self.generate_config.seed),
                        network_multiplier=self.get_image_float(image_config, 'network_multiplier', 1.0),
                        guidance_rescale=self.get_image_float(
                            image_config,
                            'guidance_rescale',
                            self.generate_config.guidance_rescale
                        ),
                        output_path=output_path,
                        output_ext=output_ext,
                        output_folder=self.output_folder,
                        add_prompt_file=self.get_image_bool(
                            image_config,
                            ['prompt_file', 'add_prompt_file'],
                            self.generate_config.prompt_file
                        ),
                        adapter_image_path=self.get_image_value(image_config, 'adapter_image_path', None),
                        adapter_conditioning_scale=self.get_image_float(
                            image_config,
                            'adapter_conditioning_scale',
                            1.0
                        ),
                        ctrl_img=self.get_image_value(image_config, 'ctrl_img', None),
                        ctrl_img_1=self.get_image_value(image_config, 'ctrl_img_1', None),
                        ctrl_img_2=self.get_image_value(image_config, 'ctrl_img_2', None),
                        ctrl_img_3=self.get_image_value(image_config, 'ctrl_img_3', None),
                        num_frames=self.get_image_int(image_config, 'num_frames', 1),
                        fps=self.get_image_int(image_config, 'fps', 15),
                        do_cfg_norm=self.get_image_bool(image_config, 'do_cfg_norm', False),
                    )
                    if sampler_name not in sampler_groups:
                        sampler_groups[sampler_name] = []
                    sampler_groups[sampler_name].append(prompt_image_config)
            # generate images
            for sampler_name, image_configs in sampler_groups.items():
                self.sd.generate_images(image_configs, sampler=sampler_name)

            print("Done generating images")
            # cleanup
            del self.sd
            gc.collect()
            torch.cuda.empty_cache()
