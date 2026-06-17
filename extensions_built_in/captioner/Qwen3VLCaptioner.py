from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
)
from collections import OrderedDict

import torch
import torch.nn.functional as F
from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner
import transformers
import logging
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def patch_qwen_vl_patch_embed(model):
    """Replace Qwen-VL patch Conv3d projections with equivalent linear GEMMs."""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if (
            isinstance(proj, torch.nn.Conv3d)
            and tuple(proj.kernel_size) == tuple(proj.stride)
        ):

            def fast_forward(hidden_states, _proj=proj):
                weight = _proj.weight.reshape(_proj.weight.shape[0], -1)
                hidden_states = hidden_states.view(-1, weight.shape[1]).to(weight.dtype)
                return F.linear(hidden_states, weight, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched


class Qwen3VLCaptioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Qwen3VLCaptioner, self).__init__(process_id, job, config, **kwargs)

    def load_model(self):
        self.print_and_status_update("Loading Qwen3VL model")
        ModelClass = (
            Qwen3VLMoeForConditionalGeneration
            if "B-A" in self.caption_config.model_name_or_path
            else Qwen3VLForConditionalGeneration
        )
        self.model = ModelClass.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        patch_qwen_vl_patch_embed(self.model)
        if not self.caption_config.low_vram:
            self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Qwen3VL model")
            quantize(self.model, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model)
            flush()
        self.processor = AutoProcessor.from_pretrained(
            self.caption_config.model_name_or_path
        )
        if self.caption_config.low_vram:
            self.model.to(self.device_torch)
        flush()

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": self.build_caption_prompt(file_path)},
                    ],
                }
            ]

            # Preparation for inference
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device_torch)

            # Inference: Generation of the output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.caption_config.max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return self.normalize_caption_output(file_path, output_text[0], image_size=img.size)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
