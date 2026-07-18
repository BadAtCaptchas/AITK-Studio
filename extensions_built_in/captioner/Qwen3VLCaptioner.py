from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
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
import traceback
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

MAX_THINKING_TOKENS = 4096


class ThinkingBudgetCriteria(StoppingCriteria):
    """Reserve the visible answer budget until a thinking block has ended."""

    def __init__(self, think_end_token_id: int, max_new_tokens: int):
        self.think_end_token_id = think_end_token_id
        self.max_new_tokens = max_new_tokens
        self.answer_start = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.answer_start is None:
            if input_ids[0, -1].item() == self.think_end_token_id:
                self.answer_start = input_ids.shape[1]
            return False
        return (input_ids.shape[1] - self.answer_start) >= self.max_new_tokens


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
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        patch_qwen_vl_patch_embed(self.model)
        if not self.caption_config.low_vram:
            self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Qwen3VL model")
            quantize(
                self.model,
                weights=get_qtype(self.caption_config.qtype),
                exclude=["lm_head", "*.lm_head"],
                quantize_device=(
                    self.device_torch if self.caption_config.low_vram else None
                ),
            )
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
                enable_thinking=self.caption_config.thinking,
            )
            inputs = inputs.to(self.device_torch)

            gen_kwargs = {"max_new_tokens": self.caption_config.max_new_tokens}
            if self.caption_config.thinking:
                think_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    "</think>"
                )
                if think_end_token_id is not None:
                    gen_kwargs = {
                        "max_new_tokens": (
                            MAX_THINKING_TOKENS
                            + self.caption_config.max_new_tokens
                        ),
                        "stopping_criteria": StoppingCriteriaList(
                            [
                                ThinkingBudgetCriteria(
                                    think_end_token_id,
                                    self.caption_config.max_new_tokens,
                                )
                            ]
                        ),
                    }

            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            caption = output_text[0]
            if "</think>" in caption:
                caption = caption.rsplit("</think>", 1)[-1]
            return self.normalize_caption_output(file_path, caption, image_size=img.size)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None
