"""Optional official Qwen Image 2.1 prompt expansion, isolated from diffusion VRAM.

Profiles and message format follow QwenLM/Qwen-Image-2.1/prompt_rewrite.
Only the parsed final answer is retained; reasoning text is neither logged nor saved.
"""
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from toolkit.image_io import open_static_image
from toolkit.paths import MODELS_PATH

REWRITER_REPOS = {
    "t2i": "Qwen/Qwen-Image-2.1-PE-T2I",
    "edit": "Qwen/Qwen-Image-2.1-PE-I2I",
}


def parse_rewrite(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Prompt rewriter did not finish its answer")
    decoder = json.JSONDecoder()
    answer = None
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        try:
            obj, size = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        index += size
        if isinstance(obj, dict):
            prompt = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
            if isinstance(prompt, str) and prompt.strip():
                answer = prompt.strip()
    if answer is None or len(answer) > 100_000:
        raise ValueError("Prompt rewriter returned no valid rewritten_prompt; original prompt was kept")
    return answer


class PresencePenalty:
    def __init__(self, penalty, prompt_length):
        self.penalty, self.prompt_length = penalty, prompt_length

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            generated = input_ids[row, self.prompt_length:]
            if generated.numel():
                scores[row, generated.unique()] -= self.penalty
        return scores


class CheckCancellation(StoppingCriteria):
    def __init__(self, check):
        self.check = check

    def __call__(self, input_ids, scores, **kwargs):
        if self.check:
            self.check()
        return False


@torch.inference_mode()
def rewrite_prompt(prompt, paths, seed, check_cancel=None):
    """Load the task-specific 9B rewriter on CPU and release it before sampling."""
    task = "edit" if paths else "t2i"
    repo = REWRITER_REPOS[task]
    if check_cancel:
        check_cancel()
    cache_dir = str(Path(MODELS_PATH) / "huggingface")
    system = Path(hf_hub_download(repo, "system_prompt.txt", cache_dir=cache_dir)).read_text(encoding="utf-8").strip()
    images = []
    for path in paths:
        image = open_static_image(path, mode="RGBA")
        rgb = image.convert("RGB")
        # Match the image model's visible conditioning, including transparency.
        rgb.paste((255, 255, 255), mask=image.getchannel("A").point(lambda alpha: 255-alpha))
        if rgb.width * rgb.height > 1024 ** 2:
            scale = (1024 ** 2 / (rgb.width * rgb.height)) ** 0.5
            rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))))
        images.append(rgb)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [*({"type": "image", "image": image} for image in images), {"type": "text", "text": prompt}]},
    ]
    model = processor = inputs = output = None
    try:
        processor = AutoProcessor.from_pretrained(repo, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, low_cpu_mem_usage=True, cache_dir=cache_dir).eval()
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=True)
        if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
            inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])
        prompt_length = inputs["input_ids"].shape[1]
        penalties = LogitsProcessorList([PresencePenalty(1.5, prompt_length)]) if task == "t2i" else LogitsProcessorList()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            output = model.generate(**inputs, max_new_tokens=16256 if task == "t2i" else 24000,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                logits_processor=penalties, stopping_criteria=StoppingCriteriaList([CheckCancellation(check_cancel)]),
                pad_token_id=processor.tokenizer.eos_token_id)
        return parse_rewrite(processor.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
    finally:
        del model, processor, inputs, output
        gc.collect()
