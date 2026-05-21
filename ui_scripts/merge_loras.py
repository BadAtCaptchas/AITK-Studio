"""Merge a list of LoRAs into a single safetensors checkpoint."""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge a list of LoRAs into a single checkpoint."
    )
    parser.add_argument(
        "--loras",
        required=True,
        help='JSON list of {"path": "...", "strength": 1.0} entries.',
    )
    parser.add_argument("--output", required=True, help="Output .safetensors path.")
    parser.add_argument(
        "--save_dtype",
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Dtype of saved tensors. Merging is always done in float32.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to merge on: cpu, cuda, cuda:1, or mps. Output is saved from CPU.",
    )
    args = parser.parse_args()

    try:
        loras = json.loads(args.loras)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse --loras JSON: {exc}", file=sys.stderr, flush=True)
        return 2

    if not isinstance(loras, list) or len(loras) == 0:
        print("--loras must be a non-empty JSON list.", file=sys.stderr, flush=True)
        return 2

    device = torch.device(args.device)
    save_dtype = DTYPE_MAP[args.save_dtype]
    merged: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    log(f"Merging {len(loras)} LoRA(s) on {device}, saving as {args.save_dtype}.")

    for index, entry in enumerate(loras):
        if not isinstance(entry, dict) or "path" not in entry:
            print(
                f"LoRA entry {index} must be an object with a 'path' field.",
                file=sys.stderr,
                flush=True,
            )
            return 2

        lora_path = entry["path"]
        strength = float(entry.get("strength", 1.0))

        if not os.path.isfile(lora_path):
            print(f"LoRA file not found: {lora_path}", file=sys.stderr, flush=True)
            return 2

        log(f"[{index + 1}/{len(loras)}] Loading {lora_path} (strength={strength})")
        state_dict = load_file(lora_path, device=str(device))

        for key, tensor in state_dict.items():
            scaled = tensor.to(torch.float32) * strength
            if key in merged:
                merged[key].add_(scaled)
            else:
                merged[key] = scaled
        del state_dict

        if index == 0:
            with safe_open(lora_path, framework="pt") as opened:
                original_metadata = opened.metadata()
                if original_metadata:
                    for meta_key in ["version", "format", "ss_base_model_version", "software"]:
                        if meta_key in original_metadata:
                            metadata[meta_key] = original_metadata[meta_key]

    log(f"Casting to {args.save_dtype} and moving to CPU")
    final = {key: tensor.to(save_dtype).cpu().contiguous() for key, tensor in merged.items()}
    merged.clear()

    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    log(f"Saving merged checkpoint to {args.output}")
    save_file(final, args.output, metadata=metadata)

    print(
        json.dumps(
            {
                "ok": True,
                "output": args.output,
                "num_loras": len(loras),
                "num_keys": len(final),
                "save_dtype": args.save_dtype,
                "device": str(device),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
