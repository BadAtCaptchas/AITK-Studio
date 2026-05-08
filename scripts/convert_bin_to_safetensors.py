#!/usr/bin/env python3
"""Convert trusted PyTorch checkpoint files (.bin/.ckpt/.pt) to .safetensors.

Security note:
    Loading PyTorch checkpoints can execute pickle payloads. Only run this script on
    trusted files in an isolated environment.
"""

import argparse
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def _load_checkpoint(path: str):
    """Load checkpoint preferring safer torch weights_only mode when available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj):
    if isinstance(obj, OrderedDict):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], (dict, OrderedDict)):
            return obj["state_dict"]
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def convert_bin_to_safetensors(input_path: str, output_path: str, overwrite: bool = False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    loaded = _load_checkpoint(input_path)
    state_dict = _extract_state_dict(loaded)

    tensor_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor_state_dict[key] = value.detach().cpu()

    if not tensor_state_dict:
        raise ValueError("No tensors found in checkpoint state dict")

    metadata = {
        "source": os.path.basename(input_path),
        "converted_by": "ai-toolkit/scripts/convert_bin_to_safetensors.py",
    }
    save_file(tensor_state_dict, output_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a trusted .bin/.ckpt/.pt checkpoint to .safetensors")
    parser.add_argument("input", help="Path to input checkpoint (.bin/.ckpt/.pt)")
    parser.add_argument("output", nargs="?", default=None, help="Path to output .safetensors")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()

    output = args.output
    if output is None:
        base, _ = os.path.splitext(args.input)
        output = f"{base}.safetensors"

    convert_bin_to_safetensors(args.input, output, overwrite=args.overwrite)
    print(f"Saved safetensors file: {output}")
