#!/usr/bin/env python3
"""Manual parity check for the local TIPSv2 DPT port.

This script intentionally loads the audited upstream Hugging Face code with
trust_remote_code=True. Run it only in a disposable maintenance environment
when updating toolkit.models.tipsv2.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from toolkit.models.tipsv2 import (  # noqa: E402
    TIPS2_BACKBONE_REPO,
    TIPS2_BACKBONE_REVISION,
    TIPS2_DPT_REPO,
    TIPS2_DPT_REVISION,
    TIPSv2DPTModel,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_remote_model(repo_id, revision, dtype):
    kwargs = {"trust_remote_code": True, "revision": revision}
    try:
        return AutoModel.from_pretrained(repo_id, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)


def tensor_stats(local_tensor, remote_tensor):
    diff = (local_tensor - remote_tensor).float().abs()
    return {
        "shape": list(local_tensor.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for parity inference.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = parser.parse_args()

    if args.image_size % 14 != 0:
        raise SystemExit("--image-size must be divisible by 14")

    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    local_model = TIPSv2DPTModel.from_pretrained(
        device=device,
        dtype=dtype,
        cache_dir=args.cache_dir,
    ).eval()

    remote_model = load_remote_model(TIPS2_DPT_REPO, TIPS2_DPT_REVISION, dtype)
    remote_backbone = load_remote_model(
        TIPS2_BACKBONE_REPO, TIPS2_BACKBONE_REVISION, dtype
    )
    remote_model._backbone = remote_backbone
    remote_model.to(device=device, dtype=dtype).eval()
    remote_model._backbone.to(device=device, dtype=dtype).eval()

    torch.manual_seed(0)
    pixel_values = torch.rand(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        local_out = local_model(pixel_values)
        remote_out = remote_model(pixel_values)

    results = {
        "dpt_repo": TIPS2_DPT_REPO,
        "dpt_revision": TIPS2_DPT_REVISION,
        "backbone_repo": TIPS2_BACKBONE_REPO,
        "backbone_revision": TIPS2_BACKBONE_REVISION,
        "stats": {
            "depth": tensor_stats(local_out.depth, remote_out.depth),
            "normals": tensor_stats(local_out.normals, remote_out.normals),
            "segmentation": tensor_stats(
                local_out.segmentation, remote_out.segmentation
            ),
        },
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    failures = [
        name
        for name, stats in results["stats"].items()
        if stats["max_abs"] > args.tolerance
    ]
    if failures:
        raise SystemExit(
            f"Parity check failed for {', '.join(failures)} at tolerance {args.tolerance}"
        )


if __name__ == "__main__":
    main()
