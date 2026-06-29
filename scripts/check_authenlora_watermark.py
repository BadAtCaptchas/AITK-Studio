import argparse
import hashlib
import json
import os
import sys

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

TOOLKIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)

from toolkit.watermarking.authenlora import AuthenLoRACodec, bit_accuracy
from toolkit.watermarking.codecs import get_builtin_codec_msg_bits, resolve_codec_path


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def secret_hash(secret: str):
    if not secret:
        return None
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def summarize_detection(bits: str, match=None, has_expected_secret: bool = False):
    zero_message = all(bit == "0" for bit in bits)
    if zero_message:
        return {
            "zero_message": True,
            "watermark_detected": False,
            "watermark_status": "not_detected",
        }
    if has_expected_secret:
        detected = bool(match)
        return {
            "zero_message": False,
            "watermark_detected": detected,
            "watermark_status": "verified" if detected else "mismatch",
        }
    return {
        "zero_message": False,
        "watermark_detected": None,
        "watermark_status": "candidate",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Decode AuthenLoRA watermark bits from an image.")
    parser.add_argument("--image", required=True, help="Image file to inspect.")
    parser.add_argument("--codec", required=True, help="Built-in codec ID or local codec checkpoint path.")
    parser.add_argument("--msg-bits", required=True, type=int, help="Expected message bit count.")
    parser.add_argument("--expected-secret", default="", help="Optional binary secret to compare against.")
    parser.add_argument("--threshold", default=0.75, type=float, help="Bit-accuracy threshold for an expected secret match.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device for decoding.")
    return parser.parse_args()


def validate_args(args):
    if args.msg_bits <= 0:
        raise ValueError("--msg-bits must be greater than 0")
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.expected_secret:
        if len(args.expected_secret) != args.msg_bits or any(bit not in {"0", "1"} for bit in args.expected_secret):
            raise ValueError("--expected-secret must be a binary string with length --msg-bits")
    builtin_msg_bits = get_builtin_codec_msg_bits(args.codec)
    if builtin_msg_bits is not None and builtin_msg_bits != args.msg_bits:
        raise ValueError(f"{args.codec} requires --msg-bits {builtin_msg_bits}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")


def load_image_tensor(path: str, device: torch.device):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        tensor = TF.to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, width, height


def main():
    args = parse_args()
    validate_args(args)

    codec_path = resolve_codec_path(args.codec)
    if not os.path.isfile(codec_path):
        raise ValueError(f"Codec file does not exist: {codec_path}")
    if not os.path.isfile(args.image):
        raise ValueError(f"Image file does not exist: {args.image}")

    device = torch.device(args.device)
    codec = AuthenLoRACodec(args.msg_bits)
    codec.load_codec_state(codec_path)
    codec.requires_grad_(False)
    codec.eval().to(device=device)

    image_tensor, width, height = load_image_tensor(args.image, device)

    with torch.no_grad():
        logits = codec.decoder(image_tensor)
        probs = logits.softmax(dim=-1)
        decoded = logits.argmax(dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()

    bits = "".join(str(int(bit.item())) for bit in decoded[0].detach().cpu())
    result = {
        "decoded_bits": bits,
        "msg_bits": args.msg_bits,
        "confidence": confidence,
        "threshold": args.threshold,
        "codec": args.codec,
        "codec_path": codec_path,
        "codec_sha256": file_sha256(codec_path),
        "image": {
            "width": width,
            "height": height,
        },
        "expected_secret_sha256": secret_hash(args.expected_secret),
        "bit_accuracy": None,
        "hamming_errors": None,
        "match": None,
    }

    match = None
    if args.expected_secret:
        expected = torch.tensor([[int(bit) for bit in args.expected_secret]], device=logits.device, dtype=torch.long)
        acc = bit_accuracy(logits, expected)
        decoded_expected = torch.tensor([[int(bit) for bit in bits]], device=expected.device, dtype=torch.long)
        errors = int((decoded_expected != expected).sum().item())
        match = acc >= args.threshold
        result["bit_accuracy"] = acc
        result["hamming_errors"] = errors
        result["match"] = match

    result.update(summarize_detection(bits, match=match, has_expected_secret=bool(args.expected_secret)))
    if result["watermark_status"] == "not_detected" and args.expected_secret:
        result["match"] = False

    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise
