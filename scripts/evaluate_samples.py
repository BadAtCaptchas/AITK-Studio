import argparse
import json
import math
import os
from pathlib import Path


def load_image(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def image_stats(path):
    import numpy as np

    img = load_image(path)
    arr = np.asarray(img).astype("float32") / 255.0
    return {
        "width": img.width,
        "height": img.height,
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
    }


def try_clip_scores(samples):
    try:
        import torch
        import open_clip
    except Exception as exc:
        return {}, f"CLIP unavailable: {exc}"

    prompted = [sample for sample in samples if sample.get("prompt")]
    if not prompted:
        return {}, "CLIP unavailable: no prompts were provided."

    try:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        scores = {}
        with torch.no_grad():
            for sample in prompted:
                image = preprocess(load_image(sample["sample_path"])).unsqueeze(0).to(device)
                text = tokenizer([sample["prompt"]]).to(device)
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores[sample["id"]] = float((image_features @ text_features.T).item())
        return scores, None
    except Exception as exc:
        return {}, f"CLIP unavailable: {exc}"


def try_lpips_scores(samples):
    try:
        import torch
        import torchvision.transforms.functional as TF
        import lpips
    except Exception as exc:
        return {}, f"LPIPS unavailable: {exc}"

    pairs = [sample for sample in samples if sample.get("reference_path") and os.path.isfile(sample["reference_path"])]
    if not pairs:
        return {}, "LPIPS unavailable: no image references were provided."

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net="alex").to(device).eval()
        scores = {}
        with torch.no_grad():
            for sample in pairs:
                a = TF.to_tensor(load_image(sample["sample_path"])).unsqueeze(0).to(device) * 2 - 1
                b = TF.to_tensor(load_image(sample["reference_path"])).unsqueeze(0).to(device) * 2 - 1
                if a.shape != b.shape:
                    b = torch.nn.functional.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
                scores[sample["id"]] = float(model(a, b).item())
        return scores, None
    except Exception as exc:
        return {}, f"LPIPS unavailable: {exc}"


def try_fid(samples):
    reference_dirs = sorted({sample.get("reference_path") for sample in samples if sample.get("reference_path")})
    reference_dirs = [ref for ref in reference_dirs if ref and os.path.isdir(ref)]
    sample_dirs = sorted({str(Path(sample["sample_path"]).parent) for sample in samples if sample.get("sample_path")})
    if not reference_dirs or not sample_dirs:
        return None, "FID unavailable: reference and sample directories are required."
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return float(calculate_fid_given_paths([sample_dirs[0], reference_dirs[0]], 32, device, 2048)), None
    except Exception as exc:
        return None, f"FID unavailable: {exc}"


def average(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    items = []
    errors = []

    clip_scores, clip_error = try_clip_scores(samples)
    if clip_error:
        errors.append(clip_error)

    lpips_scores, lpips_error = try_lpips_scores(samples)
    if lpips_error:
        errors.append(lpips_error)

    fid, fid_error = try_fid(samples)
    if fid_error:
        errors.append(fid_error)

    for sample in samples:
        sample_errors = []
        metrics = {}
        try:
            metrics.update(image_stats(sample["sample_path"]))
        except Exception as exc:
            sample_errors.append(f"Image stats unavailable: {exc}")
        if sample["id"] in clip_scores:
            metrics["clip_prompt_alignment"] = clip_scores[sample["id"]]
        if sample["id"] in lpips_scores:
            metrics["lpips"] = lpips_scores[sample["id"]]
        items.append({"id": sample["id"], "metrics": metrics, "errors": sample_errors})

    summary = {
        "sample_count": len(samples),
        "clip_prompt_alignment_avg": average([item["metrics"].get("clip_prompt_alignment") for item in items]),
        "lpips_avg": average([item["metrics"].get("lpips") for item in items]),
        "brightness_avg": average([item["metrics"].get("brightness") for item in items]),
        "contrast_avg": average([item["metrics"].get("contrast") for item in items]),
        "fid": fid,
        "unavailable": errors,
    }

    Path(args.output).write_text(json.dumps({"summary": summary, "items": items}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
