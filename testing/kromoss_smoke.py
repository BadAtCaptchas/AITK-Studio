"""Opt-in real-checkpoint Kromoss training/preview/resume validation.

Run after downloading the model components into the configured MODELS_PATH:
  .venv/Scripts/python.exe testing/kromoss_smoke.py
The harness uses only synthetic images in a new directory under .tmp. Each
variant runs one optimizer step, then resumes in a fresh process for step two.
No existing dataset, training run, or model file is modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import psutil
import torch
import yaml
from PIL import Image, ImageDraw
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full_lora", "full_lokr", "turbo_lora", "turbo_lokr")


def create_fixtures(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for index, color in enumerate(("red", "blue", "green", "orange")):
        image = Image.new("RGB", (1024, 1024), "white")
        ImageDraw.Draw(image).ellipse((200, 200, 824, 824), fill=color)
        image.save(directory / f"{index}.png")
        (directory / f"{index}.txt").write_text(
            f"A {color} circle on a white background.", encoding="utf-8"
        )


def config_for(variant: str, directory: Path, resume: bool) -> dict:
    turbo = variant.startswith("turbo")
    suffix = "_turbo" if turbo else ""
    example = ROOT / f"config/examples/train_lora_kromoss{suffix}_16gb.yaml"
    config = yaml.safe_load(example.read_text(encoding="utf-8"))
    config["config"]["name"] = variant
    process = config["config"]["process"][0]
    process["training_folder"] = str(directory / "training")
    process["network"]["type"] = variant.split("_")[1]
    process["save"].update(save_every=1, max_step_saves_to_keep=3)
    process["datasets"][0].update(
        folder_path=str(directory / "synthetic-images"), num_workers=0,
        caption_dropout_rate=0, shuffle_tokens=False,
    )
    process["train"].update(steps=2 if resume else 1, seed=42, skip_first_sample=resume)
    process["sample"].update(
        sample_every=1, prompts=["A red circle on a white background."],
        format="png", seed=42,
    )
    return config


def run_process(config_path: Path, *, training: bool = True) -> dict:
    log_path = config_path.with_suffix(".log")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               NO_ALBUMENTATIONS_UPDATE="1", PYTHONUNBUFFERED="1")
    env.setdefault("TRITON_CACHE_DIR", str(config_path.parent / "compiler-cache" / "triton"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(config_path.parent / "compiler-cache" / "inductor"))
    wrapper = (
        "import runpy, torch; torch.cuda.reset_peak_memory_stats();\n"
        "try: runpy.run_path('run.py', run_name='__main__')\n"
        "finally: print('KROMOSS_GPU_BYTES', torch.cuda.max_memory_allocated(), "
        "torch.cuda.max_memory_reserved(), flush=True)"
    )
    started = time.monotonic()
    peak_rss = 0
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-c", wrapper, str(config_path)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"PID {child.pid}: {log_path}", flush=True)
        try:
            while child.poll() is None:
                try:
                    parent = psutil.Process(child.pid)
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in
                                               [parent, *parent.children(recursive=True)]))
                except psutil.Error:
                    pass
                if time.monotonic() - started > 1800:
                    raise TimeoutError(f"Smoke phase exceeded 30 minutes: {config_path.name}")
                time.sleep(1)
        finally:
            if child.poll() is None:
                parent = psutil.Process(child.pid)
                processes = [*parent.children(recursive=True), parent]
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.Error:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=10)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.Error:
                        pass
                child.wait(timeout=10)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"KROMOSS_GPU_BYTES (\d+) (\d+)", text)
    metrics = {
        "exit_code": child.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "peak_rss_bytes": peak_rss,
        "gpu_allocated_bytes": int(match[1]) if match else None,
        "gpu_reserved_bytes": int(match[2]) if match else None,
    }
    config_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    if child.returncode:
        raise RuntimeError(f"Smoke phase failed; see {log_path}\n{text[-5000:]}")
    if match is None:
        raise AssertionError(f"GPU peak memory was not recorded: {log_path}")
    if training and config_path.stem.endswith("_resume"):
        if "Found step 1 in metadata, starting from there" not in text or "Loading optimizer state from" not in text:
            raise AssertionError(f"Adapter and optimizer resume were not confirmed: {log_path}")
        if any(message in text for message in (
            "Skipping incompatible optimizer", "Failed to load optimizer", "Skipping optimizer state"
        )):
            raise AssertionError(f"Optimizer resume was skipped: {log_path}")
    if training:
        losses = [float(value) for value in re.findall(r"loss: ([\d.eE+\-]+)", text)]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise AssertionError(f"No finite training loss reported: {log_path}")
        grad_norms = [float(value) for value in re.findall(r"grad_norm: ([\d.eE+\-]+|nan|inf)", text)]
        if not grad_norms or not all(math.isfinite(value) and value > 0 for value in grad_norms):
            raise AssertionError(f"No finite, nonzero gradient norm reported: {log_path}")
        metrics["reported_loss"] = losses[-1]
        metrics["reported_grad_norm"] = grad_norms[-1]
    return metrics


def validate_export(directory: Path, variant: str, expected_step: int, phase_start: float) -> str:
    run_directory = directory / "training" / variant
    checkpoint = run_directory / f"{variant}.safetensors"
    optimizer = torch.load(run_directory / "optimizer.pt", map_location="cpu", weights_only=True)
    if optimizer.get("step") != expected_step or not optimizer.get("optimizer_state_dict", {}).get("state"):
        raise AssertionError("Optimizer state is missing or does not match the model step")
    del optimizer
    digest = hashlib.sha256()
    with safe_open(checkpoint, framework="pt") as handle:
        keys = list(handle.keys())
        if not keys or not all(key.startswith("diffusion_model.") for key in keys):
            raise AssertionError(f"Unexpected export keys: {checkpoint}")
        metadata = handle.metadata() or {}
        training_info = json.loads(metadata.get("training_info", "{}"))
        if training_info.get("step") != expected_step:
            raise AssertionError(f"Wrong saved training step: {training_info}")
        if variant.startswith("turbo"):
            if metadata.get("aitk_distillation_lora_path") != "lodestones/Kromoss/8-steps-dmd-lora.safetensors":
                raise AssertionError("Turbo dependency missing from exported metadata")
            if metadata.get("aitk_distillation_lora_strength") != "1.0" or metadata.get("aitk_distillation_lora_required") != "true":
                raise AssertionError("Turbo metadata must require the fixed adapter at strength 1")
        elif "aitk_distillation_lora_path" in metadata:
            raise AssertionError("Full-model adapter unexpectedly requires turbo distillation")
        if variant.endswith("lokr"):
            if not all("lokr_" in key or key.endswith(".alpha") for key in keys):
                raise AssertionError("LoKr export includes weights outside the trained adapter")
            if not any(".qkv.alpha" in key for key in keys):
                raise AssertionError("LoKr export lost its fused QKV alpha")
        elif not any(key.endswith(("lora_A.weight", "lora_down.weight")) for key in keys):
            raise AssertionError("LoRA export has no down projections")
        for key in sorted(keys):
            tensor = handle.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"Nonfinite exported tensor: {key}")
            if key.endswith(("lora_A.weight", "lora_down.weight")) and tensor.shape[0] != 16:
                raise AssertionError("Export includes weights outside the rank-16 user adapter")
            digest.update(key.encode())
            digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    images = [path for path in run_directory.rglob("*.png") if path.stat().st_mtime >= phase_start]
    if not images:
        raise AssertionError("No preview images produced")
    for path in images:
        with Image.open(path) as image:
            image.load()
            if image.size != (1024, 1024):
                raise AssertionError(f"Unexpected preview size: {path}")
            extrema = image.convert("RGB").getextrema()
            if not any(high > low for low, high in extrema):
                raise AssertionError(f"Preview is a constant image: {path}")
    return digest.hexdigest()


def run_generation(source: Path, directory: Path, variant: str) -> dict:
    """Exercise ordinary GenerateProcess with a freshly loaded saved adapter."""
    training_config = yaml.safe_load((source / f"{variant}_resume.yaml").read_text())
    process = training_config["config"]["process"][0]
    model = process["model"]
    checkpoint = source / "training" / variant / f"{variant}.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model["lora_path"] = str(checkpoint.resolve())
    output = directory / variant
    sample = process["sample"]
    config = {
        "job": "generate",
        "config": {"name": f"{variant}_reload", "device": "cuda:0", "process": [{
            "type": "to_folder", "output_folder": str(output), "dtype": "bf16",
            "model": model,
            "generate": {
                "backend": "native", "sampler": sample["sampler"],
                "width": 1024, "height": 1024, "sample_steps": sample["sample_steps"],
                "guidance_scale": sample["guidance_scale"], "seed": 42,
                "images": [{"prompt": "A red circle on a white background.",
                            "network_multiplier": strength} for strength in (0, 1)],
            },
        }]},
    }
    config_path = directory / f"{variant}_generate.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metrics = run_process(config_path, training=False)
    images = sorted(output.glob("*.png"))
    if len(images) != 2:
        raise AssertionError(f"Expected two generated images: {output}")
    hashes = []
    for path in images:
        with Image.open(path) as image:
            image.load()
            if image.size != (1024, 1024):
                raise AssertionError(f"Unexpected generation size: {path}")
            rgb = image.convert("RGB")
            if not any(high > low for low, high in rgb.getextrema()):
                raise AssertionError(f"Constant generated image: {path}")
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    if hashes[0] == hashes[1]:
        raise AssertionError(f"Saved user adapter had no visible effect at strength 1: {variant}")
    metrics["image_rgb_sha256"] = hashes
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output-dir", type=Path, help="New empty validation directory")
    parser.add_argument("--generation-from", type=Path,
                        help="Skip training; generate at user strength 0/1 from a previous smoke directory")
    args = parser.parse_args()
    if args.output_dir:
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="kromoss-smoke-", dir=ROOT / ".tmp"))
    print(f"VALIDATION_DIRECTORY {directory}", flush=True)
    if args.generation_from:
        results = {}
        for variant in args.variants:
            results[variant] = run_generation(args.generation_from.resolve(), directory, variant)
            (directory / "results.json").write_text(json.dumps(results, indent=2))
            print(f"PASS {variant} generation: {results[variant]}", flush=True)
        return
    create_fixtures(directory / "synthetic-images")
    results = {}
    for variant in args.variants:
        before = None
        for resume in (False, True):
            phase = variant + ("_resume" if resume else "")
            config = directory / f"{phase}.yaml"
            config.write_text(yaml.safe_dump(config_for(variant, directory, resume), sort_keys=False))
            phase_start = time.time()
            results[phase] = run_process(config)
            digest = validate_export(directory, variant, 2 if resume else 1, phase_start)
            if resume and digest == before:
                raise AssertionError(f"Adapter weights did not change after resume: {variant}")
            before = digest
            results[phase]["tensor_sha256"] = digest
            (directory / "results.json").write_text(json.dumps(results, indent=2))
            print(f"PASS {phase}: {results[phase]}", flush=True)


if __name__ == "__main__":
    main()
