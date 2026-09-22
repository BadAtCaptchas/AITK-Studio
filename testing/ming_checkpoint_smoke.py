"""Opt-in real-checkpoint validation, never part of the weight-free test suite.

Run as a module with --checkpoint (an already downloaded snapshot), --variant,
--resolution and a new --output directory. Records actual GPU peak memory for
loading, frozen encoding, optimizer steps, LoRA reload/resume and PNG previews.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch

from extensions_built_in.diffusion_models.ming_image import MingImageDesignModel, MingImageDesignLayerModel
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.unloader import unload_text_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--variant', choices=['design', 'layer'], required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--resident-transformer', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1234)
    torch.autograd.set_detect_anomaly(args.anomaly)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    report = {'variant': args.variant, 'resolution': args.resolution,
              'target_layers': 2 if args.variant == 'layer' else None,
              'device': torch.cuda.get_device_name(), 'compute_dtype': 'bfloat16',
              'adapter_dtype': 'float32', 'lora_rank': 16, 'lora_alpha': 16,
              'qtype': 'ming_fp8', 'gradient_checkpointing': True,
              'offload_mode': 'resident' if args.resident_transformer else 'all_blocks',
              'checkpoint': str(Path(args.checkpoint).resolve()),
              'checkpoint_revision': Path(args.checkpoint).name,
              'phases': []}
    started = time.perf_counter()

    def record(name, **values):
        torch.cuda.synchronize()
        row = dict(phase=name, elapsed_seconds=time.perf_counter()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), **values)
        report['phases'].append(row)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        torch.cuda.reset_peak_memory_stats()

    cls = MingImageDesignLayerModel if args.variant == 'layer' else MingImageDesignModel
    model = cls('cuda:0', ModelConfig(arch=cls.arch, name_or_path=args.checkpoint,
        quantize=True, qtype='ming_fp8', quantize_te=False, low_vram=True,
        layer_offloading=True, layer_offloading_backend='block',
        layer_offloading_transformer_percent=1.0), dtype='bf16')
    try:
        model.load_model()
        if args.resident_transformer:
            model.model._block_offload_manager.detach()
        assert getattr(model.model, '_aitk_layer_offloading_backend', None) == 'block'
        record('load')
        size = args.resolution
        background = Image.new('RGBA', (size, size), (30, 95, 200, 255))
        graphic = Image.new('RGBA', (size, size))
        ImageDraw.Draw(graphic).ellipse((size//4, size//4, 3*size//4, 3*size//4), fill=(255, 210, 30, 210))
        composite = Image.alpha_composite(background, graphic)
        reference_path = args.output / 'reference.png'
        composite.save(reference_path)
        frames = [composite, background, graphic] if model.layered else [composite]
        pixels = torch.stack([torch.from_numpy(np.array(item).copy()).permute(2, 0, 1).float()/127.5-1 for item in frames])
        target = model.encode_images(pixels[None] if model.layered else pixels).detach()
        config = GenerateImageConfig(prompt='A yellow sun on a plain blue background.',
            width=size, height=size, num_layers=2 if model.layered else None,
            ctrl_img=str(reference_path) if model.layered else None,
            num_inference_steps=12, guidance_scale=2 if model.layered else 1,
            seed=42, output_path=str(args.output / 'preview.png'))
        model.prepare_sample_image_config_for_encoding(config)
        controls = model.load_sample_control_images(config)
        embeds = model.encode_prompt(config.prompt, control_images=controls)
        negative = model.encode_prompt('', control_images=controls)
        model.sample_prompts_cache = [{'conditional': embeds.clone(), 'unconditional': negative.clone()}]
        record('encode')
        unload_text_encoder(model)
        gc.collect()
        torch.cuda.empty_cache()
        record('unload_conditioner')
        model.model.requires_grad_(False).enable_gradient_checkpointing()
        model.model.to('cuda:0')
        lora = LoRASpecialNetwork(text_encoder=model.text_encoder, unet=model.model,
            train_text_encoder=False, train_unet=True, lora_dim=16, alpha=16,
            target_lin_modules=model.target_lora_modules, is_transformer=True,
            network_config=NetworkConfig(type='lora', linear=16, linear_alpha=16), base_model=model)
        lora.force_to('cuda:0', dtype=torch.float32)
        model.network = lora
        lora._update_torch_multiplier()
        lora.apply_to(model.text_encoder, model.model, False, True)
        lora.can_merge_in = False
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(lora.parameters(), lr=1e-4)
        model.noise_scheduler.set_train_timesteps(1000, 'cuda:0')
        if args.anomaly:
            warnings.simplefilter('always')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            noise = torch.randn_like(target)
            timestep = model.noise_scheduler.timesteps[500:501]
            noisy = model.noise_scheduler.add_noise(target, noise, timestep)
            with lora:
                prediction = model.predict_noise(noisy, conditional_embeddings=embeds,
                    timestep=timestep, guidance_scale=1.0)
                loss = (prediction.float() - (noise-target).float()).square().mean()
                assert torch.isfinite(loss), 'Nonfinite loss'
                print(f'Forward loss: {float(loss.detach())}', flush=True)
                loss.backward()
            gradients = [p.grad for p in lora.parameters() if p.grad is not None]
            invalid = [name for name, p in lora.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            assert gradients and not invalid, f'Invalid gradients: {len(gradients)} present, {invalid[:16]}'
            norm = torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
            assert float(norm) > 0, 'Zero LoRA gradients'
            optimizer.step()
            record('optimizer_step' if step == 0 else 'resumed_optimizer_step',
                   loss=float(loss.detach()), gradient_norm=float(norm))
            if step == 0:
                save_path = str(args.output / 'adapter.safetensors')
                lora.save_weights(save_path, dtype=torch.float32)
                state = {name: value.detach().cpu().clone() for name, value in lora.state_dict().items()}
                optimizer_state = optimizer.state_dict()
                with torch.no_grad():
                    for parameter in lora.parameters():
                        parameter.zero_()
                lora.load_weights(save_path)
                for name, value in lora.state_dict().items():
                    torch.testing.assert_close(value.cpu(), state[name], atol=0, rtol=0)
                optimizer = AdamW8bit(lora.parameters(), lr=1e-4)
                optimizer.load_state_dict(optimizer_state)
                del state, optimizer_state
                record('lora_reload')
        optimizer.zero_grad(set_to_none=True)
        del optimizer, prediction, loss, gradients, target, noisy, noise
        gc.collect()
        model.generate_images([config])
        assert (args.output / 'preview.png').is_file()
        if model.layered:
            output = json.loads((args.output / 'preview.png.layers.json').read_text())
            assert len(output['layers']) == 2
            assert all((args.output / path).is_file() for path in output['layers'])
        record('preview')
        report['passed'] = True
    except BaseException as error:
        report['passed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
